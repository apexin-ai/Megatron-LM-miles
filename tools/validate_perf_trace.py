#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate the perf-trace hooks on a real GPU box, without touching the installed Megatron.

Two things a CPU test cannot tell us:

1. do the hooks actually attach and produce a sane timeline on real kernels? and
2. does enabling them cost anything measurable?

So this builds a small real Megatron model (MoE included, since the A2A hook is the interesting
one), runs a few steps with tracing on and off **interleaved**, and prints:

    * which hooks attached,
    * a comparison of step times (median of the traced vs untraced steps),
    * the JSONL files written, which ``perf_trace_dashboard.py`` can then render.

Run with ``torchrun --nproc-per-node N validate_perf_trace.py``; single process works too.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch


def setup_distributed(ep=1):
    """Initialise torch.distributed (single process is fine) and the model-parallel groups."""
    import torch.distributed as dist

    if not dist.is_initialized():
        # Single process: a file:// rendezvous keeps this script usable without setting
        # MASTER_ADDR/PORT, which matters for a self-contained validation.
        import tempfile

        world = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        if world == 1:
            dist.init_process_group(
                backend='nccl',
                init_method=f'file://{tempfile.gettempdir()}/perf_trace_val_store',
                rank=0,
                world_size=1,
            )
        else:
            dist.init_process_group(backend='nccl')
    from megatron.core import parallel_state

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(0)


def build_model(args, num_experts):
    """A small GPT, optionally MoE, built from the installed Megatron."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        num_moe_experts=num_experts or None,
        moe_ffn_hidden_size=args.ffn if num_experts else None,
        moe_router_topk=min(2, num_experts) if num_experts else None,
        ffn_hidden_size=args.ffn,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        bf16=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=num_experts or None)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=args.vocab,
        max_sequence_length=args.seq,
        position_embedding_type='rope',
    )
    return model.cuda()


def one_step(model, optimizer, args, step, tracer=None):
    """One fwd+bwd+step, optionally bracketed by the tracer; returns wall seconds."""
    ids = torch.randint(0, args.vocab, (1, args.seq), device='cuda')
    labels = ids.clone()
    pos = torch.arange(args.seq, device='cuda').unsqueeze(0)
    optimizer.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    if tracer is not None:
        tracer.begin_step(step)
        tracer.mark_data_fetch(time.perf_counter() - t0)
    mask = torch.ones((1, 1, args.seq, args.seq), dtype=bool, device='cuda')
    out = model(input_ids=ids, position_ids=pos, attention_mask=mask, labels=labels)
    loss = out.float().mean()
    loss.backward()
    if tracer is not None:
        tracer.mark_span_start('optimizer')
    optimizer.step()
    if tracer is not None:
        tracer.mark_span_end('optimizer')
        tracer.end_step()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-layers', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--ffn', type=int, default=1024)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--vocab', type=int, default=1024)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--experts', type=int, default=8)
    ap.add_argument(
        '--ep', type=int, default=1, help='expert parallel size (needs >=2 to fire A2A)'
    )
    ap.add_argument('--warmup', type=int, default=4)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--pairs', type=int, default=3, help='interleaved on/off pairs for the A/B')
    ap.add_argument('--out', default='/tmp/perf-trace-validation')
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(0)

    setup_distributed(ep=args.ep)

    from perf_trace.hooks import install_a2a_hooks, install_layer_hooks
    from perf_trace.tracer import PerfTracer

    model = build_model(args, args.experts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ---- dry run (compiles kernels, warms allocations) so the A/B is not measuring warm-up
    for _ in range(args.warmup):
        one_step(model, optimizer, args, 0)

    # ---- one tracer for both the hooks and the traced arm
    tracer = PerfTracer(out_dir=args.out, rank=rank, every_n_steps=1, max_layers=64)
    tracer.start()
    n_layers = install_layer_hooks(tracer, [model], max_layers=64)
    n_a2a = install_a2a_hooks(tracer)
    print(f'[hooks] layers instrumented: {n_layers}   A2A methods wrapped: {n_a2a}', flush=True)

    # ---- interleaved on/off so drift hits both arms equally. In the off arm begin_step() is
    # not called, so the hooks record nothing (the tracer guards on the active step).
    on_times, off_times, deltas = [], [], []
    for pair in range(args.pairs):
        off = one_step(model, optimizer, args, 10_000 + pair) * 1e3
        on = one_step(model, optimizer, args, pair, tracer=tracer) * 1e3
        off_times.append(off)
        on_times.append(on)
        deltas.append(100 * (on - off) / off)
    tracer.close()

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    tag = f'[rank {rank}]'
    print(f'{tag} step ms  off: {[f"{x:.1f}" for x in off_times]}', flush=True)
    print(f'{tag} step ms   on: {[f"{x:.1f}" for x in on_times]}', flush=True)
    if off_times and on_times:
        print(
            f'{tag} median off={med(off_times):.2f} ms  on={med(on_times):.2f} ms  '
            f'delta={100 * (med(on_times) - med(off_times)) / med(off_times):+.2f}%',
            flush=True,
        )

    if rank == 0:
        path = os.path.join(args.out, f'perf-rank-{rank}.jsonl')
        if os.path.exists(path):
            with open(path) as fh:
                lines = [json.loads(x) for x in fh if x.strip()]
            print(f'[jsonl] {len(lines)} step line(s) at {path}', flush=True)
            if lines:
                rec = lines[-1]
                print('[jsonl] keys:', sorted(rec), flush=True)
                print(
                    f"[jsonl] layers recorded: {len(rec.get('layers', []))}  "
                    f"spans: {[s['n'] for s in rec.get('spans', [])]}  "
                    f"a2a: {[a['n'] for a in rec.get('a2a', [])]}",
                    flush=True,
                )
                fwd = [l.get('fwd') for l in rec.get('layers', []) if 'fwd' in l]
                bwd = [l.get('bwd') for l in rec.get('layers', []) if 'bwd' in l]
                print(f'[jsonl] layer fwd us: {fwd}', flush=True)
                print(f'[jsonl] layer bwd us: {bwd}', flush=True)
                assert fwd, 'no forward layer timings were recorded'
                assert all(v and v > 0 for v in fwd), 'forward layer timings are zero'


if __name__ == '__main__':
    main()
