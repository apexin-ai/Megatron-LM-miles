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
"""Prove that every rank records *when* each layer started, in process.

This drives exactly the entry points the training loop uses -- ``maybe_install``,
``begin_train_step``, ``phase('optimizer')``, ``end_train_step`` -- so what it exercises is the
integration, not a private API. It then reads back each rank's own JSONL and asserts the layer
start offsets are present and ordered.

Run it with ``torchrun --nproc-per-node 4 validate_layer_starts.py``.
"""

import argparse
import json
import os
import sys

import torch


def setup(ep):
    import torch.distributed as dist

    if not dist.is_initialized():
        world = int(os.environ.get('WORLD_SIZE', '1'))
        if world == 1:
            import tempfile

            dist.init_process_group(
                backend='nccl',
                init_method=f'file://{tempfile.gettempdir()}/perf_starts_store',
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


def build(args):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        num_moe_experts=args.experts or None,
        moe_ffn_hidden_size=args.ffn if args.experts else None,
        moe_router_topk=min(2, args.experts) if args.experts else None,
        ffn_hidden_size=args.ffn,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=args.experts or None)
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=args.vocab,
        max_sequence_length=args.seq,
        position_embedding_type='rope',
    )
    return model.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-layers', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=1024)
    ap.add_argument('--ffn', type=int, default=2048)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--vocab', type=int, default=1024)
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--experts', type=int, default=8)
    ap.add_argument('--ep', type=int, default=2)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--out', default='/tmp/perf-starts')
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(0)
    setup(args.ep)

    from perf_trace import hooks
    from perf_trace.tracer import PerfTracer  # noqa: F401  (ensures the module is importable)

    model = build(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mask = torch.ones((1, 1, args.seq, args.seq), dtype=bool, device='cuda')
    pos = torch.arange(args.seq, device='cuda').unsqueeze(0)

    def step(step_id):
        ids = torch.randint(0, args.vocab, (1, args.seq), device='cuda')
        optimizer.zero_grad(set_to_none=True)
        t0 = time_now()
        hooks.begin_train_step(step_id)
        out = model(input_ids=ids, position_ids=pos, attention_mask=mask, labels=ids)
        out.float().mean().backward()
        with hooks.phase('optimizer'):
            optimizer.step()
        hooks.end_train_step()
        return time_now() - t0

    def time_now():
        import time

        return time.perf_counter()

    for _ in range(args.warmup):
        step(0)

    # the same install call the training loop makes, with the same args object shape
    ns = argparse.Namespace(
        perf_trace_dir=args.out,
        perf_trace_every_n_steps=1,
        perf_trace_max_layers=256,
        perf_trace_ranks='',
    )
    tracer = hooks.maybe_install(ns, [model])
    if tracer is None:
        print(f'[rank {rank}] tracing not installed', flush=True)
        return

    for i in range(args.steps):
        step(i + 1)
    tracer.close()

    path = os.path.join(args.out, f'perf-rank-{rank}.jsonl')
    if not os.path.exists(path):
        print(f'[rank {rank}] NO FILE at {path}', flush=True)
        sys.exit(2)
    with open(path) as fh:
        recs = [json.loads(x) for x in fh if x.strip()]
    ok = True
    for rec in recs:
        layers = [l for l in rec.get('layers', []) if 'fs' in l and 'fwd' in l]
        starts = [l['fs'] for l in layers]
        if not layers:
            ok = False
            print(f"[rank {rank}] step {rec['step']}: NO layer start offsets", flush=True)
            continue
        monotonic = all(b >= a for a, b in zip(starts, starts[1:]))
        print(
            f"[rank {rank}] step {rec['step']}: {len(layers)} layer(s) with fs, "
            f'first={starts[0] / 1e6:.3f}s last={starts[-1] / 1e6:.3f}s '
            f'ordered={monotonic} step_end={rec["phases"].get("__end__", 0) / 1e6:.3f}s',
            flush=True,
        )
        if not monotonic:
            ok = False
    print(
        f'[rank {rank}] RESULT {"OK" if ok else "FAIL"} ({world} rank(s), {args.out})', flush=True
    )
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
