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
"""Tests for ``--dedupe-chained-param-sync``.

A MoE model with ``--use-distributed-optimizer`` gets a ``ChainedOptimizer`` of two
``DistributedOptimizer`` children (dense + expert) that share the same model chunks.  Each
child walks **all** bucket groups of those chunks when it syncs params, so without the flag
every bucket group is all-gathered once per child and the earlier gather is immediately
overwritten by the later one (dense and expert share bucket groups; the shards laid down by
the first child's gather are stale for the second child's parameters).

With ``--dedupe-chained-param-sync`` the chain stages every child's updated shards first and
then all-gathers each bucket group exactly once.  An all-gather only copies, so the final
parameters must be **bitwise identical** — that, plus the gather count, is what is asserted
here.  None of the deferral machinery works with ``--overlap-param-gather`` or Megatron-FSDP,
so the flag must be inert there (asserted too).
"""
import gc
import os
import sys

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import _set_timers, destroy_global_vars, set_args
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.a2a_overlap.utils import deterministic_mode
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
_NUM_EXPERTS = 8
_EP = 2


def _snapshot_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


@torch.no_grad()
def _restore(model, params0):
    for n, p in model.named_parameters():
        if n in params0:
            p.data.copy_(params0[n].to(p.device))


def _assert_bitwise_equal(a, b, what):
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert torch.equal(a, b), f"{what}: not bitwise equal (max |diff| {(a.float()-b.float()).abs().max()})"


class _GatherCounter:
    """Count ``_start_bucket_group_param_sync`` calls per model chunk."""

    def __init__(self, model):
        self.calls = []
        self.chunks = list(model)

    def __enter__(self):
        self._orig = [chunk._start_bucket_group_param_sync for chunk in self.chunks]
        for chunk, orig in zip(self.chunks, self._orig):
            def hook(bucket_group, force_sync=False, _orig=orig):
                self.calls.append(id(bucket_group))
                return _orig(bucket_group, force_sync=force_sync)

            chunk._start_bucket_group_param_sync = hook
        return self

    def __exit__(self, *exc):
        for chunk, orig in zip(self.chunks, self._orig):
            chunk._start_bucket_group_param_sync = orig
        return False

    def reset(self):
        self.calls = []


class TestDedupeChainedParamSync:
    seq_length = 128
    micro_batch_size = 1

    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        # EP>1 is what makes the experts use expert PGs (param.allreduce=False), which is
        # what splits the chain into a dense and an expert DistributedOptimizer. With EP=1
        # the two share one DistOpt and the dedupe has nothing to dedupe.
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=_EP,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(self, pre_process=True, post_process=True, **kw):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        return GPTModel(
            config=core_transformer_config_from_args(args),
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                num_experts=_NUM_EXPERTS
            ),
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
        )

    def _create_args(self, dedupe, overlap_param_gather=False):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        sys.argv = ['test_dedupe_chained_param_sync.py']
        args = parse_args()
        args.num_layers = 2
        args.vocab_size = 128
        args.hidden_size = 128
        args.ffn_hidden_size = 256
        args.num_attention_heads = 4
        args.num_experts = _NUM_EXPERTS
        args.moe_ffn_hidden_size = args.ffn_hidden_size
        args.moe_router_topk = 2
        args.max_position_embeddings = self.seq_length
        args.seq_length = self.seq_length
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.tensor_model_parallel_size = 1
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.expert_model_parallel_size = _EP
        args.train_iters = 10
        args.lr = 1e-3
        args.clip_grad = 0.0
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0
        args.attention_backend = 'unfused'
        args.optimizer = 'adam'
        args.use_distributed_optimizer = True
        args.overlap_param_gather = overlap_param_gather
        # validate_args requires the pair: async param gather rides on async grad reduce.
        args.overlap_grad_reduce = overlap_param_gather
        args.dedupe_chained_param_sync = dedupe
        args.recompute_granularity = None
        args.recompute_method = None
        args.recompute_num_layers = None
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        dp = max(1, world)  # TP=PP=1, so data_parallel_size == world_size
        args.global_batch_size = dp * self.micro_batch_size
        args.eval_global_batch_size = args.global_batch_size
        validate_args(args)
        set_args(args)
        _set_timers(args)
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1,
        )
        return args

    def _build(self, dedupe, overlap_param_gather=False):
        args = self._create_args(dedupe, overlap_param_gather)
        torch.manual_seed(_SEED)
        model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, model_provider_func=self.model_provider
        )
        return args, model, optimizer

    def _batch(self):
        ids = torch.arange(self.seq_length, dtype=torch.int64)[None, :].cuda()
        labels = 1 + ids
        pos = ids.clone()
        mask = torch.ones((1, 1, self.seq_length, self.seq_length), dtype=bool).cuda()
        loss_mask = torch.ones(self.seq_length)[None, :].cuda()
        return ids, labels, pos, mask, loss_mask

    def _step(self, args, model, optimizer):
        ids, labels, pos, mask, loss_mask = self._batch()
        model[0].zero_grad_buffer()
        optimizer.zero_grad()
        model[0].set_is_first_microbatch()
        out = model[0].forward(
            input_ids=ids, position_ids=pos, attention_mask=mask, labels=labels, loss_mask=loss_mask
        )
        out.mean().backward()
        ok, _, _ = optimizer.step()
        assert ok
        return out.detach().clone()

    def test_chain_has_two_distributed_optimizers(self):
        """The premise of the feature: dense and expert each get their own DistOpt."""
        _, _, optimizer = self._build(dedupe=True)
        assert isinstance(optimizer, ChainedOptimizer)
        dist_opts = [o for o in optimizer.chained_optimizers if isinstance(o, DistributedOptimizer)]
        assert len(dist_opts) >= 2, (
            f"expected >=2 DistributedOptimizers (dense + expert), got "
            f"{[type(o).__name__ for o in optimizer.chained_optimizers]}"
        )

    def test_flag_is_inert_with_overlap_param_gather(self):
        """The deferral machinery needs the synchronous gather path."""
        _, _, optimizer = self._build(dedupe=True, overlap_param_gather=True)
        assert not optimizer._should_dedupe_param_sync()

    def test_flag_off_by_default(self):
        _, _, optimizer = self._build(dedupe=False)
        assert not optimizer._should_dedupe_param_sync()

    def test_flag_on_enables_dedupe(self):
        _, _, optimizer = self._build(dedupe=True)
        assert optimizer._should_dedupe_param_sync()

    def test_each_bucket_group_gathered_once_per_step(self):
        """The whole point: gather count per step drops from once-per-child to once-per-group."""
        world = torch.distributed.get_world_size()
        if world < 2 * _EP:
            pytest.skip(
                f"needs dp>=2 and EP={_EP} (so both a separate expert optimizer and a real "
                f"param all-gather exist); run with --nproc-per-node>={2 * _EP}"
            )

        counts = {}
        for dedupe in (False, True):
            args, model, optimizer = self._build(dedupe=dedupe)
            with _GatherCounter(model) as counter:
                self._step(args, model, optimizer)
                if dedupe:
                    # with deferral, the gathers happen after _step(), i.e. inside optimizer.step()
                    pass
                counts[dedupe] = list(counter.calls)
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

        off, on = counts[False], counts[True]
        assert off, "no param sync was observed with the flag off; the test setup is wrong"
        assert len(on) == len(set(on)), f"bucket group synced more than once with the flag on: {on}"
        assert len(on) < len(off), (
            f"expected fewer gathers with --dedupe-chained-param-sync: "
            f"off={len(off)} on={len(on)}"
        )

    def test_on_vs_off_bitwise_identical(self):
        """An all-gather only copies, so deduping it must not change the final parameters."""
        world = torch.distributed.get_world_size()
        if world < 2 * _EP:
            pytest.skip(
                f"needs dp>=2 and EP={_EP} (so both a separate expert optimizer and a real "
                f"param all-gather exist); run with --nproc-per-node>={2 * _EP}"
            )

        n_steps = 3
        with deterministic_mode():
            off_args, off_model, off_opt = self._build(dedupe=False)
            params0 = _snapshot_params(off_model[0])
            on_args, on_model, on_opt = self._build(dedupe=True)
            _restore(on_model[0], params0)

            off_outs, off_params = [], []
            for _ in range(n_steps):
                off_outs.append(self._step(off_args, off_model, off_opt))
                off_params.append(_snapshot_params(off_model[0]))
            on_outs, on_params = [], []
            for _ in range(n_steps):
                on_outs.append(self._step(on_args, on_model, on_opt))
                on_params.append(_snapshot_params(on_model[0]))
            del off_model, on_model, off_opt, on_opt
            gc.collect()
            torch.cuda.empty_cache()

        for s in range(n_steps):
            _assert_bitwise_equal(on_outs[s], off_outs[s], f"output step {s}")
            assert on_params[s].keys() == off_params[s].keys(), f"param set mismatch step {s}"
            for name in on_params[s]:
                _assert_bitwise_equal(on_params[s][name], off_params[s][name],
                                      f"param step {s} {name}")
