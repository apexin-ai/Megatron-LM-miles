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
"""Hook installation for :mod:`megatron.core.perf_trace`.

Everything here is inert unless ``--perf-trace-dir`` is set. Installation is best-effort: a
hook that cannot be attached logs once and is skipped, because a tracing feature must never be
the reason a training run fails.
"""

import functools
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRACER: Optional[Any] = None  # module-level singleton, set by maybe_install()

# The layer whose forward is on the stack right now. MoE all-to-alls are issued from inside a
# layer's forward, so this is what lets an A2A be attributed to a layer instead of just "some
# dispatch happened". Forward is single-threaded on the main thread, so a module-level stack is
# enough; it is a plain list for the same reason.
_LAYER_STACK: List[str] = []


def current_layer() -> Optional[str]:
    """Return the layer whose forward is running, if any."""
    return _LAYER_STACK[-1] if _LAYER_STACK else None


def get_tracer() -> Optional[Any]:
    """Return the active tracer, if tracing is enabled."""
    return TRACER


def _warn_once(key: str, msg: str, *args) -> None:
    seen = getattr(_warn_once, '_seen', None)
    if seen is None:
        seen = _warn_once._seen = set()  # type: ignore[attr-defined]
    if key not in seen:
        seen.add(key)
        logger.warning(msg, *args)


# --------------------------------------------------------------------------- layers
def _iter_transformer_layers(model_chunks: List[Any]):
    """Yield ``(global_layer_id, module)`` for transformer layers in the model.

    The id is **global**, not per-stage: with pipeline parallelism each rank only builds its own
    share of the layers, so a stage-local index would make rank 0's layer 0 and rank 8's layer 0
    look like the same layer in the dashboard's rank x layer heatmap. ``vp_stage`` is the model
    chunk index, which is how virtual pipeline parallelism exposes its stages.
    """
    try:
        from megatron.core.transformer.transformer_layer import (
            TransformerLayer,
            get_transformer_layer_offset,
        )
    except Exception:  # pragma: no cover - import layout may differ
        TransformerLayer = None  # type: ignore[assignment]
        get_transformer_layer_offset = None  # type: ignore[assignment]
    for ci, chunk in enumerate(model_chunks):
        config = getattr(chunk, 'config', None)
        offset = 0
        if get_transformer_layer_offset is not None and config is not None:
            try:
                offset = int(get_transformer_layer_offset(config, vp_stage=ci))
            except TypeError:
                try:
                    offset = int(get_transformer_layer_offset(config))
                except Exception:
                    offset = 0
            except Exception:
                offset = 0
        local = 0
        for module in chunk.modules():
            if TransformerLayer is not None and isinstance(module, TransformerLayer):
                yield offset + local, module
                local += 1


def _first_tensor(out):
    """Return the first tensor in a module output (tensor, tuple or list), else None."""
    if isinstance(out, (tuple, list)):
        for item in out:
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    return out if hasattr(out, 'register_hook') else None


def install_layer_hooks(tracer: Any, model_chunks: List[Any], max_layers: int = 256) -> int:
    """Attach forward and backward markers to every transformer layer.

    Forward timing uses the plain pre/post hooks. Backward timing uses the full backward hooks,
    which fire around the module's own backward; with activation checkpointing the recomputed
    forward runs inside ``CheckpointFunction.backward``, so the layer's backward hooks still
    bracket the recompute plus the gradient work -- which is the cost we want to see.

    Returns the number of layers instrumented.
    """
    count = 0
    for layer_id, layer in _iter_transformer_layers(model_chunks):
        if count >= max_layers:
            _warn_once(
                'layer-cap',
                'perf-trace: stopped instrumenting layers at %d (--perf-trace-max-layers)',
                max_layers,
            )
            break
        name = f'l{layer_id}'
        tracer._layer_names.append(name)

        def _fwd_pre(_mod, _args, _name=name):
            try:
                _LAYER_STACK.append(_name)
                tracer.mark_layer_forward_start(_name)
            except Exception:  # tracing must never break the step
                _warn_once('hook-fwd-pre', 'perf-trace: forward pre-hook failed for %s', _name)

        def _fwd_post(_mod, _args, _out, _name=name):
            try:
                if _LAYER_STACK and _LAYER_STACK[-1] == _name:
                    _LAYER_STACK.pop()
                tracer.mark_layer_forward_end(_name)
            except Exception:
                _warn_once('hook-fwd-post', 'perf-trace: forward post-hook failed for %s', _name)
            # The module's backward hooks fire at points that do not bracket its backward (see
            # tools/validate_perf_trace.py): a tensor hook on the output fires when that tensor's
            # gradient is ready, i.e. exactly when this layer's backward starts.
            tensor = _first_tensor(_out)
            if tensor is not None and tensor.requires_grad:

                def _bwd_marker(_grad, _n=name):
                    # A tensor hook must return a tensor or None; returning the event index
                    # would make autograd fail with "expected Variable, but hook returned 'int'".
                    try:
                        tracer.mark_layer_backward_start(_n)
                    except Exception:
                        _warn_once('hook-bwd', 'perf-trace: backward hook failed for %s', _n)
                    return None

                tensor.register_hook(_bwd_marker)

        try:
            layer.register_forward_pre_hook(_fwd_pre)
            layer.register_forward_hook(_fwd_post)
            count += 1
        except Exception:
            _warn_once('fwd-hook', 'perf-trace: could not attach forward hooks to %s', name)
            continue

        # A module *post* backward hook fires when the layer's input grads are ready, i.e. when
        # its backward is done. Those end markers, sorted, give the per-layer backward durations;
        # the *pre* hooks do not bracket anything useful and are deliberately not used.
        def _bwd_end_hook(_mod, _grad_in, _grad_out, _name=name):
            try:
                tracer.mark_layer_backward_end(_name)
            except Exception:
                _warn_once('hook-bwd-end', 'perf-trace: backward end hook failed for %s', _name)

        try:
            layer.register_full_backward_hook(_bwd_end_hook)
        except Exception:
            _warn_once('hook-bwd-reg', 'perf-trace: could not register backward hook on %s', name)
    return count


# ------------------------------------------------------------------------------ A2A
def install_a2a_hooks(tracer: Any) -> int:
    """Bracket the MoE all-to-all with markers on the comm stream.

    The dispatcher's ``dispatch``/``combine`` call the four ``*_preprocess``/``*_postprocess``
    methods; wrapping them is enough to see when the A2A actually starts and ends relative to
    the layer compute that surrounds it.
    """
    try:
        from megatron.core.transformer.moe import token_dispatcher
    except Exception:
        _warn_once('a2a-import', 'perf-trace: MoE token dispatcher not importable; no A2A timing')
        return 0
    try:
        from megatron.core.pipeline_parallel.utils import get_comm_stream
    except Exception:  # pragma: no cover
        get_comm_stream = lambda: None  # type: ignore[assignment]

    # Discover the classes instead of listing them: which dispatcher runs depends on the MoE
    # configuration (all-to-all, all-gather, flex), and a new one would otherwise be missed --
    # that is exactly how the first version recorded zero A2A events at EP=2.
    method_names = (
        'dispatch_preprocess',
        'dispatch_postprocess',
        'combine_preprocess',
        'combine_postprocess',
    )
    candidates = [
        name
        for name in dir(token_dispatcher)
        if name.endswith('Dispatcher')
        and isinstance(getattr(token_dispatcher, name, None), type)
        and all(hasattr(getattr(token_dispatcher, name), m) for m in method_names)
    ]
    wrapped = 0
    for cls_name in candidates:
        cls = getattr(token_dispatcher, cls_name, None)
        if cls is None:
            continue
        for method, tag in (
            ('dispatch_preprocess', 'dispatch'),
            ('dispatch_postprocess', 'dispatch'),
            ('combine_preprocess', 'combine'),
            ('combine_postprocess', 'combine'),
        ):
            orig = getattr(cls, method, None)
            if orig is None or getattr(orig, '_perf_traced', False):
                continue
            suffix = ':start' if method.endswith('preprocess') else ':end'

            @functools.wraps(orig)
            def _wrapper(self, *args, __orig=orig, __tag=tag, __suffix=suffix, **_kw):
                try:
                    stream = get_comm_stream()
                except Exception:
                    stream = None
                tracer.mark_a2a(f'{__tag}{__suffix}', stream=stream, layer=current_layer())
                return __orig(self, *args, **_kw)

            _wrapper._perf_traced = True  # type: ignore[attr-defined]
            setattr(cls, method, _wrapper)
            wrapped += 1
    if wrapped == 0:
        _warn_once('a2a-none', 'perf-trace: no MoE dispatcher methods wrapped; no A2A timing')
    return wrapped


# ------------------------------------------------------------------------ lifecycle
class phase:
    """Context manager that brackets a region on the GPU timeline.

    Usage::

        with perf_trace.phase('optimizer'):
            optimizer.step()

    A no-op when tracing is off, so call sites stay branch-free.
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        tracer = get_tracer()
        if tracer is not None:
            tracer.mark_span_start(self.name)
        return self

    def __exit__(self, *exc):
        tracer = get_tracer()
        if tracer is not None:
            tracer.mark_span_end(self.name)
        return False


# Megatron timers worth recording: they are CPU-measured but already maintained every step, so
# they cost nothing extra and they are the only source for names like "params-all-gather".
TIMERS_OF_INTEREST = (
    'forward-backward',
    'forward-compute',
    'backward-compute',
    'all-grads-sync',
    'params-all-gather',
    'optimizer',
    'optimizer-inner-step',
    'optimizer-copy-to-main-grad',
    'optimizer-copy-main-to-model-params',
    'optimizer-unscale-and-check-inf',
    'optimizer-clip-main-grad',
)


def capture_timers(timers: Any) -> None:
    """Fold Megatron's per-step timer values into the current step (seconds)."""
    tracer = get_tracer()
    if tracer is None or timers is None:
        return
    for name in TIMERS_OF_INTEREST:
        try:
            elapsed = timers(name).elapsed(reset=False)
        except Exception:
            continue
        if elapsed:
            tracer.note_cpu_phase(name, float(elapsed))


def reset_timers(timers: Any) -> None:
    """Reset the timers we read so the next step starts clean (Megatron resets them itself)."""
    return None


def begin_train_step(iteration: Optional[int]) -> bool:
    """Called at the top of ``train_step``; returns whether this step is traced."""
    tracer = get_tracer()
    if tracer is None or iteration is None:
        return False
    return tracer.begin_step(iteration)


def end_train_step() -> None:
    """Called at the bottom of ``train_step``."""
    tracer = get_tracer()
    if tracer is not None:
        tracer.end_step()


def wrap_data_iterator(data_iterator: Any) -> Any:
    """Wrap the schedule's data iterator(s) so batch fetch time is recorded."""
    tracer = get_tracer()
    if tracer is None or data_iterator is None:
        return data_iterator
    if isinstance(data_iterator, (list, tuple)):
        return [tracer.wrap_data_iterator(it) for it in data_iterator]
    return tracer.wrap_data_iterator(data_iterator)


def maybe_install(args: Any, model_chunks: List[Any]) -> Optional[Any]:
    """Install the tracer if ``--perf-trace-dir`` is set. Returns the tracer (or None)."""
    global TRACER
    out_dir = getattr(args, 'perf_trace_dir', None)
    if not out_dir:
        return None
    if TRACER is not None:
        return TRACER
    import torch

    from .tracer import PerfTracer

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    tracer = PerfTracer(
        out_dir=out_dir,
        rank=rank,
        every_n_steps=int(getattr(args, 'perf_trace_every_n_steps', 1) or 1),
        max_layers=int(getattr(args, 'perf_trace_max_layers', 256) or 256),
    )
    tracer.start()
    n_layers = install_layer_hooks(tracer, model_chunks, tracer.max_layers)
    n_a2a = install_a2a_hooks(tracer)
    if rank == 0:
        logger.info(
            'perf-trace: writing to %s every %d step(s); %d layers, %d A2A hooks',
            out_dir,
            tracer.every_n_steps,
            n_layers,
            n_a2a,
        )
    TRACER = tracer
    return tracer
