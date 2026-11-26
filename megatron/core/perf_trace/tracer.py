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
"""Light-weight per-step, per-rank, per-layer GPU timing.

Why CUDA events and not wall-clock timestamps: launches are asynchronous, so the host clock
says nothing about when a kernel runs. ``torch.cuda.Event`` stores a marker *in the stream*,
so its position on the GPU timeline is the execution order no matter when the host enqueued
it. Event timestamps live in the device clock domain, which means an ``elapsed_time`` between
two events on **different streams of the same device** is also valid -- that is what makes the
MoE all-to-all (comm stream) comparable with layer compute (compute stream).

Cost model (48 layers x 4 microbatches, measured later on a GPU box):

* one ``record()`` per boundary, roughly 1-2 us of host time, spread over the step;
* no ``synchronize()`` and no ``elapsed_time()`` on the hot path -- a step's events are only
  converted (and its file line written) once the *next* step starts, when they must have
  completed, and only if ``query()`` says so;
* the writer runs on a background thread so file I/O never blocks the training loop.

The tracer is off unless ``--perf-trace-dir`` is given.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A record is (kind, name, event_index); kind is "layer_fwd" | "layer_bwd" | "phase" | "a2a".
Record = Tuple[str, str, int]


class _EventPool:
    """Reusable ``torch.cuda.Event`` objects, created lazily.

    Creating a fresh event per boundary would allocate on every step; the pool keeps a fixed
    set per tracer and resets it when a step is consumed.
    """

    def __init__(self, factory: Callable[[], Any], size: int = 4096):
        self._factory = factory
        self._free = [factory() for _ in range(min(size, 64))]
        self._live: List[Any] = []

    def acquire(self) -> int:
        """Take an event from the pool and return its index in ``self._live``."""
        if self._free:
            ev = self._free.pop()
        else:
            ev = self._factory()
        self._live.append(ev)
        return len(self._live) - 1

    def get(self, idx: int) -> Any:
        return self._live[idx]

    def release_all(self) -> None:
        """Return the step's events to the free list (they are done being read)."""
        self._free.extend(self._live)
        self._live = []


class PerfTracer:
    """Collect per-step GPU timings and write one JSON line per rank per step.

    Line schema (every duration is an integer number of **microseconds**, relative to the
    step's anchor unless noted)::

        {"step": 19, "rank": 3,
         "phases": {"__anchor__": 0, "__end__": 242900000},
         "layers": [{"n": "c0l0", "fwd": 3100000, "bwd": 2900000}, ...],
         "a2a": [{"n": "dispatch", "s": 4210000, "d": 4380000}, ...],
         "data": {"n": 4, "total": 25000, "max": 9000, "items": [6000, 7000, 9000, 3000]},
         "step_wall": 260100000,          # host wall clock for the step, not GPU time
         "anchor_wall": 1795147281.0}     # epoch seconds of the anchor


    Args:
        out_dir: Directory for ``perf-rank-<rank>.jsonl``.
        rank: Global rank of this process.
        every_n_steps: Sample every N steps; 0 disables.
        max_layers: Safety cap on tracked layers (keeps the event budget bounded).
        event_factory: Injection point for tests; defaults to a timing-enabled CUDA event.
        clock: Injection point for tests; defaults to ``time.time``.
    """

    def __init__(
        self,
        out_dir: str,
        rank: int,
        every_n_steps: int = 1,
        max_layers: int = 256,
        event_factory: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.time,
    ):
        if event_factory is None:
            import torch

            event_factory = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
        self.out_dir = out_dir
        self.rank = rank
        self.every_n_steps = every_n_steps
        self.max_layers = max_layers
        self._clock = clock
        self._pool = _EventPool(event_factory)
        self._records: List[Record] = []
        self._data_fetches: List[float] = []
        self._cpu_phases: Dict[str, float] = {}
        self._fwd_pass: Dict[str, int] = {}
        self._layer_names: List[str] = []
        self._step: Optional[int] = None
        self._anchor: Optional[int] = None
        self._anchor_wall: Optional[float] = None
        self._pending: List[Dict[str, Any]] = []
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._writer: Optional[threading.Thread] = None
        self._closed = False
        self.dropped_steps = 0
        self.written_steps = 0

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Create the output directory and start the background writer."""
        os.makedirs(self.out_dir, exist_ok=True)
        self._writer = threading.Thread(
            target=self._write_loop, name="perf-trace-writer", daemon=True
        )
        self._writer.start()
        atexit.register(self.close)

    def close(self) -> None:
        """Flush what is pending and stop the writer (idempotent).

        A single synchronize here is fine and necessary: the last step's tail events (the
        optimizer span, the end anchor) may still be in flight, and converting an unfinished
        event drops it silently. At shutdown there is nothing left to overlap with anyway.
        """
        if self._closed:
            return
        self._closed = True
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        self._drain(final=True)
        self._queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=5.0)

    # ------------------------------------------------------------------ writing
    def _write_loop(self) -> None:
        path = os.path.join(self.out_dir, f'perf-rank-{self.rank}.jsonl')
        with open(path, 'a', buffering=1) as fh:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                try:
                    fh.write(json.dumps(item, separators=(',', ':')) + '\n')
                except Exception:  # pragma: no cover - a broken file must not kill training
                    logger.exception('perf-trace: failed to write a line; dropping it')

    def _enqueue(self, item: Dict[str, Any]) -> None:
        self._queue.put(item)

    # ------------------------------------------------------------------ capture
    def enabled_for(self, step: int) -> bool:
        """Return whether ``step`` should be traced."""
        return self.every_n_steps > 0 and step % self.every_n_steps == 0

    def begin_step(self, step: int) -> bool:
        """Start a step: finalize the previous one, then record the anchor event.

        Returns True when this step is traced.
        """
        self._drain()
        if not self.enabled_for(step):
            self._step = None
            return False
        if len(self._layer_names) > self.max_layers:
            if not getattr(self, '_warned_layers', False):
                self._warned_layers = True
                logger.warning(
                    'perf-trace: %d candidate layers exceed --perf-trace-max-layers=%d; '
                    'later layers will not be timed',
                    len(self._layer_names),
                    self.max_layers,
                )
        self._records = []
        self._data_fetches = []
        self._cpu_phases = {}
        self._fwd_pass = {}
        self._cpu_phases = {}
        self._fwd_pass = {}
        self._step = step
        self._anchor = self._record('phase', '__anchor__')
        self._anchor_wall = self._clock()
        self._last_step_wall = None
        return True

    def end_step(self) -> None:
        """Mark the end of the traced step; conversion happens on the next ``begin_step``."""
        if self._step is None:
            return
        self._record('phase', '__end__')
        if self._anchor_wall is not None:
            self._last_step_wall = self._clock() - self._anchor_wall

    def _record(self, kind: str, name: str, stream: Any = None) -> Optional[int]:
        """Record one marker and return its event index.

        Outside a traced step nothing is recorded: the hooks stay installed for the whole run,
        so without this guard an untraced step would still pay for every layer's events and the
        on/off comparison would be meaningless.
        """
        if self._step is None:
            return None
        idx = self._pool.acquire()
        ev = self._pool.get(idx)
        if stream is not None:
            with _on_stream(stream):
                ev.record()
        else:
            ev.record()
        self._records.append((kind, name, idx))
        return idx

    # public hooks used by the hook module
    def mark_layer_forward_start(self, name: str) -> int:
        # With activation checkpointing the same layer runs forward twice in a step (once for
        # the forward, once during backward). Tag the repeat so it is not mistaken for a
        # re-measurement of the first pass -- otherwise the second overwrites the first and the
        # recompute cost silently disappears from the timeline.
        self._fwd_pass[name] = self._fwd_pass.get(name, 0) + 1
        tag = name if self._fwd_pass[name] == 1 else f'{name}#recompute'
        return self._record('layer_fwd', tag)

    def mark_layer_forward_end(self, name: str) -> int:
        tag = name if self._fwd_pass.get(name, 1) == 1 else f'{name}#recompute'
        return self._record('layer_fwd_end', tag)

    def mark_layer_backward_start(self, name: str) -> int:
        return self._record('layer_bwd', name)

    def mark_layer_backward_end(self, name: str) -> int:
        return self._record('layer_bwd_end', name)

    def mark_a2a(self, name: str, stream: Any = None) -> int:
        return self._record('a2a', name, stream=stream)

    def mark_span_start(self, name: str) -> int:
        """Open a named span on the GPU timeline (optimizer, grad sync, param sync, ...)."""
        return self._record('span', name + ':start')

    def mark_span_end(self, name: str) -> int:
        return self._record('span', name + ':end')

    def note_cpu_phase(self, name: str, seconds: float) -> None:
        """Add a host-measured phase duration (Megatron's timers) to this step.

        These are CPU numbers, not GPU timeline positions: they land in ``cpu`` rather than in
        ``spans`` so a plot can show them as a breakdown without pretending they are aligned.
        """
        if self._step is None:
            return
        self._cpu_phases[name] = self._cpu_phases.get(name, 0.0) + seconds

    # --------------------------------------------------------------- data timing
    def mark_data_fetch(self, seconds: float) -> None:
        """Record one host-side data fetch (batch generation / preprocessing).

        Data preparation is CPU work, so it is timed with the host clock rather than a CUDA
        event: the point is how long the training loop *waited* for the batch, not where it
        sits on the GPU timeline.
        """
        if self._step is None:
            return
        self._data_fetches.append(seconds)

    def wrap_data_iterator(self, iterator: Any) -> Any:
        """Return a proxy that times every ``next()`` on ``iterator``."""
        if self._step is None and self.every_n_steps <= 0:
            return iterator
        return TimedIterator(iterator, self)

    # ------------------------------------------------------------------- convert
    def _drain(self, final: bool = False) -> None:
        """Convert completed steps to JSON dicts and queue them for writing."""
        if self._step is None or not self._records:
            return
        if not final and not self._all_done():
            # The step's events are not all complete yet; try again next step. Bounded so a
            # permanently stuck query cannot grow memory without bound.
            self._pending.append(self._snapshot())
            if len(self._pending) > 4:
                self.dropped_steps += 1
                self._pending.pop(0)
            self._flush_pending(only_done=True)
            self._reset_step()
            return
        self._pending.append(self._snapshot())
        self._flush_pending(only_done=not final)
        self._reset_step()

    def _all_done(self) -> bool:
        try:
            return all(self._pool.get(idx).query() for _, _, idx in self._records)
        except Exception:  # pragma: no cover - a broken query is treated as "not done"
            return False

    def _snapshot(self) -> Dict[str, Any]:
        """Capture the current step's raw event list without converting it yet."""
        return {
            'step': self._step,
            'anchor': self._anchor,
            'anchor_wall': self._anchor_wall,
            'records': list(self._records),
            'data_fetches': list(self._data_fetches),
            'cpu_phases': dict(self._cpu_phases),
        }

    def _reset_step(self) -> None:
        self._step = None
        self._anchor = None
        self._anchor_wall = None
        self._records = []
        self._data_fetches = []
        self._cpu_phases = {}
        self._fwd_pass = {}

    def _flush_pending(self, only_done: bool) -> None:
        keep = []
        for item in self._pending:
            if only_done and not self._item_done(item):
                keep.append(item)
                continue
            self._enqueue(self._convert(item))
        self._pending = keep
        # Events of consumed steps can go back to the pool only once no snapshot uses them.
        if not self._pending and self._step is None:
            self._pool.release_all()

    def _item_done(self, item: Dict[str, Any]) -> bool:
        try:
            return all(self._pool.get(idx).query() for _, _, idx in item['records'])
        except Exception:  # pragma: no cover
            return False

    def _convert(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Turn raw events into a compact, relative-time JSON record."""
        anchor_idx = item['anchor']
        anchor = self._pool.get(anchor_idx) if anchor_idx is not None else None
        out: Dict[str, Any] = {'step': item['step'], 'rank': self.rank}
        if anchor is None:
            return out

        def rel(idx: int) -> Optional[float]:
            try:
                return anchor.elapsed_time(self._pool.get(idx))
            except Exception:
                return None

        per_layer: Dict[str, Dict[str, float]] = {}
        a2a: List[Dict[str, float]] = []
        phases: Dict[str, float] = {}
        open_fwd: Dict[str, float] = {}
        open_bwd: Dict[str, float] = {}
        open_a2a: Dict[str, float] = {}
        spans: List[Dict[str, float]] = []
        bwd_starts_in_order: List[Optional[tuple]] = []
        open_span: Dict[str, float] = {}

        for kind, name, idx in item['records']:
            t = rel(idx)
            if t is None:
                continue
            if kind == 'phase':
                phases[name] = t * 1000.0  # event time is ms; the file is all microseconds
            elif kind == 'layer_fwd':
                open_fwd[name] = t
            elif kind == 'layer_fwd_end':
                if name in open_fwd:
                    per_layer.setdefault(name, {})['fwd'] = t - open_fwd.pop(name)
            elif kind == 'layer_bwd':
                bwd_starts_in_order.append((name, t))
            elif kind == 'layer_bwd_end':
                if name in open_bwd:
                    per_layer.setdefault(name, {})['bwd'] = t - open_bwd.pop(name)
            elif kind == 'span':
                if name.endswith(':start'):
                    open_span[name[:-6]] = t
                elif name.endswith(':end') and name[:-4] in open_span:
                    st = open_span.pop(name[:-4])
                    spans.append({'n': name[:-4], 's': st * 1000.0, 'd': (t - st) * 1000.0})
            elif kind == 'a2a':
                if name.endswith(':start'):
                    open_a2a[name[:-6]] = t
                elif name.endswith(':end') and name[:-4] in open_a2a:
                    s = open_a2a.pop(name[:-4])
                    a2a.append({'name': name[:-4], 'start': s, 'dur': t - s})

        fetches = item.get('data_fetches') or []
        if fetches:
            out['data'] = {
                'n': len(fetches),
                'total': int(round(sum(fetches) * 1e6)),
                'max': int(round(max(fetches) * 1e6)),
                'items': [int(round(x * 1e6)) for x in fetches],
            }
        if getattr(self, '_last_step_wall', None) is not None:
            out['step_wall'] = int(round(self._last_step_wall * 1e6))
        # Backward durations come from consecutive layer-start markers: backward runs the
        # layers in reverse execution order, so layer i's backward lasts until layer i-1 starts.
        bwd_starts = [v for v in bwd_starts_in_order if v is not None]
        if len(bwd_starts) > 1:
            for idx in range(len(bwd_starts) - 1):
                name_i, t_i = bwd_starts[idx]
                _, t_next = bwd_starts[idx + 1]
                per_layer.setdefault(name_i, {})['bwd'] = t_next - t_i
            last_name, last_t = bwd_starts[-1]
            end_ref = phases.get('__end__', last_t) / 1000.0
            if end_ref > last_t:
                per_layer.setdefault(last_name, {})['bwd'] = end_ref - last_t

        out['anchor_wall'] = item['anchor_wall']
        out['phases'] = phases
        # Keep the file small: microsecond integers, layer order preserved.
        out['layers'] = [
            {'n': n, **{k: int(round(v * 1000)) for k, v in per_layer[n].items()}}
            for n in per_layer
        ]
        out['spans'] = [
            {'n': sp['n'], 's': int(round(sp['s'])), 'd': int(round(sp['d']))} for sp in spans
        ]
        if item.get('cpu_phases'):
            out['cpu'] = {k: int(round(v * 1e6)) for k, v in item['cpu_phases'].items()}
        out['a2a'] = [
            {'n': a['name'], 's': int(round(a['start'] * 1000)), 'd': int(round(a['dur'] * 1000))}
            for a in a2a
        ]
        return out

    # --------------------------------------------------------------------- stats
    def stats(self) -> Dict[str, int]:
        """Small counter dict, useful in tests and in the step log."""
        return {
            'written_steps': self.written_steps,
            'dropped_steps': self.dropped_steps,
            'pending': len(self._pending),
        }


class _on_stream:
    """Context helper: record on ``stream`` without disturbing stream ordering elsewhere."""

    def __init__(self, stream: Any):
        self.stream = stream
        self._prev = None

    def __enter__(self):
        import torch

        self._prev = torch.cuda.current_stream()
        torch.cuda.set_stream(self.stream)
        return self

    def __exit__(self, *exc):
        import torch

        torch.cuda.set_stream(self._prev)
        return False


class TimedIterator:
    """Transparent proxy that reports every ``next()`` duration to a :class:`PerfTracer`.

    Megatron indexes the data iterator per model chunk and calls ``next()` on it inside the
    schedule, so timing has to live here rather than around the whole step: that is the only
    place where "the loop waited for data" is visible per microbatch.
    """

    def __init__(self, iterator: Any, tracer: PerfTracer):
        self._iterator = iterator
        self._tracer = tracer

    def __next__(self):
        t0 = time.perf_counter()
        try:
            value = next(self._iterator)
        except StopIteration:
            raise  # end of data is not a data-preparation cost
        self._tracer.mark_data_fetch(time.perf_counter() - t0)
        return value

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self._iterator, name)

    def __len__(self):
        return len(self._iterator)
