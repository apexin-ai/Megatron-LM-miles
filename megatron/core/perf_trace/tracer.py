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
        skip_first_steps: int = 0,
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
        # The first steps after install can carry one-off costs (lazy NCCL setup, a fresh event
        # pool, first-touch autograd hooks) that look like a 100 ms collective. Skipping them is
        # cheaper than explaining a phantom outlier later.
        self.skip_first_steps = skip_first_steps
        self._steps_seen = 0
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
        if self.skip_first_steps:
            self._steps_seen += 1
            if self._steps_seen <= self.skip_first_steps:
                return False
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

    def mark_sub_start(self, layer: str, part: str) -> int:
        """Open a sub-layer span, e.g. the attention or MoE block inside layer ``l12``.

        Layer-level timing says which layer is expensive; this says *what inside it* is, which is
        the difference between "attention is slow" and "the experts are unbalanced".
        """
        return self._record('sub', f'{layer}|{part}:start')

    def mark_sub_end(self, layer: str, part: str) -> int:
        return self._record('sub', f'{layer}|{part}:end')

    def mark_layer_backward_end(self, name: str) -> int:
        return self._record('layer_bwd_end', name)

    def mark_a2a(self, name: str, stream: Any = None, layer: Optional[str] = None) -> int:
        """Mark a MoE all-to-all, optionally attributed to the layer that issued it."""
        return self._record('a2a', f'{layer}|{name}' if layer else name, stream=stream)

    def mark_sub_start(self, layer: str, part: str) -> int:
        """Open a sub-layer span, e.g. the attention or MoE block inside layer ``l12``.

        Layer-level timing says which layer is expensive; this says *what inside it* is, which is
        the difference between "attention is slow" and "the experts are unbalanced".
        """
        return self._record('sub', f'{layer}|{part}:start')

    def mark_sub_end(self, layer: str, part: str) -> int:
        return self._record('sub', f'{layer}|{part}:end')

    def mark_layer_backward_end(self, name: str) -> int:
        """Mark the end of a layer's backward."""
        return self._record('layer_bwd_end', name)

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
        subs: List[Dict[str, float]] = []
        open_sub: Dict[str, float] = {}
        bwd_ends: List[tuple] = []
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
                    start = open_fwd.pop(name)
                    slot = per_layer.setdefault(name, {})
                    slot['fwd'] = t - start
                    # Absolute position of this layer's forward, relative to the step anchor.
                    # Without it a timeline can only stack durations and cannot answer "when did
                    # this rank start layer 20", which is the whole point of the chart.
                    slot['fs'] = start
            elif kind == 'layer_bwd_end':
                bwd_ends.append((name, t))
            elif kind == 'sub':
                layer_name, sep, op = name.partition('|')
                if not sep:
                    continue
                key = f'{layer_name}|{op}'
                if op.endswith(':start'):
                    open_sub[key[:-6]] = t
                elif op.endswith(':end') and key[:-4] in open_sub:
                    st = open_sub.pop(key[:-4])
                    lname, _, part = key[:-4].partition('|')
                    subs.append({'l': lname, 'n': part, 's': st * 1000.0, 'd': (t - st) * 1000.0})
            elif kind == 'span':
                if name.endswith(':start'):
                    open_span[name[:-6]] = t
                elif name.endswith(':end') and name[:-4] in open_span:
                    st = open_span.pop(name[:-4])
                    spans.append({'n': name[:-4], 's': st * 1000.0, 'd': (t - st) * 1000.0})
            elif kind == 'a2a':
                # names look like "l12|dispatch:start" when the layer was known, else
                # "dispatch:start"
                layer_name, sep, op = name.partition('|')
                if not sep:
                    layer_name, op = '', name
                key = f'{layer_name}|{op}' if layer_name else op
                if op.endswith(':start'):
                    open_a2a[key[:-6]] = t
                elif op.endswith(':end') and key[:-4] in open_a2a:
                    st = open_a2a.pop(key[:-4])
                    base = key[:-4]
                    lname, sep2, opname = base.partition('|')
                    entry = {'name': opname if sep2 else lname, 'start': st, 'dur': t - st}
                    if sep2:
                        entry['layer'] = lname
                    a2a.append(entry)

        # Per-layer backward from the *end* markers. Backward finishes the last layer first, so
        # sorting the end times ascending gives the reverse execution order: each layer's duration
        # is the gap to the previous end. The first one to finish has no previous end, so its
        # duration is left unset rather than guessed from the forward.
        ends = sorted([(name, t) for name, t in bwd_ends if t is not None], key=lambda x: x[1])
        for idx in range(1, len(ends)):
            name_i, t_i = ends[idx]
            _, t_prev = ends[idx - 1]
            if t_i > t_prev:
                slot = per_layer.setdefault(name_i, {})
                slot['bwd'] = t_i - t_prev
                slot['bs'] = t_prev

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
        out['anchor_wall'] = item['anchor_wall']
        out['phases'] = phases
        # Keep the file small: microsecond integers, layer order preserved.
        out['layers'] = [
            dict({'n': n}, **{k: int(round(v * 1000)) for k, v in per_layer[n].items()})
            for n in per_layer
        ]
        out['subs'] = [
            {'l': x['l'], 'n': x['n'], 's': int(round(x['s'])), 'd': int(round(x['d']))}
            for x in subs
        ]
        out['spans'] = [
            {'n': sp['n'], 's': int(round(sp['s'])), 'd': int(round(sp['d']))} for sp in spans
        ]
        if item.get('cpu_phases'):
            out['cpu'] = {k: int(round(v * 1e6)) for k, v in item['cpu_phases'].items()}
        out['a2a'] = [
            dict(
                {
                    'n': a['name'],
                    's': int(round(a['start'] * 1000)),
                    'd': int(round(a['dur'] * 1000)),
                },
                **({'layer': a['layer']} if a.get('layer') else {}),
            )
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
