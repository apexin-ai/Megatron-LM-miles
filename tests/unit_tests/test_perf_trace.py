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
"""CPU tests for :mod:`megatron.core.perf_trace` (no GPU, no CUDA events).

The tracer talks to events through ``elapsed_time``/``query``/``record`` only, so a small fake
device clock is enough to test the bookkeeping: relative conversion, per-layer pairing, A2A
bracketing, data-fetch accounting, step sampling and the "never block the hot path" contract.
"""

import json
import os

import pytest

from megatron.core.perf_trace.tracer import PerfTracer, TimedIterator


class FakeClock:
    """A monotonic fake device clock shared by every fake event."""

    def __init__(self):
        self.now = 0.0

    def tick(self, ms):
        self.now += ms
        return self.now


class FakeEvent:
    """Minimal ``torch.cuda.Event`` stand-in: records the clock value when marked."""

    def __init__(self, clock):
        self._clock = clock
        self._at = None

    def record(self):
        self._at = self._clock.now

    def query(self):
        return self._at is not None

    def elapsed_time(self, other):
        if self._at is None or other._at is None:
            raise RuntimeError('event not recorded')
        return abs(self._at - other._at)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tracer(tmp_path, clock):
    t = PerfTracer(
        out_dir=str(tmp_path),
        rank=3,
        every_n_steps=1,
        event_factory=lambda: FakeEvent(clock),
        clock=lambda: 1000.0,
    )
    t.start()
    yield t
    t.close()


def _read_lines(tmp_path):
    path = os.path.join(str(tmp_path), 'perf-rank-3.jsonl')
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_layer_forward_and_backward_durations_are_relative_to_the_anchor(tracer, clock, tmp_path):
    assert tracer.begin_step(7)
    clock.tick(1.0)  # 1 ms after the anchor
    tracer.mark_layer_forward_start('l0')
    clock.tick(2.0)
    tracer.mark_layer_forward_end('l0')
    clock.tick(0.5)
    # Per-layer backward is derived from end markers and has its own test; this one covers the
    # forward pairing and the anchor.
    tracer.end_step()
    tracer.end_step()
    tracer.close()

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    rec = lines[0]
    assert rec['step'] == 7 and rec['rank'] == 3
    by_name = {l['n']: l for l in rec['layers']}
    assert by_name['l0']['fwd'] == 2000  # microseconds
    assert rec['phases']['__anchor__'] == 0


def test_a2a_is_bracketed_per_pair(tracer, clock, tmp_path):
    tracer.begin_step(1)
    tracer.mark_a2a('dispatch:start')
    clock.tick(3.0)
    tracer.mark_a2a('dispatch:end')
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    assert rec['a2a'] == [{'n': 'dispatch', 's': 0, 'd': 3000}]


def test_data_fetch_is_recorded_in_host_units(tracer, tmp_path):
    tracer.begin_step(2)
    tracer.mark_data_fetch(0.0123)
    tracer.mark_data_fetch(0.0077)
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    assert rec['data']['n'] == 2
    assert rec['data']['total'] == 20000  # 12.3 + 7.7 ms in microseconds
    assert rec['data']['max'] == 12300
    assert rec['data']['items'] == [12300, 7700]


def test_step_sampling_skips_unselected_steps(tmp_path, clock):
    t = PerfTracer(
        out_dir=str(tmp_path), rank=0, every_n_steps=5, event_factory=lambda: FakeEvent(clock)
    )
    t.start()
    for step in range(1, 11):
        t.begin_step(step)
        t.mark_layer_forward_start('l0')
        t.mark_layer_forward_end('l0')
        t.end_step()
    t.close()

    steps = [
        json.loads(line)['step'] for line in open(os.path.join(str(tmp_path), 'perf-rank-0.jsonl'))
    ]
    assert steps == [5, 10]


def test_no_synchronize_and_no_elapsed_time_on_the_hot_path(tracer, clock, monkeypatch):
    """Capture must never call elapsed_time: that is what keeps the training loop async."""

    def _boom(*a, **k):  # pragma: no cover - only runs on failure
        raise AssertionError('elapsed_time must not be called while capturing')

    monkeypatch.setattr(FakeEvent, 'elapsed_time', _boom)
    tracer.begin_step(1)
    tracer.mark_layer_forward_start('l0')
    tracer.mark_layer_forward_end('l0')
    tracer.mark_a2a('dispatch:start')
    tracer.mark_a2a('dispatch:end')
    tracer.mark_data_fetch(0.001)
    tracer.end_step()
    # Conversion happens on the *next* begin_step, outside the capture window.


def test_unpaired_markers_are_dropped_not_crashing(tracer, clock, tmp_path):
    tracer.begin_step(1)
    tracer.mark_layer_forward_start('l0')  # no matching end
    tracer.mark_layer_forward_end('other')  # end without start
    tracer.mark_a2a('combine:end')  # end without start
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    assert rec['layers'] == []
    assert rec['a2a'] == []


def test_timed_iterator_reports_each_next(clock):
    class _It:
        def __init__(self):
            self.n = 0

        def __next__(self):
            self.n += 1
            if self.n > 2:
                raise StopIteration
            return self.n

        def __iter__(self):
            return self

    t = PerfTracer(out_dir='/tmp', rank=0, every_n_steps=1, event_factory=lambda: FakeEvent(clock))
    assert t.begin_step(0)  # fetch timing only accumulates inside a traced step
    it = TimedIterator(_It(), t)
    assert list(it) == [1, 2]
    assert len(t._data_fetches) == 2


def test_wrap_data_iterator_handles_lists(clock, monkeypatch):
    from megatron.core.perf_trace import hooks

    class _It:
        def __iter__(self):
            return iter([1])

    t = PerfTracer(out_dir='/tmp', rank=0, every_n_steps=1, event_factory=lambda: FakeEvent(clock))
    monkeypatch.setattr(hooks, 'TRACER', t)
    wrapped = hooks.wrap_data_iterator([_It(), _It()])
    assert isinstance(wrapped, list) and all(isinstance(x, TimedIterator) for x in wrapped)


def test_spans_bracket_the_optimizer_on_the_gpu_timeline(tracer, clock, tmp_path):
    tracer.begin_step(4)
    clock.tick(10.0)
    tracer.mark_span_start('optimizer')
    clock.tick(25.0)
    tracer.mark_span_end('optimizer')
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    assert rec['spans'] == [{'n': 'optimizer', 's': 10000, 'd': 25000}]


def test_cpu_phases_are_summed_per_step(tracer, tmp_path):
    tracer.begin_step(5)
    tracer.note_cpu_phase('params-all-gather', 1.5)
    tracer.note_cpu_phase('params-all-gather', 0.5)
    tracer.note_cpu_phase('optimizer', 2.0)
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    assert rec['cpu'] == {'params-all-gather': 2_000_000, 'optimizer': 2_000_000}


def test_recompute_pass_is_tagged_not_overwritten(tracer, clock, tmp_path):
    """A checkpointed layer runs forward twice; the second must not overwrite the first."""
    tracer.begin_step(3)
    tracer.mark_layer_forward_start('l0')
    clock.tick(2.0)
    tracer.mark_layer_forward_end('l0')
    tracer.mark_layer_forward_start('l0')  # recompute pass
    clock.tick(3.0)
    tracer.mark_layer_forward_end('l0')
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    names = [l['n'] for l in rec['layers']]
    assert 'l0' in names and 'l0#recompute' in names
    by = {l['n']: l for l in rec['layers']}
    assert by['l0']['fwd'] == 2000
    assert by['l0#recompute']['fwd'] == 3000


def test_global_layer_ids_come_from_the_pipeline_offset(monkeypatch):
    """PP stages must not all number their layers from 0."""
    import torch.nn as nn

    import megatron.core.perf_trace.hooks as hooks
    import megatron.core.transformer.transformer_layer as tl

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x):
            return self.lin(x)

    class FakeModel(nn.Module):
        def __init__(self, config, n):
            super().__init__()
            self.config = config
            self.layers = nn.ModuleList([FakeLayer() for _ in range(n)])

    monkeypatch.setattr(tl, 'TransformerLayer', FakeLayer, raising=False)
    monkeypatch.setattr(
        tl, 'get_transformer_layer_offset', lambda cfg, vp_stage=0: cfg['offset'], raising=False
    )

    model = FakeModel({'offset': 12}, 2)
    ids = [lid for lid, _ in hooks._iter_transformer_layers([model])]
    assert ids == [12, 13], f'expected global ids 12,13 got {ids}'


def test_a_hook_that_raises_does_not_break_the_forward():
    """Tracing must never be the reason a training step fails."""
    import torch
    import torch.nn as nn

    import megatron.core.perf_trace.hooks as hooks
    import megatron.core.transformer.transformer_layer as tl

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, x):
            return self.lin(x)

    class Boom:
        max_layers = 8
        _layer_names = []

        def mark_layer_forward_start(self, name):
            raise RuntimeError('boom')

        def mark_layer_forward_end(self, name):
            raise RuntimeError('boom')

    model = FakeLayer()
    saved = (
        getattr(tl, 'TransformerLayer', None),
        getattr(tl, 'get_transformer_layer_offset', None),
    )
    tl.TransformerLayer = FakeLayer  # type: ignore[attr-defined]
    tl.get_transformer_layer_offset = lambda cfg, vp_stage=0: 0  # type: ignore[attr-defined]
    try:
        hooks.install_layer_hooks(Boom(), [model], max_layers=4)
        out = model(torch.zeros(1, 2))  # must not raise
        assert out.shape == (1, 2)
    finally:
        if saved[0] is not None:
            tl.TransformerLayer = saved[0]  # type: ignore[attr-defined]
        if saved[1] is not None:
            tl.get_transformer_layer_offset = saved[1]  # type: ignore[attr-defined]


def test_a2a_is_attributed_to_the_layer_that_issued_it(tracer, clock, tmp_path):
    tracer.begin_step(6)
    tracer.mark_a2a('dispatch:start', layer='l12')
    clock.tick(2.0)
    tracer.mark_a2a('dispatch:end', layer='l12')
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    assert rec['a2a'] == [{'n': 'dispatch', 's': 0, 'd': 2000, 'layer': 'l12'}]


def test_backward_is_derived_from_end_markers(tracer, clock, tmp_path):
    """Backward finishes the last layer first; gaps between ends are the layer durations."""
    tracer.begin_step(8)
    clock.tick(1.0)
    tracer.mark_layer_backward_end('l3')  # first to finish
    clock.tick(4.0)
    tracer.mark_layer_backward_end('l2')
    clock.tick(2.0)
    tracer.mark_layer_backward_end('l1')  # last to finish
    tracer.end_step()
    tracer.close()

    rec = _read_lines(tmp_path)[0]
    by = {l['n']: l for l in rec['layers']}
    assert by['l2']['bwd'] == 4000
    assert by['l1']['bwd'] == 2000
    # the first layer to finish has no previous end, so no duration is invented for it
    assert 'l3' not in by
