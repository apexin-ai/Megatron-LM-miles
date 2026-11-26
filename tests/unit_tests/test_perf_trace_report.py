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
"""CPU tests for the perf-trace store and charts (no GPU, no training stack)."""

import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


report = _load('perf_trace_report', 'megatron/core/perf_trace/report.py')

pytest.importorskip('matplotlib')


def _record(step, rank, lag=1.0, n_layers=6):
    layers = []
    for i in range(n_layers):
        fwd = int(100_000 * lag)
        layers.append({'n': f'c0l{i}', 'fwd': fwd, 'bwd': int(fwd * 1.2)})
    end = sum(l['fwd'] + l['bwd'] for l in layers)
    return {
        'step': step,
        'rank': rank,
        'phases': {'__anchor__': 0, '__end__': end},
        'layers': layers,
        'a2a': [{'n': 'dispatch', 's': 10_000, 'd': 5_000}],
        'data': {'n': 2, 'total': 4_000, 'max': 3_000, 'items': [1_000, 3_000]},
        'step_wall': end * 2,
        'anchor_wall': 1.0,
    }


@pytest.fixture
def store(tmp_path):
    for step in (5, 6):
        for rank in range(4):
            lag = 2.0 if (rank == 1 and step == 6) else 1.0
            with open(tmp_path / f'perf-rank-{rank}.jsonl', 'a') as fh:
                fh.write(json.dumps(_record(step, rank, lag)) + '\n')
    s = report.TraceStore(str(tmp_path))
    s.refresh()
    return s


def test_store_loads_steps_and_ranks(store):
    assert store.steps() == [5, 6]
    assert store.ranks(6) == [0, 1, 2, 3]
    assert store.latest_step() == 6


def test_summary_reports_median_and_max(store):
    rows = {r['step']: r for r in store.summary()}
    assert rows[5]['ranks'] == 4
    # step 6 has one rank twice as slow: median stays, max doubles
    assert rows[6]['total']['max'] > 1.8 * rows[6]['total']['median']
    assert rows[6]['fwd']['max'] > 1.8 * rows[6]['fwd']['median']


def test_rank_totals_put_the_straggler_first(store):
    rows = store.rank_totals(6)
    assert rows[0][0] == 1  # rank 1 is the lagging one
    assert rows[0][1] > rows[-1][1]


def test_layer_matrix_aligns_layers_across_ranks(store):
    names, ranks, matrix = store.layer_matrix(6, 'fwd')
    assert names == [f'c0l{i}' for i in range(6)]
    assert ranks == [0, 1, 2, 3]
    assert len(matrix) == 4 and all(len(row) == 6 for row in matrix)
    assert matrix[ranks.index(1)][0] > matrix[ranks.index(0)][0]


def test_refresh_only_reads_new_lines(store, tmp_path):
    with open(tmp_path / 'perf-rank-0.jsonl', 'a') as fh:
        fh.write(json.dumps(_record(7, 0)) + '\n')
    added = store.refresh()
    assert added == 1
    assert store.latest_step() == 7
    assert store.refresh() == 0  # nothing new


def test_partial_line_is_left_for_the_next_refresh(store, tmp_path):
    payload = json.dumps(_record(8, 0))
    with open(tmp_path / 'perf-rank-0.jsonl', 'a') as fh:
        fh.write(payload[: len(payload) // 2])  # no trailing newline yet
    assert store.refresh() == 0
    assert 8 not in store.steps()
    with open(tmp_path / 'perf-rank-0.jsonl', 'a') as fh:
        fh.write(payload[len(payload) // 2 :] + '\n')
    assert store.refresh() == 1
    assert 8 in store.steps()


def test_charts_render_png(store):
    for png in (
        report.draw_step_timeline(store, 6),
        report.draw_step_trends(store),
        report.draw_rank_skew(store, 6),
        report.draw_layer_heatmap(store, 6, 'fwd'),
        report.draw_layer_distribution(store, store.steps(), 1, 'fwd'),
    ):
        assert png[:8] == b'\x89PNG\r\n\x1a\n'
        assert len(png) > 5_000


def test_missing_step_is_an_error_not_a_crash(store):
    with pytest.raises(ValueError):
        report.draw_step_timeline(store, 999)
