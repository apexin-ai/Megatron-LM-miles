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
"""Read ``--perf-trace-dir`` output and turn it into training-wide pictures.

This module is the data + drawing half of the analysis service (the HTTP wrapper lives in
``tools/perf_trace_dashboard.py``). It answers the questions a single step cannot:

* is the step getting slower over training, and is it forward, backward, the all-to-all or the
  host-side data fetch that moved?
* which rank is the straggler, and which layer makes it a straggler?
* is one layer consistently the hot one (a fixed cost) or does the heat move (a load problem)?

The on-disk format is documented in :class:`megatron.core.perf_trace.tracer.PerfTracer`: one
JSON line per rank per step, every duration an integer number of microseconds.
"""

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------- data layer


class TraceStore:
    """Incremental reader for ``perf-rank-*.jsonl`` written by a live training run.

    Files are appended to while training runs, so the store re-reads a file only when its size
    changed and keeps a per-file offset; a dashboard can therefore poll cheaply.
    """

    def __init__(self, dirpath: str, max_steps: int = 4096):
        self.dirpath = dirpath
        self.max_steps = max_steps
        # step -> rank -> record
        self._records: Dict[int, Dict[int, dict]] = defaultdict(dict)
        self._offsets: Dict[str, int] = {}
        self._steps: List[int] = []

    # ------------------------------------------------------------------ ingestion
    def refresh(self) -> int:
        """Pick up new lines; returns the number of records added."""
        added = 0
        for path in glob.glob(os.path.join(self.dirpath, 'perf-rank-*.jsonl')):
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            offset = self._offsets.get(path, 0)
            if size <= offset:
                continue
            try:
                with open(path) as fh:
                    fh.seek(offset)
                    for line in fh:
                        if not line.endswith('\n'):
                            break  # a partial line: come back for it next refresh
                        offset += len(line)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        step, rank = rec.get('step'), rec.get('rank')
                        if step is None or rank is None:
                            continue
                        self._records[step][rank] = rec
                        added += 1
                    self._offsets[path] = offset
            except OSError:
                continue
        if added:
            self._steps = sorted(self._records)
            if len(self._steps) > self.max_steps:
                for stale in self._steps[: -self.max_steps]:
                    self._records.pop(stale, None)
                self._steps = self._steps[-self.max_steps :]
        return added

    # --------------------------------------------------------------------- queries
    def steps(self) -> List[int]:
        return list(self._steps)

    def latest_step(self) -> Optional[int]:
        return self._steps[-1] if self._steps else None

    def ranks(self, step: int) -> List[int]:
        return sorted(self._records.get(step, {}))

    def get(self, step: int, rank: int) -> Optional[dict]:
        return self._records.get(step, {}).get(rank)

    def summary(self, steps: Optional[List[int]] = None) -> List[dict]:
        """Per-step aggregates across ranks: total, forward, backward, a2a, data fetch.

        Min/median/max are reported per step because the spread *is* the signal: a growing max
        with a flat median means one rank is falling behind, not that the step got heavier.
        """
        out = []
        for step in steps if steps is not None else self._steps:
            recs = self._records.get(step, {})
            if not recs:
                continue
            totals, fwds, bwds, a2as, datas = [], [], [], [], []
            cpu_acc: Dict[str, List[float]] = defaultdict(list)
            span_acc: Dict[str, List[float]] = defaultdict(list)
            for rec in recs.values():
                layers = rec.get('layers', [])
                fwd = sum(l.get('fwd', 0) for l in layers if 'fwd' in l)
                bwd = sum(l.get('bwd', 0) for l in layers if 'bwd' in l)
                a2a = sum(a.get('d', 0) for a in rec.get('a2a', []))
                data = rec.get('data', {}).get('total', 0)
                end = rec.get('phases', {}).get('__end__', 0)
                totals.append(end or (fwd + bwd))
                fwds.append(fwd)
                bwds.append(bwd)
                a2as.append(a2a)
                datas.append(data)
                for key, usec in (rec.get('cpu') or {}).items():
                    cpu_acc[key].append(usec)
                for sp in rec.get('spans', []):
                    span_acc[sp['n']].append(sp['d'])
            out.append(
                {
                    'step': step,
                    'ranks': len(recs),
                    'total': _stats(totals),
                    'fwd': _stats(fwds),
                    'bwd': _stats(bwds),
                    'a2a': _stats(a2as),
                    'data': _stats(datas),
                    'cpu': {k: _stats(v) for k, v in cpu_acc.items()},
                    'spans': {k: _stats(v) for k, v in span_acc.items()},
                }
            )
        return out

    def rank_totals(self, step: int) -> List[Tuple[int, float, float, float, float]]:
        """Per rank: (rank, total, fwd, bwd, a2a) in seconds, sorted by total descending."""
        rows = []
        for rank in self.ranks(step):
            rec = self.get(step, rank) or {}
            layers = rec.get('layers', [])
            fwd = sum(l.get('fwd', 0) for l in layers if 'fwd' in l)
            bwd = sum(l.get('bwd', 0) for l in layers if 'bwd' in l)
            a2a = sum(a.get('d', 0) for a in rec.get('a2a', []))
            end = rec.get('phases', {}).get('__end__', 0) or (fwd + bwd)
            rows.append((rank, end / 1e6, fwd / 1e6, bwd / 1e6, a2a / 1e6))
        return sorted(rows, key=lambda r: -r[1])

    def a2a_by_layer(self, step: int) -> Dict[str, float]:
        """Per-layer total all-to-all time (seconds), summed over dispatch and combine.

        Without this an A2A spike can only be blamed on "some layer"; with it the report can point
        at the layer that issued it, which is what tells an expert-load problem apart from a
        kernel problem.
        """
        out: Dict[str, float] = defaultdict(float)
        ranks = self.ranks(step)
        for rank in ranks:
            rec = self.get(step, rank) or {}
            for a in rec.get('a2a', []):
                layer = a.get('layer') or 'unattributed'
                out[layer] += a.get('d', 0)
        n = max(1, len(ranks))
        return {k: v / 1e6 / n for k, v in out.items()}

    def layer_matrix(
        self, step: int, metric: str = 'fwd'
    ) -> Tuple[List[str], List[int], List[List[float]]]:
        """Matrix ``[rank][layer]`` of a layer metric (seconds), plus the layer names.

        Layers are matched by name across ranks so a rank missing a layer still lines up; a
        missing value becomes ``nan`` rather than 0 so it is visibly absent in a heatmap.
        """
        ranks = self.ranks(step)
        names: List[str] = []
        seen = set()
        per_rank: Dict[int, Dict[str, float]] = {}
        for rank in ranks:
            rec = self.get(step, rank) or {}
            row = {}
            for layer in rec.get('layers', []):
                name = layer.get('n')
                if name is None:
                    continue
                if name not in seen:
                    seen.add(name)
                    names.append(name)
                if metric in layer:
                    row[name] = layer[metric] / 1e6
                elif metric == 'fwd_start' and 'fs' in layer:
                    row[name] = layer['fs'] / 1e6
                elif metric == 'bwd_start' and 'bs' in layer:
                    row[name] = layer['bs'] / 1e6
            per_rank[rank] = row
        nan = float('nan')
        matrix = [[per_rank[r].get(n, nan) for n in names] for r in ranks]
        return names, ranks, matrix


def _stats(values: List[float]) -> Dict[str, float]:
    """min / median / mean / max of a list, in seconds."""
    if not values:
        return {'min': 0.0, 'median': 0.0, 'mean': 0.0, 'max': 0.0}
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {'min': s[0] / 1e6, 'median': med / 1e6, 'mean': sum(s) / n / 1e6, 'max': s[-1] / 1e6}


# ------------------------------------------------------------------------- drawing layer
_DEFAULT_COLORS = {
    'fwd': '#4C78A8',
    'bwd': '#54A24B',
    'a2a': '#F58518',
    'data': '#BAB0AC',
    'span': '#B279A2',
    'warn': '#E45756',
}


def _plt():
    """Import matplotlib lazily so the tracer never depends on it."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def draw_step_timeline(store: TraceStore, step: int, align: str = 'start') -> bytes:
    """One row per rank: layer blocks, an A2A lane and a data-fetch bar. PNG bytes."""
    import matplotlib.patches as mpatches

    plt = _plt()
    ranks = store.ranks(step)
    if not ranks:
        raise ValueError(f'no records for step {step}')
    fig_h = max(2.2, 0.26 * len(ranks) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    ends = []
    for row, rank in enumerate(ranks):
        rec = store.get(step, rank) or {}
        layers = rec.get('layers', [])
        end_us = rec.get('phases', {}).get('__end__', 0)
        ends.append((end_us / 1e6, rank))
        cursor = 0.0
        for layer in layers:
            for key in ('fwd', 'bwd'):
                if key not in layer:
                    continue
                dur = layer[key] / 1e6
                ax.add_patch(
                    mpatches.Rectangle(
                        (cursor, row - 0.34),
                        dur,
                        0.30,
                        facecolor=_DEFAULT_COLORS[key],
                        edgecolor='none',
                    )
                )
                cursor += dur
        # Comm lane: draw the A2A on a separate row band so a gap between compute and comm, or
        # an overlap with it, is visible at a glance.
        for sp in rec.get('spans', []):
            ax.add_patch(
                mpatches.Rectangle(
                    (sp['s'] / 1e6, row + 0.02),
                    sp['d'] / 1e6,
                    0.28,
                    facecolor=_DEFAULT_COLORS['span'],
                    edgecolor='none',
                )
            )
        for a in rec.get('a2a', []):
            ax.add_patch(
                mpatches.Rectangle(
                    (a['s'] / 1e6, row + 0.02),
                    a['d'] / 1e6,
                    0.28,
                    facecolor=_DEFAULT_COLORS['a2a'],
                    edgecolor='none',
                )
            )
        data = rec.get('data', {})
        if data.get('total'):
            ax.add_patch(
                mpatches.Rectangle(
                    (0, row - 0.48),
                    data['total'] / 1e6,
                    0.10,
                    facecolor=_DEFAULT_COLORS['data'],
                    edgecolor='none',
                )
            )
        ax.text(
            -0.005 * max(1.0, max(e for e, _ in ends)),
            row,
            str(rank),
            ha='right',
            va='center',
            fontsize=7,
        )

    if ends:
        end_s, end_rank = max(ends)
        ax.axvline(end_s, color=_DEFAULT_COLORS['warn'], linestyle='--', linewidth=1)
        ax.text(
            end_s,
            len(ranks) + 0.1,
            f' last: rank {end_rank} ({end_s:.2f}s)',
            color=_DEFAULT_COLORS['warn'],
            fontsize=8,
            ha='right',
            va='bottom',
        )
    ax.set_yticks([])
    ax.set_xlabel('time since step start (s)')
    ax.set_ylabel('rank')
    ax.set_ylim(-1.0, len(ranks) + 0.4)
    ax.set_title(f'step {step}: blue=fwd layers  green=bwd layers  orange=MoE A2A  grey=data fetch')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def draw_step_trends(store: TraceStore, steps: Optional[List[int]] = None) -> bytes:
    """Step time over training, split into forward / backward / A2A / data, median across ranks."""
    plt = _plt()
    rows = store.summary(steps)
    if not rows:
        raise ValueError('no steps recorded yet')
    xs = [r['step'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax = axes[0]
    ax.plot(xs, [r['total']['median'] for r in rows], label='median', color='#4C78A8')
    ax.plot(
        xs,
        [r['total']['max'] for r in rows],
        label='max',
        color=_DEFAULT_COLORS['warn'],
        linestyle='--',
        linewidth=1,
    )
    ax.plot(xs, [r['total']['min'] for r in rows], label='min', color='#54A24B', linewidth=1)
    ax.set_ylabel('step time (s)')
    ax.set_title('step time across ranks (median / min / max)')
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key, label in (
        ('fwd', 'forward'),
        ('bwd', 'backward'),
        ('a2a', 'MoE A2A'),
        ('data', 'data fetch'),
    ):
        ax.plot(xs, [r[key]['median'] for r in rows], label=label, color=_DEFAULT_COLORS.get(key))
    ax.set_ylabel('seconds (median across ranks)')
    ax.set_xlabel('step')
    ax.set_title('where the step goes')
    ax.legend()
    ax.grid(alpha=0.25)

    # CPU-measured phases from Megatron's timers: the only view of grad-sync / param-sync.
    ax = axes[2]
    interesting = ('params-all-gather', 'all-grads-sync', 'backward-compute', 'forward-compute')
    for key in interesting:
        ys = [r['cpu'].get(key, {}).get('median', 0.0) for r in rows]
        if any(ys):
            ax.plot(xs, ys, label=key, linewidth=1.2)
    for key, span in rows[-1]['spans'].items():
        if key in ('optimizer',):
            ys = [r['spans'].get(key, {}).get('median', 0.0) for r in rows]
            ax.plot(xs, ys, label=f'{key} (gpu span)', linestyle='--', linewidth=1.2)
    ax.set_ylabel('seconds (median)')
    ax.set_title('phase breakdown (CPU timers + optimizer GPU span)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def draw_rank_skew(store: TraceStore, step: int) -> bytes:
    """Bar chart of per-rank total time for one step, stragglers first."""
    plt = _plt()
    rows = store.rank_totals(step)
    if not rows:
        raise ValueError(f'no records for step {step}')
    ranks = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(ranks) + 3), 4))
    if len(ranks) <= 64:
        ax.bar(range(len(ranks)), totals, color='#4C78A8')
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(r) for r in ranks], rotation=90, fontsize=6)
        ax.set_xlabel('rank (sorted by total time)')
    else:
        ax.plot(range(len(ranks)), totals, marker='.', linestyle='none', color='#4C78A8')
        ax.set_xlabel('rank index (sorted by total time)')
    med = sorted(totals)[len(totals) // 2]
    ax.axhline(med, color='#54A24B', linestyle='--', linewidth=1, label=f'median {med:.2f}s')
    ax.set_ylabel('step time (s)')
    ax.set_title(f'step {step}: per-rank total (worst {totals[0]:.2f}s vs median {med:.2f}s)')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def draw_layer_heatmap(store: TraceStore, step: int, metric: str = 'fwd') -> bytes:
    """Heatmap rank x layer. A hot column is a fixed layer cost; a hot row is a straggler."""
    plt = _plt()
    names, ranks, matrix = store.layer_matrix(step, metric)
    if not names or not ranks:
        raise ValueError(f'no per-layer records for step {step}')
    fig, ax = plt.subplots(figsize=(max(9, 0.14 * len(names) + 4), max(3, 0.12 * len(ranks) + 2)))
    im = ax.imshow(matrix, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(r) for r in ranks], fontsize=6)
    step_x = max(1, len(names) // 40)
    ax.set_xticks(range(0, len(names), step_x))
    ax.set_xticklabels(names[::step_x], rotation=90, fontsize=6)
    ax.set_xlabel('layer')
    ax.set_ylabel('rank')
    ax.set_title(f'step {step}: {metric} time per layer per rank (s)')
    fig.colorbar(im, ax=ax, label='seconds')
    fig.tight_layout()
    return _png(fig)


def draw_layer_start_heatmap(store: TraceStore, step: int, metric: str = 'fwd_start') -> bytes:
    """Heatmap of *when* each rank started each layer (seconds since the step anchor).

    This is the imbalance view: a rank that is late shows up as a bright row across the whole
    width, while a single hot cell means one layer started late on that rank. Duration heatmaps
    cannot show waiting; start times can.
    """
    plt = _plt()
    names, ranks, matrix = store.layer_matrix(step, metric)
    if not names or not ranks:
        raise ValueError(f'no start offsets for step {step} (re-run with a newer tracer)')
    fig, ax = plt.subplots(figsize=(max(9, 0.14 * len(names) + 4), max(3, 0.12 * len(ranks) + 2)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([str(r) for r in ranks], fontsize=6)
    step_x = max(1, len(names) // 40)
    ax.set_xticks(range(0, len(names), step_x))
    ax.set_xticklabels(names[::step_x], rotation=90, fontsize=6)
    ax.set_xlabel('layer')
    ax.set_ylabel('rank')
    ax.set_title(f'step {step}: when each rank STARTED each layer (s since anchor)')
    fig.colorbar(im, ax=ax, label='start time (s)')
    fig.tight_layout()
    return _png(fig)


def draw_overlap(store: TraceStore, step: int, rank: int) -> bytes:
    """One rank's lanes over time: compute, comm and the gaps between them.

    The gaps are the point. A collective that does not overlap compute shows up as a comm block
    sitting alone with compute idle on both sides; good overlap shows orange nested inside blue.
    """
    import matplotlib.patches as mpatches

    plt = _plt()
    rec = store.get(step, rank)
    if not rec:
        raise ValueError(f'no record for rank {rank} step {step}')
    layers = rec.get('layers', [])
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for i, layer in enumerate(layers):
        if 'fs' in layer:
            ax.add_patch(
                mpatches.Rectangle(
                    (layer['fs'] / 1e6, 0.55),
                    layer.get('fwd', 0) / 1e6,
                    0.30,
                    facecolor=_DEFAULT_COLORS['fwd'],
                    edgecolor='none',
                )
            )
        if 'bs' in layer:
            ax.add_patch(
                mpatches.Rectangle(
                    (layer['bs'] / 1e6, 0.15),
                    layer.get('bwd', 0) / 1e6,
                    0.30,
                    facecolor=_DEFAULT_COLORS['bwd'],
                    edgecolor='none',
                )
            )
    for a in rec.get('a2a', []):
        ax.add_patch(
            mpatches.Rectangle(
                (a['s'] / 1e6, 0.55),
                a['d'] / 1e6,
                0.30,
                facecolor=_DEFAULT_COLORS['a2a'],
                edgecolor='none',
                alpha=0.85,
            )
        )
    for sp in rec.get('spans', []):
        ax.add_patch(
            mpatches.Rectangle(
                (sp['s'] / 1e6, 0.15),
                sp['d'] / 1e6,
                0.30,
                facecolor=_DEFAULT_COLORS['span'],
                edgecolor='none',
            )
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('time since step anchor (s)')
    ax.set_title(
        f'rank {rank} step {step}: blue=fwd (top)  green=bwd (bottom)  '
        'orange=A2A overlaid where it actually runs  purple=optimizer'
    )
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def trace_json(store: TraceStore, step: int) -> dict:
    """Compact payload for the interactive viewer: per rank, per lane, absolute intervals.

    Everything is in microseconds relative to the step anchor, which is the same origin the
    records use, so the browser needs no clock conversion -- only pixel mapping.
    """
    ranks = []
    for rank in store.ranks(step):
        rec = store.get(step, rank) or {}
        fwd, bwd = [], []
        for layer in rec.get('layers', []):
            if 'fs' in layer and layer.get('fwd'):
                fwd.append({'n': layer['n'], 's': layer['fs'], 'd': layer['fwd']})
            if 'bs' in layer and layer.get('bwd'):
                bwd.append({'n': layer['n'], 's': layer['bs'], 'd': layer['bwd']})
        comm = [
            {'n': a['n'], 'layer': a.get('layer', ''), 's': a['s'], 'd': a['d']}
            for a in rec.get('a2a', [])
        ]
        other = [{'n': sp['n'], 's': sp['s'], 'd': sp['d']} for sp in rec.get('spans', [])]
        ranks.append(
            {
                'rank': rank,
                'end': rec.get('phases', {}).get('__end__', 0),
                'fwd': fwd,
                'bwd': bwd,
                'comm': comm,
                'other': other,
            }
        )
    duration = max((r['end'] for r in ranks), default=0) or 1
    return {'step': step, 'duration_us': duration, 'ranks': ranks}


# --------------------------------------------------------------------- perfetto export
# Perfetto's UI (ui.perfetto.dev, Apache-2.0) reads Chrome Trace Event JSON. Emitting that
# format gives the whole Perfetto experience -- zoom, pan, per-slice duration, search, SQL --
# without writing a viewer. Tracks are (pid=rank, tid=lane), which is exactly our data model.
_PERFETTO_TIDS = {'fwd': 1, 'bwd': 2, 'comm': 3, 'other': 4, 'step': 5}
_PERFETTO_TID_NAMES = {1: 'forward', 2: 'backward', 3: 'comm', 4: 'optimizer/other', 5: 'step'}


def to_perfetto(store: TraceStore, steps: Optional[List[int]] = None) -> dict:
    """Convert records into a Chrome Trace Event document Perfetto can open.

    With several steps the timelines are laid end to end and a ``step N`` slice marks each
    boundary, so one file covers a whole training window instead of a single step.
    """
    steps = steps if steps is not None else store.steps()
    events: List[dict] = []
    offset = 0.0
    for step in steps:
        ranks = store.ranks(step)
        if not ranks:
            continue
        step_len = 0.0
        for rank in ranks:
            rec = store.get(step, rank) or {}
            end = rec.get('phases', {}).get('__end__', 0) or 0
            step_len = max(step_len, end)
        for rank in ranks:
            rec = store.get(step, rank) or {}
            for lane, key, name_key in (('fwd', 'fs', 'fwd'), ('bwd', 'bs', 'bwd')):
                for layer in rec.get('layers', []):
                    if key not in layer or not layer.get(name_key):
                        continue
                    events.append(
                        {
                            'ph': 'X',
                            'name': (
                                layer['n'].replace('l', 'L', 1)
                                if layer['n'].startswith('l')
                                else layer['n']
                            ),
                            'pid': rank,
                            'tid': _PERFETTO_TIDS[lane],
                            'ts': offset + layer[key],
                            'dur': layer[name_key],
                            'args': {'step': step, 'layer': layer['n'], 'lane': lane},
                        }
                    )
            for a in rec.get('a2a', []):
                events.append(
                    {
                        'ph': 'X',
                        'name': f"{a['n']} {a.get('layer', '')}".strip(),
                        'pid': rank,
                        'tid': _PERFETTO_TIDS['comm'],
                        'ts': offset + a['s'],
                        'dur': a['d'],
                        'args': {'step': step, 'layer': a.get('layer', ''), 'op': a['n']},
                    }
                )
            for sp in rec.get('spans', []):
                events.append(
                    {
                        'ph': 'X',
                        'name': sp['n'],
                        'pid': rank,
                        'tid': _PERFETTO_TIDS['other'],
                        'ts': offset + sp['s'],
                        'dur': sp['d'],
                        'args': {'step': step},
                    }
                )
            # one step-boundary slice per rank keeps the tracks aligned
            events.append(
                {
                    'ph': 'X',
                    'name': f'step {step}',
                    'pid': rank,
                    'tid': _PERFETTO_TIDS['step'],
                    'ts': offset,
                    'dur': step_len or 1,
                    'args': {'step': step},
                }
            )
        offset += (step_len or 1) + 1000.0  # 1 ms visual gap between steps

    for rank in sorted({e['pid'] for e in events}):
        events.append(
            {'ph': 'M', 'name': 'process_name', 'pid': rank, 'args': {'name': f'rank {rank}'}}
        )
    first_pid = min((e['pid'] for e in events if 'pid' in e), default=0)
    for tid, name in _PERFETTO_TID_NAMES.items():
        events.append(
            {'ph': 'M', 'name': 'thread_name', 'pid': first_pid, 'tid': tid, 'args': {'name': name}}
        )
        for rank in sorted({e['pid'] for e in events}):
            events.append(
                {'ph': 'M', 'name': 'thread_name', 'pid': rank, 'tid': tid, 'args': {'name': name}}
            )
    return {
        'traceEvents': events,
        'displayTimeUnit': 'ms',
        'metadata': {'source': 'megatron perf_trace', 'steps': list(steps)},
    }


def comm_overlap_stats(store: TraceStore, step: int) -> Dict[int, Dict[str, float]]:
    """Per rank: how much of the all-to-all time actually overlapped compute.

    This is the number behind "is comm overlapped": 100% means every comm second was hidden
    behind compute, 0% means the rank sat idle waiting for the network.
    """
    out: Dict[int, Dict[str, float]] = {}
    for rank in store.ranks(step):
        rec = store.get(step, rank) or {}
        compute = []
        for layer in rec.get('layers', []):
            if 'fs' in layer and layer.get('fwd'):
                compute.append((layer['fs'], layer['fs'] + layer['fwd']))
            if 'bs' in layer and layer.get('bwd'):
                compute.append((layer['bs'], layer['bs'] + layer['bwd']))
        comm = [(a['s'], a['s'] + a['d']) for a in rec.get('a2a', [])]
        comm_total = _union_us(comm)
        overlap = _intersection_us(compute, comm)
        out[rank] = {
            'comm_s': comm_total / 1e6,
            'overlap_s': overlap / 1e6,
            'overlap_pct': (100.0 * overlap / comm_total) if comm_total else 0.0,
        }
    return out


def _union_us(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    a, b = ivs[0]
    for x, y in ivs[1:]:
        if x > b:
            total += b - a
            a, b = x, y
        else:
            b = max(b, y)
    return total + (b - a)


def _intersection_us(a_ivs, b_ivs):
    if not a_ivs or not b_ivs:
        return 0.0
    a = sorted(a_ivs)
    b = sorted(b_ivs)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def draw_pipeline_gantt(
    store: TraceStore,
    step: int,
    layer_range: Tuple[int, int] = None,
    max_ranks: int = None,
    rank: int = None,
) -> bytes:
    """One row per rank, every layer drawn at its real start and end.

    Three lanes per rank:

    * **forward** (top): each layer a block, coloured by layer index so the sequence is readable
      and a repeated layer (recompute) is obvious;
    * **backward** (middle): same colouring, running in reverse order;
    * **comm** (bottom, thin): the all-to-all where it actually ran, so an orange block inside a
      compute block means overlapped and an orange block alone in a gap means exposed.

    Gaps between the end of a block and the start of the next are drawn as the background, which
    is exactly the idle time this chart exists to show.
    """
    import matplotlib.patches as mpatches
    from matplotlib import colormaps

    plt = _plt()
    ranks = [rank] if rank is not None else store.ranks(step)
    if max_ranks:
        ranks = ranks[:max_ranks]
    if not ranks:
        raise ValueError(f'no records for step {step}')

    stats = comm_overlap_stats(store, step)
    row_h = 1.0
    fig_h = max(2.4, row_h * len(ranks) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    cmap = colormaps['viridis']

    # layer index -> colour; layer names look like "l12" or "l12#recompute"
    all_names = []
    for rk in ranks:
        for layer in (store.get(step, rk) or {}).get('layers', []):
            if layer['n'] not in all_names:
                all_names.append(layer['n'])
    order = {n: i for i, n in enumerate(all_names)}
    n_layers = max(1, len(all_names))
    lo, hi = layer_range if layer_range else (0, n_layers - 1)

    span_hint = max(
        1.0,
        max(
            ((store.get(step, rk) or {}).get('phases', {}).get('__end__', 0) / 1e6 for rk in ranks),
            default=1.0,
        ),
    )
    for row, rk in enumerate(ranks):
        rec = store.get(step, rk) or {}
        y0 = (len(ranks) - 1 - row) * row_h  # rank 0 at the top
        for layer in rec.get('layers', []):
            idx = order.get(layer['n'], 0)
            if not (lo <= idx <= hi):
                continue
            colour = cmap(idx / max(1, n_layers - 1))
            for lane_key, start_key, y_off in (('fwd', 'fs', 0.62), ('bwd', 'bs', 0.32)):
                if start_key not in layer or not layer.get(lane_key):
                    continue
                w = max(layer[lane_key] / 1e6, 1e-5)
                ax.add_patch(
                    mpatches.Rectangle(
                        (layer[start_key] / 1e6, y0 + y_off),
                        w,
                        0.26,
                        facecolor=colour,
                        edgecolor='white',
                        linewidth=0.2,
                    )
                )
                # label the layer index whenever the block is wide enough to hold it
                if w > (span_hint * 0.018):
                    ax.text(
                        (layer[start_key] / 1e6) + w / 2,
                        y0 + y_off + 0.13,
                        layer['n'],
                        ha='center',
                        va='center',
                        fontsize=5.5,
                        color='white',
                        clip_on=True,
                    )
        for a in rec.get('a2a', []):
            ax.add_patch(
                mpatches.Rectangle(
                    (a['s'] / 1e6, y0 + 0.06),
                    max(a['d'] / 1e6, 1e-5),
                    0.18,
                    facecolor=_DEFAULT_COLORS['a2a'],
                    edgecolor='none',
                )
            )
        for sp in rec.get('spans', []):
            ax.add_patch(
                mpatches.Rectangle(
                    (sp['s'] / 1e6, y0 + 0.06),
                    max(sp['d'] / 1e6, 1e-5),
                    0.18,
                    facecolor=_DEFAULT_COLORS['span'],
                    edgecolor='none',
                    alpha=0.9,
                )
            )
        ov = stats.get(rk, {})
        ax.text(
            -0.004 * max(1e-6, _row_end(rec)),
            y0 + 0.5,
            f"r{rk}",
            ha='right',
            va='center',
            fontsize=7,
        )
        if ov:
            ax.text(
                _row_end(rec) * 1.0,
                y0 + 0.5,
                f" a2a {ov['comm_s'] * 1e3:.1f}ms overlapped {ov['overlap_pct']:.0f}%",
                ha='left',
                va='center',
                fontsize=6,
                color='#666',
            )

    ax.set_xlabel('time since step anchor (s)')
    ax.set_ylabel('rank')
    ax.set_yticks([(len(ranks) - 1 - i) * row_h + 0.5 for i in range(len(ranks))])
    ax.set_yticklabels([f'r{rk}' for rk in ranks], fontsize=7)
    ax.set_ylim(-0.1, len(ranks) * row_h + 0.1)
    ax.set_title(
        f'step {step}: per-rank pipeline, every layer at its real start/end  '
        f'(top lane=fwd, middle=bwd, colour=layer index; bottom=comm purple=optimizer)'
    )
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def _row_end(rec) -> float:
    end = rec.get('phases', {}).get('__end__', 0) / 1e6
    if end:
        return end
    last = 0.0
    for layer in rec.get('layers', []):
        last = max(last, (layer.get('fs', 0) + layer.get('fwd', 0)) / 1e6)
        last = max(last, (layer.get('bs', 0) + layer.get('bwd', 0)) / 1e6)
    return last or 1.0


def draw_layer_distribution(
    store: TraceStore, steps: List[int], rank: int = 0, metric: str = 'fwd'
) -> bytes:
    """One rank's per-layer time, min/median/max over the chosen steps."""
    plt = _plt()
    per_layer: Dict[str, List[float]] = defaultdict(list)
    order: List[str] = []
    for step in steps:
        rec = store.get(step, rank)
        if not rec:
            continue
        for layer in rec.get('layers', []):
            name = layer.get('n')
            if name is None or metric not in layer:
                continue
            if name not in per_layer:
                order.append(name)
            per_layer[name].append(layer[metric] / 1e6)
    if not order:
        raise ValueError(f'no per-layer {metric} records for rank {rank}')
    med, lo, hi = [], [], []
    for name in order:
        vals = sorted(per_layer[name])
        med.append(vals[len(vals) // 2])
        lo.append(vals[0])
        hi.append(vals[-1])
    fig, ax = plt.subplots(figsize=(max(10, 0.2 * len(order) + 4), 4))
    xs = range(len(order))
    ax.plot(xs, med, color='#4C78A8', label=f'{metric} median')
    ax.fill_between(xs, lo, hi, color='#4C78A8', alpha=0.2, label='min..max over steps')
    ax.set_xticks(list(xs)[:: max(1, len(order) // 40)])
    ax.set_xticklabels(order[:: max(1, len(order) // 40)], rotation=90, fontsize=6)
    ax.set_ylabel('seconds')
    ax.set_title(f'rank {rank}: per-layer {metric} over {len(steps)} step(s)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _png(fig)


def _png(fig) -> bytes:
    """Render a figure to PNG bytes and close it (a live dashboard opens many figures)."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140)
    _plt().close(fig)
    buf.seek(0)
    return buf.read()
