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
"""Draw a per-rank training-pipeline timeline from ``--perf-trace-dir`` output.

One row per rank, x axis in seconds relative to a chosen alignment anchor, layers as thin
blocks and the MoE all-to-all drawn on a second lane inside each row. That is what makes
"rank 37 started its MoE A2A 12 s after everyone else" visible at a glance.

Usage:
    plot_perf_trace.py --dir <perf-trace-dir> --step 19 --out pipeline.png
    plot_perf_trace.py --dir <dir> --step 19 --layers --out layers_rank37.png --rank 37

Alignment: every rank records the same collectives inside a step. By default each rank's
timeline is offset by its own ``phase.__anchor__`` (start of step) so the plot shows how
synchronised the step starts are; pass ``--align step_wall`` to instead right-align on the end
of the step, which makes backward/optimizer tails line up.
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load(dirpath: str, step: int) -> Dict[int, dict]:
    """Read every ``perf-rank-*.jsonl`` and return ``{rank: record}`` for one step."""
    out: Dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(dirpath, 'perf-rank-*.jsonl'))):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('step') == step:
                    out[rec['rank']] = rec
                    break
    return out


def _span(rec: dict) -> float:
    """A rank's timeline length in seconds (end of the last layer, else the step wall)."""
    end = 0.0
    for layer in rec.get('layers', []):
        for key in ('fwd', 'bwd'):
            if key in layer:
                end = max(end, 0.0)
    end = rec.get('phases', {}).get('__end__', 0.0)
    if not end and rec.get('step_wall'):
        end = rec['step_wall'] / 1e3
    for a in rec.get('a2a', []):
        end = max(end, (a['s'] + a['d']) / 1e3)
    return end


def plot_pipeline(records: Dict[int, dict], out_path: str, align: str = 'step_start'):
    """One row per rank: layer blocks on the compute lane, A2A blocks on a comm lane."""
    ranks = sorted(records)
    if not ranks:
        raise SystemExit('no records for that step')
    fig_h = max(2.0, 0.28 * len(ranks) + 1.6)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    for row, rank in enumerate(ranks):
        rec = records[rank]
        layers = rec.get('layers', [])
        phases = rec.get('phases', {})
        fwd_end = phases.get('__end__', 0.0)
        # Reconstruct absolute layer windows from cumulative durations in file order.
        t = phases.get('__anchor__', 0.0)
        fwd_blocks, bwd_blocks = [], []
        cursor = 0.0
        for layer in layers:
            if 'fwd' in layer:
                d = layer['fwd'] / 1e3
                fwd_blocks.append((cursor, d))
                cursor += d
            elif 'bwd' in layer:
                d = layer['bwd'] / 1e3
                bwd_blocks.append((cursor, d))
                cursor += d
        if align == 'step_end':
            shift = max(0.0, fwd_end - cursor)
        else:
            shift = 0.0
        y = row
        for start, dur in fwd_blocks:
            ax.add_patch(
                mpatches.Rectangle(
                    (start + shift, y - 0.34), dur, 0.30, facecolor='#4C78A8', edgecolor='none'
                )
            )
        for start, dur in bwd_blocks:
            ax.add_patch(
                mpatches.Rectangle(
                    (start + shift, y - 0.34), dur, 0.30, facecolor='#54A24B', edgecolor='none'
                )
            )
        for a in rec.get('a2a', []):
            ax.add_patch(
                mpatches.Rectangle(
                    (a['s'] / 1e3 + shift, y + 0.02),
                    a['d'] / 1e3,
                    0.28,
                    facecolor='#F58518',
                    edgecolor='none',
                )
            )
        data = rec.get('data')
        if data and data.get('total'):
            ax.add_patch(
                mpatches.Rectangle(
                    (0, y - 0.46), data['total'] / 1e3, 0.10, facecolor='#BAB0AC', edgecolor='none'
                )
            )
        ax.text(-0.01, y, f'{rank}', ha='right', va='center', fontsize=7)

    ax.set_yticks([])
    ax.set_ylabel('rank (one row each)')
    ax.set_xlabel('time since the start-of-step anchor (s)')
    ax.set_title(
        'training pipeline per rank  (blue=fwd layers, green=bwd layers, '
        'orange=MoE A2A, grey=data fetch)'
    )
    ax.set_ylim(-1.0, len(ranks))
    ax.grid(axis='x', alpha=0.25)

    # Mark the rank whose compute ends last: that is the straggler of this step.
    ends = []
    for rank in ranks:
        rec = records[rank]
        ph = rec.get('phases', {})
        ends.append((ph.get('__end__', 0.0), rank))
    if ends:
        end_s, end_rank = max(ends)
        ax.axvline(end_s, color='#E45756', linestyle='--', linewidth=1)
        ax.text(
            end_s,
            len(ranks) + 0.2,
            f' last: rank {end_rank} @ {end_s:.2f}s',
            color='#E45756',
            fontsize=8,
            va='bottom',
            ha='right',
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f'wrote {out_path}  ({len(ranks)} ranks, align={align})')


def plot_layers(records: Dict[int, dict], out_path: str, rank: int):
    """Per-layer durations for one rank: fwd and bwd side by side."""
    rec = records[rank]
    layers = [l for l in rec.get('layers', []) if 'fwd' in l or 'bwd' in l]
    if not layers:
        raise SystemExit(f'rank {rank} has no per-layer records in that step')
    names = [l['n'] for l in layers]
    fwd = [l.get('fwd', 0) / 1e3 for l in layers]
    bwd = [l.get('bwd', 0) / 1e3 for l in layers]

    fig, ax = plt.subplots(figsize=(max(10, 0.22 * len(layers) + 4), 4))
    idx = range(len(layers))
    ax.bar([i - 0.2 for i in idx], fwd, width=0.4, label='forward', color='#4C78A8')
    ax.bar([i + 0.2 for i in idx], bwd, width=0.4, label='backward', color='#54A24B')
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel('layer time (s)')
    ax.set_title(f'rank {rank} step {rec["step"]}: per-layer time')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f'wrote {out_path}  ({len(layers)} layers)')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--dir', required=True, help='--perf-trace-dir used by the run')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--out', default='perf_pipeline.png')
    ap.add_argument('--align', choices=['step_start', 'step_end'], default='step_start')
    ap.add_argument('--layers', action='store_true', help='plot per-layer bars instead')
    ap.add_argument('--rank', type=int, default=None, help='rank for --layers')
    args = ap.parse_args()

    records = load(args.dir, args.step)
    if not records:
        raise SystemExit(f'no perf-rank-*.jsonl with step {args.step} under {args.dir}')
    if args.layers:
        rank = args.rank if args.rank is not None else sorted(records)[0]
        plot_layers(records, args.out, rank)
    else:
        plot_pipeline(records, args.out, args.align)


if __name__ == '__main__':
    main()
