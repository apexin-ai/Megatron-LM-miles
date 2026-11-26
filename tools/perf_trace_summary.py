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
"""Text report from ``--perf-trace-dir`` output: no matplotlib, works over ssh.

The dashboard draws pictures; this prints the numbers behind them, which is what you want on a
login node or in a CI log. Three sections:

    steps     per step: window, median/min/max across ranks, and the imbalance ratio
    worst     for the chosen step (default: the worst one) the slowest ranks with their parts
    layers    for that step, the layers that cost the most, averaged over ranks

An "imbalance ratio" of ``max/median`` around 1.0 means the ranks are level; 1.3 means the slowest
rank spends 30% more time than the median rank, which is time every other rank waits for in the
next collective.

Usage:
    perf_trace_summary.py --dir <perf-trace-dir> [--step N] [--top 8]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _load_report_module():
    """Load report.py by path: this tool must not require torch just to read JSONL."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'megatron', 'core', 'perf_trace', 'report.py')
    spec = importlib.util.spec_from_file_location('perf_trace_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_report_module()


def fmt_ratio(ratio: float) -> str:
    """Human label for an imbalance ratio."""
    if ratio < 1.05:
        return 'level'
    if ratio < 1.20:
        return 'slight'
    if ratio < 1.50:
        return 'notable'
    return 'SEVERE'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--dir', required=True)
    ap.add_argument('--step', type=int, default=None, help='detail step (default: the worst)')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=4096)
    args = ap.parse_args()

    store = report.TraceStore(args.dir, max_steps=args.max_steps)
    added = store.refresh()
    if not store.steps():
        sys.exit(f'no perf records found in {args.dir} (loaded {added})')

    rows = store.summary()
    print(f'dir      : {args.dir}')
    print(f'steps    : {len(rows)}  ({store.steps()[0]} .. {store.steps()[-1]})')
    print(f'ranks    : {max(r["ranks"] for r in rows)}')
    print()

    # ---------------------------------------------------------------- per-step table
    print(
        f"{'step':>9} {'ranks':>5} {'median s':>9} {'min s':>9} {'max s':>9} "
        f"{'max/med':>8} {'fwd s':>8} {'bwd s':>8} {'a2a s':>8} {'data s':>8}  verdict"
    )
    worst_step, worst_ratio = None, 0.0
    for r in rows:
        med = r['total']['median']
        ratio = (r['total']['max'] / med) if med else 0.0
        if ratio > worst_ratio:
            worst_ratio, worst_step = ratio, r['step']
        print(
            f"{r['step']:>9} {r['ranks']:>5} {med:9.3f} {r['total']['min']:9.3f} "
            f"{r['total']['max']:9.3f} {ratio:8.2f} {r['fwd']['median']:8.3f} "
            f"{r['bwd']['median']:8.3f} {r['a2a']['median']:8.3f} {r['data']['median']:8.3f}  "
            f"{fmt_ratio(ratio)}"
        )

    step = args.step if args.step is not None else worst_step
    if step is None:
        return

    # ------------------------------------------------------------- slowest ranks
    print()
    print(f"=== step {step}: slowest ranks ===")
    print(f"{'rank':>6} {'total s':>9} {'fwd s':>8} {'bwd s':>8} {'a2a s':>8}")
    for rank, total, fwd, bwd, a2a in store.rank_totals(step)[: args.top]:
        print(f'{rank:>6} {total:9.3f} {fwd:8.3f} {bwd:8.3f} {a2a:8.3f}')

    # --------------------------------------------------------------- hot layers
    names, ranks, matrix = store.layer_matrix(step, 'fwd')
    if names and ranks:
        import statistics

        per_layer = []
        for j, name in enumerate(names):
            col = [row[j] for row in matrix if row[j] == row[j]]  # drop nan
            if col:
                per_layer.append((name, statistics.median(col), max(col)))
        per_layer.sort(key=lambda x: -x[1])
        print()
        print(f"=== step {step}: most expensive layers (forward, median over ranks) ===")
        print(f"{'layer':>10} {'median s':>9} {'max s':>9}")
        for name, med, mx in per_layer[: args.top]:
            print(f'{name:>10} {med:9.4f} {mx:9.4f}')

        # a hot column is a fixed layer cost, a hot row is a straggler: report both
        totals_by_rank = [
            sum(row[j] for j in range(len(names)) if row[j] == row[j]) for row in matrix
        ]
        if totals_by_rank:
            med_rank = statistics.median(totals_by_rank)
            worst_rank_idx = max(range(len(ranks)), key=lambda i: totals_by_rank[i])
            print()
            print(
                f'  heaviest layer column: {per_layer[0][0]} '
                f'({per_layer[0][1]:.4f} s vs median layer '
                f'{statistics.median([x[1] for x in per_layer]):.4f} s)'
            )
            print(
                f'  heaviest rank row    : {ranks[worst_rank_idx]} '
                f'({totals_by_rank[worst_rank_idx]:.3f} s vs median rank {med_rank:.3f} s)'
            )

    cpu = [r for r in rows if r['step'] == step]
    if cpu and cpu[0].get('cpu'):
        print()
        print(f'=== step {step}: phase timers (median across ranks) ===')
        for key in sorted(cpu[0]['cpu'], key=lambda k: -cpu[0]['cpu'][k]['median']):
            print(f"  {key:34s} {cpu[0]['cpu'][key]['median']:8.3f} s")


if __name__ == '__main__':
    main()
