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
"""Kernel-union analysis of PyTorch profiler traces, plus multi-rank skew.

Three things this tool answers that a raw trace does not:

1. **Where the step goes.** Kernels are grouped into pools (QSA, MoE GEMM, collectives,
   elementwise, ...) and the pool cost is the *interval union* of its kernels, never the sum of
   durations: overlapping kernels must not be counted twice.
2. **What the collectives actually are.** NCCL kernels carry ``In/Out msg nelems`` in their
   args, so a reduce-scatter's implied group size and effective bandwidth fall out of the
   trace; that is how a 49 s collective was identified as pathological rather than bandwidth
   bound.
3. **Whether ranks are skewed.** For a directory of traces it reports each rank's step window
   and its collectives, so a straggler and a synchronising collective can be told apart: a
   straggler makes the others wait (durations differ), a slow collective makes everyone slow
   (durations match).

Traces are read with ``ijson`` so a 3 GB json.gz streams through in bounded memory.

Usage:
    analyze_profile.py single  --trace <file.json.gz> [--step largest|N] [--json out.json]
    analyze_profile.py skew    --dir <trace-dir> [--glob '*.trace.json.gz'] [--json out.json]
    analyze_profile.py compare --a <file> --b <file> [--step largest]
"""

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import ijson
except ImportError:  # pragma: no cover
    sys.exit('this tool needs ijson: pip install ijson')


def _num(x):
    return float(x) if isinstance(x, Decimal) else x


# Ordered: the first matching predicate wins, so put the specific ones first.
CATEGORIES: List[Tuple[str, Tuple[str, ...]]] = [
    ('qsa_dkdv', ('_dkdv_kernel',)),
    ('qsa_dq', ('_dq_kernel',)),
    ('qsa_fwd', ('_fwd_kernel', 'score_blocks', '_qsa_')),
    ('gdn_fla', ('conv1d', 'chunk_', 'fwd_h', 'fwd_o', 'fwd_dqk', 'fwd_dv', '_fla', 'gated_delta')),
    ('comm_a2a', ('SendRecv', 'AlltoAll', 'alltoall', 'all_to_all')),
    ('comm', ('nccl', 'AllGather', 'ReduceScatter', 'AllReduce', 'Broadcast')),
    ('gemm', ('nvjet', 'cutlass', 'xmma', 'gemm', 'hgmma', 'sm90_', 'sm100_', 'sm103_')),
    ('moe_route', ('permute', 'unpermute', 'topk', 'sort', 'moe', 'router', 'scatter', 'gather')),
    ('norm', ('rmsnorm', 'layer_norm', 'layernorm')),
    ('elementwise', ('elementwise', 'vectorized', 'triton_poi', 'silu', 'swiglu', 'copy', 'cast')),
]


def categorise(name: str) -> str:
    """Return the pool name for a kernel name."""
    low = name.lower()
    for pool, needles in CATEGORIES:
        for n in needles:
            if n in name or n.lower() in low:
                return pool
    return 'other'


class Step:
    """One profiled step: its window plus the kernels and collectives inside it."""

    def __init__(self, name: str, start: float, end: float):
        self.name = name
        self.start = start
        self.end = end
        self.kernels: List[Tuple[float, float, str]] = []
        self.collectives: List[Dict] = []

    @property
    def window_s(self) -> float:
        return (self.end - self.start) / 1e6

    def union_by_pool(self) -> Dict[str, Tuple[float, int]]:
        """Interval union and call count per pool (seconds, count)."""
        buckets: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for a, b, name in self.kernels:
            buckets[categorise(name)].append((max(a, self.start), min(b, self.end)))
        out = {}
        for pool, ivs in buckets.items():
            out[pool] = (_union_seconds(ivs), len(ivs))
        return out

    def top_kernels(self, n: int = 15) -> List[Tuple[str, float, int]]:
        """Top kernels by union: (name, seconds, calls)."""
        per: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for a, b, name in self.kernels:
            per[name].append((max(a, self.start), min(b, self.end)))
        ranked = sorted(per.items(), key=lambda kv: -_union_seconds(kv[1]))[:n]
        return [(name, _union_seconds(ivs), len(ivs)) for name, ivs in ranked]


def _union_seconds(intervals: List[Tuple[float, float]]) -> float:
    """Length of the union of half-open intervals, in seconds."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += cur_b - cur_a
    return total / 1e6


def read_trace(path: str, want_step: str = 'largest') -> Optional[Step]:
    """Stream one trace and return the requested profiled step (or the largest)."""
    ann, kernels, colls = [], [], []
    with gzip.open(path, 'rb') as fh:
        for ev in ijson.items(fh, 'traceEvents.item'):
            if ev.get('ph') != 'X':
                continue
            cat = ev.get('cat')
            if cat == 'user_annotation' and 'ProfilerStep#' in str(ev.get('name', '')):
                ann.append(
                    (_num(ev['ts']), _num(ev['ts']) + _num(ev.get('dur', 0)), ev.get('name'))
                )
            elif cat == 'kernel':
                a = _num(ev.get('ts', 0))
                b = a + _num(ev.get('dur') or 0)
                kernels.append((a, b, str(ev.get('name', ''))))
                args = ev.get('args') or {}
                if 'msg nelems' in str(args) or 'Collective name' in str(args):
                    colls.append(
                        {
                            'name': str(ev.get('name', '')),
                            'dur': _num(ev.get('dur') or 0),
                            'ts': a,
                            'in': args.get('In msg nelems'),
                            'out': args.get('Out msg nelems'),
                            'coll': args.get('Collective name'),
                        }
                    )
    if not ann:
        return None
    if want_step == 'largest':
        s0, t0, name = max(ann, key=lambda x: x[1] - x[0])
    else:
        target = int(want_step)
        match = [x for x in ann if str(x[2]).endswith(f'#{target}')]
        if not match:
            return None
        s0, t0, name = match[0]
    step = Step(name, s0, t0)
    step.kernels = [k for k in kernels if k[1] > s0 and k[0] < t0]
    step.collectives = [c for c in colls if c['ts'] + c['dur'] > s0 and c['ts'] < t0]
    return step


def report_single(path: str, step: Step, as_json: Optional[str] = None) -> dict:
    """Print (and optionally dump) one step's pool table, top kernels and collectives."""
    w = step.window_s
    print(f'trace      : {os.path.basename(path)}')
    print(f'step       : {step.name}')
    print(f'window     : {w:.3f} s')
    print(f'kernels    : {len(step.kernels)}')
    print()
    pools = step.union_by_pool()
    print(f"{'pool':12s} {'union s':>9} {'%step':>7} {'calls':>8}")
    for pool, (secs, n) in sorted(pools.items(), key=lambda kv: -kv[1][0]):
        print(f'{pool:12s} {secs:9.3f} {100 * secs / w:6.1f}% {n:8d}')
    tot = sum(v[0] for v in pools.values())
    print(f"{'TOTAL':12s} {tot:9.3f} {100 * tot / w:6.1f}%")
    print()
    print('top kernels by union:')
    top = step.top_kernels(15)
    for name, secs, n in top:
        print(f'  {secs:8.3f} s {100 * secs / w:6.2f}%  n={n:6d}  {name[:70]}')
    if step.collectives:
        print()
        print('collectives with message sizes:')
        for c in sorted(step.collectives, key=lambda c: -c['dur'])[:8]:
            try:
                ins, outs = int(float(c['in'])), int(float(c['out']))
                grp = ins / outs if outs else 0
                gib, secs = ins * 4 / 2**30, c['dur'] / 1e6
                bw = gib * (grp - 1) / grp / secs if secs else 0
                print(
                    f"  {secs:8.3f} s  in={gib:6.2f} GiB  group={grp:4.0f}  {bw:7.2f} GiB/s  {c['coll']}"
                )
            except (TypeError, ValueError):
                print(f"  {c['dur'] / 1e6:8.3f} s  {c['name'][:60]}")
    data = {
        'trace': path,
        'step': step.name,
        'window_s': w,
        'kernels': len(step.kernels),
        'pools': {k: {'union_s': v[0], 'calls': v[1]} for k, v in pools.items()},
        'top_kernels': [{'name': n, 'union_s': s, 'calls': c} for n, s, c in top],
    }
    if as_json:
        with open(as_json, 'w') as fh:
            json.dump(data, fh, indent=2)
        print(f'\nwrote {as_json}')
    return data


def report_skew(dirpath: str, pattern: str, as_json: Optional[str] = None) -> dict:
    """Per-step, per-rank window and biggest collective.

    Grouping by step is essential: step durations differ by orders of magnitude in an RL run
    (a 5 s step and a 395 s step sit in the same directory), so comparing ranks across steps
    would report a huge "spread" that means nothing. Within one step the verdict is:

    * collective durations uniform  -> the collective itself is slow;
    * collective durations spread   -> some rank arrives late (a straggler).
    """
    files = sorted(glob.glob(os.path.join(dirpath, pattern)))
    if not files:
        sys.exit(f'no files matching {pattern} under {dirpath}')
    rows = []
    for f in files:
        try:
            step = read_trace(f, 'largest')
        except Exception as exc:  # a truncated trace must not kill the sweep
            print(f'  skip {os.path.basename(f)}: {exc}', file=sys.stderr)
            continue
        if step is None:
            continue
        rank = None
        base = os.path.basename(f)
        if 'rank_' in base:
            rank = int(base.split('rank_')[1].split('.')[0])
        biggest = max(step.collectives, key=lambda c: c['dur'], default=None)
        rows.append(
            {
                'file': base,
                'rank': rank,
                'step': step.name,
                'window_s': step.window_s,
                'biggest_collective_s': (biggest['dur'] / 1e6) if biggest else 0.0,
                'biggest_collective': (biggest['coll'] or biggest['name'][:40]) if biggest else '',
            }
        )
    if not rows:
        sys.exit('no readable steps')

    by_step: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_step[r['step']].append(r)

    verdicts = {}
    for step_name in sorted(by_step):
        group = sorted(by_step[step_name], key=lambda r: (r['rank'] is None, r['rank']))
        windows = [r['window_s'] for r in group]
        colls = [r['biggest_collective_s'] for r in group if r['biggest_collective_s']]
        print(f'\n=== {step_name} : {len(group)} rank(s) ===')
        print(f"{'rank':>6} {'window s':>10} {'top collective s':>17}  collective")
        for r in group:
            print(
                f"{str(r['rank']):>6} {r['window_s']:10.3f} "
                f"{r['biggest_collective_s']:17.3f}  {r['biggest_collective']}"
            )
        print(
            f'window spread    : {max(windows) - min(windows):.3f} s '
            f'({min(windows):.3f} .. {max(windows):.3f})'
        )
        if len(colls) < 2:
            verdicts[step_name] = 'single-rank sample'
            print('verdict          : single-rank sample, cannot judge skew')
            continue
        # The right question is not how much the ranks differ, but how much every rank paid no
        # matter when it arrived: that is the shortest duration. A collective everyone spends
        # 49 s in is intrinsically slow even if one rank waited 3 s longer.
        if len(colls) < 8:
            print(
                f'note             : only {len(colls)} rank(s) sampled; a straggler may be '
                'outside the sample, which also makes every sampled rank look "slow"'
            )
        lo, hi = min(colls), max(colls)
        floor_ratio = lo / hi if hi else 0.0
        if floor_ratio >= 0.9:
            verdicts[step_name] = 'all-sampled-wait'
            tail = (
                'all sampled ranks pay almost all of it -> either the collective is slow, or '
                'they are all waiting for a rank outside the sample (sample more ranks)'
            )
        elif floor_ratio >= 0.5:
            verdicts[step_name] = 'mixed'
            tail = 'about half intrinsic, half waiting -> check the arrival tail across all ranks'
        else:
            verdicts[step_name] = 'waiting'
            tail = 'the cost is mostly waiting -> a straggler inside the sample arrives late'
        print(
            f'collective       : min {lo:.3f} s / max {hi:.3f} s '
            f'(floor {100 * floor_ratio:.0f}%, spread {hi - lo:.3f} s)'
        )
        print(f'verdict          : {tail}')

    data = {'dir': dirpath, 'steps': {k: v for k, v in verdicts.items()}, 'rows': rows}
    if as_json:
        with open(as_json, 'w') as fh:
            json.dump(data, fh, indent=2)
        print(f'\nwrote {as_json}')
    return data


def report_compare(a: Step, b: Step, a_name: str, b_name: str) -> None:
    """Side-by-side pool table for two steps."""
    print(f'{"pool":12s} {("A: " + a_name)[-22:]:>22} {("B: " + b_name)[-22:]:>22} {"delta":>9}')
    pa, pb = a.union_by_pool(), b.union_by_pool()
    for pool in sorted(set(pa) | set(pb), key=lambda p: -(pa.get(p, (0, 0))[0])):
        sa, sb = pa.get(pool, (0.0, 0))[0], pb.get(pool, (0.0, 0))[0]
        d = f'{100 * (sb - sa) / sa:+.1f}%' if sa else 'n/a'
        print(
            f'{pool:12s} {sa:10.3f} s ({100*sa/a.window_s:5.1f}%) {sb:10.3f} s '
            f'({100*sb/b.window_s:5.1f}%) {d:>9}'
        )
    print(
        f'{"window":12s} {a.window_s:18.3f} s {b.window_s:18.3f} s '
        f'{100*(b.window_s-a.window_s)/a.window_s:+8.1f}%'
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    s1 = sub.add_parser('single', help='one trace')
    s1.add_argument('--trace', required=True)
    s1.add_argument('--step', default='largest')
    s1.add_argument('--json', default=None)

    s2 = sub.add_parser('skew', help='one directory, many ranks')
    s2.add_argument('--dir', required=True)
    s2.add_argument('--glob', default='*.trace.json.gz')
    s2.add_argument('--json', default=None)

    s3 = sub.add_parser('compare', help='two traces side by side')
    s3.add_argument('--a', required=True)
    s3.add_argument('--b', required=True)
    s3.add_argument('--step', default='largest')

    args = ap.parse_args()
    if args.cmd == 'single':
        step = read_trace(args.trace, args.step)
        if step is None:
            sys.exit('no ProfilerStep annotation found')
        report_single(args.trace, step, args.json)
    elif args.cmd == 'skew':
        report_skew(args.dir, args.glob, args.json)
    else:
        sa = read_trace(args.a, args.step)
        sb = read_trace(args.b, args.step)
        if sa is None or sb is None:
            sys.exit('could not read a step from one of the traces')
        report_compare(sa, sb, os.path.basename(args.a)[:22], os.path.basename(args.b)[:22])


if __name__ == '__main__':
    main()
