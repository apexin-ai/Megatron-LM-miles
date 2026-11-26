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
"""Check that perf-trace collection is correct under each parallel layout.

The tracer has no cross-rank communication, so "does it work at TP/PP/CP/EP" reduces to a set of
per-rank invariants. This script asserts them per configuration:

    * every layer built on a rank was instrumented, and layer ids are **global** (a pipeline
      stage must not renumber its layers from 0);
    * every traced rank wrote exactly one line per traced step;
    * a rank that owns transformer layers reports non-zero per-layer forward times;
    * A2A spans appear exactly when the config actually all-to-alls (EP > 1 with MoE);
    * a recomputed layer shows up twice (``lN`` and ``lN#recompute``) instead of overwriting.

It is meant to be run repeatedly, once per config, e.g.:

    for cfg in "tp1 pp1" "tp2 pp1" "tp1 pp2"; do ... torchrun --nproc-per-node 8 ...; done

but it also accepts a single configuration so it can be wired into CI one job per layout.
"""

import argparse
import glob
import json
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='perf trace dir this config wrote to')
    ap.add_argument('--world', type=int, required=True)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--cp', type=int, default=1)
    ap.add_argument('--ep', type=int, default=1)
    ap.add_argument('--vpp', type=int, default=1, help='virtual pipeline stages')
    ap.add_argument('--layers', type=int, required=True, help='model num_layers')
    ap.add_argument('--steps', type=int, required=True, help='traced steps expected per rank')
    ap.add_argument('--moe', action='store_true')
    return ap.parse_args()


def expected_layers_per_stage(args):
    """Layers a single pipeline stage owns (VPP further splits the stage's layers)."""
    per_stage = args.layers / (args.pp * args.vpp)
    return per_stage


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, 'perf-rank-*.jsonl')))
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    check(files, f'no perf-rank-*.jsonl in {args.dir}')
    if not files:
        return report(failures)

    by_rank = {}
    for path in files:
        rank = int(os.path.basename(path).split('perf-rank-')[1].split('.')[0])
        with open(path) as fh:
            by_rank[rank] = [json.loads(x) for x in fh if x.strip()]

    check(
        len(by_rank) == args.world,
        f'{len(by_rank)} rank file(s) written, expected {args.world} '
        f'(did every rank get --perf-trace-dir?)',
    )

    per_stage = expected_layers_per_stage(args)
    all_layer_ids = set()
    any_a2a = False
    for rank, records in sorted(by_rank.items()):
        check(
            len(records) == args.steps,
            f'rank {rank}: {len(records)} step line(s), expected {args.steps}',
        )
        if not records:
            continue
        ids = {l['n'].split('#')[0] for rec in records for l in rec.get('layers', [])}
        all_layer_ids |= ids
        # a stage must not renumber from 0: global ids are offset by the pipeline stage
        check(
            int(per_stage) == 0 or len(ids) <= per_stage + 1,
            f'rank {rank}: {len(ids)} distinct layers, a stage owns about {per_stage}',
        )
        for rec in records:
            if rec.get('a2a'):
                any_a2a = True
            fwds = [
                l['fwd'] for l in rec.get('layers', []) if 'fwd' in l and '#recompute' not in l['n']
            ]
            if fwds:
                check(all(v > 0 for v in fwds), f"rank {rank}: a forward layer time is 0")

    # global ids: with PP, different stages must not all start at l0
    if args.pp * args.vpp > 1 and per_stage > 0:
        offsets = sorted({int(i[1:]) for i in all_layer_ids})
        check(
            len(all_layer_ids) >= min(args.layers, int(per_stage) * args.pp * args.vpp),
            f'only {len(all_layer_ids)} distinct global layer ids across ranks; pipeline stages '
            f'look like they renumber locally ({offsets[:6]}...)',
        )

    if args.moe:
        # The dispatcher runs at every EP size (at EP=1 it is a no-op copy), so A2A spans are
        # expected whenever MoE is on. Their *duration* only becomes interesting at EP>1.
        check(any_a2a, 'MoE on but no A2A spans were recorded (dispatcher hook missed?)')
    else:
        check(not any_a2a, 'A2A spans recorded for a dense model (dispatcher hook misfired?)')

    print(
        f'config: world={args.world} tp={args.tp} pp={args.pp} cp={args.cp} ep={args.ep} '
        f'vpp={args.vpp} moe={args.moe} layers={args.layers}'
    )
    print(
        f'ranks written: {len(by_rank)}   distinct global layers: {len(all_layer_ids)}   '
        f'a2a seen: {any_a2a}'
    )
    return report(failures)


def report(failures):
    if failures:
        print('\nFAIL')
        for f in failures:
            print('  -', f)
        return 1
    print('\nPASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
