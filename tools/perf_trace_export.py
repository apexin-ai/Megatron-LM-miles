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
"""Export perf-trace records as a Chrome Trace Event file for Perfetto.

Perfetto (Apache-2.0, https://ui.perfetto.dev) opens this format directly, which gives real
zoom/pan/hover, per-slice durations, search and SQL without maintaining a viewer.

    perf_trace_export.py --dir <perf-trace-dir> --step 19 --out step19.json
    perf_trace_export.py --dir <perf-trace-dir> --all --out window.json
    perf_trace_export.py --dir <perf-trace-dir> --all --gzip --out window.json.gz

Then open https://ui.perfetto.dev and drop the file in, or run Perfetto's local trace_processor.
"""

import argparse
import gzip
import json
import os
import sys


def _load_report_module():
    """Load report.py by path so exporting needs neither torch nor an installed megatron."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'megatron', 'core', 'perf_trace', 'report.py')
    spec = importlib.util.spec_from_file_location('perf_trace_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_report_module()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--dir', required=True, help='the --perf-trace-dir the run wrote to')
    ap.add_argument('--step', type=int, default=None, help='single step (default: latest)')
    ap.add_argument('--all', action='store_true', help='every recorded step, laid end to end')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gzip', action='store_true', help='gzip the output (Perfetto accepts it)')
    ap.add_argument('--max-steps', type=int, default=4096)
    args = ap.parse_args()

    store = report.TraceStore(args.dir, max_steps=args.max_steps)
    store.refresh()
    steps = store.steps()
    if not steps:
        sys.exit(f'no perf records in {args.dir}')

    if args.all:
        chosen = steps
    else:
        chosen = [args.step if args.step is not None else steps[-1]]
        if chosen[0] not in steps:
            sys.exit(f'step {chosen[0]} not found; recorded steps: {steps[:5]}..{steps[-5:]}')

    doc = report.to_perfetto(store, chosen)
    payload = json.dumps(doc, separators=(',', ':')).encode()
    opener = gzip.open if args.gzip else open
    with opener(args.out, 'wb') as fh:
        fh.write(payload)

    slices = sum(1 for e in doc['traceEvents'] if e['ph'] == 'X')
    ranks = sorted({e['pid'] for e in doc['traceEvents'] if e['ph'] == 'X'})
    size = os.path.getsize(args.out)
    print(f'wrote {args.out}  ({size / 1e6:.2f} MB)')
    print(f'  steps  : {chosen[0]}..{chosen[-1]} ({len(chosen)})')
    print(f'  ranks  : {len(ranks)} ({ranks[0]}..{ranks[-1]})')
    print(f'  slices : {slices}')
    print('  open   : https://ui.perfetto.dev  ->  "Open trace file"  ->  pick the file')
    if size > 300e6:
        print('  note   : this is a large file; export fewer steps or more ranks selectively')


if __name__ == '__main__':
    main()
