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
"""Analysis service for ``--perf-trace-dir`` output: watch a run, draw the training pictures.

Reads the JSONL the tracer writes (one line per rank per step) and serves:

    /                       HTML dashboard (auto-refresh, step selector)
    /api/summary            per-step aggregates across ranks (JSON)
    /api/step/<n>           per-rank totals for one step (JSON)
    /png/timeline?step=N    one row per rank: fwd/bwd layers + MoE A2A + data fetch
    /png/trends             step time and its parts over training
    /png/skew?step=N        per-rank total time, stragglers first
    /png/heatmap?step=N&metric=fwd|bwd    rank x layer heatmap
    /png/layers?rank=R&metric=fwd|bwd     per-layer distribution over the sampled steps

Deliberately dependency-light: stdlib ``http.server`` plus matplotlib. It only reads files, so
it can sit next to a running job and never touches the training process.

Usage:
    perf_trace_dashboard.py --dir /cpfs/.../perf --port 8080
    perf_trace_dashboard.py --dir <dir> --once --out report.html     # static report
"""

import argparse
import base64
import html
import io
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _load_report_module():
    """Load ``report.py`` by path so this tool needs neither torch nor an installed megatron.

    Importing ``megatron.core.perf_trace`` would run ``megatron/core/__init__.py``, which imports
    torch; a plotting service that only reads JSONL should not drag the whole training stack in.
    """
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'megatron', 'core', 'perf_trace', 'report.py')
    spec = importlib.util.spec_from_file_location('perf_trace_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_report = _load_report_module()
TraceStore = _report.TraceStore
draw_layer_distribution = _report.draw_layer_distribution
draw_layer_heatmap = _report.draw_layer_heatmap
draw_rank_skew = _report.draw_rank_skew
draw_step_timeline = _report.draw_step_timeline
draw_step_trends = _report.draw_step_trends

STORE: TraceStore = None  # set in main()
REFRESH_SECONDS = 5.0


def _png_response(png: bytes) -> bytes:
    return png


def _chart(store: TraceStore, kind: str, q: dict) -> bytes:
    """Dispatch one chart request to the drawing layer."""
    if kind == 'trends':
        return draw_step_trends(store)
    step = int(q.get('step', [store.latest_step() or 0])[0])
    if kind == 'timeline':
        return draw_step_timeline(store, step)
    if kind == 'skew':
        return draw_rank_skew(store, step)
    if kind == 'heatmap':
        metric = q.get('metric', ['fwd'])[0]
        return draw_layer_heatmap(store, step, metric)
    if kind == 'layers':
        rank = int(q.get('rank', ['0'])[0])
        metric = q.get('metric', ['fwd'])[0]
        return draw_layer_distribution(store, store.steps(), rank, metric)
    raise KeyError(kind)


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>training perf dashboard</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; }}
 h1 {{ font-size: 18px; }} h2 {{ font-size: 15px; margin-top: 24px; }}
 .meta {{ color: #555; font-size: 13px; }}
 img {{ max-width: 100%; border: 1px solid #ddd; background: #fff; }}
 form {{ margin: 12px 0; font-size: 14px; }}
 .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
</style></head><body>
<h1>training perf dashboard</h1>
<div class="meta">dir: {dirpath} &nbsp;|&nbsp; steps: {n_steps} &nbsp;|&nbsp; latest: {latest} &nbsp;|&nbsp;
refreshes every 5 s</div>
<form method="get" action="/">
  <label>step <input name="step" value="{latest}" size="8"></label>
  <label>rank <input name="rank" value="0" size="6"></label>
  <label>metric
    <select name="metric"><option value="fwd">fwd</option><option value="bwd">bwd</option></select>
  </label>
  <button type="submit">update</button>
</form>
<h2>step time over training</h2>
<img src="/png/trends?t={ts}" width="1100">
<h2>step {latest}: per-rank pipeline</h2>
<img src="/png/timeline?step={step}&t={ts}" width="1200">
<div class="row">
  <div><h2>per-rank skew</h2><img src="/png/skew?step={step}&t={ts}" width="640"></div>
  <div><h2>rank x layer</h2><img src="/png/heatmap?step={step}&metric={metric}&t={ts}" width="640"></div>
</div>
<h2>per-layer distribution (rank {rank})</h2>
<img src="/png/layers?rank={rank}&metric={metric}&t={ts}" width="1100">
<script>setTimeout(function(){{ location.reload(); }}, 5000);</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        q = urllib.parse.parse_qs(parsed.query)
        STORE.refresh()
        try:
            if path == '/':
                latest = STORE.latest_step()
                step = int(q.get('step', [latest or 0])[0])
                rank = int(q.get('rank', ['0'])[0])
                metric = q.get('metric', ['fwd'])[0]
                body = INDEX_HTML.format(
                    dirpath=html.escape(STORE.dirpath),
                    n_steps=len(STORE.steps()),
                    latest=latest,
                    step=step,
                    rank=rank,
                    metric=metric,
                    ts=os.getpid(),
                ).encode()
                self._send(body, 'text/html; charset=utf-8')
            elif path == '/api/summary':
                self._send(json.dumps(STORE.summary()).encode(), 'application/json')
            elif path.startswith('/api/step/'):
                step = int(path.rsplit('/', 1)[-1])
                rows = [
                    {'rank': r, 'total_s': t, 'fwd_s': f, 'bwd_s': b, 'a2a_s': a}
                    for r, t, f, b, a in STORE.rank_totals(step)
                ]
                self._send(json.dumps({'step': step, 'ranks': rows}).encode(), 'application/json')
            elif path.startswith('/png/'):
                kind = path[len('/png/') :]
                png = _chart(STORE, kind, q)
                self._send(_png_response(png), 'image/png')
            else:
                self._send(b'not found', 'text/plain', 404)
        except Exception as exc:  # a bad step must not kill the server
            self._send(f'error: {exc}'.encode(), 'text/plain', 400)


def build_static_report(store: TraceStore, out_path: str) -> None:
    """Render every chart once and write a self-contained HTML file."""
    store.refresh()
    latest = store.latest_step()
    if latest is None:
        raise SystemExit('no perf records found')
    charts = {
        'trends': draw_step_trends(store),
        'timeline': draw_step_timeline(store, latest),
        'skew': draw_rank_skew(store, latest),
        'heatmap_fwd': draw_layer_heatmap(store, latest, 'fwd'),
        'heatmap_bwd': draw_layer_heatmap(store, latest, 'bwd'),
        'layers': draw_layer_distribution(store, store.steps(), 0, 'fwd'),
    }
    parts = [
        '<!doctype html><meta charset="utf-8"><title>training perf report</title>',
        '<style>body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa}'
        'img{max-width:100%;border:1px solid #ddd;background:#fff;margin:8px 0}'
        'h2{font-size:16px;margin-top:24px}.meta{color:#555;font-size:13px}</style>',
        f'<h1>training perf report</h1>',
        f'<div class="meta">dir: {html.escape(store.dirpath)} &nbsp;|&nbsp; step: {latest} '
        f'&nbsp;|&nbsp; steps recorded: {len(store.steps())}</div>',
    ]
    for title, png in charts.items():
        b64 = base64.b64encode(png).decode()
        parts.append(f'<h2>{html.escape(title)}</h2><img src="data:image/png;base64,{b64}">')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(parts))
    print(f'wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)')


def main():
    global STORE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--dir', required=True, help='the --perf-trace-dir used by the run')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--max-steps', type=int, default=4096)
    ap.add_argument('--once', action='store_true', help='render a static report and exit')
    ap.add_argument('--out', default='perf_report.html')
    args = ap.parse_args()

    STORE = TraceStore(args.dir, max_steps=args.max_steps)
    added = STORE.refresh()
    print(f'loaded {added} record(s) from {args.dir}; steps={len(STORE.steps())}')

    if args.once:
        build_static_report(STORE, args.out)
        return

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'serving on http://{args.host}:{args.port}/  (dir={args.dir})')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
