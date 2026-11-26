// Interactive per-rank pipeline viewer: zoom, pan, hover readout.
// Data comes from /api/trace?step=N as microseconds relative to the step anchor.
(function () {
  'use strict';

  var STEP = window.__STEP__;
  var canvas = document.getElementById('cv');
  var ctx = canvas.getContext('2d');
  var tip = document.getElementById('tip');
  var readout = document.getElementById('readout');
  var selBox = document.getElementById('sel');
  var wrap = document.getElementById('wrap');

  var data = null;
  var lanes = { fwd: true, bwd: true, comm: true, other: true };
  var view = { t0: 0, t1: 1 };          // visible time window, microseconds
  var rowsTotal = 0;
  var rowH = 34, gutter = 62, axisH = 26, pad = 6;
  var laneH = { fwd: 13, bwd: 13, comm: 7, other: 7 };
  var drag = null;
  var hover = null;
  var selected = null;

  var COLORS = {
    fwd: '#4C78A8',
    bwd: '#54A24B',
    comm: '#F58518',
    other: '#B279A2',
    axis: '#666',
    grid: '#e8e8e8',
    guide: '#c33'
  };

  function laneList() {
    return ['fwd', 'bwd', 'comm', 'other'].filter(function (k) { return lanes[k]; });
  }

  function rowHeight() {
    var h = pad;
    laneList().forEach(function (k) { h += laneH[k] + 2; });
    return Math.max(18, h);
  }

  function layerColor(name, total) {
    // stable hue per layer index, so the same layer keeps its colour in every lane
    var m = /^l(\d+)/.exec(name || '');
    var idx = m ? parseInt(m[1], 10) : 0;
    var t = total > 1 ? idx / (total - 1) : 0;
    var h = 210 - 150 * t;                       // blue -> orange
    return 'hsl(' + h.toFixed(0) + ',60%,52%)';
  }

  function layerCount() {
    var names = {};
    data.ranks.forEach(function (r) { r.fwd.forEach(function (b) { names[b.n] = 1; }); });
    return Math.max(1, Object.keys(names).length);
  }

  function xOf(us) {
    var w = canvas.clientWidth - gutter;
    return gutter + (us - view.t0) / (view.t1 - view.t0) * w;
  }

  function usOf(px) {
    var w = canvas.clientWidth - gutter;
    return view.t0 + (px - gutter) / w * (view.t1 - view.t0);
  }

  function fmt(us) {
    if (us >= 1e6) return (us / 1e6).toFixed(3) + ' s';
    if (us >= 1e3) return (us / 1e3).toFixed(2) + ' ms';
    return us.toFixed(1) + ' us';
  }

  function niceStep(span, px, minPx) {
    var target = span / Math.max(1, px / minPx);
    var pow = Math.pow(10, Math.floor(Math.log10(target)));
    var cands = [1, 2, 5, 10];
    for (var i = 0; i < cands.length; i++) {
      if (cands[i] * pow >= target) return cands[i] * pow;
    }
    return 10 * pow;
  }

  function resize() {
    var dpr = window.devicePixelRatio || 1;
    rowH = rowHeight();
    var rows = data ? data.ranks.length : 1;
    rowsTotal = rows * rowH;
    var h = axisH + rowsTotal + 8;
    canvas.style.height = h + 'px';
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function drawAxis() {
    var w = canvas.clientWidth;
    ctx.save();
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, w, axisH);
    ctx.strokeStyle = '#ddd';
    ctx.beginPath();
    ctx.moveTo(0, axisH + 0.5);
    ctx.lineTo(w, axisH + 0.5);
    ctx.stroke();

    var span = view.t1 - view.t0;
    var step = niceStep(span, w - gutter, 78);
    var t = Math.ceil(view.t0 / step) * step;
    ctx.fillStyle = COLORS.axis;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    while (t <= view.t1) {
      var x = xOf(t);
      ctx.strokeStyle = '#cfcfcf';
      ctx.beginPath();
      ctx.moveTo(x, axisH - 5);
      ctx.lineTo(x, axisH);
      ctx.stroke();
      ctx.fillText(fmt(t), x, axisH / 2 - 1);
      t += step;
    }
    ctx.restore();
  }

  function draw() {
    if (!data) return;
    var w = canvas.clientWidth;
    ctx.clearRect(0, 0, w, canvas.height);
    drawAxis();

    var keys = laneList();
    var total = layerCount();
    for (var i = 0; i < data.ranks.length; i++) {
      var r = data.ranks[i];
      var y = axisH + i * rowH;
      // row background + separator
      if (i % 2 === 1) {
        ctx.fillStyle = '#fcfcfc';
        ctx.fillRect(gutter, y, w - gutter, rowH);
      }
      ctx.strokeStyle = '#f0f0f0';
      ctx.beginPath();
      ctx.moveTo(gutter, y + rowH - 0.5);
      ctx.lineTo(w, y + rowH - 0.5);
      ctx.stroke();

      ctx.fillStyle = '#666';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText('r' + r.rank, gutter - 6, y + rowH / 2);

      var ly = y + pad;
      for (var k = 0; k < keys.length; k++) {
        var lane = keys[k];
        var blocks = r[lane] || [];
        for (var b = 0; b < blocks.length; b++) {
          var blk = blocks[b];
          var x0 = xOf(blk.s);
          var x1 = xOf(blk.s + blk.d);
          if (x1 < gutter || x0 > w) continue;
          // clip to the plot area so panning past the left edge does not smear blocks
          var left = Math.max(gutter, x0);
          var right = Math.min(w, x1);
          if (right - left < 0.5) right = left + 0.5;
          var xw = right - left;
          ctx.fillStyle =
            lane === 'fwd' ? layerColor(blk.n, total)
            : lane === 'bwd' ? layerColor(blk.n, total)
            : lane === 'comm' ? COLORS.comm
            : COLORS.other;
          if (lane === 'bwd') ctx.globalAlpha = 0.85;
          ctx.fillRect(left, ly, xw, laneH[lane]);
          ctx.globalAlpha = 1;
          if (xw > 22) {
            ctx.fillStyle = '#fff';
            ctx.font = '9px ui-monospace, monospace';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.save();
            ctx.beginPath();
            ctx.rect(left, ly, xw, laneH[lane]);
            ctx.clip();
            ctx.fillText(
              lane === 'comm'
                ? 'A2A ' + (blk.layer || blk.n)
                : (blk.n.charAt(0) === 'l' ? 'L' + blk.n.slice(1) : blk.n),
                         left + 3, ly + laneH[lane] / 2 + 0.5);
            ctx.restore();
          }
        }
        ly += laneH[lane] + 2;
      }
    }

    if (hover) {
      ctx.strokeStyle = COLORS.guide;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xOf(hover.us), axisH);
      ctx.lineTo(xOf(hover.us), axisH + rowsTotal);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function pick(mx, my) {
    if (!data) return null;
    var i = Math.floor((my - axisH) / rowH);
    if (i < 0 || i >= data.ranks.length) return null;
    var r = data.ranks[i];
    var keys = laneList();
    var ly = axisH + i * rowH + pad;
    for (var k = 0; k < keys.length; k++) {
      var lane = keys[k];
      if (my >= ly && my < ly + laneH[lane]) {
        var blocks = r[lane] || [];
        for (var b = 0; b < blocks.length; b++) {
          var blk = blocks[b];
          if (mx >= xOf(blk.s) && mx <= xOf(blk.s + blk.d)) {
            return { rank: r.rank, lane: lane, blk: blk, ly: ly };
          }
        }
        return { rank: r.rank, lane: lane, blk: null, ly: ly };
      }
      ly += laneH[lane] + 2;
    }
    return null;
  }

  function showTip(hit, mx, my) {
    if (hit && hit.blk) {
      var b = hit.blk;
      tip.innerHTML =
        '<b>' + hit.lane + '</b> ' + (b.layer ? b.layer + ' ' : '') + b.n +
        '<br>start ' + fmt(b.s) + ' &nbsp; dur ' + fmt(b.d) +
        '<br>end ' + fmt(b.s + b.d) +
        '<br><span style="color:#888">rank ' + hit.rank + '</span>';
      tip.style.display = 'block';
      tip.style.left = Math.min(mx + 14, window.innerWidth - 220) + 'px';
      tip.style.top = my + 14 + 'px';
    } else {
      tip.style.display = 'none';
    }
  }

  function setView(t0, t1) {
    var minSpan = 1;  // 1 microsecond
    if (t1 - t0 < minSpan) {
      var mid = (t0 + t1) / 2;
      t0 = mid - minSpan / 2;
      t1 = mid + minSpan / 2;
    }
    view.t0 = Math.max(0, t0);
    view.t1 = Math.min(data.duration_us * 1.02, t1);
    if (view.t1 - view.t0 < minSpan) view.t1 = view.t0 + minSpan;
    draw();
  }

  canvas.addEventListener('wheel', function (e) {
    e.preventDefault();
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var anchor = usOf(mx);
    var factor = e.deltaY > 0 ? 1.25 : 0.8;
    var span = view.t1 - view.t0;
    var newSpan = span * factor;
    var frac = (anchor - view.t0) / span;
    setView(anchor - frac * newSpan, anchor + (1 - frac) * newSpan);
  }, { passive: false });

  canvas.addEventListener('mousedown', function (e) {
    var rect = canvas.getBoundingClientRect();
    drag = { x: e.clientX - rect.left, t0: view.t0, t1: view.t1 };
  });
  window.addEventListener('mousemove', function (e) {
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (drag) {
      var w = canvas.clientWidth - gutter;
      var dt = (drag.x - mx) / w * (drag.t1 - drag.t0);
      setView(drag.t0 + dt, drag.t1 + dt);
      return;
    }
    if (my < 0 || my > canvas.clientHeight) return;
    var hit = pick(mx, my);
    hover = { us: usOf(mx) };
    if (hit) hover.rank = hit.rank;
    readout.textContent = fmt(Math.max(0, hover.us)) +
      (hover.rank !== undefined ? '   rank ' + hover.rank : '');
    showTip(hit, e.clientX, e.clientY);
    draw();
  });
  window.addEventListener('mouseup', function () { drag = null; });
  canvas.addEventListener('dblclick', function () {
    setView(0, data.duration_us);
  });
  canvas.addEventListener('click', function (e) {
    var rect = canvas.getBoundingClientRect();
    var hit = pick(e.clientX - rect.left, e.clientY - rect.top);
    selected = hit && hit.blk ? hit : null;
    selBox.textContent = selected
      ? 'selected: rank ' + selected.rank + '  ' + selected.lane + '  ' +
        (selected.blk.layer ? selected.blk.layer + ' ' : '') + selected.blk.n +
        '   start ' + fmt(selected.blk.s) + '  dur ' + fmt(selected.blk.d) +
        '  end ' + fmt(selected.blk.s + selected.blk.d)
      : '';
  });

  document.getElementById('zin').onclick = function () {
    var mid = (view.t0 + view.t1) / 2, span = (view.t1 - view.t0) * 0.6;
    setView(mid - span / 2, mid + span / 2);
  };
  document.getElementById('zout').onclick = function () {
    var mid = (view.t0 + view.t1) / 2, span = (view.t1 - view.t0) / 0.6;
    setView(mid - span / 2, mid + span / 2);
  };
  document.getElementById('fit').onclick = function () { setView(0, data.duration_us); };
  ['fwd', 'bwd', 'comm', 'other'].forEach(function (k) {
    var el = document.getElementById('lane-' + k);
    el.checked = lanes[k];
    el.onchange = function () { lanes[k] = el.checked; resize(); };
  });

  window.addEventListener('resize', resize);
  window.addEventListener('keydown', function (e) {
    if (e.key === '+' || e.key === '=') document.getElementById('zin').click();
    else if (e.key === '-') document.getElementById('zout').click();
    else if (e.key === '0') document.getElementById('fit').click();
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      var d = (view.t1 - view.t0) * 0.2 * (e.key === 'ArrowLeft' ? -1 : 1);
      setView(view.t0 + d, view.t1 + d);
    }
  });

  fetch('/api/trace?step=' + STEP)
    .then(function (r) { return r.json(); })
    .then(function (d) {
      data = d;
      view = { t0: 0, t1: d.duration_us };
      document.getElementById('meta').textContent =
        'step ' + d.step + '  ·  ' + d.ranks.length + ' rank(s)  ·  span ' +
        fmt(d.duration_us) + '  ·  wheel=zoom  drag=pan  dblclick=reset';
      resize();
    });
})();
