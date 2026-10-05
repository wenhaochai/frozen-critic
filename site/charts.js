/* Charts of the frozen-critic page, drawn from window.FC_DATA (assets/data/frozen-critic.js, built by
   site/build.py in the frozen-critic repo). A figure without data shows its axes, legend and a status word. */
ArticleCharts.define(function (C, t) {
  var D = window.FC_DATA || {}, S = D.status || {};
  var BLUE = 'var(--ch-s1)', RED = 'var(--ch-s2)', YEL = 'var(--ch-s3)', GRN = 'var(--ch-s4)', REF = 'var(--ch-ref)';
  var WORD = { running: ['Running', '运行中'], queued: ['Queued', '排队中'], planned: ['Planned', '计划中'] };
  function L(o) { return o == null ? '' : (typeof o === 'string' ? o : t(o.en, o.zh)); }
  function f2(v) { return v == null ? '—' : v.toFixed(2); }
  function f3(v) { return v == null ? '—' : v.toFixed(3); }

  /* axes: horizontal grid at y ticks, labels, baseline axis; returns the two scales */
  function frame(el, svg, b, o) {
    var X = function (v) { return b.x + (v - o.x0) / (o.x1 - o.x0) * b.w; };
    var Y = function (v) { return b.y + b.h - (v - o.y0) / (o.y1 - o.y0) * b.h; };
    (o.yt || []).forEach(function (v) {
      el('line', { x1: b.x, y1: Y(v), x2: b.x + b.w, y2: Y(v), stroke: 'var(--ch-grid)', 'stroke-width': 1 }, svg);
      if (!o.noYLab) el('text', { x: b.x - 7, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-4)', text: (o.yf || String)(v) }, svg);
    });
    (o.xt || []).forEach(function (v) {
      if (o.xgrid) el('line', { x1: X(v), y1: b.y, x2: X(v), y2: b.y + b.h, stroke: 'var(--ch-grid)', 'stroke-width': 1 }, svg);
      el('text', { x: X(v), y: b.y + b.h + 16, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-4)', text: (o.xf || String)(v) }, svg);
    });
    if (!o.noBase) el('line', { x1: b.x, y1: b.y + b.h, x2: b.x + b.w, y2: b.y + b.h, stroke: 'var(--ch-axis)', 'stroke-width': 1 }, svg);
    if (o.xl) el('text', { x: b.x + b.w / 2, y: b.y + b.h + 34, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', text: L(o.xl) }, svg);
    if (o.yl) el('text', { x: b.x - 7, y: b.y - 12, 'text-anchor': 'start', 'font-size': 11, fill: 'var(--ink-2)', text: L(o.yl) }, svg);
    if (o.name) el('text', { x: b.x + 8, y: b.y + 14, 'font-size': 11.5, 'font-weight': 600, fill: 'var(--ink-2)', text: L(o.name) }, svg);
    return { X: X, Y: Y };
  }
  function status(el, svg, b, key) {
    var w = WORD[S[key]] || WORD.queued;
    el('text', { x: b.x + b.w / 2, y: b.y + b.h / 2 + 6, 'text-anchor': 'middle', 'font-size': 16, 'font-weight': 600, fill: 'var(--ink-4)', text: t(w[0], w[1]) }, svg);
  }
  function path(el, svg, pts, color, o) {
    o = o || {};
    var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('');
    var a = { d: d, fill: 'none', stroke: color, 'stroke-width': o.w || 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' };
    if (o.dash) a['stroke-dasharray'] = o.dash;
    if (o.op) a['stroke-opacity'] = o.op;
    return el('path', a, svg);
  }
  function hoverDot(h, tp, node, title, rows) {
    node.addEventListener('pointerenter', function (ev) { h.tipRows(tp, title, rows); tp.removeAttribute('hidden'); h.placeTip(tp, ev.clientX, ev.clientY); });
    node.addEventListener('pointerleave', function () { tp.setAttribute('hidden', ''); });
  }

  /* Figure 1: AUC per generator, three readings of the value */
  var GENS = D.gens || [];
  var AUCS = [
    { k: 'auc_last', ci: 'auc_last_ci', label: { en: 'Value before the last token', zh: '最后一个 token 之前的值' }, color: BLUE },
    { k: 'auc_mean', ci: 'auc_mean_ci', label: { en: 'Mean value over the response', zh: '整份回答的平均值' }, color: YEL },
    { k: 'within_auc_last', ci: 'within_auc_last_ci', label: { en: 'Value before the last token, within each problem', zh: '最后一个 token 之前的值，只在题内比较' }, color: GRN },
    { k: 'within_auc_neg_len', ci: 'within_auc_neg_len_ci', label: { en: 'Shorter response, within each problem (no critic)', zh: '回答更短，只在题内比较（不用 critic）' }, color: REF }
  ];
  C.custom('c-auc', AUCS.map(function (s) { return { label: s.label, color: s.color, dot: true }; }), function (body, h) {
    var el = h.el, W = 640, rowH = 46, mL = 176, mR = 18, mT = 26, n = Math.max(GENS.length, 4), H = mT + n * rowH + 44;
    var b = { x: mL, y: mT, w: W - mL - mR, h: n * rowH };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'AUC per generator', zh: '各生成模型的 AUC' }) }, body);
    var s = frame(el, svg, b, { x0: 0.4, x1: 1.0, y0: 0, y1: 1, xt: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], xgrid: true, xf: function (v) { return v.toFixed(1); }, noBase: true,
      xl: { en: 'AUC (0.5 = chance)', zh: 'AUC（0.5 = 随机猜）' } });
    el('line', { x1: s.X(0.5), y1: b.y, x2: s.X(0.5), y2: b.y + b.h, stroke: 'var(--ink-2)', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, svg);
    var R = D.rank, tp = h.tip();
    GENS.forEach(function (g, i) {
      var cy = b.y + i * rowH + rowH / 2;
      el('text', { x: b.x - 10, y: cy + 4, 'text-anchor': 'end', 'font-size': 11.5, fill: 'var(--ink-2)', text: g.label }, svg);
      if (!R || !R[g.key]) return;
      AUCS.forEach(function (a, j) {
        var v = R[g.key][a.k]; if (v == null) return;
        var y = cy + (j - 1.5) * 9;
        var ci = a.ci ? R[g.key][a.ci] : null;
        if (ci) el('line', { x1: s.X(ci[0]), y1: y, x2: s.X(ci[1]), y2: y, stroke: a.color, 'stroke-width': 2, 'stroke-opacity': 0.5 }, svg);
        var c = el('circle', { cx: s.X(v), cy: y, r: 5, fill: a.color, stroke: 'var(--paper)', 'stroke-width': 1.5 }, svg);
        hoverDot(h, tp, c, g.label, [{ color: a.color, label: a.label, v: f3(v) + (ci ? '  [' + f2(ci[0]) + ', ' + f2(ci[1]) + ']' : '') }]);
      });
    });
    if (!R) status(el, svg, b, 'auc');
    else h.table([{ en: 'Generator', zh: '生成模型' }, { en: 'Accuracy', zh: '准确率' }].concat(AUCS.map(function (a) { return a.label; })),
      GENS.map(function (g) { var r = R[g.key] || {}; return [g.label, f3(r.accuracy)].concat(AUCS.map(function (a) { return f3(r[a.k]); })); }));
  });

  /* Figure 2: value along the response, correct vs incorrect, one panel per generator */
  C.custom('c-curves', [{ label: { en: 'Correct', zh: '答对' }, color: BLUE, line: true }, { label: { en: 'Incorrect', zh: '答错' }, color: RED, line: true }], function (body, h) {
    var el = h.el, W = 640, cols = 2, pw = 268, ph = 150, gapX = 56, gapY = 58, top = 30, left = 44;
    var n = Math.max(GENS.length, 4), rows = Math.ceil(n / cols), H = top + rows * ph + (rows - 1) * gapY + 52;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Critic value along the response', zh: '回答各位置的 critic 值' }) }, body);
    var R = D.rank, lo = 0.2, hi = 0.9;
    if (R) {
      var all = [];
      GENS.forEach(function (g) { var r = R[g.key]; if (r) all = all.concat(r.curve_correct || [], r.curve_incorrect || []); });
      if (all.length) { lo = Math.floor(Math.min.apply(null, all) * 10 - 0.5) / 10; hi = Math.ceil(Math.max.apply(null, all) * 10 + 0.5) / 10; }
    }
    var yt = []; for (var v = lo; v <= hi + 1e-9; v += (hi - lo > 0.6 ? 0.2 : 0.1)) yt.push(+v.toFixed(2));
    var rowsT = [], tp = h.tip();
    el('text', { x: 0, y: 12, 'font-size': 11.5, fill: 'var(--ink-2)', text: h.pick({ en: 'Mean critic value', zh: '平均 critic 值' }) }, svg);
    GENS.forEach(function (g, i) {
      var b = { x: left + (i % cols) * (pw + gapX), y: top + Math.floor(i / cols) * (ph + gapY), w: pw, h: ph };
      var s = frame(el, svg, b, { x0: 0, x1: 1, y0: lo, y1: hi, xt: [0, 0.5, 1], yt: yt, xf: function (v) { return v === 0 ? t('start', '开头') : v === 1 ? t('end', '结尾') : '0.5'; }, yf: function (v) { return v.toFixed(1); }, name: g.label });
      var r = R && R[g.key];
      if (!r) { status(el, svg, b, 'curves'); return; }
      [['curve_correct', BLUE, { en: 'Correct', zh: '答对' }], ['curve_incorrect', RED, { en: 'Incorrect', zh: '答错' }]].forEach(function (c) {
        var ys = r[c[0]]; if (!ys) return;
        var pts = ys.map(function (y, k) { return [s.X((k + 0.5) / ys.length), s.Y(y)]; });
        path(el, svg, pts, c[1]);
        rowsT.push([g.label + ' · ' + h.pick(c[2])].concat(ys.filter(function (_, k) { return k % 5 === 0 || k === ys.length - 1; }).map(f3)));
      });
    });
    el('text', { x: W / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', text: h.pick({ en: 'Position in the response', zh: '在回答中的位置' }) }, svg);
    if (R) h.table([{ en: 'Series', zh: '曲线' }, '0–5%', '25–30%', '50–55%', '75–80%', '95–100%'], rowsT);
  });

  /* Figure 3: value before the last token for real, damaged and random responses */
  var KIND = { real: [BLUE, { en: 'Real solution', zh: '真实解答' }], damaged: [YEL, { en: 'Damaged solution', zh: '受损的解答' }], random: [RED, { en: 'Random or empty text', zh: '随机或空文本' }] };
  C.custom('c-probes', Object.keys(KIND).map(function (k) { return { label: KIND[k][1], color: KIND[k][0], dot: true }; }).concat([{ label: { en: 'Prompt alone', zh: '只有题目' }, color: 'var(--ink-2)', line: true }]), function (body, h) {
    var P = D.probes || [], el = h.el, W = 640, rowH = 26, mL = 236, mR = 18, mT = 24, n = Math.max(P.length, 12), H = mT + n * rowH + 44;
    var b = { x: mL, y: mT, w: W - mL - mR, h: n * rowH };
    var x0 = 0, x1 = 1;
    if (P.length) {
      var lo = Math.min.apply(null, P.map(function (p) { return p.p10; })), hi = Math.max.apply(null, P.map(function (p) { return p.p90; }).concat([D.prompt_value || 0]));
      x0 = Math.min(0, Math.floor(lo * 5) / 5); x1 = Math.max(1, Math.ceil(hi * 5) / 5);
    }
    var xt = []; for (var v = x0; v <= x1 + 1e-9; v += 0.2) xt.push(+v.toFixed(1));
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Critic value per kind of response', zh: '各类回答的 critic 值' }) }, body);
    var s = frame(el, svg, b, { x0: x0, x1: x1, y0: 0, y1: 1, xt: xt, xgrid: true, noBase: true, xf: function (v) { return v.toFixed(1); },
      xl: { en: 'Critic value before the last token (dot: mean; line: 10th to 90th percentile)', zh: '最后一个 token 之前的 critic 值（点：均值；线：第 10 到第 90 百分位）' } });
    if (!P.length) { status(el, svg, b, 'probes'); return; }
    if (D.prompt_value != null) el('line', { x1: s.X(D.prompt_value), y1: b.y, x2: s.X(D.prompt_value), y2: b.y + b.h, stroke: 'var(--ink-2)', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, svg);
    var tp = h.tip();
    P.forEach(function (p, i) {
      var cy = b.y + i * rowH + rowH / 2, col = KIND[p.kind][0];
      el('text', { x: b.x - 10, y: cy + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-2)', text: h.pick(p.label) }, svg);
      el('line', { x1: s.X(p.p10), y1: cy, x2: s.X(p.p90), y2: cy, stroke: col, 'stroke-width': 2, 'stroke-opacity': 0.5 }, svg);
      var c = el('circle', { cx: s.X(p.mean), cy: cy, r: 5, fill: col, stroke: 'var(--paper)', 'stroke-width': 1.5 }, svg);
      hoverDot(h, tp, c, h.pick(p.label), [{ color: col, label: { en: 'mean (10th–90th pct)', zh: '均值（第 10–90 百分位）' }, v: f3(p.mean) + ' (' + f2(p.p10) + '–' + f2(p.p90) + ')' }]);
    });
    h.table([{ en: 'Response', zh: '回答' }, { en: 'Mean', zh: '均值' }, { en: '10th pct', zh: '第 10 百分位' }, { en: '90th pct', zh: '第 90 百分位' }, 'n'],
      P.map(function (p) { return [h.pick(p.label), f3(p.mean), f3(p.p10), f3(p.p90), String(p.n)]; }));
  });

  /* Figure 4: greedy value ascent */
  var POOLS = [{ k: 'full', label: { en: '512 random candidates per step', zh: '每步 512 个随机候选' }, color: RED },
               { k: 'small', label: { en: '100-token arithmetic vocabulary', zh: '100 个算术常用 token' }, color: YEL }];
  C.custom('c-ascent', POOLS.map(function (p) { return { label: p.label, color: p.color, line: true }; }).concat([{ label: { en: 'Correct solutions, middle half', zh: '正确解答，中间一半' }, color: REF }]), function (body, h) {
    var A = D.ascent, el = h.el, W = 640, H = 330, b = { x: 52, y: 30, w: 570, h: 250 };
    var y0 = 0.3, y1 = 1.0;
    if (A) {
      var all = []; POOLS.forEach(function (p) { (A[p.k] ? A[p.k].runs : []).forEach(function (r) { all = all.concat(r); }); });
      if (A.band) all = all.concat(A.band);
      if (A.v0) all.push(A.v0);
      y0 = Math.floor(Math.min.apply(null, all) * 10 - 0.5) / 10; y1 = Math.ceil(Math.max.apply(null, all) * 10 + 0.5) / 10;
    }
    var yt = []; for (var v = y0; v <= y1 + 1e-9; v += 0.1) yt.push(+v.toFixed(1));
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Value over appended tokens', zh: '接上的 token 数与 value' }) }, body);
    var s = frame(el, svg, b, { x0: 0, x1: 64, y0: y0, y1: y1, xt: [0, 16, 32, 48, 64], yt: yt, yf: function (v) { return v.toFixed(1); },
      xl: { en: 'Tokens appended to an empty response', zh: '接到空回答后面的 token 数' }, yl: { en: 'Critic value after the token', zh: '接上这个 token 后的 critic 值' } });
    if (!A) { status(el, svg, b, 'ascent'); return; }
    if (A.band) el('rect', { x: b.x, y: s.Y(A.band[1]), width: b.w, height: s.Y(A.band[0]) - s.Y(A.band[1]), fill: REF, 'fill-opacity': 0.16 }, svg);
    var rowsT = [];
    POOLS.forEach(function (p) {
      var P = A[p.k]; if (!P) return;
      P.runs.forEach(function (r) { path(el, svg, r.map(function (v, k) { return [s.X(k + 1), s.Y(v)]; }), p.color, { w: 1, op: 0.35 }); });
      path(el, svg, P.mean.map(function (v, k) { return [s.X(k + 1), s.Y(v)]; }), p.color, { w: 2.5 });
      rowsT.push([h.pick(p.label)].concat([1, 8, 16, 32, 64].map(function (k) { return f3(P.mean[k - 1]); })));
    });
    h.table([{ en: 'Candidates (mean of problems)', zh: '候选（各题平均）' }, '1', '8', '16', '32', '64'], rowsT);
  });

  /* Figure 5: time of one PPO step by stage, per actor size */
  var STAGES = [{ k: 'rollout_s', label: { en: 'Generate responses', zh: '生成回答' }, color: BLUE },
                { k: 'critic_s', label: { en: 'Critic scoring', zh: 'critic 打分' }, color: YEL },
                { k: 'actor_s', label: { en: 'Actor update', zh: '更新 actor' }, color: GRN }];
  C.custom('c-step', STAGES.map(function (s) { return { label: s.label, color: s.color }; }), function (body, h) {
    var T = D.step, sizes = ['0.5B', '1B', '2B', '4B'], el = h.el, W = 640, rowH = 40, mL = 92, mR = 70, mT = 14, H = mT + sizes.length * rowH + 44;
    var b = { x: mL, y: mT, w: W - mL - mR, h: sizes.length * rowH };
    var xmax = 10;
    if (T) { xmax = Math.max.apply(null, T.map(function (r) { return (r.rollout_s + r.critic_s + r.actor_s) / 60; })); xmax = Math.ceil(xmax / (xmax > 10 ? 5 : 1)) * (xmax > 10 ? 5 : 1); }
    var step = xmax > 20 ? 5 : xmax > 8 ? 2 : 1, xt = []; for (var v = 0; v <= xmax + 1e-9; v += step) xt.push(v);
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Minutes per PPO step', zh: '每个 PPO step 的分钟数' }) }, body);
    var s = frame(el, svg, b, { x0: 0, x1: xmax, y0: 0, y1: 1, xt: xt, xgrid: true, noBase: true, xl: { en: 'Minutes per PPO step on one H100', zh: '单张 H100 上每个 PPO step 的分钟数' } });
    var tp = h.tip();
    sizes.forEach(function (z, i) {
      var cy = b.y + i * rowH + rowH / 2;
      el('text', { x: b.x - 10, y: cy + 4, 'text-anchor': 'end', 'font-size': 11.5, fill: 'var(--ink-2)', text: t('Actor ' + z, 'actor ' + z) }, svg);
      var r = T && T.filter(function (q) { return q.size === z; })[0]; if (!r) return;
      var acc = 0;
      STAGES.forEach(function (st) {
        var m = r[st.k] / 60, rc = el('rect', { x: s.X(acc), y: cy - 11, width: Math.max(1, s.X(acc + m) - s.X(acc)), height: 22, fill: st.color }, svg);
        hoverDot(h, tp, rc, t('Actor ', 'actor ') + z, [{ color: st.color, label: st.label, v: m.toFixed(2) + t(' min', ' 分钟') }]);
        acc += m;
      });
      el('text', { x: s.X(acc) + 6, y: cy + 4, 'font-size': 11, fill: 'var(--ink)', text: acc.toFixed(1) + t(' min', ' 分钟') }, svg);
    });
    if (!T) status(el, svg, b, 'step');
    else h.table([{ en: 'Actor', zh: 'actor' }].concat(STAGES.map(function (s) { return s.label; })).concat([{ en: 'Total (min)', zh: '合计（分钟）' }]),
      T.map(function (r) { return [r.size].concat(STAGES.map(function (s) { return (r[s.k] / 60).toFixed(2); })).concat([((r.rollout_s + r.critic_s + r.actor_s) / 60).toFixed(2)]); }));
  });

  /* Figure 6: GSM8K accuracy over PPO steps per lambda (planned) */
  var LAMS = [{ k: '0.95', color: BLUE }, { k: '0.5', color: YEL }, { k: '0', color: RED }];
  C.custom('c-acc', LAMS.map(function (l) { return { label: 'λ = ' + l.k, color: l.color, line: true }; }).concat([{ label: { en: 'Qwen2.5-0.5B-Instruct', zh: 'Qwen2.5-0.5B-Instruct' }, color: REF, line: true }]), function (body, h) {
    var el = h.el, W = 640, H = 300, b = { x: 52, y: 28, w: 570, h: 220 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Accuracy over PPO steps', zh: '准确率随 PPO step 变化' }) }, body);
    var s = frame(el, svg, b, { x0: 0, x1: 400, y0: 0, y1: 0.8, xt: [0, 100, 200, 300, 400], yt: [0, 0.2, 0.4, 0.6, 0.8], yf: function (v) { return v.toFixed(1); },
      xl: { en: 'PPO step', zh: 'PPO step' }, yl: { en: 'GSM8K test accuracy', zh: 'GSM8K 测试准确率' } });
    if (D.ref_acc != null) { el('line', { x1: b.x, y1: s.Y(D.ref_acc), x2: b.x + b.w, y2: s.Y(D.ref_acc), stroke: REF, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg); }
    status(el, svg, b, 'acc');
  });

  /* Figure 7: critic value vs true accuracy of rollouts per lambda (planned) */
  C.custom('c-monitor', [{ label: { en: 'Critic value before the last token', zh: '最后一个 token 之前的 critic 值' }, color: BLUE, line: true }, { label: { en: 'Fraction correct', zh: '答对比例' }, color: GRN, line: true }], function (body, h) {
    var el = h.el, W = 640, pw = 180, gap = 25, H = 250, top = 30;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Critic value and accuracy per lambda', zh: '各 λ 的 critic 值与准确率' }) }, body);
    LAMS.forEach(function (l, i) {
      var b = { x: 36 + i * (pw + gap), y: top, w: pw, h: 160 };
      frame(el, svg, b, { x0: 0, x1: 400, y0: 0, y1: 1, xt: [0, 200, 400], yt: [0, 0.5, 1], yf: function (v) { return v.toFixed(1); }, noYLab: i > 0, name: 'λ = ' + l.k });
      status(el, svg, b, 'monitor');
    });
    el('text', { x: W / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', text: 'PPO step' }, svg);
  });
});

/* Experiment record: one row per training run, from FC_DATA.record */
(function () {
  function draw() {
    var D = window.FC_DATA || {}, R = D.record || [], box = document.getElementById('rec'); if (!box) return;
    var zh = document.documentElement.getAttribute('data-lang') === 'zh';
    var T = function (en, z) { return zh ? z : en; };
    var head = [T('Setup', '做法'), T('Actor', 'actor'), 'GAE λ', T('Tokens', 'Token'), 'H100', T('Hours', '小时'), T('Figures', '图'), T('Code', '代码'), 'W&B'];
    var h = '<table class="rec"><thead><tr>' + head.map(function (x, i) { return '<th' + ((i >= 2 && i <= 5) ? ' class="num"' : '') + '>' + x + '</th>'; }).join('') + '</tr></thead><tbody>';
    R.forEach(function (r) {
      h += '<tr><td>' + (zh ? r.setup_zh : r.setup_en) + '</td><td>' + r.actor + '</td><td class="num">' + r.lam + '</td><td class="num">' + (r.tokens || '—') + '</td><td class="num">' + (r.gpus || '—') +
        '</td><td class="num">' + (r.hours || '—') + '</td><td>' + (r.figs || []).map(function (f) { return '<a href="#' + f[1] + '">' + f[0] + '</a>'; }).join(', ') +
        '</td><td>' + (r.code ? '<a href="' + r.code + '" target="_blank" rel="noopener">' + T('code', '代码') + '</a>' : '') + '</td><td>' + (r.wandb ? '<a href="' + r.wandb + '" target="_blank" rel="noopener">' + T('run', '运行') + '</a>' : '') + '</td></tr>';
    });
    var done = R.filter(function (r) { return r.done; }).length, gh = R.reduce(function (a, r) { return a + (r.gpu_hours || 0); }, 0);
    h += '<tr class="total"><td colspan="9">' + T('Total: ' + R.length + ' runs, ' + done + ' done, ' + gh.toFixed(0) + ' GPU-hours', '合计：' + R.length + ' 个 run，完成 ' + done + ' 个，' + gh.toFixed(0) + ' GPU·时') + '</td></tr></tbody></table>';
    box.innerHTML = h;
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', draw); else draw();
  document.addEventListener('click', function (e) { if (e.target.closest && e.target.closest('.ar-lang button')) setTimeout(draw, 0); });
})();
