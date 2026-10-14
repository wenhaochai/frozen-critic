/* Charts of the frozen-critic page, drawn from window.FC_DATA (assets/data/frozen-critic.js, built by
   site/build.py in the frozen-critic repo). A figure without data shows its axes, legend and a status word. */
ArticleCharts.define(function (C, t) {
  var D = window.FC_DATA || {}, S = D.status || {};
  var BLUE = 'var(--ch-s1)', RED = 'var(--ch-s2)', YEL = 'var(--ch-s3)', GRN = 'var(--ch-s4)', REF = 'var(--ch-ref)';
  var WORD = { running: ['Running', '运行中'], queued: ['Queued', '排队中'], planned: ['Planned', '计划中'], stopped: ['Stopped', '已停止'], done: ['No data', '无数据'] };
  function L(o) { return o == null ? '' : (typeof o === 'string' ? o : t(o.en, o.zh)); }
  function f2(v) { return v == null ? '—' : v.toFixed(2); }
  function f3(v) { return v == null ? '—' : v.toFixed(3); }

  /* axes: horizontal grid at y ticks, labels, baseline axis; returns the two scales */
  function frame(el, svg, b, o) {
    var X = function (v) { return b.x + (v - o.x0) / (o.x1 - o.x0) * b.w; };
    var Y = function (v) { return b.y + b.h - (v - o.y0) / (o.y1 - o.y0) * b.h; };
    (o.yt || []).forEach(function (v) {
      el('line', { x1: b.x, y1: Y(v), x2: b.x + b.w, y2: Y(v), stroke: 'var(--ch-grid)', 'stroke-width': 1 }, svg);
      if (!o.noYLab) el('text', { class: 'tk', x: b.x - 7, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--ink-4)', text: (o.yf || String)(v) }, svg);
    });
    (o.xt || []).forEach(function (v) {
      if (o.xgrid) el('line', { x1: X(v), y1: b.y, x2: X(v), y2: b.y + b.h, stroke: 'var(--ch-grid)', 'stroke-width': 1 }, svg);
      el('text', { class: 'tk', x: X(v), y: b.y + b.h + 16, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-4)', text: (o.xf || String)(v) }, svg);
    });
    if (!o.noBase) el('line', { x1: b.x, y1: b.y + b.h, x2: b.x + b.w, y2: b.y + b.h, stroke: 'var(--ch-axis)', 'stroke-width': 1 }, svg);
    if (o.xl) el('text', { class: 'ax', x: b.x + b.w / 2, y: b.y + b.h + 34, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', text: L(o.xl) }, svg);
    if (o.yl) el('text', { class: 'ax', x: b.x - 7, y: b.y - 12, 'text-anchor': 'start', 'font-size': 11, fill: 'var(--ink-2)', text: L(o.yl) }, svg);
    if (o.name) el('text', { class: 'pn', x: b.x + 8, y: b.y + 14, 'font-size': 11.5, 'font-weight': 600, fill: 'var(--ink-2)', text: L(o.name) }, svg);
    return { X: X, Y: Y };
  }
  function status(el, svg, b, key, lam) {  /* S[key] is a state, or {lambda: state} for the per-lambda panels */
    var st = S[key]; if (st && typeof st === 'object') st = st[lam];
    var w = WORD[st] || WORD.queued;
    el('text', { class: 'big', x: b.x + b.w / 2, y: b.y + b.h / 2 + 6, 'text-anchor': 'middle', 'font-size': 16, 'font-weight': 600, fill: 'var(--ink-4)', text: t(w[0], w[1]) }, svg);
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

  /* Figure 1: AUC per generator, three readings of the value and a length baseline */
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
      el('text', { class: 'rl', x: b.x - 10, y: cy + 4, 'text-anchor': 'end', 'font-size': 11.5, fill: 'var(--ink-2)', text: g.label }, svg);
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

  /* Figure 2: the character-level actor per lambda: test accuracy, boxed-number fraction, critic value (D.phase1) */
  var LAMS = ['0.95', '0.5', '0'], P1 = D.phase1 || {};
  C.custom('c-arith', [
    { label: { en: 'GSM8K test accuracy (greedy)', zh: 'GSM8K 测试准确率（greedy）' }, color: GRN, line: true },
    { label: { en: 'Training responses ending in a boxed number', zh: '以 boxed 数字结束的训练回答比例' }, color: YEL, line: true },
    { label: { en: 'Critic value before the last token', zh: '最后一个 token 之前的 critic 值' }, color: BLUE, line: true },
    { label: { en: 'Qwen2.5-0.5B-Instruct accuracy', zh: 'Qwen2.5-0.5B-Instruct 准确率' }, color: REF, line: true }], function (body, h) {
    var el = h.el, W = 640, pw = 180, gap = 25, H = 250, top = 30, tp = h.tip();
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Accuracy, boxed answers and critic value per lambda', zh: '各 λ 的准确率、boxed 答案比例和 critic 值' }) }, body);
    LAMS.forEach(function (l, i) {
      var b = { x: 36 + i * (pw + gap), y: top, w: pw, h: 160 };
      var s = frame(el, svg, b, { x0: 0, x1: 300, y0: 0, y1: 1, xt: [0, 150, 300], yt: [0, 0.5, 1], yf: function (v) { return v.toFixed(1); }, noYLab: i > 0, name: 'λ = ' + l });
      if (D.ref_acc != null) el('line', { x1: b.x, y1: s.Y(D.ref_acc), x2: b.x + b.w, y2: s.Y(D.ref_acc), stroke: REF, 'stroke-width': 1.3, 'stroke-dasharray': '4 3' }, svg);
      var tr = (P1[l] || {}).train || [], ev = (P1[l] || {}).eval || [];
      if (!tr.length && !ev.length) { status(el, svg, b, 'arith', l); return; }
      // train rows: [rollout, critic value before the last token, fraction correct, fraction ending in a boxed number]
      if (tr.length) {
        path(el, svg, tr.map(function (q) { return [s.X(q[0] + 1), s.Y(q[3])]; }), YEL, { w: 1.6 });
        path(el, svg, tr.map(function (q) { return [s.X(q[0] + 1), s.Y(q[1])]; }), BLUE, { w: 1.6 });
      }
      // eval 0 runs before training; eval N (N > 0) runs after rollout N, that is after N + 1 PPO steps
      var e = ev.map(function (q) { return [q[0] ? q[0] + 1 : 0, q[1]]; });
      if (e.length) path(el, svg, e.map(function (q) { return [s.X(q[0]), s.Y(q[1])]; }), GRN, { w: 2 });
      var z = tr[tr.length - 1], ze = e[e.length - 1], hit = el('rect', { x: b.x, y: b.y, width: b.w, height: b.h, fill: 'transparent' }, svg);
      hoverDot(h, tp, hit, 'λ = ' + l, [
        { color: GRN, label: t('Test accuracy', '测试准确率'), v: ze ? f3(ze[1]) + t(' at step ', '，step ') + ze[0] : '—' },
        { color: YEL, label: t('Boxed-number fraction', 'boxed 数字比例'), v: z ? f3(z[3]) : '—' },
        { color: BLUE, label: t('Critic value', 'critic 值'), v: z ? f3(z[1]) : '—' }]);
    });
    el('text', { class: 'ax', x: W / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', text: 'PPO step' }, svg);
  });

  /* Figure 3: the same arm with centred critic values V - V(prompt) (D.phase1c), plus mean response length / 1,024 */
  var P1C = D.phase1c || {};
  C.custom('c-center', [
    { label: { en: 'GSM8K test accuracy (greedy)', zh: 'GSM8K 测试准确率（greedy）' }, color: GRN, line: true },
    { label: { en: 'Training responses ending in a boxed number', zh: '以 boxed 数字结束的训练回答比例' }, color: YEL, line: true },
    { label: { en: 'Mean response length / 1,024', zh: '平均回答长度 / 1,024' }, color: RED, line: true },
    { label: { en: 'Critic value before the last token', zh: '最后一个 token 之前的 critic 值' }, color: BLUE, line: true },
    { label: { en: 'Qwen2.5-0.5B-Instruct accuracy', zh: 'Qwen2.5-0.5B-Instruct 准确率' }, color: REF, line: true }], function (body, h) {
    var el = h.el, W = 640, pw = 180, gap = 25, H = 250, top = 30, tp = h.tip();
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: 'Centred values: accuracy, boxed answers, length and critic value per lambda', zh: '中心化 value：各 λ 的准确率、boxed 比例、长度和 critic 值' }) }, body);
    LAMS.forEach(function (l, i) {
      var b = { x: 36 + i * (pw + gap), y: top, w: pw, h: 160 };
      var s = frame(el, svg, b, { x0: 0, x1: 300, y0: 0, y1: 1, xt: [0, 150, 300], yt: [0, 0.5, 1], yf: function (v) { return v.toFixed(1); }, noYLab: i > 0, name: 'λ = ' + l });
      if (D.ref_acc != null) el('line', { x1: b.x, y1: s.Y(D.ref_acc), x2: b.x + b.w, y2: s.Y(D.ref_acc), stroke: REF, 'stroke-width': 1.3, 'stroke-dasharray': '4 3' }, svg);
      var tr = (P1C[l] || {}).train || [], ev = (P1C[l] || {}).eval || [];
      if (!tr.length && !ev.length) { status(el, svg, b, 'center', l); return; }
      // train rows: [rollout, critic value before the last token, fraction correct, fraction ending in a boxed number, length / 1,024]
      if (tr.length) {
        path(el, svg, tr.map(function (q) { return [s.X(q[0] + 1), s.Y(q[4])]; }), RED, { w: 1.6 });
        path(el, svg, tr.map(function (q) { return [s.X(q[0] + 1), s.Y(q[3])]; }), YEL, { w: 1.6 });
        path(el, svg, tr.map(function (q) { return [s.X(q[0] + 1), s.Y(q[1])]; }), BLUE, { w: 1.6 });
      }
      var e = ev.map(function (q) { return [q[0] ? q[0] + 1 : 0, q[1]]; });
      if (e.length) path(el, svg, e.map(function (q) { return [s.X(q[0]), s.Y(q[1])]; }), GRN, { w: 2 });
      var z = tr[tr.length - 1], ze = e[e.length - 1], hit = el('rect', { x: b.x, y: b.y, width: b.w, height: b.h, fill: 'transparent' }, svg);
      hoverDot(h, tp, hit, 'λ = ' + l, [
        { color: GRN, label: t('Test accuracy', '测试准确率'), v: ze ? f3(ze[1]) + t(' at step ', '，step ') + ze[0] : '—' },
        { color: YEL, label: t('Boxed-number fraction', 'boxed 数字比例'), v: z ? f3(z[3]) : '—' },
        { color: RED, label: t('Mean length / 1,024', '平均长度 / 1,024'), v: z ? f3(z[4]) : '—' },
        { color: BLUE, label: t('Critic value', 'critic 值'), v: z ? f3(z[1]) : '—' }]);
    });
    el('text', { class: 'ax', x: W / 2, y: H - 8, 'text-anchor': 'middle', 'font-size': 11.5, fill: 'var(--ink-2)', text: 'PPO step' }, svg);
  });

  /* Figure 4: how often the critic's top-valued \boxed{k} is the answer (D.answer): among 21 candidates, and among 0..999 */
  var A = D.answer;
  C.custom('c-answer', [
    { label: { en: "Critic's top-valued answer, the problem's own question", zh: 'critic 给分最高的答案，用本题的题目' }, color: BLUE, dot: true },
    { label: { en: "The same, under another problem's question", zh: '同上，但换成另一道题的题目' }, color: YEL, dot: true },
    { label: { en: 'Without the critic', zh: '不用 critic' }, color: REF, dot: true }], function (body, h) {
    var el = h.el, W = 640, mL = 214, mR = 24, rowH = 30, tp = h.tip();
    var P = [
      { name: { en: 'A  21 candidates: the answer and 20 other numbers', zh: 'A  21 个候选：正确答案和 20 个其他数字' }, x1: 0.15, xt: [0, 0.05, 0.1, 0.15],
        rows: A ? [{ label: { en: 'Critic, own question', zh: 'critic，本题题目' }, v: A.pick.top1, c: BLUE },
                   { label: { en: 'Chance (1 in 21)', zh: '随机猜（21 选 1）' }, v: A.pick.chance, c: REF }] : [] },
      { name: { en: 'B  All numbers 0 to 999', zh: 'B  0 到 999 的全部数字' }, x1: 0.10, xt: [0, 0.025, 0.05, 0.075, 0.1],
        rows: A ? [{ label: { en: 'Critic, own question', zh: 'critic，本题题目' }, v: A.open.own, c: BLUE },
                   { label: { en: 'Always ' + A.open.const + ' (top training answer)', zh: '总是答 ' + A.open.const + '（训练集最常见的答案）' }, v: A.open.const_200, c: REF },
                   { label: { en: "Critic, another problem's question", zh: 'critic，另一道题的题目' }, v: A.open.other, c: YEL },
                   { label: { en: 'Chance (1 in 1,000)', zh: '随机猜（1,000 选 1）' }, v: A.open.chance, c: REF }] : [] }];
    var y = 30, boxes = [];
    P.forEach(function (p) { var n = Math.max(p.rows.length, 2); boxes.push({ x: mL, y: y + 20, w: W - mL - mR, h: n * rowH }); y += 20 + n * rowH + 52; });
    var H = y - 12;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': h.pick({ en: "How often the critic's top-valued answer is right", zh: 'critic 给分最高的答案有多少是对的' }) }, body);
    P.forEach(function (p, i) {
      var b = boxes[i], pct = function (v) { return (100 * v).toFixed(1) + '%'; };
      var s = frame(el, svg, b, { x0: 0, x1: p.x1, y0: 0, y1: 1, xt: p.xt, xgrid: true, noBase: true, xf: function (v) { return (100 * v).toFixed(Math.round(v * 1000) % 10 ? 1 : 0) + '%'; },
        xl: { en: 'Problems where the answer is right', zh: '答对的题目比例' } });
      el('text', { class: 'pn', x: 8, y: b.y - 10, 'font-size': 11.5, 'font-weight': 600, fill: 'var(--ink-2)', text: L(p.name) }, svg);
      if (!A) { status(el, svg, b, 'answer'); return; }
      p.rows.forEach(function (r, j) {
        var cy = b.y + j * rowH + rowH / 2, m = Array.isArray(r.v) ? r.v[0] : r.v, ci = Array.isArray(r.v) ? r.v.slice(1) : null;
        el('text', { class: 'rl', x: b.x - 10, y: cy + 4, 'text-anchor': 'end', 'font-size': 11.5, fill: 'var(--ink-2)', text: L(r.label) }, svg);
        var bar = el('rect', { x: b.x, y: cy - 8, width: Math.max(1.5, s.X(m) - b.x), height: 16, rx: 2, fill: r.c }, svg);
        if (ci) {
          el('line', { x1: s.X(ci[0]), y1: cy, x2: s.X(ci[1]), y2: cy, stroke: 'var(--ink-2)', 'stroke-width': 1.3 }, svg);
          [ci[0], ci[1]].forEach(function (v) { el('line', { x1: s.X(v), y1: cy - 4, x2: s.X(v), y2: cy + 4, stroke: 'var(--ink-2)', 'stroke-width': 1.3 }, svg); });
        }
        el('text', { class: 'vl', x: (ci ? s.X(ci[1]) : s.X(m)) + 6, y: cy + 4, 'font-size': 11, fill: 'var(--ink-2)', text: pct(m) }, svg);
        var hit = el('rect', { x: b.x, y: cy - rowH / 2, width: b.w, height: rowH, fill: 'transparent' }, svg);
        hoverDot(h, tp, hit, L(p.name).slice(3), [{ color: r.c, label: L(r.label), v: pct(m) + (ci ? '  [' + pct(ci[0]) + ', ' + pct(ci[1]) + ']' : '') }]);
      });
    });
    if (A) h.table([{ en: 'Candidates', zh: '候选' }, { en: 'Reading', zh: '读法' }, { en: 'Answer right', zh: '答对比例' }, { en: '95% interval', zh: '95% 区间' }],
      P.reduce(function (acc, p, i) { return acc.concat(p.rows.map(function (r) {
        var v = Array.isArray(r.v) ? r.v : [r.v];
        return [i ? '0–999 (n = ' + A.open.n + ')' : t('21 (n = ', '21（n = ') + A.pick.n + t(')', '）'), L(r.label), (100 * v[0]).toFixed(1) + '%', v.length > 1 ? (100 * v[1]).toFixed(1) + '–' + (100 * v[2]).toFixed(1) + '%' : '—'];
      })); }, []));
  });
});

/* Experiment record: one row per main run and the two closing lines, from FC_RECORD (site/build_record.py) */
(function () {
  function draw() {
    var R = window.FC_RECORD, box = document.getElementById('rec'); if (!box || !R) return;
    var zh = document.documentElement.getAttribute('data-lang') === 'zh';
    var T = function (en, z) { return zh ? z : en; };
    var ST = { planned: T('planned', '计划中'), queued: T('queued', '排队中'), running: T('running', '运行中'), stopped: T('stopped', '已停止') };
    var head = [T('Setup', '做法'), T('Actor', 'actor'), 'GAE λ', 'H100', T('Hours', '小时'), T('Figures', '图'), T('Code', '代码'), 'W&B'];
    var h = '<table class="rec"><thead><tr>' + head.map(function (x, i) { return '<th' + ((i >= 2 && i <= 4) ? ' class="num"' : '') + '>' + x + '</th>'; }).join('') + '</tr></thead><tbody>';
    R.rows.forEach(function (r) {
      var hours = r.hours != null && r.state === 'done' ? r.hours.toFixed(1) : (ST[r.state] || '—');
      h += '<tr><td>' + (zh ? r.setup_zh : r.setup_en) + '</td><td>' + r.actor + '</td><td class="num">' + r.lam + '</td><td class="num">' + r.gpus +
        '</td><td class="num">' + hours + '</td><td>' + (r.figs || []).map(function (f) { return '<a href="#' + f[1] + '">' + f[0] + '</a>'; }).join(', ') +
        '</td><td>' + (r.code ? '<a href="' + r.code + '" target="_blank" rel="noopener">' + T('code', '代码') + '</a>' : '') + '</td><td>' + (r.wandb ? '<a href="' + r.wandb + '" target="_blank" rel="noopener">' + T('run', '运行') + '</a>' : '') + '</td></tr>';
    });
    h += '<tr class="total"><td colspan="8">' + T('Main runs: ' + R.main.h100h.toFixed(1) + ' H100-hours, ' + R.main.jobs + (R.main.jobs === 1 ? ' job' : ' jobs'), '正式的 job：' + R.main.h100h.toFixed(1) + ' H100·时，' + R.main.jobs + ' 个 job') + '</td></tr>';
    var a1 = R.other.a100h ? R.other.a100h.toFixed(1) : null;  // the two answer_pick jobs ran on A100
    h += '<tr class="total"><td colspan="8">' + T('Other compute: ' + R.other.h100h.toFixed(1) + ' H100-hours' + (a1 ? ' and ' + a1 + ' A100-hours' : '') + ', ' + R.other.jobs + (R.other.jobs === 1 ? ' job' : ' jobs'), 'Other compute：' + R.other.h100h.toFixed(1) + ' H100·时' + (a1 ? '、' + a1 + ' A100·时' : '') + '，' + R.other.jobs + ' 个 job') + '</td></tr></tbody></table>';
    box.innerHTML = h;
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', draw); else draw();
  document.addEventListener('click', function (e) { if (e.target.closest && e.target.closest('.ar-lang button')) setTimeout(draw, 0); });
})();
