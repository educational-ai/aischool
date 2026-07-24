// Lesson 82: gan-game-lab — move the generator, watch the optimal judge D*, JS, W1,
// precision/recall and the gradient that reaches the generator. Model example (two modes).
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("gan-game-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var L = 62, R = 24, T = 26, B = 54;
      var PB = "#fffef9";
      var N = 481, XMIN = -7, XMAX = 7, DX = (XMAX - XMIN) / (N - 1);
      var EPS = 1e-7;

      // real data: two modes (model example, fixed by hand)
      var DATA = [{ m: -2, s: 0.5, w: 0.6 }, { m: 2, s: 0.5, w: 0.4 }];
      var state = { mu: 4.5, sg: 0.5, w2: 0.0, loss: "ns" };

      var xs = new Array(N), pdata = new Array(N);
      for (var i = 0; i < N; i += 1) xs[i] = XMIN + i * DX;
      function normpdf(x, m, s) { var t = (x - m) / s; return Math.exp(-0.5 * t * t) / (s * Math.sqrt(2 * Math.PI)); }
      for (i = 0; i < N; i += 1) {
        var v = 0;
        for (var j = 0; j < DATA.length; j += 1) v += DATA[j].w * normpdf(xs[i], DATA[j].m, DATA[j].s);
        pdata[i] = v;
      }

      function genDensity(mu, sg, w2) {
        var p = new Array(N);
        for (var i2 = 0; i2 < N; i2 += 1) {
          p[i2] = (1 - w2) * normpdf(xs[i2], mu, sg) + w2 * normpdf(xs[i2], 2, sg);
        }
        return p;
      }
      function jsBits(p, q) {
        var s = 0;
        for (var i3 = 0; i3 < N; i3 += 1) {
          var a = Math.max(p[i3], EPS), b = Math.max(q[i3], EPS), m = 0.5 * (a + b);
          s += 0.5 * (a * Math.log2(a / m) + b * Math.log2(b / m)) * DX;
        }
        return s;
      }
      function w1(p, q) {
        var cp = 0, cq = 0, s = 0;
        for (var i4 = 0; i4 < N; i4 += 1) { cp += p[i4] * DX; cq += q[i4] * DX; s += Math.abs(cp - cq) * DX; }
        return s;
      }
      // fidelity / coverage with a support threshold at 2% of the peak
      function prec(p, q) {
        var mx = 0, i5;
        for (i5 = 0; i5 < N; i5 += 1) if (p[i5] > mx) mx = p[i5];
        var tau = 0.02 * mx, num = 0, den = 0;
        for (i5 = 0; i5 < N; i5 += 1) { den += q[i5] * DX; if (p[i5] >= tau) num += q[i5] * DX; }
        return den > 0 ? num / den : 0;
      }
      // generator loss under the optimal discriminator D* = p/(p+q)
      function gloss(mu, sg, w2, kind) {
        var q = genDensity(mu, sg, w2), s = 0;
        for (var i6 = 0; i6 < N; i6 += 1) {
          var p = Math.max(pdata[i6], EPS), qq = Math.max(q[i6], EPS);
          var d = p / (p + qq);
          s += (kind === "ns" ? -Math.log(d) : Math.log(Math.max(1 - d, EPS))) * qq * DX;
        }
        return s;
      }

      K.hint(root, "Модельный пример: синие данные — смесь двух мод, красный генератор вы двигаете сами. Судья не обучается вслепую: показан оптимальный при данном генераторе D*(x) = p_data/(p_data + p_g). Уведите генератор далеко вправо — и JS упрётся в плато, где градиент насыщающей цели почти нулевой, тогда как W₁ продолжает расти. Поднимите долю второй моды — и увидите, чем recall отличается от precision.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var controls2 = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Синяя заливка — p_data, красная — p_g, чёрная кривая — оптимальный судья D*(x) (правая шкала 0…1). JS измеряется в битах и упирается в 1 бит, когда носители перестают пересекаться; W₁ растёт линейно с расстоянием. Precision — доля выборки генератора внутри области данных, recall — доля области данных, покрытая генератором.");
      var cs = K.makeCanvas(stage, W, H, { label: "Плотности данных и генератора, оптимальный дискриминатор", onResize: draw, drag: false });

      function sx(x) { return L + (x - XMIN) / (XMAX - XMIN) * (W - L - R); }
      function sy(y, ymax) { return H - B - y / ymax * (H - T - B); }

      function draw() {
        var ctx = cs.ctx; ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var q = genDensity(state.mu, state.sg, state.w2);
        var ymax = 0, i;
        for (i = 0; i < N; i += 1) { if (pdata[i] > ymax) ymax = pdata[i]; if (q[i] > ymax) ymax = q[i]; }
        ymax = Math.max(ymax * 1.15, 0.05);

        // axes
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(L, H - B); ctx.lineTo(W - R, H - B); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var t = -6; t <= 6; t += 2) { ctx.fillText(String(t), sx(t), H - B + 18); }
        ctx.fillText("x", W - R + 6, H - B + 18);
        ctx.textAlign = "left"; ctx.fillText("плотность", L - 52, T + 4);
        ctx.textAlign = "right"; ctx.fillText("D*(x)", W - R, T + 4);

        // half line for D*
        ctx.strokeStyle = "rgba(110,114,106,0.45)"; ctx.setLineDash([3, 4]);
        var yHalf = H - B - 0.5 * (H - T - B);
        ctx.beginPath(); ctx.moveTo(L, yHalf); ctx.lineTo(W - R, yHalf); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("D* = 1/2 — судья не различает", L + 8, yHalf - 6);

        function fillCurve(p, col, alpha) {
          ctx.beginPath(); ctx.moveTo(sx(xs[0]), H - B);
          for (var k = 0; k < N; k += 1) ctx.lineTo(sx(xs[k]), sy(p[k], ymax));
          ctx.lineTo(sx(xs[N - 1]), H - B); ctx.closePath();
          ctx.fillStyle = alpha; ctx.fill();
          ctx.beginPath();
          for (k = 0; k < N; k += 1) { var X = sx(xs[k]), Y = sy(p[k], ymax); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
          ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.stroke();
        }
        fillCurve(pdata, C.blue, "rgba(49,95,140,0.18)");
        fillCurve(q, C.red, "rgba(185,74,59,0.18)");

        // optimal discriminator
        ctx.beginPath();
        for (i = 0; i < N; i += 1) {
          var d = Math.max(pdata[i], EPS) / (Math.max(pdata[i], EPS) + Math.max(q[i], EPS));
          var X2 = sx(xs[i]), Y2 = H - B - d * (H - T - B);
          if (i === 0) ctx.moveTo(X2, Y2); else ctx.lineTo(X2, Y2);
        }
        ctx.strokeStyle = C.ink || "#171915"; ctx.lineWidth = 1.8; ctx.stroke();

        var js = jsBits(pdata, q);
        var ww = w1(pdata, q);
        var pr = prec(pdata, q), rc = prec(q, pdata);
        var h = 0.02;
        var g = Math.abs((gloss(state.mu + h, state.sg, state.w2, state.loss) -
                          gloss(state.mu - h, state.sg, state.w2, state.loss)) / (2 * h));
        var plateau = js > 0.98;
        ctx.fillStyle = plateau ? C.red : C.muted; ctx.textAlign = "right";
        ctx.fillText(plateau ? "JS на плато: сигнал по сдвигу иссяк" : "носители пересекаются: JS ещё чувствует сдвиг", W - R, T + 24);

        output.set([
          { label: "JS(p_data, p_g), бит", value: js.toFixed(3), color: C.blue },
          { label: "W₁(p_data, p_g)", value: ww.toFixed(2), color: C.green || C.blue },
          { label: "precision (реализм)", value: pr.toFixed(2), color: C.green || C.blue },
          { label: "recall (покрытие)", value: rc.toFixed(2), color: C.gold },
          { label: "|∂L_G/∂μ| " + (state.loss === "ns" ? "(non-saturating)" : "(минимакс)"), value: g.toFixed(3), color: C.red }
        ]);
      }

      K.slider(controls, { label: "центр генератора μ", min: -7, max: 7, step: 0.1, value: state.mu, format: function (v) { return v.toFixed(1); } },
        function (v) { state.mu = v; draw(); });
      K.slider(controls, { label: "ширина генератора σ", min: 0.15, max: 2.5, step: 0.05, value: state.sg, format: function (v) { return v.toFixed(2); } },
        function (v) { state.sg = v; draw(); });
      K.slider(controls2, { label: "доля второй моды", min: 0, max: 1, step: 0.02, value: state.w2, format: function (v) { return v.toFixed(2); } },
        function (v) { state.w2 = v; draw(); });
      K.segmented(controls2, {
        label: "цель генератора", value: "ns",
        options: [{ value: "ns", label: "non-saturating" }, { value: "sat", label: "минимакс" }]
      }, function (v) { state.loss = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
