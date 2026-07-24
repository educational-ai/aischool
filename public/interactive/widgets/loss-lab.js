// Lesson 20: loss lab — shapes, outlier robustness, cross-entropy.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("loss-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 520;

      // fixed scatter with an outlier for the robustness mode
      var PTS = (function () {
        var rng = mulberry32(7);
        var arr = [];
        for (var i = 0; i < 18; i += 1) {
          var x = i / 17 * 10;
          arr.push([x, 0.5 * x + 2 + (rng() - 0.5) * 1.4]);
        }
        arr[13][1] += 8; // outlier
        return arr;
      })();
      function mulberry32(a) {
        return function () {
          a |= 0; a = a + 0x6D2B79F5 | 0;
          var t = Math.imul(a ^ a >>> 15, 1 | a);
          t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
          return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
      }

      var state = { mode: "shapes", slope: 0.5, p: 0.7 };

      K.hint(
        root,
        "Три режима: формы потерь (как штрафуется остаток), устойчивость к выбросу (двигайте прямую, сравнивайте MSE и MAE), кросс-энтропия (двигайте вероятность верного класса).",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "MSE штрафует квадратом (тянется к выбросу), MAE — модулем (держит большинство), Huber мирит их. Кросс-энтропия −log p взрывается у нуля, штрафуя уверенную ошибку. Данные с выбросом — синтетические, для наглядности.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Формы потерь, устойчивость к выбросу и кросс-энтропия",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 34, w: 900, h: 430 };

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        if (state.mode === "shapes") drawShapes(ctx);
        else if (state.mode === "outlier") drawOutlier(ctx);
        else drawCE(ctx);
      }

      function frame(ctx, xlab, ylab, xmin, xmax, ymin, ymax) {
        ctx.strokeStyle = C.line; ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText(xlab, box.x + box.w / 2, box.y + box.h + 30);
        ctx.save(); ctx.translate(box.x - 44, box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText(ylab, 0, 0); ctx.restore();
        return {
          px: function (v) { return box.x + (v - xmin) / (xmax - xmin) * box.w; },
          py: function (v) { return box.y + box.h - (v - ymin) / (ymax - ymin) * box.h; },
        };
      }

      function drawShapes(ctx) {
        var m = frame(ctx, "остаток r", "штраф", -4, 4, -0.3, 6);
        function curve(fn, col, dash) {
          ctx.strokeStyle = col; ctx.lineWidth = 2.4;
          if (dash) ctx.setLineDash([6, 4]); else ctx.setLineDash([]);
          ctx.beginPath();
          for (var r = -4; r <= 4; r += 0.03) {
            var y = Math.max(-0.3, Math.min(6, fn(r)));
            if (r === -4) ctx.moveTo(m.px(r), m.py(y)); else ctx.lineTo(m.px(r), m.py(y));
          }
          ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
        }
        curve(function (r) { return 0.5 * r * r; }, C.blue);
        curve(function (r) { return Math.abs(r); }, C.green);
        curve(function (r) { var a = Math.abs(r), d = 1.5; return a <= d ? 0.5 * r * r : d * (a - 0.5 * d); }, C.gold, true);
        ctx.textAlign = "left";
        ctx.fillStyle = C.blue; ctx.fillText("MSE (квадрат)", box.x + 20, box.y + 24);
        ctx.fillStyle = C.green; ctx.fillText("MAE (модуль)", box.x + 20, box.y + 46);
        ctx.fillStyle = C.gold; ctx.fillText("Huber (смесь)", box.x + 20, box.y + 68);
        output.set([
          { label: "MSE у r=2", value: "2,0 (штраф ×4 к r=1)", color: C.blue },
          { label: "MAE у r=2", value: "2,0 (линейно)", color: C.green },
          { label: "Смысл", value: "квадрат душит выбросами, модуль устойчив", color: C.muted },
        ]);
      }

      function drawOutlier(ctx) {
        var m = frame(ctx, "x", "y", 0, 10, 0, 16);
        // MSE fit (least squares) with current NOT used; show both optimal fits + user line
        var xs = PTS.map(function (p) { return p[0]; }), ys = PTS.map(function (p) { return p[1]; });
        var n = xs.length, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (var i = 0; i < n; i += 1) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
        var slopeMSE = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        var intMSE = (sy - slopeMSE * sx) / n;
        // MAE fit via coarse search over slope,int
        var best = null;
        for (var sl = 0.2; sl <= 0.8; sl += 0.01) {
          for (var ic = 0; ic <= 4; ic += 0.1) {
            var e = 0; for (var j = 0; j < n; j += 1) e += Math.abs(sl * xs[j] + ic - ys[j]);
            if (!best || e < best.e) best = { e: e, sl: sl, ic: ic };
          }
        }
        // points
        PTS.forEach(function (p, i) {
          ctx.beginPath(); ctx.arc(m.px(p[0]), m.py(p[1]), i === 13 ? 8 : 4.5, 0, Math.PI * 2);
          ctx.fillStyle = i === 13 ? C.red : (C.faint || C.muted); ctx.fill();
        });
        ctx.textAlign = "left"; ctx.fillStyle = C.red;
        ctx.fillText("выброс", m.px(PTS[13][0]) + 10, m.py(PTS[13][1]));
        function line(sl, ic, col, lab, yo) {
          ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.beginPath();
          ctx.moveTo(m.px(0), m.py(ic)); ctx.lineTo(m.px(10), m.py(sl * 10 + ic)); ctx.stroke();
          ctx.lineWidth = 1; ctx.fillStyle = col;
          ctx.fillText(lab, box.x + 20, box.y + yo);
        }
        line(slopeMSE, intMSE, C.blue, "MSE-прямая (кренится к выбросу)", 24);
        line(best.sl, best.ic, C.green, "MAE-прямая (держит большинство)", 46);
        output.set([
          { label: "Наклон MSE", value: slopeMSE.toFixed(2).replace(".", ","), color: C.blue },
          { label: "Наклон MAE", value: best.sl.toFixed(2).replace(".", ","), color: C.green },
          { label: "Истинный наклон", value: "0,50 — MAE ближе", color: C.muted },
        ]);
      }

      function drawCE(ctx) {
        var m = frame(ctx, "вероятность верного класса p", "-log p", 0, 1, 0, 5.5);
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.6; ctx.beginPath();
        for (var p = 0.004; p <= 1; p += 0.004) {
          var y = Math.min(5.5, -Math.log(p));
          if (p <= 0.004) ctx.moveTo(m.px(p), m.py(y)); else ctx.lineTo(m.px(p), m.py(y));
        }
        ctx.stroke(); ctx.lineWidth = 1;
        // user point
        var yv = -Math.log(state.p);
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(m.px(state.p), m.py(Math.min(5.5, yv)), 7, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = C.line; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(m.px(state.p), box.y + box.h); ctx.lineTo(m.px(state.p), m.py(Math.min(5.5, yv))); ctx.stroke();
        ctx.setLineDash([]);
        output.set([
          { label: "p верного класса", value: state.p.toFixed(2).replace(".", ","), color: C.ink },
          { label: "Кросс-энтропия -log p", value: yv.toFixed(2).replace(".", ","), color: yv > 1.5 ? C.red : C.green },
          { label: "Смысл", value: state.p > 0.5 ? "уверенно и верно — почти даром" : "неуверенно/неверно — дорого", color: C.muted },
        ]);
      }

      K.segmented(controls, { label: "Режим", value: state.mode,
        options: [
          { value: "shapes", label: "формы потерь" },
          { value: "outlier", label: "выброс: MSE vs MAE" },
          { value: "ce", label: "кросс-энтропия" },
        ] }, function (v) { state.mode = v; draw(); });
      K.slider(controls, { label: "Вероятность p (для кросс-энтропии)", min: 0.02, max: 0.99, step: 0.01, value: state.p,
        format: function (v) { return v.toFixed(2).replace(".", ","); } },
        function (v) { state.p = v; if (state.mode === "ce") draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
