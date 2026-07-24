// Lesson 53: decision-boundary-lab — direction of w, its length, the shift b and the
// acting threshold are four separate knobs. Rotating w changes the geometry, stretching w
// changes only confidence, b and the threshold move the same line in parallel.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("decision-boundary-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var PLOT = { x0: 60, y0: 30, x1: 620, y1: 440 };
      var PAPER = "#fffef9";

      // ------------------------------------------------ deterministic model data
      function lcg(seed) {
        var s = seed >>> 0;
        return function () {
          s = (1664525 * s + 1013904223) >>> 0;
          return s / 4294967296;
        };
      }
      function gauss(rand) {
        var u = Math.max(rand(), 1e-9), v = rand();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      var pts = [];
      (function build() {
        var rand = lcg(53);
        var i;
        for (i = 0; i < 150; i += 1) {
          pts.push({ x: -0.75 + 0.62 * gauss(rand), y: -0.35 + 0.72 * gauss(rand), y1: 0 });
        }
        for (i = 0; i < 60; i += 1) {
          pts.push({ x: 0.85 + 0.60 * gauss(rand), y: 0.45 + 0.64 * gauss(rand), y1: 1 });
        }
      })();

      // ------------------------------------------------------------- parameters
      var angle = 35;      // degrees: direction of w
      var mag = 2.4;       // |w|
      var bias = -0.2;     // b
      var thr = 0.5;       // acting probability threshold
      var costFN = 10;     // cost of a miss, cost of a false alarm = 1

      function nvec() {
        var a = angle * Math.PI / 180;
        return [Math.cos(a), Math.sin(a)];
      }
      function score(p) {
        var n = nvec();
        return mag * (n[0] * p.x + n[1] * p.y) + bias;
      }
      function sigma(s) { return 1 / (1 + Math.exp(-s)); }
      function logit(p) { return Math.log(p / (1 - p)); }

      function stats() {
        var tp = 0, fp = 0, fn = 0, tn = 0, grey = 0, t = logit(thr), i;
        for (i = 0; i < pts.length; i += 1) {
          var s = score(pts[i]), p = sigma(s), hit = s >= t;
          if (p > 0.2 && p < 0.8) grey += 1;
          if (hit && pts[i].y1 === 1) tp += 1;
          else if (hit) fp += 1;
          else if (pts[i].y1 === 1) fn += 1;
          else tn += 1;
        }
        return {
          tp: tp, fp: fp, fn: fn, tn: tn, grey: grey,
          acc: (tp + tn) / pts.length,
          prec: tp + fp > 0 ? tp / (tp + fp) : 1,
          rec: tp + fn > 0 ? tp / (tp + fn) : 0,
          loss: (fp * 1 + fn * costFN) / pts.length
        };
      }

      // ---------------------------------------------------------------- drawing
      function px(x) { return PLOT.x0 + (x + 3) / 6 * (PLOT.x1 - PLOT.x0); }
      function py(y) { return PLOT.y1 - (y + 3) / 6 * (PLOT.y1 - PLOT.y0); }

      function lineAt(ctx, c, color, width, dash) {
        // set of points with n·u = c, drawn across the box
        var n = nvec(), perp = [-n[1], n[0]], L = 9;
        var ax = c * n[0] - L * perp[0], ay = c * n[1] - L * perp[1];
        var bx = c * n[0] + L * perp[0], by = c * n[1] + L * perp[1];
        ctx.save();
        ctx.beginPath();
        ctx.rect(PLOT.x0, PLOT.y0, PLOT.x1 - PLOT.x0, PLOT.y1 - PLOT.y0);
        ctx.clip();
        ctx.setLineDash(dash || []);
        ctx.strokeStyle = color; ctx.lineWidth = width;
        ctx.beginPath(); ctx.moveTo(px(ax), py(ay)); ctx.lineTo(px(bx), py(by)); ctx.stroke();
        ctx.restore();
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var st = stats(), t = logit(thr), i, j;

        // probability wash
        var cols = 56, rows = 42;
        for (i = 0; i < cols; i += 1) {
          for (j = 0; j < rows; j += 1) {
            var ux = -3 + 6 * (i + 0.5) / cols, uy = -3 + 6 * (j + 0.5) / rows;
            var p = sigma(score({ x: ux, y: uy }));
            var a = Math.abs(p - 0.5) * 0.5;
            ctx.fillStyle = p >= 0.5
              ? "rgba(185,74,59," + a.toFixed(3) + ")"
              : "rgba(49,95,140," + a.toFixed(3) + ")";
            ctx.fillRect(
              PLOT.x0 + i / cols * (PLOT.x1 - PLOT.x0) - 0.5,
              PLOT.y0 + j / rows * (PLOT.y1 - PLOT.y0) - 0.5,
              (PLOT.x1 - PLOT.x0) / cols + 1,
              (PLOT.y1 - PLOT.y0) / rows + 1
            );
          }
        }
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(PLOT.x0, PLOT.y0, PLOT.x1 - PLOT.x0, PLOT.y1 - PLOT.y0);

        // s = 0 and the acting line
        lineAt(ctx, -bias / mag, C.ink, 2.6);
        lineAt(ctx, (t - bias) / mag, C.gold, 2.4, [7, 5]);

        // normal arrow from the s = 0 line
        var n = nvec(), base = [-bias / mag * n[0], -bias / mag * n[1]];
        var tip = [base[0] + 0.9 * n[0], base[1] + 0.9 * n[1]];
        ctx.strokeStyle = C.green; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(px(base[0]), py(base[1])); ctx.lineTo(px(tip[0]), py(tip[1])); ctx.stroke();
        var ang = Math.atan2(py(tip[1]) - py(base[1]), px(tip[0]) - px(base[0]));
        ctx.fillStyle = C.green;
        ctx.beginPath();
        ctx.moveTo(px(tip[0]), py(tip[1]));
        ctx.lineTo(px(tip[0]) - 11 * Math.cos(ang - 0.4), py(tip[1]) - 11 * Math.sin(ang - 0.4));
        ctx.lineTo(px(tip[0]) - 11 * Math.cos(ang + 0.4), py(tip[1]) - 11 * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
        ctx.fillText("w", px(tip[0]) + 8, py(tip[1]) - 4);

        // points
        for (i = 0; i < pts.length; i += 1) {
          var pt = pts[i], s = score(pt), hit = s >= t;
          var wrong = (hit ? 1 : 0) !== pt.y1;
          ctx.beginPath();
          ctx.arc(px(pt.x), py(pt.y), pt.y1 === 1 ? 5.5 : 4.5, 0, 7);
          ctx.fillStyle = pt.y1 === 1 ? C.red : C.blue;
          ctx.fill();
          if (wrong) {
            ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.stroke();
          } else {
            ctx.strokeStyle = PAPER; ctx.lineWidth = 1; ctx.stroke();
          }
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("признак x₁", PLOT.x1 - 96, PLOT.y1 + 20);
        ctx.save();
        ctx.translate(PLOT.x0 - 22, PLOT.y0 + 60); ctx.rotate(-Math.PI / 2);
        ctx.fillText("признак x₂", 0, 0); ctx.restore();

        // score axis on the right
        var AX = 655, AY0 = 60, AY1 = 410;
        function sy(s) { return AY1 - (s + 8) / 16 * (AY1 - AY0); }
        ctx.strokeStyle = C.axis; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(AX, AY0); ctx.lineTo(AX, AY1); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        for (var v = -8; v <= 8; v += 4) {
          ctx.beginPath(); ctx.moveTo(AX - 4, sy(v)); ctx.lineTo(AX + 4, sy(v)); ctx.stroke();
          ctx.fillText(String(v), AX - 8, sy(v) + 4);
        }
        ctx.textAlign = "left";
        ctx.fillText("шкала score", AX - 30, AY0 - 18);
        for (i = 0; i < pts.length; i += 1) {
          var sc = Math.max(-8, Math.min(8, score(pts[i])));
          ctx.globalAlpha = 0.55;
          ctx.fillStyle = pts[i].y1 === 1 ? C.red : C.blue;
          ctx.fillRect(AX + 6 + (pts[i].y1 === 1 ? 26 : 0) + (i % 7) * 2.6, sy(sc) - 1.5, 3, 3);
          ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(AX - 16, sy(0)); ctx.lineTo(AX + 70, sy(0)); ctx.stroke();
        ctx.strokeStyle = C.gold; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(AX - 16, sy(Math.max(-8, Math.min(8, t)))); ctx.lineTo(AX + 70, sy(Math.max(-8, Math.min(8, t)))); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.gold;
        ctx.fillText("порог logit(p*) = " + t.toFixed(2), AX + 76, sy(Math.max(-8, Math.min(8, t))) + 4);
        ctx.fillStyle = C.ink;
        ctx.fillText("s = 0", AX + 76, sy(0) + 4);

        out.set([
          { label: "accuracy", value: (st.acc * 100).toFixed(1) + "%", color: C.ink },
          { label: "precision", value: (st.prec * 100).toFixed(0) + "%", color: C.red },
          { label: "recall", value: (st.rec * 100).toFixed(0) + "%", color: C.blue },
          { label: "средние потери (пропуск : ложная тревога = " + costFN + " : 1)", value: st.loss.toFixed(3), color: C.gold },
          { label: "в серой зоне 0,2 < p < 0,8", value: st.grey + " из " + pts.length, color: C.violet },
          { label: "выгодный порог c/(c+1)", value: (1 / (1 + costFN)).toFixed(3), color: C.green }
        ]);
      }

      // --------------------------------------------------------------- controls
      K.hint(root, "Четыре ручки — четыре разные операции. Угол вектора w поворачивает границу и меняет геометрию; длина |w| границу не двигает вовсе, она лишь делает вероятности резче; смещение b переносит границу параллельно; порог p* тоже переносит её параллельно, но уже без переобучения весов. Модельные данные, фиксированный генератор.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Чёрная прямая — множество s(x)=0, золотая пунктирная — линия, по которой на самом деле принимается решение при пороге p*. Точки в чёрной обводке классифицированы неверно. Справа те же объекты выложены по шкале score: порог — это горизонтальная черта на ней.");
      var cs = K.makeCanvas(stage, W, H, {
        label: "Направление весов, смещение и порог решения на плоскости признаков",
        onResize: function () { draw(); },
        drag: false
      });

      K.slider(controls, {
        label: "угол вектора w", min: 0, max: 360, step: 1, value: angle,
        unit: "°", format: function (v) { return v.toFixed(0); }
      }, function (v) { angle = v; draw(); });
      K.slider(controls, {
        label: "длина |w| (крутизна вероятностей)", min: 0.3, max: 8, step: 0.1, value: mag,
        format: function (v) { return v.toFixed(1); }
      }, function (v) { mag = v; draw(); });
      K.slider(controls, {
        label: "смещение b", min: -6, max: 6, step: 0.1, value: bias,
        format: function (v) { return v.toFixed(1); }
      }, function (v) { bias = v; draw(); });
      K.slider(controls, {
        label: "порог решения p*", min: 0.02, max: 0.98, step: 0.01, value: thr,
        format: function (v) { return v.toFixed(2); }
      }, function (v) { thr = v; draw(); });
      K.segmented(controls, {
        label: "цена пропуска против ложной тревоги",
        value: 10,
        options: [{ label: "1 : 1", value: 1 }, { label: "10 : 1", value: 10 }, { label: "50 : 1", value: 50 }]
      }, function (v) { costFN = Number(v); draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
