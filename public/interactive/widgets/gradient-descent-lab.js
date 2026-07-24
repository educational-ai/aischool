// Lesson 22: gradient-descent lab — contours, gradient arrow, descent path on real MSE.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("gradient-descent-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 540;

      var GAP = {"x":[-1.283,0.054,0.089,-0.104,0.622,1.355,1.391,1.248,-1.02,1.339,-0.994,-0.272,0.222,0.609,0.367,0.489,-1.119,-1.889,-0.865,-0.736,1.395,-1.522,-0.87,0.644,-0.079,0.177,-1.274,-2.213,-0.309,0.413,-0.942,0.721,0.358,1.051,1.373,-0.721,0.065,0.163,0.008,0.028,0.584,-1.593,-1.538,1.328,1.265,0.646,-1.474,1.305,-1.054,1.19,-0.046,-1.308,-1.668,-1.128,-0.327,1.461,0.875,1.392,-0.6,-0.328,0.55,-0.156,1.478,1.133,1.217,0.209,1.293,-0.148,-0.982,-0.919,1.068,1.59,0.473,-0.931,-1.916,0.578,-1.232,-1.468,0.602,-1.233,-0.828,0.508,0.574,-0.428,0.383,-0.272,-1.408,-1.307,-0.101,-1.199,1.404,1.124,-0.516,-1.618,-0.746,1.622,1.034,-0.555,0.426,-0.207,0.218,-0.405,0.759,0.972,0.928,0.244,0.498,-1.373,-0.917,1.012,-0.866,0.424,-1.374,1.588,0.902,1.141,-1.321,0.384,1.223,-0.244,-0.556,-0.149,1.343,1.418,-0.205,1.221,-1.189,0.223,-1.356,0.875,0.186,0.316,-1.223,1.328,1.519,0.484,0.538,-0.603,-0.445,-0.654,-1.086,-1.823],"y":[-1.927,0.783,0.44,-2.018,0.691,1.183,1.066,0.717,-0.245,1.034,-0.854,-0.121,0.652,-1.353,0.447,0.499,-1.223,-1.449,-0.605,-1.378,1.134,-1.851,-1.36,0.96,0.495,0.489,-0.154,-1.708,-0.971,0.979,-1.553,0.727,0.936,0.788,0.941,-1.015,0.435,0.664,0.36,0.405,-1.282,-0.745,-1.169,1.023,1.135,-0.854,-0.628,1.031,-0.581,1.037,0.27,-0.914,-1.714,-0.506,0.265,1.264,0.526,1.226,-0.192,0.303,0.329,-0.62,0.987,1.142,1.125,0.462,1.296,0.459,-1.072,0.024,0.966,0.879,0.414,-2.029,-1.773,0.577,-0.629,-1.555,0.601,-1.042,-0.236,0.482,0.764,-0.017,0.626,0.346,-2.072,-0.41,-1.172,-0.268,1.06,1.097,0.49,-0.843,-1.675,1.096,0.718,-0.127,0.709,0.394,0.367,0.389,0.711,0.922,0.976,0.784,0.455,-1.726,-0.123,0.48,-0.328,0.581,-2.031,1.078,0.636,0.908,-1.567,-1.469,1.158,0.448,-0.703,-2.277,1.153,1.221,0.593,0.947,-1.204,0.3,-0.714,0.234,0.575,0.396,-1.286,1.032,0.934,0.779,0.56,0.602,0.533,-0.358,-2.047,-1.955]}; // {x:[...], y:[...]} standardized

      var state = { eta: 0.3, start: [1.2, -0.8] };

      K.hint(
        root,
        "Кликните по карте линий уровня — из точки протянется стрелка градиента (поперёк линий, вверх) и путь спуска против неё ко дну. Меняйте скорость обучения: мало — ползёт, много — расходится.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Поверхность потерь — MSE прямой ŷ = w₀ + w₁·x на реальных данных Gapminder-2007 (log ВВП ↔ продолжительность жизни). Минимум в w₀=0, w₁=0,81. Градиент считается точной формулой; шаг спуска w ← w − η·∇L.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Карта линий уровня потерь, стрелка градиента и путь спуска",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 30, w: 640, h: 470 };
      var W0MIN = -1.0, W0MAX = 2.0, W1MIN = -1.3, W1MAX = 1.5;

      function px(w0) { return box.x + (w0 - W0MIN) / (W0MAX - W0MIN) * box.w; }
      function py(w1) { return box.y + box.h - (w1 - W1MIN) / (W1MAX - W1MIN) * box.h; }

      function loss(w0, w1) {
        var s = 0, n = GAP.x.length;
        for (var i = 0; i < n; i += 1) { var r = w0 + w1 * GAP.x[i] - GAP.y[i]; s += r * r; }
        return s / n;
      }
      function grad(w0, w1) {
        var g0 = 0, g1 = 0, n = GAP.x.length;
        for (var i = 0; i < n; i += 1) { var r = w0 + w1 * GAP.x[i] - GAP.y[i]; g0 += r; g1 += r * GAP.x[i]; }
        return [2 * g0 / n, 2 * g1 / n];
      }

      function descentPath(w0, w1) {
        var path = [[w0, w1]];
        for (var i = 0; i < 60; i += 1) {
          var g = grad(w0, w1);
          w0 -= state.eta * g[0]; w1 -= state.eta * g[1];
          path.push([w0, w1]);
          if (!isFinite(w0) || Math.abs(w0) > 100) break;
        }
        return path;
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        // contour fill (heat)
        var step = 6;
        for (var sx = box.x; sx < box.x + box.w; sx += step) {
          for (var sy = box.y; sy < box.y + box.h; sy += step) {
            var w0 = W0MIN + (sx - box.x) / box.w * (W0MAX - W0MIN);
            var w1 = W1MIN + (box.y + box.h - sy) / box.h * (W1MAX - W1MIN);
            var L = loss(w0, w1);
            var t = Math.min(1, L / 3.5);
            var g = Math.round(230 - t * 120);
            ctx.fillStyle = "rgb(" + g + "," + (g + 8) + "," + Math.round(g + 20) + ")";
            ctx.fillRect(sx, sy, step, step);
          }
        }
        // contour lines (iso-loss)
        ctx.strokeStyle = "rgba(49,95,140,0.35)";
        [0.5, 0.8, 1.2, 1.8, 2.5, 3.3].forEach(function (lv) {
          drawContour(ctx, lv);
        });
        ctx.strokeStyle = C.line; ctx.strokeRect(box.x, box.y, box.w, box.h);

        // minimum
        ctx.fillStyle = C.green;
        ctx.beginPath(); ctx.arc(px(0), py(0.809), 7, 0, Math.PI * 2); ctx.fill();
        ctx.textAlign = "left"; ctx.fillText("минимум", px(0) + 10, py(0.809));

        // descent path
        var path = descentPath(state.start[0], state.start[1]);
        ctx.strokeStyle = C.ink; ctx.lineWidth = 2.0;
        ctx.save(); ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
        ctx.beginPath();
        path.forEach(function (p, i) {
          if (i === 0) ctx.moveTo(px(p[0]), py(p[1])); else ctx.lineTo(px(p[0]), py(p[1]));
        });
        ctx.stroke();
        ctx.restore(); ctx.lineWidth = 1;
        // start point + gradient arrow (uphill)
        var s0 = state.start;
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(px(s0[0]), py(s0[1]), 6, 0, Math.PI * 2); ctx.fill();
        var g = grad(s0[0], s0[1]); var gn = Math.hypot(g[0], g[1]);
        if (gn > 1e-6) {
          var sc = 0.35 / gn;
          drawArrow(ctx, px(s0[0]), py(s0[1]), px(s0[0] + g[0] * sc), py(s0[1] + g[1] * sc), C.red);
        }
        ctx.fillStyle = C.red; ctx.textAlign = "left";
        ctx.fillText("клик = старт; красная стрелка — градиент (вверх)", box.x + 8, box.y + 16);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("сдвиг w₀", box.x + box.w / 2, box.y + box.h + 22);
        ctx.save(); ctx.translate(box.x - 44, box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("наклон w₁", 0, 0); ctx.restore();

        var end = path[path.length - 1];
        var finalLoss = loss(end[0], end[1]);
        var diverged = !isFinite(finalLoss) || finalLoss > 5;
        var rows = [
          { label: "Скорость обучения η", value: state.eta.toFixed(2).replace(".", ","), color: C.ink },
          { label: "Шагов до сходимости", value: diverged ? "не сходится" : String(path.length - 1), color: diverged ? C.red : C.ink },
          { label: "Итоговая потеря", value: diverged ? "взорвалась" : finalLoss.toFixed(3).replace(".", ","), color: diverged ? C.red : C.green },
        ];
        if (diverged) rows.push({ label: "Итог", value: "η слишком велика — спуск расходится", color: C.red });
        else if (Math.abs(end[1] - 0.809) < 0.02) rows.push({ label: "Итог", value: "дно найдено: наклон 0,81", color: C.green });
        output.set(rows);
      }

      function drawContour(ctx, lv) {
        // marching: sample grid, draw where loss≈lv (cheap: threshold band)
        var step = 4;
        ctx.lineWidth = 1.2;
        for (var sx = box.x; sx < box.x + box.w; sx += step) {
          for (var sy = box.y; sy < box.y + box.h; sy += step) {
            var w0 = W0MIN + (sx - box.x) / box.w * (W0MAX - W0MIN);
            var w1 = W1MIN + (box.y + box.h - sy) / box.h * (W1MAX - W1MIN);
            var L = loss(w0, w1);
            if (Math.abs(L - lv) < 0.04) {
              ctx.fillStyle = "rgba(49,95,140,0.4)";
              ctx.fillRect(sx, sy, 2, 2);
            }
          }
        }
      }

      function drawArrow(ctx, x0, y0, x1, y1, col) {
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        var ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
        ctx.lineWidth = 1;
      }

      canvasState.canvas.addEventListener("click", function (ev) {
        var rect = canvasState.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        var my = (ev.clientY - rect.top) / rect.height * H;
        if (mx < box.x || mx > box.x + box.w || my < box.y || my > box.y + box.h) return;
        var w0 = W0MIN + (mx - box.x) / box.w * (W0MAX - W0MIN);
        var w1 = W1MIN + (box.y + box.h - my) / box.h * (W1MAX - W1MIN);
        state.start = [w0, w1];
        draw();
      });

      K.slider(controls, { label: "Скорость обучения η", min: 0.02, max: 1.1, step: 0.02, value: state.eta,
        format: function (v) { return v.toFixed(2).replace(".", ","); } },
        function (v) { state.eta = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
