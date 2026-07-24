// Lesson 16: hidden-layer lab — carve a region with hidden lines, bend space for XOR.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("hidden-layer-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;

      // Two-moons, from scripts/generate_lesson16_visuals.py (seed 0).
      var MOONS = [[-0.22,1.28,0],[-0.52,0.85,0],[-0.44,1.03,0],[-0.25,1.0,0],[0.14,1.08,0],[-0.35,0.86,0],[0.26,0.81,0],[-1.18,0.4,0],[-0.97,0.22,0],[0.74,1.08,0],[-0.94,0.79,0],[-0.3,0.92,0],[-0.22,1.25,0],[-1.09,0.1,0],[0.96,0.12,0],[1.14,0.1,0],[0.81,-0.01,0],[-0.94,0.81,0],[-0.61,0.66,0],[-1.11,0.53,0],[-1.16,-0.18,0],[-0.62,0.64,0],[0.27,1.04,0],[-0.63,0.53,0],[0.77,0.47,0],[-0.55,0.79,0],[0.83,0.44,0],[-1.04,-0.05,0],[-0.17,0.64,0],[0.36,0.71,0],[0.5,0.75,0],[-0.88,0.9,0],[-0.07,1.03,0],[-0.22,0.79,0],[1.08,0.03,0],[-0.24,1.06,0],[0.0,1.15,0],[-0.42,0.89,0],[-0.81,0.28,0],[-0.44,0.58,0],[0.42,0.79,0],[0.24,0.96,0],[-0.44,0.86,0],[1.11,0.11,0],[-0.65,0.8,0],[-0.51,0.92,0],[1.15,0.61,0],[0.77,0.34,0],[0.47,0.91,0],[0.17,0.92,0],[-0.19,1.01,0],[0.1,0.94,0],[-1.23,-0.04,0],[0.86,0.38,0],[0.61,0.74,0],[1.11,0.15,0],[-0.39,0.99,0],[0.6,0.65,0],[0.08,0.95,0],[0.67,0.43,0],[1.06,0.65,0],[0.81,0.11,0],[-0.39,0.79,0],[0.93,0.37,0],[0.93,0.69,0],[0.28,0.69,0],[-1.1,0.63,0],[0.76,0.22,0],[-0.97,0.48,0],[0.64,0.33,0],[-0.91,0.09,0],[0.05,1.01,0],[-0.93,-0.37,0],[-0.01,1.01,0],[-0.79,0.67,0],[1.07,0.1,0],[0.31,1.11,0],[0.91,0.53,0],[0.49,1.05,0],[0.98,0.46,0],[0.29,-0.15,1],[0.84,-0.26,1],[-0.08,0.22,1],[1.94,-0.49,1],[1.19,-0.3,1],[0.34,-0.15,1],[1.01,-0.44,1],[-0.17,0.47,1],[1.22,-0.58,1],[2.08,0.21,1],[0.25,-0.56,1],[1.61,-0.39,1],[0.06,0.27,1],[1.45,-0.39,1],[0.32,-0.27,1],[0.15,-0.09,1],[1.26,-0.48,1],[-0.11,0.31,1],[1.9,-0.15,1],[-0.19,0.44,1],[1.5,0.01,1],[0.23,-0.1,1],[1.79,-0.43,1],[2.12,0.19,1],[-0.14,-0.11,1],[0.96,-0.4,1],[1.18,-0.19,1],[1.4,-0.55,1],[0.13,-0.34,1],[1.92,0.31,1],[0.78,-0.46,1],[1.98,0.09,1],[1.46,-0.54,1],[0.62,-0.41,1],[1.73,-0.14,1],[0.39,-0.52,1],[1.85,0.23,1],[1.36,-0.47,1],[2.08,0.19,1],[1.57,-0.3,1],[1.62,-0.32,1],[0.96,-0.68,1],[2.04,0.2,1],[1.57,-0.44,1],[0.77,-0.39,1],[1.43,-0.7,1],[-0.03,0.58,1],[0.14,-0.25,1],[1.12,-0.54,1],[0.39,-0.56,1],[1.2,-0.67,1],[1.04,-0.45,1],[0.18,0.05,1],[0.35,-0.56,1],[1.17,-0.6,1],[1.42,-0.28,1],[1.47,-0.34,1],[1.37,-0.56,1],[1.58,-0.33,1],[0.5,-0.42,1],[2.08,0.19,1],[0.57,-0.54,1],[0.57,-0.35,1],[1.89,0.13,1],[2.1,0.04,1],[1.66,-0.28,1],[0.29,0.47,1],[2.12,0.34,1],[1.29,-0.26,1],[1.98,0.51,1],[0.26,-0.39,1],[1.82,0.14,1],[0.05,-0.22,1],[1.49,-0.39,1],[-0.08,0.17,1],[2.02,0.04,1],[1.95,-0.06,1],[1.15,-0.66,1],[0.7,-0.43,1],[0.16,0.27,1]];
      // XOR: label 1 on "different" corners.
      var XOR = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];

      var state = {
        mode: "moons",
        a1: 0, o1: -0.9,
        a2: 280, o2: 0.1,
        combine: "and",
      };

      K.hint(
        root,
        "Две скрытые прямые режут плоскость на полуплоскости; выходной нейрон берёт их И или ИЛИ, вырезая область. Соберите область вокруг одного класса — счётчик покажет, сколько точек разделено верно.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Каждая скрытая прямая — один нейрон: он делит плоскость надвое. Выход И оставляет пересечение полуплоскостей (клин/полоса), ИЛИ — объединение. Двух прямых хватает на XOR; «две луны» переплетены сильнее — увидите, что и композиции двух прямых мало, нужна глубина.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Плоскость с точками, двумя скрытыми прямыми и вырезанной областью",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 30, w: 940, h: 500 };

      function bounds() {
        if (state.mode === "xor") return { x0: -0.6, x1: 1.6, y0: -0.6, y1: 1.6 };
        return { x0: -1.6, x1: 2.5, y0: -1.1, y1: 1.7 };
      }
      function px(x) { var b = bounds(); return box.x + (x - b.x0) / (b.x1 - b.x0) * box.w; }
      function py(y) { var b = bounds(); return box.y + box.h - (y - b.y0) / (b.y1 - b.y0) * box.h; }

      function data() { return state.mode === "xor" ? XOR : MOONS; }

      // hidden line j: normal (cos a, sin a), point positive if n.x >= offset
      function side(x, y, ang, off) {
        var a = ang * Math.PI / 180;
        return (Math.cos(a) * x + Math.sin(a) * y) >= off ? 1 : 0;
      }
      // output: region where combination true -> predict class 1
      function predict(x, y) {
        var h1 = side(x, y, state.a1, state.o1);
        var h2 = side(x, y, state.a2, state.o2);
        return state.combine === "and" ? (h1 && h2 ? 1 : 0) : (h1 || h2 ? 1 : 0);
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var b = bounds();

        // shade carved region (class-1 prediction) via coarse grid
        var step = 7;
        for (var sx = box.x; sx < box.x + box.w; sx += step) {
          for (var sy = box.y; sy < box.y + box.h; sy += step) {
            var xd = b.x0 + (sx - box.x) / box.w * (b.x1 - b.x0);
            var yd = b.y0 + (box.y + box.h - sy) / box.h * (b.y1 - b.y0);
            if (predict(xd, yd)) {
              ctx.fillStyle = "rgba(49,95,140,0.09)";
              ctx.fillRect(sx, sy, step, step);
            }
          }
        }
        ctx.strokeStyle = C.line;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        // hidden lines
        [[state.a1, state.o1], [state.a2, state.o2]].forEach(function (ln) {
          var a = ln[0] * Math.PI / 180, off = ln[1];
          // line: cos a * x + sin a * y = off. Draw across bounds.
          var pts = [];
          [b.x0, b.x1].forEach(function (x) {
            if (Math.abs(Math.sin(a)) > 1e-6) {
              var y = (off - Math.cos(a) * x) / Math.sin(a);
              if (y >= b.y0 - 1e-6 && y <= b.y1 + 1e-6) pts.push([x, y]);
            }
          });
          [b.y0, b.y1].forEach(function (y) {
            if (Math.abs(Math.cos(a)) > 1e-6) {
              var x = (off - Math.sin(a) * y) / Math.cos(a);
              if (x >= b.x0 - 1e-6 && x <= b.x1 + 1e-6) pts.push([x, y]);
            }
          });
          if (pts.length >= 2) {
            ctx.strokeStyle = C.blue;
            ctx.lineWidth = 2.0;
            ctx.beginPath();
            ctx.moveTo(px(pts[0][0]), py(pts[0][1]));
            ctx.lineTo(px(pts[1][0]), py(pts[1][1]));
            ctx.stroke();
            ctx.lineWidth = 1;
          }
        });

        // points, colored by true class; ring if misclassified
        var D = data();
        var correct = 0;
        D.forEach(function (p) {
          var cls = p[2];
          var pr = predict(p[0], p[1]);
          if (pr === cls) correct += 1;
          ctx.beginPath();
          var r = state.mode === "xor" ? 9 : 4.5;
          if (cls === 1) ctx.arc(px(p[0]), py(p[1]), r, 0, Math.PI * 2);
          else ctx.rect(px(p[0]) - r, py(p[1]) - r, 2 * r, 2 * r);
          ctx.fillStyle = cls === 1 ? C.green : C.violet;
          ctx.globalAlpha = 0.9;
          ctx.fill();
          ctx.globalAlpha = 1;
          if (pr !== cls) {
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(px(p[0]), py(p[1]), r + 3.5, 0, Math.PI * 2);
            ctx.stroke();
            ctx.lineWidth = 1;
          }
        });

        ctx.textAlign = "left";
        ctx.fillStyle = C.green;
        ctx.fillText("● класс 1 (внутри области)", box.x + 12, box.y + 18);
        ctx.fillStyle = C.violet;
        ctx.fillText("■ класс 0 (снаружи)", box.x + 12, box.y + 40);
        ctx.fillStyle = C.blue;
        ctx.fillText("— скрытые прямые; заливка — область выхода", box.x + 12, box.y + 62);

        var total = D.length;
        var pct = Math.round(1000 * correct / total) / 10;
        var rows = [
          { label: "Верно разделено", value: correct + " из " + total + " (" + String(pct).replace(".", ",") + " %)", color: pct >= 99 ? C.green : C.ink },
        ];
        if (state.mode === "xor" && correct === 4) {
          rows.push({ label: "Итог", value: "XOR решён двумя прямыми", color: C.green });
        } else if (state.mode === "moons") {
          rows.push({ label: "Подсказка", value: "две прямые дают ~90%, последний остаток берёт глубина", color: C.muted });
        }
        output.set(rows);
      }

      var s1 = K.slider(controls, { label: "Прямая 1: угол", min: 0, max: 360, step: 1, value: state.a1,
        format: function (v) { return v + "°"; } }, function (v) { state.a1 = v; draw(); });
      var s2 = K.slider(controls, { label: "Прямая 1: сдвиг", min: -1.5, max: 2, step: 0.05, value: state.o1,
        format: function (v) { return v.toFixed(2).replace(".", ","); } }, function (v) { state.o1 = v; draw(); });
      var s3 = K.slider(controls, { label: "Прямая 2: угол", min: 0, max: 360, step: 1, value: state.a2,
        format: function (v) { return v + "°"; } }, function (v) { state.a2 = v; draw(); });
      var s4 = K.slider(controls, { label: "Прямая 2: сдвиг", min: -1.5, max: 2, step: 0.05, value: state.o2,
        format: function (v) { return v.toFixed(2).replace(".", ","); } }, function (v) { state.o2 = v; draw(); });
      K.segmented(controls, { label: "Выход комбинирует", value: state.combine,
        options: [{ value: "and", label: "И (пересечение)" }, { value: "or", label: "ИЛИ (объединение)" }] },
        function (v) { state.combine = v; draw(); });

      K.segmented(controls, { label: "Данные", value: state.mode,
        options: [{ value: "moons", label: "две луны" }, { value: "xor", label: "XOR" }] },
        function (v) {
          state.mode = v;
          if (v === "xor") { state.a1 = 45; state.o1 = 0.4; state.a2 = 225; state.o2 = -1.0; }
          else { state.a1 = 0; state.o1 = -0.9; state.a2 = 280; state.o2 = 0.1; }
          if (s1) { s1.set(state.a1); s2.set(state.o1); s3.set(state.a2); s4.set(state.o2); }
          draw();
        });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
