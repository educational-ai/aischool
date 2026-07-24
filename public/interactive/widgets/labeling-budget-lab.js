// Lesson 10: labelling budget lab — real curves from the lesson's experiments.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("labeling-budget-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;

      // Precomputed on SMS Spam (seed 7); see scripts/generate_lesson10_visuals.py.
      var DATA={"random": {"b": [25, 50, 100, 200, 300, 500], "f": [0.875, 0.895, 0.908, 0.915, 0.928, 0.94]}, "active": {"b": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500], "f": [0.875, 0.905, 0.916, 0.91, 0.915, 0.929, 0.933, 0.935, 0.942, 0.938, 0.942, 0.938, 0.936, 0.926, 0.93, 0.924, 0.923, 0.923, 0.923, 0.926]}, "mixed": {"b": [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500], "f": [0.875, 0.931, 0.932, 0.941, 0.936, 0.938, 0.938, 0.934, 0.938, 0.944, 0.94, 0.942, 0.94, 0.936, 0.928, 0.934, 0.93, 0.928, 0.924, 0.924]}, "bars": {"random100": 0.908, "active100": 0.916, "self100": 0.913, "weak": 0.672}};

      var state = { budget: 200, showMixed: "off" };

      K.hint(
        root,
        "Двигай бюджет по настоящим кривым эксперимента: разрыв стратегий, цена потолка и столбики бесплатных альтернатив при бюджете 100.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Кривые и столбики предвычислены настоящими прогонами урока на SMS-пуле (seed = 7): случайная и активная разметка, смесь 20+5, self-training на ста метках и слабая супервизия четырьмя правилами.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Кривые стратегий разметки и столбики бесплатных альтернатив",
        onResize: draw,
        drag: false,
      });

      var curveBox = { x: 74, y: 42, w: 600, h: 430 };
      var barBox = { x: 760, y: 42, w: 230, h: 430 };
      var F_MIN = 0.64, F_MAX = 0.96;

      function bx(b) { return curveBox.x + (b - 25) / (500 - 25) * curveBox.w; }
      function fy(f) {
        return curveBox.y + curveBox.h - (f - F_MIN) / (F_MAX - F_MIN) * curveBox.h;
      }

      function interp(series, budget) {
        var bs = series.b, fs = series.f;
        if (budget <= bs[0]) return fs[0];
        for (var i = 1; i < bs.length; i += 1) {
          if (budget <= bs[i]) {
            var t = (budget - bs[i - 1]) / (bs[i] - bs[i - 1]);
            return fs[i - 1] + t * (fs[i] - fs[i - 1]);
          }
        }
        return fs[fs.length - 1];
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        // Grid.
        [0.7, 0.8, 0.9, 0.94].forEach(function (f) {
          ctx.strokeStyle = f === 0.94 ? C.gold : C.grid;
          ctx.beginPath();
          ctx.moveTo(curveBox.x, fy(f));
          ctx.lineTo(curveBox.x + curveBox.w, fy(f));
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText(String(f).replace(".", ","), curveBox.x - 8, fy(f));
        });
        ctx.textAlign = "left";
        ctx.fillStyle = C.gold;
        ctx.fillText("потолок признаков", curveBox.x + 6, fy(0.94) - 11);

        function plot(series, color, width) {
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          ctx.beginPath();
          series.b.forEach(function (b, i) {
            if (i === 0) ctx.moveTo(bx(b), fy(series.f[i]));
            else ctx.lineTo(bx(b), fy(series.f[i]));
          });
          ctx.stroke();
        }
        plot(DATA.random, C.axis, 2);
        plot(DATA.active, C.blue, 2.2);
        if (state.showMixed === "on") plot(DATA.mixed, C.green, 2);

        // Budget cursor.
        ctx.strokeStyle = C.ink;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(bx(state.budget), curveBox.y);
        ctx.lineTo(bx(state.budget), curveBox.y + curveBox.h);
        ctx.stroke();
        ctx.setLineDash([]);
        [[DATA.random, C.axis], [DATA.active, C.blue]].concat(
          state.showMixed === "on" ? [[DATA.mixed, C.green]] : []
        ).forEach(function (pair) {
          var v = interp(pair[0], state.budget);
          ctx.fillStyle = pair[1];
          ctx.beginPath();
          ctx.arc(bx(state.budget), fy(v), 4.5, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        [100, 200, 300, 400, 500].forEach(function (b) {
          ctx.fillText(String(b), bx(b), curveBox.y + curveBox.h + 16);
        });
        ctx.fillText("бюджет меток", curveBox.x + curveBox.w / 2, curveBox.y + curveBox.h + 34);
        ctx.textAlign = "left";
        ctx.fillStyle = C.axis;
        ctx.fillText("случайные", bx(310), fy(interp(DATA.random, 310)) + 16);
        ctx.fillStyle = C.blue;
        ctx.fillText("активные", bx(150), fy(interp(DATA.active, 150)) - 14);

        // Bars at budget 100.
        ctx.fillStyle = C.ink;
        ctx.font = "bold 13px PT Sans, sans-serif";
        ctx.fillText("Бюджет 100 и «бесплатные» идеи", barBox.x - 24, barBox.y - 20);
        ctx.font = "12px PT Sans, sans-serif";
        var bars = [
          ["случайные", DATA.bars.random100, C.axis],
          ["активные", DATA.bars.active100, C.blue],
          ["+ псевдо", DATA.bars.self100, C.green],
          ["правила", DATA.bars.weak, C.red],
        ];
        var bw = barBox.w / 4;
        bars.forEach(function (bar, i) {
          var hgt = (bar[1] - F_MIN) / (F_MAX - F_MIN) * barBox.h;
          ctx.fillStyle = bar[2];
          ctx.globalAlpha = 0.85;
          ctx.fillRect(barBox.x + i * bw + 6, barBox.y + barBox.h - hgt, bw - 12, hgt);
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.ink;
          ctx.textAlign = "center";
          ctx.fillText(String(bar[1]).replace(".", ","), barBox.x + i * bw + bw / 2,
                       barBox.y + barBox.h - hgt - 12);
          ctx.save();
          ctx.translate(barBox.x + i * bw + bw / 2, barBox.y + barBox.h + 12);
          ctx.rotate(-0.5);
          ctx.fillStyle = C.muted;
          ctx.fillText(bar[0], 0, 0);
          ctx.restore();
          ctx.textAlign = "left";
        });

        var rows = [
          { label: "бюджет", value: String(state.budget) + " меток" },
          { label: "случайная стратегия", value: interp(DATA.random, state.budget).toFixed(3).replace(".", ",") },
          { label: "активная стратегия", value: interp(DATA.active, state.budget).toFixed(3).replace(".", ","), color: C.blue },
        ];
        if (state.showMixed === "on") {
          rows.push({ label: "смесь 20 + 5", value: interp(DATA.mixed, state.budget).toFixed(3).replace(".", ","), color: C.green });
        }
        output.set(rows);
      }

      K.slider(
        controls,
        { label: "Бюджет меток", min: 25, max: 500, step: 25, value: state.budget },
        function (value) { state.budget = value; draw(); },
      );
      K.segmented(
        controls,
        {
          label: "Смесь: 20 сомнительных + 5 случайных",
          value: state.showMixed,
          options: [
            { value: "off", label: "скрыть" },
            { value: "on", label: "показать" },
          ],
        },
        function (value) { state.showMixed = value; draw(); },
      );

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
