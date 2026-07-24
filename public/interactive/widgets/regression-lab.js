// Lesson 08: the model ladder, growth correction and quantile cushion on real data.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("regression-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 600;

      // Precomputed on the real Bike Sharing table (train 2011, test 2012)
      // by scripts/generate_lesson08_visuals.py's companion pipeline.
      var DATA = {"models": {"m0g0": {"mae": 168.3, "hist": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1256, 745, 440, 399, 381, 396, 393, 397, 386, 424, 358, 376, 306, 265, 221, 202, 171, 147, 153, 159, 159, 118, 112, 101, 87, 582]}, "m0g1": {"mae": 160.5, "hist": [0, 0, 0, 0, 0, 0, 0, 14, 255, 424, 452, 364, 341, 336, 513, 509, 431, 464, 447, 461, 452, 375, 342, 288, 248, 221, 202, 179, 164, 155, 115, 108, 107, 110, 104, 112, 80, 70, 53, 238]}, "m1g0": {"mae": 118.2, "hist": [0, 0, 0, 0, 0, 0, 0, 2, 7, 4, 23, 34, 37, 42, 104, 153, 159, 236, 349, 1366, 1093, 692, 679, 588, 500, 388, 312, 274, 214, 186, 179, 126, 138, 136, 138, 106, 91, 82, 82, 214]}, "m1g1": {"mae": 80.7, "hist": [12, 5, 6, 15, 25, 28, 36, 24, 44, 58, 77, 81, 104, 128, 202, 279, 509, 590, 953, 1656, 1085, 680, 436, 347, 242, 205, 213, 195, 159, 95, 86, 51, 49, 30, 14, 9, 2, 2, 2, 0]}, "m2g0": {"mae": 108.5, "hist": [0, 0, 0, 0, 0, 1, 2, 2, 1, 7, 12, 14, 13, 33, 64, 77, 117, 172, 264, 1076, 1613, 694, 614, 596, 545, 496, 443, 276, 274, 230, 207, 174, 151, 139, 97, 73, 78, 60, 51, 68]}, "m2g1": {"mae": 50.9, "hist": [12, 5, 12, 3, 6, 1, 11, 11, 26, 36, 48, 61, 77, 78, 110, 159, 268, 423, 743, 2322, 1815, 829, 600, 402, 277, 152, 87, 66, 41, 16, 8, 9, 10, 6, 1, 1, 1, 0, 1, 0]}, "m3g0": {"mae": 104.4, "hist": [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 3, 6, 10, 32, 44, 62, 100, 145, 277, 1100, 1671, 750, 629, 607, 560, 508, 402, 288, 260, 233, 210, 170, 149, 112, 99, 71, 74, 56, 50, 51]}, "m3g1": {"mae": 46.9, "hist": [3, 1, 0, 3, 2, 5, 8, 14, 16, 36, 40, 47, 53, 74, 96, 193, 284, 509, 888, 2491, 1718, 813, 521, 376, 194, 121, 76, 57, 30, 15, 15, 15, 9, 5, 2, 3, 0, 0, 1, 0]}}, "day": {"fact": [49, 21, 11, 7, 5, 48, 205, 557, 770, 328, 205, 232, 352, 323, 278, 318, 509, 925, 977, 635, 470, 306, 212, 127], "m0": [143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8, 143.8], "m1": [43.0, 26.6, 18.9, 10.1, 5.4, 14.6, 57.6, 156.8, 263.0, 163.9, 130.9, 155.9, 190.1, 190.5, 182.6, 188.4, 234.6, 349.7, 322.3, 236.4, 173.1, 134.6, 103.8, 69.2], "m2": [29.3, 13.7, 7.5, 4.4, 4.8, 17.8, 77.5, 214.7, 349.5, 180.6, 101.4, 117.7, 148.2, 146.5, 137.4, 146.7, 215.0, 394.9, 367.5, 259.0, 187.9, 143.6, 108.0, 68.6], "m3": [30.4, 14.2, 7.7, 4.6, 4.8, 18.5, 80.6, 228.2, 366.3, 187.1, 105.9, 123.2, 157.3, 155.7, 144.9, 155.8, 227.0, 411.3, 381.0, 268.7, 197.8, 149.4, 112.7, 71.7], "g": 2.025}, "qcov": {"0.5": {"cover": 0.579, "shortHours": 3674, "meanOver": 32.5}, "0.8": {"cover": 0.923, "shortHours": 676, "meanOver": 100.9}, "0.9": {"cover": 0.964, "shortHours": 316, "meanOver": 127.8}, "0.95": {"cover": 0.98, "shortHours": 178, "meanOver": 150.7}}};

      var MODEL_NAMES = ["константа", "+ час", "+ час и тип дня", "+ и погода"];
      var state = { model: 2, growth: "off", view: "point", q: 0.8 };

      K.hint(
        root,
        "Поднимись по лесенке моделей, затем включи поправку роста — и посмотри, как один плывущий коэффициент бьёт все признаки. Вкладка подушки показывает цену каждого уровня страховки.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Все числа предвычислены на настоящей таблице: обучение — 2011 год, проверка — 8734 часа 2012 года. Поправка роста — скользящее отношение факта к прогнозу за прошлые 28 дней; будущее в неё не подглядывает. Кривая дня — среда 12.09.2012.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Гистограмма остатков и день теста при выбранной модели",
        onResize: draw,
        drag: false,
      });

      var histBox = { x: 70, y: 46, w: 430, h: 250 };
      var dayBox = { x: 570, y: 46, w: 410, h: 250 };
      var ladderBox = { x: 70, y: 380, w: 910, h: 170 };

      function key() { return "m" + state.model + "g" + (state.growth === "on" ? 1 : 0); }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var entry = DATA.models[key()];

        // Residual histogram.
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Остатки на 2012 годе", histBox.x, histBox.y - 22);
        ctx.font = "13px PT Sans, sans-serif";
        var maxCount = Math.max.apply(null, entry.hist);
        var bw = histBox.w / entry.hist.length;
        entry.hist.forEach(function (n, i) {
          var hgt = n / maxCount * (histBox.h - 12);
          var center = -500 + 25 * i + 12.5;
          ctx.fillStyle = center < -12 ? C.green : center > 12 ? C.red : C.axis;
          ctx.globalAlpha = 0.8;
          ctx.fillRect(histBox.x + i * bw, histBox.y + histBox.h - hgt, bw - 1, hgt);
          ctx.globalAlpha = 1;
        });
        var zeroX = histBox.x + (0 + 500) / 1000 * histBox.w;
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(zeroX, histBox.y);
        ctx.lineTo(zeroX, histBox.y + histBox.h);
        ctx.stroke();
        ctx.fillStyle = C.muted;
        ctx.fillText("перелёты", histBox.x + 8, histBox.y + 12);
        ctx.textAlign = "right";
        ctx.fillStyle = C.red;
        ctx.fillText("недолёты", histBox.x + histBox.w - 8, histBox.y + 12);
        ctx.textAlign = "left";

        // Day curve.
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.fillText("Среда 12.09.2012", dayBox.x, dayBox.y - 22);
        ctx.font = "13px PT Sans, sans-serif";
        var facts = DATA.day.fact;
        var preds = DATA.day["m" + state.model].map(function (v) {
          return state.growth === "on" ? v * DATA.day.g : v;
        });
        var maxY = 1050;
        function dx(h) { return dayBox.x + h / 23 * dayBox.w; }
        function dy(v) { return dayBox.y + dayBox.h - v / maxY * dayBox.h; }
        ctx.strokeStyle = C.blue;
        ctx.lineWidth = 2;
        ctx.beginPath();
        preds.forEach(function (v, h) {
          if (h === 0) ctx.moveTo(dx(h), dy(v));
          else ctx.lineTo(dx(h), dy(v));
        });
        ctx.stroke();
        ctx.fillStyle = C.red;
        facts.forEach(function (v, h) {
          ctx.beginPath();
          ctx.arc(dx(h), dy(v), 3, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.fillStyle = C.muted;
        ctx.fillText("точки — факт, линия — прогноз", dayBox.x, dayBox.y + dayBox.h + 20);

        // Ladder.
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.fillText("MAE лесенки (тёмное — с поправкой роста)", ladderBox.x, ladderBox.y - 20);
        ctx.font = "12px PT Sans, sans-serif";
        var slot = ladderBox.w / 4;
        for (var m = 0; m < 4; m += 1) {
          ["0", "1"].forEach(function (gflag, gi) {
            var mae = DATA.models["m" + m + "g" + gflag].mae;
            var hgt = mae / 180 * (ladderBox.h - 26);
            var x = ladderBox.x + m * slot + 24 + gi * (slot / 2 - 30);
            var isCurrent = m === state.model && (state.growth === "on" ? 1 : 0) === Number(gflag);
            ctx.fillStyle = gflag === "1" ? C.blue : C.axis;
            ctx.globalAlpha = isCurrent ? 1 : 0.45;
            ctx.fillRect(x, ladderBox.y + ladderBox.h - 18 - hgt, slot / 2 - 36, hgt);
            ctx.globalAlpha = 1;
            if (isCurrent) {
              ctx.strokeStyle = C.ink;
              ctx.lineWidth = 1.6;
              ctx.strokeRect(x - 1.5, ladderBox.y + ladderBox.h - 19.5 - hgt, slot / 2 - 33, hgt + 3);
            }
            ctx.fillStyle = C.ink;
            ctx.textAlign = "center";
            ctx.fillText(String(mae).replace(".", ","), x + slot / 4 - 18, ladderBox.y + ladderBox.h - 26 - hgt);
            ctx.textAlign = "left";
          });
          ctx.fillStyle = C.muted;
          ctx.textAlign = "center";
          ctx.fillText(MODEL_NAMES[m], ladderBox.x + m * slot + slot / 2, ladderBox.y + ladderBox.h - 2);
          ctx.textAlign = "left";
        }

        var rows = [
          { label: "модель", value: MODEL_NAMES[state.model] + (state.growth === "on" ? " + рост" : "") },
          { label: "MAE на 2012", value: String(entry.mae).replace(".", ",") + " вел./час", color: entry.mae < 60 ? C.green : C.ink },
        ];
        if (state.view === "cushion") {
          var qc = DATA.qcov[String(state.q)];
          rows.push({ label: "покрытие подушки q=" + String(state.q).replace(".", ","), value: (qc.cover * 100).toFixed(1).replace(".", ",") + "%" });
          rows.push({ label: "часов с нехваткой за год", value: String(qc.shortHours), color: qc.shortHours > 1000 ? C.red : C.green });
          rows.push({ label: "средний излишек", value: String(qc.meanOver).replace(".", ",") + " вел./час", color: C.gold });
        }
        output.set(rows);
      }

      K.segmented(
        controls,
        {
          label: "Модель",
          value: String(state.model),
          options: MODEL_NAMES.map(function (n, i) { return { value: String(i), label: n }; }),
        },
        function (value) { state.model = Number(value); draw(); },
      );
      K.segmented(
        controls,
        {
          label: "Поправка роста (окно 28 дней)",
          value: state.growth,
          options: [
            { value: "off", label: "выключена" },
            { value: "on", label: "включена" },
          ],
        },
        function (value) { state.growth = value; draw(); },
      );
      K.segmented(
        controls,
        {
          label: "Подушка запаса",
          value: state.view,
          options: [
            { value: "point", label: "прогноз-точка" },
            { value: "cushion", label: "квантильная подушка" },
          ],
        },
        function (value) { state.view = value; draw(); },
      );
      K.slider(
        controls,
        {
          label: "Уровень подушки q",
          min: 0.5, max: 0.95, step: 0.05, value: state.q,
          format: function (v) { return v.toFixed(2).replace(".", ","); },
        },
        function (value) {
          var allowed = [0.5, 0.8, 0.9, 0.95];
          var best = allowed[0];
          allowed.forEach(function (a) { if (Math.abs(a - value) < Math.abs(best - value)) best = a; });
          state.q = best;
          draw();
        },
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
