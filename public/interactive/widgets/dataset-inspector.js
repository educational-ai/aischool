// Lesson 12: dataset inspector — interrogate UCI Adult before any model.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("dataset-inspector", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 540;

      // Precomputed from scripts/data/adult.data (see generate_lesson12_visuals.py).
      // age: counts for 17..90; hours: 3-hour bins 1..99; cg: $4k bins for nonzero capital_gain.
      var HIST = {"age":[395,550,712,753,720,765,877,798,841,785,835,867,813,861,888,828,875,886,876,898,858,827,816,794,808,780,770,724,734,737,708,543,577,602,595,478,464,415,419,366,358,366,355,312,300,258,230,208,178,150,151,120,108,89,72,67,64,51,45,46,29,23,22,22,20,12,6,10,3,1,1,3,0,43],"hours":[91,178,189,462,461,309,1262,317,734,1242,310,1545,663,15472,2187,648,2861,204,808,1508,30,275,16,362,69,17,136,46,16,33,4,8,98],"cg":[816,1045,144,452,2,38,49,0,5,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,159,0]};
      var STATS = {
        rows: 32561, pos: 7841, share: 24.08, wshare: 23.86,
        q: { workclass: 5.64, occupation: 5.66, country: 1.79 },
        labelsRaw: [["«<=50K»", 24720], ["«>50K»", 7841], ["«<=50K.»", 12435], ["«>50K.»", 3846]],
        testShare: 23.62,
      };

      var state = { screen: "age", scale: "log", weights: "rows" };

      K.hint(
        root,
        "Экраны инспектора — проверки паспорта: три цензурных спайка на гистограммах, пропуски-двойняшки, четыре написания метки и доля дохода по строкам против весов.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Все числа посчитаны по adult.data и adult.test заранее тем же скриптом, что строит рисунки урока. Красное на гистограммах — значения, прижатые к границе диапазона: печать top-coding, а не факт о людях.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Инспектор датасета Adult: гистограммы, пропуски, метки и веса",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 34, w: 920, h: 420 };

      function drawBars(ctx, vals, opts) {
        var n = vals.length;
        var maxV = 0;
        vals.forEach(function (v) { if (v > maxV) maxV = v; });
        var logMax = Math.log(maxV + 1);
        var bw = box.w / n;
        vals.forEach(function (v, i) {
          var frac = state.scale === "log"
            ? Math.log(v + 1) / logMax
            : v / maxV;
          var h = frac * box.h;
          ctx.fillStyle = opts.red(i) ? C.red : C.blue;
          ctx.globalAlpha = 0.85;
          ctx.fillRect(box.x + i * bw + 1, box.y + box.h - h, Math.max(bw - 2, 1.5), h);
          ctx.globalAlpha = 1;
        });
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        opts.ticks.forEach(function (t) {
          ctx.fillText(t[1], box.x + (t[0] + 0.5) * bw, box.y + box.h + 18);
        });
        ctx.fillText(opts.xlabel, box.x + box.w / 2, box.y + box.h + 40);
        ctx.fillStyle = C.red;
        ctx.textAlign = "right";
        ctx.fillText(opts.note, box.x + box.w - 8, box.y + 16);
      }

      function drawShares(ctx, items, ylabel) {
        var maxV = 0;
        items.forEach(function (it) { if (it[1] > maxV) maxV = it[1]; });
        var n = items.length;
        var slot = box.w / n;
        items.forEach(function (it, i) {
          var h = it[1] / (maxV * 1.25) * box.h;
          var x = box.x + i * slot + slot * 0.22;
          ctx.fillStyle = it[2];
          ctx.globalAlpha = 0.85;
          ctx.fillRect(x, box.y + box.h - h, slot * 0.56, h);
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.ink;
          ctx.textAlign = "center";
          ctx.fillText(it[3], x + slot * 0.28, box.y + box.h - h - 14);
          ctx.fillStyle = C.muted;
          ctx.fillText(it[0], x + slot * 0.28, box.y + box.h + 18);
        });
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText(ylabel, box.x + box.w / 2, box.y + box.h + 40);
      }

      function fmtPct(v) { return String(v).replace(".", ",") + " %"; }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var rows = [];

        if (state.screen === "age") {
          drawBars(ctx, HIST.age, {
            red: function (i) { return i === HIST.age.length - 1; },
            ticks: [[3, "20"], [23, "40"], [43, "60"], [63, "80"], [73, "90"]],
            xlabel: "возраст, лет (по одному году)",
            note: "43 анкеты ровно на 90 годах; на 89 — ноль",
          });
          rows = [
            { label: "Анкет с возрастом 90", value: "43", color: C.red },
            { label: "Анкет с возрастом 89", value: "0", color: C.ink },
            { label: "Диагноз", value: "top-coding: «90» значит «90 и старше»", color: C.muted },
          ];
        } else if (state.screen === "hours") {
          drawBars(ctx, HIST.hours, {
            red: function (i) { return i === HIST.hours.length - 1; },
            ticks: [[0, "1–3"], [13, "40"], [22, "67"], [32, "97–99"]],
            xlabel: "часов в неделю (корзины по 3 часа)",
            note: "в последней корзине 85 анкет ровно с 99",
          });
          rows = [
            { label: "Пик распределения", value: "40 часов — 15 217 анкет", color: C.blue },
            { label: "Ровно 99 часов", value: "85 анкет", color: C.red },
            { label: "Диагноз", value: "99 — потолок кодировки, а не рабочая неделя", color: C.muted },
          ];
        } else if (state.screen === "cg") {
          drawBars(ctx, HIST.cg, {
            red: function (i) { return i === 24; },
            ticks: [[0, "0–4k"], [6, "24–28k"], [12, "48–52k"], [24, "96–100k"]],
            xlabel: "прирост капитала, $ (без нулей, корзины по 4 000)",
            note: "159 анкет ровно 99 999",
          });
          rows = [
            { label: "Анкет с нулевым приростом", value: "29 849 из 32 561", color: C.blue },
            { label: "Ровно 99 999", value: "159 анкет", color: C.red },
            { label: "Диагноз", value: "цензура сверху: известен только нижний предел", color: C.muted },
          ];
        } else if (state.screen === "missing") {
          drawShares(ctx, [
            ["workclass", STATS.q.workclass, C.red, fmtPct(STATS.q.workclass)],
            ["occupation", STATS.q.occupation, C.red, fmtPct(STATS.q.occupation)],
            ["country", STATS.q.country, C.gold, fmtPct(STATS.q.country)],
          ], "доля «?» по столбцам");
          rows = [
            { label: "Общие пропуски workclass ∩ occupation", value: "1836 анкет — все", color: C.red },
            { label: "Доля >50K среди «молчунов»", value: "10,4 % против 24,9 %", color: C.violet },
            { label: "Диагноз", value: "пропуск не случаен: это отказ от блока о работе", color: C.muted },
          ];
        } else if (state.screen === "labels") {
          drawShares(ctx, [
            [STATS.labelsRaw[0][0], STATS.labelsRaw[0][1], C.blue, "24 720"],
            [STATS.labelsRaw[1][0], STATS.labelsRaw[1][1], C.red, "7 841"],
            [STATS.labelsRaw[2][0], STATS.labelsRaw[2][1], C.violet, "12 435"],
            [STATS.labelsRaw[3][0], STATS.labelsRaw[3][1], C.gold, "3 846"],
          ], "значения метки в двух файлах до чистки");
          rows = [
            { label: "Классов до чистки", value: "4 (точки в adult.test)", color: C.red },
            { label: "После чистки", value: "2; доли 24,08 % и 23,62 %", color: C.green },
            { label: "Диагноз", value: "сначала один язык значений, потом любые сравнения", color: C.muted },
          ];
        } else {
          var byRows = state.weights === "rows";
          drawShares(ctx, [
            ["по строкам", STATS.share, C.blue, fmtPct(STATS.share)],
            ["по весам fnlwgt", STATS.wshare, C.green, fmtPct(STATS.wshare)],
          ], "доля >50K двумя способами");
          rows = [
            { label: "Выбранный способ", value: byRows ? "по строкам: 24,08 %" : "по весам: 23,86 %", color: byRows ? C.blue : C.green },
            { label: "Расхождение", value: "0,22 пункта — выборка сбалансирована", color: C.ink },
            { label: "Диагноз", value: "близость — свойство этого сбора, не закон природы", color: C.muted },
          ];
        }
        output.set(rows);
      }

      K.segmented(
        controls,
        {
          label: "Экран инспектора",
          value: state.screen,
          options: [
            { value: "age", label: "возраст" },
            { value: "hours", label: "часы" },
            { value: "cg", label: "капитал" },
            { value: "missing", label: "пропуски" },
            { value: "labels", label: "метки" },
            { value: "weights", label: "веса" },
          ],
        },
        function (value) { state.screen = value; draw(); },
      );
      K.segmented(
        controls,
        {
          label: "Шкала гистограмм",
          value: state.scale,
          options: [
            { value: "log", label: "логарифмическая" },
            { value: "lin", label: "линейная" },
          ],
        },
        function (value) { state.scale = value; draw(); },
      );
      K.segmented(
        controls,
        {
          label: "Доля >50K",
          value: state.weights,
          options: [
            { value: "rows", label: "по строкам" },
            { value: "w", label: "по весам" },
          ],
        },
        function (value) { state.weights = value; state.screen = "weights"; draw(); },
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
