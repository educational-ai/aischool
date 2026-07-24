// Grade 10, module 1. Eleven purpose-built experiments share drawing helpers,
// but expose separate public names so every article has its own laboratory.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    var K = window.KonturInt;
    var C = K.COLORS;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function seeded(index, salt) {
      var x = Math.sin(index * 91.173 + salt * 17.731) * 43758.5453;
      return x - Math.floor(x);
    }

    function setup(root, hint, label, caption, height) {
      K.hint(root, hint);
      var stage = K.row(root);
      var canvasState;
      var redraw = function () {};
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, caption);
      canvasState = K.makeCanvas(stage, 920, height || 430, {
        maxWidth: 920,
        label: label,
        onResize: function () { redraw(); },
      });
      return {
        canvas: canvasState,
        controls: controls,
        output: output,
        setDraw: function (fn) { redraw = fn; },
        destroy: function () { canvasState.destroy(); },
      };
    }

    function axes(ctx, box, xLabel, yLabel, xTicks, yTicks) {
      ctx.strokeStyle = C.grid;
      ctx.lineWidth = 1;
      ctx.fillStyle = C.muted;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      xTicks.forEach(function (tick) {
        var px = box.x + tick.p * box.w;
        ctx.beginPath();
        ctx.moveTo(px, box.y);
        ctx.lineTo(px, box.y + box.h);
        ctx.stroke();
        ctx.fillText(tick.label, px, box.y + box.h + 22);
      });
      ctx.textAlign = "right";
      yTicks.forEach(function (tick) {
        var py = box.y + (1 - tick.p) * box.h;
        ctx.beginPath();
        ctx.moveTo(box.x, py);
        ctx.lineTo(box.x + box.w, py);
        ctx.stroke();
        ctx.fillText(tick.label, box.x - 10, py + 4);
      });
      ctx.strokeStyle = C.axis;
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = C.muted;
      ctx.textAlign = "center";
      ctx.fillText(xLabel, box.x + box.w / 2, box.y + box.h + 43);
      ctx.save();
      ctx.translate(20, box.y + box.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    function line(ctx, points, color, width, dash) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width || 2;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      points.forEach(function (point, index) {
        if (index) ctx.lineTo(point.x, point.y);
        else ctx.moveTo(point.x, point.y);
      });
      ctx.stroke();
      ctx.restore();
    }

    function buildModelResidual(root) {
      var ui = setup(
        root,
        "Меняй сопротивление воздуха, шум измерений и гибкость поправки. Серый участок лежит вне обучающего диапазона.",
        "Сравнение физической и гибридной модели полёта",
        "Синяя линия использует уравнение без сопротивления. Красная добавляет остаток, найденный по точкам слева от границы. Справа начинается проверка переноса.",
        440,
      );
      var state = { drag: 48, noise: 14, flex: 55 };
      var box = { x: 76, y: 42, w: 790, h: 310 };

      function truth(t) {
        return 0.93 - 0.48 * t * t + state.drag / 100 * 0.17 * t * t * t;
      }
      function physics(t) {
        return 0.93 - 0.48 * t * t;
      }
      function learned(t) {
        var residual = state.drag / 100 * 0.17 * t * t * t;
        var outside = Math.max(0, t - 0.86);
        var wobble = outside * outside * (state.flex - 45) / 100 * 0.72;
        return physics(t) + residual * state.flex / 65 + wobble;
      }
      function sx(t) { return box.x + t / 1.35 * box.w; }
      function sy(h) { return box.y + (1 - clamp(h, -0.02, 1.02) / 1.02) * box.h; }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 440);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 440);
        ctx.fillStyle = C.wash;
        ctx.fillRect(sx(0.86), box.y, sx(1.35) - sx(0.86), box.h);
        axes(
          ctx,
          box,
          "время, с",
          "нормированная высота",
          [0, 0.25, 0.5, 0.75, 1].map(function (p) {
            return { p: p, label: (p * 1.35).toFixed(1) };
          }),
          [0, 0.5, 1].map(function (p) { return { p: p, label: p.toFixed(1) }; }),
        );
        var physicalPoints = [];
        var learnedPoints = [];
        var truePoints = [];
        for (var j = 0; j <= 120; j += 1) {
          var t = j / 120 * 1.35;
          physicalPoints.push({ x: sx(t), y: sy(physics(t)) });
          learnedPoints.push({ x: sx(t), y: sy(learned(t)) });
          truePoints.push({ x: sx(t), y: sy(truth(t)) });
        }
        line(ctx, truePoints, C.green, 1.5, [5, 4]);
        line(ctx, physicalPoints, C.blue, 2.2);
        line(ctx, learnedPoints, C.red, 2.4);

        for (var i = 0; i < 17; i += 1) {
          var ti = i / 16 * 0.84;
          var noisy = truth(ti) + (seeded(i, 2) - 0.5) * state.noise / 100 * 0.16;
          ctx.beginPath();
          ctx.arc(sx(ti), sy(noisy), 3.2, 0, Math.PI * 2);
          ctx.fillStyle = C.ink;
          ctx.fill();
        }
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("обучение", sx(0.66), box.y + 20);
        ctx.fillText("новый режим", sx(0.9), box.y + 20);
        ctx.fillStyle = C.blue;
        ctx.fillText("физика", box.x + 14, box.y + box.h - 48);
        ctx.fillStyle = C.red;
        ctx.fillText("физика + остаток", box.x + 14, box.y + box.h - 30);
        ctx.fillStyle = C.green;
        ctx.fillText("скрытая траектория", box.x + 14, box.y + box.h - 12);

        var testT = 1.18;
        var physicalError = Math.abs(physics(testT) - truth(testT));
        var hybridError = Math.abs(learned(testT) - truth(testT));
        ui.output.set([
          { label: "ошибка физики вне диапазона", value: physicalError.toFixed(3), color: C.blue },
          { label: "ошибка гибрида", value: hybridError.toFixed(3), color: C.red },
          {
            label: "наблюдение",
            value: hybridError < physicalError ? "остаток помогает" : "экстраполяция сорвалась",
          },
        ]);
      }

      K.slider(ui.controls, {
        label: "Сопротивление воздуха",
        min: 0, max: 100, step: 1, value: state.drag, unit: "%",
      }, function (value) { state.drag = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум измерений",
        min: 0, max: 60, step: 1, value: state.noise, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      K.slider(ui.controls, {
        label: "Гибкость остатка",
        min: 0, max: 100, step: 1, value: state.flex, unit: "%",
      }, function (value) { state.flex = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildDataPipeline(root) {
      var ui = setup(
        root,
        "Полный поток показан серым. Цветные точки остаются в таблице после пропусков, дрейфа и отбора по времени суток.",
        "Почасовой ряд сенсора качества воздуха",
        "Среднее по видимым строкам может уходить от среднего полного потока без явного разрыва графика. Проверяй механизм появления строк, а не только диапазон столбца.",
        440,
      );
      var state = { missing: 18, drift: 12, day: 70 };
      var box = { x: 72, y: 40, w: 800, h: 300 };

      function raw(i) {
        var hour = i % 24;
        var rush = Math.exp(-Math.pow((hour - 8) / 2.6, 2))
          + 0.8 * Math.exp(-Math.pow((hour - 18) / 3.2, 2));
        return 42 + 24 * rush + 5 * Math.sin(i / 11) + (seeded(i, 7) - 0.5) * 7;
      }
      function visible(i) {
        var hour = i % 24;
        var dayFactor = hour >= 7 && hour <= 21 ? state.day / 100 : 1 - state.day / 100;
        var keepChance = 0.62 + 0.38 * dayFactor;
        var humidityGap = raw(i) > 70 ? state.missing / 100 * 1.5 : state.missing / 100;
        return seeded(i, 11) > humidityGap && seeded(i, 13) < keepChance;
      }
      function measured(i) {
        return raw(i) + state.drift / 100 * 22 * i / 95;
      }
      function sx(i) { return box.x + i / 95 * box.w; }
      function sy(value) { return box.y + (100 - value) / 80 * box.h; }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 440);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 440);
        axes(
          ctx,
          box,
          "четыре суток, часы",
          "показание сенсора",
          [0, 24, 48, 72, 95].map(function (v) {
            return { p: v / 95, label: String(v) };
          }),
          [20, 40, 60, 80, 100].map(function (v) {
            return { p: (v - 20) / 80, label: String(v) };
          }),
        );
        var all = [];
        var kept = [];
        var sumRaw = 0;
        var sumVisible = 0;
        for (var i = 0; i < 96; i += 1) {
          var r = raw(i);
          sumRaw += r;
          all.push({ x: sx(i), y: sy(r) });
          if (visible(i)) {
            var m = measured(i);
            sumVisible += m;
            kept.push({ i: i, x: sx(i), y: sy(m), value: m });
          }
        }
        line(ctx, all, C.axis, 1.1);
        kept.forEach(function (point) {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
          ctx.fillStyle = point.value > 86 ? C.red : C.blue;
          ctx.fill();
        });
        var fullMean = sumRaw / 96;
        var visibleMean = kept.length ? sumVisible / kept.length : 0;
        line(ctx, [{ x: box.x, y: sy(fullMean) }, { x: box.x + box.w, y: sy(fullMean) }], C.green, 1.5, [6, 4]);
        line(ctx, [{ x: box.x, y: sy(visibleMean) }, { x: box.x + box.w, y: sy(visibleMean) }], C.red, 1.8);
        ui.output.set([
          { label: "строк осталось", value: kept.length + " / 96" },
          { label: "среднее полного потока", value: fullMean.toFixed(1), color: C.green },
          { label: "среднее таблицы", value: visibleMean.toFixed(1), color: C.red },
          { label: "сдвиг", value: (visibleMean - fullMean).toFixed(1) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Пропуски при высоких значениях",
        min: 0, max: 55, step: 1, value: state.missing, unit: "%",
      }, function (value) { state.missing = value; draw(); });
      K.slider(ui.controls, {
        label: "Дрейф сенсора",
        min: -30, max: 30, step: 1, value: state.drift, unit: "%",
      }, function (value) { state.drift = value; draw(); });
      K.slider(ui.controls, {
        label: "Доля дневных наблюдений",
        min: 10, max: 90, step: 1, value: state.day, unit: "%",
      }, function (value) { state.day = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBlindJudge(root) {
      var ui = setup(
        root,
        "Прочитай три реплики и выбери предполагаемую стратегию. После выбора источник откроется.",
        "Слепая поведенческая проверка трёх собеседников",
        "Правильная догадка относится к этой короткой беседе. Она не доказывает устройство, сознание или надёжность собеседника в другой задаче.",
        410,
      );
      var rounds = [
        {
          question: "Почему в тени снег иногда синий?",
          replies: [
            ["Снег белый. В тени он выглядит темнее.", "template"],
            ["Прямой жёлтый свет перекрыт, а рассеянный свет неба остаётся голубым.", "human"],
            ["Цвет зависит от спектра освещения: в тени сильнее доля света от голубого неба.", "stat"],
          ],
        },
        {
          question: "Продолжи: 2, 3, 5, 8, 12, …",
          replies: [
            ["17. Разности равны 1, 2, 3, 4, затем 5.", "human"],
            ["17", "template"],
            ["Следующее число, вероятно, 17: к членам прибавляются 1, 2, 3, 4, 5.", "stat"],
          ],
        },
        {
          question: "Можно ли доверять ответу без источника?",
          replies: [
            ["Да. Ответ сформулирован уверенно.", "template"],
            ["Нет. Нужна независимая проверка утверждения.", "human"],
            ["Уверенность текста не измеряет истинность; источник надо проверить отдельно.", "stat"],
          ],
        },
      ];
      var state = { round: 0, revealed: null };
      var box = { x: 42, y: 34, w: 836, h: 330 };

      function wrap(ctx, text, x, y, width, lineHeight) {
        var words = text.split(/\s+/);
        var lineText = "";
        var lines = [];
        words.forEach(function (word) {
          var test = lineText ? lineText + " " + word : word;
          if (ctx.measureText(test).width > width && lineText) {
            lines.push(lineText);
            lineText = word;
          } else lineText = test;
        });
        if (lineText) lines.push(lineText);
        lines.slice(0, 3).forEach(function (value, index) {
          ctx.fillText(value, x, y + index * lineHeight);
        });
      }

      function draw() {
        var round = rounds[state.round];
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 410);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 410);
        ctx.fillStyle = C.ink;
        ctx.font = "600 17px ET Book, Palatino, Georgia, serif";
        ctx.fillText("Судья: " + round.question, box.x, box.y + 8);
        round.replies.forEach(function (reply, index) {
          var y = 78 + index * 98;
          ctx.fillStyle = index % 2 ? C.wash : "#faf9f4";
          ctx.fillRect(box.x, y, box.w, 78);
          ctx.strokeStyle = C.grid;
          ctx.strokeRect(box.x, y, box.w, 78);
          ctx.fillStyle = C.muted;
          ctx.font = "600 12px system-ui, sans-serif";
          ctx.fillText(String.fromCharCode(65 + index), box.x + 14, y + 23);
          ctx.fillStyle = C.ink;
          ctx.font = "15px ET Book, Palatino, Georgia, serif";
          wrap(ctx, reply[0], box.x + 42, y + 23, 620, 20);
          if (state.revealed != null) {
            var labels = { template: "жёсткий шаблон", stat: "статистическая модель", human: "человек" };
            var colors = { template: C.gold, stat: C.blue, human: C.green };
            ctx.fillStyle = colors[reply[1]];
            ctx.font = "600 12px system-ui, sans-serif";
            ctx.textAlign = "right";
            ctx.fillText(labels[reply[1]], box.x + box.w - 14, y + 43);
            ctx.textAlign = "left";
          }
        });
        ui.output.set([
          { label: "раунд", value: (state.round + 1) + " / " + rounds.length },
          {
            label: "статус",
            value: state.revealed == null ? "источники скрыты" : "стратегии открыты",
            color: state.revealed == null ? C.muted : C.green,
          },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Раунд",
        value: "0",
        options: rounds.map(function (_, index) {
          return { value: String(index), label: String(index + 1) };
        }),
      }, function (value) {
        state.round = Number(value);
        state.revealed = null;
        draw();
      });
      K.segmented(ui.controls, {
        label: "Открыть источники после своей догадки",
        value: "hidden",
        options: [
          { value: "hidden", label: "скрыто" },
          { value: "shown", label: "показать" },
        ],
      }, function (value) {
        state.revealed = value === "shown" ? true : null;
        draw();
      });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildThreeLayers(root) {
      var ui = setup(
        root,
        "Размер опроса сужает интервал. Ошибка прогноза и цена нехватки меняют решение о дополнительном автобусе.",
        "Три слоя задачи городского транспорта",
        "Статистическая оценка, прогноз и действие отвечают на разные вопросы. Последняя панель учитывает стоимость двух решений.",
        430,
      );
      var state = { sample: 400, forecast: 14, missCost: 80 };
      var panelY = 62;
      var panelH = 250;

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 430);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 430);
        var panels = [
          { x: 34, w: 260, title: "1 · оценка доли" },
          { x: 330, w: 260, title: "2 · прогноз спроса" },
          { x: 626, w: 260, title: "3 · решение" },
        ];
        panels.forEach(function (panel) {
          ctx.fillStyle = C.wash;
          ctx.fillRect(panel.x, panelY, panel.w, panelH);
          ctx.strokeStyle = C.grid;
          ctx.strokeRect(panel.x, panelY, panel.w, panelH);
          ctx.fillStyle = C.ink;
          ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
          ctx.fillText(panel.title, panel.x + 14, panelY + 24);
        });

        var p = 0.58;
        var se = Math.sqrt(p * (1 - p) / state.sample);
        var lo = p - 2 * se;
        var hi = p + 2 * se;
        var x0 = panels[0].x + 24;
        var x1 = panels[0].x + panels[0].w - 24;
        var xp = function (value) { return x0 + value * (x1 - x0); };
        ctx.strokeStyle = C.axis;
        ctx.beginPath();
        ctx.moveTo(x0, 190);
        ctx.lineTo(x1, 190);
        ctx.stroke();
        ctx.strokeStyle = C.blue;
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(xp(lo), 190);
        ctx.lineTo(xp(hi), 190);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(xp(p), 190, 7, 0, Math.PI * 2);
        ctx.fillStyle = C.blue;
        ctx.fill();
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("95% интервал", x0, 225);

        var center = 68;
        var sigma = state.forecast;
        var baseX = panels[1].x + 22;
        var baseY = 252;
        var scaleX = 2.1;
        var curve = [];
        for (var v = 20; v <= 115; v += 1) {
          var density = Math.exp(-0.5 * Math.pow((v - center) / sigma, 2));
          curve.push({ x: baseX + (v - 20) * scaleX, y: baseY - density * 112 });
        }
        line(ctx, curve, C.red, 2.5);
        ctx.fillStyle = C.muted;
        ctx.fillText("пассажиров", baseX, 282);

        var spareCost = 20;
        var threshold = spareCost / (spareCost + state.missCost);
        var probabilityOver = 0.5 * Math.exp(-Math.pow((78 - center) / (sigma * 1.45), 2));
        probabilityOver = clamp(probabilityOver + 0.18, 0.03, 0.92);
        var send = probabilityOver > threshold;
        ctx.fillStyle = send ? C.green : C.gold;
        ctx.beginPath();
        ctx.arc(panels[2].x + panels[2].w / 2, 182, 54, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = C.paper;
        ctx.textAlign = "center";
        ctx.font = "600 15px system-ui, sans-serif";
        ctx.fillText(send ? "добавить рейс" : "оставить план", panels[2].x + panels[2].w / 2, 187);
        ctx.textAlign = "left";
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("P(переполнение) = " + Math.round(probabilityOver * 100) + "%", panels[2].x + 24, 260);
        ctx.fillText("порог = " + Math.round(threshold * 100) + "%", panels[2].x + 24, 280);

        ui.output.set([
          { label: "интервал поддержки", value: Math.round(lo * 100) + "–" + Math.round(hi * 100) + "%", color: C.blue },
          { label: "разброс прогноза", value: "±" + state.forecast + " пассажиров", color: C.red },
          { label: "решение", value: send ? "добавить автобус" : "не добавлять", color: send ? C.green : C.gold },
        ]);
      }

      K.slider(ui.controls, {
        label: "Размер опроса",
        min: 40, max: 3000, step: 20, value: state.sample,
      }, function (value) { state.sample = value; draw(); });
      K.slider(ui.controls, {
        label: "Ошибка прогноза",
        min: 4, max: 30, step: 1, value: state.forecast, unit: " чел.",
      }, function (value) { state.forecast = value; draw(); });
      K.slider(ui.controls, {
        label: "Цена нехватки",
        min: 20, max: 180, step: 5, value: state.missCost,
      }, function (value) { state.missCost = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLearningCompass(root) {
      var ui = setup(
        root,
        "Выбери ситуацию и настрой доступность разметки с задержкой награды. Чем ярче луч, тем лучше режим использует такой сигнал.",
        "Компас режимов обучения",
        "Карта не выбирает конкретный алгоритм. Она помогает проверить, какой учебный сигнал существует и во сколько он обходится.",
        450,
      );
      var scenarios = {
        bike: { label: "прокат", labelNeed: 72, rewardNeed: 45, structure: 58 },
        plant: { label: "растения", labelNeed: 84, rewardNeed: 8, structure: 82 },
        robot: { label: "робот", labelNeed: 24, rewardNeed: 92, structure: 42 },
        text: { label: "текст", labelNeed: 38, rewardNeed: 12, structure: 96 },
      };
      var state = { scenario: "plant", labels: 24, delay: 18 };
      var modes = [
        { id: "class", label: "классификация", angle: -Math.PI / 2, color: C.blue },
        { id: "reg", label: "регрессия", angle: -Math.PI / 2 + Math.PI * 0.4, color: C.green },
        { id: "cluster", label: "без учителя", angle: -Math.PI / 2 + Math.PI * 0.8, color: C.violet },
        { id: "semi", label: "частичная разметка", angle: -Math.PI / 2 + Math.PI * 1.2, color: C.gold },
        { id: "rl", label: "подкрепление", angle: -Math.PI / 2 + Math.PI * 1.6, color: C.red },
      ];

      function scores() {
        var s = scenarios[state.scenario];
        var labelAmount = state.labels;
        var rewardQuality = 100 - state.delay;
        return {
          class: clamp(labelAmount * 0.85 + s.labelNeed * 0.15, 0, 100),
          reg: clamp(labelAmount * 0.75 + (state.scenario === "bike" ? 25 : 0), 0, 100),
          cluster: clamp(s.structure * 0.72 + (100 - labelAmount) * 0.28, 0, 100),
          semi: clamp(95 - Math.abs(labelAmount - 25) * 1.4 + s.structure * 0.12, 0, 100),
          rl: clamp(s.rewardNeed * 0.55 + rewardQuality * 0.45, 0, 100),
        };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 450);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 450);
        var center = { x: 460, y: 218 };
        var radius = 150;
        [25, 50, 75, 100].forEach(function (value) {
          ctx.beginPath();
          ctx.arc(center.x, center.y, radius * value / 100, 0, Math.PI * 2);
          ctx.strokeStyle = C.grid;
          ctx.stroke();
        });
        var current = scores();
        modes.forEach(function (mode) {
          var endX = center.x + Math.cos(mode.angle) * radius;
          var endY = center.y + Math.sin(mode.angle) * radius;
          ctx.beginPath();
          ctx.moveTo(center.x, center.y);
          ctx.lineTo(endX, endY);
          ctx.strokeStyle = C.grid;
          ctx.stroke();
          var score = current[mode.id];
          var pointX = center.x + Math.cos(mode.angle) * radius * score / 100;
          var pointY = center.y + Math.sin(mode.angle) * radius * score / 100;
          ctx.beginPath();
          ctx.moveTo(center.x, center.y);
          ctx.lineTo(pointX, pointY);
          ctx.strokeStyle = mode.color;
          ctx.lineWidth = 9;
          ctx.stroke();
          ctx.fillStyle = C.ink;
          ctx.font = "13px system-ui, sans-serif";
          ctx.textAlign = endX < center.x ? "right" : "left";
          ctx.fillText(mode.label, endX + (endX < center.x ? -12 : 12), endY + 4);
        });
        ctx.textAlign = "center";
        ctx.fillStyle = C.ink;
        ctx.font = "600 18px ET Book, Palatino, Georgia, serif";
        ctx.fillText(scenarios[state.scenario].label, center.x, center.y + 6);
        ctx.textAlign = "left";
        var best = modes.slice().sort(function (a, b) {
          return current[b.id] - current[a.id];
        })[0];
        ui.output.set([
          { label: "ситуация", value: scenarios[state.scenario].label },
          { label: "первый кандидат", value: best.label, color: best.color },
          { label: "пригодность", value: Math.round(current[best.id]) + "%" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Ситуация",
        value: state.scenario,
        options: Object.keys(scenarios).map(function (key) {
          return { value: key, label: scenarios[key].label };
        }),
      }, function (value) { state.scenario = value; draw(); });
      K.slider(ui.controls, {
        label: "Доля ручных меток",
        min: 0, max: 100, step: 1, value: state.labels, unit: "%",
      }, function (value) { state.labels = value; draw(); });
      K.slider(ui.controls, {
        label: "Задержка награды",
        min: 0, max: 100, step: 1, value: state.delay, unit: "%",
      }, function (value) { state.delay = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildClassification(root) {
      var ui = setup(
        root,
        "Двигай порог. Слева от него теряется часть спама, справа оказываются обычные письма. Доля спама меняет precision даже при тех же кривых.",
        "Распределения счёта спам-фильтра",
        "Синяя кривая описывает обычные письма, красная спам. Закрашенные хвосты образуют FP и FN. Порог выбирают по цене этих ошибок.",
        450,
      );
      var state = { threshold: 0.55, prevalence: 12, fnCost: 5 };
      var box = { x: 70, y: 42, w: 802, h: 285 };
      var meanNormal = 0.34;
      var meanSpam = 0.7;
      var sigmaNormal = 0.15;
      var sigmaSpam = 0.14;

      function density(x, mean, sigma) {
        return Math.exp(-0.5 * Math.pow((x - mean) / sigma, 2));
      }
      function erf(x) {
        var sign = x < 0 ? -1 : 1;
        var a = Math.abs(x);
        var t = 1 / (1 + 0.3275911 * a);
        var y = 1 - (((((1.061405429 * t - 1.453152027) * t)
          + 1.421413741) * t - 0.284496736) * t + 0.254829592)
          * t * Math.exp(-a * a);
        return sign * y;
      }
      function cdf(x, mean, sigma) {
        return 0.5 * (1 + erf((x - mean) / (sigma * Math.sqrt(2))));
      }
      function sx(value) { return box.x + value * box.w; }
      function sy(value) { return box.y + box.h - value * box.h * 0.92; }

      function fillTail(ctx, mean, sigma, from, to, color) {
        ctx.beginPath();
        ctx.moveTo(sx(from), sy(0));
        for (var i = 0; i <= 80; i += 1) {
          var x = from + (to - from) * i / 80;
          ctx.lineTo(sx(x), sy(density(x, mean, sigma)));
        }
        ctx.lineTo(sx(to), sy(0));
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.18;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 450);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 450);
        axes(
          ctx,
          box,
          "счёт модели",
          "относительная частота",
          [0, 0.25, 0.5, 0.75, 1].map(function (p) {
            return { p: p, label: p.toFixed(2) };
          }),
          [0, 0.5, 1].map(function (p) { return { p: p, label: p.toFixed(1) }; }),
        );
        fillTail(ctx, meanNormal, sigmaNormal, state.threshold, 1, C.blue);
        fillTail(ctx, meanSpam, sigmaSpam, 0, state.threshold, C.red);
        var normal = [];
        var spam = [];
        for (var i = 0; i <= 160; i += 1) {
          var x = i / 160;
          normal.push({ x: sx(x), y: sy(density(x, meanNormal, sigmaNormal)) });
          spam.push({ x: sx(x), y: sy(density(x, meanSpam, sigmaSpam)) });
        }
        line(ctx, normal, C.blue, 2.5);
        line(ctx, spam, C.red, 2.5);
        line(ctx, [
          { x: sx(state.threshold), y: box.y },
          { x: sx(state.threshold), y: box.y + box.h },
        ], C.ink, 1.5, [6, 4]);
        ctx.fillStyle = C.ink;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("порог " + state.threshold.toFixed(2), sx(state.threshold) + 8, box.y + 18);
        ctx.fillStyle = C.blue;
        ctx.fillText("обычные", sx(0.12), box.y + 42);
        ctx.fillStyle = C.red;
        ctx.fillText("спам", sx(0.79), box.y + 42);

        var tpr = 1 - cdf(state.threshold, meanSpam, sigmaSpam);
        var fpr = 1 - cdf(state.threshold, meanNormal, sigmaNormal);
        var positives = state.prevalence / 100 * 1000;
        var negatives = 1000 - positives;
        var tp = positives * tpr;
        var fn = positives - tp;
        var fp = negatives * fpr;
        var precision = tp + fp ? tp / (tp + fp) : 0;
        var cost = fp + fn * state.fnCost;
        ui.output.set([
          { label: "precision", value: Math.round(precision * 100) + "%", color: C.blue },
          { label: "recall", value: Math.round(tpr * 100) + "%", color: C.red },
          { label: "FP / FN на 1000", value: Math.round(fp) + " / " + Math.round(fn) },
          { label: "цена", value: Math.round(cost) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Порог решения",
        min: 0.05, max: 0.95, step: 0.01, value: state.threshold,
        format: function (value) { return value.toFixed(2); },
      }, function (value) { state.threshold = value; draw(); });
      K.slider(ui.controls, {
        label: "Доля спама",
        min: 1, max: 60, step: 1, value: state.prevalence, unit: "%",
      }, function (value) { state.prevalence = value; draw(); });
      K.slider(ui.controls, {
        label: "Цена пропуска спама",
        min: 1, max: 15, step: 1, value: state.fnCost, unit: "×",
      }, function (value) { state.fnCost = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildRegression(root) {
      var ui = setup(
        root,
        "Меняй шум и одно редкое опоздание. Верхняя панель показывает прогноз, нижняя остатки. Сравни MAE и RMSE.",
        "Регрессия времени прибытия и график остатков",
        "Квадратичная метрика сильнее реагирует на один большой промах. Форма облака остатков указывает на пропущенную зависимость.",
        470,
      );
      var state = { slope: 28, noise: 18, outlier: 22 };
      var top = { x: 70, y: 34, w: 802, h: 245 };
      var bottom = { x: 70, y: 334, w: 802, h: 84 };

      function points() {
        var values = [];
        for (var i = 0; i < 28; i += 1) {
          var x = 0.04 + i / 29 * 0.92;
          var wave = Math.sin(i * 1.7) * state.noise / 100 * 18;
          var jitter = (seeded(i, 22) - 0.5) * state.noise / 100 * 16;
          var y = 8 + state.slope * x + wave + jitter;
          if (i === 24) y += state.outlier;
          values.push({ x: x, y: y });
        }
        return values;
      }
      function fit(values) {
        var mx = values.reduce(function (s, p) { return s + p.x; }, 0) / values.length;
        var my = values.reduce(function (s, p) { return s + p.y; }, 0) / values.length;
        var num = 0;
        var den = 0;
        values.forEach(function (p) {
          num += (p.x - mx) * (p.y - my);
          den += Math.pow(p.x - mx, 2);
        });
        var b = den ? num / den : 0;
        return { a: my - b * mx, b: b };
      }
      function sx(x) { return top.x + x * top.w; }
      function sy(y) { return top.y + (62 - y) / 62 * top.h; }
      function ry(e) { return bottom.y + bottom.h / 2 - e / 34 * bottom.h / 2; }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 470);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 470);
        axes(
          ctx, top, "доля пройденного маршрута", "минут до прибытия",
          [0, 0.25, 0.5, 0.75, 1].map(function (p) { return { p: p, label: p.toFixed(2) }; }),
          [0, 20, 40, 60].map(function (v) { return { p: v / 62, label: String(v) }; }),
        );
        var values = points();
        var model = fit(values);
        line(ctx, [
          { x: sx(0), y: sy(model.a) },
          { x: sx(1), y: sy(model.a + model.b) },
        ], C.blue, 2.4);
        var abs = 0;
        var sq = 0;
        values.forEach(function (p, index) {
          var pred = model.a + model.b * p.x;
          var residual = p.y - pred;
          abs += Math.abs(residual);
          sq += residual * residual;
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), index === 24 ? 5.5 : 3.5, 0, Math.PI * 2);
          ctx.fillStyle = index === 24 ? C.red : C.ink;
          ctx.fill();
        });
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(bottom.x, bottom.y, bottom.w, bottom.h);
        line(ctx, [
          { x: bottom.x, y: ry(0) },
          { x: bottom.x + bottom.w, y: ry(0) },
        ], C.axis, 1);
        values.forEach(function (p, index) {
          var residual = p.y - (model.a + model.b * p.x);
          ctx.beginPath();
          ctx.arc(sx(p.x), ry(residual), index === 24 ? 5 : 3, 0, Math.PI * 2);
          ctx.fillStyle = residual > 0 ? C.red : C.blue;
          ctx.fill();
        });
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("остатки", bottom.x, bottom.y - 10);
        var mae = abs / values.length;
        var rmse = Math.sqrt(sq / values.length);
        ui.output.set([
          { label: "MAE", value: mae.toFixed(1) + " мин", color: C.blue },
          { label: "RMSE", value: rmse.toFixed(1) + " мин", color: C.red },
          { label: "наклон модели", value: model.b.toFixed(1) },
          { label: "редкий промах", value: state.outlier + " мин" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Связь с маршрутом",
        min: 8, max: 48, step: 1, value: state.slope,
      }, function (value) { state.slope = value; draw(); });
      K.slider(ui.controls, {
        label: "Обычный шум",
        min: 0, max: 60, step: 1, value: state.noise, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      K.slider(ui.controls, {
        label: "Редкое опоздание",
        min: 0, max: 45, step: 1, value: state.outlier, unit: " мин",
      }, function (value) { state.outlier = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildClustering(root) {
      var ui = setup(
        root,
        "Число центров и вес вертикального признака меняют группы. Тонкие линии связывают районы с назначенным центром.",
        "Кластеризация профилей городских районов",
        "Алгоритм оптимизирует расстояние в выбранном масштабе. Название и полезность группы появляются только после внешней проверки.",
        450,
      );
      var state = { k: 3, yWeight: 1, seed: 1 };
      var box = { x: 72, y: 38, w: 798, h: 320 };
      var palette = [C.blue, C.red, C.green, C.gold, C.violet];
      var data = [];
      [[0.22, 0.72], [0.5, 0.24], [0.76, 0.66], [0.42, 0.78]].forEach(function (center, group) {
        for (var i = 0; i < 13; i += 1) {
          data.push({
            x: clamp(center[0] + (seeded(i + group * 20, 31) - 0.5) * 0.22, 0.03, 0.97),
            y: clamp(center[1] + (seeded(i + group * 20, 37) - 0.5) * 0.22, 0.03, 0.97),
          });
        }
      });
      function sx(x) { return box.x + x * box.w; }
      function sy(y) { return box.y + (1 - y) * box.h; }
      function cluster() {
        var centers = [];
        for (var j = 0; j < state.k; j += 1) {
          var p = data[(j * 17 + state.seed * 7) % data.length];
          centers.push({ x: p.x, y: p.y });
        }
        var assignments = new Array(data.length).fill(0);
        for (var step = 0; step < 12; step += 1) {
          data.forEach(function (p, index) {
            var best = 0;
            var bestD = Infinity;
            centers.forEach(function (center, cIndex) {
              var d = Math.pow(p.x - center.x, 2)
                + state.yWeight * Math.pow(p.y - center.y, 2);
              if (d < bestD) { bestD = d; best = cIndex; }
            });
            assignments[index] = best;
          });
          centers.forEach(function (center, cIndex) {
            var members = data.filter(function (_, index) { return assignments[index] === cIndex; });
            if (!members.length) return;
            center.x = members.reduce(function (sum, p) { return sum + p.x; }, 0) / members.length;
            center.y = members.reduce(function (sum, p) { return sum + p.y; }, 0) / members.length;
          });
        }
        return { centers: centers, assignments: assignments };
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 450);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 450);
        axes(
          ctx, box, "доля утренних поездок", "вечерний ритм",
          [0, 0.25, 0.5, 0.75, 1].map(function (p) { return { p: p, label: p.toFixed(2) }; }),
          [0, 0.5, 1].map(function (p) { return { p: p, label: p.toFixed(1) }; }),
        );
        var result = cluster();
        var objective = 0;
        data.forEach(function (p, index) {
          var cIndex = result.assignments[index];
          var center = result.centers[cIndex];
          objective += Math.pow(p.x - center.x, 2)
            + state.yWeight * Math.pow(p.y - center.y, 2);
          ctx.strokeStyle = palette[cIndex] + "44";
          ctx.beginPath();
          ctx.moveTo(sx(p.x), sy(p.y));
          ctx.lineTo(sx(center.x), sy(center.y));
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 4, 0, Math.PI * 2);
          ctx.fillStyle = palette[cIndex];
          ctx.fill();
        });
        result.centers.forEach(function (center, index) {
          ctx.strokeStyle = C.paper;
          ctx.lineWidth = 6;
          ctx.beginPath();
          ctx.arc(sx(center.x), sy(center.y), 11, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = palette[index];
          ctx.beginPath();
          ctx.arc(sx(center.x), sy(center.y), 9, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = C.paper;
          ctx.font = "600 11px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(String(index + 1), sx(center.x), sy(center.y) + 4);
        });
        ctx.textAlign = "left";
        ui.output.set([
          { label: "кластеров", value: String(state.k) },
          { label: "целевая сумма", value: objective.toFixed(2) },
          { label: "вес вертикали", value: state.yWeight.toFixed(1) + "×" },
          { label: "предупреждение", value: "низкая сумма не даёт названий групп" },
        ]);
      }
      K.slider(ui.controls, {
        label: "Число кластеров k",
        min: 2, max: 5, step: 1, value: state.k,
      }, function (value) { state.k = value; draw(); });
      K.slider(ui.controls, {
        label: "Вес вечернего ритма",
        min: 0.1, max: 4, step: 0.1, value: state.yWeight,
        format: function (value) { return value.toFixed(1); },
      }, function (value) { state.yWeight = value; draw(); });
      K.slider(ui.controls, {
        label: "Начальные центры",
        min: 1, max: 8, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildFeedbackBudget(root) {
      var ui = setup(
        root,
        "Три ползунка задают относительные расходы. Доли нормируются до 100%. Сравни качество на знакомых и редких объектах.",
        "Распределение бюджета обратной связи",
        "Псевдометки быстро улучшают знакомую область, но наследуют ошибки модели. Случайное исследование помогает найти новые режимы.",
        430,
      );
      var state = { labels: 45, pseudo: 35, explore: 20 };
      var colors = [C.blue, C.gold, C.green];
      function shares() {
        var total = state.labels + state.pseudo + state.explore || 1;
        return {
          labels: state.labels / total,
          pseudo: state.pseudo / total,
          explore: state.explore / total,
        };
      }
      function drawBar(ctx, x, y, w, h, value, color, label) {
        ctx.fillStyle = C.wash;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = color;
        ctx.fillRect(x, y, w * clamp(value, 0, 1), h);
        ctx.fillStyle = C.ink;
        ctx.font = "600 14px system-ui, sans-serif";
        ctx.fillText(label, x, y - 10);
        ctx.textAlign = "right";
        ctx.fillText(Math.round(value * 100) + "%", x + w, y - 10);
        ctx.textAlign = "left";
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 430);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 430);
        var s = shares();
        var x = 70;
        var y = 72;
        var w = 780;
        ctx.fillStyle = C.ink;
        ctx.font = "600 16px ET Book, Palatino, Georgia, serif";
        ctx.fillText("100 единиц времени эксперта и среды", x, 38);
        var cursor = x;
        [
          { key: "labels", label: "ручные метки", color: colors[0] },
          { key: "pseudo", label: "псевдометки", color: colors[1] },
          { key: "explore", label: "исследование", color: colors[2] },
        ].forEach(function (item) {
          var width = w * s[item.key];
          ctx.fillStyle = item.color;
          ctx.fillRect(cursor, y, width, 54);
          if (width > 92) {
            ctx.fillStyle = C.paper;
            ctx.font = "600 12px system-ui, sans-serif";
            ctx.fillText(item.label, cursor + 10, y + 32);
          }
          cursor += width;
        });
        var familiar = clamp(0.36 + 0.55 * s.labels + 0.42 * s.pseudo + 0.12 * s.explore, 0, 0.98);
        var rare = clamp(0.18 + 0.34 * s.labels + 0.08 * s.pseudo + 0.72 * s.explore, 0, 0.96);
        var verified = clamp(0.22 + 0.68 * s.labels + 0.3 * s.explore - 0.18 * s.pseudo, 0, 0.98);
        drawBar(ctx, x, 190, w, 32, familiar, C.blue, "качество на знакомых объектах");
        drawBar(ctx, x, 272, w, 32, rare, C.green, "качество на редких объектах");
        drawBar(ctx, x, 354, w, 32, verified, C.red, "доля проверенных решений");
        ui.output.set([
          { label: "ручные метки", value: Math.round(s.labels * 100) + "%", color: C.blue },
          { label: "псевдометки", value: Math.round(s.pseudo * 100) + "%", color: C.gold },
          { label: "исследование", value: Math.round(s.explore * 100) + "%", color: C.green },
          { label: "слабое место", value: rare < verified ? "редкие случаи" : "проверка ответов" },
        ]);
      }
      K.slider(ui.controls, {
        label: "Вес ручной разметки", min: 0, max: 100, step: 1, value: state.labels,
      }, function (value) { state.labels = value; draw(); });
      K.slider(ui.controls, {
        label: "Вес псевдометок", min: 0, max: 100, step: 1, value: state.pseudo,
      }, function (value) { state.pseudo = value; draw(); });
      K.slider(ui.controls, {
        label: "Вес исследования", min: 0, max: 100, step: 1, value: state.explore,
      }, function (value) { state.explore = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildGalileo(root) {
      var ui = setup(
        root,
        "Число этажей задаёт диапазон высот. Задержка секундомера и сопротивление меняют оценку g разным образом.",
        "Опыт падения в координатах времени и квадрата времени",
        "Слева показан путь во времени. Справа прямая подгоняется к измеренным t². Систематическая задержка сдвигает начало, сопротивление искривляет зависимость.",
        460,
      );
      var state = { floors: 6, reaction: 0.06, drag: 12 };
      var left = { x: 58, y: 48, w: 365, h: 310 };
      var right = { x: 505, y: 48, w: 365, h: 310 };
      var g = 9.81;
      function observations() {
        var result = [];
        for (var i = 1; i <= state.floors; i += 1) {
          var h = i * 3;
          var ideal = Math.sqrt(2 * h / g);
          var slowed = ideal * (1 + state.drag / 100 * h / 24);
          var measured = slowed + state.reaction + (seeded(i, 51) - 0.5) * 0.035;
          result.push({ h: h, ideal: ideal, measured: measured });
        }
        return result;
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 460);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 460);
        var maxH = state.floors * 3;
        var obs = observations();
        var maxT = Math.max.apply(null, obs.map(function (o) { return o.measured; })) * 1.12;
        axes(
          ctx, left, "время, с", "путь, м",
          [0, 0.5, 1].map(function (p) { return { p: p, label: (p * maxT).toFixed(1) }; }),
          [0, 0.5, 1].map(function (p) { return { p: p, label: Math.round(p * maxH) }; }),
        );
        axes(
          ctx, right, "измеренное t², с²", "путь, м",
          [0, 0.5, 1].map(function (p) { return { p: p, label: (p * maxT * maxT).toFixed(1) }; }),
          [0, 0.5, 1].map(function (p) { return { p: p, label: Math.round(p * maxH) }; }),
        );
        function lx(t) { return left.x + t / maxT * left.w; }
        function ly(h) { return left.y + (1 - h / maxH) * left.h; }
        function rx(t2) { return right.x + t2 / (maxT * maxT) * right.w; }
        function ry(h) { return right.y + (1 - h / maxH) * right.h; }
        var curve = [];
        for (var j = 0; j <= 100; j += 1) {
          var t = j / 100 * maxT;
          curve.push({ x: lx(t), y: ly(Math.min(maxH, 0.5 * g * t * t)) });
        }
        line(ctx, curve, C.blue, 2);
        var numerator = 0;
        var denominator = 0;
        obs.forEach(function (o) {
          var u = o.measured * o.measured;
          numerator += u * o.h;
          denominator += u * u;
          ctx.fillStyle = C.red;
          ctx.beginPath();
          ctx.arc(lx(o.measured), ly(o.h), 4.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(rx(u), ry(o.h), 4.5, 0, Math.PI * 2);
          ctx.fill();
        });
        var slope = numerator / denominator;
        var gHat = 2 * slope;
        line(ctx, [
          { x: rx(0), y: ry(0) },
          { x: rx(maxT * maxT), y: ry(slope * maxT * maxT) },
        ], C.red, 2.3);
        ui.output.set([
          { label: "оценка g", value: gHat.toFixed(2) + " м/с²", color: C.red },
          { label: "справочное g", value: "9.81 м/с²", color: C.blue },
          { label: "смещение", value: (gHat - g).toFixed(2) },
          { label: "измерений", value: String(obs.length) },
        ]);
      }
      K.slider(ui.controls, {
        label: "Число этажей", min: 3, max: 10, step: 1, value: state.floors,
      }, function (value) { state.floors = value; draw(); });
      K.slider(ui.controls, {
        label: "Задержка секундомера", min: -0.08, max: 0.25, step: 0.01, value: state.reaction, unit: " с",
        format: function (value) { return value.toFixed(2); },
      }, function (value) { state.reaction = value; draw(); });
      K.slider(ui.controls, {
        label: "Сопротивление воздуха", min: 0, max: 45, step: 1, value: state.drag, unit: "%",
      }, function (value) { state.drag = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildDatasetPassport(root) {
      var ui = setup(
        root,
        "Заполни описание, снизь повторы и утечку, затем выбери тест. Индикатор оценивает перечисленные риски, а не «качество вообще».",
        "Инспектор паспорта датасета",
        "Способ разбиения должен воспроизводить будущую новизну. Документация помогает увидеть риск, но не исправляет строки сама.",
        460,
      );
      var state = { docs: 58, duplicates: 24, leakage: 18, split: "random" };
      var splitNames = { random: "случайные строки", group: "новые объекты", time: "будущее" };
      function drawCheck(ctx, y, label, value, goodWhenHigh) {
        var good = goodWhenHigh ? value >= 72 : value <= 8;
        ctx.beginPath();
        ctx.arc(78, y, 10, 0, Math.PI * 2);
        ctx.fillStyle = good ? C.green : (value < 35 ? C.gold : C.red);
        ctx.fill();
        ctx.fillStyle = C.paper;
        ctx.font = "600 12px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(good ? "✓" : "!", 78, y + 4);
        ctx.textAlign = "left";
        ctx.fillStyle = C.ink;
        ctx.font = "15px ET Book, Palatino, Georgia, serif";
        ctx.fillText(label, 100, y + 5);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(Math.round(value) + "%", 430, y + 4);
        ctx.textAlign = "left";
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 460);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 460);
        ctx.fillStyle = C.ink;
        ctx.font = "600 17px ET Book, Palatino, Georgia, serif";
        ctx.fillText("Что известно о наборе", 58, 44);
        drawCheck(ctx, 92, "источник, схема, лицензия", state.docs, true);
        drawCheck(ctx, 148, "повторы одного объекта", state.duplicates, false);
        drawCheck(ctx, 204, "признаки из будущего", state.leakage, false);
        var promiseMatch = state.split === "random"
          ? Math.max(0, 100 - state.duplicates * 1.7 - state.leakage)
          : state.split === "group"
            ? Math.max(0, 92 - state.leakage * 1.1)
            : Math.max(0, 95 - state.leakage * 1.8);
        drawCheck(ctx, 260, "тест повторяет обещание", promiseMatch, true);

        ctx.fillStyle = C.wash;
        ctx.fillRect(500, 62, 352, 266);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(500, 62, 352, 266);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("обещание теста", 522, 90);
        ctx.fillStyle = C.ink;
        ctx.font = "600 21px ET Book, Palatino, Georgia, serif";
        ctx.fillText(splitNames[state.split], 522, 124);
        var risk = clamp(
          100 - state.docs * 0.38 + state.duplicates * (state.split === "random" ? 0.5 : 0.12)
          + state.leakage * (state.split === "time" ? 0.55 : 0.35),
          0,
          100,
        );
        ctx.fillStyle = C.grid;
        ctx.fillRect(522, 190, 306, 28);
        ctx.fillStyle = risk < 30 ? C.green : (risk < 58 ? C.gold : C.red);
        ctx.fillRect(522, 190, 306 * risk / 100, 28);
        ctx.fillStyle = C.ink;
        ctx.font = "600 13px system-ui, sans-serif";
        ctx.fillText("остаточный риск: " + Math.round(risk) + "%", 522, 174);
        ctx.fillStyle = C.muted;
        ctx.font = "13px ET Book, Palatino, Georgia, serif";
        var warning = state.split === "random" && state.duplicates > 10
          ? "один объект попадёт в обе части"
          : state.split === "time" && state.leakage > 8
            ? "будущее проникло в признаки"
            : state.docs < 70
              ? "неизвестно происхождение части полей"
              : "перечисленные проверки закрыты";
        ctx.fillText(warning, 522, 258);
        ui.output.set([
          { label: "тест", value: splitNames[state.split] },
          { label: "остаточный риск", value: Math.round(risk) + "%", color: risk < 30 ? C.green : C.red },
          { label: "следующая проверка", value: warning },
        ]);
      }
      K.segmented(ui.controls, {
        label: "Способ разбиения",
        value: state.split,
        options: [
          { value: "random", label: "строки" },
          { value: "group", label: "объекты" },
          { value: "time", label: "время" },
        ],
      }, function (value) { state.split = value; draw(); });
      K.slider(ui.controls, {
        label: "Полнота паспорта", min: 0, max: 100, step: 1, value: state.docs, unit: "%",
      }, function (value) { state.docs = value; draw(); });
      K.slider(ui.controls, {
        label: "Повторы объектов", min: 0, max: 60, step: 1, value: state.duplicates, unit: "%",
      }, function (value) { state.duplicates = value; draw(); });
      K.slider(ui.controls, {
        label: "Утечка из будущего", min: 0, max: 50, step: 1, value: state.leakage, unit: "%",
      }, function (value) { state.leakage = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "model-residual-lab": buildModelResidual,
      "dataset-forensics": buildDataPipeline,
      "turing-jury": buildBlindJudge,
      "discipline-layers": buildThreeLayers,
      "learning-signals": buildLearningCompass,
      "classifier-studio": buildClassification,
      "regression-workbench": buildRegression,
      "clustering-lens": buildClustering,
      "label-budget-game": buildFeedbackBudget,
      "galileo-lab": buildGalileo,
      "dataset-passport-audit": buildDatasetPassport,
    };

    Object.keys(builders).forEach(function (name) {
      K.register(name, function (root) {
        return builders[name](root);
      });
    });

    // Compatibility for old article snapshots.
    K.register("g10-data", function (root, options) {
      var oldNames = {
        "02": "model-residual-lab",
        "03": "dataset-forensics",
        "04": "turing-jury",
        "05": "discipline-layers",
        "06": "learning-signals",
        "07": "classifier-studio",
        "08": "regression-workbench",
        "09": "clustering-lens",
        "10": "label-budget-game",
        "11": "galileo-lab",
        "12": "dataset-passport-audit",
      };
      var builder = builders[oldNames[String(options.lesson || "")]];
      return builder ? builder(root) : function () {};
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
