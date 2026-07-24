// Lesson 01: a counterfactual simulator for a model embedded in a feedback loop.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("system-feedback-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;
      var SIMULATION_SEED = 2026;
      var NEXT_HOUR_SHARE = 0.48;
      var state = {
        mode: options.mode || "advice",
        drift: Number(options.drift || 55),
        response: Number(options.response || 65),
        threshold: Number(options.threshold || 66),
      };

      K.hint(
        root,
        "Сравни серую контрфактическую нагрузку с тем, что счётчик увидит после решения. Модель не меняется — меняется мир вокруг неё.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Старая тестовая ошибка вычислена до запуска. После запуска чистый перенос сохраняет энергию: 48% снятой нагрузки возвращается через час, остальные 52% — ещё через час. Данные синтетические, seed = 2026.",
      );

      var canvasState;
      var chart = { x: 70, y: 38, w: 920, h: 306 };
      var systemY = 428;
      var nodeW = 176;
      var nodeH = 72;
      var nodeXs = [28, 296, 564, 832];

      function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }

      function seeded(i, salt) {
        var x = Math.sin((i + SIMULATION_SEED) * 19.713 + salt * 73.191) * 43758.5453;
        return x - Math.floor(x);
      }

      function latentLoad(i) {
        var hour = i % 24;
        var schoolDay = i < 72;
        var morning = Math.exp(-Math.pow((hour - 9) / 2.25, 2));
        var evening = Math.exp(-Math.pow((hour - 17) / 2.7, 2));
        var oldPattern = 46 + 26 * morning + 18 * evening;
        var changedMorning = Math.exp(-Math.pow((hour - 11.5) / 2.4, 2));
        var changedEvening = Math.exp(-Math.pow((hour - 19) / 2.2, 2));
        var newPattern = 45 + 20 * changedMorning + 31 * changedEvening;
        var mix = i < 48 ? 0 : state.drift / 100;
        var schedule = oldPattern * (1 - mix) + newPattern * mix;
        var weekendRelief = schoolDay ? 0 : -7;
        return schedule + weekendRelief + (seeded(i, 5) - 0.5) * 3.5;
      }

      function forecast(i) {
        var hour = i % 24;
        var morning = Math.exp(-Math.pow((hour - 9) / 2.25, 2));
        var evening = Math.exp(-Math.pow((hour - 17) / 2.7, 2));
        return 46 + 25 * morning + 17 * evening;
      }

      function autonomyShare() {
        if (state.mode === "forecast") return 0;
        if (state.mode === "advice") return 0.45;
        return 1;
      }

      function simulate() {
        var raw = [];
        var predicted = [];
        var observed = [];
        var actions = [];
        var incoming = new Array(98).fill(0);
        var share = autonomyShare() * state.response / 100;
        for (var i = 0; i < 96; i += 1) {
          var demand = latentLoad(i);
          var guess = forecast(i);
          var recommended = Math.max(0, guess - state.threshold) * 0.85;
          var action = i < 48 ? 0 : recommended * share;
          var actual = demand + incoming[i] - action;
          incoming[i + 1] += action * NEXT_HOUR_SHARE;
          incoming[i + 2] += action * (1 - NEXT_HOUR_SHARE);
          raw.push(demand);
          predicted.push(guess);
          observed.push(actual);
          actions.push(action);
        }
        return {
          raw: raw,
          predicted: predicted,
          observed: observed,
          actions: actions,
          pending: incoming[96] + incoming[97],
        };
      }

      function sx(i) {
        return chart.x + i / 95 * chart.w;
      }

      function sy(value) {
        return chart.y + chart.h - (value - 28) / 62 * chart.h;
      }

      function path(ctx, values, color, width, dash) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        values.forEach(function (value, i) {
          var y = sy(clamp(value, 28, 90));
          if (i === 0) ctx.moveTo(sx(i), y);
          else ctx.lineTo(sx(i), y);
        });
        ctx.stroke();
        ctx.restore();
      }

      function line(ctx, x1, y1, x2, y2, color, width) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = width || 1;
        ctx.stroke();
      }

      function arrow(ctx, x1, y1, x2, y2, color) {
        var angle = Math.atan2(y2 - y1, x2 - x1);
        var head = 8;
        line(ctx, x1, y1, x2, y2, color, 1.5);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(
          x2 - head * Math.cos(angle - Math.PI / 6),
          y2 - head * Math.sin(angle - Math.PI / 6),
        );
        ctx.lineTo(
          x2 - head * Math.cos(angle + Math.PI / 6),
          y2 - head * Math.sin(angle + Math.PI / 6),
        );
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }

      function roundedBox(ctx, x, y, w, h, fill, stroke) {
        var r = 13;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }

      function labelLine(ctx, x, y, color, text, dash) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 24, y);
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = C.ink;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(text, x + 32, y + 4);
      }

      function drawSystem(ctx, result) {
        var fills = ["#edf3f7", "#f7ece9", "#f7f1df", "#edf4ef"];
        var edges = [C.blue, C.red, C.gold, C.green];
        var titles = ["Счётчик", "Модель", "Решение", "Школа после решения"];
        var details = [
          "видит изменённую нагрузку",
          "обучена на старом расписании",
          state.mode === "forecast"
            ? "только показывает прогноз"
            : state.mode === "advice"
              ? "45% советов доходят до действия"
              : "действует автоматически",
          "48% через час, 52% через два",
        ];

        nodeXs.forEach(function (x, index) {
          roundedBox(ctx, x, systemY, nodeW, nodeH, fills[index], edges[index]);
          ctx.fillStyle = index === 1 ? C.red : C.ink;
          ctx.font = "600 14px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(titles[index], x + nodeW / 2, systemY + 28);
          ctx.fillStyle = C.muted;
          ctx.font = "11px system-ui, sans-serif";
          ctx.fillText(details[index], x + nodeW / 2, systemY + 51);
          if (index < nodeXs.length - 1) {
            arrow(
              ctx,
              x + nodeW + 4,
              systemY + nodeH / 2,
              nodeXs[index + 1] - 4,
              systemY + nodeH / 2,
              C.axis,
            );
          }
        });

        var y = systemY + nodeH + 28;
        var startX = nodeXs[3] + nodeW / 2;
        var endX = nodeXs[0] + nodeW / 2;
        ctx.strokeStyle = C.green;
        ctx.lineWidth = 1.55;
        ctx.beginPath();
        ctx.moveTo(startX, systemY + nodeH);
        ctx.bezierCurveTo(startX, y + 34, endX, y + 34, endX, systemY + nodeH);
        ctx.stroke();
        var angle = -Math.PI / 2;
        var tipX = endX;
        var tipY = systemY + nodeH;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - 8 * Math.cos(angle - Math.PI / 6), tipY - 8 * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(tipX - 8 * Math.cos(angle + Math.PI / 6), tipY - 8 * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fillStyle = C.green;
        ctx.fill();
        ctx.fillStyle = C.green;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          result.actions.some(function (value) { return value > 0.05; })
            ? "следующее наблюдение уже содержит след решения"
            : "обратная связь есть, но решение пока не меняет нагрузку",
          (startX + endX) / 2,
          systemY + nodeH + 18,
        );
      }

      function mean(values) {
        return values.reduce(function (sum, value) { return sum + value; }, 0) / values.length;
      }

      function draw() {
        var result = simulate();
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = "#f8efeb";
        ctx.fillRect(sx(48), chart.y, sx(95) - sx(48), chart.h);
        [30, 45, 60, 75, 90].forEach(function (value) {
          line(ctx, chart.x, sy(value), chart.x + chart.w, sy(value), C.grid, 1);
          ctx.fillStyle = C.muted;
          ctx.font = "11px system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(String(value), chart.x - 10, sy(value) + 4);
        });
        [0, 24, 48, 72, 95].forEach(function (value) {
          line(ctx, sx(value), chart.y, sx(value), chart.y + chart.h, C.grid, 1);
          ctx.fillStyle = C.muted;
          ctx.textAlign = "center";
          ctx.fillText(value === 95 ? "96" : String(value), sx(value), chart.y + chart.h + 22);
        });
        line(ctx, chart.x, chart.y + chart.h, chart.x + chart.w, chart.y + chart.h, C.axis, 1.1);
        line(ctx, chart.x, chart.y, chart.x, chart.y + chart.h, C.axis, 1.1);

        result.actions.forEach(function (value, i) {
          if (value < 0.08) return;
          var barW = Math.max(2, chart.w / 96 - 2);
          var barH = Math.min(44, value * 5.2);
          ctx.fillStyle = "rgba(165, 121, 32, 0.38)";
          ctx.fillRect(sx(i) - barW / 2, chart.y + chart.h - barH, barW, barH);
        });

        path(ctx, result.raw, C.axis, 1.6, [6, 5]);
        path(ctx, result.predicted, C.red, 2.25);
        path(ctx, result.observed, C.blue, 2.45);

        line(ctx, sx(48), chart.y, sx(48), chart.y + chart.h, C.ink, 1.25);
        ctx.fillStyle = C.ink;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("сменилось расписание", sx(48) + 8, chart.y + 18);
        ctx.fillStyle = C.muted;
        ctx.fillText("старая среда", chart.x + 8, chart.y + 18);

        labelLine(ctx, chart.x + 18, chart.y + chart.h - 58, C.red, "прогноз модели");
        labelLine(ctx, chart.x + 186, chart.y + chart.h - 58, C.blue, "что увидел счётчик");
        labelLine(ctx, chart.x + 384, chart.y + chart.h - 58, C.axis, "нагрузка без решения", [6, 5]);
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText("часы после запуска", chart.x + chart.w / 2, chart.y + chart.h + 42);
        ctx.save();
        ctx.translate(20, chart.y + chart.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("нагрузка, кВт", 0, 0);
        ctx.restore();

        drawSystem(ctx, result);

        var oldErrors = result.predicted.slice(0, 48).map(function (value, i) {
          return Math.abs(value - result.raw[i]);
        });
        var liveErrors = result.predicted.slice(48).map(function (value, i) {
          return Math.abs(value - result.observed[i + 48]);
        });
        var excess = result.observed.slice(48).map(function (value) {
          return Math.max(0, value - state.threshold);
        });
        var moved = result.actions.slice(48).reduce(function (sum, value) {
          return sum + value;
        }, 0);
        var rawEnergy = result.raw.reduce(function (sum, value) {
          return sum + value;
        }, 0);
        var observedEnergy = result.observed.reduce(function (sum, value) {
          return sum + value;
        }, 0);
        var energyBalance = observedEnergy + result.pending - rawEnergy;
        output.set([
          { label: "MAE на старом тесте", value: mean(oldErrors).toFixed(1) + " кВт", color: C.green },
          { label: "MAE после запуска", value: mean(liveErrors).toFixed(1) + " кВт", color: C.red },
          { label: "нагрузка выше порога", value: excess.reduce(function (a, b) { return a + b; }, 0).toFixed(0) + " кВт·ч" },
          { label: "снято с исходных часов", value: moved.toFixed(0) + " кВт·ч", color: C.gold },
          { label: "хвост после 96-го часа", value: result.pending.toFixed(1) + " кВт·ч" },
          { label: "баланс энергии с хвостом", value: energyBalance.toFixed(1) + " кВт·ч", color: C.green },
        ]);
      }

      canvasState = K.makeCanvas(stage, W, H, {
        maxWidth: W,
        label: "Симуляция прогноза школьной нагрузки внутри замкнутого контура",
        onResize: draw,
      });

      K.segmented(controls, {
        label: "Право системы",
        value: state.mode,
        options: [
          { value: "forecast", label: "только прогноз" },
          { value: "advice", label: "совет: принято 45%" },
          { value: "auto", label: "автодействие" },
        ],
      }, function (value) {
        state.mode = value;
        draw();
      });
      K.slider(controls, {
        label: "Сила смены расписания",
        min: 0,
        max: 100,
        step: 1,
        value: state.drift,
        unit: "%",
      }, function (value) {
        state.drift = value;
        draw();
      });
      K.slider(controls, {
        label: "Исполнение решения",
        min: 0,
        max: 100,
        step: 1,
        value: state.response,
        unit: "%",
      }, function (value) {
        state.response = value;
        draw();
      });
      K.slider(controls, {
        label: "Порог пика",
        min: 58,
        max: 82,
        step: 1,
        value: state.threshold,
        unit: " кВт",
      }, function (value) {
        state.threshold = value;
        draw();
      });

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
