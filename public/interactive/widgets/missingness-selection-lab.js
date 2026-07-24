// Lesson 03: a real 48-hour UCI CO series with explicit observation probabilities.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("missingness-selection-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 470;
      var co = [
        2.7, 2.8, 2.7, 4.5, 3.5, 2.6, 1.7, 1.3, 1.3, 1.0, 0.7, 0.7,
        0.5, 0.5, 0.6, 1.1, 2.7, 3.5, 2.3, 1.6, 1.3, 2.0, 1.9, 1.9,
        2.2, 2.0, 2.9, 5.2, 4.6, 2.5, 1.5, 1.2, 1.7, 1.4, 1.2, 0.6,
        0.7, 0.8, 0.9, 1.6, 3.4, 3.8, 3.1, 2.7, 2.0, 2.3, 1.9, 1.3,
      ];
      var state = {
        mode: String(options.mode || "mnar"),
        missing: Number(options.missing || 36),
        seed: Number(options.seed || 23),
        selected: Number(options.selected || 27),
      };

      K.hint(
        root,
        "Синие точки попадут в таблицу, красные кольца исчезнут. Механизм выбирает не значения, а вероятности R=1. Сравни обычное среднее и поправку обратными вероятностями.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      root.appendChild(K.element("p", "kontur-int-hint", {
        text: "Таблица ниже шире узкого экрана: проведите по ней пальцем, чтобы увидеть p(R=1) и вклад x/p.",
      }));
      var tableWrap = K.element("div", "kontur-int-table-wrap");
      var table = K.element("table", "kontur-int-data-table");
      table.setAttribute("aria-label", "Строки вокруг выбранного часа и их вероятности наблюдения");
      tableWrap.appendChild(table);
      root.appendChild(tableWrap);
      K.caption(
        root,
        "48 значений CO(GT) взяты из последовательных строк UCI Air Quality за 24–26 марта 2004 года. Симулятор берёт общие uᵢ из LCG, кладёт Rᵢ=1[uᵢ<pᵢ] и калибрует pᵢ=clip(aᵢ+c, 0,08, 0,97). MCAR: aᵢ=0. MAR: aᵢ=0,84 в местные часы 07–21, иначе 0,34; hourᵢ=(16+i) mod 24. MNAR: aᵢ=0,93−0,72zᵢ, zᵢ=(COᵢ−min CO)/(max CO−min CO). Для MNAR показана oracle-IPW.",
      );

      var canvasState;
      var redrawFrame = 0;
      var redrawFollowup = 0;
      var visibilityObserver = null;
      var seedName = null;
      var chart = { x: 64, y: 48, w: 650, h: 315 };
      var probabilityBox = { x: 782, y: 72, w: 210, h: 230 };

      function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }

      function uniforms() {
        var current = state.seed >>> 0;
        return co.map(function () {
          current = (Math.imul(1664525, current) + 1013904223) >>> 0;
          return (current + 0.5) / 4294967296;
        });
      }

      function calibrated(raw, targetObserved) {
        var lower = -2;
        var upper = 2;
        for (var iteration = 0; iteration < 80; iteration += 1) {
          var middle = (lower + upper) / 2;
          var mean = raw.reduce(function (sum, value) {
            return sum + clamp(value + middle, 0.08, 0.97);
          }, 0) / raw.length;
          if (mean < targetObserved) lower = middle;
          else upper = middle;
        }
        var shift = (lower + upper) / 2;
        return raw.map(function (value) {
          return clamp(value + shift, 0.08, 0.97);
        });
      }

      function probabilities() {
        var targetObserved = 1 - state.missing / 100;
        var raw;
        if (state.mode === "mcar") {
          raw = co.map(function () { return 0; });
        } else if (state.mode === "mar") {
          raw = co.map(function (_value, index) {
            var hour = (16 + index) % 24;
            return hour >= 7 && hour <= 21 ? 0.84 : 0.34;
          });
        } else {
          var minimum = Math.min.apply(null, co);
          var maximum = Math.max.apply(null, co);
          raw = co.map(function (value) {
            var normalized = (value - minimum) / (maximum - minimum);
            return 0.93 - 0.72 * normalized;
          });
        }
        return calibrated(raw, targetObserved);
      }

      function selection() {
        var p = probabilities();
        var u = uniforms();
        var observed = p.map(function (probability, index) {
          return u[index] < probability;
        });
        return { p: p, u: u, observed: observed };
      }

      function statistics(selectionState) {
        var fullMean = co.reduce(function (sum, value) { return sum + value; }, 0) / co.length;
        var observedValues = co.filter(function (_value, index) {
          return selectionState.observed[index];
        });
        var observedMean = observedValues.length
          ? observedValues.reduce(function (sum, value) { return sum + value; }, 0) / observedValues.length
          : NaN;
        var numerator = 0;
        var denominator = 0;
        co.forEach(function (value, index) {
          if (!selectionState.observed[index]) return;
          numerator += value / selectionState.p[index];
          denominator += 1 / selectionState.p[index];
        });
        return {
          fullMean: fullMean,
          observedMean: observedMean,
          ipwMean: denominator ? numerator / denominator : NaN,
          count: observedValues.length,
          expectedMissing: 1 - selectionState.p.reduce(function (sum, value) {
            return sum + value;
          }, 0) / selectionState.p.length,
          realizedMissing: 1 - observedValues.length / co.length,
        };
      }

      function x(index) {
        return chart.x + index / (co.length - 1) * chart.w;
      }

      function y(value) {
        return chart.y + chart.h - value / 7 * chart.h;
      }

      function probabilityX(index) {
        return probabilityBox.x + index / (co.length - 1) * probabilityBox.w;
      }

      function probabilityY(value) {
        return probabilityBox.y + probabilityBox.h - value * probabilityBox.h;
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

      function dateTime(index) {
        var startHour = 16;
        var absolute = startHour + index;
        var day = 24 + Math.floor(absolute / 24);
        var hour = absolute % 24;
        return String(day).padStart(2, "0") + ".03 · " + String(hour).padStart(2, "0") + ":00";
      }

      function modeFormula() {
        if (state.mode === "mcar") return "aᵢ = 0";
        if (state.mode === "mar") return "aᵢ = .84/.34 по часу";
        return "aᵢ = .93 − .72zᵢ";
      }

      function drawAxes(ctx) {
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "right";
        [0, 2, 4, 6].forEach(function (value) {
          var py = y(value);
          ctx.strokeStyle = C.grid;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chart.x, py);
          ctx.lineTo(chart.x + chart.w, py);
          ctx.stroke();
          ctx.fillText(String(value), chart.x - 10, py + 4);
        });
        [0, 12, 24, 36, 47].forEach(function (value) {
          var px = x(value);
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(px, chart.y);
          ctx.lineTo(px, chart.y + chart.h);
          ctx.stroke();
          ctx.textAlign = "center";
          ctx.fillText(String(value), px, chart.y + chart.h + 20);
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(chart.x, chart.y, chart.w, chart.h);
        ctx.save();
        ctx.translate(chart.x - 43, chart.y + chart.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText("CO(GT), мг/м³", 0, 0);
        ctx.restore();
        ctx.textAlign = "center";
        ctx.fillText("час от 24.03 16:00", chart.x + chart.w / 2, chart.y + chart.h + 42);
      }

      function draw() {
        var current = selection();
        var stats = statistics(current);
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, W, H);
        drawAxes(ctx);

        var fullPoints = co.map(function (value, index) {
          return { x: x(index), y: y(value) };
        });
        line(ctx, fullPoints, C.axis, 1.25);

        co.forEach(function (value, index) {
          ctx.beginPath();
          ctx.arc(x(index), y(value), current.observed[index] ? 3.4 : 5.0, 0, Math.PI * 2);
          if (current.observed[index]) {
            ctx.fillStyle = C.blue;
            ctx.fill();
          } else {
            ctx.fillStyle = C.paper;
            ctx.fill();
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 1.6;
            ctx.stroke();
          }
        });

        var selectedX = x(state.selected);
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(selectedX, chart.y);
        ctx.lineTo(selectedX, chart.y + chart.h);
        ctx.stroke();
        ctx.fillStyle = C.ink;
        ctx.font = "600 11px system-ui, sans-serif";
        ctx.textAlign = state.selected > 38 ? "right" : "left";
        ctx.fillText(
          "час " + state.selected + " · " + dateTime(state.selected),
          selectedX + (state.selected > 38 ? -7 : 7),
          chart.y + 17,
        );

        ctx.fillStyle = C.ink;
        ctx.font = "600 14px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("какие строки остаются", chart.x, 25);
        ctx.fillText("вероятность наблюдения", probabilityBox.x, 25);

        ctx.strokeStyle = C.axis;
        ctx.strokeRect(probabilityBox.x, probabilityBox.y, probabilityBox.w, probabilityBox.h);
        [0, 0.5, 1].forEach(function (value) {
          var py = probabilityY(value);
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(probabilityBox.x, py);
          ctx.lineTo(probabilityBox.x + probabilityBox.w, py);
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.font = "10px system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(value.toFixed(1), probabilityBox.x - 7, py + 3);
        });
        var probabilityPoints = current.p.map(function (value, index) {
          return { x: probabilityX(index), y: probabilityY(value) };
        });
        line(ctx, probabilityPoints, C.green, 2.0);
        ctx.beginPath();
        ctx.arc(probabilityX(state.selected), probabilityY(current.p[state.selected]), 5, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        ctx.fillStyle = C.muted;
        ctx.font = "11px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("час", probabilityBox.x + probabilityBox.w / 2, probabilityBox.y + probabilityBox.h + 22);

        var selectedObserved = current.observed[state.selected];
        ctx.fillStyle = C.wash;
        ctx.fillRect(probabilityBox.x, 337, probabilityBox.w, 82);
        ctx.fillStyle = C.ink;
        ctx.font = "600 12px ui-monospace, monospace";
        ctx.textAlign = "left";
        ctx.fillText(modeFormula(), probabilityBox.x + 12, 360);
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.fillText(
          "CO = " + co[state.selected].toFixed(1) + " мг/м³",
          probabilityBox.x + 12,
          383,
        );
        ctx.fillText(
          "p = " + current.p[state.selected].toFixed(2)
            + " · u = " + current.u[state.selected].toFixed(2)
            + " · R = " + (selectedObserved ? "1" : "0"),
          probabilityBox.x + 12,
          404,
        );

        if (seedName) seedName.textContent = "seed = " + state.seed;
        output.set([
          { label: "seed", value: String(state.seed) },
          { label: "цель пропусков", value: state.missing.toFixed(1) + "%" },
          { label: "ожидалось", value: (100 * stats.expectedMissing).toFixed(1) + "%" },
          { label: "получилось", value: (100 * stats.realizedMissing).toFixed(1) + "%" },
          { label: "строки", value: stats.count + " / " + co.length },
          { label: "среднее источника", value: stats.fullMean.toFixed(3), color: C.ink },
          {
            label: "среднее видимых",
            value: Number.isFinite(stats.observedMean) ? stats.observedMean.toFixed(3) : "нет строк",
            color: Math.abs(stats.observedMean - stats.fullMean) > 0.2 ? C.red : C.green,
          },
          {
            label: state.mode === "mnar" ? "oracle-IPW" : "IPW",
            value: Number.isFinite(stats.ipwMean) ? stats.ipwMean.toFixed(3) : "нет строк",
            color: C.green,
          },
        ]);
        renderTable(current);
      }

      function scheduleDraw() {
        if (redrawFrame) cancelAnimationFrame(redrawFrame);
        if (redrawFollowup) cancelAnimationFrame(redrawFollowup);
        redrawFrame = requestAnimationFrame(function () {
          redrawFrame = 0;
          draw();
          redrawFollowup = requestAnimationFrame(function () {
            redrawFollowup = 0;
            draw();
          });
        });
      }

      function cell(tag, text, className) {
        var element = K.element(tag, className || "", { text: text });
        return element;
      }

      function renderTable(current) {
        while (table.firstChild) table.removeChild(table.firstChild);
        var head = document.createElement("thead");
        var headRow = document.createElement("tr");
        [
          "i",
          "дата · час",
          "CO источника",
          "p(R=1)",
          "uᵢ",
          "в таблице",
          "вклад x/p",
        ].forEach(function (label) {
          headRow.appendChild(cell("th", label));
        });
        head.appendChild(headRow);
        table.appendChild(head);

        var body = document.createElement("tbody");
        var start = clamp(state.selected - 3, 0, co.length - 7);
        for (var index = start; index < start + 7; index += 1) {
          var row = document.createElement("tr");
          if (index === state.selected) row.className = "is-selected";
          row.appendChild(cell("td", String(index)));
          row.appendChild(cell("td", dateTime(index)));
          row.appendChild(cell("td", co[index].toFixed(1) + " мг/м³"));
          row.appendChild(cell("td", current.p[index].toFixed(2)));
          row.appendChild(cell("td", current.u[index].toFixed(3)));
          row.appendChild(
            cell(
              "td",
              current.observed[index] ? "R = 1" : "R = 0 · скрыто",
              current.observed[index] ? "is-observed" : "is-missing",
            ),
          );
          row.appendChild(
            cell(
              "td",
              current.observed[index]
                ? (co[index] / current.p[index]).toFixed(2)
                : "—",
            ),
          );
          body.appendChild(row);
        }
        table.appendChild(body);
      }

      canvasState = K.makeCanvas(stage, W, H, {
        maxWidth: W,
        label: "Реальный ряд CO и отбор строк при трёх механизмах пропусков",
        onResize: scheduleDraw,
      });

      K.segmented(
        controls,
        {
          label: "Механизм",
          value: state.mode,
          options: [
            { value: "mcar", label: "MCAR" },
            { value: "mar", label: "MAR" },
            { value: "mnar", label: "MNAR" },
          ],
        },
        function (value) {
          state.mode = value;
          draw();
        },
      );
      K.slider(
        controls,
        {
          label: "Целевая доля пропусков",
          min: 10,
          max: 72,
          step: 1,
          value: state.missing,
          unit: "%",
        },
        function (value) {
          state.missing = value;
          draw();
        },
      );
      K.slider(
        controls,
        {
          label: "Выбранный час",
          min: 0,
          max: 47,
          step: 1,
          value: state.selected,
          unit: "",
        },
        function (value) {
          state.selected = Math.round(value);
          draw();
        },
      );
      var action = K.element("div", "kontur-int-control");
      seedName = K.element("span", "kontur-int-label-name", { text: "seed = " + state.seed });
      action.appendChild(seedName);
      var repeat = K.element("button", "kontur-int-action", {
        type: "button",
        text: "Новый seed",
        "aria-label": "Повторить отбор строк с новым seed",
      });
      repeat.addEventListener("click", function () {
        state.seed += 1;
        draw();
      });
      action.appendChild(repeat);
      var reset = K.element("button", "kontur-int-action", {
        type: "button",
        text: "Сбросить: 23",
        "aria-label": "Вернуть seed 23",
      });
      reset.addEventListener("click", function () {
        state.seed = 23;
        draw();
      });
      action.appendChild(reset);
      controls.appendChild(action);

      draw();
      scheduleDraw();
      if (typeof IntersectionObserver !== "undefined") {
        visibilityObserver = new IntersectionObserver(function (entries) {
          if (entries.some(function (entry) { return entry.isIntersecting; })) {
            scheduleDraw();
          }
        }, { rootMargin: "120px" });
        visibilityObserver.observe(canvasState.canvas);
      }
      return function () {
        if (redrawFrame) cancelAnimationFrame(redrawFrame);
        if (redrawFollowup) cancelAnimationFrame(redrawFollowup);
        if (visibilityObserver) visibilityObserver.disconnect();
        canvasState.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
