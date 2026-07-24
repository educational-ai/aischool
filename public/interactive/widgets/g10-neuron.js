// Grade 10, module 2. Lightweight neural-network and optimization figures.
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
      var value = Math.sin(index * 84.137 + salt * 23.771) * 43758.5453;
      return value - Math.floor(value);
    }

    function setup(root, hint, label, caption, height) {
      K.hint(root, hint);
      var stage = K.row(root);
      var redraw = function () {};
      var canvasState = K.makeCanvas(stage, 920, height || 450, {
        maxWidth: 920,
        label: label,
        onResize: function () { redraw(); },
      });
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, caption);
      return {
        canvas: canvasState,
        controls: controls,
        output: output,
        setDraw: function (fn) { redraw = fn; },
        destroy: function () { canvasState.destroy(); },
      };
    }

    function line(ctx, points, color, width, dash) {
      if (!points.length) return;
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

    function arrow(ctx, x1, y1, x2, y2, color) {
      var angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 8 * Math.cos(angle - 0.45), y2 - 8 * Math.sin(angle - 0.45));
      ctx.lineTo(x2 - 8 * Math.cos(angle + 0.45), y2 - 8 * Math.sin(angle + 0.45));
      ctx.closePath();
      ctx.fill();
    }

    function buildLogicNeuron(root) {
      var ui = setup(
        root,
        "Выбери операцию или настрой веса. Цвет вершины показывает ответ, линия — порог w₁x₁+w₂x₂=b.",
        "Пороговый нейрон на четырёх бинарных входах",
        "Сдвиг порога перемещает границу параллельно. Вес меняет её наклон и вклад соответствующего входа.",
        455,
      );
      var state = { w1: 1, w2: 1, b: 2, op: "and" };
      var box = { x: 120, y: 44, w: 590, h: 330 };
      var truthX = 760;
      function sx(x) { return box.x + (x + 0.25) / 1.5 * box.w; }
      function sy(y) { return box.y + (1.25 - y) / 1.5 * box.h; }
      var operations = {
        and: { w1: 1, w2: 1, b: 2 },
        or: { w1: 1, w2: 1, b: 1 },
        x1: { w1: 1, w2: 0, b: 1 },
        custom: null,
      };
      var weight1;
      var weight2;
      var threshold;
      var selector;

      function answer(x1, x2) {
        return state.w1 * x1 + state.w2 * x2 >= state.b ? 1 : 0;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 455);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 455);
        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1;
        [-0.25, 0, 0.5, 1, 1.25].forEach(function (value) {
          ctx.beginPath();
          ctx.moveTo(sx(value), box.y);
          ctx.lineTo(sx(value), box.y + box.h);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(box.x, sy(value));
          ctx.lineTo(box.x + box.w, sy(value));
          ctx.stroke();
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("x₁", box.x + box.w + 12, sy(0) + 4);
        ctx.fillText("x₂", sx(0) - 6, box.y - 12);

        if (Math.abs(state.w2) > 0.0001) {
          var xA = -0.25;
          var xB = 1.25;
          line(ctx, [
            { x: sx(xA), y: sy((state.b - state.w1 * xA) / state.w2) },
            { x: sx(xB), y: sy((state.b - state.w1 * xB) / state.w2) },
          ], C.ink, 2.3);
        } else if (Math.abs(state.w1) > 0.0001) {
          var boundary = state.b / state.w1;
          line(ctx, [
            { x: sx(boundary), y: box.y },
            { x: sx(boundary), y: box.y + box.h },
          ], C.ink, 2.3);
        }

        var rows = [];
        [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(function (point) {
          var result = answer(point[0], point[1]);
          rows.push(point[0] + "  " + point[1] + "  →  " + result);
          ctx.beginPath();
          ctx.arc(sx(point[0]), sy(point[1]), 16, 0, Math.PI * 2);
          ctx.fillStyle = result ? C.red : C.blue;
          ctx.fill();
          ctx.strokeStyle = C.paper;
          ctx.lineWidth = 3;
          ctx.stroke();
          ctx.fillStyle = C.paper;
          ctx.font = "600 12px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(String(result), sx(point[0]), sy(point[1]) + 4);
        });
        ctx.textAlign = "left";
        ctx.fillStyle = C.ink;
        ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
        ctx.fillText("x₁  x₂      y", truthX, 88);
        ctx.font = "15px ui-monospace, monospace";
        rows.forEach(function (row, index) {
          ctx.fillText(row, truthX, 126 + index * 42);
        });
        ctx.fillStyle = C.muted;
        ctx.font = "13px system-ui, sans-serif";
        ctx.fillText("w₁x₁ + w₂x₂ ≥ b", truthX, 326);
        ui.output.set([
          { label: "веса", value: state.w1.toFixed(1) + ", " + state.w2.toFixed(1) },
          { label: "порог", value: state.b.toFixed(1) },
          { label: "единиц в таблице", value: String(rows.filter(function (_, i) {
            var p = [[0, 0], [0, 1], [1, 0], [1, 1]][i];
            return answer(p[0], p[1]);
          }).length) },
        ]);
      }

      selector = K.segmented(ui.controls, {
        label: "Операция",
        value: state.op,
        options: [
          { value: "and", label: "И" },
          { value: "or", label: "ИЛИ" },
          { value: "x1", label: "повтор x₁" },
          { value: "custom", label: "своя" },
        ],
      }, function (value) {
        state.op = value;
        var preset = operations[value];
        if (preset) {
          state.w1 = preset.w1;
          state.w2 = preset.w2;
          state.b = preset.b;
          weight1.set(state.w1);
          weight2.set(state.w2);
          threshold.set(state.b);
        }
        draw();
      });
      function custom() {
        state.op = "custom";
        selector.set("custom");
        draw();
      }
      weight1 = K.slider(ui.controls, {
        label: "Вес w₁", min: -3, max: 3, step: 0.1, value: state.w1,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.w1 = value; custom(); });
      weight2 = K.slider(ui.controls, {
        label: "Вес w₂", min: -3, max: 3, step: 0.1, value: state.w2,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.w2 = value; custom(); });
      threshold = K.slider(ui.controls, {
        label: "Порог b", min: -2, max: 4, step: 0.1, value: state.b,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.b = value; custom(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildHomeostat(root) {
      var ui = setup(
        root,
        "Усиление исправляет отклонение, задержка возвращает старую ошибку, медленная адаптация меняет сам коэффициент связи.",
        "Температура, цель и адаптивная обратная связь",
        "Синяя линия — температура, красная — цель, зелёная — текущая сила регулятора. Большая задержка превращает коррекцию в колебание.",
        465,
      );
      var state = { gain: 0.95, delay: 3, adapt: 0.006 };
      var box = { x: 66, y: 42, w: 806, h: 310 };

      function simulate() {
        var temps = [17];
        var gains = [state.gain];
        var controls = [];
        var outside = 8;
        for (var t = 1; t < 140; t += 1) {
          var target = t < 58 ? 21 : 19;
          var delayedIndex = Math.max(0, t - 1 - Math.round(state.delay));
          var error = target - temps[delayedIndex];
          var currentGain = gains[t - 1];
          var action = clamp(currentGain * error, -4.5, 4.5);
          controls.push(action);
          var next = temps[t - 1] + 0.12 * action
            - 0.035 * (temps[t - 1] - outside)
            + (seeded(t, 61) - 0.5) * 0.08;
          var hebb = state.adapt * error * action;
          var decay = state.adapt * 0.15 * currentGain;
          var nextGain = clamp(currentGain + hebb - decay, 0.05, 2.8);
          temps.push(next);
          gains.push(nextGain);
        }
        return { temps: temps, gains: gains, controls: controls };
      }
      function sx(t) { return box.x + t / 139 * box.w; }
      function sy(temp) { return box.y + (25 - temp) / 13 * box.h; }
      function gy(gain) { return box.y + box.h - gain / 3 * box.h; }

      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 465);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 465);
        ctx.strokeStyle = C.grid;
        [12, 16, 20, 24].forEach(function (temp) {
          line(ctx, [{ x: box.x, y: sy(temp) }, { x: box.x + box.w, y: sy(temp) }], C.grid, 1);
          ctx.fillStyle = C.muted;
          ctx.font = "12px system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(temp + "°", box.x - 10, sy(temp) + 4);
        });
        [0, 35, 70, 105, 139].forEach(function (t) {
          line(ctx, [{ x: sx(t), y: box.y }, { x: sx(t), y: box.y + box.h }], C.grid, 1);
          ctx.textAlign = "center";
          ctx.fillText(String(t), sx(t), box.y + box.h + 22);
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        var sim = simulate();
        var targetLine = [];
        var tempLine = [];
        var gainLine = [];
        sim.temps.forEach(function (temp, t) {
          targetLine.push({ x: sx(t), y: sy(t < 58 ? 21 : 19) });
          tempLine.push({ x: sx(t), y: sy(temp) });
          gainLine.push({ x: sx(t), y: gy(sim.gains[t]) });
        });
        line(ctx, targetLine, C.red, 1.7, [6, 4]);
        line(ctx, tempLine, C.blue, 2.6);
        line(ctx, gainLine, C.green, 1.6);
        ctx.textAlign = "left";
        ctx.fillStyle = C.red;
        ctx.fillText("цель", box.x + 12, box.y + 18);
        ctx.fillStyle = C.blue;
        ctx.fillText("температура", box.x + 62, box.y + 18);
        ctx.fillStyle = C.green;
        ctx.fillText("усиление", box.x + 156, box.y + 18);
        var tail = sim.temps.slice(-35);
        var amplitude = Math.max.apply(null, tail) - Math.min.apply(null, tail);
        ui.output.set([
          { label: "конечная температура", value: sim.temps.at(-1).toFixed(1) + "°", color: C.blue },
          { label: "конечное усиление", value: sim.gains.at(-1).toFixed(2), color: C.green },
          { label: "амплитуда последних шагов", value: amplitude.toFixed(2) + "°" },
          { label: "режим", value: amplitude > 2.5 ? "раскачка" : "устойчивый" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Начальное усиление", min: 0.1, max: 2.5, step: 0.05, value: state.gain,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.gain = value; draw(); });
      K.slider(ui.controls, {
        label: "Задержка", min: 0, max: 14, step: 1, value: state.delay, unit: " шаг.",
      }, function (value) { state.delay = value; draw(); });
      K.slider(ui.controls, {
        label: "Скорость адаптации", min: 0, max: 0.02, step: 0.001, value: state.adapt,
        format: function (v) { return v.toFixed(3); },
      }, function (value) { state.adapt = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildPerceptron(root) {
      var ui = setup(
        root,
        "Ползунок показывает последовательные предъявления точек. Красная стрелка отмечает последнее ошибочное обновление.",
        "Пошаговое обучение линейного перцептрона",
        "При пересечении классов прямая не успокаивается. Скорость меняет длину шага, но не создаёт линейную разделимость.",
        460,
      );
      var state = { steps: 12, eta: 0.18, overlap: 8 };
      var box = { x: 88, y: 38, w: 710, h: 330 };
      var data = [];
      for (var i = 0; i < 12; i += 1) {
        data.push({
          x: 0.22 + (seeded(i, 71) - 0.5) * 0.26,
          y: 0.28 + (seeded(i, 73) - 0.5) * 0.34,
          label: -1,
        });
        data.push({
          x: 0.72 + (seeded(i, 75) - 0.5) * 0.28,
          y: 0.7 + (seeded(i, 77) - 0.5) * 0.32,
          label: 1,
        });
      }
      function currentData() {
        return data.map(function (p, index) {
          if (index === 2) return { x: p.x + state.overlap / 100 * 0.62, y: p.y + state.overlap / 100 * 0.48, label: p.label };
          return p;
        });
      }
      function train() {
        var values = currentData();
        var w = { x: 0.15, y: -0.1 };
        var b = 0;
        var errors = 0;
        var last = null;
        for (var step = 0; step < state.steps; step += 1) {
          var index = step % values.length;
          var p = values[index];
          var score = w.x * p.x + w.y * p.y + b;
          var pred = score >= 0 ? 1 : -1;
          if (pred !== p.label) {
            var before = { x: w.x, y: w.y, b: b };
            w.x += state.eta * p.label * p.x;
            w.y += state.eta * p.label * p.y;
            b += state.eta * p.label;
            errors += 1;
            last = { point: p, before: before, after: { x: w.x, y: w.y, b: b } };
          }
        }
        return { values: values, w: w, b: b, errors: errors, last: last };
      }
      function sx(x) { return box.x + x * box.w; }
      function sy(y) { return box.y + (1 - y) * box.h; }

      function drawBoundary(ctx, w, b, color, dash) {
        if (Math.abs(w.y) < 0.0001) return;
        line(ctx, [
          { x: sx(0), y: sy(-b / w.y) },
          { x: sx(1), y: sy((-b - w.x) / w.y) },
        ], color, 2.2, dash);
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 460);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 460);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        var result = train();
        if (result.last) drawBoundary(ctx, result.last.before, result.last.before.b, C.axis, [5, 5]);
        drawBoundary(ctx, result.w, result.b, C.ink);
        result.values.forEach(function (p) {
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 6, 0, Math.PI * 2);
          ctx.fillStyle = p.label > 0 ? C.red : C.blue;
          ctx.fill();
          ctx.strokeStyle = C.paper;
          ctx.lineWidth = 2;
          ctx.stroke();
        });
        if (result.last) {
          var p = result.last.point;
          ctx.strokeStyle = C.gold;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 13, 0, Math.PI * 2);
          ctx.stroke();
          var start = { x: 834, y: 150 };
          arrow(ctx, start.x, start.y, start.x + p.label * p.x * 55, start.y - p.label * p.y * 55, C.red);
          ctx.fillStyle = C.muted;
          ctx.font = "12px system-ui, sans-serif";
          ctx.fillText("η y x", 812, 118);
        }
        var mistakes = result.values.filter(function (p) {
          return (result.w.x * p.x + result.w.y * p.y + result.b >= 0 ? 1 : -1) !== p.label;
        }).length;
        ui.output.set([
          { label: "предъявлений", value: String(state.steps) },
          { label: "обновлений", value: String(result.errors), color: C.red },
          { label: "ошибок сейчас", value: String(mistakes) },
          { label: "граница", value: mistakes ? "ещё спорит с точками" : "разделяет выборку" },
        ]);
      }
      K.slider(ui.controls, {
        label: "Предъявлено примеров", min: 0, max: 120, step: 1, value: state.steps,
      }, function (value) { state.steps = value; draw(); });
      K.slider(ui.controls, {
        label: "Скорость η", min: 0.02, max: 0.5, step: 0.01, value: state.eta,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.eta = value; draw(); });
      K.slider(ui.controls, {
        label: "Пересечение классов", min: 0, max: 100, step: 1, value: state.overlap, unit: "%",
      }, function (value) { state.overlap = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildXor(root) {
      var ui = setup(
        root,
        "Слева исходные координаты. Справа признаки OR и NAND. Ползунок показывает движение каждой вершины между представлениями.",
        "Преобразование XOR скрытым слоем",
        "Входные классы лежат по диагонали и не делятся прямой. Два скрытых признака совмещают положительные точки в одной вершине.",
        450,
      );
      var state = { morph: 1, h1: true, h2: true };
      var left = { x: 72, y: 64, w: 300, h: 280 };
      var right = { x: 548, y: 64, w: 300, h: 280 };
      var points = [
        { x: 0, y: 0, label: 0 },
        { x: 1, y: 0, label: 1 },
        { x: 0, y: 1, label: 1 },
        { x: 1, y: 1, label: 0 },
      ];
      function hidden(p) {
        return {
          x: state.h1 ? (p.x || p.y ? 1 : 0) : p.x,
          y: state.h2 ? (!(p.x && p.y) ? 1 : 0) : p.y,
        };
      }
      function px(box, x) { return box.x + 48 + x * (box.w - 96); }
      function py(box, y) { return box.y + box.h - 48 - y * (box.h - 96); }
      function frame(ctx, box, xLabel, yLabel) {
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText(xLabel, box.x + box.w - 30, box.y + box.h - 18);
        ctx.fillText(yLabel, box.x + 18, box.y + 22);
      }
      function drawPoint(ctx, x, y, label) {
        ctx.beginPath();
        ctx.arc(x, y, 11, 0, Math.PI * 2);
        ctx.fillStyle = label ? C.red : C.blue;
        ctx.fill();
        ctx.strokeStyle = C.paper;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = C.paper;
        ctx.font = "600 11px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(label), x, y + 4);
        ctx.textAlign = "left";
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 450);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 450);
        frame(ctx, left, "x₁", "x₂");
        frame(ctx, right, "OR", "NAND");
        ctx.fillStyle = C.ink;
        ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
        ctx.fillText("вход", left.x, 42);
        ctx.fillText("скрытые признаки", right.x, 42);
        line(ctx, [
          { x: px(right, 0.75), y: py(right, 1) },
          { x: px(right, 1), y: py(right, 0.75) },
        ], C.ink, 2);
        points.forEach(function (p) {
          var h = hidden(p);
          var lx = px(left, p.x);
          var ly = py(left, p.y);
          var rx = px(right, h.x);
          var ry = py(right, h.y);
          drawPoint(ctx, lx, ly, p.label);
          drawPoint(ctx, rx, ry, p.label);
          var middleX = lx + (rx - lx) * state.morph;
          var middleY = ly + (ry - ly) * state.morph;
          ctx.globalAlpha = 0.28;
          arrow(ctx, lx + 14, ly, rx - 14, ry, p.label ? C.red : C.blue);
          ctx.globalAlpha = 1;
          ctx.beginPath();
          ctx.arc(middleX, middleY, 5, 0, Math.PI * 2);
          ctx.fillStyle = p.label ? C.red : C.blue;
          ctx.fill();
        });
        var separable = state.h1 && state.h2;
        ui.output.set([
          { label: "OR", value: state.h1 ? "включён" : "выключен" },
          { label: "NAND", value: state.h2 ? "включён" : "выключен" },
          { label: "линейная граница справа", value: separable ? "существует" : "не хватает признака", color: separable ? C.green : C.red },
        ]);
      }
      K.slider(ui.controls, {
        label: "Переход к скрытому слою", min: 0, max: 1, step: 0.01, value: state.morph,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.morph = value; draw(); });
      K.segmented(ui.controls, {
        label: "Признак OR", value: "on",
        options: [{ value: "on", label: "включён" }, { value: "off", label: "нет" }],
      }, function (value) { state.h1 = value === "on"; draw(); });
      K.segmented(ui.controls, {
        label: "Признак NAND", value: "on",
        options: [{ value: "on", label: "включён" }, { value: "off", label: "нет" }],
      }, function (value) { state.h2 = value === "on"; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildActivation(root) {
      var ui = setup(
        root,
        "Масштаб растягивает вход перед активацией, смещение переносит рабочую область. Нижняя панель показывает производную.",
        "Активация и чувствительность нейрона",
        "Сигмоида насыщается на краях. ReLU сохраняет производную справа, но теряет её слева. Leaky ReLU оставляет малый наклон.",
        475,
      );
      var state = { type: "sigmoid", scale: 1, bias: 0 };
      var top = { x: 70, y: 35, w: 802, h: 230 };
      var bottom = { x: 70, y: 328, w: 802, h: 92 };
      function fn(z) {
        if (state.type === "sigmoid") return 1 / (1 + Math.exp(-z));
        if (state.type === "tanh") return Math.tanh(z);
        if (state.type === "relu") return Math.max(0, z);
        return z >= 0 ? z : 0.1 * z;
      }
      function derivative(z) {
        if (state.type === "sigmoid") {
          var s = fn(z);
          return s * (1 - s);
        }
        if (state.type === "tanh") return 1 - Math.pow(Math.tanh(z), 2);
        if (state.type === "relu") return z > 0 ? 1 : 0;
        return z >= 0 ? 1 : 0.1;
      }
      function sx(x) { return top.x + (x + 5) / 10 * top.w; }
      function topY(y) {
        var min = state.type === "relu" || state.type === "leaky" ? -1 : -1.2;
        var max = state.type === "relu" || state.type === "leaky" ? 5.2 : 1.2;
        return top.y + (max - y) / (max - min) * top.h;
      }
      function bottomY(y) { return bottom.y + bottom.h - y / 1.05 * bottom.h; }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 475);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 475);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(top.x, top.y, top.w, top.h);
        ctx.strokeRect(bottom.x, bottom.y, bottom.w, bottom.h);
        line(ctx, [{ x: top.x, y: topY(0) }, { x: top.x + top.w, y: topY(0) }], C.axis, 1);
        line(ctx, [{ x: sx(0), y: top.y }, { x: sx(0), y: top.y + top.h }], C.axis, 1);
        var curve = [];
        var grad = [];
        for (var i = 0; i <= 240; i += 1) {
          var x = -5 + i / 240 * 10;
          var z = state.scale * x + state.bias;
          curve.push({ x: sx(x), y: topY(fn(z)) });
          grad.push({ x: sx(x), y: bottomY(Math.min(1.05, Math.abs(state.scale * derivative(z)))) });
        }
        line(ctx, curve, C.blue, 2.7);
        line(ctx, grad, C.red, 2.3);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("φ(ax+b)", top.x + 10, top.y + 20);
        ctx.fillText("|dφ/dx|", bottom.x + 10, bottom.y + 20);
        var centerDerivative = Math.abs(state.scale * derivative(state.bias));
        var edgeDerivative = Math.abs(state.scale * derivative(state.scale * 4 + state.bias));
        ui.output.set([
          { label: "активация", value: state.type },
          { label: "производная при x=0", value: centerDerivative.toFixed(3), color: C.red },
          { label: "производная при x=4", value: edgeDerivative.toFixed(3) },
          { label: "область", value: edgeDerivative < 0.02 ? "насыщение" : "градиент проходит" },
        ]);
      }
      K.segmented(ui.controls, {
        label: "Функция", value: state.type,
        options: [
          { value: "sigmoid", label: "sigmoid" },
          { value: "tanh", label: "tanh" },
          { value: "relu", label: "ReLU" },
          { value: "leaky", label: "Leaky ReLU" },
        ],
      }, function (value) { state.type = value; draw(); });
      K.slider(ui.controls, {
        label: "Масштаб a", min: 0.2, max: 4, step: 0.1, value: state.scale,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.scale = value; draw(); });
      K.slider(ui.controls, {
        label: "Смещение b", min: -4, max: 4, step: 0.1, value: state.bias,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.bias = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLetterNetwork(root) {
      var ui = setup(
        root,
        "Выбери букву, добавь шум или нажимай клетки рисунка. Справа обновятся четыре внутренних признака и два выхода.",
        "Прямой проход сети 25–4–2",
        "Скрытые признаки заданы понятными шаблонами для демонстрации. В обучаемой сети веса возникнут из примеров и могут читаться хуже.",
        470,
      );
      var patterns = {
        c: [
          0,1,1,1,0,
          1,0,0,0,0,
          1,0,0,0,0,
          1,0,0,0,0,
          0,1,1,1,0,
        ],
        o: [
          0,1,1,1,0,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          0,1,1,1,0,
        ],
      };
      var state = { kind: "c", noise: 0, pixels: patterns.c.slice() };
      var grid = { x: 62, y: 72, size: 52 };
      var templates = [
        { label: "левая дуга", indices: [1,2,3,5,10,15,21,22,23] },
        { label: "правый край", indices: [9,14,19] },
        { label: "пустой центр", indices: [6,7,8,11,12,13,16,17,18], invert: true },
        { label: "замкнутость", indices: [1,2,3,5,9,15,19,21,22,23] },
      ];
      function visiblePixels() {
        return state.pixels.map(function (value, index) {
          return seeded(index, 91) < state.noise / 100 ? 1 - value : value;
        });
      }
      function activations(pixels) {
        return templates.map(function (template) {
          var sum = 0;
          template.indices.forEach(function (index) {
            sum += template.invert ? 1 - pixels[index] : pixels[index];
          });
          return sum / template.indices.length;
        });
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 470);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 470);
        var pixels = visiblePixels();
        var acts = activations(pixels);
        ctx.fillStyle = C.ink;
        ctx.font = "600 16px ET Book, Palatino, Georgia, serif";
        ctx.fillText("25 пикселей", grid.x, 40);
        ctx.fillText("4 скрытых признака", 400, 40);
        ctx.fillText("2 выхода", 760, 40);
        for (var row = 0; row < 5; row += 1) {
          for (var col = 0; col < 5; col += 1) {
            var index = row * 5 + col;
            ctx.fillStyle = pixels[index] ? C.ink : C.wash;
            ctx.fillRect(grid.x + col * grid.size, grid.y + row * grid.size, grid.size - 3, grid.size - 3);
            ctx.strokeStyle = C.grid;
            ctx.strokeRect(grid.x + col * grid.size, grid.y + row * grid.size, grid.size - 3, grid.size - 3);
          }
        }
        templates.forEach(function (template, index) {
          var y = 82 + index * 78;
          ctx.fillStyle = C.muted;
          ctx.font = "12px system-ui, sans-serif";
          ctx.fillText(template.label, 400, y);
          ctx.fillStyle = C.grid;
          ctx.fillRect(400, y + 14, 250, 24);
          ctx.fillStyle = [C.blue, C.red, C.green, C.gold][index];
          ctx.fillRect(400, y + 14, 250 * acts[index], 24);
          ctx.fillStyle = C.ink;
          ctx.font = "600 12px ui-monospace, monospace";
          ctx.fillText(Math.round(acts[index] * 100) + "%", 660, y + 32);
        });
        var scoreC = 1.9 * acts[0] - 1.6 * acts[1] + 0.4 * acts[2] - 0.3 * acts[3];
        var scoreO = 0.7 * acts[0] + 1.5 * acts[1] + 0.2 * acts[2] + 1.1 * acts[3];
        var maxScore = Math.max(scoreC, scoreO);
        var expC = Math.exp(scoreC - maxScore);
        var expO = Math.exp(scoreO - maxScore);
        var pC = expC / (expC + expO);
        var pO = 1 - pC;
        [["C", pC, C.blue], ["O", pO, C.red]].forEach(function (item, index) {
          var y = 130 + index * 110;
          ctx.fillStyle = C.wash;
          ctx.fillRect(748, y, 120, 70);
          ctx.fillStyle = item[2];
          ctx.fillRect(748, y + 54, 120 * item[1], 16);
          ctx.fillStyle = C.ink;
          ctx.font = "600 27px ET Book, Palatino, Georgia, serif";
          ctx.fillText(item[0], 764, y + 38);
          ctx.font = "600 13px ui-monospace, monospace";
          ctx.fillText(Math.round(item[1] * 100) + "%", 812, y + 37);
        });
        ui.output.set([
          { label: "прогноз", value: pC >= pO ? "C" : "O", color: pC >= pO ? C.blue : C.red },
          { label: "уверенность", value: Math.round(Math.max(pC, pO) * 100) + "%" },
          { label: "изменённых шумом клеток", value: String(pixels.filter(function (v, i) { return v !== state.pixels[i]; }).length) },
        ]);
      }
      function loadPattern(kind) {
        state.kind = kind;
        state.pixels = patterns[kind].slice();
        draw();
      }
      K.segmented(ui.controls, {
        label: "Начертание", value: state.kind,
        options: [{ value: "c", label: "C" }, { value: "o", label: "O" }],
      }, loadPattern);
      K.slider(ui.controls, {
        label: "Шум пикселей", min: 0, max: 45, step: 1, value: state.noise, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      var removeDrag = K.drag(ui.canvas.canvas, { w: 920, h: 470 }, {
        down: function (point) {
          var col = Math.floor((point.x - grid.x) / grid.size);
          var row = Math.floor((point.y - grid.y) / grid.size);
          if (col < 0 || col > 4 || row < 0 || row > 4) return;
          var index = row * 5 + col;
          state.pixels[index] = 1 - state.pixels[index];
          draw();
        },
      });
      ui.setDraw(draw);
      draw();
      return function () {
        removeDrag();
        ui.destroy();
      };
    }

    function buildApproximation(root) {
      var ui = setup(
        root,
        "Число изломов задаёт ширину сети. Точки на кривой соединяются линейно; такую ломаную можно записать суммой ReLU.",
        "Кусочно-линейная аппроксимация функцией ReLU",
        "Малая ошибка на отрезке не описывает продолжение за его границами. Серая область справа специально не участвовала в построении.",
        450,
      );
      var state = { target: "wave", neurons: 6, phase: 0 };
      var box = { x: 68, y: 42, w: 804, h: 310 };
      function target(x) {
        if (state.target === "wave") return 0.5 + 0.3 * Math.sin(2 * Math.PI * (x + state.phase));
        if (state.target === "peak") return 0.15 + 0.75 * Math.exp(-Math.pow((x - 0.58 - state.phase * 0.12) / 0.18, 2));
        return 0.18 + 0.2 * Math.sin(Math.PI * x) + 0.36 * Math.exp(-Math.pow((x - 0.32) / 0.12, 2))
          + 0.28 * Math.exp(-Math.pow((x - 0.78) / 0.14, 2));
      }
      function knots() {
        var values = [];
        for (var i = 0; i <= state.neurons; i += 1) {
          var x = i / state.neurons;
          values.push({ x: x, y: target(x) });
        }
        return values;
      }
      function approx(x, values) {
        if (x <= 0) return values[0].y + x * (values[1].y - values[0].y) * state.neurons;
        if (x >= 1) {
          var last = values.length - 1;
          return values[last].y + (x - 1) * (values[last].y - values[last - 1].y) * state.neurons;
        }
        var index = Math.min(state.neurons - 1, Math.floor(x * state.neurons));
        var local = x * state.neurons - index;
        return values[index].y * (1 - local) + values[index + 1].y * local;
      }
      function sx(x) { return box.x + (x + 0.1) / 1.3 * box.w; }
      function sy(y) { return box.y + (1.1 - y) / 1.2 * box.h; }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 450);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 450);
        ctx.fillStyle = C.wash;
        ctx.fillRect(sx(1), box.y, sx(1.2) - sx(1), box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        var values = knots();
        var truth = [];
        var model = [];
        var maxError = 0;
        for (var i = 0; i <= 260; i += 1) {
          var x = -0.1 + i / 260 * 1.3;
          var t = target(x);
          var a = approx(x, values);
          truth.push({ x: sx(x), y: sy(t) });
          model.push({ x: sx(x), y: sy(a) });
          if (x >= 0 && x <= 1) maxError = Math.max(maxError, Math.abs(t - a));
        }
        line(ctx, truth, C.axis, 1.7, [6, 4]);
        line(ctx, model, C.blue, 2.7);
        values.forEach(function (p) {
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 5, 0, Math.PI * 2);
          ctx.fillStyle = C.red;
          ctx.fill();
        });
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("область обучения", sx(0.05), box.y + 20);
        ctx.fillText("продолжение", sx(1.02), box.y + 20);
        ui.output.set([
          { label: "изломов", value: String(Math.max(0, state.neurons - 1)) },
          { label: "скрытых ReLU", value: String(state.neurons) },
          { label: "максимальная ошибка", value: maxError.toFixed(3), color: C.blue },
          { label: "за диапазоном", value: "гарантии нет" },
        ]);
      }
      K.segmented(ui.controls, {
        label: "Целевая функция", value: state.target,
        options: [
          { value: "wave", label: "волна" },
          { value: "peak", label: "пик" },
          { value: "load", label: "нагрузка" },
        ],
      }, function (value) { state.target = value; draw(); });
      K.slider(ui.controls, {
        label: "Ширина сети", min: 2, max: 16, step: 1, value: state.neurons,
      }, function (value) { state.neurons = value; draw(); });
      K.slider(ui.controls, {
        label: "Сдвиг функции", min: -0.2, max: 0.2, step: 0.01, value: state.phase,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.phase = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLossEditor(root) {
      var ui = setup(
        root,
        "Кривые показывают цену одного остатка. Выброс меняет оптимальное постоянное предсказание для маленькой выборки.",
        "Четыре функции потерь и один выброс",
        "Квадрат быстро растёт, модуль сохраняет линейную цену, Huber меняет режим у δ. Асимметрия сдвигает оптимум к выбранному квантилю.",
        465,
      );
      var state = { outlier: 18, delta: 3, tau: 0.75 };
      var box = { x: 68, y: 42, w: 804, h: 280 };
      function losses(e) {
        var abs = Math.abs(e);
        return {
          square: e * e / 12,
          absolute: abs,
          huber: abs <= state.delta ? 0.5 * e * e : state.delta * (abs - 0.5 * state.delta),
          quantile: e >= 0 ? state.tau * e : (state.tau - 1) * e,
        };
      }
      function sx(e) { return box.x + (e + 12) / 24 * box.w; }
      function sy(v) { return box.y + box.h - clamp(v, 0, 28) / 28 * box.h; }
      function median(values) {
        var sorted = values.slice().sort(function (a, b) { return a - b; });
        return sorted[Math.floor(sorted.length / 2)];
      }
      function quantile(values, tau) {
        var sorted = values.slice().sort(function (a, b) { return a - b; });
        return sorted[Math.min(sorted.length - 1, Math.floor(tau * sorted.length))];
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 465);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 465);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        line(ctx, [{ x: sx(0), y: box.y }, { x: sx(0), y: box.y + box.h }], C.axis, 1);
        var curves = { square: [], absolute: [], huber: [], quantile: [] };
        for (var i = 0; i <= 240; i += 1) {
          var e = -12 + i / 240 * 24;
          var value = losses(e);
          Object.keys(curves).forEach(function (key) {
            curves[key].push({ x: sx(e), y: sy(value[key]) });
          });
        }
        line(ctx, curves.square, C.red, 2.4);
        line(ctx, curves.absolute, C.blue, 2.2);
        line(ctx, curves.huber, C.green, 2.2);
        line(ctx, curves.quantile, C.gold, 2.2);
        ctx.font = "12px system-ui, sans-serif";
        [["квадрат", C.red], ["модуль", C.blue], ["Huber", C.green], ["квантиль", C.gold]]
          .forEach(function (item, index) {
            ctx.fillStyle = item[1];
            ctx.fillText(item[0], box.x + 14 + index * 102, box.y + 20);
          });
        var sample = [0, 1, 1, 2, state.outlier];
        var mean = sample.reduce(function (a, b) { return a + b; }, 0) / sample.length;
        var med = median(sample);
        var q = quantile(sample, state.tau);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, 354, box.w, 56);
        sample.forEach(function (value) {
          ctx.beginPath();
          ctx.arc(box.x + 32 + value / 30 * (box.w - 64), 382, 5, 0, Math.PI * 2);
          ctx.fillStyle = value === state.outlier ? C.red : C.ink;
          ctx.fill();
        });
        ui.output.set([
          { label: "среднее / MSE", value: mean.toFixed(1), color: C.red },
          { label: "медиана / MAE", value: med.toFixed(1), color: C.blue },
          { label: "квантиль τ", value: q.toFixed(1), color: C.gold },
          { label: "выборка", value: sample.join(", ") },
        ]);
      }
      K.slider(ui.controls, {
        label: "Величина выброса", min: 3, max: 30, step: 1, value: state.outlier,
      }, function (value) { state.outlier = value; draw(); });
      K.slider(ui.controls, {
        label: "Порог Huber δ", min: 0.5, max: 8, step: 0.5, value: state.delta,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.delta = value; draw(); });
      K.slider(ui.controls, {
        label: "Асимметрия τ", min: 0.1, max: 0.9, step: 0.05, value: state.tau,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.tau = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLandscapes(root) {
      var ui = setup(
        root,
        "Выбери рельеф и перемещай точку. Нулевая производная подсвечивается, но тип точки определяется формой рядом.",
        "Стационарная точка, касательная и кривизна",
        "У выпуклой чаши стационарная точка глобальна. Плато x⁴ имеет нулевую вторую производную в минимуме. Двойная впадина хранит локальную структуру.",
        455,
      );
      var state = { type: "convex", x: 0.8 };
      var box = { x: 68, y: 42, w: 804, h: 310 };
      function fn(x) {
        if (state.type === "convex") return 0.45 * x * x;
        if (state.type === "flat") return 0.055 * Math.pow(x, 4);
        if (state.type === "double") return 0.16 * Math.pow(x * x - 2.2, 2);
        return 0.18 * x * x * x;
      }
      function d1(x) {
        if (state.type === "convex") return 0.9 * x;
        if (state.type === "flat") return 0.22 * Math.pow(x, 3);
        if (state.type === "double") return 0.64 * x * (x * x - 2.2);
        return 0.54 * x * x;
      }
      function d2(x) {
        if (state.type === "convex") return 0.9;
        if (state.type === "flat") return 0.66 * x * x;
        if (state.type === "double") return 1.92 * x * x - 1.408;
        return 1.08 * x;
      }
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (4.4 - y) / 5.4 * box.h; }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 455);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 455);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        line(ctx, [{ x: sx(0), y: box.y }, { x: sx(0), y: box.y + box.h }], C.grid, 1);
        line(ctx, [{ x: box.x, y: sy(0) }, { x: box.x + box.w, y: sy(0) }], C.axis, 1);
        var curve = [];
        for (var i = 0; i <= 260; i += 1) {
          var x = -3 + i / 260 * 6;
          curve.push({ x: sx(x), y: sy(fn(x)) });
        }
        line(ctx, curve, C.blue, 2.7);
        var y0 = fn(state.x);
        var slope = d1(state.x);
        var tangent = [
          { x: state.x - 1.2, y: y0 - 1.2 * slope },
          { x: state.x + 1.2, y: y0 + 1.2 * slope },
        ];
        line(ctx, tangent.map(function (p) { return { x: sx(p.x), y: sy(p.y) }; }), C.red, 1.7, [6, 4]);
        ctx.beginPath();
        ctx.arc(sx(state.x), sy(y0), 8, 0, Math.PI * 2);
        ctx.fillStyle = Math.abs(slope) < 0.06 ? C.gold : C.red;
        ctx.fill();
        var curvature = d2(state.x);
        var type = Math.abs(slope) > 0.06
          ? "не стационарна"
          : curvature > 0.04
            ? "локальный минимум"
            : curvature < -0.04
              ? "локальный максимум"
              : "вторая производная не решила";
        ui.output.set([
          { label: "x", value: state.x.toFixed(2) },
          { label: "f′(x)", value: slope.toFixed(3), color: C.red },
          { label: "f″(x)", value: curvature.toFixed(3) },
          { label: "чтение", value: type },
        ]);
      }
      K.segmented(ui.controls, {
        label: "Рельеф", value: state.type,
        options: [
          { value: "convex", label: "чаша" },
          { value: "flat", label: "плато x⁴" },
          { value: "double", label: "две впадины" },
          { value: "cubic", label: "x³" },
        ],
      }, function (value) { state.type = value; draw(); });
      K.slider(ui.controls, {
        label: "Положение x", min: -3, max: 3, step: 0.02, value: state.x,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.x = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildGradientField(root) {
      var ui = setup(
        root,
        "Перетаскивай красную точку. Синие стрелки показывают спуск, чёрный штрих касается линии уровня.",
        "Поле градиента квадратичной функции",
        "Градиент перпендикулярен линии уровня в координатах выбранной метрики. Изменение масштаба оси вытягивает эллипсы и шаги.",
        470,
      );
      var state = { x: 1.5, y: -0.8, ax: 1, ay: 3 };
      var box = { x: 76, y: 44, w: 760, h: 340 };
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (3 - y) / 6 * box.h; }
      function gradient(x, y) { return { x: 2 * state.ax * x, y: 2 * state.ay * y }; }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 470);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 470);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        [0.8, 1.6, 2.4, 3.2, 4].forEach(function (level) {
          var points = [];
          for (var i = 0; i <= 160; i += 1) {
            var angle = i / 160 * Math.PI * 2;
            var x = level / Math.sqrt(state.ax) * Math.cos(angle);
            var y = level / Math.sqrt(state.ay) * Math.sin(angle);
            points.push({ x: sx(x), y: sy(y) });
          }
          line(ctx, points, C.grid, 1);
        });
        for (var gx = -2.5; gx <= 2.5; gx += 1) {
          for (var gy = -2.5; gy <= 2.5; gy += 1) {
            var g = gradient(gx, gy);
            var norm = Math.hypot(g.x, g.y) || 1;
            arrow(ctx, sx(gx), sy(gy), sx(gx - g.x / norm * 0.32), sy(gy - g.y / norm * 0.32), C.blue);
          }
        }
        var selected = gradient(state.x, state.y);
        var normSelected = Math.hypot(selected.x, selected.y) || 1;
        arrow(
          ctx, sx(state.x), sy(state.y),
          sx(state.x + selected.x / normSelected * 0.9),
          sy(state.y + selected.y / normSelected * 0.9),
          C.red,
        );
        var tangent = { x: -selected.y / normSelected, y: selected.x / normSelected };
        line(ctx, [
          { x: sx(state.x - tangent.x * 0.75), y: sy(state.y - tangent.y * 0.75) },
          { x: sx(state.x + tangent.x * 0.75), y: sy(state.y + tangent.y * 0.75) },
        ], C.ink, 2);
        ctx.beginPath();
        ctx.arc(sx(state.x), sy(state.y), 8, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        ui.output.set([
          { label: "точка", value: "(" + state.x.toFixed(2) + ", " + state.y.toFixed(2) + ")" },
          { label: "градиент", value: "(" + selected.x.toFixed(2) + ", " + selected.y.toFixed(2) + ")", color: C.red },
          { label: "норма", value: normSelected.toFixed(2) },
          { label: "скаляр с касательной", value: "0.00" },
        ]);
      }
      K.slider(ui.controls, {
        label: "Кривизна по x", min: 0.3, max: 5, step: 0.1, value: state.ax,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.ax = value; draw(); });
      K.slider(ui.controls, {
        label: "Кривизна по y", min: 0.3, max: 5, step: 0.1, value: state.ay,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.ay = value; draw(); });
      var removeDrag = K.drag(ui.canvas.canvas, { w: 920, h: 470 }, {
        down: function (point) {
          state.x = clamp((point.x - box.x) / box.w * 6 - 3, -3, 3);
          state.y = clamp(3 - (point.y - box.y) / box.h * 6, -3, 3);
          draw();
        },
        move: function (point) {
          state.x = clamp((point.x - box.x) / box.w * 6 - 3, -3, 3);
          state.y = clamp(3 - (point.y - box.y) / box.h * 6, -3, 3);
          draw();
        },
      });
      ui.setDraw(draw);
      draw();
      return function () {
        removeDrag();
        ui.destroy();
      };
    }

    function buildConstraints(root) {
      var ui = setup(
        root,
        "Перемещай целевую точку и меняй радиус. Ответ — ближайшая допустимая точка. Ромб часто касается линии уровня в углу.",
        "Условный минимум на окружности и L₁-ромбе",
        "Красная стрелка — градиент расстояния, синяя — нормаль ограничения. В гладком касании они параллельны.",
        465,
      );
      var state = { shape: "circle", tx: 2.2, ty: 1.2, radius: 1.25 };
      var box = { x: 82, y: 42, w: 690, h: 340 };
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (3 - y) / 6 * box.h; }
      function optimum() {
        if (state.shape === "circle") {
          var norm = Math.hypot(state.tx, state.ty) || 1;
          return { x: state.tx / norm * state.radius, y: state.ty / norm * state.radius };
        }
        var best = { x: 0, y: 0, d: Infinity };
        for (var i = 0; i <= 800; i += 1) {
          var t = -state.radius + 2 * state.radius * i / 800;
          var ys = [state.radius - Math.abs(t), -state.radius + Math.abs(t)];
          ys.forEach(function (y) {
            var d = Math.pow(t - state.tx, 2) + Math.pow(y - state.ty, 2);
            if (d < best.d) best = { x: t, y: y, d: d };
          });
        }
        return best;
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 465);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 465);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        if (state.shape === "circle") {
          ctx.beginPath();
          ctx.ellipse(sx(0), sy(0), state.radius / 6 * box.w, state.radius / 6 * box.h, 0, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(49,95,140,.08)";
          ctx.fill();
          ctx.strokeStyle = C.blue;
          ctx.lineWidth = 2;
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.moveTo(sx(state.radius), sy(0));
          ctx.lineTo(sx(0), sy(state.radius));
          ctx.lineTo(sx(-state.radius), sy(0));
          ctx.lineTo(sx(0), sy(-state.radius));
          ctx.closePath();
          ctx.fillStyle = "rgba(49,95,140,.08)";
          ctx.fill();
          ctx.strokeStyle = C.blue;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        var opt = optimum();
        [0.45, 0.85, 1.25].forEach(function (radius) {
          ctx.beginPath();
          ctx.ellipse(
            sx(state.tx), sy(state.ty),
            radius / 6 * box.w, radius / 6 * box.h,
            0, 0, Math.PI * 2,
          );
          ctx.strokeStyle = C.grid;
          ctx.lineWidth = 1;
          ctx.stroke();
        });
        ctx.beginPath();
        ctx.arc(sx(state.tx), sy(state.ty), 8, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(sx(opt.x), sy(opt.y), 9, 0, Math.PI * 2);
        ctx.fillStyle = C.gold;
        ctx.fill();
        var gradF = { x: 2 * (opt.x - state.tx), y: 2 * (opt.y - state.ty) };
        var gradG = state.shape === "circle"
          ? { x: 2 * opt.x, y: 2 * opt.y }
          : { x: Math.sign(opt.x), y: Math.sign(opt.y) };
        var nf = Math.hypot(gradF.x, gradF.y) || 1;
        var ng = Math.hypot(gradG.x, gradG.y) || 1;
        arrow(ctx, sx(opt.x), sy(opt.y), sx(opt.x + gradF.x / nf * 0.8), sy(opt.y + gradF.y / nf * 0.8), C.red);
        arrow(ctx, sx(opt.x), sy(opt.y), sx(opt.x + gradG.x / ng * 0.8), sy(opt.y + gradG.y / ng * 0.8), C.blue);
        var zeros = (Math.abs(opt.x) < 0.03 ? 1 : 0) + (Math.abs(opt.y) < 0.03 ? 1 : 0);
        ui.output.set([
          { label: "ответ", value: "(" + opt.x.toFixed(2) + ", " + opt.y.toFixed(2) + ")", color: C.gold },
          { label: "расстояние", value: Math.hypot(opt.x - state.tx, opt.y - state.ty).toFixed(2) },
          { label: "нулевых координат", value: String(zeros) },
          { label: "граница", value: state.shape === "circle" ? "гладкая" : "с углами" },
        ]);
      }
      K.segmented(ui.controls, {
        label: "Допустимая область", value: state.shape,
        options: [{ value: "circle", label: "L₂-круг" }, { value: "l1", label: "L₁-ромб" }],
      }, function (value) { state.shape = value; draw(); });
      K.slider(ui.controls, {
        label: "Цель по x", min: -2.8, max: 2.8, step: 0.1, value: state.tx,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.tx = value; draw(); });
      K.slider(ui.controls, {
        label: "Цель по y", min: -2.8, max: 2.8, step: 0.1, value: state.ty,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.ty = value; draw(); });
      K.slider(ui.controls, {
        label: "Радиус", min: 0.4, max: 2.4, step: 0.1, value: state.radius,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.radius = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildSgd(root) {
      var ui = setup(
        root,
        "Две траектории стартуют вместе. Чёрная использует полный градиент, красная добавляет шум мини-батча.",
        "Полный градиент и SGD на вытянутой долине",
        "Малый батч создаёт заметный разброс шага. Слишком большая скорость раскачивает узкое направление независимо от размера батча.",
        470,
      );
      var state = { batch: 16, eta: 0.12, seed: 3 };
      var box = { x: 80, y: 42, w: 730, h: 340 };
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (3 - y) / 6 * box.h; }
      function grad(x, y) { return { x: 0.7 * x + 0.45 * y, y: 0.45 * x + 3.5 * y }; }
      function path(noisy) {
        var x = -2.5;
        var y = 2.2;
        var points = [{ x: x, y: y }];
        for (var t = 0; t < 55; t += 1) {
          var g = grad(x, y);
          var scale = noisy ? 1.4 / Math.sqrt(state.batch) : 0;
          var nx = (seeded(t + state.seed * 101, 111) - 0.5) * scale;
          var ny = (seeded(t + state.seed * 101, 113) - 0.5) * scale;
          x -= state.eta * (g.x + nx);
          y -= state.eta * (g.y + ny);
          points.push({ x: x, y: y });
          if (Math.abs(x) > 5 || Math.abs(y) > 5) break;
        }
        return points;
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 470);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 470);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        [0.7, 1.2, 1.8, 2.5, 3.2].forEach(function (r) {
          var pts = [];
          for (var i = 0; i <= 160; i += 1) {
            var angle = i / 160 * Math.PI * 2;
            pts.push({ x: sx(r * Math.cos(angle)), y: sy(r * 0.34 * Math.sin(angle) - 0.13 * r * Math.cos(angle)) });
          }
          line(ctx, pts, C.grid, 1);
        });
        var full = path(false);
        var stochastic = path(true);
        line(ctx, full.map(function (p) { return { x: sx(p.x), y: sy(p.y) }; }), C.ink, 2.3);
        line(ctx, stochastic.map(function (p) { return { x: sx(p.x), y: sy(p.y) }; }), C.red, 2.1);
        stochastic.forEach(function (p, index) {
          if (index % 5) return;
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 3.5, 0, Math.PI * 2);
          ctx.fillStyle = C.red;
          ctx.fill();
        });
        var lastFull = full.at(-1);
        var lastSgd = stochastic.at(-1);
        ui.output.set([
          { label: "батч", value: String(state.batch) },
          { label: "полный путь до центра", value: Math.hypot(lastFull.x, lastFull.y).toFixed(3), color: C.ink },
          { label: "SGD до центра", value: Math.hypot(lastSgd.x, lastSgd.y).toFixed(3), color: C.red },
          { label: "шагов", value: String(stochastic.length - 1) },
        ]);
      }
      K.slider(ui.controls, {
        label: "Размер мини-батча", min: 1, max: 256, step: 1, value: state.batch,
      }, function (value) { state.batch = value; draw(); });
      K.slider(ui.controls, {
        label: "Скорость обучения", min: 0.01, max: 0.5, step: 0.01, value: state.eta,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.eta = value; draw(); });
      K.slider(ui.controls, {
        label: "Seed пачек", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBackprop(root) {
      var ui = setup(
        root,
        "Значение идёт слева направо, градиент справа налево. Вторая ветка заставляет узел x сложить два вклада.",
        "Граф вычислений для квадратичной ошибки",
        "Верхняя строка узла хранит значение прямого прохода, нижняя производную loss по этому значению.",
        455,
      );
      var state = { x: 2, w: 1.5, b: -0.5, branch: false };
      function node(ctx, x, y, title, value, grad, color) {
        ctx.fillStyle = C.wash;
        ctx.fillRect(x - 56, y - 34, 112, 68);
        ctx.strokeStyle = color || C.grid;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 56, y - 34, 112, 68);
        ctx.fillStyle = C.ink;
        ctx.font = "600 13px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(title + " = " + value.toFixed(2), x, y - 7);
        ctx.fillStyle = C.red;
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText("∂L/∂ = " + grad.toFixed(2), x, y + 17);
        ctx.textAlign = "left";
      }
      function draw() {
        var u = state.w * state.x;
        var v = u + state.b;
        var branchValue = state.branch ? 0.3 * state.x : 0;
        var total = v + branchValue;
        var loss = total * total;
        var gTotal = 2 * total;
        var gV = gTotal;
        var gU = gV;
        var gBranch = state.branch ? gTotal : 0;
        var gX = gU * state.w + gBranch * 0.3;
        var gW = gU * state.x;
        var gB = gV;
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 455);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 455);
        var positions = {
          x: { x: 95, y: 190 }, w: { x: 95, y: 90 }, u: { x: 285, y: 140 },
          b: { x: 285, y: 270 }, v: { x: 475, y: 190 }, total: { x: 655, y: 190 }, loss: { x: 825, y: 190 },
        };
        arrow(ctx, 151, 175, 229, 150, C.axis);
        arrow(ctx, 151, 105, 229, 130, C.axis);
        arrow(ctx, 341, 150, 419, 180, C.axis);
        arrow(ctx, 341, 260, 419, 205, C.axis);
        arrow(ctx, 531, 190, 599, 190, C.axis);
        arrow(ctx, 711, 190, 769, 190, C.axis);
        if (state.branch) {
          arrow(ctx, 151, 205, 599, 215, C.gold);
          ctx.fillStyle = C.gold;
          ctx.font = "12px system-ui, sans-serif";
          ctx.fillText("0.3x", 400, 237);
        }
        node(ctx, positions.x.x, positions.x.y, "x", state.x, gX, C.blue);
        node(ctx, positions.w.x, positions.w.y, "w", state.w, gW, C.blue);
        node(ctx, positions.u.x, positions.u.y, "u", u, gU);
        node(ctx, positions.b.x, positions.b.y, "b", state.b, gB, C.blue);
        node(ctx, positions.v.x, positions.v.y, "v", v, gV);
        node(ctx, positions.total.x, positions.total.y, "s", total, gTotal);
        node(ctx, positions.loss.x, positions.loss.y, "L", loss, 1, C.red);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("u = wx", 238, 88);
        ctx.fillText("v = u + b", 430, 130);
        ctx.fillText("L = s²", 785, 130);
        ui.output.set([
          { label: "loss", value: loss.toFixed(3), color: C.red },
          { label: "∂L/∂w", value: gW.toFixed(3) },
          { label: "∂L/∂x", value: gX.toFixed(3), color: C.blue },
          { label: "путей к x", value: state.branch ? "2, вклады сложены" : "1" },
        ]);
      }
      K.slider(ui.controls, {
        label: "Вход x", min: -3, max: 3, step: 0.1, value: state.x,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.x = value; draw(); });
      K.slider(ui.controls, {
        label: "Вес w", min: -3, max: 3, step: 0.1, value: state.w,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.w = value; draw(); });
      K.slider(ui.controls, {
        label: "Смещение b", min: -3, max: 3, step: 0.1, value: state.b,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.b = value; draw(); });
      K.segmented(ui.controls, {
        label: "Ветка от x", value: "off",
        options: [{ value: "off", label: "один путь" }, { value: "on", label: "два пути" }],
      }, function (value) { state.branch = value === "on"; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildMinmax(root) {
      var ui = setup(
        root,
        "Сравни одновременный шаг и шаг с прогнозом соперника. Фазовая траектория должна приблизиться к седлу в центре.",
        "Динамика min–max для F(x,y)=xy",
        "Минимизатор двигает x против y, максимизатор двигает y по x. Наивные шаги вращаются; extra-gradient смотрит на промежуточную позицию.",
        465,
      );
      var state = { method: "sim", eta: 0.18, steps: 36 };
      var box = { x: 88, y: 42, w: 650, h: 340 };
      function sx(x) { return box.x + (x + 2.5) / 5 * box.w; }
      function sy(y) { return box.y + (2.5 - y) / 5 * box.h; }
      function trajectory() {
        var x = 1.7;
        var y = 0.3;
        var points = [{ x: x, y: y }];
        var avgX = 0;
        var avgY = 0;
        for (var t = 0; t < state.steps; t += 1) {
          if (state.method === "extra") {
            var midX = x - state.eta * y;
            var midY = y + state.eta * x;
            x -= state.eta * midY;
            y += state.eta * midX;
          } else {
            var nextX = x - state.eta * y;
            var nextY = y + state.eta * x;
            x = nextX;
            y = nextY;
          }
          avgX = (avgX * t + x) / (t + 1);
          avgY = (avgY * t + y) / (t + 1);
          points.push({
            x: state.method === "avg" ? avgX : x,
            y: state.method === "avg" ? avgY : y,
          });
          if (Math.hypot(x, y) > 8) break;
        }
        return points;
      }
      function draw() {
        var ctx = ui.canvas.ctx;
        ctx.clearRect(0, 0, 920, 465);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, 920, 465);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        line(ctx, [{ x: sx(0), y: box.y }, { x: sx(0), y: box.y + box.h }], C.axis, 1);
        line(ctx, [{ x: box.x, y: sy(0) }, { x: box.x + box.w, y: sy(0) }], C.axis, 1);
        var points = trajectory();
        line(ctx, points.map(function (p) { return { x: sx(p.x), y: sy(p.y) }; }), C.red, 2.2);
        points.forEach(function (p, index) {
          if (index % 4) return;
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), index === points.length - 1 ? 6 : 3, 0, Math.PI * 2);
          ctx.fillStyle = index === points.length - 1 ? C.gold : C.red;
          ctx.fill();
        });
        ctx.beginPath();
        ctx.arc(sx(0), sy(0), 8, 0, Math.PI * 2);
        ctx.fillStyle = C.green;
        ctx.fill();
        ctx.fillStyle = C.ink;
        ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
        ctx.fillText("min x", 774, 92);
        ctx.fillText("max y", 774, 214);
        arrow(ctx, 800, 126, 800 - 42, 126, C.blue);
        arrow(ctx, 800, 248, 800, 206, C.red);
        var last = points.at(-1);
        ui.output.set([
          { label: "метод", value: state.method },
          { label: "последняя точка", value: "(" + last.x.toFixed(2) + ", " + last.y.toFixed(2) + ")" },
          { label: "расстояние до седла", value: Math.hypot(last.x, last.y).toFixed(3), color: C.red },
          { label: "шагов", value: String(points.length - 1) },
        ]);
      }
      K.segmented(ui.controls, {
        label: "Динамика", value: state.method,
        options: [
          { value: "sim", label: "одновременно" },
          { value: "extra", label: "extra-gradient" },
          { value: "avg", label: "усреднение" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Шаг η", min: 0.02, max: 0.5, step: 0.01, value: state.eta,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.eta = value; draw(); });
      K.slider(ui.controls, {
        label: "Число шагов", min: 2, max: 100, step: 1, value: state.steps,
      }, function (value) { state.steps = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "13": buildLogicNeuron,
      "14": buildHomeostat,
      "15": buildPerceptron,
      "16": buildXor,
      "17": buildActivation,
      "18": buildLetterNetwork,
      "19": buildApproximation,
      "20": buildLossEditor,
      "21": buildLandscapes,
      "22": buildGradientField,
      "23": buildConstraints,
      "24": buildSgd,
      "25": buildBackprop,
      "26": buildMinmax,
    };

    K.register("g10-neuron", function (root, options) {
      var lesson = String(options.lesson || "");
      var builder = builders[lesson];
      if (!builder) {
        root.appendChild(K.element("p", "kontur-int-error", {
          text: "Для этого урока эксперимент ещё не собран.",
        }));
        return function () {};
      }
      return builder(root);
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
