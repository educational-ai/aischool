// Grade 11, module 1. Probability, statistics and Bayesian inference.
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
      var value = Math.sin(index * 73.913 + salt * 19.731) * 43758.5453;
      return value - Math.floor(value);
    }

    function rgba(hex, alpha) {
      var value = hex.replace("#", "");
      return "rgba(" +
        parseInt(value.slice(0, 2), 16) + "," +
        parseInt(value.slice(2, 4), 16) + "," +
        parseInt(value.slice(4, 6), 16) + "," + alpha + ")";
    }

    function setup(root, hint, label, caption, height) {
      K.hint(root, hint);
      var stage = K.row(root);
      var redraw = function () {};
      var canvasState = K.makeCanvas(stage, 920, height || 470, {
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

    function small(ctx, text, x, y, color, align) {
      ctx.fillStyle = color || C.muted;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = align || "left";
      ctx.fillText(text, x, y);
    }

    function title(ctx, text, x, y) {
      ctx.fillStyle = C.ink;
      ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
      ctx.textAlign = "left";
      ctx.fillText(text, x, y);
    }

    function paper(ctx, height) {
      ctx.clearRect(0, 0, 920, height);
      ctx.fillStyle = C.paper;
      ctx.fillRect(0, 0, 920, height);
    }

    function normalDensity(x) {
      return Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
    }

    function logGamma(z) {
      var coefficients = [
        676.5203681218851, -1259.1392167224028, 771.3234287776531,
        -176.6150291621406, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7,
      ];
      if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
      z -= 1;
      var x = 0.99999999999980993;
      coefficients.forEach(function (coefficient, index) {
        x += coefficient / (z + index + 1);
      });
      var t = z + coefficients.length - 0.5;
      return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }

    function betaDensity(x, a, b) {
      if (x <= 0 || x >= 1) return 0;
      return Math.exp(
        (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)
        - (logGamma(a) + logGamma(b) - logGamma(a + b)),
      );
    }

    function gaussianSolve(matrix, vector) {
      var n = vector.length;
      var augmented = matrix.map(function (row, index) {
        return row.slice().concat([vector[index]]);
      });
      for (var col = 0; col < n; col += 1) {
        var pivot = col;
        for (var row = col + 1; row < n; row += 1) {
          if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
        }
        var swap = augmented[col];
        augmented[col] = augmented[pivot];
        augmented[pivot] = swap;
        var divisor = augmented[col][col] || 1e-9;
        for (var j = col; j <= n; j += 1) augmented[col][j] /= divisor;
        for (var r = 0; r < n; r += 1) {
          if (r === col) continue;
          var factor = augmented[r][col];
          for (var k = col; k <= n; k += 1) augmented[r][k] -= factor * augmented[col][k];
        }
      }
      return augmented.map(function (row) { return row[n]; });
    }

    function buildProbability(root) {
      var ui = setup(
        root,
        "Меняй событие и число бросков. Полупрозрачные столбцы — теория, точки — частоты одной симуляции.",
        "Распределение суммы двух кубиков",
        "Частота события колеблется вокруг вероятности. При росте числа опытов разброс уменьшается, хотя отдельная серия не обязана двигаться монотонно.",
        465,
      );
      var state = { threshold: 10, trials: 120, seed: 4 };
      var plot = { x: 70, y: 62, w: 790, h: 300 };
      var theory = [0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1].map(function (v) { return v / 36; });

      function empirical() {
        var counts = new Array(13).fill(0);
        for (var i = 0; i < state.trials; i += 1) {
          var a = 1 + Math.floor(seeded(i + state.seed * 1009, 1) * 6);
          var b = 1 + Math.floor(seeded(i + state.seed * 1009, 2) * 6);
          counts[a + b] += 1;
        }
        return counts.map(function (count) { return count / state.trials; });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 465);
        title(ctx, "P(X = сумма)", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var empiricalValues = empirical();
        var barW = plot.w / 11;
        var eventTheory = 0;
        var eventEmpirical = 0;
        for (var sum = 2; sum <= 12; sum += 1) {
          var x = plot.x + (sum - 2) * barW;
          var active = sum >= state.threshold;
          if (active) {
            eventTheory += theory[sum];
            eventEmpirical += empiricalValues[sum];
          }
          ctx.fillStyle = active ? rgba(C.red, 0.28) : rgba(C.blue, 0.24);
          var theoryH = theory[sum] / 0.18 * plot.h;
          ctx.fillRect(x + 7, plot.y + plot.h - theoryH, barW - 14, theoryH);
          var empiricalY = plot.y + plot.h - empiricalValues[sum] / 0.18 * plot.h;
          ctx.beginPath();
          ctx.arc(x + barW / 2, empiricalY, 5.5, 0, Math.PI * 2);
          ctx.fillStyle = active ? C.red : C.blue;
          ctx.fill();
          small(ctx, String(sum), x + barW / 2, plot.y + plot.h + 22, C.muted, "center");
        }
        [0.05, 0.10, 0.15].forEach(function (value) {
          var y = plot.y + plot.h - value / 0.18 * plot.h;
          line(ctx, [{ x: plot.x, y: y }, { x: plot.x + plot.w, y: y }], C.grid, 1);
          small(ctx, value.toFixed(2), plot.x - 10, y + 4, C.muted, "right");
        });
        small(ctx, "событие: X ≥ " + state.threshold, plot.x + 12, plot.y + 23, C.red);
        ui.output.set([
          { label: "теория P(A)", value: eventTheory.toFixed(3), color: C.red },
          { label: "частота", value: eventEmpirical.toFixed(3), color: C.blue },
          { label: "ошибка", value: Math.abs(eventEmpirical - eventTheory).toFixed(3) },
          { label: "испытаний", value: String(state.trials) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Порог события X ≥", min: 2, max: 12, step: 1, value: state.threshold,
      }, function (value) { state.threshold = value; draw(); });
      K.slider(ui.controls, {
        label: "Число бросков", min: 10, max: 3000, step: 10, value: state.trials,
      }, function (value) { state.trials = value; draw(); });
      K.slider(ui.controls, {
        label: "Номер серии", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBayesTest(root) {
      var ui = setup(
        root,
        "Каждая клетка — 10 человек из условной популяции 10 000. Красные и золотые клетки соответствуют положительным ответам.",
        "Тест в популяции",
        "Прогностическая ценность меняется вместе с распространённостью. Чувствительность и специфичность при этом могут оставаться прежними.",
        480,
      );
      var state = { prevalence: 0.01, sensitivity: 0.9, specificity: 0.95 };
      var grid = { x: 55, y: 68, cols: 40, rows: 25, cell: 12 };

      function counts() {
        var sick = Math.round(10000 * state.prevalence);
        var tp = Math.round(sick * state.sensitivity);
        var fn = sick - tp;
        var fp = Math.round((10000 - sick) * (1 - state.specificity));
        var tn = 10000 - sick - fp;
        return { sick: sick, tp: tp, fn: fn, fp: fp, tn: tn };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 480);
        title(ctx, "10 000 человек", grid.x, 35);
        var values = counts();
        var buckets = [
          { count: Math.round(values.tp / 10), color: C.red, kind: "TP" },
          { count: Math.round(values.fp / 10), color: C.gold, kind: "FP" },
          { count: Math.round(values.fn / 10), color: C.violet, kind: "FN" },
          { count: 1000, color: C.grid, kind: "TN" },
        ];
        var cells = [];
        buckets.forEach(function (bucket) {
          for (var i = 0; i < bucket.count && cells.length < 1000; i += 1) cells.push(bucket);
        });
        while (cells.length < 1000) cells.push(buckets[3]);
        cells.slice(0, 1000).forEach(function (bucket, index) {
          var col = index % grid.cols;
          var row = Math.floor(index / grid.cols);
          ctx.fillStyle = bucket.color;
          ctx.fillRect(grid.x + col * grid.cell, grid.y + row * grid.cell, grid.cell - 1.5, grid.cell - 1.5);
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(grid.x, grid.y, grid.cols * grid.cell, grid.rows * grid.cell);

        var sideX = 595;
        title(ctx, "Положительный результат", sideX, 35);
        var positive = values.tp + values.fp;
        var posterior = values.tp / (positive || 1);
        [
          { label: "болен, тест +", value: values.tp, color: C.red },
          { label: "здоров, тест +", value: values.fp, color: C.gold },
        ].forEach(function (item, index) {
          var y = 102 + index * 82;
          small(ctx, item.label, sideX, y, C.ink);
          ctx.fillStyle = C.wash;
          ctx.fillRect(sideX, y + 14, 245, 25);
          ctx.fillStyle = item.color;
          ctx.fillRect(sideX, y + 14, 245 * item.value / Math.max(1, positive), 25);
          small(ctx, item.value.toLocaleString("ru-RU"), sideX + 255, y + 33, item.color);
        });
        ctx.beginPath();
        ctx.arc(sideX + 100, 333, 62, -Math.PI / 2, -Math.PI / 2 + posterior * Math.PI * 2);
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 18;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(sideX + 100, 333, 62, -Math.PI / 2 + posterior * Math.PI * 2, Math.PI * 1.5);
        ctx.strokeStyle = C.gold;
        ctx.stroke();
        small(ctx, (posterior * 100).toFixed(1) + "%", sideX + 100, 339, C.ink, "center");
        small(ctx, "болезнь среди +", sideX + 100, 422, C.muted, "center");
        ui.output.set([
          { label: "истинно +", value: String(values.tp), color: C.red },
          { label: "ложно +", value: String(values.fp), color: C.gold },
          { label: "P(болезнь | +)", value: (posterior * 100).toFixed(1) + "%" },
          { label: "всего +", value: String(positive) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Распространённость", min: 0.001, max: 0.3, step: 0.001, value: state.prevalence,
        format: function (v) { return (v * 100).toFixed(1); }, unit: "%",
      }, function (value) { state.prevalence = value; draw(); });
      K.slider(ui.controls, {
        label: "Чувствительность", min: 0.5, max: 1, step: 0.01, value: state.sensitivity,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.sensitivity = value; draw(); });
      K.slider(ui.controls, {
        label: "Специфичность", min: 0.5, max: 1, step: 0.01, value: state.specificity,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.specificity = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildProsecutor(root) {
      var ui = setup(
        root,
        "Размер базы создаёт новые возможности случайного совпадения. Логарифмическая шкала сохраняет видимыми редкие частоты.",
        "Совпадение при поиске по базе",
        "Отношение правдоподобий умножает исходные шансы. Оно не знает, сколько кандидатов просмотрели и почему выбран именно этот человек.",
        470,
      );
      var state = { population: 100000, rate: 0.00001, prior: 0.00002 };
      var plot = { x: 75, y: 72, w: 760, h: 290 };

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 470);
        title(ctx, "Ожидаемые совпадения", plot.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var expectedFalse = state.population * state.rate;
        var probabilityAny = 1 - Math.pow(1 - state.rate, state.population);
        var maxIcons = 100;
        var falseIcons = Math.min(maxIcons - 1, Math.round(expectedFalse));
        for (var index = 0; index < maxIcons; index += 1) {
          var col = index % 20;
          var row = Math.floor(index / 20);
          ctx.beginPath();
          ctx.arc(plot.x + 28 + col * 36, plot.y + 32 + row * 45, 8, 0, Math.PI * 2);
          ctx.fillStyle = index === 0 ? C.red : index <= falseIcons ? C.gold : C.grid;
          ctx.fill();
        }
        small(ctx, "красный: искомый источник", plot.x + 18, plot.y + 262, C.red);
        small(ctx, "золотой: ожидаемое случайное совпадение", plot.x + 230, plot.y + 262, C.gold);
        var lr = 1 / state.rate;
        var posteriorOdds = state.prior / (1 - state.prior) * lr;
        var posterior = posteriorOdds / (1 + posteriorOdds);
        ctx.fillStyle = rgba(C.red, 0.12);
        ctx.fillRect(plot.x + 530, plot.y + 225, 200, 42);
        small(ctx, "posterior " + (posterior * 100).toFixed(2) + "%", plot.x + 630, plot.y + 251, C.red, "center");
        ui.output.set([
          { label: "ложных совпадений, ожидание", value: expectedFalse.toFixed(2), color: C.gold },
          { label: "хотя бы одно", value: (probabilityAny * 100).toFixed(1) + "%" },
          { label: "LR", value: lr.toExponential(1) },
          { label: "posterior при prior", value: (posterior * 100).toFixed(2) + "%", color: C.red },
        ]);
      }

      K.slider(ui.controls, {
        label: "Размер базы", min: 1000, max: 1000000, step: 1000, value: state.population,
      }, function (value) { state.population = value; draw(); });
      K.segmented(ui.controls, {
        label: "Частота совпадения", value: "0.00001",
        options: [
          { value: "0.001", label: "10⁻³" },
          { value: "0.0001", label: "10⁻⁴" },
          { value: "0.00001", label: "10⁻⁵" },
          { value: "0.000001", label: "10⁻⁶" },
        ],
      }, function (value) { state.rate = Number(value); draw(); });
      K.segmented(ui.controls, {
        label: "Исходный шанс H", value: "0.00002",
        options: [
          { value: "0.000001", label: "1 на миллион" },
          { value: "0.00002", label: "1 на 50 тыс." },
          { value: "0.01", label: "1 на 100" },
        ],
      }, function (value) { state.prior = Number(value); draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildClt(root) {
      var ui = setup(
        root,
        "Гистограмма строится по 1 200 стандартизированным средним. Синяя линия — плотность N(0,1).",
        "Центральная предельная теорема",
        "Скорость приближения зависит от исходной формы. Редкие большие значения сохраняют асимметрию дольше.",
        470,
      );
      var state = { distribution: "exp", n: 5, seed: 3 };
      var plot = { x: 70, y: 62, w: 790, h: 315 };

      function drawOne(index, salt) {
        var u = clamp(seeded(index, salt), 1e-6, 1 - 1e-6);
        if (state.distribution === "uniform") return { value: (u - 0.5) * Math.sqrt(12), mean: 0, sd: 1 };
        if (state.distribution === "coin") return { value: u < 0.15 ? 2.3805 : -0.4201, mean: 0, sd: 1 };
        return { value: -Math.log(u) - 1, mean: 0, sd: 1 };
      }

      function samples() {
        var values = [];
        for (var rep = 0; rep < 1200; rep += 1) {
          var sum = 0;
          for (var j = 0; j < state.n; j += 1) {
            sum += drawOne(rep * 211 + j + state.seed * 7001, 15 + j).value;
          }
          values.push(sum / Math.sqrt(state.n));
        }
        return values;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 470);
        title(ctx, "Стандартизированное среднее", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var values = samples();
        var bins = new Array(36).fill(0);
        values.forEach(function (value) {
          var index = Math.floor((clamp(value, -3.99, 3.99) + 4) / 8 * bins.length);
          bins[clamp(index, 0, bins.length - 1)] += 1;
        });
        var maxCount = Math.max.apply(null, bins);
        bins.forEach(function (count, index) {
          var x = plot.x + index / bins.length * plot.w;
          var h = count / maxCount * plot.h * 0.88;
          ctx.fillStyle = rgba(C.gold, 0.5);
          ctx.fillRect(x + 1, plot.y + plot.h - h, plot.w / bins.length - 2, h);
        });
        var curve = [];
        for (var i = 0; i <= 200; i += 1) {
          var z = -4 + i / 200 * 8;
          curve.push({
            x: plot.x + i / 200 * plot.w,
            y: plot.y + plot.h - normalDensity(z) / 0.42 * plot.h * 0.88,
          });
        }
        line(ctx, curve, C.blue, 2.5);
        [-4, -2, 0, 2, 4].forEach(function (z) {
          var x = plot.x + (z + 4) / 8 * plot.w;
          small(ctx, String(z), x, plot.y + plot.h + 22, C.muted, "center");
        });
        var skew = values.reduce(function (sum, value) { return sum + Math.pow(value, 3); }, 0) / values.length;
        ui.output.set([
          { label: "слагаемых", value: String(state.n) },
          { label: "среднее симуляции", value: (values.reduce(function (a, b) { return a + b; }, 0) / values.length).toFixed(3) },
          { label: "асимметрия", value: skew.toFixed(3), color: C.gold },
          { label: "серий", value: "1 200" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Исходное распределение", value: state.distribution,
        options: [
          { value: "exp", label: "экспоненциальное" },
          { value: "uniform", label: "равномерное" },
          { value: "coin", label: "редкий скачок" },
        ],
      }, function (value) { state.distribution = value; draw(); });
      K.slider(ui.controls, {
        label: "Число слагаемых", min: 1, max: 100, step: 1, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.slider(ui.controls, {
        label: "Серия", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLikelihood(root) {
      var ui = setup(
        root,
        "Обе кривые имеют один максимум. Логарифмическая форма показывает ширину даже при большой выборке.",
        "Правдоподобие монеты",
        "При фиксированной доле успехов увеличение n сужает пик. Точка MLE не меняется, информация о параметре растёт.",
        465,
      );
      var state = { n: 40, h: 24 };
      var plot = { x: 70, y: 62, w: 790, h: 300 };

      function logLikelihood(p) {
        if (p <= 0 || p >= 1) return -Infinity;
        return state.h * Math.log(p) + (state.n - state.h) * Math.log(1 - p);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 465);
        title(ctx, "L(p) и log L(p) − max", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var mle = state.h / state.n;
        var maximum = logLikelihood(clamp(mle, 0.0001, 0.9999));
        var likelihoodPoints = [];
        var logPoints = [];
        for (var i = 1; i < 300; i += 1) {
          var p = i / 300;
          var delta = logLikelihood(p) - maximum;
          likelihoodPoints.push({
            x: plot.x + p * plot.w,
            y: plot.y + plot.h - Math.exp(delta) * plot.h * 0.88,
          });
          logPoints.push({
            x: plot.x + p * plot.w,
            y: plot.y + plot.h - clamp((delta + 12) / 12, 0, 1) * plot.h * 0.88,
          });
        }
        line(ctx, likelihoodPoints, C.blue, 2.6);
        line(ctx, logPoints, C.red, 2.2, [6, 4]);
        var mx = plot.x + mle * plot.w;
        line(ctx, [{ x: mx, y: plot.y }, { x: mx, y: plot.y + plot.h }], C.gold, 1.5);
        small(ctx, "L(p)", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "log L, сдвинут", plot.x + 65, plot.y + 22, C.red);
        [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
          small(ctx, p.toFixed(2), plot.x + p * plot.w, plot.y + plot.h + 22, C.muted, "center");
        });
        var information = state.n / (Math.max(0.001, mle * (1 - mle)));
        ui.output.set([
          { label: "MLE", value: mle.toFixed(3), color: C.gold },
          { label: "успехи / n", value: state.h + " / " + state.n },
          { label: "наблюдаемая информация", value: information.toFixed(1) },
          { label: "примерная se", value: Math.sqrt(mle * (1 - mle) / state.n).toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Испытаний n", min: 2, max: 1000, step: 1, value: state.n,
      }, function (value) {
        var fraction = state.h / state.n;
        state.n = value;
        state.h = clamp(Math.round(fraction * value), 0, value);
        hControl.set(state.h);
        draw();
      });
      var hControl = K.slider(ui.controls, {
        label: "Успехов h", min: 0, max: 1000, step: 1, value: state.h,
      }, function (value) {
        state.h = clamp(value, 0, state.n);
        if (value > state.n) hControl.set(state.n);
        draw();
      });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function wilson(k, n, z) {
      var p = k / n;
      var den = 1 + z * z / n;
      var center = (p + z * z / (2 * n)) / den;
      var half = z / den * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
      return [center - half, center + half];
    }

    function buildCoverage(root) {
      var ui = setup(
        root,
        "Каждая строка — новая выборка. Красный интервал промахнулся мимо истинного p.",
        "Покрытие доверительных интервалов",
        "Интервал Вальда теряет покрытие у границ и при малом n. Уилсон корректирует форму без выхода за [0,1].",
        490,
      );
      var state = { p: 0.08, n: 30, method: "wilson", seed: 2 };
      var plot = { x: 70, y: 62, w: 790, h: 350 };

      function binomial(rep) {
        var k = 0;
        for (var i = 0; i < state.n; i += 1) {
          if (seeded(rep * 409 + i + state.seed * 7919, 31) < state.p) k += 1;
        }
        return k;
      }

      function interval(k) {
        var phat = k / state.n;
        if (state.method === "wilson") return wilson(k, state.n, 1.96);
        var se = Math.sqrt(phat * (1 - phat) / state.n);
        return [phat - 1.96 * se, phat + 1.96 * se];
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 490);
        title(ctx, "60 повторений", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var covered = 0;
        var intervals = [];
        for (var rep = 0; rep < 60; rep += 1) {
          var k = binomial(rep);
          var bounds = interval(k);
          var hit = bounds[0] <= state.p && bounds[1] >= state.p;
          if (hit) covered += 1;
          intervals.push({ k: k, low: bounds[0], high: bounds[1], hit: hit });
        }
        var tx = plot.x + state.p * plot.w;
        line(ctx, [{ x: tx, y: plot.y }, { x: tx, y: plot.y + plot.h }], C.gold, 2);
        intervals.forEach(function (item, index) {
          var y = plot.y + 6 + index / 59 * (plot.h - 12);
          var color = item.hit ? C.blue : C.red;
          line(ctx, [
            { x: plot.x + clamp(item.low, 0, 1) * plot.w, y: y },
            { x: plot.x + clamp(item.high, 0, 1) * plot.w, y: y },
          ], color, 1.7);
          ctx.beginPath();
          ctx.arc(plot.x + item.k / state.n * plot.w, y, 2.4, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        });
        [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
          small(ctx, p.toFixed(2), plot.x + p * plot.w, plot.y + plot.h + 22, C.muted, "center");
        });
        ui.output.set([
          { label: "метод", value: state.method === "wilson" ? "Уилсон" : "Вальд" },
          { label: "истинное p", value: state.p.toFixed(3), color: C.gold },
          { label: "покрыто", value: covered + " / 60", color: covered < 52 ? C.red : C.blue },
          { label: "частота покрытия", value: (covered / 60 * 100).toFixed(1) + "%" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Интервал", value: state.method,
        options: [{ value: "wilson", label: "Уилсон" }, { value: "wald", label: "Вальд" }],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Истинное p", min: 0.01, max: 0.99, step: 0.01, value: state.p,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.p = value; draw(); });
      K.slider(ui.controls, {
        label: "Размер n", min: 5, max: 500, step: 1, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.slider(ui.controls, {
        label: "Серия", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function drawBetaCurve(ctx, plot, a, b, color, width, dash) {
      var values = [];
      var maximum = 0;
      for (var i = 1; i < 240; i += 1) {
        var x = i / 240;
        var density = betaDensity(x, a, b);
        maximum = Math.max(maximum, density);
        values.push({ x: x, density: density });
      }
      maximum = Math.max(maximum, 0.001);
      line(ctx, values.map(function (item) {
        return {
          x: plot.x + item.x * plot.w,
          y: plot.y + plot.h - item.density / maximum * plot.h * 0.88,
        };
      }), color, width || 2.3, dash);
    }

    function buildBayesUpdate(root) {
      var ui = setup(
        root,
        "Три кривые масштабированы отдельно по высоте, поэтому сравнивай положение и ширину, а не абсолютный пик.",
        "Prior, likelihood и posterior",
        "Posterior объединяет прежние счётчики и новые наблюдения. Сильный ошибочный prior требует больше данных для исправления.",
        470,
      );
      var state = { preset: "weak", h: 7, t: 3 };
      var plot = { x: 70, y: 72, w: 790, h: 300 };
      var presets = {
        weak: { a: 1, b: 1, label: "Beta(1,1)" },
        center: { a: 12, b: 12, label: "Beta(12,12)" },
        wrong: { a: 2, b: 18, label: "Beta(2,18)" },
      };

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 470);
        title(ctx, "Плотности на p ∈ [0,1]", plot.x, 36);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var prior = presets[state.preset];
        drawBetaCurve(ctx, plot, prior.a, prior.b, C.blue, 2.2);
        drawBetaCurve(ctx, plot, state.h + 1, state.t + 1, C.gold, 2.2, [6, 4]);
        drawBetaCurve(ctx, plot, prior.a + state.h, prior.b + state.t, C.red, 3);
        [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
          small(ctx, p.toFixed(2), plot.x + p * plot.w, plot.y + plot.h + 22, C.muted, "center");
        });
        small(ctx, "prior", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "likelihood", plot.x + 70, plot.y + 22, C.gold);
        small(ctx, "posterior", plot.x + 150, plot.y + 22, C.red);
        var aPost = prior.a + state.h;
        var bPost = prior.b + state.t;
        var mean = aPost / (aPost + bPost);
        var sd = Math.sqrt(aPost * bPost / (Math.pow(aPost + bPost, 2) * (aPost + bPost + 1)));
        ui.output.set([
          { label: "prior", value: prior.label, color: C.blue },
          { label: "данные", value: state.h + " / " + (state.h + state.t) },
          { label: "posterior", value: "Beta(" + aPost + "," + bPost + ")", color: C.red },
          { label: "mean ± sd", value: mean.toFixed(3) + " ± " + sd.toFixed(3) },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Prior", value: state.preset,
        options: [
          { value: "weak", label: "слабый" },
          { value: "center", label: "уверенный 0.5" },
          { value: "wrong", label: "уверенный 0.1" },
        ],
      }, function (value) { state.preset = value; draw(); });
      K.slider(ui.controls, {
        label: "Успехи", min: 0, max: 100, step: 1, value: state.h,
      }, function (value) { state.h = value; draw(); });
      K.slider(ui.controls, {
        label: "Неудачи", min: 0, max: 100, step: 1, value: state.t,
      }, function (value) { state.t = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLaplace(root) {
      var ui = setup(
        root,
        "Слева история исходов, справа posterior. Нижняя шкала сравнивает MLE и вероятность следующего успеха.",
        "Счётчики Beta и следующее испытание",
        "Сглаживание сильнее при малой выборке. При росте n предсказание приближается к наблюдаемой доле.",
        485,
      );
      var state = { a: 1, b: 1, n: 8, h: 8 };
      var plot = { x: 470, y: 72, w: 390, h: 245 };

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        title(ctx, "Наблюдения", 60, 36);
        var cols = 10;
        for (var index = 0; index < state.n; index += 1) {
          var x = 60 + (index % cols) * 34;
          var y = 80 + Math.floor(index / cols) * 34;
          ctx.beginPath();
          ctx.arc(x, y, 11, 0, Math.PI * 2);
          ctx.fillStyle = index < state.h ? C.blue : C.grid;
          ctx.fill();
          small(ctx, index < state.h ? "1" : "0", x, y + 4, C.paper, "center");
        }
        title(ctx, "Posterior", plot.x, 36);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var aPost = state.a + state.h;
        var bPost = state.b + state.n - state.h;
        drawBetaCurve(ctx, plot, aPost, bPost, C.red, 2.8);
        [0, 0.5, 1].forEach(function (p) {
          small(ctx, p.toFixed(1), plot.x + p * plot.w, plot.y + plot.h + 20, C.muted, "center");
        });
        var mle = state.n ? state.h / state.n : 0;
        var predictive = aPost / (aPost + bPost);
        var scale = { x: 80, y: 390, w: 760 };
        line(ctx, [{ x: scale.x, y: scale.y }, { x: scale.x + scale.w, y: scale.y }], C.axis, 2);
        [0, 0.5, 1].forEach(function (p) {
          small(ctx, p.toFixed(1), scale.x + p * scale.w, scale.y + 28, C.muted, "center");
        });
        ctx.beginPath();
        ctx.arc(scale.x + mle * scale.w, scale.y, 9, 0, Math.PI * 2);
        ctx.fillStyle = C.blue;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(scale.x + predictive * scale.w, scale.y, 9, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        small(ctx, "MLE", scale.x + mle * scale.w, scale.y - 17, C.blue, "center");
        small(ctx, "следующий", scale.x + predictive * scale.w, scale.y + 48, C.red, "center");
        ui.output.set([
          { label: "prior", value: "Beta(" + state.a + "," + state.b + ")" },
          { label: "posterior", value: "Beta(" + aPost + "," + bPost + ")", color: C.red },
          { label: "MLE", value: mle.toFixed(3), color: C.blue },
          { label: "P(следующий = 1)", value: predictive.toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Наблюдений n", min: 1, max: 50, step: 1, value: state.n,
      }, function (value) {
        var fraction = state.h / state.n;
        state.n = value;
        state.h = Math.round(fraction * value);
        hSlider.set(state.h);
        draw();
      });
      var hSlider = K.slider(ui.controls, {
        label: "Успехов h", min: 0, max: 50, step: 1, value: state.h,
      }, function (value) {
        state.h = clamp(value, 0, state.n);
        if (value > state.n) hSlider.set(state.n);
        draw();
      });
      K.segmented(ui.controls, {
        label: "Prior", value: "1,1",
        options: [
          { value: "1,1", label: "Beta(1,1)" },
          { value: "2,2", label: "Beta(2,2)" },
          { value: "5,2", label: "Beta(5,2)" },
        ],
      }, function (value) {
        var parts = value.split(",").map(Number);
        state.a = parts[0]; state.b = parts[1]; draw();
      });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function regressionData(state) {
      var points = [];
      for (var i = 0; i < 34; i += 1) {
        var x = -2.8 + i / 33 * 5.6;
        var noise = (seeded(i + state.seed * 101, 61) + seeded(i + state.seed * 101, 62) - 1) * state.sigma;
        points.push({ x: x, y: state.slope * x + 0.5 + noise });
      }
      if (state.outlier) points[30].y += state.outlier;
      return points;
    }

    function fitLine(points, weights, lambda) {
      var s0 = 0;
      var sx = 0;
      var sy = 0;
      var sxx = lambda || 0;
      var sxy = 0;
      points.forEach(function (point, index) {
        var weight = weights ? weights[index] : 1;
        s0 += weight;
        sx += weight * point.x;
        sy += weight * point.y;
        sxx += weight * point.x * point.x;
        sxy += weight * point.x * point.y;
      });
      var det = s0 * sxx - sx * sx || 1e-9;
      return {
        intercept: (sy * sxx - sx * sxy) / det,
        slope: (s0 * sxy - sx * sy) / det,
      };
    }

    function buildGaussianRegression(root) {
      var ui = setup(
        root,
        "Выброс добавляется в правой части диапазона. Гистограмма показывает остатки относительно найденной прямой.",
        "Нормальный шум и MLE-прямая",
        "Квадратичная ошибка поворачивает линию к далёкому остатку. Остатки перестают быть похожими на симметричный шум.",
        475,
      );
      var state = { slope: 1.1, sigma: 0.45, outlier: 0, seed: 2 };
      var box = { x: 62, y: 62, w: 570, h: 315 };
      function sx(x) { return box.x + (x + 3.2) / 6.4 * box.w; }
      function sy(y) { return box.y + (4.2 - y) / 8.4 * box.h; }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 475);
        var points = regressionData(state);
        var fit = fitLine(points);
        title(ctx, "Данные и MLE", box.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        line(ctx, [{ x: sx(0), y: box.y }, { x: sx(0), y: box.y + box.h }], C.grid, 1);
        line(ctx, [{ x: box.x, y: sy(0) }, { x: box.x + box.w, y: sy(0) }], C.grid, 1);
        line(ctx, [
          { x: sx(-3.2), y: sy(fit.intercept - 3.2 * fit.slope) },
          { x: sx(3.2), y: sy(fit.intercept + 3.2 * fit.slope) },
        ], C.red, 2.7);
        points.forEach(function (point, index) {
          ctx.beginPath();
          ctx.arc(sx(point.x), sy(point.y), index === 30 && state.outlier ? 7 : 4.5, 0, Math.PI * 2);
          ctx.fillStyle = index === 30 && state.outlier ? C.gold : C.blue;
          ctx.fill();
        });
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        var residuals = points.map(function (point) { return point.y - fit.intercept - fit.slope * point.x; });
        var hist = { x: 690, y: 86, w: 170, h: 250 };
        title(ctx, "Остатки", hist.x, 34);
        var bins = new Array(12).fill(0);
        residuals.forEach(function (value) {
          var index = clamp(Math.floor((value + 3) / 6 * bins.length), 0, bins.length - 1);
          bins[index] += 1;
        });
        var maxBin = Math.max.apply(null, bins);
        bins.forEach(function (count, index) {
          var h = count / maxBin * hist.h;
          ctx.fillStyle = rgba(C.blue, 0.6);
          ctx.fillRect(hist.x + index / bins.length * hist.w, hist.y + hist.h - h, hist.w / bins.length - 2, h);
        });
        small(ctx, "−3", hist.x, hist.y + hist.h + 20);
        small(ctx, "3", hist.x + hist.w, hist.y + hist.h + 20, C.muted, "right");
        var mse = residuals.reduce(function (sum, value) { return sum + value * value; }, 0) / residuals.length;
        ui.output.set([
          { label: "наклон MLE", value: fit.slope.toFixed(3), color: C.red },
          { label: "истинный наклон", value: state.slope.toFixed(2) },
          { label: "RMSE", value: Math.sqrt(mse).toFixed(3) },
          { label: "средний остаток", value: (residuals.reduce(function (a, b) { return a + b; }, 0) / residuals.length).toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Истинный наклон", min: -2, max: 2, step: 0.05, value: state.slope,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.slope = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум σ", min: 0.05, max: 1.5, step: 0.05, value: state.sigma,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.sigma = value; draw(); });
      K.slider(ui.controls, {
        label: "Высота выброса", min: 0, max: 6, step: 0.1, value: state.outlier,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.outlier = value; draw(); });
      K.slider(ui.controls, {
        label: "Серия", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function basisVector(x, state) {
      if (state.kind === "poly") {
        var vector = [];
        for (var degree = 0; degree <= state.complexity; degree += 1) vector.push(Math.pow(x, degree));
        return vector;
      }
      var result = [1];
      for (var index = 0; index < state.complexity; index += 1) {
        var center = -1 + 2 * index / Math.max(1, state.complexity - 1);
        result.push(Math.exp(-Math.pow(x - center, 2) / (2 * 0.16 * 0.16)));
      }
      return result;
    }

    function fitBasis(points, state) {
      var size = basisVector(0, state).length;
      var matrix = Array.from({ length: size }, function () { return new Array(size).fill(0); });
      var vector = new Array(size).fill(0);
      points.forEach(function (point) {
        var phi = basisVector(point.x, state);
        for (var i = 0; i < size; i += 1) {
          vector[i] += phi[i] * point.y;
          for (var j = 0; j < size; j += 1) matrix[i][j] += phi[i] * phi[j];
        }
      });
      for (var d = 0; d < size; d += 1) matrix[d][d] += 1e-5;
      return gaussianSolve(matrix, vector);
    }

    function buildBasis(root) {
      var ui = setup(
        root,
        "Точки справа от пунктирной границы скрыты при обучении. Сравнивай interpolation и extrapolation отдельно.",
        "Полином и локальный базис",
        "Большая сложность может точно пройти через train и резко ошибиться за его диапазоном. Локальный базис ограничивает область влияния.",
        475,
      );
      var state = { kind: "poly", complexity: 4, noise: 0.12 };
      var box = { x: 62, y: 62, w: 796, h: 315 };
      function target(x) { return Math.sin(2.7 * x) + 0.25 * x; }
      function sx(x) { return box.x + (x + 1.2) / 2.8 * box.w; }
      function sy(y) { return box.y + (2 - y) / 4 * box.h; }

      function training() {
        var values = [];
        for (var i = 0; i < 24; i += 1) {
          var x = -1 + i / 23 * 2;
          values.push({ x: x, y: target(x) + (seeded(i, 72) - 0.5) * state.noise * 2 });
        }
        return values;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 475);
        title(ctx, "Обучение до x = 1, проверка правее", box.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = rgba(C.gold, 0.08);
        ctx.fillRect(sx(1), box.y, sx(1.6) - sx(1), box.h);
        var points = training();
        var weights = fitBasis(points, state);
        var model = [];
        var truth = [];
        var trainError = 0;
        var extraError = 0;
        var extraCount = 0;
        for (var i = 0; i <= 260; i += 1) {
          var x = -1.2 + i / 260 * 2.8;
          var phi = basisVector(x, state);
          var prediction = phi.reduce(function (sum, value, index) { return sum + value * weights[index]; }, 0);
          model.push({ x: sx(x), y: sy(prediction) });
          truth.push({ x: sx(x), y: sy(target(x)) });
          if (x > 1) {
            extraError += Math.pow(prediction - target(x), 2);
            extraCount += 1;
          }
        }
        points.forEach(function (point) {
          var phi = basisVector(point.x, state);
          var prediction = phi.reduce(function (sum, value, index) { return sum + value * weights[index]; }, 0);
          trainError += Math.pow(prediction - point.y, 2);
          ctx.beginPath();
          ctx.arc(sx(point.x), sy(point.y), 4.5, 0, Math.PI * 2);
          ctx.fillStyle = C.blue;
          ctx.fill();
        });
        line(ctx, truth, C.axis, 1.5, [5, 4]);
        line(ctx, model, C.red, 2.7);
        line(ctx, [{ x: sx(1), y: box.y }, { x: sx(1), y: box.y + box.h }], C.gold, 2);
        [-1, 0, 1, 1.5].forEach(function (x) {
          small(ctx, String(x), sx(x), box.y + box.h + 22, C.muted, "center");
        });
        ui.output.set([
          { label: "базис", value: state.kind === "poly" ? "полином" : "RBF" },
          { label: "признаков", value: String(weights.length) },
          { label: "RMSE train", value: Math.sqrt(trainError / points.length).toFixed(3), color: C.blue },
          { label: "RMSE справа", value: Math.sqrt(extraError / extraCount).toFixed(3), color: C.red },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Базис", value: state.kind,
        options: [{ value: "poly", label: "полином" }, { value: "rbf", label: "локальный RBF" }],
      }, function (value) { state.kind = value; draw(); });
      K.slider(ui.controls, {
        label: "Сложность", min: 1, max: 12, step: 1, value: state.complexity,
      }, function (value) { state.complexity = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум", min: 0, max: 0.8, step: 0.01, value: state.noise,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.noise = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildRegularization(root) {
      var ui = setup(
        root,
        "Далёкая точка влияет на OLS и Huber по-разному. Справа показано сжатие двух коррелированных коэффициентов.",
        "Выброс и путь коэффициентов",
        "Робастная ошибка ограничивает влияние остатка. Ridge стабилизирует направление весов, lasso может занулить один из похожих признаков.",
        480,
      );
      var state = { outlier: 4, lambda: 0.3, method: "ridge" };
      var box = { x: 62, y: 64, w: 525, h: 315 };
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (4 - y) / 8 * box.h; }

      function points() {
        var values = [];
        for (var i = 0; i < 28; i += 1) {
          var x = -2.6 + i / 27 * 5.2;
          values.push({ x: x, y: 0.8 * x + 0.4 + (seeded(i, 92) - 0.5) * 0.7 });
        }
        values[26].y += state.outlier;
        return values;
      }

      function robustFit(values) {
        var fit = fitLine(values, null, state.method === "ridge" ? state.lambda * values.length : 0);
        if (state.method !== "huber") return fit;
        for (var iteration = 0; iteration < 8; iteration += 1) {
          var weights = values.map(function (point) {
            var residual = point.y - fit.intercept - fit.slope * point.x;
            return Math.abs(residual) <= 1 ? 1 : 1 / Math.abs(residual);
          });
          fit = fitLine(values, weights, 0);
        }
        return fit;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 480);
        var values = points();
        var fit = robustFit(values);
        title(ctx, "Линия под влиянием точки", box.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        line(ctx, [
          { x: sx(-3), y: sy(fit.intercept - 3 * fit.slope) },
          { x: sx(3), y: sy(fit.intercept + 3 * fit.slope) },
        ], C.red, 2.7);
        values.forEach(function (point, index) {
          ctx.beginPath();
          ctx.arc(sx(point.x), sy(point.y), index === 26 ? 7 : 4.2, 0, Math.PI * 2);
          ctx.fillStyle = index === 26 ? C.gold : C.blue;
          ctx.fill();
        });
        var coeff = { x: 650, y: 90, w: 190, h: 250 };
        title(ctx, "Два похожих признака", coeff.x, 35);
        var shrink = 1 / (1 + state.lambda * 4);
        var w1 = state.method === "lasso" ? Math.max(0, 1.2 - state.lambda * 1.8) : 1.2 * shrink;
        var w2 = state.method === "lasso" ? Math.max(0, 0.7 - state.lambda * 2.2) : 0.7 * shrink;
        [
          { label: "w₁", value: w1, color: C.blue },
          { label: "w₂", value: w2, color: C.red },
        ].forEach(function (item, index) {
          var y = coeff.y + index * 105;
          small(ctx, item.label, coeff.x, y, C.ink);
          ctx.fillStyle = C.wash;
          ctx.fillRect(coeff.x, y + 16, coeff.w, 30);
          ctx.fillStyle = item.color;
          ctx.fillRect(coeff.x, y + 16, coeff.w * item.value / 1.4, 30);
          small(ctx, item.value.toFixed(2), coeff.x + coeff.w + 10, y + 37, item.color);
        });
        ui.output.set([
          { label: "метод", value: state.method },
          { label: "наклон", value: fit.slope.toFixed(3), color: C.red },
          { label: "λ", value: state.lambda.toFixed(2) },
          { label: "нулевых весов", value: String((w1 === 0 ? 1 : 0) + (w2 === 0 ? 1 : 0)) },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Метод", value: state.method,
        options: [
          { value: "ridge", label: "ridge" },
          { value: "lasso", label: "lasso" },
          { value: "huber", label: "Huber" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Высота выброса", min: 0, max: 7, step: 0.1, value: state.outlier,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.outlier = value; draw(); });
      K.slider(ui.controls, {
        label: "Регуляризация λ", min: 0, max: 1, step: 0.01, value: state.lambda,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.lambda = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBayesianRegression(root) {
      var ui = setup(
        root,
        "Серая полоса складывает шум наблюдения и неопределённость весов. Расширяй диапазон x, чтобы уточнить наклон.",
        "Posterior прямых и предсказательная полоса",
        "Точки около нуля хорошо определяют свободный член и слабо определяют наклон. Крайние измерения быстрее сужают веер линий.",
        475,
      );
      var state = { n: 8, spread: 0.7, sigma: 0.35, prior: 1.5 };
      var box = { x: 62, y: 62, w: 796, h: 315 };
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (3.2 - y) / 6.4 * box.h; }

      function posterior() {
        var points = [];
        var s00 = 1 / (state.prior * state.prior);
        var s01 = 0;
        var s11 = 1 / (state.prior * state.prior);
        var v0 = 0;
        var v1 = 0;
        for (var i = 0; i < state.n; i += 1) {
          var x = state.n === 1 ? 0 : -state.spread + i / (state.n - 1) * state.spread * 2;
          var y = 0.45 + 0.9 * x + (seeded(i, 121) - 0.5) * state.sigma * 2;
          points.push({ x: x, y: y });
          var invNoise = 1 / (state.sigma * state.sigma);
          s00 += invNoise;
          s01 += x * invNoise;
          s11 += x * x * invNoise;
          v0 += y * invNoise;
          v1 += x * y * invNoise;
        }
        var det = s00 * s11 - s01 * s01;
        var cov00 = s11 / det;
        var cov01 = -s01 / det;
        var cov11 = s00 / det;
        return {
          points: points,
          mean0: cov00 * v0 + cov01 * v1,
          mean1: cov01 * v0 + cov11 * v1,
          cov00: cov00, cov01: cov01, cov11: cov11,
        };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 475);
        var post = posterior();
        title(ctx, "Линии из posterior", box.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(box.x, box.y, box.w, box.h);
        var upper = [];
        var lower = [];
        var meanLine = [];
        for (var i = 0; i <= 220; i += 1) {
          var x = -3 + i / 220 * 6;
          var mean = post.mean0 + post.mean1 * x;
          var variance = state.sigma * state.sigma + post.cov00 + 2 * x * post.cov01 + x * x * post.cov11;
          var sd = Math.sqrt(Math.max(0, variance));
          upper.push({ x: sx(x), y: sy(mean + 1.64 * sd) });
          lower.push({ x: sx(x), y: sy(mean - 1.64 * sd) });
          meanLine.push({ x: sx(x), y: sy(mean) });
        }
        ctx.fillStyle = rgba(C.blue, 0.13);
        ctx.beginPath();
        upper.forEach(function (point, index) {
          if (index) ctx.lineTo(point.x, point.y); else ctx.moveTo(point.x, point.y);
        });
        lower.slice().reverse().forEach(function (point) { ctx.lineTo(point.x, point.y); });
        ctx.closePath();
        ctx.fill();
        for (var sample = 0; sample < 9; sample += 1) {
          var z0 = (seeded(sample, 131) - 0.5) * 3;
          var z1 = (seeded(sample, 132) - 0.5) * 3;
          var w0 = post.mean0 + z0 * Math.sqrt(post.cov00);
          var conditional = Math.max(0, post.cov11 - post.cov01 * post.cov01 / post.cov00);
          var w1 = post.mean1 + z0 * post.cov01 / Math.sqrt(post.cov00) + z1 * Math.sqrt(conditional);
          line(ctx, [
            { x: sx(-3), y: sy(w0 - 3 * w1) },
            { x: sx(3), y: sy(w0 + 3 * w1) },
          ], rgba(C.axis, 0.4), 1);
        }
        line(ctx, meanLine, C.red, 2.8);
        post.points.forEach(function (point) {
          ctx.beginPath();
          ctx.arc(sx(point.x), sy(point.y), 5, 0, Math.PI * 2);
          ctx.fillStyle = C.blue;
          ctx.fill();
        });
        line(ctx, [{ x: sx(0), y: box.y }, { x: sx(0), y: box.y + box.h }], C.grid, 1);
        line(ctx, [{ x: box.x, y: sy(0) }, { x: box.x + box.w, y: sy(0) }], C.grid, 1);
        var edgeVar = state.sigma * state.sigma + post.cov00 + 6 * post.cov01 + 9 * post.cov11;
        ui.output.set([
          { label: "posterior E[w₀]", value: post.mean0.toFixed(3) },
          { label: "posterior E[w₁]", value: post.mean1.toFixed(3), color: C.red },
          { label: "sd прогноза при x=3", value: Math.sqrt(edgeVar).toFixed(3) },
          { label: "точек", value: String(state.n) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Число точек", min: 2, max: 30, step: 1, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.slider(ui.controls, {
        label: "Диапазон x", min: 0.05, max: 2.8, step: 0.05, value: state.spread,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.spread = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум σ", min: 0.1, max: 1, step: 0.05, value: state.sigma,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.sigma = value; draw(); });
      K.slider(ui.controls, {
        label: "Масштаб prior", min: 0.2, max: 4, step: 0.1, value: state.prior,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.prior = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildClassification(root) {
      var ui = setup(
        root,
        "Поворачивай нормаль и сдвигай порог. Квадратичный признак заменяет прямую окружностью.",
        "Score и геометрия решения",
        "Порог меняет число ошибок без изменения весов. Расширение признаков меняет класс доступных границ.",
        470,
      );
      var state = { angle: 25, threshold: 0, feature: "linear" };
      var box = { x: 82, y: 54, w: 680, h: 340 };
      function sx(x) { return box.x + (x + 3) / 6 * box.w; }
      function sy(y) { return box.y + (3 - y) / 6 * box.h; }

      function labelOf(x, y) {
        return x * x + y * y < 2.5 ? 1 : 0;
      }

      function prediction(x, y) {
        if (state.feature === "quad") return x * x + y * y < 2.5 + state.threshold;
        var angle = state.angle * Math.PI / 180;
        return x * Math.cos(angle) + y * Math.sin(angle) >= state.threshold;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 470);
        title(ctx, "Пространство признаков", box.x, 31);
        var cells = 50;
        for (var row = 0; row < cells; row += 1) {
          for (var col = 0; col < cells; col += 1) {
            var x = -3 + (col + 0.5) / cells * 6;
            var y = 3 - (row + 0.5) / cells * 6;
            ctx.fillStyle = prediction(x, y) ? rgba(C.red, 0.09) : rgba(C.blue, 0.08);
            ctx.fillRect(box.x + col / cells * box.w, box.y + row / cells * box.h, box.w / cells + 0.5, box.h / cells + 0.5);
          }
        }
        var errors = 0;
        var count = 70;
        for (var i = 0; i < count; i += 1) {
          var x = (seeded(i, 141) - 0.5) * 5.5;
          var y = (seeded(i, 142) - 0.5) * 5.5;
          var label = labelOf(x, y);
          if (prediction(x, y) !== Boolean(label)) errors += 1;
          ctx.beginPath();
          ctx.arc(sx(x), sy(y), 5.2, 0, Math.PI * 2);
          ctx.fillStyle = label ? C.red : C.blue;
          ctx.fill();
          ctx.strokeStyle = C.paper;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        if (state.feature === "quad") {
          ctx.beginPath();
          var radius = Math.sqrt(Math.max(0.05, 2.5 + state.threshold));
          ctx.ellipse(sx(0), sy(0), radius / 6 * box.w, radius / 6 * box.h, 0, 0, Math.PI * 2);
          ctx.strokeStyle = C.ink;
          ctx.lineWidth = 2.5;
          ctx.stroke();
        } else {
          var angle = state.angle * Math.PI / 180;
          var dx = -Math.sin(angle);
          var dy = Math.cos(angle);
          var cx = state.threshold * Math.cos(angle);
          var cy = state.threshold * Math.sin(angle);
          line(ctx, [
            { x: sx(cx - 4 * dx), y: sy(cy - 4 * dy) },
            { x: sx(cx + 4 * dx), y: sy(cy + 4 * dy) },
          ], C.ink, 2.5);
          line(ctx, [
            { x: sx(cx), y: sy(cy) },
            { x: sx(cx + Math.cos(angle)), y: sy(cy + Math.sin(angle)) },
          ], C.gold, 3);
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        title(ctx, "класс 1", 795, 128);
        ctx.beginPath(); ctx.arc(812, 154, 7, 0, Math.PI * 2); ctx.fillStyle = C.red; ctx.fill();
        title(ctx, "класс 0", 795, 222);
        ctx.beginPath(); ctx.arc(812, 248, 7, 0, Math.PI * 2); ctx.fillStyle = C.blue; ctx.fill();
        ui.output.set([
          { label: "граница", value: state.feature === "quad" ? "окружность" : "прямая" },
          { label: "порог", value: state.threshold.toFixed(2) },
          { label: "ошибок", value: errors + " / " + count, color: errors > 20 ? C.red : C.blue },
          { label: "accuracy", value: ((count - errors) / count * 100).toFixed(1) + "%" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Признаки", value: state.feature,
        options: [{ value: "linear", label: "x₁, x₂" }, { value: "quad", label: "x₁² + x₂²" }],
      }, function (value) { state.feature = value; draw(); });
      K.slider(ui.controls, {
        label: "Угол нормали", min: 0, max: 180, step: 1, value: state.angle,
        format: function (v) { return String(v); }, unit: "°",
      }, function (value) { state.angle = value; draw(); });
      K.slider(ui.controls, {
        label: "Порог", min: -2, max: 2, step: 0.05, value: state.threshold,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.threshold = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function gaussianScore(x, y, mean, sxv, syv, rho, prior) {
      var dx = x - mean.x;
      var dy = y - mean.y;
      var oneMinus = 1 - rho * rho;
      var quad = (dx * dx / (sxv * sxv) - 2 * rho * dx * dy / (sxv * syv) + dy * dy / (syv * syv)) / oneMinus;
      return -0.5 * quad - Math.log(sxv * syv * Math.sqrt(oneMinus)) + Math.log(prior);
    }

    function ellipse(ctx, mean, sxv, syv, rho, map, color) {
      var angle = 0.5 * Math.atan2(2 * rho * sxv * syv, sxv * sxv - syv * syv);
      var trace = sxv * sxv + syv * syv;
      var disc = Math.sqrt(Math.pow(sxv * sxv - syv * syv, 2) + 4 * rho * rho * sxv * sxv * syv * syv);
      var a = Math.sqrt((trace + disc) / 2) * 1.8;
      var b = Math.sqrt((trace - disc) / 2) * 1.8;
      var points = [];
      for (var i = 0; i <= 100; i += 1) {
        var t = i / 100 * Math.PI * 2;
        var x = mean.x + a * Math.cos(t) * Math.cos(angle) - b * Math.sin(t) * Math.sin(angle);
        var y = mean.y + a * Math.cos(t) * Math.sin(angle) + b * Math.sin(t) * Math.cos(angle);
        points.push({ x: map.sx(x), y: map.sy(y) });
      }
      line(ctx, points, color, 2.2);
    }

    function buildLda(root) {
      var ui = setup(
        root,
        "Фон показывает решение, эллипсы — форму классов. В LDA ковариация общая, в QDA второй класс может растянуться.",
        "Порождающие классы",
        "Редкий prior сдвигает границу. QDA гибче, но оценка отдельной ковариации требует больше данных.",
        475,
      );
      var state = { method: "lda", prior: 0.35, ratio: 1.8 };
      var box = { x: 70, y: 58, w: 700, h: 340 };
      var map = {
        sx: function (x) { return box.x + (x + 3.5) / 7 * box.w; },
        sy: function (y) { return box.y + (3 - y) / 6 * box.h; },
      };
      var m0 = { x: -1.1, y: -0.25 };
      var m1 = { x: 1.25, y: 0.45 };

      function parameters() {
        return state.method === "lda"
          ? { s0x: 1.0, s0y: 0.7, r0: 0.25, s1x: 1.0, s1y: 0.7, r1: 0.25 }
          : { s0x: 0.9, s0y: 0.55, r0: -0.2, s1x: state.ratio, s1y: 0.65, r1: 0.5 };
      }

      function classOne(x, y) {
        var p = parameters();
        var s0 = gaussianScore(x, y, m0, p.s0x, p.s0y, p.r0, 1 - state.prior);
        var s1 = gaussianScore(x, y, m1, p.s1x, p.s1y, p.r1, state.prior);
        return s1 > s0;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 475);
        title(ctx, "Posterior decision", box.x, 32);
        var p = parameters();
        var cells = 60;
        for (var row = 0; row < cells; row += 1) {
          for (var col = 0; col < cells; col += 1) {
            var x = -3.5 + (col + 0.5) / cells * 7;
            var y = 3 - (row + 0.5) / cells * 6;
            ctx.fillStyle = classOne(x, y) ? rgba(C.red, 0.1) : rgba(C.blue, 0.09);
            ctx.fillRect(box.x + col / cells * box.w, box.y + row / cells * box.h, box.w / cells + 0.5, box.h / cells + 0.5);
          }
        }
        ellipse(ctx, m0, p.s0x, p.s0y, p.r0, map, C.blue);
        ellipse(ctx, m1, p.s1x, p.s1y, p.r1, map, C.red);
        for (var i = 0; i < 32; i += 1) {
          var angle = seeded(i, 151) * Math.PI * 2;
          var radius = Math.sqrt(-2 * Math.log(clamp(seeded(i, 152), 0.001, 0.999)));
          var x0 = m0.x + Math.cos(angle) * radius * p.s0x * 0.65;
          var y0 = m0.y + Math.sin(angle) * radius * p.s0y * 0.65;
          ctx.beginPath(); ctx.arc(map.sx(x0), map.sy(y0), 4, 0, Math.PI * 2); ctx.fillStyle = C.blue; ctx.fill();
          var x1 = m1.x + Math.cos(angle) * radius * p.s1x * 0.65;
          var y1 = m1.y + Math.sin(angle) * radius * p.s1y * 0.65;
          ctx.beginPath(); ctx.arc(map.sx(x1), map.sy(y1), 4, 0, Math.PI * 2); ctx.fillStyle = C.red; ctx.fill();
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        small(ctx, "класс 0", 806, 112, C.blue);
        small(ctx, "класс 1", 806, 158, C.red);
        ui.output.set([
          { label: "модель", value: state.method.toUpperCase() },
          { label: "prior класса 1", value: (state.prior * 100).toFixed(0) + "%" },
          { label: "ковариаций", value: state.method === "lda" ? "1 общая" : "2 отдельные" },
          { label: "форма границы", value: state.method === "lda" ? "линейная" : "квадратичная" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Модель", value: state.method,
        options: [{ value: "lda", label: "LDA" }, { value: "qda", label: "QDA" }],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Prior класса 1", min: 0.05, max: 0.95, step: 0.01, value: state.prior,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.prior = value; draw(); });
      K.slider(ui.controls, {
        label: "Растяжение класса 1", min: 0.5, max: 2.5, step: 0.05, value: state.ratio,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.ratio = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildAb(root) {
      var ui = setup(
        root,
        "Каждый отрезок — новый A/B-тест. При подглядывании серия останавливается на первом случайном пересечении порога.",
        "Эффект и преждевременная остановка",
        "Одна заранее назначенная проверка контролирует ложные сигналы. Повторный поиск момента остановки меняет процедуру.",
        485,
      );
      var state = { effect: 0.15, n: 80, peek: false, seed: 2 };
      var plot = { x: 75, y: 60, w: 760, h: 350 };

      function experiment(rep) {
        var finalN = state.n;
        var best = null;
        for (var current = state.peek ? 10 : state.n; current <= state.n; current += state.peek ? 5 : state.n) {
          var meanA = 0;
          var meanB = 0;
          for (var i = 0; i < current; i += 1) {
            meanA += (seeded(rep * 1009 + i, state.seed + 161) + seeded(rep * 1009 + i, state.seed + 162) - 1);
            meanB += state.effect + (seeded(rep * 1009 + i, state.seed + 163) + seeded(rep * 1009 + i, state.seed + 164) - 1);
          }
          meanA /= current;
          meanB /= current;
          var delta = meanB - meanA;
          var se = Math.sqrt(1 / (6 * current) + 1 / (6 * current));
          best = { delta: delta, se: se, n: current };
          if (state.peek && Math.abs(delta / se) > 1.96) {
            finalN = current;
            break;
          }
        }
        best.n = finalN;
        return best;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        title(ctx, "50 независимых экспериментов", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var minX = -0.6;
        var maxX = 0.8;
        function sx(value) { return plot.x + (value - minX) / (maxX - minX) * plot.w; }
        line(ctx, [{ x: sx(0), y: plot.y }, { x: sx(0), y: plot.y + plot.h }], C.axis, 1.7);
        var significant = 0;
        var avgN = 0;
        for (var rep = 0; rep < 50; rep += 1) {
          var result = experiment(rep);
          var low = result.delta - 1.96 * result.se;
          var high = result.delta + 1.96 * result.se;
          var hit = low > 0 || high < 0;
          if (hit) significant += 1;
          avgN += result.n;
          var y = plot.y + 7 + rep / 49 * (plot.h - 14);
          line(ctx, [{ x: sx(low), y: y }, { x: sx(high), y: y }], hit ? C.red : C.blue, 1.8);
          ctx.beginPath(); ctx.arc(sx(result.delta), y, 2.8, 0, Math.PI * 2);
          ctx.fillStyle = hit ? C.red : C.blue; ctx.fill();
        }
        [-0.5, 0, 0.5].forEach(function (value) {
          small(ctx, value.toFixed(1), sx(value), plot.y + plot.h + 22, C.muted, "center");
        });
        ui.output.set([
          { label: "истинный эффект", value: state.effect.toFixed(2), color: C.gold },
          { label: "значимых", value: significant + " / 50", color: C.red },
          { label: "средний n на группу", value: (avgN / 50).toFixed(1) },
          { label: "правило", value: state.peek ? "ежедневно смотреть" : "один финальный тест" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Истинный эффект", min: 0, max: 0.6, step: 0.01, value: state.effect,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.effect = value; draw(); });
      K.slider(ui.controls, {
        label: "Максимальный n", min: 20, max: 400, step: 10, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.segmented(ui.controls, {
        label: "Остановка", value: "fixed",
        options: [{ value: "fixed", label: "по плану" }, { value: "peek", label: "при p < 0.05" }],
      }, function (value) { state.peek = value === "peek"; draw(); });
      K.slider(ui.controls, {
        label: "Серия", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildUncertainty(root) {
      var ui = setup(
        root,
        "Порог решает, какие объекты модель отдаёт человеку. Сдвиг переносит будущие точки из плотной области.",
        "Карта отказа и coverage–risk",
        "С ростом порога автоматических решений меньше. Их ошибка обычно падает, пока сдвиг не разрушает калибровку уверенности.",
        485,
      );
      var state = { threshold: 0.68, shift: 0.6, cost: 3 };
      var map = { x: 60, y: 62, w: 560, h: 330 };
      var curve = { x: 685, y: 90, w: 175, h: 230 };

      function confidence(x, y) {
        var distance = Math.hypot(x, y);
        return clamp(0.96 - 0.18 * distance - 0.06 * Math.abs(x - y), 0.05, 0.98);
      }

      function truth(x, y) { return x + 0.55 * y > 0; }
      function prediction(x, y) { return x + 0.45 * y + 0.18 > 0; }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        title(ctx, "Будущие объекты", map.x, 35);
        var cells = 45;
        for (var row = 0; row < cells; row += 1) {
          for (var col = 0; col < cells; col += 1) {
            var x = -2.8 + (col + 0.5) / cells * 5.6;
            var y = 2.6 - (row + 0.5) / cells * 5.2;
            var conf = confidence(x, y);
            ctx.fillStyle = conf >= state.threshold ? rgba(C.blue, 0.04 + conf * 0.08) : rgba(C.gold, 0.12);
            ctx.fillRect(map.x + col / cells * map.w, map.y + row / cells * map.h, map.w / cells + 0.3, map.h / cells + 0.3);
          }
        }
        var automated = 0;
        var errors = 0;
        var manual = 0;
        for (var i = 0; i < 80; i += 1) {
          var x = (seeded(i, 171) - 0.5) * 4 + state.shift;
          var y = (seeded(i, 172) - 0.5) * 3.8;
          var conf = confidence(x, y);
          var auto = conf >= state.threshold;
          var wrong = truth(x, y) !== prediction(x, y);
          if (auto) {
            automated += 1;
            if (wrong) errors += 1;
          } else manual += 1;
          ctx.beginPath();
          ctx.arc(map.x + (x + 2.8) / 5.6 * map.w, map.y + (2.6 - y) / 5.2 * map.h, auto ? 4.5 : 6, 0, Math.PI * 2);
          ctx.fillStyle = auto ? (wrong ? C.red : C.blue) : C.gold;
          ctx.fill();
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(map.x, map.y, map.w, map.h);
        title(ctx, "Coverage–risk", curve.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(curve.x, curve.y, curve.w, curve.h);
        var riskPoints = [];
        for (var q = 0.2; q <= 0.95; q += 0.02) {
          var accepted = 0;
          var wrongCount = 0;
          for (var j = 0; j < 500; j += 1) {
            var px = (seeded(j, 173) - 0.5) * 4 + state.shift;
            var py = (seeded(j, 174) - 0.5) * 3.8;
            if (confidence(px, py) >= q) {
              accepted += 1;
              if (truth(px, py) !== prediction(px, py)) wrongCount += 1;
            }
          }
          var coverage = accepted / 500;
          var risk = wrongCount / Math.max(1, accepted);
          riskPoints.push({
            x: curve.x + coverage * curve.w,
            y: curve.y + curve.h - risk / 0.3 * curve.h,
          });
        }
        line(ctx, riskPoints, C.red, 2.5);
        small(ctx, "0", curve.x, curve.y + curve.h + 20);
        small(ctx, "coverage 1", curve.x + curve.w, curve.y + curve.h + 20, C.muted, "right");
        var risk = errors / Math.max(1, automated);
        var expectedCost = errors * state.cost + manual;
        ui.output.set([
          { label: "автоматически", value: automated + " / 80", color: C.blue },
          { label: "ручная очередь", value: String(manual), color: C.gold },
          { label: "ошибка авто", value: (risk * 100).toFixed(1) + "%", color: C.red },
          { label: "условная стоимость", value: String(expectedCost) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Порог уверенности", min: 0.2, max: 0.95, step: 0.01, value: state.threshold,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.threshold = value; draw(); });
      K.slider(ui.controls, {
        label: "Сдвиг будущих данных", min: 0, max: 2.2, step: 0.05, value: state.shift,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.shift = value; draw(); });
      K.slider(ui.controls, {
        label: "Цена ошибки", min: 1, max: 10, step: 1, value: state.cost,
      }, function (value) { state.cost = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "41": buildProbability,
      "42": buildBayesTest,
      "43": buildProsecutor,
      "44": buildClt,
      "45": buildLikelihood,
      "46": buildCoverage,
      "47": buildBayesUpdate,
      "48": buildLaplace,
      "49": buildGaussianRegression,
      "50": buildBasis,
      "51": buildRegularization,
      "52": buildBayesianRegression,
      "53": buildClassification,
      "54": buildLda,
      "55": buildAb,
      "56": buildUncertainty,
    };

    K.register("g11-stats", function (root, options) {
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
