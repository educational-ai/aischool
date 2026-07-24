// Grade 11, module 2. Statistical learning and optimization.
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
      var value = Math.sin(index * 61.337 + salt * 29.117) * 43758.5453;
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
      var canvasState = K.makeCanvas(stage, 920, height || 475, {
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

    function title(ctx, text, x, y) {
      ctx.fillStyle = C.ink;
      ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
      ctx.textAlign = "left";
      ctx.fillText(text, x, y);
    }

    function small(ctx, text, x, y, color, align) {
      ctx.fillStyle = color || C.muted;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = align || "left";
      ctx.fillText(text, x, y);
    }

    function paper(ctx, height) {
      ctx.clearRect(0, 0, 920, height);
      ctx.fillStyle = C.paper;
      ctx.fillRect(0, 0, 920, height);
    }

    function buildRisk(root) {
      var ui = setup(
        root,
        "Порог идёт по шкале вероятности. Кривые показывают accuracy, cross-entropy и цену решения для одной модели.",
        "Вероятность и решение",
        "Cross-entropy оценивает распределение вероятностей. Стоимость выбирает действие после прогноза и может требовать другого порога.",
        470,
      );
      var state = { threshold: 0.5, fp: 1, fn: 4, calibration: 0 };
      var plot = { x: 70, y: 65, w: 790, h: 300 };

      function objects() {
        var result = [];
        for (var i = 0; i < 500; i += 1) {
          var latent = seeded(i, 11);
          var truthP = 0.05 + 0.9 * latent;
          var observed = clamp(truthP + state.calibration * (truthP - 0.5), 0.01, 0.99);
          var label = seeded(i, 12) < truthP ? 1 : 0;
          result.push({ p: observed, y: label });
        }
        return result;
      }

      function metrics(threshold, data) {
        var correct = 0;
        var cost = 0;
        var ce = 0;
        data.forEach(function (item) {
          var prediction = item.p >= threshold ? 1 : 0;
          if (prediction === item.y) correct += 1;
          else cost += item.y ? state.fn : state.fp;
          ce -= item.y * Math.log(item.p) + (1 - item.y) * Math.log(1 - item.p);
        });
        return { accuracy: correct / data.length, cost: cost / data.length, ce: ce / data.length };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 470);
        title(ctx, "Метрики по порогу", plot.x, 36);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var data = objects();
        var accuracy = [];
        var cost = [];
        for (var i = 0; i <= 100; i += 1) {
          var threshold = i / 100;
          var m = metrics(threshold, data);
          accuracy.push({ x: plot.x + threshold * plot.w, y: plot.y + plot.h - m.accuracy * plot.h });
          cost.push({ x: plot.x + threshold * plot.w, y: plot.y + plot.h - clamp(m.cost / 1.8, 0, 1) * plot.h });
        }
        line(ctx, accuracy, C.blue, 2.6);
        line(ctx, cost, C.red, 2.6);
        var markerX = plot.x + state.threshold * plot.w;
        line(ctx, [{ x: markerX, y: plot.y }, { x: markerX, y: plot.y + plot.h }], C.gold, 2);
        small(ctx, "accuracy", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "цена / 1.8", plot.x + 92, plot.y + 22, C.red);
        [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
          small(ctx, v.toFixed(2), plot.x + v * plot.w, plot.y + plot.h + 22, C.muted, "center");
        });
        var current = metrics(state.threshold, data);
        var optimal = state.fp / (state.fp + state.fn);
        ui.output.set([
          { label: "accuracy", value: (current.accuracy * 100).toFixed(1) + "%", color: C.blue },
          { label: "средняя цена", value: current.cost.toFixed(3), color: C.red },
          { label: "cross-entropy", value: current.ce.toFixed(3) },
          { label: "порог по цене", value: optimal.toFixed(3), color: C.gold },
        ]);
      }

      K.slider(ui.controls, {
        label: "Порог", min: 0.01, max: 0.99, step: 0.01, value: state.threshold,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.threshold = value; draw(); });
      K.slider(ui.controls, {
        label: "Цена false positive", min: 1, max: 10, step: 1, value: state.fp,
      }, function (value) { state.fp = value; draw(); });
      K.slider(ui.controls, {
        label: "Цена false negative", min: 1, max: 10, step: 1, value: state.fn,
      }, function (value) { state.fn = value; draw(); });
      K.slider(ui.controls, {
        label: "Ошибка калибровки", min: -0.8, max: 0.8, step: 0.05, value: state.calibration,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.calibration = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildErm(root) {
      var ui = setup(
        root,
        "Синие точки — истинный риск кандидатов, золотые — оценка на выборке. Красная рамка отмечает выбранный минимум.",
        "Монте-Карло и выбор победителя",
        "Оценка фиксированной функции колеблется без систематического сдвига. Минимум среди многих получает преимущество от отрицательной случайной ошибки.",
        475,
      );
      var state = { n: 100, candidates: 12, seed: 3 };
      var plot = { x: 65, y: 62, w: 800, h: 320 };

      function candidates() {
        var result = [];
        for (var i = 0; i < state.candidates; i += 1) {
          var truth = 0.17 + 0.002 * i + 0.018 * Math.pow((i - state.candidates * 0.55) / state.candidates, 2);
          var noise = (seeded(i + state.seed * 113, 21) + seeded(i + state.seed * 113, 22) - 1) / Math.sqrt(state.n) * 0.45;
          result.push({ trueRisk: truth, empirical: clamp(truth + noise, 0.02, 0.5) });
        }
        return result;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 475);
        title(ctx, "Риск кандидатов", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var values = candidates();
        var winner = values.reduce(function (best, item, index) {
          return item.empirical < best.item.empirical ? { item: item, index: index } : best;
        }, { item: values[0], index: 0 });
        var oracle = values.reduce(function (best, item, index) {
          return item.trueRisk < best.item.trueRisk ? { item: item, index: index } : best;
        }, { item: values[0], index: 0 });
        values.forEach(function (item, index) {
          var x = plot.x + (index + 0.5) / values.length * plot.w;
          var trueY = plot.y + (item.trueRisk - 0.1) / 0.22 * plot.h;
          var empiricalY = plot.y + (item.empirical - 0.1) / 0.22 * plot.h;
          line(ctx, [{ x: x, y: trueY }, { x: x, y: empiricalY }], C.grid, 1.3);
          ctx.beginPath(); ctx.arc(x, trueY, 5, 0, Math.PI * 2); ctx.fillStyle = C.blue; ctx.fill();
          ctx.beginPath(); ctx.arc(x, empiricalY, 5, 0, Math.PI * 2); ctx.fillStyle = C.gold; ctx.fill();
          if (index === winner.index) {
            ctx.strokeStyle = C.red; ctx.lineWidth = 2;
            ctx.strokeRect(x - 11, Math.min(trueY, empiricalY) - 12, 22, Math.abs(trueY - empiricalY) + 24);
          }
        });
        small(ctx, "истинный", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "эмпирический", plot.x + 80, plot.y + 22, C.gold);
        ui.output.set([
          { label: "выбран", value: "#" + (winner.index + 1), color: C.red },
          { label: "oracle", value: "#" + (oracle.index + 1), color: C.blue },
          { label: "validation победителя", value: winner.item.empirical.toFixed(3) },
          { label: "истинный риск победителя", value: winner.item.trueRisk.toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Размер выборки", min: 20, max: 2000, step: 10, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.slider(ui.controls, {
        label: "Кандидатов", min: 2, max: 40, step: 1, value: state.candidates,
      }, function (value) { state.candidates = value; draw(); });
      K.slider(ui.controls, {
        label: "Серия", min: 1, max: 30, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildDoubleDescent(root) {
      var ui = setup(
        root,
        "Вертикальная линия отмечает отношение параметров к объектам 1. Шум меток усиливает интерполяционный пик.",
        "Классическая и современная кривые сложности",
        "После порога минимальная норма выбирает одно из многих интерполирующих решений. Форма второго спуска зависит от данных и алгоритма.",
        470,
      );
      var state = { ratio: 1.4, noise: 0.18, n: 120 };
      var plot = { x: 70, y: 62, w: 790, h: 305 };

      function errors(ratio) {
        var train = ratio < 1 ? 0.35 * Math.pow(1 - ratio, 1.4) + 0.01 : 0.004;
        var bias = 0.28 / (1 + ratio * 3);
        var peak = state.noise * 0.42 / (Math.abs(ratio - 1) + 0.075 + 20 / state.n);
        var over = ratio > 1 ? 0.08 / Math.sqrt(ratio) : 0;
        var test = clamp(0.08 + bias + peak * Math.exp(-Math.abs(ratio - 1) * 2.5) + over, 0, 0.85);
        return { train: train, test: test };
      }

      function sx(ratio) { return plot.x + ratio / 3 * plot.w; }
      function sy(error) { return plot.y + plot.h - error / 0.85 * plot.h; }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 470);
        title(ctx, "Ошибка по p / n", plot.x, 34);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var train = [];
        var test = [];
        for (var i = 0; i <= 300; i += 1) {
          var ratio = i / 100;
          var e = errors(ratio);
          train.push({ x: sx(ratio), y: sy(e.train) });
          test.push({ x: sx(ratio), y: sy(e.test) });
        }
        line(ctx, train, C.blue, 2.6);
        line(ctx, test, C.red, 2.6);
        line(ctx, [{ x: sx(1), y: plot.y }, { x: sx(1), y: plot.y + plot.h }], C.gold, 2, [5, 4]);
        var marker = errors(state.ratio);
        var mx = sx(state.ratio);
        ctx.beginPath(); ctx.arc(mx, sy(marker.test), 7, 0, Math.PI * 2); ctx.fillStyle = C.red; ctx.fill();
        small(ctx, "train", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "test", plot.x + 65, plot.y + 22, C.red);
        [0, 1, 2, 3].forEach(function (v) {
          small(ctx, String(v), sx(v), plot.y + plot.h + 22, C.muted, "center");
        });
        ui.output.set([
          { label: "параметров / объектов", value: state.ratio.toFixed(2) },
          { label: "train", value: marker.train.toFixed(3), color: C.blue },
          { label: "test", value: marker.test.toFixed(3), color: C.red },
          { label: "режим", value: state.ratio < 1 ? "до интерполяции" : "после интерполяции" },
        ]);
      }

      K.slider(ui.controls, {
        label: "p / n", min: 0.05, max: 3, step: 0.01, value: state.ratio,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.ratio = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум меток", min: 0, max: 0.5, step: 0.01, value: state.noise,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      K.slider(ui.controls, {
        label: "Объектов n", min: 30, max: 1000, step: 10, value: state.n,
      }, function (value) { state.n = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function sphereFraction(d) {
      function logGamma(z) {
        var coefficients = [
          676.5203681218851,
          -1259.1392167224028,
          771.3234287776531,
          -176.6150291621406,
          12.507343278686905,
          -0.13857109526572012,
          9.984369578019572e-6,
          1.5056327351493116e-7,
        ];
        if (z < 0.5) {
          return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
        }
        var x = 0.9999999999998099;
        var shifted = z - 1;
        coefficients.forEach(function (coefficient, index) {
          x += coefficient / (shifted + index + 1);
        });
        var t = shifted + coefficients.length - 0.5;
        return 0.5 * Math.log(2 * Math.PI)
          + (shifted + 0.5) * Math.log(t)
          - t
          + Math.log(x);
      }
      return Math.exp(d / 2 * Math.log(Math.PI) - d * Math.log(2) - logGamma(d / 2 + 1));
    }

    function buildDimension(root) {
      var ui = setup(
        root,
        "Доля шара показана по логарифмической шкале. Отношение расстояний оценено на одной детерминированной симуляции.",
        "Размерность, объём и соседство",
        "Сетка растёт экспоненциально, а расстояния концентрируются. Полезная низкоразмерная структура должна быть частью модели.",
        480,
      );
      var state = { d: 12, resolution: 10, points: 400 };
      var left = { x: 65, y: 70, w: 500, h: 290 };
      var right = { x: 650, y: 92, w: 200, h: 235 };

      function distanceStats() {
        var distances = [];
        for (var i = 0; i < state.points; i += 1) {
          var sum = 0;
          for (var j = 0; j < state.d; j += 1) {
            var value = seeded(i * 211 + j, 41) - 0.5;
            sum += value * value;
          }
          distances.push(Math.sqrt(sum));
        }
        return {
          min: Math.min.apply(null, distances),
          mean: distances.reduce(function (a, b) { return a + b; }, 0) / distances.length,
          max: Math.max.apply(null, distances),
        };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 480);
        title(ctx, "Доля единичного шара в кубе", left.x, 36);
        ctx.fillStyle = C.wash; ctx.fillRect(left.x, left.y, left.w, left.h);
        var curve = [];
        for (var d = 2; d <= 80; d += 1) {
          var fraction = Math.max(1e-30, sphereFraction(d));
          curve.push({
            x: left.x + (d - 2) / 78 * left.w,
            y: left.y + (-Math.log10(fraction)) / 30 * left.h,
          });
        }
        line(ctx, curve, C.blue, 2.6);
        var currentFraction = Math.max(1e-30, sphereFraction(state.d));
        var cx = left.x + (state.d - 2) / 78 * left.w;
        var cy = left.y + (-Math.log10(currentFraction)) / 30 * left.h;
        ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fillStyle = C.gold; ctx.fill();
        [2, 20, 40, 60, 80].forEach(function (d) {
          small(ctx, String(d), left.x + (d - 2) / 78 * left.w, left.y + left.h + 20, C.muted, "center");
        });
        title(ctx, "Расстояния от центра", right.x, 36);
        var stats = distanceStats();
        [
          { label: "ближайшее", value: stats.min, color: C.green },
          { label: "среднее", value: stats.mean, color: C.blue },
          { label: "дальнее", value: stats.max, color: C.red },
        ].forEach(function (item, index) {
          var y = right.y + index * 72;
          small(ctx, item.label, right.x, y, C.ink);
          ctx.fillStyle = C.wash; ctx.fillRect(right.x, y + 14, right.w, 22);
          ctx.fillStyle = item.color;
          ctx.fillRect(
            right.x,
            y + 14,
            Math.min(right.w, right.w * item.value / Math.sqrt(state.d) * 1.4),
            22,
          );
        });
        var logCells = state.d * Math.log10(state.resolution);
        ui.output.set([
          { label: "размерность", value: String(state.d) },
          { label: "доля шара", value: currentFraction.toExponential(2), color: C.blue },
          { label: "ячеек сетки", value: "10^" + logCells.toFixed(1), color: C.red },
          { label: "ближнее / среднее", value: (stats.min / stats.mean).toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Размерность d", min: 2, max: 80, step: 1, value: state.d,
      }, function (value) { state.d = value; draw(); });
      K.slider(ui.controls, {
        label: "Делений на ось", min: 2, max: 20, step: 1, value: state.resolution,
      }, function (value) { state.resolution = value; draw(); });
      K.slider(ui.controls, {
        label: "Точек", min: 50, max: 1000, step: 50, value: state.points,
      }, function (value) { state.points = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBatchTraining(root) {
      var ui = setup(
        root,
        "Слева — loss по эпохам, справа — путь параметров по долине. Золотая линия задаёт сохранённый checkpoint.",
        "Мини-батч и ранняя остановка",
        "Малый батч даёт шумный, но дешёвый шаг. Validation помогает выбрать момент остановки, если его не использовать как ещё одну обучающую выборку.",
        485,
      );
      var state = { batch: 32, rate: 0.08, checkpoint: 26, seed: 4 };
      var chart = { x: 55, y: 66, w: 510, h: 305 };
      var valley = { x: 625, y: 82, w: 225, h: 225 };

      function history() {
        var train = [];
        var validation = [];
        var theta = { x: -1.8, y: 1.55 };
        var path = [{ x: theta.x, y: theta.y }];
        var noiseScale = 0.16 / Math.sqrt(state.batch / 16);
        for (var epoch = 0; epoch <= 80; epoch += 1) {
          var base = 0.62 * Math.exp(-epoch * state.rate * 0.72) + 0.055;
          var wiggle = noiseScale * (seeded(epoch + state.seed * 97, 51) - 0.5);
          train.push(clamp(base + wiggle, 0.025, 0.78));
          var overfit = 0.00016 * Math.pow(Math.max(0, epoch - 30), 1.62);
          validation.push(clamp(base + 0.055 + overfit + wiggle * 0.55, 0.04, 0.82));
          var gx = theta.x * 0.24;
          var gy = theta.y * 1.8;
          theta.x -= state.rate * gx + noiseScale * (seeded(epoch + state.seed * 71, 52) - 0.5);
          theta.y -= state.rate * gy + noiseScale * (seeded(epoch + state.seed * 71, 53) - 0.5);
          path.push({ x: theta.x, y: theta.y });
        }
        return { train: train, validation: validation, path: path };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        var values = history();
        title(ctx, "Обучение", chart.x, 36);
        ctx.fillStyle = C.wash; ctx.fillRect(chart.x, chart.y, chart.w, chart.h);
        var trainPoints = values.train.map(function (value, epoch) {
          return {
            x: chart.x + epoch / 80 * chart.w,
            y: chart.y + chart.h - value / 0.85 * chart.h,
          };
        });
        var validationPoints = values.validation.map(function (value, epoch) {
          return {
            x: chart.x + epoch / 80 * chart.w,
            y: chart.y + chart.h - value / 0.85 * chart.h,
          };
        });
        line(ctx, trainPoints, C.blue, 2.4);
        line(ctx, validationPoints, C.red, 2.4);
        var checkpointX = chart.x + state.checkpoint / 80 * chart.w;
        line(ctx, [
          { x: checkpointX, y: chart.y },
          { x: checkpointX, y: chart.y + chart.h },
        ], C.gold, 2);
        small(ctx, "train", chart.x + 14, chart.y + 21, C.blue);
        small(ctx, "validation", chart.x + 62, chart.y + 21, C.red);
        [0, 20, 40, 60, 80].forEach(function (epoch) {
          small(ctx, String(epoch), chart.x + epoch / 80 * chart.w, chart.y + chart.h + 21, C.muted, "center");
        });

        title(ctx, "Путь параметров", valley.x, 36);
        ctx.fillStyle = C.wash; ctx.fillRect(valley.x, valley.y, valley.w, valley.h);
        ctx.save();
        ctx.translate(valley.x + valley.w / 2, valley.y + valley.h / 2);
        for (var ring = 5; ring >= 1; ring -= 1) {
          ctx.beginPath();
          ctx.ellipse(0, 0, ring * 20, ring * 8, -0.35, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(C.blue, 0.12 + ring * 0.035);
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.restore();
        var pathPoints = values.path.map(function (point) {
          return {
            x: valley.x + valley.w / 2 + point.x * 52,
            y: valley.y + valley.h / 2 - point.y * 52,
          };
        });
        line(ctx, pathPoints, C.blue, 1.7);
        var checkpointPoint = pathPoints[state.checkpoint];
        ctx.beginPath(); ctx.arc(checkpointPoint.x, checkpointPoint.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = C.gold; ctx.fill();
        var bestEpoch = values.validation.reduce(function (best, value, index) {
          return value < best.value ? { value: value, index: index } : best;
        }, { value: values.validation[0], index: 0 });
        ui.output.set([
          { label: "checkpoint", value: "эпоха " + state.checkpoint, color: C.gold },
          { label: "train loss", value: values.train[state.checkpoint].toFixed(3), color: C.blue },
          { label: "validation loss", value: values.validation[state.checkpoint].toFixed(3), color: C.red },
          { label: "лучший epoch", value: String(bestEpoch.index) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Размер батча", min: 8, max: 256, step: 8, value: state.batch,
      }, function (value) { state.batch = value; draw(); });
      K.slider(ui.controls, {
        label: "Шаг", min: 0.01, max: 0.16, step: 0.005, value: state.rate,
        format: function (v) { return v.toFixed(3); },
      }, function (value) { state.rate = value; draw(); });
      K.slider(ui.controls, {
        label: "Checkpoint", min: 1, max: 80, step: 1, value: state.checkpoint,
      }, function (value) { state.checkpoint = value; draw(); });
      K.slider(ui.controls, {
        label: "Перемешивание", min: 1, max: 12, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildOptimizers(root) {
      var ui = setup(
        root,
        "Все алгоритмы получают один и тот же градиент функции. Выберите метод и следите, как он проходит узкую долину.",
        "Траектории оптимизаторов",
        "Momentum хранит направление, AdaGrad делит шаг на накопленную норму, Adam соединяет обе идеи. Скорость в начале ещё не гарантирует лучший финиш.",
        485,
      );
      var state = { method: "adam", condition: 18, rate: 0.08, steps: 35 };
      var plot = { x: 95, y: 58, w: 720, h: 335 };

      function gradient(point) {
        var angle = -0.42;
        var ca = Math.cos(angle);
        var sa = Math.sin(angle);
        var u = ca * point.x - sa * point.y;
        var v = sa * point.x + ca * point.y;
        var gu = 2 * u;
        var gv = 2 * state.condition * v;
        return {
          x: (ca * gu + sa * gv) * 0.12,
          y: (-sa * gu + ca * gv) * 0.12,
        };
      }

      function trajectory(method) {
        var point = { x: -2.25, y: 1.5 };
        var velocity = { x: 0, y: 0 };
        var square = { x: 0, y: 0 };
        var first = { x: 0, y: 0 };
        var path = [{ x: point.x, y: point.y }];
        for (var step = 1; step <= state.steps; step += 1) {
          var g = gradient(point);
          if (method === "sgd") {
            point.x -= state.rate * g.x;
            point.y -= state.rate * g.y;
          } else if (method === "momentum") {
            velocity.x = 0.86 * velocity.x + g.x;
            velocity.y = 0.86 * velocity.y + g.y;
            point.x -= state.rate * velocity.x;
            point.y -= state.rate * velocity.y;
          } else if (method === "adagrad") {
            square.x += g.x * g.x;
            square.y += g.y * g.y;
            point.x -= state.rate * g.x / (Math.sqrt(square.x) + 1e-7) * 5;
            point.y -= state.rate * g.y / (Math.sqrt(square.y) + 1e-7) * 5;
          } else {
            first.x = 0.9 * first.x + 0.1 * g.x;
            first.y = 0.9 * first.y + 0.1 * g.y;
            square.x = 0.999 * square.x + 0.001 * g.x * g.x;
            square.y = 0.999 * square.y + 0.001 * g.y * g.y;
            var firstX = first.x / (1 - Math.pow(0.9, step));
            var firstY = first.y / (1 - Math.pow(0.9, step));
            var squareX = square.x / (1 - Math.pow(0.999, step));
            var squareY = square.y / (1 - Math.pow(0.999, step));
            point.x -= state.rate * firstX / (Math.sqrt(squareX) + 1e-7) * 2.2;
            point.y -= state.rate * firstY / (Math.sqrt(squareY) + 1e-7) * 2.2;
          }
          point.x = clamp(point.x, -3.2, 3.2);
          point.y = clamp(point.y, -2.4, 2.4);
          path.push({ x: point.x, y: point.y });
        }
        return path;
      }

      function loss(point) {
        var angle = -0.42;
        var u = Math.cos(angle) * point.x - Math.sin(angle) * point.y;
        var v = Math.sin(angle) * point.x + Math.cos(angle) * point.y;
        return u * u + state.condition * v * v;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        title(ctx, "Узкая квадратичная долина", plot.x, 34);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        ctx.save();
        ctx.translate(plot.x + plot.w / 2, plot.y + plot.h / 2);
        ctx.rotate(0.42);
        for (var ring = 7; ring >= 1; ring -= 1) {
          ctx.beginPath();
          ctx.ellipse(0, 0, ring * 47, ring * 47 / Math.sqrt(state.condition), 0, 0, Math.PI * 2);
          ctx.strokeStyle = rgba(C.blue, 0.1 + ring * 0.025);
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
        ctx.restore();
        var methods = [
          { key: "sgd", label: "SGD", color: C.blue },
          { key: "momentum", label: "Momentum", color: C.red },
          { key: "adagrad", label: "AdaGrad", color: C.green },
          { key: "adam", label: "Adam", color: C.gold },
        ];
        var selectedPath = null;
        ctx.save();
        ctx.beginPath();
        ctx.rect(plot.x, plot.y, plot.w, plot.h);
        ctx.clip();
        methods.forEach(function (method) {
          var path = trajectory(method.key);
          if (method.key === state.method) selectedPath = path;
          var points = path.map(function (point) {
            return {
              x: plot.x + plot.w / 2 + point.x * 125,
              y: plot.y + plot.h / 2 - point.y * 85,
            };
          });
          line(
            ctx,
            points,
            method.key === state.method ? method.color : rgba(method.color, 0.38),
            method.key === state.method ? 3.2 : 1.15,
            method.key === state.method ? [] : [3, 4],
          );
        });
        ctx.restore();
        methods.forEach(function (method, index) {
          small(ctx, method.label, plot.x + 14 + index * 92, plot.y + 22, method.color);
        });
        var end = selectedPath[selectedPath.length - 1];
        var endX = plot.x + plot.w / 2 + end.x * 125;
        var endY = plot.y + plot.h / 2 - end.y * 85;
        ctx.beginPath(); ctx.arc(endX, endY, 7, 0, Math.PI * 2);
        ctx.fillStyle = methods.filter(function (m) { return m.key === state.method; })[0].color;
        ctx.fill();
        ui.output.set([
          { label: "метод", value: state.method.toUpperCase() },
          { label: "шагов", value: String(state.steps) },
          { label: "конечный loss", value: loss(end).toFixed(4), color: C.red },
          { label: "координаты", value: end.x.toFixed(2) + "; " + end.y.toFixed(2) },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Алгоритм",
        value: state.method,
        options: [
          { label: "SGD", value: "sgd" },
          { label: "Momentum", value: "momentum" },
          { label: "AdaGrad", value: "adagrad" },
          { label: "Adam", value: "adam" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Обусловленность", min: 3, max: 30, step: 1, value: state.condition,
      }, function (value) { state.condition = value; draw(); });
      K.slider(ui.controls, {
        label: "Шаг", min: 0.01, max: 0.12, step: 0.005, value: state.rate,
        format: function (v) { return v.toFixed(3); },
      }, function (value) { state.rate = value; draw(); });
      K.slider(ui.controls, {
        label: "Итераций", min: 5, max: 70, step: 1, value: state.steps,
      }, function (value) { state.steps = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildOracle(root) {
      var ui = setup(
        root,
        "Синяя линия скрыта от исследователя. Золотая доступна на validation, красная добавляет штраф за перебор.",
        "Oracle и шум выбора",
        "Чем больше кандидатов просмотрено на одном validation, тем легче выбрать удачную флуктуацию. Отдельный test оценивает уже зафиксированное решение.",
        475,
      );
      var state = { candidates: 18, n: 240, penalty: 0.45, seed: 5 };
      var plot = { x: 65, y: 62, w: 800, h: 315 };

      function values() {
        var result = [];
        for (var i = 0; i < state.candidates; i += 1) {
          var complexity = (i + 1) / state.candidates;
          var truth = 0.115 + 0.14 * Math.pow(complexity - 0.56, 2) + 0.012 * complexity;
          var noise = (seeded(i + state.seed * 191, 61) + seeded(i + state.seed * 191, 62) - 1)
            * 0.42 / Math.sqrt(state.n);
          var validation = clamp(truth + noise, 0.05, 0.32);
          var correction = state.penalty * Math.sqrt(Math.log(state.candidates + 1) / state.n) * complexity;
          result.push({ truth: truth, validation: validation, penalized: validation + correction });
        }
        return result;
      }

      function argmin(items, key) {
        return items.reduce(function (best, item, index) {
          return item[key] < best.value ? { value: item[key], index: index } : best;
        }, { value: items[0][key], index: 0 });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 475);
        title(ctx, "Риск по сложности модели", plot.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var items = values();
        function point(value, index) {
          return {
            x: plot.x + index / Math.max(1, items.length - 1) * plot.w,
            y: plot.y + plot.h - (value - 0.05) / 0.27 * plot.h,
          };
        }
        line(ctx, items.map(function (item, index) { return point(item.truth, index); }), C.blue, 2.5);
        line(ctx, items.map(function (item, index) { return point(item.validation, index); }), C.gold, 2.3);
        line(ctx, items.map(function (item, index) { return point(item.penalized, index); }), C.red, 2.3);
        var oracle = argmin(items, "truth");
        var chosen = argmin(items, "validation");
        var corrected = argmin(items, "penalized");
        [
          { item: oracle, color: C.blue },
          { item: chosen, color: C.gold },
          { item: corrected, color: C.red },
        ].forEach(function (marker) {
          var x = plot.x + marker.item.index / Math.max(1, items.length - 1) * plot.w;
          line(ctx, [{ x: x, y: plot.y }, { x: x, y: plot.y + plot.h }], marker.color, 1.2, [4, 5]);
        });
        small(ctx, "oracle", plot.x + 14, plot.y + 21, C.blue);
        small(ctx, "validation", plot.x + 68, plot.y + 21, C.gold);
        small(ctx, "со штрафом", plot.x + 148, plot.y + 21, C.red);
        small(ctx, "простая модель", plot.x, plot.y + plot.h + 21);
        small(ctx, "сложная модель", plot.x + plot.w, plot.y + plot.h + 21, C.muted, "right");
        var optimism = items[chosen.index].truth - items[chosen.index].validation;
        ui.output.set([
          { label: "oracle", value: "#" + (oracle.index + 1), color: C.blue },
          { label: "min validation", value: "#" + (chosen.index + 1), color: C.gold },
          { label: "со штрафом", value: "#" + (corrected.index + 1), color: C.red },
          { label: "optimism", value: optimism.toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Кандидатов", min: 3, max: 60, step: 1, value: state.candidates,
      }, function (value) { state.candidates = value; draw(); });
      K.slider(ui.controls, {
        label: "Validation n", min: 30, max: 2000, step: 10, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.slider(ui.controls, {
        label: "Штраф", min: 0, max: 1.5, step: 0.05, value: state.penalty,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.penalty = value; draw(); });
      K.slider(ui.controls, {
        label: "Разбиение", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildFederated(root) {
      var ui = setup(
        root,
        "Три цвета — клиенты с разными локальными минимумами. Точки наверху показывают раунды сервера.",
        "FedAvg на неоднородных клиентах",
        "Вес клиента и число локальных шагов меняют итог. Средняя ошибка может уменьшиться одновременно с ростом ошибки малой группы.",
        485,
      );
      var state = { weightA: 55, localSteps: 4, heterogeneity: 1.35, rounds: 10 };
      var plot = { x: 65, y: 70, w: 800, h: 275 };
      var colors = [C.blue, C.red, C.green];

      function clientData() {
        var weightA = state.weightA / 100;
        return [
          { mean: -state.heterogeneity, weight: weightA, color: colors[0], name: "клиент A" },
          { mean: 0.15, weight: (1 - weightA) * 0.65, color: colors[1], name: "клиент B" },
          { mean: state.heterogeneity * 1.15, weight: (1 - weightA) * 0.35, color: colors[2], name: "клиент C" },
        ];
      }

      function rounds(clients) {
        var path = [0.8];
        var contraction = Math.pow(1 - 0.22, state.localSteps);
        for (var round = 0; round < state.rounds; round += 1) {
          var server = path[path.length - 1];
          var next = clients.reduce(function (sum, client) {
            var local = client.mean + (server - client.mean) * contraction;
            return sum + client.weight * local;
          }, 0);
          path.push(next);
        }
        return path;
      }

      function sx(value) {
        return plot.x + (value + 2.6) / 5.2 * plot.w;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        title(ctx, "Локальные функции потерь", plot.x, 36);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var clients = clientData();
        clients.forEach(function (client, clientIndex) {
          var points = [];
          for (var i = 0; i <= 160; i += 1) {
            var value = -2.6 + i / 160 * 5.2;
            var loss = 0.055 + 0.085 * Math.pow(value - client.mean, 2);
            points.push({
              x: sx(value),
              y: plot.y + plot.h - clamp(loss / 0.65, 0, 1) * plot.h,
            });
          }
          line(ctx, points, client.color, 2.2);
          small(
            ctx,
            client.name + " · " + Math.round(client.weight * 100) + "%",
            plot.x + 14 + clientIndex * 150,
            plot.y + 21,
            client.color,
          );
        });
        var path = rounds(clients);
        var pathY = plot.y + 52;
        line(ctx, path.map(function (value, index) {
          return { x: sx(value), y: pathY - index * 1.8 };
        }), C.gold, 2.5);
        path.forEach(function (value, index) {
          var x = sx(value);
          var y = pathY - index * 1.8;
          ctx.beginPath(); ctx.arc(x, y, index === path.length - 1 ? 6 : 3.5, 0, Math.PI * 2);
          ctx.fillStyle = C.gold; ctx.fill();
        });
        var final = path[path.length - 1];
        line(ctx, [
          { x: sx(final), y: plot.y },
          { x: sx(final), y: plot.y + plot.h },
        ], C.gold, 1.5, [4, 4]);
        [-2, -1, 0, 1, 2].forEach(function (value) {
          small(ctx, String(value), sx(value), plot.y + plot.h + 21, C.muted, "center");
        });
        var errors = clients.map(function (client) {
          return Math.pow(final - client.mean, 2);
        });
        var average = clients.reduce(function (sum, client, index) {
          return sum + client.weight * errors[index];
        }, 0);
        ui.output.set([
          { label: "серверная модель", value: final.toFixed(3), color: C.gold },
          { label: "средняя ошибка", value: average.toFixed(3) },
          { label: "ошибка A", value: errors[0].toFixed(3), color: C.blue },
          { label: "ошибка C", value: errors[2].toFixed(3), color: C.green },
        ]);
      }

      K.slider(ui.controls, {
        label: "Доля клиента A", min: 10, max: 80, step: 1, value: state.weightA,
        unit: "%",
      }, function (value) { state.weightA = value; draw(); });
      K.slider(ui.controls, {
        label: "Локальных шагов", min: 1, max: 16, step: 1, value: state.localSteps,
      }, function (value) { state.localSteps = value; draw(); });
      K.slider(ui.controls, {
        label: "Неоднородность", min: 0.2, max: 2, step: 0.05, value: state.heterogeneity,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.heterogeneity = value; draw(); });
      K.slider(ui.controls, {
        label: "Раундов", min: 1, max: 24, step: 1, value: state.rounds,
      }, function (value) { state.rounds = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "57": buildRisk,
      "58": buildErm,
      "59": buildDoubleDescent,
      "60": buildDimension,
      "61": buildBatchTraining,
      "62": buildOptimizers,
      "63": buildOracle,
      "64": buildFederated,
    };

    K.register("g11-ml", function (root, options) {
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
