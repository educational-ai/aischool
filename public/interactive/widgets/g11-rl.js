// Grade 11, module 3. Markov chains and reinforcement learning.
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
      var value = Math.sin(index * 72.731 + salt * 31.177) * 43758.5453;
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
      var canvasState = K.makeCanvas(stage, 920, height || 485, {
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

    function arrow(ctx, from, to, color, width) {
      var angle = Math.atan2(to.y - from.y, to.x - from.x);
      var end = {
        x: to.x - Math.cos(angle) * 25,
        y: to.y - Math.sin(angle) * 25,
      };
      var start = {
        x: from.x + Math.cos(angle) * 25,
        y: from.y + Math.sin(angle) * 25,
      };
      line(ctx, [start, end], color, width || 1.5);
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(end.x, end.y);
      ctx.lineTo(
        end.x - Math.cos(angle - Math.PI / 6) * 8,
        end.y - Math.sin(angle - Math.PI / 6) * 8,
      );
      ctx.lineTo(
        end.x - Math.cos(angle + Math.PI / 6) * 8,
        end.y - Math.sin(angle + Math.PI / 6) * 8,
      );
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function buildPageRank(root) {
      var ui = setup(
        root,
        "Размер круга задаёт точный PageRank, внешнее кольцо — частоту посещений симуляцией. Режим «ловушка» замыкает две страницы.",
        "Веб-граф и PageRank",
        "Телепортация возвращает выход из ловушек и делает предел независимым от начальной страницы.",
        485,
      );
      var state = { alpha: 0.85, clicks: 800, seed: 4, graph: "normal" };
      var nodes = [
        { x: 170, y: 135, label: "A" },
        { x: 360, y: 90, label: "B" },
        { x: 510, y: 205, label: "C" },
        { x: 320, y: 315, label: "D" },
        { x: 125, y: 320, label: "E" },
        { x: 705, y: 145, label: "F" },
      ];

      function outgoing() {
        if (state.graph === "trap") {
          return [[1, 4], [2], [1], [0, 2], [0, 3], [2, 5]];
        }
        return [[1, 4], [2, 5], [0, 3], [0, 2], [0, 3], [2]];
      }

      function exactRank(edges) {
        var rank = nodes.map(function () { return 1 / nodes.length; });
        for (var iteration = 0; iteration < 250; iteration += 1) {
          var next = nodes.map(function () { return (1 - state.alpha) / nodes.length; });
          rank.forEach(function (weight, index) {
            var targets = edges[index];
            if (!targets.length) targets = nodes.map(function (_, i) { return i; });
            targets.forEach(function (target) {
              next[target] += state.alpha * weight / targets.length;
            });
          });
          rank = next;
        }
        return rank;
      }

      function simulate(edges) {
        var visits = nodes.map(function () { return 0; });
        var current = 0;
        for (var step = 0; step < state.clicks; step += 1) {
          visits[current] += 1;
          var random = seeded(step + state.seed * 1009, 11);
          if (random > state.alpha) {
            current = Math.floor(seeded(step + state.seed * 1009, 12) * nodes.length);
          } else {
            var targets = edges[current];
            if (!targets.length) current = Math.floor(seeded(step, 13) * nodes.length);
            else current = targets[Math.floor(seeded(step + state.seed * 1009, 14) * targets.length)];
          }
        }
        return visits.map(function (count) { return count / state.clicks; });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        var edges = outgoing();
        var rank = exactRank(edges);
        var empirical = simulate(edges);
        title(ctx, "Сеть ссылок", 65, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(65, 58, 720, 335);
        edges.forEach(function (targets, source) {
          targets.forEach(function (target) {
            arrow(ctx, nodes[source], nodes[target], rgba(C.muted, 0.55), 1.4);
          });
        });
        nodes.forEach(function (node, index) {
          var radius = 18 + rank[index] * 72;
          var empiricalRadius = 18 + empirical[index] * 72;
          ctx.beginPath();
          ctx.arc(node.x, node.y, empiricalRadius + 5, 0, Math.PI * 2);
          ctx.strokeStyle = C.gold;
          ctx.lineWidth = 2.5;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = rgba(C.blue, 0.18);
          ctx.fill();
          ctx.strokeStyle = C.blue;
          ctx.lineWidth = 1.8;
          ctx.stroke();
          ctx.fillStyle = C.ink;
          ctx.font = "600 16px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(node.label, node.x, node.y);
          small(ctx, rank[index].toFixed(3), node.x, node.y + radius + 18, C.blue, "center");
        });
        small(ctx, "синий круг · точный ранг", 80, 420, C.blue);
        small(ctx, "золотое кольцо · симуляция", 260, 420, C.gold);
        var error = rank.reduce(function (sum, value, index) {
          return sum + Math.abs(value - empirical[index]);
        }, 0);
        var winner = rank.indexOf(Math.max.apply(null, rank));
        ui.output.set([
          { label: "лидер", value: nodes[winner].label, color: C.blue },
          { label: "его PageRank", value: rank[winner].toFixed(3) },
          { label: "ошибка L1", value: error.toFixed(3), color: C.gold },
          { label: "режим", value: state.graph === "trap" ? "ловушка B–C" : "связный граф" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Граф",
        value: state.graph,
        options: [
          { label: "Обычный", value: "normal" },
          { label: "Ловушка", value: "trap" },
        ],
      }, function (value) { state.graph = value; draw(); });
      K.slider(ui.controls, {
        label: "Следовать ссылке α", min: 0, max: 1, step: 0.01, value: state.alpha,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.alpha = value; draw(); });
      K.slider(ui.controls, {
        label: "Кликов", min: 50, max: 5000, step: 50, value: state.clicks,
      }, function (value) { state.clicks = value; draw(); });
      K.slider(ui.controls, {
        label: "Маршрут", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildRandomWalk(root) {
      var ui = setup(
        root,
        "Слева показаны несколько траекторий, справа — конечные положения всей группы. Дрейф можно убрать, чтобы увидеть масштаб √n.",
        "Случайные блуждания",
        "Суммарный путь растёт как n, а среднеквадратичное смещение честного блуждания — как √n.",
        490,
      );
      var state = { steps: 300, walkers: 120, drift: 0 };
      var trajectoryPlot = { x: 55, y: 65, w: 515, h: 315 };
      var histogram = { x: 635, y: 65, w: 220, h: 315 };

      function walks() {
        var finals = [];
        var paths = [];
        for (var walker = 0; walker < state.walkers; walker += 1) {
          var position = 0;
          var path = [{ x: 0, y: 0 }];
          for (var step = 1; step <= state.steps; step += 1) {
            var right = seeded(walker * 100003 + step, 21) < 0.5 + state.drift;
            position += right ? 1 : -1;
            if (walker < 14 && (step % Math.max(1, Math.floor(state.steps / 120)) === 0 || step === state.steps)) {
              path.push({ x: step, y: position });
            }
          }
          finals.push(position);
          if (walker < 14) paths.push(path);
        }
        return { finals: finals, paths: paths };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 490);
        title(ctx, "Траектории", trajectoryPlot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(trajectoryPlot.x, trajectoryPlot.y, trajectoryPlot.w, trajectoryPlot.h);
        var result = walks();
        var expected = 2 * state.drift * state.steps;
        var scale = Math.max(Math.sqrt(state.steps) * 3.5, Math.abs(expected) + Math.sqrt(state.steps) * 2.8);
        function tx(step) { return trajectoryPlot.x + step / state.steps * trajectoryPlot.w; }
        function ty(position) {
          return trajectoryPlot.y + trajectoryPlot.h / 2 - position / (2 * scale) * trajectoryPlot.h;
        }
        line(ctx, [
          { x: trajectoryPlot.x, y: ty(0) },
          { x: trajectoryPlot.x + trajectoryPlot.w, y: ty(0) },
        ], C.axis, 1);
        result.paths.forEach(function (path, index) {
          line(ctx, path.map(function (point) {
            return { x: tx(point.x), y: ty(point.y) };
          }), rgba(index === 0 ? C.red : C.blue, index === 0 ? 0.9 : 0.25), index === 0 ? 2.3 : 1.1);
        });
        small(ctx, "0", trajectoryPlot.x, trajectoryPlot.y + trajectoryPlot.h + 20);
        small(ctx, String(state.steps), trajectoryPlot.x + trajectoryPlot.w, trajectoryPlot.y + trajectoryPlot.h + 20, C.muted, "right");

        title(ctx, "Финиши", histogram.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(histogram.x, histogram.y, histogram.w, histogram.h);
        var bins = new Array(21).fill(0);
        result.finals.forEach(function (value) {
          var index = clamp(Math.floor((value + scale) / (2 * scale) * bins.length), 0, bins.length - 1);
          bins[index] += 1;
        });
        var maxBin = Math.max.apply(null, bins);
        bins.forEach(function (count, index) {
          var barWidth = histogram.w / bins.length;
          var barHeight = count / maxBin * (histogram.h - 30);
          ctx.fillStyle = C.blue;
          ctx.fillRect(
            histogram.x + index * barWidth + 1,
            histogram.y + histogram.h - barHeight,
            Math.max(1, barWidth - 2),
            barHeight,
          );
        });
        line(ctx, [
          { x: histogram.x + histogram.w / 2, y: histogram.y },
          { x: histogram.x + histogram.w / 2, y: histogram.y + histogram.h },
        ], C.gold, 1.5, [4, 4]);
        var centeredSquares = result.finals.reduce(function (sum, value) {
          return sum + Math.pow(value - expected, 2);
        }, 0) / result.finals.length;
        var mean = result.finals.reduce(function (a, b) { return a + b; }, 0) / result.finals.length;
        ui.output.set([
          { label: "средний финиш", value: mean.toFixed(1), color: C.red },
          { label: "RMS без дрейфа", value: Math.sqrt(centeredSquares).toFixed(1), color: C.blue },
          { label: "√n", value: Math.sqrt(state.steps).toFixed(1), color: C.gold },
          { label: "весь путь", value: String(state.steps) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Шагов n", min: 20, max: 1500, step: 10, value: state.steps,
      }, function (value) { state.steps = value; draw(); });
      K.slider(ui.controls, {
        label: "Траекторий", min: 20, max: 500, step: 10, value: state.walkers,
      }, function (value) { state.walkers = value; draw(); });
      K.slider(ui.controls, {
        label: "Дрейф p − 0.5", min: -0.1, max: 0.1, step: 0.005, value: state.drift,
        format: function (v) { return v.toFixed(3); },
      }, function (value) { state.drift = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function normalDensity(x, mean, sigma) {
      return Math.exp(-0.5 * Math.pow((x - mean) / sigma, 2))
        / (sigma * Math.sqrt(2 * Math.PI));
    }

    function buildMcmc(root) {
      var ui = setup(
        root,
        "Красная линия — target-плотность, синие столбики — частоты цепи. Внизу виден последний участок trace.",
        "Метрополис для двух мод",
        "Слишком малое предложение даёт сильную автокорреляцию, слишком большое — длинные серии отказов.",
        500,
      );
      var state = { width: 1.25, iterations: 1800, start: -4, seed: 3 };
      var densityPlot = { x: 55, y: 55, w: 810, h: 255 };
      var tracePlot = { x: 55, y: 355, w: 810, h: 85 };

      function target(x) {
        return 0.48 * normalDensity(x, -2, 0.62) + 0.52 * normalDensity(x, 2, 0.78);
      }

      function chain() {
        var values = [];
        var current = state.start;
        var accepted = 0;
        var switches = 0;
        var lastSide = current >= 0;
        for (var step = 0; step < state.iterations; step += 1) {
          var proposal = current + (seeded(step + state.seed * 10007, 31) * 2 - 1) * state.width;
          var ratio = target(proposal) / Math.max(1e-12, target(current));
          if (seeded(step + state.seed * 10007, 32) < Math.min(1, ratio)) {
            current = proposal;
            accepted += 1;
          }
          var side = current >= 0;
          if (side !== lastSide) switches += 1;
          lastSide = side;
          values.push(current);
        }
        return { values: values, accepted: accepted, switches: switches };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var result = chain();
        title(ctx, "Target и частоты", densityPlot.x, 32);
        ctx.fillStyle = C.wash;
        ctx.fillRect(densityPlot.x, densityPlot.y, densityPlot.w, densityPlot.h);
        var bins = new Array(64).fill(0);
        result.values.slice(Math.floor(state.iterations * 0.2)).forEach(function (value) {
          var index = Math.floor((value + 6) / 12 * bins.length);
          if (index >= 0 && index < bins.length) bins[index] += 1;
        });
        var maxBin = Math.max.apply(null, bins);
        bins.forEach(function (count, index) {
          var width = densityPlot.w / bins.length;
          var height = maxBin ? count / maxBin * densityPlot.h * 0.88 : 0;
          ctx.fillStyle = rgba(C.blue, 0.45);
          ctx.fillRect(
            densityPlot.x + index * width,
            densityPlot.y + densityPlot.h - height,
            Math.max(1, width - 1),
            height,
          );
        });
        var densityPoints = [];
        for (var i = 0; i <= 300; i += 1) {
          var x = -6 + i / 300 * 12;
          densityPoints.push({
            x: densityPlot.x + i / 300 * densityPlot.w,
            y: densityPlot.y + densityPlot.h - target(x) / 0.43 * densityPlot.h * 0.88,
          });
        }
        line(ctx, densityPoints, C.red, 2.6);
        [-6, -3, 0, 3, 6].forEach(function (value) {
          small(ctx, String(value), densityPlot.x + (value + 6) / 12 * densityPlot.w, densityPlot.y + densityPlot.h + 18, C.muted, "center");
        });

        title(ctx, "Trace последних шагов", tracePlot.x, 340);
        ctx.fillStyle = C.wash;
        ctx.fillRect(tracePlot.x, tracePlot.y, tracePlot.w, tracePlot.h);
        var tail = result.values.slice(-Math.min(600, result.values.length));
        line(ctx, tail.map(function (value, index) {
          return {
            x: tracePlot.x + index / Math.max(1, tail.length - 1) * tracePlot.w,
            y: tracePlot.y + tracePlot.h / 2 - clamp(value, -6, 6) / 12 * tracePlot.h,
          };
        }), C.blue, 1.4);
        var leftShare = result.values.slice(Math.floor(state.iterations * 0.2)).filter(function (v) { return v < 0; }).length
          / Math.max(1, Math.ceil(state.iterations * 0.8));
        ui.output.set([
          { label: "acceptance", value: (result.accepted / state.iterations * 100).toFixed(1) + "%", color: C.blue },
          { label: "смен моды", value: String(result.switches), color: C.red },
          { label: "доля слева", value: (leftShare * 100).toFixed(1) + "%" },
          { label: "burn-in", value: Math.floor(state.iterations * 0.2) + " шагов" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Ширина предложения", min: 0.05, max: 7, step: 0.05, value: state.width,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.width = value; draw(); });
      K.slider(ui.controls, {
        label: "Итераций", min: 100, max: 5000, step: 100, value: state.iterations,
      }, function (value) { state.iterations = value; draw(); });
      K.slider(ui.controls, {
        label: "Старт", min: -5, max: 5, step: 0.25, value: state.start,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.start = value; draw(); });
      K.slider(ui.controls, {
        label: "Цепь", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildCipher(root) {
      var ui = setup(
        root,
        "Шифротекст фиксирован. Цепь меняет местами две буквы ключа и оценивает результат по триграммам отдельного корпуса.",
        "MCMC-дешифровщик",
        "Короткий текст часто допускает несколько правдоподобных ключей. Повторные старты показывают эту неопределённость.",
        500,
      );
      var state = { temperature: 0.9, iterations: 5000, seed: 4 };
      var alphabet = "абвгдежзийклмнопрстуфхцчшщъыьэюя";
      var scoreAlphabet = alphabet + " ";
      var plain = (
        "случайная цепь меняет состояние шаг за шагом " +
        "дешифровщик предлагает новый ключ сравнивает частоты соседних букв " +
        "и иногда принимает худший вариант чтобы выйти из ловушки"
      );
      var corpus = (
        "вечером школьники читали книгу и спорили о сложной задаче " +
        "случайный процесс переходит между состояниями и оставляет длинную траекторию " +
        "математика связывает наблюдение формулу вычисление и независимую проверку " +
        "хороший алгоритм ищет закономерность сравнивает ответы и признает ошибку " +
        "русский текст хранит частоты букв границы слов и сочетания соседних символов " +
        "после урока класс построил модель движения и проверил результат на новом примере " +
        "марковская цепь предлагает состояние принимает или отклоняет следующий шаг " +
        "если новый ключ улучшает языковой балл переход почти всегда будет принят " +
        "иногда полезно принять слабый вариант и выбраться из локальной ловушки " +
        "длинный отрывок легче расшифровать потому что в нем повторяются слова и окончания " +
        "отдельный корпус нужен для оценки правдоподобия русской последовательности " +
        "частые сочетания букв получают высокий балл редкие сочетания получают низкий балл"
      ).replace(/[^а-я ]/g, " ");
      var encryption = alphabet.split("");
      for (var shuffle = encryption.length - 1; shuffle > 0; shuffle -= 1) {
        var swapWith = Math.floor(seeded(shuffle, 41) * (shuffle + 1));
        var temporary = encryption[shuffle];
        encryption[shuffle] = encryption[swapWith];
        encryption[swapWith] = temporary;
      }
      function encrypt(text) {
        return text.split("").map(function (character) {
          var index = alphabet.indexOf(character);
          return index < 0 ? character : encryption[index];
        }).join("");
      }
      var ciphertext = encrypt(plain);
      var scoreSize = scoreAlphabet.length;
      var counts = new Array(scoreSize * scoreSize * scoreSize).fill(0.15);
      var totals = new Array(scoreSize * scoreSize).fill(scoreSize * 0.15);
      for (var triple = 2; triple < corpus.length; triple += 1) {
        var a = scoreAlphabet.indexOf(corpus[triple - 2]);
        var b = scoreAlphabet.indexOf(corpus[triple - 1]);
        var c = scoreAlphabet.indexOf(corpus[triple]);
        if (a >= 0 && b >= 0 && c >= 0) {
          counts[(a * scoreSize + b) * scoreSize + c] += 1;
          totals[a * scoreSize + b] += 1;
        }
      }

      function decode(key) {
        return ciphertext.split("").map(function (character) {
          var cipherIndex = alphabet.indexOf(character);
          return cipherIndex < 0 ? character : alphabet[key[cipherIndex]];
        }).join("");
      }

      function score(text) {
        var total = 0;
        for (var index = 2; index < text.length; index += 1) {
          var a = scoreAlphabet.indexOf(text[index - 2]);
          var b = scoreAlphabet.indexOf(text[index - 1]);
          var c = scoreAlphabet.indexOf(text[index]);
          if (a >= 0 && b >= 0 && c >= 0) {
            total += Math.log(counts[(a * scoreSize + b) * scoreSize + c] / totals[a * scoreSize + b]);
          }
        }
        return total;
      }

      function initialKey() {
        var cipherCounts = new Array(alphabet.length).fill(0);
        var corpusCounts = new Array(alphabet.length).fill(0);
        ciphertext.split("").forEach(function (character) {
          var index = alphabet.indexOf(character);
          if (index >= 0) cipherCounts[index] += 1;
        });
        corpus.split("").forEach(function (character) {
          var index = alphabet.indexOf(character);
          if (index >= 0) corpusCounts[index] += 1;
        });
        var cipherOrder = cipherCounts.map(function (count, index) { return { count: count, index: index }; })
          .sort(function (x, y) { return y.count - x.count; });
        var corpusOrder = corpusCounts.map(function (count, index) { return { count: count, index: index }; })
          .sort(function (x, y) { return y.count - x.count; });
        var key = new Array(alphabet.length);
        cipherOrder.forEach(function (item, rank) {
          key[item.index] = corpusOrder[rank].index;
        });
        return key;
      }

      function runChain() {
        var key = initialKey();
        var currentText = decode(key);
        var currentScore = score(currentText);
        var bestText = currentText;
        var bestScore = currentScore;
        var accepted = 0;
        var history = [{ step: 0, score: bestScore }];
        for (var step = 1; step <= state.iterations; step += 1) {
          var first = Math.floor(seeded(step + state.seed * 20011, 42) * alphabet.length);
          var second = Math.floor(seeded(step + state.seed * 20011, 43) * alphabet.length);
          if (second === first) second = (second + 1) % alphabet.length;
          var proposal = key.slice();
          var swap = proposal[first];
          proposal[first] = proposal[second];
          proposal[second] = swap;
          var proposalText = decode(proposal);
          var proposalScore = score(proposalText);
          var acceptance = Math.exp((proposalScore - currentScore) / state.temperature);
          if (proposalScore >= currentScore || seeded(step + state.seed * 20011, 44) < acceptance) {
            key = proposal;
            currentText = proposalText;
            currentScore = proposalScore;
            accepted += 1;
          }
          if (currentScore > bestScore) {
            bestScore = currentScore;
            bestText = currentText;
          }
          if (step % Math.max(1, Math.floor(state.iterations / 100)) === 0) {
            history.push({ step: step, score: bestScore });
          }
        }
        return { text: bestText, score: bestScore, accepted: accepted, history: history };
      }

      function wrappedText(ctx, text, x, y, width, lineHeight, color) {
        ctx.fillStyle = color;
        ctx.font = "18px ET Book, Palatino, Georgia, serif";
        ctx.textAlign = "left";
        var words = text.split(" ");
        var current = "";
        var offset = 0;
        words.forEach(function (word) {
          var trial = current ? current + " " + word : word;
          if (ctx.measureText(trial).width > width && current) {
            ctx.fillText(current, x, y + offset);
            current = word;
            offset += lineHeight;
          } else current = trial;
        });
        ctx.fillText(current, x, y + offset);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var result = runChain();
        title(ctx, "Шифротекст", 60, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(60, 55, 800, 90);
        wrappedText(ctx, ciphertext, 78, 88, 760, 26, C.muted);
        title(ctx, "Лучший вариант цепи", 60, 180);
        ctx.fillStyle = rgba(C.blue, 0.07); ctx.fillRect(60, 200, 800, 105);
        wrappedText(ctx, result.text, 78, 236, 760, 28, C.ink);
        var plot = { x: 60, y: 350, w: 800, h: 90 };
        title(ctx, "Лучший score", plot.x, 335);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var minScore = Math.min.apply(null, result.history.map(function (item) { return item.score; }));
        var maxScore = Math.max.apply(null, result.history.map(function (item) { return item.score; }));
        line(ctx, result.history.map(function (item) {
          return {
            x: plot.x + item.step / state.iterations * plot.w,
            y: plot.y + plot.h - (item.score - minScore) / Math.max(1e-6, maxScore - minScore) * (plot.h - 10) - 5,
          };
        }), C.blue, 2.3);
        var matched = result.text.split("").filter(function (character, index) {
          return character === plain[index];
        }).length;
        ui.output.set([
          { label: "лучший score", value: result.score.toFixed(1), color: C.blue },
          { label: "принято", value: (result.accepted / state.iterations * 100).toFixed(1) + "%" },
          { label: "совпало символов", value: matched + " / " + plain.length, color: C.green },
          { label: "старт", value: "#" + state.seed },
        ]);
      }

      K.slider(ui.controls, {
        label: "Температура", min: 0.2, max: 3, step: 0.05, value: state.temperature,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.temperature = value; draw(); });
      K.slider(ui.controls, {
        label: "Итераций", min: 200, max: 12000, step: 200, value: state.iterations,
      }, function (value) { state.iterations = value; draw(); });
      K.slider(ui.controls, {
        label: "Старт цепи", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBellman(root) {
      var ui = setup(
        root,
        "Цвет клетки — ценность, стрелка — жадное действие. Награда распространяется от цели по мере итераций Беллмана.",
        "Итерация ценности",
        "Высокий дисконт смотрит дальше, а вероятность скольжения делает безопасный обход привлекательнее короткого пути.",
        500,
      );
      var state = { gamma: 0.92, slip: 0.15, iterations: 16 };
      var cols = 7;
      var rows = 5;
      var walls = { "2,1": true, "2,2": true, "4,3": true };
      var goal = { x: 6, y: 0 };
      var pit = { x: 6, y: 4 };
      var actions = [
        { x: 1, y: 0, label: "→" },
        { x: 0, y: 1, label: "↓" },
        { x: -1, y: 0, label: "←" },
        { x: 0, y: -1, label: "↑" },
      ];
      var board = { x: 55, y: 65, cell: 68 };
      var convergence = { x: 585, y: 80, w: 275, h: 260 };

      function key(x, y) { return x + "," + y; }
      function terminal(x, y) {
        return (x === goal.x && y === goal.y) || (x === pit.x && y === pit.y);
      }
      function move(x, y, action) {
        var nextX = x + actions[action].x;
        var nextY = y + actions[action].y;
        if (
          nextX < 0 || nextX >= cols || nextY < 0 || nextY >= rows
          || walls[key(nextX, nextY)]
        ) return { x: x, y: y };
        return { x: nextX, y: nextY };
      }
      function reward(x, y) {
        if (x === goal.x && y === goal.y) return 1;
        if (x === pit.x && y === pit.y) return -1;
        return -0.018;
      }
      function actionValue(values, x, y, action) {
        var left = (action + 3) % 4;
        var right = (action + 1) % 4;
        var possibilities = [
          { action: action, probability: 1 - state.slip },
          { action: left, probability: state.slip / 2 },
          { action: right, probability: state.slip / 2 },
        ];
        return possibilities.reduce(function (sum, item) {
          var next = move(x, y, item.action);
          return sum + item.probability * (
            reward(next.x, next.y)
            + (terminal(next.x, next.y) ? 0 : state.gamma * values[key(next.x, next.y)])
          );
        }, 0);
      }
      function solve(iterations) {
        var values = {};
        for (var y = 0; y < rows; y += 1) {
          for (var x = 0; x < cols; x += 1) {
            values[key(x, y)] = 0;
          }
        }
        var changes = [];
        for (var iteration = 0; iteration < iterations; iteration += 1) {
          var nextValues = Object.assign({}, values);
          var largest = 0;
          for (var row = 0; row < rows; row += 1) {
            for (var column = 0; column < cols; column += 1) {
              if (walls[key(column, row)] || terminal(column, row)) continue;
              var candidates = actions.map(function (_, action) {
                return actionValue(values, column, row, action);
              });
              nextValues[key(column, row)] = Math.max.apply(null, candidates);
              largest = Math.max(largest, Math.abs(nextValues[key(column, row)] - values[key(column, row)]));
            }
          }
          values = nextValues;
          changes.push(largest);
        }
        return { values: values, changes: changes };
      }

      function drawPolicyArrow(ctx, cx, cy, action) {
        var dx = actions[action].x * 17;
        var dy = actions[action].y * 17;
        line(ctx, [{ x: cx - dx * 0.25, y: cy - dy * 0.25 }, { x: cx + dx, y: cy + dy }], C.ink, 1.8);
        ctx.fillStyle = C.ink;
        ctx.beginPath();
        if (action === 0) {
          ctx.moveTo(cx + dx + 5, cy + dy);
          ctx.lineTo(cx + dx - 3, cy + dy - 4);
          ctx.lineTo(cx + dx - 3, cy + dy + 4);
        } else if (action === 2) {
          ctx.moveTo(cx + dx - 5, cy + dy);
          ctx.lineTo(cx + dx + 3, cy + dy - 4);
          ctx.lineTo(cx + dx + 3, cy + dy + 4);
        } else if (action === 1) {
          ctx.moveTo(cx + dx, cy + dy + 5);
          ctx.lineTo(cx + dx - 4, cy + dy - 3);
          ctx.lineTo(cx + dx + 4, cy + dy - 3);
        } else {
          ctx.moveTo(cx + dx, cy + dy - 5);
          ctx.lineTo(cx + dx - 4, cy + dy + 3);
          ctx.lineTo(cx + dx + 4, cy + dy + 3);
        }
        ctx.closePath();
        ctx.fill();
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var solution = solve(Math.max(1, state.iterations));
        title(ctx, "Город", board.x, 35);
        for (var y = 0; y < rows; y += 1) {
          for (var x = 0; x < cols; x += 1) {
            var left = board.x + x * board.cell;
            var top = board.y + y * board.cell;
            var cellKey = key(x, y);
            if (walls[cellKey]) {
              ctx.fillStyle = C.axis;
              ctx.fillRect(left, top, board.cell, board.cell);
              continue;
            }
            if (terminal(x, y)) {
              ctx.fillStyle = x === goal.x && y === goal.y ? rgba(C.green, 0.75) : rgba(C.red, 0.72);
            } else {
              var value = solution.values[cellKey];
              ctx.fillStyle = value >= 0
                ? rgba(C.blue, 0.08 + clamp(value, 0, 1) * 0.6)
                : rgba(C.red, 0.08 + clamp(-value, 0, 1) * 0.6);
            }
            ctx.fillRect(left, top, board.cell, board.cell);
            ctx.strokeStyle = C.grid;
            ctx.strokeRect(left, top, board.cell, board.cell);
            if (terminal(x, y)) {
              ctx.fillStyle = C.paper;
              ctx.font = "600 15px system-ui, sans-serif";
              ctx.textAlign = "center";
              ctx.fillText(
                x === goal.x && y === goal.y ? "+1" : "−1",
                left + board.cell / 2,
                top + board.cell / 2 + 5,
              );
            } else {
              var candidates = actions.map(function (_, action) {
                return actionValue(solution.values, x, y, action);
              });
              var bestAction = candidates.indexOf(Math.max.apply(null, candidates));
              drawPolicyArrow(ctx, left + board.cell / 2, top + board.cell / 2 - 5, bestAction);
              small(ctx, solution.values[cellKey].toFixed(2), left + board.cell / 2, top + board.cell - 9, C.muted, "center");
            }
          }
        }
        title(ctx, "Сжатие ошибки", convergence.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(convergence.x, convergence.y, convergence.w, convergence.h);
        var full = solve(50).changes;
        var maxChange = Math.max.apply(null, full);
        line(ctx, full.map(function (value, index) {
          return {
            x: convergence.x + index / 49 * convergence.w,
            y: convergence.y + convergence.h - value / maxChange * (convergence.h - 15),
          };
        }), C.blue, 2.4);
        var markerX = convergence.x + (state.iterations - 1) / 49 * convergence.w;
        line(ctx, [
          { x: markerX, y: convergence.y },
          { x: markerX, y: convergence.y + convergence.h },
        ], C.gold, 2);
        small(ctx, "1", convergence.x, convergence.y + convergence.h + 20);
        small(ctx, "50 итераций", convergence.x + convergence.w, convergence.y + convergence.h + 20, C.muted, "right");
        ui.output.set([
          { label: "итерация", value: String(state.iterations), color: C.gold },
          { label: "V старта", value: solution.values[key(0, 4)].toFixed(3), color: C.blue },
          { label: "последнее изменение", value: solution.changes[solution.changes.length - 1].toFixed(4) },
          { label: "режим", value: state.slip > 0.25 ? "скользкий" : "управляемый" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Дисконт γ", min: 0.1, max: 0.99, step: 0.01, value: state.gamma,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.gamma = value; draw(); });
      K.slider(ui.controls, {
        label: "Вероятность скольжения", min: 0, max: 0.6, step: 0.01, value: state.slip,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.slip = value; draw(); });
      K.slider(ui.controls, {
        label: "Итераций Беллмана", min: 1, max: 50, step: 1, value: state.iterations,
      }, function (value) { state.iterations = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildBandit(root) {
      var ui = setup(
        root,
        "Слева растёт накопленный regret, справа видны частоты выбора рук. Истинные средние скрыты до окончания серии.",
        "Стратегии многорукого бандита",
        "Один удачный запуск ничего не решает: меняйте серию и сравнивайте распределение regret.",
        490,
      );
      var state = { method: "ucb", horizon: 500, gap: 0.12, exploration: 0.12, seed: 4 };
      var chart = { x: 55, y: 60, w: 575, h: 315 };
      var bars = { x: 690, y: 90, w: 160, h: 260 };

      function normalSample(index, salt) {
        var total = 0;
        for (var i = 0; i < 12; i += 1) total += seeded(index * 17 + i, salt);
        return total - 6;
      }

      function simulate() {
        var means = [0.42, 0.46, 0.46 + state.gap];
        var counts = [0, 0, 0];
        var rewards = [0, 0, 0];
        var alpha = [1, 1, 1];
        var beta = [1, 1, 1];
        var regret = [];
        var cumulative = 0;
        for (var t = 1; t <= state.horizon; t += 1) {
          var action = 0;
          if (t <= 3) action = t - 1;
          else if (state.method === "epsilon") {
            if (seeded(t + state.seed * 1193, 51) < state.exploration) {
              action = Math.floor(seeded(t + state.seed * 1193, 52) * 3);
            } else {
              var averages = rewards.map(function (sum, index) { return sum / counts[index]; });
              action = averages.indexOf(Math.max.apply(null, averages));
            }
          } else if (state.method === "ucb") {
            var scores = rewards.map(function (sum, index) {
              return sum / counts[index]
                + state.exploration * 3 * Math.sqrt(2 * Math.log(t) / counts[index]);
            });
            action = scores.indexOf(Math.max.apply(null, scores));
          } else {
            var samples = alpha.map(function (a, index) {
              var b = beta[index];
              var mean = a / (a + b);
              var deviation = Math.sqrt(a * b / (Math.pow(a + b, 2) * (a + b + 1)));
              return clamp(mean + normalSample(t * 7 + index + state.seed * 101, 53) * deviation, 0, 1);
            });
            action = samples.indexOf(Math.max.apply(null, samples));
          }
          var success = seeded(t + state.seed * 10009, 54 + action) < means[action] ? 1 : 0;
          counts[action] += 1;
          rewards[action] += success;
          alpha[action] += success;
          beta[action] += 1 - success;
          cumulative += Math.max.apply(null, means) - means[action];
          regret.push(cumulative);
        }
        return { means: means, counts: counts, rewards: rewards, regret: regret };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 490);
        var result = simulate();
        title(ctx, "Накопленный regret", chart.x, 34);
        ctx.fillStyle = C.wash; ctx.fillRect(chart.x, chart.y, chart.w, chart.h);
        var maxRegret = Math.max(1, result.regret[result.regret.length - 1] * 1.08);
        line(ctx, result.regret.map(function (value, index) {
          return {
            x: chart.x + index / Math.max(1, result.regret.length - 1) * chart.w,
            y: chart.y + chart.h - value / maxRegret * chart.h,
          };
        }), C.red, 2.6);
        small(ctx, "0", chart.x, chart.y + chart.h + 20);
        small(ctx, String(state.horizon), chart.x + chart.w, chart.y + chart.h + 20, C.muted, "right");
        title(ctx, "Выбор рук", bars.x, 34);
        result.counts.forEach(function (count, index) {
          var left = bars.x + index * 55;
          var height = count / state.horizon * bars.h;
          ctx.fillStyle = [C.blue, C.gold, C.green][index];
          ctx.fillRect(left, bars.y + bars.h - height, 36, height);
          small(ctx, String.fromCharCode(65 + index), left + 18, bars.y + bars.h + 20, C.ink, "center");
          small(ctx, String(count), left + 18, bars.y + bars.h - height - 8, C.muted, "center");
        });
        var best = result.means.indexOf(Math.max.apply(null, result.means));
        var bestEstimate = result.rewards[best] / Math.max(1, result.counts[best]);
        ui.output.set([
          { label: "алгоритм", value: state.method === "epsilon" ? "ε-greedy" : state.method === "ucb" ? "UCB" : "Thompson" },
          { label: "regret", value: result.regret[result.regret.length - 1].toFixed(2), color: C.red },
          { label: "лучшая рука", value: String.fromCharCode(65 + best), color: C.green },
          { label: "её оценка", value: bestEstimate.toFixed(3) },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Алгоритм",
        value: state.method,
        options: [
          { label: "ε-greedy", value: "epsilon" },
          { label: "UCB", value: "ucb" },
          { label: "Thompson", value: "thompson" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Горизонт", min: 50, max: 2000, step: 50, value: state.horizon,
      }, function (value) { state.horizon = value; draw(); });
      K.slider(ui.controls, {
        label: "Разрыв лучшей руки", min: 0.01, max: 0.35, step: 0.01, value: state.gap,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.gap = value; draw(); });
      K.slider(ui.controls, {
        label: "Сила исследования", min: 0.01, max: 0.5, step: 0.01, value: state.exploration,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.exploration = value; draw(); });
      K.slider(ui.controls, {
        label: "Серия", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildActorCritic(root) {
      var ui = setup(
        root,
        "Кривая слева — средний возврат последних эпизодов. Справа стрелки показывают выученную стратегию.",
        "Q-learning и actor–critic",
        "Critic снижает шум обновления actor, но систематическая ошибка critic может уверенно вести стратегию в неверную сторону.",
        500,
      );
      var state = { method: "actor", episodes: 260, exploration: 0.18, critic: 0.7, seed: 3 };
      var cols = 6;
      var rows = 5;
      var walls = { "2,1": true, "2,2": true, "4,3": true };
      var start = 24;
      var goal = 5;
      var pit = 29;
      var directions = [
        { x: 1, y: 0, symbol: "→" },
        { x: 0, y: 1, symbol: "↓" },
        { x: -1, y: 0, symbol: "←" },
        { x: 0, y: -1, symbol: "↑" },
      ];
      function cellKey(index) {
        return (index % cols) + "," + Math.floor(index / cols);
      }
      function isWall(index) { return Boolean(walls[cellKey(index)]); }
      function transition(index, action, random) {
        var usedAction = random < 0.1 ? Math.floor(random * 40) % 4 : action;
        var x = index % cols;
        var y = Math.floor(index / cols);
        var nx = x + directions[usedAction].x;
        var ny = y + directions[usedAction].y;
        if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) return index;
        var next = ny * cols + nx;
        return isWall(next) ? index : next;
      }
      function softmax(values) {
        var max = Math.max.apply(null, values);
        var exp = values.map(function (value) { return Math.exp(value - max); });
        var sum = exp.reduce(function (a, b) { return a + b; }, 0);
        return exp.map(function (value) { return value / sum; });
      }
      function choose(probabilities, random) {
        var cumulative = 0;
        for (var i = 0; i < probabilities.length; i += 1) {
          cumulative += probabilities[i];
          if (random <= cumulative) return i;
        }
        return probabilities.length - 1;
      }
      function train() {
        var states = cols * rows;
        var q = new Array(states).fill(0).map(function () { return [0, 0, 0, 0]; });
        var preferences = new Array(states).fill(0).map(function () { return [0, 0, 0, 0]; });
        var values = new Array(states).fill(0);
        var returns = [];
        var lastTd = 0;
        for (var episode = 0; episode < state.episodes; episode += 1) {
          var current = start;
          var total = 0;
          for (var step = 0; step < 70; step += 1) {
            var randomAction = seeded(episode * 101 + step + state.seed * 10007, 61);
            var action;
            var probabilities;
            if (state.method === "q") {
              if (randomAction < state.exploration) action = Math.floor(seeded(episode * 101 + step, 62) * 4);
              else action = q[current].indexOf(Math.max.apply(null, q[current]));
            } else {
              probabilities = softmax(preferences[current].map(function (value) {
                return value / Math.max(0.08, state.exploration);
              }));
              action = choose(probabilities, randomAction);
            }
            var next = transition(current, action, seeded(episode * 101 + step, 63));
            var reward = next === goal ? 1 : next === pit ? -1 : -0.018;
            total += reward;
            if (state.method === "q") {
              var target = reward + ((next === goal || next === pit) ? 0 : 0.94 * Math.max.apply(null, q[next]));
              lastTd = target - q[current][action];
              q[current][action] += 0.18 * lastTd;
            } else {
              var criticTarget = reward + ((next === goal || next === pit) ? 0 : 0.94 * values[next]);
              lastTd = criticTarget - values[current];
              values[current] += (0.03 + 0.17 * state.critic) * lastTd;
              var actorStep = 0.055 * lastTd;
              preferences[current].forEach(function (_, candidate) {
                preferences[current][candidate] += actorStep * ((candidate === action ? 1 : 0) - probabilities[candidate]);
              });
            }
            current = next;
            if (current === goal || current === pit) break;
          }
          returns.push(total);
        }
        var policy = new Array(states).fill(0);
        for (var index = 0; index < states; index += 1) {
          var scores = state.method === "q" ? q[index] : preferences[index];
          policy[index] = scores.indexOf(Math.max.apply(null, scores));
        }
        return { returns: returns, policy: policy, q: q, values: values, lastTd: lastTd };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var result = train();
        var chart = { x: 50, y: 65, w: 480, h: 300 };
        title(ctx, "Возврат по эпизодам", chart.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(chart.x, chart.y, chart.w, chart.h);
        var smooth = result.returns.map(function (_, index) {
          var from = Math.max(0, index - 14);
          var slice = result.returns.slice(from, index + 1);
          return slice.reduce(function (a, b) { return a + b; }, 0) / slice.length;
        });
        line(ctx, smooth.map(function (value, index) {
          return {
            x: chart.x + index / Math.max(1, smooth.length - 1) * chart.w,
            y: chart.y + chart.h - (clamp(value, -1.5, 1) + 1.5) / 2.5 * chart.h,
          };
        }), C.blue, 2.3);
        line(ctx, [
          { x: chart.x, y: chart.y + chart.h * 0.4 },
          { x: chart.x + chart.w, y: chart.y + chart.h * 0.4 },
        ], C.axis, 1, [4, 4]);
        small(ctx, "0", chart.x, chart.y + chart.h + 20);
        small(ctx, String(state.episodes), chart.x + chart.w, chart.y + chart.h + 20, C.muted, "right");

        var board = { x: 590, y: 65, cell: 48 };
        title(ctx, "Выученная стратегия", board.x, 35);
        for (var y = 0; y < rows; y += 1) {
          for (var x = 0; x < cols; x += 1) {
            var index = y * cols + x;
            var left = board.x + x * board.cell;
            var top = board.y + y * board.cell;
            if (isWall(index)) ctx.fillStyle = C.axis;
            else if (index === goal) ctx.fillStyle = rgba(C.green, 0.65);
            else if (index === pit) ctx.fillStyle = rgba(C.red, 0.65);
            else ctx.fillStyle = C.wash;
            ctx.fillRect(left, top, board.cell, board.cell);
            ctx.strokeStyle = C.grid; ctx.strokeRect(left, top, board.cell, board.cell);
            if (!isWall(index)) {
              ctx.fillStyle = index === goal || index === pit ? C.paper : C.ink;
              ctx.font = "600 16px system-ui, sans-serif";
              ctx.textAlign = "center";
              ctx.fillText(
                index === goal ? "+1" : index === pit ? "−1" : directions[result.policy[index]].symbol,
                left + board.cell / 2,
                top + board.cell / 2 + 6,
              );
            }
          }
        }
        var lastWindow = result.returns.slice(-30);
        var recent = lastWindow.reduce(function (a, b) { return a + b; }, 0) / lastWindow.length;
        ui.output.set([
          { label: "метод", value: state.method === "q" ? "Q-learning" : "actor–critic" },
          { label: "последние 30", value: recent.toFixed(3), color: C.blue },
          { label: "последняя |TD|", value: Math.abs(result.lastTd).toFixed(3), color: C.red },
          { label: "исследование", value: Math.round(state.exploration * 100) + "%" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Метод",
        value: state.method,
        options: [
          { label: "Q-learning", value: "q" },
          { label: "Actor–critic", value: "actor" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Эпизодов", min: 30, max: 600, step: 10, value: state.episodes,
      }, function (value) { state.episodes = value; draw(); });
      K.slider(ui.controls, {
        label: "Исследование", min: 0.05, max: 0.6, step: 0.01, value: state.exploration,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.exploration = value; draw(); });
      K.slider(ui.controls, {
        label: "Скорость critic", min: 0.05, max: 1, step: 0.05, value: state.critic,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.critic = value; draw(); });
      K.slider(ui.controls, {
        label: "Запуск", min: 1, max: 15, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildSelfPlay(root) {
      var ui = setup(
        root,
        "Точка ходит по симплексу стратегий камень–бумага–ножницы. Архив смешивает прошлых соперников и гасит короткий цикл.",
        "Self-play и лига",
        "Победа над последней копией не задаёт устойчивость. Матрица матчей и exploitability видят циклические слабости.",
        500,
      );
      var state = { archive: 0.55, speed: 0.32, rounds: 80 };
      var payoff = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
      ];
      var triangle = {
        rock: { x: 120, y: 390 },
        paper: { x: 500, y: 390 },
        scissors: { x: 310, y: 70 },
      };

      function point(distribution) {
        return {
          x: distribution[0] * triangle.rock.x
            + distribution[1] * triangle.paper.x
            + distribution[2] * triangle.scissors.x,
          y: distribution[0] * triangle.rock.y
            + distribution[1] * triangle.paper.y
            + distribution[2] * triangle.scissors.y,
        };
      }
      function expectedActions(opponent) {
        return payoff.map(function (row) {
          return row.reduce(function (sum, value, index) {
            return sum + value * opponent[index];
          }, 0);
        });
      }
      function normalize(values) {
        var sum = values.reduce(function (a, b) { return a + b; }, 0);
        return values.map(function (value) { return value / sum; });
      }
      function evolve() {
        var current = [0.72, 0.14, 0.14];
        var history = [current.slice()];
        for (var round = 0; round < state.rounds; round += 1) {
          var archiveMean = history.reduce(function (sum, item) {
            return sum.map(function (value, index) { return value + item[index]; });
          }, [0, 0, 0]).map(function (value) { return value / history.length; });
          var latest = history[history.length - 1];
          var opponent = latest.map(function (value, index) {
            return (1 - state.archive) * value + state.archive * archiveMean[index];
          });
          var scores = expectedActions(opponent);
          var maxScore = Math.max.apply(null, scores);
          var response = scores.map(function (value) {
            return Math.exp((value - maxScore) * 5);
          });
          response = normalize(response);
          current = current.map(function (value, index) {
            return (1 - state.speed) * value + state.speed * response[index];
          });
          current = normalize(current);
          history.push(current.slice());
        }
        return history;
      }

      function exploitability(distribution) {
        return Math.max.apply(null, expectedActions(distribution));
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var history = evolve();
        var current = history[history.length - 1];
        title(ctx, "Пространство смешанных стратегий", 65, 35);
        ctx.fillStyle = C.wash;
        ctx.beginPath();
        ctx.moveTo(triangle.rock.x, triangle.rock.y);
        ctx.lineTo(triangle.paper.x, triangle.paper.y);
        ctx.lineTo(triangle.scissors.x, triangle.scissors.y);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        small(ctx, "камень", triangle.rock.x, triangle.rock.y + 24, C.blue, "center");
        small(ctx, "бумага", triangle.paper.x, triangle.paper.y + 24, C.red, "center");
        small(ctx, "ножницы", triangle.scissors.x, triangle.scissors.y - 14, C.green, "center");
        line(ctx, history.map(point), C.gold, 2.2);
        history.filter(function (_, index) { return index % Math.max(1, Math.floor(history.length / 18)) === 0; })
          .forEach(function (distribution) {
            var p = point(distribution);
            ctx.beginPath(); ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
            ctx.fillStyle = rgba(C.gold, 0.65); ctx.fill();
          });
        var finalPoint = point(current);
        ctx.beginPath(); ctx.arc(finalPoint.x, finalPoint.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = C.gold; ctx.fill();
        var equilibrium = point([1 / 3, 1 / 3, 1 / 3]);
        ctx.beginPath(); ctx.arc(equilibrium.x, equilibrium.y, 6, 0, Math.PI * 2);
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.stroke();

        title(ctx, "Текущая смесь", 625, 35);
        var labels = ["камень", "бумага", "ножницы"];
        current.forEach(function (value, index) {
          var y = 85 + index * 72;
          small(ctx, labels[index], 625, y, [C.blue, C.red, C.green][index]);
          ctx.fillStyle = C.wash; ctx.fillRect(625, y + 14, 220, 22);
          ctx.fillStyle = [C.blue, C.red, C.green][index];
          ctx.fillRect(625, y + 14, 220 * value, 22);
          small(ctx, (value * 100).toFixed(1) + "%", 855, y + 30, C.ink, "right");
        });
        var exploit = exploitability(current);
        var average = history.reduce(function (sum, item) {
          return sum.map(function (value, index) { return value + item[index]; });
        }, [0, 0, 0]).map(function (value) { return value / history.length; });
        ui.output.set([
          { label: "exploitability", value: exploit.toFixed(3), color: C.red },
          { label: "архив", value: Math.round(state.archive * 100) + "%" },
          { label: "средняя стратегия", value: average.map(function (v) { return Math.round(v * 100); }).join(" / ") },
          { label: "раундов", value: String(state.rounds), color: C.gold },
        ]);
      }

      K.slider(ui.controls, {
        label: "Доля архива", min: 0, max: 1, step: 0.01, value: state.archive,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.archive = value; draw(); });
      K.slider(ui.controls, {
        label: "Скорость лучшего ответа", min: 0.05, max: 0.8, step: 0.01, value: state.speed,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.speed = value; draw(); });
      K.slider(ui.controls, {
        label: "Раундов", min: 10, max: 240, step: 5, value: state.rounds,
      }, function (value) { state.rounds = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "65": buildPageRank,
      "66": buildRandomWalk,
      "67": buildMcmc,
      "68": buildCipher,
      "69": buildBellman,
      "70": buildBandit,
      "71": buildActorCritic,
      "72": buildSelfPlay,
    };

    K.register("g11-rl", function (root, options) {
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
