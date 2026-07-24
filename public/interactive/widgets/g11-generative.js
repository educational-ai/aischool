// Grade 11, module 4. Sequence models, generative AI and agents.
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
      var value = Math.sin(index * 79.113 + salt * 37.911) * 43758.5453;
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
      var canvasState = K.makeCanvas(stage, 920, height || 490, {
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

    function drawArrow(ctx, x1, y1, x2, y2, color, width) {
      var angle = Math.atan2(y2 - y1, x2 - x1);
      line(ctx, [{ x: x1, y: y1 }, { x: x2, y: y2 }], color, width || 1.8);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - Math.cos(angle - 0.5) * 9, y2 - Math.sin(angle - 0.5) * 9);
      ctx.lineTo(x2 - Math.cos(angle + 0.5) * 9, y2 - Math.sin(angle + 0.5) * 9);
      ctx.closePath();
      ctx.fill();
    }

    function buildRnn(root) {
      var ui = setup(
        root,
        "Импульс приходит в начале. Синяя линия — состояние, красная — чувствительность последнего шага к раннему сигналу.",
        "Память рекуррентной ячейки",
        "Одинаковая ячейка работает на каждом шаге; дальняя память зависит от произведения локальных производных.",
        485,
      );
      var state = { memory: 0.72, length: 32, impulse: 1 };
      var plot = { x: 60, y: 65, w: 800, h: 300 };

      function series() {
        var hidden = [];
        var sensitivity = [];
        var h = 0;
        for (var t = 0; t < state.length; t += 1) {
          var input = t === 2 ? state.impulse : 0;
          var pre = state.memory * h + input;
          h = Math.tanh(pre);
          hidden.push(h);
          sensitivity.push(Math.pow(state.memory, Math.max(0, t - 2)));
        }
        return { hidden: hidden, sensitivity: sensitivity };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        var values = series();
        title(ctx, "Сигнал во времени", plot.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        function sx(index) { return plot.x + index / Math.max(1, state.length - 1) * plot.w; }
        function sy(value) { return plot.y + plot.h / 2 - value * plot.h * 0.39; }
        line(ctx, [
          { x: plot.x, y: sy(0) },
          { x: plot.x + plot.w, y: sy(0) },
        ], C.axis, 1);
        line(ctx, values.hidden.map(function (value, index) {
          return { x: sx(index), y: sy(value) };
        }), C.blue, 2.7);
        line(ctx, values.sensitivity.map(function (value, index) {
          return { x: sx(index), y: sy(-value) };
        }), C.red, 2.3);
        ctx.fillStyle = C.gold;
        ctx.fillRect(sx(2) - 4, sy(0), 8, sy(state.impulse) - sy(0));
        small(ctx, "hidden state", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "sensitivity (внизу)", plot.x + 115, plot.y + 22, C.red);
        small(ctx, "0", plot.x, plot.y + plot.h + 20);
        small(ctx, String(state.length - 1), plot.x + plot.w, plot.y + plot.h + 20, C.muted, "right");
        var last = values.hidden[values.hidden.length - 1];
        var lastSensitivity = values.sensitivity[values.sensitivity.length - 1];
        var halfLife = state.memory > 0 && state.memory < 1
          ? Math.log(0.5) / Math.log(state.memory)
          : Infinity;
        ui.output.set([
          { label: "последнее h", value: last.toFixed(4), color: C.blue },
          { label: "чувствительность", value: lastSensitivity.toExponential(2), color: C.red },
          { label: "полураспад", value: Number.isFinite(halfLife) ? halfLife.toFixed(1) + " шага" : "не затухает" },
          { label: "параметров по времени", value: "один набор" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Сила памяти", min: 0, max: 1.08, step: 0.01, value: state.memory,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.memory = value; draw(); });
      K.slider(ui.controls, {
        label: "Длина", min: 5, max: 100, step: 1, value: state.length,
      }, function (value) { state.length = value; draw(); });
      K.slider(ui.controls, {
        label: "Импульс", min: -1.5, max: 1.5, step: 0.05, value: state.impulse,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.impulse = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildLstm(root) {
      var ui = setup(
        root,
        "Золотая линия — память LSTM, синяя — обычная рекуррентная связь. Точка запроса стоит после пустой паузы.",
        "Ворота длинной памяти",
        "Forget gate создаёт управляемый путь сохранения; output gate может скрыть память, не стирая её.",
        490,
      );
      var state = { forget: 0.94, gap: 35, rnn: 0.82, inputGate: 0.9 };
      var plot = { x: 60, y: 65, w: 800, h: 285 };

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 490);
        title(ctx, "Сохранённый импульс", plot.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var length = Math.max(5, state.gap + 3);
        var lstm = [];
        var rnn = [];
        for (var t = 0; t < length; t += 1) {
          lstm.push(t < 2 ? 0 : state.inputGate * Math.pow(state.forget, t - 2));
          rnn.push(t < 2 ? 0 : state.inputGate * Math.pow(state.rnn, t - 2));
        }
        function sx(index) { return plot.x + index / (length - 1) * plot.w; }
        function sy(value) { return plot.y + plot.h - value * (plot.h - 35) - 15; }
        line(ctx, lstm.map(function (value, index) {
          return { x: sx(index), y: sy(value) };
        }), C.gold, 2.8);
        line(ctx, rnn.map(function (value, index) {
          return { x: sx(index), y: sy(value) };
        }), C.blue, 2.4);
        var queryX = sx(length - 1);
        line(ctx, [{ x: queryX, y: plot.y }, { x: queryX, y: plot.y + plot.h }], C.red, 2);
        small(ctx, "LSTM cell", plot.x + 15, plot.y + 22, C.gold);
        small(ctx, "обычная RNN", plot.x + 100, plot.y + 22, C.blue);
        small(ctx, "запись", sx(2), plot.y + plot.h + 20, C.muted, "center");
        small(ctx, "запрос", queryX, plot.y + plot.h + 20, C.red, "right");

        title(ctx, "Ворота", 60, 395);
        var gates = [
          { label: "input", value: state.inputGate, color: C.green },
          { label: "forget", value: state.forget, color: C.gold },
          { label: "output", value: 1, color: C.blue },
        ];
        gates.forEach(function (gate, index) {
          var x = 160 + index * 210;
          small(ctx, gate.label, x, 396, gate.color);
          ctx.fillStyle = C.wash; ctx.fillRect(x, 410, 150, 20);
          ctx.fillStyle = gate.color; ctx.fillRect(x, 410, 150 * gate.value, 20);
        });
        var saved = lstm[lstm.length - 1];
        ui.output.set([
          { label: "сохранила LSTM", value: (saved * 100).toFixed(1) + "%", color: C.gold },
          { label: "сохранила RNN", value: (rnn[rnn.length - 1] * 100).toFixed(1) + "%", color: C.blue },
          { label: "пауза", value: state.gap + " шагов" },
          { label: "forget^gap", value: Math.pow(state.forget, state.gap).toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Forget gate", min: 0.5, max: 1, step: 0.005, value: state.forget,
        format: function (v) { return v.toFixed(3); },
      }, function (value) { state.forget = value; draw(); });
      K.slider(ui.controls, {
        label: "Длина паузы", min: 1, max: 100, step: 1, value: state.gap,
      }, function (value) { state.gap = value; draw(); });
      K.slider(ui.controls, {
        label: "Рекуррентный вес RNN", min: 0.5, max: 1, step: 0.01, value: state.rnn,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.rnn = value; draw(); });
      K.slider(ui.controls, {
        label: "Input gate", min: 0, max: 1, step: 0.01, value: state.inputGate,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.inputGate = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function learnBpe(corpus, mergesCount) {
      var sequences = corpus.split(/\s+/).filter(Boolean).map(function (word) {
        return ("▁" + word).split("");
      });
      var merges = [];
      for (var round = 0; round < mergesCount; round += 1) {
        var pairCounts = {};
        sequences.forEach(function (sequence) {
          for (var i = 0; i < sequence.length - 1; i += 1) {
            var key = sequence[i] + "\u0001" + sequence[i + 1];
            pairCounts[key] = (pairCounts[key] || 0) + 1;
          }
        });
        var pairs = Object.keys(pairCounts);
        if (!pairs.length) break;
        pairs.sort(function (a, b) { return pairCounts[b] - pairCounts[a]; });
        var best = pairs[0].split("\u0001");
        merges.push(best);
        sequences = sequences.map(function (sequence) {
          var next = [];
          for (var j = 0; j < sequence.length; j += 1) {
            if (j < sequence.length - 1 && sequence[j] === best[0] && sequence[j + 1] === best[1]) {
              next.push(best[0] + best[1]);
              j += 1;
            } else next.push(sequence[j]);
          }
          return next;
        });
      }
      return merges;
    }

    function applyBpe(text, merges) {
      var tokens = [];
      text.split(/\s+/).filter(Boolean).forEach(function (word) {
        var sequence = ("▁" + word).split("");
        merges.forEach(function (pair) {
          var next = [];
          for (var i = 0; i < sequence.length; i += 1) {
            if (i < sequence.length - 1 && sequence[i] === pair[0] && sequence[i + 1] === pair[1]) {
              next.push(pair[0] + pair[1]);
              i += 1;
            } else next.push(sequence[i]);
          }
          sequence = next;
        });
        tokens = tokens.concat(sequence);
      });
      return tokens;
    }

    function buildTokens(root) {
      var ui = setup(
        root,
        "Каждый прямоугольник — токен. Слияния обучаются на малом корпусе, затем без изменений применяются к выбранной строке.",
        "BPE-токенизатор",
        "Большой словарь сокращает цепь на знакомом корпусе, но редкие языки и имена всё равно могут дробиться.",
        485,
      );
      var state = { merges: 18, sample: "rare" };
      var corpus = (
        "мама мамы мамин машина машины машинный " +
        "математика математический модель модели моделирование " +
        "школа школьник школьная русский текст токен токены"
      );
      var samples = {
        common: "мама читает школьный текст",
        rare: "электрокардиографический сигнал",
        mixed: "model обучает tokenizer 2026",
      };
      var palette = [C.blue, C.red, C.green, C.gold, C.violet];

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 485);
        var merges = learnBpe(corpus, state.merges);
        var text = samples[state.sample];
        var tokens = applyBpe(text, merges);
        title(ctx, "Строка", 60, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(60, 58, 800, 60);
        ctx.fillStyle = C.ink;
        ctx.font = "20px ET Book, Palatino, Georgia, serif";
        ctx.textAlign = "left";
        ctx.fillText(text, 78, 94);
        title(ctx, "Разбиение", 60, 155);
        var x = 60;
        var y = 180;
        var row = 0;
        tokens.forEach(function (token, index) {
          ctx.font = "14px ui-monospace, monospace";
          var label = token.replace("▁", "␠");
          var width = Math.max(38, ctx.measureText(label).width + 22);
          if (x + width > 860) {
            x = 60;
            y += 52;
            row += 1;
          }
          ctx.fillStyle = rgba(palette[index % palette.length], 0.15);
          ctx.fillRect(x, y, width, 34);
          ctx.strokeStyle = palette[index % palette.length];
          ctx.strokeRect(x, y, width, 34);
          ctx.fillStyle = C.ink;
          ctx.textAlign = "center";
          ctx.fillText(label, x + width / 2, y + 22);
          x += width + 7;
        });
        title(ctx, "Последние слияния словаря", 60, 350 + row * 22);
        var recent = merges.slice(-6);
        recent.forEach(function (pair, index) {
          small(ctx, pair[0].replace("▁", "␠") + " + " + pair[1] + " → " + pair.join("").replace("▁", "␠"), 60 + (index % 3) * 270, 380 + row * 22 + Math.floor(index / 3) * 27, [C.blue, C.red, C.green][index % 3]);
        });
        var characters = text.replace(/\s/g, "").length;
        ui.output.set([
          { label: "токенов", value: String(tokens.length), color: C.blue },
          { label: "букв / токен", value: (characters / tokens.length).toFixed(2) },
          { label: "слияний", value: String(merges.length), color: C.gold },
          { label: "режим", value: state.sample === "common" ? "частая строка" : state.sample === "rare" ? "редкое слово" : "смешанный текст" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Строка",
        value: state.sample,
        options: [
          { label: "Частая", value: "common" },
          { label: "Редкая", value: "rare" },
          { label: "Смешанная", value: "mixed" },
        ],
      }, function (value) { state.sample = value; draw(); });
      K.slider(ui.controls, {
        label: "BPE-слияний", min: 0, max: 45, step: 1, value: state.merges,
      }, function (value) { state.merges = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildAttention(root) {
      var ui = setup(
        root,
        "Выбранная строка — query. Цвет клетки показывает attention weight после softmax; causal mask закрывает будущее.",
        "Карта attention",
        "Similarity выбирает позиции, value переносит содержимое. Большой вес помогает диагностике, но не служит готовым объяснением.",
        500,
      );
      var state = { query: 6, temperature: 0.8, causal: "off" };
      var words = ["Маша", "положила", "книгу", "на", "стол", "и", "она", "упала"];

      function rawScore(query, key) {
        var distance = Math.abs(query - key);
        var score = 1.5 - distance * 0.35;
        if (query === key) score += 1.2;
        if ((query === 6 && key === 2) || (query === 7 && key === 2)) score += 2.7;
        if (query === 7 && key === 4) score += 1.3;
        if (query === 2 && key === 0) score += 0.9;
        return score;
      }

      function weightsFor(query) {
        var logits = words.map(function (_, key) {
          if (state.causal === "on" && key > query) return -1e9;
          return rawScore(query, key) / state.temperature;
        });
        var max = Math.max.apply(null, logits);
        var exp = logits.map(function (value) { return value < -1e8 ? 0 : Math.exp(value - max); });
        var sum = exp.reduce(function (a, b) { return a + b; }, 0);
        return exp.map(function (value) { return value / sum; });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var cell = 44;
        var left = 220;
        var top = 75;
        title(ctx, "Query × key", 60, 35);
        words.forEach(function (word, index) {
          ctx.save();
          ctx.translate(left + index * cell + 28, top - 10);
          ctx.rotate(-0.55);
          small(ctx, word, 0, 0, C.muted);
          ctx.restore();
          small(ctx, word, left - 14, top + index * cell + 27, index === state.query ? C.red : C.muted, "right");
        });
        for (var query = 0; query < words.length; query += 1) {
          var weights = weightsFor(query);
          weights.forEach(function (weight, key) {
            ctx.fillStyle = rgba(query === state.query ? C.red : C.blue, 0.05 + weight * 0.9);
            ctx.fillRect(left + key * cell, top + query * cell, cell - 2, cell - 2);
            if (query === state.query) {
              small(ctx, Math.round(weight * 100), left + key * cell + cell / 2, top + query * cell + 27, weight > 0.45 ? C.paper : C.ink, "center");
            }
          });
        }
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 2;
        ctx.strokeRect(left - 2, top + state.query * cell - 2, words.length * cell + 2, cell + 2);

        var selected = weightsFor(state.query);
        title(ctx, "Вес контекста", 650, 35);
        selected.forEach(function (weight, index) {
          var y = 75 + index * 44;
          small(ctx, words[index], 650, y + 16, C.ink);
          ctx.fillStyle = C.wash; ctx.fillRect(720, y, 135, 22);
          ctx.fillStyle = index === state.query ? C.gold : C.blue;
          ctx.fillRect(720, y, 135 * weight, 22);
        });
        var entropy = -selected.reduce(function (sum, value) {
          return sum + (value > 0 ? value * Math.log(value) : 0);
        }, 0);
        var leader = selected.indexOf(Math.max.apply(null, selected));
        ui.output.set([
          { label: "query", value: words[state.query], color: C.red },
          { label: "главный key", value: words[leader], color: C.blue },
          { label: "его вес", value: selected[leader].toFixed(3) },
          { label: "энтропия", value: entropy.toFixed(3) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Позиция query", min: 0, max: words.length - 1, step: 1, value: state.query,
        format: function (v) { return words[v]; },
      }, function (value) { state.query = value; draw(); });
      K.slider(ui.controls, {
        label: "Температура", min: 0.2, max: 2.5, step: 0.05, value: state.temperature,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.temperature = value; draw(); });
      K.segmented(ui.controls, {
        label: "Маска",
        value: state.causal,
        options: [
          { label: "Полный контекст", value: "off" },
          { label: "Causal", value: "on" },
        ],
      }, function (value) { state.causal = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildTransformerBlock(root) {
      var ui = setup(
        root,
        "Сигнал проходит слева направо. Толщина обходной линии показывает residual, нижний граф — норму после каждого блока.",
        "Конструктор transformer block",
        "Attention смешивает позиции, MLP меняет признаки внутри позиции, residual сохраняет прямой путь.",
        500,
      );
      var state = { heads: 4, depth: 6, residual: 0.9, mlp: 4 };
      var boxes = [
        { label: "Norm", color: C.muted },
        { label: "Attention", color: C.blue },
        { label: "+ residual", color: C.gold },
        { label: "Norm", color: C.muted },
        { label: "MLP", color: C.green },
        { label: "+ residual", color: C.gold },
      ];

      function normAt(layer) {
        var attentionGain = 0.05 * Math.log2(state.heads + 1);
        var mlpGain = 0.025 * state.mlp;
        var residualStability = 0.18 * state.residual;
        return Math.exp((attentionGain + mlpGain - residualStability) * layer);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        title(ctx, "Один блок", 55, 35);
        var startX = 55;
        var y = 90;
        boxes.forEach(function (box, index) {
          var x = startX + index * 135;
          ctx.fillStyle = rgba(box.color, box.label.indexOf("residual") >= 0 ? 0.16 : 0.1);
          ctx.fillRect(x, y, 105, 58);
          ctx.strokeStyle = box.color;
          ctx.strokeRect(x, y, 105, 58);
          ctx.fillStyle = C.ink;
          ctx.font = "600 13px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(box.label, x + 52.5, y + 34);
          if (index < boxes.length - 1) drawArrow(ctx, x + 105, y + 29, x + 132, y + 29, C.muted, 1.4);
        });
        ctx.save();
        ctx.strokeStyle = C.gold;
        ctx.lineWidth = 1 + state.residual * 4;
        ctx.beginPath();
        ctx.moveTo(startX + 20, y - 8);
        ctx.bezierCurveTo(startX + 80, y - 55, startX + 320, y - 55, startX + 320, y - 8);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(startX + 425, y - 8);
        ctx.bezierCurveTo(startX + 485, y - 55, startX + 720, y - 55, startX + 720, y - 8);
        ctx.stroke();
        ctx.restore();
        small(ctx, state.heads + " heads", startX + 187, y + 79, C.blue, "center");
        small(ctx, "ширина ×" + state.mlp, startX + 592, y + 79, C.green, "center");

        var plot = { x: 55, y: 250, w: 800, h: 160 };
        title(ctx, "Норма сигнала по глубине", plot.x, 220);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var points = [];
        for (var layer = 0; layer <= state.depth; layer += 1) {
          points.push({
            x: plot.x + layer / state.depth * plot.w,
            y: plot.y + plot.h - clamp(normAt(layer), 0, 3) / 3 * plot.h,
          });
        }
        line(ctx, points, C.red, 2.5);
        line(ctx, [
          { x: plot.x, y: plot.y + plot.h * 2 / 3 },
          { x: plot.x + plot.w, y: plot.y + plot.h * 2 / 3 },
        ], C.axis, 1, [4, 4]);
        small(ctx, "0", plot.x, plot.y + plot.h + 20);
        small(ctx, state.depth + " блоков", plot.x + plot.w, plot.y + plot.h + 20, C.muted, "right");
        var attentionCost = state.depth * state.heads;
        ui.output.set([
          { label: "глубина", value: String(state.depth) },
          { label: "attention-графов", value: String(attentionCost), color: C.blue },
          { label: "норма на выходе", value: normAt(state.depth).toFixed(2), color: C.red },
          { label: "residual", value: Math.round(state.residual * 100) + "%", color: C.gold },
        ]);
      }

      K.slider(ui.controls, {
        label: "Голов внимания", min: 1, max: 16, step: 1, value: state.heads,
      }, function (value) { state.heads = value; draw(); });
      K.slider(ui.controls, {
        label: "Блоков", min: 1, max: 24, step: 1, value: state.depth,
      }, function (value) { state.depth = value; draw(); });
      K.slider(ui.controls, {
        label: "Residual", min: 0, max: 1, step: 0.01, value: state.residual,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.residual = value; draw(); });
      K.slider(ui.controls, {
        label: "Ширина MLP", min: 1, max: 8, step: 1, value: state.mlp,
        format: function (v) { return "×" + v; },
      }, function (value) { state.mlp = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildPretraining(root) {
      var ui = setup(
        root,
        "Полоса показывает реальную смесь токенов, шкалы — ожидаемые навыки. Дубли съедают бюджет без нового содержания.",
        "Смесь предобучения",
        "Вес источника задаёт частоту его градиентов. Quality-фильтр и dedup меняют состав так же сильно, как общий размер.",
        490,
      );
      var state = { code: 20, dialogue: 30, duplicates: 18, budget: 100 };
      var colors = [C.blue, C.red, C.green, C.gold];

      function mixture() {
        var remaining = Math.max(0, 100 - state.code - state.dialogue);
        return [
          state.code,
          state.dialogue,
          remaining * 0.68,
          remaining * 0.32,
        ];
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 490);
        var mix = mixture();
        var labels = ["код", "диалоги", "статьи", "таблицы"];
        title(ctx, "Смесь токенов", 60, 35);
        var x = 60;
        mix.forEach(function (value, index) {
          var width = 800 * value / 100;
          ctx.fillStyle = colors[index];
          ctx.fillRect(x, 65, width, 54);
          if (width > 60) small(ctx, labels[index] + " " + value.toFixed(0) + "%", x + width / 2, 97, C.paper, "center");
          x += width;
        });

        title(ctx, "Ожидаемые способности", 60, 165);
        var skills = [
          { label: "код", value: mix[0] / 35, color: colors[0] },
          { label: "диалог", value: mix[1] / 40, color: colors[1] },
          { label: "факты", value: mix[2] / 45, color: colors[2] },
          { label: "структура", value: (mix[0] + mix[3]) / 40, color: colors[3] },
        ];
        skills.forEach(function (skill, index) {
          var y = 200 + index * 55;
          small(ctx, skill.label, 60, y + 17, C.ink);
          ctx.fillStyle = C.wash; ctx.fillRect(155, y, 450, 23);
          ctx.fillStyle = skill.color; ctx.fillRect(155, y, 450 * clamp(skill.value, 0, 1), 23);
          small(ctx, Math.round(clamp(skill.value, 0, 1) * 100) + "%", 620, y + 17, skill.color);
        });
        title(ctx, "Бюджет", 690, 165);
        var effective = state.budget * (1 - state.duplicates / 100);
        ctx.fillStyle = C.wash; ctx.fillRect(690, 200, 160, 190);
        ctx.fillStyle = rgba(C.red, 0.35);
        ctx.fillRect(710, 220, 55, 150 * state.duplicates / 100);
        ctx.fillStyle = C.blue;
        ctx.fillRect(785, 220 + 150 * (1 - effective / Math.max(1, state.budget)), 55, 150 * effective / Math.max(1, state.budget));
        small(ctx, "дубли", 737, 410, C.red, "center");
        small(ctx, "новое", 812, 410, C.blue, "center");
        ui.output.set([
          { label: "бюджет", value: state.budget + " млрд токенов" },
          { label: "эффективно новых", value: effective.toFixed(1) + " млрд", color: C.blue },
          { label: "повторов", value: state.duplicates + "%", color: C.red },
          { label: "крупнейший источник", value: labels[mix.indexOf(Math.max.apply(null, mix))] },
        ]);
      }

      K.slider(ui.controls, {
        label: "Доля кода", min: 0, max: 70, step: 1, value: state.code,
        unit: "%",
      }, function (value) {
        state.code = Math.min(value, 100 - state.dialogue);
        draw();
      });
      K.slider(ui.controls, {
        label: "Доля диалогов", min: 0, max: 70, step: 1, value: state.dialogue,
        unit: "%",
      }, function (value) {
        state.dialogue = Math.min(value, 100 - state.code);
        draw();
      });
      K.slider(ui.controls, {
        label: "Дубли", min: 0, max: 80, step: 1, value: state.duplicates,
        unit: "%",
      }, function (value) { state.duplicates = value; draw(); });
      K.slider(ui.controls, {
        label: "Бюджет, млрд токенов", min: 10, max: 500, step: 10, value: state.budget,
      }, function (value) { state.budget = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildScaling(root) {
      var ui = setup(
        root,
        "Фон — прогнозный loss, золотая линия — тот же compute, что у выбранной точки. Белый круг отмечает лучший баланс на этой линии.",
        "Scaling law и compute",
        "Compute-optimal точка зависит от fitted law и качества корпуса; число токенов на параметр не служит вечной константой.",
        500,
      );
      var state = { parameters: 70, tokens: 1400, quality: 0.8 };
      var plot = { x: 75, y: 55, w: 700, h: 340 };

      function predictedLoss(parameters, tokens) {
        var effectiveTokens = Math.max(1, tokens * state.quality);
        return 0.92 + 1.45 / Math.pow(parameters, 0.27) + 1.7 / Math.pow(effectiveTokens / 10, 0.22);
      }

      function logX(parameters) {
        return plot.x + Math.log10(parameters) / Math.log10(400) * plot.w;
      }
      function logY(tokens) {
        return plot.y + plot.h - (Math.log10(tokens) - 1) / 3 * plot.h;
      }

      function optimum(compute) {
        var best = { parameters: 1, tokens: compute, loss: Infinity };
        for (var i = 0; i <= 600; i += 1) {
          var parameters = Math.pow(10, i / 600 * Math.log10(400));
          var tokens = compute / parameters;
          if (tokens < 10 || tokens > 10000) continue;
          var loss = predictedLoss(parameters, tokens);
          if (loss < best.loss) best = { parameters: parameters, tokens: tokens, loss: loss };
        }
        return best;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        title(ctx, "Прогноз loss по размеру и токенам", plot.x, 30);
        var cellsX = 56;
        var cellsY = 42;
        for (var ix = 0; ix < cellsX; ix += 1) {
          for (var iy = 0; iy < cellsY; iy += 1) {
            var parameters = Math.pow(10, (ix + 0.5) / cellsX * Math.log10(400));
            var tokens = Math.pow(10, 1 + (cellsY - iy - 0.5) / cellsY * 3);
            var loss = predictedLoss(parameters, tokens);
            var value = clamp((loss - 1.3) / 1.45, 0, 1);
            ctx.fillStyle = value > 0.5
              ? rgba(C.red, 0.12 + (value - 0.5) * 0.75)
              : rgba(C.blue, 0.42 - value * 0.55);
            ctx.fillRect(
              plot.x + ix / cellsX * plot.w,
              plot.y + iy / cellsY * plot.h,
              plot.w / cellsX + 1,
              plot.h / cellsY + 1,
            );
          }
        }
        var compute = state.parameters * state.tokens;
        var iso = [];
        for (var j = 0; j <= 200; j += 1) {
          var p = Math.pow(10, j / 200 * Math.log10(400));
          var d = compute / p;
          if (d >= 10 && d <= 10000) iso.push({ x: logX(p), y: logY(d) });
        }
        line(ctx, iso, C.gold, 2.6);
        var best = optimum(compute);
        ctx.beginPath(); ctx.arc(logX(best.parameters), logY(best.tokens), 8, 0, Math.PI * 2);
        ctx.fillStyle = C.paper; ctx.fill();
        ctx.strokeStyle = C.gold; ctx.lineWidth = 2.5; ctx.stroke();
        ctx.beginPath(); ctx.arc(logX(state.parameters), logY(state.tokens), 7, 0, Math.PI * 2);
        ctx.fillStyle = C.ink; ctx.fill();
        [1, 10, 100, 400].forEach(function (value) {
          small(ctx, String(value), logX(value), plot.y + plot.h + 22, C.muted, "center");
        });
        [10, 100, 1000, 10000].forEach(function (value) {
          small(ctx, value >= 1000 ? (value / 1000) + "T" : String(value), plot.x - 13, logY(value) + 4, C.muted, "right");
        });
        small(ctx, "параметры, млрд", plot.x + plot.w / 2, 455, C.muted, "center");
        small(ctx, "токены, млрд", 830, 90, C.muted);
        var currentLoss = predictedLoss(state.parameters, state.tokens);
        ui.output.set([
          { label: "прогноз loss", value: currentLoss.toFixed(3), color: C.red },
          { label: "условный compute", value: (compute / 1000).toFixed(1) + " · 10³", color: C.gold },
          { label: "optimum N", value: best.parameters.toFixed(1) + " млрд" },
          { label: "optimum D", value: best.tokens.toFixed(0) + " млрд" },
        ]);
      }

      K.slider(ui.controls, {
        label: "Параметры, млрд", min: 1, max: 400, step: 1, value: state.parameters,
      }, function (value) { state.parameters = value; draw(); });
      K.slider(ui.controls, {
        label: "Токены, млрд", min: 10, max: 10000, step: 10, value: state.tokens,
      }, function (value) { state.tokens = value; draw(); });
      K.slider(ui.controls, {
        label: "Качество корпуса", min: 0.2, max: 1.2, step: 0.01, value: state.quality,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.quality = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildPreferences(root) {
      var ui = setup(
        root,
        "Рёбра показывают долю побед первого ответа. Баллы справа fit-ятся по всем сравнениям сразу.",
        "Bradley–Terry и граф сравнений",
        "Если сравнения образуют цикл или распадаются на группы, один скрытый reward вынужден усреднять противоречия.",
        500,
      );
      var state = { ab: 72, bc: 64, cd: 58, disagreement: 28 };
      var labels = ["A", "B", "C", "D"];
      var positions = [
        { x: 130, y: 125 },
        { x: 350, y: 80 },
        { x: 350, y: 280 },
        { x: 130, y: 335 },
      ];

      function comparisons() {
        var cycle = 50 + state.disagreement * 0.42;
        return [
          { a: 0, b: 1, win: state.ab / 100, count: 40 },
          { a: 1, b: 2, win: state.bc / 100, count: 40 },
          { a: 2, b: 3, win: state.cd / 100, count: 40 },
          { a: 3, b: 0, win: cycle / 100, count: 40 },
        ];
      }

      function fitRatings() {
        var ratings = [0, 0, 0, 0];
        var edges = comparisons();
        for (var iteration = 0; iteration < 900; iteration += 1) {
          var gradient = [0, 0, 0, 0];
          edges.forEach(function (edge) {
            var probability = 1 / (1 + Math.exp(-(ratings[edge.a] - ratings[edge.b])));
            var error = edge.win - probability;
            gradient[edge.a] += edge.count * error;
            gradient[edge.b] -= edge.count * error;
          });
          ratings = ratings.map(function (value, index) {
            return value + 0.0025 * gradient[index] - 0.0004 * value;
          });
          var mean = ratings.reduce(function (a, b) { return a + b; }, 0) / ratings.length;
          ratings = ratings.map(function (value) { return value - mean; });
        }
        return ratings;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        title(ctx, "Граф голосов", 55, 35);
        var edges = comparisons();
        edges.forEach(function (edge) {
          var from = positions[edge.a];
          var to = positions[edge.b];
          drawArrow(ctx, from.x, from.y, to.x, to.y, edge.win >= 0.5 ? C.blue : C.red, 2);
          small(
            ctx,
            Math.round(edge.win * 100) + " : " + Math.round((1 - edge.win) * 100),
            (from.x + to.x) / 2,
            (from.y + to.y) / 2 - 8,
            C.muted,
            "center",
          );
        });
        positions.forEach(function (position, index) {
          ctx.beginPath(); ctx.arc(position.x, position.y, 30, 0, Math.PI * 2);
          ctx.fillStyle = C.paper; ctx.fill();
          ctx.strokeStyle = [C.blue, C.red, C.green, C.gold][index];
          ctx.lineWidth = 3; ctx.stroke();
          ctx.fillStyle = C.ink;
          ctx.font = "600 18px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(labels[index], position.x, position.y + 6);
        });

        var ratings = fitRatings();
        title(ctx, "Скрытые баллы", 560, 35);
        ratings.forEach(function (rating, index) {
          var y = 80 + index * 80;
          small(ctx, labels[index], 560, y + 18, C.ink);
          ctx.fillStyle = C.wash; ctx.fillRect(600, y, 240, 28);
          var center = 720;
          ctx.fillStyle = [C.blue, C.red, C.green, C.gold][index];
          var width = clamp(Math.abs(rating) / 1.5, 0, 1) * 115;
          ctx.fillRect(rating >= 0 ? center : center - width, y, width, 28);
          line(ctx, [{ x: center, y: y - 4 }, { x: center, y: y + 32 }], C.axis, 1);
          small(ctx, rating.toFixed(2), 855, y + 19, C.muted, "right");
        });
        var predicted = function (a, b) {
          return 1 / (1 + Math.exp(-(ratings[a] - ratings[b])));
        };
        var crossEntropy = edges.reduce(function (sum, edge) {
          var p = clamp(predicted(edge.a, edge.b), 1e-6, 1 - 1e-6);
          return sum - edge.win * Math.log(p) - (1 - edge.win) * Math.log(1 - p);
        }, 0) / edges.length;
        var winner = ratings.indexOf(Math.max.apply(null, ratings));
        ui.output.set([
          { label: "лидер", value: labels[winner], color: [C.blue, C.red, C.green, C.gold][winner] },
          { label: "его reward", value: ratings[winner].toFixed(3) },
          { label: "pairwise loss", value: crossEntropy.toFixed(3), color: C.red },
          { label: "цикл D>A", value: Math.round(edges[3].win * 100) + "%" },
        ]);
      }

      K.slider(ui.controls, {
        label: "A побеждает B", min: 5, max: 95, step: 1, value: state.ab, unit: "%",
      }, function (value) { state.ab = value; draw(); });
      K.slider(ui.controls, {
        label: "B побеждает C", min: 5, max: 95, step: 1, value: state.bc, unit: "%",
      }, function (value) { state.bc = value; draw(); });
      K.slider(ui.controls, {
        label: "C побеждает D", min: 5, max: 95, step: 1, value: state.cd, unit: "%",
      }, function (value) { state.cd = value; draw(); });
      K.slider(ui.controls, {
        label: "Разногласие групп", min: 0, max: 100, step: 1, value: state.disagreement, unit: "%",
      }, function (value) { state.disagreement = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildAlignment(root) {
      var ui = setup(
        root,
        "Серый столбик — reference, цветной — обновлённая policy. Красная точка показывает истинную полезность, золотая — proxy reward.",
        "PPO, DPO и GRPO на четырёх ответах",
        "Слабый KL позволяет policy переоптимизировать ошибку reward; высокая proxy-оценка тогда расходится с полезностью.",
        500,
      );
      var state = { method: "dpo", reward: 62, kl: 55, error: 24 };
      var base = [0.46, 0.29, 0.18, 0.07];
      var truth = [0.66, 0.86, 0.62, 0.22];
      var names = ["краткий", "проверяемый", "уверенный", "лазейка"];
      var colors = [C.blue, C.green, C.gold, C.red];

      function proxy() {
        return [0.68, 0.82, 0.65, 0.22 + state.error / 100 * 1.05];
      }

      function policy() {
        var rewards = proxy();
        var mean = rewards.reduce(function (a, b) { return a + b; }, 0) / rewards.length;
        var std = Math.sqrt(rewards.reduce(function (sum, value) {
          return sum + Math.pow(value - mean, 2);
        }, 0) / rewards.length) || 1;
        var strength = state.reward / 100 * 5 / (0.22 + state.kl / 100 * 2.2);
        if (state.method === "ppo") strength *= 0.82;
        if (state.method === "grpo") strength *= 1.05;
        var logits = base.map(function (probability, index) {
          var advantage = state.method === "grpo" ? (rewards[index] - mean) / std : rewards[index] - 0.5;
          if (state.method === "dpo" && index === 1) advantage += 0.12;
          return Math.log(probability) + strength * advantage;
        });
        var max = Math.max.apply(null, logits);
        var exp = logits.map(function (value) { return Math.exp(value - max); });
        var sum = exp.reduce(function (a, b) { return a + b; }, 0);
        return exp.map(function (value) { return value / sum; });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var updated = policy();
        var rewards = proxy();
        title(ctx, "Вероятности ответов", 55, 35);
        var chart = { x: 70, y: 75, w: 560, h: 315 };
        ctx.fillStyle = C.wash; ctx.fillRect(chart.x, chart.y, chart.w, chart.h);
        base.forEach(function (probability, index) {
          var groupX = chart.x + 50 + index * 130;
          var baseHeight = probability * chart.h * 0.88;
          var newHeight = updated[index] * chart.h * 0.88;
          ctx.fillStyle = rgba(C.muted, 0.35);
          ctx.fillRect(groupX, chart.y + chart.h - baseHeight, 34, baseHeight);
          ctx.fillStyle = colors[index];
          ctx.fillRect(groupX + 40, chart.y + chart.h - newHeight, 34, newHeight);
          small(ctx, names[index], groupX + 37, chart.y + chart.h + 23, C.muted, "center");
        });
        small(ctx, "reference", chart.x + 16, chart.y + 22, C.muted);
        small(ctx, "policy", chart.x + 92, chart.y + 22, C.blue);

        title(ctx, "Истина и proxy", 690, 35);
        names.forEach(function (name, index) {
          var y = 85 + index * 75;
          small(ctx, name, 690, y, C.ink);
          ctx.fillStyle = C.wash; ctx.fillRect(690, y + 14, 170, 20);
          ctx.fillStyle = C.red; ctx.fillRect(690, y + 14, 170 * truth[index], 8);
          ctx.fillStyle = C.gold; ctx.fillRect(690, y + 26, 170 * clamp(rewards[index], 0, 1), 8);
        });
        var expectedTruth = updated.reduce(function (sum, probability, index) {
          return sum + probability * truth[index];
        }, 0);
        var expectedProxy = updated.reduce(function (sum, probability, index) {
          return sum + probability * rewards[index];
        }, 0);
        var kl = updated.reduce(function (sum, probability, index) {
          return sum + probability * Math.log(probability / base[index]);
        }, 0);
        ui.output.set([
          { label: "метод", value: state.method.toUpperCase() },
          { label: "истинная польза", value: expectedTruth.toFixed(3), color: C.red },
          { label: "proxy reward", value: expectedProxy.toFixed(3), color: C.gold },
          { label: "KL к reference", value: kl.toFixed(3) },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Метод",
        value: state.method,
        options: [
          { label: "PPO", value: "ppo" },
          { label: "DPO", value: "dpo" },
          { label: "GRPO", value: "grpo" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Вес reward", min: 0, max: 100, step: 1, value: state.reward, unit: "%",
      }, function (value) { state.reward = value; draw(); });
      K.slider(ui.controls, {
        label: "Связь с reference", min: 0, max: 100, step: 1, value: state.kl, unit: "%",
      }, function (value) { state.kl = value; draw(); });
      K.slider(ui.controls, {
        label: "Ошибка reward на лазейке", min: 0, max: 100, step: 1, value: state.error, unit: "%",
      }, function (value) { state.error = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function gaussian(x, mean, sigma) {
      return Math.exp(-0.5 * Math.pow((x - mean) / sigma, 2))
        / (sigma * Math.sqrt(2 * Math.PI));
    }

    function buildGan(root) {
      var ui = setup(
        root,
        "Синяя плотность — настоящая, золотая — generator, красная — решение critic. Сильный дисбаланс шагов вызывает collapse.",
        "Игра generator и critic",
        "Generator должен покрыть обе моды, а critic — сохранять полезный градиент; один низкий loss не проверяет разнообразие.",
        500,
      );
      var state = { generator: 48, critic: 56, steps: 80, diversity: 65 };
      var plot = { x: 60, y: 65, w: 800, h: 310 };

      function parameters() {
        var progress = 1 - Math.exp(-state.steps / 55);
        var imbalance = (state.generator - state.critic) / 100;
        var collapse = clamp(Math.abs(imbalance) * 1.8 + (50 - state.diversity) / 100, 0, 1);
        var first = -2 * progress + 0.5 * (1 - progress);
        var second = 2 * progress - 0.5 * (1 - progress);
        if (collapse > 0.35) {
          second = second * (1 - collapse) + first * collapse;
        }
        return { first: first, second: second, collapse: collapse };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var p = parameters();
        title(ctx, "Два распределения и critic", plot.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var realPoints = [];
        var generatedPoints = [];
        var criticPoints = [];
        for (var i = 0; i <= 300; i += 1) {
          var x = -5 + i / 300 * 10;
          var real = 0.5 * gaussian(x, -2, 0.65) + 0.5 * gaussian(x, 2, 0.72);
          var generated = 0.5 * gaussian(x, p.first, 0.82) + 0.5 * gaussian(x, p.second, 0.82);
          var critic = real / Math.max(1e-6, real + generated);
          var sx = plot.x + i / 300 * plot.w;
          realPoints.push({ x: sx, y: plot.y + plot.h - real / 0.34 * plot.h * 0.7 });
          generatedPoints.push({ x: sx, y: plot.y + plot.h - generated / 0.34 * plot.h * 0.7 });
          criticPoints.push({ x: sx, y: plot.y + plot.h - critic * plot.h * 0.92 });
        }
        line(ctx, realPoints, C.blue, 2.7);
        line(ctx, generatedPoints, C.gold, 2.7);
        line(ctx, criticPoints, C.red, 1.8, [5, 4]);
        small(ctx, "real", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "generator", plot.x + 60, plot.y + 22, C.gold);
        small(ctx, "critic", plot.x + 145, plot.y + 22, C.red);
        [-5, -2, 0, 2, 5].forEach(function (value) {
          small(ctx, String(value), plot.x + (value + 5) / 10 * plot.w, plot.y + plot.h + 20, C.muted, "center");
        });
        title(ctx, "Покрытие мод", 60, 430);
        ctx.fillStyle = C.wash; ctx.fillRect(180, 413, 680, 25);
        ctx.fillStyle = p.collapse > 0.45 ? C.red : C.green;
        ctx.fillRect(180, 413, 680 * (1 - p.collapse), 25);
        ui.output.set([
          { label: "центр G₁", value: p.first.toFixed(2), color: C.gold },
          { label: "центр G₂", value: p.second.toFixed(2), color: C.gold },
          { label: "потеря покрытия", value: (p.collapse * 100).toFixed(1) + "%", color: p.collapse > 0.45 ? C.red : C.green },
          { label: "баланс шагов", value: state.generator + " : " + state.critic },
        ]);
      }

      K.slider(ui.controls, {
        label: "Сила generator", min: 5, max: 100, step: 1, value: state.generator, unit: "%",
      }, function (value) { state.generator = value; draw(); });
      K.slider(ui.controls, {
        label: "Сила critic", min: 5, max: 100, step: 1, value: state.critic, unit: "%",
      }, function (value) { state.critic = value; draw(); });
      K.slider(ui.controls, {
        label: "Шагов игры", min: 1, max: 250, step: 1, value: state.steps,
      }, function (value) { state.steps = value; draw(); });
      K.slider(ui.controls, {
        label: "Регуляризация разнообразия", min: 0, max: 100, step: 1, value: state.diversity, unit: "%",
      }, function (value) { state.diversity = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function drawFace(ctx, centerX, centerY, coarse, fine, attack, label) {
      var width = 145 + (coarse - 50) * 0.9;
      var height = 190 - (coarse - 50) * 0.35;
      ctx.save();
      ctx.fillStyle = "#e5c7aa";
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, width / 2, height / 2, 0, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      var eyeY = centerY - 28 + (coarse - 50) * 0.12;
      var eyeGap = 34 + (coarse - 50) * 0.22;
      [-1, 1].forEach(function (side) {
        ctx.beginPath(); ctx.arc(centerX + side * eyeGap, eyeY, 8, 0, Math.PI * 2);
        ctx.fillStyle = fine > 50 ? "#3f6d6a" : "#5d4939"; ctx.fill();
      });
      ctx.beginPath();
      ctx.moveTo(centerX - 26, centerY + 42);
      ctx.quadraticCurveTo(centerX, centerY + 58 + (fine - 50) * 0.2, centerX + 28, centerY + 38);
      ctx.strokeStyle = "#8d4a4a"; ctx.lineWidth = 3; ctx.stroke();
      for (var dot = 0; dot < Math.round(fine / 8); dot += 1) {
        var angle = seeded(dot, 71) * Math.PI * 2;
        var radius = 25 + seeded(dot, 72) * 52;
        ctx.beginPath();
        ctx.arc(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius * 0.8, 1.4, 0, Math.PI * 2);
        ctx.fillStyle = rgba(C.red, 0.45); ctx.fill();
      }
      if (attack > 0) {
        ctx.globalAlpha = attack / 100 * 0.32;
        for (var stripe = -70; stripe <= 70; stripe += 10) {
          ctx.strokeStyle = stripe % 20 === 0 ? C.blue : C.red;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(centerX - width / 2, centerY + stripe);
          ctx.lineTo(centerX + width / 2, centerY + stripe + 6);
          ctx.stroke();
        }
      }
      ctx.restore();
      small(ctx, label, centerX, centerY + height / 2 + 28, C.muted, "center");
    }

    function buildStyleAttack(root) {
      var ui = setup(
        root,
        "Крупный стиль меняет геометрию, мелкий — текстуру. Полосы показывают увеличенное adversarial-возмущение.",
        "Style mixing и атака",
        "Плавный latent-контроль и чувствительность classifier происходят в одном высокоразмерном пространстве, но решают разные задачи.",
        500,
      );
      var state = { coarse: 62, fine: 38, attack: 8, split: 3 };

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        title(ctx, "Два масштаба генератора", 55, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(55, 60, 570, 355);
        drawFace(ctx, 210, 230, 35, 25, 0, "код A");
        var coarseShare = clamp((5 - state.split) / 4, 0, 1);
        var fineShare = clamp((9 - state.split) / 4, 0, 1);
        var mixedCoarse = 35 * (1 - coarseShare) + state.coarse * coarseShare;
        var mixedFine = 25 * (1 - fineShare) + state.fine * fineShare;
        drawFace(ctx, 475, 230, mixedCoarse, mixedFine, state.attack, "смешанный код");

        title(ctx, "Уровни стиля", 680, 35);
        for (var layer = 1; layer <= 8; layer += 1) {
          var y = 72 + (layer - 1) * 40;
          var fromB = layer >= state.split;
          ctx.fillStyle = fromB ? rgba(C.red, 0.22) : rgba(C.blue, 0.2);
          ctx.fillRect(680, y, 175, 27);
          small(ctx, "слой " + layer, 692, y + 18, C.ink);
          small(ctx, fromB ? "B" : "A", 838, y + 18, fromB ? C.red : C.blue, "right");
        }
        var classifier = clamp(0.88 - state.attack / 100 * 1.15 + Math.abs(mixedCoarse - 50) / 200, 0.01, 0.99);
        title(ctx, "Classifier: «лицо»", 680, 425);
        ctx.fillStyle = C.wash; ctx.fillRect(680, 442, 175, 22);
        ctx.fillStyle = classifier > 0.5 ? C.green : C.red;
        ctx.fillRect(680, 442, 175 * classifier, 22);
        ui.output.set([
          { label: "крупный стиль", value: String(mixedCoarse), color: C.blue },
          { label: "мелкий стиль", value: String(mixedFine), color: C.red },
          { label: "ε атаки", value: state.attack + " / 100" },
          { label: "P(лицо)", value: classifier.toFixed(3), color: classifier > 0.5 ? C.green : C.red },
        ]);
      }

      K.slider(ui.controls, {
        label: "Крупный стиль B", min: 0, max: 100, step: 1, value: state.coarse,
      }, function (value) { state.coarse = value; draw(); });
      K.slider(ui.controls, {
        label: "Мелкий стиль B", min: 0, max: 100, step: 1, value: state.fine,
      }, function (value) { state.fine = value; draw(); });
      K.slider(ui.controls, {
        label: "Adversarial ε", min: 0, max: 30, step: 1, value: state.attack,
      }, function (value) { state.attack = value; draw(); });
      K.slider(ui.controls, {
        label: "Граница style mixing", min: 1, max: 8, step: 1, value: state.split,
      }, function (value) { state.split = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildMultimodal(root) {
      var ui = setup(
        root,
        "Круг — изображение, квадрат — его текст. Справа similarity matrix: правильные пары лежат на диагонали.",
        "Общий embedding двух модальностей",
        "Температура усиливает hard negatives; случайная пара может оказаться ложным negative, если подпись подходит нескольким снимкам.",
        500,
      );
      var state = { temperature: 0.12, negatives: 32, noise: 28 };
      var colors = [C.blue, C.red, C.green, C.gold, C.violet, C.muted];
      var centers = [
        { x: -1.6, y: -0.9 },
        { x: 1.5, y: -1.1 },
        { x: -1.2, y: 1.25 },
        { x: 1.35, y: 1.2 },
        { x: 0.1, y: -0.1 },
        { x: 0.45, y: 0.75 },
      ];

      function points() {
        var alignment = clamp(Math.log2(state.negatives + 1) / 8 / Math.sqrt(state.temperature), 0.15, 1);
        return centers.map(function (center, index) {
          var offsetX = (seeded(index, 81) - 0.5) * state.noise / 20 * (1 - alignment * 0.72);
          var offsetY = (seeded(index, 82) - 0.5) * state.noise / 20 * (1 - alignment * 0.72);
          return {
            image: { x: center.x - offsetX, y: center.y - offsetY },
            text: { x: center.x + offsetX, y: center.y + offsetY },
          };
        });
      }

      function similarity(a, b) {
        var distance = Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2);
        return -distance / state.temperature;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var pairs = points();
        var scatter = { x: 55, y: 65, w: 490, h: 340 };
        title(ctx, "Latent space", scatter.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(scatter.x, scatter.y, scatter.w, scatter.h);
        function sx(value) { return scatter.x + (value + 2.5) / 5 * scatter.w; }
        function sy(value) { return scatter.y + scatter.h - (value + 2.2) / 4.4 * scatter.h; }
        pairs.forEach(function (pair, index) {
          line(ctx, [
            { x: sx(pair.image.x), y: sy(pair.image.y) },
            { x: sx(pair.text.x), y: sy(pair.text.y) },
          ], rgba(colors[index], 0.5), 1.5, [4, 3]);
          ctx.beginPath(); ctx.arc(sx(pair.image.x), sy(pair.image.y), 8, 0, Math.PI * 2);
          ctx.fillStyle = colors[index]; ctx.fill();
          ctx.fillStyle = colors[index];
          ctx.fillRect(sx(pair.text.x) - 7, sy(pair.text.y) - 7, 14, 14);
        });
        small(ctx, "● изображение   ■ текст", scatter.x + 14, scatter.y + 22, C.muted);

        title(ctx, "Similarity", 630, 35);
        var cell = 36;
        var matrixX = 635;
        var matrixY = 90;
        var correct = 0;
        pairs.forEach(function (imagePair, row) {
          var logits = pairs.map(function (textPair) { return similarity(imagePair.image, textPair.text); });
          var leader = logits.indexOf(Math.max.apply(null, logits));
          if (leader === row) correct += 1;
          var max = Math.max.apply(null, logits);
          var exp = logits.map(function (value) { return Math.exp(value - max); });
          var sum = exp.reduce(function (a, b) { return a + b; }, 0);
          exp.forEach(function (value, column) {
            var probability = value / sum;
            ctx.fillStyle = rgba(row === column ? C.green : C.red, 0.04 + probability * 0.9);
            ctx.fillRect(matrixX + column * cell, matrixY + row * cell, cell - 2, cell - 2);
          });
        });
        for (var index = 0; index < pairs.length; index += 1) {
          small(ctx, String(index + 1), matrixX + index * cell + cell / 2, matrixY - 10, C.muted, "center");
          small(ctx, String(index + 1), matrixX - 12, matrixY + index * cell + 23, C.muted, "right");
        }
        ui.output.set([
          { label: "Recall@1", value: correct + " / " + pairs.length, color: C.green },
          { label: "температура", value: state.temperature.toFixed(2) },
          { label: "negatives", value: String(state.negatives) },
          { label: "шум пар", value: state.noise + "%", color: C.red },
        ]);
      }

      K.slider(ui.controls, {
        label: "Температура", min: 0.03, max: 1, step: 0.01, value: state.temperature,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.temperature = value; draw(); });
      K.slider(ui.controls, {
        label: "Negatives", min: 2, max: 128, step: 2, value: state.negatives,
      }, function (value) { state.negatives = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум соответствий", min: 0, max: 100, step: 1, value: state.noise, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildVae(root) {
      var ui = setup(
        root,
        "Точки — posterior-сэмплы трёх классов. Большой β тянет их к prior; справа видна интерполяция между двумя кодами.",
        "Latent space VAE",
        "Реконструкция любит раздельные острова, KL сжимает их к общему prior. Слишком сильный KL стирает полезный код.",
        500,
      );
      var state = { beta: 1, latent: 2, path: 0.5 };
      var colors = [C.blue, C.red, C.green];
      var baseCenters = [
        { x: -1.55, y: -0.7 },
        { x: 1.45, y: -0.55 },
        { x: 0.05, y: 1.4 },
      ];

      function latentPoint(classIndex, index) {
        var shrink = 1 / (1 + state.beta * 0.17);
        var spread = 0.18 + state.beta * 0.055 + 0.12 / Math.sqrt(state.latent);
        return {
          x: baseCenters[classIndex].x * shrink + (seeded(index + classIndex * 401, 91) - 0.5) * spread * 2,
          y: baseCenters[classIndex].y * shrink + (seeded(index + classIndex * 401, 92) - 0.5) * spread * 2,
        };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var scatter = { x: 55, y: 60, w: 500, h: 350 };
        title(ctx, "Posterior в двух координатах", scatter.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(scatter.x, scatter.y, scatter.w, scatter.h);
        function sx(value) { return scatter.x + (value + 2.5) / 5 * scatter.w; }
        function sy(value) { return scatter.y + scatter.h - (value + 2.25) / 4.5 * scatter.h; }
        ctx.strokeStyle = C.grid;
        ctx.beginPath(); ctx.arc(sx(0), sy(0), 95, 0, Math.PI * 2); ctx.stroke();
        for (var classIndex = 0; classIndex < 3; classIndex += 1) {
          for (var i = 0; i < 60; i += 1) {
            var point = latentPoint(classIndex, i);
            ctx.beginPath(); ctx.arc(sx(point.x), sy(point.y), 3.2, 0, Math.PI * 2);
            ctx.fillStyle = rgba(colors[classIndex], 0.55); ctx.fill();
          }
        }
        var first = latentPoint(0, 7);
        var second = latentPoint(1, 13);
        line(ctx, [{ x: sx(first.x), y: sy(first.y) }, { x: sx(second.x), y: sy(second.y) }], C.gold, 2.4);
        var pathPoint = {
          x: first.x * (1 - state.path) + second.x * state.path,
          y: first.y * (1 - state.path) + second.y * state.path,
        };
        ctx.beginPath(); ctx.arc(sx(pathPoint.x), sy(pathPoint.y), 7, 0, Math.PI * 2);
        ctx.fillStyle = C.gold; ctx.fill();
        small(ctx, "класс A", scatter.x + 14, scatter.y + 22, C.blue);
        small(ctx, "класс B", scatter.x + 84, scatter.y + 22, C.red);
        small(ctx, "класс C", scatter.x + 154, scatter.y + 22, C.green);

        title(ctx, "Декодированная интерполяция", 625, 35);
        for (var step = 0; step < 6; step += 1) {
          var t = step / 5;
          var y = 65 + step * 62;
          ctx.fillStyle = C.wash; ctx.fillRect(625, y, 220, 48);
          ctx.save();
          ctx.globalAlpha = 1 - t;
          ctx.fillStyle = C.blue;
          ctx.font = "600 40px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("3", 735, y + 39);
          ctx.restore();
          ctx.save();
          ctx.globalAlpha = t;
          ctx.fillStyle = C.red;
          ctx.font = "600 40px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("8", 735, y + 39);
          ctx.restore();
          small(ctx, t.toFixed(1), 830, y + 29, C.muted, "right");
        }
        var recon = clamp(0.98 - state.beta * 0.075 + Math.log2(state.latent + 1) * 0.025, 0.25, 0.99);
        var kl = clamp(1.35 / (0.25 + state.beta), 0.05, 2.5);
        ui.output.set([
          { label: "качество recon", value: (recon * 100).toFixed(1) + "%", color: C.blue },
          { label: "средний KL", value: kl.toFixed(3), color: C.red },
          { label: "latent dim", value: String(state.latent) },
          { label: "позиция пути", value: state.path.toFixed(2), color: C.gold },
        ]);
      }

      K.slider(ui.controls, {
        label: "Вес KL β", min: 0, max: 5, step: 0.1, value: state.beta,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.beta = value; draw(); });
      K.slider(ui.controls, {
        label: "Размер latent", min: 1, max: 16, step: 1, value: state.latent,
      }, function (value) { state.latent = value; draw(); });
      K.slider(ui.controls, {
        label: "Позиция интерполяции", min: 0, max: 1, step: 0.01, value: state.path,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.path = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function basePixel(x, y) {
      var face = Math.pow((x - 7.5) / 6.2, 2) + Math.pow((y - 7.5) / 6.2, 2) < 1;
      var eye = (Math.abs(x - 5) < 1 && Math.abs(y - 6) < 1)
        || (Math.abs(x - 10) < 1 && Math.abs(y - 6) < 1);
      var mouth = y >= 10 && y <= 12 && Math.abs(x - 7.5) < (y - 9) * 1.5;
      return face ? (eye || mouth ? -0.65 : 0.72) : -0.9;
    }

    function buildDiffusion(root) {
      var ui = setup(
        root,
        "Четыре панели: чистый объект, выбранный уровень шума, результат обратных шагов и остаточная ошибка.",
        "Прямой и обратный diffusion",
        "Noise schedule задаёт отношение сигнала к шуму; guidance усиливает условие, но может уменьшить разнообразие.",
        500,
      );
      var state = { time: 62, denoise: 28, guidance: 2.2 };
      var size = 16;

      function drawGrid(ctx, originX, originY, values, titleText, mode) {
        var cell = 12;
        title(ctx, titleText, originX, originY - 16);
        values.forEach(function (value, index) {
          var normalized = clamp((value + 1) / 2, 0, 1);
          if (mode === "error") {
            ctx.fillStyle = value >= 0
              ? rgba(C.red, 0.1 + Math.abs(value) * 0.8)
              : rgba(C.blue, 0.1 + Math.abs(value) * 0.8);
          } else {
            var shade = Math.round(normalized * 235 + 10);
            ctx.fillStyle = "rgb(" + shade + "," + shade + "," + shade + ")";
          }
          ctx.fillRect(originX + (index % size) * cell, originY + Math.floor(index / size) * cell, cell - 1, cell - 1);
        });
      }

      function values() {
        var alpha = Math.exp(-state.time / 22);
        var clean = [];
        var noisy = [];
        var restored = [];
        var error = [];
        var remaining = Math.exp(-state.denoise / 10) / (0.6 + state.guidance * 0.35);
        for (var y = 0; y < size; y += 1) {
          for (var x = 0; x < size; x += 1) {
            var base = basePixel(x, y);
            var noise = (seeded(y * size + x, 101) - 0.5) * 2;
            var corrupted = Math.sqrt(alpha) * base + Math.sqrt(1 - alpha) * noise;
            var recovered = base * (1 - remaining) + corrupted * remaining;
            clean.push(base);
            noisy.push(corrupted);
            restored.push(recovered);
            error.push(recovered - base);
          }
        }
        return { alpha: alpha, clean: clean, noisy: noisy, restored: restored, error: error };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var result = values();
        drawGrid(ctx, 45, 75, result.clean, "x₀ · сигнал");
        drawArrow(ctx, 245, 170, 280, 170, C.muted, 1.5);
        drawGrid(ctx, 290, 75, result.noisy, "xₜ · шум");
        drawArrow(ctx, 490, 170, 525, 170, C.muted, 1.5);
        drawGrid(ctx, 535, 75, result.restored, "обратные шаги");
        drawArrow(ctx, 735, 170, 770, 170, C.muted, 1.5);
        drawGrid(ctx, 780, 75, result.error, "ошибка", "error");
        var plot = { x: 45, y: 340, w: 840, h: 95 };
        title(ctx, "Signal-to-noise ratio", plot.x, 315);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var points = [];
        for (var t = 0; t <= 100; t += 1) {
          var alpha = Math.exp(-t / 22);
          var snr = alpha / Math.max(1e-5, 1 - alpha);
          points.push({
            x: plot.x + t / 100 * plot.w,
            y: plot.y + plot.h - clamp(Math.log10(snr + 1) / 3, 0, 1) * plot.h,
          });
        }
        line(ctx, points, C.blue, 2.4);
        var markerX = plot.x + state.time / 100 * plot.w;
        line(ctx, [{ x: markerX, y: plot.y }, { x: markerX, y: plot.y + plot.h }], C.gold, 2);
        var mse = result.error.reduce(function (sum, value) { return sum + value * value; }, 0) / result.error.length;
        ui.output.set([
          { label: "шаг t", value: String(state.time), color: C.gold },
          { label: "доля сигнала ᾱ", value: result.alpha.toFixed(3), color: C.blue },
          { label: "MSE после reverse", value: mse.toFixed(4), color: C.red },
          { label: "guidance", value: state.guidance.toFixed(1) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Уровень шума t", min: 0, max: 100, step: 1, value: state.time,
      }, function (value) { state.time = value; draw(); });
      K.slider(ui.controls, {
        label: "Обратных шагов", min: 0, max: 60, step: 1, value: state.denoise,
      }, function (value) { state.denoise = value; draw(); });
      K.slider(ui.controls, {
        label: "Guidance", min: 0, max: 6, step: 0.1, value: state.guidance,
        format: function (v) { return v.toFixed(1); },
      }, function (value) { state.guidance = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function erf(value) {
      var sign = value < 0 ? -1 : 1;
      var x = Math.abs(value);
      var a1 = 0.254829592;
      var a2 = -0.284496736;
      var a3 = 1.421413741;
      var a4 = -1.453152027;
      var a5 = 1.061405429;
      var p = 0.3275911;
      var t = 1 / (1 + p * x);
      var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
      return sign * y;
    }

    function normalCdf(value, mean, sigma) {
      return 0.5 * (1 + erf((value - mean) / (sigma * Math.sqrt(2))));
    }

    function buildDeepfake(root) {
      var ui = setup(
        root,
        "Синяя кривая — genuine, красная — fake. Всё справа от порога отправляется в тревогу.",
        "Deepfake-detector и base rate",
        "Новый generator сдвигает fake-распределение; prevalence определяет precision даже при тех же условных метриках.",
        500,
      );
      var state = { prevalence: 2, threshold: 0.55, shift: 25 };
      var plot = { x: 60, y: 60, w: 560, h: 300 };

      function metrics() {
        var fakeMean = 0.76 - state.shift / 100 * 0.38;
        var genuineMean = 0.27;
        var sigmaG = 0.13;
        var sigmaF = 0.15 + state.shift / 100 * 0.04;
        var fpr = 1 - normalCdf(state.threshold, genuineMean, sigmaG);
        var tpr = 1 - normalCdf(state.threshold, fakeMean, sigmaF);
        var total = 10000;
        var fake = total * state.prevalence / 100;
        var genuine = total - fake;
        var tp = Math.round(fake * tpr);
        var fn = Math.round(fake - tp);
        var fp = Math.round(genuine * fpr);
        var tn = Math.round(genuine - fp);
        return { fakeMean: fakeMean, sigmaG: sigmaG, sigmaF: sigmaF, fpr: fpr, tpr: tpr, tp: tp, fn: fn, fp: fp, tn: tn };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var m = metrics();
        title(ctx, "Score детектора", plot.x, 35);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var genuine = [];
        var fake = [];
        for (var i = 0; i <= 250; i += 1) {
          var score = i / 250;
          genuine.push({
            x: plot.x + score * plot.w,
            y: plot.y + plot.h - gaussian(score, 0.27, m.sigmaG) / 3.2 * plot.h * 0.9,
          });
          fake.push({
            x: plot.x + score * plot.w,
            y: plot.y + plot.h - gaussian(score, m.fakeMean, m.sigmaF) / 3.2 * plot.h * 0.9,
          });
        }
        line(ctx, genuine, C.blue, 2.7);
        line(ctx, fake, C.red, 2.7);
        var thresholdX = plot.x + state.threshold * plot.w;
        line(ctx, [{ x: thresholdX, y: plot.y }, { x: thresholdX, y: plot.y + plot.h }], C.gold, 2);
        small(ctx, "genuine", plot.x + 15, plot.y + 22, C.blue);
        small(ctx, "fake", plot.x + 85, plot.y + 22, C.red);
        small(ctx, "порог", thresholdX, plot.y + plot.h + 20, C.gold, "center");

        title(ctx, "10 000 файлов", 690, 35);
        var matrixX = 690;
        var matrixY = 80;
        var cells = [
          { label: "TP", value: m.tp, color: C.green },
          { label: "FN", value: m.fn, color: C.red },
          { label: "FP", value: m.fp, color: C.gold },
          { label: "TN", value: m.tn, color: C.blue },
        ];
        cells.forEach(function (cell, index) {
          var x = matrixX + (index % 2) * 92;
          var y = matrixY + Math.floor(index / 2) * 98;
          ctx.fillStyle = rgba(cell.color, 0.18);
          ctx.fillRect(x, y, 82, 82);
          small(ctx, cell.label, x + 10, y + 20, cell.color);
          ctx.fillStyle = C.ink;
          ctx.font = "600 20px ui-monospace, monospace";
          ctx.textAlign = "center";
          ctx.fillText(String(cell.value), x + 41, y + 54);
        });
        var precision = m.tp / Math.max(1, m.tp + m.fp);
        ui.output.set([
          { label: "sensitivity", value: (m.tpr * 100).toFixed(1) + "%", color: C.red },
          { label: "false positive", value: (m.fpr * 100).toFixed(2) + "%", color: C.gold },
          { label: "precision тревоги", value: (precision * 100).toFixed(1) + "%", color: C.green },
          { label: "ложных тревог", value: String(m.fp) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Доля fake", min: 0.1, max: 30, step: 0.1, value: state.prevalence,
        format: function (v) { return v.toFixed(1); }, unit: "%",
      }, function (value) { state.prevalence = value; draw(); });
      K.slider(ui.controls, {
        label: "Порог", min: 0.05, max: 0.95, step: 0.01, value: state.threshold,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.threshold = value; draw(); });
      K.slider(ui.controls, {
        label: "Сдвиг нового generator", min: 0, max: 100, step: 1, value: state.shift,
        unit: "%",
      }, function (value) { state.shift = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildVideoWorld(root) {
      var ui = setup(
        root,
        "Верхняя лента — сгенерированные кадры. Внизу синяя траектория соблюдает физику, красная показывает прогноз модели.",
        "Время и физическая согласованность",
        "Контекст уменьшает неопределённость скорости, а физический штраф не даёт кадрам быть красивыми по отдельности и невозможными вместе.",
        500,
      );
      var state = { frames: 8, physics: 55, noise: 28 };

      function trueY(t) {
        var phase = (t % 1);
        return 0.18 + 2.8 * phase * Math.pow(1 - phase, 1.25);
      }

      function predictedY(t, index) {
        var physical = trueY(t);
        var noisy = physical + (seeded(index, 111) - 0.5) * state.noise / 100 * 1.3;
        return noisy * (1 - state.physics / 100) + physical * state.physics / 100;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        title(ctx, "Кадры", 45, 35);
        var shown = 8;
        var frameWidth = 96;
        for (var i = 0; i < shown; i += 1) {
          var x = 45 + i * 108;
          var y = 60;
          ctx.fillStyle = C.wash; ctx.fillRect(x, y, frameWidth, 125);
          ctx.strokeStyle = C.grid; ctx.strokeRect(x, y, frameWidth, 125);
          var time = i / (shown - 1);
          var ballY = 165 - predictedY(time, i + state.frames) * 70;
          ctx.beginPath(); ctx.arc(x + frameWidth / 2, ballY, 9, 0, Math.PI * 2);
          ctx.fillStyle = C.gold; ctx.fill();
          line(ctx, [{ x: x + 8, y: 168 }, { x: x + frameWidth - 8, y: 168 }], C.axis, 1.5);
          small(ctx, "t" + i, x + 8, y + 18, C.muted);
        }
        var plot = { x: 45, y: 245, w: 825, h: 160 };
        title(ctx, "Траектория", plot.x, 220);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var truePoints = [];
        var predictedPoints = [];
        var squaredError = 0;
        for (var step = 0; step < 100; step += 1) {
          var t = step / 99;
          var truth = trueY(t);
          var prediction = predictedY(t, step + state.frames * 17);
          squaredError += Math.pow(truth - prediction, 2);
          truePoints.push({ x: plot.x + t * plot.w, y: plot.y + plot.h - truth / 1.2 * plot.h });
          predictedPoints.push({ x: plot.x + t * plot.w, y: plot.y + plot.h - prediction / 1.2 * plot.h });
        }
        line(ctx, truePoints, C.blue, 2.5);
        line(ctx, predictedPoints, C.red, 2.1);
        small(ctx, "физика", plot.x + 14, plot.y + 22, C.blue);
        small(ctx, "generator", plot.x + 70, plot.y + 22, C.red);
        var mse = squaredError / 100;
        var contextGain = 1 - Math.exp(-state.frames / 6);
        ui.output.set([
          { label: "temporal MSE", value: mse.toFixed(4), color: C.red },
          { label: "кадров контекста", value: String(state.frames) },
          { label: "оценка скорости", value: (contextGain * 100).toFixed(1) + "%", color: C.blue },
          { label: "связь с физикой", value: state.physics + "%", color: C.green },
        ]);
      }

      K.slider(ui.controls, {
        label: "Кадров контекста", min: 1, max: 32, step: 1, value: state.frames,
      }, function (value) { state.frames = value; draw(); });
      K.slider(ui.controls, {
        label: "Связь с физикой", min: 0, max: 100, step: 1, value: state.physics,
        unit: "%",
      }, function (value) { state.physics = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум движения", min: 0, max: 100, step: 1, value: state.noise,
        unit: "%",
      }, function (value) { state.noise = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildAgent(root) {
      var ui = setup(
        root,
        "Каждый узел может ошибиться. Verifier ловит часть ошибок, но сам добавляет стоимость и не видит то, чего нет в журнале.",
        "Агентный цикл и узкое место",
        "Scaffold полезен, когда роли обмениваются проверяемыми артефактами и имеют минимальные права.",
        500,
      );
      var state = { tools: 3, memory: 6, verifier: 72, budget: 9 };
      var nodes = [
        { label: "planner", x: 65, base: 0.91, color: C.blue },
        { label: "поиск", x: 220, base: 0.84, color: C.green },
        { label: "вычисление", x: 375, base: 0.9, color: C.gold },
        { label: "verifier", x: 530, base: 0.82, color: C.red },
        { label: "editor", x: 685, base: 0.94, color: C.violet },
      ];

      function reliabilities() {
        var toolFit = 0.72 + Math.min(state.tools, 5) * 0.045 - Math.max(0, state.tools - 5) * 0.025;
        var memoryFit = 0.78 + Math.min(state.memory, 10) * 0.015 - Math.max(0, state.memory - 12) * 0.01;
        return [
          nodes[0].base * memoryFit,
          nodes[1].base * toolFit,
          nodes[2].base * toolFit,
          0.55 + state.verifier / 100 * 0.43,
          nodes[4].base * memoryFit,
        ].map(function (value) { return clamp(value, 0.35, 0.99); });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 500);
        var reliability = reliabilities();
        title(ctx, "Цепочка артефактов", 55, 35);
        var y = 115;
        nodes.forEach(function (node, index) {
          ctx.fillStyle = rgba(node.color, 0.13);
          ctx.fillRect(node.x, y, 120, 74);
          ctx.strokeStyle = node.color; ctx.lineWidth = 1.8;
          ctx.strokeRect(node.x, y, 120, 74);
          ctx.fillStyle = C.ink;
          ctx.font = "600 13px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(node.label, node.x + 60, y + 28);
          small(ctx, (reliability[index] * 100).toFixed(1) + "%", node.x + 60, y + 53, node.color, "center");
          if (index < nodes.length - 1) {
            drawArrow(ctx, node.x + 120, y + 37, nodes[index + 1].x - 12, y + 37, C.muted, 1.4);
          }
        });
        ctx.save();
        ctx.strokeStyle = C.red;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(nodes[3].x + 60, y + 74);
        ctx.bezierCurveTo(nodes[3].x + 60, y + 145, nodes[1].x + 60, y + 145, nodes[1].x + 60, y + 76);
        ctx.stroke();
        ctx.restore();
        small(ctx, "возврат на исправление", 470, 280, C.red, "center");

        var plot = { x: 65, y: 330, w: 730, h: 70 };
        title(ctx, "Вероятность успеха всей цепи", plot.x, 315);
        ctx.fillStyle = C.wash; ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        var rawSuccess = reliability.reduce(function (a, b) { return a * b; }, 1);
        var caught = (1 - rawSuccess) * state.verifier / 100 * 0.45;
        var success = clamp(rawSuccess + caught, 0, 0.995);
        ctx.fillStyle = C.blue; ctx.fillRect(plot.x, plot.y, plot.w * success, plot.h);
        ctx.fillStyle = rgba(C.red, 0.5); ctx.fillRect(plot.x + plot.w * success, plot.y, plot.w * (1 - success), plot.h);
        var cost = Math.min(state.budget, 4 + state.tools * 0.7 + state.memory * 0.12 + state.verifier / 100 * 2);
        ui.output.set([
          { label: "успех цепи", value: (success * 100).toFixed(1) + "%", color: C.blue },
          { label: "поймано verifier", value: (caught * 100).toFixed(1) + " п.п.", color: C.red },
          { label: "условная стоимость", value: cost.toFixed(1) + " / " + state.budget },
          { label: "узкое место", value: nodes[reliability.indexOf(Math.min.apply(null, reliability))].label },
        ]);
      }

      K.slider(ui.controls, {
        label: "Инструментов", min: 0, max: 8, step: 1, value: state.tools,
      }, function (value) { state.tools = value; draw(); });
      K.slider(ui.controls, {
        label: "Шагов памяти", min: 0, max: 20, step: 1, value: state.memory,
      }, function (value) { state.memory = value; draw(); });
      K.slider(ui.controls, {
        label: "Сила verifier", min: 0, max: 100, step: 1, value: state.verifier,
        unit: "%",
      }, function (value) { state.verifier = value; draw(); });
      K.slider(ui.controls, {
        label: "Бюджет", min: 3, max: 16, step: 1, value: state.budget,
      }, function (value) { state.budget = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildOrchestra(root) {
      var ui = setup(
        root,
        "Слева — роли проекта и обязательные передачи, справа — шестигранник рубрики. Красный разрыв блокирует защиту.",
        "Оркестр финального проекта",
        "Надёжный проект связывает вопрос, источник, split, baseline, аудит и интерактив в одну воспроизводимую цепь.",
        510,
      );
      var state = { split: "on", baseline: "on", critic: "on", evidence: 72, budget: 70 };
      var roles = [
        { label: "curator", sub: "паспорт + split", color: C.blue },
        { label: "modeller", sub: "baseline + модель", color: C.green },
        { label: "critic", sub: "утечка + сдвиг", color: C.red },
        { label: "editor", sub: "интерактив + вывод", color: C.gold },
      ];

      function scores() {
        return [
          0.74 + state.evidence / 100 * 0.22,
          state.split === "on" ? 0.92 : 0.24,
          state.baseline === "on" ? 0.88 : 0.28,
          state.critic === "on" ? 0.9 : 0.22,
          state.evidence / 100,
          0.48 + state.budget / 100 * 0.47,
        ];
      }

      function drawRadar(ctx, centerX, centerY, radius, values) {
        var labels = ["вопрос", "источник", "модель", "надёжность", "артефакт", "изложение"];
        for (var ring = 1; ring <= 4; ring += 1) {
          var ringPoints = labels.map(function (_, index) {
            var angle = -Math.PI / 2 + index / labels.length * Math.PI * 2;
            return {
              x: centerX + Math.cos(angle) * radius * ring / 4,
              y: centerY + Math.sin(angle) * radius * ring / 4,
            };
          });
          line(ctx, ringPoints.concat([ringPoints[0]]), C.grid, 1);
        }
        var points = values.map(function (value, index) {
          var angle = -Math.PI / 2 + index / values.length * Math.PI * 2;
          return {
            x: centerX + Math.cos(angle) * radius * value,
            y: centerY + Math.sin(angle) * radius * value,
          };
        });
        ctx.fillStyle = rgba(C.blue, 0.17);
        ctx.beginPath();
        points.forEach(function (point, index) {
          if (index) ctx.lineTo(point.x, point.y);
          else ctx.moveTo(point.x, point.y);
        });
        ctx.closePath(); ctx.fill();
        line(ctx, points.concat([points[0]]), C.blue, 2.4);
        labels.forEach(function (label, index) {
          var angle = -Math.PI / 2 + index / labels.length * Math.PI * 2;
          small(
            ctx,
            label,
            centerX + Math.cos(angle) * (radius + 28),
            centerY + Math.sin(angle) * (radius + 20) + 4,
            C.muted,
            Math.cos(angle) > 0.3 ? "left" : Math.cos(angle) < -0.3 ? "right" : "center",
          );
        });
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paper(ctx, 510);
        title(ctx, "Передачи между ролями", 50, 35);
        var y = 90;
        roles.forEach(function (role, index) {
          var x = 50 + index * 170;
          var disabled = (index === 0 && state.split === "off")
            || (index === 1 && state.baseline === "off")
            || (index === 2 && state.critic === "off");
          ctx.fillStyle = disabled ? rgba(C.red, 0.16) : rgba(role.color, 0.13);
          ctx.fillRect(x, y, 135, 92);
          ctx.strokeStyle = disabled ? C.red : role.color;
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, 135, 92);
          ctx.fillStyle = C.ink;
          ctx.font = "600 14px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(role.label, x + 67.5, y + 32);
          small(ctx, disabled ? "пропущено" : role.sub, x + 67.5, y + 58, disabled ? C.red : C.muted, "center");
          if (index < roles.length - 1) {
            drawArrow(ctx, x + 135, y + 46, x + 158, y + 46, disabled ? C.red : C.muted, 1.6);
          }
        });

        var values = scores();
        title(ctx, "Рубрика защиты", 80, 245);
        drawRadar(ctx, 280, 365, 105, values);
        title(ctx, "Статус", 585, 245);
        var blockers = [];
        if (state.split === "off") blockers.push("нет независимого split");
        if (state.baseline === "off") blockers.push("нет baseline");
        if (state.critic === "off") blockers.push("нет красной команды");
        if (state.evidence < 45) blockers.push("слабый артефакт");
        ctx.fillStyle = blockers.length ? rgba(C.red, 0.12) : rgba(C.green, 0.12);
        ctx.fillRect(585, 275, 275, 120);
        if (blockers.length) {
          blockers.slice(0, 4).forEach(function (blocker, index) {
            small(ctx, "• " + blocker, 605, 307 + index * 25, C.red);
          });
        } else {
          ctx.fillStyle = C.green;
          ctx.font = "600 22px ET Book, Palatino, Georgia, serif";
          ctx.textAlign = "center";
          ctx.fillText("готово к защите", 722, 342);
        }
        var average = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
        ui.output.set([
          { label: "средняя рубрика", value: (average * 100).toFixed(1) + "%", color: C.blue },
          { label: "блокеров", value: String(blockers.length), color: blockers.length ? C.red : C.green },
          { label: "доказательства", value: state.evidence + "%" },
          { label: "бюджет", value: state.budget + "%" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Независимый split",
        value: state.split,
        options: [{ label: "Есть", value: "on" }, { label: "Нет", value: "off" }],
      }, function (value) { state.split = value; draw(); });
      K.segmented(ui.controls, {
        label: "Baseline",
        value: state.baseline,
        options: [{ label: "Есть", value: "on" }, { label: "Нет", value: "off" }],
      }, function (value) { state.baseline = value; draw(); });
      K.segmented(ui.controls, {
        label: "Красная команда",
        value: state.critic,
        options: [{ label: "Есть", value: "on" }, { label: "Нет", value: "off" }],
      }, function (value) { state.critic = value; draw(); });
      K.slider(ui.controls, {
        label: "Артефакты", min: 0, max: 100, step: 1, value: state.evidence, unit: "%",
      }, function (value) { state.evidence = value; draw(); });
      K.slider(ui.controls, {
        label: "Бюджет реализации", min: 10, max: 100, step: 1, value: state.budget, unit: "%",
      }, function (value) { state.budget = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "73": buildRnn,
      "74": buildLstm,
      "75": buildTokens,
      "76": buildAttention,
      "77": buildTransformerBlock,
      "78": buildPretraining,
      "79": buildScaling,
      "80": buildPreferences,
      "81": buildAlignment,
      "82": buildGan,
      "83": buildStyleAttack,
      "84": buildMultimodal,
      "85": buildVae,
      "86": buildDiffusion,
      "87": buildDeepfake,
      "88": buildVideoWorld,
      "89": buildAgent,
      "90": buildOrchestra,
    };

    K.register("g11-generative", function (root, options) {
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
