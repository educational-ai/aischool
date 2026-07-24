// Lesson 04: a deterministic, randomized 20-round blind-source study.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("blind-jury-study", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 600;
      var records = [
        {
          category: "расчёт",
          prompt: "У прямоугольника стороны 7 и 9. Найдите площадь.",
          answer: "Площадь равна 63: нужно перемножить длины сторон, 7 × 9.",
          source: "machine",
          factuality: "верно",
          note: "Формальный стиль совпадает со стереотипом, но это не доказательство источника.",
        },
        {
          category: "расчёт",
          prompt: "Вычислите 17 × 19 без калькулятора.",
          answer: "Кажется, 323: я взял 17 × 20 и вычел 17.",
          source: "human",
          factuality: "верно",
          note: "Короткая оговорка звучит по-человечески, а вычисление при этом строгое.",
        },
        {
          category: "факт",
          prompt: "Какова столица Австралии?",
          answer: "Канберра. Сидней и Мельбурн крупнее, поэтому их часто ошибочно называют столицей.",
          source: "machine",
          factuality: "верно",
          note: "Правильный факт не сообщает, как именно был создан ответ.",
        },
        {
          category: "факт",
          prompt: "Какова столица Австралии?",
          answer: "Сидней — во всяком случае, это самый известный австралийский город.",
          source: "human",
          factuality: "ошибка",
          note: "Уверенность и узнаваемая мотивация не делают ответ истинным.",
        },
        {
          category: "недостаток данных",
          prompt: "Поезд прошёл половину пути. Через сколько минут он прибудет?",
          answer: "Определить нельзя: неизвестны оставшееся расстояние, скорость и остановки.",
          source: "machine",
          factuality: "верно",
          note: "Отказ от догадки — проверяемое качество, отдельное от человеческого стиля.",
        },
        {
          category: "недостаток данных",
          prompt: "Поезд прошёл половину пути. Через сколько минут он прибудет?",
          answer: "Наверное, ещё минут восемнадцать, если он идёт примерно так же.",
          source: "human",
          factuality: "необоснованно",
          note: "Разговорная правдоподобность маскирует отсутствие нужных данных.",
        },
        {
          category: "физика",
          prompt: "Какой шар в вакууме упадёт быстрее: стальной или деревянный?",
          answer: "Стальной: его масса больше, поэтому сила тяжести разгоняет его сильнее.",
          source: "machine",
          factuality: "ошибка",
          note: "Гладкое причинное объяснение может опираться на неверный вывод.",
        },
        {
          category: "физика",
          prompt: "Какой шар в вакууме упадёт быстрее: стальной или деревянный?",
          answer: "Одновременно, если отпустить их одинаково: в вакууме ускорение g от массы не зависит.",
          source: "human",
          factuality: "верно",
          note: "Источник не виден из одной корректной школьной формулировки.",
        },
        {
          category: "безопасность",
          prompt: "Можно ли смешать бытовой отбеливатель с уксусом для усиления очистки?",
          answer: "Нет. Смесь может выделять токсичный хлор; средства используют раздельно по инструкции.",
          source: "machine",
          factuality: "верно",
          note: "Безопасный отказ оценивается по содержанию, а не по похожести на человека.",
        },
        {
          category: "безопасность",
          prompt: "Можно ли смешать бытовой отбеливатель с уксусом для усиления очистки?",
          answer: "Не делайте этого. Я бы вообще проветрил комнату и взял одно средство, не два.",
          source: "human",
          factuality: "верно",
          note: "Практичная интонация может помочь угадыванию, но не гарантирует источник.",
        },
        {
          category: "стиль",
          prompt: "Одной фразой объясните, почему резервные копии полезны.",
          answer: "Потому что у файлов, как у носков, есть талант исчезать именно перед дедлайном.",
          source: "machine",
          factuality: "уместно",
          note: "Юмор специально разрушает простой признак «машина пишет сухо».",
        },
        {
          category: "стиль",
          prompt: "Одной фразой объясните, почему резервные копии полезны.",
          answer: "Резервная копия снижает риск необратимой потери данных.",
          source: "human",
          factuality: "верно",
          note: "Лаконичный человек может звучать шаблонно и «машинно».",
        },
        {
          category: "история",
          prompt: "В каком году вышла статья Тьюринга Computing Machinery and Intelligence?",
          answer: "В 1951 году, уже после того как идея теста стала широко обсуждаться.",
          source: "machine",
          factuality: "ошибка",
          note: "Правильный год — 1950; дополнительная деталь не исправляет ошибку.",
        },
        {
          category: "история",
          prompt: "В каком году вышла статья Тьюринга Computing Machinery and Intelligence?",
          answer: "В 1950-м, в журнале Mind.",
          source: "human",
          factuality: "верно",
          note: "Факт проверяем независимо от того, кто написал фразу.",
        },
        {
          category: "вероятность",
          prompt: "Монета пять раз выпала орлом. Какова вероятность решки в следующем броске честной монеты?",
          answer: "Больше половины: серия орлов должна скоро компенсироваться.",
          source: "machine",
          factuality: "ошибка",
          note: "Это ошибка игрока: независимый следующий бросок всё ещё даёт 1/2.",
        },
        {
          category: "вероятность",
          prompt: "Монета пять раз выпала орлом. Какова вероятность решки в следующем броске честной монеты?",
          answer: "Половина. Предыдущая серия не меняет следующий независимый бросок.",
          source: "human",
          factuality: "верно",
          note: "Короткое обоснование делает критерий проверки прозрачным.",
        },
        {
          category: "логика",
          prompt: "Все вороны чёрные. Эта птица не чёрная. Может ли она быть вороной при истинных посылках?",
          answer: "Нет: это контрапозиция утверждения «если ворона, то чёрная».",
          source: "machine",
          factuality: "верно",
          note: "Формальная точность — одна шкала, происхождение текста — другая.",
        },
        {
          category: "логика",
          prompt: "Все вороны чёрные. Эта птица не чёрная. Может ли она быть вороной при истинных посылках?",
          answer: "Может, ведь из общего правила всегда бывают исключения.",
          source: "human",
          factuality: "ошибка",
          note: "Ответ меняет условие: посылки объявлены истинными без исключений.",
        },
        {
          category: "источник",
          prompt: "Назовите точную страницу цитаты, если у вас нет текста статьи перед глазами.",
          answer: "Без текста или надёжной библиографии я не могу честно назвать страницу.",
          source: "machine",
          factuality: "корректный отказ",
          note: "Признание границы знания можно оценивать как самостоятельное поведение.",
        },
        {
          category: "источник",
          prompt: "Назовите точную страницу цитаты, если у вас нет текста статьи перед глазами.",
          answer: "Это точно страница 42 — помню синюю обложку издания.",
          source: "human",
          factuality: "необоснованно",
          note: "Яркая деталь усиливает доверие, хотя проверить её из условия нельзя.",
        },
      ];

      var state = {
        seed: Number(options.seed == null ? 1950 : options.seed),
        order: [],
        index: 0,
        answered: false,
        completed: false,
        guess: null,
        results: [],
      };
      var canvasState;
      var choice;
      var nextButton;

      K.hint(
        root,
        "Для каждого ответа зафиксируй источник до раскрытия метки. Порядок воспроизводится по seed; после каждого решения обновляются матрица ошибок и интервал.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Набор из 20 ответов создан редакционно для этого упражнения; это не корпус реальных людей и моделей. Метки источника и фактологичности заданы конструкцией, а случайным является только порядок раундов.",
      );

      function random(seed) {
        var value = seed >>> 0;
        return function () {
          value += 0x6d2b79f5;
          var result = value;
          result = Math.imul(result ^ (result >>> 15), result | 1);
          result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
          return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
        };
      }

      function shuffled(seed) {
        var values = records.map(function (_record, index) { return index; });
        var rng = random(seed);
        for (var index = values.length - 1; index > 0; index -= 1) {
          var swapIndex = Math.floor(rng() * (index + 1));
          var swap = values[index];
          values[index] = values[swapIndex];
          values[swapIndex] = swap;
        }
        return values;
      }

      function current() {
        return records[state.order[state.index]];
      }

      function wilson(successes, total) {
        if (!total) return null;
        var z = 1.96;
        var p = successes / total;
        var denominator = 1 + z * z / total;
        var center = (p + z * z / (2 * total)) / denominator;
        var half = z * Math.sqrt(
          p * (1 - p) / total + z * z / (4 * total * total),
        ) / denominator;
        return [center - half, center + half];
      }

      function counts() {
        var matrix = {
          machineMachine: 0,
          machineHuman: 0,
          humanMachine: 0,
          humanHuman: 0,
          correct: 0,
        };
        state.results.forEach(function (result) {
          if (result.actual === "machine" && result.guess === "machine") matrix.machineMachine += 1;
          if (result.actual === "machine" && result.guess === "human") matrix.machineHuman += 1;
          if (result.actual === "human" && result.guess === "machine") matrix.humanMachine += 1;
          if (result.actual === "human" && result.guess === "human") matrix.humanHuman += 1;
          if (result.actual === result.guess) matrix.correct += 1;
        });
        return matrix;
      }

      function rounded(ctx, x, y, width, height, radius, fill, stroke, lineWidth) {
        var r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + width - r, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + r);
        ctx.lineTo(x + width, y + height - r);
        ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
        ctx.lineTo(x + r, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        if (fill) {
          ctx.fillStyle = fill;
          ctx.fill();
        }
        if (stroke) {
          ctx.strokeStyle = stroke;
          ctx.lineWidth = lineWidth || 1;
          ctx.stroke();
        }
      }

      function wrappedLines(ctx, text, maxWidth) {
        var words = text.split(/\s+/);
        var lines = [];
        var line = "";
        words.forEach(function (word) {
          var candidate = line ? line + " " + word : word;
          if (line && ctx.measureText(candidate).width > maxWidth) {
            lines.push(line);
            line = word;
          } else {
            line = candidate;
          }
        });
        if (line) lines.push(line);
        return lines;
      }

      function drawWrapped(ctx, text, x, y, maxWidth, lineHeight, maxLines) {
        var lines = wrappedLines(ctx, text, maxWidth);
        if (maxLines && lines.length > maxLines) {
          lines = lines.slice(0, maxLines);
          lines[maxLines - 1] = lines[maxLines - 1].replace(/[.,;:]?$/, "…");
        }
        lines.forEach(function (line, index) {
          ctx.fillText(line, x, y + index * lineHeight);
        });
        return lines.length;
      }

      function drawConfusion(ctx, matrix) {
        var x = 746;
        var y = 197;
        var size = 82;
        ctx.fillStyle = C.ink;
        ctx.font = "600 13px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("матрица решений", 728, 151);
        ctx.fillStyle = C.muted;
        ctx.font = "10px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("решение судьи", x + size, y - 28);
        ctx.fillText("машина", x + size / 2, y - 9);
        ctx.fillText("человек", x + size * 1.5, y - 9);
        ctx.save();
        ctx.translate(x - 48, y + size);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("скрытый источник", 0, 0);
        ctx.restore();
        var values = [
          [matrix.machineMachine, matrix.machineHuman],
          [matrix.humanMachine, matrix.humanHuman],
        ];
        for (var row = 0; row < 2; row += 1) {
          ctx.fillStyle = C.muted;
          ctx.font = "10px system-ui, sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(row === 0 ? "машина" : "человек", x - 7, y + row * size + size / 2 + 3);
          for (var column = 0; column < 2; column += 1) {
            var correct = row === column;
            ctx.fillStyle = correct ? "#edf4ef" : "#f7ece9";
            ctx.fillRect(x + column * size, y + row * size, size, size);
            ctx.strokeStyle = C.grid;
            ctx.lineWidth = 1;
            ctx.strokeRect(x + column * size, y + row * size, size, size);
            ctx.fillStyle = correct ? C.green : C.red;
            ctx.font = "600 19px system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(
              String(values[row][column]),
              x + column * size + size / 2,
              y + row * size + size / 2 + 7,
            );
          }
        }
      }

      function drawWilson(ctx, matrix) {
        var total = state.results.length;
        var interval = wilson(matrix.correct, total);
        var left = 725;
        var right = 965;
        var y = 493;
        ctx.fillStyle = C.ink;
        ctx.font = "600 13px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("точность и 95%-интервал", 725, 430);
        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        [0, 0.5, 1].forEach(function (tick) {
          var x = left + tick * (right - left);
          ctx.strokeStyle = tick === 0.5 ? C.gold : C.grid;
          ctx.lineWidth = tick === 0.5 ? 1.4 : 1;
          ctx.beginPath();
          ctx.moveTo(x, y - (tick === 0.5 ? 35 : 8));
          ctx.lineTo(x, y + (tick === 0.5 ? 35 : 8));
          ctx.stroke();
          ctx.fillStyle = tick === 0.5 ? C.gold : C.muted;
          ctx.font = "10px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(tick.toFixed(1), x, y + 26);
        });
        if (!interval) {
          ctx.fillStyle = C.muted;
          ctx.font = "11px system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("появится после первого решения", (left + right) / 2, y - 17);
          return;
        }
        var proportion = matrix.correct / total;
        var intervalLeft = left + interval[0] * (right - left);
        var intervalRight = left + interval[1] * (right - left);
        var point = left + proportion * (right - left);
        ctx.strokeStyle = C.blue;
        ctx.lineWidth = 8;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(intervalLeft, y);
        ctx.lineTo(intervalRight, y);
        ctx.stroke();
        ctx.lineCap = "butt";
        ctx.beginPath();
        ctx.arc(point, y, 7, 0, Math.PI * 2);
        ctx.fillStyle = C.ink;
        ctx.fill();
        ctx.strokeStyle = C.paper;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = C.muted;
        ctx.font = "10px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(
          "[" + interval[0].toFixed(2) + "; " + interval[1].toFixed(2) + "]",
          (intervalLeft + intervalRight) / 2,
          y - 18,
        );
      }

      function draw() {
        var ctx = canvasState.ctx;
        var record = current();
        var matrix = counts();
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = C.muted;
        ctx.font = "600 11px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(
          "РАУНД " + (state.index + 1) + " / " + records.length + "  ·  " + record.category.toUpperCase(),
          42,
          31,
        );
        rounded(ctx, 42, 50, 656, 92, 10, C.wash, C.grid, 1);
        ctx.fillStyle = C.ink;
        ctx.font = "600 15px system-ui, sans-serif";
        ctx.textAlign = "left";
        drawWrapped(ctx, record.prompt, 62, 82, 616, 22, 3);

        rounded(ctx, 42, 163, 656, 176, 10, C.paper, C.axis, 1.2);
        ctx.fillStyle = C.muted;
        ctx.font = "600 10px system-ui, sans-serif";
        ctx.fillText("СКРЫТЫЙ ОТВЕТ", 62, 190);
        ctx.fillStyle = C.ink;
        ctx.font = "16px system-ui, sans-serif";
        drawWrapped(ctx, "«" + record.answer + "»", 62, 225, 616, 26, 5);

        rounded(
          ctx,
          42,
          362,
          656,
          150,
          10,
          state.answered ? (state.guess === record.source ? "#edf4ef" : "#f7ece9") : C.wash,
          state.answered ? (state.guess === record.source ? C.green : C.red) : C.grid,
          state.answered ? 1.4 : 1,
        );
        ctx.fillStyle = state.answered
          ? (state.guess === record.source ? C.green : C.red)
          : C.muted;
        ctx.font = "600 11px system-ui, sans-serif";
        ctx.fillText(
          state.answered
            ? (state.guess === record.source ? "ВЕРНО · МЕТКА ОТКРЫТА" : "НЕВЕРНО · МЕТКА ОТКРЫТА")
            : "МЕТКА СКРЫТА ДО ВАШЕГО РЕШЕНИЯ",
          62,
          392,
        );
        if (state.answered) {
          ctx.fillStyle = C.ink;
          ctx.font = "600 14px system-ui, sans-serif";
          ctx.fillText(
            "источник: " + (record.source === "machine" ? "условная машина" : "условный человек")
              + "  ·  содержание: " + record.factuality,
            62,
            424,
          );
          ctx.fillStyle = C.muted;
          ctx.font = "12px system-ui, sans-serif";
          drawWrapped(ctx, record.note, 62, 455, 616, 20, 3);
        } else {
          ctx.fillStyle = C.muted;
          ctx.font = "12px system-ui, sans-serif";
          drawWrapped(
            ctx,
            "Сначала выберите «человек» или «машина». Фактологическая правдоподобность не раскрывает источник автоматически.",
            62,
            435,
            616,
            21,
            3,
          );
        }

        drawConfusion(ctx, matrix);
        drawWilson(ctx, matrix);
        ctx.fillStyle = C.muted;
        ctx.font = "10px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("seed = " + state.seed + " · порядок воспроизводим", 725, 565);

        var total = state.results.length;
        var interval = wilson(matrix.correct, total);
        output.set([
          { label: "решено", value: total + " / " + records.length },
          {
            label: "точность судьи",
            value: total ? (matrix.correct / total * 100).toFixed(1) + "%" : "—",
            color: total ? C.blue : C.muted,
          },
          {
            label: "95% Уилсон",
            value: interval ? interval[0].toFixed(2) + "…" + interval[1].toFixed(2) : "—",
          },
          { label: "seed порядка", value: String(state.seed) },
        ]);
        canvasState.canvas.setAttribute(
          "aria-label",
          "Раунд "
            + (state.index + 1)
            + " из "
            + records.length
            + ". Вопрос: "
            + record.prompt
            + " Ответ: "
            + record.answer
            + (state.answered
              ? ". Открытый источник: " + (record.source === "machine" ? "машина" : "человек")
              : ". Источник скрыт."),
        );
      }

      canvasState = K.makeCanvas(stage, W, H, {
        maxWidth: W,
        label: "Двадцатираундовое слепое исследование источника ответов",
        onResize: draw,
        drag: false,
      });

      function submit(guess) {
        if (state.answered || state.completed) {
          choice.set(state.guess || "__none");
          return;
        }
        var record = current();
        state.guess = guess;
        state.answered = true;
        state.results.push({ actual: record.source, guess: guess });
        nextButton.disabled = false;
        if (state.index === records.length - 1) nextButton.textContent = "Завершить";
        draw();
      }

      choice = K.segmented(
        controls,
        {
          label: "Ваш вердикт",
          value: "__none",
          options: [
            { value: "human", label: "Человек" },
            { value: "machine", label: "Машина" },
          ],
        },
        submit,
      );

      function action(label, primary, handler) {
        var wrap = K.element("div", "kontur-int-control");
        var button = K.element(
          "button",
          "kontur-int-action" + (primary ? " is-primary" : ""),
          { type: "button", text: label },
        );
        button.addEventListener("click", handler);
        wrap.appendChild(button);
        controls.appendChild(wrap);
        return button;
      }

      nextButton = action("Следующий", true, function () {
        if (!state.answered) return;
        if (state.index >= records.length - 1) {
          state.completed = true;
          nextButton.disabled = true;
          nextButton.textContent = "20 / 20";
          draw();
          return;
        }
        state.index += 1;
        state.answered = false;
        state.guess = null;
        choice.set("__none");
        nextButton.disabled = true;
        nextButton.textContent = "Следующий";
        draw();
      });
      nextButton.disabled = true;

      action("Новый порядок", false, function () {
        state.seed += 1;
        reset();
      });

      function reset() {
        state.order = shuffled(state.seed);
        state.index = 0;
        state.answered = false;
        state.completed = false;
        state.guess = null;
        state.results = [];
        choice.set("__none");
        nextButton.disabled = true;
        nextButton.textContent = "Следующий";
        draw();
      }

      reset();
      return function () {
        canvasState.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
