// Lesson 04: an executable binary-increment Turing machine with an exact trace.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("turing-machine-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 610;
      var examples = ["1011", "1111", "0100", "0"];
      var initial = examples.includes(String(options.input || ""))
        ? String(options.input)
        : "1011";
      var state = {
        initial: initial,
        tape: {},
        head: 0,
        machine: "q_scan",
        step: 0,
        history: [],
        running: false,
      };
      var timer = null;
      var canvasState;
      var selector;
      var runButton;

      var rules = [
        { state: "q_scan", read: "0", next: "q_scan", write: "0", move: "R" },
        { state: "q_scan", read: "1", next: "q_scan", write: "1", move: "R" },
        { state: "q_scan", read: "·", next: "q_carry", write: "·", move: "L" },
        { state: "q_carry", read: "1", next: "q_carry", write: "0", move: "L" },
        { state: "q_carry", read: "0", next: "q_halt", write: "1", move: "S" },
        { state: "q_carry", read: "·", next: "q_halt", write: "1", move: "S" },
      ];

      K.hint(
        root,
        "Нажимай «Шаг» и следи за золотой клеткой и строкой правила. «Пуск» выполняет те же переходы медленно: скрытых операций нет.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Алфавит {0, 1, ·}; знак · обозначает пустую клетку. q_scan ищет правый край, q_carry переносит единицу влево. Машина останавливается для каждой конечной двоичной строки.",
      );

      function read(position) {
        return Object.prototype.hasOwnProperty.call(state.tape, position)
          ? state.tape[position]
          : "·";
      }

      function currentRule() {
        if (state.machine === "q_halt") return null;
        var symbol = read(state.head);
        return rules.find(function (rule) {
          return rule.state === state.machine && rule.read === symbol;
        }) || null;
      }

      function tapeOutput() {
        var positions = Object.keys(state.tape)
          .map(Number)
          .filter(function (position) { return read(position) !== "·"; });
        if (!positions.length) return "·";
        var minimum = Math.min.apply(null, positions);
        var maximum = Math.max.apply(null, positions);
        var text = "";
        for (var position = minimum; position <= maximum; position += 1) {
          text += read(position);
        }
        return text;
      }

      function configuration() {
        var left = Math.min(-1, state.head - 1);
        var right = Math.max(state.initial.length, state.head + 1);
        var text = "";
        for (var position = left; position <= right; position += 1) {
          var symbol = read(position);
          text += position === state.head ? "[" + symbol + "]" : " " + symbol + " ";
        }
        return {
          step: state.step,
          machine: state.machine,
          head: state.head,
          text: text,
        };
      }

      function stopRunning() {
        state.running = false;
        if (timer) {
          window.clearInterval(timer);
          timer = null;
        }
        if (runButton) runButton.textContent = "Пуск";
      }

      function reset(nextInput) {
        stopRunning();
        state.initial = nextInput || state.initial;
        state.tape = {};
        state.initial.split("").forEach(function (symbol, index) {
          state.tape[index] = symbol;
        });
        state.head = 0;
        state.machine = "q_scan";
        state.step = 0;
        state.history = [configuration()];
        draw();
      }

      function executeStep() {
        if (state.machine === "q_halt") {
          stopRunning();
          draw();
          return false;
        }
        var rule = currentRule();
        if (!rule) {
          stopRunning();
          return false;
        }
        state.tape[state.head] = rule.write;
        state.machine = rule.next;
        if (rule.move === "R") state.head += 1;
        if (rule.move === "L") state.head -= 1;
        state.step += 1;
        state.history.push(configuration());
        if (state.history.length > 18) state.history.shift();
        if (state.machine === "q_halt") stopRunning();
        draw();
        return true;
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

      function drawTape(ctx) {
        var start = -2;
        var end = Math.max(8, state.initial.length + 2);
        var count = end - start + 1;
        var cellWidth = Math.min(79, 900 / count);
        var totalWidth = cellWidth * count;
        var left = (W - totalWidth) / 2;
        var top = 65;

        ctx.fillStyle = C.ink;
        ctx.font = "600 15px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("лента", left, 30);
        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("позиция головки: " + state.head, left + 65, 30);

        for (var position = start; position <= end; position += 1) {
          var x = left + (position - start) * cellWidth;
          var active = position === state.head;
          ctx.fillStyle = active ? "#f7f1df" : C.paper;
          ctx.strokeStyle = active ? C.gold : C.grid;
          ctx.lineWidth = active ? 2.6 : 1.1;
          ctx.fillRect(x, top, cellWidth, 70);
          ctx.strokeRect(x, top, cellWidth, 70);
          ctx.fillStyle = C.ink;
          ctx.font = (active ? "600 " : "") + "23px ui-monospace, monospace";
          ctx.textAlign = "center";
          ctx.fillText(read(position), x + cellWidth / 2, top + 45);
          ctx.fillStyle = C.muted;
          ctx.font = "10px system-ui, sans-serif";
          ctx.fillText(String(position), x + cellWidth / 2, top + 88);
        }

        var headX = left + (state.head - start) * cellWidth + cellWidth / 2;
        ctx.fillStyle = state.machine === "q_halt" ? C.green : C.gold;
        ctx.beginPath();
        ctx.moveTo(headX, top - 8);
        ctx.lineTo(headX - 8, top - 22);
        ctx.lineTo(headX + 8, top - 22);
        ctx.closePath();
        ctx.fill();
        rounded(
          ctx,
          headX - 48,
          top - 56,
          96,
          28,
          7,
          state.machine === "q_halt" ? "#edf4ef" : "#f7f1df",
          state.machine === "q_halt" ? C.green : C.gold,
          1.2,
        );
        ctx.fillStyle = state.machine === "q_halt" ? C.green : C.ink;
        ctx.font = "600 12px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(state.machine, headX, top - 37);

        var rule = currentRule();
        rounded(ctx, 145, 177, 750, 43, 8, rule ? "#f7f1df" : "#edf4ef", rule ? C.gold : C.green, 1.25);
        ctx.fillStyle = rule ? C.ink : C.green;
        ctx.font = "600 14px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(
          rule
            ? "(" + rule.state + ", " + rule.read + ") -> (" + rule.next + ", " + rule.write + ", " + rule.move + ")"
            : "q_halt · вычисление завершено",
          W / 2,
          204,
        );
      }

      function drawRules(ctx) {
        var x = 42;
        var y = 275;
        var width = 458;
        var rowHeight = 43;
        var active = currentRule();
        ctx.fillStyle = C.ink;
        ctx.font = "600 14px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("таблица переходов", x, y - 24);
        ctx.fillStyle = C.muted;
        ctx.font = "10px system-ui, sans-serif";
        ctx.fillText("состояние + символ", x, y - 6);
        ctx.fillText("новое состояние + запись + ход", x + 205, y - 6);

        rules.forEach(function (rule, index) {
          var top = y + index * rowHeight;
          var selected = active === rule;
          if (selected) {
            ctx.fillStyle = "#f7f1df";
            ctx.fillRect(x - 6, top, width + 12, rowHeight - 3);
          }
          ctx.strokeStyle = selected ? C.gold : C.grid;
          ctx.lineWidth = selected ? 1.4 : 1;
          ctx.beginPath();
          ctx.moveTo(x, top + rowHeight - 4);
          ctx.lineTo(x + width, top + rowHeight - 4);
          ctx.stroke();
          ctx.fillStyle = selected ? C.ink : C.muted;
          ctx.font = (selected ? "600 " : "") + "12px ui-monospace, monospace";
          ctx.textAlign = "left";
          ctx.fillText("(" + rule.state + ", " + rule.read + ")", x + 7, top + 25);
          ctx.fillText(
            "(" + rule.next + ", " + rule.write + ", " + rule.move + ")",
            x + 219,
            top + 25,
          );
          ctx.strokeStyle = selected ? C.gold : C.axis;
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          ctx.moveTo(x + 160, top + 20);
          ctx.lineTo(x + 198, top + 20);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x + 198, top + 20);
          ctx.lineTo(x + 191, top + 16);
          ctx.lineTo(x + 191, top + 24);
          ctx.closePath();
          ctx.fillStyle = selected ? C.gold : C.axis;
          ctx.fill();
        });
      }

      function drawTrace(ctx) {
        var x = 550;
        var y = 275;
        var width = 448;
        var rowHeight = 36;
        var rows = state.history.slice(-8);
        ctx.fillStyle = C.ink;
        ctx.font = "600 14px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("точная трасса конфигураций", x, y - 24);
        ctx.fillStyle = C.muted;
        ctx.font = "10px system-ui, sans-serif";
        ctx.fillText("квадратные скобки показывают клетку под головкой", x, y - 6);

        rows.forEach(function (item, index) {
          var top = y + index * rowHeight;
          var current = index === rows.length - 1;
          if (current) {
            ctx.fillStyle = state.machine === "q_halt" ? "#edf4ef" : C.wash;
            ctx.fillRect(x - 7, top, width + 7, rowHeight - 2);
          }
          ctx.fillStyle = current ? C.ink : C.muted;
          ctx.font = (current ? "600 " : "") + "11px ui-monospace, monospace";
          ctx.textAlign = "left";
          ctx.fillText(String(item.step).padStart(2, "0"), x, top + 23);
          ctx.fillText(item.machine.padEnd(8, " "), x + 38, top + 23);
          ctx.fillText(item.text, x + 132, top + 23);
          ctx.strokeStyle = C.grid;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, top + rowHeight - 2);
          ctx.lineTo(x + width, top + rowHeight - 2);
          ctx.stroke();
        });
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, W, H);
        drawTape(ctx);
        drawRules(ctx);
        drawTrace(ctx);
        output.set([
          { label: "шаг", value: String(state.step) },
          {
            label: "состояние",
            value: state.machine,
            color: state.machine === "q_halt" ? C.green : C.ink,
          },
          { label: "вход", value: state.initial + "₂" },
          {
            label: "лента без пустых клеток",
            value: tapeOutput() + "₂",
            color: state.machine === "q_halt" ? C.green : C.blue,
          },
        ]);
        canvasState.canvas.setAttribute(
          "aria-label",
          "Машина Тьюринга на шаге "
            + state.step
            + ", состояние "
            + state.machine
            + ", лента "
            + tapeOutput()
            + ", головка в позиции "
            + state.head,
        );
      }

      canvasState = K.makeCanvas(stage, W, H, {
        maxWidth: W,
        label: "Пошаговая машина Тьюринга для прибавления единицы",
        onResize: draw,
        drag: false,
      });

      selector = K.segmented(
        controls,
        {
          label: "Начальная лента",
          value: state.initial,
          options: examples.map(function (value) {
            return { value: value, label: value + "₂" };
          }),
        },
        function (value) {
          reset(value);
        },
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

      action("Шаг", true, function () {
        stopRunning();
        executeStep();
      });
      runButton = action("Пуск", false, function () {
        if (state.running) {
          stopRunning();
          return;
        }
        if (state.machine === "q_halt") reset(state.initial);
        state.running = true;
        runButton.textContent = "Пауза";
        timer = window.setInterval(function () {
          if (!executeStep()) stopRunning();
        }, 430);
      });
      action("Сброс", false, function () {
        reset(state.initial);
        selector.set(state.initial);
      });

      reset(state.initial);
      return function () {
        stopRunning();
        canvasState.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
