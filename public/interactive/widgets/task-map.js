// Lesson 01: a two-axis map separating learned rules from autonomy.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("task-map", function (root, options, K) {
      var C = K.COLORS;
      K.hint(
        root,
        "Выбери систему или перетащи точку. Ползунки меняют долю правила, найденную по данным, и самостоятельность решения.",
      );

      var stage = K.row(root);
      var W = 720;
      var H = 430;
      var canvasState;
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Горизонтальная ось описывает происхождение правила. Вертикальная ось описывает право системы действовать без подтверждения. Эти свойства не обязаны расти вместе.",
      );

      var examples = [
        { id: "calculator", label: "калькулятор", learned: 6, autonomy: 10, color: C.muted },
        { id: "barrier", label: "шлагбаум", learned: 8, autonomy: 84, color: C.gold },
        { id: "spam", label: "фильтр спама", learned: 76, autonomy: 68, color: C.red },
        { id: "diagnosis", label: "помощник врача", learned: 82, autonomy: 28, color: C.green },
        { id: "route", label: "маршрут", learned: 48, autonomy: 44, color: C.violet },
      ];
      var selected = examples[2];
      var state = {
        learned: Number(options.learned || selected.learned),
        autonomy: Number(options.autonomy || selected.autonomy),
      };

      function x(value) { return 82 + value / 100 * 578; }
      function y(value) { return 350 - value / 100 * 274; }

      function drawArrow(ctx, fromX, fromY, toX, toY) {
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - 6, toY + 3);
        ctx.lineTo(toX - 3, toY + 7);
        ctx.closePath();
        ctx.fill();
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = C.wash;
        ctx.fillRect(x(50), y(100), x(100) - x(50), y(0) - y(100));
        ctx.fillRect(x(0), y(50), x(100) - x(0), y(0) - y(50));

        ctx.strokeStyle = C.grid;
        ctx.lineWidth = 1;
        [0, 25, 50, 75, 100].forEach(function (value) {
          ctx.beginPath();
          ctx.moveTo(x(value), y(0));
          ctx.lineTo(x(value), y(100));
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x(0), y(value));
          ctx.lineTo(x(100), y(value));
          ctx.stroke();
        });

        ctx.strokeStyle = C.axis;
        ctx.fillStyle = C.axis;
        ctx.lineWidth = 1.2;
        drawArrow(ctx, x(0), y(0), x(100) + 14, y(0));
        drawArrow(ctx, x(0), y(0), x(0), y(100) - 14);

        ctx.fillStyle = C.muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("правило записано человеком", x(0), 392);
        ctx.textAlign = "right";
        ctx.fillText("правило найдено по данным", x(100), 392);
        ctx.save();
        ctx.translate(24, y(50));
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText("самостоятельность решения", 0, 0);
        ctx.restore();

        examples.forEach(function (item) {
          var isSelected = item.id === selected.id;
          var px = isSelected ? x(state.learned) : x(item.learned);
          var py = isSelected ? y(state.autonomy) : y(item.autonomy);
          ctx.beginPath();
          ctx.arc(px, py, isSelected ? 8 : 5, 0, Math.PI * 2);
          ctx.fillStyle = isSelected ? item.color : C.paper;
          ctx.fill();
          ctx.strokeStyle = item.color;
          ctx.lineWidth = isSelected ? 2 : 1.4;
          ctx.stroke();
          ctx.fillStyle = item.id === selected.id ? C.ink : C.muted;
          ctx.font = (isSelected ? "600 " : "") + "13px ET Book, Palatino, Georgia, serif";
          ctx.textAlign = px > x(75) ? "right" : "left";
          ctx.fillText(item.label, px + (px > x(75) ? -10 : 10), py - 9);
        });

        var zone = state.learned >= 50
          ? (state.autonomy >= 50 ? "обучаемая автономная система" : "обучаемый советник")
          : (state.autonomy >= 50 ? "автомат с фиксированным правилом" : "обычная программа");
        var control = state.autonomy >= 65
          ? "нужен контроль редких ошибок"
          : "решение можно подтвердить до действия";
        output.set([
          { label: "система", value: selected.label, color: selected.color },
          { label: "область", value: zone },
          { label: "проверка", value: control },
        ]);
      }

      canvasState = K.makeCanvas(stage, W, H, {
        maxWidth: W,
        label: "Карта систем по доле обучения и самостоятельности",
        onResize: draw,
      });

      var redraw = K.rafThrottle(draw);
      var learnedControl = K.slider(controls, {
        label: "Правило найдено по данным",
        min: 0,
        max: 100,
        step: 1,
        value: state.learned,
        unit: "%",
      }, function (value) {
        state.learned = value;
        redraw();
      });
      var autonomyControl = K.slider(controls, {
        label: "Самостоятельность решения",
        min: 0,
        max: 100,
        step: 1,
        value: state.autonomy,
        unit: "%",
      }, function (value) {
        state.autonomy = value;
        redraw();
      });
      var selector = K.segmented(controls, {
        label: "Пример",
        value: selected.id,
        options: examples.map(function (item) {
          return { value: item.id, label: item.label };
        }),
      }, function (value) {
        selected = examples.find(function (item) { return item.id === value; }) || selected;
        state.learned = selected.learned;
        state.autonomy = selected.autonomy;
        learnedControl.set(state.learned);
        autonomyControl.set(state.autonomy);
        redraw();
      });

      function nearest(point) {
        var best = null;
        var distance = 18;
        examples.forEach(function (item) {
          var px = item.id === selected.id ? x(state.learned) : x(item.learned);
          var py = item.id === selected.id ? y(state.autonomy) : y(item.autonomy);
          var current = Math.hypot(point.x - px, point.y - py);
          if (current < distance) {
            best = item;
            distance = current;
          }
        });
        return best;
      }

      var dragging = false;
      var removeDrag = K.drag(canvasState.canvas, { w: W, h: H }, {
        down: function (point) {
          var picked = nearest(point);
          if (picked) {
            selected = picked;
            selector.set(selected.id);
            state.learned = selected.learned;
            state.autonomy = selected.autonomy;
            learnedControl.set(state.learned);
            autonomyControl.set(state.autonomy);
          }
          dragging = Boolean(picked);
        },
        move: function (point) {
          if (!dragging) return;
          state.learned = Math.max(0, Math.min(100, (point.x - 82) / 578 * 100));
          state.autonomy = Math.max(0, Math.min(100, (350 - point.y) / 274 * 100));
          learnedControl.set(state.learned);
          autonomyControl.set(state.autonomy);
          redraw();
        },
        up: function () {
          dragging = false;
        },
        hover: function (point) {
          canvasState.canvas.style.cursor = nearest(point) ? "grab" : "default";
        },
      });

      draw();
      return function () {
        removeDrag();
        canvasState.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
