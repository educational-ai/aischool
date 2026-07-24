// Lesson 13: build a logic gate on one McCulloch-Pitts neuron; see the boundary line.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("neuron-logic-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 520;

      var INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
      var TARGETS = {
        and: { name: "И", f: function (a, b) { return a && b ? 1 : 0; } },
        or: { name: "ИЛИ", f: function (a, b) { return a || b ? 1 : 0; } },
        not: { name: "НЕ x₁", f: function (a) { return a ? 0 : 1; } },
        inhibit: { name: "запрет x₁∧¬x₂", f: function (a, b) { return a && !b ? 1 : 0; } },
        xor: { name: "XOR", f: function (a, b) { return a !== b ? 1 : 0; } },
      };

      var state = { w1: 1, w2: 1, b: 1.5, target: "and", twoLayer: "off" };

      K.hint(
        root,
        "Крутите веса и порог, пока таблица истинности нейрона не совпадёт с целевой. Синяя прямая — граница нейрона; для XOR её не хватит, и понадобится второй слой.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Нейрон срабатывает, когда w₁x₁+w₂x₂ ≥ порог. Слева — таблица истинности с подсветкой совпадений, справа — квадрат входов и граница. XOR не берётся одной прямой ни при каких весах; режим двух слоёв решает его тремя нейронами.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Логический нейрон: таблица истинности и граница на квадрате входов",
        onResize: draw,
        drag: false,
      });

      var tbl = { x: 60, y: 70, w: 380, h: 340 };
      var sq = { x: 560, y: 70, s: 340 };

      function neuronOut(a, b) {
        return state.w1 * a + state.w2 * b >= state.b ? 1 : 0;
      }

      function targetOut(a, b) {
        var t = TARGETS[state.target];
        return state.target === "not" ? t.f(a) : t.f(a, b);
      }

      function sx(x) { return sq.x + x * sq.s; }
      function sy(y) { return sq.y + sq.s - y * sq.s; }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "14px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        var matched = 0;
        var xorMode = state.target === "xor";

        // Truth table.
        ctx.fillStyle = C.ink;
        ctx.font = "bold 15px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Таблица истинности", tbl.x, tbl.y - 24);
        ctx.font = "14px PT Sans, sans-serif";
        var heads = ["x₁", "x₂", "нейрон", "цель"];
        var colX = [tbl.x + 30, tbl.x + 110, tbl.x + 210, tbl.x + 320];
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        heads.forEach(function (h, i) { ctx.fillText(h, colX[i], tbl.y + 8); });
        ctx.strokeStyle = C.line;
        ctx.beginPath();
        ctx.moveTo(tbl.x, tbl.y + 26);
        ctx.lineTo(tbl.x + tbl.w, tbl.y + 26);
        ctx.stroke();

        INPUTS.forEach(function (inp, r) {
          var a = inp[0], b = inp[1];
          var no = neuronOut(a, b);
          var to = targetOut(a, b);
          var ok = no === to;
          if (ok) matched += 1;
          var y = tbl.y + 58 + r * 56;
          if (!ok) {
            ctx.fillStyle = "rgba(185,74,59,0.10)";
            ctx.fillRect(tbl.x, y - 22, tbl.w, 44);
          }
          ctx.fillStyle = C.ink;
          ctx.fillText(String(a), colX[0], y);
          ctx.fillText(String(b), colX[1], y);
          ctx.fillStyle = no ? C.blue : C.muted;
          ctx.fillText(String(no), colX[2], y);
          ctx.fillStyle = to ? C.green : C.muted;
          ctx.fillText(String(to), colX[3], y);
          ctx.fillStyle = ok ? C.green : C.red;
          ctx.font = "16px PT Sans, sans-serif";
          ctx.fillText(ok ? "✓" : "✗", tbl.x + tbl.w - 8, y);
          ctx.font = "14px PT Sans, sans-serif";
        });

        // Input square with points and boundary line.
        ctx.fillStyle = C.ink;
        ctx.font = "bold 15px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Квадрат входов и граница", sq.x, tbl.y - 24);
        ctx.font = "14px PT Sans, sans-serif";
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(sq.x, sq.y, sq.s, sq.s);

        // Boundary w1*x + w2*y = b, drawn only if not two-layer.
        if (!(xorMode && state.twoLayer === "on")) {
          drawBoundary(ctx);
        } else {
          ctx.fillStyle = C.green;
          ctx.textAlign = "center";
          ctx.fillText("два слоя: три нейрона решают XOR", sq.x + sq.s / 2, sq.y + sq.s / 2);
        }

        INPUTS.forEach(function (inp) {
          var a = inp[0], b = inp[1];
          var to = targetOut(a, b);
          ctx.beginPath();
          ctx.arc(sx(a), sy(b), 12, 0, Math.PI * 2);
          ctx.fillStyle = to ? C.ink : "#ffffff";
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = C.ink;
          ctx.stroke();
          ctx.lineWidth = 1;
        });
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText("x₁", sq.x + sq.s / 2, sq.y + sq.s + 24);
        ctx.save();
        ctx.translate(sq.x - 24, sq.y + sq.s / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("x₂", 0, 0);
        ctx.restore();

        var rows = [
          { label: "Совпало строк", value: matched + " из 4", color: matched === 4 ? C.green : C.ink },
        ];
        if (matched === 4) {
          rows.push({ label: "Готово", value: "нейрон = " + TARGETS[state.target].name, color: C.green });
        } else if (xorMode && state.twoLayer === "off") {
          rows.push({ label: "XOR одной прямой", value: "недостижим — включите два слоя", color: C.red });
        } else {
          rows.push({ label: "Подсказка", value: hintFor(), color: C.muted });
        }
        output.set(rows);
      }

      function hintFor() {
        if (state.target === "and") return "И: обе единицы нужны — порог повыше";
        if (state.target === "or") return "ИЛИ: хватает одной — порог пониже";
        if (state.target === "not") return "НЕ: вес x₁ отрицательный, порог около 0";
        if (state.target === "inhibit") return "Запрет: w₂ отрицательный — вето второго входа";
        return "Двигайте прямую — четвёртую строку не поймать";
      }

      function drawBoundary(ctx) {
        // w1*x + w2*y = b within [−0.4, 1.4]^2 clipped to square.
        var w1 = state.w1, w2 = state.w2, b = state.b;
        var pts = [];
        var lo = -0.4, hi = 1.4;
        if (Math.abs(w2) > 1e-9) {
          [lo, hi].forEach(function (x) {
            var y = (b - w1 * x) / w2;
            if (y >= lo - 1e-9 && y <= hi + 1e-9) pts.push([x, y]);
          });
        }
        if (Math.abs(w1) > 1e-9) {
          [lo, hi].forEach(function (y) {
            var x = (b - w2 * y) / w1;
            if (x >= lo - 1e-9 && x <= hi + 1e-9) pts.push([x, y]);
          });
        }
        if (pts.length >= 2) {
          // shade activation side lightly
          ctx.strokeStyle = C.blue;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          ctx.moveTo(sx(clamp(pts[0][0])), sy(clamp(pts[0][1])));
          ctx.lineTo(sx(clamp(pts[1][0])), sy(clamp(pts[1][1])));
          ctx.stroke();
          ctx.lineWidth = 1;
        }
      }

      function clamp(v) { return Math.max(-0.4, Math.min(1.4, v)); }

      K.slider(controls, { label: "Вес w₁", min: -2, max: 2, step: 0.1, value: state.w1,
        format: function (v) { return v.toFixed(1).replace(".", ","); } },
        function (v) { state.w1 = v; draw(); });
      K.slider(controls, { label: "Вес w₂", min: -2, max: 2, step: 0.1, value: state.w2,
        format: function (v) { return v.toFixed(1).replace(".", ","); } },
        function (v) { state.w2 = v; draw(); });
      K.slider(controls, { label: "Порог b", min: -1, max: 3, step: 0.1, value: state.b,
        format: function (v) { return v.toFixed(1).replace(".", ","); } },
        function (v) { state.b = v; draw(); });
      K.segmented(controls, {
        label: "Целевая функция",
        value: state.target,
        options: [
          { value: "and", label: "И" },
          { value: "or", label: "ИЛИ" },
          { value: "not", label: "НЕ" },
          { value: "inhibit", label: "запрет" },
          { value: "xor", label: "XOR" },
        ],
      }, function (v) { state.target = v; draw(); });
      K.segmented(controls, {
        label: "Для XOR: два слоя",
        value: state.twoLayer,
        options: [
          { value: "off", label: "один нейрон" },
          { value: "on", label: "два слоя" },
        ],
      }, function (v) { state.twoLayer = v; draw(); });

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
