// Lesson 25: backprop graph — live forward values and backward gradients on a small graph.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("backprop-graph", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 470;

      var state = { x: 3, w: 2, b: -1, mode: "chain" };

      K.hint(
        root,
        "Двигайте входы. Вверху каждого узла — значение прямого прохода (серым), внизу — производная потери по этому узлу (красным), которую приносит обратный проход. Переключите на ветвление, чтобы увидеть, как в узле, участвующем в двух путях, вклады складываются.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Обратное распространение — это цепное правило на графе: начав с ∂L/∂L = 1, каждый узел умножает пришедшую сверху производную на свою локальную и передаёт дальше назад. Где число ветвится на два пути, приходящие производные складываются.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Граф вычислений с прямыми значениями и обратными градиентами",
        onResize: draw,
        drag: false,
      });

      function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
      }

      function node(ctx, cx, cy, top, bottom, kind) {
        var bw = 168, bh = 52;
        ctx.fillStyle = kind === "input" ? (C.wash || "#f6f5ef") : C.paper;
        ctx.strokeStyle = C.line || "#c9c8be"; ctx.lineWidth = 1.4;
        roundRect(ctx, cx - bw / 2, cy - bh / 2, bw, bh, 10);
        ctx.fill(); ctx.stroke();
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = C.ink; ctx.font = "15px PT Sans, sans-serif";
        ctx.fillText(top, cx, cy - 2);
        if (bottom) {
          ctx.fillStyle = C.red; ctx.font = "13px PT Sans, sans-serif";
          ctx.fillText(bottom, cx, cy + bh / 2 + 14);
        }
      }

      function edge(ctx, x0, y0, x1, y1, col) {
        ctx.strokeStyle = col; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }

      function fmt(v) { return (Math.round(v * 100) / 100).toString().replace(".", ","); }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        if (state.mode === "chain") drawChain(ctx); else drawBranch(ctx);
      }

      function drawChain(ctx) {
        var x = state.x, w = state.w, b = state.b;
        var u = w * x, v = u + b, L = v * v;
        var dLdv = 2 * v, dLdu = dLdv, dLdw = dLdu * x, dLdx = dLdu * w, dLdb = dLdv;

        var xN = [130, 110], wN = [130, 235], bN = [130, 360];
        var uN = [400, 175], vN = [650, 175], lN = [900, 175];
        edge(ctx, xN[0] + 84, xN[1], uN[0] - 84, uN[1], C.muted);
        edge(ctx, wN[0] + 84, wN[1], uN[0] - 84, uN[1], C.muted);
        edge(ctx, bN[0] + 84, bN[1], vN[0] - 84, vN[1], C.muted);
        edge(ctx, uN[0] + 84, uN[1], vN[0] - 84, vN[1], C.muted);
        edge(ctx, vN[0] + 84, vN[1], lN[0] - 84, lN[1], C.muted);
        // backward flow highlight
        ctx.strokeStyle = "rgba(185,74,59,0.5)"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(lN[0] - 84, lN[1] + 40); ctx.lineTo(uN[0] - 40, uN[1] + 40); ctx.stroke();
        ctx.fillStyle = C.red; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("градиент течёт назад", 470, 245);

        node(ctx, xN[0], xN[1], "x = " + fmt(x), "∂L/∂x = " + fmt(dLdx), "input");
        node(ctx, wN[0], wN[1], "w = " + fmt(w), "∂L/∂w = " + fmt(dLdw), "input");
        node(ctx, bN[0], bN[1], "b = " + fmt(b), "∂L/∂b = " + fmt(dLdb), "input");
        node(ctx, uN[0], uN[1], "u = w·x = " + fmt(u), "∂L/∂u = " + fmt(dLdu));
        node(ctx, vN[0], vN[1], "v = u+b = " + fmt(v), "∂L/∂v = " + fmt(dLdv));
        node(ctx, lN[0], lN[1], "L = v² = " + fmt(L), "∂L/∂L = 1");

        output.set([
          { label: "Потеря L", value: fmt(L), color: C.ink },
          { label: "∂L/∂w", value: fmt(dLdw) + "  (= 2·v·x)", color: C.red },
          { label: "∂L/∂b", value: fmt(dLdb) + "  (= 2·v)", color: C.red },
          { label: "∂L/∂x", value: fmt(dLdx) + "  (= 2·v·w)", color: C.red },
        ]);
      }

      function drawBranch(ctx) {
        var x = state.x;
        var a = x * x, L = a + x;
        var dLda = 1, path1 = dLda * 2 * x, path2 = 1, dLdx = path1 + path2;

        var xN = [130, 235], aN = [460, 120], xmid = [460, 350], lN = [860, 235];
        edge(ctx, xN[0] + 84, xN[1] - 30, aN[0] - 84, aN[1], C.muted);
        edge(ctx, xN[0] + 84, xN[1] + 30, xmid[0] - 84, xmid[1], C.muted);
        edge(ctx, aN[0] + 84, aN[1], lN[0] - 84, lN[1] - 30, C.muted);
        edge(ctx, xmid[0] + 84, xmid[1], lN[0] - 84, lN[1] + 30, C.muted);

        ctx.fillStyle = C.red; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("путь 1: ∂a/∂x = 2x = " + fmt(path1), 300, 150);
        ctx.fillStyle = C.blue;
        ctx.fillText("путь 2: 1", 300, 330);

        node(ctx, xN[0], xN[1], "x = " + fmt(x), "∂L/∂x = " + fmt(path1) + " + 1 = " + fmt(dLdx), "input");
        node(ctx, aN[0], aN[1], "a = x² = " + fmt(a), "∂L/∂a = 1");
        node(ctx, xmid[0], xmid[1], "x = " + fmt(x), "∂L/∂(x) = 1");
        node(ctx, lN[0], lN[1], "L = a+x = " + fmt(L), "∂L/∂L = 1");

        output.set([
          { label: "Потеря L", value: fmt(L) + "  (= x² + x)", color: C.ink },
          { label: "Путь через x²", value: "2x = " + fmt(path1), color: C.red },
          { label: "Прямой путь", value: "1", color: C.blue },
          { label: "∂L/∂x (сумма путей)", value: fmt(path1) + " + 1 = " + fmt(dLdx), color: C.green },
        ]);
      }

      var seg = K.segmented(controls, {
        label: "Граф",
        value: state.mode,
        options: [{ value: "chain", label: "цепочка  L = (w·x + b)²" }, { value: "branch", label: "ветвление  L = x² + x" }],
      }, function (v) { state.mode = v; syncControls(); draw(); });

      var sx = K.slider(controls, { label: "вход x", min: -3, max: 4, step: 0.5, value: state.x,
        format: fmt }, function (v) { state.x = v; draw(); });
      var sw = K.slider(controls, { label: "вес w", min: -3, max: 3, step: 0.5, value: state.w,
        format: fmt }, function (v) { state.w = v; draw(); });
      var sb = K.slider(controls, { label: "сдвиг b", min: -3, max: 3, step: 0.5, value: state.b,
        format: fmt }, function (v) { state.b = v; draw(); });

      function syncControls() {
        var branch = state.mode === "branch";
        [sw, sb].forEach(function (s) {
          if (!s || !s.input) return;
          var wrap = s.input.closest(".kontur-int-control");
          if (wrap) wrap.style.display = branch ? "none" : "";
        });
      }

      syncControls();
      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
