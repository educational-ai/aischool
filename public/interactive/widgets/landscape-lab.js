// Lesson 21: landscape lab — convex vs non-convex, stationary points, ball traps.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("landscape-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 520;

      var state = { bumps: 0, amp: 1.0, ball: null, balls: [] };

      K.hint(
        root,
        "Ползунок «ям» делает рельеф из выпуклого невыпуклым. Бросайте шарик из разных точек: в выпуклом он всегда в одном дне, в невыпуклом застревает в разных ловушках. Точки — стационарные (f'=0).",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Рельеф f(x) = 0,4·x² + амплитуда·sin(частота·x). При нуле ям это выпуклая парабола с единственным минимумом; добавляя ямы, получаем локальные минимумы-ловушки. Зелёная точка — глобальный минимум, красные — локальные. Клик по полю роняет шарик.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Рельеф функции со стационарными точками и падающими шариками",
        onResize: draw,
        drag: false,
      });

      var box = { x: 60, y: 30, w: 920, h: 440 };
      var XMIN = -4, XMAX = 4;

      function f(x) {
        return 0.4 * x * x + state.amp * state.bumps * 0.35 * Math.sin(state.bumps * 0.7 * x + 0.3);
      }
      function df(x) {
        return 0.8 * x + state.amp * state.bumps * 0.35 * state.bumps * 0.7 * Math.cos(state.bumps * 0.7 * x + 0.3);
      }

      function px(x) { return box.x + (x - XMIN) / (XMAX - XMIN) * box.w; }
      var YMIN = -1, YMAX = 9;
      function py(y) { return box.y + box.h - (y - YMIN) / (YMAX - YMIN) * box.h; }

      function stationaryPoints() {
        // scan for sign changes of df
        var pts = [];
        var prev = df(XMIN);
        for (var x = XMIN + 0.01; x <= XMAX; x += 0.01) {
          var d = df(x);
          if (prev < 0 && d >= 0) pts.push({ x: x, type: "min" });
          else if (prev > 0 && d <= 0) pts.push({ x: x, type: "max" });
          prev = d;
        }
        return pts;
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        ctx.strokeStyle = C.line;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        // curve
        ctx.save();
        ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.6; ctx.beginPath();
        for (var x = XMIN; x <= XMAX; x += 0.02) {
          if (x === XMIN) ctx.moveTo(px(x), py(f(x)));
          else ctx.lineTo(px(x), py(f(x)));
        }
        ctx.stroke(); ctx.lineWidth = 1;
        ctx.restore();

        // stationary points
        var sp = stationaryPoints();
        var mins = sp.filter(function (p) { return p.type === "min"; });
        var gmin = null;
        mins.forEach(function (p) { if (!gmin || f(p.x) < f(gmin.x)) gmin = p; });
        sp.forEach(function (p) {
          var col = p.type === "max" ? C.red : (p === gmin ? C.green : C.gold);
          if (p.type === "min" && p !== gmin) col = C.gold;
          ctx.fillStyle = p.type === "max" ? C.muted : (p === gmin ? C.green : C.red);
          ctx.beginPath(); ctx.arc(px(p.x), py(f(p.x)), 6, 0, Math.PI * 2); ctx.fill();
        });

        // balls (settled)
        state.balls.forEach(function (bx) {
          ctx.fillStyle = C.ink;
          ctx.beginPath(); ctx.arc(px(bx), py(f(bx)) - 8, 7, 0, Math.PI * 2); ctx.fill();
        });

        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("вес x", box.x + box.w / 2, box.y + box.h + 22);
        ctx.textAlign = "left";
        ctx.fillStyle = C.green; ctx.fillText("● глобальный минимум", box.x + 12, box.y + 18);
        ctx.fillStyle = C.red; ctx.fillText("● локальный минимум (ловушка)", box.x + 12, box.y + 38);

        var convex = state.bumps === 0;
        var rows = [
          { label: "Рельеф", value: convex ? "выпуклый" : "невыпуклый", color: convex ? C.green : C.gold },
          { label: "Локальных минимумов", value: String(mins.length), color: C.ink },
          { label: "Брошено шариков", value: String(state.balls.length), color: C.ink },
        ];
        if (convex) rows.push({ label: "Итог", value: "любой спуск найдёт единственное дно", color: C.green });
        else {
          var trapped = state.balls.filter(function (bx) { return !gmin || Math.abs(bx - gmin.x) > 0.3; }).length;
          rows.push({ label: "В ловушках", value: trapped + " из " + state.balls.length + " не в глобальном", color: trapped ? C.red : C.green });
        }
        output.set(rows);
      }

      // clicking drops a ball that rolls downhill
      canvasState.canvas.addEventListener("click", function (ev) {
        var rect = canvasState.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        if (mx < box.x || mx > box.x + box.w) return;
        var x = XMIN + (mx - box.x) / box.w * (XMAX - XMIN);
        // gradient descent
        for (var it = 0; it < 400; it += 1) {
          x = x - 0.05 * df(x);
          if (x < XMIN) x = XMIN; if (x > XMAX) x = XMAX;
        }
        state.balls.push(x);
        draw();
      });

      K.slider(controls, { label: "Локальных ям", min: 0, max: 6, step: 1, value: state.bumps },
        function (v) { state.bumps = v; state.balls = []; draw(); });
      K.slider(controls, { label: "Глубина ям", min: 0.2, max: 2, step: 0.1, value: state.amp,
        format: function (v) { return v.toFixed(1).replace(".", ","); } },
        function (v) { state.amp = v; state.balls = []; draw(); });
      K.segmented(controls, { label: "Действие", value: "info",
        options: [{ value: "info", label: "клик по полю роняет шарик" }, { value: "clear", label: "убрать шарики" }] },
        function (v) { if (v === "clear") { state.balls = []; draw(); } });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
