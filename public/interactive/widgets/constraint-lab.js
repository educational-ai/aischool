// Lesson 23: constraint lab — L1 diamond vs L2 circle, tangency and sparsity.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("constraint-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;

      // loss = A*(w1-cx)^2 + B*(w2-cy)^2 ; free minimum at target (cx,cy)
      var state = { kind: "l1", R: 1.3, target: [2.1, 0.6], A: 1.0, B: 1.4 };

      K.hint(
        root,
        "Кликните по полю, чтобы передвинуть свободный минимум потери. Меняйте бюджет R и форму ограничения. Ромб (L₁) любит цепляться углом за ось — тогда один вес становится ровно нулём. Круг (L₂) сжимает оба веса, но нулей не даёт.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Серые эллипсы — линии уровня квадратичной потери с центром в свободном минимуме. Синяя область — допустимый бюджет ‖w‖ ≤ R. Красная точка — решение с ограничением: там линия уровня впервые касается границы. У ромба касание часто приходится на угол, где одна координата равна нулю.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Линии уровня потери и допустимая область L1 или L2",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 30, w: 500, h: 500 };
      var WMIN = -2.6, WMAX = 3.2, HMIN = -2.9, HMAX = 2.9;

      function px(w) { return box.x + (w - WMIN) / (WMAX - WMIN) * box.w; }
      function py(w) { return box.y + box.h - (w - HMIN) / (HMAX - HMIN) * box.h; }

      function loss(w1, w2) {
        var dx = w1 - state.target[0], dy = w2 - state.target[1];
        return state.A * dx * dx + state.B * dy * dy;
      }

      function inside(w1, w2) {
        if (state.kind === "l2") return w1 * w1 + w2 * w2 <= state.R * state.R + 1e-9;
        return Math.abs(w1) + Math.abs(w2) <= state.R + 1e-9;
      }

      // minimise loss subject to the budget: free min if inside, else search boundary
      function solve() {
        var cx = state.target[0], cy = state.target[1];
        if (inside(cx, cy)) return { w: [cx, cy], active: false };
        var best = null, bv = Infinity;
        var N = 2400;
        for (var i = 0; i < N; i += 1) {
          var t = i / N, w1, w2;
          if (state.kind === "l2") {
            var a = t * 2 * Math.PI;
            w1 = state.R * Math.cos(a); w2 = state.R * Math.sin(a);
          } else {
            // diamond perimeter: 4 edges between axis vertices
            var seg = Math.floor(t * 4), u = (t * 4) - seg;
            var verts = [[state.R, 0], [0, state.R], [-state.R, 0], [0, -state.R]];
            var p0 = verts[seg], p1 = verts[(seg + 1) % 4];
            w1 = p0[0] + (p1[0] - p0[0]) * u; w2 = p0[1] + (p1[1] - p0[1]) * u;
          }
          var v = loss(w1, w2);
          if (v < bv) { bv = v; best = [w1, w2]; }
        }
        return { w: best, active: true };
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        // loss level curves (ellipses through several radii)
        var sol = solve();
        var lvlSol = loss(sol.w[0], sol.w[1]);
        var levels = [0.4, 1.2, 2.5, 4.5, 7.0, 10.5];
        ctx.lineWidth = 1.0;
        levels.forEach(function (lv) {
          ctx.strokeStyle = "rgba(150,153,148,0.55)";
          drawEllipse(ctx, lv);
        });
        // the level curve touching the boundary
        ctx.strokeStyle = C.red || "#b94a3b"; ctx.lineWidth = 1.6;
        drawEllipse(ctx, lvlSol);
        ctx.lineWidth = 1.0;

        // constraint region
        ctx.fillStyle = "rgba(49,95,140,0.12)";
        ctx.strokeStyle = C.blue || "#315f8c"; ctx.lineWidth = 2.0;
        ctx.beginPath();
        if (state.kind === "l2") {
          ctx.ellipse(px(0), py(0), (px(state.R) - px(0)), (py(0) - py(state.R)), 0, 0, Math.PI * 2);
        } else {
          ctx.moveTo(px(state.R), py(0));
          ctx.lineTo(px(0), py(state.R));
          ctx.lineTo(px(-state.R), py(0));
          ctx.lineTo(px(0), py(-state.R));
          ctx.closePath();
        }
        ctx.fill(); ctx.stroke(); ctx.lineWidth = 1.0;

        // axes
        ctx.strokeStyle = C.line || "#c9c8be";
        ctx.beginPath(); ctx.moveTo(px(WMIN), py(0)); ctx.lineTo(px(WMAX), py(0)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px(0), py(HMIN)); ctx.lineTo(px(0), py(HMAX)); ctx.stroke();
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        // free minimum
        ctx.fillStyle = C.faint || "#969990";
        ctx.beginPath(); ctx.arc(px(state.target[0]), py(state.target[1]), 6, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = C.muted || "#6e726a"; ctx.textAlign = "left";
        ctx.fillText("свободный минимум", px(state.target[0]) + 10, py(state.target[1]));

        // solution point
        ctx.fillStyle = C.red || "#b94a3b";
        ctx.beginPath(); ctx.arc(px(sol.w[0]), py(sol.w[1]), 7, 0, Math.PI * 2); ctx.fill();

        // gradient arrow at solution: points uphill (away from target)
        var gx = 2 * state.A * (sol.w[0] - state.target[0]);
        var gy = 2 * state.B * (sol.w[1] - state.target[1]);
        var gn = Math.hypot(gx, gy);
        if (gn > 1e-6 && sol.active) {
          var scg = 0.7 / gn;
          drawArrow(ctx, px(sol.w[0]), py(sol.w[1]), px(sol.w[0] + gx * scg), py(sol.w[1] + gy * scg), C.red || "#b94a3b");
        }

        var sparse = Math.abs(sol.w[0]) < 0.03 || Math.abs(sol.w[1]) < 0.03;
        if (sparse && sol.active) {
          var zeroX = Math.abs(sol.w[0]) < Math.abs(sol.w[1]);
          ctx.fillStyle = C.gold || "#a57920"; ctx.textAlign = "center";
          ctx.fillText(zeroX ? "w₁ = 0" : "w₂ = 0", px(sol.w[0]) + (zeroX ? 46 : 0), py(sol.w[1]) + (zeroX ? 0 : 26));
        }

        // labels
        ctx.fillStyle = C.muted || "#6e726a"; ctx.textAlign = "center";
        ctx.fillText("вес w₁", px(0) + box.w * 0.32, py(HMIN) + 20);
        ctx.save(); ctx.translate(px(WMIN) - 22, py(0) - box.h * 0.30); ctx.rotate(-Math.PI / 2);
        ctx.fillText("вес w₂", 0, 0); ctx.restore();

        var rows = [
          { label: "Ограничение", value: state.kind === "l1" ? "L₁ (ромб)" : "L₂ (круг)", color: C.ink },
          { label: "Бюджет R", value: state.R.toFixed(2).replace(".", ","), color: C.ink },
          { label: "Решение w", value: "(" + fmt(sol.w[0]) + "; " + fmt(sol.w[1]) + ")", color: C.ink },
        ];
        if (!sol.active) {
          rows.push({ label: "Итог", value: "бюджета хватает — ограничение не активно", color: C.green });
        } else if (sparse) {
          rows.push({ label: "Нулевых весов", value: "один — решение разрежено", color: C.gold || "#a57920" });
        } else {
          rows.push({ label: "Нулевых весов", value: "ноль — оба веса работают", color: C.muted });
        }
        output.set(rows);
      }

      function fmt(v) { return (Math.abs(v) < 0.03 ? 0 : v).toFixed(2).replace(".", ","); }

      function drawEllipse(ctx, lv) {
        // loss = A dx^2 + B dy^2 = lv  ->  ellipse centred at target
        var rx = Math.sqrt(lv / state.A), ry = Math.sqrt(lv / state.B);
        ctx.beginPath();
        ctx.ellipse(px(state.target[0]), py(state.target[1]),
          px(state.target[0] + rx) - px(state.target[0]),
          py(state.target[1]) - py(state.target[1] + ry), 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      function drawArrow(ctx, x0, y0, x1, y1, col) {
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        var ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 10 * Math.cos(ang - 0.4), y1 - 10 * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - 10 * Math.cos(ang + 0.4), y1 - 10 * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
        ctx.lineWidth = 1;
      }

      canvasState.canvas.addEventListener("click", function (ev) {
        var rect = canvasState.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        var my = (ev.clientY - rect.top) / rect.height * H;
        if (mx < box.x || mx > box.x + box.w || my < box.y || my > box.y + box.h) return;
        var w1 = WMIN + (mx - box.x) / box.w * (WMAX - WMIN);
        var w2 = HMIN + (box.y + box.h - my) / box.h * (HMAX - HMIN);
        state.target = [w1, w2];
        draw();
      });

      K.segmented(controls, {
        label: "Форма ограничения",
        value: state.kind,
        options: [{ value: "l1", label: "L₁ ромб" }, { value: "l2", label: "L₂ круг" }],
      }, function (v) { state.kind = v; draw(); });

      K.slider(controls, { label: "Бюджет R", min: 0.3, max: 2.6, step: 0.05, value: state.R,
        format: function (v) { return v.toFixed(2).replace(".", ","); } },
        function (v) { state.R = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
