// Lesson 14: feedback control lab — tune the gain, watch stability vs oscillation.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("feedback-control-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 520;

      var TARGET = 22, X0 = 10, X_OUT = 0, ALPHA = 1.0, BETA = 0.0;
      var STEPS = 40;

      var state = { K: 0.7, disturb: 0 };

      K.hint(
        root,
        "Крутите усиление K и смотрите на отклик комнаты: вялый подъём, гладкий выход, затухающие колебания или расходящийся вой. Порог колебаний g=1, порог устойчивости g=2 (здесь g=K).",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Модель: xₜ₊₁ = xₜ + K·(r − xₜ), цель r = 22, старт 10. Отклонение за шаг умножается на (1−K): при K<1 монотонно к цели, 1<K<2 — затухающие колебания, K>2 — расходимость. Возмущение сбивает комнату в середине пути — контур сам её возвращает.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Отклик термостата во времени при выбранном усилении",
        onResize: draw,
        drag: false,
      });

      var box = { x: 64, y: 34, w: 930, h: 420 };
      var YMIN = -6, YMAX = 50;

      function px(t) { return box.x + t / STEPS * box.w; }
      function py(v) { return box.y + box.h - (v - YMIN) / (YMAX - YMIN) * box.h; }

      function simulate() {
        var xs = [X0];
        var x = X0;
        for (var t = 1; t <= STEPS; t += 1) {
          x = x + state.K * (TARGET - x) - BETA * (x - X_OUT);
          if (state.disturb && t === Math.floor(STEPS * 0.45)) x += state.disturb;
          xs.push(x);
        }
        return xs;
      }

      function regime() {
        var g = state.K;
        if (g <= 0) return { name: "нет управления", color: C.muted };
        if (g < 1) return { name: "вяло, монотонно к цели", color: C.blue };
        if (g < 2) return { name: "затухающие колебания", color: C.gold };
        if (g === 2 || Math.abs(g - 2) < 1e-9) return { name: "незатухающие колебания", color: C.violet };
        return { name: "расходимость (вой)", color: C.red };
      }

      function settleTime(xs) {
        // first step after which stays within 2% of target range
        var tol = 0.02 * Math.abs(TARGET - X0);
        for (var t = 0; t < xs.length; t += 1) {
          var ok = true;
          for (var k = t; k < xs.length; k += 1) {
            if (Math.abs(xs[k] - TARGET) > tol) { ok = false; break; }
          }
          if (ok) return t;
        }
        return null;
      }

      function overshoot(xs) {
        var mx = Math.max.apply(null, xs);
        return Math.max(0, mx - TARGET);
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        // grid + target line
        ctx.strokeStyle = C.grid;
        ctx.fillStyle = C.muted;
        ctx.textAlign = "right";
        [0, 10, 20, 30, 40, 50].forEach(function (v) {
          if (v < YMIN || v > YMAX) return;
          ctx.beginPath();
          ctx.moveTo(box.x, py(v));
          ctx.lineTo(box.x + box.w, py(v));
          ctx.stroke();
          ctx.fillStyle = C.grid;
          ctx.fillStyle = C.muted;
          ctx.fillText(String(v), box.x - 8, py(v));
        });
        ctx.strokeStyle = C.ink;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(box.x, py(TARGET));
        ctx.lineTo(box.x + box.w, py(TARGET));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.ink;
        ctx.textAlign = "left";
        ctx.fillText("цель r = 22", box.x + box.w - 96, py(TARGET) - 12);

        var xs = simulate();
        var reg = regime();
        // clip drawing to box
        ctx.save();
        ctx.beginPath();
        ctx.rect(box.x, box.y, box.w, box.h);
        ctx.clip();
        ctx.strokeStyle = reg.color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        xs.forEach(function (v, t) {
          if (t === 0) ctx.moveTo(px(t), py(v));
          else ctx.lineTo(px(t), py(v));
        });
        ctx.stroke();
        ctx.fillStyle = reg.color;
        xs.forEach(function (v, t) {
          if (t % 1 === 0 && v >= YMIN && v <= YMAX) {
            ctx.beginPath();
            ctx.arc(px(t), py(v), 2.6, 0, Math.PI * 2);
            ctx.fill();
          }
        });
        // disturbance marker
        if (state.disturb) {
          var dt = Math.floor(STEPS * 0.45);
          ctx.strokeStyle = C.muted;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(px(dt), box.y);
          ctx.lineTo(px(dt), box.y + box.h);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.restore();
        ctx.lineWidth = 1;
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText("шаг времени", box.x + box.w / 2, box.y + box.h + 24);
        if (state.disturb) {
          ctx.fillStyle = C.muted;
          ctx.fillText("возмущение", px(Math.floor(STEPS * 0.45)), box.y + 12);
        }

        var st = settleTime(xs);
        var rows = [
          { label: "Режим", value: reg.name, color: reg.color },
          { label: "Время установления", value: st === null ? "не устанавливается" : st + " шагов", color: st === null ? C.red : C.ink },
          { label: "Перерегулирование", value: overshoot(xs).toFixed(1).replace(".", ",") + " °", color: C.ink },
        ];
        output.set(rows);
      }

      K.slider(controls, { label: "Усиление K", min: 0.1, max: 2.4, step: 0.05, value: state.K,
        format: function (v) { return v.toFixed(2).replace(".", ","); } },
        function (v) { state.K = v; draw(); });
      K.segmented(controls, {
        label: "Возмущение в середине",
        value: String(state.disturb),
        options: [
          { value: "0", label: "нет" },
          { value: "12", label: "толчок +12°" },
        ],
      }, function (v) { state.disturb = Number(v); draw(); });

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
