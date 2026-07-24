// Lesson 73: recurrent-memory-lab — one recurrent multiplier sets both the memory horizon
// and the reach of the gradient. Move u, watch the state decay and the Jacobian product.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("recurrent-memory-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 520;
      var T = 60;

      var state = { u: 0.8, w: 1.0, amp: 1.0, nonlin: "tanh", signal: "impulse" };

      K.hint(
        root,
        "Ячейка считает h_t = φ(w·x_t + u·h_{t−1}). Один множитель u задаёт сразу две вещи: сколько шагов держится след входа и как далеко назад доходит градиент. Двигайте u около единицы, включайте tanh и большую амплитуду — и смотрите, где память есть, а обучающего сигнала уже нет.",
      );

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Сверху — вход x_t (серые столбики) и состояние h_t (синяя линия). Снизу — модуль производной ∂h_T/∂h_t по лагу T−t в логарифмическом масштабе: это множитель, на который умножается обучающий сигнал из будущего. При |u| < 1 обе кривые гаснут геометрически, при u > 1 обе взрываются. Нелинейность tanh держит состояние в коридоре (−1; 1), но её производная 1 − h² при насыщении почти нулевая, поэтому память может остаться, а градиент — исчезнуть.",
      );

      var cs = K.makeCanvas(stage, W, H, {
        label: "Состояние рекуррентной ячейки и дальность градиента",
        onResize: draw,
        drag: false,
      });

      function inputAt(t) {
        if (state.signal === "impulse") return t === 0 ? state.amp : 0;
        if (state.signal === "step") return t < 10 ? state.amp : 0;
        if (state.signal === "daily") return state.amp * Math.sin((2 * Math.PI * t) / 24);
        // deterministic pseudo-noise, stable across redraws
        var s = Math.sin(t * 12.9898 + 78.233) * 43758.5453;
        return state.amp * (2 * (s - Math.floor(s)) - 1);
      }

      function run() {
        var h = 0, hs = [], xs = [], der = [];
        for (var t = 0; t < T; t += 1) {
          var x = inputAt(t);
          var z = state.w * x + state.u * h;
          h = state.nonlin === "tanh" ? Math.tanh(z) : z;
          xs.push(x);
          hs.push(h);
          der.push(state.nonlin === "tanh" ? 1 - h * h : 1);
        }
        // product of Jacobians from the last step backwards
        var jac = [1], p = 1;
        for (var k = T - 1; k >= 1; k -= 1) {
          p = p * der[k] * state.u;
          jac.push(Math.abs(p));
        }
        return { xs: xs, hs: hs, jac: jac, der: der };
      }

      function halfLife() {
        var a = Math.abs(state.u);
        if (a >= 1 || a <= 0) return null;
        return 1 + Math.log(0.5) / Math.log(a);
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var r = run();

        var gx = 68, gw = W - 110;
        function X(t) { return gx + (t / (T - 1)) * gw; }

        // ---------- top panel: input and state
        var ty = 34, th = 200, mid = ty + th / 2;
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, ty, gw, th);
        var span = 1.0;
        for (var i = 0; i < T; i += 1) {
          span = Math.max(span, Math.abs(r.hs[i]), Math.abs(r.xs[i]));
        }
        span = Math.min(span, 6);
        function Y(v) { return mid - Math.max(-span, Math.min(span, v)) / span * (th / 2 - 8); }

        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.beginPath(); ctx.moveTo(gx, mid); ctx.lineTo(gx + gw, mid); ctx.stroke();

        ctx.fillStyle = "rgba(150,153,144,0.55)";
        for (i = 0; i < T; i += 1) {
          var bh = Y(r.xs[i]) - mid;
          ctx.fillRect(X(i) - 3, Math.min(mid, mid + bh), 6, Math.abs(bh));
        }
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.4; ctx.beginPath();
        for (i = 0; i < T; i += 1) {
          if (i === 0) ctx.moveTo(X(i), Y(r.hs[i])); else ctx.lineTo(X(i), Y(r.hs[i]));
        }
        ctx.stroke();

        // half-life marker
        var hl = halfLife();
        if (state.signal === "impulse" && hl && hl < T) {
          ctx.strokeStyle = C.red; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
          ctx.beginPath(); ctx.moveTo(X(hl - 1), ty + 4); ctx.lineTo(X(hl - 1), ty + th - 4); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = C.red; ctx.textAlign = "left";
          ctx.fillText("половина импульса", X(hl - 1) + 5, ty + 16);
        }
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("вход x_t (серое) и состояние h_t (синее)", gx, ty - 9);
        ctx.font = "11px PT Sans, sans-serif"; ctx.fillStyle = C.muted; ctx.textAlign = "right";
        ctx.fillText("+" + span.toFixed(1).replace(".", ","), gx - 8, ty + 12);
        ctx.fillText("0", gx - 8, mid + 4);
        ctx.fillText("−" + span.toFixed(1).replace(".", ","), gx - 8, ty + th - 4);

        // ---------- bottom panel: |dh_T/dh_t| in log scale
        var by = 296, bh2 = 176;
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, by, gw, bh2);
        var LO = -18, HI = 6;
        function YL(v) { return by + bh2 - (Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO) * bh2; }
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "10.5px PT Sans, sans-serif";
        for (var g = LO; g <= HI; g += 6) {
          ctx.beginPath(); ctx.moveTo(gx, YL(g)); ctx.lineTo(gx + gw, YL(g)); ctx.stroke();
          ctx.fillText("10^" + g, gx - 8, YL(g) + 4);
        }
        // band of practical invisibility
        ctx.fillStyle = "rgba(185,74,59,0.08)";
        ctx.fillRect(gx + 1, YL(-7), gw - 2, YL(LO) - YL(-7));
        ctx.fillStyle = C.red; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("сигнал тонет в шуме оптимизации", gx + 8, YL(-7) + 15);

        ctx.strokeStyle = C.gold; ctx.lineWidth = 2.4; ctx.beginPath();
        for (i = 0; i < r.jac.length; i += 1) {
          var lv = Math.log(r.jac[i] + 1e-300) / Math.LN10;
          var px = X(i), py = YL(lv);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("|∂h_T/∂h_t| по лагу T−t (десятичный логарифм)", gx, by - 9);
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "center";
        for (var tk = 0; tk < T; tk += 10) ctx.fillText(String(tk), X(tk), by + bh2 + 16);
        ctx.fillText("лаг", gx + gw + 18, by + bh2 + 16);

        var lag30 = r.jac.length > 30 ? Math.log(r.jac[30] + 1e-300) / Math.LN10 : 0;
        var reach = 0;
        for (i = 0; i < r.jac.length; i += 1) {
          if (r.jac[i] >= 1e-7) reach = i;
        }
        output.set([
          { label: "время полузабывания", value: hl ? hl.toFixed(1).replace(".", ",") + " шага" : "память не гаснет", color: C.blue },
          { label: "|h| на 30-м шаге", value: Math.abs(r.hs[Math.min(29, T - 1)]).toFixed(3).replace(".", ","), color: C.blue },
          { label: "градиент на лаге 30", value: "10^" + lag30.toFixed(1).replace(".", ","), color: C.gold },
          { label: "дальность обучения (до 10⁻⁷)", value: reach + " шагов", color: C.red },
        ]);
      }

      K.slider(controls, {
        label: "Рекуррентный множитель u", min: -1.2, max: 1.2, step: 0.01, value: state.u,
        format: function (v) { return v.toFixed(2).replace(".", ","); },
      }, function (v) { state.u = v; draw(); });

      K.slider(controls, {
        label: "Амплитуда входа", min: 0.2, max: 6, step: 0.1, value: state.amp,
        format: function (v) { return v.toFixed(1).replace(".", ","); },
      }, function (v) { state.amp = v; draw(); });

      K.segmented(controls, {
        label: "Нелинейность", value: state.nonlin,
        options: [{ value: "none", label: "нет (линейная)" }, { value: "tanh", label: "tanh" }],
      }, function (v) { state.nonlin = v; draw(); });

      K.segmented(controls, {
        label: "Вход", value: state.signal,
        options: [
          { value: "impulse", label: "импульс" },
          { value: "step", label: "ступенька" },
          { value: "daily", label: "суточный ритм" },
          { value: "noise", label: "шум" },
        ],
      }, function (v) { state.signal = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
