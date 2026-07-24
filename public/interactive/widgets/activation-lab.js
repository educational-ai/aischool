// Lesson 17: activation lab — function, its slope, and gradient decay through layers.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("activation-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 520;

      function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
      var ACTS = {
        step: { name: "ступенька", f: function (z) { return z >= 0 ? 1 : 0; },
          d: function (z) { return 0; }, maxd: 0, lo: -0.15, hi: 1.15 },
        sigmoid: { name: "сигмоида", f: sigmoid,
          d: function (z) { var s = sigmoid(z); return s * (1 - s); }, maxd: 0.25, lo: -0.15, hi: 1.15 },
        tanh: { name: "tanh", f: function (z) { return Math.tanh(z); },
          d: function (z) { var t = Math.tanh(z); return 1 - t * t; }, maxd: 1, lo: -1.2, hi: 1.2 },
        relu: { name: "ReLU", f: function (z) { return Math.max(0, z); },
          d: function (z) { return z > 0 ? 1 : 0; }, maxd: 1, lo: -0.5, hi: 5 },
        leaky: { name: "дырявая ReLU", f: function (z) { return z >= 0 ? z : 0.01 * z; },
          d: function (z) { return z >= 0 ? 1 : 0.01; }, maxd: 1, lo: -0.5, hi: 5 },
      };

      var state = { act: "sigmoid", layers: 6 };

      K.hint(
        root,
        "Выбирайте активацию: слева её график, справа наклон. Затем крутите число слоёв и следите за множителем градиента — сигмоида гаснет как (макс. наклон)^L, ReLU держится.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Множитель градиента через L слоёв оценён как (максимальный наклон активации)^L — грубая верхняя оценка. У сигмоиды это (1/4)^L и гаснет; у ReLU (1)^L и держится. Наклон-колокол показывает, где активация «насыщается» и обучается медленно.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "График активации, её наклон и затухание градиента по слоям",
        onResize: draw,
        drag: false,
      });

      var fbox = { x: 60, y: 40, w: 300, h: 210 };
      var dbox = { x: 60, y: 290, w: 300, h: 200 };
      var vbox = { x: 470, y: 40, w: 520, h: 450 };

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var A = ACTS[state.act];

        // function plot
        plotCurve(ctx, fbox, function (z) { return A.f(z); }, A.lo, A.hi, C.blue,
          "функция " + A.name);
        // derivative plot
        plotCurve(ctx, dbox, function (z) { return A.d(z); }, -0.1,
          Math.max(0.3, A.maxd * 1.15), C.violet, "наклон (производная)");
        if (A.maxd > 0) {
          ctx.fillStyle = C.violet;
          ctx.textAlign = "right";
          ctx.fillText("макс " + (A.maxd === 0.25 ? "1/4" : A.maxd),
            dbox.x + dbox.w - 6, dbox.y + 14);
        } else {
          ctx.fillStyle = C.red;
          ctx.textAlign = "right";
          ctx.fillText("всюду 0 — не учится", dbox.x + dbox.w - 6, dbox.y + 14);
        }

        // vanishing plot: (maxd)^L for L=1..12, log scale
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Множитель градиента через L слоёв", vbox.x, vbox.y - 12);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.strokeStyle = C.line;
        ctx.strokeRect(vbox.x, vbox.y, vbox.w, vbox.h);
        var logMin = -8, logMax = 0.5;
        function vx(L) { return vbox.x + (L - 1) / 11 * vbox.w; }
        function vy(val) {
          var lg = Math.log10(Math.max(val, 1e-9));
          return vbox.y + vbox.h - (lg - logMin) / (logMax - logMin) * vbox.h;
        }
        // grid decades
        ctx.textAlign = "right";
        for (var e = 0; e >= -8; e -= 2) {
          var yy = vy(Math.pow(10, e));
          ctx.strokeStyle = C.grid;
          ctx.beginPath(); ctx.moveTo(vbox.x, yy); ctx.lineTo(vbox.x + vbox.w, yy); ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.fillText("10^" + e, vbox.x - 6, yy);
        }
        var md = A.maxd > 0 ? A.maxd : 0.0001;
        // sigmoid-style curve and relu reference
        drawSeries(ctx, vx, vy, function (L) { return Math.pow(md, L); }, C.blue, true);
        drawSeries(ctx, vx, vy, function (L) { return 1; }, C.green, false);
        // marker at current layers
        var cur = Math.pow(md, state.layers);
        ctx.fillStyle = C.red;
        ctx.beginPath(); ctx.arc(vx(state.layers), vy(cur), 6, 0, Math.PI * 2); ctx.fill();
        ctx.textAlign = "center";
        ctx.fillStyle = C.muted;
        [1, 4, 8, 12].forEach(function (L) { ctx.fillText("L=" + L, vx(L), vbox.y + vbox.h + 16); });

        var multiplier = A.maxd > 0 ? Math.pow(A.maxd, state.layers) : 0;
        var rows = [
          { label: "Активация", value: A.name, color: C.blue },
          { label: "Макс. наклон", value: A.maxd === 0.25 ? "1/4" : String(A.maxd), color: C.violet },
          { label: "Множитель через " + state.layers + " слоёв", value: fmtSci(multiplier), color: multiplier < 1e-3 ? C.red : C.ink },
        ];
        if (state.act === "step") rows.push({ label: "Итог", value: "наклон 0 — градиентом не обучить", color: C.red });
        else if (A.maxd >= 1) rows.push({ label: "Итог", value: "градиент проходит сквозь глубину", color: C.green });
        else if (multiplier < 1e-4) rows.push({ label: "Итог", value: "градиент погас — ранние слои не учатся", color: C.red });
        output.set(rows);
      }

      function fmtSci(v) {
        if (v === 0) return "0";
        if (v >= 0.01) return v.toFixed(3).replace(".", ",");
        var e = Math.floor(Math.log10(v));
        var m = v / Math.pow(10, e);
        return m.toFixed(1).replace(".", ",") + "·10^" + e;
      }

      function drawSeries(ctx, vx, vy, fn, color, dashed) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        if (dashed) ctx.setLineDash([]);
        ctx.beginPath();
        for (var L = 1; L <= 12; L += 1) {
          if (L === 1) ctx.moveTo(vx(L), vy(fn(L)));
          else ctx.lineTo(vx(L), vy(fn(L)));
        }
        ctx.stroke();
        ctx.lineWidth = 1;
      }

      function plotCurve(ctx, bx, fn, lo, hi, color, title) {
        ctx.strokeStyle = C.line;
        ctx.strokeRect(bx.x, bx.y, bx.w, bx.h);
        ctx.fillStyle = C.ink;
        ctx.font = "bold 13px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(title, bx.x, bx.y - 8);
        ctx.font = "13px PT Sans, sans-serif";
        var zmin = -5, zmax = 5;
        function xx(z) { return bx.x + (z - zmin) / (zmax - zmin) * bx.w; }
        function yy(v) { return bx.y + bx.h - (v - lo) / (hi - lo) * bx.h; }
        // zero axes
        ctx.strokeStyle = C.grid;
        ctx.beginPath(); ctx.moveTo(bx.x, yy(0)); ctx.lineTo(bx.x + bx.w, yy(0)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xx(0), bx.y); ctx.lineTo(xx(0), bx.y + bx.h); ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        var started = false, prevBig = false;
        for (var z = zmin; z <= zmax; z += 0.05) {
          var v = fn(z);
          var vy2 = yy(Math.max(lo, Math.min(hi, v)));
          if (!started) { ctx.moveTo(xx(z), vy2); started = true; }
          else ctx.lineTo(xx(z), vy2);
        }
        ctx.stroke();
        ctx.lineWidth = 1;
      }

      K.segmented(controls, { label: "Активация", value: state.act,
        options: [
          { value: "step", label: "ступенька" },
          { value: "sigmoid", label: "сигмоида" },
          { value: "tanh", label: "tanh" },
          { value: "relu", label: "ReLU" },
          { value: "leaky", label: "дырявая ReLU" },
        ] }, function (v) { state.act = v; draw(); });
      K.slider(controls, { label: "Число слоёв L", min: 1, max: 12, step: 1, value: state.layers },
        function (v) { state.layers = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
