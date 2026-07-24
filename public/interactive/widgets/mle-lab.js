// Lesson 45: mle-lab — slide the data, watch the likelihood peak and sharpen; the maximum is the estimate.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("mle-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var n = 20, frac = 0.7;

      K.hint(
        root,
        "Функция правдоподобия сравнивает значения параметра по тому, как хорошо каждое объясняет данные. Двигайте долю успехов и число наблюдений: вершина кривой — это оценка максимального правдоподобия, а её ширина — неопределённость. Больше данных — острее пик и точнее оценка.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Сверху — правдоподобие L(p) = p^h (1−p)^(n−h): произведение вероятностей данных как функция параметра. Снизу — лог-правдоподобие ℓ(p): его максимум в той же точке p̂ = h/n. Заштрихован интервал правдоподобия (падение логарифма меньше 1,92). Его ширина убывает как 1/√n.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Правдоподобие и его максимум", onResize: draw, drag: false });

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var h = Math.round(frac * n), t = n - h;
        var phat = h / n;
        var gx = 60, gw = W - 100;

        function ell(p) { var e = 0; if (h > 0) e += h * Math.log(Math.max(1e-12, p)); if (t > 0) e += t * Math.log(Math.max(1e-12, 1 - p)); return e; }
        var lmax = ell(Math.min(0.999999, Math.max(1e-6, phat)));

        // top strip: likelihood L(p) normalised
        var ty = 40, th = 120;
        ctx.strokeStyle = C.line; ctx.strokeRect(gx, ty, gw, th);
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("правдоподобие L(p)", gx, ty - 8);
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.0; ctx.beginPath();
        for (var s = 0; s <= 200; s += 1) {
          var p = s / 200, L = Math.exp(ell(p) - lmax);
          var px = gx + p * gw, py = ty + th - L * (th - 6) - 3;
          if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // main: log-likelihood
        var my = 210, mh = 200;
        ctx.strokeStyle = C.line; ctx.strokeRect(gx, my, gw, mh);
        ctx.fillStyle = C.ink; ctx.fillText("лог-правдоподобие ℓ(p) − ℓ(p̂)", gx, my - 8);
        var ylo = -8;
        function Y(v) { return my + mh - (v - ylo) / (0 - ylo) * mh; }
        // 95% likelihood interval: drop < 1.92
        var loI = null, hiI = null;
        for (var q = 0; q <= 400; q += 1) {
          var pp = q / 400, drop = ell(pp) - lmax;
          if (drop >= -1.92) { if (loI === null) loI = pp; hiI = pp; }
        }
        if (loI !== null) {
          ctx.fillStyle = "rgba(165,121,32,0.16)";
          ctx.fillRect(gx + loI * gw, my, (hiI - loI) * gw, mh);
        }
        // curve
        ctx.strokeStyle = C.green; ctx.lineWidth = 2.2; ctx.beginPath();
        var started = false;
        for (var s2 = 0; s2 <= 400; s2 += 1) {
          var p2 = s2 / 400, d = ell(p2) - lmax;
          if (d < ylo) { started = false; continue; }
          var px2 = gx + p2 * gw, py2 = Y(d);
          if (!started) { ctx.moveTo(px2, py2); started = true; } else ctx.lineTo(px2, py2);
        }
        ctx.stroke();
        // MLE line
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(gx + phat * gw, my); ctx.lineTo(gx + phat * gw, my + mh); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.red; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("p̂ = " + phat.toFixed(2), gx + phat * gw, my - 8 < my ? my + 16 : my + 16);
        // x ticks
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        for (var xt = 0; xt <= 5; xt += 1) ctx.fillText((xt / 5).toFixed(1), gx + xt / 5 * gw, my + mh + 16);
        ctx.fillText("параметр p", gx + gw / 2, my + mh + 32);

        var se = Math.sqrt(Math.max(1e-9, phat * (1 - phat) / n));
        output.set([
          { label: "Оценка p̂ = h/n", value: h + "/" + n + " = " + phat.toFixed(3), color: C.red },
          { label: "Станд. ошибка ≈ √(p(1−p)/n)", value: se.toFixed(3), color: C.blue },
          { label: "Интервал правдоподобия", value: loI !== null ? "[" + loI.toFixed(2) + ", " + hiI.toFixed(2) + "]" : "—", color: C.gold },
        ]);
      }

      K.slider(controls, { label: "Число наблюдений n", min: 2, max: 500, step: 1, value: 20, format: function (v) { return String(v); } }, function (v) { n = v; draw(); });
      K.slider(controls, { label: "Доля успехов", min: 0, max: 1, step: 0.05, value: 0.7, format: function (v) { return v.toFixed(2); } }, function (v) { frac = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
