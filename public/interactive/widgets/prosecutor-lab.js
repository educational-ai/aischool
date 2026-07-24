// Lesson 43: prosecutor-lab — how database size and prior turn a rare match into (or away from) guilt.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("prosecutor-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var logp = -6, logN = 6, logPriorDen = 6, sens = 0.98; // p=1e-6, N=1e6, prior 1:1e6

      K.hint(
        root,
        "«Один шанс на миллион» звучит как приговор, но всё решают ещё два числа: сколько записей проверили и каковы были шансы виновности до улики. Двигайте ползунки и смотрите, как одна и та же редкая улика даёт совсем разную вероятность вины.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Отношение правдоподобий LR = чувствительность / вероятность случайного совпадения. Апостериорные шансы = априорные шансы × LR. При поиске по огромной базе даже редкое совпадение почти наверняка встретится у кого-то невиновного, поэтому база и prior решают исход, а не одна частота совпадения.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Редкая улика, размер базы и апостериор", onResize: draw, drag: false });

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var p = Math.pow(10, logp), N = Math.round(Math.pow(10, logN));
        var priorOdds = 1 / Math.pow(10, logPriorDen);
        var LR = sens / p;
        var postOdds = priorOdds * LR;
        var postProb = postOdds / (1 + postOdds);
        var expected = (N - 1) * p;
        var atLeast = 1 - Math.pow(1 - p, N);

        // database curve panel
        var gx = 60, gy = 55, gw = W - 100, gh = 175;
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, gy, gw, gh);
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("вероятность хотя бы одного случайного совпадения в базе", gx, gy - 12);
        var lnLo = 0, lnHi = 7; // log10 N axis
        function X(ln) { return gx + (ln - lnLo) / (lnHi - lnLo) * gw; }
        function Y(pr) { return gy + gh - pr * gh; }
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.2; ctx.beginPath();
        for (var s = 0; s <= 100; s += 1) {
          var ln = lnLo + (lnHi - lnLo) * s / 100, nn = Math.pow(10, ln);
          var pr = 1 - Math.pow(1 - p, nn);
          if (s === 0) ctx.moveTo(X(ln), Y(pr)); else ctx.lineTo(X(ln), Y(pr));
        }
        ctx.stroke();
        // current N marker
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(X(logN), gy); ctx.lineTo(X(logN), gy + gh); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(X(logN), Y(atLeast), 6, 0, 7); ctx.fill();
        ctx.fillStyle = C.ink; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText((atLeast * 100).toFixed(atLeast > 0.99 ? 1 : 0) + "%", X(logN), Y(atLeast) - 10);
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        for (var t = 0; t <= 7; t += 1) ctx.fillText("10" + supr(t), X(t), gy + gh + 15);
        ctx.fillText("размер базы N", (gx + gw / 2), gy + gh + 32);

        // odds shift panel
        var oy = 300;
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("улика умножает шансы виновности на LR = " + fmt(LR), gx, oy - 6);
        var bx = gx, bw = gw, by = oy + 18;
        function OX(logodds) { return bx + (logodds + 8) / 12 * bw; } // log10 odds from -8 to 4
        ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(bx, by + 10); ctx.lineTo(bx + bw, by + 10); ctx.stroke();
        var lpr = Math.log10(priorOdds), lpo = Math.log10(postOdds);
        ctx.strokeStyle = C.gold; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(OX(lpr), by + 10); ctx.lineTo(OX(Math.min(4, lpo)), by + 10); ctx.stroke();
        ctx.fillStyle = C.blue; ctx.beginPath(); ctx.arc(OX(lpr), by + 10, 7, 0, 7); ctx.fill();
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(OX(Math.min(4, lpo)), by + 10, 7, 0, 7); ctx.fill();
        ctx.fillStyle = C.blue; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("до улики", OX(lpr), by + 30);
        ctx.fillStyle = C.red; ctx.fillText("после улики", OX(Math.min(4, lpo)), by - 6);

        // big posterior
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "15px PT Sans, sans-serif";
        ctx.fillText("Вероятность вины после улики:", gx, oy + 92);
        ctx.font = "30px PT Sans, sans-serif"; ctx.fillStyle = postProb > 0.5 ? C.red : C.green;
        ctx.fillText((postProb * 100).toFixed(postProb < 0.1 ? 1 : 0) + "%", gx + 260, oy + 95);

        output.set([
          { label: "Ожидается случайных совпадений", value: expected < 0.01 ? expected.toExponential(1) : expected.toFixed(2), color: C.blue },
          { label: "LR (сила улики)", value: fmt(LR), color: C.gold },
          { label: "Вероятность вины (апостериор)", value: (postProb * 100).toFixed(postProb < 0.1 ? 2 : 0) + "%", color: postProb > 0.5 ? C.red : C.green },
        ]);
      }
      function supr(n) { var m = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷" }; return m[n] || ""; }
      function fmt(v) { return v >= 1000 ? v.toExponential(1) : v.toFixed(v < 10 ? 1 : 0); }

      K.slider(controls, { label: "Случайное совпадение p", min: -7, max: -3, step: 0.5, value: -6, format: function (v) { return "10^" + v.toFixed(1); } }, function (v) { logp = v; draw(); });
      K.slider(controls, { label: "Размер базы N", min: 0, max: 7, step: 0.5, value: 6, format: function (v) { return "10^" + v.toFixed(1); } }, function (v) { logN = v; draw(); });
      K.slider(controls, { label: "Априорные шансы вины 1 к", min: 1, max: 7, step: 0.5, value: 6, format: function (v) { return "10^" + v.toFixed(1); } }, function (v) { logPriorDen = v; draw(); });
      K.slider(controls, { label: "Чувствительность анализа", min: 0.8, max: 1, step: 0.01, value: 0.98, format: function (v) { return (v * 100).toFixed(0) + "%"; } }, function (v) { sens = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
