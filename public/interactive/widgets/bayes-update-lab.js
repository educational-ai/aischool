// Lesson 47: bayes-update-lab — prior times likelihood equals posterior, shown as proper densities.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("bayes-update-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var pa = 2, pb = 2, h = 3, t = 1;   // prior Beta(pa,pb), data h successes, t failures
      var GRID = 400;

      K.hint(
        root,
        "Байесовское обновление: prior × правдоподобие = апостериор. Двигайте prior (псевдо-успехи и псевдо-неудачи) и данные (успехи и неудачи). Все три кривые — настоящие плотности, площадь каждой равна единице. Сильный prior почти не сдвигается данными, слабый — сильно. Заштрихован 95% credible-интервал.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Prior Beta(α, β) кодирует знание до данных; параметры — как псевдосчётчики. Данные прибавляют успехи к α и неудачи к β: апостериор = Beta(α+h, β+t). Credible-интервал — область, где параметр лежит с апостериорной вероятностью 0,95 (при принятой модели и prior).",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Prior, правдоподобие и апостериор", onResize: draw, drag: false });

      function betaDensity(a, b) {
        var d = new Float64Array(GRID + 1), dx = 1 / GRID, area = 0;
        for (var i = 0; i <= GRID; i += 1) {
          var x = Math.min(1 - 1e-9, Math.max(1e-9, i * dx));
          d[i] = Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x));
          area += d[i] * dx;
        }
        for (var j = 0; j <= GRID; j += 1) d[j] /= area;
        return d;
      }
      function credible(dens) {
        var dx = 1 / GRID, c = 0, lo = 0, hi = 1, foundLo = false;
        for (var i = 0; i <= GRID; i += 1) {
          c += dens[i] * dx;
          if (!foundLo && c >= 0.025) { lo = i * dx; foundLo = true; }
          if (c >= 0.975) { hi = i * dx; break; }
        }
        return [lo, hi];
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        var prior = betaDensity(pa, pb);
        var like = betaDensity(h + 1, t + 1);
        var post = betaDensity(pa + h, pb + t);
        var ci = credible(post);
        var pm = (pa + h) / (pa + pb + h + t);

        var gx = 55, gy = 45, gw = W - 90, gh = 340;
        var mx = 0;
        for (var i = 0; i <= GRID; i += 1) mx = Math.max(mx, prior[i], like[i], post[i]);
        mx *= 1.08;
        function X(x) { return gx + x * gw; }
        function Y(v) { return gy + gh - v / mx * gh; }
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, gy, gw, gh);

        // credible shading
        ctx.fillStyle = "rgba(49,95,140,0.10)";
        ctx.fillRect(X(ci[0]), gy, X(ci[1]) - X(ci[0]), gh);

        function curve(d, col, lw, dash) {
          ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash || []);
          ctx.beginPath();
          for (var i = 0; i <= GRID; i += 1) { var px = X(i / GRID), py = Y(d[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
          ctx.stroke(); ctx.setLineDash([]);
        }
        curve(prior, C.blue, 2.0);
        curve(like, C.gold, 1.8, [6, 4]);
        // posterior fill
        ctx.fillStyle = "rgba(185,74,59,0.10)"; ctx.beginPath(); ctx.moveTo(X(0), Y(0));
        for (var k = 0; k <= GRID; k += 1) ctx.lineTo(X(k / GRID), Y(post[k]));
        ctx.lineTo(X(1), Y(0)); ctx.closePath(); ctx.fill();
        curve(post, C.red, 2.6);
        // mean line
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.0; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(pm), gy); ctx.lineTo(X(pm), gy + gh); ctx.stroke(); ctx.setLineDash([]);

        // x ticks
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        for (var xt = 0; xt <= 5; xt += 1) ctx.fillText((xt / 5).toFixed(1), X(xt / 5), gy + gh + 16);
        ctx.fillStyle = C.ink; ctx.fillText("параметр θ", gx + gw / 2, gy + gh + 32);
        // legend
        ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = C.blue; ctx.fillText("— prior Beta(" + pa + "," + pb + ")", gx + 12, gy + 18);
        ctx.fillStyle = C.gold; ctx.fillText("– – правдоподобие " + h + "/" + (h + t), gx + 12, gy + 36);
        ctx.fillStyle = C.red; ctx.fillText("— posterior Beta(" + (pa + h) + "," + (pb + t) + ")", gx + 12, gy + 54);

        output.set([
          { label: "Апостериорное среднее", value: pm.toFixed(3), color: C.red },
          { label: "95% credible interval", value: "[" + ci[0].toFixed(2) + ", " + ci[1].toFixed(2) + "]", color: C.blue },
          { label: "Сила prior (α+β) vs данные (n)", value: (pa + pb) + " против " + (h + t), color: C.gold },
        ]);
      }

      K.slider(controls, { label: "prior: псевдо-успехи α", min: 1, max: 50, step: 1, value: 2, format: function (v) { return String(v); } }, function (v) { pa = v; draw(); });
      K.slider(controls, { label: "prior: псевдо-неудачи β", min: 1, max: 50, step: 1, value: 2, format: function (v) { return String(v); } }, function (v) { pb = v; draw(); });
      K.slider(controls, { label: "данные: успехи h", min: 0, max: 100, step: 1, value: 3, format: function (v) { return String(v); } }, function (v) { h = v; draw(); });
      K.slider(controls, { label: "данные: неудачи t", min: 0, max: 100, step: 1, value: 1, format: function (v) { return String(v); } }, function (v) { t = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
