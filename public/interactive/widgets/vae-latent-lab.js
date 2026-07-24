// Lesson 85: vae-latent-lab — точная (аналитически решаемая) модель спора реконструкции и KL.
// Объекты с «истинными» кодами t_i, декодер g(z)=z, посты q(z|x_i)=N(mu_i, sigma^2).
// Минимум (mu-t)^2 + sigma^2 + beta*KL даёт mu = 2t/(2+beta), sigma^2 = beta/(beta+2).
// Двигая beta, видно всё сразу: сжатие кодов, расширение облаков, падение узнаваемости,
// коллапс при большом beta и кривую «скорость — искажение».
(function () {
  "use strict";

  function erf(x) {
    // Abramowitz & Stegun 7.1.26, |eps| < 1.5e-7
    var s = x < 0 ? -1 : 1;
    var a = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * a);
    var y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  }
  function Phi(x) { return 0.5 * (1 + erf(x / Math.SQRT2)); }
  function npdf(x, m, s) { return Math.exp(-0.5 * ((x - m) / s) * ((x - m) / s)) / (s * Math.sqrt(2 * Math.PI)); }

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("vae-latent-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 520;
      var beta = 1.0, spread = 1.6, nObj = 5;
      var SIG_MIN = 0.02;

      function model(b, sp, n) {
        var shrink = 2 / (2 + b);
        var sigma = Math.max(Math.sqrt(b / (b + 2)), SIG_MIN);
        var t = [], mu = [], i;
        for (i = 0; i < n; i += 1) {
          var ti = sp * (i - (n - 1) / 2);
          t.push(ti); mu.push(shrink * ti);
        }
        var dist = 0, rate = 0, acc = 0;
        for (i = 0; i < n; i += 1) {
          dist += (mu[i] - t[i]) * (mu[i] - t[i]) + sigma * sigma;
          rate += 0.5 * (mu[i] * mu[i] + sigma * sigma - Math.log(sigma * sigma) - 1);
          var lo = i === 0 ? -Infinity : 0.5 * (t[i - 1] + t[i]);
          var hi = i === n - 1 ? Infinity : 0.5 * (t[i] + t[i + 1]);
          var pl = lo === -Infinity ? 0 : Phi((lo - mu[i]) / sigma);
          var ph = hi === Infinity ? 1 : Phi((hi - mu[i]) / sigma);
          acc += ph - pl;
        }
        return {
          shrink: shrink, sigma: sigma, t: t, mu: mu,
          dist: dist / n, rate: rate / n, acc: acc / n
        };
      }

      K.hint(root, "Модель решается на бумаге: если декодер — тождественное отображение, а объект хочет попасть в свою точку t, то оптимум даёт mu = 2t/(2+β) и σ² = β/(β+2). Двигайте β и смотрите, как коды сползают к нулю, облака расширяются, а узнаваемость объектов падает. При β = 0 получается обычный автокодировщик (σ = 0, коды где угодно), при большом β — коллапс: все q(z|x) совпали с prior.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Сверху: prior N(0,1) серым, цветные колокола — апостериорные облака отдельных объектов, штрихи внизу — их «истинные» коды t. Снизу: кривая «скорость — искажение», которую заметает β; красная точка — текущий режим. Узнаваемость — доля жребиев z, попадающих ближе к своему t, чем к чужому; при коллапсе она падает до 1/K.");
      var cs = K.makeCanvas(stage, W, H, { label: "Апостериорные облака, prior и кривая скорость — искажение", onResize: draw, drag: false });

      K.slider(controls, { label: "β — вес KL", min: 0, max: 8, step: 0.05, value: beta, format: function (v) { return v.toFixed(2); } }, function (v) { beta = v; draw(); });
      K.slider(controls, { label: "разнос объектов", min: 0.6, max: 3, step: 0.05, value: spread, format: function (v) { return v.toFixed(2); } }, function (v) { spread = v; draw(); });
      K.segmented(controls, {
        label: "число объектов",
        value: 5,
        options: [{ value: 3, label: "3" }, { value: 5, label: "5" }, { value: 7, label: "7" }]
      }, function (v) { nObj = v; draw(); });

      var PAL = [C.blue, C.red, C.green, C.gold, C.violet || "#6f5a8f", C.blue, C.red];

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var m = model(beta, spread, nObj);

        // ---------- panel 1: densities
        var P1 = { x0: 60, x1: W - 30, y0: 30, y1: 258 };
        var zmax = Math.max(3.2, spread * (nObj - 1) / 2 + 2.4);
        function zx(z) { return P1.x0 + (z + zmax) / (2 * zmax) * (P1.x1 - P1.x0); }
        var pmax = 1 / (Math.max(m.sigma, 0.16) * Math.sqrt(2 * Math.PI));
        pmax = Math.max(pmax, 0.42);
        function py(p) { return P1.y1 - p / pmax * (P1.y1 - P1.y0); }

        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(P1.x0, P1.y1); ctx.lineTo(P1.x1, P1.y1); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var g = -Math.floor(zmax); g <= Math.floor(zmax); g += 1) {
          ctx.fillText(String(g), zx(g), P1.y1 + 16);
          ctx.strokeStyle = "rgba(222,221,212,0.7)";
          ctx.beginPath(); ctx.moveTo(zx(g), P1.y0); ctx.lineTo(zx(g), P1.y1); ctx.stroke();
        }
        ctx.fillText("z", P1.x1 - 6, P1.y1 + 32);

        // prior
        ctx.strokeStyle = "rgba(110,114,106,0.85)"; ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (var k = 0; k <= 240; k += 1) {
          var z = -zmax + 2 * zmax * k / 240;
          var pv = npdf(z, 0, 1);
          if (k === 0) ctx.moveTo(zx(z), py(pv)); else ctx.lineTo(zx(z), py(pv));
        }
        ctx.stroke(); ctx.setLineDash([]);

        // posteriors
        for (var i = 0; i < nObj; i += 1) {
          var col = PAL[i % PAL.length];
          ctx.strokeStyle = col; ctx.lineWidth = 2;
          ctx.beginPath();
          for (var j = 0; j <= 240; j += 1) {
            var zz = -zmax + 2 * zmax * j / 240;
            var q = npdf(zz, m.mu[i], m.sigma);
            var yy = Math.max(py(Math.min(q, pmax)), P1.y0 - 2);
            if (j === 0) ctx.moveTo(zx(zz), yy); else ctx.lineTo(zx(zz), yy);
          }
          ctx.stroke();
          // target tick and mu marker
          ctx.strokeStyle = col; ctx.lineWidth = 2.4;
          ctx.beginPath(); ctx.moveTo(zx(m.t[i]), P1.y1 + 2); ctx.lineTo(zx(m.t[i]), P1.y1 + 12); ctx.stroke();
          ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(zx(m.mu[i]), P1.y1, 4.5, 0, 7); ctx.fill();
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("штрих — цель t, кружок — код μ", P1.x0, P1.y1 + 32);
        ctx.textAlign = "right";
        ctx.fillText("штриховая серая — prior N(0,1)", P1.x1, P1.y0 + 4);
        ctx.font = "12px PT Sans, sans-serif";

        // ---------- panel 2: rate-distortion curve
        var P2 = { x0: 60, x1: W - 30, y0: 320, y1: H - 40 };
        var pts = [], b;
        for (var s2 = 0; s2 <= 160; s2 += 1) {
          b = 0.02 + (8 - 0.02) * s2 / 160;
          var mm = model(b, spread, nObj);
          pts.push({ b: b, r: mm.rate, d: mm.dist });
        }
        var rmax = 0, dmax = 0;
        for (var q2 = 0; q2 < pts.length; q2 += 1) {
          if (pts[q2].r > rmax) rmax = pts[q2].r;
          if (pts[q2].d > dmax) dmax = pts[q2].d;
        }
        rmax = Math.max(rmax, m.rate) * 1.06;
        dmax = Math.max(dmax, m.dist) * 1.08;
        function rx(r) { return P2.x0 + Math.min(r / rmax, 1) * (P2.x1 - P2.x0); }
        function dy(d) { return P2.y1 - Math.min(d / dmax, 1) * (P2.y1 - P2.y0); }

        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(P2.x0, P2.y0); ctx.lineTo(P2.x0, P2.y1); ctx.lineTo(P2.x1, P2.y1);
        ctx.stroke();
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (var u = 0; u < pts.length; u += 1) {
          var X = rx(pts[u].r), Y = dy(pts[u].d);
          if (u === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        }
        ctx.stroke();
        ctx.fillStyle = C.red;
        ctx.beginPath(); ctx.arc(rx(m.rate), dy(m.dist), 7, 0, 7); ctx.fill();
        ctx.strokeStyle = "#fffef9"; ctx.lineWidth = 1.4; ctx.stroke();

        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11.5px PT Sans, sans-serif";
        ctx.fillText("rate: KL на объект, нат", (P2.x0 + P2.x1) / 2, P2.y1 + 26);
        ctx.save();
        ctx.translate(P2.x0 - 40, (P2.y0 + P2.y1) / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("distortion", 0, 0); ctx.restore();
        ctx.textAlign = "left";
        ctx.fillText("большое β: код пуст, ошибка максимальна", P2.x0 + 8, P2.y0 + 4);
        ctx.textAlign = "right";
        ctx.fillText("малое β: точная реконструкция дорогим кодом", P2.x1 - 4, P2.y1 - 8);
        ctx.font = "12px PT Sans, sans-serif";

        var collapsed = m.rate < 0.01;
        out.set([
          { label: "сжатие кода μ/t = 2/(2+β)", value: m.shrink.toFixed(3), color: C.blue },
          { label: "ширина облака σ", value: m.sigma.toFixed(3), color: C.gold },
          { label: "rate: KL на объект, нат", value: m.rate.toFixed(3), color: C.green },
          { label: "distortion: ошибка на объект", value: m.dist.toFixed(3), color: C.red },
          { label: "узнаваемость объекта", value: (100 * m.acc).toFixed(1) + "% (случайно " + (100 / nObj).toFixed(1) + "%)", color: collapsed ? C.red : C.blue }
        ]);
      }

      draw();
      return function destroy() { if (cs && cs.destroy) cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
