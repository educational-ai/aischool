// Lesson 50: basis-lab — fit a chosen basis (polynomials / radial bells / cyclic sines)
// to a fixed bimodal target, watch training fit vs behaviour outside the data.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("basis-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var OX = 64, OY = 410, SX = 58, SY = 34, Y0 = -2;
      var PB = "#fffef9";
      var XMAX = 13, XDATA = 10;              // data live in [0,10], [10,13] is extrapolation
      var basisType = 0, m = 6;               // 0 poly, 1 rbf, 2 cyclic

      // fixed bimodal target sampled from two gaussian bumps (morning + evening peak)
      var DX = [], DY = [];
      var NZ = [0.4, -0.3, 0.5, -0.2, 0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, -0.6, 0.2, -0.35, 0.5, -0.25, 0.4, -0.3];
      (function build() {
        for (var i = 0; i < 18; i += 1) {
          var x = 0.4 + i * 0.55;
          var f = 5.2 * Math.exp(-((x - 3.3) * (x - 3.3)) / 2.0) + 8.4 * Math.exp(-((x - 7.0) * (x - 7.0)) / 2.88);
          DX.push(x); DY.push(f + NZ[i]);
        }
      })();

      function u(x) { return x / (XDATA / 2) - 1; }   // map [0,10] -> [-1,1]
      function phi(x) {
        var uu = u(x), v = [], k;
        if (basisType === 0) { for (k = 0; k < m; k += 1) v.push(Math.pow(uu, k)); }
        else if (basisType === 1) {
          v.push(1);
          var cnt = m - 1, wdt = 2.0 / Math.max(2, cnt);
          for (k = 0; k < cnt; k += 1) { var c = -1 + (k + 0.5) * (2.0 / cnt); v.push(Math.exp(-((uu - c) * (uu - c)) / (2 * wdt * wdt))); }
        } else {
          v.push(1);
          for (k = 1; v.length < m; k += 1) { v.push(Math.sin(Math.PI * k * uu)); if (v.length < m) v.push(Math.cos(Math.PI * k * uu)); }
        }
        return v;
      }

      // solve (ΦᵀΦ + λI) w = Φᵀy with tiny ridge for stability, Gaussian elimination
      function solve(A, b) {
        var n = b.length, i, j, k;
        for (i = 0; i < n; i += 1) {
          var p = i; for (j = i + 1; j < n; j += 1) if (Math.abs(A[j][i]) > Math.abs(A[p][i])) p = j;
          var t = A[i]; A[i] = A[p]; A[p] = t; var tb = b[i]; b[i] = b[p]; b[p] = tb;
          if (Math.abs(A[i][i]) < 1e-12) continue;
          for (j = i + 1; j < n; j += 1) {
            var f = A[j][i] / A[i][i];
            for (k = i; k < n; k += 1) A[j][k] -= f * A[i][k];
            b[j] -= f * b[i];
          }
        }
        var w = new Array(n).fill(0);
        for (i = n - 1; i >= 0; i -= 1) {
          var s = b[i]; for (j = i + 1; j < n; j += 1) s -= A[i][j] * w[j];
          w[i] = Math.abs(A[i][i]) < 1e-12 ? 0 : s / A[i][i];
        }
        return w;
      }
      function fit() {
        var d = DX.length, P = [], i, j;
        for (i = 0; i < d; i += 1) P.push(phi(DX[i]));
        var A = [], b = [];
        for (i = 0; i < m; i += 1) { A.push(new Array(m).fill(0)); b.push(0); }
        for (i = 0; i < m; i += 1) {
          for (j = 0; j < m; j += 1) { var s = 0; for (var r = 0; r < d; r += 1) s += P[r][i] * P[r][j]; A[i][j] = s + (i === j ? 1e-6 : 0); }
          var sb = 0; for (var q = 0; q < d; q += 1) sb += P[q][i] * DY[q]; b[i] = sb;
        }
        var w = solve(A, b);
        var ssr = 0; for (i = 0; i < d; i += 1) { var e = DY[i] - dot(P[i], w); ssr += e * e; }
        return { w: w, rmse: Math.sqrt(ssr / d) };
      }
      function dot(a, w) { var s = 0; for (var i = 0; i < a.length; i += 1) s += a[i] * w[i]; return s; }
      function evalx(x, w) { return dot(phi(x), w); }

      K.hint(root, "Одна и та же цель — два пика, как утренний и вечерний прокат велосипедов. Выберите семейство признаков и их число, и модель подберёт вес каждого. Следите за двумя вещами: насколько кривая описывает данные (серая зона) и что она вытворяет ПРАВЕЕ данных, где ей не на что опереться.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Полиномы — глобальные: правишь один вес — меняется вся кривая, а за краем данных степенной хвост улетает. Колокола локальны: каждый отвечает за свой участок. Синусы навязывают периодичность. Больше признаков → лучше на данных, но опаснее вне них.");
      var cs = K.makeCanvas(stage, W, H, { label: "Цель, подобранная кривая и зона экстраполяции", onResize: draw });

      function m2s(x, y) { return [OX + x * SX, OY - (y - Y0) * SY]; }
      function draw() {
        var ctx = cs.ctx; ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var f = fit();
        // extrapolation band
        var ed0 = m2s(XDATA, 11)[0], ey0 = m2s(0, 11)[1], eyb = m2s(0, Y0)[1];
        ctx.fillStyle = "rgba(185,74,59,0.06)"; ctx.fillRect(ed0, ey0, m2s(XMAX, 0)[0] - ed0, eyb - ey0);
        ctx.fillStyle = "rgba(56,115,93,0.05)"; ctx.fillRect(m2s(0, 0)[0], ey0, ed0 - m2s(0, 0)[0], eyb - ey0);
        // axes
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var o = m2s(0, 0), xe = m2s(XMAX, 0), ye = m2s(0, 11);
        ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(xe[0], xe[1]); ctx.moveTo(m2s(0, Y0)[0], m2s(0, Y0)[1]); ctx.lineTo(ye[0], ye[1]); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var t = 2; t <= 12; t += 2) { var p = m2s(t, 0); ctx.fillText(String(t), p[0], p[1] + 16); }
        ctx.fillText("x", xe[0] + 12, xe[1] + 4);
        ctx.textAlign = "left"; ctx.fillStyle = "rgba(185,74,59,0.75)";
        ctx.fillText("вне данных →", ed0 + 8, ey0 + 16);
        // fitted curve
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.6; ctx.beginPath();
        var started = false;
        for (var x = 0; x <= XMAX; x += 0.06) {
          var yv = evalx(x, f.w); yv = Math.max(Y0 - 1, Math.min(12, yv));
          var s = m2s(x, yv); if (!started) { ctx.moveTo(s[0], s[1]); started = true; } else ctx.lineTo(s[0], s[1]);
        }
        ctx.stroke();
        // data points
        for (var i = 0; i < DX.length; i += 1) { var sp = m2s(DX[i], DY[i]); ctx.fillStyle = C.blue; ctx.beginPath(); ctx.arc(sp[0], sp[1], 5, 0, 7); ctx.fill(); ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.stroke(); }
        var names = ["полиномы", "колокола (RBF)", "синусы"];
        var xtra = evalx(12, f.w);
        output.set([
          { label: "Семейство", value: names[basisType], color: C.red },
          { label: "Параметров", value: String(m), color: C.blue },
          { label: "RMSE на данных", value: f.rmse.toFixed(2), color: C.green },
          { label: "Прогноз в x=12 (вне данных)", value: xtra.toFixed(1), color: C.gold }
        ]);
      }

      K.segmented(controls, { label: "Семейство признаков", value: 0, options: [{ label: "полиномы", value: 0 }, { label: "колокола (RBF)", value: 1 }, { label: "синусы", value: 2 }] }, function (v) { basisType = v; draw(); });
      K.slider(controls, { label: "Число признаков", min: 2, max: 13, step: 1, value: 6, format: function (v) { return String(v); } }, function (v) { m = v; draw(); });
      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
