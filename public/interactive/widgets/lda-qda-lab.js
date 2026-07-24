// Lesson 54: lda-qda-lab — LDA, QDA and naive Bayes on the same two clouds.
// Stretch and rotate the second class, move its prior, and watch the decision region change
// while the points stay exactly where they were.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("lda-qda-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 480;
      var OX = 60, OY = 430, S = 40;          // screen mapping: x_screen = OX + x*S
      var PB = "#fffef9";
      var NA = 60, NB = 60;
      var model = "lda", stretch = 1.0, angle = 0, priorB = 0.5, showEll = true;

      // ---- deterministic normal sample (mulberry32 + Box-Muller), fixed seed
      function rng(seed) {
        return function () {
          seed |= 0; seed = seed + 0x6D2B79F5 | 0;
          var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
          t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
          return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
      }
      function normals(n, seed) {
        var r = rng(seed), out = [], i;
        for (i = 0; i < n; i += 1) {
          var u = Math.max(r(), 1e-9), v = r();
          out.push([Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v),
                    Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v)]);
        }
        return out;
      }
      var zA = normals(NA, 54), zB = normals(NB, 154);
      var muA = [4.2, 4.8], muB = [6.0, 4.0];
      var LA = [[0.95, 0], [0.42, 0.62]];      // lower-triangular factor of class A covariance

      function factorB() {                      // stretch + rotate class B
        var c = Math.cos(angle), s = Math.sin(angle);
        var d = [[0.62 * stretch, 0], [0, 0.62 / Math.sqrt(stretch)]];
        return [[c * d[0][0], -s * d[1][1]], [s * d[0][0], c * d[1][1]]];
      }
      function pointsOf(z, mu, L) {
        var out = [], i;
        for (i = 0; i < z.length; i += 1) {
          out.push([mu[0] + L[0][0] * z[i][0] + L[0][1] * z[i][1],
                    mu[1] + L[1][0] * z[i][0] + L[1][1] * z[i][1]]);
        }
        return out;
      }
      // ---- 2x2 linear algebra
      function mean(P) {
        var mx = 0, my = 0, i;
        for (i = 0; i < P.length; i += 1) { mx += P[i][0]; my += P[i][1]; }
        return [mx / P.length, my / P.length];
      }
      function cov(P, m) {
        var a = 0, b = 0, c2 = 0, i, n = P.length;
        for (i = 0; i < n; i += 1) {
          var dx = P[i][0] - m[0], dy = P[i][1] - m[1];
          a += dx * dx; b += dx * dy; c2 += dy * dy;
        }
        return [[a / (n - 1), b / (n - 1)], [b / (n - 1), c2 / (n - 1)]];
      }
      function blend(S1, S2, w1, w2) {
        var t = w1 + w2;
        return [[(w1 * S1[0][0] + w2 * S2[0][0]) / t, (w1 * S1[0][1] + w2 * S2[0][1]) / t],
                [(w1 * S1[1][0] + w2 * S2[1][0]) / t, (w1 * S1[1][1] + w2 * S2[1][1]) / t]];
      }
      function diagOnly(S) { return [[S[0][0], 0], [0, S[1][1]]]; }
      function det(S) { return S[0][0] * S[1][1] - S[0][1] * S[1][0]; }
      function logDens(x, m, S) {
        var d = det(S);
        if (d < 1e-9) d = 1e-9;
        var dx = x[0] - m[0], dy = x[1] - m[1];
        var q = (S[1][1] * dx * dx - 2 * S[0][1] * dx * dy + S[0][0] * dy * dy) / d;
        return -0.5 * q - 0.5 * Math.log(d) - Math.log(2 * Math.PI);
      }
      function fitted() {
        var A = pointsOf(zA, muA, LA), B = pointsOf(zB, muB, factorB());
        var mA = mean(A), mB = mean(B), SA = cov(A, mA), SB = cov(B, mB);
        var pooled = blend(SA, SB, A.length - 1, B.length - 1);
        var use;
        if (model === "lda") use = [pooled, pooled];
        else if (model === "qda") use = [SA, SB];
        else use = [diagOnly(SA), diagOnly(SB)];
        return { A: A, B: B, mA: mA, mB: mB, SA: use[0], SB: use[1], eA: SA, eB: SB };
      }
      function disc(f, x) {   // >0 => class B (rare class)
        return logDens(x, f.mB, f.SB) + Math.log(priorB)
             - logDens(x, f.mA, f.SA) - Math.log(1 - priorB);
      }
      function params() { return model === "qda" ? 10 : (model === "nb" ? 8 : 7); }

      K.hint(root, "Одни и те же точки, три порождающие модели. LDA заставляет оба класса делить общую матрицу разброса и всегда даёт прямую границу. QDA даёт каждому классу свою матрицу — граница гнётся. Naive Bayes оставляет только диагональ, то есть объявляет признаки условно независимыми, и эллипсы становятся строго по осям. Ползунок prior не двигает ни одной точки, но перекраивает области решения.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Заливка — область, где модель выбирает соответствующий класс; жирная линия — граница. Обратите внимание: при сильном растяжении и повороте красного класса прямая LDA уже не может обойти облако, а QDA огибает его дугой. Уменьшая prior редкого класса, вы сжимаете его область, не трогая данные, — и точность на этих же точках падает, потому что мы требуем более убедительного совпадения.");
      var cs = K.makeCanvas(stage, W, H, { label: "Два облака, эллипсы моделей и область решения", onResize: draw, drag: false });

      function m2s(x, y) { return [OX + x * S, OY - y * S]; }

      function drawEllipse(ctx, m, Sm, color, k) {
        var tr = Sm[0][0] + Sm[1][1], dt = det(Sm);
        var disc2 = Math.max(tr * tr / 4 - dt, 0);
        var l1 = tr / 2 + Math.sqrt(disc2), l2 = tr / 2 - Math.sqrt(disc2);
        var vx, vy;
        if (Math.abs(Sm[0][1]) > 1e-9) { vx = l1 - Sm[1][1]; vy = Sm[0][1]; }
        else if (Sm[0][0] >= Sm[1][1]) { vx = 1; vy = 0; }
        else { vx = 0; vy = 1; }            // diagonal Σ, taller than wide (naive Bayes case)
        var nrm = Math.hypot(vx, vy); vx /= nrm; vy /= nrm;
        var th = Math.atan2(vy, vx);
        ctx.save();
        var c = m2s(m[0], m[1]);
        ctx.translate(c[0], c[1]); ctx.rotate(-th);
        ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.ellipse(0, 0, k * Math.sqrt(Math.max(l1, 1e-9)) * S, k * Math.sqrt(Math.max(l2, 1e-9)) * S, 0, 0, 7);
        ctx.stroke();
        ctx.restore();
      }

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var f = fitted();

        // decision regions on a coarse grid
        var step = 6;
        for (var px = OX; px < W - 10; px += step) {
          for (var py = 40; py < OY; py += step) {
            var mx = (px - OX) / S, my = (OY - py) / S;
            var v = disc(f, [mx, my]);
            ctx.fillStyle = v > 0 ? "rgba(185,74,59,0.12)" : "rgba(49,95,140,0.10)";
            if (Math.abs(v) < 0.35) ctx.fillStyle = "rgba(23,25,21,0.55)";
            ctx.fillRect(px, py, step, step);
          }
        }
        // axes
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var o = m2s(0, 0), xe = m2s(13, 0), ye = m2s(0, 9.6);
        ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(xe[0], xe[1]);
        ctx.moveTo(o[0], o[1]); ctx.lineTo(ye[0], ye[1]); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("признак 1", (OX + xe[0]) / 2, OY + 26);
        ctx.save(); ctx.translate(OX - 34, (OY + 60) / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("признак 2", 0, 0); ctx.restore();

        // ellipses of the model actually used
        if (showEll) {
          drawEllipse(ctx, f.mA, f.SA, C.blue, 1.6);
          drawEllipse(ctx, f.mB, f.SB, C.red, 1.6);
        }
        // points
        var errA = 0, errB = 0;
        for (i = 0; i < f.A.length; i += 1) {
          var pa = m2s(f.A[i][0], f.A[i][1]);
          var wrongA = disc(f, f.A[i]) > 0; if (wrongA) errA += 1;
          ctx.fillStyle = C.blue; ctx.globalAlpha = wrongA ? 1 : 0.8;
          ctx.beginPath(); ctx.arc(pa[0], pa[1], wrongA ? 6 : 4.5, 0, 7); ctx.fill();
          if (wrongA) { ctx.strokeStyle = "#171915"; ctx.lineWidth = 1.2; ctx.stroke(); }
          ctx.globalAlpha = 1;
        }
        for (i = 0; i < f.B.length; i += 1) {
          var pb = m2s(f.B[i][0], f.B[i][1]);
          var wrongB = disc(f, f.B[i]) <= 0; if (wrongB) errB += 1;
          ctx.fillStyle = C.red; ctx.globalAlpha = wrongB ? 1 : 0.8;
          ctx.beginPath(); ctx.arc(pb[0], pb[1], wrongB ? 6 : 4.5, 0, 7); ctx.fill();
          if (wrongB) { ctx.strokeStyle = "#171915"; ctx.lineWidth = 1.2; ctx.stroke(); }
          ctx.globalAlpha = 1;
        }
        // centres
        [[f.mA, C.blue], [f.mB, C.red]].forEach(function (pair) {
          var c = m2s(pair[0][0], pair[0][1]);
          ctx.strokeStyle = pair[1]; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(c[0] - 8, c[1]); ctx.lineTo(c[0] + 8, c[1]);
          ctx.moveTo(c[0], c[1] - 8); ctx.lineTo(c[0], c[1] + 8); ctx.stroke();
        });

        var acc = (NA + NB - errA - errB) / (NA + NB);
        var recall = (NB - errB) / NB;
        var names = { lda: "LDA (общая Σ)", qda: "QDA (своя Σ каждому)", nb: "naive Bayes (диагональ)" };
        output.set([
          { label: "Модель", value: names[model], color: C.gold },
          { label: "Точность на этих точках", value: (acc * 100).toFixed(1) + "%", color: C.blue },
          { label: "Поймано редкого класса", value: (recall * 100).toFixed(0) + "% (" + (NB - errB) + " из " + NB + ")", color: C.red },
          { label: "Параметров модели", value: String(params()), color: C.gold }
        ]);
      }

      K.segmented(controls, {
        label: "Порождающая модель", value: 0,
        options: [{ label: "LDA", value: 0 }, { label: "QDA", value: 1 }, { label: "naive Bayes", value: 2 }]
      }, function (v) { model = v === 1 ? "qda" : (v === 2 ? "nb" : "lda"); draw(); });
      K.slider(controls, {
        label: "Растяжение красного класса", min: 0.6, max: 3.2, step: 0.05, value: 1,
        format: function (v) { return "×" + Number(v).toFixed(2); }
      }, function (v) { stretch = Number(v); draw(); });
      K.slider(controls, {
        label: "Поворот красного класса", min: -90, max: 90, step: 1, value: 0,
        format: function (v) { return v + "°"; }
      }, function (v) { angle = Number(v) * Math.PI / 180; draw(); });
      K.slider(controls, {
        label: "Prior редкого класса", min: 0.02, max: 0.5, step: 0.01, value: 0.5,
        format: function (v) { return (Number(v) * 100).toFixed(0) + "%"; }
      }, function (v) { priorB = Number(v); draw(); });
      K.segmented(controls, {
        label: "Эллипсы модели", value: 1,
        options: [{ label: "скрыть", value: 0 }, { label: "показать", value: 1 }]
      }, function (v) { showEll = v === 1; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
