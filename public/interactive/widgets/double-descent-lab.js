// Lesson 59: double-descent-lab — двигайте число признаков p через порог интерполяции p = n
// и смотрите: ровно у порога прогноз взрывается, а дальше, в переопределённом режиме,
// решение минимальной нормы снова становится осмысленным.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("double-descent-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var N = 40;      // обучающих объектов
      var D = 8;       // размерность объекта
      var NTE = 240;   // тестовых объектов
      var PGRID = [2, 4, 6, 9, 13, 18, 24, 30, 34, 37, 39, 40, 41, 43, 46, 52, 62, 80, 110, 160, 240, 360, 550, 800, 1200];
      var PMAX = PGRID[PGRID.length - 1];

      var pIndex = 6, noise = 0.30, lam = 0, seed = 11;
      var Xtr = [], Xte = [], ftr = [], fte = [], eps = [], Wf = [], bf = [];

      // ---- детерминированный генератор: у всех учеников одна и та же картинка
      function mulberry(a) {
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          var t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function gauss(rnd) {
        var u = Math.max(1e-9, rnd()), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      function build() {
        var rnd = mulberry(seed), i, j, k;
        // учитель: маленькая сеть из шести ReLU-нейронов
        var A = [], bT = [], cT = [];
        for (j = 0; j < D; j += 1) {
          A.push([]);
          for (k = 0; k < 6; k += 1) A[j].push(gauss(rnd));
        }
        for (k = 0; k < 6; k += 1) { bT.push(gauss(rnd)); cT.push(gauss(rnd)); }
        function teacher(x) {
          var s = 0, kk, jj;
          for (kk = 0; kk < 6; kk += 1) {
            var h = bT[kk];
            for (jj = 0; jj < D; jj += 1) h += x[jj] * A[jj][kk];
            if (h > 0) s += h * cT[kk];
          }
          return s;
        }
        Xtr = []; Xte = []; ftr = []; fte = []; eps = [];
        for (i = 0; i < N; i += 1) {
          var x = [];
          for (j = 0; j < D; j += 1) x.push(gauss(rnd));
          Xtr.push(x); ftr.push(teacher(x)); eps.push(gauss(rnd));
        }
        for (i = 0; i < NTE; i += 1) {
          var z = [];
          for (j = 0; j < D; j += 1) z.push(gauss(rnd));
          Xte.push(z); fte.push(teacher(z));
        }
        // приводим ответ к единичному разбросу: шум меток читается в тех же единицах
        var m = 0; for (i = 0; i < N; i += 1) m += ftr[i]; m /= N;
        var v2 = 0; for (i = 0; i < N; i += 1) v2 += (ftr[i] - m) * (ftr[i] - m);
        var sd = Math.sqrt(v2 / N) || 1;
        for (i = 0; i < N; i += 1) ftr[i] = (ftr[i] - m) / sd;
        for (i = 0; i < NTE; i += 1) fte[i] = (fte[i] - m) / sd;
        // банк случайных признаков: первые p штук и есть модель размера p
        Wf = []; bf = [];
        for (j = 0; j < D; j += 1) {
          Wf.push([]);
          for (k = 0; k < PMAX; k += 1) Wf[j].push(gauss(rnd) / Math.sqrt(D));
        }
        for (k = 0; k < PMAX; k += 1) bf.push(-1 + 2 * rnd());
      }

      function features(x, pp) {
        var z = new Array(pp), k, j;
        for (k = 0; k < pp; k += 1) {
          var h = bf[k];
          for (j = 0; j < D; j += 1) h += x[j] * Wf[j][k];
          z[k] = h > 0 ? h : 0;
        }
        return z;
      }
      function labels() {
        var y = [], i;
        for (i = 0; i < N; i += 1) y.push(ftr[i] + noise * eps[i]);
        return y;
      }

      // ---- симметричная система A z = b, метод Гаусса с выбором главного элемента
      function solve(A, b) {
        var m = b.length, i, j, k;
        for (i = 0; i < m; i += 1) {
          var piv = i;
          for (k = i + 1; k < m; k += 1) if (Math.abs(A[k][i]) > Math.abs(A[piv][i])) piv = k;
          var tr = A[i]; A[i] = A[piv]; A[piv] = tr;
          var tb = b[i]; b[i] = b[piv]; b[piv] = tb;
          var dd = A[i][i];
          if (Math.abs(dd) < 1e-12) dd = dd >= 0 ? 1e-12 : -1e-12;
          for (j = i; j < m; j += 1) A[i][j] /= dd;
          b[i] /= dd;
          for (k = i + 1; k < m; k += 1) {
            var f = A[k][i];
            if (f === 0) continue;
            for (j = i; j < m; j += 1) A[k][j] -= f * A[i][j];
            b[k] -= f * b[i];
          }
        }
        var out = new Array(m).fill(0);
        for (i = m - 1; i >= 0; i -= 1) {
          var s = b[i];
          for (j = i + 1; j < m; j += 1) s -= A[i][j] * out[j];
          out[i] = s;
        }
        return out;
      }

      // ---- обучение: p <= n — обычный МНК, p > n — решение минимальной нормы
      function fit(pp, y, lambda, wantPred) {
        var i, j, k, Z = [];
        for (i = 0; i < N; i += 1) Z.push(features(Xtr[i], pp));
        var ridge = lambda + 1e-11;
        var w = new Array(pp).fill(0);
        if (pp <= N) {
          var A = [], b = new Array(pp).fill(0);
          for (j = 0; j < pp; j += 1) A.push(new Array(pp).fill(0));
          for (j = 0; j < pp; j += 1) {
            for (k = j; k < pp; k += 1) {
              var s = 0;
              for (i = 0; i < N; i += 1) s += Z[i][j] * Z[i][k];
              A[j][k] = s; A[k][j] = s;
            }
            A[j][j] += ridge;
            var t = 0;
            for (i = 0; i < N; i += 1) t += Z[i][j] * y[i];
            b[j] = t;
          }
          w = solve(A, b);
        } else {
          var G = [];
          for (i = 0; i < N; i += 1) G.push(new Array(N).fill(0));
          for (i = 0; i < N; i += 1) {
            for (k = i; k < N; k += 1) {
              var g = 0;
              for (j = 0; j < pp; j += 1) g += Z[i][j] * Z[k][j];
              G[i][k] = g; G[k][i] = g;
            }
            G[i][i] += ridge;
          }
          var alpha = solve(G, y.slice());
          for (j = 0; j < pp; j += 1) {
            var acc = 0;
            for (i = 0; i < N; i += 1) acc += alpha[i] * Z[i][j];
            w[j] = acc;
          }
        }
        var nrm = 0, trErr = 0;
        for (j = 0; j < pp; j += 1) nrm += w[j] * w[j];
        for (i = 0; i < N; i += 1) {
          var pr = 0;
          for (j = 0; j < pp; j += 1) pr += Z[i][j] * w[j];
          trErr += (pr - y[i]) * (pr - y[i]);
        }
        var teErr = 0, preds = wantPred ? [] : null;
        for (i = 0; i < NTE; i += 1) {
          var f = features(Xte[i], pp), q = 0;
          for (j = 0; j < pp; j += 1) q += f[j] * w[j];
          teErr += (q - fte[i]) * (q - fte[i]);
          if (wantPred) preds.push(q);
        }
        return {
          norm: Math.sqrt(nrm), preds: preds,
          train: Math.sqrt(trErr / N), test: Math.sqrt(teErr / NTE)
        };
      }

      K.hint(root, "Обучающих объектов ровно " + N + ", каждый описан " + D + " числами; ответ порождён небольшой сетью-учителем, к нему добавлен шум. Модель — линейная комбинация p случайных ReLU-признаков. Ведите ползунок p слева направо: сначала ошибка на тесте падает, у p = n она взлетает — единственный интерполятор вынужден повторить и шум, — а дальше интерполяторов становится много, алгоритм берёт из них решение минимальной нормы, и ошибка снова падает. Ridge убирает пик целиком.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева: прогноз против истины на тестовых объектах; чем плотнее облако прижато к диагонали, тем лучше модель. Справа: ошибка на обучении (синяя) и на тесте (красная) сразу для всех p. Золотая вертикаль — порог интерполяции p = n = " + N + ", золотая точка — текущее положение ползунка.");
      var cs = K.makeCanvas(stage, W, H, { label: "Прогноз против истины и кривая ошибки по числу признаков", onResize: draw, drag: false });

      build();

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var y = labels();
        var p = PGRID[pIndex];
        var cur = fit(p, y, lam, true);
        var tests = [], trains = [];
        for (i = 0; i < PGRID.length; i += 1) {
          var f = PGRID[i] === p ? cur : fit(PGRID[i], y, lam, false);
          tests.push(f.test); trains.push(f.train);
        }

        // ---------------- левая панель: прогноз против истины
        var L = { x0: 62, x1: 420, y0: 52, y1: 408 };
        var M = 3.2;
        function lx(v) { return L.x0 + (Math.max(-M, Math.min(M, v)) + M) / (2 * M) * (L.x1 - L.x0); }
        function ly(v) { return L.y1 - (Math.max(-M, Math.min(M, v)) + M) / (2 * M) * (L.y1 - L.y0); }
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0);
        ctx.strokeStyle = C.muted; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(lx(-M), ly(-M)); ctx.lineTo(lx(M), ly(M)); ctx.stroke();
        ctx.setLineDash([]);
        for (i = 0; i < NTE; i += 1) {
          var pv = cur.preds[i], outside = Math.abs(pv) > M;
          ctx.fillStyle = outside ? "rgba(185,74,59,0.85)" : "rgba(49,95,140,0.55)";
          ctx.beginPath(); ctx.arc(lx(fte[i]), ly(pv), outside ? 3.4 : 2.8, 0, 7); ctx.fill();
        }
        ctx.fillStyle = C.ink; ctx.textAlign = "left";
        ctx.fillText("прогноз против истины,  p = " + p, L.x0, L.y0 - 14);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("истинное значение", (L.x0 + L.x1) / 2, L.y1 + 24);
        ctx.save();
        ctx.translate(L.x0 - 24, (L.y0 + L.y1) / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("прогноз модели", 0, 0); ctx.restore();
        ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillStyle = C.red;
        ctx.fillText("красные точки вылетели за рамку", L.x1 - 6, L.y1 - 10);
        ctx.font = "12px PT Sans, sans-serif";

        // ---------------- правая панель: ошибка от p
        var R = { x0: 512, x1: 872, y0: 52, y1: 408 };
        var lo = 0.02, hi = 20;
        function rx(pp) {
          return R.x0 + (Math.log(pp) - Math.log(PGRID[0])) /
            (Math.log(PMAX) - Math.log(PGRID[0])) * (R.x1 - R.x0);
        }
        function ry(e) {
          var v = Math.max(lo, Math.min(hi, e));
          return R.y1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (R.y1 - R.y0);
        }
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(R.x0, R.y0, R.x1 - R.x0, R.y1 - R.y0);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        [2, 10, 40, 160, 1200].forEach(function (t) {
          ctx.strokeStyle = "rgba(222,221,212,0.9)";
          ctx.beginPath(); ctx.moveTo(rx(t), R.y0); ctx.lineTo(rx(t), R.y1); ctx.stroke();
          ctx.fillText(String(t), rx(t), R.y1 + 18);
        });
        ctx.textAlign = "right";
        [0.05, 0.2, 1, 5].forEach(function (t) {
          ctx.strokeStyle = "rgba(222,221,212,0.9)";
          ctx.beginPath(); ctx.moveTo(R.x0, ry(t)); ctx.lineTo(R.x1, ry(t)); ctx.stroke();
          ctx.fillText(String(t), R.x0 - 8, ry(t) + 4);
        });
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(rx(N), R.y0); ctx.lineTo(rx(N), R.y1); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.gold; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("p = n", rx(N) + 6, R.y0 + 14);
        ctx.font = "12px PT Sans, sans-serif";
        function poly(arr, color, width) {
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
          for (var k = 0; k < PGRID.length; k += 1) {
            var X = rx(PGRID[k]), Y = ry(arr[k]);
            if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
          }
          ctx.stroke();
        }
        poly(tests, C.red, 2.6);
        poly(trains, C.blue, 2.0);
        ctx.fillStyle = C.gold;
        ctx.beginPath(); ctx.arc(rx(p), ry(cur.test), 6, 0, 7); ctx.fill();
        ctx.strokeStyle = C.paper; ctx.lineWidth = 1.4; ctx.stroke();
        ctx.textAlign = "left";
        ctx.fillStyle = C.red; ctx.fillText("тест", R.x0 + 10, R.y0 + 16);
        ctx.fillStyle = C.blue; ctx.fillText("обучение", R.x0 + 10, R.y0 + 34);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("число признаков p (лог. шкала)", (R.x0 + R.x1) / 2, R.y1 + 42);
        ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("RMSE, лог. шкала", R.x0, R.y0 - 14);

        output.set([
          { label: "p / n", value: p + " / " + N + (p === N ? "  — порог интерполяции" : ""), color: p === N ? C.gold : C.ink },
          { label: "RMSE на обучении", value: cur.train.toFixed(3), color: C.blue },
          { label: "RMSE на тесте (против истины)", value: cur.test.toFixed(3), color: C.red },
          { label: "норма весов ‖w‖", value: cur.norm.toFixed(1), color: C.violet }
        ]);
      }

      K.slider(controls, {
        label: "Число признаков p", min: 0, max: PGRID.length - 1, step: 1, value: pIndex,
        format: function (v) { return String(PGRID[v]); }
      }, function (v) { pIndex = v; draw(); });
      K.slider(controls, {
        label: "Шум меток σ", min: 0, max: 0.6, step: 0.05, value: noise,
        format: function (v) { return v.toFixed(2); }
      }, function (v) { noise = v; draw(); });
      K.segmented(controls, {
        label: "Ridge λ", value: 0,
        options: [{ label: "0", value: 0 }, { label: "0,01", value: 0.01 }, { label: "1", value: 1 }]
      }, function (v) { lam = v; draw(); });

      var reseed = K.element("button", "kontur-int-segment", { type: "button", text: "другая выборка" });
      reseed.style.margin = "0 6px";
      reseed.addEventListener("click", function () { seed = (seed * 31 + 17) % 9973; build(); draw(); });
      controls.appendChild(reseed);

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
