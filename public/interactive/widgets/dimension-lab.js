// Lesson 60: dimension-lab — add noise coordinates to a task whose signal lives in two of them.
// Watch the distance contrast collapse, the ball vanish inside the cube and k-NN accuracy fall,
// then switch to the two informative axes and see the geometry come back.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("dimension-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var PB = "#fffef9";
      var DMAX = 200, KNN = 7;
      var DGRID = [2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200];
      var d = 20, nTrain = 300, useOnlyInformative = false;
      var Xtr = [], ytr = [], Xte = [], yte = [], curveAll = [], curveTwo = 0;
      var NTEST = 150;

      // deterministic generator so every reader sees the same picture
      function rng(seed) {
        var s = seed >>> 0;
        return function () {
          s = (s + 0x6d2b79f5) >>> 0;
          var t = s;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function gauss(r) {
        var u = 1 - r(), v = r();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // two classes separated along coordinate 0; coordinate 1 carries a weaker signal;
      // coordinates 2..DMAX-1 are pure noise, identical for both classes.
      function build() {
        var r = rng(60), i, j;
        Xtr = []; ytr = []; Xte = []; yte = [];
        function sample(store, labels, count) {
          for (i = 0; i < count; i += 1) {
            var cls = r() < 0.5 ? 0 : 1, row = new Float64Array(DMAX);
            row[0] = gauss(r) + (cls === 1 ? 1.3 : -1.3);
            row[1] = gauss(r) + (cls === 1 ? 0.7 : -0.7);
            for (j = 2; j < DMAX; j += 1) row[j] = gauss(r);
            store.push(row); labels.push(cls);
          }
        }
        sample(Xtr, ytr, nTrain);
        sample(Xte, yte, NTEST);
        curveAll = DGRID.map(function (dd) { return accuracy(dd); });
        curveTwo = accuracy(2);
      }

      function dist2(a, b, dim) {
        var s = 0;
        for (var j = 0; j < dim; j += 1) { var e = a[j] - b[j]; s += e * e; }
        return s;
      }

      function accuracy(dim) {
        var right = 0, i, t;
        // allocate the scratch rows once per call and mutate them: dragging the
        // sliders re-runs this eleven times, so per-pair object churn is visible
        var idx = new Array(Xtr.length);
        for (i = 0; i < Xtr.length; i += 1) idx[i] = { v: 0, y: 0 };
        for (t = 0; t < Xte.length; t += 1) {
          for (i = 0; i < Xtr.length; i += 1) {
            idx[i].v = dist2(Xte[t], Xtr[i], dim);
            idx[i].y = ytr[i];
          }
          idx.sort(function (p, q) { return p.v - q.v; });
          var votes = 0;
          for (i = 0; i < KNN; i += 1) votes += idx[i].y;
          if ((votes * 2 > KNN ? 1 : 0) === yte[t]) right += 1;
        }
        return right / Xte.length;
      }

      // fraction of the cube [-1,1]^d occupied by the inscribed unit ball, exactly
      // V_1 = 2, V_2 = pi, V_k = V_{k-2} * 2pi/k — exact recursion, no gamma needed
      function ballVolume(dim) {
        var v = dim % 2 === 0 ? Math.PI : 2, k;
        for (k = (dim % 2 === 0 ? 4 : 3); k <= dim; k += 2) v = v * 2 * Math.PI / k;
        return v;
      }
      function ballFraction(dim) {
        return ballVolume(dim) / Math.pow(2, dim);
      }

      function distStats(dim) {
        var q = Xte[0], ds = [], i;
        for (i = 0; i < Xtr.length; i += 1) ds.push(Math.sqrt(dist2(q, Xtr[i], dim)));
        ds.sort(function (a, b) { return a - b; });
        var mean = 0;
        for (i = 0; i < ds.length; i += 1) mean += ds[i];
        mean /= ds.length;
        return { ds: ds, min: ds[0], max: ds[ds.length - 1], mean: mean };
      }

      K.hint(root, "Сигнал живёт только в двух координатах: классы сдвинуты по первой и слабее по второй. Все остальные оси — чистый шум, одинаковый для обоих классов. Двигайте ползунок размерности и следите за тремя вещами сразу: за разбросом расстояний от одной точки до всех остальных, за долей шара внутри куба и за точностью k-NN. Затем переключитесь на «только 2 информативные» — данные те же, лишней информации мы не добавили, но геометрия возвращается.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева — гистограмма расстояний от одной тестовой точки до всех обучающих, поделённых на своё среднее: с ростом d она сжимается в иглу, и «ближайший» перестаёт быть по-настоящему ближайшим. Справа — точность k-NN по всем d координатам (красная) против точности по двум информативным (зелёная). Разрыв между кривыми и есть цена лишних измерений.");
      var cs = K.makeCanvas(stage, W, H, { label: "Гистограмма расстояний и точность k-NN в зависимости от размерности", onResize: draw });

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var dim = useOnlyInformative ? 2 : d;
        var st = distStats(dim);
        var contrast = (st.max - st.min) / st.min;

        // ---------------- left: histogram of normalised distances
        var LX = 60, LY = 330, LW = 350, LH = 240;
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(LX, LY - LH); ctx.lineTo(LX, LY); ctx.lineTo(LX + LW, LY); ctx.stroke();
        var BINS = 44, lo = 0.0, hi = 2.2, hist = new Array(BINS).fill(0), i;
        for (i = 0; i < st.ds.length; i += 1) {
          var v = st.ds[i] / st.mean;
          var b = Math.floor((v - lo) / (hi - lo) * BINS);
          if (b >= 0 && b < BINS) hist[b] += 1;
        }
        var hmax = 1;
        for (i = 0; i < BINS; i += 1) if (hist[i] > hmax) hmax = hist[i];
        var bw = LW / BINS;
        ctx.fillStyle = "rgba(49,95,140,0.55)";
        for (i = 0; i < BINS; i += 1) {
          var h = hist[i] / hmax * (LH - 24);
          ctx.fillRect(LX + i * bw, LY - h, bw - 1, h);
        }
        // marks for min / max
        function xOf(v) { return LX + (v - lo) / (hi - lo) * LW; }
        ctx.strokeStyle = C.green; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(xOf(st.min / st.mean), LY); ctx.lineTo(xOf(st.min / st.mean), LY - LH + 16); ctx.stroke();
        ctx.strokeStyle = C.red;
        ctx.beginPath(); ctx.moveTo(xOf(st.max / st.mean), LY); ctx.lineTo(xOf(st.max / st.mean), LY - LH + 16); ctx.stroke();
        ctx.fillStyle = C.green; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("ближайший", xOf(st.min / st.mean), LY - LH + 8);
        ctx.fillStyle = C.red;
        ctx.fillText("самый далёкий", xOf(st.max / st.mean), LY - LH - 4);
        ctx.fillStyle = C.muted; ctx.font = "12px PT Sans, sans-serif";
        for (var t = 0.5; t <= 2.0; t += 0.5) { ctx.fillText(String(t).replace(".", ","), xOf(t), LY + 16); }
        ctx.fillText("расстояние / среднее расстояние", LX + LW / 2, LY + 36);
        ctx.fillStyle = C.ink || "#171915"; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("расстояния от одной точки до всех обучающих", LX - 6, LY - LH - 22);

        // ---------------- right: accuracy vs dimension
        var RX = 500, RY = 330, RW = 330, RH = 240;
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(RX, RY - RH); ctx.lineTo(RX, RY); ctx.lineTo(RX + RW, RY); ctx.stroke();
        var A0 = 0.5, A1 = 1.0;
        function yOfAcc(a) { return RY - (a - A0) / (A1 - A0) * (RH - 20); }
        function xOfDim(dd) { return RX + Math.log(dd / 2) / Math.log(DMAX / 2) * RW; }
        ctx.strokeStyle = C.muted; ctx.setLineDash([2, 3]); ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(RX, yOfAcc(0.75)); ctx.lineTo(RX + RW, yOfAcc(0.75)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("75%", RX - 6, yOfAcc(0.75) + 4);
        ctx.fillText("100%", RX - 6, yOfAcc(1.0) + 4);
        ctx.fillText("50%", RX - 6, yOfAcc(0.5) + 4);
        // green line: only the two informative coordinates
        ctx.strokeStyle = C.green; ctx.lineWidth = 2.0;
        ctx.beginPath(); ctx.moveTo(RX, yOfAcc(curveTwo)); ctx.lineTo(RX + RW, yOfAcc(curveTwo)); ctx.stroke();
        ctx.fillStyle = C.green; ctx.textAlign = "left";
        ctx.fillText("только 2 информативные: " + (curveTwo * 100).toFixed(1) + "%", RX + 8, yOfAcc(curveTwo) - 8);
        // red curve: all d coordinates
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.6; ctx.beginPath();
        for (i = 0; i < DGRID.length; i += 1) {
          var px = xOfDim(DGRID[i]), py = yOfAcc(curveAll[i]);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        for (i = 0; i < DGRID.length; i += 1) {
          ctx.fillStyle = C.red; ctx.beginPath();
          ctx.arc(xOfDim(DGRID[i]), yOfAcc(curveAll[i]), 3.5, 0, 7); ctx.fill();
        }
        // marker at the current d
        var accNow = useOnlyInformative ? curveTwo : accuracy(d);
        if (!useOnlyInformative) {
          ctx.fillStyle = C.gold; ctx.beginPath();
          ctx.arc(xOfDim(d), yOfAcc(accNow), 7, 0, 7); ctx.fill();
          ctx.strokeStyle = PB; ctx.lineWidth = 1.4; ctx.stroke();
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        for (i = 0; i < DGRID.length; i += 1) {
          if (DGRID[i] === 2 || DGRID[i] === 8 || DGRID[i] === 30 || DGRID[i] === 200) {
            ctx.fillText(String(DGRID[i]), xOfDim(DGRID[i]), RY + 16);
          }
        }
        ctx.fillText("число используемых координат d", RX + RW / 2, RY + 36);
        ctx.fillStyle = C.ink || "#171915"; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("точность k-NN (k = 7) на отложенных точках", RX - 6, RY - RH - 22);

        var bf = ballFraction(dim);
        output.set([
          { label: "Координат в расстоянии", value: String(dim) + (useOnlyInformative ? " (обе полезные)" : " (2 полезные + " + (dim - 2) + " шумовых)"), color: C.blue },
          { label: "Контраст (max−min)/min", value: contrast.toFixed(2), color: C.red },
          { label: "Доля шара в кубе V_d/2^d", value: bf < 1e-4 ? bf.toExponential(1) : bf.toFixed(4), color: C.violet || C.gold },
          { label: "Точность k-NN", value: (accNow * 100).toFixed(1) + "%", color: C.green }
        ]);
      }

      build();
      K.slider(controls, {
        label: "Размерность d (2 полезные + шум)", min: 2, max: DMAX, step: 1, value: d,
        format: function (v) { return String(v); }
      }, function (v) { d = v; draw(); });
      K.slider(controls, {
        label: "Обучающих точек", min: 100, max: 600, step: 50, value: nTrain,
        format: function (v) { return String(v); }
      }, function (v) { nTrain = v; build(); draw(); });
      K.segmented(controls, {
        label: "Какие координаты в расстоянии",
        value: 0,
        options: [{ label: "все d", value: 0 }, { label: "только 2 информативные", value: 1 }]
      }, function (v) { useOnlyInformative = v === 1; draw(); });
      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
