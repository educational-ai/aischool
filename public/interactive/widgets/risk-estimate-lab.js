// Lesson 58: risk-estimate-lab — эмпирический риск как случайная величина
// и систематический оптимизм минимума по K кандидатам.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("risk-estimate-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 440;
      var L = 70, R = 24, T = 34, B = 62;
      var REPS = 500;

      var state = { mode: "fixed", n: 200, cands: 40, spread: 0.0 };
      var R0 = 0.20;              // истинный риск базового правила
      var destroyed = false;

      function mulberry32(a) {
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          var t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function gauss(rnd) {
        var u = 1 - rnd(), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // истинные риски кандидатов: детерминированная «лестница» вокруг R0
      function trueRisks(k, spread) {
        var out = [], i;
        for (i = 0; i < k; i += 1) {
          var u = k === 1 ? 0.5 : i / (k - 1);
          out.push(R0 + spread * (u - 0.5));
        }
        return out;
      }

      // одна серия Монте-Карло: возвращает выборки (оценка победителя, его истинный риск)
      function simulate() {
        var k = state.mode === "fixed" ? 1 : state.cands;
        var risks = trueRisks(k, state.mode === "fixed" ? 0 : state.spread);
        var rnd = mulberry32(20250958 + k * 7919 + state.n * 131 + Math.round(state.spread * 1e4));
        var est = new Float64Array(REPS), tru = new Float64Array(REPS);
        var n = state.n, i, j;
        for (i = 0; i < REPS; i += 1) {
          var bestV = Infinity, bestT = 0;
          for (j = 0; j < k; j += 1) {
            var p = risks[j];
            var sd = Math.sqrt(p * (1 - p) / n);
            var v = p + sd * gauss(rnd);          // нормальное приближение биномиальной доли
            v = Math.round(v * n) / n;            // доля ошибок кратна 1/n
            if (v < 0) v = 0; if (v > 1) v = 1;
            if (v < bestV) { bestV = v; bestT = p; }
          }
          est[i] = bestV; tru[i] = bestT;
        }
        var me = 0, mt = 0, se = 0;
        for (i = 0; i < REPS; i += 1) { me += est[i]; mt += tru[i]; }
        me /= REPS; mt /= REPS;
        for (i = 0; i < REPS; i += 1) se += (est[i] - me) * (est[i] - me);
        se = Math.sqrt(se / (REPS - 1));
        return {
          est: est, tru: tru, meanEst: me, meanTrue: mt, sd: se,
          best: Math.min.apply(null, risks), k: k,
        };
      }

      K.hint(root, "Каждая перерисовка — это 500 параллельных вселенных: в каждой мы взяли свою контрольную выборку размера n и посчитали долю ошибок. Синее облако — то, что показывает контроль; красная вертикаль — истинный риск того правила, которое мы по этому контролю выбрали. В режиме «одна заранее фиксированная функция» линии совпадают в среднем — оценка несмещённая. Включите перебор кандидатов, и синее облако систематически уползёт влево от красного: минимум шумных оценок оптимистичен, и это смещение не лечится повторным взглядом на тот же контроль.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Гистограмма — распределение той оценки, которую вы объявите результатом. Красная вертикаль — средний истинный риск выбранного правила, синяя — среднее объявленной оценки. Расстояние между ними и есть плата за перебор: она растёт с числом кандидатов и падает с ростом n. Доли ошибок моделируются нормальным приближением биномиальной доли с фиксированным зерном, поэтому картинка воспроизводима.");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Распределение оценки риска и истинный риск выбранного правила",
        onResize: draw, drag: false,
      });

      function x2p(x, lo, hi) { return L + (x - lo) / (hi - lo) * (W - L - R); }

      function draw() {
        if (destroyed) return;
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var s = simulate();

        // диапазон подстраивается под текущий разброс, иначе при малом n
        // хвосты упирались бы в край и рисовали ложные пики
        var sdRef = Math.sqrt(R0 * (1 - R0) / state.n);
        var half = Math.max(0.055, 4.2 * sdRef + state.spread * 0.6);
        var lo = R0 - half, hi = R0 + half;
        if (lo < 0) lo = 0;
        var BINS = 66, i;
        var hist = new Array(BINS).fill(0);
        for (i = 0; i < REPS; i += 1) {
          var b = Math.floor((s.est[i] - lo) / (hi - lo) * BINS);
          if (b < 0) b = 0; if (b >= BINS) b = BINS - 1;
          hist[b] += 1;
        }
        var mx = 1;
        for (i = 0; i < BINS; i += 1) if (hist[i] > mx) mx = hist[i];

        // оси
        ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(L, T); ctx.lineTo(L, H - B); ctx.lineTo(W - R, H - B);
        ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        var stepT = (hi - lo) > 0.28 ? 0.05 : ((hi - lo) > 0.14 ? 0.02 : 0.01);
        var t0 = Math.ceil(lo / stepT) * stepT;
        for (var t = t0; t <= hi + 1e-9; t += stepT) {
          var px = x2p(t, lo, hi);
          ctx.strokeStyle = C.grid; ctx.beginPath();
          ctx.moveTo(px, T); ctx.lineTo(px, H - B); ctx.stroke();
          ctx.fillText(t.toFixed(stepT < 0.02 ? 3 : 2).replace(".", ","), px, H - B + 18);
        }
        ctx.fillText("объявленная оценка риска (доля ошибок на контроле)", (L + W - R) / 2, H - 14);

        // гистограмма
        var bw = (W - L - R) / BINS;
        ctx.fillStyle = "rgba(49,95,140,0.55)";
        for (i = 0; i < BINS; i += 1) {
          if (!hist[i]) continue;
          var h = (H - B - T) * hist[i] / mx;
          ctx.fillRect(L + i * bw + 0.5, H - B - h, bw - 1, h);
        }

        function vline(x, color, label, dy) {
          var px = x2p(x, lo, hi);
          if (px < L || px > W - R) return;
          ctx.strokeStyle = color; ctx.lineWidth = 2.2;
          ctx.beginPath(); ctx.moveTo(px, T - 6); ctx.lineTo(px, H - B); ctx.stroke();
          ctx.fillStyle = color; ctx.textAlign = "left";
          ctx.fillText(label, Math.min(px + 6, W - R - 190), T + dy);
        }
        vline(s.meanTrue, C.red, "истинный риск выбранного: " + s.meanTrue.toFixed(3), 12);
        vline(s.meanEst, C.blue, "среднее оценки: " + s.meanEst.toFixed(3), 30);

        // подпись оптимизма
        var opt = s.meanTrue - s.meanEst;
        if (opt > 0.0015) {
          var a = x2p(s.meanEst, lo, hi), b2 = x2p(s.meanTrue, lo, hi), y = T + 52;
          ctx.strokeStyle = C.gold; ctx.lineWidth = 1.8;
          ctx.beginPath(); ctx.moveTo(a, y); ctx.lineTo(b2, y); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(a, y - 4); ctx.lineTo(a, y + 4);
          ctx.moveTo(b2, y - 4); ctx.lineTo(b2, y + 4); ctx.stroke();
          ctx.fillStyle = C.gold; ctx.textAlign = "center";
          ctx.fillText("оптимизм " + opt.toFixed(3), (a + b2) / 2, y - 8);
        }

        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("частота по " + REPS + " независимым контрольным выборкам", L + 6, T - 12);

        out.set([
          { label: "Кандидатов K", value: String(s.k), color: C.ink },
          { label: "Размер контроля n", value: String(state.n), color: C.ink },
          { label: "Среднее оценки", value: s.meanEst.toFixed(3), color: C.blue },
          { label: "Истинный риск выбранного", value: s.meanTrue.toFixed(3), color: C.red },
          { label: "Оптимизм (смещение вниз)", value: (s.meanTrue - s.meanEst).toFixed(3), color: C.gold },
          { label: "Разброс оценки, se", value: s.sd.toFixed(3), color: C.violet },
        ]);
      }

      K.segmented(controls, {
        label: "Что мы оцениваем",
        value: "fixed",
        options: [
          { value: "fixed", label: "одна заранее фиксированная функция" },
          { value: "pick", label: "минимум по K кандидатам" },
        ],
      }, function (v) { state.mode = v; draw(); });

      K.slider(controls, {
        label: "Размер контрольной выборки n", min: 25, max: 2000, step: 25, value: 200,
        format: function (v) { return String(Math.round(v)); },
      }, function (v) { state.n = Math.round(v); draw(); });

      K.slider(controls, {
        label: "Число кандидатов K (в режиме перебора)", min: 1, max: 200, step: 1, value: 40,
        format: function (v) { return String(Math.round(v)); },
      }, function (v) { state.cands = Math.round(v); draw(); });

      K.slider(controls, {
        label: "Разброс истинных рисков кандидатов", min: 0, max: 0.08, step: 0.005, value: 0,
        format: function (v) { return v.toFixed(3).replace(".", ","); },
      }, function (v) { state.spread = v; draw(); });

      draw();
      return function () { destroyed = true; cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
