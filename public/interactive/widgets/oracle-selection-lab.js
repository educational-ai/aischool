// Lesson 63: oracle-selection-lab — how much does looking at M candidates cost?
// Move M, the validation size n and the real spread of risks: watch the observed
// minimum sink below the truth (optimism) and the chosen model drift away from the
// oracle's best one (regret). The theoretical price sqrt(ln(2M/delta)/(2n)) is drawn too.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("oracle-selection-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 480;
      var R0 = 0.20;              // base true risk of a candidate
      var M = 100;                // number of candidates in the catalogue
      var n = 500;                // size of the validation set
      var spread = 0.0;           // how different the candidates really are
      var mode = 0;               // 0 = all equal, 1 = fan, 2 = one true leader
      var REP = 200;              // repeats used for the averaged readouts

      var seed = 20250701;
      function rnd() {            // mulberry32: deterministic, fast, 32-bit safe
        seed = (seed + 0x6d2b79f5) | 0;
        var t = seed;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      }
      var spare = null;
      function gauss() {
        if (spare !== null) { var s = spare; spare = null; return s; }
        var u = 0, v = 0, r = 0;
        do {
          u = rnd() * 2 - 1; v = rnd() * 2 - 1; r = u * u + v * v;
        } while (r <= 1e-12 || r >= 1);
        var f = Math.sqrt(-2 * Math.log(r) / r);
        spare = v * f;
        return u * f;
      }

      function trueRisks() {
        var out = new Float64Array(M), i;
        if (mode === 0) {
          for (i = 0; i < M; i += 1) out[i] = R0;
        } else if (mode === 1) {
          for (i = 0; i < M; i += 1) out[i] = R0 + spread * (M > 1 ? i / (M - 1) : 0);
        } else {
          for (i = 0; i < M; i += 1) out[i] = R0;
          out[0] = R0 - spread;
        }
        return out;
      }

      function observe(R) {       // noisy validation estimate of a 0/1 loss
        var out = new Float64Array(M);
        for (var i = 0; i < M; i += 1) {
          var se = Math.sqrt(Math.max(R[i] * (1 - R[i]), 1e-6) / n);
          out[i] = R[i] + se * gauss();
        }
        return out;
      }

      function stats() {
        var R = trueRisks(), bestTrue = Infinity, i;
        for (i = 0; i < M; i += 1) if (R[i] < bestTrue) bestTrue = R[i];
        var draw = observe(R), win = 0;
        for (i = 1; i < M; i += 1) if (draw[i] < draw[win]) win = i;

        var optSum = 0, regSum = 0, hits = 0;
        for (var r = 0; r < REP; r += 1) {
          var d = observe(R), w = 0;
          for (i = 1; i < M; i += 1) if (d[i] < d[w]) w = i;
          optSum += R[w] - d[w];
          regSum += R[w] - bestTrue;
          if (Math.abs(R[w] - bestTrue) < 1e-12) hits += 1;
        }
        return {
          R: R, draw: draw, win: win, bestTrue: bestTrue,
          obsMin: draw[win], trueOfWin: R[win],
          optimism: optSum / REP, regret: regSum / REP, hit: hits / REP,
          eps: Math.sqrt(Math.log(2 * M / 0.05) / (2 * n))
        };
      }

      K.hint(
        root,
        "Каталог из M кандидатов проверяют на одной validation-выборке размера n и берут лучшего. " +
        "Сначала оставьте разброс истинных рисков нулевым: все кандидаты на самом деле одинаковы, " +
        "но наблюдаемый минимум всё равно опускается ниже истины — это цена просмотра, а не качество. " +
        "Затем включите режим с настоящим лидером и найдите, при каком n процедура начинает его находить."
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Серая линия — истинный риск каждого кандидата, синие точки — его шумная оценка на validation. " +
        "Красная точка — тот, кого мы выберем; зелёная — тот, кого выбрал бы оракул, знающий истину. " +
        "Оптимизм = насколько наблюдаемый минимум ниже истинного риска выбранного. " +
        "Регрет = насколько выбранный хуже настоящего лучшего. Оба усреднены по 200 повторам."
      );

      var cs = K.makeCanvas(stage, W, H, {
        label: "Каталог кандидатов, шумные оценки и цена выбора",
        onResize: draw, drag: false
      });

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var s = stats();
        var gx = 62, gy = 56, gw = 500, gh = 340;

        var lo = Math.min(s.bestTrue, s.obsMin) - 0.02;
        var hi = Math.max(R0 + Math.max(spread, 0.01), s.obsMin) + 0.035;
        function Y(v) { return gy + gh - (v - lo) / (hi - lo) * gh; }
        function X(i) { return gx + (M > 1 ? i / (M - 1) : 0.5) * gw; }

        ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
        ctx.strokeRect(gx, gy, gw, gh);
        ctx.strokeStyle = C.grid || "#deddd4";
        var step = (hi - lo) / 5, t;
        ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        for (t = 0; t <= 5; t += 1) {
          var v = lo + t * step, yy = Y(v);
          ctx.beginPath(); ctx.moveTo(gx, yy); ctx.lineTo(gx + gw, yy); ctx.stroke();
          ctx.fillText((v * 100).toFixed(1) + "%", gx - 8, yy + 4);
        }

        // observed estimates
        ctx.fillStyle = "rgba(49,95,140,0.55)";
        var stepDots = Math.max(1, Math.floor(M / 400));
        for (var i = 0; i < M; i += stepDots) {
          ctx.beginPath(); ctx.arc(X(i), Y(s.draw[i]), M > 200 ? 1.6 : 3, 0, 7); ctx.fill();
        }
        // true risk curve
        ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
        ctx.beginPath();
        for (i = 0; i < M; i += 1) {
          var px = X(i), py = Y(s.R[i]);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        if (M === 1) { ctx.moveTo(gx, Y(s.R[0])); ctx.lineTo(gx + gw, Y(s.R[0])); }
        ctx.stroke();

        // oracle and winner
        var oracle = 0;
        for (i = 1; i < M; i += 1) if (s.R[i] < s.R[oracle]) oracle = i;
        ctx.fillStyle = C.green; ctx.beginPath(); ctx.arc(X(oracle), Y(s.R[oracle]), 7, 0, 7); ctx.fill();
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(X(s.win), Y(s.obsMin), 7, 0, 7); ctx.fill();
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(X(s.win), Y(s.obsMin)); ctx.lineTo(X(s.win), Y(s.trueOfWin)); ctx.stroke();
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(X(s.win), Y(s.trueOfWin), 4, 0, 7); ctx.fill();

        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("кандидаты каталога, от 1 до " + M, gx + gw / 2, gy + gh + 20);
        ctx.save();
        ctx.translate(16, gy + gh / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("риск (доля ошибок)", 0, 0);
        ctx.restore();
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("истинный риск", gx + 6, gy + 16);
        ctx.fillStyle = C.red;
        ctx.fillText("наблюдаемый минимум (его и выберем)", gx + 6, gy + 32);

        // right panel: the three prices
        var px0 = 596, bw = 240;
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("во что обошёлся просмотр", px0, gy + 4);
        var bars = [
          { t: "оптимизм оценки", v: s.optimism, c: C.red },
          { t: "регрет (хуже оракула)", v: s.regret, c: C.gold },
          { t: "гарантия Хёфдинга ε", v: s.eps, c: C.blue }
        ];
        var scale = Math.max(s.eps, s.optimism, s.regret, 1e-4);
        for (var b = 0; b < bars.length; b += 1) {
          var by = gy + 40 + b * 66;
          ctx.fillStyle = C.muted; ctx.font = "11.5px PT Sans, sans-serif";
          ctx.fillText(bars[b].t, px0, by);
          ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
          ctx.strokeRect(px0, by + 8, bw, 16);
          ctx.fillStyle = bars[b].c;
          ctx.fillRect(px0, by + 8, bw * Math.min(bars[b].v / scale, 1), 16);
          ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
          ctx.fillText((bars[b].v * 100).toFixed(2).replace(".", ",") + " п.п.", px0, by + 40);
        }
        var degenerate = spread < 1e-9 || mode === 0;
        ctx.fillStyle = C.muted; ctx.font = "11.5px PT Sans, sans-serif";
        ctx.fillText("нашли настоящего лучшего", px0, gy + 250);
        if (degenerate) {
          ctx.fillStyle = C.muted; ctx.font = "26px PT Sans, sans-serif";
          ctx.fillText("—", px0, gy + 282);
          ctx.font = "10.5px PT Sans, sans-serif";
          ctx.fillText("все кандидаты равны: лучшего нет", px0, gy + 300);
        } else {
          ctx.fillStyle = s.hit > 0.9 ? C.green : (s.hit > 0.4 ? C.gold : C.red);
          ctx.font = "26px PT Sans, sans-serif";
          ctx.fillText((s.hit * 100).toFixed(0) + "%", px0, gy + 282);
          ctx.fillStyle = C.muted; ctx.font = "10.5px PT Sans, sans-serif";
          ctx.fillText("в " + (s.hit * 100).toFixed(0) + " случаях из 100", px0, gy + 300);
        }

        output.set([
          { label: "Наблюдаемый минимум", value: (s.obsMin * 100).toFixed(2) + "%", color: C.red },
          { label: "Истинный риск выбранного", value: (s.trueOfWin * 100).toFixed(2) + "%", color: C.ink },
          { label: "Истинный риск лучшего (оракул)", value: (s.bestTrue * 100).toFixed(2) + "%", color: C.green },
          { label: "Цена просмотра ε при δ=0,05", value: (s.eps * 100).toFixed(2) + " п.п.", color: C.blue }
        ]);
      }

      K.slider(controls, {
        label: "Кандидатов M", min: 0, max: 3, step: 0.25, value: 2,
        format: function (v) { return String(Math.round(Math.pow(10, v))); }
      }, function (v) { M = Math.max(1, Math.round(Math.pow(10, v))); draw(); });

      K.slider(controls, {
        label: "Размер validation n", min: 50, max: 5000, step: 50, value: 500,
        format: function (v) { return String(v); }
      }, function (v) { n = v; draw(); });

      K.slider(controls, {
        label: "Настоящее различие рисков", min: 0, max: 0.1, step: 0.005, value: 0,
        format: function (v) { return (v * 100).toFixed(1).replace(".", ",") + " п.п."; }
      }, function (v) { spread = v; draw(); });

      K.segmented(controls, {
        label: "Каталог", value: 0, options: [
          { label: "все одинаковы", value: 0 },
          { label: "веер качества", value: 1 },
          { label: "один лидер", value: 2 }
        ]
      }, function (v) { mode = v; draw(); });

      var b1 = K.element("button", "kontur-int-segment", { type: "button", text: "новый жребий" });
      b1.style.margin = "0 6px";
      b1.addEventListener("click", function () { draw(); });
      controls.appendChild(b1);

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
