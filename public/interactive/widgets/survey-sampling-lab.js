// Lesson 05: one hundred repeated surveys under three sampling schemes.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("survey-sampling-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 540;
      var REPS = 100;
      var Z = 1.96;

      // The hidden city: three commute strata with their own support levels.
      var STRATA = [
        { key: "walk", share: 0.5, support: 0.47 },
        { key: "transit", share: 0.3, support: 0.82 },
        { key: "car", share: 0.2, support: 0.35 },
      ];
      // Landline directories overrepresent car owners, as in 1936.
      var PHONE_WEIGHTS = [0.25, 0.15, 0.6];

      var TRUE_P = STRATA.reduce(function (acc, s) { return acc + s.share * s.support; }, 0);
      var PHONE_P = STRATA.reduce(function (acc, s, i) { return acc + PHONE_WEIGHTS[i] * s.support; }, 0);

      var state = {
        scheme: options.scheme || "random",
        n: Number(options.n || 500),
      };

      K.hint(
        root,
        "Сто раз повтори один и тот же опрос и посмотри на сто интервалов сразу. Размер выборки сжимает облако; честность рамки решает, вокруг чего оно сожмётся.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Истинная поддержка в городе 55,1%. Телефонная книга накрывает автомобилистов втрое чаще пешеходов, поэтому её центр — 45,1%. Выборочная доля моделируется нормальным приближением биномиального распределения; при n от 200 оно точнее толщины линии. Расчёт детерминированный, seed = 1936.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Сто доверительных интервалов при выбранной схеме опроса",
        onResize: draw,
        drag: false,
      });

      function mulberry(seed) {
        var a = seed >>> 0;
        return function () {
          a = (a + 0x6d2b79f5) >>> 0;
          var t = a;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }

      function gauss(i, salt) {
        var next = mulberry(1936 + i * 641 + salt * 7919);
        var u = Math.max(next(), 1e-9);
        var v = next();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      function schemeIndex() {
        if (state.scheme === "phone") return 1;
        if (state.scheme === "strat") return 2;
        return 0;
      }

      function simulate() {
        var idx = schemeIndex();
        var center = idx === 1 ? PHONE_P : TRUE_P;
        var variance;
        if (idx === 2) {
          variance = STRATA.reduce(function (acc, s) {
            return acc + s.share * s.support * (1 - s.support);
          }, 0) / state.n;
        } else {
          variance = center * (1 - center) / state.n;
        }
        var sd = Math.sqrt(variance);
        var runs = [];
        var covered = 0;
        var widthSum = 0;
        var meanSum = 0;
        for (var k = 0; k < REPS; k += 1) {
          var phat = center + sd * gauss(k, idx * 7 + 3);
          var se = Math.sqrt(Math.max(phat * (1 - phat), 1e-6) / state.n);
          if (idx === 2) se = se * Math.sqrt(variance / (center * (1 - center) / state.n));
          var lo = phat - Z * se;
          var hi = phat + Z * se;
          var hit = lo <= TRUE_P && TRUE_P <= hi;
          if (hit) covered += 1;
          widthSum += hi - lo;
          meanSum += phat;
          runs.push({ phat: phat, lo: lo, hi: hi, hit: hit });
        }
        return {
          runs: runs,
          covered: covered,
          meanWidth: widthSum / REPS,
          meanEstimate: meanSum / REPS,
          center: center,
        };
      }

      var chart = { x: 74, y: 44, w: 900, h: 400 };
      var Y_MIN = 0.34;
      var Y_MAX = 0.72;

      function toY(value) {
        var clamped = Math.max(Y_MIN, Math.min(Y_MAX, value));
        return chart.y + chart.h - (clamped - Y_MIN) / (Y_MAX - Y_MIN) * chart.h;
      }

      function draw() {
        var ctx = canvasState.ctx;
        var result = simulate();
        ctx.clearRect(0, 0, W, H);

        ctx.font = "13px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        // Grid and axis.
        for (var g = 0.35; g <= 0.71; g += 0.05) {
          var gy = toY(g);
          ctx.strokeStyle = C.grid;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chart.x, gy);
          ctx.lineTo(chart.x + chart.w, gy);
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText((g * 100).toFixed(0) + "%", chart.x - 10, gy);
          ctx.textAlign = "left";
        }

        // Intervals.
        var step = chart.w / REPS;
        result.runs.forEach(function (run, k) {
          var x = chart.x + step * (k + 0.5);
          ctx.strokeStyle = run.hit ? C.axis : C.red;
          ctx.lineWidth = run.hit ? 1.4 : 2.2;
          ctx.beginPath();
          ctx.moveTo(x, toY(run.lo));
          ctx.lineTo(x, toY(run.hi));
          ctx.stroke();
          ctx.fillStyle = run.hit ? C.axis : C.red;
          ctx.beginPath();
          ctx.arc(x, toY(run.phat), 2.1, 0, Math.PI * 2);
          ctx.fill();
        });

        // True share.
        var trueY = toY(TRUE_P);
        ctx.strokeStyle = C.blue;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(chart.x, trueY);
        ctx.lineTo(chart.x + chart.w, trueY);
        ctx.stroke();
        ctx.fillStyle = C.blue;
        ctx.fillText("истинная доля 55,1%", chart.x + chart.w - 156, trueY - 12);

        // Frame centre for the biased scheme.
        if (schemeIndex() === 1) {
          var frameY = toY(PHONE_P);
          ctx.strokeStyle = C.gold;
          ctx.lineWidth = 1.6;
          ctx.setLineDash([6, 5]);
          ctx.beginPath();
          ctx.moveTo(chart.x, frameY);
          ctx.lineTo(chart.x + chart.w, frameY);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = C.gold;
          ctx.fillText("центр телефонной рамки 45,1%", chart.x + chart.w - 226, frameY + 14);
        }

        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.fillText(
          "Сто повторов опроса, n = " + state.n + " в каждом",
          chart.x,
          chart.y - 20,
        );
        ctx.font = "13px PT Sans, sans-serif";
        ctx.fillStyle = result.covered >= 85 ? C.green : C.red;
        ctx.fillText(
          "покрытие: " + result.covered + " из " + REPS,
          chart.x,
          chart.y + chart.h + 26,
        );
        ctx.fillStyle = C.muted;
        ctx.fillText(
          "красные интервалы не накрыли истинную долю",
          chart.x + 210,
          chart.y + chart.h + 26,
        );

        output.set([
          { label: "истинная доля", value: (TRUE_P * 100).toFixed(1).replace(".", ",") + "%" },
          {
            label: "средняя оценка",
            value: (result.meanEstimate * 100).toFixed(1).replace(".", ",") + "%",
            color: Math.abs(result.meanEstimate - TRUE_P) > 0.02 ? C.red : C.green,
          },
          {
            label: "покрытие интервалов",
            value: result.covered + " / " + REPS,
            color: result.covered >= 85 ? C.green : C.red,
          },
          {
            label: "средняя ширина",
            value: "±" + (result.meanWidth / 2 * 100).toFixed(1).replace(".", ",") + " п.п.",
          },
        ]);
      }

      K.segmented(
        controls,
        {
          label: "Схема выборки",
          value: state.scheme,
          options: [
            { value: "random", label: "случайная" },
            { value: "phone", label: "телефонная книга" },
            { value: "strat", label: "стратифицированная" },
          ],
        },
        function (value) {
          state.scheme = value;
          draw();
        },
      );
      K.slider(
        controls,
        { label: "Размер выборки n", min: 200, max: 3200, step: 100, value: state.n },
        function (value) {
          state.n = value;
          draw();
        },
      );

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
