// Lesson 06: label propagation over two clouds of days, with a creeping error.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("pseudo-label-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;
      var N = 300;

      var state = {
        seeds: Number(options.seeds || 20),
        threshold: Number(options.threshold || 0.9),
        rounds: Number(options.rounds || 6),
      };

      K.hint(
        root,
        "Квадраты — настоящие метки, всё остальное модель подписывает сама. Понижай порог уверенности и смотри, как покрытие растёт, а точность проваливается волной от границы облаков.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Триста дней-точек в координатах «доля аренд в 8:00 и в 13:00»: облако будних, облако выходных и мост из пограничных дней вроде предпраздничных пятниц. Уверенность псевдометки равна доле голосов пяти ближайших уже размеченных соседей. Расчёт детерминированный, seed = 730.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Распространение псевдометок по двум облакам дней",
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

      // Two clouds shaped after the real day features from the article figure.
      var points = (function () {
        var rnd = mulberry(730);
        var list = [];
        for (var i = 0; i < N; i += 1) {
          var u1 = rnd(), u2 = rnd(), u3 = rnd(), u4 = rnd();
          var g1 = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-9))) * Math.cos(2 * Math.PI * u2);
          var g2 = Math.sqrt(-2 * Math.log(Math.max(u3, 1e-9))) * Math.cos(2 * Math.PI * u4);
          if (i % 5 === 4) {
            // Bridge days: pre-holiday Fridays with a mixed rhythm. The true
            // type follows geography with occasional exceptions, so a careful
            // threshold stalls here and a lax one paints the bridge with errors.
            var bx = 6.3 + g1 * 1.6;
            var by = 6.4 + g2 * 1.1;
            var geo = bx < 6.3 ? 0 : 1;
            list.push({
              x: bx,
              y: by,
              truth: rnd() < 0.22 ? 1 - geo : geo,
            });
            continue;
          }
          var weekendish = i % 3 === 0;
          var cx = weekendish ? 3.2 : 9.4;
          var cy = weekendish ? 8.6 : 4.6;
          list.push({
            x: cx + g1 * (weekendish ? 1.5 : 1.9),
            y: cy + g2 * (weekendish ? 1.35 : 1.05),
            truth: weekendish ? 0 : 1,
          });
        }
        return list;
      })();

      function neighbors(i, labels, k) {
        var dists = [];
        for (var j = 0; j < N; j += 1) {
          if (j === i || labels[j] < 0) continue;
          var dx = points[i].x - points[j].x;
          var dy = points[i].y - points[j].y;
          dists.push({ j: j, d: dx * dx + dy * dy });
        }
        dists.sort(function (a, b) { return a.d - b.d; });
        return dists.slice(0, k);
      }

      function simulate() {
        var rnd = mulberry(42);
        var order = [];
        for (var i = 0; i < N; i += 1) order.push(i);
        // Deterministic shuffle for seed selection.
        for (var s = order.length - 1; s > 0; s -= 1) {
          var q = Math.floor(rnd() * (s + 1));
          var tmp = order[s];
          order[s] = order[q];
          order[q] = tmp;
        }
        var labels = new Array(N).fill(-1);
        var isSeed = new Array(N).fill(false);
        for (var k = 0; k < state.seeds; k += 1) {
          labels[order[k]] = points[order[k]].truth;
          isSeed[order[k]] = true;
        }
        for (var round = 0; round < state.rounds; round += 1) {
          var assigned = [];
          for (var p = 0; p < N; p += 1) {
            if (labels[p] >= 0) continue;
            var near = neighbors(p, labels, 5);
            if (near.length < 3) continue;
            var votes = 0;
            near.forEach(function (item) { votes += labels[item.j]; });
            var confidence = Math.max(votes / near.length, 1 - votes / near.length);
            if (confidence >= state.threshold) {
              assigned.push({ p: p, label: votes / near.length >= 0.5 ? 1 : 0 });
            }
          }
          if (!assigned.length) break;
          assigned.forEach(function (item) { labels[item.p] = item.label; });
        }
        var covered = 0;
        var errors = 0;
        for (var m = 0; m < N; m += 1) {
          if (labels[m] >= 0 && !isSeed[m]) {
            covered += 1;
            if (labels[m] !== points[m].truth) errors += 1;
          }
        }
        return { labels: labels, isSeed: isSeed, covered: covered, errors: errors };
      }

      var chart = { x: 64, y: 30, w: 920, h: 460 };
      var X_MIN = -3, X_MAX = 17, Y_MIN = 0.5, Y_MAX = 13.5;

      function sx(v) { return chart.x + (v - X_MIN) / (X_MAX - X_MIN) * chart.w; }
      function sy(v) { return chart.y + chart.h - (v - Y_MIN) / (Y_MAX - Y_MIN) * chart.h; }

      function draw() {
        var ctx = canvasState.ctx;
        var result = simulate();
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        for (var i = 0; i < N; i += 1) {
          var label = result.labels[i];
          var x = sx(points[i].x);
          var y = sy(points[i].y);
          if (label < 0) {
            ctx.fillStyle = C.grid;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
            continue;
          }
          ctx.fillStyle = label === 1 ? C.blue : C.gold;
          ctx.beginPath();
          ctx.arc(x, y, 3.6, 0, Math.PI * 2);
          ctx.fill();
          if (!result.isSeed[i] && label !== points[i].truth) {
            ctx.strokeStyle = C.red;
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, Math.PI * 2);
            ctx.stroke();
          }
          if (result.isSeed[i]) {
            ctx.strokeStyle = C.ink;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x - 6, y - 6, 12, 12);
          }
        }

        ctx.fillStyle = C.muted;
        ctx.textAlign = "left";
        ctx.fillText("доля аренд в 8:00 →", chart.x, chart.y + chart.h + 24);
        ctx.save();
        ctx.translate(chart.x - 34, chart.y + chart.h);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("доля аренд в 13:00 →", 0, 0);
        ctx.restore();

        var accuracy = result.covered
          ? Math.round((1 - result.errors / result.covered) * 1000) / 10
          : 100;
        output.set([
          { label: "настоящих меток", value: String(state.seeds) },
          { label: "псевдометок", value: result.covered + " из " + (N - state.seeds) },
          {
            label: "ошибочных псевдометок",
            value: String(result.errors),
            color: result.errors > 12 ? C.red : C.green,
          },
          {
            label: "точность псевдометок",
            value: String(accuracy).replace(".", ",") + "%",
            color: accuracy > 95 ? C.green : C.red,
          },
        ]);
      }

      K.slider(
        controls,
        { label: "Настоящих меток", min: 4, max: 60, step: 2, value: state.seeds },
        function (value) { state.seeds = value; draw(); },
      );
      K.slider(
        controls,
        {
          label: "Порог уверенности",
          min: 0.6,
          max: 1.0,
          step: 0.05,
          value: state.threshold,
          format: function (v) { return v.toFixed(2).replace(".", ","); },
        },
        function (value) { state.threshold = value; draw(); },
      );
      K.slider(
        controls,
        { label: "Итераций распространения", min: 1, max: 14, step: 1, value: state.rounds },
        function (value) { state.rounds = value; draw(); },
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
