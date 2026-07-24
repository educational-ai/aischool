// Lesson 09: clustering 142 Gapminder countries — metric, K and stability live.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("gapminder-cluster-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;

      // 142 countries of Gapminder 2007: [gdpPercap, lifeExp].
      var PTS=[[974.6, 43.8], [5937.0, 76.4], [6223.4, 72.3], [4797.2, 42.7], [12779.4, 75.3], [34435.4, 81.2], [36126.5, 79.8], [29796.0, 75.6], [1391.3, 64.1], [33692.6, 79.4], [1441.3, 56.7], [3822.1, 65.6], [7446.3, 74.9], [12569.9, 50.7], [9065.8, 72.4], [10680.8, 73.0], [1217.0, 52.3], [430.1, 49.6], [1713.8, 59.7], [2042.1, 50.4], [36319.2, 80.7], [706.0, 44.7], [1704.1, 50.7], [13171.6, 78.6], [4959.1, 73.0], [7006.6, 72.9], [986.1, 65.2], [277.6, 46.5], [3632.6, 55.3], [9645.1, 78.8], [1544.8, 48.3], [14619.2, 75.7], [8948.1, 78.3], [22833.3, 76.5], [35278.4, 78.3], [2082.5, 54.8], [6025.4, 72.2], [6873.3, 75.0], [5581.2, 71.3], [5728.4, 71.9], [12154.1, 51.6], [641.4, 58.0], [690.8, 52.9], [33207.1, 79.3], [30470.0, 80.7], [13206.5, 56.7], [752.7, 59.4], [32170.4, 79.4], [1327.6, 60.0], [27538.4, 79.5], [5186.1, 70.3], [942.7, 56.0], [579.2, 46.4], [1201.6, 60.9], [3548.3, 70.2], [39725.0, 82.2], [18008.9, 73.3], [36180.8, 81.8], [2452.2, 64.7], [3540.7, 70.7], [11605.7, 71.0], [4471.1, 59.5], [40676.0, 78.9], [25523.3, 80.7], [28569.7, 80.5], [7320.9, 72.6], [31656.1, 82.6], [4519.5, 72.5], [1463.2, 54.1], [1593.1, 67.3], [23348.1, 78.6], [47307.0, 77.6], [10461.1, 72.0], [1569.3, 42.6], [414.5, 45.7], [12057.5, 74.0], [1044.8, 59.4], [759.3, 48.3], [12451.7, 74.2], [1042.6, 54.5], [1803.2, 64.2], [10957.0, 72.8], [11977.6, 76.2], [3095.8, 66.8], [9253.9, 74.5], [3820.2, 71.2], [823.7, 42.1], [944.0, 62.1], [4811.1, 52.9], [1091.4, 63.8], [36797.9, 79.8], [25185.0, 80.2], [2749.3, 72.9], [619.7, 56.9], [2014.0, 46.9], [49357.2, 80.2], [22316.2, 75.6], [2605.9, 65.5], [9809.2, 75.5], [4172.8, 71.8], [7408.9, 71.4], [3190.5, 71.7], [15389.9, 75.6], [20509.6, 78.1], [19328.7, 78.7], [7670.1, 76.4], [10808.5, 72.5], [863.1, 46.2], [1598.4, 65.5], [21654.8, 72.8], [1712.5, 63.1], [9786.5, 74.0], [862.5, 42.6], [47143.2, 80.0], [18678.3, 74.7], [25768.3, 77.9], [926.1, 48.2], [9269.7, 49.3], [28821.1, 80.9], [3970.1, 72.4], [2602.4, 58.6], [4513.5, 39.6], [33859.7, 80.9], [37506.4, 81.7], [4184.5, 74.1], [28718.3, 78.4], [1107.5, 52.5], [7458.4, 70.6], [883.0, 58.4], [18008.5, 69.8], [7092.9, 73.9], [8458.3, 71.8], [1056.4, 51.5], [33203.3, 79.4], [42951.7, 78.2], [10611.5, 76.4], [11415.8, 73.7], [2441.6, 74.2], [3025.3, 73.4], [2280.8, 62.7], [1271.2, 42.4], [469.7, 43.5]];
      var MARKS=[[0, "Афганистан"], [13, "Ботсвана"], [24, "Китай"], [29, "Коста-Рика"], [32, "Куба"], [40, "Экв. Гвинея"], [66, "Япония"], [70, "Юж. Корея"], [95, "Норвегия"], [117, "ЮАР"]];
      
      var state = { norm: "raw", k: 3, seed: 1 };

      K.hint(
        root,
        "Сначала сырые единицы: ось жизни мертва. Переключи нормировку — мир перестроится. На K=2 меняй попытки и лови два конкурирующих разреза; силуэт в сводке подскажет, когда группы настоящие.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Кластеризация выполняется в выбранной метрике, а рисунок всегда показывает страны в осях «лог-ВВП × жизнь» для читаемости. Кольцами отмечены страны-диссиденты из текста. Алгоритм — k-средние с фиксированным по «попытке» стартом; всё считается в браузере.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Страны мира, раскрашенные найденными кластерами",
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

      function normalized() {
        var gs = PTS.map(function (p) { return p[0]; });
        var ls = PTS.map(function (p) { return p[1]; });
        function stats(v) {
          var m = v.reduce(function (a, b) { return a + b; }) / v.length;
          var s = Math.sqrt(v.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / (v.length - 1));
          return [m, s];
        }
        if (state.norm === "raw") {
          return PTS.map(function (p) { return [p[0], p[1]]; });
        }
        if (state.norm === "z") {
          var g = stats(gs), l = stats(ls);
          return PTS.map(function (p) { return [(p[0] - g[0]) / g[1], (p[1] - l[0]) / l[1]]; });
        }
        var logs = gs.map(function (v) { return Math.log(v); });
        var lg = stats(logs), l2 = stats(ls);
        return PTS.map(function (p) {
          return [(Math.log(p[0]) - lg[0]) / lg[1], (p[1] - l2[0]) / l2[1]];
        });
      }

      function kmeans(pts, k, seed) {
        var rnd = mulberry(1000 + seed * 77);
        var idx = [];
        while (idx.length < k) {
          var cand = Math.floor(rnd() * pts.length);
          if (idx.indexOf(cand) < 0) idx.push(cand);
        }
        var cents = idx.map(function (i) { return pts[i].slice(); });
        var labels = new Array(pts.length).fill(0);
        for (var it = 0; it < 60; it += 1) {
          for (var i = 0; i < pts.length; i += 1) {
            var best = 0, bd = Infinity;
            for (var j = 0; j < k; j += 1) {
              var dx = pts[i][0] - cents[j][0];
              var dy = pts[i][1] - cents[j][1];
              var d = dx * dx + dy * dy;
              if (d < bd) { bd = d; best = j; }
            }
            labels[i] = best;
          }
          for (var j2 = 0; j2 < k; j2 += 1) {
            var sx = 0, sy = 0, n = 0;
            for (var i2 = 0; i2 < pts.length; i2 += 1) {
              if (labels[i2] === j2) { sx += pts[i2][0]; sy += pts[i2][1]; n += 1; }
            }
            if (n) cents[j2] = [sx / n, sy / n];
          }
        }
        return labels;
      }

      function silhouette(pts, labels, k) {
        var n = pts.length;
        var total = 0;
        var borderline = 0;
        for (var i = 0; i < n; i += 1) {
          var sums = new Array(k).fill(0);
          var counts = new Array(k).fill(0);
          for (var j = 0; j < n; j += 1) {
            if (j === i) continue;
            var dx = pts[i][0] - pts[j][0];
            var dy = pts[i][1] - pts[j][1];
            var d = Math.sqrt(dx * dx + dy * dy);
            sums[labels[j]] += d;
            counts[labels[j]] += 1;
          }
          var a = counts[labels[i]] ? sums[labels[i]] / counts[labels[i]] : 0;
          var b = Infinity;
          for (var c = 0; c < k; c += 1) {
            if (c === labels[i] || !counts[c]) continue;
            b = Math.min(b, sums[c] / counts[c]);
          }
          var s = (b - a) / Math.max(a, b);
          total += s;
          if (s < 0.1) borderline += 1;
        }
        return { mean: total / n, borderline: borderline };
      }

      var chart = { x: 84, y: 40, w: 900, h: 430 };
      var PALETTE = [C.red, C.gold, C.green, C.blue, C.violet, C.muted];

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        var pts = normalized();
        var labels = kmeans(pts, state.k, state.seed);
        var sil = silhouette(pts, labels, state.k);

        // Order clusters by mean life so colours stay comparable.
        var meanLife = [];
        for (var j = 0; j < state.k; j += 1) {
          var s = 0, n = 0;
          for (var i = 0; i < PTS.length; i += 1) {
            if (labels[i] === j) { s += PTS[i][1]; n += 1; }
          }
          meanLife.push([j, n ? s / n : 0]);
        }
        meanLife.sort(function (a, b) { return a[1] - b[1]; });
        var recolor = {};
        meanLife.forEach(function (pair, pos) { recolor[pair[0]] = pos; });

        var LOG_MIN = Math.log(250), LOG_MAX = Math.log(52000);
        function sx(g) { return chart.x + (Math.log(g) - LOG_MIN) / (LOG_MAX - LOG_MIN) * chart.w; }
        function sy(l) { return chart.y + chart.h - (l - 38) / (85 - 38) * chart.h; }

        [500, 2000, 8000, 32000].forEach(function (g) {
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(sx(g), chart.y);
          ctx.lineTo(sx(g), chart.y + chart.h);
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "center";
          ctx.fillText((g / 1000).toString().replace(".", ",") + " тыс. $", sx(g), chart.y + chart.h + 16);
        });
        [40, 55, 70, 85].forEach(function (l) {
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(chart.x, sy(l));
          ctx.lineTo(chart.x + chart.w, sy(l));
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText(l + " лет", chart.x - 8, sy(l));
        });
        ctx.textAlign = "left";

        for (var i3 = 0; i3 < PTS.length; i3 += 1) {
          ctx.fillStyle = PALETTE[recolor[labels[i3]] % PALETTE.length];
          ctx.beginPath();
          ctx.arc(sx(PTS[i3][0]), sy(PTS[i3][1]), 4, 0, Math.PI * 2);
          ctx.fill();
        }
        var LABEL_SHIFT = {
          "Куба": [-14, 18], "Коста-Рика": [10, -12], "Юж. Корея": [-30, 16],
          "Норвегия": [-72, 14], "Япония": [10, -10]
        };
        MARKS.forEach(function (mk) {
          var i4 = mk[0];
          ctx.strokeStyle = C.ink;
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.arc(sx(PTS[i4][0]), sy(PTS[i4][1]), 7.5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.fillStyle = C.ink;
          var shift = LABEL_SHIFT[mk[1]] || [10, -8];
          ctx.fillText(mk[1], sx(PTS[i4][0]) + shift[0], sy(PTS[i4][1]) + shift[1]);
        });

        var sizes = [];
        for (var j3 = 0; j3 < state.k; j3 += 1) {
          sizes.push(labels.filter(function (l) { return l === j3; }).length);
        }
        sizes.sort(function (a, b) { return b - a; });

        output.set([
          {
            label: "средний силуэт",
            value: sil.mean.toFixed(3).replace(".", ","),
            color: sil.mean > 0.55 ? C.green : sil.mean > 0.45 ? C.gold : C.red,
          },
          { label: "размеры кластеров", value: sizes.join(" / ") },
          {
            label: "пограничников (s < 0,1)",
            value: String(sil.borderline),
            color: sil.borderline > 15 ? C.red : C.ink,
          },
        ]);
      }

      K.segmented(
        controls,
        {
          label: "Метрика",
          value: state.norm,
          options: [
            { value: "raw", label: "сырые единицы" },
            { value: "z", label: "сигмы" },
            { value: "logz", label: "лог-ВВП + сигмы" },
          ],
        },
        function (value) { state.norm = value; draw(); },
      );
      K.slider(
        controls,
        { label: "Число кластеров K", min: 2, max: 6, step: 1, value: state.k },
        function (value) { state.k = value; draw(); },
      );
      K.slider(
        controls,
        { label: "Попытка (случайный старт)", min: 1, max: 10, step: 1, value: state.seed },
        function (value) { state.seed = value; draw(); },
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
