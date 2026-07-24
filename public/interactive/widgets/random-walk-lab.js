// Lesson 66: random-walk-lab — 300 walks at once; change n, p and the normalisation
// and watch the sqrt(n) law, the drift beating the noise, and the collapse under S_n/n.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("random-walk-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var ML = 70, MR = 200, MT = 26, MB = 46;
      var PLOTW = W - ML - MR, PLOTH = H - MT - MB;
      var TRIALS = 300, SHOW = 10, CK = 200;
      var n = 1600, p = 0.5, mode = "raw";
      var sim = null;

      var seed = 0;
      function rnd() {
        seed = (seed + 0x6d2b79f5) | 0;
        var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      }

      function simulate() {
        seed = 66066;
        var marks = new Int32Array(CK + 1), j;
        for (j = 0; j <= CK; j += 1) marks[j] = Math.round((j * n) / CK);
        var paths = [], ends = new Float64Array(TRIALS);
        var sum = 0, sum2 = 0, w, k, idx, s, path;
        for (w = 0; w < TRIALS; w += 1) {
          s = 0; idx = 1;
          path = w < SHOW ? new Float64Array(CK + 1) : null;
          for (k = 1; k <= n; k += 1) {
            s += rnd() < p ? 1 : -1;
            while (idx <= CK && marks[idx] === k) {
              if (path) path[idx] = s;
              idx += 1;
            }
          }
          if (path) paths.push(path);
          ends[w] = s; sum += s; sum2 += s * s;
        }
        var mean = sum / TRIALS;
        var sd = Math.sqrt(Math.max(0, sum2 / TRIALS - mean * mean));
        var muT = n * (2 * p - 1);
        var sdT = 2 * Math.sqrt(n * p * (1 - p));
        var out = 0;
        for (w = 0; w < TRIALS; w += 1) if (Math.abs(ends[w] - muT) > 2 * sdT) out += 1;
        return {
          marks: marks, paths: paths, ends: ends, mean: mean, sd: sd,
          muT: muT, sdT: sdT, outside: out / TRIALS
        };
      }

      function scaleOf() {
        if (mode === "sqrt") return Math.sqrt(n);
        if (mode === "n") return n;
        return 1;
      }

      function yRange() {
        var d = scaleOf();
        var c = sim.muT / d, h;
        if (mode === "n") h = 0.42;
        else if (mode === "sqrt") h = 4.2;
        else h = 4.2 * sim.sdT + 1;
        return [c - h, c + h];
      }

      K.hint(root, "Здесь 300 независимых блужданий бросают монету одновременно. Ползунок n меняет длину пути, ползунок p — перекос монеты. Переключатель нормировки решает главное: делите на √n — облако имеет постоянную ширину при любом n; делите на n — всё схлопывается в точку. Именно поэтому масштаб случайности корневой, а не линейный.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева — десять траекторий и коридор ±2σ вокруг сноса n(2p−1); справа — гистограмма 300 конечных положений. Теоретические σ и снос считаются по формулам урока, опытные — по этим же 300 траекториям: расхождение показывает цену конечной выборки.");
      var cs = K.makeCanvas(stage, W, H, { label: "Ансамбль случайных блужданий, коридор двух сигм и гистограмма конечных точек", onResize: draw, drag: false });

      function draw() {
        if (!sim) sim = simulate();
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var d = scaleOf(), r = yRange(), lo = r[0], hi = r[1];
        function px(k) { return ML + (k / n) * PLOTW; }
        function py(v) { return MT + PLOTH * (1 - (v - lo) / (hi - lo)); }

        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        var t, ticks = 6;
        for (t = 0; t <= ticks; t += 1) {
          var v = lo + ((hi - lo) * t) / ticks, y = py(v);
          ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(ML + PLOTW, y); ctx.stroke();
          ctx.fillText(Math.abs(v) < 1000 ? v.toFixed(mode === "n" ? 2 : 1) : v.toFixed(0), ML - 8, y + 4);
        }
        ctx.textAlign = "center";
        for (t = 0; t <= 4; t += 1) {
          var kk = (n * t) / 4;
          ctx.fillText(String(Math.round(kk)), px(kk), MT + PLOTH + 20);
        }
        ctx.fillText("шаг k", ML + PLOTW / 2, MT + PLOTH + 40);

        // коридор ±2σ и снос
        ctx.beginPath();
        var k1;
        for (k1 = 0; k1 <= 120; k1 += 1) {
          var kx = (n * k1) / 120;
          var mu = kx * (2 * p - 1), sg = 2 * Math.sqrt(kx * p * (1 - p));
          ctx.lineTo(px(kx), py((mu + 2 * sg) / d));
        }
        for (k1 = 120; k1 >= 0; k1 -= 1) {
          var kx2 = (n * k1) / 120;
          var mu2 = kx2 * (2 * p - 1), sg2 = 2 * Math.sqrt(kx2 * p * (1 - p));
          ctx.lineTo(px(kx2), py((mu2 - 2 * sg2) / d));
        }
        ctx.closePath();
        ctx.fillStyle = "rgba(49,95,140,0.10)"; ctx.fill();

        ctx.strokeStyle = C.blue; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (k1 = 0; k1 <= 120; k1 += 1) {
          var kx3 = (n * k1) / 120, sg3 = 2 * Math.sqrt(kx3 * p * (1 - p));
          ctx.lineTo(px(kx3), py((kx3 * (2 * p - 1) + sg3) / d));
        }
        ctx.stroke();
        ctx.beginPath();
        for (k1 = 0; k1 <= 120; k1 += 1) {
          var kx4 = (n * k1) / 120, sg4 = 2 * Math.sqrt(kx4 * p * (1 - p));
          ctx.lineTo(px(kx4), py((kx4 * (2 * p - 1) - sg4) / d));
        }
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.strokeStyle = C.red; ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(px(0), py(0));
        ctx.lineTo(px(n), py(sim.muT / d));
        ctx.stroke();

        var palette = [C.ink, C.blue, C.green, C.gold, C.violet, C.red, C.muted, C.blue, C.green, C.gold];
        var i, jj;
        ctx.lineWidth = 0.9;
        for (i = 0; i < sim.paths.length; i += 1) {
          ctx.strokeStyle = palette[i % palette.length];
          ctx.globalAlpha = 0.75;
          ctx.beginPath();
          for (jj = 0; jj <= CK; jj += 1) ctx.lineTo(px(sim.marks[jj]), py(sim.paths[i][jj] / d));
          ctx.stroke();
        }
        ctx.globalAlpha = 1;

        // гистограмма конечных положений
        var BX = ML + PLOTW + 16, BW = MR - 40, bins = 34;
        var counts = new Array(bins).fill(0), mx = 0;
        for (i = 0; i < TRIALS; i += 1) {
          var vv = sim.ends[i] / d;
          var b = Math.floor(((vv - lo) / (hi - lo)) * bins);
          if (b >= 0 && b < bins) counts[b] += 1;
        }
        for (i = 0; i < bins; i += 1) if (counts[i] > mx) mx = counts[i];
        var bh = PLOTH / bins;
        ctx.fillStyle = "rgba(49,95,140,0.55)";
        for (i = 0; i < bins; i += 1) {
          if (!counts[i]) continue;
          var wdt = mx ? (counts[i] / mx) * BW : 0;
          ctx.fillRect(BX, MT + PLOTH - (i + 1) * bh + 1, wdt, bh - 1.5);
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("300 конечных точек", BX, MT - 8);
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(BX, py(sim.muT / d)); ctx.lineTo(BX + BW, py(sim.muT / d)); ctx.stroke();

        ctx.fillStyle = C.muted; ctx.save();
        ctx.translate(18, MT + PLOTH / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText(mode === "raw" ? "S_k" : mode === "sqrt" ? "S_k / √n" : "S_k / n", 0, 0);
        ctx.restore();

        var gap = Math.abs(2 * p - 1);
        var fair = gap < 1e-9;
        var cross = fair ? 0 : Math.ceil(Math.pow((4 * Math.sqrt(p * (1 - p))) / gap, 2));
        output.set([
          { label: "теория: снос n(2p−1)", value: sim.muT.toFixed(1), color: C.red },
          { label: "опыт: среднее конечных точек", value: sim.mean.toFixed(1), color: C.red },
          { label: "теория: σ = 2√(np(1−p))", value: sim.sdT.toFixed(1), color: C.blue },
          { label: "опыт: sd конечных точек", value: sim.sd.toFixed(1), color: C.blue },
          { label: "σ/√n", value: (sim.sd / Math.sqrt(n)).toFixed(3), color: C.green },
          { label: "вне ±2σ (теория 4.55%)", value: (100 * sim.outside).toFixed(1) + "%", color: C.gold },
          { label: "снос обгонит 2σ при n ≥", value: fair ? "никогда (p = 0.5)" : String(cross), color: C.violet }
        ]);
      }

      K.segmented(controls, {
        label: "Нормировка", value: "raw",
        options: [
          { label: "как есть S", value: "raw" },
          { label: "делить на √n", value: "sqrt" },
          { label: "делить на n", value: "n" }
        ]
      }, function (v) { mode = v; draw(); });
      K.slider(controls, {
        label: "Число шагов n", min: 100, max: 6400, step: 100, value: n,
        format: function (v) { return String(v); }
      }, function (v) { n = v; sim = simulate(); draw(); });
      K.slider(controls, {
        label: "Вероятность шага вправо p", min: 0.44, max: 0.56, step: 0.005, value: p,
        format: function (v) { return v.toFixed(3); }
      }, function (v) { p = v; sim = simulate(); draw(); });

      sim = simulate();
      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
