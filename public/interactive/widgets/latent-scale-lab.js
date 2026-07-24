// Lesson 83: latent-scale-lab — mix two latent codes at a chosen cutoff and watch which
// measurable feature is inherited from which source; then compare a directed perturbation
// with random noise of exactly the same l-infinity budget.
(function () {
  "use strict";

  // coarse -> fine ordering of separable cosine modes of an 8x8 image
  var MODES = [[0, 1], [1, 0], [1, 1], [2, 0], [0, 2], [2, 2], [3, 2], [2, 3], [3, 3], [4, 4]];
  var N = 8, DIM = MODES.length;

  function basis(j) {
    var u = MODES[j][0], v = MODES[j][1], out = new Float64Array(N * N);
    for (var y = 0; y < N; y += 1) {
      for (var x = 0; x < N; x += 1) {
        out[y * N + x] = Math.cos(Math.PI * (x + 0.5) * u / N) * Math.cos(Math.PI * (y + 0.5) * v / N);
      }
    }
    return out;
  }
  var BASIS = [], AMP = [];
  for (var j = 0; j < DIM; j += 1) { BASIS.push(basis(j)); AMP.push(6.4 / (1 + 0.45 * j)); }

  function mulberry(seed) {
    var a = seed >>> 0;
    return function () {
      a += 0x6D2B79F5;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("latent-scale-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 430;
      var mode = 0, cut = 3, eps = 0.12;
      var seedA = 83, seedB = 191;
      var wA, wB, signs;

      function codeFrom(seed) {
        var rnd = mulberry(seed), w = [];
        for (var i = 0; i < DIM; i += 1) w.push((rnd() * 2 - 1) * 1.15);
        return w;
      }
      function render(w) {
        var img = new Float64Array(N * N);
        for (var p = 0; p < N * N; p += 1) {
          var s = 8.0;
          for (var i = 0; i < DIM; i += 1) s += w[i] * AMP[i] * BASIS[i][p];
          img[p] = Math.max(0, Math.min(16, s));
        }
        return img;
      }
      function mix(c) {
        var w = [];
        for (var i = 0; i < DIM; i += 1) w.push(i < c ? wA[i] : wB[i]);
        return w;
      }
      function fCoarse(img) {   // horizontal moment: the largest scale
        var s = 0;
        for (var y = 0; y < N; y += 1) for (var x = 0; x < N; x += 1) s += (x - 3.5) * img[y * N + x];
        return s / (N * N);
      }
      function fFine(img) {     // checkerboard mode: the smallest scale
        var s = 0;
        for (var y = 0; y < N; y += 1) for (var x = 0; x < N; x += 1) s += ((x + y) % 2 ? -1 : 1) * img[y * N + x];
        return s / (N * N);
      }
      function share(f, m, a, b) {
        var den = f(a) - f(b);
        if (Math.abs(den) < 1e-9) return 0;
        return (f(m) - f(b)) / den * 100;
      }

      // linear score used for the attack half: s(x) = v . x, |v| = 1
      var V = (function () {
        var v = BASIS[2].slice(), nrm = 0, i;
        for (i = 0; i < v.length; i += 1) nrm += v[i] * v[i];
        nrm = Math.sqrt(nrm);
        for (i = 0; i < v.length; i += 1) v[i] /= nrm;
        return v;
      })();
      function score(img) {
        var s = 0;
        for (var i = 0; i < img.length; i += 1) s += V[i] * img[i] / 16;
        return s;
      }
      function perturb(img, dir) {
        var out = new Float64Array(img.length);
        for (var i = 0; i < img.length; i += 1) {
          out[i] = Math.max(0, Math.min(16, img[i] + 16 * eps * dir[i]));
        }
        return out;
      }
      function newSigns() {
        var rnd = mulberry(seedA * 7 + seedB), s = new Float64Array(N * N);
        for (var i = 0; i < s.length; i += 1) s[i] = rnd() < 0.5 ? -1 : 1;
        signs = s;
      }
      function reseed() {
        wA = codeFrom(seedA); wB = codeFrom(seedB); newSigns();
      }
      reseed();

      K.hint(root, "Слева — два кода одного и того же генератора. Ползунок «граница смешения» решает, сколько РАННИХ координат взято у источника A: остальные приходят от B. Следите за двумя измеримыми признаками — крупным (горизонтальный момент) и мелким (шахматная мода): крупный переходит к A почти сразу, мелкий держится за B до самого конца. Во второй вкладке к картинке добавляется возмущение с одинаковым бюджетом по каждому пикселю: направленное вдоль весов и случайное. Норма одна, сдвиг решения — разный.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Смешение кодов: w = (w_A[0..c), w_B[c..]). Доля признака, унаследованная от A, равна (f(mix) − f(B)) / (f(A) − f(B)). Атака: при бюджете ε по норме l∞ направленное возмущение даёт сдвиг ε·Σ|v_i|, случайное — порядка ε·‖v‖, то есть в √d раз меньше по модулю ожидания.");

      var cs = K.makeCanvas(stage, W, H, { label: "Смешение латентных кодов и направленное возмущение", onResize: draw, drag: false });

      K.segmented(controls, {
        label: "Что изучаем",
        value: 0,
        options: [{ label: "смешение масштабов", value: 0 }, { label: "направленное возмущение", value: 1 }],
      }, function (v) { mode = v; draw(); });

      K.slider(controls, {
        label: "граница смешения c", min: 0, max: DIM, step: 1, value: cut,
        format: function (n) { return String(Math.round(n)); },
      }, function (v) { cut = Math.round(v); draw(); });

      K.slider(controls, {
        label: "бюджет ε (доля шкалы яркости)", min: 0, max: 0.3, step: 0.01, value: eps,
        format: function (n) { return n.toFixed(2); },
      }, function (v) { eps = v; draw(); });

      var btn = K.element("button", "kontur-int-segment", { type: "button", text: "новые источники" });
      btn.style.margin = "0 6px";
      btn.addEventListener("click", function () {
        seedA = (seedA * 7 + 13) % 9973; seedB = (seedB * 5 + 29) % 9973;
        reseed(); draw();
      });
      controls.appendChild(btn);

      function drawImage(ctx, img, x0, y0, cell, title, color) {
        for (var y = 0; y < N; y += 1) {
          for (var x = 0; x < N; x += 1) {
            var t = Math.max(0, Math.min(1, img[y * N + x] / 16));
            var g = Math.round(255 - t * 235);
            ctx.fillStyle = "rgb(" + g + "," + g + "," + g + ")";
            ctx.fillRect(x0 + x * cell, y0 + y * cell, cell, cell);
          }
        }
        ctx.strokeStyle = "#c9c8be"; ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, N * cell - 1, N * cell - 1);
        if (title) {
          ctx.fillStyle = color || C.muted;
          ctx.textAlign = "center";
          ctx.font = "13px PT Sans, sans-serif";
          ctx.fillText(title, x0 + N * cell / 2, y0 - 10);
        }
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.textBaseline = "alphabetic";
        var cell = 26, size = N * cell;
        var imgA = render(wA), imgB = render(wB);

        if (mode === 0) {
          var imgM = render(mix(cut));
          drawImage(ctx, imgB, 60, 90, cell, "источник B (поздние координаты)", C.green);
          drawImage(ctx, imgM, 60 + size + 90, 90, cell, "смесь при c = " + cut, C.red);
          drawImage(ctx, imgA, 60 + 2 * (size + 90), 90, cell, "источник A (ранние координаты)", C.blue);
          var sC = share(fCoarse, imgM, imgA, imgB), sF = share(fFine, imgM, imgA, imgB);
          // scale bar: which coordinates came from A
          var bx = 60, bw = 2 * (size + 90) + size;
          for (var i = 0; i < DIM; i += 1) {
            ctx.fillStyle = i < cut ? C.blue : C.green;
            ctx.fillRect(bx + i * (bw / DIM) + 2, H - 62, bw / DIM - 4, 16);
          }
          ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
          ctx.fillText("координаты латента: слева крупный масштаб, справа мелкий", bx, H - 70);
          ctx.textAlign = "right";
          ctx.fillText("синие взяты у A, зелёные у B", bx + bw, H - 70);
          output.set([
            { label: "крупный признак от A", value: sC.toFixed(0) + " %", color: C.blue },
            { label: "мелкий признак от A", value: sF.toFixed(0) + " %", color: C.red },
            { label: "координат от A", value: cut + " из " + DIM, color: C.muted }
          ]);
        } else {
          var dirSign = new Float64Array(N * N), i2;
          for (i2 = 0; i2 < dirSign.length; i2 += 1) dirSign[i2] = V[i2] >= 0 ? 1 : -1;
          var base = imgA;
          var adv = perturb(base, dirSign), rnd = perturb(base, signs);
          drawImage(ctx, base, 60, 90, cell, "исходная картинка", C.muted);
          drawImage(ctx, adv, 60 + size + 90, 90, cell, "направленное возмущение", C.red);
          drawImage(ctx, rnd, 60 + 2 * (size + 90), 90, cell, "случайный шум той же нормы", C.green);
          var s0 = score(base), sa = score(adv), sr = score(rnd);
          ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
          ctx.fillText("каждый пиксель сдвинут не более чем на ε·16 уровней яркости в обоих случаях", 60, H - 62);
          output.set([
            { label: "сдвиг оценки: направленно", value: (sa - s0).toFixed(3), color: C.red },
            { label: "сдвиг оценки: случайно", value: (sr - s0).toFixed(3), color: C.green },
            { label: "во сколько раз больше", value: Math.abs(sr - s0) > 1e-6 ? (Math.abs(sa - s0) / Math.abs(sr - s0)).toFixed(1) + "×" : "—", color: C.blue }
          ]);
        }
        ctx.fillStyle = C.ink || "#171915";
        ctx.textAlign = "center"; ctx.font = "15px PT Sans, sans-serif";
        ctx.fillText(mode === 0
          ? "Ранние координаты правят геометрию, поздние — мелкую фактуру"
          : "Одинаковый бюджет по норме, совершенно разный сдвиг решения", W / 2, 40);
      }

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
