// Lesson 44: clt-lab — average n draws from a chosen distribution and watch the mean turn into a bell (or not, for Cauchy).
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("clt-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var src = 0, n = 5;
      var means = [];

      // source distributions: sampler, mu, sigma (null = no finite variance)
      var SOURCES = [
        { name: "скошенное", mu: 1, sd: 1, draw: function () { return -Math.log(1 - Math.random()); } },
        { name: "равномерное", mu: 0.5, sd: 1 / Math.sqrt(12), draw: function () { return Math.random(); } },
        { name: "двугорбое", mu: 1, sd: 1.0, draw: function () { return Math.random() < 0.5 ? 0.15 * gauss() : 2 + 0.15 * gauss(); } },
        { name: "хвосты Коши", mu: null, sd: null, draw: function () { return Math.tan(Math.PI * (Math.random() - 0.5)); } },
      ];
      function gauss() { var u = 1 - Math.random(), v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

      function reset() { means = []; }
      function accumulate(k) {
        var s = SOURCES[src];
        for (var m = 0; m < k; m += 1) {
          var sum = 0; for (var i = 0; i < n; i += 1) sum += s.draw();
          means.push(sum / n);
        }
        draw();
      }

      K.hint(
        root,
        "Центральная предельная теорема живьём. Выберите исходное распределение — хоть сильно скошенное, хоть двугорбое — и усредняйте по n наблюдений. Гистограмма средних набирается из тысяч опытов и стягивается к колоколу. Но у распределения Коши нет конечной дисперсии, и никакое n не спасает.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Слева — форма исходного распределения. Справа — распределение выборочного среднего из n наблюдений. Красная кривая — нормаль N(μ, σ²/n), к которой ведёт ЦПТ. С ростом n колокол становится уже (как σ/√n). Для Коши дисперсия бесконечна, и приближения нет.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Исходное распределение и распределение среднего", onResize: draw, drag: false });

      function hist(data, lo, hi, bins) {
        var h = new Array(bins).fill(0), w = (hi - lo) / bins;
        for (var i = 0; i < data.length; i += 1) {
          var b = Math.floor((data[i] - lo) / w);
          if (b >= 0 && b < bins) h[b] += 1;
        }
        var area = data.length * w;
        return h.map(function (v) { return v / area; });
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var s = SOURCES[src];

        // left: source shape (sample many raw draws once)
        var raw = [];
        var rr = 0; for (; rr < 4000; rr += 1) raw.push(s.draw());
        var sx = 55, sy = 70, sw = 250, sh = 300;
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("исходное: " + s.name, sx, sy - 14);
        var rlo = s.name === "хвосты Коши" ? -6 : Math.min.apply(null, raw), rhi = s.name === "хвосты Коши" ? 6 : Math.max.apply(null, raw);
        var rh = hist(raw.map(function (v) { return Math.max(rlo, Math.min(rhi, v)); }), rlo, rhi, 40);
        var rmx = Math.max.apply(null, rh);
        ctx.fillStyle = "rgba(110,114,106,0.55)";
        for (var b = 0; b < 40; b += 1) { var hh = rh[b] / rmx * sh; ctx.fillRect(sx + b * sw / 40, sy + sh - hh, sw / 40 - 1, hh); }
        ctx.strokeStyle = C.line; ctx.strokeRect(sx, sy, sw, sh);

        // right: distribution of means
        var mx0 = 360, my = 70, mw = W - mx0 - 30, mh = 300;
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("распределение среднего (n = " + n + ")", mx0, my - 14);
        ctx.strokeStyle = C.line; ctx.strokeRect(mx0, my, mw, mh);
        if (means.length > 5) {
          var finite = s.sd !== null;
          var mlo, mhi;
          if (finite) { mlo = s.mu - 4 * s.sd / Math.sqrt(n); mhi = s.mu + 4 * s.sd / Math.sqrt(n); }
          else { mlo = -6; mhi = 6; }
          var clipped = means.map(function (v) { return Math.max(mlo, Math.min(mhi, v)); });
          var mh2 = hist(clipped, mlo, mhi, 44);
          var normPk = finite ? 1 / (s.sd / Math.sqrt(n) * Math.sqrt(2 * Math.PI)) : 0;
          var mmx = Math.max(Math.max.apply(null, mh2), normPk * 1.05);
          ctx.fillStyle = "rgba(49,95,140,0.72)";
          for (var k = 0; k < 44; k += 1) { var hk = mh2[k] / mmx * mh; ctx.fillRect(mx0 + k * mw / 44, my + mh - hk, mw / 44 - 1, hk); }
          if (finite) {
            ctx.strokeStyle = C.red; ctx.lineWidth = 2.2; ctx.beginPath();
            for (var t = 0; t <= 100; t += 1) {
              var xv = mlo + (mhi - mlo) * t / 100;
              var d = Math.exp(-((xv - s.mu) ** 2) / (2 * (s.sd / Math.sqrt(n)) ** 2)) / (s.sd / Math.sqrt(n) * Math.sqrt(2 * Math.PI));
              var px = mx0 + (xv - mlo) / (mhi - mlo) * mw, py = my + mh - d / mmx * mh;
              if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
          } else {
            ctx.fillStyle = C.red; ctx.textAlign = "center"; ctx.font = "13px PT Sans, sans-serif";
            ctx.fillText("дисперсия бесконечна — колокол не появляется", mx0 + mw / 2, my + 30);
          }
        } else {
          ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "13px PT Sans, sans-serif";
          ctx.fillText("нажмите «набрать средние»", mx0 + mw / 2, my + mh / 2);
        }

        var empSd = std(means);
        var theoSd = s.sd !== null ? s.sd / Math.sqrt(n) : null;
        output.set([
          { label: "Собрано средних", value: String(means.length), color: C.ink },
          { label: "Разброс средних (факт)", value: means.length > 5 ? empSd.toFixed(3) : "—", color: C.blue },
          { label: "Теория σ/√n", value: theoSd !== null ? theoSd.toFixed(3) : "∞ (нет дисперсии)", color: theoSd !== null ? C.red : C.gold },
        ]);
      }
      function std(a) { if (a.length < 2) return 0; var m = a.reduce(function (x, y) { return x + y; }, 0) / a.length; return Math.sqrt(a.reduce(function (x, y) { return x + (y - m) * (y - m); }, 0) / (a.length - 1)); }

      K.segmented(controls, { label: "Исходное распределение", value: 0, options: SOURCES.map(function (s, i) { return { label: s.name, value: i }; }) }, function (v) { src = v; reset(); draw(); });
      K.slider(controls, { label: "Сколько усредняем n", min: 1, max: 100, step: 1, value: 5, format: function (v) { return String(v); } }, function (v) { n = v; reset(); draw(); });
      var b1 = K.element("button", "kontur-int-segment", { type: "button", text: "набрать средние" });
      var b2 = K.element("button", "kontur-int-segment", { type: "button", text: "сброс" });
      b1.style.margin = b2.style.margin = "0 6px";
      b1.addEventListener("click", function () { accumulate(3000); });
      b2.addEventListener("click", function () { reset(); draw(); });
      controls.appendChild(b1); controls.appendChild(b2);

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
