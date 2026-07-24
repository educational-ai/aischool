// Lesson 42: bayes-lab — a natural-frequency medical test; move prevalence/sensitivity/specificity, watch the posterior.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("bayes-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var prev = 0.01, sens = 0.90, spec = 0.95;
      var GREEN = "#38735d", RED = "#b94a3b", GOLD = "#a57920", LBLUE = "#cdd7e0";

      K.hint(
        root,
        "Классическая ловушка базовой частоты. Возьмём 10 000 человек и хороший тест. Двигайте распространённость болезни и качество теста — и следите, какая доля людей с положительным ответом действительно больна. Для редкой болезни даже точный тест даёт много ложных тревог.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Каждая клетка — один человек из 10 000. Зелёные больны и получили «+» (верно), красные здоровы, но получили «+» (ложная тревога), золотые больны, но «−», голубые здоровы и «−». Ценность положительного теста P(болен | +) = зелёные / (зелёные + красные).",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "10 000 человек и результат теста", onResize: draw, drag: false });

      // offscreen 100x100 person grid
      var off = document.createElement("canvas"); off.width = 100; off.height = 100;
      var octx = off.getContext("2d");

      function compute() {
        var N = 10000, sick = Math.round(N * prev), healthy = N - sick;
        var tp = Math.round(sick * sens), fn = sick - tp;
        var fp = Math.round(healthy * (1 - spec)), tn = healthy - fp;
        return { sick: sick, healthy: healthy, tp: tp, fn: fn, fp: fp, tn: tn,
                 ppv: tp + fp > 0 ? tp / (tp + fp) : 0, npv: tn + fn > 0 ? tn / (tn + fn) : 0 };
      }
      function rgb(hex) { return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)]; }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var s = compute();

        // fill person grid (row-major: TP, FN, FP, TN)
        var id = octx.createImageData(100, 100), d = id.data;
        var cols = [rgb(GREEN), rgb(GOLD), rgb(RED), rgb(LBLUE)];
        var blocks = [s.tp, s.fn, s.fp, s.tn], bi = 0, left = blocks[0];
        for (var p = 0; p < 10000; p += 1) {
          while (left <= 0 && bi < 3) { bi += 1; left = blocks[bi]; }
          left -= 1;
          var o = p * 4, col = cols[bi];
          d[o] = col[0]; d[o + 1] = col[1]; d[o + 2] = col[2]; d[o + 3] = 255;
        }
        octx.putImageData(id, 0, 0);
        var gx = 50, gy = 60, gs = 320;
        ctx.imageSmoothingEnabled = false; ctx.drawImage(off, gx, gy, gs, gs);
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, gy, gs, gs);
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("10 000 человек", gx, gy - 14);

        // legend
        var lx = gx + gs + 40, ly = 70;
        var leg = [[GREEN, "больны, «+» верно", s.tp], [RED, "здоровы, «+» ложно", s.fp],
                   [GOLD, "больны, «−» пропуск", s.fn], [LBLUE, "здоровы, «−» верно", s.tn]];
        ctx.font = "13px PT Sans, sans-serif";
        for (var i = 0; i < 4; i += 1) {
          ctx.fillStyle = leg[i][0]; ctx.fillRect(lx, ly + i * 26, 16, 16);
          ctx.fillStyle = C.ink; ctx.textAlign = "left";
          ctx.fillText(leg[i][1] + ": " + leg[i][2], lx + 24, ly + i * 26 + 13);
        }

        // PPV highlight
        var py = ly + 4 * 26 + 22;
        ctx.fillStyle = C.ink; ctx.font = "15px PT Sans, sans-serif";
        ctx.fillText("После «+» болезнь у:", lx, py);
        ctx.font = "30px PT Sans, sans-serif"; ctx.fillStyle = RED;
        ctx.fillText((s.ppv * 100).toFixed(1) + "%", lx, py + 38);
        ctx.font = "12px PT Sans, sans-serif"; ctx.fillStyle = C.muted;
        ctx.fillText(s.tp + " из " + (s.tp + s.fp) + " положительных", lx, py + 60);

        // among positives bar
        var bx = lx, by = py + 78, bw = 250, bh = 20, tot = s.tp + s.fp;
        if (tot > 0) {
          ctx.fillStyle = GREEN; ctx.fillRect(bx, by, bw * s.tp / tot, bh);
          ctx.fillStyle = RED; ctx.fillRect(bx + bw * s.tp / tot, by, bw * s.fp / tot, bh);
        }
        ctx.strokeStyle = C.line; ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("среди всех «+»: зелёные — настоящие больные", bx, by + bh + 15);

        output.set([
          { label: "Ценность «+» (P болен | +)", value: (s.ppv * 100).toFixed(1) + "%", color: s.ppv > 0.5 ? C.green : RED },
          { label: "Ложных тревог на 1 больного", value: s.tp > 0 ? (s.fp / s.tp).toFixed(1) : "∞", color: C.gold },
          { label: "Ценность «−» (P здоров | −)", value: (s.npv * 100).toFixed(1) + "%", color: C.blue },
        ]);
      }

      K.slider(controls, { label: "Распространённость болезни", min: 0.1, max: 30, step: 0.1, value: 1, format: function (v) { return v.toFixed(1) + "%"; } }, function (v) { prev = v / 100; draw(); });
      K.slider(controls, { label: "Чувствительность (находит больных)", min: 50, max: 100, step: 1, value: 90, format: function (v) { return v + "%"; } }, function (v) { sens = v / 100; draw(); });
      K.slider(controls, { label: "Специфичность (не пугает здоровых)", min: 50, max: 100, step: 0.5, value: 95, format: function (v) { return v + "%"; } }, function (v) { spec = v / 100; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
