// Lesson 57: cost-loss-lab — one fixed model, changing price list.
// Move the two costs and watch the optimal threshold slide; break calibration and
// watch the ranking (AUC) survive while the money decision falls apart.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("cost-loss-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 500;
      var PB = "#fffef9";
      var L = 74, R = 856, TOP = 46, BOT = 318;          // рамка графика цены
      var RUG_TOP = 372, RUG_BOT = 460;                   // полоса объектов

      // ---------- детерминированная выборка: score калиброван по построению ----------
      var N = 600;
      var base = [];
      (function build() {
        var seed = 20240057;
        function rnd() {                                  // LCG, фиксированное зерно
          seed = (1103515245 * seed + 12345) % 2147483648;
          return seed / 2147483648;
        }
        for (var i = 0; i < N; i += 1) {
          var u = rnd();
          var p = u * u * u;                              // редкое событие: мало больших p
          base.push({ p: p, y: rnd() < p ? 1 : 0 });
        }
      })();

      var state = { cfp: 1, cfn: 10, t: 0.5, gamma: 1 };

      function reported(p) {
        return Math.pow(p, state.gamma);
      }

      function stats(t) {
        var fp = 0, fn = 0, tp = 0, tn = 0;
        for (var i = 0; i < N; i += 1) {
          var a = reported(base[i].p) >= t ? 1 : 0, y = base[i].y;
          if (a === 1 && y === 1) tp += 1;
          else if (a === 1 && y === 0) fp += 1;
          else if (a === 0 && y === 1) fn += 1;
          else tn += 1;
        }
        return { fp: fp, fn: fn, tp: tp, tn: tn, cost: state.cfp * fp + state.cfn * fn };
      }

      var GRIDN = 241;
      function curve() {
        var xs = [], ys = [], best = 0, bestCost = Infinity, maxCost = 0;
        for (var k = 0; k < GRIDN; k += 1) {
          var t = k / (GRIDN - 1);
          var c = stats(t).cost;
          xs.push(t); ys.push(c);
          if (c < bestCost) { bestCost = c; best = t; }
          if (c > maxCost) maxCost = c;
        }
        return { xs: xs, ys: ys, best: best, bestCost: bestCost, maxCost: maxCost };
      }

      function auc() {                                    // ранги: инвариант монотонных искажений
        var arr = base.map(function (o) { return { s: reported(o.p), y: o.y }; });
        arr.sort(function (a, b) { return a.s - b.s; });
        var rank = new Array(arr.length), i = 0;
        while (i < arr.length) {
          var j = i;
          while (j + 1 < arr.length && arr[j + 1].s === arr[i].s) j += 1;
          var mean = (i + j) / 2 + 1;
          for (var k = i; k <= j; k += 1) rank[k] = mean;
          i = j + 1;
        }
        var n1 = 0, sum = 0;
        for (var m = 0; m < arr.length; m += 1) if (arr[m].y === 1) { n1 += 1; sum += rank[m]; }
        var n0 = arr.length - n1;
        if (n0 === 0 || n1 === 0) return 0.5;
        return (sum - n1 * (n1 + 1) / 2) / (n0 * n1);
      }

      K.hint(root, "Модель здесь не меняется ни разу: список её оценок фиксирован. Двигаются только цены двух ошибок — и вместе с ними едет оптимальный порог. Затем сломайте калибровку: порядок объектов (AUC) не шелохнётся, а решение по деньгам подорожает.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Синяя кривая — суммарная стоимость ошибок на всей выборке при пороге t. Зелёная штриховая — теоретический порог c_FP/(c_FP+c_FN), красная точка — эмпирический минимум, золотая — ваш текущий порог. Внизу каждая точка — объект: вверху спам (y=1), внизу обычные (y=0), по горизонтали — объявленная моделью вероятность.");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Кривая стоимости по порогу и разметка объектов",
        onResize: draw,
        drag: false,
      });

      function xmap(t) { return L + t * (R - L); }
      function ymap(c, maxC) { return BOT - (c / maxC) * (BOT - TOP); }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var cv = curve();
        var maxC = Math.max(cv.maxCost, 1);
        var tTheory = state.cfp / (state.cfp + state.cfn);
        var here = stats(state.t);
        var atTheory = stats(tTheory).cost;

        // сетка и оси
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var g = 0; g <= 10; g += 1) {
          var gx = xmap(g / 10);
          ctx.moveTo(gx, TOP); ctx.lineTo(gx, BOT);
        }
        ctx.stroke();
        ctx.strokeStyle = C.muted || "#6e726a";
        ctx.beginPath();
        ctx.moveTo(L, TOP); ctx.lineTo(L, BOT); ctx.lineTo(R, BOT);
        ctx.stroke();

        ctx.fillStyle = C.muted || "#6e726a";
        ctx.textAlign = "center";
        for (var q = 0; q <= 10; q += 2) {
          ctx.fillText((q / 10).toFixed(1), xmap(q / 10), BOT + 18);
        }
        ctx.fillText("порог t по объявленной вероятности", (L + R) / 2, BOT + 38);
        ctx.save();
        ctx.translate(22, (TOP + BOT) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("стоимость ошибок, у.е.", 0, 0);
        ctx.restore();
        ctx.textAlign = "right";
        ctx.fillText(String(Math.round(maxC)), L - 8, TOP + 5);
        ctx.fillText("0", L - 8, BOT + 4);

        // кривая стоимости
        ctx.strokeStyle = C.blue || "#315f8c";
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        for (var i = 0; i < cv.xs.length; i += 1) {
          var px = xmap(cv.xs[i]), py = ymap(cv.ys[i], maxC);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // теоретический порог
        ctx.strokeStyle = C.green || "#38735d";
        ctx.lineWidth = 1.8;
        ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(xmap(tTheory), TOP); ctx.lineTo(xmap(tTheory), BOT); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.green || "#38735d";
        ctx.textAlign = "left";
        ctx.fillText("теория " + tTheory.toFixed(3), xmap(tTheory) + 6, TOP + 14);

        // эмпирический минимум
        ctx.fillStyle = C.red || "#b94a3b";
        ctx.beginPath();
        ctx.arc(xmap(cv.best), ymap(cv.bestCost, maxC), 6.5, 0, 7);
        ctx.fill();
        ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.fillText("минимум " + cv.best.toFixed(3), xmap(cv.best) + 10, ymap(cv.bestCost, maxC) - 8);

        // текущий порог
        ctx.strokeStyle = C.gold || "#a57920";
        ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(xmap(state.t), TOP); ctx.lineTo(xmap(state.t), RUG_BOT); ctx.stroke();

        // полоса объектов
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(L, (RUG_TOP + RUG_BOT) / 2); ctx.lineTo(R, (RUG_TOP + RUG_BOT) / 2);
        ctx.stroke();
        for (var k2 = 0; k2 < N; k2 += 1) {
          var o = base[k2];
          var s = reported(o.p);
          var yy = o.y === 1
            ? RUG_TOP + 6 + ((k2 * 37) % 26)
            : RUG_BOT - 6 - ((k2 * 53) % 26);
          var decided = s >= state.t ? 1 : 0;
          var wrong = decided !== o.y;
          ctx.fillStyle = o.y === 1 ? (C.red || "#b94a3b") : (C.blue || "#315f8c");
          ctx.globalAlpha = wrong ? 1 : 0.28;
          ctx.beginPath();
          ctx.arc(xmap(s), yy, wrong ? 3.2 : 2.2, 0, 7);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = C.muted || "#6e726a";
        ctx.textAlign = "left";
        ctx.fillText("y = 1 (событие)", L, RUG_TOP - 4);
        ctx.fillText("y = 0 (фон)", L, RUG_BOT + 16);
        ctx.textAlign = "right";
        ctx.fillText("ярко — ошибка при текущем пороге", R, RUG_BOT + 16);

        output.set([
          { label: "Теоретический порог", value: tTheory.toFixed(3), color: C.green },
          { label: "Эмпирический оптимум", value: cv.best.toFixed(3), color: C.red },
          { label: "Цена при вашем t", value: String(Math.round(here.cost)), color: C.gold },
          { label: "Цена при теории", value: String(Math.round(atTheory)), color: C.green },
          { label: "Ложных тревог / пропусков", value: here.fp + " / " + here.fn, color: C.blue },
          { label: "AUC (ранжирование)", value: auc().toFixed(4), color: C.blue },
        ]);
      }

      K.slider(controls, {
        label: "Цена ложной тревоги c_FP", min: 1, max: 40, step: 1, value: state.cfp,
        format: function (v) { return String(v); },
      }, function (v) { state.cfp = v; draw(); });

      K.slider(controls, {
        label: "Цена пропуска c_FN", min: 1, max: 40, step: 1, value: state.cfn,
        format: function (v) { return String(v); },
      }, function (v) { state.cfn = v; draw(); });

      K.slider(controls, {
        label: "Ваш порог t", min: 0, max: 1, step: 0.005, value: state.t,
        format: function (v) { return v.toFixed(3); },
      }, function (v) { state.t = v; draw(); });

      K.segmented(controls, {
        label: "Калибровка модели",
        value: "fair",
        options: [
          { value: "fair", label: "честная" },
          { value: "under", label: "занижает (p²)" },
          { value: "over", label: "завышает (√p)" },
        ],
      }, function (v) {
        state.gamma = v === "fair" ? 1 : (v === "under" ? 2 : 0.5);
        draw();
      });

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
