// Lesson 72: self-play-league-lab — best-response self-play in Rock-Paper-Scissors-Lizard-Spock.
// If the new version best-responds only to the previous one (archive of size 1), the champion
// spins around a cycle and the whole run stays exploitable. Let it best-respond to a mixture of
// past versions and the league average slides down to the equilibrium. Evaluation noise (games
// per match) shows what happens when the champion is chosen from a short, noisy series.
(function () {
  "use strict";

  var LABELS = ["камень", "бумага", "ножницы", "ящерица", "Спок"];
  // BEATS[i] — два действия, которые побеждает действие i
  var BEATS = [[2, 3], [0, 4], [1, 3], [4, 1], [2, 0]];

  function payoff() {
    var A = [], i, j;
    for (i = 0; i < 5; i += 1) A.push([0, 0, 0, 0, 0]);
    for (i = 0; i < 5; i += 1) {
      for (j = 0; j < BEATS[i].length; j += 1) {
        A[i][BEATS[i][j]] = 1;
        A[BEATS[i][j]][i] = -1;
      }
    }
    return A;
  }

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("self-play-league-lab", function (root, options, K) {
      var C = K.COLORS;
      var INK = C.ink || "#171915";
      var W = 900, H = 440;
      var A = payoff();

      var archive = 1;     // сколько последних версий входит в цель обучения
      var games = 400;     // партий на оценку одного действия
      var eps = 0.10;      // доля равномерного шума в новой версии
      var panel = 0;       // 0 — состав смеси лиги, 1 — лента чемпионов
      var seed0 = 20720721;

      var seed, hist, champs, sum, total, curveLast, curveMix, t, timer = null;

      function rnd() {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        if (seed < 0) seed += 2147483648;
        return seed / 2147483648;
      }
      function gauss() {
        var u = Math.max(rnd(), 1e-12), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      function mul(q) {
        var r = [], a, s, j;
        for (a = 0; a < 5; a += 1) {
          s = 0;
          for (j = 0; j < 5; j += 1) s += A[a][j] * q[j];
          r.push(s);
        }
        return r;
      }
      function windowMix() {
        var m = [0, 0, 0, 0, 0], from = Math.max(0, hist.length - archive), i, j, k = 0;
        for (i = from; i < hist.length; i += 1) { for (j = 0; j < 5; j += 1) m[j] += hist[i][j]; k += 1; }
        for (j = 0; j < 5; j += 1) m[j] /= k;
        return m;
      }
      function leagueMix() {
        var m = [], j;
        for (j = 0; j < 5; j += 1) m.push(sum[j] / total);
        return m;
      }
      function exploit(p) {                    // значение симметричной игры равно нулю
        var s = mul(p), m = -9, a;
        for (a = 0; a < 5; a += 1) if (s[a] > m) m = s[a];
        return m;
      }

      function reset() {
        seed = seed0;
        var start = [1, 0, 0, 0, 0];
        hist = [start];
        champs = [0];
        sum = [1, 0, 0, 0, 0];
        total = 1;
        t = 0;
        curveLast = [exploit(start)];
        curveMix = [exploit(start)];
      }

      function step() {
        var target = windowMix(), mu = mul(target), a, best = 0, bestVal = -9, est, varOne;
        for (a = 0; a < 5; a += 1) {
          // одна партия даёт -1/0/+1; её дисперсия равна (1 - q_a) - mu_a^2
          varOne = Math.max((1 - target[a]) - mu[a] * mu[a], 0);
          est = mu[a] + Math.sqrt(varOne / games) * gauss();
          if (est > bestVal) { bestVal = est; best = a; }
        }
        var next = [], j;
        for (j = 0; j < 5; j += 1) next.push(eps / 5 + (j === best ? 1 - eps : 0));
        hist.push(next);
        champs.push(best);
        for (j = 0; j < 5; j += 1) sum[j] += next[j];
        total += 1;
        t += 1;
        curveLast.push(exploit(next));
        curveMix.push(exploit(leagueMix()));
        if (curveLast.length > 500) { curveLast.shift(); curveMix.shift(); }
        if (champs.length > 400) champs.shift();
        if (hist.length > archive + 400) hist.splice(0, 200);
      }

      K.hint(root, "Каждая итерация — новая версия агента, обученная как лучший ответ на цель. Поставьте размер архива в 1: цель — только предыдущая версия, это игра с зеркалом, чемпион вечно бегает по кругу «камень → бумага → ножницы», и красная линия эксплуатируемости не падает. Увеличьте архив: цель становится смесью прошлых версий, и синяя линия — эксплуатируемость средней стратегии всей лиги — уходит вниз. Уменьшите число партий на оценку до нескольких десятков: чемпиона начинает выбирать шум, а не сила.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Эксплуатируемость — сколько очков отберёт у стратегии наилучший ответ на неё; 0 означает равновесие, 1 — полностью читаемую игру. Красная линия относится к последней версии, синяя — к средней стратегии всего прогона (это и есть «лига»). Игра: камень-ножницы-бумага-ящерица-Спок, у каждого действия ровно два побеждённых и два победителя, равновесие — равномерная смесь по 0,2.");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Эксплуатируемость последней версии и смеси лиги; состав смеси или лента чемпионов",
        onResize: draw, drag: false
      });

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var L = 62, R = 545, T = 52, B = 380;

        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        for (i = 0; i <= 5; i += 1) {
          var gy = B - (B - T) * i / 5;
          ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(R, gy); ctx.stroke();
          ctx.fillStyle = C.muted; ctx.textAlign = "right";
          ctx.fillText((i / 5).toFixed(1), L - 8, gy + 4);
        }
        ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, B); ctx.lineTo(R, B); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("итерации self-play", (L + R) / 2, B + 30);
        ctx.save(); ctx.translate(18, (T + B) / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("эксплуатируемость", 0, 0); ctx.restore();
        ctx.textAlign = "left"; ctx.fillStyle = INK;
        ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("Насколько чемпиона можно наказать", L, T - 20);
        ctx.font = "12px PT Sans, sans-serif";

        var n = curveLast.length, span = Math.max(n - 1, 1);
        function px(k) { return L + (R - L) * k / span; }
        function py(v) { return B - (B - T) * Math.max(0, Math.min(1, v)); }
        function line(arr, col, w) {
          ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath();
          for (var k = 0; k < arr.length; k += 1) {
            if (k === 0) ctx.moveTo(px(k), py(arr[k])); else ctx.lineTo(px(k), py(arr[k]));
          }
          ctx.stroke();
        }
        line(curveLast, C.red, 2.0);
        line(curveMix, C.blue, 2.6);
        ctx.fillStyle = C.red; ctx.fillText("последняя версия", L + 12, T + 18);
        ctx.fillStyle = C.blue; ctx.fillText("смесь лиги (средняя за прогон)", L + 12, T + 36);

        var BX = 606, BY = 380, BW = 268, BH = B - T;
        ctx.fillStyle = INK; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText(panel === 0 ? "Состав смеси лиги" : "Кого выбирали чемпионом", BX, T - 20);
        ctx.font = "12px PT Sans, sans-serif";

        if (panel === 0) {
          var mix = leagueMix();
          var eq = BY - BH * 0.2 / 0.7;
          ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(BX, eq); ctx.lineTo(BX + BW, eq); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = C.muted; ctx.textAlign = "left";
          ctx.fillText("равновесие 0,20", BX + BW - 110, eq - 7);
          var bw = BW / 5 * 0.6;
          for (i = 0; i < 5; i += 1) {
            var cx = BX + BW * (i + 0.5) / 5;
            var h = BH * Math.min(mix[i], 0.7) / 0.7;
            ctx.fillStyle = Math.abs(mix[i] - 0.2) > 0.1 ? C.red : C.green;
            ctx.fillRect(cx - bw / 2, BY - h, bw, h);
            ctx.fillStyle = C.muted; ctx.textAlign = "center";
            ctx.save(); ctx.translate(cx, BY + 16); ctx.rotate(-Math.PI / 9);
            ctx.fillText(LABELS[i], 0, 0); ctx.restore();
            ctx.fillStyle = INK;
            ctx.fillText(mix[i].toFixed(2), cx, BY - h - 7);
          }
        } else {
          var rows = 5, m = champs.length, show = Math.min(m, 120);
          var cellW = BW / show, cellH = BH / rows;
          for (i = 0; i < show; i += 1) {
            var act = champs[m - show + i];
            ctx.fillStyle = i === show - 1 ? C.gold : C.blue;
            ctx.fillRect(BX + i * cellW, T + act * cellH + 2, Math.max(cellW - 0.5, 1), cellH - 4);
          }
          ctx.fillStyle = C.muted; ctx.textAlign = "right";
          for (i = 0; i < 5; i += 1) ctx.fillText(LABELS[i], BX - 6, T + i * cellH + cellH / 2 + 4);
          ctx.textAlign = "center";
          ctx.fillText("последние " + show + " версий →", BX + BW / 2, BY + 22);
        }
        ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(BX, BY); ctx.lineTo(BX + BW, BY); ctx.stroke();

        var eLast = curveLast[curveLast.length - 1], eMix = curveMix[curveMix.length - 1];
        out.set([
          { label: "итераций", value: String(t), color: C.muted },
          { label: "эксплуатируемость последней версии", value: eLast.toFixed(3), color: C.red },
          { label: "эксплуатируемость смеси лиги", value: eMix.toFixed(3), color: C.blue },
          { label: "во сколько раз смесь безопаснее", value: (eLast / Math.max(eMix, 1e-3)).toFixed(1) + "×", color: C.gold }
        ]);
      }

      function tick() { step(); step(); draw(); }
      function run() { if (timer === null) timer = window.setInterval(tick, 80); }
      function stop() { if (timer !== null) { window.clearInterval(timer); timer = null; } }
      function restart() { stop(); reset(); draw(); run(); }

      K.slider(controls, {
        label: "Размер архива (1 — только зеркало)",
        min: 1, max: 200, step: 1, value: archive,
        format: function (v) { return String(v); }
      }, function (v) { archive = v; restart(); });

      K.slider(controls, {
        label: "Партий на оценку одного действия",
        min: 5, max: 2000, step: 5, value: games,
        format: function (v) { return String(v); }
      }, function (v) { games = v; restart(); });

      K.slider(controls, {
        label: "Доля равномерного шума ε",
        min: 0, max: 0.4, step: 0.02, value: eps,
        format: function (v) { return Number(v).toFixed(2); }
      }, function (v) { eps = v; restart(); });

      K.segmented(controls, {
        label: "Правая панель",
        value: 0,
        options: [
          { label: "состав смеси", value: 0 },
          { label: "лента чемпионов", value: 1 }
        ]
      }, function (v) { panel = v; draw(); });

      reset();
      draw();
      run();

      return function () { stop(); cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
