// Lesson 70: bandit-lab — five real MovieLens arms, four policies, one budget.
// Watch how the shape of the regret curve (linear vs flattening) depends on the policy
// and on how much exploration you buy; break the environment and see stale confidence.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("bandit-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 500;
      var PB = "#fffef9";
      var NAMES = ["Star Wars", "Godfather", "Fargo", "Contact", "Liar Liar"];
      var BASE = [0.859, 0.850, 0.799, 0.676, 0.406];   // real share of "liked" (rating >= 4)
      var COLS = [C.blue, C.violet || "#6f5a8f", C.green, C.gold, C.red];
      var TMAX = 3000;
      var policy = 2, eps = 0.10, cUCB = 0.5, speed = 20;
      var p = BASE.slice(), t = 0, cnt = [], sum = [], reg = 0, hist = [], last = -1;
      var swapped = false, timer = null, running = false;

      // deterministic PRNG so that a given experiment is reproducible
      var seed = 20250701;
      function rnd() {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        var x = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
      }
      // Beta(a,b) via two Gammas (Marsaglia-Tsang), integer-friendly and fast enough here
      function gammaSample(a) {
        if (a < 1) return gammaSample(a + 1) * Math.pow(rnd(), 1 / a);
        var d = a - 1 / 3, c = 1 / Math.sqrt(9 * d), x, v, u;
        for (;;) {
          do { x = normal(); v = 1 + c * x; } while (v <= 0);
          v = v * v * v; u = rnd();
          if (u < 1 - 0.0331 * x * x * x * x) return d * v;
          if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
      }
      function normal() {
        var u1 = Math.max(1e-12, rnd()), u2 = rnd();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      function betaSample(a, b) { var g = gammaSample(a); return g / (g + gammaSample(b)); }

      function reset(keepSwap) {
        cnt = [0, 0, 0, 0, 0]; sum = [0, 0, 0, 0, 0];
        t = 0; reg = 0; hist = []; last = -1; seed = 20250701;
        if (!keepSwap) { swapped = false; p = BASE.slice(); }
      }
      function mean(a) { return cnt[a] > 0 ? sum[a] / cnt[a] : 0; }
      function best() { var b = 0, i; for (i = 1; i < 5; i += 1) if (p[i] > p[b]) b = i; return b; }

      function choose() {
        var i, k;
        for (i = 0; i < 5; i += 1) if (cnt[i] === 0) return i;      // one pull of each first
        if (policy === 0) k = argmaxMean();
        else if (policy === 1) k = rnd() < eps ? Math.min(4, Math.floor(rnd() * 5)) : argmaxMean();
        else if (policy === 2) {
          var bi = 0, bv = -1e9;
          for (i = 0; i < 5; i += 1) {
            var v = mean(i) + cUCB * Math.sqrt(Math.log(Math.max(2, t)) / cnt[i]);
            if (v > bv) { bv = v; bi = i; }
          }
          k = bi;
        } else {
          var ti = 0, tv = -1;
          for (i = 0; i < 5; i += 1) {
            var s = betaSample(1 + sum[i], 1 + cnt[i] - sum[i]);
            if (s > tv) { tv = s; ti = i; }
          }
          k = ti;
        }
        return k;
      }
      function argmaxMean() {
        var bi = 0, i;
        for (i = 1; i < 5; i += 1) if (mean(i) > mean(bi)) bi = i;
        return bi;
      }
      function step() {
        if (t >= TMAX) { stop(); return; }
        var a = choose();
        var r = rnd() < p[a] ? 1 : 0;
        cnt[a] += 1; sum[a] += r; t += 1; last = a;
        reg += p[best()] - p[a];
        hist.push(reg);
      }

      K.hint(root, "Пять «рук» — реальные фильмы из MovieLens 100K, вероятность награды равна настоящей доле оценок «нравится». Две верхние руки почти неразличимы (0,859 и 0,850), нижняя очевидно слаба. Выберите политику показа, задайте силу исследования и запустите бюджет из 3000 показов. Смотрите не на счёт, а на ФОРМУ кривой regret: у постоянного исследования она прямая, у оптимизма и Томпсона — загибается.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var buttons = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Верх: сколько показов получила каждая рука (полоса), её эмпирическое среднее (точка) и верхняя граница «среднее + бонус» (штрих). Низ: накопленный regret — сколько «нравится» потеряно относительно всезнающего выбора. Кнопка «сломать среду» меняет местами лучшую и четвёртую руку: старые счётчики продолжают убеждать алгоритм в том, чего уже нет.");
      var cs = K.makeCanvas(stage, W, H, { label: "Показы по рукам, доверительные границы и кривая regret", onResize: draw });

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var X0 = 118, X1 = 700, TOP = 34, RH = 34;
        var bi = best();
        ctx.textAlign = "left"; ctx.fillStyle = C.muted;
        ctx.fillText("доля показов и оценка каждой руки", X0, TOP - 12);
        for (i = 0; i < 5; i += 1) {
          var y = TOP + i * RH;
          var share = t > 0 ? cnt[i] / t : 0;
          ctx.fillStyle = COLS[i]; ctx.globalAlpha = 0.20;
          ctx.fillRect(X0, y, (X1 - X0) * share, RH - 12);
          ctx.globalAlpha = 1;
          ctx.textAlign = "right"; ctx.fillStyle = i === bi ? C.ink || "#171915" : C.muted;
          ctx.fillText(NAMES[i] + (i === bi ? " ★" : ""), X0 - 10, y + 14);
          ctx.textAlign = "left"; ctx.fillStyle = COLS[i];
          ctx.fillText("N=" + cnt[i] + (cnt[i] > 0 ? ", μ̂=" + mean(i).toFixed(3) : ""), X1 + 14, y + 14);
          if (i === last) {
            ctx.fillStyle = C.gold; ctx.beginPath();
            ctx.arc(X0 - 92, y + 9, 4, 0, 7); ctx.fill();
          }
          // mean and upper bound on a 0..1 scale drawn inside the row
          if (cnt[i] > 0) {
            var sc = function (v) { return X0 + (X1 - X0) * Math.max(0, Math.min(1, v)); };
            var mx = sc(mean(i));
            var bonus = cUCB * Math.sqrt(Math.log(Math.max(2, t)) / cnt[i]);
            var ux = sc(mean(i) + bonus);
            ctx.strokeStyle = COLS[i]; ctx.lineWidth = 1.4;
            ctx.beginPath(); ctx.moveTo(mx, y + 4); ctx.lineTo(ux, y + 4); ctx.stroke();
            ctx.fillStyle = COLS[i]; ctx.beginPath(); ctx.arc(mx, y + 4, 4, 0, 7); ctx.fill();
            ctx.strokeStyle = COLS[i]; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(ux, y - 2); ctx.lineTo(ux, y + 10); ctx.stroke();
          }
        }
        // regret panel
        var GY0 = TOP + 5 * RH + 26, GY1 = H - 34, GX0 = X0, GX1 = X1 + 100;
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(GX0, GY0); ctx.lineTo(GX0, GY1); ctx.lineTo(GX1, GY1); ctx.stroke();
        var top = Math.max(20, reg * 1.15);
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        ctx.fillText(top.toFixed(0), GX0 - 8, GY0 + 5);
        ctx.fillText("0", GX0 - 8, GY1 + 4);
        ctx.textAlign = "left";
        ctx.fillText("накопленный regret", GX0 + 6, GY0 - 8);
        ctx.fillText("раунд t → " + TMAX, GX1 - 120, GY1 + 18);
        if (hist.length > 1) {
          ctx.strokeStyle = C.red; ctx.lineWidth = 2.4; ctx.beginPath();
          for (i = 0; i < hist.length; i += 1) {
            var px = GX0 + (GX1 - GX0) * (i / TMAX);
            var py = GY1 - (GY1 - GY0) * (hist[i] / top);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
          // straight reference from origin through the current point: linear growth
          ctx.strokeStyle = "rgba(110,114,106,0.55)"; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(GX0, GY1);
          ctx.lineTo(GX0 + (GX1 - GX0) * (hist.length / TMAX), GY1 - (GY1 - GY0) * (reg / top));
          ctx.stroke(); ctx.setLineDash([]);
        }
        if (swapped) {
          ctx.fillStyle = C.red; ctx.textAlign = "left";
          ctx.fillText("среда сломана: лучшая и четвёртая руки поменялись", GX0 + 6, GY1 - 8);
        }
        var names = ["чистая жадность", "ε-greedy", "UCB", "Thompson"];
        var shareBest = t > 0 ? cnt[bi] / t : 0;
        output.set([
          { label: "Политика", value: names[policy], color: C.red },
          { label: "Раундов сыграно", value: String(t), color: C.blue },
          { label: "Накопленный regret", value: reg.toFixed(1), color: C.gold },
          { label: "Доля показов лучшей руке", value: (100 * shareBest).toFixed(1) + "%", color: C.green },
          { label: "Regret на раунд (последняя 1000)", value: perRound().toFixed(4), color: C.muted }
        ]);
      }
      function perRound() {
        if (hist.length < 2) return 0;
        var n = Math.min(1000, hist.length - 1);
        return (hist[hist.length - 1] - hist[hist.length - 1 - n]) / n;
      }

      function tick() {
        var i;
        for (i = 0; i < speed; i += 1) step();
        draw();
        if (running) timer = window.requestAnimationFrame(tick);
      }
      function stop() { running = false; if (timer) { window.cancelAnimationFrame(timer); timer = null; } btnRun.textContent = "пуск"; }

      K.segmented(controls, {
        label: "Политика", value: 2, options: [
          { label: "жадность", value: 0 }, { label: "ε-greedy", value: 1 },
          { label: "UCB", value: 2 }, { label: "Thompson", value: 3 }]
      }, function (v) { policy = v; draw(); });
      K.slider(controls, { label: "ε (для ε-greedy)", min: 0, max: 0.4, step: 0.01, value: 0.10, format: function (v) { return v.toFixed(2); } }, function (v) { eps = v; draw(); });
      K.slider(controls, { label: "c — цена бонуса в UCB", min: 0, max: 2, step: 0.05, value: 0.5, format: function (v) { return v.toFixed(2); } }, function (v) { cUCB = v; draw(); });
      K.slider(controls, { label: "раундов за кадр", min: 1, max: 60, step: 1, value: 20, format: function (v) { return String(v); } }, function (v) { speed = v; });

      var btnRun = K.element("button", "kontur-int-segment", { type: "button", text: "пуск" });
      var btnStep = K.element("button", "kontur-int-segment", { type: "button", text: "+100 показов" });
      var btnBreak = K.element("button", "kontur-int-segment", { type: "button", text: "сломать среду" });
      var btnReset = K.element("button", "kontur-int-segment", { type: "button", text: "сброс" });
      [btnRun, btnStep, btnBreak, btnReset].forEach(function (b) { b.style.margin = "0 6px"; buttons.appendChild(b); });
      btnRun.addEventListener("click", function () {
        if (running) { stop(); return; }
        if (t >= TMAX) reset(true);
        running = true; btnRun.textContent = "пауза"; tick();
      });
      btnStep.addEventListener("click", function () { var i; for (i = 0; i < 100; i += 1) step(); draw(); });
      btnBreak.addEventListener("click", function () {
        swapped = !swapped;
        var a = p[0]; p[0] = p[3]; p[3] = a;
        draw();
      });
      btnReset.addEventListener("click", function () { stop(); reset(false); draw(); });

      draw();
      return function () { stop(); cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
