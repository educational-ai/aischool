// Lesson 79: scaling-law-lab — распределите бюджет между параметрами и токенами.
// Константы закона L = L_inf + A*N^-alpha + B*D^-beta взяты из измерений урока
// (24 обученные символьные модели на корпусе SMS, scripts/generate_lesson79_visuals.py).
(function () {
  "use strict";
  var FIT = { Linf: 0.10695, A: 4.54647, alpha: 0.28576, B: 5.44326, beta: 0.08710 };

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("scaling-law-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var PB = "#fffef9";
      var logC = 9;        // бюджет C = 6*N*D
      var logN = 3.3;      // положение вдоль линии бюджета
      var quality = 1;     // доля новой информации в токене
      var logQ = 0;        // число будущих запросов (0 = не учитываем)

      function budget() { return Math.pow(10, logC); }
      function dataFor(n) { return budget() / (6 * n); }
      function loss(n, d) {
        var de = Math.max(d * quality, 1);
        return FIT.Linf + FIT.A * Math.pow(n, -FIT.alpha) + FIT.B * Math.pow(de, -FIT.beta);
      }
      function totalCost(n, d) {
        return 6 * n * d + Math.pow(10, logQ) * 2 * n;
      }
      // минимум loss вдоль линии бюджета
      function optimum() {
        var lo = 1.5, hi = 8, best = lo, bl = Infinity;
        for (var i = 0; i <= 600; i += 1) {
          var l = lo + (hi - lo) * i / 600, n = Math.pow(10, l), v = loss(n, dataFor(n));
          if (v < bl) { bl = v; best = l; }
        }
        return { logN: best, N: Math.pow(10, best), loss: bl };
      }
      // минимум полной стоимости при целевом loss (только когда учитываем запросы)
      function deployOpt(target) {
        var lo = 1.5, hi = 8, best = null, bc = Infinity;
        for (var i = 0; i <= 600; i += 1) {
          var l = lo + (hi - lo) * i / 600, n = Math.pow(10, l);
          var rest = target - FIT.Linf - FIT.A * Math.pow(n, -FIT.alpha);
          if (rest <= 0) continue;
          var d = Math.pow(FIT.B / rest, 1 / FIT.beta) / quality;
          var c = totalCost(n, d);
          if (c < bc) { bc = c; best = { logN: l, N: n, D: d, cost: c }; }
        }
        return best;
      }

      K.hint(root, "Бюджет вычислений жёстко связывает модель и корпус: C ≈ 6·N·D. Двигая N, вы автоматически двигаете D в другую сторону вдоль гиперболы. Кривая loss оказывается U-образной: слишком маленькая модель упирается в capacity, слишком большая — недоучена. Крестик — ваш выбор, кружок — оптимум бюджета.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева: loss вдоль линии постоянного бюджета. Справа: плоскость (N, D) — линия бюджета и compute-optimal ridge. Ухудшите качество токена — правая часть закона сдвинется вверх, и оптимум уедет к большей модели: при плохих данных выгоднее capacity, а не объём. Включите цену инференса — и «лучшая» модель станет меньше, чем training-optimal.");
      var cs = K.makeCanvas(stage, W, H, { label: "U-образный loss вдоль бюджета и плоскость параметров-токенов", onResize: draw });

      var LX = 60, LW = 400, TY = 60, PH = 330;
      var RX = 545, RW = 320;

      function xL(l) { return LX + (l - 1.8) / (7.2 - 1.8) * LW; }
      function yL(v, lo, hi) { return TY + PH - (v - lo) / (hi - lo) * PH; }
      function xR(l) { return RX + (l - 1.8) / (7.2 - 1.8) * RW; }
      function yR(l) { return TY + PH - (l - 3.5) / (9.5 - 3.5) * PH; }

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var opt = optimum();
        var N = Math.pow(10, logN), D = dataFor(N), L = loss(N, D);

        // --- левая панель: U-кривая
        var vals = [], lo = Infinity, hi = -Infinity;
        for (i = 0; i <= 220; i += 1) {
          var l = 1.8 + (7.2 - 1.8) * i / 220, n = Math.pow(10, l), v = loss(n, dataFor(n));
          vals.push([l, v]); if (v < lo) lo = v; if (v > hi) hi = v;
        }
        hi = Math.min(hi, lo + 1.6);
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.strokeRect(LX, TY, LW, PH);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (i = 2; i <= 7; i += 1) { ctx.fillText("10^" + i, xL(i), TY + PH + 18); }
        ctx.fillText("параметры N (лог)", LX + LW / 2, TY + PH + 38);
        ctx.save(); ctx.translate(LX - 42, TY + PH / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("val loss, нат/символ", 0, 0); ctx.restore();
        ctx.textAlign = "right";
        for (i = 0; i <= 4; i += 1) {
          var vv = lo + (hi - lo) * i / 4;
          ctx.fillText(vv.toFixed(2), LX - 8, yL(vv, lo, hi) + 4);
          ctx.strokeStyle = "rgba(222,221,212,0.8)";
          ctx.beginPath(); ctx.moveTo(LX, yL(vv, lo, hi)); ctx.lineTo(LX + LW, yL(vv, lo, hi)); ctx.stroke();
        }
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.6; ctx.beginPath();
        for (i = 0; i < vals.length; i += 1) {
          var px = xL(vals[i][0]), py = yL(Math.min(vals[i][1], hi), lo, hi);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        // оптимум
        ctx.strokeStyle = C.green; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(xL(opt.logN), TY); ctx.lineTo(xL(opt.logN), TY + PH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = PB; ctx.strokeStyle = C.green; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.arc(xL(opt.logN), yL(opt.loss, lo, hi), 7, 0, 7); ctx.fill(); ctx.stroke();
        // выбор пользователя
        var cx = xL(logN), cy = yL(Math.min(L, hi), lo, hi);
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(cx - 8, cy - 8); ctx.lineTo(cx + 8, cy + 8);
        ctx.moveTo(cx + 8, cy - 8); ctx.lineTo(cx - 8, cy + 8);
        ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("недостаточно параметров", LX + 8, TY + 18);
        ctx.textAlign = "right";
        ctx.fillText("недостаточно токенов", LX + LW - 8, TY + 18);

        // --- правая панель: плоскость (N, D)
        ctx.font = "12px PT Sans, sans-serif";
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.strokeRect(RX, TY, RW, PH);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (i = 2; i <= 7; i += 1) ctx.fillText("10^" + i, xR(i), TY + PH + 18);
        ctx.fillText("параметры N (лог)", RX + RW / 2, TY + PH + 38);
        ctx.textAlign = "right";
        for (i = 4; i <= 9; i += 1) ctx.fillText("10^" + i, RX - 6, yR(i) + 4);
        // ridge: для набора бюджетов найти оптимум
        ctx.strokeStyle = C.violet || "#6f5a8f"; ctx.lineWidth = 2.2; ctx.beginPath();
        var started = false;
        for (var b = 7.5; b <= 12.5; b += 0.25) {
          var saved = logC; logC = b;
          var o = optimum(); var dOpt = dataFor(o.N);
          logC = saved;
          var px2 = xR(o.logN), py2 = yR(Math.log(dOpt) / Math.LN10);
          if (!started) { ctx.moveTo(px2, py2); started = true; } else ctx.lineTo(px2, py2);
        }
        ctx.stroke();
        // линия бюджета
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.0; ctx.setLineDash([5, 3]);
        ctx.beginPath();
        for (i = 0; i <= 60; i += 1) {
          var ll = 1.8 + (7.2 - 1.8) * i / 60;
          var dd = Math.log(dataFor(Math.pow(10, ll))) / Math.LN10;
          var ppx = xR(ll), ppy = yR(dd);
          if (i === 0) ctx.moveTo(ppx, ppy); else ctx.lineTo(ppx, ppy);
        }
        ctx.stroke(); ctx.setLineDash([]);
        // текущая точка и оптимум бюджета
        var dNow = Math.log(D) / Math.LN10;
        ctx.fillStyle = C.green;
        ctx.beginPath(); ctx.arc(xR(opt.logN), yR(Math.log(dataFor(opt.N)) / Math.LN10), 6, 0, 7); ctx.fill();
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.6;
        var qx = xR(logN), qy = yR(dNow);
        ctx.beginPath();
        ctx.moveTo(qx - 7, qy - 7); ctx.lineTo(qx + 7, qy + 7);
        ctx.moveTo(qx + 7, qy - 7); ctx.lineTo(qx - 7, qy + 7);
        ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("ridge оптимумов", RX + 8, TY + 18);
        ctx.fillText("линия бюджета ND = C/6", RX + 8, TY + 34);
        ctx.save(); ctx.translate(RX - 40, TY + PH / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center"; ctx.fillText("токены D (лог)", 0, 0); ctx.restore();

        var rows = [
          { label: "N (параметры)", value: fmt(N), color: C.red },
          { label: "D (токены)", value: fmt(D), color: C.blue },
          { label: "токенов на параметр", value: (D / N).toFixed(1), color: C.gold },
          { label: "loss вашего выбора", value: L.toFixed(3), color: C.red },
          { label: "loss оптимума бюджета", value: opt.loss.toFixed(3), color: C.green },
          { label: "переплата по loss", value: "+" + (L - opt.loss).toFixed(3), color: C.muted }
        ];
        if (logQ > 0) {
          var dep = deployOpt(L);
          if (dep) {
            rows.push({ label: "запросов Q", value: fmt(Math.pow(10, logQ)), color: C.violet });
            rows.push({ label: "тот же loss дешевле всего при N", value: fmt(dep.N), color: C.violet });
            rows.push({ label: "полная стоимость: ваша / лучшая", value: fmt(totalCost(N, D)) + " / " + fmt(dep.cost), color: C.gold });
          }
        }
        output.set(rows);
      }

      function fmt(x) {
        if (x >= 1e6) return (x / 1e6).toFixed(x >= 1e7 ? 0 : 1) + " млн";
        if (x >= 1e3) return (x / 1e3).toFixed(x >= 1e4 ? 0 : 1) + " тыс.";
        return x.toFixed(0);
      }

      K.slider(controls, {
        label: "Бюджет вычислений C, FLOP", min: 7.5, max: 12, step: 0.25, value: logC,
        format: function (v) { return "10^" + v.toFixed(2); }
      }, function (v) { logC = v; draw(); });
      K.slider(controls, {
        label: "Размер модели N", min: 1.8, max: 7.2, step: 0.05, value: logN,
        format: function (v) { return "10^" + v.toFixed(2); }
      }, function (v) { logN = v; draw(); });
      K.segmented(controls, {
        label: "Качество токена (доля новой информации)", value: 1,
        options: [
          { label: "чистые данные", value: 1 },
          { label: "половина — повторы", value: 0.5 },
          { label: "четверть полезного", value: 0.25 }
        ]
      }, function (v) { quality = v; draw(); });
      K.segmented(controls, {
        label: "Учитывать цену инференса", value: 0,
        options: [
          { label: "нет", value: 0 },
          { label: "Q = 10⁶", value: 6 },
          { label: "Q = 10⁹", value: 9 },
          { label: "Q = 10¹²", value: 12 }
        ]
      }, function (v) { logQ = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
