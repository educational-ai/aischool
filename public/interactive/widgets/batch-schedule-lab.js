// Lesson 61: batch-schedule-lab — fixed data budget, variable batch size, step size and
// schedule. Left panel: trajectory of SGD in a narrow valley. Right panel: loss vs the number
// of examples actually read. Teaches: noise ~ 1/sqrt(b), stability threshold 2/L, and the fact
// that a constant step leaves a noise floor that a decaying schedule removes.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("batch-schedule-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 440;
      var PB = "#fffef9";
      var LAM1 = 1.0, LAM2 = 0.12;        // curvature of the valley; L = 1, so 2/L = 2
      var SIGMA = 0.9;                    // per-example gradient noise
      var BUDGET = 8192;                  // examples read — the honest common budget
      var batch = 16, etaFrac = 0.5, sched = 0, showCloud = true;

      // deterministic pseudo-random generator, so the picture is reproducible
      function rng(seed) {
        var s = seed >>> 0;
        return function () {
          s = (s + 0x6d2b79f5) >>> 0;
          var t = s;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function gauss(r) {
        var u = Math.max(1e-9, r()), v = r();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      function loss(w) { return 0.5 * (LAM1 * w[0] * w[0] + LAM2 * w[1] * w[1]); }

      function rate(eta, t, steps) {
        if (sched === 1) return eta / (1 + 8 * t / Math.max(1, steps));
        if (sched === 2) return eta * 0.5 * (1 + Math.cos(Math.PI * t / Math.max(1, steps)));
        return eta;
      }

      function simulate() {
        var steps = Math.max(1, Math.floor(BUDGET / batch));
        var r = rng(61), w = [2.3, 2.4], path = [[w[0], w[1]]], curve = [], t, lr, n1, n2, sd;
        sd = SIGMA / Math.sqrt(batch);
        var diverged = false;
        for (t = 0; t < steps; t += 1) {
          lr = rate(2.0 * etaFrac, t, steps);        // eta = 2*frac / L, threshold at frac = 1
          n1 = gauss(r) * sd; n2 = gauss(r) * sd;
          w = [w[0] - lr * (LAM1 * w[0] + n1), w[1] - lr * (LAM2 * w[1] + n2)];
          if (!isFinite(w[0]) || Math.abs(w[0]) > 1e6) { diverged = true; break; }
          if (path.length < 900) path.push([w[0], w[1]]);
          curve.push([(t + 1) * batch, loss(w)]);
        }
        var tail = 0, cnt = 0;
        for (t = Math.floor(curve.length * 0.85); t < curve.length; t += 1) { tail += curve[t][1]; cnt += 1; }
        return {
          steps: steps, path: path, curve: curve, sd: sd, diverged: diverged,
          floor: cnt ? tail / cnt : NaN, eta: 2.0 * etaFrac
        };
      }

      K.hint(root, "Бюджет фиксирован: алгоритм читает ровно 8192 примера — как ни дели их на батчи. Малый батч даёт много дешёвых, но шумных обновлений; большой — мало точных. Шаг измеряется в долях порога устойчивости 2/L: перейдите за единицу и посмотрите, что будет. Затем включите угасание шага и сравните финальный уровень потерь при том же бюджете.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева — узкая долина и траектория обучения: точки тем разбросаннее, чем меньше батч. Справа — потери по числу прочитанных примеров (лог), пунктир — точный минимум. Постоянный шаг оставляет «пол» из шума высотой порядка η·σ²/b; косинусное угасание опускает траекторию на дно, не меняя бюджета.");
      var cs = K.makeCanvas(stage, W, H, { label: "Траектория SGD в долине и кривая потерь при фиксированном бюджете", onResize: draw });

      var LX = 56, LY = 220, LS = 74;            // valley panel: origin and scale
      function m2s(x, y) { return [LX + 180 + x * LS, LY - y * LS * 0.62]; }
      var RX = 545, RW = 320, RY0 = 60, RH = 300;
      function c2s(ex, ls) {
        var x = RX + RW * Math.max(0, Math.log(Math.max(1, ex) / 16)) / Math.log(BUDGET / 16);
        var v = Math.min(1, Math.max(0, (Math.log10(Math.max(1e-5, ls)) + 5) / 6));
        return [x, RY0 + RH - v * RH];
      }

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var sim = simulate();

        // ---- left panel: contours of the quadratic valley
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        for (i = 1; i <= 5; i += 1) {
          var lv = i * 0.55;
          ctx.beginPath();
          for (var a = 0; a <= 64; a += 1) {
            var th = a / 64 * 2 * Math.PI;
            var p = m2s(Math.sqrt(2 * lv / LAM1) * Math.cos(th), Math.sqrt(2 * lv / LAM2) * Math.sin(th));
            if (a === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
          }
          ctx.stroke();
        }
        var o = m2s(0, 0);
        ctx.fillStyle = C.green || "#38735d";
        ctx.beginPath(); ctx.arc(o[0], o[1], 5, 0, 7); ctx.fill();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("минимум", o[0], o[1] + 20);

        // ---- trajectory
        ctx.strokeStyle = sim.diverged ? (C.red || "#b94a3b") : (C.blue || "#315f8c");
        ctx.lineWidth = 1.4; ctx.beginPath();
        for (i = 0; i < sim.path.length; i += 1) {
          var q = m2s(sim.path[i][0], sim.path[i][1]);
          if (q[0] < -3000 || q[0] > 3000 || q[1] < -3000 || q[1] > 3000) break;
          if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
        }
        ctx.stroke();
        if (showCloud) {
          ctx.fillStyle = "rgba(49,95,140,0.45)";
          for (i = 0; i < sim.path.length; i += Math.max(1, Math.floor(sim.path.length / 220))) {
            var s = m2s(sim.path[i][0], sim.path[i][1]);
            if (s[0] > 20 && s[0] < 500 && s[1] > 10 && s[1] < H - 10) {
              ctx.beginPath(); ctx.arc(s[0], s[1], 1.8, 0, 7); ctx.fill();
            }
          }
        }
        var st = m2s(2.3, 2.4);
        ctx.fillStyle = C.gold || "#a57920";
        ctx.beginPath(); ctx.arc(st[0], st[1], 5, 0, 7); ctx.fill();
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("старт", st[0] + 9, st[1] + 4);
        ctx.textAlign = "center"; ctx.fillStyle = C.ink || "#171915";
        ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("узкая долина: траектория обучения", LX + 180, 34);
        ctx.font = "12px PT Sans, sans-serif";

        // ---- right panel: loss vs examples read
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(RX, RY0); ctx.lineTo(RX, RY0 + RH); ctx.lineTo(RX + RW, RY0 + RH); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        for (i = -4; i <= 1; i += 1) {
          var yy = c2s(BUDGET, Math.pow(10, i))[1];
          ctx.fillText("10" + (i < 0 ? "⁻" : "") + "⁰¹²³⁴".charAt(Math.abs(i)), RX - 8, yy + 4);
          ctx.strokeStyle = "rgba(222,221,212,0.7)";
          ctx.beginPath(); ctx.moveTo(RX, yy); ctx.lineTo(RX + RW, yy); ctx.stroke();
        }
        ctx.textAlign = "center";
        ctx.fillText("прочитано примеров (лог)", RX + RW / 2, RY0 + RH + 34);
        ctx.fillText("16", RX, RY0 + RH + 16);
        ctx.fillText(String(BUDGET), RX + RW, RY0 + RH + 16);
        ctx.fillStyle = C.ink || "#171915"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("потери при общем бюджете данных", RX + RW / 2, 34);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.strokeStyle = sim.diverged ? (C.red || "#b94a3b") : (C.blue || "#315f8c");
        ctx.lineWidth = 1.8; ctx.beginPath();
        var started = false;
        for (i = 0; i < sim.curve.length; i += 1) {
          var pc = c2s(sim.curve[i][0], sim.curve[i][1]);
          if (!isFinite(pc[1])) continue;
          if (!started) { ctx.moveTo(pc[0], pc[1]); started = true; } else ctx.lineTo(pc[0], pc[1]);
        }
        ctx.stroke();
        if (!sim.diverged && isFinite(sim.floor)) {
          var fy = c2s(BUDGET, sim.floor)[1];
          ctx.strokeStyle = C.gold || "#a57920"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(RX, fy); ctx.lineTo(RX + RW, fy); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = C.gold || "#a57920"; ctx.textAlign = "right";
          ctx.fillText("пол из шума", RX + RW, fy - 7);
        }

        output.set([
          { label: "Обновлений при бюджете " + BUDGET, value: String(sim.steps), color: C.blue },
          { label: "Шум оценки σ/√b", value: sim.sd.toFixed(3), color: C.violet || C.blue },
          { label: "Шаг η (порог 2/L = 2)", value: sim.eta.toFixed(2) + (etaFrac >= 1 ? " — за порогом" : ""), color: etaFrac >= 1 ? C.red : C.green },
          { label: "Финальные потери", value: sim.diverged ? "расходится" : sim.floor.toExponential(2), color: sim.diverged ? C.red : C.gold }
        ]);
      }

      K.segmented(controls, {
        label: "Размер батча b", value: 16,
        options: [{ label: "1", value: 1 }, { label: "4", value: 4 }, { label: "16", value: 16 }, { label: "64", value: 64 }, { label: "256", value: 256 }]
      }, function (v) { batch = v; draw(); });
      K.segmented(controls, {
        label: "Расписание шага", value: 0,
        options: [{ label: "постоянный", value: 0 }, { label: "1/t", value: 1 }, { label: "косинус", value: 2 }]
      }, function (v) { sched = v; draw(); });
      K.slider(controls, {
        label: "Шаг η в долях порога 2/L", min: 0.05, max: 1.3, step: 0.05, value: 0.5,
        format: function (v) { return v.toFixed(2) + "·(2/L)"; }
      }, function (v) { etaFrac = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
