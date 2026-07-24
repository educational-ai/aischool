// Lesson 64: federated-lab — FedAvg на восьми неодинаковых клиентах: локальные шаги,
// неоднородность данных, выбор весов агрегации и шум приватности.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("federated-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1000, H = 520;
      var PB = "#fffef9";
      var NK = 8;                       // число клиентов
      var LR = 0.12;                    // локальная скорость обучения
      var MAXR = 80;                    // раундов в прогоне

      var state = { E: 5, het: 0.7, ineq: 1.45, frac: 1.0, weight: "records", sigma: 0 };
      var anim = { raf: 0, w: [0, 0], round: 0, path: [], local: [], drift: 0, done: false, tick: 0 };

      // --- детерминированный генератор (mulberry32)
      function mulberry32(a) {
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          var t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      var rnd = mulberry32(64);
      function gauss() {
        var u = Math.max(1e-9, rnd()), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // --- клиенты: центр локального оптимума и число строк
      function clients() {
        var list = [], i;
        for (i = 0; i < NK; i += 1) {
          var ang = 2 * Math.PI * i / NK;
          var rad = 1 + 0.45 * (((i * 7) % 5) / 4);
          list.push({
            m: [state.het * rad * Math.cos(ang) * 2.1, state.het * rad * Math.sin(ang) * 1.5],
            n: Math.max(1, Math.round(12 * Math.pow(state.ineq, NK - 1 - i)))
          });
        }
        return list;
      }
      var CL = clients();

      function optima() {
        var sr = [0, 0], su = [0, 0], tot = 0, i;
        for (i = 0; i < CL.length; i += 1) {
          sr[0] += CL[i].n * CL[i].m[0]; sr[1] += CL[i].n * CL[i].m[1];
          su[0] += CL[i].m[0]; su[1] += CL[i].m[1]; tot += CL[i].n;
        }
        return { rec: [sr[0] / tot, sr[1] / tot], uni: [su[0] / CL.length, su[1] / CL.length], total: tot };
      }

      function reset() {
        CL = clients();
        rnd = mulberry32(64);
        if (anim.raf) cancelAnimationFrame(anim.raf);
        anim = { raf: 0, w: [0, 0], round: 0, path: [[0, 0]], local: [], drift: 0, done: false, tick: 0 };
        anim.raf = requestAnimationFrame(step);
      }

      function oneRound() {
        var i, j, sel = [], op = optima();
        for (i = 0; i < CL.length; i += 1) if (rnd() < state.frac || state.frac >= 1) sel.push(i);
        if (sel.length === 0) sel.push(Math.floor(rnd() * CL.length));
        var deltas = [], sizes = [], paths = [];
        for (j = 0; j < sel.length; j += 1) {
          var c = CL[sel[j]];
          var w = [anim.w[0], anim.w[1]], p = [[w[0], w[1]]];
          for (i = 0; i < state.E; i += 1) {
            w[0] -= LR * 2 * (w[0] - c.m[0]);
            w[1] -= LR * 2 * (w[1] - c.m[1]);
            p.push([w[0], w[1]]);
          }
          deltas.push([w[0] - anim.w[0], w[1] - anim.w[1]]);
          sizes.push(c.n); paths.push(p);
        }
        var tot = 0, agg = [0, 0];
        for (j = 0; j < deltas.length; j += 1) tot += (state.weight === "records" ? sizes[j] : 1);
        for (j = 0; j < deltas.length; j += 1) {
          var wgt = (state.weight === "records" ? sizes[j] : 1) / tot;
          agg[0] += wgt * deltas[j][0]; agg[1] += wgt * deltas[j][1];
        }
        // средний разброс локальных обновлений — client drift
        var mean = [0, 0], d = 0;
        for (j = 0; j < deltas.length; j += 1) { mean[0] += deltas[j][0] / deltas.length; mean[1] += deltas[j][1] / deltas.length; }
        for (j = 0; j < deltas.length; j += 1) d += Math.hypot(deltas[j][0] - mean[0], deltas[j][1] - mean[1]) / deltas.length;
        anim.drift = d;
        if (state.sigma > 0) {
          var scale = state.sigma * 0.35 / Math.sqrt(deltas.length);
          agg[0] += scale * gauss(); agg[1] += scale * gauss();
        }
        anim.w = [anim.w[0] + agg[0], anim.w[1] + agg[1]];
        anim.local = paths;
        anim.path.push([anim.w[0], anim.w[1]]);
        if (anim.path.length > 200) anim.path.shift();
        anim.round += 1;
        if (anim.round >= MAXR) anim.done = true;
        return op;
      }

      function step() {
        anim.tick += 1;
        if (anim.tick % 5 === 0) oneRound();
        draw();
        if (!anim.done) anim.raf = requestAnimationFrame(step);
      }

      K.hint(
        root,
        "Восемь клиентов, у каждого свои данные (звезда — минимум его локальной потери) и свой объём строк (размер круга). Сервер рассылает общую точку, клиенты делают E локальных шагов, сервер усредняет обновления. Увеличивайте неоднородность и число локальных шагов — и следите, как разлетаются тонкие локальные траектории. Переключатель весов сдвигает саму цель: синяя мишень — оптимум «средней строки», красная — оптимум «среднего клиента»."
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Локальная потеря клиента — чаша с минимумом в его звезде; глобальная цель — взвешенная сумма чаш. При E = 1 усреднение обновлений равно одному общему градиентному шагу. При большом E каждый клиент успевает уйти к своему минимуму, и среднее промахивается мимо обеих мишеней тем сильнее, чем дальше разнесены данные. Шум приватности не даёт траектории застыть в точке."
      );

      var cs = K.makeCanvas(stage, W, H, {
        label: "Клиенты, локальные траектории и путь глобальной модели",
        onResize: draw, drag: false
      });

      var box = { x: 60, y: 34, w: 600, h: 452 };
      var XMIN = -4.2, XMAX = 4.2, YMIN = -3.0, YMAX = 3.0;
      function px(x) { return box.x + (x - XMIN) / (XMAX - XMIN) * box.w; }
      function py(y) { return box.y + box.h - (y - YMIN) / (YMAX - YMIN) * box.h; }

      function star(ctx, x, y, r, col) {
        ctx.fillStyle = col; ctx.beginPath();
        for (var i = 0; i < 10; i += 1) {
          var rr = i % 2 === 0 ? r : r * 0.45, a = -Math.PI / 2 + i * Math.PI / 5;
          var xx = x + rr * Math.cos(a), yy = y + rr * Math.sin(a);
          if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }
        ctx.closePath(); ctx.fill();
      }

      function draw() {
        var ctx = cs.ctx, i, j;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var op = optima();

        // сетка
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        for (i = -4; i <= 4; i += 2) { ctx.beginPath(); ctx.moveTo(px(i), box.y); ctx.lineTo(px(i), box.y + box.h); ctx.stroke(); }
        for (i = -3; i <= 3; i += 1) { ctx.beginPath(); ctx.moveTo(box.x, py(i)); ctx.lineTo(box.x + box.w, py(i)); ctx.stroke(); }
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("w₁", box.x + box.w / 2, box.y + box.h + 22);
        ctx.save(); ctx.translate(box.x - 34, box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("w₂", 0, 0); ctx.restore();

        // клиенты
        var maxn = 1;
        for (i = 0; i < CL.length; i += 1) maxn = Math.max(maxn, CL[i].n);
        for (i = 0; i < CL.length; i += 1) {
          var c = CL[i], sx = px(c.m[0]), sy = py(c.m[1]);
          var rad = 8 + 26 * Math.sqrt(c.n / maxn);
          ctx.fillStyle = "rgba(49,95,140,0.10)";
          ctx.beginPath(); ctx.arc(sx, sy, rad, 0, 7); ctx.fill();
          ctx.strokeStyle = "rgba(49,95,140,0.35)"; ctx.lineWidth = 1; ctx.stroke();
          star(ctx, sx, sy, 7, C.blue);
          ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(String(c.n), sx, sy + rad + 13);
        }

        // локальные траектории текущего раунда
        ctx.strokeStyle = "rgba(165,121,32,0.75)"; ctx.lineWidth = 1.4;
        for (j = 0; j < anim.local.length; j += 1) {
          var p = anim.local[j];
          ctx.beginPath(); ctx.moveTo(px(p[0][0]), py(p[0][1]));
          for (i = 1; i < p.length; i += 1) ctx.lineTo(px(p[i][0]), py(p[i][1]));
          ctx.stroke();
          var last = p[p.length - 1];
          ctx.fillStyle = C.gold; ctx.beginPath(); ctx.arc(px(last[0]), py(last[1]), 3.4, 0, 7); ctx.fill();
        }

        // мишени
        star(ctx, px(op.rec[0]), py(op.rec[1]), 13, "rgba(49,95,140,0.95)");
        star(ctx, px(op.uni[0]), py(op.uni[1]), 13, "rgba(185,74,59,0.95)");
        ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillStyle = C.blue; ctx.fillText("цель «по строкам»", px(op.rec[0]) + 15, py(op.rec[1]) - 6);
        ctx.fillStyle = C.red; ctx.fillText("цель «по клиентам»", px(op.uni[0]) + 15, py(op.uni[1]) + 16);

        // траектория глобальной модели
        ctx.strokeStyle = C.green; ctx.lineWidth = 2.4; ctx.beginPath();
        for (i = 0; i < anim.path.length; i += 1) {
          var s = anim.path[i];
          if (i === 0) ctx.moveTo(px(s[0]), py(s[1])); else ctx.lineTo(px(s[0]), py(s[1]));
        }
        ctx.stroke();
        ctx.fillStyle = C.green; ctx.beginPath(); ctx.arc(px(anim.w[0]), py(anim.w[1]), 7, 0, 7); ctx.fill();
        ctx.strokeStyle = PB; ctx.lineWidth = 1.4; ctx.stroke();

        // правая панель: локальная потеря каждого клиента
        var bx = box.x + box.w + 62, bw = W - bx - 30, bh = 26;
        ctx.fillStyle = C.ink || "#171915"; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("Потеря у каждого клиента", bx, box.y + 6);
        var losses = [], worst = 0, mean = 0;
        for (i = 0; i < CL.length; i += 1) {
          var L = Math.pow(anim.w[0] - CL[i].m[0], 2) + Math.pow(anim.w[1] - CL[i].m[1], 2);
          losses.push(L); worst = Math.max(worst, L); mean += L / CL.length;
        }
        var scale = Math.max(1e-6, worst);
        for (i = 0; i < CL.length; i += 1) {
          var yy = box.y + 26 + i * (bh + 12);
          ctx.fillStyle = "rgba(110,114,106,0.14)"; ctx.fillRect(bx, yy, bw, bh);
          ctx.fillStyle = losses[i] >= worst - 1e-9 ? C.red : C.blue;
          ctx.fillRect(bx, yy, bw * losses[i] / scale, bh);
          ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "left";
          ctx.fillText("клиент " + (i + 1) + " · " + CL[i].n + " строк", bx + 6, yy + bh + 10);
          ctx.textAlign = "right"; ctx.fillStyle = C.ink || "#171915";
          ctx.fillText(losses[i].toFixed(2), bx + bw - 6, yy + 17);
          ctx.textAlign = "left";
        }

        output.set([
          { label: "Раунд", value: String(anim.round) + " / " + MAXR, color: C.green },
          { label: "До цели «по строкам»", value: Math.hypot(anim.w[0] - op.rec[0], anim.w[1] - op.rec[1]).toFixed(2), color: C.blue },
          { label: "До цели «по клиентам»", value: Math.hypot(anim.w[0] - op.uni[0], anim.w[1] - op.uni[1]).toFixed(2), color: C.red },
          { label: "Client drift (разброс обновлений)", value: anim.drift.toFixed(2), color: C.gold },
          { label: "Худший клиент / средний", value: worst.toFixed(2) + " / " + mean.toFixed(2), color: C.violet || C.red }
        ]);
      }

      K.slider(controls, { label: "Локальных эпох E", min: 1, max: 20, step: 1, value: state.E,
        format: function (v) { return String(v); } }, function (v) { state.E = v; reset(); });

      K.slider(controls, { label: "Неоднородность данных", min: 0, max: 1, step: 0.05, value: state.het,
        format: function (v) { return v.toFixed(2).replace(".", ","); } }, function (v) { state.het = v; reset(); });

      K.slider(controls, { label: "Неравенство размеров", min: 1, max: 1.8, step: 0.05, value: state.ineq,
        format: function (v) { return "×" + v.toFixed(2).replace(".", ","); } }, function (v) { state.ineq = v; reset(); });

      K.slider(controls, { label: "Шум приватности σ", min: 0, max: 2, step: 0.1, value: state.sigma,
        format: function (v) { return v.toFixed(1).replace(".", ","); } }, function (v) { state.sigma = v; reset(); });

      K.segmented(controls, {
        label: "Веса агрегации",
        value: "records",
        options: [{ value: "records", label: "по строкам" }, { value: "clients", label: "по клиентам" }]
      }, function (v) { state.weight = v; reset(); });

      K.segmented(controls, {
        label: "Доля клиентов в раунде",
        value: "1",
        options: [{ value: "0.4", label: "40 %" }, { value: "0.7", label: "70 %" }, { value: "1", label: "все" }]
      }, function (v) { state.frac = parseFloat(v); reset(); });

      reset();
      return function () { if (anim.raf) cancelAnimationFrame(anim.raf); cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
