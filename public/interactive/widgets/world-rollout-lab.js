// Lesson 88: world-rollout-lab — teacher forcing vs free rollout of a world model.
// A slightly wrong model of gravity looks perfect one step ahead and drifts away over a
// long rollout; periodic observation resets the accumulated error.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("world-rollout-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 500;
      var TOPH = 300, BOTY = 330, BOTH = 140;
      var PB = "#fffef9";

      var eps = 4;        // ошибка модели в гравитации, %
      var Hsteps = 90;    // длина прогноза
      var kobs = 0;       // коррекция каждые k шагов, 0 = никогда
      var mode = 0;       // 0 — обе линии, 1 — только rollout

      // --- истинная физика (единицы: метры, секунды)
      var DT = 1 / 30, G = 9.81, E = 0.78;
      var X0 = 0.2, Y0 = 2.2, VX0 = 1.6, VY0 = 0.6;

      function step(s, g) {
        var vy = s.vy - g * DT;
        var x = s.x + s.vx * DT, y = s.y + vy * DT;
        var vx = s.vx;
        if (y < 0) { y = -y * E; vy = -vy * E; }
        if (x > 4.6) { x = 9.2 - x; vx = -vx; }
        if (x < 0) { x = -x; vx = -vx; }
        return { x: x, y: y, vx: vx, vy: vy };
      }
      function copy(s) { return { x: s.x, y: s.y, vx: s.vx, vy: s.vy }; }

      function simulate() {
        var gm = G * (1 + eps / 100);
        var truth = [{ x: X0, y: Y0, vx: VX0, vy: VY0 }];
        var i;
        for (i = 0; i < Hsteps; i += 1) truth.push(step(truth[i], G));
        var free = [copy(truth[0])], tf = [copy(truth[0])], errFree = [0], errTF = [0];
        var st = copy(truth[0]), nobs = 0;
        for (i = 0; i < Hsteps; i += 1) {
          st = step(st, gm);
          if (kobs > 0 && (i + 1) % kobs === 0) { st = copy(truth[i + 1]); nobs += 1; }
          free.push(copy(st));
          errFree.push(Math.hypot(st.x - truth[i + 1].x, st.y - truth[i + 1].y));
          var one = step(truth[i], gm);          // teacher forcing: старт из истины
          tf.push(one);
          errTF.push(Math.hypot(one.x - truth[i + 1].x, one.y - truth[i + 1].y));
        }
        var m1 = 0;
        for (i = 1; i <= Hsteps; i += 1) m1 += errTF[i];
        return {
          truth: truth, free: free, tf: tf, errFree: errFree, errTF: errTF,
          one: m1 / Hsteps, last: errFree[Hsteps], nobs: nobs
        };
      }

      K.hint(root, "Модель мира ошибается в гравитации всего на несколько процентов. Прогноз на один шаг вперёд из настоящего состояния (teacher forcing) при этом почти идеален — а свободный rollout, где модель кормит саму себя, уходит от истины всё дальше. Включите наблюдения: каждая коррекция обнуляет накопленный дрейф.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Сверху — траектории мяча: чёрная истинная, красная свободный rollout, синие крестики — одношаговые прогнозы из истинного состояния. Снизу — модуль ошибки положения по шагам. Одношаговая ошибка почти не зависит от горизонта; ошибка rollout растёт, пока не приходит наблюдение.");
      var cs = K.makeCanvas(stage, W, H, { label: "Траектории и рост ошибки свободного прогноза", onResize: draw });

      var OX = 60, OY = TOPH - 30, SX = 165, SY = 105;
      function m2s(x, y) { return [OX + x * SX, OY - y * SY]; }

      function draw() {
        var ctx = cs.ctx, i, p;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var S = simulate();

        // ---------- верх: сцена
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var o = m2s(0, 0), xe = m2s(4.8, 0), ye = m2s(0, 2.5);
        ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(xe[0], xe[1]);
        ctx.moveTo(o[0], o[1]); ctx.lineTo(ye[0], ye[1]); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("пол", xe[0] - 20, o[1] + 16);
        ctx.save(); ctx.translate(OX - 34, m2s(0, 1.2)[1]); ctx.rotate(-Math.PI / 2);
        ctx.fillText("высота, м", 0, 0); ctx.restore();

        ctx.strokeStyle = "#171915"; ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (i = 0; i < S.truth.length; i += 1) {
          p = m2s(S.truth[i].x, S.truth[i].y);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();

        ctx.strokeStyle = C.red; ctx.lineWidth = 2.4;
        ctx.beginPath();
        for (i = 0; i < S.free.length; i += 1) {
          p = m2s(S.free[i].x, S.free[i].y);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();

        if (mode === 0) {
          ctx.strokeStyle = C.blue; ctx.lineWidth = 1.2;
          for (i = 1; i < S.tf.length; i += 3) {
            p = m2s(S.tf[i].x, S.tf[i].y);
            ctx.beginPath();
            ctx.moveTo(p[0] - 3, p[1] - 3); ctx.lineTo(p[0] + 3, p[1] + 3);
            ctx.moveTo(p[0] + 3, p[1] - 3); ctx.lineTo(p[0] - 3, p[1] + 3);
            ctx.stroke();
          }
        }
        // конечные точки
        var pt = m2s(S.truth[Hsteps].x, S.truth[Hsteps].y);
        var pf = m2s(S.free[Hsteps].x, S.free[Hsteps].y);
        ctx.fillStyle = "#171915"; ctx.beginPath(); ctx.arc(pt[0], pt[1], 6, 0, 7); ctx.fill();
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(pf[0], pf[1], 6, 0, 7); ctx.fill();
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(pt[0], pt[1]); ctx.lineTo(pf[0], pf[1]); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.gold; ctx.textAlign = "left";
        ctx.fillText("расхождение " + S.last.toFixed(2) + " м", Math.min(pt[0], pf[0]) + 10, Math.min(pt[1], pf[1]) - 10);

        ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = "#171915"; ctx.fillText("истина", W - 250, 24);
        ctx.fillStyle = C.red; ctx.fillText("свободный rollout", W - 250, 42);
        if (mode === 0) { ctx.fillStyle = C.blue; ctx.fillText("одношаговые прогнозы", W - 250, 60); }

        // ---------- низ: ошибка по шагам
        var maxE = 0;
        for (i = 0; i <= Hsteps; i += 1) if (S.errFree[i] > maxE) maxE = S.errFree[i];
        maxE = Math.max(maxE, 0.05);
        function e2s(i, e) { return [OX + i / Hsteps * (W - OX - 40), BOTY + BOTH - e / maxE * BOTH]; }
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(OX, BOTY); ctx.lineTo(OX, BOTY + BOTH);
        ctx.lineTo(W - 40, BOTY + BOTH); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        ctx.fillText(maxE.toFixed(2) + " м", OX - 8, BOTY + 6);
        ctx.fillText("0", OX - 8, BOTY + BOTH + 4);
        ctx.textAlign = "center";
        ctx.fillText("шаг прогноза", (OX + W - 40) / 2, BOTY + BOTH + 26);

        ctx.strokeStyle = C.red; ctx.lineWidth = 2.2; ctx.beginPath();
        for (i = 0; i <= Hsteps; i += 1) {
          p = e2s(i, S.errFree[i]);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();
        ctx.strokeStyle = C.blue; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]); ctx.beginPath();
        for (i = 0; i <= Hsteps; i += 1) {
          p = e2s(i, S.errTF[i]);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke(); ctx.setLineDash([]);
        if (kobs > 0) {
          ctx.strokeStyle = "rgba(56,115,93,0.45)"; ctx.lineWidth = 1;
          for (i = kobs; i <= Hsteps; i += kobs) {
            p = e2s(i, 0);
            ctx.beginPath(); ctx.moveTo(p[0], BOTY); ctx.lineTo(p[0], BOTY + BOTH); ctx.stroke();
          }
          ctx.fillStyle = C.green; ctx.textAlign = "left";
          ctx.fillText("зелёные линии — наблюдения", OX + 8, BOTY - 6);
        }

        output.set([
          { label: "Ошибка одного шага (среднее)", value: S.one.toFixed(4) + " м", color: C.blue },
          { label: "Ошибка на шаге " + Hsteps, value: S.last.toFixed(3) + " м", color: C.red },
          { label: "Во сколько раз хуже", value: (S.last / Math.max(S.one, 1e-9)).toFixed(0) + "×", color: C.gold },
          { label: "Наблюдений за прогноз", value: String(S.nobs), color: C.green }
        ]);
      }

      K.slider(controls, {
        label: "Ошибка модели в гравитации", min: 0, max: 10, step: 0.5, value: eps,
        format: function (v) { return v.toFixed(1) + " %"; }
      }, function (v) { eps = v; draw(); });
      K.slider(controls, {
        label: "Горизонт прогноза", min: 20, max: 180, step: 5, value: Hsteps,
        format: function (v) { return v + " шагов"; }
      }, function (v) { Hsteps = v; draw(); });
      K.slider(controls, {
        label: "Наблюдение каждые", min: 0, max: 60, step: 5, value: kobs,
        format: function (v) { return v === 0 ? "никогда" : v + " шагов"; }
      }, function (v) { kobs = v; draw(); });
      K.segmented(controls, {
        label: "Показывать", value: 0,
        options: [{ label: "оба режима", value: 0 }, { label: "только rollout", value: 1 }]
      }, function (v) { mode = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
