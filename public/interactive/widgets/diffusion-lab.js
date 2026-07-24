// Lesson 86: diffusion-lab — forward noising and the learned reverse chain on a 2D toy set.
// The score is the EXACT score of the (kernel-smoothed) empirical distribution, so the
// reader watches the optimal denoiser itself: schedule sets the difficulty, the number of
// steps sets the fidelity, guidance trades condition-matching against diversity.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("diffusion-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var PAD = 54, PLOT = 400;                 // square plotting area on the left
      var T = 1000;
      var mode = "forward";                     // forward | reverse
      var sched = "cosine";
      var tNow = 400, steps = 40, guide = 1, cls = -1;
      var seed = 12345;

      // ---------- deterministic pseudo-random (so the picture is reproducible)
      function mulberry(a) {
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          var t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      var rnd = mulberry(seed);
      function gauss() {
        var u = 0, v = 0;
        while (u === 0) u = rnd();
        while (v === 0) v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // ---------- data: two modes, class 0 = ring arc, class 1 = tight blob
      var DATA = [];
      (function build() {
        var r = mulberry(7);
        var i;
        for (i = 0; i < 70; i += 1) {
          var a = Math.PI * (0.15 + 1.25 * i / 70);
          DATA.push({ x: -0.75 + 0.85 * Math.cos(a) + 0.07 * (r() - 0.5),
                      y: 0.15 + 0.85 * Math.sin(a) + 0.07 * (r() - 0.5), c: 0 });
        }
        for (i = 0; i < 50; i += 1) {
          DATA.push({ x: 0.95 + 0.22 * (r() - 0.5) * 2,
                      y: -0.55 + 0.22 * (r() - 0.5) * 2, c: 1 });
        }
      })();
      var KERNEL = 0.05;                        // bandwidth of the smoothed empirical law

      // ---------- schedules: abar[t], t = 0..T
      function makeSchedule(kind) {
        var ab = new Float64Array(T + 1), t;
        if (kind === "linear") {
          var p = 1;
          ab[0] = 1;
          for (t = 1; t <= T; t += 1) { p *= 1 - (1e-4 + (0.02 - 1e-4) * (t - 1) / (T - 1)); ab[t] = p; }
        } else {
          var s = 0.008;
          var f0 = Math.pow(Math.cos((s / (1 + s)) * Math.PI / 2), 2);
          for (t = 0; t <= T; t += 1) {
            ab[t] = Math.pow(Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2), 2) / f0;
          }
        }
        return ab;
      }
      var AB = { linear: makeSchedule("linear"), cosine: makeSchedule("cosine") };
      function abar(t) { return Math.min(1, Math.max(1e-8, AB[sched][Math.max(0, Math.min(T, Math.round(t)))])); }

      // ---------- exact posterior mean E[x0|xt] for the smoothed empirical law
      function posterior(px, py, ab, klass) {
        var v = (1 - ab) + ab * KERNEL * KERNEL;    // variance of the diffused kernel
        var sa = Math.sqrt(ab);
        var best = -Infinity, i, lg = new Array(DATA.length);
        for (i = 0; i < DATA.length; i += 1) {
          if (klass >= 0 && DATA[i].c !== klass) { lg[i] = -Infinity; continue; }
          var dx = px - sa * DATA[i].x, dy = py - sa * DATA[i].y;
          lg[i] = -(dx * dx + dy * dy) / (2 * v);
          if (lg[i] > best) best = lg[i];
        }
        var sw = 0, mx = 0, my = 0, ent = 0, wts = new Array(DATA.length);
        for (i = 0; i < DATA.length; i += 1) {
          var w = lg[i] === -Infinity ? 0 : Math.exp(lg[i] - best);
          wts[i] = w; sw += w;
        }
        for (i = 0; i < DATA.length; i += 1) {
          var q = wts[i] / sw;
          mx += q * DATA[i].x; my += q * DATA[i].y;
          if (q > 1e-12) ent -= q * Math.log(q);
        }
        return { x: mx, y: my, neff: Math.exp(ent) };
      }
      function epsOf(px, py, ab, klass) {
        var p = posterior(px, py, ab, klass), sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
        if (sn < 1e-6) return { ex: 0, ey: 0, neff: p.neff };
        return { ex: (px - sa * p.x) / sn, ey: (py - sa * p.y) / sn, neff: p.neff };
      }

      // ---------- reverse chain
      var samples = [], trace = [];
      function runChain() {
        rnd = mulberry(seed);
        var n = 40, i, j;
        var ts = [];
        for (i = 0; i < steps; i += 1) ts.push(Math.max(1, Math.round(T - i * (T - 1) / (steps - 1))));
        samples = [];
        trace = [];
        for (i = 0; i < n; i += 1) {
          var x = gauss(), y = gauss(), path = [[x, y]];
          for (j = 0; j < ts.length; j += 1) {
            var t = ts[j], sNext = j + 1 < ts.length ? ts[j + 1] : 0;
            var ab = abar(t), abS = abar(sNext);
            var ex, ey;
            if (cls < 0) {
              var e0 = epsOf(x, y, ab, -1); ex = e0.ex; ey = e0.ey;
            } else {
              var eu = epsOf(x, y, ab, -1), ec = epsOf(x, y, ab, cls);
              ex = eu.ex + guide * (ec.ex - eu.ex);
              ey = eu.ey + guide * (ec.ey - eu.ey);
            }
            var sa = Math.sqrt(ab);
            var x0x = (x - Math.sqrt(1 - ab) * ex) / sa, x0y = (y - Math.sqrt(1 - ab) * ey) / sa;
            x0x = Math.max(-2.5, Math.min(2.5, x0x)); x0y = Math.max(-2.5, Math.min(2.5, x0y));
            if (sNext > 0) {
              var sig = Math.sqrt(Math.max(0, (1 - abS) / (1 - ab) * (1 - ab / abS)));
              var dir = Math.sqrt(Math.max(0, 1 - abS - sig * sig));
              x = Math.sqrt(abS) * x0x + dir * ex + sig * gauss();
              y = Math.sqrt(abS) * x0y + dir * ey + sig * gauss();
            } else { x = x0x; y = x0y; }
            if (i < 4) path.push([x, y]);
          }
          samples.push({ x: x, y: y });
          if (i < 4) trace.push(path);
        }
      }

      // ---------- metrics
      function metrics() {
        var i, j, near = 0, hit = 0, div = 0, cnt = 0, arc = 0;
        for (i = 0; i < samples.length; i += 1) {
          var bd = Infinity, bc = 0;
          for (j = 0; j < DATA.length; j += 1) {
            var dx = samples[i].x - DATA[j].x, dy = samples[i].y - DATA[j].y;
            var d = Math.sqrt(dx * dx + dy * dy);
            if (d < bd) { bd = d; bc = DATA[j].c; }
          }
          near += bd;
          if (cls < 0 ? bd < 0.15 : bc === cls) hit += 1;
          if (bc === 0) arc += 1;
          for (j = i + 1; j < samples.length; j += 1) {
            var ax = samples[i].x - samples[j].x, ay = samples[i].y - samples[j].y;
            div += Math.sqrt(ax * ax + ay * ay); cnt += 1;
          }
        }
        return {
          near: near / Math.max(1, samples.length),
          hit: hit / Math.max(1, samples.length),
          div: div / Math.max(1, cnt),
          arc: arc / Math.max(1, samples.length),
        };
      }

      // ---------- layout
      K.hint(root, "Слева — плоскость данных: два «класса» (дуга и пятно). В режиме «прямой процесс» ползунок t показывает, как одна и та же выборка растворяется в гауссовом шуме; справа видно, какая доля дисперсии ещё сигнал. В режиме «обратный процесс» цепочка стартует из чистого шума и идёт назад: меняйте число шагов и силу guidance и смотрите, что происходит с попаданием в класс и с разнообразием.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Score здесь не приближённый, а точный: это score сглаженного эмпирического распределения показанных точек. Поэтому всё, что вы видите, — свойства самой схемы diffusion, а не недоучившейся сети.");
      var cs = K.makeCanvas(stage, W, H, { label: "Прямой и обратный процессы diffusion на плоскости", onResize: draw, drag: false });

      function px(x) { return PAD + (x + 2.6) / 5.2 * PLOT; }
      function py(y) { return PAD + PLOT - (y + 2.6) / 5.2 * PLOT; }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        // frame
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(PAD, PAD, PLOT, PLOT);
        var ab = abar(tNow), sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab), i, j;

        if (mode === "forward") {
          rnd = mulberry(seed);
          for (i = 0; i < DATA.length; i += 1) {
            var nx = sa * DATA[i].x + sn * gauss(), ny = sa * DATA[i].y + sn * gauss();
            ctx.fillStyle = DATA[i].c === 0 ? C.blue : C.red;
            ctx.globalAlpha = 0.85;
            ctx.beginPath(); ctx.arc(px(nx), py(ny), 4, 0, 7); ctx.fill();
          }
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.muted; ctx.textAlign = "left";
          ctx.fillText("выборка на уровне шума t = " + tNow, PAD, PAD - 14);
        } else {
          // faint data cloud for reference
          ctx.globalAlpha = 0.28;
          for (i = 0; i < DATA.length; i += 1) {
            ctx.fillStyle = DATA[i].c === 0 ? C.blue : C.red;
            ctx.beginPath(); ctx.arc(px(DATA[i].x), py(DATA[i].y), 3.4, 0, 7); ctx.fill();
          }
          ctx.globalAlpha = 1;
          ctx.strokeStyle = "rgba(111,90,143,0.55)"; ctx.lineWidth = 1.2;
          for (i = 0; i < trace.length; i += 1) {
            ctx.beginPath();
            for (j = 0; j < trace[i].length; j += 1) {
              var p = trace[i][j];
              if (j === 0) ctx.moveTo(px(p[0]), py(p[1])); else ctx.lineTo(px(p[0]), py(p[1]));
            }
            ctx.stroke();
          }
          for (i = 0; i < samples.length; i += 1) {
            ctx.fillStyle = C.green;
            ctx.beginPath(); ctx.arc(px(samples[i].x), py(samples[i].y), 4.5, 0, 7); ctx.fill();
            ctx.strokeStyle = C.paper; ctx.lineWidth = 1.1; ctx.stroke();
          }
          ctx.fillStyle = C.muted; ctx.textAlign = "left";
          ctx.fillText("зелёные — сгенерированные образцы, фиолетовые нити — четыре траектории", PAD, PAD - 14);
        }

        // ---------- right panel: SNR curve of the current schedule
        var RX = PAD + PLOT + 64, RW = W - RX - 30, RY = PAD + 26, RH = PLOT - 60;
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(RX, RY, RW, RH);
        function sy(logsnr) {                    // log10 SNR from +4 down to -4
          return RY + (4 - logsnr) / 8 * RH;
        }
        ctx.strokeStyle = C.muted; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(RX, sy(0)); ctx.lineTo(RX + RW, sy(0)); ctx.stroke();
        ctx.setLineDash([]);
        ["linear", "cosine"].forEach(function (kind) {
          ctx.strokeStyle = kind === sched ? C.red : "rgba(110,114,106,0.45)";
          ctx.lineWidth = kind === sched ? 2.4 : 1.2;
          ctx.beginPath();
          for (var t = 1; t <= T; t += 5) {
            var a = AB[kind][t], s = Math.log(a / (1 - a)) / Math.LN10;
            s = Math.max(-4, Math.min(4, s));
            var X = RX + t / T * RW;
            if (t === 1) ctx.moveTo(X, sy(s)); else ctx.lineTo(X, sy(s));
          }
          ctx.stroke();
        });
        var curS = Math.max(-4, Math.min(4, Math.log(ab / (1 - ab)) / Math.LN10));
        var mx = RX + tNow / T * RW;
        ctx.strokeStyle = C.blue; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(mx, RY); ctx.lineTo(mx, RY + RH); ctx.stroke();
        ctx.fillStyle = C.blue; ctx.beginPath(); ctx.arc(mx, sy(curS), 5, 0, 7); ctx.fill();
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("lg SNR = +4", RX + 4, RY + 12);
        ctx.fillText("SNR = 1", RX + 4, sy(0) - 5);
        ctx.fillText("lg SNR = −4", RX + 4, RY + RH - 5);
        ctx.textAlign = "center";
        ctx.fillText("t", RX + RW / 2, RY + RH + 18);
        ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = C.ink;
        ctx.fillText("расписание " + sched, RX, RY - 10);

        // ---------- bar of signal vs noise
        var BY = PAD + PLOT + 22, BW = PLOT;
        ctx.fillStyle = "rgba(49,95,140,0.75)";
        ctx.fillRect(PAD, BY, BW * ab, 16);
        ctx.fillStyle = "rgba(185,74,59,0.65)";
        ctx.fillRect(PAD + BW * ab, BY, BW * (1 - ab), 16);
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("доля дисперсии: сигнал " + (100 * ab).toFixed(1) + "%  /  шум " + (100 * (1 - ab)).toFixed(1) + "%", PAD, BY + 30);

        var items = [
          { label: "√ᾱ (сигнал)", value: sa.toFixed(3), color: C.blue },
          { label: "√(1−ᾱ) (шум)", value: sn.toFixed(3), color: C.red },
          { label: "SNR", value: (ab / (1 - ab)).toFixed(3), color: C.gold }
        ];
        if (mode === "forward") {
          var pinfo = posterior(sa * DATA[0].x + sn * 0.3, sa * DATA[0].y + sn * -0.2, ab, -1);
          items.push({ label: "n_eff одного ответа", value: pinfo.neff.toFixed(1), color: C.violet });
        } else {
          var m = metrics();
          items.push({ label: "шагов цепочки", value: String(steps), color: C.ink });
          items.push({ label: cls < 0 ? "легло на данные (<0,15)" : "попало в нужный класс", value: (100 * m.hit).toFixed(0) + "%", color: C.blue });
          items.push({ label: "доля моды «дуга» (в данных 58%)", value: (100 * m.arc).toFixed(0) + "%", color: C.violet });
          items.push({ label: "разнообразие", value: m.div.toFixed(2), color: C.red });
          items.push({ label: "до ближайшей точки", value: m.near.toFixed(3), color: C.green });
        }
        out.set(items);
      }

      // ---------- controls
      K.segmented(controls, {
        label: "режим", value: "forward",
        options: [{ value: "forward", label: "прямой процесс" }, { value: "reverse", label: "обратный процесс" }]
      }, function (v) { mode = v; if (mode === "reverse") runChain(); draw(); });

      K.segmented(controls, {
        label: "расписание", value: "cosine",
        options: [{ value: "linear", label: "linear" }, { value: "cosine", label: "cosine" }]
      }, function (v) { sched = v; if (mode === "reverse") runChain(); draw(); });

      K.segmented(controls, {
        label: "условие", value: "-1",
        options: [{ value: "-1", label: "без условия" }, { value: "0", label: "класс «дуга»" }, { value: "1", label: "класс «пятно»" }]
      }, function (v) { cls = Number(v); if (mode === "reverse") runChain(); draw(); });

      K.slider(controls, { label: "уровень шума t", min: 0, max: 1000, step: 10, value: tNow },
        function (v) { tNow = v; draw(); });
      K.slider(controls, { label: "шагов в обратной цепочке", min: 2, max: 120, step: 1, value: steps },
        function (v) { steps = v; if (mode === "reverse") { runChain(); draw(); } });
      K.slider(controls, { label: "guidance w", min: 0, max: 6, step: 0.25, value: guide },
        function (v) { guide = v; if (mode === "reverse" && cls >= 0) { runChain(); draw(); } });

      var again = K.element("button", "kontur-int-segment", { type: "button", text: "новый шум" });
      again.style.margin = "0 6px";
      again.addEventListener("click", function () {
        seed = (seed * 1103515245 + 12345) % 2147483647;
        if (mode === "reverse") runChain();
        draw();
      });
      controls.appendChild(again);

      runChain();
      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
