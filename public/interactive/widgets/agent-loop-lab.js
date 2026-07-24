// Lesson 89: agent-loop-lab — chain of steps, verifier quality and retry budget.
// Move step accuracy, chain length, verifier type and retry budget; watch the
// end-to-end success probability, the cost in calls and one simulated trace.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("agent-loop-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;

      // Detection / false-flag rates measured on the real SMS pipeline of the lesson.
      var MODES = {
        none: { label: "нет проверки", d: 0, f: 0, verify: false },
        restate: { label: "пересказ", d: 0, f: 0, verify: true },
        conf: { label: "порог уверенности", d: 0.344, f: 0.021, verify: true },
        indep: { label: "независимая улика", d: 0.639, f: 0.021, verify: true }
      };

      var state = { p: 0.9765, n: 40, mode: "none", k: 1 };

      // ---- deterministic PRNG so the picture never flickers between redraws
      function lcg(seed) {
        var s = seed >>> 0;
        return function () {
          s = (1664525 * s + 1013904223) >>> 0;
          return s / 4294967296;
        };
      }

      // Probability that one step ends correct with up to k retries, and the
      // expected number of attempts spent on it.
      function stepMath(p, m, k) {
        var d = m.verify ? m.d : 0;
        var f = m.verify ? m.f : 0;
        var again = p * f + (1 - p) * d;
        var a = 0, c = 0, i;
        if (!m.verify) return { a: p, calls: 1 };
        for (i = 0; i <= k; i += 1) {
          a = p * (1 - f) + again * a;
          c = 1 + again * c;
        }
        return { a: a, calls: 2 * c };
      }

      // One simulated chain: per step returns 0 ok, 1 caught-and-fixed, 2 silent error,
      // 3 gave up after the retry budget (flagged but never fixed).
      function simulate(rnd, p, m, k, n) {
        var out = [], i, j, code, done, fixed;
        for (i = 0; i < n; i += 1) {
          code = 0; done = false; fixed = false;
          for (j = 0; j <= k && !done; j += 1) {
            var good = rnd() < p;
            var flagged = m.verify && (good ? rnd() < m.f : rnd() < m.d);
            if (!flagged) { code = good ? (fixed ? 1 : 0) : 2; done = true; }
            else { fixed = true; }
          }
          if (!done) code = 3;
          out.push(code);
        }
        return out;
      }

      K.hint(root, "Цепочка обязательных шагов рвётся там, где рвётся любой её шаг. Двигайте точность шага и длину задачи — и следите не за шагом, а за вероятностью пройти всю цепь. Затем меняйте verifier: «пересказ» не приносит новой улики и не меняет ничего, независимая проверка распрямляет кривую. Бюджет повторов почти весь окупается первым повтором.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева — один смоделированный прогон: квадрат на каждый шаг. Зелёный — шаг прошёл сразу, синий — ошибка поймана проверкой и исправлена повтором, красный — ошибка молча ушла дальше, серый — бюджет повторов исчерпан. Справа — вероятность безошибочной задачи как функция длины цепочки: пунктир показывает ту же цепочку без проверки.");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Цепочка шагов агента, verifier и вероятность довести задачу до конца",
        onResize: draw,
        drag: false
      });

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var m = MODES[state.mode];
        var k = m.verify ? state.k : 0;
        var st = stepMath(state.p, m, k);
        var bare = state.p;

        // ---------------------------------------------------------- trace panel
        var TX = 40, TY = 54, cell = 26, gap = 5, cols = 8;
        ctx.fillStyle = C.ink;
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("один прогон задачи из " + state.n + " шагов", TX, 34);

        var rnd = lcg(20260089);
        var trace = simulate(rnd, state.p, m, k, state.n);
        var silent = 0, caught = 0, gaveup = 0;
        for (var i = 0; i < trace.length; i += 1) {
          var col = i % cols, rowI = Math.floor(i / cols);
          var x = TX + col * (cell + gap), y = TY + rowI * (cell + gap);
          var code = trace[i];
          if (code === 2) silent += 1;
          if (code === 1) caught += 1;
          if (code === 3) gaveup += 1;
          ctx.fillStyle = code === 0 ? "rgba(56,115,93,0.75)"
            : code === 1 ? "rgba(49,95,140,0.85)"
              : code === 2 ? "rgba(185,74,59,0.9)" : "rgba(110,114,106,0.55)";
          ctx.fillRect(x, y, cell, cell);
          ctx.strokeStyle = C.paper; ctx.lineWidth = 1.4;
          ctx.strokeRect(x, y, cell, cell);
        }
        var traceBottom = TY + Math.ceil(state.n / cols) * (cell + gap) + 14;
        ctx.fillStyle = silent > 0 ? C.red : C.green;
        ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText(silent > 0
          ? "итог прогона: " + silent + " ошибок ушло в ответ"
          : (gaveup > 0 ? "итог прогона: остановка, бюджет исчерпан" : "итог прогона: задача выполнена"),
          TX, traceBottom);
        ctx.fillStyle = C.muted;
        ctx.fillText("исправлено повтором: " + caught, TX, traceBottom + 20);

        // ---------------------------------------------------------- curve panel
        var GX = 400, GY = 60, GW = 440, GH = 330;
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(GX, GY); ctx.lineTo(GX, GY + GH); ctx.lineTo(GX + GW, GY + GH);
        ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        for (var t = 0; t <= 1.0001; t += 0.25) {
          var yy = GY + GH - t * GH;
          ctx.fillText(t.toFixed(2).replace(".", ","), GX - 8, yy + 4);
          ctx.strokeStyle = "rgba(222,221,212,0.7)";
          ctx.beginPath(); ctx.moveTo(GX, yy); ctx.lineTo(GX + GW, yy); ctx.stroke();
        }
        ctx.textAlign = "center";
        var NMAX = 60;
        for (var nn = 0; nn <= NMAX; nn += 15) {
          var xx = GX + (nn / NMAX) * GW;
          ctx.fillText(String(nn), xx, GY + GH + 18);
        }
        ctx.fillText("число обязательных шагов", GX + GW / 2, GY + GH + 38);
        ctx.save();
        ctx.translate(GX - 44, GY + GH / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("вероятность довести задачу", 0, 0);
        ctx.restore();

        function curve(a, color, dashed, width) {
          ctx.strokeStyle = color; ctx.lineWidth = width;
          if (dashed) ctx.setLineDash([5, 4]); else ctx.setLineDash([]);
          ctx.beginPath();
          for (var q = 1; q <= NMAX; q += 1) {
            var px = GX + (q / NMAX) * GW, py = GY + GH - Math.pow(a, q) * GH;
            if (q === 1) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
          ctx.setLineDash([]);
        }
        curve(bare, "rgba(110,114,106,0.9)", true, 1.6);
        curve(st.a, m.verify && m.d > 0 ? C.green : C.red, false, 2.6);

        var mx = GX + (state.n / NMAX) * GW;
        var my = GY + GH - Math.pow(st.a, state.n) * GH;
        ctx.strokeStyle = "rgba(110,114,106,0.5)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mx, GY); ctx.lineTo(mx, GY + GH); ctx.stroke();
        ctx.fillStyle = C.ink;
        ctx.beginPath(); ctx.arc(mx, my, 6, 0, 7); ctx.fill();
        ctx.strokeStyle = C.paper; ctx.lineWidth = 1.4; ctx.stroke();

        ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.fillText("пунктир — та же цепочка без проверки", GX + 8, GY - 14);

        var pTask = Math.pow(st.a, state.n);
        output.set([
          { label: "точность шага p", value: state.p.toFixed(4).replace(".", ","), color: C.blue },
          { label: "эффективная точность шага", value: st.a.toFixed(4).replace(".", ","), color: m.d > 0 ? C.green : C.muted },
          { label: "вероятность довести задачу", value: pTask.toFixed(3).replace(".", ","), color: pTask > 0.8 ? C.green : (pTask > 0.5 ? C.gold : C.red) },
          { label: "без проверки было бы", value: Math.pow(bare, state.n).toFixed(3).replace(".", ","), color: C.muted },
          { label: "вызовов на шаг", value: st.calls.toFixed(2).replace(".", ","), color: C.violet }
        ]);
      }

      K.slider(controls, {
        label: "точность одного шага p", min: 0.9, max: 0.999, step: 0.0005, value: state.p,
        format: function (v) { return v.toFixed(4).replace(".", ","); }
      }, function (v) { state.p = v; draw(); });

      K.slider(controls, {
        label: "длина цепочки n", min: 5, max: 60, step: 1, value: state.n,
        format: function (v) { return String(Math.round(v)); }
      }, function (v) { state.n = Math.round(v); draw(); });

      K.slider(controls, {
        label: "повторов на шаг k", min: 0, max: 3, step: 1, value: state.k,
        format: function (v) { return String(Math.round(v)); }
      }, function (v) { state.k = Math.round(v); draw(); });

      K.segmented(controls, {
        label: "verifier",
        value: state.mode,
        options: [
          { value: "none", label: "нет" },
          { value: "restate", label: "пересказ" },
          { value: "conf", label: "порог уверенности" },
          { value: "indep", label: "независимая улика" }
        ]
      }, function (v) { state.mode = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
