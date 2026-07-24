// Lesson 71: td-learning-lab — учим таблицу по одному переходу.
// Три ученика (Q-learning, SARSA, табличный actor–critic) живут в мире с обрывом.
// Двигая alpha и epsilon, видно главный тезис урока: off-policy цель оценивает
// жадное продолжение и ведёт по краю, on-policy цель помнит про собственный шум
// и отходит от обрыва, а actor–critic меняет вероятности по знаку TD-ошибки.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("td-learning-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var ROWS = 4, COLS = 8, NS = ROWS * COLS, NA = 4;
      var START = (ROWS - 1) * COLS, GOAL = NS - 1;
      var CELL = 52, GX = 24, GY = 74;
      var PLOT_X = GX + COLS * CELL + 44, PLOT_Y = GY, PLOT_W = W - PLOT_X - 26, PLOT_H = ROWS * CELL;

      var algo = "q", alpha = 0.5, eps = 0.1;
      var Q, theta, V, episodes, returns, falls, running = true, lastPath = [];
      var seed = 20250771;

      function rnd() {                        // детерминированный ГПСЧ: опыт воспроизводим
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        return seed / 4294967296;
      }

      function reset() {
        Q = []; theta = []; V = [];
        for (var s = 0; s < NS; s += 1) {
          Q.push([0, 0, 0, 0]); theta.push([0, 0, 0, 0]); V.push(0);
        }
        episodes = 0; returns = []; falls = []; lastPath = [];
      }

      function step(s, a) {
        var r = (s / COLS) | 0, c = s % COLS;
        if (a === 0) r = Math.max(0, r - 1);
        else if (a === 1) c = Math.min(COLS - 1, c + 1);
        else if (a === 2) r = Math.min(ROWS - 1, r + 1);
        else c = Math.max(0, c - 1);
        var ns = r * COLS + c;
        if (r === ROWS - 1 && c >= 1 && c <= COLS - 2) return { s: START, r: -100, done: false, fell: true };
        if (ns === GOAL) return { s: ns, r: -1, done: true, fell: false };
        return { s: ns, r: -1, done: false, fell: false };
      }

      function argmax(v) {
        var best = 0, i;
        for (i = 1; i < v.length; i += 1) if (v[i] > v[best]) best = i;
        return best;
      }

      function softmax(v) {
        var m = Math.max(v[0], v[1], v[2], v[3]), e = [], sum = 0, i;
        for (i = 0; i < 4; i += 1) { e.push(Math.exp(Math.min(6, v[i] - m))); sum += e[i]; }
        for (i = 0; i < 4; i += 1) e[i] /= sum;
        return e;
      }

      function pick(s) {
        if (algo === "ac") {
          var p = softmax(theta[s]), u = rnd(), acc = 0;
          for (var i = 0; i < 4; i += 1) { acc += p[i]; if (u <= acc) return i; }
          return 3;
        }
        if (rnd() < eps) return Math.min(3, Math.floor(rnd() * 4));
        return argmax(Q[s]);
      }

      function episode() {
        var s = START, a = pick(s), total = 0, fell = 0, steps = 0, done = false;
        var path = [s];
        while (!done && steps < 240) {
          var t = step(s, a);
          total += t.r;
          if (t.fell) fell += 1;
          var na = pick(t.s);
          if (algo === "ac") {
            var delta = t.r + (t.done ? 0 : V[t.s]) - V[s];
            V[s] += 0.2 * delta;
            var p = softmax(theta[s]);
            for (var i = 0; i < 4; i += 1) {
              theta[s][i] += alpha * 0.25 * delta * ((i === a ? 1 : 0) - p[i]);
              theta[s][i] = Math.max(-8, Math.min(8, theta[s][i]));
            }
          } else {
            var boot = 0;
            if (!t.done) boot = algo === "q" ? Q[t.s][argmax(Q[t.s])] : Q[t.s][na];
            Q[s][a] += alpha * (t.r + boot - Q[s][a]);
          }
          s = t.s; a = na; steps += 1;
          path.push(s);
          done = t.done;
        }
        episodes += 1;
        returns.push(total); falls.push(fell);
        if (returns.length > 400) { returns.shift(); falls.shift(); }
        lastPath = path;
      }

      function scoreOf(s) { return algo === "ac" ? V[s] : Q[s][argmax(Q[s])]; }
      function policyOf(s) { return algo === "ac" ? argmax(theta[s]) : argmax(Q[s]); }

      function greedyPath() {
        var s = START, path = [s], seen = {}, guard = 0;
        seen[s] = true;
        while (s !== GOAL && guard < 40) {
          var t = step(s, policyOf(s));
          path.push(t.s);
          if (seen[t.s]) break;
          seen[t.s] = true;
          s = t.s; guard += 1;
        }
        return path;
      }

      function heat(v, lo, hi) {
        var u = hi > lo ? (v - lo) / (hi - lo) : 0;
        u = Math.max(0, Math.min(1, u));
        var r = Math.round(245 - 60 * u), g = Math.round(243 - 30 * u), b = Math.round(234 + 12 * u);
        return "rgb(" + r + "," + g + "," + b + ")";
      }

      function centre(s) {
        return [GX + (s % COLS) * CELL + CELL / 2, GY + (((s / COLS) | 0)) * CELL + CELL / 2];
      }

      function arrow(ctx, cx, cy, a, len, color, wdt) {
        var dx = a === 1 ? 1 : a === 3 ? -1 : 0, dy = a === 2 ? 1 : a === 0 ? -1 : 0;
        ctx.strokeStyle = color; ctx.lineWidth = wdt;
        ctx.beginPath();
        ctx.moveTo(cx - dx * len / 2, cy - dy * len / 2);
        ctx.lineTo(cx + dx * len / 2, cy + dy * len / 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx + dx * len / 2, cy + dy * len / 2);
        ctx.lineTo(cx + dx * len / 2 - dx * 6 - dy * 4, cy + dy * len / 2 - dy * 6 - dx * 4);
        ctx.lineTo(cx + dx * len / 2 - dx * 6 + dy * 4, cy + dy * len / 2 - dy * 6 + dx * 4);
        ctx.closePath();
        ctx.fillStyle = color; ctx.fill();
      }

      function mean(arr, n) {
        if (!arr.length) return 0;
        var k = Math.min(n, arr.length), sum = 0;
        for (var i = arr.length - k; i < arr.length; i += 1) sum += arr[i];
        return sum / k;
      }

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var lo = 1e9, hi = -1e9;
        for (i = 0; i < NS; i += 1) {
          var v = scoreOf(i);
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("мир с обрывом: цвет — ценность состояния, стрелка — жадное действие", GX, 34);
        ctx.fillStyle = C.muted; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("нижний ряд между S и G — обрыв (−100 и назад на старт); каждый шаг стоит −1", GX, 54);

        for (i = 0; i < NS; i += 1) {
          var r = (i / COLS) | 0, c = i % COLS;
          var x = GX + c * CELL, y = GY + r * CELL;
          var cliff = r === ROWS - 1 && c >= 1 && c <= COLS - 2;
          ctx.fillStyle = cliff ? "#eddcd7" : heat(scoreOf(i), lo, hi);
          ctx.fillRect(x, y, CELL, CELL);
          ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
          if (!cliff && i !== GOAL && episodes > 0) {
            arrow(ctx, x + CELL / 2, y + CELL / 2, policyOf(i), 22, "rgba(23,25,21,0.45)", 1.6);
          }
        }
        ctx.textAlign = "center"; ctx.font = "13px PT Sans, sans-serif";
        var sc = centre(START), gc = centre(GOAL);
        ctx.fillStyle = C.ink; ctx.fillText("S", sc[0], sc[1] + 5);
        ctx.fillStyle = C.green; ctx.fillText("G", gc[0], gc[1] + 5);
        ctx.fillStyle = C.red; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("обрыв", GX + COLS * CELL / 2, GY + (ROWS - 1) * CELL + CELL / 2 + 4);

        var gp = greedyPath();
        ctx.strokeStyle = algo === "q" ? C.red : algo === "sarsa" ? C.blue : C.violet;
        ctx.lineWidth = 3.2; ctx.beginPath();
        for (i = 0; i < gp.length; i += 1) {
          var p = centre(gp[i]);
          if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
        }
        ctx.stroke();

        // кривая возвратов
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(PLOT_X + 0.5, PLOT_Y + 0.5, PLOT_W, PLOT_H);
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("возврат за эпизод", PLOT_X, PLOT_Y - 10);
        ctx.fillText("−120", PLOT_X + 4, PLOT_Y + PLOT_H - 6);
        ctx.fillText("0", PLOT_X + 4, PLOT_Y + 14);
        var n = returns.length;
        if (n > 1) {
          ctx.strokeStyle = algo === "q" ? C.red : algo === "sarsa" ? C.blue : C.violet;
          ctx.lineWidth = 1.8; ctx.beginPath();
          for (i = 0; i < n; i += 1) {
            var xx = PLOT_X + PLOT_W * i / (n - 1);
            var yy = PLOT_Y + PLOT_H * Math.min(1, Math.max(0, -returns[i] / 120));
            if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
          }
          ctx.stroke();
        }
        var edge = 0;
        for (i = 0; i < gp.length; i += 1) if (((gp[i] / COLS) | 0) === ROWS - 2) edge += 1;
        var safe = edge <= 2;
        output.set([
          { label: "эпизодов прожито", value: String(episodes), color: C.ink },
          { label: "средний возврат (20 посл.)", value: episodes ? mean(returns, 20).toFixed(1) : "—", color: C.blue },
          { label: "падений в обрыв (20 посл.)", value: episodes ? mean(falls, 20).toFixed(2) : "—", color: C.red },
          {
            label: "жадный маршрут",
            value: episodes === 0 ? "—" : (gp[gp.length - 1] !== GOAL ? "ещё не дошёл" : (safe ? "в обход обрыва" : "впритык к обрыву")),
            color: safe ? C.green : C.gold,
          },
        ]);
      }

      K.hint(root, "Ученик не знает ни карты, ни вероятностей — только собственные переходы (s, a, r, s′). Запустите обучение и следите за двумя вещами сразу: как цвет ценности расползается от цели назад к старту и какой маршрут выбирает жадная политика. Q-learning в цели берёт max по действиям и потому оценивает идеальное продолжение — его маршрут жмётся к обрыву. SARSA подставляет действие, которое реально будет сделано, включая ε-случайные шаги, и отходит подальше. Actor–critic не хранит Q: critic держит V, actor двигает вероятности по знаку TD-ошибки.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var buttons = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Цвет клетки — оценка ценности, серые стрелки — жадное действие таблицы, толстая линия — маршрут из S при выключенном исследовании. Большое α делает таблицу нервной: она повторяет последний шумный опыт. При ε = 0 обучение слепнет — непроверенные клетки остаются с начальной оценкой навсегда.");

      var cs = K.makeCanvas(stage, W, H, { label: "Обучение таблицы ценностей по одному переходу", onResize: draw, drag: false });

      K.segmented(controls, {
        label: "ученик",
        value: "q",
        options: [
          { value: "q", label: "Q-learning" },
          { value: "sarsa", label: "SARSA" },
          { value: "ac", label: "actor–critic" },
        ],
      }, function (v) { algo = v; reset(); draw(); });

      K.slider(controls, {
        label: "шаг α", min: 0.05, max: 0.9, step: 0.05, value: 0.5,
        format: function (v) { return v.toFixed(2); },
      }, function (v) { alpha = v; draw(); });

      K.slider(controls, {
        label: "исследование ε", min: 0, max: 0.5, step: 0.02, value: 0.1,
        format: function (v) { return v.toFixed(2); },
      }, function (v) { eps = v; draw(); });

      var runBtn = K.element("button", "kontur-int-segment is-active", { type: "button", text: "пауза" });
      var stepBtn = K.element("button", "kontur-int-segment", { type: "button", text: "+50 эпизодов" });
      var resetBtn = K.element("button", "kontur-int-segment", { type: "button", text: "забыть всё" });
      runBtn.style.margin = stepBtn.style.margin = resetBtn.style.margin = "0 6px";
      runBtn.addEventListener("click", function () {
        running = !running;
        runBtn.textContent = running ? "пауза" : "продолжить";
        runBtn.classList.toggle("is-active", running);
      });
      stepBtn.addEventListener("click", function () {
        for (var i = 0; i < 50; i += 1) episode();
        draw();
      });
      resetBtn.addEventListener("click", function () { reset(); draw(); });
      buttons.appendChild(runBtn); buttons.appendChild(stepBtn); buttons.appendChild(resetBtn);

      reset();
      draw();

      var loop = K.visibleLoop(root, function () {
        if (!running) return;
        for (var i = 0; i < 4; i += 1) episode();
        draw();
      });

      return function () {
        loop.stop();
        cs.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
