// Lesson 69: mdp-bellman-lab — value iteration on a 5x5 grid MDP.
// Move gamma, the sideways slip and the number of Bellman backups; watch the
// wave of value spread from the goal, the greedy policy detour around the pit,
// and the gap between the best and the second-best action.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("mdp-bellman-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var NR = 5, NC = 5;
      var CELL = 74, OX = 40, OY = 52;
      var WALLS = [[1, 1], [2, 1], [3, 3]];
      var GOAL = [0, 4];
      var START = [4, 0];
      var pit = [1, 3];
      var GOAL_R = 10, PIT_R = -8;
      var ACT = ["U", "R", "D", "L"];
      var DR = { U: -1, R: 0, D: 1, L: 0 };
      var DC = { U: 0, R: 1, D: 0, L: -1 };
      var PERP = { U: ["L", "R"], D: ["R", "L"], R: ["U", "D"], L: ["D", "U"] };

      var gamma = 0.9, slip = 0.1, steps = 20, mode = "value";

      function isWall(r, c) {
        for (var i = 0; i < WALLS.length; i += 1) if (WALLS[i][0] === r && WALLS[i][1] === c) return true;
        return false;
      }
      function isGoal(r, c) { return r === GOAL[0] && c === GOAL[1]; }
      function isPit(r, c) { return r === pit[0] && c === pit[1]; }
      function isTerminal(r, c) { return isGoal(r, c) || isPit(r, c); }
      function idx(r, c) { return r * NC + c; }

      function move(r, c, a) {
        var nr = r + DR[a], nc = c + DC[a];
        if (nr < 0 || nr >= NR || nc < 0 || nc >= NC || isWall(nr, nc)) return [r, c];
        return [nr, nc];
      }
      function rewardOf(r, c) {
        if (isGoal(r, c)) return GOAL_R;
        if (isPit(r, c)) return PIT_R;
        return 0;
      }
      function backup(V, r, c, a) {
        var parts = [[a, 1 - 2 * slip], [PERP[a][0], slip], [PERP[a][1], slip]];
        var total = 0;
        for (var i = 0; i < parts.length; i += 1) {
          var t = move(r, c, parts[i][0]);
          total += parts[i][1] * (rewardOf(t[0], t[1]) + gamma * V[idx(t[0], t[1])]);
        }
        return total;
      }
      function sweep(V) {
        var Vn = V.slice();
        for (var r = 0; r < NR; r += 1) {
          for (var c = 0; c < NC; c += 1) {
            if (isWall(r, c) || isTerminal(r, c)) { Vn[idx(r, c)] = 0; continue; }
            var best = -1e9;
            for (var i = 0; i < 4; i += 1) best = Math.max(best, backup(V, r, c, ACT[i]));
            Vn[idx(r, c)] = best;
          }
        }
        return Vn;
      }
      function iterate(n) {
        var V = new Array(NR * NC).fill(0), k;
        for (k = 0; k < n; k += 1) V = sweep(V);
        return V;
      }
      function convergence() {
        var V = new Array(NR * NC).fill(0), k, res;
        for (k = 1; k <= 4000; k += 1) {
          var Vn = sweep(V);
          res = 0;
          for (var i = 0; i < Vn.length; i += 1) res = Math.max(res, Math.abs(Vn[i] - V[i]));
          V = Vn;
          if (res < 1e-6) break;
        }
        return { V: V, k: k };
      }
      function qvalues(V, r, c) {
        var qs = [];
        for (var i = 0; i < 4; i += 1) qs.push({ a: ACT[i], q: backup(V, r, c, ACT[i]) });
        qs.sort(function (x, y) { return y.q - x.q; });
        return qs;
      }

      K.hint(root, "Пять на пять клеток, четыре действия, боковой снос и дисконт. Ползунок «шагов итерации» — это буквально число применений оператора Беллмана к нулевой ценности: смотрите, как волна ненулевого значения расходится от цели ровно на клетку за шаг. Щелчок по свободной клетке переносит яму: стратегия перестраивается целиком, а не только рядом с ней.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Цвет — ценность V_k(s), стрелка — жадное действие по текущей V_k, режим «разрыв Q» показывает, насколько лучшее действие опережает второе. Тонкая стрелка означает почти ничью: там оценка переходов легко переворачивает выбор, почти не меняя возврата.");
      var cs = K.makeCanvas(stage, W, H, { label: "Сетка MDP, ценность и жадная стратегия", onResize: draw, drag: false });

      K.slider(controls, {
        label: "дисконт γ", min: 0.05, max: 0.99, step: 0.01, value: gamma,
        format: function (v) { return v.toFixed(2).replace(".", ","); }
      }, function (v) { gamma = v; draw(); });
      K.slider(controls, {
        label: "боковой снос", min: 0, max: 0.4, step: 0.01, value: slip,
        format: function (v) { return v.toFixed(2).replace(".", ","); }
      }, function (v) { slip = v; draw(); });
      K.slider(controls, {
        label: "шагов итерации k", min: 0, max: 60, step: 1, value: steps,
        format: function (v) { return String(Math.round(v)); }
      }, function (v) { steps = Math.round(v); draw(); });
      K.segmented(controls, {
        label: "что показывать", value: mode, options: [
          { value: "value", label: "ценность V_k" },
          { value: "policy", label: "стратегия" },
          { value: "gap", label: "разрыв Q" }
        ]
      }, function (v) { mode = v; draw(); });

      function cellRect(r, c) { return [OX + c * CELL, OY + r * CELL, CELL - 5, CELL - 5]; }

      function heat(v, lo, hi) {
        var t = hi - lo < 1e-9 ? 0 : (v - lo) / (hi - lo);
        t = Math.max(0, Math.min(1, t));
        var r = Math.round(245 + (56 - 245) * t);
        var g = Math.round(243 + (115 - 243) * t);
        var b = Math.round(234 + (93 - 234) * t);
        return "rgb(" + r + "," + g + "," + b + ")";
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.textBaseline = "alphabetic";
        var V = iterate(steps);
        var conv = convergence();
        var lo = 0, hi = 0.001, r, c, i;
        for (i = 0; i < V.length; i += 1) hi = Math.max(hi, V[i]);

        var gaps = [], minGap = Infinity, minCell = null;
        for (r = 0; r < NR; r += 1) {
          for (c = 0; c < NC; c += 1) {
            if (isWall(r, c) || isTerminal(r, c)) continue;
            var qs = qvalues(conv.V, r, c);
            var g = qs[0].q - qs[1].q;
            gaps.push(g);
            if (g < minGap) { minGap = g; minCell = [r, c]; }
          }
        }
        var maxGap = 0.001;
        for (i = 0; i < gaps.length; i += 1) maxGap = Math.max(maxGap, gaps[i]);

        for (r = 0; r < NR; r += 1) {
          for (c = 0; c < NC; c += 1) {
            var rect = cellRect(r, c);
            if (isWall(r, c)) {
              ctx.fillStyle = "#d5d6ce";
              ctx.fillRect(rect[0], rect[1], rect[2], rect[3]);
              continue;
            }
            if (isGoal(r, c)) ctx.fillStyle = C.green;
            else if (isPit(r, c)) ctx.fillStyle = C.red;
            else if (mode === "gap") ctx.fillStyle = heat(gapAt(gaps, r, c), 0, maxGap);
            else ctx.fillStyle = heat(V[idx(r, c)], lo, hi);
            ctx.fillRect(rect[0], rect[1], rect[2], rect[3]);
            ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
            ctx.strokeRect(rect[0] + 0.5, rect[1] + 0.5, rect[2], rect[3]);
          }
        }

        ctx.font = "13px PT Sans, sans-serif";
        ctx.textAlign = "center";
        for (r = 0; r < NR; r += 1) {
          for (c = 0; c < NC; c += 1) {
            if (isWall(r, c)) continue;
            var q = cellRect(r, c), cx = q[0] + q[2] / 2, cy = q[1] + q[3] / 2;
            if (isGoal(r, c) || isPit(r, c)) {
              ctx.fillStyle = C.paper;
              ctx.fillText(isGoal(r, c) ? "+10" : "−8", cx, cy + 5);
              continue;
            }
            var val = V[idx(r, c)];
            ctx.fillStyle = val > 0.55 * hi ? C.paper : C.ink;
            if (mode === "gap") {
              ctx.fillText(gapAt(gaps, r, c).toFixed(2).replace(".", ","), cx, cy + 26);
            } else {
              ctx.fillText(val.toFixed(2).replace(".", ","), cx, cy + 26);
            }
            if (mode !== "value") {
              var best = qvalues(V, r, c);
              var a = best[0].a;
              var strength = mode === "gap" ? Math.max(0.9, 1 + 5 * gapAt(gaps, r, c) / maxGap) : 2.4;
              arrow(ctx, cx, cy - 6, DC[a], DR[a], strength, val > 0.55 * hi ? C.paper : C.ink);
            }
            if (r === START[0] && c === START[1]) {
              ctx.strokeStyle = C.blue; ctx.lineWidth = 2.4;
              ctx.strokeRect(q[0] + 1, q[1] + 1, q[2] - 2, q[3] - 2);
            }
          }
        }

        ctx.textAlign = "left";
        ctx.fillStyle = C.muted;
        ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("синяя рамка — старт; серые клетки — стены; щелчок переносит яму", OX, OY + NR * CELL + 26);
        ctx.fillText("гарантия сжатия: ошибка не больше γ^k от начальной", OX, OY + NR * CELL + 46);
        ctx.fillText("k = " + steps + " применений оператора Беллмана", OX, OY - 18);

        var vs = conv.V[idx(START[0], START[1])];
        var pol = qvalues(conv.V, START[0], START[1])[0].a;
        var names = { U: "вверх", R: "вправо", D: "вниз", L: "влево" };
        output.set([
          { label: "V*(старт)", value: vs.toFixed(3).replace(".", ","), color: C.blue },
          { label: "V_k(старт) при текущем k", value: V[idx(START[0], START[1])].toFixed(3).replace(".", ","), color: C.gold },
          { label: "шагов до точности 10⁻⁶", value: String(conv.k), color: C.violet },
          { label: "лучшее действие в старте", value: names[pol], color: C.green },
          { label: "минимальный разрыв Q", value: minGap.toFixed(3).replace(".", ",") + (minCell ? " (строка " + (minCell[0] + 1) + ", столбец " + (minCell[1] + 1) + ")" : ""), color: C.red }
        ]);
      }

      function gapAt(gaps, r, c) {
        var n = 0;
        for (var rr = 0; rr < NR; rr += 1) {
          for (var cc = 0; cc < NC; cc += 1) {
            if (isWall(rr, cc) || isTerminal(rr, cc)) continue;
            if (rr === r && cc === c) return gaps[n];
            n += 1;
          }
        }
        return 0;
      }

      function arrow(ctx, x, y, dx, dy, lw, color) {
        var L = 17;
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(x - dx * L * 0.5, y - dy * L * 0.5);
        ctx.lineTo(x + dx * L * 0.5, y + dy * L * 0.5);
        ctx.stroke();
        var hx = x + dx * L * 0.5, hy = y + dy * L * 0.5;
        ctx.beginPath();
        ctx.moveTo(hx + dx * 6, hy + dy * 6);
        ctx.lineTo(hx - dy * 5 - dx * 2, hy + dx * 5 - dy * 2);
        ctx.lineTo(hx + dy * 5 - dx * 2, hy - dx * 5 - dy * 2);
        ctx.closePath();
        ctx.fill();
      }

      function onClick(ev) {
        var rect = cs.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        var my = (ev.clientY - rect.top) / rect.height * H;
        var c = Math.floor((mx - OX) / CELL), r = Math.floor((my - OY) / CELL);
        if (r < 0 || r >= NR || c < 0 || c >= NC) return;
        if (isWall(r, c) || isGoal(r, c)) return;
        if (r === START[0] && c === START[1]) return;
        pit = [r, c];
        draw();
      }
      cs.canvas.addEventListener("click", onClick);

      draw();
      return function () {
        cs.canvas.removeEventListener("click", onClick);
        cs.destroy();
      };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
