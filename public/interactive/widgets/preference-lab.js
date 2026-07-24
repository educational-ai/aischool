// Lesson 80: preference-lab — попарные голоса, модель Брэдли–Терри и её пределы.
// Голоса порождаются заданной «истиной»: одномерная шкала, смесь двух вкусов или цикл.
// Модель всегда одна и та же — одномерная. С ростом числа голосов интервалы сужаются
// ВСЕГДА, а вот остаток «факт минус модель» уходит к нулю только в первом режиме.
// Активный выбор пары ускоряет сужение интервалов и никак не лечит неверную модель.
(function () {
  "use strict";

  var NAMES = ["A", "B", "C", "D", "E", "F"];
  var N = 6;

  // «Истины»: матрица p[i][j] = вероятность, что i победит j
  function truthMatrix(kind) {
    var r1 = [1.3, 0.8, 0.25, -0.35, -0.9, -1.1];
    var r2 = [-1.1, 0.35, 1.2, 0.7, -0.25, -0.9];
    var P = [], i, j;
    for (i = 0; i < N; i += 1) { P.push([]); for (j = 0; j < N; j += 1) P[i].push(0.5); }
    for (i = 0; i < N; i += 1) {
      for (j = 0; j < N; j += 1) {
        if (i === j) continue;
        if (kind === 0) P[i][j] = sig(r1[i] - r1[j]);
        else if (kind === 1) P[i][j] = 0.5 * sig(r1[i] - r1[j]) + 0.5 * sig(r2[i] - r2[j]);
        else P[i][j] = sig(r1[i] - r1[j]);
      }
    }
    if (kind === 2) {                       // цикл A > B > C > A на трёх первых
      P[0][1] = 0.70; P[1][0] = 0.30;
      P[1][2] = 0.70; P[2][1] = 0.30;
      P[2][0] = 0.70; P[0][2] = 0.30;
    }
    var groups = [];
    if (kind === 1) {
      groups.push(matFromScores(r1));
      groups.push(matFromScores(r2));
    } else {
      groups.push(P);
    }
    return { P: P, r: kind === 1 ? null : r1, groups: groups };
  }

  function matFromScores(r) {
    var M = [], i, j;
    for (i = 0; i < N; i += 1) {
      M.push([]);
      for (j = 0; j < N; j += 1) M[i].push(i === j ? 0.5 : sig(r[i] - r[j]));
    }
    return M;
  }

  function sig(z) { return 1 / (1 + Math.exp(-z)); }

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("preference-lab", function (root, options, K) {
      var C = K.COLORS;
      var INK = C.ink || "#171915";
      var MUT = C.muted || "#6e726a";
      var GRID = C.grid || "#deddd4";
      var PAPER = "#fffef9";
      var W = 900, H = 470;

      var truth = 0;        // 0 — одна шкала, 1 — два вкуса, 2 — цикл
      var votes = 300;      // сколько голосов собрано
      var active = 0;       // 0 — случайные пары, 1 — активный выбор
      var state = null;

      function rndGen(seed) {
        var s = seed >>> 0;
        return function () {
          s = (s * 1664525 + 1013904223) >>> 0;
          return s / 4294967296;
        };
      }

      // ---- модель Брэдли–Терри: MM-итерации Цермело
      function fit(Wm) {
        var p = [], i, j, it;
        for (i = 0; i < N; i += 1) p.push(1);
        var wins = [], nij = [];
        for (i = 0; i < N; i += 1) {
          var s = 0; nij.push([]);
          for (j = 0; j < N; j += 1) { s += Wm[i][j]; nij[i].push(Wm[i][j] + Wm[j][i]); }
          wins.push(s);
        }
        for (it = 0; it < 300; it += 1) {
          var np = [], logsum = 0, live = 0;
          for (i = 0; i < N; i += 1) {
            var den = 0;
            for (j = 0; j < N; j += 1) if (j !== i) den += nij[i][j] / (p[i] + p[j]);
            np.push(den > 0 && wins[i] > 0 ? wins[i] / den : p[i]);
          }
          for (i = 0; i < N; i += 1) logsum += Math.log(Math.max(np[i], 1e-9));
          var g = Math.exp(logsum / N);
          for (i = 0; i < N; i += 1) { np[i] /= g; live += Math.abs(np[i] - p[i]); }
          p = np;
          if (live < 1e-11) break;
        }
        var r = [], mean = 0;
        for (i = 0; i < N; i += 1) { r.push(Math.log(Math.max(p[i], 1e-9))); mean += r[i]; }
        mean /= N;
        for (i = 0; i < N; i += 1) r[i] -= mean;
        return r;
      }

      // приближённые стандартные ошибки по информации Фишера
      function stderr(Wm, r) {
        var se = [], i, j;
        for (i = 0; i < N; i += 1) {
          var info = 0;
          for (j = 0; j < N; j += 1) {
            if (i === j) continue;
            var n = Wm[i][j] + Wm[j][i];
            var q = sig(r[i] - r[j]);
            info += n * q * (1 - q);
          }
          se.push(info > 1e-9 ? 1 / Math.sqrt(info) : 3.0);
        }
        return se;
      }

      function empty() {
        var M = [], i, j;
        for (i = 0; i < N; i += 1) { M.push([]); for (j = 0; j < N; j += 1) M[i].push(0); }
        return M;
      }

      // ---- сбор голосов: детерминированно от нуля до текущего бюджета
      function simulate() {
        var T = truthMatrix(truth);
        var rnd = rndGen(80808081);
        var Wm = empty(), r = [], se = [], i, j, k;
        for (i = 0; i < N; i += 1) r.push(0);
        var pairs = [];
        for (i = 0; i < N; i += 1) for (j = i + 1; j < N; j += 1) pairs.push([i, j]);
        var block = 10;
        var done = 0;
        while (done < votes) {
          var pick;
          if (active === 1 && done >= 30) {
            se = stderr(Wm, r);
            var best = -1, bi = 0;
            for (k = 0; k < pairs.length; k += 1) {
              var a = pairs[k][0], b = pairs[k][1];
              var q = sig(r[a] - r[b]);
              var score = (se[a] + se[b]) * q * (1 - q);
              if (score > best) { best = score; bi = k; }
            }
            pick = pairs[bi];
          } else {
            pick = pairs[Math.floor(rnd() * pairs.length) % pairs.length];
          }
          var m = Math.min(block, votes - done);
          for (k = 0; k < m; k += 1) {
            if (rnd() < T.P[pick[0]][pick[1]]) Wm[pick[0]][pick[1]] += 1;
            else Wm[pick[1]][pick[0]] += 1;
          }
          done += m;
          if (active === 1) r = fit(Wm);
        }
        r = fit(Wm);
        se = stderr(Wm, r);

        // остаток: эмпирическая доля побед против предсказанной
        var maxRes = 0, cover = 0, pairsSeen = 0, ll = 0, nll = 0;
        for (i = 0; i < N; i += 1) {
          for (j = i + 1; j < N; j += 1) {
            var n = Wm[i][j] + Wm[j][i];
            if (n >= 12) {
              var emp = Wm[i][j] / n;
              var fitp = sig(r[i] - r[j]);
              maxRes = Math.max(maxRes, Math.abs(emp - fitp));
              pairsSeen += 1;
            }
            if (n > 0) cover += 1;
            // насколько модель промахивается мимо истины отдельной группы
            var pf = sig(r[i] - r[j]), gk;
            for (gk = 0; gk < T.groups.length; gk += 1) {
              ll = Math.max(ll, Math.abs(pf - T.groups[gk][i][j]));
            }
            nll += 1;
          }
        }
        var width = 0;
        for (i = 0; i < N; i += 1) width += 2 * 1.96 * se[i];
        width /= N;
        return {
          Wm: Wm, r: r, se: se, T: T, maxRes: maxRes, seen: pairsSeen,
          cover: cover, width: width, ll: ll
        };
      }

      function recompute() { state = simulate(); }

      K.hint(root, "Голоса порождаются выбранной «истиной», а модель всегда одна: один скрытый балл на вариант. Увеличивайте число голосов — интервалы сужаются в любом режиме, это всего лишь статистика. Но остаток «доля побед минус предсказание модели» падает к нулю только тогда, когда истина действительно одномерна. В режимах «два вкуса» и «цикл» модель становится всё увереннее, оставаясь неверной, — именно так reward model уверенно повторяет то, чего в данных нет.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Слева — граф сравнений: толщина ребра равна числу голосов, золотой цвет отмечает пары, где голоса почти поровну. Справа — оценённые баллы с 95% интервалом (информация Фишера) и чёрными штрихами истинные баллы там, где они существуют. Остаток считается по парам, где набралось хотя бы 12 голосов.");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Граф попарных сравнений, оценённые баллы и остатки модели",
        onResize: draw,
        drag: false
      });

      function nodePos(i) {
        var cx = 230, cy = 245, R = 150;
        var a = -Math.PI / 2 + (2 * Math.PI * i) / N;
        return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
      }

      function draw() {
        var ctx = cs.ctx, i, j;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var S = state;

        // ---------- левая панель: граф
        ctx.fillStyle = MUT; ctx.textAlign = "center";
        ctx.fillText("граф сравнений: " + S.cover + " пар из 15 получили голоса", 230, 40);
        var maxN = 1;
        for (i = 0; i < N; i += 1) for (j = 0; j < N; j += 1) maxN = Math.max(maxN, S.Wm[i][j] + S.Wm[j][i]);
        for (i = 0; i < N; i += 1) {
          for (j = i + 1; j < N; j += 1) {
            var n = S.Wm[i][j] + S.Wm[j][i];
            if (n === 0) continue;
            var a = nodePos(i), b = nodePos(j);
            var frac = S.Wm[i][j] / n;
            ctx.strokeStyle = Math.abs(frac - 0.5) < 0.08 ? (C.gold || "#a57920") : (C.blue || "#315f8c");
            ctx.globalAlpha = 0.28 + 0.6 * (n / maxN);
            ctx.lineWidth = 1 + 6 * (n / maxN);
            ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
            ctx.globalAlpha = 1;
          }
        }
        for (i = 0; i < N; i += 1) {
          var p = nodePos(i);
          ctx.fillStyle = C.blue || "#315f8c";
          ctx.beginPath(); ctx.arc(p[0], p[1], 17, 0, 7); ctx.fill();
          ctx.strokeStyle = PAPER; ctx.lineWidth = 2; ctx.stroke();
          ctx.fillStyle = PAPER; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "center";
          ctx.fillText(NAMES[i], p[0], p[1] + 5);
        }
        ctx.font = "12px PT Sans, sans-serif";

        // ---------- правая панель: баллы с интервалами
        var X0 = 520, X1 = 860, mid = (X0 + X1) / 2, scale = (X1 - X0) / 2 / 2.4;
        ctx.strokeStyle = GRID; ctx.lineWidth = 1;
        for (var t = -2; t <= 2; t += 1) {
          var x = mid + t * scale;
          ctx.beginPath(); ctx.moveTo(x, 70); ctx.lineTo(x, 415); ctx.stroke();
          ctx.fillStyle = MUT; ctx.textAlign = "center";
          ctx.fillText(String(t), x, 435);
        }
        ctx.fillStyle = MUT; ctx.textAlign = "center";
        ctx.fillText("оценённый балл и 95% интервал", mid, 40);
        ctx.fillText("скрытый балл", mid, 455);
        for (i = 0; i < N; i += 1) {
          var y = 90 + i * 55;
          var xc = mid + S.r[i] * scale;
          var half = 1.96 * S.se[i] * scale;
          ctx.strokeStyle = C.blue || "#315f8c"; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(xc - half, y); ctx.lineTo(xc + half, y); ctx.stroke();
          ctx.fillStyle = C.blue || "#315f8c";
          ctx.beginPath(); ctx.arc(xc, y, 6, 0, 7); ctx.fill();
          if (S.T.r) {
            var xt = mid + S.T.r[i] * scale;
            ctx.strokeStyle = INK; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(xt, y - 12); ctx.lineTo(xt, y + 12); ctx.stroke();
          }
          ctx.fillStyle = INK; ctx.textAlign = "right";
          ctx.fillText(NAMES[i], X0 - 14, y + 5);
        }
        if (!S.T.r) {
          ctx.fillStyle = MUT; ctx.textAlign = "center";
          ctx.fillText("истинного одного балла здесь не существует", mid, 70);
        }

        var resColor = S.maxRes > 0.08 ? (C.red || "#b94a3b") : (C.green || "#38735d");
        out.set([
          { label: "Голосов собрано", value: String(votes), color: C.blue },
          { label: "Средняя ширина 95% интервала", value: S.width.toFixed(2), color: C.gold },
          { label: "Макс. остаток |факт − модель|", value: S.maxRes.toFixed(3), color: resColor },
          { label: "Макс. промах мимо истины группы", value: S.ll.toFixed(3), color: S.ll > 0.08 ? (C.red || "#b94a3b") : (C.green || "#38735d") }
        ]);
      }

      K.slider(controls, {
        label: "Число голосов",
        min: 30, max: 3000, step: 30, value: votes,
        format: function (v) { return String(v); }
      }, function (v) { votes = v; recompute(); draw(); });

      K.segmented(controls, {
        label: "Что на самом деле думают люди",
        value: 0,
        options: [
          { label: "одна шкала", value: 0 },
          { label: "два вкуса", value: 1 },
          { label: "цикл A≻B≻C≻A", value: 2 }
        ]
      }, function (v) { truth = v; recompute(); draw(); });

      K.segmented(controls, {
        label: "Как выбираем следующую пару",
        value: 0,
        options: [
          { label: "случайно", value: 0 },
          { label: "активно", value: 1 }
        ]
      }, function (v) { active = v; recompute(); draw(); });

      recompute();
      draw();

      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
