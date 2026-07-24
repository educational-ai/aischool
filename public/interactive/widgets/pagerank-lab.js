// Lesson 65: pagerank-lab — a random reader walks a small graph while the power method
// computes the exact stationary vector. Change alpha, the graph and the teleport vector
// and watch empirical frequencies chase (or fail to chase) the exact PageRank.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("pagerank-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var PAPER = "#fffef9";

      var GRAPHS = {
        cycle: {
          title: "цикл: A→B, A→C, B→C, C→A",
          labels: ["A", "B", "C"],
          edges: [[0, 1], [0, 2], [1, 2], [2, 0]],
          pos: [[0.50, 0.16], [0.80, 0.74], [0.20, 0.74]]
        },
        trap: {
          title: "ловушка: из D, E, F выхода нет",
          labels: ["A", "B", "C", "D", "E", "F"],
          edges: [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3]],
          pos: [[0.16, 0.20], [0.08, 0.72], [0.40, 0.78], [0.72, 0.20], [0.92, 0.66], [0.60, 0.62]]
        },
        dead: {
          title: "тупик: у E нет исходящих ссылок",
          labels: ["A", "B", "C", "D", "E"],
          edges: [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4]],
          pos: [[0.14, 0.24], [0.10, 0.76], [0.42, 0.52], [0.70, 0.24], [0.88, 0.74]]
        }
      };

      var state = {
        graph: "trap",
        alpha: 0.85,
        teleport: "uniform",
        walker: 0,
        steps: 0,
        visits: null,
        timer: 0
      };

      function g() { return GRAPHS[state.graph]; }

      function outLinks(n) {
        var gr = g(), out = [], i;
        for (i = 0; i < gr.labels.length; i += 1) out.push([]);
        for (i = 0; i < gr.edges.length; i += 1) out[gr.edges[i][0]].push(gr.edges[i][1]);
        return out;
      }

      function teleVector() {
        var n = g().labels.length, v = [], i;
        if (state.teleport === "uniform") {
          for (i = 0; i < n; i += 1) v.push(1 / n);
        } else {
          for (i = 0; i < n; i += 1) v.push(i === 0 ? 0.7 : 0.3 / (n - 1));
        }
        return v;
      }

      // exact PageRank by the power method; also returns the number of iterations
      function exact() {
        var gr = g(), n = gr.labels.length, out = outLinks(n), v = teleVector();
        var p = v.slice(), i, j, t, d, q, mass;
        var iters = 0;
        for (t = 1; t <= 4000; t += 1) {
          q = [];
          for (i = 0; i < n; i += 1) q.push((1 - state.alpha) * v[i]);
          for (i = 0; i < n; i += 1) {
            if (out[i].length === 0) {
              for (j = 0; j < n; j += 1) q[j] += state.alpha * p[i] * v[j];  // dangling repair
            } else {
              for (j = 0; j < out[i].length; j += 1) q[out[i][j]] += state.alpha * p[i] / out[i].length;
            }
          }
          mass = 0;
          for (i = 0; i < n; i += 1) mass += q[i];
          for (i = 0; i < n; i += 1) q[i] /= mass;
          d = 0;
          for (i = 0; i < n; i += 1) d += Math.abs(q[i] - p[i]);
          p = q;
          iters = t;
          if (d < 1e-12) break;
        }
        return { pi: p, iters: iters };
      }

      function resetWalk() {
        var n = g().labels.length, i;
        state.visits = [];
        for (i = 0; i < n; i += 1) state.visits.push(0);
        state.walker = 0;
        state.steps = 0;
      }

      function pick(v) {
        var r = Math.random(), acc = 0, i;
        for (i = 0; i < v.length; i += 1) { acc += v[i]; if (r <= acc) return i; }
        return v.length - 1;
      }

      function walk(k) {
        var out = outLinks(), v = teleVector(), i, links;
        for (i = 0; i < k; i += 1) {
          links = out[state.walker];
          if (Math.random() > state.alpha || links.length === 0) {
            state.walker = pick(v);
          } else {
            state.walker = links[Math.floor(Math.random() * links.length)];
          }
          state.visits[state.walker] += 1;
          state.steps += 1;
        }
      }

      K.hint(root, "Слева случайный читатель ходит по графу: с вероятностью α он идёт по случайной исходящей ссылке, иначе телепортируется по вектору v (из тупика — всегда телепортация). Справа серые столбцы — доля времени, проведённого читателем в вершине, красные точки — точный вектор π, вычисленный степенным методом. Уменьшите α до нуля — ранг станет равен v; поднимите до 1 на графе с ловушкой — и весь поток утечёт в D, E, F.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var buttons = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева: размер и цвет вершины пропорциональны π. Справа: эмпирическая частота против точного PageRank. Совпадение столбцов и точек — это и есть частотный смысл стационарного распределения: π — доля времени, а не «оценка качества».");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Граф, случайный читатель и стационарное распределение",
        onResize: draw,
        drag: false
      });

      function nodeXY(i) {
        var p = g().pos[i];
        return [40 + p[0] * 400, 40 + p[1] * 370];
      }

      function arrow(ctx, a, b, color, lw) {
        var A = nodeXY(a), B = nodeXY(b);
        var dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy) || 1;
        var ux = dx / L, uy = dy / L, r = 22;
        var x0 = A[0] + ux * r, y0 = A[1] + uy * r, x1 = B[0] - ux * r, y1 = B[1] - uy * r;
        ctx.strokeStyle = color; ctx.lineWidth = lw;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        var ah = 9;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - ah * (ux * 0.92 + uy * 0.5), y1 - ah * (uy * 0.92 - ux * 0.5));
        ctx.lineTo(x1 - ah * (ux * 0.92 - uy * 0.5), y1 - ah * (uy * 0.92 + ux * 0.5));
        ctx.closePath();
        ctx.fillStyle = color; ctx.fill();
      }

      function draw() {
        var ctx = cs.ctx, gr = g(), n = gr.labels.length, i;
        var res = exact(), pi = res.pi;
        var out = outLinks();
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        // ---- left: the graph
        for (i = 0; i < gr.edges.length; i += 1) arrow(ctx, gr.edges[i][0], gr.edges[i][1], "#c9c8be", 2);
        for (i = 0; i < n; i += 1) {
          var xy = nodeXY(i);
          var dangling = out[i].length === 0;
          var r = 13 + 26 * Math.min(1, pi[i] * n / 2.6);
          ctx.beginPath(); ctx.arc(xy[0], xy[1], r, 0, 7);
          ctx.fillStyle = dangling ? C.gold : (i === state.walker ? C.red : C.blue);
          ctx.fill();
          ctx.strokeStyle = PAPER; ctx.lineWidth = 2; ctx.stroke();
          ctx.fillStyle = PAPER; ctx.textAlign = "center"; ctx.font = "13px PT Sans, sans-serif";
          ctx.fillText(gr.labels[i], xy[0], xy[1] + 5);
          ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(pi[i].toFixed(3), xy[0], xy[1] + r + 14);
        }
        ctx.fillStyle = C.red; ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("красная вершина — где читатель сейчас", 30, H - 14);
        ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText(gr.title, 30, 24);

        // ---- right: bars
        var X0 = 520, X1 = W - 30, Y0 = 60, Y1 = H - 60;
        var maxv = 0.05, emp = [];
        for (i = 0; i < n; i += 1) {
          emp.push(state.steps > 0 ? state.visits[i] / state.steps : 0);
          maxv = Math.max(maxv, pi[i], emp[i]);
        }
        maxv = Math.min(1, maxv * 1.2);
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X0, Y1); ctx.lineTo(X1, Y1); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        for (var t = 0; t <= 4; t += 1) {
          var val = maxv * t / 4, y = Y1 - (Y1 - Y0) * t / 4;
          ctx.fillText(val.toFixed(2), X0 - 8, y + 4);
          ctx.strokeStyle = C.grid;
          ctx.beginPath(); ctx.moveTo(X0, y); ctx.lineTo(X1, y); ctx.stroke();
        }
        var bw = (X1 - X0) / n;
        for (i = 0; i < n; i += 1) {
          var cx = X0 + bw * (i + 0.5);
          var he = (Y1 - Y0) * Math.min(1, emp[i] / maxv);
          ctx.fillStyle = "rgba(110,114,106,0.45)";
          ctx.fillRect(cx - bw * 0.28, Y1 - he, bw * 0.56, he);
          var yp = Y1 - (Y1 - Y0) * Math.min(1, pi[i] / maxv);
          ctx.fillStyle = C.red;
          ctx.beginPath(); ctx.arc(cx, yp, 6, 0, 7); ctx.fill();
          ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
          ctx.fillText(gr.labels[i], cx, Y1 + 18);
        }
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("частота читателя (серое) и точный π (красное)", X0, 34);

        // ---- readout
        var best = 0, gap = 0;
        for (i = 0; i < n; i += 1) if (pi[i] > pi[best]) best = i;
        for (i = 0; i < n; i += 1) gap += Math.abs(emp[i] - pi[i]);
        var extra = [];
        if (state.graph === "trap") {
          var tm = pi[3] + pi[4] + pi[5];
          extra.push({ label: "масса в ловушке D+E+F", value: tm.toFixed(3), color: C.gold });
        }
        output.set([
          { label: "лидер", value: gr.labels[best] + " (π = " + pi[best].toFixed(3) + ")", color: C.blue },
          { label: "шагов читателя", value: String(state.steps), color: C.muted },
          { label: "расхождение частот и π (L₁)", value: state.steps ? gap.toFixed(3) : "—", color: C.red },
          { label: "итераций степенного метода", value: String(res.iters), color: C.green }
        ].concat(extra));
      }

      K.segmented(controls, {
        label: "граф",
        value: state.graph,
        options: [
          { value: "cycle", label: "цикл" },
          { value: "trap", label: "ловушка" },
          { value: "dead", label: "тупик" }
        ]
      }, function (v) { state.graph = v; resetWalk(); draw(); });

      K.slider(controls, {
        label: "α — вероятность пойти по ссылке",
        min: 0, max: 0.99, step: 0.01, value: state.alpha,
        format: function (x) { return x.toFixed(2); }
      }, function (v) { state.alpha = v; resetWalk(); draw(); });

      K.segmented(controls, {
        label: "телепортация v",
        value: state.teleport,
        options: [
          { value: "uniform", label: "равномерная" },
          { value: "toA", label: "70% в A" }
        ]
      }, function (v) { state.teleport = v; resetWalk(); draw(); });

      var bWalk = K.element("button", "kontur-int-segment", { type: "button", text: "+2000 шагов" });
      var bRun = K.element("button", "kontur-int-segment", { type: "button", text: "гулять" });
      var bReset = K.element("button", "kontur-int-segment", { type: "button", text: "сброс прогулки" });
      bWalk.style.margin = bRun.style.margin = bReset.style.margin = "0 6px";
      bWalk.addEventListener("click", function () { walk(2000); draw(); });
      bReset.addEventListener("click", function () { resetWalk(); draw(); });
      bRun.addEventListener("click", function () {
        if (state.timer) {
          clearInterval(state.timer); state.timer = 0; bRun.textContent = "гулять";
        } else {
          state.timer = setInterval(function () { walk(40); draw(); }, 60);
          bRun.textContent = "стоп";
        }
      });
      buttons.appendChild(bRun); buttons.appendChild(bWalk); buttons.appendChild(bReset);

      resetWalk();
      draw();

      return function () {
        if (state.timer) clearInterval(state.timer);
        cs.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
