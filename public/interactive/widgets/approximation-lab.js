// Lesson 19: universal approximation — build a target curve from sigmoid/ReLU blocks.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("approximation-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 520;

      var BIKE = {"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],"y":[36.8,16.6,8.7,4.9,5.4,24.9,102.5,290.6,477.0,241.5,135.4,158.2,200.8,198.4,183.6,201.3,293.1,525.3,492.2,348.4,249.7,186.3,138.4,88.7]}; // {x:[hours], y:[demand]}

      // Targets on x in [0,1].
      var TARGETS = {
        bike: {
          name: "спрос проката по часам",
          xs: BIKE.x.map(function (h) { return h / 23; }),
          ys: BIKE.y.slice(),
          xlabel: "час дня", xraw: BIKE.x, ylabel: "спрос",
        },
        sine: {
          name: "синусоида",
          xs: null, ys: null, fn: function (t) { return 0.5 + 0.4 * Math.sin(2 * Math.PI * t * 2); },
          xlabel: "x", ylabel: "y",
        },
        zigzag: {
          name: "зигзаг",
          xs: null, ys: null, fn: function (t) { return Math.abs((t * 4 % 2) - 1); },
          xlabel: "x", ylabel: "y",
        },
      };
      // materialize sine/zigzag on a grid
      Object.keys(TARGETS).forEach(function (key) {
        var T = TARGETS[key];
        if (!T.xs) {
          T.xs = []; T.ys = [];
          for (var i = 0; i <= 40; i += 1) {
            var t = i / 40; T.xs.push(t); T.ys.push(T.fn(t));
          }
          T.xraw = T.xs.map(function (t) { return Math.round(t * 100) / 100; });
        }
      });

      var state = { target: "bike", blocks: 4, kind: "sigmoid" };

      K.hint(
        root,
        "Крутите число блоков — сеть подгоняет столько сигмоид (или ReLU-изломов), сколько разрешено, и складывает из них приближение. Следите, как ошибка падает: больше блоков — точнее, как обещает теорема.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Блоки размещены равномерно по оси; веса выходного слоя подобраны прямым методом наименьших квадратов, без долгого обучения. Сигмоидные блоки дают гладкое приближение горбиками, ReLU — кусочно-линейное изломами. Целевая кривая проката — реальные данные из урока 08.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Целевая кривая и её приближение суммой блоков",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 30, w: 940, h: 430 };

      function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

      // least-squares fit: design matrix of K basis + intercept, solve normal equations.
      function fit(T, Kb, kind) {
        var xs = T.xs, ys = T.ys, n = xs.length;
        var centers = [];
        for (var c = 0; c < Kb; c += 1) centers.push(Kb === 1 ? 0.5 : c / (Kb - 1));
        var cols = Kb + (kind === "relu" ? 2 : 1);
        // build Phi (n x cols)
        var Phi = [];
        for (var i = 0; i < n; i += 1) {
          var row = [];
          for (var j = 0; j < Kb; j += 1) {
            if (kind === "sigmoid") row.push(sigmoid(12 * (xs[i] - centers[j])));
            else row.push(Math.max(0, xs[i] - centers[j]));
          }
          row.push(1);
          if (kind === "relu") row.push(xs[i]);
          Phi.push(row);
        }
        // normal equations A = Phi^T Phi (cols x cols), rhs = Phi^T y
        var A = [], rhs = [];
        for (var a = 0; a < cols; a += 1) {
          A.push(new Array(cols).fill(0)); rhs.push(0);
        }
        for (var i2 = 0; i2 < n; i2 += 1) {
          for (var a2 = 0; a2 < cols; a2 += 1) {
            rhs[a2] += Phi[i2][a2] * ys[i2];
            for (var b = 0; b < cols; b += 1) A[a2][b] += Phi[i2][a2] * Phi[i2][b];
          }
        }
        // ridge for stability
        for (var d = 0; d < cols; d += 1) A[d][d] += 1e-7;
        var v = solve(A, rhs);
        // predict on dense grid
        var dense = [];
        for (var g = 0; g <= 200; g += 1) {
          var t = g / 200, val = 0;
          for (var j2 = 0; j2 < Kb; j2 += 1) {
            var bb = kind === "sigmoid" ? sigmoid(12 * (t - centers[j2])) : Math.max(0, t - centers[j2]);
            val += v[j2] * bb;
          }
          val += v[Kb];
          if (kind === "relu") val += v[Kb + 1] * t;
          dense.push([t, val]);
        }
        // rmse on data
        var se = 0;
        for (var i3 = 0; i3 < n; i3 += 1) {
          var pv = 0;
          for (var j3 = 0; j3 < Kb; j3 += 1) {
            var bv = kind === "sigmoid" ? sigmoid(12 * (xs[i3] - centers[j3])) : Math.max(0, xs[i3] - centers[j3]);
            pv += v[j3] * bv;
          }
          pv += v[Kb]; if (kind === "relu") pv += v[Kb + 1] * xs[i3];
          se += (pv - ys[i3]) * (pv - ys[i3]);
        }
        return { dense: dense, rmse: Math.sqrt(se / n) };
      }

      function solve(A, b) {
        var n = A.length;
        var M = A.map(function (r, i) { return r.concat([b[i]]); });
        for (var col = 0; col < n; col += 1) {
          var piv = col;
          for (var r = col + 1; r < n; r += 1) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
          var tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
          var d = M[col][col] || 1e-9;
          for (var c2 = col; c2 <= n; c2 += 1) M[col][c2] /= d;
          for (var r2 = 0; r2 < n; r2 += 1) {
            if (r2 === col) continue;
            var f = M[r2][col];
            for (var c3 = col; c3 <= n; c3 += 1) M[r2][c3] -= f * M[col][c3];
          }
        }
        return M.map(function (r) { return r[n]; });
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var T = TARGETS[state.target];
        var ymin = Math.min.apply(null, T.ys), ymax = Math.max.apply(null, T.ys);
        var pad = (ymax - ymin) * 0.12 + 1e-6;
        ymin -= pad; ymax += pad;
        function px(t) { return box.x + t * box.w; }
        function py(v) { return box.y + box.h - (v - ymin) / (ymax - ymin) * box.h; }

        ctx.strokeStyle = C.line;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        // x ticks
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var t = 0; t <= 1.0001; t += 0.25) {
          var lab = state.target === "bike" ? Math.round(t * 23) : t.toFixed(2).replace(".", ",");
          ctx.fillText(String(lab), px(t), box.y + box.h + 16);
        }
        ctx.fillText(T.xlabel, box.x + box.w / 2, box.y + box.h + 34);

        var res = fit(T, state.blocks, state.kind);
        // target points
        ctx.fillStyle = C.faint || C.muted;
        T.xs.forEach(function (xx, i) {
          ctx.beginPath(); ctx.arc(px(xx), py(T.ys[i]), 3.4, 0, Math.PI * 2); ctx.fill();
        });
        // approximation (clip first, then draw the curve)
        ctx.save();
        ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.6;
        ctx.beginPath();
        res.dense.forEach(function (p, i) {
          if (i === 0) ctx.moveTo(px(p[0]), py(p[1]));
          else ctx.lineTo(px(p[0]), py(p[1]));
        });
        ctx.stroke();
        ctx.restore();
        ctx.lineWidth = 1;
        ctx.fillStyle = C.blue; ctx.textAlign = "left";
        ctx.fillText("серые точки — цель, синяя — сумма " + state.blocks + " блоков", box.x + 10, box.y + 16);

        var unit = state.target === "bike" ? " поездок" : "";
        var rmse = state.target === "bike" ? res.rmse.toFixed(0) : res.rmse.toFixed(3).replace(".", ",");
        var rows = [
          { label: "Блоков", value: String(state.blocks) + " (" + (state.kind === "sigmoid" ? "сигмоиды" : "ReLU") + ")", color: C.ink },
          { label: "Ошибка приближения", value: rmse + unit, color: res.rmse < (state.target === "bike" ? 30 : 0.03) ? C.green : C.ink },
        ];
        var good = state.target === "bike" ? res.rmse < 30 : res.rmse < 0.03;
        if (good) rows.push({ label: "Итог", value: "сумма блоков уверенно повторяет кривую", color: C.green });
        else rows.push({ label: "Подсказка", value: "добавьте блоков — приближение станет точнее", color: C.muted });
        output.set(rows);
      }

      K.slider(controls, { label: "Число блоков", min: 1, max: 24, step: 1, value: state.blocks },
        function (v) { state.blocks = v; draw(); });
      K.segmented(controls, { label: "Тип блока", value: state.kind,
        options: [{ value: "sigmoid", label: "сигмоида (горбики)" }, { value: "relu", label: "ReLU (изломы)" }] },
        function (v) { state.kind = v; draw(); });
      K.segmented(controls, { label: "Целевая кривая", value: state.target,
        options: [
          { value: "bike", label: "спрос проката" },
          { value: "sine", label: "синусоида" },
          { value: "zigzag", label: "зигзаг" },
        ] }, function (v) { state.target = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
