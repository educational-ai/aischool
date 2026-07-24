// Lesson 37: matmul-cost-lab — measure the cubic law for real, and see how parenthesization changes cost.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("matmul-cost-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 430;
      var mode = 0;              // 0 = benchmark, 1 = chain order
      var points = [];           // {n, ms} measured
      var running = false;
      var p2 = 5;                // shared inner dim in the chain
      var P0 = 40, P1 = 100, P3 = 60;

      K.hint(
        root,
        "Два взгляда на цену умножения матриц. «Замер» по-настоящему прогоняет наивное умножение в браузере и рисует измеренное время — наклон выходит около трёх. «Порядок скобок» показывает, что перестановка скобок меняет число операций в разы, хотя ответ один.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Замер: наивное умножение n×n делает примерно n³ операций, и измеренное время растёт как прямая на лог-лог осях с наклоном 3. Порядок скобок: цепочка A·B·C даёт один ответ, но (A·B)·C и A·(B·C) стоят разного числа умножений.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Цена умножения матриц", onResize: draw, drag: false });

      var runBtn = K.element("button", "kontur-int-segment", { type: "button", text: "Запустить замер" });
      runBtn.style.margin = "0 8px";
      runBtn.addEventListener("click", runBenchmark);

      K.segmented(controls, { label: "Что показать", value: 0, options: [
        { label: "закон n³ (замер)", value: 0 }, { label: "порядок скобок", value: 1 } ] }, function (v) {
        mode = v; syncControls(); draw();
      });
      controls.appendChild(runBtn);
      var p2Slider = K.slider(controls, { label: "средний размер (B: 100×k, C: k×60)", min: 5, max: 100, step: 5, value: p2, format: function (v) { return "k=" + v; } }, function (v) { p2 = v; draw(); });

      function syncControls() {
        runBtn.style.display = mode === 0 ? "" : "none";
        p2Slider.input.parentNode.style.display = mode === 1 ? "" : "none";
      }

      // ---- naive matmul in JS (real work) ----
      function naive(n, A, B) {
        var Cm = new Float64Array(n * n);
        for (var i = 0; i < n; i += 1) {
          for (var k = 0; k < n; k += 1) {
            var a = A[i * n + k];
            for (var j = 0; j < n; j += 1) Cm[i * n + j] += a * B[k * n + j];
          }
        }
        return Cm;
      }
      function timeSize(n) {
        var A = new Float64Array(n * n), B = new Float64Array(n * n);
        for (var t = 0; t < n * n; t += 1) { A[t] = (t * 2654435761 % 1000) / 1000 - 0.5; B[t] = (t * 40503 % 1000) / 1000 - 0.5; }
        naive(n, A, B); // warm
        // repeat enough times that total time clears timer resolution, then divide
        var best = Infinity;
        for (var trial = 0; trial < 3; trial += 1) {
          var reps = 1, el = 0;
          while (true) {
            var s = performance.now();
            for (var r = 0; r < reps; r += 1) naive(n, A, B);
            el = performance.now() - s;
            if (el >= 8 || reps >= 8192) break;
            reps *= 2;
          }
          best = Math.min(best, el / reps);
        }
        return best;
      }
      function runBenchmark() {
        if (running) return; running = true; points = [];
        var sizes = [16, 24, 32, 48, 64, 80, 96, 128];
        var idx = 0;
        runBtn.textContent = "Измеряю…";
        function step() {
          if (idx >= sizes.length) { running = false; runBtn.textContent = "Запустить замер"; draw(); return; }
          var n = sizes[idx];
          var ms = timeSize(n);
          points.push({ n: n, ms: ms });
          idx += 1; draw();
          setTimeout(step, 30);
        }
        setTimeout(step, 30);
      }

      function fitSlope() {
        if (points.length < 2) return null;
        var xs = points.map(function (p) { return Math.log(p.n); });
        var ys = points.map(function (p) { return Math.log(p.ms); });
        var mx = xs.reduce(function (a, b) { return a + b; }, 0) / xs.length;
        var my = ys.reduce(function (a, b) { return a + b; }, 0) / ys.length;
        var num = 0, den = 0;
        for (var i = 0; i < xs.length; i += 1) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) * (xs[i] - mx); }
        return num / den;
      }

      function drawBenchmark(ctx) {
        var L = 74, R = W - 40, T = 54, Bt = H - 56;
        // axes box
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(L, T, R - L, Bt - T);
        ctx.fillStyle = C.muted; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("размер n (лог-шкала)", (L + R) / 2, H - 20);
        ctx.save(); ctx.translate(24, (T + Bt) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("время, мс (лог-шкала)", 0, 0); ctx.restore();
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "15px PT Sans, sans-serif";
        ctx.fillText("Наивное умножение: измеряем время сами", L, T - 22);

        var nlo = Math.log(12), nhi = Math.log(160);
        function X(n) { return L + (Math.log(n) - nlo) / (nhi - nlo) * (R - L); }
        if (points.length === 0) {
          ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "14px PT Sans, sans-serif";
          ctx.fillText("Нажмите «Запустить замер» — умножение выполнится прямо в браузере.", (L + R) / 2, (T + Bt) / 2);
          return;
        }
        var mss = points.map(function (p) { return p.ms; });
        var ylo = Math.log(Math.min.apply(null, mss) * 0.6), yhi = Math.log(Math.max.apply(null, mss) * 1.6);
        function Y(ms) { return Bt - (Math.log(ms) - ylo) / (yhi - ylo) * (Bt - T); }
        // n^3 reference anchored at first point
        var p0 = points[0];
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
        ctx.beginPath();
        for (var gi = 0; gi <= 40; gi += 1) {
          var n = 12 * Math.pow(160 / 12, gi / 40);
          var ref = p0.ms * Math.pow(n / p0.n, 3);
          var yy = Y(Math.max(ref, Math.exp(ylo)));
          if (gi === 0) ctx.moveTo(X(n), yy); else ctx.lineTo(X(n), yy);
        }
        ctx.stroke(); ctx.setLineDash([]);
        // measured points
        ctx.fillStyle = C.blue;
        points.forEach(function (p) { ctx.beginPath(); ctx.arc(X(p.n), Y(p.ms), 5, 0, 7); ctx.fill(); });
        // legend
        ctx.font = "12px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillStyle = C.blue; ctx.fillText("● измеренное время", R - 190, T + 18);
        ctx.fillStyle = C.red; ctx.fillText("– – закон n³", R - 190, T + 36);

        var slope = fitSlope();
        output.set([
          { label: "Измерено размеров", value: String(points.length) + " из 8", color: C.ink },
          { label: "Последний размер", value: points[points.length - 1].n + "×" + points[points.length - 1].n + " за " + points[points.length - 1].ms.toFixed(2) + " мс", color: C.blue },
          { label: "Наклон на лог-лог (≈3 — куб)", value: slope ? slope.toFixed(2) : "—", color: slope && slope > 2.6 ? C.green : C.gold },
        ]);
      }

      function drawChain(ctx) {
        var abc = P0 * P1 * p2 + P0 * p2 * P3;   // (AB)C
        var a_bc = P1 * p2 * P3 + P0 * P1 * P3;  // A(BC)
        ctx.fillStyle = C.ink; ctx.font = "15px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("Цепочка A·B·C: один ответ, разная цена", 60, 40);
        // matrices, placed at fixed slots so labels never collide
        var sc = 0.42;
        function mat(cx, y, rows, cols, col, name) {
          var w = cols * sc, h = rows * sc;
          var x = cx - w / 2;
          ctx.fillStyle = col; ctx.globalAlpha = 0.22; ctx.fillRect(x, y - h, w, h); ctx.globalAlpha = 1;
          ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.strokeRect(x, y - h, w, h);
          ctx.fillStyle = col; ctx.font = "12px PT Sans, sans-serif"; ctx.textAlign = "center";
          ctx.fillText(name + " " + rows + "×" + cols, cx, 230);
        }
        var baseY = 190;
        mat(120, baseY, P0, P1, C.blue, "A");
        mat(240, baseY, P1, p2, C.green, "B");
        mat(340, baseY, p2, P3, C.gold, "C");
        ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("Ответ у обоих порядков одинаковый — цена разная.", 60, 270);
        // cost bars
        var bx = 470, bw = 210, by = 96, bh = 32, mx = Math.max(abc, a_bc);
        ctx.textAlign = "left";
        [["(A·B)·C", abc, C.green], ["A·(B·C)", a_bc, C.red]].forEach(function (row, i) {
          var y = by + i * 74;
          ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif"; ctx.fillText(row[0], bx, y - 7);
          ctx.fillStyle = row[2]; ctx.fillRect(bx, y, bw * row[1] / mx, bh);
          ctx.fillStyle = C.ink; ctx.font = "12px PT Sans, sans-serif";
          ctx.fillText(fmt(row[1]), bx + bw * row[1] / mx + 8, y + bh - 9);
        });
        var ratio = Math.max(abc, a_bc) / Math.min(abc, a_bc);
        ctx.fillStyle = C.muted; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("дешёвый порядок выгоднее в " + ratio.toFixed(1) + " раза", bx, by + 190);

        output.set([
          { label: "(A·B)·C", value: fmt(abc) + " умножений", color: abc <= a_bc ? C.green : C.ink },
          { label: "A·(B·C)", value: fmt(a_bc) + " умножений", color: a_bc < abc ? C.green : C.ink },
          { label: "Выгода порядка", value: "в " + ratio.toFixed(1) + " раза дешевле", color: C.blue },
        ]);
      }
      function fmt(v) { return String(v).replace(/\B(?=(\d{3})+(?!\d))/g, " "); }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.textBaseline = "alphabetic";
        if (mode === 0) drawBenchmark(ctx); else drawChain(ctx);
      }

      syncControls();
      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
