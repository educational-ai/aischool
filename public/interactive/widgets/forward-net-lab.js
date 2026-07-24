// Lesson 18: forward-pass lab — run a real iris flower through a trained 4-6-3 net.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("forward-net-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 540;

      // Trained on iris (seed 0), 98.7% acc. See generate_lesson18_visuals.py.
      var NET = {"mu":[5.843,3.054,3.759,1.199],"sd":[0.825,0.432,1.759,0.761],"W1":[[0.4,0.522,0.366,0.719,0.865,-0.518],[0.47,-0.648,0.383,-0.543,-0.355,0.992],[0.224,0.862,-1.963,0.422,0.849,-0.442],[0.011,0.37,-1.825,0.637,0.46,-0.696]],"b1":[-0.432,0.461,2.412,-0.009,0.819,-0.001],"W2":[[0.745,-0.639,0.324],[-1.167,1.27,1.305],[0.852,2.412,-3.442],[-2.02,-0.018,0.952],[-0.689,1.334,0.377],[1.237,-0.923,-1.7]],"b2":[0.118,0.124,-0.242],"examples":[[5.1,3.5,1.4,0.2],[5.9,3.2,4.8,1.8],[6.9,3.2,5.7,2.3]],"acc":0.987};
      var EX = [
        { name: "setosa (ясный)", x: NET.examples[0] },
        { name: "versicolor (пограничный)", x: NET.examples[1] },
        { name: "virginica (ясный)", x: NET.examples[2] },
      ];
      var FEATS = ["длина чашел.", "ширина чашел.", "длина леп.", "ширина леп."];
      var RANGE = [[4.3, 7.9], [2.0, 4.4], [1.0, 6.9], [0.1, 2.5]];
      var SPECIES = ["setosa", "versicolor", "virginica"];
      var SPCOL = [C.green, C.violet, C.red];

      var state = { x: EX[0].x.slice() };

      K.hint(
        root,
        "Двигайте четыре измерения цветка (или выбирайте пример) и следите, как сигнал течёт по сети: стандартизация входа, скрытый слой tanh, softmax-вероятности видов. Пограничный цветок даёт честную неуверенность.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Настоящая сеть 4→6→3, обученная на ирисах Фишера (точность 98,7%): вход стандартизуется, скрытый слой h=tanh(W₁x+b₁), выход softmax(W₂h+b₂). Веса зашиты; здесь только прогон (forward pass), без обучения.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Сигнал течёт по сети: вход, скрытый слой, вероятности видов",
        onResize: draw,
        drag: false,
      });

      function forward(xraw) {
        var xn = xraw.map(function (v, i) { return (v - NET.mu[i]) / NET.sd[i]; });
        var h = [];
        for (var j = 0; j < NET.b1.length; j += 1) {
          var s = NET.b1[j];
          for (var i = 0; i < 4; i += 1) s += xn[i] * NET.W1[i][j];
          h.push(Math.tanh(s));
        }
        var z = [];
        for (var k = 0; k < NET.b2.length; k += 1) {
          var t = NET.b2[k];
          for (var jj = 0; jj < h.length; jj += 1) t += h[jj] * NET.W2[jj][k];
          z.push(t);
        }
        var mx = Math.max.apply(null, z);
        var e = z.map(function (v) { return Math.exp(v - mx); });
        var sum = e.reduce(function (a, b) { return a + b; }, 0);
        var p = e.map(function (v) { return v / sum; });
        return { xn: xn, h: h, z: z, p: p };
      }

      var col = { inX: 90, inW: 150, hX: 420, hW: 150, outX: 780, outW: 180 };

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var r = forward(state.x);

        // input bars (standardized)
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("вход (станд.)", col.inX, 40);
        ctx.font = "12px PT Sans, sans-serif";
        var iy0 = 80, ih = 70;
        r.xn.forEach(function (v, i) {
          var y = iy0 + i * ih;
          var cx = col.inX + 60;
          var w = Math.max(-2.5, Math.min(2.5, v)) / 2.5 * 55;
          ctx.fillStyle = C.blue;
          ctx.globalAlpha = 0.85;
          ctx.fillRect(cx, y - 9, w, 18);
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText(FEATS[i], cx - 6, y);
          ctx.strokeStyle = C.line;
          ctx.beginPath(); ctx.moveTo(cx, y - 14); ctx.lineTo(cx, y + 14); ctx.stroke();
        });

        // hidden bars (tanh -1..1)
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("скрытый слой (tanh)", col.hX - 20, 40);
        ctx.font = "12px PT Sans, sans-serif";
        var hy0 = 70, hh = 62;
        r.h.forEach(function (v, j) {
          var y = hy0 + j * hh;
          var cx = col.hX + 55;
          ctx.strokeStyle = C.line;
          ctx.beginPath(); ctx.moveTo(cx, y - 12); ctx.lineTo(cx, y + 12); ctx.stroke();
          ctx.fillStyle = v >= 0 ? C.violet : C.gold;
          ctx.globalAlpha = 0.85;
          ctx.fillRect(cx, y - 8, v * 50, 16);
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText("h" + (j + 1), cx - 8, y);
        });

        // arrows between columns
        ctx.strokeStyle = C.line;
        drawFan(ctx, col.inX + 130, 220, col.hX + 30, 220);
        drawFan(ctx, col.hX + 120, 220, col.outX + 20, 260);

        // output probability bars
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("вероятности (softmax)", col.outX - 10, 40);
        ctx.font = "12px PT Sans, sans-serif";
        var oy0 = 130, oh = 90;
        var best = r.p.indexOf(Math.max.apply(null, r.p));
        r.p.forEach(function (v, k) {
          var y = oy0 + k * oh;
          var cx = col.outX + 10;
          ctx.fillStyle = SPCOL[k];
          ctx.globalAlpha = k === best ? 0.9 : 0.5;
          ctx.fillRect(cx, y - 14, v * 150, 28);
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.ink;
          ctx.textAlign = "left";
          ctx.fillText(SPECIES[k] + "  " + (v * 100).toFixed(1).replace(".", ",") + "%", cx, y - 26);
        });
        ctx.strokeStyle = C.line;
        ctx.beginPath(); ctx.moveTo(col.outX + 10, oy0 - 30); ctx.lineTo(col.outX + 10, oy0 + 2 * oh + 20); ctx.stroke();

        var conf = Math.max.apply(null, r.p);
        var rows = [
          { label: "Предсказание", value: SPECIES[best], color: SPCOL[best] },
          { label: "Уверенность", value: (conf * 100).toFixed(1).replace(".", ",") + " %", color: conf > 0.9 ? C.green : C.gold },
        ];
        if (conf < 0.7) rows.push({ label: "Итог", value: "пограничный цветок — сеть честно сомневается", color: C.gold });
        else rows.push({ label: "Итог", value: "ясный цветок — уверенное решение", color: C.green });
        output.set(rows);
      }

      function drawFan(ctx, x0, y0, x1, y1) {
        ctx.strokeStyle = C.line;
        ctx.globalAlpha = 0.5;
        for (var d = -60; d <= 60; d += 30) {
          ctx.beginPath();
          ctx.moveTo(x0, y0 + d);
          ctx.lineTo(x1, y1 + d);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // sliders for four features
      var handles = [];
      for (var i = 0; i < 4; i += 1) {
        (function (idx) {
          var hnd = K.slider(controls, { label: FEATS[idx] + ", см", min: RANGE[idx][0], max: RANGE[idx][1], step: 0.1, value: state.x[idx],
            format: function (v) { return v.toFixed(1).replace(".", ","); } },
            function (v) { state.x[idx] = v; draw(); });
          handles[idx] = hnd;
        })(i);
      }
      K.segmented(controls, { label: "Готовый пример",
        value: "0",
        options: EX.map(function (e, i) { return { value: String(i), label: e.name }; }) },
        function (v) {
          state.x = EX[Number(v)].x.slice();
          handles.forEach(function (hnd, i) { if (hnd) hnd.set(state.x[i]); });
          draw();
        });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
