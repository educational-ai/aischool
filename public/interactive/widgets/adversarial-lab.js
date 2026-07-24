// Lesson 26: adversarial lab — FGSM attack flips a real iris classifier.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("adversarial-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 470;

      var D = {"W":[[-1.547,1.5781,-2.6204,-2.4593],[0.9446,-0.4562,-0.3719,-1.1919],[0.6024,-1.1219,2.9924,3.6512]],"b":[-0.5318,2.8094,-2.2776],"sd":[0.825,0.432,1.759,0.761],"classes":["setosa","versicolor","virginica"],"examples":[{"x":[-1.628,-1.745,-1.398,-1.182],"t":0},{"x":[0.553,-1.282,0.649,0.396],"t":1},{"x":[0.311,-1.051,1.047,0.265],"t":2}],"featc":["длина чаш.","ширина чаш.","длина леп.","ширина леп."]};

      var state = { pick: 1, eps: 0 };

      K.hint(
        root,
        "Настоящий классификатор ирисов, обученный до 97 % точности. Двигайте бюджет искажения ε: вход сдвигается в сторону, сильнее всего запутывающую модель (метод FGSM). Смотрите, при каком крошечном искажении уверенный ответ переворачивается на неверный.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Атака FGSM сдвигает вход по знаку градиента потери: x' = x + ε·sign(∇ₓL). Классификатор — softmax-регрессия на реальных данных ириса. ε мерится в долях стандартного отклонения признаков; рядом показан сдвиг в сантиметрах.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Признаки цветка и предсказание классификатора под адверсариальной атакой",
        onResize: draw,
        drag: false,
      });

      function softmax(z) {
        var m = Math.max(z[0], z[1], z[2]);
        var e = [Math.exp(z[0] - m), Math.exp(z[1] - m), Math.exp(z[2] - m)];
        var s = e[0] + e[1] + e[2];
        return [e[0] / s, e[1] / s, e[2] / s];
      }
      function logits(x) {
        var z = [0, 0, 0];
        for (var k = 0; k < 3; k += 1) { z[k] = D.b[k]; for (var j = 0; j < 4; j += 1) z[k] += D.W[k][j] * x[j]; }
        return z;
      }
      function gradx(x, t) {
        var p = softmax(logits(x));
        var g = [0, 0, 0, 0];
        for (var k = 0; k < 3; k += 1) {
          var d = p[k] - (k === t ? 1 : 0);
          for (var j = 0; j < 4; j += 1) g[j] += d * D.W[k][j];
        }
        return g;
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";

        var ex = D.examples[state.pick];
        var t = ex.t;
        var g = gradx(ex.x, t);
        var xadv = ex.x.slice();
        for (var j = 0; j < 4; j += 1) xadv[j] += state.eps * (g[j] > 0 ? 1 : g[j] < 0 ? -1 : 0);
        var p0 = softmax(logits(ex.x));
        var padv = softmax(logits(xadv));
        var pred = padv[0] > padv[1] && padv[0] > padv[2] ? 0 : (padv[1] > padv[2] ? 1 : 2);
        var flipped = pred !== t;

        // left panel: feature bars
        var LX = 70, LY = 60, LW = 400, LH = 300;
        ctx.strokeStyle = C.line; ctx.strokeRect(LX, LY, LW, LH);
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("признаки цветка (стандартизованы)", LX, LY - 12);
        var mid = LY + LH / 2;
        ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(LX, mid); ctx.lineTo(LX + LW, mid); ctx.stroke();
        var bw = LW / 4;
        for (var f = 0; f < 4; f += 1) {
          var cx = LX + bw * f + bw / 2;
          var h0 = ex.x[f] * 55, ha = xadv[f] * 55;
          ctx.fillStyle = C.blue;
          ctx.fillRect(cx - 26, mid - Math.max(h0, 0), 22, Math.abs(h0));
          ctx.fillStyle = "rgba(185,74,59,0.85)";
          ctx.fillRect(cx + 4, mid - Math.max(ha, 0), 22, Math.abs(ha));
          ctx.fillStyle = C.muted; ctx.textAlign = "center";
          ctx.fillText(D.featc[f], cx, LY + LH + 16);
        }
        ctx.fillStyle = C.blue; ctx.textAlign = "left"; ctx.fillText("■ исходный", LX + 6, LY + 14);
        ctx.fillStyle = C.red; ctx.fillText("■ после атаки", LX + 110, LY + 14);

        // right panel: prediction bars
        var RX = 560, RY = 60, RW = 400, RH = 300;
        ctx.strokeStyle = C.line; ctx.strokeRect(RX, RY, RW, RH);
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("уверенность классификатора", RX, RY - 12);
        var pbw = RW / 3;
        for (var c = 0; c < 3; c += 1) {
          var px = RX + pbw * c + pbw / 2;
          var bh = padv[c] * (RH - 30);
          var isPred = c === pred;
          ctx.fillStyle = isPred ? (flipped ? C.red : C.green) : "rgba(110,114,106,0.35)";
          ctx.fillRect(px - 34, RY + RH - bh, 68, bh);
          ctx.fillStyle = C.ink; ctx.textAlign = "center";
          ctx.fillText(D.classes[c], px, RY + RH + 16);
          ctx.fillText((Math.round(padv[c] * 100)) + "%", px, RY + RH - bh - 10);
        }
        // true-class marker
        var tx = RX + pbw * t + pbw / 2;
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("↑ верный", tx, RY + RH + 32);

        var pcm = [];
        for (var q = 0; q < 4; q += 1) pcm.push(((xadv[q] - ex.x[q]) * D.sd[q]).toFixed(2).replace(".", ","));
        var rows = [
          { label: "Верный класс", value: D.classes[t], color: C.ink },
          { label: "Бюджет искажения ε", value: state.eps.toFixed(2).replace(".", ","), color: C.ink },
          { label: "Ответ модели", value: D.classes[pred] + " (" + Math.round(padv[pred] * 100) + "%)", color: flipped ? C.red : C.green },
          { label: "Сдвиг, см", value: pcm.join("; "), color: C.muted },
        ];
        rows.push(flipped
          ? { label: "Итог", value: "атака удалась — модель обманута", color: C.red }
          : { label: "Итог", value: "ответ пока верный", color: C.green });
        output.set(rows);
      }

      K.segmented(controls, {
        label: "Цветок",
        value: String(state.pick),
        options: [{ value: "0", label: "setosa" }, { value: "1", label: "versicolor" }, { value: "2", label: "virginica" }],
      }, function (v) { state.pick = parseInt(v, 10); state.eps = 0; if (epsSlider) epsSlider.set(0); draw(); });

      var epsSlider = K.slider(controls, { label: "Бюджет искажения ε", min: 0, max: 1.2, step: 0.05, value: state.eps,
        format: function (v) { return v.toFixed(2).replace(".", ","); } },
        function (v) { state.eps = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
