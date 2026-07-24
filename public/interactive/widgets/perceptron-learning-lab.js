// Lesson 15: perceptron learning on Iris — watch the boundary rotate to fit.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("perceptron-learning-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;

      // Iris petal length/width, label ±1. From scripts/data/iris.data.
      var DATA = {"sep":[[1.4,0.2,1],[1.4,0.2,1],[1.3,0.2,1],[1.5,0.2,1],[1.4,0.2,1],[1.7,0.4,1],[1.4,0.3,1],[1.5,0.2,1],[1.4,0.2,1],[1.5,0.1,1],[1.5,0.2,1],[1.6,0.2,1],[1.4,0.1,1],[1.1,0.1,1],[1.2,0.2,1],[1.5,0.4,1],[1.3,0.4,1],[1.4,0.3,1],[1.7,0.3,1],[1.5,0.3,1],[1.7,0.2,1],[1.5,0.4,1],[1.0,0.2,1],[1.7,0.5,1],[1.9,0.2,1],[1.6,0.2,1],[1.6,0.4,1],[1.5,0.2,1],[1.4,0.2,1],[1.6,0.2,1],[1.6,0.2,1],[1.5,0.4,1],[1.5,0.1,1],[1.4,0.2,1],[1.5,0.1,1],[1.2,0.2,1],[1.3,0.2,1],[1.5,0.1,1],[1.3,0.2,1],[1.5,0.2,1],[1.3,0.3,1],[1.3,0.3,1],[1.3,0.2,1],[1.6,0.6,1],[1.9,0.4,1],[1.4,0.3,1],[1.6,0.2,1],[1.4,0.2,1],[1.5,0.2,1],[1.4,0.2,1],[4.7,1.4,-1],[4.5,1.5,-1],[4.9,1.5,-1],[4.0,1.3,-1],[4.6,1.5,-1],[4.5,1.3,-1],[4.7,1.6,-1],[3.3,1.0,-1],[4.6,1.3,-1],[3.9,1.4,-1],[3.5,1.0,-1],[4.2,1.5,-1],[4.0,1.0,-1],[4.7,1.4,-1],[3.6,1.3,-1],[4.4,1.4,-1],[4.5,1.5,-1],[4.1,1.0,-1],[4.5,1.5,-1],[3.9,1.1,-1],[4.8,1.8,-1],[4.0,1.3,-1],[4.9,1.5,-1],[4.7,1.2,-1],[4.3,1.3,-1],[4.4,1.4,-1],[4.8,1.4,-1],[5.0,1.7,-1],[4.5,1.5,-1],[3.5,1.0,-1],[3.8,1.1,-1],[3.7,1.0,-1],[3.9,1.2,-1],[5.1,1.6,-1],[4.5,1.5,-1],[4.5,1.6,-1],[4.7,1.5,-1],[4.4,1.3,-1],[4.1,1.3,-1],[4.0,1.3,-1],[4.4,1.2,-1],[4.6,1.4,-1],[4.0,1.2,-1],[3.3,1.0,-1],[4.2,1.3,-1],[4.2,1.2,-1],[4.2,1.3,-1],[4.3,1.3,-1],[3.0,1.1,-1],[4.1,1.3,-1]],"non":[[4.7,1.4,1],[4.5,1.5,1],[4.9,1.5,1],[4.0,1.3,1],[4.6,1.5,1],[4.5,1.3,1],[4.7,1.6,1],[3.3,1.0,1],[4.6,1.3,1],[3.9,1.4,1],[3.5,1.0,1],[4.2,1.5,1],[4.0,1.0,1],[4.7,1.4,1],[3.6,1.3,1],[4.4,1.4,1],[4.5,1.5,1],[4.1,1.0,1],[4.5,1.5,1],[3.9,1.1,1],[4.8,1.8,1],[4.0,1.3,1],[4.9,1.5,1],[4.7,1.2,1],[4.3,1.3,1],[4.4,1.4,1],[4.8,1.4,1],[5.0,1.7,1],[4.5,1.5,1],[3.5,1.0,1],[3.8,1.1,1],[3.7,1.0,1],[3.9,1.2,1],[5.1,1.6,1],[4.5,1.5,1],[4.5,1.6,1],[4.7,1.5,1],[4.4,1.3,1],[4.1,1.3,1],[4.0,1.3,1],[4.4,1.2,1],[4.6,1.4,1],[4.0,1.2,1],[3.3,1.0,1],[4.2,1.3,1],[4.2,1.2,1],[4.2,1.3,1],[4.3,1.3,1],[3.0,1.1,1],[4.1,1.3,1],[6.0,2.5,-1],[5.1,1.9,-1],[5.9,2.1,-1],[5.6,1.8,-1],[5.8,2.2,-1],[6.6,2.1,-1],[4.5,1.7,-1],[6.3,1.8,-1],[5.8,1.8,-1],[6.1,2.5,-1],[5.1,2.0,-1],[5.3,1.9,-1],[5.5,2.1,-1],[5.0,2.0,-1],[5.1,2.4,-1],[5.3,2.3,-1],[5.5,1.8,-1],[6.7,2.2,-1],[6.9,2.3,-1],[5.0,1.5,-1],[5.7,2.3,-1],[4.9,2.0,-1],[6.7,2.0,-1],[4.9,1.8,-1],[5.7,2.1,-1],[6.0,1.8,-1],[4.8,1.8,-1],[4.9,1.8,-1],[5.6,2.1,-1],[5.8,1.6,-1],[6.1,1.9,-1],[6.4,2.0,-1],[5.6,2.2,-1],[5.1,1.5,-1],[5.6,1.4,-1],[6.1,2.3,-1],[5.6,2.4,-1],[5.5,1.8,-1],[4.8,1.8,-1],[5.4,2.1,-1],[5.6,2.4,-1],[5.1,2.3,-1],[5.1,1.9,-1],[5.9,2.3,-1],[5.7,2.5,-1],[5.2,2.3,-1],[5.0,1.9,-1],[5.2,2.0,-1],[5.4,2.3,-1],[5.1,1.8,-1]]};

      var state = { pair: "sep", eta: 0.1, w: [0, 0, 0], idx: 0, epoch: 1, errThis: 0, errLast: null, timer: null };

      K.hint(
        root,
        "Жмите «шаг»: перцептрон берёт точки по очереди и на каждой ошибке доворачивает границу. Разделимая пара сходится к нулю ошибок; неразделимая мечется без конца.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Реальные ирисы Фишера: длина и ширина лепестка. setosa/versicolor разделимы широким зазором — граница находится за пару проходов; versicolor/virginica перекрываются, и ошибки не гаснут. Граница = прямая, перпендикулярная вектору весов; порог спрятан в вес при постоянном входе.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Точки ирисов и разделяющая прямая перцептрона",
        onResize: draw,
        drag: false,
      });

      var box = { x: 70, y: 34, w: 660, h: 480 };
      var XMIN = 0.5, XMAX = 7.2, YMIN = -0.2, YMAX = 2.7;

      function px(v) { return box.x + (v - XMIN) / (XMAX - XMIN) * box.w; }
      function py(v) { return box.y + box.h - (v - YMIN) / (YMAX - YMIN) * box.h; }

      function pts() { return DATA[state.pair]; }

      function reset() {
        stopAuto();
        state.w = [0, 0, 0];
        state.idx = 0;
        state.epoch = 1;
        state.errThis = 0;
        state.errLast = null;
        draw();
      }

      function step() {
        var P = pts();
        var p = P[state.idx];
        var s = state.w[0] * p[0] + state.w[1] * p[1] + state.w[2];
        var pred = s >= 0 ? 1 : -1;
        if (pred !== p[2]) {
          state.w[0] += state.eta * p[2] * p[0];
          state.w[1] += state.eta * p[2] * p[1];
          state.w[2] += state.eta * p[2] * 1;
          state.errThis += 1;
        }
        state.idx += 1;
        if (state.idx >= P.length) {
          state.idx = 0;
          state.errLast = state.errThis;
          state.errThis = 0;
          state.epoch += 1;
          if (state.errLast === 0) stopAuto();
        }
        draw();
      }

      function startAuto() {
        if (state.timer) return;
        state.timer = setInterval(function () {
          step();
          if (state.errLast === 0 && state.idx === 0) stopAuto();
        }, 60);
      }
      function stopAuto() {
        if (state.timer) { clearInterval(state.timer); state.timer = null; }
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";
        var P = pts();

        // axes
        ctx.strokeStyle = C.line;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        [1, 2, 3, 4, 5, 6, 7].forEach(function (v) {
          ctx.fillText(String(v), px(v), box.y + box.h + 16);
        });
        ctx.fillText("длина лепестка, см", box.x + box.w / 2, box.y + box.h + 36);
        ctx.textAlign = "right";
        [0, 0.5, 1, 1.5, 2, 2.5].forEach(function (v) {
          ctx.fillText(String(v).replace(".", ","), box.x - 8, py(v));
        });
        ctx.save();
        ctx.translate(box.x - 40, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText("ширина лепестка, см", 0, 0);
        ctx.restore();

        // boundary line: w0*x + w1*y + w2 = 0
        ctx.save();
        ctx.beginPath();
        ctx.rect(box.x, box.y, box.w, box.h);
        ctx.clip();
        if (Math.abs(state.w[1]) > 1e-9 || Math.abs(state.w[0]) > 1e-9) {
          ctx.strokeStyle = C.blue;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          if (Math.abs(state.w[1]) >= Math.abs(state.w[0])) {
            var y1 = -(state.w[0] * XMIN + state.w[2]) / state.w[1];
            var y2 = -(state.w[0] * XMAX + state.w[2]) / state.w[1];
            ctx.moveTo(px(XMIN), py(y1));
            ctx.lineTo(px(XMAX), py(y2));
          } else {
            var x1 = -(state.w[1] * YMIN + state.w[2]) / state.w[0];
            var x2 = -(state.w[1] * YMAX + state.w[2]) / state.w[0];
            ctx.moveTo(px(x1), py(YMIN));
            ctx.lineTo(px(x2), py(YMAX));
          }
          ctx.stroke();
          ctx.lineWidth = 1;
        }
        ctx.restore();

        // points; highlight current
        P.forEach(function (p, i) {
          var pos = p[2] === 1;
          ctx.beginPath();
          if (pos) ctx.arc(px(p[0]), py(p[1]), 5, 0, Math.PI * 2);
          else { ctx.rect(px(p[0]) - 4.5, py(p[1]) - 4.5, 9, 9); }
          ctx.fillStyle = pos ? C.green : C.violet;
          ctx.globalAlpha = 0.9;
          ctx.fill();
          ctx.globalAlpha = 1;
          if (i === state.idx && !state.timer) {
            ctx.strokeStyle = C.gold;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.arc(px(p[0]), py(p[1]), 10, 0, Math.PI * 2);
            ctx.stroke();
            ctx.lineWidth = 1;
          }
        });

        var labels = state.pair === "sep"
          ? ["setosa (+1)", "versicolor (−1)"]
          : ["versicolor (+1)", "virginica (−1)"];
        ctx.textAlign = "left";
        ctx.fillStyle = C.green;
        ctx.fillText("● " + labels[0], box.x + 12, box.y + 18);
        ctx.fillStyle = C.violet;
        ctx.fillText("■ " + labels[1], box.x + 12, box.y + 40);

        var rows = [
          { label: "Проход (эпоха)", value: String(state.epoch), color: C.ink },
          { label: "Ошибок в текущем проходе", value: String(state.errThis), color: C.ink },
          { label: "Ошибок в прошлом проходе", value: state.errLast === null ? "—" : String(state.errLast), color: state.errLast === 0 ? C.green : C.ink },
        ];
        if (state.errLast === 0) {
          rows.push({ label: "Итог", value: "разделено, граница найдена", color: C.green });
        } else if (state.pair === "non" && state.epoch > 6) {
          rows.push({ label: "Итог", value: "мечется: пара неразделима", color: C.red });
        }
        output.set(rows);
      }

      K.segmented(controls, {
        label: "Пара видов",
        value: state.pair,
        options: [
          { value: "sep", label: "setosa / versicolor (разделимы)" },
          { value: "non", label: "versicolor / virginica (нет)" },
        ],
      }, function (v) { state.pair = v; reset(); });

      function actionButton(parent, label, onClick) {
        var b = K.element("button", "kontur-int-segment", { type: "button", text: label });
        b.addEventListener("click", onClick);
        parent.appendChild(b);
        return b;
      }
      var btnWrap = K.element("fieldset", "kontur-int-control kontur-int-control--segmented");
      btnWrap.appendChild(K.element("legend", "kontur-int-label-name", { text: "Управление" }));
      var btnGroup = K.element("div", "kontur-int-segments");
      btnWrap.appendChild(btnGroup);
      controls.appendChild(btnWrap);
      actionButton(btnGroup, "шаг", function () { stopAuto(); step(); });
      actionButton(btnGroup, "автопрогон", function () { startAuto(); });
      actionButton(btnGroup, "пауза", function () { stopAuto(); });
      actionButton(btnGroup, "сброс", function () { reset(); });

      draw();
      return function () {
        stopAuto();
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
