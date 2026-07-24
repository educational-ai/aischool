// Lesson 32: overfitting-lab — slide model complexity, watch train and validation diverge.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("overfitting-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 500;

      var D = {"depths":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"train":[0.197,0.319,0.494,0.608,0.718,0.837,0.915,0.945,0.965,0.983,0.994,0.997,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"val":[0.201,0.32,0.476,0.557,0.66,0.727,0.797,0.788,0.802,0.797,0.797,0.802,0.797,0.797,0.797,0.797,0.797,0.797,0.797,0.797],"best":8};  // train/validation accuracy of a decision tree by depth, on real digits
      var depths = D.depths, TR = D.train, VA = D.val, BEST = D.best;
      var state = { i: 4 };  // index into depths (start at depth 5)

      K.hint(
        root,
        "Двигайте сложность модели — глубину дерева. Синяя линия (точность на обучении) неудержимо ползёт к 100%: дерево заучивает обучающие примеры. Зелёная (на отложенной проверке) сперва растёт вместе, но потом застывает и даже падает: заученное не переносится на новые данные. Расстояние между линиями — и есть переобучение.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Дерево решений растущей глубины на реальных рукописных цифрах. Точность на обучении почти всегда растёт со сложностью; настоящая цель — точность на новых данных (validation), которая имеет максимум. Лучшая сложность — там, где validation максимальна, а не там, где train.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Кривые точности на обучении и проверке по сложности модели", onResize: draw, drag: false });

      var box = { x: 90, y: 50, w: 860, h: 360 };
      function px(i) { return box.x + i / (depths.length - 1) * box.w; }
      function py(a) { return box.y + box.h - (a - 0.1) / 0.95 * box.h; }

      function curve(ctx, arr, col) {
        ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.beginPath();
        for (var i = 0; i < arr.length; i += 1) { var x = px(i), y = py(arr[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
        ctx.stroke();
        for (var j = 0; j < arr.length; j += 1) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px(j), py(arr[j]), 3, 0, Math.PI * 2); ctx.fill(); }
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        // axes + gridlines
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        [0.25, 0.5, 0.75, 1.0].forEach(function (a) {
          var y = py(a); ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
          ctx.fillStyle = C.muted; ctx.fillText(Math.round(a * 100) + "%", box.x - 8, y + 4);
        });

        // best-complexity line
        ctx.strokeStyle = "rgba(56,115,93,0.5)"; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(px(BEST), box.y); ctx.lineTo(px(BEST), box.y + box.h); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.green; ctx.textAlign = "center"; ctx.fillText("лучшая сложность", px(BEST), box.y - 8);

        // current complexity marker
        var i = state.i;
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(px(i), box.y); ctx.lineTo(px(i), box.y + box.h); ctx.stroke(); ctx.setLineDash([]);

        curve(ctx, TR, C.blue);
        curve(ctx, VA, C.green);
        // gap at current
        ctx.strokeStyle = "rgba(185,74,59,0.6)"; ctx.lineWidth = 6;
        ctx.beginPath(); ctx.moveTo(px(i), py(TR[i])); ctx.lineTo(px(i), py(VA[i])); ctx.stroke();
        ctx.lineWidth = 1;

        // labels on curves
        ctx.fillStyle = C.blue; ctx.textAlign = "left"; ctx.fillText("на обучении", px(depths.length - 1) - 90, py(TR[TR.length - 1]) - 8);
        ctx.fillStyle = C.green; ctx.fillText("на проверке (validation)", px(depths.length - 1) - 190, py(VA[VA.length - 1]) + 20);
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.fillText("сложность модели (глубина дерева)", box.x + box.w / 2, box.y + box.h + 34);

        var gap = TR[i] - VA[i];
        var verdict, vcol;
        if (VA[i] < 0.7 && TR[i] < 0.8) { verdict = "недообучение: обе точности низки"; vcol = C.muted; }
        else if (i <= BEST + 1 && i >= BEST - 1) { verdict = "хороший баланс — здесь и остановиться"; vcol = C.green; }
        else if (i > BEST + 1) { verdict = "переобучение: train растёт, validation стоит"; vcol = C.red; }
        else { verdict = "модель ещё простовата"; vcol = C.gold || "#a57920"; }
        output.set([
          { label: "Глубина дерева", value: String(depths[i]), color: C.ink },
          { label: "Точность на обучении", value: Math.round(TR[i] * 100) + "%", color: C.blue },
          { label: "Точность на проверке", value: Math.round(VA[i] * 100) + "%", color: C.green },
          { label: "Разрыв (переобучение)", value: Math.round(gap * 100) + "%", color: gap > 0.12 ? C.red : C.muted },
          { label: "Итог", value: verdict, color: vcol },
        ]);
      }

      K.slider(controls, { label: "Сложность модели (глубина дерева)", min: 0, max: depths.length - 1, step: 1, value: state.i,
        format: function (v) { return String(depths[v]); } }, function (v) { state.i = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
