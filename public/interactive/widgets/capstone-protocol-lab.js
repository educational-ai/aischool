// Lesson 90: capstone-protocol-lab — the protocol decides what you will believe.
// All numbers are real MAE values of HistGradientBoostingRegressor on the UCI Bike Sharing
// hourly file, produced by scripts/generate_lesson90_visuals.py (seed 2026, max_iter=300).
// Rows: reported error inside the chosen protocol vs error on the untouched future block
// (October–December 2012, 2168 hours, never used for training in any configuration).
(function () {
  "use strict";

  var FRACS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
  var TRAIN_N = [1300, 2600, 5201, 7801, 10402, 13003];
  var DATA = {
    "time-honest": {
      reported: [43.3, 43.6, 47.5, 57.4, 40.6, 40.4],
      future: [75.9, 53.0, 56.1, 52.8, 46.1, 45.5]
    },
    "random-honest": {
      reported: [36.4, 29.0, 26.0, 24.4, 23.9, 23.1],
      future: [56.4, 54.7, 44.7, 45.4, 43.0, 42.3]
    },
    "time-leaky": {
      reported: [5.4, 4.4, 4.4, 4.4, 4.6, 4.9],
      future: [4.5, 3.4, 3.0, 3.0, 3.2, 3.4]
    },
    "random-leaky": {
      reported: [4.2, 3.0, 2.1, 1.9, 1.8, 1.6],
      future: [6.5, 4.3, 3.0, 2.5, 2.5, 2.3]
    }
  };
  var HOUR_BASELINE = 88.5;

  function fmt(v, d) {
    return v.toFixed(d === undefined ? 1 : d).replace(".", ",");
  }

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("capstone-protocol-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var split = "time", feat = "honest", fi = 5;

      K.hint(root, "Одна и та же модель, одни и те же реальные данные велопроката (17 379 часов). " +
        "Меняется только протокол: как разбита выборка и какие признаки разрешены. Золотая полоса — " +
        "ошибка, о которой вы отчитываетесь по своей валидации. Синяя — ошибка на честно отложенном " +
        "будущем: октябрь–декабрь 2012 года, эти часы не участвуют в обучении ни в одной конфигурации.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Случайное перемешивание почасового ряда занижает ошибку почти вдвое: соседний час " +
        "того же дня попадает и в обучение, и в проверку. Утечка признака (casual + registered = cnt) " +
        "не ловится никаким разбиением — только вопросом «когда этот признак становится известен?». " +
        "Кривая внизу показывает обе ошибки при разном объёме обучающих данных.");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Ошибка, о которой отчитались, и ошибка на будущем блоке при разных протоколах",
        onResize: draw,
        drag: false
      });

      function series() {
        return DATA[split + "-" + feat];
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var s = series();
        var rep = s.reported[fi], fut = s.future[fi];
        var maxV = 80;
        // ---- bars
        var BX = 80, BY0 = 40, BH = 190, scale = BH / maxV;
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.lineWidth = 1;
        for (var g = 0; g <= maxV; g += 20) {
          var yy = BY0 + BH - g * scale;
          ctx.beginPath(); ctx.moveTo(BX - 10, yy); ctx.lineTo(W - 40, yy); ctx.stroke();
          ctx.fillStyle = C.muted; ctx.textAlign = "right";
          ctx.fillText(String(g), BX - 18, yy + 4);
        }
        function bar(x, v, color, title, sub) {
          var h = Math.min(v, maxV) * scale;
          ctx.fillStyle = color;
          ctx.fillRect(x, BY0 + BH - h, 120, h);
          ctx.fillStyle = C.ink || "#171915";
          ctx.textAlign = "center";
          ctx.font = "17px PT Sans, sans-serif";
          ctx.fillText(fmt(v), x + 60, BY0 + BH - h - 10);
          ctx.font = "13px PT Sans, sans-serif";
          ctx.fillStyle = C.muted;
          ctx.fillText(title, x + 60, BY0 + BH + 22);
          ctx.fillText(sub, x + 60, BY0 + BH + 40);
        }
        bar(BX + 30, rep, C.gold, "отчёт по своей", "валидации");
        bar(BX + 230, fut, C.blue, "проверка на будущем", "октябрь–декабрь");
        // honest baseline line
        var yb = BY0 + BH - Math.min(HOUR_BASELINE, maxV) * scale;
        ctx.strokeStyle = C.red; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(BX - 10, yb); ctx.lineTo(W - 40, yb); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.red; ctx.textAlign = "left";
        ctx.fillText("baseline «среднее по часу и типу дня» — " + fmt(HOUR_BASELINE), BX + 420, yb - 8);

        var ratio = fut / rep;
        ctx.fillStyle = ratio > 1.35 ? C.red : C.green;
        ctx.font = "16px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("разрыв ×" + fmt(ratio, 2) + (ratio > 1.35 ? "  — отчёт врёт" : "  — отчёт честен"),
          BX + 420, BY0 + 40);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.fillStyle = C.muted;
        var note = feat === "leaky"
          ? "MAE ≈ 3 при среднем спросе 189,5 — «слишком хорошо». Признаки casual и registered в сумме дают target и известны только постфактум."
          : (split === "random"
            ? "Соседние часы одного дня попали и в train, и в валидацию: протокол сам себе подсказал."
            : "Обучение до 30 июня, валидация июль–сентябрь: проверка устроена так же, как будущее.");
        wrap(ctx, note, BX + 420, BY0 + 66, 400, 18);

        // ---- curve over train size
        var CX0 = BX + 30, CX1 = W - 60, CY0 = 300, CH = 130;
        var cmax = 80;
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.beginPath(); ctx.moveTo(CX0, CY0 + CH); ctx.lineTo(CX1, CY0 + CH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(CX0, CY0); ctx.lineTo(CX0, CY0 + CH); ctx.stroke();
        function px(i) { return CX0 + (CX1 - CX0) * i / (FRACS.length - 1); }
        function py(v) { return CY0 + CH - Math.min(v, cmax) / cmax * CH; }
        function poly(arr, color) {
          ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.beginPath();
          for (var i = 0; i < arr.length; i += 1) {
            var x = px(i), y = py(arr[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
          for (var j = 0; j < arr.length; j += 1) {
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(px(j), py(arr[j]), j === fi ? 6.5 : 3.5, 0, 7); ctx.fill();
          }
        }
        poly(s.reported, C.gold);
        poly(s.future, C.blue);
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        for (var t = 0; t < FRACS.length; t += 1) {
          ctx.fillText(String(TRAIN_N[t]), px(t), CY0 + CH + 18);
        }
        ctx.fillText("размер обучающей выборки, часов", (CX0 + CX1) / 2, CY0 + CH + 40);
        ctx.textAlign = "right"; ctx.fillText("MAE", CX0 - 12, CY0 + 10);

        output.set([
          { label: "Отчётная ошибка (своя валидация)", value: fmt(rep) + " поездок/час", color: C.gold },
          { label: "Ошибка на будущем блоке", value: fmt(fut) + " поездок/час", color: C.blue },
          { label: "Во сколько раз отчёт оптимистичнее", value: "×" + fmt(ratio, 2), color: ratio > 1.35 ? C.red : C.green },
          { label: "Обучающих часов", value: String(TRAIN_N[fi]), color: C.muted }
        ]);
      }

      function wrap(ctx, text, x, y, maxw, lh) {
        var words = text.split(" "), line = "", lines = [];
        for (var i = 0; i < words.length; i += 1) {
          var probe = line ? line + " " + words[i] : words[i];
          if (ctx.measureText(probe).width > maxw && line) { lines.push(line); line = words[i]; }
          else line = probe;
        }
        if (line) lines.push(line);
        ctx.textAlign = "left";
        for (var k = 0; k < lines.length; k += 1) ctx.fillText(lines[k], x, y + k * lh);
      }

      K.segmented(controls, {
        label: "Разбиение",
        value: "time",
        options: [{ label: "по времени", value: "time" }, { label: "случайное", value: "random" }]
      }, function (v) { split = v; draw(); });

      K.segmented(controls, {
        label: "Признаки",
        value: "honest",
        options: [
          { label: "честные (известны заранее)", value: "honest" },
          { label: "с утечкой (casual, registered)", value: "leaky" }
        ]
      }, function (v) { feat = v; draw(); });

      K.slider(controls, {
        label: "Доля обучающих данных",
        min: 0, max: 5, step: 1, value: 5,
        format: function (v) { return Math.round(FRACS[v] * 100) + "% (" + TRAIN_N[v] + " ч)"; }
      }, function (v) { fi = Math.max(0, Math.min(5, Math.round(v))); draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
