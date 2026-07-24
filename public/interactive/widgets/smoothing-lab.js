// Lesson 48: smoothing-lab — a pseudo-count prior pulls small groups toward the common rate; large groups barely move.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("smoothing-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var mean = 0.2, strength = 10;  // prior Beta(alpha, beta), alpha=mean*strength
      var groups = [[0, 3], [1, 8], [4, 20], [20, 100], [210, 1000]];

      K.hint(
        root,
        "Сопряжённый prior работает как псевдосчётчики: он прибавляет воображаемые успехи и неудачи к данным. Двигайте среднее и силу prior и смотрите, как оценки групп стягиваются к общему уровню. Малые группы (мало данных) сдвигаются сильно, крупные почти не двигаются — а группа с нулём событий получает не ноль, а маленькую положительную оценку.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Сглаженная оценка = (h + α) / (n + α + β), где α = среднее × сила, β = (1 − среднее) × сила. Это апостериорное среднее Beta-prior. Пустой кружок — сырая доля h/n, заполненный — сглаженная. Пунктир — среднее prior, к которому всё стягивается.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Стягивание оценок групп к общему уровню", onResize: draw, drag: false });

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var alpha = mean * strength, beta = (1 - mean) * strength;

        var gx = 130, gy = 55, gw = W - 260, gh = 320;
        var xhi = 0.5;
        function X(v) { return gx + v / xhi * gw; }
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, gy, gw, gh);
        // prior mean line
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(X(mean), gy); ctx.lineTo(X(mean), gy + gh); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.ink; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("среднее prior " + mean.toFixed(2), X(mean), gy - 8);

        var rows = groups.length;
        for (var i = 0; i < rows; i += 1) {
          var h = groups[i][0], n = groups[i][1];
          var raw = h / n, sm = (h + alpha) / (n + alpha + beta);
          var y = gy + gh - (i + 0.5) / rows * gh;
          ctx.strokeStyle = C.line; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(X(raw), y); ctx.lineTo(X(sm), y); ctx.stroke();
          // arrow head toward sm
          var dir = sm > raw ? 1 : -1;
          ctx.beginPath(); ctx.moveTo(X(sm), y); ctx.lineTo(X(sm) - dir * 7, y - 4); ctx.lineTo(X(sm) - dir * 7, y + 4); ctx.closePath();
          ctx.fillStyle = C.gold; ctx.fill();
          ctx.strokeStyle = C.blue; ctx.fillStyle = "white"; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.arc(X(raw), y, 6, 0, 7); ctx.fill(); ctx.stroke();
          ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(X(sm), y, 5.5, 0, 7); ctx.fill();
          // labels
          ctx.fillStyle = C.ink; ctx.textAlign = "right"; ctx.font = "12px PT Sans, sans-serif";
          ctx.fillText(h + "/" + n, gx - 12, y + 4);
        }
        // x ticks
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        for (var t = 0; t <= 5; t += 1) ctx.fillText((t / 5 * xhi).toFixed(1), X(t / 5 * xhi), gy + gh + 16);
        ctx.fillText("доля / оценка", gx + gw / 2, gy + gh + 32);
        // legend
        ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.strokeStyle = C.blue; ctx.fillStyle = "white"; ctx.beginPath(); ctx.arc(gx + gw + 22, gy + 16, 6, 0, 7); ctx.fill(); ctx.stroke();
        ctx.fillStyle = C.ink; ctx.fillText("сырая", gx + gw + 34, gy + 20);
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(gx + gw + 22, gy + 40, 5.5, 0, 7); ctx.fill();
        ctx.fillStyle = C.ink; ctx.fillText("сглаж.", gx + gw + 34, gy + 44);

        // zero-group succession
        var z = groups[0], zraw = 0, zsm = (0 + alpha) / (z[1] + alpha + beta);
        output.set([
          { label: "Prior Beta(α, β)", value: "Beta(" + alpha.toFixed(1) + ", " + beta.toFixed(1) + ")", color: C.blue },
          { label: "Группа 0/3: сырая → сглаж.", value: "0 → " + zsm.toFixed(3), color: C.red },
          { label: "Крупная 210/1000: сдвиг", value: "мал: " + (210 / 1000).toFixed(3) + " → " + ((210 + alpha) / (1000 + alpha + beta)).toFixed(3), color: C.muted },
        ]);
      }

      K.slider(controls, { label: "Среднее prior", min: 0.05, max: 0.5, step: 0.01, value: 0.2, format: function (v) { return v.toFixed(2); } }, function (v) { mean = v; draw(); });
      K.slider(controls, { label: "Сила prior (α+β)", min: 1, max: 100, step: 1, value: 10, format: function (v) { return String(v); } }, function (v) { strength = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
