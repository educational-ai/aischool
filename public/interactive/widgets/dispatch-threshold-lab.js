// Lesson 05: a month of dispatch decisions — probability, cost threshold, feedback.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("dispatch-threshold-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 640;
      var DAYS = 28;
      // Weekday overflow risk echoes the MTA weekly rhythm from the article.
      var BASE_RISK = [0.32, 0.38, 0.4, 0.36, 0.3, 0.1, 0.06];

      var state = {
        threshold: Number(options.threshold || 0.5),
        extra: Number(options.extra || 20),
        miss: Number(options.miss || 80),
        noise: Number(options.noise || 0.12),
        feedback: options.feedback === "on" ? "on" : "off",
      };

      K.hint(
        root,
        "Каждое утро модель называет вероятность переполнения, порог превращает её в решение. Сдвинь порог к расчётному q* и сравни счёт за месяц с интуитивным порогом 0,5.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Месяц из 28 дней с недельным ритмом риска. Кривая затрат пересчитывается для тех же самых дней, поэтому сравнение порогов честное. Обратная связь: выпущенный резерв приманивает пассажиров и поднимает риск того же дня следующей недели. Данные синтетические, seed = 2024.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Решения диспетчера за месяц и кривая месячных затрат по порогу",
        onResize: draw,
        drag: false,
      });

      function seeded(i, salt) {
        var x = Math.sin((i + 2024) * 17.271 + salt * 91.7) * 43758.5453;
        return x - Math.floor(x);
      }

      function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }

      function gauss(i, salt) {
        var u = Math.max(seeded(i, salt), 1e-9);
        var v = seeded(i, salt + 13);
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // Latent risk, model estimate and overflow draw for a fixed month.
      function baseDay(t) {
        var wobble = 0.05 * gauss(t, 3);
        return {
          latent: clamp(BASE_RISK[t % 7] + wobble, 0.02, 0.9),
          estimateNoise: gauss(t, 7),
          overflowDraw: seeded(t, 21),
        };
      }

      function runMonth(threshold, feedbackOn) {
        var lift = new Array(DAYS + 7).fill(0);
        var total = 0;
        var releases = 0;
        var missed = 0;
        var latentOverflows = 0;
        var observedOverflows = 0;
        var days = [];
        for (var t = 0; t < DAYS; t += 1) {
          var day = baseDay(t);
          var latent = clamp(day.latent + lift[t], 0.02, 0.95);
          var estimate = clamp(latent + state.noise * day.estimateNoise, 0.01, 0.99);
          var release = estimate >= threshold;
          var wouldOverflow = day.overflowDraw < latent;
          if (wouldOverflow) latentOverflows += 1;
          var cost = 0;
          if (release) {
            cost = state.extra;
            releases += 1;
            if (feedbackOn) lift[t + 7] += 0.06;
          } else if (wouldOverflow) {
            cost = state.miss;
            missed += 1;
            observedOverflows += 1;
          }
          total += cost;
          days.push({
            latent: latent,
            estimate: estimate,
            release: release,
            missed: !release && wouldOverflow,
          });
        }
        return {
          days: days,
          total: total,
          releases: releases,
          missed: missed,
          latentOverflows: latentOverflows,
          observedOverflows: observedOverflows,
        };
      }

      function costCurve(feedbackOn) {
        var points = [];
        var best = { threshold: 0, total: Infinity };
        for (var q = 0.02; q <= 0.98; q += 0.02) {
          var total = runMonth(q, feedbackOn).total;
          points.push({ q: q, total: total });
          if (total < best.total) best = { threshold: q, total: total };
        }
        return { points: points, best: best };
      }

      var strip = { x: 70, y: 46, w: 910, h: 210 };
      var curve = { x: 70, y: 356, w: 910, h: 230 };

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        var feedbackOn = state.feedback === "on";
        var month = runMonth(state.threshold, feedbackOn);
        var sweep = costCurve(feedbackOn);
        var qStar = clamp(state.extra / state.miss, 0.02, 0.98);

        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        // --- Daily strip -----------------------------------------------
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Месяц решений: вероятность переполнения и действие", strip.x, strip.y - 24);
        ctx.font = "13px PT Sans, sans-serif";

        var barW = strip.w / DAYS;
        // Weekend columns first, behind the bars.
        for (var wd = 0; wd < DAYS; wd += 1) {
          if (wd % 7 === 5 || wd % 7 === 6) {
            ctx.fillStyle = C.wash;
            ctx.fillRect(strip.x + barW * wd, strip.y, barW, strip.h);
          }
        }
        for (var g = 0; g <= 1.0001; g += 0.25) {
          var gy = strip.y + strip.h - g * strip.h;
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(strip.x, gy);
          ctx.lineTo(strip.x + strip.w, gy);
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText(g.toFixed(2).replace(".", ","), strip.x - 8, gy);
        }
        month.days.forEach(function (day, t) {
          var x = strip.x + barW * t;
          var barH = day.estimate * strip.h;
          var color = day.release ? C.gold : day.missed ? C.red : C.axis;
          ctx.fillStyle = color;
          ctx.globalAlpha = day.release || day.missed ? 0.95 : 0.55;
          ctx.fillRect(x + barW * 0.18, strip.y + strip.h - barH, barW * 0.64, barH);
          ctx.globalAlpha = 1;
        });
        var thrY = strip.y + strip.h - state.threshold * strip.h;
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        ctx.moveTo(strip.x, thrY);
        ctx.lineTo(strip.x + strip.w, thrY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.textAlign = "left";
        ctx.fillStyle = C.ink;
        ctx.fillText("порог " + state.threshold.toFixed(2).replace(".", ","), strip.x + strip.w - 92, thrY - 12);

        // Legend under the strip.
        var ly = strip.y + strip.h + 22;
        function legendDot(x, color, text) {
          ctx.fillStyle = color;
          ctx.fillRect(x, ly - 6, 12, 12);
          ctx.fillStyle = C.muted;
          ctx.fillText(text, x + 18, ly);
          return x + 18 + ctx.measureText(text).width + 26;
        }
        var lx = strip.x;
        lx = legendDot(lx, C.gold, "резерв выпущен (цена " + state.extra + ")");
        lx = legendDot(lx, C.red, "переполнение пропущено (цена " + state.miss + ")");
        legendDot(lx, C.axis, "тихий день без действия");

        // --- Cost curve -------------------------------------------------
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.fillText("Затраты за месяц в зависимости от порога", curve.x, curve.y - 24);
        ctx.font = "13px PT Sans, sans-serif";

        var maxCost = sweep.points.reduce(function (m, p) { return Math.max(m, p.total); }, 1);
        function cx(q) { return curve.x + q * curve.w; }
        function cy(total) { return curve.y + curve.h - total / maxCost * (curve.h - 18); }

        for (var gq = 0; gq <= 1.0001; gq += 0.25) {
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(cx(gq), curve.y);
          ctx.lineTo(cx(gq), curve.y + curve.h);
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "center";
          ctx.fillText(gq.toFixed(2).replace(".", ","), cx(gq), curve.y + curve.h + 16);
        }
        [0.25, 0.5, 0.75, 1].forEach(function (frac) {
          var total = Math.round(maxCost * frac);
          var yy = cy(maxCost * frac);
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(curve.x, yy);
          ctx.lineTo(curve.x + curve.w, yy);
          ctx.stroke();
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText(String(total), curve.x - 8, yy);
        });
        ctx.textAlign = "left";
        ctx.fillStyle = C.muted;
        ctx.fillText("ед.", curve.x - 40, curve.y - 8);

        ctx.strokeStyle = C.blue;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        sweep.points.forEach(function (p, i) {
          if (i === 0) ctx.moveTo(cx(p.q), cy(p.total));
          else ctx.lineTo(cx(p.q), cy(p.total));
        });
        ctx.stroke();

        // Computed threshold from prices.
        ctx.strokeStyle = C.green;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(cx(qStar), curve.y + 6);
        ctx.lineTo(cx(qStar), curve.y + curve.h);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.green;
        ctx.textAlign = qStar > 0.75 ? "right" : "left";
        ctx.fillText(
          "расчётный порог q* = " + qStar.toFixed(2).replace(".", ","),
          cx(qStar) + (qStar > 0.75 ? -8 : 8),
          curve.y + 18,
        );

        // Current position.
        var current = runMonth(state.threshold, feedbackOn).total;
        ctx.fillStyle = C.ink;
        ctx.beginPath();
        ctx.arc(cx(state.threshold), cy(current), 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.textAlign = state.threshold > 0.72 ? "right" : "left";
        ctx.fillText(
          "твой порог: " + current + " единиц",
          cx(state.threshold) + (state.threshold > 0.72 ? -10 : 10),
          cy(current) - 14,
        );
        ctx.textAlign = "left";

        output.set([
          { label: "затраты за месяц", value: String(month.total) + " ед." },
          { label: "расчётный q*", value: qStar.toFixed(2).replace(".", ",") },
          {
            label: "лучший порог на кривой",
            value: sweep.best.threshold.toFixed(2).replace(".", ","),
            color: C.blue,
          },
          {
            label: "пропущено переполнений",
            value: month.missed + " из " + month.latentOverflows,
            color: month.missed > 0 ? C.red : C.green,
          },
          {
            label: "наблюдаемая доля переполнений",
            value: (month.observedOverflows / DAYS * 100).toFixed(0) + "%",
            color: C.gold,
          },
        ]);
      }

      K.slider(
        controls,
        {
          label: "Порог выпуска резерва",
          min: 0.05,
          max: 0.95,
          step: 0.05,
          value: state.threshold,
          format: function (v) { return v.toFixed(2).replace(".", ","); },
        },
        function (value) {
          state.threshold = value;
          draw();
        },
      );
      K.slider(
        controls,
        { label: "Цена лишнего рейса", min: 5, max: 60, step: 5, value: state.extra },
        function (value) {
          state.extra = value;
          draw();
        },
      );
      K.slider(
        controls,
        { label: "Цена пропуска", min: 20, max: 200, step: 10, value: state.miss },
        function (value) {
          state.miss = value;
          draw();
        },
      );
      K.slider(
        controls,
        {
          label: "Шум модели σ",
          min: 0,
          max: 0.3,
          step: 0.03,
          value: state.noise,
          format: function (v) { return v.toFixed(2).replace(".", ","); },
        },
        function (value) {
          state.noise = value;
          draw();
        },
      );
      K.segmented(
        controls,
        {
          label: "Обратная связь спроса",
          value: state.feedback,
          options: [
            { value: "off", label: "выключена" },
            { value: "on", label: "включена" },
          ],
        },
        function (value) {
          state.feedback = value;
          draw();
        },
      );

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
