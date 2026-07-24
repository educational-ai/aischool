// Lesson 87: forensic-threshold-lab — threshold, base rate and generator shift for a fake detector.
// Move the threshold and the share of fakes in the stream: watch PPV collapse, watch the
// cost-optimal threshold move, and watch the whole picture die when the generator changes.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("forensic-threshold-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 520;
      var PL = 62, PR = 40, PT = 40, PB = 300;        // density panel box
      var N = 10000;
      var GRIDN = 401;

      // Score models: normal densities on [0,1], calibrated on the measured lesson numbers.
      var REAL = { m: 0.16, s: 0.205 };
      var MODES = {
        own: { label: "свой генератор", fake: { m: 0.80, s: 0.24 } },
        jpeg: { label: "он же после пережатия", fake: { m: 0.45, s: 0.25 } },
        alien: { label: "незнакомый генератор", fake: { m: 0.10, s: 0.14 } }
      };

      var mode = "own";
      var t = 0.5;
      var pi = 0.01;
      var ratio = 20;
      var dragging = false;

      function grid() {
        var xs = new Array(GRIDN), i;
        for (i = 0; i < GRIDN; i += 1) xs[i] = i / (GRIDN - 1);
        return xs;
      }
      var XS = grid();

      function density(par) {
        var d = new Array(GRIDN), sum = 0, i, z;
        for (i = 0; i < GRIDN; i += 1) {
          z = (XS[i] - par.m) / par.s;
          d[i] = Math.exp(-0.5 * z * z);
          sum += d[i];
        }
        for (i = 0; i < GRIDN; i += 1) d[i] /= sum;   // discrete probabilities
        return d;
      }

      function tail(d, thr) {                          // P(score >= thr)
        var acc = 0, i;
        for (i = 0; i < GRIDN; i += 1) if (XS[i] >= thr) acc += d[i];
        return acc;
      }

      function stats(thr) {
        var dr = density(REAL), df = density(MODES[mode].fake);
        var sens = tail(df, thr);
        var fpr = tail(dr, thr);
        var spec = 1 - fpr;
        var tp = N * pi * sens, fn = N * pi * (1 - sens);
        var fp = N * (1 - pi) * fpr, tn = N * (1 - pi) * spec;
        var alarms = tp + fp;
        var ppv = alarms > 1e-9 ? tp / alarms : 0;
        var cost = ratio * fn + fp;
        return { dr: dr, df: df, sens: sens, spec: spec, tp: tp, fn: fn, fp: fp, tn: tn,
                 alarms: alarms, ppv: ppv, cost: cost };
      }

      function bestThreshold() {
        var best = 0, bc = Infinity, i, s;
        for (i = 1; i < GRIDN; i += 1) {
          s = stats(XS[i]);
          if (s.cost < bc) { bc = s.cost; best = XS[i]; }
        }
        return best;
      }

      K.hint(root, "Слева — распределения оценок детектора для настоящих файлов и для подделок; вертикальная красная линия — ваш порог (её можно тащить прямо по картинке). Справа внизу — что получится, если прогнать через этот порог 10 000 файлов. Двигайте долю подделок: при редкой подделке даже хороший детектор выдаёт в основном ложные тревоги. Переключите генератор на незнакомый — и оба распределения слипнутся.");

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Распределения оценок модельные: они подобраны так, чтобы при пороге 0,5 детектор на своём генераторе имел чувствительность около 0,87 и специфичность около 0,94 — примерно как измеренный в уроке. Золотая штриховая линия — порог, минимизирующий ожидаемую цену при выбранных base rate и отношении цен c_FN/c_FP. Он не равен 0,5 и сдвигается вместе с редкостью подделок и с ценой ошибки. PPV — доля истинных подделок среди поднятых тревог.");

      var cs = K.makeCanvas(stage, W, H, { label: "Порог детектора, редкость подделок и цена ошибок", onResize: draw, drag: true });

      K.segmented(controls, {
        label: "что проверяем",
        value: mode,
        options: [
          { value: "own", label: "свой генератор" },
          { value: "jpeg", label: "после пережатия" },
          { value: "alien", label: "незнакомый" }
        ]
      }, function (value) { mode = value; draw(); });

      K.slider(controls, {
        label: "порог t", min: 0.02, max: 0.98, step: 0.01, value: t,
        format: function (v) { return v.toFixed(2); }
      }, function (v) { t = v; draw(); });

      K.slider(controls, {
        label: "доля подделок в потоке", min: 0.1, max: 40, step: 0.1, value: 1,
        format: function (v) { return v.toFixed(1) + " %"; }
      }, function (v) { pi = v / 100; draw(); });

      K.slider(controls, {
        label: "цена пропуска / цена ложной тревоги", min: 1, max: 60, step: 1, value: ratio,
        format: function (v) { return v.toFixed(0) + " : 1"; }
      }, function (v) { ratio = v; draw(); });

      function x2p(x) { return PL + x * (W - PL - PR); }
      function p2x(px) { return (px - PL) / (W - PL - PR); }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var s = stats(t);
        var tb = bestThreshold();
        var top = PT, bottom = H - PB, i, px, py;
        var peak = 0;
        for (i = 0; i < GRIDN; i += 1) {
          if (s.dr[i] > peak) peak = s.dr[i];
          if (s.df[i] > peak) peak = s.df[i];
        }
        function y2p(v) { return bottom - (v / peak) * (bottom - top); }

        // axes
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(PL, bottom); ctx.lineTo(W - PR, bottom); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (i = 0; i <= 10; i += 2) {
          px = x2p(i / 10);
          ctx.beginPath(); ctx.moveTo(px, bottom); ctx.lineTo(px, bottom + 5); ctx.stroke();
          ctx.fillText((i / 10).toFixed(1), px, bottom + 20);
        }
        ctx.fillText("оценка детектора «это подделка»", (PL + W - PR) / 2, bottom + 40);

        // shaded tails above the threshold
        function tailPath(d, color, alpha) {
          ctx.beginPath();
          var started = false;
          for (i = 0; i < GRIDN; i += 1) {
            if (XS[i] < t) continue;
            px = x2p(XS[i]); py = y2p(d[i]);
            if (!started) { ctx.moveTo(px, bottom); started = true; }
            ctx.lineTo(px, py);
          }
          if (started) {
            ctx.lineTo(x2p(1), bottom);
            ctx.closePath();
            ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
          }
        }
        tailPath(s.dr, C.red, 0.28);
        tailPath(s.df, C.blue, 0.22);

        function line(d, color) {
          ctx.beginPath();
          for (i = 0; i < GRIDN; i += 1) {
            px = x2p(XS[i]); py = y2p(d[i]);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.stroke();
        }
        line(s.dr, C.green);
        line(s.df, C.blue);

        // legend
        ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = C.green; ctx.fillText("настоящие файлы", PL + 8, top + 6);
        ctx.fillStyle = C.blue; ctx.fillText("подделки: " + MODES[mode].label, PL + 8, top + 24);
        ctx.fillStyle = C.red; ctx.fillText("закрашено красным — ложные тревоги", PL + 8, top + 42);

        // best threshold and current threshold
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(x2p(tb), top - 10); ctx.lineTo(x2p(tb), bottom); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.gold; ctx.textAlign = "center";
        ctx.fillText("t* = " + tb.toFixed(2), x2p(tb), top - 16);

        ctx.strokeStyle = C.red; ctx.lineWidth = 2.6;
        ctx.beginPath(); ctx.moveTo(x2p(t), top - 10); ctx.lineTo(x2p(t), bottom + 6); ctx.stroke();
        ctx.fillStyle = C.red;
        ctx.beginPath(); ctx.arc(x2p(t), bottom + 6, 6, 0, 7); ctx.fill();

        // ------- flow panel
        var by = H - PB + 70;
        var barX = PL, barW = W - PL - PR, barH = 30;
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("10 000 файлов на входе", barX, by - 10);
        var wf = Math.max(2, barW * pi);
        ctx.fillStyle = C.blue; ctx.fillRect(barX, by, wf, barH);
        ctx.fillStyle = C.green; ctx.fillRect(barX + wf, by, barW - wf, barH);
        ctx.fillStyle = C.paper; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("подделок " + Math.round(N * pi), barX + 6, by + 20);
        ctx.textAlign = "right";
        ctx.fillText("настоящих " + Math.round(N * (1 - pi)), barX + barW - 8, by + 20);

        var ay = by + 74;
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("поднято тревог: " + Math.round(s.alarms)
          + "   (истинных " + Math.round(s.tp) + ", ложных " + Math.round(s.fp) + ")", barX, ay - 10);
        var wt = s.alarms > 1e-9 ? barW * (s.tp / s.alarms) : 0;
        ctx.fillStyle = C.blue; ctx.fillRect(barX, ay, wt, barH);
        ctx.fillStyle = C.red; ctx.fillRect(barX + wt, ay, barW - wt, barH);
        ctx.fillStyle = C.paper; ctx.font = "12px PT Sans, sans-serif";
        if (wt > 90) ctx.fillText("верные " + (s.ppv * 100).toFixed(1) + " %", barX + 6, ay + 20);
        ctx.textAlign = "right";
        if (barW - wt > 90) ctx.fillText("ложные " + ((1 - s.ppv) * 100).toFixed(1) + " %", barX + barW - 8, ay + 20);
        ctx.textAlign = "left"; ctx.fillStyle = C.muted; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("пропущено подделок: " + Math.round(s.fn), barX, ay + barH + 22);

        output.set([
          { label: "sensitivity (доля найденных подделок)", value: s.sens.toFixed(3), color: C.blue },
          { label: "specificity", value: s.spec.toFixed(3), color: C.green },
          { label: "PPV: тревога верна с вероятностью", value: (s.ppv * 100).toFixed(1) + " %", color: C.red },
          { label: "ожидаемая цена (в ложных тревогах)", value: Math.round(s.cost), color: C.gold },
          { label: "порог минимальной цены t*", value: tb.toFixed(2), color: C.violet }
        ]);
      }

      function pointerT(ev) {
        var rect = cs.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        return Math.max(0.02, Math.min(0.98, p2x(mx)));
      }
      function down(ev) { dragging = true; t = pointerT(ev); draw(); ev.preventDefault(); }
      function move(ev) { if (!dragging) return; t = pointerT(ev); draw(); }
      function up() { dragging = false; }
      cs.canvas.addEventListener("pointerdown", down);
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);

      draw();

      return function () {
        cs.canvas.removeEventListener("pointerdown", down);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        cs.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
