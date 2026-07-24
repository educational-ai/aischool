// Lesson 62: optimizer-lab — one valley, four update rules.
// Change the condition number, the learning rate, the momentum and the gradient noise,
// and watch which rule spends its budget of 80 gradients best.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("optimizer-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var STEPS = 80;
      var X0 = 9, Y0 = 1;
      var kappa = 100, logEta = -1.2, beta = 0.9, sigma = 0;
      var focus = "all", etaMode = "own";

      var METHODS = [
        { id: "gd", name: "GD", color: C.red },
        { id: "momentum", name: "Momentum", color: C.blue },
        { id: "adagrad", name: "AdaGrad", color: C.gold },
        { id: "adam", name: "Adam", color: C.violet }
      ];

      function noiseSeq(seed) {
        // deterministic pseudo-noise: the same picture for the same settings
        var s = seed >>> 0;
        return function () {
          s = (s * 1664525 + 1013904223) >>> 0;
          return (s / 4294967296) * 2 - 1;
        };
      }

      function fval(x, y) { return 0.5 * (x * x + kappa * y * y); }

      function run(id, eta) {
        var x = X0, y = Y0;
        var vx = 0, vy = 0, gx2 = 0, gy2 = 0, mx = 0, my = 0, sx = 0, sy = 0;
        var pts = [{ x: x, y: y, f: fval(x, y) }];
        var rnd = noiseSeq(20250062);
        var eps = 1e-8, b2 = 0.999, cross = 0;
        for (var t = 1; t <= STEPS; t += 1) {
          var gx = x + sigma * rnd();
          var gy = kappa * y + sigma * kappa * 0.1 * rnd();
          if (id === "gd") {
            x -= eta * gx; y -= eta * gy;
          } else if (id === "momentum") {
            vx = beta * vx + gx; vy = beta * vy + gy;
            x -= eta * vx; y -= eta * vy;
          } else if (id === "adagrad") {
            gx2 += gx * gx; gy2 += gy * gy;
            x -= eta * gx / (Math.sqrt(gx2) + eps);
            y -= eta * gy / (Math.sqrt(gy2) + eps);
          } else {
            mx = 0.9 * mx + 0.1 * gx; my = 0.9 * my + 0.1 * gy;
            sx = b2 * sx + (1 - b2) * gx * gx; sy = b2 * sy + (1 - b2) * gy * gy;
            var mhx = mx / (1 - Math.pow(0.9, t)), mhy = my / (1 - Math.pow(0.9, t));
            var shx = sx / (1 - Math.pow(b2, t)), shy = sy / (1 - Math.pow(b2, t));
            x -= eta * mhx / (Math.sqrt(shx) + eps);
            y -= eta * mhy / (Math.sqrt(shy) + eps);
          }
          if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6) {
            pts.push({ x: NaN, y: NaN, f: Infinity });
            return { pts: pts, last: Infinity, cross: cross, blown: true };
          }
          if (pts[pts.length - 1].y * y < 0) cross += 1;
          pts.push({ x: x, y: y, f: fval(x, y) });
        }
        return { pts: pts, last: pts[pts.length - 1].f, cross: cross, blown: false };
      }

      function bestEta(id) {
        var best = null, bestVal = Infinity;
        for (var k = 0; k <= 48; k += 1) {
          var lr = Math.pow(10, -3 + k * (3.6 / 48));
          var r = run(id, lr);
          if (r.last < bestVal) { bestVal = r.last; best = lr; }
        }
        return best === null ? Math.pow(10, logEta) : best;
      }

      K.hint(root, "Одна и та же чаша f(x,y) = (x² + κ·y²)/2 и четыре правила обновления. Бюджет у всех одинаковый: 80 вычисленных градиентов, и по умолчанию каждому методу подбирается свой лучший η из сетки. Поднимите обусловленность κ — и посмотрите, кто первым начнёт зигзагом биться о стенки оврага. Переключите режим на общий η — и увидите, как нечестное сравнение само рождает «победителя». Добавьте шум градиента σ — и порядок методов снова изменится.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Слева — траектории в овраге (звезда — минимум), справа — падение f по числу вычисленных градиентов в логарифмической шкале. Обрыв кривой означает расходимость. Момент накапливает скорость вдоль пологого направления; AdaGrad и Adam делят шаг на накопленный масштаб координаты и потому почти не замечают κ.");

      var cs = K.makeCanvas(stage, W, H, { label: "Траектории четырёх оптимизаторов в анизотропном овраге", onResize: draw, drag: false });

      var LX0 = 34, LX1 = 500, LY0 = 46, LY1 = 386;
      var RX0 = 566, RX1 = 872, RY0 = 46, RY1 = 386;

      function mx2s(x) { return LX0 + (x + 11) / 22 * (LX1 - LX0); }
      function my2s(y) { return (LY0 + LY1) / 2 - y / 1.7 * (LY1 - LY0) / 2; }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var eta = Math.pow(10, logEta);
        var etas = METHODS.map(function (m) {
          return etaMode === "own" ? bestEta(m.id) : eta;
        });
        var res = METHODS.map(function (m, k) { return run(m.id, etas[k]); });

        // ---- left: contours
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        [2, 10, 30, 70, 130, 220].forEach(function (L) {
          var ax = Math.sqrt(2 * L), ay = Math.sqrt(2 * L / kappa);
          ctx.beginPath();
          for (var i = 0; i <= 96; i += 1) {
            var th = i / 96 * Math.PI * 2;
            var px = mx2s(ax * Math.cos(th)), py = my2s(ay * Math.sin(th));
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
        });
        ctx.strokeStyle = C.axis; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(LX0, my2s(0)); ctx.lineTo(LX1, my2s(0)); ctx.stroke();
        ctx.fillStyle = C.ink; ctx.textAlign = "center";
        ctx.font = "16px PT Sans, sans-serif";
        ctx.fillText("★", mx2s(0), my2s(0) + 6);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.fillText("x — пологое направление", (LX0 + LX1) / 2, LY1 + 26);
        ctx.save();
        ctx.translate(LX0 - 16, (LY0 + LY1) / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("y — крутое", 0, 0);
        ctx.restore();
        ctx.textAlign = "left";
        ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("Овраг: κ = " + kappa
          + (etaMode === "own" ? ",  у каждого свой лучший η" : ",  общий η = " + eta.toPrecision(2)),
          LX0, LY0 - 16);

        METHODS.forEach(function (m, k) {
          var dim = focus !== "all" && focus !== m.id;
          var pts = res[k].pts;
          ctx.strokeStyle = m.color; ctx.globalAlpha = dim ? 0.16 : 1;
          ctx.lineWidth = dim ? 1.2 : 1.8;
          ctx.beginPath();
          var started = false;
          for (var i = 0; i < pts.length; i += 1) {
            if (!isFinite(pts[i].x)) break;
            var px = mx2s(pts[i].x), py = my2s(pts[i].y);
            if (px < LX0 - 60 || px > LX1 + 60 || py < LY0 - 60 || py > LY1 + 60) { started = false; continue; }
            if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
          }
          ctx.stroke();
          if (!dim) {
            ctx.fillStyle = m.color;
            for (var j = 0; j < pts.length; j += 4) {
              if (!isFinite(pts[j].x)) break;
              var qx = mx2s(pts[j].x), qy = my2s(pts[j].y);
              if (qx < LX0 || qx > LX1 || qy < LY0 || qy > LY1) continue;
              ctx.beginPath(); ctx.arc(qx, qy, 2.1, 0, 7); ctx.fill();
            }
          }
          ctx.globalAlpha = 1;
        });

        // ---- right: loss curves, log scale
        var lo = -6, hi = Math.log10(Math.max(fval(X0, Y0), 1)) + 0.4;
        function ly(f) {
          var v = Math.log10(Math.max(f, 1e-6));
          return RY1 - (v - lo) / (hi - lo) * (RY1 - RY0);
        }
        function lx(t) { return RX0 + t / STEPS * (RX1 - RX0); }
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        for (var p = Math.ceil(lo); p <= hi; p += 2) {
          var yy = ly(Math.pow(10, p));
          ctx.beginPath(); ctx.moveTo(RX0, yy); ctx.lineTo(RX1, yy); ctx.stroke();
          ctx.fillText("1e" + p, RX0 - 6, yy + 4);
        }
        ctx.strokeStyle = C.axis;
        ctx.beginPath(); ctx.moveTo(RX0, RY0); ctx.lineTo(RX0, RY1); ctx.lineTo(RX1, RY1); ctx.stroke();
        ctx.textAlign = "center"; ctx.fillStyle = C.muted; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("число вычисленных градиентов", (RX0 + RX1) / 2, RY1 + 26);
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("Потеря f при общем бюджете 80", RX0 - 14, LY0 - 16);

        METHODS.forEach(function (m, k) {
          var dim = focus !== "all" && focus !== m.id;
          ctx.strokeStyle = m.color; ctx.globalAlpha = dim ? 0.16 : 1;
          ctx.lineWidth = dim ? 1.2 : 2.1;
          ctx.beginPath();
          var pts = res[k].pts, on = false;
          for (var i = 0; i < pts.length; i += 1) {
            if (!isFinite(pts[i].f)) break;
            var px = lx(i), py = ly(pts[i].f);
            if (py < RY0) { on = false; continue; }
            if (!on) { ctx.moveTo(px, py); on = true; } else ctx.lineTo(px, py);
          }
          ctx.stroke();
          ctx.globalAlpha = 1;
        });

        var best = null;
        res.forEach(function (r, k) { if (best === null || r.last < res[best].last) best = k; });
        var items = METHODS.map(function (m, k) {
          var r = res[k];
          var value = r.blown ? "разошёлся" : r.last.toExponential(2).replace(".", ",");
          return {
            label: m.name + (k === best && !r.blown ? " ← лучший" : "")
              + " (η = " + etas[k].toPrecision(2) + "), пересечений оси: " + r.cross,
            value: value,
            color: m.color
          };
        });
        out.set(items);
      }

      K.slider(controls, {
        label: "обусловленность κ", min: 1, max: 300, step: 1, value: kappa,
        format: function (v) { return String(Math.round(v)); }
      }, function (v) { kappa = Math.round(v); draw(); });
      K.segmented(controls, {
        label: "learning rate", value: "own",
        options: [
          { value: "own", label: "свой лучший у каждого" },
          { value: "common", label: "общий для всех" }
        ]
      }, function (v) { etaMode = v; draw(); });
      K.slider(controls, {
        label: "общий η (действует во втором режиме)", min: -3, max: 0.5, step: 0.02, value: logEta,
        format: function (v) { return Math.pow(10, v).toPrecision(2); }
      }, function (v) { logEta = v; draw(); });
      K.slider(controls, {
        label: "инерция β (Momentum)", min: 0, max: 0.95, step: 0.01, value: beta,
        format: function (v) { return v.toFixed(2).replace(".", ","); }
      }, function (v) { beta = v; draw(); });
      K.slider(controls, {
        label: "шум градиента σ", min: 0, max: 3, step: 0.05, value: sigma,
        format: function (v) { return v.toFixed(2).replace(".", ","); }
      }, function (v) { sigma = v; draw(); });
      K.segmented(controls, {
        label: "показать", value: "all",
        options: [
          { value: "all", label: "все" },
          { value: "gd", label: "GD" },
          { value: "momentum", label: "Momentum" },
          { value: "adagrad", label: "AdaGrad" },
          { value: "adam", label: "Adam" }
        ]
      }, function (v) { focus = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
