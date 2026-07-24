// Lesson 56: uncertainty-map-lab — two different signals of doubt (confidence and distance to the
// training data), the reject threshold, and what happens to coverage, selective risk and cost when
// the population shifts. Deterministic pseudo-data: the same seed gives the same picture every time.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("uncertainty-map-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 480;
      var N = 320;
      var confT = 0.85, distT = 2.6, mode = 0, pop = 0;
      var pts = [];

      function rng(seed) {
        var s = seed >>> 0;
        return function () {
          s += 0x6D2B79F5;
          var t = s;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function sigmoid(v) { return 1 / (1 + Math.exp(-v)); }

      // модель: p = sigmoid(4*x), обучена на «обычной» популяции возле начала координат
      function phat(x) { return sigmoid(4 * x.x1); }
      // расстояние до обучающей области (масштаб взят из разброса обучающих признаков)
      function dist(x) { return Math.sqrt((x.x1 / 1.6) * (x.x1 / 1.6) + (x.x2 / 1.0) * (x.x2 / 1.0)); }

      function build() {
        var r = rng(5656), i;
        pts = [];
        for (i = 0; i < N; i += 1) {
          var g1 = 0, g2 = 0, k;
          for (k = 0; k < 6; k += 1) { g1 += r(); g2 += r(); }
          g1 = (g1 - 3) / Math.sqrt(0.5); g2 = (g2 - 3) / Math.sqrt(0.5);
          var ood = pop === 1 && i % 3 === 0;              // треть объектов — новая среда
          var p;
          if (ood) {
            var o = { x1: 0.9 * g1, x2: 3.4 + 0.45 * g2 };
            p = sigmoid(3 * o.x1 - 3.2);                    // в новой среде правило другое
            o.y = r() < p ? 1 : 0; o.ood = true;
            pts.push(o);
          } else {
            var q = { x1: 1.25 * g1, x2: 0.95 * g2 };
            p = sigmoid(3 * q.x1);                          // честная случайность исхода
            q.y = r() < p ? 1 : 0; q.ood = false;
            pts.push(q);
          }
        }
        for (i = 0; i < pts.length; i += 1) {
          pts[i].p = phat(pts[i]);
          pts[i].conf = Math.max(pts[i].p, 1 - pts[i].p);
          pts[i].d = dist(pts[i]);
          pts[i].wrong = (pts[i].p >= 0.5 ? 1 : 0) !== pts[i].y;
        }
      }

      function accepted(pt) {
        var okConf = mode === 1 ? true : pt.conf >= confT;
        var okDist = mode === 0 ? true : pt.d <= distT;
        return okConf && okDist;
      }
      function score(pt) {
        var a = (pt.conf - 0.5) * 2;                        // 0 — сомнение, 1 — уверенность
        var b = Math.max(0, 1 - pt.d / 4.5);
        if (mode === 0) return a;
        if (mode === 1) return b;
        return Math.min(a, b);
      }

      function stats() {
        var auto = 0, err = 0, errRej = 0, rej = 0, i;
        for (i = 0; i < pts.length; i += 1) {
          if (accepted(pts[i])) { auto += 1; if (pts[i].wrong) err += 1; }
          else { rej += 1; if (pts[i].wrong) errRej += 1; }
        }
        var cost = (err * 1000 + rej * 50) / pts.length;
        return {
          cov: auto / pts.length,
          risk: auto > 0 ? err / auto : 0,
          riskRej: rej > 0 ? errRej / rej : 0,
          err: err, rej: rej, cost: cost
        };
      }

      function curve() {
        var arr = pts.slice().sort(function (a, b) { return score(b) - score(a); });
        var out = [], e = 0, i;
        for (i = 0; i < arr.length; i += 1) {
          if (arr[i].wrong) e += 1;
          out.push({ cov: (i + 1) / arr.length, risk: e / (i + 1) });
        }
        return out;
      }

      K.hint(
        root,
        "Одна цифра прогноза скрывает разные сомнения. Здесь два независимых сигнала: уверенность модели (насколько p далеко от 0,5) и расстояние до обучающей области. Двигайте пороги и следите за парой «coverage — selective risk». Затем включите сдвиг популяции: треть объектов приходит из новой среды, где правило другое. Уверенность там по-прежнему высока — и порог, выбранный на validation, перестаёт выполнять обещание. Расстояние ловит именно этих чужаков.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Слева — карта: серая эллиптическая область показывает обучающую популяцию, кружки — объекты, крестики внутри них — ошибки модели. Полупрозрачные объекты переданы человеку. Справа — кривая risk-coverage для выбранного сигнала и текущая рабочая точка. Цена посчитана по правилу: ошибка автомата 1000 ₽, передача человеку 50 ₽, верное автоматическое решение бесплатно.",
      );

      var cs = K.makeCanvas(stage, W, H, { label: "Карта неопределённости: сигналы, порог, охват и риск", onResize: draw, drag: false });

      var GX = 40, GY = 40, GW = 430, GH = 400;
      function X(v) { return GX + (v + 5.2) / 10.4 * GW; }
      function Y(v) { return GY + GH - (v + 4.2) / 9.4 * GH; }

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        // ----- левая панель: карта признаков
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(GX, GY, GW, GH);
        // обучающая область: эллипс двух сигм
        ctx.save();
        ctx.beginPath(); ctx.rect(GX, GY, GW, GH); ctx.clip();
        ctx.beginPath();
        ctx.ellipse(X(0), Y(0), (X(distT * 1.6) - X(0)), (Y(0) - Y(distT * 1.0)), 0, 0, 7);
        ctx.fillStyle = "rgba(110,114,106,0.10)"; ctx.fill();
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); ctx.stroke();
        ctx.setLineDash([]); ctx.restore();
        // полоса низкой уверенности вокруг границы p = 0,5
        var xEdge = Math.log(confT / (1 - confT)) / 4;
        ctx.save();
        ctx.beginPath(); ctx.rect(GX, GY, GW, GH); ctx.clip();
        ctx.fillStyle = "rgba(49,95,140,0.10)";
        ctx.fillRect(X(-xEdge), GY, X(xEdge) - X(-xEdge), GH);
        ctx.restore();
        ctx.strokeStyle = C.blue; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(X(0), GY); ctx.lineTo(X(0), GY + GH); ctx.stroke();

        for (i = 0; i < pts.length; i += 1) {
          var pt = pts[i], acc = accepted(pt), px = X(pt.x1), py = Y(pt.x2);
          if (px < GX || px > GX + GW || py < GY || py > GY + GH) continue;
          ctx.globalAlpha = acc ? 1 : 0.22;
          ctx.fillStyle = pt.y === 1 ? C.red : C.blue;
          ctx.beginPath(); ctx.arc(px, py, 4.2, 0, 7); ctx.fill();
          if (pt.wrong) {
            ctx.strokeStyle = acc ? C.ink : C.muted; ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.moveTo(px - 3, py - 3); ctx.lineTo(px + 3, py + 3);
            ctx.moveTo(px + 3, py - 3); ctx.lineTo(px - 3, py + 3);
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("признак 1 (по нему модель и решает)", GX + GW / 2, GY + GH + 20);
        ctx.save();
        ctx.translate(GX - 14, GY + GH / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("признак 2", 0, 0); ctx.restore();
        ctx.fillStyle = C.gold; ctx.textAlign = "left";
        ctx.fillText("граница «своей» области", GX + 8, GY + 16);
        ctx.fillStyle = C.blue;
        ctx.fillText("полоса низкой уверенности", GX + 8, GY + 32);
        if (pop === 1) {
          ctx.fillStyle = C.red;
          ctx.fillText("новая среда: другое правило", GX + 8, GY + 48);
        }

        // ----- правая панель: risk-coverage
        var PX = 540, PY = 42, PW = 320, PH = 190;
        var cur = curve(), s = stats();
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(PX, PY, PW, PH);
        var maxR = 0.02;
        for (i = 0; i < cur.length; i += 1) if (cur[i].risk > maxR) maxR = cur[i].risk;
        maxR = Math.max(maxR, s.risk) * 1.15 + 0.01;
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.2; ctx.beginPath();
        for (i = 0; i < cur.length; i += 1) {
          var cx = PX + cur[i].cov * PW, cy = PY + PH - (cur[i].risk / maxR) * PH;
          if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        // случайный отказ: горизонталь на полной ошибке
        var full = cur[cur.length - 1].risk;
        ctx.strokeStyle = C.muted; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(PX, PY + PH - (full / maxR) * PH); ctx.lineTo(PX + PW, PY + PH - (full / maxR) * PH);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "10px PT Sans, sans-serif";
        ctx.fillText("случайный отказ: " + (full * 100).toFixed(1) + "%", PX + 6, PY + PH - (full / maxR) * PH - 5);
        // рабочая точка
        var opx = PX + s.cov * PW, opy = PY + PH - (s.risk / maxR) * PH;
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(opx, opy, 6, 0, 7); ctx.fill();
        ctx.strokeStyle = "#fffef9"; ctx.lineWidth = 1.4; ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("coverage, %", PX + PW / 2, PY + PH + 18);
        ctx.textAlign = "left";
        ctx.fillText("0", PX - 2, PY + PH + 14); ctx.textAlign = "right";
        ctx.fillText("100", PX + PW, PY + PH + 14);
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("selective risk vs coverage", PX, PY - 10);

        // ----- цифры
        var TY = PY + PH + 52;
        ctx.font = "12px PT Sans, sans-serif"; ctx.fillStyle = C.muted;
        ctx.fillText("автоматически решено", PX, TY);
        ctx.fillText("ошибок на автомате", PX, TY + 46);
        ctx.fillText("ошибок среди переданных", PX, TY + 92);
        ctx.font = "22px PT Sans, sans-serif"; ctx.fillStyle = C.ink;
        ctx.fillText((s.cov * 100).toFixed(0) + "%  (риск " + (s.risk * 100).toFixed(1) + "%)", PX, TY + 24);
        ctx.fillStyle = s.err > 0 ? C.red : C.green;
        ctx.fillText(String(s.err) + " из " + (pts.length - s.rej), PX, TY + 70);
        ctx.fillStyle = C.gold;
        ctx.fillText((s.riskRej * 100).toFixed(1) + "%", PX, TY + 116);

        output.set([
          { label: "Coverage", value: (s.cov * 100).toFixed(1) + "%", color: C.blue },
          { label: "Selective risk", value: (s.risk * 100).toFixed(1) + "%", color: s.risk > 0.05 ? C.red : C.green },
          { label: "Передано человеку", value: String(s.rej), color: C.gold },
          { label: "Цена на случай", value: s.cost.toFixed(0) + " ₽", color: C.ink },
        ]);
      }

      K.segmented(controls, {
        label: "Сигнал отказа", value: 0, options: [
          { label: "уверенность", value: 0 },
          { label: "расстояние", value: 1 },
          { label: "оба", value: 2 }]
      }, function (v) { mode = v; draw(); });
      K.segmented(controls, {
        label: "Популяция", value: 0, options: [
          { label: "как на обучении", value: 0 },
          { label: "сдвиг: новая среда", value: 1 }]
      }, function (v) { pop = v; build(); draw(); });
      K.slider(controls, {
        label: "Порог уверенности", min: 0.5, max: 0.999, step: 0.001, value: confT,
        format: function (v) { return v.toFixed(3); }
      }, function (v) { confT = v; draw(); });
      K.slider(controls, {
        label: "Порог расстояния", min: 0.5, max: 5, step: 0.05, value: distT,
        format: function (v) { return v.toFixed(2); }
      }, function (v) { distT = v; draw(); });

      build();
      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
