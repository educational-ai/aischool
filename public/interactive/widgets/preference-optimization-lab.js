// Lesson 81: preference-optimization-lab — tilt a reference policy by a learned reward,
// watch KL grow, and see the independent (hidden) value peak and fall when the reward
// has a surface loophole. Group normalisation is shown as a change of effective beta.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("preference-optimization-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var PB = "#fffef9";

      var NAMES = ["краткий точный", "подробный точный", "многословный",
        "уверенный, но неточный", "уклончивый", "с оговорками"];
      var PREF = [0.22, 0.20, 0.18, 0.16, 0.14, 0.10];      // reference policy
      var BASE = [0.70, 0.80, 0.55, 0.35, 0.30, 0.60];      // learned proxy reward
      var HACKV = [0.00, 0.00, 1.00, 0.35, 0.00, 0.00];     // surface loophole direction
      var TRUE = [0.92, 0.88, 0.55, 0.25, 0.40, 0.70];      // hidden human value
      var N = NAMES.length;

      var logBeta = Math.log(0.6) / Math.LN10;   // slider works in log10(beta)
      var hack = 0.45;                            // strength of the loophole
      var grouped = 0;                           // 0 — raw advantage, 1 — group-normalised

      function reward() {
        var r = [], i;
        for (i = 0; i < N; i += 1) r.push(BASE[i] + hack * HACKV[i]);
        return r;
      }
      function spread(r) {
        var m = 0, i, s = 0;
        for (i = 0; i < N; i += 1) m += r[i] / N;
        for (i = 0; i < N; i += 1) s += (r[i] - m) * (r[i] - m) / N;
        return Math.sqrt(s);
      }
      function effBeta(b, r) {
        return grouped ? b * spread(r) : b;   // деление advantage на s ≡ деление beta на s
      }
      function policy(b, r) {
        var z = [], i, mx = -1e9, p = [], s = 0;
        for (i = 0; i < N; i += 1) { z.push(Math.log(PREF[i]) + r[i] / b); if (z[i] > mx) mx = z[i]; }
        for (i = 0; i < N; i += 1) { p.push(Math.exp(z[i] - mx)); s += p[i]; }
        for (i = 0; i < N; i += 1) p[i] /= s;
        return p;
      }
      function stats(p, r) {
        var kl = 0, pr = 0, tv = 0, top = 0, i;
        for (i = 0; i < N; i += 1) {
          if (p[i] > 1e-12) kl += p[i] * Math.log(p[i] / PREF[i]);
          pr += p[i] * r[i]; tv += p[i] * TRUE[i];
          if (p[i] > top) top = p[i];
        }
        return { kl: kl, proxy: pr, value: tv, top: top };
      }
      function sweep(r) {
        var pts = [], t;
        for (t = 0; t <= 120; t += 1) {
          var b = Math.exp(Math.log(4.0) + (Math.log(0.05) - Math.log(4.0)) * t / 120);
          pts.push(stats(policy(effBeta(b, r), r), r));
        }
        return pts;
      }

      K.hint(root, "Слева — распределение ответов: серое было у reference-модели, цветное получилось после оптимизации предпочтений. Справа — та самая ловушка: по горизонтали KL (насколько ушли от reference), по вертикали два числа. Красное — обученный reward, синее — скрытая настоящая польза, которую оптимизатор не видит. Уменьшайте β и включайте лазейку: красная кривая растёт всегда, синяя разворачивается.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Модельный пример из шести ответов: числа условные, механика настоящая. π* ∝ π_ref·exp(r/β). Лазейка добавляет reward многословному и самоуверенному ответу, не меняя их пользы. Нормировка advantage внутри группы делит r на разброс s, то есть тайно уменьшает β — та же кривая, но проходится быстрее.");
      var cs = K.makeCanvas(stage, W, H, { label: "Распределение ответов и кривая Гудхарта", onResize: draw, drag: false });

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var r = reward();
        var b = Math.pow(10, logBeta);
        var be = effBeta(b, r);
        var p = policy(be, r);
        var st = stats(p, r);
        var pts = sweep(r);

        // ---------------- left panel: distribution over responses
        var LX = 50, LY = 60, LW = 380, LH = 340;
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("вероятности ответов", LX, LY - 22);
        var rowH = LH / N;
        for (var i = 0; i < N; i += 1) {
          var y = LY + i * rowH;
          var wRef = PREF[i] * LW * 1.9, wPi = p[i] * LW * 1.9;
          ctx.fillStyle = "rgba(110,114,106,0.35)";
          ctx.fillRect(LX, y + 6, wRef, 12);
          ctx.fillStyle = HACKV[i] > 0 && hack > 0.01 ? C.red : C.blue;
          ctx.fillRect(LX, y + 20, wPi, 16);
          ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(NAMES[i], LX, y + 2);
          ctx.fillStyle = C.ink;
          ctx.fillText(p[i].toFixed(2), LX + Math.max(wPi, wRef) + 8, y + 33);
          ctx.fillStyle = C.muted;
          ctx.fillText("r = " + r[i].toFixed(2) + ",  польза " + TRUE[i].toFixed(2),
            LX + 250, y + 33);
        }
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("серая полоска — π_ref, цветная — π после оптимизации", LX, LY + LH + 18);

        // ---------------- right panel: Goodhart curve
        var GX = 520, GY = 60, GW = 330, GH = 300;
        var klMax = 0, k;
        for (k = 0; k < pts.length; k += 1) if (pts[k].kl > klMax) klMax = pts[k].kl;
        klMax = Math.max(klMax, 0.2);
        var vLo = 1e9, vHi = -1e9, pLo = 1e9, pHi = -1e9;
        for (k = 0; k < pts.length; k += 1) {
          vLo = Math.min(vLo, pts[k].value); vHi = Math.max(vHi, pts[k].value);
          pLo = Math.min(pLo, pts[k].proxy); pHi = Math.max(pHi, pts[k].proxy);
        }
        if (vHi - vLo < 1e-6) { vHi = vLo + 1e-3; }
        if (pHi - pLo < 1e-6) { pHi = pLo + 1e-3; }
        function gx(kl) { return GX + GW * Math.min(kl, klMax) / klMax; }
        function gyV(v) { return GY + GH - GH * (v - vLo) / (vHi - vLo); }
        function gyP(v) { return GY + GH - GH * (v - pLo) / (pHi - pLo); }
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(GX, GY); ctx.lineTo(GX, GY + GH); ctx.lineTo(GX + GW, GY + GH); ctx.stroke();
        ctx.strokeStyle = C.red; ctx.lineWidth = 2;
        ctx.beginPath();
        for (k = 0; k < pts.length; k += 1) {
          var x1 = gx(pts[k].kl), y1 = gyP(pts[k].proxy);
          if (k === 0) ctx.moveTo(x1, y1); else ctx.lineTo(x1, y1);
        }
        ctx.stroke();
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.6;
        ctx.beginPath();
        for (k = 0; k < pts.length; k += 1) {
          var x2 = gx(pts[k].kl), y2 = gyV(pts[k].value);
          if (k === 0) ctx.moveTo(x2, y2); else ctx.lineTo(x2, y2);
        }
        ctx.stroke();
        // peak of the hidden value
        var best = 0;
        for (k = 0; k < pts.length; k += 1) if (pts[k].value > pts[best].value) best = k;
        ctx.fillStyle = C.gold;
        ctx.beginPath(); ctx.arc(gx(pts[best].kl), gyV(pts[best].value), 6, 0, 7); ctx.fill();
        ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("максимум пользы", gx(pts[best].kl), gyV(pts[best].value) - 12);
        // current position
        ctx.fillStyle = C.ink;
        ctx.beginPath(); ctx.arc(gx(st.kl), gyV(st.value), 5, 0, 7); ctx.fill();
        ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.strokeStyle = "rgba(23,25,21,0.35)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(gx(st.kl), GY); ctx.lineTo(gx(st.kl), GY + GH); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("KL к reference →", GX, GY + GH + 20);
        ctx.fillStyle = C.red; ctx.fillText("обученный reward", GX, GY - 26);
        ctx.fillStyle = C.blue; ctx.fillText("настоящая польза (скрыта от оптимизатора)", GX, GY - 10);

        output.set([
          { label: "β (штраф за уход)", value: b.toFixed(2), color: C.ink },
          { label: grouped ? "эффективное β после нормировки" : "эффективное β",
            value: be.toFixed(2), color: C.violet },
          { label: "KL(π‖π_ref)", value: st.kl.toFixed(2), color: C.gold },
          { label: "средний proxy-reward", value: st.proxy.toFixed(3), color: C.red },
          { label: "средняя настоящая польза", value: st.value.toFixed(3), color: C.blue },
          { label: "доля самого частого ответа", value: st.top.toFixed(2), color: C.green }
        ]);
      }

      K.slider(controls, {
        label: "β: штраф за уход от reference",
        min: -1.3, max: 0.75, step: 0.02, value: logBeta,
        format: function (v) { return Math.pow(10, v).toFixed(2); }
      }, function (v) { logBeta = v; draw(); });
      K.slider(controls, {
        label: "сила лазейки в reward",
        min: 0, max: 1.2, step: 0.05, value: hack,
        format: function (v) { return Number(v).toFixed(2); }
      }, function (v) { hack = v; draw(); });
      K.segmented(controls, {
        label: "advantage",
        value: 0,
        options: [{ label: "сырой (PPO)", value: 0 }, { label: "нормированный в группе (GRPO)", value: 1 }]
      }, function (v) { grouped = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
