// Lesson 46: confidence-lab — run many surveys, watch what fraction of confidence intervals actually catch the fixed truth.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("confidence-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var truep = 0.6, n = 100, z = 1.96, level = "95%";
      var runs = 0, covered = 0, recent = [];

      function reset() { runs = 0; covered = 0; recent = []; }

      function sample() {
        var h = 0;
        for (var i = 0; i < n; i += 1) if (Math.random() < truep) h += 1;
        var ph = h / n, se = Math.sqrt(ph * (1 - ph) / n);
        var lo = ph - z * se, hi = ph + z * se;
        return { ph: ph, lo: lo, hi: hi, hit: lo <= truep && truep <= hi };
      }
      function run(k) {
        for (var m = 0; m < k; m += 1) {
          var s = sample(); runs += 1; if (s.hit) covered += 1;
          recent.push(s); if (recent.length > 45) recent.shift();
        }
        draw();
      }

      K.hint(
        root,
        "Что на самом деле значит «95% доверительный интервал»? Истинная доля фиксирована. Проводите опрос за опросом: каждый даёт свой интервал, и примерно 95 из 100 накрывают истину, а около пяти промахиваются. Двигайте истинную долю к краю (например, 0,05) — и увидите, что простой интервал Вальда там накрывает реже обещанного.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Каждая горизонтальная линия — один опрос и его интервал Вальда p̂ ± z·√(p̂(1−p̂)/n). Синие накрывают истинную долю (вертикальная линия), красные — нет. Доля накрывших стремится к заявленному уровню при верных предпосылках, но у края (p близко к 0 или 1) простой интервал систематически промахивается чаще.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Повторные опросы и покрытие интервалов", onResize: draw, drag: false });

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        // interval plot region
        var gx = 60, gy = 50, gw = 560, gh = 360;
        var lo = Math.max(0, truep - 0.45), hi = Math.min(1, truep + 0.45);
        function X(v) { return gx + (v - lo) / (hi - lo) * gw; }
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, gy, gw, gh);
        // truth line
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(X(truep), gy); ctx.lineTo(X(truep), gy + gh); ctx.stroke();
        ctx.fillStyle = C.ink; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("истина p = " + truep.toFixed(2), X(truep), gy - 8);
        // recent intervals
        var rows = recent.length;
        for (var i = 0; i < rows; i += 1) {
          var s = recent[i], y = gy + gh - (i + 0.5) / 45 * gh;
          var c = s.hit ? C.blue : C.red;
          ctx.strokeStyle = c; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(X(Math.max(lo, s.lo)), y); ctx.lineTo(X(Math.min(hi, s.hi)), y); ctx.stroke();
          ctx.fillStyle = c; ctx.beginPath(); ctx.arc(X(s.ph), y, 2, 0, 7); ctx.fill();
        }
        // x ticks
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        for (var t = 0; t <= 4; t += 1) { var v = lo + (hi - lo) * t / 4; ctx.fillText(v.toFixed(2), X(v), gy + gh + 16); }
        ctx.fillText("доля и её интервал", gx + gw / 2, gy + gh + 32);

        // coverage panel
        var px = 660;
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("накрыли истину:", px, gy + 30);
        var cov = runs > 0 ? covered / runs : 0;
        ctx.font = "34px PT Sans, sans-serif"; ctx.fillStyle = Math.abs(cov - parseInt(level) / 100) < 0.03 ? C.green : C.gold;
        ctx.fillText(runs > 0 ? (cov * 100).toFixed(1) + "%" : "—", px, gy + 72);
        ctx.font = "12px PT Sans, sans-serif"; ctx.fillStyle = C.muted;
        ctx.fillText(covered + " из " + runs + " опросов", px, gy + 94);
        ctx.fillText("обещано: " + level, px, gy + 112);
        // bar
        var by = gy + 140, bw = 180, bh = 18;
        ctx.strokeStyle = C.line; ctx.strokeRect(px, by, bw, bh);
        ctx.fillStyle = C.blue; ctx.fillRect(px, by, bw * cov, bh);
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.4;
        var nx = px + bw * parseInt(level) / 100;
        ctx.beginPath(); ctx.moveTo(nx, by - 4); ctx.lineTo(nx, by + bh + 4); ctx.stroke();
        ctx.fillStyle = C.red; ctx.font = "10px PT Sans, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("цель", nx, by + bh + 16);

        output.set([
          { label: "Опросов проведено", value: String(runs), color: C.ink },
          { label: "Фактическое покрытие", value: runs > 0 ? (cov * 100).toFixed(1) + "%" : "—", color: Math.abs(cov - parseInt(level) / 100) < 0.03 ? C.green : C.gold },
          { label: "Обещанный уровень", value: level, color: C.red },
        ]);
      }

      K.slider(controls, { label: "Истинная доля p", min: 0.02, max: 0.98, step: 0.01, value: 0.6, format: function (v) { return v.toFixed(2); } }, function (v) { truep = v; reset(); draw(); });
      K.slider(controls, { label: "Размер опроса n", min: 10, max: 500, step: 5, value: 100, format: function (v) { return String(v); } }, function (v) { n = v; reset(); draw(); });
      K.segmented(controls, { label: "Уровень доверия", value: 1, options: [
        { label: "90%", value: 0 }, { label: "95%", value: 1 }, { label: "99%", value: 2 } ] }, function (v) {
        z = [1.645, 1.96, 2.576][v]; level = ["90%", "95%", "99%"][v]; reset(); draw();
      });
      var b1 = K.element("button", "kontur-int-segment", { type: "button", text: "провести 100 опросов" });
      var b2 = K.element("button", "kontur-int-segment", { type: "button", text: "сброс" });
      b1.style.margin = b2.style.margin = "0 6px";
      b1.addEventListener("click", function () { run(100); });
      b2.addEventListener("click", function () { reset(); draw(); });
      controls.appendChild(b1); controls.appendChild(b2);

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
