// Lesson 49: ols-lab — drag points, watch the least-squares line and residuals; feel leverage.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("ols-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 460;
      var OX = 70, OY = 400, S = 34;
      var PB = "#fffef9";
      var pts, drag = null, showSq = false;

      function reset() {
        pts = [];
        var xs = [1, 2, 2.6, 3.4, 4, 5, 5.6, 6.4, 7, 8];
        var nz = [0.6, -0.4, 0.9, -0.7, 0.3, -0.5, 0.8, -0.2, 0.5, -0.6];
        for (var i = 0; i < xs.length; i += 1) pts.push({ x: xs[i], y: 1.2 + 0.85 * xs[i] + nz[i] });
        pts.push({ x: 9.2, y: 8.5 });
      }
      function fit() {
        var n = pts.length, mx = 0, my = 0, i;
        for (i = 0; i < n; i += 1) { mx += pts[i].x; my += pts[i].y; }
        mx /= n; my /= n;
        var sxy = 0, sxx = 0, syy = 0;
        for (i = 0; i < n; i += 1) { sxy += (pts[i].x - mx) * (pts[i].y - my); sxx += (pts[i].x - mx) * (pts[i].x - mx); syy += (pts[i].y - my) * (pts[i].y - my); }
        var w1 = sxx > 1e-9 ? sxy / sxx : 0, w0 = my - w1 * mx;
        var ssr = 0; for (i = 0; i < n; i += 1) { var e = pts[i].y - (w0 + w1 * pts[i].x); ssr += e * e; }
        var r2 = syy > 1e-9 ? 1 - ssr / syy : 0;
        return { w0: w0, w1: w1, r2: r2, rmse: Math.sqrt(ssr / n) };
      }
      K.hint(root, "Линия наименьших квадратов минимизирует сумму квадратов вертикальных остатков. Перетаскивайте точки и следите, как линия подстраивается. Особенно поучительна крайняя правая точка: у неё большой рычаг (leverage), и даже небольшой её сдвиг заметно поворачивает всю прямую.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Красная линия минимизирует сумму квадратов серых остатков. Наклон — прирост ответа на единицу x; R² — доля разброса ответа, объяснённая линией. Точки с необычным x имеют высокий leverage: тянут линию сильнее.");
      var cs = K.makeCanvas(stage, W, H, { label: "Точки, линия наименьших квадратов и остатки", onResize: draw, drag: false });
      reset();
      function m2s(x, y) { return [OX + x * S, OY - y * S]; }
      function s2m(px, py) { return [(px - OX) / S, (OY - py) / S]; }
      function draw() {
        var ctx = cs.ctx; ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var f = fit();
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var o = m2s(0, 0), xe = m2s(10.5, 0), ye = m2s(0, 10.5);
        ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(xe[0], xe[1]); ctx.moveTo(o[0], o[1]); ctx.lineTo(ye[0], ye[1]); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var t = 2; t <= 10; t += 2) { var p = m2s(t, 0); ctx.fillText(String(t), p[0], p[1] + 16); var q = m2s(0, t); ctx.textAlign = "right"; ctx.fillText(String(t), q[0] - 8, q[1] + 4); ctx.textAlign = "center"; }
        ctx.fillText("x", xe[0] + 10, xe[1] + 4); ctx.fillText("y", ye[0], ye[1] - 8);
        for (var i = 0; i < pts.length; i += 1) {
          var pm = pts[i], sp = m2s(pm.x, pm.y), yl = f.w0 + f.w1 * pm.x, fp = m2s(pm.x, yl);
          ctx.strokeStyle = "rgba(110,114,106,0.6)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(sp[0], sp[1]); ctx.lineTo(fp[0], fp[1]); ctx.stroke();
          if (showSq) { var side = Math.abs(sp[1] - fp[1]); ctx.fillStyle = "rgba(185,74,59,0.10)"; ctx.fillRect(sp[0], Math.min(sp[1], fp[1]), side, side); }
        }
        var a = m2s(0, f.w0), b = m2s(10.5, f.w0 + f.w1 * 10.5);
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.6; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        for (var k = 0; k < pts.length; k += 1) {
          var s = m2s(pts[k].x, pts[k].y), lever = pts[k].x > 8.5;
          ctx.fillStyle = lever ? C.gold : C.blue; ctx.beginPath(); ctx.arc(s[0], s[1], lever ? 8 : 6, 0, 7); ctx.fill();
          ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.stroke();
        }
        ctx.fillStyle = C.gold; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        var lv = m2s(9.2, 0); ctx.fillText("высокий leverage — тащи по вертикали", lv[0] + 20, lv[1] + 34);
        output.set([
          { label: "Линия", value: "y = " + f.w0.toFixed(2) + " + " + f.w1.toFixed(2) + "·x", color: C.red },
          { label: "R² (объяснённая доля)", value: f.r2.toFixed(3), color: C.blue },
          { label: "RMSE (типичный остаток)", value: f.rmse.toFixed(2), color: C.gold }
        ]);
      }
      cs.canvas.addEventListener("mousedown", function (ev) {
        var rect = cs.canvas.getBoundingClientRect(), mx = (ev.clientX - rect.left) / rect.width * W, my = (ev.clientY - rect.top) / rect.height * H;
        for (var i = 0; i < pts.length; i += 1) { var s = m2s(pts[i].x, pts[i].y); if (Math.hypot(mx - s[0], my - s[1]) < 16) { drag = i; return; } }
      });
      window.addEventListener("mousemove", function (ev) {
        if (drag === null) return;
        var rect = cs.canvas.getBoundingClientRect(), mx = (ev.clientX - rect.left) / rect.width * W, my = (ev.clientY - rect.top) / rect.height * H, m = s2m(mx, my);
        pts[drag].y = Math.max(0, Math.min(10, Math.round(m[1] * 20) / 20));
        pts[drag].x = Math.max(0.3, Math.min(10, Math.round(m[0] * 20) / 20));
        draw();
      });
      window.addEventListener("mouseup", function () { drag = null; });
      var b1 = K.element("button", "kontur-int-segment", { type: "button", text: "показать квадраты" });
      var b2 = K.element("button", "kontur-int-segment", { type: "button", text: "сброс" });
      b1.style.margin = b2.style.margin = "0 6px";
      b1.addEventListener("click", function () { showSq = !showSq; b1.textContent = showSq ? "скрыть квадраты" : "показать квадраты"; draw(); });
      b2.addEventListener("click", function () { reset(); draw(); });
      controls.appendChild(b1); controls.appendChild(b2);
      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
