// Lesson 38: pca-lab — rotate a projection axis over real iris data; centering toggle actually re-centers.
(function () {
  "use strict";

  var DATA = {"pts":[[1.4,0.2],[1.4,0.2],[1.3,0.2],[1.5,0.2],[1.4,0.2],[1.7,0.4],[1.4,0.3],[1.5,0.2],[1.4,0.2],[1.5,0.1],[1.5,0.2],[1.6,0.2],[1.4,0.1],[1.1,0.1],[1.2,0.2],[1.5,0.4],[1.3,0.4],[1.4,0.3],[1.7,0.3],[1.5,0.3],[1.7,0.2],[1.5,0.4],[1.0,0.2],[1.7,0.5],[1.9,0.2],[1.6,0.2],[1.6,0.4],[1.5,0.2],[1.4,0.2],[1.6,0.2],[1.6,0.2],[1.5,0.4],[1.5,0.1],[1.4,0.2],[1.5,0.2],[1.2,0.2],[1.3,0.2],[1.4,0.1],[1.3,0.2],[1.5,0.2],[1.3,0.3],[1.3,0.3],[1.3,0.2],[1.6,0.6],[1.9,0.4],[1.4,0.3],[1.6,0.2],[1.4,0.2],[1.5,0.2],[1.4,0.2],[4.7,1.4],[4.5,1.5],[4.9,1.5],[4.0,1.3],[4.6,1.5],[4.5,1.3],[4.7,1.6],[3.3,1.0],[4.6,1.3],[3.9,1.4],[3.5,1.0],[4.2,1.5],[4.0,1.0],[4.7,1.4],[3.6,1.3],[4.4,1.4],[4.5,1.5],[4.1,1.0],[4.5,1.5],[3.9,1.1],[4.8,1.8],[4.0,1.3],[4.9,1.5],[4.7,1.2],[4.3,1.3],[4.4,1.4],[4.8,1.4],[5.0,1.7],[4.5,1.5],[3.5,1.0],[3.8,1.1],[3.7,1.0],[3.9,1.2],[5.1,1.6],[4.5,1.5],[4.5,1.6],[4.7,1.5],[4.4,1.3],[4.1,1.3],[4.0,1.3],[4.4,1.2],[4.6,1.4],[4.0,1.2],[3.3,1.0],[4.2,1.3],[4.2,1.2],[4.2,1.3],[4.3,1.3],[3.0,1.1],[4.1,1.3],[6.0,2.5],[5.1,1.9],[5.9,2.1],[5.6,1.8],[5.8,2.2],[6.6,2.1],[4.5,1.7],[6.3,1.8],[5.8,1.8],[6.1,2.5],[5.1,2.0],[5.3,1.9],[5.5,2.1],[5.0,2.0],[5.1,2.4],[5.3,2.3],[5.5,1.8],[6.7,2.2],[6.9,2.3],[5.0,1.5],[5.7,2.3],[4.9,2.0],[6.7,2.0],[4.9,1.8],[5.7,2.1],[6.0,1.8],[4.8,1.8],[4.9,1.8],[5.6,2.1],[5.8,1.6],[6.1,1.9],[6.4,2.0],[5.6,2.2],[5.1,1.5],[5.6,1.4],[6.1,2.3],[5.6,2.4],[5.5,1.8],[4.8,1.8],[5.4,2.1],[5.6,2.4],[5.1,2.3],[5.1,1.9],[5.9,2.3],[5.7,2.5],[5.2,2.3],[5.0,1.9],[5.2,2.0],[5.4,2.3],[5.1,1.8]],"y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],"mean":[3.758,1.199]};

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("pca-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 440;
      var OX = 250, OY = 250, S = 46;   // math->screen (petal cm), y up
      var angle = 0.4;                  // radians of projection axis
      var centered = true;
      var CLS = [C.blue, C.green, C.gold];

      // precompute covariance eigenstuff on centered data (for the max marker)
      var mean = DATA.mean;
      function centeredPts() {
        return DATA.pts.map(function (p) { return centered ? [p[0] - mean[0], p[1] - mean[1]] : [p[0], p[1]]; });
      }
      function projVariance(a) {
        // mean square of projections on the CURRENT points: equals variance only if
        // the points were centred. Without centring it is the second moment about the
        // origin, which is maximised by pointing at the cloud, not along its spread.
        var ux = Math.cos(a), uy = Math.sin(a), pts = centeredPts();
        var s2 = 0;
        for (var i = 0; i < pts.length; i += 1) { var z = pts[i][0] * ux + pts[i][1] * uy; s2 += z * z; }
        return s2 / pts.length;
      }
      // find max variance angle by scan (on centered data always, for reference marker)
      function bestAngle() {
        var save = centered; centered = true; var best = 0, bv = -1;
        for (var d = 0; d < 180; d += 1) { var a = d * Math.PI / 180; var v = projVariance(a); if (v > bv) { bv = v; best = a; } }
        centered = save; return { a: best, v: bv };
      }
      var BEST = bestAngle();

      K.hint(
        root,
        "Настоящие ирисы: длина и ширина лепестка. Вращайте ось проекции и следите за дисперсией спроецированных точек — она максимальна вдоль вытянутой оси облака, это и есть первая главная компонента. Переключатель центрирования по-настоящему пересчитывает данные: без вычитания среднего ось уводит к началу координат.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Дисперсия проекции равна u^T S u, где S — ковариация. Максимум по направлению u достигается на собственном векторе S с наибольшим собственным значением — это первая главная компонента. Без центрирования S считается вокруг нуля, и ось теряет смысл разброса.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Проекция облака ирисов на ось", onResize: draw, drag: false });

      function m2s(x, y) { return [OX + x * S, OY - y * S]; }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var pts = centeredPts();
        var ux = Math.cos(angle), uy = Math.sin(angle);

        // axes
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var xa = m2s(-2, 0), xb = m2s(6, 0), ya = m2s(0, -2), yb = m2s(0, 4);
        if (!centered) { xa = m2s(-1, 0); xb = m2s(8, 0); ya = m2s(0, -1); yb = m2s(0, 4); }
        ctx.beginPath(); ctx.moveTo(xa[0], xa[1]); ctx.lineTo(xb[0], xb[1]); ctx.moveTo(ya[0], ya[1]); ctx.lineTo(yb[0], yb[1]); ctx.stroke();

        // projection axis (through origin)
        var far = 7;
        var e0 = m2s(-far * ux, -far * uy), e1 = m2s(far * ux, far * uy);
        ctx.strokeStyle = C.red; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(e0[0], e0[1]); ctx.lineTo(e1[0], e1[1]); ctx.stroke();

        // points + projection feet
        for (var i = 0; i < pts.length; i += 1) {
          var p = pts[i], sp = m2s(p[0], p[1]);
          var t = p[0] * ux + p[1] * uy;      // projection scalar
          var fp = m2s(t * ux, t * uy);
          ctx.strokeStyle = "rgba(110,114,106,0.25)"; ctx.lineWidth = 0.6;
          ctx.beginPath(); ctx.moveTo(sp[0], sp[1]); ctx.lineTo(fp[0], fp[1]); ctx.stroke();
          ctx.fillStyle = CLS[DATA.y[i]]; ctx.globalAlpha = 0.75;
          ctx.beginPath(); ctx.arc(sp[0], sp[1], 3, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
        }
        // projection feet (dark)
        for (var j = 0; j < pts.length; j += 1) {
          var pj = pts[j], tj = pj[0] * ux + pj[1] * uy, f = m2s(tj * ux, tj * uy);
          ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(f[0], f[1], 1.8, 0, 7); ctx.fill();
        }
        // labels
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("длина лепестка" + (centered ? " − среднее" : "") + ", см", xb[0] - 150, xb[1] + 20);
        ctx.save(); ctx.translate(yb[0] - 16, yb[1] + 12); ctx.fillText("ширина лепестка, см", 0, 0); ctx.restore();

        var v = projVariance(angle);
        var atMax = Math.abs(v - BEST.v) < 0.02 * BEST.v && centered;
        // mini variance gauge (right side)
        var gx = 620, gy = 70, gw = 210, gh = 260;
        ctx.strokeStyle = C.line; ctx.strokeRect(gx, gy, gw, gh);
        ctx.fillStyle = C.ink; ctx.textAlign = "center"; ctx.fillText("дисперсия проекции", gx + gw / 2, gy - 10);
        var frac = Math.min(1, v / (BEST.v * 1.05));
        ctx.fillStyle = atMax ? C.green : C.blue;
        ctx.fillRect(gx + 40, gy + gh - 20 - frac * (gh - 40), gw - 80, frac * (gh - 40));
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("макс = λ₁ = " + BEST.v.toFixed(2), gx + gw / 2, gy + 18);
        ctx.fillStyle = C.ink; ctx.fillText(v.toFixed(2), gx + gw / 2, gy + gh - 6);

        output.set([
          { label: "Угол оси", value: Math.round(angle * 180 / Math.PI) + "°", color: C.ink },
          { label: centered ? "Дисперсия проекции" : "Средний квадрат (не дисперсия!)", value: v.toFixed(2), color: atMax ? C.green : C.blue },
          { label: centered ? "Центрировано" : "БЕЗ центрирования", value: atMax ? "это первая компонента" : (centered ? "поверните к максимуму" : "ось тянется к облаку, не вдоль разброса"), color: atMax ? C.green : (centered ? C.gold : C.red) },
        ]);
      }

      K.slider(controls, { label: "Угол оси проекции", min: 0, max: 180, step: 1, value: Math.round(angle * 180 / Math.PI), format: function (v) { return v + "°"; } }, function (val) { angle = val * Math.PI / 180; draw(); });
      K.segmented(controls, { label: "Центрирование", value: 1, options: [ { label: "без центра", value: 0 }, { label: "центрировать", value: 1 } ] }, function (val) { centered = val === 1; draw(); });
      var toMax = K.element("button", "kontur-int-segment", { type: "button", text: "К максимуму" });
      toMax.style.margin = "0 8px";
      toMax.addEventListener("click", function () { centered = true; angle = BEST.a; draw(); });
      controls.appendChild(toMax);

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
