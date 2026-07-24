// Lesson 52: bayes-regression-lab — real evening bike counts, a normal prior on the two
// weights, and the exact posterior. Move the prior width, the noise level and the number of
// observed evenings; watch the fan of lines, the band for the mean and the band for a new
// evening. Click on the canvas to add your own evening and see the fan collapse there.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("bayes-regression-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 500;
      var OX = 78, OY = 430, TOP = 40;
      var XMAX = 1.0, YMAX = 900;
      var PB = "#fffef9";
      var CENTER = 0.5;

      // real evenings: 17:00, working days of 2011 (temp, cnt) from bike-sharing-hour.csv
      var REAL = [
        [0.20, 139], [0.74, 601], [0.46, 255],
        [0.90, 556], [0.32, 206], [0.72, 569],
        [0.36, 177], [0.76, 557], [0.84, 468],
        [0.56, 568], [0.72, 590], [0.46, 262]
      ];
      var nData = 6, tau = 300, sigma = 111, mode = 0, extra = [];

      // fixed standard normal pairs (Box-Muller over a small LCG) so the fan does not flicker
      var Z = (function () {
        var s = 20520, out = [], i;
        function rnd() { s = (1103515245 * s + 12345) % 2147483648; return (s + 1) / 2147483649; }
        for (i = 0; i < 40; i += 1) {
          var u1 = rnd(), u2 = rnd(), r = Math.sqrt(-2 * Math.log(u1));
          out.push([r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)]);
        }
        return out;
      })();

      function points() {
        var p = REAL.slice(0, nData), i;
        for (i = 0; i < extra.length; i += 1) p.push(extra[i]);
        return p;
      }

      // posterior of w=(w0,w1) for phi(x)=(1, x-0.5), prior N(0, tau^2 I), noise sigma^2
      function posterior() {
        var p = points(), a = 1 / (tau * tau), b = 0, c = 1 / (tau * tau), i;
        var r0 = 0, r1 = 0, s2 = sigma * sigma;
        for (i = 0; i < p.length; i += 1) {
          var x = p[i][0] - CENTER, y = p[i][1];
          a += 1 / s2; b += x / s2; c += x * x / s2;
          r0 += y / s2; r1 += x * y / s2;
        }
        var det = a * c - b * b;
        var S = [[c / det, -b / det], [-b / det, a / det]];
        return { m: [S[0][0] * r0 + S[0][1] * r1, S[1][0] * r0 + S[1][1] * r1], S: S };
      }
      function ols() {
        var p = points(), n = p.length, mx = 0, my = 0, i;
        if (n < 2) return null;
        for (i = 0; i < n; i += 1) { mx += p[i][0]; my += p[i][1]; }
        mx /= n; my /= n;
        var sxy = 0, sxx = 0;
        for (i = 0; i < n; i += 1) { sxy += (p[i][0] - mx) * (p[i][1] - my); sxx += (p[i][0] - mx) * (p[i][0] - mx); }
        if (sxx < 1e-9) return null;
        var w1 = sxy / sxx;
        return [my - w1 * (mx - CENTER), w1];
      }
      function varf(x, S) {
        var d = x - CENTER;
        return S[0][0] + 2 * d * S[0][1] + d * d * S[1][1];
      }

      K.hint(root, "Шесть реальных вечеров велопроката: по горизонтали температура, по вертикали число поездок в 17:00. Модель не выбирает одну прямую — она держит целое облако прямых, согласованных и с данными, и с prior. Двигайте ширину prior, уровень шума и число наблюдений; щёлкните по полю, чтобы добавить свой вечер, и посмотрите, где веер схлопывается.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Синяя полоса — 95% для среднего x·w: она узка там, где есть точки, и расходится веером по краям. Золотая полоса шире на постоянную величину σ²: это шум самого мира, который не исчезает никогда. Узкий prior стягивает наклон к нулю, широкий отпускает его к МНК.");
      var cs = K.makeCanvas(stage, W, H, { label: "Веер posterior-прямых, полоса среднего и полоса нового вечера", onResize: draw, drag: false });

      function sx(x) { return OX + x / XMAX * (W - OX - 40); }
      function sy(y) { return OY - y / YMAX * (OY - TOP); }
      function ix(px) { return (px - OX) / (W - OX - 40) * XMAX; }
      function iy(py) { return (OY - py) / (OY - TOP) * YMAX; }

      function draw() {
        var ctx = cs.ctx, i, x, s;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var po = posterior(), m = po.m, S = po.S;
        var p = points(), xmin = 1, xmax = 0;
        for (i = 0; i < p.length; i += 1) { if (p[i][0] < xmin) xmin = p[i][0]; if (p[i][0] > xmax) xmax = p[i][0]; }

        // зона наблюдений
        ctx.fillStyle = "rgba(110,114,106,0.07)";
        ctx.fillRect(sx(xmin), TOP, sx(xmax) - sx(xmin), OY - TOP);

        var N = 120, gx = [], mu = [], sf = [];
        for (i = 0; i <= N; i += 1) {
          x = i / N * XMAX; gx.push(x);
          mu.push(m[0] + m[1] * (x - CENTER));
          sf.push(Math.sqrt(Math.max(0, varf(x, S))));
        }
        function bandPath(mult, noise) {
          ctx.beginPath();
          for (i = 0; i <= N; i += 1) { s = Math.sqrt(sf[i] * sf[i] + noise); ctx.lineTo(sx(gx[i]), sy(mu[i] + mult * s)); }
          for (i = N; i >= 0; i -= 1) { s = Math.sqrt(sf[i] * sf[i] + noise); ctx.lineTo(sx(gx[i]), sy(mu[i] - mult * s)); }
          ctx.closePath(); ctx.fill();
        }
        ctx.fillStyle = "rgba(165,121,32,0.18)"; bandPath(1.96, sigma * sigma);
        ctx.fillStyle = "rgba(49,95,140,0.22)"; bandPath(1.96, 0);

        if (mode === 1) {
          // линии из posterior: w = m + L z, L — Холецкий S
          var l11 = Math.sqrt(Math.max(1e-12, S[0][0]));
          var l21 = S[0][1] / l11;
          var l22 = Math.sqrt(Math.max(1e-12, S[1][1] - l21 * l21));
          ctx.strokeStyle = "rgba(49,95,140,0.30)"; ctx.lineWidth = 1;
          for (i = 0; i < Z.length; i += 1) {
            var w0 = m[0] + l11 * Z[i][0], w1 = m[1] + l21 * Z[i][0] + l22 * Z[i][1];
            ctx.beginPath();
            ctx.moveTo(sx(0), sy(w0 + w1 * (0 - CENTER)));
            ctx.lineTo(sx(XMAX), sy(w0 + w1 * (XMAX - CENTER)));
            ctx.stroke();
          }
        }

        // оси
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(OX, OY); ctx.lineTo(sx(XMAX) + 12, OY);
        ctx.moveTo(OX, OY); ctx.lineTo(OX, TOP); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (i = 0; i <= 5; i += 1) { x = i / 5; ctx.fillText(x.toFixed(1), sx(x), OY + 18); }
        ctx.textAlign = "right";
        for (i = 0; i <= 3; i += 1) { var yv = i * 300; ctx.fillText(String(yv), OX - 8, sy(yv) + 4); }
        ctx.textAlign = "center";
        ctx.fillText("температура (temp)", (OX + sx(XMAX)) / 2, OY + 38);
        ctx.save(); ctx.translate(20, (TOP + OY) / 2); ctx.rotate(-Math.PI / 2);
        ctx.fillText("поездок в 17:00", 0, 0); ctx.restore();

        // МНК для сравнения
        var o = ols();
        if (o) {
          ctx.strokeStyle = C.red; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]);
          ctx.beginPath();
          ctx.moveTo(sx(0), sy(o[0] + o[1] * (0 - CENTER)));
          ctx.lineTo(sx(XMAX), sy(o[0] + o[1] * (XMAX - CENTER)));
          ctx.stroke(); ctx.setLineDash([]);
        }
        // posterior mean
        ctx.strokeStyle = C.blue; ctx.lineWidth = 2.8;
        ctx.beginPath();
        ctx.moveTo(sx(0), sy(mu[0])); ctx.lineTo(sx(XMAX), sy(mu[N]));
        ctx.stroke();

        for (i = 0; i < p.length; i += 1) {
          ctx.fillStyle = i < nData ? C.ink || "#171915" : C.green;
          ctx.beginPath(); ctx.arc(sx(p[i][0]), sy(p[i][1]), 6, 0, 7); ctx.fill();
          ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.stroke();
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("щёлкните, чтобы добавить вечер", OX + 8, TOP + 14);

        var sdSlope = Math.sqrt(S[1][1]);
        var half05 = 1.96 * Math.sqrt(varf(0.5, S));
        var half10 = 1.96 * Math.sqrt(varf(1.0, S));
        output.set([
          { label: "Наклон: posterior mean ± sd", value: m[1].toFixed(0) + " ± " + sdSlope.toFixed(0), color: C.blue },
          { label: "Наклон по МНК", value: o ? o[1].toFixed(0) : "не определён", color: C.red },
          { label: "Полоса среднего при temp=0,5", value: "±" + half05.toFixed(0) + " поездок", color: C.blue },
          { label: "Полоса среднего при temp=1,0", value: "±" + half10.toFixed(0) + " поездок", color: C.gold },
          { label: "Полоса нового вечера при temp=1,0", value: "±" + (1.96 * Math.sqrt(varf(1.0, S) + sigma * sigma)).toFixed(0), color: C.gold }
        ]);
      }

      cs.canvas.addEventListener("click", function (ev) {
        var rect = cs.canvas.getBoundingClientRect();
        var px = (ev.clientX - rect.left) / rect.width * W;
        var py = (ev.clientY - rect.top) / rect.height * H;
        var x = ix(px), y = iy(py);
        if (x < 0 || x > XMAX || y < 0 || y > YMAX) return;
        extra.push([Math.round(x * 100) / 100, Math.round(y)]);
        draw();
      });

      K.slider(controls, {
        label: "Ширина prior τ (поездок)", min: 25, max: 800, step: 25, value: 300,
        format: function (v) { return String(v); }
      }, function (v) { tau = v; draw(); });
      K.slider(controls, {
        label: "Шум σ (поездок)", min: 30, max: 260, step: 5, value: 111,
        format: function (v) { return String(v); }
      }, function (v) { sigma = v; draw(); });
      K.segmented(controls, {
        label: "Наблюдённых вечеров", value: 6,
        options: [{ label: "2", value: 2 }, { label: "3", value: 3 }, { label: "6", value: 6 }, { label: "12", value: 12 }]
      }, function (v) { nData = v; draw(); });
      K.segmented(controls, {
        label: "Показывать", value: 0,
        options: [{ label: "полосы", value: 0 }, { label: "полосы и линии", value: 1 }]
      }, function (v) { mode = v; draw(); });
      var reset = K.element("button", "kontur-int-segment", { type: "button", text: "убрать мои точки" });
      reset.style.margin = "0 6px";
      reset.addEventListener("click", function () { extra = []; draw(); });
      controls.appendChild(reset);

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
