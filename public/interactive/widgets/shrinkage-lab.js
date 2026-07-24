// Lesson 51: shrinkage-lab.
// Scene A — the real diabetes table (442x10, z-scored): move the penalty and watch the ten
// coefficients shrink; ridge slides them smoothly, lasso snaps them to exact zero. The pair
// s1/s2 is shown separately, because that is where the penalty does its visible work.
// Scene B — the real bike-sharing evening hours: drag one broken row and compare the
// least-squares line with the Huber line.
//
// Scene A carries no raw table: everything the two solvers need is the Gram matrix of the
// z-scored features and X^T y, computed once in scripts/generate_lesson51_visuals.py from
// load_diabetes(scaled=False) + StandardScaler. Standardization is therefore baked in, and
// alpha = 0 reproduces the figures exactly: w(s1) = -37.7, w(s2) = +22.7.
(function () {
  "use strict";

  var NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"];
  var N = 442;
  var SYY = 2621009.12;
  var G = [
    [442.0000, 76.7918, 81.8074, 148.2590, 114.9469, 96.9055, -33.2300, 90.0977, 119.6822, 133.3651],
    [76.7918, 442.0000, 38.9673, 106.5266, 15.5924, 63.0457, -167.5576, 146.7949, 66.2629, 91.9949],
    [81.8074, 38.9673, 442.0000, 174.7716, 110.4016, 115.4371, -162.1305, 182.9025, 197.2012, 171.7966],
    [148.2590, 106.5266, 174.7716, 442.0000, 107.1691, 82.0124, -79.0126, 113.8813, 173.9182, 172.5701],
    [114.9469, 15.5924, 110.4016, 107.1691, 442.0000, 396.3250, 22.7716, 239.6556, 227.8523, 143.9668],
    [96.9055, 63.0457, 115.4371, 82.0124, 396.3250, 442.0000, -86.8332, 291.6391, 140.7136, 128.4454],
    [-33.2300, -167.5576, -162.1305, -79.0126, 22.7716, -86.8332, 442.0000, -326.4138, -176.1712, -120.9742],
    [90.0977, 146.7949, 182.9025, 113.8813, 239.6556, 291.6391, -326.4138, 442.0000, 273.0937, 184.4078],
    [119.6822, 66.2629, 197.2012, 173.9182, 227.8523, 140.7136, -176.1712, 273.0937, 442.0000, 205.3836],
    [133.3651, 91.9949, 171.7966, 172.5701, 143.9668, 128.4454, -120.9742, 184.4078, 205.3836, 442.0000]
  ];
  var XTY = [6395.083, 1465.681, 19960.733, 15026.511, 7216.512, 5924.182, -13437.260, 14651.127, 19260.685, 13018.414];

  // Real evening hours (17:00) of the bike-sharing table: temperature in Celsius, rides per hour.
  var BIKE = [[-0.48, 136], [2.34, 159], [6.1, 222], [1.4, 166], [13.62, 255], [8.92, 248], [12.68, 111], [19.26, 428], [10.8, 410], [19.26, 354], [19.26, 274], [26.78, 341], [26.78, 383], [31.48, 495], [26.78, 638], [29.6, 529], [31.48, 572], [24.9, 425], [29.6, 532], [29.6, 383], [25.84, 611], [16.44, 566], [16.44, 359], [18.32, 325], [23.02, 563], [20.2, 99], [18.32, 585], [23.02, 459], [21.14, 479], [18.32, 620], [20.2, 625], [11.74, 480], [8.92, 177], [15.5, 451], [9.86, 163], [13.62, 262], [10.8, 431], [6.1, 240], [1.4, 281], [7.98, 289], [18.32, 600], [18.32, 612], [20.2, 732], [23.02, 852], [21.14, 724], [25.84, 869], [20.2, 723], [31.48, 747], [35.24, 689], [37.12, 279], [27.72, 795], [23.02, 715], [26.78, 812], [23.96, 846], [23.02, 967], [6.1, 124], [7.98, 638], [14.56, 300], [6.1, 527], [14.56, 616]];

  // ------------------------------------------------------------------ solvers
  function solveRidge(alpha) {
    var n = 10, i, j, k, a = [], w = new Array(n).fill(0);
    for (i = 0; i < n; i += 1) {
      a.push([]);
      for (j = 0; j < n; j += 1) a[i].push(G[i][j] + (i === j ? alpha : 0));
      a[i].push(XTY[i]);
    }
    for (i = 0; i < n; i += 1) {                       // Gaussian elimination with partial pivot
      var piv = i;
      for (k = i + 1; k < n; k += 1) if (Math.abs(a[k][i]) > Math.abs(a[piv][i])) piv = k;
      var tmp = a[i]; a[i] = a[piv]; a[piv] = tmp;
      var d = a[i][i];
      if (Math.abs(d) < 1e-12) continue;
      for (j = i; j <= n; j += 1) a[i][j] /= d;
      for (k = 0; k < n; k += 1) {
        if (k === i) continue;
        var f = a[k][i];
        if (f === 0) continue;
        for (j = i; j <= n; j += 1) a[k][j] -= f * a[i][j];
      }
    }
    for (i = 0; i < n; i += 1) w[i] = a[i][n];
    return w;
  }

  function solveLasso(alpha) {
    // Coordinate descent with soft-thresholding on the same objective sklearn uses:
    // (1/2n)||y - Xw||^2 + alpha*||w||_1  =>  w_j = soft(X_j^T r, n*alpha) / G_jj.
    var n = 10, w = new Array(n).fill(0), thr = N * alpha, pass, j, k;
    for (pass = 0; pass < 400; pass += 1) {
      var moved = 0;
      for (j = 0; j < n; j += 1) {
        var rho = XTY[j];
        for (k = 0; k < n; k += 1) if (k !== j) rho -= G[j][k] * w[k];
        var next = 0;
        if (rho > thr) next = (rho - thr) / G[j][j];
        else if (rho < -thr) next = (rho + thr) / G[j][j];
        moved = Math.max(moved, Math.abs(next - w[j]));
        w[j] = next;
      }
      if (moved < 1e-9) break;
    }
    return w;
  }

  function quality(w) {
    var i, j, quad = 0, lin = 0, l1 = 0, l2 = 0, nnz = 0;
    for (i = 0; i < 10; i += 1) {
      lin += w[i] * XTY[i];
      l1 += Math.abs(w[i]);
      l2 += w[i] * w[i];
      if (Math.abs(w[i]) > 1e-8) nnz += 1;
      for (j = 0; j < 10; j += 1) quad += w[i] * G[i][j] * w[j];
    }
    var rss = SYY - 2 * lin + quad;
    return { r2: 1 - rss / SYY, l1: l1, norm: Math.sqrt(l2), nnz: nnz };
  }

  function ols2(pts) {
    var n = pts.length, i, mx = 0, my = 0;
    for (i = 0; i < n; i += 1) { mx += pts[i][0]; my += pts[i][1]; }
    mx /= n; my /= n;
    var sxy = 0, sxx = 0;
    for (i = 0; i < n; i += 1) { sxy += (pts[i][0] - mx) * (pts[i][1] - my); sxx += (pts[i][0] - mx) * (pts[i][0] - mx); }
    var b = sxx > 1e-9 ? sxy / sxx : 0;
    return [my - b * mx, b];
  }

  function median(values) {
    var s = values.slice().sort(function (p, q) { return p - q; });
    var m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function huber2(pts, deltaInSigma) {
    var fit = ols2(pts), i, pass;
    for (pass = 0; pass < 60; pass += 1) {
      var res = pts.map(function (p) { return p[1] - (fit[0] + fit[1] * p[0]); });
      var med = median(res);
      var sigma = 1.4826 * median(res.map(function (r) { return Math.abs(r - med); })) || 1;
      var delta = deltaInSigma * sigma;
      var sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
      for (i = 0; i < pts.length; i += 1) {
        var a = Math.abs(res[i]);
        var wt = a <= delta ? 1 : delta / a;
        sw += wt; swx += wt * pts[i][0]; swy += wt * pts[i][1];
        swxx += wt * pts[i][0] * pts[i][0]; swxy += wt * pts[i][0] * pts[i][1];
      }
      var den = sw * swxx - swx * swx;
      if (Math.abs(den) < 1e-9) break;
      var slope = (sw * swxy - swx * swy) / den;
      var inter = (swy - slope * swx) / sw;
      var step = Math.abs(slope - fit[1]) + Math.abs(inter - fit[0]);
      fit = [inter, slope];
      if (step < 1e-9) break;
    }
    return fit;
  }

  function ru(value, digits) {
    return value.toFixed(digits == null ? 2 : digits).replace(".", ",");
  }

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("shrinkage-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 430;
      var scene = "path";
      var mode = "ridge", logAlpha = 1;            // alpha = 10^logAlpha
      var lossMode = "huber", deltaSigma = 1.35;
      var OUT_INDEX = 1;                            // the row [2.34; 159] is the one you break
      var REST = BIKE.filter(function (p, i) { return i !== OUT_INDEX; });
      var outlier = [BIKE[OUT_INDEX][0], BIKE[OUT_INDEX][1] + 2000];
      var dragging = false;

      K.hint(root, "Слева — реальная таблица диабета (442 больных, 10 z-стандартизованных признаков): двигайте штраф и смотрите, как ridge плавно сжимает все веса, а lasso зануляет их по одному. Переключитесь на вторую сцену и потащите сломанную строку вверх: квадратичная ошибка побежит за ней, Хьюбер — нет.");

      var sceneRow = K.row(root, "controls");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      var cap = K.caption(root, "");

      var cs = K.makeCanvas(stage, W, H, {
        label: "Коэффициенты при разном штрафе и робастная подгонка прямой",
        onResize: draw,
        drag: true
      });

      K.segmented(sceneRow, {
        label: "сцена",
        value: scene,
        options: [
          { value: "path", label: "штраф и коэффициенты" },
          { value: "robust", label: "выброс и робастность" }
        ]
      }, function (next) { scene = next; syncControls(); draw(); });

      var pathBox = K.element("div", "kontur-int-controls-group");
      var robustBox = K.element("div", "kontur-int-controls-group");
      controls.appendChild(pathBox);
      controls.appendChild(robustBox);

      K.segmented(pathBox, {
        label: "штраф",
        value: mode,
        options: [{ value: "ridge", label: "ridge (L2)" }, { value: "lasso", label: "lasso (L1)" }]
      }, function (next) { mode = next; draw(); });

      K.slider(pathBox, {
        label: "log₁₀ α", min: -2, max: 3.5, step: 0.05, value: logAlpha,
        format: function (v) { return ru(Math.pow(10, v), Math.pow(10, v) < 1 ? 2 : 1); },
        unit: ""
      }, function (v) { logAlpha = v; draw(); });

      K.segmented(robustBox, {
        label: "функция потерь",
        value: lossMode,
        options: [{ value: "mse", label: "квадрат" }, { value: "huber", label: "Хьюбер" }]
      }, function (next) { lossMode = next; draw(); });

      K.slider(robustBox, {
        label: "порог δ", min: 0.3, max: 4, step: 0.05, value: deltaSigma,
        format: function (v) { return ru(v, 2); }, unit: " σ̂"
      }, function (v) { deltaSigma = v; draw(); });

      function syncControls() {
        pathBox.style.display = scene === "path" ? "" : "none";
        robustBox.style.display = scene === "robust" ? "" : "none";
      }
      syncControls();

      // ------------------------------------------------------------- scene A
      function drawPath(ctx) {
        var alpha = Math.pow(10, logAlpha);
        var w = mode === "ridge" ? solveRidge(alpha) : solveLasso(alpha);
        var q = quality(w);
        // Веса ходят от −37,7 до +34, поэтому ноль держим посередине поля:
        // иначе отрицательный столбик s1 уходит под нижнюю границу канвы.
        var left = 64, right = W - 30, top = 40, bottom = H - 34;
        var base = (top + bottom) / 2;
        var span = (right - left) / 10;
        var scale = (bottom - base) / 42;

        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        for (var v = -40; v <= 40; v += 10) {
          var yy = base - v * scale;
          ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
          ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(String(v), left - 8, yy + 4);
        }
        ctx.strokeStyle = C.axis; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(left, base); ctx.lineTo(right, base); ctx.stroke();

        for (var j = 0; j < 10; j += 1) {
          var cx = left + span * (j + 0.5);
          var zero = Math.abs(w[j]) < 1e-8;
          var pair = j === 4 || j === 5;
          var height = w[j] * scale;
          ctx.fillStyle = zero ? "rgba(110,114,106,0.20)" : (pair ? C.violet : C.blue);
          ctx.fillRect(cx - span * 0.3, base - Math.max(height, 0), span * 0.6, Math.abs(height) || 2);
          ctx.fillStyle = zero ? C.muted : (pair ? C.violet : C.ink);
          ctx.textAlign = "center"; ctx.font = (pair ? "bold " : "") + "12px PT Sans, sans-serif";
          // Имена признаков — под всей областью, а не у нулевой линии:
          // у нуля они наезжали на значения коротких столбиков.
          ctx.fillText(NAMES[j], cx, H - 12);
          ctx.font = "11px PT Sans, sans-serif";
          ctx.fillStyle = zero ? C.muted : C.muted;
          ctx.fillText(zero ? "0" : ru(w[j], 1), cx, base - height + (height >= 0 ? -8 : 16));
        }

        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText(mode === "ridge"
          ? "ridge: (XᵀX + αI)w = Xᵀy — все веса живы, но короче"
          : "lasso: мягкий порог обнуляет координаты одну за другой", left, 24);
        if (mode === "lasso") {
          ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText("серые столбики — точные нули", right, 24);
        }

        output.set([
          { label: "α", value: ru(alpha, alpha < 1 ? 3 : 1), color: C.gold },
          { label: "вес s1", value: ru(w[4], 2), color: C.violet },
          { label: "вес s2", value: ru(w[5], 2), color: C.violet },
          { label: "ненулевых весов", value: q.nnz + " из 10", color: C.ink },
          { label: "‖w‖₂", value: ru(q.norm, 1), color: C.blue },
          { label: "R² на обучении", value: ru(q.r2, 3), color: C.green }
        ]);
        cap.textContent = "Реальная таблица диабета, признаки z-стандартизованы. При α → 0 повторяются числа урока: s1 = −37,7 и s2 = +22,7. Ridge стягивает пару к нулю согласованно, lasso сначала обнуляет s2 и оставляет s1 представителем группы.";
      }

      // ------------------------------------------------------------- scene B
      function bx(t) { return 70 + (t + 3) / 44 * (W - 110); }
      function by(c) { return H - 60 - c / 2400 * (H - 100); }
      function bxInv(px) { return (px - 70) / (W - 110) * 44 - 3; }
      function byInv(py) { return (H - 60 - py) / (H - 100) * 2400; }

      function drawRobust(ctx) {
        var pts = REST.map(function (p) { return [p[0], p[1]]; });
        pts.push([outlier[0], outlier[1]]);
        var line = ols2(pts);
        var rob = huber2(pts, deltaSigma);
        var clean = ols2(BIKE);

        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        var g, p;
        for (g = 0; g <= 2400; g += 400) {
          ctx.beginPath(); ctx.moveTo(bx(-3), by(g)); ctx.lineTo(bx(41), by(g)); ctx.stroke();
          ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(String(g), bx(-3) - 8, by(g) + 4);
        }
        for (g = 0; g <= 40; g += 10) {
          ctx.fillStyle = C.muted; ctx.textAlign = "center";
          ctx.fillText(g + "°", bx(g), by(0) + 20);
        }
        for (p = 0; p < REST.length; p += 1) {
          ctx.fillStyle = "rgba(23,25,21,0.62)";
          ctx.beginPath(); ctx.arc(bx(REST[p][0]), by(REST[p][1]), 4.5, 0, 7); ctx.fill();
        }

        function stroke(fit, color, width) {
          ctx.strokeStyle = color; ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(bx(-2), by(fit[0] + fit[1] * -2));
          ctx.lineTo(bx(40), by(fit[0] + fit[1] * 40));
          ctx.stroke();
        }
        ctx.setLineDash([6, 4]);
        stroke(clean, C.muted, 1.4);
        ctx.setLineDash([]);
        stroke(line, C.red, lossMode === "mse" ? 3.2 : 1.8);
        stroke(rob, C.green, lossMode === "huber" ? 3.2 : 1.8);

        var ox = bx(outlier[0]), oy = by(outlier[1]);
        ctx.fillStyle = C.gold;
        ctx.beginPath(); ctx.arc(ox, oy, 9, 0, 7); ctx.fill();
        ctx.strokeStyle = C.paper; ctx.lineWidth = 1.6; ctx.stroke();
        ctx.fillStyle = C.gold; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("тяните эту строку", ox + 14, oy + 4);

        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("вечерний час велопроката: температура → поездки", bx(-3), 24);

        var resid = outlier[1] - (line[0] + line[1] * outlier[0]);
        var residRob = outlier[1] - (rob[0] + rob[1] * outlier[0]);
        output.set([
          { label: "МНК: наклон", value: ru(line[1], 1) + " поездки/°C", color: C.red },
          { label: "МНК: свободный член", value: ru(line[0], 0), color: C.red },
          { label: "Хьюбер: наклон", value: ru(rob[1], 1) + " поездки/°C", color: C.green },
          { label: "Хьюбер: свободный член", value: ru(rob[0], 0), color: C.green },
          { label: "остаток выброса (МНК)", value: ru(resid, 0), color: C.gold },
          { label: "остаток выброса (Хьюбер)", value: ru(residRob, 0), color: C.gold }
        ]);
        cap.textContent = "Серая штриховая прямая — МНК по чистым данным (наклон 13,7). Тащите жёлтую точку вверх: красная прямая уходит за ней, зелёная держится, пока δ не станет слишком большим. При δ около 4σ̂ Хьюбер почти совпадает с квадратом.";
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.textBaseline = "alphabetic";
        ctx.font = "12px PT Sans, sans-serif";
        if (scene === "path") drawPath(ctx); else drawRobust(ctx);
      }

      var stopDrag = K.drag(cs.canvas, { w: W, h: H }, {
        down: function (point) {
          if (scene !== "robust") return;
          var dx = point.x - bx(outlier[0]), dy = point.y - by(outlier[1]);
          dragging = Math.hypot(dx, dy) < 26;
        },
        move: function (point) {
          if (!dragging || scene !== "robust") return;
          outlier = [
            Math.max(-1, Math.min(38, Math.round(bxInv(point.x) * 10) / 10)),
            Math.max(0, Math.min(2380, Math.round(byInv(point.y))))
          ];
          draw();
        },
        up: function () { dragging = false; }
      });

      draw();
      return function () { stopDrag(); cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
