// Lesson 77: transformer-block-lab — включайте и выключайте подслои блока и смотрите,
// что происходит с потоком: норма представления, память о своём embedding, влияние первого
// токена на последний. Веса случайные и фиксированные (seeded), обучения нет — измеряется
// именно маршрут сигнала через архитектуру.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("transformer-block-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var N = 8, D = 16, HEADS = 2, EXP = 2, LMAX = 16;

      // ---- детерминированный генератор весов -------------------------------
      function mulberry32(a) {
        return function () {
          a |= 0; a = (a + 0x6D2B79F5) | 0;
          var t = Math.imul(a ^ (a >>> 15), 1 | a);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }
      function gauss(rnd) {
        var u = Math.max(rnd(), 1e-9), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      function mat(rnd, r, c, s) {
        var m = new Array(r), i, j;
        for (i = 0; i < r; i += 1) { m[i] = new Float64Array(c); for (j = 0; j < c; j += 1) m[i][j] = gauss(rnd) * s; }
        return m;
      }
      function mm(A, B) {                       // A: r×k, B: k×c
        var r = A.length, k = B.length, c = B[0].length, out = new Array(r), i, j, t, s;
        for (i = 0; i < r; i += 1) {
          out[i] = new Float64Array(c);
          for (t = 0; t < k; t += 1) {
            s = A[i][t]; if (s === 0) continue;
            for (j = 0; j < c; j += 1) out[i][j] += s * B[t][j];
          }
        }
        return out;
      }
      function clone(A) { return A.map(function (r) { return Float64Array.from(r); }); }

      var rnd = mulberry32(77);
      var X0 = mat(rnd, N, D, 1);
      var UDIR = Float64Array.from(mat(rnd, 1, D, 1)[0]);
      (function () { var nrm = 0, i; for (i = 0; i < D; i += 1) nrm += UDIR[i] * UDIR[i]; nrm = Math.sqrt(nrm); for (i = 0; i < D; i += 1) UDIR[i] /= nrm; })();
      var LAYERS = [];
      for (var l = 0; l < LMAX; l += 1) {
        var s = 1 / Math.sqrt(D);
        LAYERS.push({
          Wq: mat(rnd, D, D, s), Wk: mat(rnd, D, D, s), Wv: mat(rnd, D, D, s),
          Wo: mat(rnd, D, D, s), W1: mat(rnd, D, EXP * D, s),
          W2: mat(rnd, EXP * D, D, 1 / Math.sqrt(EXP * D))
        });
      }

      // ---- подслои ---------------------------------------------------------
      function layerNorm(Xm) {
        var out = new Array(Xm.length), i, j, mu, va, r;
        for (i = 0; i < Xm.length; i += 1) {
          r = Xm[i]; mu = 0; for (j = 0; j < D; j += 1) mu += r[j]; mu /= D;
          va = 0; for (j = 0; j < D; j += 1) va += (r[j] - mu) * (r[j] - mu); va /= D;
          out[i] = new Float64Array(D);
          for (j = 0; j < D; j += 1) out[i][j] = (r[j] - mu) / Math.sqrt(va + 1e-5);
        }
        return out;
      }
      function gelu(x) {
        return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
      }
      function attn(Xm, p, gain) {
        var Q = mm(Xm, p.Wq), Kk = mm(Xm, p.Wk), V = mm(Xm, p.Wv);
        var dh = D / HEADS, out = new Array(N), i, j, h, t, sc, mx, sum, w, acc;
        for (i = 0; i < N; i += 1) out[i] = new Float64Array(D);
        for (h = 0; h < HEADS; h += 1) {
          var off = h * dh;
          for (i = 0; i < N; i += 1) {
            var sco = new Float64Array(i + 1);              // causal
            mx = -Infinity;
            for (j = 0; j <= i; j += 1) {
              sc = 0; for (t = 0; t < dh; t += 1) sc += Q[i][off + t] * Kk[j][off + t];
              sco[j] = sc / Math.sqrt(dh); if (sco[j] > mx) mx = sco[j];
            }
            sum = 0; for (j = 0; j <= i; j += 1) { sco[j] = Math.exp(sco[j] - mx); sum += sco[j]; }
            for (j = 0; j <= i; j += 1) {
              w = sco[j] / sum;
              for (t = 0; t < dh; t += 1) out[i][off + t] += w * V[j][off + t];
            }
          }
        }
        acc = mm(out, p.Wo);
        for (i = 0; i < N; i += 1) for (j = 0; j < D; j += 1) acc[i][j] *= gain;
        return acc;
      }
      function mlp(Xm, p, gain) {
        var Hh = mm(Xm, p.W1), i, j;
        for (i = 0; i < N; i += 1) for (j = 0; j < EXP * D; j += 1) Hh[i][j] = gelu(Hh[i][j]);
        var o = mm(Hh, p.W2);
        for (i = 0; i < N; i += 1) for (j = 0; j < D; j += 1) o[i][j] *= gain;
        return o;
      }
      function add(A, B) {
        var i, j, o = new Array(A.length);
        for (i = 0; i < A.length; i += 1) { o[i] = new Float64Array(D); for (j = 0; j < D; j += 1) o[i][j] = A[i][j] + B[i][j]; }
        return o;
      }
      function apply(Xm, F, cfg) {
        if (cfg.norm === "post") return cfg.res ? layerNorm(add(Xm, F(Xm))) : layerNorm(F(Xm));
        var pre = cfg.norm === "pre" ? layerNorm(Xm) : Xm;
        var f = F(pre);
        return cfg.res ? add(Xm, f) : f;
      }

      function forward(Xstart, cfg) {
        var x = clone(Xstart), traceN = [1], traceC = [1], i, j;
        var n0 = fro(Xstart);
        for (var li = 0; li < cfg.L; li += 1) {
          var p = LAYERS[li];
          if (cfg.attn) x = apply(x, function (z) { return attn(z, p, cfg.gain); }, cfg);
          if (cfg.mlp) x = apply(x, function (z) { return mlp(z, p, cfg.gain); }, cfg);
          traceN.push(fro(x) / n0);
          var cs = 0;
          for (i = 0; i < N; i += 1) {
            var dot = 0, a = 0, b = 0;
            for (j = 0; j < D; j += 1) { dot += x[i][j] * Xstart[i][j]; a += x[i][j] * x[i][j]; b += Xstart[i][j] * Xstart[i][j]; }
            cs += dot / (Math.sqrt(a * b) + 1e-12);
          }
          traceC.push(cs / N);
        }
        return { x: x, norms: traceN, coss: traceC };
      }
      function fro(A) { var s = 0, i, j; for (i = 0; i < A.length; i += 1) for (j = 0; j < D; j += 1) s += A[i][j] * A[i][j]; return Math.sqrt(s); }

      function influence(cfg) {          // ||d h_last / d x_first|| по фиксированному направлению
        var eps = 1e-4, i, Xp = clone(X0), Xm2 = clone(X0);
        for (i = 0; i < D; i += 1) { Xp[0][i] += eps * UDIR[i]; Xm2[0][i] -= eps * UDIR[i]; }
        var a = forward(Xp, cfg).x[N - 1], b = forward(Xm2, cfg).x[N - 1], s = 0;
        for (i = 0; i < D; i += 1) s += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.sqrt(s) / (2 * eps);
      }

      // ---- интерфейс -------------------------------------------------------
      var cfg = { L: 12, gain: 1.0, attn: true, mlp: true, res: true, norm: "pre" };

      K.hint(root, "Стек из случайных, но фиксированных блоков: обучения нет, измеряется только маршрут сигнала. Слева — норма представления по глубине, справа — средний косинус между представлением позиции и её исходным embedding. Отключите residual — и через десяток блоков от начального вектора не останется ничего; отключите attention — и последняя позиция перестанет зависеть от первой (нижняя строка отчёта обнулится).");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Влияние первого токена оценено центральной разностью: насколько сдвигается представление последней позиции, если чуть шевельнуть embedding первой. Ровно ноль означает, что архитектура физически не может передать информацию между позициями.");

      var cs = K.makeCanvas(stage, W, H, { label: "Норма и косинус по глубине стека блоков", onResize: draw, drag: false });

      function seg(label, key, opts) {
        K.segmented(controls, { label: label, value: String(cfg[key]), options: opts }, function (v) {
          cfg[key] = (v === "true" || v === "false") ? (v === "true") : v;
          draw();
        });
      }
      seg("attention", "attn", [{ value: "true", label: "вкл" }, { value: "false", label: "выкл" }]);
      seg("MLP", "mlp", [{ value: "true", label: "вкл" }, { value: "false", label: "выкл" }]);
      seg("residual", "res", [{ value: "true", label: "вкл" }, { value: "false", label: "выкл" }]);
      seg("нормировка", "norm", [{ value: "pre", label: "pre-LN" }, { value: "post", label: "post-LN" }, { value: "none", label: "нет" }]);
      K.slider(controls, { label: "глубина L", min: 1, max: LMAX, step: 1, value: cfg.L, format: function (v) { return String(v); } }, function (v) { cfg.L = v; draw(); });
      K.slider(controls, { label: "масштаб подслоя", min: 0.2, max: 2, step: 0.1, value: cfg.gain, format: function (v) { return v.toFixed(1); } }, function (v) { cfg.gain = v; draw(); });

      // ---- отрисовка -------------------------------------------------------
      function panel(ctx, x0, y0, w, h, title, ylab) {
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.strokeRect(x0, y0, w, h);
        ctx.fillStyle = C.muted; ctx.font = "12px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText(title, x0, y0 - 22);
        ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText(ylab, x0, y0 - 6);
      }
      function curve(ctx, x0, y0, w, h, vals, lo, hi, color, dash) {
        ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.clip();
        ctx.strokeStyle = color; ctx.lineWidth = 2.4;
        if (dash) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (var i = 0; i < vals.length; i += 1) {
          var px = x0 + (vals.length === 1 ? w : w * i / (vals.length - 1));
          var py = y0 + h - h * (vals[i] - lo) / (hi - lo);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke(); ctx.setLineDash([]);
        for (var j = 0; j < vals.length; j += 1) {
          var qx = x0 + (vals.length === 1 ? w : w * j / (vals.length - 1));
          var qy = y0 + h - h * (vals[j] - lo) / (hi - lo);
          ctx.fillStyle = color; ctx.beginPath(); ctx.arc(qx, qy, 2.6, 0, 7); ctx.fill();
        }
        ctx.restore();
      }
      function axisTicks(ctx, x0, y0, w, h, lo, hi, n, fmt) {
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "right";
        for (var i = 0; i <= n; i += 1) {
          var v = lo + (hi - lo) * i / n, py = y0 + h - h * i / n;
          ctx.fillText(fmt(v), x0 - 6, py + 4);
          ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 0.6;
          ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x0 + w, py); ctx.stroke();
        }
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        var full = forward(X0, cfg);
        var ref = forward(X0, { L: cfg.L, gain: cfg.gain, attn: cfg.attn, mlp: cfg.mlp, res: !cfg.res, norm: cfg.norm });

        var PW = 350, PH = 300, X1 = 70, X2 = 520, Y = 90;
        var maxN = Math.max(2, Math.min(6, Math.max.apply(null, full.norms.concat(ref.norms))));
        panel(ctx, X1, Y, PW, PH, "Норма представления", "‖X⁽ˡ⁾‖ / ‖X⁽⁰⁾‖");
        axisTicks(ctx, X1, Y, PW, PH, 0, maxN, 4, function (v) { return v.toFixed(1); });
        curve(ctx, X1, Y, PW, PH, ref.norms, 0, maxN, C.muted || "#6e726a", true);
        curve(ctx, X1, Y, PW, PH, full.norms, 0, maxN, C.blue);

        panel(ctx, X2, Y, PW, PH, "Память о собственном embedding", "средний cos(x⁽ˡ⁾, x⁽⁰⁾)");
        axisTicks(ctx, X2, Y, PW, PH, -0.4, 1, 7, function (v) { return v.toFixed(1); });
        curve(ctx, X2, Y, PW, PH, ref.coss, -0.4, 1, C.muted || "#6e726a", true);
        curve(ctx, X2, Y, PW, PH, full.coss, -0.4, 1, C.blue);

        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "center";
        ctx.fillText("номер блока ℓ", X1 + PW / 2, Y + PH + 26);
        ctx.fillText("номер блока ℓ", X2 + PW / 2, Y + PH + 26);
        ctx.textAlign = "left";
        ctx.fillStyle = C.blue; ctx.fillText("текущая настройка", X1, Y + PH + 46);
        ctx.fillStyle = C.muted; ctx.fillText(cfg.res ? "— — та же сеть без residual" : "— — та же сеть с residual", X1 + 140, Y + PH + 46);

        var inf = influence(cfg);
        out.set([
          { label: "норма после L блоков", value: "×" + full.norms[full.norms.length - 1].toFixed(2), color: C.blue },
          { label: "cos с исходным embedding", value: full.coss[full.coss.length - 1].toFixed(3), color: C.green || "#38735d" },
          { label: "влияние первого токена на последний", value: inf < 1e-12 ? "0 (пути нет)" : inf.toFixed(3), color: inf < 1e-12 ? C.red : C.gold }
        ]);
      }

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
