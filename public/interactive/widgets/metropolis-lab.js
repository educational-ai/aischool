// Lesson 67: metropolis-lab — run a random-walk Metropolis chain on a target you choose,
// watch acceptance rate, mode crossings, autocorrelation and the effective sample size.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("metropolis-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var PB = "#fffef9";
      // plotting frame for the density/histogram panel and the trace panel
      var LX = 60, LW = 300, TX = 400, TW = 470, TOP = 40, BOT = 420;
      var XMIN = -9, XMAX = 9;

      var target = "two";           // "one" | "two" | "narrow"
      var sigma = 1.2;
      var startAt = 0;
      var running = false;
      var raf = null;

      // ---- deterministic RNG (mulberry32) so a rerun repeats exactly
      var seed = 6701;
      var rngState = seed;
      function rnd() {
        rngState |= 0; rngState = (rngState + 0x6D2B79F5) | 0;
        var t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      }
      function gauss() {
        var u = 1 - rnd(), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // ---- targets (unnormalised); trueLeft = exact probability of x < 0
      function dens(x) {
        if (target === "one") return Math.exp(-x * x / 2);
        if (target === "two") {
          return 0.5 * Math.exp(-(x + 3) * (x + 3) / (2 * 0.7 * 0.7)) / 0.7 +
                 0.5 * Math.exp(-(x - 3) * (x - 3) / (2 * 0.7 * 0.7)) / 0.7;
        }
        // narrow + wide: equal mass, very different widths
        return 0.5 * Math.exp(-(x + 3) * (x + 3) / (2 * 0.3 * 0.3)) / 0.3 +
               0.5 * Math.exp(-(x - 3) * (x - 3) / (2 * 2.0 * 2.0)) / 2.0;
      }
      function trueLeft() { return target === "one" ? 0.5 : 0.5; }

      // ---- chain state
      var chain, x, accepted, steps, crossings, lastSide, hist, HB = 90;
      function reset() {
        rngState = seed;
        x = startAt;
        chain = [];
        hist = new Float64Array(HB);
        accepted = 0; steps = 0; crossings = 0; lastSide = 0;
        record(x);
      }
      function record(v) {
        chain.push(v);
        if (chain.length > 40000) chain.shift();
        var b = Math.floor((v - XMIN) / (XMAX - XMIN) * HB);
        if (b >= 0 && b < HB) hist[b] += 1;
        var side = v >= 1.2 ? 1 : (v <= -1.2 ? -1 : 0);
        if (side !== 0) {
          if (lastSide !== 0 && side !== lastSide) crossings += 1;
          lastSide = side;
        }
      }
      function step() {
        var y = x + sigma * gauss();
        var r = dens(y) / dens(x);
        if (rnd() < r) { x = y; accepted += 1; }
        steps += 1;
        record(x);
      }
      function advance(n) { for (var i = 0; i < n; i += 1) step(); }

      // ---- diagnostics
      function tauInt() {
        var n = chain.length;
        if (n < 200) return NaN;
        var burn = Math.floor(n * 0.2);
        var a = chain.slice(burn);
        if (a.length > 6000) a = a.slice(a.length - 6000);   // keep the redraw cheap
        var m = a.length, mean = 0, i;
        for (i = 0; i < m; i += 1) mean += a[i];
        mean /= m;
        var d0 = 0;
        for (i = 0; i < m; i += 1) d0 += (a[i] - mean) * (a[i] - mean);
        if (d0 < 1e-12) return NaN;
        var maxlag = Math.min(200, Math.floor(m / 4));
        var rho = new Float64Array(maxlag + 1);
        for (var k = 0; k <= maxlag; k += 1) {
          var s = 0;
          for (i = 0; i + k < m; i += 1) s += (a[i] - mean) * (a[i + k] - mean);
          rho[k] = s / d0;
        }
        var tau = 1, p = 1;
        while (2 * p < maxlag) {
          var pair = rho[2 * p - 1] + rho[2 * p];
          if (pair <= 0) break;
          tau += 2 * pair; p += 1;
        }
        return Math.max(tau, 1);
      }
      function fracLeft() {
        var n = chain.length, burn = Math.floor(n * 0.2), c = 0;
        for (var i = burn; i < n; i += 1) if (chain[i] < 0) c += 1;
        return n - burn > 0 ? c / (n - burn) : NaN;
      }

      K.hint(root, "Цепь Метрополиса не рисует картинку — она гуляет. Выберите цель, задайте ширину предложения σ и точку старта, затем добавляйте такты. Следите не за долей принятия, а за тремя вещами: сколько раз цепь перешла между модами, насколько гистограмма похожа на цель и каков эффективный размер выборки N_eff. При σ = 0,05 принимается почти всё — и N_eff остаётся крошечным.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var buttons = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева: чёрная кривая — целевая плотность π, золотые столбики — гистограмма состояний цепи (все такты, включая повторы после отказа). Справа: трасса x_t. Доля x<0 сравнивается с точным значением 0,50; расхождение при большом числе тактов означает, что цепь не перемешалась.");

      var cs = K.makeCanvas(stage, W, H, { label: "Целевая плотность, гистограмма цепи и трасса", onResize: draw, drag: false });

      K.segmented(controls, {
        label: "Цель", value: "two",
        options: [
          { label: "одна мода", value: "one" },
          { label: "две моды", value: "two" },
          { label: "узкая + широкая", value: "narrow" }
        ]
      }, function (v) { target = v; reset(); draw(); });

      K.segmented(controls, {
        label: "Старт", value: "0",
        options: [
          { label: "x = 0", value: "0" },
          { label: "x = −3", value: "-3" },
          { label: "x = 8", value: "8" }
        ]
      }, function (v) { startAt = parseFloat(v); reset(); draw(); });

      K.slider(controls, {
        label: "Ширина предложения σ", min: 0.05, max: 8, step: 0.05, value: sigma,
        format: function (v) { return v.toFixed(2); }
      }, function (v) { sigma = v; reset(); draw(); });

      function mkBtn(text, fn) {
        var b = K.element("button", "kontur-int-segment", { type: "button", text: text });
        b.style.margin = "0 6px";
        b.addEventListener("click", fn);
        buttons.appendChild(b);
        return b;
      }
      mkBtn("+200 тактов", function () { advance(200); draw(); });
      mkBtn("+5000 тактов", function () { advance(5000); draw(); });
      var runBtn = mkBtn("пуск", function () {
        running = !running;
        runBtn.textContent = running ? "пауза" : "пуск";
        if (running) loop(); else if (raf) { cancelAnimationFrame(raf); raf = null; }
      });
      mkBtn("сброс", function () {
        running = false; runBtn.textContent = "пуск";
        if (raf) { cancelAnimationFrame(raf); raf = null; }
        reset(); draw();
      });

      function loop() {
        if (!running) return;
        advance(120);
        draw();
        raf = requestAnimationFrame(loop);
      }

      function xToPx(v) { return LX + (v - XMIN) / (XMAX - XMIN) * LW; }

      function draw() {
        var ctx = cs.ctx, i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        // ---------- left panel: target + histogram
        var maxd = 0;
        for (i = 0; i <= 200; i += 1) {
          var xv = XMIN + (XMAX - XMIN) * i / 200;
          maxd = Math.max(maxd, dens(xv));
        }
        var maxh = 1;
        for (i = 0; i < HB; i += 1) maxh = Math.max(maxh, hist[i]);

        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(LX, BOT); ctx.lineTo(LX + LW, BOT);
        ctx.moveTo(LX, TOP); ctx.lineTo(LX, BOT);
        ctx.stroke();

        // histogram bars
        ctx.fillStyle = "rgba(165,121,32,0.55)";
        var bw = LW / HB;
        for (i = 0; i < HB; i += 1) {
          if (hist[i] <= 0) continue;
          var hgt = (hist[i] / maxh) * (BOT - TOP - 10);
          ctx.fillRect(LX + i * bw, BOT - hgt, bw - 0.6, hgt);
        }
        // target curve
        ctx.strokeStyle = C.ink || "#171915"; ctx.lineWidth = 2.2;
        ctx.beginPath();
        for (i = 0; i <= 300; i += 1) {
          var xv2 = XMIN + (XMAX - XMIN) * i / 300;
          var py = BOT - (dens(xv2) / maxd) * (BOT - TOP - 10);
          if (i === 0) ctx.moveTo(xToPx(xv2), py); else ctx.lineTo(xToPx(xv2), py);
        }
        ctx.stroke();
        // zero line
        ctx.strokeStyle = "rgba(110,114,106,0.7)"; ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(xToPx(0), TOP); ctx.lineTo(xToPx(0), BOT); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var t = -6; t <= 6; t += 3) ctx.fillText(String(t), xToPx(t), BOT + 18);
        ctx.fillText("x", LX + LW + 14, BOT + 5);
        ctx.fillStyle = C.ink || "#171915"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("цель π и гистограмма цепи", LX + LW / 2, TOP - 14);

        // ---------- right panel: trace
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(TX, BOT); ctx.lineTo(TX + TW, BOT);
        ctx.moveTo(TX, TOP); ctx.lineTo(TX, BOT);
        ctx.stroke();
        var n = chain.length;
        var show = Math.min(n, 1500);
        var from = n - show;
        ctx.strokeStyle = C.blue; ctx.lineWidth = 1;
        ctx.beginPath();
        for (i = 0; i < show; i += 1) {
          var v = chain[from + i];
          var px = TX + (i / Math.max(show - 1, 1)) * TW;
          var py = BOT - (v - XMIN) / (XMAX - XMIN) * (BOT - TOP);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.strokeStyle = "rgba(110,114,106,0.7)"; ctx.setLineDash([4, 3]);
        var pyz = BOT - (0 - XMIN) / (XMAX - XMIN) * (BOT - TOP);
        ctx.beginPath(); ctx.moveTo(TX, pyz); ctx.lineTo(TX + TW, pyz); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.ink || "#171915"; ctx.textAlign = "center"; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("трасса: последние " + show + " тактов", TX + TW / 2, TOP - 14);
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "right";
        ctx.fillText("x", TX - 8, TOP + 10);

        // ---------- readout
        var tau = tauInt();
        var neff = isFinite(tau) ? Math.floor(chain.length * 0.8) / tau : NaN;
        var fl = fracLeft();
        var accRate = steps > 0 ? accepted / steps : 0;
        var flColor = Math.abs(fl - trueLeft()) > 0.05 ? C.red : C.green;
        output.set([
          { label: "тактов сделано", value: String(steps), color: C.blue },
          { label: "доля принятия", value: (accRate * 100).toFixed(1) + "%", color: C.gold },
          { label: "переходов между модами", value: String(crossings), color: crossings === 0 && steps > 3000 ? C.red : C.green },
          { label: "доля x<0 (точно 0,50)", value: isFinite(fl) ? fl.toFixed(3) : "—", color: flColor },
          { label: "τ_int", value: isFinite(tau) ? tau.toFixed(1) : "—", color: C.violet || C.muted },
          { label: "N_eff", value: isFinite(neff) ? Math.round(neff).toString() : "—", color: C.red }
        ]);
      }

      reset();
      draw();

      return function () {
        running = false;
        if (raf) cancelAnimationFrame(raf);
        cs.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
