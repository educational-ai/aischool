// Lesson 41: probability-lab — roll two dice, watch a random variable's distribution and the law of large numbers.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("probability-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var mode = 0; // 0 sum, 1 max, 2 |diff|
      var counts = {}, total = 0, last = null, history = [];

      function rv(i, j) { return mode === 0 ? i + j : mode === 1 ? Math.max(i, j) : Math.abs(i - j); }
      function values() { return mode === 0 ? range(2, 12) : mode === 1 ? range(1, 6) : range(0, 5); }
      function range(a, b) { var r = []; for (var v = a; v <= b; v += 1) r.push(v); return r; }
      function theo() {
        var t = {}; values().forEach(function (v) { t[v] = 0; });
        for (var i = 1; i <= 6; i += 1) for (var j = 1; j <= 6; j += 1) t[rv(i, j)] += 1 / 36;
        return t;
      }
      // event for the LLN panel: RV >= threshold (its rarest-ish upper tail)
      function eventThreshold() { return mode === 0 ? 10 : mode === 1 ? 6 : 4; }
      function inEvent(v) { return v >= eventThreshold(); }
      function theoEvent() { var t = theo(), p = 0; values().forEach(function (v) { if (inEvent(v)) p += t[v]; }); return p; }

      function reset() { counts = {}; total = 0; last = null; history = []; values().forEach(function (v) { counts[v] = 0; }); }

      function roll(k) {
        for (var n = 0; n < k; n += 1) {
          var i = 1 + Math.floor(Math.random() * 6), j = 1 + Math.floor(Math.random() * 6);
          var v = rv(i, j); counts[v] += 1; total += 1; last = [i, j, v];
          if (history.length < 4000) history.push(inEvent(v) ? 1 : 0);
        }
        draw();
      }

      K.hint(
        root,
        "Бросайте два кубика и следите за случайной величиной. Слева — пространство исходов, раскрашенное по её значению; справа — распределение, которое набирается из бросков и стремится к теоретическому. Внизу видно, как доля события сходится к его вероятности — это закон больших чисел.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Каждая из 36 пар кубиков равновероятна (1/36). Случайная величина — функция от исхода: она приписывает паре число. Красные точки — точные вероятности значений; столбцы — накопленные частоты. С ростом числа бросков частоты приближаются к вероятностям.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Пространство исходов и распределение", onResize: draw, drag: false });
      reset();

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        // sample space grid
        var gx = 55, gy = 60, cs = 40;
        var vals = values(), vmin = vals[0], vmax = vals[vals.length - 1];
        ctx.textAlign = "left"; ctx.fillStyle = C.ink; ctx.font = "14px PT Sans, sans-serif";
        ctx.fillText("пространство исходов", gx, gy - 16);
        for (var i = 1; i <= 6; i += 1) for (var j = 1; j <= 6; j += 1) {
          var v = rv(i, j), fr = (v - vmin) / (vmax - vmin + 1e-9);
          var x = gx + (i - 1) * cs, y = gy + (6 - j) * cs;
          ctx.fillStyle = mix(fr); ctx.fillRect(x, y, cs - 2, cs - 2);
          ctx.fillStyle = fr > 0.6 ? "#fff" : C.ink; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(String(v), x + cs / 2 - 1, y + cs / 2 + 3);
          if (inEvent(v)) { ctx.strokeStyle = C.red; ctx.lineWidth = 2; ctx.strokeRect(x, y, cs - 2, cs - 2); }
          if (last && last[0] === i && last[1] === j) { ctx.strokeStyle = C.gold; ctx.lineWidth = 3; ctx.strokeRect(x - 1, y - 1, cs, cs); }
        }
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
        for (var d = 1; d <= 6; d += 1) { ctx.fillText(String(d), gx + (d - 1) * cs + cs / 2 - 1, gy + 6 * cs + 14); ctx.fillText(String(d), gx - 12, gy + (6 - d) * cs + cs / 2 + 3); }
        ctx.fillStyle = C.red; ctx.textAlign = "left"; ctx.fillText("красная рамка — событие", gx, gy + 6 * cs + 34);

        // histogram: empirical vs theoretical
        var hx = 420, hy = 60, hw = 420, hh = 220;
        var t = theo(), mx = 0;
        vals.forEach(function (v) { mx = Math.max(mx, t[v], total ? counts[v] / total : 0); });
        mx = Math.max(mx, 0.05);
        var bw = hw / vals.length;
        ctx.fillStyle = C.ink; ctx.font = "14px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("распределение величины", hx, hy - 16);
        vals.forEach(function (v, k) {
          var emp = total ? counts[v] / total : 0;
          var bx = hx + k * bw;
          ctx.fillStyle = "rgba(49,95,140,0.75)";
          ctx.fillRect(bx + 3, hy + hh - emp / mx * hh, bw - 6, emp / mx * hh);
          // theoretical dot
          var ty = hy + hh - t[v] / mx * hh;
          ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(bx + bw / 2, ty, 3.5, 0, 7); ctx.fill();
          ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "10px PT Sans, sans-serif";
          ctx.fillText(String(v), bx + bw / 2, hy + hh + 14);
        });
        ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(hx, hy + hh); ctx.lineTo(hx + hw, hy + hh); ctx.stroke();
        ctx.fillStyle = C.blue; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("столбцы — накопленная частота", hx, hy + hh + 32);
        ctx.fillStyle = C.red; ctx.fillText("точки — точная вероятность", hx + 210, hy + hh + 32);

        // LLN convergence panel
        var lx = 420, ly = 330, lw = 420, lh = 92;
        var pe = theoEvent();
        ctx.strokeStyle = C.line; ctx.strokeRect(lx, ly, lw, lh);
        var lo = pe - 0.25, hi = pe + 0.25;
        function Y(p) { return ly + lh - (p - lo) / (hi - lo) * lh; }
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(lx, Y(pe)); ctx.lineTo(lx + lw, Y(pe)); ctx.stroke();
        if (history.length > 1) {
          ctx.strokeStyle = C.blue; ctx.lineWidth = 1.4; ctx.beginPath();
          var run = 0;
          for (var h = 0; h < history.length; h += 1) {
            run += history[h]; var p = run / (h + 1);
            var px = lx + h / (history.length - 1) * lw, py = Math.max(ly, Math.min(ly + lh, Y(p)));
            if (h === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
        }
        ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("доля события сходится к вероятности " + pe.toFixed(3), lx, ly - 6);

        var empE = total ? vals.reduce(function (a, v) { return a + (inEvent(v) ? counts[v] : 0); }, 0) / total : 0;
        output.set([
          { label: "Бросков", value: String(total), color: C.ink },
          { label: "Доля события (частота)", value: total ? empE.toFixed(3) : "—", color: C.blue },
          { label: "Вероятность события (модель)", value: pe.toFixed(3), color: C.red },
        ]);
      }

      function mix(fr) {
        var r = Math.round(250 - fr * 210), g = Math.round(248 - fr * 120), b = Math.round(200 + fr * 40);
        return "rgb(" + r + "," + g + "," + b + ")";
      }

      K.segmented(controls, { label: "Случайная величина", value: 0, options: [
        { label: "сумма", value: 0 }, { label: "максимум", value: 1 }, { label: "|разность|", value: 2 } ] }, function (v) { mode = v; reset(); draw(); });
      var b1 = K.element("button", "kontur-int-segment", { type: "button", text: "бросить 10" });
      var b2 = K.element("button", "kontur-int-segment", { type: "button", text: "бросить 1000" });
      var b3 = K.element("button", "kontur-int-segment", { type: "button", text: "сброс" });
      b1.style.margin = b2.style.margin = b3.style.margin = "0 6px";
      b1.addEventListener("click", function () { roll(10); });
      b2.addEventListener("click", function () { roll(1000); });
      b3.addEventListener("click", function () { reset(); draw(); });
      controls.appendChild(b1); controls.appendChild(b2); controls.appendChild(b3);

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
