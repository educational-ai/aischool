// Lesson 74: gated-memory-lab — one forget gate sets the time scale of memory AND the reach of
// the gradient; a constant input gate cannot both write the key and ignore the chatter. Two runs
// (key = +1 and key = −1) with identical noise show how much of the answer survives the pause.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("gated-memory-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 920, H = 570;
      var T = 90;                 // steps
      var KEY = 5;                // the key arrives here
      var QUERY = 70;             // the answer is asked here

      var state = { f: 0.95, i: 0.6, o: 0.9, noise: 0.0, cell: "lstm" };

      K.hint(
        root,
        "Ключ (+1 или −1) приходит на шаге 5, ответ спрашивают на шаге 70: между ними 65 шагов паузы, " +
          "заполненной посторонним шумом. Ворота здесь — не человечки, а три постоянных числа: f хранит, " +
          "i записывает, o показывает. Две кривые — один и тот же прогон с ключом +1 и −1; важно не то, " +
          "куда пришла память, а расстояние между кривыми: только оно и есть сохранённый ответ.",
      );

      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Сверху — вход: импульс-ключ, посторонний шум в паузе и запрос. В середине — память c_t для двух " +
          "значений ключа; серая полоса показывает, где ответ уже неразличим на выходе h = o·tanh(c). " +
          "Снизу — |∂c_T/∂c_t| по лагу: для аддитивной памяти это f^k, для простой RNN добавляется " +
          "множитель 1 − h², и кривая падает круче. Поднимите шум при большом i — увидите главное: " +
          "постоянными воротами нельзя одновременно записать ключ и не записать болтовню, поэтому " +
          "в настоящей сети ворота являются функцией входа.",
      );

      var cs = K.makeCanvas(stage, W, H, {
        label: "Ворота LSTM: время жизни памяти, помехи в паузе и дальность градиента",
        onResize: draw,
        drag: false,
      });

      function chatter(t) {
        var s = Math.sin(t * 12.9898 + 78.233) * 43758.5453;
        return 2 * (s - Math.floor(s)) - 1;
      }

      function inputAt(t, key) {
        if (t === KEY) return key;
        if (t > KEY && t < QUERY) return state.noise * chatter(t);
        return 0;
      }

      function run(key) {
        var c = 0, h = 0, cs_ = [], hs_ = [], der = [];
        for (var t = 0; t < T; t += 1) {
          var x = inputAt(t, key);
          if (state.cell === "lstm") {
            c = state.f * c + state.i * Math.tanh(2 * x);
            h = state.o * Math.tanh(c);
            der.push(state.f);
          } else {
            h = Math.tanh(state.f * h + 2 * state.i * x);
            c = h;
            der.push(state.f * (1 - h * h));
          }
          cs_.push(c); hs_.push(h);
        }
        var jac = [], p = 1;
        for (var k = 0; k < T; k += 1) {
          jac.push(Math.abs(p));
          p = p * der[Math.max(0, T - 1 - k)];
        }
        return { cs: cs_, hs: hs_, jac: jac };
      }

      function halfLife() {
        var a = state.f;
        if (a >= 1 || a <= 0) return null;
        return Math.log(0.5) / Math.log(a);
      }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var rp = run(1), rm = run(-1);

        var gx = 76, gw = W - 118;
        function X(t) { return gx + (t / (T - 1)) * gw; }
        function events(y0, y1) {
          ctx.save();
          ctx.setLineDash([4, 3]); ctx.lineWidth = 1; ctx.strokeStyle = C.muted;
          [KEY, QUERY].forEach(function (t) {
            ctx.beginPath(); ctx.moveTo(X(t), y0); ctx.lineTo(X(t), y1); ctx.stroke();
          });
          ctx.restore();
        }

        // ---------- panel 1: input
        var ay = 30, ah = 70, amid = ay + ah / 2;
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, ay, gw, ah);
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.beginPath(); ctx.moveTo(gx, amid); ctx.lineTo(gx + gw, amid); ctx.stroke();
        events(ay, ay + ah);
        ctx.fillStyle = "rgba(150,153,144,0.6)";
        for (var t2 = KEY + 1; t2 < QUERY; t2 += 1) {
          var v = state.noise * chatter(t2);
          var bh0 = -v * (ah / 2 - 6);
          ctx.fillRect(X(t2) - 2, Math.min(amid, amid + bh0), 4, Math.abs(bh0));
        }
        ctx.fillStyle = C.gold;
        ctx.fillRect(X(KEY) - 3, amid - (ah / 2 - 6), 6, ah / 2 - 6);
        ctx.fillStyle = C.ink; ctx.font = "12px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("ключ ±1", X(KEY) + 6, ay + 15);
        ctx.fillText("запрос", X(QUERY) + 6, ay + 15);
        ctx.fillStyle = C.muted;
        ctx.fillText("вход: ключ, посторонний шум в паузе, запрос через " + (QUERY - KEY) + " шагов", gx, ay - 8);

        // ---------- panel 2: memory for key = +1 and key = −1
        var by = 136, bh = 200, mid = by + bh / 2;
        ctx.strokeStyle = C.line; ctx.strokeRect(gx, by, gw, bh);
        var span = 1.05;
        for (var i = 0; i < T; i += 1) {
          span = Math.max(span, Math.abs(rp.cs[i]), Math.abs(rm.cs[i]));
        }
        span = Math.min(span, 8);
        function Y(v) {
          var q = Math.max(-span, Math.min(span, v));
          return mid - (q / span) * (bh / 2 - 10);
        }
        // band where the difference on the output is indistinguishable
        var dead = 0.1 / Math.max(state.o, 1e-6);
        if (dead < span) {
          ctx.fillStyle = "rgba(110,114,106,0.10)";
          ctx.fillRect(gx + 1, Y(dead), gw - 2, Y(-dead) - Y(dead));
        }
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(gx, mid); ctx.lineTo(gx + gw, mid); ctx.stroke();
        events(by, by + bh);

        function poly(vals, color, width) {
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
          for (var k = 0; k < T; k += 1) {
            if (k === 0) ctx.moveTo(X(k), Y(vals[k])); else ctx.lineTo(X(k), Y(vals[k]));
          }
          ctx.stroke();
        }
        poly(rp.cs, C.blue, 2.6);
        poly(rm.cs, C.gold, 2.2);

        ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText(
          state.cell === "lstm"
            ? "память c_t: ключ +1 (синяя) и ключ −1 (жёлтая)"
            : "простая RNN, состояние h_t = tanh(f·h + i·x): ключ +1 и ключ −1",
          gx, by - 8,
        );
        ctx.font = "11px PT Sans, sans-serif"; ctx.fillStyle = C.muted; ctx.textAlign = "right";
        ctx.fillText("+" + span.toFixed(1).replace(".", ","), gx - 8, by + 12);
        ctx.fillText("0", gx - 8, mid + 4);
        ctx.fillText("−" + span.toFixed(1).replace(".", ","), gx - 8, by + bh - 4);
        ctx.textAlign = "left";
        if (dead < span) ctx.fillText("на выходе неразличимо", gx + 6, Y(dead) - 4);

        var hl = halfLife();
        if (hl && KEY + hl < T) {
          ctx.strokeStyle = C.red; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(X(KEY + hl), by + 4); ctx.lineTo(X(KEY + hl), by + bh - 4);
          ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = C.red; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText("полураспад " + hl.toFixed(1).replace(".", ",") + " шага",
            X(KEY + hl) + 5, by + bh - 10);
        }

        // ---------- panel 3: gradient reach
        var cy = 380, ch = 152;
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(gx, cy, gw, ch);
        var LO = -14, HI = 1;
        function YL(v) {
          return cy + ch - (Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO) * ch;
        }
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.font = "10.5px PT Sans, sans-serif";
        for (var g = LO; g <= HI; g += 5) {
          ctx.beginPath(); ctx.moveTo(gx, YL(g)); ctx.lineTo(gx + gw, YL(g)); ctx.stroke();
          ctx.fillText("10^" + g, gx - 8, YL(g) + 4);
        }
        ctx.fillStyle = "rgba(185,74,59,0.08)";
        ctx.fillRect(gx + 1, YL(-6), gw - 2, YL(LO) - YL(-6));
        ctx.fillStyle = C.red; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("сигнал обучения тонет в шуме оптимизации", gx + 8, YL(-6) + 14);

        ctx.strokeStyle = C.green; ctx.lineWidth = 2.4; ctx.beginPath();
        for (i = 0; i < T; i += 1) {
          var lv = Math.log(rp.jac[i] + 1e-300) / Math.LN10;
          if (i === 0) ctx.moveTo(X(i), YL(lv)); else ctx.lineTo(X(i), YL(lv));
        }
        ctx.stroke();
        ctx.fillStyle = C.ink; ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("|∂c_T/∂c_t| по лагу T − t (десятичный логарифм)", gx, cy - 8);
        ctx.fillStyle = C.muted; ctx.font = "11px PT Sans, sans-serif"; ctx.textAlign = "center";
        for (var tk = 0; tk < T; tk += 15) ctx.fillText(String(tk), X(tk), cy + ch + 16);
        ctx.fillText("лаг", gx + gw + 20, cy + ch + 16);

        // ---------- readout
        var lag = QUERY - KEY;
        var sepC = Math.abs(rp.cs[QUERY] - rm.cs[QUERY]);
        var sepH = Math.abs(rp.hs[QUERY] - rm.hs[QUERY]);
        var grad = rp.jac[Math.min(lag, T - 1)];
        output.set([
          {
            label: "полураспад памяти",
            value: hl ? hl.toFixed(1).replace(".", ",") + " шага" : "не гаснет",
            color: C.blue,
          },
          {
            label: "разделение в памяти на запросе",
            value: sepC.toFixed(3).replace(".", ","),
            color: C.blue,
          },
          {
            label: "разделение на выходе h",
            value: sepH.toFixed(3).replace(".", ",") + (sepH >= 0.1 ? " — ответ читается" : " — потерян"),
            color: sepH >= 0.1 ? C.green : C.red,
          },
          {
            label: "градиент через " + lag + " шагов",
            value: "10^" + (Math.log(grad + 1e-300) / Math.LN10).toFixed(1).replace(".", ","),
            color: C.gold,
          },
        ]);
      }

      K.slider(controls, {
        label: "Forget gate f (хранить)", min: 0.5, max: 1.0, step: 0.005, value: state.f,
        format: function (v) { return v.toFixed(3).replace(".", ","); },
      }, function (v) { state.f = v; draw(); });

      K.slider(controls, {
        label: "Input gate i (записать)", min: 0, max: 1, step: 0.01, value: state.i,
        format: function (v) { return v.toFixed(2).replace(".", ","); },
      }, function (v) { state.i = v; draw(); });

      K.slider(controls, {
        label: "Output gate o (показать)", min: 0, max: 1, step: 0.01, value: state.o,
        format: function (v) { return v.toFixed(2).replace(".", ","); },
      }, function (v) { state.o = v; draw(); });

      K.slider(controls, {
        label: "Шум в паузе", min: 0, max: 1, step: 0.02, value: state.noise,
        format: function (v) { return v.toFixed(2).replace(".", ","); },
      }, function (v) { state.noise = v; draw(); });

      K.segmented(controls, {
        label: "Ячейка", value: state.cell,
        options: [
          { value: "lstm", label: "LSTM: c = f·c + i·c̃" },
          { value: "rnn", label: "RNN: h = tanh(f·h + i·x)" },
        ],
      }, function (v) { state.cell = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
