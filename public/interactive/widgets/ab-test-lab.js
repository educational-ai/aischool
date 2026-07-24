// Lesson 55: ab-test-lab — 400 повторных A/B-экспериментов при заданном истинном эффекте.
// Видно три вещи сразу: разброс оценки, долю отклонений H0 (уровень или мощность)
// и завышение эффекта у «победителей» — особенно при остановке по первому p<0.05.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("ab-test-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 430;
      var PB = "#fffef9";
      var TRIALS = 400, LOOKS = 20, Z95 = 1.959964;
      var base = 0.08;
      var effect = 0.0;      // истинный эффект в процентных пунктах
      var nPer = 5000;       // объём одной группы
      var rule = "fixed";    // fixed | peek
      var seed = 20240055;

      // детерминированный генератор: одинаковые настройки — одинаковая картинка
      function mkRng(s) {
        var x = s >>> 0;
        return function () {
          x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
          return x / 4294967296;
        };
      }
      function normal(rnd) {
        var u = 1 - rnd(), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }

      // одна серия экспериментов; возвращает список исходов
      function simulate() {
        var rnd = mkRng(seed + Math.round(effect * 1000) * 7919 + nPer * 31 + (rule === "peek" ? 104729 : 0));
        var pA = base, pB = base + effect / 100;
        var batch = Math.max(1, Math.floor(nPer / LOOKS));
        var res = [], i, k;
        for (i = 0; i < TRIALS; i += 1) {
          var sA = 0, sB = 0, n = 0, stopped = false, out = null;
          var looks = rule === "peek" ? LOOKS : 1;
          var step = rule === "peek" ? batch : nPer;
          for (k = 0; k < looks && !stopped; k += 1) {
            // приращение успехов ~ нормальное приближение биномиального
            sA += step * pA + Math.sqrt(step * pA * (1 - pA)) * normal(rnd);
            sB += step * pB + Math.sqrt(step * pB * (1 - pB)) * normal(rnd);
            n += step;
            var qA = Math.min(0.999, Math.max(0.001, sA / n));
            var qB = Math.min(0.999, Math.max(0.001, sB / n));
            var se = Math.sqrt(qA * (1 - qA) / n + qB * (1 - qB) / n);
            var d = qB - qA, z = d / se;
            out = { d: d * 100, z: z, n: n };
            if (rule === "peek" && Math.abs(z) > Z95) stopped = true;
          }
          out.rejected = Math.abs(out.z) > Z95;
          res.push(out);
        }
        return res;
      }

      K.hint(root, "Каждый запуск — 400 повторов одного и того же эксперимента при истинном эффекте, который вы задали сами. Смотрите на три вещи: ширину облака оценок, долю отклонений нулевой гипотезы и среднюю оценку эффекта среди «победителей». Поставьте истинный эффект в ноль — доля отклонений должна держаться около 5%; включите остановку по первому p<0,05 — и она взлетит, хотя формула теста не менялась.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Гистограмма — оценки эффекта в 400 повторах, зелёная линия — истина, красным отмечены повторы, где тест отклонил H₀. Золотая линия — средняя оценка среди отклонивших. При положительном эффекте она заметно правее зелёной: победителями становятся удачные шумовые отклонения, и объявленный эффект систематически преувеличен тем сильнее, чем меньше мощность.");

      var cs = K.makeCanvas(stage, W, H, { label: "Распределение оценок эффекта в 400 экспериментах", onResize: draw, drag: false });

      K.slider(controls, {
        label: "истинный эффект, п.п.", min: 0, max: 2, step: 0.1, value: effect,
        format: function (v) { return v.toFixed(1) + " п.п."; }
      }, function (v) { effect = v; draw(); });
      K.slider(controls, {
        label: "объём одной группы n", min: 500, max: 40000, step: 500, value: nPer,
        format: function (v) { return String(v); }
      }, function (v) { nPer = v; draw(); });
      K.segmented(controls, {
        label: "правило остановки", value: rule,
        options: [
          { value: "fixed", label: "фиксированный горизонт" },
          { value: "peek", label: "остановка при p<0,05" }
        ]
      }, function (v) { rule = v; draw(); });

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var res = simulate();

        var lo = -3.0, hi = 3.0, BINS = 60;
        var i, hAll = new Array(BINS), hRej = new Array(BINS);
        for (i = 0; i < BINS; i += 1) { hAll[i] = 0; hRej[i] = 0; }
        var rej = 0, sumRej = 0, sumAll = 0;
        for (i = 0; i < res.length; i += 1) {
          var d = res[i].d;
          var b = Math.floor((Math.min(hi - 1e-9, Math.max(lo, d)) - lo) / (hi - lo) * BINS);
          hAll[b] += 1;
          sumAll += d;
          if (res[i].rejected) { hRej[b] += 1; rej += 1; sumRej += d; }
        }
        var top = 0;
        for (i = 0; i < BINS; i += 1) if (hAll[i] > top) top = hAll[i];

        var OX = 66, OY = 340, PW = W - OX - 40, PH = 250;
        function sx(v) { return OX + (v - lo) / (hi - lo) * PW; }
        function sy(c) { return OY - (top ? c / top : 0) * PH; }

        // оси
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(OX, OY); ctx.lineTo(OX + PW, OY); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        for (var t = -3; t <= 3; t += 1) {
          ctx.beginPath(); ctx.moveTo(sx(t), OY); ctx.lineTo(sx(t), OY + 5); ctx.stroke();
          ctx.fillText(t.toFixed(0), sx(t), OY + 20);
        }
        ctx.fillText("оценённый эффект, процентные пункты", OX + PW / 2, OY + 44);

        // столбики
        var bw = PW / BINS;
        for (i = 0; i < BINS; i += 1) {
          if (!hAll[i]) continue;
          var x0 = OX + i * bw;
          ctx.fillStyle = "rgba(49,95,140,0.30)";
          ctx.fillRect(x0, sy(hAll[i]), bw - 0.8, OY - sy(hAll[i]));
          if (hRej[i]) {
            ctx.fillStyle = "rgba(185,74,59,0.75)";
            ctx.fillRect(x0, sy(hRej[i]), bw - 0.8, OY - sy(hRej[i]));
          }
        }

        // истинный эффект
        ctx.strokeStyle = C.green || "#38735d"; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(sx(effect), OY - PH - 14); ctx.lineTo(sx(effect), OY); ctx.stroke();
        ctx.fillStyle = C.green || "#38735d"; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("истинный эффект " + effect.toFixed(1), sx(effect) + 6, OY - PH - 4);

        // средняя оценка среди «победителей»
        if (rej > 0) {
          var mRej = sumRej / rej;
          ctx.strokeStyle = C.gold || "#a57920"; ctx.lineWidth = 2.2;
          ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(sx(mRej), OY - PH - 14); ctx.lineTo(sx(mRej), OY); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = C.gold || "#a57920"; ctx.textAlign = "right";
          ctx.fillText("среднее у «значимых» " + mRej.toFixed(2), sx(mRej) - 6, OY - PH + 12);
        }

        // ноль
        ctx.strokeStyle = "rgba(23,25,21,0.45)"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(sx(0), OY); ctx.lineTo(sx(0), OY - PH - 14); ctx.stroke();

        var rate = rej / res.length;
        var label = effect === 0 ? "доля ложных открытий" : "мощность (доля отклонений H₀)";
        var expected = effect === 0 ? "ожидалось 5%" : "";
        var inflate = rej > 0 && effect > 0 ? ((sumRej / rej) / effect) : 0;
        out.set([
          { label: label, value: (rate * 100).toFixed(1) + "%" + (expected ? "  (" + expected + ")" : ""), color: effect === 0 && rate > 0.08 ? C.red : C.blue },
          { label: "средняя оценка по всем 400", value: (sumAll / res.length).toFixed(3) + " п.п.", color: C.blue },
          { label: rej ? "средняя оценка у «значимых»" : "значимых нет", value: rej ? (sumRej / rej).toFixed(3) + " п.п." + (inflate ? "  (×" + inflate.toFixed(1) + " от истины)" : "") : "—", color: C.gold }
        ]);

        ctx.textAlign = "left"; ctx.fillStyle = C.muted; ctx.font = "11.5px PT Sans, sans-serif";
        ctx.fillText("база " + (base * 100).toFixed(0) + "%, n = " + nPer + " на группу, " +
          (rule === "peek" ? LOOKS + " промежуточных проверок с остановкой" : "одна проверка в конце") +
          ", 400 повторов", OX, 26);
        ctx.fillStyle = "rgba(185,74,59,0.9)";
        ctx.fillText("красные — повторы, объявленные значимыми", OX, 46);
      }

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
