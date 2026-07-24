// Lesson 76: attention-lab — one head, live. Pick a query, watch scores turn into
// competing weights, see the output as a convex mix of VALUES (not keys), and feel
// what the 1/sqrt(d_k) scaling, the causal mask and extra distractor tokens do.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("attention-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 500;
      var PB = "#fffef9";

      var BASE = ["робот", "положил", "стакан", "на", "стол", "потому", "что", "он"];
      var VAL0 = [-1.6, -0.8, 0.6, -0.2, 1.8, 0.2, -0.4, 1.0];
      var QUERIES = [
        { name: "он", idx: 7, s: [0.4, 0.2, 1.4, 0.0, 1.9, 0.3, 0.2, 0.9] },
        { name: "стакан", idx: 2, s: [1.2, 1.5, 1.6, 0.1, 0.9, 0.0, 0.1, 0.5] },
        { name: "потому", idx: 5, s: [0.3, 0.9, 0.4, 0.0, 0.5, 1.3, 1.1, 0.6] }
      ];
      var DK = [4, 16, 64, 256];
      var qi = 0, dkIdx = 2, scaled = 1, bump = 0, causal = 0, distract = 0, swapped = 0;

      function tokens() {
        var t = BASE.slice(), i;
        if (distract) for (i = 1; i <= 6; i += 1) t.push("шум" + i);
        return t;
      }
      function values() {
        var v = VAL0.slice(), i;
        v[4] = swapped ? -1.5 : 1.8;
        if (distract) for (i = 0; i < 6; i += 1) v.push(0);
        return v;
      }
      function rawScores() {
        var q = QUERIES[qi], s = q.s.slice(), i;
        s[4] += bump;
        if (distract) for (i = 0; i < 6; i += 1) s.push(0.5);
        return s;
      }
      function state() {
        var dk = DK[dkIdx], q = QUERIES[qi], s = rawScores(), n = s.length, i;
        // the underlying dot product grows like sqrt(d_k); scaling divides it back
        var logits = [];
        for (i = 0; i < n; i += 1) {
          var raw = s[i] * Math.sqrt(dk);
          logits.push(scaled ? raw / Math.sqrt(dk) : raw);
        }
        var allowed = [];
        for (i = 0; i < n; i += 1) allowed.push(causal ? (i <= q.idx) : true);
        var mx = -1e9;
        for (i = 0; i < n; i += 1) if (allowed[i] && logits[i] > mx) mx = logits[i];
        var e = [], sum = 0;
        for (i = 0; i < n; i += 1) { e.push(allowed[i] ? Math.exp(logits[i] - mx) : 0); sum += e[i]; }
        var a = [], ent = 0, mw = 0, out = 0, v = values();
        for (i = 0; i < n; i += 1) {
          a.push(e[i] / sum);
          if (a[i] > 1e-12) ent -= a[i] * Math.log(a[i]);
          if (a[i] > mw) mw = a[i];
          out += a[i] * v[i];
        }
        return { a: a, logits: logits, allowed: allowed, out: out, ent: ent, mw: mw, n: n, v: v, dk: dk };
      }

      K.hint(root, "Одна голова внимания вживую. Запрос сравнивается с ключами всех слов, softmax превращает совместимость в конкурирующие веса, а выход собирается из VALUE. Поднимите score одного ключа — и увидите, что чужие веса падают, хотя их scores не менялись. Отключите деление на корень размерности при большом d_k — распределение схлопнется в одну позицию. Добавьте шумовые токены — вес опоры упадёт от одного роста знаменателя.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var buttons = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Верх: scores (совместимость) и веса (доля бюджета внимания). Низ: ось value — выход всегда лежит внутри отрезка между переносимыми значениями, потому что веса неотрицательны и дают в сумме единицу. Кнопка «подменить value у слова стол» меняет содержимое, не трогая адресацию: веса те же, выход другой.");
      var cs = K.makeCanvas(stage, W, H, { label: "Scores, веса внимания и выход как смесь value", onResize: draw });

      function draw() {
        var ctx = cs.ctx, st = state(), i;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var toks = tokens(), n = st.n;
        var L = 60, R = W - 30, colW = (R - L) / n;
        var yS = 150, yW = 320;          // baselines of the two bar groups
        var hS = 90, hW = 120;

        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(L, yS); ctx.lineTo(R, yS); ctx.moveTo(L, yW); ctx.lineTo(R, yW); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("scores  s_j = q·k_j" + (scaled ? " / √d_k" : "  (без деления)"), L, yS - hS - 14);
        ctx.fillText("веса  a_j = softmax(s)_j", L, yW - hW - 14);

        var maxAbs = 0.6;
        for (i = 0; i < n; i += 1) maxAbs = Math.max(maxAbs, Math.abs(st.logits[i]));
        for (i = 0; i < n; i += 1) {
          var x = L + i * colW + colW * 0.16, bw = colW * 0.68;
          var blocked = !st.allowed[i];
          var isAnchor = (i === 4), isQuery = (i === QUERIES[qi].idx);
          var col = blocked ? "#c9c8be" : (isAnchor ? C.red : (isQuery ? C.gold : C.blue));
          // scores
          var hs = Math.min(hS, Math.abs(st.logits[i]) / maxAbs * hS);
          ctx.fillStyle = blocked ? "rgba(150,153,144,0.35)" : col;
          ctx.globalAlpha = 0.55;
          ctx.fillRect(x, st.logits[i] >= 0 ? yS - hs : yS, bw, hs);
          ctx.globalAlpha = 1;
          // weights
          var hw = st.a[i] * hW;
          ctx.fillStyle = blocked ? "rgba(150,153,144,0.3)" : col;
          ctx.fillRect(x, yW - hw, bw, hw);
          ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "10.5px PT Sans, sans-serif";
          if (st.a[i] >= 0.02) ctx.fillText(st.a[i].toFixed(2), x + bw / 2, yW - hw - 5);
          ctx.fillStyle = blocked ? "#969990" : C.ink || "#171915";
          ctx.font = (isQuery ? "bold " : "") + "11px PT Sans, sans-serif";
          ctx.save();
          ctx.translate(x + bw / 2, yW + 14);
          if (n > 10) { ctx.rotate(-0.5); ctx.textAlign = "right"; }
          ctx.fillText(toks[i], 0, 0);
          ctx.restore();
          if (blocked) {
            ctx.strokeStyle = "#b94a3b"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, yW - 6); ctx.lineTo(x + bw, yW - 26); ctx.stroke();
          }
        }
        ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif"; ctx.fillStyle = C.gold;
        ctx.fillText("запрос: «" + QUERIES[qi].name + "»", L, 26);
        ctx.fillStyle = C.red;
        ctx.fillText("красный столбец — ключ «стол», его score двигает ползунок", L + 190, 26);

        // ---- value axis
        var vy = 430, vL = L + 40, vR = R - 40;
        var vmin = -2.2, vmax = 2.2;
        function vx(v) { return vL + (v - vmin) / (vmax - vmin) * (vR - vL); }
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(vL, vy); ctx.lineTo(vR, vy); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.font = "11.5px PT Sans, sans-serif";
        ctx.fillText("ось value: переносимое содержимое", vL, vy - 44);
        var lo = 9, hi = -9;
        for (i = 0; i < n; i += 1) {
          if (!st.allowed[i]) continue;
          lo = Math.min(lo, st.v[i]); hi = Math.max(hi, st.v[i]);
          ctx.fillStyle = i === 4 ? C.red : "rgba(49,95,140,0.75)";
          ctx.beginPath(); ctx.arc(vx(st.v[i]), vy, 4 + 8 * st.a[i], 0, 7); ctx.fill();
        }
        ctx.strokeStyle = "rgba(110,114,106,0.6)"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(vx(lo), vy + 16); ctx.lineTo(vx(hi), vy + 16); ctx.stroke();
        ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.font = "10.5px PT Sans, sans-serif";
        ctx.fillText("выпуклая оболочка доступных value", vx((lo + hi) / 2), vy + 32);
        ctx.fillStyle = C.green;
        ctx.beginPath();
        ctx.moveTo(vx(st.out), vy - 12); ctx.lineTo(vx(st.out) - 8, vy - 26); ctx.lineTo(vx(st.out) + 8, vy - 26);
        ctx.closePath(); ctx.fill();
        ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("выход z = " + st.out.toFixed(3), vx(st.out), vy - 32);

        var anchor = st.a[4], unif = 1 / n;
        output.set([
          { label: "Максимальный вес", value: st.mw.toFixed(3), color: C.red },
          { label: "Энтропия карты (макс " + Math.log(n).toFixed(2) + ")", value: st.ent.toFixed(3), color: C.blue },
          { label: "Вес ключа «стол»", value: anchor.toFixed(3) + " (равномерно " + unif.toFixed(3) + ")", color: C.gold },
          { label: "Выход z", value: st.out.toFixed(3), color: C.green }
        ]);
      }

      K.segmented(controls, {
        label: "Запрос", value: 0,
        options: [{ label: "«он»", value: 0 }, { label: "«стакан»", value: 1 }, { label: "«потому»", value: 2 }]
      }, function (v) { qi = v; draw(); });
      K.segmented(controls, {
        label: "Масштаб logits", value: 1,
        options: [{ label: "делим на √d_k", value: 1 }, { label: "без деления", value: 0 }]
      }, function (v) { scaled = v; draw(); });
      K.slider(controls, {
        label: "Размерность d_k", min: 0, max: 3, step: 1, value: 2,
        format: function (v) { return String(DK[v]); }
      }, function (v) { dkIdx = v; draw(); });
      K.slider(controls, {
        label: "Прибавка к score ключа «стол»", min: -2, max: 4, step: 0.1, value: 0,
        format: function (v) { return (v >= 0 ? "+" : "") + Number(v).toFixed(1); }
      }, function (v) { bump = Number(v); draw(); });

      function toggle(text, get, set) {
        var b = K.element("button", "kontur-int-segment", { type: "button", text: text(get()) });
        b.style.margin = "0 6px";
        b.addEventListener("click", function () { set(get() ? 0 : 1); b.textContent = text(get()); draw(); });
        buttons.appendChild(b);
        return b;
      }
      toggle(function (v) { return v ? "causal mask: включена" : "causal mask: выключена"; },
        function () { return causal; }, function (v) { causal = v; });
      toggle(function (v) { return v ? "убрать 6 шумовых токенов" : "добавить 6 шумовых токенов"; },
        function () { return distract; }, function (v) { distract = v; });
      toggle(function (v) { return v ? "вернуть value слова «стол»" : "подменить value слова «стол»"; },
        function () { return swapped; }, function (v) { swapped = v; });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
