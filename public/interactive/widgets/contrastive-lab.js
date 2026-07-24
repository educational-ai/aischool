// Lesson 84: contrastive-lab — общее пространство на окружности.
// Тащите снимки и подписи, меняйте температуру, включайте вторую верную подпись
// и делайте шаги обучения: видно, кто кого притягивает и кого зря отталкивает.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("contrastive-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var CX = 232, CY = 238, R = 152;
      var PB = "#fffef9";
      var N = 5, tau = 0.1, multi = false;
      var img = [], cap = [], drag = null;

      function reset() {
        img = []; cap = [];
        for (var i = 0; i < N; i += 1) {
          var a = -Math.PI / 2 + (2 * Math.PI * i) / N;
          img.push(a);
          cap.push(a + 0.28);
        }
        // вторая подпись подозрительно близка к первому снимку: кандидат в ложные negatives
        if (N > 1) cap[1] = img[0] + 0.42;
      }

      function posMask(i) {
        var m = [], j;
        for (j = 0; j < N; j += 1) m.push(i === j ? 1 : 0);
        if (multi && i === 0 && N > 1) m[1] = 1;
        return m;
      }

      function rowProbs(i) {
        var s = [], j, mx = -Infinity;
        for (j = 0; j < N; j += 1) {
          s.push(Math.cos(img[i] - cap[j]) / tau);
          if (s[j] > mx) mx = s[j];
        }
        var sum = 0, p = [];
        for (j = 0; j < N; j += 1) { p.push(Math.exp(s[j] - mx)); sum += p[j]; }
        for (j = 0; j < N; j += 1) p[j] /= sum;
        return p;
      }

      function rowLoss(i) {
        var p = rowProbs(i), m = posMask(i), acc = 0, j;
        for (j = 0; j < N; j += 1) if (m[j]) acc += p[j];
        return -Math.log(Math.max(acc, 1e-12));
      }

      function meanLoss() {
        var acc = 0, i;
        for (i = 0; i < N; i += 1) acc += rowLoss(i);
        return acc / N;
      }

      function hardestShare(i) {
        var p = rowProbs(i), m = posMask(i), best = 0, sum = 0, j;
        for (j = 0; j < N; j += 1) {
          if (m[j]) continue;
          sum += p[j];
          if (p[j] > best) best = p[j];
        }
        return sum > 1e-12 ? best / sum : 0;
      }

      function step(times) {
        var lr = 0.08, t, i, j;
        for (t = 0; t < times; t += 1) {
          var gi = new Array(N), gc = new Array(N);
          for (i = 0; i < N; i += 1) { gi[i] = 0; gc[i] = 0; }
          for (i = 0; i < N; i += 1) {
            var p = rowProbs(i), m = posMask(i), npos = 0;
            for (j = 0; j < N; j += 1) npos += m[j];
            for (j = 0; j < N; j += 1) {
              // dL/ds_ij = p_ij - y_ij, s_ij = cos(img_i - cap_j)/tau
              var g = (p[j] - (m[j] ? 1 / npos : 0)) / tau;
              var d = Math.sin(img[i] - cap[j]);
              gi[i] += -g * d;
              gc[j] += g * d;
            }
          }
          for (i = 0; i < N; i += 1) {
            img[i] -= lr * gi[i] / N;
            cap[i] -= lr * gc[i] / N;
          }
        }
      }

      K.hint(root, "Косинусная близость после нормировки — это только угол, поэтому всё пространство помещается на окружности. Круги — снимки, квадраты — подписи. Тащите их мышью, меняйте температуру и нажимайте «шаг обучения». Стрелки показывают силы для первого снимка: зелёная тянет к своей подписи, красные отталкивают соперников. Включите вторую верную подпись — и посмотрите, что делает с ней обычный loss.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var out = K.readout(root);
      K.caption(root, "Малая температура делает softmax резким: почти весь штраф достаётся одному ближайшему сопернику. Если этот соперник на самом деле подходит запросу (ложный negative), обучение расталкивает то, что должно стоять рядом. Режим «две верных подписи» переводит числитель в multi-positive и снимает эту силу.");
      var cs = K.makeCanvas(stage, W, H, { label: "Общее пространство на окружности и softmax строки", onResize: draw, drag: false });
      reset();

      function px(a) { return [CX + R * Math.cos(a), CY + R * Math.sin(a)]; }

      function arrow(ctx, x, y, dx, dy, color) {
        var len = Math.hypot(dx, dy);
        if (len < 1.5) return;
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx, y + dy); ctx.stroke();
        var ux = dx / len, uy = dy / len;
        ctx.beginPath();
        ctx.moveTo(x + dx, y + dy);
        ctx.lineTo(x + dx - 8 * ux + 4 * uy, y + dy - 8 * uy - 4 * ux);
        ctx.lineTo(x + dx - 8 * ux - 4 * uy, y + dy - 8 * uy + 4 * ux);
        ctx.closePath(); ctx.fill();
      }

      function draw() {
        var ctx = cs.ctx, i, j, p0 = rowProbs(0), m0 = posMask(0);
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic"; ctx.textAlign = "center";
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(CX, CY, R, 0, 7); ctx.stroke();

        // силы для строки 0
        var q0 = px(img[0]);
        for (j = 0; j < N; j += 1) {
          var pj = px(cap[j]);
          var force = (p0[j] - (m0[j] ? 1 : 0));
          var tx = -Math.sin(cap[j]), ty = Math.cos(cap[j]);
          var dir = Math.sin(img[0] - cap[j]) >= 0 ? 1 : -1;
          var mag = Math.min(70, Math.abs(force) * 150);
          var col = m0[j] ? C.green : C.red;
          var sgn = m0[j] ? dir : -dir;
          arrow(ctx, pj[0], pj[1], sgn * tx * mag, sgn * ty * mag, col);
          ctx.strokeStyle = m0[j] ? "rgba(56,115,93,0.55)" : "rgba(185,74,59,0.18)";
          ctx.lineWidth = m0[j] ? 2 : 1;
          ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(pj[0], pj[1]); ctx.stroke();
        }

        for (i = 0; i < N; i += 1) {
          var a = px(img[i]);
          ctx.fillStyle = i === 0 ? C.blue : "rgba(49,95,140,0.45)";
          ctx.beginPath(); ctx.arc(a[0], a[1], i === 0 ? 10 : 7, 0, 7); ctx.fill();
          ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.stroke();
          ctx.fillStyle = C.ink || "#171915";
          ctx.fillText("I" + (i + 1), a[0] + 20 * Math.cos(img[i]), a[1] + 20 * Math.sin(img[i]) + 4);
          var b = px(cap[i]);
          ctx.fillStyle = m0[i] && i !== 0 ? C.green : (i === 0 ? C.gold : "rgba(165,121,32,0.5)");
          ctx.fillRect(b[0] - 7, b[1] - 7, 14, 14);
          ctx.strokeStyle = PB; ctx.lineWidth = 1.2; ctx.strokeRect(b[0] - 7, b[1] - 7, 14, 14);
          ctx.fillStyle = C.ink || "#171915";
          ctx.fillText("T" + (i + 1), b[0] + 22 * Math.cos(cap[i]), b[1] + 22 * Math.sin(cap[i]) + 4);
        }
        ctx.fillStyle = C.muted;
        ctx.fillText("общее пространство: важен только угол", CX, CY + R + 46);

        // правая панель: softmax строки 1
        var BX = 500, BY = 372, BW = 340, BH = 250;
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(BX, BY - BH); ctx.lineTo(BX, BY); ctx.lineTo(BX + BW, BY); ctx.stroke();
        var bw = BW / N * 0.6;
        for (j = 0; j < N; j += 1) {
          var x = BX + (BW / N) * (j + 0.5), h = p0[j] * BH;
          ctx.fillStyle = m0[j] ? C.green : C.red;
          ctx.fillRect(x - bw / 2, BY - h, bw, h);
          ctx.fillStyle = C.muted;
          ctx.fillText("T" + (j + 1), x, BY + 16);
          ctx.fillStyle = C.ink || "#171915";
          ctx.fillText(p0[j].toFixed(2), x, BY - h - 6);
        }
        ctx.fillStyle = C.muted;
        ctx.fillText("softmax строки I1: чем ниже τ, тем острее пик", BX + BW / 2, BY - BH - 12);

        var pos = 0;
        for (j = 0; j < N; j += 1) if (m0[j]) pos += p0[j];
        out.set([
          { label: "средний loss по строкам", value: meanLoss().toFixed(3), color: C.blue },
          { label: "вероятность верного (строка I1)", value: pos.toFixed(3), color: C.green },
          { label: "доля отталкивания у ближайшего соперника", value: (hardestShare(0) * 100).toFixed(1) + "%", color: C.red },
          { label: "температура τ", value: tau.toFixed(2), color: C.gold }
        ]);
      }

      cs.canvas.addEventListener("mousedown", function (ev) {
        var rect = cs.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        var my = (ev.clientY - rect.top) / rect.height * H;
        var i;
        for (i = 0; i < N; i += 1) {
          var a = px(img[i]);
          if (Math.hypot(mx - a[0], my - a[1]) < 16) { drag = { kind: "i", idx: i }; return; }
          var b = px(cap[i]);
          if (Math.hypot(mx - b[0], my - b[1]) < 16) { drag = { kind: "c", idx: i }; return; }
        }
      });
      function move(ev) {
        if (!drag) return;
        var rect = cs.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        var my = (ev.clientY - rect.top) / rect.height * H;
        var ang = Math.atan2(my - CY, mx - CX);
        if (drag.kind === "i") img[drag.idx] = ang; else cap[drag.idx] = ang;
        draw();
      }
      function up() { drag = null; }
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);

      K.slider(controls, {
        label: "температура τ", min: 0.02, max: 1, step: 0.02, value: tau,
        format: function (v) { return v.toFixed(2); }
      }, function (v) { tau = v; draw(); });
      K.slider(controls, {
        label: "число пар B", min: 3, max: 8, step: 1, value: N,
        format: function (v) { return String(v); }
      }, function (v) { N = Math.round(v); reset(); draw(); });
      K.segmented(controls, {
        label: "верные подписи для I1", value: "одна",
        options: [{ label: "одна", value: "одна" }, { label: "две (multi-positive)", value: "две" }]
      }, function (v) { multi = v === "две"; draw(); });

      var b1 = K.element("button", "kontur-int-segment", { type: "button", text: "шаг обучения ×20" });
      var b2 = K.element("button", "kontur-int-segment", { type: "button", text: "сброс" });
      b1.style.margin = b2.style.margin = "0 6px";
      b1.addEventListener("click", function () { step(20); draw(); });
      b2.addEventListener("click", function () { reset(); draw(); });
      controls.appendChild(b1); controls.appendChild(b2);

      draw();
      return function () {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        cs.destroy();
      };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
