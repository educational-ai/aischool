// Lesson 29: pooling-lab — max/avg pooling buys shift-invariance on a real digit.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("pooling-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 480;

      var DIG = {"w":32,"h":32,"px":[0,0,0,0,0,14,40,65,90,116,144,173,202,231,234,227,220,212,195,152,108,65,22,13,9,6,2,0,0,0,0,0,0,7,13,20,26,42,66,89,113,136,152,169,185,202,208,210,211,213,202,160,118,75,33,22,16,9,3,0,0,0,0,0,0,13,26,39,52,70,92,114,136,155,160,164,169,173,182,193,203,214,210,168,127,85,43,30,22,13,5,0,0,0,0,0,0,20,39,59,78,98,118,139,159,175,167,160,152,144,156,176,195,215,217,176,136,95,54,39,28,17,6,0,0,0,0,0,0,26,52,78,104,126,145,163,182,195,175,155,135,115,130,158,187,216,225,185,145,105,65,48,34,21,8,0,0,0,0,0,0,26,52,78,104,124,139,154,169,180,164,147,131,115,130,158,187,215,223,183,142,102,61,45,32,20,7,0,0,0,0,0,0,21,42,63,85,99,110,120,130,139,138,137,136,135,150,171,192,213,216,174,131,89,47,33,24,15,5,0,0,0,0,0,0,16,33,49,65,75,81,86,91,98,113,127,142,156,170,184,197,211,208,164,121,77,33,22,16,9,3,0,0,0,0,0,0,11,23,34,46,51,51,52,52,57,87,117,147,177,190,196,203,209,200,155,110,64,19,10,7,4,2,0,0,0,0,0,0,7,14,21,28,29,26,22,19,23,66,109,153,196,208,207,207,206,193,146,100,54,7,0,0,0,0,0,0,0,0,0,0,5,11,16,21,23,22,22,21,26,70,115,159,203,213,209,205,202,186,142,98,54,10,3,2,1,1,0,0,0,0,0,0,4,7,11,15,17,19,21,22,30,75,120,165,210,218,211,204,197,180,138,97,55,14,6,4,3,1,0,0,0,0,0,0,2,4,6,8,12,16,20,24,34,80,125,171,217,223,212,202,192,173,134,95,56,17,9,7,4,1,0,0,0,0,0,0,0,1,1,2,6,12,19,25,37,84,131,177,224,227,214,200,187,167,130,94,57,20,12,9,5,2,0,0,0,0,0,0,0,0,0,0,3,9,16,22,32,72,112,152,192,199,193,188,182,169,139,109,79,49,36,27,17,8,2,2,1,1,0,0,0,0,0,0,3,7,11,16,24,54,84,114,144,157,164,170,177,175,153,131,109,87,69,51,34,16,6,4,3,1,0,0,0,0,0,0,2,4,7,10,15,36,56,76,97,116,134,153,172,181,167,153,139,125,101,76,50,24,9,7,4,2,0,0,0,0,0,0,1,2,3,4,7,17,28,38,49,74,105,136,167,187,181,175,170,164,134,100,66,33,12,9,6,3,0,0,0,0,0,0,0,0,0,0,0,4,7,11,14,42,79,116,153,181,183,185,187,189,157,120,82,45,21,16,10,5,0,0,0,0,0,0,0,0,0,0,0,3,5,8,11,32,61,90,120,143,152,162,171,181,156,126,96,66,44,33,22,11,0,0,0,0,0,0,0,0,0,0,0,2,4,5,7,23,44,65,86,106,122,139,156,172,155,132,109,87,66,50,33,17,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,13,26,39,52,68,92,116,140,164,154,139,123,108,89,67,45,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,9,13,18,31,62,93,125,156,153,145,137,129,112,84,56,28,0,0,0,0,0,0,3,9,14,20,24,21,19,16,13,15,19,22,26,38,70,102,135,167,164,154,145,136,118,88,59,29,0,0,0,0,0,0,7,19,31,43,53,47,41,34,28,29,33,37,40,52,84,117,149,181,176,164,153,141,121,91,61,30,0,0,0,0,0,0,11,29,48,66,81,72,62,53,44,44,47,51,54,66,99,131,163,196,188,174,160,146,124,93,62,31,0,0,0,0,0,0,14,39,64,90,109,97,84,72,59,58,62,65,69,81,113,145,178,210,201,184,167,151,128,96,64,32,0,0,0,0,0,0,16,45,73,102,124,113,103,92,81,80,83,86,90,100,128,156,184,211,199,179,160,141,117,88,59,29,0,0,0,0,0,0,16,43,71,99,122,119,116,114,111,112,114,117,119,126,143,160,178,195,178,155,133,110,88,66,44,22,0,0,0,0,0,0,15,42,69,96,120,125,130,136,141,144,145,147,148,152,158,165,172,178,157,131,105,80,59,44,29,15,0,0,0,0,0,0,15,41,67,93,117,131,144,157,171,175,176,177,178,177,173,170,166,162,136,107,78,49,29,22,15,7,0,0,0,0,0,0,14,40,65,90,115,136,158,179,201,207,207,207,207,203,189,174,160,145,116,83,51,19,0,0,0,0,0]};  // 32x32 upscaled real handwritten digit (sklearn load_digits)
      var iw = DIG.w, ih = DIG.h, px = DIG.px;

      var state = { shift: 0, mode: "max" };

      K.hint(
        root,
        "Настоящая рукописная цифра. Двигайте её вбок по пикселю и следите за двумя числами внизу: карта краёв меняется сильно, а та же карта после max-пулинга — заметно слабее. Так пулинг покупает устойчивость к небольшому сдвигу, огрубляя положение.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Слева — цифра, сдвинутая на выбранное число пикселей; справа — карта после пулинга 2×2. Изменение считается как относительная норма разности с несдвинутым образцом. Max-пулинг берёт в каждом окне 2×2 сильнейшее значение, average — среднее.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Рукописная цифра и её карта после пулинга при сдвиге",
        onResize: draw,
        drag: false,
      });

      function shifted(dx) {
        var out = new Float32Array(iw * ih);
        for (var y = 0; y < ih; y += 1) for (var x = 0; x < iw; x += 1) {
          var sx = x - dx; var v = (sx >= 0 && sx < iw) ? px[y * iw + sx] : 0;
          out[y * iw + x] = v / 255;
        }
        return out;
      }
      // Sobel edge magnitude — the "feature map" we pool (matches the lesson figure)
      function edge(a) {
        var out = new Float32Array(iw * ih), mx = 1e-6;
        function at(y, x) { y = y < 0 ? 0 : y >= ih ? ih - 1 : y; x = x < 0 ? 0 : x >= iw ? iw - 1 : x; return a[y * iw + x]; }
        for (var y = 0; y < ih; y += 1) for (var x = 0; x < iw; x += 1) {
          var gx = at(y - 1, x + 1) + 2 * at(y, x + 1) + at(y + 1, x + 1) - at(y - 1, x - 1) - 2 * at(y, x - 1) - at(y + 1, x - 1);
          var gy = at(y + 1, x - 1) + 2 * at(y + 1, x) + at(y + 1, x + 1) - at(y - 1, x - 1) - 2 * at(y - 1, x) - at(y - 1, x + 1);
          var m = Math.sqrt(gx * gx + gy * gy); out[y * iw + x] = m; if (m > mx) mx = m;
        }
        for (var i = 0; i < out.length; i += 1) out[i] /= mx;
        return out;
      }
      function pool(a, mode) {
        var ow = iw >> 1, oh = ih >> 1, out = new Float32Array(ow * oh);
        for (var y = 0; y < oh; y += 1) for (var x = 0; x < ow; x += 1) {
          var v00 = a[(2 * y) * iw + 2 * x], v01 = a[(2 * y) * iw + 2 * x + 1];
          var v10 = a[(2 * y + 1) * iw + 2 * x], v11 = a[(2 * y + 1) * iw + 2 * x + 1];
          out[y * ow + x] = mode === "max" ? Math.max(v00, v01, v10, v11) : (v00 + v01 + v10 + v11) / 4;
        }
        return { d: out, w: ow, h: oh };
      }
      function relChange(a, b) {
        var num = 0, den = 0;
        for (var i = 0; i < a.length; i += 1) { var d = b[i] - a[i]; num += d * d; den += a[i] * a[i]; }
        return Math.sqrt(num) / (Math.sqrt(den) + 1e-9);
      }
      function ramp(t) {
        t = Math.max(0, Math.min(1, t));
        var st = [[0, 0, 4], [40, 11, 84], [139, 26, 108], [222, 73, 104], [251, 159, 58], [252, 253, 191]];
        var f = t * (st.length - 1), i = Math.floor(f), r = f - i;
        if (i >= st.length - 1) return st[st.length - 1];
        var a = st[i], b = st[i + 1];
        return [a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r, a[2] + (b[2] - a[2]) * r];
      }
      function blit(ctx, fill, ww, hh, dx, dy, dw, dh) {
        var off = document.createElement("canvas"); off.width = ww; off.height = hh;
        var octx = off.getContext("2d"); var id = octx.createImageData(ww, hh);
        for (var y = 0; y < hh; y += 1) for (var x = 0; x < ww; x += 1) {
          var col = fill(y, x); var o = (y * ww + x) * 4;
          id.data[o] = col[0]; id.data[o + 1] = col[1]; id.data[o + 2] = col[2]; id.data[o + 3] = 255;
        }
        octx.putImageData(id, 0, 0);
        ctx.imageSmoothingEnabled = false; ctx.drawImage(off, dx, dy, dw, dh);
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "14px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        var cur = shifted(state.shift);
        var base = shifted(0);
        var ecur = edge(cur), ebase = edge(base);
        var boxW = 300, oy = 60, ox = 90, rx = 640;
        blit(ctx, function (y, x) { var v = Math.round(cur[y * iw + x] * 255); return [v, v, v]; }, iw, ih, ox, oy, boxW, boxW);
        var pc = pool(ecur, state.mode), pb = pool(ebase, state.mode);
        blit(ctx, function (y, x) { return ramp(pc.d[y * pc.w + x]); }, pc.w, pc.h, rx, oy, boxW, boxW);
        ctx.strokeStyle = C.line; ctx.strokeRect(ox, oy, boxW, boxW); ctx.strokeRect(rx, oy, boxW, boxW);
        ctx.fillStyle = C.ink; ctx.textAlign = "center";
        ctx.fillText("цифра, сдвиг " + state.shift + " пикс.", ox + boxW / 2, oy - 14);
        ctx.fillText((state.mode === "max" ? "max" : "average") + "-пулинг карты краёв  (" + pc.w + "×" + pc.h + ")", rx + boxW / 2, oy - 14);

        var rawChange = relChange(ebase, ecur);
        var poolChange = relChange(pb.d, pc.d);
        var factor = poolChange > 1e-6 ? (rawChange / poolChange) : 0;
        output.set([
          { label: "Сдвиг", value: state.shift + " пикс.", color: C.ink },
          { label: "Изменение карты краёв", value: (rawChange * 100).toFixed(0) + "%", color: C.red },
          { label: "Изменение после пулинга", value: (poolChange * 100).toFixed(0) + "%", color: C.green },
          { label: "Пулинг устойчивее в", value: state.shift === 0 ? "— (нет сдвига)" : factor.toFixed(1) + " раза", color: C.blue },
        ]);
      }

      K.segmented(controls, {
        label: "Тип пулинга",
        value: state.mode,
        options: [{ value: "max", label: "max" }, { value: "avg", label: "average" }],
      }, function (v) { state.mode = v; draw(); });

      K.slider(controls, { label: "Сдвиг цифры, пикс.", min: -6, max: 6, step: 1, value: state.shift,
        format: function (v) { return String(v); } }, function (v) { state.shift = v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
