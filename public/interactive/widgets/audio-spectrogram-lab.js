// Lesson 40: audio-spectrogram-lab — build a signal, run a REAL FFT/STFT in the browser, see the true spectrogram.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("audio-spectrogram-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470, FS = 8000, NS = 4096;
      var preset = 0, win = 128;

      // ---- iterative radix-2 FFT (in place) ----
      function fft(re, im) {
        var n = re.length, j = 0, i, k, m;
        for (i = 1; i < n; i += 1) {
          var bit = n >> 1;
          for (; j & bit; bit >>= 1) j ^= bit;
          j ^= bit;
          if (i < j) { var tr = re[i]; re[i] = re[j]; re[j] = tr; var ti = im[i]; im[i] = im[j]; im[j] = ti; }
        }
        for (var len = 2; len <= n; len <<= 1) {
          var ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
          for (i = 0; i < n; i += len) {
            var cr = 1, ci = 0;
            for (k = 0; k < len / 2; k += 1) {
              var a = i + k, b = i + k + len / 2;
              var xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
              re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
              var ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
            }
          }
        }
      }

      function signal() {
        var x = new Float64Array(NS), t;
        for (var n = 0; n < NS; n += 1) {
          t = n / FS;
          if (preset === 0) x[n] = Math.sin(2 * Math.PI * 500 * t) + 0.8 * Math.sin(2 * Math.PI * 900 * t);
          else if (preset === 1) { x[n] = Math.sin(2 * Math.PI * 800 * t) + Math.sin(2 * Math.PI * 920 * t); if (n >= NS / 2 && n < NS / 2 + 24) x[n] += 4; }
          else if (preset === 2) { var f = 300 + (1800 - 300) * t / (NS / FS); x[n] = Math.sin(2 * Math.PI * f * t); }
          else { x[n] = Math.sin(2 * Math.PI * 600 * t) + 0.7 * (pseudo(n) * 2 - 1); }
        }
        return x;
      }
      function pseudo(n) { var s = Math.sin(n * 12.9898) * 43758.5453; return s - Math.floor(s); }

      function stft(x, nper) {
        var hop = nper / 2, bins = nper / 2, frames = Math.floor((NS - nper) / hop) + 1;
        var mag = [], hann = new Float64Array(nper);
        for (var i = 0; i < nper; i += 1) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (nper - 1));
        var re = new Float64Array(nper), im = new Float64Array(nper);
        for (var fr = 0; fr < frames; fr += 1) {
          for (var k = 0; k < nper; k += 1) { re[k] = x[fr * hop + k] * hann[k]; im[k] = 0; }
          fft(re, im);
          var col = new Float64Array(bins);
          for (var b = 0; b < bins; b += 1) col[b] = Math.sqrt(re[b] * re[b] + im[b] * im[b]);
          mag.push(col);
        }
        return { mag: mag, frames: frames, bins: bins };
      }

      function magmaColor(v) { // v in [0,1] -> dark->purple->orange->white
        v = Math.max(0, Math.min(1, v));
        var r = Math.min(255, 255 * Math.pow(v, 0.7) * 1.4);
        var g = Math.max(0, 255 * (v - 0.35) / 0.65) * Math.pow(v, 0.5);
        var bl = 60 * Math.sin(Math.PI * v) + 255 * Math.pow(v, 3);
        return [Math.round(r), Math.round(Math.max(0, g)), Math.round(Math.min(255, bl))];
      }

      K.hint(
        root,
        "Настоящее преобразование Фурье прямо в браузере. Соберите сигнал, и виджет посчитает реальную СТФТ — спектрограмму. Двигайте ширину окна: короткое окно точно ловит момент события, но смазывает близкие частоты; длинное — наоборот. Это фундаментальный компромисс времени и частоты.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Сверху — волна сигнала во времени. Снизу — спектрограмма: по горизонтали время, по вертикали частота, яркость = сила частоты. Всё посчитано настоящим быстрым преобразованием Фурье по перекрывающимся окнам Ханна.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Сигнал и его спектрограмма", onResize: draw, drag: false });

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";
        var x = signal();

        // waveform (top strip)
        var wx = 60, wy = 30, ww = W - 90, wh = 70;
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(wx, wy, ww, wh);
        ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.fillText("волна (время →)", wx, wy - 8);
        ctx.strokeStyle = C.blue; ctx.lineWidth = 1;
        ctx.beginPath();
        var step = Math.ceil(NS / ww);
        for (var i = 0; i < NS; i += step) {
          var px = wx + i / NS * ww, py = wy + wh / 2 - x[i] / 5 * wh;
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // spectrogram
        var S = stft(x, win);
        var sx = 60, sy = 130, sw = W - 90, sh = 300;
        var off = document.createElement("canvas"); off.width = S.frames; off.height = S.bins;
        var octx = off.getContext("2d"); var id = octx.createImageData(S.frames, S.bins);
        var mx = 0; for (var f = 0; f < S.frames; f += 1) for (var b = 0; b < S.bins; b += 1) if (S.mag[f][b] > mx) mx = S.mag[f][b];
        for (f = 0; f < S.frames; f += 1) for (b = 0; b < S.bins; b += 1) {
          var v = Math.log(1 + 8 * S.mag[f][b] / (mx + 1e-9)) / Math.log(9);
          var col = magmaColor(v);
          var o = ((S.bins - 1 - b) * S.frames + f) * 4;
          id.data[o] = col[0]; id.data[o + 1] = col[1]; id.data[o + 2] = col[2]; id.data[o + 3] = 255;
        }
        octx.putImageData(id, 0, 0);
        ctx.imageSmoothingEnabled = true; ctx.drawImage(off, sx, sy, sw, sh);
        ctx.strokeStyle = C.line; ctx.strokeRect(sx, sy, sw, sh);
        // axes labels
        ctx.fillStyle = C.muted; ctx.textAlign = "right";
        ctx.fillText((FS / 2) + " Гц", sx - 6, sy + 6); ctx.fillText("0", sx - 6, sy + sh);
        ctx.textAlign = "center"; ctx.fillText("время →", sx + sw / 2, sy + sh + 20);
        ctx.save(); ctx.translate(sx - 34, sy + sh / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("частота", 0, 0); ctx.restore();

        var winMs = (win / FS * 1000), df = FS / win;
        output.set([
          { label: "Ширина окна", value: win + " отсч. = " + winMs.toFixed(1) + " мс", color: C.ink },
          { label: "Разрешение по частоте", value: "≈ " + df.toFixed(0) + " Гц на корзину", color: C.blue },
          { label: "Компромисс", value: win <= 64 ? "точное время, грубая частота" : win >= 256 ? "точная частота, грубое время" : "баланс", color: C.gold },
        ]);
      }

      K.segmented(controls, { label: "Сигнал", value: 0, options: [
        { label: "два тона", value: 0 }, { label: "щелчок и тоны", value: 1 }, { label: "свип", value: 2 }, { label: "тон и шум", value: 3 } ] }, function (v) { preset = v; draw(); });
      K.slider(controls, { label: "Ширина окна СТФТ", min: 5, max: 9, step: 1, value: 7, format: function (v) { return (1 << v) + " отсч."; } }, function (v) { win = 1 << v; draw(); });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
