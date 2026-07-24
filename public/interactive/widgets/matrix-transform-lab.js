// Lesson 36: matrix-transform-lab — drag the matrix columns, warp a real photo, read det and singular values.
(function () {
  "use strict";

  var IMG = {"w":40,"h":48,"rows":["8888767765556567777898877777667666654433","8767656555566666788888777888676655655443","7876666544567778bba986677998999875555454","777676655677899aaabca7566667767897544544","766765556789bba9558bb8545556766665556665","878766777789abaa6348ba435777665555666665","77777678778aaba984479b646777766677666665","76766667899abca78858aa746789a97777667775","8766567889bbba94597588876678999767888777","8865456778abba8448a788787777889879888656","8767557768abbb84798987876789888777776555","97767756779ccb9abba9a9888889867666786565","a976765789bccbcdcaaaa9999777777778788655","998775679abccdddcaabbaab8888787677678755","99a8667888abcddcccccccccb889986668887777","a9b97556689cddcbccddddeeec9acb456aa87766","aaa9766678bcdcccddeeedeffe9afa35aec99898","aaab856779bddcdccdedc877796af837ea9dddcb","aaaa9556788acccccaa85444442be53ae6aeeddc","aba9b757777abccbb8543455442ad56d924468ac","abbaaa88778a9bcba85443553439c68a23433347","aabaaba978889aaa98645644555897a533245333","abaabbb98879a9789877556458a9a98644454233","aaababba9669989898788767addaa9aa64332233","6779bba986799888989988acddcbaabb85333344","45568999967a99778889989cccaabbcba8765544","665556789888876779a9988bea5779ddcb866544","699a7776664677999a998877b54444bcb9744555","8aaac735688888988aa998866466439a86656555","9abaaa86544324568aa9aa998423237655556665","4aabbbb73233445679999aabba64456665455566","18babbba8733333578889abbbb96334565555544","14aabbbb9772133346778abbbaa9755566555565","029abbaab87842334467799abaaaaa8744454345","1069bbb9aa779642433655579aba995455545566","1048aabaaba87985445343434689664344466676","103889aa99b98888735656610124445545567776","1137899aa7aa9987755456202222222223333343","1134889aa89a9988773762111111111111111112","2422889aa9699999788665211111111101111113","32127789aa679998879754311111111101111122","11136778aa759988887486211101111100101111","113457789a857988799349710101111000111112","233447789a956888689625982011111000111222","214337688aa65788667753599301121001112212","1251376789a75678667766459a51121111222224","15413676889856677677677459a7211122222333","341136767899556675775687469a831121122121"]};

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("matrix-transform-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var OX = 300, OY = 360, S = 108;         // math->screen: sx=OX+x*S, sy=OY-y*S (y up)
      var a1 = { x: 1, y: 0 }, a2 = { x: 0, y: 1 };  // columns of A
      var drag = null;

      // offscreen photo (grayscale) at native small size
      var photo = document.createElement("canvas"); photo.width = IMG.w; photo.height = IMG.h;
      (function () {
        var pctx = photo.getContext("2d"); var id = pctx.createImageData(IMG.w, IMG.h);
        for (var j = 0; j < IMG.h; j += 1) { var row = IMG.rows[j]; for (var i = 0; i < IMG.w; i += 1) {
          var v = parseInt(row[i], 16) / 15 * 255; var o = (j * IMG.w + i) * 4;
          id.data[o] = v; id.data[o + 1] = v; id.data[o + 2] = v; id.data[o + 3] = 255; } }
        pctx.putImageData(id, 0, 0);
      })();
      var photoData = photo.getContext("2d").getImageData(0, 0, IMG.w, IMG.h).data;

      // warp buffer (inverse gather)
      var RW = 200, RH = 200;
      var xmin = -1.0, xmax = 2.8, ymin = -1.0, ymax = 2.6;
      var disp = document.createElement("canvas"); disp.width = RW; disp.height = RH;
      var dctx = disp.getContext("2d");
      var buf = dctx.createImageData(RW, RH);

      K.hint(
        root,
        "Столбцы матрицы — это две стрелки: куда переходят базисные векторы. Перетаскивайте их концы и смотрите, как деформируется реальная фотография и координатная сетка. Определитель — во сколько раз меняется площадь; число обусловленности κ — во сколько эллипс вытянут.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Фотография занимает единичный квадрат; матрица A с колонками из двух стрелок переводит его в параллелограмм. Определитель det A = площадь параллелограмма со знаком. Сингулярные числа σ — полуоси эллипса, в который переходит окружность; κ = σ_max/σ_min.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Матрица деформирует фотографию и сетку", onResize: draw, drag: false });

      function m2s(x, y) { return [OX + x * S, OY - y * S]; }
      function svd2() {
        var a = a1.x, c = a1.y, b = a2.x, d = a2.y;
        // A^T A
        var p = a * a + c * c, q = a * b + c * d, r = b * b + d * d;
        var tr = p + r, det2 = p * r - q * q;
        var disc = Math.sqrt(Math.max(0, tr * tr / 4 - det2));
        var l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        return [Math.sqrt(Math.max(0, l1)), Math.sqrt(Math.max(0, l2))];
      }

      function warp() {
        var a = a1.x, c = a1.y, b = a2.x, d = a2.y;
        var det = a * d - b * c;
        var data = buf.data;
        if (Math.abs(det) < 1e-6) { for (var z = 0; z < RW * RH; z += 1) { var oo = z * 4; data[oo] = 245; data[oo + 1] = 243; data[oo + 2] = 234; data[oo + 3] = 255; } dctx.putImageData(buf, 0, 0); return; }
        var ia = d / det, ib = -b / det, ic = -c / det, id2 = a / det;
        for (var jj = 0; jj < RH; jj += 1) {
          var my = ymax - (jj / RH) * (ymax - ymin);
          for (var ii = 0; ii < RW; ii += 1) {
            var mx = xmin + (ii / RW) * (xmax - xmin);
            var sx = ia * mx + ib * my, sy = ic * mx + id2 * my;   // image-space coord in [0,1]^2
            var o = (jj * RW + ii) * 4;
            if (sx >= 0 && sx < 1 && sy >= 0 && sy < 1) {
              var px = Math.min(IMG.w - 1, Math.floor(sx * IMG.w));
              var py = Math.min(IMG.h - 1, Math.floor((1 - sy) * IMG.h));
              var so = (py * IMG.w + px) * 4; var g = photoData[so];
              data[o] = g; data[o + 1] = g; data[o + 2] = g; data[o + 3] = 255;
            } else { data[o] = 255; data[o + 1] = 254; data[o + 2] = 249; data[o + 3] = 255; }
          }
        }
        dctx.putImageData(buf, 0, 0);
      }

      function arrow(ctx, x0, y0, x1, y1, col) {
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        var ang = Math.atan2(y1 - y0, x1 - x0);
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 11 * Math.cos(ang - 0.4), y1 - 11 * Math.sin(ang - 0.4));
        ctx.lineTo(x1 - 11 * Math.cos(ang + 0.4), y1 - 11 * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
      }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "14px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        warp();
        var dTL = m2s(xmin, ymax), dBR = m2s(xmax, ymin);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(disp, dTL[0], dTL[1], dBR[0] - dTL[0], dBR[1] - dTL[1]);

        // transformed grid of the image square (0..1 in both image axes)
        ctx.strokeStyle = C.line; ctx.lineWidth = 0.8;
        for (var t = 0; t <= 4; t += 1) {
          var u = t / 4;
          var p0 = m2s(a1.x * u, a1.y * u), p1 = m2s(a1.x * u + a2.x, a1.y * u + a2.y);
          ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
          var q0 = m2s(a2.x * u, a2.y * u), q1 = m2s(a2.x * u + a1.x, a2.y * u + a1.y);
          ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q1[0], q1[1]); ctx.stroke();
        }

        // axes
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var ax0 = m2s(xmin, 0), ax1 = m2s(xmax, 0), ay0 = m2s(0, ymin), ay1 = m2s(0, ymax);
        ctx.beginPath(); ctx.moveTo(ax0[0], ax0[1]); ctx.lineTo(ax1[0], ax1[1]); ctx.moveTo(ay0[0], ay0[1]); ctx.lineTo(ay1[0], ay1[1]); ctx.stroke();

        // column vectors
        var o = m2s(0, 0), t1 = m2s(a1.x, a1.y), t2 = m2s(a2.x, a2.y);
        arrow(ctx, o[0], o[1], t1[0], t1[1], C.red);
        arrow(ctx, o[0], o[1], t2[0], t2[1], C.blue);
        ctx.fillStyle = C.red; ctx.textAlign = "left";
        ctx.fillText("столбец 1 (" + a1.x.toFixed(1) + ", " + a1.y.toFixed(1) + ")", t1[0] + 8, t1[1] + 4);
        ctx.fillStyle = C.blue;
        ctx.fillText("столбец 2 (" + a2.x.toFixed(1) + ", " + a2.y.toFixed(1) + ")", t2[0] + 8, t2[1] - 4);
        // draggable knobs
        ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(t1[0], t1[1], 6, 0, 7); ctx.fill();
        ctx.fillStyle = C.blue; ctx.beginPath(); ctx.arc(t2[0], t2[1], 6, 0, 7); ctx.fill();

        var det = a1.x * a2.y - a2.x * a1.y;
        var sv = svd2(); var kappa = sv[1] > 1e-9 ? sv[0] / sv[1] : Infinity;
        output.set([
          { label: "Определитель", value: det.toFixed(2) + (det < 0 ? " (зеркало)" : ""), color: Math.abs(det) < 0.05 ? C.red : C.ink },
          { label: "Сингулярные числа", value: "σ₁=" + sv[0].toFixed(2) + ", σ₂=" + sv[1].toFixed(2), color: C.blue },
          { label: "Число обусловленности κ", value: isFinite(kappa) ? kappa.toFixed(1) : "∞ (необратима)", color: !isFinite(kappa) || kappa > 12 ? C.red : C.green },
        ]);
      }

      function pick(ev) {
        var rect = canvasState.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W;
        var my = (ev.clientY - rect.top) / rect.height * H;
        return { mx: mx, my: my, x: (mx - OX) / S, y: (OY - my) / S };
      }
      canvasState.canvas.addEventListener("mousedown", function (ev) {
        var p = pick(ev); var t1 = m2s(a1.x, a1.y), t2 = m2s(a2.x, a2.y);
        if (Math.hypot(p.mx - t1[0], p.my - t1[1]) < 16) drag = "a1";
        else if (Math.hypot(p.mx - t2[0], p.my - t2[1]) < 16) drag = "a2";
      });
      window.addEventListener("mousemove", function (ev) {
        if (!drag) return; var p = pick(ev);
        var gx = Math.max(-3, Math.min(3.5, Math.round(p.x * 10) / 10));
        var gy = Math.max(-3, Math.min(3.5, Math.round(p.y * 10) / 10));
        if (drag === "a1") { a1.x = gx; a1.y = gy; } else { a2.x = gx; a2.y = gy; }
        draw();
      });
      window.addEventListener("mouseup", function () { drag = null; });

      K.segmented(controls, { label: "Пример", value: 0, options: [
        { label: "единичная", value: 0 }, { label: "сдвиг", value: 1 },
        { label: "поворот", value: 2 }, { label: "почти вырождена", value: 3 } ] }, function (v) {
        if (v === 0) { a1 = { x: 1, y: 0 }; a2 = { x: 0, y: 1 }; }
        else if (v === 1) { a1 = { x: 1, y: 0 }; a2 = { x: 1, y: 1 }; }
        else if (v === 2) { a1 = { x: 0.7, y: 0.7 }; a2 = { x: -0.7, y: 0.7 }; }
        else { a1 = { x: 1, y: 0.9 }; a2 = { x: 1, y: 1 }; }
        draw();
      });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
