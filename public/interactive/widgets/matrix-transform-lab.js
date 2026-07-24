// Lesson 36: matrix-transform-lab — drag the matrix columns, warp a real photo, read det and singular values.
(function () {
  "use strict";

  var IMG = {"w":40,"h":48,"rows":["239232e232cd3313737665666767667767777777","212312a232ed3222ce7776767777677777776788","3222225232832232be3234777767777777777788","3222233222222122d1aeee267677777878777777","3122123222223229f2ed8e23f777777777777778","33422231222231ef622efe11fff7776777777778","3df32de2122222e1123dfe2222f7777777877777","3352ce622d3431112ddd2cce4d31777877777778","33221e827b25211111112212e111177777787787","2332223212271111111111111111177887878777","3331232212291011101111111111178878887878","3222233212180114aaba98aa9811188788788887","232223222228106aacbbcccbb983188887887888","2d8122221218138ab79bca868997188888888888","ced1e322121827221219ca198d35398888888888","2932e2317128a365213577534565299989898899","12323122711847955a9cba89ab86898989988999","22313222122977798bc7ccacaa87a99999999999","222132221119c88879baccb69a8a999989999999","2232322112a9bb989b569836a88b999999999999","b22122211aa99bb9889989846899899999999889","aa2132218ab9b7b8a39bcca969659999a9998899","6b1721113ba9b625982849a8b949999999999999","221a11113aa8959299a855889879999999999999","2211111338a996647899a99b6879999999999999","2211213434aa7785868999985999999999999999","12111334349a66a0872322158819999a99999999","2212a434338a511ca76422677e11199999999999","121a84342311111c38745577bc01111a9a999a99","111aac2111111110d9856778e111121211999989","921a401111111111e375557ee111121212111999","b11a111111111111ef5966efe011112221111199","b112210011111101eff9e8ff1111121121111219","ba111111111111117ff2112e1111112222111129","a9411111111111111f111112112111222222111a","ba71111111110111111156211111122211111115","aa61111111110111111dff011211112111111111","ba1111110011111111cffd111131221112111111","a91111181111011111bfe2112211111111111111","a911110a8eba8111112e81212222f66b11111111","a91111489aaa9111112b2112113b8a8856921111","9911011011111111111321112496a35b65611112","a911101111111111101111211212121221211111","3811111111111111111111111221221111111111","4811111111011111111110111212111121111111","4311111111111111111111111112222111101111","411111111111111111a911111111111111111111","4111111111111111110911111111111111110111"]};

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
