// Lesson 39: recommender-lab — drag your taste vector over the real MovieLens map; dot product ranks films.
(function () {
  "use strict";

  var DATA = {"mu":3.529,"movies":[{"t":"Star Wars","x":-0.91,"y":0.397,"b":0.853,"n":583},{"t":"Contact","x":-0.784,"y":0.179,"b":0.33,"n":509},{"t":"Fargo","x":-0.044,"y":0.974,"b":0.562,"n":508},{"t":"Return of the Jedi","x":-0.96,"y":0.062,"b":0.524,"n":507},{"t":"Liar Liar","x":-0.598,"y":-0.704,"b":-0.286,"n":485},{"t":"English Patient, The","x":-0.282,"y":0.768,"b":0.13,"n":481},{"t":"Scream","x":-0.677,"y":-0.078,"b":-0.025,"n":478},{"t":"Toy Story","x":-0.551,"y":-0.074,"b":0.408,"n":452},{"t":"Air Force One","x":-0.507,"y":-0.99,"b":0.134,"n":431},{"t":"Independence Day","x":-0.81,"y":-0.983,"b":0.019,"n":429},{"t":"Raiders of the Lost Ark","x":-1.019,"y":0.015,"b":0.801,"n":420},{"t":"Godfather, The","x":-0.405,"y":1.095,"b":0.701,"n":413},{"t":"Pulp Fiction","x":0.15,"y":1.067,"b":0.534,"n":394},{"t":"Twelve Monkeys","x":0.263,"y":0.248,"b":0.339,"n":392},{"t":"Silence of the Lambs, Th","x":-0.486,"y":0.503,"b":0.788,"n":390},{"t":"Jerry Maguire","x":-0.823,"y":-0.095,"b":0.207,"n":384},{"t":"Rock, The","x":-0.332,"y":-0.583,"b":0.269,"n":378},{"t":"Empire Strikes Back, The","x":-0.79,"y":0.288,"b":0.74,"n":367},{"t":"Star Trek: First Contact","x":-0.615,"y":-0.293,"b":0.23,"n":365},{"t":"Titanic","x":-0.993,"y":-0.143,"b":0.762,"n":350},{"t":"Back to the Future","x":-0.805,"y":-0.11,"b":0.412,"n":350},{"t":"Mission: Impossible","x":-0.319,"y":-0.509,"b":-0.103,"n":344},{"t":"Fugitive, The","x":-0.678,"y":-0.08,"b":0.591,"n":336},{"t":"Indiana Jones and the La","x":-0.62,"y":-0.304,"b":0.489,"n":331},{"t":"Willy Wonka and the Choc","x":-0.256,"y":0.661,"b":0.152,"n":326},{"t":"Princess Bride, The","x":-0.405,"y":0.108,"b":0.715,"n":324},{"t":"Forrest Gump","x":-1.002,"y":-0.091,"b":0.362,"n":321},{"t":"Monty Python and the Hol","x":0.116,"y":0.367,"b":0.638,"n":316},{"t":"Saint, The","x":-0.218,"y":-1.007,"b":-0.33,"n":316},{"t":"Full Monty, The","x":0.231,"y":0.485,"b":0.468,"n":315},{"t":"Men in Black","x":-0.443,"y":-0.314,"b":0.292,"n":303},{"t":"Terminator, The","x":-0.415,"y":0.014,"b":0.507,"n":301},{"t":"E.T. the Extra-Terrestri","x":-0.789,"y":0.186,"b":0.385,"n":300},{"t":"Dead Man Walking","x":0.333,"y":0.741,"b":0.4,"n":299},{"t":"Schindler's List","x":-0.503,"y":0.238,"b":0.971,"n":298},{"t":"Leaving Las Vegas","x":0.282,"y":0.914,"b":0.184,"n":298},{"t":"L.A. Confidential","x":-0.099,"y":0.369,"b":0.704,"n":297},{"t":"Braveheart","x":-0.938,"y":-0.064,"b":0.695,"n":297},{"t":"Terminator 2: Judgment D","x":-0.481,"y":-0.313,"b":0.605,"n":295},{"t":"Conspiracy Theory","x":-0.647,"y":-0.838,"b":-0.024,"n":295},{"t":"Birdcage, The","x":0.229,"y":-0.219,"b":0.065,"n":293},{"t":"Mr. Holland's Opus","x":-0.518,"y":-0.319,"b":0.294,"n":293},{"t":"Twister","x":-0.478,"y":-0.826,"b":-0.26,"n":293},{"t":"Alien","x":-0.066,"y":0.304,"b":0.567,"n":291},{"t":"When Harry Met Sally...","x":-0.346,"y":-0.03,"b":0.478,"n":290},{"t":"Aliens","x":-0.245,"y":0.131,"b":0.505,"n":284},{"t":"Shawshank Redemption, Th","x":-0.642,"y":0.206,"b":0.96,"n":283},{"t":"Jaws","x":-0.565,"y":0.547,"b":0.243,"n":280}]};

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("recommender-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var OX = 250, OY = 235, S = 150;   // taste-space -> screen
      var mov = DATA.movies, mu = DATA.mu;
      var you = { x: -0.6, y: 0.35 };    // user taste vector
      var drag = false, hover = -1;

      K.hint(
        root,
        "Настоящая карта фильмов MovieLens: близкие точки нравятся похожей публике. Перетаскивайте свой вектор вкуса — прогноз оценки каждого фильма равен средней плюс репутация фильма плюс скалярное произведение вашего вектора на вектор фильма. Справа — какие фильмы система порекомендует именно вам.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Прогноз оценки = μ + смещение фильма b + (ваш вектор · вектор фильма). Чем сильнее ваш вектор смотрит в сторону фильма, тем выше прогноз. Оси карты не имеют закреплённого смысла — важна только геометрия близости.",
      );

      var canvasState = K.makeCanvas(stage, W, H, { label: "Карта вкусов и рекомендации", onResize: draw, drag: false });

      function m2s(x, y) { return [OX + x * S, OY - y * S]; }
      function predict(mv1) { return mu + mv1.b + you.x * mv1.x + you.y * mv1.y; }

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif"; ctx.textBaseline = "alphabetic";

        // rank movies by prediction
        var ranked = mov.map(function (m, i) { return { i: i, p: predict(m) }; }).sort(function (a, b) { return b.p - a.p; });
        var topset = {}; for (var t = 0; t < 5; t += 1) topset[ranked[t].i] = t + 1;

        // axes
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        var ax = m2s(-1.4, 0), bx = m2s(0.7, 0), ay = m2s(0, -1.2), by = m2s(0, 1.4);
        ctx.beginPath(); ctx.moveTo(ax[0], ax[1]); ctx.lineTo(bx[0], bx[1]); ctx.moveTo(ay[0], ay[1]); ctx.lineTo(by[0], by[1]); ctx.stroke();

        // movie points
        for (var k = 0; k < mov.length; k += 1) {
          var mk = mov[k], s = m2s(mk.x, mk.y);
          var isTop = topset[k], isHover = k === hover;
          ctx.fillStyle = isTop ? C.red : "rgba(49,95,140,0.35)";
          ctx.beginPath(); ctx.arc(s[0], s[1], isTop ? 6 : 3 + Math.sqrt(mk.n) / 12, 0, 7); ctx.fill();
          if (isTop) { ctx.fillStyle = C.gold; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif"; ctx.fillText(String(topset[k]), s[0], s[1] - 9); }
          if (isHover) {
            ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.font = "13px PT Sans, sans-serif";
            ctx.fillText(mk.t, s[0] + 8, s[1] + 4);
          }
        }

        // your taste vector
        var o = m2s(0, 0), yv = m2s(you.x, you.y);
        ctx.strokeStyle = C.gold; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(yv[0], yv[1]); ctx.stroke();
        ctx.fillStyle = C.gold; ctx.beginPath(); ctx.arc(yv[0], yv[1], 8, 0, 7); ctx.fill();
        ctx.fillStyle = C.ink; ctx.textAlign = "center"; ctx.fillText("вы", yv[0], yv[1] - 12);

        // recommendation panel
        var px = 620, py = 60;
        ctx.fillStyle = C.ink; ctx.font = "14px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("Рекомендуем вам:", px, py - 12);
        for (var r = 0; r < 5; r += 1) {
          var mm = mov[ranked[r].i];
          ctx.fillStyle = C.red; ctx.font = "13px PT Sans, sans-serif";
          ctx.fillText((r + 1) + ".", px, py + r * 30 + 6);
          ctx.fillStyle = C.ink; ctx.fillText(mm.t.length > 20 ? mm.t.slice(0, 20) : mm.t, px + 20, py + r * 30 + 6);
          ctx.fillStyle = C.green; ctx.textAlign = "left";
          ctx.fillText(ranked[r].p.toFixed(2), px + 200, py + r * 30 + 6); ctx.textAlign = "left";
        }
        ctx.fillStyle = C.muted; ctx.font = "12px PT Sans, sans-serif";
        ctx.fillText("прогноз", px + 200, py - 12);

        output.set([
          { label: "Ваш вектор вкуса", value: "(" + you.x.toFixed(2) + ", " + you.y.toFixed(2) + ")", color: C.gold },
          { label: "Лучший прогноз", value: mov[ranked[0].i].t.slice(0, 22) + " — " + ranked[0].p.toFixed(2), color: C.red },
          { label: "Худший прогноз", value: mov[ranked[ranked.length - 1].i].t.slice(0, 18) + " — " + ranked[ranked.length - 1].p.toFixed(2), color: C.muted },
        ]);
      }

      function pick(ev) {
        var rect = canvasState.canvas.getBoundingClientRect();
        var mx = (ev.clientX - rect.left) / rect.width * W, my = (ev.clientY - rect.top) / rect.height * H;
        return { mx: mx, my: my, x: (mx - OX) / S, y: (OY - my) / S };
      }
      canvasState.canvas.addEventListener("mousedown", function (ev) {
        var p = pick(ev), yv = m2s(you.x, you.y);
        if (Math.hypot(p.mx - yv[0], p.my - yv[1]) < 20) drag = true;
      });
      window.addEventListener("mousemove", function (ev) {
        if (!drag) return; var p = pick(ev);
        you.x = Math.max(-1.4, Math.min(0.7, Math.round(p.x * 100) / 100));
        you.y = Math.max(-1.2, Math.min(1.4, Math.round(p.y * 100) / 100));
        draw();
      });
      window.addEventListener("mouseup", function () { drag = false; });
      canvasState.canvas.addEventListener("mousemove", function (ev) {
        if (drag) return; var p = pick(ev), best = -1, bd = 14;
        for (var k = 0; k < mov.length; k += 1) { var s = m2s(mov[k].x, mov[k].y); var d = Math.hypot(p.mx - s[0], p.my - s[1]); if (d < bd) { bd = d; best = k; } }
        if (best !== hover) { hover = best; draw(); }
      });

      K.segmented(controls, { label: "Готовые профили вкуса", value: 0, options: [
        { label: "боевики/фантастика", value: 0 }, { label: "драма/классика", value: 1 }, { label: "семейное", value: 2 } ] }, function (v) {
        if (v === 0) you = { x: -0.7, y: -0.1 };
        else if (v === 1) you = { x: -0.3, y: 0.7 };
        else you = { x: -0.5, y: -0.1 };
        draw();
      });

      draw();
      return function () { canvasState.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
