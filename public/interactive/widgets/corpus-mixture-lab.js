// Lesson 78: corpus-mixture-lab — настоящая мини-языковая модель (биграммы с add-k),
// обучаемая прямо в браузере на смеси трёх источников. Двигая веса, повтор и утечку
// benchmark, ученик видит: (1) обмен между доменами, (2) эффективные эпохи малого
// источника, (3) memorization при повторе, (4) рост метрики от утечки без роста навыка.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("corpus-mixture-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;

      // ---------- детерминированный генератор (LCG, фиксированный seed) ----------
      function rngFactory(seed) {
        var s = seed >>> 0;
        return function () {
          s = (s * 1664525 + 1013904223) >>> 0;
          return s / 4294967296;
        };
      }

      // Три источника: у каждого свой словарь плюс общие служебные слова.
      var SHARED = ["и", "в", "не", "что", "на", "."];
      var SOURCES = [
        { name: "код", color: C.violet || "#6f5a8f",
          words: ["def", "return", "if", "else", "for", "range", "print", "value", "self", "import"] },
        { name: "диалоги", color: C.blue || "#315f8c",
          words: ["привет", "ладно", "давай", "когда", "спасибо", "ок", "сегодня", "во", "сколько", "пока"] },
        { name: "статьи", color: C.green || "#38735d",
          words: ["однако", "исследование", "модель", "данные", "результат", "показывает", "выборка", "оценка", "метод", "анализ"] }
      ];

      // Для каждого источника строим свою марковскую цепь: вероятность перехода
      // задаётся один раз фиксированным seed, поэтому «корпус» всегда один и тот же.
      function buildChain(src, seed) {
        var vocab = src.words.concat(SHARED);
        var rnd = rngFactory(seed);
        var table = {};
        vocab.forEach(function (w) {
          var row = [], sum = 0, i;
          for (i = 0; i < vocab.length; i += 1) {
            var p = Math.pow(rnd(), 3) + 0.02;      // разреженные, «острые» переходы
            row.push(p); sum += p;
          }
          for (i = 0; i < row.length; i += 1) row[i] /= sum;
          table[w] = row;
        });
        return { vocab: vocab, table: table };
      }

      function sampleDocs(chain, seed, nDocs, docLen) {
        var rnd = rngFactory(seed), docs = [], d, i, j;
        for (d = 0; d < nDocs; d += 1) {
          var doc = ["<s>"];
          var cur = chain.vocab[Math.floor(rnd() * chain.vocab.length)];
          for (i = 0; i < docLen; i += 1) {
            doc.push(cur);
            var row = chain.table[cur], u = rnd(), acc = 0, next = chain.vocab[0];
            for (j = 0; j < row.length; j += 1) {
              acc += row[j];
              if (u <= acc) { next = chain.vocab[j]; break; }
            }
            cur = next;
          }
          doc.push("</s>");
          docs.push(doc);
        }
        return docs;
      }

      var chains = SOURCES.map(function (s, i) { return buildChain(s, 7801 + i * 17); });
      // Пулы: «код» намеренно маленький — 12 документов, остальные по 60.
      var POOL_DOCS = [12, 60, 60], DOC_LEN = 24;
      var pools = chains.map(function (ch, i) { return sampleDocs(ch, 100 + i, POOL_DOCS[i], DOC_LEN); });
      var valid = chains.map(function (ch, i) { return sampleDocs(ch, 900 + i, 25, DOC_LEN); });
      // Benchmark: отдельная выборка из тех же трёх цепей (по 8 документов).
      var bench = [];
      chains.forEach(function (ch, i) { bench = bench.concat(sampleDocs(ch, 4400 + i, 8, DOC_LEN)); });

      var VOCAB = {};
      SOURCES.forEach(function (s) { s.words.forEach(function (w) { VOCAB[w] = 1; }); });
      SHARED.forEach(function (w) { VOCAB[w] = 1; });
      VOCAB["<s>"] = 1; VOCAB["</s>"] = 1;
      var VSIZE = Object.keys(VOCAB).length;

      function poolTokens(docs) {
        var n = 0;
        docs.forEach(function (d) { n += d.length - 1; });
        return n;
      }
      var poolSize = pools.map(poolTokens);

      // ---------- состояние ----------
      var w = [0.2, 0.5, 0.3];       // веса источников (нормируются)
      var repeat = 1;                 // повтор «премиального» источника «код»
      var leak = 0;                   // доля benchmark, попавшая в обучение
      var BUDGET = 4200;              // токенов в обучающем потоке

      function normWeights() {
        var s = w[0] + w[1] + w[2];
        if (s < 1e-6) return [1 / 3, 1 / 3, 1 / 3];
        return [w[0] / s, w[1] / s, w[2] / s];
      }

      // ---------- обучение биграммной модели ----------
      function train() {
        var nw = normWeights();
        var uni = {}, bi = {}, i, j, k;
        var streamDocs = [];
        var leakDocs = Math.round(leak * bench.length);
        var leakTok = 0;
        for (i = 0; i < leakDocs; i += 1) { streamDocs.push(bench[i]); leakTok += bench[i].length - 1; }
        var rest = Math.max(0, BUDGET - leakTok);
        var epochs = [0, 0, 0];
        for (i = 0; i < 3; i += 1) {
          var quota = Math.round(rest * nw[i]);
          var mult = i === 0 ? repeat : 1;      // повторяем только малый источник
          var got = 0, idx = 0;
          while (got < quota) {
            var doc = pools[i][idx % pools[i].length];
            for (k = 0; k < mult && got < quota; k += 1) { streamDocs.push(doc); got += doc.length - 1; }
            idx += 1;
          }
          epochs[i] = got / poolSize[i];
        }
        for (i = 0; i < streamDocs.length; i += 1) {
          var d = streamDocs[i];
          for (j = 0; j + 1 < d.length; j += 1) {
            uni[d[j]] = (uni[d[j]] || 0) + 1;
            var key = d[j] + "" + d[j + 1];
            bi[key] = (bi[key] || 0) + 1;
          }
        }
        return { uni: uni, bi: bi, epochs: epochs, leakTok: leakTok, docs: streamDocs };
      }

      function nll(model, docs) {
        var kSm = 0.4, tot = 0, n = 0, i, j;
        for (i = 0; i < docs.length; i += 1) {
          var d = docs[i];
          for (j = 0; j + 1 < d.length; j += 1) {
            var c = model.bi[d[j] + "" + d[j + 1]] || 0;
            var u = model.uni[d[j]] || 0;
            tot -= Math.log((c + kSm) / (u + kSm * VSIZE));
            n += 1;
          }
        }
        return n ? tot / n : 0;
      }

      // ---------- интерфейс ----------
      K.hint(root, "Внутри виджета живёт настоящая маленькая языковая модель: биграммы со сглаживанием, обученные на потоке фиксированного размера. Веса решают, чьи токены займут этот бюджет. Смотрите не на среднее, а на три отдельных столбика: улучшение одного домена всегда чем-то оплачено.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Слева — состав обучающего потока и эффективное число проходов E по каждому источнику. Справа — отложенный NLL по каждому домену (чем ниже, тем лучше) и две контрольные величины: loss на самом обучающем потоке и accuracy на benchmark. Повтор малого источника роняет loss на увиденном тексте, а утечка benchmark поднимает его оценку, не трогая отложенные домены.");
      var cs = K.makeCanvas(stage, W, H, { label: "Состав корпуса и отложенный loss по доменам", onResize: draw, drag: false });

      K.slider(controls, { label: "Вес источника «код» (малый пул)", min: 0, max: 1, step: 0.05, value: w[0], format: fmt2 }, function (v) { w[0] = v; draw(); });
      K.slider(controls, { label: "Вес источника «диалоги»", min: 0, max: 1, step: 0.05, value: w[1], format: fmt2 }, function (v) { w[1] = v; draw(); });
      K.slider(controls, { label: "Вес источника «статьи»", min: 0, max: 1, step: 0.05, value: w[2], format: fmt2 }, function (v) { w[2] = v; draw(); });
      K.slider(controls, { label: "Повтор документов «кода»", min: 1, max: 16, step: 1, value: 1, format: function (v) { return "×" + v; } }, function (v) { repeat = v; draw(); });
      K.segmented(controls, {
        label: "Утечка benchmark в обучение", value: 0,
        options: [{ label: "нет", value: 0 }, { label: "четверть", value: 0.25 }, { label: "половина", value: 0.5 }, { label: "весь", value: 1 }]
      }, function (v) { leak = v; draw(); });

      function fmt2(v) { return v.toFixed(2).replace(".", ","); }
      function fmt3(v) { return v.toFixed(3).replace(".", ","); }

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";

        var model = train();
        var nw = normWeights();
        var losses = [nll(model, valid[0]), nll(model, valid[1]), nll(model, valid[2])];
        var avg = (losses[0] + losses[1] + losses[2]) / 3;
        var trainLoss = nll(model, model.docs);
        var benchLoss = nll(model, bench);

        // ---- левая панель: состав потока ----
        var x0 = 56, y0 = 60, barW = 300, barH = 42;
        ctx.fillStyle = C.ink || "#171915";
        ctx.textAlign = "left";
        ctx.font = "13px PT Sans, sans-serif";
        ctx.fillText("Из чего состоит обучающий поток", x0, y0 - 18);
        var acc = 0, i;
        for (i = 0; i < 3; i += 1) {
          var seg = barW * nw[i] * (1 - model.leakTok / BUDGET);
          ctx.fillStyle = SOURCES[i].color;
          ctx.globalAlpha = 0.85;
          ctx.fillRect(x0 + acc, y0, seg, barH);
          ctx.globalAlpha = 1;
          if (seg > 34) {
            ctx.fillStyle = "#fffef9"; ctx.textAlign = "center";
            ctx.fillText(Math.round(100 * nw[i] * (1 - model.leakTok / BUDGET)) + "%", x0 + acc + seg / 2, y0 + 27);
          }
          acc += seg;
        }
        if (model.leakTok > 0) {
          var lw = barW * model.leakTok / BUDGET;
          ctx.fillStyle = C.red || "#b94a3b";
          ctx.fillRect(x0 + acc, y0, lw, barH);
          ctx.fillStyle = "#fffef9"; ctx.textAlign = "center"; ctx.font = "11px PT Sans, sans-serif";
          if (lw > 40) ctx.fillText("утечка", x0 + acc + lw / 2, y0 + 26);
        }

        // легенда + эффективные эпохи
        ctx.font = "12px PT Sans, sans-serif";
        for (i = 0; i < 3; i += 1) {
          var ly = y0 + 78 + i * 30;
          ctx.fillStyle = SOURCES[i].color;
          ctx.fillRect(x0, ly - 10, 12, 12);
          ctx.textAlign = "left"; ctx.fillStyle = C.ink || "#171915";
          ctx.fillText(SOURCES[i].name + ": пул " + poolSize[i] + " токенов", x0 + 20, ly);
          var ep = model.epochs[i];
          ctx.fillStyle = ep > 3 ? (C.red || "#b94a3b") : (C.muted || "#6e726a");
          ctx.textAlign = "right";
          ctx.fillText("E = " + fmt2(ep), x0 + barW, ly);
        }
        ctx.textAlign = "left";
        ctx.fillStyle = C.muted || "#6e726a";
        ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("E — сколько раз поток проходит по источнику целиком", x0, y0 + 186);
        ctx.fillText("при повторе ×" + repeat + " каждый документ «кода» идёт подряд", x0, y0 + 204);

        // ---- правая панель: отложенный loss по доменам ----
        var px = 470, py = 380, pw = 380, ph = 268;
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, py - ph); ctx.lineTo(px, py); ctx.lineTo(px + pw, py); ctx.stroke();
        var lo = 2.2, hi = 3.6;
        function ly2(v) { return py - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * ph; }
        ctx.strokeStyle = C.grid || "#deddd4";
        ctx.fillStyle = C.muted || "#6e726a"; ctx.textAlign = "right"; ctx.font = "11px PT Sans, sans-serif";
        for (var t = lo; t <= hi + 1e-9; t += 0.35) {
          var gy = ly2(t);
          ctx.beginPath(); ctx.moveTo(px, gy); ctx.lineTo(px + pw, gy); ctx.stroke();
          ctx.fillText(fmt2(t), px - 8, gy + 4);
        }
        var bw = 58, gap = 34;
        for (i = 0; i < 3; i += 1) {
          var bx = px + 30 + i * (bw + gap);
          var by = ly2(losses[i]);
          ctx.fillStyle = SOURCES[i].color; ctx.globalAlpha = 0.88;
          ctx.fillRect(bx, by, bw, py - by);
          ctx.globalAlpha = 1;
          ctx.fillStyle = C.ink || "#171915"; ctx.textAlign = "center"; ctx.font = "12px PT Sans, sans-serif";
          ctx.fillText(fmt2(losses[i]), bx + bw / 2, by - 8);
          ctx.fillStyle = C.muted || "#6e726a"; ctx.font = "11px PT Sans, sans-serif";
          ctx.fillText(SOURCES[i].name, bx + bw / 2, py + 16);
        }
        // линия среднего
        ctx.strokeStyle = C.gold || "#a57920"; ctx.lineWidth = 1.8;
        ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(px, ly2(avg)); ctx.lineTo(px + pw, ly2(avg)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.gold || "#a57920"; ctx.textAlign = "left"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("среднее " + fmt2(avg), px + pw - 96, ly2(avg) - 6);
        ctx.fillStyle = C.ink || "#171915"; ctx.font = "13px PT Sans, sans-serif"; ctx.textAlign = "left";
        ctx.fillText("Отложенный NLL по доменам", px, py - ph - 14);
        ctx.fillStyle = C.muted || "#6e726a"; ctx.font = "11px PT Sans, sans-serif";
        ctx.fillText("отложенные тексты не участвуют в обучении никогда", px, py + 38);

        var gapMem = trainLoss - (losses[0] + losses[1] + losses[2]) / 3;
        output.set([
          { label: "NLL: код / диалоги / статьи", value: fmt2(losses[0]) + " / " + fmt2(losses[1]) + " / " + fmt2(losses[2]) },
          { label: "Среднее по доменам", value: fmt3(avg), color: C.gold },
          { label: "Loss на самом потоке (что видит оптимизатор)", value: fmt3(trainLoss), color: C.red },
          { label: "Разрыв поток − отложенное", value: fmt3(gapMem), color: gapMem < -0.35 ? C.red : C.muted },
          { label: "NLL на benchmark", value: fmt3(benchLoss), color: leak > 0 ? C.red : C.blue },
          { label: "Эффективных эпох по «коду»", value: fmt2(model.epochs[0]), color: model.epochs[0] > 3 ? C.red : C.ink }
        ]);
      }

      draw();
      return function () { cs.destroy(); };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
