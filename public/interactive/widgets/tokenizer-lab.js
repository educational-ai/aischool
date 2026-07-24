// Lesson 75: tokenizer-lab — train a miniature BPE live: add merges one by one on a chosen
// corpus and watch how the same probe string collapses from characters into subwords.
// The lesson's thesis is visible directly: merges learned on one corpus barely help another.
(function () {
  "use strict";
  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("tokenizer-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 880, H = 470;
      var PB = "#fffef9";

      var CORPORA = [
        {
          name: "русская проза",
          text: "нейросеть учится на примерах. сеть учит признаки, а признаки учат сеть. " +
            "модель обучается, обучение идёт, обученная модель предсказывает. " +
            "данные для обучения, данные для проверки, данные для теста. " +
            "признак, признаки, признаком, признакам, признаковое пространство. " +
            "учитель учит ученика, ученик учится у учителя, обучение с учителем. " +
            "лес, лесной, лесник, лесничество, перелесок, лесная тропа. " +
            "нейрон, нейроны, нейронная сеть, нейросетевой подход, нейросети учатся."
        },
        {
          name: "английский текст",
          text: "the model learns from examples. learning is slow, the learner learns. " +
            "data for learning, data for testing, data for the test set. " +
            "feature, features, featured, featuring, feature space, feature map. " +
            "the teacher teaches the student, the student learns from the teacher. " +
            "forest, forester, forestry, forests, a forest path in the forest. " +
            "neuron, neurons, neural network, neural networks learn features."
        },
        {
          name: "программный код",
          text: "def train(data, model): return model.fit(data) " +
            "for i in range(len(data)): model.step(data[i]) " +
            "if model.loss < best_loss: best_loss = model.loss " +
            "import numpy as np; x = np.zeros(len(data)); y = np.ones(len(data)) " +
            "class Model: def __init__(self, data): self.data = data " +
            "print(model.loss, model.step, model.fit, model.data)"
        }
      ];
      var PROBES = [
        { name: "русская фраза", text: "нейросеть учится на признаках" },
        { name: "английская фраза", text: "the model learns features" },
        { name: "строка кода", text: "model.fit(data)" }
      ];

      var corpus = 0, probe = 0, nMerges = 0, MAXM = 120;
      var cache = {};   // corpusIndex -> array of merges (pairs of strings)

      function words(text) {
        var raw = text.split(/\s+/), out = {}, i;
        for (i = 0; i < raw.length; i += 1) {
          if (!raw[i]) continue;
          var w = "_" + raw[i];
          out[w] = (out[w] || 0) + 1;
        }
        return out;
      }

      // train up to MAXM merges once per corpus, deterministic (ties -> lexicographic)
      function trainAll(ci) {
        if (cache[ci]) return cache[ci];
        var wf = words(CORPORA[ci].text), keys = Object.keys(wf);
        var seq = [], freq = [], i, j;
        for (i = 0; i < keys.length; i += 1) { seq.push(keys[i].split("")); freq.push(wf[keys[i]]); }
        var merges = [];
        for (var step = 0; step < MAXM; step += 1) {
          var cnt = {};
          for (i = 0; i < seq.length; i += 1) {
            for (j = 0; j + 1 < seq[i].length; j += 1) {
              var k = seq[i][j] + "\u0000" + seq[i][j + 1];
              cnt[k] = (cnt[k] || 0) + freq[i];
            }
          }
          var bk = null, bv = 0, ks = Object.keys(cnt).sort();
          for (i = 0; i < ks.length; i += 1) if (cnt[ks[i]] > bv) { bv = cnt[ks[i]]; bk = ks[i]; }
          if (bk === null || bv < 2) break;
          var pr = bk.split("\u0000"), joined = pr[0] + pr[1];
          merges.push(pr);
          for (i = 0; i < seq.length; i += 1) {
            var s = seq[i], out = [];
            for (j = 0; j < s.length;) {
              if (j + 1 < s.length && s[j] === pr[0] && s[j + 1] === pr[1]) { out.push(joined); j += 2; }
              else { out.push(s[j]); j += 1; }
            }
            seq[i] = out;
          }
        }
        cache[ci] = merges;
        return merges;
      }

      function encodeWord(w, merges, m) {
        var s = w.split(""), i, best, bi;
        for (;;) {
          best = -1; bi = -1;
          for (i = 0; i + 1 < s.length; i += 1) {
            for (var r = 0; r < m; r += 1) {
              if (merges[r][0] === s[i] && merges[r][1] === s[i + 1]) {
                if (best < 0 || r < best) { best = r; bi = i; }
                break;
              }
            }
          }
          if (best < 0) break;
          s = s.slice(0, bi).concat([s[bi] + s[bi + 1]], s.slice(bi + 2));
        }
        return s;
      }

      function encodeText(text, merges, m) {
        var raw = text.split(/\s+/), out = [], i, k;
        for (i = 0; i < raw.length; i += 1) {
          if (!raw[i]) continue;
          var p = encodeWord("_" + raw[i], merges, m);
          for (k = 0; k < p.length; k += 1) out.push(p[k]);
        }
        return out;
      }

      function per100(text, merges, m) {
        return 100 * encodeText(text, merges, m).length / text.replace(/\s/g, "").length;
      }

      K.hint(root, "Мини-BPE обучается прямо здесь. Двигайте число слияний: алгоритм каждый раз склеивает самую частую соседнюю пару в обучающем корпусе и добавляет её в словарь. Смотрите, как проверочная строка сжимается из букв в куски слов — и что происходит, если корпус и строка на разных языках.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Первые слияния дают огромную экономию, дальше отдача падает: это та самая кривая из урока. Обучите словарь на русском и подайте английскую строку (или наоборот) — чужой текст почти не сжимается, потому что нужные пары в корпусе не встречались. Символ «_» отмечает начало слова: ' дом' и 'дом' — разные токены.");
      var cs = K.makeCanvas(stage, W, H, { label: "Обученные слияния и разбиение проверочной строки", onResize: draw });

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "alphabetic";
        var merges = trainAll(corpus);
        var m = Math.min(nMerges, merges.length);
        var probeText = PROBES[probe].text;
        var toks = encodeText(probeText, merges, m);
        var base = encodeText(probeText, merges, 0).length;

        // --- probe segmentation boxes
        ctx.fillStyle = C.muted;
        ctx.textAlign = "left";
        ctx.fillText("проверочная строка «" + probeText + "»:", 24, 34);
        var x = 24, y = 52, rowH = 40, maxX = W - 24;
        for (var i = 0; i < toks.length; i += 1) {
          var label = toks[i].replace(/_/g, "·");
          var w = Math.max(20, ctx.measureText(label).width + 16);
          if (x + w > maxX) { x = 24; y += rowH; }
          var deep = toks[i].length > 1;
          ctx.fillStyle = deep ? "rgba(49,95,140,0.14)" : "rgba(185,74,59,0.10)";
          ctx.strokeStyle = deep ? C.blue : C.red;
          ctx.lineWidth = 1.1;
          ctx.fillRect(x, y, w, 30);
          ctx.strokeRect(x, y, w, 30);
          ctx.fillStyle = C.ink || "#171915";
          ctx.textAlign = "center";
          ctx.fillText(label, x + w / 2, y + 20);
          x += w + 6;
        }
        var listTop = y + rowH + 18;

        // --- merge list (last learned merges)
        ctx.textAlign = "left";
        ctx.fillStyle = C.muted;
        ctx.fillText("последние выученные слияния (корпус: " + CORPORA[corpus].name + "):", 24, listTop);
        var shown = merges.slice(Math.max(0, m - 12), m);
        var mx = 24, my = listTop + 24;
        for (var j = shown.length - 1; j >= 0; j -= 1) {
          var idx = Math.max(0, m - 12) + j + 1;
          var txt = idx + ". " + (shown[j][0] + " + " + shown[j][1] + " → " + shown[j][0] + shown[j][1]).replace(/_/g, "·");
          var tw = ctx.measureText(txt).width + 18;
          if (mx + tw > W - 24) { mx = 24; my += 26; }
          ctx.fillStyle = j === shown.length - 1 ? C.gold : C.muted;
          ctx.fillText(txt, mx, my);
          mx += tw;
        }

        // --- compression bar
        var barY = H - 66;
        ctx.fillStyle = C.muted;
        ctx.fillText("длина строки в токенах:", 24, barY - 10);
        var full = W - 260;
        ctx.fillStyle = "rgba(110,114,106,0.16)";
        ctx.fillRect(200, barY, full, 22);
        ctx.fillStyle = "rgba(56,115,93,0.65)";
        ctx.fillRect(200, barY, full * toks.length / base, 22);
        ctx.fillStyle = C.green;
        ctx.textAlign = "left";
        ctx.fillText(toks.length + " из " + base + " (" + Math.round(100 * toks.length / base) + "%)", 206 + full, barY + 17);

        var own = per100(CORPORA[corpus].text, merges, m);
        output.set([
          { label: "Размер словаря (буквы + слияния)", value: String(m) + " слияний", color: C.blue },
          { label: "Токенов в проверочной строке", value: String(toks.length), color: C.green },
          { label: "Токенов на 100 знаков строки", value: per100(probeText, merges, m).toFixed(1), color: C.red },
          { label: "Токенов на 100 знаков своего корпуса", value: own.toFixed(1), color: C.gold }
        ]);
      }

      K.segmented(controls, {
        label: "Корпус для обучения словаря", value: 0,
        options: [
          { label: "русская проза", value: 0 },
          { label: "английский текст", value: 1 },
          { label: "программный код", value: 2 }
        ]
      }, function (v) { corpus = v; draw(); });

      K.segmented(controls, {
        label: "Проверочная строка", value: 0,
        options: [
          { label: "русская фраза", value: 0 },
          { label: "английская фраза", value: 1 },
          { label: "строка кода", value: 2 }
        ]
      }, function (v) { probe = v; draw(); });

      K.slider(controls, {
        label: "Число слияний BPE", min: 0, max: MAXM, step: 1, value: 0,
        format: function (v) { return String(v); }
      }, function (v) { nMerges = v; draw(); });

      draw();
      return function () { cs.destroy(); };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
