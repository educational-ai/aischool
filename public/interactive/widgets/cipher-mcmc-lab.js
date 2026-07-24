// Lesson 68: cipher-mcmc-lab — break a substitution cipher with a Metropolis chain.
// The language model is a character bigram model trained in the browser on a real
// fragment of this textbook; the ciphertext is a held-out fragment of lesson 45.
(function () {
  "use strict";
  var ALPHA = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
  var M = 33, A = 34;
  var TRAIN = "ения меньше секунды ошибка на выборке и ошибка в мире эмпирический риск вычисляется по конечному набору пользователя интересует ожидаемая потеря на будущей паре случайных величин распределение здесь не обязательно известно формулой это сокращённая запись механизма порождающего будущие случаи времена суток погоду расписания поведение людей и работу датчиков тестовая выборка даёт ещё одну оценку но не превращается в само распределение обобщение модель обобщает в выбранном сценарии если сохраняет приемлемое качество на объектах которые не использовались при выборе параметров и настроек и порождаются условиями соответствующими этому сценарию окончание определения принципиально модель обученная на одной школе может прекрасно обобщать на новые недели той же школы и плохо на другую школу это не логическое противоречие слово новый всегда требует уточнения новый час человек город прибор сезон или тип события если потери ограничены между и и тест состоит из независимых наблюдений разброс среднего порядка это не точная универсальная формула доверительного интервала а масштаб увеличение теста в четыре раза уменьшает типичный шум оценки примерно вдвое а не в четыре раза для доли ошибок грубая стандартная ошибка равна если классификатор ошибся раз на объектах а разумно ожидать неопределённость в несколько процентных пунктов а не рассказывать о точности до сотых процента при малом редких событиях и зависимых наблюдениях нужны более аккуратные методы но уже эта оценка защищает от ложной точности сколько даёт ещё один ноль сравните стандартную ошибку доли промахов при и во сколько раз нужно увеличить тест чтобы типичный разброс уменьшился примерно в десять раз почему десять почти одинаковых кадров одной видеозаписи не обязательно эквивалентны десяти независимым объектам наконец усреднение прячет структуру одинаковая может состоять из частых малых отклонений или редких катастрофических промахов средняя точность может расти пока качество для небольшой группы падает поэтому рядом с итогом показывают ошибку по времени режиму величине целевого значения и группам которые имеют смысл в задаче кому досталась средняя ошибка автор среднее отвечает на вопрос о сумме но не о распределении последствий перед автоматизацией спросите кто получает ошибку когда она возникает можно ли её заметить и кто успевает отменить действие почему перебор моделей требует больше данных в пороговом примере мы выбрали лучшее правило из нескольких чем больше вариантов разрешено попробовать тем легче случайно найти правило которому повезло именно на этой выборке это рассуждение можно превратить в небольшую теорему пусть потери лежат между и а тестовые объекты независимы и одинаково распределены для одного заранее зафиксированного правила неравенство хёффдинга даёт смысл правой части прост вероятность большого случайного расхождения эмпирической и ожидаемой ошибки убывает экспоненциально с размером теста пока не будем доказывать само неравенство хёффдинга нас интересует что произойдёт при выборе из конечного семейства одновременная оценка конечного семейства с вероятностью не меньше для всех заранее зафиксированных правил одновременно выполнено почему в оценке появляется логарифм числа правил для каждого вероятность отклонения больше не превосходит вероятность того что плохо поведёт себя хотя бы одно из правил не больше суммы этих вероятностей потребуем чтобы правая часть была не больше прологарифмируем и выразим получится заявленная граница единственный новый приём оценка объединения событий вероятность хотя бы одного провала не больше суммы вероятностей отдельных провалов возьмём и правая часть границы примерно равна гарантия довольно груба она разрешает расхождение около процентного пункта и ничего не знает о близости правил друг к другу но она показывает важную структуру размер теста находится в знаменателе под корнем а число вариантов входит под логарифмом увеличить семейство в тысячу раз не то же самое что увеличить неопределённость в тысячу раз однако бесплатного перебора нет есть две принципиальные оговорки во первых семейство должно быть определено до просмотра тестовых ответов если после каждой ошибки исследователь изобретает новый признак фактическое множество решений включает весь его адаптивный поиск во вторых тестовые объекты должны соответствовать сценарию и не быть почти копиями теорема не исправляет утечку времени смену школы или десять кадров одного события граница не предсказание ошибки теорема на полях верхняя оценка говорит с такой вероятностью отклонение не превысит она не утверждает что реальное отклонение обязано быть близко к правой части гарантии часто намеренно пессимистичны зато видимы их предпосылки объём проверки из требования из условия выразите минимальный оцените его при и затем объясните почему полученное число нельзя просто подставить как обязательный размер теста для почасового ряда с сильной зависимостью соседних часов эта маленькая теорема связывает три темы курса вероятность сложность выбора и честную проверку позже более тонкие оценки заменят простое число характеристикой богатства семейства пока достаточно помнить качественный вывод чем свободнее мы искали удачное правило тем строже должен быть независимый экзамен потеря модели и цена решения до сих пор функция потерь выглядела как техническая мера расхождения но модель обычно выдаёт число а система совершает действие между ними находится правило решения смешивать эти два уровня нельзя пусть модель оценивает вероятность вечернего пика для обучения вероятности можно использовать если истинный ответ потеря равна она велика при уверенном неверном прогнозе и мала при если получаем такая потеря поощряет осмысленные вероятности но не говорит при каком переносить зарядку решение зависит от последствий допустим ложная тревога стоит условные единицы сотрудник зря проверит щитовую пропущенный пик стоит единиц сработает дорогая защита обозначим действия через предупредить и не предупреждать ожидаемая цена предупреждения при вероятности пика равна потому что рас";
  var PLAIN = "я производная при значит найденная точка максимум оценка просто доля успехов потрогай правдоподобие данные логарифм и ширина максимума сохраните долю успехов и увеличивайте число наблюдений вершина останется на месте а интервал близких значений сузится сравните и одинаковая точечная оценка несёт совершенно разную неопределённость наконец поставьте все успехи и ни одной неудачи максимум уйдёт на гр";

  function install() {
    if (!window.KonturInt) return;
    window.KonturInt.register("cipher-mcmc-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 900, H = 470;
      var idx = {}, i, j;
      for (i = 0; i < M; i += 1) idx[ALPHA.charAt(i)] = i;
      idx[" "] = M;

      function encode(s) {
        var out = [], k;
        for (k = 0; k < s.length; k += 1) {
          var c = idx[s.charAt(k)];
          if (c === undefined) c = M;
          out.push(c);
        }
        return out;
      }
      var trainIds = encode(TRAIN), plainAll = encode(PLAIN);

      // ---- language model: unigram and bigram log-probabilities, add-0.5 smoothing
      var LAM = 0.5;
      var uniC = new Float64Array(A), biC = new Float64Array(A * A);
      for (i = 0; i < trainIds.length; i += 1) uniC[trainIds[i]] += 1;
      for (i = 0; i + 1 < trainIds.length; i += 1) biC[trainIds[i] * A + trainIds[i + 1]] += 1;
      var LU = new Float64Array(A), LB = new Float64Array(A * A), tot = 0;
      for (i = 0; i < A; i += 1) tot += uniC[i];
      for (i = 0; i < A; i += 1) LU[i] = Math.log((uniC[i] + LAM) / (tot + LAM * A));
      for (i = 0; i < A; i += 1) {
        var rs = 0;
        for (j = 0; j < A; j += 1) rs += biC[i * A + j];
        for (j = 0; j < A; j += 1) LB[i * A + j] = Math.log((biC[i * A + j] + LAM) / (rs + LAM * A));
      }

      // ---- state
      var n = 200, order = 2, tempMode = 1;   // 0 = greedy, 1 = fixed temperature, 2 = annealing
      var tempScale = 0.03;
      var plain, cipher, key, inv, cur, best, bestInv, steps, accepted, hist;
      var running = false, raf = null;

      function newKey(seed) {
        var perm = [], r = seed || 1, k, t, tmp;
        for (k = 0; k < M; k += 1) perm.push(k);
        for (k = M - 1; k > 0; k -= 1) {
          r = (r * 1103515245 + 12345) % 2147483648;
          t = Math.floor(r / 2147483648 * (k + 1));
          tmp = perm[k]; perm[k] = perm[t]; perm[t] = tmp;
        }
        var key2 = [];
        for (k = 0; k < A; k += 1) key2.push(k);
        for (k = 0; k < M; k += 1) key2[k] = perm[k];
        return key2;
      }
      function score(iv) {
        var s = 0, k;
        if (order === 1) {
          for (k = 0; k < n; k += 1) s += LU[iv[cipher[k]]];
        } else {
          for (k = 0; k + 1 < n; k += 1) s += LB[iv[cipher[k]] * A + iv[cipher[k + 1]]];
        }
        return s;
      }
      function resetChain() {
        var k, t, tmp;
        inv = [];
        for (k = 0; k < A; k += 1) inv.push(k);
        for (k = M - 1; k > 0; k -= 1) {
          t = Math.floor(Math.random() * (k + 1));
          tmp = inv[k]; inv[k] = inv[t]; inv[t] = tmp;
        }
        cur = score(inv); best = cur; bestInv = inv.slice();
        steps = 0; accepted = 0; hist = [cur];
      }
      function rebuild(seed) {
        var k;
        plain = plainAll.slice(0, n);
        key = newKey(seed || (Date.now() % 100000) + 7);
        cipher = [];
        for (k = 0; k < n; k += 1) cipher.push(key[plain[k]]);
        resetChain();
      }
      function temperature() {
        if (tempMode === 0) return 0;
        if (tempMode === 1) return tempScale * n;
        var f = Math.pow(0.03, Math.min(1, steps / 20000));
        return Math.max(0.02 * n * tempScale / 0.03, tempScale * n * f);
      }
      function proposal() {
        var a = Math.floor(Math.random() * M), b = Math.floor(Math.random() * M);
        if (a === b) return;
        var tmp = inv[a]; inv[a] = inv[b]; inv[b] = tmp;
        var nw = score(inv), d = nw - cur, T = temperature(), ok;
        if (T <= 0) ok = d > 0;
        else ok = d >= 0 || Math.random() < Math.exp(d / T);
        if (ok) {
          cur = nw; accepted += 1;
          if (cur > best) { best = cur; bestInv = inv.slice(); }
        } else {
          tmp = inv[a]; inv[a] = inv[b]; inv[b] = tmp;
        }
        steps += 1;
        if (steps % 20 === 0) { hist.push(best); if (hist.length > 900) hist.shift(); }
      }
      function accuracy() {
        var good = 0, k;
        for (k = 0; k < n; k += 1) if (bestInv[cipher[k]] === plain[k]) good += 1;
        return good / n;
      }

      K.hint(root, "Модель языка здесь настоящая: биграммы считаются прямо в браузере по фрагменту этого же учебника, а шифротекст — кусок урока 45, замкнутый случайной перестановкой 33 букв. Запустите цепь и смотрите, как текст проявляется. Сравните жадный подъём и цепь с температурой, а потом сократите текст до 60 знаков.");
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var buttons = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, "Верхняя строка каждой пары — шифротекст, нижняя — лучшая расшифровка: зелёные знаки угаданы, серые нет. Кривая внизу — лучший балл L по шагам цепи. Жадный режим быстро упирается в локальный максимум; температура позволяет уйти вниз и найти лучший ключ. На коротком тексте балл растёт, а смысл не появляется — информации не хватает.");
      var cs = K.makeCanvas(stage, W, H, { label: "Текущая расшифровка и кривая балла", onResize: draw });

      rebuild(20240);

      function draw() {
        var ctx = cs.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.textBaseline = "alphabetic";
        var perLine = 62, x0 = 18, y0 = 40, lh = 21, k, line, col;
        ctx.font = "12px PT Sans, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("шифротекст (серый) и лучшая расшифровка (зелёное — верно)", x0, 20);
        ctx.font = "14px \"PT Mono\", ui-monospace, monospace";
        var cw = 14.0;
        var rows = Math.min(6, Math.ceil(n / perLine));
        for (line = 0; line < rows; line += 1) {
          var yc = y0 + line * lh * 2.15;
          for (k = 0; k < perLine; k += 1) {
            var pos = line * perLine + k;
            if (pos >= n) break;
            ctx.fillStyle = "rgba(110,114,106,0.45)";
            ctx.fillText(cipher[pos] === M ? " " : ALPHA.charAt(cipher[pos]), x0 + k * cw, yc);
            var g = bestInv[cipher[pos]];
            col = g === plain[pos] ? C.green : C.red;
            ctx.fillStyle = col;
            ctx.fillText(g === M ? " " : ALPHA.charAt(g), x0 + k * cw, yc + lh);
          }
        }
        var gx = 20, gy = 320, gw = W - 40, gh = 120;
        ctx.strokeStyle = C.grid || "#deddd4"; ctx.lineWidth = 1;
        ctx.strokeRect(gx, gy, gw, gh);
        if (hist.length > 1) {
          var lo = hist[0], hi = hist[0];
          for (k = 0; k < hist.length; k += 1) {
            if (hist[k] < lo) lo = hist[k];
            if (hist[k] > hi) hi = hist[k];
          }
          if (hi - lo < 1e-6) hi = lo + 1;
          ctx.strokeStyle = C.blue; ctx.lineWidth = 2;
          ctx.beginPath();
          for (k = 0; k < hist.length; k += 1) {
            var px = gx + gw * k / (hist.length - 1);
            var py = gy + gh - gh * (hist[k] - lo) / (hi - lo);
            if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
        }
        ctx.font = "12px PT Sans, sans-serif"; ctx.fillStyle = C.muted;
        ctx.fillText("лучший балл L", gx + 8, gy + 16);
        ctx.textAlign = "right";
        ctx.fillText("шаги цепи", gx + gw - 10, gy + gh - 8);
        ctx.textAlign = "left";
        var acc = accuracy();
        output.set([
          { label: "балл L (лучший)", value: best.toFixed(1), color: C.blue },
          { label: "верных знаков", value: (acc * 100).toFixed(0) + "%", color: acc > 0.9 ? C.green : C.red },
          { label: "шагов цепи", value: String(steps), color: C.gold },
          { label: "принято предложений", value: steps ? (accepted / steps * 100).toFixed(1) + "%" : "0%", color: C.muted }
        ]);
      }

      function frame() {
        if (!running) return;
        for (var q = 0; q < 400; q += 1) proposal();
        draw();
        raf = window.requestAnimationFrame(frame);
      }

      K.segmented(controls, {
        label: "Модель языка", value: 2,
        options: [{ label: "униграммы", value: 1 }, { label: "биграммы", value: 2 }]
      }, function (v) {
        order = v; cur = score(inv); best = score(bestInv); hist = [best]; draw();
      });

      K.segmented(controls, {
        label: "Режим поиска", value: 1,
        options: [{ label: "жадный", value: 0 }, { label: "цепь", value: 1 }, { label: "отжиг", value: 2 }]
      }, function (v) { tempMode = v; draw(); });

      K.slider(controls, {
        label: "Длина шифротекста", min: 60, max: 400, step: 20, value: 200,
        format: function (v) { return v + " знаков"; }
      }, function (v) { n = v; rebuild(20240); draw(); });

      K.slider(controls, {
        label: "Температура на знак", min: 0.005, max: 0.12, step: 0.005, value: 0.03,
        format: function (v) { return v.toFixed(3); }
      }, function (v) { tempScale = v; draw(); });

      var bRun = K.element("button", "kontur-int-segment", { type: "button", text: "запустить" });
      var bStep = K.element("button", "kontur-int-segment", { type: "button", text: "1000 шагов" });
      var bReset = K.element("button", "kontur-int-segment", { type: "button", text: "новый старт" });
      var bKey = K.element("button", "kontur-int-segment", { type: "button", text: "другой ключ" });
      bRun.style.margin = "0 6px";
      bStep.style.margin = "0 6px";
      bReset.style.margin = "0 6px";
      bKey.style.margin = "0 6px";
      buttons.appendChild(bRun); buttons.appendChild(bStep);
      buttons.appendChild(bReset); buttons.appendChild(bKey);
      bRun.addEventListener("click", function () {
        running = !running;
        bRun.textContent = running ? "пауза" : "запустить";
        if (running) raf = window.requestAnimationFrame(frame);
      });
      bStep.addEventListener("click", function () {
        for (var q = 0; q < 1000; q += 1) proposal();
        draw();
      });
      bReset.addEventListener("click", function () { resetChain(); draw(); });
      bKey.addEventListener("click", function () { rebuild(0); draw(); });

      draw();
      return function () {
        running = false;
        if (raf) window.cancelAnimationFrame(raf);
        cs.destroy();
      };
    });
  }
  install();
  window.addEventListener("kontur-int-ready", install);
})();
