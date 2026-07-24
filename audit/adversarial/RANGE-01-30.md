# Сводка адверсального аудита: уроки 01–30

## Итог диапазона

Все 30 уроков получают **общий FAIL**. Причина не в отсутствии отдельных
хороших абзацев: центральные идеи чаще всего выбраны разумно, задачи заметно
лучше основного объёма, а несколько интерактивов действительно вычисляют
нужную модель. Причина в несоответствии жанру. Ни один текст не является
самодостаточной журнальной статьёй масштаба Лагутина/«Кванта»: до задач обычно
около 1300–1450 слов, ровно три SVG, один большой canvas, пять-шесть
текстовых заметок и одно inline-упражнение.

Начиная примерно с урока 16 особенно заметен машинный шаблон: первые 4–5
разделов кратко вводят тему, а следующие 4–5 почти повторяют её «более
подробно». XOR доказывается дважды, формы сети и softmax вводятся дважды,
потери, выпуклость, градиент, SGD, свёртка и pooling проходят повторным кругом.
Это увеличивает число заголовков, но не число математических переходов.

## Что ученик действительно получает

- устойчивую общую карту: данные → модели → neuron → optimization → vision;
- много правильных cautionary phrases о leakage, calibration, units,
  nonseparability, residuals and shift;
- самодостаточные финальные задачи с числами, seeds and algorithms во многих
  уроках;
- несколько настоящих derivations: OLS through origin (11), perceptron bound
  sketch (15), XOR contradiction (16), ReLU piecewise construction (19),
  convex local→global (21), scalar backprop (25), bilinear min–max (26);
- useful backlinks between neighboring topics.

Этого достаточно для хорошего расширенного outline, но недостаточно для
учебника: выводы обрываются, real-data examples остаются ссылками, figures не
образуют доказательство, а sidenotes почти всегда текстовые.

## Системные провалы

### 1. Объём и композиция

Каждая статья в 3–5 раз короче целевого журнального масштаба. Нет длинного
сквозного исследования с конфликтом гипотез, несколькими случаями,
возвращением к данным и развязкой. Ритм одинаков:

`короткое вступление → определение → формула → SVG → sidenote → widget →
ещё SVG → пять задач`.

Содержание не должно раздуваться повторением. Нужны новые derivations,
worked examples, counterexamples, data fragments and solutions.

### 2. Rich sidenotes фактически отсутствуют

В уроках есть текстовые callouts, но нет sidenote-изображений, портретов,
малых чертежей, проверяемых афоризмов/цитат и JS-calculations. История обычно
занимает один абзац и не связывает конкретную работу с текущей формулой.
Целевой набор на урок: 10–16 заметок, среди них 2–3 визуальных, 1 small JS,
2 counterexamples, 2 reveal-questions, 1 sourced historical fragment and
links backward/forward.

### 3. Три SVG стали чек-листом

Контактные листы 01–30 показывают повторяющуюся grammar:

- коробки-пайплайны;
- point clouds with one boundary;
- paired small plots;
- heatmaps/bars;
- faint arrows and small captions.

Полезные исключения есть (линеаризация Галилея, XOR coordinates, ReLU
construction, loss+derivative, convex chord, finite-difference U-curve,
pooling aliasing), но даже они не получают numerical source table,
reproducible parameters and close-up proof sequence. Нет sidenote visuals,
real raster fragments, documentary images and large explanatory diagrams.

### 4. Интерактивы часто не реализуют то, что обещает подпись

Критические случаи, требующие исправления до использования:

| Урок | Дефект |
|---:|---|
| 05 | вероятность над порогом считается ad hoc формулой вместо определённой probabilistic model |
| 06 | suitability percentages — произвольные linear combinations |
| 10 | «качество» budget allocation — произвольные linear scores, не active learning/RL |
| 11 | drag — скрытая эвристика, а promised residual plot отсутствует |
| 12 | residual risk — произвольная формула из sliders, не audit данных |
| 14 | adaptation update не совпадает с Hebb/Oja formulas статьи |
| 15 | widget synthetic and changes one point; это не заявленный Spambase experiment |
| 16 | morph slider двигает decorative marker, не вычисляет промежуточное representation |
| 18 | handcrafted index averages выдаются как network 25–4–2; «confidence» не calibrated |
| 19 | target автоматически interpolated at equal knots; это oracle, не ReLU constructor/training |
| 20 | quadratic loss произвольно делится на 12, prediction нельзя перетаскивать вопреки инструкции |
| 22 | selected and field arrows have opposite roles; dot product output hard-coded `0.00` |
| 23 | target inside feasible set ошибочно projected to boundary; inactive constraint невозможен |
| 24 | SGD заменён analytic gradient + invented isotropic noise, mini-batches не существуют |
| 27 | response is `abs(cos Δangle)`, not dot product displayed kernel/image; width has no effect |
| 28 | «EuroSAT fragment» procedurally invented in JS; response maps normalize separately |
| 29 | shift also changes noise seed, so translation experiment is confounded |
| 30 | skip toggle only draws a line between shape-incompatible tensors |

До расширения дизайна нужен принцип: **каждая цифра canvas должна либо
вычисляться из выписанной модели/данных, либо быть явно названа
иллюстративной**.

### 5. Реальные данные чаще служат декорацией

UCI, EMNIST, EuroSAT, MNIST, GTSRB and transit sources встречаются в prose and
homework, но main article почти никогда не содержит fixed snapshot, data
fragment, license/date/hash, exact filtering and reproducible result. Самый
жёсткий случай — урок 28, где synthetic generator прямо назван EuroSAT.

### 6. Домашка лучше статьи, но всё ещё не целевой листок

Финальные пять задач часто самодостаточны и содержательны: seeds, formulas,
exact arrays and requested artifacts указаны. Это сильная часть проекта.
Однако целевой формат требует 6–10 задач, 3–6 inline exercises, solution key,
visual construction, proof/counterexample and a genuinely embedded real-data
task. Сейчас обычно одна inline-задача; редакционных решений нет.

## Приоритет исправления

### P0 — убрать ложную математику и несоответствие интерфейса

Уроки 05, 06, 10, 12, 14, 19, 20, 23, 24, 27, 28, 29, 30. До этого их
widgets нельзя использовать как evidence в тексте.

### P1 — построить образцовые длинные статьи

Сначала переписать 11, 15, 19, 20, 22, 25, 28 and 29. Они покрывают
experiment, theorem/proof, approximation, loss, gradient, backprop,
convolution and sampling. На них следует закрепить единый article framework:
5000–7500 words, 8–10 main visuals, 10–16 rich notes, central lab + small
calculators, 4–6 inline and 6–10 final tasks.

### P2 — затем переносить форму по смысловым блокам

1. 01–06: scientific model, data and modes of learning.
2. 07–12: supervised/unsupervised methods and experimental protocol.
3. 13–20: neuron, representation, activation, approximation and loss.
4. 21–26: optimization, constraints, stochasticity, AD and games.
5. 27–30: local vision, convolution, pooling and CNN architecture.

Копировать можно editorial skeleton, но не количество figures, sequence of
headings or widget grammar. У каждого урока должен быть свой mathematical act:
fit, reconstruct, prove, search, sample, trace, attack, filter or compare.

## Минимальная приёмка после переписывания

- article body passes 4000-word partial threshold and normally reaches
  5000–7500 without repetition;
- no repeated introductory/expanded section pair;
- every formula has a derivation or a clearly marked status;
- each real-data claim points to a fixed reproducible fragment;
- every widget has a visible model, reset/step/scenario and calculated
  diagnostics;
- every SVG receives a 390/768/1440 close-up and 200% zoom check;
- notes include images, sourced history, small calculations and
  counterexamples;
- homework has only numbering+points in UI, complete conditions and an
  editorial solution key.
