# Диапазонный adversarial audit: уроки 31–60

## Итог без смягчения

Ни один из тридцати уроков сейчас не соответствует заявленному эталону
журнальной статьи уровня «Квант»/Лагутина и критериям `RUBRIC.md`. Это не
проблема отдельных опечаток или «ещё пары sidenotes». В диапазоне работает
генератор коротких конспектов с тремя однотипными SVG и одним canvas-widget,
тогда как нужен набор самодостаточных математических глав, где вывод,
численный пример, рисунок, интерактив и задачи проверяют один и тот же объект.

Суммарно здесь 39 979 слов, то есть в среднем 1332,6 слова на урок при целевых
5000–7500. Минимум — 1262 слова (урок 47), максимум — 1438 (урок 33):
малый разброс показывает не редакционную адаптацию к теме, а шаблонный лимит.
Есть 355 заголовков, 239 display-формул, 146 sidenotes и ровно 90 SVG, но
количество не превращается в плотность: формулы часто предъявлены без полного
вывода, sidenotes почти все текстовые, а у каждого урока механически ровно три
рисунка.

В уроках 31–40 есть всего 10 упражнений по ходу — строго по одному на урок; в
41–60 нет ни одного. В каждом уроке ровно 5 итоговых задач, то есть 150 на
диапазон, но рубрика требует 6–10 плюс 3–6 встроенных упражнений. Наборы часто
названы, но воспроизводимый snapshot, версия, лицензия, split manifest и
локальный provenance почти нигде не образуют единый эксперимент.

## Приоритет исправления

### P0 — ложное или внутренне противоречивое вычисление

Эти уроки нельзя «дополнять контентом» до исправления ядра: нынешний интерактив
создаёт у школьника неверное представление о том, что было вычислено.

1. **35 — автокодировщик.** `buildAutoencoder` смешивает реконструкцию с
   известным чистым target `clean`; это oracle leakage на inference. Удалить
   виджет и заменить честным linear-PCA AE или frozen denoiser.
2. **38 — PCA.** Toggle «среднее оставить» не работает по смыслу:
   `varianceAt` всё равно вычитает sample mean. До исправления виджет
   опровергает центральный тезис урока о центрировании.
3. **40 — звук.** Spectrogram рисуется Gaussian blobs, embedding — ручной
   формулой из centroid; FFT/STFT отсутствуют. Это имитация научного
   вычисления. Нужен настоящий typed-array FFT.
4. **32 — split.** «Видимая accuracy» равна
   `0.71 + leaking/8*0.18`; модели и labels нет. Либо реально обучать 1-NN на
   patient fingerprint, либо убрать метрику.
5. **33 и 59 — double descent.** Оба widget рисуют phenomenon аналитической
   авторской формулой, хотя подписи обещают обучения/запуски. Нужен реальный
   random-feature/SVD experiment; два урока надо развести или слить.
6. **37 — цена матричного умножения.** «Время» получается из операций,
   умноженных на ручные коэффициенты. Нужны cache simulation и настоящий
   benchmark с warmup/median/checksum.
7. **54 — LDA/QDA.** Ellipses и Gaussian scores используют correlation
   \(\rho\), но нарисованные samples сгенерированы без этой correlation. Data
   cloud не соответствует заявленной плотности.
8. **51 — regularization.** Lasso coefficient bars заданы эвристическими
   формулами и не решают objective; Huber/ridge/lasso panels относятся к
   разным вычислениям, но выдаются за один эксперимент.
9. **43 — ошибка прокурора.** Код берёт `LR=1/rate`, неявно ставя
   sensitivity=1, хотя текст использует 0,98. Исправить competing hypotheses,
   sensitivity и database-selection protocol.
10. **46 — coverage.** Wald endpoints визуально clamped к [0,1], поэтому
    скрывается основной дефект метода у границы. Нельзя обрезать математически
    неверный интервал молча.
11. **47 — Bayesian update.** Prior, likelihood и posterior независимо
    нормируются по peak; сравнивать концентрацию/площадь нельзя. Нужна общая
    density scale или явный режим.
12. **31 — digits.** Softmax от MSE до четырёх нарисованных шаблонов называется
    probability распознавания. Нет обученной модели, confusion matrix,
    calibration или EMNIST.
13. **56 — uncertainty.** Confidence — ручная функция расстояния до начала
    координат, не uncertainty модели. Coverage–risk вычисляется, но
    педагогически приписывается неподтверждённому confidence estimator.
14. **58 — ERM.** Individual losses не сэмплируются: true risk curve и noise
    сконструированы вручную; y-axis визуально инвертирована без ticks, а
    empirical value называется validation.
15. **52 — Bayesian regression.** Posterior lines используют равномерные
    псевдо-z вместо Gaussian draws; band hardcodes 90% через 1,64 без
    достаточной маркировки.

### P1 — восстановить математическую главу

После P0 надо не «нарастить абзацы», а переписать каждый урок по одному
проверяемому пути:

`реальная постановка → малая таблица → вывод → численный расчёт →
контрпример → edge cases → вычислительная лаборатория → реальные данные →
упражнения → итоговые задачи`.

Наиболее срочные главы:

- **31–35:** сейчас это серия лозунгов о vision, split, overfit, masks и
  representation; именно здесь пользователь уже увидел повторяемость и
  бессодержательность.
- **41–48:** базовая вероятность/статистика без единого встроенного упражнения
  формирует иллюзию понимания. Для каждого урока нужны 4–5 ручных задач прямо
  между выводами.
- **56:** проектный урок имеет лишь две display-формулы и не содержит
  воспроизводимого проекта.
- **57–60:** финал статистического блока повторяет старые темы (risk, split,
  overfit) без достаточного повышения математического уровня.

### P2 — заменить визуальный конвейер

После содержательного rewrite SVG надо строить от математики и одного
эксперимента, а не выбирать один из трёх шаблонов. Центральное требование:
каждый visual unit должен позволять проверить число, преобразование или
причинную связь. Все labels, scales, legends и units проверяются на реальном
desktop screenshot и в ширине основного полотна, а не только в исходном SVG.

### P3 — редакционная связность и backlinks

Backlinks следует превратить из ссылок по ключевым словам в учебный граф:
каждая ссылка отвечает «какой результат прошлого урока сейчас используется»
или «какая нерешённая проблема станет следующей». Дублирующиеся главы надо
развести по уровню результата.

## Повторяющиеся визуальные шаблоны

### Тройные heatmap/colorbar

| Урок | Рисунок | Что повторяется | Почему это мешает |
|---|---|---|---|
| 31 | fig.1 | три матрицы + узкая colorbar | числа/знаменатели слишком малы |
| 32 | fig.2 | три табличных split-панели | даты и group leaks не читаются |
| 34 | fig.1, fig.3 | image/mask triptych | colorbar и boundary details тесны |
| 35 | fig.3 | noise-transfer matrices | источник/масштаб условны |
| 37 | fig.1 | три access heatmaps | cache traces и labels исчезают |
| 39 | fig.1 | sparse/factor heatmaps | выбранный прогноз не прослеживается |

Это не шесть случаев «подходящей общей стилистики», а один способ заполнить
полосу. Для 31 нужна linked confusion matrix, для 32 — реестр визитов, для 34
— real crop/error map, для 35 — reconstruction gallery/transfer table, для 37
— адресная трасса, для 39 — factorization ledger.

### Округлые блоки и длинная зелёная нижняя стрелка

`32/fig-1`, `34/fig-2`, `35/fig-1`, `37/fig-2`, `40/fig-3`,
`47/fig-3`, `48/fig-1`, `50/fig-1`, `55/fig-1`, `58/fig-3`.

Общие дефекты: arrowheads входят в boxes; нижняя линия часто не имеет
однозначного источника/приёмника; маленькие shape labels заменяют explanation;
один и тот же pipeline grammar используется для доступа к test, U-Net,
autoencoder, cache, audio, Bayes, feature map, randomization и split — тем,
которые требуют принципиально разной геометрии. Этот шаблон надо запретить как
default. Он допустим только после ответа на вопрос, какое exact state несёт
каждое ребро и что вычисляет каждый узел.

### Гладкие кривые без эксперимента

`33/fig-2,3`, `39/fig-3`, `44/fig-1`, `45/fig-1,2`, `50/fig-2`,
`51/fig-1`, `56/fig-3`, `57/fig-1,2`, `58/fig-1,2`,
`59/fig-1,2`, `60/fig-2`.

Повторяются линии с подписями прямо на curve/right edge, без points, units,
seed, n, uncertainty band или источника. Там, где кривая теоретическая, нужна
формула и проверяемая точка. Там, где эмпирическая, нужны raw points,
replicates and intervals. «Красиво проведённая линия» не может одновременно
изображать theorem, simulation и benchmark.

### Малые multiples, которые невозможно читать

`33/fig-1`, `35/fig-2`, `37/fig-3`, `40/fig-2`, `44/fig-3`,
`46/fig-1`, `49/fig-3`, `52/fig-2`, `55/fig-3`, `59/fig-3`,
`60/fig-3`.

На desktop screenshot они технически помещаются, но labels, residual vectors,
folds, paths и axes оказываются ниже полезного размера. Решение не «увеличить
весь SVG», а выбрать один главный panel, linked inset и interactive reveal.

## Повторяющиеся интерактивные грамматики

### Ползунок меняет авторскую формулу, а не объект урока

- 31: MSE до четырёх templates → pseudo-probability;
- 32: count leaks → fabricated accuracy;
- 33: complexity → hand double-descent formula;
- 37: operation count → pseudo-time;
- 39: λ → ручное shrink dot product fixed factors;
- 40: frequency → нарисованные blobs/pseudo-embedding;
- 43: rarity → LR с sensitivity 1;
- 56: distance from origin → arbitrary confidence;
- 58: n/M → synthetic risk noise;
- 59: p/n → hand double-descent formula.

Общий критерий замены: если readout содержит accuracy, probability, time,
risk, spectrum или posterior, это число должно быть пересчитываемо из видимых
данных/модели, а не из дизайнерской формулы, придуманной для красивого
движения.

### Честная основа, но слишком узкое действие

- 34: реальные IoU/Dice fixed masks, но нет image/logits/threshold;
- 36: матричная геометрия хороша, но нет composition/conditioning;
- 41: dice Monte Carlo хорош, но нет event constructor/joint dependence;
- 42: exact Bayes counts, но rounded icon array and no repeated test;
- 44: CLT simulation, но нет Cauchy/dependence;
- 45: likelihood curves, но boundary/score/interval не раскрыты;
- 48: sequential Beta update, но нет groups/overdispersion;
- 49: fitted regression, но нет leverage separation/QQ/SE;
- 50: actual basis fit, но hardcoded RBF width/no conditioning;
- 53: actual geometric classifier, но threshold masquerades as probability;
- 55: repeated A/B experiments, но null/type-I and power conflated;
- 57: threshold/cost calculation, но caption promises a CE curve that is not
  drawn;
- 60: exact volume, но distances from centre substitute neighbour
  concentration.

Эти widgets можно сохранять как micro-tools, но центральный app должен
совершать полный педагогический цикл: predict before interaction, manipulate,
observe invariant/failure, explain from formula, verify on data.

### Внутренне несогласованные вычислительные слои

- 38: UI toggle and math disagree;
- 47: three curves have incomparable vertical normalization;
- 51: line fit and coefficient bars come from different procedures;
- 52: Gaussian posterior and non-Gaussian samples disagree;
- 54: density ellipses and drawn samples use different covariance.

Это отдельный класс QA: проверять не только отсутствие exception, а equality
between caption, state, formula, data generator, readout and pixels.

## Дубли и недостроенные мосты

| Пара/цепочка | Сейчас | Что должно отличаться |
|---|---|---|
| 31 ↔ 56 | calibration + selective risk повторены | 31 — classification diagnostics; 56 — decomposition, shift and project policy |
| 32 ↔ 58 | split/leakage повторены | 32 — protocol/access control; 58 — uniform selection optimism and bounds |
| 33 ↔ 59 | почти один double-descent урок | 33 — interpolation intuition; 59 — SVD/random-matrix derivation and experiment |
| 35 ↔ 38 | linear AE=PCA назван | один общий fully worked rank-1 example and backlink equation |
| 38 ↔ 39 ↔ 40 | vector representations | PCA compression → learned factors → signal retrieval with same notation |
| 42 ↔ 43 | Bayes then prosecutor | 43 must add selection, LR and dependent evidence, not repeat base rates |
| 45 ↔ 46 ↔ 47 | likelihood/CI/posterior | one shared coin dataset, then three inference readings compared explicitly |
| 49 ↔ 51 ↔ 52 | OLS/regularized/Bayesian | one shared X,y exposes change of objective and uncertainty |
| 53 ↔ 54 | discriminative/generative | same dataset, compare fitted scores, calibration and boundary assumptions |
| 55 ↔ 58 | multiple looks/multiple models | sequential decision versus adaptive model selection |

## Системные дефекты текста

1. **Фиксированная длина.** Все темы — от определения вероятности до
   double descent — сжаты до одного объёма. Математическая статья должна
   заканчиваться, когда завершён ход, а не когда достигнут шаблон 1300 слов.
2. **Формула как карточка, не как вывод.** Display math есть, но переходы,
   denominators, domains and assumptions часто пропущены. Нужен один
   непрерывный derivation thread на урок.
3. **Нет вычисленного сквозного объекта.** SVG используют условные данные,
   widget — другой hand dataset, homework просит третий. Ученик не может
   проследить одно число через статью.
4. **Контрпримеры перечислены.** Cauchy, leakage, OOD, boundary, cold start,
   singularity названы, но редко рассчитаны. Каждый главный тезис нуждается в
   worked counterexample.
5. **Edge cases не определяют API знания.** Empty mask, empty prediction,
   h=0/n, zero coverage, singular covariance, p≥n должны иметь явную
   convention/diagnostic.
6. **Real case не равен real data.** Ссылка на MNIST/MovieLens/SpaceNet/FMA
   без встроенного snapshot, version, license, split and baseline не создаёт
   практику.
7. **Sidenotes однотипны.** 146 заметок выглядят как текстовые «Вопрос
   автора/История/Будущее», но почти нет required mini-images, aphorisms,
   JS-calculators, exact counterexamples and facsimiles.
8. **Домашняя работа отделена от доказательств.** Пять задач часто
   самодостаточнее статьи, но нет промежуточной practice ladder и feedback.

## Системные дефекты visual QA

- labels frequently sit on curves (`33,38,39,42,50,51,53,56,57,59,60`);
- legends/labels approach right edge (`31,33,38,42,50,57,60`);
- colorbars crowd the third panel (`31,34,35,37,39`);
- arrowheads enter rounded blocks (`32,34,35,37,40,42,47,48,50,55,58`);
- clipped curves hide the claimed failure (`50`, partially `54`);
- too many paths make the claim unreadable (`46,55`);
- diagrams lack units, n, seed and data provenance even when they look
  empirical (`33,37,39,40,44,46,55,58,59,60`).

Visual validation must include, per page:

1. screenshot at target desktop width;
2. screenshot at narrower article width;
3. automated overflow/overlap probes where possible;
4. manual read of every label at 100%;
5. cross-check of every plotted number against source data;
6. explicit decision: theorem figure, simulation figure or empirical figure.

## Минимальный редакционный контракт для переписывания

Для каждого урока 31–60:

- 5000–7500 words before final tasks, 12–18 substantive sections;
- one running dataset/example, introduced early and reused in equations,
  visuals, widget and exercises;
- 6–10 visual units: at least one derivation diagram, one numeric plot, one
  counterexample, two small sidenote images;
- one unique central widget performing the main mathematical action plus 1–3
  micro-calculators;
- 10–16 diverse sidenotes, not four renamed text boxes;
- 3–6 in-body exercises and 6–10 final self-contained problems with points;
- dataset snapshot, source, license, version, units, split, seed and baseline;
- backlinks with a stated dependency/result;
- desktop visual QA and mathematical checksum before PASS.

## Рекомендуемый порядок производства

1. Исправить P0 correctness defects и поставить automated numeric assertions
   для each widget.
2. Слить/развести 33–59 и утвердить dependency graph 31–60.
3. Создать четыре shared real-data spines:
   `EMNIST/SpaceNet`, `MovieLens/FMA`, `medical/binomial`, `regression/A-B`.
4. Написать сначала одну эталонную главу каждого типа: 31 diagnostics, 36
   mathematics, 45 inference, 55 experiment/project.
5. Утвердить visual grammar на этих четырёх главах.
6. Переписать остальные, не копируя число/тип figures.
7. Провести independent mathematical review and screenshot audit.
8. Только после этого считать диапазон пригодным для ученического чтения.

**Общий вердикт диапазона 31–60: FAIL. Текущий материал годится как
редакционный outline, но не как готовый интерактивный учебник.**
