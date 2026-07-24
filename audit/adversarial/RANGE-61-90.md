# Adversarial audit: уроки 61–90

## Итог

**Все 30 уроков получают общий вердикт FAIL.** Это не означает, что в них нет
полезного материала. В диапазоне есть сильные конспекты и местами очень хорошие
условия финальных задач. Провал относится к заявленному стандарту
самодостаточной журнальной статьи: объём в 3–5 раз меньше цели, на каждый урок
приходятся ровно три SVG и один ползунковый widget, rich sidenotes и inline
exercises почти отсутствуют, а значительная часть интерактивов математически не
реализует названный метод.

Проверены:

- `site/audit/adversarial/RUBRIC.md` целиком;
- 30 markdown-файлов `site/content/lessons/61.md`–`90.md`;
- все 90 SVG в `site/public/figures/lessons/61`–`90`;
- соответствующие desktop screenshots и figure contact sheets;
- mapping и implementations
  `buildBatchTraining`–`buildOrchestra` в трёх widget bundles.

## Количественный разрыв

| Объект | Фактически, уроки 61–90 | Минимум рубрики | Разрыв |
|---|---:|---:|---:|
| Слова до задач | 44 436, среднее 1481 | 150 000 | −105 564 |
| Разделы | 344, среднее 11,5 | 360 | −16, при этом многие разделы состоят из 1–3 абзацев |
| SVG/рисунки | 90, ровно 3 на урок | 180 | −90 |
| Центральные widgets | 30 | 30 | номинально выполнено |
| Малые JS-опыты сверх центрального | 0 | 30–90 | −30 как минимум |
| Sidenotes | 121, в среднем 4,0 | 300 | −179 |
| Формальные inline exercises | 0 | 90 | −90 |
| Финальные задачи | 150, ровно 5 на урок | 180 | −30 |
| Display-формулы | 241 | отдельного минимума нет | формул достаточно, выводов недостаточно |

Медиана статьи — около 1483 слов; shortest 61 — 1309, longest 85 — 1655.
Вместо диапазона 5000–7500 слов получился единообразный шаблон примерно на
1500 слов. Ровно три фигуры, четыре notes, один widget и пять tasks почти в
каждом уроке — след генератора структуры, а не редакционного решения по теме.

## Самый серьёзный дефект: интерактив часто лжёт о вычислении

В `61`–`90` нет ни одной кнопки `run`, `step` или `reset`: все центральные
«лаборатории» управляются только sliders/segmented controls и перерисовывают
один canvas/readout. Это уже не соответствует рубрике. Хуже того, часть
виджетов подменяет алгоритм авторской формулой.

| Урок | Builder | Что происходит фактически | Вердикт |
|---:|---|---|---|
| 61 | `buildBatchTraining` | рисованные loss curves и шум, без данных/gradient updates | FAIL |
| 62 | `buildOptimizers` | quadratic считается, но AdaGrad и Adam тайно получают разные множители learning rate | FAIL |
| 63 | `buildOracle` | deterministic parabola + игрушечный validation noise; реальный test отсутствует | FAIL |
| 64 | `buildFederated` | три scalar quadratics; server rounds визуально сдвигаются пикселями | FAIL |
| 65 | `buildPageRank` | настоящий PageRank и seeded click simulation, но без residual/editable \(v\)/step | PARTIAL |
| 66 | `buildRandomWalk` | настоящие walks, но без seed/control, hitting time, ESS-like diagnostics | PARTIAL |
| 67 | `buildMcmc` | настоящая bimodal chain, но histogram и target имеют несовместимые нормировки | PARTIAL |
| 68 | `buildCipher` | реальный toy trigram MCMC; лучший widget диапазона, но corpus микроскопичен | PARTIAL |
| 69 | `buildBellman` | настоящее value iteration на fixed grid, без пошагового backup/Q-gap | PARTIAL |
| 70 | `buildBandit` | «Thompson» — clipped Gaussian approximation, не Beta sampling | FAIL |
| 71 | `buildActorCritic` | один slider означает epsilon у Q и temperature у actor; сравнение несопоставимо | FAIL |
| 72 | `buildSelfPlay` | soft best response к среднему прошлому, без matches, league и MCTS | FAIL |
| 73 | `buildRnn` | sensitivity = `memory^(t-2)`, полностью игнорирует tanh Jacobians | FAIL |
| 74 | `buildLstm` | сравнение двух экспонент, без LSTM gates/state/tanh/training | FAIL |
| 75 | `buildTokens` | честный tiny BPE, но fixed corpus/strings и без frequencies/bytes | PARTIAL |
| 76 | `buildAttention` | нужная связь hard-coded условием `query===6 && key===2`, не \(QK^\top\) | FAIL |
| 77 | `buildTransformerBlock` | invented scalar norm formula, transformer block не выполняется | FAIL |
| 78 | `buildPretraining` | «skills» — произвольные линейные функции mixture sliders | FAIL |
| 79 | `buildScaling` | invented coefficients/quality; нет fit, source, residual or uncertainty | FAIL |
| 80 | `buildPreferences` | Bradley–Terry fit частично настоящий, но disagreement — один hard-coded cycle | PARTIAL |
| 81 | `buildAlignment` | PPO/DPO/GRPO отличаются только произвольными multipliers | FAIL |
| 82 | `buildGan` | collapse заранее задан; игроков и minimax updates нет | FAIL |
| 83 | `buildStyleAttack` | декоративное лицо, arbitrary classifier formula, no gradient/model | FAIL |
| 84 | `buildMultimodal` | synthetic centers and hand-written alignment, no encoders/images/text | FAIL |
| 85 | `buildVae` | arbitrary KL/reconstruction; interpolation — alpha-blend glyphs `3`/`8` | FAIL |
| 86 | `buildDiffusion` | reverse uses clean \(x_0\) directly; oracle crossfade вместо denoiser | FAIL |
| 87 | `buildDeepfake` | полезный base-rate calculator, но Gaussian scores/shift invented and unlabeled | PARTIAL |
| 88 | `buildVideoWorld` | physics slider moves prediction toward known ground truth | FAIL |
| 89 | `buildAgent` | independent reliability formula, no tools/actions/injection/gate | FAIL |
| 90 | `buildOrchestra` | checklist/radar with arbitrary scores, no artifacts/gates | FAIL |

Уроки 73, 76, 81, 82, 85 и 86 особенно опасны: интерфейс показывает числа,
формы и trajectories, которые выглядят как результат заявленного алгоритма,
но вычисляются другим способом. До исправления эти widgets следует либо
снять с публикации, либо явно назвать «схематической иллюстрацией» и не делать
по ним предметных выводов.

## Визуальный аудит

Проблема не в минимализме как таковом. Minimal article layout близок к
нужному направлению, но editorial density низка: в viewport почти всегда один
абзац, формула/рисунок и один узкий текстовый note. Нет плотного ритма
«доказательство — схема — вычисление — контрпример — вопрос на полях».

Наиболее конкретные дефекты:

1. `64/figure-1.svg`: клиентские стрелки сходятся в unreadable spaghetti,
   arrowheads перекрывают \(w_t\to w_{t+1}\).
2. `70/figure-2.svg`: подписи показывают буквальный текст `A\nN=20` и т. п.
3. `72/figure-1.svg`: подпись `ножницы` обрезана слева; matrix lacks values/
   colorbar.
4. `73/figure-1.svg`: hidden lines идут под cells, направление слабое.
5. `74/figure-2.svg`: labels gates сталкиваются с нулевыми ticks/baselines.
6. `77/figure-1.svg`: residual bypass routes visually disconnected from
   add nodes.
7. `78/figure-2.svg`: arrows and bars imply quantitative token flow without
   axes/counts/source.
8. `88/figure-2.svg`: epipolar geometry incomplete; planes/rays/projected
   points and red line do not form a correct teaching construction.
9. Все три figures урока 89 практически arrowless, хотя описывают state,
   evidence and trust flows.
10. `90/figure-3.svg`: rotated x labels collide with the printed figure footer
    in contact render.

Часть фигур 65–70 и 82/85/86 выглядит профессионально как matplotlib plate,
но captions систематически обещают больше, чем источник содержит: реальные
данные, uncertainty, arrows, fit bands или metrics не видны. Каждый caption
должен проходить буквальную проверку: любой заявленный элемент обязан быть в
SVG и читаться на фактической ширине страницы.

## Что ученик действительно получает

Сильнее всего сейчас работают:

- 65–70: Markov/PageRank/random walk/MCMC/cipher/Bellman/bandit образуют
  разумную математическую линию;
- 73–77: текстовая линия RNN→LSTM→tokens→attention→transformer логична;
- 80–82: preferences→alignment→GAN имеют понятные локальные идеи;
- 85–87: ELBO/diffusion/base-rate содержат важные формулы и хорошие
  предостережения;
- 89–90: system-level безопасность и воспроизводимый capstone сформулированы
  зрелее большинства школьных материалов.

Но understanding останавливается на уровне «могу пересказать механизм».
Ученику редко дают:

- полностью вычисленную модель от сырых строк до результата;
- proof chain with assumptions named;
- настоящий контрпример, который можно запустить;
- real data slice embedded next to plot;
- interactive state that corresponds exactly to equations;
- uncertainty/repeated runs;
- inline practice before a large homework specification.

## Задачи: сильные условия, слабая лестница

Финальные задачи 65–90 гораздо содержательнее статей. Они часто фиксируют
version, revision, seed, split, preprocessing, metrics and acceptance
criteria — это достоинство. Однако они не образуют школьную лестницу:

- ровно пять задач вместо 6–10;
- 0 formal inline exercises before homework;
- задачи 6–7 часто требуют downloads, GPU training, 30–50 seeds, human
  raters, large archives and hours/days compute;
- article does not derive or demonstrate many procedures demanded by task;
- отсутствуют intermediate answer checks and expected artifacts;
- points 3–7 do not correspond to reliable workload increments.

Нужно сохранить полноту условий, но разбить большие задачи на checkpoints:
ручной calculation → tiny browser replica → fixed precomputed artifact →
optional full research extension. Для каждого урока нужны как минимум два
proof/derivation задания, два exact numeric, drawing, counterexample,
seeded browser experiment, real-data task and open transfer.

## Sidenotes: основной авторский голос пока не построен

Вместо требуемых 10–16 notes на урок сейчас почти везде четыре коротких
текстовых callout. Нет sidenote-images, interactive marginalia, sourced
quotes, paired counterexamples, annotated historical documents. Notes часто
повторяют шаблон `Контроль: ...`, `Автор`, `Проверка`, не образуя
параллельный авторский маршрут.

На каждый урок нужен план полей:

1. 2–3 image notes (portrait, manuscript/device/data fragment);
2. one exact sourced quote/aphorism;
3. one tiny JS calculator;
4. two counterexamples;
5. two questions with delayed reveal;
6. one historical note, preferably with Russian scientist only where
   connection is real and specific;
7. one bridge back and one forward;
8. one warning about assumptions/data/units.

## Системные данные и backlinks

Названия NASA C-MAPSS, NOAA drifters, Web-Google, OPSD, HH-RLHF, FineWeb,
Fashion-MNIST, FaceForensics++, Waymo/Argoverse and others appear often, but
article body rarely includes a reproducible sample. Data references need one
uniform card: origin, licence/terms, snapshot/revision, checksum, unit of
observation, columns/units, split and limitations.

Wikipedia-style links are not mechanically trustworthy. Concrete example:
lesson 65 links `/lesson/63` as Markov chains and `/lesson/62` as eigenvectors,
while those pages are oracle selection and optimizers. Before expanding
backlinks, add a link audit that compares anchor semantics with target title.

## Поурочные приоритеты

| Урок | Сильное ядро | Первое обязательное исправление |
|---:|---|---|
| 61 | variance mini-batch, early stopping | настоящий SGD engine и derivation variance |
| 62 | adaptive optimizers | убрать скрытые LR multipliers, вывести dynamics |
| 63 | oracle selection | proof \(2\epsilon\) and repeated selection experiment |
| 64 | non-IID/FedAvg | exact two-client drift and readable central scheme |
| 65 | PageRank fixed point | contraction/residual, editable \(v\), fix backlinks |
| 66 | \(\sqrt n\) scaling | hitting-time lab and real drifter provenance |
| 67 | MH diagnostics | common histogram normalization + multiple chains |
| 68 | cipher MCMC | corpus/smoothing/restarts, richer diagram |
| 69 | Bellman | stepwise backup/Q-gap and contraction proof |
| 70 | regret/UCB/Thompson | implement actual Beta Thompson; fix literal `\n` |
| 71 | value vs actor | fair exploration semantics and actual updates |
| 72 | self-play/league | real matches, population and MCTS backup |
| 73 | BPTT memory | replace false power curve by tanh Jacobian |
| 74 | gated memory | replace exponential demo by actual LSTM |
| 75 | tokenizer economics | editable corpus/BPE trace/Unicode microscope |
| 76 | Q/K/V | remove hard-coded semantic score |
| 77 | block grammar | execute real tiny transformer block |
| 78 | corpus pipeline | real documents/filter ledger instead of skill sliders |
| 79 | compute optimum | fit actual run table and derive optimum |
| 80 | preference graph | group votes, uncertainty and cycles |
| 81 | KL alignment | implement real finite PPO/DPO/GRPO updates |
| 82 | minimax game | actual two-player gradient dynamics |
| 83 | style + attacks | split or double article; two real experiments |
| 84 | contrastive pairs | real image/text embeddings and exact cosine loss |
| 85 | ELBO | real fixed VAE, no glyph crossfade |
| 86 | reverse diffusion | remove oracle access to \(x_0\) |
| 87 | base rates/evidence | real detector-score snapshot and ROC–PR |
| 88 | time/3D/action | correct epipolar drawing and true simulator |
| 89 | agent loop | tool/action/injection trace with capability gate |
| 90 | capstone method | one fully worked project and artifact validator |

## Рекомендуемая последовательность переработки

### Волна 0: остановить математически ложные demonstrations

73, 76, 77, 78, 81, 82, 83, 84, 85, 86, 88, 89, 90; затем 70, 71, 72.
До замены не использовать их outputs как evidence в тексте.

### Волна 1: сделать эталонные статьи

Выбрать 65 (дискретная математика/граф), 68 (алгоритмический практикум), 76
(линейная алгебра/нейросеть), 87 (decision under uncertainty), 90 (project).
Довести каждую до полного стандарта рубрики и использовать как five genre
templates, а не один универсальный шаблон.

### Волна 2: расширить сильные математические цепочки

61–64, 66–72, 73–75, 77–82, 84–89. Для каждого сначала написать derivation
spine and data contract, затем figures/widgets, затем sidenotes/tasks.

### Волна 3: визуальная и ссылочная QA

Проверить every SVG at actual column width and retina scale; caption-to-source
check; arrowhead/text collision scan; keyboard/fallback states; link-anchor
audit; browser screenshots for initial, edge and mobile states.

## Финальный вердикт по категориям

| Категория | Диапазон 61–90 |
|---|---|
| A. Глубина | FAIL: 44 436 слов вместо минимум 150 000 |
| B. Математический ход | PARTIAL: формул много, выводы/assumptions редки; несколько widgets противоречат формулам |
| C. Рисунки | FAIL: ровно 3 вместо 6–10, почти нет photo/vector marginalia, есть конкретные layout errors |
| D. Интерактив | FAIL: 0 run/step/reset, 0 small JS notes, много hand-authored proxies |
| E. Sidenotes | FAIL: 121 вместо минимум 300, почти только текстовые вопросы |
| F. Упражнения | PARTIAL: финальные условия сильны, но 0 inline и research cliff |
| G. Данные | PARTIAL: названия datasets есть, reproducible article snapshots почти отсутствуют |
| H. Связность | PARTIAL: тематическая линия хороша, backlinks неаудированы и местами неверны |

**Общий вердикт диапазона: FAIL.**
