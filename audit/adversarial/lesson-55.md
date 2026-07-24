# Урок 55. Статистический практикум: опрос и A/B-тест

## Главная педагогическая идея

Рандомизация делает группы сравнимыми в среднем, а не гарантированно.
Эффект, uncertainty, power, stopping rule, multiplicity и interference должны
быть определены до анализа; p-value — вероятность хвоста при null, а не
вероятность истинности null.

## Фактическая инвентаризация

1321 слово, 11 заголовков, 7 display-формул, 4 текстовых sidenotes, 3 SVG,
0 упражнений по ходу и 5 итоговых задач. Интерактив:
`public/interactive/widgets/g11-stats.js`, `buildAb` (около строки 1473),
ключ `55`.

Ученик понимает randomization/potential outcomes, p-value, power,
pre-registration, multiple metrics и peeking. Он не выполняет полный
randomization inference, не рассчитывает размер выборки и не анализирует
attrition/noncompliance.

## Чего не хватает

1. Конкретного назначения 20 людей и exact randomization distribution.
2. Estimand: разность means/proportions, ITT против treatment-on-treated.
3. Вывода standard error и minimal detectable effect.
4. Ошибок I/II рода и power curve по effect/sample.
5. Крайних случаев: imbalance, attrition, cluster interference, skew metric.
6. Реального анонимизированного snapshot опыта и preregistration manifest.

## Рисунки

### `figure-1.svg` — «Рандомизация создаёт сравнимые группы»

Снова block pipeline с наконечниками внутри узлов; distributions после split
схематичны и всегда выглядят сбалансированными. Показать одну реальную
рандомизацию с различием covariates, затем distribution imbalance по 1000
назначениям. Рандомизация гарантирует распределение, не равенство каждой
таблицы.

### `figure-2.svg` — «p-value как хвост нулевого распределения»

Это самый ясный рисунок, но нужны observed statistic, точная сумма хвоста и
численное различие one/two-sided. Добавить CI эффекта и отметить, что p не
измеряет величину эффекта.

### `figure-3.svg` — «Подглядывание раздувает ложные открытия»

Сто траекторий нечитаемы; нет выделенного пути и доли пересечений threshold.
Оставить 20 бледных paths и 3 выделенных примера, running false-positive
proportion; явно подписать время/число просмотров и boundary.

## Интерактив

`buildAb` моделирует 50 экспериментов, confidence intervals и fixed/peeking
stop; основа содержательна. Но красные интервалы называются “significant” и
при ненулевом эффекте, поэтому output смешивает false positives и power.
Данные — суммы uniform, SE фиксирован аналитически; реальной binary metric нет.
Не показаны p-value, randomization, multiplicity, attrition или stopping
correction.

Добавить вкладки null/power и сообщать ошибку I рода только при effect=0,
power — при заданном эффекте. Показывать sequential trajectories и stopping
times. Реализовать exact permutation test для \(n\le20\), normal approximation
для больших n и сравнение Bonferroni/FDR при нескольких metrics.

## Обязательные rich sidenotes

- JS-рандомизация 20 именованных анонимных units;
- афоризм: «Рандомизация балансирует процедуру, не каждую таблицу»;
- портрет Николая Смирнова с точным вкладом в testing;
- контрпример p=.049 при ничтожном эффекте;
- вопрос автора о влиянии школьников друг на друга;
- калькулятор power/MDE;
- анимация peeking;
- крайний случай dropout только в B;
- карточка ITT vs per-protocol;
- preregistration JSON;
- мост к выбору по empirical risk урока 58.

## Недостающие математические упражнения

1. Outcomes восьми units: `2,3,3,4,5,6,7,8`; четыре назначаются в B. Для
   заданного assignment вычислите разность means, затем переберите все 70
   assignments и exact two-sided p-value.
2. Доли 0.10 и 0.12 при 500 пользователях в каждой группе. Вычислите standard
   error, z-statistic и 95% CI разности; разделите absolute/relative lift.
3. При \(\alpha=.05\) и 20 независимых metrics вычислите family-wise
   false-positive probability и Bonferroni threshold.
4. В B 20% пользователей уходят после treatment, в A — 5%. Объясните, почему
   complete-case ломает randomization, и задайте ITT analysis.
5. Спроектируйте A/B школьного расписания: unit, randomization, primary metric,
   horizon, MDE, stopping rule, interference risk и итоговую таблицу.

## Пошаговый план переписывания

1. Начать с одного конечного randomization experiment.
2. Определить estimand и вычислить exact null distribution.
3. Раздельно ввести p-value, CI и эффект.
4. Вывести SE и power/MDE.
5. Добавить peeking/multiplicity.
6. Добавить attrition/interference/cluster design.
7. Расширить виджет до null/power/sequential lab.
8. Добавить 4–5 упражнений, preregistration и 7–8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | PARTIAL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
