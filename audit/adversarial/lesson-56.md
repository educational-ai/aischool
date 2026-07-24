# Урок 56. Проект: карта неопределённости

## Главная педагогическая идея

Прогноз без границ знания неполон. Надо разделять aleatoric uncertainty,
epistemic uncertainty и distribution shift; проверять calibration/support и
строить selective system, где threshold выбирается по coverage, risk и цене
ручной проверки. Для проекта это должен быть полный журнал происхождения
данных, модели и решения.

## Фактическая инвентаризация

1325 слов, 12 заголовков, 2 display-формулы, 5 текстовых sidenotes, 3 SVG,
0 упражнений по ходу и 5 итоговых задач. Интерактив:
`public/interactive/widgets/g11-stats.js`, `buildUncertainty` (около строки
1562), ключ `56`.

Ученик назовёт три вида uncertainty, поймёт reliability diagram и
coverage–risk, увидит abstention. Но математическое содержание особенно
тонкое: decomposition только декларируется, нет построения интервала,
proper score, shift detector или проектного real-data pipeline.

## Чего не хватает

1. Закона полной variance
   \(Var(Y|x,D)=E[Var(Y|x,\theta)|D]+Var(E[Y|x,\theta]|D)\) с численным
   примером.
2. Чёткого различия confidence, entropy, variance, residual и error.
3. Расчёта ECE/Brier/NLL и неопределённости calibration.
4. Определений selective risk \(R(\tau)\), coverage \(C(\tau)\) и expected
   decision cost.
5. Крайних случаев: confident OOD, отсутствие принятых объектов, subgroup
   miscalibration.
6. Реального snapshot weather/demand map с spatial-temporal split, лицензией,
   uncertainty target и checklist оценки.

## Рисунки

### `figure-1.svg` — «Три причины одинаково широкого прогноза»

Три общих panels не имеют единых единиц или численных распределений; «одинаковую
ширину» нельзя проверить. Заменить одним predictive interval, разложенным на
цветные variance components, и тремя механизмами генерации, каждый с
проверяемой диагностикой.

### `figure-2.svg` — «Карта калибровки и плотность данных»

Reliability+histogram полезны, но подписи bins/counts малы, нет confidence
interval или subgroup. Добавить Wilson/bootstrap bands, bin mass и linked
brushing; не ставить labels линий у границ.

### `figure-3.svg` — «Кривая selective risk и охват»

Подписи сталкиваются около threshold/curves, у рабочей точки нет вывода cost.
Использовать coverage как монотонную x по uncertainty, показать
accepted/rejected counts, baseline случайного ranking и iso-cost lines;
отметить неопределённость risk при coverage 0.

## Интерактив

`buildUncertainty` рисует карту отказа и coverage–risk, но его “confidence” —
ручная функция, убывающая с расстоянием от начала, а не результат probability,
ensemble или residual модели. Ошибка возникает из двух ручных линейных
границ; shift двигает точки без переобучения. Это может показать провал, но
выдаётся за калиброванную уверенность. Cost равна `errors*cost + manual` для 80
точек без нормировки и группового breakdown.

Заменить прозрачной generative model, где отдельно видны true
\(P(Y=1|x)\), fitted model и uncertainty estimator. Режимы:
well-calibrated, overconfident, ensemble epistemic, OOD shift. Считать ECE,
Brier, coverage-risk и cost на фиксированных train/calibration/test snapshots;
threshold выбирать на calibration и замораживать для shifted test.

## Обязательные rich sidenotes

- JS-калькулятор разложения variance;
- афоризм: «Уверенность — прогноз модели о себе, а не гарантия мира»;
- портрет Владимира Вовка и точный вклад в conformal prediction;
- контрпример confident OOD;
- вопрос автора: какие объекты уходят человеку;
- анимация uncertainty calibration-bin;
- карточка entropy vs epistemic uncertainty;
- крайний случай zero coverage;
- мини-график subgroup calibration;
- мост к conformal prediction без преувеличений;
- паспорт проекта: data/version/license.

## Недостающие математические упражнения

1. Posterior содержит две равновероятные модели с прогнозами .2 и .8; каждое
   наблюдение имеет Bernoulli noise. Вычислите aleatoric, epistemic и total
   variance.
2. Для десяти пар confidence/correctness создайте два bins и вычислите ECE и
   Brier. Постройте вторую модель с той же accuracy, но худшим Brier.
3. Sorted uncertainty даёт counts/errors
   `(100,20),(80,8),(50,2),(10,0)`. Вычислите coverage, selective risk и cost
   при manual=1,error=15; выберите рабочую точку.
4. Общий ECE=.02, но у группы A .01, у B .18, а B — 5% данных. Объясните
   masking агрегатом и предложите отчёт.
5. Спроектируйте spatial-temporal split demand map: точные
   train/calibration/test периоды/регионы, shift, interval metric и abstention.

## Пошаговый план переписывания

1. Начать с одного прогноза и трёх источников uncertainty.
2. Численно вывести total variance.
3. Развести confidence/calibration/error.
4. Вывести coverage-risk-cost.
5. Добавить shift/OOD/subgroups.
6. Построить протокол реального map-project.
7. Заменить произвольный confidence проверяемой моделью.
8. Добавить 4–5 упражнений, rubric проекта и 7–8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | FAIL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
