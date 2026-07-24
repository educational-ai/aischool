# Урок 58. Эмпирический риск: учимся по выборке

## Главная педагогическая идея

True risk — ожидание по будущему распределению, empirical risk — среднее по
выборке. Для фиксированной функции оценка колеблется, но выбор минимума среди
многих кандидатов эксплуатирует случайный шум; generalization требует
контролировать размер класса, split и distribution shift.

## Фактическая инвентаризация

1293 слова, 12 заголовков, 6 display-формул, 5 текстовых sidenotes, 3 SVG,
0 упражнений по ходу и 5 итоговых задач. Интерактив:
`public/interactive/widgets/g11-ml.js`, `buildErm` (около строки 172), ключ
`58`.

Ученик отличит \(R\) и empirical \(R\), поймёт winner's curse и интуицию
Hoeffding, свяжет split с вопросом deployment. Он не выведет uniform bound, не
проведёт опыт среднего по individual losses и не отделит validation selection
от test estimation.

## Чего не хватает

1. Видимого списка individual losses, среднее которого равно empirical risk.
2. Численной инверсии Hoeffding для фиксированной h и union bound по M.
3. Вывода/симуляции optimism выбора на repeated validations.
4. Structural risk/regularization как штрафа сложности.
5. Крайних случаев: adaptive candidates и зависимые примеры.
6. Реального snapshot поездок с row/time/driver split и shift report.

## Рисунки

### `figure-1.svg` — «Эмпирический риск по повторным тестовым выборкам»

Три density curves передают \(1/\sqrt n\), но true-risk line и sampling
distributions требуют общей шкалы, n/числа повторов и confidence. Заменить
гладкие придуманные плотности гистограммами воспроизводимой Bernoulli-loss
simulation.

### `figure-2.svg` — «Победитель получает удачный шум»

Пары true/validation полезны, но легенда не объясняет вертикаль и selected
index; оси пусты. Добавить oracle/selected lines, distribution selection gap
по повторам и held-out test point.

### `figure-3.svg` — «Три разбиения отвечают на три вопроса»

Снова rounded-block split diagram; стрелки и подписи малы. Использовать одну
таблицу поездок с driver/time, связями leakage и точным описанием будущей
population для каждого split.

## Интерактив

`buildErm` не сэмплирует individual losses. Он вручную строит smooth
“true risk” по индексу кандидата и добавляет шум \(1/\sqrt n\), затем выбирает
winner/oracle. Эффект виден, но y-mapping инвертирован без ticks: больший risk
рисуется ниже. Empirical value называется “validation”, хотя validation sample
нет. Нет distribution repeated selection или test set.

Заменить actual Bernoulli losses для M фиксированных classifiers на n
объектах. Ученик открывает loss matrix, усредняет столбцы, выбирает validation
winner и раскрывает independent test. 500 seeds показывают optimism по M,n;
поверх — Hoeffding+union bound. Temporal/group split вынести на real snapshot.

## Обязательные rich sidenotes

- JS-среднее видимых 0/1 losses;
- aphorism: «Минимум измерений наследует самый удачный шум»;
- портреты Владимира Вапника/Алексея Червоненкиса и точный контекст uniform
  convergence;
- контрпример adaptive reuse validation;
- вопрос автора: сколько моделей пробовали;
- калькулятор Hoeffding bound;
- анимация selection gap;
- крайний случай коррелированных кандидатов;
- карточка target каждого split;
- мост назад к train/val/test урока 32 и вперёд к double descent;
- журнал эксперимента.

## Недостающие математические упражнения

1. Losses трёх моделей на пяти объектах:
   `A:0,1,0,0,1; B:0,0,1,0,0; C:1,0,0,0,0`. Вычислите empirical risks и
   ERM; как один новый объект меняет winner?
2. Hoeffding says \(P(|\hat R-R|>\epsilon)\le2e^{-2n\epsilon^2}\).
   При n=200, ε=.1 вычислите bound; примените union bound к M=50.
3. После 20 model searches validation winner имеет .16, independent test .23;
   baseline .21/.22. Оцените optimism и выберите deployment.
4. Постройте зависимые losses, нарушающие iid: десять duplicate rides на
   driver. Объясните effective sample и group split.
5. Задайте воспроизводимый selection experiment: M,n,seeds, validation,
   untouched test, plotted quantiles и stopping rule.

## Пошаговый план переписывания

1. Начать с видимых строк loss.
2. Определить true/empirical risk.
3. Показать concentration фиксированной функции.
4. Добавить union bound/model selection.
5. Показать winner's curse на independent test.
6. Добавить split/shift/dependence.
7. Заменить hand-noise виджет симуляцией loss matrix.
8. Добавить 4–5 упражнений и 7–8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
