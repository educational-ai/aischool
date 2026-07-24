# Урок 57. От статистики к машинному обучению

## Главная педагогическая идея

Loss кодирует, что значит хороший вероятностный прогноз или полезное решение.
Likelihood часто выводит loss из модели, но инженерная стоимость действует
после прогноза; proper scoring rules, surrogate, quantile и ranking losses
выбирают разные статистические функционалы.

## Фактическая инвентаризация

1347 слов, 12 заголовков, 11 display-формул, 4 текстовых sidenotes, 3 SVG,
0 упражнений по ходу и 5 итоговых задач. Интерактив:
`public/interactive/widgets/g11-ml.js`, `buildRisk` (около строки 83), ключ
`57`.

Ученик отличит likelihood-loss от decision cost, поймёт proper scores,
surrogate, quantile и ranking examples. Он не докажет properness, не построит
Bayes action для нескольких losses и не увидит несовпадение training surrogate
с product metric.

## Чего не хватает

1. Минимизации expected loss: squared→mean, absolute→median,
   pinball→quantile.
2. Доказательства properness cross-entropy для Bernoulli через производную.
3. Вывода decision threshold из цен FP/FN.
4. Различия calibration/discrimination и threshold-free ranking.
5. Крайних случаев: нулевые вероятности в log loss, class imbalance и metric
   gaming.
6. Реального поискового/медицинского mini-dataset и карточки метрики.

## Рисунки

### `figure-1.svg` — «Вероятность одна, решения различны»

Линии expected loss полезны, но подписи лежат на кривых; значения/единицы cost
и уравнение threshold не показаны. Добавить вертикальный marker p и две
арифметические карточки loss.

### `figure-2.svg` — «0/1-loss и три суррогата»

Подписи перекрывают кривые, формы подходят к верхнему clipping. Использовать
aligned small multiples или внешнюю легенду, отметить производные и указать,
какая loss является верхней оценкой 0/1.

### `figure-3.svg` — «Разные потери выбирают разные точки распределения»

Это самый сильный рисунок: markers распределения передают
mean/median/quantile. Но нижние формы loss малы и не связаны вертикальными
направляющими. Добавить численную выборку и empirical minimizers.

## Интерактив

`buildRisk` генерирует calibrated/miscalibrated probabilities и перебирает
threshold, рисуя только accuracy и cost. Setup обещает curves accuracy,
cross-entropy и cost, но cross-entropy не зависит от threshold и существует
лишь как постоянный readout. Несоответствие смешивает качество score с
decision threshold. Calibration transform монотонна, поэтому discrimination
фиксирована, но это не объясняется.

Показывать cross-entropy/Brier на отдельной calibration panel по параметру
калибровки, не threshold. Threshold panel должна показывать accuracy, FP/FN
cost и confusion. Добавить ожидаемый Bayes threshold и finite-sample optimum.
Третий режим выбирает squared/absolute/pinball и показывает появление
mean/median/quantile.

## Обязательные rich sidenotes

- JS-минимизатор expected loss;
- афоризм: «Прогноз описывает веру; порог выбирает действие»;
- портрет Владимира Вапника и связь с минимизацией риска;
- контрпример: идеальный ranking, плохая calibration;
- вопрос автора: чья cost задаёт loss;
- карточка нулевой вероятности в log loss;
- производная proper score;
- анимация quantile;
- пример metric gaming;
- мост назад к likelihood и вперёд к empirical risk;
- паспорт product metric.

## Недостающие математические упражнения

1. \(Y=0\) с вероятностью .7 и \(Y=10\) с вероятностью .3. Найдите прогноз,
   минимизирующий expected squared, absolute и pinball loss при \(\tau=.9\).
2. Для Bernoulli с истинной q продифференцируйте expected log loss
   \(-q\log p-(1-q)\log(1-p)\) и докажите минимум при \(p=q\).
3. Выведите optimal threshold при FP=2,FN=8; сравните действия при
   \(p=.15,.25,.7\).
4. Две модели имеют один ranking, но вероятности `(.9,.8,.2,.1)` и
   `(.6,.55,.45,.4)`. На labels `1,0,1,0` сравните AUC и вычислите log loss.
5. Спроектируйте audit surrogate/product metric для top-5 поиска: training
   loss, offline metric, online cost, failure slice и stopping rule.

## Пошаговый план переписывания

1. Начать с Bayes action при заданной loss.
2. Вывести mean/median/quantile.
3. Вывести log loss/properness.
4. Развести scoring и decision cost.
5. Объяснить surrogate и ranking.
6. Добавить calibration/metric gaming.
7. Разделить виджет на threshold/scoring panels.
8. Добавить 4–5 упражнений и 7–8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | PARTIAL |
| D. Интерактив | PARTIAL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
