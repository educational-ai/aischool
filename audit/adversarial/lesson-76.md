# Урок 76. Внимание и трансформер

## Главная педагогическая идея

Attention разделяет адрес \(Q/K\) и переносимое содержание \(V\), нормирует
scores softmax и создаёт для каждой позиции динамическую выпуклую смесь.
Маска меняет допустимое множество, а positional information возвращает
порядок. Центральная лаборатория обязана вычислять \(QK^\top/\sqrt d\), иначе
урок опровергает сам себя.

## Что есть сейчас

1523 слова до задач, 11 разделов, 9 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildAttention`) и 5 задач. Проверены все SVG, source и screenshot. Текст
вводит Q/K/V, scaling, masks, multi-head, positions, quadratic cost,
interventions и GAP-coreference task. Финальные задачи сильные.

Ученик поймёт формулы attention и различит key/value. Но интерактив не
исполняет эти формулы; значит главный опыт формирует ложную причинную модель.

## Чего не хватает в рассуждении

1. Полного числового примера \(X,W_Q,W_K,W_V\) для 3 tokens с matrix shapes,
   scores, row-softmax и \(AV\).
2. Вывода variance dot product с явными covariance terms; сейчас он дан
   словесно.
3. Jacobian softmax \(\partial a_i/\partial s_j\), который объясняет
   конкуренцию weights.
4. Доказательства permutation equivariance without positions.
5. Counterexample high attention/zero causal effect через value=0 или
   compensating residual path.
6. Exact memory units for \(n^2\) matrix at fp16/fp32.
7. Реального attention/intervention trace с model revision.

## Рисунки и интерактив

### `figure-1.svg`

Query-lines визуально направлены от query к positions, но неясно, где
сравниваются keys и где смешиваются values; arrow labels мелки. Нужны три
последовательные панели `scores → softmax → weighted values` с матрицами.

### `figure-2.svg`

Heatmaps читаются, но color scale/числа и row-sum checks отсутствуют. Добавить
одну выбранную row, denominator до/после mask и exact zeros.

### `figure-3.svg`

Кривые стоимости выглядят убедительно, но правая «доля дальних связей»
не имеет data source, task или uncertainty. Нужны exact memory MiB/GiB для
\(H,n,dtype\) и отдельный reproducible dependency-length dataset.

### `buildAttention`

Критический дефект: scores не вычисляются из \(QK^\top\). В коде вручную
зашито `if (query === 6 && key === 2) score += 2.7`; это авторская связь
местоимения со словом, замаскированная под attention. Controls меняют query,
temperature и mask, но не vectors, key/value. Текст просит изменить value при
фиксированных weights — невозможно. Заменить на matrix laboratory с
редактируемыми 2D \(q,k,v\), exact dot products, mask, row sum, output vector
и interventions.

## Какие rich sidenotes нужны

- портрет Владимира Фока/линейно-алгебраический российский мост уместен лишь
  с конкретной историей; лучше карточка авторов Transformer с первоисточником;
- цитата «Attention is not Explanation» с точной citation;
- JS softmax row calculator;
- counterexample high weight but zero value;
- counterexample same weights/different output;
- рисунок softmax Jacobian;
- вопрос о row sums after mask;
- мост назад к token fertility и вперёд к transformer block;
- warning о heatmap as causal proof.

## Недостающие упражнения

1. Полностью вычислить \(Q,K,V,S,A,AV\) для \(3\times2\) matrices.
2. Доказать variance scaling при независимых unit-variance components.
3. Доказать permutation equivariance self-attention без positions.
4. Сконструировать attention map с max weight .9, но нулевым вкладом
   выбранного token.
5. Browser-experiment seed 7605: 100 random matrices, compare raw/scale,
   entropy and gradient norms by \(d=4,16,64,256\).

## План переписывания

1. Начать ручным 3-token matrix example.
2. Развести address/content визуально и алгебраически.
3. Вывести scaling и softmax competition.
4. Добавить masks и permutation argument.
5. Пересобрать widget без hard-coded semantics.
6. Провести causal intervention versus heatmap.
7. Посчитать memory/computation in units.
8. Довести до 10 sidenotes, 4 inline exercises и 8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | PARTIAL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | FAIL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
