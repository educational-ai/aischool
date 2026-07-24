# Урок 73. Рекуррентная сеть: состояние во времени

## Главная педагогическая идея

RNN повторяет одну ячейку и переносит фиксированное скрытое состояние; дальняя
память и обучение определяются произведением Jacobians. Линейный scalar case
должен быть рентгеном настоящей tanh-RNN, а не оправданием ложного ползунка.

## Что есть сейчас

1477 слов до задач, 11 разделов, 7 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildRnn`) и 5 задач. Проверены SVG, screenshot и implementation. Текст
содержит recurrence, linear unfolding, BPTT/Jacobian product, clipping,
truncated BPTT, synthetic memory tasks, NASA C-MAPSS и temporal occlusion.

Ученик поймёт weight sharing, hidden state, vanishing/exploding gradients и
честный split по engines. Однако лаборатория математически противоречит
статье, а для tanh dynamics нет ни одного полного числового прохода.

## Чего не хватает в рассуждении

1. Явного nested expansion для `АБ`/`БА`, отвечающего первой sidenote.
2. Числовой tanh-RNN на 4 шагах: \(h_t\), \(\hat y_t\), loss и gradients.
3. Разложения Jacobian norm bound с singular values и factors
   \(1-h_t^2\); одного spectral-radius лозунга недостаточно.
4. Контрпримера: \(W_h\) norm \(>1\), но saturation убивает gradient; и
   norm \(<1\), но short task учится.
5. Полного различия forward memory и trainable credit assignment.
6. Parameter-count/fair-budget comparison RNN/1D-CNN/seasonal baseline в
   основном тексте.
7. C-MAPSS snapshot, units sensors и figure provenance.

## Рисунки и интерактив

### `figure-1.svg`

Input arrows заканчиваются на границе cells, horizontal hidden lines проходят
под boxes, направление и shared weights не настолько явны, как обещает
caption. Нужны color-coded matrices \(W_x,W_h\), time arrows и impulse trace.

### `figure-2.svg`

Log-gradient graph чистый, но использует «effective multipliers» без связи с
реальной tanh state. Добавить две кривые: exact scalar derivative и bound,
показать saturation.

### `figure-3.svg`

Траектории RUL убедительны, но dataset fields/units/engine IDs и intervals
отсутствуют. Нужен реальный мини-snapshot и uncertainty across engines.

### `buildRnn`

Критический дефект: «sensitivity» вычисляется как
`Math.pow(memory, t - 2)`. Это не Jacobian tanh-RNN, игнорирует
\((1-h_k^2)\), input amplitude, saturation и weights; slider разрешает
`memory>1`. Interface не включает нелинейность, хотя текст просит её
«включить», и не выполняет delayed-copy training. Нет run/reset. Заменить на
настоящий scalar recurrence \(h_t=\tanh(uh_{t-1}+wx_t)\), automatic
derivative chain и отдельный tiny trained delayed-copy model.

## Какие rich sidenotes нужны

- портрет Александра Ляпунова и связь stability без приписывания ему RNN;
- цитата Elman 1990 с источником;
- JS-развёртка `АБ`/`БА`;
- counterexample saturation при \(u=2,x_1=10\);
- counterexample forward dependence без usable gradient;
- рисунок computation graph и reverse arrows;
- вопрос о state leakage между engines;
- мост назад к dynamical systems и вперёд к LSTM/attention;
- warning о shuffled temporal split.

## Недостающие упражнения

1. Развернуть \(h_2\) для `АБ` и `БА` и дать условия равенства.
2. Для scalar tanh-RNN вручную вычислить 4 states и
   \(\partial h_4/\partial h_0\).
3. Доказать bound произведения operator norms.
4. Сконструировать насыщенный пример с большой state dependency и малым local
   gradient.
5. Browser-experiment seed 7305: impulse amplitudes \(0.1,1,10\),
   \(u=.8,1,1.2\), exact state/gradient/half-life.

## План переписывания

1. Начать nested formulas для порядка.
2. Полностью вычислить scalar linear и tanh cases.
3. Нарисовать BPTT graph и вывести Jacobian product.
4. Развести memory, gradient и capacity.
5. Пересобрать laboratory на реальной recurrence.
6. Добавить delayed-copy training и baseline comparison.
7. Дать C-MAPSS data card.
8. Расширить до 10 sidenotes, 4 inline exercises и 8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | PARTIAL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
