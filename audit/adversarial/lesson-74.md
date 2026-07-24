# Урок 74. LSTM и проблема длинной памяти

## Главная педагогическая идея

LSTM создаёт аддитивный cell-state path и обучаемые gates; постоянный forget
gate задаёт экспоненциальное время памяти, но полный gradient имеет
дополнительные пути и память не бесконечна.

## Что есть сейчас

1595 слов до задач, 11 разделов, 8 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildLstm`) и 5 задач. Проверены visuals, source и screenshot. Есть полные
LSTM equations, half-life proposition, masking, GRU comparison, household
power case, gate interventions и state-reset checks.

Ученик поймёт роли \(f,i,o,\tilde c\), вычислит half-life и запомнит
padding/state contracts. Не увидит полный derivative graph, parameter counts
или честно обученную gate dynamics.

## Чего не хватает в рассуждении

1. Полного числового шага vector LSTM с sigmoid preactivations, не только
   готовыми gates.
2. Различия partial derivative direct path \(\approx f_t\) и total derivative,
   где gates зависят от \(h_{t-1}\).
3. Parameter-count formula RNN/GRU/LSTM при одинаковых \(d_x,d_h\).
4. Counterexample \(f\approx1,i>0\): \(c_t\) растёт, \(h_t\) saturates и
   gradient наружу исчезает.
5. Quantitative mask bug на двух padded sequences.
6. Честного learned-gate experiment; текст просит «разрешить модели подобрать
   ворота», widget этого не делает.
7. OPSD/UCI data snapshot and licence внутри статьи.

## Рисунки и интерактив

### `figure-1.svg`

Схема памяти слишком упрощена: actual gates/equations не привязаны к
операциям, output arrow заканчивается на border. Нужен стандартный, но
минималистичный computational graph с multiply/add nodes и dimensions.

### `figure-2.svg`

Labels `forget/input/cell/output` сталкиваются с нулевым tick/baselines;
масштабы gates/state смешаны. Перенести labels в отдельную legend, сделать
shared event markers и числовые gate values.

### `figure-3.svg`

Load/RUL-style panel чист, но gate profile может быть принят за causal
explanation. Нужны intervention curve и uncertainty по seeds рядом, плюс
units и real snapshot.

### `buildLstm`

Виджет не исполняет LSTM. Он сравнивает
`inputGate * forget^(t-2)` с линейной RNN
`inputGate * rnn^(t-2)`; output gate фиксирован 1, candidate/tanh/state
отсутствуют. Нельзя обучить gates, управлять \(o\), padding или reset.
Это переименованный exponential-decay demo. Замена: exact scalar LSTM
recurrence с four gates, impulse/false marker sequences, step/run/reset и
tiny trained delayed-copy preset, где learned profiles можно сравнить с
manual.

## Какие rich sidenotes нужны

- портрет Зеппа Хохрайтера/Юргена Шмидхубера и точная citation 1997;
- российский мост к теории устойчивости с портретом Ляпунова, без ложной
  преемственности;
- JS-calculator одного LSTM step;
- counterexample accumulating cell;
- counterexample informative missingness;
- рисунок direct vs indirect gradient paths;
- вопрос: почему \(o_t=0\) не стирает \(c_t\);
- мост назад к RNN и вперёд к residual transformer;
- warning state leakage between batch items.

## Недостающие упражнения

1. По заданным gate preactivations вычислить sigmoids, \(c_t,h_t\).
2. Доказать half-life formula и оценить sensitivity \(dt_{1/2}/df\).
3. Вывести exact parameter counts RNN/GRU/LSTM.
4. На batch длины 2/5 показать prediction dependence on padding при
   неправильной маске.
5. Browser-experiment seed 7405: false/true markers, three forget biases,
   recovery delay и gate curves.

## План переписывания

1. Начать delayed-copy с ручной gate strategy.
2. Собрать ячейку по operation graph.
3. Вычислить один полный vector step.
4. Вывести direct gradient/half-life и оговорить total paths.
5. Пересобрать laboratory как настоящую LSTM.
6. Разобрать masking/reset через executable unit tests.
7. Добавить real load snapshot и fair GRU comparison.
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
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
