# Урок 77. Трансформер: блок за блоком

## Главная педагогическая идея

Transformer block — не одно attention: MHA смешивает позиции, MLP смешивает
каналы, residual даёт identity path, LayerNorm управляет масштабом. Нужно
проследить один тензор через реальные операции и размеры.

## Что есть сейчас

1440 слов до задач, 12 разделов, 7 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildTransformerBlock`) и 5 задач. Проверены visuals, source и screenshot.
Текст хорошо разделяет MHA/MLP, pre/post norm, decoder/encoder, probing,
patching, TinyStories и compute. Задачи дают точные parameter counts.

Ученик поймёт архитектурную грамматику и identity path. Не увидит реального
block forward: ни matrix shapes, ни LayerNorm numbers, ни residual vector
before/after.

## Чего не хватает в рассуждении

1. Полного tiny-block \(n=3,d=2\): LN, one-head attention, residual, ReLU-MLP,
   residual.
2. Exact parameter/FLOP formulas in text, not only tasks.
3. Distinction pre-norm gradient identity path from post-norm.
4. Counterexample: residual does not preserve information if later projection
   discards it; decodability is basis-dependent.
5. LayerNorm edge case near-zero variance and role of \(\varepsilon\).
6. KV-cache tensor shapes and one cached decode step.
7. TinyStories snapshot/revision and actual ablation data.

## Рисунки и интерактив

### `figure-1.svg`

Residual bypass lines visually do not connect cleanly to \(X\) and add nodes;
some tiny labels cannot be read at page width. Перерисовать as two repeated
units with explicit fork/join circles, dimensions and pre-norm ordering.

### `figure-2.svg`

Log axes and curves are readable, but synthetic experiment has no
configuration, seeds or common scale explanation. Add architecture table and
median/IQR.

### `figure-3.svg`

Ablation panel is serviceable, but rotated labels and absent source make it
look invented. Show actual minimal-pair examples and one confidence interval.

### `buildTransformerBlock`

Виджет не выполняет transformer. Он вычисляет invented scalar
`normAt = exp((heads/MLP/residual formula) * layer)`; heads, MLP and residual
enter arbitrary coefficients. Нет tokens, \(Q/K/V\), LN, MLP tensors or
training. Это классический renamed slider template. Замена: tiny exact block
with editable \(X\), on/off sublayers, per-operation tensor inspector,
step/reset and probe on order-sensitive synthetic task.

## Какие rich sidenotes нужны

- точная цитата Vaswani et al. и схема original post-norm vs modern pre-norm;
- российский мост к residual numerical methods через академика Тихонова
  только с аккуратной предметной связью;
- JS LayerNorm calculator;
- counterexample zero-variance token;
- counterexample no positions;
- рисунок path expansion through two residual blocks;
- вопрос: где именно positions mix;
- мост назад к LSTM additive path и вперёд к pretraining;
- warning probe ≠ use.

## Недостающие упражнения

1. Вычислить LN vector and invariance to shift/positive scale.
2. Провести tiny \(3\times2\) block forward.
3. Раскрыть two residual blocks as sum of four paths.
4. Exact parameter/FLOP counts MHA/MLP at \(n=512,d=256\).
5. Browser-experiment seed 7705: train tiny order task with no-position,
   no-MHA, no-residual, five seeds and matched parameters.

## План переписывания

1. Начать тензором \(3\times2\) и одним block forward.
2. Развести position/channel mixing.
3. Сравнить pre/post norm computation graphs.
4. Вывести residual path expansion.
5. Пересобрать widget как tensor microscope.
6. Добавить KV-cache and compute crossover.
7. Подкрепить ablation actual dataset/revision.
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
