# Урок 75. Последовательности, токены и предсказание

## Главная педагогическая идея

Токенизатор — часть модели и вычислительного бюджета: его corpus-derived
merge rules задают длину последовательности, embedding table, доступный
контекст и неравную цену языков. Сравнение моделей требует общей единицы
вроде bits per byte.

## Что есть сейчас

1391 слово до задач, 11 разделов, 6 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildTokens`) и 5 задач. Проверены visuals, code и screenshot. Есть BPE,
Unicode/NFC/ZWJ, next-token loss, BPB, sampling, multilingual fertility,
tokenizer audit и числовые токены. Финальные задачи очень подробны и
воспроизводимы.

Ученик поймёт компромисс vocab–length, механизм BPE и проблемы Unicode.
Но основной текст гораздо слабее задач: нет полного merge trace,
pre-tokenization conventions или реального multilingual snapshot.

## Чего не хватает в рассуждении

1. Полной таблицы pair counts на 4–6 BPE merges с tie-breaks.
2. Различия byte-level BPE, WordPiece и unigram LM; сейчас «подслова» почти
   синоним BPE.
3. Чёткого контракта normalization/pre-tokenizer/special tokens.
4. Вывода, почему sequence likelihood корректно сравнивать через byte
   denominator, включая EOS/normalization.
5. Контрпример combining marks/ZWJ, где naive slicing ломает round trip.
6. Exact context-compute calculation: fertility × \(n^2\).
7. Малого OPUS parallel sample и tokenizer versions прямо в статье.

## Рисунки и интерактив

### `figure-1.svg`

Byte glyphs и token IDs слишком мелки, исходная фраза не считывается с
первого взгляда; «доля словаря» для одного примера непонятна. Нужен hoverable
strip с byte offsets, unicode scalars и token boundaries.

### `figure-2.svg`

Code-point boxes крошечны, mixed English labels, arrows едва видны. Нужен
пошаговый NFC transformation с hexadecimal code points и обязательным
round-trip test.

### `figure-3.svg`

Error bars аккуратны, но dataset, tokenizer, seed и language sample не
указаны; выглядит как статистика без происхождения. Нужны 1000 OPUS pairs,
exact revision, median/10–90% and sample strings-outliers.

### `buildTokens`

Виджет выполняет маленький BPE честнее многих соседей, но corpus фиксирован
микроскопически, доступны только три preset строки и slider merge count.
Нельзя редактировать corpus/input, посмотреть pair frequencies, byte offsets,
token IDs, BPB или сравнить tokenizer. Нет step/run/reset как отдельных
операций. Пересобрать как BPE workbench: editable five-line corpus, merge
table, one-step button, deterministic tie-break, encode/decode tests и side-by-side
two corpora.

## Какие rich sidenotes нужны

- портрет Андрея Маркова у языковых chains только как исторический мост;
- карточка Филипа Гейджа и BPE 1994 с источником;
- JS Unicode microscope;
- counterexample NFC-sensitive domain;
- counterexample number boundary `999`→`1000`;
- рисунок bytes/code points/graphemes/tokens;
- вопрос: считается ли `<EOS>` байтом при BPB;
- мост вперёд к attention compute;
- warning о perplexity across tokenizers.

## Недостающие упражнения

1. Выполнить 4 BPE merges с exact tie-break и pair table.
2. Для двух tokenizations одного byte string сравнить NLL/token и BPB.
3. Разобрать emoji family на code points/UTF-8 bytes/grapheme cluster.
4. Сконструировать corpus, где первый greedy merge мешает глобально лучшему
   двумерджевому сжатию.
5. Browser-experiment: OPUS 200 pairs, fixed tokenizer revision, fertility,
   context retained at 512 tokens и intervals.

## План переписывания

1. Открыть Unicode microscope одной строкой.
2. Провести BPE trace без пропусков.
3. Развести normalization, pre-tokenization и merge model.
4. Вывести next-token loss/BPB на одном файле.
5. Пересобрать интерактив как editable workbench.
6. Показать multilingual/context cost на реальных pairs.
7. Добавить number/code counterexamples.
8. Довести до 10 sidenotes, 4 inline exercises и 8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | PARTIAL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
