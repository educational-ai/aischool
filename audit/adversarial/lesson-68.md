# Урок 68. Практикум MCMC: шифр Диакониса

## Главная педагогическая идея

Ключ простой подстановки можно искать локальными перестановками букв, если
языковая статистика превращена в log-likelihood; Metropolis позволяет иногда
идти вниз и выходить из локальных максимумов. Это редкий урок, где интерактив
действительно исполняет алгоритм, а не только рисует метафору.

## Что есть сейчас

1400 слов до задач, 10 разделов, 5 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildCipher`) и 5 задач. Проверены SVG, screenshot и implementation.
Текст вводит substitution key, n-gram score, proposal swap, acceptance,
temperature/annealing, diagnostics, Taiga corpus и историю frequency analysis.

Ученик поймёт, как строка получает score и почему случайный downhill move
полезен. Не сможет вывести acceptance ratio для temperature target,
оценить неопределённость ключа или отделить переобучение language model от
качества расшифровки.

## Чего не хватает в рассуждении

1. Полностью ручного расчёта bigram score для двух ключей на строке длины
   8–12.
2. Вывода \(\min(1,\exp((S'-S)/T))\) из target
   \(\pi(k)\propto e^{S(k)/T}\).
3. Обработки zero counts: add-\(\alpha\) smoothing, alphabet, spaces,
   punctuation.
4. Контрпримера короткого ciphertext, где несколько keys равноправны.
5. Отличия fixed-temperature MCMC от simulated annealing; текущий текст
   сближает sampling и optimization.
6. Train/test language score: key, максимизирующий corpus trigrams, может
   породить правдоподобную, но неверную фразу.
7. Reproducible corpus slice с licence/checksum, а не только название Taiga.

## Рисунки и интерактив

### `figure-1.svg`

Фигура слишком пустая: крошечные повторяющиеся символы и стрелки вниз, хотя
caption обещает table/key/bigrams. Нет реального alphabet mapping и ни одного
вычисленного n-gram. Нужна лента ciphertext, строка key, decoded text и
подсвеченные биграммы с суммой log-probabilities.

### `figure-2.svg`

Граф key swaps не имеет читаемых направлений/наконечников и не показывает
sample strings, хотя это главный смысл state. Нужен graph из 6 ключей,
accepted/rejected edges, score и temperature.

### `figure-3.svg`

Trace/acceptance/result — хорошая основа. Не хватает нескольких restarts,
best-so-far и held-out score; одна красная линия создаёт иллюзию доказанной
сходимости.

### `buildCipher`

Виджет реально обучает trigram statistics на маленьком встроенном corpus,
шифрует строку и делает MCMC. Это лучше соседей. Но corpus микроскопический,
temperature фиксирована, нет bigram/trigram switch, длины текста, annealing,
multiple restarts, step/reset. Readout показывает долю совпавших символов с
истинным plaintext — удобный synthetic oracle, недоступный в реальной задаче.
Нужно явно пометить oracle и добавить held-out language score, restart
distribution и ручной inspection key.

## Какие rich sidenotes нужны

- портрет Перси Диакониса и схема конкретного cipher experiment;
- исторический портрет аль-Кинди и точная ссылка на трактат;
- JS-калькулятор score одной строки по таблице bigrams;
- афоризм «правдоподобный язык не равен правильному сообщению» с авторской
  маркировкой, не псевдоцитата;
- контрпример `АБАБАБ`, где частоты не идентифицируют alphabet;
- контрпример corpus shift: дореформенный текст против новостей;
- вопрос о сохранении пробелов атакующим;
- мост назад к MH diagnostics и вперёд к tokenization;
- рисунок smoothing zero count.

## Недостающие упражнения

1. По заданной \(3\times3\) bigram table вычислить scores двух decodings.
2. Вывести acceptance probability из tempered target.
3. Для ciphertext длины 12 построить два разных keys с одинаковым decoded
   bigram multiset.
4. Сравнить fixed \(T=1\) и schedule \(T_t=2/\log(t+2)\) на игрушечном
   landscape; назвать sampling/optimization objective.
5. Browser-experiment seed 6805: 20 restarts, 3 corpus sizes, report exact-key
   rate, held-out trigram score и time-to-best.

## План переписывания

1. Открыть реальным коротким ciphertext и двумя competing decodings.
2. Построить n-gram model со smoothing вручную.
3. Вывести tempered target и MH acceptance.
4. Проследить пять swaps в таблице.
5. Расширить виджет step/restart/annealing diagnostics.
6. Разобрать ambiguity короткого текста и corpus shift.
7. Опубликовать corpus snapshot, licence и checksum.
8. Довести статью до 10 sidenotes, 4 inline exercises и 8 задач.

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
