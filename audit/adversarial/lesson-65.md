# Урок 65. Случайные блуждания и PageRank

## Главная педагогическая идея

PageRank — стационарное распределение случайного перехода по ориентированному
графу с телепортацией; он измеряет поток, определённый графом и prior-вектором,
а не истинность или качество страницы. Это один из более содержательных
уроков диапазона.

## Что есть сейчас

1542 слова до задач, 9 разделов, 8 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildPageRank`) и 5 финальных задач. Проверены SVG, desktop screenshot и
код виджета. Есть ручной stationary example \((0.4,0.2,0.4)\), teleportation,
Perron-style утверждение, sparse power iteration, personalized vector,
Web-Google и sensitivity. Финальные задачи заметно полнее первых уроков.

Ученик действительно поймёт переходную матрицу, stationary equation и
назначение teleportation. Он не увидит доказательства существования/единственности,
критерия сходимости, residual или ошибки power iteration.

## Чего не хватает в рассуждении

1. Явного соглашения row/column stochastic с одной полностью проверенной
   матрицей; сейчас направление умножения легко перепутать.
2. Обработки dangling nodes как изменения матрицы, а не словесной ремарки.
3. Доказательства contraction в \(L_1\):
   \(\|Gx-Gy\|_1\le\alpha\|x-y\|_1\), которое сразу даёт uniqueness и rate.
4. Residual-based stopping и границы ошибки после \(k\) iterations.
5. Контрпримера при \(\alpha=1\): несколько closed classes и зависимость от
   start.
6. Реального subgraph snapshot с лицензией и кодом получения figure-3.
7. Исправления backlinks: `/lesson/63` назван цепями Маркова, но урок 63 —
   oracle selection; `/lesson/62` назван eigenvectors, но урок 62 —
   optimizers. Это разрушает wikipedia-style навигацию.

## Рисунки и интерактив

### `figure-1.svg`

Композиция graph–matrix–bars удачна, но направления рёбер в graph на
фактическом render почти не читаются: явных наконечников нет. Матрица и bars
тесны, dangling convention не показана. Нужны номерованные рёбра, цвет row
matrix при hover и проверка суммы каждого столбца/строки.

### `figure-2.svg`

Teleport edges слишком тонкие и образуют спутанный узор; непонятно, это
реальные ссылки или виртуальные переходы. Перерисовать как смесь двух
операторов: с вероятностью \(\alpha\) — один graph edge, иначе — выбор из
\(v\), с двумя игральными жетонами.

### `figure-3.svg`

Scatter степени и PageRank выглядит как авторская synthetic cloud: не указаны
snapshot, число вершин, preprocessing, seed и uncertainty. Подписи outliers
не связаны с локальной структурой. Нужен опубликованный Web-Google subset,
checksum, \(n,m\), axes units и inset ego-graphs двух исключений.

### `buildPageRank`

Это наиболее честный виджет диапазона: PageRank вычисляется итерационно, а
seeded click simulation сравнивается с точным rank. Но \(\alpha=1\)
допускается без предупреждения о неединственности; всегда выполняется 250
iterations без residual. Нельзя менять personalized \(v\), старт, ребро или
dangling policy; нет `step/run/reset`. Требуется matrix inspector, один
power-step, residual plot, editable graph и preset с двумя closed classes.

## Какие rich sidenotes нужны

- портрет Андрея Маркова и короткий фрагмент его работы 1906 года;
- цитата Page–Brin с точным источником;
- JS-калькулятор одного matrix-vector product;
- контрпример: link farm и высокий rank без качества;
- контрпример при \(\alpha=1\) с двумя stationary distributions;
- рисунок teleportation как mixture coin;
- вопрос, почему personalized ranks линейны по \(v\);
- мост назад к eigenvectors с корректной ссылкой и вперёд к random walk/MCMC;
- warning об интерпретации link как endorsement.

## Недостающие упражнения

1. Для трёхвершинного graph построить stochastic matrix, исправить dangling
   node и проверить суммы.
2. Доказать \(L_1\)-contraction Google operator при \(0<\alpha<1\).
3. При \(\alpha=1\) сконструировать graph с двумя stationary distributions и
   показать зависимость power iteration от start.
4. Вывести residual \(\|G\pi_k-\pi_k\|_1\) и оценить distance до fixed point
   через contraction.
5. Browser-experiment seed 6505: добавить одно ребро в 50-node graph,
   построить rank-change и объяснить top-3 structural outliers.

## План переписывания

1. Начать точным блужданием на трёх страницах и матрицей.
2. Разобрать dangling node и две stochastic conventions.
3. Ввести teleportation как mixture и доказать contraction.
4. Показать power iteration с residual и stopping.
5. Встроить editable graph laboratory.
6. Перейти к personalized rank и link-farm counterexample.
7. Добавить воспроизводимый real graph и исправить backlinks.
8. Расширить до 10 sidenotes, 4 inline exercises и 8 финальных задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | PARTIAL |
| D. Интерактив | PARTIAL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | PARTIAL |
| H. Связность | FAIL |

**Общий вердикт: FAIL.**
