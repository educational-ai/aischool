# Урок 66. Случайное блуждание и диффузионный масштаб

## Главная педагогическая идея

Сумма независимых центрированных шагов имеет типичный масштаб \(\sqrt n\), а
после пространственно-временного масштабирования ведёт к броуновскому
движению. Важно различить среднее смещение, RMS-расстояние, максимум и время
достижения границы.

## Что есть сейчас

1632 слова до задач, 9 разделов, 13 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildRandomWalk`) и 5 задач. Проверены все SVG, widget source и desktop
screenshot. Есть точное биномиальное распределение, drift, gambler's ruin,
Brownian limit, NOAA drifter case и необычно подробные финальные задачи.

Ученик поймёт, почему \(\mathbb E S_n=0\) не означает неподвижность и почему
\(\sqrt{\mathbb E S_n^2}=\sqrt n\). Он не получит вывода среднего времени
разорения, принципа отражения или аккуратного перехода от Donsker scaling к
конкретным units.

## Чего не хватает в рассуждении

1. Полного вычисления \(\mathbb E S_n^2\), где cross-terms исчезают именно
   из-за независимости и нулевого среднего.
2. Отличия \(\mathbb E|S_n|\) от RMS и асимптотики
   \(\mathbb E|S_n|\sim\sqrt{2n/\pi}\).
3. Вывода hitting probability gambler's ruin через harmonic recurrence и
   граничные условия.
4. Вывода mean duration \(E_i=i(N-i)\) для честного walk; сейчас результат
   сообщается без доказательства.
5. Контрпримера correlated/persistent walk, где масштаб становится почти
   \(n\), и mean-reverting walk, где рост останавливается.
6. Реального drifter snapshot: buoy ID, даты, coordinates, sampling interval,
   licence и code для fitted diffusivity.

## Рисунки и интерактив

### `figure-1.svg`

Три панели траекторий читаемы, но масштабы меняются, направление времени
обозначено слабо, а theoretical envelope не отделён от empirical quantiles.
Нужны одинаковые normalized axes \(S_k/\sqrt n\), start/end markers и
подпись, сколько paths формирует полосу.

### `figure-2.svg`

Распределение и first-passage panel полезны, но histogram не показывает
цензурированные траектории и boundary convention. Следует нарисовать survival
curve \(P(\tau>t)\), отдельно долю не достигших границы.

### `figure-3.svg`

Реальный drifter — хороший выбор, но в render нет цветовой шкалы времени,
units скорости и явно видимой серой полосы диапазона fit, обещанной подписью.
Нужны карта с scale bar, timestamp colorbar и log–log MSD с interval и
диапазоном, использованным для оценки \(D\).

### `buildRandomWalk`

Виджет рисует seeded paths и finish histogram, но seed скрыт и не управляется,
нет `run/reset`, максимум steps равен 1500, тогда как текст предлагает 1600.
Output «RMS без дрейфа» фактически выводит теоретическую центрированную
величину, а histogram не имеет x-scale. Нет maxima, hitting time, boundary или
correlation. Замена: пошаговая симуляция с seed, распределениями
\(S_n/\sqrt n\), максимумом, first-passage и переключателем independent /
persistent steps.

## Какие rich sidenotes нужны

- портрет Андрея Колмогорова и мост к invariance principle;
- цитата Эйнштейна 1905 года о броуновском движении с источником;
- JS-таблица 16 \(\pm1\)-шагов, где cross-terms можно включать;
- контрпример persistent walk;
- counterexample drift: mean растёт как \(n\), std как \(\sqrt n\);
- рисунок reflection principle;
- вопрос: почему одна траектория не проверяет distribution;
- мост назад к PageRank и вперёд к diffusion models;
- warning о longitude degrees как неравных километрах.

## Недостающие упражнения

1. Раскрыть \(S_n^2\) и доказать \(\operatorname{Var}S_n=n\).
2. Решить recurrence для вероятности достичь \(N\) раньше 0.
3. Решить recurrence среднего absorption time \(E_i=i(N-i)\).
4. Сконструировать Markov steps с
   \(P(X_{t+1}=X_t)=0.9\) и сравнить empirical scaling exponent.
5. NOAA-browser опыт: фиксированный buoy snapshot, seed 6605 для bootstrap,
   оценка MSD exponent и интервала на двух временных диапазонах.

## План переписывания

1. Начать одной ручной траекторией и различить mean/RMS.
2. Вывести variance суммы без пропусков.
3. Показать CLT scaling и empirical collapse distributions.
4. Доказать две gambler's-ruin recurrence.
5. Встроить лабораторию first passage и correlated counterexample.
6. Разобрать реальный drifter с units/provenance.
7. Добавить 10 rich notes и 4 inline exercises.
8. Закончить 8 задачами, включая два proofs и реальный data audit.

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
