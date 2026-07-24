# Урок 67. MCMC и расшифровка текста

## Главная педагогическая идея

MCMC строит зависимую цепь, стационарное распределение которой равно сложному
target; Metropolis–Hastings корректирует удобное proposal через acceptance
ratio. Качество цепи определяется не красивым histogram, а mixing,
autocorrelation, ESS и воспроизводимой диагностикой.

## Что есть сейчас

1622 слова до задач, 10 разделов, 16 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildMcmc`) и 5 задач. Проверены все visuals и desktop screenshot. Текст
содержит detailed balance, MH ratio, burn-in, ESS, diagnostics, Bayesian
logistic example и calibration. Финальные задания сильнее среднего.

Ученик поймёт proposal/accept/reject, сможет проверить detailed balance и
объяснить зависимость samples. Но не получит доказательства stationarity из
detailed balance в суммарной форме, не увидит irreducibility/aperiodicity и не
потрогает диагностику, о которой читает.

## Чего не хватает в рассуждении

1. Строгого шага
   \(\sum_x\pi(x)P(x,y)=\sum_x\pi(y)P(y,x)=\pi(y)\).
2. Явных условий ergodicity и контрпримеров: две несвязные компоненты,
   периодическая цепь.
3. Ручной MH на 4 состояниях с transition matrix и проверкой stationary
   vector.
4. Вывода acceptance ratio для asymmetric proposal.
5. Autocorrelation plot, integrated autocorrelation time и расчёта ESS на
   короткой числовой последовательности.
6. Различия convergence-to-stationarity и Monte Carlo error после
   convergence.
7. Consistent real case: текст называет Heart Disease, задача — Breast
   Cancer; нет embedded sample, licence и snapshot.

## Рисунки и интерактив

### `figure-1.svg`

Target, trace и histogram составлены хорошо, но одна удачная цепь скрывает
mode trapping. Нужны четыре chains из dispersed starts, burn-in boundary и
mode occupancy.

### `figure-2.svg`

Многопанельная диагностика сильна визуально, но формулы ESS/R-hat не связаны с
конкретными цифрами. Добавить callout, показывающий один lag-product и
effective sample count.

### `figure-3.svg`

Траектории на posterior/parameter plane информативны, но плотная легенда и
линии затрудняют чтение. Нужны одинаковые axes, contours target, proposal
ellipse и accepted/rejected marks.

### `buildMcmc`

Виджет симулирует bimodal target, но histogram нормируется на свой максимум,
а target density масштабируется отдельно: overlay количественно ложен. Fixed
20% burn-in не обоснован. Нет ACF, ESS, \(\hat R\), multiple chains,
acceptance rate или `run/reset`; только sliders. Замена: 4 parallel chains,
step/run buttons, shared normalization density, acceptance/rejection marks,
ACF, ESS и preset «узкое proposal», «широкое», «непересекающиеся modes».

## Какие rich sidenotes нужны

- портрет Андрея Маркова и отдельный портрет Перси Диакониса с источниками;
- короткая цитата Гельмана о multiple chains;
- JS-проверка detailed balance для матрицы \(4\times4\);
- контрпример periodic chain \(0\leftrightarrow1\);
- контрпример reducible chain с правильным локальным histogram;
- рисунок proposal ellipse и target contours;
- вопрос: почему thinning обычно не создаёт информацию;
- мост назад к random walk и вперёд к cipher-MCMC;
- warning «ESS 1000» без определения функции интереса.

## Недостающие упражнения

1. Построить MH transition matrix для target \((1,2,4,3)/10\) и symmetric
   neighbor proposal.
2. Доказать stationarity из detailed balance.
3. Для sequence autocorrelations \((.8,.5,.2,0)\) вычислить integrated time и
   ESS при \(N=10000\).
4. Сконструировать chain, которая проходит trace-test внутри одной mode, но
   имеет неверный общий mean.
5. Browser-experiment seed 6705: 4 starts, 3 proposal widths, 5000 steps;
   публиковать bias mean, ESS и mode-switch count.

## План переписывания

1. Начать конечной четырёхточечной цепью.
2. Вывести MH acceptance и transition matrix.
3. Доказать stationarity и назвать ergodicity assumptions.
4. Перейти к continuous bimodal target.
5. Пересобрать лабораторию вокруг multiple-chain diagnostics.
6. Разобрать ESS/Monte Carlo error численно.
7. Выбрать один реальный medical dataset и дать manifest.
8. Добавить 10 sidenotes, 4 inline exercises и 8 финальных задач.

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
