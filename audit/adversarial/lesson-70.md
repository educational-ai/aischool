# Урок 70. Многорукий бандит

## Главная педагогическая идея

Bandit решает конфликт исследования и использования; regret можно разложить
по числу выборов неоптимальных arms, а UCB и Thompson sampling кодируют
неопределённость разными способами. Это хороший сюжет для настоящего
многоразового browser experiment.

## Что есть сейчас

1510 слов до задач, 9 разделов, 9 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildBandit`) и 5 задач. Проверены source, SVG и screenshot. Есть regret
identity, UCB, Thompson, IPS, off-policy bias, dynamic environment и сильные
финальные задания.

Ученик поймёт exploration/exploitation, cumulative regret и принцип optimism.
Не сможет доказать логарифмический regret UCB даже в outline, вывести
Beta–Bernoulli posterior или отличить one-run curve от expected regret.

## Чего не хватает в рассуждении

1. Полного Beta–Bernoulli update с prior \(\mathrm{Beta}(1,1)\) и credible
   intervals.
2. Вывода UCB bonus из Hoeffding и union bound хотя бы на уровне
   \(P(\mu_i>\hat\mu_i+c)\).
3. Regret decomposition
   \(R_T=\sum_i\Delta_i\mathbb E N_i(T)\) с выводом, не только формулой.
4. Distribution по seeds; один stochastic run не демонстрирует expected
   guarantee.
5. Контрпримера nonstationary rewards, где UCB/Thompson накапливают устаревшие
   данные.
6. Различия contextual и non-contextual bandit.
7. Реального click/log snapshot с logging propensity и licence.

## Рисунки и интерактив

### `figure-1.svg`

Choices/reward/regret читаются, но один run без uncertainty выглядит как
свойство метода. Нужны median и band 100 seeds, одинаковые reward streams и
vertical marks exploration.

### `figure-2.svg`

В x labels буквально отображаются `A\nN=20`, `B\nN=200`, `C\nN=8` — явный
визуальный дефект. Posterior должен показывать Beta density, mean и interval,
а не только bars.

### `figure-3.svg`

Simpson reversal показан, но отсутствуют размеры групп и raw counts, поэтому
проверить агрегат невозможно. Нужна таблица impressions/clicks по strata и
weighted aggregation.

### `buildBandit`

Epsilon-greedy и UCB симулируются, но «Thompson» реализован не выборкой Beta
posterior, а clipped Gaussian approximation. Это математически неверная
подмена. Один slider `explore` имеет разные смыслы у методов, одна trajectory
выдаётся без intervals, нет nonstationarity, context, `run/reset`. Требуется
настоящая Beta sampling, common random numbers, 100-seed mode, step/action
log и switch environment на \(t=300\).

## Какие rich sidenotes нужны

- портрет Герберта Роббинса и первоисточник 1952;
- цитата «optimism in face of uncertainty» с корректной атрибуцией;
- JS Beta-update после каждого клика;
- контрпример winner's curse после одной победы;
- counterexample nonstationary arm;
- рисунок exploration cost vs information gain;
- вопрос: почему click не равен user utility;
- мост назад к confidence bounds и вперёд к actor–critic;
- warning про IPS weights при малой propensity.

## Недостающие упражнения

1. Обновить Beta priors для трёх arms после заданных clicks.
2. Из Hoeffding получить UCB radius для \(t=1000,N_i=20,\delta=t^{-4}\).
3. Доказать regret decomposition по counts arms.
4. Построить Simpson reversal с целыми impressions и проверить его вручную.
5. Browser-experiment seed 7005: 500 seeds, horizon 1000, stationary и
   switched arms; median/IQR regret трёх методов.

## План переписывания

1. Открыть реальным A/B/click журналом.
2. Вывести regret identity на пяти rounds.
3. Получить UCB из confidence interval.
4. Вывести Beta posterior и настоящий Thompson sampling.
5. Пересобрать интерактив с repeated runs.
6. Добавить nonstationary/contextual counterexamples и IPS.
7. Дать data snapshot и logging propensities.
8. Добавить 10 sidenotes, 4 inline exercises и 8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | FAIL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
