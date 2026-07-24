# Урок 79. Scaling laws и Chinchilla

## Главная педагогическая идея

Power-law fits связывают loss с parameters/data/compute внутри измеренного
режима; при fixed training compute optimum находится балансом model- и
data-limited terms. Fit должен предсказывать скрытые точки и не подменять
decision about deployment cost.

## Что есть сейчас

1494 слова до задач, 12 разделов, 6 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildScaling`) и 5 задач. Проверены figures, source, screenshot. Текст
вводит power law, \(C\approx6ND\), joint surface, Chinchilla, fit residuals,
emergence metrics, effective data and inference cost. Финальные задания
серьёзные.

Ученик поймёт log–log slope и compute trade-off. Но аналитический optimum
только обещан словами, исходные empirical coefficients не привязаны к
источнику, widget выдаёт авторскую формулу за law.

## Чего не хватает в рассуждении

1. Дифференцирования
   \(AN^{-\alpha}+B(6N/C)^\beta\) и exact scaling
   \(N_*(C),D_*(C)\).
2. Units and factor 6 derivation: forward/backward parameter-token FLOPs.
3. Uncertainty/covariance of \(L_\infty,\alpha,A\); \(L_\infty\) strongly
   confounds slope.
4. Counterexample regime break/architecture change.
5. Held-out prediction with numerical table from a published source.
6. Distinction fitted loss and downstream threshold with actual paired data.
7. Energy/latency measurements for named device.

## Рисунки и интерактив

### `figure-1.svg`

Linear/log views are pedagogically useful, but points and fitted law have no
source or uncertainty. Need a published data table, residual inset and holdout
point revealed after fit.

### `figure-2.svg`

Contour plot lacks colorbar/uncertainty/source; white contours are low
contrast and `compute-optimal` label hugs top. Add numeric \(N,D,C\), fitted
interval ribbon and measured-domain boundary.

### `figure-3.svg`

Probability/accuracy threshold illustrates emergence, but threshold/source
are unnamed. Need same examples/models and a robustness sweep over scoring
rule.

### `buildScaling`

All coefficients and «quality» scalar are invented; compute uses \(N D\)
without factor/units, no data fit, residuals, uncertainty or hidden point.
Sliders merely evaluate a hand-authored surface. Replace with table of 12
actual small-model runs, fit button, parameter intervals, holdout reveal and
deployment-cost layer with named units.

## Какие rich sidenotes нужны

- exact Kaplan/Hoffmann/Chinchilla source card and figure licence;
- quote about extrapolation from a primary source;
- JS log–log slope calculator;
- counterexample two regimes with excellent local \(R^2\);
- counterexample wrong \(L_\infty\);
- illustration fixed-compute hyperbola with units;
- question: training-optimal vs deployment-optimal;
- bridge back to corpus quality and forward to preference optimization;
- warning fitted FLOPs ≠ wall-clock/energy.

## Недостающие упражнения

1. Derive \(N_*(C),D_*(C)\) for general \(\alpha,\beta\).
2. Compute training FLOPs and token/parameter ratios for three configs.
3. Fit same points with/without \(L_\infty\), compare hidden prediction.
4. Construct regime-break dataset where one power law misleads.
5. Browser-experiment: 12 fixed runs, bootstrap fit seed 7905, holdout
   coverage and optimum distribution.

## План переписывания

1. Start with a published table, not a smooth curve.
2. Derive log slope and factor \(6ND\).
3. Derive compute-optimal scaling.
4. Fit with uncertainty and holdout.
5. Rebuild widget as fit laboratory.
6. Separate loss, capability and deployment decision.
7. Add regime breaks/data-quality counterexamples.
8. Reach 10 sidenotes, 4 inline exercises and 8 tasks.

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
