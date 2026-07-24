# Урок 81. RLHF, PPO, DPO и GRPO

## Главная педагогическая идея

Preference optimization shifts answer probabilities toward a proxy while a
reference/KL term constrains distributional departure. PPO, DPO and GRPO are
different algorithms with different data flows; they cannot be represented by
three arbitrary «strength» multipliers.

## Что есть сейчас

1488 слов до задач, 12 разделов, 6 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildAlignment`) и 5 задач. Проверены source, SVG and screenshot. Text
contains KL-regularized optimum, PPO clipping, DPO loss, GRPO normalization,
length hacking, evaluation and Pareto KL–win-rate. Tasks are detailed.

Ученик поймёт роли reference, \(\beta\), online/offline data and proxy
exploitation. Не увидит derivation Gibbs optimum/DPO objective and получит
ложное представление из widget, где algorithms names are cosmetic.

## Чего не хватает в рассуждении

1. Lagrange-multiplier derivation
   \(\pi^*\propto\pi_{\rm ref}e^{r/\beta}\).
2. Sequence-level vs token-level KL and length dependence.
3. PPO clipping graph separately for positive/negative advantage.
4. DPO derivation assumptions and role of chosen/rejected support.
5. GRPO zero-variance group handling and bias from group sampling.
6. Counterexample both responses bad; fixed preference only ranks them.
7. Real small preference snapshot and actual policy probability shifts.

## Рисунки и интерактив

### `figure-1.svg`

Pipeline boxes are usable, but arrowheads/routing are small and distinctions
data/parameters/frozen evaluation are not legible at page scale. Need swim
lanes for dataset, model copies and evaluator.

### `figure-2.svg`

Mixed Russian/English and small routed arrows; PPO/DPO comparison remains a
flowchart, not a mathematical comparison. Add one pair with log-probs and
actual objective terms.

### `figure-3.svg`

Four mini-plots are clean but x-axis meaning/source and uncertainty missing.
Need a controlled optimization trace with checkpoints and independent human
score intervals.

### `buildAlignment`

This is severe mathematical fakery. PPO, DPO and GRPO differ only through
arbitrary strength multipliers and bonuses; no clipping, ratios, pairwise loss,
group normalization, sampling or model probabilities are implemented.
Hand-coded true/proxy functions guarantee a Goodhart curve. Replace with a
finite 4-answer policy: exact reference probabilities/rewards, closed-form
optimum; separate tabs implementing actual PPO one-batch update, DPO pair
gradient and GRPO group advantages, all exposing state and reset.

## Какие rich sidenotes нужны

- primary-source cards PPO, DPO, GRPO with exact equations/citations;
- quote on Goodhart with verified attribution;
- JS Gibbs-policy calculator;
- counterexample both answers unsafe;
- counterexample average KL hiding one prompt;
- plot positive/negative PPO clipping;
- question why long answer accumulates more token KL;
- bridge back to Bradley–Terry and forward to agent system safety;
- warning judge model position bias.

## Недостающие упражнения

1. Derive KL-regularized optimum by Lagrangian.
2. Compute PPO clipped terms for both signs of advantage.
3. Differentiate DPO loss with respect to chosen/rejected log-odds.
4. Construct GRPO group with tiny variance and analyze normalization.
5. Browser-experiment seed 8105: finite 6-answer proxy with length bias,
   exact updates and Pareto true utility–KL.

## План переписывания

1. Open with finite answer distribution and exact reference/rewards.
2. Derive Gibbs optimum.
3. Implement PPO on one sampled batch.
4. Derive/compute DPO on one pair and GRPO on one group.
5. Rebuild widget with actual algorithms.
6. Add length/support/average-KL counterexamples.
7. Ground evaluation in a fixed preference sample.
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
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
