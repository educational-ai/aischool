# Урок 80. Модель Брэдли—Терри и reward modeling

## Главная педагогическая идея

Pairwise preferences identify score differences through a probabilistic graph,
not absolute quality. Connectivity, repeated votes, annotator groups and
cycles determine what a scalar reward can legitimately represent.

## Что есть сейчас

1451 слово до задач, 12 разделов, 6 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildPreferences`) и 5 задач. Проверены visuals, source and screenshot.
Есть Bradley–Terry likelihood, graph connectivity, cycles, annotation
protocol, calibration, Goodhart, HH-RLHF and active pairing.

Ученик поймёт log-odds and score-shift invariance, увидит structural
disagreement. Не получит likelihood gradient/Hessian, uncertainty intervals
or identifiability proof by connected components.

## Чего не хватает в рассуждении

1. Derivation MLE for one pair and multi-node gradient.
2. Identifiability proof: one additive constant per connected component.
3. Standard errors from Fisher/Hessian or bootstrap.
4. Explicit Davidson/tie likelihood.
5. Counterexample mixture of two opposite annotator populations, where
   aggregate 50/50 is not uncertainty of one taste.
6. Active information gain rather than verbal «choose near 0.5».
7. Embedded HH-RLHF sample with revision/licence/provenance.

## Рисунки и интерактив

### `figure-1.svg`

Logistic curve works, but should show inverse logit and uncertainty from finite
votes; otherwise score difference looks observed.

### `figure-2.svg`

Bottom interval line has no scale/labels; single red bridge lacks vote count
and interval computation. Need before/after Fisher information and exact
counts on edges.

### `figure-3.svg`

Reward-vs-human trajectories appear hand-fashioned and lack data/source.
Replace with controlled length exploit experiment and intervals.

### `buildPreferences`

The four-edge graph and Bradley–Terry fit are partly real, but counts are fixed
at 40 and no uncertainty/raw votes are visible. «Disagreement groups» is only
a special toggle that changes one `D>A` cycle; it does not model group
labels. No adding arbitrary comparisons, ties, run/reset or active proposal.
Rebuild as editable comparison graph with votes by group, MLE optimizer,
bootstrap intervals, residual-cycle map and suggested next edge.

## Какие rich sidenotes нужны

- portrait R. A. Bradley/M. Terry and original citation;
- Russian historical bridge to paired-comparison psychometrics only if a
  primary source exists;
- JS inverse-logit/MLE calculator;
- counterexample disconnected tournaments;
- counterexample opposite stable groups;
- drawing likelihood ridge under score shift;
- question whether «both bad» pair identifies quality;
- bridge back to Elo/self-play and forward to DPO;
- warning position/length bias.

## Недостающие упражнения

1. Derive one-pair MLE \(\Delta=\log(w/l)\).
2. Prove additive non-identifiability per component.
3. Compute gradient/Hessian for three-node vote graph.
4. Fit scalar model to \(A>B>B>C>C>A\) cyclic rates and inspect residuals.
5. Browser-experiment seed 8005: two annotator groups, active vs random edges,
   rank correlation and interval widths.

## План переписывания

1. Open with ten votes on one pair and derive log-odds.
2. Build a connected graph and likelihood.
3. Prove identifiability and show intervals.
4. Introduce cycles/groups/ties as model failures.
5. Rebuild widget as comparison-graph workbench.
6. Quantify annotation bias and reward exploitation.
7. Add real HH-RLHF sample and data card.
8. Expand to 10 sidenotes, 4 inline exercises and 8 tasks.

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
