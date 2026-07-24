# Урок 87. Редактирование, deepfake и AI-safety

## Главная педагогическая идея

Generative editing is governed by a mask/identity contract; deepfake
detection is decision-making under base rate, domain shift and adaptive
threats. A detector flag is evidence for review, not a claim about intent.

## Что есть сейчас

1462 слова до задач, 12 разделов, 4 display-формулы, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildDeepfake`) и 5 tasks. Checked source, figures and screenshot. Text
covers inpainting mask, identity/style, threat model, Bayes PPV, costs,
generator shift, provenance/C2PA, FaceForensics++ and selective review.

Ученик действительно поймёт base-rate fallacy and evidence pipeline. Это
один из лучших conceptual lessons, но editing/deepfake share too little
mathematics and widget uses synthetic score distributions without disclosure.

## Чего не хватает в рассуждении

1. Confusion-matrix derivation of PPV, NPV and expected cost as function of
   threshold.
2. ROC vs PR under prevalence change.
3. Calibration/likelihood-ratio view and how prior odds update.
4. Exact mask loss with inside/outside weights and dilation.
5. Counterexample high AUC/poor PPV at 0.1% prevalence.
6. Counterexample detector recognizing codec.
7. Fixed benchmark score snapshot, model revision, transformations and
   licence.

## Рисунки и интерактив

### `figure-1.svg`

Four generic shapes have too much whitespace and are not real pixels; the
difference map cannot teach mask leakage. Use a licensed still, binary/dilated
mask and measured outside-mask \(L_1\).

### `figure-2.svg`

Base-rate blocks are mathematically useful and PPV 15.4% is correct. Add
multiple prevalence rows and explicit denominator among flags.

### `figure-3.svg`

Evidence tree is a good idea but arrows/directions are not visible in render;
states/actions and failure branches need arrowheads and thresholds.

### `buildDeepfake`

Base-rate calculator is useful, but scores come from assumed Gaussian
distributions and «new generator shift» is an arbitrary mean shift, with no
label that this is synthetic. No ROC/PR threshold sweep, calibration,
transformation or real scores. Add a fixed public detector-score snapshot,
prevalence reweighting, threshold/cost, calibration and a synthetic tab
clearly separated from evidence.

## Какие rich sidenotes нужны

- provenance/C2PA primary-spec card;
- short verified quote on Bayesian evidence;
- JS prior-odds × likelihood-ratio calculator;
- counterexample codec shortcut;
- counterexample missing watermark ≠ real;
- mask dilation illustration;
- question whether detector evaluates file or author;
- bridge back to adversarial threat model and forward to video consistency;
- warning biometric data/access.

## Недостающие упражнения

1. Derive PPV/NPV for three prevalences.
2. Compute expected cost for threshold table.
3. Construct same ROC but different PR under prevalence.
4. Compute inside/outside mask loss and choose penalty.
5. Browser-experiment with fixed score file: threshold, prevalence
   0.1–30%, compression strata, calibration and selective coverage.

## План переписывания

1. Open with one edited image/mask contract.
2. Quantify outside-mask change.
3. Define threat model and confusion outcomes.
4. Derive Bayes PPV/cost and ROC–PR distinction.
5. Rebuild widget on real/synthetic-separated scores.
6. Add codec/new-generator shift and selective review.
7. Draw provenance pipeline with actions.
8. Expand to 10 sidenotes, 4 inline exercises and 8 tasks.

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
