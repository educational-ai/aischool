# Урок 85. VAE: вероятностный латентный мир

## Главная педагогическая идея

VAE turns encoding into variational inference: reconstruction likelihood and
KL form ELBO, reparameterization makes sampling differentiable, and \(\beta\)
sets a rate–distortion trade-off. A good-looking interpolation is not evidence
of a trained probabilistic latent space.

## Что есть сейчас

1655 слов до задач, 14 разделов, 11 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildVae`) и 5 tasks. Checked visuals, source and screenshot. Text contains
ELBO identity, Gaussian KL, reparameterization, beta-VAE, posterior collapse,
aggregate posterior, uncertainty and Fashion-MNIST. Tasks are mathematically
and experimentally rich.

Ученик поймёт ELBO components and sampling modes. Но статья не выводит
Gaussian KL, and widget does not contain an encoder, decoder or ELBO.

## Чего не хватает в рассуждении

1. ELBO derivation expanded from Bayes/Jensen line by line in main article.
2. One-dimensional Gaussian KL derivation, then diagonal sum.
3. Numeric Monte Carlo reparameterization gradient estimate.
4. Likelihood choice: Bernoulli/MSE/Gaussian with units and constants.
5. Mutual information/rate relation and aggregate posterior mismatch.
6. Counterexample smooth interpolation through low-prior-density region.
7. Real Fashion-MNIST grid and metrics/revisions in article.

## Рисунки и интерактив

### `figure-1.svg`

Latent plots are readable but synthetic, with no decoded outputs or density
scale. Add prior contours, aggregate posterior samples and holes.

### `figure-2.svg`

Encoder diagram has no clear arrowheads/direction; \(\varepsilon\) enters
diagonally and decoder path is incomplete. Show computation graph and gradient
paths with dimensions.

### `figure-3.svg`

Rate–distortion/modes are visually tidy but icons are not tied to actual
\(\beta\), seeds or data. Replace with measured frontier and random prior
samples, not selected glyphs.

### `buildVae`

Posterior clouds and reconstruction/KL are arbitrary functions of sliders.
Interpolation simply alpha-blends drawn glyphs `3` and `8`; no decoder,
encoder, stochastic \(z\), ELBO or training exists. This is materially false.
Replace with a precomputed tiny 2D VAE: load fixed encoder means/variances and
decoder weights (or a small browser model), calculate exact KL, sample with
seed, decode grid and expose prior/aggregate mismatch.

## Какие rich sidenotes нужны

- primary-source Kingma–Welling card and exact quote;
- Russian bridge to variational methods through Dobrushin/Tikhonov only with
  precise relevance;
- JS Gaussian KL calculator;
- counterexample posterior collapse;
- counterexample linear interpolation through atypical radius;
- drawing Jensen/ELBO gap;
- question \(G(Ez)\) vs \(E G(z)\);
- bridge back to autoencoder and forward to latent diffusion;
- warning cherry-picked samples.

## Недостающие упражнения

1. Derive diagonal Gaussian KL.
2. Prove ELBO identity and equality condition.
3. Compute two reparameterized samples and pathwise gradients.
4. Construct decoder where \(G(Ez)\ne EG(z)\).
5. Browser-experiment seed 8505: fixed 2D VAE, \(\beta=.1,1,4\), rate,
   distortion, active units, prior-vs-aggregate classifier.

## План переписывания

1. Begin deterministic autoencoder holes with actual decoded points.
2. Derive ELBO line by line.
3. Derive Gaussian KL and reparameterization numerically.
4. Explain likelihood choice.
5. Rebuild widget on a real fixed VAE.
6. Diagnose rate–distortion and collapse.
7. Add aggregate posterior/prior and real Fashion-MNIST evidence.
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
