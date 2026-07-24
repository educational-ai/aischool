# Урок 82. GAN: генератор против дискриминатора

## Главная педагогическая идея

GAN is a two-player game: optimal discriminator induces a divergence, but
alternating gradient dynamics need not follow a descending scalar potential.
Fidelity, coverage and memorization must be measured separately.

## Что есть сейчас

1531 слово до задач, 12 разделов, 8 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildGan`) и 5 задач. Проверены source, SVG, screenshot. Text includes
minimax, \(D^*\), JS, saturating/non-saturating losses, collapse, bilinear
rotation, WGAN-GP, Fashion-MNIST and temporal diagnostics. Tasks are strong.

Ученик поймёт game structure and mode collapse. Не увидит derivation
\(D^*\)/JS in full and widget falsely presents a predetermined interpolation
as training.

## Чего не хватает в рассуждении

1. Pointwise maximization derivation of
   \(D^*=p_d/(p_d+p_g)\).
2. Substitution yielding \(-\log4+2JS\).
3. Generator gradient comparison saturating vs non-saturating on actual
   logits.
4. Matrix eigenvalue derivation for bilinear game in article.
5. Distinction discriminator and WGAN critic with Lipschitz enforcement
   diagnostics.
6. Counterexample high precision/low recall and memorization/high coverage.
7. Actual Fashion-MNIST random grid and metrics with seeds.

## Рисунки и интерактив

### `figure-1.svg`

Gradient loop is crowded; backward arrow and label collide, stop-gradient
semantics are not visually clear, objective signs absent. Split into D-step
and G-step panels with frozen blocks.

### `figure-2.svg`

Eight-mode scatter is visually strong, but metrics are synthetic and no seed/
sample size. Add common real samples, assignment rule and time heatmap.

### `figure-3.svg`

Icons/bar/CDF make a good audit frame but look fabricated. Use actual
Fashion-MNIST samples and classifier/retrieval provenance.

### `buildGan`

No minimax updates occur. Collapse is hand-determined from slider imbalance
and diversity; generated density interpolates toward a preset form. The
displayed critic is always analytic \(p/(p+q)\), and «critic strength» barely
affects it. Steps mean interpolation frames, not optimizer steps. No
run/step/reset. Replace with two parameterized 1D/2D distributions, actual
simultaneous/alternating gradient, critic parameters, losses/vector field and
seeded samples.

## Какие rich sidenotes нужны

- portraits Ian Goodfellow and Kantorovich (Wasserstein bridge) with sources;
- primary-source quote on adversarial nets;
- JS pointwise \(D^*\) calculator;
- counterexample two-mode collapse;
- counterexample memorized training set;
- vector-field drawing bilinear game;
- question why discriminator accuracy 50% can mean equilibrium or failure;
- bridge back to game/self-play and forward to VAE/diffusion;
- warning hand-picked samples.

## Недостающие упражнения

1. Derive \(D^*\) pointwise.
2. Complete JS-divergence substitution.
3. Compare generator gradients at \(D(G(z))=.001\).
4. Prove bilinear simultaneous gradient radius grows by
   \(\sqrt{1+\eta^2}\).
5. Browser-experiment seed 8205: true eight-mode mixture, 15 runs, mode
   coverage/precision/recall over checkpoints.

## План переписывания

1. Start with two discrete distributions and pointwise discriminator.
2. Derive \(D^*\) and JS.
3. Compare generator losses numerically.
4. Explain game rotation by vector field.
5. Rebuild widget as actual optimizer.
6. Add collapse/memorization diagnostics.
7. Ground Fashion-MNIST visuals in real snapshots.
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
