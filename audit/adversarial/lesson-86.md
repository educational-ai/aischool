# Урок 86. Diffusion: учимся обращать шум

## Главная педагогическая идея

Forward diffusion has a known Gaussian closed form; a network predicts noise
or score, and a reverse sampler composes local conditional steps. Schedule,
solver and classifier-free guidance are part of the model result. A widget
that secretly uses clean \(x_0\) to reconstruct is not diffusion.

## Что есть сейчас

1627 слов до задач, 15 разделов, 9 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildDiffusion`) и 5 tasks. Checked source, visuals and screenshot. Text
contains forward process/closed form, noise objective, score link, reverse
mean, CFG, latent diffusion, SDE/ODE, CIFAR metrics and many mini-studies.

Ученик поймёт coefficients signal/noise, schedule and guidance trade-off.
Не получит derivation closed form/reverse posterior and widget teaches an
oracle interpolation instead of denoising.

## Чего не хватает в рассуждении

1. Induction deriving
   \(q(x_t|x_0)=N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)\).
2. Posterior \(q(x_{t-1}|x_t,x_0)\) and substitution of predicted noise.
3. Exact relation noise/v/x0 parameterizations.
4. Score derivation for Gaussian corruption.
5. Counterexample high guidance reduces diversity/oversaturation with actual
   samples and metrics.
6. Solver step comparison on identical initial noise.
7. Real image provenance and fixed model/scheduler configuration.

## Рисунки и интерактив

### `figure-1.svg`

Noise stages/SNR communicate the idea, but image provenance, schedule values
and selected markers are absent. Add exact \(\bar\alpha_t\), seed and source.

### `figure-2.svg`

Four spectra have unlabeled axes/units and appear repetitive; caption promises
correction magnitudes not legible. Use shared frequency axis, normalized PSD
and actual denoising trajectory.

### `figure-3.svg`

Vector field is visually strongest, but scale/normalization and density model
are missing. Add score units, probability contours and time labels.

### `buildDiffusion`

Fatal defect: reverse «recovery» uses clean ground truth
`base`/`x0` directly:
`recovered = base*(1-remaining) + corrupted*remaining`. Guidance directly
shrinks error. No denoiser, score, noise prediction or reverse transition.
This is an oracle crossfade dressed as diffusion. Replace with exact 1D/2D
mixture score (analytically computable) and Euler reverse steps, plus optional
precomputed tiny image denoiser outputs. Must have seed, step/run/reset and
same \(x_T\) across schedulers.

## Какие rich sidenotes нужны

- primary-source DDPM/Song score-model cards;
- portrait Andrey Kolmogorov for diffusion equations only with precise bridge;
- JS forward closed-form calculator;
- counterexample oracle denoising using \(x_0\);
- counterexample excessive CFG;
- drawing Gaussian conditioning triangle;
- question why reverse can be stochastic;
- bridge back to random walk and VAE, forward to deepfake/video;
- warning `50 steps` insufficient without scheduler/solver.

## Недостающие упражнения

1. Prove forward closed form by induction.
2. Derive scalar Gaussian posterior \(q(x_{t-1}|x_t,x_0)\).
3. Convert \(\varepsilon\)-prediction to \(\hat x_0\).
4. Compute CFG geometry and show \(w>1\) extrapolation.
5. Browser-experiment seed 8605: analytic two-Gaussian score, Euler steps
   10/25/100, same starts, Wasserstein/coverage metrics.

## План переписывания

1. Start with exact scalar Gaussian chain.
2. Derive closed form and SNR.
3. Derive training objective/score relation.
4. Derive one reverse posterior step.
5. Rebuild widget on an actual score field.
6. Add solver/schedule/guidance paired experiments.
7. Ground image panels in fixed checkpoint/data.
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
