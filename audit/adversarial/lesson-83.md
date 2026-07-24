# Урок 83. StyleGAN и состязательные атаки

## Главная педагогическая идея

StyleGAN injects latent controls at multiple spatial scales; adversarial
attacks reveal a different local sensitivity of a classifier. Both concern
geometry of learned functions, but combining them in one 1493-word article
creates two half-lessons rather than one coherent Quant-style investigation.

## Что есть сейчас

1493 слова до задач, 11 разделов, 8 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildStyleAttack`) и 5 tasks. Checked visuals, source and screenshot.
Text covers mapping \(z\to w\), modulation/demodulation, style mixing,
inversion, FGSM/PGD, adversarial training, FFHQ and extensive mini-studies.

Ученик поймёт coarse/fine style and norm-bounded attack formulas, но
переход между темами риторический, not derived; neither StyleGAN nor attack
actually runs in the widget.

## Чего не хватает в рассуждении

1. Correct indexing/dimensions in modulation/demodulation and one numeric
   convolution example.
2. Difference \(W,W^+,S\), noise injection and affine transforms.
3. A measurable style-mixing case with real generated images, not icons.
4. Linear classifier derivation showing FGSM solves first-order
   \(\ell_\infty\) inner maximization.
5. PGD projection geometry for \(\ell_2\) vs \(\ell_\infty\).
6. Counterexample gradient masking and adaptive attack.
7. A decision: split into two lessons or explicitly frame one common
   Jacobian experiment.

## Рисунки и интерактив

### `figure-1.svg`

Progressive layer boxes overlap, branch lines disappear under boxes, labels
`4²/16²/...` are too small. No modulation math appears. Use a vertical scale
ladder with one channel-style vector and visible affine routes.

### `figure-2.svg`

Style grid is readable but consists of generic synthetic faces/tiles, not
actual StyleGAN outputs. It cannot support claims about identity/texture.
Replace with fixed checkpoint/seeds and measurable feature rows.

### `figure-3.svg`

Pattern label `10×δ` lacks norm/scale; decision graph is generic and no actual
classifier/image provenance. Need real CIFAR sample, unamplified difference
range, amplified panel and logits per PGD step.

### `buildStyleAttack`

The drawn face and stripes are decorative. «Classifier» is an arbitrary
formula that directly subtracts attack; no gradient, norm ball, generator,
latent vector or image model exists. Coarse/fine split is invented arithmetic.
Replace with either real precomputed StyleGAN frames + feature measurements
and a separate exact 2D classifier attack, or two focused interactives.

## Какие rich sidenotes нужны

- StyleGAN2 paper/checkpoint card and licence;
- quote from adversarial examples primary paper;
- JS numeric demodulation;
- counterexample latent direction entangling pose/identity;
- counterexample gradient masking;
- norm-ball diagram;
- question whether pixel-small means perceptually small;
- bridge back to GAN and forward to deepfake threat models;
- warning about face data/consent.

## Недостающие упражнения

1. Numeric modulation/demodulation for a \(2\times2\) kernel.
2. Derive FGSM from first-order loss under \(\ell_\infty\) constraint.
3. Project a point onto \(\ell_\infty\) and \(\ell_2\) balls.
4. Construct a 2D gradient-masked classifier defeated by finite difference.
5. Browser-experiment: fixed CIFAR checkpoint, seed 8305, PGD restarts,
   success/norm/logit trace plus random-noise control.

## План переписывания

1. Split StyleGAN and adversarial attack into two articles, or double length
   and make Jacobian the explicit spine.
2. Ground StyleGAN in real fixed outputs.
3. Derive modulation numerically.
4. Measure style mixing and collateral effects.
5. Derive FGSM/PGD geometrically.
6. Rebuild interactives with real computations.
7. Add robustness evaluation and ethical data card.
8. Reach 10–12 sidenotes and 8 tasks per resulting article.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | FAIL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | PARTIAL |
| H. Связность | FAIL |

**Общий вердикт: FAIL.**
