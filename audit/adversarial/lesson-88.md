# Урок 88. От изображения к видео, 3D и симуляции

## Главная педагогическая идея

Video adds temporal consistency, 3D adds multi-view geometry, world models add
action-dependent counterfactual dynamics. One-step photorealism is not enough;
rollouts require measured invariants, uncertainty and interventions.

## Что есть сейчас

1523 слова до задач, 14 разделов, 12 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildVideoWorld`) and 5 tasks. Checked source, visuals, screenshot. Text
spans optical flow, recurrent dynamics, camera projection, NeRF rendering,
world models, intervention, Waymo/Argoverse, exploitation and conservation.

Ученик увидит связь time–geometry–action and strong evaluation ideas. Но
охват слишком широк для объёма: pinhole/epipolar/NeRF/world model each gets
half-page, widget implements none.

## Чего не хватает в рассуждении

1. One exact optical-flow/warping calculation with occlusion mask.
2. Homogeneous projection with camera intrinsics and units in article.
3. Correct epipolar constraint \(u_2^\top F u_1=0\) and numeric residual.
4. Discretized NeRF volume rendering weights before continuous integral.
5. Error compounding derivation and teacher-forcing distribution shift.
6. Counterexample action-confounded video where model ignores action.
7. One fixed Argoverse/Waymo mini-scene with licence and data card.

## Рисунки и интерактив

### `figure-1.svg`

Ball trajectory is readable, but arrows/time/uncertainty could be stronger;
one physical law dominates a broad video lesson. Add frame strip tied to
position/residual and occlusion case.

### `figure-2.svg`

Epipolar drawing is pedagogically incomplete: image planes are loose
horizontal segments, projected points/rays are not properly connected, and
the red epipolar line looks detached. Redraw with two camera centers, baseline,
epipolar plane, \(u_1,u_2\), ray and wrong point residual.

### `figure-3.svg`

Three intervention paths/cones are useful but direction arrowheads and
training-support boundary are weak. Add action values, horizon, units and
ensemble calibration.

### `buildVideoWorld`

No video/world model is simulated. `frames` mostly changes seed and a displayed
context-gain label; «physics» directly interpolates prediction toward known
ground truth, which is an oracle. Actions do not drive dynamics, camera/3D are
absent, no rollout/reset. Replace with an actual 2D bouncing-ball simulator
and learned/fitted predictor: same observed prefix, intervention actions,
autoregressive rollout, physics residual and ensemble uncertainty.

## Какие rich sidenotes нужны

- portrait Andrei Kolmogorov/control-school bridge and primary NeRF card;
- exact quote on model-based control from a primary source;
- JS pinhole projection calculator;
- counterexample MSE averaging two futures;
- counterexample physics prior wrong after bounce;
- epipolar residual drawing;
- question observational correlation vs action effect;
- bridge back to MDP/RNN and forward to agents;
- warning neighboring-frame leakage.

## Недостающие упражнения

1. Project three 3D points with a given \(K[R|t]\).
2. Compute epipolar residual for one matching/wrong point.
3. Discretize NeRF rendering weights and transmittance.
4. Prove linear accumulation of constant velocity bias in autoregressive
   rollout.
5. Browser-experiment seed 8805: bouncing-ball fit, action intervention,
   50-step RMSE, conservation residual and uncertainty.

## План переписывания

1. Split lesson into temporal consistency, geometry, action; make one spine
   scene cross all three.
2. Compute optical-flow residual.
3. Derive projection/epipolar geometry correctly.
4. Compute discrete volume rendering.
5. Rebuild widget as true dynamical simulator.
6. Add action counterfactual/support exploitation.
7. Ground in one real motion scene.
8. Expand to 10–12 sidenotes, 4 inline exercises and 8 tasks.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
