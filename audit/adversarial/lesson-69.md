# Урок 69. Управляемый Марковский процесс и Беллман

## Главная педагогическая идея

Bellman operator превращает длинную задачу управления в локальное сравнение
«reward сейчас + value следующего состояния»; contraction при
\(\gamma<1\) делает value iteration проверяемой процедурой, а не магией
обучения.

## Что есть сейчас

1580 слов до задач, 10 разделов, 13 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildBellman`) и 5 задач. Проверены visuals, implementation и screenshot.
Есть MDP tuple, Bellman expectation/optimality, contraction proposition,
value iteration, reward shaping, CityLearn и uncertainty.

Ученик поймёт state/action/reward/transition, различит \(V^\pi\) и optimal
value и сможет выполнить update. Он не увидит proof contraction в статье,
policy extraction/gap или точного анализа stochastic slip.

## Чего не хватает в рассуждении

1. Доказательства sup-norm contraction через
   \(|\max_a f_a-\max_a g_a|\le\max_a|f_a-g_a|\).
2. Error bound
   \(\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty\) и stopping по
   Bellman residual.
3. Полного two-route MDP: transitions, rewards, \(V_1,V_2\), policy switch
   при изменении \(\gamma\).
4. Контрпримера \(\gamma=1\) без terminal state, где contraction исчезает.
5. Разницы reward shaping, сохраняющего policy
   \(F(s,s')=\gamma\Phi(s')-\Phi(s)\), и произвольного bonus.
6. Q-gap как устойчивости выбранного action.
7. CityLearn snapshot/episode, units kWh/рубли/CO2 и baseline controller.

## Рисунки и интерактив

### `figure-1.svg`

Branch lines входят в boxes и ellipses, наконечники/подпись
\(R+\gamma V\) висят отдельно. Рисунок слишком пуст и не показывает
expectation по вероятностям. Нужен decision node, chance nodes с \(p\), суммы
по ветвям и числовой backup.

### `figure-2.svg`

Четыре heatmap без colorbar, чисел, координат и policy arrows, хотя caption
обещает arrows. По ним нельзя проверить convergence. Добавить общий scale,
iteration labels, greedy arrows и residual.

### `figure-3.svg`

Три stacked plots читаемы, но не связаны вертикальными event markers и units.
Нужен реальный load/price/storage trace, baseline и накопленная стоимость.

### `buildBellman`

Grid-world value iteration математически честен, но world фиксирован, нет
редактирования rewards/goal/walls, `one step/run/reset`, Q-values, policy gap
или uncertainty. Текст просит сравнивать маршруты и intervention, которых
controls не дают. Пересобрать как Bellman microscope: клик по клетке показывает
четыре \(Q(s,a)\), weighted successors и update; `step` двигает ровно одну
итерацию, `run` строит residual.

## Какие rich sidenotes нужны

- портрет Ричарда Беллмана и точная цитата о curse of dimensionality;
- русская историческая карточка Льва Понтрягина как параллельной школы
  optimal control;
- JS-backup одной клетки;
- контрпример \(\gamma=1\) с бесконечной положительной петлёй;
- контрпример reward hacking от неправильного bonus;
- рисунок policy gap/uncertain action;
- вопрос о Markov state: достаточно ли текущей температуры без времени;
- мост назад к PageRank fixed point и вперёд к Q-learning;
- warning о reward units.

## Недостающие упражнения

1. Для two-state, two-action MDP вручную сделать три value iterations.
2. Доказать contraction Bellman optimality operator.
3. Из residual \(\|TV-V\|_\infty\le\epsilon\) вывести bound до \(V^*\).
4. Построить potential shaping и проверить неизменность action ranking.
5. Browser-experiment seed 6905: stochastic slip, 5 seeds, сравнить policy
   switch и Q-gap при \(\gamma=.5,.9,.99\).

## План переписывания

1. Начать одним полностью числовым decision tree.
2. Ввести MDP tuple только после дерева.
3. Вывести Bellman backup и contraction.
4. Связать residual, iterations и policy gap.
5. Пересобрать grid laboratory с пошаговым backup.
6. Добавить reward-shaping counterexample.
7. Разобрать один CityLearn day с units и provenance.
8. Добавить 10 sidenotes, 4 inline exercises и 8 задач.

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
