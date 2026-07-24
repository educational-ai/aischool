# Урок 71. Q-learning, actor и critic

## Главная педагогическая идея

Value-based метод учит цену действий, policy-based — непосредственно
распределение действий, actor–critic использует critic как baseline/оценку
advantage. Сравнение имеет смысл только при одной среде, одинаковой квоте
взаимодействий и честно определённом exploration.

## Что есть сейчас

1652 слова до задач, 11 разделов, 14 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildActorCritic`) и 5 задач. Проверены SVG, source и screenshot. Текст
охватывает Q-learning, SARSA, DQN, replay/target network, actor–critic,
advantage, offline extrapolation и overestimation.

Ученик поймёт TD-error и различие on/off-policy, увидит, зачем нужен critic и
target network. Не выведет policy-gradient theorem даже в конечном случае и
не отделит variance reduction baseline от bias critic.

## Чего не хватает в рассуждении

1. Ручной episode на 3 состояниях с таблицей Q до/после каждого transition.
2. Точного сравнения SARSA и Q-learning на cliff-walking: одинаковые
   transitions, разные targets.
3. Вывода \(\nabla_\theta\log\pi_\theta(a|s)A(s,a)\) из log-derivative trick.
4. Доказательства, что action-independent baseline не меняет expected
   gradient.
5. Контрпримера biased critic, который уменьшает variance, но ведёт actor не
   туда.
6. Double-Q derivation для overestimation.
7. Реального reproducible environment snapshot и learning curves по seeds.

## Рисунки и интерактив

### `figure-1.svg`

State chain занимает мало площади; \(\delta\)-подписи висят под узлами и не
связаны стрелками с конкретными transitions. Нужна episode tape с
\((s,a,r,s')\), target, old Q, new Q.

### `figure-2.svg`

DQN diagram почти без arrowheads; route target-copy непонятен, линии
обрываются у рамок. Перерисовать в два временных масштаба: online update каждый
step, frozen target copy каждые \(K\) steps.

### `figure-3.svg`

Learning curves визуально неплохи, но часть переменных/units не подписана и
неясно, сколько seeds образует band. Добавить environment, interactions,
median/IQR и evaluation policy without exploration.

### `buildActorCritic`

Сравнение нечестно: один slider `exploration` означает epsilon для Q-learning,
но softmax temperature для actor. Critic learning-rate задаётся произвольной
формулой; нет trajectories, critic error, TD table, multiple seeds,
run/reset. Это две рисованные кривые на fixed grid, не лаборатория методов.
Нужны одинаковый action sampler, common transitions, stepwise Q/actor/critic
updates и распределение returns.

## Какие rich sidenotes нужны

- портрет Ричарда Саттона и Андрея Колмогорова/Понтрягина как исторические
  мосты, без натянутой «национальной» связи;
- точная цитата про deadly triad;
- JS TD-update одной строки;
- контрпример cliff: off-policy optimum рискован под exploration;
- counterexample offline unseen action;
- рисунок bias–variance critic;
- вопрос: можно ли оценивать policy на тех же exploratory episodes;
- мост назад к Bellman и вперёд к self-play;
- warning о environment steps как честной единице бюджета.

## Недостающие упражнения

1. Выполнить Q-learning и SARSA update на заданном episode.
2. Доказать нулевое expectation baseline term
   \(\sum_a\pi(a|s)\nabla\log\pi(a|s)b(s)=0\).
3. Для двух noisy action estimates вычислить max-bias Monte Carlo/аналитически.
4. Сконструировать offline dataset без action `right` и показать
   неидентифицируемость его value.
5. Browser-experiment seed 7105: cliff grid, 50 seeds, equal 50k steps,
   returns/train failures/evaluation returns.

## План переписывания

1. Открыть одной episode tape и Q-table.
2. Развести SARSA/Q-learning targets.
3. Объяснить DQN как engineering response к correlation/moving target.
4. Вывести policy gradient и baseline.
5. Собрать actor–critic пошагово.
6. Пересобрать лабораторию с common random streams.
7. Добавить offline/overestimation counterexamples и real environment.
8. Довести до 10 sidenotes, 4 inline exercises и 8 задач.

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
