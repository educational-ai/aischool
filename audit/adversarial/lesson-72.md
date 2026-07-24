# Урок 72. Self-play: от AlphaGo к AlphaZero

## Главная педагогическая идея

Self-play превращает текущую policy в генератор возрастающих задач, но
стабильность требует population/league, оценки против фиксированных
оппонентов и поиска вроде MCTS. Один рейтинг не описывает нетранзитивную игру.

## Что есть сейчас

1460 слов до задач, 11 разделов, 7 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-rl`
(`buildSelfPlay`) и 5 задач. Проверены visuals, source и screenshot. Текст
разбирает RPS/minimax, MCTS, policy/value network, AlphaZero loop, league/Elo,
OpenSpiel и нетранзитивность.

Ученик поймёт, почему игра с собой поставляет labels и зачем хранить старых
оппонентов. Но виджет не реализует почти ничего из этого, а статья не выводит
minimax equilibrium и MCTS backup на числовом дереве.

## Чего не хватает в рассуждении

1. Решения RPS equilibrium через indifference/linear constraints.
2. Числового MCTS tree: \(N,W,Q,P\), selection UCB/PUCT, expansion, backup
   после одного rollout.
3. Различия policy evaluation, self-play training и league matchmaking.
4. Контрпримера best-response cycle: новый агент побеждает current champion,
   но проигрывает старому.
5. Elo likelihood и ограничения транзитивной scalar model.
6. Exploitability/NashConv как метрики, не только win rate.
7. Reproducible OpenSpiel matrix/game snapshot и seeds.

## Рисунки и интерактив

### `figure-1.svg`

У левой оси обрезана подпись `ножницы`; matrix не имеет colorbar/чисел,
graph directions неясны. Нужна exact payoff matrix с \(-1,0,1\), simplex и
best-response arrows.

### `figure-2.svg`

Схема MCTS содержит английский footer, почти нет arrowheads, а caption обещает
\((N,Q,P)\) до/после, которых в figure не видно. Нужен один concrete path,
числа в каждом node и выделенный backup.

### `figure-3.svg`

League heatmaps без color scale и readable values; подписи мелки. Нужна
antisymmetric payoff matrix, opponent sampling weights и exploitability over
time.

### `buildSelfPlay`

Виджет не симулирует matches, MCTS или league. Он детерминированно двигает
RPS mixture к soft best response на mixture истории. `population` — просто
усреднение прошлого, не roster agents; нет stochastic games, match matrix,
evaluation, run/reset. Заменить на population из explicit strategies,
round-robin matches с seed, selector opponent distribution, step generation
best response и dashboard current-vs-history/exploitability.

## Какие rich sidenotes нужны

- портрет Джона фон Неймана и точная minimax citation;
- карточка Александра Кронрода и советских работ по игровым программам;
- JS MCTS node, где один click делает selection/backup;
- контрпример rock–paper–scissors к Elo;
- counterexample champion forgetting old opponent;
- рисунок simplex RPS;
- вопрос: кто выдаёт «правильный ход» в self-play;
- мост назад к actor–critic и вперёд к preference tournaments;
- warning о тесте против собственной последней версии.

## Недостающие упражнения

1. Доказать uniform mixed equilibrium RPS и value 0.
2. Выполнить два PUCT selections и backups на заданном tree.
3. Построить три strategies с cyclic payoff и показать невозможность
   scalar ordering.
4. Рассчитать Elo update и показать зависимость от assumed transitivity.
5. Browser-experiment seed 7205: league 12 agents, 1000 matches per round,
   compare latest-only vs population sampling по exploitability.

## План переписывания

1. Открыть RPS simplex и решить minimax.
2. Показать naive latest-opponent cycle.
3. Разобрать MCTS одним числовым rollout.
4. Собрать AlphaZero loop из отдельных sources/targets.
5. Ввести league и exploitability.
6. Пересобрать widget в реальный tournament laboratory.
7. Добавить OpenSpiel snapshot и failure cases.
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
