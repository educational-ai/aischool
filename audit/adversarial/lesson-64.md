# Урок 64. Федеративное обучение и честные модели

## Главная педагогическая идея

FedAvg — оптимизация по распределённым неоднородным данным, а веса
агрегации, выбор клиентов, privacy и group metrics входят в математическую
постановку. Пересылка весов вместо строк сама по себе не означает ни
приватность, ни справедливость.

## Что есть сейчас

1393 слова до задач, 13 разделов, 7 display-формул, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-ml`
(`buildFederated`) и 5 задач. Проверены все SVG и desktop screenshot. Текст
охватывает FedAvg, non-IID drift, client weighting, secure aggregation,
differential privacy, leakage и LEAF. Охват широк, но каждый механизм получает
по абзацу вместо выводимой модели.

Ученик поймёт цикл broadcast–local update–aggregate и различит равный вес
клиента от веса по числу примеров. Не сможет посчитать drift при нескольких
local steps, доказать secure sum на игрушечном протоколе или выбрать шум для
заданной privacy guarantee.

## Чего не хватает в рассуждении

1. Двухклиентского quadratic с точным сравнением centralized gradient,
   FedAvg \(E=1\) и \(E>1\); это центральный механизм client drift.
2. Вывода, когда weighted local gradients дают unbiased global gradient, и
   что ломает partial participation.
3. Контрпримера fairness: 990 объектов клиента A и 10 клиента B, где
   sample-weighted accuracy отлична, а B полностью проиграна.
4. Secure aggregation на трёх числах с парными масками, которые взаимно
   сокращаются.
5. Численного DP-примера: clipping \(C\), Gaussian noise, privacy accountant
   хотя бы как таблица зависимости \(\sigma\), rounds и \(\varepsilon\).
6. Различия record-level и client-level privacy.
7. Реального snapshot FEMNIST/LEAF с лицензией, client histogram и готовым
   маленьким manifest.

## Рисунки и интерактив

### `figure-1.svg`

Схема практически нечитаема: все клиентские и серверные стрелки пересекаются
в центре, несколько наконечников лежат поверх `w_t → w_{t+1}`. Именно
главный рисунок урока выглядит как spaghetti. Нужны три горизонтальные фазы
и отдельная aggregation bar, где длина сегмента равна весу клиента.

### `figure-2.svg`

Local trajectories и FedAvg visual понятны, но нет стрелок времени, weights,
centralized reference и масштаба local steps. Добавить две quadratics,
траектории \(E=1,5,20\), central step и вектор drift.

### `figure-3.svg`

Треугольник privacy–utility–fairness неколичественный, смешивает русский и
английский, подписи прижаты к границам. Это плакат, не аргумент. Заменить на
три реальные кривые: noise–accuracy, client sampling–worst-group error и
rounds–privacy budget, с единицами и uncertainty.

### `buildFederated`

Четыре sliders двигают три скалярные quadratic и рисуют условную траекторию.
Server rounds искусственно разнесены вверх на 1.8 px, поэтому вертикальное
изменение не является параметром модели. Нет client sampling, dropout,
privacy noise, secure aggregation или group metric; текст предлагает менять
weighting rule, но интерфейс не даёт такого переключателя. Нужен настоящий
двух/пятиклиентский engine, кнопки `один local step`, `один round`,
`run/reset`, выбор sample/equal weights, dropout и clipping/noise; рядом
centralized counterfactual.

## Какие rich sidenotes нужны

- портрет/биографическая карточка Якова Цыпкина или советской школы
  распределённых адаптивных систем с точным источником;
- цитата из McMahan et al. 2017 о communication efficiency;
- JS secure-sum из трёх телефонов и взаимно уничтожающихся масок;
- контрпример «веса не являются анонимными» через gradient inversion;
- рисунок record-level vs client-level adjacency;
- вопрос: кто представлен в раунде, если телефон ночью offline;
- мост назад к mini-batch sampling и вперёд к group evaluation;
- warning: secure aggregation не исправляет poisoned update;
- маленькая историческая линия от distributed optimization до on-device
  learning.

## Недостающие упражнения

1. Два клиента имеют \(F_A(w)=\frac12(w-0)^2\),
   \(F_B(w)=\frac12(w-4)^2\), веса .9/.1. Вычислить centralized optimum и
   один FedAvg round при \(E=1,5\).
2. Доказать сокращение парных масок для трёх клиентов и разобрать dropout
   одного участника.
3. Для размеров 990/10 сравнить sample-weighted и equal-client objectives и
   их minimizers.
4. При clipping \(C=1\) и noise std 0.5 сэмплировать 1000 aggregate updates;
   измерить bias/variance, seed 6404.
5. Browser-experiment: 20 clients с non-IID label skew, 50 rounds,
   пять seeds; график mean и worst-client accuracy.

## План переписывания

1. Открыть двумя телефонами с противоположными quadratics.
2. Вывести centralized objective и FedAvg \(E=1\).
3. Рассчитать client drift при \(E>1\).
4. Ввести weighting/participation через одну таблицу population.
5. Отдельно разобрать secure aggregation и DP, не смешивая обещания.
6. Пересобрать лабораторию как настоящий round simulator.
7. Добавить LEAF snapshot, 10 rich sidenotes и 4 inline exercises.
8. Завершить 8 задачами с proof, privacy calculation и reproducible run.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
