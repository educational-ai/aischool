# Урок 60. Проклятие размерности и no free lunch

## Главная педагогическая идея

При фиксированном разрешении число ячеек растёт как \(m^d\), объём шара
исчезает относительно куба, а расстояния концентрируются; локальные методы
теряют контраст. Спасает не «ещё больше данных вообще», а структура:
low-dimensional manifold, sparsity, invariance, metric learning или projection.
No-free-lunch напоминает, что inductive bias неизбежен.

## Фактическая инвентаризация

1331 слово, 12 заголовков, 4 display-формулы, 4 текстовых sidenotes, 3 SVG,
0 упражнений по ходу и 5 итоговых задач. Интерактив:
`public/interactive/widgets/g11-ml.js`, `buildDimension` (около строки 342),
ключ `60`.

Ученик почувствует \(m^d\), узнает отношение объёмов ball/cube, concentration
расстояний, remedies PCA/JL/structure и no-free-lunch. Он не выводит formula/
asymptotic объёма, не анализирует nearest-neighbor sample complexity и не
различает ambient/intrinsic dimension.

## Чего не хватает

1. Точной таблицы \(m^d\) с единицами storage/sample density.
2. Вывода \(V_d=\pi^{d/2}/\Gamma(d/2+1)\) или recurrence
   \(V_d=2\pi V_{d-2}/d\), затем доли в кубе.
3. Mean/variance квадрата расстояния в кубе и масштаба concentration.
4. Nearest-neighbor radius \(r\approx (k/(n c_d))^{1/d}\).
5. Крайних случаев: sparse/relevant coordinates и данные на 1D manifold в
   100D.
6. Точной формулировки no-free-lunch: усреднение по всем target functions при
   uniform setup, а не «все алгоритмы равны в каждой задаче».

## Рисунки

### `figure-1.svg` — «Сетка расходует данные экспоненциально»

Сравнение 1D/2D/6D ясно, но 6D превращается в декоративный счётчик; нет bytes
или samples на cell. Добавить таблицу для d=1,2,6,20 и расчёт density при
n=\(10^6\), не только blocks.

### `figure-2.svg` — «Доля шара в кубе по размерности»

Подписи `d` близки к curve, смысл логарифмической y слаб. Добавить явные
\(\log_{10}\)-ticks, exact values при d=2,10,50 и recurrence inset; не ставить
labels на линии.

### `figure-3.svg` — «Ближайший и дальний сосед теряют контраст»

Подписи peaks перекрываются; n/seed/distribution отсутствуют. Общая y-scale
неясна. Показать distribution pairwise distances и ratio min/max по repeated
queries с quantile band; подписать ambient data model.

## Интерактив

`buildDimension` корректно считает ball/cube fraction через log-Gamma и cells
\(m^d\). Но “distance concentration” сэмплирует расстояния от начала до
uniform cube points и сравнивает min/mean/max. Это не nearest-vs-farthest
neighbor опыт текста/рисунка, а min сильно зависит от числа точек. Нет
intrinsic structure или JL projection.

Добавить режимы `from centre`, `pairwise`, `nearest neighbour`; повторять
queries, показывать quantiles и зависимость от n. Toggle manifold генерирует
1D-кривую в d dimensions и различает ambient/intrinsic dimension. Projection
mode применяет random JL matrix и строит histogram distortion расстояний.

## Обязательные rich sidenotes

- JS-калькулятор bytes для \(m^d\);
- aphorism: «Высокая размерность пуста, пока структура не говорит, где искать»;
- портрет Владимира Вапника и inductive bias с точным источником;
- контрпример: 1000D-данные на прямой;
- вопрос автора: какая metric делает соседство осмысленным;
- анимация recurrence ball/cube;
- калькулятор nearest-neighbor radius;
- крайний случай irrelevant noise coordinates;
- точная карточка no-free-lunch;
- micro-demo JL distortion;
- мост назад к PCA и вперёд к optimization/representation learning.

## Недостающие математические упражнения

1. При m=10 и d=1,2,6,12 вычислите cells и memory по 4 bytes/cell. При
   \(10^6\) samples найдите среднее samples per cell.
2. Используйте \(V_d=2\pi V_{d-2}/d\), \(V_0=1,V_1=2\), чтобы вычислить
   \(V_2\dots V_6\) и доли внутри куба \([-1,1]^d\).
3. Если координаты \(X,Y\) iid Uniform[0,1], найдите
   \(E[(X-Y)^2]=1/6\); обобщите expected squared distance на d dimensions и
   обсудите масштаб относительных колебаний.
4. Оцените nearest-neighbor radius \(r=(1/n)^{1/d}\) при \(n=10^6\) и
   d=2,10,100. Интерпретируйте.
5. Постройте 100D-точки \((t,0,\dots,0)\) и добавьте 99 noise coordinates с
   sd \(\sigma\). Сравните Euclidean ranking соседей при росте \(\sigma\);
   предложите projection.

## Пошаговый план переписывания

1. Начать с таблицы cells и физической памяти.
2. Вывести recurrence/fraction объёма.
3. Вычислить moments расстояния.
4. Связать их с neighbour radius/sample complexity.
5. Противопоставить ambient/intrinsic dimension.
6. Точно сформулировать no-free-lunch и перечислить biases.
7. Исправить подмену neighbour расстоянием до центра.
8. Добавить 4–5 упражнений, simulation protocol и 7–8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | PARTIAL |
| D. Интерактив | PARTIAL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
