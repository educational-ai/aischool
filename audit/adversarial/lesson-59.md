# Урок 59. Переобучение и двойной спуск

## Главная педагогическая идея

Классическая U-кривая bias–variance описывает один режим сложности; около
интерполяции \(p\approx n\) малые singular values усиливают noise и создают
пик, а minimum-norm решения после порога могут обобщать лучше. Double descent
зависит от данных, алгоритма и шума, а не является универсальным законом.

## Фактическая инвентаризация

1281 слово, 11 заголовков, 3 display-формулы, 4 текстовых sidenotes, 3 SVG,
0 упражнений по ходу и 5 итоговых задач. Интерактив:
`public/interactive/widgets/g11-ml.js`, `buildDoubleDescent` (около строки
242), ключ `59`.

Ученик увидит U- и double-descent curves, minimum norm, SVD amplification и
роль label noise. Но этот урок почти дублирует урок 33 и при этом содержит
меньше математики: ученик всё ещё не воспроизводит феномен на данных.

## Чего не хватает

1. Вывода bias–variance, сейчас компоненты только названы.
2. Exact under/overdetermined систем около \(p=n\).
3. SVD-вывода minimum norm и усиления noise.
4. Repeated random-matrix simulation с median/quantiles.
5. Крайних случаев: noiseless labels, ridge убирает пик, anisotropic features.
6. Явного редакционного отличия от урока 33: backlinks не оправдывают дубль.

## Рисунки

### `figure-1.svg` — «Классическая U-кривая и её составные части»

Подписи лежат на curves, у компонентов неясные шкалы, равенство
total=bias²+variance+noise нельзя проверить. Использовать linked values при
выбранной complexity и отделить expected decomposition от illustration.

### `figure-2.svg` — «Двойной спуск около интерполяционного порога»

Кривая передаёт phenomenon, но снова является гладкой авторской формой без
данных, seeds и uncertainty. Показать scatter/median/IQR реального
random-feature experiment, точно отметить p,n и rank.

### `figure-3.svg` — «Малое сингулярное число усиливает шум»

Bars и curves наложены при неясных осях; singular values и inverse coefficients
нужны прямыми парами
\((\sigma_i,u_i^\top y,\text{coefficient})\). Подписи у peaks сталкиваются.
Сделать таблицу и linked vector reconstruction.

## Интерактив

`buildDoubleDescent` использует ручную аналитическую формулу train/test error,
bias, peak и overparameterized tail. Движение p/n, noise и n лишь меняет эту
формулу. Нет \(X\), labels, fitting, SVD, repeats или seeds. Как и урок 33, он
выдаёт спорную empirical shape за детерминированную анимацию.

Заменить browser random-features/ridge regression: генерировать X при n≤120,
менять p, решать через SVD/pseudoinverse, считать train/test по 30 seeds и
показывать spectrum. Noise/ridge toggles должны давать наличие и отсутствие
peak. Либо слить урок с 33, либо оставить 33 концептуальным, а 59 сделать
вычислительным выводом.

## Обязательные rich sidenotes

- JS-pseudoinverse матрицы 2×3;
- афоризм: «На интерполяции самое малое singular direction получает самый
  громкий микрофон»;
- портрет Андрея Тихонова и filter factors;
- контрпример: нет double descent при сильном ridge;
- вопрос автора: что считать p в сети;
- анимация SVD noise;
- крайний случай noiseless labels;
- журнал random-matrix experiment;
- геометрия minimum norm;
- таблица различий уроков 33 и 59;
- мост к проклятию размерности.

## Недостающие математические упражнения

1. Для \(X=(1\ 1),y=2\) найдите все interpolating weights и minimum-norm
   решение; сравните test-прогноз при (1,-1).
2. Singular values: 3,.3,.01, response projections: 1,.2,.02. Вычислите
   pseudoinverse coefficients и ridge filter при \(\lambda=.1\).
3. По таблице bias²/variance/noise для пяти complexities сложите expected
   error и найдите U-minimum; почему это ничего не говорит о post-interpolation?
4. Спроектируйте опыт n=60, p=5…150, Gaussian X, teacher/noise, minimum-norm
   fit, 30 seeds, test n=2000. Задайте все метрики.
5. Сравните noise=0/.2 и ridge=0/.1; сформулируйте проверяемые прогнозы для
   высоты peak и second descent.

## Пошаговый план переписывания

1. Устранить дубль с уроком 33.
2. Вывести геометрию малой линейной интерполяции.
3. Ввести SVD/pseudoinverse.
4. Показать усиление noise.
5. Провести repeated random-feature experiment.
6. Менять noise/ridge/data geometry.
7. Заменить analytic-curve виджет.
8. Добавить 4–5 упражнений и 7–8 задач.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | FAIL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | FAIL |
| H. Связность | FAIL |

**Общий вердикт: FAIL.**
