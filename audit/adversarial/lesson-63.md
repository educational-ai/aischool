# Урок 63. Oracle inequalities и выбор модели

## Главная педагогическая идея

Выбранную по данным модель надо сравнивать не с недостижимым абсолютным
идеалом, а с лучшим кандидатом внутри заданного семейства плюс ценой выбора.
Это содержательная идея статистического мышления: validation сама становится
источником overfitting, когда кандидатов много.

## Что есть сейчас

1340 слов до задач, 12 разделов, 10 display-формул, 1 определение,
0 встроенных упражнений, 5 sidenotes, 3 SVG, виджет `g11-ml`
(`buildOracle`) и 5 задач. Просмотрены три SVG и desktop screenshot.
Есть Hoeffding/union-bound, схема cross-validation, nested CV и хорошая
формулировка oracle bound. Финальная задача просит доказать bound \(2\epsilon\),
но доказательство не дано в статье.

Ученик поймёт, почему минимум validation error оптимистичен, как число
кандидатов входит логарифмически и зачем нужен nested split. Не поймёт, где
именно используется независимость, что меняется для зависимых кандидатов и как
выглядит selection bias в числах.

## Чего не хватает в рассуждении

1. Полного доказательства:
   если \(\sup_m|\hat R_m-R_m|\le\epsilon\), то
   \(R_{\hat m}\le R_{m^*}+2\epsilon\), с подписанными тремя неравенствами.
2. Вывода union bound от индивидуальной Hoeffding-оценки с явной
   вероятностью \(1-\delta\), а не только итоговой формулы.
3. Monte Carlo из 100 одинаково плохих кандидатов: истинный risk один,
   минимум validation всё равно падает.
4. Различия model selection и hyperparameter tuning; correlated candidates
   уменьшают эффективное число сравнений, но не устраняют bias.
5. Контрпримера: выбрать degree polynomial после просмотра test, затем
   честного nested protocol.
6. Structural risk minimization или penalized criterion как альтернативы.
7. Малого реального набора с manifest и зафиксированным candidate grid.

## Рисунки и интерактив

### `figure-1.svg`

Четыре boxplot выглядят убедительно, но не показаны отдельные candidates,
истинный общий risk, seed и число симуляций. Эта картинка не позволяет
проверить механизм extreme minimum. Нужен swarm из validation estimates,
выбранный минимум и test estimate того же кандидата при \(M=2,10,100\).

### `figure-2.svg`

Кривые bound аккуратны, но \(\delta\) отсутствует визуально, а вертикальная
шкала не разделяет empirical risk и penalty. Нужна интерактивная derivation:
confidence bands каждого кандидата и общая simultaneous band.

### `figure-3.svg`

Стрелки fold 1–4 сходятся в одну точку рамки, наконечники накладываются,
fold 5 визуально вынесен в сторону; nested CV читается как схема проводов.
Перерисовать в два ряда: outer test-fold и inner validation-fold, с отдельным
цветом данных, которые никогда не выбирают hyperparameters.

### `buildOracle`

Четыре sliders управляют candidates, validation size, penalty и seed, но
«истина» — детерминированная парабола, шум добавлен только к validation, а
penalty — произвольный линейный коэффициент сложности. Реального test
sampling нет, хотя подпись обещает разницу risk. Нет `run/reset`, repeat
distribution, candidate correlations или nested split. Замена: генератор
train/validation/test из известного polynomial, fit degrees 0–12, 200 повторов
с seed и overlay oracle/selected risk; кнопка «подсмотреть test» должна
наглядно загрязнять следующий выбор.

## Какие rich sidenotes нужны

- портрет Владимира Вапника и точное место oracle-style сравнения в истории
  statistical learning;
- цитата Джорджа Бокса об «all models are wrong» с источником и оговоркой;
- JS-опыт «100 честных монет, выбери лучшую»;
- контрпример: одна модель, но 100 checkpoints — тоже 100 кандидатов;
- иллюстрация simultaneous vs pointwise intervals;
- вопрос: можно ли считать seed новым кандидатом;
- мост назад к confidence intervals и вперёд к scaling-law fit;
- warning о adaptive leaderboard;
- маленькая историческая справка про AIC/SRM.

## Недостающие упражнения

1. Полностью доказать \(R_{\hat m}\le R_{m^*}+2\epsilon\) на событии
   uniform deviation.
2. Из Hoeffding
   \(P(|\hat R_m-R_m|>\epsilon)\le2e^{-2n\epsilon^2}\) вывести simultaneous
   bound для \(M\) и решить его относительно \(\epsilon\).
3. Для десяти validation errors и известных true risks вручную выбрать
   empirical winner, oracle и optimism.
4. Сконструировать два зависимых кандидата, где union bound сильно
   консервативен, и два независимых, где minimum bias велик.
5. Browser-experiment seed 6305: degrees 0–12, \(n=120\), 500 repeats;
   сравнить single split, 5-fold и nested protocol.

## План переписывания

1. Начать с конкурса одинаковых монет и неожиданно хорошего минимума.
2. Вывести pointwise Hoeffding, union bound и oracle inequality без пропусков.
3. Показать точную числовую симуляцию selection bias.
4. Развести train, selection validation и untouched test.
5. Пересобрать nested CV-рисунок.
6. Добавить коррелированные candidates, checkpoints и leaderboard как
   контрпримеры.
7. Дать реальный polynomial/data case и 4 inline-упражнения.
8. Завершить 8 задачами, включая proof и reproducible browser experiment.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | FAIL |
| G. Данные | FAIL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
