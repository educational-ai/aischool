# Урок 90. Итоговый проект: оркестр моделей

## Главная педагогическая идея

The capstone should turn one falsifiable question into a chain of versioned
artifacts—data, split, baseline, model, error audit, interactive and
reproduction—not reward architecture complexity. This is the correct course
ending, but the current page is still an assignment brief, not a worked
magazine article.

## Что есть сейчас

1316 слов до задач, 12 разделов, 0 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildOrchestra`) and 5 tasks. Checked source, all SVG and screenshot. Text
contains question contract, project passport, baselines, roles, interactive
requirements, error taxonomy, red teaming, rubric and defence. Homework is
exceptionally complete, especially the air-quality project.

Ученик поймёт what makes a reproducible project and why baseline/error analysis
matter. Но урок не показывает один completed project with actual tables,
plots, repository tree and decisions; it tells the student to do it.

## Чего не хватает в рассуждении

1. A fully worked mini-project from raw 30-row snapshot to split, baseline,
   model, uncertainty, error and final claim.
2. Exact metric formulas (AP, Brier, bootstrap delta) in article.
3. Decision log showing one hypothesis falsified and plan changed.
4. Counterexample high metric/no action and polished interactive/no evidence.
5. Reproduction manifest/checksum/environment example.
6. Team role handoffs as artifacts, not labels.
7. Backlink audit: several links across course have wrong targets; capstone
   should not advertise a network that is not mechanically checked.

## Рисунки и интерактив

### `figure-1.svg`

Funnel is clean but generic; no actual question evolves through stages.
Replace labels with one air-quality case and show what information is lost if
target/horizon/baseline is omitted.

### `figure-2.svg`

Pipeline boxes connect without visible arrowheads; reviewer diamonds are
detached from stages, feedback/test-freeze unclear. Redraw as artifact DAG
with gate inputs/outputs and immutable test boundary.

### `figure-3.svg`

Dashboard concept is good, but in contact render rotated x labels collide with
the printed `Урок 90 · рисунок 3` caption. Data are generic icons/curves.
Use real project slice table, intervals and linked error examples.

### `buildOrchestra`

The widget is a checklist/radar with arbitrary scores. Budget only changes a
score; no pipeline is assembled, no artifacts are produced or validated, and
there is no run/reset. It fails the lesson's own line «ползунок без решения —
игрушка». Replace with a project-contract validator: drag real artifact cards
(snapshot, split manifest, metric table, interactive spec) into a DAG, run
explicit gates, expose exact missing evidence and export a reproducibility
manifest.

## Какие rich sidenotes нужны

- portrait Mikhail Lagutin/visual-statistics tradition only if directly tied
  to the reference-book method, with licensed image;
- quote by Tukey on data analysis with exact source;
- JS paired-bootstrap mini-calculator;
- counterexample leaked target with superb score;
- counterexample negative result as success;
- illustration immutable test seal;
- question «what action follows 0.73?» with answer branches;
- bridge map to 8 specific prior lessons, verified links;
- warning UNKNOWN provenance must stay UNKNOWN.

## Недостающие упражнения

1. Given a 30-row time series, construct leakage-safe origins and target.
2. Compute baseline MAE/AP/Brier by hand on 8 predictions.
3. Run paired day-bootstrap with seed 9005 and interpret interval.
4. Create a counterexample where a stronger model loses under decision cost.
5. Browser mini-project: fixed bike-sharing snapshot, baseline vs tree,
   threshold interactive, manifest and independent reproduction checklist.

## План переписывания

1. Make one air-quality mini-project the narrative spine.
2. Publish data card/snapshot and leakage-safe target table.
3. Compute baseline and candidate metrics.
4. Show uncertainty/error slices and one falsified claim.
5. Build the exact browser interactive used by the decision.
6. Rebuild orchestra widget as artifact/gate validator.
7. Add reproduction run and verified backlink map.
8. Expand to 12 rich sidenotes, 4 inline exercises and 8 tasks; retain the
   excellent large project task as capstone.

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
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
