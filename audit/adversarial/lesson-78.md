# Урок 78. Предобучение и самообучение LLM

## Главная педагогическая идея

Self-supervised next-token objective дешёво создаёт targets, но corpus pipeline
и mixture скрыто программируют распределение ошибок модели. Extraction,
deduplication, filtering, privacy and contamination должны быть измерены по
дороге, а не перечислены.

## Что есть сейчас

1399 слов до задач, 12 разделов, 4 display-формулы, 0 определений,
0 встроенных упражнений, 4 sidenotes, 3 SVG, виджет `g11-generative`
(`buildPretraining`) и 5 задач. Проверены visuals, source и screenshot. Текст
охватывает next-token labels, pipeline, Jaccard dedup, mixture epochs, PII,
benchmark leakage, transfer, FineWeb and genre-filter audits. Финальные задачи
очень детальны.

Ученик поймёт, что corpus не нейтрален, duplicates меняют weights, validation
нужна по domains. Но не увидит ни одного реального document record до/после
filters и не сможет проверить widget scores.

## Чего не хватает в рассуждении

1. Exact document-to-examples table with BOS/EOS, packing and loss mask.
2. MinHash derivation/false collision probability; Jaccard alone не
   объясняет scalable dedup.
3. Quantitative pipeline ledger: documents/tokens removed at each stage,
   with uncertainty/manual precision audit.
4. Counterexample quality filter selecting encyclopedic style over useful
   dialect.
5. Mixture optimization on real validation losses, not invented skill
   coefficients.
6. Memorization experiment as exposure/membership, not mere warning.
7. FineWeb sample manifest, date, licence/terms, fields and actual 20
   redacted records.

## Рисунки и интерактив

### `figure-1.svg`

Shifted targets are clear but does not show packed-document boundary, EOS or
loss masking. Add two short documents packed into one sequence and mark
forbidden cross-document target.

### `figure-2.svg`

Bars/arrows are arbitrary, no axes, counts or uncertainty; arrowheads pass
near bar corners. Caption claims token-width flows but reader cannot verify.
Replace with real Sankey/table from 5000 documents and manual-error samples.

### `figure-3.svg`

Contamination curves are smooth synthetic claims without model, data or seed.
Need real small-model experiment, paired structural variants and intervals.

### `buildPretraining`

Widget calculates four «skills» by arbitrary linear formulas of mixture
sliders; duplicates simply discard budget. No documents, tokenizer, model,
loss, quality filter (despite text), training or held-out domains. It is a
renamed radar chart. Replace with fixed real microcorpus, unigram/bigram
language model or tiny in-browser neural model, explicit sampling stream,
dedup toggle, per-domain validation BPB and run/reset.

## Какие rich sidenotes нужны

- portrait of Claude Shannon with source, tied to compression objective;
- exact quote from dataset documentation, not marketing;
- JS shingle/Jaccard inspector;
- counterexample copied answer across train/test;
- counterexample quality filter deleting poetry/dialect;
- illustration data lineage for one document;
- question: what is a «token epoch» in mixture;
- bridge back to tokenizer and forward to scaling laws;
- warning public webpage ≠ unrestricted training licence.

## Недостающие упражнения

1. Build causal target/mask for two packed documents and identify leakage.
2. Compute Jaccard and one MinHash collision estimate.
3. Given source sizes/weights, calculate effective epochs and duplication.
4. Construct threshold confusion table for quality filter by genre.
5. Browser/data experiment: 5000 fixed Common Crawl records, seed 7805 manual
   sample, before/after token ledger and domain validation BPB.

## План переписывания

1. Open with one document's complete lineage.
2. Build exact next-token examples and packing mask.
3. Quantify extraction/filter/dedup pipeline.
4. Derive Jaccard/MinHash and show failure cases.
5. Replace widget with real micro-training stream.
6. Treat mixture as measured multi-domain optimization.
7. Add PII/licence/contamination evidence cards.
8. Expand to 10 sidenotes, 4 inline exercises and 8 tasks.

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
