# Урок 84. Metric learning и мультимодальность

## Главная педагогическая идея

Contrastive image–text learning uses batch negatives to create a shared
geometry, but false negatives, temperature, prompt templates and geographic
split determine what that geometry means. Similarity is ranking evidence, not
probability.

## Что есть сейчас

1430 слов до задач, 11 разделов, 6 display-формул, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildMultimodal`) и 5 tasks. Checked source, visuals and screenshot. Text
contains symmetric contrastive loss, temperature, multi-positive, retrieval,
zero-shot prompting, remote sensing, RSICD and ANN audit. Tasks are strong.

Ученик поймёт batch similarity matrix and false negatives. Не вычислит
gradient одного hard negative and won't see actual image/text pairs in the
lesson or widget.

## Чего не хватает в рассуждении

1. Full \(3\times3\) normalized embeddings, logits, two directional losses.
2. Derivative \(\partial L/\partial s_{ij}=p_{ij}-1_{i=j}\).
3. Multi-positive gradient comparison and false-negative conflict.
4. Prompt-ensemble vector normalization order.
5. Retrieval metrics with multiple relevant items and nDCG.
6. Counterexample normalization discarding informative norm.
7. Actual satellite image/caption snapshot, geocoordinates, licence and
   region split manifest.

## Рисунки и интерактив

### `figure-1.svg`

Heatmap has no numeric color scale; the promised false negative cannot be
quantified. Add exact logits, selected row softmax at two temperatures and
gradient signs.

### `figure-2.svg`

Embedding scatter is clear but a 2D projection may invent neighborhoods.
Need original-space similarities alongside projection and marked graded
relevance.

### `figure-3.svg`

Cards are placeholders with tiny captions/similarities, not real
images/texts. This violates the core multimodal premise. Replace with licensed
RSICD/EuroSAT thumbnails, exact checkpoint and retrieval scores.

### `buildMultimodal`

Synthetic centers are generated and controls directly improve a hand-written
«alignment» formula. More negatives do not train encoders; no images, text,
loss gradient or multi-positive option. Similarity uses negative squared
distance, while article defines normalized cosine. Replace with real fixed
embeddings for 12 licensed image-caption pairs, exact cosine/logits,
single/multi-positive losses, prompt edits and retrieval ranking.

## Какие rich sidenotes нужны

- CLIP primary-source card and exact model revision;
- satellite-image licence/provenance thumbnail;
- JS one-row contrastive gradient;
- counterexample two valid captions treated as negatives;
- counterexample watermark shortcut;
- drawing original high-dimensional vs 2D projection;
- question similarity vs calibrated probability;
- bridge back to token prompts and forward to diffusion conditioning;
- warning geographic leakage.

## Недостающие упражнения

1. Compute symmetric contrastive loss for \(3\times3\) logits.
2. Derive logit gradient.
3. Compare single/multi-positive gradient numerically.
4. Build relevance set where Recall@1 marks a semantically valid result wrong.
5. Browser/data experiment: fixed 1000 COCO pairs or EuroSAT split, exact
   checkpoint, prompt variants, R@K and manual alternative audit.

## План переписывания

1. Open with three actual image-caption pairs.
2. Compute normalized embeddings and matrix.
3. Derive loss/gradient and false negative.
4. Introduce retrieval and graded relevance.
5. Rebuild widget on fixed real embeddings.
6. Add prompts/domain/geographic split.
7. Include ANN error separately.
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
