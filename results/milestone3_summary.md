# Milestone 3 — Multimodal Sampling for Surgical Phase Recognition

These are preliminary frame-level results using frozen visual embeddings and a lightweight classifier. The goal is not to maximize absolute surgical phase recognition performance yet, but to test whether the choice of training subset affects data efficiency.

## Experimental setup

| Item | Setting |
|------|---------|
| Dataset | CholecTrack20 |
| Task | Frame-level surgical phase classification |
| Data | 1 fps annotated frames |
| Split | `split_mode=video` — train=12738 frames, val=4365 frames, test=2287 frames | train=8 videos, val=2 videos, test=2 videos |
| Classifier | Logistic regression on frozen visual embeddings |
| Sampling budgets | 5%, 10% (+ 100% full-data upper bound) |
| Methods | random, stratified random, vision-only k-means, multimodal k-means |
| Metrics | accuracy, macro F1, weighted F1, per-class F1 |

- Visual embedding backend: `clip`
- Classifier flag: `logreg` (always trained on **visual** embeddings only)
- Split seed: 0
- Sampling seeds: [0, 1, 2, 3, 4] — reported metrics are the mean over 5 seeds (std in `*_std` columns of `results.csv`, per-seed rows in `results_raw.csv`).
- Multimodal concat weights: alpha=1.0, beta=1.0
- Subset selection uses only the **train** split; evaluation is on the **test** split (val held out for future hyperparameter tuning).

## Dataset / split

- Metadata CSV: `data/cholectrack20_metadata.csv`
- #phase classes: 7 (CalotTriangleDissection, CleaningCoagulation, ClippingCutting, GallbladderDissection, GallbladderExtraction, GallbladderPackaging, Preparation)
- #train / #val / #test **frames**: 12738 / 4365 / 2287
- train=12738 frames, val=4365 frames, test=2287 frames | train=8 videos, val=2 videos, test=2 videos

## Methods compared

- **random**: uniform random subset of the train split.
- **stratified**: random subset that preserves per-phase class proportions. Key control: because the multimodal embedding concatenates the (true) phase-label text vector, multimodal k-means clusters by label, so any gain may just be class balancing.
- **vision**: K-means on L2-normalized visual embeddings; pick the training example closest to each cluster center.
- **multimodal**: same as vision, but K-means is run on a concatenation of visual embeddings and phase-label text embeddings (CLIP text encoder when available, otherwise TF-IDF over the prompt `"A laparoscopic surgery frame showing the phase: {phase}"`).
- **full**: train on 100% of the train split as an upper bound.

## Main results

| method     |   budget |   n_train |   accuracy |   macro_f1 |   weighted_f1 |
|:-----------|---------:|----------:|-----------:|-----------:|--------------:|
| full       |   1.0000 |     12738 |     0.5627 |     0.4761 |        0.5892 |
| multimodal |   0.0500 |       637 |     0.4754 |     0.3617 |        0.4973 |
| multimodal |   0.1000 |      1274 |     0.4689 |     0.3628 |        0.4945 |
| random     |   0.0500 |       637 |     0.4448 |     0.3130 |        0.4657 |
| random     |   0.1000 |      1274 |     0.4753 |     0.3585 |        0.4979 |
| stratified |   0.0500 |       637 |     0.4483 |     0.3237 |        0.4682 |
| stratified |   0.1000 |      1274 |     0.4801 |     0.3771 |        0.5037 |
| vision     |   0.0500 |       637 |     0.4661 |     0.3355 |        0.4835 |
| vision     |   0.1000 |      1274 |     0.4886 |     0.3638 |        0.5101 |

## Key observations

- Mean macro-F1 across budgets per method: multimodal=0.3623, random=0.3358, stratified=0.3504, vision=0.3496.
- Averaged over 5 seeds; typical per-cell macro-F1 std ~0.0197. Treat method gaps smaller than ~1 std as noise.
- Full-data upper bound: accuracy=0.5627, macro F1=0.4761.
- The multimodal sampler performs best at lower data budgets, suggesting that incorporating phase semantics helps preserve useful diversity in the reduced training set.

## Limitations

- Preliminary subset of data; results may not generalize.
- Frame-level classification ignores temporal structure between consecutive frames.
- CLIP / ResNet features are not surgical-domain-specific.
- Only one (or very few) random seed(s) were evaluated.
- Phase-label text is very short, which limits how much extra signal text embeddings can add.
- Using phase labels in the sampler assumes labels are available for the training pool — in practice we typically want to sample BEFORE labeling.

## Next steps

- Run on more videos and average across multiple seeds.
- Add a stratified random baseline (random sampling within phase).
- Use richer text descriptions for surgical phases (e.g. clinical definitions).
- Add per-class analysis focused on rare phases (e.g. ClippingCutting).
- Evaluate temporal smoothing or a small sequence model over predicted frames.
- Compare against full CholecTrack20 (or Cholec80) training as a stronger upper bound.

## Fine-tuning stage (partial encoder fine-tuning)

Same selected subsets as above, but the visual encoder's last blocks are fine-tuned (early-stopped on the video-disjoint val split). The question: does a better-selected subset matter *more* once the model can adapt its representation?

| method     |   budget |   n_train |   accuracy |   macro_f1 |   weighted_f1 |
|:-----------|---------:|----------:|-----------:|-----------:|--------------:|
| multimodal |   0.0500 |       637 |     0.6495 |     0.4756 |        0.6337 |
| multimodal |   0.1000 |      1274 |     0.6517 |     0.4622 |        0.6454 |
| random     |   0.0500 |       637 |     0.6154 |     0.3521 |        0.5791 |
| random     |   0.1000 |      1274 |     0.6515 |     0.4580 |        0.6292 |
| stratified |   0.0500 |       637 |     0.6511 |     0.4221 |        0.6464 |
| stratified |   0.1000 |      1274 |     0.6572 |     0.4632 |        0.6421 |
| vision     |   0.0500 |       637 |     0.6032 |     0.3880 |        0.5992 |
| vision     |   0.1000 |      1274 |     0.6677 |     0.4725 |        0.6579 |

### Frozen vs fine-tuned — does the gap widen?

macro-F1 gap of multimodal over the stratified / random baselines, frozen vs fine-tuned (positive = multimodal better):

|   budget | baseline   |   frozen_gap |   finetuned_gap | widened   |
|---------:|:-----------|-------------:|----------------:|:----------|
|   0.0500 | stratified |       0.0380 |          0.0535 | yes       |
|   0.0500 | random     |       0.0486 |          0.1235 | yes       |
|   0.1000 | stratified |      -0.0142 |         -0.0011 | yes       |
|   0.1000 | random     |       0.0043 |          0.0041 | no        |

- Multimodal's advantage widened after fine-tuning in 3/4 (budget, baseline) cells. A consistent widening supports the hypothesis that selection matters more when the encoder can learn; no widening suggests the frozen-feature ceiling, not sampling, was the limiting factor.
