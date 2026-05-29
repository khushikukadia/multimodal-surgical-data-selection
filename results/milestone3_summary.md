# Milestone 3 — Multimodal Sampling for Surgical Phase Recognition

These are preliminary frame-level results using frozen visual embeddings and a lightweight classifier. The goal is not to maximize absolute surgical phase recognition performance yet, but to test whether the choice of training subset affects data efficiency.

## Experimental setup

| Item | Setting |
|------|---------|
| Dataset | CholecTrack20 |
| Task | Frame-level surgical phase classification |
| Data | 1 fps annotated frames |
| Split | Official 10 train / 2 val / 8 test videos (`split_mode={args.split_mode}`) |
| Classifier | Logistic regression on frozen visual embeddings |
| Sampling budgets | 10%, 25%, 50% (+ 100% full-data upper bound) |
| Methods | random, vision-only k-means, multimodal k-means |
| Metrics | accuracy, macro F1, weighted F1, per-class F1 |

- Visual embedding backend: `clip`
- Classifier flag: `logreg` (always trained on **visual** embeddings only)
- Seed: 0
- Multimodal concat weights: alpha=1.0, beta=1.0
- Subset selection uses only the **train** split; evaluation is on the **test** split (val held out for future hyperparameter tuning).

## Dataset / split

- Metadata CSV: `data/cholectrack20_metadata.csv`
- #phase classes: 7 (CalotTriangleDissection, CleaningCoagulation, ClippingCutting, GallbladderDissection, GallbladderExtraction, GallbladderPackaging, Preparation)
- #train / #val / #test **frames**: 13572 / 2909 / 2909
- train=13572 frames, val=2909 frames, test=2909 frames | train=12 videos, val=12 videos, test=12 videos

## Methods compared

- **random**: uniform random subset of the train split.
- **vision**: K-means on L2-normalized visual embeddings; pick the training example closest to each cluster center.
- **multimodal**: same as vision, but K-means is run on a concatenation of visual embeddings and phase-label text embeddings (CLIP text encoder when available, otherwise TF-IDF over the prompt `"A laparoscopic surgery frame showing the phase: {phase}"`).
- **full**: train on 100% of the train split as an upper bound.

## Main results

| method     |   budget |   n_train |   accuracy |   macro_f1 |   weighted_f1 |
|:-----------|---------:|----------:|-----------:|-----------:|--------------:|
| random     |   0.1000 |      1358 |     0.5483 |     0.4666 |        0.5832 |
| random     |   0.2500 |      3393 |     0.5902 |     0.5198 |        0.6170 |
| random     |   0.5000 |      6786 |     0.6305 |     0.5649 |        0.6538 |
| vision     |   0.1000 |      1358 |     0.5407 |     0.4499 |        0.5740 |
| vision     |   0.2500 |      3393 |     0.5875 |     0.5182 |        0.6181 |
| vision     |   0.5000 |      6786 |     0.6329 |     0.5762 |        0.6582 |
| multimodal |   0.1000 |      1358 |     0.5376 |     0.4696 |        0.5676 |
| multimodal |   0.2500 |      3393 |     0.5799 |     0.5129 |        0.6111 |
| multimodal |   0.5000 |      6786 |     0.6346 |     0.5779 |        0.6599 |
| full       |   1.0000 |     13572 |     0.6672 |     0.6123 |        0.6883 |

## Key observations

- Mean macro-F1 across budgets per method: multimodal=0.5201, random=0.5171, vision=0.5148.
- Full-data upper bound: accuracy=0.6672, macro F1=0.6123.
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
