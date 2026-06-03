# Milestone 3 — Experimental Protocol

| Item | Setting |
|------|---------|
| **Dataset** | CholecTrack20 |
| **Task** | Frame-level surgical phase classification |
| **Data** | 1 fps annotated frames |
| **Split** | **Video-disjoint** (whole videos to train/val/test — no frame leakage). Official 10/2/8 split when all videos are extracted; otherwise a video-grouped 70/15/15 split via `split_mode=video`. |
| **Classifier** | Logistic regression on frozen visual embeddings |
| **Sampling budgets** | 1%, 5%, 10%, 25%, 50% of train frames (+ 100% full-data upper bound) |
| **Methods** | random, **stratified random**, vision-only k-means, multimodal k-means |
| **Seeds** | average over ≥5 sampling seeds; report mean ± std |
| **Metrics** | accuracy, macro F1, weighted F1, per-class F1 |

## End-to-end commands

```bash
# 1. Download (after setting env vars — see data/.env.example)
python data/get_dataset.py

# 2. Metadata CSV (official splits preserved in `split` column)
python scripts/prepare_metadata.py cholectrack20 \
  --root_dir data/cholectrack20 \
  --output_csv data/cholectrack20_metadata.csv

# 3. Experiment — video-disjoint split, low budgets, 5 sampling seeds
python scripts/run_experiment.py \
  --metadata_csv data/cholectrack20_metadata.csv \
  --output_dir results \
  --budgets 0.01 0.05 0.1 0.25 0.5 \
  --seed 0 \
  --seeds 0 1 2 3 4 \
  --split_mode video \
  --embedding_backend clip
```

`split_mode=auto` also picks a video-disjoint split automatically whenever a
`video_id` column is present; use `split_mode=video` to force it. Avoid
`split_mode=random` for reported numbers — it splits at the frame level and
leaks near-duplicate frames across train/test.

## Stage 2 — fine-tuning the visual encoder (GPU)

Tests the stronger question: does a better-selected subset matter *more* once
the encoder can adapt? Reuses the same subsets, partially fine-tunes the last
few backbone blocks, and early-stops on the (video-disjoint) val split.

```bash
python scripts/run_experiment.py \
  --metadata_csv data/cholectrack20_metadata.csv \
  --output_dir results \
  --budgets 0.05 0.1 0.25 \
  --seeds 0 1 2 3 4 \
  --split_mode video \
  --embedding_backend clip \
  --finetune \
  --finetune_budgets 0.05 0.1 0.25 \
  --finetune_seeds 0 1 2 \
  --unfreeze_blocks 3 \
  --backbone_lr 1e-5 --head_lr 1e-3 \
  --ft_max_epochs 10 --ft_batch_size 32
```

Notes:
- **Use a CUDA GPU.** Fine-tuning ViT-B/32 on CPU/MPS is impractical at this scale.
- Skip the 1% budget for fine-tuning (a few hundred frames just overfit an 86M-param ViT).
- Keep `--finetune_seeds` small (e.g. 3); the grid is methods x budgets x seeds runs.
- `--unfreeze_blocks N` unfreezes the last N transformer blocks (CLIP) / residual stages (ResNet) plus a fresh head; the rest stays frozen.
- On a GPU box, raise `FineTuneConfig.num_workers` (or edit the default) for faster image loading.

## Outputs

- `results/results.csv` — frozen aggregated mean (+ `*_std`) per (method, budget).
- `results/results_raw.csv` — frozen, one row per (method, budget, seed).
- `results/results_finetune.csv` / `results_finetune_raw.csv` — fine-tuning results (when `--finetune`).
- `results/finetune_vs_frozen_macro_f1.png` — headline plot: frozen (dashed) vs fine-tuned (solid).
- `results/per_class_f1*.csv`, budget plots, `milestone3_summary.md` (with a fine-tuning + gap-analysis section).

## Design notes

- **Sampling** may use multimodal (visual + text) embeddings; the **classifier** always sees visual embeddings only.
- **Train** split = only pool for subset selection; **test** split = all reported metrics.
- **Val** split is reserved (not used in the default loop).
- **stratified** is the control for **multimodal**: since the multimodal embedding concatenates the ground-truth phase-label text vector, multimodal k-means effectively clusters by label. If multimodal ≈ stratified, the gain is just class balancing.
