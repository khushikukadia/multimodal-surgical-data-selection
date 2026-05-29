# Milestone 3 — Experimental Protocol

| Item | Setting |
|------|---------|
| **Dataset** | CholecTrack20 |
| **Task** | Frame-level surgical phase classification |
| **Data** | 1 fps annotated frames |
| **Split** | Official **10 train / 2 val / 8 test** videos (folder-based) |
| **Classifier** | Logistic regression on frozen visual embeddings |
| **Sampling budgets** | 10%, 25%, 50% of train frames (+ 100% full-data upper bound) |
| **Methods** | random, vision-only k-means, multimodal k-means |
| **Metrics** | accuracy, macro F1, weighted F1, per-class F1 |

## End-to-end commands

```bash
# 1. Download (after setting env vars — see data/.env.example)
python data/get_dataset.py

# 2. Metadata CSV (official splits preserved in `split` column)
python scripts/prepare_metadata.py cholectrack20 \
  --root_dir data/cholectrack20 \
  --output_csv data/cholectrack20_metadata.csv

# 3. Experiment (auto-detects official splits)
python scripts/run_experiment.py \
  --metadata_csv data/cholectrack20_metadata.csv \
  --output_dir results \
  --budgets 0.1 0.25 0.5 \
  --seed 0 \
  --split_mode auto \
  --embedding_backend clip
```

## Design notes

- **Sampling** may use multimodal (visual + text) embeddings; the **classifier** always sees visual embeddings only.
- **Train** split = only pool for subset selection; **test** split = all reported metrics.
- **Val** split is reserved (not used in the default loop).
