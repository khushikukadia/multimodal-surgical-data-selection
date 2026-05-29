# Less Data, Better Models: Multimodal Sampling for Surgical AI

CS231N project on data-efficient **surgical phase recognition**. We test whether
sampling the training set with *both* visual and language embeddings of the
phase labels selects more useful frames than random or vision-only sampling,
especially at low data budgets.

The repository implements a minimal, reproducible pipeline:

1. Load a metadata CSV of surgical frames and phase labels.
2. Cache frozen visual embeddings (CLIP ViT-B/32, with ResNet18 as a fallback).
3. Cache text embeddings for the unique phase labels (CLIP text encoder, with
   TF-IDF as a fallback).
4. Pick a training subset using one of: `random`, `vision`-only K-means
   diversity, `multimodal` (visual+text) K-means diversity, or `full`-data.
5. Train a logistic-regression head on the **visual** embeddings of the
   selected subset and evaluate on a fixed held-out test split.
6. Compare accuracy / macro F1 / per-class F1 across data budgets and dump
   plots and a short markdown summary.

The classifier always sees **visual embeddings only** — only the *sampling*
method differs between conditions.

---

## Dataset: CholecTrack20 (recommended)

We use **[CholecTrack20](https://github.com/CAMMA-public/cholectrack20)** instead of
full Cholec80 when disk space is limited:

| | CholecTrack20 | Cholec80 |
|---|----------------|----------|
| Videos | 20 | 80 |
| Frame rate in release | **1 fps** (pre-sampled) | Often 1 fps extracted from 25 fps |
| Phase labels | Yes (7 phases, in JSON) | Yes |
| Typical size | Smaller zip; ~35k labeled frames | Much larger |

**Download** (requires DUA + request form — see the [CholecTrack20 README](https://github.com/CAMMA-public/cholectrack20)):

1. Complete the [request form](https://docs.google.com/forms/d/e/1FAIpQLSdewhAi0vGmZj5DLOMWdLf85BhUtTedS28YzvHS58ViwuEX5w/viewform) for an access key.
2. Download from [Synapse](https://www.synapse.org/Synapse:syn53182642/wiki/).

**Save disk space:** each video folder contains `images/` (needed) and sometimes
an `.mp4` (optional for this project). You can delete the `.mp4` files after
extracting — the pipeline only reads `images/*.png`.

**Pilot run on a few videos:**

```bash
python scripts/prepare_metadata.py cholectrack20 \
  --root_dir /path/to/CholecTrack20 \
  --output_csv data/cholectrack20_metadata.csv \
  --splits training validation \
  --video_ids VID01 VID02 VID04
```

---

## Expected metadata format

The pipeline is driven by a CSV with two required columns:

```
frame_path,phase
/path/to/CholecTrack20/training/VID01/images/000042.png,CalotTriangleDissection
```

- `frame_path` — path to a single RGB frame (`.png` or `.jpg`).
- `phase` — surgical phase label string.

Extra columns (`video_id`, `split`, `frame_id`) from `prepare_metadata.py` are
preserved but unused by the experiment script.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `open-clip-torch` is hard to install, use ResNet18:

```bash
python scripts/run_experiment.py ... --embedding_backend resnet
```

---

## Build the metadata CSV

### Option A — CholecTrack20 (recommended)

After unzipping, point at the dataset root (folder that contains `training/`,
`validation/`, `testing/`):

```bash
python scripts/prepare_metadata.py cholectrack20 \
  --root_dir /path/to/CholecTrack20 \
  --output_csv data/cholectrack20_metadata.csv
```

Expected layout per video:

```
CholecTrack20/training/VID01/
    VID01.json          # frame_id -> list of tool records (includes "phase")
    images/000000.png   # 1 fps frames
    images/000001.png
```

Phase IDs in JSON map to names such as `Preparation`, `CalotTriangleDissection`,
`GallbladderDissection`, `ClippingCutting`, etc. (see `scripts/prepare_metadata.py`).

### Option B — legacy Cholec80

If you already have Cholec80 extracted:

```bash
python scripts/prepare_metadata.py cholec80 \
  --cholec80_dir /path/to/cholec80 \
  --output_csv data/cholec80_metadata.csv
```

### Option C — synthetic smoke test

```bash
python scripts/prepare_metadata.py synthetic \
  --output_csv data/synthetic_metadata.csv \
  --frames_dir data/synthetic_frames
```

Do **not** use synthetic results in your milestone write-up.

---

## Run the experiment

```bash
python scripts/run_experiment.py \
  --metadata_csv data/cholectrack20_metadata.csv \
  --output_dir results \
  --budgets 0.1 0.25 0.5 \
  --seed 0 \
  --embedding_backend clip
```

Useful flags:

| flag | default | purpose |
|------|---------|---------|
| `--embedding_backend` | `clip` | `clip` or `resnet` |
| `--classifier` | `logreg` | `logreg` or `mlp` |
| `--budgets` | `0.1 0.25 0.5` | fraction of train split used |
| `--force_recompute_embeddings` | off | ignore cached `.npz` |
| `--max_frames` | none | cap rows for quick tests |

Embeddings are cached under `results/cache/`.

---

## Outputs

Under `results/`:

- `results.csv`, `per_class_f1.csv`
- `performance_vs_budget_accuracy.png`, `performance_vs_budget_macro_f1.png`
- `confusion_matrix_<method>_<budget>.png`
- `milestone3_summary.md`, `config.json`, `splits.csv`

---

## Layout

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_experiment.py
│   ├── prepare_metadata.py   # cholectrack20 | cholec80 | synthetic
│   └── make_plots.py
├── src/
│   ├── data.py
│   ├── embeddings.py
│   ├── sampling.py
│   ├── classifier.py
│   ├── evaluation.py
│   └── utils.py
└── results/
```
