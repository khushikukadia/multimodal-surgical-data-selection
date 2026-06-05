# Changelog

All notable changes to this project are tracked here.

---

## [Unreleased]

### In Progress
- Paper writeup (CVPR format)

---

## 2026-06-05

### Added
- `results/paper_figures/` folder with all figures and tables ready for Overleaf upload
  - `fig1_phase_examples.png` — 2×4 grid of representative frames per surgical phase
  - `fig3_frozen_vs_finetuned.png` — frozen vs fine-tuned macro-F1 comparison
  - `fig4_per_class_f1.png` — per-phase F1 at 10% budget
  - `fig5_confusion_matrices.png` — confusion matrices side-by-side
  - `latex_figures_and_tables.tex` — ready-to-paste LaTeX for all figures and tables
- `results/results_finetune.csv` and `results_finetune_raw.csv` — fine-tuning results
- `results/finetune_vs_frozen_macro_f1.png` — headline fine-tuning comparison plot

### Changed
- `src/data.py` — added `video_grouped_split()` for proper video-disjoint splits (no frame leakage); fixed KeyError bug when cached `splits.csv` conflicts with existing split column in metadata CSV

### Experiments Run
- **Step 1 (Frozen):** 4 methods × 5 budgets (1%, 5%, 10%, 25%, 50%) × 5 seeds, `split_mode=video`, CLIP backend. Key result: multimodal K-means best at 5% budget (0.362 macro-F1), methods converge at 10%+.
- **Step 2 (Fine-tuning):** 4 methods × 2 budgets (5%, 10%) × 2 seeds, partial encoder fine-tuning (last 3 transformer blocks, 8 epochs, MPS). Key result: multimodal fine-tuned at 5% reaches 0.476 macro-F1, matching full-data frozen baseline.

---

## 2026-06-04

### Added
- `src/finetune.py` — partial CLIP encoder fine-tuning with early stopping on video-disjoint val split (pulled from Khushi's commit `ae482ce`)
- Multi-seed averaging with mean ± std and error bars across all methods
- Stratified random sampling baseline added to `src/sampling.py`

### Changed
- `scripts/run_experiment.py` — major update: added `--seeds`, `--finetune`, `--finetune_budgets`, `--finetune_seeds`, `--split_mode` flags
- `src/evaluation.py` — added per-seed aggregation and error bar support
- Removed stale ResNet cache files (`results/cache/visual_resnet.npz`, etc.)

---

## 2026-06-03

### Added
- Full CholecTrack20 dataset downloaded via Synapse (33 GB, 19,390 frames, 12 videos, 7 phases)
- `data/cholectrack20_metadata.csv` — frame-path + phase label CSV built by `scripts/prepare_metadata.py`
- Initial frozen-feature experiment: CLIP ViT-B/32 embeddings + logistic regression, random split (later replaced with video-disjoint)
- `results/results.csv`, performance plots, confusion matrices, `milestone3_summary.md`

### Fixed
- `data/get_dataset.py` — installed missing `synapseclient` and `tabulate` dependencies
- Initial run used frame-level random split (inflated macro-F1 from 0.476 → 0.612 due to near-duplicate leakage); corrected to video-disjoint split

---

## Notes on AI Assistance

> The following section is for **debugging and development reference only** — not part of the scientific record.

- **Environment / dependency issues:** Python path (`/opt/miniconda3/bin/python`), missing `synapseclient`, `tabulate`, `torchvision`, `open-clip-torch` — all installed via pip with AI assistance.
- **Bug fix (`src/data.py`):** KeyError when loading cached `splits.csv` if metadata CSV already had a `split` column — fixed by dropping existing column before merge. One-line fix.
- **Chart generation:** `results/make_presentation_charts.py` and `results/make_paper_figures.py` — helper scripts written with AI assistance for generating publication-ready figures from `results.csv` and `results_finetune.csv`.
- **BibTeX lookup:** References for related work section fetched via AI web search (19 entries).
- **LaTeX formatting:** `latex_figures_and_tables.tex` template generated with AI assistance; all numbers are directly from experiment CSVs.
- **`.gitignore` update:** Added `data/cholectrack20/` and `results/cache/` to prevent accidental large-file commits.
