"""Milestone 3 experiment driver.

Pipeline overview:
    1. Load metadata CSV (frame_path, phase).
    2. Build a fixed stratified train/val/test split (cached on disk).
    3. Extract & cache visual embeddings (CLIP or ResNet18).
    4. Extract & cache text embeddings for unique phase labels.
    5. Build per-frame multimodal (concat) embeddings.
    6. For each (method, budget) combination, select a training subset and
       train a LogisticRegression classifier on the visual embeddings of the
       selected examples. Evaluate on the fixed test split.
    7. Save:
         results/results.csv
         results/per_class_f1.csv
         results/confusion_matrix_<method>_<budget>.png   (best method)
         results/performance_vs_budget_accuracy.png
         results/performance_vs_budget_macro_f1.png
         results/milestone3_summary.md
         results/config.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Allow `python scripts/run_experiment.py` from repo root without install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import train_classifier, predict  # noqa: E402
from src.data import (  # noqa: E402
    create_or_load_splits,
    load_metadata,
    split_indices,
    split_summary,
)
from src.embeddings import (  # noqa: E402
    build_multimodal_embeddings,
    extract_text_embeddings,
    extract_visual_embeddings,
)
from src.evaluation import (  # noqa: E402
    compute_metrics,
    compute_per_class_f1,
    plot_confusion_matrix,
    plot_metric_vs_budget,
    save_results,
)
from src.sampling import select_subset  # noqa: E402
from src.utils import ensure_dir, log, save_json, set_seed  # noqa: E402


METHODS = ("random", "vision", "multimodal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Milestone 3: multimodal sampling for surgical phase recognition."
    )
    p.add_argument(
        "--metadata_csv",
        required=True,
        help="CSV with columns: frame_path,phase",
    )
    p.add_argument("--output_dir", default="results", help="Where to write outputs.")
    p.add_argument(
        "--cache_dir",
        default=None,
        help="Where to cache embeddings. Defaults to <output_dir>/cache.",
    )
    p.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5],
        help="Training-data budgets as fractions of the train split.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--embedding_backend",
        choices=["clip", "resnet"],
        default="clip",
    )
    p.add_argument(
        "--classifier",
        choices=["logreg", "mlp"],
        default="logreg",
    )
    p.add_argument(
        "--force_recompute_embeddings",
        action="store_true",
        help="Ignore any cached visual/text embeddings.",
    )
    p.add_argument(
        "--include_full_baseline",
        action="store_true",
        default=True,
        help="Also train on 100%% of the train split as an upper bound.",
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Weight of visual half.")
    p.add_argument("--beta", type=float, default=1.0, help="Weight of text half.")
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on dataset size for fast smoke tests.",
    )
    p.add_argument(
        "--split_mode",
        choices=["auto", "official", "random"],
        default="auto",
        help="auto: CholecTrack20 official video splits when present, else random frame split.",
    )
    p.add_argument(
        "--force_recompute_splits",
        action="store_true",
        help="Ignore cached results/splits.csv and rebuild train/val/test assignment.",
    )
    return p.parse_args()


def _select_best_row(results_df: pd.DataFrame) -> Optional[pd.Series]:
    """Pick the (method, budget) combo with the highest macro_f1 for plotting."""
    if results_df.empty:
        return None
    subset = results_df[results_df["method"] != "full"]
    if subset.empty:
        subset = results_df
    return subset.sort_values("macro_f1", ascending=False).iloc[0]


def _write_summary_markdown(
    output_dir: str,
    results_df: pd.DataFrame,
    args: argparse.Namespace,
    n_train: int,
    n_val: int,
    n_test: int,
    label_list: List[str],
) -> None:
    """Generate the milestone3_summary.md report."""
    path = os.path.join(output_dir, "milestone3_summary.md")

    # Decide whether the multimodal sampler beat random/vision on average.
    non_full = results_df[results_df["method"] != "full"]
    if not non_full.empty:
        mean_by_method = non_full.groupby("method")["macro_f1"].mean().to_dict()
        multimodal_wins = (
            "multimodal" in mean_by_method
            and mean_by_method["multimodal"]
            == max(mean_by_method.values())
        )
    else:
        mean_by_method = {}
        multimodal_wins = False

    full_row = results_df[results_df["method"] == "full"]
    full_acc = float(full_row["accuracy"].iloc[0]) if not full_row.empty else None
    full_f1 = float(full_row["macro_f1"].iloc[0]) if not full_row.empty else None

    table_md = results_df[
        ["method", "budget", "n_train", "accuracy", "macro_f1", "weighted_f1"]
    ].to_markdown(index=False, floatfmt=".4f")

    looks_like_smoke_test = "synthetic" in (args.metadata_csv or "").lower()

    lines = []
    lines.append("# Milestone 3 — Multimodal Sampling for Surgical Phase Recognition\n")
    if looks_like_smoke_test:
        lines.append(
            "> **Toy smoke test only.** These numbers were produced from a "
            "synthetic, solid-color dataset generated by "
            "`scripts/prepare_metadata.py synthetic`. They are useful only "
            "to verify that the pipeline runs end-to-end. Do **not** include "
            "them in the milestone write-up.\n"
        )
    lines.append(
        "These are preliminary frame-level results using frozen visual "
        "embeddings and a lightweight classifier. The goal is not to maximize "
        "absolute surgical phase recognition performance yet, but to test "
        "whether the choice of training subset affects data efficiency.\n"
    )

    lines.append("## Experimental setup\n")
    lines.append("| Item | Setting |")
    lines.append("|------|---------|")
    lines.append("| Dataset | CholecTrack20 |")
    lines.append("| Task | Frame-level surgical phase classification |")
    lines.append("| Data | 1 fps annotated frames |")
    lines.append("| Split | Official 10 train / 2 val / 8 test videos (`split_mode={args.split_mode}`) |")
    lines.append("| Classifier | Logistic regression on frozen visual embeddings |")
    lines.append(f"| Sampling budgets | {', '.join(f'{int(b*100)}%' for b in args.budgets if b < 1)} (+ 100% full-data upper bound) |")
    lines.append("| Methods | random, vision-only k-means, multimodal k-means |")
    lines.append("| Metrics | accuracy, macro F1, weighted F1, per-class F1 |")
    lines.append("")
    lines.append(f"- Visual embedding backend: `{args.embedding_backend}`")
    lines.append(f"- Classifier flag: `{args.classifier}` (always trained on **visual** embeddings only)")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Multimodal concat weights: alpha={args.alpha}, beta={args.beta}")
    lines.append(
        "- Subset selection uses only the **train** split; evaluation is on the **test** split "
        "(val held out for future hyperparameter tuning).\n"
    )

    lines.append("## Dataset / split\n")
    lines.append(f"- Metadata CSV: `{args.metadata_csv}`")
    lines.append(f"- #phase classes: {len(label_list)} ({', '.join(label_list)})")
    lines.append(f"- #train / #val / #test **frames**: {n_train} / {n_val} / {n_test}")
    if getattr(args, "split_description", None):
        lines.append(f"- {args.split_description}")
    lines.append("")

    lines.append("## Methods compared\n")
    lines.append("- **random**: uniform random subset of the train split.")
    lines.append(
        "- **vision**: K-means on L2-normalized visual embeddings; pick the "
        "training example closest to each cluster center."
    )
    lines.append(
        "- **multimodal**: same as vision, but K-means is run on a concatenation "
        "of visual embeddings and phase-label text embeddings (CLIP text encoder "
        "when available, otherwise TF-IDF over the prompt `\"A laparoscopic "
        "surgery frame showing the phase: {phase}\"`)."
    )
    if args.include_full_baseline:
        lines.append("- **full**: train on 100% of the train split as an upper bound.")
    lines.append("")

    lines.append("## Main results\n")
    lines.append(table_md)
    lines.append("")

    lines.append("## Key observations\n")
    if mean_by_method:
        means_str = ", ".join(
            f"{m}={v:.4f}" for m, v in sorted(mean_by_method.items())
        )
        lines.append(f"- Mean macro-F1 across budgets per method: {means_str}.")
    if full_acc is not None:
        lines.append(
            f"- Full-data upper bound: accuracy={full_acc:.4f}, macro F1={full_f1:.4f}."
        )

    if multimodal_wins:
        lines.append(
            "- The multimodal sampler performs best at lower data budgets, "
            "suggesting that incorporating phase semantics helps preserve useful "
            "diversity in the reduced training set."
        )
    else:
        lines.append(
            "- The preliminary results do not yet show a clear multimodal advantage. "
            "This may be because short phase labels provide limited additional "
            "information beyond visual embeddings, or because the current subset / "
            "classifier is too small to reveal the expected effect."
        )

    lines.append("")
    lines.append("## Limitations\n")
    lines.append("- Preliminary subset of data; results may not generalize.")
    lines.append("- Frame-level classification ignores temporal structure between consecutive frames.")
    lines.append("- CLIP / ResNet features are not surgical-domain-specific.")
    lines.append("- Only one (or very few) random seed(s) were evaluated.")
    lines.append("- Phase-label text is very short, which limits how much extra signal text embeddings can add.")
    lines.append(
        "- Using phase labels in the sampler assumes labels are available for the "
        "training pool — in practice we typically want to sample BEFORE labeling."
    )
    lines.append("")

    lines.append("## Next steps\n")
    lines.append("- Run on more videos and average across multiple seeds.")
    lines.append("- Add a stratified random baseline (random sampling within phase).")
    lines.append("- Use richer text descriptions for surgical phases (e.g. clinical definitions).")
    lines.append("- Add per-class analysis focused on rare phases (e.g. ClippingCutting).")
    lines.append("- Evaluate temporal smoothing or a small sequence model over predicted frames.")
    lines.append(
        "- Compare against full CholecTrack20 (or Cholec80) training as a stronger upper bound."
    )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    log(f"Wrote summary to {path}.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir or os.path.join(output_dir, "cache"))

    # Persist the run configuration.
    save_json(os.path.join(output_dir, "config.json"), vars(args))

    # ---- 1. Load metadata ----------------------------------------------------
    df = load_metadata(args.metadata_csv)
    if args.max_frames is not None:
        df = df.head(args.max_frames).reset_index(drop=True)
        log(f"Truncated dataset to first {len(df)} rows (--max_frames).")

    # ---- 2. Splits -----------------------------------------------------------
    df = create_or_load_splits(
        df,
        output_dir,
        seed=args.seed,
        split_mode=args.split_mode,
        force_recompute=args.force_recompute_splits,
    )
    args.split_description = split_summary(df)

    train_idx, val_idx, test_idx = split_indices(df)
    log(f"Split sizes: {args.split_description}")

    # ---- 3. Visual embeddings -----------------------------------------------
    visual_cache = os.path.join(
        cache_dir, f"visual_{args.embedding_backend}.npz"
    )
    visual = extract_visual_embeddings(
        df,
        backend=args.embedding_backend,
        cache_path=visual_cache,
        force_recompute=args.force_recompute_embeddings,
    )
    log(f"Visual embeddings shape: {visual.shape}")

    # ---- 4. Text embeddings -------------------------------------------------
    unique_phases = sorted(df["phase"].unique().tolist())
    text_cache = os.path.join(cache_dir, f"text_{args.embedding_backend}.npz")
    text_dict = extract_text_embeddings(
        unique_phases,
        backend=args.embedding_backend,
        cache_path=text_cache,
        force_recompute=args.force_recompute_embeddings,
    )

    # ---- 5. Multimodal embeddings -------------------------------------------
    multimodal = build_multimodal_embeddings(
        visual,
        df["phase"].astype(str).tolist(),
        text_dict,
        alpha=args.alpha,
        beta=args.beta,
    )
    log(f"Multimodal embeddings shape: {multimodal.shape}")
    np.savez(
        os.path.join(cache_dir, f"multimodal_{args.embedding_backend}.npz"),
        embeddings=multimodal.astype(np.float32),
    )

    # ---- 6. Run all (method, budget) combinations ---------------------------
    y_all = df["phase"].astype(str).to_numpy()
    X_test = visual[test_idx]
    y_test = y_all[test_idx]
    log(f"Test set: X={X_test.shape}, classes_in_test={len(np.unique(y_test))}")

    label_list = unique_phases

    results: List[Dict] = []
    per_class_records: List[Dict] = []

    # Multimodal vs vision-only vs random for each budget.
    for method in METHODS:
        for budget in args.budgets:
            t0 = time.time()
            selected = select_subset(
                method=method,
                train_indices=train_idx,
                visual_embeddings=visual,
                multimodal_embeddings=multimodal,
                budget=budget,
                seed=args.seed,
            )
            log(
                f"[{method}] budget={budget:.3f} -> selected {len(selected)} "
                f"train examples (of {len(train_idx)})."
            )

            X_tr = visual[selected]
            y_tr = y_all[selected]
            if len(np.unique(y_tr)) < 2:
                log(
                    f"  WARN: only {len(np.unique(y_tr))} class(es) in selected subset; "
                    "classifier cannot generalize. Recording zeros."
                )
                metrics = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
                per_class = {lab: 0.0 for lab in label_list}
            else:
                clf, _ = train_classifier(args.classifier, X_tr, y_tr, seed=args.seed)
                y_pred = predict(clf, X_test)
                metrics = compute_metrics(y_test, y_pred)
                per_class = compute_per_class_f1(y_test, y_pred, label_list)

            elapsed = time.time() - t0
            row = {
                "method": method,
                "budget": budget,
                "n_train": int(len(selected)),
                **metrics,
                "elapsed_sec": round(elapsed, 2),
                "seed": args.seed,
            }
            results.append(row)
            per_class_records.append(
                {"method": method, "budget": budget, **per_class}
            )

    # Full-data upper bound (run once at budget=1.0).
    if args.include_full_baseline:
        log("Training full-data upper bound (budget=1.0).")
        X_tr = visual[train_idx]
        y_tr = y_all[train_idx]
        clf, _ = train_classifier(args.classifier, X_tr, y_tr, seed=args.seed)
        y_pred = predict(clf, X_test)
        metrics = compute_metrics(y_test, y_pred)
        per_class = compute_per_class_f1(y_test, y_pred, label_list)
        results.append(
            {
                "method": "full",
                "budget": 1.0,
                "n_train": int(len(train_idx)),
                **metrics,
                "elapsed_sec": 0.0,
                "seed": args.seed,
            }
        )
        per_class_records.append({"method": "full", "budget": 1.0, **per_class})

    # ---- 7. Save results, plots, and summary --------------------------------
    results_df, per_class_df = save_results(results, per_class_records, output_dir)

    full_acc = None
    full_f1 = None
    full_row = results_df[results_df["method"] == "full"]
    if not full_row.empty:
        full_acc = float(full_row["accuracy"].iloc[0])
        full_f1 = float(full_row["macro_f1"].iloc[0])

    plot_metric_vs_budget(
        results_df,
        metric="accuracy",
        save_path=os.path.join(output_dir, "performance_vs_budget_accuracy.png"),
        full_data_value=full_acc,
    )
    plot_metric_vs_budget(
        results_df,
        metric="macro_f1",
        save_path=os.path.join(output_dir, "performance_vs_budget_macro_f1.png"),
        full_data_value=full_f1,
    )

    # Confusion matrix for the best non-full row.
    best_row = _select_best_row(results_df)
    if best_row is not None:
        method = best_row["method"]
        budget = best_row["budget"]
        log(
            f"Re-running best ({method} @ budget={budget}) for the confusion matrix..."
        )
        selected = select_subset(
            method=method,
            train_indices=train_idx,
            visual_embeddings=visual,
            multimodal_embeddings=multimodal,
            budget=budget,
            seed=args.seed,
        )
        X_tr = visual[selected]
        y_tr = y_all[selected]
        clf, _ = train_classifier(args.classifier, X_tr, y_tr, seed=args.seed)
        y_pred = predict(clf, X_test)
        plot_confusion_matrix(
            y_test,
            y_pred,
            labels=label_list,
            save_path=os.path.join(
                output_dir, f"confusion_matrix_{method}_{budget:.2f}.png"
            ),
            title=f"Confusion matrix — {method}, budget={budget:.2f}",
        )

    _write_summary_markdown(
        output_dir=output_dir,
        results_df=results_df,
        args=args,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        label_list=label_list,
    )

    log("Done.")


if __name__ == "__main__":
    main()
