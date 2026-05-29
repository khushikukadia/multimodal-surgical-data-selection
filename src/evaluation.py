"""Metrics and plotting helpers."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend so this works on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from .utils import ensure_dir, log


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Top-line metrics for the results CSV."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]
) -> Dict[str, float]:
    """Per-class F1 keyed by label. Missing labels get 0."""
    f1_arr = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: float(s) for lab, s in zip(labels, f1_arr)}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: str,
    title: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Normalize by row (true class) so per-class behavior is readable when
    # classes are very imbalanced.
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.6)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalized)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"Saved confusion matrix to {save_path}.")


def plot_metric_vs_budget(
    results_df: pd.DataFrame,
    metric: str,
    save_path: str,
    full_data_value: Optional[float] = None,
) -> None:
    """Line plot of ``metric`` vs ``budget`` with one line per sampling method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = sorted(m for m in results_df["method"].unique() if m != "full")
    for method in methods:
        sub = results_df[results_df["method"] == method].sort_values("budget")
        ax.plot(sub["budget"], sub[metric], marker="o", label=method)
    if full_data_value is not None:
        ax.axhline(
            full_data_value,
            linestyle="--",
            color="grey",
            label=f"full-data ({full_data_value:.3f})",
        )
    ax.set_xlabel("Training-data budget (fraction of train split)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs training-data budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"Saved plot to {save_path}.")


def save_results(
    results: List[Dict],
    per_class_records: List[Dict],
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)
    results_df = pd.DataFrame(results)
    per_class_df = pd.DataFrame(per_class_records)

    results_path = os.path.join(output_dir, "results.csv")
    per_class_path = os.path.join(output_dir, "per_class_f1.csv")
    results_df.to_csv(results_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    log(f"Wrote {results_path} ({len(results_df)} rows) and {per_class_path}.")
    return results_df, per_class_df
