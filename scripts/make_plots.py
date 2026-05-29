"""Re-render the milestone 3 plots from a saved `results.csv`.

Useful if you tweak plotting after a long experiment run and do not want to
recompute embeddings or classifiers.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import plot_metric_vs_budget  # noqa: E402
from src.utils import log  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_csv", default="results/results.csv")
    p.add_argument("--output_dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(args.results_csv)
    df = pd.read_csv(args.results_csv)

    full_row = df[df["method"] == "full"]
    full_acc = float(full_row["accuracy"].iloc[0]) if not full_row.empty else None
    full_f1 = float(full_row["macro_f1"].iloc[0]) if not full_row.empty else None

    plot_metric_vs_budget(
        df,
        metric="accuracy",
        save_path=os.path.join(args.output_dir, "performance_vs_budget_accuracy.png"),
        full_data_value=full_acc,
    )
    plot_metric_vs_budget(
        df,
        metric="macro_f1",
        save_path=os.path.join(args.output_dir, "performance_vs_budget_macro_f1.png"),
        full_data_value=full_f1,
    )
    log("Plots regenerated.")


if __name__ == "__main__":
    main()
