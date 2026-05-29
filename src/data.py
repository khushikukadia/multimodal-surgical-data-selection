"""Metadata loading and fixed train/val/test split creation."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import ensure_dir, log

REQUIRED_COLUMNS = ("frame_path", "phase")

# CholecTrack20 official folder / CSV split names -> internal names.
OFFICIAL_SPLIT_MAP = {
    "training": "train",
    "train": "train",
    "tr": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "testing": "test",
    "test": "test",
}


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load a CSV with columns frame_path,phase.

    Extra columns (e.g. video_id, split, frame_id) are allowed and preserved.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_path}\n"
            "Expected a CSV with columns: frame_path,phase\n"
            "Run `python scripts/prepare_metadata.py --help` for details."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata CSV {csv_path} is missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )
    n_before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if len(df) < n_before:
        log(
            f"Dropped {n_before - len(df)} rows with missing frame_path/phase "
            f"(kept {len(df)})."
        )
    df["phase"] = df["phase"].astype(str)
    log(f"Loaded metadata: {len(df)} frames, {df['phase'].nunique()} unique phases.")
    return df


def _stratified_safe(
    df: pd.DataFrame, label_col: str, min_per_class: int = 3
) -> bool:
    counts = df[label_col].value_counts()
    return bool((counts >= min_per_class).all())


def _infer_split_from_path(frame_path: str) -> Optional[str]:
    """Infer CholecTrack20 split from a frame path like .../training/VID01/..."""
    parts = frame_path.replace("\\", "/").lower().split("/")
    for part in parts:
        if part in OFFICIAL_SPLIT_MAP:
            return OFFICIAL_SPLIT_MAP[part]
    return None


def has_official_splits(df: pd.DataFrame) -> bool:
    """True if metadata carries CholecTrack20-style official video splits."""
    if "split" in df.columns:
        normalized = (
            df["split"].astype(str).str.strip().str.lower().map(OFFICIAL_SPLIT_MAP)
        )
        if normalized.notna().mean() >= 0.95:
            return True
    # Fallback: infer from frame_path subfolders (training/, validation/, testing/).
    if "frame_path" in df.columns:
        inferred = df["frame_path"].astype(str).map(_infer_split_from_path)
        if inferred.notna().mean() >= 0.95:
            return True
    return False


def apply_official_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Map CholecTrack20 official splits to internal train / val / test labels."""
    out = df.copy()
    if "split" in out.columns:
        mapped = (
            out["split"].astype(str).str.strip().str.lower().map(OFFICIAL_SPLIT_MAP)
        )
        if mapped.isna().any():
            # Mixed column: fill missing from path.
            inferred = out["frame_path"].astype(str).map(_infer_split_from_path)
            mapped = mapped.fillna(inferred)
    else:
        mapped = out["frame_path"].astype(str).map(_infer_split_from_path)

    if mapped.isna().any():
        bad = int(mapped.isna().sum())
        raise ValueError(
            f"Could not assign official split for {bad} rows. "
            "Ensure metadata was built with `prepare_metadata.py cholectrack20` "
            "or that frame_path contains training/, validation/, or testing/."
        )

    out["split"] = mapped
    return out


def split_summary(df: pd.DataFrame) -> str:
    """Human-readable split stats for logging / the milestone summary."""
    counts = df["split"].value_counts().to_dict()
    parts = [f"{k}={counts.get(k, 0)} frames" for k in ("train", "val", "test")]
    line = ", ".join(parts)
    if "video_id" in df.columns:
        for split_name in ("train", "val", "test"):
            vids = df.loc[df["split"] == split_name, "video_id"].nunique()
            parts.append(f"{split_name}_videos={vids}")
        line += " | " + ", ".join(
            f"{k}={df.loc[df['split'] == k, 'video_id'].nunique()} videos"
            for k in ("train", "val", "test")
            if k in counts
        )
    return line


def create_or_load_splits(
    df: pd.DataFrame,
    output_dir: str,
    seed: int,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    split_mode: str = "auto",
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Create or load fixed train/val/test assignments.

    ``split_mode``:
        - ``auto`` (default): use official CholecTrack20 video splits when the
          metadata CSV (or frame paths) indicate training/validation/testing;
          otherwise fall back to a stratified random frame split.
        - ``official``: require official splits (raises if not found).
        - ``random``: always use stratified random frame split (70/15/15).

    Saves ``<output_dir>/splits.csv`` so all sampling methods share the same
    partition on subsequent runs.
    """
    ensure_dir(output_dir)
    splits_path = os.path.join(output_dir, "splits.csv")

    use_official = split_mode == "official" or (
        split_mode == "auto" and has_official_splits(df)
    )

    if os.path.exists(splits_path) and not force_recompute:
        existing = pd.read_csv(splits_path)
        if len(existing) == len(df) and "split" in existing.columns:
            log(f"Reusing cached splits at {splits_path}.")
            merged = df.copy().drop(columns=["split"], errors="ignore")
            merged = merged.merge(
                existing[["frame_path", "split"]], on="frame_path", how="left"
            )
            if merged["split"].isna().any():
                log("Cached splits did not cover every frame; recomputing.")
            else:
                log(f"Split summary: {split_summary(merged)}")
                return merged

    if use_official:
        if split_mode == "official" and not has_official_splits(df):
            raise ValueError(
                "split_mode=official but metadata has no CholecTrack20 split "
                "column or path hints. Build metadata with:\n"
                "  python scripts/prepare_metadata.py cholectrack20 --root_dir ..."
            )
        out = apply_official_splits(df)
        log(
            "Using CholecTrack20 official video splits "
            "(10 train / 2 val / 8 test videos when all splits are included)."
        )
    else:
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6
        log(
            f"Using stratified random frame split "
            f"({train_size:.0%}/{val_size:.0%}/{test_size:.0%}, seed={seed})."
        )
        stratify = df["phase"] if _stratified_safe(df, "phase") else None
        if stratify is None:
            log(
                "Some phase classes have <3 examples; falling back to a "
                "non-stratified random split."
            )

        remainder_size = train_size + val_size
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

        if stratify is not None:
            stratify_inner = train_val_df["phase"]
            if not _stratified_safe(train_val_df, "phase", min_per_class=2):
                stratify_inner = None
        else:
            stratify_inner = None

        relative_val = val_size / remainder_size
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=seed,
            stratify=stratify_inner,
        )

        train_df = train_df.assign(split="train")
        val_df = val_df.assign(split="val")
        test_df = test_df.assign(split="test")
        out = pd.concat([train_df, val_df, test_df], ignore_index=True)
        out = (
            out.set_index("frame_path")
            .loc[df["frame_path"].tolist()]
            .reset_index()
        )

    out.to_csv(splits_path, index=False)
    log(f"Wrote splits to {splits_path}: {split_summary(out)}")
    return out


def split_indices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer indices for train, val, test relative to df."""
    train_idx = np.flatnonzero((df["split"] == "train").to_numpy())
    val_idx = np.flatnonzero((df["split"] == "val").to_numpy())
    test_idx = np.flatnonzero((df["split"] == "test").to_numpy())
    return train_idx, val_idx, test_idx
