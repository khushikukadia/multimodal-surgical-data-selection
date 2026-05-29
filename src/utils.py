"""Shared utilities: seeding, logging helpers, embedding normalization."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and (if available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not installed; that is fine for parts of the pipeline that
        # do not need it (e.g. random sampling + sklearn classifier).
        pass


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization. Safe for zero vectors."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def log(msg: str) -> None:
    """Tiny logger so progress is visible without configuring a real logger."""
    print(f"[milestone3] {msg}", flush=True)
