"""Subset selection strategies used to build the training set.

All functions operate on integer indices into the train split, so the caller
controls how those indices map back to the full dataframe.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .utils import l2_normalize, log


def _budget_to_k(budget: float, n: int) -> int:
    """Convert a budget (fraction in (0, 1]) into an integer subset size."""
    if budget <= 0 or budget > 1.0:
        raise ValueError(f"budget must be in (0, 1]; got {budget}")
    k = max(1, int(math.ceil(budget * n)))
    return min(k, n)


def random_sample(indices: np.ndarray, budget: float, seed: int) -> np.ndarray:
    """Uniform-random subset of `indices`."""
    n = len(indices)
    k = _budget_to_k(budget, n)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=k, replace=False)
    chosen.sort()
    return indices[chosen]


def kmeans_diversity_sample(
    indices: np.ndarray,
    embeddings: np.ndarray,
    budget: float,
    seed: int,
    minibatch_threshold: int = 5000,
) -> np.ndarray:
    """K-means diversity sampling.

    Steps:
      1. Run k-means with n_clusters = ceil(budget * N) on ``embeddings`` of
         the candidate pool.
      2. For each cluster center, pick the candidate whose embedding is
         closest (L2) to the center.

    For large pools (>= ``minibatch_threshold``) we use ``MiniBatchKMeans``
    for speed; for smaller pools the exact ``KMeans`` is more accurate.
    Embeddings are L2-normalized so distances behave like cosine distances.
    """
    n = len(indices)
    if n != len(embeddings):
        raise ValueError(
            f"indices ({n}) and embeddings ({len(embeddings)}) must have the same length"
        )
    k = _budget_to_k(budget, n)

    if k >= n:
        log(f"budget covers entire pool (k={k} >= n={n}); returning all indices.")
        return indices.copy()

    X = l2_normalize(embeddings)

    if n >= minibatch_threshold:
        log(f"Running MiniBatchKMeans (k={k}, n={n})")
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            n_init=3,
            batch_size=min(1024, n),
            max_iter=100,
        )
    else:
        log(f"Running KMeans (k={k}, n={n})")
        km = KMeans(n_clusters=k, random_state=seed, n_init=4)
    km.fit(X)

    centers = km.cluster_centers_  # [k, D]
    # For each center, find the nearest example in X.
    # Use squared distances; argmin is identical.
    chosen_pool_positions = set()
    selected = []
    # Pre-compute distances in chunks to keep memory bounded.
    chunk = 4096
    # Distance matrix [k, n] could be heavy; do it center-by-center if huge.
    if k * n <= 8_000_000:
        # Cheaper to vectorize.
        d2 = ((X[None, :, :] - centers[:, None, :]) ** 2).sum(axis=-1)  # [k, n]
        order = np.argsort(d2, axis=1)  # nearest first
        for ci in range(k):
            for cand in order[ci]:
                if cand not in chosen_pool_positions:
                    chosen_pool_positions.add(int(cand))
                    selected.append(int(cand))
                    break
    else:
        # Memory-light path: per-center search with masking.
        mask = np.ones(n, dtype=bool)
        for ci in range(k):
            diff = X - centers[ci][None, :]
            d2 = (diff * diff).sum(axis=-1)
            d2[~mask] = np.inf
            cand = int(np.argmin(d2))
            mask[cand] = False
            selected.append(cand)

    selected_arr = np.array(sorted(selected), dtype=np.int64)
    return indices[selected_arr]


def select_subset(
    method: str,
    train_indices: np.ndarray,
    visual_embeddings: np.ndarray,
    multimodal_embeddings: Optional[np.ndarray],
    budget: float,
    seed: int,
) -> np.ndarray:
    """Dispatch to the requested subset-selection method.

    ``train_indices`` should be the integer indices of the TRAIN split into the
    full dataset (so that embeddings can be looked up via these indices).
    """
    if method == "random":
        return random_sample(train_indices, budget, seed)
    if method == "vision":
        emb_train = visual_embeddings[train_indices]
        return kmeans_diversity_sample(train_indices, emb_train, budget, seed)
    if method == "multimodal":
        if multimodal_embeddings is None:
            raise ValueError("multimodal_embeddings must be provided for method='multimodal'")
        emb_train = multimodal_embeddings[train_indices]
        return kmeans_diversity_sample(train_indices, emb_train, budget, seed)
    if method == "full":
        return train_indices.copy()
    raise ValueError(f"Unknown sampling method: {method}")
