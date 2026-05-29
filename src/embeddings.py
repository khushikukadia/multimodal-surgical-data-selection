"""Visual and text embedding extraction.

Visual backends:
    - "clip"   : open_clip ViT-B/32 (preferred; provides aligned text encoder).
    - "resnet" : torchvision ResNet18, ImageNet-pretrained (fallback).

Text backends (selected automatically based on visual backend):
    - "clip"   : CLIP text encoder (aligned with visual features).
    - "tfidf"  : sklearn TF-IDF over phase label tokens (fallback).

All embeddings are cached as .npz on disk and L2-normalized at the end.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import ensure_dir, l2_normalize, log

# Defer heavy torch / open_clip imports to inside functions so the module
# imports cleanly even when torch is not installed (e.g. for the unit-style
# smoke test that uses random fallback features).


PHASE_PROMPT_TEMPLATE = "A laparoscopic surgery frame showing the phase: {phase}"


# -----------------------------------------------------------------------------
# Visual embeddings
# -----------------------------------------------------------------------------


def _load_clip_model(device: str):
    import open_clip  # type: ignore
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    return model, preprocess, tokenizer


def _load_resnet_model(device: str):
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    model = tvm.resnet18(weights=weights)
    # Drop the classification head -> pooled 512-d feature.
    model.fc = torch.nn.Identity()  # type: ignore[assignment]
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess


def _read_image(path: str):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return img


def _embed_with_clip(
    frame_paths: List[str], device: str, batch_size: int = 32
) -> np.ndarray:
    import torch

    model, preprocess, _ = _load_clip_model(device)
    features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(frame_paths), batch_size), desc="CLIP image"):
            batch_paths = frame_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                img = _read_image(p)
                tensors.append(preprocess(img))
            batch = torch.stack(tensors, dim=0).to(device)
            feats = model.encode_image(batch).float().cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)


def _embed_with_resnet(
    frame_paths: List[str], device: str, batch_size: int = 32
) -> np.ndarray:
    import torch

    model, preprocess = _load_resnet_model(device)
    features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(frame_paths), batch_size), desc="ResNet image"):
            batch_paths = frame_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                img = _read_image(p)
                tensors.append(preprocess(img))
            batch = torch.stack(tensors, dim=0).to(device)
            feats = model(batch).float().cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)


def extract_visual_embeddings(
    df: pd.DataFrame,
    backend: str,
    cache_path: str,
    force_recompute: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return L2-normalized visual embeddings of shape [N, D].

    Embeddings are aligned with ``df`` row order.
    """
    if device is None:
        device = _pick_device()

    if os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = cached["embeddings"]
        cached_paths = cached["frame_paths"].tolist()
        if cached_paths == df["frame_path"].tolist():
            log(f"Loaded cached visual embeddings ({embeddings.shape}) from {cache_path}.")
            return l2_normalize(embeddings)
        log("Cached visual embeddings do not match current df; recomputing.")

    log(f"Extracting visual embeddings with backend={backend} on device={device}...")
    paths = df["frame_path"].tolist()

    if backend == "clip":
        try:
            embeddings = _embed_with_clip(paths, device=device)
        except Exception as e:
            log(
                f"CLIP backend failed ({e}). Falling back to ResNet18. "
                "Install open-clip-torch to use CLIP."
            )
            embeddings = _embed_with_resnet(paths, device=device)
            backend = "resnet"
    elif backend == "resnet":
        embeddings = _embed_with_resnet(paths, device=device)
    else:
        raise ValueError(f"Unknown visual backend: {backend}")

    ensure_dir(os.path.dirname(cache_path) or ".")
    np.savez(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        frame_paths=np.array(paths),
        backend=backend,
    )
    log(f"Saved visual embeddings to {cache_path} (shape={embeddings.shape}).")
    return l2_normalize(embeddings)


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        # Apple silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


# -----------------------------------------------------------------------------
# Text embeddings
# -----------------------------------------------------------------------------


def _embed_text_with_clip(prompts: List[str], device: str) -> np.ndarray:
    import torch

    model, _, tokenizer = _load_clip_model(device)
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens).float().cpu().numpy()
    return feats


def _embed_text_with_tfidf(prompts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    X = vec.fit_transform(prompts).toarray()
    return X.astype(np.float32)


def extract_text_embeddings(
    unique_phases: List[str],
    backend: str,
    cache_path: str,
    force_recompute: bool = False,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Return {phase_label: L2-normalized embedding}.

    ``backend`` is the visual backend; when "clip" we use the matching CLIP
    text encoder, otherwise we fall back to TF-IDF over the prompts.
    """
    if device is None:
        device = _pick_device()

    unique_phases = sorted(set(unique_phases))
    prompts = [PHASE_PROMPT_TEMPLATE.format(phase=p) for p in unique_phases]

    if os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path, allow_pickle=True)
        cached_phases = cached["phases"].tolist()
        if cached_phases == unique_phases:
            log(f"Loaded cached text embeddings from {cache_path}.")
            embs = cached["embeddings"]
            return {p: l2_normalize(embs[i : i + 1])[0] for i, p in enumerate(unique_phases)}
        log("Cached text embeddings do not match phases; recomputing.")

    text_backend = "clip" if backend == "clip" else "tfidf"
    log(f"Extracting text embeddings with backend={text_backend} for {len(unique_phases)} phases.")

    if text_backend == "clip":
        try:
            embeddings = _embed_text_with_clip(prompts, device=device)
        except Exception as e:
            log(f"CLIP text encoder failed ({e}); falling back to TF-IDF.")
            embeddings = _embed_text_with_tfidf(prompts)
    else:
        embeddings = _embed_text_with_tfidf(prompts)

    ensure_dir(os.path.dirname(cache_path) or ".")
    np.savez(
        cache_path,
        embeddings=embeddings.astype(np.float32),
        phases=np.array(unique_phases),
        backend=text_backend,
    )
    log(f"Saved text embeddings to {cache_path} (shape={embeddings.shape}).")
    return {p: l2_normalize(embeddings[i : i + 1])[0] for i, p in enumerate(unique_phases)}


# -----------------------------------------------------------------------------
# Multimodal embeddings
# -----------------------------------------------------------------------------


def build_multimodal_embeddings(
    visual_embeddings: np.ndarray,
    labels: List[str],
    text_embedding_dict: Dict[str, np.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Concatenate visual and text embeddings per row.

    Both halves are L2-normalized; the visual/text halves are scaled by
    ``alpha``/``beta`` so the relative weighting can be tuned.
    """
    visual = l2_normalize(visual_embeddings) * float(alpha)

    # Stack the text embedding for each row's phase label.
    text_dim = next(iter(text_embedding_dict.values())).shape[-1]
    text = np.zeros((len(labels), text_dim), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab not in text_embedding_dict:
            raise KeyError(
                f"Phase label '{lab}' has no text embedding. "
                "Did you call extract_text_embeddings with all unique phases?"
            )
        text[i] = text_embedding_dict[lab]
    text = l2_normalize(text) * float(beta)

    multimodal = np.concatenate([visual, text], axis=1).astype(np.float32)
    return multimodal


# -----------------------------------------------------------------------------
# Convenience: random-fallback embeddings (used by tests / smoke test if the
# real backends are unavailable). Not used by the main run by default.
# -----------------------------------------------------------------------------


def random_visual_fallback(
    df: pd.DataFrame, dim: int = 64, seed: int = 0
) -> np.ndarray:
    """Deterministic pseudo-random visual features for smoke tests."""
    rng = np.random.default_rng(seed)
    # Encode a weak signal of the phase so K-means clustering is meaningful.
    phases = df["phase"].astype(str).tolist()
    phase_to_id = {p: i for i, p in enumerate(sorted(set(phases)))}
    centers = rng.standard_normal((len(phase_to_id), dim)).astype(np.float32) * 2.0
    noise = rng.standard_normal((len(df), dim)).astype(np.float32) * 0.5
    out = np.array([centers[phase_to_id[p]] for p in phases], dtype=np.float32) + noise
    return out
