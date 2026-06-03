"""End-to-end (partial) fine-tuning of the visual encoder on a selected subset.

Motivation
----------
The frozen-features + logistic-regression pipeline answers "which subset gives
the most linearly separable CLIP features." Fine-tuning answers the stronger
question this project actually cares about: **which subset teaches the encoder
the most?** If multimodal sampling selects a more useful training set, the gap
between sampling methods should *widen* once the model can adapt its
representation.

Design choices (kept deliberately conservative for low-data regimes):
    - **Partial fine-tuning**: freeze the backbone, unfreeze only the last
      ``unfreeze_blocks`` transformer blocks (CLIP) / residual stages (ResNet)
      plus a fresh linear head. Full fine-tuning of an 86M-param ViT on a few
      hundred frames just overfits.
    - **Early stopping on the held-out (video-disjoint) val split**, by macro
      F1 — so low-budget runs don't overfit and the comparison stays fair.
    - **Class-weighted cross-entropy**, mirroring the ``class_weight="balanced"``
      logistic-regression baseline.

Everything except the *selected training subset* is held constant across the
sampling methods, so differences are attributable to the data selection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .embeddings import _load_clip_model, _load_resnet_model, _pick_device, _read_image
from .evaluation import compute_metrics, compute_per_class_f1
from .utils import log


@dataclass
class FineTuneConfig:
    backend: str = "clip"            # "clip" or "resnet"
    unfreeze_blocks: int = 3         # last N transformer blocks / residual stages
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    max_epochs: int = 10
    patience: int = 3                # early-stopping patience on val macro-F1
    num_workers: int = 0             # bump to 2-4 on a GPU box for faster I/O
    seed: int = 0
    device: Optional[str] = None


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


class FrameDataset(Dataset):
    """Reads frames from disk and applies the backbone's preprocess transform.

    Defined at module level (not a closure) so it is picklable by DataLoader
    workers (``num_workers > 0``).
    """

    def __init__(self, frame_paths: List[str], label_ids: np.ndarray, preprocess):
        self.paths = list(frame_paths)
        self.labels = np.asarray(label_ids, dtype=np.int64)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = _read_image(self.paths[i])
        x = self.preprocess(img)
        return x, int(self.labels[i])


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def _build_model_and_preprocess(backend: str, n_classes: int, unfreeze_blocks: int, device: str):
    """Return (model, preprocess). ``model`` outputs class logits and has only
    the last ``unfreeze_blocks`` backbone blocks + head trainable."""
    if backend == "clip":
        clip_model, preprocess, _ = _load_clip_model(device)
        embed_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, "output_dim") else 512

        # Freeze everything, then selectively unfreeze.
        for p in clip_model.parameters():
            p.requires_grad_(False)
        visual = clip_model.visual
        resblocks = visual.transformer.resblocks
        n_unfreeze = max(0, min(unfreeze_blocks, len(resblocks)))
        for blk in list(resblocks)[len(resblocks) - n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if hasattr(visual, "ln_post"):
            for p in visual.ln_post.parameters():
                p.requires_grad_(True)
        if getattr(visual, "proj", None) is not None and isinstance(visual.proj, nn.Parameter):
            visual.proj.requires_grad_(True)

        class CLIPClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = clip_model
                self.head = nn.Linear(embed_dim, n_classes)

            def forward(self, x):
                feats = self.clip.encode_image(x).float()
                return self.head(feats)

            def backbone_parameters(self):
                return (p for n, p in self.clip.named_parameters() if p.requires_grad)

        model = CLIPClassifier()

    elif backend == "resnet":
        backbone, preprocess = _load_resnet_model(device)  # fc already Identity -> 512-d
        embed_dim = 512
        for p in backbone.parameters():
            p.requires_grad_(False)
        # Unfreeze the last residual stages: layer4, then layer3, ...
        stages = [backbone.layer4, backbone.layer3, backbone.layer2, backbone.layer1]
        for stage in stages[: max(0, unfreeze_blocks)]:
            for p in stage.parameters():
                p.requires_grad_(True)

        class ResNetClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(embed_dim, n_classes)

            def forward(self, x):
                feats = self.backbone(x).float()
                return self.head(feats)

            def backbone_parameters(self):
                return (p for n, p in self.backbone.named_parameters() if p.requires_grad)

        model = ResNetClassifier()
    else:
        raise ValueError(f"Unknown backend for fine-tuning: {backend}")

    model.to(device)
    return model, preprocess


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------


def _class_weights(label_ids: np.ndarray, n_classes: int):
    counts = np.bincount(label_ids, minlength=n_classes).astype(np.float64)
    # sklearn "balanced": n_samples / (n_classes * count_c); guard zero counts.
    counts[counts == 0] = 1.0
    w = label_ids.shape[0] / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _predict_ids(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def finetune_and_eval(
    train_paths: List[str],
    train_labels: List[str],
    val_paths: List[str],
    val_labels: List[str],
    test_paths: List[str],
    test_labels: List[str],
    label_list: List[str],
    cfg: FineTuneConfig,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Partial fine-tune on (train_*), early-stop on (val_*), evaluate on (test_*).

    Returns ``(metrics, per_class_f1, info)`` where ``info`` reports the best
    epoch and best val macro-F1.
    """
    device = cfg.device or _pick_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    label_to_id = {lab: i for i, lab in enumerate(label_list)}
    n_classes = len(label_list)

    y_tr = np.array([label_to_id[l] for l in train_labels], dtype=np.int64)
    y_val = np.array([label_to_id[l] for l in val_labels], dtype=np.int64)
    y_te = np.array([label_to_id[l] for l in test_labels], dtype=np.int64)

    model, preprocess = _build_model_and_preprocess(
        cfg.backend, n_classes, cfg.unfreeze_blocks, device
    )

    pin = device == "cuda"
    train_loader = DataLoader(
        FrameDataset(train_paths, y_tr, preprocess),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        FrameDataset(val_paths, y_val, preprocess),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        FrameDataset(test_paths, y_te, preprocess),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )

    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_classes).to(device))
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": cfg.backbone_lr},
            {"params": model.head.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    from sklearn.metrics import f1_score

    best_val = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.size(0)
        train_loss = running / max(1, len(train_loader.dataset))

        val_pred = _predict_ids(model, val_loader, device)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        log(
            f"    [ft] epoch {epoch + 1}/{cfg.max_epochs} "
            f"train_loss={train_loss:.4f} val_macroF1={val_f1:.4f}"
        )

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                log(f"    [ft] early stop at epoch {epoch + 1} (best epoch {best_epoch}).")
                break

    model.load_state_dict(best_state)
    test_pred_ids = _predict_ids(model, test_loader, device)

    id_to_label = {i: lab for lab, i in label_to_id.items()}
    y_pred = np.array([id_to_label[i] for i in test_pred_ids])
    y_true = np.array(test_labels)

    metrics = compute_metrics(y_true, y_pred)
    per_class = compute_per_class_f1(y_true, y_pred, label_list)
    info = {"best_epoch": best_epoch, "best_val_macro_f1": best_val}

    # Free GPU memory between runs.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return metrics, per_class, info
