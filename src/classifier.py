"""Lightweight classifiers used to score each selected training subset.

The classifier always sees visual embeddings only. This is intentional: the
sampling method is what differs between conditions, so the classifier itself
should be held constant and should not get extra language information.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .utils import log


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a multinomial logistic regression on visual embeddings."""
    n_classes = len(np.unique(y_train))
    log(
        f"Training LogisticRegression on X_train shape={X_train.shape}, "
        f"{n_classes} unique classes in subset."
    )
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=None,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
    hidden_dim: int = 256,
    max_iter: int = 200,
) -> MLPClassifier:
    """Optional small MLP classifier. Slower than LR; keep for ablations."""
    log(f"Training MLPClassifier(hidden={hidden_dim}) on X_train shape={X_train.shape}.")
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        max_iter=max_iter,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=10,
    )
    clf.fit(X_train, y_train)
    return clf


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)


def train_classifier(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 0,
) -> Tuple[object, str]:
    if classifier_name == "logreg":
        return train_logistic_regression(X_train, y_train, seed=seed), "logreg"
    if classifier_name == "mlp":
        return train_mlp(X_train, y_train, seed=seed), "mlp"
    raise ValueError(f"Unknown classifier: {classifier_name}")
