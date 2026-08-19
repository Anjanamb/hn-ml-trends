"""Evaluation metrics for classification and clustering.

Deliberately implemented from arithmetic where possible so the
notebooks can walk the reader through what each number measures.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction of predictions equal to the truth."""
    y_true = list(y_true); y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        return float("nan")
    n_correct = sum(a == b for a, b in zip(y_true, y_pred))
    return n_correct / len(y_true)


def per_class_report(
    y_true: Sequence[str], y_pred: Sequence[str],
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Precision / recall / F1 per class, dependency-free."""
    y_true = list(y_true); y_pred = list(y_pred)
    if categories is None:
        categories = sorted(set(y_true) | set(y_pred))

    rows = []
    for c in categories:
        tp = sum((a == c) and (b == c) for a, b in zip(y_true, y_pred))
        fp = sum((a != c) and (b == c) for a, b in zip(y_true, y_pred))
        fn = sum((a == c) and (b != c) for a, b in zip(y_true, y_pred))
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / support if support else float("nan")
        if precision != precision or recall != recall or (precision + recall) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall)
        rows.append({
            "category": c, "support": support,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows).set_index("category")


def cluster_purity(
    labels: Sequence[str], clusters: Sequence[int],
) -> float:
    """Fraction of points assigned to the majority label of their cluster.

    Cluster label -1 (HDBSCAN noise) is excluded from the calculation.
    """
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    mask = clusters >= 0
    if not mask.any():
        return float("nan")
    labels, clusters = labels[mask], clusters[mask]
    correct = 0
    for c in np.unique(clusters):
        in_c = labels[clusters == c]
        _, counts = np.unique(in_c, return_counts=True)
        correct += counts.max()
    return correct / len(labels)
