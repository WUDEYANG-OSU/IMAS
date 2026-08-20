"""Binary classification metrics used across IMAS benchmark workflows.

Implementations are NumPy-only so the core package can evaluate cached scores
without importing scikit-learn. Inputs are filtered to finite scores and labels
in {0, 1}.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BinaryMetricResult:
    n: int
    n_positive: int
    n_negative: int
    auroc: float
    auprc: float


def _clean(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have equal length")
    mask = np.isfinite(s) & np.isfinite(y) & np.isin(y, [0, 1])
    return y[mask].astype(np.int8), s[mask]


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _clean(y_true, scores)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    distinct = np.r_[True, s[1:] != s[:-1]]
    tp = np.cumsum(y == 1)[distinct]
    fp = np.cumsum(y == 0)[distinct]
    thresholds = s[distinct]
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def precision_recall_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _clean(y_true, scores)
    positives = int((y == 1).sum())
    if positives == 0:
        return np.array([0.0]), np.array([0.0]), np.array([])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    distinct_end = np.r_[s[1:] != s[:-1], True]
    tp = np.cumsum(y == 1)[distinct_end].astype(float)
    fp = np.cumsum(y == 0)[distinct_end].astype(float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    thresholds = s[distinct_end]
    return precision, recall, thresholds


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    y, _ = _clean(y_true, scores)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(np.trapezoid(tpr, fpr))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    y, _ = _clean(y_true, scores)
    if (y == 1).sum() == 0:
        return float("nan")
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def evaluate_binary(y_true: np.ndarray, scores: np.ndarray) -> BinaryMetricResult:
    y, s = _clean(y_true, scores)
    return BinaryMetricResult(
        n=int(y.size),
        n_positive=int((y == 1).sum()),
        n_negative=int((y == 0).sum()),
        auroc=auroc(y, s),
        auprc=average_precision(y, s),
    )
