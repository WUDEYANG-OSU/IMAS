from __future__ import annotations
import numpy as np
from imas.utils.stats import zscore_safe


def make_pos_neg_labels(scores, positive_quantile=0.80, negative_quantile=0.20):
    scores = np.asarray(scores, np.float32); finite = np.isfinite(scores); values = scores[finite]
    labels = np.full(scores.shape, -1, np.int8)
    if values.size < 10: return labels, labels >= 0
    labels[(scores >= np.quantile(values, positive_quantile)) & finite] = 1
    labels[(scores <= np.quantile(values, negative_quantile)) & finite] = 0
    return labels, labels >= 0


def make_pos_neg_labels_grouped(scores, groups, positive_quantile=0.80, negative_quantile=0.20, min_n=30):
    scores, groups = np.asarray(scores, np.float32), np.asarray(groups)
    labels = np.full(scores.shape, -1, np.int8); finite = np.isfinite(scores)
    for group in np.unique(groups):
        mask = (groups == group) & finite
        if mask.sum() < min_n: continue
        values = scores[mask]
        labels[mask & (scores >= np.quantile(values, positive_quantile))] = 1
        labels[mask & (scores <= np.quantile(values, negative_quantile))] = 0
    return labels, labels >= 0


def mix_tf_proxies(aucell: np.ndarray, mean_log1p: np.ndarray, alpha: float = 0.6):
    if aucell.shape != mean_log1p.shape: raise ValueError("Proxy matrices must have the same shape")
    out = np.empty_like(aucell, np.float32)
    for column in range(aucell.shape[1]):
        out[:, column] = alpha * zscore_safe(aucell[:, column]) + (1-alpha) * zscore_safe(mean_log1p[:, column])
    return out
