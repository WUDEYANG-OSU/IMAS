"""Correlation utilities with tie-aware ranks."""
from __future__ import annotations

import numpy as np


def rank_average_ties(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        out[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("x and y must have equal length")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("x and y must have equal length")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return pearson(rank_average_ties(a[mask]), rank_average_ties(b[mask]))
