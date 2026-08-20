"""Probability calibration metrics."""
from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0, 1])
    if not mask.any():
        return float("nan")
    return float(np.mean((np.clip(p[mask], 0.0, 1.0) - y[mask]) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0, 1])
    y, p = y[mask], np.clip(p[mask], 0.0, 1.0)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_id = np.minimum(np.digitize(p, edges[1:-1]), int(n_bins) - 1)
    ece = 0.0
    for i in range(int(n_bins)):
        m = bin_id == i
        if m.any():
            ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)
