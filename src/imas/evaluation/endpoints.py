"""Unified endpoint evaluation used by stage-wise IMAS ablations."""
from __future__ import annotations

import numpy as np
from .classification import evaluate_binary
from .ranking import retrieval_metrics
from .calibration import brier_score, expected_calibration_error


def minmax_probability(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    mask = np.isfinite(x)
    if not mask.any():
        return out
    lo, hi = float(np.min(x[mask])), float(np.max(x[mask]))
    out[mask] = 0.5 if hi <= lo else (x[mask] - lo) / (hi - lo)
    return out


def evaluate_endpoint(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    binary = evaluate_binary(y_true, scores)
    probability = minmax_probability(scores)
    result = {
        "n": float(binary.n),
        "n_positive": float(binary.n_positive),
        "n_negative": float(binary.n_negative),
        "auroc": binary.auroc,
        "auprc": binary.auprc,
        "brier": brier_score(y_true, probability),
        "ece": expected_calibration_error(y_true, probability),
    }
    result.update(retrieval_metrics(y_true, scores, k=k))
    return result
