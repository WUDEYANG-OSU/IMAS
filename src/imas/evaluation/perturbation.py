"""Metrics for comparing virtual perturbation responses with observed trends."""
from __future__ import annotations

import numpy as np


def sign3(values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros(x.shape, dtype=np.int8)
    out[x > float(threshold)] = 1
    out[x < -float(threshold)] = -1
    return out


def trend_consistency(
    truth_delta: np.ndarray,
    predicted_delta: np.ndarray,
    truth_threshold: float = 0.0,
    prediction_threshold: float = 0.0,
) -> dict[str, float]:
    truth = np.asarray(truth_delta, dtype=float).reshape(-1)
    pred = np.asarray(predicted_delta, dtype=float).reshape(-1)
    if truth.size != pred.size:
        raise ValueError("truth_delta and predicted_delta must have equal length")
    mask = np.isfinite(truth) & np.isfinite(pred)
    t = sign3(truth[mask], truth_threshold)
    p = sign3(pred[mask], prediction_threshold)
    nonzero = t != 0
    up = t == 1
    down = t == -1
    up_recall = float(np.mean(p[up] == 1)) if up.any() else float("nan")
    down_recall = float(np.mean(p[down] == -1)) if down.any() else float("nan")
    return {
        "n_overlap": float(mask.sum()),
        "direction_accuracy_3class": float(np.mean(t == p)) if t.size else float("nan"),
        "direction_accuracy_nonzero": float(np.mean(t[nonzero] == p[nonzero])) if nonzero.any() else float("nan"),
        "truth_up_recall": up_recall,
        "truth_down_recall": down_recall,
        "balanced_accuracy_up_down": float(np.nanmean([up_recall, down_recall])),
        "predicted_nonflat_rate": float(np.mean(p != 0)) if p.size else float("nan"),
        "truth_nonflat_rate": float(np.mean(t != 0)) if t.size else float("nan"),
    }
