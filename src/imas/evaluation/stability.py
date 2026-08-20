"""Bootstrap helpers for score and top-k stability."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np


def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> np.ndarray:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    if y.size != s.size:
        raise ValueError("y_true and scores must have equal length")
    rng = np.random.default_rng(seed)
    values = np.full(int(n_bootstrap), np.nan, dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, y.size, size=y.size)
        values[i] = metric(y[idx], s[idx])
    return values


def summarize_bootstrap(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "ci_low": float(np.quantile(x, alpha / 2)),
        "ci_high": float(np.quantile(x, 1 - alpha / 2)),
    }
