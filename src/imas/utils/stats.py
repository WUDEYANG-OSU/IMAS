from __future__ import annotations
import numpy as np
from scipy import stats


def zscore_safe(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / (std + eps)).astype(np.float32)


def minmax01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return out
    lo, hi = float(np.nanmin(x[finite])), float(np.nanmax(x[finite]))
    if hi - lo < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def rank01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() <= 1:
        out[finite] = 0.5
        return out
    ranks = stats.rankdata(x[finite], method="average")
    out[finite] = (ranks - 1.0) / max(1, finite.sum() - 1)
    return out


def robust01(values: np.ndarray, qlo: float = 1.0, qhi: float = 99.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return out
    lo = float(np.nanpercentile(x[finite], qlo))
    hi = float(np.nanpercentile(x[finite], qhi))
    if hi - lo < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = np.clip((x[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(pvalues, dtype=float), 1e-300, 1.0)
    if p.size == 0:
        return p
    order = np.argsort(p)
    adjusted = p[order] * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out
