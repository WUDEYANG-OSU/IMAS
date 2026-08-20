"""Task-aligned virtual-perturbation controls used by the IMAS repository.

These functions are deliberately named ``*_style``. They are compact controls
inspired by masking, latent-effect, and graph-propagation principles; they are
not drop-in reproductions of the official scGPT, CPA, or GEARS software.
"""
from __future__ import annotations

import numpy as np


def scgpt_style_masking(expression: np.ndarray, target_index: int, mask_value: float = 0.0) -> np.ndarray:
    x = np.asarray(expression, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expression must be cells x genes")
    if not 0 <= int(target_index) < x.shape[1]:
        raise IndexError("target_index out of range")
    out = x.copy()
    out[:, int(target_index)] = float(mask_value)
    return out


def cpa_style_shared_effect(
    expression: np.ndarray,
    target_index: int,
    effect_scale: float = 0.2,
    clip_quantile: float = 0.95,
) -> np.ndarray:
    """Apply one population-derived target association vector to every cell."""
    x = np.asarray(expression, dtype=np.float32)
    target = x[:, int(target_index)].astype(float)
    centered = x.astype(float) - x.mean(axis=0, keepdims=True)
    t = target - target.mean()
    denom = float(np.dot(t, t)) + 1e-12
    effect = np.dot(t, centered) / denom
    clip = float(np.quantile(np.abs(effect), clip_quantile)) if effect.size else 0.0
    if clip > 0:
        effect = np.clip(effect, -clip, clip)
    strength = float(np.quantile(target, 0.75))
    out = x - float(effect_scale) * strength * effect[None, :]
    out[:, int(target_index)] = 0.0
    return np.clip(out, 0.0, None).astype(np.float32)


def gears_style_graph_propagation(
    expression: np.ndarray,
    target_index: int,
    adjacency: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    """Propagate a target KO through a fixed gene-gene adjacency matrix."""
    x = np.asarray(expression, dtype=np.float32)
    a = np.asarray(adjacency, dtype=np.float32)
    if a.shape != (x.shape[1], x.shape[1]):
        raise ValueError("adjacency must be genes x genes")
    row_sum = np.sum(np.abs(a), axis=1, keepdims=True)
    norm = a / np.maximum(row_sum, 1e-8)
    target_strength = float(np.quantile(x[:, int(target_index)], 0.75))
    seed = np.zeros(x.shape[1], dtype=np.float32)
    seed[int(target_index)] = -target_strength
    propagated = seed + float(alpha) * (seed @ norm)
    out = x + propagated[None, :]
    out[:, int(target_index)] = 0.0
    return np.clip(out, 0.0, None).astype(np.float32)
