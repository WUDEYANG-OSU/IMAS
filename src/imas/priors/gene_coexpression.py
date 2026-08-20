from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def select_genes_by_variance(
    expression: np.ndarray,
    *,
    max_genes: int | None = None,
    min_detection_fraction: float = 0.0,
) -> np.ndarray:
    matrix = np.asarray(expression, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("expression must be cells×genes")
    detection = np.mean(matrix != 0, axis=0)
    variance = np.nanvar(matrix, axis=0)
    valid = np.flatnonzero(np.isfinite(variance) & (detection >= min_detection_fraction))
    if max_genes is not None and valid.size > max_genes:
        order = np.lexsort((valid, -variance[valid]))[: int(max_genes)]
        valid = valid[order]
    return valid.astype(np.int64)


def compute_topk_correlation_edges(
    expression: np.ndarray,
    gene_names: Sequence[object],
    *,
    topk: int = 32,
    min_abs_correlation: float = 0.1,
    include_negative: bool = True,
    block_size: int = 512,
) -> pd.DataFrame:
    """Compute deterministic top-k gene correlations without materialising full G×G."""
    x = np.asarray(expression, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(gene_names):
        raise ValueError("expression must be cells×genes and match gene_names")
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    valid = np.isfinite(std[0]) & (std[0] > 1e-8)
    z = np.zeros_like(x, dtype=np.float32)
    z[:, valid] = (x[:, valid] - mean[:, valid]) / std[:, valid]
    n = max(1, z.shape[0] - 1)
    records: list[tuple[str, str, float]] = []
    for start in range(0, z.shape[1], int(block_size)):
        stop = min(z.shape[1], start + int(block_size))
        corr = (z[:, start:stop].T @ z) / float(n)
        for local_row, src in enumerate(range(start, stop)):
            values = corr[local_row].astype(np.float64, copy=True)
            values[src] = np.nan
            scores = np.abs(values) if include_negative else values
            candidate = np.flatnonzero(np.isfinite(scores) & (scores >= min_abs_correlation))
            if candidate.size > topk:
                selected = candidate[np.argpartition(scores[candidate], -topk)[-topk:]]
            else:
                selected = candidate
            selected = selected[np.lexsort((selected, -scores[selected]))]
            for dst in selected:
                records.append((f"Gene:{gene_names[src]}", f"Gene:{gene_names[dst]}", float(values[dst])))
    return pd.DataFrame(records, columns=["src_name", "dst_name", "weight"])
