from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import sparse


def build_peak_gene_matrix(
    peak_indices: np.ndarray,
    gene_indices: np.ndarray,
    weights: np.ndarray | None,
    *,
    n_peaks: int,
    n_genes: int,
    aggregate_duplicates: bool = True,
) -> sparse.csr_matrix:
    """Build a sparse peak-to-gene prior matrix from indexed edges."""
    peak_indices = np.asarray(peak_indices, dtype=np.int64)
    gene_indices = np.asarray(gene_indices, dtype=np.int64)
    if peak_indices.shape != gene_indices.shape:
        raise ValueError("peak_indices and gene_indices must have identical shapes")
    data = (
        np.ones(peak_indices.size, dtype=np.float32)
        if weights is None
        else np.asarray(weights, dtype=np.float32)
    )
    if data.shape != peak_indices.shape:
        raise ValueError("weights must match edge indices")
    valid = (
        (peak_indices >= 0)
        & (peak_indices < int(n_peaks))
        & (gene_indices >= 0)
        & (gene_indices < int(n_genes))
        & np.isfinite(data)
    )
    matrix = sparse.coo_matrix(
        (data[valid], (peak_indices[valid], gene_indices[valid])),
        shape=(int(n_peaks), int(n_genes)),
    ).tocsr()
    if aggregate_duplicates:
        matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def compute_gene_activity(
    atac: sparse.spmatrix,
    peak_gene: sparse.spmatrix,
    *,
    log1p: bool = False,
    binary_atac: bool = False,
    dtype=np.float32,
) -> sparse.csr_matrix:
    """Aggregate accessible peaks into gene-activity scores using a peak-gene prior."""
    atac_csr = atac.tocsr(copy=True)
    prior = peak_gene.tocsr()
    if atac_csr.shape[1] != prior.shape[0]:
        raise ValueError(
            f"ATAC has {atac_csr.shape[1]} peaks but prior has {prior.shape[0]} peak rows"
        )
    if binary_atac:
        atac_csr.data = np.ones_like(atac_csr.data, dtype=np.float32)
    activity = (atac_csr @ prior).tocsr().astype(dtype)
    activity.sum_duplicates()
    if log1p:
        activity.data = np.log1p(np.maximum(activity.data, 0)).astype(dtype, copy=False)
    activity.sort_indices()
    return activity
