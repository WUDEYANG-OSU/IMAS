from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .sparse_io import load_csr_npz, save_csr_npz, topk_csr_per_row


@dataclass(frozen=True)
class ProjectionSummary:
    n_rows: int
    local_dim: int
    global_dim: int
    mapped_local_features: int
    unique_global_features: int
    duplicate_global_mappings: int
    nnz: int

    @property
    def mapping_coverage(self) -> float:
        return self.mapped_local_features / max(1, self.local_dim)


def _mapping_summary(local_to_global: np.ndarray, global_dim: int, n_rows: int, nnz: int) -> ProjectionSummary:
    mapping = np.asarray(local_to_global, dtype=np.int64)
    valid = mapping[(mapping >= 0) & (mapping < int(global_dim))]
    unique = np.unique(valid)
    return ProjectionSummary(
        n_rows=int(n_rows),
        local_dim=int(mapping.size),
        global_dim=int(global_dim),
        mapped_local_features=int(valid.size),
        unique_global_features=int(unique.size),
        duplicate_global_mappings=int(valid.size - unique.size),
        nnz=int(nnz),
    )


def dense_row_to_sparse_global(
    row: np.ndarray,
    local_to_global: np.ndarray,
    global_dim: int,
) -> sparse.csr_matrix:
    """Project one dense local row into a one-row global CSR matrix."""
    values = np.asarray(row).reshape(-1)
    mapping = np.asarray(local_to_global, dtype=np.int64)
    if values.size != mapping.size:
        raise ValueError(f"row has {values.size} features but mapping has {mapping.size}")
    keep = (mapping >= 0) & (mapping < int(global_dim)) & np.isfinite(values) & (values != 0)
    if not np.any(keep):
        return sparse.csr_matrix((1, int(global_dim)), dtype=np.float32)
    projected = sparse.coo_matrix(
        (values[keep].astype(np.float32), (np.zeros(int(keep.sum())), mapping[keep])),
        shape=(1, int(global_dim)),
    ).tocsr()
    projected.sum_duplicates()
    return projected


def project_dense_npy_to_global_csr(
    npy_path: str | Path,
    local_to_global: np.ndarray,
    global_dim: int,
    *,
    topk: int | None = None,
    row_chunk_size: int = 1024,
    output_path: str | Path | None = None,
) -> tuple[sparse.csr_matrix, ProjectionSummary]:
    """Project a dense memmapped matrix into the shared global feature space."""
    matrix = np.load(Path(npy_path), mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional array, got {matrix.shape}")
    mapping = np.asarray(local_to_global, dtype=np.int64)
    if matrix.shape[1] != mapping.size:
        raise ValueError(
            f"matrix has {matrix.shape[1]} columns but mapping contains {mapping.size} entries"
        )
    valid_local = np.flatnonzero((mapping >= 0) & (mapping < int(global_dim)))
    valid_global = mapping[valid_local]
    chunks: list[sparse.csr_matrix] = []
    for start in range(0, matrix.shape[0], int(row_chunk_size)):
        stop = min(matrix.shape[0], start + int(row_chunk_size))
        block = np.asarray(matrix[start:stop, valid_local], dtype=np.float32)
        rows, local_cols = np.nonzero(np.isfinite(block) & (block != 0))
        values = block[rows, local_cols]
        projected = sparse.coo_matrix(
            (values, (rows, valid_global[local_cols])),
            shape=(stop - start, int(global_dim)),
        ).tocsr()
        projected.sum_duplicates()
        if topk is not None:
            projected = topk_csr_per_row(projected, topk)
        chunks.append(projected)
    result = sparse.vstack(chunks, format="csr") if chunks else sparse.csr_matrix((0, int(global_dim)))
    result.sum_duplicates()
    result.sort_indices()
    if output_path is not None:
        save_csr_npz(output_path, result)
    return result, _mapping_summary(mapping, global_dim, result.shape[0], result.nnz)


def project_sparse_csr_to_global(
    matrix_or_path: sparse.spmatrix | str | Path,
    local_to_global: np.ndarray,
    global_dim: int,
    *,
    topk: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[sparse.csr_matrix, ProjectionSummary]:
    """Project a local CSR matrix into a global feature space and merge duplicates."""
    local = (
        load_csr_npz(matrix_or_path)
        if isinstance(matrix_or_path, (str, Path))
        else matrix_or_path.tocsr()
    )
    mapping = np.asarray(local_to_global, dtype=np.int64)
    if local.shape[1] != mapping.size:
        raise ValueError(
            f"matrix has {local.shape[1]} columns but mapping contains {mapping.size} entries"
        )
    valid_local = np.flatnonzero((mapping >= 0) & (mapping < int(global_dim)))
    if valid_local.size == 0:
        result = sparse.csr_matrix((local.shape[0], int(global_dim)), dtype=np.float32)
    else:
        selection = sparse.coo_matrix(
            (
                np.ones(valid_local.size, dtype=np.float32),
                (valid_local, mapping[valid_local]),
            ),
            shape=(local.shape[1], int(global_dim)),
        ).tocsr()
        result = (local[:, valid_local] @ selection[valid_local, :]).tocsr()
        result.sum_duplicates()
        if topk is not None:
            result = topk_csr_per_row(result, topk)
        result.sort_indices()
    if output_path is not None:
        save_csr_npz(output_path, result)
    return result, _mapping_summary(mapping, global_dim, result.shape[0], result.nnz)


project_atac_csr_to_global_csr = project_sparse_csr_to_global
