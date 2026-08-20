from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy import sparse

from .projection import ProjectionSummary


def validate_local_to_global_mapping(mapping: np.ndarray, global_dim: int) -> dict[str, float | int]:
    values = np.asarray(mapping, dtype=np.int64)
    invalid_positive = int(np.sum(values >= int(global_dim)))
    valid = values[(values >= 0) & (values < int(global_dim))]
    unique = np.unique(valid)
    return {
        "local_dim": int(values.size),
        "mapped": int(valid.size),
        "unmapped": int(np.sum(values < 0)),
        "out_of_range": invalid_positive,
        "unique_global": int(unique.size),
        "duplicate_mappings": int(valid.size - unique.size),
        "coverage": float(valid.size / max(1, values.size)),
    }


def validate_projected_matrix(
    matrix: sparse.spmatrix,
    expected_global_dim: int,
    *,
    require_finite: bool = True,
) -> dict[str, float | int]:
    csr = matrix.tocsr()
    if csr.shape[1] != int(expected_global_dim):
        raise ValueError(
            f"Projected matrix has {csr.shape[1]} columns; expected {expected_global_dim}"
        )
    if require_finite and not np.isfinite(csr.data).all():
        raise ValueError("Projected matrix contains non-finite values")
    return {
        "n_rows": int(csr.shape[0]),
        "n_cols": int(csr.shape[1]),
        "nnz": int(csr.nnz),
        "density": float(csr.nnz / max(1, csr.shape[0] * csr.shape[1])),
    }
