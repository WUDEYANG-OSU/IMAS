from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


def get_npy_shape(path: str | Path) -> tuple[int, ...]:
    """Return a NumPy array shape without loading the full array."""
    arr = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    return tuple(int(x) for x in arr.shape)


def get_npz_csr_shape(path: str | Path) -> tuple[int, int]:
    """Read the shape of an IMAS CSR ``.npz`` file."""
    with np.load(Path(path), allow_pickle=False) as payload:
        if "shape" in payload.files:
            shape = tuple(int(x) for x in payload["shape"].tolist())
            if len(shape) != 2:
                raise ValueError(f"Expected a 2-D CSR shape, got {shape}")
            return shape
        if "indptr" not in payload.files or "indices" not in payload.files:
            raise KeyError(f"{path}: missing CSR shape and index arrays")
        n_rows = int(len(payload["indptr"]) - 1)
        n_cols = int(payload["indices"].max()) + 1 if payload["indices"].size else 0
        return n_rows, n_cols


def save_csr_npz(
    path: str | Path,
    matrix: sparse.spmatrix,
    *,
    compressed: bool = True,
) -> Path:
    """Save a sparse matrix using the stable IMAS CSR interchange schema."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    writer = np.savez_compressed if compressed else np.savez
    writer(
        out,
        data=csr.data.astype(np.float32, copy=False),
        indices=csr.indices.astype(np.int32, copy=False),
        indptr=csr.indptr.astype(np.int64, copy=False),
        shape=np.asarray(csr.shape, dtype=np.int64),
    )
    return out


def load_csr_npz(path: str | Path) -> sparse.csr_matrix:
    """Load either an IMAS CSR archive or a SciPy ``save_npz`` archive."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"data", "indices", "indptr", "shape"}
            if required.issubset(payload.files):
                shape = tuple(int(x) for x in payload["shape"].tolist())
                matrix = sparse.csr_matrix(
                    (payload["data"], payload["indices"], payload["indptr"]),
                    shape=shape,
                )
                matrix.sum_duplicates()
                matrix.sort_indices()
                return matrix
    except (ValueError, OSError):
        pass
    matrix = sparse.load_npz(path).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def topk_csr_per_row(
    matrix: sparse.spmatrix,
    k: int | None,
    *,
    by_absolute_value: bool = False,
) -> sparse.csr_matrix:
    """Retain at most ``k`` entries per row using deterministic descending order."""
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    if k is None:
        csr.sort_indices()
        return csr
    if k < 1:
        return sparse.csr_matrix(csr.shape, dtype=csr.dtype)

    data_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    indptr = [0]
    for row in range(csr.shape[0]):
        start, stop = int(csr.indptr[row]), int(csr.indptr[row + 1])
        idx = csr.indices[start:stop]
        values = csr.data[start:stop]
        if values.size > k:
            scores = np.abs(values) if by_absolute_value else values
            selected = np.argpartition(scores, -k)[-k:]
            idx = idx[selected]
            values = values[selected]
            scores = scores[selected]
            order = np.lexsort((idx, -scores))
            idx = idx[order]
            values = values[order]
        elif values.size:
            scores = np.abs(values) if by_absolute_value else values
            order = np.lexsort((idx, -scores))
            idx = idx[order]
            values = values[order]
        data_parts.append(np.asarray(values))
        index_parts.append(np.asarray(idx, dtype=np.int32))
        indptr.append(indptr[-1] + len(values))

    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=csr.dtype)
    indices = (
        np.concatenate(index_parts).astype(np.int32, copy=False)
        if index_parts
        else np.empty(0, dtype=np.int32)
    )
    out = sparse.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=csr.shape,
    )
    out.sum_duplicates()
    return out
