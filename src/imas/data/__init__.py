"""Data loading, cell pairing, feature projection and gene-activity utilities."""

from .gene_activity import build_peak_gene_matrix, compute_gene_activity
from .manifests import infer_column, load_manifest, normalize_manifest_path
from .pairing import PairingResult, pair_cells
from .projection import (
    ProjectionSummary,
    dense_row_to_sparse_global,
    project_dense_npy_to_global_csr,
    project_sparse_csr_to_global,
)
from .sparse_io import (
    get_npy_shape,
    get_npz_csr_shape,
    load_csr_npz,
    save_csr_npz,
    topk_csr_per_row,
)
from .vocabulary import (
    build_global_vocab_from_node_index,
    build_local_to_global_index,
    write_vocab_txt,
)

__all__ = [
    "PairingResult",
    "ProjectionSummary",
    "build_global_vocab_from_node_index",
    "build_local_to_global_index",
    "build_peak_gene_matrix",
    "compute_gene_activity",
    "dense_row_to_sparse_global",
    "get_npy_shape",
    "get_npz_csr_shape",
    "infer_column",
    "load_csr_npz",
    "load_manifest",
    "normalize_manifest_path",
    "pair_cells",
    "project_dense_npy_to_global_csr",
    "project_sparse_csr_to_global",
    "save_csr_npz",
    "topk_csr_per_row",
    "write_vocab_txt",
]
