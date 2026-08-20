from __future__ import annotations

# =============================================================================
# 11_integrated_UMAP_native_vs_MLTD_mapped_8samples_no_sample7.py
# =============================================================================
# Purpose
# -------
# Build two directly comparable joint UMAPs for the same eight modeled kidney
# cancer samples (Sample-1–6, Sample-8 and Sample-9):
#
# A) Native-expression UMAP
#    - based only on the measured RNA expression matrix;
#    - no expression replacement, imputation or MLTD-derived value is written
#      into the native count matrix.
#
# B) MLTD-mapped UMAP
#    - keeps the native RNA PCA as an anchor;
#    - adds Stage-B adjusted cell representation;
#    - adds cell-mapped sender / receiver / TF-response features;
#    - adds communication-pseudotime features;
#    - constructs a new dependency-aware representation and neighbour graph.
#
# Therefore, the second UMAP is a representation-level mapping of MLTD effects,
# not a modified or imputed expression matrix.
#
# Main outputs
# ------------
# - native_vs_MLTD_mapped_cell_coordinates.tsv
# - MLTD_mapped_representation.npy
# - native_vs_MLTD_mapped_lightweight.h5ad
# - UMAP_native_vs_MLTD_by_sample.png/.svg
# - UMAP_native_vs_MLTD_by_cluster.png/.svg
# - UMAP_MLTD_mapping_displacement.png/.svg
# - MLTD_mapping_displacement_by_sample_cluster.tsv
# - cluster_top1000_marker_genes_for_annotation.xlsx
# - run_summary.json
# =============================================================================

import gc
import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.linalg import orthogonal_procrustes

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import anndata as ad
    import scanpy as sc
except ImportError as exc:
    raise ImportError(
        "Missing anndata/scanpy. In the active WSL Jupyter kernel run:\n"
        "%pip install -U anndata scanpy igraph leidenalg scikit-misc"
    ) from exc

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIG
# =============================================================================

BASE = get_project_root()

RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "stageB_RNA_only_Kidney_cancer_8samples_exclude_sample7_joint"
)
DATASET_NAME = "scVerifyKC-Kidney-cancer-8samples-exclude-sample7-joint"
RUN_STAGEB_DATASET_DIR = RUN_STAGEB_DIR / DATASET_NAME

# The completed no-Sample-7 Stage-B manifest is the sole authority for modeled
# cells. The source count matrix may still contain Sample-7 rows; only manifest
# row_index values are used.
INPUT_MANIFEST = RUN_STAGEB_DIR / "input_manifest_excluding_sample7.tsv"

CCI_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_cci_jointIntegratedUMAP_with_sample_labels"
)
RECLUSTER_DIR = CCI_DIR / "recluster"
PT_DIR = CCI_DIR / "communication_pseudotime_sampleClusterGNN_LTC"

CELL_MAP_TSV = (
    RECLUSTER_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_cell_level_mapping_with_sample_labels.tsv"
)
NATIVE_H5AD = (
    RECLUSTER_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_joint_integrated_style_UMAP_scanpy.h5ad"
)
NATIVE_COORD_TSV = (
    RECLUSTER_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_joint_UMAP_coordinates_with_sample_labels.tsv"
)
HVG_TSV = RECLUSTER_DIR / "selected_HVGs.tsv"

ADJUSTED_Z_NPY = CCI_DIR / "EIGHT_SAMPLES_NO_SAMPLE7_stageB_adjusted_Z.npy"
PROFILE_PT_CSV = PT_DIR / "cluster_pseudotime.csv"
CELL_PT_CSV = PT_DIR / "cell_projected_pseudotime.csv"

OUT_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_native_vs_MLTD_mapped_integrated_UMAP"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_COORD_TSV = OUT_DIR / "native_vs_MLTD_mapped_cell_coordinates.tsv"
OUT_REP_NPY = OUT_DIR / "MLTD_mapped_representation.npy"
OUT_H5AD = OUT_DIR / "native_vs_MLTD_mapped_lightweight.h5ad"
OUT_DISPLACEMENT_TSV = (
    OUT_DIR / "MLTD_mapping_displacement_by_sample_cluster.tsv"
)
OUT_SUMMARY_JSON = OUT_DIR / "run_summary.json"
OUT_MARKER_XLSX = OUT_DIR / "cluster_top1000_marker_genes_for_annotation.xlsx"

# Expected sample structure.
EXPECTED_SAMPLE_COUNT = 8
EXPECTED_EXCLUDED_SAMPLE_NUMBERS = {7}

# Native-expression representation. These values match the previous joint UMAP
# code where applicable.
N_TOP_HVG = 2000
N_NATIVE_PCS = 15
N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.03
UMAP_SPREAD = 2.0
LEIDEN_RESOLUTION = 0.4
SEED = 42

# Stage-B embedding compression.
N_STAGEB_PCS = 30
STAGEB_PCA_FIT_MAX_CELLS = 50000
STAGEB_PCA_TRANSFORM_CHUNK = 4096

# MLTD-mapped representation block weights. Each block is standardized and
# divided by sqrt(number of block dimensions), so these are true block-level
# weights rather than dimension-count weights.
NATIVE_ANCHOR_WEIGHT = 1.00
STAGEB_DEPENDENCY_WEIGHT = 1.00
COMMUNICATION_WEIGHT = 0.70
TEMPORAL_WEIGHT = 0.70

COMMUNICATION_FEATURES = [
    "sender_strength",
    "receiver_strength",
    "receiver_tf_response",
    "net_flow",
    "io_ratio",
]
TEMPORAL_PROFILE_FEATURES = [
    "t_hat_norm_mean",
    "sigma_t",
    "tau_mean",
    "weak_teacher_rank",
]
TEMPORAL_CELL_FEATURES = [
    "cell_t_proj",
]

# Reuse the native UMAP generated during the CCI preparation whenever possible.
# Native PCA is still loaded because it anchors the mapped representation.
REUSE_EXISTING_NATIVE_UMAP = True

# For easier side-by-side interpretation, rotate/reflect/rescale the mapped UMAP
# onto the native UMAP by orthogonal Procrustes. This changes only displayed
# coordinates, not the mapped neighbour graph or mapped Leiden labels.
ALIGN_MAPPED_UMAP_TO_NATIVE_FOR_DISPLAY = True

# Optional mapped representation clustering.
RUN_MLTD_LEIDEN = True

# Memory controls used only if the previous native h5ad/PCA is unavailable.
DENSE_TO_SPARSE_CHUNK = 512
FULL_HVG_BUILD_CHUNK = 1024
MAX_DENSE_SCALE_GB = 6.0

# Plot settings.
FIG_DPI = 320
POINT_SIZE_MIN = 0.10
POINT_SIZE_MAX = 2.8
DISPLACEMENT_CLIP_QUANTILE = 0.99

# Marker-gene export for cluster annotation. Marker statistics are always
# calculated from the native measured RNA matrix. MLTD coordinates or mapped
# features are never treated as gene expression.
MARKER_TOP_N = 1000
MARKER_GENE_BATCH = 256
MARKER_CELL_CHUNK = 512
MARKER_TARGET_SUM = 1e4
MARKER_MIN_CELLS_PER_CLUSTER = 10
MARKER_MIN_PCT_IN = 0.0
MARKER_EXPORT_NATIVE_AND_MLTD_LEIDEN = True

np.random.seed(SEED)


# =============================================================================
# 1. HELPERS
# =============================================================================


def log(*args) -> None:
    print(*args, flush=True)


def read_table_auto(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def normalize_cluster_str(value: str) -> str:
    text = str(value).strip()
    if text.startswith("C") and text[1:].isdigit():
        return text[1:]
    return text


def normalize_cell_id(value: str) -> str:
    text = str(value)
    return text.split("::", 1)[-1] if "::" in text else text


def make_cell_key(sample_id: str, cell_id: str) -> str:
    return f"{sample_id}::{normalize_cell_id(cell_id)}"


def extract_sample_number(sample_id: str) -> int | None:
    match = re.search(r"Sample[-_ ]?(\d+)", str(sample_id), flags=re.I)
    return int(match.group(1)) if match else None


def choose_point_size(n_cells: int) -> float:
    if n_cells <= 0:
        return POINT_SIZE_MIN
    return float(
        np.clip(110000.0 / float(n_cells), POINT_SIZE_MIN, POINT_SIZE_MAX)
    )


def save_figure_both(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def safe_write_h5ad(adata: ad.AnnData, path: Path) -> None:
    try:
        adata.write_h5ad(path)
        log("[SAVE]", path)
    except Exception as exc:
        log("[H5AD WARN] could not write", path, repr(exc))


def standardize_block(matrix: np.ndarray, block_name: str) -> np.ndarray:
    """Standardize a feature block and balance its total dimensional weight."""
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)

    finite = np.isfinite(x)
    if not finite.all():
        medians = np.nanmedian(np.where(finite, x, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        bad_rows, bad_cols = np.where(~finite)
        x = x.copy()
        x[bad_rows, bad_cols] = medians[bad_cols]

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    x = scaler.fit_transform(x).astype(np.float32)
    x = np.clip(x, -10.0, 10.0)

    # Equalize total variance contribution per block before applying the
    # user-facing block weight.
    x /= float(math.sqrt(max(1, x.shape[1])))
    log(
        f"[BLOCK] {block_name}: shape={x.shape}; "
        f"mean_abs={float(np.mean(np.abs(x))):.5f}"
    )
    return x


def align_umap_to_reference(
    reference: np.ndarray,
    moving: np.ndarray,
) -> np.ndarray:
    """Orthogonally align moving 2D coordinates to reference coordinates."""
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.shape[1] != 2:
        raise ValueError("UMAP alignment requires two arrays with shape (n, 2).")

    ref_mean = ref.mean(axis=0, keepdims=True)
    mov_mean = mov.mean(axis=0, keepdims=True)
    ref0 = ref - ref_mean
    mov0 = mov - mov_mean

    ref_norm = np.linalg.norm(ref0)
    mov_norm = np.linalg.norm(mov0)
    if ref_norm < 1e-12 or mov_norm < 1e-12:
        return mov.astype(np.float32)

    mov_scaled = mov0 * (ref_norm / mov_norm)
    rotation, _ = orthogonal_procrustes(mov_scaled, ref0)
    aligned = mov_scaled @ rotation + ref_mean
    return aligned.astype(np.float32)


# =============================================================================
# 2. LOAD AND VALIDATE THE COMPLETED EIGHT-SAMPLE CELL SET
# =============================================================================


def load_manifest() -> tuple[pd.DataFrame, Path, Path, np.ndarray]:
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(INPUT_MANIFEST)

    manifest = pd.read_csv(INPUT_MANIFEST, sep="\t")
    rename_map = {}
    if "sid" in manifest.columns and "sample_id" not in manifest.columns:
        rename_map["sid"] = "sample_id"
    if "barcode" in manifest.columns and "cell_id" not in manifest.columns:
        rename_map["barcode"] = "cell_id"
    manifest = manifest.rename(columns=rename_map)

    required = ["sample_id", "cell_id", "row_index", "counts_npy", "meta_npz"]
    for column in required:
        if column not in manifest.columns:
            raise KeyError(f"{INPUT_MANIFEST.name}: missing {column}")

    manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
    manifest["cell_id"] = manifest["cell_id"].astype(str).map(normalize_cell_id)
    manifest["row_index"] = pd.to_numeric(
        manifest["row_index"], errors="raise"
    ).astype(np.int64)
    manifest["cell_key"] = [
        make_cell_key(s, c)
        for s, c in zip(manifest["sample_id"], manifest["cell_id"])
    ]

    if manifest["cell_key"].duplicated().any():
        duplicated = manifest.loc[
            manifest["cell_key"].duplicated(), "cell_key"
        ].head(10)
        raise ValueError(f"Duplicate manifest cell_key values: {duplicated.tolist()}")

    sample_labels = sorted(
        manifest["sample_id"].unique().tolist(), key=natural_sort_key
    )
    if len(sample_labels) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} modeled samples; found "
            f"{len(sample_labels)}: {sample_labels}"
        )

    sample_numbers = {
        n for n in (extract_sample_number(x) for x in sample_labels) if n is not None
    }
    unexpected_excluded = sample_numbers & EXPECTED_EXCLUDED_SAMPLE_NUMBERS
    if unexpected_excluded:
        raise ValueError(
            "The completed no-Sample-7 manifest unexpectedly contains excluded "
            f"sample numbers: {sorted(unexpected_excluded)}"
        )

    count_paths = sorted(set(manifest["counts_npy"].astype(str)))
    meta_paths = sorted(set(manifest["meta_npz"].astype(str)))
    if len(count_paths) != 1 or len(meta_paths) != 1:
        raise ValueError(
            "Expected one combined counts matrix and one combined metadata file. "
            f"counts={count_paths}; meta={meta_paths}"
        )

    counts_path = Path(count_paths[0])
    meta_path = Path(meta_paths[0])
    if not counts_path.exists():
        raise FileNotFoundError(counts_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = np.load(meta_path, allow_pickle=True)
    if "gene_names" not in meta:
        raise KeyError(f"{meta_path.name}: missing gene_names")
    gene_names = np.asarray(meta["gene_names"]).astype(str)

    counts_mm = np.load(counts_path, mmap_mode="r")
    if counts_mm.ndim != 2:
        raise ValueError(f"Counts matrix must be 2D: {counts_mm.shape}")
    if counts_mm.shape[1] != len(gene_names):
        raise ValueError(
            f"Counts columns {counts_mm.shape[1]} != genes {len(gene_names)}"
        )
    if manifest["row_index"].min() < 0 or manifest["row_index"].max() >= counts_mm.shape[0]:
        raise IndexError("Manifest row_index is outside the combined counts matrix.")

    log("[MANIFEST]", INPUT_MANIFEST)
    log("[COUNTS]", counts_path, counts_mm.shape)
    log("[META]", meta_path)
    log("[SAMPLES]", sample_labels)
    log(manifest["sample_id"].value_counts().sort_index().to_string())

    return manifest.reset_index(drop=True), counts_path, meta_path, gene_names


def standardize_cell_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = read_table_auto(path)

    rename_map = {}
    if "sid" in frame.columns and "sample_id" not in frame.columns:
        rename_map["sid"] = "sample_id"
    if "barcode" in frame.columns and "cell_id" not in frame.columns:
        rename_map["barcode"] = "cell_id"
    if "new_leiden" in frame.columns and "new_cluster" not in frame.columns:
        rename_map["new_leiden"] = "new_cluster"
    if "cluster_norm" in frame.columns and "new_cluster" not in frame.columns:
        rename_map["cluster_norm"] = "new_cluster"
    frame = frame.rename(columns=rename_map)

    required = ["sample_id", "new_cluster"]
    for column in required:
        if column not in frame.columns:
            raise KeyError(f"{path.name}: missing {column}")

    frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
    frame["new_cluster"] = frame["new_cluster"].astype(str).map(
        normalize_cluster_str
    )

    if "cell_id" in frame.columns:
        frame["cell_id"] = frame["cell_id"].astype(str).map(normalize_cell_id)
    elif "barcode" in frame.columns:
        frame["cell_id"] = frame["barcode"].astype(str).map(normalize_cell_id)

    if "cell_key" not in frame.columns:
        if "cell_id" not in frame.columns:
            raise KeyError(f"{path.name}: missing cell_key and cell_id")
        frame["cell_key"] = [
            make_cell_key(s, c)
            for s, c in zip(frame["sample_id"], frame["cell_id"])
        ]
    frame["cell_key"] = frame["cell_key"].astype(str)

    if "profile_key" not in frame.columns:
        frame["profile_key"] = [
            f"{s}::{c}"
            for s, c in zip(frame["sample_id"], frame["new_cluster"])
        ]
    frame["profile_key"] = frame["profile_key"].astype(str)

    if frame["cell_key"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate cell_key values")

    return frame.reset_index(drop=True)


def align_cell_map_to_manifest(
    cell_map: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    manifest_small = manifest[
        ["cell_key", "sample_id", "cell_id", "row_index"]
    ].copy()
    left = cell_map.copy()
    left["_cell_map_order"] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(
        manifest_small,
        on="cell_key",
        how="inner",
        suffixes=("", "_manifest"),
        validate="one_to_one",
    )

    if len(merged) != len(manifest) or len(merged) != len(cell_map):
        raise ValueError(
            "Cell map and completed Stage-B manifest do not contain exactly the "
            f"same cells: cell_map={len(cell_map)}, manifest={len(manifest)}, "
            f"matched={len(merged)}"
        )

    if "sample_id_manifest" in merged.columns:
        mismatch = merged["sample_id"].astype(str) != merged[
            "sample_id_manifest"
        ].astype(str)
        if mismatch.any():
            raise ValueError("sample_id mismatch between cell map and manifest")
        merged = merged.drop(columns=["sample_id_manifest"])

    if "cell_id_manifest" in merged.columns:
        if "cell_id" not in merged.columns:
            merged["cell_id"] = merged["cell_id_manifest"]
        merged = merged.drop(columns=["cell_id_manifest"])

    if "row_index_manifest" in merged.columns:
        if "row_index" in merged.columns:
            old_row = pd.to_numeric(merged["row_index"], errors="coerce")
            new_row = pd.to_numeric(merged["row_index_manifest"], errors="raise")
            mismatch = old_row.notna() & (old_row.astype(np.int64) != new_row.astype(np.int64))
            if mismatch.any():
                raise ValueError("row_index mismatch between cell map and manifest")
        merged["row_index"] = merged["row_index_manifest"].astype(np.int64)
        merged = merged.drop(columns=["row_index_manifest"])

    # Preserve the exact cell-map order because adjusted_Z was generated in it.
    merged = merged.sort_values("_cell_map_order").drop(columns=["_cell_map_order"])
    return merged.reset_index(drop=True)


# =============================================================================
# 3. NATIVE RNA PCA / UMAP
# =============================================================================


def dense_memmap_rows_to_csr(
    counts_mm,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    chunk_size: int,
) -> sp.csr_matrix:
    parts: list[sp.csr_matrix] = []
    row_indices = np.asarray(row_indices, dtype=np.int64)
    column_indices = np.asarray(column_indices, dtype=np.int64)

    for start in range(0, len(row_indices), int(chunk_size)):
        end = min(start + int(chunk_size), len(row_indices))
        rows = row_indices[start:end]
        block = np.asarray(
            counts_mm[rows[:, None], column_indices[None, :]],
            dtype=np.float32,
        )
        parts.append(sp.csr_matrix(block))
        del block
        gc.collect()

    return sp.vstack(parts, format="csr")


def load_hvg_indices(gene_names: np.ndarray) -> np.ndarray:
    if not HVG_TSV.exists():
        raise FileNotFoundError(
            f"Missing {HVG_TSV}. It should have been generated by the previous "
            "eight-sample CCI/reclustering code."
        )

    hvg = read_table_auto(HVG_TSV)
    if "gene_index_in_union_vocab" in hvg.columns:
        idx = pd.to_numeric(
            hvg["gene_index_in_union_vocab"], errors="raise"
        ).astype(np.int64).to_numpy()
    elif "gene" in hvg.columns:
        gene_to_i = {g: i for i, g in enumerate(gene_names.astype(str))}
        idx = np.asarray(
            [gene_to_i[g] for g in hvg["gene"].astype(str) if g in gene_to_i],
            dtype=np.int64,
        )
    else:
        raise KeyError(f"{HVG_TSV.name}: missing gene index/gene column")

    idx = np.unique(idx)
    idx = idx[(idx >= 0) & (idx < len(gene_names))]
    if len(idx) == 0:
        raise RuntimeError("No usable HVG indices were loaded.")
    if len(idx) > N_TOP_HVG:
        idx = idx[:N_TOP_HVG]
    return idx


def align_existing_native_arrays(
    cell_map: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load existing native PCA, UMAP and Leiden in exact cell-map order."""
    if not NATIVE_H5AD.exists():
        raise FileNotFoundError(NATIVE_H5AD)

    native = ad.read_h5ad(NATIVE_H5AD, backed="r")
    if "X_pca" not in native.obsm:
        raise KeyError(f"{NATIVE_H5AD.name}: missing obsm['X_pca']")

    if "cell_key" in native.obs.columns:
        native_keys = native.obs["cell_key"].astype(str).to_numpy()
    else:
        native_keys = native.obs_names.astype(str).to_numpy()

    key_to_row = {key: i for i, key in enumerate(native_keys.tolist())}
    missing = [key for key in cell_map["cell_key"] if key not in key_to_row]
    if missing:
        raise ValueError(
            f"Existing native h5ad is missing {len(missing)} modeled cells; "
            f"examples={missing[:5]}"
        )

    order = np.asarray(
        [key_to_row[key] for key in cell_map["cell_key"].astype(str)],
        dtype=np.int64,
    )
    pca = np.asarray(native.obsm["X_pca"])[order].astype(np.float32)

    umap = None
    if "X_umap" in native.obsm:
        umap = np.asarray(native.obsm["X_umap"])[order].astype(np.float32)

    leiden = None
    for column in ["new_leiden", "leiden", "new_cluster"]:
        if column in native.obs.columns:
            leiden = native.obs[column].astype(str).to_numpy()[order]
            break

    try:
        native.file.close()
    except Exception:
        pass
    del native
    gc.collect()
    log("[NATIVE] loaded existing PCA/UMAP from", NATIVE_H5AD)
    return pca, umap, leiden


def rebuild_native_pca_umap(
    cell_map: pd.DataFrame,
    counts_path: Path,
    gene_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hvg_indices = load_hvg_indices(gene_names)
    counts_mm = np.load(counts_path, mmap_mode="r")
    rows = cell_map["row_index"].to_numpy(dtype=np.int64)

    log("[NATIVE] rebuilding sparse HVG matrix ...")
    x_hvg = dense_memmap_rows_to_csr(
        counts_mm,
        rows,
        hvg_indices,
        chunk_size=FULL_HVG_BUILD_CHUNK,
    )

    obs = cell_map[["cell_key", "sample_id", "new_cluster", "row_index"]].copy()
    obs.index = pd.Index(obs["cell_key"].astype(str), name="cell_key_index")
    native = ad.AnnData(
        X=x_hvg,
        obs=obs,
        var=pd.DataFrame(index=gene_names[hvg_indices].astype(str)),
    )

    sc.pp.normalize_total(native, target_sum=1e4)
    sc.pp.log1p(native)

    estimated_dense_gb = native.n_obs * native.n_vars * 4.0 / 1024**3
    if estimated_dense_gb <= MAX_DENSE_SCALE_GB:
        sc.pp.scale(native, max_value=10)
    else:
        sc.pp.scale(native, zero_center=False, max_value=10)

    n_pcs = min(N_NATIVE_PCS, native.n_vars - 1, native.n_obs - 1)
    if n_pcs < 2:
        raise RuntimeError("Too few cells/genes for native PCA.")

    sc.tl.pca(
        native,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=SEED,
    )
    sc.pp.neighbors(
        native,
        n_neighbors=N_NEIGHBORS,
        n_pcs=n_pcs,
        use_rep="X_pca",
        key_added="native_neighbors",
        random_state=SEED,
    )
    sc.tl.umap(
        native,
        neighbors_key="native_neighbors",
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=SEED,
    )
    sc.tl.leiden(
        native,
        neighbors_key="native_neighbors",
        resolution=LEIDEN_RESOLUTION,
        key_added="native_leiden",
        random_state=SEED,
    )

    pca = np.asarray(native.obsm["X_pca"], dtype=np.float32)
    umap = np.asarray(native.obsm["X_umap"], dtype=np.float32)
    leiden = native.obs["native_leiden"].astype(str).to_numpy()

    del native, x_hvg
    gc.collect()
    log("[NATIVE] rebuilt native PCA and UMAP")
    return pca, umap, leiden


def load_or_build_native_representation(
    cell_map: pd.DataFrame,
    counts_path: Path,
    gene_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if NATIVE_H5AD.exists():
        try:
            pca, umap, leiden = align_existing_native_arrays(cell_map)
            if umap is not None and REUSE_EXISTING_NATIVE_UMAP:
                return pca, umap, leiden
            # If PCA exists but UMAP is absent, calculate a UMAP from that PCA.
            tmp = ad.AnnData(obs=cell_map[["cell_key", "sample_id"]].copy())
            tmp.obsm["X_pca"] = pca[:, : min(N_NATIVE_PCS, pca.shape[1])]
            sc.pp.neighbors(
                tmp,
                n_neighbors=N_NEIGHBORS,
                use_rep="X_pca",
                key_added="native_neighbors",
                random_state=SEED,
            )
            sc.tl.umap(
                tmp,
                neighbors_key="native_neighbors",
                min_dist=UMAP_MIN_DIST,
                spread=UMAP_SPREAD,
                random_state=SEED,
            )
            umap = np.asarray(tmp.obsm["X_umap"], dtype=np.float32)
            if RUN_MLTD_LEIDEN:
                try:
                    sc.tl.leiden(
                        tmp,
                        neighbors_key="native_neighbors",
                        resolution=LEIDEN_RESOLUTION,
                        key_added="native_leiden",
                        random_state=SEED,
                    )
                    leiden = tmp.obs["native_leiden"].astype(str).to_numpy()
                except Exception as exc:
                    log("[NATIVE LEIDEN WARN]", repr(exc))
            return pca, umap, leiden
        except Exception as exc:
            log("[NATIVE WARN] existing h5ad could not be reused:", repr(exc))

    return rebuild_native_pca_umap(cell_map, counts_path, gene_names)


# =============================================================================
# 4. LOAD CELL-MAPPED MLTD FEATURES
# =============================================================================


def standardize_profile_pt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)

    rename_map = {}
    if "cluster_norm" in frame.columns and "new_cluster" not in frame.columns:
        rename_map["cluster_norm"] = "new_cluster"
    if "cluster_label" in frame.columns and "new_cluster" not in frame.columns:
        rename_map["cluster_label"] = "new_cluster"
    if "new_leiden" in frame.columns and "new_cluster" not in frame.columns:
        rename_map["new_leiden"] = "new_cluster"
    frame = frame.rename(columns=rename_map)

    if "sample_id" not in frame.columns:
        raise KeyError(f"{path.name}: missing sample_id")
    if "new_cluster" not in frame.columns:
        raise KeyError(
            f"{path.name}: missing cluster column; found {frame.columns.tolist()}"
        )

    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["new_cluster"] = frame["new_cluster"].astype(str).map(
        normalize_cluster_str
    )
    if "profile_key" not in frame.columns:
        frame["profile_key"] = [
            f"{s}::{c}"
            for s, c in zip(frame["sample_id"], frame["new_cluster"])
        ]
    frame["profile_key"] = frame["profile_key"].astype(str)

    if frame["profile_key"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate profile_key")

    return frame


def standardize_cell_pt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)

    rename_map = {}
    if "sid" in frame.columns and "sample_id" not in frame.columns:
        rename_map["sid"] = "sample_id"
    if "barcode" in frame.columns and "cell_id" not in frame.columns:
        rename_map["barcode"] = "cell_id"
    frame = frame.rename(columns=rename_map)

    if "cell_key" not in frame.columns:
        if "sample_id" not in frame.columns or "cell_id" not in frame.columns:
            raise KeyError(f"{path.name}: missing cell_key or sample_id/cell_id")
        frame["cell_key"] = [
            make_cell_key(s, c)
            for s, c in zip(frame["sample_id"], frame["cell_id"])
        ]
    frame["cell_key"] = frame["cell_key"].astype(str)
    if frame["cell_key"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate cell_key")
    return frame


def map_mltd_features_to_cells(cell_map: pd.DataFrame) -> pd.DataFrame:
    profile = standardize_profile_pt(PROFILE_PT_CSV)
    cell_pt = standardize_cell_pt(CELL_PT_CSV)

    profile_keep = ["profile_key"] + [
        c
        for c in COMMUNICATION_FEATURES + TEMPORAL_PROFILE_FEATURES
        if c in profile.columns
    ]
    missing_profile = [
        c
        for c in COMMUNICATION_FEATURES + TEMPORAL_PROFILE_FEATURES
        if c not in profile.columns
    ]
    if missing_profile:
        log("[MLTD WARN] missing profile features:", missing_profile)

    mapped = cell_map.merge(
        profile[profile_keep],
        on="profile_key",
        how="left",
        validate="many_to_one",
    )

    cell_keep = ["cell_key"] + [
        c for c in TEMPORAL_CELL_FEATURES if c in cell_pt.columns
    ]
    missing_cell = [c for c in TEMPORAL_CELL_FEATURES if c not in cell_pt.columns]
    if missing_cell:
        log("[MLTD WARN] missing cell temporal features:", missing_cell)

    mapped = mapped.merge(
        cell_pt[cell_keep],
        on="cell_key",
        how="left",
        validate="one_to_one",
    )

    used = [
        c
        for c in COMMUNICATION_FEATURES
        + TEMPORAL_PROFILE_FEATURES
        + TEMPORAL_CELL_FEATURES
        if c in mapped.columns
    ]
    if not used:
        raise RuntimeError("No communication/temporal MLTD features were mapped.")

    for column in used:
        mapped[column] = pd.to_numeric(mapped[column], errors="coerce")
        if mapped[column].isna().all():
            raise ValueError(f"Mapped feature is entirely missing: {column}")
        mapped[column] = mapped[column].fillna(mapped[column].median())

    log("[MLTD] mapped cell features:", used)
    return mapped


# =============================================================================
# 5. BUILD THE MLTD-MAPPED REPRESENTATION
# =============================================================================


def compress_stageb_adjusted_z(
    adjusted_z_path: Path,
    sample_ids: np.ndarray,
) -> np.ndarray:
    if not adjusted_z_path.exists():
        raise FileNotFoundError(adjusted_z_path)

    z_mm = np.load(adjusted_z_path, mmap_mode="r")
    if z_mm.ndim != 2:
        raise ValueError(f"Adjusted Z must be 2D: {z_mm.shape}")
    if z_mm.shape[0] != len(sample_ids):
        raise ValueError(
            f"Adjusted Z rows {z_mm.shape[0]} != modeled cells {len(sample_ids)}"
        )

    n_components = min(N_STAGEB_PCS, z_mm.shape[1], z_mm.shape[0] - 1)
    if n_components < 2:
        raise RuntimeError("Too few dimensions/cells for Stage-B PCA.")

    rng = np.random.default_rng(SEED)
    fit_indices: list[np.ndarray] = []
    unique_samples = sorted(np.unique(sample_ids), key=natural_sort_key)
    per_sample = max(1, STAGEB_PCA_FIT_MAX_CELLS // len(unique_samples))

    for sample_id in unique_samples:
        idx = np.where(sample_ids == sample_id)[0]
        take = min(len(idx), per_sample)
        if len(idx) > take:
            idx = np.sort(rng.choice(idx, size=take, replace=False))
        fit_indices.append(idx)

    fit_idx = np.concatenate(fit_indices)
    if len(fit_idx) > STAGEB_PCA_FIT_MAX_CELLS:
        fit_idx = np.sort(
            rng.choice(
                fit_idx,
                size=STAGEB_PCA_FIT_MAX_CELLS,
                replace=False,
            )
        )

    z_fit = np.asarray(z_mm[fit_idx], dtype=np.float32)
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=SEED,
    )
    pca.fit(z_fit)
    del z_fit
    gc.collect()

    transformed = np.empty((z_mm.shape[0], n_components), dtype=np.float32)
    for start in range(0, z_mm.shape[0], STAGEB_PCA_TRANSFORM_CHUNK):
        end = min(start + STAGEB_PCA_TRANSFORM_CHUNK, z_mm.shape[0])
        transformed[start:end] = pca.transform(
            np.asarray(z_mm[start:end], dtype=np.float32)
        ).astype(np.float32)
        if start == 0 or end == z_mm.shape[0] or start % 20000 == 0:
            log(f"[STAGE-B PCA] {start}:{end}/{z_mm.shape[0]}")

    log(
        "[STAGE-B PCA] fit cells=",
        len(fit_idx),
        "components=",
        n_components,
        "explained_variance=",
        round(float(pca.explained_variance_ratio_.sum()), 5),
    )
    return transformed


def build_mltd_mapped_representation(
    native_pca: np.ndarray,
    stageb_pca: np.ndarray,
    mapped_features: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    native_dim = min(N_NATIVE_PCS, native_pca.shape[1])
    native_block = standardize_block(
        native_pca[:, :native_dim], "native_RNA_PCA_anchor"
    )
    stageb_block = standardize_block(stageb_pca, "StageB_adjusted_dependency")

    comm_cols = [c for c in COMMUNICATION_FEATURES if c in mapped_features.columns]
    temporal_cols = [
        c
        for c in TEMPORAL_PROFILE_FEATURES + TEMPORAL_CELL_FEATURES
        if c in mapped_features.columns
    ]

    comm_block = standardize_block(
        mapped_features[comm_cols].to_numpy(dtype=np.float32),
        "communication_sender_receiver_TF",
    )
    temporal_block = standardize_block(
        mapped_features[temporal_cols].to_numpy(dtype=np.float32),
        "communication_pseudotime",
    )

    blocks = [
        NATIVE_ANCHOR_WEIGHT * native_block,
        STAGEB_DEPENDENCY_WEIGHT * stageb_block,
        COMMUNICATION_WEIGHT * comm_block,
        TEMPORAL_WEIGHT * temporal_block,
    ]
    representation = np.concatenate(blocks, axis=1).astype(np.float32)

    np.save(OUT_REP_NPY, representation)
    log("[SAVE]", OUT_REP_NPY, representation.shape)

    metadata = {
        "native_anchor_columns": int(native_block.shape[1]),
        "stageb_dependency_columns": int(stageb_block.shape[1]),
        "communication_columns": comm_cols,
        "temporal_columns": temporal_cols,
        "weights": {
            "native_anchor": NATIVE_ANCHOR_WEIGHT,
            "stageb_dependency": STAGEB_DEPENDENCY_WEIGHT,
            "communication": COMMUNICATION_WEIGHT,
            "temporal": TEMPORAL_WEIGHT,
        },
        "representation_shape": list(representation.shape),
    }
    return representation, metadata


def run_mltd_umap(
    representation: np.ndarray,
    cell_map: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:
    obs = cell_map[
        ["cell_key", "sample_id", "new_cluster", "profile_key", "row_index"]
    ].copy()
    obs.index = pd.Index(obs["cell_key"].astype(str), name="cell_key_index")

    mapped = ad.AnnData(obs=obs)
    mapped.obsm["X_mltd_mapped"] = representation

    sc.pp.neighbors(
        mapped,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_mltd_mapped",
        key_added="mltd_mapped_neighbors",
        random_state=SEED,
    )
    sc.tl.umap(
        mapped,
        neighbors_key="mltd_mapped_neighbors",
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=SEED,
    )

    leiden = None
    if RUN_MLTD_LEIDEN:
        try:
            sc.tl.leiden(
                mapped,
                neighbors_key="mltd_mapped_neighbors",
                resolution=LEIDEN_RESOLUTION,
                key_added="mltd_leiden",
                random_state=SEED,
            )
            leiden = mapped.obs["mltd_leiden"].astype(str).to_numpy()
        except Exception as exc:
            log("[MLTD LEIDEN WARN]", repr(exc))

    umap = np.asarray(mapped.obsm["X_umap"], dtype=np.float32)
    del mapped
    gc.collect()
    return umap, leiden



# =============================================================================
# 6. NATIVE-EXPRESSION MARKER GENES FOR CLUSTER ANNOTATION
# =============================================================================


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def find_native_log1p_matrix(counts_path: Path) -> Path | None:
    """Find a precomputed native log1p-normalized matrix beside counts."""
    candidates = [
        counts_path.with_name(
            counts_path.name.replace(
                ".X_rna_counts_dense.npy",
                ".X_rna_log1p_norm_dense.npy",
            )
        ),
        counts_path.with_name(
            counts_path.name.replace("counts_dense", "log1p_norm_dense")
        ),
        counts_path.with_name(
            counts_path.name.replace("counts", "log1p_norm")
        ),
        counts_path.parent / "ALL_SAMPLES.X_rna_log1p_norm_dense.npy",
        counts_path.parent / "ALL_SAMPLES.X_rna_log1p_dense.npy",
    ]
    seen = set()
    for path in candidates:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            try:
                arr = np.load(path, mmap_mode="r")
                if arr.ndim == 2:
                    log("[MARKER] using precomputed native log1p matrix:", path)
                    return path
            except Exception:
                continue
    return None


def load_depth_from_meta(
    meta_path: Path,
    row_indices: np.ndarray,
    n_source_cells: int,
) -> np.ndarray | None:
    """Read a per-cell library-size vector from common metadata keys."""
    meta = np.load(meta_path, allow_pickle=True)
    candidate_keys = [
        "total_counts",
        "library_size",
        "library_sizes",
        "depth",
        "depths",
        "rna_depth",
        "rna_depths",
        "nCount_RNA",
        "cell_sums",
        "counts_per_cell",
    ]
    for key in candidate_keys:
        if key not in meta:
            continue
        values = np.asarray(meta[key]).reshape(-1)
        if values.size == n_source_cells:
            depth = values[row_indices].astype(np.float64)
        elif values.size == len(row_indices):
            depth = values.astype(np.float64)
        else:
            continue
        if np.all(np.isfinite(depth)) and np.nanmax(depth) > 0:
            log("[MARKER] using library depth from meta key:", key)
            return np.maximum(depth, 1.0)
    return None


def compute_depth_from_counts(
    counts_mm,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Fallback library sizes. Reads only modeled cells and never changes counts."""
    log("[MARKER] no usable depth metadata; computing cell library sizes from counts")
    depth = np.zeros(len(row_indices), dtype=np.float64)
    for start in range(0, len(row_indices), MARKER_CELL_CHUNK):
        end = min(start + MARKER_CELL_CHUNK, len(row_indices))
        rows = row_indices[start:end]
        block = np.asarray(counts_mm[rows, :], dtype=np.float32)
        depth[start:end] = np.asarray(block.sum(axis=1), dtype=np.float64)
        del block
        gc.collect()
        if start == 0 or end == len(row_indices) or end % 10000 < MARKER_CELL_CHUNK:
            log(f"[MARKER DEPTH] {end}/{len(row_indices)} cells")
    return np.maximum(depth, 1.0)


def make_cluster_membership(labels: np.ndarray):
    labels = np.asarray(labels).astype(str)
    unique_labels = sorted(np.unique(labels).tolist(), key=natural_sort_key)
    label_to_code = {label: i for i, label in enumerate(unique_labels)}
    codes = np.asarray([label_to_code[x] for x in labels], dtype=np.int64)
    counts = np.bincount(codes, minlength=len(unique_labels)).astype(np.int64)

    valid = counts >= MARKER_MIN_CELLS_PER_CLUSTER
    if not np.any(valid):
        raise RuntimeError(
            "No cluster has enough cells for marker analysis: "
            f"minimum={MARKER_MIN_CELLS_PER_CLUSTER}"
        )
    if not np.all(valid):
        dropped = [
            f"{unique_labels[i]}(n={counts[i]})"
            for i in range(len(unique_labels))
            if not valid[i]
        ]
        log("[MARKER WARN] dropping small clusters:", dropped)
        keep_labels = [unique_labels[i] for i in range(len(unique_labels)) if valid[i]]
        remap = {label: i for i, label in enumerate(keep_labels)}
        keep_cell = np.asarray([label in remap for label in labels], dtype=bool)
        labels_kept = labels[keep_cell]
        codes = np.asarray([remap[x] for x in labels_kept], dtype=np.int64)
        unique_labels = keep_labels
        counts = np.bincount(codes, minlength=len(unique_labels)).astype(np.int64)
    else:
        keep_cell = np.ones(len(labels), dtype=bool)

    n_cells = int(keep_cell.sum())
    membership = sp.csr_matrix(
        (
            np.ones(n_cells, dtype=np.float32),
            (codes, np.arange(n_cells, dtype=np.int64)),
        ),
        shape=(len(unique_labels), n_cells),
    )
    return unique_labels, counts, keep_cell, membership


def compute_top_markers_for_labels(
    scheme_name: str,
    labels: np.ndarray,
    counts_path: Path,
    meta_path: Path,
    row_indices: np.ndarray,
    gene_names: np.ndarray,
    native_log1p_path: Path | None,
    depth_all: np.ndarray | None,
) -> pd.DataFrame:
    """
    Compute one-cluster-versus-rest Welch statistics from native RNA.

    The displayed MLTD UMAP can define a grouping (for example mltd_leiden),
    but all marker values are calculated from measured RNA expression.
    """
    labels = np.asarray(labels).astype(str)
    unique_labels, n_per_cluster, keep_cell, membership = make_cluster_membership(labels)
    rows = np.asarray(row_indices, dtype=np.int64)[keep_cell]
    n_cells = len(rows)
    n_clusters = len(unique_labels)
    n_genes = len(gene_names)

    if n_cells < 2:
        raise RuntimeError(f"{scheme_name}: too few cells for marker analysis")
    if np.any(n_cells - n_per_cluster < 2):
        raise RuntimeError(
            f"{scheme_name}: at least one cluster leaves fewer than two rest cells"
        )

    log(
        f"[MARKER] scheme={scheme_name}; cells={n_cells}; "
        f"clusters={n_clusters}; genes={n_genes}"
    )

    mean_in_all = np.empty((n_clusters, n_genes), dtype=np.float32)
    mean_rest_all = np.empty((n_clusters, n_genes), dtype=np.float32)
    pct_in_all = np.empty((n_clusters, n_genes), dtype=np.float32)
    pct_rest_all = np.empty((n_clusters, n_genes), dtype=np.float32)
    score_all = np.empty((n_clusters, n_genes), dtype=np.float32)
    pval_all = np.empty((n_clusters, n_genes), dtype=np.float64)

    counts_mm = np.load(counts_path, mmap_mode="r")
    log1p_mm = (
        np.load(native_log1p_path, mmap_mode="r")
        if native_log1p_path is not None
        else None
    )
    depth = None if depth_all is None else np.asarray(depth_all)[keep_cell]

    for start in range(0, n_genes, MARKER_GENE_BATCH):
        end = min(start + MARKER_GENE_BATCH, n_genes)
        gene_idx = np.arange(start, end, dtype=np.int64)

        if log1p_mm is not None:
            x = np.asarray(
                log1p_mm[rows[:, None], gene_idx[None, :]],
                dtype=np.float32,
            )
        else:
            if depth is None:
                raise RuntimeError("Marker depth vector is unavailable")
            raw = np.asarray(
                counts_mm[rows[:, None], gene_idx[None, :]],
                dtype=np.float32,
            )
            x = np.log1p(
                raw
                * (float(MARKER_TARGET_SUM) / np.maximum(depth, 1.0))[:, None]
            ).astype(np.float32)
            del raw

        total_sum = np.asarray(x.sum(axis=0), dtype=np.float64)
        total_sumsq = np.asarray(np.square(x, dtype=np.float32).sum(axis=0), dtype=np.float64)
        total_detect = np.asarray((x > 0).sum(axis=0), dtype=np.float64)

        sum_in = np.asarray(membership @ x, dtype=np.float64)
        sumsq_in = np.asarray(membership @ np.square(x, dtype=np.float32), dtype=np.float64)
        detect_in = np.asarray(membership @ (x > 0).astype(np.float32), dtype=np.float64)

        n_in = n_per_cluster.astype(np.float64)[:, None]
        n_rest = (n_cells - n_per_cluster).astype(np.float64)[:, None]
        sum_rest = total_sum[None, :] - sum_in
        sumsq_rest = total_sumsq[None, :] - sumsq_in
        detect_rest = total_detect[None, :] - detect_in

        mean_in = sum_in / np.maximum(n_in, 1.0)
        mean_rest = sum_rest / np.maximum(n_rest, 1.0)
        pct_in = detect_in / np.maximum(n_in, 1.0)
        pct_rest = detect_rest / np.maximum(n_rest, 1.0)

        var_in = (
            sumsq_in - np.square(sum_in) / np.maximum(n_in, 1.0)
        ) / np.maximum(n_in - 1.0, 1.0)
        var_rest = (
            sumsq_rest - np.square(sum_rest) / np.maximum(n_rest, 1.0)
        ) / np.maximum(n_rest - 1.0, 1.0)
        var_in = np.maximum(var_in, 0.0)
        var_rest = np.maximum(var_rest, 0.0)

        se2 = var_in / np.maximum(n_in, 1.0) + var_rest / np.maximum(n_rest, 1.0)
        score = (mean_in - mean_rest) / np.sqrt(np.maximum(se2, 1e-20))
        dof_num = np.square(se2)
        dof_den = (
            np.square(var_in / np.maximum(n_in, 1.0))
            / np.maximum(n_in - 1.0, 1.0)
            + np.square(var_rest / np.maximum(n_rest, 1.0))
            / np.maximum(n_rest - 1.0, 1.0)
        )
        dof = dof_num / np.maximum(dof_den, 1e-20)
        pval = stats.t.sf(score, df=np.maximum(dof, 1.0))
        pval = np.clip(pval, 1e-300, 1.0)

        mean_in_all[:, start:end] = mean_in.astype(np.float32)
        mean_rest_all[:, start:end] = mean_rest.astype(np.float32)
        pct_in_all[:, start:end] = pct_in.astype(np.float32)
        pct_rest_all[:, start:end] = pct_rest.astype(np.float32)
        score_all[:, start:end] = score.astype(np.float32)
        pval_all[:, start:end] = pval

        del x, total_sum, total_sumsq, total_detect
        del sum_in, sumsq_in, detect_in, sum_rest, sumsq_rest, detect_rest
        gc.collect()
        if start == 0 or end == n_genes or end % (MARKER_GENE_BATCH * 10) == 0:
            log(f"[MARKER {scheme_name}] {end}/{n_genes} genes")

    output_tables = []
    genes = np.asarray(gene_names).astype(str)
    for cluster_index, cluster_label in enumerate(unique_labels):
        score = score_all[cluster_index].astype(float)
        mean_in = mean_in_all[cluster_index].astype(float)
        mean_rest = mean_rest_all[cluster_index].astype(float)
        pct_in = pct_in_all[cluster_index].astype(float)
        pct_rest = pct_rest_all[cluster_index].astype(float)
        pval = pval_all[cluster_index].astype(float)
        fdr = bh_fdr(pval)
        mean_diff = mean_in - mean_rest

        table = pd.DataFrame(
            {
                "scheme": scheme_name,
                "cluster": str(cluster_label),
                "gene": genes,
                "welch_t_score": score,
                "pval_one_sided": pval,
                "fdr": fdr,
                "mean_log1p_in_cluster": mean_in,
                "mean_log1p_rest": mean_rest,
                "mean_log1p_difference": mean_diff,
                "pct_in_cluster": pct_in,
                "pct_rest": pct_rest,
                "pct_difference": pct_in - pct_rest,
                "n_cells_cluster": int(n_per_cluster[cluster_index]),
                "n_cells_rest": int(n_cells - n_per_cluster[cluster_index]),
            }
        )
        table = table[
            np.isfinite(table["welch_t_score"])
            & (table["mean_log1p_difference"] > 0)
            & (table["pct_in_cluster"] >= MARKER_MIN_PCT_IN)
        ].copy()
        table = table.sort_values(
            ["welch_t_score", "mean_log1p_difference", "pct_difference"],
            ascending=[False, False, False],
        ).head(MARKER_TOP_N).reset_index(drop=True)
        table.insert(2, "marker_rank", np.arange(1, len(table) + 1))
        output_tables.append(table)

    if not output_tables:
        return pd.DataFrame()
    return pd.concat(output_tables, ignore_index=True)


def excel_safe_sheet_name(name: str, used: set[str]) -> str:
    value = re.sub(r"[\\/*?:\[\]]+", "_", str(name)).strip() or "Sheet"
    value = value[:31]
    base = value
    suffix = 1
    while value in used:
        tail = f"_{suffix}"
        value = base[: 31 - len(tail)] + tail
        suffix += 1
    used.add(value)
    return value


def write_marker_workbook(
    marker_tables: dict[str, pd.DataFrame],
    sample_labels: list[str],
) -> None:
    try:
        import xlsxwriter  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Marker Excel export requires xlsxwriter. Run: %pip install xlsxwriter"
        ) from exc

    used_sheets: set[str] = set()
    with pd.ExcelWriter(OUT_MARKER_XLSX, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#365F91",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        num_fmt = workbook.add_format({"num_format": "0.0000"})
        sci_fmt = workbook.add_format({"num_format": "0.00E+00"})

        readme = pd.DataFrame(
            {
                "Item": [
                    "Dataset",
                    "Samples",
                    "Native matrix modified",
                    "Marker expression source",
                    "Comparison",
                    "Ranking",
                    "Top genes per cluster",
                    "Important interpretation",
                ],
                "Value": [
                    DATASET_NAME,
                    "; ".join(sample_labels),
                    "No",
                    "Measured native RNA; log1p normalized to 10,000 counts per cell",
                    "Each cluster versus all remaining modeled cells",
                    "One-sided Welch t score; positive cluster-enriched genes only",
                    str(MARKER_TOP_N),
                    (
                        "MLTD/Native Leiden labels may define cell groups, but marker "
                        "expression is always calculated from the original RNA matrix."
                    ),
                ],
            }
        )
        readme_sheet = excel_safe_sheet_name("README", used_sheets)
        readme.to_excel(writer, sheet_name=readme_sheet, index=False)
        ws = writer.sheets[readme_sheet]
        ws.freeze_panes(1, 0)
        ws.set_column("A:A", 28)
        ws.set_column("B:B", 90, note_fmt)
        ws.set_row(0, 24, header_fmt)

        for scheme_name, table in marker_tables.items():
            if table is None or len(table) == 0:
                continue
            all_sheet = excel_safe_sheet_name(f"{scheme_name}_All", used_sheets)
            table.to_excel(writer, sheet_name=all_sheet, index=False)
            ws = writer.sheets[all_sheet]
            ws.freeze_panes(1, 3)
            ws.autofilter(0, 0, len(table), len(table.columns) - 1)
            ws.set_row(0, 24, header_fmt)
            ws.set_column("A:A", 18)
            ws.set_column("B:B", 12)
            ws.set_column("C:C", 11)
            ws.set_column("D:D", 18)
            ws.set_column("E:E", 15, num_fmt)
            ws.set_column("F:G", 14, sci_fmt)
            ws.set_column("H:M", 20, num_fmt)
            ws.set_column("N:O", 16)

            # The fixed original/global clusters are the labels displayed in the
            # cluster-coloured comparison figure. Give each one its own sheet.
            if scheme_name == "OriginalCluster":
                for cluster, sub in table.groupby("cluster", sort=False):
                    sheet = excel_safe_sheet_name(f"Cluster_{cluster}", used_sheets)
                    sub.to_excel(writer, sheet_name=sheet, index=False)
                    cws = writer.sheets[sheet]
                    cws.freeze_panes(1, 3)
                    cws.autofilter(0, 0, len(sub), len(sub.columns) - 1)
                    cws.set_row(0, 24, header_fmt)
                    cws.set_column("A:A", 18)
                    cws.set_column("B:B", 12)
                    cws.set_column("C:C", 11)
                    cws.set_column("D:D", 18)
                    cws.set_column("E:E", 15, num_fmt)
                    cws.set_column("F:G", 14, sci_fmt)
                    cws.set_column("H:M", 20, num_fmt)
                    cws.set_column("N:O", 16)

    log("[SAVE]", OUT_MARKER_XLSX)


def export_cluster_marker_excel(
    output: pd.DataFrame,
    counts_path: Path,
    meta_path: Path,
    gene_names: np.ndarray,
    sample_labels: list[str],
) -> dict[str, int]:
    row_indices = output["row_index"].to_numpy(dtype=np.int64)
    counts_mm = np.load(counts_path, mmap_mode="r")
    native_log1p_path = find_native_log1p_matrix(counts_path)

    depth_all = None
    if native_log1p_path is None:
        depth_all = load_depth_from_meta(
            meta_path,
            row_indices=row_indices,
            n_source_cells=counts_mm.shape[0],
        )
        if depth_all is None:
            depth_all = compute_depth_from_counts(counts_mm, row_indices)

    schemes: dict[str, np.ndarray] = {
        "OriginalCluster": output["new_cluster"].astype(str).to_numpy(),
    }
    if MARKER_EXPORT_NATIVE_AND_MLTD_LEIDEN:
        if "native_leiden" in output.columns:
            schemes["NativeLeiden"] = output["native_leiden"].astype(str).to_numpy()
        if "mltd_leiden" in output.columns:
            schemes["MLTDLeiden"] = output["mltd_leiden"].astype(str).to_numpy()

    marker_tables: dict[str, pd.DataFrame] = {}
    summary: dict[str, int] = {}
    for scheme_name, labels in schemes.items():
        marker_table = compute_top_markers_for_labels(
            scheme_name=scheme_name,
            labels=labels,
            counts_path=counts_path,
            meta_path=meta_path,
            row_indices=row_indices,
            gene_names=gene_names,
            native_log1p_path=native_log1p_path,
            depth_all=depth_all,
        )
        marker_tables[scheme_name] = marker_table
        summary[scheme_name] = int(marker_table["cluster"].nunique()) if len(marker_table) else 0

    write_marker_workbook(marker_tables, sample_labels)
    return summary


# =============================================================================
# 6. PLOTS
# =============================================================================


def plot_by_category(
    axes: Iterable[plt.Axes],
    native_umap: np.ndarray,
    mapped_umap: np.ndarray,
    categories: np.ndarray,
    category_name: str,
    native_title: str,
    mapped_title: str,
    annotate_clusters: bool = False,
) -> None:
    axes = list(axes)
    labels = sorted(np.unique(categories.astype(str)), key=natural_sort_key)
    cmap = plt.get_cmap("tab20")
    point_size = choose_point_size(len(categories))

    for ax, coords, title in zip(
        axes,
        [native_umap, mapped_umap],
        [native_title, mapped_title],
    ):
        for i, label in enumerate(labels):
            mask = categories.astype(str) == str(label)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                alpha=0.74,
                linewidths=0,
                rasterized=True,
                color=cmap(i % cmap.N),
                label=str(label),
            )
            if annotate_clusters and np.any(mask):
                ax.text(
                    float(np.median(coords[mask, 0])),
                    float(np.median(coords[mask, 1])),
                    str(label),
                    fontsize=8,
                    ha="center",
                    va="center",
                )

        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].legend(
        title=category_name,
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        markerscale=4,
        fontsize=8,
    )


def make_plots(
    output: pd.DataFrame,
) -> None:
    native = output[["native_umap_1", "native_umap_2"]].to_numpy(
        dtype=np.float32
    )
    mapped = output[["mltd_umap_1", "mltd_umap_2"]].to_numpy(
        dtype=np.float32
    )

    # A. By sample.
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    plot_by_category(
        axes,
        native,
        mapped,
        output["sample_id"].astype(str).to_numpy(),
        "Sample",
        "Native RNA expression UMAP",
        "MLTD-mapped dependency-aware UMAP",
        annotate_clusters=False,
    )
    fig.suptitle(
        "Eight-sample integrated UMAP: native expression versus MLTD mapping",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    save_figure_both(fig, OUT_DIR / "UMAP_native_vs_MLTD_by_sample")

    # B. By fixed original/global Leiden cluster. The same labels are used in
    # both panels so cell movement can be interpreted directly. Cluster IDs
    # are kept only in the legend; no numbers are printed over the UMAP points.
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    plot_by_category(
        axes,
        native,
        mapped,
        output["new_cluster"].astype(str).to_numpy(),
        "Original global cluster",
        "Native RNA expression UMAP",
        "MLTD-mapped dependency-aware UMAP",
        annotate_clusters=False,
    )
    fig.suptitle(
        "Same cells and fixed cluster labels before and after MLTD mapping",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    save_figure_both(fig, OUT_DIR / "UMAP_native_vs_MLTD_by_cluster")

    # C. Mapping displacement shown on the native coordinates.
    displacement = output["mltd_mapping_displacement"].to_numpy(dtype=float)
    vmax = float(np.quantile(displacement, DISPLACEMENT_CLIP_QUANTILE))
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    scatter = ax.scatter(
        native[:, 0],
        native[:, 1],
        c=np.clip(displacement, 0.0, vmax),
        s=choose_point_size(len(output)),
        cmap="viridis",
        alpha=0.80,
        linewidths=0,
        rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("MLTD mapping displacement")
    ax.set_title("Cell-level influence of MLTD mapping")
    ax.set_xlabel("Native UMAP 1")
    ax.set_ylabel("Native UMAP 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure_both(fig, OUT_DIR / "UMAP_MLTD_mapping_displacement")

    log("[SAVE] UMAP comparison figures in", OUT_DIR)


# =============================================================================
# 7. MAIN
# =============================================================================


def main() -> None:
    required = [
        INPUT_MANIFEST,
        CELL_MAP_TSV,
        ADJUSTED_Z_NPY,
        PROFILE_PT_CSV,
        CELL_PT_CSV,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest, counts_path, meta_path, gene_names = load_manifest()
    cell_map_raw = standardize_cell_map(CELL_MAP_TSV)
    cell_map = align_cell_map_to_manifest(cell_map_raw, manifest)

    sample_labels = sorted(
        cell_map["sample_id"].astype(str).unique().tolist(),
        key=natural_sort_key,
    )
    if len(sample_labels) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Cell map contains {len(sample_labels)} samples, expected "
            f"{EXPECTED_SAMPLE_COUNT}: {sample_labels}"
        )
    if any(extract_sample_number(x) == 7 for x in sample_labels):
        raise ValueError("Sample-7 unexpectedly entered the UMAP input.")

    adjusted_z = np.load(ADJUSTED_Z_NPY, mmap_mode="r")
    if adjusted_z.shape[0] != len(cell_map):
        raise ValueError(
            f"Adjusted Z rows {adjusted_z.shape[0]} != cell map rows {len(cell_map)}"
        )

    log("=" * 110)
    log("[RUN] Native-expression UMAP vs MLTD-mapped UMAP")
    log("[CELLS]", len(cell_map))
    log("[SAMPLES]", sample_labels)
    log("[RULE] Native counts remain unchanged; MLTD acts only on representation.")
    log("[OUT]", OUT_DIR)
    log("=" * 110)

    # -------------------------------------------------------------------------
    # A. Native RNA representation and UMAP.
    # -------------------------------------------------------------------------
    native_pca, native_umap, native_leiden = load_or_build_native_representation(
        cell_map=cell_map,
        counts_path=counts_path,
        gene_names=gene_names,
    )
    if native_pca.shape[0] != len(cell_map) or native_umap.shape[0] != len(cell_map):
        raise ValueError("Native PCA/UMAP rows do not match modeled cells.")

    # -------------------------------------------------------------------------
    # B. MLTD-mapped dependency-aware representation and UMAP.
    # -------------------------------------------------------------------------
    mapped_features = map_mltd_features_to_cells(cell_map)
    stageb_pca = compress_stageb_adjusted_z(
        ADJUSTED_Z_NPY,
        cell_map["sample_id"].astype(str).to_numpy(),
    )
    representation, representation_meta = build_mltd_mapped_representation(
        native_pca=native_pca,
        stageb_pca=stageb_pca,
        mapped_features=mapped_features,
    )
    mltd_umap_raw, mltd_leiden = run_mltd_umap(representation, cell_map)

    if ALIGN_MAPPED_UMAP_TO_NATIVE_FOR_DISPLAY:
        mltd_umap = align_umap_to_reference(native_umap, mltd_umap_raw)
    else:
        mltd_umap = mltd_umap_raw.copy()

    # -------------------------------------------------------------------------
    # C. Export matched cell coordinates and direct mapping displacement.
    # -------------------------------------------------------------------------
    output = cell_map[
        [
            "cell_key",
            "sample_id",
            "cell_id",
            "row_index",
            "new_cluster",
            "profile_key",
        ]
    ].copy()
    output["native_umap_1"] = native_umap[:, 0]
    output["native_umap_2"] = native_umap[:, 1]
    output["mltd_umap_raw_1"] = mltd_umap_raw[:, 0]
    output["mltd_umap_raw_2"] = mltd_umap_raw[:, 1]
    output["mltd_umap_1"] = mltd_umap[:, 0]
    output["mltd_umap_2"] = mltd_umap[:, 1]

    if native_leiden is not None:
        output["native_leiden"] = np.asarray(native_leiden).astype(str)
    if mltd_leiden is not None:
        output["mltd_leiden"] = np.asarray(mltd_leiden).astype(str)

    output["mltd_mapping_displacement"] = np.sqrt(
        (output["mltd_umap_1"] - output["native_umap_1"]) ** 2
        + (output["mltd_umap_2"] - output["native_umap_2"]) ** 2
    ).astype(np.float32)

    for column in (
        COMMUNICATION_FEATURES
        + TEMPORAL_PROFILE_FEATURES
        + TEMPORAL_CELL_FEATURES
    ):
        if column in mapped_features.columns:
            output[column] = mapped_features[column].to_numpy()

    output.to_csv(OUT_COORD_TSV, sep="\t", index=False)
    log("[SAVE]", OUT_COORD_TSV)

    displacement_summary = (
        output.groupby(["sample_id", "new_cluster"], as_index=False)
        .agg(
            n_cells=("cell_key", "count"),
            mean_mapping_displacement=("mltd_mapping_displacement", "mean"),
            median_mapping_displacement=("mltd_mapping_displacement", "median"),
            q90_mapping_displacement=(
                "mltd_mapping_displacement",
                lambda x: float(np.quantile(x, 0.90)),
            ),
        )
        .sort_values(
            ["mean_mapping_displacement", "sample_id", "new_cluster"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    displacement_summary.to_csv(OUT_DISPLACEMENT_TSV, sep="\t", index=False)
    log("[SAVE]", OUT_DISPLACEMENT_TSV)

    # Marker genes for annotation. Groups may come from the original, native
    # Leiden or MLTD Leiden labels, but expression is always native measured RNA.
    marker_cluster_counts = export_cluster_marker_excel(
        output=output,
        counts_path=counts_path,
        meta_path=meta_path,
        gene_names=gene_names,
        sample_labels=sample_labels,
    )

    # Lightweight h5ad: no modified expression matrix. Both UMAPs and the
    # dependency-aware representation are stored in obsm.
    obs = output[
        [
            "cell_key",
            "sample_id",
            "cell_id",
            "row_index",
            "new_cluster",
            "profile_key",
            "mltd_mapping_displacement",
        ]
        + (["native_leiden"] if "native_leiden" in output.columns else [])
        + (["mltd_leiden"] if "mltd_leiden" in output.columns else [])
    ].copy()
    obs.index = pd.Index(obs["cell_key"].astype(str), name="cell_key_index")
    combined = ad.AnnData(obs=obs)
    combined.obsm["X_pca_native"] = native_pca[:, : min(N_NATIVE_PCS, native_pca.shape[1])]
    combined.obsm["X_umap_native"] = native_umap
    combined.obsm["X_mltd_mapped"] = representation
    combined.obsm["X_umap_mltd_raw"] = mltd_umap_raw
    combined.obsm["X_umap_mltd"] = mltd_umap
    combined.uns["mltd_mapping_definition"] = representation_meta
    combined.uns["native_matrix_modified"] = False
    safe_write_h5ad(combined, OUT_H5AD)

    make_plots(output)

    summary = {
        "dataset_name": DATASET_NAME,
        "n_cells": int(len(output)),
        "n_samples": int(len(sample_labels)),
        "samples": sample_labels,
        "input_manifest": str(INPUT_MANIFEST),
        "counts_matrix": str(counts_path),
        "meta_npz": str(meta_path),
        "cell_map": str(CELL_MAP_TSV),
        "native_h5ad_reused": bool(NATIVE_H5AD.exists()),
        "adjusted_z": str(ADJUSTED_Z_NPY),
        "profile_pseudotime": str(PROFILE_PT_CSV),
        "cell_pseudotime": str(CELL_PT_CSV),
        "native_matrix_modified": False,
        "cluster_marker_expression_source": "native measured RNA",
        "cluster_marker_top_n": MARKER_TOP_N,
        "marker_cluster_counts": marker_cluster_counts,
        "mapped_representation": representation_meta,
        "umap_parameters": {
            "n_neighbors": N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "spread": UMAP_SPREAD,
            "seed": SEED,
            "display_procrustes_alignment": ALIGN_MAPPED_UMAP_TO_NATIVE_FOR_DISPLAY,
        },
        "outputs": {
            "coordinates_tsv": str(OUT_COORD_TSV),
            "mapped_representation_npy": str(OUT_REP_NPY),
            "lightweight_h5ad": str(OUT_H5AD),
            "displacement_summary_tsv": str(OUT_DISPLACEMENT_TSV),
            "cluster_marker_excel": str(OUT_MARKER_XLSX),
            "figures_dir": str(OUT_DIR),
        },
    }
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    log("[SAVE]", OUT_SUMMARY_JSON)

    log("\n" + "=" * 110)
    log("[DONE] Native and MLTD-mapped integrated UMAPs are in:")
    log(OUT_DIR.resolve())
    log("[INTERPRETATION]")
    log(" - Native UMAP: measured RNA expression only.")
    log(" - MLTD UMAP: native PCA anchor + mapped dependency/communication/time layers.")
    log(" - Original counts were never replaced or modified.")
    log(" - UMAP cluster IDs are shown only in the legend, not over the points.")
    log(" - Cluster Top1000 native-RNA marker genes:", OUT_MARKER_XLSX)
    log("=" * 110)


if __name__ == "__main__":
    main()
