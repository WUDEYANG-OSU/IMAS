from __future__ import annotations

# =============================================================================
# 10.4_pseudobulk_log2_Welch_MLTD_tumour_vs_normal_volcano.py
# =============================================================================
# Purpose
# -------
# Pure-Python sample-level pseudobulk differential-expression analysis:
#
#   eight retained kidney-cancer samples vs independent normal-kidney samples
#
# Workflow
# --------
# 1. Sum raw single-cell counts within each biological sample.
# 2. Intersect normal and tumour gene vocabularies.
# 3. Build a sample-by-gene pseudobulk raw-count matrix.
# 4. Normalize pseudobulk counts by the positive-count median-ratio method.
# 5. Test tumour vs normal on log2-normalized sample values using Welch t.
# 6. Plot a volcano:
#       x = log2 fold change of normalized pseudobulk means
#       y = -log10(sample-level Welch P value)
# 7. Overlay genes selected by the latest MLTD-mapped heatmap-v2 workflow.
# 8. Export a pseudobulk sample PCA for cohort/batch inspection.
#
# IMPORTANT
# ---------
# - This script does not call R, Rscript or PyDESeq2.
# - Cells are not treated as biological replicates.
# - Tumour biological replicates are Sample-1–6, Sample-8 and Sample-9.
# - Normal biological replicates are recovered from normal manifest column
#   "original_sample_id", retained when GSE202109_ALL was assembled.
# - MLTD does not replace RNA counts. MLTD-selected genes are only highlighted.
# - No P-value or y-axis cap is imposed. The y range is determined by the test.
# - Normal and tumour originate from different cohorts, so condition remains
#   confounded with cohort/batch even after pseudobulk correction.
# =============================================================================

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import t as student_t


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE = get_project_root()
DATASET_NAME = "scVerifyKC-Kidney-cancer-8samples-exclude-sample7-joint"
NORMAL_DATASET_ID = "GSE202109_ALL"

# -------------------------
# Tumour source
# -------------------------
TUMOUR_RUN_DIR = (
    BASE
    / "runs"
    / "stageB_RNA_only_Kidney_cancer_8samples_exclude_sample7_joint"
)
TUMOUR_MANIFEST = TUMOUR_RUN_DIR / "input_manifest_excluding_sample7.tsv"
TUMOUR_SAMPLE_COLUMN = "sample_id"
TUMOUR_ROW_COLUMN = "row_index"

EXPECTED_TUMOUR_SAMPLES = [
    "Kidney-cancer-Sample-1",
    "Kidney-cancer-Sample-2",
    "Kidney-cancer-Sample-3",
    "Kidney-cancer-Sample-4",
    "Kidney-cancer-Sample-5",
    "Kidney-cancer-Sample-6",
    "Kidney-cancer-Sample-8",
    "Kidney-cancer-Sample-9",
]
EXPECTED_EXCLUDED_SAMPLE_NUMBERS = {7}

# -------------------------
# Normal source
# -------------------------
NORMAL_ROOT = (
    BASE
    / "LLM-WorldWord"
    / "Independent-Verify-NormalKidney"
    / "RNA_all_gene_input_matrices_per_sample_pureR_v2_allgroups"
    / NORMAL_DATASET_ID
    / "processed_RNA_all_genes"
)
NORMAL_MANIFEST = NORMAL_ROOT / "manifest_rna.tsv"
NORMAL_COUNTS = NORMAL_ROOT / f"{NORMAL_DATASET_ID}.X_rna_counts_dense.npy"
NORMAL_META = NORMAL_ROOT / f"{NORMAL_DATASET_ID}.rna_meta.npz"
NORMAL_SAMPLE_COLUMN = "original_sample_id"
NORMAL_ROW_COLUMN = "row_index"

# -------------------------
# MLTD mapped-gene overlay
# -------------------------
RUN_STAGEB_DATASET_DIR = TUMOUR_RUN_DIR / DATASET_NAME
HEATMAP_PARENT = RUN_STAGEB_DATASET_DIR
HEATMAP_DIR_PATTERN = (
    "EIGHT_SAMPLES_NO_SAMPLE7_native_vs_MLTD_cluster_normal_heatmaps_v2*"
)
REQUIRE_MLTD_GENE_OVERLAY = True

# The old version used only selected_genes_final.csv. That file is the compact
# 72-gene set chosen specifically for heatmap display, so the volcano legend
# could never contain more than 72 MLTD genes. For a volcano overlay, use the
# broader MLTD-supported candidate pool instead.
#
# Supported values:
#   "candidate_pool" -> candidate_gene_pool.csv, filtered by cross-sample MLTD
#                       support (recommended for the volcano)
#   "heatmap_final"  -> selected_genes_final.csv, the compact heatmap gene set
MLTD_GENE_SOURCE = "candidate_pool"

# Cross-sample support thresholds used only when MLTD_GENE_SOURCE is
# "candidate_pool". These match the broad candidate-selection logic used in
# the heatmap workflow, without restricting the overlay to the final 72 rows.
MLTD_MIN_SAMPLES_PRESENT = 3
MLTD_MIN_DIRECTION_CONSISTENCY = 0.55
MLTD_REQUIRE_POSITIVE_CONSENSUS_NDS = True

# None keeps every candidate passing the thresholds. Set an integer such as
# 300 or 500 only when a deliberately smaller overlay is desired.
MLTD_HIGHLIGHT_TOP_N: int | None = None

# -------------------------
# Output
# -------------------------
PREFERRED_OUT_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_pseudobulk_log2_Welch_tumour_vs_normal_broader_MLTD_v1"
)

# -------------------------
# Pseudobulk construction/QC
# -------------------------
COUNT_CHUNK_CELLS = 256
MIN_CELLS_PER_PSEUDOBULK = 20
MIN_LIBRARY_SIZE = 1000
MIN_REPLICATES_PER_CONDITION = 2

# -------------------------
# Sample-level statistical settings
# -------------------------
# Raw pseudobulk counts are normalized with a positive-count median-ratio
# method, then tested on log2-normalized sample values with Welch's t test.
# This uses biological samples as replicates and lets between-sample variance
# determine the P values. No display cap and no permutation-resolution cap are
# used.
SIZE_FACTOR_METHOD = "poscounts_median_ratio"
LOG2_TRANSFORM_PSEUDOCOUNT = 0.5
LOG2FC_PSEUDOCOUNT = 0.5
WELCH_VARIANCE_EPS = 1e-12

# Gene prefilter before the sample-level test.
MIN_TOTAL_COUNT = 15
MIN_COUNT_IN_SAMPLE = 10
MIN_SAMPLES_WITH_MIN_COUNT = 2

# Volcano significance.
# The volcano y-axis always uses the raw sample-level Welch P value.
SIGNIFICANCE_COLUMN = "pvalue"  # "pvalue" or "padj"
PVALUE_THRESHOLD = 0.05
PADJ_THRESHOLD = 0.05
ABS_LOG2FC_THRESHOLD = 0.50

# No y clipping. X limits use the actual finite range, not a fixed cap.
Y_DISPLAY_CAP: float | None = None
X_MARGIN_FRACTION = 0.06
MIN_X_LIMIT = 2.0

# # PCA settings.
PCA_TOP_VARIABLE_GENES = 2000

# Labels/style.
LABEL_TOP_N = 18
LABEL_MLTD_ONLY = True
FIG_DPI = 320
FIGSIZE = (9.2, 7.8)
COLOR_DOWN = "#64C2D7"
COLOR_NOTSIG = "#BDBDBD"
COLOR_UP = "#F4A38C"
COLOR_MLTD = "#E31A1C"
POINT_SIZE = 20
POINT_ALPHA = 0.72
MLTD_POINT_SIZE = 42


# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================


def log(*args) -> None:
    print(*args, flush=True)


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def extract_sample_number(value: str) -> int | None:
    match = re.search(r"Sample[-_ ]?(\d+)", str(value), flags=re.I)
    return int(match.group(1)) if match else None


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def read_table_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def resolve_new_output_dir(preferred: Path) -> Path:
    preferred = Path(preferred)
    if not preferred.exists():
        preferred.mkdir(parents=True, exist_ok=False)
        return preferred
    try:
        if not any(preferred.iterdir()):
            return preferred
    except OSError:
        pass
    for run_number in range(2, 1000):
        candidate = preferred.with_name(f"{preferred.name}_run{run_number:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Could not allocate output directory beside {preferred}")


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def make_unique_gene_map(genes: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, gene in enumerate(map(str, genes)):
        if gene not in mapping:
            mapping[gene] = index
    return mapping


# =============================================================================
# 2. LOAD SOURCES
# =============================================================================


def load_tumour_source() -> tuple[pd.DataFrame, Path, Path, np.ndarray]:
    manifest = read_table_auto(TUMOUR_MANIFEST)
    ensure_columns(
        manifest,
        [TUMOUR_SAMPLE_COLUMN, TUMOUR_ROW_COLUMN, "counts_npy", "meta_npz"],
        TUMOUR_MANIFEST.name,
    )
    manifest[TUMOUR_SAMPLE_COLUMN] = (
        manifest[TUMOUR_SAMPLE_COLUMN].astype(str).str.strip()
    )
    manifest[TUMOUR_ROW_COLUMN] = pd.to_numeric(
        manifest[TUMOUR_ROW_COLUMN], errors="raise"
    ).astype(np.int64)

    observed_samples = sorted(
        manifest[TUMOUR_SAMPLE_COLUMN].unique().tolist(), key=natural_sort_key
    )
    expected_samples = sorted(EXPECTED_TUMOUR_SAMPLES, key=natural_sort_key)
    if observed_samples != expected_samples:
        raise ValueError(
            "Tumour samples differ from the completed eight-sample set.\n"
            f"Expected: {expected_samples}\nObserved: {observed_samples}"
        )
    if any(
        extract_sample_number(sample) in EXPECTED_EXCLUDED_SAMPLE_NUMBERS
        for sample in observed_samples
    ):
        raise ValueError("Sample-7 unexpectedly appears in tumour manifest")

    counts_paths = sorted(set(manifest["counts_npy"].astype(str)))
    meta_paths = sorted(set(manifest["meta_npz"].astype(str)))
    if len(counts_paths) != 1 or len(meta_paths) != 1:
        raise ValueError(
            "Expected one combined tumour count matrix and one metadata file; "
            f"counts={counts_paths}; meta={meta_paths}"
        )
    counts_path = Path(counts_paths[0])
    meta_path = Path(meta_paths[0])
    if not counts_path.exists():
        raise FileNotFoundError(counts_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    metadata = np.load(meta_path, allow_pickle=True)
    if "gene_names" not in metadata:
        raise KeyError(f"{meta_path.name} missing gene_names")
    genes = np.asarray(metadata["gene_names"]).astype(str)

    matrix = np.load(counts_path, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[1] != len(genes):
        raise ValueError(
            f"Tumour matrix/gene mismatch: matrix={matrix.shape}, genes={len(genes)}"
        )
    rows = manifest[TUMOUR_ROW_COLUMN].to_numpy(dtype=np.int64)
    if rows.min() < 0 or rows.max() >= matrix.shape[0]:
        raise IndexError("Tumour manifest row indices exceed tumour matrix")
    if manifest[TUMOUR_ROW_COLUMN].duplicated().any():
        raise ValueError("Tumour manifest contains duplicate row_index values")

    log("[TUMOUR COUNTS]", counts_path, matrix.shape)
    log("[TUMOUR SAMPLES]", observed_samples)
    return manifest, counts_path, meta_path, genes


def load_normal_source() -> tuple[pd.DataFrame, Path, Path, np.ndarray]:
    manifest = read_table_auto(NORMAL_MANIFEST)
    ensure_columns(
        manifest,
        [NORMAL_SAMPLE_COLUMN, NORMAL_ROW_COLUMN],
        NORMAL_MANIFEST.name,
    )
    if not NORMAL_COUNTS.exists():
        raise FileNotFoundError(NORMAL_COUNTS)
    if not NORMAL_META.exists():
        raise FileNotFoundError(NORMAL_META)

    manifest[NORMAL_SAMPLE_COLUMN] = (
        manifest[NORMAL_SAMPLE_COLUMN].astype(str).str.strip()
    )
    manifest[NORMAL_ROW_COLUMN] = pd.to_numeric(
        manifest[NORMAL_ROW_COLUMN], errors="raise"
    ).astype(np.int64)

    invalid = manifest[NORMAL_SAMPLE_COLUMN].isin(
        ["", "nan", "None", "<NA>", NORMAL_DATASET_ID]
    )
    if invalid.any():
        examples = manifest.loc[invalid, NORMAL_SAMPLE_COLUMN].head().tolist()
        raise ValueError(
            f"Normal column {NORMAL_SAMPLE_COLUMN} contains pooled/invalid labels: "
            f"{examples}. Independent normal sample IDs are required."
        )

    normal_samples = sorted(
        manifest[NORMAL_SAMPLE_COLUMN].unique().tolist(), key=natural_sort_key
    )
    if len(normal_samples) < MIN_REPLICATES_PER_CONDITION:
        raise ValueError(
            f"Only {len(normal_samples)} normal biological replicate(s) found: "
            f"{normal_samples}"
        )

    metadata = np.load(NORMAL_META, allow_pickle=True)
    if "gene_names" not in metadata:
        raise KeyError(f"{NORMAL_META.name} missing gene_names")
    genes = np.asarray(metadata["gene_names"]).astype(str)

    matrix = np.load(NORMAL_COUNTS, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[1] != len(genes):
        raise ValueError(
            f"Normal matrix/gene mismatch: matrix={matrix.shape}, genes={len(genes)}"
        )
    rows = manifest[NORMAL_ROW_COLUMN].to_numpy(dtype=np.int64)
    if rows.min() < 0 or rows.max() >= matrix.shape[0]:
        raise IndexError("Normal manifest row indices exceed normal matrix")
    if manifest[NORMAL_ROW_COLUMN].duplicated().any():
        raise ValueError("Normal manifest contains duplicate row_index values")

    log("[NORMAL COUNTS]", NORMAL_COUNTS, matrix.shape)
    log("[NORMAL BIOLOGICAL SAMPLES]", normal_samples)
    return manifest, NORMAL_COUNTS, NORMAL_META, genes


# =============================================================================
# 3. PSEUDOBULK AGGREGATION
# =============================================================================


def aggregate_pseudobulk(
    manifest: pd.DataFrame,
    counts_path: Path,
    sample_column: str,
    row_column: str,
    gene_indices: np.ndarray,
    condition: str,
    sample_prefix: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    samples = sorted(
        manifest[sample_column].astype(str).unique(), key=natural_sort_key
    )
    sample_to_code = {sample: index for index, sample in enumerate(samples)}
    codes = (
        manifest[sample_column]
        .astype(str)
        .map(sample_to_code)
        .to_numpy(dtype=np.int64)
    )
    rows = manifest[row_column].to_numpy(dtype=np.int64)

    matrix = np.load(counts_path, mmap_mode="r")
    pseudobulk = np.zeros((len(samples), len(gene_indices)), dtype=np.float64)
    n_cells = np.bincount(codes, minlength=len(samples)).astype(np.int64)

    log(
        f"[PSEUDOBULK {condition}] cells={len(rows)}, samples={len(samples)}, "
        f"genes={len(gene_indices)}"
    )
    for start in range(0, len(rows), COUNT_CHUNK_CELLS):
        end = min(start + COUNT_CHUNK_CELLS, len(rows))
        chunk_rows = rows[start:end]
        chunk_codes = codes[start:end]
        block = np.asarray(
            matrix[chunk_rows[:, None], gene_indices[None, :]],
            dtype=np.float64,
        )
        np.add.at(pseudobulk, chunk_codes, block)
        del block
        if start == 0 or end == len(rows) or end % 10000 < COUNT_CHUNK_CELLS:
            log(f"[PSEUDOBULK {condition}] {end}/{len(rows)} cells")

    rounded = np.rint(pseudobulk)
    max_fractional = float(np.max(np.abs(pseudobulk - rounded)))
    if max_fractional > 1e-3:
        raise ValueError(
            f"{condition} source does not appear to be raw integer counts; "
            f"maximum fractional deviation={max_fractional}"
        )
    pseudobulk = rounded.astype(np.int64)
    library_sizes = pseudobulk.sum(axis=1).astype(np.int64)

    sample_meta = pd.DataFrame(
        {
            "sample_id": [f"{sample_prefix}{sample}" for sample in samples],
            "original_sample_id": samples,
            "condition": condition,
            "n_cells": n_cells,
            "library_size": library_sizes,
        }
    )

    keep = (
        (sample_meta["n_cells"].to_numpy(dtype=int) >= MIN_CELLS_PER_PSEUDOBULK)
        & (
            sample_meta["library_size"].to_numpy(dtype=np.int64)
            >= MIN_LIBRARY_SIZE
        )
    )
    if not np.all(keep):
        log("[PSEUDOBULK DROP]\n", sample_meta.loc[~keep].to_string(index=False))
        pseudobulk = pseudobulk[keep]
        sample_meta = sample_meta.loc[keep].reset_index(drop=True)

    if len(sample_meta) < MIN_REPLICATES_PER_CONDITION:
        raise RuntimeError(
            f"{condition} has only {len(sample_meta)} valid pseudobulk replicates"
        )
    return pseudobulk, sample_meta


# =============================================================================
# 4. MLTD GENE OVERLAY
# =============================================================================


def find_latest_heatmap_run(parent_dir: Path) -> Path | None:
    candidates = [path for path in parent_dir.glob(HEATMAP_DIR_PATTERN) if path.is_dir()]
    if not candidates:
        if REQUIRE_MLTD_GENE_OVERLAY:
            raise FileNotFoundError(
                f"No MLTD heatmap-v2 directory found under {parent_dir} "
                f"matching {HEATMAP_DIR_PATTERN}"
            )
        return None

    def key(path: Path):
        match = re.search(r"_run(\d+)$", path.name)
        run_number = int(match.group(1)) if match else 1
        return run_number, path.stat().st_mtime

    return sorted(candidates, key=key)[-1]


def load_mltd_genes(heatmap_dir: Path | None) -> pd.DataFrame:
    """Load a volcano-appropriate MLTD-supported gene set.

    selected_genes_final.csv is intentionally small because it was produced for
    heatmap display. candidate_gene_pool.csv retains the broader cross-sample
    MLTD-supported set and is therefore the preferred volcano overlay source.
    """
    if heatmap_dir is None:
        return pd.DataFrame(columns=["gene"])

    selected_dir = heatmap_dir / "01_selected_genes"

    if MLTD_GENE_SOURCE == "candidate_pool":
        path = selected_dir / "candidate_gene_pool.csv"
    elif MLTD_GENE_SOURCE == "heatmap_final":
        path = selected_dir / "selected_genes_final.csv"
    else:
        raise ValueError(
            "MLTD_GENE_SOURCE must be 'candidate_pool' or 'heatmap_final'; "
            f"got {MLTD_GENE_SOURCE!r}"
        )

    if not path.exists():
        if REQUIRE_MLTD_GENE_OVERLAY:
            raise FileNotFoundError(path)
        return pd.DataFrame(columns=["gene"])

    table = pd.read_csv(path, low_memory=False)
    ensure_columns(table, ["gene"], path.name)
    table["gene"] = table["gene"].astype(str)
    table = table.drop_duplicates("gene", keep="first").copy()

    if MLTD_GENE_SOURCE == "candidate_pool":
        required_support_columns = [
            "consensus_NDS",
            "n_samples_present",
            "direction_consistency",
        ]
        ensure_columns(table, required_support_columns, path.name)

        for column in [
            "consensus_NDS",
            "consensus_rank",
            "n_samples_present",
            "direction_consistency",
            "candidate_rank",
        ]:
            if column in table.columns:
                table[column] = pd.to_numeric(table[column], errors="coerce")

        keep = (
            table["consensus_NDS"].notna()
            & (table["n_samples_present"] >= MLTD_MIN_SAMPLES_PRESENT)
            & (
                table["direction_consistency"]
                >= MLTD_MIN_DIRECTION_CONSISTENCY
            )
        )
        if MLTD_REQUIRE_POSITIVE_CONSENSUS_NDS:
            keep &= table["consensus_NDS"] > 0
        table = table.loc[keep].copy()

        sort_columns: list[str] = []
        ascending: list[bool] = []
        if "consensus_rank" in table.columns:
            sort_columns.append("consensus_rank")
            ascending.append(True)
        if "consensus_NDS" in table.columns:
            sort_columns.append("consensus_NDS")
            ascending.append(False)
        if "candidate_rank" in table.columns:
            sort_columns.append("candidate_rank")
            ascending.append(True)
        if sort_columns:
            table = table.sort_values(
                sort_columns,
                ascending=ascending,
                na_position="last",
            )
    else:
        if "final_gene_score" in table.columns:
            table["final_gene_score"] = pd.to_numeric(
                table["final_gene_score"], errors="coerce"
            )
            table = table.sort_values(
                "final_gene_score",
                ascending=False,
                na_position="last",
            )

    if MLTD_HIGHLIGHT_TOP_N is not None:
        table = table.head(int(MLTD_HIGHLIGHT_TOP_N)).copy()

    table["MLTD_overlay_source"] = MLTD_GENE_SOURCE
    table["MLTD_overlay_input_file"] = str(path)
    return table.reset_index(drop=True)


# =============================================================================
# 5. NORMALIZATION AND SAMPLE-LEVEL WELCH TEST
# =============================================================================


def prefilter_count_matrix(
    counts_samples_by_genes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = counts_samples_by_genes.copy()
    total_count = counts.sum(axis=0)
    n_samples_min_count = (counts >= MIN_COUNT_IN_SAMPLE).sum(axis=0)
    keep = (
        (total_count >= MIN_TOTAL_COUNT)
        & (n_samples_min_count >= MIN_SAMPLES_WITH_MIN_COUNT)
    )
    qc = pd.DataFrame(
        {
            "gene": counts.columns.astype(str),
            "total_count": total_count.to_numpy(dtype=np.int64),
            "n_samples_count_ge_threshold": n_samples_min_count.to_numpy(dtype=np.int64),
            "retained": keep.to_numpy(dtype=bool),
        }
    )
    filtered = counts.loc[:, keep]
    if filtered.shape[1] < 100:
        raise RuntimeError(
            f"Only {filtered.shape[1]} genes retained after count filtering"
        )
    return filtered.astype(np.int64), qc


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction with NaN preservation."""
    pvalues = np.asarray(pvalues, dtype=float)
    output = np.full(pvalues.shape, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not np.any(finite):
        return output

    p = np.clip(pvalues[finite], 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    output[finite] = restored
    return output


def estimate_poscounts_size_factors(counts: np.ndarray) -> np.ndarray:
    """
    Estimate DESeq2-style positive-count median-ratio size factors.

    For each gene, the geometric mean is calculated from its positive sample
    counts. For each sample, the size factor is the median of count/geometric
    mean ratios for genes with positive counts in that sample.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"Counts must be sample-by-gene, observed {counts.shape}")
    if np.any(counts < 0):
        raise ValueError("Counts contain negative values")

    positive = counts > 0
    n_positive = positive.sum(axis=0)

    sum_log = np.zeros(counts.shape[1], dtype=np.float64)
    for sample_index in range(counts.shape[0]):
        row_positive = positive[sample_index]
        sum_log[row_positive] += np.log(counts[sample_index, row_positive])

    geometric_mean = np.zeros(counts.shape[1], dtype=np.float64)
    valid_gene = n_positive > 0
    geometric_mean[valid_gene] = np.exp(
        sum_log[valid_gene] / n_positive[valid_gene]
    )

    size_factors = np.ones(counts.shape[0], dtype=np.float64)
    for sample_index in range(counts.shape[0]):
        valid = (
            valid_gene
            & (geometric_mean > 0)
            & np.isfinite(geometric_mean)
            & (counts[sample_index] > 0)
        )
        ratios = counts[sample_index, valid] / geometric_mean[valid]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if len(ratios) == 0:
            raise RuntimeError(
                f"No valid median-ratio genes for sample index {sample_index}"
            )
        size_factors[sample_index] = float(np.median(ratios))

    if np.any(~np.isfinite(size_factors)) or np.any(size_factors <= 0):
        raise RuntimeError(f"Invalid size factors: {size_factors}")

    # Geometric centring changes only the common scale, not group contrasts.
    centre = float(np.exp(np.mean(np.log(size_factors))))
    return size_factors / centre


def run_sample_level_welch(
    counts_samples_by_genes: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize pseudobulk raw counts and perform a two-sided Welch t test on
    log2-normalized values for each gene.

    The result table deliberately retains PyDESeq2-like column names where
    useful (baseMean, log2FoldChange, lfcSE, stat, pvalue, padj), so the
    downstream volcano workflow remains structurally similar to the supplied
    script.
    """
    counts = counts_samples_by_genes.copy().astype(np.int64)
    metadata = sample_metadata.copy().set_index("sample_id", drop=False)
    metadata = metadata.loc[counts.index].copy()
    metadata["condition"] = metadata["condition"].astype(str).str.lower().str.strip()

    observed_conditions = set(metadata["condition"])
    if observed_conditions != {"normal", "tumor"}:
        raise ValueError(
            f"Metadata condition must contain normal/tumor; observed={observed_conditions}"
        )

    raw = counts.to_numpy(dtype=np.float64)
    size_factor_values = estimate_poscounts_size_factors(raw)
    normalized = raw / size_factor_values[:, None]
    log2_normalized = np.log2(
        normalized + float(LOG2_TRANSFORM_PSEUDOCOUNT)
    )

    condition = metadata["condition"].to_numpy(dtype=str)
    normal_mask = condition == "normal"
    tumour_mask = condition == "tumor"
    n_normal = int(normal_mask.sum())
    n_tumour = int(tumour_mask.sum())

    if n_normal < MIN_REPLICATES_PER_CONDITION or n_tumour < MIN_REPLICATES_PER_CONDITION:
        raise RuntimeError(
            f"Welch test requires at least {MIN_REPLICATES_PER_CONDITION} replicates "
            f"per condition; normal={n_normal}, tumour={n_tumour}"
        )

    normal_values = log2_normalized[normal_mask]
    tumour_values = log2_normalized[tumour_mask]

    mean_log_normal = normal_values.mean(axis=0)
    mean_log_tumour = tumour_values.mean(axis=0)
    var_log_normal = normal_values.var(axis=0, ddof=1)
    var_log_tumour = tumour_values.var(axis=0, ddof=1)

    se2_normal = var_log_normal / float(n_normal)
    se2_tumour = var_log_tumour / float(n_tumour)
    se2 = se2_normal + se2_tumour
    standard_error = np.sqrt(np.maximum(se2, 0.0))

    mean_log_difference = mean_log_tumour - mean_log_normal
    statistic = np.divide(
        mean_log_difference,
        standard_error,
        out=np.zeros_like(mean_log_difference, dtype=np.float64),
        where=standard_error > WELCH_VARIANCE_EPS,
    )

    denominator_df = (
        (se2_normal * se2_normal) / float(n_normal - 1)
        + (se2_tumour * se2_tumour) / float(n_tumour - 1)
    )
    degrees_freedom = np.divide(
        se2 * se2,
        denominator_df,
        out=np.full_like(se2, float(n_normal + n_tumour - 2)),
        where=denominator_df > WELCH_VARIANCE_EPS,
    )
    degrees_freedom = np.maximum(degrees_freedom, 1.0)

    # For genes with exactly zero estimated variance in both groups:
    # - no group difference -> P=1;
    # - non-zero difference -> infinite t. This is a property of the data, not
    #   a graphical cap. Student-t log survival is used to avoid numerical
    #   underflow when the statistic is large.
    zero_se = standard_error <= WELCH_VARIANCE_EPS
    zero_se_difference = zero_se & (np.abs(mean_log_difference) > WELCH_VARIANCE_EPS)
    statistic[zero_se_difference] = np.sign(
        mean_log_difference[zero_se_difference]
    ) * np.inf

    abs_statistic = np.abs(statistic)
    log_pvalue = np.log(2.0) + student_t.logsf(
        abs_statistic,
        degrees_freedom,
    )
    log_pvalue = np.minimum(log_pvalue, 0.0)
    log_pvalue[zero_se & ~zero_se_difference] = 0.0

    minus_log10_pvalue = -log_pvalue / np.log(10.0)

    # Numeric pvalue is exported for conventional downstream use. The volcano
    # y coordinate is calculated directly from log_pvalue, so it is not capped
    # by a chosen P-value floor.
    minimum_log_float = float(np.log(np.finfo(np.float64).tiny))
    pvalue = np.exp(np.maximum(log_pvalue, minimum_log_float))
    pvalue[zero_se_difference] = 0.0
    padj = benjamini_hochberg(pvalue)

    mean_normalized_normal = normalized[normal_mask].mean(axis=0)
    mean_normalized_tumour = normalized[tumour_mask].mean(axis=0)
    log2_fold_change = np.log2(
        (mean_normalized_tumour + float(LOG2FC_PSEUDOCOUNT))
        / (mean_normalized_normal + float(LOG2FC_PSEUDOCOUNT))
    )

    result = pd.DataFrame(
        {
            "gene": counts.columns.astype(str),
            "baseMean": normalized.mean(axis=0),
            "log2FoldChange": log2_fold_change,
            "lfcSE": standard_error,
            "stat": statistic,
            "welch_df": degrees_freedom,
            "pvalue": pvalue,
            "padj": padj,
            "log_pvalue_natural": log_pvalue,
            "minus_log10_pvalue": minus_log10_pvalue,
            "mean_log2_normal": mean_log_normal,
            "mean_log2_tumour": mean_log_tumour,
            "mean_normalized_normal": mean_normalized_normal,
            "mean_normalized_tumour": mean_normalized_tumour,
        }
    )
    result = result.sort_values(
        ["pvalue", "log2FoldChange"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    normalized_counts = pd.DataFrame(
        normalized,
        index=counts.index,
        columns=counts.columns,
    )
    normalized_counts.index.name = "sample_id"

    log2_normalized_counts = pd.DataFrame(
        log2_normalized,
        index=counts.index,
        columns=counts.columns,
    )
    log2_normalized_counts.index.name = "sample_id"

    size_factors = pd.DataFrame(
        {
            "sample_id": counts.index.astype(str),
            "original_sample_id": metadata["original_sample_id"].astype(str).to_numpy(),
            "condition": metadata["condition"].astype(str).to_numpy(),
            "library_size": counts.sum(axis=1).to_numpy(dtype=np.int64),
            "size_factor": size_factor_values,
        }
    )

    result.to_csv(
        output_dir / "Welch_pseudobulk_tumour_vs_normal.csv",
        index=False,
    )
    normalized_counts.reset_index().to_csv(
        output_dir / "median_ratio_normalized_counts_sample_by_gene.csv",
        index=False,
    )
    log2_normalized_counts.reset_index().to_csv(
        output_dir / "log2_normalized_counts_sample_by_gene.csv",
        index=False,
    )
    size_factors.to_csv(
        output_dir / "median_ratio_size_factors.csv",
        index=False,
    )
    log("[SAVE] sample-level Welch results and normalized matrices")

    return result, normalized_counts, log2_normalized_counts, size_factors


# =============================================================================
# 6. VOLCANO AND PCA
# =============================================================================


def prepare_volcano_table(
    result: pd.DataFrame,
    mltd_genes: pd.DataFrame,
) -> pd.DataFrame:
    frame = result.copy()
    numeric_columns = [
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "stat",
        "welch_df",
        "pvalue",
        "padj",
        "minus_log10_pvalue",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["gene"] = frame["gene"].astype(str)
    frame = frame.dropna(
        subset=["log2FoldChange", "pvalue", "minus_log10_pvalue"]
    ).copy()

    mltd_set = set(mltd_genes["gene"].astype(str))
    frame["is_MLTD_selected"] = frame["gene"].isin(mltd_set)

    if SIGNIFICANCE_COLUMN == "pvalue":
        significant_stat = frame["pvalue"] <= PVALUE_THRESHOLD
    elif SIGNIFICANCE_COLUMN == "padj":
        significant_stat = frame["padj"].fillna(1.0) <= PADJ_THRESHOLD
    else:
        raise ValueError(f"Unsupported SIGNIFICANCE_COLUMN: {SIGNIFICANCE_COLUMN}")

    frame["is_significant"] = significant_stat & (
        np.abs(frame["log2FoldChange"]) >= ABS_LOG2FC_THRESHOLD
    )
    frame["regulation"] = "Not sig"
    frame.loc[
        frame["is_significant"] & (frame["log2FoldChange"] > 0),
        "regulation",
    ] = "Up regulated"
    frame.loc[
        frame["is_significant"] & (frame["log2FoldChange"] < 0),
        "regulation",
    ] = "Down regulated"

    # No clipping: plotted y equals the actual sample-level test statistic.
    frame["volcano_y_display"] = frame["minus_log10_pvalue"]
    frame["y_was_clipped"] = False

    mapped_extra = [
        column
        for column in [
            "final_gene_order",
            "final_gene_score",
            "consensus_NDS",
            "native_cluster_range_log2fc",
            "mltd_cluster_range_log2fc",
        ]
        if column in mltd_genes.columns
    ]
    if mapped_extra:
        frame = frame.merge(
            mltd_genes[["gene"] + mapped_extra],
            on="gene",
            how="left",
        )

    frame["label_score"] = (
        frame["minus_log10_pvalue"]
        * (np.abs(frame["log2FoldChange"]) + 0.25)
        * np.where(frame["is_MLTD_selected"], 2.0, 1.0)
    )
    return frame


def choose_balanced_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if LABEL_MLTD_ONLY:
        source = frame[
            frame["is_MLTD_selected"] & frame["is_significant"]
        ].copy()
        if len(source) == 0:
            source = frame[frame["is_MLTD_selected"]].copy()
    else:
        source = frame[frame["is_significant"]].copy()

    per_side = max(1, LABEL_TOP_N // 2)
    left = (
        source[source["log2FoldChange"] < 0]
        .sort_values(["label_score", "pvalue"], ascending=[False, True])
        .head(per_side)
    )
    right = (
        source[source["log2FoldChange"] > 0]
        .sort_values(["label_score", "pvalue"], ascending=[False, True])
        .head(per_side)
    )
    labels = pd.concat([left, right], ignore_index=True)

    if len(labels) < LABEL_TOP_N:
        used = set(labels["gene"].astype(str))
        remaining = (
            source[~source["gene"].astype(str).isin(used)]
            .sort_values(["label_score", "pvalue"], ascending=[False, True])
            .head(LABEL_TOP_N - len(labels))
        )
        labels = pd.concat([labels, remaining], ignore_index=True)

    return labels.head(LABEL_TOP_N)


def plot_volcano(frame: pd.DataFrame, output_base: Path) -> None:
    finite = frame[
        np.isfinite(frame["log2FoldChange"])
        & np.isfinite(frame["volcano_y_display"])
    ].copy()
    if len(finite) == 0:
        raise RuntimeError("No finite volcano coordinates")

    finite_x = finite["log2FoldChange"].to_numpy(dtype=float)
    absolute_x = float(np.max(np.abs(finite_x)))
    x_limit = max(MIN_X_LIMIT, absolute_x * (1.0 + X_MARGIN_FRACTION))

    finite_y = finite["volcano_y_display"].to_numpy(dtype=float)
    y_max = float(np.max(finite_y))
    y_limit = max(2.0, y_max * 1.08 + 0.05)

    figure, axis = plt.subplots(
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    plot_order = [
        ("Not sig", COLOR_NOTSIG, 1),
        ("Down regulated", COLOR_DOWN, 2),
        ("Up regulated", COLOR_UP, 2),
    ]
    counts_by_class: dict[str, int] = {}
    for group, color, zorder in plot_order:
        subset = finite[finite["regulation"] == group]
        counts_by_class[group] = int(len(subset))
        axis.scatter(
            subset["log2FoldChange"],
            subset["volcano_y_display"],
            s=POINT_SIZE,
            c=color,
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
            zorder=zorder,
            clip_on=True,
        )

    mapped = finite[finite["is_MLTD_selected"]].copy()
    axis.scatter(
        mapped["log2FoldChange"],
        mapped["volcano_y_display"],
        s=MLTD_POINT_SIZE,
        facecolors="none",
        edgecolors=COLOR_MLTD,
        linewidths=0.8,
        alpha=0.95,
        rasterized=True,
        zorder=4,
        clip_on=True,
    )

    axis.axvline(
        -ABS_LOG2FC_THRESHOLD,
        color="#777777",
        linestyle=":",
        linewidth=0.9,
    )
    axis.axvline(
        ABS_LOG2FC_THRESHOLD,
        color="#777777",
        linestyle=":",
        linewidth=0.9,
    )
    if SIGNIFICANCE_COLUMN == "pvalue":
        axis.axhline(
            -math.log10(PVALUE_THRESHOLD),
            color="#777777",
            linestyle=":",
            linewidth=0.9,
        )

    labels = choose_balanced_labels(finite)
    x_label_offset = max(0.015 * x_limit, 0.04)
    y_label_offset = max(0.018 * y_limit, 0.03)

    for _, record in labels.iterrows():
        x = float(record["log2FoldChange"])
        y = float(record["volcano_y_display"])
        is_right = x >= 0
        label_x = x - x_label_offset if is_right else x + x_label_offset
        label_x = float(np.clip(label_x, -0.97 * x_limit, 0.97 * x_limit))
        label_y = float(min(y + y_label_offset, 0.975 * y_limit))

        axis.text(
            label_x,
            label_y,
            str(record["gene"]),
            fontsize=8,
            color=COLOR_MLTD if bool(record["is_MLTD_selected"]) else "#444444",
            ha="right" if is_right else "left",
            va="bottom",
            clip_on=True,
            zorder=5,
        )

    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(-0.03, y_limit)
    axis.set_xlabel(
        r"$\log_{2}$(Fold Change), tumour vs normal",
        fontsize=13,
    )
    axis.set_ylabel(
        r"$-\log_{10}$(sample-level Welch P value)",
        fontsize=13,
    )
    axis.set_title(
        "Sample-level pseudobulk tumour versus normal\n"
        "Welch test on log2 median-ratio-normalized counts",
        fontsize=14,
        pad=35,
    )

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLOR_DOWN, markeredgecolor="none", markersize=8,
            label=f"Down regulated ({counts_by_class['Down regulated']})",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLOR_NOTSIG, markeredgecolor="none", markersize=8,
            label=f"Not sig ({counts_by_class['Not sig']})",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLOR_UP, markeredgecolor="none", markersize=8,
            label=f"Up regulated ({counts_by_class['Up regulated']})",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="none", markeredgecolor=COLOR_MLTD,
            markeredgewidth=1.0, markersize=8,
            label=f"MLTD supported ({len(mapped)})",
        ),
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=9,
        handlelength=1.0,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, linestyle="-", linewidth=0.45, alpha=0.24)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches='tight': labels are clipped to the axes and
    # constrained_layout controls the canvas, preventing right-side blank space.
    figure.savefig(output_base.with_suffix(".png"), dpi=FIG_DPI)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)

    log("[SAVE]", output_base.with_suffix(".png"))
    log("[SAVE]", output_base.with_suffix(".svg"))
    log("[SAVE]", output_base.with_suffix(".pdf"))


def build_and_plot_pca(
    log2_normalized_counts: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    output_table: Path,
    output_base: Path,
) -> None:
    x = log2_normalized_counts.to_numpy(dtype=float)
    variances = np.var(x, axis=0, ddof=1)
    n_keep = min(PCA_TOP_VARIABLE_GENES, x.shape[1])
    top_indices = np.argsort(variances)[-n_keep:]

    pca_model = PCA(n_components=2, random_state=42)
    coordinates = pca_model.fit_transform(x[:, top_indices])

    meta = sample_metadata.set_index("sample_id").loc[
        log2_normalized_counts.index
    ].copy()
    pca_table = pd.DataFrame(
        {
            "sample_id": log2_normalized_counts.index.astype(str),
            "condition": meta["condition"].astype(str).to_numpy(),
            "original_sample_id": meta["original_sample_id"].astype(str).to_numpy(),
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "PC1_variance_percent": pca_model.explained_variance_ratio_[0] * 100.0,
            "PC2_variance_percent": pca_model.explained_variance_ratio_[1] * 100.0,
        }
    )
    pca_table.to_csv(output_table, index=False)

    figure, axis = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    for condition, color, marker in [
        ("normal", COLOR_DOWN, "o"),
        ("tumor", COLOR_UP, "s"),
    ]:
        subset = pca_table[pca_table["condition"] == condition]
        axis.scatter(
            subset["PC1"],
            subset["PC2"],
            s=70,
            c=color,
            marker=marker,
            alpha=0.88,
            edgecolors="#444444",
            linewidths=0.4,
            label=condition.capitalize(),
        )
        for _, record in subset.iterrows():
            axis.text(
                float(record["PC1"]),
                float(record["PC2"]),
                str(record["original_sample_id"]),
                fontsize=7,
                ha="left",
                va="bottom",
            )

    axis.set_xlabel(
        f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}%)"
    )
    axis.set_ylabel(
        f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}%)"
    )
    axis.set_title("Pseudobulk sample PCA")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)

    figure.savefig(output_base.with_suffix(".png"), dpi=FIG_DPI)
    figure.savefig(output_base.with_suffix(".svg"))
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)

    log("[SAVE]", output_table)
    log("[SAVE]", output_base.with_suffix(".png"))


# =============================================================================
# 7. MAIN
# =============================================================================


def main() -> None:
    out_dir = resolve_new_output_dir(PREFERRED_OUT_DIR)
    dir_input = out_dir / "01_pseudobulk_inputs"
    dir_stats = out_dir / "02_log2_Welch"
    dir_fig = out_dir / "03_figures"
    for directory in [dir_input, dir_stats, dir_fig]:
        directory.mkdir(parents=True, exist_ok=True)

    log("=" * 110)
    log("[START] Pure-Python sample-level pseudobulk log2-Welch analysis")
    log("[OUT]", out_dir)
    log("[RULE] No P-value floor is used for the volcano y coordinate.")
    log("[RULE] No y-axis display cap is used.")
    log("=" * 110)

    tumour_manifest, tumour_counts_path, tumour_meta_path, tumour_genes = (
        load_tumour_source()
    )
    normal_manifest, normal_counts_path, normal_meta_path, normal_genes = (
        load_normal_source()
    )

    tumour_gene_map = make_unique_gene_map(tumour_genes)
    normal_gene_map = make_unique_gene_map(normal_genes)
    common_genes = [gene for gene in tumour_gene_map if gene in normal_gene_map]
    if len(common_genes) < 1000:
        raise RuntimeError(f"Too few common genes: {len(common_genes)}")

    tumour_gene_indices = np.asarray(
        [tumour_gene_map[gene] for gene in common_genes],
        dtype=np.int64,
    )
    normal_gene_indices = np.asarray(
        [normal_gene_map[gene] for gene in common_genes],
        dtype=np.int64,
    )
    log("[COMMON GENES]", len(common_genes))

    tumour_pb, tumour_sample_meta = aggregate_pseudobulk(
        manifest=tumour_manifest,
        counts_path=tumour_counts_path,
        sample_column=TUMOUR_SAMPLE_COLUMN,
        row_column=TUMOUR_ROW_COLUMN,
        gene_indices=tumour_gene_indices,
        condition="tumor",
        sample_prefix="T__",
    )
    normal_pb, normal_sample_meta = aggregate_pseudobulk(
        manifest=normal_manifest,
        counts_path=normal_counts_path,
        sample_column=NORMAL_SAMPLE_COLUMN,
        row_column=NORMAL_ROW_COLUMN,
        gene_indices=normal_gene_indices,
        condition="normal",
        sample_prefix="N__",
    )

    sample_meta = pd.concat(
        [normal_sample_meta, tumour_sample_meta],
        ignore_index=True,
    )
    combined_samples_by_gene = np.vstack([normal_pb, tumour_pb])
    sample_ids = sample_meta["sample_id"].astype(str).tolist()
    if combined_samples_by_gene.shape != (len(sample_ids), len(common_genes)):
        raise ValueError("Combined pseudobulk matrix shape mismatch")

    counts_samples_by_genes = pd.DataFrame(
        combined_samples_by_gene,
        index=pd.Index(sample_ids, name="sample_id"),
        columns=common_genes,
    )
    counts_filtered, gene_filter_qc = prefilter_count_matrix(
        counts_samples_by_genes
    )
    log("[GENES RETAINED]", counts_filtered.shape[1], "/", len(common_genes))

    # Export raw pseudobulk matrices in both orientations.
    counts_samples_by_genes.reset_index().to_csv(
        dir_input / "pseudobulk_raw_counts_sample_by_gene.tsv",
        sep="\t",
        index=False,
    )
    counts_gene_by_sample = counts_samples_by_genes.T.copy()
    counts_gene_by_sample.index.name = "gene"
    counts_gene_by_sample.reset_index().to_csv(
        dir_input / "pseudobulk_raw_counts_gene_by_sample.tsv",
        sep="\t",
        index=False,
    )
    counts_filtered.reset_index().to_csv(
        dir_input / "pseudobulk_filtered_counts_sample_by_gene.tsv",
        sep="\t",
        index=False,
    )
    sample_meta.to_csv(
        dir_input / "pseudobulk_sample_metadata.tsv",
        sep="\t",
        index=False,
    )
    sample_meta.to_csv(
        dir_input / "pseudobulk_sample_QC.csv",
        index=False,
    )
    gene_filter_qc.to_csv(
        dir_input / "gene_prefilter_QC.csv",
        index=False,
    )
    pd.DataFrame({"gene": common_genes}).to_csv(
        dir_input / "common_gene_vocabulary.csv",
        index=False,
    )

    heatmap_run = find_latest_heatmap_run(HEATMAP_PARENT)
    mltd_genes = load_mltd_genes(heatmap_run)
    mltd_genes.to_csv(
        dir_input / "MLTD_selected_genes_used.csv",
        index=False,
    )
    log("[MLTD HEATMAP RUN]", heatmap_run)
    log("[MLTD GENE SOURCE]", MLTD_GENE_SOURCE)
    log("[MLTD GENES AFTER SUPPORT FILTER]", len(mltd_genes))

    (
        welch_result,
        normalized_counts,
        log2_normalized_counts,
        size_factors,
    ) = run_sample_level_welch(
        counts_samples_by_genes=counts_filtered,
        sample_metadata=sample_meta,
        output_dir=dir_stats,
    )

    volcano_table = prepare_volcano_table(welch_result, mltd_genes)
    volcano_table_path = (
        dir_stats / "Welch_pseudobulk_volcano_plot_table.csv"
    )
    volcano_table.to_csv(volcano_table_path, index=False)
    log("[SAVE]", volcano_table_path)

    plot_volcano(
        volcano_table,
        dir_fig / "pseudobulk_log2_Welch_tumour_vs_normal_MLTD_volcano",
    )
    build_and_plot_pca(
        log2_normalized_counts=log2_normalized_counts,
        sample_metadata=sample_meta,
        output_table=dir_stats / "log2_Welch_PCA_coordinates.csv",
        output_base=dir_fig / "pseudobulk_log2_normalized_sample_PCA",
    )

    n_normal = int((sample_meta["condition"] == "normal").sum())
    n_tumour = int((sample_meta["condition"] == "tumor").sum())
    regulation_counts = volcano_table["regulation"].value_counts().to_dict()
    finite_y = volcano_table["minus_log10_pvalue"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    summary = {
        "dataset_name": DATASET_NAME,
        "analysis": (
            "sample-level pseudobulk positive-count median-ratio normalization; "
            "Welch t test on log2-normalized counts"
        ),
        "normal_dataset": NORMAL_DATASET_ID,
        "normal_sample_column": NORMAL_SAMPLE_COLUMN,
        "tumour_sample_column": TUMOUR_SAMPLE_COLUMN,
        "n_normal_replicates": n_normal,
        "n_tumour_replicates": n_tumour,
        "tumour_samples": tumour_sample_meta["original_sample_id"].tolist(),
        "normal_samples": normal_sample_meta["original_sample_id"].tolist(),
        "n_common_genes": len(common_genes),
        "n_genes_after_prefilter": int(counts_filtered.shape[1]),
        "n_tested_genes": int(len(volcano_table)),
        "MLTD_gene_source": MLTD_GENE_SOURCE,
        "MLTD_highlight_top_n": MLTD_HIGHLIGHT_TOP_N,
        "MLTD_min_samples_present": MLTD_MIN_SAMPLES_PRESENT,
        "MLTD_min_direction_consistency": MLTD_MIN_DIRECTION_CONSISTENCY,
        "n_MLTD_supported_genes_in_overlay_file": int(len(mltd_genes)),
        "n_MLTD_supported_genes_tested": int(
            volcano_table["is_MLTD_selected"].sum()
        ),
        "regulation_counts": regulation_counts,
        "normalization": {
            "method": SIZE_FACTOR_METHOD,
            "log2_transform_pseudocount": LOG2_TRANSFORM_PSEUDOCOUNT,
            "log2fc_pseudocount": LOG2FC_PSEUDOCOUNT,
        },
        "volcano": {
            "x": "log2 fold change of median-ratio-normalized pseudobulk means",
            "y": "-log10(two-sided sample-level Welch pvalue)",
            "color_significance_column": SIGNIFICANCE_COLUMN,
            "pvalue_threshold": PVALUE_THRESHOLD,
            "padj_threshold": PADJ_THRESHOLD,
            "absolute_log2fc_threshold": ABS_LOG2FC_THRESHOLD,
            "y_display_cap": None,
            "maximum_finite_minus_log10_pvalue": (
                float(finite_y.max()) if len(finite_y) else None
            ),
        },
        "warning": (
            "The sample-level Welch test lets biological-sample variability "
            "control significance and does not impose a y cap. However, normal "
            "and tumour originate from separate cohorts, so condition remains "
            "confounded with cohort/batch. Inspect the pseudobulk PCA."
        ),
        "inputs": {
            "tumour_manifest": TUMOUR_MANIFEST,
            "tumour_counts": tumour_counts_path,
            "tumour_meta": tumour_meta_path,
            "normal_manifest": NORMAL_MANIFEST,
            "normal_counts": normal_counts_path,
            "normal_meta": normal_meta_path,
            "MLTD_heatmap_run": heatmap_run,
        },
        "outputs": {
            "root": out_dir,
            "sample_metadata": dir_input / "pseudobulk_sample_metadata.tsv",
            "Welch_results": dir_stats / "Welch_pseudobulk_tumour_vs_normal.csv",
            "normalized_counts": (
                dir_stats / "median_ratio_normalized_counts_sample_by_gene.csv"
            ),
            "log2_normalized_counts": (
                dir_stats / "log2_normalized_counts_sample_by_gene.csv"
            ),
            "size_factors": dir_stats / "median_ratio_size_factors.csv",
            "volcano_table": volcano_table_path,
            "volcano_png": (
                dir_fig
                / "pseudobulk_log2_Welch_tumour_vs_normal_MLTD_volcano.png"
            ),
            "volcano_svg": (
                dir_fig
                / "pseudobulk_log2_Welch_tumour_vs_normal_MLTD_volcano.svg"
            ),
            "volcano_pdf": (
                dir_fig
                / "pseudobulk_log2_Welch_tumour_vs_normal_MLTD_volcano.pdf"
            ),
            "pca_png": (
                dir_fig / "pseudobulk_log2_normalized_sample_PCA.png"
            ),
        },
    }

    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_safe,
        )
    log("[SAVE]", summary_path)

    log("\n" + "=" * 110)
    log("[DONE]")
    log("[OUTPUT ROOT]", out_dir.resolve())
    log("[BIOLOGICAL REPLICATES] normal=", n_normal, "tumour=", n_tumour)
    log("[VOLCANO Y] actual -log10(sample-level Welch pvalue); no cap")
    log("=" * 110)


if __name__ == "__main__":
    main()
