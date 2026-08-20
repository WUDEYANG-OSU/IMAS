from __future__ import annotations

# =============================================================================
# NDS preparation for the completed eight-sample kidney-cancer model
# (Kidney-cancer-Sample-7 excluded during Stage-B training)
# =============================================================================
# Method retained from the supplied single-sample script:
#   1) global DEG: normal kidney reference versus tumour RNA;
#   2) IMAS projected candidate cascades: ligand -> receptor -> TF;
#   3) candidate scoring combines CCI, receptor-TF, DEG alignment,
#      biological-axis hint and hidden-mechanism support.
#
# Multi-sample adaptation:
#   - directly reuse the completed eight-sample Stage-B manifest;
#   - use Sample-1..6, Sample-8 and Sample-9 only;
#   - compute one pooled eight-sample DEG table for the canonical NDS input;
#   - compute independent normal-vs-tumour DEG tables for every sample;
#   - choose the best available CCI table separately for every sample;
#   - build candidate cascades only from within-sample CCI edges;
#   - export per-sample candidates plus a cross-sample consensus table;
#   - never re-filter Sample-7 silently: its presence is treated as an error.
#
# Canonical NDS outputs:
#   01_global_DEG/global_DEG_normal_all_vs_tumor_all.csv
#   03_global_IMAS_projection/all_global_imas_projected_candidate_cascades.csv
# =============================================================================

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE = get_project_root()

# Normal-kidney reference retained from the supplied code.
NORMAL_SAMPLE_ID = "GSE202109_ALL"
NORMAL_ROOT = (
    BASE
    / "LLM-WorldWord"
    / "Independent-Verify-NormalKidney"
    / "RNA_all_gene_input_matrices_per_sample_pureR_v2_allgroups"
    / NORMAL_SAMPLE_ID
    / "processed_RNA_all_genes"
)
NORMAL_META_NPZ = NORMAL_ROOT / f"{NORMAL_SAMPLE_ID}.rna_meta.npz"
NORMAL_LOG1P = NORMAL_ROOT / f"{NORMAL_SAMPLE_ID}.X_rna_log1p_norm_dense.npy"

# Exact completed Stage-B run with Sample-7 excluded.
RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "stageB_RNA_only_Kidney_cancer_8samples_exclude_sample7_joint"
)
DATASET_NAME = "scVerifyKC-Kidney-cancer-8samples-exclude-sample7-joint"
RUN_STAGEB_DATASET_DIR = RUN_STAGEB_DIR / DATASET_NAME
STAGEB_MANIFEST = RUN_STAGEB_DIR / "input_manifest_excluding_sample7.tsv"
STAGEB_CHECKPOINT = (
    RUN_STAGEB_DATASET_DIR
    / "best_stageB_Kidney_cancer_8samples_exclude_sample7_joint.pt"
)

EXPECTED_SAMPLE_COUNT = 8
EXPECTED_EXCLUDED_SAMPLE_NUMBERS = {7}
POOLED_TUMOR_ID = "Kidney-cancer-8samples-exclude-sample7-pooled"

# Exact CCI output generated from the completed eight-sample model.
CCI_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_cci_jointIntegratedUMAP_with_sample_labels"
)
CCI_TFKEPT_CSV = CCI_DIR / "interactions_within_sample_highconf_TFkept.csv"
CCI_HIGHCONF_CSV = CCI_DIR / "interactions_within_sample_highconf.csv"
CCI_ALL_CSV = CCI_DIR / "interactions_within_sample_cluster_level.csv"
RECEPTOR_TF_SIG_CSV = CCI_DIR / "receptor_tf_signature_across_sample_clusters.csv"

# Output root. The canonical NDS file names are kept unchanged.
OUT_ROOT = (
    BASE
    / "LLM-WorldWord"
    / "HiddenMechanism_NormalRef_IMAS"
    / "GLOBAL_GSE202109_ALL_vs_Kidney_cancer_8samples_exclude_sample7_DEG_IMAS_volcano"
)
DIR_INPUT_QC = OUT_ROOT / "00_input_QC"
DIR_DEG = OUT_ROOT / "01_global_DEG"
DIR_FIG = OUT_ROOT / "02_figures"
DIR_IMAS = OUT_ROOT / "03_global_IMAS_projection"
DIR_PER_SAMPLE = OUT_ROOT / "04_per_sample"
DIR_CONSENSUS = OUT_ROOT / "05_cross_sample_consensus"

# DEG settings retained from the supplied script.
MIN_PCT_EITHER = 0.001
MIN_MEAN_EITHER = 0.001
PSEUDOCOUNT = 0.02
GENE_BATCH = 512
# GENE_BATCH remains the algorithmic cap. The effective batch can be reduced
# automatically to keep one dense block below this memory budget.
MAX_EXPRESSION_BLOCK_GB = 0.55
NORMALIZATION_TARGET_SUM = 1e4

# Plot settings retained from the supplied script.
FIG_DPI = 300
VOLCANO_Y_SOFT_CAP = 80.0
VOLCANO_Y_DISPLAY_MAX = 90.0
LABEL_TOP_N = 40

# IMAS candidate settings retained from the supplied script.
MAX_CCI_ROWS_USE = 500000
MAX_SIG_ROWS_USE = 500000
TOP_TF_PER_RECEPTOR = 20
MIN_CCI_SCORE_PERCENTILE_KEEP = 0.00
MIN_TF_SCORE_PERCENTILE_KEEP = 0.00
MAX_CANDIDATES_EXPORT = 1000000
TOP_CANDIDATES_PER_SAMPLE_EXPORT = 5000
TOP_GLOBAL_CANDIDATES_EXPORT = 5000

# Candidate scoring weights retained from the supplied script.
W_CCI = 0.35
W_TF = 0.30
W_DEG = 0.20
W_AXIS = 0.10
W_HIDDEN = 0.05

SEED = 42
np.random.seed(SEED)


# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================


def log(*args) -> None:
    print(*args, flush=True)


def to_wsl_path(value: str | Path) -> str:
    value = str(value).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", value)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return value


def safe_name(value: str, n: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]+", "_", str(value))[:n]


def normalize_cluster_str(value: str) -> str:
    value = str(value).strip()
    if value.startswith("C") and value[1:].isdigit():
        return value[1:]
    return value


def normalize_sample_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def sample_number_from_label(sample_id: str) -> int | None:
    key = normalize_sample_key(sample_id)
    match = re.search(r"sample0*([0-9]+)$", key)
    return int(match.group(1)) if match else None


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() <= 1:
        out[finite] = 0.5
        return out
    ranks = stats.rankdata(values[finite], method="average")
    out[finite] = (ranks - 1.0) / max(1, finite.sum() - 1)
    return out


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def soft_cap_y(values: np.ndarray, cap: float = 80.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)
    return cap * (1.0 - np.exp(-values / cap))


def find_first_existing_col(
    frame: pd.DataFrame,
    candidate_columns: Sequence[str],
) -> Optional[str]:
    for column in candidate_columns:
        if column in frame.columns:
            return column
    return None


def find_column_by_tokens(
    frame: pd.DataFrame,
    required_tokens: Sequence[str],
) -> Optional[str]:
    for column in frame.columns:
        lower = str(column).lower()
        if all(token.lower() in lower for token in required_tokens):
            return column
    return None


def is_human_style_symbol(gene: str) -> bool:
    gene = str(gene).strip()
    if gene in {"", "NA", "nan", "None", "<NA>"}:
        return False
    if len(gene) > 35:
        return False
    if re.search(r"[a-z]", gene):
        return False
    return re.match(r"^[A-Z0-9.\-]+$", gene) is not None


def ensure_output_directories() -> None:
    for directory in [
        OUT_ROOT,
        DIR_INPUT_QC,
        DIR_DEG,
        DIR_FIG,
        DIR_IMAS,
        DIR_PER_SAMPLE,
        DIR_CONSENSUS,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def effective_gene_batch(n_cells: int) -> int:
    if n_cells <= 0:
        return int(GENE_BATCH)
    byte_budget = float(MAX_EXPRESSION_BLOCK_GB) * (1024**3)
    # One float32 dense expression block dominates memory. Reserve ~25% for
    # temporary arrays used in mean/variance calculations.
    estimated = int(byte_budget / (max(1, n_cells) * 4.0 * 1.25))
    return max(1, min(int(GENE_BATCH), estimated))


# =============================================================================
# 2. LOAD AND VALIDATE THE COMPLETED EIGHT-SAMPLE INPUT
# =============================================================================


@dataclass
class TumorInput:
    manifest: pd.DataFrame
    sample_labels: List[str]
    gene_names: np.ndarray
    counts_path: Path
    depth_path: Path
    log1p_path: Optional[Path]
    meta_path: Path
    matrix_n_rows: int
    row_indices: np.ndarray
    sample_positions: Dict[str, np.ndarray]


def load_completed_stageb_manifest() -> TumorInput:
    if not STAGEB_MANIFEST.exists():
        raise FileNotFoundError(STAGEB_MANIFEST)
    if not STAGEB_CHECKPOINT.exists():
        raise FileNotFoundError(STAGEB_CHECKPOINT)

    manifest = pd.read_csv(STAGEB_MANIFEST, sep="\t", low_memory=False)
    required = [
        "cell_id",
        "sample_id",
        "row_index",
        "counts_npy",
        "depth_npy",
        "meta_npz",
    ]
    for column in required:
        if column not in manifest.columns:
            raise KeyError(f"{STAGEB_MANIFEST.name}: missing {column}")

    manifest = manifest.copy()
    manifest["cell_id"] = manifest["cell_id"].astype(str).str.strip()
    manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
    manifest["row_index"] = pd.to_numeric(
        manifest["row_index"], errors="raise"
    ).astype(np.int64)

    for column in ["counts_npy", "depth_npy", "log1p_npy", "meta_npz"]:
        if column in manifest.columns:
            manifest[column] = manifest[column].map(
                lambda x: to_wsl_path(x)
                if pd.notna(x) and str(x).strip()
                else ""
            )

    manifest = manifest.sort_values("row_index").reset_index(drop=True)
    if len(manifest) == 0:
        raise ValueError("The completed Stage-B manifest is empty.")
    if manifest["row_index"].duplicated().any():
        raise ValueError("The completed Stage-B manifest has duplicate row_index values.")
    if manifest[["sample_id", "cell_id"]].duplicated().any():
        raise ValueError("sample_id + cell_id must be unique.")

    sample_labels = sorted(
        manifest["sample_id"].unique().tolist(), key=natural_sort_key
    )
    if len(sample_labels) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} completed samples, found "
            f"{len(sample_labels)}: {sample_labels}"
        )

    excluded_found = {
        number
        for number in (sample_number_from_label(x) for x in sample_labels)
        if number in EXPECTED_EXCLUDED_SAMPLE_NUMBERS
    }
    if excluded_found:
        raise ValueError(
            "The completed Stage-B manifest unexpectedly contains excluded "
            f"sample number(s): {sorted(excluded_found)}"
        )

    counts_paths = sorted(
        {x for x in manifest["counts_npy"].astype(str).tolist() if x.strip()}
    )
    depth_paths = sorted(
        {x for x in manifest["depth_npy"].astype(str).tolist() if x.strip()}
    )
    meta_paths = sorted(
        {x for x in manifest["meta_npz"].astype(str).tolist() if x.strip()}
    )
    log1p_paths = []
    if "log1p_npy" in manifest.columns:
        log1p_paths = sorted(
            {x for x in manifest["log1p_npy"].astype(str).tolist() if x.strip()}
        )

    if len(counts_paths) != 1:
        raise ValueError(f"Expected one combined counts matrix: {counts_paths}")
    if len(depth_paths) != 1:
        raise ValueError(f"Expected one combined depth array: {depth_paths}")
    if len(meta_paths) != 1:
        raise ValueError(f"Expected one combined metadata NPZ: {meta_paths}")
    if len(log1p_paths) > 1:
        raise ValueError(f"Expected at most one combined log1p matrix: {log1p_paths}")

    counts_path = Path(counts_paths[0])
    depth_path = Path(depth_paths[0])
    meta_path = Path(meta_paths[0])
    log1p_path = Path(log1p_paths[0]) if log1p_paths else None

    for path in [counts_path, depth_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if log1p_path is not None and not log1p_path.exists():
        log("[WARN] manifest log1p matrix does not exist; counts fallback:", log1p_path)
        log1p_path = None

    counts_mm = np.load(counts_path, mmap_mode="r")
    depth_mm = np.load(depth_path, mmap_mode="r")
    meta_npz = np.load(meta_path, allow_pickle=True)
    gene_names = np.asarray(meta_npz["gene_names"]).astype(str)

    if counts_mm.ndim != 2:
        raise ValueError("The combined tumour counts matrix must be two-dimensional.")
    if counts_mm.shape[0] != depth_mm.shape[0]:
        raise ValueError("Combined counts rows and depth length differ.")
    if counts_mm.shape[1] != len(gene_names):
        raise ValueError("Combined counts gene dimension and metadata differ.")
    if manifest["row_index"].min() < 0:
        raise ValueError("Negative row_index in completed manifest.")
    if manifest["row_index"].max() >= counts_mm.shape[0]:
        raise ValueError("row_index exceeds combined matrix bounds.")

    if log1p_path is not None:
        log1p_mm = np.load(log1p_path, mmap_mode="r")
        if log1p_mm.shape != counts_mm.shape:
            raise ValueError(
                "Combined log1p matrix shape differs from counts matrix: "
                f"{log1p_mm.shape} vs {counts_mm.shape}"
            )

    selected_rows = manifest["row_index"].to_numpy(dtype=np.int64)
    if "cell_ids" in meta_npz:
        all_cell_ids = np.asarray(meta_npz["cell_ids"]).astype(str)
        if len(all_cell_ids) != counts_mm.shape[0]:
            raise ValueError("metadata cell_ids length differs from matrix rows.")
        if not np.array_equal(
            all_cell_ids[selected_rows], manifest["cell_id"].to_numpy(dtype=str)
        ):
            raise ValueError("Manifest cell IDs do not match metadata at row_index.")
    if "sample_id" in meta_npz:
        all_sample_ids = np.asarray(meta_npz["sample_id"]).astype(str)
        if len(all_sample_ids) != counts_mm.shape[0]:
            raise ValueError("metadata sample_id length differs from matrix rows.")
        if not np.array_equal(
            all_sample_ids[selected_rows], manifest["sample_id"].to_numpy(dtype=str)
        ):
            raise ValueError("Manifest sample IDs do not match metadata at row_index.")

    sample_positions: Dict[str, np.ndarray] = {}
    sample_array = manifest["sample_id"].to_numpy(dtype=str)
    for sample_id in sample_labels:
        sample_positions[sample_id] = np.where(sample_array == sample_id)[0].astype(
            np.int64
        )
        if len(sample_positions[sample_id]) == 0:
            raise ValueError(f"No modeled cells for {sample_id}")

    counts_by_sample = (
        manifest.groupby("sample_id", sort=True).size().rename("n_cells").reset_index()
    )
    counts_by_sample.to_csv(DIR_INPUT_QC / "tumor_sample_cell_counts.tsv", sep="\t", index=False)
    manifest.to_csv(DIR_INPUT_QC / "completed_stageB_manifest_snapshot.tsv", sep="\t", index=False)

    log("[MANIFEST]", STAGEB_MANIFEST)
    log("[COUNTS]", counts_path, tuple(counts_mm.shape))
    log("[DEPTH]", depth_path, tuple(depth_mm.shape))
    log("[LOG1P]", log1p_path if log1p_path is not None else "counts/depth fallback")
    log("[META]", meta_path)
    log("[SAMPLES]", sample_labels)
    log(manifest.groupby("sample_id").size().to_string())

    return TumorInput(
        manifest=manifest,
        sample_labels=sample_labels,
        gene_names=gene_names,
        counts_path=counts_path,
        depth_path=depth_path,
        log1p_path=log1p_path,
        meta_path=meta_path,
        matrix_n_rows=int(counts_mm.shape[0]),
        row_indices=selected_rows,
        sample_positions=sample_positions,
    )


# =============================================================================
# 3. EXPRESSION SUMMARIES
# =============================================================================


class NormalExpression:
    def __init__(self, meta_npz: Path, log1p_path: Path):
        if not meta_npz.exists():
            raise FileNotFoundError(meta_npz)
        if not log1p_path.exists():
            raise FileNotFoundError(log1p_path)
        meta = np.load(meta_npz, allow_pickle=True)
        self.genes = np.asarray(meta["gene_names"]).astype(str)
        self.X = np.load(log1p_path, mmap_mode="r")
        if self.X.ndim != 2 or self.X.shape[1] != len(self.genes):
            raise ValueError("Normal log1p matrix and metadata gene names differ.")
        self.n_cells = int(self.X.shape[0])
        log("[LOAD NORMAL] cells=", self.n_cells, "genes=", len(self.genes))

    def summarize_all_genes(self) -> pd.DataFrame:
        n_genes = len(self.genes)
        batch_size = effective_gene_batch(self.n_cells)
        log("[NORMAL SUMMARY] effective gene batch =", batch_size)
        means: List[np.ndarray] = []
        variances: List[np.ndarray] = []
        percentages: List[np.ndarray] = []

        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            block = np.asarray(self.X[:, start:end], dtype=np.float32)
            means.append(np.nanmean(block, axis=0).astype(np.float32))
            if self.n_cells > 1:
                variances.append(np.nanvar(block, axis=0, ddof=1).astype(np.float32))
            else:
                variances.append(np.zeros((end - start,), dtype=np.float32))
            percentages.append(np.nanmean(block > 0, axis=0).astype(np.float32))
            if start == 0 or end % (batch_size * 10) == 0 or end == n_genes:
                log(f"[NORMAL SUMMARY] {end}/{n_genes} genes")
            del block

        return pd.DataFrame(
            {
                "gene": self.genes.astype(str),
                "mean": np.concatenate(means),
                "variance": np.concatenate(variances),
                "pct": np.concatenate(percentages),
                "n_cells": self.n_cells,
            }
        )


class GroupedTumorExpression:
    def __init__(self, data: TumorInput):
        self.data = data
        self.counts = np.load(data.counts_path, mmap_mode="r")
        self.depth = np.load(data.depth_path, mmap_mode="r")
        self.log1p = (
            np.load(data.log1p_path, mmap_mode="r")
            if data.log1p_path is not None
            else None
        )

    def read_selected_log1p_block(self, start: int, end: int) -> np.ndarray:
        rows = self.data.row_indices
        columns = np.arange(start, end, dtype=np.int64)
        if self.log1p is not None:
            block = np.asarray(
                self.log1p[rows[:, None], columns[None, :]], dtype=np.float32
            )
            return block

        counts_block = np.asarray(
            self.counts[rows[:, None], columns[None, :]], dtype=np.float32
        )
        depth_selected = np.asarray(self.depth[rows], dtype=np.float32)
        normalized = (
            counts_block
            / np.maximum(depth_selected[:, None], 1.0)
            * float(NORMALIZATION_TARGET_SUM)
        )
        np.log1p(np.clip(normalized, 0.0, None), out=normalized)
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _summarize_block(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_cells = int(block.shape[0])
        mean = np.nanmean(block, axis=0).astype(np.float32)
        if n_cells > 1:
            variance = np.nanvar(block, axis=0, ddof=1).astype(np.float32)
        else:
            variance = np.zeros((block.shape[1],), dtype=np.float32)
        pct = np.nanmean(block > 0, axis=0).astype(np.float32)
        return mean, variance, pct

    def summarize_pooled_and_per_sample(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        n_selected = len(self.data.row_indices)
        n_genes = len(self.data.gene_names)
        batch_size = effective_gene_batch(n_selected)
        log("[TUMOR SUMMARY] selected cells =", n_selected)
        log("[TUMOR SUMMARY] effective gene batch =", batch_size)

        pooled_acc = {"mean": [], "variance": [], "pct": []}
        sample_acc = {
            sample_id: {"mean": [], "variance": [], "pct": []}
            for sample_id in self.data.sample_labels
        }

        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            block = self.read_selected_log1p_block(start, end)
            p_mean, p_var, p_pct = self._summarize_block(block)
            pooled_acc["mean"].append(p_mean)
            pooled_acc["variance"].append(p_var)
            pooled_acc["pct"].append(p_pct)

            for sample_id in self.data.sample_labels:
                positions = self.data.sample_positions[sample_id]
                sub = block[positions, :]
                s_mean, s_var, s_pct = self._summarize_block(sub)
                sample_acc[sample_id]["mean"].append(s_mean)
                sample_acc[sample_id]["variance"].append(s_var)
                sample_acc[sample_id]["pct"].append(s_pct)
                del sub

            if start == 0 or end % (batch_size * 10) == 0 or end == n_genes:
                log(f"[TUMOR SUMMARY] {end}/{n_genes} genes")
            del block

        pooled = pd.DataFrame(
            {
                "gene": self.data.gene_names.astype(str),
                "mean": np.concatenate(pooled_acc["mean"]),
                "variance": np.concatenate(pooled_acc["variance"]),
                "pct": np.concatenate(pooled_acc["pct"]),
                "n_cells": n_selected,
            }
        )

        per_sample: Dict[str, pd.DataFrame] = {}
        for sample_id in self.data.sample_labels:
            acc = sample_acc[sample_id]
            per_sample[sample_id] = pd.DataFrame(
                {
                    "gene": self.data.gene_names.astype(str),
                    "mean": np.concatenate(acc["mean"]),
                    "variance": np.concatenate(acc["variance"]),
                    "pct": np.concatenate(acc["pct"]),
                    "n_cells": len(self.data.sample_positions[sample_id]),
                }
            )
        return pooled, per_sample


# =============================================================================
# 4. DEG AND VOLCANO
# =============================================================================


def compute_global_deg_from_summaries(
    normal_summary: pd.DataFrame,
    tumor_summary: pd.DataFrame,
    tumor_label: str,
) -> pd.DataFrame:
    normal = normal_summary.rename(
        columns={
            "mean": "mean_normal",
            "variance": "variance_normal",
            "pct": "pct_normal",
            "n_cells": "n_normal",
        }
    )
    tumor = tumor_summary.rename(
        columns={
            "mean": "mean_tumor",
            "variance": "variance_tumor",
            "pct": "pct_tumor",
            "n_cells": "n_tumor",
        }
    )
    merged = normal.merge(tumor, on="gene", how="inner")
    if len(merged) == 0:
        raise RuntimeError(f"No common genes for normal vs {tumor_label}")

    mean_n = merged["mean_normal"].to_numpy(dtype=float)
    mean_t = merged["mean_tumor"].to_numpy(dtype=float)
    var_n = merged["variance_normal"].to_numpy(dtype=float)
    var_t = merged["variance_tumor"].to_numpy(dtype=float)
    pct_n = merged["pct_normal"].to_numpy(dtype=float)
    pct_t = merged["pct_tumor"].to_numpy(dtype=float)
    n_n = int(merged["n_normal"].iloc[0])
    n_t = int(merged["n_tumor"].iloc[0])

    denom = np.sqrt(var_n / max(n_n, 1) + var_t / max(n_t, 1))
    t_stat = (mean_t - mean_n) / np.maximum(denom, 1e-12)
    df_num = (var_n / max(n_n, 1) + var_t / max(n_t, 1)) ** 2
    df_den = (
        (var_n**2) / ((max(n_n, 1) ** 2) * max(n_n - 1, 1))
        + (var_t**2) / ((max(n_t, 1) ** 2) * max(n_t - 1, 1))
    )
    dof = df_num / np.maximum(df_den, 1e-12)
    pval = 2.0 * stats.t.sf(np.abs(t_stat), df=np.maximum(dof, 1.0))
    pval = np.clip(pval, 1e-300, 1.0)

    log2fc = np.log2((mean_t + PSEUDOCOUNT) / (mean_n + PSEUDOCOUNT))
    pct_delta = pct_t - pct_n
    mean_either = np.maximum(mean_n, mean_t)
    pct_either = np.maximum(pct_n, pct_t)
    support = (mean_either >= MIN_MEAN_EITHER) & (pct_either >= MIN_PCT_EITHER)

    deg = pd.DataFrame(
        {
            "gene": merged["gene"].astype(str),
            "mean_normal": mean_n,
            "mean_tumor": mean_t,
            "pct_normal": pct_n,
            "pct_tumor": pct_t,
            "pct_delta_tumor_minus_normal": pct_delta,
            "log2FC_tumor_vs_normal": log2fc,
            "t_stat": t_stat,
            "pval": pval,
            "mean_either_max": mean_either,
            "pct_either_max": pct_either,
            "gene_supported": support.astype(bool),
            "normal_n_cells": n_n,
            "tumor_n_cells": n_t,
            "tumor_sample_id": tumor_label,
        }
    )
    deg["fdr"] = bh_fdr(deg["pval"].to_numpy(dtype=float))
    deg["abs_log2FC"] = np.abs(deg["log2FC_tumor_vs_normal"].to_numpy(dtype=float))
    deg["neglog10_fdr"] = -np.log10(
        np.clip(deg["fdr"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    deg["DEG_rank_score"] = (
        rank01(deg["abs_log2FC"].to_numpy(dtype=float)) * 0.45
        + rank01(deg["neglog10_fdr"].to_numpy(dtype=float)) * 0.45
        + rank01(deg["pct_either_max"].to_numpy(dtype=float)) * 0.10
    )
    deg = deg.sort_values(
        ["DEG_rank_score", "abs_log2FC", "neglog10_fdr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    deg.insert(0, "DEG_rank", np.arange(1, len(deg) + 1))
    return deg


def plot_global_deg_volcano(
    deg: pd.DataFrame,
    tumor_label: str,
    output_base: Path,
) -> Dict[str, object]:
    frame = deg.copy()
    frame["plot_x"] = frame["log2FC_tumor_vs_normal"].replace(
        [np.inf, -np.inf], np.nan
    )
    y_raw = -np.log10(
        np.clip(frame["fdr"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    frame["plot_y"] = soft_cap_y(y_raw, cap=VOLCANO_Y_SOFT_CAP)
    significant = (
        (frame["fdr"] < 0.05)
        & (np.abs(frame["log2FC_tumor_vs_normal"]) > 0.25)
        & frame["gene_supported"].astype(bool)
    )
    frame["is_sig"] = significant

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    background = frame[~frame["is_sig"]]
    highlighted = frame[frame["is_sig"]]
    ax.scatter(
        background["plot_x"],
        background["plot_y"],
        s=6,
        c="#C7C7C7",
        alpha=0.45,
        linewidths=0,
        zorder=1,
        label="All genes",
    )
    ax.scatter(
        highlighted["plot_x"],
        highlighted["plot_y"],
        s=15,
        c="#D62728",
        alpha=0.75,
        linewidths=0,
        zorder=2,
        label="DEG",
    )
    labels = highlighted.sort_values("DEG_rank_score", ascending=False).head(
        LABEL_TOP_N
    )
    for _, row in labels.iterrows():
        ax.text(
            float(row["plot_x"]),
            float(row["plot_y"]),
            str(row["gene"]),
            fontsize=7.5,
            color="#7F0000",
            ha="center",
            va="bottom",
            zorder=4,
        )

    threshold_y = soft_cap_y(
        np.array([-np.log10(0.05)]), cap=VOLCANO_Y_SOFT_CAP
    )[0]
    ax.axhline(threshold_y, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(0.25, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(-0.25, color="#777777", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#444444", linestyle="-", linewidth=0.5)

    finite_x = frame["plot_x"].dropna().to_numpy(dtype=float)
    xmax = np.nanpercentile(np.abs(finite_x), 99.5) if len(finite_x) else 2.0
    xmax = min(max(float(xmax), 2.0), 8.0)
    ax.set_xlim(-xmax * 1.08, xmax * 1.08)
    ax.set_ylim(-1, VOLCANO_Y_DISPLAY_MAX)
    ax.set_xlabel("log2FC tumor vs normal", fontsize=12)
    ax.set_ylabel("Soft-capped -log10(FDR)", fontsize=12)
    ax.set_title(
        f"Global DEG volcano\n{NORMAL_SAMPLE_ID} vs {tumor_label}",
        fontsize=14,
        pad=14,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    fig.tight_layout()

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(png),
        "svg": str(svg),
        "pdf": str(pdf),
        "n_deg_sig": int(significant.sum()),
    }


# =============================================================================
# 5. CCI AND RECEPTOR-TF INPUTS
# =============================================================================


def guess_cci_columns(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    output: Dict[str, Optional[str]] = {
        "sample_id": None,
        "ligand": None,
        "receptor": None,
        "sender_cluster": None,
        "receiver_cluster": None,
        "sender_profile_key": None,
        "receiver_profile_key": None,
        "score": None,
        "confidence": None,
    }
    for column in frame.columns:
        lower = str(column).lower()
        if output["sample_id"] is None and lower in {"sample_id", "sid", "sample"}:
            output["sample_id"] = column
        if output["ligand"] is None and "ligand" in lower:
            output["ligand"] = column
        if output["receptor"] is None and "receptor" in lower:
            output["receptor"] = column
        if output["sender_cluster"] is None and (
            "sender_cluster" in lower or "source_cluster" in lower
        ):
            output["sender_cluster"] = column
        if output["receiver_cluster"] is None and (
            "receiver_cluster" in lower or "target_cluster" in lower
        ):
            output["receiver_cluster"] = column
        if output["sender_profile_key"] is None and "sender_profile_key" in lower:
            output["sender_profile_key"] = column
        if output["receiver_profile_key"] is None and "receiver_profile_key" in lower:
            output["receiver_profile_key"] = column

    output["score"] = find_first_existing_col(
        frame,
        ["edge_score", "score", "expr_only", "confidence", "n_pairs", "weight"],
    )
    output["confidence"] = find_first_existing_col(
        frame, ["confidence", "conf", "prob"]
    )
    return output


def guess_signature_columns(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    output: Dict[str, Optional[str]] = {
        "receptor": None,
        "tf": None,
        "score": None,
    }
    for column in frame.columns:
        lower = str(column).lower()
        if output["receptor"] is None and "receptor" in lower:
            output["receptor"] = column
        if output["tf"] is None and (
            lower == "tf"
            or lower == "tf_name"
            or "tf_name" in lower
            or lower.endswith("_tf")
        ):
            output["tf"] = column

    output["score"] = find_first_existing_col(
        frame,
        [
            "tf_score",
            "corr_across_sample_clusters",
            "corr",
            "score",
            "weight",
            "tf_corr",
            "signature_score",
        ],
    )
    if output["score"] is None:
        output["score"] = find_column_by_tokens(frame, ["corr"])
    return output


def load_cci_tables() -> Dict[str, pd.DataFrame]:
    paths = {
        "TFkept": CCI_TFKEPT_CSV,
        "highconf": CCI_HIGHCONF_CSV,
        "all": CCI_ALL_CSV,
    }
    tables: Dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        if path.exists():
            table = pd.read_csv(path, low_memory=False)
            tables[label] = table
            log(f"[CCI {label}]", path, table.shape)
    if not tables:
        raise FileNotFoundError(
            "No eight-sample CCI interaction table found under:\n" f"{CCI_DIR}"
        )
    return tables


def choose_cci_for_sample(
    tables: Dict[str, pd.DataFrame],
    sample_id: str,
) -> Tuple[str, pd.DataFrame]:
    for source in ["TFkept", "highconf", "all"]:
        table = tables.get(source)
        if table is None or len(table) == 0:
            continue
        columns = guess_cci_columns(table)
        sample_col = columns["sample_id"]
        if sample_col is None:
            raise KeyError(f"CCI {source} table is missing sample_id.")
        subset = table[table[sample_col].astype(str) == str(sample_id)].copy()
        if len(subset) > 0:
            return source, subset.reset_index(drop=True)
    raise RuntimeError(f"No usable CCI rows found for {sample_id}")


def prepare_cci_table(
    cci: pd.DataFrame,
    sample_id: str,
    source_table: str,
) -> pd.DataFrame:
    columns = guess_cci_columns(cci)
    for name in ["sample_id", "ligand", "receptor"]:
        if columns[name] is None:
            raise KeyError(f"CCI table missing {name} column")

    frame = cci.copy()
    frame["sample_id"] = frame[columns["sample_id"]].astype(str)
    frame = frame[frame["sample_id"] == str(sample_id)].copy()
    frame["ligand"] = frame[columns["ligand"]].astype(str)
    frame["receptor"] = frame[columns["receptor"]].astype(str)

    if columns["sender_cluster"] is not None:
        frame["sender_cluster"] = frame[columns["sender_cluster"]].astype(str).map(
            normalize_cluster_str
        )
    else:
        frame["sender_cluster"] = ""
    if columns["receiver_cluster"] is not None:
        frame["receiver_cluster"] = frame[columns["receiver_cluster"]].astype(str).map(
            normalize_cluster_str
        )
    else:
        frame["receiver_cluster"] = ""

    if columns["sender_profile_key"] is not None:
        frame["sender_profile_key"] = frame[columns["sender_profile_key"]].astype(str)
    else:
        frame["sender_profile_key"] = [
            f"{sample_id}::{cluster}" for cluster in frame["sender_cluster"]
        ]
    if columns["receiver_profile_key"] is not None:
        frame["receiver_profile_key"] = frame[
            columns["receiver_profile_key"]
        ].astype(str)
    else:
        frame["receiver_profile_key"] = [
            f"{sample_id}::{cluster}" for cluster in frame["receiver_cluster"]
        ]

    for key_column in ["sender_profile_key", "receiver_profile_key"]:
        bad = ~frame[key_column].astype(str).str.startswith(f"{sample_id}::")
        if bad.any():
            examples = frame.loc[bad, key_column].astype(str).head(5).tolist()
            raise ValueError(
                f"Cross-sample or malformed {key_column} in {sample_id}: {examples}"
            )

    if columns["score"] is not None:
        frame["cci_score"] = pd.to_numeric(
            frame[columns["score"]], errors="coerce"
        ).fillna(0.0)
    else:
        frame["cci_score"] = 1.0
    if columns["confidence"] is not None:
        frame["confidence"] = pd.to_numeric(
            frame[columns["confidence"]], errors="coerce"
        ).fillna(1.0)
    else:
        frame["confidence"] = 1.0

    # Preserve the supplied method: CCI evidence is multiplied by confidence.
    frame["cci_score"] = (
        frame["cci_score"].astype(float) * frame["confidence"].astype(float)
    )
    frame = frame[
        frame["ligand"].map(is_human_style_symbol)
        & frame["receptor"].map(is_human_style_symbol)
    ].copy()
    frame = frame.sort_values("cci_score", ascending=False).head(MAX_CCI_ROWS_USE)
    frame = frame.reset_index(drop=True)
    if len(frame) == 0:
        raise RuntimeError(f"No valid CCI rows remain for {sample_id}")
    frame["cci_score_percentile"] = rank01(frame["cci_score"].to_numpy(dtype=float))
    frame = frame[
        frame["cci_score_percentile"] >= MIN_CCI_SCORE_PERCENTILE_KEEP
    ].copy()
    frame["cci_source_table"] = source_table
    return frame[
        [
            "sample_id",
            "ligand",
            "receptor",
            "sender_cluster",
            "receiver_cluster",
            "sender_profile_key",
            "receiver_profile_key",
            "cci_score",
            "confidence",
            "cci_score_percentile",
            "cci_source_table",
        ]
    ].reset_index(drop=True)


def prepare_receptor_tf_signature(signature: pd.DataFrame) -> pd.DataFrame:
    columns = guess_signature_columns(signature)
    if columns["receptor"] is None:
        raise KeyError("receptor-TF signature table is missing receptor.")
    if columns["tf"] is None:
        raise KeyError("receptor-TF signature table is missing TF.")

    frame = signature.copy()
    frame["receptor"] = frame[columns["receptor"]].astype(str)
    frame["tf"] = frame[columns["tf"]].astype(str)
    if columns["score"] is not None:
        frame["tf_score"] = pd.to_numeric(
            frame[columns["score"]], errors="coerce"
        ).fillna(0.0)
    else:
        log("[WARN] receptor-TF signature score was not detected; using 1.0")
        frame["tf_score"] = 1.0

    frame = frame[
        frame["receptor"].map(is_human_style_symbol)
        & frame["tf"].map(is_human_style_symbol)
    ].copy()
    frame = frame.sort_values("tf_score", ascending=False).head(MAX_SIG_ROWS_USE)
    frame["tf_score_percentile"] = rank01(frame["tf_score"].to_numpy(dtype=float))
    frame = frame[
        frame["tf_score_percentile"] >= MIN_TF_SCORE_PERCENTILE_KEEP
    ].copy()
    frame = (
        frame.sort_values("tf_score", ascending=False)
        .groupby("receptor", as_index=False)
        .head(TOP_TF_PER_RECEPTOR)
        .reset_index(drop=True)
    )
    return frame[["receptor", "tf", "tf_score", "tf_score_percentile"]]


# =============================================================================
# 6. CANDIDATE CASCADES
# =============================================================================


def make_axis_hint(ligand: str, receptor: str, tf: str) -> str:
    combined = " ".join([ligand, receptor, tf]).upper()
    if any(
        token in combined
        for token in [
            "FGF",
            "FGFR",
            "PDGF",
            "PDGFR",
            "EGF",
            "EGFR",
            "ERBB",
            "MET",
            "AXL",
            "KIT",
        ]
    ):
        return "RTK_related"
    if any(
        token in combined
        for token in ["COL", "FN1", "LAMB", "LAMC", "ITGA", "ITGB", "SPP1", "THBS"]
    ):
        return "ECM_integrin"
    if any(
        token in combined
        for token in ["CXCL", "CCL", "CCR", "CXCR", "IL", "TNF", "IFN"]
    ):
        return "inflammatory_cytokine"
    if any(
        token in combined
        for token in ["VEGF", "FLT", "KDR", "ANGPT", "TEK", "PECAM", "VWF", "ENG"]
    ):
        return "vascular_angiogenic"
    if any(token in combined for token in ["TGFB", "BMP", "SMAD"]):
        return "TGF_beta_BMP"
    if any(token in combined for token in ["WNT", "FZD", "LRP5", "LRP6"]):
        return "WNT_related"
    return "other"


def build_deg_lookup_arrays(
    cascades: pd.DataFrame,
    deg: pd.DataFrame,
    gene_column: str,
    output_prefix: str,
) -> pd.DataFrame:
    lookup = deg.set_index("gene")
    for source_column, suffix, default in [
        ("log2FC_tumor_vs_normal", "log2FC_tumor_vs_normal", 0.0),
        ("fdr", "fdr", 1.0),
        ("DEG_rank_score", "DEG_rank_score", 0.0),
    ]:
        mapping = lookup[source_column].to_dict()
        cascades[f"{output_prefix}_{suffix}"] = (
            cascades[gene_column].map(mapping).fillna(default).astype(float)
        )
    return cascades


def build_candidate_cascades(
    deg: pd.DataFrame,
    cci_prepared: pd.DataFrame,
    signature_prepared: pd.DataFrame,
    sample_id: str,
) -> pd.DataFrame:
    merged = cci_prepared.merge(signature_prepared, on="receptor", how="inner")
    log(f"[CASCADE MERGED] {sample_id}", merged.shape)
    if len(merged) == 0:
        raise RuntimeError(
            f"No ligand-receptor-TF cascades after CCI/signature merge for {sample_id}"
        )

    merged = build_deg_lookup_arrays(merged, deg, "ligand", "ligand")
    merged = build_deg_lookup_arrays(merged, deg, "receptor", "receptor")
    merged = build_deg_lookup_arrays(merged, deg, "tf", "tf")

    fc_matrix = np.column_stack(
        [
            merged["ligand_log2FC_tumor_vs_normal"].to_numpy(dtype=float),
            merged["receptor_log2FC_tumor_vs_normal"].to_numpy(dtype=float),
            merged["tf_log2FC_tumor_vs_normal"].to_numpy(dtype=float),
        ]
    )
    mean_fc = np.nanmean(fc_matrix, axis=1)
    direction_consistency = np.mean(
        np.sign(fc_matrix) == np.sign(mean_fc)[:, None], axis=1
    )
    fc_magnitude = np.nanmean(np.abs(fc_matrix), axis=1)
    deg_support = np.nanmean(
        np.column_stack(
            [
                merged["ligand_DEG_rank_score"].to_numpy(dtype=float),
                merged["receptor_DEG_rank_score"].to_numpy(dtype=float),
                merged["tf_DEG_rank_score"].to_numpy(dtype=float),
            ]
        ),
        axis=1,
    )
    phenotype_alignment = (
        0.55 * direction_consistency
        + 0.25 * np.minimum(1.0, fc_magnitude / 2.0)
        + 0.20 * deg_support
    )

    fdr_matrix = np.column_stack(
        [
            merged["ligand_fdr"].to_numpy(dtype=float),
            merged["receptor_fdr"].to_numpy(dtype=float),
            merged["tf_fdr"].to_numpy(dtype=float),
        ]
    )
    neglog = -np.log10(np.clip(fdr_matrix, 1e-300, 1.0))
    global_signature_deviation = 0.65 * deg_support + 0.35 * np.minimum(
        1.0, np.nanmean(neglog, axis=1) / 50.0
    )

    axis_hint = [
        make_axis_hint(ligand, receptor, tf)
        for ligand, receptor, tf in zip(
            merged["ligand"].astype(str),
            merged["receptor"].astype(str),
            merged["tf"].astype(str),
        )
    ]
    axis_score_map = {
        "RTK_related": 1.00,
        "vascular_angiogenic": 0.95,
        "ECM_integrin": 0.90,
        "TGF_beta_BMP": 0.85,
        "inflammatory_cytokine": 0.80,
        "WNT_related": 0.75,
        "other": 0.50,
    }
    axis_hint_score = np.asarray(
        [axis_score_map.get(x, 0.50) for x in axis_hint], dtype=float
    )

    cci_rank = merged["cci_score_percentile"].to_numpy(dtype=float)
    tf_rank = merged["tf_score_percentile"].to_numpy(dtype=float)
    hidden_mechanism_score = (
        0.35 * cci_rank + 0.35 * tf_rank + 0.30 * global_signature_deviation
    )
    imas_score = (
        W_CCI * cci_rank
        + W_TF * tf_rank
        + W_DEG * phenotype_alignment
        + W_AXIS * axis_hint_score
        + W_HIDDEN * hidden_mechanism_score
    )

    merged["cascade"] = (
        merged["ligand"].astype(str)
        + "->"
        + merged["receptor"].astype(str)
        + "->"
        + merged["tf"].astype(str)
    )
    merged["direction_consistency"] = direction_consistency
    merged["fc_magnitude_mean"] = fc_magnitude
    merged["deg_support_mean"] = deg_support
    merged["phenotype_alignment_score"] = phenotype_alignment
    merged["global_signature_deviation_score"] = global_signature_deviation
    merged["axis_hint"] = axis_hint
    merged["axis_hint_score"] = axis_hint_score
    merged["hidden_mechanism_score"] = hidden_mechanism_score
    merged["imas_cascade_score"] = imas_score
    merged["tumor_sample_id"] = sample_id

    keep_columns = [
        "tumor_sample_id",
        "cascade",
        "ligand",
        "receptor",
        "tf",
        "sender_cluster",
        "receiver_cluster",
        "sender_profile_key",
        "receiver_profile_key",
        "cci_source_table",
        "cci_score",
        "confidence",
        "tf_score",
        "cci_score_percentile",
        "tf_score_percentile",
        "phenotype_alignment_score",
        "global_signature_deviation_score",
        "axis_hint",
        "axis_hint_score",
        "hidden_mechanism_score",
        "imas_cascade_score",
        "direction_consistency",
        "fc_magnitude_mean",
        "deg_support_mean",
        "ligand_log2FC_tumor_vs_normal",
        "receptor_log2FC_tumor_vs_normal",
        "tf_log2FC_tumor_vs_normal",
        "ligand_fdr",
        "receptor_fdr",
        "tf_fdr",
        "ligand_DEG_rank_score",
        "receptor_DEG_rank_score",
        "tf_DEG_rank_score",
    ]
    keep_columns = [column for column in keep_columns if column in merged.columns]
    candidates = merged[keep_columns].copy()
    candidates = candidates.sort_values(
        ["imas_cascade_score", "cci_score", "tf_score"],
        ascending=[False, False, False],
    ).head(MAX_CANDIDATES_EXPORT)
    candidates = candidates.reset_index(drop=True)
    candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))
    return candidates


def plot_candidate_score_summary(
    candidates: pd.DataFrame,
    label: str,
    output_base: Path,
) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    x = np.arange(len(candidates))
    y = candidates["imas_cascade_score"].to_numpy(dtype=float)
    if len(x) > 5000:
        indices = np.linspace(0, len(x) - 1, 5000).astype(int)
        x = x[indices]
        y = y[indices]
    ax.plot(x + 1, y, linewidth=1.2)
    ax.set_xlabel("Candidate cascade rank", fontsize=12)
    ax.set_ylabel("IMAS cascade score", fontsize=12)
    ax.set_title(
        f"IMAS projected candidate cascades\n{NORMAL_SAMPLE_ID} vs {label}",
        fontsize=14,
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    fig.tight_layout()

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf)}


# =============================================================================
# 7. CROSS-SAMPLE CONSENSUS
# =============================================================================


def build_cross_sample_consensus(
    all_candidates: pd.DataFrame,
    n_expected_samples: int,
) -> pd.DataFrame:
    if len(all_candidates) == 0:
        return pd.DataFrame()

    numeric_columns = [
        "imas_cascade_score",
        "cci_score_percentile",
        "tf_score_percentile",
        "phenotype_alignment_score",
        "global_signature_deviation_score",
        "hidden_mechanism_score",
        "ligand_log2FC_tumor_vs_normal",
        "receptor_log2FC_tumor_vs_normal",
        "tf_log2FC_tumor_vs_normal",
    ]
    aggregation = {
        "n_samples_supported": ("tumor_sample_id", "nunique"),
        "sample_ids": (
            "tumor_sample_id",
            lambda x: ";".join(sorted(set(map(str, x)), key=natural_sort_key)),
        ),
        "best_imas_cascade_score": ("imas_cascade_score", "max"),
        "mean_imas_cascade_score": ("imas_cascade_score", "mean"),
        "median_imas_cascade_score": ("imas_cascade_score", "median"),
        "mean_cci_percentile": ("cci_score_percentile", "mean"),
        "mean_tf_percentile": ("tf_score_percentile", "mean"),
        "mean_phenotype_alignment": ("phenotype_alignment_score", "mean"),
        "mean_global_signature_deviation": (
            "global_signature_deviation_score",
            "mean",
        ),
        "mean_hidden_mechanism_score": ("hidden_mechanism_score", "mean"),
        "mean_ligand_log2FC": ("ligand_log2FC_tumor_vs_normal", "mean"),
        "mean_receptor_log2FC": ("receptor_log2FC_tumor_vs_normal", "mean"),
        "mean_tf_log2FC": ("tf_log2FC_tumor_vs_normal", "mean"),
        "axis_hint": ("axis_hint", lambda x: x.value_counts().index[0]),
    }
    consensus = all_candidates.groupby(
        ["cascade", "ligand", "receptor", "tf"], as_index=False
    ).agg(**aggregation)
    consensus["sample_coverage_fraction"] = (
        consensus["n_samples_supported"] / float(max(1, n_expected_samples))
    )
    consensus["consensus_score"] = (
        0.55 * consensus["mean_imas_cascade_score"]
        + 0.25 * consensus["best_imas_cascade_score"]
        + 0.20 * consensus["sample_coverage_fraction"]
    )
    consensus = consensus.sort_values(
        [
            "consensus_score",
            "n_samples_supported",
            "mean_imas_cascade_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
    return consensus


# =============================================================================
# 8. MAIN
# =============================================================================


def main() -> None:
    ensure_output_directories()

    log("=" * 110)
    log("[START] NDS preparation for completed eight-sample model")
    log("[NORMAL]", NORMAL_SAMPLE_ID)
    log("[STAGEB]", RUN_STAGEB_DATASET_DIR)
    log("[CCI]", CCI_DIR)
    log("[OUT]", OUT_ROOT)
    log("=" * 110)

    for path in [
        NORMAL_META_NPZ,
        NORMAL_LOG1P,
        STAGEB_MANIFEST,
        STAGEB_CHECKPOINT,
        RECEPTOR_TF_SIG_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    tumor_input = load_completed_stageb_manifest()
    normal_expression = NormalExpression(NORMAL_META_NPZ, NORMAL_LOG1P)
    tumor_expression = GroupedTumorExpression(tumor_input)

    # -------------------------------------------------------------------------
    # Expression summaries are computed once. The combined tumour matrix is
    # read one gene block at a time, and all sample statistics are derived from
    # that same block to avoid eight redundant full-matrix passes.
    # -------------------------------------------------------------------------
    log("\n[STEP 1] Summarize normal expression")
    normal_summary = normal_expression.summarize_all_genes()
    normal_summary.to_csv(
        DIR_INPUT_QC / "normal_expression_summary.tsv", sep="\t", index=False
    )

    log("\n[STEP 2] Summarize pooled and per-sample tumour expression")
    pooled_tumor_summary, per_sample_tumor_summary = (
        tumor_expression.summarize_pooled_and_per_sample()
    )
    pooled_tumor_summary.to_csv(
        DIR_INPUT_QC / "pooled_tumor_expression_summary.tsv",
        sep="\t",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Canonical pooled DEG output required by the next NDS step.
    # -------------------------------------------------------------------------
    log("\n[STEP 3] Compute pooled eight-sample global DEG")
    pooled_deg = compute_global_deg_from_summaries(
        normal_summary,
        pooled_tumor_summary,
        POOLED_TUMOR_ID,
    )
    pooled_deg_csv = DIR_DEG / "global_DEG_normal_all_vs_tumor_all.csv"
    pooled_deg.to_csv(pooled_deg_csv, index=False)
    pooled_deg_plot = plot_global_deg_volcano(
        pooled_deg,
        POOLED_TUMOR_ID,
        DIR_FIG / "global_DEG_volcano_normal_vs_pooled_8samples",
    )
    log("[WRITE]", pooled_deg_csv)

    # -------------------------------------------------------------------------
    # Load CCI once, then select the best available table independently for
    # each sample. Signature scores are derived from all sample-cluster units,
    # exactly as generated by the eight-sample CCI workflow.
    # -------------------------------------------------------------------------
    log("\n[STEP 4] Load CCI and receptor-TF signature")
    cci_tables = load_cci_tables()
    signature_raw = pd.read_csv(RECEPTOR_TF_SIG_CSV, low_memory=False)
    signature_prepared = prepare_receptor_tf_signature(signature_raw)
    signature_prepared.to_csv(
        DIR_INPUT_QC / "receptor_tf_signature_prepared.tsv",
        sep="\t",
        index=False,
    )
    log("[SIGNATURE PREPARED]", signature_prepared.shape)

    all_candidate_parts: List[pd.DataFrame] = []
    sample_summary_rows: List[dict] = []
    failed_rows: List[dict] = []

    # -------------------------------------------------------------------------
    # Per-sample DEG and candidate construction.
    # -------------------------------------------------------------------------
    log("\n[STEP 5] Per-sample DEG + IMAS candidate cascades")
    for sample_id in tumor_input.sample_labels:
        sample_dir = DIR_PER_SAMPLE / safe_name(sample_id)
        sample_deg_dir = sample_dir / "01_global_DEG"
        sample_fig_dir = sample_dir / "02_figures"
        sample_imas_dir = sample_dir / "03_global_IMAS_projection"
        for directory in [sample_deg_dir, sample_fig_dir, sample_imas_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        log("\n" + "-" * 100)
        log("[SAMPLE]", sample_id)
        try:
            sample_deg = compute_global_deg_from_summaries(
                normal_summary,
                per_sample_tumor_summary[sample_id],
                sample_id,
            )
            sample_deg_csv = (
                sample_deg_dir / "global_DEG_normal_all_vs_tumor_all.csv"
            )
            sample_deg.to_csv(sample_deg_csv, index=False)
            sample_deg_plot = plot_global_deg_volcano(
                sample_deg,
                sample_id,
                sample_fig_dir / "global_DEG_volcano_normal_vs_tumor",
            )

            cci_source, cci_raw = choose_cci_for_sample(cci_tables, sample_id)
            cci_prepared = prepare_cci_table(cci_raw, sample_id, cci_source)
            cci_prepared.to_csv(
                sample_imas_dir / "prepared_sample_cci.tsv",
                sep="\t",
                index=False,
            )
            candidates = build_candidate_cascades(
                deg=sample_deg,
                cci_prepared=cci_prepared,
                signature_prepared=signature_prepared,
                sample_id=sample_id,
            )
            candidates_csv = (
                sample_imas_dir
                / "all_global_imas_projected_candidate_cascades.csv"
            )
            top_candidates_csv = (
                sample_imas_dir
                / "top_global_imas_projected_candidate_cascades.csv"
            )
            candidates.to_csv(candidates_csv, index=False)
            candidates.head(TOP_CANDIDATES_PER_SAMPLE_EXPORT).to_csv(
                top_candidates_csv, index=False
            )
            candidate_plot = plot_candidate_score_summary(
                candidates,
                sample_id,
                sample_fig_dir / "IMAS_candidate_cascade_score_rank_curve",
            )
            all_candidate_parts.append(candidates)

            sample_summary_rows.append(
                {
                    "sample_id": sample_id,
                    "n_tumor_cells": int(
                        len(tumor_input.sample_positions[sample_id])
                    ),
                    "n_deg_rows": int(len(sample_deg)),
                    "n_deg_sig": int(sample_deg_plot["n_deg_sig"]),
                    "cci_source_table": cci_source,
                    "n_cci_rows_prepared": int(len(cci_prepared)),
                    "n_candidates": int(len(candidates)),
                    "deg_csv": str(sample_deg_csv),
                    "candidate_csv": str(candidates_csv),
                    "status": "success",
                }
            )
            log("[SAMPLE DONE]", sample_id, "candidates=", len(candidates))
        except Exception as exc:
            failed_rows.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            log("[SAMPLE FAILED]", sample_id, repr(exc))

    sample_summary = pd.DataFrame(sample_summary_rows)
    failed = pd.DataFrame(failed_rows)
    sample_summary.to_csv(
        DIR_INPUT_QC / "per_sample_run_summary.tsv", sep="\t", index=False
    )
    failed.to_csv(DIR_INPUT_QC / "failed_samples.tsv", sep="\t", index=False)

    if failed_rows:
        raise RuntimeError(
            "One or more samples failed. See: "
            f"{DIR_INPUT_QC / 'failed_samples.tsv'}"
        )
    if len(all_candidate_parts) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected candidate output for {EXPECTED_SAMPLE_COUNT} samples, "
            f"obtained {len(all_candidate_parts)}."
        )

    # -------------------------------------------------------------------------
    # Canonical all-sample IMAS projection required by the next NDS step.
    # Each candidate retains its tumour sample and sample-specific profile keys.
    # -------------------------------------------------------------------------
    log("\n[STEP 6] Build canonical all-sample candidate table")
    all_candidates = pd.concat(all_candidate_parts, axis=0, ignore_index=True)
    all_candidates = all_candidates.sort_values(
        ["imas_cascade_score", "cci_score", "tf_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    all_candidates.insert(
        0, "global_candidate_rank", np.arange(1, len(all_candidates) + 1)
    )

    all_candidates_csv = (
        DIR_IMAS / "all_global_imas_projected_candidate_cascades.csv"
    )
    top_all_candidates_csv = (
        DIR_IMAS / "top_global_imas_projected_candidate_cascades.csv"
    )
    all_candidates.to_csv(all_candidates_csv, index=False)
    all_candidates.head(TOP_GLOBAL_CANDIDATES_EXPORT).to_csv(
        top_all_candidates_csv, index=False
    )
    all_candidate_plot = plot_candidate_score_summary(
        all_candidates,
        "eight retained kidney-cancer samples",
        DIR_FIG / "IMAS_candidate_cascade_score_rank_curve_all_samples",
    )

    # -------------------------------------------------------------------------
    # Cross-sample consensus is supplementary and does not replace the canonical
    # row-level NDS input above.
    # -------------------------------------------------------------------------
    log("\n[STEP 7] Build cross-sample candidate consensus")
    consensus = build_cross_sample_consensus(
        all_candidates,
        n_expected_samples=EXPECTED_SAMPLE_COUNT,
    )
    consensus_csv = DIR_CONSENSUS / "cross_sample_candidate_consensus.csv"
    top_consensus_csv = DIR_CONSENSUS / "top_cross_sample_candidate_consensus.csv"
    consensus.to_csv(consensus_csv, index=False)
    consensus.head(TOP_GLOBAL_CANDIDATES_EXPORT).to_csv(
        top_consensus_csv, index=False
    )

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------
    summary = {
        "normal_sample_id": NORMAL_SAMPLE_ID,
        "tumor_dataset_name": DATASET_NAME,
        "pooled_tumor_id": POOLED_TUMOR_ID,
        "sample_labels": tumor_input.sample_labels,
        "excluded_sample_numbers": sorted(EXPECTED_EXCLUDED_SAMPLE_NUMBERS),
        "normal_cells": int(normal_expression.n_cells),
        "pooled_tumor_cells": int(len(tumor_input.row_indices)),
        "normal_genes": int(len(normal_expression.genes)),
        "tumor_genes": int(len(tumor_input.gene_names)),
        "pooled_common_genes": int(len(pooled_deg)),
        "stageB_manifest": str(STAGEB_MANIFEST),
        "stageB_checkpoint": str(STAGEB_CHECKPOINT),
        "cci_dir": str(CCI_DIR),
        "receptor_tf_signature_csv": str(RECEPTOR_TF_SIG_CSV),
        "n_pooled_deg_rows": int(len(pooled_deg)),
        "n_all_candidate_rows": int(len(all_candidates)),
        "n_consensus_cascades": int(len(consensus)),
        "canonical_nds_outputs": {
            "deg_csv": str(pooled_deg_csv),
            "candidate_csv": str(all_candidates_csv),
        },
        "additional_outputs": {
            "top_candidate_csv": str(top_all_candidates_csv),
            "consensus_csv": str(consensus_csv),
            "per_sample_summary": str(
                DIR_INPUT_QC / "per_sample_run_summary.tsv"
            ),
            "pooled_deg_plot": pooled_deg_plot,
            "all_candidate_plot": all_candidate_plot,
        },
        "out_root": str(OUT_ROOT),
    }
    summary_json = OUT_ROOT / "GLOBAL_DEG_IMAS_volcano_summary.json"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    log("\n" + "=" * 110)
    log("[DONE]")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    log("\n[NDS REQUIRED OUTPUTS]")
    log(" -", pooled_deg_csv)
    log(" -", all_candidates_csv)
    log("=" * 110)

    show_columns = [
        "global_candidate_rank",
        "tumor_sample_id",
        "cascade",
        "ligand",
        "receptor",
        "tf",
        "sender_cluster",
        "receiver_cluster",
        "cci_source_table",
        "cci_score",
        "tf_score",
        "phenotype_alignment_score",
        "global_signature_deviation_score",
        "axis_hint",
        "imas_cascade_score",
    ]
    show_columns = [column for column in show_columns if column in all_candidates]
    log("\n[TOP CANDIDATE CASCADES]")
    log(all_candidates[show_columns].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
