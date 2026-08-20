from __future__ import annotations

# =============================================================================
# IMAS Network Differential Score (NDS) matrix construction for the completed
# eight-sample kidney-cancer workflow (Sample-7 excluded during Stage-B)
# =============================================================================
# Method retained from the supplied single-sample NDS script:
#   NDS = expression difference
#       + graph-position difference
#       + force-field / propagation-weight difference
#
# Inputs are taken directly from the preceding eight-sample NDS-preparation
# workflow:
#   1) pooled normal-vs-eight-sample DEG table;
#   2) pooled row-level IMAS ligand -> receptor -> TF candidate table;
#   3) per-sample normal-vs-tumour DEG tables;
#   4) the completed Stage-B manifest used to validate the eight retained
#      biological samples.
#
# Multi-sample adaptation:
#   - construct one canonical pooled normal-vs-eight-sample NDS network;
#   - construct eight independent normal-vs-sample NDS networks;
#   - never create or compare a network edge across biological samples;
#   - combine samples only after independent scoring, at the node-score level;
#   - export NDS / signed-NDS / component matrices and a cross-sample consensus;
#   - treat any appearance of Sample-7 as an error instead of silently removing it.
# =============================================================================

import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE = get_project_root()
NORMAL_SAMPLE_ID = "GSE202109_ALL"
POOLED_TUMOR_ID = "Kidney-cancer-8samples-exclude-sample7-pooled"

# Exact completed Stage-B run. This is used for sample identity validation only;
# the NDS matrices are built from the outputs of the preceding preparation step.
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

# Outputs from 9_NDS_preparation_8samples_no_sample7.py.
PREP_ROOT = (
    BASE
    / "LLM-WorldWord"
    / "HiddenMechanism_NormalRef_IMAS"
    / "GLOBAL_GSE202109_ALL_vs_Kidney_cancer_8samples_exclude_sample7_DEG_IMAS_volcano"
)
POOLED_DEG_CSV = (
    PREP_ROOT
    / "01_global_DEG"
    / "global_DEG_normal_all_vs_tumor_all.csv"
)
ALL_CANDIDATE_CSV = (
    PREP_ROOT
    / "03_global_IMAS_projection"
    / "all_global_imas_projected_candidate_cascades.csv"
)
PER_SAMPLE_PREP_ROOT = PREP_ROOT / "04_per_sample"
PREP_SAMPLE_SUMMARY_TSV = PREP_ROOT / "00_input_QC" / "per_sample_run_summary.tsv"

# New NDS output root.
OUT_ROOT = (
    BASE
    / "LLM-WorldWord"
    / "HiddenMechanism_NormalRef_IMAS"
    / "IMAS_NetworkDifferential_GSE202109_ALL_vs_Kidney_cancer_8samples_exclude_sample7_v1"
)
DIR_QC = OUT_ROOT / "00_input_QC"
DIR_EDGE = OUT_ROOT / "01_network_edges"
DIR_NODE = OUT_ROOT / "02_node_scores"
DIR_FIG = OUT_ROOT / "03_figures"
DIR_PER_SAMPLE = OUT_ROOT / "04_per_sample"
DIR_CONSENSUS = OUT_ROOT / "05_cross_sample_consensus"

# Network filtering: retained from the supplied script.
MAX_CANDIDATES_USE = 300000
MIN_CCI_SCORE_PERCENTILE = 0.10
MIN_TF_SCORE_PERCENTILE = 0.10
MIN_NODE_PCT_EITHER = 0.001
MIN_NODE_MEAN_EITHER = 0.001
FILTER_NON_HUMAN_STYLE_TF = True

# NDS component weights: retained exactly.
W_EXPR = 0.34
W_POS = 0.33
W_FIELD = 0.33

# Node-field weights: retained exactly.
W_FIELD_IN = 0.45
W_FIELD_OUT = 0.45
W_FIELD_EXPR = 0.10

# High differential node threshold: retained exactly.
RED_NDS_PERCENTILE = 0.90
MAX_RED_NODES = 120

# Plot settings: retained exactly.
FIG_DPI = 300
LABEL_TOP_N = 35
FC_SHRINK_PSEUDOCOUNT = 0.05
DEG_Y_SOFT_CAP = 80.0
DEG_Y_DISPLAY_MAX = 90.0

# Memory-aware graph-position calculation.
# Up to this many nodes, the supplied dense joint-PCA calculation is used
# exactly. Larger networks use a sparse joint TruncatedSVD fallback while
# retaining the same incoming/outgoing neighbourhood representation.
DENSE_POSITION_MAX_NODES = 4000
SPARSE_SVD_COMPONENTS = 2

# Cross-sample consensus weights. These are applied only after each sample has
# been scored independently and therefore do not modify the NDS method itself.
CONSENSUS_W_MEAN = 0.55
CONSENSUS_W_MAX = 0.25
CONSENSUS_W_COVERAGE = 0.20

SEED = 42
np.random.seed(SEED)


# =============================================================================
# 1. BASIC HELPERS
# =============================================================================


def log(*args) -> None:
    print(*args, flush=True)


def safe_name(value: str, n: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]+", "_", str(value))[:n]


def normalize_sample_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def sample_number_from_label(sample_id: str) -> int | None:
    key = normalize_sample_key(sample_id)
    match = re.search(r"sample0*([0-9]+)$", key)
    return int(match.group(1)) if match else None


def natural_sort_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


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


def robust01(
    values: np.ndarray,
    qlo: float = 1.0,
    qhi: float = 99.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return out
    lo = float(np.nanpercentile(values[finite], qlo))
    hi = float(np.nanpercentile(values[finite], qhi))
    if hi - lo < 1e-12:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((values[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def soft_cap_y(values: np.ndarray, cap: float = 80.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0.0)
    return cap * (1.0 - np.exp(-values / cap))


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if len(pvalues) == 0:
        return pvalues
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def empirical_p_from_score(score: np.ndarray) -> np.ndarray:
    """Higher NDS corresponds to a smaller one-sided empirical rank P."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    if n == 0:
        return np.zeros((0,), dtype=float)
    order = np.argsort(-score)
    pvalues = np.empty(n, dtype=float)
    for rank, index in enumerate(order, start=1):
        pvalues[index] = rank / (n + 1.0)
    return pvalues


def is_human_style_symbol(gene: str) -> bool:
    gene = str(gene).strip()
    if gene in {"", "NA", "nan", "None", "<NA>"}:
        return False
    if len(gene) > 25:
        return False
    if re.search(r"[a-z]", gene):
        return False
    return re.match(r"^[A-Z0-9.\-]+$", gene) is not None


def join_unique(values: Iterable[str], sep: str = ";") -> str:
    cleaned = {
        str(value).strip()
        for value in values
        if str(value).strip() not in {"", "nan", "None", "<NA>"}
    }
    return sep.join(sorted(cleaned, key=natural_sort_key))


def ensure_directories(directories: Sequence[Path]) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. INPUT VALIDATION
# =============================================================================


def load_completed_sample_labels() -> Tuple[pd.DataFrame, List[str]]:
    if not STAGEB_MANIFEST.exists():
        raise FileNotFoundError(STAGEB_MANIFEST)
    manifest = pd.read_csv(STAGEB_MANIFEST, sep="\t", low_memory=False)
    required = {"sample_id", "cell_id", "row_index"}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(
            f"{STAGEB_MANIFEST.name} missing columns: {sorted(missing)}"
        )
    manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
    manifest["row_index"] = pd.to_numeric(
        manifest["row_index"], errors="raise"
    ).astype(np.int64)
    manifest = manifest.sort_values("row_index").reset_index(drop=True)

    sample_labels = sorted(
        manifest["sample_id"].unique().tolist(),
        key=natural_sort_key,
    )
    if len(sample_labels) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} samples in Stage-B manifest; "
            f"found {len(sample_labels)}: {sample_labels}"
        )
    excluded_hits = [
        sample_id
        for sample_id in sample_labels
        if sample_number_from_label(sample_id) in EXPECTED_EXCLUDED_SAMPLE_NUMBERS
    ]
    if excluded_hits:
        raise ValueError(
            "The completed Stage-B manifest unexpectedly contains Sample-7: "
            f"{excluded_hits}"
        )

    log("[MANIFEST]", STAGEB_MANIFEST)
    log("[SAMPLES]", sample_labels)
    log(manifest.groupby("sample_id").size().to_string())
    return manifest, sample_labels


def validate_deg_table(deg: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "gene",
        "mean_normal",
        "mean_tumor",
        "pct_normal",
        "pct_tumor",
        "fdr",
    }
    missing = required.difference(deg.columns)
    if missing:
        raise KeyError(f"{label} DEG table missing columns: {sorted(missing)}")

    frame = deg.copy()
    frame["gene"] = frame["gene"].astype(str)
    numeric = [
        "mean_normal",
        "mean_tumor",
        "pct_normal",
        "pct_tumor",
        "fdr",
        "log2FC_tumor_vs_normal",
        "DEG_rank_score",
    ]
    for column in numeric:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "log2FC_tumor_vs_normal" not in frame.columns:
        frame["log2FC_tumor_vs_normal"] = np.log2(
            (frame["mean_tumor"].fillna(0.0) + 0.02)
            / (frame["mean_normal"].fillna(0.0) + 0.02)
        )
    if "DEG_rank_score" not in frame.columns:
        neglog = -np.log10(np.clip(frame["fdr"].fillna(1.0), 1e-300, 1.0))
        frame["DEG_rank_score"] = (
            0.45 * rank01(np.abs(frame["log2FC_tumor_vs_normal"]))
            + 0.45 * rank01(neglog)
            + 0.10
            * rank01(
                np.maximum(
                    frame["pct_normal"].fillna(0.0),
                    frame["pct_tumor"].fillna(0.0),
                )
            )
        )

    frame = frame.drop_duplicates("gene", keep="first").reset_index(drop=True)
    if len(frame) == 0:
        raise RuntimeError(f"{label} DEG table is empty.")
    return frame


def validate_candidate_table(
    candidates: pd.DataFrame,
    expected_samples: Sequence[str],
) -> pd.DataFrame:
    required = {"ligand", "receptor", "tf", "tumor_sample_id"}
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(
            f"{ALL_CANDIDATE_CSV.name} missing columns: {sorted(missing)}"
        )

    frame = candidates.copy()
    for column in ["ligand", "receptor", "tf", "tumor_sample_id"]:
        frame[column] = frame[column].astype(str).str.strip()

    observed = sorted(
        frame["tumor_sample_id"].unique().tolist(),
        key=natural_sort_key,
    )
    expected = sorted(map(str, expected_samples), key=natural_sort_key)
    if observed != expected:
        raise ValueError(
            "Candidate sample labels differ from the completed Stage-B manifest.\n"
            f"Expected: {expected}\nObserved: {observed}"
        )
    bad_sample7 = [
        sample_id
        for sample_id in observed
        if sample_number_from_label(sample_id) in EXPECTED_EXCLUDED_SAMPLE_NUMBERS
    ]
    if bad_sample7:
        raise ValueError(f"Candidate table unexpectedly contains Sample-7: {bad_sample7}")

    # Validate sample-specific profile keys when available.
    for role in ["sender", "receiver"]:
        key_column = f"{role}_profile_key"
        if key_column not in frame.columns:
            continue
        values = frame[key_column].astype(str)
        expected_prefix = frame["tumor_sample_id"].astype(str) + "::"
        malformed = ~np.fromiter(
            (
                value.startswith(prefix)
                for value, prefix in zip(values.tolist(), expected_prefix.tolist())
            ),
            dtype=bool,
            count=len(frame),
        )
        if malformed.any():
            examples = frame.loc[
                malformed,
                ["tumor_sample_id", key_column],
            ].head(5)
            raise ValueError(
                f"Malformed or cross-sample {key_column}:\n"
                + examples.to_string(index=False)
            )

    return frame.reset_index(drop=True)


def per_sample_deg_path(sample_id: str) -> Path:
    return (
        PER_SAMPLE_PREP_ROOT
        / safe_name(sample_id)
        / "01_global_DEG"
        / "global_DEG_normal_all_vs_tumor_all.csv"
    )


# =============================================================================
# 3. CANDIDATE PREPARATION AND EXPRESSION ATTACHMENT
# =============================================================================


def prepare_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the supplied NDS candidate filtering within the current network."""
    frame = candidates.copy()
    for column in ["ligand", "receptor", "tf"]:
        if column not in frame.columns:
            raise KeyError(f"Candidate table missing column: {column}")
        frame[column] = frame[column].astype(str).str.strip()

    frame = frame[
        (frame["ligand"] != "")
        & (frame["receptor"] != "")
        & (frame["tf"] != "")
        & (~frame["tf"].isin(["NA", "nan", "None", "<NA>"]))
    ].copy()
    if FILTER_NON_HUMAN_STYLE_TF:
        frame = frame[frame["tf"].map(is_human_style_symbol)].copy()

    numeric_columns = [
        "cci_score",
        "tf_score",
        "phenotype_alignment_score",
        "global_signature_deviation_score",
        "axis_hint_score",
        "imas_cascade_score",
        "hidden_mechanism_score",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    # Retain the supplied method: percentiles are recalculated inside the
    # network currently being scored. Per-sample runs therefore cannot be
    # influenced by another biological sample.
    frame["cci_score_percentile_for_filter"] = rank01(
        frame["cci_score"].to_numpy(dtype=float)
    )
    frame["tf_score_percentile_for_filter"] = rank01(
        frame["tf_score"].to_numpy(dtype=float)
    )
    frame = frame[
        (frame["cci_score_percentile_for_filter"] >= MIN_CCI_SCORE_PERCENTILE)
        & (frame["tf_score_percentile_for_filter"] >= MIN_TF_SCORE_PERCENTILE)
    ].copy()
    frame = (
        frame.sort_values("imas_cascade_score", ascending=False)
        .head(MAX_CANDIDATES_USE)
        .reset_index(drop=True)
    )
    if len(frame) == 0:
        raise RuntimeError("No candidate cascades remain after NDS filtering.")
    return frame


def expression_table_from_deg(deg: pd.DataFrame) -> pd.DataFrame:
    frame = validate_deg_table(deg, "expression source")
    return frame[
        [
            "gene",
            "mean_normal",
            "pct_normal",
            "mean_tumor",
            "pct_tumor",
            "log2FC_tumor_vs_normal",
            "fdr",
            "DEG_rank_score",
        ]
    ].copy()


def attach_node_expression(
    candidates: pd.DataFrame,
    deg: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genes: List[str] = []
    for role in ["ligand", "receptor", "tf"]:
        genes.extend(candidates[role].astype(str).tolist())
    genes = sorted(set(genes))

    expression_all = expression_table_from_deg(deg)
    expression = pd.DataFrame({"gene": genes}).merge(
        expression_all,
        on="gene",
        how="left",
    )
    for column in [
        "mean_normal",
        "pct_normal",
        "mean_tumor",
        "pct_tumor",
        "log2FC_tumor_vs_normal",
        "DEG_rank_score",
    ]:
        expression[column] = pd.to_numeric(
            expression[column], errors="coerce"
        ).fillna(0.0)
    expression["fdr"] = pd.to_numeric(
        expression["fdr"], errors="coerce"
    ).fillna(1.0)
    expression["mean_either_max"] = expression[
        ["mean_normal", "mean_tumor"]
    ].max(axis=1)
    expression["pct_either_max"] = expression[
        ["pct_normal", "pct_tumor"]
    ].max(axis=1)
    expression["node_expr_supported"] = (
        (expression["mean_either_max"] >= MIN_NODE_MEAN_EITHER)
        & (expression["pct_either_max"] >= MIN_NODE_PCT_EITHER)
    )

    output = candidates.copy()
    for role in ["ligand", "receptor", "tf"]:
        renamed = expression.rename(
            columns={
                "gene": role,
                "mean_normal": f"{role}_normal_mean",
                "pct_normal": f"{role}_normal_pct",
                "mean_tumor": f"{role}_tumor_mean",
                "pct_tumor": f"{role}_tumor_pct",
                "node_expr_supported": f"{role}_expr_supported",
            }
        )
        keep = [
            role,
            f"{role}_normal_mean",
            f"{role}_normal_pct",
            f"{role}_tumor_mean",
            f"{role}_tumor_pct",
            f"{role}_expr_supported",
        ]
        output = output.merge(renamed[keep], on=role, how="left")
        for column in keep:
            if column == role:
                continue
            if column.endswith("_expr_supported"):
                output[column] = output[column].fillna(False).astype(bool)
            else:
                output[column] = pd.to_numeric(
                    output[column], errors="coerce"
                ).fillna(0.0)

    output["cascade_expr_supported_either"] = (
        output["ligand_expr_supported"].astype(bool)
        & output["receptor_expr_supported"].astype(bool)
        & output["tf_expr_supported"].astype(bool)
    )
    output = output[output["cascade_expr_supported_either"]].reset_index(drop=True)
    if len(output) == 0:
        raise RuntimeError(
            "No candidate cascades have expression support for ligand, receptor and TF."
        )
    return output, expression


def pair_expr_weight(
    mean_a: np.ndarray,
    pct_a: np.ndarray,
    mean_b: np.ndarray,
    pct_b: np.ndarray,
) -> np.ndarray:
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    pct_a = np.asarray(pct_a, dtype=float)
    pct_b = np.asarray(pct_b, dtype=float)
    mean_part = np.sqrt(np.maximum(mean_a, 0.0) * np.maximum(mean_b, 0.0))
    pct_part = np.sqrt(
        np.clip(pct_a, 0.0, 1.0) * np.clip(pct_b, 0.0, 1.0)
    )
    return mean_part * pct_part


# =============================================================================
# 4. BUILD NORMAL AND TUMOUR NETWORKS
# =============================================================================


def build_condition_edges(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy().reset_index(drop=True)
    frame["cci01"] = rank01(frame["cci_score"].to_numpy(dtype=float))
    frame["tf01"] = rank01(frame["tf_score"].to_numpy(dtype=float))

    normal_lr_expr = pair_expr_weight(
        frame["ligand_normal_mean"],
        frame["ligand_normal_pct"],
        frame["receptor_normal_mean"],
        frame["receptor_normal_pct"],
    )
    tumor_lr_expr = pair_expr_weight(
        frame["ligand_tumor_mean"],
        frame["ligand_tumor_pct"],
        frame["receptor_tumor_mean"],
        frame["receptor_tumor_pct"],
    )
    normal_rtf_expr = pair_expr_weight(
        frame["receptor_normal_mean"],
        frame["receptor_normal_pct"],
        frame["tf_normal_mean"],
        frame["tf_normal_pct"],
    )
    tumor_rtf_expr = pair_expr_weight(
        frame["receptor_tumor_mean"],
        frame["receptor_tumor_pct"],
        frame["tf_tumor_mean"],
        frame["tf_tumor_pct"],
    )

    rows: List[dict] = []
    for index, record in frame.iterrows():
        ligand = str(record["ligand"])
        receptor = str(record["receptor"])
        tf = str(record["tf"])
        axis = str(record.get("axis_hint", "unassigned"))
        cascade = str(record.get("cascade", f"{ligand}->{receptor}->{tf}"))
        sample_id = str(record.get("tumor_sample_id", POOLED_TUMOR_ID))

        rows.append(
            {
                "source": ligand,
                "target": receptor,
                "edge_type": "LR",
                "axis_hint": axis,
                "cascade": cascade,
                "tumor_sample_id": sample_id,
                "base_score": float(record["cci01"]),
                "normal_weight": float(record["cci01"] * normal_lr_expr[index]),
                "tumor_weight": float(record["cci01"] * tumor_lr_expr[index]),
            }
        )
        rows.append(
            {
                "source": receptor,
                "target": tf,
                "edge_type": "R_TF",
                "axis_hint": axis,
                "cascade": cascade,
                "tumor_sample_id": sample_id,
                "base_score": float(record["tf01"]),
                "normal_weight": float(record["tf01"] * normal_rtf_expr[index]),
                "tumor_weight": float(record["tf01"] * tumor_rtf_expr[index]),
            }
        )

    edges = pd.DataFrame(rows)
    if len(edges) == 0:
        raise RuntimeError("No network edges were constructed.")

    aggregated = (
        edges.groupby(["source", "target", "edge_type"], as_index=False)
        .agg(
            normal_weight=("normal_weight", "sum"),
            tumor_weight=("tumor_weight", "sum"),
            base_score=("base_score", "max"),
            axis_hints=("axis_hint", join_unique),
            example_cascade=("cascade", "first"),
            n_supporting_cascades=("cascade", "count"),
            n_supporting_samples=("tumor_sample_id", "nunique"),
            supporting_samples=("tumor_sample_id", join_unique),
        )
        .reset_index(drop=True)
    )
    aggregated["edge_delta_tumor_minus_normal"] = (
        aggregated["tumor_weight"] - aggregated["normal_weight"]
    )
    aggregated["edge_abs_delta"] = np.abs(
        aggregated["edge_delta_tumor_minus_normal"].to_numpy(dtype=float)
    )
    aggregated["edge_log2_ratio_tumor_vs_normal"] = np.log2(
        (aggregated["tumor_weight"].to_numpy(dtype=float) + 1e-8)
        / (aggregated["normal_weight"].to_numpy(dtype=float) + 1e-8)
    )
    return aggregated


# =============================================================================
# 5. NODE-LEVEL NDS
# =============================================================================


def build_adjacency_dense(
    edges: pd.DataFrame,
    nodes: Sequence[str],
    weight_column: str,
) -> np.ndarray:
    node_to_index = {node: index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for record in edges.itertuples(index=False):
        source = str(record.source)
        target = str(record.target)
        if source not in node_to_index or target not in node_to_index:
            continue
        matrix[node_to_index[source], node_to_index[target]] += float(
            getattr(record, weight_column)
        )
    return matrix


def build_adjacency_sparse(
    edges: pd.DataFrame,
    nodes: Sequence[str],
    weight_column: str,
) -> sp.csr_matrix:
    node_to_index = {node: index for index, node in enumerate(nodes)}
    row: List[int] = []
    col: List[int] = []
    data: List[float] = []
    for record in edges.itertuples(index=False):
        source = str(record.source)
        target = str(record.target)
        if source not in node_to_index or target not in node_to_index:
            continue
        row.append(node_to_index[source])
        col.append(node_to_index[target])
        data.append(float(getattr(record, weight_column)))
    matrix = sp.coo_matrix(
        (np.asarray(data, dtype=np.float32), (row, col)),
        shape=(len(nodes), len(nodes)),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def row_normalize_dense(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(denominator, 1e-12)


def row_normalize_sparse(matrix: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    good = row_sum > 1e-12
    inverse[good] = 1.0 / row_sum[good]
    return sp.diags(inverse, format="csr") @ matrix


def cosine_distance_rows_dense(
    normal: np.ndarray,
    tumor: np.ndarray,
) -> np.ndarray:
    numerator = np.sum(normal * tumor, axis=1)
    denominator = (
        np.sqrt(np.sum(normal * normal, axis=1))
        * np.sqrt(np.sum(tumor * tumor, axis=1))
    )
    cosine = numerator / np.maximum(denominator, 1e-12)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def cosine_distance_rows_sparse(
    normal: sp.csr_matrix,
    tumor: sp.csr_matrix,
) -> np.ndarray:
    numerator = np.asarray(normal.multiply(tumor).sum(axis=1)).reshape(-1)
    norm_normal = np.sqrt(
        np.asarray(normal.multiply(normal).sum(axis=1)).reshape(-1)
    )
    norm_tumor = np.sqrt(
        np.asarray(tumor.multiply(tumor).sum(axis=1)).reshape(-1)
    )
    cosine = numerator / np.maximum(norm_normal * norm_tumor, 1e-12)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def pca_2d_joint_dense(
    normal_features: np.ndarray,
    tumor_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact joint-PCA implementation retained from the supplied script."""
    combined = np.vstack([normal_features, tumor_features]).astype(np.float64)
    combined = combined - combined.mean(axis=0, keepdims=True)
    standard_deviation = combined.std(axis=0)
    keep = standard_deviation > 1e-12
    if keep.sum() < 2:
        zeros = np.zeros((normal_features.shape[0], 2), dtype=np.float32)
        return zeros.copy(), zeros.copy()
    combined = combined[:, keep]
    combined = combined / (combined.std(axis=0, keepdims=True) + 1e-12)
    left, singular, _ = np.linalg.svd(combined, full_matrices=False)
    coordinates = left[:, :2] * singular[:2]
    n_normal = normal_features.shape[0]
    return coordinates[:n_normal], coordinates[n_normal:]


def pca_2d_joint_sparse(
    normal_features: sp.csr_matrix,
    tumor_features: sp.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-safe joint embedding for unusually large candidate networks."""
    combined = sp.vstack([normal_features, tumor_features], format="csr")
    if combined.shape[1] < 2 or combined.nnz == 0:
        zeros = np.zeros((normal_features.shape[0], 2), dtype=np.float32)
        return zeros.copy(), zeros.copy()
    n_components = min(
        SPARSE_SVD_COMPONENTS,
        max(1, combined.shape[0] - 1),
        max(1, combined.shape[1] - 1),
    )
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    coordinates = svd.fit_transform(combined)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack(
            [coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)]
        )
    n_normal = normal_features.shape[0]
    return coordinates[:n_normal, :2], coordinates[n_normal:, :2]


def calculate_position_and_strengths(
    edges: pd.DataFrame,
    nodes: Sequence[str],
) -> Dict[str, np.ndarray]:
    n_nodes = len(nodes)
    if n_nodes <= DENSE_POSITION_MAX_NODES:
        normal_adjacency = build_adjacency_dense(edges, nodes, "normal_weight")
        tumor_adjacency = build_adjacency_dense(edges, nodes, "tumor_weight")
        normal_features = np.hstack(
            [normal_adjacency, normal_adjacency.T]
        ).astype(np.float32)
        tumor_features = np.hstack(
            [tumor_adjacency, tumor_adjacency.T]
        ).astype(np.float32)
        normal_normalized = row_normalize_dense(normal_features)
        tumor_normalized = row_normalize_dense(tumor_features)
        neighborhood_distance = cosine_distance_rows_dense(
            normal_normalized,
            tumor_normalized,
        )
        normal_coordinates, tumor_coordinates = pca_2d_joint_dense(
            normal_normalized,
            tumor_normalized,
        )
        in_normal = normal_adjacency.sum(axis=0).astype(float)
        out_normal = normal_adjacency.sum(axis=1).astype(float)
        in_tumor = tumor_adjacency.sum(axis=0).astype(float)
        out_tumor = tumor_adjacency.sum(axis=1).astype(float)
        mode = "dense_exact"
    else:
        log(
            "[POSITION WARN]",
            f"n_nodes={n_nodes} exceeds {DENSE_POSITION_MAX_NODES};",
            "using sparse joint-SVD fallback.",
        )
        normal_adjacency = build_adjacency_sparse(edges, nodes, "normal_weight")
        tumor_adjacency = build_adjacency_sparse(edges, nodes, "tumor_weight")
        normal_features = sp.hstack(
            [normal_adjacency, normal_adjacency.T], format="csr"
        )
        tumor_features = sp.hstack(
            [tumor_adjacency, tumor_adjacency.T], format="csr"
        )
        normal_normalized = row_normalize_sparse(normal_features)
        tumor_normalized = row_normalize_sparse(tumor_features)
        neighborhood_distance = cosine_distance_rows_sparse(
            normal_normalized,
            tumor_normalized,
        )
        normal_coordinates, tumor_coordinates = pca_2d_joint_sparse(
            normal_normalized,
            tumor_normalized,
        )
        in_normal = np.asarray(normal_adjacency.sum(axis=0)).reshape(-1)
        out_normal = np.asarray(normal_adjacency.sum(axis=1)).reshape(-1)
        in_tumor = np.asarray(tumor_adjacency.sum(axis=0)).reshape(-1)
        out_tumor = np.asarray(tumor_adjacency.sum(axis=1)).reshape(-1)
        mode = "sparse_fallback"

    pca_position_distance = np.sqrt(
        np.sum((tumor_coordinates - normal_coordinates) ** 2, axis=1)
    )
    position_diff_raw = (
        0.55 * robust01(neighborhood_distance)
        + 0.45 * robust01(pca_position_distance)
    )
    return {
        "neighborhood_cosine_distance": neighborhood_distance,
        "pca_position_distance": pca_position_distance,
        "position_diff_raw": position_diff_raw,
        "normal_x": normal_coordinates[:, 0],
        "normal_y": normal_coordinates[:, 1],
        "tumor_x": tumor_coordinates[:, 0],
        "tumor_y": tumor_coordinates[:, 1],
        "normal_in_strength": in_normal,
        "normal_out_strength": out_normal,
        "tumor_in_strength": in_tumor,
        "tumor_out_strength": out_tumor,
        "position_calculation_mode": np.repeat(mode, n_nodes),
    }


def compute_node_scores(
    edges: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    nodes = sorted(
        set(edges["source"].astype(str))
        | set(edges["target"].astype(str))
    )
    if len(nodes) == 0:
        raise RuntimeError("The condition-specific network contains no nodes.")

    graph = calculate_position_and_strengths(edges, nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    expression_lookup = expression.set_index("gene").to_dict(orient="index")

    normal_mean = np.zeros(len(nodes), dtype=float)
    tumor_mean = np.zeros(len(nodes), dtype=float)
    normal_pct = np.zeros(len(nodes), dtype=float)
    tumor_pct = np.zeros(len(nodes), dtype=float)
    for node, index in node_to_index.items():
        values = expression_lookup.get(node, {})
        normal_mean[index] = float(values.get("mean_normal", 0.0))
        tumor_mean[index] = float(values.get("mean_tumor", 0.0))
        normal_pct[index] = float(values.get("pct_normal", 0.0))
        tumor_pct[index] = float(values.get("pct_tumor", 0.0))

    normal_expression_strength = np.sqrt(
        np.maximum(normal_mean, 0.0) * np.clip(normal_pct, 0.0, 1.0)
    )
    tumor_expression_strength = np.sqrt(
        np.maximum(tumor_mean, 0.0) * np.clip(tumor_pct, 0.0, 1.0)
    )

    normal_field = (
        W_FIELD_IN * graph["normal_in_strength"]
        + W_FIELD_OUT * graph["normal_out_strength"]
        + W_FIELD_EXPR * normal_expression_strength
    )
    tumor_field = (
        W_FIELD_IN * graph["tumor_in_strength"]
        + W_FIELD_OUT * graph["tumor_out_strength"]
        + W_FIELD_EXPR * tumor_expression_strength
    )
    field_log2_ratio = np.log2(
        (tumor_field + 1e-8) / (normal_field + 1e-8)
    )
    field_abs_diff_raw = np.abs(field_log2_ratio)

    expression_log2fc = np.log2(
        (tumor_mean + 0.02) / (normal_mean + 0.02)
    )
    expression_pct_delta = tumor_pct - normal_pct
    expression_diff_raw = (
        np.abs(expression_log2fc)
        * np.sqrt(np.maximum(tumor_pct, normal_pct) + 1e-6)
    )
    expression_diff_raw += 0.50 * np.abs(expression_pct_delta)

    expression_diff01 = robust01(expression_diff_raw)
    position_diff01 = robust01(graph["position_diff_raw"])
    field_diff01 = robust01(field_abs_diff_raw)
    nds = (
        W_EXPR * expression_diff01
        + W_POS * position_diff01
        + W_FIELD * field_diff01
    )

    sign_expression = np.sign(expression_log2fc)
    sign_field = np.sign(field_log2_ratio)
    signed_direction = np.where(
        np.abs(expression_log2fc) >= 0.05,
        sign_expression,
        sign_field,
    )
    signed_direction = np.where(signed_direction == 0, 1.0, signed_direction)
    signed_nds = signed_direction * nds

    empirical_p = empirical_p_from_score(nds)
    empirical_q = bh_fdr(empirical_p)

    output = pd.DataFrame(
        {
            "node": nodes,
            "normal_mean": normal_mean,
            "tumor_mean": tumor_mean,
            "normal_pct": normal_pct,
            "tumor_pct": tumor_pct,
            "expr_log2FC_tumor_vs_normal": expression_log2fc,
            "expr_pct_delta_tumor_minus_normal": expression_pct_delta,
            "expr_diff_raw": expression_diff_raw,
            "expr_diff01": expression_diff01,
            "normal_in_strength": graph["normal_in_strength"],
            "tumor_in_strength": graph["tumor_in_strength"],
            "normal_out_strength": graph["normal_out_strength"],
            "tumor_out_strength": graph["tumor_out_strength"],
            "normal_field_strength": normal_field,
            "tumor_field_strength": tumor_field,
            "field_log2_ratio_tumor_vs_normal": field_log2_ratio,
            "field_abs_diff_raw": field_abs_diff_raw,
            "field_diff01": field_diff01,
            "neighborhood_cosine_distance": graph[
                "neighborhood_cosine_distance"
            ],
            "pca_position_distance": graph["pca_position_distance"],
            "position_diff_raw": graph["position_diff_raw"],
            "position_diff01": position_diff01,
            "normal_x": graph["normal_x"],
            "normal_y": graph["normal_y"],
            "tumor_x": graph["tumor_x"],
            "tumor_y": graph["tumor_y"],
            "position_calculation_mode": graph["position_calculation_mode"],
            "NDS": nds,
            "signed_NDS": signed_nds,
            "empirical_p_NDS": empirical_p,
            "empirical_q_NDS": empirical_q,
        }
    )
    output["NDS_percentile"] = rank01(output["NDS"].to_numpy(dtype=float))
    output["dominant_NDS_component"] = np.asarray(
        ["expression", "position", "field"], dtype=object
    )[
        np.argmax(
            np.column_stack(
                [expression_diff01, position_diff01, field_diff01]
            ),
            axis=1,
        )
    ]

    role_rows: List[dict] = []
    for record in edges.itertuples(index=False):
        role_rows.append(
            {
                "node": str(record.source),
                "role": "source",
                "axis": str(getattr(record, "axis_hints", "")),
                "n_samples": int(getattr(record, "n_supporting_samples", 1)),
            }
        )
        role_rows.append(
            {
                "node": str(record.target),
                "role": "target",
                "axis": str(getattr(record, "axis_hints", "")),
                "n_samples": int(getattr(record, "n_supporting_samples", 1)),
            }
        )
    role_frame = pd.DataFrame(role_rows)
    if len(role_frame) > 0:
        role_aggregated = role_frame.groupby("node", as_index=False).agg(
            network_roles=("role", join_unique),
            axis_hints=("axis", lambda x: join_unique(";".join(map(str, x)).split(";"))),
            max_edge_sample_support=("n_samples", "max"),
        )
        output = output.merge(role_aggregated, on="node", how="left")
    else:
        output["network_roles"] = ""
        output["axis_hints"] = ""
        output["max_edge_sample_support"] = 0

    output["network_roles"] = output["network_roles"].fillna("")
    output["axis_hints"] = output["axis_hints"].fillna("")
    output["max_edge_sample_support"] = pd.to_numeric(
        output["max_edge_sample_support"], errors="coerce"
    ).fillna(0).astype(int)
    output = output.sort_values("NDS", ascending=False).reset_index(drop=True)
    output.insert(0, "NDS_rank", np.arange(1, len(output) + 1))
    return output


# =============================================================================
# 6. PLOTS
# =============================================================================


def plot_network_volcano(
    node_scores: pd.DataFrame,
    output_base: Path,
    title_label: str,
) -> Dict[str, object]:
    frame = node_scores.copy()
    red = frame[frame["NDS_percentile"] >= RED_NDS_PERCENTILE].copy()
    red = red.sort_values("NDS", ascending=False).head(MAX_RED_NODES)
    red_nodes = set(red["node"].astype(str))
    frame["is_red"] = frame["node"].astype(str).isin(red_nodes)
    y = -np.log10(
        np.clip(frame["empirical_p_NDS"].to_numpy(dtype=float), 1e-300, 1.0)
    )

    figure, axis = plt.subplots(figsize=(9.2, 7.8))
    background = frame[~frame["is_red"]]
    highlighted = frame[frame["is_red"]]
    axis.scatter(
        background["signed_NDS"],
        y[~frame["is_red"].to_numpy()],
        s=18,
        c="#BDBDBD",
        alpha=0.60,
        linewidths=0,
        zorder=1,
        label="All network nodes",
    )
    axis.scatter(
        highlighted["signed_NDS"],
        -np.log10(
            np.clip(
                highlighted["empirical_p_NDS"].to_numpy(dtype=float),
                1e-300,
                1.0,
            )
        ),
        s=42,
        c="#D62728",
        alpha=0.95,
        edgecolors="#7F0000",
        linewidths=0.25,
        zorder=3,
        label="High-NDS nodes",
    )
    for _, record in highlighted.sort_values("NDS", ascending=False).head(
        LABEL_TOP_N
    ).iterrows():
        axis.text(
            float(record["signed_NDS"]),
            -np.log10(max(float(record["empirical_p_NDS"]), 1e-300)),
            str(record["node"]),
            fontsize=8,
            color="#7F0000",
            ha="center",
            va="bottom",
            zorder=4,
        )
    axis.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Signed IMAS Network Differential Score", fontsize=12)
    axis.set_ylabel("-log10(empirical P from NDS rank)", fontsize=12)
    axis.set_title(
        f"IMAS network differential volcano\n{title_label}",
        fontsize=14,
        pad=14,
    )
    axis.legend(frameon=False, loc="upper right", fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    figure.tight_layout()

    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    figure.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "png": str(png),
        "svg": str(svg),
        "pdf": str(pdf),
        "n_red_nodes": int(len(highlighted)),
    }


def plot_deg_background_with_nds_red(
    deg: pd.DataFrame,
    node_scores: pd.DataFrame,
    output_base: Path,
    title_label: str,
) -> Dict[str, object]:
    frame = validate_deg_table(deg, title_label)
    frame["log2FC_shrunk"] = np.log2(
        (frame["mean_tumor"].to_numpy(dtype=float) + FC_SHRINK_PSEUDOCOUNT)
        / (frame["mean_normal"].to_numpy(dtype=float) + FC_SHRINK_PSEUDOCOUNT)
    )
    frame["neglog10_fdr_raw"] = -np.log10(
        np.clip(frame["fdr"].to_numpy(dtype=float), 1e-300, 1.0)
    )
    frame["neglog10_fdr_soft"] = soft_cap_y(
        frame["neglog10_fdr_raw"].to_numpy(dtype=float),
        cap=DEG_Y_SOFT_CAP,
    )

    red_nodes = (
        node_scores[node_scores["NDS_percentile"] >= RED_NDS_PERCENTILE]
        .sort_values("NDS", ascending=False)
        .head(MAX_RED_NODES)
    )
    red_set = set(red_nodes["node"].astype(str))
    frame["NDS_red"] = frame["gene"].astype(str).isin(red_set)
    nds_lookup = node_scores.set_index("node")["NDS"].to_dict()
    frame["NDS"] = frame["gene"].map(nds_lookup).fillna(0.0)

    figure, axis = plt.subplots(figsize=(9.2, 7.8))
    axis.scatter(
        frame["log2FC_shrunk"],
        frame["neglog10_fdr_soft"],
        s=7,
        c="#C7C7C7",
        alpha=0.45,
        linewidths=0,
        zorder=1,
        label="Ordinary DEG background",
    )
    highlighted = frame[frame["NDS_red"]]
    if len(highlighted) > 0:
        axis.scatter(
            highlighted["log2FC_shrunk"],
            highlighted["neglog10_fdr_soft"],
            s=38,
            c="#D62728",
            alpha=0.95,
            edgecolors="#7F0000",
            linewidths=0.25,
            zorder=3,
            label="High-NDS IMAS nodes",
        )
        for _, record in highlighted.sort_values("NDS", ascending=False).head(
            LABEL_TOP_N
        ).iterrows():
            axis.text(
                float(record["log2FC_shrunk"]),
                float(record["neglog10_fdr_soft"]),
                str(record["gene"]),
                fontsize=8,
                color="#7F0000",
                ha="center",
                va="bottom",
                zorder=4,
            )

    axis.axhline(
        soft_cap_y(
            np.asarray([-np.log10(0.05)]), cap=DEG_Y_SOFT_CAP
        )[0],
        color="#777777",
        linestyle="--",
        linewidth=0.8,
    )
    axis.axvline(0.25, color="#777777", linestyle="--", linewidth=0.8)
    axis.axvline(-0.25, color="#777777", linestyle="--", linewidth=0.8)
    finite_x = frame["log2FC_shrunk"].replace([np.inf, -np.inf], np.nan).dropna()
    xmax = float(np.nanpercentile(np.abs(finite_x), 99.5)) if len(finite_x) else 2.0
    xmax = min(max(xmax, 2.0), 8.0)
    axis.set_xlim(-xmax * 1.08, xmax * 1.08)
    axis.set_ylim(-1, DEG_Y_DISPLAY_MAX)
    axis.set_xlabel("Shrunken log2FC tumor vs normal", fontsize=12)
    axis.set_ylabel("Soft-capped -log10(FDR)", fontsize=12)
    axis.set_title(
        f"DEG background with IMAS network-differential nodes\n{title_label}",
        fontsize=14,
        pad=14,
    )
    axis.legend(frameon=False, loc="upper right", fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    figure.tight_layout()

    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    figure.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {
        "png": str(png),
        "svg": str(svg),
        "pdf": str(pdf),
        "n_red_nodes_on_deg": int(len(highlighted)),
    }


# =============================================================================
# 7. ONE COMPLETE NDS RUN
# =============================================================================


def run_one_nds_network(
    run_label: str,
    deg: pd.DataFrame,
    candidates_raw: pd.DataFrame,
    output_root: Path,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    edge_dir = output_root / "01_network_edges"
    node_dir = output_root / "02_node_scores"
    figure_dir = output_root / "03_figures"
    ensure_directories([edge_dir, node_dir, figure_dir])

    log("[NDS RUN]", run_label)
    candidates = prepare_candidates(candidates_raw)
    candidates_expression, node_expression = attach_node_expression(
        candidates,
        deg,
    )

    node_expression_csv = node_dir / "node_expression_normal_tumor.csv"
    candidate_expression_csv = (
        edge_dir / "candidate_cascades_with_normal_tumor_expression.csv"
    )
    node_expression.to_csv(node_expression_csv, index=False)
    candidates_expression.to_csv(candidate_expression_csv, index=False)

    edges = build_condition_edges(candidates_expression)
    edges_csv = edge_dir / "condition_specific_network_edges_normal_vs_tumor.csv"
    edges.to_csv(edges_csv, index=False)

    node_scores = compute_node_scores(edges, node_expression)
    node_scores_csv = node_dir / "IMAS_network_differential_node_scores_NDS.csv"
    top_nodes_csv = node_dir / "IMAS_network_differential_top_nodes_NDS.csv"
    red_nodes_csv = node_dir / "IMAS_network_differential_red_nodes.csv"
    node_scores.to_csv(node_scores_csv, index=False)
    node_scores.head(300).to_csv(top_nodes_csv, index=False)
    red_nodes = (
        node_scores[node_scores["NDS_percentile"] >= RED_NDS_PERCENTILE]
        .sort_values("NDS", ascending=False)
        .head(MAX_RED_NODES)
    )
    red_nodes.to_csv(red_nodes_csv, index=False)

    network_plot = plot_network_volcano(
        node_scores,
        figure_dir / "IMAS_network_differential_volcano_NDS",
        f"{NORMAL_SAMPLE_ID} vs {run_label}",
    )
    deg_plot = plot_deg_background_with_nds_red(
        deg,
        node_scores,
        figure_dir / "DEG_background_with_IMAS_NDS_red_nodes",
        f"{NORMAL_SAMPLE_ID} vs {run_label}",
    )

    summary = {
        "run_label": run_label,
        "n_candidates_raw": int(len(candidates_raw)),
        "n_candidates_prepared": int(len(candidates)),
        "n_candidates_expression_supported": int(len(candidates_expression)),
        "n_condition_specific_edges": int(len(edges)),
        "n_network_nodes": int(len(node_scores)),
        "n_red_nodes": int(len(red_nodes)),
        "position_calculation_mode": join_unique(
            node_scores["position_calculation_mode"].astype(str)
        ),
        "NDS_weights": {
            "W_EXPR": W_EXPR,
            "W_POS": W_POS,
            "W_FIELD": W_FIELD,
            "W_FIELD_IN": W_FIELD_IN,
            "W_FIELD_OUT": W_FIELD_OUT,
            "W_FIELD_EXPR": W_FIELD_EXPR,
        },
        "outputs": {
            "candidate_expression_csv": str(candidate_expression_csv),
            "edges_csv": str(edges_csv),
            "node_expression_csv": str(node_expression_csv),
            "node_scores_csv": str(node_scores_csv),
            "top_nodes_csv": str(top_nodes_csv),
            "red_nodes_csv": str(red_nodes_csv),
            "network_plot": network_plot,
            "deg_background_plot": deg_plot,
        },
    }
    summary_json = output_root / "IMAS_network_differential_summary_NDS_v1.json"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary, node_scores, edges


# =============================================================================
# 8. CROSS-SAMPLE NDS MATRICES AND CONSENSUS
# =============================================================================


def save_node_metric_matrix(
    long_scores: pd.DataFrame,
    metric: str,
    sample_labels: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    matrix = long_scores.pivot_table(
        index="node",
        columns="sample_id",
        values=metric,
        aggfunc="first",
    )
    matrix = matrix.reindex(columns=list(sample_labels))
    matrix = matrix.sort_index()
    matrix.reset_index().to_csv(output_path, index=False)
    return matrix


def build_cross_sample_node_consensus(
    long_scores: pd.DataFrame,
    sample_labels: Sequence[str],
) -> pd.DataFrame:
    if len(long_scores) == 0:
        return pd.DataFrame()

    def positive_fraction(values: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        return float(np.mean(values > 0)) if len(values) else np.nan

    def negative_fraction(values: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        return float(np.mean(values < 0)) if len(values) else np.nan

    consensus = long_scores.groupby("node", as_index=False).agg(
        n_samples_present=("sample_id", "nunique"),
        sample_ids=("sample_id", join_unique),
        mean_NDS=("NDS", "mean"),
        median_NDS=("NDS", "median"),
        max_NDS=("NDS", "max"),
        min_NDS=("NDS", "min"),
        sd_NDS=("NDS", "std"),
        mean_signed_NDS=("signed_NDS", "mean"),
        median_signed_NDS=("signed_NDS", "median"),
        positive_fraction=("signed_NDS", positive_fraction),
        negative_fraction=("signed_NDS", negative_fraction),
        mean_expr_diff01=("expr_diff01", "mean"),
        mean_position_diff01=("position_diff01", "mean"),
        mean_field_diff01=("field_diff01", "mean"),
        max_expr_diff01=("expr_diff01", "max"),
        max_position_diff01=("position_diff01", "max"),
        max_field_diff01=("field_diff01", "max"),
        network_roles=("network_roles", lambda x: join_unique(";".join(map(str, x)).split(";"))),
        axis_hints=("axis_hints", lambda x: join_unique(";".join(map(str, x)).split(";"))),
    )
    consensus["sd_NDS"] = consensus["sd_NDS"].fillna(0.0)
    consensus["sample_coverage_fraction"] = (
        consensus["n_samples_present"] / float(len(sample_labels))
    )
    consensus["direction_consistency"] = np.maximum(
        consensus["positive_fraction"].fillna(0.0),
        consensus["negative_fraction"].fillna(0.0),
    )
    consensus["consensus_NDS"] = (
        CONSENSUS_W_MEAN * consensus["mean_NDS"]
        + CONSENSUS_W_MAX * consensus["max_NDS"]
        + CONSENSUS_W_COVERAGE * consensus["sample_coverage_fraction"]
    )
    consensus_direction = np.sign(consensus["mean_signed_NDS"].to_numpy(dtype=float))
    consensus_direction = np.where(consensus_direction == 0, 1.0, consensus_direction)
    consensus["consensus_signed_NDS"] = (
        consensus_direction * consensus["consensus_NDS"]
    )
    consensus["empirical_p_consensus_NDS"] = empirical_p_from_score(
        consensus["consensus_NDS"].to_numpy(dtype=float)
    )
    consensus["empirical_q_consensus_NDS"] = bh_fdr(
        consensus["empirical_p_consensus_NDS"].to_numpy(dtype=float)
    )
    consensus["consensus_NDS_percentile"] = rank01(
        consensus["consensus_NDS"].to_numpy(dtype=float)
    )
    consensus = consensus.sort_values(
        ["consensus_NDS", "n_samples_present", "mean_NDS"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
    return consensus


def plot_consensus_volcano(
    consensus: pd.DataFrame,
    output_base: Path,
) -> Dict[str, object]:
    frame = consensus.copy()
    highlighted = frame[
        frame["consensus_NDS_percentile"] >= RED_NDS_PERCENTILE
    ].head(MAX_RED_NODES)
    highlighted_nodes = set(highlighted["node"].astype(str))
    frame["is_red"] = frame["node"].astype(str).isin(highlighted_nodes)
    y = -np.log10(
        np.clip(
            frame["empirical_p_consensus_NDS"].to_numpy(dtype=float),
            1e-300,
            1.0,
        )
    )

    figure, axis = plt.subplots(figsize=(9.2, 7.8))
    background = frame[~frame["is_red"]]
    red = frame[frame["is_red"]]
    axis.scatter(
        background["consensus_signed_NDS"],
        y[~frame["is_red"].to_numpy()],
        s=18,
        c="#BDBDBD",
        alpha=0.60,
        linewidths=0,
    )
    axis.scatter(
        red["consensus_signed_NDS"],
        -np.log10(
            np.clip(
                red["empirical_p_consensus_NDS"].to_numpy(dtype=float),
                1e-300,
                1.0,
            )
        ),
        s=42,
        c="#D62728",
        alpha=0.95,
        edgecolors="#7F0000",
        linewidths=0.25,
    )
    for _, record in red.head(LABEL_TOP_N).iterrows():
        axis.text(
            float(record["consensus_signed_NDS"]),
            -np.log10(max(float(record["empirical_p_consensus_NDS"]), 1e-300)),
            str(record["node"]),
            fontsize=8,
            color="#7F0000",
            ha="center",
            va="bottom",
        )
    axis.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Signed cross-sample consensus NDS", fontsize=12)
    axis.set_ylabel("-log10(empirical P from consensus rank)", fontsize=12)
    axis.set_title(
        "Cross-sample IMAS network-differential consensus\n"
        "Sample-1-6, Sample-8 and Sample-9",
        fontsize=14,
        pad=14,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    figure.tight_layout()

    png = output_base.with_suffix(".png")
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    figure.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf)}


# =============================================================================
# 9. MAIN
# =============================================================================


def main() -> None:
    ensure_directories(
        [DIR_QC, DIR_EDGE, DIR_NODE, DIR_FIG, DIR_PER_SAMPLE, DIR_CONSENSUS]
    )

    log("=" * 110)
    log("[START] IMAS NDS matrix construction for eight retained samples")
    log("[PREP ROOT]", PREP_ROOT)
    log("[OUT ROOT]", OUT_ROOT)
    log("=" * 110)

    for path in [
        STAGEB_MANIFEST,
        STAGEB_CHECKPOINT,
        POOLED_DEG_CSV,
        ALL_CANDIDATE_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest, sample_labels = load_completed_sample_labels()
    pooled_deg = validate_deg_table(
        pd.read_csv(POOLED_DEG_CSV, low_memory=False),
        "pooled",
    )
    all_candidates = validate_candidate_table(
        pd.read_csv(ALL_CANDIDATE_CSV, low_memory=False),
        sample_labels,
    )

    # Record exact sample/cell provenance.
    sample_counts = (
        manifest.groupby("sample_id").size().rename("n_cells").reset_index()
    )
    sample_counts.to_csv(DIR_QC / "retained_sample_cell_counts.tsv", sep="\t", index=False)
    all_candidates.groupby("tumor_sample_id").size().rename(
        "n_candidate_rows"
    ).reset_index().to_csv(
        DIR_QC / "candidate_rows_per_sample.tsv", sep="\t", index=False
    )

    # -------------------------------------------------------------------------
    # 1) Canonical pooled NDS network.
    # -------------------------------------------------------------------------
    log("\n[STEP 1] Canonical pooled eight-sample NDS")
    pooled_summary, pooled_node_scores, pooled_edges = run_one_nds_network(
        run_label=POOLED_TUMOR_ID,
        deg=pooled_deg,
        candidates_raw=all_candidates,
        output_root=OUT_ROOT,
    )

    # -------------------------------------------------------------------------
    # 2) Eight independent per-sample NDS networks.
    # -------------------------------------------------------------------------
    log("\n[STEP 2] Independent per-sample NDS networks")
    per_sample_node_parts: List[pd.DataFrame] = []
    sample_summary_rows: List[dict] = []
    failed_rows: List[dict] = []

    for sample_id in sample_labels:
        log("\n" + "-" * 100)
        log("[SAMPLE]", sample_id)
        sample_output_root = DIR_PER_SAMPLE / safe_name(sample_id)
        try:
            deg_path = per_sample_deg_path(sample_id)
            if not deg_path.exists():
                raise FileNotFoundError(deg_path)
            sample_deg = validate_deg_table(
                pd.read_csv(deg_path, low_memory=False),
                sample_id,
            )
            sample_candidates = all_candidates[
                all_candidates["tumor_sample_id"].astype(str) == str(sample_id)
            ].copy()
            if len(sample_candidates) == 0:
                raise RuntimeError(f"No candidate cascades found for {sample_id}")

            summary, node_scores, _ = run_one_nds_network(
                run_label=sample_id,
                deg=sample_deg,
                candidates_raw=sample_candidates,
                output_root=sample_output_root,
            )
            node_scores = node_scores.copy()
            node_scores.insert(1, "sample_id", sample_id)
            per_sample_node_parts.append(node_scores)
            sample_summary_rows.append(
                {
                    "sample_id": sample_id,
                    "n_cells": int(
                        sample_counts.loc[
                            sample_counts["sample_id"] == sample_id,
                            "n_cells",
                        ].iloc[0]
                    ),
                    "n_candidates_raw": summary["n_candidates_raw"],
                    "n_candidates_prepared": summary["n_candidates_prepared"],
                    "n_condition_specific_edges": summary[
                        "n_condition_specific_edges"
                    ],
                    "n_network_nodes": summary["n_network_nodes"],
                    "n_red_nodes": summary["n_red_nodes"],
                    "position_calculation_mode": summary[
                        "position_calculation_mode"
                    ],
                    "node_scores_csv": summary["outputs"]["node_scores_csv"],
                    "status": "success",
                }
            )
            log("[SAMPLE DONE]", sample_id, "nodes=", len(node_scores))
        except Exception as error:
            failed_rows.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            log("[SAMPLE FAILED]", sample_id, repr(error))

    sample_summary = pd.DataFrame(sample_summary_rows)
    failed = pd.DataFrame(failed_rows)
    sample_summary.to_csv(
        DIR_QC / "per_sample_NDS_run_summary.tsv", sep="\t", index=False
    )
    failed.to_csv(DIR_QC / "failed_samples.tsv", sep="\t", index=False)
    if failed_rows:
        raise RuntimeError(
            "One or more per-sample NDS runs failed. See: "
            f"{DIR_QC / 'failed_samples.tsv'}"
        )
    if len(per_sample_node_parts) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLE_COUNT} per-sample node-score tables; "
            f"obtained {len(per_sample_node_parts)}."
        )

    # -------------------------------------------------------------------------
    # 3) NDS matrices and node-level cross-sample consensus.
    # -------------------------------------------------------------------------
    log("\n[STEP 3] Cross-sample NDS matrices and consensus")
    long_scores = pd.concat(per_sample_node_parts, axis=0, ignore_index=True)
    long_scores.to_csv(
        DIR_CONSENSUS / "per_sample_node_scores_long.csv",
        index=False,
    )

    matrix_paths: Dict[str, str] = {}
    for metric in [
        "NDS",
        "signed_NDS",
        "NDS_percentile",
        "expr_diff01",
        "position_diff01",
        "field_diff01",
        "expr_log2FC_tumor_vs_normal",
        "field_log2_ratio_tumor_vs_normal",
        "empirical_p_NDS",
        "empirical_q_NDS",
    ]:
        output_path = DIR_CONSENSUS / f"{metric}_matrix_by_sample.csv"
        save_node_metric_matrix(long_scores, metric, sample_labels, output_path)
        matrix_paths[metric] = str(output_path)

    consensus = build_cross_sample_node_consensus(long_scores, sample_labels)
    consensus_csv = DIR_CONSENSUS / "cross_sample_node_consensus_NDS.csv"
    top_consensus_csv = DIR_CONSENSUS / "top_cross_sample_node_consensus_NDS.csv"
    consensus.to_csv(consensus_csv, index=False)
    consensus.head(300).to_csv(top_consensus_csv, index=False)
    consensus_plot = plot_consensus_volcano(
        consensus,
        DIR_CONSENSUS / "cross_sample_consensus_NDS_volcano",
    )

    # Optional comparison of pooled NDS and per-sample consensus.
    pooled_compare = pooled_node_scores[
        [
            "node",
            "NDS",
            "signed_NDS",
            "expr_diff01",
            "position_diff01",
            "field_diff01",
        ]
    ].rename(
        columns={
            "NDS": "pooled_NDS",
            "signed_NDS": "pooled_signed_NDS",
            "expr_diff01": "pooled_expr_diff01",
            "position_diff01": "pooled_position_diff01",
            "field_diff01": "pooled_field_diff01",
        }
    )
    pooled_consensus_comparison = consensus.merge(
        pooled_compare,
        on="node",
        how="outer",
    )
    pooled_consensus_comparison.to_csv(
        DIR_CONSENSUS / "pooled_NDS_vs_cross_sample_consensus.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 4) Final summary.
    # -------------------------------------------------------------------------
    final_summary = {
        "normal_sample_id": NORMAL_SAMPLE_ID,
        "pooled_tumor_id": POOLED_TUMOR_ID,
        "dataset_name": DATASET_NAME,
        "sample_labels": sample_labels,
        "excluded_sample_numbers": sorted(EXPECTED_EXCLUDED_SAMPLE_NUMBERS),
        "stageB_manifest": str(STAGEB_MANIFEST),
        "stageB_checkpoint": str(STAGEB_CHECKPOINT),
        "preparation_root": str(PREP_ROOT),
        "pooled_deg_csv": str(POOLED_DEG_CSV),
        "all_candidate_csv": str(ALL_CANDIDATE_CSV),
        "pooled_NDS_summary": pooled_summary,
        "n_per_sample_node_rows": int(len(long_scores)),
        "n_unique_nodes_across_samples": int(long_scores["node"].nunique()),
        "n_consensus_nodes": int(len(consensus)),
        "NDS_weights": {
            "W_EXPR": W_EXPR,
            "W_POS": W_POS,
            "W_FIELD": W_FIELD,
            "W_FIELD_IN": W_FIELD_IN,
            "W_FIELD_OUT": W_FIELD_OUT,
            "W_FIELD_EXPR": W_FIELD_EXPR,
        },
        "consensus_weights": {
            "mean_NDS": CONSENSUS_W_MEAN,
            "max_NDS": CONSENSUS_W_MAX,
            "sample_coverage": CONSENSUS_W_COVERAGE,
        },
        "matrix_paths": matrix_paths,
        "consensus_csv": str(consensus_csv),
        "top_consensus_csv": str(top_consensus_csv),
        "consensus_plot": consensus_plot,
        "per_sample_summary": str(
            DIR_QC / "per_sample_NDS_run_summary.tsv"
        ),
        "out_root": str(OUT_ROOT),
    }
    summary_json = OUT_ROOT / "IMAS_network_differential_summary_NDS_v1.json"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, ensure_ascii=False, indent=2)

    log("\n" + "=" * 110)
    log("[DONE]")
    log(json.dumps(final_summary, indent=2, ensure_ascii=False))
    log("\n[PRIMARY OUTPUTS]")
    log(" -", DIR_NODE / "IMAS_network_differential_node_scores_NDS.csv")
    log(" -", DIR_CONSENSUS / "NDS_matrix_by_sample.csv")
    log(" -", consensus_csv)
    log("=" * 110)

    show_columns = [
        "consensus_rank",
        "node",
        "consensus_NDS",
        "consensus_signed_NDS",
        "n_samples_present",
        "sample_coverage_fraction",
        "direction_consistency",
        "mean_expr_diff01",
        "mean_position_diff01",
        "mean_field_diff01",
        "axis_hints",
    ]
    show_columns = [column for column in show_columns if column in consensus.columns]
    log("\n[TOP CROSS-SAMPLE CONSENSUS NDS NODES]")
    log(consensus[show_columns].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
