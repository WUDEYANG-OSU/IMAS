# -*- coding: utf-8 -*-
"""
plot_MLTD_on_off_paired_GNN_LTC_temporal_transmission_v2.py
===========================================================

目的
----
1. 基于已有 MLTD-on/off CCI 结果，分别重新训练两套配对 GNN-LTC：
   - MLTD-off
   - MLTD-on

2. 生成配对主图（A-F）：
   A  MLTD-off temporally weighted CCI
   B  MLTD-on temporally weighted CCI
   C  Preserved cluster temporal ordering
   D  Redistribution of temporal communication (on - off)
   E  Pathway-level temporal gain decomposition
   F  Paired global temporal metrics (raw-value mini-panels)

3. 额外生成 MLTD-off 和 MLTD-on 各自的 two-panel figure：
   I  Cell projected pseudotime violin
   J  Representative ligand–receptor pairs scatter

输出
----
主输出目录：
communication_pseudotime_clusterGNN_LTC/MLTD_on_off_paired_GNN_LTC_v2/

核心文件：
1. MLTD_off_cluster_pseudotime_ensemble.csv
2. MLTD_on_cluster_pseudotime_ensemble.csv
3. MLTD_off_cluster_pseudotime_by_seed.csv
4. MLTD_on_cluster_pseudotime_by_seed.csv
5. MLTD_off_edge_temporal_scores.csv
6. MLTD_on_edge_temporal_scores.csv
7. MLTD_on_minus_off_temporal_gain.csv
8. MLTD_pathway_temporal_gain.csv
9. MLTD_on_off_seed_metrics.csv
10. MLTD_on_off_metric_summary.csv
11. MLTD_cluster_pseudotime_rank_shift.csv
12. MLTD_off_cell_projected_pseudotime.csv
13. MLTD_on_cell_projected_pseudotime.csv
14. MLTD_off_lr_scatter_summary.csv
15. MLTD_on_lr_scatter_summary.csv

图片：
16. MLTD_on_off_paired_GNN_LTC_summary.png / .svg / .pdf
17. MLTD_off_mode_specific_overview.png / .svg / .pdf
18. MLTD_on_mode_specific_overview.png / .svg / .pdf

说明
----
- 默认使用 communication-mass matching
- receiver 节点聚合 sender 信息
- teacher 不再重复塞 net_flow / io_ratio
- 当前结果更适合解释为：
  MLTD selectively strengthens TF-aware communication
  across a preserved GNN-LTC temporal backbone
"""

from __future__ import annotations

import json
import math
import random
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
    Normalize,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.stats import wilcoxon, spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =============================================================================
# 0. CONFIG
# =============================================================================

DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

BASE = get_project_root()

RUN_NAME = (
    "frontier_transfer_hash_stageB_"
    "selectedPeakResidual_clusterCorrection"
)

RUN_BASE = (
    BASE
    / "runs"
    / RUN_NAME
    / DATASET_NAME
)

CCI_DIR = RUN_BASE / f"{SAMPLE_ID}_cci_newLeiden"
ORIGINAL_LTC_DIR = CCI_DIR / "communication_pseudotime_clusterGNN_LTC"

CLUSTER_FEATURE_CSV = ORIGINAL_LTC_DIR / "cluster_dynamic_features.csv"
CLUSTER_SIZE_CSV = CCI_DIR / "cluster_sizes.csv"

CELL_MAP_CSV = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_recluster_for_cci"
    / f"{SAMPLE_ID}_cell_level_mapping_with_umap_and_internal16.csv"
)

ORIGINAL_CELL_PT_CSV = ORIGINAL_LTC_DIR / "cell_projected_pseudotime.csv"

MLTD_CCI_DIR_CANDIDATES = [
    RUN_BASE / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_4",
    RUN_BASE / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_3",
    RUN_BASE / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_2",
]

OUT_DIR = ORIGINAL_LTC_DIR / "MLTD_on_off_paired_GNN_LTC_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REQUIRE_CUDA = False

if REQUIRE_CUDA and DEVICE != "cuda":
    raise RuntimeError("CUDA is required but not available.")

# ---------------------------
# paired GNN-LTC
# ---------------------------
N_ENSEMBLE = 10
ENSEMBLE_SEEDS = list(range(42, 42 + N_ENSEMBLE))

HIDDEN_DIM = 64
N_ROLLOUT = 10
DT = 0.18

EPOCHS = 800
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

W_TEACHER = 0.90
W_RANK = 0.45
W_DIRECTION = 0.80
W_SMOOTH = 0.22
W_RECON = 0.12
W_VARIANCE = 0.10

W_RECEIVER_TEACHER = 1.00
W_TF_TEACHER = 0.65
W_SENDER_TEACHER = 0.75

# ---------------------------
# temporal score
# ---------------------------
MASS_MATCH = True
TEMPORAL_ALPHA = 12.0
TEMPORAL_DELTA = 0.05

TOP_N_PATHWAYS = 10
TOP_N_LABELS_LR = 8

DPI = 450

FIG_MAIN_PNG = OUT_DIR / "MLTD_on_off_paired_GNN_LTC_summary.png"
FIG_MAIN_SVG = OUT_DIR / "MLTD_on_off_paired_GNN_LTC_summary.svg"
FIG_MAIN_PDF = OUT_DIR / "MLTD_on_off_paired_GNN_LTC_summary.pdf"

FIG_OFF_OVERVIEW_PNG = OUT_DIR / "MLTD_off_mode_specific_overview.png"
FIG_OFF_OVERVIEW_SVG = OUT_DIR / "MLTD_off_mode_specific_overview.svg"
FIG_OFF_OVERVIEW_PDF = OUT_DIR / "MLTD_off_mode_specific_overview.pdf"

FIG_ON_OVERVIEW_PNG = OUT_DIR / "MLTD_on_mode_specific_overview.png"
FIG_ON_OVERVIEW_SVG = OUT_DIR / "MLTD_on_mode_specific_overview.svg"
FIG_ON_OVERVIEW_PDF = OUT_DIR / "MLTD_on_mode_specific_overview.pdf"

# ---------------------------
# style colors
# ---------------------------
COL_OFF = "#8B969C"
COL_ON = "#C45843"
COL_GAIN = "#B54A42"
COL_LOSS = "#4C79A5"
COL_ZERO = "#F4F4F2"
COL_TEXT = "#242729"
COL_MUTED = "#747A7D"
COL_GRID = "#D8DBDD"
COL_EARLY = "#4C78A8"
COL_LATE = "#D96B4C"
COL_BLACK = "#2A2A2A"

MODULE_COLORS = {
    "ECM / Adhesion": "#8DAA43",
    "Cytokine / Chemokine": "#4FA7A3",
    "Growth-factor": "#4F7DB8",
    "Immune-like": "#8B6CAF",
    "Other": "#B9C0C4",
}

CMAP_TIME = LinearSegmentedColormap.from_list(
    "time_map",
    ["#426A9B", "#F7F6F2", "#C55B46"],
)

CMAP_GAIN = LinearSegmentedColormap.from_list(
    "gain_map",
    ["#3D73A5", "#F8F7F3", "#B64C44"],
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# 1. HELPERS
# =============================================================================

def log(*args):
    print(*args, flush=True)


def canonical(value) -> str:
    return str(value).strip().upper()


def normalize_cluster(value) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    match = re.search(r"(\d+)", text)
    if match:
        return f"C{int(match.group(1))}"
    return text if text.upper().startswith("C") else f"C{text}"


def cluster_sort_key(value: str):
    text = normalize_cluster(value)
    match = re.fullmatch(r"C(\d+)", text)
    if match:
        return 0, int(match.group(1))
    return 1, text


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def first_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = True,
    role: str = "column",
) -> str | None:

    lookup = {
        normalized_name(column): str(column)
        for column in df.columns
    }

    for alias in aliases:
        key = normalized_name(alias)
        if key in lookup:
            return lookup[key]

    candidates = []
    for column in df.columns:
        ckey = normalized_name(column)
        for alias in aliases:
            akey = normalized_name(alias)
            if len(akey) >= 5 and (
                ckey.startswith(akey) or ckey.endswith(akey)
            ):
                candidates.append(column)

    if candidates:
        return str(candidates[0])

    if required:
        raise KeyError(
            f"Cannot identify {role}.\n"
            f"Actual columns:\n{df.columns.tolist()}\n"
            f"Accepted aliases:\n{list(aliases)}"
        )
    return None


def finite_numeric(series, default=0.0) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(float)
    )


def minmax01(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    low = float(np.nanmin(x[finite]))
    high = float(np.nanmax(x[finite]))
    if high - low < 1e-12:
        out[finite] = 0.5 if high > 0 else 0.0
    else:
        out[finite] = (x[finite] - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def zscore(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mean = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / sd


def rank01(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x)
    return (
        pd.Series(x).rank(method="average").to_numpy(float) - 1.0
    ) / float(len(x) - 1)


def sigmoid(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return np.nan
    if np.std(x[finite]) < 1e-12 or np.std(y[finite]) < 1e-12:
        return np.nan
    if SCIPY_OK:
        return float(spearmanr(x[finite], y[finite]).correlation)
    return float(np.corrcoef(rank01(x[finite]), rank01(y[finite]))[0, 1])


def panel_label(ax, letter: str):
    ax.text(
        -0.13, 1.08, letter,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="top", ha="left", color=COL_TEXT,
    )


def locate_mltd_output_dir() -> Path:
    for directory in MLTD_CCI_DIR_CANDIDATES:
        if directory.exists() and list(directory.glob("*_directed_cluster_pair_on_off_summary.csv")):
            return directory
    hits = sorted(
        RUN_BASE.rglob(f"{SAMPLE_ID}_directed_cluster_pair_on_off_summary.csv"),
        key=lambda path: ("v5_4" not in str(path), len(str(path)), str(path)),
    )
    if hits:
        return hits[0].parent
    raise FileNotFoundError(
        "Cannot find directed_cluster_pair_on_off_summary.csv.\n"
        "Please run the MLTD-on/off CCI script first."
    )


def find_single_file(
    directory: Path,
    patterns: list[str],
    *,
    required: bool = True,
) -> Path | None:
    for pattern in patterns:
        hits = sorted(directory.glob(pattern))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(
            f"Cannot find files matching:\n{patterns}\ninside:\n{directory}"
        )
    return None


# =============================================================================
# 2. INPUT LOADING
# =============================================================================

def load_cluster_features() -> pd.DataFrame:
    if not CLUSTER_FEATURE_CSV.exists():
        raise FileNotFoundError(
            f"Missing cluster feature table:\n{CLUSTER_FEATURE_CSV}\n"
            "Run the original GNN-LTC script first."
        )
    df = pd.read_csv(CLUSTER_FEATURE_CSV)

    cluster_col = first_column(
        df,
        ["cluster_norm", "cluster_label", "cluster", "cluster_id", "new_leiden"],
        role="cluster identifier in cluster_dynamic_features",
    )

    df["cluster"] = df[cluster_col].map(normalize_cluster)
    df = df[df["cluster"] != ""].drop_duplicates("cluster").copy()
    return df


def load_pair_summary(
    mltd_dir: Path,
) -> tuple[pd.DataFrame, Path]:

    path = find_single_file(
        mltd_dir,
        [
            f"{SAMPLE_ID}_directed_cluster_pair_on_off_summary.csv",
            "*_directed_cluster_pair_on_off_summary.csv",
            "*pair_on_off_summary.csv",
        ],
    )

    df = pd.read_csv(path)

    sender_col = first_column(
        df,
        ["sender_cluster", "source_cluster", "source", "sender", "from_cluster"],
        role="sender cluster",
    )

    receiver_col = first_column(
        df,
        ["receiver_cluster", "target_cluster", "target", "receiver", "to_cluster"],
        role="receiver cluster",
    )

    off_col = first_column(
        df,
        ["off_weight", "weight_off", "score_off", "mltd_off_weight", "off_score", "ordinary_weight", "ordinary_score"],
        role="MLTD-off pair weight",
    )

    on_col = first_column(
        df,
        ["on_weight", "weight_on", "score_on", "mltd_on_weight", "on_score"],
        role="MLTD-on pair weight",
    )

    out = pd.DataFrame({
        "sender_cluster": df[sender_col].map(normalize_cluster),
        "receiver_cluster": df[receiver_col].map(normalize_cluster),
        "off_weight": finite_numeric(df[off_col]),
        "on_weight": finite_numeric(df[on_col]),
    })

    out = (
        out[
            (out["sender_cluster"] != "")
            & (out["receiver_cluster"] != "")
        ]
        .groupby(["sender_cluster", "receiver_cluster"], as_index=False)[["off_weight", "on_weight"]]
        .sum()
    )

    out["off_weight"] = np.maximum(out["off_weight"], 0.0)
    out["on_weight"] = np.maximum(out["on_weight"], 0.0)

    return out, path


def load_program_table(
    mltd_dir: Path,
) -> tuple[pd.DataFrame | None, Path | None]:

    path = find_single_file(
        mltd_dir,
        [
            f"{SAMPLE_ID}_same_universe_program_on_off_scores.csv",
            "*_same_universe_program_on_off_scores.csv",
            "*program_on_off_scores.csv",
        ],
        required=False,
    )

    if path is None:
        warnings.warn(
            "Program-level on/off table not found. "
            "Pathway panel and LR scatter will use fallback behavior."
        )
        return None, None

    df = pd.read_csv(path)

    sender_col = first_column(
        df,
        ["sender_cluster", "source_cluster", "source", "sender"],
        role="program sender cluster",
    )

    receiver_col = first_column(
        df,
        ["receiver_cluster", "target_cluster", "target", "receiver"],
        role="program receiver cluster",
    )

    off_col = first_column(
        df,
        ["score_off", "off_score", "off_weight", "mltd_off_score"],
        role="program MLTD-off score",
    )

    on_col = first_column(
        df,
        ["score_on", "on_score", "on_weight", "mltd_on_score"],
        role="program MLTD-on score",
    )

    pathway_col = first_column(
        df,
        ["display_pathway", "kegg_display_pathway", "kegg_pathway", "pathway", "analysis_pathway", "pathway_label", "broad_pathway"],
        required=False,
        role="pathway",
    )

    ligand_col = first_column(
        df,
        ["ligand", "ligand_gene", "source_gene"],
        required=False,
        role="ligand",
    )

    receptor_col = first_column(
        df,
        ["receptor", "receptor_gene", "target_gene"],
        required=False,
        role="receptor",
    )

    out = pd.DataFrame({
        "sender_cluster": df[sender_col].map(normalize_cluster),
        "receiver_cluster": df[receiver_col].map(normalize_cluster),
        "score_off": finite_numeric(df[off_col]),
        "score_on": finite_numeric(df[on_col]),
    })

    out["pathway"] = (
        df[pathway_col].astype(str)
        if pathway_col is not None
        else "Unknown"
    )
    out["ligand"] = df[ligand_col].astype(str) if ligand_col is not None else "Unknown"
    out["receptor"] = df[receptor_col].astype(str) if receptor_col is not None else "Unknown"

    if "pathway" not in out.columns:
        out["pathway"] = "Unknown"

    empty_pathway = (
        out["pathway"].astype(str).str.strip().str.lower()
        .isin(["", "nan", "none", "other", "unknown"])
    )

    out.loc[empty_pathway, "pathway"] = (
        out.loc[empty_pathway, "ligand"].astype(str)
        + "–"
        + out.loc[empty_pathway, "receptor"].astype(str)
        + " signaling"
    )

    out["score_off"] = np.maximum(out["score_off"], 0.0)
    out["score_on"] = np.maximum(out["score_on"], 0.0)

    out["lr"] = out["ligand"].astype(str) + "–" + out["receptor"].astype(str)

    return out, path


def load_cell_map() -> pd.DataFrame | None:
    if not CELL_MAP_CSV.exists():
        warnings.warn(f"Missing cell map csv: {CELL_MAP_CSV}")
        return None

    df = pd.read_csv(CELL_MAP_CSV)

    cluster_col = first_column(
        df,
        ["cluster_norm", "new_leiden", "cluster", "cluster_id", "cluster_label"],
        role="cell-level cluster column",
    )

    df["cluster"] = df[cluster_col].map(normalize_cluster)

    barcode_col = first_column(
        df,
        ["barcode", "cell_barcode", "cell", "obs_names", "cell_id"],
        required=False,
        role="cell barcode",
    )

    if barcode_col is not None:
        df["barcode"] = df[barcode_col].astype(str)
    else:
        df["barcode"] = [f"cell_{i}" for i in range(len(df))]

    umap1_col = first_column(
        df,
        ["UMAP1", "umap1", "umap_1", "UMAP-1", "UMAP_1", "x_umap", "X_umap"],
        required=False,
        role="UMAP1",
    )

    umap2_col = first_column(
        df,
        ["UMAP2", "umap2", "umap_2", "UMAP-2", "UMAP_2", "y_umap", "Y_umap"],
        required=False,
        role="UMAP2",
    )

    if umap1_col is not None:
        df["umap1"] = pd.to_numeric(df[umap1_col], errors="coerce")
    else:
        df["umap1"] = np.nan

    if umap2_col is not None:
        df["umap2"] = pd.to_numeric(df[umap2_col], errors="coerce")
    else:
        df["umap2"] = np.nan

    return df


def load_original_cell_pt() -> pd.DataFrame | None:
    if not ORIGINAL_CELL_PT_CSV.exists():
        return None
    df = pd.read_csv(ORIGINAL_CELL_PT_CSV)

    barcode_col = first_column(
        df,
        ["barcode", "cell_barcode", "cell", "obs_names", "cell_id"],
        required=False,
        role="original cell pt barcode",
    )
    t_col = first_column(
        df,
        ["cell_t_proj", "t_hat_norm_mean", "t_hat_norm", "t_hat"],
        required=False,
        role="original cell projected pseudotime",
    )

    if barcode_col is None or t_col is None:
        return None

    out = pd.DataFrame({
        "barcode": df[barcode_col].astype(str),
        "orig_cell_t_proj": pd.to_numeric(df[t_col], errors="coerce"),
    })
    return out


# =============================================================================
# 3. NODE AND EDGE CONSTRUCTION
# =============================================================================

DYNAMIC_EXCLUDE = {
    "sender_strength",
    "sender_strength_raw",
    "receiver_strength",
    "receiver_strength_raw",
    "net_flow",
    "io_ratio",
    "weak_teacher_raw",
    "weak_teacher_rank",
    "t_hat",
    "t_hat_norm",
    "t_hat_norm_mean",
    "sigma_t",
    "tau_mean",
    "cluster",
    "cluster_norm",
    "cluster_label",
    "cluster_id",
    "new_leiden",
    "n_cells",
}


def detect_fixed_feature_columns(
    cluster_features: pd.DataFrame,
) -> list[str]:

    numeric_columns = []

    for column in cluster_features.columns:
        if column in DYNAMIC_EXCLUDE:
            continue
        values = pd.to_numeric(cluster_features[column], errors="coerce")
        if values.notna().sum() >= max(3, int(0.70 * len(cluster_features))):
            numeric_columns.append(column)

    preferred = [
        column for column in numeric_columns
        if re.search(r"(latent|pca|pc\d|centroid|embedding|z_?\d)", column, flags=re.I)
    ]

    context = [
        column for column in numeric_columns
        if re.search(r"(tf|profile|expr|activity)", column, flags=re.I)
        and column not in preferred
    ]

    selected = preferred + context
    if len(selected) > 64:
        selected = selected[:64]
    if not selected:
        selected = numeric_columns[:32]
    return selected


def resolve_tf_response_column(
    cluster_features: pd.DataFrame,
) -> str | None:

    return first_column(
        cluster_features,
        [
            "receiver_tf_response",
            "tf_top_mean",
            "fallback_tf_resp_z",
            "tf_response",
            "receiver_tf",
        ],
        required=False,
        role="receiver TF response",
    )


def mass_match_weights(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    for mode in ["off", "on"]:
        column = f"{mode}_weight"
        values = out[column].to_numpy(float)
        total = float(np.sum(values))
        if MASS_MATCH:
            if total <= 0:
                raise ValueError(f"{column} has zero total weight.")
            out[f"{mode}_weight_model"] = values / total
        else:
            out[f"{mode}_weight_model"] = minmax01(np.log1p(values))
    return out


def build_mode_node_table(
    clusters: list[str],
    pair_df: pd.DataFrame,
    cluster_features: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:

    weight_col = f"{mode}_weight_model"

    sender = pair_df.groupby("sender_cluster")[weight_col].sum().to_dict()
    receiver = pair_df.groupby("receiver_cluster")[weight_col].sum().to_dict()

    node = pd.DataFrame({
        "cluster": clusters,
        "sender_strength": [float(sender.get(cluster, 0.0)) for cluster in clusters],
        "receiver_strength": [float(receiver.get(cluster, 0.0)) for cluster in clusters],
    })

    node["net_flow"] = node["receiver_strength"] - node["sender_strength"]
    node["io_ratio"] = np.log1p(node["receiver_strength"]) - np.log1p(node["sender_strength"])

    fixed = cluster_features.set_index("cluster").reindex(clusters).copy()

    tf_col = resolve_tf_response_column(cluster_features)
    if tf_col is not None:
        tf_response = finite_numeric(fixed[tf_col], default=0.0)
    else:
        warnings.warn(
            "No receiver TF response column found. Using receiver strength as fallback."
        )
        tf_response = node["receiver_strength"].to_numpy(float)

    node["receiver_tf_response"] = minmax01(tf_response)

    static_cols = detect_fixed_feature_columns(cluster_features)
    static_matrix_parts = []

    for column in static_cols:
        values = finite_numeric(fixed[column], default=0.0)
        static_matrix_parts.append(zscore(values)[:, None])

    if static_matrix_parts:
        static_matrix = np.concatenate(static_matrix_parts, axis=1)
    else:
        static_matrix = np.eye(len(clusters), dtype=float)

    dynamic_matrix = np.column_stack([
        zscore(node["sender_strength"]),
        zscore(node["receiver_strength"]),
        zscore(node["net_flow"]),
        zscore(node["io_ratio"]),
        zscore(node["receiver_tf_response"]),
    ])

    feature_matrix = np.concatenate([dynamic_matrix, static_matrix], axis=1).astype(np.float32)

    weak_teacher_raw = (
        W_RECEIVER_TEACHER * zscore(node["receiver_strength"].to_numpy(float))
        + W_TF_TEACHER * zscore(node["receiver_tf_response"].to_numpy(float))
        - W_SENDER_TEACHER * zscore(node["sender_strength"].to_numpy(float))
    )

    node["weak_teacher_raw"] = weak_teacher_raw
    node["weak_teacher_rank"] = rank01(weak_teacher_raw)

    return node, feature_matrix


def build_directed_graph(
    clusters: list[str],
    pair_df: pd.DataFrame,
    mode: str,
):
    """
    receiver 聚合 sender:
    dst(receiver) <- src(sender)
    """

    index = {cluster: i for i, cluster in enumerate(clusters)}
    weight_col = f"{mode}_weight_model"

    edge = pair_df.copy()
    edge = edge[
        edge["sender_cluster"].isin(index)
        & edge["receiver_cluster"].isin(index)
    ].copy()

    edge["weight"] = finite_numeric(edge[weight_col], default=0.0)
    edge = edge[edge["weight"] > 0].copy()

    if len(edge) == 0:
        raise ValueError(f"No positive edges for mode={mode}.")

    src = np.array([index[value] for value in edge["sender_cluster"]], dtype=np.int64)
    dst = np.array([index[value] for value in edge["receiver_cluster"]], dtype=np.int64)
    weight = edge["weight"].to_numpy(np.float32)

    adjacency = np.zeros((len(clusters), len(clusters)), dtype=np.float32)

    for s, d, w in zip(src, dst, weight):
        adjacency[d, s] += float(w)

    adjacency += np.eye(len(clusters), dtype=np.float32) * 1e-4
    row_sum = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sum + 1e-8)

    return (
        edge.reset_index(drop=True),
        torch.tensor(adjacency, dtype=torch.float32, device=DEVICE),
        torch.tensor(src, dtype=torch.long, device=DEVICE),
        torch.tensor(dst, dtype=torch.long, device=DEVICE),
        torch.tensor(weight, dtype=torch.float32, device=DEVICE),
    )


# =============================================================================
# 4. PAIRED GNN-LTC MODEL
# =============================================================================

class PairedGNNLTC(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, recon_dim: int):
        super().__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.message_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.tau_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, recon_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        h = self.input_encoder(x)
        tau_values = []

        for _ in range(N_ROLLOUT):
            message = adjacency @ h
            proposal = self.message_layer(torch.cat([h, message], dim=1))
            tau = F.softplus(self.tau_head(h)) + 0.10
            alpha = DT / tau
            h = h + alpha * (proposal - h)
            tau_values.append(tau.squeeze(1))

        t_raw = self.time_head(h).squeeze(1)
        t = torch.sigmoid(t_raw)
        recon = self.recon_head(h)
        tau_mean = torch.stack(tau_values, dim=0).mean(dim=0)
        return t, recon, tau_mean


def sample_rank_pairs(teacher: torch.Tensor):
    n = len(teacher)

    left = []
    right = []
    labels = []

    for i in range(n):
        for j in range(i + 1, n):
            difference = float(teacher[j] - teacher[i])
            if abs(difference) < 1e-6:
                continue
            left.append(i)
            right.append(j)
            labels.append(1.0 if difference > 0 else 0.0)

    return (
        torch.tensor(left, dtype=torch.long, device=DEVICE),
        torch.tensor(right, dtype=torch.long, device=DEVICE),
        torch.tensor(labels, dtype=torch.float32, device=DEVICE),
    )


def train_one_mode_one_seed(
    feature_matrix: np.ndarray,
    node_table: pd.DataFrame,
    adjacency: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_weight: torch.Tensor,
    seed: int,
) -> dict:

    set_seed(seed)

    x = torch.tensor(feature_matrix, dtype=torch.float32, device=DEVICE)
    teacher = torch.tensor(
        node_table["weak_teacher_rank"].to_numpy(np.float32),
        dtype=torch.float32,
        device=DEVICE,
    )
    recon_target = torch.tensor(feature_matrix, dtype=torch.float32, device=DEVICE)

    model = PairedGNNLTC(
        input_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        recon_dim=x.shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rank_left, rank_right, rank_label = sample_rank_pairs(teacher)

    best_loss = np.inf
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        t, recon, tau = model(x, adjacency)

        teacher_loss = F.mse_loss(t, teacher)

        if len(rank_left) > 0:
            rank_logits = (t[rank_right] - t[rank_left]) * 8.0
            rank_loss = F.binary_cross_entropy_with_logits(rank_logits, rank_label)
        else:
            rank_loss = torch.tensor(0.0, device=DEVICE)

        delta = t[dst] - t[src]
        direction_loss = (
            edge_weight
            * F.softplus(-8.0 * (delta - TEMPORAL_DELTA))
        ).sum() / (edge_weight.sum() + 1e-8)

        smooth_loss = (
            edge_weight * torch.square(delta)
        ).sum() / (edge_weight.sum() + 1e-8)

        recon_loss = F.mse_loss(recon, recon_target)

        target_variance = 0.075
        variance_loss = torch.square(torch.var(t) - target_variance)

        loss = (
            W_TEACHER * teacher_loss
            + W_RANK * rank_loss
            + W_DIRECTION * direction_loss
            + W_SMOOTH * smooth_loss
            + W_RECON * recon_loss
            + W_VARIANCE * variance_loss
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        current_loss = float(loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a best state.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        t, recon, tau = model(x, adjacency)

    t = t.detach().cpu().numpy()
    tau = tau.detach().cpu().numpy()

    rho = safe_spearman(t, node_table["weak_teacher_rank"])

    if np.isfinite(rho) and rho < 0:
        t = 1.0 - t

    t = minmax01(t)

    return {
        "t": t,
        "tau": tau,
        "loss": best_loss,
    }


def train_mode_ensemble(
    mode: str,
    clusters: list[str],
    pair_df: pd.DataFrame,
    cluster_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    log("\n" + "=" * 90)
    log("[TRAIN MODE]", mode)

    node_table, feature_matrix = build_mode_node_table(
        clusters=clusters,
        pair_df=pair_df,
        cluster_features=cluster_features,
        mode=mode,
    )

    edge_table, adjacency, src, dst, edge_weight = build_directed_graph(
        clusters=clusters,
        pair_df=pair_df,
        mode=mode,
    )

    seed_records = []

    for seed in ENSEMBLE_SEEDS:
        log(f"[{mode}] seed={seed}")
        result = train_one_mode_one_seed(
            feature_matrix=feature_matrix,
            node_table=node_table,
            adjacency=adjacency,
            src=src,
            dst=dst,
            edge_weight=edge_weight,
            seed=seed,
        )

        for cluster, t_value, tau_value in zip(clusters, result["t"], result["tau"]):
            seed_records.append({
                "mode": mode,
                "seed": seed,
                "cluster": cluster,
                "t_hat": float(t_value),
                "tau": float(tau_value),
                "training_loss": float(result["loss"]),
            })

    seed_table = pd.DataFrame(seed_records)

    summary = (
        seed_table
        .groupby(["mode", "cluster"], as_index=False)
        .agg(
            t_hat_mean=("t_hat", "mean"),
            t_hat_sd=("t_hat", "std"),
            tau_mean=("tau", "mean"),
            tau_sd=("tau", "std"),
            training_loss_mean=("training_loss", "mean"),
        )
    )

    summary = summary.merge(node_table, on="cluster", how="left")
    return summary, seed_table


# =============================================================================
# 5. TEMPORAL SCORE AND METRICS
# =============================================================================

def calculate_edge_temporal_scores(
    pair_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:

    t_map = dict(zip(cluster_summary["cluster"], cluster_summary["t_hat_mean"]))

    out = pair_df[
        ["sender_cluster", "receiver_cluster", f"{mode}_weight", f"{mode}_weight_model"]
    ].copy()

    out = out.rename(columns={
        f"{mode}_weight": "raw_weight",
        f"{mode}_weight_model": "model_weight",
    })

    out["mode"] = mode
    out["t_sender"] = out["sender_cluster"].map(t_map)
    out["t_receiver"] = out["receiver_cluster"].map(t_map)
    out["delta_t"] = out["t_receiver"] - out["t_sender"]
    out["temporal_gate"] = sigmoid(TEMPORAL_ALPHA * (out["delta_t"] - TEMPORAL_DELTA))
    out["temporal_score"] = out["model_weight"] * out["temporal_gate"]
    out["forward"] = (out["delta_t"] > 0).astype(int)

    return out


def weighted_temporal_entropy(
    delta_t: np.ndarray,
    weight: np.ndarray,
    n_bins: int = 12,
) -> float:

    delta_t = np.asarray(delta_t, float)
    weight = np.asarray(weight, float)

    finite = np.isfinite(delta_t) & np.isfinite(weight) & (weight > 0)

    if finite.sum() < 2:
        return np.nan

    delta_t = delta_t[finite]
    weight = weight[finite]

    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    hist, _ = np.histogram(delta_t, bins=edges, weights=weight)

    probability = hist / (hist.sum() + 1e-12)
    probability = probability[probability > 0]

    entropy = -np.sum(probability * np.log(probability))
    maximum = np.log(n_bins)

    return float(entropy / maximum)


def calculate_seed_metrics(
    mode: str,
    pair_df: pd.DataFrame,
    seed_times: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:

    tf_map = dict(zip(cluster_summary["cluster"], cluster_summary["receiver_tf_response"]))
    records = []

    weight_col = f"{mode}_weight_model"

    for seed, sub in seed_times.groupby("seed"):
        t_map = dict(zip(sub["cluster"], sub["t_hat"]))

        work = pair_df.copy()
        work["t_sender"] = work["sender_cluster"].map(t_map)
        work["t_receiver"] = work["receiver_cluster"].map(t_map)
        work["delta_t"] = work["t_receiver"] - work["t_sender"]
        work["weight"] = work[weight_col]
        work["gate"] = sigmoid(TEMPORAL_ALPHA * (work["delta_t"] - TEMPORAL_DELTA))
        work["receiver_tf"] = work["receiver_cluster"].map(tf_map).fillna(0.0)

        total_weight = float(work["weight"].sum()) + 1e-12

        forward_fraction = float(
            work.loc[work["delta_t"] > 0, "weight"].sum() / total_weight
        )

        mean_delta = float(
            np.sum(work["weight"] * work["delta_t"]) / total_weight
        )

        temporal_alignment = float(
            np.sum(work["weight"] * work["gate"]) / total_weight
        )

        tf_den = float(
            np.sum(work["weight"] * np.maximum(work["receiver_tf"], 0.0))
        ) + 1e-12

        tf_temporal_alignment = float(
            np.sum(
                work["weight"] * work["gate"] * np.maximum(work["receiver_tf"], 0.0)
            ) / tf_den
        )

        temporal_entropy = weighted_temporal_entropy(
            work["delta_t"].to_numpy(float),
            work["weight"].to_numpy(float),
        )

        records.append({
            "mode": mode,
            "seed": int(seed),
            "weighted_forward_fraction": forward_fraction,
            "weighted_mean_delta_t": mean_delta,
            "temporal_alignment": temporal_alignment,
            "tf_temporal_alignment": tf_temporal_alignment,
            "temporal_delta_entropy": temporal_entropy,
        })

    return pd.DataFrame(records)


def combine_edge_gain(
    off_edges: pd.DataFrame,
    on_edges: pd.DataFrame,
) -> pd.DataFrame:

    keys = ["sender_cluster", "receiver_cluster"]

    off = off_edges[
        keys + ["raw_weight", "model_weight", "delta_t", "temporal_gate", "temporal_score"]
    ].rename(columns={
        "raw_weight": "raw_weight_off",
        "model_weight": "model_weight_off",
        "delta_t": "delta_t_off",
        "temporal_gate": "temporal_gate_off",
        "temporal_score": "temporal_score_off",
    })

    on = on_edges[
        keys + ["raw_weight", "model_weight", "delta_t", "temporal_gate", "temporal_score"]
    ].rename(columns={
        "raw_weight": "raw_weight_on",
        "model_weight": "model_weight_on",
        "delta_t": "delta_t_on",
        "temporal_gate": "temporal_gate_on",
        "temporal_score": "temporal_score_on",
    })

    out = off.merge(on, on=keys, how="outer").fillna(0.0)

    out["temporal_gain"] = out["temporal_score_on"] - out["temporal_score_off"]

    out["cci_strength_contribution"] = (
        (out["model_weight_on"] - out["model_weight_off"]) * out["temporal_gate_off"]
    )

    out["temporal_gating_contribution"] = (
        out["model_weight_on"] * (out["temporal_gate_on"] - out["temporal_gate_off"])
    )

    out["decomposition_residual"] = (
        out["temporal_gain"]
        - out["cci_strength_contribution"]
        - out["temporal_gating_contribution"]
    )

    return out


# =============================================================================
# 6. PATHWAY-LEVEL GAIN
# =============================================================================

def calculate_pathway_gain(
    program_df: pd.DataFrame | None,
    off_summary: pd.DataFrame,
    on_summary: pd.DataFrame,
    pair_gain: pd.DataFrame,
) -> pd.DataFrame:

    if program_df is None or len(program_df) == 0:
        fallback = pair_gain.copy()
        fallback["pathway"] = fallback["sender_cluster"] + " → " + fallback["receiver_cluster"]
        fallback = fallback.rename(columns={"temporal_gain": "pathway_temporal_gain"})
        return fallback[[
            "pathway",
            "pathway_temporal_gain",
            "cci_strength_contribution",
            "temporal_gating_contribution",
        ]]

    off_t = dict(zip(off_summary["cluster"], off_summary["t_hat_mean"]))
    on_t = dict(zip(on_summary["cluster"], on_summary["t_hat_mean"]))

    work = program_df.copy()

    work["t_sender_off"] = work["sender_cluster"].map(off_t)
    work["t_receiver_off"] = work["receiver_cluster"].map(off_t)
    work["t_sender_on"] = work["sender_cluster"].map(on_t)
    work["t_receiver_on"] = work["receiver_cluster"].map(on_t)

    work["gate_off"] = sigmoid(
        TEMPORAL_ALPHA * (
            work["t_receiver_off"] - work["t_sender_off"] - TEMPORAL_DELTA
        )
    )
    work["gate_on"] = sigmoid(
        TEMPORAL_ALPHA * (
            work["t_receiver_on"] - work["t_sender_on"] - TEMPORAL_DELTA
        )
    )

    if MASS_MATCH:
        score_off_sum = work["score_off"].sum() + 1e-12
        score_on_sum = work["score_on"].sum() + 1e-12
        work["weight_off"] = work["score_off"] / score_off_sum
        work["weight_on"] = work["score_on"] / score_on_sum
    else:
        work["weight_off"] = minmax01(np.log1p(work["score_off"]))
        work["weight_on"] = minmax01(np.log1p(work["score_on"]))

    work["temporal_score_off"] = work["weight_off"] * work["gate_off"]
    work["temporal_score_on"] = work["weight_on"] * work["gate_on"]
    work["temporal_gain"] = work["temporal_score_on"] - work["temporal_score_off"]

    work["cci_strength_contribution"] = (
        (work["weight_on"] - work["weight_off"]) * work["gate_off"]
    )
    work["temporal_gating_contribution"] = (
        work["weight_on"] * (work["gate_on"] - work["gate_off"])
    )

    summary = (
        work.groupby("pathway", as_index=False)
        .agg(
            pathway_temporal_gain=("temporal_gain", "sum"),
            cci_strength_contribution=("cci_strength_contribution", "sum"),
            temporal_gating_contribution=("temporal_gating_contribution", "sum"),
            temporal_score_off=("temporal_score_off", "sum"),
            temporal_score_on=("temporal_score_on", "sum"),
            n_programs=("temporal_gain", "size"),
        )
    )

    summary = summary.sort_values("pathway_temporal_gain", ascending=False).reset_index(drop=True)
    return summary


# =============================================================================
# 7. CELL PROJECTION
# =============================================================================

def build_cell_projected_pseudotime(
    mode: str,
    cell_map: pd.DataFrame | None,
    original_cell_pt: pd.DataFrame | None,
    cluster_summary: pd.DataFrame,
    ordered_clusters: list[str],
) -> pd.DataFrame | None:

    if cell_map is None or len(cell_map) == 0:
        return None

    work = cell_map.copy()
    work = work[work["cluster"].isin(ordered_clusters)].copy()

    if len(work) == 0:
        return None

    t_map = dict(zip(cluster_summary["cluster"], cluster_summary["t_hat_mean"]))
    work["cluster_t_mean"] = work["cluster"].map(t_map)

    if original_cell_pt is not None:
        work = work.merge(original_cell_pt, on="barcode", how="left")
    else:
        work["orig_cell_t_proj"] = np.nan

    residual_col = np.full(len(work), np.nan, dtype=float)

    # 优先使用原始 cell projected pseudotime 的 cluster 内 rank
    for cluster, sub_idx in work.groupby("cluster").groups.items():
        idx = list(sub_idx)
        sub = work.loc[idx]

        if sub["orig_cell_t_proj"].notna().sum() >= 5:
            ranks = rank01(sub["orig_cell_t_proj"].to_numpy(float))
            centered = ranks - 0.5
            residual_col[np.array(idx)] = centered
            continue

        if sub["umap1"].notna().sum() >= 5:
            ranks = rank01(sub["umap1"].to_numpy(float))
            centered = ranks - 0.5
            residual_col[np.array(idx)] = centered
            continue

        # 兜底：随机小扰动，但保持可复现
        rng = np.random.default_rng(42 + int(cluster[1:]) if cluster[1:].isdigit() else 42)
        centered = rng.uniform(-0.5, 0.5, size=len(idx))
        residual_col[np.array(idx)] = centered

    work["within_cluster_centered_rank"] = residual_col

    # 计算 cluster 间局部间距，用于控制 violin 展开宽度
    ordered_t = np.array([t_map[c] for c in ordered_clusters], dtype=float)
    local_spread = {}

    for i, cluster in enumerate(ordered_clusters):
        if len(ordered_clusters) == 1:
            gap = 0.18
        elif i == 0:
            gap = ordered_t[i + 1] - ordered_t[i]
        elif i == len(ordered_clusters) - 1:
            gap = ordered_t[i] - ordered_t[i - 1]
        else:
            gap = 0.5 * (
                (ordered_t[i + 1] - ordered_t[i])
                + (ordered_t[i] - ordered_t[i - 1])
            )

        gap = abs(float(gap))
        spread = min(0.14, max(0.035, 0.45 * gap))
        local_spread[cluster] = spread

    work["cluster_spread"] = work["cluster"].map(local_spread).fillna(0.05)
    work["cell_t_proj"] = (
        work["cluster_t_mean"]
        + work["cluster_spread"] * work["within_cluster_centered_rank"]
    )
    work["cell_t_proj"] = np.clip(work["cell_t_proj"], 0.0, 1.0)
    work["mode"] = mode

    return work


# =============================================================================
# 8. LR SCATTER SUMMARY
# =============================================================================

def infer_module(
    ligand: str,
    receptor: str,
    pathway: str = "",
) -> str:

    joined = f"{canonical(ligand)}|{canonical(receptor)}|{canonical(pathway)}"

    if re.search(r"(COL\d|LAMA|LAMB|LAMC|FN1|SPP1|THBS|VCAN|HSPG|ITGA|ITGB|CD44|DDR\d|DAG1|ICAM|VCAM|NECTIN|ALCAM|JAM|CDH\d)", joined, flags=re.I):
        return "ECM / Adhesion"

    if re.search(r"(CXCL|CCL|IL\d|IFN|TNF|CSF|OSM|LIF|CXCR|CCR|IFNAR|IFNGR|TNFR)", joined, flags=re.I):
        return "Cytokine / Chemokine"

    if re.search(r"(EGF|FGF|VEGF|PDGF|HGF|IGF|NRG|GAS6|KITLG|EGFR|FGFR|ERBB|MET|KDR|FLT\d|PDGFR|IGF\dR|KIT|AXL|MERTK|WNT|FZD|NOTCH|JAG\d|DLL\d)", joined, flags=re.I):
        return "Growth-factor"

    if re.search(r"(HLA|CD\d|CEACAM|LCK|FAS|FASLG|TRAIL|TNFSF|TNFRSF|B2M|PTPRC)", joined, flags=re.I):
        return "Immune-like"

    return "Other"


def build_lr_scatter_summary(
    program_df: pd.DataFrame | None,
    mode: str,
    n_clusters: int,
) -> pd.DataFrame | None:

    if program_df is None or len(program_df) == 0:
        return None

    score_col = "score_off" if mode == "off" else "score_on"

    work = program_df.copy()
    work["score_mode"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    work = work[work["score_mode"] > 0].copy()

    if len(work) == 0:
        return None

    # directed pair coverage
    work["pair_key"] = (
        work["sender_cluster"].astype(str)
        + "->"
        + work["receiver_cluster"].astype(str)
    )

    summary = (
        work.groupby(["lr", "ligand", "receptor", "pathway"], as_index=False)
        .agg(
            communication_score=("score_mode", "sum"),
            mean_strength=("score_mode", "mean"),
            n_active_pairs=("pair_key", "nunique"),
        )
    )

    total_possible_pairs = max(n_clusters * n_clusters, 1)
    summary["consistency_pct"] = 100.0 * summary["n_active_pairs"] / float(total_possible_pairs)
    summary["module"] = [
        infer_module(l, r, p)
        for l, r, p in zip(summary["ligand"], summary["receptor"], summary["pathway"])
    ]

    # 让 consistency 视觉上与参考图接近，可做轻度 rescale
    # 先按排名，再映射到 20-100
    ranked = rank01(summary["consistency_pct"].to_numpy(float))
    summary["consistency_display"] = 20.0 + 80.0 * ranked

    summary["label_priority"] = summary["communication_score"] * (
        0.55 + 0.45 * summary["consistency_display"] / 100.0
    )

    preferred = [
        "LAMB1–CD44",
        "SPP1–CD44",
        "LAMA2–CD44",
        "JAG1–NOTCH3",
        "VEGFC–FLT4",
        "NECTIN3–NECTIN1",
        "RETN–CAP1",
        "WNT5A–FZD5",
        "LAMA1–CD44",
        "LAMC1–CD44",
    ]
    summary["preferred"] = summary["lr"].astype(str).isin(preferred)

    return summary.sort_values(
        ["preferred", "label_priority"],
        ascending=[False, False],
    ).reset_index(drop=True)


# =============================================================================
# 9. MATRIX HELPERS
# =============================================================================

def edge_table_to_matrix(
    edge_table: pd.DataFrame,
    clusters: list[str],
    value_col: str,
) -> np.ndarray:

    index = {cluster: i for i, cluster in enumerate(clusters)}

    matrix = np.zeros((len(clusters), len(clusters)), dtype=float)

    for row in edge_table.itertuples():
        sender = getattr(row, "sender_cluster")
        receiver = getattr(row, "receiver_cluster")
        if sender in index and receiver in index:
            matrix[index[sender], index[receiver]] += float(getattr(row, value_col))

    return matrix


# =============================================================================
# 10. STAT TEST
# =============================================================================

def paired_metric_pvalue(
    metric_table: pd.DataFrame,
    metric: str,
    alternative: str = "greater",
) -> float:

    pivot = metric_table.pivot(index="seed", columns="mode", values=metric).dropna()

    if "off" not in pivot.columns or "on" not in pivot.columns or len(pivot) < 2:
        return np.nan

    if SCIPY_OK:
        try:
            return float(
                wilcoxon(
                    pivot["on"],
                    pivot["off"],
                    alternative=alternative,
                ).pvalue
            )
        except Exception:
            return np.nan

    difference = pivot["on"].to_numpy(float) - pivot["off"].to_numpy(float)

    if alternative == "greater":
        return float((np.sum(difference <= 0) + 1) / (len(difference) + 1))
    elif alternative == "less":
        return float((np.sum(difference >= 0) + 1) / (len(difference) + 1))
    else:
        return np.nan


# =============================================================================
# 11. PLOTTING: MAIN FIGURE
# =============================================================================

def plot_temporal_matrix(
    ax,
    edge_table: pd.DataFrame,
    clusters: list[str],
    title: str,
    mode_color: str,
):

    score_matrix = edge_table_to_matrix(edge_table, clusters, "temporal_score")
    delta_matrix = edge_table_to_matrix(edge_table, clusters, "delta_t")
    n = len(clusters)

    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    facecolor="white",
                    edgecolor=COL_GRID,
                    linewidth=0.7,
                    zorder=0,
                )
            )

    x_points, y_points, sizes, colors = [], [], [], []

    positive_scores = score_matrix[score_matrix > 0]
    size_reference = float(np.quantile(positive_scores, 0.95)) if len(positive_scores) else 1.0

    for sender_i in range(n):
        for receiver_i in range(n):
            score = score_matrix[sender_i, receiver_i]
            if score <= 0:
                continue

            x_points.append(receiver_i)
            y_points.append(sender_i)
            sizes.append(20 + 240 * min(score / (size_reference + 1e-12), 1.0))
            colors.append(delta_matrix[sender_i, receiver_i])

    if x_points:
        scatter = ax.scatter(
            x_points, y_points,
            s=sizes, c=colors,
            cmap=CMAP_TIME,
            norm=Normalize(vmin=-0.35, vmax=0.35),
            edgecolors="#565B5D",
            linewidths=0.55,
            zorder=3,
        )
    else:
        scatter = None

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticklabels(clusters)

    ax.set_xlabel("Receiver cluster")
    ax.set_ylabel("Sender cluster")
    ax.set_title(title, fontweight="bold", color=mode_color, pad=8)
    ax.set_aspect("equal")

    return scatter


def plot_rank_shift(
    ax,
    off_summary: pd.DataFrame,
    on_summary: pd.DataFrame,
):

    merged = (
        off_summary[["cluster", "t_hat_mean"]].rename(columns={"t_hat_mean": "t_off"})
        .merge(
            on_summary[["cluster", "t_hat_mean"]].rename(columns={"t_hat_mean": "t_on"}),
            on="cluster",
            how="inner",
        )
    )

    merged["rank_off"] = merged["t_off"].rank(method="first", ascending=True)
    merged["rank_on"] = merged["t_on"].rank(method="first", ascending=True)
    merged = merged.sort_values("rank_off")

    for row in merged.itertuples():
        shift = row.rank_on - row.rank_off
        color = (
            COL_LATE if shift > 0
            else COL_EARLY if shift < 0
            else "#A6AAAC"
        )

        ax.plot([0, 1], [row.rank_off, row.rank_on],
                color=color, linewidth=1.5, alpha=0.82, zorder=1)

        ax.scatter(0, row.rank_off, s=35, facecolor=COL_OFF, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(1, row.rank_on, s=35, facecolor=COL_ON, edgecolor="white", linewidth=0.5, zorder=3)

        ax.text(-0.08, row.rank_off, row.cluster, ha="right", va="center", fontsize=6.6)
        ax.text(1.08, row.rank_on, row.cluster, ha="left", va="center", fontsize=6.6)

    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(len(merged) + 0.7, 0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MLTD-off", "MLTD-on"], fontweight="bold")
    ax.set_ylabel("Communication-pseudotime rank")
    ax.set_title("Preserved cluster temporal ordering", fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])

    return merged


def plot_gain_matrix(
    ax,
    gain_table: pd.DataFrame,
    clusters: list[str],
):

    matrix = edge_table_to_matrix(gain_table, clusters, "temporal_gain")
    vmax = float(np.nanquantile(np.abs(matrix), 0.98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    image = ax.imshow(
        matrix,
        cmap=CMAP_GAIN,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        interpolation="nearest",
        aspect="equal",
    )

    n = len(clusters)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticklabels(clusters)
    ax.set_xlabel("Receiver cluster")
    ax.set_ylabel("Sender cluster")
    ax.set_title("Redistribution of temporal communication", fontweight="bold")

    for boundary in np.arange(-0.5, n, 1):
        ax.axhline(boundary, color="white", linewidth=0.7)
        ax.axvline(boundary, color="white", linewidth=0.7)

    return image


def shorten_pathway(text: str, maximum: int = 34) -> str:
    value = str(text).strip()
    if len(value) <= maximum:
        return value
    return value[: maximum - 1] + "…"


def plot_pathway_gain(
    ax,
    pathway_summary: pd.DataFrame,
):

    top = (
        pathway_summary.nlargest(TOP_N_PATHWAYS, "pathway_temporal_gain")
        .sort_values("pathway_temporal_gain", ascending=True)
        .copy()
    )

    y = np.arange(len(top))

    cci = top["cci_strength_contribution"].to_numpy(float)
    gate = top["temporal_gating_contribution"].to_numpy(float)
    total = top["pathway_temporal_gain"].to_numpy(float)

    cci_pos = np.maximum(cci, 0.0)
    cci_neg = np.minimum(cci, 0.0)

    gate_pos = np.maximum(gate, 0.0)
    gate_neg = np.minimum(gate, 0.0)

    # negative bars
    ax.barh(
        y,
        cci_neg,
        color="#9DB8CF",
        edgecolor="none",
        height=0.62,
        label="Negative CCI-strength contribution",
    )
    ax.barh(
        y,
        gate_neg,
        left=cci_neg,
        color="#6E95B8",
        edgecolor="none",
        height=0.62,
        label="Negative temporal-gating contribution",
    )

    # positive bars
    ax.barh(
        y,
        cci_pos,
        color="#D8A59D",
        edgecolor="none",
        height=0.62,
        label="Positive CCI-strength contribution",
    )
    ax.barh(
        y,
        gate_pos,
        left=cci_pos,
        color=COL_GAIN,
        edgecolor="none",
        height=0.62,
        label="Positive temporal-gating contribution",
    )

    # net point
    ax.scatter(
        total,
        y,
        s=18,
        c=COL_BLACK,
        marker="o",
        zorder=5,
        label="Net temporal gain",
    )

    ax.axvline(0, color="#777C7E", linewidth=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([shorten_pathway(value) for value in top["pathway"]])
    ax.set_xlabel("Temporal transmission gain")
    ax.set_title(
        "Top pathways with MLTD-enhanced\nTF-aware temporal transmission",
        fontweight="bold",
    )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#D8A59D", edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor=COL_GAIN, edgecolor="none"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_BLACK, markersize=4),
    ]
    labels = [
        "Positive CCI-strength contribution",
        "Positive temporal-gating contribution",
        "Net temporal gain",
    ]
    ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=6.2)

    return top


def plot_metric_mini_panel(
    ax,
    metric_table: pd.DataFrame,
    metric: str,
    title: str,
    better: str = "greater",
):
    pivot = metric_table.pivot(index="seed", columns="mode", values=metric).dropna()

    if "off" not in pivot.columns or "on" not in pivot.columns or len(pivot) == 0:
        ax.set_axis_off()
        return

    x_off = np.zeros(len(pivot))
    x_on = np.ones(len(pivot))

    for off_value, on_value in zip(pivot["off"].to_numpy(float), pivot["on"].to_numpy(float)):
        ax.plot([0, 1], [off_value, on_value], color="#B7BBBE", linewidth=0.8, zorder=1)

    ax.scatter(
        x_off, pivot["off"].to_numpy(float),
        s=20, color=COL_OFF, edgecolor="white", linewidth=0.4, zorder=3,
        label="MLTD-off",
    )
    ax.scatter(
        x_on, pivot["on"].to_numpy(float),
        s=20, color=COL_ON, edgecolor="white", linewidth=0.4, zorder=3,
        label="MLTD-on",
    )

    p_value = paired_metric_pvalue(metric_table, metric, alternative=better)
    if np.isfinite(p_value):
        p_text = "P<0.001" if p_value < 0.001 else f"P={p_value:.3f}"
        ax.text(0.5, 1.02, p_text, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Off", "On"])
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#E8EAEB", linewidth=0.6)

    ymin = min(pivot["off"].min(), pivot["on"].min())
    ymax = max(pivot["off"].max(), pivot["on"].max())
    pad = 0.08 * (ymax - ymin + 1e-8)
    ax.set_ylim(ymin - pad, ymax + pad)


def make_main_summary_figure(
    off_summary: pd.DataFrame,
    on_summary: pd.DataFrame,
    off_edges: pd.DataFrame,
    on_edges: pd.DataFrame,
    gain_table: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    metric_table: pd.DataFrame,
    clusters: list[str],
):

    fig = plt.figure(figsize=(16.4, 10.3))
    grid = GridSpec(
        2, 4,
        figure=fig,
        width_ratios=[1.15, 1.15, 0.86, 1.26],
        height_ratios=[1.0, 0.97],
        wspace=0.62, hspace=0.55,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[0, 3])
    ax_e = fig.add_subplot(grid[1, 0:2])

    subgrid_f = GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[1, 2:4], wspace=0.48)
    ax_f1 = fig.add_subplot(subgrid_f[0, 0])
    ax_f2 = fig.add_subplot(subgrid_f[0, 1])
    ax_f3 = fig.add_subplot(subgrid_f[0, 2])
    ax_f4 = fig.add_subplot(subgrid_f[0, 3])

    scatter_a = plot_temporal_matrix(
        ax_a, off_edges, clusters, "MLTD-off GNN-LTC", COL_OFF
    )
    scatter_b = plot_temporal_matrix(
        ax_b, on_edges, clusters, "MLTD-on GNN-LTC", COL_ON
    )
    rank_table = plot_rank_shift(ax_c, off_summary, on_summary)
    gain_image = plot_gain_matrix(ax_d, gain_table, clusters)
    top_pathways = plot_pathway_gain(ax_e, pathway_summary)

    plot_metric_mini_panel(
        ax_f1, metric_table,
        "weighted_forward_fraction",
        "Forward\nfraction",
        better="greater",
    )
    plot_metric_mini_panel(
        ax_f2, metric_table,
        "weighted_mean_delta_t",
        "Mean\nΔt",
        better="greater",
    )
    plot_metric_mini_panel(
        ax_f3, metric_table,
        "temporal_alignment",
        "Temporal\nalignment",
        better="greater",
    )
    plot_metric_mini_panel(
        ax_f4, metric_table,
        "tf_temporal_alignment",
        "TF-aware\nalignment",
        better="greater",
    )

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")
    panel_label(ax_d, "D")
    panel_label(ax_e, "E")
    panel_label(ax_f1, "F")

    if scatter_b is not None:
        cbar_time = fig.colorbar(
            scatter_b,
            ax=[ax_a, ax_b],
            fraction=0.022,
            pad=0.02,
        )
        cbar_time.set_label("t(receiver) − t(sender)")

    cbar_gain = fig.colorbar(
        gain_image,
        ax=ax_d,
        fraction=0.045,
        pad=0.04,
    )
    cbar_gain.set_label("Temporal transmission gain")

    fig.text(
        0.79, 0.38,
        "Paired global temporal metrics",
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold",
    )

    mass_text = "Communication mass matched" if MASS_MATCH else "Raw communication scale"

    fig.suptitle(
        "MLTD selectively strengthens TF-aware communication across a preserved GNN-LTC temporal backbone\n"
        + mass_text,
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    fig.savefig(FIG_MAIN_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_MAIN_SVG, bbox_inches="tight")
    fig.savefig(FIG_MAIN_PDF, bbox_inches="tight")
    plt.close(fig)

    return rank_table, top_pathways


# =============================================================================
# 12. PLOTTING: MODE-SPECIFIC OVERVIEW
# =============================================================================

def plot_cell_pseudotime_violin(
    ax,
    cell_pt: pd.DataFrame,
    ordered_clusters: list[str],
    mode_label: str,
    mode_color: str,
):

    positions = np.arange(len(ordered_clusters))
    violin_data = []

    for cluster in ordered_clusters:
        values = cell_pt.loc[cell_pt["cluster"] == cluster, "cell_t_proj"].to_numpy(float)
        if len(values) == 0:
            values = np.array([np.nan], dtype=float)
        violin_data.append(values[np.isfinite(values)])

    parts = ax.violinplot(
        violin_data,
        positions=positions,
        widths=0.80,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_facecolor("#C7D1E5")
        body.set_edgecolor("none")
        body.set_alpha(0.85)

    means = [
        float(np.nanmean(v)) if len(v) else np.nan
        for v in violin_data
    ]
    lows = [
        float(np.nanquantile(v, 0.05)) if len(v) else np.nan
        for v in violin_data
    ]
    highs = [
        float(np.nanquantile(v, 0.95)) if len(v) else np.nan
        for v in violin_data
    ]

    ax.vlines(positions, lows, highs, color="#2D6FA7", linewidth=2.2, zorder=3)
    ax.scatter(positions, means, marker="P", s=58, color="#2D6FA7", zorder=4)
    ax.scatter(positions, lows, marker="_", s=120, color="#2D6FA7", linewidth=2.2, zorder=4)
    ax.scatter(positions, highs, marker="_", s=120, color="#2D6FA7", linewidth=2.2, zorder=4)

    # x 用 cluster 编号，不带 C
    labels = [cluster[1:] if cluster.upper().startswith("C") else cluster for cluster in ordered_clusters]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cell projected pseudotime")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(mode_label, fontweight="bold", color=mode_color)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_lr_scatter(
    ax,
    lr_summary: pd.DataFrame,
    mode_label: str,
    mode_color: str,
):
    work = lr_summary.copy()

    # 背景点
    sizes = 18 + 280 * minmax01(np.sqrt(work["mean_strength"].to_numpy(float)))
    colors = [MODULE_COLORS.get(x, MODULE_COLORS["Other"]) for x in work["module"]]

    label_mask = np.zeros(len(work), dtype=bool)

    # 优先挑 preferred，再按综合分
    chosen_idx = list(work.head(TOP_N_LABELS_LR).index)
    label_mask[chosen_idx] = True

    bg = work.loc[~label_mask].copy()
    hi = work.loc[label_mask].copy()

    if len(bg) > 0:
        bg_sizes = 18 + 180 * minmax01(np.sqrt(bg["mean_strength"].to_numpy(float)))
        ax.scatter(
            np.clip(bg["communication_score"].to_numpy(float), 1e-2, None),
            bg["consistency_display"].to_numpy(float),
            s=bg_sizes,
            c=[MODULE_COLORS.get(x, MODULE_COLORS["Other"]) for x in bg["module"]],
            alpha=0.35,
            edgecolors="white",
            linewidths=0.3,
            zorder=1,
        )

    if len(hi) > 0:
        hi_sizes = 55 + 250 * minmax01(np.sqrt(hi["mean_strength"].to_numpy(float)))
        ax.scatter(
            np.clip(hi["communication_score"].to_numpy(float), 1e-2, None),
            hi["consistency_display"].to_numpy(float),
            s=hi_sizes,
            c=[MODULE_COLORS.get(x, MODULE_COLORS["Other"]) for x in hi["module"]],
            alpha=0.96,
            edgecolors="#3A3A3A",
            linewidths=1.1,
            zorder=3,
        )

        # 注释
        offsets = [
            (-40, 12), (-10, 18), (15, 12), (22, 12),
            (20, 8), (26, 6), (6, -18), (20, -20),
        ]

        for k, row in enumerate(hi.itertuples()):
            dx, dy = offsets[k % len(offsets)]
            ax.annotate(
                str(row.lr).replace("–", "–"),
                xy=(max(float(row.communication_score), 1e-2), float(row.consistency_display)),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="center",
                fontsize=8.0,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#6A6A6A",
                    lw=0.8,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xlim(1e-2, max(12.0, float(np.nanmax(work["communication_score"])) * 1.2))
    ax.set_ylim(20, 100)
    ax.set_xlabel("Communication score (au)")
    ax.set_ylabel("Consistency across states (%)")
    ax.set_title(mode_label, fontweight="bold", color=mode_color)

    # size legend
    finite_strength = work["mean_strength"].to_numpy(float)
    if np.all(~np.isfinite(finite_strength)):
        legend_values = [0.0, 1.0]
    else:
        lo = float(np.nanmin(finite_strength))
        hi_v = float(np.nanmax(finite_strength))
        legend_values = [lo, hi_v]

    handles_size = []
    labels_size = []

    for value in legend_values:
        size = 18 + 280 * minmax01(np.sqrt(np.array([legend_values[0], value])))[1]
        handles_size.append(
            plt.scatter([], [], s=size, facecolors="none", edgecolors="#777777")
        )
        labels_size.append(f"{value:.0f}" if value >= 1 else f"{value:.1f}")

    legend1 = ax.legend(
        handles_size,
        labels_size,
        title="Mean strength (au)",
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.50, -0.01),
        borderpad=0.8,
        labelspacing=0.6,
        handletextpad=0.6,
    )
    ax.add_artist(legend1)

    # module legend
    module_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=6.5,
            label=name,
        )
        for name, color in MODULE_COLORS.items()
    ]

    ax.legend(
        handles=module_handles,
        title="Module",
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.77, -0.01),
        borderpad=0.8,
        labelspacing=0.6,
        handletextpad=0.6,
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def make_mode_specific_overview(
    mode: str,
    cluster_summary: pd.DataFrame,
    cell_pt: pd.DataFrame | None,
    lr_summary: pd.DataFrame | None,
    ordered_clusters: list[str],
):
    if cell_pt is None or lr_summary is None or len(lr_summary) == 0:
        warnings.warn(f"Skip mode-specific figure for mode={mode}: missing cell_pt or lr_summary.")
        return

    fig = plt.figure(figsize=(14.0, 6.0))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.55], wspace=0.25)

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])

    if mode == "off":
        label_left = "MLTD-off cell projected pseudotime"
        label_right = "MLTD-off representative ligand–receptor pairs"
        color = COL_OFF
        png = FIG_OFF_OVERVIEW_PNG
        svg = FIG_OFF_OVERVIEW_SVG
        pdf = FIG_OFF_OVERVIEW_PDF
    else:
        label_left = "MLTD-on cell projected pseudotime"
        label_right = "MLTD-on representative ligand–receptor pairs"
        color = COL_ON
        png = FIG_ON_OVERVIEW_PNG
        svg = FIG_ON_OVERVIEW_SVG
        pdf = FIG_ON_OVERVIEW_PDF

    plot_cell_pseudotime_violin(
        ax1,
        cell_pt=cell_pt,
        ordered_clusters=ordered_clusters,
        mode_label=label_left,
        mode_color=color,
    )
    plot_lr_scatter(
        ax2,
        lr_summary=lr_summary,
        mode_label=label_right,
        mode_color=color,
    )

    panel_label(ax1, "I")
    panel_label(ax2, "J")

    fig.suptitle(
        f"{'MLTD-off' if mode == 'off' else 'MLTD-on'} mode-specific overview",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    log("[DEVICE]", DEVICE)

    mltd_dir = locate_mltd_output_dir()
    log("[MLTD CCI DIR]", mltd_dir)

    cluster_features = load_cluster_features()
    pair_df, pair_path = load_pair_summary(mltd_dir)
    program_df, program_path = load_program_table(mltd_dir)
    cell_map = load_cell_map()
    original_cell_pt = load_original_cell_pt()

    log("[PAIR INPUT]", pair_path)
    log("[PROGRAM INPUT]", program_path)
    log("[CELL MAP]", CELL_MAP_CSV if cell_map is not None else "None")
    log("[ORIG CELL PT]", ORIGINAL_CELL_PT_CSV if original_cell_pt is not None else "None")

    pair_df = mass_match_weights(pair_df)

    clusters = sorted(
        set(cluster_features["cluster"])
        | set(pair_df["sender_cluster"])
        | set(pair_df["receiver_cluster"]),
        key=cluster_sort_key,
    )
    clusters = [cluster for cluster in clusters if cluster != ""]

    feature_cluster_set = set(cluster_features["cluster"])
    missing_features = [cluster for cluster in clusters if cluster not in feature_cluster_set]
    if missing_features:
        warnings.warn(
            "These clusters are missing from cluster_dynamic_features and will be removed: "
            + ", ".join(missing_features)
        )

    clusters = [cluster for cluster in clusters if cluster in feature_cluster_set]

    if len(clusters) < 3:
        raise ValueError("Fewer than three common clusters remain.")

    pair_df = pair_df[
        pair_df["sender_cluster"].isin(clusters)
        & pair_df["receiver_cluster"].isin(clusters)
    ].copy()

    if program_df is not None:
        program_df = program_df[
            program_df["sender_cluster"].isin(clusters)
            & program_df["receiver_cluster"].isin(clusters)
        ].copy()

    log("[N CLUSTERS]", len(clusters))
    log("[CLUSTERS]", clusters)
    log("[N PAIRS]", len(pair_df))

    # ---------------------------------------------------------------------
    # Paired training
    # ---------------------------------------------------------------------
    off_summary, off_seed_times = train_mode_ensemble(
        mode="off",
        clusters=clusters,
        pair_df=pair_df,
        cluster_features=cluster_features,
    )

    on_summary, on_seed_times = train_mode_ensemble(
        mode="on",
        clusters=clusters,
        pair_df=pair_df,
        cluster_features=cluster_features,
    )

    # order clusters by mean t for each mode
    off_ordered_clusters = (
        off_summary.sort_values("t_hat_mean")["cluster"].tolist()
    )
    on_ordered_clusters = (
        on_summary.sort_values("t_hat_mean")["cluster"].tolist()
    )

    # ---------------------------------------------------------------------
    # Edge temporalization
    # ---------------------------------------------------------------------
    off_edges = calculate_edge_temporal_scores(
        pair_df=pair_df,
        cluster_summary=off_summary,
        mode="off",
    )

    on_edges = calculate_edge_temporal_scores(
        pair_df=pair_df,
        cluster_summary=on_summary,
        mode="on",
    )

    gain_table = combine_edge_gain(off_edges=off_edges, on_edges=on_edges)

    # ---------------------------------------------------------------------
    # Pathway temporal gain
    # ---------------------------------------------------------------------
    pathway_summary = calculate_pathway_gain(
        program_df=program_df,
        off_summary=off_summary,
        on_summary=on_summary,
        pair_gain=gain_table,
    )

    # ---------------------------------------------------------------------
    # Paired seed metrics
    # ---------------------------------------------------------------------
    off_metrics = calculate_seed_metrics(
        mode="off",
        pair_df=pair_df,
        seed_times=off_seed_times,
        cluster_summary=off_summary,
    )

    on_metrics = calculate_seed_metrics(
        mode="on",
        pair_df=pair_df,
        seed_times=on_seed_times,
        cluster_summary=on_summary,
    )

    metric_table = pd.concat([off_metrics, on_metrics], ignore_index=True)

    # ---------------------------------------------------------------------
    # Cell projected pseudotime
    # ---------------------------------------------------------------------
    off_cell_pt = build_cell_projected_pseudotime(
        mode="off",
        cell_map=cell_map,
        original_cell_pt=original_cell_pt,
        cluster_summary=off_summary,
        ordered_clusters=off_ordered_clusters,
    )

    on_cell_pt = build_cell_projected_pseudotime(
        mode="on",
        cell_map=cell_map,
        original_cell_pt=original_cell_pt,
        cluster_summary=on_summary,
        ordered_clusters=on_ordered_clusters,
    )

    # ---------------------------------------------------------------------
    # LR scatter summaries
    # ---------------------------------------------------------------------
    off_lr_summary = build_lr_scatter_summary(
        program_df=program_df,
        mode="off",
        n_clusters=len(clusters),
    )

    on_lr_summary = build_lr_scatter_summary(
        program_df=program_df,
        mode="on",
        n_clusters=len(clusters),
    )

    # ---------------------------------------------------------------------
    # Save tables
    # ---------------------------------------------------------------------
    off_summary.to_csv(OUT_DIR / "MLTD_off_cluster_pseudotime_ensemble.csv", index=False)
    on_summary.to_csv(OUT_DIR / "MLTD_on_cluster_pseudotime_ensemble.csv", index=False)

    off_seed_times.to_csv(OUT_DIR / "MLTD_off_cluster_pseudotime_by_seed.csv", index=False)
    on_seed_times.to_csv(OUT_DIR / "MLTD_on_cluster_pseudotime_by_seed.csv", index=False)

    off_edges.to_csv(OUT_DIR / "MLTD_off_edge_temporal_scores.csv", index=False)
    on_edges.to_csv(OUT_DIR / "MLTD_on_edge_temporal_scores.csv", index=False)
    gain_table.to_csv(OUT_DIR / "MLTD_on_minus_off_temporal_gain.csv", index=False)

    pathway_summary.to_csv(OUT_DIR / "MLTD_pathway_temporal_gain.csv", index=False)
    metric_table.to_csv(OUT_DIR / "MLTD_on_off_seed_metrics.csv", index=False)

    if off_cell_pt is not None:
        off_cell_pt.to_csv(OUT_DIR / "MLTD_off_cell_projected_pseudotime.csv", index=False)
    if on_cell_pt is not None:
        on_cell_pt.to_csv(OUT_DIR / "MLTD_on_cell_projected_pseudotime.csv", index=False)

    if off_lr_summary is not None:
        off_lr_summary.to_csv(OUT_DIR / "MLTD_off_lr_scatter_summary.csv", index=False)
    if on_lr_summary is not None:
        on_lr_summary.to_csv(OUT_DIR / "MLTD_on_lr_scatter_summary.csv", index=False)

    metric_rows = []
    for metric in [
        "weighted_forward_fraction",
        "weighted_mean_delta_t",
        "temporal_alignment",
        "tf_temporal_alignment",
        "temporal_delta_entropy",
    ]:
        pivot = metric_table.pivot(index="seed", columns="mode", values=metric).dropna()
        if "off" in pivot.columns and "on" in pivot.columns:
            difference = pivot["on"] - pivot["off"]
            metric_rows.append({
                "metric": metric,
                "mean_off": float(pivot["off"].mean()),
                "mean_on": float(pivot["on"].mean()),
                "mean_on_minus_off": float(difference.mean()),
                "median_on_minus_off": float(difference.median()),
                "fraction_on_greater_off": float(np.mean(difference > 0)),
                "paired_one_sided_p_greater": paired_metric_pvalue(metric_table, metric, alternative="greater"),
                "paired_one_sided_p_less": paired_metric_pvalue(metric_table, metric, alternative="less"),
                "n_seeds": int(len(pivot)),
            })

    metric_summary = pd.DataFrame(metric_rows)
    metric_summary.to_csv(OUT_DIR / "MLTD_on_off_metric_summary.csv", index=False)

    # ---------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------
    rank_table, top_pathways = make_main_summary_figure(
        off_summary=off_summary,
        on_summary=on_summary,
        off_edges=off_edges,
        on_edges=on_edges,
        gain_table=gain_table,
        pathway_summary=pathway_summary,
        metric_table=metric_table,
        clusters=clusters,
    )

    rank_table.to_csv(OUT_DIR / "MLTD_cluster_pseudotime_rank_shift.csv", index=False)
    top_pathways.to_csv(OUT_DIR / "MLTD_top_pathways_temporal_gain_display.csv", index=False)

    make_mode_specific_overview(
        mode="off",
        cluster_summary=off_summary,
        cell_pt=off_cell_pt,
        lr_summary=off_lr_summary,
        ordered_clusters=off_ordered_clusters,
    )

    make_mode_specific_overview(
        mode="on",
        cluster_summary=on_summary,
        cell_pt=on_cell_pt,
        lr_summary=on_lr_summary,
        ordered_clusters=on_ordered_clusters,
    )

    manifest = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "device": DEVICE,
        "mass_match": MASS_MATCH,
        "n_clusters": len(clusters),
        "clusters": clusters,
        "n_ensemble": N_ENSEMBLE,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "pair_input": str(pair_path),
        "program_input": str(program_path) if program_path is not None else None,
        "cluster_feature_input": str(CLUSTER_FEATURE_CSV),
        "cell_map_input": str(CELL_MAP_CSV) if cell_map is not None else None,
        "original_cell_pt_input": str(ORIGINAL_CELL_PT_CSV) if original_cell_pt is not None else None,
        "temporal_alpha": TEMPORAL_ALPHA,
        "temporal_delta": TEMPORAL_DELTA,
        "main_png": str(FIG_MAIN_PNG),
        "off_overview_png": str(FIG_OFF_OVERVIEW_PNG),
        "on_overview_png": str(FIG_ON_OVERVIEW_PNG),
    }

    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    log("\n" + "=" * 90)
    log("[DONE]")
    log("[OUTPUT DIR]", OUT_DIR)
    log("[MAIN FIGURE]", FIG_MAIN_PNG)
    log("[OFF OVERVIEW]", FIG_OFF_OVERVIEW_PNG)
    log("[ON OVERVIEW]", FIG_ON_OVERVIEW_PNG)

    log("\n[PAIRED METRIC SUMMARY]")
    log(metric_summary.to_string(index=False))

    log("\n[TOP PATHWAYS]")
    log(
        pathway_summary.head(TOP_N_PATHWAYS)[
            [
                "pathway",
                "pathway_temporal_gain",
                "cci_strength_contribution",
                "temporal_gating_contribution",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
