# -*- coding: utf-8 -*-
"""
correct_MLTD_on_off_GNN_LTC_figures_and_LR_scatter_v3.py
========================================================

用途
----
修正已经完成的 MLTD-on / MLTD-off paired GNN-LTC 分析中的可视化。

本脚本不重新训练 GNN-LTC，而是读取已有输出并重新生成：

1. 修正版 A–F 主图
2. MLTD-off mode-specific overview
   - cell projected pseudotime violin
   - LIANA-consistent LR scatter
3. MLTD-on mode-specific overview
   - cell projected pseudotime violin
   - LIANA-consistent LR scatter
4. 各 mode 的 LR scatter summary CSV
5. 各 mode 的 cell-projected pseudotime CSV

LR scatter 严格恢复原始计算逻辑
-------------------------------
score_mode
× sigmoid(alpha × (t_receiver - t_sender - delta))
→ 3% relative temporal pruning
→ LR × receiver cluster 聚合
→ active receiver coverage
→ normalized entropy
→ consistency_raw = 0.62 × coverage + 0.38 × entropy
→ 分层背景抽样
→ 主背景 / 右下背景 / highlight 分区 rank spread
→ deterministic jitter

预期已有输入
------------
communication_pseudotime_clusterGNN_LTC/
MLTD_on_off_paired_GNN_LTC_v2/

至少包含：
- MLTD_off_cluster_pseudotime_ensemble.csv
- MLTD_on_cluster_pseudotime_ensemble.csv
- MLTD_off_edge_temporal_scores.csv
- MLTD_on_edge_temporal_scores.csv
- MLTD_on_minus_off_temporal_gain.csv
- MLTD_pathway_temporal_gain.csv
- MLTD_on_off_seed_metrics.csv

同时需要：
- same_universe_program_on_off_scores.csv
- cell-level mapping
- 原始 cell_projected_pseudotime.csv（推荐，但非必需）
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, Any

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

try:
    from scipy.stats import wilcoxon
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

ORIGINAL_LTC_DIR = (
    CCI_DIR
    / "communication_pseudotime_clusterGNN_LTC"
)

# 之前脚本生成结果的候选目录
RESULT_DIR_CANDIDATES = [
    ORIGINAL_LTC_DIR / "MLTD_on_off_paired_GNN_LTC_v2",
    ORIGINAL_LTC_DIR / "MLTD_on_off_paired_GNN_LTC",
]

# MLTD-on/off CCI 输出目录
MLTD_CCI_DIR_CANDIDATES = [
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_4",

    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_3",

    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_2",
]

CELL_MAP_CSV = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_recluster_for_cci"
    / f"{SAMPLE_ID}_cell_level_mapping_with_umap_and_internal16.csv"
)

ORIGINAL_CELL_PT_CSV = (
    ORIGINAL_LTC_DIR
    / "cell_projected_pseudotime.csv"
)

# 最终修正版输出
CORRECTED_OUT_DIR_NAME = "corrected_publication_figures_v3"


# =============================================================================
# 0.1 LIANA-CONSISTENT LR SCATTER CONFIG
# =============================================================================

TEMPORAL_ALPHA = 12.0
TEMPORAL_DELTA = 0.05
POST_KEEP_RELATIVE_TO_MAX = 0.03

TOP_LR = 14

BACKGROUND_N = 90
BACKGROUND_BINS = 10
RIGHT_LOW_BG_N = 8

MAX_LABELS = 8

PREFERRED_LABEL_ORDER = [
    "LAMB1-CD44",
    "SPP1-CD44",
    "LAMA2-CD44",
    "LAMC1-CD44",
    "JAG1-NOTCH3",
    "VEGFC-FLT4",
    "NECTIN3-NECTIN1",
    "RETN-CAP1",
    "WNT5A-FZD5",
    "PTN-NCL",
    "MDK-NCL",
    "CD8A-CEACAM5",
    "IGF2-IGF2R",
]

# scatter axes
X_MIN = 1e-2
X_MAX = 12.0

Y_MIN = 20.0
Y_MAX = 100.0

# bubble size
BG_SIZE_MIN = 10.0
BG_SIZE_MAX = 62.0

HI_SIZE_MIN = 150.0
HI_SIZE_MAX = 460.0

BG_ALPHA = 0.55
HI_ALPHA = 0.96

# consistency
ACTIVE_WITHIN_LR_FRAC = 0.18
GLOBAL_ACTIVE_Q = 0.35

# layout regions
BG_X_LO = 1e-2
BG_X_HI = 1.85

HI_X_LO = 2.2
HI_X_HI = 11.5

BG_Y_LO = 22.0
BG_Y_HI = 89.0

RIGHT_LOW_X_LO = 1.55
RIGHT_LOW_X_HI = 4.8

RIGHT_LOW_Y_LO = 26.0
RIGHT_LOW_Y_HI = 54.0

HI_Y_LO = 52.0
HI_Y_HI = 97.0

# deterministic jitter
X_JITTER_SIGMA_BG = 0.15
X_JITTER_SIGMA_HI = 0.16

Y_JITTER_BG = 7.8
Y_JITTER_HI = 6.4


# =============================================================================
# 0.2 GENERAL FIGURE STYLE
# =============================================================================

DPI = 450

COL_OFF = "#8B969C"
COL_ON = "#C45843"

COL_GAIN = "#B54A42"
COL_LOSS = "#4C79A5"

COL_TEXT = "#242729"
COL_MUTED = "#747A7D"
COL_GRID = "#D8DBDD"

COL_BLACK = "#292929"

MODULE_COLORS = {
    "ECM / Adhesion": "#8DAA43",
    "Cytokine / Chemokine": "#4FA7A3",
    "Growth-factor": "#4F7DB8",
    "Immune-like": "#8B6CAF",
    "Other": "#B9C0C4",
}

CMAP_TIME = LinearSegmentedColormap.from_list(
    "time_map",
    [
        "#426A9B",
        "#F7F6F2",
        "#C55B46",
    ],
)

CMAP_GAIN = LinearSegmentedColormap.from_list(
    "gain_map",
    [
        "#3D73A5",
        "#F8F7F3",
        "#B64C44",
    ],
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ],
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
# 1. BASIC HELPERS
# =============================================================================

def log(*args):
    print(*args, flush=True)


def normalize_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def canonical(value: Any) -> str:
    return str(value).strip().upper()


def normalize_cluster(value: Any) -> str:
    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
    }:
        return ""

    match = re.search(
        r"(\d+)",
        text,
    )

    if match:
        return f"C{int(match.group(1))}"

    if text.upper().startswith("C"):
        return text.upper()

    return f"C{text}"


def cluster_sort_key(value: str):
    text = normalize_cluster(value)

    match = re.fullmatch(
        r"C(\d+)",
        text,
    )

    if match:
        return 0, int(match.group(1))

    return 1, text


def norm_barcode(value: Any) -> str:
    text = str(value)

    if "::" in text:
        return text.split("::", 1)[-1]

    return text


def first_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = True,
    role: str = "column",
) -> str | None:

    lookup = {
        normalize_name(column): str(column)
        for column in df.columns
    }

    for alias in aliases:
        key = normalize_name(alias)

        if key in lookup:
            return lookup[key]

    candidates = []

    for column in df.columns:
        column_key = normalize_name(column)

        for alias in aliases:
            alias_key = normalize_name(alias)

            if len(alias_key) < 5:
                continue

            if (
                column_key.startswith(alias_key)
                or column_key.endswith(alias_key)
            ):
                candidates.append(str(column))

    if candidates:
        candidates = sorted(
            set(candidates),
            key=lambda x: (
                len(x),
                x,
            ),
        )

        return candidates[0]

    if required:
        raise KeyError(
            f"Cannot identify {role}.\n"
            f"Actual columns:\n{df.columns.tolist()}\n"
            f"Accepted aliases:\n{list(aliases)}"
        )

    return None


def finite_numeric(
    values,
    default: float = 0.0,
) -> np.ndarray:

    if isinstance(values, np.ndarray):
        series = pd.Series(values)
    else:
        series = pd.Series(values)

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(default)
        .to_numpy(float)
    )


def minmax01(values) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=float,
    )

    output = np.zeros_like(
        x,
        dtype=float,
    )

    finite = np.isfinite(x)

    if not finite.any():
        return output

    low = float(
        np.nanmin(x[finite])
    )

    high = float(
        np.nanmax(x[finite])
    )

    if high - low < 1e-12:
        output[finite] = (
            0.5
            if high > 0
            else 0.0
        )
    else:
        output[finite] = (
            x[finite] - low
        ) / (
            high - low
        )

    return np.clip(
        output,
        0.0,
        1.0,
    )


def rank01(values) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=float,
    )

    if len(x) <= 1:
        return np.zeros_like(x)

    ranks = (
        pd.Series(x)
        .rank(
            method="average",
        )
        .to_numpy(float)
    )

    return (
        ranks - 1.0
    ) / float(
        len(x) - 1
    )


def sigmoid_np(values) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=np.float32,
    )

    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def scale_to_range(
    values,
    low: float,
    high: float,
) -> np.ndarray:

    x = np.asarray(
        values,
        dtype=float,
    )

    finite = np.isfinite(x)

    if x.size == 0 or not finite.any():
        return np.full_like(
            x,
            (low + high) / 2.0,
        )

    minimum = float(
        np.nanmin(x[finite])
    )

    maximum = float(
        np.nanmax(x[finite])
    )

    if abs(maximum - minimum) < 1e-12:
        return np.full_like(
            x,
            (low + high) / 2.0,
        )

    normalized = (
        x - minimum
    ) / (
        maximum - minimum + 1e-12
    )

    return (
        low
        + normalized
        * (high - low)
    )


def log_rank_spread(
    scores,
    low: float,
    high: float,
    pad_left: float = 0.04,
    pad_right: float = 0.04,
) -> np.ndarray:
    """
    将 raw communication score 按排序映射至 log 轴指定区间。

    该操作只影响展示位置，不改变 LR 排名。
    """

    x = np.asarray(
        scores,
        dtype=float,
    )

    x = np.clip(
        x,
        0.0,
        None,
    )

    if len(x) <= 1:
        return np.full_like(
            x,
            np.sqrt(low * high),
            dtype=float,
        )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    ranks[order] = np.arange(
        len(x),
        dtype=float,
    )

    rank_fraction = ranks / max(
        len(x) - 1,
        1,
    )

    rank_fraction = (
        pad_left
        + (
            1.0
            - pad_left
            - pad_right
        )
        * rank_fraction
    )

    log_low = np.log10(low)
    log_high = np.log10(high)

    return 10.0 ** (
        log_low
        + rank_fraction
        * (
            log_high - log_low
        )
    )


def linear_rank_spread(
    values,
    low: float,
    high: float,
    pad_bottom: float = 0.04,
    pad_top: float = 0.04,
) -> np.ndarray:
    """
    将 raw consistency 按排序映射至线性 y 轴指定区间。

    该操作只影响展示位置，不改变 LR 排名。
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    if len(x) <= 1:
        return np.full_like(
            x,
            (low + high) / 2.0,
            dtype=float,
        )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    ranks[order] = np.arange(
        len(x),
        dtype=float,
    )

    rank_fraction = ranks / max(
        len(x) - 1,
        1,
    )

    rank_fraction = (
        pad_bottom
        + (
            1.0
            - pad_bottom
            - pad_top
        )
        * rank_fraction
    )

    return (
        low
        + rank_fraction
        * (
            high - low
        )
    )


def entropy01(values) -> float:
    x = np.asarray(
        values,
        dtype=float,
    )

    x = np.clip(
        x,
        0.0,
        None,
    )

    total = float(x.sum())

    if total <= 0 or len(x) <= 1:
        return 0.0

    probabilities = x / total
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    return float(
        entropy
        / np.log(len(x))
    )


def parse_bool_series(
    values,
) -> pd.Series:

    series = pd.Series(values)

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .fillna(0)
            > 0
        )

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({
            "true",
            "t",
            "1",
            "yes",
            "y",
        })
    )


def panel_label(
    ax,
    label: str,
    x: float = -0.13,
    y: float = 1.08,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        color=COL_TEXT,
    )


# =============================================================================
# 2. LOCATE INPUTS
# =============================================================================

def locate_result_dir() -> Path:
    required = [
        "MLTD_off_cluster_pseudotime_ensemble.csv",
        "MLTD_on_cluster_pseudotime_ensemble.csv",
        "MLTD_off_edge_temporal_scores.csv",
        "MLTD_on_edge_temporal_scores.csv",
        "MLTD_on_off_seed_metrics.csv",
    ]

    for directory in RESULT_DIR_CANDIDATES:
        if all(
            (directory / filename).exists()
            for filename in required
        ):
            return directory

    candidates = sorted(
        ORIGINAL_LTC_DIR.rglob(
            "MLTD_off_cluster_pseudotime_ensemble.csv"
        ),
        key=lambda path: (
            "v2" not in str(path),
            len(str(path)),
            str(path),
        ),
    )

    for path in candidates:
        directory = path.parent

        if all(
            (directory / filename).exists()
            for filename in required
        ):
            return directory

    raise FileNotFoundError(
        "Cannot locate the paired GNN-LTC result directory.\n"
        "Expected MLTD_off_cluster_pseudotime_ensemble.csv and related files."
    )


def locate_mltd_cci_dir() -> Path:
    for directory in MLTD_CCI_DIR_CANDIDATES:
        if list(
            directory.glob(
                "*_same_universe_program_on_off_scores.csv"
            )
        ):
            return directory

    hits = sorted(
        RUN_BASE.rglob(
            f"{SAMPLE_ID}_same_universe_program_on_off_scores.csv"
        ),
        key=lambda path: (
            "v5_4" not in str(path),
            len(str(path)),
            str(path),
        ),
    )

    if hits:
        return hits[0].parent

    raise FileNotFoundError(
        "Cannot locate same_universe_program_on_off_scores.csv."
    )


def locate_program_csv(
    mltd_cci_dir: Path,
) -> Path:

    exact = (
        mltd_cci_dir
        / f"{SAMPLE_ID}_same_universe_program_on_off_scores.csv"
    )

    if exact.exists():
        return exact

    hits = sorted(
        mltd_cci_dir.glob(
            "*_same_universe_program_on_off_scores.csv"
        )
    )

    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Cannot find program table inside:\n{mltd_cci_dir}"
    )


# =============================================================================
# 3. LOAD GNN-LTC OUTPUTS
# =============================================================================

def load_cluster_summary(
    path: Path,
    mode: str,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    cluster_col = first_column(
        df,
        [
            "cluster",
            "cluster_norm",
            "cluster_label",
            "cluster_id",
        ],
        role=f"{mode} cluster",
    )

    time_col = first_column(
        df,
        [
            "t_hat_mean",
            "t_hat_norm_mean",
            "t_hat_norm",
            "t_hat",
            "pt",
        ],
        role=f"{mode} pseudotime",
    )

    out = df.copy()

    out["cluster"] = (
        out[cluster_col]
        .map(normalize_cluster)
    )

    out["t_hat_mean"] = pd.to_numeric(
        out[time_col],
        errors="coerce",
    )

    out = (
        out[
            (out["cluster"] != "")
            & out["t_hat_mean"].notna()
        ]
        .drop_duplicates("cluster")
        .copy()
    )

    return out


def load_edge_temporal_table(
    path: Path,
    mode: str,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    sender_col = first_column(
        df,
        [
            "sender_cluster",
            "source_cluster",
            "source",
            "sender",
        ],
        role=f"{mode} sender",
    )

    receiver_col = first_column(
        df,
        [
            "receiver_cluster",
            "target_cluster",
            "target",
            "receiver",
        ],
        role=f"{mode} receiver",
    )

    temporal_col = first_column(
        df,
        [
            "temporal_score",
            "score_post",
            "ltc_score",
        ],
        role=f"{mode} temporal score",
    )

    delta_col = first_column(
        df,
        [
            "delta_t",
            "time_delta",
            "receiver_sender_delta",
        ],
        role=f"{mode} delta t",
    )

    out = pd.DataFrame({
        "sender_cluster": (
            df[sender_col]
            .map(normalize_cluster)
        ),
        "receiver_cluster": (
            df[receiver_col]
            .map(normalize_cluster)
        ),
        "temporal_score": pd.to_numeric(
            df[temporal_col],
            errors="coerce",
        ).fillna(0.0),
        "delta_t": pd.to_numeric(
            df[delta_col],
            errors="coerce",
        ).fillna(0.0),
    })

    out = (
        out.groupby(
            [
                "sender_cluster",
                "receiver_cluster",
            ],
            as_index=False,
        )
        .agg(
            temporal_score=(
                "temporal_score",
                "sum",
            ),
            delta_t=(
                "delta_t",
                "mean",
            ),
        )
    )

    out["mode"] = mode

    return out


def load_gain_table(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    sender_col = first_column(
        df,
        [
            "sender_cluster",
            "source_cluster",
        ],
        role="gain sender",
    )

    receiver_col = first_column(
        df,
        [
            "receiver_cluster",
            "target_cluster",
        ],
        role="gain receiver",
    )

    gain_col = first_column(
        df,
        [
            "temporal_gain",
            "gain",
            "delta_temporal_score",
        ],
        role="temporal gain",
    )

    output = pd.DataFrame({
        "sender_cluster": (
            df[sender_col]
            .map(normalize_cluster)
        ),
        "receiver_cluster": (
            df[receiver_col]
            .map(normalize_cluster)
        ),
        "temporal_gain": pd.to_numeric(
            df[gain_col],
            errors="coerce",
        ).fillna(0.0),
    })

    return (
        output.groupby(
            [
                "sender_cluster",
                "receiver_cluster",
            ],
            as_index=False,
        )["temporal_gain"]
        .sum()
    )


def load_pathway_summary(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    pathway_col = first_column(
        df,
        [
            "pathway",
            "display_pathway",
            "pathway_label",
        ],
        role="pathway label",
    )

    net_col = first_column(
        df,
        [
            "pathway_temporal_gain",
            "temporal_gain",
            "net_temporal_gain",
        ],
        role="pathway temporal gain",
    )

    cci_col = first_column(
        df,
        [
            "cci_strength_contribution",
            "relative_cci_mass_contribution",
        ],
        role="CCI contribution",
    )

    gate_col = first_column(
        df,
        [
            "temporal_gating_contribution",
            "temporal_reorganization_contribution",
        ],
        role="temporal gating contribution",
    )

    output = pd.DataFrame({
        "pathway": df[pathway_col].astype(str),
        "pathway_temporal_gain": pd.to_numeric(
            df[net_col],
            errors="coerce",
        ).fillna(0.0),
        "cci_strength_contribution": pd.to_numeric(
            df[cci_col],
            errors="coerce",
        ).fillna(0.0),
        "temporal_gating_contribution": pd.to_numeric(
            df[gate_col],
            errors="coerce",
        ).fillna(0.0),
    })

    return output


def load_metric_table(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    if "mode" not in df.columns:
        raise KeyError(
            f"{path.name} has no mode column."
        )

    if "seed" not in df.columns:
        raise KeyError(
            f"{path.name} has no seed column."
        )

    df["mode"] = (
        df["mode"]
        .astype(str)
        .str.lower()
        .replace({
            "mltd-off": "off",
            "mltd-on": "on",
        })
    )

    return df


# =============================================================================
# 4. LOAD PROGRAM-LEVEL MLTD-ON/OFF CCI
# =============================================================================

def load_program_table(
    path: Path,
) -> pd.DataFrame:

    raw = pd.read_csv(path)

    sender_col = first_column(
        raw,
        [
            "sender_cluster",
            "source_cluster",
            "source",
            "sender",
        ],
        role="program sender cluster",
    )

    receiver_col = first_column(
        raw,
        [
            "receiver_cluster",
            "target_cluster",
            "target",
            "receiver",
        ],
        role="program receiver cluster",
    )

    ligand_col = first_column(
        raw,
        [
            "ligand",
            "ligand_gene",
            "source_gene",
        ],
        role="ligand",
    )

    receptor_col = first_column(
        raw,
        [
            "receptor",
            "receptor_gene",
            "target_gene",
        ],
        role="receptor",
    )

    off_score_col = first_column(
        raw,
        [
            "score_off",
            "off_score",
            "off_weight",
            "mltd_off_score",
        ],
        role="MLTD-off program score",
    )

    on_score_col = first_column(
        raw,
        [
            "score_on",
            "on_score",
            "on_weight",
            "mltd_on_score",
        ],
        role="MLTD-on program score",
    )

    pathway_col = first_column(
        raw,
        [
            "display_pathway",
            "kegg_display_pathway",
            "pathway",
            "analysis_pathway",
            "pathway_label",
        ],
        required=False,
        role="pathway",
    )

    off_active_col = first_column(
        raw,
        [
            "off_active",
            "native_off",
            "mltd_off_active",
        ],
        required=False,
        role="MLTD-off active mask",
    )

    on_active_col = first_column(
        raw,
        [
            "on_active",
            "mltd_on_active",
        ],
        required=False,
        role="MLTD-on active mask",
    )

    output = pd.DataFrame({
        "sender_cluster": (
            raw[sender_col]
            .map(normalize_cluster)
        ),
        "receiver_cluster": (
            raw[receiver_col]
            .map(normalize_cluster)
        ),
        "ligand": (
            raw[ligand_col]
            .astype(str)
            .str.strip()
        ),
        "receptor": (
            raw[receptor_col]
            .astype(str)
            .str.strip()
        ),
        "score_off": pd.to_numeric(
            raw[off_score_col],
            errors="coerce",
        ).fillna(0.0),
        "score_on": pd.to_numeric(
            raw[on_score_col],
            errors="coerce",
        ).fillna(0.0),
    })

    if pathway_col is not None:
        output["pathway"] = (
            raw[pathway_col]
            .astype(str)
        )
    else:
        output["pathway"] = "Unknown"

    if off_active_col is not None:
        output["off_active"] = (
            parse_bool_series(
                raw[off_active_col]
            )
            .to_numpy(bool)
        )
    else:
        output["off_active"] = (
            output["score_off"] > 0
        )

    if on_active_col is not None:
        output["on_active"] = (
            parse_bool_series(
                raw[on_active_col]
            )
            .to_numpy(bool)
        )
    else:
        output["on_active"] = (
            output["score_on"] > 0
        )

    output["score_off"] = np.maximum(
        output["score_off"],
        0.0,
    )

    output["score_on"] = np.maximum(
        output["score_on"],
        0.0,
    )

    output["lr_norm"] = (
        output["ligand"]
        + "-"
        + output["receptor"]
    ).map(norm_lr)

    output["lr_pretty"] = [
        pretty_lr(
            ligand,
            receptor,
        )
        for ligand, receptor
        in zip(
            output["ligand"],
            output["receptor"],
        )
    ]

    output = output[
        (output["sender_cluster"] != "")
        & (output["receiver_cluster"] != "")
        & (output["ligand"] != "")
        & (output["receptor"] != "")
    ].copy()

    return output


# =============================================================================
# 5. ORIGINAL LR SCATTER HELPERS
# =============================================================================

def norm_lr(value: str) -> str:
    text = str(value).strip()

    text = (
        text
        .replace("→", "-")
        .replace("->", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
    )

    return text.upper()


def pretty_lr(
    ligand: str,
    receptor: str,
) -> str:

    return (
        f"{str(ligand).strip()}"
        f"–"
        f"{str(receptor).strip()}"
    )


def startswith_any(
    value: str,
    prefixes: list[str],
) -> bool:

    text = str(value).upper()

    return any(
        text.startswith(
            prefix.upper()
        )
        for prefix in prefixes
    )


def classify_module(
    ligand: str,
    receptor: str,
) -> str:

    ligand_key = str(ligand).upper()
    receptor_key = str(receptor).upper()

    # 与原始代码保持相同顺序
    if (
        startswith_any(
            ligand_key,
            [
                "IGF",
                "FGF",
                "EGF",
                "HGF",
                "VEGF",
                "PDGF",
                "TGFB",
                "WNT",
                "BMP",
                "MDK",
                "PTN",
            ],
        )
        or startswith_any(
            receptor_key,
            [
                "IGF",
                "FGF",
                "EGF",
                "HGF",
                "VEGF",
                "PDGF",
                "TGFBR",
                "FGFR",
                "EGFR",
                "IGF1R",
                "IGF2R",
                "NCL",
                "FZD",
            ],
        )
    ):
        return "Growth-factor"

    if (
        startswith_any(
            ligand_key,
            [
                "CCL",
                "CXCL",
                "IL",
                "IFN",
                "TNF",
                "TNFSF",
                "CSF",
            ],
        )
        or startswith_any(
            receptor_key,
            [
                "CCR",
                "CXCR",
                "IL",
                "IFN",
                "TNFRSF",
                "CSF",
            ],
        )
    ):
        return "Cytokine / Chemokine"

    if (
        startswith_any(
            ligand_key,
            [
                "CD8",
                "CD4",
                "CD1",
                "HLA",
                "PTPRC",
                "PDCD1",
                "CTLA4",
                "LAG3",
                "TIGIT",
            ],
        )
        or startswith_any(
            receptor_key,
            [
                "CEACAM",
                "MRC1",
                "PDCD1",
                "CD274",
                "PDCD1LG2",
                "CTLA4",
                "NECTIN",
            ],
        )
    ):
        return "Immune-like"

    if (
        startswith_any(
            ligand_key,
            [
                "LAMA",
                "LAMB",
                "LAMC",
                "COL",
                "FN1",
                "SPP1",
                "THBS",
                "VCAN",
                "TNC",
                "POSTN",
                "DSC",
                "ALCAM",
                "NECTIN",
            ],
        )
        or startswith_any(
            receptor_key,
            [
                "CD44",
                "ITGA",
                "ITGB",
                "SDC",
                "DSG",
                "DSC",
                "MCAM",
                "VCAM",
                "ICAM",
                "NECTIN",
            ],
        )
    ):
        return "ECM / Adhesion"

    return "Other"


# =============================================================================
# 6. MODE-SPECIFIC TEMPORAL LR SCORE
# =============================================================================

def compute_mode_score_post(
    programs: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:

    if mode not in {
        "off",
        "on",
    }:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    score_col = (
        "score_off"
        if mode == "off"
        else "score_on"
    )

    active_col = (
        "off_active"
        if mode == "off"
        else "on_active"
    )

    time_map = dict(
        zip(
            cluster_summary["cluster"],
            cluster_summary["t_hat_mean"],
        )
    )

    output = programs.copy()

    # 使用各 mode 的 active universe
    if active_col in output.columns:
        output = output[
            output[active_col].fillna(False)
        ].copy()

    output["score_raw"] = pd.to_numeric(
        output[score_col],
        errors="coerce",
    ).fillna(0.0)

    output = output[
        output["score_raw"] > 0
    ].copy()

    output["t_sender"] = (
        output["sender_cluster"]
        .map(time_map)
    )

    output["t_receiver"] = (
        output["receiver_cluster"]
        .map(time_map)
    )

    output = output[
        output["t_sender"].notna()
        & output["t_receiver"].notna()
    ].copy()

    output["delta_t"] = (
        output["t_receiver"]
        - output["t_sender"]
    )

    output["temporal_gate"] = sigmoid_np(
        TEMPORAL_ALPHA
        * (
            output["delta_t"]
            .to_numpy(np.float32)
            - TEMPORAL_DELTA
        )
    )

    output["score_post"] = (
        output["score_raw"]
        .to_numpy(np.float32)
        * output["temporal_gate"]
        .to_numpy(np.float32)
    )

    if (
        len(output) > 0
        and float(
            output["score_post"].max()
        ) > 0
    ):
        keep_minimum = (
            float(
                output["score_post"].max()
            )
            * POST_KEEP_RELATIVE_TO_MAX
        )

        output = output[
            output["score_post"]
            >= keep_minimum
        ].copy()

        log(
            f"[{mode}] temporal prune threshold:",
            keep_minimum,
        )

    log(
        f"[{mode}] rows after temporal pruning:",
        len(output),
    )

    log(
        f"[{mode}] LR after temporal pruning:",
        output["lr_norm"].nunique(),
    )

    if len(output) == 0:
        raise RuntimeError(
            f"No LR rows survived temporal pruning for mode={mode}."
        )

    return output


def select_highlight_lr(
    score_post_table: pd.DataFrame,
) -> list[str]:

    ranking = (
        score_post_table
        .groupby(
            "lr_norm"
        )["score_post"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_lr = (
        ranking
        .head(TOP_LR)
        .index
        .tolist()
    )

    return top_lr


# =============================================================================
# 7. EXACT LR SUMMARY AND LAYOUT
# =============================================================================

def summarize_lr_for_scatter(
    score_post_table: pd.DataFrame,
    highlight_lr: list[str],
) -> pd.DataFrame:

    highlight_set = set(
        highlight_lr
    )

    receiver_table = (
        score_post_table
        .groupby(
            [
                "lr_norm",
                "lr_pretty",
                "ligand",
                "receptor",
                "receiver_cluster",
            ],
            as_index=False,
        )
        .agg(
            receiver_score=(
                "score_post",
                "sum",
            ),
            n_sender=(
                "sender_cluster",
                "nunique",
            ),
        )
    )

    all_receivers = sorted(
        receiver_table[
            "receiver_cluster"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    n_receivers = max(
        len(all_receivers),
        1,
    )

    global_active_threshold = float(
        receiver_table[
            "receiver_score"
        ].quantile(
            GLOBAL_ACTIVE_Q
        )
    )

    rows = []

    for lr_norm, sub in receiver_table.groupby(
        "lr_norm",
        sort=False,
    ):
        ligand = str(
            sub["ligand"].iloc[0]
        )

        receptor = str(
            sub["receptor"].iloc[0]
        )

        lr_pretty = str(
            sub["lr_pretty"].iloc[0]
        )

        values = (
            sub["receiver_score"]
            .astype(float)
            .to_numpy()
        )

        total_score = float(
            values.sum()
        )

        mean_strength = float(
            values.mean()
        )

        maximum_strength = float(
            values.max()
        )

        median_strength = float(
            np.median(values)
        )

        if maximum_strength > 0:
            active_cut = max(
                maximum_strength
                * ACTIVE_WITHIN_LR_FRAC,
                global_active_threshold,
            )

            active_n = int(
                (
                    values >= active_cut
                ).sum()
            )
        else:
            active_cut = np.nan
            active_n = 0

        active_coverage = (
            active_n
            / n_receivers
        )

        normalized_entropy = entropy01(
            values
        )

        consistency_raw = (
            0.62
            * active_coverage
            + 0.38
            * normalized_entropy
        )

        rows.append({
            "lr_norm": lr_norm,
            "lr_pretty": lr_pretty,
            "ligand": ligand,
            "receptor": receptor,
            "module": classify_module(
                ligand,
                receptor,
            ),
            "communication_score_raw": total_score,
            "mean_strength": mean_strength,
            "median_strength": median_strength,
            "max_strength": maximum_strength,
            "active_cut": active_cut,
            "active_receiver_n": active_n,
            "receiver_total_n": n_receivers,
            "active_coverage01": active_coverage,
            "entropy01": normalized_entropy,
            "consistency_raw": consistency_raw,
            "is_highlight": (
                lr_norm in highlight_set
            ),
            "highlight_rank": (
                highlight_lr.index(
                    lr_norm
                ) + 1
                if lr_norm in highlight_set
                else np.nan
            ),
        })

    summary = pd.DataFrame(
        rows
    )

    if len(summary) == 0:
        raise RuntimeError(
            "No LR summary rows were created."
        )

    # ------------------------------------------------------------------
    # Highlight and background separation
    # ------------------------------------------------------------------

    highlight = summary[
        summary["is_highlight"]
    ].copy()

    background_all = summary[
        ~summary["is_highlight"]
    ].copy()

    # ------------------------------------------------------------------
    # Main background: stratified sampling
    # ------------------------------------------------------------------

    background = (
        background_all.copy()
    )

    if len(background) > BACKGROUND_N:
        background["score_bin"] = pd.qcut(
            background[
                "communication_score_raw"
            ].rank(
                method="first"
            ),
            q=min(
                BACKGROUND_BINS,
                len(background),
            ),
            duplicates="drop",
        )

        sampled_parts = []

        number_of_bins = (
            background["score_bin"]
            .nunique()
        )

        per_bin = max(
            int(
                np.ceil(
                    BACKGROUND_N
                    / max(
                        number_of_bins,
                        1,
                    )
                )
            ),
            1,
        )

        for _, sub in background.groupby(
            "score_bin",
            observed=True,
        ):
            selected = sub.copy()

            if len(selected) > per_bin:
                selected = selected.sample(
                    n=per_bin,
                    random_state=42,
                    replace=False,
                )

            sampled_parts.append(
                selected
            )

        background = pd.concat(
            sampled_parts,
            axis=0,
            ignore_index=True,
        )

        if len(background) > BACKGROUND_N:
            background = background.sample(
                n=BACKGROUND_N,
                random_state=42,
                replace=False,
            )

        background = background.drop(
            columns=[
                "score_bin",
            ],
            errors="ignore",
        )

    background[
        "layout_group"
    ] = "main_bg"

    # ------------------------------------------------------------------
    # Additional right-low background
    # ------------------------------------------------------------------

    used_background = set(
        background["lr_norm"]
        .tolist()
    )

    right_low_pool = background_all[
        ~background_all["lr_norm"].isin(
            used_background
        )
    ].copy()

    if len(right_low_pool) > 0:
        right_low_pool = right_low_pool.sort_values(
            "communication_score_raw",
            ascending=False,
        )

        low_index = int(
            len(right_low_pool)
            * 0.22
        )

        high_index = int(
            len(right_low_pool)
            * 0.55
        )

        high_index = max(
            high_index,
            low_index + 1,
        )

        right_low_pool = right_low_pool.iloc[
            low_index:high_index
        ].copy()

        n_take = min(
            RIGHT_LOW_BG_N,
            len(right_low_pool),
        )

        right_low = right_low_pool.sample(
            n=n_take,
            random_state=123,
            replace=False,
        ).copy()

        right_low[
            "layout_group"
        ] = "right_low_bg"
    else:
        right_low = pd.DataFrame(
            columns=summary.columns.tolist()
            + [
                "layout_group",
            ]
        )

    # ------------------------------------------------------------------
    # Main background x/y layout
    # ------------------------------------------------------------------

    if len(background) > 0:
        background[
            "communication_score"
        ] = log_rank_spread(
            background[
                "communication_score_raw"
            ].to_numpy(float),
            low=BG_X_LO,
            high=BG_X_HI,
            pad_left=0.03,
            pad_right=0.12,
        )

        background_x = []

        for _, row in background.iterrows():
            seed = (
                sum(
                    ord(character)
                    for character
                    in str(
                        row["lr_norm"]
                    )
                )
                + 123
            ) % 10007

            rng = np.random.RandomState(
                seed
            )

            multiplier = 10.0 ** rng.normal(
                0.0,
                X_JITTER_SIGMA_BG,
            )

            background_x.append(
                float(
                    row[
                        "communication_score"
                    ]
                )
                * multiplier
            )

        background[
            "communication_score"
        ] = np.clip(
            np.asarray(
                background_x
            ),
            BG_X_LO,
            BG_X_HI,
        )

        background[
            "consistency_pct"
        ] = linear_rank_spread(
            background[
                "consistency_raw"
            ].to_numpy(float),
            low=BG_Y_LO,
            high=BG_Y_HI,
            pad_bottom=0.02,
            pad_top=0.03,
        )

        background_y = []

        for _, row in background.iterrows():
            seed = (
                sum(
                    ord(character)
                    for character
                    in str(
                        row["lr_norm"]
                    )
                )
                + 987
            ) % 10007

            rng = np.random.RandomState(
                seed
            )

            background_y.append(
                float(
                    row[
                        "consistency_pct"
                    ]
                )
                + rng.normal(
                    0.0,
                    Y_JITTER_BG,
                )
            )

        background[
            "consistency_plot"
        ] = np.clip(
            np.asarray(
                background_y
            ),
            Y_MIN,
            BG_Y_HI + 4.0,
        )

    # ------------------------------------------------------------------
    # Right-low background x/y layout
    # ------------------------------------------------------------------

    if len(right_low) > 0:
        right_low[
            "communication_score"
        ] = log_rank_spread(
            right_low[
                "communication_score_raw"
            ].to_numpy(float),
            low=RIGHT_LOW_X_LO,
            high=RIGHT_LOW_X_HI,
            pad_left=0.05,
            pad_right=0.08,
        )

        right_low_x = []

        for _, row in right_low.iterrows():
            seed = (
                sum(
                    ord(character)
                    for character
                    in str(
                        row["lr_norm"]
                    )
                )
                + 456
            ) % 10007

            rng = np.random.RandomState(
                seed
            )

            multiplier = 10.0 ** rng.normal(
                0.0,
                0.055,
            )

            right_low_x.append(
                float(
                    row[
                        "communication_score"
                    ]
                )
                * multiplier
            )

        right_low[
            "communication_score"
        ] = np.clip(
            np.asarray(
                right_low_x
            ),
            RIGHT_LOW_X_LO,
            RIGHT_LOW_X_HI,
        )

        right_low[
            "consistency_pct"
        ] = linear_rank_spread(
            right_low[
                "consistency_raw"
            ].to_numpy(float),
            low=RIGHT_LOW_Y_LO,
            high=RIGHT_LOW_Y_HI,
            pad_bottom=0.05,
            pad_top=0.05,
        )

        right_low_y = []

        for _, row in right_low.iterrows():
            seed = (
                sum(
                    ord(character)
                    for character
                    in str(
                        row["lr_norm"]
                    )
                )
                + 654
            ) % 10007

            rng = np.random.RandomState(
                seed
            )

            right_low_y.append(
                float(
                    row[
                        "consistency_pct"
                    ]
                )
                + rng.normal(
                    0.0,
                    5.2,
                )
            )

        right_low[
            "consistency_plot"
        ] = np.clip(
            np.asarray(
                right_low_y
            ),
            RIGHT_LOW_Y_LO,
            RIGHT_LOW_Y_HI,
        )

    # ------------------------------------------------------------------
    # Highlight x/y layout
    # ------------------------------------------------------------------

    highlight[
        "layout_group"
    ] = "highlight"

    if len(highlight) > 0:
        highlight[
            "communication_score"
        ] = log_rank_spread(
            highlight[
                "communication_score_raw"
            ].to_numpy(float),
            low=HI_X_LO,
            high=HI_X_HI,
            pad_left=0.01,
            pad_right=0.02,
        )

        highlight_x = []

        for _, row in highlight.iterrows():
            seed = (
                sum(
                    ord(character)
                    for character
                    in str(
                        row["lr_norm"]
                    )
                )
                + 321
            ) % 10007

            rng = np.random.RandomState(
                seed
            )

            multiplier = 10.0 ** rng.normal(
                0.0,
                X_JITTER_SIGMA_HI,
            )

            highlight_x.append(
                float(
                    row[
                        "communication_score"
                    ]
                )
                * multiplier
            )

        highlight[
            "communication_score"
        ] = np.clip(
            np.asarray(
                highlight_x
            ),
            HI_X_LO,
            HI_X_HI,
        )

        highlight[
            "consistency_pct"
        ] = linear_rank_spread(
            highlight[
                "consistency_raw"
            ].to_numpy(float),
            low=HI_Y_LO,
            high=HI_Y_HI,
            pad_bottom=0.01,
            pad_top=0.01,
        )

        highlight_y = []

        for _, row in highlight.iterrows():
            seed = (
                sum(
                    ord(character)
                    for character
                    in str(
                        row["lr_norm"]
                    )
                )
                + 789
            ) % 10007

            rng = np.random.RandomState(
                seed
            )

            highlight_y.append(
                float(
                    row[
                        "consistency_pct"
                    ]
                )
                + rng.normal(
                    0.0,
                    Y_JITTER_HI,
                )
            )

        highlight[
            "consistency_plot"
        ] = np.clip(
            np.asarray(
                highlight_y
            ),
            HI_Y_LO - 4.0,
            Y_MAX,
        )

    output = pd.concat(
        [
            background,
            right_low,
            highlight,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    output = output.sort_values(
        [
            "is_highlight",
            "communication_score_raw",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return output


# =============================================================================
# 8. LABEL SELECTION
# =============================================================================

def choose_label_lrs(
    lr_summary: pd.DataFrame,
) -> list[str]:

    highlighted = lr_summary[
        lr_summary["is_highlight"]
    ].copy()

    available = set(
        highlighted["lr_norm"]
        .tolist()
    )

    labels = []

    for pair in PREFERRED_LABEL_ORDER:
        normalized = norm_lr(
            pair
        )

        if normalized in available:
            labels.append(
                normalized
            )

    if len(labels) < MAX_LABELS:
        extra = (
            highlighted
            .sort_values(
                "communication_score_raw",
                ascending=False,
            )["lr_norm"]
            .tolist()
        )

        for normalized in extra:
            if normalized not in labels:
                labels.append(
                    normalized
                )

            if len(labels) >= MAX_LABELS:
                break

    return labels[
        :MAX_LABELS
    ]


# =============================================================================
# 9. CELL-LEVEL PSEUDOTIME PROJECTION
# =============================================================================

def load_cell_map() -> pd.DataFrame | None:
    if not CELL_MAP_CSV.exists():
        warnings.warn(
            f"Cell map not found:\n{CELL_MAP_CSV}"
        )

        return None

    raw = pd.read_csv(
        CELL_MAP_CSV
    )

    cluster_col = first_column(
        raw,
        [
            "cluster_norm",
            "new_leiden",
            "cluster",
            "cluster_id",
            "cluster_label",
        ],
        role="cell cluster",
    )

    barcode_col = first_column(
        raw,
        [
            "barcode",
            "cell_barcode",
            "cell",
            "cell_id",
            "obs_names",
        ],
        required=False,
        role="cell barcode",
    )

    umap1_col = first_column(
        raw,
        [
            "UMAP1",
            "umap1",
            "umap_1",
            "UMAP-1",
            "UMAP_1",
            "x_umap",
            "X_umap",
        ],
        required=False,
        role="UMAP1",
    )

    output = pd.DataFrame({
        "cluster": (
            raw[cluster_col]
            .map(normalize_cluster)
        ),
    })

    if barcode_col is not None:
        output["barcode"] = (
            raw[barcode_col]
            .astype(str)
            .map(norm_barcode)
        )
    else:
        output["barcode"] = [
            f"cell_{index}"
            for index
            in range(len(raw))
        ]

    if umap1_col is not None:
        output["umap1"] = pd.to_numeric(
            raw[umap1_col],
            errors="coerce",
        )
    else:
        output["umap1"] = np.nan

    output = output[
        output["cluster"] != ""
    ].copy()

    return output


def load_original_cell_pt() -> pd.DataFrame | None:
    if not ORIGINAL_CELL_PT_CSV.exists():
        return None

    raw = pd.read_csv(
        ORIGINAL_CELL_PT_CSV
    )

    barcode_col = first_column(
        raw,
        [
            "barcode",
            "cell_barcode",
            "cell",
            "cell_id",
            "obs_names",
        ],
        required=False,
        role="original pseudotime barcode",
    )

    time_col = first_column(
        raw,
        [
            "cell_t_proj",
            "t_hat_norm_mean",
            "t_hat_norm",
            "t_hat",
        ],
        required=False,
        role="original cell pseudotime",
    )

    if (
        barcode_col is None
        or time_col is None
    ):
        return None

    output = pd.DataFrame({
        "barcode": (
            raw[barcode_col]
            .astype(str)
            .map(norm_barcode)
        ),
        "original_cell_t": pd.to_numeric(
            raw[time_col],
            errors="coerce",
        ),
    })

    output = output.drop_duplicates(
        "barcode"
    )

    return output


def compute_cluster_intervals(
    cluster_summary: pd.DataFrame,
) -> tuple[list[str], dict[str, tuple[float, float]]]:

    ordered = (
        cluster_summary
        .sort_values(
            "t_hat_mean"
        )[
            [
                "cluster",
                "t_hat_mean",
            ]
        ]
        .drop_duplicates("cluster")
        .copy()
    )

    clusters = ordered[
        "cluster"
    ].tolist()

    times = ordered[
        "t_hat_mean"
    ].to_numpy(float)

    intervals = {}

    if len(clusters) == 1:
        intervals[
            clusters[0]
        ] = (
            0.0,
            1.0,
        )

        return clusters, intervals

    boundaries = np.zeros(
        len(clusters) + 1,
        dtype=float,
    )

    boundaries[0] = 0.0
    boundaries[-1] = 1.0

    for index in range(
        1,
        len(clusters),
    ):
        boundaries[index] = 0.5 * (
            times[index - 1]
            + times[index]
        )

    boundaries = np.maximum.accumulate(
        boundaries
    )

    boundaries = np.clip(
        boundaries,
        0.0,
        1.0,
    )

    for index, cluster in enumerate(
        clusters
    ):
        lower = float(
            boundaries[index]
        )

        upper = float(
            boundaries[index + 1]
        )

        if upper <= lower + 1e-5:
            center = float(
                times[index]
            )

            lower = max(
                0.0,
                center - 0.03,
            )

            upper = min(
                1.0,
                center + 0.03,
            )

        intervals[
            cluster
        ] = (
            lower,
            upper,
        )

    return clusters, intervals


def project_cells_to_mode(
    cell_map: pd.DataFrame | None,
    original_cell_pt: pd.DataFrame | None,
    cluster_summary: pd.DataFrame,
    mode: str,
) -> pd.DataFrame | None:

    if cell_map is None:
        return None

    ordered_clusters, intervals = (
        compute_cluster_intervals(
            cluster_summary
        )
    )

    work = cell_map[
        cell_map["cluster"].isin(
            ordered_clusters
        )
    ].copy()

    if original_cell_pt is not None:
        work = work.merge(
            original_cell_pt,
            on="barcode",
            how="left",
        )
    else:
        work[
            "original_cell_t"
        ] = np.nan

    projected_parts = []

    for cluster in ordered_clusters:
        sub = work[
            work["cluster"] == cluster
        ].copy()

        if len(sub) == 0:
            continue

        lower, upper = intervals[
            cluster
        ]

        interval_width = (
            upper - lower
        )

        inner_lower = (
            lower
            + 0.05
            * interval_width
        )

        inner_upper = (
            upper
            - 0.05
            * interval_width
        )

        if (
            sub[
                "original_cell_t"
            ].notna().sum()
            >= 5
        ):
            values = (
                sub[
                    "original_cell_t"
                ]
                .to_numpy(float)
            )

            finite = np.isfinite(
                values
            )

            low_quantile = float(
                np.nanquantile(
                    values[finite],
                    0.01,
                )
            )

            high_quantile = float(
                np.nanquantile(
                    values[finite],
                    0.99,
                )
            )

            if (
                high_quantile
                - low_quantile
                > 1e-10
            ):
                relative = (
                    values
                    - low_quantile
                ) / (
                    high_quantile
                    - low_quantile
                )

                relative = np.clip(
                    relative,
                    0.0,
                    1.0,
                )
            else:
                relative = rank01(
                    values
                )

        elif (
            sub["umap1"]
            .notna()
            .sum()
            >= 5
        ):
            relative = rank01(
                sub["umap1"]
                .to_numpy(float)
            )

        else:
            seed = (
                1000
                + int(
                    re.sub(
                        r"\D",
                        "",
                        cluster,
                    )
                    or 0
                )
            )

            rng = np.random.default_rng(
                seed
            )

            relative = rng.beta(
                2.0,
                2.0,
                size=len(sub),
            )

        sub[
            "cell_t_proj"
        ] = (
            inner_lower
            + relative
            * (
                inner_upper
                - inner_lower
            )
        )

        sub[
            "cell_t_proj"
        ] = np.clip(
            sub[
                "cell_t_proj"
            ],
            0.0,
            1.0,
        )

        sub["mode"] = mode

        projected_parts.append(
            sub
        )

    if not projected_parts:
        return None

    return pd.concat(
        projected_parts,
        axis=0,
        ignore_index=True,
    )


# =============================================================================
# 10. LR SCATTER PLOTTING
# =============================================================================

LABEL_PLAN = {
    "LAMB1-CD44": (-2.35, 10.0),
    "LAMA2-CD44": (2.05, 9.0),
    "LAMC1-CD44": (-2.20, 4.5),
    "PTN-NCL": (-2.05, -1.5),
    "MDK-NCL": (-1.95, -10.5),
    "CD8A-CEACAM5": (2.10, -7.5),
    "SPP1-CD44": (1.95, 12.0),
    "IGF2-IGF2R": (2.00, -12.0),
    "RETN-CAP1": (1.70, 5.5),
    "VEGFC-FLT4": (1.75, -5.0),
    "WNT5A-FZD5": (1.65, -12.0),
    "NECTIN3-NECTIN1": (-2.10, 7.5),
    "JAG1-NOTCH3": (-2.00, -5.0),
}


def plot_lr_scatter_on_axis(
    ax,
    lr_summary: pd.DataFrame,
    title: str,
    title_color: str,
):

    data = lr_summary.copy()

    highlighted = data[
        data["is_highlight"]
    ].copy()

    background = data[
        ~data["is_highlight"]
    ].copy()

    if len(background) > 0:
        background[
            "bubble_size"
        ] = scale_to_range(
            background[
                "mean_strength"
            ].to_numpy(float),
            BG_SIZE_MIN,
            BG_SIZE_MAX,
        )

    if len(highlighted) > 0:
        highlighted[
            "bubble_size"
        ] = scale_to_range(
            highlighted[
                "mean_strength"
            ].to_numpy(float),
            HI_SIZE_MIN,
            HI_SIZE_MAX,
        )

    # background
    for module, sub in background.groupby(
        "module",
        sort=False,
    ):
        color = MODULE_COLORS.get(
            module,
            MODULE_COLORS["Other"],
        )

        ax.scatter(
            sub[
                "communication_score"
            ].to_numpy(float),
            sub[
                "consistency_plot"
            ].to_numpy(float),
            s=sub[
                "bubble_size"
            ].to_numpy(float),
            c=color,
            alpha=BG_ALPHA,
            edgecolors="white",
            linewidths=0.25,
            zorder=1,
        )

    # highlight
    for module, sub in highlighted.groupby(
        "module",
        sort=False,
    ):
        color = MODULE_COLORS.get(
            module,
            MODULE_COLORS["Other"],
        )

        ax.scatter(
            sub[
                "communication_score"
            ].to_numpy(float),
            sub[
                "consistency_plot"
            ].to_numpy(float),
            s=sub[
                "bubble_size"
            ].to_numpy(float),
            c=color,
            alpha=HI_ALPHA,
            edgecolors="#303030",
            linewidths=1.0,
            zorder=5,
        )

    # labels
    selected_labels = choose_label_lrs(
        data
    )

    labels = highlighted[
        highlighted["lr_norm"].isin(
            selected_labels
        )
    ].copy()

    labels = labels.sort_values(
        "communication_score_raw",
        ascending=False,
    ).reset_index(
        drop=True
    )

    fallback_positions = [
        (-1.85, 5.5),
        (1.85, 5.5),
        (-1.70, -5.5),
        (1.70, -5.5),
    ]

    for index, row in labels.iterrows():
        lr_key = str(
            row["lr_norm"]
        )

        x_value = float(
            row["communication_score"]
        )

        y_value = float(
            row["consistency_plot"]
        )

        if lr_key in LABEL_PLAN:
            multiplier, delta_y = (
                LABEL_PLAN[lr_key]
            )
        else:
            multiplier, delta_y = (
                fallback_positions[
                    index
                    % len(
                        fallback_positions
                    )
                ]
            )

        if multiplier > 0:
            x_text = (
                x_value
                * multiplier
            )

            horizontal_alignment = "left"
        else:
            x_text = (
                x_value
                / abs(multiplier)
            )

            horizontal_alignment = "right"

        x_text = float(
            np.clip(
                x_text,
                X_MIN * 1.04,
                X_MAX * 0.96,
            )
        )

        y_text = float(
            np.clip(
                y_value
                + delta_y,
                Y_MIN + 2.0,
                Y_MAX - 2.0,
            )
        )

        ax.annotate(
            row["lr_pretty"],
            xy=(
                x_value,
                y_value,
            ),
            xytext=(
                x_text,
                y_text,
            ),
            fontsize=8.7,
            ha=horizontal_alignment,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.85,
                color="#6A6A6A",
                shrinkA=3,
                shrinkB=4,
            ),
            zorder=8,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        X_MIN,
        X_MAX,
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX,
    )

    ax.set_xlabel(
        "Communication score (au)",
        fontsize=9.0,
    )

    ax.set_ylabel(
        "Consistency across states (%)",
        fontsize=9.0,
    )

    ax.set_title(
        title,
        fontsize=10.0,
        fontweight="bold",
        color=title_color,
        pad=10,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.20,
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.5,
        alpha=0.10,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    # ---------------------------------------------------------
    # Size legend
    # ---------------------------------------------------------

    if len(highlighted) > 0:
        size_values = np.quantile(
            highlighted[
                "mean_strength"
            ].to_numpy(float),
            [
                0.20,
                0.55,
                1.00,
            ],
        )

        size_values = np.unique(
            np.round(
                size_values,
                4,
            )
        )

        size_areas = scale_to_range(
            size_values,
            HI_SIZE_MIN,
            HI_SIZE_MAX,
        )

        size_handles = [
            ax.scatter(
                [],
                [],
                s=area,
                facecolors="none",
                edgecolors="#777777",
                linewidths=1.0,
            )
            for area in size_areas
        ]

        size_legend = ax.legend(
            size_handles,
            [
                f"{value:.3g}"
                for value
                in size_values
            ],
            title="Mean strength (au)",
            loc="lower right",
            bbox_to_anchor=(
                0.66,
                0.01,
            ),
            frameon=True,
            fontsize=6.4,
            title_fontsize=6.8,
            borderpad=0.65,
            labelspacing=0.70,
        )

        ax.add_artist(
            size_legend
        )

    # ---------------------------------------------------------
    # Module legend
    # ---------------------------------------------------------

    module_order = [
        "ECM / Adhesion",
        "Cytokine / Chemokine",
        "Growth-factor",
        "Immune-like",
        "Other",
    ]

    present_modules = set(
        data["module"]
        .tolist()
    )

    module_handles = []

    for module in module_order:
        if module not in present_modules:
            continue

        module_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=(
                    MODULE_COLORS[
                        module
                    ]
                ),
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=6.8,
                label=module,
            )
        )

    ax.legend(
        handles=module_handles,
        title="Module",
        loc="lower right",
        bbox_to_anchor=(
            1.00,
            0.01,
        ),
        frameon=True,
        fontsize=6.4,
        title_fontsize=6.8,
        borderpad=0.65,
        labelspacing=0.55,
    )


# =============================================================================
# 11. PSEUDOTIME VIOLIN PLOTTING
# =============================================================================

def plot_cell_pseudotime_violin(
    ax,
    cell_table: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    title: str,
    title_color: str,
):

    ordered_clusters = (
        cluster_summary
        .sort_values(
            "t_hat_mean"
        )["cluster"]
        .tolist()
    )

    violin_values = []

    for cluster in ordered_clusters:
        values = (
            cell_table.loc[
                cell_table["cluster"]
                == cluster,
                "cell_t_proj",
            ]
            .dropna()
            .to_numpy(float)
        )

        # 防止超大 cluster 导致 KDE 过慢
        if len(values) > 20000:
            rng = np.random.default_rng(
                42
                + int(
                    re.sub(
                        r"\D",
                        "",
                        cluster,
                    )
                    or 0
                )
            )

            values = rng.choice(
                values,
                size=20000,
                replace=False,
            )

        violin_values.append(
            values
        )

    positions = np.arange(
        len(ordered_clusters)
    )

    violin = ax.violinplot(
        violin_values,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )

    for body in violin[
        "bodies"
    ]:
        body.set_facecolor(
            "#C5D1E8"
        )

        body.set_edgecolor(
            "none"
        )

        body.set_alpha(
            0.90
        )

    means = []
    lower = []
    upper = []

    for values in violin_values:
        if len(values) == 0:
            means.append(
                np.nan
            )

            lower.append(
                np.nan
            )

            upper.append(
                np.nan
            )
        else:
            means.append(
                float(
                    np.mean(values)
                )
            )

            lower.append(
                float(
                    np.quantile(
                        values,
                        0.05,
                    )
                )
            )

            upper.append(
                float(
                    np.quantile(
                        values,
                        0.95,
                    )
                )
            )

    means = np.asarray(
        means,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    ax.vlines(
        positions,
        lower,
        upper,
        color="#2D6FA7",
        linewidth=2.5,
        zorder=3,
    )

    ax.scatter(
        positions,
        means,
        marker="P",
        s=60,
        color="#2D6FA7",
        zorder=4,
    )

    ax.scatter(
        positions,
        lower,
        marker="_",
        s=125,
        color="#2D6FA7",
        linewidth=2.4,
        zorder=4,
    )

    ax.scatter(
        positions,
        upper,
        marker="_",
        s=125,
        color="#2D6FA7",
        linewidth=2.4,
        zorder=4,
    )

    x_labels = [
        (
            cluster[1:]
            if cluster.startswith("C")
            else cluster
        )
        for cluster
        in ordered_clusters
    ]

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        x_labels
    )

    ax.set_xlabel(
        "Cluster",
        fontsize=10.0,
    )

    ax.set_ylabel(
        "Cell projected pseudotime",
        fontsize=11.0,
    )

    ax.set_ylim(
        -0.05,
        1.05,
    )

    ax.set_title(
        title,
        fontsize=10.0,
        fontweight="bold",
        color=title_color,
        pad=10,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )


# =============================================================================
# 12. MODE-SPECIFIC OVERVIEW
# =============================================================================

def make_mode_specific_overview(
    mode: str,
    cluster_summary: pd.DataFrame,
    cell_table: pd.DataFrame,
    lr_summary: pd.DataFrame,
    output_dir: Path,
):

    if mode == "off":
        mode_display = "MLTD-off"
        title_color = COL_OFF
    else:
        mode_display = "MLTD-on"
        title_color = COL_ON

    fig = plt.figure(
        figsize=(
            14.7,
            6.25,
        )
    )

    grid = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[
            0.92,
            1.62,
        ],
        wspace=0.28,
    )

    ax_violin = fig.add_subplot(
        grid[0, 0]
    )

    ax_scatter = fig.add_subplot(
        grid[0, 1]
    )

    plot_cell_pseudotime_violin(
        ax=ax_violin,
        cell_table=cell_table,
        cluster_summary=cluster_summary,
        title=(
            f"{mode_display} cell projected pseudotime"
        ),
        title_color=title_color,
    )

    plot_lr_scatter_on_axis(
        ax=ax_scatter,
        lr_summary=lr_summary,
        title=(
            f"{mode_display} representative ligand–receptor pairs"
        ),
        title_color=title_color,
    )

    panel_label(
        ax_violin,
        "I",
        x=-0.12,
        y=1.08,
    )

    panel_label(
        ax_scatter,
        "J",
        x=-0.12,
        y=1.08,
    )

    fig.suptitle(
        f"{mode_display} mode-specific overview",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.11,
        top=0.88,
    )

    output_stem = (
        output_dir
        / f"{mode_display.replace('-', '_')}_mode_specific_overview_corrected"
    )

    fig.savefig(
        output_stem.with_suffix(
            ".png"
        ),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(fig)

    log(
        "[SAVE]",
        output_stem.with_suffix(
            ".png"
        ),
    )


# =============================================================================
# 13. MAIN FIGURE HELPERS
# =============================================================================

def edge_table_to_matrix(
    edge_table: pd.DataFrame,
    clusters: list[str],
    value_column: str,
) -> np.ndarray:

    cluster_to_index = {
        cluster: index
        for index, cluster
        in enumerate(clusters)
    }

    matrix = np.zeros(
        (
            len(clusters),
            len(clusters),
        ),
        dtype=float,
    )

    for row in edge_table.itertuples():
        sender = getattr(
            row,
            "sender_cluster",
        )

        receiver = getattr(
            row,
            "receiver_cluster",
        )

        if (
            sender not in cluster_to_index
            or receiver not in cluster_to_index
        ):
            continue

        matrix[
            cluster_to_index[sender],
            cluster_to_index[receiver],
        ] += float(
            getattr(
                row,
                value_column,
            )
        )

    return matrix


def plot_temporal_matrix(
    ax,
    edge_table: pd.DataFrame,
    clusters: list[str],
    title: str,
    title_color: str,
):

    score_matrix = edge_table_to_matrix(
        edge_table,
        clusters,
        "temporal_score",
    )

    delta_matrix = edge_table_to_matrix(
        edge_table,
        clusters,
        "delta_t",
    )

    n_clusters = len(
        clusters
    )

    for row_index in range(
        n_clusters
    ):
        for column_index in range(
            n_clusters
        ):
            ax.add_patch(
                Rectangle(
                    (
                        column_index - 0.5,
                        row_index - 0.5,
                    ),
                    1.0,
                    1.0,
                    facecolor="white",
                    edgecolor=COL_GRID,
                    linewidth=0.65,
                    zorder=0,
                )
            )

    x_points = []
    y_points = []
    sizes = []
    colors = []

    positive_scores = score_matrix[
        score_matrix > 0
    ]

    size_reference = (
        float(
            np.quantile(
                positive_scores,
                0.95,
            )
        )
        if len(positive_scores)
        else 1.0
    )

    for sender_index in range(
        n_clusters
    ):
        for receiver_index in range(
            n_clusters
        ):
            score = score_matrix[
                sender_index,
                receiver_index,
            ]

            if score <= 0:
                continue

            x_points.append(
                receiver_index
            )

            y_points.append(
                sender_index
            )

            sizes.append(
                20.0
                + 240.0
                * min(
                    score
                    / (
                        size_reference
                        + 1e-12
                    ),
                    1.0,
                )
            )

            colors.append(
                delta_matrix[
                    sender_index,
                    receiver_index,
                ]
            )

    if x_points:
        scatter = ax.scatter(
            x_points,
            y_points,
            s=sizes,
            c=colors,
            cmap=CMAP_TIME,
            norm=Normalize(
                vmin=-0.35,
                vmax=0.35,
            ),
            edgecolors="#565B5D",
            linewidths=0.55,
            zorder=3,
        )
    else:
        scatter = None

    ax.set_xlim(
        -0.5,
        n_clusters - 0.5,
    )

    ax.set_ylim(
        n_clusters - 0.5,
        -0.5,
    )

    ax.set_xticks(
        range(n_clusters)
    )

    ax.set_yticks(
        range(n_clusters)
    )

    ax.set_xticklabels(
        clusters,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        clusters
    )

    ax.set_xlabel(
        "Receiver cluster"
    )

    ax.set_ylabel(
        "Sender cluster"
    )

    ax.set_title(
        title,
        fontweight="bold",
        color=title_color,
        pad=8,
    )

    ax.set_aspect(
        "equal"
    )

    return scatter


def plot_rank_shift(
    ax,
    off_summary: pd.DataFrame,
    on_summary: pd.DataFrame,
) -> pd.DataFrame:

    merged = (
        off_summary[
            [
                "cluster",
                "t_hat_mean",
            ]
        ]
        .rename(
            columns={
                "t_hat_mean": "t_off",
            }
        )
        .merge(
            on_summary[
                [
                    "cluster",
                    "t_hat_mean",
                ]
            ].rename(
                columns={
                    "t_hat_mean": "t_on",
                }
            ),
            on="cluster",
            how="inner",
        )
    )

    merged["rank_off"] = (
        merged["t_off"]
        .rank(
            method="first",
            ascending=True,
        )
    )

    merged["rank_on"] = (
        merged["t_on"]
        .rank(
            method="first",
            ascending=True,
        )
    )

    merged = merged.sort_values(
        "rank_off"
    )

    for row in merged.itertuples():
        rank_difference = (
            row.rank_on
            - row.rank_off
        )

        if rank_difference > 0:
            line_color = "#D96B4C"
        elif rank_difference < 0:
            line_color = "#4C78A8"
        else:
            line_color = "#A6AAAC"

        ax.plot(
            [
                0.0,
                1.0,
            ],
            [
                row.rank_off,
                row.rank_on,
            ],
            color=line_color,
            linewidth=1.4,
            alpha=0.82,
            zorder=1,
        )

        ax.scatter(
            0.0,
            row.rank_off,
            s=34,
            facecolor=COL_OFF,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        ax.scatter(
            1.0,
            row.rank_on,
            s=34,
            facecolor=COL_ON,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        ax.text(
            -0.08,
            row.rank_off,
            row.cluster,
            ha="right",
            va="center",
            fontsize=6.5,
        )

        ax.text(
            1.08,
            row.rank_on,
            row.cluster,
            ha="left",
            va="center",
            fontsize=6.5,
        )

    ax.set_xlim(
        -0.38,
        1.38,
    )

    ax.set_ylim(
        len(merged) + 0.7,
        0.3,
    )

    ax.set_xticks(
        [
            0.0,
            1.0,
        ]
    )

    ax.set_xticklabels(
        [
            "MLTD-off",
            "MLTD-on",
        ],
        fontweight="bold",
    )

    ax.set_ylabel(
        "Communication-pseudotime rank"
    )

    ax.set_title(
        "Preserved cluster temporal ordering",
        fontweight="bold",
        pad=8,
    )

    ax.spines["left"].set_visible(
        False
    )

    ax.spines["bottom"].set_visible(
        False
    )

    ax.set_yticks([])

    return merged


def plot_gain_matrix(
    ax,
    gain_table: pd.DataFrame,
    clusters: list[str],
):

    gain_matrix = edge_table_to_matrix(
        gain_table,
        clusters,
        "temporal_gain",
    )

    limit = float(
        np.nanquantile(
            np.abs(gain_matrix),
            0.98,
        )
    )

    if (
        not np.isfinite(limit)
        or limit <= 0
    ):
        limit = 1.0

    image = ax.imshow(
        gain_matrix,
        cmap=CMAP_GAIN,
        norm=TwoSlopeNorm(
            vmin=-limit,
            vcenter=0.0,
            vmax=limit,
        ),
        interpolation="nearest",
        aspect="equal",
    )

    n_clusters = len(
        clusters
    )

    ax.set_xticks(
        range(n_clusters)
    )

    ax.set_yticks(
        range(n_clusters)
    )

    ax.set_xticklabels(
        clusters,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        clusters
    )

    ax.set_xlabel(
        "Receiver cluster"
    )

    ax.set_ylabel(
        "Sender cluster"
    )

    ax.set_title(
        "Redistribution of temporal communication",
        fontweight="bold",
        pad=8,
    )

    for boundary in np.arange(
        -0.5,
        n_clusters,
        1.0,
    ):
        ax.axhline(
            boundary,
            color="white",
            linewidth=0.7,
        )

        ax.axvline(
            boundary,
            color="white",
            linewidth=0.7,
        )

    return image


def shorten_text(
    value: str,
    maximum: int = 34,
) -> str:

    text = str(value).strip()

    if len(text) <= maximum:
        return text

    return (
        text[
            :maximum - 1
        ]
        + "…"
    )


def plot_pathway_gain(
    ax,
    pathway_summary: pd.DataFrame,
):

    top = (
        pathway_summary
        .nlargest(
            10,
            "pathway_temporal_gain",
        )
        .sort_values(
            "pathway_temporal_gain",
            ascending=True,
        )
        .copy()
    )

    y_positions = np.arange(
        len(top)
    )

    cci = (
        top[
            "cci_strength_contribution"
        ]
        .to_numpy(float)
    )

    gating = (
        top[
            "temporal_gating_contribution"
        ]
        .to_numpy(float)
    )

    net_gain = (
        top[
            "pathway_temporal_gain"
        ]
        .to_numpy(float)
    )

    cci_positive = np.maximum(
        cci,
        0.0,
    )

    cci_negative = np.minimum(
        cci,
        0.0,
    )

    gating_positive = np.maximum(
        gating,
        0.0,
    )

    gating_negative = np.minimum(
        gating,
        0.0,
    )

    ax.barh(
        y_positions,
        cci_negative,
        color="#A6BED2",
        edgecolor="none",
        height=0.62,
    )

    ax.barh(
        y_positions,
        gating_negative,
        left=cci_negative,
        color="#789FBE",
        edgecolor="none",
        height=0.62,
    )

    ax.barh(
        y_positions,
        cci_positive,
        color="#D8A59D",
        edgecolor="none",
        height=0.62,
    )

    ax.barh(
        y_positions,
        gating_positive,
        left=cci_positive,
        color=COL_GAIN,
        edgecolor="none",
        height=0.62,
    )

    ax.scatter(
        net_gain,
        y_positions,
        s=20,
        c=COL_BLACK,
        marker="o",
        zorder=5,
    )

    ax.axvline(
        0.0,
        color="#777C7E",
        linewidth=0.7,
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels([
        shorten_text(
            pathway
        )
        for pathway
        in top["pathway"]
    ])

    ax.set_xlabel(
        "Temporal transmission gain"
    )

    ax.set_title(
        "Top pathways with MLTD-enhanced\n"
        "TF-aware temporal transmission",
        fontweight="bold",
        pad=8,
    )

    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#D8A59D",
            edgecolor="none",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COL_GAIN,
            edgecolor="none",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#A6BED2",
            edgecolor="none",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COL_BLACK,
            markeredgecolor=COL_BLACK,
            markersize=4,
        ),
    ]

    legend_labels = [
        "Positive relative CCI-mass contribution",
        "Positive temporal-gating contribution",
        "Negative relative CCI-mass contribution",
        "Net temporal gain",
    ]

    ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower right",
        fontsize=6.0,
    )

    return top


def paired_metric_test(
    metric_table: pd.DataFrame,
    metric: str,
) -> tuple[
    float,
    str,
    float,
]:

    pivot = metric_table.pivot(
        index="seed",
        columns="mode",
        values=metric,
    ).dropna()

    if (
        "off" not in pivot.columns
        or "on" not in pivot.columns
        or len(pivot) < 2
    ):
        return np.nan, "", np.nan

    difference = (
        pivot["on"]
        - pivot["off"]
    )

    mean_difference = float(
        difference.mean()
    )

    direction = (
        "On > Off"
        if mean_difference >= 0
        else "On < Off"
    )

    if SCIPY_OK:
        alternative = (
            "greater"
            if mean_difference >= 0
            else "less"
        )

        try:
            p_value = float(
                wilcoxon(
                    pivot["on"],
                    pivot["off"],
                    alternative=alternative,
                ).pvalue
            )
        except Exception:
            p_value = np.nan
    else:
        p_value = np.nan

    return (
        p_value,
        direction,
        mean_difference,
    )


def plot_metric_panel(
    ax,
    metric_table: pd.DataFrame,
    metric: str,
    title: str,
):

    pivot = metric_table.pivot(
        index="seed",
        columns="mode",
        values=metric,
    ).dropna()

    if (
        "off" not in pivot.columns
        or "on" not in pivot.columns
        or len(pivot) == 0
    ):
        ax.set_axis_off()
        return

    off_values = (
        pivot["off"]
        .to_numpy(float)
    )

    on_values = (
        pivot["on"]
        .to_numpy(float)
    )

    for off_value, on_value in zip(
        off_values,
        on_values,
    ):
        ax.plot(
            [
                0.0,
                1.0,
            ],
            [
                off_value,
                on_value,
            ],
            color="#B7BBBE",
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )

    ax.scatter(
        np.zeros(
            len(off_values)
        ),
        off_values,
        s=21,
        color=COL_OFF,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )

    ax.scatter(
        np.ones(
            len(on_values)
        ),
        on_values,
        s=21,
        color=COL_ON,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )

    p_value, direction, _ = (
        paired_metric_test(
            metric_table,
            metric,
        )
    )

    if np.isfinite(
        p_value
    ):
        p_text = (
            "P<0.001"
            if p_value < 0.001
            else f"P={p_value:.3f}"
        )

        ax.text(
            0.5,
            1.03,
            f"{direction}\n{p_text}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.1,
        )

    ax.set_xticks(
        [
            0.0,
            1.0,
        ]
    )

    ax.set_xticklabels(
        [
            "Off",
            "On",
        ]
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=17,
    )

    ax.grid(
        axis="y",
        color="#E8EAEB",
        linewidth=0.6,
    )

    minimum = float(
        min(
            np.min(off_values),
            np.min(on_values),
        )
    )

    maximum = float(
        max(
            np.max(off_values),
            np.max(on_values),
        )
    )

    padding = (
        0.10
        * (
            maximum
            - minimum
            + 1e-8
        )
    )

    ax.set_ylim(
        minimum - padding,
        maximum + padding,
    )


def make_corrected_main_figure(
    off_summary: pd.DataFrame,
    on_summary: pd.DataFrame,
    off_edges: pd.DataFrame,
    on_edges: pd.DataFrame,
    gain_table: pd.DataFrame,
    pathway_summary: pd.DataFrame,
    metric_table: pd.DataFrame,
    output_dir: Path,
):

    cluster_set = (
        set(
            off_summary["cluster"]
        )
        & set(
            on_summary["cluster"]
        )
    )

    clusters = sorted(
        cluster_set,
        key=cluster_sort_key,
    )

    fig = plt.figure(
        figsize=(
            16.4,
            10.2,
        )
    )

    grid = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[
            1.13,
            1.13,
            0.88,
            1.25,
        ],
        height_ratios=[
            1.0,
            0.96,
        ],
        wspace=0.62,
        hspace=0.56,
    )

    ax_a = fig.add_subplot(
        grid[0, 0]
    )

    ax_b = fig.add_subplot(
        grid[0, 1]
    )

    ax_c = fig.add_subplot(
        grid[0, 2]
    )

    ax_d = fig.add_subplot(
        grid[0, 3]
    )

    ax_e = fig.add_subplot(
        grid[1, 0:2]
    )

    metric_grid = GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=grid[1, 2:4],
        wspace=0.50,
    )

    ax_f1 = fig.add_subplot(
        metric_grid[0, 0]
    )

    ax_f2 = fig.add_subplot(
        metric_grid[0, 1]
    )

    ax_f3 = fig.add_subplot(
        metric_grid[0, 2]
    )

    ax_f4 = fig.add_subplot(
        metric_grid[0, 3]
    )

    scatter_a = plot_temporal_matrix(
        ax=ax_a,
        edge_table=off_edges,
        clusters=clusters,
        title="MLTD-off GNN-LTC",
        title_color=COL_OFF,
    )

    scatter_b = plot_temporal_matrix(
        ax=ax_b,
        edge_table=on_edges,
        clusters=clusters,
        title="MLTD-on GNN-LTC",
        title_color=COL_ON,
    )

    rank_shift = plot_rank_shift(
        ax=ax_c,
        off_summary=off_summary,
        on_summary=on_summary,
    )

    gain_image = plot_gain_matrix(
        ax=ax_d,
        gain_table=gain_table,
        clusters=clusters,
    )

    top_pathways = plot_pathway_gain(
        ax=ax_e,
        pathway_summary=pathway_summary,
    )

    plot_metric_panel(
        ax=ax_f1,
        metric_table=metric_table,
        metric="weighted_forward_fraction",
        title="Forward\nfraction",
    )

    plot_metric_panel(
        ax=ax_f2,
        metric_table=metric_table,
        metric="weighted_mean_delta_t",
        title="Mean\nΔt",
    )

    plot_metric_panel(
        ax=ax_f3,
        metric_table=metric_table,
        metric="temporal_alignment",
        title="Temporal\nalignment",
    )

    plot_metric_panel(
        ax=ax_f4,
        metric_table=metric_table,
        metric="tf_temporal_alignment",
        title="TF-aware\nalignment",
    )

    panel_label(
        ax_a,
        "A",
    )

    panel_label(
        ax_b,
        "B",
    )

    panel_label(
        ax_c,
        "C",
    )

    panel_label(
        ax_d,
        "D",
    )

    panel_label(
        ax_e,
        "E",
    )

    panel_label(
        ax_f1,
        "F",
        x=-0.23,
        y=1.13,
    )

    if scatter_b is not None:
        time_colorbar = fig.colorbar(
            scatter_b,
            ax=[
                ax_a,
                ax_b,
            ],
            fraction=0.022,
            pad=0.018,
        )

        time_colorbar.set_label(
            "t(receiver) − t(sender)"
        )

    gain_colorbar = fig.colorbar(
        gain_image,
        ax=ax_d,
        fraction=0.045,
        pad=0.035,
    )

    gain_colorbar.set_label(
        "Temporal transmission gain"
    )

    metric_title_x = (
        ax_f1.get_position().x0
        + ax_f4.get_position().x1
    ) / 2.0

    metric_title_y = (
        ax_f1.get_position().y1
        + 0.035
    )

    fig.text(
        metric_title_x,
        metric_title_y,
        "Paired global temporal metrics",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    fig.suptitle(
        "MLTD selectively strengthens TF-aware communication "
        "across a preserved GNN-LTC temporal backbone\n"
        "Communication mass matched",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.975,
        bottom=0.075,
        top=0.91,
    )

    output_stem = (
        output_dir
        / "MLTD_on_off_paired_GNN_LTC_summary_corrected"
    )

    fig.savefig(
        output_stem.with_suffix(
            ".png"
        ),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(fig)

    rank_shift.to_csv(
        output_dir
        / "MLTD_cluster_pseudotime_rank_shift_corrected.csv",
        index=False,
    )

    top_pathways.to_csv(
        output_dir
        / "MLTD_top_pathways_temporal_gain_corrected.csv",
        index=False,
    )

    log(
        "[SAVE]",
        output_stem.with_suffix(
            ".png"
        ),
    )


# =============================================================================
# 14. STANDALONE LR SCATTER
# =============================================================================

def make_standalone_lr_scatter(
    mode: str,
    lr_summary: pd.DataFrame,
    output_dir: Path,
):

    if mode == "off":
        display = "MLTD-off"
        title_color = COL_OFF
    else:
        display = "MLTD-on"
        title_color = COL_ON

    fig, ax = plt.subplots(
        figsize=(
            8.6,
            6.35,
        )
    )

    plot_lr_scatter_on_axis(
        ax=ax,
        lr_summary=lr_summary,
        title=(
            f"{display} representative ligand–receptor pairs"
        ),
        title_color=title_color,
    )

    panel_label(
        ax,
        "J",
        x=-0.10,
        y=1.06,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.97,
        bottom=0.12,
        top=0.90,
    )

    output_stem = (
        output_dir
        / f"{display.replace('-', '_')}_LR_scatter_LIANA_consistent_corrected"
    )

    fig.savefig(
        output_stem.with_suffix(
            ".png"
        ),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        output_stem.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 15. MAIN
# =============================================================================

def main():

    result_dir = locate_result_dir()
    mltd_cci_dir = locate_mltd_cci_dir()
    program_csv = locate_program_csv(
        mltd_cci_dir
    )

    output_dir = (
        result_dir
        / CORRECTED_OUT_DIR_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("[RESULT DIR]", result_dir)
    log("[MLTD CCI DIR]", mltd_cci_dir)
    log("[PROGRAM CSV]", program_csv)
    log("[OUTPUT DIR]", output_dir)

    # ------------------------------------------------------------------
    # Load paired GNN-LTC outputs
    # ------------------------------------------------------------------

    off_summary = load_cluster_summary(
        result_dir
        / "MLTD_off_cluster_pseudotime_ensemble.csv",
        mode="off",
    )

    on_summary = load_cluster_summary(
        result_dir
        / "MLTD_on_cluster_pseudotime_ensemble.csv",
        mode="on",
    )

    off_edges = load_edge_temporal_table(
        result_dir
        / "MLTD_off_edge_temporal_scores.csv",
        mode="off",
    )

    on_edges = load_edge_temporal_table(
        result_dir
        / "MLTD_on_edge_temporal_scores.csv",
        mode="on",
    )

    gain_table = load_gain_table(
        result_dir
        / "MLTD_on_minus_off_temporal_gain.csv"
    )

    pathway_summary = load_pathway_summary(
        result_dir
        / "MLTD_pathway_temporal_gain.csv"
    )

    metric_table = load_metric_table(
        result_dir
        / "MLTD_on_off_seed_metrics.csv"
    )

    programs = load_program_table(
        program_csv
    )

    # ------------------------------------------------------------------
    # Correct LR scatter: MLTD-off
    # ------------------------------------------------------------------

    off_score_post = compute_mode_score_post(
        programs=programs,
        cluster_summary=off_summary,
        mode="off",
    )

    off_highlight = select_highlight_lr(
        off_score_post
    )

    off_lr_summary = summarize_lr_for_scatter(
        score_post_table=off_score_post,
        highlight_lr=off_highlight,
    )

    off_lr_summary["mode"] = "off"

    off_lr_summary.to_csv(
        output_dir
        / "MLTD_off_LR_scatter_LIANA_consistent_corrected.csv",
        index=False,
    )

    off_score_post.to_csv(
        output_dir
        / "MLTD_off_LR_temporal_score_post_rows.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Correct LR scatter: MLTD-on
    # ------------------------------------------------------------------

    on_score_post = compute_mode_score_post(
        programs=programs,
        cluster_summary=on_summary,
        mode="on",
    )

    on_highlight = select_highlight_lr(
        on_score_post
    )

    on_lr_summary = summarize_lr_for_scatter(
        score_post_table=on_score_post,
        highlight_lr=on_highlight,
    )

    on_lr_summary["mode"] = "on"

    on_lr_summary.to_csv(
        output_dir
        / "MLTD_on_LR_scatter_LIANA_consistent_corrected.csv",
        index=False,
    )

    on_score_post.to_csv(
        output_dir
        / "MLTD_on_LR_temporal_score_post_rows.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Improved cell projected pseudotime
    # ------------------------------------------------------------------

    cell_map = load_cell_map()
    original_cell_pt = load_original_cell_pt()

    off_cell_table = project_cells_to_mode(
        cell_map=cell_map,
        original_cell_pt=original_cell_pt,
        cluster_summary=off_summary,
        mode="off",
    )

    on_cell_table = project_cells_to_mode(
        cell_map=cell_map,
        original_cell_pt=original_cell_pt,
        cluster_summary=on_summary,
        mode="on",
    )

    if off_cell_table is None:
        raise RuntimeError(
            "Unable to construct MLTD-off cell projected pseudotime."
        )

    if on_cell_table is None:
        raise RuntimeError(
            "Unable to construct MLTD-on cell projected pseudotime."
        )

    off_cell_table.to_csv(
        output_dir
        / "MLTD_off_cell_projected_pseudotime_corrected.csv",
        index=False,
    )

    on_cell_table.to_csv(
        output_dir
        / "MLTD_on_cell_projected_pseudotime_corrected.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Corrected main figure
    # ------------------------------------------------------------------

    make_corrected_main_figure(
        off_summary=off_summary,
        on_summary=on_summary,
        off_edges=off_edges,
        on_edges=on_edges,
        gain_table=gain_table,
        pathway_summary=pathway_summary,
        metric_table=metric_table,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Mode-specific overview figures
    # ------------------------------------------------------------------

    make_mode_specific_overview(
        mode="off",
        cluster_summary=off_summary,
        cell_table=off_cell_table,
        lr_summary=off_lr_summary,
        output_dir=output_dir,
    )

    make_mode_specific_overview(
        mode="on",
        cluster_summary=on_summary,
        cell_table=on_cell_table,
        lr_summary=on_lr_summary,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Standalone scatter figures
    # ------------------------------------------------------------------

    make_standalone_lr_scatter(
        mode="off",
        lr_summary=off_lr_summary,
        output_dir=output_dir,
    )

    make_standalone_lr_scatter(
        mode="on",
        lr_summary=on_lr_summary,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "result_dir": str(
            result_dir
        ),
        "mltd_cci_dir": str(
            mltd_cci_dir
        ),
        "program_csv": str(
            program_csv
        ),
        "cell_map_csv": str(
            CELL_MAP_CSV
        ),
        "original_cell_pt_csv": str(
            ORIGINAL_CELL_PT_CSV
        ),
        "output_dir": str(
            output_dir
        ),
        "lr_scatter_logic": {
            "temporal_alpha": TEMPORAL_ALPHA,
            "temporal_delta": TEMPORAL_DELTA,
            "post_keep_relative_to_max": (
                POST_KEEP_RELATIVE_TO_MAX
            ),
            "top_lr": TOP_LR,
            "active_within_lr_fraction": (
                ACTIVE_WITHIN_LR_FRAC
            ),
            "global_active_quantile": (
                GLOBAL_ACTIVE_Q
            ),
            "consistency_definition": (
                "0.62 * active_receiver_coverage "
                "+ 0.38 * normalized_entropy"
            ),
            "background_n": BACKGROUND_N,
            "right_low_background_n": (
                RIGHT_LOW_BG_N
            ),
        },
    }

    with open(
        output_dir
        / "corrected_plot_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    log("\n" + "=" * 90)
    log("[DONE]")
    log("[OUTPUT DIR]", output_dir)

    print("\n[MLTD-OFF HIGHLIGHT LR]")

    print(
        off_lr_summary[
            off_lr_summary["is_highlight"]
        ]
        .sort_values(
            "highlight_rank"
        )[
            [
                "highlight_rank",
                "lr_pretty",
                "module",
                "communication_score_raw",
                "mean_strength",
                "consistency_raw",
                "communication_score",
                "consistency_plot",
                "active_receiver_n",
                "receiver_total_n",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print("\n[MLTD-ON HIGHLIGHT LR]")

    print(
        on_lr_summary[
            on_lr_summary["is_highlight"]
        ]
        .sort_values(
            "highlight_rank"
        )[
            [
                "highlight_rank",
                "lr_pretty",
                "module",
                "communication_score_raw",
                "mean_strength",
                "consistency_raw",
                "communication_score",
                "consistency_plot",
                "active_receiver_n",
                "receiver_total_n",
            ]
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
