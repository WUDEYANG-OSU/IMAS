# -*- coding: utf-8 -*-
"""
correct_MLTD_on_off_LR_scatter_from_top10_pathways_v4.py
========================================================

核心修改
--------
1. 高亮 LR 不再从所有 LR 中按 score_post 全局取 Top。
2. 高亮 LR 必须来自：
   <SAMPLE_ID>_top10_unique_MLTD_gain_pathways.csv
3. 每条 Top pathway 选择一个代表 LR：
   a. 优先使用 representative_ligand–representative_receptor；
   b. 若代表 LR 不存在，则在同一 display pathway 中选择
      MLTD-off + MLTD-on temporal score 总和最高的 LR；
   c. 若该 pathway 无任何候选 LR，则记录为 missing。
4. MLTD-off 和 MLTD-on 使用完全相同的一组高亮 LR。
5. 若代表 LR 在某个 mode 中低于 temporal pruning 阈值，
   仍强制加入该 mode 的 scatter，保证严格配对比较。
6. 散点计算保持原始 LIANA-consistent 逻辑：
   score_mode
   × temporal gate
   → relative pruning
   → LR × receiver cluster aggregation
   → receiver-state coverage + normalized entropy
   → rank-preserving display spread + deterministic jitter

本脚本不重新训练 GNN-LTC。

需要已有输出
------------
communication_pseudotime_clusterGNN_LTC/
MLTD_on_off_paired_GNN_LTC_v2/

至少包含：
- MLTD_off_cluster_pseudotime_ensemble.csv
- MLTD_on_cluster_pseudotime_ensemble.csv

需要 MLTD CCI 输出：
- <SAMPLE_ID>_same_universe_program_on_off_scores.csv
- <SAMPLE_ID>_top10_unique_MLTD_gain_pathways.csv

输出
----
corrected_top10_pathway_LR_v4/

- shared_top10_pathway_representative_LR.csv
- missing_top10_pathways.csv
- MLTD_off_LR_temporal_rows_all.csv
- MLTD_on_LR_temporal_rows_all.csv
- MLTD_off_LR_temporal_rows_for_plot.csv
- MLTD_on_LR_temporal_rows_for_plot.csv
- MLTD_off_LR_scatter_summary.csv
- MLTD_on_LR_scatter_summary.csv
- MLTD_off_cell_projected_pseudotime.csv
- MLTD_on_cell_projected_pseudotime.csv

- MLTD_off_mode_specific_overview_top10LR.png/svg/pdf
- MLTD_on_mode_specific_overview_top10LR.png/svg/pdf
- MLTD_off_LR_scatter_top10_pathways.png/svg/pdf
- MLTD_on_LR_scatter_top10_pathways.png/svg/pdf
- MLTD_off_on_shared_LR_shift.png/svg/pdf
"""

from __future__ import annotations

import json
import math
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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

LTC_DIR = (
    CCI_DIR
    / "communication_pseudotime_clusterGNN_LTC"
)

# -------------------------------------------------------------------------
# 已完成的 paired GNN-LTC 结果目录
# -------------------------------------------------------------------------

PAIRED_RESULT_DIR_CANDIDATES = [
    LTC_DIR / "MLTD_on_off_paired_GNN_LTC_v2",
    LTC_DIR / "MLTD_on_off_paired_GNN_LTC",
]

# -------------------------------------------------------------------------
# MLTD-on/off CCI 输出目录
# -------------------------------------------------------------------------

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
    LTC_DIR
    / "cell_projected_pseudotime.csv"
)

OUTPUT_FOLDER_NAME = "corrected_top10_pathway_LR_v4"


# =============================================================================
# 0.1 TEMPORAL LR SETTINGS
# =============================================================================

TEMPORAL_ALPHA = 12.0
TEMPORAL_DELTA = 0.05

# 普通背景 LR 使用的 relative pruning
POST_KEEP_RELATIVE_TO_MAX = 0.03

# 代表 LR 即使低于 pruning 阈值也强制保留
FORCE_KEEP_SELECTED_LR = True

# 每条 Top pathway 仅保留一个代表 LR
MAX_LR_PER_PATHWAY = 1

# 若最终不足 10 个，是否允许从 Top pathway 内补充第二个 LR
ALLOW_SECOND_LR_FROM_PATHWAY = False


# =============================================================================
# 0.2 CONSISTENCY SETTINGS
# =============================================================================

ACTIVE_WITHIN_LR_FRAC = 0.18
GLOBAL_ACTIVE_Q = 0.35

CONSISTENCY_COVERAGE_WEIGHT = 0.62
CONSISTENCY_ENTROPY_WEIGHT = 0.38


# =============================================================================
# 0.3 SCATTER LAYOUT SETTINGS
# =============================================================================

BACKGROUND_N = 90
BACKGROUND_BINS = 10
RIGHT_LOW_BG_N = 8

X_MIN = 1e-2
X_MAX = 12.0

Y_MIN = 20.0
Y_MAX = 100.0

BG_X_LO = 1e-2
BG_X_HI = 1.85

RIGHT_LOW_X_LO = 1.55
RIGHT_LOW_X_HI = 4.8

HI_X_LO = 2.2
HI_X_HI = 11.5

BG_Y_LO = 22.0
BG_Y_HI = 89.0

RIGHT_LOW_Y_LO = 26.0
RIGHT_LOW_Y_HI = 54.0

HI_Y_LO = 52.0
HI_Y_HI = 97.0

X_JITTER_SIGMA_BG = 0.15
X_JITTER_SIGMA_HI = 0.16

Y_JITTER_BG = 7.8
Y_JITTER_HI = 6.4

BG_SIZE_MIN = 10.0
BG_SIZE_MAX = 62.0

HI_SIZE_MIN = 145.0
HI_SIZE_MAX = 455.0

BG_ALPHA = 0.48
HI_ALPHA = 0.97

DPI = 450


# =============================================================================
# 0.4 COLORS
# =============================================================================

COL_OFF = "#8B969C"
COL_ON = "#C45843"

COL_TEXT = "#242729"
COL_MUTED = "#747A7D"

MODULE_COLORS = {
    "ECM / Adhesion": "#8DAA43",
    "Cytokine / Chemokine": "#4FA7A3",
    "Growth-factor": "#4F7DB8",
    "Immune-like": "#8B6CAF",
    "Other": "#B9C0C4",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ],
    "font.size": 8.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
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

def log(*args: Any) -> None:
    print(*args, flush=True)


def normalized_name(value: Any) -> str:
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


def norm_barcode(value: Any) -> str:
    text = str(value)

    if "::" in text:
        return text.split(
            "::",
            1,
        )[-1]

    return text


def norm_lr(value: Any) -> str:
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
    ligand: Any,
    receptor: Any,
) -> str:
    return (
        f"{str(ligand).strip()}"
        f"–"
        f"{str(receptor).strip()}"
    )


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

    candidates: list[str] = []

    for column in df.columns:
        column_key = normalized_name(
            column
        )

        for alias in aliases:
            alias_key = normalized_name(
                alias
            )

            if len(alias_key) < 5:
                continue

            if (
                column_key.startswith(alias_key)
                or column_key.endswith(alias_key)
            ):
                candidates.append(
                    str(column)
                )

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


def parse_bool_series(
    values,
) -> pd.Series:

    series = pd.Series(values)

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .fillna(0.0)
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
        return np.zeros_like(
            x,
            dtype=float,
        )

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


def scale_to_range(
    values,
    low: float,
    high: float,
) -> np.ndarray:

    scaled = minmax01(
        values
    )

    return (
        low
        + scaled
        * (
            high - low
        )
    )


def sigmoid_np(values) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=float,
    )

    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                x,
                -30.0,
                30.0,
            )
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

    total = float(
        x.sum()
    )

    if (
        total <= 0
        or len(x) <= 1
    ):
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


def log_rank_spread(
    values,
    low: float,
    high: float,
    pad_left: float = 0.03,
    pad_right: float = 0.08,
) -> np.ndarray:

    x = np.asarray(
        values,
        dtype=float,
    )

    if len(x) <= 1:
        return np.full_like(
            x,
            np.sqrt(
                low * high
            ),
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

    fractions = ranks / max(
        len(x) - 1,
        1,
    )

    fractions = (
        pad_left
        + (
            1.0
            - pad_left
            - pad_right
        )
        * fractions
    )

    log_low = np.log10(low)
    log_high = np.log10(high)

    return 10.0 ** (
        log_low
        + fractions
        * (
            log_high - log_low
        )
    )


def linear_rank_spread(
    values,
    low: float,
    high: float,
    pad_bottom: float = 0.03,
    pad_top: float = 0.03,
) -> np.ndarray:

    x = np.asarray(
        values,
        dtype=float,
    )

    if len(x) <= 1:
        return np.full_like(
            x,
            (
                low + high
            ) / 2.0,
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

    fractions = ranks / max(
        len(x) - 1,
        1,
    )

    fractions = (
        pad_bottom
        + (
            1.0
            - pad_bottom
            - pad_top
        )
        * fractions
    )

    return (
        low
        + fractions
        * (
            high - low
        )
    )


def deterministic_seed(
    text: Any,
    offset: int = 0,
) -> int:
    return int(
        (
            sum(
                ord(character)
                for character
                in str(text)
            )
            + offset
        )
        % 2_147_483_647
    )


def panel_label(
    ax,
    label: str,
    x: float = -0.12,
    y: float = 1.07,
) -> None:

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
# 2. PATHWAY MAPPING
# =============================================================================

def infer_display_kegg_pathway(
    ligand: Any,
    receptor: Any,
    analysis_pathway: Any = "",
) -> str:

    ligand_key = canonical(
        ligand
    )

    receptor_key = canonical(
        receptor
    )

    analysis_key = canonical(
        analysis_pathway
    )

    joined = (
        ligand_key
        + "|"
        + receptor_key
        + "|"
        + analysis_key
    )

    rules = [
        (
            "Protein processing in endoplasmic reticulum",
            r"(HSPA5|HSP90B1|CANX|CALR|PDIA|DNAJ|SEC61|CDNF)",
        ),
        (
            "Complement and coagulation cascades",
            r"(\bF2\b|\bF3\b|\bF5\b|\bF7\b|\bF8\b|\bF9\b|"
            r"\bF10\b|\bF11\b|\bF12\b|FGA|FGB|FGG|PLG|"
            r"SERPINC1|PROC|PROS1|CTSG)",
        ),
        (
            "mTOR signaling pathway",
            r"(MTOR|RPTOR|RICTOR|RPS6KB1|RPS6KB2|EIF4EBP|AKT1S1)",
        ),
        (
            "JAK–STAT signaling pathway",
            r"(EPOR|JAK\d|STAT\d|IFNAR|IFNGR|IL\d+R|CSF\dR|"
            r"LIFR|OSMR|CNTFR)",
        ),
        (
            "ErbB signaling pathway",
            r"(ERBB|EGFR|LRIG|NRG\d|EREG|AREG|HBEGF|BTC|EPGN)",
        ),
        (
            "Axon guidance",
            r"(SEMA|PLXN|NRP\d|EPHA|EPHB|EFNA|EFNB|ROBO|SLIT)",
        ),
        (
            "ECM–receptor interaction",
            r"(COL\d|LAMA|LAMB|LAMC|FN1|HSPG|HSPG2|AGRN|"
            r"VCAN|SDC\d|CD44|DAG1|DDR1|DDR2)",
        ),
        (
            "Cell adhesion molecules",
            r"(ALCAM|NCAM|ICAM|VCAM|PECAM|NECTIN|JAM|CDH\d|"
            r"CLDN|LRRN|PTPR[ABCDMOPRSZ])",
        ),
        (
            "Chemokine signaling pathway",
            r"(CXCL|CXCR|CCL\d|CCR\d|XCL\d|XCR1)",
        ),
        (
            "Cytokine–cytokine receptor interaction",
            r"(IL\d|IFN|TNF|TNFR|CSF|OSM|LIF|CNTF|FLT3LG)",
        ),
        (
            "TGF-beta signaling pathway",
            r"(TGFB|TGFBR|BMP\d|BMPR|ACVR|INHBA|INHBB|SMAD)",
        ),
        (
            "Wnt signaling pathway",
            r"(WNT\d|FZD\d|LRP5|LRP6|ROR1|ROR2|RYK)",
        ),
        (
            "Notch signaling pathway",
            r"(NOTCH\d|JAG1|JAG2|DLL1|DLL3|DLL4)",
        ),
        (
            "Hedgehog signaling pathway",
            r"(SHH|IHH|DHH|PTCH1|PTCH2|SMO|HHIP)",
        ),
        (
            "VEGF signaling pathway",
            r"(VEGFA|VEGFB|VEGFC|VEGFD|KDR|FLT1|FLT4)",
        ),
        (
            "PI3K–Akt signaling pathway",
            r"(KITLG|KIT|IGF|IGF1R|PDGF|PDGFR|FGF|FGFR|"
            r"HGF|MET|GAS6|AXL|MERTK|TEK|ANGPT)",
        ),
        (
            "MAPK signaling pathway",
            r"(MAPK|ERK|RAF|MEK|NTRK|NGF|BDNF)",
        ),
        (
            "Focal adhesion",
            r"(ITGA|ITGB|PXN|PTK2|TLN|VCL|THBS|SPP1)",
        ),
        (
            "Apoptosis",
            r"(FAS|FASLG|TNFSF10|TNFRSF10|TRAIL|BCL2|BAX|CASP)",
        ),
        (
            "Calcium signaling pathway",
            r"(P2RY|P2RX|ADRA|ADRB|CHRM|CHRNA|GRM|HTR)",
        ),
        (
            "Drug metabolism–cytochrome P450",
            r"(CYP\d|UGT\d|GST[AMPT]|ALDH\d)",
        ),
    ]

    for pathway_name, pattern in rules:
        if re.search(
            pattern,
            joined,
            flags=re.I,
        ):
            return pathway_name

    analysis_rules = [
        (
            "ErbB signaling pathway",
            r"ERBB",
        ),
        (
            "ECM–receptor interaction",
            r"ECM|MATRIX",
        ),
        (
            "Focal adhesion",
            r"ADHESION",
        ),
        (
            "Axon guidance",
            r"SEMAPHORIN|EPH",
        ),
        (
            "Chemokine signaling pathway",
            r"CHEMOKINE",
        ),
        (
            "Cytokine–cytokine receptor interaction",
            r"CYTOKINE",
        ),
        (
            "TGF-beta signaling pathway",
            r"TGF|BMP",
        ),
        (
            "Wnt signaling pathway",
            r"WNT",
        ),
        (
            "Notch signaling pathway",
            r"NOTCH",
        ),
        (
            "PI3K–Akt signaling pathway",
            r"GROWTH FACTOR|RTK",
        ),
    ]

    for pathway_name, pattern in analysis_rules:
        if re.search(
            pattern,
            analysis_key,
            flags=re.I,
        ):
            return pathway_name

    return "Other signaling pathways"


def classify_module(
    ligand: Any,
    receptor: Any,
) -> str:

    ligand_key = canonical(
        ligand
    )

    receptor_key = canonical(
        receptor
    )

    joined = (
        ligand_key
        + "|"
        + receptor_key
    )

    if re.search(
        r"(COL\d|LAMA|LAMB|LAMC|FN1|SPP1|THBS|VCAN|HSPG|"
        r"ITGA|ITGB|CD44|DDR\d|DAG1|ICAM|VCAM|NECTIN|"
        r"ALCAM|JAM|CDH\d)",
        joined,
        flags=re.I,
    ):
        return "ECM / Adhesion"

    if re.search(
        r"(CXCL|CCL|IL\d|IFN|TNF|CSF|OSM|LIF|"
        r"CXCR|CCR|IFNAR|IFNGR|TNFR)",
        joined,
        flags=re.I,
    ):
        return "Cytokine / Chemokine"

    if re.search(
        r"(EGF|FGF|VEGF|PDGF|HGF|IGF|NRG|GAS6|KITLG|"
        r"EGFR|FGFR|ERBB|MET|KDR|FLT\d|PDGFR|IGF\dR|"
        r"KIT|AXL|MERTK|WNT|FZD|NOTCH|JAG\d|DLL\d|"
        r"SEMA|EPHA|EPHB|EFNA|EFNB)",
        joined,
        flags=re.I,
    ):
        return "Growth-factor"

    if re.search(
        r"(HLA|CD\d|CEACAM|LCK|FAS|FASLG|TRAIL|TNFSF|"
        r"TNFRSF|B2M|PTPRC)",
        joined,
        flags=re.I,
    ):
        return "Immune-like"

    return "Other"


# =============================================================================
# 3. LOCATE FILES
# =============================================================================

def locate_paired_result_dir() -> Path:

    required = [
        "MLTD_off_cluster_pseudotime_ensemble.csv",
        "MLTD_on_cluster_pseudotime_ensemble.csv",
    ]

    for directory in PAIRED_RESULT_DIR_CANDIDATES:
        if all(
            (
                directory
                / filename
            ).exists()
            for filename in required
        ):
            return directory

    hits = sorted(
        LTC_DIR.rglob(
            "MLTD_off_cluster_pseudotime_ensemble.csv"
        ),
        key=lambda path: (
            "v2" not in str(path),
            len(str(path)),
            str(path),
        ),
    )

    for path in hits:
        directory = path.parent

        if all(
            (
                directory
                / filename
            ).exists()
            for filename in required
        ):
            return directory

    raise FileNotFoundError(
        "Cannot locate paired GNN-LTC outputs."
    )


def locate_mltd_cci_dir() -> Path:

    required_patterns = [
        "*_same_universe_program_on_off_scores.csv",
        "*_top10_unique_MLTD_gain_pathways.csv",
    ]

    for directory in MLTD_CCI_DIR_CANDIDATES:
        if all(
            len(
                list(
                    directory.glob(
                        pattern
                    )
                )
            ) > 0
            for pattern in required_patterns
        ):
            return directory

    top10_hits = sorted(
        RUN_BASE.rglob(
            f"{SAMPLE_ID}_top10_unique_MLTD_gain_pathways.csv"
        ),
        key=lambda path: (
            "v5_4" not in str(path),
            len(str(path)),
            str(path),
        ),
    )

    for top10_path in top10_hits:
        directory = top10_path.parent

        program_hits = list(
            directory.glob(
                "*_same_universe_program_on_off_scores.csv"
            )
        )

        if program_hits:
            return directory

    raise FileNotFoundError(
        "Cannot locate MLTD CCI directory containing both "
        "program scores and Top 10 pathway table."
    )


def first_matching_file(
    directory: Path,
    patterns: list[str],
) -> Path:

    for pattern in patterns:
        hits = sorted(
            directory.glob(
                pattern
            )
        )

        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"Cannot find any matching file in {directory}:\n"
        + "\n".join(patterns)
    )


# =============================================================================
# 4. LOAD CLUSTER PSEUDOTIME
# =============================================================================

def load_cluster_summary(
    path: Path,
    mode: str,
) -> pd.DataFrame:

    raw = pd.read_csv(
        path
    )

    cluster_col = first_column(
        raw,
        [
            "cluster",
            "cluster_norm",
            "cluster_label",
            "cluster_id",
        ],
        role=f"{mode} cluster",
    )

    time_col = first_column(
        raw,
        [
            "t_hat_mean",
            "t_hat_norm_mean",
            "t_hat_norm",
            "t_hat",
            "pt",
        ],
        role=f"{mode} pseudotime",
    )

    output = raw.copy()

    output["cluster"] = (
        output[cluster_col]
        .map(normalize_cluster)
    )

    output["t_hat_mean"] = pd.to_numeric(
        output[time_col],
        errors="coerce",
    )

    output = (
        output[
            (output["cluster"] != "")
            & output["t_hat_mean"].notna()
        ]
        .drop_duplicates("cluster")
        .copy()
    )

    output["mode"] = mode

    return output


# =============================================================================
# 5. LOAD PROGRAM TABLE
# =============================================================================

def load_program_table(
    path: Path,
) -> pd.DataFrame:

    raw = pd.read_csv(
        path
    )

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
        role="MLTD-off score",
    )

    on_score_col = first_column(
        raw,
        [
            "score_on",
            "on_score",
            "on_weight",
            "mltd_on_score",
        ],
        role="MLTD-on score",
    )

    pathway_col = first_column(
        raw,
        [
            "pathway",
            "analysis_pathway",
            "pathway_label",
            "broad_pathway",
        ],
        required=False,
        role="analysis pathway",
    )

    off_active_col = first_column(
        raw,
        [
            "off_active",
            "native_off",
            "mltd_off_active",
        ],
        required=False,
        role="MLTD-off active",
    )

    on_active_col = first_column(
        raw,
        [
            "on_active",
            "mltd_on_active",
        ],
        required=False,
        role="MLTD-on active",
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
        output["analysis_pathway"] = (
            raw[pathway_col]
            .astype(str)
        )
    else:
        output["analysis_pathway"] = ""

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

    output["display_pathway"] = [
        infer_display_kegg_pathway(
            ligand=ligand,
            receptor=receptor,
            analysis_pathway=analysis_pathway,
        )
        for ligand, receptor, analysis_pathway
        in zip(
            output["ligand"],
            output["receptor"],
            output["analysis_pathway"],
        )
    ]

    output = output[
        (output["sender_cluster"] != "")
        & (output["receiver_cluster"] != "")
        & (output["ligand"] != "")
        & (output["receptor"] != "")
        & (output["lr_norm"] != "")
    ].copy()

    return output


# =============================================================================
# 6. LOAD TOP 10 PATHWAY TABLE
# =============================================================================

def load_top10_pathways(
    path: Path,
) -> pd.DataFrame:

    raw = pd.read_csv(
        path
    )

    pathway_col = first_column(
        raw,
        [
            "pathway",
            "display_pathway",
            "kegg_pathway",
            "pathway_name",
        ],
        role="Top pathway name",
    )

    ligand_col = first_column(
        raw,
        [
            "representative_ligand",
            "ligand",
            "top_ligand",
        ],
        required=False,
        role="representative ligand",
    )

    receptor_col = first_column(
        raw,
        [
            "representative_receptor",
            "receptor",
            "top_receptor",
        ],
        required=False,
        role="representative receptor",
    )

    rank_col = first_column(
        raw,
        [
            "rank",
            "pathway_rank",
            "order",
        ],
        required=False,
        role="pathway rank",
    )

    p_col = first_column(
        raw,
        [
            "unit_simes_p_value",
            "display_simes_p_value",
            "p_value",
            "pvalue",
        ],
        required=False,
        role="pathway P value",
    )

    gain_col = first_column(
        raw,
        [
            "selected_gain_weight_actual",
            "selected_gain_weight",
            "total_positive_gain",
            "gain_weight",
        ],
        required=False,
        role="pathway gain",
    )

    output = pd.DataFrame({
        "pathway": (
            raw[pathway_col]
            .astype(str)
            .str.strip()
        ),
    })

    if ligand_col is not None:
        output["representative_ligand"] = (
            raw[ligand_col]
            .astype(str)
            .str.strip()
        )
    else:
        output["representative_ligand"] = ""

    if receptor_col is not None:
        output["representative_receptor"] = (
            raw[receptor_col]
            .astype(str)
            .str.strip()
        )
    else:
        output["representative_receptor"] = ""

    if rank_col is not None:
        output["rank"] = pd.to_numeric(
            raw[rank_col],
            errors="coerce",
        )
    else:
        output["rank"] = np.arange(
            1,
            len(output) + 1,
        )

    if p_col is not None:
        output["pathway_p_value"] = pd.to_numeric(
            raw[p_col],
            errors="coerce",
        )
    else:
        output["pathway_p_value"] = np.nan

    if gain_col is not None:
        output["pathway_gain"] = pd.to_numeric(
            raw[gain_col],
            errors="coerce",
        )
    else:
        output["pathway_gain"] = np.nan

    output["representative_lr_norm"] = [
        norm_lr(
            f"{ligand}-{receptor}"
        )
        if (
            str(ligand).strip()
            and str(receptor).strip()
            and str(ligand).lower() != "nan"
            and str(receptor).lower() != "nan"
        )
        else ""
        for ligand, receptor
        in zip(
            output["representative_ligand"],
            output["representative_receptor"],
        )
    ]

    output = (
        output.sort_values(
            "rank"
        )
        .drop_duplicates(
            "pathway",
            keep="first",
        )
        .head(10)
        .reset_index(drop=True)
    )

    return output


# =============================================================================
# 7. COMPUTE UNPRUNED TEMPORAL SCORES
# =============================================================================

def compute_mode_temporal_rows(
    programs: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:

    if mode not in {
        "off",
        "on",
    }:
        raise ValueError(
            f"Invalid mode: {mode}"
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

    output["score_raw"] = pd.to_numeric(
        output[score_col],
        errors="coerce",
    ).fillna(0.0)

    output["mode_active"] = (
        output[active_col]
        .fillna(False)
        .astype(bool)
    )

    # 保留所有有正分数的行，代表 LR 后续可被强制纳入；
    # mode_active 用于普通背景筛选。
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
            - TEMPORAL_DELTA
        )
    )

    output["score_post"] = (
        output["score_raw"]
        * output["temporal_gate"]
    )

    output["mode"] = mode

    return output


# =============================================================================
# 8. SELECT SHARED LR FROM TOP 10 PATHWAYS
# =============================================================================

def build_combined_lr_score(
    off_rows: pd.DataFrame,
    on_rows: pd.DataFrame,
) -> pd.DataFrame:

    key_columns = [
        "lr_norm",
        "lr_pretty",
        "ligand",
        "receptor",
        "display_pathway",
    ]

    off_summary = (
        off_rows.groupby(
            key_columns,
            as_index=False,
        )
        .agg(
            temporal_score_off=(
                "score_post",
                "sum",
            ),
            raw_score_off=(
                "score_raw",
                "sum",
            ),
        )
    )

    on_summary = (
        on_rows.groupby(
            key_columns,
            as_index=False,
        )
        .agg(
            temporal_score_on=(
                "score_post",
                "sum",
            ),
            raw_score_on=(
                "score_raw",
                "sum",
            ),
        )
    )

    combined = off_summary.merge(
        on_summary,
        on=key_columns,
        how="outer",
    )

    numeric_columns = [
        "temporal_score_off",
        "raw_score_off",
        "temporal_score_on",
        "raw_score_on",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        ).fillna(0.0)

    combined["combined_temporal_score"] = (
        combined["temporal_score_off"]
        + combined["temporal_score_on"]
    )

    combined["combined_raw_score"] = (
        combined["raw_score_off"]
        + combined["raw_score_on"]
    )

    return combined


def select_shared_top10_pathway_lr(
    top10: pd.DataFrame,
    combined_lr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    selected_rows = []
    missing_rows = []

    used_lr: set[str] = set()

    for top_row in top10.itertuples():

        pathway = str(
            top_row.pathway
        ).strip()

        representative_lr = str(
            top_row.representative_lr_norm
        ).strip()

        pathway_candidates = combined_lr[
            combined_lr["display_pathway"]
            .astype(str)
            .str.strip()
            == pathway
        ].copy()

        selected = None
        selection_source = ""

        # -------------------------------------------------------------
        # 1. 优先使用 Top 10 表中记录的代表性 LR
        # -------------------------------------------------------------

        if representative_lr:
            exact = combined_lr[
                combined_lr["lr_norm"]
                == representative_lr
            ].copy()

            if len(exact) > 0:
                exact = exact.sort_values(
                    [
                        "combined_temporal_score",
                        "combined_raw_score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )

                selected = exact.iloc[0]
                selection_source = (
                    "top10_representative_lr"
                )

        # -------------------------------------------------------------
        # 2. 代表 LR 不存在：同 pathway 内选择 combined temporal 最大 LR
        # -------------------------------------------------------------

        if selected is None:
            available = pathway_candidates[
                ~pathway_candidates["lr_norm"]
                .isin(used_lr)
            ].copy()

            if len(available) == 0:
                available = pathway_candidates.copy()

            if len(available) > 0:
                available = available.sort_values(
                    [
                        "combined_temporal_score",
                        "combined_raw_score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )

                selected = available.iloc[0]
                selection_source = (
                    "same_pathway_highest_combined_temporal_score"
                )

        # -------------------------------------------------------------
        # 3. 该 pathway 无任何候选
        # -------------------------------------------------------------

        if selected is None:
            missing_rows.append({
                "rank": int(
                    top_row.rank
                ),
                "pathway": pathway,
                "representative_ligand": str(
                    top_row.representative_ligand
                ),
                "representative_receptor": str(
                    top_row.representative_receptor
                ),
                "representative_lr_norm": representative_lr,
                "reason": (
                    "No LR candidate found for this display pathway"
                ),
            })

            continue

        selected_lr = str(
            selected["lr_norm"]
        )

        # 同一个 LR 不重复代表两个 pathway；
        # 若重复，尝试寻找同 pathway 的下一个候选。
        if selected_lr in used_lr:
            alternatives = pathway_candidates[
                ~pathway_candidates["lr_norm"]
                .isin(used_lr)
            ].copy()

            if len(alternatives) > 0:
                alternatives = alternatives.sort_values(
                    [
                        "combined_temporal_score",
                        "combined_raw_score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )

                selected = alternatives.iloc[0]
                selected_lr = str(
                    selected["lr_norm"]
                )

                selection_source += (
                    "_deduplicated_alternative"
                )

        used_lr.add(
            selected_lr
        )

        selected_rows.append({
            "rank": int(
                top_row.rank
            ),
            "pathway": pathway,
            "pathway_p_value": getattr(
                top_row,
                "pathway_p_value",
                np.nan,
            ),
            "pathway_gain": getattr(
                top_row,
                "pathway_gain",
                np.nan,
            ),
            "top10_representative_ligand": str(
                top_row.representative_ligand
            ),
            "top10_representative_receptor": str(
                top_row.representative_receptor
            ),
            "top10_representative_lr_norm": representative_lr,
            "selected_ligand": str(
                selected["ligand"]
            ),
            "selected_receptor": str(
                selected["receptor"]
            ),
            "selected_lr_norm": selected_lr,
            "selected_lr_pretty": str(
                selected["lr_pretty"]
            ),
            "selected_display_pathway": str(
                selected["display_pathway"]
            ),
            "selection_source": selection_source,
            "temporal_score_off": float(
                selected["temporal_score_off"]
            ),
            "temporal_score_on": float(
                selected["temporal_score_on"]
            ),
            "combined_temporal_score": float(
                selected["combined_temporal_score"]
            ),
            "raw_score_off": float(
                selected["raw_score_off"]
            ),
            "raw_score_on": float(
                selected["raw_score_on"]
            ),
        })

    selected_table = pd.DataFrame(
        selected_rows
    )

    missing_table = pd.DataFrame(
        missing_rows
    )

    if len(selected_table) == 0:
        raise RuntimeError(
            "No representative LR could be selected from Top 10 pathways."
        )

    selected_table = selected_table.sort_values(
        "rank"
    ).reset_index(drop=True)

    if selected_table[
        "selected_lr_norm"
    ].duplicated().any():
        warnings.warn(
            "The selected LR table still contains duplicated LR identities."
        )

    return (
        selected_table,
        missing_table,
    )


# =============================================================================
# 9. PRUNE BACKGROUND BUT FORCE KEEP SHARED LR
# =============================================================================

def prepare_mode_rows_for_plot(
    all_rows: pd.DataFrame,
    selected_lr: list[str],
    mode: str,
) -> pd.DataFrame:

    active_rows = all_rows[
        all_rows["mode_active"]
    ].copy()

    if len(active_rows) == 0:
        raise RuntimeError(
            f"No active rows for mode={mode}."
        )

    maximum = float(
        active_rows["score_post"].max()
    )

    threshold = (
        maximum
        * POST_KEEP_RELATIVE_TO_MAX
    )

    background_rows = active_rows[
        active_rows["score_post"]
        >= threshold
    ].copy()

    selected_rows = all_rows[
        all_rows["lr_norm"]
        .isin(selected_lr)
    ].copy()

    selected_rows[
        "forced_selected_lr"
    ] = True

    background_rows[
        "forced_selected_lr"
    ] = False

    if FORCE_KEEP_SELECTED_LR:
        output = pd.concat(
            [
                background_rows,
                selected_rows,
            ],
            axis=0,
            ignore_index=True,
        )

        output = output.drop_duplicates(
            subset=[
                "sender_cluster",
                "receiver_cluster",
                "lr_norm",
                "score_raw",
                "score_post",
            ],
            keep="last",
        )
    else:
        output = background_rows.copy()

    output["is_selected_lr"] = (
        output["lr_norm"]
        .isin(selected_lr)
    )

    output["temporal_prune_threshold"] = threshold

    log(
        f"[{mode}] temporal threshold:",
        threshold,
    )

    log(
        f"[{mode}] active rows:",
        len(active_rows),
    )

    log(
        f"[{mode}] background rows after pruning:",
        len(background_rows),
    )

    log(
        f"[{mode}] selected LR rows forced in:",
        len(selected_rows),
    )

    log(
        f"[{mode}] LR for plot:",
        output["lr_norm"].nunique(),
    )

    return output


# =============================================================================
# 10. BUILD LR SCATTER SUMMARY
# =============================================================================

def summarize_lr_for_scatter(
    rows_for_plot: pd.DataFrame,
    selected_table: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:

    selected_lr = (
        selected_table[
            "selected_lr_norm"
        ]
        .astype(str)
        .tolist()
    )

    pathway_map = dict(
        zip(
            selected_table[
                "selected_lr_norm"
            ],
            selected_table[
                "pathway"
            ],
        )
    )

    pathway_rank_map = dict(
        zip(
            selected_table[
                "selected_lr_norm"
            ],
            selected_table[
                "rank"
            ],
        )
    )

    receiver_table = (
        rows_for_plot.groupby(
            [
                "lr_norm",
                "lr_pretty",
                "ligand",
                "receptor",
                "display_pathway",
                "receiver_cluster",
            ],
            as_index=False,
        )
        .agg(
            receiver_score=(
                "score_post",
                "sum",
            ),
            n_sender_clusters=(
                "sender_cluster",
                "nunique",
            ),
            forced_selected_lr=(
                "forced_selected_lr",
                "max",
            ),
        )
    )

    all_receivers = sorted(
        rows_for_plot[
            "receiver_cluster"
        ]
        .dropna()
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

    summary_rows = []

    for lr_norm, sub in receiver_table.groupby(
        "lr_norm",
        sort=False,
    ):
        values = (
            sub["receiver_score"]
            .astype(float)
            .to_numpy()
        )

        ligand = str(
            sub["ligand"].iloc[0]
        )

        receptor = str(
            sub["receptor"].iloc[0]
        )

        maximum = float(
            np.max(values)
        )

        if maximum > 0:
            active_cut = max(
                maximum
                * ACTIVE_WITHIN_LR_FRAC,
                global_active_threshold,
            )

            active_n = int(
                np.sum(
                    values >= active_cut
                )
            )
        else:
            active_cut = np.nan
            active_n = 0

        coverage = (
            active_n
            / n_receivers
        )

        normalized_entropy = entropy01(
            values
        )

        consistency = (
            CONSISTENCY_COVERAGE_WEIGHT
            * coverage
            + CONSISTENCY_ENTROPY_WEIGHT
            * normalized_entropy
        )

        is_highlight = (
            lr_norm in selected_lr
        )

        summary_rows.append({
            "mode": mode,
            "lr_norm": lr_norm,
            "lr_pretty": str(
                sub["lr_pretty"].iloc[0]
            ),
            "ligand": ligand,
            "receptor": receptor,
            "display_pathway": str(
                sub["display_pathway"].iloc[0]
            ),
            "top10_pathway": (
                pathway_map.get(
                    lr_norm,
                    "",
                )
            ),
            "top10_pathway_rank": (
                pathway_rank_map.get(
                    lr_norm,
                    np.nan,
                )
            ),
            "module": classify_module(
                ligand,
                receptor,
            ),
            "communication_score_raw": float(
                np.sum(values)
            ),
            "mean_strength": float(
                np.mean(values)
            ),
            "median_strength": float(
                np.median(values)
            ),
            "max_strength": maximum,
            "active_cut": active_cut,
            "active_receiver_n": active_n,
            "receiver_total_n": n_receivers,
            "active_coverage01": coverage,
            "entropy01": normalized_entropy,
            "consistency_raw": consistency,
            "is_highlight": is_highlight,
            "highlight_rank": (
                pathway_rank_map.get(
                    lr_norm,
                    np.nan,
                )
            ),
            "forced_selected_lr": bool(
                sub["forced_selected_lr"].max()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # 代表 LR 若该 mode 完全没有正分数，则不会出现在 receiver_table 中。
    # 将其加入 summary，显示在接近坐标轴下界的位置。
    missing_highlights = [
        lr
        for lr in selected_lr
        if lr not in set(
            summary["lr_norm"]
        )
    ]

    for lr_norm in missing_highlights:
        selected_row = selected_table[
            selected_table[
                "selected_lr_norm"
            ]
            == lr_norm
        ].iloc[0]

        summary = pd.concat(
            [
                summary,
                pd.DataFrame([{
                    "mode": mode,
                    "lr_norm": lr_norm,
                    "lr_pretty": selected_row[
                        "selected_lr_pretty"
                    ],
                    "ligand": selected_row[
                        "selected_ligand"
                    ],
                    "receptor": selected_row[
                        "selected_receptor"
                    ],
                    "display_pathway": selected_row[
                        "selected_display_pathway"
                    ],
                    "top10_pathway": selected_row[
                        "pathway"
                    ],
                    "top10_pathway_rank": selected_row[
                        "rank"
                    ],
                    "module": classify_module(
                        selected_row[
                            "selected_ligand"
                        ],
                        selected_row[
                            "selected_receptor"
                        ],
                    ),
                    "communication_score_raw": 0.0,
                    "mean_strength": 0.0,
                    "median_strength": 0.0,
                    "max_strength": 0.0,
                    "active_cut": np.nan,
                    "active_receiver_n": 0,
                    "receiver_total_n": n_receivers,
                    "active_coverage01": 0.0,
                    "entropy01": 0.0,
                    "consistency_raw": 0.0,
                    "is_highlight": True,
                    "highlight_rank": selected_row[
                        "rank"
                    ],
                    "forced_selected_lr": True,
                }]),
            ],
            ignore_index=True,
        )

    return apply_scatter_layout(
        summary
    )


# =============================================================================
# 11. SCATTER DISPLAY LAYOUT
# =============================================================================

def stratified_background_sample(
    background_all: pd.DataFrame,
) -> pd.DataFrame:

    background = background_all.copy()

    if len(background) <= BACKGROUND_N:
        return background

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

    n_bins = max(
        int(
            background["score_bin"]
            .nunique()
        ),
        1,
    )

    per_bin = max(
        int(
            np.ceil(
                BACKGROUND_N
                / n_bins
            )
        ),
        1,
    )

    for _, sub in background.groupby(
        "score_bin",
        observed=True,
    ):
        if len(sub) > per_bin:
            sub = sub.sample(
                n=per_bin,
                random_state=42,
                replace=False,
            )

        sampled_parts.append(
            sub
        )

    background = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    if len(background) > BACKGROUND_N:
        background = background.sample(
            n=BACKGROUND_N,
            random_state=42,
            replace=False,
        )

    return background.drop(
        columns=[
            "score_bin",
        ],
        errors="ignore",
    )


def add_deterministic_log_jitter(
    values,
    keys,
    sigma: float,
    offset: int,
    lower: float,
    upper: float,
) -> np.ndarray:

    output = []

    for value, key in zip(
        values,
        keys,
    ):
        rng = np.random.RandomState(
            deterministic_seed(
                key,
                offset=offset,
            )
        )

        multiplier = 10.0 ** rng.normal(
            0.0,
            sigma,
        )

        output.append(
            float(
                np.clip(
                    value * multiplier,
                    lower,
                    upper,
                )
            )
        )

    return np.asarray(
        output,
        dtype=float,
    )


def add_deterministic_linear_jitter(
    values,
    keys,
    sigma: float,
    offset: int,
    lower: float,
    upper: float,
) -> np.ndarray:

    output = []

    for value, key in zip(
        values,
        keys,
    ):
        rng = np.random.RandomState(
            deterministic_seed(
                key,
                offset=offset,
            )
        )

        output.append(
            float(
                np.clip(
                    value
                    + rng.normal(
                        0.0,
                        sigma,
                    ),
                    lower,
                    upper,
                )
            )
        )

    return np.asarray(
        output,
        dtype=float,
    )


def apply_scatter_layout(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    highlight = summary[
        summary["is_highlight"]
    ].copy()

    background_all = summary[
        ~summary["is_highlight"]
    ].copy()

    background = stratified_background_sample(
        background_all
    )

    background["layout_group"] = (
        "main_background"
    )

    used_background = set(
        background["lr_norm"]
        .tolist()
    )

    right_low_pool = background_all[
        ~background_all["lr_norm"]
        .isin(used_background)
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

        high_index = max(
            int(
                len(right_low_pool)
                * 0.55
            ),
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

        right_low["layout_group"] = (
            "right_low_background"
        )
    else:
        right_low = pd.DataFrame(
            columns=summary.columns.tolist()
            + [
                "layout_group",
            ]
        )

    highlight["layout_group"] = (
        "highlight"
    )

    # ------------------------------------------------------------------
    # Main background
    # ------------------------------------------------------------------

    if len(background) > 0:
        background[
            "communication_score"
        ] = log_rank_spread(
            background[
                "communication_score_raw"
            ],
            low=BG_X_LO,
            high=BG_X_HI,
            pad_left=0.03,
            pad_right=0.12,
        )

        background[
            "communication_score"
        ] = add_deterministic_log_jitter(
            background[
                "communication_score"
            ],
            background["lr_norm"],
            sigma=X_JITTER_SIGMA_BG,
            offset=123,
            lower=BG_X_LO,
            upper=BG_X_HI,
        )

        background[
            "consistency_plot"
        ] = linear_rank_spread(
            background[
                "consistency_raw"
            ],
            low=BG_Y_LO,
            high=BG_Y_HI,
            pad_bottom=0.02,
            pad_top=0.03,
        )

        background[
            "consistency_plot"
        ] = add_deterministic_linear_jitter(
            background[
                "consistency_plot"
            ],
            background["lr_norm"],
            sigma=Y_JITTER_BG,
            offset=987,
            lower=Y_MIN,
            upper=BG_Y_HI + 4.0,
        )

    # ------------------------------------------------------------------
    # Right-low background
    # ------------------------------------------------------------------

    if len(right_low) > 0:
        right_low[
            "communication_score"
        ] = log_rank_spread(
            right_low[
                "communication_score_raw"
            ],
            low=RIGHT_LOW_X_LO,
            high=RIGHT_LOW_X_HI,
            pad_left=0.05,
            pad_right=0.08,
        )

        right_low[
            "communication_score"
        ] = add_deterministic_log_jitter(
            right_low[
                "communication_score"
            ],
            right_low["lr_norm"],
            sigma=0.055,
            offset=456,
            lower=RIGHT_LOW_X_LO,
            upper=RIGHT_LOW_X_HI,
        )

        right_low[
            "consistency_plot"
        ] = linear_rank_spread(
            right_low[
                "consistency_raw"
            ],
            low=RIGHT_LOW_Y_LO,
            high=RIGHT_LOW_Y_HI,
            pad_bottom=0.05,
            pad_top=0.05,
        )

        right_low[
            "consistency_plot"
        ] = add_deterministic_linear_jitter(
            right_low[
                "consistency_plot"
            ],
            right_low["lr_norm"],
            sigma=5.2,
            offset=654,
            lower=RIGHT_LOW_Y_LO,
            upper=RIGHT_LOW_Y_HI,
        )

    # ------------------------------------------------------------------
    # Highlight: rank by Top pathway rank, not independent mode score
    # ------------------------------------------------------------------

    if len(highlight) > 0:
        highlight = highlight.sort_values(
            [
                "highlight_rank",
                "communication_score_raw",
            ],
            ascending=[
                True,
                False,
            ],
        )

        # x 仍以各 mode 的 score 排序展开，
        # 但同一批 LR 身份保持一致。
        highlight[
            "communication_score"
        ] = log_rank_spread(
            highlight[
                "communication_score_raw"
            ],
            low=HI_X_LO,
            high=HI_X_HI,
            pad_left=0.01,
            pad_right=0.02,
        )

        highlight[
            "communication_score"
        ] = add_deterministic_log_jitter(
            highlight[
                "communication_score"
            ],
            highlight["lr_norm"],
            sigma=X_JITTER_SIGMA_HI,
            offset=321,
            lower=HI_X_LO,
            upper=HI_X_HI,
        )

        highlight[
            "consistency_plot"
        ] = linear_rank_spread(
            highlight[
                "consistency_raw"
            ],
            low=HI_Y_LO,
            high=HI_Y_HI,
            pad_bottom=0.01,
            pad_top=0.01,
        )

        highlight[
            "consistency_plot"
        ] = add_deterministic_linear_jitter(
            highlight[
                "consistency_plot"
            ],
            highlight["lr_norm"],
            sigma=Y_JITTER_HI,
            offset=789,
            lower=HI_Y_LO - 4.0,
            upper=Y_MAX,
        )

    output = pd.concat(
        [
            background,
            right_low,
            highlight,
        ],
        ignore_index=True,
        sort=False,
    )

    output["bubble_size"] = np.nan

    background_mask = (
        ~output["is_highlight"]
    )

    highlight_mask = (
        output["is_highlight"]
    )

    if background_mask.any():
        output.loc[
            background_mask,
            "bubble_size",
        ] = scale_to_range(
            output.loc[
                background_mask,
                "mean_strength",
            ],
            BG_SIZE_MIN,
            BG_SIZE_MAX,
        )

    if highlight_mask.any():
        output.loc[
            highlight_mask,
            "bubble_size",
        ] = scale_to_range(
            output.loc[
                highlight_mask,
                "mean_strength",
            ],
            HI_SIZE_MIN,
            HI_SIZE_MAX,
        )

    return output


# =============================================================================
# 12. CELL PSEUDOTIME
# =============================================================================

def load_cell_map() -> pd.DataFrame:

    if not CELL_MAP_CSV.exists():
        raise FileNotFoundError(
            CELL_MAP_CSV
        )

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

    return output[
        output["cluster"] != ""
    ].copy()


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
        role="original cell barcode",
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

    return output.drop_duplicates(
        "barcode"
    )


def project_cells_to_mode(
    cell_map: pd.DataFrame,
    original_cell_pt: pd.DataFrame | None,
    cluster_summary: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:

    ordered = (
        cluster_summary.sort_values(
            "t_hat_mean"
        )[
            [
                "cluster",
                "t_hat_mean",
            ]
        ]
        .drop_duplicates("cluster")
        .reset_index(drop=True)
    )

    clusters = ordered[
        "cluster"
    ].tolist()

    times = ordered[
        "t_hat_mean"
    ].to_numpy(float)

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
        np.clip(
            boundaries,
            0.0,
            1.0,
        )
    )

    work = cell_map[
        cell_map["cluster"]
        .isin(clusters)
    ].copy()

    if original_cell_pt is not None:
        work = work.merge(
            original_cell_pt,
            on="barcode",
            how="left",
        )
    else:
        work["original_cell_t"] = np.nan

    projected_parts = []

    for index, cluster in enumerate(
        clusters
    ):
        sub = work[
            work["cluster"]
            == cluster
        ].copy()

        if len(sub) == 0:
            continue

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

        width = upper - lower

        inner_lower = (
            lower
            + 0.05 * width
        )

        inner_upper = (
            upper
            - 0.05 * width
        )

        if (
            sub["original_cell_t"]
            .notna()
            .sum()
            >= 5
        ):
            relative = rank01(
                sub["original_cell_t"]
                .fillna(
                    sub[
                        "original_cell_t"
                    ].median()
                )
                .to_numpy(float)
            )

        elif (
            sub["umap1"]
            .notna()
            .sum()
            >= 5
        ):
            relative = rank01(
                sub["umap1"]
                .fillna(
                    sub["umap1"]
                    .median()
                )
                .to_numpy(float)
            )

        else:
            rng = np.random.default_rng(
                deterministic_seed(
                    cluster,
                    offset=1000,
                )
            )

            relative = rng.beta(
                2.0,
                2.0,
                size=len(sub),
            )

        sub["cell_t_proj"] = (
            inner_lower
            + relative
            * (
                inner_upper
                - inner_lower
            )
        )

        sub["mode"] = mode

        projected_parts.append(
            sub
        )

    return pd.concat(
        projected_parts,
        ignore_index=True,
    )


# =============================================================================
# 13. SCATTER PLOTTING
# =============================================================================

def make_label_positions(
    highlight: pd.DataFrame,
) -> dict[str, tuple[float, float, str]]:

    ordered = highlight.sort_values(
        [
            "consistency_plot",
            "communication_score",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    positions = {}

    left_offsets = [
        (-1.85, 8.0),
        (-1.75, 3.5),
        (-1.75, -4.0),
        (-1.65, -9.0),
        (-1.60, 12.0),
    ]

    right_offsets = [
        (1.75, 8.0),
        (1.75, 3.5),
        (1.70, -4.0),
        (1.65, -9.0),
        (1.60, 12.0),
    ]

    for index, row in ordered.iterrows():
        if index % 2 == 0:
            multiplier, delta_y = left_offsets[
                (index // 2)
                % len(left_offsets)
            ]

            alignment = "right"
        else:
            multiplier, delta_y = right_offsets[
                (index // 2)
                % len(right_offsets)
            ]

            alignment = "left"

        positions[
            row["lr_norm"]
        ] = (
            multiplier,
            delta_y,
            alignment,
        )

    return positions


def plot_lr_scatter(
    ax,
    summary: pd.DataFrame,
    title: str,
    title_color: str,
) -> None:

    background = summary[
        ~summary["is_highlight"]
    ].copy()

    highlight = summary[
        summary["is_highlight"]
    ].copy()

    for module, sub in background.groupby(
        "module",
        sort=False,
    ):
        ax.scatter(
            sub["communication_score"],
            sub["consistency_plot"],
            s=sub["bubble_size"],
            color=MODULE_COLORS.get(
                module,
                MODULE_COLORS["Other"],
            ),
            alpha=BG_ALPHA,
            edgecolors="white",
            linewidths=0.25,
            zorder=1,
        )

    for module, sub in highlight.groupby(
        "module",
        sort=False,
    ):
        ax.scatter(
            sub["communication_score"],
            sub["consistency_plot"],
            s=sub["bubble_size"],
            color=MODULE_COLORS.get(
                module,
                MODULE_COLORS["Other"],
            ),
            alpha=HI_ALPHA,
            edgecolors="#303030",
            linewidths=1.05,
            zorder=5,
        )

    label_positions = make_label_positions(
        highlight
    )

    for row in highlight.itertuples():
        multiplier, delta_y, alignment = (
            label_positions[
                row.lr_norm
            ]
        )

        x_value = float(
            row.communication_score
        )

        y_value = float(
            row.consistency_plot
        )

        if multiplier < 0:
            x_text = (
                x_value
                / abs(multiplier)
            )
        else:
            x_text = (
                x_value
                * multiplier
            )

        x_text = float(
            np.clip(
                x_text,
                X_MIN * 1.05,
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

        pathway_text = textwrap.fill(
            str(
                row.top10_pathway
            ),
            width=25,
        )

        label_text = (
            f"{row.lr_pretty}\n"
            f"[{pathway_text}]"
        )

        ax.annotate(
            label_text,
            xy=(
                x_value,
                y_value,
            ),
            xytext=(
                x_text,
                y_text,
            ),
            fontsize=7.2,
            ha=alignment,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color="#686868",
                shrinkA=2,
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
        "Communication score (au)"
    )

    ax.set_ylabel(
        "Consistency across states (%)"
    )

    ax.set_title(
        title,
        color=title_color,
        fontweight="bold",
        pad=10,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.65,
        alpha=0.20,
    )

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.5,
        alpha=0.10,
    )

    module_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6.5,
            label=module,
        )
        for module, color
        in MODULE_COLORS.items()
        if module in set(
            summary["module"]
        )
    ]

    ax.legend(
        handles=module_handles,
        title="Module",
        loc="lower right",
        frameon=True,
        fontsize=6.5,
        title_fontsize=7.0,
    )


# =============================================================================
# 14. VIOLIN PLOT
# =============================================================================

def plot_pseudotime_violin(
    ax,
    cell_table: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    title: str,
    title_color: str,
) -> None:

    ordered_clusters = (
        cluster_summary.sort_values(
            "t_hat_mean"
        )["cluster"]
        .tolist()
    )

    values_by_cluster = []

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

        if len(values) > 20000:
            rng = np.random.default_rng(
                deterministic_seed(
                    cluster,
                    offset=2000,
                )
            )

            values = rng.choice(
                values,
                size=20000,
                replace=False,
            )

        values_by_cluster.append(
            values
        )

    positions = np.arange(
        len(ordered_clusters)
    )

    violin = ax.violinplot(
        values_by_cluster,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )

    for body in violin["bodies"]:
        body.set_facecolor(
            "#C5D1E8"
        )

        body.set_edgecolor(
            "none"
        )

        body.set_alpha(
            0.90
        )

    means = np.asarray([
        np.mean(values)
        if len(values)
        else np.nan
        for values
        in values_by_cluster
    ])

    lower = np.asarray([
        np.quantile(
            values,
            0.05,
        )
        if len(values)
        else np.nan
        for values
        in values_by_cluster
    ])

    upper = np.asarray([
        np.quantile(
            values,
            0.95,
        )
        if len(values)
        else np.nan
        for values
        in values_by_cluster
    ])

    ax.vlines(
        positions,
        lower,
        upper,
        color="#2D6FA7",
        linewidth=2.4,
        zorder=3,
    )

    ax.scatter(
        positions,
        means,
        marker="P",
        s=58,
        color="#2D6FA7",
        zorder=4,
    )

    ax.scatter(
        positions,
        lower,
        marker="_",
        s=120,
        color="#2D6FA7",
        linewidth=2.4,
        zorder=4,
    )

    ax.scatter(
        positions,
        upper,
        marker="_",
        s=120,
        color="#2D6FA7",
        linewidth=2.4,
        zorder=4,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels([
        cluster[1:]
        if cluster.startswith("C")
        else cluster
        for cluster
        in ordered_clusters
    ])

    ax.set_xlabel(
        "Cluster"
    )

    ax.set_ylabel(
        "Cell projected pseudotime"
    )

    ax.set_ylim(
        -0.05,
        1.05,
    )

    ax.set_title(
        title,
        color=title_color,
        fontweight="bold",
        pad=10,
    )


# =============================================================================
# 15. MODE-SPECIFIC OVERVIEW
# =============================================================================

def save_figure(
    fig,
    stem: Path,
) -> None:

    fig.savefig(
        stem.with_suffix(
            ".png"
        ),
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    log(
        "[SAVE]",
        stem.with_suffix(
            ".png"
        ),
    )


def make_mode_overview(
    mode: str,
    cluster_summary: pd.DataFrame,
    cell_table: pd.DataFrame,
    scatter_summary: pd.DataFrame,
    output_dir: Path,
) -> None:

    if mode == "off":
        display = "MLTD-off"
        title_color = COL_OFF
    else:
        display = "MLTD-on"
        title_color = COL_ON

    fig = plt.figure(
        figsize=(
            15.3,
            6.4,
        )
    )

    grid = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[
            0.90,
            1.70,
        ],
        wspace=0.28,
    )

    ax_violin = fig.add_subplot(
        grid[0, 0]
    )

    ax_scatter = fig.add_subplot(
        grid[0, 1]
    )

    plot_pseudotime_violin(
        ax=ax_violin,
        cell_table=cell_table,
        cluster_summary=cluster_summary,
        title=(
            f"{display} cell projected pseudotime"
        ),
        title_color=title_color,
    )

    plot_lr_scatter(
        ax=ax_scatter,
        summary=scatter_summary,
        title=(
            f"{display} Top-pathway representative "
            f"ligand–receptor pairs"
        ),
        title_color=title_color,
    )

    panel_label(
        ax_violin,
        "I",
    )

    panel_label(
        ax_scatter,
        "J",
    )

    fig.suptitle(
        f"{display} mode-specific overview",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.11,
        top=0.88,
    )

    save_figure(
        fig,
        output_dir
        / f"{display.replace('-', '_')}_mode_specific_overview_top10LR",
    )


def make_standalone_scatter(
    mode: str,
    scatter_summary: pd.DataFrame,
    output_dir: Path,
) -> None:

    if mode == "off":
        display = "MLTD-off"
        title_color = COL_OFF
    else:
        display = "MLTD-on"
        title_color = COL_ON

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            6.7,
        )
    )

    plot_lr_scatter(
        ax=ax,
        summary=scatter_summary,
        title=(
            f"{display} Top-pathway representative "
            f"ligand–receptor pairs"
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
        right=0.98,
        bottom=0.12,
        top=0.90,
    )

    save_figure(
        fig,
        output_dir
        / f"{display.replace('-', '_')}_LR_scatter_top10_pathways",
    )


# =============================================================================
# 16. SHARED LR OFF-ON SHIFT FIGURE
# =============================================================================

def make_shared_lr_shift_figure(
    off_summary: pd.DataFrame,
    on_summary: pd.DataFrame,
    selected_table: pd.DataFrame,
    output_dir: Path,
) -> None:

    columns = [
        "lr_norm",
        "communication_score",
        "consistency_plot",
        "communication_score_raw",
        "consistency_raw",
    ]

    off = (
        off_summary[
            off_summary["is_highlight"]
        ][columns]
        .rename(
            columns={
                "communication_score": (
                    "communication_score_off"
                ),
                "consistency_plot": (
                    "consistency_plot_off"
                ),
                "communication_score_raw": (
                    "communication_score_raw_off"
                ),
                "consistency_raw": (
                    "consistency_raw_off"
                ),
            }
        )
    )

    on = (
        on_summary[
            on_summary["is_highlight"]
        ][columns]
        .rename(
            columns={
                "communication_score": (
                    "communication_score_on"
                ),
                "consistency_plot": (
                    "consistency_plot_on"
                ),
                "communication_score_raw": (
                    "communication_score_raw_on"
                ),
                "consistency_raw": (
                    "consistency_raw_on"
                ),
            }
        )
    )

    merged = (
        selected_table.merge(
            off,
            left_on="selected_lr_norm",
            right_on="lr_norm",
            how="left",
        )
        .drop(
            columns=[
                "lr_norm",
            ],
            errors="ignore",
        )
        .merge(
            on,
            left_on="selected_lr_norm",
            right_on="lr_norm",
            how="left",
        )
        .drop(
            columns=[
                "lr_norm",
            ],
            errors="ignore",
        )
    )

    merged.to_csv(
        output_dir
        / "shared_top10_pathway_LR_off_on_positions.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            6.8,
        )
    )

    for row in merged.itertuples():
        if (
            not np.isfinite(
                row.communication_score_off
            )
            or not np.isfinite(
                row.communication_score_on
            )
            or not np.isfinite(
                row.consistency_plot_off
            )
            or not np.isfinite(
                row.consistency_plot_on
            )
        ):
            continue

        ax.annotate(
            "",
            xy=(
                row.communication_score_on,
                row.consistency_plot_on,
            ),
            xytext=(
                row.communication_score_off,
                row.consistency_plot_off,
            ),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#8C8C8C",
                lw=1.0,
                mutation_scale=10,
                alpha=0.80,
            ),
            zorder=1,
        )

        module = classify_module(
            row.selected_ligand,
            row.selected_receptor,
        )

        color = MODULE_COLORS.get(
            module,
            MODULE_COLORS["Other"],
        )

        ax.scatter(
            row.communication_score_off,
            row.consistency_plot_off,
            s=95,
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )

        ax.scatter(
            row.communication_score_on,
            row.consistency_plot_on,
            s=115,
            facecolor=color,
            edgecolor="#303030",
            linewidth=0.9,
            zorder=4,
        )

        ax.text(
            row.communication_score_on
            * 1.06,
            row.consistency_plot_on,
            row.selected_lr_pretty,
            fontsize=7.5,
            ha="left",
            va="center",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        X_MIN,
        X_MAX * 1.20,
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX,
    )

    ax.set_xlabel(
        "Communication score (display scale)"
    )

    ax.set_ylabel(
        "Consistency across states (%)"
    )

    ax.set_title(
        "Top MLTD-gained pathway LR shifts: "
        "MLTD-off → MLTD-on",
        fontweight="bold",
        pad=10,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.20,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#666666",
            markersize=7,
            label="MLTD-off",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#777777",
            markeredgecolor="#303030",
            markersize=7,
            label="MLTD-on",
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
    )

    save_figure(
        fig,
        output_dir
        / "MLTD_off_on_shared_LR_shift",
    )


# =============================================================================
# 17. MAIN
# =============================================================================

def main() -> None:

    paired_result_dir = (
        locate_paired_result_dir()
    )

    mltd_cci_dir = (
        locate_mltd_cci_dir()
    )

    program_csv = first_matching_file(
        mltd_cci_dir,
        [
            f"{SAMPLE_ID}_same_universe_program_on_off_scores.csv",
            "*_same_universe_program_on_off_scores.csv",
        ],
    )

    top10_csv = first_matching_file(
        mltd_cci_dir,
        [
            f"{SAMPLE_ID}_top10_unique_MLTD_gain_pathways.csv",
            "*_top10_unique_MLTD_gain_pathways.csv",
        ],
    )

    output_dir = (
        paired_result_dir
        / OUTPUT_FOLDER_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "[PAIRED RESULT DIR]",
        paired_result_dir,
    )

    log(
        "[MLTD CCI DIR]",
        mltd_cci_dir,
    )

    log(
        "[PROGRAM CSV]",
        program_csv,
    )

    log(
        "[TOP10 CSV]",
        top10_csv,
    )

    log(
        "[OUTPUT DIR]",
        output_dir,
    )

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------

    off_cluster = load_cluster_summary(
        paired_result_dir
        / "MLTD_off_cluster_pseudotime_ensemble.csv",
        mode="off",
    )

    on_cluster = load_cluster_summary(
        paired_result_dir
        / "MLTD_on_cluster_pseudotime_ensemble.csv",
        mode="on",
    )

    programs = load_program_table(
        program_csv
    )

    top10 = load_top10_pathways(
        top10_csv
    )

    top10.to_csv(
        output_dir
        / "input_top10_unique_MLTD_gain_pathways.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Compute unpruned temporal scores for both modes
    # ------------------------------------------------------------------

    off_all_rows = compute_mode_temporal_rows(
        programs=programs,
        cluster_summary=off_cluster,
        mode="off",
    )

    on_all_rows = compute_mode_temporal_rows(
        programs=programs,
        cluster_summary=on_cluster,
        mode="on",
    )

    off_all_rows.to_csv(
        output_dir
        / "MLTD_off_LR_temporal_rows_all.csv",
        index=False,
    )

    on_all_rows.to_csv(
        output_dir
        / "MLTD_on_LR_temporal_rows_all.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Select one shared representative LR per Top pathway
    # ------------------------------------------------------------------

    combined_lr = build_combined_lr_score(
        off_rows=off_all_rows,
        on_rows=on_all_rows,
    )

    combined_lr.to_csv(
        output_dir
        / "all_LR_combined_off_on_temporal_scores.csv",
        index=False,
    )

    selected_table, missing_table = (
        select_shared_top10_pathway_lr(
            top10=top10,
            combined_lr=combined_lr,
        )
    )

    selected_table.to_csv(
        output_dir
        / "shared_top10_pathway_representative_LR.csv",
        index=False,
    )

    missing_table.to_csv(
        output_dir
        / "missing_top10_pathways.csv",
        index=False,
    )

    selected_lr = (
        selected_table[
            "selected_lr_norm"
        ]
        .astype(str)
        .tolist()
    )

    log(
        "[SHARED SELECTED LR]",
        selected_lr,
    )

    # ------------------------------------------------------------------
    # Mode-specific pruning + forced representative LR retention
    # ------------------------------------------------------------------

    off_plot_rows = prepare_mode_rows_for_plot(
        all_rows=off_all_rows,
        selected_lr=selected_lr,
        mode="off",
    )

    on_plot_rows = prepare_mode_rows_for_plot(
        all_rows=on_all_rows,
        selected_lr=selected_lr,
        mode="on",
    )

    off_plot_rows.to_csv(
        output_dir
        / "MLTD_off_LR_temporal_rows_for_plot.csv",
        index=False,
    )

    on_plot_rows.to_csv(
        output_dir
        / "MLTD_on_LR_temporal_rows_for_plot.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Scatter summaries
    # ------------------------------------------------------------------

    off_scatter = summarize_lr_for_scatter(
        rows_for_plot=off_plot_rows,
        selected_table=selected_table,
        mode="off",
    )

    on_scatter = summarize_lr_for_scatter(
        rows_for_plot=on_plot_rows,
        selected_table=selected_table,
        mode="on",
    )

    off_scatter.to_csv(
        output_dir
        / "MLTD_off_LR_scatter_summary.csv",
        index=False,
    )

    on_scatter.to_csv(
        output_dir
        / "MLTD_on_LR_scatter_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Cell pseudotime
    # ------------------------------------------------------------------

    cell_map = load_cell_map()
    original_cell_pt = load_original_cell_pt()

    off_cell = project_cells_to_mode(
        cell_map=cell_map,
        original_cell_pt=original_cell_pt,
        cluster_summary=off_cluster,
        mode="off",
    )

    on_cell = project_cells_to_mode(
        cell_map=cell_map,
        original_cell_pt=original_cell_pt,
        cluster_summary=on_cluster,
        mode="on",
    )

    off_cell.to_csv(
        output_dir
        / "MLTD_off_cell_projected_pseudotime.csv",
        index=False,
    )

    on_cell.to_csv(
        output_dir
        / "MLTD_on_cell_projected_pseudotime.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    make_mode_overview(
        mode="off",
        cluster_summary=off_cluster,
        cell_table=off_cell,
        scatter_summary=off_scatter,
        output_dir=output_dir,
    )

    make_mode_overview(
        mode="on",
        cluster_summary=on_cluster,
        cell_table=on_cell,
        scatter_summary=on_scatter,
        output_dir=output_dir,
    )

    make_standalone_scatter(
        mode="off",
        scatter_summary=off_scatter,
        output_dir=output_dir,
    )

    make_standalone_scatter(
        mode="on",
        scatter_summary=on_scatter,
        output_dir=output_dir,
    )

    make_shared_lr_shift_figure(
        off_summary=off_scatter,
        on_summary=on_scatter,
        selected_table=selected_table,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "paired_result_dir": str(
            paired_result_dir
        ),
        "mltd_cci_dir": str(
            mltd_cci_dir
        ),
        "program_csv": str(
            program_csv
        ),
        "top10_pathway_csv": str(
            top10_csv
        ),
        "output_dir": str(
            output_dir
        ),
        "highlight_definition": (
            "One shared representative LR per Top 10 "
            "unique MLTD-gained display pathway"
        ),
        "representative_priority": [
            "representative_ligand–representative_receptor from Top 10 table",
            "highest combined off+on temporal score within the same display pathway",
        ],
        "same_lr_used_in_off_and_on": True,
        "selected_lr_forced_through_pruning": (
            FORCE_KEEP_SELECTED_LR
        ),
        "temporal_alpha": (
            TEMPORAL_ALPHA
        ),
        "temporal_delta": (
            TEMPORAL_DELTA
        ),
        "background_relative_pruning": (
            POST_KEEP_RELATIVE_TO_MAX
        ),
        "consistency_definition": (
            f"{CONSISTENCY_COVERAGE_WEIGHT} × active receiver coverage "
            f"+ {CONSISTENCY_ENTROPY_WEIGHT} × normalized receiver entropy"
        ),
        "n_top_pathways_input": int(
            len(top10)
        ),
        "n_representative_lr_selected": int(
            len(selected_table)
        ),
        "n_missing_pathways": int(
            len(missing_table)
        ),
    }

    with open(
        output_dir
        / "run_manifest.json",
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

    print(
        "\n"
        + "=" * 100
    )

    print(
        "[SELECTED TOP-PATHWAY REPRESENTATIVE LR]"
    )

    print(
        selected_table[
            [
                "rank",
                "pathway",
                "selected_lr_pretty",
                "selection_source",
                "temporal_score_off",
                "temporal_score_on",
                "combined_temporal_score",
            ]
        ].to_string(
            index=False
        )
    )

    if len(missing_table) > 0:
        print(
            "\n[MISSING PATHWAYS]"
        )

        print(
            missing_table.to_string(
                index=False
            )
        )

    print(
        "\n[OFF HIGHLIGHT POSITIONS]"
    )

    print(
        off_scatter[
            off_scatter["is_highlight"]
        ][
            [
                "top10_pathway_rank",
                "top10_pathway",
                "lr_pretty",
                "communication_score_raw",
                "consistency_raw",
                "communication_score",
                "consistency_plot",
            ]
        ]
        .sort_values(
            "top10_pathway_rank"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n[ON HIGHLIGHT POSITIONS]"
    )

    print(
        on_scatter[
            on_scatter["is_highlight"]
        ][
            [
                "top10_pathway_rank",
                "top10_pathway",
                "lr_pretty",
                "communication_score_raw",
                "consistency_raw",
                "communication_score",
                "consistency_plot",
            ]
        ]
        .sort_values(
            "top10_pathway_rank"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n[DONE]"
    )

    print(
        "[OUTPUT DIR]",
        output_dir
    )


if __name__ == "__main__":
    main()
