# -*- coding: utf-8 -*-
"""
sox2_p1_p3_propagation_weighted_cci.py

目的
====
基于前面 SOX2 KO propagation P0-P3 结果，对 P1/P2/P3 分别计算
propagation-weighted cell-cell communication (CCI)。

核心设计
========
不是简单地“每个 P 时间点各自算一次静态 CCI”，而是做 frontier CCI：

P1: sender = P0 已激活区域       receiver = P1 新到达区域
P2: sender = P0+P1 已激活区域    receiver = P2 新到达区域
P3: sender = P0+P1+P2 已激活区域 receiver = P3 新到达区域

每条 LR 边的分数：
    LR_score(Pt)
      = mean_ligand(sender group)
        × mean_receptor(receiver group)
        × sender_propagation_activity
        × receiver_frontier_activity
        × LR_response_support

其中 propagation activity 来自前面 propagation 脚本输出的：
    cell_response_score / stage5_module_score / target_expr / propagation_time

输出
====
OUT_DIR 下：
- SOX2_ko_P1_P3_propagation_weighted_CCI_all_edges.csv
- SOX2_ko_P1_P3_top_edges.csv
- SOX2_ko_P1_P3_LR_by_P_heatmap_matrix.csv
- SOX2_ko_P1_P3_group_pair_by_P_heatmap_matrix.csv
- SOX2_ko_P1_P3_network.png/svg
- SOX2_ko_P1_P3_LR_heatmap.png/svg
- SOX2_ko_P1_P3_group_pair_heatmap.png/svg
- SOX2_ko_P1_P3_CCI_qc_summary.csv

运行前需要
==========
1) 前面 propagation 脚本已经跑完，并输出：
   SOX2_cell_level_propagation_response.csv
   SOX2_cluster_propagation_order.csv

2) CA_all 表达矩阵与基因名存在：
   CA_all.X_rna_log1p_norm_dense.npy
   CA_all.rna_meta.npz
   genes.tsv

3) Stage5 baseline / KO node state 存在，用于 LR_response_support。
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

TARGET_GENE = "SOX2"
MODE = "ko"
TARGET_GENE_UPPER = TARGET_GENE.upper()

BASE = get_project_root()

EXPLOIT_ROOT_CANDIDATES = [
    BASE / "LLM-Worldword" / "Exploit-HNSCC",
    BASE / "LLM-WorldWord" / "Exploit-HNSCC",
]
EXPLOIT_HNSCC_ROOT = None
for _root in EXPLOIT_ROOT_CANDIDATES:
    if (_root / "processed_CA" / "RNA_CA_all").exists():
        EXPLOIT_HNSCC_ROOT = _root
        break
if EXPLOIT_HNSCC_ROOT is None:
    raise FileNotFoundError(
        "Cannot find Exploit-HNSCC processed_CA/RNA_CA_all under:\n"
        + "\n".join(str(x) for x in EXPLOIT_ROOT_CANDIDATES)
    )

RNA_DIR = EXPLOIT_HNSCC_ROOT / "processed_CA" / "RNA_CA_all"
ASSET_DIR = EXPLOIT_HNSCC_ROOT / "assets_CA"
LOG1P_NPY = RNA_DIR / "CA_all.X_rna_log1p_norm_dense.npy"
META_NPZ = RNA_DIR / "CA_all.rna_meta.npz"
GENES_TSV = ASSET_DIR / "genes.tsv"

RUN_STAGEB_DATASET_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_rna_only_stageB_clusterCorrection_HNSCC_CAall_only"
    / "HNSCC-CA_all"
)
PT_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "CA_all_single_cci_newLeiden"
    / "communication_pseudotime_clusterGNN_LTC"
)
STAGE5_DIR = PT_DIR / "five_stage_network_viz_out"

# 手动指定 propagation 输出目录；None 时自动找最新的 SOX2_cell_level_propagation_response.csv
FORCE_PROP_DIR: Optional[str] = None

# 输出目录
OUT_DIR = STAGE5_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_propagation_weighted_CCI"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 如果 propagation CSV 里没有 display_propagation_stage，则按 propagation_rank 推断 P0-P3
N_DISPLAY_STAGES = 4
DISPLAY_SKIP_INITIAL_RANKS = 1
TARGET_P_STAGES = [1, 2, 3]

# CCI 计算设置
MIN_CELLS_PER_GROUP = 10
MIN_EXPR_MEAN = 0.02
MIN_EXPR_PCT = 0.03
TOP_EDGES_PER_P = 28
TOP_LR_FOR_HEATMAP = 35
TOP_GROUP_PAIRS_FOR_HEATMAP = 35

# sender / receiver 权重
SENDER_ACTIVITY_FLOOR = 0.35
RECEIVER_ACTIVITY_FLOOR = 0.35
PROP_ACTIVITY_RESPONSE_WEIGHT = 0.50
PROP_ACTIVITY_MODULE_WEIGHT = 0.30
PROP_ACTIVITY_TARGET_WEIGHT = 0.10
PROP_ACTIVITY_TIME_WEIGHT = 0.10

# LR response support 权重
LR_SUPPORT_STRENGTH = 1.15
STAGE5_DIRECT_SUPPORT_WEIGHT = 0.75
MALIGNANT_TOPGENE_SUPPORT_WEIGHT = 0.35

# 是否额外读取 malignant subgroup top1000 作为 LR support
USE_MALIGNANT_TOPGENE_SUPPORT = True
MALIGNANT_OUT_DIR = (
    STAGE5_DIR
    / f"malignant_balanced3_subgroups_{TARGET_GENE}_{MODE}_top1000_tripleLayerTop10_weightedPathwayEnrichment_plots"
)

# 可选本地 LR 数据库；没有就用内置 curated fallback
LR_DB_CSV_CANDIDATES = [
    STAGE5_DIR / "lr_db.csv",
    STAGE5_DIR / "LR_db.csv",
    PT_DIR / "lr_db.csv",
    PT_DIR / "LR_db.csv",
    EXPLOIT_HNSCC_ROOT / "lr_db.csv",
    BASE / "networks" / "lr_db.csv",
    BASE / "networks" / "cellchat_lr_db.csv",
]

# 作图设置
DPI = 340
SEED = 42
np.random.seed(SEED)

NATURE_RED = "#B2182B"
NATURE_BLUE = "#2166AC"
NATURE_GREEN = "#4D9221"
NATURE_PURPLE = "#6A51A3"
NATURE_ORANGE = "#E6953B"
GREY = "#D0D0D0"
DARK_GREY = "#4D4D4D"

P_STAGE_COLORS = {
    1: "#B2182B",
    2: "#2166AC",
    3: "#4D9221",
}

# Network nodes: use solid colored points + legend, not hollow nodes with text labels.
# Keep colors stable across P1/P2/P3 panels.
CELLTYPE_NODE_COLORS = {
    "B cells": "#1F77B4",
    "B.cells": "#1F77B4",
    "Dendritic cells": "#FF7F0E",
    "Dendritic.cells": "#FF7F0E",
    "Endothelial cells": "#2CA02C",
    "Endothelial.cells": "#2CA02C",
    "Fibroblasts": "#D62728",
    "Macrophages": "#9467BD",
    "Malignant cells": "#8C564B",
    "Malignant.cells": "#8C564B",
    "Mast cells": "#E377C2",
    "Mast.cells": "#E377C2",
    "Myocytes": "#7F7F7F",
    "T cells": "#17BECF",
    "T.cells": "#17BECF",
}
FALLBACK_NODE_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
]
NETWORK_NODE_SIZE = 210
NETWORK_NODE_EDGE_COLOR = "white"
NETWORK_NODE_EDGE_WIDTH = 0.55
NETWORK_SHOW_NODE_TEXT = False
NETWORK_LEGEND_MARKER_SIZE = 6.0

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["svg.fonttype"] = "none"


# ============================================================
# BASIC HELPERS
# ============================================================

def log(*args):
    print(*args, flush=True)


def is_target_gene(g: str) -> bool:
    return str(g).upper() == TARGET_GENE_UPPER


def safe_name(x: str, n: int = 160) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n]


def norm_cell_id(x: str) -> str:
    x = str(x).strip()
    if "::" in x:
        x = x.split("::", 1)[-1]
    if ":" in x:
        x = x.rsplit(":", 1)[-1]
    return x.replace("-", ".")


def minmax_1d(x: np.ndarray, constant_value: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if int(finite.sum()) == 0:
        return out
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if hi - lo < 1e-12:
        out[finite] = constant_value
    else:
        out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if int(finite.sum()) == 0:
        return out
    mu = float(np.nanmean(x[finite]))
    sd = float(np.nanstd(x[finite]))
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    out[finite] = (x[finite] - mu) / sd
    return out


def shorten_label(x: str, n: int = 23) -> str:
    x = str(x).strip().replace("_", " ").replace(".", " ")
    x = " ".join(x.split())
    if x == "" or x.lower() in {"nan", "none"}:
        x = "Unknown"
    if len(x) > n:
        return x[: n - 1] + "…"
    return x


def sort_labels(xs: List[str]) -> List[str]:
    def _key(v):
        s = str(v)
        if s.startswith("P") and s[1:].isdigit():
            return (0, int(s[1:]))
        if s.startswith("C") and s[1:].isdigit():
            return (0, int(s[1:]))
        if s.isdigit():
            return (0, int(s))
        return (1, s)
    return sorted([str(x) for x in xs], key=_key)


def save_fig(fig: plt.Figure, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("[SAVE]", out_base.with_suffix(".png"))
    log("[SAVE]", out_base.with_suffix(".svg"))


# ============================================================
# LOAD GENES / META / EXPRESSION
# ============================================================

def read_gene_names() -> List[str]:
    if not GENES_TSV.exists():
        raise FileNotFoundError(f"Missing genes.tsv: {GENES_TSV}")
    df = pd.read_csv(GENES_TSV, sep="\t")
    for c in ["gene", "gene_name", "Gene", "name", "symbol"]:
        if c in df.columns:
            return df[c].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def load_meta_cell_ids() -> np.ndarray:
    if not META_NPZ.exists():
        raise FileNotFoundError(f"Missing meta npz: {META_NPZ}")
    z = np.load(META_NPZ, allow_pickle=True)
    if "cell_ids" in z.files:
        return z["cell_ids"].astype(str)
    if "metadata_columns" in z.files and "metadata_table" in z.files:
        cols = z["metadata_columns"].astype(str).tolist()
        tab = z["metadata_table"]
        meta = pd.DataFrame(tab, columns=cols)
        for c in ["cell_id", "barcode_counts", "barcode"]:
            if c in meta.columns:
                return meta[c].astype(str).values
    raise KeyError(f"Cannot recover cell_ids from {META_NPZ}. files={z.files}")


def build_expression_row_matcher() -> Tuple[Dict[str, int], Dict[str, int]]:
    cell_ids = load_meta_cell_ids()
    exact_map = {str(x): i for i, x in enumerate(cell_ids)}

    norm_all = [norm_cell_id(x) for x in cell_ids]
    vc = pd.Series(norm_all).value_counts()
    nondup_norm = set(vc[vc == 1].index.astype(str).tolist())
    norm_map = {
        norm_cell_id(cid): i
        for i, cid in enumerate(cell_ids)
        if norm_cell_id(cid) in nondup_norm
    }
    return exact_map, norm_map


def match_cell_df_to_expression_rows(cell_df: pd.DataFrame) -> np.ndarray:
    exact_map, norm_map = build_expression_row_matcher()
    candidate_cols = ["cell_id", "barcode", "barcode_counts", "barcode_meta", "CellID"]
    rows = []
    missing = []
    for i, r in cell_df.iterrows():
        candidates = []
        for c in candidate_cols:
            if c in cell_df.columns:
                candidates.append(str(r[c]))
        candidates.append(str(i))
        candidates += [norm_cell_id(x) for x in list(candidates)]
        hit = None
        for x in candidates:
            if x in exact_map:
                hit = exact_map[x]
                break
            xn = norm_cell_id(x)
            if xn in norm_map:
                hit = norm_map[xn]
                break
        if hit is None:
            missing.append(str(r.get("cell_id", i)))
        else:
            rows.append(hit)
    if missing:
        raise RuntimeError(
            f"Cannot match {len(missing)} cells to expression rows. Examples: {missing[:8]}"
        )
    return np.asarray(rows, dtype=np.int64)


def load_expression(row_idx: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    if not LOG1P_NPY.exists():
        raise FileNotFoundError(f"Missing expression matrix: {LOG1P_NPY}")
    X = np.load(LOG1P_NPY, mmap_mode="r")
    return np.asarray(X[np.ix_(row_idx.astype(int), gene_idx.astype(int))], dtype=np.float32)


# ============================================================
# PROPAGATION OUTPUT
# ============================================================

def find_propagation_dir() -> Path:
    if FORCE_PROP_DIR is not None and str(FORCE_PROP_DIR).strip():
        d = Path(FORCE_PROP_DIR)
        if not d.exists():
            raise FileNotFoundError(f"FORCE_PROP_DIR does not exist: {d}")
        return d

    candidates = []
    for p in STAGE5_DIR.rglob(f"{TARGET_GENE}_cell_level_propagation_response.csv"):
        if p.is_file():
            candidates.append(p.parent)
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {TARGET_GENE}_cell_level_propagation_response.csv under:\n{STAGE5_DIR}\n"
            "请先运行 P0-P3 propagation UMAP 脚本。"
        )
    candidates = sorted(
        candidates,
        key=lambda d: (d / f"{TARGET_GENE}_cell_level_propagation_response.csv").stat().st_mtime,
        reverse=True,
    )
    log("[PROP] using propagation dir =", candidates[0])
    return candidates[0]


def derive_display_stage_from_rank(prop_df: pd.DataFrame) -> pd.DataFrame:
    prop_df = prop_df.copy()
    if "display_propagation_stage" in prop_df.columns:
        prop_df["display_propagation_stage"] = pd.to_numeric(
            prop_df["display_propagation_stage"], errors="coerce"
        )
        return prop_df

    if "propagation_rank" not in prop_df.columns:
        raise KeyError("propagation order CSV needs display_propagation_stage or propagation_rank")

    rank = pd.to_numeric(prop_df["propagation_rank"], errors="coerce")
    eligible = rank >= DISPLAY_SKIP_INITIAL_RANKS
    if int(eligible.sum()) == 0:
        eligible = rank >= np.nanmin(rank.values)
    rmin = int(np.nanmin(rank[eligible]))
    rmax = int(np.nanmax(rank[eligible]))
    span = max(1, rmax - rmin + 1)

    stage = np.full(len(prop_df), np.nan, dtype=float)
    for i, r in enumerate(rank.values):
        if not np.isfinite(r) or r < rmin:
            continue
        s = int(np.floor((float(r) - rmin) * N_DISPLAY_STAGES / span))
        s = max(0, min(N_DISPLAY_STAGES - 1, s))
        stage[i] = s
    prop_df["display_propagation_stage"] = stage
    prop_df["display_stage_derived_from_rank"] = 1
    return prop_df


def load_propagation_tables() -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    prop_dir = find_propagation_dir()
    cell_csv = prop_dir / f"{TARGET_GENE}_cell_level_propagation_response.csv"
    prop_csv = prop_dir / f"{TARGET_GENE}_cluster_propagation_order.csv"

    if not cell_csv.exists():
        raise FileNotFoundError(f"Missing: {cell_csv}")
    if not prop_csv.exists():
        raise FileNotFoundError(f"Missing: {prop_csv}")

    cell_df = pd.read_csv(cell_csv, low_memory=False)
    prop_df = pd.read_csv(prop_csv, low_memory=False)

    if "cell_id" not in cell_df.columns:
        raise KeyError(f"{cell_csv} must contain cell_id")
    if "cluster_label" not in cell_df.columns:
        raise KeyError(f"{cell_csv} must contain cluster_label")
    if "cluster_label" not in prop_df.columns:
        raise KeyError(f"{prop_csv} must contain cluster_label")

    cell_df["cell_id"] = cell_df["cell_id"].astype(str)
    cell_df["cluster_label"] = cell_df["cluster_label"].astype(str)
    prop_df["cluster_label"] = prop_df["cluster_label"].astype(str)
    prop_df = derive_display_stage_from_rank(prop_df)

    # map display stage into cell_df if not already present
    if "display_propagation_stage" not in cell_df.columns:
        stage_map = prop_df.set_index("cluster_label")["display_propagation_stage"].to_dict()
        cell_df["display_propagation_stage"] = cell_df["cluster_label"].map(stage_map)
    else:
        cell_df["display_propagation_stage"] = pd.to_numeric(
            cell_df["display_propagation_stage"], errors="coerce"
        )

    # group/celltype label
    if "major_celltype" in prop_df.columns:
        group_map = prop_df.set_index("cluster_label")["major_celltype"].astype(str).to_dict()
    elif "xtick_label" in prop_df.columns:
        group_map = prop_df.set_index("cluster_label")["xtick_label"].astype(str).to_dict()
    else:
        group_map = {cl: f"C{cl}" for cl in prop_df["cluster_label"].astype(str).tolist()}

    cell_df["cci_group"] = cell_df["cluster_label"].map(group_map).astype(str)
    cell_df["cci_group"] = cell_df["cci_group"].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    cell_df["cci_group_short"] = cell_df["cci_group"].map(shorten_label)

    # activity features
    for c in ["cell_response_score", "stage5_module_score", "target_expr", "propagation_time"]:
        if c not in cell_df.columns:
            cell_df[c] = 0.0
        cell_df[c] = pd.to_numeric(cell_df[c], errors="coerce").fillna(0.0)

    resp = minmax_1d(cell_df["cell_response_score"].values, constant_value=0.0)
    mod = minmax_1d(cell_df["stage5_module_score"].values, constant_value=0.0)
    tgt = minmax_1d(cell_df["target_expr"].values, constant_value=0.0)
    tim = minmax_1d(cell_df["propagation_time"].values, constant_value=0.0)

    activity = (
        PROP_ACTIVITY_RESPONSE_WEIGHT * resp
        + PROP_ACTIVITY_MODULE_WEIGHT * mod
        + PROP_ACTIVITY_TARGET_WEIGHT * tgt
        + PROP_ACTIVITY_TIME_WEIGHT * tim
    )
    cell_df["propagation_activity"] = minmax_1d(activity, constant_value=0.0)

    # remove skipped/no-stage cells
    cell_df = cell_df[np.isfinite(pd.to_numeric(cell_df["display_propagation_stage"], errors="coerce"))].copy()
    cell_df["display_propagation_stage"] = cell_df["display_propagation_stage"].astype(int)

    log("[PROP] cells with display stages:")
    log(cell_df["display_propagation_stage"].value_counts().sort_index().to_string())
    log("[PROP] cci groups:")
    log(cell_df["cci_group_short"].value_counts().to_string())

    return cell_df, prop_df, prop_dir


# ============================================================
# LR DATABASE
# ============================================================

def fallback_lr_database() -> pd.DataFrame:
    """Curated fallback LR list focused on HNSCC / tumor microenvironment / SOX2-KO relevant axes."""
    rows = [
        # ECM / adhesion / invasion
        ("SPP1", "CD44", "SPP1-CD44"),
        ("SPP1", "ITGAV", "SPP1-integrin"),
        ("SPP1", "ITGB1", "SPP1-integrin"),
        ("FN1", "ITGB1", "ECM-integrin"),
        ("FN1", "ITGA5", "ECM-integrin"),
        ("FN1", "ITGAV", "ECM-integrin"),
        ("LAMB1", "ITGB1", "laminin-integrin"),
        ("LAMB1", "ITGA6", "laminin-integrin"),
        ("LAMC1", "ITGB1", "laminin-integrin"),
        ("LAMC1", "ITGA6", "laminin-integrin"),
        ("COL1A1", "ITGA2", "collagen-integrin"),
        ("COL1A1", "ITGB1", "collagen-integrin"),
        ("COL1A2", "ITGA2", "collagen-integrin"),
        ("COL1A2", "ITGB1", "collagen-integrin"),
        ("THBS1", "CD47", "THBS-CD47"),
        ("THBS1", "ITGB1", "THBS-integrin"),
        ("LGALS1", "CD45", "galectin"),
        ("LGALS1", "ITGB1", "galectin-integrin"),
        ("LGALS3", "CD44", "galectin-CD44"),
        # RTK / growth factor
        ("AREG", "EGFR", "EGFR"),
        ("EREG", "EGFR", "EGFR"),
        ("HBEGF", "EGFR", "EGFR"),
        ("TGFA", "EGFR", "EGFR"),
        ("NRG1", "ERBB3", "ERBB"),
        ("NRG1", "ERBB4", "ERBB"),
        ("FGF2", "FGFR1", "FGF-FGFR"),
        ("FGF2", "FGFR2", "FGF-FGFR"),
        ("FGF7", "FGFR2", "FGF-FGFR"),
        ("FGF10", "FGFR2", "FGF-FGFR"),
        ("PDGFA", "PDGFRA", "PDGF-PDGFR"),
        ("PDGFA", "PDGFRB", "PDGF-PDGFR"),
        ("PDGFB", "PDGFRA", "PDGF-PDGFR"),
        ("PDGFB", "PDGFRB", "PDGF-PDGFR"),
        ("IGF1", "IGF1R", "IGF"),
        ("IGF2", "IGF1R", "IGF"),
        ("HGF", "MET", "HGF-MET"),
        ("GAS6", "AXL", "GAS6-AXL"),
        ("KITLG", "KIT", "KIT"),
        # angiogenesis / endothelial / EndMT related
        ("VEGFA", "KDR", "VEGF"),
        ("VEGFA", "FLT1", "VEGF"),
        ("VEGFA", "NRP1", "VEGF"),
        ("VEGFB", "FLT1", "VEGF"),
        ("VEGFC", "FLT4", "VEGF"),
        ("PGF", "FLT1", "VEGF"),
        ("ANGPT1", "TEK", "ANGPT-TEK"),
        ("ANGPT2", "TEK", "ANGPT-TEK"),
        ("EFNB2", "EPHB4", "Ephrin"),
        ("EFNA1", "EPHA2", "Ephrin"),
        ("EDN1", "EDNRA", "Endothelin"),
        ("EDN1", "EDNRB", "Endothelin"),
        ("TGFB1", "TGFBR1", "TGF-beta"),
        ("TGFB1", "TGFBR2", "TGF-beta"),
        ("TGFB2", "TGFBR1", "TGF-beta"),
        ("TGFB2", "TGFBR2", "TGF-beta"),
        # immune / cytokine / chemokine
        ("IL6", "IL6R", "IL6"),
        ("IL6", "IL6ST", "IL6"),
        ("IL11", "IL11RA", "IL11"),
        ("LIF", "LIFR", "LIF"),
        ("OSM", "OSMR", "OSM"),
        ("CXCL8", "CXCR1", "CXCL"),
        ("CXCL8", "CXCR2", "CXCL"),
        ("CXCL12", "CXCR4", "CXCL"),
        ("CXCL12", "ACKR3", "CXCL"),
        ("CXCL9", "CXCR3", "CXCL"),
        ("CXCL10", "CXCR3", "CXCL"),
        ("CCL2", "CCR2", "CCL"),
        ("CCL3", "CCR1", "CCL"),
        ("CCL4", "CCR5", "CCL"),
        ("CCL5", "CCR5", "CCL"),
        ("CCL5", "ACKR1", "CCL"),
        ("CCL16", "HRH4", "CCL-HRH4"),
        ("TNF", "TNFRSF1A", "TNF"),
        ("TNF", "TNFRSF1B", "TNF"),
        ("FASLG", "FAS", "apoptosis"),
        ("CD274", "PDCD1", "PD-L1"),
        ("LGALS9", "HAVCR2", "TIM3"),
        # developmental / SOX2 axis
        ("JAG1", "NOTCH1", "Notch"),
        ("JAG1", "NOTCH2", "Notch"),
        ("JAG2", "NOTCH1", "Notch"),
        ("DLL1", "NOTCH1", "Notch"),
        ("DLL3", "NOTCH1", "Notch"),
        ("WNT5A", "FZD2", "WNT"),
        ("WNT5A", "FZD5", "WNT"),
        ("WNT7B", "FZD5", "WNT"),
        ("BMP2", "BMPR1A", "BMP"),
        ("BMP4", "BMPR1A", "BMP"),
        ("SEMA3E", "PLXND1", "Semaphorin"),
        ("SEMA3E", "PLXNA3", "Semaphorin"),
        ("SEMA4D", "PLXNB1", "Semaphorin"),
        ("SEMA6B", "PLXNA2", "Semaphorin"),
    ]
    return pd.DataFrame(rows, columns=["ligand", "receptor", "pathway"])


def standardize_lr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {str(c).lower(): c for c in df.columns}

    lig_col = None
    rec_col = None
    path_col = None
    for c in ["ligand", "ligand_symbol", "source_gene", "ligand_gene", "LIGAND"]:
        if c.lower() in lower:
            lig_col = lower[c.lower()]
            break
    for c in ["receptor", "receptor_symbol", "target_gene", "receptor_gene", "RECEPTOR"]:
        if c.lower() in lower:
            rec_col = lower[c.lower()]
            break
    for c in ["pathway", "pathway_name", "annotation", "category", "interaction_name"]:
        if c.lower() in lower:
            path_col = lower[c.lower()]
            break

    if lig_col is None or rec_col is None:
        raise ValueError(f"Cannot identify ligand/receptor columns. Available={list(df.columns)}")

    out = pd.DataFrame({
        "ligand": df[lig_col].astype(str),
        "receptor": df[rec_col].astype(str),
    })
    if path_col is not None:
        out["pathway"] = df[path_col].astype(str)
    else:
        out["pathway"] = out["ligand"] + "-" + out["receptor"]

    # remove complexes like A_B if not directly present; keep first gene if comma separated
    out["ligand"] = out["ligand"].str.split("[,;|]").str[0].str.strip()
    out["receptor"] = out["receptor"].str.split("[,;|]").str[0].str.strip()
    out = out[(out["ligand"] != "") & (out["receptor"] != "")].copy()
    out = out.drop_duplicates(["ligand", "receptor", "pathway"]).reset_index(drop=True)
    return out


def load_lr_database(genes_all: List[str]) -> pd.DataFrame:
    gene_upper = {str(g).upper(): g for g in genes_all}
    chosen = None
    for p in LR_DB_CSV_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            chosen = p
            break

    if chosen is not None:
        try:
            lr = standardize_lr_columns(pd.read_csv(chosen, low_memory=False))
            log("[LR] using local LR db =", chosen)
        except Exception as e:
            log("[LR][WARN] failed to read local LR db:", chosen, repr(e))
            lr = fallback_lr_database()
            log("[LR] using fallback curated LR db")
    else:
        lr = fallback_lr_database()
        log("[LR] using fallback curated LR db")

    # canonical gene symbol mapping by uppercase
    lr["ligand_upper"] = lr["ligand"].astype(str).str.upper()
    lr["receptor_upper"] = lr["receptor"].astype(str).str.upper()
    lr = lr[lr["ligand_upper"].isin(gene_upper.keys()) & lr["receptor_upper"].isin(gene_upper.keys())].copy()
    lr["ligand"] = lr["ligand_upper"].map(gene_upper)
    lr["receptor"] = lr["receptor_upper"].map(gene_upper)
    lr = lr[~lr["ligand"].map(is_target_gene) & ~lr["receptor"].map(is_target_gene)].copy()
    lr = lr.drop_duplicates(["ligand", "receptor"]).reset_index(drop=True)
    lr["lr_pair"] = lr["ligand"].astype(str) + "→" + lr["receptor"].astype(str)

    if len(lr) == 0:
        raise RuntimeError("No LR pairs left after matching to genes.tsv")
    log("[LR] usable LR pairs =", len(lr))
    return lr[["ligand", "receptor", "pathway", "lr_pair"]].copy()


# ============================================================
# RESPONSE SUPPORT
# ============================================================

def load_stage5_gene_support(genes_all: List[str]) -> pd.DataFrame:
    gene_upper_to_gene = {g.upper(): g for g in genes_all}
    out = pd.DataFrame({"gene": genes_all, "stage5_direct_support": np.zeros(len(genes_all), dtype=float)})
    baseline = STAGE5_DIR / f"{TARGET_GENE}_stage5_baseline_node_state.csv"
    pert = STAGE5_DIR / f"{TARGET_GENE}_stage5_{MODE}_node_state.csv"
    if not baseline.exists() or not pert.exists():
        log("[SUPPORT][WARN] Stage5 node states not found. direct support = 0")
        return out
    b = pd.read_csv(baseline)
    p = pd.read_csv(pert)
    if not {"node", "state_final"}.issubset(b.columns) or not {"node", "state_final"}.issubset(p.columns):
        log("[SUPPORT][WARN] Stage5 files missing node/state_final. direct support = 0")
        return out
    df = (
        b[["node", "state_final"]].rename(columns={"state_final": "state_baseline"})
        .merge(p[["node", "state_final"]].rename(columns={"state_final": "state_perturbed"}), on="node", how="inner")
    )
    df["node"] = df["node"].astype(str)
    df = df[~df["node"].map(is_target_gene)].copy()
    df["state_baseline"] = pd.to_numeric(df["state_baseline"], errors="coerce")
    df["state_perturbed"] = pd.to_numeric(df["state_perturbed"], errors="coerce")
    df = df.dropna(subset=["state_baseline", "state_perturbed"]).copy()
    if MODE.lower() == "ko":
        impact = np.maximum(0.0, df["state_baseline"].values - df["state_perturbed"].values)
    elif MODE.lower() == "oe":
        impact = np.maximum(0.0, df["state_perturbed"].values - df["state_baseline"].values)
    else:
        impact = np.abs(df["state_perturbed"].values - df["state_baseline"].values)
    df["support"] = minmax_1d(impact, constant_value=0.0)

    support_map = {}
    for _, r in df.iterrows():
        gu = str(r["node"]).upper()
        if gu in gene_upper_to_gene:
            g = gene_upper_to_gene[gu]
            support_map[g] = max(float(support_map.get(g, 0.0)), float(r["support"]))
    out["stage5_direct_support"] = [support_map.get(g, 0.0) for g in genes_all]
    log("[SUPPORT] Stage5 direct support genes >0 =", int((out["stage5_direct_support"] > 0).sum()))
    return out


def find_malignant_topgene_file() -> Optional[Path]:
    if not MALIGNANT_OUT_DIR.exists():
        return None
    exact = MALIGNANT_OUT_DIR / "malignant_subgroup_top1000_response_genes.csv"
    if exact.exists():
        return exact
    cands = sorted(MALIGNANT_OUT_DIR.glob("malignant_subgroup_top*_response_genes.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_malignant_topgene_support(genes_all: List[str]) -> pd.DataFrame:
    out = pd.DataFrame({"gene": genes_all, "malignant_topgene_support": np.zeros(len(genes_all), dtype=float)})
    if not USE_MALIGNANT_TOPGENE_SUPPORT:
        return out
    p = find_malignant_topgene_file()
    if p is None:
        log("[SUPPORT][WARN] malignant topgene file not found. malignant support = 0")
        return out
    df = pd.read_csv(p, low_memory=False)
    if "gene" not in df.columns:
        log("[SUPPORT][WARN] malignant topgene file has no gene column. malignant support = 0")
        return out
    gene_upper_to_gene = {g.upper(): g for g in genes_all}
    df["gene"] = df["gene"].astype(str)

    weight_col = None
    for c in [
        "response_weight", "final_score_raw", "selection_score", "impact_eff_with_prior_floor",
        "axis_selection_score", "lane_specific_response", "lane_natural_sox2_axis", "rank"
    ]:
        if c in df.columns:
            weight_col = c
            break
    if weight_col is None:
        df["_w"] = 1.0
    elif weight_col == "rank":
        rk = pd.to_numeric(df["rank"], errors="coerce").fillna(999999).values.astype(float)
        df["_w"] = 1.0 / np.sqrt(np.maximum(rk, 1.0))
    else:
        df["_w"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).values.astype(float)
    df["_w01"] = minmax_1d(df["_w"].values, constant_value=0.0)

    support_map = {}
    for _, r in df.iterrows():
        gu = str(r["gene"]).upper()
        if gu in gene_upper_to_gene and not is_target_gene(gene_upper_to_gene[gu]):
            g = gene_upper_to_gene[gu]
            support_map[g] = max(float(support_map.get(g, 0.0)), float(r["_w01"]))
    out["malignant_topgene_support"] = [support_map.get(g, 0.0) for g in genes_all]
    log("[SUPPORT] malignant topgene support genes >0 =", int((out["malignant_topgene_support"] > 0).sum()), "file=", p)
    return out


def build_gene_support(genes_all: List[str]) -> pd.DataFrame:
    s1 = load_stage5_gene_support(genes_all)
    s2 = load_malignant_topgene_support(genes_all)
    df = s1.merge(s2, on="gene", how="left")
    df["malignant_topgene_support"] = pd.to_numeric(df["malignant_topgene_support"], errors="coerce").fillna(0.0)
    df["combined_gene_support"] = np.clip(
        STAGE5_DIRECT_SUPPORT_WEIGHT * df["stage5_direct_support"].values.astype(float)
        + MALIGNANT_TOPGENE_SUPPORT_WEIGHT * df["malignant_topgene_support"].values.astype(float),
        0.0,
        1.0,
    )
    return df


# ============================================================
# CCI COMPUTATION
# ============================================================

def prepare_expression_for_lr(cell_df: pd.DataFrame, lr_df: pd.DataFrame, genes_all: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    gene_to_idx = {g: i for i, g in enumerate(genes_all)}
    lr_genes = sorted(set(lr_df["ligand"].astype(str).tolist()) | set(lr_df["receptor"].astype(str).tolist()))
    lr_genes = [g for g in lr_genes if g in gene_to_idx]
    if len(lr_genes) == 0:
        raise RuntimeError("No LR genes available in expression matrix.")
    gene_idx = np.asarray([gene_to_idx[g] for g in lr_genes], dtype=np.int64)

    row_idx = match_cell_df_to_expression_rows(cell_df)
    X = load_expression(row_idx, gene_idx)
    log("[EXPR] loaded LR expression matrix:", X.shape)
    return cell_df.copy(), X, lr_genes


def compute_group_gene_stats(
    X: np.ndarray,
    cell_df: pd.DataFrame,
    genes: List[str],
    mask: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Return mean_expr[group][gene], pct_expr[group][gene], activity[group], n_cells[group].
    """
    sub = cell_df.loc[mask].copy()
    Xs = X[mask, :]
    groups = sort_labels(sub["cci_group_short"].astype(str).unique().tolist())

    mean_expr: Dict[str, Dict[str, float]] = {}
    pct_expr: Dict[str, Dict[str, float]] = {}
    activity: Dict[str, Dict[str, float]] = {}
    n_cells: Dict[str, int] = {}

    group_arr = sub["cci_group_short"].astype(str).values
    prop_act = sub["propagation_activity"].values.astype(float)
    resp = sub["cell_response_score"].values.astype(float)
    mod = sub["stage5_module_score"].values.astype(float)
    tgt = sub["target_expr"].values.astype(float)
    ptime = sub["propagation_time"].values.astype(float)

    for g in groups:
        m = group_arr == g
        n = int(m.sum())
        if n < MIN_CELLS_PER_GROUP:
            continue
        Xi = Xs[m, :]
        mean_vals = np.nanmean(Xi, axis=0)
        pct_vals = np.nanmean(Xi > 0, axis=0)
        mean_expr[g] = {gene: float(v) for gene, v in zip(genes, mean_vals)}
        pct_expr[g] = {gene: float(v) for gene, v in zip(genes, pct_vals)}
        activity[g] = {
            "propagation_activity_mean": float(np.nanmean(prop_act[m])),
            "cell_response_score_mean": float(np.nanmean(resp[m])),
            "stage5_module_score_mean": float(np.nanmean(mod[m])),
            "target_expr_mean": float(np.nanmean(tgt[m])),
            "propagation_time_mean": float(np.nanmean(ptime[m])),
        }
        n_cells[g] = n
    return mean_expr, pct_expr, activity, n_cells


def _get_nested(d: Dict[str, Dict[str, float]], a: str, b: str, default: float = 0.0) -> float:
    return float(d.get(a, {}).get(b, default))


def compute_stage_cci(
    p_stage: int,
    cell_df: pd.DataFrame,
    X: np.ndarray,
    lr_df: pd.DataFrame,
    lr_genes: List[str],
    gene_support_map: Dict[str, float],
) -> pd.DataFrame:
    """
    Frontier CCI for P stage:
    sender = stages < p_stage
    receiver = stage == p_stage
    """
    st = cell_df["display_propagation_stage"].values.astype(int)
    sender_mask = st < int(p_stage)
    receiver_mask = st == int(p_stage)

    if int(sender_mask.sum()) < MIN_CELLS_PER_GROUP or int(receiver_mask.sum()) < MIN_CELLS_PER_GROUP:
        log(f"[CCI][P{p_stage}][WARN] too few sender/receiver cells:", int(sender_mask.sum()), int(receiver_mask.sum()))
        return pd.DataFrame()

    s_mean, s_pct, s_act, s_n = compute_group_gene_stats(X, cell_df, lr_genes, sender_mask)
    r_mean, r_pct, r_act, r_n = compute_group_gene_stats(X, cell_df, lr_genes, receiver_mask)

    rows = []
    for _, lr in lr_df.iterrows():
        ligand = str(lr["ligand"])
        receptor = str(lr["receptor"])
        pathway = str(lr.get("pathway", ""))
        lr_pair = str(lr["lr_pair"])
        if ligand not in lr_genes or receptor not in lr_genes:
            continue

        gene_support = max(float(gene_support_map.get(ligand, 0.0)), float(gene_support_map.get(receptor, 0.0)))
        lr_response_support = 1.0 + LR_SUPPORT_STRENGTH * gene_support

        for sg in s_mean.keys():
            lig_mean = _get_nested(s_mean, sg, ligand)
            lig_pct = _get_nested(s_pct, sg, ligand)
            if lig_mean < MIN_EXPR_MEAN or lig_pct < MIN_EXPR_PCT:
                continue

            sender_activity = SENDER_ACTIVITY_FLOOR + (1.0 - SENDER_ACTIVITY_FLOOR) * float(s_act[sg]["propagation_activity_mean"])

            for rg in r_mean.keys():
                rec_mean = _get_nested(r_mean, rg, receptor)
                rec_pct = _get_nested(r_pct, rg, receptor)
                if rec_mean < MIN_EXPR_MEAN or rec_pct < MIN_EXPR_PCT:
                    continue

                receiver_activity = RECEIVER_ACTIVITY_FLOOR + (1.0 - RECEIVER_ACTIVITY_FLOOR) * float(r_act[rg]["propagation_activity_mean"])

                expr_score = float(lig_mean) * float(rec_mean)
                pct_score = math.sqrt(max(float(lig_pct), 0.0) * max(float(rec_pct), 0.0))
                cci_score = expr_score * pct_score * sender_activity * receiver_activity * lr_response_support

                rows.append({
                    "P_stage": f"P{p_stage}",
                    "P_stage_int": int(p_stage),
                    "sender_region": "P0" if p_stage == 1 else f"P0-P{p_stage - 1}",
                    "receiver_region": f"P{p_stage}",
                    "sender_group": sg,
                    "receiver_group": rg,
                    "ligand": ligand,
                    "receptor": receptor,
                    "lr_pair": lr_pair,
                    "pathway": pathway,
                    "sender_ligand_mean": float(lig_mean),
                    "sender_ligand_pct": float(lig_pct),
                    "receiver_receptor_mean": float(rec_mean),
                    "receiver_receptor_pct": float(rec_pct),
                    "sender_n_cells": int(s_n[sg]),
                    "receiver_n_cells": int(r_n[rg]),
                    "sender_propagation_activity": float(sender_activity),
                    "receiver_frontier_activity": float(receiver_activity),
                    "lr_response_support": float(lr_response_support),
                    "gene_support_max": float(gene_support),
                    "expr_score": float(expr_score),
                    "pct_score": float(pct_score),
                    "propagation_weighted_cci_score": float(cci_score),
                    "sender_cell_response_score_mean": float(s_act[sg]["cell_response_score_mean"]),
                    "receiver_cell_response_score_mean": float(r_act[rg]["cell_response_score_mean"]),
                    "sender_stage5_module_score_mean": float(s_act[sg]["stage5_module_score_mean"]),
                    "receiver_stage5_module_score_mean": float(r_act[rg]["stage5_module_score_mean"]),
                    "sender_target_expr_mean": float(s_act[sg]["target_expr_mean"]),
                    "receiver_target_expr_mean": float(r_act[rg]["target_expr_mean"]),
                })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        log(f"[CCI][P{p_stage}] no valid LR edges")
        return out

    out["score_rank_in_P"] = out["propagation_weighted_cci_score"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("propagation_weighted_cci_score", ascending=False).reset_index(drop=True)
    log(f"[CCI][P{p_stage}] edges = {len(out)}")
    log(out[["sender_group", "receiver_group", "lr_pair", "propagation_weighted_cci_score"]].head(10).to_string(index=False))
    return out


def compute_all_cci(cell_df: pd.DataFrame, lr_df: pd.DataFrame, genes_all: List[str], gene_support_df: pd.DataFrame) -> pd.DataFrame:
    cell_df, X, lr_genes = prepare_expression_for_lr(cell_df, lr_df, genes_all)
    support_map = dict(zip(gene_support_df["gene"].astype(str), gene_support_df["combined_gene_support"].astype(float)))

    all_edges = []
    for p in TARGET_P_STAGES:
        ed = compute_stage_cci(p, cell_df, X, lr_df, lr_genes, support_map)
        if len(ed) > 0:
            all_edges.append(ed)
    if not all_edges:
        raise RuntimeError("No CCI edges were computed for P1-P3.")
    out = pd.concat(all_edges, axis=0, ignore_index=True)

    # normalized scores within P and global
    out["score_norm_within_P"] = 0.0
    for p, sub_idx in out.groupby("P_stage_int").groups.items():
        vals = out.loc[sub_idx, "propagation_weighted_cci_score"].values.astype(float)
        out.loc[sub_idx, "score_norm_within_P"] = minmax_1d(vals, constant_value=0.0)
    out["score_norm_global"] = minmax_1d(out["propagation_weighted_cci_score"].values.astype(float), constant_value=0.0)
    return out


# ============================================================
# PLOTS
# ============================================================

def make_heatmap_cmap():
    return LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#F2B6A6", "#B2182B"], N=256)


def plot_lr_heatmap(edges: pd.DataFrame, out_base: Path):
    # Aggregate by LR pair x P stage
    agg = edges.groupby(["lr_pair", "P_stage"], as_index=False)["propagation_weighted_cci_score"].max()
    total = agg.groupby("lr_pair")["propagation_weighted_cci_score"].sum().sort_values(ascending=False)
    keep_lr = total.head(TOP_LR_FOR_HEATMAP).index.astype(str).tolist()
    mat = agg.pivot(index="lr_pair", columns="P_stage", values="propagation_weighted_cci_score").fillna(0.0)
    mat = mat.reindex(index=keep_lr, columns=[f"P{x}" for x in TARGET_P_STAGES]).fillna(0.0)
    mat_norm = mat.copy()
    for col in mat_norm.columns:
        mat_norm[col] = minmax_1d(mat_norm[col].values.astype(float), constant_value=0.0)

    mat.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_LR_by_P_heatmap_matrix.csv")
    mat_norm.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_LR_by_P_heatmap_matrix_norm_within_P.csv")

    fig_h = max(5.2, min(12.0, 0.26 * len(mat_norm) + 1.8))
    fig, ax = plt.subplots(figsize=(4.4, fig_h), dpi=DPI)
    im = ax.imshow(mat_norm.values, aspect="auto", interpolation="nearest", cmap=make_heatmap_cmap(), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(mat_norm.columns)))
    ax.set_xticklabels(mat_norm.columns)
    ax.set_yticks(np.arange(len(mat_norm.index)))
    ax.set_yticklabels(mat_norm.index, fontsize=6.5)
    ax.set_title(f"{TARGET_GENE} KO propagation-weighted LR activity")
    ax.set_xlabel("Propagation time point")
    ax.set_ylabel("LR pair")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Normalized CCI score")
    save_fig(fig, out_base)


def plot_group_pair_heatmap(edges: pd.DataFrame, out_base: Path):
    edges = edges.copy()
    edges["group_pair"] = edges["sender_group"].astype(str) + " → " + edges["receiver_group"].astype(str)
    agg = edges.groupby(["group_pair", "P_stage"], as_index=False)["propagation_weighted_cci_score"].sum()
    total = agg.groupby("group_pair")["propagation_weighted_cci_score"].sum().sort_values(ascending=False)
    keep = total.head(TOP_GROUP_PAIRS_FOR_HEATMAP).index.astype(str).tolist()
    mat = agg.pivot(index="group_pair", columns="P_stage", values="propagation_weighted_cci_score").fillna(0.0)
    mat = mat.reindex(index=keep, columns=[f"P{x}" for x in TARGET_P_STAGES]).fillna(0.0)
    mat_norm = mat.copy()
    for col in mat_norm.columns:
        mat_norm[col] = minmax_1d(mat_norm[col].values.astype(float), constant_value=0.0)

    mat.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_group_pair_by_P_heatmap_matrix.csv")
    mat_norm.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_group_pair_by_P_heatmap_matrix_norm_within_P.csv")

    fig_h = max(5.2, min(12.0, 0.26 * len(mat_norm) + 1.8))
    fig, ax = plt.subplots(figsize=(5.3, fig_h), dpi=DPI)
    im = ax.imshow(mat_norm.values, aspect="auto", interpolation="nearest", cmap=make_heatmap_cmap(), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(mat_norm.columns)))
    ax.set_xticklabels(mat_norm.columns)
    ax.set_yticks(np.arange(len(mat_norm.index)))
    ax.set_yticklabels(mat_norm.index, fontsize=6.5)
    ax.set_title(f"{TARGET_GENE} KO propagation-weighted group-pair CCI")
    ax.set_xlabel("Propagation time point")
    ax.set_ylabel("Sender → receiver")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Normalized CCI score")
    save_fig(fig, out_base)


def circular_layout(nodes: List[str]) -> Dict[str, Tuple[float, float]]:
    n = len(nodes)
    if n == 0:
        return {}
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return {node: (float(np.cos(a)), float(np.sin(a))) for node, a in zip(nodes, angles)}


def build_stable_node_layout(edges: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    nodes = sorted(set(edges["sender_group"].astype(str)) | set(edges["receiver_group"].astype(str)))
    # Try networkx spring layout for stable, prettier network.
    try:
        import networkx as nx
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        agg = edges.groupby(["sender_group", "receiver_group"], as_index=False)["propagation_weighted_cci_score"].sum()
        for _, r in agg.iterrows():
            G.add_edge(str(r["sender_group"]), str(r["receiver_group"]), weight=float(r["propagation_weighted_cci_score"]))
        pos = nx.spring_layout(G, seed=SEED, weight="weight", k=0.9, iterations=200)
        return {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    except Exception:
        return circular_layout(nodes)


def draw_arrow(ax, x1, y1, x2, y2, color, lw, alpha=0.82, rad=0.10):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=7.0 + 1.2 * lw,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=12,
        shrinkB=12,
        zorder=2,
    )
    ax.add_patch(arr)


def canonical_celltype_label(x: str) -> str:
    """Make legend labels readable while preserving known cell-type names."""
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return "Unknown"
    return shorten_label(s, 28)


def build_node_color_map(nodes: List[str]) -> Dict[str, str]:
    """Assign stable solid colors to cell-type nodes."""
    out = {}
    used = 0
    for n in sort_labels(nodes):
        if n in CELLTYPE_NODE_COLORS:
            out[n] = CELLTYPE_NODE_COLORS[n]
        else:
            # also try the dot/space-normalized version
            n_space = str(n).replace(".", " ")
            n_dot = str(n).replace(" ", ".")
            if n_space in CELLTYPE_NODE_COLORS:
                out[n] = CELLTYPE_NODE_COLORS[n_space]
            elif n_dot in CELLTYPE_NODE_COLORS:
                out[n] = CELLTYPE_NODE_COLORS[n_dot]
            else:
                out[n] = FALLBACK_NODE_COLORS[used % len(FALLBACK_NODE_COLORS)]
                used += 1
    return out


def plot_networks(edges: pd.DataFrame, out_base: Path):
    # use only top edges per P
    top_list = []
    for p in TARGET_P_STAGES:
        sub = (
            edges[edges["P_stage_int"] == p]
            .sort_values("propagation_weighted_cci_score", ascending=False)
            .head(TOP_EDGES_PER_P)
            .copy()
        )
        top_list.append(sub)
    top_edges = pd.concat(top_list, axis=0, ignore_index=True)
    top_edges.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_top_edges.csv", index=False)

    pos = build_stable_node_layout(top_edges)
    all_nodes = sort_labels(sorted(set(top_edges["sender_group"].astype(str)) | set(top_edges["receiver_group"].astype(str))))
    node_color_map = build_node_color_map(all_nodes)

    node_color_df = pd.DataFrame({
        "cell_group": all_nodes,
        "display_label": [canonical_celltype_label(x) for x in all_nodes],
        "node_color": [node_color_map[x] for x in all_nodes],
    })
    node_color_df.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_network_node_color_legend.csv", index=False)

    all_scores = top_edges["propagation_weighted_cci_score"].values.astype(float)
    score01 = minmax_1d(all_scores, constant_value=0.5)
    score_map = dict(zip(top_edges.index.tolist(), score01))

    # Wider canvas because legend is placed outside the right side.
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), dpi=DPI)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, p in zip(axes, TARGET_P_STAGES):
        sub = top_edges[top_edges["P_stage_int"] == p].copy()
        nodes = sort_labels(sorted(set(sub["sender_group"].astype(str)) | set(sub["receiver_group"].astype(str))))
        edge_color = P_STAGE_COLORS.get(p, DARK_GREY)

        # draw edges first
        for idx, r in sub.iterrows():
            s = str(r["sender_group"])
            t = str(r["receiver_group"])
            if s not in pos or t not in pos:
                continue
            x1, y1 = pos[s]
            x2, y2 = pos[t]
            v = float(score_map.get(idx, 0.5))
            lw = 0.35 + 3.8 * v
            rad = 0.08 if s <= t else -0.08
            draw_arrow(ax, x1, y1, x2, y2, color=edge_color, lw=lw, alpha=0.68, rad=rad)

        # draw solid colored nodes; no node text labels on the plot
        for n in nodes:
            if n not in pos:
                continue
            x, y = pos[n]
            ax.scatter(
                [x], [y],
                s=NETWORK_NODE_SIZE,
                c=node_color_map.get(n, DARK_GREY),
                edgecolors=NETWORK_NODE_EDGE_COLOR,
                linewidths=NETWORK_NODE_EDGE_WIDTH,
                zorder=4,
            )
            if NETWORK_SHOW_NODE_TEXT:
                ax.text(
                    x, y, shorten_label(n, 13),
                    ha="center", va="center", fontsize=6.2,
                    color="#111111", zorder=5,
                )

        # annotate top LR names on right side of each panel
        ann = sub.sort_values("propagation_weighted_cci_score", ascending=False).head(8)
        txt = "\n".join([f"{i+1}. {r.lr_pair}" for i, r in enumerate(ann.itertuples())])
        ax.text(1.04, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#333333")

        sender_region = "P0" if p == 1 else f"P0-P{p-1}"
        ax.set_title(f"P{p}: {sender_region} → P{p}", color=edge_color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # One shared legend: colored node point -> cell group name.
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=NETWORK_LEGEND_MARKER_SIZE,
            markerfacecolor=node_color_map[n],
            markeredgecolor=NETWORK_NODE_EDGE_COLOR,
            markeredgewidth=0.4,
            label=canonical_celltype_label(n),
        )
        for n in all_nodes
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.94),
        frameon=False,
        fontsize=7.2,
        title="Cell group",
        title_fontsize=7.6,
        labelspacing=0.35,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.suptitle(f"{TARGET_GENE} KO propagation-weighted frontier CCI", y=1.03, fontsize=12)
    # Leave space at right for node legend.
    fig.subplots_adjust(left=0.03, right=0.82, wspace=0.30, top=0.88, bottom=0.06)
    save_fig(fig, out_base)


def plot_stage_score_bars(edges: pd.DataFrame, out_base: Path):
    stage_sum = edges.groupby("P_stage", as_index=False).agg(
        total_score=("propagation_weighted_cci_score", "sum"),
        n_edges=("propagation_weighted_cci_score", "size"),
        mean_score=("propagation_weighted_cci_score", "mean"),
    )
    stage_sum["P_stage_int"] = stage_sum["P_stage"].str.replace("P", "", regex=False).astype(int)
    stage_sum = stage_sum.sort_values("P_stage_int")
    stage_sum.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_total_CCI_by_P.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=DPI)
    colors = [P_STAGE_COLORS.get(int(p), DARK_GREY) for p in stage_sum["P_stage_int"]]
    ax.bar(stage_sum["P_stage"], stage_sum["total_score"], color=colors, alpha=0.88)
    ax.set_ylabel("Total propagation-weighted CCI score")
    ax.set_xlabel("Propagation time point")
    ax.set_title(f"{TARGET_GENE} KO total frontier CCI")
    for sp in ax.spines.values():
        sp.set_visible(False)
    save_fig(fig, out_base)


# ============================================================
# QC
# ============================================================

def write_qc(cell_df: pd.DataFrame, lr_df: pd.DataFrame, edges: pd.DataFrame, prop_dir: Path, gene_support_df: pd.DataFrame):
    rows = []
    rows.append({"metric": "propagation_dir", "value": str(prop_dir)})
    rows.append({"metric": "n_cells_used_total", "value": int(len(cell_df))})
    rows.append({"metric": "n_lr_pairs_usable", "value": int(len(lr_df))})
    rows.append({"metric": "n_cci_edges_total", "value": int(len(edges))})
    rows.append({"metric": "n_genes_with_stage5_support", "value": int((gene_support_df["stage5_direct_support"] > 0).sum())})
    rows.append({"metric": "n_genes_with_malignant_topgene_support", "value": int((gene_support_df["malignant_topgene_support"] > 0).sum())})
    for p in sorted(cell_df["display_propagation_stage"].unique().tolist()):
        rows.append({"metric": f"n_cells_P{int(p)}", "value": int((cell_df["display_propagation_stage"] == int(p)).sum())})
    for p in TARGET_P_STAGES:
        sub = edges[edges["P_stage_int"] == p]
        rows.append({"metric": f"n_cci_edges_P{p}", "value": int(len(sub))})
        rows.append({"metric": f"total_cci_score_P{p}", "value": float(sub["propagation_weighted_cci_score"].sum()) if len(sub) else 0.0})
    qc = pd.DataFrame(rows)
    qc.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_CCI_qc_summary.csv", index=False)
    log("[SAVE]", OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_CCI_qc_summary.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 100)
    log("[START] SOX2 KO P1-P3 propagation-weighted CCI")
    log("[TARGET_GENE]", TARGET_GENE)
    log("[MODE]", MODE)
    log("[OUT_DIR]", OUT_DIR)
    log("=" * 100)

    for p in [LOG1P_NPY, META_NPZ, GENES_TSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    genes_all = read_gene_names()
    cell_df, prop_df, prop_dir = load_propagation_tables()

    # keep only P0-P3 cells; P1-P3 use P0...P(t-1) -> Pt
    cell_df = cell_df[cell_df["display_propagation_stage"].isin([0, 1, 2, 3])].copy()
    if len(cell_df) == 0:
        raise RuntimeError("No P0-P3 cells after filtering display_propagation_stage.")

    lr_df = load_lr_database(genes_all)
    gene_support_df = build_gene_support(genes_all)
    gene_support_df.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_gene_response_support_used_for_LR.csv", index=False)
    lr_df.to_csv(OUT_DIR / f"{TARGET_GENE}_{MODE}_LR_pairs_used.csv", index=False)
    log("[SAVE]", OUT_DIR / f"{TARGET_GENE}_{MODE}_gene_response_support_used_for_LR.csv")
    log("[SAVE]", OUT_DIR / f"{TARGET_GENE}_{MODE}_LR_pairs_used.csv")

    edges = compute_all_cci(cell_df, lr_df, genes_all, gene_support_df)
    edges_csv = OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_propagation_weighted_CCI_all_edges.csv"
    edges.to_csv(edges_csv, index=False)
    log("[SAVE]", edges_csv)

    # plots
    plot_networks(edges, OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_network")
    plot_lr_heatmap(edges, OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_LR_heatmap")
    plot_group_pair_heatmap(edges, OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_group_pair_heatmap")
    plot_stage_score_bars(edges, OUT_DIR / f"{TARGET_GENE}_{MODE}_P1_P3_total_CCI_bar")

    write_qc(cell_df, lr_df, edges, prop_dir, gene_support_df)

    log("=" * 100)
    log("[DONE] outputs saved to:", OUT_DIR)
    log("=" * 100)


if __name__ == "__main__":
    main()
