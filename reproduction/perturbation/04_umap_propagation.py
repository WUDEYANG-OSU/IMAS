# -*- coding: utf-8 -*-
"""
hnscc_caall_sox2KO_umap_propagation_direction_cluster_gene_response_heatmap.py

目标
====
参照两个已有思路：
1) 单基因 KO 在 UMAP 上的趋势/传播方向可视化：用 target expression / KO response / UMAP 邻域
   自动推断扰动传播方向，而不是手动指定起点。
2) 敲除后各 cluster top1000 response gene heatmap：用 Stage5 perturbation impact + cluster expression
   specificity / owner bonus / row-wise z-score，让不同 cluster 的响应基因块明显不同。

本脚本针对 HNSCC CA_all / SOX2 KO：
- 自动识别 SOX2-high source cluster：不直接指定 malignant cells；只用 SOX2 表达、SOX2-positive fraction、
  local response sensitivity 等数据证据计算 seed score。因为 SOX2 在 malignant cells 中高表达，正常情况下
  source 会自动落在 malignant-like cluster。
- 计算 UMAP 上的扰动传播方向：cluster graph shortest path + cell-neighborhood vector field。
- 生成 gene x cluster heatmap：
    纵轴 = 全基因（按扰动 owner cluster 和传播顺序排序，不显示 gene label）
    横轴 = 扰动行进到的位置（cluster，按 propagation order 排列）
    颜色 = 该 gene 在该 cluster 的 KO response degree，row-wise z-score 后显示。
- 另输出 top1000-per-cluster block heatmap，便于看到每个 cluster 的响应基因明显不同。

主要输出
========
OUT_DIR 下：
- SOX2_cluster_propagation_order.csv
- SOX2_cell_level_propagation_response.csv
- SOX2_umap_ko_propagation_vector_field.png/svg
- SOX2_umap_cluster_propagation_order.png/svg
- SOX2_all_genes_cluster_response_heatmap_matrix.csv
- SOX2_all_genes_cluster_response_heatmap.png/svg
- SOX2_cluster_top1000_response_genes_by_propagation.csv
- SOX2_top1000_block_response_heatmap.png/svg
- SOX2_algorithm_qc_summary.csv

运行前确认
==========
- CA_all log1p matrix / genes.tsv / rna_meta.npz 已存在；
- celltypeN-guided h5ad 已存在；
- Stage5 SOX2 baseline / KO node state 已存在。
"""

from __future__ import annotations

import math
import heapq
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as e:
    raise ImportError("This script needs anndata. Please install/import anndata in your environment.") from e

try:
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    raise ImportError("This script needs scikit-learn. Please install/import sklearn in your environment.") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
TARGET_GENE = "SOX2"
MODE = "ko"  # ko / oe
IMPACT_MODE = "auto"  # auto / down / up / abs

BASE = get_project_root()

# ---------- CA_all root: compatible with Worldword / WorldWord ----------
EXPLOIT_ROOT_CANDIDATES = [
    BASE / "LLM-Worldword" / "Exploit-HNSCC",
    BASE / "LLM-WorldWord" / "Exploit-HNSCC",
]
EXPLOIT_ROOT = None
for _root in EXPLOIT_ROOT_CANDIDATES:
    if (_root / "processed_CA" / "RNA_CA_all").exists():
        EXPLOIT_ROOT = _root
        break
if EXPLOIT_ROOT is None:
    # Keep a readable error. Do not silently use the wrong path.
    raise FileNotFoundError(
        "Cannot find Exploit-HNSCC processed_CA/RNA_CA_all under:\n"
        + "\n".join(str(x) for x in EXPLOIT_ROOT_CANDIDATES)
    )

RNA_DIR = EXPLOIT_ROOT / "processed_CA" / "RNA_CA_all"
ASSET_DIR = EXPLOIT_ROOT / "assets_CA"
LOG1P_NPY = RNA_DIR / "CA_all.X_rna_log1p_norm_dense.npy"
META_NPZ = RNA_DIR / "CA_all.rna_meta.npz"
GENES_TSV = ASSET_DIR / "genes.tsv"

RUN_STAGEB_DIR = BASE / "runs" / "frontier_transfer_rna_only_stageB_clusterCorrection_HNSCC_CAall_only"
RUN_STAGEB_DATASET_DIR = RUN_STAGEB_DIR / "HNSCC-CA_all"
RECLUSTER_DIR = RUN_STAGEB_DATASET_DIR / "single_pool_CA_all_single_recluster_for_cci"
ADATA_UMAP = RECLUSTER_DIR / "CA_all_single_matchedcells_scanpy.h5ad"

CURRENT_STAGE5_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "CA_all_single_cci_newLeiden"
    / "communication_pseudotime_clusterGNN_LTC"
    / "five_stage_network_viz_out"
)
STAGE5_DIR_CANDIDATES = [
    CURRENT_STAGE5_DIR,
    (get_external_root("SIG_LLM", "external/Sig-LLM-Frag") / "BreastCancer/five_stage_network_viz_out_hnscc_caall/1"),
    (get_external_root("SIG_LLM", "external/Sig-LLM-Frag") / "BreastCancer/five_stage_network_viz_out"),
]

# Output
OUT_DIR_NAME = f"{TARGET_GENE}_{MODE}_umap_propagation_direction_cluster_gene_response_heatmap"

# ---------- source inference ----------
# 不手动指定 malignant 起点。source_score 完全由 SOX2 expression / SOX2-positive fraction / cluster specificity 推断。
SOURCE_POS_Q = 0.80
SOURCE_MIN_RELATIVE_TO_MAX = 0.70
SOURCE_MIN_N_CLUSTERS = 1
SOURCE_MAX_N_CLUSTERS = 3
SOURCE_TARGET_MEAN_POWER = 1.00
SOURCE_TARGET_PCT_POWER = 1.20
SOURCE_TARGET_SPEC_POWER = 1.00

# ---------- cluster graph / propagation ----------
CLUSTER_GRAPH_K = 4
CLUSTER_GRAPH_DISTANCE_POWER = 1.0
CLUSTER_SENSITIVITY_EDGE_BONUS = 0.75
CELL_VECTOR_KNN = 22
CELL_VECTOR_LATER_TIME_EPS = 1e-5
CELL_VECTOR_RESPONSE_POWER = 1.0
GRID_NX = 55
GRID_NY = 42
GRID_MIN_CELLS = 6
ARROW_SCALE = 1.0
ARROW_WIDTH = 0.0017
ARROW_ALPHA = 0.88
BG_ALPHA = 0.82

# ---------- display compression for cluster propagation UMAP ----------
# 原始传播顺序可能是 P0-P8；这里仅用于 UMAP 显示层压缩成 P0-P3。
# 不改变 prop_df["propagation_rank"]，因此不会影响后续 heatmap / gene selection。
#
# 重要：最早的 source / 启动位点通常细胞数很少，直接作为显示 P0 会不好看。
# 所以这里先丢掉最前面的原始传播 rank，再把剩余区域压缩成显示 P0-P3。
# 默认 DISPLAY_SKIP_INITIAL_RANKS=1，即原始 P0/source 区域不参与四张累计 UMAP。
# 如果还觉得 P0 太少，可以改成 2；如果想恢复不丢起点，改成 0。
DISPLAY_PROPAGATION_N_STAGES = 4
DISPLAY_SKIP_INITIAL_RANKS = 1
DISPLAY_SKIP_INITIAL_FRACTION = 0.0  # 例如 0.10 表示至少丢掉最前 10% 原始 rank；默认不用比例。
DISPLAY_STAGE_PREFIX = "P"
DISPLAY_STAGE_POINT_SIZE = 3.4
DISPLAY_STAGE_ALPHA = 0.88
DISPLAY_CURRENT_STAGE_EDGE_ALPHA = 0.35

# ---------- response heatmap gene universe ----------
USE_ALL_CA_GENES_FOR_HEATMAP = True
RESTRICT_TO_STAGE5_NODES_ONLY = False
NON_STAGE5_IMPACT_FLOOR = 0.16
STAGE5_DIRECT_IMPACT_BOOST = 1.45
GENE_CHUNK = 512

# ---------- response scoring ----------
TOP_N_GENES_PER_CLUSTER = 1000
TOP_N_PREVIEW_PER_CLUSTER = 200
ALPHA_IMPACT = 1.0
BETA_CLUSTER_SPECIFICITY = 2.35
GAMMA_OWNER_BONUS = 1.75
GAMMA_ROWZ_BONUS = 1.10
MEAN_EXPR_BONUS_WEIGHT = 0.25
PROPAGATION_REACH_WEIGHT = 0.35
MIN_PCT_EXPR = 0.06
MIN_SPEC_Z = 0.0
UNIQUE_GENE_ROWS = True
ALLOW_DUPLICATE_IF_NOT_ENOUGH = True
RAW_SCORE_SCALE_FOR_ROWZ = 80.0
HEATMAP_Z_CLIP = 2.3

# ---------- grouping ----------
GROUP_BY_ORIGINAL_CELLTYPE = True
CELLTYPE_COL_CANDIDATES = [
    "cell.type", "cell_type", "CellType", "original_cell_type",
    "integrated_label", "annotation", "cell_annotation",
]
ANALYSIS_CELLTYPE_COL = "analysis_celltype"
ORIGINAL_LEIDEN_COL_BACKUP = "leiden_cluster_backup"
MIN_CELLS_PER_CELLTYPE_GROUP = 1
PREFERRED_CELLTYPE_ORDER = [
    "B.cells", "B cells", "B_Plasma.cells", "B/Plasma cells",
    "Dendritic.cells", "Dendritic cells", "Endothelial.cells", "Endothelial cells",
    "Fibroblasts", "Macrophages", "Malignant.cells", "Malignant cells",
    "Mast.cells", "Mast cells", "Myocytes", "T.cells", "T cells",
]
MAX_CELLTYPE_LABEL_LEN = 25

# ---------- plotting ----------
DPI = 320
FIG_DPI_VECTOR = 260
POINT_SIZE = 2.2
NATURE_BLUE = "#3B6FB6"
NATURE_WHITE = "#F7F7F7"
NATURE_RED = "#B2182B"
SOFT_ORANGE = "#E6953B"
SOFT_PURPLE = "#6A51A3"

SEED = 42
np.random.seed(SEED)

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["svg.fonttype"] = "none"

# ============================================================
# HELPERS
# ============================================================

def log(*args):
    print(*args, flush=True)


def safe_name(x: str) -> str:
    x = str(x)
    for bad in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
        x = x.replace(bad, "_")
    return x


def save_fig(fig: plt.Figure, out_base: Path, dpi: int = DPI, tight: bool = True):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches=bbox, facecolor="white")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches=bbox, facecolor="white")
    plt.close(fig)
    log("[SAVE]", out_base.with_suffix(".png"))
    log("[SAVE]", out_base.with_suffix(".svg"))


def minmax_1d(x: np.ndarray, constant_value: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if finite.sum() == 0:
        return out
    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])
    if hi - lo < 1e-12:
        out[finite] = float(constant_value)
        return out
    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if finite.sum() == 0:
        return out
    mu = np.nanmean(x[finite])
    sd = np.nanstd(x[finite])
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    out[finite] = (x[finite] - mu) / sd
    return out


def zscore_by_gene(cluster_by_gene: np.ndarray) -> np.ndarray:
    mu = np.nanmean(cluster_by_gene, axis=0, keepdims=True)
    sd = np.nanstd(cluster_by_gene, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((cluster_by_gene - mu) / sd).astype(np.float32)


def row_zscore(gene_by_cluster: np.ndarray) -> np.ndarray:
    mu = np.nanmean(gene_by_cluster, axis=1, keepdims=True)
    sd = np.nanstd(gene_by_cluster, axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((gene_by_cluster - mu) / sd).astype(np.float32)


def safe_pow(x: np.ndarray, p: float) -> np.ndarray:
    return np.power(np.clip(np.asarray(x, dtype=np.float64), 0.0, None), float(p))


def norm_barcode(x: str) -> str:
    x = str(x)
    if "::" in x:
        return x.split("::", 1)[-1]
    if ":" in x:
        return x.split(":", 1)[-1]
    return x


def sort_cluster_labels(xs: List[str]) -> List[str]:
    def _key(v):
        s = str(v)
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted([str(x) for x in xs], key=_key)


def shorten_label(x: str, max_len: int = MAX_CELLTYPE_LABEL_LEN) -> str:
    x = str(x).strip().replace("_", " ").replace(".", " ")
    x = " ".join(x.split())
    if x == "" or x.lower() in {"nan", "none"}:
        x = "Unknown"
    if len(x) > max_len:
        return x[: max_len - 1] + "…"
    return x


def is_target_gene(g: str) -> bool:
    return str(g).upper() == TARGET_GENE.upper()


def make_diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "NatureBlueWhiteRed", [NATURE_BLUE, NATURE_WHITE, NATURE_RED], N=256
    )


def make_white_orange_cmap():
    return LinearSegmentedColormap.from_list("white_orange", ["#FFFFFF", "#F6C99B", "#D95F02"], N=256)

# ============================================================
# PATH RESOLUTION / DATA LOADING
# ============================================================

def resolve_stage5_dir() -> Path:
    baseline_name = f"{TARGET_GENE}_stage5_baseline_node_state.csv"
    pert_name = f"{TARGET_GENE}_stage5_{MODE}_node_state.csv"
    for d in STAGE5_DIR_CANDIDATES:
        if (d / baseline_name).exists() and (d / pert_name).exists():
            return d
    msg = [f"Cannot find Stage5 files for {TARGET_GENE}/{MODE}. Tried:"]
    for d in STAGE5_DIR_CANDIDATES:
        msg.append(f" - {d}")
        if d.exists():
            cands = sorted(d.glob("*_stage5_baseline_node_state.csv"))[:40]
            if cands:
                msg.append("   available: " + ", ".join(p.name.replace("_stage5_baseline_node_state.csv", "") for p in cands))
    raise FileNotFoundError("\n".join(msg))


def read_gene_names() -> List[str]:
    if GENES_TSV.exists():
        df = pd.read_csv(GENES_TSV, sep="\t")
        for c in ["gene", "gene_name", "symbol"]:
            if c in df.columns:
                return df[c].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    z = np.load(META_NPZ, allow_pickle=True)
    if "gene_names" in z.files:
        return np.asarray(z["gene_names"]).astype(str).tolist()
    raise FileNotFoundError(f"Cannot find gene names from {GENES_TSV} or {META_NPZ}")


def detect_cluster_col(adata_umap: ad.AnnData) -> str:
    for c in ["new_cluster", "leiden", "leiden_new", "original_cluster"]:
        if c in adata_umap.obs.columns:
            return c
    raise KeyError(f"Cannot find cluster column in h5ad obs. Available: {list(adata_umap.obs.columns)}")


def detect_celltype_col(adata_umap: ad.AnnData) -> Optional[str]:
    for c in CELLTYPE_COL_CANDIDATES:
        if c in adata_umap.obs.columns:
            return c
    return None


def choose_celltype_order(vals: pd.Series) -> List[str]:
    vals = vals.astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    counts = vals.value_counts(dropna=False)
    existing = counts.index.astype(str).tolist()
    ordered: List[str] = []
    existing_by_norm = {shorten_label(x).lower(): x for x in existing}
    for p in PREFERRED_CELLTYPE_ORDER:
        key = shorten_label(p).lower()
        if key in existing_by_norm and existing_by_norm[key] not in ordered:
            ordered.append(existing_by_norm[key])
    rest = [x for x in existing if x not in ordered]
    rest = sorted(rest, key=lambda z: (-int(counts.get(z, 0)), shorten_label(z)))
    ordered.extend(rest)
    ordered = [x for x in ordered if int(counts.get(x, 0)) >= int(MIN_CELLS_PER_CELLTYPE_GROUP)]
    return ordered


def apply_original_celltype_as_cluster(adata_umap: ad.AnnData) -> Tuple[ad.AnnData, pd.DataFrame]:
    celltype_col = detect_celltype_col(adata_umap)
    if celltype_col is None:
        raise KeyError(
            "GROUP_BY_ORIGINAL_CELLTYPE=True, but no cell.type-like column found. "
            f"Tried {CELLTYPE_COL_CANDIDATES}. Available obs columns: {list(adata_umap.obs.columns)}"
        )
    adata_umap = adata_umap.copy()
    if "new_cluster" in adata_umap.obs.columns:
        adata_umap.obs[ORIGINAL_LEIDEN_COL_BACKUP] = adata_umap.obs["new_cluster"].astype(str)
    ct_raw = adata_umap.obs[celltype_col].astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    ordered_ct = choose_celltype_order(ct_raw)
    ct2cluster = {ct: str(i) for i, ct in enumerate(ordered_ct)}
    keep = ct_raw.isin(ct2cluster.keys()).values
    if int(keep.sum()) == 0:
        raise RuntimeError("No cells left after original cell.type grouping.")
    if int(keep.sum()) < adata_umap.n_obs:
        log(f"[CELLTYPE] filtered by MIN_CELLS_PER_CELLTYPE_GROUP: {adata_umap.n_obs} -> {int(keep.sum())}")
        adata_umap = adata_umap[keep].copy()
        ct_raw = adata_umap.obs[celltype_col].astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    adata_umap.obs[ANALYSIS_CELLTYPE_COL] = ct_raw.values
    adata_umap.obs["new_cluster"] = ct_raw.map(ct2cluster).astype(str).values
    rows = []
    for ct in ordered_ct:
        cl = ct2cluster[ct]
        n_cells = int((ct_raw.values == ct).sum())
        rows.append({
            "cluster_label": str(cl),
            "cluster_display": f"C{cl}",
            "celltype_col_used": celltype_col,
            "major_celltype": str(ct),
            "major_celltype_short": shorten_label(ct),
            "n_cells": n_cells,
            "xtick_label": f"C{cl}: {shorten_label(ct)}",
        })
    label_df = pd.DataFrame(rows)
    log("[CELLTYPE] using original cell.type as analysis cluster; col =", celltype_col)
    log(label_df[["cluster_display", "major_celltype", "n_cells"]].to_string(index=False))
    return adata_umap, label_df


def infer_cluster_celltype_labels(adata_umap: ad.AnnData, clusters: List[str]) -> pd.DataFrame:
    celltype_col = detect_celltype_col(adata_umap)
    cl_arr = adata_umap.obs["new_cluster"].astype(str)
    rows = []
    for cl in clusters:
        m = cl_arr == str(cl)
        n_cells = int(m.sum())
        if celltype_col is None:
            major = "Unknown"
        else:
            vals = adata_umap.obs.loc[m, celltype_col].astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
            major = str(vals.value_counts().index[0]) if len(vals) else "Unknown"
        rows.append({
            "cluster_label": str(cl),
            "cluster_display": f"C{cl}",
            "celltype_col_used": celltype_col or "",
            "major_celltype": major,
            "major_celltype_short": shorten_label(major),
            "n_cells": n_cells,
            "xtick_label": f"C{cl}: {shorten_label(major)}",
        })
    return pd.DataFrame(rows)


def get_cell_row_indices(adata_umap: ad.AnnData, meta_npz_path: Path) -> np.ndarray:
    z = np.load(meta_npz_path, allow_pickle=True)
    assert "cell_ids" in z.files, "cell_ids not found in CA_all.rna_meta.npz"
    cell_ids_all = z["cell_ids"].astype(str)
    exact_map = {cid: i for i, cid in enumerate(cell_ids_all)}
    obs_ids = adata_umap.obs_names.astype(str).tolist()
    if all(cid in exact_map for cid in obs_ids):
        return np.array([exact_map[cid] for cid in obs_ids], dtype=int)
    if "cell_id" in adata_umap.obs.columns:
        obs_cell_ids = adata_umap.obs["cell_id"].astype(str).tolist()
        if all(cid in exact_map for cid in obs_cell_ids):
            return np.array([exact_map[cid] for cid in obs_cell_ids], dtype=int)
    norm_all = [norm_barcode(x) for x in cell_ids_all]
    vc = pd.Series(norm_all).value_counts()
    nondup = set(vc[vc == 1].index.astype(str).tolist())
    norm_map = {norm_barcode(cid): i for i, cid in enumerate(cell_ids_all) if norm_barcode(cid) in nondup}
    for col in ["barcode_counts", "barcode", "barcode_meta"]:
        if col in adata_umap.obs.columns:
            keys = adata_umap.obs[col].astype(str).map(norm_barcode).tolist()
            if all(k in norm_map for k in keys):
                log("[META] matched rows by obs column:", col)
                return np.array([norm_map[k] for k in keys], dtype=int)
    missing = [cid for cid in obs_ids if cid not in exact_map][:8]
    raise ValueError("UMAP cells cannot be matched to CA_all.rna_meta.npz. Example missing: " + str(missing))


def load_umap_and_clusters() -> Tuple[ad.AnnData, np.ndarray, List[str], pd.DataFrame]:
    assert ADATA_UMAP.exists(), f"Missing h5ad: {ADATA_UMAP}"
    adata_umap = ad.read_h5ad(str(ADATA_UMAP))
    assert "X_umap" in adata_umap.obsm, "X_umap not found in h5ad."
    cluster_col = detect_cluster_col(adata_umap)
    adata_umap.obs["new_cluster"] = adata_umap.obs[cluster_col].astype(str)
    log("[UMAP] h5ad =", ADATA_UMAP)
    log("[UMAP] n_obs before grouping =", adata_umap.n_obs)
    log("[UMAP] original cluster_col =", cluster_col)
    if GROUP_BY_ORIGINAL_CELLTYPE:
        adata_umap, label_df = apply_original_celltype_as_cluster(adata_umap)
    else:
        clusters_tmp = sort_cluster_labels(adata_umap.obs["new_cluster"].astype(str).unique().tolist())
        label_df = infer_cluster_celltype_labels(adata_umap, clusters_tmp)
    row_idx = get_cell_row_indices(adata_umap, META_NPZ)
    genes = read_gene_names()
    log("[UMAP] n_obs used =", adata_umap.n_obs)
    log("[UMAP] analysis n_cluster =", adata_umap.obs["new_cluster"].astype(str).nunique())
    return adata_umap, row_idx, genes, label_df


def load_stage5_impact(stage5_dir: Path) -> pd.DataFrame:
    baseline_csv = stage5_dir / f"{TARGET_GENE}_stage5_baseline_node_state.csv"
    pert_csv = stage5_dir / f"{TARGET_GENE}_stage5_{MODE}_node_state.csv"
    b = pd.read_csv(baseline_csv)
    k = pd.read_csv(pert_csv)
    need = {"node", "state_final"}
    if not need.issubset(b.columns) or not need.issubset(k.columns):
        raise ValueError("Stage5 node state CSVs must contain columns: node, state_final")
    df = (
        b[["node", "state_final"]].rename(columns={"state_final": "state_baseline"})
        .merge(k[["node", "state_final"]].rename(columns={"state_final": "state_perturbed"}), on="node", how="inner")
    )
    df["node"] = df["node"].astype(str)
    before = len(df)
    df = df[~df["node"].map(is_target_gene)].copy()
    log(f"[STAGE5] exclude target gene {TARGET_GENE}: {before} -> {len(df)}")
    df["state_baseline"] = pd.to_numeric(df["state_baseline"], errors="coerce")
    df["state_perturbed"] = pd.to_numeric(df["state_perturbed"], errors="coerce")
    df = df.dropna(subset=["state_baseline", "state_perturbed"]).copy()
    df["delta_perturbed_minus_base"] = df["state_perturbed"] - df["state_baseline"]
    imode = IMPACT_MODE.lower().strip()
    if imode == "auto":
        imode = "down" if MODE.lower() == "ko" else "up"
    if imode == "down":
        df["perturb_impact"] = np.maximum(0.0, df["state_baseline"] - df["state_perturbed"])
    elif imode == "up":
        df["perturb_impact"] = np.maximum(0.0, df["state_perturbed"] - df["state_baseline"])
    elif imode == "abs":
        df["perturb_impact"] = np.abs(df["delta_perturbed_minus_base"])
    else:
        raise ValueError(f"Unsupported IMPACT_MODE: {IMPACT_MODE}")
    df = df.sort_values("perturb_impact", ascending=False).reset_index(drop=True)
    log("[STAGE5] impact mode used =", imode)
    log("[STAGE5] top impact nodes:")
    log(df[["node", "state_baseline", "state_perturbed", "perturb_impact"]].head(20).to_string(index=False))
    return df

# ============================================================
# EXPRESSION FEATURES
# ============================================================

def compute_cluster_expression_features_chunked(
    log1p_npy: Path,
    row_idx: np.ndarray,
    genes: List[str],
    adata_umap: ad.AnnData,
    gene_keep_mask: np.ndarray,
    gene_chunk: int = GENE_CHUNK,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    genes_arr = np.asarray(genes, dtype=object)
    keep_idx_all = np.where(gene_keep_mask)[0].astype(int)
    keep_genes = genes_arr[keep_idx_all].astype(str).tolist()
    clusters = sort_cluster_labels(adata_umap.obs["new_cluster"].astype(str).unique().tolist())
    cl_arr = adata_umap.obs["new_cluster"].astype(str).values
    cluster_masks = [(cl_arr == str(cl)) for cl in clusters]
    n_cluster = len(clusters)
    n_gene = len(keep_idx_all)
    mean_expr = np.zeros((n_cluster, n_gene), dtype=np.float32)
    pct_expr = np.zeros((n_cluster, n_gene), dtype=np.float32)
    X_mmap = np.load(log1p_npy, mmap_mode="r")
    log("[RNA] full log1p matrix shape =", X_mmap.shape)
    log("[RNA] kept gene universe =", n_gene)
    for st in range(0, n_gene, gene_chunk):
        ed = min(st + gene_chunk, n_gene)
        cols = keep_idx_all[st:ed]
        X = np.asarray(X_mmap[np.ix_(row_idx, cols)], dtype=np.float32)
        for ci, m in enumerate(cluster_masks):
            if int(m.sum()) == 0:
                continue
            Xi = X[m, :]
            mean_expr[ci, st:ed] = Xi.mean(axis=0).astype(np.float32)
            pct_expr[ci, st:ed] = (Xi > 0).mean(axis=0).astype(np.float32)
        if st == 0 or ed == n_gene or (st // gene_chunk) % 10 == 0:
            log(f"[RNA] processed genes {ed}/{n_gene}")
    spec_z = zscore_by_gene(mean_expr)
    mean_expr_scaled = np.zeros_like(mean_expr, dtype=np.float32)
    for j in range(n_gene):
        mean_expr_scaled[:, j] = minmax_1d(mean_expr[:, j], constant_value=0.0).astype(np.float32)
    return keep_genes, mean_expr, pct_expr, spec_z, mean_expr_scaled


def get_target_expression(row_idx: np.ndarray, genes: List[str]) -> np.ndarray:
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    if TARGET_GENE not in gene_to_idx:
        # case-insensitive fallback
        upper_map = {str(g).upper(): i for i, g in enumerate(genes)}
        if TARGET_GENE.upper() not in upper_map:
            raise KeyError(f"{TARGET_GENE} not found in CA_all gene list.")
        gi = upper_map[TARGET_GENE.upper()]
    else:
        gi = gene_to_idx[TARGET_GENE]
    X = np.load(LOG1P_NPY, mmap_mode="r")
    return np.asarray(X[row_idx, gi], dtype=np.float32)

# ============================================================
# PROPAGATION SOURCE AND GRAPH
# ============================================================

def compute_source_seed_scores(
    adata_umap: ad.AnnData,
    clusters: List[str],
    target_expr: np.ndarray,
    cluster_label_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cl_arr = adata_umap.obs["new_cluster"].astype(str).values
    positive_vals = target_expr[target_expr > 0]
    if len(positive_vals) >= 20:
        pos_thr = float(np.quantile(positive_vals, SOURCE_POS_Q))
    else:
        pos_thr = float(np.quantile(target_expr, SOURCE_POS_Q))
    target_pos = target_expr >= pos_thr
    rows = []
    for ci, cl in enumerate(clusters):
        m = cl_arr == str(cl)
        vals = target_expr[m]
        rows.append({
            "cluster_label": str(cl),
            "cluster_idx": ci,
            "n_cells": int(m.sum()),
            "target_mean_expr": float(np.nanmean(vals)) if len(vals) else 0.0,
            "target_positive_fraction": float(np.mean(target_pos[m])) if int(m.sum()) else 0.0,
        })
    df = pd.DataFrame(rows)
    target_mean_scaled = minmax_1d(df["target_mean_expr"].values, constant_value=0.0)
    target_pct_scaled = minmax_1d(df["target_positive_fraction"].values, constant_value=0.0)
    target_spec = np.maximum(zscore_1d(df["target_mean_expr"].values), 0.0)
    target_spec_scaled = minmax_1d(target_spec, constant_value=0.0)
    seed_raw = (
        safe_pow(target_mean_scaled, SOURCE_TARGET_MEAN_POWER)
        * safe_pow(target_pct_scaled, SOURCE_TARGET_PCT_POWER)
        * (0.35 + 0.65 * safe_pow(target_spec_scaled, SOURCE_TARGET_SPEC_POWER))
    )
    seed_score = minmax_1d(seed_raw, constant_value=0.0)
    if np.nanmax(seed_score) <= 1e-12:
        seed_score = target_mean_scaled.copy()
    df["target_mean_scaled"] = target_mean_scaled
    df["target_positive_fraction_scaled"] = target_pct_scaled
    df["target_specificity_scaled"] = target_spec_scaled
    df["source_seed_score"] = seed_score
    label_cols = ["cluster_label", "cluster_display", "major_celltype", "xtick_label"]
    df = df.merge(cluster_label_df[[c for c in label_cols if c in cluster_label_df.columns]], on="cluster_label", how="left")
    order = np.argsort(seed_score)[::-1]
    maxv = float(np.nanmax(seed_score)) if len(seed_score) else 0.0
    source_mask = np.zeros_like(seed_score, dtype=bool)
    if maxv > 0:
        candidates = [i for i in order if seed_score[i] >= SOURCE_MIN_RELATIVE_TO_MAX * maxv]
    else:
        candidates = list(order[:SOURCE_MIN_N_CLUSTERS])
    candidates = candidates[:SOURCE_MAX_N_CLUSTERS]
    if len(candidates) < SOURCE_MIN_N_CLUSTERS and len(order) > 0:
        candidates = list(order[:SOURCE_MIN_N_CLUSTERS])
    source_mask[candidates] = True
    df["is_inferred_source"] = source_mask
    log("[SOURCE] target positive threshold =", pos_thr)
    log("[SOURCE] inferred source clusters, based only on target expression statistics:")
    show_cols = ["cluster_label", "cluster_display", "major_celltype", "n_cells", "target_mean_expr", "target_positive_fraction", "source_seed_score", "is_inferred_source"]
    log(df[[c for c in show_cols if c in df.columns]].sort_values("source_seed_score", ascending=False).to_string(index=False))
    return df, seed_score.astype(np.float64), source_mask


def build_cluster_centroids(adata_umap: ad.AnnData, clusters: List[str]) -> np.ndarray:
    xy = np.asarray(adata_umap.obsm["X_umap"], dtype=np.float64)
    cl_arr = adata_umap.obs["new_cluster"].astype(str).values
    cent = np.zeros((len(clusters), 2), dtype=np.float64)
    for i, cl in enumerate(clusters):
        m = cl_arr == str(cl)
        if int(m.sum()) == 0:
            cent[i, :] = np.nan
        else:
            cent[i, :] = np.nanmedian(xy[m, :], axis=0)
    return cent


def dijkstra_multi_source(adj: np.ndarray, sources: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full(n, np.inf, dtype=np.float64)
    pq: List[Tuple[float, int]] = []
    for i in np.where(sources)[0].tolist():
        dist[i] = 0.0
        heapq.heappush(pq, (0.0, int(i)))
    if len(pq) == 0:
        dist[0] = 0.0
        heapq.heappush(pq, (0.0, 0))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(n):
            w = adj[u, v]
            if not np.isfinite(w) or w <= 0:
                continue
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    finite = np.isfinite(dist)
    if finite.sum() < n:
        # Fallback for disconnected clusters: use Euclidean distance to nearest source centroid in graph distance scale.
        max_f = float(np.nanmax(dist[finite])) if finite.sum() else 1.0
        dist[~finite] = max_f + 1.0
    return dist


def compute_cluster_propagation(
    adata_umap: ad.AnnData,
    clusters: List[str],
    seed_score: np.ndarray,
    source_mask: np.ndarray,
    cluster_label_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    cent = build_cluster_centroids(adata_umap, clusters)
    n = len(clusters)
    if n <= 1:
        adj = np.zeros((n, n), dtype=np.float64)
        dist = np.zeros(n, dtype=np.float64)
    else:
        k = min(CLUSTER_GRAPH_K, max(1, n - 1))
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(cent)
        dmat, imat = nn.kneighbors(cent, return_distance=True)
        adj = np.full((n, n), np.inf, dtype=np.float64)
        sensitivity = 0.20 + 0.80 * minmax_1d(seed_score, constant_value=0.0)
        for i in range(n):
            for dd, j in zip(dmat[i, 1:], imat[i, 1:]):
                j = int(j)
                sens_pair = math.sqrt(max(0.0, float(sensitivity[i])) * max(0.0, float(sensitivity[j])))
                denom = 1.0 + CLUSTER_SENSITIVITY_EDGE_BONUS * sens_pair
                w = (float(dd) ** CLUSTER_GRAPH_DISTANCE_POWER) / max(denom, 1e-6)
                adj[i, j] = min(adj[i, j], w)
                adj[j, i] = min(adj[j, i], w)
        dist = dijkstra_multi_source(adj, source_mask)
    ptime = minmax_1d(dist, constant_value=0.0)
    source_potential = 1.0 - ptime
    order_idx = np.argsort(ptime)
    rank = np.empty_like(order_idx)
    rank[order_idx] = np.arange(len(order_idx))
    prop_df = pd.DataFrame({
        "cluster_label": clusters,
        "cluster_idx": np.arange(n),
        "propagation_distance_raw": dist,
        "propagation_time": ptime,
        "source_potential": source_potential,
        "propagation_rank": rank.astype(int),
        "is_inferred_source": source_mask.astype(int),
        "source_seed_score": seed_score,
        "centroid_umap1": cent[:, 0],
        "centroid_umap2": cent[:, 1],
    })
    label_cols = ["cluster_label", "cluster_display", "major_celltype", "xtick_label", "n_cells"]
    prop_df = prop_df.merge(cluster_label_df[[c for c in label_cols if c in cluster_label_df.columns]], on="cluster_label", how="left")
    prop_df = prop_df.sort_values("propagation_rank").reset_index(drop=True)
    log("[PROPAGATION] cluster order inferred from source score + UMAP graph:")
    show_cols = ["propagation_rank", "cluster_label", "cluster_display", "major_celltype", "n_cells", "propagation_time", "source_seed_score", "is_inferred_source"]
    log(prop_df[[c for c in show_cols if c in prop_df.columns]].to_string(index=False))
    return prop_df, adj, cent, ptime

# ============================================================
# CELL-LEVEL RESPONSE AND VECTOR FIELD
# ============================================================

def compute_stage5_module_score_cells(
    row_idx: np.ndarray,
    genes: List[str],
    impact_df: pd.DataFrame,
    topn: int = 40,
) -> Tuple[np.ndarray, pd.DataFrame]:
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows = []
    for _, r in impact_df.sort_values("perturb_impact", ascending=False).iterrows():
        g = str(r["node"])
        if g in gene_to_idx and not is_target_gene(g):
            rows.append({"gene": g, "gene_idx": int(gene_to_idx[g]), "weight": float(r["perturb_impact"])})
        if len(rows) >= topn:
            break
    module_df = pd.DataFrame(rows)
    if len(module_df) == 0:
        return np.zeros(len(row_idx), dtype=np.float32), module_df
    weights = module_df["weight"].values.astype(np.float64)
    weights = minmax_1d(weights, constant_value=0.0) + 1e-3
    weights = weights / np.sum(weights)
    X = np.load(LOG1P_NPY, mmap_mode="r")
    cols = module_df["gene_idx"].values.astype(int)
    Xm = np.asarray(X[np.ix_(row_idx, cols)], dtype=np.float32)
    # z-score each module gene across cells
    mu = np.nanmean(Xm, axis=0, keepdims=True)
    sd = np.nanstd(Xm, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xz = (Xm - mu) / sd
    score = np.asarray(Xz @ weights, dtype=np.float32)
    score = zscore_1d(score).astype(np.float32)
    module_df["weight_used"] = weights
    return score, module_df


def compute_cell_vector_field(
    adata_umap: ad.AnnData,
    clusters: List[str],
    cluster_time: np.ndarray,
    target_expr: np.ndarray,
    module_score: np.ndarray,
) -> pd.DataFrame:
    xy = np.asarray(adata_umap.obsm["X_umap"], dtype=np.float64)
    cl_arr = adata_umap.obs["new_cluster"].astype(str).values
    cl_to_i = {str(c): i for i, c in enumerate(clusters)}
    cell_time = np.asarray([cluster_time[cl_to_i[str(c)]] for c in cl_arr], dtype=np.float64)
    target_scaled = minmax_1d(target_expr, constant_value=0.0)
    response_scaled = minmax_1d(module_score, constant_value=0.0)
    cell_response = minmax_1d(0.45 * target_scaled + 0.55 * response_scaled, constant_value=0.0)
    # kNN in UMAP. For each cell, arrow points toward neighboring cells with later propagation time.
    k = min(CELL_VECTOR_KNN + 1, max(2, xy.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(xy)
    dist, idx = nn.kneighbors(xy, return_distance=True)
    dx = np.zeros(xy.shape[0], dtype=np.float64)
    dy = np.zeros(xy.shape[0], dtype=np.float64)
    for i in range(xy.shape[0]):
        js = idx[i, 1:]
        ds = dist[i, 1:]
        later = cell_time[js] > cell_time[i] + CELL_VECTOR_LATER_TIME_EPS
        if later.sum() == 0:
            # if already terminal, use local response gradient as fallback
            later = cell_response[js] < cell_response[i]
        if later.sum() == 0:
            continue
        js2 = js[later]
        ds2 = ds[later]
        dt = np.maximum(cell_time[js2] - cell_time[i], 0.0)
        w = np.exp(-ds2 / (np.nanmedian(ds2) + 1e-6)) * (0.25 + 0.75 * (dt + 1e-6))
        w = w * (0.25 + 0.75 * np.power(cell_response[js2], CELL_VECTOR_RESPONSE_POWER))
        if np.sum(w) <= 1e-12:
            continue
        v = (xy[js2, :] - xy[i, :])
        vv = (v * w[:, None]).sum(axis=0) / (np.sum(w) + 1e-12)
        dx[i], dy[i] = vv[0], vv[1]
    out = pd.DataFrame({
        "cell_id": adata_umap.obs_names.astype(str),
        "cluster_label": cl_arr,
        "UMAP1": xy[:, 0],
        "UMAP2": xy[:, 1],
        "target_expr": target_expr.astype(float),
        "stage5_module_score": module_score.astype(float),
        "cell_response_score": cell_response.astype(float),
        "propagation_time": cell_time.astype(float),
        "vector_dx": dx.astype(float),
        "vector_dy": dy.astype(float),
    })
    return out


def aggregate_grid(cell_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    x = cell_df["UMAP1"].values.astype(float)
    y = cell_df["UMAP2"].values.astype(float)
    dx = cell_df["vector_dx"].values.astype(float)
    dy = cell_df["vector_dy"].values.astype(float)
    sc = cell_df["cell_response_score"].values.astype(float)
    t = cell_df["propagation_time"].values.astype(float)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad_x = 0.02 * max(1e-6, xmax - xmin)
    pad_y = 0.02 * max(1e-6, ymax - ymin)
    x_edges = np.linspace(xmin - pad_x, xmax + pad_x, GRID_NX + 1)
    y_edges = np.linspace(ymin - pad_y, ymax + pad_y, GRID_NY + 1)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, GRID_NX - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, GRID_NY - 1)
    count = np.zeros((GRID_NY, GRID_NX), dtype=np.int32)
    sum_dx = np.zeros((GRID_NY, GRID_NX), dtype=np.float64)
    sum_dy = np.zeros((GRID_NY, GRID_NX), dtype=np.float64)
    sum_sc = np.zeros((GRID_NY, GRID_NX), dtype=np.float64)
    sum_t = np.zeros((GRID_NY, GRID_NX), dtype=np.float64)
    for i in range(len(x)):
        gx = int(ix[i]); gy = int(iy[i])
        count[gy, gx] += 1
        sum_dx[gy, gx] += dx[i]
        sum_dy[gy, gx] += dy[i]
        sum_sc[gy, gx] += sc[i]
        sum_t[gy, gx] += t[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_dx = np.where(count > 0, sum_dx / np.maximum(count, 1), np.nan)
        mean_dy = np.where(count > 0, sum_dy / np.maximum(count, 1), np.nan)
        mean_sc = np.where(count > 0, sum_sc / np.maximum(count, 1), np.nan)
        mean_t = np.where(count > 0, sum_t / np.maximum(count, 1), np.nan)
    valid = count >= int(GRID_MIN_CELLS)
    mean_dx[~valid] = np.nan
    mean_dy[~valid] = np.nan
    mean_sc[~valid] = np.nan
    mean_t[~valid] = np.nan
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(xc, yc)
    return {
        "x_edges": x_edges, "y_edges": y_edges,
        "XX": XX, "YY": YY,
        "count": count,
        "mean_dx": mean_dx, "mean_dy": mean_dy,
        "mean_response": mean_sc,
        "mean_time": mean_t,
        "valid": valid,
    }

# ============================================================
# RESPONSE HEATMAP MATRICES
# ============================================================

def build_gene_keep_mask(genes: List[str], impact_df: pd.DataFrame) -> np.ndarray:
    if RESTRICT_TO_STAGE5_NODES_ONLY:
        nodes = set(impact_df["node"].astype(str).tolist())
        keep = np.asarray([(g in nodes) and (not is_target_gene(g)) for g in genes], dtype=bool)
    else:
        keep = np.asarray([not is_target_gene(g) for g in genes], dtype=bool)
    return keep


def build_response_matrices(
    clusters: List[str],
    keep_genes: List[str],
    mean_expr: np.ndarray,
    pct_expr: np.ndarray,
    spec_z: np.ndarray,
    mean_expr_scaled: np.ndarray,
    impact_df: pd.DataFrame,
    prop_df: pd.DataFrame,
) -> Dict[str, object]:
    n_cluster, n_gene = mean_expr.shape
    impact_df = impact_df.copy()
    impact_df["node"] = impact_df["node"].astype(str)
    impact_map = impact_df.set_index("node")["perturb_impact"].to_dict()
    state_base_map = impact_df.set_index("node")["state_baseline"].to_dict()
    state_pert_map = impact_df.set_index("node")["state_perturbed"].to_dict()
    delta_map = impact_df.set_index("node")["delta_perturbed_minus_base"].to_dict()
    direct_impact = np.asarray([float(impact_map.get(g, 0.0)) for g in keep_genes], dtype=np.float32)
    direct_flag = np.asarray([1 if g in impact_map else 0 for g in keep_genes], dtype=np.int8)
    direct_impact_norm = minmax_1d(direct_impact, constant_value=0.0).astype(np.float32)
    if RESTRICT_TO_STAGE5_NODES_ONLY:
        impact_weight = direct_impact_norm.copy()
    else:
        impact_weight = np.full((n_gene,), float(NON_STAGE5_IMPACT_FLOOR), dtype=np.float32)
        impact_weight = np.maximum(impact_weight, direct_impact_norm)
    impact_weight = impact_weight * (1.0 + (STAGE5_DIRECT_IMPACT_BOOST - 1.0) * direct_flag.astype(np.float32))
    impact_weight = np.clip(impact_weight, 0.0, 1.0)

    # cluster propagation reach: source and downstream clusters can both respond, but later clusters are downweighted softly.
    prop_time_map = prop_df.set_index("cluster_label")["propagation_time"].to_dict()
    prop_time = np.asarray([float(prop_time_map.get(str(cl), 1.0)) for cl in clusters], dtype=np.float32)
    reach_weight = 1.0 - PROPAGATION_REACH_WEIGHT * prop_time
    reach_weight = np.clip(reach_weight, 0.25, 1.0).astype(np.float32)

    spec_pos = np.maximum(spec_z, 0.0).astype(np.float32)
    cluster_specificity = (spec_pos * pct_expr).astype(np.float32)
    cluster_specificity = cluster_specificity * (1.0 + float(MEAN_EXPR_BONUS_WEIGHT) * mean_expr_scaled)
    raw_response = (
        safe_pow(impact_weight[None, :], ALPHA_IMPACT)
        * safe_pow(cluster_specificity, BETA_CLUSTER_SPECIFICITY)
        * reach_weight[:, None]
    ).astype(np.float64)
    invalid = (pct_expr < float(MIN_PCT_EXPR)) | (spec_z <= float(MIN_SPEC_Z))
    raw_response[invalid] = 0.0
    # target gene excluded if accidentally present.
    for j, g in enumerate(keep_genes):
        if is_target_gene(g):
            raw_response[:, j] = 0.0

    owner_idx = np.argmax(raw_response, axis=0).astype(np.int64)
    best = np.max(raw_response, axis=0).astype(np.float64)
    best_other = np.zeros_like(raw_response, dtype=np.float64)
    for ci in range(n_cluster):
        if n_cluster == 1:
            best_other[ci, :] = 0.0
        else:
            other = [x for x in range(n_cluster) if x != ci]
            best_other[ci, :] = np.max(raw_response[other, :], axis=0)
    ratio = raw_response / (raw_response + best_other + 1e-12)
    ratio = np.clip(ratio, 0.0, 1.0)
    log_raw = np.log1p(float(RAW_SCORE_SCALE_FOR_ROWZ) * raw_response.T).astype(np.float32)
    heatmap_weight = row_zscore(log_raw)
    heatmap_weight = np.clip(heatmap_weight, -float(HEATMAP_Z_CLIP), float(HEATMAP_Z_CLIP)).astype(np.float32)
    positive_rowz = np.maximum(heatmap_weight.T, 0.0)
    owner_bonus = np.power(1e-6 + ratio, float(GAMMA_OWNER_BONUS))
    rowz_bonus = np.power(1e-6 + positive_rowz, float(GAMMA_ROWZ_BONUS))
    selection_score = raw_response * owner_bonus * (1.0 + rowz_bonus)
    selection_score = selection_score * (1.0 + 0.25 * direct_flag[None, :].astype(np.float64))

    gene_meta = pd.DataFrame({
        "gene": keep_genes,
        "direct_stage5_flag": direct_flag.astype(int),
        "direct_impact": direct_impact.astype(float),
        "direct_impact_norm": direct_impact_norm.astype(float),
        "impact_weight_used": impact_weight.astype(float),
        "owner_cluster_idx": owner_idx.astype(int),
        "best_raw_response": best.astype(float),
        "state_baseline": [float(state_base_map.get(g, np.nan)) for g in keep_genes],
        "state_perturbed": [float(state_pert_map.get(g, np.nan)) for g in keep_genes],
        "delta_perturbed_minus_base": [float(delta_map.get(g, np.nan)) for g in keep_genes],
    })
    return {
        "raw_response": raw_response,
        "selection_score": selection_score,
        "heatmap_weight_gene_by_cluster": heatmap_weight,
        "ratio": ratio,
        "owner_idx": owner_idx,
        "impact_weight": impact_weight,
        "direct_flag": direct_flag,
        "gene_meta": gene_meta,
        "propagation_reach_weight": reach_weight,
    }


def order_all_genes_by_owner_and_propagation(
    clusters: List[str],
    keep_genes: List[str],
    matrices: Dict[str, object],
    prop_df: pd.DataFrame,
) -> Tuple[List[int], pd.DataFrame]:
    raw_response = np.asarray(matrices["raw_response"], dtype=np.float64)
    selection_score = np.asarray(matrices["selection_score"], dtype=np.float64)
    owner_idx = np.asarray(matrices["owner_idx"], dtype=np.int64)
    best = np.max(raw_response, axis=0)
    prop_rank_map = prop_df.set_index("cluster_label")["propagation_rank"].to_dict()
    cluster_prop_rank = np.asarray([int(prop_rank_map.get(str(cl), 999)) for cl in clusters], dtype=int)
    rows = []
    for gj, g in enumerate(keep_genes):
        oi = int(owner_idx[gj]) if len(clusters) else 0
        owner_cluster = str(clusters[oi]) if len(clusters) else "0"
        owner_rank = int(cluster_prop_rank[oi]) if len(clusters) else 999
        rows.append({
            "gene": str(g),
            "gene_idx_in_keep": int(gj),
            "owner_cluster": owner_cluster,
            "owner_cluster_propagation_rank": owner_rank,
            "best_raw_response": float(best[gj]),
            "best_selection_score": float(np.nanmax(selection_score[:, gj])) if selection_score.size else 0.0,
            "responsive_flag": int(best[gj] > 0),
        })
    order_df = pd.DataFrame(rows)
    order_df = order_df.sort_values(
        ["responsive_flag", "owner_cluster_propagation_rank", "best_selection_score", "best_raw_response", "gene"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    order_df["heatmap_row_index"] = np.arange(len(order_df), dtype=int)
    order_idx = order_df["gene_idx_in_keep"].values.astype(int).tolist()
    return order_idx, order_df


def select_top_genes_per_cluster_by_propagation(
    clusters: List[str],
    keep_genes: List[str],
    mean_expr: np.ndarray,
    pct_expr: np.ndarray,
    spec_z: np.ndarray,
    matrices: Dict[str, object],
    prop_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    selection_score = np.asarray(matrices["selection_score"], dtype=np.float64)
    raw_response = np.asarray(matrices["raw_response"], dtype=np.float64)
    heatmap_weight = np.asarray(matrices["heatmap_weight_gene_by_cluster"], dtype=np.float32)
    ratio = np.asarray(matrices["ratio"], dtype=np.float64)
    owner_idx = np.asarray(matrices["owner_idx"], dtype=np.int64)
    impact_weight = np.asarray(matrices["impact_weight"], dtype=np.float32)
    direct_flag = np.asarray(matrices["direct_flag"], dtype=np.int8)

    prop_order_clusters = prop_df.sort_values("propagation_rank")["cluster_label"].astype(str).tolist()
    cl_to_i = {str(c): i for i, c in enumerate(clusters)}
    used_genes = set()
    top_rows = []
    row_ann = []
    row_indices = []

    for cl in prop_order_clusters:
        if str(cl) not in cl_to_i:
            continue
        ci = cl_to_i[str(cl)]
        fs = selection_score[ci, :]
        order = np.argsort(fs)[::-1]
        picked = 0

        def try_pick(require_owner: bool, allow_used: bool, pass_name: str, picked_count: int):
            local_rows = []
            local_idx = []
            for gj in order:
                if picked_count + len(local_idx) >= TOP_N_GENES_PER_CLUSTER:
                    break
                g = str(keep_genes[int(gj)])
                if is_target_gene(g):
                    continue
                if (not allow_used) and g in used_genes:
                    continue
                if require_owner and int(owner_idx[int(gj)]) != int(ci):
                    continue
                if float(raw_response[ci, gj]) <= 0 or float(fs[gj]) <= 0:
                    continue
                local_idx.append(int(gj))
                local_rows.append({
                    "cluster_label": str(cl),
                    "propagation_rank": int(prop_df.set_index("cluster_label").loc[str(cl), "propagation_rank"]),
                    "rank": int(picked_count + len(local_idx)),
                    "gene": g,
                    "selection_pass": pass_name,
                    "selection_score": float(fs[gj]),
                    "raw_response": float(raw_response[ci, gj]),
                    "heatmap_weight_in_owner_cluster": float(heatmap_weight[gj, ci]),
                    "cluster_response_ratio_vs_best_other": float(ratio[ci, gj]),
                    "impact_weight_used": float(impact_weight[gj]),
                    "direct_stage5_flag": int(direct_flag[gj]),
                    "cluster_mean_expr": float(mean_expr[ci, gj]),
                    "cluster_pct_expr": float(pct_expr[ci, gj]),
                    "cluster_spec_z": float(spec_z[ci, gj]),
                    "owner_cluster_by_raw_response": str(clusters[int(owner_idx[gj])]),
                })
            return local_rows, local_idx

        rows1, idx1 = try_pick(require_owner=True, allow_used=UNIQUE_GENE_ROWS, pass_name="primary_owner_unique", picked_count=picked)
        top_rows.extend(rows1); row_indices.extend(idx1)
        for gj in idx1:
            used_genes.add(str(keep_genes[gj]))
        picked += len(idx1)

        if picked < TOP_N_GENES_PER_CLUSTER:
            rows2, idx2 = try_pick(require_owner=False, allow_used=UNIQUE_GENE_ROWS, pass_name="fallback_nonowner_unique", picked_count=picked)
            top_rows.extend(rows2); row_indices.extend(idx2)
            for gj in idx2:
                used_genes.add(str(keep_genes[gj]))
            picked += len(idx2)

        if picked < TOP_N_GENES_PER_CLUSTER and ALLOW_DUPLICATE_IF_NOT_ENOUGH:
            rows3, idx3 = try_pick(require_owner=False, allow_used=True, pass_name="fallback_allow_duplicate", picked_count=picked)
            top_rows.extend(rows3); row_indices.extend(idx3)
            picked += len(idx3)

        cur_rows = [r for r in top_rows if str(r["cluster_label"]) == str(cl)]
        for rr in cur_rows:
            row_ann.append({
                "row_index": len(row_ann),
                "row_block_cluster": str(cl),
                "row_block_label": f"C{cl}_top{TOP_N_GENES_PER_CLUSTER}",
                "propagation_rank": int(rr["propagation_rank"]),
                "gene": rr["gene"],
                "rank_in_cluster": rr["rank"],
                "selection_pass": rr["selection_pass"],
                "selection_score": rr["selection_score"],
                "raw_response": rr["raw_response"],
                "heatmap_weight_in_owner_cluster": rr["heatmap_weight_in_owner_cluster"],
                "direct_stage5_flag": rr["direct_stage5_flag"],
                "owner_cluster_by_raw_response": rr["owner_cluster_by_raw_response"],
            })
        log(f"[SELECT] propagation C{cl}: picked {picked}/{TOP_N_GENES_PER_CLUSTER}")

    top_df = pd.DataFrame(top_rows)
    row_ann_df = pd.DataFrame(row_ann)
    # Rebuild row indices by gene order in annotation, because duplicate rows may be allowed.
    gene_to_idx = {str(g): i for i, g in enumerate(keep_genes)}
    row_indices_final = [gene_to_idx[str(g)] for g in row_ann_df["gene"].astype(str).tolist()] if len(row_ann_df) else []
    return top_df, row_ann_df, row_indices_final

# ============================================================
# PLOTS
# ============================================================

def plot_umap_propagation_vector_field(
    adata_umap: ad.AnnData,
    cell_df: pd.DataFrame,
    grid: Dict[str, np.ndarray],
    prop_df: pd.DataFrame,
    out_base: Path,
):
    xy = np.asarray(adata_umap.obsm["X_umap"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.8, 7.6), dpi=FIG_DPI_VECTOR)
    fig.patch.set_facecolor("white")
    bg = np.asarray(grid["mean_response"], dtype=float)
    finite = bg[np.isfinite(bg)]
    vmax = float(np.nanquantile(finite, 0.98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.pcolormesh(
        grid["x_edges"], grid["y_edges"], np.ma.masked_invalid(bg),
        cmap=make_white_orange_cmap(), shading="auto", vmin=0, vmax=vmax,
        alpha=BG_ALPHA, zorder=1,
    )
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c="#bdbdbd", alpha=0.22, linewidths=0, zorder=2)
    XX = np.asarray(grid["XX"], dtype=float)
    YY = np.asarray(grid["YY"], dtype=float)
    U = np.asarray(grid["mean_dx"], dtype=float)
    V = np.asarray(grid["mean_dy"], dtype=float)
    valid = np.asarray(grid["valid"], dtype=bool) & np.isfinite(U) & np.isfinite(V)
    mag = np.sqrt(U * U + V * V)
    mag_f = mag[valid]
    if mag_f.size:
        scale_norm = float(np.nanquantile(mag_f, 0.90))
        if not np.isfinite(scale_norm) or scale_norm <= 1e-12:
            scale_norm = 1.0
        U = U / scale_norm
        V = V / scale_norm
    ax.quiver(
        XX[valid], YY[valid], U[valid], V[valid],
        angles="xy", scale_units="xy", scale=15.0 / max(ARROW_SCALE, 1e-6),
        width=ARROW_WIDTH, headwidth=3.5, headlength=4.8, headaxislength=4.0,
        color="#1f1f1f", alpha=ARROW_ALPHA, zorder=4,
    )
    # cluster centroids and propagation labels
    for _, r in prop_df.iterrows():
        x = float(r["centroid_umap1"]); y = float(r["centroid_umap2"])
        rank = int(r["propagation_rank"])
        cl = str(r["cluster_label"])
        is_src = int(r.get("is_inferred_source", 0)) == 1
        ax.scatter([x], [y], s=70 if is_src else 44, c=SOFT_PURPLE if is_src else "#ffffff",
                   edgecolors="#111111", linewidths=0.8, zorder=5)
        ax.text(x, y, f"P{rank}\nC{cl}", ha="center", va="center", fontsize=6.8,
                color="white" if is_src else "#111111", zorder=6)
    ax.set_title(f"{TARGET_GENE} KO perturbation propagation on UMAP\nsource inferred from target-expression statistics")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Cell-level KO response intensity")
    save_fig(fig, out_base, dpi=FIG_DPI_VECTOR, tight=True)


def add_compressed_propagation_display_stages(
    prop_df: pd.DataFrame,
    n_stages: int = DISPLAY_PROPAGATION_N_STAGES,
    skip_initial_ranks: int = DISPLAY_SKIP_INITIAL_RANKS,
    skip_initial_fraction: float = DISPLAY_SKIP_INITIAL_FRACTION,
) -> pd.DataFrame:
    """
    Compress original cluster propagation ranks into display stages P0-P{n_stages-1},
    after excluding the earliest source/start segment.

    Important:
    - This is only for UMAP visualization.
    - Original propagation_rank / propagation_time are kept unchanged for downstream heatmap logic.
    - The earliest rank(s), usually the inferred source/start point, are marked as
      display_excluded_initial=1 and are not shown in the P0-P3 cumulative snapshots.
    - The remaining clusters are split as evenly as possible into P0-P3 by propagation rank.

    Example for 9 original ranks and 4 display stages, with skip_initial_ranks=1:
        original P0      -> excluded initial/start segment
        original P1-P2   -> display P0
        original P3-P4   -> display P1
        original P5-P6   -> display P2
        original P7-P8   -> display P3
    """
    prop_df = prop_df.copy()
    if "propagation_rank" not in prop_df.columns:
        raise KeyError("prop_df must contain column: propagation_rank")

    n_stages = int(max(1, n_stages))
    rank = pd.to_numeric(prop_df["propagation_rank"], errors="coerce").fillna(0).astype(int).values
    n_cluster = len(rank)

    if n_cluster <= 0:
        prop_df["display_propagation_stage"] = []
        prop_df["display_propagation_label"] = []
        prop_df["display_stage_rank_range"] = []
        prop_df["display_excluded_initial"] = []
        prop_df["display_skip_initial_ranks_used"] = []
        return prop_df

    # Work out how many earliest rank positions to exclude. Keep at least one rank eligible.
    n_unique_rank = len(np.unique(rank))
    frac_skip = int(math.floor(float(max(0.0, skip_initial_fraction)) * float(max(n_unique_rank, 1))))
    skip_n = int(max(0, int(skip_initial_ranks), frac_skip))
    skip_n = min(skip_n, max(0, n_unique_rank - 1))

    # Eligible = post-start area to be compressed into P0-P3.
    stage = np.full(n_cluster, -1, dtype=int)
    excluded_initial = rank < skip_n
    eligible = ~excluded_initial

    if int(eligible.sum()) <= 0:
        # Defensive fallback; should not happen because skip_n is clipped above.
        eligible[:] = True
        excluded_initial[:] = False
        skip_n = 0

    eligible_indices_sorted = np.where(eligible)[0][np.argsort(rank[eligible])]
    n_eligible = len(eligible_indices_sorted)
    if n_eligible <= 1 or n_stages == 1:
        stage[eligible_indices_sorted] = 0
    else:
        # Equal-count rank compression after skipping the start segment.
        # This is more stable than propagation_time thresholds and prevents P0 from containing
        # only the tiny source cluster.
        for pos, idx in enumerate(eligible_indices_sorted):
            s = int(math.floor(pos * n_stages / max(n_eligible, 1)))
            s = min(max(s, 0), n_stages - 1)
            stage[idx] = s

    prop_df["display_propagation_stage"] = stage.astype(int)
    prop_df["display_excluded_initial"] = excluded_initial.astype(int)
    prop_df["display_skip_initial_ranks_used"] = int(skip_n)
    prop_df["display_propagation_label"] = [
        "excluded_start" if int(x) < 0 else f"{DISPLAY_STAGE_PREFIX}{int(x)}"
        for x in stage
    ]
    prop_df["display_stage_rank_range"] = ""
    for s in range(n_stages):
        m = prop_df["display_propagation_stage"].astype(int).values == s
        if np.any(m):
            rmin = int(np.min(rank[m])); rmax = int(np.max(rank[m]))
            prop_df.loc[m, "display_stage_rank_range"] = f"original_P{rmin}-P{rmax}"
    if np.any(excluded_initial):
        rmin = int(np.min(rank[excluded_initial])); rmax = int(np.max(rank[excluded_initial]))
        prop_df.loc[excluded_initial, "display_stage_rank_range"] = f"excluded_start_original_P{rmin}-P{rmax}"

    log(
        f"[DISPLAY] skip earliest original propagation ranks: {skip_n}; "
        f"compress remaining {int(eligible.sum())} clusters into {n_stages} display stages."
    )
    show_cols = [
        "propagation_rank", "cluster_label", "major_celltype",
        "display_propagation_label", "display_stage_rank_range", "display_excluded_initial"
    ]
    log(prop_df[[c for c in show_cols if c in prop_df.columns]].sort_values("propagation_rank").to_string(index=False))
    return prop_df

def _stage_cmap_and_norm(n_stages: int):
    """
    Keep the original propagation-order color scheme.

    Use continuous viridis for the colorbar, while the plotted cells still take compressed
    display-stage values P0-P3. This keeps the original yellow-green-purple time-gradient
    feeling and avoids a discrete block-style legend.
    """
    n_stages = int(max(1, n_stages))
    cmap = plt.get_cmap("viridis")
    vmax = max(1.0, float(n_stages - 1))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    return cmap, norm


def plot_cluster_propagation_order(adata_umap: ad.AnnData, prop_df: pd.DataFrame, out_base: Path):
    """
    Generate propagation UMAPs with compressed display stages P0-P3.

    Outputs:
    1) out_base.png/svg
       Full UMAP, all cells shown, colored by compressed display stage P0-P3.
    2) out_base_P0_visible.png/svg ... out_base_P3_visible.png/svg
       Cumulative UMAPs. At Pk, only cells belonging to display stages <= Pk are shown;
       future stages and the skipped initial/source segment are completely hidden.
    """
    if "display_propagation_stage" not in prop_df.columns:
        prop_df = add_compressed_propagation_display_stages(prop_df, DISPLAY_PROPAGATION_N_STAGES)

    xy = np.asarray(adata_umap.obsm["X_umap"], dtype=np.float64)
    cl_arr = adata_umap.obs["new_cluster"].astype(str).values
    stage_map = prop_df.set_index("cluster_label")["display_propagation_stage"].to_dict()
    # Cells belonging to excluded initial/source clusters receive stage -1 and are not drawn
    # in the P0-P3 snapshots.
    cell_stage = np.asarray([int(stage_map.get(str(c), -1)) for c in cl_arr], dtype=int)
    n_stages = int(DISPLAY_PROPAGATION_N_STAGES)
    cmap, norm = _stage_cmap_and_norm(n_stages)

    def _draw_one(max_stage: Optional[int], suffix: str, title_extra: str):
        fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=DPI)
        fig.patch.set_facecolor("white")
        if max_stage is None:
            visible = cell_stage >= 0
        else:
            visible = (cell_stage >= 0) & (cell_stage <= int(max_stage))
        # Future stages and the skipped initial/source segment are intentionally not plotted.
        if int(visible.sum()) > 0:
            # Draw earlier stages first, current stage last, so the newest visible front is not buried.
            # IMPORTANT: sort only inside the visible subset. Sorting the full cell_stage array and then
            # indexing np.where(visible)[0] can produce out-of-bounds indices when the skipped start cells
            # are hidden. Also guard against rare obs/UMAP row-length mismatches by keeping only indices
            # that exist in xy.
            idx_visible = np.where(visible)[0]
            idx_visible = idx_visible[idx_visible < xy.shape[0]]
            if idx_visible.size > 0:
                order = np.argsort(cell_stage[idx_visible], kind="mergesort")
                idx_visible = idx_visible[order]
                ax.scatter(
                    xy[idx_visible, 0], xy[idx_visible, 1],
                    c=cell_stage[idx_visible],
                    s=DISPLAY_STAGE_POINT_SIZE,
                    cmap=cmap,
                    norm=norm,
                    linewidths=0,
                    alpha=DISPLAY_STAGE_ALPHA,
                    rasterized=True,
                )
        # Do not draw P/C text labels on UMAP. The stage information is represented
        # only by the viridis time-gradient colorbar.
        ax.set_title(
            f"{TARGET_GENE} KO post-start compressed propagation order ({DISPLAY_STAGE_PREFIX}0-{DISPLAY_STAGE_PREFIX}{n_stages-1}){title_extra}",
            pad=8,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # Keep the original style: a continuous viridis time-gradient colorbar.
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(np.linspace(0, n_stages - 1, 256))
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, ticks=np.arange(n_stages))
        cb.ax.set_yticklabels([f"{DISPLAY_STAGE_PREFIX}{i}" for i in range(n_stages)])
        cb.set_label("Compressed propagation time")
        save_fig(fig, out_base.with_name(out_base.name + suffix), dpi=DPI, tight=True)

    # Full overview with all post-start P0-P3 stages visible.
    _draw_one(max_stage=None, suffix="", title_extra=" | post-start stages visible")

    # Cumulative snapshots after skipping the start segment: P0 only, P0-P1, P0-P2, P0-P3.
    for s in range(DISPLAY_PROPAGATION_N_STAGES):
        _draw_one(
            max_stage=s,
            suffix=f"_{DISPLAY_STAGE_PREFIX}{s}_visible",
            title_extra=f" | visible: {DISPLAY_STAGE_PREFIX}0-{DISPLAY_STAGE_PREFIX}{s}",
        )


def plot_heatmap_matrix(
    mat_gene_by_cluster: np.ndarray,
    row_ann: pd.DataFrame,
    clusters_ordered: List[str],
    prop_df: pd.DataFrame,
    out_base: Path,
    title: str,
    max_height: float = 30.0,
    show_block_lines: bool = True,
):
    n_rows, n_cols = mat_gene_by_cluster.shape
    fig_h = min(max_height, max(7.5, n_rows / 850.0))
    fig_w = max(6.0, min(13.5, n_cols * 0.78 + 3.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    im = ax.imshow(
        mat_gene_by_cluster,
        aspect="auto",
        interpolation="nearest",
        cmap=make_diverging_cmap(),
        vmin=-float(HEATMAP_Z_CLIP),
        vmax=float(HEATMAP_Z_CLIP),
    )
    label_map = prop_df.set_index("cluster_label")["xtick_label"].to_dict() if "xtick_label" in prop_df.columns else {}
    xticks = []
    for cl in clusters_ordered:
        rank = int(prop_df.set_index("cluster_label").loc[str(cl), "propagation_rank"])
        lab = label_map.get(str(cl), f"C{cl}")
        xticks.append(f"P{rank} | {lab}")
    ax.set_xticks(np.arange(len(clusters_ordered)))
    ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel("Genes ordered by perturbation-response owner cluster")
    ax.set_title(title, pad=10)
    if show_block_lines and row_ann is not None and len(row_ann) > 0 and "row_block_cluster" in row_ann.columns:
        blocks = row_ann["row_block_cluster"].astype(str).values
        for i in range(1, len(blocks)):
            if blocks[i] != blocks[i - 1]:
                ax.axhline(i - 0.5, color="#111111", lw=0.35, alpha=0.28)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Row-wise response z-score")
    save_fig(fig, out_base, dpi=DPI, tight=True)

# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 100)
    log("[START] HNSCC CA_all SOX2 KO UMAP propagation direction + cluster-gene heatmap")
    log("=" * 100)
    stage5_dir = resolve_stage5_dir()
    out_dir = stage5_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    log("[STAGE5_DIR]", stage5_dir)
    log("[OUT_DIR]", out_dir)

    adata_umap, row_idx, genes, cluster_label_df = load_umap_and_clusters()
    clusters = sort_cluster_labels(adata_umap.obs["new_cluster"].astype(str).unique().tolist())
    cluster_label_df.to_csv(out_dir / f"{TARGET_GENE}_cluster_labels.csv", index=False)
    assert LOG1P_NPY.exists(), f"Missing log1p matrix: {LOG1P_NPY}"
    assert META_NPZ.exists(), f"Missing meta npz: {META_NPZ}"

    impact_df = load_stage5_impact(stage5_dir)
    impact_df.to_csv(out_dir / f"{TARGET_GENE}_stage5_impact_used.csv", index=False)

    target_expr = get_target_expression(row_idx, genes)
    source_df, seed_score, source_mask = compute_source_seed_scores(adata_umap, clusters, target_expr, cluster_label_df)
    source_df.to_csv(out_dir / f"{TARGET_GENE}_source_seed_score_by_cluster.csv", index=False)

    prop_df, cluster_adj, centroids, cluster_time = compute_cluster_propagation(
        adata_umap, clusters, seed_score, source_mask, cluster_label_df
    )
    # Add post-start compressed display stages P0-P3 for UMAP snapshots only.
    # The earliest source/start rank is excluded from display by DISPLAY_SKIP_INITIAL_RANKS.
    # Original propagation_rank / propagation_time are kept unchanged for downstream analysis.
    prop_df = add_compressed_propagation_display_stages(
        prop_df,
        n_stages=DISPLAY_PROPAGATION_N_STAGES,
        skip_initial_ranks=DISPLAY_SKIP_INITIAL_RANKS,
        skip_initial_fraction=DISPLAY_SKIP_INITIAL_FRACTION,
    )
    prop_df.to_csv(out_dir / f"{TARGET_GENE}_cluster_propagation_order.csv", index=False)
    pd.DataFrame(cluster_adj, index=[f"C{x}" for x in clusters], columns=[f"C{x}" for x in clusters]).to_csv(
        out_dir / f"{TARGET_GENE}_cluster_graph_edge_weight_matrix.csv"
    )

    module_score, module_df = compute_stage5_module_score_cells(row_idx, genes, impact_df, topn=40)
    module_df.to_csv(out_dir / f"{TARGET_GENE}_stage5_module_genes_for_cell_response.csv", index=False)
    cell_df = compute_cell_vector_field(adata_umap, clusters, cluster_time, target_expr, module_score)
    cell_df.to_csv(out_dir / f"{TARGET_GENE}_cell_level_propagation_response.csv", index=False)
    grid = aggregate_grid(cell_df)
    plot_umap_propagation_vector_field(
        adata_umap, cell_df, grid, prop_df,
        out_dir / f"{TARGET_GENE}_umap_ko_propagation_vector_field"
    )
    plot_cluster_propagation_order(
        adata_umap, prop_df,
        out_dir / f"{TARGET_GENE}_umap_cluster_propagation_order"
    )

    log("[HEATMAP] compute cluster expression features over all genes ...")
    gene_keep_mask = build_gene_keep_mask(genes, impact_df)
    keep_genes, mean_expr, pct_expr, spec_z, mean_expr_scaled = compute_cluster_expression_features_chunked(
        LOG1P_NPY, row_idx, genes, adata_umap, gene_keep_mask, gene_chunk=GENE_CHUNK
    )
    matrices = build_response_matrices(
        clusters=clusters,
        keep_genes=keep_genes,
        mean_expr=mean_expr,
        pct_expr=pct_expr,
        spec_z=spec_z,
        mean_expr_scaled=mean_expr_scaled,
        impact_df=impact_df,
        prop_df=prop_df,
    )
    gene_meta = matrices["gene_meta"]
    assert isinstance(gene_meta, pd.DataFrame)
    gene_meta.to_csv(out_dir / f"{TARGET_GENE}_response_gene_meta.csv", index=False)

    prop_order_clusters = prop_df.sort_values("propagation_rank")["cluster_label"].astype(str).tolist()
    cluster_to_i = {str(c): i for i, c in enumerate(clusters)}
    cluster_order_idx = [cluster_to_i[str(c)] for c in prop_order_clusters]

    # All-gene heatmap: all kept genes ordered by response owner and propagation order.
    all_order_idx, all_order_df = order_all_genes_by_owner_and_propagation(clusters, keep_genes, matrices, prop_df)
    all_order_df.to_csv(out_dir / f"{TARGET_GENE}_all_gene_heatmap_row_order.csv", index=False)
    heatmap_weight = np.asarray(matrices["heatmap_weight_gene_by_cluster"], dtype=np.float32)
    all_mat = heatmap_weight[np.asarray(all_order_idx, dtype=int), :][:, cluster_order_idx]
    all_matrix_df = pd.DataFrame(all_mat, columns=[f"P{int(prop_df.set_index('cluster_label').loc[str(cl), 'propagation_rank'])}_C{cl}" for cl in prop_order_clusters])
    all_matrix_df.insert(0, "gene", [keep_genes[i] for i in all_order_idx])
    all_matrix_df.to_csv(out_dir / f"{TARGET_GENE}_all_genes_cluster_response_heatmap_matrix.csv", index=False)
    # Make row annotation with owner blocks for all genes.
    all_row_ann = all_order_df.rename(columns={"owner_cluster": "row_block_cluster"}).copy()
    plot_heatmap_matrix(
        all_mat,
        row_ann=all_row_ann,
        clusters_ordered=prop_order_clusters,
        prop_df=prop_df,
        out_base=out_dir / f"{TARGET_GENE}_all_genes_cluster_response_heatmap",
        title=f"{TARGET_GENE} KO propagation-ordered all-gene response heatmap",
        max_height=34.0,
        show_block_lines=True,
    )

    # Top1000-per-cluster block heatmap: stronger visual block separation.
    top_df, row_ann_df, top_row_indices = select_top_genes_per_cluster_by_propagation(
        clusters=clusters,
        keep_genes=keep_genes,
        mean_expr=mean_expr,
        pct_expr=pct_expr,
        spec_z=spec_z,
        matrices=matrices,
        prop_df=prop_df,
    )
    top_df.to_csv(out_dir / f"{TARGET_GENE}_cluster_top1000_response_genes_by_propagation.csv", index=False)
    row_ann_df.to_csv(out_dir / f"{TARGET_GENE}_top1000_block_heatmap_row_annotation.csv", index=False)
    top_mat = heatmap_weight[np.asarray(top_row_indices, dtype=int), :][:, cluster_order_idx] if len(top_row_indices) else np.zeros((0, len(cluster_order_idx)))
    top_matrix_df = pd.DataFrame(top_mat, columns=[f"P{int(prop_df.set_index('cluster_label').loc[str(cl), 'propagation_rank'])}_C{cl}" for cl in prop_order_clusters])
    if len(row_ann_df):
        top_matrix_df.insert(0, "gene", row_ann_df["gene"].astype(str).values)
        top_matrix_df.insert(0, "row_block_cluster", row_ann_df["row_block_cluster"].astype(str).values)
    top_matrix_df.to_csv(out_dir / f"{TARGET_GENE}_top1000_block_response_heatmap_matrix.csv", index=False)
    plot_heatmap_matrix(
        top_mat,
        row_ann=row_ann_df,
        clusters_ordered=prop_order_clusters,
        prop_df=prop_df,
        out_base=out_dir / f"{TARGET_GENE}_top1000_block_response_heatmap",
        title=f"{TARGET_GENE} KO top1000 response genes per propagation cluster",
        max_height=30.0,
        show_block_lines=True,
    )

    # Preview top200 per cluster, easier to inspect.
    if len(row_ann_df) > 0:
        preview_rows = []
        for cl in prop_order_clusters:
            sub = row_ann_df[row_ann_df["row_block_cluster"].astype(str) == str(cl)].head(TOP_N_PREVIEW_PER_CLUSTER)
            preview_rows.append(sub)
        preview_ann = pd.concat(preview_rows, axis=0, ignore_index=True) if preview_rows else pd.DataFrame()
        if len(preview_ann) > 0:
            gene_to_keep_idx = {str(g): i for i, g in enumerate(keep_genes)}
            preview_idx = [gene_to_keep_idx[str(g)] for g in preview_ann["gene"].astype(str).tolist()]
            preview_mat = heatmap_weight[np.asarray(preview_idx, dtype=int), :][:, cluster_order_idx]
            preview_ann.to_csv(out_dir / f"{TARGET_GENE}_top{TOP_N_PREVIEW_PER_CLUSTER}_preview_row_annotation.csv", index=False)
            plot_heatmap_matrix(
                preview_mat,
                row_ann=preview_ann,
                clusters_ordered=prop_order_clusters,
                prop_df=prop_df,
                out_base=out_dir / f"{TARGET_GENE}_top{TOP_N_PREVIEW_PER_CLUSTER}_preview_response_heatmap",
                title=f"{TARGET_GENE} KO preview: top{TOP_N_PREVIEW_PER_CLUSTER} response genes per propagation cluster",
                max_height=22.0,
                show_block_lines=True,
            )

    qc = pd.DataFrame([{
        "target_gene": TARGET_GENE,
        "mode": MODE,
        "n_cells": int(adata_umap.n_obs),
        "n_clusters": int(len(clusters)),
        "n_genes_kept": int(len(keep_genes)),
        "n_stage5_impact_nodes": int(len(impact_df)),
        "n_inferred_source_clusters": int(np.sum(source_mask)),
        "source_clusters": ";".join([str(clusters[i]) for i in np.where(source_mask)[0].tolist()]),
        "top_source_cluster": str(clusters[int(np.argmax(seed_score))]) if len(seed_score) else "",
        "top_source_score": float(np.nanmax(seed_score)) if len(seed_score) else np.nan,
        "top1000_rows": int(len(row_ann_df)),
        "out_dir": str(out_dir),
    }])
    qc.to_csv(out_dir / f"{TARGET_GENE}_algorithm_qc_summary.csv", index=False)
    log("[DONE] all outputs saved under:", out_dir)
    log("=" * 100)


if __name__ == "__main__":
    main()
