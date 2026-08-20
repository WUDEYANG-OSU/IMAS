# -*- coding: utf-8 -*-
"""
hnscc_caall_stage5_targetKO_celltypeN_cluster_response_top1000_heatmap.py

目标：
- 参照“25.CA-SOX2敲除后cluster基因umap.docx”的 Stage5 perturbation + CA_all cluster 表达特征思路；
- 但不再画 top1 UMAP，而是设计 cluster-specific perturbation response weight；
- 生成热图：横轴 = 原始 cell.type 内部注释映射成的 C0-Cn；纵轴 = 各 cell.type 响应 top1000 genes 组成的 gene blocks；
- 纵轴不显示 gene 名，只显示 cluster block 分隔；
- 颜色 = cluster-specific perturbation response weight：
  高于该 gene 跨 cluster 背景的 cluster 为 Nature red；
  低于该 gene 跨 cluster 背景的 cluster 为 Nature blue。

核心算法设计：
1) gene universe：默认使用 CA_all 全部 genes.tsv 中的基因，而不只限 Stage5 网络节点。
   - Stage5 直接节点使用真实 perturbation impact；
   - 非 Stage5 节点给予较低 background impact floor，使热图可以形成 top1000 gene blocks。
2) raw_response(cluster, gene)
   = impact_weight(gene)^ALPHA
     * [positive_cluster_specificity(cluster,gene) * pct_expr(cluster,gene) * mean_expr_bonus]^BETA
3) selection_score(cluster,gene)
   = raw_response(cluster,gene)
     * owner_bonus(cluster,gene)
     * positive_row_z_bonus(cluster,gene)
   其中 owner_bonus 鼓励每个 gene 主要归属到一个 cluster，从而形成块状红色。
4) heatmap_weight(gene, cluster)
   = row-wise zscore(log1p(raw_response * SCALE))
   - 同一 gene 横向比较不同 cluster 的 response；
   - row-z 高：红；row-z 低：蓝；
   - 这样每个 cluster 的 top1000 gene block 主要在自己的 cluster 列显红，其余列多为蓝或白，少数共享响应为浅红。

输出：
- CA_all_originalCelltype_cluster_response_top1000_genes.csv
- CA_all_originalCelltype_heatmap_row_annotation.csv
- CA_all_originalCelltype_heatmap_weight_matrix.csv
- CA_all_originalCelltype_cluster_response_top1000_heatmap.png/svg
- CA_all_originalCelltype_cluster_response_top1000_rowblock_annotation.png/svg
- CA_all_originalCelltype_cluster_response_top200_preview_heatmap.png/svg
- CA_all_originalCelltype_cluster_response_top200_preview_rowblock_annotation.png/svg
- CA_all_originalCelltype_cluster_celltype_labels.csv
- CA_all_originalCelltype_cluster_response_algorithm_qc.csv
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ============================================================
# 0) CONFIG
# ============================================================
plt.rcParams["font.family"] = "DejaVu Sans"

# 按需修改：SOX2 / FGFR1 / AXL / MET / KIT / PDGFA / IGF2R / CD47 等
TARGET_GENE = "SOX2"
TARGET_GENE_UPPER = TARGET_GENE.upper()

# "ko" 或 "oe"
MODE = "ko"

# impact mode:
# auto: ko 用 baseline - perturbed；oe 用 perturbed - baseline
# down: baseline - perturbed
# up: perturbed - baseline
# abs: abs(perturbed - baseline)
IMPACT_MODE = "auto"

BASE = get_project_root()

# ---------- CA_all root：兼容 Worldword / WorldWord ----------
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
    raise FileNotFoundError(
        "Cannot find Exploit-HNSCC processed_CA/RNA_CA_all under:\n"
        + "\n".join(str(x) for x in EXPLOIT_ROOT_CANDIDATES)
    )

RNA_DIR = EXPLOIT_ROOT / "processed_CA" / "RNA_CA_all"
ASSET_DIR = EXPLOIT_ROOT / "assets_CA"
LOG1P_NPY = RNA_DIR / "CA_all.X_rna_log1p_norm_dense.npy"
META_NPZ = RNA_DIR / "CA_all.rna_meta.npz"
GENES_TSV = ASSET_DIR / "genes.tsv"

# ---------- 使用 celltypeN-guided recluster UMAP / cluster ----------
RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_rna_only_stageB_clusterCorrection_HNSCC_CAall_only"
)
RUN_STAGEB_DATASET_DIR = RUN_STAGEB_DIR / "HNSCC-CA_all"
RECLUSTER_DIR = RUN_STAGEB_DATASET_DIR / "single_pool_CA_all_single_recluster_for_cci"
ADATA_CELLTYPE_N_UMAP = RECLUSTER_DIR / "CA_all_single_matchedcells_scanpy.h5ad"

# ---------- Stage5 output ----------
CURRENT_STAGE5_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "CA_all_single_cci_newLeiden"
    / "communication_pseudotime_clusterGNN_LTC"
    / "five_stage_network_viz_out"
)

# 兼容旧临时目录；自动找含有 TARGET 文件的目录。
STAGE5_DIR_CANDIDATES = [
    CURRENT_STAGE5_DIR,
    (get_external_root("SIG_LLM", "external/Sig-LLM-Frag") / "BreastCancer/five_stage_network_viz_out_hnscc_caall/1"),
    (get_external_root("SIG_LLM", "external/Sig-LLM-Frag") / "BreastCancer/five_stage_network_viz_out_hnscc_lnall/1"),
    (get_external_root("SIG_LLM", "external/Sig-LLM-Frag") / "BreastCancer/five_stage_network_viz_out"),
]

# ---------- gene universe ----------
# True: 为了真正得到每个 cluster top1000，使用 CA_all 全基因作为候选；
# Stage5 节点有真实 perturb impact，非 Stage5 gene 用低 background impact floor。
USE_ALL_CA_GENES_FOR_HEATMAP = True

# 如果只想完全限制在 Stage5 节点，设为 True；但通常 Stage5 节点数量不足以组成 top1000。
RESTRICT_TO_STAGE5_NODES_ONLY = False

# 对非 Stage5 节点的 background impact floor。
# 这个值不能太高，否则热图退化成普通 marker 热图；也不能为 0，否则无法填充 top1000。
NON_STAGE5_IMPACT_FLOOR = 0.18
STAGE5_DIRECT_IMPACT_BOOST = 1.45

# ---------- scoring ----------
TOP_N_GENES_PER_CLUSTER = 1000
TOP_N_PREVIEW_PER_CLUSTER = 200

ALPHA_IMPACT = 1.0
BETA_CLUSTER_SPECIFICITY = 2.4
GAMMA_OWNER_BONUS = 1.8
GAMMA_ROWZ_BONUS = 1.2

MIN_PCT_EXPR = 0.08
MIN_SPEC_Z = 0.00
MEAN_EXPR_BONUS_WEIGHT = 0.25

# primary 阶段优先只选 owner cluster == 当前 cluster 的 genes，保证每个 block 自己发红。
REQUIRE_OWNER_CLUSTER_PRIMARY = True
UNIQUE_GENE_ROWS = True
ALLOW_DUPLICATE_IF_NOT_ENOUGH = True

# heatmap display
RAW_SCORE_SCALE_FOR_ROWZ = 80.0
HEATMAP_Z_CLIP = 2.3

# ---------- chunking ----------
GENE_CHUNK = 512

# ---------- plotting ----------
DPI = 320
FIG_W = 7.8
FIG_H_MIN = 8.0
FIG_H_MAX = 30.0
PREVIEW_FIG_H_MAX = 20.0

# 横轴 cluster 标签：C0: cell type
CELLTYPE_COL_CANDIDATES = [
    "cell.type",
    "cell_type",
    "CellType",
    "original_cell_type",
    "integrated_label",
    "annotation",
    "cell_annotation",
]
MAX_CELLTYPE_LABEL_LEN = 26
X_LABEL_ROTATION = 45


# 关键修正：这版不再用 Leiden cluster 的多数 cell.type 做标签；
# 而是直接把原始 cell.type 作为分析分组，再映射成 C0-Cn。
# 这样 C0-Cn 与原始 cell.type UMAP 图例一一对应，不会出现多个 cluster 重复同一个细胞类型。
GROUP_BY_ORIGINAL_CELLTYPE = True
ORIGINAL_LEIDEN_COL_BACKUP = "leiden_cluster_backup"
ANALYSIS_CELLTYPE_COL = "analysis_celltype"
MIN_CELLS_PER_CELLTYPE_GROUP = 1

# 尽量按你之前 UMAP 图例的顺序排列；没有列出的类型会自动追加。
PREFERRED_CELLTYPE_ORDER = [
    "B.cells",
    "B cells",
    "B_Plasma.cells",
    "B/Plasma cells",
    "Dendritic.cells",
    "Dendritic cells",
    "Endothelial.cells",
    "Endothelial cells",
    "Fibroblasts",
    "Macrophages",
    "Malignant.cells",
    "Malignant cells",
    "Mast.cells",
    "Mast cells",
    "Myocytes",
    "T.cells",
    "T cells",
]

# 单独输出 row block 对标色卡；主热图不再显示右侧 block 标签。
BLOCK_ANNOTATION_WIDTH = 2.6
BLOCK_COLOR_ALPHA = 0.95

# Nature-like diverging colors
NATURE_BLUE = "#3B6FB6"
NATURE_WHITE = "#F7F7F7"
NATURE_RED = "#B2182B"

SEED = 42
np.random.seed(SEED)

# ============================================================
# 1) HELPERS
# ============================================================
def log(*args):
    print(*args, flush=True)


def safe_name(x: str) -> str:
    x = str(x)
    for bad in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
        x = x.replace(bad, "_")
    return x


def is_target_gene(g: str) -> bool:
    return str(g).upper() == TARGET_GENE_UPPER


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


def minmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x, dtype=np.float64)
    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])
    if hi - lo < 1e-12:
        out = np.zeros_like(x, dtype=np.float64)
        out[finite] = 0.5
        return out
    out = np.zeros_like(x, dtype=np.float64)
    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def zscore_by_gene(X_cluster_by_gene: np.ndarray) -> np.ndarray:
    """对每个 gene 在 cluster 维度做 zscore。"""
    mu = np.nanmean(X_cluster_by_gene, axis=0, keepdims=True)
    sd = np.nanstd(X_cluster_by_gene, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((X_cluster_by_gene - mu) / sd).astype(np.float32)


def row_zscore(mat_gene_by_cluster: np.ndarray) -> np.ndarray:
    """对每个 gene row 在 cluster 维度做 zscore。"""
    m = np.nanmean(mat_gene_by_cluster, axis=1, keepdims=True)
    s = np.nanstd(mat_gene_by_cluster, axis=1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return ((mat_gene_by_cluster - m) / s).astype(np.float32)


def safe_pow(x: np.ndarray, p: float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    return np.power(x, p)


def save_fig_both(fig: plt.Figure, png_path: Path, svg_path: Path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(svg_path), bbox_inches="tight")
    plt.close(fig)


def resolve_stage5_dir() -> Path:
    baseline_name = f"{TARGET_GENE}_stage5_baseline_node_state.csv"
    pert_name = f"{TARGET_GENE}_stage5_{MODE}_node_state.csv"

    for d in STAGE5_DIR_CANDIDATES:
        if (d / baseline_name).exists() and (d / pert_name).exists():
            return d

    msg = [
        f"Cannot find Stage5 files for TARGET_GENE={TARGET_GENE}, MODE={MODE}.",
        "Tried:",
    ]
    for d in STAGE5_DIR_CANDIDATES:
        msg.append(f"  - {d}")
        if d.exists():
            cands = sorted(d.glob("*_stage5_baseline_node_state.csv"))[:40]
            if cands:
                prefixes = [p.name.replace("_stage5_baseline_node_state.csv", "") for p in cands]
                msg.append("    available target prefixes: " + ", ".join(prefixes))
    raise FileNotFoundError("\n".join(msg))


def resolve_adata_path() -> Path:
    if ADATA_CELLTYPE_N_UMAP.exists():
        return ADATA_CELLTYPE_N_UMAP
    raise FileNotFoundError(
        "Cannot find celltypeN-guided CA_all recluster UMAP h5ad. Tried:\n"
        f"  {ADATA_CELLTYPE_N_UMAP}\n"
        "Please rerun hnscc_caall_cci_newLeiden_celltypeN_metaFixed.py first."
    )

# ============================================================
# 2) LOAD STAGE5 PERTURBATION IMPACT
# ============================================================
def load_stage5_impact(
    baseline_csv: Path,
    pert_csv: Path,
    mode: str,
    impact_mode: str = "auto",
) -> pd.DataFrame:
    b = pd.read_csv(baseline_csv)
    k = pd.read_csv(pert_csv)

    need = {"node", "state_final"}
    assert need.issubset(set(b.columns)), f"baseline 缺少列: {need - set(b.columns)}"
    assert need.issubset(set(k.columns)), f"perturbed 缺少列: {need - set(k.columns)}"

    df = (
        b[["node", "state_final"]]
        .rename(columns={"state_final": "state_baseline"})
        .merge(
            k[["node", "state_final"]]
            .rename(columns={"state_final": "state_perturbed"}),
            on="node",
            how="inner",
        )
    )
    df["node"] = df["node"].astype(str)

    before_n = len(df)
    df = df[~df["node"].map(is_target_gene)].copy()
    after_n = len(df)
    log(f"[FILTER] exclude target gene {TARGET_GENE}: {before_n} -> {after_n}")

    df["state_baseline"] = pd.to_numeric(df["state_baseline"], errors="coerce")
    df["state_perturbed"] = pd.to_numeric(df["state_perturbed"], errors="coerce")
    df = df.dropna().copy()

    df["delta_perturbed_minus_base"] = df["state_perturbed"] - df["state_baseline"]

    imode = impact_mode.lower().strip()
    if imode == "auto":
        imode = "down" if str(mode).lower() == "ko" else "up"

    if imode == "down":
        df["perturb_impact"] = np.maximum(0.0, df["state_baseline"] - df["state_perturbed"])
    elif imode == "up":
        df["perturb_impact"] = np.maximum(0.0, df["state_perturbed"] - df["state_baseline"])
    elif imode == "abs":
        df["perturb_impact"] = np.abs(df["delta_perturbed_minus_base"])
    else:
        raise ValueError(f"Unsupported IMPACT_MODE: {impact_mode}")

    return df.sort_values("perturb_impact", ascending=False).reset_index(drop=True)

# ============================================================
# 3) LOAD UMAP / MATCH ROWS
# ============================================================
def get_cell_row_indices(adata_umap: ad.AnnData, meta_npz_path: Path) -> np.ndarray:
    meta_npz = np.load(meta_npz_path, allow_pickle=True)
    assert "cell_ids" in meta_npz.files, "cell_ids not found in CA_all.rna_meta.npz"

    cell_ids_all = meta_npz["cell_ids"].astype(str)
    exact_map = {cid: i for i, cid in enumerate(cell_ids_all)}

    obs_ids = adata_umap.obs_names.astype(str).tolist()
    if all(cid in exact_map for cid in obs_ids):
        return np.array([exact_map[cid] for cid in obs_ids], dtype=int)

    if "cell_id" in adata_umap.obs.columns:
        obs_cell_ids = adata_umap.obs["cell_id"].astype(str).tolist()
        if all(cid in exact_map for cid in obs_cell_ids):
            return np.array([exact_map[cid] for cid in obs_cell_ids], dtype=int)

    # barcode_counts fallback：仅在 barcode 不重复时使用。
    norm_all = [norm_barcode(x) for x in cell_ids_all]
    norm_counts = pd.Series(norm_all).value_counts()
    non_dup = set(norm_counts[norm_counts == 1].index.astype(str).tolist())
    norm_map = {norm_barcode(cid): i for i, cid in enumerate(cell_ids_all) if norm_barcode(cid) in non_dup}

    for col in ["barcode_counts", "barcode", "barcode_meta"]:
        if col in adata_umap.obs.columns:
            keys = adata_umap.obs[col].astype(str).map(norm_barcode).tolist()
            if all(k in norm_map for k in keys):
                log(f"[META] matched rows by obs column: {col}")
                return np.array([norm_map[k] for k in keys], dtype=int)

    missing = [cid for cid in obs_ids if cid not in exact_map][:8]
    raise ValueError(
        "UMAP cells cannot be matched to CA_all.rna_meta.npz. "
        f"Example missing obs_names: {missing}"
    )


def detect_cluster_col(adata_umap: ad.AnnData) -> str:
    candidates = ["new_cluster", "leiden", "leiden_new", "original_cluster"]
    for c in candidates:
        if c in adata_umap.obs.columns:
            return c
    raise KeyError("Cannot find cluster column in UMAP h5ad.")


def detect_celltype_col(adata_umap: ad.AnnData) -> str | None:
    """Find original internal cell annotation column if present."""
    for c in CELLTYPE_COL_CANDIDATES:
        if c in adata_umap.obs.columns:
            return c
    return None


def pretty_celltype_label(x: str) -> str:
    """Make original cell.type values readable on figures."""
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "none"}:
        return "Unknown"
    x = x.replace("_", " ").replace(".", " ")
    x = " ".join(x.split())
    return x


def shorten_label(x: str, max_len: int = MAX_CELLTYPE_LABEL_LEN) -> str:
    x = pretty_celltype_label(x)
    if len(x) <= max_len:
        return x
    return x[: max_len - 1] + "…"


def choose_celltype_order(vals: pd.Series) -> List[str]:
    """Order original cell.type labels to match the previous cell-type UMAP legend as much as possible."""
    vals = vals.astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    counts = vals.value_counts(dropna=False)
    existing = counts.index.astype(str).tolist()

    # If categorical order exists upstream, keep that first when available.
    ordered = []

    pref_norm = {pretty_celltype_label(x).lower(): x for x in PREFERRED_CELLTYPE_ORDER}
    existing_by_norm = {pretty_celltype_label(x).lower(): x for x in existing}

    for p in PREFERRED_CELLTYPE_ORDER:
        key = pretty_celltype_label(p).lower()
        if key in existing_by_norm and existing_by_norm[key] not in ordered:
            ordered.append(existing_by_norm[key])

    # Append unlisted cell types by abundance, then name.
    rest = [x for x in existing if x not in ordered]
    rest = sorted(rest, key=lambda z: (-int(counts.get(z, 0)), pretty_celltype_label(z)))
    ordered.extend(rest)

    ordered = [x for x in ordered if int(counts.get(x, 0)) >= int(MIN_CELLS_PER_CELLTYPE_GROUP)]
    return ordered


def apply_original_celltype_as_cluster(adata_umap: ad.AnnData) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Use original internal cell annotation as the analysis cluster.

    This is the key correction for the current request:
    - preserve the existing UMAP coordinates unchanged;
    - do not use Leiden clusters and then label them by majority cell.type;
    - instead, directly group cells by original cell.type and map those cell types to C0-Cn.
    Therefore C0-Cn are one-to-one with cell types and cannot have duplicated names.
    """
    celltype_col = detect_celltype_col(adata_umap)
    if celltype_col is None:
        raise KeyError(
            "GROUP_BY_ORIGINAL_CELLTYPE=True, but no cell.type-like column was found in h5ad obs. "
            f"Tried: {CELLTYPE_COL_CANDIDATES}. Available obs columns: {list(adata_umap.obs.columns)}"
        )

    adata_umap = adata_umap.copy()
    if "new_cluster" in adata_umap.obs.columns:
        adata_umap.obs[ORIGINAL_LEIDEN_COL_BACKUP] = adata_umap.obs["new_cluster"].astype(str)

    celltype_raw = (
        adata_umap.obs[celltype_col]
        .astype(str)
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    )

    ordered_celltypes = choose_celltype_order(celltype_raw)
    ct2cluster = {ct: str(i) for i, ct in enumerate(ordered_celltypes)}

    keep_mask = celltype_raw.isin(ct2cluster.keys()).values
    if int(keep_mask.sum()) == 0:
        raise RuntimeError("No cells left after original cell.type grouping.")
    if int(keep_mask.sum()) < adata_umap.n_obs:
        log(f"[CELLTYPE_AS_CLUSTER] filtered cells by MIN_CELLS_PER_CELLTYPE_GROUP: {adata_umap.n_obs} -> {int(keep_mask.sum())}")
        adata_umap = adata_umap[keep_mask].copy()
        celltype_raw = celltype_raw.iloc[np.where(keep_mask)[0]].reset_index(drop=True)
    else:
        celltype_raw = pd.Series(celltype_raw.values, index=adata_umap.obs_names)

    # Recompute with filtered obs index to avoid alignment issues.
    celltype_raw = (
        adata_umap.obs[celltype_col]
        .astype(str)
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    )
    adata_umap.obs[ANALYSIS_CELLTYPE_COL] = celltype_raw.values
    adata_umap.obs["new_cluster"] = celltype_raw.map(ct2cluster).astype(str).values

    rows = []
    for ct in ordered_celltypes:
        cl = ct2cluster[ct]
        n_cells = int((celltype_raw.values == ct).sum())
        rows.append({
            "cluster_label": str(cl),
            "cluster_display": f"C{cl}",
            "celltype_col_used": celltype_col,
            "major_celltype": str(ct),
            "major_celltype_short": shorten_label(ct),
            "major_celltype_n": n_cells,
            "major_celltype_frac": 1.0,
            "n_cells": n_cells,
            "xtick_label": f"C{cl}: {shorten_label(ct)}",
        })

    label_df = pd.DataFrame(rows)
    log("[CELLTYPE_AS_CLUSTER] celltype_col =", celltype_col)
    log("[CELLTYPE_AS_CLUSTER] original cell.type -> analysis cluster:")
    log(label_df[["cluster_display", "major_celltype", "n_cells"]].to_string(index=False))
    return adata_umap, label_df


def infer_cluster_celltype_labels(adata_umap: ad.AnnData, clusters: List[str]) -> pd.DataFrame:
    """For fallback only: label Leiden clusters by majority cell.type."""
    celltype_col = detect_celltype_col(adata_umap)
    cl_arr = adata_umap.obs["new_cluster"].astype(str)

    rows = []
    for cl in clusters:
        m = cl_arr == str(cl)
        n_cells = int(m.sum())

        if celltype_col is None:
            major = "Unknown"
            major_n = 0
            major_frac = np.nan
        else:
            vals = (
                adata_umap.obs.loc[m, celltype_col]
                .astype(str)
                .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
            )
            vc = vals.value_counts(dropna=False)
            if len(vc) == 0:
                major = "Unknown"
                major_n = 0
                major_frac = np.nan
            else:
                major = str(vc.index[0])
                major_n = int(vc.iloc[0])
                major_frac = float(major_n / max(n_cells, 1))

        c_label = f"C{cl}"
        xtick_label = f"{c_label}: {shorten_label(major)}"
        rows.append({
            "cluster_label": str(cl),
            "cluster_display": c_label,
            "celltype_col_used": celltype_col if celltype_col is not None else "",
            "major_celltype": major,
            "major_celltype_short": shorten_label(major),
            "major_celltype_n": major_n,
            "major_celltype_frac": major_frac,
            "n_cells": n_cells,
            "xtick_label": xtick_label,
        })

    out = pd.DataFrame(rows)
    log("[CELLTYPE_LABEL][FALLBACK_MAJOR] celltype_col =", celltype_col)
    if len(out) > 0:
        log("[CELLTYPE_LABEL][FALLBACK_MAJOR] Leiden cluster -> major cell type:")
        log(out[["cluster_display", "major_celltype", "major_celltype_n", "major_celltype_frac", "n_cells"]].to_string(index=False))
    return out


def build_block_color_map(clusters: List[str]) -> Dict[str, tuple]:
    """Stable block colors for separate row-block annotation map."""
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for cmap in cmaps:
        for i in range(cmap.N):
            colors.append(cmap(i))
    if len(clusters) > len(colors):
        extra = plt.get_cmap("hsv")
        for i in range(len(clusters) - len(colors)):
            colors.append(extra(i / max(1, len(clusters) - len(colors))))
    return {str(cl): colors[i] for i, cl in enumerate(clusters)}


def load_umap_cluster_and_gene_index(
    adata_path: Path,
    meta_npz_path: Path,
    genes_tsv: Path,
) -> Tuple[ad.AnnData, np.ndarray, List[str], pd.DataFrame]:
    adata_umap = ad.read_h5ad(str(adata_path))
    assert "X_umap" in adata_umap.obsm, "UMAP coordinates not found in h5ad"

    cluster_col = detect_cluster_col(adata_umap)
    adata_umap.obs["new_cluster"] = adata_umap.obs[cluster_col].astype(str)

    log("[UMAP] h5ad =", adata_path)
    log("[UMAP] n_obs before grouping =", adata_umap.n_obs)
    log("[UMAP] original Leiden cluster_col =", cluster_col)
    log("[UMAP] original Leiden n_cluster =", adata_umap.obs["new_cluster"].nunique())

    if GROUP_BY_ORIGINAL_CELLTYPE:
        adata_umap, cluster_label_df = apply_original_celltype_as_cluster(adata_umap)
    else:
        clusters_tmp = sort_cluster_labels(adata_umap.obs["new_cluster"].astype(str).unique().tolist())
        cluster_label_df = infer_cluster_celltype_labels(adata_umap, clusters_tmp)

    # Important: row_idx must be computed AFTER optional filtering by original cell.type groups.
    row_idx = get_cell_row_indices(adata_umap, meta_npz_path)
    genes = pd.read_csv(genes_tsv, sep="\t")["gene"].astype(str).tolist()

    log("[UMAP] n_obs used =", adata_umap.n_obs)
    log("[UMAP] analysis group =", "original cell.type" if GROUP_BY_ORIGINAL_CELLTYPE else "Leiden cluster")
    log("[UMAP] analysis n_cluster =", adata_umap.obs["new_cluster"].nunique())
    log("[UMAP] analysis cluster counts:")
    log(adata_umap.obs["new_cluster"].astype(str).value_counts().sort_index())

    return adata_umap, row_idx, genes, cluster_label_df

# ============================================================
# 4) CHUNKED CLUSTER EXPRESSION FEATURES
# ============================================================
def compute_cluster_expression_features_chunked(
    log1p_npy: Path,
    row_idx: np.ndarray,
    genes: List[str],
    adata_umap: ad.AnnData,
    gene_keep_mask: np.ndarray,
    gene_chunk: int = GENE_CHUNK,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    keep_genes, mean_expr(cluster x gene), pct_expr(cluster x gene), spec_z(cluster x gene), mean_expr_scaled(cluster x gene)
    """
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
    log("[RNA] gene universe kept =", n_gene)
    log("[RNA] computing cluster mean/pct in chunks ...")

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

    spec_z = zscore_by_gene(mean_expr).astype(np.float32)

    mean_expr_scaled = np.zeros_like(mean_expr, dtype=np.float32)
    for j in range(n_gene):
        mean_expr_scaled[:, j] = minmax_1d(mean_expr[:, j]).astype(np.float32)

    return keep_genes, mean_expr, pct_expr, spec_z, mean_expr_scaled

# ============================================================
# 5) RESPONSE WEIGHT ALGORITHM
# ============================================================
def build_response_weight_matrices(
    genes: List[str],
    keep_genes: List[str],
    mean_expr: np.ndarray,
    pct_expr: np.ndarray,
    spec_z: np.ndarray,
    mean_expr_scaled: np.ndarray,
    impact_df: pd.DataFrame,
) -> Dict[str, np.ndarray | pd.DataFrame | List[str]]:
    """构建 raw response / selection score / heatmap display weight。"""
    n_cluster, n_gene = mean_expr.shape

    impact_df = impact_df.copy()
    impact_df["node"] = impact_df["node"].astype(str)
    impact_map = impact_df.set_index("node")["perturb_impact"].to_dict()
    state_base_map = impact_df.set_index("node")["state_baseline"].to_dict()
    state_pert_map = impact_df.set_index("node")["state_perturbed"].to_dict()
    delta_map = impact_df.set_index("node")["delta_perturbed_minus_base"].to_dict()

    direct_impact = np.asarray([float(impact_map.get(g, 0.0)) for g in keep_genes], dtype=np.float32)
    direct_flag = np.asarray([1 if g in impact_map else 0 for g in keep_genes], dtype=np.int8)

    direct_impact_norm = minmax_1d(direct_impact).astype(np.float32)

    # 关键：非 Stage5 gene 保留低 background impact；Stage5 direct gene 获得额外 boost。
    if RESTRICT_TO_STAGE5_NODES_ONLY:
        impact_weight = direct_impact_norm.copy()
    else:
        impact_weight = np.full((n_gene,), float(NON_STAGE5_IMPACT_FLOOR), dtype=np.float32)
        impact_weight = np.maximum(impact_weight, direct_impact_norm)
        impact_weight = impact_weight * (1.0 + (STAGE5_DIRECT_IMPACT_BOOST - 1.0) * direct_flag.astype(np.float32))
        impact_weight = np.clip(impact_weight, 0.0, 1.0)

    spec_pos = np.maximum(spec_z, 0.0).astype(np.float32)
    cluster_specificity = (spec_pos * pct_expr).astype(np.float32)
    cluster_specificity = cluster_specificity * (1.0 + float(MEAN_EXPR_BONUS_WEIGHT) * mean_expr_scaled)

    raw_response = (
        safe_pow(impact_weight[None, :], ALPHA_IMPACT)
        * safe_pow(cluster_specificity, BETA_CLUSTER_SPECIFICITY)
    ).astype(np.float64)

    invalid = (pct_expr < float(MIN_PCT_EXPR)) | (spec_z <= float(MIN_SPEC_Z))
    raw_response[invalid] = 0.0

    # gene owner：这个 gene 最偏向哪个 cluster。
    owner_idx = np.argmax(raw_response, axis=0).astype(np.int64)
    best = np.max(raw_response, axis=0).astype(np.float64)

    # 对每个 cluster 计算 best_other 和 margin/ratio。
    best_other = np.zeros_like(raw_response, dtype=np.float64)
    for ci in range(n_cluster):
        if n_cluster == 1:
            best_other[ci, :] = 0.0
        else:
            other = [x for x in range(n_cluster) if x != ci]
            best_other[ci, :] = np.max(raw_response[other, :], axis=0)

    ratio = raw_response / (raw_response + best_other + 1e-12)
    ratio = np.clip(ratio, 0.0, 1.0)

    # display matrix：每个 gene 横向比较 cluster。
    log_raw = np.log1p(float(RAW_SCORE_SCALE_FOR_ROWZ) * raw_response.T).astype(np.float32)
    heatmap_weight = row_zscore(log_raw)
    heatmap_weight = np.clip(heatmap_weight, -float(HEATMAP_Z_CLIP), float(HEATMAP_Z_CLIP)).astype(np.float32)

    positive_rowz = np.maximum(heatmap_weight.T, 0.0)

    # selection score：让每个 cluster 优先选 owner 为自己的 gene，形成块状红色。
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
    }


def select_top_genes_per_cluster(
    clusters: List[str],
    keep_genes: List[str],
    mean_expr: np.ndarray,
    pct_expr: np.ndarray,
    spec_z: np.ndarray,
    matrices: Dict[str, np.ndarray | pd.DataFrame | List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[int]]:
    selection_score = matrices["selection_score"]
    raw_response = matrices["raw_response"]
    heatmap_weight = matrices["heatmap_weight_gene_by_cluster"]  # gene x cluster
    ratio = matrices["ratio"]
    owner_idx = matrices["owner_idx"]
    impact_weight = matrices["impact_weight"]
    direct_flag = matrices["direct_flag"]

    n_cluster, n_gene = selection_score.shape
    used_genes = set()

    top_rows = []
    row_ann = []
    heatmap_row_indices = []

    for ci, cl in enumerate(clusters):
        fs = np.asarray(selection_score[ci, :], dtype=np.float64)
        order = np.argsort(fs)[::-1]

        picked = 0
        pass_id = 1

        def try_pick(require_owner: bool, allow_used: bool, pass_name: str, picked: int):
            local_rows = []
            local_indices = []
            for gj in order:
                if picked + len(local_indices) >= TOP_N_GENES_PER_CLUSTER:
                    break
                g = str(keep_genes[int(gj)])
                if is_target_gene(g):
                    continue
                if (not allow_used) and g in used_genes:
                    continue
                if require_owner and int(owner_idx[int(gj)]) != int(ci):
                    continue
                if float(raw_response[ci, gj]) <= 0:
                    continue
                if float(fs[gj]) <= 0:
                    continue

                local_indices.append(int(gj))
                local_rows.append({
                    "cluster_label": str(cl),
                    "rank": int(picked + len(local_indices)),
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
            return local_rows, local_indices

        # Pass 1: owner == current cluster，且不重复。
        if REQUIRE_OWNER_CLUSTER_PRIMARY:
            rows1, idx1 = try_pick(require_owner=True, allow_used=False, pass_name="primary_owner_unique", picked=picked)
        else:
            rows1, idx1 = try_pick(require_owner=False, allow_used=False, pass_name="primary_unique", picked=picked)
        top_rows.extend(rows1)
        heatmap_row_indices.extend(idx1)
        for gj in idx1:
            used_genes.add(str(keep_genes[gj]))
        picked += len(idx1)

        # Pass 2: 如果 owner genes 不够，放宽 owner，但仍不重复。
        if picked < TOP_N_GENES_PER_CLUSTER:
            rows2, idx2 = try_pick(require_owner=False, allow_used=False, pass_name="fallback_nonowner_unique", picked=picked)
            top_rows.extend(rows2)
            heatmap_row_indices.extend(idx2)
            for gj in idx2:
                used_genes.add(str(keep_genes[gj]))
            picked += len(idx2)

        # Pass 3: 如果仍不够，允许重复 gene 出现在不同 cluster block。
        if picked < TOP_N_GENES_PER_CLUSTER and ALLOW_DUPLICATE_IF_NOT_ENOUGH:
            rows3, idx3 = try_pick(require_owner=False, allow_used=True, pass_name="fallback_allow_duplicate", picked=picked)
            top_rows.extend(rows3)
            heatmap_row_indices.extend(idx3)
            picked += len(idx3)

        # 为 row annotation 建 block 信息。注意 top_rows 是全局 append，所以重新从 top_rows 中截取当前 cluster。
        cur_rows = [r for r in top_rows if str(r["cluster_label"]) == str(cl)]
        for rr in cur_rows:
            row_ann.append({
                "row_index": len(row_ann),
                "row_block_cluster": str(cl),
                "row_block_label": f"C{cl}_top{TOP_N_GENES_PER_CLUSTER}",
                "gene": rr["gene"],
                "rank_in_cluster": rr["rank"],
                "selection_pass": rr["selection_pass"],
                "selection_score": rr["selection_score"],
                "raw_response": rr["raw_response"],
                "heatmap_weight_in_owner_cluster": rr["heatmap_weight_in_owner_cluster"],
                "direct_stage5_flag": rr["direct_stage5_flag"],
                "owner_cluster_by_raw_response": rr["owner_cluster_by_raw_response"],
            })

        log(f"[SELECT] cluster C{cl}: picked {picked}/{TOP_N_GENES_PER_CLUSTER}")

    top_df = pd.DataFrame(top_rows)
    row_ann_df = pd.DataFrame(row_ann)

    # 重新构建 heatmap row indices：按 row_ann 的 gene 顺序。
    gene_to_indices = {}
    for i, g in enumerate(keep_genes):
        gene_to_indices.setdefault(str(g), []).append(i)

    row_indices_final = []
    for g in row_ann_df["gene"].astype(str).tolist():
        # keep_genes 中 gene 唯一，一般取第一个。
        row_indices_final.append(gene_to_indices[str(g)][0])

    return top_df, row_ann_df, heatmap_weight, row_indices_final

# ============================================================
# 6) PLOT HEATMAP
# ============================================================
def make_nature_cmap():
    return LinearSegmentedColormap.from_list(
        "NatureBlueWhiteRed",
        [NATURE_BLUE, NATURE_WHITE, NATURE_RED],
        N=256,
    )


def plot_response_heatmap(
    mat: np.ndarray,
    row_ann_df: pd.DataFrame,
    clusters: List[str],
    cluster_label_df: pd.DataFrame,
    out_png: Path,
    out_svg: Path,
    title: str,
    topn_label: str,
    fig_h_max: float,
):
    """
    Main heatmap only:
    - x axis = C0: concrete cell name
    - y axis = hidden completely
    - no right-side row-block labels here, because they are generated as a separate figure.
    """
    if mat.shape[0] == 0:
        log("[WARN] heatmap matrix has zero rows, skip plot:", out_png)
        return

    n_rows, n_cols = mat.shape
    fig_h = min(float(fig_h_max), max(float(FIG_H_MIN), n_rows / 360.0))

    fig, ax = plt.subplots(figsize=(FIG_W, fig_h), dpi=DPI)
    cmap = make_nature_cmap()

    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-float(HEATMAP_Z_CLIP),
        vmax=float(HEATMAP_Z_CLIP),
    )

    label_map = dict(zip(cluster_label_df["cluster_label"].astype(str), cluster_label_df["xtick_label"].astype(str)))
    xtick_labels = [label_map.get(str(x), f"C{x}") for x in clusters]

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        xtick_labels,
        rotation=X_LABEL_ROTATION,
        ha="right",
        fontsize=8,
    )

    # 纵轴完全不显示 gene 名，也不显示 cluster-gene 数量数字。
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0)

    # 仅保留 block 分割线，不在主热图右侧标数字/名称，避免对标块错位。
    if len(row_ann_df) > 0:
        y0 = 0
        for _, sub in row_ann_df.groupby("row_block_cluster", sort=False):
            n = len(sub)
            if y0 > 0:
                ax.axhline(y0 - 0.5, color="black", linewidth=0.35, alpha=0.45)
            y0 += n

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("original cell-type group", fontsize=10)

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label(
        "cluster-specific perturbation response weight\nrow-z of log response; red=higher, blue=lower",
        fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)

    fig.text(
        0.02,
        0.01,
        f"Rows: {topn_label}; gene names hidden; row-block labels are saved as a separate annotation figure.",
        fontsize=7,
        ha="left",
        va="bottom",
        color="#444444",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    save_fig_both(fig, out_png, out_svg)


def plot_row_block_annotation(
    row_ann_df: pd.DataFrame,
    clusters: List[str],
    cluster_label_df: pd.DataFrame,
    out_png: Path,
    out_svg: Path,
    title: str,
    fig_h_max: float,
):
    """
    Separate row-block annotation / color-card figure.
    This replaces the old right-side block labels that could misalign with the heatmap.
    """
    if len(row_ann_df) == 0:
        log("[WARN] row annotation is empty, skip block annotation:", out_png)
        return

    n_rows = len(row_ann_df)
    fig_h = min(float(fig_h_max), max(float(FIG_H_MIN), n_rows / 360.0))

    block_color_map = build_block_color_map(clusters)
    label_info = cluster_label_df.set_index("cluster_label").to_dict(orient="index")

    # Build one-column color image, one pixel per heatmap row.
    rgba = np.zeros((n_rows, 1, 4), dtype=float)
    for i, cl in enumerate(row_ann_df["row_block_cluster"].astype(str).tolist()):
        rgba[i, 0, :] = block_color_map.get(str(cl), (0.75, 0.75, 0.75, 1.0))
        rgba[i, 0, 3] = BLOCK_COLOR_ALPHA

    fig, ax = plt.subplots(figsize=(BLOCK_ANNOTATION_WIDTH, fig_h), dpi=DPI)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=10)

    # Add block separators and labels centered in each block.
    y0 = 0
    for cl, sub in row_ann_df.groupby("row_block_cluster", sort=False):
        n = len(sub)
        center = y0 + 0.5 * max(n - 1, 0)
        info = label_info.get(str(cl), {})
        cell_name = shorten_label(info.get("major_celltype", "Unknown"), max_len=34)
        n_cells = info.get("n_cells", "")
        label = f"C{cl}\n{cell_name}"
        ax.text(
            0.52,
            center,
            label,
            ha="left",
            va="center",
            fontsize=7.0,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
        )
        if y0 > 0:
            ax.axhline(y0 - 0.5, color="black", linewidth=0.35, alpha=0.45)
        y0 += n

    ax.set_xlim(-0.5, 2.8)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.axis("off")

    plt.tight_layout()
    save_fig_both(fig, out_png, out_svg)

# ============================================================
# 7) MAIN
# ============================================================
# 7) MAIN
# ============================================================
def main():
    stage5_dir = resolve_stage5_dir()
    adata_path = resolve_adata_path()

    out_dir = stage5_dir / (
        f"cluster_response_top{TOP_N_GENES_PER_CLUSTER}_heatmap_"
        f"originalCelltype_as_C0Cn_from_stage5_{MODE}_exclude_self"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stage5_baseline_node = stage5_dir / f"{TARGET_GENE}_stage5_baseline_node_state.csv"
    stage5_pert_node = stage5_dir / f"{TARGET_GENE}_stage5_{MODE}_node_state.csv"

    log("=" * 100)
    log("[INFO] TARGET_GENE =", TARGET_GENE)
    log("[INFO] MODE =", MODE)
    log("[INFO] IMPACT_MODE =", IMPACT_MODE)
    log("[INFO] STAGE5_DIR =", stage5_dir)
    log("[INFO] ADATA_SOURCE_UMAP =", adata_path)
    log("[INFO] GROUP_BY_ORIGINAL_CELLTYPE =", GROUP_BY_ORIGINAL_CELLTYPE)
    log("[INFO] LOG1P_NPY =", LOG1P_NPY)
    log("[INFO] META_NPZ =", META_NPZ)
    log("[INFO] GENES_TSV =", GENES_TSV)
    log("[INFO] OUT_DIR =", out_dir)
    log("[INFO] USE_ALL_CA_GENES_FOR_HEATMAP =", USE_ALL_CA_GENES_FOR_HEATMAP)
    log("[INFO] RESTRICT_TO_STAGE5_NODES_ONLY =", RESTRICT_TO_STAGE5_NODES_ONLY)
    log("=" * 100)

    for p in [stage5_baseline_node, stage5_pert_node, adata_path, LOG1P_NPY, META_NPZ, GENES_TSV]:
        assert p.exists(), f"Missing: {p}"

    impact_df = load_stage5_impact(
        baseline_csv=stage5_baseline_node,
        pert_csv=stage5_pert_node,
        mode=MODE,
        impact_mode=IMPACT_MODE,
    )
    log("[LOAD] Stage5 nodes after excluding target =", len(impact_df))
    log("[LOAD] nonzero perturb impact nodes =", int((impact_df["perturb_impact"] > 0).sum()))

    adata_umap, row_idx, genes, cluster_label_df = load_umap_cluster_and_gene_index(
        adata_path=adata_path,
        meta_npz_path=META_NPZ,
        genes_tsv=GENES_TSV,
    )

    clusters = sort_cluster_labels(adata_umap.obs["new_cluster"].astype(str).unique().tolist())

    # gene universe
    genes_arr = np.asarray(genes, dtype=object).astype(str)
    gene_keep_mask = np.ones((len(genes_arr),), dtype=bool)
    gene_keep_mask &= np.asarray([not is_target_gene(g) for g in genes_arr], dtype=bool)

    if RESTRICT_TO_STAGE5_NODES_ONLY:
        stage5_gene_set = set(impact_df["node"].astype(str).tolist())
        gene_keep_mask &= np.asarray([g in stage5_gene_set for g in genes_arr], dtype=bool)
    elif not USE_ALL_CA_GENES_FOR_HEATMAP:
        stage5_gene_set = set(impact_df["node"].astype(str).tolist())
        gene_keep_mask &= np.asarray([g in stage5_gene_set for g in genes_arr], dtype=bool)

    n_keep = int(gene_keep_mask.sum())
    if n_keep == 0:
        raise RuntimeError("No genes kept for heatmap after filtering target gene / stage5 restriction.")

    log("[GENE_UNIVERSE] total genes.tsv =", len(genes_arr))
    log("[GENE_UNIVERSE] kept genes =", n_keep)

    keep_genes, mean_expr, pct_expr, spec_z, mean_expr_scaled = compute_cluster_expression_features_chunked(
        log1p_npy=LOG1P_NPY,
        row_idx=row_idx,
        genes=genes,
        adata_umap=adata_umap,
        gene_keep_mask=gene_keep_mask,
        gene_chunk=GENE_CHUNK,
    )

    log("[SCORE] building cluster-specific perturbation response matrices ...")
    matrices = build_response_weight_matrices(
        genes=genes,
        keep_genes=keep_genes,
        mean_expr=mean_expr,
        pct_expr=pct_expr,
        spec_z=spec_z,
        mean_expr_scaled=mean_expr_scaled,
        impact_df=impact_df,
    )

    top_df, row_ann_df, heatmap_weight_all, row_indices_final = select_top_genes_per_cluster(
        clusters=clusters,
        keep_genes=keep_genes,
        mean_expr=mean_expr,
        pct_expr=pct_expr,
        spec_z=spec_z,
        matrices=matrices,
    )

    if len(row_indices_final) == 0:
        raise RuntimeError("No rows selected for heatmap. Consider relaxing MIN_PCT_EXPR / MIN_SPEC_Z.")

    heatmap_mat = heatmap_weight_all[np.asarray(row_indices_final, dtype=int), :]

    # Save outputs
    top_csv = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_GENES_PER_CLUSTER}_genes.csv"
    row_ann_csv = out_dir / "CA_all_originalCelltype_heatmap_row_annotation.csv"
    heatmap_csv = out_dir / "CA_all_originalCelltype_heatmap_weight_matrix.csv"
    qc_csv = out_dir / "CA_all_originalCelltype_cluster_response_algorithm_qc.csv"
    cluster_label_csv = out_dir / "CA_all_originalCelltype_cluster_celltype_labels.csv"

    top_df.to_csv(top_csv, index=False)
    row_ann_df.to_csv(row_ann_csv, index=False)

    heatmap_df = pd.DataFrame(heatmap_mat, columns=[f"C{x}" for x in clusters])
    heatmap_df.insert(0, "gene", row_ann_df["gene"].astype(str).values)
    heatmap_df.insert(0, "row_block_cluster", row_ann_df["row_block_cluster"].astype(str).values)
    heatmap_df.to_csv(heatmap_csv, index=False)

    qc_rows = []
    for cl, sub in row_ann_df.groupby("row_block_cluster", sort=False):
        qc_rows.append({
            "cluster_label": str(cl),
            "n_selected_rows": int(len(sub)),
            "n_direct_stage5_genes": int((sub["direct_stage5_flag"].astype(int) == 1).sum()),
            "mean_selection_score": float(np.nanmean(sub["selection_score"].values.astype(float))),
            "mean_owner_heatmap_weight": float(np.nanmean(sub["heatmap_weight_in_owner_cluster"].values.astype(float))),
            "n_primary_owner_unique": int((sub["selection_pass"] == "primary_owner_unique").sum()),
            "n_fallback_nonowner_unique": int((sub["selection_pass"] == "fallback_nonowner_unique").sum()),
            "n_fallback_allow_duplicate": int((sub["selection_pass"] == "fallback_allow_duplicate").sum()),
        })
    pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)
    cluster_label_df.to_csv(cluster_label_csv, index=False)

    log("[SAVE]", top_csv)
    log("[SAVE]", row_ann_csv)
    log("[SAVE]", heatmap_csv)
    log("[SAVE]", qc_csv)
    log("[SAVE]", cluster_label_csv)

    # Full top1000 heatmap
    full_png = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_GENES_PER_CLUSTER}_heatmap.png"
    full_svg = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_GENES_PER_CLUSTER}_heatmap.svg"
    log("[PLOT] full top response heatmap ...")
    plot_response_heatmap(
        mat=heatmap_mat,
        row_ann_df=row_ann_df,
        clusters=clusters,
        cluster_label_df=cluster_label_df,
        out_png=full_png,
        out_svg=full_svg,
        title=(
            f"CA_all cluster-specific response genes after {TARGET_GENE} {MODE.upper()}\n"
            f"rows = top{TOP_N_GENES_PER_CLUSTER} response genes per cluster; columns = original cell.type groups"
        ),
        topn_label=f"top{TOP_N_GENES_PER_CLUSTER} per cluster",
        fig_h_max=FIG_H_MAX,
    )
    log("[SAVE]", full_png)

    full_block_png = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_GENES_PER_CLUSTER}_rowblock_annotation.png"
    full_block_svg = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_GENES_PER_CLUSTER}_rowblock_annotation.svg"
    log("[PLOT] full row-block annotation as a separate figure ...")
    plot_row_block_annotation(
        row_ann_df=row_ann_df,
        clusters=clusters,
        cluster_label_df=cluster_label_df,
        out_png=full_block_png,
        out_svg=full_block_svg,
        title=f"Row blocks: top{TOP_N_GENES_PER_CLUSTER} genes per cluster",
        fig_h_max=FIG_H_MAX,
    )
    log("[SAVE]", full_block_png)

    # Preview top200 heatmap for easier inspection
    preview_blocks = []
    preview_indices = []
    start = 0
    for cl, sub in row_ann_df.groupby("row_block_cluster", sort=False):
        idx = sub.index.values[:TOP_N_PREVIEW_PER_CLUSTER]
        preview_indices.extend(idx.tolist())
    preview_row_ann = row_ann_df.iloc[preview_indices].copy().reset_index(drop=True)
    preview_mat = heatmap_mat[np.asarray(preview_indices, dtype=int), :]

    preview_png = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_PREVIEW_PER_CLUSTER}_preview_heatmap.png"
    preview_svg = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_PREVIEW_PER_CLUSTER}_preview_heatmap.svg"
    log("[PLOT] preview top response heatmap ...")
    plot_response_heatmap(
        mat=preview_mat,
        row_ann_df=preview_row_ann,
        clusters=clusters,
        cluster_label_df=cluster_label_df,
        out_png=preview_png,
        out_svg=preview_svg,
        title=(
            f"Preview: CA_all cluster-specific response genes after {TARGET_GENE} {MODE.upper()}\n"
            f"rows = top{TOP_N_PREVIEW_PER_CLUSTER} response genes per cluster"
        ),
        topn_label=f"top{TOP_N_PREVIEW_PER_CLUSTER} preview per cluster",
        fig_h_max=PREVIEW_FIG_H_MAX,
    )
    log("[SAVE]", preview_png)

    preview_block_png = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_PREVIEW_PER_CLUSTER}_preview_rowblock_annotation.png"
    preview_block_svg = out_dir / f"CA_all_originalCelltype_cluster_response_top{TOP_N_PREVIEW_PER_CLUSTER}_preview_rowblock_annotation.svg"
    log("[PLOT] preview row-block annotation as a separate figure ...")
    plot_row_block_annotation(
        row_ann_df=preview_row_ann,
        clusters=clusters,
        cluster_label_df=cluster_label_df,
        out_png=preview_block_png,
        out_svg=preview_block_svg,
        title=f"Row blocks: top{TOP_N_PREVIEW_PER_CLUSTER} genes per cluster",
        fig_h_max=PREVIEW_FIG_H_MAX,
    )
    log("[SAVE]", preview_block_png)

    # Save brief top20 list for quick reading
    brief_rows = []
    for cl, sub in top_df.groupby("cluster_label", sort=False):
        sub2 = sub.sort_values("rank").head(20)
        brief_rows.append({
            "cluster_label": str(cl),
            "top20_genes": json.dumps(sub2["gene"].astype(str).tolist(), ensure_ascii=False),
        })
    brief_csv = out_dir / "CA_all_originalCelltype_cluster_response_top20_gene_list_brief.csv"
    pd.DataFrame(brief_rows).to_csv(brief_csv, index=False)
    log("[SAVE]", brief_csv)

    log("\n[DONE] outputs in:")
    log(out_dir.resolve())


if __name__ == "__main__":
    main()
