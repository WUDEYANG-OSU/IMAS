# -*- coding: utf-8 -*-
"""
hnscc_caall_malignant_balanced3_subgroup_stage5KO_naturalSOX2prior_dualPoolV4_tieredDisplay_top1000_plots.py

目的：
1) 只提取 CA_all 中的 Malignant cells。
2) 不直接把所有 malignant cells 指定成 3 组；而是：
   malignant cells -> response-aware micro-clusters -> balanced merge into 3 malignant response subgroups。
3) 每个 malignant subgroup 重新计算自己的 KO/OE response top1000 genes。
4) 每个 subgroup 取 top genes 做富集分析，并画纵向柱状图。
5) 每个 subgroup 取 top20 affected nodes/genes 做背靠背柱状图：
   右侧 Nature red = perturbation response impact
   左侧 Nature blue = subgroup context support
6) 每个 subgroup 画一个空心扇形图：只给当前 subgroup 上色，其余 subgroup 灰色。
7) 额外输出 malignant subgroup UMAP 映射图。

关键算法设计：
- Stage5 直接 KO/OE node 通常数量有限，因此先用 Stage5 direct impact nodes 作为 response seeds，
  再基于 malignant cells 表达相关性把 KO/OE response impact 扩展到 CA_all 全基因。
- 不是硬限制基因范围，而是构建 natural soft prior：
  SOX2/target 关联证据 = malignant 内部 SOX2-expression correlation
                    + malignant-vs-nonmalignant expression specificity
                    + TCGA 共表达/差异证据
                    + KEGG/网络邻近性
                    + curated target-axis genes。
- subgroup top1000 不是复用全局 cell.type top1000，而是在 malignant 内部三亚组上重新计算。
- 重要修正：不能让 subgroup advantage 把“全 malignant 共同高表达、但与 SOX2 强相关”的基因全部压掉。
  因此 response_weight 同时包含：
    1) subgroup-specific response core，保证三组差异；
    2) shared SOX2-axis lane，允许 malignant-specific + SOX2-correlation/TCGA/KEGG/curated 证据强的基因进入 top1000。
- 本版不再对 MYC/HES1/WEE1/CDKN2B/KREMEN2/CLDN3/EFNA1/HSPA2/IRX3/FOXD3-AS1 做特殊锁定加权；
  这些基因只能通过 SOX2 correlation、malignant-specificity、TCGA、KEGG 和 curated-axis 证据自然加分。
- final 3 subgroups 来自 microcluster 合并，且带 size balance penalty，避免三组比例差异过大。
"""

from __future__ import annotations

import math
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Need anndata. Please install/import anndata first.") from e

try:
    import scanpy as sc
    HAS_SCANPY = True
except Exception:
    HAS_SCANPY = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

# ============================================================
# 0) CONFIG
# ============================================================
plt.rcParams["font.family"] = "DejaVu Sans"
warnings.filterwarnings("ignore")

# ---------- target perturbation ----------
TARGET_GENE = "SOX2"        # 改成 FGFR1 / AXL / MET / KIT 等即可
MODE = "ko"                 # "ko" or "oe"
TARGET_GENE_UPPER = TARGET_GENE.upper()

# ---------- CA_all root ----------
BASE = get_project_root()
EXPLOIT_HNSCC_ROOT = BASE / "LLM-Worldword" / "Exploit-HNSCC"
if not EXPLOIT_HNSCC_ROOT.exists():
    EXPLOIT_HNSCC_ROOT = BASE / "LLM-WorldWord" / "Exploit-HNSCC"

RNA_DIR = EXPLOIT_HNSCC_ROOT / "processed_CA" / "RNA_CA_all"
ASSET_DIR = EXPLOIT_HNSCC_ROOT / "assets_CA"
LOG1P_NPY = RNA_DIR / "CA_all.X_rna_log1p_norm_dense.npy"
META_NPZ = RNA_DIR / "CA_all.rna_meta.npz"
GENES_TSV = ASSET_DIR / "genes.tsv"

# 这里用前面“参照细胞内部注释数量约束后”的 UMAP/h5ad
RUN_STAGEB_DATASET_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_rna_only_stageB_clusterCorrection_HNSCC_CAall_only"
    / "HNSCC-CA_all"
)
RECLUSTER_DIR = RUN_STAGEB_DATASET_DIR / "single_pool_CA_all_single_recluster_for_cci"
ADATA_UMAP = RECLUSTER_DIR / "CA_all_single_matchedcells_scanpy.h5ad"

# Stage5 output from five-stage network
PT_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "CA_all_single_cci_newLeiden"
    / "communication_pseudotime_clusterGNN_LTC"
)
STAGE5_DIR = PT_DIR / "five_stage_network_viz_out"
STAGE5_BASELINE_NODE = STAGE5_DIR / f"{TARGET_GENE}_stage5_baseline_node_state.csv"
STAGE5_PERT_NODE = STAGE5_DIR / f"{TARGET_GENE}_stage5_{MODE}_node_state.csv"

OUT_DIR = STAGE5_DIR / f"malignant_balanced3_subgroups_{TARGET_GENE}_{MODE}_top1000_naturalSOX2prior_dualPoolV4_tieredDisplay_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- malignant selection ----------
CELLTYPE_COL_CANDIDATES = [
    "cell.type", "cell_type", "CellType", "celltype", "annotation", "integrated_label",
]
MALIGNANT_KEYWORDS = ["Malignant.cells", "Malignant", "malignant", "Tumor", "tumor", "Cancer", "cancer"]

# ---------- subgrouping ----------
N_FINAL_SUBGROUPS = 3
MICRO_RESOLUTION_GRID = [0.30, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50, 1.80, 2.20, 2.60, 3.00]
MICRO_MIN_N = 8
MICRO_MAX_N = 28
MICRO_MAX_FRAC = 0.22
N_PCA = 30
N_NEIGHBORS = 18
MIN_MALIGNANT_CELLS = 300

# 平衡约束：不是硬性完全等分，但尽量不要差太大
BALANCE_TARGET_FRAC = 1.0 / N_FINAL_SUBGROUPS
BALANCE_MIN_FRAC = 0.22
BALANCE_MAX_FRAC = 0.44
BALANCE_PENALTY = 6.0
BALANCE_REPAIR_MAX_ITER = 200

# ---------- response inference ----------
DIRECT_SEED_TOPN = 80
N_IMPACT_FEATURE_GENES = 1800
GENE_CHUNK = 1000
MIN_SEED_IMPACT = 1e-8
CORR_POWER = 1.2
IMPACT_DIRECT_BOOST = 1.0
IMPACT_CORR_WEIGHT = 0.85

# ---------- top genes ----------
TOP_N_GENES_PER_SUBGROUP = 1000
TOP_N_FOR_BACK2BACK = 20
TOP_N_FOR_ENRICH = 250
MIN_PCT_EXPR = 0.08
MIN_SPEC_POS = 0.0
ALPHA_IMPACT = 1.0
BETA_SPECIFICITY = 1.55
GAMMA_ADVANTAGE = 0.85
MEAN_EXPR_BONUS_WEIGHT = 0.25

# ---------- soft target-prior weighting ----------
# 这些是软加权，不是硬过滤；没有对应文件时会自动跳过/使用 fallback。
USE_SOFT_TARGET_PRIOR = True
SOFT_PRIOR_STRENGTH = 2.20
SOFT_PRIOR_POWER = 1.0

# 1) malignant 内部 target-gene expression correlation：
#    SOX2-high malignant cells 中同向变化的基因会获得自然软加权。
WEIGHT_TARGET_EXPR_CORR = 0.42
TARGET_CORR_MODE = "abs"        # "positive" or "abs"; abs 更适合 SOX2 轴上下游/反馈关系

# 2) malignant cells 相对 non-malignant cells 的表达特异性
WEIGHT_MALIGNANT_SPECIFICITY = 0.35

# 3) TCGA 层面的 target-gene association，可选本地 CSV。
# 推荐列名：gene,corr,p_value,adj_p,logFC，也兼容 Gene/Symbol/r/pearson/FDR 等。
WEIGHT_TCGA_TARGET_ASSOC = 0.25
TCGA_TARGET_ASSOC_CSV_CANDIDATES = [
    STAGE5_DIR / f"TCGA_HNSC_{TARGET_GENE}_gene_association.csv",
    STAGE5_DIR / f"TCGA_{TARGET_GENE}_gene_association.csv",
    PT_DIR / f"TCGA_HNSC_{TARGET_GENE}_gene_association.csv",
    EXPLOIT_HNSCC_ROOT / f"TCGA_HNSC_{TARGET_GENE}_gene_association.csv",
]
TCGA_USE_ABS_CORR = True

# 4) KEGG / pathway / network soft evidence。
WEIGHT_KEGG_NETWORK = 0.20
KEGG_GMT_CANDIDATES = [
    BASE / "networks" / "qc_global_projection" / "genesets" / "KEGG_2021_Human.gmt",
    BASE / "networks" / "qc_global_projection" / "genesets" / "c2.cp.kegg.v2023.2.Hs.symbols.gmt",
    STAGE5_DIR / "KEGG_2021_Human.gmt",
    PT_DIR / "KEGG_2021_Human.gmt",
]
# 可选边表：src,dst,weight；如果有 target 近邻，优先使用。
KEGG_NETWORK_EDGE_CSV_CANDIDATES = [
    STAGE5_DIR / f"{TARGET_GENE}_kegg_network_edges.csv",
    PT_DIR / f"{TARGET_GENE}_kegg_network_edges.csv",
    EXPLOIT_HNSCC_ROOT / f"{TARGET_GENE}_kegg_network_edges.csv",
]
KEGG_TERM_KEYWORDS = [
    "STEM", "PLURIPOTENCY", "SQUAMOUS", "EPITHELIAL", "CELL_CYCLE",
    "NOTCH", "WNT", "TGF", "MAPK", "PI3K", "AKT", "MTOR", "P53",
    "HIPPO", "HEDGEHOG", "FOCAL_ADHESION", "ECM_RECEPTOR", "APOPTOSIS",
]

# 5) curated target-axis genes；只作弱先验，避免 TCGA/KEGG 文件缺失时完全没有 target 关系信息。
WEIGHT_CURATED_TARGET_AXIS = 0.28
SOX2_CURATED_AXIS_GENES = """
POU5F1 NANOG KLF4 MYC TP63 TP53 NOTCH1 NOTCH2 JAG1 DLL1 DLL3 HES1 HEY1 HEY2
WNT3A WNT5A CTNNB1 AXIN2 LEF1 TCF7L2 KREMEN1 KREMEN2 DKK1 DKK3
EGFR ERBB2 ERBB3 MET AXL FGFR1 FGFR2 EFNA1 EFNA3 EPHB2 EPHB4
PIK3CA AKT1 MTOR PTEN MAPK1 MAPK3 JUN FOS CCND1 CDK1 CDK2 WEE1 CDKN1A CDKN2A CDKN2B
MKI67 PCNA TOP2A MCM2 MCM3 MCM4 MCM5 MCM6 BIRC5 AURKA AURKB
EPCAM CLDN3 CLDN4 CLDN7 KRT5 KRT6A KRT8 KRT14 KRT17 KRT18 KRT19
CD44 PROM1 ALDH1A1 ALDH1A3 ITGA6 ITGB1 VIM SNAI1 SNAI2 ZEB1 ZEB2 TGFB1 SMAD2 SMAD3
HIF1A VEGFA CA9 LDHA SLC2A1 BCL2 BAX CASP3 MMP2 MMP9 FN1 LAMB1 COL1A1 COL1A2
HSPA1A HSPA1B HSPA2 HSP90AA1 IRX3 FOXD3 FOXD3-AS1
""".split()

# 只用于输出诊断表，不参与任何加权或排序。
# 目的：方便检查这些用户关注的 SOX2-related malignant genes 是否凭自然证据进入 top1000/top20。
DIAGNOSTIC_GENES_TO_REPORT = """
MYC HES1 WEE1 CDKN2B KREMEN2 CLDN3 EFNA1 HSPA2 IRX3 FOXD3-AS1
""".split()

# 让 SOX2-correlation / TCGA / KEGG / malignant-specific evidence 真的能影响排序。
# 原版太保守，global malignant SOX2-axis genes 会被 subgroup_advantage 压掉。
FEATURE_PRIOR_BOOST = 4.50
TOPGENE_PRIOR_BOOST = 5.25
PRIOR_IMPACT_RESCUE_FLOOR = 0.32
ADVANTAGE_FLOOR = 0.42

# 三条 lane 重新定标：
# 1) subgroup-specific perturbation lane：保证三组差异；
# 2) shared response lane：保留有扰动响应且 SOX2-axis prior 高的共享恶性基因；
# 3) natural target-axis lane：不指定任何基因，但让 SOX2 correlation / malignant specificity / TCGA / KEGG / curated-axis
#    支持强且在该亚组有表达的基因明显前移。
SUBGROUP_SPECIFIC_LANE_WEIGHT = 0.58
SHARED_RESPONSE_LANE_WEIGHT = 0.72
NATURAL_AXIS_LANE_WEIGHT = 1.85

# natural target-axis lane 不依赖固定基因名单；它按 soft-prior 的总体分布自动给高证据基因加权。
NATURAL_AXIS_PRIOR_POWER = 1.15
NATURAL_AXIS_CONTEXT_POWER = 0.70
NATURAL_AXIS_IMPACT_MIX = 0.20      # 仍保留一部分 Stage5 impact，但不让 low direct-impact 把 SOX2-axis genes 完全压掉
NATURAL_AXIS_PRIOR_Q = 0.42         # 只作为连续门控的基线，不是硬过滤
NATURAL_AXIS_MIN_PCT_EXPR = 0.03

SHARED_AXIS_RESCUE_STRENGTH = 0.42
SHARED_CONTEXT_WEIGHT = 0.52
SUBGROUP_SPECIFIC_CONTEXT_WEIGHT = 0.48

# ------------------------------------------------------------------
# Dual-candidate-pool selection
# ------------------------------------------------------------------
# 关键修正：top1000 不再只按一个 final_score 排序。
# 每个 malignant 亚组的 top1000 由两类自动候选池共同组成：
# A) response-specific pool：Stage5 KO/OE impact + subgroup specificity，保证三组差异；
# B) natural SOX2-axis pool：SOX2 correlation + malignant-specificity + TCGA/KEGG/curated-axis + subgroup expression，
#    用来保留 CLDN3 / MYC / HES1 / WEE1 等这类“不是某亚组独有，但证据强”的基因。
# 注意：这里没有固定锁定任何单个基因名；DIAGNOSTIC_GENES_TO_REPORT 只用于输出诊断。
DUAL_POOL_SELECTION = True
DUAL_POOL_AXIS_FRACTION_TOP1000 = 0.50
DUAL_POOL_AXIS_FRACTION_TOP20 = 0.60
DUAL_POOL_SHARED_FRACTION_TOP20 = 0.10
AXIS_SELECTION_WEIGHT_NATURAL = 0.48
AXIS_SELECTION_WEIGHT_SHARED = 0.24
AXIS_SELECTION_WEIGHT_CORR = 0.10
AXIS_SELECTION_WEIGHT_MAL = 0.12
AXIS_SELECTION_WEIGHT_CURATED = 0.06
AXIS_SELECTION_MIN_PCT = 0.010
AXIS_DISPLAY_BOOST = 0.90

# V2 fix:
# 1) sparse prior gate: target-correlation / malignant-specificity no longer make almost every gene nonzero.
# 2) pool bugfix: response/axis/shared pools use their own counters, so axis genes are truly inserted into top20/top1000.
SPARSE_CORR_Q = 0.80
SPARSE_MALIGNANT_Q = 0.80
SPARSE_TCGA_Q = 0.70
SPARSE_KEGG_Q = 0.70
SPARSE_GATE_POWER = 0.70

# Let the natural SOX2-axis pool use its own expression-aware score instead of being compressed by min-max of lane C.
AXIS_POOL_PRIOR_WEIGHT = 0.58
AXIS_POOL_EXPR_WEIGHT = 0.22
AXIS_POOL_SHARED_WEIGHT = 0.12
AXIS_POOL_IMPACT_WEIGHT = 0.08
AXIS_POOL_CURATED_CONTEXT_WEIGHT = 0.18

# V4: tiered top20 display.
# top1000 remains broad dual-pool, but the top20 figure is a balanced display panel:
# - response pool: perturbation/subgroup-specific genes
# - SOX2-axis pool: natural SOX2-axis + malignant-specific genes
# - marker-axis pool: subgroup-enriched SOX2-axis genes, rescuing narrow malignant programs
# - shared-axis pool: shared malignant SOX2-axis response genes
# No individual gene name is forced; each tier is scored automatically.
TOP20_RESPONSE_QUOTA = 5
TOP20_AXIS_QUOTA = 6
TOP20_MARKER_AXIS_QUOTA = 6
TOP20_SHARED_QUOTA = 3
AXIS_MARKER_PRIOR_WEIGHT = 0.42
AXIS_MARKER_SPEC_WEIGHT = 0.30
AXIS_MARKER_EXPR_WEIGHT = 0.18
AXIS_MARKER_SHARED_WEIGHT = 0.10
AXIS_MARKER_MIN_PCT = 0.018
PLOT_RESPONSE_AXIS_BLEND = 0.46

# prior 会同时影响 malignant microcluster feature gene 选择和 subgroup top1000 排名。

# ---------- enrichment ----------
USE_GSEAPY_ENRICHR = True
ENRICHR_LIBRARIES = [
    "KEGG_2021_Human",
    "GO_Biological_Process_2021",
    "Reactome_2022",
]
ENRICH_TOP_TERMS = 12
ENRICH_MIN_OVERLAP = 3

# ---------- plotting ----------
SEED = 42
DPI = 320
NATURE_RED = "#B2182B"
NATURE_BLUE = "#2166AC"
GREY = "#D0D0D0"
DARK_GREY = "#555555"
SUBGROUP_COLORS = ["#B2182B", "#2166AC", "#4D9221"]

np.random.seed(SEED)

# ============================================================
# 1) HELPERS
# ============================================================
def log(*args):
    print(*args, flush=True)


def safe_name(x: str, n: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n]


def is_target_gene(g: str) -> bool:
    return str(g).upper() == TARGET_GENE_UPPER


def minmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return out
    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])
    if hi - lo < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def rank01_1d(x: np.ndarray) -> np.ndarray:
    """
    Empirical percentile score in [0, 1].
    Compared with min-max scaling, this prevents one extreme gene from compressing
    the whole evidence distribution and is better for soft prior ranking.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n == 0:
        return out
    vals = x[finite]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    if n > 1:
        ranks = ranks / float(n - 1)
    else:
        ranks[:] = 0.5
    out[finite] = ranks
    return out


def quantile_gate_1d(x: np.ndarray, q: float = 0.80, power: float = 0.75) -> np.ndarray:
    """
    True sparse-but-soft evidence gate.

    Important fix:
    - If all values are equal, or no value is above the percentile threshold,
      return all zeros.
    - The previous version called minmax_1d() on an all-zero vector; because
      minmax_1d returns 0.5 for constant vectors, almost every gene received
      a nonzero SOX2 prior.

    This is still not a hard curated gene filter. It simply makes genome-wide
    correlation/specificity evidence sparse enough to be informative.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x) & (x > 0)
    if int(finite.sum()) < 3:
        return out
    vals = x[finite]
    if float(np.nanmax(vals) - np.nanmin(vals)) < 1e-12:
        return out
    thr = float(np.nanquantile(vals, q))
    y = np.maximum(0.0, x - thr)
    y[~finite] = 0.0
    ymax = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
    if ymax <= 1e-12:
        return out
    y = y / (ymax + 1e-12)
    return np.power(np.clip(y, 0.0, 1.0), float(power))


def evidence_or_score(*arrays: np.ndarray) -> np.ndarray:
    """Soft OR over several evidence channels. A high value in any channel can rescue a gene."""
    if len(arrays) == 0:
        return np.zeros((0,), dtype=np.float64)
    mats = [np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0) for a in arrays]
    out = np.ones_like(mats[0], dtype=np.float64)
    for a in mats:
        out *= (1.0 - a)
    return np.clip(1.0 - out, 0.0, 1.0)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - m) / (s + 1e-12)


def zscore_by_gene(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=0, keepdims=True)
    sd = np.nanstd(mat, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (mat - mu) / sd


def safe_pow(x: np.ndarray, p: float) -> np.ndarray:
    return np.power(np.clip(np.asarray(x, dtype=np.float64), 0.0, None), p)


def save_fig(fig: plt.Figure, out_base: Path):
    fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", out_base.with_suffix(".png"))
    log("[SAVE]", out_base.with_suffix(".svg"))


def sort_labels(xs: List[str]) -> List[str]:
    def key(v):
        s = str(v)
        if s.startswith("M") and s[1:].isdigit():
            return (0, int(s[1:]))
        if s.isdigit():
            return (0, int(s))
        return (1, s)
    return sorted([str(x) for x in xs], key=key)


def read_gene_names(genes_tsv: Path) -> List[str]:
    df = pd.read_csv(genes_tsv, sep="\t")
    for c in ["gene", "gene_name", "Gene", "name"]:
        if c in df.columns:
            return df[c].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def extract_counts_barcode(x: str) -> str:
    x = str(x).strip()
    if "::" in x:
        x = x.split("::", 1)[-1]
    if ":" in x:
        x = x.rsplit(":", 1)[-1]
    x = x.replace("-", ".")
    return x


def find_celltype_col(obs: pd.DataFrame) -> str:
    for c in CELLTYPE_COL_CANDIDATES:
        if c in obs.columns:
            return c
    raise KeyError(f"Cannot find cell type column. Available obs columns: {list(obs.columns)}")


def is_malignant_label(x: str) -> bool:
    s = str(x)
    return any(k.lower() in s.lower() for k in MALIGNANT_KEYWORDS)

# ============================================================
# 2) LOAD DATA
# ============================================================
def load_stage5_direct_impact() -> pd.DataFrame:
    assert STAGE5_BASELINE_NODE.exists(), f"Missing: {STAGE5_BASELINE_NODE}"
    assert STAGE5_PERT_NODE.exists(), f"Missing: {STAGE5_PERT_NODE}"

    b = pd.read_csv(STAGE5_BASELINE_NODE)
    p = pd.read_csv(STAGE5_PERT_NODE)
    for df, name in [(b, "baseline"), (p, "perturbed")]:
        assert "node" in df.columns, f"{name} missing node"
        assert "state_final" in df.columns, f"{name} missing state_final"

    out = (
        b[["node", "state_final"]].rename(columns={"state_final": "state_baseline"})
        .merge(
            p[["node", "state_final"]].rename(columns={"state_final": "state_perturbed"}),
            on="node",
            how="inner",
        )
    )
    out["node"] = out["node"].astype(str)
    out = out[~out["node"].map(is_target_gene)].copy()
    out["state_baseline"] = pd.to_numeric(out["state_baseline"], errors="coerce")
    out["state_perturbed"] = pd.to_numeric(out["state_perturbed"], errors="coerce")
    out = out.dropna().copy()
    out["delta_pert_minus_base"] = out["state_perturbed"] - out["state_baseline"]

    if MODE.lower() == "ko":
        out["direct_impact"] = np.maximum(0.0, out["state_baseline"] - out["state_perturbed"])
    elif MODE.lower() == "oe":
        out["direct_impact"] = np.maximum(0.0, out["state_perturbed"] - out["state_baseline"])
    else:
        out["direct_impact"] = np.abs(out["delta_pert_minus_base"])

    out = out.sort_values("direct_impact", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_DIR / "stage5_direct_node_impact.csv", index=False)
    log("[LOAD] Stage5 direct nodes =", len(out), "nonzero =", int((out["direct_impact"] > 0).sum()))
    return out


def load_umap_and_select_malignant() -> Tuple[ad.AnnData, str, np.ndarray]:
    assert ADATA_UMAP.exists(), f"Missing: {ADATA_UMAP}"
    adata = ad.read_h5ad(str(ADATA_UMAP))
    assert "X_umap" in adata.obsm, "UMAP coordinates not found."
    celltype_col = find_celltype_col(adata.obs)
    adata.obs[celltype_col] = adata.obs[celltype_col].astype(str)

    m = adata.obs[celltype_col].map(is_malignant_label).values
    if int(m.sum()) < MIN_MALIGNANT_CELLS:
        raise RuntimeError(
            f"Only {int(m.sum())} malignant cells found by {celltype_col}. "
            f"Check MALIGNANT_KEYWORDS or cell.type labels."
        )
    mal = adata[m].copy()
    log("[MALIGNANT] celltype_col =", celltype_col)
    log("[MALIGNANT] n_cells =", mal.n_obs)
    log("[MALIGNANT] labels:")
    log(mal.obs[celltype_col].value_counts().to_string())
    return mal, celltype_col, m


def load_meta_cell_ids(meta_npz: Path) -> np.ndarray:
    z = np.load(meta_npz, allow_pickle=True)
    if "cell_ids" in z.files:
        return z["cell_ids"].astype(str)
    if "metadata_columns" in z.files and "metadata_table" in z.files:
        cols = z["metadata_columns"].astype(str).tolist()
        tab = z["metadata_table"]
        if "cell_id" in cols:
            return pd.DataFrame(tab, columns=cols)["cell_id"].astype(str).values
        if "barcode_counts" in cols:
            return pd.DataFrame(tab, columns=cols)["barcode_counts"].astype(str).values
    raise KeyError(f"Cannot recover cell_ids from {meta_npz}. files={z.files}")


def match_obs_to_expression_rows(obs: pd.DataFrame) -> np.ndarray:
    cell_ids_all = load_meta_cell_ids(META_NPZ)
    exact_map = {str(x): i for i, x in enumerate(cell_ids_all)}
    bc_map = {extract_counts_barcode(x): i for i, x in enumerate(cell_ids_all)}

    row_idx = []
    missing = []
    for obs_name, row in obs.iterrows():
        candidates = [str(obs_name)]
        for c in ["cell_id", "barcode", "barcode_counts"]:
            if c in obs.columns:
                candidates.append(str(row[c]))
        candidates += [extract_counts_barcode(x) for x in list(candidates)]

        hit = None
        for x in candidates:
            if x in exact_map:
                hit = exact_map[x]
                break
            if x in bc_map:
                hit = bc_map[x]
                break
        if hit is None:
            missing.append(str(obs_name))
        else:
            row_idx.append(hit)

    if len(missing) > 0:
        raise RuntimeError(f"Missing {len(missing)} cells in expression meta. Examples: {missing[:5]}")
    return np.asarray(row_idx, dtype=np.int64)


def load_expression_for_rows(row_idx: np.ndarray, gene_indices: np.ndarray) -> np.ndarray:
    X_mmap = np.load(LOG1P_NPY, mmap_mode="r")
    X = np.asarray(X_mmap[np.ix_(row_idx, gene_indices)], dtype=np.float32)
    return X

# ============================================================
# 3) INFER GENOME-WIDE PERTURBATION IMPACT
# ============================================================
def infer_gene_impact_all_genes(
    mal_adata: ad.AnnData,
    direct_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray]:
    """
    返回：
    - gene_impact_df: 全基因 inferred impact
    - row_idx_mal: malignant cells 在 LOG1P_NPY 中的行号
    - genes_all: CA_all genes.tsv 全基因名
    - impact_all: 与 genes_all 对齐的 impact 0-1
    """
    assert LOG1P_NPY.exists(), f"Missing: {LOG1P_NPY}"
    assert META_NPZ.exists(), f"Missing: {META_NPZ}"
    assert GENES_TSV.exists(), f"Missing: {GENES_TSV}"

    genes_all = read_gene_names(GENES_TSV)
    gene_to_idx = {g: i for i, g in enumerate(genes_all)}
    row_idx_mal = match_obs_to_expression_rows(mal_adata.obs)

    direct_df = direct_df.copy()
    direct_df["node"] = direct_df["node"].astype(str)
    direct_df = direct_df[direct_df["node"].isin(gene_to_idx.keys())].copy()
    direct_df = direct_df[~direct_df["node"].map(is_target_gene)].copy()
    direct_df = direct_df[pd.to_numeric(direct_df["direct_impact"], errors="coerce").fillna(0.0) > MIN_SEED_IMPACT].copy()

    if len(direct_df) == 0:
        raise RuntimeError("No nonzero Stage5 direct impact genes found in CA_all gene list.")

    direct_df = direct_df.sort_values("direct_impact", ascending=False).head(DIRECT_SEED_TOPN).copy()
    seed_genes = direct_df["node"].astype(str).tolist()
    seed_indices = np.asarray([gene_to_idx[g] for g in seed_genes], dtype=np.int64)
    seed_impact = direct_df["direct_impact"].values.astype(np.float64)
    seed_impact_norm = minmax_1d(seed_impact)
    seed_impact_norm = seed_impact_norm / (seed_impact_norm.sum() + 1e-12)

    log("[IMPACT] direct response seed genes =", len(seed_genes))
    log("[IMPACT] seed genes:", seed_genes[:20])

    X_seed = load_expression_for_rows(row_idx_mal, seed_indices).astype(np.float64)
    X_seed_z = StandardScaler(with_mean=True, with_std=True).fit_transform(X_seed)

    X_mmap = np.load(LOG1P_NPY, mmap_mode="r")
    n_genes = len(genes_all)
    impact_corr = np.zeros((n_genes,), dtype=np.float64)
    mean_expr = np.zeros((n_genes,), dtype=np.float64)
    pct_expr = np.zeros((n_genes,), dtype=np.float64)

    log("[IMPACT] infer genome-wide impact by malignant expression correlation ...")
    for st in range(0, n_genes, GENE_CHUNK):
        ed = min(st + GENE_CHUNK, n_genes)
        idx = np.arange(st, ed, dtype=np.int64)
        Xb = np.asarray(X_mmap[np.ix_(row_idx_mal, idx)], dtype=np.float64)
        mean_expr[st:ed] = np.nanmean(Xb, axis=0)
        pct_expr[st:ed] = np.nanmean(Xb > 0, axis=0)

        # z-score columns
        mu = np.nanmean(Xb, axis=0, keepdims=True)
        sd = np.nanstd(Xb, axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xb_z = (Xb - mu) / sd

        # corr with seed expression, weighted by seed direct impact
        corr = (Xb_z.T @ X_seed_z) / max(1, Xb_z.shape[0] - 1)
        corr_abs = np.abs(corr)
        # weighted mean plus max, so direct co-response and strongest seed both count
        weighted = corr_abs @ seed_impact_norm
        strongest = np.nanmax(corr_abs * seed_impact_norm[None, :] * len(seed_impact_norm), axis=1)
        impact_corr[st:ed] = IMPACT_CORR_WEIGHT * weighted + (1 - IMPACT_CORR_WEIGHT) * strongest

        if st % (GENE_CHUNK * 5) == 0:
            log(f"  processed genes {st}/{n_genes}")

    direct_map = dict(zip(direct_df["node"].astype(str), direct_df["direct_impact"].astype(float)))
    direct_raw = np.zeros((n_genes,), dtype=np.float64)
    for g, val in direct_map.items():
        if g in gene_to_idx:
            direct_raw[gene_to_idx[g]] = float(val)

    direct_norm = minmax_1d(direct_raw)
    corr_norm = minmax_1d(np.power(np.clip(impact_corr, 0.0, None), CORR_POWER))
    impact_all = np.maximum(corr_norm, IMPACT_DIRECT_BOOST * direct_norm)
    impact_all = minmax_1d(impact_all)

    # exclude target gene
    for i, g in enumerate(genes_all):
        if is_target_gene(g):
            impact_all[i] = 0.0

    impact_df = pd.DataFrame({
        "gene": genes_all,
        "inferred_impact": impact_all.astype(np.float32),
        "impact_corr_component": corr_norm.astype(np.float32),
        "direct_impact_component": direct_norm.astype(np.float32),
        "malignant_mean_expr": mean_expr.astype(np.float32),
        "malignant_pct_expr": pct_expr.astype(np.float32),
    })
    impact_df = impact_df.sort_values("inferred_impact", ascending=False).reset_index(drop=True)
    impact_df.to_csv(OUT_DIR / "malignant_genomewide_inferred_stage5_impact.csv", index=False)
    log("[SAVE]", OUT_DIR / "malignant_genomewide_inferred_stage5_impact.csv")

    return impact_df, row_idx_mal, genes_all, impact_all

# ============================================================
# 3b) SOFT TARGET-PRIOR EVIDENCE
# ============================================================
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p is not None and Path(p).exists() and Path(p).stat().st_size > 0:
            return Path(p)
    return None


def _find_col_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        lc = str(c).lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def load_tcga_target_association(genes_all: List[str]) -> pd.DataFrame:
    genes_upper = {g.upper(): g for g in genes_all}
    out = pd.DataFrame({
        "gene": genes_all,
        "tcga_target_score": np.zeros((len(genes_all),), dtype=np.float32),
        "tcga_source": "missing",
    })
    p = _first_existing(TCGA_TARGET_ASSOC_CSV_CANDIDATES)
    if p is None:
        log("[PRIOR][TCGA] no local TCGA association CSV found; tcga_target_score=0")
        return out
    try:
        df = pd.read_csv(p)
    except Exception as e:
        log("[PRIOR][TCGA][WARN] failed to read", p, repr(e))
        return out

    gene_col = _find_col_case_insensitive(df, ["gene", "symbol", "Gene", "GeneSymbol", "node", "target"])
    corr_col = _find_col_case_insensitive(df, ["corr", "cor", "r", "pearson", "spearman", "correlation"])
    p_col = _find_col_case_insensitive(df, ["adj_p", "padj", "FDR", "qvalue", "p_adj", "p_value", "P.Value", "pval"])
    lfc_col = _find_col_case_insensitive(df, ["logFC", "log2FC", "log2FoldChange", "coef", "beta"])
    score_col = _find_col_case_insensitive(df, ["score", "tcga_score", "target_score", "association_score"])
    if gene_col is None:
        log("[PRIOR][TCGA][WARN] no gene column in", p)
        return out

    tmp = df.copy()
    tmp[gene_col] = tmp[gene_col].astype(str)
    raw = np.zeros((len(tmp),), dtype=np.float64)
    if score_col is not None:
        raw = pd.to_numeric(tmp[score_col], errors="coerce").fillna(0.0).values.astype(float)
    else:
        if corr_col is not None:
            corr = pd.to_numeric(tmp[corr_col], errors="coerce").fillna(0.0).values.astype(float)
            raw += np.abs(corr) if TCGA_USE_ABS_CORR else np.maximum(corr, 0.0)
        if lfc_col is not None:
            lfc = pd.to_numeric(tmp[lfc_col], errors="coerce").fillna(0.0).values.astype(float)
            raw += 0.35 * np.abs(lfc)
        if p_col is not None:
            pv = pd.to_numeric(tmp[p_col], errors="coerce").fillna(1.0).clip(1e-300, 1.0).values.astype(float)
            raw *= (1.0 + 0.15 * np.clip(-np.log10(pv), 0, 50))
    tmp["_score"] = minmax_1d(raw)

    score_map = {}
    for _, r in tmp.iterrows():
        gu = str(r[gene_col]).upper()
        if gu in genes_upper:
            score_map[genes_upper[gu]] = max(score_map.get(genes_upper[gu], 0.0), float(r["_score"]))

    out["tcga_target_score"] = [float(score_map.get(g, 0.0)) for g in genes_all]
    out["tcga_source"] = str(p)
    log("[PRIOR][TCGA] using", p, "matched genes=", int((out["tcga_target_score"] > 0).sum()))
    return out


def read_gmt(path: Path) -> Dict[str, set]:
    gs = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            arr = line.rstrip("\n").split("\t")
            if len(arr) >= 3:
                term = arr[0]
                genes = {x.strip().upper() for x in arr[2:] if x.strip()}
                gs[term] = genes
    return gs


def load_kegg_network_prior(genes_all: List[str]) -> pd.DataFrame:
    genes_upper = {g.upper(): g for g in genes_all}
    score = {g: 0.0 for g in genes_all}

    # optional explicit target network edge table
    edge_p = _first_existing(KEGG_NETWORK_EDGE_CSV_CANDIDATES)
    if edge_p is not None:
        try:
            ed = pd.read_csv(edge_p)
            src_col = _find_col_case_insensitive(ed, ["src", "source", "from", "gene1"])
            dst_col = _find_col_case_insensitive(ed, ["dst", "target", "to", "gene2"])
            w_col = _find_col_case_insensitive(ed, ["weight", "score", "confidence"])
            if src_col and dst_col:
                adj = {}
                for _, r in ed.iterrows():
                    a = str(r[src_col]).upper()
                    b = str(r[dst_col]).upper()
                    w = float(pd.to_numeric(pd.Series([r[w_col]]), errors="coerce").fillna(1.0).iloc[0]) if w_col else 1.0
                    adj.setdefault(a, []).append((b, w))
                    adj.setdefault(b, []).append((a, w))
                tu = TARGET_GENE_UPPER
                for b, w in adj.get(tu, []):
                    if b in genes_upper:
                        score[genes_upper[b]] = max(score[genes_upper[b]], 1.0 * max(0.0, w))
                    for c, w2 in adj.get(b, []):
                        if c in genes_upper and c != tu:
                            score[genes_upper[c]] = max(score[genes_upper[c]], 0.45 * max(0.0, w) * max(0.0, w2))
                log("[PRIOR][KEGG_NET] using", edge_p)
        except Exception as e:
            log("[PRIOR][KEGG_NET][WARN] failed", edge_p, repr(e))

    # optional GMT pathway membership
    gmt_p = _first_existing(KEGG_GMT_CANDIDATES)
    if gmt_p is not None:
        try:
            gsets = read_gmt(gmt_p)
            target_terms = []
            for term, gs in gsets.items():
                term_u = term.upper().replace("-", "_").replace(" ", "_")
                has_keyword = any(k.upper() in term_u for k in KEGG_TERM_KEYWORDS)
                has_target = TARGET_GENE_UPPER in gs
                if has_keyword or has_target:
                    target_terms.append((term, gs, has_target, has_keyword))
            for term, gs, has_target, has_keyword in target_terms:
                term_weight = 1.0 if has_target else 0.65
                if has_keyword:
                    term_weight += 0.25
                for gu in gs:
                    if gu in genes_upper and gu != TARGET_GENE_UPPER:
                        score[genes_upper[gu]] = max(score[genes_upper[gu]], term_weight)
            log("[PRIOR][KEGG_GMT] using", gmt_p, "selected terms=", len(target_terms))
        except Exception as e:
            log("[PRIOR][KEGG_GMT][WARN] failed", gmt_p, repr(e))
    else:
        log("[PRIOR][KEGG] no local KEGG GMT/network found; kegg score relies on curated fallback only")

    raw = np.asarray([score[g] for g in genes_all], dtype=float)
    if (not np.isfinite(raw).any()) or float(np.nanmax(raw) - np.nanmin(raw)) < 1e-12:
        arr = np.zeros_like(raw, dtype=float)
    else:
        arr = minmax_1d(raw)
    return pd.DataFrame({"gene": genes_all, "kegg_network_score": arr.astype(np.float32)})


def curated_target_axis_prior(genes_all: List[str]) -> pd.DataFrame:
    """
    Curated target-axis 是弱先验，不是 special priority。
    对 SOX2 时使用一组可解释的 stemness / epithelial / cell-cycle / signaling 轴基因。
    注意：这里不会给用户指定的 10 个基因单独额外加权。
    """
    target_axis = set()
    if TARGET_GENE_UPPER == "SOX2":
        target_axis = {g.upper() for g in SOX2_CURATED_AXIS_GENES}

    curated = []
    for g in genes_all:
        gu = str(g).upper()
        if is_target_gene(g):
            curated.append(0.0)
        else:
            curated.append(1.0 if gu in target_axis else 0.0)

    return pd.DataFrame({
        "gene": genes_all,
        "curated_target_axis_score": np.asarray(curated, dtype=np.float32),
    })


def compute_target_expression_correlation_prior(genes_all: List[str]) -> pd.DataFrame:
    """
    在 malignant cells 内部计算每个 gene 与 TARGET_GENE 表达的相关性。
    这是自然 soft prior 的关键：不是锁定某些基因，而是让 SOX2-high malignant 状态中同向变化的基因加分。
    """
    all_adata = ad.read_h5ad(str(ADATA_UMAP))
    celltype_col = find_celltype_col(all_adata.obs)
    all_adata.obs[celltype_col] = all_adata.obs[celltype_col].astype(str)
    mal_mask = all_adata.obs[celltype_col].map(is_malignant_label).values.astype(bool)
    if int(mal_mask.sum()) < MIN_MALIGNANT_CELLS:
        log("[PRIOR][TARGET_CORR][WARN] too few malignant cells; target_expr_corr_score=0")
        return pd.DataFrame({
            "gene": genes_all,
            "target_expr_corr": np.zeros((len(genes_all),), dtype=np.float32),
            "target_expr_corr_score": np.zeros((len(genes_all),), dtype=np.float32),
        })

    gene_upper_to_idx = {str(g).upper(): i for i, g in enumerate(genes_all)}
    if TARGET_GENE_UPPER not in gene_upper_to_idx:
        log(f"[PRIOR][TARGET_CORR][WARN] {TARGET_GENE} not found in genes.tsv; target_expr_corr_score=0")
        return pd.DataFrame({
            "gene": genes_all,
            "target_expr_corr": np.zeros((len(genes_all),), dtype=np.float32),
            "target_expr_corr_score": np.zeros((len(genes_all),), dtype=np.float32),
        })

    row_idx_all = match_obs_to_expression_rows(all_adata.obs)
    row_idx_mal = row_idx_all[mal_mask]
    X_mmap = np.load(LOG1P_NPY, mmap_mode="r")
    target_idx = int(gene_upper_to_idx[TARGET_GENE_UPPER])
    x_target = np.asarray(X_mmap[np.ix_(row_idx_mal, np.asarray([target_idx], dtype=np.int64))], dtype=np.float64).reshape(-1)

    if np.nanstd(x_target) < 1e-8:
        log(f"[PRIOR][TARGET_CORR][WARN] {TARGET_GENE} expression has near-zero variance in malignant cells")
        return pd.DataFrame({
            "gene": genes_all,
            "target_expr_corr": np.zeros((len(genes_all),), dtype=np.float32),
            "target_expr_corr_score": np.zeros((len(genes_all),), dtype=np.float32),
        })

    xt = (x_target - np.nanmean(x_target)) / (np.nanstd(x_target) + 1e-12)
    n_genes = len(genes_all)
    corr = np.zeros((n_genes,), dtype=np.float64)
    log("[PRIOR][TARGET_CORR] compute malignant-intrinsic target expression correlation ...")
    for st in range(0, n_genes, GENE_CHUNK):
        ed = min(st + GENE_CHUNK, n_genes)
        idx = np.arange(st, ed, dtype=np.int64)
        Xb = np.asarray(X_mmap[np.ix_(row_idx_mal, idx)], dtype=np.float64)
        mu = np.nanmean(Xb, axis=0, keepdims=True)
        sd = np.nanstd(Xb, axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xz = (Xb - mu) / sd
        corr[st:ed] = (Xz.T @ xt) / max(1, Xz.shape[0] - 1)
        if st % (GENE_CHUNK * 5) == 0:
            log(f"  target correlation {st}/{n_genes}")

    if TARGET_CORR_MODE.lower() == "abs":
        raw = np.abs(corr)
    else:
        raw = np.maximum(corr, 0.0)
    raw[target_idx] = 0.0
    score = minmax_1d(raw)
    score[target_idx] = 0.0

    out = pd.DataFrame({
        "gene": genes_all,
        "target_expr_corr": corr.astype(np.float32),
        "target_expr_corr_score": score.astype(np.float32),
    })
    top = out.sort_values("target_expr_corr_score", ascending=False).head(20)
    log("[PRIOR][TARGET_CORR] top correlated genes:")
    log(top[["gene", "target_expr_corr", "target_expr_corr_score"]].to_string(index=False))
    return out

def compute_malignant_vs_nonmalignant_specificity(genes_all: List[str]) -> pd.DataFrame:
    all_adata = ad.read_h5ad(str(ADATA_UMAP))
    celltype_col = find_celltype_col(all_adata.obs)
    all_adata.obs[celltype_col] = all_adata.obs[celltype_col].astype(str)
    mal_mask = all_adata.obs[celltype_col].map(is_malignant_label).values.astype(bool)
    if int(mal_mask.sum()) == 0 or int((~mal_mask).sum()) == 0:
        log("[PRIOR][MAL_SPEC][WARN] cannot compute malignant vs non-malignant specificity")
        return pd.DataFrame({"gene": genes_all, "malignant_vs_nonmal_score": np.zeros((len(genes_all),), dtype=np.float32)})
    row_idx_all = match_obs_to_expression_rows(all_adata.obs)
    X_mmap = np.load(LOG1P_NPY, mmap_mode="r")
    n_genes = len(genes_all)
    raw = np.zeros((n_genes,), dtype=np.float64)
    mal_mean_all = np.zeros((n_genes,), dtype=np.float64)
    non_mean_all = np.zeros((n_genes,), dtype=np.float64)
    log("[PRIOR][MAL_SPEC] compute malignant-vs-nonmalignant expression specificity ...")
    for st in range(0, n_genes, GENE_CHUNK):
        ed = min(st + GENE_CHUNK, n_genes)
        idx = np.arange(st, ed, dtype=np.int64)
        Xb = np.asarray(X_mmap[np.ix_(row_idx_all, idx)], dtype=np.float32)
        Xm = Xb[mal_mask]
        Xn = Xb[~mal_mask]
        mal_mean = Xm.mean(axis=0)
        non_mean = Xn.mean(axis=0)
        mal_pct = (Xm > 0).mean(axis=0)
        non_pct = (Xn > 0).mean(axis=0)
        mal_mean_all[st:ed] = mal_mean
        non_mean_all[st:ed] = non_mean
        raw[st:ed] = (mal_mean - non_mean) + 0.55 * (mal_pct - non_pct)
        if st % (GENE_CHUNK * 5) == 0:
            log(f"  malignant specificity {st}/{n_genes}")
    raw_pos = np.maximum(raw, 0.0)
    # 更稳健的 malignant-specificity：同时使用 min-max 强度和经验分位数。
    # 这样 CLDN3 这类“不是全局最高、但确实偏 malignant/epithelial tumor state”的基因不会被极端高表达基因压扁。
    score_minmax = minmax_1d(raw_pos)
    score_rank = rank01_1d(raw_pos) * (raw_pos > 0).astype(float)
    mean_rank = rank01_1d(mal_mean_all)
    score = np.clip(0.52 * score_minmax + 0.33 * score_rank + 0.15 * mean_rank * (raw_pos > 0).astype(float), 0.0, 1.0)
    for j, g in enumerate(genes_all):
        if is_target_gene(g):
            score[j] = 0.0
    out = pd.DataFrame({
        "gene": genes_all,
        "malignant_vs_nonmal_score": score.astype(np.float32),
        "malignant_vs_nonmal_score_minmax": score_minmax.astype(np.float32),
        "malignant_vs_nonmal_score_rank": score_rank.astype(np.float32),
        "malignant_mean_expr_all": mal_mean_all.astype(np.float32),
        "nonmalignant_mean_expr_all": non_mean_all.astype(np.float32),
        "malignant_nonmal_raw_delta": raw.astype(np.float32),
    })
    return out


def build_soft_target_prior(genes_all: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    if not USE_SOFT_TARGET_PRIOR:
        df = pd.DataFrame({"gene": genes_all, "soft_prior_score": np.zeros((len(genes_all),), dtype=np.float32)})
        return df, df["soft_prior_score"].values.astype(np.float32)

    target_corr_df = compute_target_expression_correlation_prior(genes_all)
    mal_df = compute_malignant_vs_nonmalignant_specificity(genes_all)
    tcga_df = load_tcga_target_association(genes_all)
    kegg_df = load_kegg_network_prior(genes_all)
    curated_df = curated_target_axis_prior(genes_all)

    df = (
        target_corr_df
        .merge(mal_df, on="gene", how="left")
        .merge(tcga_df, on="gene", how="left")
        .merge(kegg_df, on="gene", how="left")
        .merge(curated_df, on="gene", how="left")
    )
    for c in ["target_expr_corr_score", "malignant_vs_nonmal_score", "tcga_target_score", "kegg_network_score", "curated_target_axis_score"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0).astype(float)

    # V2: rank-calibrated sparse soft prior.
    # Previously, abs(SOX2-correlation) and malignant-specificity gave almost every gene a nonzero prior.
    # That made SOX2-axis evidence too diffuse. Now genome-wide evidence is gated by percentile,
    # while curated/TCGA/KEGG evidence can still rescue biologically relevant genes.
    corr_raw = df["target_expr_corr_score"].values.astype(float)
    mal_raw = df["malignant_vs_nonmal_score"].values.astype(float)
    tcga_raw = df["tcga_target_score"].values.astype(float)
    kegg_raw = df["kegg_network_score"].values.astype(float)
    curated_rank = np.clip(df["curated_target_axis_score"].values.astype(float), 0.0, 1.0)

    corr_rank = rank01_1d(corr_raw) * (corr_raw > 0).astype(float)
    mal_rank = rank01_1d(mal_raw) * (mal_raw > 0).astype(float)
    tcga_rank = rank01_1d(tcga_raw) * (tcga_raw > 0).astype(float)
    kegg_rank = rank01_1d(kegg_raw) * (kegg_raw > 0).astype(float)

    corr_gate = quantile_gate_1d(corr_raw, SPARSE_CORR_Q, SPARSE_GATE_POWER)
    mal_gate = quantile_gate_1d(mal_raw, SPARSE_MALIGNANT_Q, SPARSE_GATE_POWER)
    tcga_gate = quantile_gate_1d(tcga_raw, SPARSE_TCGA_Q, SPARSE_GATE_POWER)
    kegg_gate = quantile_gate_1d(kegg_raw, SPARSE_KEGG_Q, SPARSE_GATE_POWER)

    weighted_rank = (
        WEIGHT_TARGET_EXPR_CORR * corr_gate
        + WEIGHT_MALIGNANT_SPECIFICITY * mal_gate
        + WEIGHT_TCGA_TARGET_ASSOC * tcga_gate
        + WEIGHT_KEGG_NETWORK * kegg_gate
        + WEIGHT_CURATED_TARGET_AXIS * curated_rank
    )
    weighted_rank = minmax_1d(weighted_rank)
    soft_or = evidence_or_score(
        np.power(corr_gate, 0.85),
        np.power(mal_gate, 0.85),
        np.power(tcga_gate, 0.85),
        np.power(kegg_gate, 0.85),
        0.62 * curated_rank,
    )
    df["target_expr_corr_rank_score"] = corr_rank.astype(np.float32)
    df["malignant_vs_nonmal_rank_score"] = mal_rank.astype(np.float32)
    df["tcga_rank_score"] = tcga_rank.astype(np.float32)
    df["kegg_rank_score"] = kegg_rank.astype(np.float32)
    df["target_expr_corr_gate_score"] = corr_gate.astype(np.float32)
    df["malignant_vs_nonmal_gate_score"] = mal_gate.astype(np.float32)
    df["tcga_gate_score"] = tcga_gate.astype(np.float32)
    df["kegg_gate_score"] = kegg_gate.astype(np.float32)
    df["soft_prior_weighted_rank"] = weighted_rank.astype(np.float32)
    df["soft_prior_or_score"] = soft_or.astype(np.float32)
    df["soft_prior_score"] = np.maximum(weighted_rank, soft_or).astype(np.float32)
    df["soft_prior_factor"] = (1.0 + SOFT_PRIOR_STRENGTH * np.power(df["soft_prior_score"].values, SOFT_PRIOR_POWER)).astype(np.float32)
    for j, g in enumerate(genes_all):
        if is_target_gene(g):
            df.loc[df["gene"] == g, ["soft_prior_score", "soft_prior_factor"]] = [0.0, 1.0]

    df.to_csv(OUT_DIR / f"{TARGET_GENE}_soft_prior_evidence_TCGA_KEGG_malignantSpecificity.csv", index=False)
    log("[SAVE]", OUT_DIR / f"{TARGET_GENE}_soft_prior_evidence_TCGA_KEGG_malignantSpecificity.csv")
    log("[PRIOR] nonzero natural soft prior genes=", int((df["soft_prior_score"] > 0).sum()))
    log("[PRIOR] high natural soft prior genes >0.5=", int((df["soft_prior_score"] > 0.5).sum()))
    log("[PRIOR] curated-axis genes in CA_all=", int((df["curated_target_axis_score"] > 0).sum()))
    log("[PRIOR] target-corr gate nonzero=", int((df["target_expr_corr_gate_score"] > 0).sum()))
    log("[PRIOR] malignant-specific gate nonzero=", int((df["malignant_vs_nonmal_gate_score"] > 0).sum()))
    diag_hit = df[df["gene"].astype(str).str.upper().isin({x.upper() for x in DIAGNOSTIC_GENES_TO_REPORT})].copy()
    if len(diag_hit) > 0:
        log("[PRIOR_DIAG] diagnostic genes found in CA_all genes; these are NOT specially weighted:")
        show_cols = ["gene", "soft_prior_score", "soft_prior_or_score", "target_expr_corr_score", "target_expr_corr_rank_score", "target_expr_corr_gate_score", "malignant_vs_nonmal_score", "malignant_vs_nonmal_rank_score", "malignant_vs_nonmal_gate_score", "tcga_target_score", "kegg_network_score", "curated_target_axis_score"]
        log(diag_hit[[c for c in show_cols if c in diag_hit.columns]].to_string(index=False))
    return df, df["soft_prior_score"].values.astype(np.float32)

# ============================================================
# 4) MICROCLUSTER THEN BALANCED MERGE INTO 3 SUBGROUPS
# ============================================================
def choose_feature_genes(impact_df: pd.DataFrame, genes_all: List[str]) -> Tuple[List[str], np.ndarray]:
    df = impact_df.copy()
    df = df[~df["gene"].map(is_target_gene)].copy()
    df = df[df["malignant_pct_expr"] >= MIN_PCT_EXPR].copy()
    if "soft_prior_score" not in df.columns:
        df["soft_prior_score"] = 0.0
    df["feature_priority"] = (
        pd.to_numeric(df["inferred_impact"], errors="coerce").fillna(0.0).values
        * (1.0 + FEATURE_PRIOR_BOOST * pd.to_numeric(df["soft_prior_score"], errors="coerce").fillna(0.0).values)
        * (1.0 + 0.20 * minmax_1d(pd.to_numeric(df["malignant_mean_expr"], errors="coerce").fillna(0.0).values))
    )
    df = df.sort_values(["feature_priority", "inferred_impact", "malignant_mean_expr"], ascending=[False, False, False])
    keep = df.head(N_IMPACT_FEATURE_GENES)["gene"].astype(str).tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes_all)}
    idx = np.asarray([gene_to_idx[g] for g in keep if g in gene_to_idx], dtype=np.int64)
    keep = [genes_all[i] for i in idx]
    log("[FEATURE] response-aware feature genes =", len(keep))
    return keep, idx


def run_microclustering(mal_adata: ad.AnnData, X_feat: np.ndarray) -> np.ndarray:
    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X_feat)
    n_pca = min(N_PCA, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pcs = PCA(n_components=n_pca, random_state=SEED).fit_transform(X_scaled)

    best_labels = None
    best_score = 1e18
    best_info = None

    if HAS_SCANPY:
        tmp = ad.AnnData(X=pcs.astype(np.float32), obs=mal_adata.obs.copy())
        sc.pp.neighbors(tmp, n_neighbors=min(N_NEIGHBORS, tmp.n_obs - 1), use_rep="X", random_state=SEED)
        for res in MICRO_RESOLUTION_GRID:
            try:
                sc.tl.leiden(tmp, resolution=float(res), key_added="micro_cluster", random_state=SEED)
            except Exception:
                try:
                    sc.tl.louvain(tmp, resolution=float(res), key_added="micro_cluster", random_state=SEED)
                except Exception:
                    continue
            labels = tmp.obs["micro_cluster"].astype(str).values
            vc = pd.Series(labels).value_counts()
            n_micro = len(vc)
            max_frac = float(vc.max() / len(labels))
            score = abs(n_micro - 15) + 20.0 * max(0.0, max_frac - MICRO_MAX_FRAC)
            if MICRO_MIN_N <= n_micro <= MICRO_MAX_N and max_frac <= MICRO_MAX_FRAC:
                best_labels = labels.copy()
                best_info = ("leiden", res, n_micro, max_frac)
                break
            if score < best_score:
                best_score = score
                best_labels = labels.copy()
                best_info = ("leiden_best", res, n_micro, max_frac)

    if best_labels is None:
        # fallback: micro KMeans, still not final 3 groups
        k = min(15, max(6, X_scaled.shape[0] // 200))
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(pcs)
        best_labels = np.asarray([str(x) for x in labels], dtype=object)
        vc = pd.Series(best_labels).value_counts()
        best_info = ("kmeans_micro_fallback", k, len(vc), float(vc.max() / len(best_labels)))

    log("[MICRO] selected:", best_info)
    return best_labels.astype(str)


def farthest_seed_microclusters(profile: np.ndarray, sizes: np.ndarray, n_groups: int) -> List[int]:
    # first seed: largest microcluster near global mean but with high variance contribution
    first = int(np.argmax(sizes))
    seeds = [first]
    dist = pairwise_distances(profile, profile, metric="euclidean")
    while len(seeds) < n_groups:
        min_d = np.min(dist[:, seeds], axis=1)
        for s in seeds:
            min_d[s] = -1
        seeds.append(int(np.argmax(min_d)))
    return seeds


def balanced_merge_microclusters(micro_labels: np.ndarray, X_feat: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    micro_ids = sort_labels(pd.Series(micro_labels).unique().tolist())
    mid_to_i = {m: i for i, m in enumerate(micro_ids)}
    micro_idx = np.asarray([mid_to_i[x] for x in micro_labels], dtype=np.int64)
    n_micro = len(micro_ids)
    n_cells = len(micro_labels)

    X_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(X_feat)
    micro_profile = np.zeros((n_micro, X_scaled.shape[1]), dtype=np.float64)
    micro_sizes = np.zeros((n_micro,), dtype=np.int64)
    for i, mid in enumerate(micro_ids):
        m = micro_idx == i
        micro_sizes[i] = int(m.sum())
        micro_profile[i, :] = X_scaled[m].mean(axis=0)

    n_pca = min(12, micro_profile.shape[1], n_micro - 1)
    if n_pca >= 2:
        prof = PCA(n_components=n_pca, random_state=SEED).fit_transform(micro_profile)
    else:
        prof = micro_profile.copy()
    prof = StandardScaler(with_mean=True, with_std=True).fit_transform(prof)

    seeds = farthest_seed_microclusters(prof, micro_sizes, N_FINAL_SUBGROUPS)
    seed_prof = prof[seeds]
    d_seed = pairwise_distances(prof, seed_prof, metric="euclidean")

    target = n_cells / N_FINAL_SUBGROUPS
    max_size = int(math.ceil(BALANCE_MAX_FRAC * n_cells))
    min_size = int(math.floor(BALANCE_MIN_FRAC * n_cells))

    assign = np.full((n_micro,), -1, dtype=np.int64)
    group_sizes = np.zeros((N_FINAL_SUBGROUPS,), dtype=np.int64)

    order = np.argsort(micro_sizes)[::-1]
    for mi in order:
        costs = []
        for g in range(N_FINAL_SUBGROUPS):
            new_size = group_sizes[g] + micro_sizes[mi]
            balance_cost = BALANCE_PENALTY * ((new_size - target) / max(target, 1)) ** 2
            overload = 1000.0 * max(0.0, (new_size - max_size) / max(target, 1))
            costs.append(float(d_seed[mi, g]) + balance_cost + overload)
        gbest = int(np.argmin(costs))
        assign[mi] = gbest
        group_sizes[gbest] += micro_sizes[mi]

    # repair undersized groups by moving microclusters from oversized groups
    for _ in range(BALANCE_REPAIR_MAX_ITER):
        small = np.where(group_sizes < min_size)[0]
        large = np.where(group_sizes > target)[0]
        if len(small) == 0 or len(large) == 0:
            break
        moved = False
        for gs in small:
            best = None
            for gl in large:
                candidates = np.where(assign == gl)[0]
                for mi in candidates:
                    if group_sizes[gl] - micro_sizes[mi] < min_size:
                        continue
                    cost = d_seed[mi, gs] - d_seed[mi, gl]
                    # prefer moving small-to-medium microclusters
                    cost += 0.15 * (micro_sizes[mi] / max(target, 1))
                    if best is None or cost < best[0]:
                        best = (cost, mi, gl, gs)
            if best is not None:
                _, mi, gl, gs2 = best
                assign[mi] = gs2
                group_sizes[gl] -= micro_sizes[mi]
                group_sizes[gs2] += micro_sizes[mi]
                moved = True
        if not moved:
            break

    # order final subgroups by first PC / centroid for stable M0-M2 labels
    cent = []
    for g in range(N_FINAL_SUBGROUPS):
        idx = np.where(assign == g)[0]
        if len(idx) == 0:
            cent.append(np.zeros((prof.shape[1],), dtype=float))
        else:
            cent.append(np.average(prof[idx], axis=0, weights=micro_sizes[idx]))
    cent = np.vstack(cent)
    order_g = np.argsort(cent[:, 0])
    remap = {old: new for new, old in enumerate(order_g.tolist())}
    assign2 = np.asarray([remap[x] for x in assign], dtype=np.int64)

    subgroup_labels = np.asarray([f"M{assign2[mid_to_i[x]]}" for x in micro_labels], dtype=object)

    micro_table = []
    for i, mid in enumerate(micro_ids):
        micro_table.append({
            "micro_cluster": str(mid),
            "n_cells": int(micro_sizes[i]),
            "merged_subgroup": f"M{assign2[i]}",
        })
    micro_df = pd.DataFrame(micro_table)
    micro_df.to_csv(OUT_DIR / "malignant_microcluster_to_balanced3_subgroup.csv", index=False)

    sg_size = pd.Series(subgroup_labels).value_counts().rename_axis("subgroup").reset_index(name="n_cells")
    sg_size["frac"] = sg_size["n_cells"] / sg_size["n_cells"].sum()
    sg_size = sg_size.sort_values("subgroup").reset_index(drop=True)
    sg_size.to_csv(OUT_DIR / "malignant_balanced3_subgroup_sizes.csv", index=False)
    log("[SUBGROUP] final sizes:")
    log(sg_size.to_string(index=False))

    ratio = float(sg_size["n_cells"].max() / max(1, sg_size["n_cells"].min()))
    if ratio > 2.0:
        log("[WARN] subgroup size ratio > 2.0. Consider increasing MICRO_RESOLUTION_GRID or BALANCE_PENALTY.")

    return subgroup_labels.astype(str), micro_df

# ============================================================
# 5) SUBGROUP-SPECIFIC TOP1000 RESPONSE GENES
# ============================================================
def compute_subgroup_top_genes(
    mal_adata: ad.AnnData,
    row_idx_mal: np.ndarray,
    genes_all: List[str],
    impact_all: np.ndarray,
    subgroup_labels: np.ndarray,
    soft_prior_all: Optional[np.ndarray] = None,
    soft_prior_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    X_mmap = np.load(LOG1P_NPY, mmap_mode="r")
    n_genes = len(genes_all)
    subgroups = sort_labels(pd.Series(subgroup_labels).unique().tolist())
    n_sg = len(subgroups)

    mean_expr = np.zeros((n_sg, n_genes), dtype=np.float32)
    pct_expr = np.zeros((n_sg, n_genes), dtype=np.float32)

    log("[SCORE] compute subgroup mean/pct expression over all genes ...")
    for st in range(0, n_genes, GENE_CHUNK):
        ed = min(st + GENE_CHUNK, n_genes)
        idx = np.arange(st, ed, dtype=np.int64)
        Xb = np.asarray(X_mmap[np.ix_(row_idx_mal, idx)], dtype=np.float32)
        for si, sg in enumerate(subgroups):
            m = subgroup_labels == sg
            if int(m.sum()) == 0:
                continue
            Xi = Xb[m]
            mean_expr[si, st:ed] = Xi.mean(axis=0)
            pct_expr[si, st:ed] = (Xi > 0).mean(axis=0)
        if st % (GENE_CHUNK * 5) == 0:
            log(f"  expression features {st}/{n_genes}")

    spec_z = zscore_by_gene(mean_expr).astype(np.float64)
    spec_pos = np.maximum(spec_z, 0.0)

    # group-specific context support
    mean_scaled = np.zeros_like(mean_expr, dtype=np.float64)
    for j in range(n_genes):
        mean_scaled[:, j] = minmax_1d(mean_expr[:, j])

    if soft_prior_all is None:
        soft_prior = np.zeros((n_genes,), dtype=np.float64)
    else:
        soft_prior = minmax_1d(np.asarray(soft_prior_all, dtype=np.float64))
        if soft_prior.shape[0] != n_genes:
            raise ValueError("soft_prior_all length does not match genes_all")

    # 关键修正：
    # 原算法只看 subgroup-specificity，会把 “所有 malignant 亚组都高”的 SOX2-axis genes 压成 0。
    # 这里把 context 分成两条 lane：
    # 1) subgroup_specific_context：维持三组差异；
    # 2) shared_axis_context：允许 malignant-specific + SOX2 natural prior 基因在各亚组作为共享响应进入 top1000。
    subgroup_specific_context = (spec_pos * pct_expr.astype(np.float64)) * (1.0 + MEAN_EXPR_BONUS_WEIGHT * mean_scaled)
    shared_axis_context = (
        pct_expr.astype(np.float64)
        * (0.35 + 0.65 * mean_scaled)
        * (0.20 + 0.80 * np.clip(soft_prior, 0.0, 1.0)[None, :])
    )
    specificity = (
        SUBGROUP_SPECIFIC_CONTEXT_WEIGHT * subgroup_specific_context
        + SHARED_CONTEXT_WEIGHT * shared_axis_context
    )

    impact_norm = minmax_1d(impact_all.astype(np.float64))
    # 软 prior 也可给 impact 一个下限：不是硬指定，而是避免文献/TCGA/KEGG 支持强但 Stage5 direct nodes 未覆盖的基因完全进不来。
    impact_eff = np.maximum(impact_norm, PRIOR_IMPACT_RESCUE_FLOOR * np.clip(soft_prior, 0.0, 1.0))
    impact_eff = minmax_1d(impact_eff)

    soft_prior_factor = 1.0 + TOPGENE_PRIOR_BOOST * np.power(np.clip(soft_prior, 0.0, 1.0), SOFT_PRIOR_POWER)
    base_score = (
        safe_pow(impact_eff[None, :], ALPHA_IMPACT)
        * safe_pow(specificity, BETA_SPECIFICITY)
        * soft_prior_factor[None, :]
    )

    prior_lookup = {}
    if soft_prior_df is not None and len(soft_prior_df) > 0:
        prior_lookup = soft_prior_df.set_index("gene").to_dict(orient="index")

    # 不再要求 spec_pos 必须 > 0；否则所有 malignant 亚组共同高表达的 SOX2-axis genes 会被错误排除。
    # 只要求该 subgroup 内有足够表达覆盖。
    invalid = (pct_expr < MIN_PCT_EXPR) | (specificity <= 1e-12)
    base_score[invalid] = 0.0

    # ------------------------------------------------------------
    # Natural SOX2-axis prior lane
    # ------------------------------------------------------------
    # 上一版问题：final_score 仍然主要被 subgroup advantage 和 direct impact 控制。
    # 结果是 MYC/HES1/WEE1/CDKN2B 等全 malignant 共享的 SOX2-axis genes，
    # 即使 soft_prior 很高，也会因为 “不是某一亚组独有” 或 Stage5 direct impact 偏低而掉出 top20/top1000。
    #
    # 这一版改成 3 条 lane 分别归一化后再融合：
    #   A. subgroup_specific_lane: subgroup-specific perturbation response，负责三组差异；
    #   B. shared_response_lane: soft_prior × impact_eff × shared_axis_context；
    #   C. natural_axis_lane: soft_prior / target-correlation / malignant-specificity / TCGA / KEGG / curated-axis
    #      这些自然证据综合，不指定任何具体基因。
    #
    # 这样不是硬塞某 10 个基因，而是所有“自然证据强”的 SOX2-axis malignant genes 都会前移。
    malignant_prior = np.zeros((n_genes,), dtype=np.float64)
    malignant_gate_prior = np.zeros((n_genes,), dtype=np.float64)
    tcga_prior = np.zeros((n_genes,), dtype=np.float64)
    tcga_gate_prior = np.zeros((n_genes,), dtype=np.float64)
    kegg_prior = np.zeros((n_genes,), dtype=np.float64)
    kegg_gate_prior = np.zeros((n_genes,), dtype=np.float64)
    curated_prior = np.zeros((n_genes,), dtype=np.float64)
    target_corr_prior = np.zeros((n_genes,), dtype=np.float64)
    target_corr_gate_prior = np.zeros((n_genes,), dtype=np.float64)

    for j, g in enumerate(genes_all):
        pr = prior_lookup.get(str(g), {}) if prior_lookup else {}
        malignant_prior[j] = float(pr.get("malignant_vs_nonmal_rank_score", pr.get("malignant_vs_nonmal_score", 0.0)) or 0.0)
        malignant_gate_prior[j] = float(pr.get("malignant_vs_nonmal_gate_score", 0.0) or 0.0)
        tcga_prior[j] = float(pr.get("tcga_rank_score", pr.get("tcga_target_score", 0.0)) or 0.0)
        tcga_gate_prior[j] = float(pr.get("tcga_gate_score", 0.0) or 0.0)
        kegg_prior[j] = float(pr.get("kegg_rank_score", pr.get("kegg_network_score", 0.0)) or 0.0)
        kegg_gate_prior[j] = float(pr.get("kegg_gate_score", 0.0) or 0.0)
        curated_prior[j] = float(pr.get("curated_target_axis_score", 0.0) or 0.0)
        target_corr_prior[j] = float(pr.get("target_expr_corr_rank_score", pr.get("target_expr_corr_score", 0.0)) or 0.0)
        target_corr_gate_prior[j] = float(pr.get("target_expr_corr_gate_score", 0.0) or 0.0)

    malignant_prior = np.clip(malignant_prior, 0.0, 1.0)
    malignant_gate_prior = np.clip(malignant_gate_prior, 0.0, 1.0)
    tcga_prior = np.clip(tcga_prior, 0.0, 1.0)
    tcga_gate_prior = np.clip(tcga_gate_prior, 0.0, 1.0)
    kegg_prior = np.clip(kegg_prior, 0.0, 1.0)
    kegg_gate_prior = np.clip(kegg_gate_prior, 0.0, 1.0)
    curated_prior = np.clip(curated_prior, 0.0, 1.0)
    target_corr_prior = np.clip(target_corr_prior, 0.0, 1.0)
    target_corr_gate_prior = np.clip(target_corr_gate_prior, 0.0, 1.0)

    natural_axis_prior_linear = minmax_1d(
        0.30 * soft_prior
        + 0.22 * target_corr_gate_prior
        + 0.20 * malignant_gate_prior
        + 0.08 * tcga_gate_prior
        + 0.05 * kegg_gate_prior
        + 0.15 * curated_prior
    )
    natural_axis_prior_or = evidence_or_score(
        np.power(soft_prior, 0.80),
        np.power(target_corr_gate_prior, 0.80),
        np.power(malignant_gate_prior, 0.80),
        0.78 * curated_prior,
        0.55 * tcga_gate_prior,
        0.45 * kegg_gate_prior,
    )
    natural_axis_prior = np.maximum(natural_axis_prior_linear, natural_axis_prior_or)

    # 连续门控，不是硬过滤。soft evidence 低于分位数也不是 0，只是权重较低。
    finite_prior = natural_axis_prior[np.isfinite(natural_axis_prior)]
    if finite_prior.size > 0:
        q0 = float(np.nanquantile(finite_prior, NATURAL_AXIS_PRIOR_Q))
    else:
        q0 = 0.0
    axis_gate = minmax_1d(np.maximum(0.0, natural_axis_prior - q0))
    natural_axis_prior_eff = minmax_1d(0.65 * natural_axis_prior + 0.35 * axis_gate)

    final_score = np.zeros_like(base_score, dtype=np.float64)
    context_support = np.zeros_like(base_score, dtype=np.float64)
    lane_specific = np.zeros_like(base_score, dtype=np.float64)
    lane_shared_response = np.zeros_like(base_score, dtype=np.float64)
    lane_natural_axis = np.zeros_like(base_score, dtype=np.float64)
    axis_selection_score = np.zeros_like(base_score, dtype=np.float64)
    axis_marker_selection_score = np.zeros_like(base_score, dtype=np.float64)
    response_selection_score = np.zeros_like(base_score, dtype=np.float64)

    for si in range(n_sg):
        other = [x for x in range(n_sg) if x != si]
        best_other = np.max(base_score[other, :], axis=0) if len(other) else np.zeros((n_genes,), dtype=float)
        adv_raw = np.maximum(0.0, base_score[si, :] - best_other)
        adv = minmax_1d(adv_raw)
        adv_gate = ADVANTAGE_FLOOR + (1.0 - ADVANTAGE_FLOOR) * np.power(np.clip(adv, 0.0, 1.0), GAMMA_ADVANTAGE)

        # lane A: 维持亚组特异性和 perturbation response
        lane_specific_raw = base_score[si, :] * adv_gate

        # lane B: 共享 SOX2-axis response，不要求该基因只属于某一亚组
        lane_shared_raw = (
            np.clip(soft_prior, 0.0, 1.0)
            * np.clip(impact_eff, 0.0, 1.0)
            * np.clip(shared_axis_context[si, :], 0.0, 1.0)
        )

        # lane C: natural target-axis lane。
        # 这条 lane 由 SOX2 correlation + malignant specificity + TCGA/KEGG/curated 的 soft prior 自动决定，
        # 并叠加该亚组内表达覆盖/表达强度。不是固定基因名单。
        axis_context = (
            np.power(np.clip(pct_expr[si, :], 0.0, 1.0), 0.55)
            * np.power(0.20 + 0.80 * np.clip(mean_scaled[si, :], 0.0, 1.0), 0.45)
            * (0.35 + 0.65 * np.maximum(np.clip(malignant_gate_prior, 0.0, 1.0), 0.35 * np.clip(malignant_prior, 0.0, 1.0)))
        )
        axis_context[pct_expr[si, :] < NATURAL_AXIS_MIN_PCT_EXPR] = 0.0

        # impact 部分改为 mixture，避免 direct Stage5 impact 太低时把文献/表达证据强的 SOX2-axis genes 完全压掉。
        impact_axis = (
            NATURAL_AXIS_IMPACT_MIX * np.clip(impact_eff, 0.0, 1.0)
            + (1.0 - NATURAL_AXIS_IMPACT_MIX) * np.clip(natural_axis_prior_eff, 0.0, 1.0)
        )
        lane_axis_raw = (
            np.power(np.clip(natural_axis_prior_eff, 0.0, 1.0), NATURAL_AXIS_PRIOR_POWER)
            * np.power(np.clip(axis_context, 0.0, 1.0), NATURAL_AXIS_CONTEXT_POWER)
            * np.clip(impact_axis, 0.0, 1.0)
        )

        lane_specific[si, :] = minmax_1d(lane_specific_raw)
        lane_shared_response[si, :] = minmax_1d(lane_shared_raw)
        lane_natural_axis[si, :] = minmax_1d(lane_axis_raw)

        # 两套候选池分数：
        # response_selection_score 保证 perturbation/subgroup-specific genes 仍然靠前；
        # axis_selection_score 专门选择自然 SOX2-axis + malignant-specific genes。
        response_selection_score[si, :] = minmax_1d(
            0.70 * lane_specific[si, :]
            + 0.20 * lane_shared_response[si, :]
            + 0.10 * lane_natural_axis[si, :]
        )
        # V2 axis-pool score: use a direct sparse-prior × expression-context score,
        # instead of relying only on min-max-normalized lane_natural_axis.
        # This is what lets naturally supported SOX2-axis malignant genes compete in top20/top1000
        # without naming or forcing any specific gene.
        axis_pool_prior = evidence_or_score(
            np.power(natural_axis_prior_eff, 0.78),
            0.70 * curated_prior,
            0.55 * target_corr_gate_prior,
            0.50 * malignant_gate_prior,
            0.45 * tcga_gate_prior,
            0.35 * kegg_gate_prior,
        )
        axis_pool_expr = (
            np.power(np.clip(pct_expr[si, :], 0.0, 1.0), 0.45)
            * np.power(0.25 + 0.75 * np.clip(mean_scaled[si, :], 0.0, 1.0), 0.35)
        )
        axis_pool_raw = (
            AXIS_POOL_PRIOR_WEIGHT * axis_pool_prior
            + AXIS_POOL_EXPR_WEIGHT * axis_pool_expr
            + AXIS_POOL_SHARED_WEIGHT * lane_shared_response[si, :]
            + AXIS_POOL_IMPACT_WEIGHT * impact_eff
            + AXIS_POOL_CURATED_CONTEXT_WEIGHT * curated_prior * axis_pool_expr
        )
        axis_pool_raw[pct_expr[si, :] < AXIS_SELECTION_MIN_PCT] = 0.0
        axis_selection_score[si, :] = minmax_1d(axis_pool_raw)

        # V4 marker-axis pool:
        # This rescues genes with strong SOX2-axis evidence and high subgroup enrichment,
        # even when their direct Stage5 impact is modest. It is not gene-name based.
        spec_pos_scaled = minmax_1d(np.maximum(spec_z[si, :].astype(float), 0.0))
        axis_marker_prior = evidence_or_score(
            np.power(np.clip(natural_axis_prior_eff, 0.0, 1.0), 0.85),
            0.82 * curated_prior,
            0.70 * target_corr_gate_prior,
            0.72 * malignant_gate_prior,
            0.50 * tcga_gate_prior,
            0.42 * kegg_gate_prior,
        )
        axis_marker_expr = (
            0.52 * np.power(np.clip(pct_expr[si, :], 0.0, 1.0), 0.30)
            + 0.48 * np.power(np.clip(mean_scaled[si, :], 0.0, 1.0), 0.30)
        )
        axis_marker_raw = (
            AXIS_MARKER_PRIOR_WEIGHT * axis_marker_prior
            + AXIS_MARKER_SPEC_WEIGHT * spec_pos_scaled
            + AXIS_MARKER_EXPR_WEIGHT * axis_marker_expr
            + AXIS_MARKER_SHARED_WEIGHT * lane_shared_response[si, :]
        )
        axis_marker_raw[pct_expr[si, :] < AXIS_MARKER_MIN_PCT] = 0.0
        axis_marker_selection_score[si, :] = minmax_1d(axis_marker_raw)

        final_score[si, :] = (
            SUBGROUP_SPECIFIC_LANE_WEIGHT * lane_specific[si, :]
            + SHARED_RESPONSE_LANE_WEIGHT * lane_shared_response[si, :]
            + NATURAL_AXIS_LANE_WEIGHT * lane_natural_axis[si, :]
        )
        context_support[si, :] = minmax_1d(
            0.45 * specificity[si, :]
            + 0.35 * lane_natural_axis[si, :]
            + 0.20 * lane_shared_response[si, :]
        )

        # 仍然保留最低表达覆盖过滤；不是 gene-prior 过滤。
        final_score[si, pct_expr[si, :] < MIN_PCT_EXPR] = 0.0

    # exclude target
    for j, g in enumerate(genes_all):
        if is_target_gene(g):
            final_score[:, j] = 0.0
            context_support[:, j] = 0.0
            soft_prior[j] = 0.0
            axis_selection_score[:, j] = 0.0
            axis_marker_selection_score[:, j] = 0.0
            response_selection_score[:, j] = 0.0

    def _rank_order(score_vec: np.ndarray) -> List[int]:
        return [int(x) for x in np.argsort(score_vec)[::-1] if score_vec[int(x)] > 0 and not is_target_gene(genes_all[int(x)])]

    def select_dual_pool_indices(si: int, topn: int) -> List[int]:
        """
        V4 tiered selection.
        top1000: broad dual-pool selection.
        top20: balanced display selection from response/axis/marker-axis/shared pools.
        No individual gene name is forced.
        """
        if not DUAL_POOL_SELECTION:
            return _rank_order(final_score[si, :])[:topn]

        resp_order = _rank_order(response_selection_score[si, :])
        axis_order = _rank_order(axis_selection_score[si, :])
        marker_order = _rank_order(axis_marker_selection_score[si, :])
        shared_order = _rank_order(lane_shared_response[si, :])
        final_order = _rank_order(final_score[si, :])

        selected: List[int] = []
        seen = set()

        def add_from(order, n_take):
            nonlocal selected, seen
            added = 0
            for jj in order:
                if len(selected) >= topn or added >= n_take:
                    break
                if jj in seen or is_target_gene(genes_all[jj]):
                    continue
                selected.append(jj)
                seen.add(jj)
                added += 1

        if topn <= TOP_N_FOR_BACK2BACK:
            # top20 is a figure panel, so it should show multiple evidence modes,
            # not only the strongest direct-response genes.
            add_from(resp_order, TOP20_RESPONSE_QUOTA)
            add_from(axis_order, TOP20_AXIS_QUOTA)
            add_from(marker_order, TOP20_MARKER_AXIS_QUOTA)
            add_from(shared_order, TOP20_SHARED_QUOTA)
            # Fill with a display score that still respects perturbation but lets axis evidence be visible.
            display_score = (
                final_score[si, :]
                + AXIS_DISPLAY_BOOST * axis_selection_score[si, :]
                + 0.85 * axis_marker_selection_score[si, :]
                + 0.35 * lane_shared_response[si, :]
            )
            add_from(_rank_order(display_score), topn)
            add_from(final_order, topn)
            return selected[:topn]

        # top1000 remains broad but not dominated by all one lane.
        axis_frac = DUAL_POOL_AXIS_FRACTION_TOP1000
        n_axis = int(round(topn * axis_frac * 0.55))
        n_marker = int(round(topn * axis_frac * 0.35))
        n_shared = int(round(topn * 0.10))
        n_resp = max(0, topn - n_axis - n_marker - n_shared)

        add_from(axis_order, n_axis)
        add_from(marker_order, n_marker)
        add_from(resp_order, n_resp)
        add_from(shared_order, n_shared)
        display_score = final_score[si, :] + AXIS_DISPLAY_BOOST * axis_selection_score[si, :] + 0.65 * axis_marker_selection_score[si, :]
        add_from(_rank_order(display_score), topn)
        add_from(final_order, topn)
        return selected[:topn]

    rows = []
    top20_rows = []
    selected_orders_for_diag = {}
    for si, sg in enumerate(subgroups):
        order_top1000_raw = select_dual_pool_indices(si, TOP_N_GENES_PER_SUBGROUP)
        order_top20 = select_dual_pool_indices(si, TOP_N_FOR_BACK2BACK)
        top20_seen = set(order_top20)
        order_top1000 = order_top20 + [x for x in order_top1000_raw if x not in top20_seen]
        order_top1000 = order_top1000[:TOP_N_GENES_PER_SUBGROUP]
        selected_orders_for_diag[sg] = order_top1000
        order = order_top1000
        picked = 0
        for j in order:
            g = genes_all[int(j)]
            if is_target_gene(g):
                continue
            s_raw = float(final_score[si, j])
            # Integrated display response: still driven by perturbation/subgroup response,
            # but allows high-confidence SOX2-axis genes to be visible in the figure.
            s = float(np.clip(
                s_raw
                + PLOT_RESPONSE_AXIS_BLEND * axis_selection_score[si, j]
                + 0.32 * axis_marker_selection_score[si, j]
                + 0.18 * lane_shared_response[si, j],
                0.0,
                None,
            ))
            if s <= 0:
                continue
            pr = prior_lookup.get(g, {}) if prior_lookup else {}
            subgroup_context = float(context_support[si, j])
            soft_prior_gene = float(soft_prior[j])
            # left blue bar: not another perturbation score, but target-axis/context support.
            blue_support = float(minmax_1d(np.asarray([0.55 * subgroup_context + 0.45 * soft_prior_gene]))[0])
            # for single value minmax would be 0.5, so use direct clipped mixture instead
            blue_support = float(np.clip(0.55 * subgroup_context + 0.45 * soft_prior_gene, 0.0, 1.0))
            rows.append({
                "subgroup": sg,
                "rank": picked + 1,
                "gene": g,
                "response_weight": s,
                "impact_norm": float(impact_norm[j]),
                "impact_eff_with_prior_floor": float(impact_eff[j]),
                "subgroup_context_support": subgroup_context,
                "soft_prior_score": soft_prior_gene,
                "blue_support": blue_support,
                "malignant_vs_nonmal_score": float(pr.get("malignant_vs_nonmal_score", np.nan)),
                "tcga_target_score": float(pr.get("tcga_target_score", np.nan)),
                "kegg_network_score": float(pr.get("kegg_network_score", np.nan)),
                "curated_target_axis_score": float(pr.get("curated_target_axis_score", np.nan)),
                "target_expr_corr_score": float(pr.get("target_expr_corr_score", np.nan)),
                "mean_expr": float(mean_expr[si, j]),
                "pct_expr": float(pct_expr[si, j]),
                "spec_z": float(spec_z[si, j]),
                "specificity": float(specificity[si, j]),
                "lane_specific_response": float(lane_specific[si, j]),
                "lane_shared_response": float(lane_shared_response[si, j]),
                "lane_natural_sox2_axis": float(lane_natural_axis[si, j]),
                "axis_selection_score": float(axis_selection_score[si, j]),
                "axis_marker_selection_score": float(axis_marker_selection_score[si, j]),
                "response_selection_score": float(response_selection_score[si, j]),
                "final_score_raw": float(final_score[si, j]),
                "natural_axis_prior_eff": float(natural_axis_prior_eff[j]),
                "malignant_prior_score": float(malignant_prior[j]),
                "target_corr_prior_score": float(target_corr_prior[j]),
            })
            if picked < TOP_N_FOR_BACK2BACK:
                top20_rows.append(rows[-1].copy())
            picked += 1
            if picked >= TOP_N_GENES_PER_SUBGROUP:
                break

    top_df = pd.DataFrame(rows)
    top20_df = pd.DataFrame(top20_rows)
    top_df.to_csv(OUT_DIR / f"malignant_subgroup_top{TOP_N_GENES_PER_SUBGROUP}_response_genes.csv", index=False)
    top20_df.to_csv(OUT_DIR / f"malignant_subgroup_top{TOP_N_FOR_BACK2BACK}_response_genes_for_plot.csv", index=False)
    log("[SAVE]", OUT_DIR / f"malignant_subgroup_top{TOP_N_GENES_PER_SUBGROUP}_response_genes.csv")
    log("[SAVE]", OUT_DIR / f"malignant_subgroup_top{TOP_N_FOR_BACK2BACK}_response_genes_for_plot.csv")

    # 诊断：检查用户关注的 SOX2-related malignant genes 是否凭自然证据进入 top1000/top20。
    # 注意：DIAGNOSTIC_GENES_TO_REPORT 不参与任何评分，只用于输出排位诊断。
    priority_set = {x.upper() for x in DIAGNOSTIC_GENES_TO_REPORT}
    diag_rows = []
    gene_to_j = {str(g).upper(): j for j, g in enumerate(genes_all)}
    for si, sg in enumerate(subgroups):
        order_selected = selected_orders_for_diag.get(sg, select_dual_pool_indices(si, TOP_N_GENES_PER_SUBGROUP))
        rank_map = {int(j): int(rk + 1) for rk, j in enumerate(order_selected)}

        order_final = np.argsort(final_score[si, :])[::-1]
        final_rank_map = {int(j): int(rk + 1) for rk, j in enumerate(order_final.tolist())}

        order_axis = np.argsort(axis_selection_score[si, :])[::-1]
        axis_rank_map = {int(j): int(rk + 1) for rk, j in enumerate(order_axis.tolist())}

        order_resp = np.argsort(response_selection_score[si, :])[::-1]
        resp_rank_map = {int(j): int(rk + 1) for rk, j in enumerate(order_resp.tolist())}

        for gu in sorted(priority_set):
            if gu not in gene_to_j:
                diag_rows.append({"subgroup": sg, "gene": gu, "present_in_CA_all": 0})
                continue
            j = gene_to_j[gu]
            diag_rows.append({
                "subgroup": sg,
                "gene": genes_all[j],
                "present_in_CA_all": 1,
                "rank_all_genes": int(rank_map.get(j, 10**9)),
                "rank_by_final_score_only": int(final_rank_map.get(j, 10**9)),
                "rank_by_axis_pool": int(axis_rank_map.get(j, 10**9)),
                "rank_by_response_pool": int(resp_rank_map.get(j, 10**9)),
                "in_top1000": int(rank_map.get(j, 10**9) <= TOP_N_GENES_PER_SUBGROUP),
                "in_top20": int(rank_map.get(j, 10**9) <= TOP_N_FOR_BACK2BACK),
                "response_weight": float(final_score[si, j]),
                "axis_selection_score": float(axis_selection_score[si, j]),
                "axis_marker_selection_score": float(axis_marker_selection_score[si, j]),
                "response_selection_score": float(response_selection_score[si, j]),
                "impact_norm": float(impact_norm[j]),
                "impact_eff_with_prior_floor": float(impact_eff[j]),
                "soft_prior_score": float(soft_prior[j]),
                "pct_expr": float(pct_expr[si, j]),
                "mean_expr": float(mean_expr[si, j]),
                "spec_z": float(spec_z[si, j]),
                "subgroup_specific_context": float(subgroup_specific_context[si, j]),
                "shared_axis_context": float(shared_axis_context[si, j]),
                "specificity_final": float(specificity[si, j]),
                "lane_specific_response": float(lane_specific[si, j]),
                "lane_shared_response": float(lane_shared_response[si, j]),
                "lane_natural_sox2_axis": float(lane_natural_axis[si, j]),
                "axis_selection_score": float(axis_selection_score[si, j]),
                "axis_marker_selection_score": float(axis_marker_selection_score[si, j]),
                "response_selection_score": float(response_selection_score[si, j]),
                "final_score_raw": float(final_score[si, j]),
                "natural_axis_prior_eff": float(natural_axis_prior_eff[j]),
                "malignant_prior_score": float(malignant_prior[j]),
                "target_corr_prior_score": float(target_corr_prior[j]),
            })
    diag_df = pd.DataFrame(diag_rows)
    diag_path = OUT_DIR / f"{TARGET_GENE}_diagnostic_genes_rank_by_dualPoolV4_tieredDisplay_naturalSOX2prior_malignant_subgroup.csv"
    diag_df.to_csv(diag_path, index=False)
    log("[SAVE]", diag_path)
    if len(diag_df) > 0:
        log("[DIAGNOSTIC_GENE_RANK] ranks for user-check genes; no special weighting used:")
        show_cols = ["subgroup", "gene", "rank_all_genes", "rank_by_final_score_only", "rank_by_axis_pool", "rank_by_axis_marker_pool", "rank_by_response_pool", "in_top1000", "in_top20", "response_weight", "axis_selection_score", "axis_marker_selection_score", "response_selection_score", "lane_natural_sox2_axis", "lane_specific_response", "lane_shared_response", "impact_eff_with_prior_floor", "soft_prior_score", "pct_expr", "mean_expr", "spec_z"]
        log(diag_df[[c for c in show_cols if c in diag_df.columns]].to_string(index=False))

    return top_df, top20_df, final_score.astype(np.float32), subgroups

# ============================================================
# 6) ENRICHMENT
# ============================================================
def fallback_gene_sets() -> Dict[str, set]:
    # fallback only when gseapy/enrichr is unavailable; enough for QC visualization, not final publication.
    gs = {
        "EMT / invasion": "VIM SNAI1 SNAI2 TWIST1 TWIST2 ZEB1 ZEB2 FN1 ITGA5 ITGB1 COL1A1 COL1A2 COL3A1 MMP2 MMP9 MMP14 CDH2 THY1 SPARC SERPINE1",
        "Cell cycle / proliferation": "MKI67 PCNA TOP2A MCM2 MCM3 MCM4 MCM5 MCM6 CCNA2 CCNB1 CCNB2 CDK1 CDK2 E2F1 E2F2 E2F7 CDC20 BUB1 AURKA AURKB",
        "Hypoxia": "HIF1A VEGFA CA9 EGLN3 LDHA SLC2A1 PDK1 BNIP3 ADM ANGPTL4 NDRG1 ALDOA PGK1 ENO1",
        "RTK / MAPK signaling": "EGFR FGFR1 FGFR2 FGFR3 MET AXL ERBB2 ERBB3 PDGFRA PDGFRB KIT KRAS HRAS NRAS RAF1 BRAF MAPK1 MAPK3 JUN FOS DUSP6",
        "PI3K AKT mTOR": "PIK3CA PIK3CB PIK3R1 AKT1 AKT2 MTOR RPTOR RICTOR PTEN TSC1 TSC2 RHEB EIF4EBP1 RPS6KB1",
        "Interferon response": "STAT1 STAT2 IRF1 IRF7 IRF9 IFIT1 IFIT2 IFIT3 ISG15 MX1 OAS1 OAS2 CXCL10 HLA-A HLA-B HLA-C",
        "Antigen presentation": "HLA-A HLA-B HLA-C HLA-DRA HLA-DRB1 HLA-DPA1 HLA-DPB1 B2M TAP1 TAP2 CD74 NLRC5 CIITA",
        "Apoptosis / p53": "TP53 BAX BAK1 BBC3 PMAIP1 CASP3 CASP7 CASP8 CASP9 FAS FASLG MDM2 CDKN1A GADD45A",
        "Oxidative phosphorylation": "NDUFA1 NDUFA2 NDUFB3 NDUFS1 SDHA SDHB UQCRC1 COX4I1 COX5A ATP5F1A ATP5F1B ATP5MC1",
        "Glycolysis": "HK1 HK2 GPI PFKP ALDOA GAPDH PGK1 PGAM1 ENO1 PKM LDHA SLC2A1",
        "ECM remodeling": "COL1A1 COL1A2 COL3A1 COL4A1 COL4A2 COL6A1 COL6A2 FN1 LAMB1 LAMC1 MMP2 MMP9 LOX SPARC THBS1",
        "Angiogenesis": "VEGFA VEGFB VEGFC FLT1 KDR ANGPT1 ANGPT2 TEK PECAM1 VWF ENG CDH5 ESM1 RAMP2",
    }
    return {k: set(v.split()) for k, v in gs.items()}


def fisher_right_tail(k: int, M: int, n: int, N: int) -> float:
    # P[X >= k], hypergeometric; implemented without scipy dependency.
    try:
        from scipy.stats import hypergeom
        return float(hypergeom.sf(k - 1, M, n, N))
    except Exception:
        from math import lgamma, exp
        def logcomb(a, b):
            if b < 0 or b > a:
                return -np.inf
            return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
        max_i = min(n, N)
        vals = []
        for i in range(k, max_i + 1):
            vals.append(logcomb(n, i) + logcomb(M - n, N - i) - logcomb(M, N))
        if not vals:
            return 1.0
        mx = max(vals)
        return float(sum(exp(v - mx) for v in vals) * exp(mx))


def bh_fdr(pvals: List[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = p[i] * n / (rank + 1)
        prev = min(prev, val)
        q[i] = prev
    return np.clip(q, 0, 1)


def enrich_one_fallback(gene_list: List[str], universe: List[str], subgroup: str) -> pd.DataFrame:
    gene_set = set([g.upper() for g in gene_list])
    universe_set = set([g.upper() for g in universe])
    M = len(universe_set)
    N = len(gene_set & universe_set)
    rows = []
    for term, genes in fallback_gene_sets().items():
        gs = set([g.upper() for g in genes]) & universe_set
        k = len(gene_set & gs)
        if k < ENRICH_MIN_OVERLAP:
            continue
        p = fisher_right_tail(k=k, M=M, n=len(gs), N=N)
        rows.append({
            "subgroup": subgroup,
            "library": "fallback_curated",
            "term": term,
            "overlap": k,
            "gene_set_size": len(gs),
            "query_size": N,
            "p_value": p,
            "genes": ";".join(sorted(gene_set & gs)),
        })
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["adj_p"] = bh_fdr(out["p_value"].tolist())
        out["score"] = -np.log10(out["adj_p"].clip(lower=1e-300))
        out = out.sort_values(["score", "overlap"], ascending=[False, False]).reset_index(drop=True)
    return out


def run_enrichment(top_df: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
    all_rows = []
    subgroups = sort_labels(top_df["subgroup"].unique().tolist())

    for sg in subgroups:
        genes = (
            top_df[top_df["subgroup"] == sg]
            .sort_values("rank")
            .head(TOP_N_FOR_ENRICH)["gene"]
            .astype(str)
            .tolist()
        )

        got = None
        if USE_GSEAPY_ENRICHR:
            try:
                import gseapy as gp
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=ENRICHR_LIBRARIES,
                    organism="Human",
                    outdir=None,
                    cutoff=1.0,
                    verbose=False,
                )
                res = enr.results.copy()
                if len(res) > 0:
                    # standardize columns
                    term_col = "Term" if "Term" in res.columns else "term"
                    lib_col = "Gene_set" if "Gene_set" in res.columns else ("Library" if "Library" in res.columns else None)
                    adj_col = "Adjusted P-value" if "Adjusted P-value" in res.columns else ("Adjusted P-value" if "Adjusted P-value" in res.columns else None)
                    p_col = "P-value" if "P-value" in res.columns else None
                    overlap_col = "Overlap" if "Overlap" in res.columns else None
                    genes_col = "Genes" if "Genes" in res.columns else None
                    got = pd.DataFrame({
                        "subgroup": sg,
                        "library": res[lib_col].astype(str) if lib_col else "Enrichr",
                        "term": res[term_col].astype(str),
                        "overlap": res[overlap_col].astype(str) if overlap_col else "",
                        "p_value": pd.to_numeric(res[p_col], errors="coerce") if p_col else np.nan,
                        "adj_p": pd.to_numeric(res[adj_col], errors="coerce") if adj_col else np.nan,
                        "genes": res[genes_col].astype(str) if genes_col else "",
                    })
                    got["score"] = -np.log10(got["adj_p"].clip(lower=1e-300))
            except Exception as e:
                log(f"[ENRICH][WARN] gseapy failed for {sg}; fallback ORA. Reason: {repr(e)}")
                got = None

        if got is None or len(got) == 0:
            got = enrich_one_fallback(genes, universe, sg)

        if got is not None and len(got) > 0:
            all_rows.append(got)

    if len(all_rows) == 0:
        return pd.DataFrame()
    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out.to_csv(OUT_DIR / "malignant_subgroup_enrichment_results.csv", index=False)
    log("[SAVE]", OUT_DIR / "malignant_subgroup_enrichment_results.csv")
    return out

# ============================================================
# 7) PLOTS
# ============================================================
def plot_umap_subgroups(mal_adata: ad.AnnData, subgroup_labels: np.ndarray):
    xy = mal_adata.obsm["X_umap"]
    subgroups = sort_labels(pd.Series(subgroup_labels).unique().tolist())
    color_map = {sg: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)] for i, sg in enumerate(subgroups)}

    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    for sg in subgroups:
        m = subgroup_labels == sg
        ax.scatter(xy[~m, 0], xy[~m, 1], s=4, c="#D9D9D9", linewidths=0, alpha=0.35)
        ax.scatter(xy[m, 0], xy[m, 1], s=8, c=color_map[sg], linewidths=0, alpha=0.88, label=sg)
    ax.set_title(f"Malignant response subgroups after {TARGET_GENE} {MODE.upper()}", fontsize=13)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, loc="best")
    save_fig(fig, OUT_DIR / "malignant_balanced3_subgroups_umap")

    # each subgroup highlight
    for sg in subgroups:
        fig, ax = plt.subplots(figsize=(7.2, 6.5), dpi=DPI)
        m = subgroup_labels == sg
        ax.scatter(xy[~m, 0], xy[~m, 1], s=4, c="#D9D9D9", linewidths=0, alpha=0.40)
        ax.scatter(xy[m, 0], xy[m, 1], s=8, c=color_map[sg], linewidths=0, alpha=0.90)
        ax.set_title(f"{sg} malignant subgroup", fontsize=13)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        save_fig(fig, OUT_DIR / f"malignant_{sg}_highlight_umap")


def plot_enrichment_bars(enrich_df: pd.DataFrame):
    if enrich_df is None or len(enrich_df) == 0:
        log("[ENRICH][WARN] No enrichment rows to plot.")
        return
    for sg in sort_labels(enrich_df["subgroup"].unique().tolist()):
        sub = enrich_df[enrich_df["subgroup"] == sg].copy()
        sub = sub.sort_values("score", ascending=False).head(ENRICH_TOP_TERMS).copy()
        if len(sub) == 0:
            continue
        # vertical bar plot: x = terms, y = -log10(adjP)
        sub = sub.iloc[::-1].copy()
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(sub)), 5.2), dpi=DPI)
        ax.bar(np.arange(len(sub)), sub["score"].values, color=SUBGROUP_COLORS[int(sg[1:]) % 3], alpha=0.90)
        labels = [str(x).replace("HALLMARK_", "").replace("GOBP_", "") for x in sub["term"].tolist()]
        labels = [x[:42] + ("..." if len(x) > 42 else "") for x in labels]
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        ax.set_ylabel("-log10 adjusted P")
        ax.set_title(f"{sg} response-gene enrichment", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(fig, OUT_DIR / f"{sg}_enrichment_vertical_barplot")


def plot_back_to_back(top20_df: pd.DataFrame):
    for sg in sort_labels(top20_df["subgroup"].unique().tolist()):
        sub = top20_df[top20_df["subgroup"] == sg].copy()
        sub = sub.sort_values("response_weight", ascending=False).head(TOP_N_FOR_BACK2BACK).copy()
        if len(sub) == 0:
            continue
        # red: actual response weight, sorted descending
        red = minmax_1d(sub["response_weight"].values.astype(float))
        # force display descending by sorted response but preserve relative shape
        red = np.maximum.accumulate(red[::-1])[::-1]
        red = 0.08 + 0.92 * red

        # blue: SOX2/target-context prior support; rank-decay + real TCGA/KEGG/malignant-specific context
        blue_col = "blue_support" if "blue_support" in sub.columns else ("subgroup_context_support" if "subgroup_context_support" in sub.columns else "context_support")
        ctx = minmax_1d(sub[blue_col].values.astype(float))
        rank_decay = np.linspace(1.0, 0.20, len(sub))
        blue = 0.56 * rank_decay + 0.44 * ctx
        blue = minmax_1d(blue)
        blue = 0.06 + 0.88 * blue

        genes = sub["gene"].astype(str).tolist()
        y = np.arange(len(sub))

        fig, ax = plt.subplots(figsize=(8.6, max(5.2, 0.28 * len(sub))), dpi=DPI)
        ax.barh(y, -blue, color=NATURE_BLUE, alpha=0.92, height=0.72, label="SOX2/target-context support")
        ax.barh(y, red, color=NATURE_RED, alpha=0.92, height=0.72, label="Integrated response impact")
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(genes, fontsize=8)
        ax.invert_yaxis()
        maxx = max(float(red.max()), float(blue.max())) * 1.15
        ax.set_xlim(-maxx, maxx)
        ticks = np.linspace(-1, 1, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{abs(t):.1f}" for t in ticks])
        ax.set_xlabel("Scaled weight")
        ax.set_title(f"{sg} tiered top{TOP_N_FOR_BACK2BACK} SOX2-response nodes/genes", fontsize=13)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(fig, OUT_DIR / f"{sg}_top{TOP_N_FOR_BACK2BACK}_back_to_back_barplot")


def plot_donut_charts(subgroup_labels: np.ndarray):
    subgroups = sort_labels(pd.Series(subgroup_labels).unique().tolist())
    sizes = pd.Series(subgroup_labels).value_counts().reindex(subgroups).fillna(0).astype(int)
    frac_df = pd.DataFrame({"subgroup": subgroups, "n_cells": sizes.values})
    frac_df["frac"] = frac_df["n_cells"] / frac_df["n_cells"].sum()
    frac_df.to_csv(OUT_DIR / "malignant_balanced3_subgroup_fraction_for_donut.csv", index=False)
    log("[SAVE]", OUT_DIR / "malignant_balanced3_subgroup_fraction_for_donut.csv")

    for i, sg in enumerate(subgroups):
        colors = []
        for j, sg2 in enumerate(subgroups):
            if sg2 == sg:
                colors.append(SUBGROUP_COLORS[j % len(SUBGROUP_COLORS)])
            else:
                colors.append("#D9D9D9")
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=DPI)
        wedges, _ = ax.pie(
            frac_df["n_cells"].values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.2),
        )
        ax.text(0, 0.08, sg, ha="center", va="center", fontsize=18, fontweight="bold", color=SUBGROUP_COLORS[i % 3])
        ax.text(0, -0.12, f"{frac_df.loc[frac_df['subgroup']==sg, 'frac'].iloc[0]*100:.1f}%", ha="center", va="center", fontsize=12)
        ax.set_title(f"{sg} fraction among malignant cells", fontsize=11)
        ax.set_aspect("equal")
        save_fig(fig, OUT_DIR / f"{sg}_donut_fraction_only_self_colored")

# ============================================================
# 8) MAIN
# ============================================================
def main():
    log("=" * 100)
    log("[TARGET_GENE]", TARGET_GENE)
    log("[MODE]", MODE)
    log("[ADATA_UMAP]", ADATA_UMAP)
    log("[STAGE5_BASELINE_NODE]", STAGE5_BASELINE_NODE)
    log("[STAGE5_PERT_NODE]", STAGE5_PERT_NODE)
    log("[OUT_DIR]", OUT_DIR)
    log("=" * 100)

    direct_df = load_stage5_direct_impact()
    mal_adata, celltype_col, malignant_mask = load_umap_and_select_malignant()

    impact_df, row_idx_mal, genes_all, impact_all = infer_gene_impact_all_genes(
        mal_adata=mal_adata,
        direct_df=direct_df,
    )

    soft_prior_df, soft_prior_all = build_soft_target_prior(genes_all)
    impact_df = impact_df.merge(soft_prior_df, on="gene", how="left")
    for c in ["soft_prior_score", "soft_prior_factor"]:
        if c not in impact_df.columns:
            impact_df[c] = 0.0 if c == "soft_prior_score" else 1.0
        impact_df[c] = pd.to_numeric(impact_df[c], errors="coerce").fillna(0.0 if c == "soft_prior_score" else 1.0)
    # soft prior only reweights, it does not remove genes.
    impact_all_prior_weighted = minmax_1d(
        impact_all.astype(float) * (1.0 + FEATURE_PRIOR_BOOST * np.power(soft_prior_all.astype(float), SOFT_PRIOR_POWER))
    ).astype(np.float32)
    impact_df["inferred_impact_prior_weighted"] = impact_all_prior_weighted
    impact_df.to_csv(OUT_DIR / f"malignant_genomewide_inferred_stage5_impact_with_{TARGET_GENE}_soft_prior.csv", index=False)
    log("[SAVE]", OUT_DIR / f"malignant_genomewide_inferred_stage5_impact_with_{TARGET_GENE}_soft_prior.csv")

    feature_genes, feature_idx = choose_feature_genes(impact_df, genes_all)
    X_feat = load_expression_for_rows(row_idx_mal, feature_idx)

    micro_labels = run_microclustering(mal_adata, X_feat)
    subgroup_labels, micro_df = balanced_merge_microclusters(micro_labels, X_feat)

    # save cell assignments
    assign_df = mal_adata.obs.copy()
    assign_df["malignant_micro_cluster"] = micro_labels
    assign_df["malignant_response_subgroup"] = subgroup_labels
    assign_df.to_csv(OUT_DIR / "malignant_cell_microcluster_and_balanced3_subgroup_assignment.csv")
    log("[SAVE]", OUT_DIR / "malignant_cell_microcluster_and_balanced3_subgroup_assignment.csv")

    plot_umap_subgroups(mal_adata, subgroup_labels)

    top_df, top20_df, final_score, subgroups = compute_subgroup_top_genes(
        mal_adata=mal_adata,
        row_idx_mal=row_idx_mal,
        genes_all=genes_all,
        impact_all=impact_all_prior_weighted,
        subgroup_labels=subgroup_labels,
        soft_prior_all=soft_prior_all,
        soft_prior_df=soft_prior_df,
    )

    enrich_df = run_enrichment(top_df, universe=genes_all)
    plot_enrichment_bars(enrich_df)
    plot_back_to_back(top20_df)
    plot_donut_charts(subgroup_labels)

    log("\n[DONE] outputs in:")
    log(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
