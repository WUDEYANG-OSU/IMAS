# -*- coding: utf-8 -*-
"""
sox2_p0_malignant_response_flow_scatter_only_meaningful_axes.py

目的
====
只生成一张图：
- P0 malignant cells 的 SOX2-KO response cytometry-like scatter
- 不画圈、不画 ellipse
- 用已有 malignant 三亚群 assignment + subgroup top1000 response genes 给点上色
- 横纵坐标不使用 M0/M1/M2 差值，而使用有实际意义的连续分数：
    x = SOX2-KO perturbation response score
    y = SOX2-axis malignant-state score
- 展示坐标额外做 sharpen：同一 response mode 点更聚焦，不同 response mode 中心更远

输入依赖
========
1) 前面恶性细胞三亚群脚本输出：
   malignant_cell_microcluster_and_balanced3_subgroup_assignment.csv
   malignant_subgroup_top1000_response_genes.csv

2) 前面 P0-P3 propagation 脚本输出：
   SOX2_cell_level_propagation_response.csv
   SOX2_cluster_propagation_order.csv

3) CA_all 表达矩阵与基因名：
   CA_all.X_rna_log1p_norm_dense.npy
   CA_all.rna_meta.npz
   genes.tsv

输出
====
OUT_DIR 下：
- SOX2_ko_P0_malignant_response_cytometry_like_scatter.png
- SOX2_ko_P0_malignant_response_cytometry_like_scatter.svg
- SOX2_ko_P0_malignant_response_cytometry_like_scatter_values.csv
- SOX2_ko_P0_flow_scatter_gene_weights_used.csv

说明
====
- display_x/display_y 只是展示坐标，用于让图更像流式散点图；
- 原始 SOX2-KO response score 与 SOX2-axis score 仍保存在 CSV 中；
- 不改变 malignant_response_subgroup 结果。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

TARGET_GENE = "SOX2"
MODE = "ko"
TARGET_GENE_UPPER = TARGET_GENE.upper()

BASE = get_project_root()

# ---------- CA_all root: compatible with Worldword / WorldWord ----------
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

# 前面 malignant 三亚群脚本的输出目录
MALIGNANT_OUT_DIR = (
    STAGE5_DIR
    / f"malignant_balanced3_subgroups_{TARGET_GENE}_{MODE}_top1000_tripleLayerTop10_weightedPathwayEnrichment_plots"
)

# 本脚本只输出这一张 scatter
OUT_DIR = MALIGNANT_OUT_DIR / f"{TARGET_GENE}_{MODE}_P0_malignant_response_flow_scatter_only_meaningful_axes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 如果想手动指定 propagation 输出目录，就填绝对路径；否则自动找最新的 SOX2_cell_level_propagation_response.csv
FORCE_PROP_DIR: Optional[str] = None

# P0 设置：优先读取 display_propagation_stage；没有的话按 propagation_rank 跳过起始段后压缩成 P0-P3
N_DISPLAY_STAGES = 4
DISPLAY_STAGE_TO_PLOT = 0
DISPLAY_SKIP_INITIAL_RANKS = 1

# 每个 response mode 使用多少个 top response genes 来算 cell-level weighted response score
TOP_N_SCORE_GENES_PER_SUBGROUP = 300

# 横纵坐标使用实际含义的两条 signature，不再用 M0/M1/M2 差值
TOP_N_AXIS_GENES_X = 420
TOP_N_AXIS_GENES_Y = 320
X_AXIS_LABEL = "SOX2-KO perturbation response score"
Y_AXIS_LABEL = "SOX2-axis response-state score (x-independent)"

# 让纵轴和横轴不要高度共线：
# 1) y 轴基因优先使用 subgroup-discriminative 的 SOX2-axis genes；
# 2) 再对 cell-level y score 做对 x 的线性残差化。
Y_AXIS_REQUIRE_SUBGROUP_SPECIFICITY = True
Y_AXIS_EXCLUDE_TOP_X_GENES = True
Y_AXIS_SUBGROUP_SPEC_POWER = 1.20
Y_AXIS_AXIS_SUPPORT_POWER = 1.00
Y_AXIS_RESIDUALIZE_AGAINST_X = True

# P0 malignant 细胞太少时的处理
MIN_P0_MALIGNANT_CELLS = 30
FALLBACK_TO_ALL_MALIGNANT_IF_P0_TOO_FEW = False

# ---------- display sharpening ----------
# 只影响散点图展示坐标，不改变 assignment / raw semantic x/y score
FLOW_SHARPEN_FOR_DISPLAY = True

# 同一 M 亚群内部收缩；越小越聚焦。当前这版更激进，适合把同亚群点收得更紧
FLOW_WITHIN_GROUP_SHRINK = 0.28

# 不同 M 亚群中心推开；越大组间越远。当前这版更激进，适合把不同亚群拉得更开
FLOW_BETWEEN_GROUP_EXPAND = 3.10

# sharpen 后重新 z-score，使坐标轴尺度好看
FLOW_RE_ZSCORE_AFTER_SHARPEN = True

# ---------- plotting ----------
DPI = 360
SEED = 42
np.random.seed(SEED)

SUBGROUP_COLORS = ["#B2182B", "#2166AC", "#4D9221"]  # red / blue / green
POINT_SIZE = 12
POINT_ALPHA = 0.78
BACKGROUND_ALPHA = 0.10

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# BASIC HELPERS
# ============================================================

def log(*args):
    print(*args, flush=True)


def is_target_gene(g: str) -> bool:
    return str(g).upper() == TARGET_GENE_UPPER


def safe_name(x: str, n: int = 140) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n]


def norm_cell_id(x: str) -> str:
    """Normalize common barcode/cell-id forms for matching."""
    x = str(x).strip()
    if "::" in x:
        x = x.split("::", 1)[-1]
    if ":" in x:
        x = x.rsplit(":", 1)[-1]
    return x.replace("-", ".")


def minmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if int(finite.sum()) == 0:
        return out
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if hi - lo < 1e-12:
        out[finite] = 0.5
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


def sort_labels(xs: List[str]) -> List[str]:
    def _key(v):
        s = str(v)
        if s.startswith("M") and s[1:].isdigit():
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
# LOAD GENE / META / EXPRESSION
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
    cell_ids_all = load_meta_cell_ids()
    exact_map = {str(x): int(i) for i, x in enumerate(cell_ids_all)}

    norm_all = [norm_cell_id(x) for x in cell_ids_all]
    vc = pd.Series(norm_all).value_counts()
    nondup_norm = set(vc[vc == 1].index.astype(str).tolist())
    norm_map = {
        norm_cell_id(cid): int(i)
        for i, cid in enumerate(cell_ids_all)
        if norm_cell_id(cid) in nondup_norm
    }
    return exact_map, norm_map


def match_one_cell_to_expression_row(
    obs_name: str,
    row: pd.Series,
    exact_map: Dict[str, int],
    norm_map: Dict[str, int],
) -> Optional[int]:
    candidate_cols = [
        "cell_id", "barcode", "barcode_counts", "barcode_meta", "CellID",
        "cell", "Cell", "orig.ident",
    ]
    candidates = [str(obs_name), norm_cell_id(obs_name)]
    for c in candidate_cols:
        if c in row.index:
            candidates.append(str(row[c]))
            candidates.append(norm_cell_id(row[c]))

    for x in candidates:
        if x in exact_map:
            return int(exact_map[x])
        nx = norm_cell_id(x)
        if nx in norm_map:
            return int(norm_map[nx])
    return None


def match_assignment_rows_to_expression(assign_df: pd.DataFrame) -> pd.Series:
    exact_map, norm_map = build_expression_row_matcher()
    row_idx = []
    missing = []

    for obs_name, row in assign_df.iterrows():
        hit = match_one_cell_to_expression_row(str(obs_name), row, exact_map, norm_map)
        if hit is None:
            missing.append(str(obs_name))
        else:
            row_idx.append(hit)

    if len(missing) > 0:
        raise RuntimeError(
            f"Missing {len(missing)} assignment cells in expression meta. Examples: {missing[:8]}"
        )

    return pd.Series(np.asarray(row_idx, dtype=np.int64), index=assign_df.index.astype(str))


def load_expression_for_rows(row_idx: np.ndarray, gene_indices: np.ndarray) -> np.ndarray:
    if not LOG1P_NPY.exists():
        raise FileNotFoundError(f"Missing expression matrix: {LOG1P_NPY}")
    X_mmap = np.load(LOG1P_NPY, mmap_mode="r")
    return np.asarray(X_mmap[np.ix_(row_idx.astype(np.int64), gene_indices.astype(np.int64))], dtype=np.float32)


# ============================================================
# LOAD EXISTING MALIGNANT SUBGROUP OUTPUTS
# ============================================================

def load_assignment() -> pd.DataFrame:
    p = MALIGNANT_OUT_DIR / "malignant_cell_microcluster_and_balanced3_subgroup_assignment.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing malignant assignment CSV:\n{p}\n"
            "请先跑完恶性细胞三亚群那段代码。"
        )

    df = pd.read_csv(p, index_col=0, low_memory=False)
    if "malignant_response_subgroup" not in df.columns:
        cand = [c for c in df.columns if "subgroup" in str(c).lower()]
        if not cand:
            raise KeyError(f"Cannot find malignant_response_subgroup in {p}. Available={list(df.columns)}")
        df["malignant_response_subgroup"] = df[cand[0]].astype(str)
    else:
        df["malignant_response_subgroup"] = df["malignant_response_subgroup"].astype(str)

    df.index = df.index.astype(str)
    log("[LOAD] assignment =", p)
    log("[LOAD] assignment cells =", len(df))
    log(df["malignant_response_subgroup"].value_counts().sort_index().to_string())
    return df


def find_top_response_file() -> Path:
    exact = MALIGNANT_OUT_DIR / "malignant_subgroup_top1000_response_genes.csv"
    if exact.exists():
        return exact

    cands = sorted(
        MALIGNANT_OUT_DIR.glob("malignant_subgroup_top*_response_genes.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    # Avoid using *_for_plot unless nothing else exists.
    cands_non_plot = [p for p in cands if "for_plot" not in p.name]
    if cands_non_plot:
        return cands_non_plot[0]
    if cands:
        return cands[0]

    raise FileNotFoundError(
        "Cannot find malignant_subgroup_top*_response_genes.csv under:\n"
        f"{MALIGNANT_OUT_DIR}\n"
        "请先跑完恶性细胞三亚群 top1000 response genes 那段代码。"
    )


def load_top_response_genes() -> pd.DataFrame:
    p = find_top_response_file()
    df = pd.read_csv(p, low_memory=False)

    need = {"subgroup", "gene"}
    if not need.issubset(df.columns):
        raise KeyError(f"{p} must contain columns: {need}. Available={list(df.columns)}")

    df["subgroup"] = df["subgroup"].astype(str)
    df["gene"] = df["gene"].astype(str)

    # 优先使用原 top1000 中保存的 response_weight；没有时使用 final_score_raw 或其他可用权重。
    weight_col = None
    for c in [
        "response_weight",
        "final_score_raw",
        "response_selection_score",
        "impact_eff_with_prior_floor",
        "impact_norm",
        "lane_specific_response",
        "lane_shared_response",
        "lane_natural_sox2_axis",
        "axis_selection_score",
    ]:
        if c in df.columns:
            weight_col = c
            break

    if weight_col is None:
        log("[WARN] no response weight column found. Use rank-decay weight.")
        if "rank" in df.columns:
            df["_weight_for_score"] = 1.0 / np.maximum(
                1.0,
                pd.to_numeric(df["rank"], errors="coerce").fillna(9999).values.astype(float),
            )
        else:
            df["_weight_for_score"] = 1.0
        weight_col = "_weight_for_score"
    else:
        df["_weight_for_score"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).values.astype(float)

    log("[LOAD] top response genes =", p)
    log("[LOAD] weight column used =", weight_col)
    log("[LOAD] rows =", len(df))
    return df


# ============================================================
# P0 CELL FILTER FROM PROPAGATION OUTPUT
# ============================================================

def find_propagation_dir() -> Path:
    if FORCE_PROP_DIR is not None and str(FORCE_PROP_DIR).strip():
        d = Path(FORCE_PROP_DIR)
        if not d.exists():
            raise FileNotFoundError(f"FORCE_PROP_DIR does not exist: {d}")
        return d

    cands = []

    # 常见命名：SOX2_ko_umap_propagation_direction_cluster_gene_response_heatmap*
    for d in STAGE5_DIR.glob(f"{TARGET_GENE}_{MODE}_umap_propagation_direction_cluster_gene_response_heatmap*"):
        p = d / f"{TARGET_GENE}_cell_level_propagation_response.csv"
        if d.is_dir() and p.exists():
            cands.append(d)

    # 保险：递归找 cell-level propagation 文件
    for p in STAGE5_DIR.rglob(f"{TARGET_GENE}_cell_level_propagation_response.csv"):
        if p.parent not in cands:
            cands.append(p.parent)

    if not cands:
        raise FileNotFoundError(
            f"Cannot find {TARGET_GENE}_cell_level_propagation_response.csv under:\n{STAGE5_DIR}\n"
            "请先跑完 P0-P3 propagation UMAP 那段代码。"
        )

    cands = sorted(
        cands,
        key=lambda d: (d / f"{TARGET_GENE}_cell_level_propagation_response.csv").stat().st_mtime,
        reverse=True,
    )
    log("[PROP] using propagation dir =", cands[0])
    return cands[0]


def derive_display_stage_from_rank(prop_df: pd.DataFrame) -> pd.DataFrame:
    prop_df = prop_df.copy()

    if "display_propagation_stage" in prop_df.columns:
        prop_df["display_propagation_stage"] = pd.to_numeric(prop_df["display_propagation_stage"], errors="coerce")
        return prop_df

    if "propagation_rank" not in prop_df.columns:
        raise KeyError(
            "cluster_propagation_order.csv has neither display_propagation_stage nor propagation_rank."
        )

    rank = pd.to_numeric(prop_df["propagation_rank"], errors="coerce")
    eligible = rank >= int(DISPLAY_SKIP_INITIAL_RANKS)
    if int(eligible.sum()) == 0:
        log("[WARN] no eligible ranks after DISPLAY_SKIP_INITIAL_RANKS. Use all ranks.")
        eligible = rank >= np.nanmin(rank.values)

    rmin = int(np.nanmin(rank[eligible]))
    rmax = int(np.nanmax(rank[eligible]))
    span = max(1, rmax - rmin + 1)

    stage = np.full(len(prop_df), np.nan, dtype=float)
    for i, r in enumerate(rank.values):
        if not np.isfinite(r) or r < rmin:
            continue
        s = int(np.floor((float(r) - rmin) * int(N_DISPLAY_STAGES) / span))
        s = max(0, min(int(N_DISPLAY_STAGES) - 1, s))
        stage[i] = s

    prop_df["display_propagation_stage"] = stage
    prop_df["display_stage_derived_from_rank"] = 1
    return prop_df


def load_p0_cell_id_set() -> Tuple[set, str, Path]:
    prop_dir = find_propagation_dir()
    cell_csv = prop_dir / f"{TARGET_GENE}_cell_level_propagation_response.csv"
    prop_csv = prop_dir / f"{TARGET_GENE}_cluster_propagation_order.csv"

    if not cell_csv.exists():
        raise FileNotFoundError(f"Missing: {cell_csv}")

    cell_df = pd.read_csv(cell_csv, low_memory=False)
    if "cell_id" not in cell_df.columns:
        raise KeyError(f"{cell_csv} must contain column: cell_id")

    if "display_propagation_stage" in cell_df.columns:
        cell_df["display_propagation_stage"] = pd.to_numeric(cell_df["display_propagation_stage"], errors="coerce")
    else:
        if not prop_csv.exists():
            raise FileNotFoundError(f"Missing {prop_csv}; cannot map clusters to display P0.")
        prop_df = pd.read_csv(prop_csv, low_memory=False)
        if "cluster_label" not in prop_df.columns:
            raise KeyError(f"{prop_csv} must contain cluster_label")
        prop_df["cluster_label"] = prop_df["cluster_label"].astype(str)
        prop_df = derive_display_stage_from_rank(prop_df)
        stage_map = prop_df.set_index("cluster_label")["display_propagation_stage"].to_dict()

        if "cluster_label" not in cell_df.columns:
            raise KeyError(
                f"{cell_csv} has no display_propagation_stage and no cluster_label. Cannot infer P0 cells."
            )
        cell_df["display_propagation_stage"] = cell_df["cluster_label"].astype(str).map(stage_map)

    stage = pd.to_numeric(cell_df["display_propagation_stage"], errors="coerce")
    p0 = cell_df[stage == int(DISPLAY_STAGE_TO_PLOT)].copy()

    p0_ids_exact = set(p0["cell_id"].astype(str).tolist())
    p0_ids_norm = {norm_cell_id(x) for x in p0_ids_exact}
    p0_ids = p0_ids_exact.union(p0_ids_norm)

    log("[P0] propagation cell file =", cell_csv)
    log(f"[P0] total cells in display P{DISPLAY_STAGE_TO_PLOT} =", len(p0_ids_exact))
    return p0_ids, f"P{DISPLAY_STAGE_TO_PLOT}", prop_dir


def filter_assignment_to_p0(assign_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, Path]:
    p0_ids, tag, prop_dir = load_p0_cell_id_set()

    candidate_cols = ["cell_id", "barcode", "barcode_counts", "barcode_meta", "CellID"]
    keep = []
    for obs_name, row in assign_df.iterrows():
        candidates = [str(obs_name), norm_cell_id(obs_name)]
        for c in candidate_cols:
            if c in assign_df.columns:
                candidates.append(str(row[c]))
                candidates.append(norm_cell_id(row[c]))
        keep.append(any(x in p0_ids for x in candidates))

    keep = np.asarray(keep, dtype=bool)
    n_keep = int(keep.sum())

    if n_keep < int(MIN_P0_MALIGNANT_CELLS):
        msg = f"[WARN] P0 malignant cells too few: {n_keep}."
        if FALLBACK_TO_ALL_MALIGNANT_IF_P0_TOO_FEW:
            log(msg, "Fallback to all malignant cells.")
            return assign_df.copy(), "all_malignant_fallback", prop_dir
        raise RuntimeError(
            msg
            + "\n如果你确实想强制画少量 P0 malignant cells，可以降低 MIN_P0_MALIGNANT_CELLS；"
            + "\n如果想自动退回全部 malignant cells，把 FALLBACK_TO_ALL_MALIGNANT_IF_P0_TOO_FEW=True。"
        )

    out = assign_df.loc[keep].copy()
    log(f"[P0] malignant cells kept = {n_keep}/{len(assign_df)}")
    log(out["malignant_response_subgroup"].value_counts().sort_index().to_string())
    return out, tag, prop_dir


# ============================================================
# BUILD WEIGHTED RESPONSE SIGNATURES
# ============================================================

def build_subgroup_gene_weights(
    top_df: pd.DataFrame,
    genes_all: List[str],
    subgroups: List[str],
) -> Dict[str, pd.DataFrame]:
    gene_upper_to_idx = {str(g).upper(): i for i, g in enumerate(genes_all)}
    out: Dict[str, pd.DataFrame] = {}

    for sg in subgroups:
        sub = top_df[top_df["subgroup"].astype(str) == str(sg)].copy()
        if len(sub) == 0:
            raise RuntimeError(f"No top response genes found for subgroup {sg}")

        if "rank" in sub.columns:
            sub["_rank_sort"] = pd.to_numeric(sub["rank"], errors="coerce").fillna(999999)
            sub = sub.sort_values(["_rank_sort", "_weight_for_score"], ascending=[True, False])
        else:
            sub = sub.sort_values("_weight_for_score", ascending=False)

        rows = []
        seen = set()
        for _, r in sub.iterrows():
            g = str(r["gene"])
            gu = g.upper()
            if gu in seen:
                continue
            if is_target_gene(g):
                continue
            if gu not in gene_upper_to_idx:
                continue

            w = float(pd.to_numeric(pd.Series([r["_weight_for_score"]]), errors="coerce").fillna(0.0).iloc[0])
            if not np.isfinite(w) or w <= 0:
                continue

            rows.append({
                "subgroup": sg,
                "gene": genes_all[gene_upper_to_idx[gu]],
                "gene_idx": int(gene_upper_to_idx[gu]),
                "raw_weight": w,
            })
            seen.add(gu)

            if len(rows) >= int(TOP_N_SCORE_GENES_PER_SUBGROUP):
                break

        if len(rows) == 0:
            raise RuntimeError(f"No valid weighted genes for subgroup {sg}")

        gdf = pd.DataFrame(rows)
        # Robust weight: min-max + rank decay, avoiding one extremely large weight dominating the cell score.
        raw = gdf["raw_weight"].values.astype(float)
        raw01 = minmax_1d(raw)
        rank_decay = 1.0 / np.sqrt(np.arange(1, len(gdf) + 1, dtype=float))
        w = (0.72 * raw01 + 0.28 * rank_decay)
        w = np.maximum(w, 1e-12)
        w = w / (np.sum(w) + 1e-12)
        gdf["weight"] = w.astype(float)

        out[sg] = gdf
        log(f"[WEIGHT] {sg}: genes used = {len(gdf)}")

    return out


def _numeric_series_or_zero(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(np.float64)


def _combine_weight_columns(df: pd.DataFrame, columns: List[str], fallback_col: str = "_weight_for_score") -> np.ndarray:
    """
    Combine available gene-level evidence columns into one 0-1 weight vector.
    Each column is min-max normalized first, then the row-wise max is used.
    This makes the axis robust to slightly different saved CSV versions.
    """
    mats = []
    used = []
    for c in columns:
        if c in df.columns:
            v = _numeric_series_or_zero(df, c)
            if np.nanmax(v) > 0:
                mats.append(minmax_1d(v))
                used.append(c)
    if not mats:
        v = _numeric_series_or_zero(df, fallback_col)
        if np.nanmax(v) <= 0:
            v = np.ones(len(df), dtype=np.float64)
        mats = [minmax_1d(v)]
        used = [fallback_col]
    score = np.max(np.vstack(mats), axis=0)
    return score.astype(np.float64)


def build_semantic_axis_gene_weights(top_df: pd.DataFrame, genes_all: List[str]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Build two less-collinear, more subgroup-separating semantic axes.

    X axis: SOX2-KO perturbation response score
      - prefers globally strong perturbation-response genes.

    Y axis: SOX2-axis response-state score (x-independent)
      - prefers SOX2-axis / malignant-state genes that are also subgroup-discriminative,
        so the figure is less like one continuous diagonal line.
    """
    df = top_df.copy()
    df["gene"] = df["gene"].astype(str)
    df["subgroup"] = df["subgroup"].astype(str)
    gene_upper_to_idx = {str(g).upper(): i for i, g in enumerate(genes_all)}

    perturb_cols = [
        "perturbation_response_weight",
        "response_selection_score",
        "lane_specific_response",
        "impact_eff_with_prior_floor",
        "impact_norm",
        "inferred_impact",
        "direct_impact",
        "direct_impact_component",
        "_weight_for_score",
    ]
    axis_cols = [
        "natural_SOX2_axis_weight",
        "natural_sox2_axis_weight",
        "axis_selection_score",
        "axis_marker_selection_score",
        "lane_natural_sox2_axis",
        "soft_prior_score",
        "target_expr_corr_score",
        "malignant_vs_nonmal_score",
        "curated_target_axis_score",
        "tumor_expr_selection_score",
    ]

    df["_perturb_axis_weight_raw"] = _combine_weight_columns(df, perturb_cols)
    df["_sox2_axis_weight_raw"] = _combine_weight_columns(df, axis_cols)

    rows = []
    grouped = df.assign(gene_upper=df["gene"].str.upper()).groupby("gene_upper", sort=False)
    for gu, sub in grouped:
        if gu not in gene_upper_to_idx:
            continue
        g = genes_all[gene_upper_to_idx[gu]]
        if is_target_gene(g):
            continue

        # global supports
        perturb_support = float(np.nanmax(sub["_perturb_axis_weight_raw"].values))
        axis_support = float(np.nanmax(sub["_sox2_axis_weight_raw"].values))

        # subgroup discriminative support: the more uneven across the 3 response modes, the higher.
        per_sg = []
        for sg, sub2 in sub.groupby("subgroup", sort=False):
            per_sg.append(float(np.nanmax(sub2["_sox2_axis_weight_raw"].values)))
        per_sg = np.asarray(per_sg, dtype=float)
        if len(per_sg) <= 1:
            subgroup_spec = 0.0
        else:
            subgroup_spec = float(np.nanmax(per_sg) - np.nanmedian(per_sg))
            subgroup_spec = max(subgroup_spec, 0.0)

        rows.append({
            "gene": g,
            "gene_idx": int(gene_upper_to_idx[gu]),
            "perturb_axis_weight_raw": perturb_support,
            "sox2_axis_weight_raw": axis_support,
            "subgroup_specificity_raw": subgroup_spec,
            "n_source_rows": int(len(sub)),
            "source_subgroups": ";".join(sorted(set(sub["subgroup"].astype(str).tolist()))),
        })

    agg = pd.DataFrame(rows)
    if len(agg) == 0:
        raise RuntimeError("No valid genes available for semantic axes.")

    agg["perturb_axis_weight_raw01"] = minmax_1d(agg["perturb_axis_weight_raw"].values.astype(float))
    agg["sox2_axis_weight_raw01"] = minmax_1d(agg["sox2_axis_weight_raw"].values.astype(float))
    agg["subgroup_specificity_raw01"] = minmax_1d(agg["subgroup_specificity_raw"].values.astype(float))

    # X-axis genes: globally strong perturbation responders.
    xdf = agg.sort_values(["perturb_axis_weight_raw", "sox2_axis_weight_raw"], ascending=[False, False]).head(int(TOP_N_AXIS_GENES_X)).copy()
    raw = xdf["perturb_axis_weight_raw01"].values.astype(float)
    raw01 = minmax_1d(raw)
    rank_decay = 1.0 / np.sqrt(np.arange(1, len(xdf) + 1, dtype=float))
    wx = 0.82 * raw01 + 0.18 * rank_decay
    wx = np.maximum(wx, 1e-12)
    wx = wx / (wx.sum() + 1e-12)
    xdf["axis_name"] = "SOX2_KO_perturbation_response"
    xdf["weight"] = wx.astype(float)

    # Y-axis genes: SOX2-axis / malignant-state support, but with subgroup-discriminative emphasis.
    ydf = agg.copy()
    if Y_AXIS_EXCLUDE_TOP_X_GENES:
        x_top_genes = set(xdf["gene"].astype(str).tolist())
        ydf = ydf[~ydf["gene"].astype(str).isin(x_top_genes)].copy()
    y_support = np.power(np.clip(ydf["sox2_axis_weight_raw01"].values.astype(float), 0.0, None), Y_AXIS_AXIS_SUPPORT_POWER)
    if Y_AXIS_REQUIRE_SUBGROUP_SPECIFICITY:
        y_spec = np.power(np.clip(ydf["subgroup_specificity_raw01"].values.astype(float), 0.0, None), Y_AXIS_SUBGROUP_SPEC_POWER)
        y_raw = y_support * (0.30 + 0.70 * y_spec)
    else:
        y_raw = y_support
    ydf["_y_axis_priority"] = y_raw
    ydf = ydf.sort_values(["_y_axis_priority", "sox2_axis_weight_raw", "subgroup_specificity_raw"], ascending=[False, False, False]).head(int(TOP_N_AXIS_GENES_Y)).copy()
    raw = ydf["_y_axis_priority"].values.astype(float)
    raw01 = minmax_1d(raw)
    rank_decay = 1.0 / np.sqrt(np.arange(1, len(ydf) + 1, dtype=float))
    wy = 0.80 * raw01 + 0.20 * rank_decay
    wy = np.maximum(wy, 1e-12)
    wy = wy / (wy.sum() + 1e-12)
    ydf["axis_name"] = "SOX2_axis_response_state_xindependent"
    ydf["weight"] = wy.astype(float)

    axis_tables = {
        "SOX2_KO_perturbation_response": xdf[["axis_name", "gene", "gene_idx", "weight", "perturb_axis_weight_raw", "sox2_axis_weight_raw", "subgroup_specificity_raw", "n_source_rows", "source_subgroups"]].copy(),
        "SOX2_axis_response_state_xindependent": ydf[["axis_name", "gene", "gene_idx", "weight", "perturb_axis_weight_raw", "sox2_axis_weight_raw", "subgroup_specificity_raw", "n_source_rows", "source_subgroups"]].copy(),
    }
    all_axis_weight_df = pd.concat(axis_tables.values(), axis=0, ignore_index=True)
    log(f"[AXIS] SOX2_KO_perturbation_response: genes used = {len(xdf)}")
    log(f"[AXIS] SOX2_axis_response_state_xindependent: genes used = {len(ydf)}")
    return axis_tables, all_axis_weight_df


def compute_signature_score(row_idx_use: np.ndarray, gene_weight_df: pd.DataFrame) -> np.ndarray:
    idx = gene_weight_df["gene_idx"].values.astype(np.int64)
    w = gene_weight_df["weight"].values.astype(np.float64)
    X = load_expression_for_rows(row_idx_use, idx).astype(np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xz = (X - mu) / sd
    score = Xz @ w
    return zscore_1d(score).astype(np.float64)


def compute_weighted_scores(
    assign_p0: pd.DataFrame,
    row_idx_p0: np.ndarray,
    genes_all: List[str],
    subgroup_gene_weights: Dict[str, pd.DataFrame],
    semantic_axis_weights: Dict[str, pd.DataFrame],
    subgroups: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute:
    1) subgroup-weighted response scores, only for confidence / color consistency;
    2) semantic x/y scores used as actual plot axes.
    """
    score_cols = {}
    weight_tables = []

    for sg in subgroups:
        gdf = subgroup_gene_weights[sg].copy()
        score = compute_signature_score(row_idx_p0, gdf)
        score_cols[f"{sg}_weighted_response_score"] = score.astype(float)
        weight_tables.append(gdf.assign(axis_name="subgroup_response_signature"))

    score_mat = np.vstack([score_cols[f"{sg}_weighted_response_score"] for sg in subgroups]).T
    max_score = np.max(score_mat, axis=1)
    second_score = np.partition(score_mat, -2, axis=1)[:, -2]
    confidence = minmax_1d(max_score - second_score)
    pred_idx = np.argmax(score_mat, axis=1)
    pred_sg = [subgroups[i] for i in pred_idx]

    perturb_score = compute_signature_score(
        row_idx_p0,
        semantic_axis_weights["SOX2_KO_perturbation_response"],
    )
    sox2_axis_score = compute_signature_score(
        row_idx_p0,
        semantic_axis_weights["SOX2_axis_response_state_xindependent"],
    )

    # reduce x-y collinearity: y becomes the component of SOX2-axis state independent of x
    if Y_AXIS_RESIDUALIZE_AGAINST_X:
        x0 = np.asarray(perturb_score, dtype=float)
        y0 = np.asarray(sox2_axis_score, dtype=float)
        varx = float(np.nanvar(x0))
        if np.isfinite(varx) and varx > 1e-12:
            beta = float(np.nanmean((x0 - np.nanmean(x0)) * (y0 - np.nanmean(y0))) / (varx + 1e-12))
            y0 = y0 - beta * x0
            sox2_axis_score = zscore_1d(y0).astype(np.float64)

    out_df = pd.DataFrame({
        "cell_id": assign_p0.index.astype(str).values,
        "malignant_response_subgroup": assign_p0["malignant_response_subgroup"].astype(str).values,
        "predicted_response_subgroup_by_score": pred_sg,
        "sox2_ko_perturbation_response_score_raw": perturb_score.astype(float),
        "sox2_axis_malignant_state_score_raw": sox2_axis_score.astype(float),
        "response_confidence": confidence.astype(float),
    })

    # Labels for legend: avoid exposing M0/M1/M2 on the figure.
    label_map = {sg: f"Response mode {i + 1}" for i, sg in enumerate(subgroups)}
    out_df["response_mode_label"] = [label_map.get(x, str(x)) for x in out_df["malignant_response_subgroup"].astype(str)]

    for c, v in score_cols.items():
        out_df[c] = v

    wt_df = pd.concat(weight_tables, axis=0, ignore_index=True)
    out_df = sharpen_flow_coordinates_for_display(out_df, subgroups)
    return out_df, wt_df


# ============================================================
# DISPLAY SHARPENING
# ============================================================

def sharpen_flow_coordinates_for_display(flow_df: pd.DataFrame, subgroups: List[str]) -> pd.DataFrame:
    """
    只用于展示的坐标 sharpen：
    1) 同组点向本组中心收缩；
    2) 不同 response mode 中心相对全局中心推远；
    3) 原始语义分数不覆盖，另存 display_x / display_y。
    """
    out = flow_df.copy()
    x = out["sox2_ko_perturbation_response_score_raw"].values.astype(float)
    y = out["sox2_axis_malignant_state_score_raw"].values.astype(float)
    labels = out["malignant_response_subgroup"].astype(str).values

    if (not FLOW_SHARPEN_FOR_DISPLAY) or len(subgroups) < 2:
        out["display_x"] = x
        out["display_y"] = y
        out["display_sharpening_used"] = 0
        return out

    gx = float(np.nanmedian(x))
    gy = float(np.nanmedian(y))

    x_new = x.copy()
    y_new = y.copy()

    for sg in subgroups:
        m = labels == str(sg)
        if int(m.sum()) == 0:
            continue

        cx = float(np.nanmedian(x[m]))
        cy = float(np.nanmedian(y[m]))

        cx2 = gx + float(FLOW_BETWEEN_GROUP_EXPAND) * (cx - gx)
        cy2 = gy + float(FLOW_BETWEEN_GROUP_EXPAND) * (cy - gy)

        x_new[m] = cx2 + float(FLOW_WITHIN_GROUP_SHRINK) * (x[m] - cx)
        y_new[m] = cy2 + float(FLOW_WITHIN_GROUP_SHRINK) * (y[m] - cy)

    if FLOW_RE_ZSCORE_AFTER_SHARPEN:
        x_new = zscore_1d(x_new)
        y_new = zscore_1d(y_new)

    out["display_x"] = x_new.astype(float)
    out["display_y"] = y_new.astype(float)
    out["display_sharpening_used"] = 1
    out["display_within_group_shrink"] = float(FLOW_WITHIN_GROUP_SHRINK)
    out["display_between_group_expand"] = float(FLOW_BETWEEN_GROUP_EXPAND)
    return out


# ============================================================
# PLOT
# ============================================================

def plot_flow_scatter(flow_df: pd.DataFrame, subgroups: List[str], p0_tag: str):
    color_map = {sg: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)] for i, sg in enumerate(subgroups)}
    label_map = {sg: f"Response mode {i + 1}" for i, sg in enumerate(subgroups)}

    # Use sharpened display coordinates. Raw semantic scores remain in CSV.
    x = flow_df["display_x"].values.astype(float)
    y = flow_df["display_y"].values.astype(float)

    fig, ax = plt.subplots(figsize=(6.6, 6.1), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax.scatter(
        x, y,
        s=POINT_SIZE * 0.68,
        c="#D0D0D0",
        alpha=BACKGROUND_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    for sg in subgroups:
        m = flow_df["malignant_response_subgroup"].astype(str).values == str(sg)
        if int(m.sum()) == 0:
            continue
        ax.scatter(
            x[m], y[m],
            s=POINT_SIZE,
            c=color_map[sg],
            alpha=POINT_ALPHA,
            linewidths=0,
            rasterized=True,
            label=f"{label_map[sg]}  n={int(m.sum())}",
            zorder=3,
        )

    # No circles / no ellipses / no gates.
    ax.axhline(0, color="#777777", linewidth=0.60, alpha=0.33, zorder=0)
    ax.axvline(0, color="#777777", linewidth=0.60, alpha=0.33, zorder=0)

    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_title(f"{TARGET_GENE} KO response-state landscape at {p0_tag}")

    ax.legend(frameon=False, loc="best", fontsize=8, markerscale=1.35)

    if len(x) > 0 and np.isfinite(x).any() and np.isfinite(y).any():
        xpad = 0.11 * max(1e-6, float(np.nanmax(x) - np.nanmin(x)))
        ypad = 0.11 * max(1e-6, float(np.nanmax(y) - np.nanmin(y)))
        ax.set_xlim(float(np.nanmin(x)) - xpad, float(np.nanmax(x)) + xpad)
        ax.set_ylim(float(np.nanmin(y)) - ypad, float(np.nanmax(y)) + ypad)

    out_base = OUT_DIR / f"{TARGET_GENE}_{MODE}_{p0_tag}_malignant_response_cytometry_like_scatter_meaningful_axes"
    save_fig(fig, out_base)


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 100)
    log("[START] P0 malignant SOX2-KO response cytometry-like scatter only")
    log("[TARGET_GENE]", TARGET_GENE)
    log("[MODE]", MODE)
    log("[MALIGNANT_OUT_DIR]", MALIGNANT_OUT_DIR)
    log("[OUT_DIR]", OUT_DIR)
    log("=" * 100)

    if not LOG1P_NPY.exists():
        raise FileNotFoundError(f"Missing log1p matrix: {LOG1P_NPY}")
    if not META_NPZ.exists():
        raise FileNotFoundError(f"Missing meta npz: {META_NPZ}")
    if not GENES_TSV.exists():
        raise FileNotFoundError(f"Missing genes.tsv: {GENES_TSV}")

    genes_all = read_gene_names()
    assign_df = load_assignment()
    top_df = load_top_response_genes()

    subgroups = sort_labels(assign_df["malignant_response_subgroup"].astype(str).unique().tolist())
    if len(subgroups) != 3:
        raise RuntimeError(f"Expected 3 malignant response subgroups, got {len(subgroups)}: {subgroups}")
    log("[SUBGROUPS]", subgroups)

    # Filter to P0 malignant cells.
    assign_p0, p0_tag, prop_dir = filter_assignment_to_p0(assign_df)

    # Match all assignment cells to expression rows, then subset to P0.
    row_idx_series = match_assignment_rows_to_expression(assign_df)
    row_idx_p0 = row_idx_series.loc[assign_p0.index.astype(str)].values.astype(np.int64)

    # Build subgroup response gene weights for coloring/confidence.
    subgroup_gene_weights = build_subgroup_gene_weights(
        top_df=top_df,
        genes_all=genes_all,
        subgroups=subgroups,
    )

    # Build semantic x/y axis gene weights. These make the axis labels biologically meaningful,
    # rather than showing M1-M0 / M2-M0 differences.
    semantic_axis_weights, axis_wt_df = build_semantic_axis_gene_weights(
        top_df=top_df,
        genes_all=genes_all,
    )

    # Cell-level semantic scores and sharpened display coordinates.
    flow_df, wt_df = compute_weighted_scores(
        assign_p0=assign_p0,
        row_idx_p0=row_idx_p0,
        genes_all=genes_all,
        subgroup_gene_weights=subgroup_gene_weights,
        semantic_axis_weights=semantic_axis_weights,
        subgroups=subgroups,
    )

    # Save values.
    values_csv = OUT_DIR / f"{TARGET_GENE}_{MODE}_{p0_tag}_malignant_response_cytometry_like_scatter_values.csv"
    weights_csv = OUT_DIR / f"{TARGET_GENE}_{MODE}_{p0_tag}_flow_scatter_gene_weights_used.csv"
    axis_weights_csv = OUT_DIR / f"{TARGET_GENE}_{MODE}_{p0_tag}_semantic_axis_gene_weights_used.csv"
    qc_csv = OUT_DIR / f"{TARGET_GENE}_{MODE}_{p0_tag}_flow_scatter_qc.csv"

    flow_df.to_csv(values_csv, index=False)
    wt_df.to_csv(weights_csv, index=False)
    axis_wt_df.to_csv(axis_weights_csv, index=False)

    qc = pd.DataFrame([{
        "target_gene": TARGET_GENE,
        "mode": MODE,
        "n_p0_malignant_cells": int(len(assign_p0)),
        "subgroups": ";".join(subgroups),
        "subgroup_counts": ";".join([f"{sg}:{int((assign_p0['malignant_response_subgroup'].astype(str)==sg).sum())}" for sg in subgroups]),
        "top_n_score_genes_per_subgroup": int(TOP_N_SCORE_GENES_PER_SUBGROUP),
        "top_n_axis_genes": int(TOP_N_AXIS_GENES),
        "x_axis_label": X_AXIS_LABEL,
        "y_axis_label": Y_AXIS_LABEL,
        "flow_sharpen_for_display": int(FLOW_SHARPEN_FOR_DISPLAY),
        "flow_within_group_shrink": float(FLOW_WITHIN_GROUP_SHRINK),
        "flow_between_group_expand": float(FLOW_BETWEEN_GROUP_EXPAND),
        "propagation_dir": str(prop_dir),
        "malignant_out_dir": str(MALIGNANT_OUT_DIR),
        "out_dir": str(OUT_DIR),
    }])
    qc.to_csv(qc_csv, index=False)

    log("[SAVE]", values_csv)
    log("[SAVE]", weights_csv)
    log("[SAVE]", axis_weights_csv)
    log("[SAVE]", qc_csv)

    # Plot only this one scatter.
    plot_flow_scatter(flow_df, subgroups, p0_tag=p0_tag)

    log("=" * 100)
    log("[DONE] output in:", OUT_DIR)
    log("=" * 100)


if __name__ == "__main__":
    main()
