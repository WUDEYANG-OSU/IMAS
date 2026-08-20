from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# 0) 路径配置
# ============================================================
OUT_ROOT_CANDIDATES = [
    get_runs_root() / "frontier_transfer_hash_stageB_data_regime_comparison" / "benchmark_umap_dependency_structure_posthoc",
    Path.cwd(),
]

DATASETS = ["CRC-Tissue", "Glioma-Tissue", "Osteosarcoma-Tissue"]
REGIME_ORDER = ["multiomic_only", "mixed", "scrna_only"]
REGIME_LABEL = {
    "multiomic_only": "Multiomic-only",
    "mixed": "Mixed",
    "scrna_only": "scRNA-only",
}

# 如果 Osteosarcoma 文件名前缀和这里不同，可以在这里补 alias
DATASET_ALIASES = {
    "CRC-Tissue": ["CRC-Tissue", "CRC"],
    "Glioma-Tissue": ["Glioma-Tissue", "Glioma"],
    "Osteosarcoma-Tissue": [
        "Osteosarcoma-Tissue",
        "Osteosarcoma",
        "Osteosarcoma-Cellline",
        "Osteosarcoma-CellLine",
        "OS-Tissue",
        "OS",
    ],
}

# ============================================================
# 1) 画风参数（按你要的“水彩散点感”）
# ============================================================
PLOT_DOT_SIZE = 0.45      # 建议 0.30 ~ 0.70
PLOT_DOT_ALPHA = 0.18     # 建议 0.12 ~ 0.25
FIG_W = 12.5
FIG_H = 12.0
DPI = 400

SUPTITLE = "Benchmark-style UMAP of IMAS dependent-structure manifolds"

# ============================================================
# 2) 工具函数
# ============================================================
def detect_out_root():
    for root in OUT_ROOT_CANDIDATES:
        if root.exists():
            # 只要能找到任意一个 regime 文件就认为目录可用
            hits = list(root.glob("*_multiomic_only_cell_structure_scores_with_DepStruct.csv"))
            if len(hits) > 0:
                return root
    raise FileNotFoundError("Cannot detect OUT_ROOT. Please check OUT_ROOT_CANDIDATES.")

def normalize_name(x):
    return str(x).lower().replace("-", "").replace("_", "").replace(" ", "")

def resolve_dataset_prefix(out_root: Path, requested_name: str):
    """
    根据已有文件解析 dataset prefix。
    """
    suffix = "_multiomic_only_cell_structure_scores_with_DepStruct.csv"
    existing_prefixes = sorted([f.name[:-len(suffix)] for f in out_root.glob(f"*{suffix}")])

    aliases = DATASET_ALIASES.get(requested_name, [requested_name])
    candidates = aliases + [requested_name]

    # 1) 精确命中
    for a in candidates:
        if a in existing_prefixes:
            return a

    # 2) 归一化模糊命中
    req_low = normalize_name(requested_name)
    for p in existing_prefixes:
        p_low = normalize_name(p)
        if req_low in p_low or p_low in req_low:
            return p

    # 3) alias 归一化命中
    for a in candidates:
        a_low = normalize_name(a)
        for p in existing_prefixes:
            p_low = normalize_name(p)
            if a_low in p_low or p_low in a_low:
                return p

    raise FileNotFoundError(f"Cannot resolve dataset prefix for: {requested_name}")

def pick_col(df: pd.DataFrame, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in low:
            return low[a.lower()]
    return None

def cluster_sort_key(x):
    s = str(x)
    nums = re.findall(r"\d+", s)
    if nums:
        return (0, int(nums[0]), s)
    return (1, math.inf, s)

def build_palette(n):
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent", "Dark2"]
    colors = []
    for c in cmaps:
        cmap = mpl.colormaps[c]
        if hasattr(cmap, "colors"):
            colors.extend(list(cmap.colors))
        else:
            colors.extend([cmap(i) for i in np.linspace(0, 1, 20)])
    uniq = []
    for col in colors:
        if col not in uniq:
            uniq.append(col)
    if len(uniq) < n:
        raise ValueError(f"Not enough colors: {len(uniq)} < {n}")
    return uniq[:n]

def compute_depstruct_cell(df: pd.DataFrame):
    """
    如果文件里没有 depstruct_score_cell，就按当前 posthoc 权重补算。
    """
    if "depstruct_score_cell" in df.columns:
        return pd.to_numeric(df["depstruct_score_cell"], errors="coerce").values

    weights = {
        "dependency_support_score_global": 0.18,
        "re_traceability_score_global": 0.37,
        "dependency_core_score_global": 0.30,
        "response_anchor_score_global": 0.15,
    }

    v = np.zeros(len(df), dtype=float)
    ok_any = False
    for col, w in weights.items():
        if col in df.columns:
            v += w * pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
            ok_any = True

    if not ok_any:
        return np.full(len(df), np.nan)
    return v

def load_panel_score_lookup(out_root: Path):
    """
    尝试读取已有 summary 分数。
    若不存在，后面会 fallback 到 mean depstruct_score_cell。
    """
    score_file = out_root / "dependency_structure_benchmark_scores.csv"
    if score_file.exists():
        try:
            s = pd.read_csv(score_file)
            if {"dataset", "data_regime", "DepStruct"}.issubset(set(s.columns)):
                out = {}
                for _, r in s.iterrows():
                    out[(str(r["dataset"]), str(r["data_regime"]))] = float(r["DepStruct"])
                return out
        except Exception:
            pass
    return {}

def load_one_table(out_root: Path, dataset_name: str, regime: str):
    prefix = resolve_dataset_prefix(out_root, dataset_name)
    f = out_root / f"{prefix}_{regime}_cell_structure_scores_with_DepStruct.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")
    df = pd.read_csv(f)

    xcol = pick_col(df, ["umap_1", "umap1", "UMAP_1", "UMAP1"])
    ycol = pick_col(df, ["umap_2", "umap2", "UMAP_2", "UMAP2"])
    ccol = pick_col(df, ["new_leiden", "leiden", "cluster", "cluster_id"])

    if xcol is None or ycol is None:
        raise ValueError(f"UMAP columns not found in {f.name}")
    if ccol is None:
        raise ValueError(f"Cluster column not found in {f.name}")

    out = pd.DataFrame({
        "umap_1": pd.to_numeric(df[xcol], errors="coerce"),
        "umap_2": pd.to_numeric(df[ycol], errors="coerce"),
        "new_leiden": df[ccol].astype(str),
    })

    # 尝试补 DepStruct
    out["depstruct_score_cell"] = compute_depstruct_cell(df)

    # 去掉无效坐标
    out = out[np.isfinite(out["umap_1"].values) & np.isfinite(out["umap_2"].values)].copy()
    out["dataset"] = dataset_name
    out["data_regime"] = regime
    return out

def plot_single_panel(ax, tab: pd.DataFrame, panel_title: str, score_value=None):
    clusters = sorted(tab["new_leiden"].astype(str).unique().tolist(), key=cluster_sort_key)
    palette = build_palette(len(clusters))
    cl2color = {cl: palette[i] for i, cl in enumerate(clusters)}

    # 白底、水彩散点风格
    ax.set_facecolor("white")

    # 按 cluster 分层画，保留原图逻辑，但改成小点+低透明度
    for cl in clusters:
        sub = tab[tab["new_leiden"].astype(str) == str(cl)]
        ax.scatter(
            sub["umap_1"].to_numpy(),
            sub["umap_2"].to_numpy(),
            s=PLOT_DOT_SIZE,
            c=[cl2color[cl]],
            alpha=PLOT_DOT_ALPHA,
            linewidths=0,
            rasterized=True,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 等比例，避免 panel 被压扁
    ax.set_aspect("equal", adjustable="box")

    # 两行标题
    ax.text(
        0.5, 1.07, panel_title,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=15, fontweight="bold"
    )
    if score_value is not None and np.isfinite(score_value):
        ax.text(
            0.5, 0.97, f"DepStruct = {score_value:.3f}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold"
        )

# ============================================================
# 3) 主流程
# ============================================================
OUT_ROOT = detect_out_root()
print("[INFO] OUT_ROOT =", OUT_ROOT)

score_lookup = load_panel_score_lookup(OUT_ROOT)

all_tabs = {}
for ds in DATASETS:
    all_tabs[ds] = {}
    for reg in REGIME_ORDER:
        tab = load_one_table(OUT_ROOT, ds, reg)
        all_tabs[ds][reg] = tab
        if (ds, reg) not in score_lookup:
            # 如果没有已有 benchmark score，就用 mean cell-level depstruct 兜底
            vv = pd.to_numeric(tab["depstruct_score_cell"], errors="coerce").values
            score_lookup[(ds, reg)] = float(np.nanmean(vv)) if len(vv) else np.nan

# ============================================================
# 4) 画 3×3 图
# ============================================================
n_rows = len(DATASETS)
n_cols = len(REGIME_ORDER)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(FIG_W, FIG_H),
    facecolor="white",
    squeeze=False
)

for i, ds in enumerate(DATASETS):
    for j, reg in enumerate(REGIME_ORDER):
        ax = axes[i, j]
        tab = all_tabs[ds][reg]
        title = REGIME_LABEL.get(reg, reg)
        score = score_lookup.get((ds, reg), np.nan)
        plot_single_panel(ax, tab, title, score)

        # 左侧行标签
        if j == 0:
            ax.text(
                -0.10, 0.50, ds,
                transform=ax.transAxes,
                rotation=90,
                ha="center", va="center",
                fontsize=15, fontweight="bold"
            )

fig.suptitle(SUPTITLE, fontsize=16, fontweight="bold", y=0.995)

fig.text(
    0.5, 0.012,
    "Each panel is re-rendered using the existing regime-specific UMAP coordinates, with a watercolor-like scatter style for clearer manifold comparison.",
    ha="center", va="bottom", fontsize=9
)

fig.subplots_adjust(
    left=0.08, right=0.985,
    top=0.94, bottom=0.05,
    wspace=0.06, hspace=0.16
)

# ============================================================
# 5) 保存
# ============================================================
out_stem = OUT_ROOT / "dependency_structure_benchmark_umap_watercolor"

for ext in ["png", "pdf", "svg"]:
    out = out_stem.with_suffix(f".{ext}")
    fig.savefig(out, dpi=DPI if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("[SAVE]", out)

plt.show()
plt.close(fig)
