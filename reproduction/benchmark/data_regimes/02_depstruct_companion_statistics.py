# -*- coding: utf-8 -*-
"""
IMAS DepStruct non-UMAP companion figures, v2
================================================

This script keeps and improves the component-decomposition dot plot,
adds pairwise P-values/FDR labels, and replaces the other non-UMAP
figures with cleaner companion plots.

Input: outputs from the benchmark UMAP DepStruct posthoc script
    <OUT_ROOT>/<dataset>_<regime>_cell_structure_scores_with_DepStruct.csv

Main outputs:
    01_component_decomposition_with_pvalues.{png,pdf,svg}
    02_depstruct_raincloud_with_pvalues.{png,pdf,svg}
    03_component_profile_lines.{png,pdf,svg}
    04_effect_size_forest.{png,pdf,svg}
    companion_cluster_score_summary_v2.csv
    companion_pairwise_pvalues_v2.csv
    companion_effect_sizes_v2.csv

No UMAP is regenerated here.
No heatmap is generated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy import stats
    from scipy.stats import gaussian_kde
except Exception as e:
    raise RuntimeError("This script requires scipy. Please run: pip install scipy") from e

warnings.filterwarnings("ignore")

# ============================================================
# 0. USER CONFIG
# ============================================================

OUT_ROOT_CANDIDATES = [
    get_runs_root() / "frontier_transfer_hash_stageB_data_regime_comparison" / "benchmark_umap_dependency_structure_posthoc",
    Path.cwd(),
]

# Datasets to include in the companion figures.
# CRC/Glioma are read from existing posthoc outputs; no model/UMAP computation is run here.
# Osteosarcoma is added as a downstream plotting/statistics input once its posthoc
# <dataset>_<regime>_cell_structure_scores_with_DepStruct.csv files exist.
DATASETS_REQUESTED = ["CRC-Tissue", "Glioma-Tissue", "Osteosarcoma-Tissue"]
REGIMES = ["multiomic_only", "mixed", "scrna_only"]

# Allow a few common local folder/output names for the osteosarcoma run.
# The script will use the first existing dataset prefix found in OUT_ROOT.
DATASET_ALIASES = {
    "CRC-Tissue": ["CRC-Tissue"],
    "Glioma-Tissue": ["Glioma-Tissue"],
    "Osteosarcoma-Tissue": [
        "Osteosarcoma-Tissue",
        "Osteosarcoma",
        "Osteosarcoma-Cellline",
        "Osteosarcoma-CellLine",
        "OS-Tissue",
        "OS",
    ],
}

# Resolved after OUT_ROOT is detected.
DATASETS = list(DATASETS_REQUESTED)

REGIME_LABEL = {
    "multiomic_only": "Multiomic-only",
    "mixed": "Mixed",
    "scrna_only": "scRNA-only",
}

# Keep close to the colors in your current figure
REGIME_COLOR = {
    "multiomic_only": "#2ca25f",  # green
    "mixed": "#8c6bb1",           # purple
    "scrna_only": "#bdbdbd",      # grey
}

COMPONENTS = [
    ("dependency_support_mean", "Dependency support"),
    ("re_traceability_mean", "RE traceability"),
    ("dependency_core_mean", "Dependency core"),
    ("response_anchor_mean", "Response anchor"),
]

PAIRWISE = [
    ("multiomic_only", "mixed"),
    ("multiomic_only", "scrna_only"),
    ("mixed", "scrna_only"),
]

# Plot settings
POINT_ALPHA = 0.70
POINT_SIZE_BASE = 18
POINT_SIZE_SCALE = 0.85
JITTER_SCALE = 0.075
MIN_CLUSTER_FOR_STATS = 3
RNG = np.random.default_rng(42)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def log(*args):
    print(*args, flush=True)


def detect_out_root() -> Path:
    for p in OUT_ROOT_CANDIDATES:
        if p.exists() and len(list(p.glob("*_cell_structure_scores_with_DepStruct.csv"))) > 0:
            log("[FOUND OUT_ROOT]", p)
            return p
    msg = ["Cannot detect OUT_ROOT automatically. Checked:"]
    msg += [str(p) for p in OUT_ROOT_CANDIDATES]
    msg += ["Please edit OUT_ROOT_CANDIDATES at the top of this script."]
    raise FileNotFoundError("\n".join(msg))


OUT_ROOT = detect_out_root()

def _dataset_prefix_from_cell_score_file(path: Path) -> Optional[str]:
    """Return dataset prefix from <dataset>_<regime>_cell_structure_scores_with_DepStruct.csv."""
    name = path.name
    suffix = "_cell_structure_scores_with_DepStruct.csv"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    for reg in REGIMES:
        reg_suffix = f"_{reg}"
        if stem.endswith(reg_suffix):
            ds = stem[: -len(reg_suffix)]
            return ds if ds else None
    return None

def resolve_dataset_names(requested: list[str]) -> list[str]:
    """Resolve requested dataset names to prefixes that actually exist in OUT_ROOT.

    This lets the companion plotting script include Osteosarcoma without forcing a
    particular local folder/output prefix. CRC/Glioma are simply reused from their
    existing CSVs; no model computation is performed in this script.
    """
    existing = []
    for f in sorted(OUT_ROOT.glob("*_cell_structure_scores_with_DepStruct.csv")):
        ds = _dataset_prefix_from_cell_score_file(f)
        if ds is not None and ds not in existing:
            existing.append(ds)
    existing_lower = {x.lower(): x for x in existing}

    resolved = []
    for req in requested:
        aliases = DATASET_ALIASES.get(req, [req])
        hit = None
        for cand in aliases:
            if cand in existing:
                hit = cand
                break
            if cand.lower() in existing_lower:
                hit = existing_lower[cand.lower()]
                break
        if hit is None:
            log(f"[WARN dataset missing] requested={req}; aliases={aliases}; available={existing}")
            # Keep the requested name in DATASETS so the missing files are reported clearly.
            hit = req
        if hit not in resolved:
            resolved.append(hit)
    return resolved

DATASETS = resolve_dataset_names(DATASETS_REQUESTED)
log("[DATASETS for companion figures]", DATASETS)


def savefig_multi(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=450 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        log("[SAVE]", out)
    plt.close(fig)


def pick_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def jitter(n: int, scale: float = JITTER_SCALE) -> np.ndarray:
    return RNG.uniform(-scale, scale, int(n))


def nice_dataset_label(x: str) -> str:
    return str(x).replace("-Tissue", "")


def safe_mwu(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < MIN_CLUSTER_FOR_STATS or y.size < MIN_CLUSTER_FOR_STATS:
        return np.nan
    try:
        return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[valid] = tmp
    return out


def p_to_star(q: float) -> str:
    if not np.isfinite(q):
        return "n.s."
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return "n.s."


def format_q(q: float) -> str:
    if not np.isfinite(q):
        return "q=NA"
    if q < 1e-4:
        return f"q={q:.1e}"
    return f"q={q:.3f}"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    # Efficient enough for cluster-level vectors.
    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    return float((gt - lt) / (x.size * y.size))


def bootstrap_median_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan
    diffs = []
    for _ in range(int(n_boot)):
        xb = RNG.choice(x, size=x.size, replace=True)
        yb = RNG.choice(y, size=y.size, replace=True)
        diffs.append(np.nanmedian(xb) - np.nanmedian(yb))
    diffs = np.asarray(diffs, dtype=float)
    return float(np.nanmedian(x) - np.nanmedian(y)), float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


# ============================================================
# 2. LOAD CELL TABLES AND AGGREGATE TO CLUSTER LEVEL
# ============================================================

def load_all_cell_tables() -> pd.DataFrame:
    tables = []
    for ds in DATASETS:
        for reg in REGIMES:
            f = OUT_ROOT / f"{ds}_{reg}_cell_structure_scores_with_DepStruct.csv"
            if not f.exists():
                log("[WARN missing]", f)
                continue
            df = pd.read_csv(f)
            df["dataset"] = ds
            df["data_regime"] = reg
            tables.append(df)
            log("[LOAD]", f, "rows=", len(df))
    if not tables:
        raise RuntimeError(f"No *_cell_structure_scores_with_DepStruct.csv files found for DATASETS={DATASETS} in OUT_ROOT={OUT_ROOT}.")
    return pd.concat(tables, axis=0, ignore_index=True)


def aggregate_cluster_scores(cell_df: pd.DataFrame) -> pd.DataFrame:
    cluster_col = pick_col(cell_df, ["new_leiden", "leiden", "cluster"])
    dep_col = pick_col(cell_df, ["depstruct_score_cell", "DepStruct", "depstruct_score"])
    dep_support_col = pick_col(cell_df, ["dependency_support_score_global", "dependency_support_score"])
    re_col = pick_col(cell_df, ["re_traceability_score_global", "re_traceability_score"])
    core_col = pick_col(cell_df, ["dependency_core_score_global", "dependency_core_score"])
    resp_col = pick_col(cell_df, ["response_anchor_score_global", "response_anchor_score"])

    required = {
        "new_leiden": cluster_col,
        "depstruct_mean": dep_col,
        "dependency_support_mean": dep_support_col,
        "re_traceability_mean": re_col,
        "dependency_core_mean": core_col,
        "response_anchor_mean": resp_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns in cell tables: {missing}\nColumns={cell_df.columns.tolist()}")

    work = cell_df.copy()
    work["new_leiden"] = work[cluster_col].astype(str)
    agg = (
        work.groupby(["dataset", "data_regime", "new_leiden"], as_index=False)
        .agg(
            n_cells=("new_leiden", "size"),
            depstruct_mean=(dep_col, "mean"),
            depstruct_median=(dep_col, "median"),
            dependency_support_mean=(dep_support_col, "mean"),
            re_traceability_mean=(re_col, "mean"),
            dependency_core_mean=(core_col, "mean"),
            response_anchor_mean=(resp_col, "mean"),
        )
    )
    agg["depstruct_rank"] = (
        agg.groupby(["dataset", "data_regime"])["depstruct_mean"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    return agg


cell_df = load_all_cell_tables()
cluster_df = aggregate_cluster_scores(cell_df)
cluster_df.to_csv(OUT_ROOT / "companion_cluster_score_summary_v2.csv", index=False)
log("[SAVE]", OUT_ROOT / "companion_cluster_score_summary_v2.csv")


# ============================================================
# 3. STATS TABLES
# ============================================================

def compute_pairwise_stats(cluster_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [("depstruct_mean", "DepStruct")] + COMPONENTS
    for ds in DATASETS:
        for metric, metric_label in metrics:
            for a, b in PAIRWISE:
                xa = cluster_df[(cluster_df["dataset"] == ds) & (cluster_df["data_regime"] == a)][metric].values
                xb = cluster_df[(cluster_df["dataset"] == ds) & (cluster_df["data_regime"] == b)][metric].values
                p = safe_mwu(xa, xb)
                med_diff, ci_lo, ci_hi = bootstrap_median_diff(xa, xb)
                rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "metric_label": metric_label,
                    "group_a": a,
                    "group_b": b,
                    "group_a_label": REGIME_LABEL[a],
                    "group_b_label": REGIME_LABEL[b],
                    "n_a": int(np.isfinite(xa).sum()),
                    "n_b": int(np.isfinite(xb).sum()),
                    "median_a": float(np.nanmedian(xa)) if np.isfinite(xa).any() else np.nan,
                    "median_b": float(np.nanmedian(xb)) if np.isfinite(xb).any() else np.nan,
                    "median_diff_a_minus_b": med_diff,
                    "median_diff_ci95_low": ci_lo,
                    "median_diff_ci95_high": ci_hi,
                    "cliffs_delta_a_vs_b": cliffs_delta(xa, xb),
                    "p_mannwhitney": p,
                })
    stat_df = pd.DataFrame(rows)
    # FDR over all pairwise tests in this companion analysis.
    stat_df["q_bh_global"] = bh_fdr(stat_df["p_mannwhitney"].values)
    # FDR within each dataset + metric.
    stat_df["q_bh_within_metric"] = np.nan
    for (ds, metric), idx in stat_df.groupby(["dataset", "metric"]).groups.items():
        idx = list(idx)
        stat_df.loc[idx, "q_bh_within_metric"] = bh_fdr(stat_df.loc[idx, "p_mannwhitney"].values)
    stat_df["stars"] = [p_to_star(q) for q in stat_df["q_bh_within_metric"].values]
    return stat_df


pairwise_stats = compute_pairwise_stats(cluster_df)
pairwise_stats.to_csv(OUT_ROOT / "companion_pairwise_pvalues_v2.csv", index=False)
log("[SAVE]", OUT_ROOT / "companion_pairwise_pvalues_v2.csv")


def get_q_for(dataset: str, metric: str, a: str, b: str) -> tuple[float, str]:
    sub = pairwise_stats[
        (pairwise_stats["dataset"] == dataset)
        & (pairwise_stats["metric"] == metric)
        & (pairwise_stats["group_a"] == a)
        & (pairwise_stats["group_b"] == b)
    ]
    if sub.empty:
        return np.nan, "n.s."
    q = float(sub.iloc[0]["q_bh_within_metric"])
    return q, p_to_star(q)


# ============================================================
# 4. PLOT HELPERS
# ============================================================

def add_sig_bracket(ax, x1: float, x2: float, y: float, h: float, text: str, color: str = "#222222"):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.75, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=7.5, color=color)


def draw_group_dots(ax, values: np.ndarray, x0: float, color: str, sizes: Optional[np.ndarray] = None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if sizes is None:
        s = np.full(vals.size, 28.0)
    else:
        s = np.asarray(sizes, dtype=float)
        if s.size != vals.size:
            s = np.full(vals.size, 28.0)
    x = np.full(vals.size, x0) + jitter(vals.size)
    ax.scatter(x, vals, s=s, color=color, alpha=POINT_ALPHA, edgecolor="white", linewidth=0.45, zorder=3)
    med = float(np.nanmedian(vals))
    ax.plot([x0 - 0.16, x0 + 0.16], [med, med], color="black", lw=1.65, zorder=5)


def auto_ylim_with_brackets(ax, ys: list[float], base_min: float = 0.0):
    finite = [float(y) for y in ys if np.isfinite(y)]
    ymax = max(finite) if finite else 1.0
    ymin = min(finite) if finite else base_min
    pad = max(0.06, (ymax - ymin) * 0.22)
    ax.set_ylim(min(base_min, ymin - pad * 0.25), ymax + pad)
    return ymax, pad


# ============================================================
# 5. FIGURE 1: COMPONENT DECOMPOSITION WITH P-VALUES
# ============================================================

def plot_component_decomposition_with_pvalues():
    fig, axes = plt.subplots(
        len(DATASETS),
        len(COMPONENTS),
        figsize=(15.6, max(7.5, 3.65 * len(DATASETS))),
        facecolor="white",
        squeeze=False,
    )

    x_pos = {r: i for i, r in enumerate(REGIMES)}

    for i, ds in enumerate(DATASETS):
        for j, (metric, label) in enumerate(COMPONENTS):
            ax = axes[i, j]
            ds_sub = cluster_df[cluster_df["dataset"] == ds]
            all_y = []
            for reg in REGIMES:
                ss = ds_sub[ds_sub["data_regime"] == reg]
                vals = ss[metric].values.astype(float)
                all_y.extend(vals[np.isfinite(vals)].tolist())
                sizes = POINT_SIZE_BASE + POINT_SIZE_SCALE * np.sqrt(ss["n_cells"].values.astype(float))
                draw_group_dots(ax, vals, x_pos[reg], REGIME_COLOR[reg], sizes=sizes)

            ymax, pad = auto_ylim_with_brackets(ax, all_y)
            h = pad * 0.10
            # Brackets: show all three comparisons as stars; full p/q in CSV.
            y0 = ymax + pad * 0.05
            for k, (a, b) in enumerate(PAIRWISE):
                _, star = get_q_for(ds, metric, a, b)
                add_sig_bracket(ax, x_pos[a], x_pos[b], y0 + k * pad * 0.23, h, star)

            ax.set_xticks([x_pos[r] for r in REGIMES])
            ax.set_xticklabels([REGIME_LABEL[r] for r in REGIMES], rotation=35, ha="right")
            ax.set_title(label, fontweight="bold", pad=7)
            ax.grid(axis="y", alpha=0.16, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if j == 0:
                ax.set_ylabel(nice_dataset_label(ds), fontweight="bold", fontsize=12)
            else:
                ax.set_ylabel("")

    fig.suptitle("Cluster-level decomposition of DepStruct components", fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.015,
        "Dots denote newly computed Leiden clusters. Black lines show medians. Brackets show Mann–Whitney tests with BH-FDR within each dataset/component: * q<0.05, ** q<0.01, *** q<0.001, **** q<1e-4.",
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )
    savefig_multi(fig, OUT_ROOT / "01_component_decomposition_with_pvalues")


plot_component_decomposition_with_pvalues()


# ============================================================
# 6. FIGURE 2: DEPSTRUCT RAINCLOUD WITH P-VALUES
# ============================================================

def draw_half_violin(ax, vals: np.ndarray, x0: float, color: str, width: float = 0.33):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3 or np.nanstd(vals) < 1e-8:
        return
    y_grid = np.linspace(max(0.0, vals.min() - 0.03), min(1.05, vals.max() + 0.03), 160)
    try:
        kde = gaussian_kde(vals)
        dens = kde(y_grid)
    except Exception:
        return
    dens = dens / (dens.max() + 1e-12) * width
    ax.fill_betweenx(y_grid, x0, x0 + dens, color=color, alpha=0.23, lw=0, zorder=1)
    ax.plot(x0 + dens, y_grid, color=color, alpha=0.65, lw=0.8, zorder=2)


def plot_depstruct_raincloud():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(max(11.4, 4.4 * len(DATASETS)), 5.1), facecolor="white", squeeze=False)
    axes = axes[0]
    x_pos = {r: i for i, r in enumerate(REGIMES)}

    for ax, ds in zip(axes, DATASETS):
        ds_sub = cluster_df[cluster_df["dataset"] == ds]
        all_y = []
        for reg in REGIMES:
            ss = ds_sub[ds_sub["data_regime"] == reg]
            vals = ss["depstruct_mean"].values.astype(float)
            all_y.extend(vals[np.isfinite(vals)].tolist())
            x0 = x_pos[reg]
            draw_half_violin(ax, vals, x0, REGIME_COLOR[reg])
            sizes = POINT_SIZE_BASE + POINT_SIZE_SCALE * np.sqrt(ss["n_cells"].values.astype(float))
            # Put dots slightly left of the half-violin.
            x = np.full(len(vals), x0 - 0.08) + jitter(len(vals), scale=0.055)
            ax.scatter(x, vals, s=sizes, color=REGIME_COLOR[reg], alpha=0.72, edgecolor="white", linewidth=0.45, zorder=4)
            med = np.nanmedian(vals)
            ax.plot([x0 - 0.18, x0 + 0.18], [med, med], color="black", lw=1.8, zorder=6)

        ymax, pad = auto_ylim_with_brackets(ax, all_y)
        h = pad * 0.10
        y0 = ymax + pad * 0.05
        for k, (a, b) in enumerate(PAIRWISE):
            _, star = get_q_for(ds, "depstruct_mean", a, b)
            add_sig_bracket(ax, x_pos[a], x_pos[b], y0 + k * pad * 0.23, h, star)

        ax.set_xticks([x_pos[r] for r in REGIMES])
        ax.set_xticklabels([REGIME_LABEL[r] for r in REGIMES], rotation=25, ha="right")
        ax.set_ylabel("Cluster-level DepStruct")
        ax.set_title(nice_dataset_label(ds), fontweight="bold")
        ax.grid(axis="y", alpha=0.16, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Cluster-level DepStruct distributions", fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "Half-violins show density; dots show clusters; black lines show medians. Significance notation as in Fig. 1.", ha="center", fontsize=8)
    savefig_multi(fig, OUT_ROOT / "02_depstruct_raincloud_with_pvalues")


plot_depstruct_raincloud()


# ============================================================
# 7. FIGURE 3: COMPONENT PROFILE LINES
# ============================================================

def bootstrap_ci_mean(vals: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    boots = []
    for _ in range(int(n_boot)):
        boots.append(np.nanmean(RNG.choice(vals, size=vals.size, replace=True)))
    boots = np.asarray(boots, dtype=float)
    return float(np.nanmean(vals)), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def plot_component_profile_lines():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(max(12.2, 4.5 * len(DATASETS)), 4.8), facecolor="white", squeeze=False)
    axes = axes[0]
    x = np.arange(len(COMPONENTS))

    for ax, ds in zip(axes, DATASETS):
        ds_sub = cluster_df[cluster_df["dataset"] == ds]
        for reg in REGIMES:
            means = []
            lows = []
            highs = []
            for metric, _ in COMPONENTS:
                vals = ds_sub[ds_sub["data_regime"] == reg][metric].values.astype(float)
                m, lo, hi = bootstrap_ci_mean(vals)
                means.append(m)
                lows.append(lo)
                highs.append(hi)
            means = np.asarray(means)
            lows = np.asarray(lows)
            highs = np.asarray(highs)
            ax.plot(x, means, marker="o", lw=2.0, ms=5.5, color=REGIME_COLOR[reg], label=REGIME_LABEL[reg])
            ax.fill_between(x, lows, highs, color=REGIME_COLOR[reg], alpha=0.14, lw=0)

        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in COMPONENTS], rotation=25, ha="right")
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("Mean cluster-level component score")
        ax.set_title(nice_dataset_label(ds), fontweight="bold")
        ax.grid(axis="y", alpha=0.16, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("DepStruct component profile across data regimes", fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "Lines show cluster-level means; shaded bands show bootstrap 95% CIs.", ha="center", fontsize=8)
    savefig_multi(fig, OUT_ROOT / "03_component_profile_lines")


plot_component_profile_lines()


# ============================================================
# 8. FIGURE 4: EFFECT-SIZE FOREST
# ============================================================

def build_effect_size_table() -> pd.DataFrame:
    rows = []
    comparisons = [("multiomic_only", "mixed"), ("multiomic_only", "scrna_only")]
    metrics = [("depstruct_mean", "DepStruct")] + COMPONENTS
    for ds in DATASETS:
        for metric, metric_label in metrics:
            for a, b in comparisons:
                xa = cluster_df[(cluster_df["dataset"] == ds) & (cluster_df["data_regime"] == a)][metric].values
                xb = cluster_df[(cluster_df["dataset"] == ds) & (cluster_df["data_regime"] == b)][metric].values
                med_diff, lo, hi = bootstrap_median_diff(xa, xb)
                p = safe_mwu(xa, xb)
                rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "metric_label": metric_label,
                    "comparison": f"{REGIME_LABEL[a]} − {REGIME_LABEL[b]}",
                    "group_a": a,
                    "group_b": b,
                    "median_diff": med_diff,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "cliffs_delta": cliffs_delta(xa, xb),
                    "p_mannwhitney": p,
                })
    out = pd.DataFrame(rows)
    out["q_bh_global"] = bh_fdr(out["p_mannwhitney"].values)
    out["stars"] = [p_to_star(q) for q in out["q_bh_global"].values]
    return out


effect_df = build_effect_size_table()
effect_df.to_csv(OUT_ROOT / "companion_effect_sizes_v2.csv", index=False)
log("[SAVE]", OUT_ROOT / "companion_effect_sizes_v2.csv")


def plot_effect_size_forest():
    metrics = ["depstruct_mean"] + [m for m, _ in COMPONENTS]
    metric_labels = {"depstruct_mean": "DepStruct"}
    metric_labels.update({m: lab for m, lab in COMPONENTS})
    comp_order = ["Multiomic-only − Mixed", "Multiomic-only − scRNA-only"]
    comp_color = {
        "Multiomic-only − Mixed": "#3182bd",
        "Multiomic-only − scRNA-only": "#de2d26",
    }

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(max(12.5, 4.7 * len(DATASETS)), 5.3), facecolor="white", squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, DATASETS):
        sub = effect_df[effect_df["dataset"] == ds]
        y_base = np.arange(len(metrics))[::-1]
        offset = {comp_order[0]: 0.13, comp_order[1]: -0.13}

        for comp in comp_order:
            ss = sub[sub["comparison"] == comp]
            for yi, metric in zip(y_base, metrics):
                r = ss[ss["metric"] == metric]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = yi + offset[comp]
                x = float(r["median_diff"])
                lo = float(r["ci95_low"])
                hi = float(r["ci95_high"])
                col = comp_color[comp]
                ax.plot([lo, hi], [y, y], color=col, lw=1.5, alpha=0.85)
                ax.scatter([x], [y], s=52, color=col, edgecolor="white", linewidth=0.6, zorder=5)
                ax.text(hi + 0.012, y, str(r["stars"]), va="center", fontsize=8, color=col)

        ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.65)
        ax.set_yticks(y_base)
        ax.set_yticklabels([metric_labels[m] for m in metrics])
        ax.set_xlabel("Median difference in cluster-level score")
        ax.set_title(nice_dataset_label(ds), fontweight="bold")
        ax.grid(axis="x", alpha=0.16, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [Line2D([0], [0], marker="o", color=comp_color[c], label=c, lw=1.8, markersize=6) for c in comp_order]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Effect-size summary of multiomic constraint advantage", fontweight="bold", y=1.10)
    fig.text(0.5, -0.02, "Points show median differences; horizontal lines show bootstrap 95% CIs; stars show BH-FDR across effect-size tests.", ha="center", fontsize=8)
    savefig_multi(fig, OUT_ROOT / "04_effect_size_forest")


plot_effect_size_forest()

log("\n[DONE] Non-UMAP companion figures v2 generated.")
log("[OUT_ROOT]", OUT_ROOT)
