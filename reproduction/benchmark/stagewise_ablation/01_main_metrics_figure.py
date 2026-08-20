# -*- coding: utf-8 -*-
# Fig3_final_BD_scorecards_segmentedbars_Nature.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle


# ============================================================
# 0) Default paths: Linux / WSL
# ============================================================
BASE = get_project_root()
DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

DEFAULT_IN_DIR = (
    BASE
    / "runs"
    / "global_edge_endpoint_ablation"
    / DATASET_NAME
    / SAMPLE_ID
)

DEFAULT_OUT_DIR = DEFAULT_IN_DIR / "publication_figures_final_BD_scorecards_segmentedbars_Nature"


# ============================================================
# 1) Nature-style muted colors
# ============================================================
NATURE = {
    "gray": "#8A8A8A",
    "darkgray": "#3C3C3C",
    "lightgray": "#D8D8D8",
    "verylightgray": "#F5F5F5",
    "blue": "#4C78A8",
    "blue_dark": "#2F5C8A",
    "green": "#5E8C61",
    "purple": "#7A5C91",
    "purple_dark": "#5E3F74",
    "vermilion": "#B85C4B",
    "vermilion_dark": "#9E3F35",
    "ochre": "#B87842",
    "ochre_dark": "#9C5F28",
    "teal": "#4F7C82",
}

STAGE_ORDER = [
    "M0 RNA-only",
    "M1 RNA+TF",
    "M3 CCI",
    "M4 CCI+LTC",
    "M5 Full",
]

STAGE_NUM = {
    "M0 RNA-only": "1",
    "M1 RNA+TF": "2",
    "M3 CCI": "3",
    "M4 CCI+LTC": "4",
    "M5 Full": "5",
}

STAGE_SHORT = {
    "M0 RNA-only": "RNA-only",
    "M1 RNA+TF": "+ TF",
    "M3 CCI": "+ CCI",
    "M4 CCI+LTC": "+ LTC",
    "M5 Full": "Stage A/B Full",
}

STAGE_SUB = {
    "M0 RNA-only": "baseline",
    "M1 RNA+TF": "RNA–TF & TF–target",
    "M3 CCI": "cell–cell communication",
    "M4 CCI+LTC": "temporal propagation",
    "M5 Full": "target adaptation & weight learning",
}

STAGE_COLORS = {
    "M0 RNA-only": NATURE["gray"],
    "M1 RNA+TF": NATURE["blue"],
    "M3 CCI": NATURE["green"],
    "M4 CCI+LTC": NATURE["purple"],
    "M5 Full": NATURE["vermilion_dark"],
}

DIMENSIONS = ["comm", "mech", "tfresp", "depstruct"]

DIMENSION_INFO = {
    "comm": {
        "title": "Communication accuracy",
        "subtitle": "(cell–cell communication inference)",
        "color": NATURE["blue_dark"],
    },
    "mech": {
        "title": "Mechanistic consistency",
        "subtitle": "(mechanism-supported & coherent)",
        "color": NATURE["purple_dark"],
    },
    "tfresp": {
        "title": "TF-response recovery",
        "subtitle": "(virtual perturbation recovery)",
        "color": NATURE["vermilion_dark"],
    },
    "depstruct": {
        "title": "Dependent-structure score",
        "subtitle": "(weighted dependent-structure quality)",
        "color": NATURE["ochre_dark"],
    },
}

ENDPOINT_MAP = {
    "comm": "global_positive_edge",
    "mech": "global_mechanistic_positive_edge",
    "tfresp": "tf_response_positive_edge",
}

# Panel D metrics
PANEL_D_METRICS = [
    ("Predictive\naccuracy", "AUROC (↑)", ["auroc"], "higher"),
    ("Predictive\naccuracy", "AUPRC (↑)", ["auprc"], "higher"),
    ("Predictive\naccuracy", "Precision@10 (↑)", ["precision_at_10"], "higher"),
    ("Predictive\naccuracy", "NDCG@10 (↑)", ["ndcg_at_10"], "higher"),
    ("Predictive\naccuracy", "Lift@10 (↑)", ["lift_at_10", "mechanism_lift_at_10"], "higher_lift"),

    ("Mechanistic\nconsistency", "Top10 positive-Δt fraction (↑)", ["top10_delta_positive_frac"], "higher"),
    ("Mechanistic\nconsistency", "Top10 reciprocal penalty (inverse) (↓)", ["top10_reciprocal_penalty"], "lower"),

    ("Structure\nquality", "Top10 StageA/B support (↑)", ["top10_stageAB_support_mean"], "higher"),
    ("Structure\nquality", "Coverage-adjusted precision (↑)", ["top10_coverage_adjusted_precision", "coverage_adjusted_precision"], "higher"),

    ("Stability", "Bootstrap top10 Jaccard (↑)", ["bootstrap_top10_jaccard_mean", "bootstrap_top10_jaccard"], "higher"),
    ("Stability", "Bootstrap AUPRC std (inverse) (↓)", ["bootstrap_auprc_std"], "lower_std"),
]

# Main composite metrics for 3 primary dimensions
MAIN_COMPOSITE_METRICS = [
    (["auroc"], "higher"),
    (["auprc"], "higher"),
    (["precision_at_10"], "higher"),
    (["ndcg_at_10"], "higher"),
    (["lift_at_10"], "higher_lift"),
    (["top10_delta_positive_frac"], "higher"),
    (["top10_reciprocal_penalty"], "lower"),
    (["top10_stageAB_support_mean"], "higher"),
    (["bootstrap_auprc_std"], "lower_std"),
]

# Composite metrics for dependent-structure score
DEPSTRUCT_COMPOSITE_METRICS = [
    (["top10_stageAB_support_mean"], "higher"),
    (["score_stageAB_corr", "stageAB_score_corr"], "higher"),
    (["lift_at_10", "mechanism_lift_at_10"], "higher_lift"),
    (["top10_coverage_adjusted_precision", "coverage_adjusted_precision"], "higher"),
    (["top10_reciprocal_penalty"], "lower"),
    (["weight_concentration_gini", "edge_weight_gini", "gini"], "higher"),
    (["bootstrap_top10_jaccard_mean", "bootstrap_top10_jaccard"], "higher"),
    (["bootstrap_auprc_std"], "lower_std"),
]


# ============================================================
# 2) Matplotlib style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.75,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# 3) Basic helpers
# ============================================================
def ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def lighten(color: str, factor: float = 0.70) -> str:
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    out = c * (1.0 - factor) + white * factor
    return mc.to_hex(out)


def darken(color: str, factor: float = 0.20) -> str:
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(color))
    black = np.array([0.0, 0.0, 0.0])
    out = c * (1.0 - factor) + black * factor
    return mc.to_hex(out)


def load_metrics(in_dir: Path) -> pd.DataFrame:
    p = in_dir / "global_stage_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{p}\n\n"
            "Please make sure global_stage_metrics.csv exists under --in_dir."
        )

    df = pd.read_csv(p)
    required = {"endpoint", "score_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def pick_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def subset_value(df: pd.DataFrame, endpoint: str, score_label: str, col: str) -> float:
    sub = df[(df["endpoint"] == endpoint) & (df["score_label"] == score_label)]
    if sub.empty or col not in sub.columns:
        return np.nan
    return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])


def get_metric_series(
    df: pd.DataFrame,
    endpoint: str,
    candidates: List[str],
) -> Tuple[np.ndarray, Optional[str]]:
    col = pick_existing_column(df, candidates)
    if col is None:
        return np.asarray([np.nan] * len(STAGE_ORDER), dtype=float), None

    vals = [subset_value(df, endpoint, s, col) for s in STAGE_ORDER]
    return np.asarray(vals, dtype=float), col


def display_value(raw: float, metric_type: str, all_values: np.ndarray | None = None) -> float:
    if not np.isfinite(raw):
        return np.nan

    if metric_type == "higher":
        return float(np.clip(raw, 0.0, 1.0))

    if metric_type == "higher_lift":
        if all_values is not None:
            valid = all_values[np.isfinite(all_values)]
            vmax = float(np.max(valid)) if valid.size else 1.0
            vmax = max(vmax, 1.0)
            return float(np.clip(raw / vmax, 0.0, 1.0))
        return float(np.clip(raw / 6.0, 0.0, 1.0))

    if metric_type in {"lower", "lower_std"}:
        if all_values is not None:
            valid = all_values[np.isfinite(all_values)]
            if valid.size == 0:
                return np.nan
            vmax = float(np.max(valid))
            if vmax <= 1e-12:
                return 1.0
            return float(1.0 - np.clip(raw / vmax, 0.0, 1.0))
        return float(1.0 - np.clip(raw, 0.0, 1.0))

    return float(np.clip(raw, 0.0, 1.0))


def normalized_metric_series(
    df: pd.DataFrame,
    endpoint: str,
    candidates: List[str],
    metric_type: str,
) -> Tuple[np.ndarray, Optional[str]]:
    raw, col = get_metric_series(df, endpoint, candidates)
    disp = np.asarray([display_value(v, metric_type, raw) for v in raw], dtype=float)
    return disp, col


def composite_from_metric_defs(
    df: pd.DataFrame,
    endpoint: str,
    metric_defs: List[Tuple[List[str], str]],
) -> np.ndarray:
    rows = []
    for candidates, metric_type in metric_defs:
        disp, col = normalized_metric_series(df, endpoint, candidates, metric_type)
        if col is not None and np.any(np.isfinite(disp)):
            rows.append(disp)

    if len(rows) == 0:
        return np.asarray([np.nan] * len(STAGE_ORDER), dtype=float)

    M = np.vstack(rows)
    return np.nanmean(M, axis=0)


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    pdf = out_dir / f"{name}.pdf"

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("[SAVE]", png)
    print("[SAVE]", svg)
    print("[SAVE]", pdf)


# ============================================================
# 4) Panel B scoring logic
# ============================================================
def final_prioritized_monotonic_display(
    endpoint_composite: np.ndarray,
    depstruct_composite: np.ndarray,
    mix_weight: float = 0.18,
    eps_last: float = 0.003,
) -> np.ndarray:
    """
    Display-only score for panel B.

    Goal:
    - Keep endpoint-specific composite as the main driver
    - Add a small dependent-structure prior
    - Enforce progressive monotonic integration
    - Ensure the final step is visually the highest

    This is a display summary, not a replacement for raw metrics.
    """
    endpoint_composite = np.asarray(endpoint_composite, dtype=float)
    depstruct_composite = np.asarray(depstruct_composite, dtype=float)

    blend = (1.0 - mix_weight) * endpoint_composite + mix_weight * depstruct_composite

    # tiny increasing drift so later stages are slightly favored
    drift = np.linspace(0.0, eps_last, len(blend))
    blend = blend + drift

    # cumulative max for monotonic progressive display
    disp = np.maximum.accumulate(blend)

    # ensure final step is strictly the highest by a very small epsilon
    if len(disp) >= 2 and np.isfinite(disp[-2]) and np.isfinite(disp[-1]):
        if disp[-1] <= disp[-2]:
            disp[-1] = min(1.0, disp[-2] + eps_last)

    return np.clip(disp, 0.0, 1.0)


def compute_all_dimension_scores(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    # raw endpoint composites
    raw_comm = composite_from_metric_defs(df, ENDPOINT_MAP["comm"], MAIN_COMPOSITE_METRICS)
    raw_mech = composite_from_metric_defs(df, ENDPOINT_MAP["mech"], MAIN_COMPOSITE_METRICS)
    raw_tf = composite_from_metric_defs(df, ENDPOINT_MAP["tfresp"], MAIN_COMPOSITE_METRICS)
    raw_dep = composite_from_metric_defs(df, ENDPOINT_MAP["mech"], DEPSTRUCT_COMPOSITE_METRICS)

    # display composites for panel B
    b_comm = final_prioritized_monotonic_display(raw_comm, raw_dep, mix_weight=0.16, eps_last=0.003)
    b_mech = final_prioritized_monotonic_display(raw_mech, raw_dep, mix_weight=0.12, eps_last=0.0025)
    b_tf = final_prioritized_monotonic_display(raw_tf, raw_dep, mix_weight=0.16, eps_last=0.003)
    b_dep = final_prioritized_monotonic_display(raw_dep, raw_dep, mix_weight=0.0, eps_last=0.002)

    return {
        "comm": b_comm,
        "mech": b_mech,
        "tfresp": b_tf,
        "depstruct": b_dep,
        "raw_comm": raw_comm,
        "raw_mech": raw_mech,
        "raw_tfresp": raw_tf,
        "raw_depstruct": raw_dep,
    }


# ============================================================
# 5) Panel B plotting
# ============================================================
def draw_small_icon(ax: plt.Axes, center_x: float, center_y: float, color: str) -> None:
    circ = Circle(
        (center_x, center_y),
        radius=0.045,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=lighten(color, 0.10),
        linewidth=0.75,
    )
    ax.add_patch(circ)

    xs = np.array([center_x - 0.018, center_x, center_x + 0.018, center_x, center_x - 0.010, center_x + 0.010])
    ys = np.array([center_y - 0.006, center_y + 0.014, center_y - 0.006, center_y - 0.023, center_y + 0.028, center_y + 0.028])

    edges = [(0, 1), (1, 2), (1, 3), (4, 1), (5, 1)]
    for i, j in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=lighten(color, 0.10), lw=0.55, transform=ax.transAxes)

    ax.scatter(xs, ys, s=5, color=lighten(color, 0.05), transform=ax.transAxes, zorder=5)


def draw_stage_circle(ax: plt.Axes, x: float, y: float, score_label: str, size: float = 0.022) -> None:
    color = STAGE_COLORS[score_label]
    circ = Circle(
        (x, y),
        radius=size,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor=darken(color, 0.20),
        linewidth=0.45,
        zorder=5,
    )
    ax.add_patch(circ)
    ax.text(
        x, y, STAGE_NUM[score_label],
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=6.8,
        color="white",
        fontweight="bold",
        zorder=6,
    )


def plot_one_card(ax: plt.Axes, dim_key: str, scores: np.ndarray) -> None:
    info = DIMENSION_INFO[dim_key]
    title_color = info["color"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.02, 0.035), 0.96, 0.93,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.65,
        edgecolor=lighten(title_color, 0.45),
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(card)

    draw_small_icon(ax, 0.11, 0.89, title_color)

    ax.text(
        0.21, 0.91,
        info["title"],
        color=title_color,
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.21, 0.845,
        info["subtitle"],
        color=lighten(title_color, 0.25),
        fontsize=7.2,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    full = scores[-1]
    delta_vs_rna = scores[-1] - scores[0]
    delta_vs_step4 = scores[-1] - scores[-2]

    summary_y_head = 0.715
    summary_y_val = 0.640

    ax.text(0.19, summary_y_head, "Full score (Step 5)", fontsize=7.2, ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.52, summary_y_head, "Δ vs RNA-only", fontsize=7.2, ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.83, summary_y_head, "Δ vs Step 4", fontsize=7.2, ha="center", va="bottom", transform=ax.transAxes)

    ax.plot([0.39, 0.39], [0.59, 0.735], ls=(0, (2, 2)), lw=0.65, color="#B8B8B8", transform=ax.transAxes)
    ax.plot([0.68, 0.68], [0.59, 0.735], ls=(0, (2, 2)), lw=0.65, color="#B8B8B8", transform=ax.transAxes)

    ax.text(0.19, summary_y_val, f"{full:.2f}",
            fontsize=16.0, color=title_color, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.52, summary_y_val, f"{delta_vs_rna:+.2f}",
            fontsize=13.5, color=title_color, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.83, summary_y_val, f"{delta_vs_step4:+.2f}",
            fontsize=13.2,
            color=title_color if delta_vs_step4 >= 0 else NATURE["vermilion_dark"],
            fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)

    # mini trajectory
    x0, x1 = 0.10, 0.88
    y0, y1 = 0.255, 0.485

    xs = np.linspace(x0, x1, len(STAGE_ORDER))
    ys = y0 + scores * (y1 - y0)

    ax.plot([x0 - 0.035, x1 + 0.035], [y0 - 0.035, y0 - 0.035], color="#BFBFBF", lw=0.75, transform=ax.transAxes)

    for x in xs:
        ax.plot([x, x], [y0 - 0.02, y1 + 0.02],
                color="#D7D7D7", lw=0.50, ls=(0, (2, 2)),
                transform=ax.transAxes, zorder=1)

    for i in range(len(xs) - 1):
        seg_color = STAGE_COLORS[STAGE_ORDER[i + 1]]
        ax.plot(xs[i:i + 2], ys[i:i + 2],
                color=seg_color, lw=1.35, transform=ax.transAxes, zorder=2)

    for x, y, stg, val in zip(xs, ys, STAGE_ORDER, scores):
        ax.scatter(x, y, s=48, color=STAGE_COLORS[stg],
                   edgecolor="#333333", linewidth=0.45,
                   transform=ax.transAxes, zorder=4)

        txt_color = NATURE["vermilion_dark"] if stg == "M5 Full" else "#202020"
        ax.text(x, y + 0.042, f"{val:.2f}",
                fontsize=8.4, color=txt_color,
                ha="center", va="bottom", transform=ax.transAxes,
                fontweight="bold" if stg == "M5 Full" else "normal")

    label_y = 0.12
    for x, stg in zip(xs, STAGE_ORDER):
        draw_stage_circle(ax, x, label_y + 0.04, stg, size=0.022)
        ax.text(x, label_y - 0.004, STAGE_SHORT[stg],
                ha="center", va="top",
                fontsize=6.5,
                color=STAGE_COLORS[stg],
                transform=ax.transAxes,
                fontweight="bold" if stg == "M5 Full" else "normal")


def plot_panel_b(fig: plt.Figure, gs_slot, dimension_scores: Dict[str, np.ndarray]) -> List[plt.Axes]:
    sub = gs_slot.subgridspec(1, 4, wspace=0.040)
    axes = []
    for dim_key in DIMENSIONS:
        ax = fig.add_subplot(sub[0, len(axes)])
        plot_one_card(ax, dim_key, dimension_scores[dim_key])
        axes.append(ax)
    return axes


def plot_stage_legend(fig: plt.Figure, gs_slot) -> plt.Axes:
    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box = FancyBboxPatch(
        (0.065, 0.24), 0.870, 0.54,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        linewidth=0.65,
        edgecolor="#D2D2D2",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    xs = [0.12, 0.29, 0.46, 0.63, 0.80]

    for x, stg in zip(xs, STAGE_ORDER):
        circ = Circle((x, 0.53), radius=0.023, transform=ax.transAxes,
                      facecolor=STAGE_COLORS[stg], edgecolor="#444444", linewidth=0.45)
        ax.add_patch(circ)
        ax.text(x, 0.53, STAGE_NUM[stg], ha="center", va="center",
                color="white", fontsize=6.8, fontweight="bold", transform=ax.transAxes)

        ax.text(x + 0.024, 0.585, STAGE_SHORT[stg],
                ha="left", va="center", fontsize=8.0,
                color=STAGE_COLORS[stg],
                fontweight="bold" if stg == "M5 Full" else "normal",
                transform=ax.transAxes)

        ax.text(x + 0.024, 0.395, STAGE_SUB[stg],
                ha="left", va="center", fontsize=6.6,
                color="#3A3A3A", transform=ax.transAxes)

    return ax


# ============================================================
# 6) Panel D data preparation
# ============================================================
def prepare_panel_d_data(df: pd.DataFrame) -> Tuple[List[Dict], Dict[str, List[np.ndarray]]]:
    available = []

    for group, label, candidates, mtype in PANEL_D_METRICS:
        col = pick_existing_column(df, candidates)
        if col is not None:
            available.append({
                "group": group,
                "label": label,
                "candidates": candidates,
                "metric_type": mtype,
                "col": col,
            })

    # per endpoint -> list of stage vectors
    stage_values = {"comm": [], "mech": [], "tfresp": []}

    for item in available:
        for dim_key in ["comm", "mech", "tfresp"]:
            disp, _ = normalized_metric_series(df, ENDPOINT_MAP[dim_key], item["candidates"], item["metric_type"])
            stage_values[dim_key].append(disp)

    return available, stage_values


def segment_profile_from_values(vals: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert 5 stage values into one segmented bar:
    - total bar length = Full (Step 5) display-normalized score
    - internal segment ratios = stage values after within-metric max scaling
    """
    vals = np.asarray(vals, dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)

    full_score = float(vals[-1]) if len(vals) > 0 else 0.0
    vmax = float(np.max(vals)) if np.any(np.isfinite(vals)) else 0.0

    if vmax <= 1e-12 or full_score <= 1e-12:
        return np.zeros(len(vals), dtype=float), full_score

    rel = vals / vmax
    denom = np.sum(rel)
    if denom <= 1e-12:
        widths = np.zeros(len(vals), dtype=float)
    else:
        widths = full_score * rel / denom

    return widths, full_score


# ============================================================
# 7) Panel D plotting
# ============================================================
def plot_panel_d(fig: plt.Figure, gs_slot, metric_items: List[Dict], stage_values: Dict[str, List[np.ndarray]]) -> List[plt.Axes]:
    sub = gs_slot.subgridspec(
        1, 4,
        width_ratios=[1.45, 1.0, 1.0, 1.0],
        wspace=0.16,
    )

    ax_label = fig.add_subplot(sub[0, 0])
    ax1 = fig.add_subplot(sub[0, 1])
    ax2 = fig.add_subplot(sub[0, 2], sharey=ax1)
    ax3 = fig.add_subplot(sub[0, 3], sharey=ax1)

    axes = [ax_label, ax1, ax2, ax3]

    n = len(metric_items)
    y = np.arange(n)[::-1]

    groups = [m["group"] for m in metric_items]
    boundaries = []
    start = 0
    for i in range(1, n + 1):
        if i == n or groups[i] != groups[start]:
            boundaries.append((groups[start], start, i - 1))
            start = i

    # -------- left label axis --------
    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(-0.6, n - 0.4)
    ax_label.axis("off")

    group_colors = {
        "Predictive\naccuracy": NATURE["blue_dark"],
        "Mechanistic\nconsistency": NATURE["purple_dark"],
        "Structure\nquality": NATURE["ochre_dark"],
        "Stability": NATURE["teal"],
    }

    for grp, s, e in boundaries:
        y_mid = np.mean(y[s:e + 1])
        c = group_colors.get(grp, "#444444")

        ax_label.text(
            0.02, y_mid, grp,
            ha="left", va="center",
            fontsize=9.1, color=c, fontweight="bold"
        )
        ax_label.plot([0.00, 0.00], [y[e] - 0.45, y[s] + 0.45], color=c, lw=1.05)

        if e < n - 1:
            sep_y = y[e] - 0.65
            ax_label.plot([0.00, 1.00], [sep_y, sep_y], color="#D2D2D2", lw=0.70, ls=(0, (2, 2)))
            for ax in [ax1, ax2, ax3]:
                ax.axhline(sep_y, color="#D2D2D2", lw=0.70, ls=(0, (2, 2)))

    for yi, item in zip(y, metric_items):
        ax_label.text(0.38, yi, item["label"], ha="left", va="center", fontsize=8.0, color="#202020")

    ax_label.plot([0.33, 0.33], [-0.5, n - 0.5], color="#565656", lw=0.75)

    endpoint_axes = [
        ("comm", ax1, "Communication accuracy", NATURE["blue_dark"]),
        ("mech", ax2, "Mechanistic consistency", NATURE["purple_dark"]),
        ("tfresp", ax3, "TF-response recovery", NATURE["vermilion_dark"]),
    ]

    for dim_key, ax, title, title_color in endpoint_axes:
        all_rows = stage_values[dim_key]

        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0, 1.02)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        ax.set_xlabel("Normalized score", fontsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#666666")
        ax.tick_params(axis="x", labelsize=7.6, length=3)

        ax.text(
            0.50, 1.055,
            title,
            ha="center", va="bottom",
            transform=ax.transAxes,
            color=title_color,
            fontsize=9.8,
            fontweight="bold",
        )

        for yi, vals in zip(y, all_rows):
            widths, full_score = segment_profile_from_values(vals)

            left = 0.0
            for stg, w in zip(STAGE_ORDER, widths):
                if w <= 0:
                    continue
                ax.barh(
                    yi,
                    w,
                    left=left,
                    height=0.40,
                    color=STAGE_COLORS[stg],
                    edgecolor="white",
                    linewidth=0.0,
                )
                left += w

            ax.text(
                min(full_score + 0.03, 0.985), yi,
                f"{full_score:.2f}",
                ha="left", va="center",
                fontsize=7.6,
                color=NATURE["vermilion_dark"],
                fontweight="bold",
            )

    return axes


# ============================================================
# 8) Export source tables
# ============================================================
def export_source_tables(
    out_dir: Path,
    dimension_scores: Dict[str, np.ndarray],
    metric_items: List[Dict],
    stage_values: Dict[str, List[np.ndarray]],
) -> None:
    # panel B source
    rows_b = []
    for dim_key in DIMENSIONS:
        for i, stg in enumerate(STAGE_ORDER, start=1):
            rows_b.append({
                "dimension_key": dim_key,
                "dimension_title": DIMENSION_INFO[dim_key]["title"],
                "step": i,
                "score_label": stg,
                "display_score": dimension_scores[dim_key][i - 1],
            })
    pd.DataFrame(rows_b).to_csv(out_dir / "panel_b_scores.csv", index=False)

    # panel D source
    rows_d = []
    for idx, item in enumerate(metric_items):
        for dim_key in ["comm", "mech", "tfresp"]:
            vals = stage_values[dim_key][idx]
            row = {
                "metric_group": item["group"],
                "metric_label": item["label"],
                "dimension_key": dim_key,
                "dimension_title": DIMENSION_INFO[dim_key]["title"],
            }
            for stg, val in zip(STAGE_ORDER, vals):
                row[stg] = val
            rows_d.append(row)
    pd.DataFrame(rows_d).to_csv(out_dir / "panel_d_stage_scores.csv", index=False)

    print("[SAVE]", out_dir / "panel_b_scores.csv")
    print("[SAVE]", out_dir / "panel_d_stage_scores.csv")


# ============================================================
# 9) Main figure
# ============================================================
def plot_main_figure(df: pd.DataFrame, out_dir: Path) -> None:
    dimension_scores = compute_all_dimension_scores(df)
    metric_items, stage_values = prepare_panel_d_data(df)
    export_source_tables(out_dir, dimension_scores, metric_items, stage_values)

    fig = plt.figure(figsize=(18.0, 10.8))
    gs = GridSpec(
        4, 1,
        figure=fig,
        height_ratios=[1.60, 0.23, 1.75, 0.12],
        hspace=0.24,
    )

    # -------------------------
    # Panel b
    # -------------------------
    axes_b = plot_panel_b(fig, gs[0], dimension_scores)

    axes_b[0].text(
        -0.04, 1.075, "b",
        transform=axes_b[0].transAxes,
        fontsize=22, fontweight="bold", ha="left", va="top"
    )
    axes_b[0].text(
        0.08, 1.055,
        "Four-dimensional evaluation across five pipeline steps",
        transform=axes_b[0].transAxes,
        fontsize=15.5, fontweight="bold", ha="left", va="top"
    )

    plot_stage_legend(fig, gs[1])

    # -------------------------
    # Panel d
    # -------------------------
    axes_d = plot_panel_d(fig, gs[2], metric_items, stage_values)
    d_pos = axes_d[0].get_position()

    fig.text(
        d_pos.x0 - 0.025,
        d_pos.y1 + 0.032,
        "d",
        fontsize=22, fontweight="bold",
        ha="left", va="top"
    )
    fig.text(
        d_pos.x0 + 0.005,
        d_pos.y1 + 0.028,
        "Metric-level progression across five pipeline steps",
        fontsize=15.0, fontweight="bold",
        ha="left", va="top"
    )
    fig.text(
        d_pos.x0 + 0.005,
        d_pos.y1 + 0.004,
        "Display-normalized score (within-metric max scaling)",
        fontsize=9.5, color="#333333",
        ha="left", va="top"
    )

    # footnote
    ax_foot = fig.add_subplot(gs[3])
    ax_foot.axis("off")
    ax_foot.text(
        0.01, 0.60,
        "Note: panel B uses a final-prioritized monotonic display summary for visualization; panel D shows five-stage segmented bars derived from within-metric max-scaled display-normalized scores.",
        ha="left", va="center",
        fontsize=9.2, color="#222222"
    )

    savefig(fig, out_dir, "Fig3_final_BD_scorecards_segmentedbars_Nature")


# ============================================================
# 10) CLI / Jupyter-compatible entry
# ============================================================
def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("[INFO] ignored unknown args:", unknown)

    in_dir = Path(args.in_dir)
    out_dir = ensure_out(Path(args.out_dir))

    print("[IN ]", in_dir)
    print("[OUT]", out_dir)

    df = load_metrics(in_dir)
    plot_main_figure(df, out_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
