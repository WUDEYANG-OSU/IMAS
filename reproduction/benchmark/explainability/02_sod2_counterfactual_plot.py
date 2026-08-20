#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replot SOD2 counterfactual validation panels in a compact main-figure style.

This script does not recompute any model quantities. It only reads the CSV
outputs written by:
  SOD2_counterfactual_support_deletion_matched_null_full_standalone_v2_duplicate_gene_resolver.py

Expected files inside the output directory:
  - SOD2_counterfactual_retention_curves_long.csv
  - SOD2_counterfactual_matched_null_mean_fragility_distribution.csv
  - SOD2_counterfactual_cell_level_summary.csv
  - SOD2_counterfactual_associated_TF_transitions.csv

Main output:
  - SOD2_counterfactual_triptych_compact_Cellstyle.png/.pdf/.svg
Also saves individual panels A/B/C.
"""

from __future__ import annotations
from imas.config import get_runs_root

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ============================================================
# Paths
# ============================================================

DEFAULT_OUTDIR_CANDIDATES = [
    get_runs_root()
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / "CRC-Tissue"
    / "fig2a_mltd_support_umap"
    / "SOD2_counterfactual_support_deletion_matched_null",
    Path.cwd() / "SOD2_counterfactual_support_deletion_matched_null",
]

REQUIRED_FILES = {
    "curves": "SOD2_counterfactual_retention_curves_long.csv",
    "null_dist": "SOD2_counterfactual_matched_null_mean_fragility_distribution.csv",
    "summary": "SOD2_counterfactual_cell_level_summary.csv",
    "transitions": "SOD2_counterfactual_associated_TF_transitions.csv",
}

# ============================================================
# Style
# ============================================================

COLOR_RATIONALE = "#C5486B"
COLOR_RATIONALE_LIGHT = "#E8AFC0"
COLOR_NULL = "#7A8796"
COLOR_NULL_LIGHT = "#D9DEE6"
COLOR_TEXT = "#1F1F1F"
COLOR_GRID = "#E7E7E7"
COLOR_MUTED = "#9A9A9A"
COLOR_ARROW = "#D86A89"
COLOR_NODE = "#E5B94F"
COLOR_NODE_EDGE = "#333333"
COLOR_SELF = "#D7D7D7"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.9,
        "axes.labelsize": 9.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    }
)

# ============================================================
# Helpers
# ============================================================


def log(*x):
    print(*x, flush=True)



def find_output_dir() -> Path:
    env = os.environ.get("SOD2_COUNTERFACTUAL_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    for candidate in DEFAULT_OUTDIR_CANDIDATES:
        if candidate.exists() and all((candidate / fname).exists() for fname in REQUIRED_FILES.values()):
            return candidate.resolve()

    # light recursive search under common roots
    search_roots = [get_runs_root(), Path.cwd()]
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("SOD2_counterfactual_cell_level_summary.csv"):
                parent = path.parent
                if all((parent / fname).exists() for fname in REQUIRED_FILES.values()):
                    return parent.resolve()
        except Exception:
            pass

    checked = "\n".join(str(x) for x in DEFAULT_OUTDIR_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate the SOD2 counterfactual output directory.\n"
        "Set environment variable SOD2_COUNTERFACTUAL_DIR or place outputs in one of:\n"
        f"{checked}"
    )



def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.9, length=3.2, color=COLOR_TEXT)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.xaxis.grid(False)



def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in [".png", ".pdf", ".svg"]:
        out = Path(str(stem) + suffix)
        fig.savefig(out, dpi=300)
        log("[SAVE]", out)



def bootstrap_mean_interval(
    matrix: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    boot = np.empty((n_bootstrap, matrix.shape[1]), dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.nanmean(matrix[idx], axis=0)
    mean = np.nanmean(matrix, axis=0)
    lower = np.nanpercentile(boot, 2.5, axis=0)
    upper = np.nanpercentile(boot, 97.5, axis=0)
    return mean, lower, upper



def empirical_upper_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    return (1.0 + np.sum(null_values >= observed)) / (1.0 + null_values.size)



def first_existing_column(
    table: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:
    """
    Resolve a column across output-version naming differences.
    """
    for column in candidates:
        if column in table.columns:
            return column

    raise KeyError(
        f"Could not resolve {description}. "
        f"Tried columns={list(candidates)}; "
        f"available columns={table.columns.tolist()}"
    )



def normalise_density_curve(values: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # light smoothing without scipy
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return centers, smooth


# ============================================================
# Load data
# ============================================================


def load_tables(out_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        key: pd.read_csv(out_dir / fname)
        for key, fname in REQUIRED_FILES.items()
    }
    return tables


# ============================================================
# Panel A
# ============================================================


def panel_retention(ax: plt.Axes, curves: pd.DataFrame) -> None:
    rationale = curves.loc[curves["mode"] == "rationale"].copy()
    null_df = curves.loc[curves["mode"] == "matched_null"].copy()

    rationale_wide = rationale.pivot_table(
        index=["sid", "rid"],
        columns="deletion_step",
        values="retention",
        aggfunc="mean",
    ).sort_index(axis=1)

    null_repeat_mean = null_df.groupby(["null_repeat", "deletion_step"])["retention"].mean().unstack("deletion_step")
    null_repeat_mean = null_repeat_mean.sort_index(axis=1)

    x = np.asarray(rationale_wide.columns, dtype=int)
    rat_mean, rat_lo, rat_hi = bootstrap_mean_interval(rationale_wide.to_numpy(), n_bootstrap=2000, seed=11)
    null_mean = np.nanmean(null_repeat_mean.to_numpy(), axis=0)
    null_lo = np.nanpercentile(null_repeat_mean.to_numpy(), 5, axis=0)
    null_hi = np.nanpercentile(null_repeat_mean.to_numpy(), 95, axis=0)

    ax.fill_between(x, null_lo, null_hi, color=COLOR_NULL_LIGHT, alpha=0.75, linewidth=0, zorder=1)
    ax.plot(x, null_mean, color=COLOR_NULL, linewidth=1.8, zorder=2)

    ax.fill_between(x, rat_lo, rat_hi, color=COLOR_RATIONALE_LIGHT, alpha=0.65, linewidth=0, zorder=3)
    ax.plot(x, rat_mean, color=COLOR_RATIONALE, linewidth=2.6, marker="o", markersize=3.4, zorder=4)

    # direct labels
    ax.text(x[-1] + 0.15, null_mean[-1], "Matched null", color=COLOR_NULL, fontsize=8.2, va="center")
    ax.text(x[-1] + 0.15, rat_mean[-1], "MLTD-prioritized", color=COLOR_RATIONALE, fontsize=8.2, va="center")

    ax.axhline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.9)
    ymin = min(np.nanmin(rat_lo), np.nanmin(null_lo))
    ymax = max(np.nanmax(rat_hi), np.nanmax(null_hi))
    pad_low = max(0.008, (ymax - ymin) * 0.35)
    ax.set_ylim(ymin - pad_low, min(1.012, ymax + 0.004))
    ax.set_xlim(x.min() - 0.5, x.max() + 1.0)
    ax.set_xticks(x)
    ax.set_xlabel("Sequentially removed support genes")
    ax.set_ylabel("Retained SOD2 score")
    ax.set_title("A  Support deletion collapses SOD2 dependency", loc="left", fontweight="bold")
    style_axis(ax)


# ============================================================
# Panel B
# ============================================================


def panel_null(ax: plt.Axes, null_dist: pd.DataFrame, summary: pd.DataFrame) -> None:
    null_column = first_existing_column(
        null_dist,
        [
            "mean_fragility_across_cells",
            "null_mean_fragility",
            "mean_null_fragility",
        ],
        description="matched-null mean fragility",
    )

    observed_column = first_existing_column(
        summary,
        [
            # Current SOD2 counterfactual output:
            "rationale_fragility_core",
            # Older/reversed naming variants:
            "fragility_core_rationale",
            "fragility_core",
            "observed_fragility_core",
        ],
        description="observed rationale fragility",
    )

    values = null_dist[
        null_column
    ].to_numpy(
        dtype=float
    )

    observed = float(
        summary[
            observed_column
        ].mean()
    )

    null_mean = float(
        np.mean(
            values
        )
    )

    p_emp = empirical_upper_p(
        observed,
        values,
    )

    bins = min(30, max(12, int(np.sqrt(values.size) * 2)))
    ax.hist(values, bins=bins, color=COLOR_NULL_LIGHT, edgecolor="white", linewidth=0.8, alpha=0.95)
    xs, ys = normalise_density_curve(values, bins=bins)
    binw = np.diff(np.histogram_bin_edges(values, bins=bins)).mean()
    ax.plot(xs, ys * values.size * binw, color=COLOR_NULL, linewidth=1.5)

    ax.axvline(null_mean, color=COLOR_NULL, linestyle=(0, (4, 2)), linewidth=1.5)
    ax.axvline(observed, color=COLOR_RATIONALE, linestyle=(0, (4, 2)), linewidth=2.2)

    ymax = ax.get_ylim()[1]
    ax.text(null_mean, ymax * 0.95, "Null mean", color=COLOR_NULL, ha="left", va="top", fontsize=7.8)
    ax.text(observed, ymax * 0.95, "Observed", color=COLOR_RATIONALE, ha="left", va="top", fontsize=7.8)
    ax.text(0.04, 0.96, f"Empirical P = {p_emp:.3g}", transform=ax.transAxes, ha="left", va="top", fontsize=8.6)

    ax.set_xlabel("Mean fragility score")
    ax.set_ylabel("Matched-null repetitions")
    ax.set_title("B  Matched-null fragility", loc="left", fontweight="bold")
    style_axis(ax)


# ============================================================
# Panel C: slope chart
# ============================================================


def _rank_map(series: pd.Series) -> dict[str, int]:
    order = list(series.sort_values(ascending=False).index)
    return {name: i for i, name in enumerate(order)}



def panel_transitions(ax: plt.Axes, summary: pd.DataFrame, transitions: pd.DataFrame) -> None:
    df = summary.copy()
    if not {"associated_tf_before", "associated_tf_after"}.issubset(df.columns):
        ax.text(0.5, 0.5, "TF transition columns not found", ha="center", va="center")
        ax.axis("off")
        return

    before_counts = df["associated_tf_before"].value_counts()
    after_counts = df["associated_tf_after"].value_counts()

    switch_tbl = transitions.loc[
        transitions["associated_tf_before"] != transitions["associated_tf_after"]
    ].copy()

    if switch_tbl.empty:
        ax.text(0.5, 0.5, "No TF re-ranking observed", ha="center", va="center")
        ax.axis("off")
        return

    switch_tbl = switch_tbl.sort_values(["n_cells", "mean_delta_fragility"], ascending=[False, False]).reset_index(drop=True)

    # choose compact top set of nodes from switched transitions
    top_edges = switch_tbl.head(6).copy()
    node_set = set(top_edges["associated_tf_before"]) | set(top_edges["associated_tf_after"])
    # add high-frequency nodes if needed
    for name in list(before_counts.index) + list(after_counts.index):
        if len(node_set) >= 8:
            break
        node_set.add(name)

    nodes = list(node_set)
    before_rank = _rank_map(before_counts.reindex(nodes).fillna(0))
    after_rank = _rank_map(after_counts.reindex(nodes).fillna(0))

    # y positions top to bottom
    def y_from_rank(rank: int, total: int) -> float:
        if total <= 1:
            return 0.5
        return 1.0 - rank / (total - 1)

    n_nodes = len(nodes)
    x_left, x_right = 0.18, 0.82

    # faint self-lines to show persistence for major nodes
    shared_major = sorted(node_set, key=lambda n: before_counts.get(n, 0) + after_counts.get(n, 0), reverse=True)[:6]
    for node in shared_major:
        y0 = y_from_rank(before_rank[node], n_nodes)
        y1 = y_from_rank(after_rank[node], n_nodes)
        ax.plot([x_left, x_right], [y0, y1], color=COLOR_SELF, linewidth=1.1, alpha=0.7, zorder=1)

    # switched edges
    edge_max = max(top_edges["n_cells"].max(), 1)
    for _, row in top_edges.iterrows():
        b = str(row["associated_tf_before"])
        a = str(row["associated_tf_after"])
        y0 = y_from_rank(before_rank[b], n_nodes)
        y1 = y_from_rank(after_rank[a], n_nodes)
        width = 0.8 + 2.2 * (float(row["n_cells"]) / edge_max)
        patch = FancyArrowPatch(
            (x_left, y0),
            (x_right, y1),
            connectionstyle="arc3,rad=0.12",
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=width,
            color=COLOR_ARROW,
            alpha=0.75,
            zorder=3,
        )
        ax.add_patch(patch)

    # nodes and labels
    for node in nodes:
        y0 = y_from_rank(before_rank[node], n_nodes)
        y1 = y_from_rank(after_rank[node], n_nodes)
        s0 = 40 + 130 * (before_counts.get(node, 0) / max(before_counts.max(), 1))
        s1 = 40 + 130 * (after_counts.get(node, 0) / max(after_counts.max(), 1))

        ax.scatter([x_left], [y0], s=s0, facecolor=COLOR_NODE, edgecolor=COLOR_NODE_EDGE, linewidth=0.9, zorder=4)
        ax.scatter([x_right], [y1], s=s1, facecolor=COLOR_NODE, edgecolor=COLOR_NODE_EDGE, linewidth=0.9, zorder=4)
        ax.text(x_left - 0.03, y0, node, ha="right", va="center", fontsize=8.0)
        ax.text(x_right + 0.03, y1, node, ha="left", va="center", fontsize=8.0)

    switch_rate = float((df["associated_tf_before"] != df["associated_tf_after"]).mean())
    ax.text(0.02, 0.98, f"Switch rate = {switch_rate:.1%}\nN cells = {df.shape[0]}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.4)
    ax.text(x_left, 1.04, "Before deletion", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax.text(x_right, 1.04, "After deletion", ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.08, 1.08)
    ax.set_title("C  Associated TF re-ranking", loc="left", fontweight="bold")
    ax.axis("off")


# ============================================================
# Main figure assembly
# ============================================================


def make_compact_triptych(out_dir: Path) -> None:
    tables = load_tables(out_dir)
    curves = tables["curves"]
    null_dist = tables["null_dist"]
    summary = tables["summary"]
    transitions = tables["transitions"]

    log(
        "[SUMMARY COLUMNS]",
        summary.columns.tolist(),
    )

    log(
        "[NULL-DISTRIBUTION COLUMNS]",
        null_dist.columns.tolist(),
    )

    fig = plt.figure(figsize=(11.8, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 1.10], wspace=0.34)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_retention(ax_a, curves)
    panel_null(ax_b, null_dist, summary)
    panel_transitions(ax_c, summary, transitions)

    fig.suptitle(
        "Counterfactual validation of the SOD2 MLTD dependency structure",
        y=1.03,
        fontsize=16,
        fontweight="bold",
    )

    save_figure(fig, out_dir / "SOD2_counterfactual_triptych_compact_Cellstyle")
    plt.close(fig)

    # Save individual panels too
    for name, panel_func in [
        ("SOD2_counterfactual_panelA_compact_retention", lambda ax: panel_retention(ax, curves)),
        ("SOD2_counterfactual_panelB_compact_null", lambda ax: panel_null(ax, null_dist, summary)),
        ("SOD2_counterfactual_panelC_compact_TF_reranking", lambda ax: panel_transitions(ax, summary, transitions)),
    ]:
        fig_i, ax_i = plt.subplots(figsize=(3.8, 3.3))
        panel_func(ax_i)
        save_figure(fig_i, out_dir / name)
        plt.close(fig_i)

    manifest = {
        "source_output_dir": str(out_dir),
        "used_files": {k: str(out_dir / v) for k, v in REQUIRED_FILES.items()},
        "main_output": str(out_dir / "SOD2_counterfactual_triptych_compact_Cellstyle.png"),
    }
    with open(out_dir / "SOD2_counterfactual_triptych_compact_Cellstyle_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log("[SAVE]", out_dir / "SOD2_counterfactual_triptych_compact_Cellstyle_manifest.json")



def main() -> None:
    out_dir = find_output_dir()
    log("[OUT DIR]", out_dir)
    make_compact_triptych(out_dir)
    log("[DONE]", "Finished compact Cell-style replot.")


if __name__ == "__main__":
    main()
