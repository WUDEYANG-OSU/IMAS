from __future__ import annotations

# ============================================================
# plot_supplementary_ablation_metrics.py
# ------------------------------------------------------------
# Purpose:
#   Make supplementary figures for the remaining ablation metrics
#   from global_edge_endpoint_ablation.py outputs.
#
# Inputs:
#   global_stage_metrics.csv
#   global_stage_scores_long.csv optional, not required
#   global_edge_endpoint_table.csv optional, used for endpoint overview only
#
# Design:
#   Main-text plots should stay compact. This script makes supplementary
#   heatmaps and overview plots for the extended metrics:
#     - extended recovery metrics
#     - calibration and separation metrics
#     - score-feature correlations
#     - TopK CCI / TF / RNA-TF / StageA/B support metrics
#     - topology and coverage metrics
#     - robustness and bootstrap metrics
#     - full metric source tables
#
# Usage:
#   python plot_supplementary_ablation_metrics.py \
#     --in_dir $IMAS_PROJECT_ROOT/runs/global_edge_endpoint_ablation/CRC-Tissue/CRC_RNA_TO009 \
#     --out_dir $IMAS_PROJECT_ROOT/runs/global_edge_endpoint_ablation/CRC-Tissue/CRC_RNA_TO009/supplementary_figures
#
# Notes:
#   - This script is read-only for benchmark outputs.
#   - It does not recompute metrics.
#   - It tolerates missing columns and writes a missing metric report.
# ============================================================

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0) DEFAULT PATHS
# ============================================================
BASE = get_project_root()
DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"
DEFAULT_IN_DIR = BASE / "runs" / "global_edge_endpoint_ablation" / DATASET_NAME / SAMPLE_ID
DEFAULT_OUT_DIR = DEFAULT_IN_DIR / "supplementary_figures"


# ============================================================
# 1) DISPLAY CONFIG
# ============================================================
SCORE_ORDER = [
    "N0 Random",
    "N1 Degree-only",
    "N2 Direction-flipped",
    "N3 TF receiver-shuffled",
    "N4 CCI-shuffled only",
    "N5 CCI-shuffled + LTC",
    "M0 RNA-only",
    "M1 RNA+TF",
    "M2 RNA-TF matrix",
    "M3 CCI",
    "M4 CCI+LTC",
    "M5 Full",
]

SCORE_SHORT = {
    "N0 Random": "Random",
    "N1 Degree-only": "Degree-only",
    "N2 Direction-flipped": "Direction-flipped",
    "N3 TF receiver-shuffled": "TF receiver-shuffled",
    "N4 CCI-shuffled only": "CCI-shuffled only",
    "N5 CCI-shuffled + LTC": "CCI-shuffled + LTC",
    "M0 RNA-only": "RNA-only",
    "M1 RNA+TF": "RNA-TF reg.",
    "M2 RNA-TF matrix": "RNA-TF matrix",
    "M3 CCI": "CCI",
    "M4 CCI+LTC": "CCI+LTC",
    "M5 Full": "Full StageA/B",
}

ENDPOINT_ORDER = [
    "global_positive_edge",
    "global_mechanistic_positive_edge",
    "tf_response_positive_edge",
    "cci_consensus_positive_edge",
]

ENDPOINT_LABEL = {
    "global_positive_edge": "Directed communication",
    "global_mechanistic_positive_edge": "Mechanism-supported",
    "tf_response_positive_edge": "TF-response endpoint",
    "cci_consensus_positive_edge": "CCI consensus",
}

# Main-text metrics are not removed from all supplementary tables, but we do not
# emphasize them in category-specific supplementary figures.
MAIN_TEXT_METRICS = {
    "auroc",
    "auprc",
    "precision_at_10",
    "ndcg_at_10",
    "lift_at_10",
    "top10_delta_positive_frac",
    "top10_reciprocal_penalty",
    "top10_stageAB_support_mean",
    "bootstrap_auprc_std",
}

SUPPLEMENT_GROUPS: dict[str, list[tuple[str, str, str]]] = {
    "S1_extended_recovery": [
        ("mrr", "MRR", "higher"),
        ("mean_rank_positive_norm_inv", "Mean positive rank\nnormalized inverse", "higher"),
        ("median_rank_positive_norm_inv", "Median positive rank\nnormalized inverse", "higher"),
        ("recall_at_5", "Recall@5", "higher"),
        ("recall_at_10", "Recall@10", "higher"),
        ("recall_at_20", "Recall@20", "higher"),
        ("recall_at_50", "Recall@50", "higher"),
        ("hit_at_5", "Hit@5", "higher"),
        ("hit_at_10", "Hit@10", "higher"),
        ("hit_at_20", "Hit@20", "higher"),
        ("precision_at_20", "Precision@20", "higher"),
        ("precision_at_50", "Precision@50", "higher"),
        ("ndcg_at_20", "NDCG@20", "higher"),
        ("ndcg_at_50", "NDCG@50", "higher"),
        ("lift_at_20", "Lift@20", "higher_lift"),
        ("lift_at_50", "Lift@50", "higher_lift"),
    ],
    "S2_calibration_separation": [
        ("brier", "Brier score", "lower"),
        ("ece", "ECE", "lower"),
        ("score_separation", "Positive-negative\nscore separation", "signed"),
        ("cohen_d", "Cohen's d", "signed"),
        ("cliffs_delta", "Cliff's delta", "signed_bounded"),
    ],
    "S3_score_feature_correlations": [
        ("score_delta_corr", "Score-Δt corr.", "corr"),
        ("score_tf_response_corr", "Score-TF response corr.", "corr"),
        ("score_rna_tf_corr", "Score-RNA-TF corr.", "corr"),
        ("score_cci_corr", "Score-CCI corr.", "corr"),
        ("score_stageAB_corr", "Score-StageA/B corr.", "corr"),
    ],
    "S4_topk_biological_support": [
        ("top5_highconf_rate", "Top5 highconf rate", "higher"),
        ("top10_highconf_rate", "Top10 highconf rate", "higher"),
        ("top20_highconf_rate", "Top20 highconf rate", "higher"),
        ("top50_highconf_rate", "Top50 highconf rate", "higher"),
        ("top5_tfkept_rate", "Top5 TF-kept rate", "higher"),
        ("top10_tfkept_rate", "Top10 TF-kept rate", "higher"),
        ("top20_tfkept_rate", "Top20 TF-kept rate", "higher"),
        ("top50_tfkept_rate", "Top50 TF-kept rate", "higher"),
        ("top5_cci_score_mean", "Top5 CCI score", "higher"),
        ("top10_cci_score_mean", "Top10 CCI score", "higher"),
        ("top20_cci_score_mean", "Top20 CCI score", "higher"),
        ("top50_cci_score_mean", "Top50 CCI score", "higher"),
        ("top5_receiver_tf_support_mean", "Top5 receiver TF", "higher"),
        ("top10_receiver_tf_support_mean", "Top10 receiver TF", "higher"),
        ("top20_receiver_tf_support_mean", "Top20 receiver TF", "higher"),
        ("top50_receiver_tf_support_mean", "Top50 receiver TF", "higher"),
        ("top5_rna_tf_support_mean", "Top5 RNA-TF support", "higher"),
        ("top10_rna_tf_support_mean", "Top10 RNA-TF support", "higher"),
        ("top20_rna_tf_support_mean", "Top20 RNA-TF support", "higher"),
        ("top50_rna_tf_support_mean", "Top50 RNA-TF support", "higher"),
    ],
    "S5_stageAB_mechanism_support": [
        ("top5_stageA_support_mean", "Top5 StageA", "higher"),
        ("top10_stageA_support_mean", "Top10 StageA", "higher"),
        ("top20_stageA_support_mean", "Top20 StageA", "higher"),
        ("top50_stageA_support_mean", "Top50 StageA", "higher"),
        ("top5_stageB_support_mean", "Top5 StageB", "higher"),
        ("top10_stageB_support_mean", "Top10 StageB", "higher"),
        ("top20_stageB_support_mean", "Top20 StageB", "higher"),
        ("top50_stageB_support_mean", "Top50 StageB", "higher"),
        ("top5_stageAB_support_mean", "Top5 StageA/B", "higher"),
        ("top20_stageAB_support_mean", "Top20 StageA/B", "higher"),
        ("top50_stageAB_support_mean", "Top50 StageA/B", "higher"),
        ("mechanism_lift_at_5", "Mechanism lift@5", "higher_lift"),
        ("mechanism_lift_at_10", "Mechanism lift@10", "higher_lift"),
        ("mechanism_lift_at_20", "Mechanism lift@20", "higher_lift"),
        ("mechanism_lift_at_50", "Mechanism lift@50", "higher_lift"),
    ],
    "S6_topology_coverage": [
        ("top5_sender_coverage_frac", "Top5 sender coverage", "higher"),
        ("top10_sender_coverage_frac", "Top10 sender coverage", "higher"),
        ("top20_sender_coverage_frac", "Top20 sender coverage", "higher"),
        ("top50_sender_coverage_frac", "Top50 sender coverage", "higher"),
        ("top5_receiver_coverage_frac", "Top5 receiver coverage", "higher"),
        ("top10_receiver_coverage_frac", "Top10 receiver coverage", "higher"),
        ("top20_receiver_coverage_frac", "Top20 receiver coverage", "higher"),
        ("top50_receiver_coverage_frac", "Top50 receiver coverage", "higher"),
        ("top5_edge_pair_diversity_frac", "Top5 edge diversity", "higher"),
        ("top10_edge_pair_diversity_frac", "Top10 edge diversity", "higher"),
        ("top20_edge_pair_diversity_frac", "Top20 edge diversity", "higher"),
        ("top50_edge_pair_diversity_frac", "Top50 edge diversity", "higher"),
        ("top5_reciprocal_penalty", "Top5 reciprocal penalty", "lower"),
        ("top20_reciprocal_penalty", "Top20 reciprocal penalty", "lower"),
        ("top50_reciprocal_penalty", "Top50 reciprocal penalty", "lower"),
        ("top5_coverage_adjusted_precision", "Top5 coverage-adjusted P", "higher"),
        ("top10_coverage_adjusted_precision", "Top10 coverage-adjusted P", "higher"),
        ("top20_coverage_adjusted_precision", "Top20 coverage-adjusted P", "higher"),
        ("top50_coverage_adjusted_precision", "Top50 coverage-adjusted P", "higher"),
    ],
    "S7_robustness_bootstrap": [
        ("bootstrap_auroc_mean", "Bootstrap AUROC mean", "higher"),
        ("bootstrap_auroc_std", "Bootstrap AUROC std", "lower_std"),
        ("bootstrap_auprc_mean", "Bootstrap AUPRC mean", "higher"),
        ("bootstrap_precision_at_10_mean", "Bootstrap P@10 mean", "higher"),
        ("bootstrap_precision_at_10_std", "Bootstrap P@10 std", "lower_std"),
        ("bootstrap_top10_jaccard_mean", "Bootstrap Top10 Jaccard", "higher"),
        ("bootstrap_top10_jaccard_std", "Bootstrap Top10 Jaccard std", "lower_std"),
    ],
}

# Endpoint-specific figures can become too many. These endpoints are shown by default.
DEFAULT_ENDPOINTS_FOR_GROUP_FIGS = [
    "global_positive_edge",
    "global_mechanistic_positive_edge",
    "tf_response_positive_edge",
    "cci_consensus_positive_edge",
]


# ============================================================
# 2) HELPERS
# ============================================================
def ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_metrics(in_dir: Path) -> pd.DataFrame:
    p = in_dir / "global_stage_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find metrics table: {p}")
    df = pd.read_csv(p)
    required = {"endpoint", "score_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"global_stage_metrics.csv missing required columns: {missing}")
    return df


def load_edge_table(in_dir: Path) -> pd.DataFrame | None:
    p = in_dir / "global_edge_endpoint_table.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE]", png)
    print("[SAVE]", svg)


def existing_score_order(metrics: pd.DataFrame) -> list[str]:
    present = set(metrics["score_label"].dropna().astype(str))
    ordered = [s for s in SCORE_ORDER if s in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def existing_endpoint_order(metrics: pd.DataFrame) -> list[str]:
    present = set(metrics["endpoint"].dropna().astype(str))
    ordered = [e for e in ENDPOINT_ORDER if e in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.8)


def subset_metric(df: pd.DataFrame, endpoint: str, score_label: str, metric: str) -> float:
    if metric not in df.columns:
        return np.nan
    sub = df[(df["endpoint"] == endpoint) & (df["score_label"] == score_label)]
    if sub.empty:
        return np.nan
    return float(pd.to_numeric(sub[metric], errors="coerce").iloc[0])


def metric_display_value(raw: float, metric_type: str, values: np.ndarray | None = None) -> float:
    if not np.isfinite(raw):
        return np.nan

    if metric_type == "higher":
        return float(np.clip(raw, 0, 1))

    if metric_type == "higher_lift":
        if values is not None:
            valid = values[np.isfinite(values)]
            vmax = max(float(np.nanmax(valid)), 1.0) if valid.size else 6.0
            return float(np.clip(raw / vmax, 0, 1))
        return float(np.clip(raw / 6.0, 0, 1))

    if metric_type in {"lower", "lower_std"}:
        if values is not None:
            valid = values[np.isfinite(values)]
            vmax = float(np.nanmax(valid)) if valid.size else 1.0
            if vmax <= 1e-12:
                return 1.0
            return float(1.0 - np.clip(raw / vmax, 0, 1))
        return float(1.0 - np.clip(raw, 0, 1))

    if metric_type == "corr":
        return float((np.clip(raw, -1, 1) + 1) / 2)

    if metric_type == "signed":
        if values is not None:
            valid = values[np.isfinite(values)]
            vmax = max(abs(float(np.nanmin(valid))), abs(float(np.nanmax(valid))), 1e-12) if valid.size else 1.0
            return float((np.clip(raw / vmax, -1, 1) + 1) / 2)
        return float((np.tanh(raw) + 1) / 2)

    if metric_type == "signed_bounded":
        return float((np.clip(raw, -1, 1) + 1) / 2)

    return float(np.clip(raw, 0, 1))


def text_color_for_display(display_val: float) -> str:
    return "white" if np.isfinite(display_val) and display_val < 0.42 else "black"


def format_value(raw: float, metric_type: str) -> str:
    if not np.isfinite(raw):
        return ""
    if metric_type in {"lower_std"} or abs(raw) < 0.1:
        return f"{raw:.3f}"
    if metric_type in {"higher_lift"} and abs(raw) >= 10:
        return f"{raw:.1f}"
    return f"{raw:.2f}"


def metric_group_available(metrics: pd.DataFrame, group: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return [x for x in group if x[0] in metrics.columns]


def write_missing_metric_report(metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for group_name, group in SUPPLEMENT_GROUPS.items():
        for metric, label, metric_type in group:
            rows.append({
                "group": group_name,
                "metric": metric,
                "label": label.replace("\n", " "),
                "metric_type": metric_type,
                "available": metric in metrics.columns,
            })
    report = pd.DataFrame(rows)
    p = out_dir / "supplementary_missing_metric_report.csv"
    report.to_csv(p, index=False)
    print("[SAVE]", p)
    return report


def build_supplement_source_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoints = existing_endpoint_order(metrics)
    scores = existing_score_order(metrics)
    for group_name, group in SUPPLEMENT_GROUPS.items():
        for metric, metric_label, metric_type in group:
            if metric not in metrics.columns:
                continue
            for endpoint in endpoints:
                for score_label in scores:
                    rows.append({
                        "group": group_name,
                        "endpoint": endpoint,
                        "endpoint_label": ENDPOINT_LABEL.get(endpoint, endpoint),
                        "score_label": score_label,
                        "score_short": SCORE_SHORT.get(score_label, score_label),
                        "metric": metric,
                        "metric_label": metric_label.replace("\n", " "),
                        "metric_type": metric_type,
                        "value": subset_metric(metrics, endpoint, score_label, metric),
                    })
    return pd.DataFrame(rows)


# ============================================================
# 3) CORE HEATMAP PLOTTERS
# ============================================================
def plot_group_heatmap_for_endpoint(
    metrics: pd.DataFrame,
    endpoint: str,
    group_name: str,
    group: list[tuple[str, str, str]],
    out_dir: Path,
) -> None:
    available = metric_group_available(metrics, group)
    if not available:
        print(f"[SKIP] {group_name} for {endpoint}: no metrics available")
        return

    scores = existing_score_order(metrics)
    metric_names = [m for m, _, _ in available]
    metric_labels = [label for _, label, _ in available]
    metric_types = {m: t for m, _, t in available}

    raw_mat = np.full((len(metric_names), len(scores)), np.nan, dtype=float)
    display_mat = np.full_like(raw_mat, np.nan)

    for i, metric in enumerate(metric_names):
        vals = np.array([subset_metric(metrics, endpoint, score, metric) for score in scores], dtype=float)
        raw_mat[i, :] = vals
        for j, raw in enumerate(vals):
            display_mat[i, j] = metric_display_value(raw, metric_types[metric], vals)

    height = max(4.5, 0.36 * len(metric_names) + 2.0)
    width = max(8.0, 0.62 * len(scores) + 4.0)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(display_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels([SCORE_SHORT.get(s, s) for s in scores], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=8)
    ax.set_title(f"{group_name}: {ENDPOINT_LABEL.get(endpoint, endpoint)}", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("display value; lower-is-better metrics inverted")

    for i in range(raw_mat.shape[0]):
        metric_type = metric_types[metric_names[i]]
        for j in range(raw_mat.shape[1]):
            raw = raw_mat[i, j]
            disp = display_mat[i, j]
            if np.isfinite(raw) and np.isfinite(disp):
                ax.text(j, i, format_value(raw, metric_type), ha="center", va="center", fontsize=5.8, color=text_color_for_display(disp))

    fig.tight_layout()
    safe_name = f"{group_name}_{endpoint}".replace("/", "_").replace(" ", "_")
    savefig(fig, out_dir, safe_name)


def plot_endpoint_summary_by_group(metrics: pd.DataFrame, group_name: str, group: list[tuple[str, str, str]], out_dir: Path) -> None:
    available = metric_group_available(metrics, group)
    if not available:
        print(f"[SKIP] {group_name} endpoint summary: no metrics available")
        return

    endpoints = existing_endpoint_order(metrics)
    scores = existing_score_order(metrics)

    # Average display value across metrics for each endpoint-score pair.
    raw_summary = np.full((len(endpoints), len(scores)), np.nan, dtype=float)
    for i, endpoint in enumerate(endpoints):
        for j, score in enumerate(scores):
            display_vals = []
            for metric, _, metric_type in available:
                raw = subset_metric(metrics, endpoint, score, metric)
                vals = np.array([subset_metric(metrics, endpoint, s, metric) for s in scores], dtype=float)
                disp = metric_display_value(raw, metric_type, vals)
                if np.isfinite(disp):
                    display_vals.append(disp)
            raw_summary[i, j] = float(np.mean(display_vals)) if display_vals else np.nan

    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(scores) + 3.0), max(3.6, 0.5 * len(endpoints) + 2.0)))
    im = ax.imshow(raw_summary, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels([SCORE_SHORT.get(s, s) for s in scores], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(endpoints)))
    ax.set_yticklabels([ENDPOINT_LABEL.get(e, e) for e in endpoints], fontsize=8)
    ax.set_title(f"{group_name}: average normalized display value", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("mean display value")
    for i in range(raw_summary.shape[0]):
        for j in range(raw_summary.shape[1]):
            val = raw_summary[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color_for_display(val))
    fig.tight_layout()
    savefig(fig, out_dir, f"{group_name}_endpoint_summary")


# ============================================================
# 4) SPECIAL SUPPLEMENTARY PLOTS
# ============================================================
def plot_rank_metric_lines(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Line plots for Precision/Recall/NDCG/Lift across K=5/10/20/50."""
    endpoints = [e for e in ["global_positive_edge", "global_mechanistic_positive_edge", "tf_response_positive_edge"] if e in set(metrics["endpoint"])]
    stage_scores = [s for s in ["M0 RNA-only", "M1 RNA+TF", "M3 CCI", "M4 CCI+LTC", "M5 Full"] if s in set(metrics["score_label"])]
    ks = [5, 10, 20, 50]
    metric_prefixes = [
        ("precision_at_", "Precision@K"),
        ("recall_at_", "Recall@K"),
        ("ndcg_at_", "NDCG@K"),
        ("lift_at_", "Lift@K"),
    ]

    for endpoint in endpoints:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
        axes = axes.ravel()
        for ax, (prefix, title) in zip(axes, metric_prefixes):
            for score in stage_scores:
                vals = []
                for k in ks:
                    metric = f"{prefix}{k}"
                    vals.append(subset_metric(metrics, endpoint, score, metric))
                ax.plot(ks, vals, marker="o", linewidth=2, label=SCORE_SHORT.get(score, score))
            ax.set_xscale("linear")
            ax.set_xticks(ks)
            ax.set_xlabel("K")
            ax.set_ylabel(title)
            ax.set_title(title)
            clean_axes(ax)
            if title != "Lift@K":
                ax.set_ylim(0, 1.05)
        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Rank metrics across K: {ENDPOINT_LABEL.get(endpoint, endpoint)}", fontsize=13, y=1.02)
        fig.tight_layout()
        savefig(fig, out_dir, f"S_rank_metrics_acrossK_{endpoint}")


def plot_topk_support_lines(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Line plots for topK support metrics across K."""
    endpoint = "global_mechanistic_positive_edge"
    if endpoint not in set(metrics["endpoint"]):
        endpoint = existing_endpoint_order(metrics)[0]
    stage_scores = [s for s in ["M0 RNA-only", "M1 RNA+TF", "M3 CCI", "M4 CCI+LTC", "M5 Full"] if s in set(metrics["score_label"])]
    ks = [5, 10, 20, 50]
    support_specs = [
        ("top{k}_highconf_rate", "Highconf rate"),
        ("top{k}_tfkept_rate", "TF-kept rate"),
        ("top{k}_receiver_tf_support_mean", "Receiver TF support"),
        ("top{k}_stageAB_support_mean", "StageA/B support"),
        ("top{k}_sender_coverage_frac", "Sender coverage"),
        ("top{k}_receiver_coverage_frac", "Receiver coverage"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2))
    axes = axes.ravel()
    for ax, (template, title) in zip(axes, support_specs):
        for score in stage_scores:
            vals = [subset_metric(metrics, endpoint, score, template.format(k=k)) for k in ks]
            ax.plot(ks, vals, marker="o", linewidth=2, label=SCORE_SHORT.get(score, score))
        ax.set_xticks(ks)
        ax.set_xlabel("K")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        clean_axes(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"TopK support and coverage across K: {ENDPOINT_LABEL.get(endpoint, endpoint)}", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, out_dir, "S_topK_support_coverage_acrossK_mechanistic_endpoint")


def plot_endpoint_positive_rates(metrics: pd.DataFrame, edge_df: pd.DataFrame | None, out_dir: Path) -> None:
    endpoints = existing_endpoint_order(metrics)
    rates = []
    counts = []
    for endpoint in endpoints:
        sub = metrics[metrics["endpoint"] == endpoint]
        if not sub.empty and "positive_rate" in sub.columns:
            rate = float(pd.to_numeric(sub["positive_rate"], errors="coerce").dropna().iloc[0])
            npos = int(pd.to_numeric(sub["n_positive"], errors="coerce").dropna().iloc[0]) if "n_positive" in sub.columns else np.nan
        elif edge_df is not None and endpoint in edge_df.columns:
            rate = float(edge_df[endpoint].mean())
            npos = int(edge_df[endpoint].sum())
        else:
            rate = np.nan
            npos = np.nan
        rates.append(rate)
        counts.append(npos)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(endpoints))
    ax.bar(x, rates, color="#9DB7C9", edgecolor="black", linewidth=0.5)
    for xi, rate, npos in zip(x, rates, counts):
        if np.isfinite(rate):
            ax.text(xi, rate + 0.025, f"{rate:.2f}\n(n={npos})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ENDPOINT_LABEL.get(e, e) for e in endpoints], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Positive rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Endpoint positive rates")
    clean_axes(ax)
    fig.tight_layout()
    savefig(fig, out_dir, "S_endpoint_positive_rates")


def plot_correlation_bar_panel(metrics: pd.DataFrame, out_dir: Path) -> None:
    endpoint = "global_mechanistic_positive_edge"
    if endpoint not in set(metrics["endpoint"]):
        endpoint = existing_endpoint_order(metrics)[0]
    scores = [s for s in ["M0 RNA-only", "M1 RNA+TF", "M3 CCI", "M4 CCI+LTC", "M5 Full"] if s in set(metrics["score_label"])]
    corr_metrics = [
        ("score_delta_corr", "Δt"),
        ("score_tf_response_corr", "TF response"),
        ("score_rna_tf_corr", "RNA-TF"),
        ("score_cci_corr", "CCI"),
        ("score_stageAB_corr", "StageA/B"),
    ]
    corr_metrics = [x for x in corr_metrics if x[0] in metrics.columns]
    if not corr_metrics:
        return

    mat = np.full((len(corr_metrics), len(scores)), np.nan)
    for i, (metric, _) in enumerate(corr_metrics):
        for j, score in enumerate(scores):
            mat[i, j] = subset_metric(metrics, endpoint, score, metric)

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels([SCORE_SHORT.get(s, s) for s in scores], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(corr_metrics)))
    ax.set_yticklabels([lab for _, lab in corr_metrics], fontsize=8)
    ax.set_title(f"Score-feature correlations: {ENDPOINT_LABEL.get(endpoint, endpoint)}")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(val) > 0.55 else "black")
    fig.tight_layout()
    savefig(fig, out_dir, "S_score_feature_correlation_panel")


# ============================================================
# 5) MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(DEFAULT_IN_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument(
        "--only_main_endpoints",
        action="store_true",
        help="Only plot global_positive_edge, global_mechanistic_positive_edge, and tf_response_positive_edge for group heatmaps.",
    )
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[INFO] ignored unknown CLI args:", unknown)

    in_dir = Path(args.in_dir)
    out_dir = ensure_out(Path(args.out_dir))
    metrics = load_metrics(in_dir)
    edge_df = load_edge_table(in_dir)

    write_missing_metric_report(metrics, out_dir)
    source = build_supplement_source_table(metrics)
    source_path = out_dir / "supplementary_metric_source_table.csv"
    source.to_csv(source_path, index=False)
    print("[SAVE]", source_path)

    endpoints = existing_endpoint_order(metrics)
    if args.only_main_endpoints:
        endpoints = [e for e in endpoints if e in {"global_positive_edge", "global_mechanistic_positive_edge", "tf_response_positive_edge"}]

    # 1) Heatmaps for each metric group and endpoint.
    for group_name, group in SUPPLEMENT_GROUPS.items():
        group_dir = ensure_out(out_dir / group_name)
        for endpoint in endpoints:
            plot_group_heatmap_for_endpoint(metrics, endpoint, group_name, group, group_dir)
        plot_endpoint_summary_by_group(metrics, group_name, group, group_dir)

    # 2) Special overview plots.
    overview_dir = ensure_out(out_dir / "S0_overview")
    plot_endpoint_positive_rates(metrics, edge_df, overview_dir)
    plot_rank_metric_lines(metrics, overview_dir)
    plot_topk_support_lines(metrics, overview_dir)
    plot_correlation_bar_panel(metrics, overview_dir)

    print("[DONE] supplementary figures written to", out_dir)


if __name__ == "__main__":
    main()
