from __future__ import annotations

# ============================================================
# plot_tf_rna_auc_bowling_combined.py
# ------------------------------------------------------------
# 在同一张图里画 TF 和 RNA 的 AUC 保龄球图（violin）
# 输出两版：
#   1) with scatter
#   2) no scatter
#
# 输入：
#   TF:
#     <SAMPLE_ID>_tf_auc_overall.csv
#
#   RNA:
#     <SAMPLE_ID>_rna2rna_cell_aucs_with_newcluster.csv
#
# 输出：
#   <SAMPLE_ID>_tf_rna_auc_bowling_with_scatter.png / svg
#   <SAMPLE_ID>_tf_rna_auc_bowling_no_scatter.png / svg
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 0) CONFIG
# ============================================================

DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

BASE = get_project_root()

RUN_BASE = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)

# ---------- TF ROC ----------
TF_ROC_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_newcluster_roc_aligned"
TF_AUC_CSV = TF_ROC_DIR / f"{SAMPLE_ID}_tf_auc_overall.csv"

# ---------- RNA→RNA ROC ----------
RNA_ROC_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_newcluster_roc_rna2rna_aligned"
RNA_AUC_CSV = RNA_ROC_DIR / f"{SAMPLE_ID}_rna2rna_cell_aucs_with_newcluster.csv"

# ---------- OUTPUT ----------
OUT_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_combined_auc_bowling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_WITH_SCATTER_PNG = OUT_DIR / f"{SAMPLE_ID}_tf_rna_auc_bowling_with_scatter.png"
OUT_WITH_SCATTER_SVG = OUT_DIR / f"{SAMPLE_ID}_tf_rna_auc_bowling_with_scatter.svg"

OUT_NO_SCATTER_PNG = OUT_DIR / f"{SAMPLE_ID}_tf_rna_auc_bowling_no_scatter.png"
OUT_NO_SCATTER_SVG = OUT_DIR / f"{SAMPLE_ID}_tf_rna_auc_bowling_no_scatter.svg"

# ---------- plotting ----------
SEED = 42
FIG_DPI = 320

FIG_W = 5.4
FIG_H = 7.2

Y_MIN = 0.0
Y_MAX = 1.0

MAX_POINTS = 5000

# 颜色
TF_FILL = "#9ecae1"     # 浅蓝
TF_DOT = "#3182bd"      # 深蓝

RNA_FILL = "#fdd0a2"    # 浅橙
RNA_DOT = "#e6550d"     # 深橙


# ============================================================
# 1) UTILS
# ============================================================

def log(*args):
    print(*args, flush=True)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def find_auc_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    auc_like = [c for c in df.columns if "auc" in c.lower()]
    if len(auc_like) > 0:
        return auc_like[0]

    raise ValueError(
        f"Cannot find AUC column. candidates={candidates}, "
        f"available={list(df.columns)}"
    )


def load_auc_values(csv_path: Path, auc_col_candidates: list[str]) -> np.ndarray:
    assert csv_path.exists(), f"Missing AUC CSV: {csv_path}"

    log("[LOAD]", csv_path)
    df = pd.read_csv(csv_path)

    auc_col = find_auc_column(df, auc_col_candidates)
    log("[AUC COLUMN]", auc_col)

    auc = pd.to_numeric(df[auc_col], errors="coerce").values.astype(np.float64)
    auc = auc[np.isfinite(auc)]
    auc = auc[(auc >= 0.0) & (auc <= 1.0)]

    log("[INFO] n =", len(auc))
    log("[INFO] mean =", float(np.mean(auc)) if len(auc) else np.nan)
    log("[INFO] median =", float(np.median(auc)) if len(auc) else np.nan)

    return auc


def downsample_for_scatter(x: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= max_points:
        return x

    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx]


def add_group_stats(ax, x_center: float, values: np.ndarray, text_x_shift: float = 0.33):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    q25 = float(np.quantile(values, 0.25))
    q75 = float(np.quantile(values, 0.75))

    # IQR 竖线
    ax.vlines(
        x_center,
        q25,
        q75,
        color="black",
        linewidth=2.0,
        alpha=0.35,
        zorder=4,
    )

    # mean 实线
    ax.hlines(
        mean_v,
        x_center - 0.28,
        x_center + 0.28,
        color="black",
        linewidth=1.6,
        linestyle="-",
        zorder=5,
    )

    # median 虚线
    ax.hlines(
        median_v,
        x_center - 0.28,
        x_center + 0.28,
        color="0.35",
        linewidth=1.4,
        linestyle="--",
        zorder=5,
    )

    # 文本标注
    ax.text(
        x_center + text_x_shift,
        median_v,
        f"median={median_v:.3f}",
        ha="left",
        va="center",
        fontsize=8.8,
        color="0.35",
    )

    ax.text(
        x_center + text_x_shift,
        mean_v,
        f"mean={mean_v:.3f}",
        ha="left",
        va="center",
        fontsize=8.8,
        color="black",
    )

    # n
    ax.text(
        x_center,
        Y_MIN + 0.02 * (Y_MAX - Y_MIN),
        f"n={values.size}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )


# ============================================================
# 2) MAIN PLOT
# ============================================================

def plot_combined_bowling(
    tf_auc: np.ndarray,
    rna_auc: np.ndarray,
    *,
    show_scatter: bool,
    out_png: Path,
    out_svg: Path | None = None,
):
    tf_auc = np.asarray(tf_auc, dtype=np.float64)
    tf_auc = tf_auc[np.isfinite(tf_auc)]
    tf_auc = tf_auc[(tf_auc >= Y_MIN) & (tf_auc <= Y_MAX)]

    rna_auc = np.asarray(rna_auc, dtype=np.float64)
    rna_auc = rna_auc[np.isfinite(rna_auc)]
    rna_auc = rna_auc[(rna_auc >= Y_MIN) & (rna_auc <= Y_MAX)]

    assert tf_auc.size > 0, "No valid TF AUC values."
    assert rna_auc.size > 0, "No valid RNA AUC values."

    ensure_parent(out_png)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    positions = [0, 1]
    all_data = [tf_auc, rna_auc]

    # violin
    parts = ax.violinplot(
        all_data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.22,
    )

    fill_colors = [TF_FILL, RNA_FILL]
    for body, c in zip(parts["bodies"], fill_colors):
        body.set_facecolor(c)
        body.set_edgecolor("none")
        body.set_alpha(0.75)

    # scatter
    if show_scatter:
        rng = np.random.default_rng(SEED)

        tf_scatter = downsample_for_scatter(tf_auc, MAX_POINTS, SEED)
        rna_scatter = downsample_for_scatter(rna_auc, MAX_POINTS, SEED + 1)

        tf_jitter = rng.normal(loc=0.0, scale=0.045, size=tf_scatter.size)
        tf_jitter = np.clip(tf_jitter, -0.16, 0.16)

        rna_jitter = rng.normal(loc=0.0, scale=0.045, size=rna_scatter.size)
        rna_jitter = np.clip(rna_jitter, -0.16, 0.16)

        ax.scatter(
            positions[0] + tf_jitter,
            tf_scatter,
            s=10,
            color=TF_DOT,
            alpha=0.28,
            linewidths=0,
            zorder=3,
        )

        ax.scatter(
            positions[1] + rna_jitter,
            rna_scatter,
            s=10,
            color=RNA_DOT,
            alpha=0.28,
            linewidths=0,
            zorder=3,
        )

    # mean / median / n
    add_group_stats(ax, positions[0], tf_auc, text_x_shift=0.25)
    add_group_stats(ax, positions[1], rna_auc, text_x_shift=0.25)

    # title
    ax.set_title(
        "AUC distribution",
        fontsize=17,
        fontweight="bold",
        pad=10,
    )

    ax.set_ylabel("AUC", fontsize=11)
    ax.set_xlabel("Group", fontsize=11)

    ax.set_xticks(positions)
    ax.set_xticklabels(["TF", "RNA"], fontsize=11)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-0.65, 1.65)

    ax.set_yticks(np.linspace(Y_MIN, Y_MAX, 6))
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=TF_FILL, edgecolor="none", alpha=0.75, label="TF"),
        Patch(facecolor=RNA_FILL, edgecolor="none", alpha=0.75, label="RNA"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    log("[SAVE]", out_png)

    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight")
        log("[SAVE]", out_svg)

    plt.close(fig)


# ============================================================
# 3) MAIN
# ============================================================

def main():
    tf_auc = load_auc_values(
        TF_AUC_CSV,
        auc_col_candidates=["auc", "tf_auc", "mean_auc"],
    )

    rna_auc = load_auc_values(
        RNA_AUC_CSV,
        auc_col_candidates=["cell_auc", "auc", "rna_auc", "mean_auc"],
    )

    log("\n" + "=" * 80)
    log("[PLOT] combined bowling with scatter")
    log("=" * 80)
    plot_combined_bowling(
        tf_auc,
        rna_auc,
        show_scatter=True,
        out_png=OUT_WITH_SCATTER_PNG,
        out_svg=OUT_WITH_SCATTER_SVG,
    )

    log("\n" + "=" * 80)
    log("[PLOT] combined bowling without scatter")
    log("=" * 80)
    plot_combined_bowling(
        tf_auc,
        rna_auc,
        show_scatter=False,
        out_png=OUT_NO_SCATTER_PNG,
        out_svg=OUT_NO_SCATTER_SVG,
    )

    log("\n[DONE]")


if __name__ == "__main__":
    main()
