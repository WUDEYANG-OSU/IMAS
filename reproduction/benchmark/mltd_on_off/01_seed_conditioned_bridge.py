# -*- coding: utf-8 -*-
"""
plot_CRC_MLTD_bridge_promotion_independent_sides_NSC_v1_3.py

Nature/Science/Cell-style seed-conditioned RNA–TF–RE bridge analysis.

核心问题
--------
对于同一个 RNA perturbation seed，MLTD 是否让：

    seed-specific RNA→TF support

与：

    TF-linked RE support

围绕相同 TF program 发生更协调的配对？

与上一版的区别
--------------
1. 统计单位改为 RNA seed，而不是把 10 个 TF 当成独立重复。
2. 每个 seed × TF 都有独立 bridge score。
3. Full 与 no-MLTD / ablation 使用相同 RNA、TF、RE 候选空间。
4. 两侧都转换为 seed内/模式内 percentile，控制分数范围与传播预算。
5. TF-label rewiring 同时保持每个 seed 的两侧边际分布，只破坏TF身份配对。
6. 主图改成期刊正文风格的六个紧凑panel：
   A 机制示意
   B 代表性seed的TF配对位移
   C 所有seed×TF的bridge gain热图
   D seed级Full vs Separate paired comparison
   E Full相对各对照的paired effect size
   F native pairing vs TF-label rewired null

严格bridge定义
--------------
For seed s and TF t:

    U_st = percentile(RNA_s → TF_t support)
    R_st = percentile(TF_t → RE support under that mode)
    B_st = min(U_st, R_st)

任何一侧很弱，B_st都会被限制。

解释边界
--------
这是模型内部的seed-conditioned structural transmission analysis，
不是实验扰动准确率验证。

推荐表述：
    MLTD strengthens seed-conditioned RNA–TF–RE bridging
    through TF-identity-matched cross-layer coupling.

输入
----
RNA top:
  single_sample_<SAMPLE_ID>_network_aware_top_rna/
  <SAMPLE_ID>_final_top10_rnas.csv

TF top:
  single_sample_<SAMPLE_ID>_network_aware_top_tf/
  <SAMPLE_ID>_final_top10_tfs.csv

RNA–TF decomposition:
  single_sample_<SAMPLE_ID>_rna_tf_top10_network_heatmap/qc/
  <SAMPLE_ID>_qc_score_decomposition_table.csv

TF-linked RE exports:
  single_sample_<SAMPLE_ID>_rna_tf_re_three_layer_v1/
  <SAMPLE_ID>_TF_linked_RE_peaks_v1.csv
  <SAMPLE_ID>_TF_linked_RE_peak_arcs_v1.csv

Optional TF program ablations:
  single_sample_<SAMPLE_ID>_network_aware_top_tf_qc_allinone/04_ablation/

运行
----
%run /path/13_plot_CRC_MLTD_seed_conditioned_bridge_promotion_NSC.py
"""

from __future__ import annotations

import itertools
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

try:
    from scipy.stats import spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =============================================================================
# 0. CONFIG
# =============================================================================

DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

BASE = get_project_root()
RUN_BASE = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)

RNA_TOP_CSV = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_network_aware_top_rna"
    / f"{SAMPLE_ID}_final_top10_rnas.csv"
)
TF_TOP_CSV = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_network_aware_top_tf"
    / f"{SAMPLE_ID}_final_top10_tfs.csv"
)
RNA_TF_DECOMP_CSV = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_rna_tf_top10_network_heatmap"
    / "qc"
    / f"{SAMPLE_ID}_qc_score_decomposition_table.csv"
)
THREE_LAYER_DIR = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_rna_tf_re_three_layer_v1"
)
TF_RE_PEAK_CSV = THREE_LAYER_DIR / f"{SAMPLE_ID}_TF_linked_RE_peaks_v1.csv"
TF_RE_ARC_CSV = THREE_LAYER_DIR / f"{SAMPLE_ID}_TF_linked_RE_peak_arcs_v1.csv"

TF_ABLATION_DIR = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_network_aware_top_tf_qc_allinone"
    / "04_ablation"
)

OUT_DIR = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_bridge_promotion_independent_sides_NSC_v1_3"
)

OUT_PAIR_TABLE = OUT_DIR / f"{SAMPLE_ID}_seed_TF_bridge_scores.csv"
OUT_SEED_SUMMARY = OUT_DIR / f"{SAMPLE_ID}_seed_bridge_summary.csv"
OUT_MODE_EFFECTS = OUT_DIR / f"{SAMPLE_ID}_bridge_mode_effects.csv"
OUT_NULL = OUT_DIR / f"{SAMPLE_ID}_TF_label_rewired_null.csv"
OUT_STATS = OUT_DIR / f"{SAMPLE_ID}_bridge_statistics.csv"
OUT_MANIFEST = OUT_DIR / f"{SAMPLE_ID}_bridge_manifest.json"

OUT_PNG = OUT_DIR / f"{SAMPLE_ID}_bridge_promotion_independent_sides_NSC_v1_3.png"
OUT_SVG = OUT_DIR / f"{SAMPLE_ID}_bridge_promotion_independent_sides_NSC_v1_3.svg"
OUT_PDF = OUT_DIR / f"{SAMPLE_ID}_bridge_promotion_independent_sides_NSC_v1_3.pdf"

# RNA–TF decomposition weights used by the original network-aware heatmap.
W_CORR = 0.45
W_LATENT = 0.40
W_PRIOR = 0.15

# TF-linked RE evidence.
RE_W_PEAK_COUNT = 0.22
RE_W_TARGET_BREADTH = 0.18
RE_W_TARGET_RANK = 0.15
RE_W_ARC_STRENGTH = 0.32
RE_W_ARC_BREADTH = 0.13

# The TF-program gate is applied ONLY to the RNA-facing upstream score.
# It is never applied to the independently constructed TF→RE score.
# This prevents the same TF gate from mathematically inducing correlation
# between the two sides of the bridge.
UPSTREAM_GATE_FLOOR = 0.20

# Complete bridge requires both sides in the upper tail.
COMPLETE_THRESHOLD = 0.65

# Statistics.
N_BOOTSTRAP = 10000
N_REWIRED = 10000
RNG_SEED = 42

# Figure.
FIG_W_IN = 7.25
FIG_H_IN = 8.20
DPI = 400

RNA_COLOR = "#45B8B5"
TF_COLOR = "#E56B67"
RE_COLOR = "#6677B8"
FULL_COLOR = "#3E6E9E"
BASELINE_COLOR = "#A2A6AA"
ABLATION_COLOR = "#B68A4C"
POS_COLOR = "#B64845"
NEG_COLOR = "#4677A8"
NULL_COLOR = "#D7D9DD"
DARK = "#202124"
MID_GREY = "#6F7478"
LIGHT_GREY = "#D9DDE1"
GRID_GREY = "#E9EBED"

# Selected labels in dense panels.
N_LABEL_TF_REPRESENTATIVE = 5


# =============================================================================
# 1. STYLE
# =============================================================================

def set_publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def clean_axis(ax, keep=("left", "bottom")):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(direction="out", pad=2)


def panel_label(ax, letter: str):
    ax.text(
        -0.16,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=8.6,
        fontweight="bold",
        va="top",
        ha="left",
        color=DARK,
    )


# =============================================================================
# 2. HELPERS
# =============================================================================

def log(*args):
    print(*args, flush=True)


def canonical(x: Any) -> str:
    return str(x).strip().upper()


def minmax01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return out
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if hi - lo < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (x[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    """
    Average-tie percentile ranks in [0, 1].
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([1.0], dtype=float)

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i + 1
        while j < n and np.isclose(xs[j], xs[i], rtol=0, atol=1e-12):
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j

    return ranks / float(n - 1)


def safe_spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    if np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    if SCIPY_OK:
        return float(spearmanr(x[m], y[m]).correlation)
    return float(
        np.corrcoef(
            rank_percentile(x[m]),
            rank_percentile(y[m]),
        )[0, 1]
    )


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out


def exact_signflip_greater(differences: np.ndarray) -> dict[str, float]:
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "exact_signflip_p": np.nan,
            "n_seed": 0,
        }

    observed = float(np.mean(d))
    signs = np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=n)),
        dtype=float,
    )
    null = (signs * d[None, :]).mean(axis=1)
    p = float(
        (np.sum(null >= observed - 1e-15) + 1)
        / (len(null) + 1)
    )
    return {
        "mean_difference": observed,
        "median_difference": float(np.median(d)),
        "exact_signflip_p": p,
        "n_seed": int(n),
    }


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(RNG_SEED)
    means = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        means[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    return (
        float(np.mean(x)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def save_figure(fig):
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", OUT_PNG)
    log("[SAVE]", OUT_SVG)
    log("[SAVE]", OUT_PDF)


# =============================================================================
# 2B. ROBUST COLUMN RESOLUTION
# =============================================================================

def normalized_column_name(name: Any) -> str:
    """Normalize a column name for case-/separator-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_column(
    df: pd.DataFrame,
    aliases: list[str],
    *,
    role: str,
    path: Path | None = None,
    required: bool = True,
) -> str | None:
    """
    Resolve a column by common aliases.

    Resolution order:
    1. case-/separator-insensitive exact alias match;
    2. normalized prefix/suffix match for aliases of length >= 4.

    A detailed error lists the actual columns and accepted aliases.
    """
    normalized_to_actual: dict[str, str] = {}
    for col in df.columns:
        normalized_to_actual.setdefault(normalized_column_name(col), str(col))

    for alias in aliases:
        key = normalized_column_name(alias)
        if key in normalized_to_actual:
            return normalized_to_actual[key]

    # Conservative fuzzy fallback for names such as selected_gene_symbol.
    candidates: list[tuple[int, str]] = []
    alias_keys = [
        normalized_column_name(x)
        for x in aliases
        if len(normalized_column_name(x)) >= 4
    ]
    for col in df.columns:
        col_key = normalized_column_name(col)
        for alias_key in alias_keys:
            if col_key.startswith(alias_key) or col_key.endswith(alias_key):
                candidates.append((abs(len(col_key) - len(alias_key)), str(col)))
                break

    if candidates:
        candidates.sort(key=lambda x: (x[0], len(x[1]), x[1]))
        chosen = candidates[0][1]
        log(
            f"[COLUMN] {role}: using fuzzy match {chosen!r}; "
            f"accepted aliases={aliases}"
        )
        return chosen

    if not required:
        return None

    location = f" in {path}" if path is not None else ""
    raise KeyError(
        f"Unable to identify the {role} column{location}.\n"
        f"Actual columns: {df.columns.tolist()}\n"
        f"Accepted aliases: {aliases}"
    )


RNA_ITEM_ALIASES = [
    "rna",
    "rna_name",
    "rna_symbol",
    "gene",
    "gene_symbol",
    "gene_name",
    "symbol",
    "node",
    "feature",
    "feature_name",
    "candidate_gene",
    "target_gene",
    "selected_gene",
]

TF_ITEM_ALIASES = [
    "tf",
    "tf_name",
    "tf_symbol",
    "transcription_factor",
    "transcriptionfactor",
    "regulator",
    "factor",
    "gene",
    "gene_symbol",
    "gene_name",
    "symbol",
    "node",
    "feature",
    "feature_name",
    "selected_tf",
]

RANKED_SCORE_ALIASES = [
    "final_score",
    "finalscore",
    "score",
    "rank_score",
    "network_score",
    "mltd_score",
    "combined_score",
    "total_score",
    "pre_score",
    "activity_norm",
    "activity_score",
    "importance",
    "weight",
    "selection_score",
]

RANK_COLUMN_ALIASES = [
    "rank",
    "final_rank",
    "selection_rank",
    "network_rank",
    "order",
]

DECOMP_RNA_ALIASES = RNA_ITEM_ALIASES
DECOMP_TF_ALIASES = TF_ITEM_ALIASES

DECOMP_CORR_ALIASES = [
    "corr_abs",
    "abs_corr",
    "correlation_abs",
    "absolute_correlation",
    "corr",
    "correlation",
    "expression_correlation",
]

DECOMP_EDGE_ALIASES = [
    "edge",
    "edge_score",
    "latent_edge",
    "latent_score",
    "model_edge",
    "mltd_edge",
    "representation_edge",
]

DECOMP_PRIOR_ALIASES = [
    "prior",
    "prior_score",
    "tf_gene_prior",
    "regulatory_prior",
    "prior_support",
]

DECOMP_FINAL_ALIASES = [
    "final_raw_abs",
    "final_score",
    "pair_score",
    "network_score",
    "combined_score",
    "rna_tf_score",
    "mltd_score",
]


# =============================================================================
# 3. INPUTS
# =============================================================================

def validate_inputs():
    required = [
        RNA_TOP_CSV,
        TF_TOP_CSV,
        RNA_TF_DECOMP_CSV,
        TF_RE_PEAK_CSV,
        TF_RE_ARC_CSV,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n"
            + "\n".join(missing)
            + "\n\nRun the RNA/TF ranking, RNA–TF QC, and three-layer "
              "architecture scripts first."
        )


def load_ranked_list(path: Path, item_col: str) -> pd.DataFrame:
    """
    Load a ranked RNA or TF table while accepting common column aliases.

    The returned table always contains canonical columns:
      - RNA: rna, rna_key
      - TF:  tf, tf_key
      - rank_score_norm

    If no score column exists, the existing row order is treated as the rank.
    """
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"Ranked table is empty: {path}")

    if item_col == "rna":
        aliases = RNA_ITEM_ALIASES
        role = "RNA/gene identifier"
    elif item_col == "tf":
        aliases = TF_ITEM_ALIASES
        role = "TF identifier"
    else:
        aliases = [item_col]
        role = item_col

    actual_item_col = resolve_column(
        df,
        aliases,
        role=role,
        path=path,
        required=True,
    )

    score_col = resolve_column(
        df,
        RANKED_SCORE_ALIASES,
        role=f"{item_col} ranking score",
        path=path,
        required=False,
    )

    rank_col = None
    if score_col is None:
        rank_col = resolve_column(
            df,
            RANK_COLUMN_ALIASES,
            role=f"{item_col} rank",
            path=path,
            required=False,
        )

    out = df.copy()
    if actual_item_col != item_col:
        if item_col in out.columns:
            # Avoid silently overwriting an unrelated pre-existing column.
            out = out.rename(columns={item_col: f"original_{item_col}"})
        out = out.rename(columns={actual_item_col: item_col})

    out[item_col] = out[item_col].astype(str).str.strip()
    out = out[
        out[item_col].notna()
        & (out[item_col] != "")
        & (out[item_col].str.lower() != "nan")
    ].copy()

    if score_col is not None:
        if score_col == actual_item_col:
            raise ValueError(
                f"The identifier and score resolved to the same column "
                f"{score_col!r} in {path}. Actual columns={df.columns.tolist()}"
            )
        score_values = pd.to_numeric(
            df.loc[out.index, score_col],
            errors="coerce",
        )
        if score_values.notna().sum() == 0:
            warnings.warn(
                f"{path}: resolved score column {score_col!r} contains no "
                "numeric values; preserving file order instead."
            )
            out["_resolved_rank_score"] = np.arange(
                len(out), 0, -1, dtype=float
            )
            score_source = "file order"
        else:
            minimum = float(score_values.dropna().min())
            out["_resolved_rank_score"] = score_values.fillna(
                minimum - 1.0
            ).to_numpy(float)
            score_source = score_col

    elif rank_col is not None:
        rank_values = pd.to_numeric(
            df.loc[out.index, rank_col],
            errors="coerce",
        )
        if rank_values.notna().sum() == 0:
            out["_resolved_rank_score"] = np.arange(
                len(out), 0, -1, dtype=float
            )
            score_source = "file order"
        else:
            maximum = float(rank_values.dropna().max())
            # Smaller rank means stronger candidate.
            out["_resolved_rank_score"] = -rank_values.fillna(
                maximum + 1.0
            ).to_numpy(float)
            score_source = f"negative {rank_col}"

    else:
        out["_resolved_rank_score"] = np.arange(
            len(out), 0, -1, dtype=float
        )
        score_source = "file order"

    out = (
        out.sort_values("_resolved_rank_score", ascending=False)
        .drop_duplicates(item_col, keep="first")
        .reset_index(drop=True)
    )
    out[f"{item_col}_key"] = out[item_col].map(canonical)
    out["rank_score_norm"] = minmax01(
        out["_resolved_rank_score"].to_numpy(float)
    )
    out["resolved_identifier_column"] = actual_item_col
    out["resolved_score_source"] = score_source

    log(
        f"[LOAD] {path.name}: {item_col} column={actual_item_col!r}; "
        f"score source={score_source!r}; n={len(out)}"
    )
    return out


def load_decomposition(
    rna_keys: set[str],
    tf_keys: set[str],
) -> pd.DataFrame:
    """
    Load RNA–TF pair decomposition with robust alias resolution.

    RNA, TF and expression-correlation columns are required. Missing latent
    or prior columns are set to zero with an explicit warning so the script
    can still run, while the manifest/output preserves that limitation.
    """
    raw = pd.read_csv(RNA_TF_DECOMP_CSV)
    if len(raw) == 0:
        raise ValueError(f"RNA–TF decomposition is empty: {RNA_TF_DECOMP_CSV}")

    rna_col = resolve_column(
        raw,
        DECOMP_RNA_ALIASES,
        role="RNA identifier in RNA–TF decomposition",
        path=RNA_TF_DECOMP_CSV,
        required=True,
    )
    tf_col = resolve_column(
        raw,
        DECOMP_TF_ALIASES,
        role="TF identifier in RNA–TF decomposition",
        path=RNA_TF_DECOMP_CSV,
        required=True,
    )
    corr_col = resolve_column(
        raw,
        DECOMP_CORR_ALIASES,
        role="RNA–TF expression-association score",
        path=RNA_TF_DECOMP_CSV,
        required=True,
    )
    edge_col = resolve_column(
        raw,
        DECOMP_EDGE_ALIASES,
        role="RNA–TF latent/model edge score",
        path=RNA_TF_DECOMP_CSV,
        required=False,
    )
    prior_col = resolve_column(
        raw,
        DECOMP_PRIOR_ALIASES,
        role="RNA–TF prior score",
        path=RNA_TF_DECOMP_CSV,
        required=False,
    )
    final_col = resolve_column(
        raw,
        DECOMP_FINAL_ALIASES,
        role="saved final RNA–TF pair score",
        path=RNA_TF_DECOMP_CSV,
        required=False,
    )

    df = pd.DataFrame({
        "rna": raw[rna_col].astype(str).str.strip(),
        "tf": raw[tf_col].astype(str).str.strip(),
        "corr_abs": pd.to_numeric(
            raw[corr_col], errors="coerce"
        ).fillna(0.0).abs(),
    })

    if edge_col is None:
        warnings.warn(
            f"{RNA_TF_DECOMP_CSV}: no latent/model-edge column was found. "
            "The latent component is set to zero."
        )
        df["edge"] = 0.0
    else:
        df["edge"] = pd.to_numeric(
            raw[edge_col], errors="coerce"
        ).fillna(0.0).abs()

    if prior_col is None:
        warnings.warn(
            f"{RNA_TF_DECOMP_CSV}: no prior column was found. "
            "The prior component is set to zero."
        )
        df["prior"] = 0.0
    else:
        df["prior"] = pd.to_numeric(
            raw[prior_col], errors="coerce"
        ).fillna(0.0).abs()

    df["rna_key"] = df["rna"].map(canonical)
    df["tf_key"] = df["tf"].map(canonical)
    df = df[
        df["rna_key"].isin(rna_keys)
        & df["tf_key"].isin(tf_keys)
    ].copy()

    if len(df) == 0:
        raise RuntimeError(
            "No RNA–TF decomposition rows remain after matching the ranked "
            "RNA and TF lists.\n"
            f"Ranked RNA keys (first 10): {sorted(rna_keys)[:10]}\n"
            f"Ranked TF keys (first 10): {sorted(tf_keys)[:10]}\n"
            f"Resolved decomposition columns: RNA={rna_col!r}, TF={tf_col!r}"
        )

    # Collapse accidental duplicate RNA–TF pairs by retaining their strongest
    # evidence on each component.
    df = (
        df.groupby(["rna", "rna_key", "tf", "tf_key"], as_index=False)
        .agg(
            corr_abs=("corr_abs", "max"),
            edge=("edge", "max"),
            prior=("prior", "max"),
        )
    )

    df["score_full"] = (
        W_CORR * df["corr_abs"]
        + W_LATENT * df["edge"]
        + W_PRIOR * df["prior"]
    )
    df["score_separate"] = df["corr_abs"]
    df["score_no_prior"] = (
        W_CORR * df["corr_abs"]
        + W_LATENT * df["edge"]
    )
    df["score_no_latent"] = (
        W_CORR * df["corr_abs"]
        + W_PRIOR * df["prior"]
    )
    df["score_corr_only"] = df["corr_abs"]

    if final_col is not None:
        final_values = pd.to_numeric(
            raw[final_col], errors="coerce"
        ).fillna(0.0).abs()
        raw_check = pd.DataFrame({
            "rna_key": raw[rna_col].astype(str).map(canonical),
            "tf_key": raw[tf_col].astype(str).map(canonical),
            "saved_final": final_values,
        })
        raw_check = (
            raw_check.groupby(["rna_key", "tf_key"], as_index=False)
            .agg(saved_final=("saved_final", "max"))
        )
        check = df.merge(
            raw_check,
            on=["rna_key", "tf_key"],
            how="left",
        )
        max_diff = float(
            np.nanmax(
                np.abs(
                    check["score_full"].to_numpy(float)
                    - check["saved_final"].fillna(0.0).to_numpy(float)
                )
            )
        )
        if max_diff > 1e-6:
            warnings.warn(
                "Recomputed RNA–TF full scores differ from the saved final "
                f"score column {final_col!r}; maximum difference={max_diff:.3g}. "
                "The transparent component-based recomputation is used."
            )

    df["resolved_rna_column"] = rna_col
    df["resolved_tf_column"] = tf_col
    df["resolved_corr_column"] = corr_col
    df["resolved_edge_column"] = edge_col or "missing→0"
    df["resolved_prior_column"] = prior_col or "missing→0"
    df["resolved_final_column"] = final_col or "not available"

    log(
        "[LOAD] RNA–TF decomposition columns:",
        {
            "rna": rna_col,
            "tf": tf_col,
            "corr": corr_col,
            "edge": edge_col,
            "prior": prior_col,
            "final": final_col,
        },
    )
    log("[LOAD] matched RNA–TF pairs:", len(df))
    return df


def load_re_tables(tf_keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    peaks = pd.read_csv(TF_RE_PEAK_CSV)
    arcs = pd.read_csv(TF_RE_ARC_CSV)

    peak_tf_col = resolve_column(
        peaks,
        TF_ITEM_ALIASES,
        role="TF identifier in TF-linked RE peak table",
        path=TF_RE_PEAK_CSV,
        required=True,
    )
    arc_tf_col = resolve_column(
        arcs,
        TF_ITEM_ALIASES,
        role="TF identifier in TF-linked RE arc table",
        path=TF_RE_ARC_CSV,
        required=True,
    )

    if peak_tf_col != "tf":
        peaks = peaks.rename(columns={peak_tf_col: "tf"})
    if arc_tf_col != "tf":
        arcs = arcs.rename(columns={arc_tf_col: "tf"})

    peaks["tf_key"] = peaks["tf"].map(canonical)
    arcs["tf_key"] = arcs["tf"].map(canonical)
    peaks = peaks[peaks["tf_key"].isin(tf_keys)].copy()
    arcs = arcs[arcs["tf_key"].isin(tf_keys)].copy()

    if "peak_id" not in peaks.columns:
        peaks["peak_id"] = np.arange(len(peaks))
    if "gene" not in peaks.columns:
        peaks["gene"] = "unknown"
    if "target_gene_rank" not in peaks.columns:
        peaks["target_gene_rank"] = 1.0

    peaks["target_gene_rank"] = pd.to_numeric(
        peaks["target_gene_rank"], errors="coerce"
    ).fillna(1.0)

    if "arc_weight" not in arcs.columns:
        raise KeyError(f"{TF_RE_ARC_CSV} lacks arc_weight")
    arcs["arc_weight"] = pd.to_numeric(
        arcs["arc_weight"], errors="coerce"
    ).fillna(0.0)
    return peaks, arcs


# =============================================================================
# 4. MODE-SPECIFIC TF GATES
# =============================================================================

def load_optional_gate(
    path: Path,
    tf_keys: list[str],
) -> tuple[np.ndarray | None, str]:
    if not path.exists():
        return None, "missing"

    df = pd.read_csv(path)
    if "tf" not in df.columns:
        return None, "missing_tf"

    score_col = next(
        (
            c for c in [
                "final_score",
                "pre_score",
                "activity_norm",
                "activity_score",
            ]
            if c in df.columns
        ),
        None,
    )
    if score_col is None:
        return None, "missing_score"

    df["tf_key"] = df["tf"].map(canonical)
    score_map = dict(
        zip(
            df["tf_key"],
            pd.to_numeric(df[score_col], errors="coerce").fillna(0.0),
        )
    )
    values = np.asarray(
        [score_map.get(k, 0.0) for k in tf_keys],
        dtype=float,
    )
    return minmax01(values), f"{path.name}:{score_col}"


def build_mode_gates(
    tf_table: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    tf_keys = tf_table["tf_key"].tolist()
    full_gate = tf_table["rank_score_norm"].to_numpy(float)

    gates = {
        "Full MLTD": full_gate,
        "Separate layers": np.ones(len(tf_keys), dtype=float),
    }
    sources = {
        "Full MLTD": f"{TF_TOP_CSV.name}:full final score",
        "Separate layers": "no joint TF-program gate",
    }

    mode_files = {
        "Prior removed": TF_ABLATION_DIR / "corr_plus_latent_final_top_tfs.csv",
        "Latent removed": TF_ABLATION_DIR / "corr_plus_prior_final_top_tfs.csv",
        "Continuity removed": TF_ABLATION_DIR / "full_no_continuity_final_top_tfs.csv",
        "Corr only": TF_ABLATION_DIR / "corr_only_final_top_tfs.csv",
    }

    for mode, path in mode_files.items():
        gate, source = load_optional_gate(path, tf_keys)
        if gate is None:
            gate = full_gate.copy()
            source = (
                f"{source}; fallback to Full gate while the RNA–TF "
                "component score is changed"
            )
        gates[mode] = gate
        sources[mode] = source

    return gates, sources


# =============================================================================
# 5. TF-LINKED RE SUPPORT
# =============================================================================

def aggregate_re_support(
    peaks: pd.DataFrame,
    arcs: pd.DataFrame,
    tf_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in tf_table.iterrows():
        tf = row["tf"]
        key = row["tf_key"]
        p = peaks[peaks["tf_key"] == key]
        a = arcs[arcs["tf_key"] == key]

        peak_n = int(p["peak_id"].nunique()) if len(p) else 0
        gene_n = int(p["gene"].nunique()) if len(p) else 0
        arc_n = int(len(a))

        rank_quality = (
            float(np.mean(1.0 / (p["target_gene_rank"] + 1.0)))
            if len(p) else 0.0
        )
        arc_top = (
            float(np.mean(
                np.sort(a["arc_weight"].to_numpy(float))[::-1]
                [:min(10, len(a))]
            ))
            if len(a) else 0.0
        )

        rows.append({
            "tf": tf,
            "tf_key": key,
            "peak_n": peak_n,
            "target_gene_n": gene_n,
            "arc_n": arc_n,
            "target_rank_quality": rank_quality,
            "top_arc_weight": arc_top,
        })

    out = pd.DataFrame(rows)
    out["peak_n_norm"] = minmax01(out["peak_n"].to_numpy(float))
    out["target_gene_n_norm"] = minmax01(
        out["target_gene_n"].to_numpy(float)
    )
    out["target_rank_norm"] = minmax01(
        out["target_rank_quality"].to_numpy(float)
    )
    out["arc_strength_norm"] = minmax01(
        out["top_arc_weight"].to_numpy(float)
    )
    out["arc_n_norm"] = minmax01(out["arc_n"].to_numpy(float))

    out["re_intrinsic"] = (
        RE_W_PEAK_COUNT * out["peak_n_norm"]
        + RE_W_TARGET_BREADTH * out["target_gene_n_norm"]
        + RE_W_TARGET_RANK * out["target_rank_norm"]
        + RE_W_ARC_STRENGTH * out["arc_strength_norm"]
        + RE_W_ARC_BREADTH * out["arc_n_norm"]
    )
    return out


# =============================================================================
# 6. SEED-CONDITIONED BRIDGES
# =============================================================================

MODE_SCORE_COLUMNS = {
    "Full MLTD": "score_full",
    "Separate layers": "score_separate",
    "Prior removed": "score_no_prior",
    "Latent removed": "score_no_latent",
    "Continuity removed": "score_full",
    "Corr only": "score_corr_only",
}


def build_seed_tf_bridge_table(
    rna_table: pd.DataFrame,
    tf_table: pd.DataFrame,
    decomp: pd.DataFrame,
    re_support: pd.DataFrame,
    gates: dict[str, np.ndarray],
    gate_sources: dict[str, str],
) -> pd.DataFrame:
    """
    Construct seed-conditioned RNA–TF–RE bridges with strictly independent
    RNA-facing and RE-facing scores.

    RNA-facing side
    ---------------
    U_st is produced only from seed-specific RNA→TF information:
      - expression association;
      - latent/model RNA→TF edge;
      - RNA→TF prior;
      - optional upstream-only TF-program continuity gate.

    RE-facing side
    --------------
    R_t is produced only from independently exported TF→RE evidence:
      - TF-linked peak number;
      - TF target-gene breadth/rank;
      - peak-arc strength/breadth.

    Crucially, the TF-program gate is NOT multiplied into R_t. Therefore the
    two sides cannot become correlated merely because they share a gate.

    For every mode and seed:
        U_percentile = rank_TF(U_st)
        R_percentile = rank_TF(R_t)
        bridge = min(U_percentile, R_percentile)

    R_t and its percentile are identical across modes and seeds. Only the
    independently derived RNA-facing side changes.
    """
    tf_order = tf_table[["tf", "tf_key"]].copy()

    re_map = (
        re_support
        .set_index("tf_key")["re_intrinsic"]
        .to_dict()
    )
    independent_re_raw = np.asarray(
        [
            float(re_map.get(tf_key, 0.0))
            for tf_key in tf_order["tf_key"]
        ],
        dtype=float,
    )
    independent_re_pct = rank_percentile(independent_re_raw)

    rows = []

    for _, seed_row in rna_table.iterrows():
        seed = str(seed_row["rna"])
        seed_key = str(seed_row["rna_key"])

        seed_pairs = decomp[
            decomp["rna_key"] == seed_key
        ].copy()
        seed_pairs = tf_order.merge(
            seed_pairs.drop(columns=["tf"], errors="ignore"),
            on="tf_key",
            how="left",
            validate="one_to_one",
        )
        seed_pairs = seed_pairs.fillna(0.0)

        for mode, score_col in MODE_SCORE_COLUMNS.items():
            upstream_gate = np.asarray(
                gates[mode],
                dtype=float,
            )
            upstream_gate_effective = (
                UPSTREAM_GATE_FLOOR
                + (1.0 - UPSTREAM_GATE_FLOOR) * upstream_gate
            )

            # Separate layers deliberately uses no joint TF-program gate.
            if mode == "Separate layers":
                upstream_gate_effective = np.ones_like(
                    upstream_gate_effective
                )

            upstream_component_raw = pd.to_numeric(
                seed_pairs[score_col],
                errors="coerce",
            ).fillna(0.0).to_numpy(float)

            # Only the RNA-facing side receives the upstream program gate.
            upstream_raw = (
                upstream_component_raw
                * upstream_gate_effective
            )
            upstream_pct = rank_percentile(upstream_raw)

            # Strictly independent, fixed TF→RE anchor.
            re_raw = independent_re_raw.copy()
            re_pct = independent_re_pct.copy()

            bridge = np.minimum(upstream_pct, re_pct)
            harmonic = (
                2.0 * upstream_pct * re_pct
                / (upstream_pct + re_pct + 1e-12)
            )
            balance = 1.0 - np.abs(upstream_pct - re_pct)
            complete = (
                (upstream_pct >= COMPLETE_THRESHOLD)
                & (re_pct >= COMPLETE_THRESHOLD)
            )

            for j, tf_row in tf_order.iterrows():
                rows.append({
                    "seed_rna": seed,
                    "seed_key": seed_key,
                    "tf": str(tf_row["tf"]),
                    "tf_key": str(tf_row["tf_key"]),
                    "mode": mode,

                    # RNA-facing side.
                    "rna_tf_component_raw": float(
                        upstream_component_raw[j]
                    ),
                    "rna_tf_upstream_gate": float(
                        upstream_gate[j]
                    ),
                    "rna_tf_gate_effective": float(
                        upstream_gate_effective[j]
                    ),
                    "rna_tf_raw": float(upstream_raw[j]),
                    "rna_tf_percentile": float(upstream_pct[j]),

                    # Independent RE-facing side.
                    "tf_re_raw": float(re_raw[j]),
                    "tf_re_percentile": float(re_pct[j]),
                    "tf_re_source": (
                        "independent TF-target/gene-peak/peak-arc evidence"
                    ),
                    "tf_re_is_mode_invariant": True,

                    # Bridge outputs.
                    "bridge_score": float(bridge[j]),
                    "harmonic_bridge": float(harmonic[j]),
                    "bridge_balance": float(balance[j]),
                    "complete_bridge": bool(complete[j]),

                    # Backward-compatible metadata.
                    "gate_value": float(upstream_gate[j]),
                    "gate_source": gate_sources[mode],
                    "shared_gate_across_sides": False,
                })

    out = pd.DataFrame(rows)

    # Hard QC: the RE-facing score must be invariant across every mode/seed
    # for the same TF. Failure here means independence was accidentally lost.
    re_qc = (
        out.groupby("tf_key")
        .agg(
            n_re_raw=("tf_re_raw", "nunique"),
            n_re_pct=("tf_re_percentile", "nunique"),
        )
    )
    if (
        (re_qc["n_re_raw"] != 1).any()
        or (re_qc["n_re_pct"] != 1).any()
    ):
        bad = re_qc[
            (re_qc["n_re_raw"] != 1)
            | (re_qc["n_re_pct"] != 1)
        ]
        raise RuntimeError(
            "Independent-side QC failed: TF→RE scores changed across "
            f"modes/seeds for TFs {bad.index.tolist()}"
        )

    return out


def summarize_by_seed(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, mode), g in pair_table.groupby(
        ["seed_rna", "mode"], sort=False
    ):
        rows.append({
            "seed_rna": seed,
            "mode": mode,
            "mean_bridge_score": float(g["bridge_score"].mean()),
            "median_bridge_score": float(g["bridge_score"].median()),
            "mean_harmonic_bridge": float(g["harmonic_bridge"].mean()),
            "mean_bridge_balance": float(g["bridge_balance"].mean()),
            "complete_bridge_fraction": float(g["complete_bridge"].mean()),
            "complete_bridge_n": int(g["complete_bridge"].sum()),
            "rna_re_spearman": safe_spearman(
                g["rna_tf_percentile"],
                g["tf_re_percentile"],
            ),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 7. PAIRED MODE EFFECTS
# =============================================================================

def paired_mode_effects(seed_summary: pd.DataFrame) -> pd.DataFrame:
    full = seed_summary[
        seed_summary["mode"] == "Full MLTD"
    ][
        [
            "seed_rna",
            "mean_bridge_score",
            "complete_bridge_fraction",
            "mean_harmonic_bridge",
        ]
    ].copy()

    rows = []
    control_modes = [
        x for x in MODE_SCORE_COLUMNS
        if x != "Full MLTD"
    ]

    for mode in control_modes:
        other = seed_summary[
            seed_summary["mode"] == mode
        ][
            [
                "seed_rna",
                "mean_bridge_score",
                "complete_bridge_fraction",
                "mean_harmonic_bridge",
            ]
        ].copy()

        merged = full.merge(
            other,
            on="seed_rna",
            suffixes=("_full", "_control"),
            validate="one_to_one",
        )

        for metric in [
            "mean_bridge_score",
            "complete_bridge_fraction",
            "mean_harmonic_bridge",
        ]:
            difference = (
                merged[f"{metric}_full"]
                - merged[f"{metric}_control"]
            ).to_numpy(float)
            test = exact_signflip_greater(difference)
            mean, lo, hi = bootstrap_mean_ci(difference)
            rows.append({
                "comparison": f"Full MLTD vs {mode}",
                "control_mode": mode,
                "metric": metric,
                "full_mean": float(merged[f"{metric}_full"].mean()),
                "control_mean": float(
                    merged[f"{metric}_control"].mean()
                ),
                "mean_difference": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "exact_signflip_p": test["exact_signflip_p"],
                "n_seed": test["n_seed"],
            })

    out = pd.DataFrame(rows)
    out["fdr"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        idx = list(idx)
        out.loc[idx, "fdr"] = bh_fdr(
            out.loc[idx, "exact_signflip_p"].to_numpy(float)
        )
    return out


# =============================================================================
# 8. TF-LABEL REWIRED NULL
# =============================================================================

def run_tf_label_rewired_null(
    pair_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pair_table[
        pair_table["mode"] == "Full MLTD"
    ].copy()

    observed_seed = (
        full.groupby("seed_rna")["bridge_score"]
        .mean()
    )
    observed = float(observed_seed.mean())

    rng = np.random.default_rng(RNG_SEED)
    null_values = np.empty(N_REWIRED, dtype=float)

    groups = [
        g.sort_values("tf_key").copy()
        for _, g in full.groupby("seed_rna", sort=False)
    ]

    for b in range(N_REWIRED):
        seed_means = []
        for g in groups:
            upstream = g["rna_tf_percentile"].to_numpy(float)
            re_side = rng.permutation(
                g["tf_re_percentile"].to_numpy(float)
            )
            seed_means.append(
                float(np.minimum(upstream, re_side).mean())
            )
        null_values[b] = float(np.mean(seed_means))

    p = float(
        (np.sum(null_values >= observed - 1e-15) + 1)
        / (len(null_values) + 1)
    )

    null = pd.DataFrame({
        "rewire_id": np.arange(N_REWIRED),
        "mean_seed_bridge_score": null_values,
    })
    summary = pd.DataFrame([{
        "metric": "mean seed bridge score",
        "observed_native_matching": observed,
        "rewired_mean": float(np.mean(null_values)),
        "rewired_median": float(np.median(null_values)),
        "rewired_sd": float(np.std(null_values)),
        "native_minus_rewired_median": float(
            observed - np.median(null_values)
        ),
        "empirical_p_native_greater": p,
        "n_rewired": N_REWIRED,
    }])
    return null, summary


# =============================================================================
# 9. REPRESENTATIVE SEED AND MATRICES
# =============================================================================

def representative_seed(
    seed_summary: pd.DataFrame,
) -> str:
    full = seed_summary[
        seed_summary["mode"] == "Full MLTD"
    ][["seed_rna", "mean_bridge_score"]]
    base = seed_summary[
        seed_summary["mode"] == "Separate layers"
    ][["seed_rna", "mean_bridge_score"]]
    merged = full.merge(
        base,
        on="seed_rna",
        suffixes=("_full", "_base"),
    )
    merged["gain"] = (
        merged["mean_bridge_score_full"]
        - merged["mean_bridge_score_base"]
    )
    return str(
        merged.sort_values("gain", ascending=False)
        .iloc[0]["seed_rna"]
    )


def gain_matrix(
    pair_table: pd.DataFrame,
    rna_order: list[str],
    tf_order: list[str],
) -> pd.DataFrame:
    full = pair_table[
        pair_table["mode"] == "Full MLTD"
    ][["seed_rna", "tf", "bridge_score"]].rename(
        columns={"bridge_score": "full"}
    )
    base = pair_table[
        pair_table["mode"] == "Separate layers"
    ][["seed_rna", "tf", "bridge_score"]].rename(
        columns={"bridge_score": "base"}
    )
    m = full.merge(base, on=["seed_rna", "tf"])
    m["gain"] = m["full"] - m["base"]
    return (
        m.pivot(index="seed_rna", columns="tf", values="gain")
        .reindex(index=rna_order, columns=tf_order)
    )


# =============================================================================
# 10. FIGURE PANELS — INDEPENDENT SIDES + HYBRID EVIDENCE
# =============================================================================

def representative_pair_table(
    pair_table: pd.DataFrame,
    seed: str,
) -> pd.DataFrame:
    full = pair_table[
        (pair_table["mode"] == "Full MLTD")
        & (pair_table["seed_rna"] == seed)
    ][
        [
            "tf",
            "tf_key",
            "rna_tf_percentile",
            "tf_re_percentile",
            "bridge_score",
        ]
    ].copy()

    base = pair_table[
        (pair_table["mode"] == "Separate layers")
        & (pair_table["seed_rna"] == seed)
    ][
        [
            "tf_key",
            "rna_tf_percentile",
            "tf_re_percentile",
            "bridge_score",
        ]
    ].copy()

    merged = full.merge(
        base,
        on="tf_key",
        suffixes=("_full", "_base"),
        validate="one_to_one",
    )
    merged["bridge_gain"] = (
        merged["bridge_score_full"]
        - merged["bridge_score_base"]
    )

    # Independence QC: the RE anchor must be exactly the same.
    if not np.allclose(
        merged["tf_re_percentile_full"],
        merged["tf_re_percentile_base"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Representative-pair QC failed: TF→RE anchors differ between "
            "Full MLTD and Separate layers."
        )

    return merged.sort_values(
        ["bridge_score_full", "bridge_gain"],
        ascending=False,
    ).reset_index(drop=True)


def draw_panel_a(ax):
    panel_label(ax, "A")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Independent incoming and outgoing support",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    # Full row.
    y = 0.70
    ax.text(
        0.01, y, "Full MLTD",
        fontsize=6.5,
        fontweight="bold",
        color=FULL_COLOR,
        ha="left",
        va="center",
    )

    x_rna, x_tf, x_re = 0.31, 0.61, 0.89
    nodes = [
        (x_rna, "RNA seed", RNA_COLOR),
        (x_tf, "TF program", TF_COLOR),
        (x_re, "RE evidence", RE_COLOR),
    ]
    for x, label, color in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.095, y - 0.105),
                0.19,
                0.21,
                boxstyle=(
                    "round,pad=0.008,rounding_size=0.016"
                ),
                facecolor="white",
                edgecolor=color,
                linewidth=0.95,
            )
        )
        ax.text(
            x, y, label,
            fontsize=5.7,
            color=DARK,
            ha="center",
            va="center",
        )

    ax.add_patch(
        FancyArrowPatch(
            (x_rna + 0.10, y),
            (x_tf - 0.10, y),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=1.15,
            color=TF_COLOR,
        )
    )
    # Independent RE anchor points toward the TF, rather than sharing the
    # RNA-facing gate.
    ax.add_patch(
        FancyArrowPatch(
            (x_re - 0.10, y),
            (x_tf + 0.10, y),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=1.15,
            color=RE_COLOR,
        )
    )
    ax.text(
        0.46, y + 0.13,
        "RNA→TF only",
        fontsize=5.1,
        color=MID_GREY,
        ha="center",
    )
    ax.text(
        0.75, y + 0.13,
        "independent TF→RE evidence",
        fontsize=5.1,
        color=MID_GREY,
        ha="center",
    )

    # Separate row.
    y = 0.28
    ax.text(
        0.01, y, "Separate layers",
        fontsize=6.5,
        fontweight="bold",
        color=BASELINE_COLOR,
        ha="left",
        va="center",
    )
    for x, label, _ in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.095, y - 0.105),
                0.19,
                0.21,
                boxstyle=(
                    "round,pad=0.008,rounding_size=0.016"
                ),
                facecolor="white",
                edgecolor=LIGHT_GREY,
                linewidth=0.70,
            )
        )
        ax.text(
            x, y, label,
            fontsize=5.7,
            color=MID_GREY,
            ha="center",
            va="center",
        )
    ax.plot(
        [x_rna + 0.10, x_tf - 0.10],
        [y, y],
        color=BASELINE_COLOR,
        linewidth=0.70,
        linestyle=(0, (2, 2)),
    )
    ax.add_patch(
        FancyArrowPatch(
            (x_re - 0.10, y),
            (x_tf + 0.10, y),
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=0.85,
            color=RE_COLOR,
            alpha=0.65,
        )
    )
    ax.text(
        0.50,
        0.02,
        "The RE-facing anchor is identical in every mode; only RNA→TF "
        "support changes.",
        ha="center",
        va="bottom",
        fontsize=5.15,
        color=MID_GREY,
    )


def draw_panel_b_point_to_point(
    ax,
    pair_table: pd.DataFrame,
    seed: str,
):
    """
    Explicit point-to-point legend:
      open grey circle = Separate-layer RNA→TF support;
      filled teal circle = Full-MLTD RNA→TF support;
      purple square = independent, mode-invariant TF→RE support.
    """
    panel_label(ax, "B")
    ax.set_title(
        f"Point-to-point TF bridges: {seed}",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    m = representative_pair_table(pair_table, seed)
    y_positions = np.arange(len(m))[::-1].astype(float)

    for yy, (_, row) in zip(y_positions, m.iterrows()):
        x_base = float(row["rna_tf_percentile_base"])
        x_full = float(row["rna_tf_percentile_full"])
        x_re = float(row["tf_re_percentile_full"])

        # Baseline distance to the same RE anchor.
        ax.plot(
            [x_base, x_re],
            [yy + 0.105, yy + 0.105],
            color="#C4C7CA",
            linewidth=0.65,
            zorder=1,
        )
        # Full distance to the same RE anchor.
        full_line_color = (
            FULL_COLOR
            if row["bridge_score_full"] >= COMPLETE_THRESHOLD
            else "#8297AB"
        )
        ax.plot(
            [x_full, x_re],
            [yy - 0.105, yy - 0.105],
            color=full_line_color,
            linewidth=1.05,
            zorder=2,
        )

        # Separate RNA→TF.
        ax.scatter(
            [x_base],
            [yy + 0.105],
            marker="o",
            s=18,
            facecolor="white",
            edgecolor=BASELINE_COLOR,
            linewidth=0.75,
            zorder=3,
        )
        # Full RNA→TF.
        ax.scatter(
            [x_full],
            [yy - 0.105],
            marker="o",
            s=20,
            color=RNA_COLOR,
            edgecolor="white",
            linewidth=0.40,
            zorder=4,
        )
        # One independent RE anchor for the TF. Draw midway between rails.
        ax.scatter(
            [x_re],
            [yy],
            marker="s",
            s=20,
            color=RE_COLOR,
            edgecolor="white",
            linewidth=0.40,
            zorder=5,
        )

    ax.axvline(
        COMPLETE_THRESHOLD,
        color=LIGHT_GREY,
        linewidth=0.65,
        linestyle=(0, (2, 2)),
    )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.55, len(m) - 0.45)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(m["tf"].tolist(), fontsize=5.6)
    ax.set_xlabel("Within-layer support percentile")
    ax.grid(
        axis="x",
        color=GRID_GREY,
        linewidth=0.45,
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=BASELINE_COLOR,
            markeredgewidth=0.75,
            markersize=4.2,
            label="Separate RNA→TF",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=RNA_COLOR,
            markeredgecolor=RNA_COLOR,
            markersize=4.2,
            label="Full MLTD RNA→TF",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor=RE_COLOR,
            markeredgecolor=RE_COLOR,
            markersize=4.1,
            label="Independent TF→RE",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower right",
        borderpad=0.1,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    clean_axis(ax)


def draw_panel_c_displacement(
    ax,
    pair_table: pd.DataFrame,
    seed: str,
):
    """
    Because the RE side is independent and fixed, all displacement arrows are
    horizontal. This is intentional: MLTD changes the RNA-facing support
    relative to a fixed RE anchor.
    """
    panel_label(ax, "C")
    ax.set_title(
        "RNA-facing shift toward fixed RE anchors",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    m = representative_pair_table(pair_table, seed)

    for _, row in m.iterrows():
        y = float(row["tf_re_percentile_full"])
        ax.add_patch(
            FancyArrowPatch(
                (
                    float(row["rna_tf_percentile_base"]),
                    y,
                ),
                (
                    float(row["rna_tf_percentile_full"]),
                    y,
                ),
                arrowstyle="-|>",
                mutation_scale=5.7,
                linewidth=0.52,
                color="#BFC3C6",
                alpha=0.75,
                zorder=1,
            )
        )

    ax.scatter(
        m["rna_tf_percentile_base"],
        m["tf_re_percentile_full"],
        s=16,
        facecolor="white",
        edgecolor=BASELINE_COLOR,
        linewidth=0.70,
        label="Separate RNA→TF",
        zorder=2,
    )
    ax.scatter(
        m["rna_tf_percentile_full"],
        m["tf_re_percentile_full"],
        s=21,
        color=RNA_COLOR,
        edgecolor="white",
        linewidth=0.40,
        label="Full MLTD RNA→TF",
        zorder=3,
    )
    ax.scatter(
        m["tf_re_percentile_full"],
        m["tf_re_percentile_full"],
        marker="s",
        s=13,
        color=RE_COLOR,
        alpha=0.42,
        linewidth=0,
        label="TF→RE anchor",
        zorder=0,
    )

    label_rows = m.nlargest(
        min(N_LABEL_TF_REPRESENTATIVE, len(m)),
        "bridge_gain",
    )
    for _, row in label_rows.iterrows():
        ax.text(
            float(row["rna_tf_percentile_full"]) + 0.020,
            float(row["tf_re_percentile_full"]) + 0.013,
            str(row["tf"]),
            fontsize=5.2,
            color=DARK,
        )

    ax.axvline(
        COMPLETE_THRESHOLD,
        color=LIGHT_GREY,
        linestyle=(0, (2, 2)),
        linewidth=0.65,
    )
    ax.axhline(
        COMPLETE_THRESHOLD,
        color=LIGHT_GREY,
        linestyle=(0, (2, 2)),
        linewidth=0.65,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#E4E6E8",
        linewidth=0.55,
        zorder=0,
    )
    ax.set_xlim(-0.04, 1.07)
    ax.set_ylim(-0.04, 1.07)
    ax.set_xlabel("RNA→TF support percentile")
    ax.set_ylabel("Independent TF→RE percentile")
    ax.legend(
        frameon=False,
        loc="lower right",
        borderpad=0.1,
        handletextpad=0.30,
        labelspacing=0.22,
    )
    clean_axis(ax)


def draw_panel_d_heatmap(
    ax,
    matrix: pd.DataFrame,
):
    panel_label(ax, "D")
    ax.set_title(
        "Bridge gain across RNA–TF pairs",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    row_order = (
        matrix.mean(axis=1)
        .sort_values(ascending=False)
        .index
    )
    col_order = (
        matrix.mean(axis=0)
        .sort_values(ascending=False)
        .index
    )
    matrix = matrix.reindex(
        index=row_order,
        columns=col_order,
    )

    arr = matrix.to_numpy(float)
    vmax = max(
        float(np.nanmax(np.abs(arr))),
        1e-6,
    )
    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    im = ax.imshow(
        arr,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
        fontsize=5.25,
    )
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(
        matrix.index,
        fontsize=5.25,
    )
    ax.set_xlabel("TF program")
    ax.set_ylabel("RNA seed")

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.018,
    )
    cbar.ax.tick_params(
        labelsize=5.1,
        width=0.4,
        length=1.8,
    )
    cbar.set_label(
        "Bridge gain\nFull − separate",
        fontsize=5.4,
    )
    clean_axis(ax, keep=())


def draw_panel_e_seed_level(
    ax,
    seed_summary: pd.DataFrame,
    stats: pd.DataFrame,
):
    panel_label(ax, "E")
    ax.set_title(
        "Seed-level bridge gain",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    full = seed_summary[
        seed_summary["mode"] == "Full MLTD"
    ][["seed_rna", "mean_bridge_score"]]
    base = seed_summary[
        seed_summary["mode"] == "Separate layers"
    ][["seed_rna", "mean_bridge_score"]]

    merged = base.merge(
        full,
        on="seed_rna",
        suffixes=("_base", "_full"),
        validate="one_to_one",
    )

    for _, row in merged.iterrows():
        ax.plot(
            [0, 1],
            [
                row["mean_bridge_score_base"],
                row["mean_bridge_score_full"],
            ],
            color="#BEC2C5",
            linewidth=0.70,
            zorder=1,
        )

    ax.scatter(
        np.zeros(len(merged)),
        merged["mean_bridge_score_base"],
        s=18,
        color=BASELINE_COLOR,
        edgecolor="white",
        linewidth=0.40,
        zorder=2,
    )
    ax.scatter(
        np.ones(len(merged)),
        merged["mean_bridge_score_full"],
        s=21,
        color=FULL_COLOR,
        edgecolor="white",
        linewidth=0.40,
        zorder=2,
    )

    for xpos, values in [
        (
            0,
            merged["mean_bridge_score_base"].to_numpy(float),
        ),
        (
            1,
            merged["mean_bridge_score_full"].to_numpy(float),
        ),
    ]:
        mean, lo, hi = bootstrap_mean_ci(values)
        ax.errorbar(
            [xpos],
            [mean],
            yerr=[[mean - lo], [hi - mean]],
            fmt="D",
            markersize=3.5,
            color=DARK,
            ecolor=DARK,
            elinewidth=0.75,
            capsize=2.0,
            zorder=4,
        )

    stat = stats[
        (stats["control_mode"] == "Separate layers")
        & (stats["metric"] == "mean_bridge_score")
    ]
    delta = (
        float(stat["mean_difference"].iloc[0])
        if len(stat) else np.nan
    )
    p = (
        float(stat["exact_signflip_p"].iloc[0])
        if len(stat) else np.nan
    )
    q = (
        float(stat["fdr"].iloc[0])
        if len(stat) else np.nan
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Separate\nlayers", "Full\nMLTD"]
    )
    ax.set_xlim(-0.34, 1.34)
    ax.set_ylabel(
        "Mean bridge score per RNA seed"
    )
    ax.text(
        0.98,
        0.98,
        f"Δ={delta:.3f}\nexact P={p:.3g}; FDR={q:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.35,
        color=MID_GREY,
    )
    ax.grid(
        axis="y",
        color=GRID_GREY,
        linewidth=0.45,
    )
    clean_axis(ax)


def draw_panel_f_components(
    ax,
    stats: pd.DataFrame,
):
    panel_label(ax, "F")
    ax.set_title(
        "MLTD component contributions",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    sub = stats[
        stats["metric"] == "mean_bridge_score"
    ].copy()

    order = [
        "Separate layers",
        "Prior removed",
        "Latent removed",
        "Continuity removed",
        "Corr only",
    ]
    labels = {
        "Separate layers": "Separate layers",
        "Prior removed": "Prior removed",
        "Latent removed": "Latent removed",
        "Continuity removed": "Continuity removed",
        "Corr only": "Correlation only",
    }

    sub["control_mode"] = pd.Categorical(
        sub["control_mode"],
        categories=order,
        ordered=True,
    )
    sub = sub.sort_values(
        "control_mode"
    ).dropna(
        subset=["control_mode"]
    )
    y = np.arange(len(sub))[::-1]

    x = sub["mean_difference"].to_numpy(float)
    lo = sub["ci95_low"].to_numpy(float)
    hi = sub["ci95_high"].to_numpy(float)

    ax.axvline(
        0,
        color="#797D80",
        linewidth=0.65,
    )
    ax.hlines(
        y,
        lo,
        hi,
        color="#8E9397",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        x,
        y,
        s=25,
        color=[
            FULL_COLOR if value > 0
            else BASELINE_COLOR
            for value in x
        ],
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )

    x_span = max(
        float(np.nanmax(hi) - np.nanmin(lo)),
        0.05,
    )
    text_offset = 0.025 * x_span
    for yy, (_, row), upper in zip(
        y,
        sub.iterrows(),
        hi,
    ):
        ax.text(
            upper + text_offset,
            yy,
            f"q={row['fdr']:.2g}",
            fontsize=5.15,
            color=MID_GREY,
            va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            labels[str(value)]
            for value in sub["control_mode"]
        ]
    )
    ax.set_xlabel(
        "Paired gain in mean bridge score\n"
        "Full MLTD − control"
    )
    ax.grid(
        axis="x",
        color=GRID_GREY,
        linewidth=0.45,
    )
    clean_axis(ax)


def draw_panel_g_rewired(
    ax,
    null: pd.DataFrame,
    null_summary: pd.DataFrame,
):
    panel_label(ax, "G")
    ax.set_title(
        "Native TF matching versus rewired null",
        loc="left",
        pad=5,
        fontweight="bold",
    )

    values = null[
        "mean_seed_bridge_score"
    ].to_numpy(float)
    observed = float(
        null_summary[
            "observed_native_matching"
        ].iloc[0]
    )
    p = float(
        null_summary[
            "empirical_p_native_greater"
        ].iloc[0]
    )
    n_perm = int(
        null_summary["n_rewired"].iloc[0]
    )

    ax.hist(
        values,
        bins=34,
        color=NULL_COLOR,
        edgecolor="white",
        linewidth=0.30,
    )
    ax.axvline(
        observed,
        color=FULL_COLOR,
        linewidth=1.65,
    )

    min_empirical = 1.0 / (n_perm + 1.0)
    if p <= min_empirical + 1e-15:
        p_text = f"Pemp≤{min_empirical:.1e}"
    else:
        p_text = f"Pemp={p:.2g}"

    ymax = ax.get_ylim()[1]
    x_text = min(
        observed,
        ax.get_xlim()[1] - 0.01
    )
    ax.text(
        x_text,
        0.94 * ymax,
        f"native\n{p_text}\n{n_perm:,} rewires",
        fontsize=5.4,
        color=FULL_COLOR,
        ha="right",
        va="top",
    )
    ax.set_xlabel(
        "Mean seed bridge score after "
        "TF-label rewiring"
    )
    ax.set_ylabel("Frequency")
    clean_axis(ax)


# =============================================================================
# 11. MAIN FIGURE — THREE-ROW PUBLICATION LAYOUT
# =============================================================================

def make_figure(
    pair_table: pd.DataFrame,
    seed_summary: pd.DataFrame,
    stats: pd.DataFrame,
    null: pd.DataFrame,
    null_summary: pd.DataFrame,
    rna_order: list[str],
    tf_order: list[str],
):
    set_publication_style()

    seed = representative_seed(seed_summary)
    matrix = gain_matrix(
        pair_table,
        rna_order=rna_order,
        tf_order=tf_order,
    )

    fig = plt.figure(
        figsize=(FIG_W_IN, FIG_H_IN),
        dpi=DPI,
    )
    gs = fig.add_gridspec(
        3,
        6,
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.975,
        wspace=0.78,
        hspace=0.58,
        height_ratios=[0.90, 1.00, 0.92],
    )

    # Row 1: schematic + wide point-to-point panel.
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:6])

    # Row 2: compact displacement + wide global heatmap.
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:6])

    # Row 3: three statistical summaries.
    ax_e = fig.add_subplot(gs[2, 0:2])
    ax_f = fig.add_subplot(gs[2, 2:4])
    ax_g = fig.add_subplot(gs[2, 4:6])

    draw_panel_a(ax_a)
    draw_panel_b_point_to_point(
        ax_b,
        pair_table,
        seed,
    )
    draw_panel_c_displacement(
        ax_c,
        pair_table,
        seed,
    )
    draw_panel_d_heatmap(
        ax_d,
        matrix,
    )
    draw_panel_e_seed_level(
        ax_e,
        seed_summary,
        stats,
    )
    draw_panel_f_components(
        ax_f,
        stats,
    )
    draw_panel_g_rewired(
        ax_g,
        null,
        null_summary,
    )

    save_figure(fig)


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_inputs()

    rna_table = load_ranked_list(RNA_TOP_CSV, "rna")
    tf_table = load_ranked_list(TF_TOP_CSV, "tf")

    rna_keys = set(rna_table["rna_key"])
    tf_keys = set(tf_table["tf_key"])

    decomp = load_decomposition(rna_keys, tf_keys)
    peaks, arcs = load_re_tables(tf_keys)
    re_support = aggregate_re_support(peaks, arcs, tf_table)
    gates, gate_sources = build_mode_gates(tf_table)

    pair_table = build_seed_tf_bridge_table(
        rna_table=rna_table,
        tf_table=tf_table,
        decomp=decomp,
        re_support=re_support,
        gates=gates,
        gate_sources=gate_sources,
    )
    seed_summary = summarize_by_seed(pair_table)
    mode_effects = paired_mode_effects(seed_summary)
    null, null_summary = run_tf_label_rewired_null(pair_table)

    pair_table.to_csv(OUT_PAIR_TABLE, index=False)
    seed_summary.to_csv(OUT_SEED_SUMMARY, index=False)
    mode_effects.to_csv(OUT_MODE_EFFECTS, index=False)
    null.to_csv(OUT_NULL, index=False)

    stats = pd.concat(
        [
            mode_effects,
            null_summary.assign(
                comparison="Native TF matching vs rewired",
                control_mode="TF-label rewired",
                fdr=np.nan,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    stats.to_csv(OUT_STATS, index=False)

    rna_order = rna_table["rna"].tolist()
    tf_order = tf_table["tf"].tolist()

    make_figure(
        pair_table=pair_table,
        seed_summary=seed_summary,
        stats=mode_effects,
        null=null,
        null_summary=null_summary,
        rna_order=rna_order,
        tf_order=tf_order,
    )

    main_stat = mode_effects[
        (mode_effects["control_mode"] == "Separate layers")
        & (mode_effects["metric"] == "mean_bridge_score")
    ].iloc[0]
    null_p = float(
        null_summary["empirical_p_native_greater"].iloc[0]
    )
    at_least_one_ablation = bool(
        (
            mode_effects[
                (mode_effects["metric"] == "mean_bridge_score")
                & (
                    mode_effects["control_mode"].isin([
                        "Prior removed",
                        "Latent removed",
                        "Continuity removed",
                    ])
                )
            ]["mean_difference"] > 0
        ).any()
    )

    promotion_supported = bool(
        main_stat["mean_difference"] > 0
        and main_stat["fdr"] < 0.05
        and null_p < 0.05
        and at_least_one_ablation
    )

    manifest = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "analysis": (
            "Independent-sided, hybrid point-to-point and seed-conditioned "
            "TF-identity-matched RNA–TF–RE bridging"
        ),
        "n_rna_seed": int(len(rna_table)),
        "n_tf": int(len(tf_table)),
        "n_seed_tf_pairs_per_mode": int(
            len(rna_table) * len(tf_table)
        ),
        "bridge_score": (
            "min(seed-within-mode RNA→TF percentile, "
            "independently derived mode-invariant TF→RE percentile)"
        ),
        "independence_constraint": {
            "rna_facing_score": (
                "seed-specific RNA→TF association + latent edge + prior + "
                "optional upstream-only continuity/program gate"
            ),
            "re_facing_score": (
                "TF-target + gene-peak + peak-arc evidence only"
            ),
            "shared_gate_across_sides": False,
            "tf_re_score_mode_invariant": True,
            "hard_qc_enforced": True
        },
        "complete_threshold": COMPLETE_THRESHOLD,
        "primary_comparison": "Full MLTD vs Separate layers",
        "primary_mean_gain": float(main_stat["mean_difference"]),
        "primary_exact_p": float(main_stat["exact_signflip_p"]),
        "primary_fdr": float(main_stat["fdr"]),
        "rewired_empirical_p": null_p,
        "promotion_supported_by_prespecified_criteria": (
            promotion_supported
        ),
        "prespecified_criteria": {
            "full_vs_separate_gain_positive": True,
            "full_vs_separate_fdr_below_0.05": True,
            "native_vs_rewired_p_below_0.05": True,
            "at_least_one_component_ablation_reduces_bridge": True,
        },
        "gate_sources": gate_sources,
        "figure_logic": [
            "independent RNA-facing and RE-facing mechanism schematic",
            "TF-level point-to-point bridges to a fixed independent RE anchor",
            "horizontal RNA-facing displacement toward fixed RE anchors",
            "global RNA-seed by TF bridge-gain heatmap",
            "seed-level paired bridge gain",
            "component effect sizes",
            "TF-label rewired null"
        ],
        "interpretation_boundary": (
            "Internal seed-conditioned structural transmission analysis; "
            "not experimental perturbation accuracy."
        ),
        "input_files": {
            "rna_top": str(RNA_TOP_CSV),
            "tf_top": str(TF_TOP_CSV),
            "rna_tf_decomposition": str(RNA_TF_DECOMP_CSV),
            "tf_re_peaks": str(TF_RE_PEAK_CSV),
            "tf_re_arcs": str(TF_RE_ARC_CSV),
        },
    }
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("[SAVE]", OUT_PAIR_TABLE)
    log("[SAVE]", OUT_SEED_SUMMARY)
    log("[SAVE]", OUT_MODE_EFFECTS)
    log("[SAVE]", OUT_NULL)
    log("[SAVE]", OUT_STATS)
    log("[SAVE]", OUT_MANIFEST)
    log("[DONE]", OUT_DIR)


if __name__ == "__main__":
    main()
