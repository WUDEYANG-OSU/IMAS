# -*- coding: utf-8 -*-
"""
25_plot_CRC_MLTD_sparse_pathway_directional_CCI_NSC_v5_4.py

Sparse-pathway, cluster-specific MLTD communication — V5.4
=======================================================

Why V3 failed
-------------
V3 defined MLTD-on augmentation through several simultaneous hard gates:

    sender molecular bridge >= threshold
    receiver molecular bridge >= threshold
    sender cluster bridge >= threshold
    receiver cluster bridge >= threshold
    augmentation >= threshold

If either side narrowly failed, augmentation was set to exactly zero. In real
data this can make:

    MLTD-on == MLTD-off
    Top pathways == 0

V4 removes that brittle definition.

Core scientific contrast
------------------------
The same complete LR candidate universe and the same observed CCI context are
used for both modes.

MLTD-off bridge:
    correlation-only RNA→TF
    + independent TF→RE support
    + cluster-specific expression / TF activity

MLTD-on bridge:
    full MLTD RNA→TF
      = correlation + latent/model edge + prior
    + the same independent TF→RE support
    + the same cluster-specific expression / TF activity

For every gene–TF edge:

    full_support = max(correlation_support, calibrated_full_MLTD_support)

Therefore, at the molecular level:

    bridge_on >= bridge_off

For each LR program:

    observed_context
      = LR confidence
        × database support
        × cluster-specific LR expression

    pair_bridge_mode
      = noisy-OR(sender bridge, receiver bridge)

    score_mode
      = noisy-OR(native CCI core,
                 observed_context × pair_bridge_mode)

    MLTD_gain
      = score_on - score_off >= 0

No absolute bridge threshold is used to create the gain. Statistical
gene-identity permutations determine which positive gains represent potential
MLTD-specific communication programs.

Program selection
-----------------
MLTD-off active programs:
    original high-confidence + TF-kept CCI programs.

MLTD-on active programs:
    all MLTD-off active programs
    + potential gained programs satisfying:

        gain >= MIN_PROGRAM_GAIN
        empirical P_gain <= POTENTIAL_GAIN_P_THRESHOLD

The empirical P threshold is intentionally used for "potential communication".
BH-adjusted Q values are still reported and used as the bubble color.

The script never silently emits two identical maps:
    if no positive gain exists, it raises with coverage diagnostics;
    if positive gains exist but none pass the potential-gain rule, it raises
    with gain/P-value quantiles instead of creating a misleading figure.

Direction-preserving triangular map
-----------------------------------
The same clusters are on both axes; only the upper triangle is retained.

Within an unordered cluster-pair cell:
    upper-left bubble  = lower-index cluster → higher-index cluster
    lower-right bubble = higher-index cluster → lower-index cluster
    center bubble      = self-communication

Bubble size:
    state-specific directed communication weight

Bubble color:
    MLTD-off panel: -log10(BH-adjusted native-state P)
    MLTD-on panel:  -log10(BH-adjusted MLTD-gain P)

Top pathways
------------
Top pathway units are ranked by total positive MLTD gain.

Each pathway label appears only once globally. When the input lacks a useful
pathway annotation, the script creates an LR-family label, for example:

    LAMB–CD44 signaling
    CXCL–CXCR signaling

Thus the pathway panel cannot collapse into a single repeated "Other".
"""

from __future__ import annotations

import json
import math
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


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
CCI_DIR = RUN_BASE / f"{SAMPLE_ID}_cci_newLeiden"

# Optional manual paths.
ALL_CCI_MANUAL: Path | None = None
NATIVE_CCI_MANUAL: Path | None = None
GENE_TF_MANUAL: Path | None = None
TF_RE_MANUAL: Path | None = None
TF_RE_PEAKS_MANUAL: Path | None = None
TF_RE_ARCS_MANUAL: Path | None = None
RECEPTOR_TF_MANUAL: Path | None = None
HYBRID_TF_MANUAL: Path | None = None

ALL_CCI_CANDIDATES = [
    CCI_DIR / "interactions_cluster_level.csv",
]
NATIVE_CCI_CANDIDATES = [
    CCI_DIR / "interactions_cluster_level_highconf_TFkept.csv",
    CCI_DIR / "interactions_cluster_level_highconf.csv",
]
GENE_TF_CANDIDATES = [
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_network_aware_top_tf"
    / f"{SAMPLE_ID}_gene_topk_tf_edges.csv",
]
TF_RE_COMPONENT_CANDIDATES = [
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_tf_centered_bridge_promotion_v1"
    / f"{SAMPLE_ID}_TF_RE_intrinsic_components.csv",
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_bridge_promotion_independent_sides_NSC_v1_3"
    / f"{SAMPLE_ID}_TF_RE_intrinsic_components.csv",
]
TF_RE_PEAK_CANDIDATES = [
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_rna_tf_re_three_layer_v1"
    / f"{SAMPLE_ID}_TF_linked_RE_peaks_v1.csv",
]
TF_RE_ARC_CANDIDATES = [
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_rna_tf_re_three_layer_v1"
    / f"{SAMPLE_ID}_TF_linked_RE_peak_arcs_v1.csv",
]
RECEPTOR_TF_CANDIDATES = [
    CCI_DIR / "receptor_tf_signature.csv",
]
HYBRID_TF_CANDIDATES = [
    CCI_DIR
    / "tfaware_liana_dotplot_hybridTF_named"
    / "dotplot_hybridTF_table.csv",
    CCI_DIR
    / "tfaware_liana_dotplot_hybridTF"
    / "dotplot_hybridTF_table.csv",
]

OUT_DIR = (
    RUN_BASE
    / f"single_sample_{SAMPLE_ID}_MLTD_sparse_pathway_directional_CCI_NSC_v5_4"
)

OUT_GENE_TF = OUT_DIR / f"{SAMPLE_ID}_gene_TF_on_off_support.csv"
OUT_GENE_TF_QC = OUT_DIR / f"{SAMPLE_ID}_gene_TF_component_gain_QC.csv"
OUT_TF_RE = OUT_DIR / f"{SAMPLE_ID}_TF_RE_support.csv"
OUT_RECEPTOR_TF = OUT_DIR / f"{SAMPLE_ID}_deduplicated_receptor_TF_support.csv"
OUT_RECEPTOR_TF_QC = OUT_DIR / f"{SAMPLE_ID}_receptor_TF_duplicate_QC.csv"
OUT_CLUSTER_TF = OUT_DIR / f"{SAMPLE_ID}_cluster_TF_activity_with_backfill.csv"
OUT_SENDER = OUT_DIR / f"{SAMPLE_ID}_sender_cluster_gene_on_off_bridges.csv"
OUT_RECEIVER = OUT_DIR / f"{SAMPLE_ID}_receiver_cluster_gene_on_off_bridges.csv"
OUT_PROGRAMS = OUT_DIR / f"{SAMPLE_ID}_same_universe_program_on_off_scores.csv"
OUT_GAINED = OUT_DIR / f"{SAMPLE_ID}_potential_MLTD_gained_programs.csv"
OUT_PAIR = OUT_DIR / f"{SAMPLE_ID}_directed_cluster_pair_on_off_summary.csv"
OUT_UNIT = OUT_DIR / f"{SAMPLE_ID}_directed_pair_pathway_on_off_summary.csv"
OUT_KEGG_DISPLAY = OUT_DIR / f"{SAMPLE_ID}_KEGG_style_display_pathway_summary.csv"
OUT_TOP10 = OUT_DIR / f"{SAMPLE_ID}_top10_unique_MLTD_gain_pathways.csv"
OUT_NULL = OUT_DIR / f"{SAMPLE_ID}_MLTD_gain_permutation_null.csv"
OUT_QC = OUT_DIR / f"{SAMPLE_ID}_MLTD_gain_CCI_QC.csv"
OUT_DIAGNOSTIC = OUT_DIR / f"{SAMPLE_ID}_MLTD_gain_quantile_diagnostics.csv"
OUT_MANIFEST = OUT_DIR / f"{SAMPLE_ID}_MLTD_gain_CCI_manifest.json"

OUT_OFF_STEM = OUT_DIR / f"{SAMPLE_ID}_MLTD_off_directional_triangular_CCI"
OUT_ON_STEM = OUT_DIR / f"{SAMPLE_ID}_MLTD_on_directional_triangular_CCI"
OUT_COMBINED_STEM = (
    OUT_DIR / f"{SAMPLE_ID}_MLTD_off_on_directional_triangular_CCI_comparison"
)

# RNA→TF scoring.
W_CORR = 0.40
W_LATENT = 0.35
W_PRIOR = 0.25

# Cross-gene calibration.
W_GLOBAL_RANK = 0.60
W_TF_STRATIFIED_RANK = 0.40

# MLTD-specific RNA→TF evidence channels.
#
# V4.1 compared a single full-score ranking with the correlation ranking.
# In the real CRC table the full ranking never exceeded the correlation
# ranking, so every gain became zero. V4.2 treats representation and prior as
# additional evidence channels, which is the actual distinction between
# correlation-only and full MLTD.
FULL_RESIDUAL_RELIABILITY = 0.30
LATENT_CHANNEL_RELIABILITY = 0.45
PRIOR_ASSISTED_RELIABILITY = 0.30

# Prior alone cannot create a bridge. It strengthens a model/representation
# edge through a prior-assisted joint channel.
PRIOR_ASSIST_BASE = 0.15

# Sparse TF activity backfill.
#
# The hybrid CCI table contains selected TFs, not a dense cluster×TF matrix.
# Missing cluster×TF entries must not be interpreted as biological zero.
TF_ACTIVITY_GLOBAL_BACKFILL = 0.35
TF_ACTIVITY_MIN_BACKFILL = 0.05

# Receptor→TF identity fallback.
#
# Direct receptor_tf_signature evidence is preferred. If it is missing for a
# receptor–TF edge, gene→TF support can be used only with this penalty.
RECEPTOR_TF_FALLBACK_PENALTY = 0.30

# Same candidate universe.
MIN_CANDIDATE_SOURCES = 1
MIN_CANDIDATE_EXPR = 0.0

# Observed/native score.
NATIVE_SCORE_FLOOR = 0.08
W_CONTEXT_CONFIDENCE = 0.30
W_CONTEXT_SOURCES = 0.25
W_CONTEXT_LR_EXPRESSION = 0.25
W_CONTEXT_LIGAND_EXPRESSION = 0.10
W_CONTEXT_RECEPTOR_EXPRESSION = 0.10

# Score composition.
NATIVE_CORE_WEIGHT = 0.55
BRIDGE_CORE_WEIGHT = 0.45

# Sparse-signal hierarchical evidence rules.
#
# The real CRC run showed that summing gain over all ~936k candidate programs
# is the wrong statistic: many irrelevant candidates dilute a small subset of
# identity-specific programs, and the permutation-null mean can exceed the
# observed total for every cluster pair.
#
# V5 therefore tests sparse enrichment in two stages:
#
#   program empirical P
#       -> Simes combination within directed pair × pathway
#       -> Simes combination across pathways within a directed pair
#
# A gained program can enter MLTD-on only when its pathway unit is supported
# and the program itself has positive MLTD gain plus a sufficiently small
# empirical P value.
POSITIVE_GAIN_EPS = 1e-12

PROGRAM_P_THRESHOLD_POTENTIAL = 0.01
PROGRAM_P_THRESHOLD_CONFIRMED = 0.05
MAX_GAINED_PROGRAMS_PER_UNIT = 5

UNIT_SIME_P_THRESHOLD = 0.05
UNIT_CONFIRMED_Q_THRESHOLD = 0.10

OFF_PAIR_P_THRESHOLD = 0.05
OFF_PAIR_Q_THRESHOLD = 0.10

REQUIRE_NEW_DIRECTED_PAIR = True

# Permutations.
N_PERMUTATIONS = 1000
PERMUTATION_SEED = 42

# Pathways.
TOP_N_PATHWAYS = 10

# Matrix.
INCLUDE_SELF_INTERACTIONS = True
CLUSTER_ORDER_MANUAL: list[str] | None = None
# Display.
DISPLAY_NATIVE_ONLY = True
# off map: original native programs only.
# on map: native programs + potential gained programs.

# Bubble areas are expressed in points^2.
#
# V5.2 displays one bubble per unordered cluster pair. Opposite directions
# are summed in the matrix and remain available in the Top pathway table.
BUBBLE_SIZE_MIN = 22.0
BUBBLE_SIZE_MAX = 270.0
BUBBLE_WEIGHT_CAP_QUANTILE = 0.95
BUBBLE_WEIGHT_LEGEND_QUANTILES = (0.25, 0.50, 0.85)
BUBBLE_LOG_POWER = 0.90

# Raw significance-effect score:
#
#   raw score = -log10(Simes P across active directions)
#               + COLOR_EFFECT_BONUS × robust normalized log1p(weight)
#
# MLTD-off and MLTD-on have different score distributions. V5.4 retains the
# raw score for numerical comparison but maps colors independently within
# each panel, so both panels display their internal blue–white–orange pattern.
COLOR_EFFECT_BONUS = 1.35
COLOR_SCORE_CAP = 6.0
COLOR_EFFECT_LOWER_QUANTILE = 0.05
COLOR_EFFECT_UPPER_QUANTILE = 0.95

PANEL_COLOR_LOWER_QUANTILE = 0.05
PANEL_COLOR_UPPER_QUANTILE = 0.95
PANEL_COLOR_MIN = 0.0
PANEL_COLOR_MAX = 1.0

STANDALONE_FIGSIZE = (10.4, 7.5)
COMBINED_FIGSIZE = (15.8, 7.5)
DPI = 450

GRID_COLOR = "#D8DADD"
EDGE_COLOR = "#BFC3C7"
TEXT_COLOR = "#202124"
MUTED_TEXT = "#70757A"
GAIN_COLOR = "#B64540"
OFF_COLOR = "#7B858C"


# =============================================================================
# 1. HELPERS
# =============================================================================

def log(*args: Any) -> None:
    print(*args, flush=True)


def canonical(value: Any) -> str:
    return str(value).strip().upper()


def normalize_cluster(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text if text.upper().startswith("C") else f"C{text}"


def cluster_sort_key(value: str) -> tuple[int, Any]:
    text = normalize_cluster(value)
    match = re.fullmatch(r"C(\d+)", text, flags=re.I)
    return (0, int(match.group(1))) if match else (1, text)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def sigmoid(values: np.ndarray | pd.Series | float) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -12.0, 12.0)))


def minmax01(values: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    output = np.zeros_like(x)
    finite = np.isfinite(x)
    if not finite.any():
        return output

    low = float(np.nanmin(x[finite]))
    high = float(np.nanmax(x[finite]))

    if high - low < 1e-12:
        output[finite] = 0.5 if high > 0 else 0.0
    else:
        output[finite] = (x[finite] - low) / (high - low)

    return np.clip(output, 0.0, 1.0)


def rank_percentile(values: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n = len(x)

    if n == 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([1.0], dtype=float)

    finite = np.isfinite(x)
    fill_value = (
        float(np.nanmin(x[finite])) - 1.0
        if finite.any()
        else -1.0
    )
    z = x.copy()
    z[~finite] = fill_value

    order = np.argsort(z, kind="mergesort")
    sorted_values = z[order]
    ranks = np.empty(n, dtype=float)

    start = 0
    while start < n:
        end = start + 1
        while (
            end < n
            and np.isclose(
                sorted_values[end],
                sorted_values[start],
                atol=1e-12,
                rtol=0.0,
            )
        ):
            end += 1

        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end

    return ranks / float(n - 1)


def cross_gene_tf_calibration(
    table: pd.DataFrame,
    value_column: str,
    *,
    prefix: str,
) -> np.ndarray:
    """
    Calibrate one gene–TF evidence channel across genes.

    The calibration combines:
        1. global percentile across all gene–TF edges;
        2. TF-stratified percentile across genes for the same TF.

    It deliberately avoids within-gene ranking.
    """
    global_column = f"{prefix}_global_pct"
    tf_column = f"{prefix}_tf_pct"

    table[global_column] = rank_percentile(
        table[value_column].to_numpy(float)
    )
    table[tf_column] = (
        table.groupby("tf_key")[value_column]
        .transform(
            lambda values: rank_percentile(
                values.to_numpy(float)
            )
        )
    )

    return (
        np.power(
            np.clip(
                table[global_column].to_numpy(float),
                1e-8,
                1.0,
            ),
            W_GLOBAL_RANK,
        )
        * np.power(
            np.clip(
                table[tf_column].to_numpy(float),
                1e-8,
                1.0,
            ),
            W_TF_STRATIFIED_RANK,
        )
    )


def noisy_or_many(
    *channels: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Combine independent evidence channels without requiring one channel to
    outrank another. Each channel must already be in [0, 1].
    """
    if not channels:
        return np.asarray([], dtype=float)

    output = np.zeros(
        len(np.asarray(channels[0])),
        dtype=float,
    )
    for channel in channels:
        output = noisy_or(
            output,
            np.clip(
                np.asarray(channel, dtype=float),
                0.0,
                1.0,
            ),
        )
    return output


def bh_fdr(values: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    output = np.full_like(p, np.nan)
    finite = np.isfinite(p)

    if not finite.any():
        return output

    p_finite = np.clip(p[finite], 0.0, 1.0)
    n = len(p_finite)
    order = np.argsort(p_finite)
    ranked = p_finite[order]

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]

    restored = np.empty(n, dtype=float)
    restored[order] = np.clip(q, 0.0, 1.0)
    output[finite] = restored
    return output


def first_existing(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = True,
    role: str = "column",
) -> str | None:
    lookup = {
        re.sub(r"[^a-z0-9]+", "", str(column).lower()): str(column)
        for column in df.columns
    }

    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "", str(alias).lower())
        if key in lookup:
            return lookup[key]

    if required:
        raise KeyError(
            f"Unable to identify {role}.\n"
            f"Actual columns: {df.columns.tolist()}\n"
            f"Accepted aliases: {list(aliases)}"
        )

    return None


def resolve_path(
    manual: Path | None,
    candidates: Iterable[Path],
    *,
    root: Path,
    filename: str,
) -> Path:
    if manual is not None:
        path = Path(manual)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()

    for path in candidates:
        if path.exists():
            return path.resolve()

    hits = sorted(
        root.rglob(filename),
        key=lambda path: (len(str(path)), str(path)),
    )
    if hits:
        return hits[0].resolve()

    raise FileNotFoundError(
        f"Unable to find {filename}.\n"
        + "\n".join(str(path) for path in candidates)
    )


def optional_path(
    manual: Path | None,
    candidates: Iterable[Path],
    *,
    root: Path,
    filename: str,
) -> Path | None:
    try:
        return resolve_path(
            manual,
            candidates,
            root=root,
            filename=filename,
        )
    except FileNotFoundError:
        return None


def weighted_geometric(
    components: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.clip(np.asarray(components, dtype=float), 1e-8, 1.0)
    weight = np.asarray(weights, dtype=float)
    weight = weight / weight.sum()
    return np.exp(np.sum(np.log(matrix) * weight[None, :], axis=1))


def noisy_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
    right = np.clip(np.asarray(b, dtype=float), 0.0, 1.0)
    return 1.0 - (1.0 - left) * (1.0 - right)


def gene_family(gene: str) -> str:
    value = canonical(gene)

    special = [
        "CXCL", "CCL", "IL", "IFN", "TNF", "TGFB", "BMP",
        "VEGF", "FGF", "EGF", "PDGF", "WNT", "JAG", "DLL",
        "NOTCH", "SEMA", "EFNA", "EFNB", "EPHA", "EPHB",
        "LAMA", "LAMB", "LAMC", "COL", "ITGA", "ITGB",
    ]
    for prefix in special:
        if value.startswith(prefix):
            return prefix

    match = re.match(r"([A-Z]+)", value)
    return match.group(1) if match else value


def infer_broad_pathway(ligand: str, receptor: str) -> str:
    pair = f"{canonical(ligand)}|{canonical(receptor)}"

    rules = [
        (
            "ECM–adhesion",
            r"(COL\d|LAMA|LAMB|LAMC|FN1|SPP1|THBS|VCAN|HSPG|"
            r"ITGA|ITGB|CD44|DDR\d|DAG1|ICAM|VCAM|SELE|SELL|SELP)",
        ),
        (
            "Growth factor–RTK",
            r"(EGF|FGF|VEGF|PDGF|HGF|IGF|NRG|GAS6|KITLG|"
            r"EGFR|FGFR|ERBB|MET|KDR|FLT\d|PDGFR|IGF\dR|KIT|AXL|MERTK)",
        ),
        (
            "Cytokine–chemokine",
            r"(IL\d|CXCL|CCL|IFN|TNF|CSF|OSM|LIF|"
            r"IL\dR|CXCR|CCR|IFNAR|IFNGR|TNFR|CSFR)",
        ),
        ("TGFβ–BMP", r"(TGFB|TGFBR|BMP\d|BMPR|ACVR|INHBA|INHBB)"),
        ("Notch", r"(JAG\d|DLL\d|NOTCH\d)"),
        ("WNT", r"(WNT\d|FZD\d|LRP5|LRP6|ROR\d)"),
        ("Semaphorin–Eph", r"(SEMA|PLXN|NRP\d|EFNA|EFNB|EPHA|EPHB)"),
    ]

    for pathway, pattern in rules:
        if re.search(pattern, pair, flags=re.I):
            return pathway

    return "Other"


def pathway_label(
    raw_pathway: Any,
    ligand: str,
    receptor: str,
) -> str:
    raw = str(raw_pathway).strip()

    unusable = {
        "",
        "NAN",
        "NONE",
        "OTHER",
        "UNKNOWN",
        "NA",
    }

    if canonical(raw) not in unusable:
        return raw

    broad = infer_broad_pathway(ligand, receptor)
    if broad != "Other":
        # Keep distinct LR-family signals within a broad pathway.
        return (
            f"{gene_family(ligand)}–{gene_family(receptor)} "
            f"{broad}"
        )

    return (
        f"{gene_family(ligand)}–{gene_family(receptor)} signaling"
    )


def infer_display_kegg_pathway(
    ligand: str,
    receptor: str,
    analysis_pathway: Any = "",
) -> str:
    """
    Post-hoc KEGG-style display mapping.

    This function MUST NOT be used to define `programs["pathway"]` or
    `pathway_unit`. The statistical analysis remains at the original V5.2
    fine-grained LR/pathway-unit level.

    The labels below are broad KEGG-style categories for figure display.
    They are rule-based annotations rather than a formal KEGG enrichment test.
    """
    ligand_key = canonical(ligand)
    receptor_key = canonical(receptor)
    analysis_key = canonical(analysis_pathway)

    joined = (
        ligand_key
        + "|"
        + receptor_key
        + "|"
        + analysis_key
    )

    # More specific receptor/downstream signatures are evaluated first so
    # KITLG–RPS6KB1, KITLG–EPOR, etc. do not all collapse into PI3K–Akt.
    rules = [
        (
            "Protein processing in endoplasmic reticulum",
            r"(HSPA5|HSP90B1|CANX|CALR|PDIA|DNAJ|SEC61|CDNF)",
        ),
        (
            "Complement and coagulation cascades",
            r"(\bF2\b|\bF3\b|\bF5\b|\bF7\b|\bF8\b|\bF9\b|"
            r"\bF10\b|\bF11\b|\bF12\b|FGA|FGB|FGG|PLG|"
            r"SERPINC1|PROC|PROS1|CTSG)",
        ),
        (
            "mTOR signaling pathway",
            r"(MTOR|RPTOR|RICTOR|RPS6KB1|RPS6KB2|EIF4EBP|AKT1S1)",
        ),
        (
            "JAK–STAT signaling pathway",
            r"(EPOR|JAK\d|STAT\d|IFNAR|IFNGR|IL\d+R|CSF\dR|"
            r"LIFR|OSMR|CNTFR)",
        ),
        (
            "ErbB signaling pathway",
            r"(ERBB|EGFR|LRIG|NRG\d|EREG|AREG|HBEGF|BTC|EPGN)",
        ),
        (
            "Axon guidance",
            r"(SEMA|PLXN|NRP\d|EPHA|EPHB|EFNA|EFNB|ROBO|SLIT)",
        ),
        (
            "ECM–receptor interaction",
            r"(COL\d|LAMA|LAMB|LAMC|FN1|HSPG|HSPG2|AGRN|"
            r"VCAN|SDC\d|CD44|DAG1|DDR1|DDR2)",
        ),
        (
            "Cell adhesion molecules",
            r"(ALCAM|NCAM|ICAM|VCAM|PECAM|NECTIN|JAM|CDH\d|"
            r"CLDN|LRRN|PTPR[ABCDMOPRSZ])",
        ),
        (
            "Chemokine signaling pathway",
            r"(CXCL|CXCR|CCL\d|CCR\d|XCL\d|XCR1)",
        ),
        (
            "Cytokine–cytokine receptor interaction",
            r"(IL\d|IFN|TNF|TNFR|CSF|OSM|LIF|CNTF|FLT3LG)",
        ),
        (
            "TGF-beta signaling pathway",
            r"(TGFB|TGFBR|BMP\d|BMPR|ACVR|INHBA|INHBB|SMAD)",
        ),
        (
            "Wnt signaling pathway",
            r"(WNT\d|FZD\d|LRP5|LRP6|ROR1|ROR2|RYK)",
        ),
        (
            "Notch signaling pathway",
            r"(NOTCH\d|JAG1|JAG2|DLL1|DLL3|DLL4)",
        ),
        (
            "Hedgehog signaling pathway",
            r"(SHH|IHH|DHH|PTCH1|PTCH2|SMO|HHIP)",
        ),
        (
            "VEGF signaling pathway",
            r"(VEGFA|VEGFB|VEGFC|VEGFD|KDR|FLT1|FLT4)",
        ),
        (
            "PI3K–Akt signaling pathway",
            r"(KITLG|KIT|IGF|IGF1R|PDGF|PDGFR|FGF|FGFR|"
            r"HGF|MET|GAS6|AXL|MERTK|TEK|ANGPT)",
        ),
        (
            "MAPK signaling pathway",
            r"(MAPK|ERK|RAF|MEK|NTRK|NGF|BDNF)",
        ),
        (
            "Focal adhesion",
            r"(ITGA|ITGB|PXN|PTK2|TLN|VCL|THBS|SPP1)",
        ),
        (
            "Apoptosis",
            r"(FAS|FASLG|TNFSF10|TNFRSF10|TRAIL|BCL2|BAX|CASP)",
        ),
        (
            "Calcium signaling pathway",
            r"(P2RY|P2RX|ADRA|ADRB|CHRM|CHRNA|GRM|HTR)",
        ),
        (
            "Drug metabolism–cytochrome P450",
            r"(CYP\d|UGT\d|GST[AMPT]|ALDH\d)",
        ),
    ]

    for pathway_name, pattern in rules:
        if re.search(
            pattern,
            joined,
            flags=re.I,
        ):
            return pathway_name

    # Fallback from the original fine-grained analysis label.
    analysis_rules = [
        ("ErbB signaling pathway", r"ERBB"),
        ("ECM–receptor interaction", r"ECM|MATRIX"),
        ("Focal adhesion", r"ADHESION"),
        ("Axon guidance", r"SEMAPHORIN|EPH"),
        ("Chemokine signaling pathway", r"CHEMOKINE"),
        (
            "Cytokine–cytokine receptor interaction",
            r"CYTOKINE",
        ),
        ("TGF-beta signaling pathway", r"TGF|BMP"),
        ("Wnt signaling pathway", r"WNT"),
        ("Notch signaling pathway", r"NOTCH"),
        ("PI3K–Akt signaling pathway", r"GROWTH FACTOR|RTK"),
    ]

    for pathway_name, pattern in analysis_rules:
        if re.search(
            pattern,
            analysis_key,
            flags=re.I,
        ):
            return pathway_name

    return "Other signaling pathways"


def build_kegg_display_pathway_summary(
    unit_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate already-selected fine-grained pathway units for display only.

    Input:
        the V5.2 fine-grained directed-pair × pathway summary.

    Output:
        one row per broad KEGG-style display pathway.

    This function cannot change:
        off_active/on_active,
        pair_summary,
        triangular-map shape,
        program selection,
        or Simes testing units.
    """
    candidates = unit_summary[
        unit_summary[
            "potential_gain_unit"
        ].fillna(False)
        & (
            pd.to_numeric(
                unit_summary[
                    "n_selected_gained_programs_actual"
                ],
                errors="coerce",
            ).fillna(0.0)
            > 0
        )
    ].copy()

    if len(candidates) == 0:
        raise RuntimeError(
            "No selected fine-grained pathway unit is available for the "
            "post-hoc KEGG-style display summary."
        )

    candidates["display_pathway"] = [
        infer_display_kegg_pathway(
            ligand=ligand,
            receptor=receptor,
            analysis_pathway=analysis_pathway,
        )
        for ligand, receptor, analysis_pathway in zip(
            candidates[
                "representative_ligand"
            ],
            candidates[
                "representative_receptor"
            ],
            candidates[
                "pathway"
            ],
        )
    ]

    summary_rows = []

    for display_pathway, group in candidates.groupby(
        "display_pathway",
        sort=False,
    ):
        group = group.copy()

        representative = group.sort_values(
            [
                "unit_simes_p_value",
                "selected_gain_weight_actual",
                "total_positive_gain",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        ).iloc[0]

        unit_p_values = pd.to_numeric(
            group[
                "unit_simes_p_value"
            ],
            errors="coerce",
        ).fillna(1.0).to_numpy(float)

        summary_rows.append({
            "pathway": display_pathway,
            "display_simes_p_value": simes_p_value(
                unit_p_values
            ),
            "selected_gain_weight_actual": float(
                pd.to_numeric(
                    group[
                        "selected_gain_weight_actual"
                    ],
                    errors="coerce",
                ).fillna(0.0).sum()
            ),
            "total_positive_gain": float(
                pd.to_numeric(
                    group[
                        "total_positive_gain"
                    ],
                    errors="coerce",
                ).fillna(0.0).sum()
            ),
            "n_fine_pathway_units": int(
                group[
                    "pathway_unit"
                ].nunique()
            ),
            "n_directed_pairs": int(
                group[
                    "directed_pair"
                ].nunique()
            ),
            "n_selected_gained_programs": int(
                pd.to_numeric(
                    group[
                        "n_selected_gained_programs_actual"
                    ],
                    errors="coerce",
                ).fillna(0.0).sum()
            ),
            "directed_pair": str(
                representative[
                    "directed_pair"
                ]
            ),
            "representative_fine_pathway": str(
                representative[
                    "pathway"
                ]
            ),
            "representative_ligand": str(
                representative[
                    "representative_ligand"
                ]
            ),
            "representative_receptor": str(
                representative[
                    "representative_receptor"
                ]
            ),
            "representative_unit_p_value": float(
                representative[
                    "unit_simes_p_value"
                ]
            ),
            "representative_unit_q_value": float(
                representative[
                    "unit_simes_adjusted_p_value"
                ]
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )
    summary[
        "display_simes_adjusted_p_value"
    ] = bh_fdr(
        summary[
            "display_simes_p_value"
        ].to_numpy(float)
    )

    summary[
        "unit_evidence_tier"
    ] = np.select(
        [
            summary[
                "display_simes_adjusted_p_value"
            ] <= UNIT_CONFIRMED_Q_THRESHOLD,
            summary[
                "display_simes_p_value"
            ] <= UNIT_SIME_P_THRESHOLD,
        ],
        [
            "confirmed_q",
            "potential_simes_p",
        ],
        default="selected_unit_supported",
    )

    tier_order = {
        "confirmed_q": 0,
        "potential_simes_p": 1,
        "selected_unit_supported": 2,
    }
    summary[
        "evidence_tier_order"
    ] = (
        summary[
            "unit_evidence_tier"
        ]
        .map(tier_order)
        .fillna(99)
    )

    summary = summary.sort_values(
        [
            "evidence_tier_order",
            "display_simes_p_value",
            "selected_gain_weight_actual",
            "total_positive_gain",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    if summary[
        "pathway"
    ].duplicated().any():
        raise AssertionError(
            "Post-hoc KEGG-style display summary contains duplicate pathways."
        )

    return summary


def select_top_unique_pathways(
    display_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select up to TOP_N_PATHWAYS from the post-hoc display summary.

    Every broad pathway appears at most once by construction. No new
    statistical filtering is applied here, preventing the display layer from
    collapsing a valid analysis to only one or two pathways.
    """
    top = (
        display_summary
        .head(
            TOP_N_PATHWAYS
        )
        .copy()
        .reset_index(
            drop=True
        )
    )
    top.insert(
        0,
        "rank",
        np.arange(
            1,
            len(top) + 1,
        ),
    )

    if len(top) == 0:
        raise RuntimeError(
            "No KEGG-style display pathway is available for the Top panel."
        )

    if top[
        "pathway"
    ].duplicated().any():
        raise AssertionError(
            "A KEGG-style pathway appears more than once in the Top panel."
        )

    # Plotting function expects these conventional column names.
    top[
        "unit_simes_p_value"
    ] = top[
        "display_simes_p_value"
    ]
    top[
        "unit_simes_adjusted_p_value"
    ] = top[
        "display_simes_adjusted_p_value"
    ]

    return top


# =============================================================================
# 2. LOADERS
# =============================================================================

def load_cci(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    sender_col = first_existing(
        raw,
        ["sender_cluster", "source_cluster", "sender", "source"],
        role="sender cluster",
    )
    receiver_col = first_existing(
        raw,
        ["receiver_cluster", "target_cluster", "receiver", "target"],
        role="receiver cluster",
    )
    ligand_col = first_existing(
        raw,
        ["ligand", "ligand_gene"],
        role="ligand",
    )
    receptor_col = first_existing(
        raw,
        ["receptor", "receptor_gene"],
        role="receptor",
    )

    aliases = {
        "score": ["score", "edge_score", "cci_score", "weight"],
        "confidence": ["confidence", "conf"],
        "n_sources": ["n_sources", "source_count", "n_databases"],
        "expr_only": ["expr_only", "expression_score", "lr_expression"],
        "ligand_ev_z": ["ligand_ev_z", "ligand_z", "ligand_evidence_z"],
        "receptor_ev_z": [
            "receptor_ev_z",
            "receptor_z",
            "receptor_evidence_z",
        ],
        "pathway_raw": [
            "pathway",
            "pathway_name",
            "family",
            "interaction_family",
            "lr_family",
            "interaction_name",
        ],
    }

    table = pd.DataFrame({
        "sender_cluster": raw[sender_col].map(normalize_cluster),
        "receiver_cluster": raw[receiver_col].map(normalize_cluster),
        "ligand": raw[ligand_col].astype(str).str.strip(),
        "receptor": raw[receptor_col].astype(str).str.strip(),
    })

    for output, candidates in aliases.items():
        column = first_existing(raw, candidates, required=False)
        if column is None:
            table[output] = "" if output == "pathway_raw" else np.nan
        else:
            table[output] = raw[column]

    for column in [
        "score",
        "confidence",
        "n_sources",
        "expr_only",
        "ligand_ev_z",
        "receptor_ev_z",
    ]:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    table["score"] = table["score"].fillna(table["expr_only"]).fillna(0.0)
    table["confidence"] = table["confidence"].fillna(1.0)
    table["n_sources"] = table["n_sources"].fillna(1.0)
    table["expr_only"] = table["expr_only"].fillna(0.0)

    table["ligand_key"] = table["ligand"].map(canonical)
    table["receptor_key"] = table["receptor"].map(canonical)
    table["program_key"] = (
        table["sender_cluster"]
        + "|"
        + table["receiver_cluster"]
        + "|"
        + table["ligand_key"]
        + "|"
        + table["receptor_key"]
    )
    table["pathway"] = [
        pathway_label(pathway, ligand, receptor)
        for pathway, ligand, receptor in zip(
            table["pathway_raw"],
            table["ligand"],
            table["receptor"],
        )
    ]

    table = table[
        table["sender_cluster"].ne("")
        & table["receiver_cluster"].ne("")
        & table["ligand_key"].ne("")
        & table["receptor_key"].ne("")
    ].copy()

    return (
        table.sort_values(["score", "confidence"], ascending=False)
        .drop_duplicates("program_key", keep="first")
        .reset_index(drop=True)
    )


def load_gene_tf(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build correlation-only and full-MLTD RNA→TF support.

    Off:
        correlation evidence only.

    On:
        noisy-OR of
            correlation evidence,
            calibrated full-score residual,
            calibrated latent/model representation evidence,
            prior-assisted representation evidence.

    The prior is not allowed to create an edge by itself. It can only
    strengthen latent/full model evidence.
    """
    raw = pd.read_csv(path)

    gene_col = first_existing(
        raw,
        [
            "gene",
            "source_gene",
            "rna",
            "seed_rna",
            "source_rna",
            "gene_name",
        ],
        role="gene",
    )
    tf_col = first_existing(
        raw,
        [
            "tf",
            "tf_name",
            "target_tf",
            "target",
        ],
        role="TF",
    )
    full_col = first_existing(
        raw,
        [
            "final_score",
            "combined_score",
            "mltd_score",
            "edge_score",
            "score",
        ],
        required=False,
    )
    corr_col = first_existing(
        raw,
        [
            "corr01",
            "abs_corr",
            "abs_spearman",
            "spearman",
            "corr",
            "correlation",
        ],
        required=False,
    )
    latent_col = first_existing(
        raw,
        [
            "edge01",
            "latent01",
            "latent_score",
            "edge_cos",
            "edge_weight",
            "latent",
        ],
        required=False,
    )
    prior_col = first_existing(
        raw,
        [
            "prior01",
            "prior",
            "prior_support",
        ],
        required=False,
    )

    if corr_col is None:
        raise KeyError(
            "The all-gene RNA–TF table contains no correlation column. "
            "MLTD-off cannot be defined without correlation-only evidence."
        )

    def numeric(
        column: str | None,
    ) -> np.ndarray:
        if column is None:
            return np.zeros(
                len(raw),
                dtype=float,
            )
        return (
            pd.to_numeric(
                raw[column],
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(float)
        )

    corr_raw = numeric(corr_col)
    latent_raw = numeric(latent_col)
    prior_raw = numeric(prior_col)
    explicit_full_raw = numeric(full_col)

    table = pd.DataFrame({
        "gene": (
            raw[gene_col]
            .astype(str)
            .str.strip()
        ),
        "tf": (
            raw[tf_col]
            .astype(str)
            .str.strip()
        ),
    })
    table["gene_key"] = (
        table["gene"]
        .map(canonical)
    )
    table["tf_key"] = (
        table["tf"]
        .map(canonical)
    )

    table["corr_raw"] = np.clip(
        np.abs(corr_raw),
        0.0,
        1.0,
    )

    if latent_col is None:
        table["latent01"] = 0.0
    elif np.nanmin(latent_raw) < 0.0:
        table["latent01"] = np.clip(
            (latent_raw + 1.0) * 0.5,
            0.0,
            1.0,
        )
    else:
        # Preserve a genuine [0,1] latent column; otherwise min–max scale.
        if (
            np.nanmin(latent_raw) >= 0.0
            and np.nanmax(latent_raw) <= 1.0
        ):
            table["latent01"] = np.clip(
                latent_raw,
                0.0,
                1.0,
            )
        else:
            table["latent01"] = minmax01(
                latent_raw
            )

    table["prior01"] = np.clip(
        prior_raw,
        0.0,
        1.0,
    )

    if full_col is not None:
        if (
            np.nanmin(explicit_full_raw) >= 0.0
            and np.nanmax(explicit_full_raw) <= 1.0
        ):
            table["full_raw"] = np.clip(
                explicit_full_raw,
                0.0,
                1.0,
            )
        else:
            table["full_raw"] = minmax01(
                explicit_full_raw
            )
        full_source = full_col
    else:
        table["full_raw"] = (
            W_CORR
            * table["corr_raw"]
            + W_LATENT
            * table["latent01"]
            + W_PRIOR
            * table["prior01"]
        )
        full_source = "reconstructed_from_components"

    table = table[
        table["gene_key"].ne("")
        & table["tf_key"].ne("")
    ].copy()

    table = (
        table.sort_values(
            [
                "full_raw",
                "latent01",
                "corr_raw",
            ],
            ascending=False,
        )
        .drop_duplicates(
            [
                "gene_key",
                "tf_key",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Correlation-only evidence.
    table["off_support"] = (
        cross_gene_tf_calibration(
            table,
            "corr_raw",
            prefix="corr",
        )
    )

    # Full-score evidence is retained as one channel, but only the part not
    # already represented by correlation is considered incremental.
    table["full_calibrated"] = (
        cross_gene_tf_calibration(
            table,
            "full_raw",
            prefix="full",
        )
    )
    table["full_residual"] = np.clip(
        table["full_calibrated"]
        - table["off_support"],
        0.0,
        1.0,
    )

    # Representation/model evidence is an independent MLTD channel.
    if latent_col is not None and table["latent01"].nunique() > 1:
        table["latent_support"] = (
            cross_gene_tf_calibration(
                table,
                "latent01",
                prefix="latent",
            )
        )
    else:
        table["latent_global_pct"] = 0.0
        table["latent_tf_pct"] = 0.0
        table["latent_support"] = 0.0

    # Prior cannot create a bridge alone. It modulates representation/full
    # model evidence through a joint channel.
    table["model_support_for_prior"] = np.maximum(
        table["latent_support"],
        table["full_calibrated"],
    )
    table["prior_assisted_support"] = (
        table["model_support_for_prior"]
        * (
            PRIOR_ASSIST_BASE
            + (
                1.0
                - PRIOR_ASSIST_BASE
            )
            * table["prior01"]
        )
    )

    table["full_residual_channel"] = (
        FULL_RESIDUAL_RELIABILITY
        * table["full_residual"]
    )
    table["latent_channel"] = (
        LATENT_CHANNEL_RELIABILITY
        * table["latent_support"]
    )
    table["prior_assisted_channel"] = (
        PRIOR_ASSISTED_RELIABILITY
        * table["prior_assisted_support"]
    )

    table["mltd_added_evidence"] = (
        noisy_or_many(
            table["full_residual_channel"],
            table["latent_channel"],
            table["prior_assisted_channel"],
        )
    )

    # Correlation is preserved; MLTD adds independent evidence channels.
    table["on_support"] = noisy_or(
        table["off_support"].to_numpy(float),
        table["mltd_added_evidence"].to_numpy(float),
    )
    table["rna_tf_gain"] = (
        table["on_support"]
        - table["off_support"]
    )

    if (
        table["on_support"]
        < (
            table["off_support"]
            - 1e-12
        )
    ).any():
        raise AssertionError(
            "RNA–TF invariant violated: on_support < off_support."
        )

    qc_rows = []

    component_specs = [
        (
            "correlation_raw",
            "corr_raw",
        ),
        (
            "latent_raw_scaled",
            "latent01",
        ),
        (
            "prior_raw",
            "prior01",
        ),
        (
            "full_raw",
            "full_raw",
        ),
        (
            "off_support",
            "off_support",
        ),
        (
            "full_residual",
            "full_residual",
        ),
        (
            "latent_support",
            "latent_support",
        ),
        (
            "prior_assisted_support",
            "prior_assisted_support",
        ),
        (
            "mltd_added_evidence",
            "mltd_added_evidence",
        ),
        (
            "rna_tf_gain",
            "rna_tf_gain",
        ),
    ]

    for metric, column in component_specs:
        values = pd.to_numeric(
            table[column],
            errors="coerce",
        )
        qc_rows.append({
            "metric": metric,
            "source_column": (
                corr_col
                if metric == "correlation_raw"
                else latent_col
                if metric == "latent_raw_scaled"
                else prior_col
                if metric == "prior_raw"
                else full_source
                if metric == "full_raw"
                else "derived"
            ),
            "n_edges": int(len(values)),
            "n_nonzero": int(
                (values > 1e-12).sum()
            ),
            "fraction_nonzero": float(
                (values > 1e-12).mean()
            ),
            "minimum": float(
                values.min()
            ),
            "median": float(
                values.median()
            ),
            "mean": float(
                values.mean()
            ),
            "maximum": float(
                values.max()
            ),
            "n_unique": int(
                values.nunique(
                    dropna=True
                )
            ),
        })

    qc_rows.extend([
        {
            "metric": "source_columns",
            "source_column": (
                f"corr={corr_col}; "
                f"latent={latent_col}; "
                f"prior={prior_col}; "
                f"full={full_source}"
            ),
            "n_edges": int(len(table)),
            "n_nonzero": np.nan,
            "fraction_nonzero": np.nan,
            "minimum": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "maximum": np.nan,
            "n_unique": np.nan,
        },
        {
            "metric": "positive_rna_tf_gain_edges",
            "source_column": "derived",
            "n_edges": int(len(table)),
            "n_nonzero": int(
                (
                    table["rna_tf_gain"]
                    > 1e-12
                ).sum()
            ),
            "fraction_nonzero": float(
                (
                    table["rna_tf_gain"]
                    > 1e-12
                ).mean()
            ),
            "minimum": float(
                table["rna_tf_gain"].min()
            ),
            "median": float(
                table["rna_tf_gain"].median()
            ),
            "mean": float(
                table["rna_tf_gain"].mean()
            ),
            "maximum": float(
                table["rna_tf_gain"].max()
            ),
            "n_unique": int(
                table["rna_tf_gain"].nunique()
            ),
        },
    ])

    return (
        table,
        pd.DataFrame(qc_rows),
    )


def load_tf_re(
    component_path: Path | None,
    peak_path: Path | None,
    arc_path: Path | None,
) -> pd.DataFrame:
    if component_path is not None and component_path.exists():
        raw = pd.read_csv(component_path)

        tf_col = first_existing(
            raw,
            ["tf", "tf_name", "tf_key"],
            role="TF in TF–RE support",
        )
        support_col = first_existing(
            raw,
            [
                "re_intrinsic",
                "tf_re_support",
                "tf_re_score",
                "re_support",
                "score",
            ],
            required=False,
        )

        table = pd.DataFrame({
            "tf": raw[tf_col].astype(str).str.strip(),
        })

        if support_col is not None:
            table["tf_re_raw"] = (
                pd.to_numeric(raw[support_col], errors="coerce")
                .fillna(0.0)
            )
        else:
            numeric_columns = [
                column
                for column in raw.columns
                if column != tf_col
                and pd.api.types.is_numeric_dtype(raw[column])
            ]
            if not numeric_columns:
                raise ValueError(f"No numeric TF–RE support in {component_path}")

            table["tf_re_raw"] = (
                raw[numeric_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .mean(axis=1)
            )

    else:
        if peak_path is None or not peak_path.exists():
            raise FileNotFoundError(
                "Neither TF_RE_intrinsic_components nor TF-linked RE peaks exist."
            )

        peaks = pd.read_csv(peak_path)
        tf_col = first_existing(
            peaks,
            ["tf", "tf_name", "tf_key"],
            role="TF in TF-linked RE peaks",
        )
        score_col = first_existing(
            peaks,
            ["score", "weight", "support", "peak_score", "tf_re_score"],
            required=False,
        )

        peak_table = pd.DataFrame({
            "tf": peaks[tf_col].astype(str).str.strip(),
            "score": (
                pd.to_numeric(peaks[score_col], errors="coerce").fillna(1.0)
                if score_col is not None
                else 1.0
            ),
        })

        table = (
            peak_table.groupby("tf", as_index=False)
            .agg(
                n_peaks=("score", "size"),
                mean_peak_score=("score", "mean"),
                max_peak_score=("score", "max"),
            )
        )
        table["tf_re_raw"] = (
            0.45 * minmax01(np.log1p(table["n_peaks"]))
            + 0.35 * minmax01(table["mean_peak_score"])
            + 0.20 * minmax01(table["max_peak_score"])
        )

        if arc_path is not None and arc_path.exists():
            arcs = pd.read_csv(arc_path)
            arc_tf_col = first_existing(
                arcs,
                ["tf", "tf_name", "tf_key"],
                role="TF in TF-linked RE arcs",
            )
            arc_score_col = first_existing(
                arcs,
                ["score", "weight", "support", "arc_score", "tf_re_score"],
                required=False,
            )
            arc_table = pd.DataFrame({
                "tf": arcs[arc_tf_col].astype(str).str.strip(),
                "arc_score": (
                    pd.to_numeric(
                        arcs[arc_score_col],
                        errors="coerce",
                    ).fillna(1.0)
                    if arc_score_col is not None
                    else 1.0
                ),
            })
            arc_summary = (
                arc_table.groupby("tf", as_index=False)
                .agg(
                    n_arcs=("arc_score", "size"),
                    mean_arc_score=("arc_score", "mean"),
                )
            )
            table = table.merge(arc_summary, on="tf", how="left")
            table["n_arcs"] = table["n_arcs"].fillna(0.0)
            table["mean_arc_score"] = table["mean_arc_score"].fillna(0.0)
            table["tf_re_raw"] = (
                0.75 * table["tf_re_raw"]
                + 0.15 * minmax01(np.log1p(table["n_arcs"]))
                + 0.10 * minmax01(table["mean_arc_score"])
            )

    table["tf_key"] = table["tf"].map(canonical)
    table = (
        table.groupby("tf_key", as_index=False)
        .agg(
            tf=("tf", "first"),
            tf_re_raw=("tf_re_raw", "max"),
        )
    )
    table["tf_re_support"] = rank_percentile(table["tf_re_raw"])
    return table


def load_receptor_tf(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load receptor–TF identity support and collapse duplicated receptor×TF rows.

    receptor_tf_signature.csv may contain repeated receptor–TF records because
    the same identity can be contributed by multiple source rows, methods or
    contexts. These duplicates are evidence replicates, not independent
    communication programs. A many-to-many merge would duplicate downstream
    bridge rows and inflate weights.

    Aggregation rule:
        signature_raw = strongest available record for receptor×TF
        n_signature_records = number of source rows
        signature_mean_raw = mean source-row score for QC

    Percentile calibration is performed only after deduplication.
    """
    raw = pd.read_csv(path)

    receptor_col = first_existing(
        raw,
        ["receptor", "receptor_gene"],
        role="receptor in receptor–TF signature",
    )
    tf_col = first_existing(
        raw,
        ["tf", "tf_name", "TF_name"],
        role="TF in receptor–TF signature",
    )
    score_col = first_existing(
        raw,
        ["corr", "score", "weight", "tf_score", "tf_corr"],
        required=False,
    )

    table = pd.DataFrame({
        "receptor": (
            raw[receptor_col]
            .astype(str)
            .str.strip()
        ),
        "tf": (
            raw[tf_col]
            .astype(str)
            .str.strip()
        ),
        "signature_raw": (
            pd.to_numeric(
                raw[score_col],
                errors="coerce",
            ).fillna(1.0)
            if score_col is not None
            else 1.0
        ),
    })
    table["receptor_key"] = table["receptor"].map(canonical)
    table["tf_key"] = table["tf"].map(canonical)

    table = table[
        table["receptor_key"].ne("")
        & table["tf_key"].ne("")
    ].copy()

    if len(table) == 0:
        raise RuntimeError(
            f"No valid receptor–TF rows remained after reading {path}."
        )

    duplicate_counts = (
        table.groupby(
            ["receptor_key", "tf_key"],
            as_index=False,
        )
        .agg(
            receptor=("receptor", "first"),
            tf=("tf", "first"),
            n_signature_records=("signature_raw", "size"),
            signature_min_raw=("signature_raw", "min"),
            signature_mean_raw=("signature_raw", "mean"),
            signature_max_raw=("signature_raw", "max"),
        )
    )
    duplicate_counts["is_duplicated_key"] = (
        duplicate_counts["n_signature_records"] > 1
    )

    # Use the strongest source record once. This preserves evidence strength
    # without multiplying the downstream receptor bridge.
    table = duplicate_counts.rename(columns={
        "signature_max_raw": "signature_raw",
    })[[
        "receptor",
        "tf",
        "receptor_key",
        "tf_key",
        "signature_raw",
        "signature_mean_raw",
        "signature_min_raw",
        "n_signature_records",
        "is_duplicated_key",
    ]].copy()

    if table.duplicated(
        ["receptor_key", "tf_key"]
    ).any():
        raise AssertionError(
            "Receptor–TF deduplication failed to create unique keys."
        )

    table["signature_global_pct"] = rank_percentile(
        table["signature_raw"].to_numpy(float)
    )
    table["signature_receptor_pct"] = (
        table.groupby("receptor_key")["signature_raw"]
        .transform(
            lambda values: rank_percentile(
                values.to_numpy(float)
            )
        )
    )
    table["signature_support"] = (
        np.power(
            np.clip(
                table["signature_global_pct"],
                1e-8,
                1.0,
            ),
            0.60,
        )
        * np.power(
            np.clip(
                table["signature_receptor_pct"],
                1e-8,
                1.0,
            ),
            0.40,
        )
    )

    qc = pd.DataFrame([
        {
            "metric": "raw_receptor_TF_rows",
            "value": int(len(raw)),
        },
        {
            "metric": "valid_receptor_TF_rows",
            "value": int(
                duplicate_counts[
                    "n_signature_records"
                ].sum()
            ),
        },
        {
            "metric": "unique_receptor_TF_keys",
            "value": int(len(table)),
        },
        {
            "metric": "duplicated_receptor_TF_keys",
            "value": int(
                table["is_duplicated_key"].sum()
            ),
        },
        {
            "metric": "rows_collapsed_by_deduplication",
            "value": int(
                duplicate_counts[
                    "n_signature_records"
                ].sum()
                - len(table)
            ),
        },
        {
            "metric": "maximum_records_per_key",
            "value": int(
                table[
                    "n_signature_records"
                ].max()
            ),
        },
    ])

    return table, qc


def load_cluster_tf(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    cluster_col = first_existing(
        raw,
        [
            "receiver_cluster",
            "target_cluster",
            "cluster",
            "cluster_id",
            "receiver",
            "target",
        ],
        role="cluster in hybrid TF table",
    )
    tf_col = first_existing(
        raw,
        ["tf_label", "selected_tf", "tf", "best_tf"],
        role="TF in hybrid TF table",
    )
    weight_col = first_existing(
        raw,
        ["tfact_weight", "tf_activity_weight", "tf_weight"],
        required=False,
    )
    z_col = first_existing(
        raw,
        ["tf_act_z", "tf_activity_z", "receiver_tf_z"],
        required=False,
    )

    if weight_col is None and z_col is None:
        raise KeyError(
            "Hybrid TF table has neither tfact_weight nor TF-activity z-score."
        )

    activity = (
        pd.to_numeric(raw[weight_col], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy(float)
        if weight_col is not None
        else sigmoid(
            pd.to_numeric(raw[z_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
    )

    table = pd.DataFrame({
        "cluster": raw[cluster_col].map(normalize_cluster),
        "tf": raw[tf_col].astype(str).str.strip(),
        "tf_activity_observed": activity,
    })
    table["tf_key"] = table["tf"].map(canonical)

    table = table[
        table["cluster"].ne("")
        & table["tf_key"].ne("")
    ].copy()

    return (
        table.groupby(["cluster", "tf_key"], as_index=False)
        .agg(
            tf=("tf", "first"),
            tf_activity_observed=("tf_activity_observed", "max"),
        )
    )


# =============================================================================
# 3. SAME CANDIDATE UNIVERSE
# =============================================================================

def prepare_universe(
    all_cci: pd.DataFrame,
    native_cci: pd.DataFrame,
) -> pd.DataFrame:
    universe = all_cci[
        (all_cci["n_sources"] >= MIN_CANDIDATE_SOURCES)
        & (all_cci["expr_only"] >= MIN_CANDIDATE_EXPR)
    ].copy()

    native = native_cci[["program_key", "score", "confidence"]].copy()

    if len(native):
        native["score_rank"] = rank_percentile(native["score"])
        native["confidence01"] = minmax01(native["confidence"])
        native["native_core"] = (
            NATIVE_SCORE_FLOOR
            + (1.0 - NATIVE_SCORE_FLOOR) * native["score_rank"]
        ) * (0.5 + 0.5 * native["confidence01"])

        native = (
            native.groupby("program_key", as_index=False)
            .agg(native_core=("native_core", "max"))
        )
    else:
        native = pd.DataFrame({
            "program_key": [],
            "native_core": [],
        })

    universe = universe.merge(
        native,
        on="program_key",
        how="left",
        validate="one_to_one",
    )
    universe["native_core"] = universe["native_core"].fillna(0.0)
    universe["native_off"] = universe["native_core"] > 0.0

    confidence01 = minmax01(universe["confidence"])
    source01 = (
        np.log1p(np.clip(universe["n_sources"].to_numpy(float), 0.0, None))
        / math.log1p(max(float(universe["n_sources"].max()), 1.0))
    )
    lr_expression01 = (
        np.clip(
            universe["expr_only"].to_numpy(float)
            / max(float(universe["expr_only"].quantile(0.99)), 1e-12),
            0.0,
            1.0,
        )
    )

    ligand_expression = np.where(
        universe["ligand_ev_z"].notna(),
        sigmoid(universe["ligand_ev_z"].fillna(0.0)),
        np.sqrt(np.clip(universe["expr_only"], 0.0, 1.0)),
    )
    receptor_expression = np.where(
        universe["receptor_ev_z"].notna(),
        sigmoid(universe["receptor_ev_z"].fillna(0.0)),
        np.sqrt(np.clip(universe["expr_only"], 0.0, 1.0)),
    )

    universe["ligand_expression"] = ligand_expression
    universe["receptor_expression"] = receptor_expression
    universe["observed_context"] = weighted_geometric(
        np.column_stack([
            confidence01,
            source01,
            lr_expression01,
            ligand_expression,
            receptor_expression,
        ]),
        np.asarray([
            W_CONTEXT_CONFIDENCE,
            W_CONTEXT_SOURCES,
            W_CONTEXT_LR_EXPRESSION,
            W_CONTEXT_LIGAND_EXPRESSION,
            W_CONTEXT_RECEPTOR_EXPRESSION,
        ]),
    )

    return universe


# =============================================================================
# 4. DENSE CLUSTER TF ACTIVITY WITH SHRUNK BACKFILL
# =============================================================================

def complete_cluster_tf_activity(
    observed: pd.DataFrame,
    clusters: list[str],
    tf_keys: list[str],
) -> pd.DataFrame:
    global_activity = (
        observed.groupby("tf_key")["tf_activity_observed"]
        .mean()
        .to_dict()
    )
    global_median = (
        float(observed["tf_activity_observed"].median())
        if len(observed)
        else 0.25
    )

    observed_map = {
        (row.cluster, row.tf_key): float(row.tf_activity_observed)
        for row in observed.itertuples(index=False)
    }

    rows = []
    for cluster in clusters:
        for tf_key in tf_keys:
            key = (cluster, tf_key)

            if key in observed_map:
                activity = observed_map[key]
                source = "cluster_observed"
            else:
                baseline = global_activity.get(tf_key, global_median)
                activity = max(
                    TF_ACTIVITY_MIN_BACKFILL,
                    TF_ACTIVITY_GLOBAL_BACKFILL * baseline,
                )
                source = "global_shrunk_backfill"

            rows.append({
                "cluster": cluster,
                "tf_key": tf_key,
                "tf_activity": float(np.clip(activity, 0.0, 1.0)),
                "activity_source": source,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 5. CLUSTER-SPECIFIC ON/OFF GENE BRIDGES
# =============================================================================

def cluster_gene_expression(
    universe: pd.DataFrame,
    *,
    side: str,
) -> pd.DataFrame:
    if side == "sender":
        cluster_col = "sender_cluster"
        gene_col = "ligand"
        gene_key_col = "ligand_key"
        expression_col = "ligand_expression"
    elif side == "receiver":
        cluster_col = "receiver_cluster"
        gene_col = "receptor"
        gene_key_col = "receptor_key"
        expression_col = "receptor_expression"
    else:
        raise ValueError(side)

    return (
        universe[[
            cluster_col,
            gene_col,
            gene_key_col,
            expression_col,
        ]]
        .groupby([cluster_col, gene_key_col], as_index=False)
        .agg(
            gene=(gene_col, "first"),
            gene_expression=(expression_col, "max"),
        )
        .rename(columns={
            cluster_col: "cluster",
            gene_key_col: "gene_key",
        })
    )


def build_sender_bridge(
    expression: pd.DataFrame,
    gene_tf: pd.DataFrame,
    tf_re: pd.DataFrame,
    cluster_tf: pd.DataFrame,
) -> pd.DataFrame:
    table = expression.merge(
        gene_tf[[
            "gene_key",
            "tf",
            "tf_key",
            "off_support",
            "on_support",
            "rna_tf_gain",
        ]],
        on="gene_key",
        how="inner",
        validate="many_to_many",
    )
    table = table.merge(
        tf_re[["tf_key", "tf_re_support"]],
        on="tf_key",
        how="inner",
        validate="many_to_one",
    )
    table = table.merge(
        cluster_tf[["cluster", "tf_key", "tf_activity", "activity_source"]],
        on=["cluster", "tf_key"],
        how="left",
        validate="many_to_one",
    )

    table["tf_activity"] = table["tf_activity"].fillna(
        TF_ACTIVITY_MIN_BACKFILL
    )
    table["activity_source"] = table["activity_source"].fillna(
        "minimum_backfill"
    )

    table["molecular_off"] = np.minimum(
        table["off_support"],
        table["tf_re_support"],
    )
    table["molecular_on"] = np.minimum(
        table["on_support"],
        table["tf_re_support"],
    )

    cluster_context = np.sqrt(
        np.clip(
            table["gene_expression"] * table["tf_activity"],
            0.0,
            1.0,
        )
    )
    table["cluster_context"] = cluster_context
    table["bridge_off"] = np.sqrt(
        np.clip(table["molecular_off"] * cluster_context, 0.0, 1.0)
    )
    table["bridge_on"] = np.sqrt(
        np.clip(table["molecular_on"] * cluster_context, 0.0, 1.0)
    )
    table["bridge_gain"] = table["bridge_on"] - table["bridge_off"]

    rows = []
    for (cluster, gene_key), group in table.groupby(
        ["cluster", "gene_key"],
        sort=False,
    ):
        # Select the TF explaining the largest MLTD gain. When all gains are
        # tied, prefer the strongest full bridge.
        best = group.sort_values(
            ["bridge_gain", "bridge_on", "molecular_on"],
            ascending=False,
        ).iloc[0]

        rows.append({
            "cluster": cluster,
            "gene_key": gene_key,
            "gene": best["gene"],
            "best_tf": best["tf"],
            "best_tf_key": best["tf_key"],
            "gene_expression": float(best["gene_expression"]),
            "tf_activity": float(best["tf_activity"]),
            "activity_source": best["activity_source"],
            "molecular_off": float(best["molecular_off"]),
            "molecular_on": float(best["molecular_on"]),
            "bridge_off": float(best["bridge_off"]),
            "bridge_on": float(best["bridge_on"]),
            "bridge_gain": float(best["bridge_gain"]),
            "n_tf_tested": int(group["tf_key"].nunique()),
        })

    return pd.DataFrame(rows)


def build_receiver_bridge(
    expression: pd.DataFrame,
    gene_tf: pd.DataFrame,
    tf_re: pd.DataFrame,
    receptor_tf: pd.DataFrame,
    cluster_tf: pd.DataFrame,
) -> pd.DataFrame:
    base = expression.merge(
        gene_tf[[
            "gene_key",
            "tf",
            "tf_key",
            "off_support",
            "on_support",
            "rna_tf_gain",
        ]],
        on="gene_key",
        how="inner",
        validate="many_to_many",
    )
    base = base.merge(
        tf_re[["tf_key", "tf_re_support"]],
        on="tf_key",
        how="inner",
        validate="many_to_one",
    )
    base = base.merge(
        receptor_tf[[
            "receptor_key",
            "tf_key",
            "signature_support",
        ]],
        left_on=["gene_key", "tf_key"],
        right_on=["receptor_key", "tf_key"],
        how="left",
        validate="many_to_one",
    )

    base["identity_source"] = np.where(
        base["signature_support"].notna(),
        "receptor_tf_signature",
        "penalized_gene_tf_fallback",
    )
    base["identity_support"] = base["signature_support"].fillna(
        RECEPTOR_TF_FALLBACK_PENALTY
    )

    base = base.merge(
        cluster_tf[["cluster", "tf_key", "tf_activity", "activity_source"]],
        on=["cluster", "tf_key"],
        how="left",
        validate="many_to_one",
    )
    base["tf_activity"] = base["tf_activity"].fillna(
        TF_ACTIVITY_MIN_BACKFILL
    )
    base["activity_source"] = base["activity_source"].fillna(
        "minimum_backfill"
    )

    base["molecular_off"] = np.minimum.reduce([
        base["off_support"].to_numpy(float),
        base["tf_re_support"].to_numpy(float),
        base["identity_support"].to_numpy(float),
    ])
    base["molecular_on"] = np.minimum.reduce([
        base["on_support"].to_numpy(float),
        base["tf_re_support"].to_numpy(float),
        base["identity_support"].to_numpy(float),
    ])

    cluster_context = np.sqrt(
        np.clip(
            base["gene_expression"] * base["tf_activity"],
            0.0,
            1.0,
        )
    )
    base["cluster_context"] = cluster_context
    base["bridge_off"] = np.sqrt(
        np.clip(base["molecular_off"] * cluster_context, 0.0, 1.0)
    )
    base["bridge_on"] = np.sqrt(
        np.clip(base["molecular_on"] * cluster_context, 0.0, 1.0)
    )
    base["bridge_gain"] = base["bridge_on"] - base["bridge_off"]

    rows = []
    for (cluster, gene_key), group in base.groupby(
        ["cluster", "gene_key"],
        sort=False,
    ):
        best = group.sort_values(
            [
                "bridge_gain",
                "bridge_on",
                "identity_support",
                "molecular_on",
            ],
            ascending=False,
        ).iloc[0]

        rows.append({
            "cluster": cluster,
            "gene_key": gene_key,
            "gene": best["gene"],
            "best_tf": best["tf"],
            "best_tf_key": best["tf_key"],
            "gene_expression": float(best["gene_expression"]),
            "tf_activity": float(best["tf_activity"]),
            "activity_source": best["activity_source"],
            "identity_support": float(best["identity_support"]),
            "identity_source": best["identity_source"],
            "molecular_off": float(best["molecular_off"]),
            "molecular_on": float(best["molecular_on"]),
            "bridge_off": float(best["bridge_off"]),
            "bridge_on": float(best["bridge_on"]),
            "bridge_gain": float(best["bridge_gain"]),
            "n_tf_tested": int(group["tf_key"].nunique()),
        })

    return pd.DataFrame(rows)


# =============================================================================
# 6. PROGRAM SCORES
# =============================================================================

def attach_program_scores(
    universe: pd.DataFrame,
    sender: pd.DataFrame,
    receiver: pd.DataFrame,
) -> pd.DataFrame:
    sender_columns = sender[[
        "cluster",
        "gene_key",
        "best_tf",
        "best_tf_key",
        "gene_expression",
        "tf_activity",
        "activity_source",
        "molecular_off",
        "molecular_on",
        "bridge_off",
        "bridge_on",
        "bridge_gain",
    ]].rename(columns={
        "cluster": "sender_cluster",
        "gene_key": "ligand_key",
        "best_tf": "sender_tf",
        "best_tf_key": "sender_tf_key",
        "gene_expression": "sender_gene_expression",
        "tf_activity": "sender_tf_activity",
        "activity_source": "sender_activity_source",
        "molecular_off": "sender_molecular_off",
        "molecular_on": "sender_molecular_on",
        "bridge_off": "sender_bridge_off",
        "bridge_on": "sender_bridge_on",
        "bridge_gain": "sender_bridge_gain",
    })

    receiver_columns = receiver[[
        "cluster",
        "gene_key",
        "best_tf",
        "best_tf_key",
        "gene_expression",
        "tf_activity",
        "activity_source",
        "identity_support",
        "identity_source",
        "molecular_off",
        "molecular_on",
        "bridge_off",
        "bridge_on",
        "bridge_gain",
    ]].rename(columns={
        "cluster": "receiver_cluster",
        "gene_key": "receptor_key",
        "best_tf": "receiver_tf",
        "best_tf_key": "receiver_tf_key",
        "gene_expression": "receiver_gene_expression",
        "tf_activity": "receiver_tf_activity",
        "activity_source": "receiver_activity_source",
        "identity_support": "receiver_identity_support",
        "identity_source": "receiver_identity_source",
        "molecular_off": "receiver_molecular_off",
        "molecular_on": "receiver_molecular_on",
        "bridge_off": "receiver_bridge_off",
        "bridge_on": "receiver_bridge_on",
        "bridge_gain": "receiver_bridge_gain",
    })

    programs = universe.merge(
        sender_columns,
        on=["sender_cluster", "ligand_key"],
        how="left",
        validate="many_to_one",
    )
    programs = programs.merge(
        receiver_columns,
        on=["receiver_cluster", "receptor_key"],
        how="left",
        validate="many_to_one",
    )

    numeric_bridge_columns = [
        "sender_gene_expression",
        "sender_tf_activity",
        "sender_molecular_off",
        "sender_molecular_on",
        "sender_bridge_off",
        "sender_bridge_on",
        "sender_bridge_gain",
        "receiver_gene_expression",
        "receiver_tf_activity",
        "receiver_identity_support",
        "receiver_molecular_off",
        "receiver_molecular_on",
        "receiver_bridge_off",
        "receiver_bridge_on",
        "receiver_bridge_gain",
    ]

    for column in numeric_bridge_columns:
        programs[column] = (
            pd.to_numeric(programs[column], errors="coerce").fillna(0.0)
        )

    pair_bridge_off = noisy_or(
        programs["sender_bridge_off"].to_numpy(float),
        programs["receiver_bridge_off"].to_numpy(float),
    )
    pair_bridge_on = noisy_or(
        programs["sender_bridge_on"].to_numpy(float),
        programs["receiver_bridge_on"].to_numpy(float),
    )

    programs["pair_bridge_off"] = pair_bridge_off
    programs["pair_bridge_on"] = np.maximum(pair_bridge_off, pair_bridge_on)
    programs["pair_bridge_gain"] = (
        programs["pair_bridge_on"] - programs["pair_bridge_off"]
    )

    latent_off = programs["observed_context"].to_numpy(float) * pair_bridge_off
    latent_on = (
        programs["observed_context"].to_numpy(float)
        * programs["pair_bridge_on"].to_numpy(float)
    )

    native_component = (
        NATIVE_CORE_WEIGHT * programs["native_core"].to_numpy(float)
    )
    bridge_off_component = BRIDGE_CORE_WEIGHT * latent_off
    bridge_on_component = BRIDGE_CORE_WEIGHT * latent_on

    programs["score_off"] = noisy_or(
        native_component,
        bridge_off_component,
    )
    programs["score_on"] = noisy_or(
        native_component,
        bridge_on_component,
    )
    programs["mltd_gain"] = programs["score_on"] - programs["score_off"]

    if (programs["score_on"] < programs["score_off"] - 1e-12).any():
        raise AssertionError("Program invariant violated: score_on < score_off.")

    return programs


# =============================================================================
# 7. PAIRS AND LOOKUPS
# =============================================================================

def get_cluster_order(programs: pd.DataFrame) -> list[str]:
    clusters = sorted(
        set(programs["sender_cluster"]) | set(programs["receiver_cluster"]),
        key=cluster_sort_key,
    )

    if CLUSTER_ORDER_MANUAL is not None:
        manual = [normalize_cluster(value) for value in CLUSTER_ORDER_MANUAL]
        missing = sorted(set(clusters) - set(manual))
        if missing:
            raise ValueError(f"CLUSTER_ORDER_MANUAL omits clusters: {missing}")
        clusters = manual

    return clusters


def add_pair_columns(
    programs: pd.DataFrame,
    clusters: list[str],
) -> pd.DataFrame:
    order = {cluster: index for index, cluster in enumerate(clusters)}
    output = programs.copy()

    output["directed_pair"] = (
        output["sender_cluster"] + "→" + output["receiver_cluster"]
    )
    output["pathway_unit"] = output["directed_pair"] + "|" + output["pathway"]

    pair_left = []
    pair_right = []
    direction_class = []

    for sender, receiver in zip(
        output["sender_cluster"],
        output["receiver_cluster"],
    ):
        sender_index = order[sender]
        receiver_index = order[receiver]

        if sender_index < receiver_index:
            pair_left.append(sender)
            pair_right.append(receiver)
            direction_class.append("low_to_high")
        elif sender_index > receiver_index:
            pair_left.append(receiver)
            pair_right.append(sender)
            direction_class.append("high_to_low")
        else:
            pair_left.append(sender)
            pair_right.append(receiver)
            direction_class.append("self")

    output["pair_left"] = pair_left
    output["pair_right"] = pair_right
    output["unordered_pair"] = output["pair_left"] + "–" + output["pair_right"]
    output["direction_class"] = direction_class
    return output


def categorical_codes(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    labels = sorted(values.astype(str).unique().tolist())
    lookup = {label: index for index, label in enumerate(labels)}
    return values.astype(str).map(lookup).to_numpy(int), labels


def bridge_bundle_lookup(
    programs: pd.DataFrame,
    *,
    side: str,
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    if side == "sender":
        cluster_col = "sender_cluster"
        gene_col = "ligand_key"
        off_col = "sender_bridge_off"
        on_col = "sender_bridge_on"
    elif side == "receiver":
        cluster_col = "receiver_cluster"
        gene_col = "receptor_key"
        off_col = "receiver_bridge_off"
        on_col = "receiver_bridge_on"
    else:
        raise ValueError(side)

    lookup = (
        programs[[cluster_col, gene_col, off_col, on_col]]
        .drop_duplicates([cluster_col, gene_col])
        .rename(columns={
            cluster_col: "cluster",
            gene_col: "gene_key",
            off_col: "bridge_off",
            on_col: "bridge_on",
        })
        .reset_index(drop=True)
    )
    lookup["lookup_index"] = np.arange(len(lookup))

    program_index = (
        programs[[cluster_col, gene_col]]
        .merge(
            lookup[["cluster", "gene_key", "lookup_index"]],
            left_on=[cluster_col, gene_col],
            right_on=["cluster", "gene_key"],
            how="left",
            validate="many_to_one",
        )["lookup_index"]
        .to_numpy(int)
    )

    cluster_groups = [
        group["lookup_index"].to_numpy(int)
        for _, group in lookup.groupby("cluster", sort=False)
    ]

    return lookup, program_index, cluster_groups


# =============================================================================
# 8. EMPIRICAL NULLS
# =============================================================================

def simes_p_value(
    p_values: np.ndarray | pd.Series,
) -> float:
    """
    Simes combined P value for sparse positive signals.

    This is used hierarchically:
        program P values -> pathway-unit P
        pathway-unit P values -> directed-pair P
    """
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]

    if len(p) == 0:
        return 1.0

    p = np.clip(p, 0.0, 1.0)
    p.sort()
    m = len(p)

    adjusted = (
        p
        * float(m)
        / np.arange(1, m + 1)
    )
    return float(
        np.clip(
            np.min(adjusted),
            0.0,
            1.0,
        )
    )


def empirical_nulls(
    programs: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Compute:
        1. per-program empirical MLTD-gain P values;
        2. native/off directed-pair P values;
        3. compact permutation diagnostics.

    V5 deliberately does NOT use total pair gain versus pair-gain null as the
    selection statistic. That statistic failed because broad candidate noise
    diluted sparse identity-specific programs.
    """
    pair_codes, pair_labels = categorical_codes(
        programs["directed_pair"]
    )

    n_programs = len(programs)
    n_pairs = len(pair_labels)

    observed_gain = programs[
        "mltd_gain"
    ].to_numpy(float)
    observed_off_score = programs[
        "score_off"
    ].to_numpy(float)

    observed_pair_off = np.bincount(
        pair_codes,
        weights=np.where(
            programs["native_off"],
            observed_off_score,
            0.0,
        ),
        minlength=n_pairs,
    )

    pathway_values = (
        programs["pathway"]
        .astype(str)
        .to_numpy()
    )
    pathway_groups = [
        np.where(
            pathway_values == pathway
        )[0]
        for pathway in pd.unique(
            pathway_values
        )
    ]

    (
        sender_lookup,
        sender_program_index,
        sender_cluster_groups,
    ) = bridge_bundle_lookup(
        programs,
        side="sender",
    )
    (
        receiver_lookup,
        receiver_program_index,
        receiver_cluster_groups,
    ) = bridge_bundle_lookup(
        programs,
        side="receiver",
    )

    sender_off_native = (
        sender_lookup[
            "bridge_off"
        ].to_numpy(float)
    )
    sender_on_native = (
        sender_lookup[
            "bridge_on"
        ].to_numpy(float)
    )
    receiver_off_native = (
        receiver_lookup[
            "bridge_off"
        ].to_numpy(float)
    )
    receiver_on_native = (
        receiver_lookup[
            "bridge_on"
        ].to_numpy(float)
    )

    native_component = (
        NATIVE_CORE_WEIGHT
        * programs[
            "native_core"
        ].to_numpy(float)
    )
    observed_context = programs[
        "observed_context"
    ].to_numpy(float)
    native_mask = programs[
        "native_off"
    ].to_numpy(bool)

    rng = np.random.default_rng(
        PERMUTATION_SEED
    )

    program_gain_exceed = np.zeros(
        n_programs,
        dtype=np.int64,
    )
    pair_off_exceed = np.zeros(
        n_pairs,
        dtype=np.int64,
    )

    null_rows = []

    for permutation_id in range(
        N_PERMUTATIONS
    ):
        # Native CCI null.
        off_native_null = np.where(
            native_mask,
            observed_off_score,
            0.0,
        ).copy()

        for indices in pathway_groups:
            if len(indices) > 1:
                off_native_null[
                    indices
                ] = rng.permutation(
                    off_native_null[
                        indices
                    ]
                )

        pair_off_null = np.bincount(
            pair_codes,
            weights=off_native_null,
            minlength=n_pairs,
        )
        pair_off_exceed += (
            pair_off_null
            >= (
                observed_pair_off
                - 1e-15
            )
        )

        # Gene-identity-matched MLTD null.
        sender_off = (
            sender_off_native.copy()
        )
        sender_on = (
            sender_on_native.copy()
        )

        for indices in (
            sender_cluster_groups
        ):
            if len(indices) > 1:
                permutation = (
                    rng.permutation(
                        len(indices)
                    )
                )
                sender_off[
                    indices
                ] = (
                    sender_off_native[
                        indices
                    ][permutation]
                )
                sender_on[
                    indices
                ] = (
                    sender_on_native[
                        indices
                    ][permutation]
                )

        receiver_off = (
            receiver_off_native.copy()
        )
        receiver_on = (
            receiver_on_native.copy()
        )

        for indices in (
            receiver_cluster_groups
        ):
            if len(indices) > 1:
                permutation = (
                    rng.permutation(
                        len(indices)
                    )
                )
                receiver_off[
                    indices
                ] = (
                    receiver_off_native[
                        indices
                    ][permutation]
                )
                receiver_on[
                    indices
                ] = (
                    receiver_on_native[
                        indices
                    ][permutation]
                )

        sender_off_program = (
            sender_off[
                sender_program_index
            ]
        )
        sender_on_program = (
            sender_on[
                sender_program_index
            ]
        )
        receiver_off_program = (
            receiver_off[
                receiver_program_index
            ]
        )
        receiver_on_program = (
            receiver_on[
                receiver_program_index
            ]
        )

        pair_bridge_off_null = noisy_or(
            sender_off_program,
            receiver_off_program,
        )
        pair_bridge_on_null = noisy_or(
            sender_on_program,
            receiver_on_program,
        )
        pair_bridge_on_null = np.maximum(
            pair_bridge_off_null,
            pair_bridge_on_null,
        )

        score_off_null = noisy_or(
            native_component,
            (
                BRIDGE_CORE_WEIGHT
                * observed_context
                * pair_bridge_off_null
            ),
        )
        score_on_null = noisy_or(
            native_component,
            (
                BRIDGE_CORE_WEIGHT
                * observed_context
                * pair_bridge_on_null
            ),
        )
        gain_null = (
            score_on_null
            - score_off_null
        )

        program_gain_exceed += (
            gain_null
            >= (
                observed_gain
                - 1e-15
            )
        )

        null_rows.append({
            "permutation_id": (
                permutation_id
            ),
            "total_gain": float(
                gain_null.sum()
            ),
            "max_program_gain": float(
                gain_null.max()
            ),
            "n_positive_gain_programs": int(
                (
                    gain_null > 0
                ).sum()
            ),
        })

    program_p = (
        program_gain_exceed + 1
    ) / float(
        N_PERMUTATIONS + 1
    )
    pair_off_p = (
        pair_off_exceed + 1
    ) / float(
        N_PERMUTATIONS + 1
    )

    program_stats = programs[[
        "program_key",
        "directed_pair",
        "pathway_unit",
        "mltd_gain",
    ]].copy()
    program_stats[
        "gain_p_value"
    ] = program_p
    program_stats[
        "gain_adjusted_p_value"
    ] = bh_fdr(
        program_p
    )

    pair_rows = []
    for index, label in enumerate(
        pair_labels
    ):
        sender, receiver = (
            label.split("→", 1)
        )
        pair_rows.append({
            "directed_pair": label,
            "sender_cluster": sender,
            "receiver_cluster": receiver,
            "native_off_weight": float(
                observed_pair_off[
                    index
                ]
            ),
            "off_p_value": float(
                pair_off_p[
                    index
                ]
            ),
        })

    native_pair_stats = pd.DataFrame(
        pair_rows
    )
    native_pair_stats[
        "off_adjusted_p_value"
    ] = bh_fdr(
        native_pair_stats[
            "off_p_value"
        ].to_numpy(float)
    )
    native_pair_stats[
        "off_pair_selected"
    ] = (
        (
            native_pair_stats[
                "native_off_weight"
            ] > 0
        )
        & (
            (
                native_pair_stats[
                    "off_adjusted_p_value"
                ]
                <= OFF_PAIR_Q_THRESHOLD
            )
            | (
                native_pair_stats[
                    "off_p_value"
                ]
                <= OFF_PAIR_P_THRESHOLD
            )
        )
    )
    native_pair_stats[
        "off_evidence_tier"
    ] = np.select(
        [
            (
                native_pair_stats[
                    "off_adjusted_p_value"
                ]
                <= OFF_PAIR_Q_THRESHOLD
            ),
            native_pair_stats[
                "off_pair_selected"
            ],
        ],
        [
            "confirmed_q",
            "potential_p",
        ],
        default="not_selected",
    )

    return (
        program_stats,
        native_pair_stats,
        pd.DataFrame(
            null_rows
        ),
    )


def build_sparse_gain_hierarchy(
    programs: pd.DataFrame,
    program_stats: pd.DataFrame,
    native_pair_stats: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Hierarchical sparse-signal selection.

    Stage 1:
        combine non-native candidate program P values inside each
        directed pair × pathway using Simes.

    Stage 2:
        combine pathway-unit P values inside each directed pair using Simes.

    Program activation:
        positive gain
        + parent pathway unit selected
        + program empirical P below the tier-specific threshold
        + at most MAX_GAINED_PROGRAMS_PER_UNIT per unit.
    """
    output = programs.merge(
        program_stats[[
            "program_key",
            "gain_p_value",
            "gain_adjusted_p_value",
        ]],
        on="program_key",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------------
    # Pathway-unit statistics.
    # ---------------------------------------------------------------------
    unit_rows = []

    for pathway_unit, group in output.groupby(
        "pathway_unit",
        sort=False,
    ):
        non_native = group[
            ~group[
                "native_off"
            ]
        ].copy()
        positive = non_native[
            non_native[
                "mltd_gain"
            ] > POSITIVE_GAIN_EPS
        ].copy()

        p_values = (
            non_native[
                "gain_p_value"
            ].fillna(1.0)
            .to_numpy(float)
        )

        representative = (
            positive.sort_values(
                [
                    "gain_p_value",
                    "mltd_gain",
                ],
                ascending=[
                    True,
                    False,
                ],
            ).iloc[0]
            if len(positive)
            else group.sort_values(
                "mltd_gain",
                ascending=False,
            ).iloc[0]
        )

        unit_rows.append({
            "pathway_unit": pathway_unit,
            "directed_pair": (
                group.iloc[0][
                    "directed_pair"
                ]
            ),
            "sender_cluster": (
                group.iloc[0][
                    "sender_cluster"
                ]
            ),
            "receiver_cluster": (
                group.iloc[0][
                    "receiver_cluster"
                ]
            ),
            "pathway": (
                group.iloc[0][
                    "pathway"
                ]
            ),
            "unit_simes_p_value": (
                simes_p_value(
                    p_values
                )
            ),
            "n_non_native_candidates": int(
                len(non_native)
            ),
            "n_positive_gain_programs": int(
                len(positive)
            ),
            "n_program_p_le_0_01": int(
                (
                    positive[
                        "gain_p_value"
                    ]
                    <= (
                        PROGRAM_P_THRESHOLD_POTENTIAL
                    )
                ).sum()
            ),
            "n_program_p_le_0_05": int(
                (
                    positive[
                        "gain_p_value"
                    ]
                    <= (
                        PROGRAM_P_THRESHOLD_CONFIRMED
                    )
                ).sum()
            ),
            "minimum_program_p": float(
                positive[
                    "gain_p_value"
                ].min()
                if len(positive)
                else 1.0
            ),
            "total_positive_gain": float(
                positive[
                    "mltd_gain"
                ].sum()
            ),
            "maximum_program_gain": float(
                positive[
                    "mltd_gain"
                ].max()
                if len(positive)
                else 0.0
            ),
            "representative_ligand": (
                representative[
                    "ligand"
                ]
            ),
            "representative_receptor": (
                representative[
                    "receptor"
                ]
            ),
            "representative_sender_tf": (
                representative.get(
                    "sender_tf",
                    "",
                )
            ),
            "representative_receiver_tf": (
                representative.get(
                    "receiver_tf",
                    "",
                )
            ),
        })

    unit_stats = pd.DataFrame(
        unit_rows
    )
    unit_stats[
        "unit_simes_adjusted_p_value"
    ] = bh_fdr(
        unit_stats[
            "unit_simes_p_value"
        ].to_numpy(float)
    )

    unit_stats[
        "confirmed_gain_unit"
    ] = (
        (
            unit_stats[
                "n_positive_gain_programs"
            ] > 0
        )
        & (
            unit_stats[
                "unit_simes_adjusted_p_value"
            ]
            <= UNIT_CONFIRMED_Q_THRESHOLD
        )
        & (
            unit_stats[
                "n_program_p_le_0_05"
            ] >= 1
        )
    )

    unit_stats[
        "potential_gain_unit"
    ] = (
        unit_stats[
            "confirmed_gain_unit"
        ]
        | (
            (
                unit_stats[
                    "n_positive_gain_programs"
                ] > 0
            )
            & (
                unit_stats[
                    "unit_simes_p_value"
                ]
                <= UNIT_SIME_P_THRESHOLD
            )
            & (
                unit_stats[
                    "n_program_p_le_0_01"
                ] >= 1
            )
        )
    )

    unit_stats[
        "unit_evidence_tier"
    ] = np.select(
        [
            unit_stats[
                "confirmed_gain_unit"
            ],
            unit_stats[
                "potential_gain_unit"
            ],
        ],
        [
            "confirmed_q",
            "potential_simes_p",
        ],
        default="not_selected",
    )

    # ---------------------------------------------------------------------
    # Select a bounded number of gained programs per selected pathway unit.
    # ---------------------------------------------------------------------
    selected_program_keys: set[str] = set()

    selected_units = unit_stats[
        unit_stats[
            "potential_gain_unit"
        ]
    ]

    unit_tier_lookup = (
        selected_units
        .set_index(
            "pathway_unit"
        )[
            "unit_evidence_tier"
        ]
        .to_dict()
    )

    for pathway_unit, tier in (
        unit_tier_lookup.items()
    ):
        threshold = (
            PROGRAM_P_THRESHOLD_CONFIRMED
            if tier == "confirmed_q"
            else PROGRAM_P_THRESHOLD_POTENTIAL
        )

        candidates = output[
            (
                output[
                    "pathway_unit"
                ] == pathway_unit
            )
            & (
                ~output[
                    "native_off"
                ]
            )
            & (
                output[
                    "mltd_gain"
                ] > POSITIVE_GAIN_EPS
            )
            & (
                output[
                    "gain_p_value"
                ] <= threshold
            )
        ].sort_values(
            [
                "gain_p_value",
                "mltd_gain",
                "score_on",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )

        candidates = candidates.head(
            MAX_GAINED_PROGRAMS_PER_UNIT
        )

        selected_program_keys.update(
            candidates[
                "program_key"
            ].astype(str)
        )

    output[
        "potential_gained_program"
    ] = (
        output[
            "program_key"
        ].astype(str)
        .isin(
            selected_program_keys
        )
    )

    # Add actual selected-program summaries to units.
    selected_program_summary = (
        output[
            output[
                "potential_gained_program"
            ]
        ]
        .groupby(
            "pathway_unit",
            as_index=False,
        )
        .agg(
            n_selected_gained_programs=(
                "program_key",
                "nunique",
            ),
            selected_gain_weight=(
                "mltd_gain",
                "sum",
            ),
            selected_on_weight=(
                "score_on",
                "sum",
            ),
        )
    )

    unit_stats = unit_stats.merge(
        selected_program_summary,
        on="pathway_unit",
        how="left",
        validate="one_to_one",
    )
    for column in [
        "n_selected_gained_programs",
        "selected_gain_weight",
        "selected_on_weight",
    ]:
        unit_stats[column] = (
            unit_stats[column]
            .fillna(0.0)
        )

    # ---------------------------------------------------------------------
    # Directed-pair sparse combination.
    # ---------------------------------------------------------------------
    pair_rows = []

    for directed_pair, group in unit_stats.groupby(
        "directed_pair",
        sort=False,
    ):
        informative = group[
            group[
                "n_positive_gain_programs"
            ] > 0
        ]
        selected = group[
            group[
                "potential_gain_unit"
            ]
        ]

        pair_rows.append({
            "directed_pair": directed_pair,
            "sender_cluster": (
                group.iloc[0][
                    "sender_cluster"
                ]
            ),
            "receiver_cluster": (
                group.iloc[0][
                    "receiver_cluster"
                ]
            ),
            "pair_simes_p_value": (
                simes_p_value(
                    informative[
                        "unit_simes_p_value"
                    ].to_numpy(float)
                )
                if len(informative)
                else 1.0
            ),
            "n_pathway_units": int(
                len(group)
            ),
            "n_positive_pathway_units": int(
                len(informative)
            ),
            "n_selected_pathway_units": int(
                len(selected)
            ),
            "n_selected_gained_programs": int(
                selected[
                    "n_selected_gained_programs"
                ].sum()
            ),
            "selected_gain_weight": float(
                selected[
                    "selected_gain_weight"
                ].sum()
            ),
        })

    pair_sparse = pd.DataFrame(
        pair_rows
    )
    pair_sparse[
        "pair_simes_adjusted_p_value"
    ] = bh_fdr(
        pair_sparse[
            "pair_simes_p_value"
        ].to_numpy(float)
    )
    pair_sparse[
        "gain_pair_selected"
    ] = (
        pair_sparse[
            "n_selected_pathway_units"
        ] > 0
    )
    pair_sparse[
        "gain_pair_evidence_tier"
    ] = np.select(
        [
            (
                pair_sparse[
                    "gain_pair_selected"
                ]
                & (
                    pair_sparse[
                        "pair_simes_adjusted_p_value"
                    ]
                    <= UNIT_CONFIRMED_Q_THRESHOLD
                )
            ),
            pair_sparse[
                "gain_pair_selected"
            ],
        ],
        [
            "confirmed_q",
            "pathway_supported",
        ],
        default="not_selected",
    )

    pair_stats = (
        native_pair_stats.merge(
            pair_sparse,
            on=[
                "directed_pair",
                "sender_cluster",
                "receiver_cluster",
            ],
            how="outer",
            validate="one_to_one",
        )
    )

    fill_zero_columns = [
        "native_off_weight",
        "n_pathway_units",
        "n_positive_pathway_units",
        "n_selected_pathway_units",
        "n_selected_gained_programs",
        "selected_gain_weight",
    ]
    for column in fill_zero_columns:
        pair_stats[column] = (
            pd.to_numeric(
                pair_stats[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    fill_one_columns = [
        "off_p_value",
        "off_adjusted_p_value",
        "pair_simes_p_value",
        "pair_simes_adjusted_p_value",
    ]
    for column in fill_one_columns:
        pair_stats[column] = (
            pd.to_numeric(
                pair_stats[column],
                errors="coerce",
            )
            .fillna(1.0)
        )

    for column in [
        "off_pair_selected",
        "gain_pair_selected",
    ]:
        pair_stats[column] = (
            pair_stats[column]
            .fillna(False)
            .astype(bool)
        )

    pair_stats[
        "off_evidence_tier"
    ] = pair_stats[
        "off_evidence_tier"
    ].fillna(
        "not_selected"
    )
    pair_stats[
        "gain_pair_evidence_tier"
    ] = pair_stats[
        "gain_pair_evidence_tier"
    ].fillna(
        "not_selected"
    )

    # ---------------------------------------------------------------------
    # Activate programs.
    # ---------------------------------------------------------------------
    pair_flag_columns = pair_stats[[
        "directed_pair",
        "off_pair_selected",
        "gain_pair_selected",
        "off_p_value",
        "off_adjusted_p_value",
        "pair_simes_p_value",
        "pair_simes_adjusted_p_value",
        "off_evidence_tier",
        "gain_pair_evidence_tier",
    ]]

    output = output.merge(
        pair_flag_columns,
        on="directed_pair",
        how="left",
        validate="many_to_one",
    )

    output[
        "off_active"
    ] = (
        output[
            "native_off"
        ]
        & output[
            "off_pair_selected"
        ].fillna(False)
    )
    output[
        "on_active"
    ] = (
        output[
            "off_active"
        ]
        | output[
            "potential_gained_program"
        ]
    )

    return (
        output,
        pair_stats,
        unit_stats,
    )


def fail_if_no_useful_gain(
    programs: pd.DataFrame,
    pair_stats: pd.DataFrame,
    unit_stats: pd.DataFrame,
    diagnostic_path: Path,
) -> None:
    diagnostic_rows = []

    for level, table, metrics in [
        (
            "program",
            programs,
            [
                "mltd_gain",
                "gain_p_value",
                "gain_adjusted_p_value",
            ],
        ),
        (
            "pathway_unit",
            unit_stats,
            [
                "unit_simes_p_value",
                "unit_simes_adjusted_p_value",
                "total_positive_gain",
                "n_program_p_le_0_01",
            ],
        ),
        (
            "directed_pair",
            pair_stats,
            [
                "off_p_value",
                "off_adjusted_p_value",
                "pair_simes_p_value",
                "pair_simes_adjusted_p_value",
                "n_selected_pathway_units",
            ],
        ),
    ]:
        for metric in metrics:
            values = pd.to_numeric(
                table[metric],
                errors="coerce",
            )

            for quantile in [
                0.0,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.0,
            ]:
                diagnostic_rows.append({
                    "level": level,
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(
                        values.quantile(
                            quantile
                        )
                    ),
                })

    pd.DataFrame(
        diagnostic_rows
    ).to_csv(
        diagnostic_path,
        index=False,
    )
    log(
        "[SAVE]",
        diagnostic_path,
    )

    positive_programs = int(
        (
            programs[
                "mltd_gain"
            ] > POSITIVE_GAIN_EPS
        ).sum()
    )
    selected_units = int(
        unit_stats[
            "potential_gain_unit"
        ].sum()
    )
    selected_programs = int(
        programs[
            "potential_gained_program"
        ].sum()
    )
    off_pairs = set(
        pair_stats.loc[
            pair_stats[
                "off_pair_selected"
            ],
            "directed_pair",
        ]
    )
    gain_pairs = set(
        pair_stats.loc[
            pair_stats[
                "gain_pair_selected"
            ],
            "directed_pair",
        ]
    )
    new_pairs = (
        gain_pairs
        - off_pairs
    )

    if positive_programs == 0:
        raise RuntimeError(
            "No program has positive MLTD gain. "
            f"See: {diagnostic_path}"
        )

    if selected_units == 0:
        best_units = (
            unit_stats.sort_values(
                [
                    "unit_simes_p_value",
                    "minimum_program_p",
                    "total_positive_gain",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            .head(15)[[
                "directed_pair",
                "pathway",
                "unit_simes_p_value",
                "unit_simes_adjusted_p_value",
                "minimum_program_p",
                "n_program_p_le_0_01",
                "n_program_p_le_0_05",
                "total_positive_gain",
            ]]
        )

        raise RuntimeError(
            "Positive program-level gains exist, but no pathway unit passed "
            "the sparse Simes rule.\n"
            f"Potential rule: unit Simes P <= {UNIT_SIME_P_THRESHOLD} "
            f"and at least one program P <= "
            f"{PROGRAM_P_THRESHOLD_POTENTIAL}; "
            f"confirmed rule: unit Q <= "
            f"{UNIT_CONFIRMED_Q_THRESHOLD}.\n\n"
            "Best pathway units:\n"
            f"{best_units.to_string(index=False)}\n\n"
            f"See: {diagnostic_path}"
        )

    if selected_programs == 0:
        raise RuntimeError(
            "Pathway-unit evidence exists, but no non-native program passed "
            "the tier-specific program P threshold. "
            f"See: {diagnostic_path}"
        )

    if len(off_pairs) == 0:
        raise RuntimeError(
            "No native directed pair passed the off-state significance rule. "
            "The off map would be empty. "
            f"See: {diagnostic_path}"
        )

    if (
        REQUIRE_NEW_DIRECTED_PAIR
        and len(new_pairs) == 0
    ):
        best_gain_pairs = (
            pair_stats[
                pair_stats[
                    "gain_pair_selected"
                ]
            ]
            .sort_values(
                [
                    "pair_simes_p_value",
                    "selected_gain_weight",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(15)[[
                "directed_pair",
                "pair_simes_p_value",
                "pair_simes_adjusted_p_value",
                "n_selected_pathway_units",
                "n_selected_gained_programs",
                "selected_gain_weight",
                "off_pair_selected",
            ]]
        )

        raise RuntimeError(
            "Sparse MLTD pathway evidence was detected, but every supported "
            "gain occurs inside a directed pair already present in MLTD-off. "
            "Therefore the pair-level topology is unchanged and the script "
            "refuses to claim communication expansion.\n\n"
            "Supported gain pairs:\n"
            f"{best_gain_pairs.to_string(index=False)}\n\n"
            "Set REQUIRE_NEW_DIRECTED_PAIR=False only when the intended claim "
            "is pathway enrichment within existing cluster pairs rather than "
            "new pair-level communication.\n"
            f"See: {diagnostic_path}"
        )


def aggregate_directed_pairs(
    programs: pd.DataFrame,
    pair_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for directed_pair, group in programs.groupby(
        "directed_pair",
        sort=False,
    ):
        off_group = group[
            group[
                "off_active"
            ]
        ]
        on_group = group[
            group[
                "on_active"
            ]
        ]

        rows.append({
            "directed_pair": directed_pair,
            "sender_cluster": (
                group.iloc[0][
                    "sender_cluster"
                ]
            ),
            "receiver_cluster": (
                group.iloc[0][
                    "receiver_cluster"
                ]
            ),
            "off_weight": float(
                off_group[
                    "score_off"
                ].sum()
            ),
            "on_weight": float(
                on_group[
                    "score_on"
                ].sum()
            ),
            "selected_gain_weight": float(
                group.loc[
                    group[
                        "potential_gained_program"
                    ],
                    "mltd_gain",
                ].sum()
            ),
            "n_off_programs": int(
                off_group[
                    "program_key"
                ].nunique()
            ),
            "n_on_programs": int(
                on_group[
                    "program_key"
                ].nunique()
            ),
            "n_potential_gained_programs": int(
                group[
                    "potential_gained_program"
                ].sum()
            ),
        })

    summary = pd.DataFrame(
        rows
    ).merge(
        pair_stats,
        on=[
            "directed_pair",
            "sender_cluster",
            "receiver_cluster",
        ],
        how="left",
        validate="one_to_one",
    )

    # Native-only pairs retain native significance in the on panel.
    # Pairs with sparse MLTD gains use their hierarchical Simes Q.
    summary[
        "on_gain_adjusted_p_value"
    ] = np.where(
        summary[
            "gain_pair_selected"
        ].fillna(False),
        summary[
            "pair_simes_adjusted_p_value"
        ],
        summary[
            "off_adjusted_p_value"
        ],
    )

    return summary


def aggregate_pathway_units(
    programs: pd.DataFrame,
    unit_stats: pd.DataFrame,
) -> pd.DataFrame:
    active_summary = (
        programs.groupby(
            "pathway_unit",
            as_index=False,
        )
        .agg(
            off_weight=(
                "score_off",
                lambda values: 0.0,
            ),
        )
    )
    # Rebuild explicitly to respect active masks.
    rows = []

    for pathway_unit, group in programs.groupby(
        "pathway_unit",
        sort=False,
    ):
        off_group = group[
            group[
                "off_active"
            ]
        ]
        on_group = group[
            group[
                "on_active"
            ]
        ]
        selected = group[
            group[
                "potential_gained_program"
            ]
        ]

        rows.append({
            "pathway_unit": pathway_unit,
            "off_weight": float(
                off_group[
                    "score_off"
                ].sum()
            ),
            "on_weight": float(
                on_group[
                    "score_on"
                ].sum()
            ),
            "selected_gain_weight_actual": float(
                selected[
                    "mltd_gain"
                ].sum()
            ),
            "n_selected_gained_programs_actual": int(
                selected[
                    "program_key"
                ].nunique()
            ),
        })

    active_summary = pd.DataFrame(
        rows
    )

    return unit_stats.merge(
        active_summary,
        on="pathway_unit",
        how="left",
        validate="one_to_one",
    )




# =============================================================================
# 10. PLOTTING
# =============================================================================

def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "font.size": 7.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 5.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def robust01(
    values: np.ndarray | pd.Series,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> np.ndarray:
    """
    Robustly map finite values to [0,1] using quantile clipping.
    """
    array = np.asarray(
        values,
        dtype=float,
    )
    output = np.zeros_like(
        array,
        dtype=float,
    )

    finite = np.isfinite(
        array
    )
    if not finite.any():
        return output

    finite_values = array[
        finite
    ]
    low = float(
        np.quantile(
            finite_values,
            lower_quantile,
        )
    )
    high = float(
        np.quantile(
            finite_values,
            upper_quantile,
        )
    )

    if high - low <= 1e-15:
        output[finite] = (
            0.5
            if high > 0
            else 0.0
        )
        return output

    clipped = np.clip(
        array[finite],
        low,
        high,
    )
    output[finite] = (
        clipped - low
    ) / (
        high - low
    )

    return np.clip(
        output,
        0.0,
        1.0,
    )


def direction_p_for_plot(
    pair_summary: pd.DataFrame,
    *,
    mode: str,
) -> pd.Series:
    """
    Return one evidence P value for each directed pair before combining the
    two directions into an unordered cluster pair.

    Off:
        native empirical pair P.

    On:
        hierarchical pair Simes P for MLTD-gained directions;
        native empirical P for retained native-only directions.
    """
    if mode == "off":
        return pd.to_numeric(
            pair_summary[
                "off_p_value"
            ],
            errors="coerce",
        ).fillna(1.0)

    if mode == "on":
        gain_selected = (
            pair_summary[
                "gain_pair_selected"
            ]
            .fillna(False)
            .astype(bool)
        )
        gain_p = pd.to_numeric(
            pair_summary[
                "pair_simes_p_value"
            ],
            errors="coerce",
        ).fillna(1.0)
        native_p = pd.to_numeric(
            pair_summary[
                "off_p_value"
            ],
            errors="coerce",
        ).fillna(1.0)

        return pd.Series(
            np.where(
                gain_selected,
                gain_p,
                native_p,
            ),
            index=pair_summary.index,
            dtype=float,
        )

    raise ValueError(mode)


def unordered_pair_table(
    pair_summary: pd.DataFrame,
    clusters: list[str],
    *,
    mode: str,
) -> pd.DataFrame:
    """
    Collapse both communication directions into one upper-triangle bubble.

    Example:
        C0→C3
        C3→C0
    become:
        C0–C3

    Matrix weight:
        sum of active directional weights.

    Pair P:
        Simes combination of the active directional P values.

    Direction is not discarded from the analysis tables. It remains in the
    directed-pair summary and the Top pathway panel.
    """
    order = {
        cluster: index
        for index, cluster
        in enumerate(clusters)
    }

    if mode == "off":
        weight_column = (
            "off_weight"
        )
    elif mode == "on":
        weight_column = (
            "on_weight"
        )
    else:
        raise ValueError(mode)

    table = pair_summary[[
        "directed_pair",
        "sender_cluster",
        "receiver_cluster",
        weight_column,
    ]].copy()
    table["direction_p_value"] = (
        direction_p_for_plot(
            pair_summary,
            mode=mode,
        )
    )
    table = table.rename(columns={
        weight_column: "direction_weight",
    })

    table["direction_weight"] = (
        pd.to_numeric(
            table[
                "direction_weight"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )
    table["direction_p_value"] = (
        pd.to_numeric(
            table[
                "direction_p_value"
            ],
            errors="coerce",
        )
        .fillna(1.0)
        .clip(0.0, 1.0)
    )

    left_clusters = []
    right_clusters = []
    matrix_rows = []
    matrix_columns = []

    for sender, receiver in zip(
        table["sender_cluster"],
        table["receiver_cluster"],
    ):
        sender_index = order[
            sender
        ]
        receiver_index = order[
            receiver
        ]

        low_index = min(
            sender_index,
            receiver_index,
        )
        high_index = max(
            sender_index,
            receiver_index,
        )

        left_clusters.append(
            clusters[
                low_index
            ]
        )
        right_clusters.append(
            clusters[
                high_index
            ]
        )
        matrix_rows.append(
            low_index
        )
        matrix_columns.append(
            high_index
        )

    table["pair_left"] = (
        left_clusters
    )
    table["pair_right"] = (
        right_clusters
    )
    table["unordered_pair"] = (
        table["pair_left"]
        + "–"
        + table["pair_right"]
    )
    table["matrix_row"] = (
        matrix_rows
    )
    table["matrix_column"] = (
        matrix_columns
    )

    # Only active directions contribute to the matrix.
    active = table[
        table[
            "direction_weight"
        ] > 1e-12
    ].copy()

    rows = []

    for unordered_pair, group in active.groupby(
        "unordered_pair",
        sort=False,
    ):
        p_values = group[
            "direction_p_value"
        ].to_numpy(float)

        rows.append({
            "unordered_pair": (
                unordered_pair
            ),
            "pair_left": (
                group.iloc[0][
                    "pair_left"
                ]
            ),
            "pair_right": (
                group.iloc[0][
                    "pair_right"
                ]
            ),
            "matrix_row": int(
                group.iloc[0][
                    "matrix_row"
                ]
            ),
            "matrix_column": int(
                group.iloc[0][
                    "matrix_column"
                ]
            ),
            "weight": float(
                group[
                    "direction_weight"
                ].sum()
            ),
            "combined_p_value": (
                simes_p_value(
                    p_values
                )
            ),
            "n_active_directions": int(
                len(group)
            ),
            "active_directions": "; ".join(
                group[
                    "directed_pair"
                ]
                .astype(str)
                .tolist()
            ),
        })

    output = pd.DataFrame(
        rows
    )

    if len(output) == 0:
        return pd.DataFrame({
            "unordered_pair": [],
            "pair_left": [],
            "pair_right": [],
            "matrix_row": [],
            "matrix_column": [],
            "weight": [],
            "combined_p_value": [],
            "n_active_directions": [],
            "active_directions": [],
        })

    if output.duplicated(
        "unordered_pair"
    ).any():
        raise AssertionError(
            "Unordered-pair plotting table contains duplicate cells."
        )

    return output.sort_values(
        [
            "matrix_row",
            "matrix_column",
        ]
    ).reset_index(
        drop=True
    )


def add_color_scores(
    off_table: pd.DataFrame,
    on_table: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Add a continuous significance-effect color score.

    Pure empirical P values are discrete at the permutation floor. For example,
    with 1,000 permutations, many supported results have P=0.000999 and would
    all receive the same color.

    The displayed score is explicitly:
        -log10(combined directional Simes P)
        + COLOR_EFFECT_BONUS × robust normalized log1p(pair weight)

    It is not labelled as a P value.
    """
    tables = []

    for mode, table in [
        ("off", off_table),
        ("on", on_table),
    ]:
        copied = table.copy()
        copied[
            "_mode"
        ] = mode
        tables.append(
            copied
        )

    combined = pd.concat(
        tables,
        ignore_index=True,
        sort=False,
    )

    if len(combined) == 0:
        return (
            off_table.assign(
                color_score=[]
            ),
            on_table.assign(
                color_score=[]
            ),
        )

    log_weight = np.log1p(
        np.clip(
            combined[
                "weight"
            ].to_numpy(float),
            0.0,
            None,
        )
    )
    effect01 = robust01(
        log_weight,
        lower_quantile=(
            COLOR_EFFECT_LOWER_QUANTILE
        ),
        upper_quantile=(
            COLOR_EFFECT_UPPER_QUANTILE
        ),
    )
    significance = (
        -np.log10(
            np.clip(
                combined[
                    "combined_p_value"
                ].to_numpy(float),
                1e-300,
                1.0,
            )
        )
    )

    combined[
        "significance_component"
    ] = significance
    combined[
        "effect_component01"
    ] = effect01
    combined[
        "color_score_raw"
    ] = np.clip(
        significance
        + (
            COLOR_EFFECT_BONUS
            * effect01
        ),
        0.0,
        COLOR_SCORE_CAP,
    )

    # Visualization-only normalization. It is intentionally performed after
    # the analysis and independently within MLTD-off and MLTD-on.
    combined[
        "color_display01"
    ] = 0.5

    for mode_name, group_indices in (
        combined.groupby(
            "_mode",
            sort=False,
        ).groups.items()
    ):
        group_indices = np.asarray(
            list(group_indices),
            dtype=int,
        )
        mode_values = combined.loc[
            group_indices,
            "color_score_raw",
        ].to_numpy(float)

        combined.loc[
            group_indices,
            "color_display01",
        ] = robust01(
            mode_values,
            lower_quantile=(
                PANEL_COLOR_LOWER_QUANTILE
            ),
            upper_quantile=(
                PANEL_COLOR_UPPER_QUANTILE
            ),
        )

    off_output = (
        combined[
            combined[
                "_mode"
            ] == "off"
        ]
        .drop(
            columns=[
                "_mode",
            ]
        )
        .reset_index(
            drop=True
        )
    )
    on_output = (
        combined[
            combined[
                "_mode"
            ] == "on"
        ]
        .drop(
            columns=[
                "_mode",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        off_output,
        on_output,
    )


def build_plot_tables(
    pair_summary: pd.DataFrame,
    clusters: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    off_table = (
        unordered_pair_table(
            pair_summary,
            clusters,
            mode="off",
        )
    )
    on_table = (
        unordered_pair_table(
            pair_summary,
            clusters,
            mode="on",
        )
    )

    return add_color_scores(
        off_table,
        on_table,
    )


def build_bubble_scale(
    off_table: pd.DataFrame,
    on_table: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build one common robust size scale from unordered-pair weights.
    """
    arrays = []

    for table in [
        off_table,
        on_table,
    ]:
        if len(table):
            arrays.append(
                pd.to_numeric(
                    table[
                        "weight"
                    ],
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(float)
            )

    if arrays:
        values = np.concatenate(
            arrays
        )
    else:
        values = np.asarray(
            [],
            dtype=float,
        )

    positive = values[
        np.isfinite(
            values
        )
        & (
            values > 0
        )
    ]

    if len(positive) == 0:
        return {
            "cap": 1.0,
            "legend_values": (
                np.asarray(
                    [
                        0.25,
                        0.50,
                        1.00,
                    ],
                    dtype=float,
                )
            ),
        }

    cap = float(
        np.quantile(
            positive,
            BUBBLE_WEIGHT_CAP_QUANTILE,
        )
    )
    cap = max(
        cap,
        float(
            np.min(
                positive
            )
        ),
        1e-12,
    )

    legend_values = np.quantile(
        positive,
        BUBBLE_WEIGHT_LEGEND_QUANTILES,
    )
    legend_values = np.clip(
        legend_values,
        1e-12,
        cap,
    )

    cleaned = []

    for value in (
        legend_values
    ):
        value = float(
            value
        )

        if (
            not cleaned
            or not np.isclose(
                value,
                cleaned[-1],
                rtol=0.02,
                atol=1e-12,
            )
        ):
            cleaned.append(
                value
            )

    if len(cleaned) < 3:
        cleaned = [
            cap * 0.20,
            cap * 0.50,
            cap,
        ]

    return {
        "cap": cap,
        "legend_values": (
            np.asarray(
                cleaned[-3:],
                dtype=float,
            )
        ),
    }


def size_scale(
    values: np.ndarray | pd.Series,
    scale: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = np.clip(
        values,
        0.0,
        None,
    )

    cap = max(
        float(
            scale[
                "cap"
            ]
        ),
        1e-12,
    )
    clipped = np.minimum(
        values,
        cap,
    )

    denominator = np.log1p(
        cap
    )

    if denominator <= 1e-15:
        ratio = np.zeros_like(
            clipped
        )
    else:
        ratio = (
            np.log1p(
                clipped
            )
            / denominator
        )

    ratio = np.power(
        np.clip(
            ratio,
            0.0,
            1.0,
        ),
        BUBBLE_LOG_POWER,
    )

    return (
        BUBBLE_SIZE_MIN
        + (
            BUBBLE_SIZE_MAX
            - BUBBLE_SIZE_MIN
        )
        * ratio
    )


def draw_triangle(
    ax,
    table: pd.DataFrame,
    clusters: list[str],
    *,
    title: str,
    bubble_scale: dict[str, Any],
    cmap,
    norm: Normalize,
):
    """
    Draw one centered bubble per unordered cluster-pair cell.
    """
    n_clusters = len(
        clusters
    )

    for row in range(
        n_clusters
    ):
        for column in range(
            row,
            n_clusters,
        ):
            ax.add_patch(
                Rectangle(
                    (
                        column - 0.5,
                        row - 0.5,
                    ),
                    1.0,
                    1.0,
                    facecolor="white",
                    edgecolor=GRID_COLOR,
                    linewidth=0.62,
                    zorder=0,
                )
            )

    scatter = ax.scatter(
        table[
            "matrix_column"
        ],
        table[
            "matrix_row"
        ],
        s=size_scale(
            table[
                "weight"
            ],
            bubble_scale,
        ),
        c=table[
            "color_display01"
        ],
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.78,
        alpha=0.96,
        zorder=3,
        clip_on=True,
    )

    for row in range(
        n_clusters
    ):
        ax.plot(
            [
                row - 0.5,
                n_clusters - 0.5,
            ],
            [
                row - 0.5,
                row - 0.5,
            ],
            color=EDGE_COLOR,
            linewidth=0.72,
            zorder=1,
        )
        ax.plot(
            [
                row - 0.5,
                row - 0.5,
            ],
            [
                row - 0.5,
                row + 0.5,
            ],
            color=EDGE_COLOR,
            linewidth=0.72,
            zorder=1,
        )

    ax.set_xlim(
        -0.72,
        n_clusters - 0.28,
    )
    ax.set_ylim(
        n_clusters - 0.28,
        -0.72,
    )
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xticks(
        np.arange(
            n_clusters
        )
    )
    ax.set_xticklabels(
        clusters
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position(
        "top"
    )
    ax.set_xlabel(
        "Paired cluster",
        labelpad=8,
    )

    ax.set_yticks(
        np.arange(
            n_clusters
        )
    )
    ax.set_yticklabels(
        clusters
    )
    ax.set_ylabel(
        "Paired cluster"
    )

    ax.set_title(
        title,
        fontweight="bold",
        fontsize=9.0,
        pad=15,
    )
    ax.tick_params(
        length=0,
        pad=3,
    )

    for spine in (
        ax.spines.values()
    ):
        spine.set_visible(
            False
        )

    ax.text(
        0.01,
        -0.064,
        (
            "one bubble = both directions "
            "within one cluster pair"
        ),
        transform=ax.transAxes,
        fontsize=5.1,
        color=MUTED_TEXT,
        ha="left",
        va="top",
    )

    return scatter


def draw_size_legend(
    ax,
    bubble_scale: dict[str, Any],
) -> None:
    ax.axis(
        "off"
    )

    values = np.asarray(
        bubble_scale[
            "legend_values"
        ],
        dtype=float,
    )
    handles = []

    for value in values:
        area = float(
            size_scale(
                np.asarray(
                    [
                        value,
                    ]
                ),
                bubble_scale,
            )[0]
        )
        handles.append(
            Line2D(
                [
                    0,
                ],
                [
                    0,
                ],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=MUTED_TEXT,
                markeredgewidth=0.70,
                markersize=math.sqrt(
                    area
                ),
                label=f"{value:.3g}",
            )
        )

    legend = ax.legend(
        handles=handles,
        title=(
            "Bidirectional cluster-pair weight "
            "(95th-percentile capped)"
        ),
        frameon=False,
        loc="center",
        ncol=len(
            handles
        ),
        handletextpad=0.55,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    plt.setp(
        legend.get_title(),
        fontsize=5.8,
    )


def compact_label(
    value: Any,
    *,
    width: int,
    max_lines: int,
) -> str:
    text_value = re.sub(
        r"\s+",
        " ",
        str(
            value
        ).strip(),
    )
    lines = textwrap.wrap(
        text_value,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    if len(
        lines
    ) > max_lines:
        lines = lines[
            :max_lines
        ]
        lines[-1] = (
            lines[-1][:-1]
            + "…"
            if len(
                lines[-1]
            ) > 1
            else "…"
        )

    return "\n".join(
        lines
    )


def draw_top_pathways(
    ax,
    top: pd.DataFrame,
) -> None:
    ax.axis(
        "off"
    )
    ax.set_xlim(
        0.0,
        1.0,
    )
    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.text(
        0.00,
        0.985,
        (
            f"Top {len(top)} unique "
            "MLTD-gained pathways"
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    if len(
        top
    ) == 0:
        ax.text(
            0.00,
            0.88,
            (
                "No pathway passed the sparse "
                "hierarchical evidence rule."
            ),
            fontsize=5.8,
            color=MUTED_TEXT,
        )
        return

    gain_column = (
        "selected_gain_weight_actual"
        if (
            "selected_gain_weight_actual"
            in top.columns
        )
        else "selected_gain_weight"
    )
    gain_values = pd.to_numeric(
        top[
            gain_column
        ],
        errors="coerce",
    ).fillna(
        0.0
    )
    maximum = max(
        float(
            gain_values.max()
        ),
        1e-12,
    )

    y_positions = np.linspace(
        0.885,
        0.075,
        len(
            top
        ),
    )

    for y, (_, row) in zip(
        y_positions,
        top.iterrows(),
    ):
        gain = finite_float(
            row.get(
                gain_column,
                0.0,
            )
        )
        fraction = np.clip(
            gain / maximum,
            0.0,
            1.0,
        )

        pathway = compact_label(
            row.get(
                "pathway",
                "",
            ),
            width=31,
            max_lines=2,
        )
        ligand = str(
            row.get(
                "representative_ligand",
                "",
            )
        )
        receptor = str(
            row.get(
                "representative_receptor",
                "",
            )
        )
        pair = str(
            row.get(
                "directed_pair",
                "",
            )
        )
        simes_p = finite_float(
            row.get(
                "unit_simes_p_value",
                1.0,
            ),
            default=1.0,
        )
        simes_q = finite_float(
            row.get(
                "unit_simes_adjusted_p_value",
                1.0,
            ),
            default=1.0,
        )
        tier = str(
            row.get(
                "unit_evidence_tier",
                "",
            )
        )

        ax.text(
            0.00,
            y + 0.006,
            str(
                int(
                    row[
                        "rank"
                    ]
                )
            ),
            fontsize=6.7,
            fontweight="bold",
            va="center",
        )
        ax.text(
            0.055,
            y + 0.017,
            pathway,
            fontsize=5.55,
            fontweight="semibold",
            va="center",
            ha="left",
            linespacing=1.05,
            color=TEXT_COLOR,
        )

        detail = (
            f"gain={gain:.3g}; "
            f"P={simes_p:.2g}; "
            f"Q={simes_q:.2g}; "
            f"{tier}"
        )
        ax.text(
            0.055,
            y - 0.028,
            compact_label(
                detail,
                width=48,
                max_lines=2,
            ),
            fontsize=4.45,
            color=MUTED_TEXT,
            va="center",
            ha="left",
            linespacing=1.05,
        )

        ax.add_patch(
            Rectangle(
                (
                    0.625,
                    y - 0.014,
                ),
                0.105,
                0.028,
                facecolor="#F1F3F4",
                edgecolor="none",
            )
        )
        ax.add_patch(
            Rectangle(
                (
                    0.625,
                    y - 0.014,
                ),
                0.105
                * fraction,
                0.028,
                facecolor=GAIN_COLOR,
                edgecolor="none",
                alpha=0.82,
            )
        )

        ax.add_patch(
            FancyBboxPatch(
                (
                    0.765,
                    y - 0.027,
                ),
                0.215,
                0.054,
                boxstyle=(
                    "round,pad=0.006,"
                    "rounding_size=0.010"
                ),
                facecolor="white",
                edgecolor=GAIN_COLOR,
                linewidth=0.85,
            )
        )
        ax.text(
            0.8725,
            y,
            pair,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=GAIN_COLOR,
        )


def save_figure(
    fig,
    stem: Path,
) -> None:
    for suffix in [
        "png",
        "svg",
        "pdf",
    ]:
        path = stem.with_suffix(
            f".{suffix}"
        )
        fig.savefig(
            path,
            dpi=(
                DPI
                if suffix
                == "png"
                else None
            ),
            bbox_inches="tight",
            pad_inches=0.06,
        )
        log(
            "[SAVE]",
            path,
        )

    plt.close(
        fig
    )


def make_standalone(
    table: pd.DataFrame,
    top: pd.DataFrame,
    clusters: list[str],
    *,
    title: str,
    bubble_scale: dict[str, Any],
    cmap,
    norm: Normalize,
    output_stem: Path,
) -> None:
    fig = plt.figure(
        figsize=STANDALONE_FIGSIZE
    )
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[
            1.0,
            0.075,
            0.045,
        ],
        width_ratios=[
            1.65,
            1.05,
        ],
        left=0.065,
        right=0.985,
        bottom=0.085,
        top=0.925,
        hspace=0.20,
        wspace=0.17,
    )

    matrix_ax = fig.add_subplot(
        grid[
            0,
            0,
        ]
    )
    top_ax = fig.add_subplot(
        grid[
            :,
            1,
        ]
    )
    size_ax = fig.add_subplot(
        grid[
            1,
            0,
        ]
    )
    color_ax = fig.add_subplot(
        grid[
            2,
            0,
        ]
    )

    scatter = draw_triangle(
        matrix_ax,
        table,
        clusters,
        title=title,
        bubble_scale=bubble_scale,
        cmap=cmap,
        norm=norm,
    )
    draw_size_legend(
        size_ax,
        bubble_scale,
    )
    draw_top_pathways(
        top_ax,
        top,
    )

    colorbar = fig.colorbar(
        scatter,
        cax=color_ax,
        orientation="horizontal",
    )
    colorbar.set_label(
        (
            "Within-panel relative significance–effect score"
        ),
        fontsize=6.2,
    )
    colorbar.ax.tick_params(
        labelsize=5.6,
        length=2,
        width=0.5,
    )

    save_figure(
        fig,
        output_stem,
    )


def make_combined(
    off_table: pd.DataFrame,
    on_table: pd.DataFrame,
    top: pd.DataFrame,
    clusters: list[str],
    *,
    bubble_scale: dict[str, Any],
    cmap,
    norm: Normalize,
) -> None:
    fig = plt.figure(
        figsize=COMBINED_FIGSIZE
    )
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[
            1.0,
            0.070,
            0.043,
        ],
        width_ratios=[
            1.0,
            1.0,
            1.00,
        ],
        left=0.040,
        right=0.987,
        bottom=0.080,
        top=0.925,
        hspace=0.22,
        wspace=0.17,
    )

    off_ax = fig.add_subplot(
        grid[
            0,
            0,
        ]
    )
    on_ax = fig.add_subplot(
        grid[
            0,
            1,
        ]
    )
    top_ax = fig.add_subplot(
        grid[
            :,
            2,
        ]
    )
    size_ax = fig.add_subplot(
        grid[
            1,
            0:2,
        ]
    )
    off_color_ax = fig.add_subplot(
        grid[
            2,
            0,
        ]
    )
    on_color_ax = fig.add_subplot(
        grid[
            2,
            1,
        ]
    )

    off_scatter = draw_triangle(
        off_ax,
        off_table,
        clusters,
        title=(
            "MLTD-off: observed "
            "high-confidence CCI"
        ),
        bubble_scale=bubble_scale,
        cmap=cmap,
        norm=norm,
    )
    on_scatter = draw_triangle(
        on_ax,
        on_table,
        clusters,
        title=(
            "MLTD-on: RNA–TF–RE-gained "
            "potential CCI"
        ),
        bubble_scale=bubble_scale,
        cmap=cmap,
        norm=norm,
    )

    draw_size_legend(
        size_ax,
        bubble_scale,
    )
    draw_top_pathways(
        top_ax,
        top,
    )

    for color_ax, scatter, mode_label in [
        (
            off_color_ax,
            off_scatter,
            "MLTD-off",
        ),
        (
            on_color_ax,
            on_scatter,
            "MLTD-on",
        ),
    ]:
        colorbar = fig.colorbar(
            scatter,
            cax=color_ax,
            orientation="horizontal",
        )
        colorbar.set_label(
            (
                f"{mode_label} within-panel relative "
                "significance–effect score"
            ),
            fontsize=5.9,
        )
        colorbar.ax.tick_params(
            labelsize=5.5,
            length=2,
        )

    save_figure(
        fig,
        OUT_COMBINED_STEM,
    )


# =============================================================================
# 11. MAIN
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cci_path = resolve_path(
        ALL_CCI_MANUAL,
        ALL_CCI_CANDIDATES,
        root=RUN_BASE,
        filename="interactions_cluster_level.csv",
    )
    native_cci_path = resolve_path(
        NATIVE_CCI_MANUAL,
        NATIVE_CCI_CANDIDATES,
        root=RUN_BASE,
        filename="interactions_cluster_level_highconf_TFkept.csv",
    )
    gene_tf_path = resolve_path(
        GENE_TF_MANUAL,
        GENE_TF_CANDIDATES,
        root=RUN_BASE,
        filename=f"{SAMPLE_ID}_gene_topk_tf_edges.csv",
    )
    receptor_tf_path = resolve_path(
        RECEPTOR_TF_MANUAL,
        RECEPTOR_TF_CANDIDATES,
        root=RUN_BASE,
        filename="receptor_tf_signature.csv",
    )
    hybrid_tf_path = resolve_path(
        HYBRID_TF_MANUAL,
        HYBRID_TF_CANDIDATES,
        root=RUN_BASE,
        filename="dotplot_hybridTF_table.csv",
    )

    tf_re_component_path = optional_path(
        TF_RE_MANUAL,
        TF_RE_COMPONENT_CANDIDATES,
        root=RUN_BASE,
        filename=f"{SAMPLE_ID}_TF_RE_intrinsic_components.csv",
    )
    tf_re_peak_path = None
    tf_re_arc_path = None

    if tf_re_component_path is None:
        tf_re_peak_path = resolve_path(
            TF_RE_PEAKS_MANUAL,
            TF_RE_PEAK_CANDIDATES,
            root=RUN_BASE,
            filename=f"{SAMPLE_ID}_TF_linked_RE_peaks_v1.csv",
        )
        tf_re_arc_path = optional_path(
            TF_RE_ARCS_MANUAL,
            TF_RE_ARC_CANDIDATES,
            root=RUN_BASE,
            filename=f"{SAMPLE_ID}_TF_linked_RE_peak_arcs_v1.csv",
        )

    log("[INPUT all CCI]", all_cci_path)
    log("[INPUT native CCI]", native_cci_path)
    log("[INPUT all-gene RNA-TF]", gene_tf_path)
    log("[INPUT TF-RE]", tf_re_component_path or tf_re_peak_path)
    log("[INPUT receptor-TF]", receptor_tf_path)
    log("[INPUT hybrid TF]", hybrid_tf_path)

    all_cci = load_cci(all_cci_path)
    native_cci = load_cci(native_cci_path)
    (
        gene_tf,
        gene_tf_qc,
    ) = load_gene_tf(
        gene_tf_path
    )
    tf_re = load_tf_re(
        tf_re_component_path,
        tf_re_peak_path,
        tf_re_arc_path,
    )

    positive_rna_tf_gain = int(
        (
            gene_tf["rna_tf_gain"]
            > 1e-12
        ).sum()
    )
    log(
        "[RNA-TF GAIN]",
        positive_rna_tf_gain,
        "/",
        len(gene_tf),
        "gene-TF edges have positive full-MLTD evidence gain",
    )

    if positive_rna_tf_gain == 0:
        OUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        gene_tf.to_csv(
            OUT_GENE_TF,
            index=False,
        )
        gene_tf_qc.to_csv(
            OUT_GENE_TF_QC,
            index=False,
        )
        raise RuntimeError(
            "No RNA-TF edge has positive MLTD-specific evidence after "
            "component-aware scoring. The script stopped before communication "
            "analysis. Inspect the source columns and component coverage in "
            f"{OUT_GENE_TF_QC}."
        )
    (
        receptor_tf,
        receptor_tf_qc,
    ) = load_receptor_tf(
        receptor_tf_path
    )
    observed_cluster_tf = load_cluster_tf(hybrid_tf_path)

    universe = prepare_universe(all_cci, native_cci)

    if not INCLUDE_SELF_INTERACTIONS:
        universe = universe[
            universe["sender_cluster"] != universe["receiver_cluster"]
        ].copy()

    clusters = get_cluster_order(universe)
    tf_keys = sorted(
        set(gene_tf["tf_key"]) & set(tf_re["tf_key"])
    )
    cluster_tf = complete_cluster_tf_activity(
        observed_cluster_tf,
        clusters,
        tf_keys,
    )

    sender_expression = cluster_gene_expression(universe, side="sender")
    receiver_expression = cluster_gene_expression(universe, side="receiver")

    sender_bridge = build_sender_bridge(
        sender_expression,
        gene_tf,
        tf_re,
        cluster_tf,
    )
    receiver_bridge = build_receiver_bridge(
        receiver_expression,
        gene_tf,
        tf_re,
        receptor_tf,
        cluster_tf,
    )

    programs = attach_program_scores(
        universe,
        sender_bridge,
        receiver_bridge,
    )
    programs = add_pair_columns(programs, clusters)

    (
        program_stats,
        native_pair_stats,
        null_summary,
    ) = empirical_nulls(
        programs
    )

    (
        programs,
        pair_stats,
        unit_stats,
    ) = build_sparse_gain_hierarchy(
        programs,
        program_stats,
        native_pair_stats,
    )
    fail_if_no_useful_gain(
        programs,
        pair_stats,
        unit_stats,
        OUT_DIAGNOSTIC,
    )

    pair_summary = aggregate_directed_pairs(programs, pair_stats)
    unit_summary = aggregate_pathway_units(programs, unit_stats)

    # Post-hoc display aggregation only. This occurs after pair/program
    # selection and therefore cannot change triangular-map topology.
    kegg_display_summary = build_kegg_display_pathway_summary(
        unit_summary
    )
    top10 = select_top_unique_pathways(
        kegg_display_summary
    )

    # Invariants after active-set selection.
    off_pairs = set(
        pair_summary.loc[pair_summary["off_weight"] > 0, "directed_pair"]
    )
    on_pairs = set(
        pair_summary.loc[pair_summary["on_weight"] > 0, "directed_pair"]
    )

    if not off_pairs.issubset(on_pairs):
        raise AssertionError("Active-set invariant violated: off pair missing in on.")

    new_pairs = on_pairs - off_pairs

    # Save tables.
    gene_tf.to_csv(
        OUT_GENE_TF,
        index=False,
    )
    gene_tf_qc.to_csv(
        OUT_GENE_TF_QC,
        index=False,
    )
    tf_re.to_csv(OUT_TF_RE, index=False)
    receptor_tf.to_csv(
        OUT_RECEPTOR_TF,
        index=False,
    )
    receptor_tf_qc.to_csv(
        OUT_RECEPTOR_TF_QC,
        index=False,
    )
    cluster_tf.to_csv(OUT_CLUSTER_TF, index=False)
    sender_bridge.to_csv(OUT_SENDER, index=False)
    receiver_bridge.to_csv(OUT_RECEIVER, index=False)
    programs.to_csv(OUT_PROGRAMS, index=False)
    programs[programs["potential_gained_program"]].to_csv(
        OUT_GAINED,
        index=False,
    )
    pair_summary.to_csv(OUT_PAIR, index=False)
    unit_summary.to_csv(OUT_UNIT, index=False)
    kegg_display_summary.to_csv(
        OUT_KEGG_DISPLAY,
        index=False,
    )
    top10.to_csv(OUT_TOP10, index=False)
    null_summary.to_csv(OUT_NULL, index=False)

    for path in [
        OUT_GENE_TF,
        OUT_GENE_TF_QC,
        OUT_TF_RE,
        OUT_RECEPTOR_TF,
        OUT_RECEPTOR_TF_QC,
        OUT_CLUSTER_TF,
        OUT_SENDER,
        OUT_RECEIVER,
        OUT_PROGRAMS,
        OUT_GAINED,
        OUT_PAIR,
        OUT_UNIT,
        OUT_KEGG_DISPLAY,
        OUT_TOP10,
        OUT_NULL,
    ]:
        log("[SAVE]", path)

    qc = pd.DataFrame([
        {
            "metric": "candidate_programs_same_universe",
            "value": int(len(programs)),
        },
        {
            "metric": "native_off_programs",
            "value": int(programs["native_off"].sum()),
        },
        {
            "metric": "positive_RNA_TF_gain_edges",
            "value": int(
                (
                    gene_tf[
                        "rna_tf_gain"
                    ] > 1e-12
                ).sum()
            ),
        },
        {
            "metric": "positive_MLTD_gain_programs",
            "value": int((programs["mltd_gain"] > 0).sum()),
        },
        {
            "metric": "potential_gained_programs",
            "value": int(programs["potential_gained_program"].sum()),
        },
        {
            "metric": "selected_gain_directed_pairs",
            "value": int(
                pair_stats[
                    "gain_pair_selected"
                ].sum()
            ),
        },
        {
            "metric": "selected_off_directed_pairs",
            "value": int(
                pair_stats[
                    "off_pair_selected"
                ].sum()
            ),
        },
        {
            "metric": "confirmed_sparse_pathway_units",
            "value": int(
                unit_stats[
                    "confirmed_gain_unit"
                ].sum()
            ),
        },
        {
            "metric": "potential_sparse_pathway_units",
            "value": int(
                unit_stats[
                    "potential_gain_unit"
                ].sum()
            ),
        },
        {
            "metric": "programs_capped_per_pathway_unit",
            "value": int(
                MAX_GAINED_PROGRAMS_PER_UNIT
            ),
        },
        {
            "metric": "off_active_programs",
            "value": int(programs["off_active"].sum()),
        },
        {
            "metric": "on_active_programs",
            "value": int(programs["on_active"].sum()),
        },
        {
            "metric": "off_nonzero_directed_pairs",
            "value": int(len(off_pairs)),
        },
        {
            "metric": "on_nonzero_directed_pairs",
            "value": int(len(on_pairs)),
        },
        {
            "metric": "new_directed_pairs_in_on",
            "value": int(len(new_pairs)),
        },
        {
            "metric": "on_score_below_off_violations",
            "value": int(
                (programs["score_on"] < programs["score_off"] - 1e-12).sum()
            ),
        },
        {
            "metric": "unique_top_pathways",
            "value": int(top10["pathway"].nunique()),
        },
        {
            "metric": "top_pathway_rows",
            "value": int(len(top10)),
        },
        {
            "metric": "sender_activity_backfill_fraction",
            "value": float(
                programs["sender_activity_source"]
                .eq("global_shrunk_backfill")
                .mean()
            ),
        },
        {
            "metric": "receiver_activity_backfill_fraction",
            "value": float(
                programs["receiver_activity_source"]
                .eq("global_shrunk_backfill")
                .mean()
            ),
        },
        {
            "metric": "receiver_identity_fallback_fraction",
            "value": float(
                programs["receiver_identity_source"]
                .eq("penalized_gene_tf_fallback")
                .mean()
            ),
        },
        {
            "metric": "duplicated_receptor_TF_keys",
            "value": int(
                receptor_tf[
                    "is_duplicated_key"
                ].sum()
            ),
        },
        {
            "metric": "receptor_TF_rows_collapsed",
            "value": int(
                receptor_tf[
                    "n_signature_records"
                ].sum()
                - len(receptor_tf)
            ),
        },
    ])
    qc.to_csv(OUT_QC, index=False)
    log("[SAVE]", OUT_QC)

    (
        off_plot_table,
        on_plot_table,
    ) = build_plot_tables(
        pair_summary,
        clusters,
    )
    bubble_scale = build_bubble_scale(
        off_plot_table,
        on_plot_table,
    )

    off_plot_table.to_csv(
        OUT_DIR
        / f"{SAMPLE_ID}_MLTD_off_unordered_pair_plot_table.csv",
        index=False,
    )
    on_plot_table.to_csv(
        OUT_DIR
        / f"{SAMPLE_ID}_MLTD_on_unordered_pair_plot_table.csv",
        index=False,
    )

    log(
        "[PLOT TABLE]",
        "off unordered pairs=",
        len(off_plot_table),
        "| on unordered pairs=",
        len(on_plot_table),
    )
    log(
        "[PLOT SCALE]",
        "weight cap=",
        bubble_scale["cap"],
        "| legend=",
        bubble_scale["legend_values"].tolist(),
    )
    log(
        "[PLOT COLOR]",
        "off score range=",
        (
            float(
                off_plot_table[
                    "color_display01"
                ].min()
            )
            if len(
                off_plot_table
            )
            else np.nan
        ),
        "to",
        (
            float(
                off_plot_table[
                    "color_display01"
                ].max()
            )
            if len(
                off_plot_table
            )
            else np.nan
        ),
        "| on score range=",
        (
            float(
                on_plot_table[
                    "color_display01"
                ].min()
            )
            if len(
                on_plot_table
            )
            else np.nan
        ),
        "to",
        (
            float(
                on_plot_table[
                    "color_display01"
                ].max()
            )
            if len(
                on_plot_table
            )
            else np.nan
        ),
    )

    cmap = LinearSegmentedColormap.from_list(
        "MLTD_blue_white_orange",
        [
            "#2C7BB6",
            "#F7F7F7",
            "#F46D43",
        ],
        N=256,
    )
    norm = Normalize(
        vmin=PANEL_COLOR_MIN,
        vmax=PANEL_COLOR_MAX,
    )

    make_standalone(
        off_plot_table,
        top10,
        clusters,
        title=(
            "MLTD-off: observed "
            "high-confidence CCI"
        ),
        bubble_scale=bubble_scale,
        cmap=cmap,
        norm=norm,
        output_stem=OUT_OFF_STEM,
    )
    make_standalone(
        on_plot_table,
        top10,
        clusters,
        title=(
            "MLTD-on: RNA–TF–RE-gained "
            "potential CCI"
        ),
        bubble_scale=bubble_scale,
        cmap=cmap,
        norm=norm,
        output_stem=OUT_ON_STEM,
    )
    make_combined(
        off_plot_table,
        on_plot_table,
        top10,
        clusters,
        bubble_scale=bubble_scale,
        cmap=cmap,
        norm=norm,
    )

    manifest = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "same_candidate_universe": True,
        "off_definition": (
            "original high-confidence TF-kept CCI active set; score uses "
            "correlation-only RNA-TF bridge"
        ),
        "on_definition": (
            "off active set plus positive full-MLTD-versus-correlation bridge "
            "gains passing empirical potential-gain P threshold"
        ),
        "full_support_contains_correlation_support": True,
        "rna_tf_on_support_definition": (
            "noisy-OR of correlation support, full-score residual, "
            "latent/model representation evidence, and prior-assisted "
            "model evidence"
        ),
        "rna_tf_channel_reliabilities": {
            "full_residual": FULL_RESIDUAL_RELIABILITY,
            "latent": LATENT_CHANNEL_RELIABILITY,
            "prior_assisted": PRIOR_ASSISTED_RELIABILITY,
            "prior_assist_base": PRIOR_ASSIST_BASE,
        },
        "positive_rna_tf_gain_edges": int(
            (
                gene_tf[
                    "rna_tf_gain"
                ] > 1e-12
            ).sum()
        ),
        "absolute_bridge_hard_gate_removed": True,
        "sparse_gain_rule": {
            "selection_level": (
                "directed pair × pathway, followed by directed pair"
            ),
            "program_empirical_p_threshold_potential": (
                PROGRAM_P_THRESHOLD_POTENTIAL
            ),
            "program_empirical_p_threshold_confirmed": (
                PROGRAM_P_THRESHOLD_CONFIRMED
            ),
            "unit_simes_p_threshold": (
                UNIT_SIME_P_THRESHOLD
            ),
            "unit_confirmed_q_threshold": (
                UNIT_CONFIRMED_Q_THRESHOLD
            ),
            "maximum_gained_programs_per_unit": (
                MAX_GAINED_PROGRAMS_PER_UNIT
            ),
            "off_pair_p_threshold": (
                OFF_PAIR_P_THRESHOLD
            ),
            "off_pair_q_threshold": (
                OFF_PAIR_Q_THRESHOLD
            ),
            "require_new_directed_pair": (
                REQUIRE_NEW_DIRECTED_PAIR
            ),
            "reason": (
                "sparse identity-specific programs were diluted by total-gain "
                "aggregation over approximately 936k candidates"
            ),
        },
        "activity_backfill": {
            "reason": (
                "hybrid TF output is sparse selected-TF evidence; absence is "
                "not treated as biological zero"
            ),
            "global_shrinkage": TF_ACTIVITY_GLOBAL_BACKFILL,
            "minimum": TF_ACTIVITY_MIN_BACKFILL,
        },
        "receiver_identity_fallback_penalty": (
            RECEPTOR_TF_FALLBACK_PENALTY
        ),
        "receptor_tf_deduplication": {
            "merge_key": [
                "receptor_key",
                "tf_key",
            ],
            "aggregation": (
                "maximum source-row signature score; "
                "duplicate count retained for QC"
            ),
            "unique_keys": int(
                len(receptor_tf)
            ),
            "duplicated_keys": int(
                receptor_tf[
                    "is_duplicated_key"
                ].sum()
            ),
            "rows_collapsed": int(
                receptor_tf[
                    "n_signature_records"
                ].sum()
                - len(receptor_tf)
            ),
        },
        "n_candidate_programs": int(len(programs)),
        "n_native_programs": int(programs["native_off"].sum()),
        "n_positive_gain_programs": int((programs["mltd_gain"] > 0).sum()),
        "n_potential_gained_programs": int(
            programs["potential_gained_program"].sum()
        ),
        "n_selected_gain_pairs": int(
            pair_stats[
                "gain_pair_selected"
            ].sum()
        ),
        "n_selected_off_pairs": int(
            pair_stats[
                "off_pair_selected"
            ].sum()
        ),
        "n_confirmed_sparse_units": int(
            unit_stats[
                "confirmed_gain_unit"
            ].sum()
        ),
        "n_potential_sparse_units": int(
            unit_stats[
                "potential_gain_unit"
            ].sum()
        ),
        "n_off_pairs": int(len(off_pairs)),
        "n_on_pairs": int(len(on_pairs)),
        "new_on_pairs": sorted(new_pairs),
        "top_pathways_unique": bool(
            top10["pathway"].nunique() == len(top10)
        ),
        "n_fine_grained_selected_pathway_units": int(
            unit_summary[
                "potential_gain_unit"
            ].sum()
        ),
        "n_broad_KEGG_style_display_pathways": int(
            len(kegg_display_summary)
        ),
        "plotting": {
            "one_bubble_per_unordered_cluster_pair": True,
            "opposite_directions": (
                "summed in the triangular matrix; retained in directed "
                "analysis tables and Top pathway labels"
            ),
            "unordered_pair_p": (
                "Simes combination across active directional P values"
            ),
            "bubble_weight_transform": "log1p",
            "bubble_weight_cap_quantile": (
                BUBBLE_WEIGHT_CAP_QUANTILE
            ),
            "bubble_area_min": BUBBLE_SIZE_MIN,
            "bubble_area_max": BUBBLE_SIZE_MAX,
            "bubble_color_raw": (
                "-log10(unordered-pair Simes P) plus robust normalized "
                "log1p(weight) effect bonus"
            ),
            "bubble_color_display": (
                "raw significance-effect score normalized independently "
                "within each panel to a blue-white-orange gradient"
            ),
            "cross_panel_color_warning": (
                "display colors indicate relative structure within each "
                "panel; use color_score_raw, P and Q for absolute cross-panel "
                "comparison"
            ),
            "color_effect_bonus": (
                COLOR_EFFECT_BONUS
            ),
            "color_score_cap": (
                COLOR_SCORE_CAP
            ),
            "analysis_pathway_level": (
                "original V5.2 fine-grained LR/pathway units; used for all "
                "Simes testing and communication topology"
            ),
            "display_pathway_level": (
                "post-hoc rule-based KEGG-style broad labels; never fed back "
                "into program, pathway-unit or pair selection"
            ),
        },
        "permutations": N_PERMUTATIONS,
        "inputs": {
            "all_cci": str(all_cci_path),
            "native_cci": str(native_cci_path),
            "gene_tf": str(gene_tf_path),
            "tf_re": str(tf_re_component_path or tf_re_peak_path),
            "receptor_tf": str(receptor_tf_path),
            "hybrid_tf": str(hybrid_tf_path),
        },
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    log("[SAVE]", OUT_MANIFEST)

    log("\n[DONE]")
    log("  candidate programs:", len(programs))
    log("  native off programs:", int(programs["native_off"].sum()))
    log(
        "  positive gain programs:",
        int((programs["mltd_gain"] > 0).sum()),
    )
    log(
        "  potential gained programs:",
        int(programs["potential_gained_program"].sum()),
    )
    log("  off directed pairs:", len(off_pairs))
    log("  on directed pairs:", len(on_pairs))
    log("  new directed pairs:", len(new_pairs))
    log("  top unique pathways:", len(top10))
    log("  output:", OUT_DIR)


if __name__ == "__main__":
    main()
