from __future__ import annotations

# ============================================================
# compare_SOD2_MLTD_vs_traditional_pathway_recovery_full_standalone.py
# ============================================================
#
# Purpose
# -------
# Test whether conventional transcriptomic strategies can recover the same
# predefined SOD2-compatible biological programs highlighted by
# MLTD-defined dependency sensitivity.
#
# Strategies
# ----------
# 1. MLTD dependency response
#    - Uses the existing expression-matched MLTD-high versus MLTD-low
#      native-RNA ranking generated for Fig. 2h.
#
# 2. Native SOD2-expression stratification
#    - Within the same CCI clusters represented in the MLTD comparison,
#      selects SOD2-expression-high and SOD2-expression-low cells.
#    - Matches cells within sample and cluster for total UMI and detected
#      genes.
#    - Uses the same number and cluster composition of pairs whenever
#      sufficient cells are available.
#
# 3. Conventional SOD2 co-expression
#    - Computes gene-wise partial Spearman correlation with native SOD2
#      expression while controlling for sample, CCI cluster, total UMI and
#      detected genes.
#
# 4. Matched-label null
#    - Randomly swaps the high/low direction within the existing MLTD
#      expression-matched pairs.
#    - Generates a null distribution for an aggregate predefined-program
#      recovery index.
#
# Core fairness principle
# -----------------------
# All observed strategies use:
#   - the same native RNA matrices;
#   - the same common gene vocabulary;
#   - the same curated SOD2-compatible modules;
#   - the same local preranked-GSEA implementation and permutation count.
#
# SOD2 itself is excluded from enrichment and recovery scoring to avoid
# circularity.
#
# Main figure
# -----------
# A. Dot heatmap of module recovery across strategies
# B. Recovery-completeness summary
# C. Matched-label null distribution of aggregate program recovery
# D. Module-level MLTD advantage over the best traditional method
#
# Main outputs
# ------------
# SOD2_MLTD_vs_traditional_pathway_recovery.png/.pdf/.svg
# SOD2_pathway_recovery_all_methods_GSEA.csv
# SOD2_expression_stratified_pairs.csv
# SOD2_expression_stratified_matching_QC.csv
# SOD2_expression_stratified_native_RNA_DE.csv
# SOD2_partial_Spearman_coexpression_ranking.csv
# SOD2_matched_label_null_recovery_index.csv
# SOD2_pathway_recovery_method_summary.csv
# SOD2_pathway_recovery_module_advantage.csv
# SOD2_pathway_recovery_manifest.json
# ============================================================

from pathlib import Path
from typing import Any, Iterable
import json
import math
import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
except Exception as error:
    raise ImportError(
        "This script requires scikit-learn for expression-stratified cell matching."
    ) from error


# ============================================================
# 0) CONFIGURATION
# ============================================================

TARGET_GENE = "SOD2"
DATASET_NAME = "CRC-Tissue"

BASE_CANDIDATES = [get_project_root()]

BIOLOGICAL_OUTPUT_SUBDIRECTORIES = [
    "SOD2_MLTD_high_low_biological_concordance_v3_red_blue_pairwise_matched",
    "SOD2_MLTD_high_low_biological_concordance_v2_cluster_context_matched",
    "SOD2_MLTD_high_low_biological_concordance",
]

# Native-expression comparator.
EXPRESSION_HIGH_QUANTILE = 0.80
EXPRESSION_LOW_QUANTILE = 0.20
EXPRESSION_MATCH_NEIGHBORS = 64
EXPRESSION_MATCH_DISTANCE_CALIPER = 1.50
EXPRESSION_MATCH_WITHOUT_REPLACEMENT = True

# Native RNA.
NORMALIZE_TARGET_SUM = 1.0e4
GENE_BLOCK_SIZE = 256
MIN_EXPRESSED_CELL_FRACTION = 0.01
MIN_EXPRESSED_CELLS = 12

# Co-expression.
COEXPRESSION_GENE_BLOCK_SIZE = 192
COEXPRESSION_MAX_CELLS = None
COEXPRESSION_SEED = 20260721

# GSEA.
N_GSEA_PERMUTATIONS = 1000
GSEA_MIN_SET_SIZE = 5
GSEA_MAX_SET_SIZE = 250
GSEA_WEIGHT_POWER = 1.0

# Label-null recovery index.
N_MATCHED_LABEL_NULL = 200
N_METHOD_GENESET_NULL = 1000
METHOD_GENESET_NULL_SEED = 20260724
LABEL_NULL_SEED = 20260722

# Plotting.
DPI = 500
MAX_SIGNIFICANCE_SIZE = 320.0
MIN_SIGNIFICANCE_SIZE = 24.0

COLORS = {
    "MLTD dependency": "#B33F62",
    "SOD2 expression": "#D68A27",
    "SOD2 co-expression": "#35639A",
    "null": "#B8BDC5",
    "positive": "#B2182B",
    "negative": "#2166AC",
    "grid": "#E9E9E9",
    "text": "#222222",
    "panel": "#F7F7F7",
}

CURATED_SOD2_MODULES: dict[str, list[str]] = {'Mitochondrial ROS detoxification': ['SOD2',
                                      'PRDX3',
                                      'PRDX5',
                                      'TXN2',
                                      'TXNRD2',
                                      'GPX1',
                                      'GPX4',
                                      'GSR',
                                      'NNT',
                                      'IDH2',
                                      'ME3',
                                      'SIRT3',
                                      'FOXO3',
                                      'AIFM1',
                                      'GLRX2'],
 'NRF2 antioxidant response': ['NFE2L2',
                               'KEAP1',
                               'HMOX1',
                               'NQO1',
                               'GCLC',
                               'GCLM',
                               'SRXN1',
                               'TXNRD1',
                               'PRDX1',
                               'PRDX2',
                               'FTH1',
                               'FTL',
                               'SLC7A11',
                               'G6PD',
                               'PGD',
                               'TALDO1',
                               'GSTM1',
                               'GSTM3'],
 'Mitochondrial respiratory chain': ['NDUFA1',
                                     'NDUFA2',
                                     'NDUFA4',
                                     'NDUFA6',
                                     'NDUFA8',
                                     'NDUFA9',
                                     'NDUFB3',
                                     'NDUFB5',
                                     'NDUFB7',
                                     'NDUFB8',
                                     'NDUFS1',
                                     'NDUFS2',
                                     'NDUFS3',
                                     'SDHA',
                                     'SDHB',
                                     'SDHC',
                                     'UQCRC1',
                                     'UQCRC2',
                                     'UQCRFS1',
                                     'CYC1',
                                     'COX4I1',
                                     'COX5A',
                                     'COX5B',
                                     'COX6A1',
                                     'COX6B1',
                                     'ATP5F1A',
                                     'ATP5F1B',
                                     'ATP5MC1',
                                     'ATP5MC2',
                                     'ATP5PB'],
 'Oxidative-stress damage response': ['ATM',
                                      'ATR',
                                      'TP53',
                                      'SESN1',
                                      'SESN2',
                                      'DDIT3',
                                      'DDIT4',
                                      'GADD45A',
                                      'JUN',
                                      'FOS',
                                      'MAPK8',
                                      'MAPK14',
                                      'HSPA1A',
                                      'HSPA1B',
                                      'HSP90AA1',
                                      'DNAJB1',
                                      'ATF3',
                                      'ATF4'],
 'Hypoxia-redox adaptation': ['HIF1A',
                              'EGLN1',
                              'EGLN3',
                              'VEGFA',
                              'SLC2A1',
                              'LDHA',
                              'PDK1',
                              'BNIP3',
                              'BNIP3L',
                              'CA9',
                              'DDIT4',
                              'ADM',
                              'ENO1',
                              'PFKFB3',
                              'ALDOA'],
 'Mitochondrial quality control': ['PPARGC1A',
                                   'NRF1',
                                   'TFAM',
                                   'POLG',
                                   'LONP1',
                                   'CLPP',
                                   'HSPD1',
                                   'HSPE1',
                                   'OPA1',
                                   'MFN1',
                                   'MFN2',
                                   'DNM1L',
                                   'PINK1',
                                   'PRKN',
                                   'TOMM20',
                                   'TIMM23',
                                   'YME1L1',
                                   'OMA1']}


# ============================================================
# 1) GENERAL HELPERS
# ============================================================

def log(*items: Any) -> None:
    print(*items, flush=True)


def normalize_gene_name(value: Any) -> str:
    return str(value).strip().upper()


def norm_barcode(value: Any) -> str:
    text = str(value)
    return text.split("::", 1)[-1] if "::" in text else text


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    output = np.full(p_values.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(p_values)
    if not np.any(valid):
        return output

    values = p_values[valid]
    order = np.argsort(values)
    ranked = values[order]
    n = ranked.size

    adjusted = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output[valid] = restored
    return output


def standardized_mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    pooled = math.sqrt(
        max(
            (np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0,
            1e-12,
        )
    )
    return float((np.mean(x) - np.mean(y)) / pooled)


def save_figure_all_formats(figure: plt.Figure, stem: Path) -> None:
    for extension in ["png", "pdf", "svg"]:
        output_path = stem.with_suffix(f".{extension}")
        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if extension == "png":
            kwargs["dpi"] = DPI

        figure.savefig(output_path, **kwargs)
        log("[SAVE]", output_path)


def first_existing_column(
    table: pd.DataFrame,
    candidates: Iterable[str],
    description: str,
) -> str:
    for column in candidates:
        if column in table.columns:
            return column

    raise KeyError(
        f"Could not resolve {description}. "
        f"Tried={list(candidates)}; available={table.columns.tolist()}"
    )


def discover_paths() -> dict[str, Path]:
    checked = []

    for base in BASE_CANDIDATES:
        figure_root = (
            base
            / "runs"
            / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
            / DATASET_NAME
            / "fig2a_mltd_support_umap"
        )

        rna_dir = (
            base
            / "LLM-WorldWord"
            / DATASET_NAME
            / "processed"
            / "RNA"
        )

        manifest_path = rna_dir / "manifest_rna.tsv"

        for subdirectory in BIOLOGICAL_OUTPUT_SUBDIRECTORIES:
            output_dir = figure_root / subdirectory

            required = {
                "all_cells": output_dir / f"{TARGET_GENE}_all_cells_with_native_QC.csv",
                "mltd_pairs": output_dir / f"{TARGET_GENE}_expression_matched_pairs.csv",
                "mltd_cells": output_dir / f"{TARGET_GENE}_expression_matched_cells.csv",
                "mltd_de": output_dir / f"{TARGET_GENE}_matched_pair_native_RNA_DE.csv",
                "mltd_gsea": output_dir / f"{TARGET_GENE}_curated_module_preranked_GSEA.csv",
                "manifest": manifest_path,
            }

            checked.append(required)

            if all(path.exists() for path in required.values()):
                comparison_dir = (
                    output_dir
                    / "traditional_method_pathway_recovery_comparison"
                )
                comparison_dir.mkdir(parents=True, exist_ok=True)

                return {
                    "base": base,
                    "figure_root": figure_root,
                    "rna_dir": rna_dir,
                    "source_output_dir": output_dir,
                    "comparison_dir": comparison_dir,
                    **required,
                }

    lines = [
        "Could not locate the existing SOD2 biological-concordance outputs.",
        "Checked:",
    ]

    for item in checked:
        lines.append(json.dumps({k: str(v) for k, v in item.items()}, indent=2))

    raise FileNotFoundError("\n".join(lines))


# ============================================================
# 2) NATIVE RNA FILE HANDLING
# ============================================================

def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t")
    required = ["sample_id", "row_index"]

    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    manifest["sample_id"] = manifest["sample_id"].astype(str)
    return manifest


def resolve_counts_path(
    sample_id: str,
    rna_dir: Path,
    manifest_group: pd.DataFrame,
) -> Path:
    conventional = rna_dir / f"{sample_id}.X_rna_counts_dense.npy"
    if conventional.exists():
        return conventional

    for column in [
        "counts_npy",
        "counts_path",
        "rna_counts_path",
        "X_rna_counts_dense",
    ]:
        if column not in manifest_group.columns:
            continue

        raw_value = str(manifest_group[column].iloc[0]).strip()
        if not raw_value or raw_value.lower() == "nan":
            continue

        raw_path = Path(raw_value)
        candidates = [raw_path]

        if not raw_path.is_absolute():
            candidates.extend(
                [
                    rna_dir / raw_path,
                    rna_dir / raw_path.name,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

    local_candidates = sorted(rna_dir.glob(f"{sample_id}*counts*.npy"))
    if len(local_candidates) == 1:
        return local_candidates[0]

    raise FileNotFoundError(
        f"Cannot resolve native RNA count matrix for sample {sample_id}."
    )


def load_sample_metadata(sample_id: str, rna_dir: Path) -> dict[str, Any]:
    metadata_path = rna_dir / f"{sample_id}.rna_meta.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing RNA metadata: {metadata_path}")

    archive = np.load(metadata_path, allow_pickle=True)
    if "gene_names" not in archive:
        raise KeyError(f"{metadata_path.name} contains no gene_names.")

    gene_names = np.asarray(archive["gene_names"]).astype(str)
    normalized_names = np.asarray(
        [normalize_gene_name(gene) for gene in gene_names],
        dtype=object,
    )

    gene_to_index: dict[str, int] = {}
    for index, gene in enumerate(normalized_names.tolist()):
        if gene not in gene_to_index:
            gene_to_index[gene] = int(index)

    return {
        "metadata_path": metadata_path,
        "gene_names": gene_names,
        "gene_to_index": gene_to_index,
    }


def build_sample_info(
    sample_ids: Iterable[str],
    manifest: pd.DataFrame,
    rna_dir: Path,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    for sample_id in sorted(set(map(str, sample_ids))):
        manifest_group = manifest[
            manifest["sample_id"] == sample_id
        ]

        if manifest_group.shape[0] == 0:
            raise RuntimeError(f"No manifest rows for sample {sample_id}.")

        metadata = load_sample_metadata(sample_id, rna_dir)
        counts_path = resolve_counts_path(sample_id, rna_dir, manifest_group)
        counts = np.load(counts_path, mmap_mode="r")

        output[sample_id] = {
            **metadata,
            "counts_path": counts_path,
            "counts_shape": tuple(map(int, counts.shape)),
        }

        log(
            "[SAMPLE]",
            sample_id,
            f"cells={counts.shape[0]}",
            f"genes={counts.shape[1]}",
        )

    return output


def load_selected_counts_dense(
    cells: pd.DataFrame,
    genes: np.ndarray,
    sample_info: dict[str, dict[str, Any]],
) -> np.ndarray:
    genes = np.asarray([normalize_gene_name(gene) for gene in genes], dtype=object)

    output = np.zeros(
        (cells.shape[0], genes.size),
        dtype=np.float32,
    )

    for sample_id, group in cells.groupby("sid", sort=True):
        sample_id = str(sample_id)
        info = sample_info[sample_id]

        gene_indices = np.asarray(
            [
                info["gene_to_index"].get(str(gene), -1)
                for gene in genes.tolist()
            ],
            dtype=np.int64,
        )

        present = gene_indices >= 0
        if int(present.sum()) == 0:
            continue

        counts = np.load(info["counts_path"], mmap_mode="r")
        row_positions = group.index.to_numpy(dtype=np.int64)
        rid_values = group["rid"].to_numpy(dtype=np.int64)

        block = np.asarray(
            counts[rid_values][:, gene_indices[present]],
            dtype=np.float32,
        )

        output[np.ix_(row_positions, np.where(present)[0])] = block

    return output


def normalize_log1p_dense(
    counts: np.ndarray,
    total_counts: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)

    if total_counts is None:
        total_counts = np.sum(counts, axis=1, dtype=np.float64)

    total_counts = np.asarray(total_counts, dtype=np.float64)

    return np.log1p(
        counts
        / np.maximum(total_counts[:, None], 1.0)
        * float(NORMALIZE_TARGET_SUM)
    ).astype(np.float32)


# ============================================================
# 3) EXPRESSION-STRATIFIED MATCHING
# ============================================================

def choose_quantile_extremes(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values(
        ["SOD2_log1p", "rid"],
        ascending=[True, True],
    )

    n = int(group.shape[0])
    n_tail = max(5, int(math.floor(n * min(
        EXPRESSION_HIGH_QUANTILE,
        1.0 - EXPRESSION_LOW_QUANTILE,
    ))))

    n_tail = min(n_tail, n // 2)

    low = group.head(n_tail).copy()
    high = group.tail(n_tail).copy()

    overlap = set(low.index.tolist()) & set(high.index.tolist())
    if overlap:
        low = low.loc[~low.index.isin(overlap)].copy()
        high = high.loc[~high.index.isin(overlap)].copy()

    return high, low


def match_one_expression_stratum(
    high: pd.DataFrame,
    low: pd.DataFrame,
    quota: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if high.shape[0] == 0 or low.shape[0] == 0 or quota <= 0:
        return []

    covariates = [
        "log1p_total_counts",
        "log1p_n_genes",
    ]

    combined = pd.concat(
        [
            high[covariates],
            low[covariates],
        ],
        axis=0,
    ).to_numpy(dtype=np.float64)

    scaler = StandardScaler().fit(combined)

    high_scaled = scaler.transform(
        high[covariates].to_numpy(dtype=np.float64)
    )
    low_scaled = scaler.transform(
        low[covariates].to_numpy(dtype=np.float64)
    )

    n_neighbors = min(
        int(EXPRESSION_MATCH_NEIGHBORS),
        low.shape[0],
    )

    nearest = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit(low_scaled)

    distances, neighbors = nearest.kneighbors(high_scaled)

    # Harder cells first, with seeded jitter to avoid deterministic tie bias.
    difficulty = distances[:, 0] + rng.uniform(
        0.0,
        1e-8,
        size=distances.shape[0],
    )
    high_order = np.argsort(difficulty)[::-1]

    high_indices = high.index.to_numpy(dtype=np.int64)
    low_indices = low.index.to_numpy(dtype=np.int64)

    used_low: set[int] = set()
    rows = []

    for high_local in high_order.tolist():
        if len(rows) >= int(quota):
            break

        high_global = int(high_indices[high_local])
        selected_low = None
        selected_distance = None

        for rank in range(n_neighbors):
            low_local = int(neighbors[high_local, rank])
            low_global = int(low_indices[low_local])

            if (
                EXPRESSION_MATCH_WITHOUT_REPLACEMENT
                and low_global in used_low
            ):
                continue

            distance = float(distances[high_local, rank])
            if distance > float(EXPRESSION_MATCH_DISTANCE_CALIPER):
                continue

            selected_low = low_global
            selected_distance = distance
            break

        if selected_low is None:
            continue

        used_low.add(selected_low)

        high_row = high.loc[high_global]
        low_row = low.loc[selected_low]

        rows.append(
            {
                "sid": str(high_row["sid"]),
                "CCI_cluster_position": int(
                    high_row["CCI_cluster_position"]
                ),
                "CCI_cluster_name": str(
                    high_row.get("CCI_cluster_name", "")
                ),
                "high_cell_index": int(high_global),
                "low_cell_index": int(selected_low),
                "match_distance": float(selected_distance),
                "high_SOD2_log1p": float(high_row["SOD2_log1p"]),
                "low_SOD2_log1p": float(low_row["SOD2_log1p"]),
                "SOD2_log1p_difference": float(
                    high_row["SOD2_log1p"]
                    - low_row["SOD2_log1p"]
                ),
            }
        )

    return rows


def construct_expression_stratified_pairs(
    all_cells: pd.DataFrame,
    mltd_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cells = all_cells.copy()

    required = [
        "sid",
        "rid",
        "CCI_cluster_position",
        "SOD2_log1p",
        "log1p_total_counts",
        "log1p_n_genes",
    ]
    missing = [column for column in required if column not in all_cells.columns]
    if missing:
        raise KeyError(
            f"All-cell table is missing columns required for expression matching: {missing}"
        )

    all_cells["sid"] = all_cells["sid"].astype(str)
    all_cells["rid"] = all_cells["rid"].astype(np.int64)
    all_cells["CCI_cluster_position"] = all_cells[
        "CCI_cluster_position"
    ].astype(np.int64)

    # Preserve the same sample/cluster pair composition used by the MLTD contrast.
    quota_table = (
        mltd_pairs.groupby(
            ["sid", "high_cluster_position"],
            sort=True,
        )
        .size()
        .reset_index(name="target_pairs")
    )

    rng = np.random.default_rng(20260723)
    match_rows = []

    for _, quota_row in quota_table.iterrows():
        sample_id = str(quota_row["sid"])
        cluster_position = int(quota_row["high_cluster_position"])
        target_pairs = int(quota_row["target_pairs"])

        stratum = all_cells[
            (all_cells["sid"] == sample_id)
            & (
                all_cells["CCI_cluster_position"]
                == cluster_position
            )
            & np.isfinite(all_cells["SOD2_log1p"])
            & np.isfinite(all_cells["log1p_total_counts"])
            & np.isfinite(all_cells["log1p_n_genes"])
        ].copy()

        high, low = choose_quantile_extremes(stratum)

        rows = match_one_expression_stratum(
            high=high,
            low=low,
            quota=target_pairs,
            rng=rng,
        )

        match_rows.extend(rows)

        log(
            "[EXPRESSION MATCH]",
            f"sample={sample_id}",
            f"cluster={cluster_position}",
            f"target={target_pairs}",
            f"obtained={len(rows)}",
        )

    matches = pd.DataFrame(match_rows)
    if matches.shape[0] == 0:
        raise RuntimeError("No expression-stratified matched pairs were obtained.")

    matches["pair_id"] = np.arange(matches.shape[0], dtype=np.int64)

    high_cells = all_cells.loc[
        matches["high_cell_index"].to_numpy(dtype=np.int64)
    ].copy()
    low_cells = all_cells.loc[
        matches["low_cell_index"].to_numpy(dtype=np.int64)
    ].copy()

    high_cells["pair_id"] = matches["pair_id"].to_numpy(dtype=np.int64)
    low_cells["pair_id"] = matches["pair_id"].to_numpy(dtype=np.int64)
    high_cells["traditional_group"] = "SOD2-expression-high"
    low_cells["traditional_group"] = "SOD2-expression-low"

    selected_cells = pd.concat(
        [high_cells, low_cells],
        axis=0,
        ignore_index=True,
    )

    qc_rows = []
    for covariate in [
        "SOD2_log1p",
        "log1p_total_counts",
        "log1p_n_genes",
    ]:
        qc_rows.append(
            {
                "covariate": covariate,
                "SMD_high_minus_low": standardized_mean_difference(
                    high_cells[covariate].to_numpy(dtype=np.float64),
                    low_cells[covariate].to_numpy(dtype=np.float64),
                ),
                "high_mean": float(high_cells[covariate].mean()),
                "low_mean": float(low_cells[covariate].mean()),
            }
        )

    qc = pd.DataFrame(qc_rows)
    return matches, selected_cells, qc


# ============================================================
# 4) PAIRED NATIVE-RNA DIFFERENTIAL RANKING
# ============================================================

def paired_differential_ranking(
    counts: np.ndarray,
    genes: np.ndarray,
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_pairs = int(matches.shape[0])

    if counts.shape[0] != 2 * n_pairs:
        raise RuntimeError(
            f"Expected 2 × n_pairs rows, got counts={counts.shape}, pairs={n_pairs}."
        )

    total_counts = np.sum(counts, axis=1, dtype=np.float64)
    log_expression = normalize_log1p_dense(counts, total_counts)

    high_log = log_expression[:n_pairs]
    low_log = log_expression[n_pairs:]

    high_counts = counts[:n_pairs]
    low_counts = counts[n_pairs:]

    pair_difference = (
        high_log.astype(np.float32)
        - low_log.astype(np.float32)
    )

    mean_difference = np.mean(pair_difference, axis=0, dtype=np.float64)
    median_difference = np.median(pair_difference, axis=0)
    difference_sd = np.std(pair_difference, axis=0, ddof=1, dtype=np.float64)

    standard_error = difference_sd / math.sqrt(max(n_pairs, 1))
    t_statistic = np.divide(
        mean_difference,
        standard_error,
        out=np.zeros_like(mean_difference),
        where=standard_error > 1e-12,
    )

    p_value = 2.0 * stats.t.sf(
        np.abs(t_statistic),
        df=max(n_pairs - 1, 1),
    )

    fraction_high = np.mean(high_counts > 0, axis=0)
    fraction_low = np.mean(low_counts > 0, axis=0)

    min_expressed = max(
        int(MIN_EXPRESSED_CELLS),
        int(math.ceil(n_pairs * float(MIN_EXPRESSED_CELL_FRACTION))),
    )

    expressed_cells = (fraction_high + fraction_low) * n_pairs
    tested = expressed_cells >= float(min_expressed)

    adjusted = np.full(genes.size, np.nan, dtype=np.float64)
    adjusted[tested] = benjamini_hochberg(p_value[tested])

    cluster_column = first_existing_column(
        matches,
        [
            "high_cluster_position",
            "CCI_cluster_position",
        ],
        "pair high-cluster position",
    )

    pair_clusters = matches[cluster_column].to_numpy(dtype=np.int64)
    unique_clusters = sorted(np.unique(pair_clusters).tolist())

    cluster_effect_matrix = np.full(
        (len(unique_clusters), genes.size),
        np.nan,
        dtype=np.float64,
    )

    for cluster_row, cluster_position in enumerate(unique_clusters):
        mask = pair_clusters == int(cluster_position)
        if int(mask.sum()) == 0:
            continue
        cluster_effect_matrix[cluster_row] = np.mean(
            pair_difference[mask],
            axis=0,
            dtype=np.float64,
        )

    cluster_balanced_effect = np.nanmean(cluster_effect_matrix, axis=0)
    positive_fraction = np.nanmean(cluster_effect_matrix > 0, axis=0)
    negative_fraction = np.nanmean(cluster_effect_matrix < 0, axis=0)

    direction_consistency = np.maximum(
        positive_fraction,
        negative_fraction,
    )

    cluster_consensus_rank_score = (
        cluster_balanced_effect
        * np.clip(
            2.0 * direction_consistency - 1.0,
            0.0,
            1.0,
        )
    )

    gene_names = np.asarray(
        [normalize_gene_name(gene) for gene in genes],
        dtype=object,
    )

    table = pd.DataFrame(
        {
            "gene": gene_names.astype(str),
            "paired_mean_log_expression_difference": mean_difference,
            "paired_median_log_expression_difference": median_difference,
            "paired_difference_sd": difference_sd,
            "paired_t_statistic": t_statistic,
            "p_value": p_value,
            "FDR": adjusted,
            "fraction_expressed_high": fraction_high,
            "fraction_expressed_low": fraction_low,
            "tested": tested.astype(np.uint8),
            "cluster_balanced_effect": cluster_balanced_effect,
            "positive_cluster_fraction": positive_fraction,
            "negative_cluster_fraction": negative_fraction,
            "direction_consistency": direction_consistency,
            "cluster_consensus_rank_score": cluster_consensus_rank_score,
            "excluded_from_primary_enrichment": (
                gene_names == TARGET_GENE
            ).astype(np.uint8),
        }
    )

    return table, pair_difference


# ============================================================
# 5) PARTIAL SPEARMAN CO-EXPRESSION
# ============================================================

def build_covariate_design(all_cells: pd.DataFrame) -> np.ndarray:
    continuous = all_cells[
        [
            "log1p_total_counts",
            "log1p_n_genes",
        ]
    ].to_numpy(dtype=np.float64)

    continuous = (
        continuous
        - np.mean(continuous, axis=0, keepdims=True)
    ) / np.maximum(
        np.std(continuous, axis=0, ddof=1, keepdims=True),
        1e-8,
    )

    categorical = pd.DataFrame(
        {
            "sid": all_cells["sid"].astype(str),
            "cluster": all_cells["CCI_cluster_position"].astype(str),
        }
    )

    dummy = pd.get_dummies(
        categorical,
        columns=["sid", "cluster"],
        drop_first=True,
        dtype=float,
    ).to_numpy(dtype=np.float64)

    design = np.column_stack(
        [
            np.ones(all_cells.shape[0], dtype=np.float64),
            continuous,
            dummy,
        ]
    )

    # Remove rank-deficient columns.
    _, independent_indices = np.unique(
        np.round(design, 10),
        axis=1,
        return_index=True,
    )
    design = design[:, np.sort(independent_indices)]

    q, _ = np.linalg.qr(design, mode="reduced")
    return q


def coexpression_partial_spearman(
    all_cells: pd.DataFrame,
    genes: np.ndarray,
    sample_info: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    cells = all_cells.copy().reset_index(drop=True)

    valid = (
        np.isfinite(cells["SOD2_log1p"])
        & np.isfinite(cells["log1p_total_counts"])
        & np.isfinite(cells["log1p_n_genes"])
    )
    cells = cells.loc[valid].reset_index(drop=True)

    if COEXPRESSION_MAX_CELLS is not None and cells.shape[0] > int(
        COEXPRESSION_MAX_CELLS
    ):
        rng = np.random.default_rng(COEXPRESSION_SEED)
        selected = rng.choice(
            cells.index.to_numpy(dtype=np.int64),
            size=int(COEXPRESSION_MAX_CELLS),
            replace=False,
        )
        cells = cells.loc[np.sort(selected)].reset_index(drop=True)

    q = build_covariate_design(cells)

    target_rank = stats.rankdata(
        cells["SOD2_log1p"].to_numpy(dtype=np.float64),
        method="average",
    )
    target_residual = target_rank - q @ (q.T @ target_rank)
    target_residual -= np.mean(target_residual)
    target_norm = np.sqrt(np.sum(target_residual ** 2)) + 1e-12

    genes = np.asarray(
        [normalize_gene_name(gene) for gene in genes],
        dtype=object,
    )

    correlations = np.full(genes.size, np.nan, dtype=np.float64)
    p_values = np.full(genes.size, np.nan, dtype=np.float64)
    expressed_fraction = np.zeros(genes.size, dtype=np.float64)

    sample_groups = {
        str(sample_id): group.index.to_numpy(dtype=np.int64)
        for sample_id, group in cells.groupby("sid", sort=True)
    }

    for start in range(0, genes.size, int(COEXPRESSION_GENE_BLOCK_SIZE)):
        end = min(start + int(COEXPRESSION_GENE_BLOCK_SIZE), genes.size)
        block_genes = genes[start:end]

        raw_block = np.zeros(
            (cells.shape[0], block_genes.size),
            dtype=np.float32,
        )

        for sample_id, row_positions in sample_groups.items():
            info = sample_info[sample_id]
            counts = np.load(info["counts_path"], mmap_mode="r")

            sample_cells = cells.loc[row_positions]
            rid_values = sample_cells["rid"].to_numpy(dtype=np.int64)

            gene_indices = np.asarray(
                [
                    info["gene_to_index"].get(str(gene), -1)
                    for gene in block_genes.tolist()
                ],
                dtype=np.int64,
            )

            present = gene_indices >= 0
            if int(present.sum()) == 0:
                continue

            block = np.asarray(
                counts[rid_values][:, gene_indices[present]],
                dtype=np.float32,
            )

            raw_block[np.ix_(row_positions, np.where(present)[0])] = block

        expressed_fraction[start:end] = np.mean(raw_block > 0, axis=0)

        total_counts = cells["total_counts"].to_numpy(dtype=np.float64)
        log_block = np.log1p(
            raw_block
            / np.maximum(total_counts[:, None], 1.0)
            * float(NORMALIZE_TARGET_SUM)
        )

        ranked_block = stats.rankdata(
            log_block,
            axis=0,
            method="average",
        ).astype(np.float64)

        residual_block = ranked_block - q @ (q.T @ ranked_block)
        residual_block -= np.mean(residual_block, axis=0, keepdims=True)

        denominator = (
            target_norm
            * np.sqrt(np.sum(residual_block ** 2, axis=0))
            + 1e-12
        )

        block_correlation = (
            target_residual @ residual_block
        ) / denominator

        block_correlation = np.clip(block_correlation, -0.999999, 0.999999)

        degrees_of_freedom = max(
            cells.shape[0] - q.shape[1] - 2,
            1,
        )

        t_stat = (
            block_correlation
            * np.sqrt(
                degrees_of_freedom
                / np.maximum(1.0 - block_correlation ** 2, 1e-12)
            )
        )

        block_p = 2.0 * stats.t.sf(
            np.abs(t_stat),
            df=degrees_of_freedom,
        )

        correlations[start:end] = block_correlation
        p_values[start:end] = block_p

        log("[COEXPRESSION]", f"{end}/{genes.size} genes")

    tested = (
        expressed_fraction
        >= max(
            float(MIN_EXPRESSED_CELL_FRACTION),
            float(MIN_EXPRESSED_CELLS) / max(cells.shape[0], 1),
        )
    )

    fdr = np.full(genes.size, np.nan, dtype=np.float64)
    fdr[tested] = benjamini_hochberg(p_values[tested])

    result = pd.DataFrame(
        {
            "gene": genes.astype(str),
            "partial_Spearman_rho": correlations,
            "p_value": p_values,
            "FDR": fdr,
            "fraction_expressed": expressed_fraction,
            "tested": tested.astype(np.uint8),
            "cluster_consensus_rank_score": correlations,
            "excluded_from_primary_enrichment": (
                genes == TARGET_GENE
            ).astype(np.uint8),
            "n_cells": int(cells.shape[0]),
            "n_covariate_columns": int(q.shape[1]),
        }
    )

    return result


# ============================================================
# 6) LOCAL PRERANKED GSEA
# ============================================================

def enrichment_score(
    ranked_scores: np.ndarray,
    hit_mask: np.ndarray,
    weight_power: float,
) -> float:
    ranked_scores = np.asarray(ranked_scores, dtype=np.float64)
    hit_mask = np.asarray(hit_mask, dtype=bool)

    n_total = ranked_scores.size
    n_hits = int(hit_mask.sum())

    if n_hits <= 0 or n_hits >= n_total:
        return np.nan

    hit_weights = np.power(
        np.abs(ranked_scores[hit_mask]),
        float(weight_power),
    )

    hit_denominator = max(float(np.sum(hit_weights)), 1e-12)

    running_increment = np.full(
        n_total,
        -1.0 / max(n_total - n_hits, 1),
        dtype=np.float64,
    )

    running_increment[hit_mask] = (
        np.power(
            np.abs(ranked_scores[hit_mask]),
            float(weight_power),
        )
        / hit_denominator
    )

    running = np.cumsum(running_increment)

    maximum = float(np.max(running))
    minimum = float(np.min(running))

    return maximum if abs(maximum) >= abs(minimum) else minimum


def run_curated_preranked_gsea(
    ranking_table: pd.DataFrame,
    method_name: str,
    seed: int,
) -> pd.DataFrame:
    score_column = first_existing_column(
        ranking_table,
        [
            "cluster_consensus_rank_score",
            "partial_Spearman_rho",
        ],
        "ranking score",
    )

    tested_column = (
        "tested"
        if "tested" in ranking_table.columns
        else None
    )

    exclusion_column = (
        "excluded_from_primary_enrichment"
        if "excluded_from_primary_enrichment" in ranking_table.columns
        else None
    )

    mask = np.isfinite(
        ranking_table[score_column].to_numpy(dtype=np.float64)
    )

    if tested_column is not None:
        mask &= ranking_table[tested_column].to_numpy(dtype=int) == 1

    if exclusion_column is not None:
        mask &= (
            ranking_table[exclusion_column].to_numpy(dtype=int)
            == 0
        )

    rank_table = ranking_table.loc[
        mask,
        ["gene", score_column],
    ].copy()

    rank_table["gene"] = rank_table["gene"].map(normalize_gene_name)

    rank_table = rank_table.sort_values(
        score_column,
        ascending=False,
    ).drop_duplicates("gene").reset_index(drop=True)

    ranked_genes = rank_table["gene"].to_numpy(dtype=str)
    ranked_scores = rank_table[score_column].to_numpy(dtype=np.float64)

    gene_to_position = {
        gene: int(index)
        for index, gene in enumerate(ranked_genes.tolist())
    }

    rng = np.random.default_rng(int(seed))
    rows = []

    for module_name, module_genes in CURATED_SOD2_MODULES.items():
        cleaned = sorted(
            {
                normalize_gene_name(gene)
                for gene in module_genes
                if normalize_gene_name(gene) != TARGET_GENE
            }
        )

        present = [
            gene
            for gene in cleaned
            if gene in gene_to_position
        ]

        if (
            len(present) < int(GSEA_MIN_SET_SIZE)
            or len(present) > int(GSEA_MAX_SET_SIZE)
        ):
            continue

        hit_mask = np.zeros(ranked_genes.size, dtype=bool)
        hit_positions = [
            gene_to_position[gene]
            for gene in present
        ]
        hit_mask[hit_positions] = True

        observed_es = enrichment_score(
            ranked_scores,
            hit_mask,
            float(GSEA_WEIGHT_POWER),
        )

        null_es = np.zeros(
            int(N_GSEA_PERMUTATIONS),
            dtype=np.float64,
        )

        for permutation_index in range(int(N_GSEA_PERMUTATIONS)):
            random_positions = rng.choice(
                ranked_genes.size,
                size=len(present),
                replace=False,
            )

            random_mask = np.zeros(ranked_genes.size, dtype=bool)
            random_mask[random_positions] = True

            null_es[permutation_index] = enrichment_score(
                ranked_scores,
                random_mask,
                float(GSEA_WEIGHT_POWER),
            )

        if observed_es >= 0:
            same_sign = null_es[null_es >= 0]
            normalization = (
                float(np.mean(same_sign))
                if same_sign.size > 0
                else np.nan
            )
            p_value = (
                1.0 + np.sum(null_es >= observed_es)
            ) / (
                null_es.size + 1.0
            )
        else:
            same_sign = np.abs(null_es[null_es < 0])
            normalization = (
                float(np.mean(same_sign))
                if same_sign.size > 0
                else np.nan
            )
            p_value = (
                1.0 + np.sum(null_es <= observed_es)
            ) / (
                null_es.size + 1.0
            )

        nes = float(
            observed_es
            / max(abs(normalization), 1e-12)
        )

        running_increment = np.full(
            ranked_genes.size,
            -1.0 / max(ranked_genes.size - len(present), 1),
            dtype=np.float64,
        )

        hit_weights = np.power(
            np.abs(ranked_scores[hit_mask]),
            float(GSEA_WEIGHT_POWER),
        )

        running_increment[hit_mask] = (
            hit_weights
            / max(float(np.sum(hit_weights)), 1e-12)
        )

        running = np.cumsum(running_increment)

        if observed_es >= 0:
            peak_position = int(np.argmax(running))
            leading_edge = ranked_genes[
                hit_mask
                & (
                    np.arange(ranked_genes.size)
                    <= peak_position
                )
            ].tolist()
        else:
            trough_position = int(np.argmin(running))
            leading_edge = ranked_genes[
                hit_mask
                & (
                    np.arange(ranked_genes.size)
                    >= trough_position
                )
            ].tolist()

        rows.append(
            {
                "method": method_name,
                "module": module_name,
                "n_genes_in_module": len(cleaned),
                "n_genes_present": len(present),
                "ES": float(observed_es),
                "NES": nes,
                "p_value": float(p_value),
                "leading_edge_genes": ";".join(leading_edge),
                "n_leading_edge_genes": int(len(leading_edge)),
                "leading_edge_fraction": float(
                    len(leading_edge) / max(len(present), 1)
                ),
                "present_genes": ";".join(present),
                "SOD2_excluded": 1,
            }
        )

    result = pd.DataFrame(rows)

    if result.shape[0] > 0:
        result["FDR"] = benjamini_hochberg(
            result["p_value"].to_numpy(dtype=np.float64)
        )

    return result


# ============================================================
# 7) RECOVERY INDEX AND MATCHED-LABEL NULL
# ============================================================

def module_rank_recovery(
    ranking_table: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    score_column = first_existing_column(
        ranking_table,
        [
            "cluster_consensus_rank_score",
            "partial_Spearman_rho",
        ],
        "recovery ranking score",
    )

    mask = np.isfinite(
        ranking_table[score_column].to_numpy(dtype=np.float64)
    )

    if "tested" in ranking_table.columns:
        mask &= ranking_table["tested"].to_numpy(dtype=int) == 1

    if "excluded_from_primary_enrichment" in ranking_table.columns:
        mask &= (
            ranking_table[
                "excluded_from_primary_enrichment"
            ].to_numpy(dtype=int)
            == 0
        )

    rank_table = ranking_table.loc[
        mask,
        ["gene", score_column],
    ].copy()

    rank_table["gene"] = rank_table["gene"].map(normalize_gene_name)

    rank_table = rank_table.sort_values(
        score_column,
        ascending=False,
    ).drop_duplicates("gene").reset_index(drop=True)

    genes = rank_table["gene"].to_numpy(dtype=str)
    n = genes.size

    top_percentile = 1.0 - np.arange(n, dtype=np.float64) / max(n - 1, 1)

    gene_to_percentile = {
        gene: float(percentile)
        for gene, percentile in zip(
            genes.tolist(),
            top_percentile.tolist(),
        )
    }

    rows = []

    for module_name, module_genes in CURATED_SOD2_MODULES.items():
        present = [
            normalize_gene_name(gene)
            for gene in module_genes
            if (
                normalize_gene_name(gene) != TARGET_GENE
                and normalize_gene_name(gene) in gene_to_percentile
            )
        ]

        if len(present) < int(GSEA_MIN_SET_SIZE):
            continue

        mean_top_percentile = float(
            np.mean(
                [
                    gene_to_percentile[gene]
                    for gene in present
                ]
            )
        )

        centered_recovery = float(
            2.0 * (mean_top_percentile - 0.5)
        )

        rows.append(
            {
                "module": module_name,
                "n_genes_present": int(len(present)),
                "mean_top_rank_percentile": mean_top_percentile,
                "centered_rank_recovery": centered_recovery,
            }
        )

    module_table = pd.DataFrame(rows)
    aggregate = float(
        module_table["centered_rank_recovery"].mean()
    )

    return aggregate, module_table


def consensus_score_from_pair_differences(
    pair_difference: np.ndarray,
    pair_clusters: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    pair_difference = np.asarray(pair_difference, dtype=np.float32)
    pair_clusters = np.asarray(pair_clusters, dtype=np.int64)
    signs = np.asarray(signs, dtype=np.float32)

    unique_clusters = sorted(np.unique(pair_clusters).tolist())
    cluster_effects = []

    for cluster_position in unique_clusters:
        mask = pair_clusters == int(cluster_position)
        if int(mask.sum()) == 0:
            continue

        cluster_effects.append(
            np.mean(
                pair_difference[mask]
                * signs[mask, None],
                axis=0,
                dtype=np.float64,
            )
        )

    cluster_effect_matrix = np.stack(cluster_effects, axis=0)

    balanced = np.mean(cluster_effect_matrix, axis=0)
    positive_fraction = np.mean(cluster_effect_matrix > 0, axis=0)
    negative_fraction = np.mean(cluster_effect_matrix < 0, axis=0)

    consistency = np.maximum(positive_fraction, negative_fraction)

    return (
        balanced
        * np.clip(
            2.0 * consistency - 1.0,
            0.0,
            1.0,
        )
    )


def run_matched_label_null(
    mltd_pair_difference: np.ndarray,
    mltd_pairs: pd.DataFrame,
    genes: np.ndarray,
) -> pd.DataFrame:
    cluster_column = first_existing_column(
        mltd_pairs,
        ["high_cluster_position", "CCI_cluster_position"],
        "MLTD pair cluster",
    )

    pair_clusters = mltd_pairs[
        cluster_column
    ].to_numpy(dtype=np.int64)

    genes = np.asarray(
        [normalize_gene_name(gene) for gene in genes],
        dtype=object,
    )

    tested = np.ones(genes.size, dtype=np.uint8)
    excluded = (genes == TARGET_GENE).astype(np.uint8)

    rng = np.random.default_rng(int(LABEL_NULL_SEED))
    rows = []

    for repeat in range(int(N_MATCHED_LABEL_NULL)):
        signs = rng.choice(
            [-1.0, 1.0],
            size=mltd_pair_difference.shape[0],
            replace=True,
        )

        null_score = consensus_score_from_pair_differences(
            pair_difference=mltd_pair_difference,
            pair_clusters=pair_clusters,
            signs=signs,
        )

        ranking = pd.DataFrame(
            {
                "gene": genes.astype(str),
                "cluster_consensus_rank_score": null_score,
                "tested": tested,
                "excluded_from_primary_enrichment": excluded,
            }
        )

        aggregate, module_table = module_rank_recovery(ranking)

        rows.append(
            {
                "null_repeat": int(repeat),
                "aggregate_program_recovery_index": float(aggregate),
                **{
                    f"{row['module']}__rank_recovery": float(
                        row["centered_rank_recovery"]
                    )
                    for _, row in module_table.iterrows()
                },
            }
        )

        if (repeat + 1) % 20 == 0:
            log(
                "[LABEL NULL]",
                f"{repeat + 1}/{N_MATCHED_LABEL_NULL}",
            )

    return pd.DataFrame(rows)


# ============================================================
# 8) SUMMARIES
# ============================================================


def extract_ranked_gene_percentiles(
    ranking_table: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Convert a method-specific ranking into top-rank percentiles.

    A percentile of 1 indicates the top-ranked gene; 0 indicates the
    bottom-ranked gene. SOD2 is removed to avoid circularity.
    """
    score_column = first_existing_column(
        ranking_table,
        [
            "cluster_consensus_rank_score",
            "partial_Spearman_rho",
        ],
        "method ranking score",
    )

    mask = np.isfinite(
        ranking_table[score_column].to_numpy(dtype=np.float64)
    )

    if "tested" in ranking_table.columns:
        mask &= ranking_table["tested"].to_numpy(dtype=int) == 1

    if "excluded_from_primary_enrichment" in ranking_table.columns:
        mask &= (
            ranking_table[
                "excluded_from_primary_enrichment"
            ].to_numpy(dtype=int)
            == 0
        )

    table = ranking_table.loc[
        mask,
        ["gene", score_column],
    ].copy()

    table["gene"] = table["gene"].map(normalize_gene_name)

    table = (
        table.sort_values(
            score_column,
            ascending=False,
        )
        .drop_duplicates("gene")
        .reset_index(drop=True)
    )

    genes = table["gene"].to_numpy(dtype=str)
    n_genes = int(genes.size)

    percentiles = (
        1.0
        - np.arange(
            n_genes,
            dtype=np.float64,
        )
        / max(
            n_genes - 1,
            1,
        )
    )

    return (
        genes,
        {
            gene: float(percentile)
            for gene, percentile in zip(
                genes.tolist(),
                percentiles.tolist(),
            )
        },
    )


def run_method_specific_gene_set_null(
    rankings: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate an independent size-matched random-module null for each method.

    For each repeat and each predefined module:
      - preserve the number of module genes present in that method's ranking;
      - sample random genes without replacement;
      - calculate the centered mean top-rank percentile;
      - average across modules.

    This avoids comparing expression and co-expression methods with an MLTD-
    specific label-permutation null.
    """
    rows = []

    for method_index, (method_name, ranking_table) in enumerate(
        rankings.items()
    ):
        ranked_genes, gene_to_percentile = (
            extract_ranked_gene_percentiles(
                ranking_table
            )
        )

        module_sizes = []

        for module_name, module_genes in CURATED_SOD2_MODULES.items():
            present = {
                normalize_gene_name(gene)
                for gene in module_genes
                if (
                    normalize_gene_name(gene) != TARGET_GENE
                    and normalize_gene_name(gene)
                    in gene_to_percentile
                )
            }

            if len(present) >= int(GSEA_MIN_SET_SIZE):
                module_sizes.append(
                    (
                        module_name,
                        int(len(present)),
                    )
                )

        rng = np.random.default_rng(
            int(METHOD_GENESET_NULL_SEED)
            + method_index * 100003
        )

        percentile_array = np.asarray(
            [
                gene_to_percentile[gene]
                for gene in ranked_genes.tolist()
            ],
            dtype=np.float64,
        )

        for repeat in range(
            int(N_METHOD_GENESET_NULL)
        ):
            module_recoveries = []

            for module_name, module_size in module_sizes:
                selected = rng.choice(
                    ranked_genes.size,
                    size=int(module_size),
                    replace=False,
                )

                mean_top_percentile = float(
                    np.mean(
                        percentile_array[selected]
                    )
                )

                centered_recovery = float(
                    2.0
                    * (
                        mean_top_percentile
                        - 0.5
                    )
                )

                module_recoveries.append(
                    centered_recovery
                )

            rows.append(
                {
                    "method": method_name,
                    "null_repeat": int(repeat),
                    "aggregate_program_recovery_index": float(
                        np.mean(
                            module_recoveries
                        )
                    ),
                    "n_modules": int(
                        len(module_recoveries)
                    ),
                }
            )

            if (
                repeat + 1
            ) % 200 == 0:
                log(
                    "[METHOD GENE-SET NULL]",
                    method_name,
                    f"{repeat + 1}/{N_METHOD_GENESET_NULL}",
                )

    return pd.DataFrame(rows)


def summarize_methods(
    all_gsea: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    method_null: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_modules = [
        "Oxidative-stress damage response",
        "Mitochondrial respiratory chain",
        "Mitochondrial ROS detoxification",
        "NRF2 antioxidant response",
        "Mitochondrial quality control",
    ]

    rows = []

    for method_name, ranking in rankings.items():
        gsea = all_gsea[
            all_gsea["method"] == method_name
        ].copy()

        core_gsea = gsea[
            gsea["module"].isin(
                core_modules
            )
        ].copy()

        recovery_index, _ = module_rank_recovery(
            ranking
        )

        null_values = method_null.loc[
            method_null["method"] == method_name,
            "aggregate_program_recovery_index",
        ].to_numpy(dtype=np.float64)

        empirical_p = float(
            (
                1
                + np.sum(
                    null_values
                    >= recovery_index
                )
            )
            / (
                null_values.size
                + 1
            )
        )

        rows.append(
            {
                "method": method_name,
                "n_modules_tested": int(
                    gsea.shape[0]
                ),
                "n_core_modules": int(
                    core_gsea.shape[0]
                ),
                "n_positive_core_modules": int(
                    (
                        core_gsea["NES"]
                        > 0
                    ).sum()
                ),
                "core_directional_coherence": float(
                    (
                        core_gsea["NES"]
                        > 0
                    ).mean()
                ),
                "median_core_NES": float(
                    core_gsea["NES"].median()
                ),
                "mean_core_NES": float(
                    core_gsea["NES"].mean()
                ),
                "mean_leading_edge_fraction": float(
                    gsea[
                        "leading_edge_fraction"
                    ].mean()
                ),
                "aggregate_program_recovery_index": float(
                    recovery_index
                ),
                "empirical_P_vs_method_specific_gene_set_null": (
                    empirical_p
                ),
            }
        )

    method_summary = pd.DataFrame(
        rows
    )

    pivot = all_gsea.pivot(
        index="module",
        columns="method",
        values="NES",
    )

    traditional_columns = [
        column
        for column in [
            "SOD2 expression",
            "SOD2 co-expression",
        ]
        if column in pivot.columns
    ]

    advantage_rows = []

    for module_name in pivot.index.tolist():
        mltd_nes = float(
            pivot.loc[
                module_name,
                "MLTD dependency",
            ]
        )

        traditional_values = pivot.loc[
            module_name,
            traditional_columns,
        ].to_numpy(dtype=np.float64)

        best_traditional = float(
            np.nanmax(
                traditional_values
            )
        )

        best_method = traditional_columns[
            int(
                np.nanargmax(
                    traditional_values
                )
            )
        ]

        advantage_rows.append(
            {
                "module": module_name,
                "MLTD_NES": mltd_nes,
                "best_traditional_NES": best_traditional,
                "best_traditional_method": best_method,
                "MLTD_minus_best_traditional_NES": float(
                    mltd_nes
                    - best_traditional
                ),
            }
        )

    advantage = (
        pd.DataFrame(
            advantage_rows
        )
        .sort_values(
            "MLTD_minus_best_traditional_NES",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return (
        method_summary,
        advantage,
    )


# ============================================================
# 9) PLOTTING
# ============================================================

def style_axis(axis: plt.Axes, grid_axis: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.9)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.tick_params(width=0.8, length=3.2, labelsize=8.0)

    if grid_axis is not None:
        axis.grid(
            axis=grid_axis,
            color=COLORS["grid"],
            linewidth=0.75,
        )
        axis.set_axisbelow(True)


def smooth_histogram_density(
    values: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(
        values,
        bins=bins,
        density=True,
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    kernel = np.asarray(
        [1.0, 2.0, 3.0, 2.0, 1.0],
        dtype=np.float64,
    )
    kernel /= kernel.sum()

    smoothed = np.convolve(
        counts,
        kernel,
        mode="same",
    )

    if np.max(smoothed) > 0:
        smoothed = (
            smoothed
            / np.max(smoothed)
        )

    return centers, smoothed


def plot_pathway_recovery_figure(
    all_gsea: pd.DataFrame,
    method_summary: pd.DataFrame,
    method_null: pd.DataFrame,
    module_advantage: pd.DataFrame,
    output_dir: Path,
) -> None:
    method_order = [
        "MLTD dependency",
        "SOD2 expression",
        "SOD2 co-expression",
    ]

    module_order = [
        "Oxidative-stress damage response",
        "Mitochondrial respiratory chain",
        "Mitochondrial ROS detoxification",
        "NRF2 antioxidant response",
        "Mitochondrial quality control",
        "Hypoxia-redox adaptation",
    ]

    available_modules = [
        module
        for module in module_order
        if module
        in set(
            all_gsea[
                "module"
            ].tolist()
        )
    ]

    figure = plt.figure(
        figsize=(
            12.6,
            8.0,
        )
    )

    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[
            1.26,
            1.0,
        ],
        height_ratios=[
            1.08,
            1.0,
        ],
        wspace=0.38,
        hspace=0.45,
    )

    axis_a = figure.add_subplot(
        grid[
            0,
            0
        ]
    )

    axis_b = figure.add_subplot(
        grid[
            0,
            1
        ]
    )

    axis_c = figure.add_subplot(
        grid[
            1,
            0
        ]
    )

    axis_d = figure.add_subplot(
        grid[
            1,
            1
        ]
    )

    # --------------------------------------------------------
    # A. Rectangular NES heatmap
    # --------------------------------------------------------
    heatmap_table = all_gsea.pivot(
        index="module",
        columns="method",
        values="NES",
    ).reindex(
        index=available_modules,
        columns=method_order,
    )

    fdr_table = all_gsea.pivot(
        index="module",
        columns="method",
        values="FDR",
    ).reindex(
        index=available_modules,
        columns=method_order,
    )

    heatmap_values = heatmap_table.to_numpy(
        dtype=np.float64
    )

    limit = max(
        1.5,
        float(
            np.nanmax(
                np.abs(
                    heatmap_values
                )
            )
        )
        * 1.05,
    )

    image = axis_a.imshow(
        heatmap_values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )

    axis_a.set_xticks(
        np.arange(
            len(
                method_order
            )
        )
    )

    axis_a.set_xticklabels(
        method_order,
        rotation=24,
        ha="right",
    )

    axis_a.set_yticks(
        np.arange(
            len(
                available_modules
            )
        )
    )

    axis_a.set_yticklabels(
        available_modules
    )

    for row_index, module in enumerate(
        available_modules
    ):
        for column_index, method in enumerate(
            method_order
        ):
            value = float(
                heatmap_table.loc[
                    module,
                    method,
                ]
            )

            fdr = float(
                fdr_table.loc[
                    module,
                    method,
                ]
            )

            text_color = (
                "white"
                if abs(
                    value
                )
                >= 0.90
                else COLORS[
                    "text"
                ]
            )

            annotation = f"{value:.2f}"

            if np.isfinite(
                fdr
            ) and fdr < 0.05:
                annotation += "•"

            axis_a.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.0,
                fontweight=(
                    "bold"
                    if (
                        np.isfinite(
                            fdr
                        )
                        and fdr
                        < 0.05
                    )
                    else "normal"
                ),
            )

    axis_a.set_title(
        "A  SOD2-compatible program landscape",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )

    axis_a.tick_params(
        length=0
    )

    for spine in axis_a.spines.values():
        spine.set_visible(
            False
        )

    for boundary in np.arange(
        -0.5,
        len(
            available_modules
        ),
        1.0,
    ):
        axis_a.axhline(
            boundary,
            color="white",
            linewidth=1.3,
        )

    for boundary in np.arange(
        -0.5,
        len(
            method_order
        ),
        1.0,
    ):
        axis_a.axvline(
            boundary,
            color="white",
            linewidth=1.3,
        )

    colorbar = figure.colorbar(
        image,
        ax=axis_a,
        fraction=0.042,
        pad=0.025,
    )

    colorbar.set_label(
        "Normalized enrichment score",
        fontsize=8.2,
    )

    colorbar.ax.tick_params(
        labelsize=7.5
    )

    axis_a.text(
        1.0,
        -0.16,
        "• FDR < 0.05",
        transform=axis_a.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color="#666666",
    )

    # --------------------------------------------------------
    # B. Core-program directional coherence
    # --------------------------------------------------------
    summary_plot = (
        method_summary.set_index(
            "method"
        )
        .reindex(
            method_order
        )
        .reset_index()
    )

    y_positions = np.arange(
        len(
            method_order
        )
    )

    bars = axis_b.barh(
        y_positions,
        summary_plot[
            "core_directional_coherence"
        ].to_numpy(
            dtype=np.float64
        ),
        color=[
            COLORS[
                method
            ]
            for method in method_order
        ],
        alpha=0.88,
        height=0.54,
    )

    for y_position, bar, (_, row) in zip(
        y_positions.tolist(),
        bars,
        summary_plot.iterrows(),
    ):
        axis_b.text(
            min(
                float(
                    row[
                        "core_directional_coherence"
                    ]
                )
                + 0.03,
                0.97,
            ),
            y_position,
            (
                f"{int(row['n_positive_core_modules'])}/"
                f"{int(row['n_core_modules'])}"
            ),
            ha=(
                "right"
                if float(
                    row[
                        "core_directional_coherence"
                    ]
                )
                > 0.90
                else "left"
            ),
            va="center",
            fontsize=8.2,
            color=(
                "white"
                if float(
                    row[
                        "core_directional_coherence"
                    ]
                )
                > 0.90
                else COLORS[
                    "text"
                ]
            ),
            fontweight="bold",
        )

        axis_b.text(
            1.03,
            y_position,
            (
                f"median NES={float(row['median_core_NES']):.2f}"
            ),
            ha="left",
            va="center",
            fontsize=7.6,
            color="#555555",
        )

    axis_b.set_yticks(
        y_positions
    )

    axis_b.set_yticklabels(
        method_order
    )

    axis_b.invert_yaxis()

    axis_b.set_xlim(
        0.0,
        1.42,
    )

    axis_b.set_xticks(
        [
            0.0,
            0.5,
            1.0,
        ]
    )

    axis_b.set_xticklabels(
        [
            "0%",
            "50%",
            "100%",
        ]
    )

    axis_b.set_xlabel(
        "Core SOD2 modules with NES > 0"
    )

    axis_b.set_title(
        "B  Core-program directional coherence",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )

    style_axis(
        axis_b,
        grid_axis="x",
    )

    # --------------------------------------------------------
    # C. Method-specific random-module nulls
    # --------------------------------------------------------
    offsets = [
        2.0,
        1.0,
        0.0,
    ]

    global_min = float(
        min(
            method_null[
                "aggregate_program_recovery_index"
            ].min(),
            summary_plot[
                "aggregate_program_recovery_index"
            ].min(),
        )
    )

    global_max = float(
        max(
            method_null[
                "aggregate_program_recovery_index"
            ].max(),
            summary_plot[
                "aggregate_program_recovery_index"
            ].max(),
        )
    )

    for method_index, method in enumerate(
        method_order
    ):
        values = method_null.loc[
            method_null[
                "method"
            ]
            == method,
            "aggregate_program_recovery_index",
        ].to_numpy(
            dtype=np.float64
        )

        centers, density = smooth_histogram_density(
            values,
            bins=36,
        )

        offset = offsets[
            method_index
        ]

        axis_c.fill_between(
            centers,
            offset,
            offset
            + 0.72
            * density,
            color=COLORS[
                "null"
            ],
            alpha=0.48,
            linewidth=0,
        )

        axis_c.plot(
            centers,
            offset
            + 0.72
            * density,
            color="#8C929A",
            linewidth=1.2,
        )

        observed_row = summary_plot[
            summary_plot[
                "method"
            ]
            == method
        ].iloc[
            0
        ]

        observed = float(
            observed_row[
                "aggregate_program_recovery_index"
            ]
        )

        empirical_p = float(
            observed_row[
                "empirical_P_vs_method_specific_gene_set_null"
            ]
        )

        axis_c.vlines(
            observed,
            offset,
            offset
            + 0.78,
            color=COLORS[
                method
            ],
            linewidth=2.4,
        )

        axis_c.scatter(
            [
                observed
            ],
            [
                offset
                + 0.78
            ],
            s=38,
            color=COLORS[
                method
            ],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

        axis_c.text(
            global_max
            + 0.02
            * (
                global_max
                - global_min
            ),
            offset
            + 0.34,
            (
                f"{method}\n"
                f"P={empirical_p:.3g}"
            ),
            ha="left",
            va="center",
            fontsize=7.7,
            color=COLORS[
                method
            ],
        )

    axis_c.set_xlim(
        global_min
        - 0.05
        * (
            global_max
            - global_min
        ),
        global_max
        + 0.45
        * (
            global_max
            - global_min
        ),
    )

    axis_c.set_ylim(
        -0.12,
        2.92,
    )

    axis_c.set_yticks(
        []
    )

    axis_c.set_xlabel(
        "Aggregate predefined-program recovery index"
    )

    axis_c.set_title(
        "C  Recovery relative to method-specific random modules",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )

    axis_c.spines[
        "left"
    ].set_visible(
        False
    )

    axis_c.spines[
        "top"
    ].set_visible(
        False
    )

    axis_c.spines[
        "right"
    ].set_visible(
        False
    )

    axis_c.tick_params(
        width=0.8,
        length=3.2,
        labelsize=8.0,
    )

    axis_c.grid(
        axis="x",
        color=COLORS[
            "grid"
        ],
        linewidth=0.75,
    )

    axis_c.set_axisbelow(
        True
    )

    # --------------------------------------------------------
    # D. Differential module recovery
    # --------------------------------------------------------
    advantage = module_advantage.copy()

    y_positions = np.arange(
        advantage.shape[
            0
        ]
    )

    values = advantage[
        "MLTD_minus_best_traditional_NES"
    ].to_numpy(
        dtype=np.float64
    )

    axis_d.axvline(
        0.0,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
    )

    for y_position, value in zip(
        y_positions.tolist(),
        values.tolist(),
    ):
        color = (
            COLORS[
                "MLTD dependency"
            ]
            if value
            >= 0
            else COLORS[
                "SOD2 co-expression"
            ]
        )

        axis_d.plot(
            [
                0.0,
                value,
            ],
            [
                y_position,
                y_position,
            ],
            color=color,
            linewidth=1.9,
        )

        axis_d.scatter(
            value,
            y_position,
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    axis_d.set_yticks(
        y_positions
    )

    axis_d.set_yticklabels(
        advantage[
            "module"
        ].tolist()
    )

    axis_d.set_xlabel(
        "ΔNES: MLTD − best traditional strategy"
    )

    axis_d.set_title(
        "D  Module-level differential recovery",
        loc="left",
        fontsize=11.2,
        fontweight="bold",
    )

    axis_d.text(
        0.02,
        0.02,
        (
            "Positive: preferential MLTD recovery\n"
            "Negative: stronger traditional recovery"
        ),
        transform=axis_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#666666",
    )

    style_axis(
        axis_d,
        grid_axis="x",
    )

    figure.suptitle(
        (
            "MLTD resolves a more directionally coherent core SOD2 program "
            "than conventional transcriptomic strategies"
        ),
        fontsize=13.8,
        fontweight="bold",
        y=1.02,
    )

    figure.tight_layout()

    save_figure_all_formats(
        figure,
        output_dir
        / "SOD2_MLTD_vs_traditional_pathway_recovery_v2",
    )

    plt.close(
        figure
    )


# ============================================================
# 10) MAIN
# ============================================================

def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    paths = discover_paths()
    output_dir = paths["comparison_dir"]

    log("[SOURCE OUTPUT]", paths["source_output_dir"])
    log("[COMPARISON OUTPUT]", output_dir)

    all_cells = pd.read_csv(paths["all_cells"])
    mltd_pairs = pd.read_csv(paths["mltd_pairs"])
    mltd_cells = pd.read_csv(paths["mltd_cells"])
    mltd_de = pd.read_csv(paths["mltd_de"])
    existing_mltd_gsea = pd.read_csv(paths["mltd_gsea"])
    manifest = load_manifest(paths["manifest"])

    for table in [all_cells, mltd_cells]:
        table["sid"] = table["sid"].astype(str)
        table["rid"] = table["rid"].astype(np.int64)

    genes = (
        mltd_de["gene"]
        .astype(str)
        .map(normalize_gene_name)
        .drop_duplicates()
        .to_numpy(dtype=object)
    )

    sample_ids = sorted(
        set(all_cells["sid"].astype(str).tolist())
        | set(mltd_cells["sid"].astype(str).tolist())
    )

    sample_info = build_sample_info(
        sample_ids=sample_ids,
        manifest=manifest,
        rna_dir=paths["rna_dir"],
    )

    # --------------------------------------------------------
    # MLTD method: preserve the exact existing Fig. 2h ranking.
    # --------------------------------------------------------
    mltd_ranking = mltd_de.copy()
    mltd_ranking["gene"] = mltd_ranking["gene"].map(normalize_gene_name)

    # Recompute GSEA with the same implementation used for all methods,
    # ensuring strict method comparability.
    mltd_gsea = run_curated_preranked_gsea(
        ranking_table=mltd_ranking,
        method_name="MLTD dependency",
        seed=3301,
    )

    # --------------------------------------------------------
    # Native SOD2-expression stratification.
    # --------------------------------------------------------
    (
        expression_pairs,
        expression_cells,
        expression_qc,
    ) = construct_expression_stratified_pairs(
        all_cells=all_cells,
        mltd_pairs=mltd_pairs,
    )

    expression_pairs_path = (
        output_dir / "SOD2_expression_stratified_pairs.csv"
    )
    expression_cells_path = (
        output_dir / "SOD2_expression_stratified_cells.csv"
    )
    expression_qc_path = (
        output_dir / "SOD2_expression_stratified_matching_QC.csv"
    )

    expression_pairs.to_csv(expression_pairs_path, index=False)
    expression_cells.to_csv(expression_cells_path, index=False)
    expression_qc.to_csv(expression_qc_path, index=False)

    for path in [
        expression_pairs_path,
        expression_cells_path,
        expression_qc_path,
    ]:
        log("[SAVE]", path)

    expression_counts = load_selected_counts_dense(
        cells=expression_cells,
        genes=genes,
        sample_info=sample_info,
    )

    expression_de, _ = paired_differential_ranking(
        counts=expression_counts,
        genes=genes,
        matches=expression_pairs,
    )

    expression_de_path = (
        output_dir / "SOD2_expression_stratified_native_RNA_DE.csv"
    )
    expression_de.to_csv(expression_de_path, index=False)
    log("[SAVE]", expression_de_path)

    expression_gsea = run_curated_preranked_gsea(
        ranking_table=expression_de,
        method_name="SOD2 expression",
        seed=4301,
    )

    # --------------------------------------------------------
    # Conventional partial-Spearman co-expression.
    # --------------------------------------------------------
    coexpression_ranking = coexpression_partial_spearman(
        all_cells=all_cells,
        genes=genes,
        sample_info=sample_info,
    )

    coexpression_path = (
        output_dir / "SOD2_partial_Spearman_coexpression_ranking.csv"
    )
    coexpression_ranking.to_csv(coexpression_path, index=False)
    log("[SAVE]", coexpression_path)

    coexpression_gsea = run_curated_preranked_gsea(
        ranking_table=coexpression_ranking,
        method_name="SOD2 co-expression",
        seed=5301,
    )

    # --------------------------------------------------------
    # MLTD matched-label null.
    # --------------------------------------------------------
    mltd_counts = load_selected_counts_dense(
        cells=mltd_cells,
        genes=genes,
        sample_info=sample_info,
    )

    _, mltd_pair_difference = paired_differential_ranking(
        counts=mltd_counts,
        genes=genes,
        matches=mltd_pairs,
    )

    label_null = run_matched_label_null(
        mltd_pair_difference=mltd_pair_difference,
        mltd_pairs=mltd_pairs,
        genes=genes,
    )

    label_null_path = (
        output_dir / "SOD2_matched_label_null_recovery_index.csv"
    )
    label_null.to_csv(label_null_path, index=False)
    log("[SAVE]", label_null_path)

    # --------------------------------------------------------
    # Combined outputs.
    # --------------------------------------------------------
    all_gsea = pd.concat(
        [
            mltd_gsea,
            expression_gsea,
            coexpression_gsea,
        ],
        axis=0,
        ignore_index=True,
    )

    all_gsea_path = (
        output_dir / "SOD2_pathway_recovery_all_methods_GSEA.csv"
    )
    all_gsea.to_csv(all_gsea_path, index=False)
    log("[SAVE]", all_gsea_path)

    rankings = {
        "MLTD dependency": mltd_ranking,
        "SOD2 expression": expression_de,
        "SOD2 co-expression": coexpression_ranking,
    }

    method_null = run_method_specific_gene_set_null(
        rankings=rankings,
    )

    method_null_path = (
        output_dir
        / "SOD2_method_specific_random_module_null.csv"
    )

    method_null.to_csv(
        method_null_path,
        index=False,
    )

    log(
        "[SAVE]",
        method_null_path,
    )

    method_summary, module_advantage = summarize_methods(
        all_gsea=all_gsea,
        rankings=rankings,
        method_null=method_null,
    )

    method_summary_path = (
        output_dir
        / "SOD2_pathway_recovery_method_summary_v2.csv"
    )

    advantage_path = (
        output_dir
        / "SOD2_pathway_recovery_module_differential_v2.csv"
    )

    method_summary.to_csv(
        method_summary_path,
        index=False,
    )

    module_advantage.to_csv(
        advantage_path,
        index=False,
    )

    log(
        "[SAVE]",
        method_summary_path,
    )

    log(
        "[SAVE]",
        advantage_path,
    )

    plot_pathway_recovery_figure(
        all_gsea=all_gsea,
        method_summary=method_summary,
        method_null=method_null,
        module_advantage=module_advantage,
        output_dir=output_dir,
    )

    manifest_output = {
        "dataset": DATASET_NAME,
        "target_gene": TARGET_GENE,
        "source_output_directory": str(paths["source_output_dir"]),
        "comparison_output_directory": str(output_dir),
        "methods": {
            "MLTD dependency": (
                "Existing expression-matched MLTD-high versus MLTD-low "
                "native-RNA cluster-consensus ranking."
            ),
            "SOD2 expression": (
                "Within-sample and within-CCI-cluster SOD2-expression-high "
                "versus SOD2-expression-low matched comparison, controlling "
                "total UMI and detected genes."
            ),
            "SOD2 co-expression": (
                "Gene-wise partial Spearman correlation with native SOD2 "
                "expression, controlling sample, CCI cluster, total UMI and "
                "detected genes."
            ),
            "matched_label_null": (
                "Random high/low direction swaps within the existing MLTD "
                "expression-matched pairs."
            ),
        },
        "curated_modules": CURATED_SOD2_MODULES,
        "SOD2_excluded_from_enrichment": True,
        "GSEA_permutations": int(N_GSEA_PERMUTATIONS),
        "matched_label_null_repeats": int(N_MATCHED_LABEL_NULL),
        "method_specific_random_module_null_repeats": int(N_METHOD_GENESET_NULL),
        "expression_matched_pairs_obtained": int(expression_pairs.shape[0]),
        "MLTD_matched_pairs": int(mltd_pairs.shape[0]),
        "method_specific_null_definition": (
            "For each analytical strategy, predefined module sizes are "
            "preserved while gene identities are randomly sampled from "
            "that strategy's tested ranking."
        ),
        "recovery_index_definition": (
            "Mean across predefined modules of "
            "2 × (mean top-rank percentile of module genes − 0.5)."
        ),
        "interpretation_boundary": (
            "The comparison tests recovery of predefined SOD2-compatible "
            "native transcriptional programs. It does not establish direct "
            "causal regulation by SOD2 or by MLTD edges."
        ),
    }

    manifest_output_path = (
        output_dir / "SOD2_pathway_recovery_manifest.json"
    )

    with open(
        manifest_output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest_output,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    log("[SAVE]", manifest_output_path)
    log("[DONE]", "MLTD versus traditional SOD2 pathway-recovery comparison completed.")


if __name__ == "__main__":
    main()
