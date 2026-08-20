from __future__ import annotations

# =============================================================================
# Endothelial and pericyte statistics across eight kidney-cancer samples
#
# Samples:
# Sample-1–6, Sample-8 and Sample-9
#
# Updated annotation:
# Cluster 2 = Pericytes
#
# Input:
# native_vs_MLTD_mapped_cell_coordinates.tsv
#
# Cluster column:
# new_cluster
# =============================================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. PATH CONFIGURATION
# =============================================================================

BASE = get_project_root()

RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "stageB_RNA_only_Kidney_cancer_8samples_exclude_sample7_joint"
)

DATASET_NAME = "scVerifyKC-Kidney-cancer-8samples-exclude-sample7-joint"

RUN_STAGEB_DATASET_DIR = (
    RUN_STAGEB_DIR
    / DATASET_NAME
)

UMAP_OUT_DIR = (
    RUN_STAGEB_DATASET_DIR
    / "EIGHT_SAMPLES_NO_SAMPLE7_native_vs_MLTD_mapped_integrated_UMAP"
)

CELL_COORD_TSV = (
    UMAP_OUT_DIR
    / "native_vs_MLTD_mapped_cell_coordinates.tsv"
)

OUT_DIR = (
    UMAP_OUT_DIR
    / "endothelial_pericyte_statistics_by_sample"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_SUMMARY_TSV = (
    OUT_DIR
    / "endothelial_pericyte_counts_by_sample.tsv"
)

OUT_ENDOTHELIAL_LONG_TSV = (
    OUT_DIR
    / "endothelial_subtype_counts_long.tsv"
)

OUT_CLUSTER_COUNTS_TSV = (
    OUT_DIR
    / "all_cluster_counts_by_sample.tsv"
)

OUT_CELL_METADATA_TSV_GZ = (
    OUT_DIR
    / "cell_metadata_with_vascular_annotation.tsv.gz"
)


# =============================================================================
# 1. EXPECTED SAMPLE STRUCTURE
# =============================================================================

EXPECTED_SAMPLE_NUMBERS = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
]

# Histological labels from the supplied screenshot.
SAMPLE_HISTOLOGY = {
    1: "Papillary RCC",
    2: "Clear cell RCC",
    3: "Chromophobe RCC",
    4: "Clear cell RCC",
    5: "Chromophobe RCC",
    6: "Chromophobe RCC",
    8: "Clear cell RCC",
    9: "Clear cell RCC",
}


# =============================================================================
# 2. UPDATED CLUSTER ANNOTATION
# =============================================================================

CLUSTER_ANNOTATION = {
    "0": "SPARCL1+ endothelial cells",
    "1": "APOC1+ macrophages",

    # Updated annotation.
    "2": "Pericytes",

    "3": "IL1B+ macrophages",
    "4": "NK cells",
    "5": "OLR1+ monocytes",
    "6": "KRT17+ tumour cells",
    "7": "JCHAIN+ plasma cells",
    "8": "CCL4+ macrophages",
    "9": "HLA-DRA+ macrophages",
    "10": "SERPINA1+ macrophages",
    "11": "IFI27+ endothelial cells",
    "12": "S100A9+ neutrophils",
    "13": "CCL3+ macrophages",
    "14": "IGHG+ plasma cells",
    "15": "CCL14+ endothelial cells",
    "16": "CCL4+ T/NK cells",
    "17": "SPP1+ macrophages",
    "18": "FCGR3B+ neutrophils",
    "19": "Proximal tubular cells",
}

ENDOTHELIAL_SUBTYPE_CLUSTERS = {
    "SPARCL1+ endothelial cells": "0",
    "IFI27+ endothelial cells": "11",
    "CCL14+ endothelial cells": "15",
}

PERICYTE_SUBTYPE_CLUSTERS = {
    "Pericytes": "2",
}

ENDOTHELIAL_CLUSTER_IDS = set(
    ENDOTHELIAL_SUBTYPE_CLUSTERS.values()
)

PERICYTE_CLUSTER_IDS = set(
    PERICYTE_SUBTYPE_CLUSTERS.values()
)


# =============================================================================
# 3. FIGURE SETTINGS
# =============================================================================

FIG_DPI = 400

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Preserve editable text in SVG/PDF.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ENDOTHELIAL_COLORS = {
    "SPARCL1+ endothelial cells": "#BFD9D5",
    "IFI27+ endothelial cells": "#68AAA5",
    "CCL14+ endothelial cells": "#176C70",
}

TOTAL_ENDOTHELIAL_COLOR = "#287C80"
PERICYTE_COLOR = "#C28B5A"


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

def normalize_cluster(value) -> str:
    """
    Normalize cluster labels.

    Examples
    --------
    C2       -> 2
    Cluster2 -> 2
    2.0      -> 2
    ' 2 '    -> 2
    """

    text = str(value).strip()

    cluster_match = re.fullmatch(
        r"(?:C|Cluster)[-_ ]?(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if cluster_match is not None:
        return str(
            int(cluster_match.group(1))
        )

    if re.fullmatch(r"\d+\.0+", text):
        return str(
            int(float(text))
        )

    if re.fullmatch(r"\d+", text):
        return str(
            int(text)
        )

    return text


def extract_sample_number(sample_id: str) -> int:
    """
    Extract sample number from a sample label.

    Examples
    --------
    Kidney-cancer-Sample-1 -> 1
    Sample_08              -> 8
    Sample 9               -> 9
    """

    text = str(sample_id).strip()

    match = re.search(
        r"Sample[-_ ]?0*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            "Could not extract the sample number from "
            f"sample_id={sample_id!r}"
        )

    return int(
        match.group(1)
    )


def save_figure_all_formats(
    figure: plt.Figure,
    output_base: Path,
) -> None:
    """
    Save PNG, SVG and PDF versions of one figure.
    """

    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
    )

    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(figure)


def add_count_labels(
    axis: plt.Axes,
    bars,
    fontsize: float = 8.0,
) -> None:
    """
    Add integer count labels above bars.
    """

    for bar in bars:
        height = float(
            bar.get_height()
        )

        if not np.isfinite(height):
            continue

        if height <= 0:
            continue

        axis.annotate(
            f"{int(round(height)):,}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def add_percentage_labels(
    axis: plt.Axes,
    bars,
    fontsize: float = 7.5,
) -> None:
    """
    Add percentage labels above bars.
    """

    for bar in bars:
        height = float(
            bar.get_height()
        )

        if not np.isfinite(height):
            continue

        if height <= 0:
            continue

        axis.annotate(
            f"{height:.1f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# =============================================================================
# 5. LOAD CELL-LEVEL DATA
# =============================================================================

if not CELL_COORD_TSV.exists():
    raise FileNotFoundError(
        "The cell-level coordinate table was not found:\n"
        f"{CELL_COORD_TSV}"
    )

cells = pd.read_csv(
    CELL_COORD_TSV,
    sep="\t",
    low_memory=False,
)

required_columns = {
    "sample_id",
    "new_cluster",
}

missing_columns = (
    required_columns
    .difference(cells.columns)
)

if missing_columns:
    raise KeyError(
        f"{CELL_COORD_TSV.name} is missing required columns: "
        f"{sorted(missing_columns)}\n"
        f"Available columns: {cells.columns.tolist()}"
    )

cells["sample_id"] = (
    cells["sample_id"]
    .astype(str)
    .str.strip()
)

cells["sample_number"] = (
    cells["sample_id"]
    .map(extract_sample_number)
    .astype(np.int64)
)

cells["cluster"] = (
    cells["new_cluster"]
    .map(normalize_cluster)
)

cells["histology"] = (
    cells["sample_number"]
    .map(SAMPLE_HISTOLOGY)
)


# =============================================================================
# 6. CREATE OR VALIDATE A UNIQUE CELL KEY
# =============================================================================

if "cell_key" in cells.columns:
    cells["cell_unique_key"] = (
        cells["cell_key"]
        .astype(str)
    )

elif "cell_id" in cells.columns:
    cells["cell_unique_key"] = (
        cells["sample_id"].astype(str)
        + "::"
        + cells["cell_id"].astype(str)
    )

else:
    # As a final fallback, each input row is treated as one cell.
    cells["cell_unique_key"] = (
        cells.index
        .astype(str)
    )

duplicated_cells = (
    cells["cell_unique_key"]
    .duplicated()
)

if duplicated_cells.any():
    duplicate_examples = (
        cells.loc[
            duplicated_cells,
            "cell_unique_key",
        ]
        .head(10)
        .tolist()
    )

    raise ValueError(
        "Duplicate cell identifiers were detected. "
        f"Examples: {duplicate_examples}"
    )


# =============================================================================
# 7. VALIDATE SAMPLE STRUCTURE
# =============================================================================

observed_sample_numbers = sorted(
    cells["sample_number"]
    .unique()
    .tolist()
)

if 7 in observed_sample_numbers:
    raise ValueError(
        "Sample-7 unexpectedly entered the statistics."
    )

if observed_sample_numbers != EXPECTED_SAMPLE_NUMBERS:
    raise ValueError(
        "Observed samples differ from the expected eight samples.\n"
        f"Expected: {EXPECTED_SAMPLE_NUMBERS}\n"
        f"Observed: {observed_sample_numbers}"
    )

if cells["histology"].isna().any():
    missing_histology_samples = sorted(
        cells.loc[
            cells["histology"].isna(),
            "sample_number",
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Histological labels are missing for sample numbers: "
        f"{missing_histology_samples}"
    )


# =============================================================================
# 8. ATTACH CELL-TYPE ANNOTATIONS
# =============================================================================

cells["celltype_annotation"] = (
    cells["cluster"]
    .map(CLUSTER_ANNOTATION)
)

unknown_clusters = sorted(
    cells.loc[
        cells["celltype_annotation"].isna(),
        "cluster",
    ]
    .astype(str)
    .unique()
    .tolist()
)

if unknown_clusters:
    raise ValueError(
        "The input contains clusters that are not covered by "
        f"CLUSTER_ANNOTATION: {unknown_clusters}"
    )

cells["vascular_class"] = "Other"

cells.loc[
    cells["cluster"].isin(
        ENDOTHELIAL_CLUSTER_IDS
    ),
    "vascular_class",
] = "Endothelial cells"

cells.loc[
    cells["cluster"].isin(
        PERICYTE_CLUSTER_IDS
    ),
    "vascular_class",
] = "Pericytes"

endothelial_cluster_to_subtype = {
    cluster_id: subtype
    for subtype, cluster_id
    in ENDOTHELIAL_SUBTYPE_CLUSTERS.items()
}

cells["endothelial_subtype"] = (
    cells["cluster"]
    .map(endothelial_cluster_to_subtype)
)


# =============================================================================
# 9. BUILD SAMPLE REFERENCE TABLE
# =============================================================================

sample_id_table = (
    cells[
        [
            "sample_number",
            "sample_id",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

duplicated_sample_numbers = (
    sample_id_table["sample_number"]
    .duplicated(keep=False)
)

if duplicated_sample_numbers.any():
    duplicated_table = (
        sample_id_table.loc[
            duplicated_sample_numbers
        ]
        .sort_values(
            by=[
                "sample_number",
                "sample_id",
            ],
            kind="stable",
        )
    )

    raise ValueError(
        "More than one sample_id is associated with the same sample number:\n"
        + duplicated_table.to_string(index=False)
    )

sample_reference = pd.DataFrame(
    {
        "sample_number": EXPECTED_SAMPLE_NUMBERS,
    }
)

sample_reference = (
    sample_reference
    .merge(
        sample_id_table,
        on="sample_number",
        how="left",
        validate="one_to_one",
    )
)

if sample_reference["sample_id"].isna().any():
    missing_sample_ids = (
        sample_reference.loc[
            sample_reference["sample_id"].isna(),
            "sample_number",
        ]
        .tolist()
    )

    raise ValueError(
        "No sample_id was found for sample numbers: "
        f"{missing_sample_ids}"
    )

sample_reference["histology"] = (
    sample_reference["sample_number"]
    .map(SAMPLE_HISTOLOGY)
)

sample_reference["plot_label"] = (
    sample_reference.apply(
        lambda row: (
            f"Sample-{int(row['sample_number'])}\n"
            f"{row['histology']}"
        ),
        axis=1,
    )
)


# =============================================================================
# 10. COUNT ALL CELLS IN EACH SAMPLE
# =============================================================================

total_counts = (
    cells.groupby(
        "sample_number",
        observed=True,
    )
    .size()
    .reindex(
        EXPECTED_SAMPLE_NUMBERS,
        fill_value=0,
    )
    .astype(np.int64)
)

summary = (
    sample_reference
    .copy()
)

summary["n_total_cells"] = (
    summary["sample_number"]
    .map(total_counts)
    .fillna(0)
    .astype(np.int64)
)


# =============================================================================
# 11. COUNT ENDOTHELIAL SUBTYPES
# =============================================================================

endothelial_long_parts = []
endothelial_count_columns = []

for subtype, cluster_id in ENDOTHELIAL_SUBTYPE_CLUSTERS.items():

    marker_name = (
        subtype
        .split("+", 1)[0]
    )

    output_column = (
        f"n_{marker_name}_positive_endothelial"
    )

    endothelial_count_columns.append(
        output_column
    )

    subtype_counts = (
        cells.loc[
            cells["cluster"] == cluster_id
        ]
        .groupby(
            "sample_number",
            observed=True,
        )
        .size()
        .reindex(
            EXPECTED_SAMPLE_NUMBERS,
            fill_value=0,
        )
        .astype(np.int64)
    )

    summary[output_column] = (
        summary["sample_number"]
        .map(subtype_counts)
        .fillna(0)
        .astype(np.int64)
    )

    subtype_long_table = pd.DataFrame(
        {
            "sample_number": EXPECTED_SAMPLE_NUMBERS,
            "endothelial_subtype": subtype,
            "cluster": cluster_id,
            "n_cells": [
                int(
                    subtype_counts.loc[
                        sample_number
                    ]
                )
                for sample_number
                in EXPECTED_SAMPLE_NUMBERS
            ],
        }
    )

    endothelial_long_parts.append(
        subtype_long_table
    )

endothelial_long = pd.concat(
    endothelial_long_parts,
    ignore_index=True,
)

endothelial_long = (
    endothelial_long
    .merge(
        sample_reference[
            [
                "sample_number",
                "sample_id",
                "histology",
            ]
        ],
        on="sample_number",
        how="left",
        validate="many_to_one",
    )
)

summary["n_endothelial_total"] = (
    summary[
        endothelial_count_columns
    ]
    .sum(axis=1)
    .astype(np.int64)
)


# =============================================================================
# 12. COUNT PERICYTES
# =============================================================================

pericyte_counts = (
    cells.loc[
        cells["cluster"].isin(
            PERICYTE_CLUSTER_IDS
        )
    ]
    .groupby(
        "sample_number",
        observed=True,
    )
    .size()
    .reindex(
        EXPECTED_SAMPLE_NUMBERS,
        fill_value=0,
    )
    .astype(np.int64)
)

summary["n_pericytes"] = (
    summary["sample_number"]
    .map(pericyte_counts)
    .fillna(0)
    .astype(np.int64)
)


# =============================================================================
# 13. CALCULATE PROPORTIONS AND RATIOS
# =============================================================================

summary["pct_endothelial"] = np.where(
    summary["n_total_cells"] > 0,
    (
        summary["n_endothelial_total"]
        / summary["n_total_cells"]
        * 100.0
    ),
    np.nan,
)

summary["pct_pericytes"] = np.where(
    summary["n_total_cells"] > 0,
    (
        summary["n_pericytes"]
        / summary["n_total_cells"]
        * 100.0
    ),
    np.nan,
)

summary["n_endothelial_and_pericytes"] = (
    summary["n_endothelial_total"]
    + summary["n_pericytes"]
)

summary["pct_endothelial_and_pericytes"] = np.where(
    summary["n_total_cells"] > 0,
    (
        summary["n_endothelial_and_pericytes"]
        / summary["n_total_cells"]
        * 100.0
    ),
    np.nan,
)

summary["endothelial_to_pericyte_ratio"] = np.where(
    summary["n_pericytes"] > 0,
    (
        summary["n_endothelial_total"]
        / summary["n_pericytes"]
    ),
    np.nan,
)


# =============================================================================
# 14. ALL-CLUSTER AUDIT TABLE
#
# Important correction:
# Do not use natural_sort_key returning lists inside pandas sort_values(key=...).
# Instead, construct a numeric helper column for safe sorting.
# =============================================================================

all_cluster_counts = (
    cells.groupby(
        [
            "sample_number",
            "sample_id",
            "histology",
            "cluster",
            "celltype_annotation",
            "vascular_class",
        ],
        observed=True,
        dropna=False,
    )
    .size()
    .rename("n_cells")
    .reset_index()
)

# Extract a numeric cluster value for sorting.
all_cluster_counts["cluster_sort_number"] = (
    all_cluster_counts["cluster"]
    .astype(str)
    .str.extract(
        r"(\d+)",
        expand=False,
    )
    .pipe(
        pd.to_numeric,
        errors="coerce",
    )
)

# Put non-numeric clusters at the end.
all_cluster_counts["cluster_sort_number"] = (
    all_cluster_counts["cluster_sort_number"]
    .fillna(np.inf)
)

all_cluster_counts = (
    all_cluster_counts
    .sort_values(
        by=[
            "sample_number",
            "cluster_sort_number",
            "cluster",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="stable",
    )
    .drop(
        columns=[
            "cluster_sort_number",
        ]
    )
    .reset_index(drop=True)
)


# =============================================================================
# 15. SAVE OUTPUT TABLES
# =============================================================================

summary_columns = [
    "sample_number",
    "sample_id",
    "histology",
    "n_total_cells",
    "n_SPARCL1_positive_endothelial",
    "n_IFI27_positive_endothelial",
    "n_CCL14_positive_endothelial",
    "n_endothelial_total",
    "pct_endothelial",
    "n_pericytes",
    "pct_pericytes",
    "n_endothelial_and_pericytes",
    "pct_endothelial_and_pericytes",
    "endothelial_to_pericyte_ratio",
]

summary[
    summary_columns
].to_csv(
    OUT_SUMMARY_TSV,
    sep="\t",
    index=False,
    na_rep="NA",
)

endothelial_long.to_csv(
    OUT_ENDOTHELIAL_LONG_TSV,
    sep="\t",
    index=False,
)

all_cluster_counts.to_csv(
    OUT_CLUSTER_COUNTS_TSV,
    sep="\t",
    index=False,
)

cell_metadata_columns = [
    "cell_unique_key",
    "sample_id",
    "sample_number",
    "histology",
    "cluster",
    "celltype_annotation",
    "vascular_class",
    "endothelial_subtype",
]

cells[
    cell_metadata_columns
].to_csv(
    OUT_CELL_METADATA_TSV_GZ,
    sep="\t",
    index=False,
    na_rep="NA",
    compression="gzip",
)


# =============================================================================
# 16. FIGURE 1
# STACKED ENDOTHELIAL SUBTYPE COUNTS
# =============================================================================

figure, axis = plt.subplots(
    figsize=(12.0, 6.3)
)

x_positions = np.arange(
    len(summary)
)

stack_bottom = np.zeros(
    len(summary),
    dtype=float,
)

for subtype, cluster_id in ENDOTHELIAL_SUBTYPE_CLUSTERS.items():

    marker_name = (
        subtype
        .split("+", 1)[0]
    )

    count_column = (
        f"n_{marker_name}_positive_endothelial"
    )

    values = (
        summary[count_column]
        .to_numpy(dtype=float)
    )

    axis.bar(
        x_positions,
        values,
        bottom=stack_bottom,
        width=0.68,
        color=ENDOTHELIAL_COLORS[subtype],
        edgecolor="white",
        linewidth=0.7,
        label=subtype,
    )

    stack_bottom += values

maximum_endothelial = max(
    float(
        summary["n_endothelial_total"]
        .max()
    ),
    1.0,
)

label_offset = (
    maximum_endothelial
    * 0.018
)

for position, total in zip(
    x_positions,
    summary["n_endothelial_total"]
    .to_numpy(dtype=float),
):
    axis.text(
        position,
        total + label_offset,
        f"{int(total):,}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

axis.set_xticks(
    x_positions
)

axis.set_xticklabels(
    summary["plot_label"],
    rotation=0,
    ha="center",
)

axis.set_ylabel(
    "Number of endothelial cells"
)

axis.set_xlabel("")

axis.set_title(
    "Endothelial cell composition across kidney cancer samples",
    pad=14,
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.28,
)

axis.set_axisbelow(True)

axis.legend(
    title="Endothelial subtype",
    frameon=False,
    bbox_to_anchor=(1.02, 1.0),
    loc="upper left",
)

axis.set_ylim(
    bottom=0,
    top=maximum_endothelial * 1.14,
)

axis.margins(
    x=0.025
)

figure.tight_layout()

save_figure_all_formats(
    figure,
    OUT_DIR
    / "endothelial_subtype_counts_by_sample",
)


# =============================================================================
# 17. FIGURE 2
# TOTAL ENDOTHELIAL CELLS VERSUS PERICYTES
# =============================================================================

figure, axis = plt.subplots(
    figsize=(12.0, 6.3)
)

x_positions = np.arange(
    len(summary)
)

bar_width = 0.36

endothelial_bars = axis.bar(
    x_positions - bar_width / 2,
    summary["n_endothelial_total"]
    .to_numpy(dtype=float),
    width=bar_width,
    color=TOTAL_ENDOTHELIAL_COLOR,
    edgecolor="white",
    linewidth=0.6,
    label="Endothelial cells",
)

pericyte_bars = axis.bar(
    x_positions + bar_width / 2,
    summary["n_pericytes"]
    .to_numpy(dtype=float),
    width=bar_width,
    color=PERICYTE_COLOR,
    edgecolor="white",
    linewidth=0.6,
    label="Pericytes",
)

add_count_labels(
    axis,
    endothelial_bars,
    fontsize=8,
)

add_count_labels(
    axis,
    pericyte_bars,
    fontsize=8,
)

axis.set_xticks(
    x_positions
)

axis.set_xticklabels(
    summary["plot_label"],
    rotation=0,
    ha="center",
)

axis.set_ylabel(
    "Number of cells"
)

axis.set_xlabel("")

axis.set_title(
    "Endothelial and pericyte cell counts across kidney cancer samples",
    pad=14,
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.28,
)

axis.set_axisbelow(True)

axis.legend(
    frameon=False,
    loc="upper left",
)

maximum_count = max(
    float(
        summary[
            [
                "n_endothelial_total",
                "n_pericytes",
            ]
        ]
        .to_numpy(dtype=float)
        .max()
    ),
    1.0,
)

axis.set_ylim(
    bottom=0,
    top=maximum_count * 1.16,
)

axis.margins(
    x=0.025
)

figure.tight_layout()

save_figure_all_formats(
    figure,
    OUT_DIR
    / "endothelial_vs_pericyte_counts_by_sample",
)


# =============================================================================
# 18. FIGURE 3
# ENDOTHELIAL AND PERICYTE PERCENTAGES
# =============================================================================

figure, axis = plt.subplots(
    figsize=(12.0, 6.3)
)

x_positions = np.arange(
    len(summary)
)

bar_width = 0.36

endothelial_percentage_bars = axis.bar(
    x_positions - bar_width / 2,
    summary["pct_endothelial"]
    .to_numpy(dtype=float),
    width=bar_width,
    color=TOTAL_ENDOTHELIAL_COLOR,
    edgecolor="white",
    linewidth=0.6,
    label="Endothelial cells",
)

pericyte_percentage_bars = axis.bar(
    x_positions + bar_width / 2,
    summary["pct_pericytes"]
    .to_numpy(dtype=float),
    width=bar_width,
    color=PERICYTE_COLOR,
    edgecolor="white",
    linewidth=0.6,
    label="Pericytes",
)

add_percentage_labels(
    axis,
    endothelial_percentage_bars,
)

add_percentage_labels(
    axis,
    pericyte_percentage_bars,
)

axis.set_xticks(
    x_positions
)

axis.set_xticklabels(
    summary["plot_label"],
    rotation=0,
    ha="center",
)

axis.set_ylabel(
    "Percentage of all cells (%)"
)

axis.set_xlabel("")

axis.set_title(
    "Endothelial and pericyte fractions across kidney cancer samples",
    pad=14,
)

axis.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.28,
)

axis.set_axisbelow(True)

axis.legend(
    frameon=False,
    loc="upper left",
)

maximum_percentage = max(
    float(
        summary[
            [
                "pct_endothelial",
                "pct_pericytes",
            ]
        ]
        .to_numpy(dtype=float)
        .max()
    ),
    1.0,
)

axis.set_ylim(
    bottom=0,
    top=maximum_percentage * 1.18,
)

axis.margins(
    x=0.025
)

figure.tight_layout()

save_figure_all_formats(
    figure,
    OUT_DIR
    / "endothelial_vs_pericyte_percentages_by_sample",
)


# =============================================================================
# 19. PRINT RESULTS
# =============================================================================

print(
    "\n"
    + "=" * 125
)

print(
    "[DONE] Endothelial and pericyte statistics"
)

print(
    "=" * 125
)

print(
    summary[
        [
            "sample_number",
            "sample_id",
            "histology",
            "n_total_cells",
            "n_SPARCL1_positive_endothelial",
            "n_IFI27_positive_endothelial",
            "n_CCL14_positive_endothelial",
            "n_endothelial_total",
            "pct_endothelial",
            "n_pericytes",
            "pct_pericytes",
            "endothelial_to_pericyte_ratio",
        ]
    ].to_string(
        index=False,
        na_rep="NA",
        formatters={
            "pct_endothelial": (
                lambda value:
                "NA"
                if pd.isna(value)
                else f"{value:.3f}"
            ),
            "pct_pericytes": (
                lambda value:
                "NA"
                if pd.isna(value)
                else f"{value:.3f}"
            ),
            "endothelial_to_pericyte_ratio": (
                lambda value:
                "NA"
                if pd.isna(value)
                else f"{value:.3f}"
            ),
        },
    )
)

print("\n[UPDATED ANNOTATION]")
print("Cluster 2 = Pericytes")

print("\n[INPUT]")
print(CELL_COORD_TSV)

print("\n[OUTPUT DIRECTORY]")
print(OUT_DIR)

print("\n[OUTPUT TABLES]")
print(OUT_SUMMARY_TSV)
print(OUT_ENDOTHELIAL_LONG_TSV)
print(OUT_CLUSTER_COUNTS_TSV)
print(OUT_CELL_METADATA_TSV_GZ)

print("\n[OUTPUT FIGURES]")

figure_names = [
    "endothelial_subtype_counts_by_sample",
    "endothelial_vs_pericyte_counts_by_sample",
    "endothelial_vs_pericyte_percentages_by_sample",
]

for figure_name in figure_names:
    print(
        OUT_DIR
        / f"{figure_name}.png"
    )

    print(
        OUT_DIR
        / f"{figure_name}.svg"
    )

    print(
        OUT_DIR
        / f"{figure_name}.pdf"
    )

print(
    "\n"
    + "=" * 125
)