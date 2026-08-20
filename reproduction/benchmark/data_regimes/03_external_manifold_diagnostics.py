# -*- coding: utf-8 -*-
"""
External benchmark-like diagnostics for IMAS DepStruct comparison
-----------------------------------------------------------------
This script makes NON-UMAP / NON-heatmap companion figures that are independent
from the DepStruct formula.

Inputs expected from the previous benchmark-UMAP / DepStruct script:
    OUT_ROOT/<dataset>_<regime>_cell_structure_scores_with_DepStruct.csv
    OUT_ROOT/<dataset>_<regime>_cluster_top2000_markers_long.csv   optional, for marker separability

Main outputs:
    01_external_benchmark_diagnostics_with_pvalues.{png,pdf,svg}
    02_depstruct_vs_external_diagnostics.{png,pdf,svg}
    external_cluster_level_diagnostics.csv
    external_dataset_level_diagnostics.csv
    external_pairwise_pvalues.csv
    external_effect_sizes.csv

Metrics:
    1. ASW / silhouette on the regime-specific UMAP manifold
    2. kNN purity on the regime-specific UMAP manifold
    3. graph connectivity of each Leiden cluster in the kNN graph
    4. marker separability from one-vs-rest cluster marker statistics

These metrics are not used to define DepStruct. They are benchmark-like external
diagnostics to test whether DepStruct advantages are accompanied by more coherent
manifold structure and marker separability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================
# 0. USER CONFIG
# ============================================================

OUT_ROOT_CANDIDATES = [
    get_runs_root() / "frontier_transfer_hash_stageB_data_regime_comparison" / "benchmark_umap_dependency_structure_posthoc",
    Path.cwd(),
]

# Figure-level dataset order. CRC/Glioma are expected to have been computed already;
# Osteosarcoma is added here as the dataset to be supplemented.
DATASETS_REQUESTED = ["CRC-Tissue", "Glioma-Tissue", "Osteosarcoma-Tissue"]

# Only these datasets are newly computed when their external diagnostics are absent.
# Existing CRC/Glioma external diagnostics are reused instead of recomputing ASW/kNN/graph metrics.
DATASETS_TO_CALCULATE_REQUESTED = ["Osteosarcoma-Tissue"]

# Reuse previously generated external_cluster_level_diagnostics.csv when possible.
REUSE_EXISTING_EXTERNAL_DIAGNOSTICS = True

# Name aliases for the Osteosarcoma output folder/prefix. The first existing alias will be used.
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

REGIMES = ["multiomic_only", "mixed", "scrna_only"]

REGIME_LABEL = {
    "multiomic_only": "Multiomic-only",
    "mixed": "Mixed",
    "scrna_only": "scRNA-only",
}

REGIME_COLOR = {
    "multiomic_only": "#2ca25f",  # green
    "mixed": "#8c6bb1",           # purple
    "scrna_only": "#bdbdbd",      # grey
}

METRIC_LABEL = {
    "asw_silhouette": "ASW / silhouette",
    "knn_purity": "kNN purity",
    "graph_connectivity": "Graph connectivity",
    "marker_separability": "Marker separability",
}

METRIC_ORDER = [
    "asw_silhouette",
    "knn_purity",
    "graph_connectivity",
    "marker_separability",
]

# k for kNN purity and graph connectivity
KNN_K = 15

# cap silhouette_samples to keep runtime stable; purity/connectivity still uses all cells by default
MAX_CELLS_FOR_SILHOUETTE = 8000

# marker separability uses top N ranked marker genes per cluster
TOP_MARKERS_FOR_SEPARABILITY = 100

# plotting
POINT_ALPHA = 0.74
POINT_SIZE_BASE = 22
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.85,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ============================================================
# 1. DEPENDENCIES
# ============================================================

def _install_if_missing(pkg: str, import_name: Optional[str] = None):
    import importlib
    import subprocess
    import sys
    name = import_name or pkg
    try:
        return importlib.import_module(name)
    except Exception:
        print(f"[INSTALL] {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(name)

sk_metrics = _install_if_missing("scikit-learn", "sklearn.metrics")
sk_neighbors = _install_if_missing("scikit-learn", "sklearn.neighbors")
scipy_stats = _install_if_missing("scipy", "scipy.stats")
scipy_sparse = _install_if_missing("scipy", "scipy.sparse")

silhouette_samples = sk_metrics.silhouette_samples
NearestNeighbors = sk_neighbors.NearestNeighbors
mannwhitneyu = scipy_stats.mannwhitneyu
coo_matrix = scipy_sparse.coo_matrix
csgraph = _install_if_missing("scipy", "scipy.sparse.csgraph")
connected_components = csgraph.connected_components

# ============================================================
# 2. BASIC HELPERS
# ============================================================

def detect_out_root() -> Path:
    for p in OUT_ROOT_CANDIDATES:
        if not p.exists():
            continue
        hits = list(p.glob("*_cell_structure_scores_with_DepStruct.csv"))
        if len(hits) > 0:
            print("[FOUND OUT_ROOT]", p)
            return p
    raise FileNotFoundError(
        "Cannot detect OUT_ROOT automatically.\n"
        "Please edit OUT_ROOT_CANDIDATES to your benchmark_umap_dependency_structure_posthoc folder."
    )

OUT_ROOT = detect_out_root()


def resolve_dataset_name(requested: str, require_all_regimes: bool = False) -> str:
    """Resolve a requested dataset label to the actual filename prefix under OUT_ROOT."""
    candidates = DATASET_ALIASES.get(requested, [requested])
    candidates = list(dict.fromkeys([requested] + candidates))
    for cand in candidates:
        exists = [
            (OUT_ROOT / f"{cand}_{reg}_cell_structure_scores_with_DepStruct.csv").exists()
            for reg in REGIMES
        ]
        if all(exists) or ((not require_all_regimes) and any(exists)):
            if cand != requested:
                print(f"[DATASET ALIAS] {requested} -> {cand}")
            return cand
    print(f"[DATASET ALIAS][WARN] cannot resolve {requested}; using requested name directly")
    return requested


DATASETS = [resolve_dataset_name(ds, require_all_regimes=False) for ds in DATASETS_REQUESTED]
DATASETS_TO_CALCULATE = [resolve_dataset_name(ds, require_all_regimes=False) for ds in DATASETS_TO_CALCULATE_REQUESTED]
DATASETS_TO_CALCULATE = list(dict.fromkeys(DATASETS_TO_CALCULATE))
print("[DATASETS_FOR_FIGURE]", DATASETS)
print("[DATASETS_TO_CALCULATE]", DATASETS_TO_CALCULATE)


def nice_dataset_label(x: str) -> str:
    x = str(x)
    return (
        x.replace("-Tissue", "")
         .replace("-Cellline", "")
         .replace("-CellLine", "")
    )


def savefig_multi(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=420 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("[SAVE]", out)
    plt.close(fig)


def pick_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    low_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in low_map:
            return low_map[a.lower()]
    return None


def jitter(n: int, scale: float = 0.08) -> np.ndarray:
    return RNG.uniform(-scale, scale, int(n))


def q_to_star(q: float) -> str:
    if not np.isfinite(q):
        return "n.s."
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 5e-2:
        return "*"
    return "n.s."


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return q
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(m) + 1.0)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    tmp = np.empty_like(q_ranked)
    tmp[order] = q_ranked
    q[ok] = tmp
    return q


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # efficient enough for cluster-level small N; fallback sampling if huge
    if len(x) * len(y) > 2_000_000:
        rng = np.random.default_rng(42)
        x = rng.choice(x, size=min(len(x), 1500), replace=False)
        y = rng.choice(y, size=min(len(y), 1500), replace=False)
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def numeric01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.clip(x, 0.0, 1.0)

# ============================================================
# 3. LOAD INPUT TABLES
# ============================================================

def load_cell_table(dataset: str, regime: str) -> Optional[pd.DataFrame]:
    f = OUT_ROOT / f"{dataset}_{regime}_cell_structure_scores_with_DepStruct.csv"
    if not f.exists():
        print("[WARN missing cell score]", f)
        return None
    df = pd.read_csv(f)
    df["dataset"] = dataset
    df["data_regime"] = regime
    return df


def load_marker_table(dataset: str, regime: str) -> Optional[pd.DataFrame]:
    # v3/v4 marker file name
    candidates = [
        OUT_ROOT / f"{dataset}_{regime}_cluster_top2000_markers_long.csv",
        OUT_ROOT / f"{dataset}_{regime}_cluster_top500_markers_long.csv",
    ]
    for f in candidates:
        if f.exists():
            df = pd.read_csv(f)
            df["dataset"] = dataset
            df["data_regime"] = regime
            return df
    print("[WARN missing marker table]", candidates[0])
    return None

# ============================================================
# 4. EXTERNAL DIAGNOSTICS
# ============================================================

def compute_asw_cluster(tab: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    coords = tab[["umap_1", "umap_2"]].astype(float).values
    labels = tab[cluster_col].astype(str).values
    ok = np.isfinite(coords).all(axis=1)

    out = pd.DataFrame({
        "cell_index": np.arange(len(tab)),
        "new_leiden": labels,
        "asw_silhouette_cell": np.nan,
    })

    coords_ok = coords[ok]
    labels_ok = labels[ok]
    if coords_ok.shape[0] < 20 or len(np.unique(labels_ok)) < 2:
        return out.groupby("new_leiden", as_index=False).agg(asw_silhouette=("asw_silhouette_cell", "mean"))

    # cap if too many cells
    ok_idx = np.where(ok)[0]
    if coords_ok.shape[0] > MAX_CELLS_FOR_SILHOUETTE:
        use_idx_rel = RNG.choice(np.arange(coords_ok.shape[0]), size=MAX_CELLS_FOR_SILHOUETTE, replace=False)
        coords_s = coords_ok[use_idx_rel]
        labels_s = labels_ok[use_idx_rel]
        abs_idx = ok_idx[use_idx_rel]
    else:
        coords_s = coords_ok
        labels_s = labels_ok
        abs_idx = ok_idx

    try:
        s = silhouette_samples(coords_s, labels_s, metric="euclidean")
        # Convert from [-1, 1] to [0, 1] for easier comparison
        s01 = 0.5 * (s + 1.0)
        out.loc[abs_idx, "asw_silhouette_cell"] = s01
    except Exception as e:
        print("[ASW][WARN]", repr(e))

    return out.groupby("new_leiden", as_index=False).agg(asw_silhouette=("asw_silhouette_cell", "mean"))


def compute_knn_purity_cluster(tab: pd.DataFrame, cluster_col: str, k: int = KNN_K) -> pd.DataFrame:
    coords = tab[["umap_1", "umap_2"]].astype(float).values
    labels = tab[cluster_col].astype(str).values
    ok = np.isfinite(coords).all(axis=1)
    purity = np.full(len(tab), np.nan, dtype=float)

    coords_ok = coords[ok]
    labels_ok = labels[ok]
    ok_idx = np.where(ok)[0]
    if coords_ok.shape[0] < max(5, k + 2):
        return pd.DataFrame({"new_leiden": sorted(np.unique(labels)), "knn_purity": np.nan})

    try:
        kk = min(int(k), coords_ok.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean").fit(coords_ok)
        nn = nbrs.kneighbors(return_distance=False)[:, 1:]
        same = (labels_ok[nn] == labels_ok[:, None]).mean(axis=1)
        purity[ok_idx] = same
    except Exception as e:
        print("[KNN purity][WARN]", repr(e))

    tmp = pd.DataFrame({"new_leiden": labels, "knn_purity_cell": purity})
    return tmp.groupby("new_leiden", as_index=False).agg(knn_purity=("knn_purity_cell", "mean"))


def compute_graph_connectivity_cluster(tab: pd.DataFrame, cluster_col: str, k: int = KNN_K) -> pd.DataFrame:
    coords = tab[["umap_1", "umap_2"]].astype(float).values
    labels = tab[cluster_col].astype(str).values
    ok = np.isfinite(coords).all(axis=1)
    coords_ok = coords[ok]
    labels_ok = labels[ok]

    rows = []
    if coords_ok.shape[0] < max(5, k + 2):
        for cl in sorted(np.unique(labels)):
            rows.append({"new_leiden": str(cl), "graph_connectivity": np.nan})
        return pd.DataFrame(rows)

    try:
        kk = min(int(k), coords_ok.shape[0] - 1)
        nbrs = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean").fit(coords_ok)
        nn = nbrs.kneighbors(return_distance=False)[:, 1:]
        row = np.repeat(np.arange(coords_ok.shape[0]), kk)
        col = nn.reshape(-1)
        data = np.ones_like(row, dtype=np.float32)
        A = coo_matrix((data, (row, col)), shape=(coords_ok.shape[0], coords_ok.shape[0])).tocsr()
        A = A.maximum(A.T)

        for cl in sorted(np.unique(labels_ok), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            idx = np.where(labels_ok == cl)[0]
            if len(idx) <= 1:
                conn = 1.0
            else:
                subA = A[idx, :][:, idx]
                n_comp, comp_lab = connected_components(subA, directed=False, return_labels=True)
                if n_comp <= 0:
                    conn = np.nan
                else:
                    cnt = np.bincount(comp_lab)
                    conn = float(cnt.max() / max(1, len(idx)))
            rows.append({"new_leiden": str(cl), "graph_connectivity": conn})
    except Exception as e:
        print("[Graph connectivity][WARN]", repr(e))
        for cl in sorted(np.unique(labels)):
            rows.append({"new_leiden": str(cl), "graph_connectivity": np.nan})

    return pd.DataFrame(rows)


def compute_marker_separability_cluster(marker_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if marker_df is None or len(marker_df) == 0:
        return pd.DataFrame(columns=["new_leiden", "marker_separability"])

    cl_col = pick_col(marker_df, ["new_leiden", "cluster", "cluster_id", "leiden"])
    if cl_col is None:
        return pd.DataFrame(columns=["new_leiden", "marker_separability"])

    rank_col = pick_col(marker_df, ["rank", "Rank"])
    logfc_col = pick_col(marker_df, ["logfoldchange", "logfoldchanges", "log2fc", "logFC"])
    p_col = pick_col(marker_df, ["pvals_adj", "padj", "FDR", "adj_p", "p_val_adj"])
    score_col = pick_col(marker_df, ["score", "scores", "zscore"])

    rows = []
    for cl, sub in marker_df.groupby(marker_df[cl_col].astype(str), sort=False):
        ss = sub.copy()
        if rank_col is not None:
            ss = ss.sort_values(rank_col, ascending=True)
        elif score_col is not None:
            ss = ss.sort_values(score_col, ascending=False)
        ss = ss.head(int(TOP_MARKERS_FOR_SEPARABILITY)).copy()

        vals = []
        if logfc_col is not None:
            lfc = pd.to_numeric(ss[logfc_col], errors="coerce").values.astype(float)
            lfc_pos = np.clip(lfc, 0, None)
            # logFC component: saturate around 2.0
            vals.append(np.nanmean(np.clip(lfc_pos / 2.0, 0, 1)))
        if p_col is not None:
            p = pd.to_numeric(ss[p_col], errors="coerce").values.astype(float)
            sig = -np.log10(np.clip(p, 1e-300, 1.0))
            # significance component: q=1e-10 already saturated
            vals.append(np.nanmean(np.clip(sig / 10.0, 0, 1)))
        if score_col is not None:
            sc = pd.to_numeric(ss[score_col], errors="coerce").values.astype(float)
            sc = sc[np.isfinite(sc)]
            if sc.size > 0:
                # rank statistic component, robustly squashed
                vals.append(float(1.0 / (1.0 + np.exp(-np.nanmedian(sc) / 3.0))))

        if len(vals) == 0 or not np.isfinite(vals).any():
            sep = np.nan
        else:
            sep = float(np.nanmean(vals))
        rows.append({"new_leiden": str(cl), "marker_separability": sep})

    return pd.DataFrame(rows)


def compute_external_for_one(dataset: str, regime: str) -> Optional[pd.DataFrame]:
    tab = load_cell_table(dataset, regime)
    if tab is None or len(tab) == 0:
        return None

    cluster_col = pick_col(tab, ["new_leiden", "cluster", "leiden"])
    if cluster_col is None:
        raise ValueError(f"Cannot find cluster column for {dataset}/{regime}")
    if "umap_1" not in tab.columns or "umap_2" not in tab.columns:
        raise ValueError(f"Need umap_1/umap_2 in {dataset}/{regime} cell score table")

    base = (
        tab.assign(new_leiden=tab[cluster_col].astype(str))
        .groupby("new_leiden", as_index=False)
        .agg(n_cells=("new_leiden", "size"))
    )

    dep_col = pick_col(tab, ["depstruct_score_cell", "DepStruct", "depstruct_score"])
    if dep_col is not None:
        dep = (
            tab.assign(new_leiden=tab[cluster_col].astype(str))
            .groupby("new_leiden", as_index=False)
            .agg(depstruct_mean=(dep_col, "mean"))
        )
        base = base.merge(dep, on="new_leiden", how="left")
    else:
        base["depstruct_mean"] = np.nan

    asw = compute_asw_cluster(tab, cluster_col)
    pur = compute_knn_purity_cluster(tab, cluster_col, KNN_K)
    conn = compute_graph_connectivity_cluster(tab, cluster_col, KNN_K)
    markers = compute_marker_separability_cluster(load_marker_table(dataset, regime))

    out = base.merge(asw, on="new_leiden", how="left")
    out = out.merge(pur, on="new_leiden", how="left")
    out = out.merge(conn, on="new_leiden", how="left")
    out = out.merge(markers, on="new_leiden", how="left")

    out["dataset"] = dataset
    out["data_regime"] = regime
    return out


def _standard_external_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "data_regime", "new_leiden", "n_cells", "depstruct_mean"] + METRIC_ORDER
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out["dataset"] = out["dataset"].astype(str)
    out["data_regime"] = out["data_regime"].astype(str)
    out["new_leiden"] = out["new_leiden"].astype(str)
    return out[cols]


def _existing_external_table() -> Optional[pd.DataFrame]:
    f = OUT_ROOT / "external_cluster_level_diagnostics.csv"
    if not (REUSE_EXISTING_EXTERNAL_DIAGNOSTICS and f.exists()):
        return None
    try:
        df = pd.read_csv(f)
        df = _standard_external_columns(df)
        df = df[df["dataset"].isin(DATASETS) & df["data_regime"].isin(REGIMES)].copy()
        if len(df) == 0:
            return None
        print("[REUSE external_cluster_level_diagnostics]", f, "rows=", len(df))
        return df
    except Exception as e:
        print("[REUSE][WARN] cannot read existing external diagnostics:", repr(e))
        return None


def _dataset_complete_in_external(df: Optional[pd.DataFrame], dataset: str) -> bool:
    if df is None or len(df) == 0:
        return False
    sub = df[df["dataset"].astype(str) == str(dataset)]
    if len(sub) == 0:
        return False
    for reg in REGIMES:
        rr = sub[sub["data_regime"].astype(str) == str(reg)]
        if len(rr) == 0:
            return False
        if rr["new_leiden"].nunique() == 0:
            return False
    return True


def compute_all_external() -> pd.DataFrame:
    existing = _existing_external_table()
    tables = []
    used_existing = set()

    for ds in DATASETS:
        if _dataset_complete_in_external(existing, ds):
            print(f"[SKIP/REUSE] {ds}: use existing external diagnostics")
            tables.append(existing[existing["dataset"].astype(str) == str(ds)].copy())
            used_existing.add(ds)
            continue

        if ds not in DATASETS_TO_CALCULATE:
            print(f"[SKIP][WARN] {ds}: existing external diagnostics not complete, "
                  f"but dataset is not in DATASETS_TO_CALCULATE; skipping recomputation")
            continue

        for reg in REGIMES:
            print(f"[COMPUTE] {ds}/{reg}")
            df = compute_external_for_one(ds, reg)
            if df is not None and len(df) > 0:
                tables.append(_standard_external_columns(df))

    if len(tables) == 0:
        raise RuntimeError("No external diagnostic tables could be reused or computed")

    all_df = pd.concat(tables, axis=0, ignore_index=True)
    all_df = _standard_external_columns(all_df)
    all_df = all_df.drop_duplicates(["dataset", "data_regime", "new_leiden"], keep="last")
    all_df = all_df[all_df["dataset"].isin(DATASETS) & all_df["data_regime"].isin(REGIMES)].copy()
    all_df["dataset"] = pd.Categorical(all_df["dataset"], categories=DATASETS, ordered=True)
    all_df["data_regime"] = pd.Categorical(all_df["data_regime"], categories=REGIMES, ordered=True)
    all_df = all_df.sort_values(["dataset", "data_regime", "new_leiden"]).reset_index(drop=True)
    all_df["dataset"] = all_df["dataset"].astype(str)
    all_df["data_regime"] = all_df["data_regime"].astype(str)
    return all_df


external_df = compute_all_external()
external_df.to_csv(OUT_ROOT / "external_cluster_level_diagnostics.csv", index=False)
print("[SAVE]", OUT_ROOT / "external_cluster_level_diagnostics.csv")

# dataset-level summary: weighted mean over clusters by cluster size
summary_rows = []
for (ds, reg), sub in external_df.groupby(["dataset", "data_regime"], sort=False):
    row = {"dataset": ds, "data_regime": reg, "n_clusters": int(len(sub)), "n_cells": int(sub["n_cells"].sum())}
    w = sub["n_cells"].astype(float).values
    for m in METRIC_ORDER + ["depstruct_mean"]:
        x = pd.to_numeric(sub[m], errors="coerce").values.astype(float)
        ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
        row[m + "_weighted_mean"] = float(np.average(x[ok], weights=w[ok])) if ok.any() else np.nan
        row[m + "_median"] = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_ROOT / "external_dataset_level_diagnostics.csv", index=False)
print("[SAVE]", OUT_ROOT / "external_dataset_level_diagnostics.csv")

# ============================================================
# 5. PAIRWISE TESTS + EFFECT SIZES
# ============================================================

def compute_pairwise_tests(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [("multiomic_only", "mixed"), ("multiomic_only", "scrna_only"), ("mixed", "scrna_only")]
    rows = []
    eff_rows = []
    for ds in DATASETS:
        for metric in METRIC_ORDER:
            local_idx = []
            for a, b in pairs:
                xa = pd.to_numeric(df[(df["dataset"] == ds) & (df["data_regime"] == a)][metric], errors="coerce").dropna().values
                xb = pd.to_numeric(df[(df["dataset"] == ds) & (df["data_regime"] == b)][metric], errors="coerce").dropna().values
                if len(xa) >= 2 and len(xb) >= 2:
                    try:
                        p = float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
                    except Exception:
                        p = np.nan
                    delta = cliffs_delta(xa, xb)
                    med_a = float(np.nanmedian(xa))
                    med_b = float(np.nanmedian(xb))
                    diff = med_a - med_b
                else:
                    p, delta, med_a, med_b, diff = np.nan, np.nan, np.nan, np.nan, np.nan
                rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "metric_label": METRIC_LABEL[metric],
                    "group_a": a,
                    "group_b": b,
                    "n_a": int(len(xa)),
                    "n_b": int(len(xb)),
                    "median_a": med_a,
                    "median_b": med_b,
                    "median_diff_a_minus_b": diff,
                    "p_value": p,
                    "cliffs_delta_a_vs_b": delta,
                })
                local_idx.append(len(rows) - 1)
                eff_rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "metric_label": METRIC_LABEL[metric],
                    "contrast": f"{REGIME_LABEL[a]} vs {REGIME_LABEL[b]}",
                    "effect_median_diff": diff,
                    "cliffs_delta": delta,
                })
            # BH-FDR within dataset x metric across the 3 pairwise tests
            pvals = np.array([rows[i]["p_value"] for i in local_idx], dtype=float)
            qvals = bh_fdr(pvals)
            for i, q in zip(local_idx, qvals):
                rows[i]["q_value_BH_within_dataset_metric"] = q
                rows[i]["significance"] = q_to_star(q)
    return pd.DataFrame(rows), pd.DataFrame(eff_rows)

pval_df, effect_df = compute_pairwise_tests(external_df)
pval_df.to_csv(OUT_ROOT / "external_pairwise_pvalues.csv", index=False)
effect_df.to_csv(OUT_ROOT / "external_effect_sizes.csv", index=False)
print("[SAVE]", OUT_ROOT / "external_pairwise_pvalues.csv")
print("[SAVE]", OUT_ROOT / "external_effect_sizes.csv")

# ============================================================
# 6. PLOTTING
# ============================================================

def _add_pvalue_bracket(ax, x1, x2, y, h, label, color="black"):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=8, color=color, clip_on=False)


def plot_external_diagnostics(df: pd.DataFrame, pvals: pd.DataFrame):
    nrow = len(DATASETS)
    ncol = len(METRIC_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.25 * nrow), facecolor="white", squeeze=False)

    x_pos = {r: i for i, r in enumerate(REGIMES)}

    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(METRIC_ORDER):
            ax = axes[i, j]
            sub = df[df["dataset"] == ds].copy()

            ymax = 0.0
            ymin = 1.0
            for reg in REGIMES:
                ss = sub[sub["data_regime"] == reg].copy()
                y = pd.to_numeric(ss[metric], errors="coerce").values.astype(float)
                y = y[np.isfinite(y)]
                if len(y) == 0:
                    continue
                sizes = POINT_SIZE_BASE + 1.6 * np.sqrt(ss.loc[np.isfinite(pd.to_numeric(ss[metric], errors="coerce")), "n_cells"].astype(float).values)
                x = np.full(len(y), x_pos[reg], dtype=float) + jitter(len(y), 0.075)
                ax.scatter(
                    x, y,
                    s=sizes,
                    color=REGIME_COLOR[reg],
                    alpha=POINT_ALPHA,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
                med = np.nanmedian(y)
                q1 = np.nanpercentile(y, 25)
                q3 = np.nanpercentile(y, 75)
                ax.plot([x_pos[reg], x_pos[reg]], [q1, q3], color="black", lw=1.1, zorder=4)
                ax.plot([x_pos[reg] - 0.11, x_pos[reg] + 0.11], [med, med], color="black", lw=2.0, zorder=5)
                ymax = max(ymax, float(np.nanmax(y)))
                ymin = min(ymin, float(np.nanmin(y)))

            # Pairwise brackets: show only Multiomic vs Mixed and Multiomic vs scRNA to avoid clutter
            pp = pvals[(pvals["dataset"] == ds) & (pvals["metric"] == metric)].copy()
            pp = pp[pp["group_a"].eq("multiomic_only")]
            if len(pp) > 0:
                yrange = max(0.05, ymax - ymin)
                base = min(1.03, ymax + 0.06 * yrange)
                step = 0.10 * yrange
                h = 0.025 * yrange
                for kk, (_, r) in enumerate(pp.iterrows()):
                    a, b = r["group_a"], r["group_b"]
                    label = str(r.get("significance", "n.s."))
                    yb = base + kk * step
                    _add_pvalue_bracket(ax, x_pos[a], x_pos[b], yb, h, label)

            ax.set_xticks([x_pos[r] for r in REGIMES])
            ax.set_xticklabels([REGIME_LABEL[r] for r in REGIMES], rotation=35, ha="right")
            ax.set_ylim(max(-0.03, ymin - 0.08), min(1.25, max(1.05, ymax + 0.25)))
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)
            ax.set_title(METRIC_LABEL[metric], fontweight="bold")
            if j == 0:
                ax.set_ylabel(nice_dataset_label(ds), fontweight="bold", fontsize=12)
            else:
                ax.set_ylabel("")

    handles = [Line2D([0], [0], marker="o", linestyle="", color=REGIME_COLOR[r], label=REGIME_LABEL[r], markersize=8) for r in REGIMES]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("External benchmark-like diagnostics of dependency manifolds", fontweight="bold", y=1.075)
    fig.text(
        0.5, -0.015,
        "Each point is one newly computed Leiden cluster. Significance: two-sided Mann–Whitney U test with BH-FDR correction within each dataset × metric.",
        ha="center", va="top", fontsize=9,
    )
    savefig_multi(fig, OUT_ROOT / "01_external_benchmark_diagnostics_with_pvalues")


def plot_depstruct_vs_external(df: pd.DataFrame):
    # Scatter relation between internal DepStruct and external diagnostics; useful as a sanity check.
    nrow = len(DATASETS)
    ncol = len(METRIC_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.25 * nrow), facecolor="white", squeeze=False)

    for i, ds in enumerate(DATASETS):
        for j, metric in enumerate(METRIC_ORDER):
            ax = axes[i, j]
            sub = df[df["dataset"] == ds].copy()
            for reg in REGIMES:
                ss = sub[sub["data_regime"] == reg].copy()
                x = pd.to_numeric(ss["depstruct_mean"], errors="coerce").values.astype(float)
                y = pd.to_numeric(ss[metric], errors="coerce").values.astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() == 0:
                    continue
                ax.scatter(
                    x[ok], y[ok],
                    s=POINT_SIZE_BASE + 1.3 * np.sqrt(ss.loc[ok, "n_cells"].astype(float).values),
                    color=REGIME_COLOR[reg],
                    alpha=0.70,
                    edgecolor="white",
                    linewidth=0.45,
                    label=REGIME_LABEL[reg] if (i == 0 and j == 0) else None,
                )
            # robust trend line across all points
            xall = pd.to_numeric(sub["depstruct_mean"], errors="coerce").values.astype(float)
            yall = pd.to_numeric(sub[metric], errors="coerce").values.astype(float)
            ok = np.isfinite(xall) & np.isfinite(yall)
            if ok.sum() >= 4:
                try:
                    coef = np.polyfit(xall[ok], yall[ok], deg=1)
                    xx = np.linspace(np.nanmin(xall[ok]), np.nanmax(xall[ok]), 50)
                    ax.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.0, alpha=0.55)
                    rho = scipy_stats.spearmanr(xall[ok], yall[ok]).correlation
                    ax.text(0.04, 0.94, f"Spearman r={rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
                except Exception:
                    pass
            ax.set_xlabel("Cluster-level DepStruct")
            ax.set_ylabel(METRIC_LABEL[metric])
            ax.grid(alpha=0.16, linewidth=0.5)
            if i == 0:
                ax.set_title(METRIC_LABEL[metric], fontweight="bold")
            if j == 0:
                ax.text(-0.35, 0.5, nice_dataset_label(ds), rotation=90, transform=ax.transAxes, ha="center", va="center", fontweight="bold", fontsize=12)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=REGIME_COLOR[r], label=REGIME_LABEL[r], markersize=8) for r in REGIMES]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Relationship between DepStruct and external diagnostics", fontweight="bold", y=1.075)
    savefig_multi(fig, OUT_ROOT / "02_depstruct_vs_external_diagnostics")

plot_external_diagnostics(external_df, pval_df)
plot_depstruct_vs_external(external_df)

print("\n[DONE] External benchmark-like diagnostics generated.")
print("[OUT_ROOT]", OUT_ROOT)
