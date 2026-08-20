from __future__ import annotations

# =============================================================================
# CRC Part 2 | Dataset-specific perturbation-inference space
# =============================================================================
#
# Figure logic
# ------------
# A  Part 1 perturbation-sensitive MLTD -> one RNA input context
# B  RNA / TF / RE state-recovery benchmark
# C  One specific RNA input and its reconstructed RNA, predicted TF and RE
# D  Foundation vs adapted joint-state identity matrices
# E  Joint RNA-TF-RE inference-space geometry
# F  Joint metrics and interface to five-dimensional propagation
#
# Required files in PROFILE_DIR
# --------------------------------
# cluster_profile_full_test_rna_baseline.npz
# cluster_profile_full_test_rna_frontier.npz
# cluster_profile_full_test_tf_baseline.npz
# cluster_profile_full_test_tf_frontier.npz
# cluster_profile_full_test_re_baseline.npz
# cluster_profile_full_test_re_frontier.npz
#
# Required keys:
# cluster_ids, cluster_counts, true_cluster, pred_cluster
#
# Optional external benchmarks
# ----------------------------
# Put METHOD__rna.npz, METHOD__tf.npz or METHOD__re.npz under
# BENCHMARK_INPUT_DIR. Each file needs cluster_ids and pred_cluster.
# feature_ids is optional; without it, feature order must exactly match IMAS.
# =============================================================================

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


# =============================================================================
# 0) CONFIG
# =============================================================================

BASE = get_project_root()
DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

RUN_ROOT = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)
PROFILE_DIR = (
    RUN_ROOT
    / "structure_stability_full_eval_frontier_vs_baseline_rna_tf"
)
OUT_DIR = RUN_ROOT / "fig3_dataset_specific_perturbation_inference_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_INPUT_DIR = RUN_ROOT / "part2_inference_space_benchmark_inputs"
BENCHMARK_INPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ("rna", "tf", "re")
TASK_DISPLAY = {
    "rna": "RNA reconstruction",
    "tf": "TF-state prediction",
    "re": "RE-state prediction",
}
TASK_SHORT = {"rna": "RNA", "tf": "TF", "re": "RE"}

PROFILE_PATHS = {
    task: {
        "Foundation": PROFILE_DIR
        / f"cluster_profile_full_test_{task}_baseline.npz",
        "IMAS-Adapted": PROFILE_DIR
        / f"cluster_profile_full_test_{task}_frontier.npz",
    }
    for task in TASKS
}

REPRESENTATIVE_CLUSTER: Optional[int] = None
PROFILE_N_FEATURES = {"rna": 28, "tf": 24, "re": 28}
TOP_K = 25
BOOTSTRAP_N = 2000
SEED = 42
DPI = 400
REQUIRE_EXTERNAL_BENCHMARKS = False

CLUSTER_ORDER = list(range(10))

FOUNDATION_COLOR = "#6C87B5"
ADAPTED_COLOR = "#C95064"
TRUTH_COLOR = "#4A4A4A"
EXTERNAL_COLOR = "#C79A33"
GRID_COLOR = "#E7E7E7"
TEXT_COLOR = "#292929"
LAYER_COLORS = {"rna": "#AF5D6B", "tf": "#657BAA", "re": "#A38A50"}

IDENTITY_CMAP = LinearSegmentedColormap.from_list(
    "identity",
    ["#F5F1E9", "#D7DCE8", "#8599C5", "#354E8A"],
    N=256,
)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.linewidth": 0.75,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# =============================================================================
# 1) DATA OBJECTS AND HELPERS
# =============================================================================

@dataclass
class Layer:
    task: str
    cluster_ids: np.ndarray
    cluster_counts: np.ndarray
    feature_ids: np.ndarray
    feature_names: np.ndarray
    positional_features: bool
    truth: np.ndarray
    foundation: np.ndarray
    adapted: np.ndarray


@dataclass
class ExternalPrediction:
    task: str
    method: str
    cluster_ids: np.ndarray
    canonical_cluster_indices: np.ndarray
    canonical_feature_indices: np.ndarray
    prediction: np.ndarray


def log(*values: object) -> None:
    print(*values, flush=True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("[FIG]", stem.with_suffix(".png"))


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def first_key(npz, candidates: list[str], description: str, required: bool = True):
    for key in candidates:
        if key in npz.files:
            return key
    if required:
        raise KeyError(
            f"Cannot find {description}; candidates={candidates}; keys={list(npz.files)}"
        )
    return None


def decode(values: np.ndarray) -> np.ndarray:
    out = []
    for value in np.asarray(values).reshape(-1).tolist():
        out.append(
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else str(value)
        )
    return np.asarray(out, dtype=object)


def orient(matrix: np.ndarray, n_clusters: int, label: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{label}: expected 2D matrix, got {matrix.shape}")
    if matrix.shape[0] == n_clusters:
        return matrix
    if matrix.shape[1] == n_clusters:
        return matrix.T
    raise ValueError(f"{label}: shape={matrix.shape}, n_clusters={n_clusters}")


def row_zscore(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(matrix, axis=1, keepdims=True)
    sd = np.nanstd(matrix, axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return np.nan_to_num((matrix - mean) / sd)


def vector_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    out = np.full_like(values, np.nan)
    if valid.sum() == 0:
        return out
    mean = float(np.mean(values[valid]))
    sd = float(np.std(values[valid]))
    out[valid] = values[valid] - mean if sd < 1e-8 else (values[valid] - mean) / sd
    return out


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    aa, bb = a[valid], b[valid]
    if np.std(aa) < 1e-8 or np.std(bb) < 1e-8:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    return float(spearmanr(a[valid], b[valid]).statistic)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    aa, bb = a[valid], b[valid]
    denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
    return np.nan if denominator < 1e-12 else float(np.dot(aa, bb) / denominator)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    aa, bb = a[valid], b[valid]
    scale = max(float(np.std(aa)), 1e-8)
    return float(np.sqrt(np.mean((aa - bb) ** 2)) / scale)


def top_overlap(a: np.ndarray, b: np.ndarray, k: int = TOP_K) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid_idx = np.where(np.isfinite(a) & np.isfinite(b))[0]
    if valid_idx.size == 0:
        return np.nan
    k_use = min(k, valid_idx.size)
    ia = valid_idx[np.argsort(a[valid_idx])[-k_use:]]
    ib = valid_idx[np.argsort(b[valid_idx])[-k_use:]]
    return len(set(ia.tolist()) & set(ib.tolist())) / float(k_use)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        draws[i] = np.mean(rng.choice(values, size=values.size, replace=True))
    return mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def short_label(value: object, length: int = 13) -> str:
    text = str(value)
    return text if len(text) <= length else text[: length - 1] + "…"


# =============================================================================
# 2) LOAD FOUNDATION / ADAPTED PROFILES
# =============================================================================

def load_raw_profile(task: str, model: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing profile: {path}")

    z = np.load(path, allow_pickle=True)
    cluster_key = first_key(z, ["cluster_ids", "clusters", "cluster_id"], "cluster IDs")
    truth_key = first_key(z, ["true_cluster", "truth_cluster", "true"], "truth matrix")
    pred_key = first_key(z, ["pred_cluster", "prediction_cluster", "pred"], "prediction matrix")

    cluster_ids = np.asarray(z[cluster_key]).astype(np.int64)
    truth = orient(z[truth_key], len(cluster_ids), f"{path.name}:{truth_key}")
    pred = orient(z[pred_key], len(cluster_ids), f"{path.name}:{pred_key}")
    if truth.shape != pred.shape:
        raise ValueError(f"{path.name}: truth/pred mismatch {truth.shape} vs {pred.shape}")

    counts = (
        np.asarray(z["cluster_counts"]).astype(np.int64)
        if "cluster_counts" in z.files
        else np.ones(len(cluster_ids), dtype=np.int64)
    )

    id_key = first_key(
        z,
        ["feature_ids", "gene_ids", "tf_ids", "peak_ids", "features"],
        "feature IDs",
        required=False,
    )
    name_key = first_key(
        z,
        ["feature_names", "gene_names", "tf_names", "peak_names"],
        "feature names",
        required=False,
    )

    positional = id_key is None
    feature_ids = (
        np.asarray([f"{task}_position_{i}" for i in range(truth.shape[1])], dtype=object)
        if id_key is None
        else decode(z[id_key])
    )
    feature_names = (
        np.asarray([f"{TASK_SHORT[task]}_{i:04d}" for i in range(truth.shape[1])], dtype=object)
        if name_key is None
        else decode(z[name_key])
    )

    if len(feature_ids) != truth.shape[1]:
        raise ValueError(f"{path.name}: feature ID count does not match matrix width")
    if len(feature_names) != truth.shape[1]:
        feature_names = np.asarray(
            [f"{TASK_SHORT[task]}_{i:04d}" for i in range(truth.shape[1])],
            dtype=object,
        )

    return {
        "task": task,
        "model": model,
        "path": path,
        "keys": list(z.files),
        "cluster_ids": cluster_ids,
        "cluster_counts": counts,
        "feature_ids": feature_ids,
        "feature_names": feature_names,
        "positional": positional,
        "truth": truth,
        "pred": pred,
    }


def align_layer(task: str) -> Layer:
    foundation = load_raw_profile(task, "Foundation", PROFILE_PATHS[task]["Foundation"])
    adapted = load_raw_profile(task, "IMAS-Adapted", PROFILE_PATHS[task]["IMAS-Adapted"])

    foundation_cluster_index = {
        int(cid): i for i, cid in enumerate(foundation["cluster_ids"].tolist())
    }
    adapted_cluster_index = {
        int(cid): i for i, cid in enumerate(adapted["cluster_ids"].tolist())
    }
    clusters = [
        cid
        for cid in CLUSTER_ORDER
        if cid in foundation_cluster_index and cid in adapted_cluster_index
    ]
    if not clusters:
        clusters = [
            int(cid)
            for cid in foundation["cluster_ids"].tolist()
            if int(cid) in adapted_cluster_index
        ]
    if not clusters:
        raise RuntimeError(f"{task}: no shared clusters")

    f_rows = [foundation_cluster_index[cid] for cid in clusters]
    a_rows = [adapted_cluster_index[cid] for cid in clusters]

    if not foundation["positional"] and not adapted["positional"]:
        adapted_feature_index = {
            str(fid): i for i, fid in enumerate(adapted["feature_ids"].tolist())
        }
        f_cols, a_cols = [], []
        for i, fid in enumerate(foundation["feature_ids"].tolist()):
            if str(fid) in adapted_feature_index:
                f_cols.append(i)
                a_cols.append(adapted_feature_index[str(fid)])
    else:
        n = min(foundation["truth"].shape[1], adapted["truth"].shape[1])
        f_cols = list(range(n))
        a_cols = list(range(n))

    if not f_cols:
        raise RuntimeError(f"{task}: no shared features")

    f_cols = np.asarray(f_cols, dtype=np.int64)
    a_cols = np.asarray(a_cols, dtype=np.int64)

    layer = Layer(
        task=task,
        cluster_ids=np.asarray(clusters, dtype=np.int64),
        cluster_counts=np.maximum(
            foundation["cluster_counts"][f_rows],
            adapted["cluster_counts"][a_rows],
        ),
        feature_ids=foundation["feature_ids"][f_cols],
        feature_names=foundation["feature_names"][f_cols],
        positional_features=foundation["positional"] or adapted["positional"],
        truth=adapted["truth"][a_rows][:, a_cols],
        foundation=foundation["pred"][f_rows][:, f_cols],
        adapted=adapted["pred"][a_rows][:, a_cols],
    )
    log("[PROFILE]", task, layer.truth.shape)
    return layer


def load_layers() -> dict[str, Layer]:
    layers = {task: align_layer(task) for task in TASKS}
    pd.DataFrame(
        [
            {
                "task": task,
                "n_clusters": len(layer.cluster_ids),
                "n_features": len(layer.feature_ids),
                "positional_features": layer.positional_features,
            }
            for task, layer in layers.items()
        ]
    ).to_csv(OUT_DIR / "profile_loading_qc.csv", index=False)
    return layers


# =============================================================================
# 3) OPTIONAL EXTERNAL BENCHMARKS
# =============================================================================

def write_benchmark_readme() -> None:
    text = """Standardized benchmark input

File name:
  METHOD__rna.npz
  METHOD__tf.npz
  METHOD__re.npz

Required keys:
  cluster_ids
  pred_cluster

Optional:
  feature_ids
  feature_names

Without feature_ids, prediction width and feature order must exactly match
the corresponding IMAS profile. Use the same held-out CCI10 cells/reference.
"""
    (BENCHMARK_INPUT_DIR / "README_standardized_inputs.txt").write_text(
        text,
        encoding="utf-8",
    )


def parse_method_task(path: Path) -> tuple[Optional[str], Optional[str]]:
    if "__" not in path.stem:
        return None, None
    left, right = path.stem.split("__", 1)
    if right.lower() in TASKS:
        return left, right.lower()
    if left.lower() in TASKS:
        return right, left.lower()
    return None, None


def load_external(path: Path, method: str, task: str, layer: Layer) -> ExternalPrediction:
    z = np.load(path, allow_pickle=True)
    cluster_key = first_key(z, ["cluster_ids", "clusters", "cluster_id"], "cluster IDs")
    pred_key = first_key(z, ["pred_cluster", "prediction_cluster", "pred"], "prediction")

    ext_cluster_ids = np.asarray(z[cluster_key]).astype(np.int64)
    pred = orient(z[pred_key], len(ext_cluster_ids), f"{path.name}:{pred_key}")
    ext_cluster_index = {int(cid): i for i, cid in enumerate(ext_cluster_ids.tolist())}

    canonical_cluster_indices, ext_rows = [], []
    for canonical_i, cid in enumerate(layer.cluster_ids.tolist()):
        if int(cid) in ext_cluster_index:
            canonical_cluster_indices.append(canonical_i)
            ext_rows.append(ext_cluster_index[int(cid)])
    if not ext_rows:
        raise RuntimeError(f"{path.name}: no overlapping clusters")

    feature_key = first_key(
        z,
        ["feature_ids", "gene_ids", "tf_ids", "peak_ids", "features"],
        "feature IDs",
        required=False,
    )
    if feature_key is None:
        if pred.shape[1] != len(layer.feature_ids):
            raise ValueError(
                f"{path.name}: width={pred.shape[1]}, canonical={len(layer.feature_ids)}"
            )
        canonical_feature_indices = np.arange(len(layer.feature_ids), dtype=np.int64)
        ext_cols = canonical_feature_indices
    else:
        ext_feature_ids = decode(z[feature_key])
        ext_feature_index = {str(fid): i for i, fid in enumerate(ext_feature_ids.tolist())}
        canonical_feature_indices, ext_cols = [], []
        for canonical_i, fid in enumerate(layer.feature_ids.tolist()):
            if str(fid) in ext_feature_index:
                canonical_feature_indices.append(canonical_i)
                ext_cols.append(ext_feature_index[str(fid)])
        if not ext_cols:
            raise RuntimeError(f"{path.name}: no overlapping features")
        canonical_feature_indices = np.asarray(canonical_feature_indices, dtype=np.int64)
        ext_cols = np.asarray(ext_cols, dtype=np.int64)

    return ExternalPrediction(
        task=task,
        method=method,
        cluster_ids=layer.cluster_ids[np.asarray(canonical_cluster_indices, dtype=np.int64)],
        canonical_cluster_indices=np.asarray(canonical_cluster_indices, dtype=np.int64),
        canonical_feature_indices=np.asarray(canonical_feature_indices, dtype=np.int64),
        prediction=pred[np.asarray(ext_rows, dtype=np.int64)][:, ext_cols],
    )


def load_external_benchmarks(
    layers: dict[str, Layer],
) -> dict[str, dict[str, ExternalPrediction]]:
    write_benchmark_readme()
    external = {task: {} for task in TASKS}
    errors = []

    for path in sorted(BENCHMARK_INPUT_DIR.glob("*.npz")):
        method, task = parse_method_task(path)
        if method is None or task is None:
            errors.append({"file": str(path), "error": "Use METHOD__task.npz naming"})
            continue
        try:
            external[task][method] = load_external(path, method, task, layers[task])
            log("[BENCHMARK]", task, method)
        except Exception as exc:
            errors.append({"file": str(path), "error": repr(exc)})

    if errors:
        pd.DataFrame(errors).to_csv(
            OUT_DIR / "external_benchmark_loading_errors.csv",
            index=False,
        )
    if REQUIRE_EXTERNAL_BENCHMARKS and sum(len(x) for x in external.values()) == 0:
        raise RuntimeError("No external benchmark loaded")
    return external


# =============================================================================
# 4) METRICS
# =============================================================================

def evaluate(
    task: str,
    method: str,
    cluster_ids: np.ndarray,
    counts: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, cid in enumerate(cluster_ids.tolist()):
        error = nrmse(truth[i], prediction[i])
        rows.append(
            {
                "task": task,
                "method": method,
                "cluster_id": int(cid),
                "n_cells": int(counts[i]),
                "pearson": safe_pearson(truth[i], prediction[i]),
                "spearman": safe_spearman(truth[i], prediction[i]),
                "cosine": safe_cosine(truth[i], prediction[i]),
                "rmse_similarity": 1.0 / (1.0 + error) if np.isfinite(error) else np.nan,
                "top_feature_overlap": top_overlap(truth[i], prediction[i]),
                "n_features": truth.shape[1],
            }
        )
    return pd.DataFrame(rows)


def build_benchmark_tables(
    layers: dict[str, Layer],
    external: dict[str, dict[str, ExternalPrediction]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for task, layer in layers.items():
        tables.append(
            evaluate(
                task,
                "Foundation",
                layer.cluster_ids,
                layer.cluster_counts,
                layer.truth,
                layer.foundation,
            )
        )
        tables.append(
            evaluate(
                task,
                "IMAS-Adapted",
                layer.cluster_ids,
                layer.cluster_counts,
                layer.truth,
                layer.adapted,
            )
        )
        for method, ext in external[task].items():
            ci = ext.canonical_cluster_indices
            fi = ext.canonical_feature_indices
            tables.append(
                evaluate(
                    task,
                    method,
                    layer.cluster_ids[ci],
                    layer.cluster_counts[ci],
                    layer.truth[ci][:, fi],
                    ext.prediction,
                )
            )

    cluster_metrics = pd.concat(tables, ignore_index=True)
    cluster_metrics.to_csv(OUT_DIR / "layer_benchmark_cluster_metrics.csv", index=False)

    summary_rows = []
    metrics = ["pearson", "spearman", "cosine", "rmse_similarity", "top_feature_overlap"]
    for (task, method), group in cluster_metrics.groupby(["task", "method"], sort=False):
        row = {
            "task": task,
            "method": method,
            "n_clusters": group["cluster_id"].nunique(),
            "n_features": int(group["n_features"].median()),
        }
        for metric in metrics:
            point, low, high = bootstrap_ci(
                group[metric].to_numpy(dtype=float),
                seed=SEED + sum(ord(c) for c in task + method + metric),
            )
            row[f"{metric}_mean"] = point
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row["composite_score"] = float(
            np.nanmean([row[f"{metric}_mean"] for metric in metrics])
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "layer_benchmark_summary.csv", index=False)
    return cluster_metrics, summary


# =============================================================================
# 5) REPRESENTATIVE INPUT
# =============================================================================

def choose_representative_cluster(
    layers: dict[str, Layer],
    cluster_metrics: pd.DataFrame,
) -> int:
    common = sorted(set.intersection(*[set(layer.cluster_ids.tolist()) for layer in layers.values()]))

    if REPRESENTATIVE_CLUSTER is not None:
        if int(REPRESENTATIVE_CLUSTER) not in common:
            raise ValueError("Requested representative cluster is not shared by all layers")
        return int(REPRESENTATIVE_CLUSTER)

    internal = cluster_metrics[
        cluster_metrics["method"].isin(["Foundation", "IMAS-Adapted"])
    ]
    pivot = internal.pivot_table(
        index=["task", "cluster_id"],
        columns="method",
        values="pearson",
        aggfunc="mean",
    ).reset_index()

    if not {"Foundation", "IMAS-Adapted"}.issubset(pivot.columns):
        return int(common[0])

    pivot["gain"] = pivot["IMAS-Adapted"] - pivot["Foundation"]
    gain = (
        pivot[pivot["cluster_id"].isin(common)]
        .groupby("cluster_id")["gain"]
        .mean()
        .sort_values(ascending=False)
    )
    return int(gain.index[0]) if not gain.empty else int(common[0])


def choose_features(layer: Layer, cluster_id: int, n: int) -> np.ndarray:
    row = int(np.where(layer.cluster_ids == int(cluster_id))[0][0])
    variance = np.nanvar(layer.truth, axis=0)
    difference = np.abs(layer.adapted[row] - layer.foundation[row])
    magnitude = np.abs(vector_zscore(layer.truth[row]))

    def rank(values):
        values = np.nan_to_num(values, nan=0.0)
        return np.argsort(np.argsort(values)) / max(len(values) - 1, 1)

    score = 0.45 * rank(variance) + 0.35 * rank(difference) + 0.20 * rank(magnitude)
    valid = np.where(
        np.isfinite(layer.truth[row])
        | np.isfinite(layer.foundation[row])
        | np.isfinite(layer.adapted[row])
    )[0]
    if valid.size == 0:
        valid = np.arange(layer.truth.shape[1])
    selected = valid[np.argsort(score[valid])[-min(n, valid.size) :]]
    return selected[np.argsort(layer.truth[row, selected])]


def best_external_method(task: str, summary: pd.DataFrame) -> Optional[str]:
    subset = summary[
        (summary["task"] == task)
        & ~summary["method"].isin(["Foundation", "IMAS-Adapted"])
    ].sort_values(["pearson_mean", "composite_score"], ascending=False)
    return None if subset.empty else str(subset.iloc[0]["method"])


def get_external_profile(
    task: str,
    method: str,
    cluster_id: int,
    selected_features: np.ndarray,
    external: dict[str, dict[str, ExternalPrediction]],
) -> Optional[np.ndarray]:
    ext = external[task].get(method)
    if ext is None:
        return None
    cluster_hit = np.where(ext.cluster_ids == int(cluster_id))[0]
    if cluster_hit.size == 0:
        return None
    feature_map = {
        int(canonical_i): method_i
        for method_i, canonical_i in enumerate(ext.canonical_feature_indices.tolist())
    }
    if any(int(i) not in feature_map for i in selected_features):
        return None
    method_cols = [feature_map[int(i)] for i in selected_features]
    return ext.prediction[int(cluster_hit[0]), method_cols]


# =============================================================================
# 6) JOINT SPACE
# =============================================================================

def common_joint_clusters(layers: dict[str, Layer]) -> np.ndarray:
    common = sorted(set.intersection(*[set(layer.cluster_ids.tolist()) for layer in layers.values()]))
    if not common:
        raise RuntimeError("No clusters shared by RNA, TF and RE")
    return np.asarray(common, dtype=np.int64)


def rows_for(layer: Layer, clusters: np.ndarray) -> np.ndarray:
    index = {int(cid): i for i, cid in enumerate(layer.cluster_ids.tolist())}
    return np.asarray([index[int(cid)] for cid in clusters], dtype=np.int64)


def joint_matrix(layers: dict[str, Layer], state: str, clusters: np.ndarray) -> np.ndarray:
    blocks = []
    for task in TASKS:
        layer = layers[task]
        r = rows_for(layer, clusters)
        matrix = (
            layer.truth[r]
            if state == "truth"
            else layer.foundation[r]
            if state == "foundation"
            else layer.adapted[r]
        )
        matrix = row_zscore(matrix) / math.sqrt(max(matrix.shape[1], 1))
        blocks.append(matrix)
    return np.concatenate(blocks, axis=1)


def cosine_matrix(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p = row_zscore(prediction)
    t = row_zscore(truth)
    p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return p @ t.T


def joint_identity(layers: dict[str, Layer], state: str, clusters: np.ndarray) -> np.ndarray:
    matrices = []
    for task in TASKS:
        layer = layers[task]
        r = rows_for(layer, clusters)
        pred = layer.foundation[r] if state == "foundation" else layer.adapted[r]
        matrices.append(cosine_matrix(pred, layer.truth[r]))
    return np.mean(np.stack(matrices), axis=0)


def identity_metrics(similarity: np.ndarray) -> dict:
    predicted = np.argmax(similarity, axis=1)
    n = similarity.shape[0]
    diagonal = np.diag(similarity)
    off_max = []
    for i in range(n):
        row = similarity[i].copy()
        row[i] = -np.inf
        off_max.append(np.max(row))
    return {
        "identity_accuracy": float(np.mean(predicted == np.arange(n))),
        "identity_margin": float(np.mean(diagonal - np.asarray(off_max))),
        "mean_diagonal_similarity": float(np.mean(diagonal)),
    }


def geometry_correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    dt = pdist(truth, metric="cosine")
    dp = pdist(prediction, metric="cosine")
    return safe_spearman(dt, dp)


def build_joint_results(
    layers: dict[str, Layer],
    cluster_metrics: pd.DataFrame,
):
    clusters = common_joint_clusters(layers)
    truth = joint_matrix(layers, "truth", clusters)
    foundation = joint_matrix(layers, "foundation", clusters)
    adapted = joint_matrix(layers, "adapted", clusters)

    identity = {
        "Foundation": joint_identity(layers, "foundation", clusters),
        "IMAS-Adapted": joint_identity(layers, "adapted", clusters),
    }

    rows = []
    for method, matrix in [("Foundation", foundation), ("IMAS-Adapted", adapted)]:
        subset = cluster_metrics[
            (cluster_metrics["method"] == method)
            & cluster_metrics["cluster_id"].isin(clusters)
        ]
        rows.append(
            {
                "method": method,
                **identity_metrics(identity[method]),
                "geometry_correlation": geometry_correlation(truth, matrix),
                "mean_layer_pearson": float(subset["pearson"].mean()),
                "mean_layer_cosine": float(subset["cosine"].mean()),
                "mean_top_feature_overlap": float(subset["top_feature_overlap"].mean()),
            }
        )
    joint_metrics = pd.DataFrame(rows)
    joint_metrics.to_csv(OUT_DIR / "joint_inference_space_metrics.csv", index=False)

    stacked = np.vstack([truth, foundation, adapted])
    pca = PCA(n_components=2, random_state=SEED)
    coordinates = pca.fit_transform(stacked)

    coordinate_rows = []
    offset = 0
    for state in ["Truth", "Foundation", "IMAS-Adapted"]:
        for cid, xy in zip(clusters.tolist(), coordinates[offset : offset + len(clusters)]):
            coordinate_rows.append(
                {
                    "state": state,
                    "cluster_id": int(cid),
                    "PC1": float(xy[0]),
                    "PC2": float(xy[1]),
                    "explained_PC1": float(pca.explained_variance_ratio_[0]),
                    "explained_PC2": float(pca.explained_variance_ratio_[1]),
                }
            )
        offset += len(clusters)

    pca_df = pd.DataFrame(coordinate_rows)
    pca_df.to_csv(OUT_DIR / "joint_inference_space_pca_coordinates.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "joint_identity_similarity_matrices.npz",
        cluster_ids=clusters,
        foundation=identity["Foundation"],
        adapted=identity["IMAS-Adapted"],
    )
    return joint_metrics, identity, clusters, pca_df


# =============================================================================
# 7) PLOTTING
# =============================================================================

def draw_concept(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Dataset-specific perturbation-inference space", loc="left", fontweight="bold")

    def box(x, y, width, height, text, face, edge, fontsize=7.2):
        patch = patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.14,rounding_size=0.13",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "-|>",
                "linewidth": 0.9,
                "color": "#555555",
                "shrinkA": 7,
                "shrinkB": 7,
            },
        )

    box(5, 8.8, 6.0, 0.9, "Part 1: perturbation-sensitive MLTD structure", "#F0F0F0", "#777777")
    box(5, 7.1, 3.7, 0.9, "Specific RNA input context", "#F7EAED", ADAPTED_COLOR)
    box(1.8, 4.8, 2.7, 1.2, "Reconstructed\nRNA state", "#F7EAED", LAYER_COLORS["rna"])
    box(5.0, 4.8, 2.7, 1.2, "Predicted\nTF state", "#EDF0F7", LAYER_COLORS["tf"])
    box(8.2, 4.8, 2.7, 1.2, "Predicted\nRE state", "#F5F0E4", LAYER_COLORS["re"])
    box(5, 2.1, 6.7, 1.1, "Joint RNA–TF–RE molecular coordinates", "#F4F2EE", "#7A6A55", 7.6)
    box(5, 0.65, 6.9, 0.75, "Space for downstream five-dimensional propagation", "#F0F0F0", "#8A8A8A", 6.8)

    arrow(5, 8.3, 5, 7.6)
    arrow(4.3, 6.7, 2.3, 5.45)
    arrow(5, 6.6, 5, 5.45)
    arrow(5.7, 6.7, 7.7, 5.45)
    arrow(2.4, 4.2, 4.2, 2.65)
    arrow(5, 4.2, 5, 2.7)
    arrow(7.6, 4.2, 5.8, 2.65)
    arrow(5, 1.5, 5, 1.02)


def ordered_methods(methods: list[str]) -> list[str]:
    output = [method for method in ["IMAS-Adapted", "Foundation"] if method in methods]
    output.extend(sorted(method for method in methods if method not in output))
    return output


def draw_benchmark(ax: plt.Axes, task: str, summary: pd.DataFrame) -> None:
    subset = summary[summary["task"] == task].copy()
    methods = ordered_methods(subset["method"].astype(str).unique().tolist())
    subset = subset.set_index("method").reindex(methods).reset_index()

    for i, row in subset.iterrows():
        method = str(row["method"])
        color = (
            ADAPTED_COLOR
            if method == "IMAS-Adapted"
            else FOUNDATION_COLOR
            if method == "Foundation"
            else EXTERNAL_COLOR
        )
        ax.hlines(i, row["pearson_ci_low"], row["pearson_ci_high"], color=color, linewidth=1.3)
        ax.scatter(
            row["pearson_mean"],
            i,
            s=48 if method == "IMAS-Adapted" else 34,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.axvline(0, color="#BEBEBE", linewidth=0.65, linestyle="--")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlim(-0.15, 1.02)
    ax.set_xlabel("Mean cluster-level Pearson r")
    ax.set_title(TASK_DISPLAY[task], fontweight="bold")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_profile(
    ax: plt.Axes,
    task: str,
    layer: Layer,
    cluster_id: int,
    external: dict[str, dict[str, ExternalPrediction]],
    summary: pd.DataFrame,
) -> None:
    row = int(np.where(layer.cluster_ids == int(cluster_id))[0][0])
    selected = choose_features(layer, cluster_id, PROFILE_N_FEATURES[task])
    x = np.arange(len(selected))

    truth = vector_zscore(layer.truth[row, selected])
    foundation = vector_zscore(layer.foundation[row, selected])
    adapted = vector_zscore(layer.adapted[row, selected])

    ax.axhline(0, color="#C8C8C8", linewidth=0.6)
    ax.plot(x, truth, color=TRUTH_COLOR, linewidth=1.35, label="Observed/reference")
    ax.plot(x, foundation, color=FOUNDATION_COLOR, linewidth=1.0, label="Foundation")
    ax.plot(x, adapted, color=ADAPTED_COLOR, linewidth=1.45, label="IMAS-Adapted")

    best = best_external_method(task, summary)
    if best is not None:
        profile = get_external_profile(task, best, cluster_id, selected, external)
        if profile is not None:
            ax.plot(
                x,
                vector_zscore(profile),
                color=EXTERNAL_COLOR,
                linewidth=1.0,
                linestyle="--",
                label=best,
            )

    r0 = safe_pearson(layer.truth[row, selected], layer.foundation[row, selected])
    r1 = safe_pearson(layer.truth[row, selected], layer.adapted[row, selected])
    ax.text(
        0.995,
        0.92,
        f"r: {r0:.2f} → {r1:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )

    labels = [short_label(layer.feature_names[i]) for i in selected]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=62, ha="right", fontsize=6)
    ax.set_ylabel("Within-state\nz-score")
    ax.set_title(f"{TASK_SHORT[task]} output from RNA input C{cluster_id}", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_identity(ax: plt.Axes, matrix: np.ndarray, clusters: np.ndarray, title: str):
    low = float(np.quantile(matrix, 0.05))
    high = float(np.quantile(matrix, 0.98))
    if high <= low:
        high = low + 1e-6
    image = ax.imshow(matrix, cmap=IDENTITY_CMAP, norm=Normalize(vmin=low, vmax=high))
    labels = [f"C{int(cid)}" for cid in clusters]
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_yticks(np.arange(len(clusters)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Observed joint state")
    ax.set_ylabel("Predicted joint state")
    metrics = identity_metrics(matrix)
    ax.set_title(
        f"{title}\nidentity={metrics['identity_accuracy']:.0%}, margin={metrics['identity_margin']:.2f}",
        fontweight="bold",
        fontsize=8,
    )
    return image


def draw_pca(ax: plt.Axes, coordinates: pd.DataFrame) -> None:
    styles = {
        "Truth": ("o", TRUTH_COLOR, 38),
        "Foundation": ("x", FOUNDATION_COLOR, 38),
        "IMAS-Adapted": ("^", ADAPTED_COLOR, 44),
    }

    for cid in sorted(coordinates["cluster_id"].unique()):
        table = coordinates[coordinates["cluster_id"] == cid].set_index("state")
        truth_xy = table.loc["Truth", ["PC1", "PC2"]].to_numpy(dtype=float)
        for state, color in [("Foundation", FOUNDATION_COLOR), ("IMAS-Adapted", ADAPTED_COLOR)]:
            xy = table.loc[state, ["PC1", "PC2"]].to_numpy(dtype=float)
            ax.plot(
                [truth_xy[0], xy[0]],
                [truth_xy[1], xy[1]],
                color=color,
                linewidth=0.7,
                alpha=0.5,
            )

    for state, (marker, color, size) in styles.items():
        subset = coordinates[coordinates["state"] == state]
        ax.scatter(subset["PC1"], subset["PC2"], marker=marker, s=size, color=color, label=state, zorder=3)
        if state == "Truth":
            for _, row in subset.iterrows():
                ax.text(row["PC1"], row["PC2"], f"C{int(row['cluster_id'])}", fontsize=6, ha="left", va="bottom")

    pc1 = float(coordinates.iloc[0]["explained_PC1"])
    pc2 = float(coordinates.iloc[0]["explained_PC2"])
    ax.set_xlabel(f"Joint PC1 ({pc1:.1%})")
    ax.set_ylabel(f"Joint PC2 ({pc2:.1%})")
    ax.set_title("Joint RNA–TF–RE inference-space geometry", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color=GRID_COLOR, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_joint_metrics(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    definitions = [
        ("identity_accuracy", "State identity"),
        ("geometry_correlation", "Geometry"),
        ("mean_layer_pearson", "Layer recovery"),
        ("mean_top_feature_overlap", "Top-feature\nrecovery"),
    ]
    x = np.arange(len(definitions))
    width = 0.30

    for j, method in enumerate(["Foundation", "IMAS-Adapted"]):
        row = metrics[metrics["method"] == method].iloc[0]
        values = [float(row[key]) for key, _ in definitions]
        ax.bar(
            x + (-width / 2 if j == 0 else width / 2),
            values,
            width,
            color=FOUNDATION_COLOR if method == "Foundation" else ADAPTED_COLOR,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in definitions])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Recovery score")
    ax.set_title("Joint perturbation-inference space", fontweight="bold")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.5,
        -0.33,
        "Joint molecular coordinates\n↓\nFive-dimensional propagation constraints",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "#F2F2F2",
            "edgecolor": "#B5B5B5",
            "linewidth": 0.7,
        },
        clip_on=False,
    )


def build_figure(
    layers,
    external,
    summary,
    representative_cluster,
    joint_metrics,
    identity,
    clusters,
    pca_df,
):
    fig = plt.figure(figsize=(16.2, 12.4))
    outer = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.62, 1.27],
        width_ratios=[1.05, 1.0, 1.0, 1.0],
        hspace=0.55,
        wspace=0.48,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    draw_concept(ax_a)
    panel_letter(ax_a, "A")

    benchmark_grid = outer[0, 1:4].subgridspec(1, 3, wspace=0.52)
    benchmark_axes = []
    for i, task in enumerate(TASKS):
        ax = fig.add_subplot(benchmark_grid[0, i])
        draw_benchmark(ax, task, summary)
        benchmark_axes.append(ax)
    panel_letter(benchmark_axes[0], "B", x=-0.28)

    profile_grid = outer[1, :].subgridspec(3, 1, hspace=0.46)
    profile_axes = []
    for i, task in enumerate(TASKS):
        ax = fig.add_subplot(profile_grid[i, 0])
        draw_profile(ax, task, layers[task], representative_cluster, external, summary)
        profile_axes.append(ax)
    profile_axes[0].legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.30))
    panel_letter(profile_axes[0], "C", x=-0.035, y=1.30)
    profile_axes[0].text(
        0.5,
        1.30,
        f"Specific RNA input context: C{representative_cluster}",
        transform=profile_axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    identity_grid = outer[2, 0:2].subgridspec(1, 2, wspace=0.32)
    ax_d1 = fig.add_subplot(identity_grid[0, 0])
    ax_d2 = fig.add_subplot(identity_grid[0, 1])
    image = draw_identity(ax_d1, identity["Foundation"], clusters, "Foundation")
    draw_identity(ax_d2, identity["IMAS-Adapted"], clusters, "IMAS-Adapted")
    panel_letter(ax_d1, "D", x=-0.30)
    cbar = fig.colorbar(image, ax=[ax_d1, ax_d2], fraction=0.020, pad=0.025, aspect=28)
    cbar.set_label("Mean cross-layer cosine similarity", fontsize=7)

    ax_e = fig.add_subplot(outer[2, 2])
    draw_pca(ax_e, pca_df)
    panel_letter(ax_e, "E", x=-0.22)

    ax_f = fig.add_subplot(outer[2, 3])
    draw_joint_metrics(ax_f, joint_metrics)
    panel_letter(ax_f, "F", x=-0.22)

    fig.suptitle(
        "Joint RNA, TF and regulatory-element state recovery constructs "
        "a dataset-specific perturbation-inference space",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    save_figure(
        fig,
        OUT_DIR / "fig3_dataset_specific_perturbation_inference_space_main",
    )


# =============================================================================
# 8) MAIN
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    log("[1/6] Loading RNA, TF and RE profiles")
    layers = load_layers()

    log("[2/6] Loading optional benchmark predictions")
    external = load_external_benchmarks(layers)

    log("[3/6] Evaluating layer-wise recovery")
    cluster_metrics, summary = build_benchmark_tables(layers, external)

    log("[4/6] Selecting a specific RNA input context")
    representative_cluster = choose_representative_cluster(layers, cluster_metrics)
    log("[INPUT CONTEXT]", f"C{representative_cluster}")

    log("[5/6] Constructing the joint RNA-TF-RE space")
    joint_metrics, identity, clusters, pca_df = build_joint_results(layers, cluster_metrics)

    log("[6/6] Building the main Part 2 figure")
    build_figure(
        layers,
        external,
        summary,
        representative_cluster,
        joint_metrics,
        identity,
        clusters,
        pca_df,
    )

    with open(
        OUT_DIR / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "sample_id": SAMPLE_ID,
                "representative_cluster": int(representative_cluster),
                "external_benchmarks_loaded": {
                    task: sorted(external[task].keys()) for task in TASKS
                },
                "n_clusters_by_layer": {
                    task: int(len(layer.cluster_ids)) for task, layer in layers.items()
                },
                "n_features_by_layer": {
                    task: int(len(layer.feature_ids)) for task, layer in layers.items()
                },
                "scope": (
                    "Joint RNA reconstruction, TF-state prediction and RE-state "
                    "prediction define the dataset-specific perturbation-inference "
                    "space before five-dimensional propagation."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    log("[DONE]", OUT_DIR)


if __name__ == "__main__":
    main()
