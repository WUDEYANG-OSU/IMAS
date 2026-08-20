from __future__ import annotations

# ============================================================
# global_edge_endpoint_ablation.py
# ------------------------------------------------------------
# 全局统一 endpoint 的 stage-wise ablation + 全套 control 指标。
#
# 这个脚本做的不是“每个阶段各算各的指标”，而是：
#   1) 构建同一批 candidate directed cluster edges: C_i -> C_j
#   2) 定义多个统一 endpoint labels
#   3) 每个阶段/对照方法都给同一批 edges 打分
#   4) 对所有 score 使用同一套指标评价
#
# Endpoints:
#   - global_positive_edge:
#       high-confidence / TF-kept CCI + positive communication delta + endpoint score high
#   - global_mechanistic_positive_edge:
#       global_positive_edge + StageA/B mechanism support high
#   - cci_consensus_positive_edge:
#       high-confidence / TF-kept CCI consensus endpoint，不直接依赖 pseudotime
#   - tf_response_positive_edge:
#       receiver TF response high + positive communication delta
#
# Scores:
#   Null controls:
#     N0_Random
#     N1_Degree_only
#     N2_Direction_flipped
#     N3_TF_shuffled
#     N4_CCI_score_shuffled
#   Progressive stages:
#     M0_RNA_only
#     M1_RNA_TF
#     M2_RNA_TF_matrix
#     M3_CCI
#     M4_CCI_LTC
#     M5_Full
#
# Metrics:
#   Recovery:
#     AUROC, AUPRC, MRR, positive rank, Hit@K, Precision@K, Recall@K,
#     NDCG@K, Lift@K / Enrichment@K
#   Calibration / separation:
#     Brier, ECE, positive-negative score separation, Cohen's d
#   Directionality:
#     topK delta-positive fraction, mean delta@K, delta separation,
#     score-delta correlation
#   Regulatory:
#     topK TF response, topK RNA-TF support, score-TF corr, score-RNA-TF corr
#   CCI:
#     topK highconf rate, topK TFkept rate, topK CCI score, score-CCI corr
#   Mechanism:
#     topK StageA support, StageB support, StageAB support, score-StageAB corr
#   Topology / coverage:
#     sender/receiver coverage@K, edge pair diversity@K,
#     reciprocal penalty@K, temporal DAG consistency@K,
#     coverage-adjusted precision@K
#   Robustness:
#     bootstrap AUROC/AUPRC/Precision@10 std,
#     bootstrap topK Jaccard stability
#
# 用法：
#   python global_edge_endpoint_ablation.py
#   python global_edge_endpoint_ablation.py --n_bootstrap 200
# ============================================================

import argparse
import itertools
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
OUT_DIR = BASE / "runs" / "global_edge_endpoint_ablation" / DATASET_NAME / SAMPLE_ID

NULL_SCORE_ORDER = [
    "N0_Random",
    "N1_Degree_only",
    "N2_Direction_flipped",
    "N3_TF_receiver_shuffled",
    "N4_CCI_shuffled_only",
    "N5_CCI_shuffled_plus_LTC",
]

STAGE_ORDER = [
    "M0_RNA_only",
    "M1_RNA_TF",
    "M2_RNA_TF_matrix",
    "M3_CCI",
    "M4_CCI_LTC",
    "M5_Full",
]

SCORE_ORDER = NULL_SCORE_ORDER + STAGE_ORDER

SCORE_LABEL = {
    "N0_Random": "N0 Random",
    "N1_Degree_only": "N1 Degree-only",
    "N2_Direction_flipped": "N2 Direction-flipped",
    "N3_TF_receiver_shuffled": "N3 TF receiver-shuffled",
    "N4_CCI_shuffled_only": "N4 CCI-shuffled only",
    "N5_CCI_shuffled_plus_LTC": "N5 CCI-shuffled + LTC",
    "M0_RNA_only": "M0 RNA-only",
    "M1_RNA_TF": "M1 RNA+TF",
    "M2_RNA_TF_matrix": "M2 RNA-TF matrix",
    "M3_CCI": "M3 CCI",
    "M4_CCI_LTC": "M4 CCI+LTC",
    "M5_Full": "M5 Full",
}

ENDPOINTS = [
    "global_positive_edge",
    "global_mechanistic_positive_edge",
    "cci_consensus_positive_edge",
    "tf_response_positive_edge",
]

DEFAULT_LABEL_MODE = "strict"  # strict / tfkept_directional / top_quantile
POSITIVE_SCORE_Q = 0.70
MECHANISTIC_SCORE_Q = 0.70
CCI_CONSENSUS_Q = 0.70
TF_RESPONSE_Q = 0.70
MIN_POSITIVE_DELTA = 0.0
TOPK_LIST = [5, 10, 20, 50]
DEFAULT_BOOTSTRAP_N = 200
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_ECE_BINS = 10

PRIMARY_METRICS = [
    "auroc",
    "auprc",
    "mrr",
    "mean_rank_positive_norm_inv",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "lift_at_10",
    "brier",
    "ece",
    "score_separation",
    "cohen_d",
    "top10_delta_positive_frac",
    "top10_temporal_dag_consistency",
    "top10_reciprocal_penalty",
    "top10_highconf_rate",
    "top10_tfkept_rate",
    "top10_receiver_tf_support_mean",
    "top10_rna_tf_support_mean",
    "top10_stageAB_support_mean",
    "top10_sender_coverage_frac",
    "top10_receiver_coverage_frac",
    "top10_coverage_adjusted_precision",
    "bootstrap_auroc_std",
    "bootstrap_auprc_std",
    "bootstrap_top10_jaccard_mean",
]


# ============================================================
# 1) BASIC HELPERS
# ============================================================
def log(*args: Any) -> None:
    print(*args, flush=True)


def safe_read_csv(path: Path, missing: Optional[list[dict[str, Any]]] = None, label: str = "") -> Optional[pd.DataFrame]:
    if path is None or not Path(path).exists():
        if missing is not None:
            missing.append({"label": label, "path": str(path), "reason": "missing"})
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        if missing is not None:
            missing.append({"label": label, "path": str(path), "reason": f"read_error:{e}"})
        return None


def glob_first(root: Path, patterns: list[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def glob_all(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(root.glob(pat)))
    seen = set()
    final = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            final.append(p)
    return final


def first_existing_col(df: Optional[pd.DataFrame], candidates: list[str]) -> Optional[str]:
    if df is None:
        return None
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def cols_containing(df: Optional[pd.DataFrame], keywords: list[str]) -> list[str]:
    if df is None:
        return []
    out = []
    for c in df.columns:
        lc = c.lower()
        if any(k.lower() in lc for k in keywords):
            out.append(c)
    return out


def normalize_cluster_str(x: Any) -> str:
    s = str(x).strip()
    if s.startswith("C") and s[1:].isdigit():
        return s[1:]
    if s.endswith(".0"):
        s2 = s[:-2]
        if s2.isdigit():
            return s2
    return s


def edge_key(sender: Any, receiver: Any) -> str:
    return f"{normalize_cluster_str(sender)}->{normalize_cluster_str(receiver)}"


def parse_edge_key(k: str) -> tuple[str, str]:
    kk = str(k).replace("C", "")
    if "->" in kk:
        a, b = kk.split("->", 1)
        return normalize_cluster_str(a), normalize_cluster_str(b)
    if "→" in kk:
        a, b = kk.split("→", 1)
        return normalize_cluster_str(a), normalize_cluster_str(b)
    raise ValueError(f"Cannot parse edge key: {k}")


def stable_numeric_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").values.astype(float)


def minmax01(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out
    lo = np.nanmin(x[m])
    hi = np.nanmax(x[m])
    if hi <= lo + 1e-12:
        out[m] = 0.5
    else:
        out[m] = (x[m] - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0)


def safe_mean(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def safe_median(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def safe_corr(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m]
    bb = b[m]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def gini_nonnegative(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n)


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Z = X / norm
    return Z @ Z.T


def sigmoid01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    out[~np.isfinite(x)] = np.nan
    return out


# ============================================================
# 2) METRIC HELPERS
# ============================================================
def auc_binary(y_true: Any, score: Any) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(all_s), dtype=float)
    r_pos = ranks[: len(pos)]
    auc = (r_pos.sum() - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg) + 1e-12)
    return float(auc)


def pr_curve(y_true: Any, score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if len(y) == 0 or np.sum(y == 1) == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (np.sum(y == 1) + 1e-12)
    thresholds = s[order]
    return precision, recall, thresholds


def roc_curve(y_true: Any, score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if len(y) == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)
    tpr = tp / (np.sum(y == 1) + 1e-12)
    fpr = fp / (np.sum(y == 0) + 1e-12)
    thresholds = s[order]
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])
    return fpr, tpr, thresholds


def auprc_binary(y_true: Any, score: Any) -> float:
    precision, recall, _ = pr_curve(y_true, score)
    if len(precision) == 0:
        return np.nan
    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))


def precision_recall_at_k(y_true: Any, score: Any, k: int) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if len(y) == 0:
        return np.nan, np.nan
    k = min(int(k), len(y))
    if k <= 0:
        return np.nan, np.nan
    order = np.argsort(-s)[:k]
    tp = float(np.sum(y[order] == 1))
    precision = tp / k
    recall = tp / (np.sum(y == 1) + 1e-12) if np.sum(y == 1) > 0 else np.nan
    return float(precision), float(recall)


def ndcg_at_k(y_true: Any, score: Any, k: int) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if len(y) == 0:
        return np.nan
    k = min(int(k), len(y))
    if k <= 0:
        return np.nan
    order = np.argsort(-s)[:k]
    gains = y[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(y)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    if idcg <= 1e-12:
        return np.nan
    return float(dcg / idcg)


def rank_stats(y_true: Any, score: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if len(y) == 0 or np.sum(y == 1) == 0:
        return {
            "mrr": np.nan,
            "mean_rank_positive": np.nan,
            "median_rank_positive": np.nan,
            "mean_rank_positive_norm_inv": np.nan,
            "median_rank_positive_norm_inv": np.nan,
        }
    order = np.argsort(-s)
    y_ranked = y[order]
    pos_ranks = np.where(y_ranked == 1)[0] + 1
    mrr = float(np.mean(1.0 / pos_ranks))
    mean_rank = float(np.mean(pos_ranks))
    med_rank = float(np.median(pos_ranks))
    n = len(y)
    mean_inv = float(1.0 - (mean_rank - 1.0) / max(n - 1.0, 1.0))
    med_inv = float(1.0 - (med_rank - 1.0) / max(n - 1.0, 1.0))
    return {
        "mrr": mrr,
        "mean_rank_positive": mean_rank,
        "median_rank_positive": med_rank,
        "mean_rank_positive_norm_inv": mean_inv,
        "median_rank_positive_norm_inv": med_inv,
    }


def score_to_probability(score: Any) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    p = minmax01(s)
    # minmax can become 0.5 for constant. Keep valid range.
    p = np.clip(p, 0.0, 1.0)
    return p


def brier_score(y_true: Any, score: Any) -> float:
    y = np.asarray(y_true, dtype=float)
    p = score_to_probability(score)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((p[m] - y[m]) ** 2))


def expected_calibration_error(y_true: Any, score: Any, n_bins: int = DEFAULT_ECE_BINS) -> float:
    y = np.asarray(y_true, dtype=float)
    p = score_to_probability(score)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)
        if idx.sum() == 0:
            continue
        conf = float(np.mean(p[idx]))
        acc = float(np.mean(y[idx]))
        ece += (idx.sum() / len(y)) * abs(acc - conf)
    return float(ece)


def separation_stats(y_true: Any, score: Any) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"score_separation": np.nan, "cohen_d": np.nan, "cliffs_delta": np.nan}
    sep = float(np.mean(pos) - np.mean(neg))
    sp = np.sqrt(((len(pos) - 1) * np.var(pos) + (len(neg) - 1) * np.var(neg)) / max(len(pos) + len(neg) - 2, 1))
    cohen = float(sep / (sp + 1e-12))
    # Cliff's delta: P(pos>neg)-P(pos<neg)
    comp = pos.reshape(-1, 1) - neg.reshape(1, -1)
    cliffs = float((np.sum(comp > 0) - np.sum(comp < 0)) / comp.size)
    return {"score_separation": sep, "cohen_d": cohen, "cliffs_delta": cliffs}


def topk_values(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    if score_col not in df.columns:
        return df.iloc[0:0].copy()
    d = df.copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d[np.isfinite(d[score_col])]
    if len(d) == 0:
        return d
    return d.sort_values(score_col, ascending=False).head(min(k, len(d)))


def topk_mean_by_score(df: pd.DataFrame, score_col: str, value_col: str, k: int) -> float:
    if value_col not in df.columns:
        return np.nan
    d = topk_values(df, score_col, k)
    if len(d) == 0:
        return np.nan
    return safe_mean(pd.to_numeric(d[value_col], errors="coerce").values)


def top_bottom_separation(df: pd.DataFrame, score_col: str, value_col: str, k: int) -> float:
    if score_col not in df.columns or value_col not in df.columns:
        return np.nan
    d = df[[score_col, value_col]].copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[np.isfinite(d[score_col]) & np.isfinite(d[value_col])]
    if len(d) < 2:
        return np.nan
    k = min(int(k), len(d) // 2 if len(d) >= 2 else len(d))
    if k <= 0:
        return np.nan
    dd = d.sort_values(score_col, ascending=False)
    return float(dd.head(k)[value_col].mean() - dd.tail(k)[value_col].mean())


def reciprocal_penalty_topk(df: pd.DataFrame, score_col: str, k: int) -> float:
    d = topk_values(df, score_col, k)
    if len(d) == 0:
        return np.nan
    edges = set(d["edge_key"].astype(str))
    rec_count = 0
    for e in edges:
        try:
            a, b = parse_edge_key(e)
            if f"{b}->{a}" in edges:
                rec_count += 1
        except Exception:
            continue
    # Each reciprocal pair counted twice. Fraction of topK edges involved in reciprocal pairs.
    return float(rec_count / max(len(edges), 1))


def coverage_metrics_topk(df: pd.DataFrame, score_col: str, k: int, n_clusters: int) -> dict[str, float]:
    d = topk_values(df, score_col, k)
    if len(d) == 0:
        return {
            "sender_coverage_frac": np.nan,
            "receiver_coverage_frac": np.nan,
            "edge_pair_diversity_frac": np.nan,
        }
    senders = set(d["sender"].astype(str))
    receivers = set(d["receiver"].astype(str))
    pairs = set(d["edge_key"].astype(str))
    max_edges = n_clusters * max(n_clusters - 1, 1)
    return {
        "sender_coverage_frac": len(senders) / max(n_clusters, 1),
        "receiver_coverage_frac": len(receivers) / max(n_clusters, 1),
        "edge_pair_diversity_frac": len(pairs) / max(min(k, max_edges), 1),
    }


def bootstrap_metrics(df: pd.DataFrame, y_col: str, score_col: str, n_bootstrap: int, seed: int, k: int = 10) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    y = df[y_col].values.astype(int)
    s = pd.to_numeric(df[score_col], errors="coerce").values.astype(float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    edge_keys = df.loc[m, "edge_key"].astype(str).values
    n = len(y)
    if n < 5 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return {
            "bootstrap_auroc_mean": np.nan,
            "bootstrap_auroc_std": np.nan,
            "bootstrap_auprc_mean": np.nan,
            "bootstrap_auprc_std": np.nan,
            f"bootstrap_precision_at_{k}_mean": np.nan,
            f"bootstrap_precision_at_{k}_std": np.nan,
            f"bootstrap_top{k}_jaccard_mean": np.nan,
            f"bootstrap_top{k}_jaccard_std": np.nan,
        }

    original_top = set(edge_keys[np.argsort(-s)[: min(k, n)]])
    aucs, aps, ps, jacs = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        ss = s[idx]
        kk = edge_keys[idx]
        if np.sum(yy == 1) == 0 or np.sum(yy == 0) == 0:
            continue
        aucs.append(auc_binary(yy, ss))
        aps.append(auprc_binary(yy, ss))
        p, _ = precision_recall_at_k(yy, ss, k)
        ps.append(p)
        boot_top = set(kk[np.argsort(-ss)[: min(k, len(ss))]])
        union = original_top | boot_top
        inter = original_top & boot_top
        jacs.append(len(inter) / max(len(union), 1))
    return {
        "bootstrap_auroc_mean": safe_mean(aucs),
        "bootstrap_auroc_std": float(np.nanstd(aucs)) if len(aucs) else np.nan,
        "bootstrap_auprc_mean": safe_mean(aps),
        "bootstrap_auprc_std": float(np.nanstd(aps)) if len(aps) else np.nan,
        f"bootstrap_precision_at_{k}_mean": safe_mean(ps),
        f"bootstrap_precision_at_{k}_std": float(np.nanstd(ps)) if len(ps) else np.nan,
        f"bootstrap_top{k}_jaccard_mean": safe_mean(jacs),
        f"bootstrap_top{k}_jaccard_std": float(np.nanstd(jacs)) if len(jacs) else np.nan,
    }


# ============================================================
# 3) PATHS AND LOADERS
# ============================================================
class Paths:
    def __init__(self, run_base: Path, sample_id: str):
        self.run_base = Path(run_base)
        self.sample_id = sample_id

    @property
    def rna_tf_dir(self) -> Path:
        return self.run_base / f"single_sample_{self.sample_id}_rna_tf_top10_network_heatmap"

    @property
    def cci_dir(self) -> Path:
        return self.run_base / f"{self.sample_id}_cci_newLeiden"

    @property
    def recluster_dir(self) -> Path:
        return self.run_base / f"single_sample_{self.sample_id}_recluster_for_cci"

    @property
    def pt_dir(self) -> Path:
        return self.cci_dir / "communication_pseudotime_clusterGNN_LTC"

    @property
    def stageA_dir(self) -> Path:
        return self.pt_dir / "stageA_cluster_plus_edgeC"

    @property
    def stageB_dir(self) -> Path:
        return self.pt_dir / "seed_edge_clusterpath"


def load_cluster_list(paths: Paths, missing: list[dict[str, Any]]) -> list[str]:
    candidates = [
        paths.pt_dir / "cluster_pseudotime.csv",
        paths.cci_dir / "cluster_profiles.csv",
        paths.cci_dir / "cluster_sizes.csv",
        paths.recluster_dir / f"{paths.sample_id}_cell_level_mapping_with_umap_and_internal16.csv",
    ]
    clusters: set[str] = set()
    for p in candidates:
        df = safe_read_csv(p)
        if df is None or len(df) == 0:
            continue
        ccol = first_existing_col(df, ["cluster", "new_cluster", "new_leiden", "leiden", "cluster_id", "new_leiden_cluster"])
        if ccol is not None:
            clusters |= set(df[ccol].map(normalize_cluster_str).astype(str))
    if not clusters:
        missing.append({"label": "cluster list", "path": str(paths.run_base), "reason": "no_cluster_column_found"})

    def sort_key(x: str):
        return (0, int(x)) if str(x).isdigit() else (1, str(x))

    return sorted(clusters, key=sort_key)


def make_candidate_edges(clusters: list[str], include_self_edges: bool = False) -> pd.DataFrame:
    rows = []
    for s, r in itertools.product(clusters, clusters):
        if (not include_self_edges) and s == r:
            continue
        rows.append({"sender": s, "receiver": r, "edge_key": f"{s}->{r}"})
    return pd.DataFrame(rows)


def load_cluster_profiles(paths: Paths) -> Optional[pd.DataFrame]:
    return safe_read_csv(paths.cci_dir / "cluster_profiles.csv")


def load_cluster_pseudotime(paths: Paths, missing: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
    return safe_read_csv(paths.pt_dir / "cluster_pseudotime.csv", missing, "cluster_pseudotime")


def load_edge_pseudotime(paths: Paths, missing: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
    return safe_read_csv(paths.pt_dir / "edge_pseudotime_summary.csv", missing, "edge_pseudotime_summary")


def load_cci_tables(paths: Paths, missing: list[dict[str, Any]]) -> dict[str, Optional[pd.DataFrame]]:
    return {
        "all": safe_read_csv(paths.cci_dir / "interactions_cluster_level.csv", missing, "CCI all"),
        "highconf": safe_read_csv(paths.cci_dir / "interactions_cluster_level_highconf.csv", missing, "CCI highconf"),
        "tfkept": safe_read_csv(paths.cci_dir / "interactions_cluster_level_highconf_TFkept.csv", missing, "CCI TFkept"),
        "signature": safe_read_csv(paths.cci_dir / "receptor_tf_signature.csv", missing, "receptor_tf_signature"),
    }


def load_rna_tf_matrix(paths: Paths, missing: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
    p = glob_first(paths.rna_tf_dir, ["*_network_score_matrix.csv", "*score_matrix.csv", "*matrix*.csv"])
    if p is None:
        missing.append({"label": "RNA-TF matrix", "path": str(paths.rna_tf_dir), "reason": "not_found"})
        return None
    try:
        return pd.read_csv(p, index_col=0)
    except Exception as e:
        missing.append({"label": "RNA-TF matrix", "path": str(p), "reason": f"read_error:{e}"})
        return None


# ============================================================
# 4) FEATURE BUILDERS
# ============================================================
def add_cluster_pseudotime_features(edge_df: pd.DataFrame, cl: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = edge_df.copy()
    for c in [
        "t_sender", "t_receiver", "delta_t_receiver_minus_sender",
        "receiver_tf_response", "sender_strength", "receiver_strength",
        "net_flow_receiver", "io_ratio_receiver",
    ]:
        out[c] = np.nan
    if cl is None or len(cl) == 0:
        return out
    ccol = first_existing_col(cl, ["cluster", "new_cluster", "new_leiden", "leiden", "cluster_id", "new_leiden_cluster"])
    if ccol is None:
        return out
    tmp = cl.copy()
    tmp["_cluster"] = tmp[ccol].map(normalize_cluster_str).astype(str)

    def map_col(candidates: list[str]) -> dict[str, float]:
        col = first_existing_col(tmp, candidates)
        if col is None:
            return {}
        return dict(zip(tmp["_cluster"], pd.to_numeric(tmp[col], errors="coerce")))

    tmap = map_col(["t_hat_norm_mean", "pseudotime", "t", "time"])
    tfmap = map_col(["receiver_tf_response", "tf_response", "receiver_tf", "tf_resp"])
    smap = map_col(["sender_strength", "send_strength"])
    rmap = map_col(["receiver_strength", "recv_strength"])
    netmap = map_col(["net_flow", "flow"])
    iomap = map_col(["io_ratio", "in_out_ratio"])

    out["t_sender"] = out["sender"].map(tmap)
    out["t_receiver"] = out["receiver"].map(tmap)
    out["delta_t_receiver_minus_sender"] = out["t_receiver"] - out["t_sender"]
    out["receiver_tf_response"] = out["receiver"].map(tfmap)
    out["sender_strength"] = out["sender"].map(smap)
    out["receiver_strength"] = out["receiver"].map(rmap)
    out["net_flow_receiver"] = out["receiver"].map(netmap)
    out["io_ratio_receiver"] = out["receiver"].map(iomap)
    return out


def aggregate_edge_table(df: Optional[pd.DataFrame], prefix: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["edge_key"])
    sender_col = first_existing_col(df, ["sender_cluster", "source_cluster", "sender", "source"])
    receiver_col = first_existing_col(df, ["receiver_cluster", "target_cluster", "receiver", "target"])
    if sender_col is None or receiver_col is None:
        return pd.DataFrame(columns=["edge_key"])
    d = df.copy()
    d["edge_key"] = [edge_key(s, r) for s, r in zip(d[sender_col], d[receiver_col])]
    score_col = first_existing_col(d, ["score", "edge_score", "final_score", "confidence", "expr_only", "n_pairs", "weight"])
    if score_col is None:
        score_candidates = cols_containing(d, ["score", "confidence", "expr"])
        score_col = score_candidates[0] if score_candidates else None
    d["_score"] = pd.to_numeric(d[score_col], errors="coerce").fillna(0) if score_col else 1.0
    tf_cols = [
        c for c in d.columns
        if "tf" in c.lower() and any(k in c.lower() for k in ["score", "support", "response", "z", "gate", "kept"])
    ]
    if tf_cols:
        d["_tf_support"] = pd.concat([pd.to_numeric(d[c], errors="coerce") for c in tf_cols], axis=1).mean(axis=1)
    else:
        d["_tf_support"] = np.nan
    return d.groupby("edge_key").agg(
        **{
            f"{prefix}_edge_count": ("edge_key", "size"),
            f"{prefix}_score_sum": ("_score", "sum"),
            f"{prefix}_score_mean": ("_score", "mean"),
            f"{prefix}_score_max": ("_score", "max"),
            f"{prefix}_receiver_tf_support_mean": ("_tf_support", "mean"),
        }
    ).reset_index()


def add_cci_features(edge_df: pd.DataFrame, cci: dict[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    out = edge_df.copy()
    for prefix, df in [("cci_all", cci.get("all")), ("cci_highconf", cci.get("highconf")), ("cci_tfkept", cci.get("tfkept"))]:
        agg = aggregate_edge_table(df, prefix)
        out = out.merge(agg, on="edge_key", how="left")
    count_cols = [c for c in out.columns if c.endswith("_edge_count")]
    score_cols = [c for c in out.columns if c.endswith("_score_sum") or c.endswith("_score_mean") or c.endswith("_score_max")]
    for c in count_cols + score_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["cci_all_supported"] = (out.get("cci_all_edge_count", 0) > 0).astype(float)
    out["cci_highconf_supported"] = (out.get("cci_highconf_edge_count", 0) > 0).astype(float)
    out["cci_tfkept_supported"] = (out.get("cci_tfkept_edge_count", 0) > 0).astype(float)
    pieces = []
    for c in ["cci_tfkept_score_sum", "cci_highconf_score_sum", "cci_all_score_sum"]:
        if c in out.columns:
            pieces.append(minmax01(out[c].values))
    out["cci_unified_score"] = np.nanmean(np.vstack(pieces), axis=0) if pieces else np.nan
    tf_cols = [c for c in out.columns if c.endswith("receiver_tf_support_mean")]
    out["cci_receiver_tf_support"] = (
        pd.concat([pd.to_numeric(out[c], errors="coerce") for c in tf_cols], axis=1).mean(axis=1)
        if tf_cols else np.nan
    )
    return out


def add_edge_pseudotime_features(edge_df: pd.DataFrame, edge_pt: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = edge_df.copy()
    if edge_pt is None or len(edge_pt) == 0:
        return out
    sender_col = first_existing_col(edge_pt, ["sender_cluster", "source_cluster", "sender", "source"])
    receiver_col = first_existing_col(edge_pt, ["receiver_cluster", "target_cluster", "receiver", "target"])
    if sender_col is None or receiver_col is None:
        return out
    d = edge_pt.copy()
    d["edge_key"] = [edge_key(s, r) for s, r in zip(d[sender_col], d[receiver_col])]
    keep_cols = ["edge_key"]
    rename: dict[str, str] = {}
    for src, dst in [
        ("t_sender", "edgept_t_sender"),
        ("t_receiver", "edgept_t_receiver"),
        ("delta_t_receiver_minus_sender", "edgept_delta_t"),
        ("delta_t", "edgept_delta_t"),
        ("dt", "edgept_delta_t"),
        ("edge_strength_log1p", "edgept_strength"),
        ("edge_weight", "edgept_strength"),
        ("edge_score", "edgept_strength"),
        ("score", "edgept_strength"),
        ("receiver_cluster_tf_response", "edgept_receiver_tf_response"),
        ("receiver_tf_response", "edgept_receiver_tf_response"),
        ("sender_cluster_sender_strength", "edgept_sender_strength"),
        ("receiver_cluster_receiver_strength", "edgept_receiver_strength"),
    ]:
        if src in d.columns and dst not in rename.values():
            keep_cols.append(src)
            rename[src] = dst
    d = d[keep_cols].rename(columns=rename)
    num_cols = [c for c in d.columns if c != "edge_key"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if num_cols:
        agg = d.groupby("edge_key")[num_cols].mean().reset_index()
        out = out.merge(agg, on="edge_key", how="left")
    if "edgept_delta_t" in out.columns:
        m = np.isfinite(pd.to_numeric(out["edgept_delta_t"], errors="coerce"))
        out.loc[m, "delta_t_receiver_minus_sender"] = pd.to_numeric(out.loc[m, "edgept_delta_t"], errors="coerce")
    if "edgept_receiver_tf_response" in out.columns:
        m = np.isfinite(pd.to_numeric(out["edgept_receiver_tf_response"], errors="coerce"))
        out.loc[m, "receiver_tf_response"] = pd.to_numeric(out.loc[m, "edgept_receiver_tf_response"], errors="coerce")
    return out


def add_rna_profile_similarity(edge_df: pd.DataFrame, prof: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = edge_df.copy()
    out["rna_cluster_similarity"] = np.nan
    out["rna_cluster_distance"] = np.nan
    out["rna_sender_strength_proxy"] = np.nan
    out["rna_receiver_strength_proxy"] = np.nan
    if prof is None or len(prof) == 0:
        return out
    ccol = first_existing_col(prof, ["cluster", "new_cluster", "new_leiden", "leiden", "cluster_id"])
    if ccol is None:
        return out
    d = prof.copy()
    d["_cluster"] = d[ccol].map(normalize_cluster_str).astype(str)
    bad_keywords = ["cluster", "size", "count", "n_cell", "barcode"]
    num_cols = []
    for c in d.columns:
        if c == "_cluster" or c == ccol:
            continue
        lc = c.lower()
        if any(k in lc for k in bad_keywords):
            continue
        x = pd.to_numeric(d[c], errors="coerce")
        if x.notna().sum() >= max(2, int(0.3 * len(d))):
            num_cols.append(c)
    if not num_cols:
        return out
    X = d[num_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    clusters = d["_cluster"].tolist()
    S = cosine_sim_matrix(X)
    sim_map = {f"{s}->{r}": float(S[i, j]) for i, s in enumerate(clusters) for j, r in enumerate(clusters)}
    out["rna_cluster_similarity"] = out["edge_key"].map(sim_map)
    out["rna_cluster_distance"] = 1.0 - out["rna_cluster_similarity"]
    strength = np.nanmean(np.abs(X), axis=1)
    smap = dict(zip(clusters, strength))
    out["rna_sender_strength_proxy"] = out["sender"].map(smap)
    out["rna_receiver_strength_proxy"] = out["receiver"].map(smap)
    return out


def add_rna_tf_matrix_support(edge_df: pd.DataFrame, rna_tf: Optional[pd.DataFrame], prof: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = edge_df.copy()
    out["rna_tf_regulatory_support"] = np.nan
    out["rna_tf_global_structure_support"] = np.nan
    if rna_tf is None or rna_tf.empty:
        return out
    Mdf = rna_tf.copy()
    Mdf.index = Mdf.index.astype(str)
    Mdf.columns = Mdf.columns.astype(str)
    M = Mdf.values.astype(float)
    global_structure = gini_nonnegative(np.maximum(M.ravel(), 0))
    out["rna_tf_global_structure_support"] = global_structure
    if prof is None or len(prof) == 0:
        out["rna_tf_regulatory_support"] = global_structure
        return out
    ccol = first_existing_col(prof, ["cluster", "new_cluster", "new_leiden", "leiden", "cluster_id"])
    if ccol is None:
        out["rna_tf_regulatory_support"] = global_structure
        return out
    d = prof.copy()
    d["_cluster"] = d[ccol].map(normalize_cluster_str).astype(str)
    prof_cols = set(map(str, d.columns))
    rna_names = [g for g in Mdf.index.astype(str).tolist() if g in prof_cols]
    tf_names = [t for t in Mdf.columns.astype(str).tolist() if t in prof_cols]
    if len(rna_names) < 2 or len(tf_names) < 2:
        out["rna_tf_regulatory_support"] = global_structure
        return out
    Msub = Mdf.loc[rna_names, tf_names].values.astype(float)
    Msub01 = minmax01(Msub.ravel()).reshape(Msub.shape)
    cluster_to_rna = {}
    cluster_to_tf = {}
    for _, row in d.iterrows():
        c = row["_cluster"]
        rv = pd.to_numeric(row[rna_names], errors="coerce").values.astype(float)
        tv = pd.to_numeric(row[tf_names], errors="coerce").values.astype(float)
        rv = np.maximum(np.nan_to_num(rv, nan=0.0), 0)
        tv = np.maximum(np.nan_to_num(tv, nan=0.0), 0)
        if rv.sum() > 0:
            rv = rv / (rv.sum() + 1e-12)
        if tv.sum() > 0:
            tv = tv / (tv.sum() + 1e-12)
        cluster_to_rna[c] = rv
        cluster_to_tf[c] = tv
    vals = []
    for _, row in out.iterrows():
        s = row["sender"]
        r = row["receiver"]
        if s in cluster_to_rna and r in cluster_to_tf:
            vals.append(float(cluster_to_rna[s] @ Msub01 @ cluster_to_tf[r]))
        else:
            vals.append(np.nan)
    out["rna_tf_regulatory_support"] = vals
    if np.isfinite(pd.to_numeric(out["rna_tf_regulatory_support"], errors="coerce")).sum() == 0:
        out["rna_tf_regulatory_support"] = global_structure
    return out


def add_stageA_stageB_support(edge_df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    out = edge_df.copy()
    out["stageA_dynamic_support"] = np.nan
    out["stageC_edge_support"] = np.nan
    out["stageB_path_support"] = np.nan

    # StageC edge metrics.
    cdf = safe_read_csv(paths.stageA_dir / "C_edge_metrics.csv")
    if cdf is not None and len(cdf) > 0:
        sender_col = first_existing_col(cdf, ["sender_cluster", "source_cluster", "sender", "source"])
        receiver_col = first_existing_col(cdf, ["receiver_cluster", "target_cluster", "receiver", "target"])
        if sender_col is not None and receiver_col is not None:
            d = cdf.copy()
            d["edge_key"] = [edge_key(s, r) for s, r in zip(d[sender_col], d[receiver_col])]
            score_cols = []
            for key in ["edge_weight", "delta_t", "receiver_tf", "edge_score", "rank_score", "score"]:
                score_cols.extend(cols_containing(d, [key]))
            score_cols = list(dict.fromkeys(score_cols))
            usable = [c for c in score_cols if pd.to_numeric(d[c], errors="coerce").notna().sum() > 0]
            if usable:
                tmp = d[["edge_key"] + usable].copy()
                for c in usable:
                    tmp[c] = minmax01(pd.to_numeric(tmp[c], errors="coerce").values)
                tmp["stageC_edge_support"] = tmp[usable].mean(axis=1)
                agg = tmp.groupby("edge_key")["stageC_edge_support"].mean().reset_index()
                out = out.drop(columns=["stageC_edge_support"], errors="ignore").merge(agg, on="edge_key", how="left")

    # StageA cluster-level dynamic metrics mapped to receiver cluster.
    adf = safe_read_csv(paths.stageA_dir / "A_cluster_feature_metrics.csv")
    if adf is not None and len(adf) > 0:
        ccol = first_existing_col(adf, ["cluster", "new_cluster", "leiden", "cluster_id"])
        if ccol is not None:
            metric_cols = []
            for key in ["amp", "coverage", "max_speed", "t50"]:
                metric_cols.extend(cols_containing(adf, [key]))
            metric_cols = list(dict.fromkeys(metric_cols))
            usable = [c for c in metric_cols if pd.to_numeric(adf[c], errors="coerce").notna().sum() > 0]
            if usable:
                tmp = adf[[ccol] + usable].copy()
                tmp["_cluster"] = tmp[ccol].map(normalize_cluster_str).astype(str)
                for c in usable:
                    tmp[c] = minmax01(pd.to_numeric(tmp[c], errors="coerce").values)
                tmp["stageA_dynamic_support"] = tmp[usable].mean(axis=1)
                amap = dict(zip(tmp["_cluster"], tmp["stageA_dynamic_support"]))
                out["stageA_dynamic_support"] = out["receiver"].map(amap)

    # StageB seed-edge path files.
    bfiles = glob_all(paths.stageB_dir, ["edge_enriched_table.csv", "seed_*_top*_edges.csv", "seed_*_top*_clusterpath_summary.csv"])
    bvals = []
    for p in bfiles:
        df = safe_read_csv(p)
        if df is None or len(df) == 0:
            continue
        sender_col = first_existing_col(df, ["sender_cluster", "source_cluster", "sender", "source"])
        receiver_col = first_existing_col(df, ["receiver_cluster", "target_cluster", "receiver", "target"])
        edge_col = first_existing_col(df, ["edge", "edge_name", "edge_key", "sender_receiver"])
        d = df.copy()

        if sender_col is not None and receiver_col is not None:
            d["edge_key"] = [edge_key(s, r) for s, r in zip(d[sender_col], d[receiver_col])]
        elif edge_col is not None:
            keys = []
            for x in d[edge_col].astype(str):
                try:
                    a, b = parse_edge_key(x)
                    keys.append(f"{a}->{b}")
                except Exception:
                    keys.append(None)
            d["edge_key"] = keys
        else:
            continue

        score_col = first_existing_col(d, ["similarity_score", "composite_score", "edge_rank_score", "rank_score", "score"])
        if score_col is None:
            scols = cols_containing(d, ["score", "similarity", "rank"])
            score_col = scols[0] if scols else None
        if score_col is None:
            continue

        d["_stageB"] = pd.to_numeric(d[score_col], errors="coerce")
        bvals.append(d[["edge_key", "_stageB"]].dropna())

    if bvals:
        b = pd.concat(bvals, axis=0)
        b["stageB_path_support"] = minmax01(b["_stageB"].values)
        agg = b.groupby("edge_key")["stageB_path_support"].mean().reset_index()
        out = out.drop(columns=["stageB_path_support"], errors="ignore").merge(agg, on="edge_key", how="left")

    for c in ["stageA_dynamic_support", "stageC_edge_support", "stageB_path_support"]:
        if c not in out.columns:
            out[c] = np.nan
    out["stageAB_support"] = pd.concat(
        [pd.to_numeric(out[c], errors="coerce") for c in ["stageA_dynamic_support", "stageC_edge_support", "stageB_path_support"]],
        axis=1,
    ).mean(axis=1)
    return out


# ============================================================
# 5) LABELS AND SCORES
# ============================================================
def add_global_endpoint_labels(edge_df: pd.DataFrame, label_mode: str = DEFAULT_LABEL_MODE) -> pd.DataFrame:
    d = edge_df.copy()

    cci_part = np.nanmax(np.vstack([
        pd.to_numeric(d.get("cci_tfkept_supported", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float),
        pd.to_numeric(d.get("cci_highconf_supported", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float),
        minmax01(pd.to_numeric(d.get("cci_unified_score", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float)),
    ]), axis=0)
    delta = pd.to_numeric(d.get("delta_t_receiver_minus_sender", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float)
    delta_pos = (delta > MIN_POSITIVE_DELTA).astype(float)
    delta_mag = minmax01(delta)

    tf_support = pd.concat([
        pd.to_numeric(d.get("receiver_tf_response", pd.Series(np.nan, index=d.index)), errors="coerce"),
        pd.to_numeric(d.get("cci_receiver_tf_support", pd.Series(np.nan, index=d.index)), errors="coerce"),
        pd.to_numeric(d.get("edgept_receiver_tf_response", pd.Series(np.nan, index=d.index)), errors="coerce") if "edgept_receiver_tf_response" in d.columns else pd.Series(np.nan, index=d.index),
    ], axis=1).mean(axis=1).values.astype(float)
    d["tf_response_unified"] = tf_support
    tf_part = minmax01(tf_support)

    rna_tf_part = minmax01(pd.to_numeric(d.get("rna_tf_regulatory_support", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float))
    stage_part = minmax01(pd.to_numeric(d.get("stageAB_support", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float))

    comp = []
    for arr in [cci_part, delta_pos, delta_mag, tf_part, rna_tf_part, stage_part]:
        arr2 = np.asarray(arr, dtype=float)
        arr2[~np.isfinite(arr2)] = 0.0
        comp.append(arr2)
    d["global_endpoint_score"] = (
        0.30 * comp[0]
        + 0.22 * comp[1]
        + 0.12 * comp[2]
        + 0.14 * comp[3]
        + 0.10 * comp[4]
        + 0.12 * comp[5]
    )

    tfkept_or_highconf = (
        pd.to_numeric(d.get("cci_tfkept_supported", pd.Series(0, index=d.index)), errors="coerce").fillna(0).values > 0
    ) | (
        pd.to_numeric(d.get("cci_highconf_supported", pd.Series(0, index=d.index)), errors="coerce").fillna(0).values > 0
    )
    positive_delta = delta > MIN_POSITIVE_DELTA

    if label_mode == "tfkept_directional":
        y = tfkept_or_highconf & positive_delta
    elif label_mode == "top_quantile":
        thr = np.nanquantile(d["global_endpoint_score"].values.astype(float), POSITIVE_SCORE_Q)
        y = d["global_endpoint_score"].values.astype(float) >= thr
    elif label_mode == "strict":
        thr = np.nanquantile(d["global_endpoint_score"].values.astype(float), POSITIVE_SCORE_Q)
        y = tfkept_or_highconf & positive_delta & (d["global_endpoint_score"].values.astype(float) >= thr)
        if np.sum(y) < max(3, int(0.02 * len(d))):
            y = tfkept_or_highconf & positive_delta
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    d["global_positive_edge"] = y.astype(int)

    stage_raw = pd.to_numeric(d.get("stageAB_support", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float)
    if np.isfinite(stage_raw).sum() > 0:
        mech_thr = np.nanquantile(stage_raw[np.isfinite(stage_raw)], MECHANISTIC_SCORE_Q)
        mech_support = stage_raw >= mech_thr
        y_mech = y & mech_support
        if np.sum(y_mech) < max(2, int(0.01 * len(d))):
            score_thr = np.nanquantile(d["global_endpoint_score"].values.astype(float), 0.80)
            y_mech = positive_delta & (d["global_endpoint_score"].values.astype(float) >= score_thr) & mech_support
    else:
        y_mech = np.zeros(len(d), dtype=bool)
    d["global_mechanistic_positive_edge"] = y_mech.astype(int)

    cci_score = pd.to_numeric(d.get("cci_unified_score", pd.Series(np.nan, index=d.index)), errors="coerce").values.astype(float)
    if np.isfinite(cci_score).sum() > 0:
        valid = np.isfinite(cci_score)
        cci_thr = np.nanquantile(cci_score[valid], CCI_CONSENSUS_Q)
        cci_consensus = tfkept_or_highconf & (cci_score >= cci_thr)
        if np.mean(cci_consensus) > 0.75:
            cci_thr = np.nanquantile(cci_score[valid], 0.80)
            cci_consensus = tfkept_or_highconf & (cci_score >= cci_thr)
        if np.sum(cci_consensus) < max(3, int(0.02 * len(d))):
            cci_thr = np.nanquantile(cci_score[valid], 0.50)
            cci_consensus = tfkept_or_highconf & (cci_score >= cci_thr)
    else:
        cci_consensus = tfkept_or_highconf
    d["cci_consensus_positive_edge"] = cci_consensus.astype(int)

    if np.isfinite(tf_support).sum() > 0:
        tf_thr = np.nanquantile(tf_support[np.isfinite(tf_support)], TF_RESPONSE_Q)
        tf_pos = (tf_support >= tf_thr) & positive_delta
    else:
        tf_pos = np.zeros(len(d), dtype=bool)
    d["tf_response_positive_edge"] = tf_pos.astype(int)
    return d


def add_stage_and_null_scores(edge_df: pd.DataFrame, seed: int = DEFAULT_BOOTSTRAP_SEED) -> pd.DataFrame:
    d = edge_df.copy()
    rng = np.random.default_rng(seed)

    def nz(col: str) -> np.ndarray:
        if col not in d.columns:
            return np.full(len(d), np.nan)
        return pd.to_numeric(d[col], errors="coerce").values.astype(float)

    def score_from_cols(cols: list[str], weights: Optional[list[float]] = None) -> np.ndarray:
        arrs = [minmax01(nz(c)) for c in cols]
        if not arrs:
            return np.full(len(d), np.nan)
        M = np.vstack(arrs)
        if weights is None:
            return np.nanmean(M, axis=0)
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        valid = np.isfinite(M).astype(float)
        M2 = np.nan_to_num(M, nan=0.0)
        denom = np.sum(w * valid, axis=0) + 1e-12
        return np.sum(w * M2, axis=0) / denom

    d["score_M0_RNA_only"] = score_from_cols(
        ["rna_cluster_similarity", "rna_cluster_distance", "rna_sender_strength_proxy", "rna_receiver_strength_proxy"],
        weights=[0.25, 0.25, 0.25, 0.25],
    )
    d["score_M1_RNA_TF"] = score_from_cols(
        ["score_M0_RNA_only", "receiver_tf_response", "tf_response_unified"],
        weights=[0.55, 0.25, 0.20],
    )
    d["score_M2_RNA_TF_matrix"] = score_from_cols(
        ["score_M1_RNA_TF", "rna_tf_regulatory_support", "rna_tf_global_structure_support"],
        weights=[0.45, 0.40, 0.15],
    )
    d["score_M3_CCI"] = score_from_cols(
        ["cci_unified_score", "cci_highconf_supported", "cci_tfkept_supported", "cci_receiver_tf_support"],
        weights=[0.45, 0.20, 0.25, 0.10],
    )

    delta = nz("delta_t_receiver_minus_sender")
    d["positive_delta_score"] = (delta > MIN_POSITIVE_DELTA).astype(float)
    d["negative_delta_score"] = (delta < -MIN_POSITIVE_DELTA).astype(float)
    d["delta_magnitude_score"] = minmax01(delta)
    d["negative_delta_magnitude_score"] = minmax01(-delta)
    d["score_M4_CCI_LTC"] = score_from_cols(
        ["score_M3_CCI", "positive_delta_score", "delta_magnitude_score", "edgept_strength"],
        weights=[0.45, 0.30, 0.15, 0.10],
    )
    d["score_M5_Full"] = score_from_cols(
        ["score_M4_CCI_LTC", "stageA_dynamic_support", "stageC_edge_support", "stageB_path_support", "stageAB_support"],
        weights=[0.50, 0.15, 0.15, 0.15, 0.05],
    )

    d["score_N0_Random"] = rng.random(len(d))

    sender_degree = d.groupby("sender")["cci_all_supported"].sum().to_dict() if "cci_all_supported" in d.columns else {}
    receiver_degree = d.groupby("receiver")["cci_all_supported"].sum().to_dict() if "cci_all_supported" in d.columns else {}
    deg_score = [sender_degree.get(row["sender"], 0.0) + receiver_degree.get(row["receiver"], 0.0) for _, row in d.iterrows()]
    d["score_N1_Degree_only"] = minmax01(deg_score)

    d["score_N2_Direction_flipped"] = score_from_cols(
        ["score_M3_CCI", "negative_delta_score", "negative_delta_magnitude_score"],
        weights=[0.45, 0.35, 0.20],
    )

    tf = nz("tf_response_unified")
    d["tf_response_shuffled"] = np.nan
    if np.isfinite(tf).sum() > 0:
        receiver_tf = (
            pd.DataFrame({"receiver": d["receiver"].astype(str), "tf": tf})
            .groupby("receiver")["tf"]
            .mean()
        )
        receivers = receiver_tf.index.to_numpy(copy=True)
        values = receiver_tf.values.copy()
        rng.shuffle(values)
        shuffled_map = dict(zip(receivers, values))
        d["tf_response_shuffled"] = d["receiver"].astype(str).map(shuffled_map)
    d["score_N3_TF_receiver_shuffled"] = score_from_cols(
        ["score_M0_RNA_only", "tf_response_shuffled"],
        weights=[0.60, 0.40],
    )

    cci_score = nz("cci_unified_score")
    cci_shuf = cci_score.copy()
    finite = np.where(np.isfinite(cci_shuf))[0]
    rng.shuffle(cci_shuf[finite])
    d["cci_score_shuffled"] = cci_shuf
    d["score_N4_CCI_shuffled_only"] = minmax01(d["cci_score_shuffled"].values)
    d["score_N5_CCI_shuffled_plus_LTC"] = score_from_cols(
        ["cci_score_shuffled", "positive_delta_score", "delta_magnitude_score"],
        weights=[0.45, 0.35, 0.20],
    )

    prev = None
    for name in SCORE_ORDER:
        col = f"score_{name}"
        if col not in d.columns:
            continue
        vals = pd.to_numeric(d[col], errors="coerce").values.astype(float)
        if np.isfinite(vals).sum() == 0 and prev is not None:
            d[col] = d[prev]
        elif np.isfinite(vals).sum() > 0:
            minv = np.nanmin(vals[np.isfinite(vals)])
            vals[~np.isfinite(vals)] = minv
            d[col] = vals
        prev = col
    return d


# ============================================================
# 6) BUILD TABLE
# ============================================================
def build_global_edge_endpoint_table(
    paths: Paths,
    missing: list[dict[str, Any]],
    include_self_edges: bool = False,
    label_mode: str = DEFAULT_LABEL_MODE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    clusters = load_cluster_list(paths, missing)
    if not clusters:
        raise RuntimeError("No clusters found. Check cluster_pseudotime.csv / cluster_profiles.csv / cluster_sizes.csv.")
    log(f"[INFO] clusters found: {len(clusters)} | {clusters}")
    edge_df = make_candidate_edges(clusters, include_self_edges=include_self_edges)
    log(f"[INFO] candidate directed edges: {len(edge_df)}")

    prof = load_cluster_profiles(paths)
    cl = load_cluster_pseudotime(paths, missing)
    edge_pt = load_edge_pseudotime(paths, missing)
    cci = load_cci_tables(paths, missing)
    rna_tf = load_rna_tf_matrix(paths, missing)

    edge_df = add_rna_profile_similarity(edge_df, prof)
    edge_df = add_cluster_pseudotime_features(edge_df, cl)
    edge_df = add_cci_features(edge_df, cci)
    edge_df = add_edge_pseudotime_features(edge_df, edge_pt)
    edge_df = add_rna_tf_matrix_support(edge_df, rna_tf, prof)
    edge_df = add_stageA_stageB_support(edge_df, paths)
    edge_df = add_global_endpoint_labels(edge_df, label_mode=label_mode)
    edge_df = add_stage_and_null_scores(edge_df, seed=seed)
    return edge_df


# ============================================================
# 7) EVALUATION
# ============================================================
def evaluate_score_for_endpoint(
    edge_df: pd.DataFrame,
    endpoint_col: str,
    score_name: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    score_col = f"score_{score_name}"
    y = edge_df[endpoint_col].values.astype(int)
    s = pd.to_numeric(edge_df[score_col], errors="coerce").values.astype(float)
    base_rate = float(np.mean(y == 1)) if len(y) else np.nan
    n_clusters = len(set(edge_df["sender"].astype(str)) | set(edge_df["receiver"].astype(str)))

    row: dict[str, Any] = {
        "endpoint": endpoint_col,
        "score_name": score_name,
        "score_label": SCORE_LABEL.get(score_name, score_name),
        "score_type": "null_control" if score_name in NULL_SCORE_ORDER else "progressive_stage",
        "score_index": SCORE_ORDER.index(score_name),
        "n_edges": int(len(edge_df)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "positive_rate": base_rate,
        "auroc": auc_binary(y, s),
        "auprc": auprc_binary(y, s),
        "brier": brier_score(y, s),
        "ece": expected_calibration_error(y, s),
    }
    row.update(rank_stats(y, s))
    row.update(separation_stats(y, s))
    row.update(bootstrap_metrics(edge_df, endpoint_col, score_col, n_bootstrap=n_bootstrap, seed=seed, k=10))

    row["score_delta_corr"] = safe_corr(s, edge_df.get("delta_t_receiver_minus_sender", np.nan))
    row["score_tf_response_corr"] = safe_corr(s, edge_df.get("tf_response_unified", np.nan))
    row["score_rna_tf_corr"] = safe_corr(s, edge_df.get("rna_tf_regulatory_support", np.nan))
    row["score_cci_corr"] = safe_corr(s, edge_df.get("cci_unified_score", np.nan))
    row["score_stageAB_corr"] = safe_corr(s, edge_df.get("stageAB_support", np.nan))

    for k in TOPK_LIST:
        p, r = precision_recall_at_k(y, s, k)
        row[f"precision_at_{k}"] = p
        row[f"recall_at_{k}"] = r
        row[f"hit_at_{k}"] = float(p > 0) if np.isfinite(p) else np.nan
        row[f"ndcg_at_{k}"] = ndcg_at_k(y, s, k)
        row[f"lift_at_{k}"] = float(p / base_rate) if base_rate > 0 and np.isfinite(p) else np.nan
        row[f"enrichment_at_{k}"] = row[f"lift_at_{k}"]

        tmp = edge_df.assign(_score=s)
        row[f"top{k}_delta_positive_frac"] = topk_mean_by_score(tmp, "_score", "positive_delta_score", k)
        row[f"top{k}_delta_mean"] = topk_mean_by_score(tmp, "_score", "delta_t_receiver_minus_sender", k)
        row[f"top{k}_delta_separation"] = top_bottom_separation(tmp, "_score", "delta_t_receiver_minus_sender", k)
        row[f"top{k}_temporal_dag_consistency"] = row[f"top{k}_delta_positive_frac"]
        row[f"top{k}_reciprocal_penalty"] = reciprocal_penalty_topk(tmp, "_score", k)

        row[f"top{k}_highconf_rate"] = topk_mean_by_score(tmp, "_score", "cci_highconf_supported", k)
        row[f"top{k}_tfkept_rate"] = topk_mean_by_score(tmp, "_score", "cci_tfkept_supported", k)
        row[f"top{k}_cci_score_mean"] = topk_mean_by_score(tmp, "_score", "cci_unified_score", k)

        row[f"top{k}_receiver_tf_support_mean"] = topk_mean_by_score(tmp, "_score", "tf_response_unified", k)
        row[f"top{k}_rna_tf_support_mean"] = topk_mean_by_score(tmp, "_score", "rna_tf_regulatory_support", k)

        row[f"top{k}_stageA_support_mean"] = topk_mean_by_score(tmp, "_score", "stageA_dynamic_support", k)
        row[f"top{k}_stageB_support_mean"] = topk_mean_by_score(tmp, "_score", "stageB_path_support", k)
        row[f"top{k}_stageAB_support_mean"] = topk_mean_by_score(tmp, "_score", "stageAB_support", k)

        cov = coverage_metrics_topk(tmp, "_score", k, n_clusters=n_clusters)
        row[f"top{k}_sender_coverage_frac"] = cov["sender_coverage_frac"]
        row[f"top{k}_receiver_coverage_frac"] = cov["receiver_coverage_frac"]
        row[f"top{k}_edge_pair_diversity_frac"] = cov["edge_pair_diversity_frac"]
        cov_mean = safe_mean([cov["sender_coverage_frac"], cov["receiver_coverage_frac"]])
        row[f"top{k}_coverage_adjusted_precision"] = p * cov_mean if np.isfinite(p) and np.isfinite(cov_mean) else np.nan

        mech_col = "global_mechanistic_positive_edge" if "global_mechanistic_positive_edge" in edge_df.columns else endpoint_col
        mech_base = edge_df[mech_col].mean()
        row[f"mechanism_lift_at_{k}"] = (
            topk_mean_by_score(tmp, "_score", mech_col, k) / (mech_base + 1e-12)
            if mech_base > 0 else np.nan
        )
    return row


def evaluate_all(edge_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        if endpoint not in edge_df.columns:
            continue
        for i, score_name in enumerate(SCORE_ORDER):
            score_col = f"score_{score_name}"
            if score_col not in edge_df.columns:
                continue
            rows.append(evaluate_score_for_endpoint(edge_df, endpoint, score_name, n_bootstrap, seed + 1000 * i))
    return pd.DataFrame(rows)


def stage_scores_long(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_cols = {
        "endpoint", "score_name", "score_label", "score_type", "score_index",
        "n_edges", "n_positive", "n_negative", "positive_rate",
    }
    for _, r in metrics.iterrows():
        for c in metrics.columns:
            if c in id_cols:
                continue
            rows.append({
                "endpoint": r["endpoint"],
                "score_name": r["score_name"],
                "score_label": r["score_label"],
                "score_type": r["score_type"],
                "metric": c,
                "value": r[c],
            })
    return pd.DataFrame(rows)


# ============================================================
# 8) PLOTS
# ============================================================
def save_metric_heatmap(metrics: pd.DataFrame, endpoint: str, out_dir: Path) -> None:
    d = metrics[metrics["endpoint"] == endpoint].copy()
    if d.empty:
        return
    cols = [c for c in PRIMARY_METRICS if c in d.columns]
    mat = d.set_index("score_label")[cols].astype(float)
    plot_mat = mat.copy()
    for c in plot_mat.columns:
        x = plot_mat[c].values.astype(float)
        if c in {"brier", "ece", "bootstrap_auroc_std", "bootstrap_auprc_std"}:
            mx = np.nanmax(x[np.isfinite(x)]) if np.isfinite(x).sum() else np.nan
            plot_mat[c] = 1.0 - (x / (mx + 1e-12)) if np.isfinite(mx) and mx > 0 else x
        elif c.endswith("penalty"):
            plot_mat[c] = 1.0 - np.clip(x, 0, 1)
        else:
            plot_mat[c] = np.clip(x, 0, 1)
    plt.figure(figsize=(max(14, 0.8 * len(cols)), max(6, 0.42 * len(plot_mat.index) + 2)), dpi=300)
    ax = plt.gca()
    data = plot_mat.values
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(plot_mat.index)))
    ax.set_yticklabels(plot_mat.index.tolist(), fontsize=9)
    ax.set_title(f"Global endpoint ablation metrics: {endpoint}", fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("display value 0-1; lower-is-better metrics inverted")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            raw = mat.values[i, j]
            if np.isfinite(raw):
                ax.text(j, i, f"{raw:.2f}", ha="center", va="center", fontsize=5.5)
    plt.tight_layout()
    png = out_dir / f"global_stage_metric_heatmap_{endpoint}.png"
    svg = out_dir / f"global_stage_metric_heatmap_{endpoint}.svg"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    log("[SAVE]", png)
    log("[SAVE]", svg)


def save_roc_pr_curves(edge_df: pd.DataFrame, endpoint: str, out_dir: Path) -> None:
    y = edge_df[endpoint].values.astype(int)
    plot_names = [
        "N0_Random",
        "N2_Direction_flipped",
        "N3_TF_receiver_shuffled",
        "N4_CCI_shuffled_only",
        "N5_CCI_shuffled_plus_LTC",
    ] + STAGE_ORDER

    plt.figure(figsize=(7.2, 5.8), dpi=300)
    for name in plot_names:
        col = f"score_{name}"
        if col not in edge_df.columns:
            continue
        s = pd.to_numeric(edge_df[col], errors="coerce").values.astype(float)
        fpr, tpr, _ = roc_curve(y, s)
        if len(fpr):
            auc = auc_binary(y, s)
            plt.plot(fpr, tpr, label=f"{SCORE_LABEL[name]} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC: recovering {endpoint}")
    plt.legend(fontsize=6)
    plt.tight_layout()
    png = out_dir / f"global_stage_roc_curve_{endpoint}.png"
    svg = out_dir / f"global_stage_roc_curve_{endpoint}.svg"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    log("[SAVE]", png)
    log("[SAVE]", svg)

    plt.figure(figsize=(7.2, 5.8), dpi=300)
    base_rate = float(np.mean(y == 1)) if len(y) else np.nan
    for name in plot_names:
        col = f"score_{name}"
        if col not in edge_df.columns:
            continue
        s = pd.to_numeric(edge_df[col], errors="coerce").values.astype(float)
        precision, recall, _ = pr_curve(y, s)
        if len(precision):
            ap = auprc_binary(y, s)
            plt.plot(recall, precision, label=f"{SCORE_LABEL[name]} AUPRC={ap:.3f}")
    if np.isfinite(base_rate):
        plt.axhline(base_rate, linestyle="--", linewidth=1, label=f"base={base_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR: recovering {endpoint}")
    plt.legend(fontsize=6)
    plt.tight_layout()
    png = out_dir / f"global_stage_pr_curve_{endpoint}.png"
    svg = out_dir / f"global_stage_pr_curve_{endpoint}.svg"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    log("[SAVE]", png)
    log("[SAVE]", svg)


def save_rank_enrichment(edge_df: pd.DataFrame, endpoint: str, out_dir: Path) -> None:
    y = edge_df[endpoint].values.astype(int)
    plot_names = [
        "N0_Random",
        "N2_Direction_flipped",
        "N3_TF_receiver_shuffled",
        "N4_CCI_shuffled_only",
        "N5_CCI_shuffled_plus_LTC",
    ] + STAGE_ORDER
    plt.figure(figsize=(9, 5.2), dpi=300)
    for name in plot_names:
        col = f"score_{name}"
        if col not in edge_df.columns:
            continue
        s = pd.to_numeric(edge_df[col], errors="coerce").values.astype(float)
        m = np.isfinite(s)
        if m.sum() == 0:
            continue
        order = np.argsort(-s[m])
        yy = y[m][order]
        cum = np.cumsum(yy == 1) / (np.arange(len(yy)) + 1)
        x = np.arange(1, len(yy) + 1)
        plt.plot(x, cum, label=SCORE_LABEL[name])
    plt.xscale("log")
    plt.xlabel("Top-ranked edges K")
    plt.ylabel("Cumulative positive fraction")
    plt.title(f"Rank enrichment: {endpoint}")
    plt.legend(fontsize=6)
    plt.tight_layout()
    png = out_dir / f"global_rank_enrichment_curve_{endpoint}.png"
    svg = out_dir / f"global_rank_enrichment_curve_{endpoint}.svg"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    log("[SAVE]", png)
    log("[SAVE]", svg)


def save_score_boxplots(edge_df: pd.DataFrame, endpoint: str, out_dir: Path) -> None:
    y = edge_df[endpoint].values.astype(int)
    names = STAGE_ORDER
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    positions = []
    labels = []
    data = []
    pos = 1.0
    for name in names:
        col = f"score_{name}"
        if col not in edge_df.columns:
            continue
        s = pd.to_numeric(edge_df[col], errors="coerce").values.astype(float)
        p = score_to_probability(s)
        data.append(p[y == 0])
        positions.append(pos)
        labels.append(f"{SCORE_LABEL[name]}\\nneg")
        data.append(p[y == 1])
        positions.append(pos + 0.35)
        labels.append(f"{SCORE_LABEL[name]}\\npos")
        pos += 1.1
    if not data:
        plt.close()
        return
    ax.boxplot(data, positions=positions, widths=0.25, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("min-max normalized stage score")
    ax.set_title(f"Score separation by endpoint label: {endpoint}")
    plt.tight_layout()
    png = out_dir / f"global_score_boxplot_{endpoint}.png"
    svg = out_dir / f"global_score_boxplot_{endpoint}.svg"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    log("[SAVE]", png)
    log("[SAVE]", svg)


# ============================================================
# 9) MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_base", default=str(RUN_BASE))
    ap.add_argument("--sample_id", default=SAMPLE_ID)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--label_mode", default=DEFAULT_LABEL_MODE, choices=["strict", "tfkept_directional", "top_quantile"])
    ap.add_argument("--include_self_edges", action="store_true")
    ap.add_argument("--n_bootstrap", type=int, default=DEFAULT_BOOTSTRAP_N)
    ap.add_argument("--bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args, unknown = ap.parse_known_args()
    if unknown:
        log("[INFO] ignored unknown CLI args:", unknown)

    run_base = Path(args.run_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Paths(run_base=run_base, sample_id=args.sample_id)

    log("[INFO] run_base:", run_base)
    log("[INFO] sample_id:", args.sample_id)
    log("[INFO] out_dir:", out_dir)
    log("[INFO] label_mode:", args.label_mode)
    log("[INFO] n_bootstrap:", args.n_bootstrap)

    missing: list[dict[str, Any]] = []
    edge_df = build_global_edge_endpoint_table(
        paths,
        missing,
        include_self_edges=args.include_self_edges,
        label_mode=args.label_mode,
        seed=args.bootstrap_seed,
    )
    metrics = evaluate_all(edge_df, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)
    metrics_long = stage_scores_long(metrics)
    missing_df = pd.DataFrame(missing)

    p_edge = out_dir / "global_edge_endpoint_table.csv"
    p_metrics = out_dir / "global_stage_metrics.csv"
    p_long = out_dir / "global_stage_scores_long.csv"
    p_missing = out_dir / "global_endpoint_missing_inputs.csv"

    edge_df.to_csv(p_edge, index=False)
    metrics.to_csv(p_metrics, index=False)
    metrics_long.to_csv(p_long, index=False)
    missing_df.to_csv(p_missing, index=False)

    log("[SAVE]", p_edge)
    log("[SAVE]", p_metrics)
    log("[SAVE]", p_long)
    log("[SAVE]", p_missing)

    for endpoint in ENDPOINTS:
        if endpoint in edge_df.columns:
            save_metric_heatmap(metrics, endpoint, out_dir)
            save_roc_pr_curves(edge_df, endpoint, out_dir)
            save_rank_enrichment(edge_df, endpoint, out_dir)
            save_score_boxplots(edge_df, endpoint, out_dir)

    log("[DONE]")
    for endpoint in ENDPOINTS:
        if endpoint in edge_df.columns:
            n_pos = int(edge_df[endpoint].sum())
            log(f"[SUMMARY] {endpoint}: edges={len(edge_df)} positives={n_pos} positive_rate={n_pos / max(len(edge_df), 1):.4f}")
    show_cols = [
        "endpoint", "score_label", "score_type", "auroc", "auprc", "mrr", "precision_at_10", "recall_at_10",
        "ndcg_at_10", "lift_at_10", "brier", "ece", "top10_delta_positive_frac",
        "top10_highconf_rate", "top10_tfkept_rate", "top10_stageAB_support_mean",
        "top10_reciprocal_penalty", "top10_sender_coverage_frac", "top10_receiver_coverage_frac",
        "bootstrap_auroc_std", "bootstrap_auprc_std", "bootstrap_top10_jaccard_mean",
    ]
    show_cols = [c for c in show_cols if c in metrics.columns]
    log(metrics[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
