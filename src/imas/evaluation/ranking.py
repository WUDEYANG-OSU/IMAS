"""Ranking and retrieval metrics for RNA, TF, RE, and edge recovery."""
from __future__ import annotations

import numpy as np


def _order(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size:
        raise ValueError("y_true and scores must have equal length")
    mask = np.isfinite(s) & np.isfinite(y)
    return y[mask][np.argsort(-s[mask], kind="mergesort")]


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _order(y_true, scores)
    k = min(max(int(k), 1), ranked.size)
    return float(np.mean(ranked[:k] > 0)) if ranked.size else float("nan")


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _order(y_true, scores)
    n_pos = int(np.sum(ranked > 0))
    if n_pos == 0:
        return float("nan")
    k = min(max(int(k), 1), ranked.size)
    return float(np.sum(ranked[:k] > 0) / n_pos)


def hit_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _order(y_true, scores)
    if ranked.size == 0:
        return float("nan")
    k = min(max(int(k), 1), ranked.size)
    return float(np.any(ranked[:k] > 0))


def reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    ranked = _order(y_true, scores)
    positions = np.flatnonzero(ranked > 0)
    return float(1.0 / (positions[0] + 1)) if positions.size else 0.0


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _order(y_true, scores).astype(float)
    if ranked.size == 0:
        return float("nan")
    k = min(max(int(k), 1), ranked.size)
    gains = np.maximum(ranked[:k], 0.0)
    discounts = np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(np.maximum(ranked, 0.0))[::-1][:k]
    idcg = float(np.sum(ideal / discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    ranked = _order(y_true, scores)
    if ranked.size == 0:
        return float("nan")
    prevalence = float(np.mean(ranked > 0))
    if prevalence <= 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / prevalence


def topk_jaccard(a_scores: np.ndarray, b_scores: np.ndarray, k: int) -> float:
    a = np.asarray(a_scores, dtype=float).reshape(-1)
    b = np.asarray(b_scores, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("score arrays must have equal length")
    k = min(max(int(k), 1), a.size)
    ia = set(np.argsort(-a, kind="mergesort")[:k].tolist())
    ib = set(np.argsort(-b, kind="mergesort")[:k].tolist())
    union = ia | ib
    return float(len(ia & ib) / len(union)) if union else float("nan")


def retrieval_metrics(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    return {
        f"hit_at_{k}": hit_at_k(y_true, scores, k),
        f"precision_at_{k}": precision_at_k(y_true, scores, k),
        f"recall_at_{k}": recall_at_k(y_true, scores, k),
        f"ndcg_at_{k}": ndcg_at_k(y_true, scores, k),
        f"lift_at_{k}": lift_at_k(y_true, scores, k),
        "mrr": reciprocal_rank(y_true, scores),
    }
