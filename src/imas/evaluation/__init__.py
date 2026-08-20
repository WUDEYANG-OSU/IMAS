"""Shared evaluation metrics for IMAS benchmarks and ablations."""
from .classification import BinaryMetricResult, auroc, average_precision, evaluate_binary
from .ranking import (
    hit_at_k,
    lift_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    topk_jaccard,
)
from .correlation import pearson, spearman
from .calibration import brier_score, expected_calibration_error
from .perturbation import sign3, trend_consistency
from .stability import bootstrap_metric, summarize_bootstrap
from .endpoints import evaluate_endpoint, minmax_probability

__all__ = [
    "BinaryMetricResult", "auroc", "average_precision", "evaluate_binary",
    "hit_at_k", "lift_at_k", "ndcg_at_k", "precision_at_k", "recall_at_k",
    "reciprocal_rank", "retrieval_metrics", "topk_jaccard", "pearson", "spearman",
    "brier_score", "expected_calibration_error", "sign3", "trend_consistency",
    "bootstrap_metric", "summarize_bootstrap", "evaluate_endpoint", "minmax_probability",
]
