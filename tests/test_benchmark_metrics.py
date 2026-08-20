from __future__ import annotations

import numpy as np

from imas.benchmark import (
    cpa_style_shared_effect,
    dependent_structure_score,
    gears_style_graph_propagation,
    get_ablation_mode,
    get_regime_spec,
    scgpt_style_masking,
)
from imas.evaluation import (
    auroc,
    average_precision,
    evaluate_endpoint,
    ndcg_at_k,
    spearman,
    trend_consistency,
)


def test_binary_metrics_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(auroc(y, s), 1.0)
    assert np.isclose(average_precision(y, s), 1.0)
    assert np.isclose(ndcg_at_k(y, s, 2), 1.0)


def test_spearman_with_ties():
    x = np.array([1.0, 2.0, 2.0, 4.0])
    y = np.array([10.0, 20.0, 20.0, 40.0])
    assert np.isclose(spearman(x, y), 1.0)


def test_trend_consistency():
    truth = np.array([-2.0, 0.0, 3.0])
    pred = np.array([-1.0, 0.0, 2.0])
    out = trend_consistency(truth, pred)
    assert np.isclose(out["direction_accuracy_3class"], 1.0)
    assert np.isclose(out["balanced_accuracy_up_down"], 1.0)


def test_style_baselines_shapes_and_target_mask():
    x = np.array([[2.0, 1.0, 0.0], [4.0, 2.0, 3.0]], dtype=np.float32)
    masked = scgpt_style_masking(x, 0)
    cpa = cpa_style_shared_effect(x, 0)
    adjacency = np.eye(3, dtype=np.float32)
    gears = gears_style_graph_propagation(x, 0, adjacency)
    for out in (masked, cpa, gears):
        assert out.shape == x.shape
        assert np.allclose(out[:, 0], 0.0)


def test_regime_and_ablation_definitions():
    assert get_regime_spec("scrna_only").enable_re_branch is False
    assert get_regime_spec("multiomic_only").enable_selected_peak_residual is True
    assert get_ablation_mode("no_cluster_correction").cluster_correction is False
    assert get_ablation_mode("full_stageB").cluster_correction is True


def test_depstruct_and_endpoint():
    x = np.array([0.0, 1.0])
    score = dependent_structure_score(x, x, x, x)
    assert np.allclose(score, x)
    result = evaluate_endpoint(np.array([0, 1]), np.array([0.0, 1.0]), k=1)
    assert np.isclose(result["auroc"], 1.0)
