from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .profiles import build_cluster_profiles
from .scoring import score_cluster_interactions


@dataclass
class CCIConfig:
    min_mean_expression: float = 0.0
    min_fraction_expressing: float = 0.0
    high_confidence_quantile: float = 0.75


def run_cci(expression: np.ndarray, clusters, gene_names, lr_table: pd.DataFrame, config: CCIConfig = CCIConfig()):
    profiles, sizes = build_cluster_profiles(expression, clusters, gene_names)
    interactions = score_cluster_interactions(profiles, lr_table, config.min_mean_expression, config.min_fraction_expressing)
    if interactions.empty: high = interactions.copy()
    else: high = interactions[interactions["score"] >= interactions["score"].quantile(config.high_confidence_quantile)].copy()
    return {"profiles": profiles, "cluster_sizes": sizes, "interactions": interactions, "high_confidence": high}
