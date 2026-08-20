from __future__ import annotations
import numpy as np


def build_cluster_response_weights(expression_mean: np.ndarray, expression_fraction: np.ndarray,
                                   gene_impact: np.ndarray, alpha: float = 1.0, beta: float = 1.55):
    """Cluster-by-gene response weight used by the top-response heatmap workflow."""
    mean=np.asarray(expression_mean,float); fraction=np.asarray(expression_fraction,float); impact=np.asarray(gene_impact,float).reshape(1,-1)
    specificity=mean/(mean.mean(0,keepdims=True)+1e-8)
    raw=np.power(np.clip(impact,0,None)+1e-12,alpha)*np.power(np.clip(specificity,0,None)*np.clip(fraction,0,1)+1e-12,beta)
    return raw
