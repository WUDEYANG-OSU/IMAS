from __future__ import annotations
import numpy as np
import pandas as pd


def build_cluster_profiles(expression: np.ndarray, clusters, gene_names) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = np.asarray(expression, np.float32); clusters = np.asarray(clusters).astype(str)
    rows, sizes = [], []
    for cluster in sorted(np.unique(clusters)):
        mask = clusters == cluster; mean = expression[mask].mean(0); pct = (expression[mask] > 0).mean(0)
        sizes.append({"cluster": cluster, "n_cells": int(mask.sum())})
        rows.extend({"cluster": cluster, "gene": str(gene), "mean_expr": float(m), "pct_expr": float(p)}
                    for gene, m, p in zip(gene_names, mean, pct))
    return pd.DataFrame(rows), pd.DataFrame(sizes)
