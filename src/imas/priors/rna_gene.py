from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_rna_gene_edges(
    counts: np.ndarray,
    gene_names: Sequence[object],
    *,
    rna_node: str,
    aggregate: str = "sum",
    min_weight: float = 0.0,
    topk: int | None = None,
) -> pd.DataFrame:
    """Aggregate a dataset-level RNA node to gene-expression support edges."""
    matrix = np.asarray(counts)
    if matrix.ndim != 2 or matrix.shape[1] != len(gene_names):
        raise ValueError("counts must be cells×genes and match gene_names")
    if aggregate == "sum":
        values = np.nansum(matrix, axis=0)
    elif aggregate == "mean":
        values = np.nanmean(matrix, axis=0)
    elif aggregate == "detected_fraction":
        values = np.mean(matrix > 0, axis=0)
    else:
        raise ValueError(f"Unsupported aggregate: {aggregate}")
    out = pd.DataFrame(
        {
            "src_name": str(rna_node),
            "dst_name": [f"Gene:{str(g).strip()}" for g in gene_names],
            "weight": values.astype(np.float64),
        }
    )
    out = out[np.isfinite(out["weight"]) & (out["weight"] > float(min_weight))]
    if topk is not None:
        out = out.sort_values(["weight", "dst_name"], ascending=[False, True]).head(int(topk))
    return out.reset_index(drop=True)
