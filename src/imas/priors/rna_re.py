from __future__ import annotations

import pandas as pd


def build_rna_re_edges(
    rna_gene: pd.DataFrame,
    re_gene: pd.DataFrame,
    *,
    topk: int | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compose RNA→Gene and RE→Gene evidence into RNA→RE edges."""
    left = rna_gene.rename(columns={"dst_name": "gene", "weight": "w_rna_gene"})
    right = re_gene.rename(columns={"src_name": "re", "dst_name": "gene", "weight": "w_re_gene"})
    joined = left[["src_name", "gene", "w_rna_gene"]].merge(
        right[["re", "gene", "w_re_gene"]], on="gene", how="inner"
    )
    joined["contribution"] = joined["w_rna_gene"] * joined["w_re_gene"]
    out = (
        joined.groupby(["src_name", "re"], as_index=False)["contribution"].sum()
        .rename(columns={"re": "dst_name", "contribution": "weight"})
    )
    if topk is not None:
        out = (
            out.sort_values(["src_name", "weight", "dst_name"], ascending=[True, False, True])
            .groupby("src_name", group_keys=False)
            .head(int(topk))
        )
    if normalize and len(out):
        denom = out.groupby("src_name")["weight"].transform("sum").clip(lower=1e-12)
        out["weight"] = out["weight"] / denom
    return out.reset_index(drop=True)
