from __future__ import annotations

import pandas as pd


def build_rna_tf_edges(
    rna_re: pd.DataFrame,
    tf_re: pd.DataFrame,
    *,
    topk: int | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compose RNA→RE and TF→RE evidence into RNA→TF edges."""
    left = rna_re.rename(columns={"dst_name": "re", "weight": "w_rna_re"})
    right = tf_re.rename(columns={"src_name": "tf", "dst_name": "re", "weight": "w_tf_re"})
    joined = left[["src_name", "re", "w_rna_re"]].merge(
        right[["tf", "re", "w_tf_re"]], on="re", how="inner"
    )
    joined["contribution"] = joined["w_rna_re"] * joined["w_tf_re"]
    out = (
        joined.groupby(["src_name", "tf"], as_index=False)["contribution"].sum()
        .rename(columns={"tf": "dst_name", "contribution": "weight"})
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
