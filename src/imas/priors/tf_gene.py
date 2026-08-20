from __future__ import annotations

import pandas as pd


def build_tf_gene_edges(
    tf_re: pd.DataFrame,
    re_gene: pd.DataFrame,
    *,
    topk_per_tf: int | None = None,
    normalize_per_tf: bool = False,
) -> pd.DataFrame:
    """Compose TF→RE and RE→Gene evidence into TF→Gene edges."""
    left = tf_re.rename(columns={"src_name": "tf", "dst_name": "re", "weight": "w_tf_re"})
    right = re_gene.rename(columns={"src_name": "re", "dst_name": "gene", "weight": "w_re_gene"})
    joined = left[["tf", "re", "w_tf_re"]].merge(
        right[["re", "gene", "w_re_gene"]], on="re", how="inner"
    )
    joined["contribution"] = joined["w_tf_re"] * joined["w_re_gene"]
    out = joined.groupby(["tf", "gene"], as_index=False)["contribution"].sum()
    out = out.rename(columns={"tf": "src_name", "gene": "dst_name", "contribution": "weight"})
    if topk_per_tf is not None:
        out = (
            out.sort_values(["src_name", "weight", "dst_name"], ascending=[True, False, True])
            .groupby("src_name", group_keys=False)
            .head(int(topk_per_tf))
        )
    if normalize_per_tf and len(out):
        denom = out.groupby("src_name")["weight"].transform("sum").clip(lower=1e-12)
        out["weight"] = out["weight"] / denom
    return out.reset_index(drop=True)
