from __future__ import annotations
import numpy as np
import pandas as pd
from imas.utils.stats import rank01


def build_candidate_cascades(cci: pd.DataFrame, receptor_tf: pd.DataFrame, deg: pd.DataFrame,
                             weights=(0.35,0.30,0.20,0.10,0.05), axis_hint: dict[str,float] | None = None,
                             hidden_support: dict[str,float] | None = None):
    axis_hint,hidden_support=axis_hint or {},hidden_support or {}
    merged=cci.merge(receptor_tf,on="receptor",how="inner").merge(deg[["gene","log2FC_tumor_vs_normal"]],left_on="tf",right_on="gene",how="left")
    cci_col=next(c for c in ["score","edge_score","expr_only","confidence"] if c in merged.columns)
    tf_col=next(c for c in ["corr","tf_score","score_y","weight"] if c in merged.columns)
    merged["cci_rank"]=rank01(pd.to_numeric(merged[cci_col],errors="coerce").fillna(0).to_numpy())
    merged["tf_rank"]=rank01(pd.to_numeric(merged[tf_col],errors="coerce").fillna(0).abs().to_numpy())
    merged["deg_rank"]=rank01(pd.to_numeric(merged.log2FC_tumor_vs_normal,errors="coerce").fillna(0).abs().to_numpy())
    merged["axis_score"]=merged.tf.astype(str).map(axis_hint).fillna(0.0); merged["hidden_score"]=merged.tf.astype(str).map(hidden_support).fillna(0.0)
    merged["candidate_score"]=sum(w*merged[col] for w,col in zip(weights,["cci_rank","tf_rank","deg_rank","axis_score","hidden_score"]))
    return merged.sort_values("candidate_score",ascending=False).reset_index(drop=True)
