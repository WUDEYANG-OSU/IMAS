from __future__ import annotations
import numpy as np
import pandas as pd


def add_temporal_annotations(communication_annotations: list[dict], edge_timing: pd.DataFrame,
                             cluster_time: pd.DataFrame | None = None, top_n: int = 16):
    if not communication_annotations:return []
    score_col=next((c for c in ["edge_score","score","weight","confidence"] if c in edge_timing.columns),None)
    timing=edge_timing.copy(); timing["_score"]=pd.to_numeric(timing[score_col],errors="coerce").fillna(0.0) if score_col else 1.0
    result=[]
    for annotation in communication_annotations:
        item=dict(annotation); candidates=timing
        if "receptor" in timing.columns:candidates=candidates[timing.receptor.astype(str)==str(annotation.get("receptor",""))]
        if len(candidates):
            row=candidates.nlargest(1,"_score").iloc[0]
            for key in ["t_sender","t_receiver","delta_t_receiver_minus_sender"]:
                item[key]=float(row[key]) if key in row and pd.notna(row[key]) else np.nan
        result.append(item)
    return result[:top_n]
