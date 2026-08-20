from __future__ import annotations
import numpy as np
import pandas as pd


def score_cluster_interactions(profile_table: pd.DataFrame, lr_table: pd.DataFrame,
                               min_mean: float = 0.0, min_pct: float = 0.0) -> pd.DataFrame:
    required = {"cluster", "gene", "mean_expr", "pct_expr"}
    if not required.issubset(profile_table.columns): raise KeyError(required - set(profile_table.columns))
    lookup = profile_table.set_index(["cluster", "gene"])[["mean_expr", "pct_expr"]].to_dict("index")
    clusters = sorted(profile_table["cluster"].astype(str).unique())
    rows = []
    for sender in clusters:
        for receiver in clusters:
            for lr in lr_table.itertuples(index=False):
                ligand, receptor = str(lr.ligand), str(lr.receptor)
                ls, rs = lookup.get((sender, ligand)), lookup.get((receiver, receptor))
                if ls is None or rs is None: continue
                if ls["mean_expr"] < min_mean or rs["mean_expr"] < min_mean: continue
                if ls["pct_expr"] < min_pct or rs["pct_expr"] < min_pct: continue
                score = float(ls["mean_expr"] * rs["mean_expr"] * np.sqrt(ls["pct_expr"] * rs["pct_expr"]))
                rows.append({"sender_cluster": sender, "receiver_cluster": receiver,
                             "ligand": ligand, "receptor": receptor, "expr_only": score, "score": score})
    return pd.DataFrame(rows)
