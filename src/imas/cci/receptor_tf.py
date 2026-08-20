from __future__ import annotations
import numpy as np
import pandas as pd


def build_receptor_tf_signature(receptor_expression: np.ndarray, tf_scores: np.ndarray,
                                receptors: list[str], tfs: list[str], min_abs_corr: float = 0.18,
                                top_tfs_per_receptor: int = 10) -> pd.DataFrame:
    rows = []
    for i, receptor in enumerate(receptors):
        x = receptor_expression[:, i]
        for j, tf in enumerate(tfs):
            y = tf_scores[:, j]
            finite = np.isfinite(x) & np.isfinite(y)
            corr = float(np.corrcoef(x[finite], y[finite])[0,1]) if finite.sum() >= 3 else 0.0
            if np.isfinite(corr) and abs(corr) >= min_abs_corr:
                rows.append({"receptor": receptor, "tf": tf, "corr": corr, "abs_corr": abs(corr)})
    frame = pd.DataFrame(rows)
    if frame.empty: return frame
    return (frame.sort_values(["receptor", "abs_corr"], ascending=[True, False])
                 .groupby("receptor", group_keys=False).head(top_tfs_per_receptor).reset_index(drop=True))
