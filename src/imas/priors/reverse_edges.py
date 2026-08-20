from __future__ import annotations

import pandas as pd


def reverse_edge_table(
    frame: pd.DataFrame,
    *,
    rel_type: str | None = None,
    direction: str = "rev",
) -> pd.DataFrame:
    """Reverse an edge table exactly once while preserving evidence columns."""
    required = {"src_name", "dst_name", "weight"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"Edge table missing columns: {sorted(missing)}")
    out = frame.copy()
    src = out["src_name"].copy()
    out["src_name"] = out["dst_name"].to_numpy()
    out["dst_name"] = src.to_numpy()
    if rel_type is not None:
        out["rel_type"] = str(rel_type)
    if "direction" in out.columns:
        out["direction"] = str(direction)
    return out.reset_index(drop=True)
