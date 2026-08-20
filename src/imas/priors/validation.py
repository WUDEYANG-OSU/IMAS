from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED = {"src_name", "dst_name", "weight"}


def validate_edge_table(frame: pd.DataFrame, *, allow_self_edges: bool = True) -> dict[str, object]:
    if missing := _REQUIRED.difference(frame.columns):
        raise KeyError(f"Edge table missing columns: {sorted(missing)}")
    src = frame["src_name"].astype(str).str.strip()
    dst = frame["dst_name"].astype(str).str.strip()
    weights = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(dtype=float)
    if src.eq("").any() or dst.eq("").any():
        raise ValueError("Edge table contains empty node names")
    if not np.isfinite(weights).all():
        raise ValueError("Edge table contains non-finite weights")
    self_edges = int((src == dst).sum())
    if self_edges and not allow_self_edges:
        raise ValueError(f"Edge table contains {self_edges} self edges")
    duplicate_cols = [c for c in ["src_name", "dst_name", "rel_type", "split", "direction", "dataset"] if c in frame.columns]
    duplicates = int(frame.duplicated(duplicate_cols, keep=False).sum())
    return {
        "n_edges": int(len(frame)),
        "n_src": int(src.nunique()),
        "n_dst": int(dst.nunique()),
        "self_edges": self_edges,
        "duplicate_rows": duplicates,
        "weight_min": float(np.min(weights)) if len(weights) else np.nan,
        "weight_median": float(np.median(weights)) if len(weights) else np.nan,
        "weight_max": float(np.max(weights)) if len(weights) else np.nan,
    }


def validate_reverse_pair(
    forward: pd.DataFrame,
    reverse: pd.DataFrame,
    *,
    weight_tolerance: float = 1e-8,
) -> dict[str, int | float]:
    f = forward[["src_name", "dst_name", "weight"]].copy()
    r = reverse[["src_name", "dst_name", "weight"]].copy()
    f["key"] = f["src_name"].astype(str) + "\x1f" + f["dst_name"].astype(str)
    r["key"] = r["dst_name"].astype(str) + "\x1f" + r["src_name"].astype(str)
    joined = f.merge(r, on="key", how="outer", suffixes=("_fwd", "_rev"), indicator=True)
    matched = joined["_merge"].eq("both")
    weight_ok = np.isclose(
        pd.to_numeric(joined.loc[matched, "weight_fwd"], errors="coerce"),
        pd.to_numeric(joined.loc[matched, "weight_rev"], errors="coerce"),
        atol=float(weight_tolerance), rtol=0.0,
    )
    return {
        "forward_edges": int(len(f)),
        "reverse_edges": int(len(r)),
        "matched_edges": int(matched.sum()),
        "missing_reverse": int((joined["_merge"] == "left_only").sum()),
        "extra_reverse": int((joined["_merge"] == "right_only").sum()),
        "weight_mismatches": int((~weight_ok).sum()),
    }
