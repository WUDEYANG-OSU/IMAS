from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EDGE_COLUMNS = [
    "src_name",
    "dst_name",
    "weight",
    "rel_type",
    "split",
    "direction",
    "dataset",
]


@dataclass(frozen=True)
class EdgeSchema:
    rel_type: str
    split: str = "train"
    direction: str = "fwd"
    dataset: str = ""


def canonicalize_edge_table(
    frame: pd.DataFrame,
    *,
    rel_type: str | None = None,
    split: str = "train",
    direction: str = "fwd",
    dataset: str = "",
    src_candidates: Iterable[str] = ("src_name", "src", "source", "src_id"),
    dst_candidates: Iterable[str] = ("dst_name", "dst", "target", "dst_id"),
    weight_candidates: Iterable[str] = ("weight", "w0", "score", "evidence"),
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Convert heterogeneous prior tables to the common IMAS edge schema."""
    src = next((name for name in src_candidates if name in frame.columns), None)
    dst = next((name for name in dst_candidates if name in frame.columns), None)
    weight = next((name for name in weight_candidates if name in frame.columns), None)
    if src is None or dst is None:
        raise KeyError("Could not infer source and destination columns")
    out = pd.DataFrame(
        {
            "src_name": frame[src].astype(str).str.strip(),
            "dst_name": frame[dst].astype(str).str.strip(),
            "weight": (
                pd.to_numeric(frame[weight], errors="coerce")
                if weight is not None
                else np.ones(len(frame), dtype=np.float64)
            ),
        }
    )
    inferred_rel = rel_type
    if inferred_rel is None and "rel_type" in frame.columns:
        values = frame["rel_type"].dropna().astype(str).unique()
        inferred_rel = values[0] if len(values) == 1 else "mixed"
    out["rel_type"] = inferred_rel or "unknown"
    out["split"] = frame["split"].astype(str) if "split" in frame.columns else str(split)
    out["direction"] = (
        frame["direction"].astype(str) if "direction" in frame.columns else str(direction)
    )
    out["dataset"] = frame["dataset"].astype(str) if "dataset" in frame.columns else str(dataset)
    valid = (
        out["src_name"].ne("")
        & out["dst_name"].ne("")
        & np.isfinite(out["weight"].to_numpy(dtype=float))
    )
    if drop_invalid:
        out = out.loc[valid].copy()
    elif not bool(valid.all()):
        raise ValueError("Edge table contains empty nodes or non-finite weights")
    return out[EDGE_COLUMNS].reset_index(drop=True)
