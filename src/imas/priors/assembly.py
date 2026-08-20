from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .node_index import build_global_node_index
from .schema import EDGE_COLUMNS, canonicalize_edge_table


def assemble_edge_tables(
    tables: Iterable[pd.DataFrame],
    *,
    node_index: Mapping[str, int] | None = None,
    aggregate_duplicates: str = "sum",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assemble heterogeneous edge shards into the shared pan-cancer graph schema."""
    canonical = [canonicalize_edge_table(table) for table in tables]
    if not canonical:
        return pd.DataFrame(columns=EDGE_COLUMNS + ["src_id", "dst_id"]), {}
    merged = pd.concat(canonical, ignore_index=True)
    keys = ["src_name", "dst_name", "rel_type", "split", "direction", "dataset"]
    if aggregate_duplicates == "sum":
        merged = merged.groupby(keys, as_index=False)["weight"].sum()
    elif aggregate_duplicates == "max":
        merged = merged.groupby(keys, as_index=False)["weight"].max()
    elif aggregate_duplicates == "first":
        merged = merged.drop_duplicates(keys, keep="first")
    else:
        raise ValueError("aggregate_duplicates must be sum, max or first")
    index = dict(node_index) if node_index is not None else build_global_node_index([merged])
    missing_names = sorted(
        (set(merged["src_name"]) | set(merged["dst_name"])) - set(index)
    )
    if missing_names:
        next_id = max(index.values(), default=-1) + 1
        for offset, name in enumerate(missing_names):
            index[name] = next_id + offset
    merged["src_id"] = merged["src_name"].map(index).astype(np.int64)
    merged["dst_id"] = merged["dst_name"].map(index).astype(np.int64)
    merged = merged.sort_values(
        ["rel_type", "split", "direction", "dataset", "src_id", "dst_id"]
    ).reset_index(drop=True)
    return merged, {str(k): int(v) for k, v in index.items()}
