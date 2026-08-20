from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def load_node_index(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(name): int(node_id) for name, node_id in payload.items()}


def save_node_index(path: str | Path, index: Mapping[str, int]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(((str(k), int(v)) for k, v in index.items()), key=lambda x: x[1]))
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)
    return out


def invert_node_index(index: Mapping[str, int]) -> dict[int, str]:
    inverse: dict[int, str] = {}
    for name, node_id in index.items():
        node_id = int(node_id)
        if node_id in inverse and inverse[node_id] != str(name):
            raise ValueError(f"Node ID {node_id} maps to multiple names")
        inverse[node_id] = str(name)
    return inverse


def ensure_node(index: dict[str, int], name: str) -> int:
    key = str(name).strip()
    if not key:
        raise ValueError("Node name may not be empty")
    if key not in index:
        index[key] = max(index.values(), default=-1) + 1
    return int(index[key])


def build_global_node_index(tables, *, start: int = 0) -> dict[str, int]:
    names: set[str] = set()
    for table in tables:
        names.update(table["src_name"].astype(str))
        names.update(table["dst_name"].astype(str))
    return {name: i for i, name in enumerate(sorted(names), start=int(start))}
