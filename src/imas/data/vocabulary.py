from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np


def write_vocab_txt(path: str | Path, values: Iterable[object]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            text = str(value).strip()
            if text:
                handle.write(text + "\n")
    return out


def _strip_prefix(name: object, prefix: str) -> str:
    text = str(name).strip()
    expected = f"{prefix}:"
    return text[len(expected):] if text.startswith(expected) else text


def build_global_vocab_from_node_index(
    node_index: Mapping[str, int],
    prefix: str,
) -> list[str]:
    """Return names with a selected node prefix, ordered by global node ID."""
    selected = [
        (int(node_id), _strip_prefix(name, prefix))
        for name, node_id in node_index.items()
        if str(name).startswith(f"{prefix}:")
    ]
    selected.sort(key=lambda item: item[0])
    return [name for _, name in selected]


def build_local_to_global_index(
    local_names: Iterable[object],
    global_name_to_index: Mapping[str, int],
    *,
    prefix: str = "",
    case_sensitive: bool = True,
) -> np.ndarray:
    """Map local feature names to a global vocabulary, returning ``-1`` for misses."""
    if case_sensitive:
        lookup = {str(k): int(v) for k, v in global_name_to_index.items()}
        transform = str
    else:
        lookup = {str(k).upper(): int(v) for k, v in global_name_to_index.items()}
        transform = lambda x: str(x).upper()
    # Materialise once so generators remain supported.
    names = local_names if isinstance(local_names, list) else list(local_names)
    out = np.full(len(names), -1, dtype=np.int64)
    for i, name in enumerate(names):
        text = str(name).strip()
        if prefix and not text.startswith(f"{prefix}:"):
            text = f"{prefix}:{text}"
        key = transform(text)
        if key in lookup:
            out[i] = lookup[key]
    return out
