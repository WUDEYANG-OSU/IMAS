from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import re

import pandas as pd


def infer_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Find the first matching column, using a case-insensitive fallback."""
    for name in candidates:
        if name in frame.columns:
            return name
    lower = {str(name).lower(): str(name) for name in frame.columns}
    for name in candidates:
        if str(name).lower() in lower:
            return lower[str(name).lower()]
    return None


def normalize_manifest_path(value: object, manifest_dir: str | Path) -> Path | None:
    """Resolve Windows, WSL and relative paths recorded in an IMAS manifest."""
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return None
    if re.match(r"^[A-Za-z]:[\\/].*", raw):
        return Path(raw)
    normalized = raw.replace("\\", "/")
    wsl = re.match(r"^/mnt/([A-Za-z])/(.*)$", normalized)
    if wsl:
        drive = wsl.group(1).upper()
        tail = wsl.group(2).replace("/", "\\")
        return Path(drive + ":\\" + tail)
    path = Path(raw)
    return path if path.is_absolute() else (Path(manifest_dir) / path).resolve()


def find_manifest(dataset_root: str | Path, kind: str) -> Path | None:
    """Locate ``processed/<kind>/manifest_<kind>.tsv`` for one dataset."""
    root = Path(dataset_root)
    expected = root / "processed" / kind / f"manifest_{kind.lower()}.tsv"
    return expected if expected.exists() else None


def load_manifest(
    path: str | Path,
    *,
    required_columns: Iterable[str] = (),
    resolve_path_columns: Iterable[str] = (),
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Load, validate and optionally resolve path columns in a TSV manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [name for name in required_columns if name not in frame.columns]
    if missing:
        raise KeyError(f"{path.name}: missing required columns {missing}")
    for column in resolve_path_columns:
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: normalize_manifest_path(value, path.parent)
            )
    if sort_by is not None:
        if sort_by not in frame.columns:
            raise KeyError(f"{path.name}: missing sort column {sort_by!r}")
        frame = frame.sort_values(sort_by).reset_index(drop=True)
    return frame
