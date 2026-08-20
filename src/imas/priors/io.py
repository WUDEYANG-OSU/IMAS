from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import gzip
import os
import tempfile

import pandas as pd


@contextmanager
def open_text(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, mode, encoding=None if "b" in mode else "utf-8") as handle:
            yield handle
    else:
        with open(path, mode, encoding=None if "b" in mode else "utf-8") as handle:
            yield handle


def read_edge_table(path: str | Path, *, chunksize: int | None = None, **kwargs):
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt", ".gz"} else ","
    return pd.read_csv(path, sep=sep, chunksize=chunksize, low_memory=False, **kwargs)


def atomic_write_tsv(frame: pd.DataFrame, path: str | Path, *, compression: str | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp" + (".gz" if out.suffix == ".gz" else "")
    temp = out.with_name(out.name + suffix)
    frame.to_csv(temp, sep="\t", index=False, compression=compression or ("gzip" if out.suffix == ".gz" else None))
    os.replace(temp, out)
    return out
