#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import platform
import sys

from imas import __version__
from imas.config import ProjectPaths

OPTIONAL = ["anndata", "scanpy", "networkx", "umap", "gseapy", "igraph", "leidenalg"]


def main() -> None:
    paths = ProjectPaths.discover()
    optional = {}
    for name in OPTIONAL:
        try:
            module = importlib.import_module(name)
            optional[name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            optional[name] = f"unavailable: {type(exc).__name__}"
    report = {
        "imas_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "paths": {key: str(value) for key, value in paths.__dict__.items()},
        "optional_dependencies": optional,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
