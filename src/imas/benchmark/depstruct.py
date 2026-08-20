"""Dependent-structure score used in multiomic/scRNA regime diagnostics."""
from __future__ import annotations

import numpy as np


DEFAULT_WEIGHTS = {
    "regulatory_support": 0.30,
    "re_traceability": 0.30,
    "dependency_core": 0.25,
    "response_anchor": 0.15,
}


def dependent_structure_score(
    regulatory_support: np.ndarray,
    re_traceability: np.ndarray,
    dependency_core: np.ndarray,
    response_anchor: np.ndarray,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    w = DEFAULT_WEIGHTS if weights is None else weights
    arrays = [
        np.asarray(regulatory_support, dtype=float),
        np.asarray(re_traceability, dtype=float),
        np.asarray(dependency_core, dtype=float),
        np.asarray(response_anchor, dtype=float),
    ]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("all DepStruct components must have identical shape")
    return (
        w["regulatory_support"] * arrays[0]
        + w["re_traceability"] * arrays[1]
        + w["dependency_core"] * arrays[2]
        + w["response_anchor"] * arrays[3]
    )
