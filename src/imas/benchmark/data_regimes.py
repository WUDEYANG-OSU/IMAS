"""Definitions for paired multiomic, mixed, and scRNA-only comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataRegime(str, Enum):
    MULTIOMIC_ONLY = "multiomic_only"
    MIXED = "mixed"
    SCRNA_ONLY = "scrna_only"


@dataclass(frozen=True)
class DataRegimeSpec:
    use_paired_only: bool
    enable_re_branch: bool
    enable_selected_peak_residual: bool
    re_loss_on_matched_cells_only: bool


REGIME_SPECS = {
    DataRegime.MULTIOMIC_ONLY: DataRegimeSpec(True, True, True, True),
    DataRegime.MIXED: DataRegimeSpec(False, True, True, True),
    DataRegime.SCRNA_ONLY: DataRegimeSpec(False, False, False, False),
}


def get_regime_spec(regime: str | DataRegime) -> DataRegimeSpec:
    return REGIME_SPECS[DataRegime(regime)]
