"""Stage-B component-ablation definitions."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StageBComponents:
    lowrank_adapter: bool = True
    cluster_correction: bool = True
    selected_peak_residual: bool = True
    group_scale_bias: bool = True
    cellgroup_adapter: bool = True
    simple_finetune: bool = False


ABLATION_MODES = {
    "full_stageB": StageBComponents(),
    "simple_finetune": StageBComponents(
        lowrank_adapter=False,
        cluster_correction=False,
        selected_peak_residual=False,
        group_scale_bias=False,
        cellgroup_adapter=False,
        simple_finetune=True,
    ),
    "no_lowrank_adapter": replace(StageBComponents(), lowrank_adapter=False),
    "no_cluster_correction": replace(StageBComponents(), cluster_correction=False),
    "no_selected_peak_residual": replace(StageBComponents(), selected_peak_residual=False),
    "no_group_scale_bias": replace(StageBComponents(), group_scale_bias=False),
    "no_cellgroup_adapter": replace(StageBComponents(), cellgroup_adapter=False),
}


def get_ablation_mode(name: str) -> StageBComponents:
    try:
        return ABLATION_MODES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Stage-B ablation mode: {name}") from exc
