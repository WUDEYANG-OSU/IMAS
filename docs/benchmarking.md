# IMAS benchmark and ablation workflows

This repository groups the benchmark code into four complementary levels.

## 1. Task-aligned external controls

The repository contains `scGPT-style`, `CPA-style`, and `GEARS-style` virtual
perturbation controls. These names describe the design principle used by the
control, not an execution of the official packages or official checkpoints.

- `scGPT-style`: target masking followed by context reconstruction.
- `CPA-style`: a population-derived shared perturbation-effect vector.
- `GEARS-style`: target-effect propagation through a fixed gene graph.

The runnable source scripts are:

```text
scripts/benchmark/01_run_virtual_ko_style_baselines.py
scripts/benchmark/02_evaluate_virtual_ko_trend_consistency.py
```

The evaluator compares predicted mean shifts with observed DEG directions and
reports three-class direction accuracy, non-flat accuracy, up/down recall, and
balanced up/down accuracy.

Additional architectural controls are retained under
`scripts/benchmark/baselines/`: a minimal multimodal VAE, a MIDAS-like shared
latent model, linear regression, and an MLP. Files labelled `*-like` are not
claimed as exact official implementations.

## 2. Data-regime comparison

The same Stage-B workflow is run in three regimes:

- `multiomic_only`: paired cells, all RNA/TF/RE branches enabled.
- `mixed`: all RNA cells; RE supervision is restricted to matched cells.
- `scrna_only`: no target-domain RE branch or selected-peak residual.

The main runner is `scripts/benchmark/05_run_data_regime_comparison.py`.
Post-hoc DepStruct, manifold, marker-separability, and unpaired-dilution analyses
are under `reproduction/benchmark/data_regimes/`.

## 3. Stage-B component ablation

The component runner compares:

```text
full_stageB
simple_finetune
no_lowrank_adapter
no_cluster_correction
no_selected_peak_residual
no_group_scale_bias
no_cellgroup_adapter
```

Training and independent checkpoint evaluation are separated:

```text
scripts/benchmark/03_run_stageb_component_ablation.py
scripts/benchmark/04_evaluate_stageb_component_ablation.py
```

## 4. MLTD-on/off and full stage-wise ablation

MLTD-on/off is evaluated at four levels:

1. seed-conditioned RNA-TF-RE bridge coupling;
2. directional CCI in the same LR candidate universe;
3. paired GNN-LTC temporal transmission;
4. fixed-reference trajectory analysis.

The relevant reproduction scripts are under
`reproduction/benchmark/mltd_on_off/`.

The global stage-wise analysis scores the same directed cluster-edge universe
with null controls and progressive model stages, then evaluates unified
endpoints using AUROC, AUPRC, MRR, Precision/Recall/Hit@K, NDCG, Lift,
calibration, directionality, topology, mechanism support, and bootstrap
stability.

## Shared library APIs

Reusable metrics are in `src/imas/evaluation/`. Benchmark schemas, compact
style baselines, data-regime definitions, component-ablation definitions, and
the DepStruct formula are in `src/imas/benchmark/`.

The large files under `scripts/benchmark/` and `reproduction/benchmark/` are
faithful extractions of the final workflow documents. They retain dataset paths
and should be configured before execution. They are not yet thin wrappers around
all functions in `src/imas`.
