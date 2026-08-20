# IMAS

**IMAS (Integrative Multi-omic Augmentation System)** is a research software
framework for tumour single-cell multiomics. It combines a fixed pan-cancer
RNA–ATAC prior, leave-one-dataset-out foundation training, target adaptation,
multi-layer target dependencies (MLTDs), cell–cell communication, GNN–LTC
communication pseudotime and virtual perturbation analysis.

> **Repository status:** research code, version 0.3.0. The scientific workflows
> and benchmark/ablation scripts are included. Large datasets, pan-cancer prior
> caches and trained checkpoints are not bundled. A software license has not yet
> been selected; until a `LICENSE` file is added, reuse rights are not granted by
> this repository.

## Main capabilities

- Global RNA/ATAC vocabulary construction and projected sparse inputs.
- Pan-cancer RE→gene, TF→RE, TF→gene, RNA→gene, RNA→RE, RNA→TF and
  gene–gene prior construction.
- Leave-one-dataset-out foundation training with a fixed pan-cancer prior graph.
- Target adaptation with low-rank adaptors, selected-peak residuals and
  cluster-aware correction.
- MLTD assembly across prior, representation, communication and temporal layers.
- RNA→TF, RNA→RE and RNA→RNA evaluation; CCI and GNN–LTC analysis.
- Virtual KO/OE, perturbation propagation, response subgroups and NDS mapping.
- Benchmarking and ablation for MLTD-on/off, multiomic-only/mixed/scRNA-only,
  Stage-B components, retrieval, AUROC/AUPRC, Spearman and top-k metrics.

## Repository layout

```text
src/imas/          Reusable Python package
scripts/           Main executable workflows
reproduction/      Dataset- and figure-specific analyses
configs/           Portable YAML configuration
scripts/benchmark/ Benchmark and ablation workflows
tests/             Synthetic unit and smoke tests
docs/              Installation, configuration and reproduction notes
```

## Installation

```bash
conda env create -f environment.yml
conda activate imas
pip install -e .
```

For an existing Python environment:

```bash
pip install -e ".[full]"
```

PyTorch/CUDA builds are machine-specific. Install a PyTorch build compatible
with the local driver before running GPU workflows.

## Configure paths

No script should require a fixed drive letter or `/mnt/e` mount. Copy the
example configuration and set the workspace root:

```bash
cp configs/project.default.yaml configs/project.yaml
export IMAS_PROJECT_ROOT=/path/to/imas-workspace
```

The expected workspace convention is:

```text
$IMAS_PROJECT_ROOT/
├── LLM-WorldWord/
├── networks/qc_global_projection/
├── checkpoints/
├── resources/
└── runs/
```

More granular overrides are available, including `IMAS_WORLD_ROOT`,
`IMAS_RUNS_ROOT`, `IMAS_GLOBAL_ROOT`, `IMAS_PRIORS_ROOT` and
`IMAS_CHECKPOINT_ROOT`. See [configuration documentation](docs/configuration.md).

Validate the installation and resolved paths:

```bash
python scripts/validate_environment.py
pytest -q
```

## Typical workflow order

```text
1. scripts/data/
2. scripts/priors/
3. scripts/foundation/
4. scripts/adaptation/
5. scripts/mltd/
6. scripts/cci/
7. scripts/gnn_ltc/
8. scripts/perturbation/
9. scripts/nds/
10. scripts/benchmark/ and reproduction/
```

The complete data and checkpoint files are not included. Each workflow checks
for its required inputs and writes outputs below the configured runs directory.

## Benchmark naming boundary

Files labelled **scGPT-style**, **CPA-style**, **GEARS-style**, **MIDAS-like**
or **multiDGD-style** are task-aligned architectural controls. They are not
runs of the official packages unless an official implementation and checkpoint
are explicitly supplied.

## Reproducibility scope

The included tests validate reusable modules on synthetic inputs and check
Python syntax/imports. Full multi-terabyte corpus training and checkpoint-level
GPU reproduction were not executed inside this packaged environment. See
[reproducibility notes](docs/reproducibility.md).

## Citation

Citation metadata are provided in `CITATION.cff`. Author order, repository URL,
manuscript title and DOI remain editable placeholders until the project team
confirms them.

## License

A license is intentionally not included in this release. Add the selected
license before public distribution.
