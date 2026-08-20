# IMAS

**IMAS (Integrative Multi-omic Augmentation System)** is a research software
framework for tumour single-cell multiomics. It combines pan-cancer matched
RNA–ATAC foundation learning, leave-one-dataset-out training, target-specific
adaptation, multi-layer target dependencies (MLTDs), receiver-aware cell–cell
communication, GNN–LTC communication pseudotime and virtual perturbation analysis.

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
  target-adaptation components, retrieval, AUROC/AUPRC, Spearman and top-k metrics.

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

Benchmark comparisons include **scGPT**, **CPA**, **GEARS**, **MIDAS** and
**multiDGD** where applicable. Dataset-specific benchmark configurations,
evaluation procedures and source provenance are documented in the accompanying
scripts and `benchmark_source_provenance.tsv`.

## Reproducibility scope

The included tests validate reusable modules, configuration resolution,
input/output schemas and core workflows using lightweight synthetic inputs.
Full foundation training requires the complete pan-cancer multiomic corpus and
dedicated GPU infrastructure and is therefore not part of the lightweight test
suite. See [reproducibility notes](docs/reproducibility.md).
## Data availability

The single-cell RNA-sequencing data generated from the eight patient-derived
renal tumours have been deposited in the NCBI Gene Expression Omnibus under
accession **GSE344302**.

Public datasets used for foundation training, validation and application
analyses are documented in the repository manifests,
`benchmark_source_provenance.tsv` and the accompanying manuscript. Large raw
datasets and model checkpoints are not stored directly in this Git repository.
## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the accompanying
manuscript:

> Wu, D., Yamashiro, T. & Inubushi, T. *Matched tumour multiomics resolves perturbation-sensitive regulation*.

The citation information will be updated with the final journal reference and
DOI upon publication.
## License

A license is intentionally not included in this release. Add the selected
license before public distribution.
