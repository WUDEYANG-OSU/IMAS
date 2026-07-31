# IMAS
Integrative Multiomic Augmentation System for target-specific regulatory dependency reconstruction

IMAS is a matched tumour multiomic foundation and target-adaptation framework for recovering compact, interpretable and dataset-specific regulatory architectures from single-cell data.

The framework learns transferable relationships among genes, transcription factors (TFs) and regulatory elements (REs) from matched single-cell RNA and chromatin-accessibility profiles. It then adapts this information to a target dataset while preserving the observed expression matrix. The resulting multi-layer target dependencies (MLTDs) integrate molecular, regulatory, communication, temporal and perturbation-sensitive evidence for mechanism prioritisation.

This repository accompanies the manuscript:

Matched tumour multiomics resolves perturbation-sensitive regulationDeyang Wu, Takashi Yamashiro and Toshihiro Inubushi

Overview

Single-cell RNA-sequencing datasets provide broad cellular coverage but do not directly observe regulatory-element usage in the same cells. IMAS addresses this limitation through two linked stages:

Matched multiomic foundation learningA pan-cancer foundation is trained exclusively on paired single-cell RNA–ATAC measurements. The model learns direction-aware relationships among genes, TFs and REs in a common typed feature space.

Target-aware adaptationThe frozen foundation is adapted to a target dataset using low-rank updates, group-aware correction and task-specific residual refinement. The measured target expression matrix is not replaced or imputed.

The adapted outputs are organized as MLTDs, which support downstream analyses including:

RNA–TF dependency inference

RNA–RE and TF–RE support reconstruction

RNA–TF–RE bridge analysis

receiver-aware cell–cell communication

GNN–LTC communication pseudotime

target-centred virtual perturbation

perturbation-field and edge-flux analysis

dataset-specific mechanism prioritisation

Conceptual workflow

Matched single-cell RNA–ATAC datasets
                │
                ▼
Pan-cancer foundation learning
                │
                ├── RNA–ATAC alignment
                ├── direction-aware graph learning
                └── shared gene / TF / RE representation
                │
                ▼
RNA-only or matched target dataset
                │
                ▼
Target-aware adaptation
                │
                ├── low-rank parameter updates
                ├── group-aware cell-state correction
                ├── RNA-state preservation
                ├── TF-response supervision
                └── optional RE residual refinement
                │
                ▼
Multi-layer target dependencies
                │
                ├── RNA–TF–RE propagation
                ├── cell–cell communication
                ├── GNN–LTC communication pseudotime
                └── virtual perturbation and field/flux analysis

Main features

Matched multiomic foundation

The foundation model is trained on a pan-cancer corpus of matched RNA–ATAC datasets. In the accompanying study, the unified corpus contains:

48 matched single-cell multiomic datasets

24 tumour groups

approximately 2 million paired cells

33,373 genes

a global RE vocabulary of approximately 28.7 million intervals

Genes, TFs and REs are represented as separate typed entities. Direction-specific embeddings distinguish source and destination roles for each relation.

Target-aware adaptation

The adaptation framework:

freezes the pan-cancer foundation backbone;

applies rank-32 target-specific low-rank updates;

assigns target cells to group-aware state summaries;

preserves the measured RNA state through gene-retrieval supervision;

derives TF-response supervision from target-gene activity;

optionally refines selected REs when matched chromatin-derived gene activity is available;

uses an independent validation partition for checkpoint selection.

Multi-layer target dependencies

Each MLTD stores complementary support components for a directed relationship:

prior support
representation support
communication support
temporal support
perturbation field / flux support

MLTD scores are comparative support values rather than calibrated probabilities. They should not be interpreted as direct measurements of TF binding, signalling, chronological time or causal intervention effects.

Receiver-aware communication

Cell–cell communication analysis combines:

ligand–receptor expression;

interaction confidence;

sender-side molecular support;

receiver-side receptor–TF support;

independent TF–RE evidence;

permutation-based significance estimation.

GNN–LTC communication pseudotime

GNN–LTC orders cell states according to sender activity, receiver response and downstream TF context. It is a model-derived communication ordering rather than developmental or transcriptomic pseudotime.

Virtual perturbation

Target-centred virtual perturbation is performed on a fixed directed MLTD graph. Baseline and perturbed propagation are compared using node-state displacement and edge-flux redistribution.

Applications in the accompanying study

Colorectal cancer

SOD2-centred perturbation-sensitive dependency reconstruction

topology-matched localization testing

RNA–TF–RE bridge analysis

receiver-aware communication

GNN–LTC communication pseudotime

LAMB1-centred extracellular-matrix dependency recovery

Head and neck squamous cell carcinoma

SOX2-centred target adaptation

malignant-state-specific response stratification

state-specific communication programmes

virtual SOX2 perturbation

comparison with independent perturbation-prediction baselines

Renal cancer

joint modelling of untreated renal tumour samples

tumour–vascular dependency reconstruction

endothelial and pericyte state analysis

sample-restricted communication and pseudotime

independent Xenium spatial support

Repository structure

IMAS/
├── README.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── src/
│   └── imas/
│       ├── data/
│       │   ├── multiome.py
│       │   ├── scrna.py
│       │   ├── feature_space.py
│       │   └── manifests.py
│       ├── foundation/
│       │   ├── encoders.py
│       │   ├── graph_prior.py
│       │   ├── losses.py
│       │   └── train.py
│       ├── adaptation/
│       │   ├── low_rank_adapter.py
│       │   ├── group_correction.py
│       │   ├── peak_residual.py
│       │   └── train.py
│       ├── mltd/
│       │   ├── construction.py
│       │   ├── dependency_scores.py
│       │   ├── propagation.py
│       │   └── perturbation.py
│       ├── communication/
│       │   ├── ligand_receptor.py
│       │   ├── receiver_tf.py
│       │   └── permutation.py
│       ├── gnn_ltc/
│       │   ├── graph.py
│       │   ├── model.py
│       │   ├── pseudotime.py
│       │   └── controls.py
│       └── utils/
│           ├── io.py
│           ├── seeds.py
│           ├── statistics.py
│           └── plotting.py
│
├── configs/
│   ├── foundation.yaml
│   ├── target_adaptation.yaml
│   ├── crc_sod2.yaml
│   ├── crc_lamb1.yaml
│   ├── hnscc_sox2.yaml
│   ├── renal_cancer.yaml
│   └── xenium.yaml
│
├── scripts/
│   ├── 01_prepare_multiome.py
│   ├── 02_build_feature_space.py
│   ├── 03_train_foundation.py
│   ├── 04_adapt_target.py
│   ├── 05_construct_mltd.py
│   ├── 06_run_communication.py
│   ├── 07_run_gnn_ltc.py
│   └── 08_run_virtual_perturbation.py
│
├── reproduction/
│   ├── main_figures/
│   └── extended_data/
│
├── examples/
│   └── crc_demo/
│
├── manifests/
│   ├── foundation_datasets.tsv
│   ├── validation_datasets.tsv
│   ├── expected_outputs.tsv
│   └── checksums.tsv
│
├── tests/
└── docs/

Installation

Recommended environment

Linux

Python 3.10 or later

CUDA-enabled NVIDIA GPU for full training

PyTorch with CUDA support

Conda or Mamba

Clone the repository:

git clone https://github.com/WUDEYANG-OSU/IMAS.git
cd IMAS

Create the environment:

conda env create -f environment.yml
conda activate imas

Install IMAS in editable mode:

pip install -e .

A pip-only installation can be created with:

pip install -r requirements.txt
pip install -e .

Exact package versions used for the manuscript analyses are recorded in environment.yml and requirements.txt.

Input data

Matched RNA–ATAC foundation datasets

Each matched dataset requires:

an RNA count matrix;

an ATAC fragment file or peak-by-cell matrix;

matched cell barcodes;

sample identifiers;

gene annotations;

peak coordinates.

RNA and ATAC cells are paired within biological sample using normalized sample identifiers and harmonized cell barcodes.

Target scRNA-seq datasets

Each target dataset requires:

a raw gene-by-cell or cell-by-gene count matrix;

cell barcodes;

sample metadata;

optional cell annotations.

Raw counts are retained. Log-normalized values are generated for encoding and downstream summaries, but IMAS does not replace the original matrix.

Configuration files

All model and analysis parameters are stored in YAML files under configs/. Dataset-specific paths should be supplied through configuration files rather than hard-coded into source code.

Example:

dataset:
  name: CRC_RNA_TO009
  counts: /path/to/counts.h5ad
  metadata: /path/to/metadata.csv

model:
  checkpoint: /path/to/foundation_checkpoint.pt
  steps: 10000
  learning_rate: 7.0e-4
  latent_dim: 256
  adapter_rank: 32

output:
  directory: outputs/crc_sod2

Quick start

A lightweight example is provided under examples/crc_demo/.

python scripts/run_demo.py \
  --config examples/crc_demo/demo_config.yaml

The demonstration performs:

target-data loading;

typed gene and TF mapping;

target-cell embedding;

MLTD construction;

export of a small RNA–TF–RE support table.

The demonstration does not retrain the full pan-cancer foundation.

Full workflow

1. Prepare matched multiomic datasets

python scripts/01_prepare_multiome.py \
  --config configs/foundation.yaml

2. Build the unified feature space and regulatory prior

python scripts/02_build_feature_space.py \
  --config configs/foundation.yaml

3. Train the pan-cancer foundation

python scripts/03_train_foundation.py \
  --config configs/foundation.yaml

4. Adapt the foundation to a target dataset

python scripts/04_adapt_target.py \
  --config configs/target_adaptation.yaml

5. Construct MLTDs

python scripts/05_construct_mltd.py \
  --config configs/crc_sod2.yaml

6. Run receiver-aware communication analysis

python scripts/06_run_communication.py \
  --config configs/crc_sod2.yaml

7. Run GNN–LTC communication pseudotime

python scripts/07_run_gnn_ltc.py \
  --config configs/crc_sod2.yaml

8. Run target-centred virtual perturbation

python scripts/08_run_virtual_perturbation.py \
  --config configs/crc_sod2.yaml

Reproducing the manuscript figures

Main-figure scripts are stored in:

reproduction/main_figures/

Extended Data scripts are stored in:

reproduction/extended_data/

The intended commands are:

bash reproduction/reproduce_main_figures.sh
bash reproduction/reproduce_extended_data.sh

Each figure workflow uses frozen processed inputs, configuration files and expected-output checksums.

The manuscript figures correspond to:

Figure

Analysis

Fig. 1

Pan-cancer foundation and target-aware adaptation

Fig. 2

SOD2 perturbation-sensitive dependency architecture

Fig. 3

RNA–TF–RE propagation, communication and GNN–LTC

Fig. 4

LAMB1-centred perturbation-space reconstruction

Fig. 5

SOX2-centred HNSCC programmes

Fig. 6

Renal tumour–vascular coupling

Extended Data

Foundation controls, ablations, benchmarks, application-specific validation and Xenium analysis

Computational requirements

Full foundation training

The manuscript foundation was trained with:

latent dimension: 256

hidden dimension: 512

dropout: 0.1

maximum RNA tokens per cell: 2,048

maximum ATAC tokens per cell: 1,024

RE hash-table size: 2,000,000

contrastive temperature: 0.07

four graph negatives per positive edge

30,000 optimization steps

AdamW optimizer

learning rate: 2 × 10⁻⁴

weight decay: 10⁻²

gradient clipping: 1.0

Full training requires a CUDA-enabled GPU and substantial local or cluster storage for processed multiomic matrices and fragment-derived files.

Target adaptation

The manuscript target-adaptation models used:

16 target-cell groups

rank-32 low-rank updates

10,000 optimization steps

learning rate: 7 × 10⁻⁴

cell batches of up to 2,000

gradient clipping: 5.0

validation TF AUC for checkpoint selection

A 32-GB NVIDIA GPU is sufficient for development-scale target adaptation and figure-level analyses, although full foundation training is better suited to a multi-GPU or high-memory computing environment.

Outputs

Depending on the selected workflow, IMAS writes:

trained foundation checkpoints;

target-adapted checkpoints;

typed feature manifests;

RNA–TF support tables;

RNA–RE and TF–RE support tables;

MLTD node and edge tables;

ligand–receptor communication tables;

receiver-TF support matrices;

cluster and cell communication pseudotime;

virtual perturbation node-state changes;

edge-flux changes;

permutation and bootstrap results;

figure-ready tables.

Large intermediate objects are excluded from Git and should be stored in a separate data directory.

Reproducibility

The manuscript analyses use:

deterministic typed feature identifiers;

explicit sample–barcode pairing keys;

fixed candidate universes;

deterministic data splits;

saved train, validation and test manifests;

fixed random seeds;

frozen YAML configurations;

stored checkpoints;

permutation outputs;

expected-output checksums.

Random seed 42 is used unless an ensemble or permutation analysis specifies additional seeds.

The repository includes tests for:

feature mapping;

model forward passes;

MLTD construction;

communication-score calculation;

deterministic split recovery;

output-schema validation.

Data availability

Public single-cell datasets used in the manuscript are available from their original repositories. Foundation and validation dataset manifests are provided in:

manifests/foundation_datasets.tsv
manifests/validation_datasets.tsv

Public accession codes include datasets from the Gene Expression Omnibus and the 10x Genomics datasets portal. The renal Xenium dataset is distributed by 10x Genomics under the title:

Xenium In Situ Gene and Protein Expression data for FFPE Human Renal Cell Carcinoma

The accession for the primary renal tumour dataset will be added before publication.

Raw single-cell matrices, ATAC fragment files, Xenium objects and large checkpoints are not stored directly in this Git repository. Processed data and model files that can be redistributed will be deposited in an archival repository and linked from the release page.

Pretrained models

Pretrained foundation and target-adapted checkpoints will be distributed separately because of file size.

Each released checkpoint will include:

model version;

training configuration;

source-data manifest;

feature-vocabulary checksum;

checkpoint checksum;

compatible code release.

Interpretation and limitations

IMAS is a hypothesis-prioritisation framework.

Model outputs should not be interpreted as:

direct measurements of protein–DNA binding;

experimentally observed ligand–receptor signalling;

chronological developmental time;

causal intervention effects;

complete reconstruction of an unmeasured molecular modality.

Virtual perturbation and spatial correspondence require independent experimental validation.

Citation

Please cite the associated manuscript when using IMAS:

Wu, D., Yamashiro, T. & Inubushi, T. Matched tumour multiomics resolves perturbation-sensitive regulation. Manuscript in preparation (2026).

A machine-readable citation is provided in CITATION.cff. The citation will be updated with the final journal reference and DOI after publication.

Licence

This repository is currently distributed without an open-source licence. All rights are reserved.

A formal software licence will be added after institutional and co-author review.

Contact

Deyang WuDepartment of Orthodontics and Dentofacial OrthopedicsGraduate School of Dentistry, Osaka UniversityEmail: scu_wudeyang@163.com

Toshihiro InubushiDepartment of Orthodontics and Dentofacial OrthopedicsGraduate School of Dentistry, Osaka UniversityEmail: inubushitoshihiro.dent@osaka-u.ac.jp
