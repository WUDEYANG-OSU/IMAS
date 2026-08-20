# Benchmark coverage inventory

| Requested comparison | Repository coverage | Main locations |
|---|---|---|
| IMAS/MLTD vs scGPT-style, CPA-style, GEARS-style | Complete for the task-aligned style controls used in the project; not official package reproductions | `scripts/benchmark/01_*`, `02_*`, `src/imas/benchmark/baselines.py` |
| MLTD-on vs MLTD-off | Molecular bridge, directional CCI, paired GNN-LTC, and fixed-reference temporal comparison | `reproduction/benchmark/mltd_on_off/` |
| Multiomic-only vs mixed vs scRNA-only | Shared Stage-B runner plus CRC, Glioma, and Osteosarcoma post-hoc diagnostics | `scripts/benchmark/05_*`, `reproduction/benchmark/data_regimes/` |
| Selected-peak residual and cluster correction | Retrained component ablations and independent checkpoint evaluation | `scripts/benchmark/03_*`, `04_*` |
| Other Stage-B components | Low-rank adaptor, group scale/bias, cell-group adaptor, simple fine-tuning | `src/imas/benchmark/ablation.py` and component scripts |
| AUROC/AUPRC | Shared implementations plus RNA/TF/RE and edge-endpoint workflows | `src/imas/evaluation/classification.py`, `reproduction/benchmark/task_evaluation/` |
| Spearman | Shared tie-aware implementation and model-specific QC | `src/imas/evaluation/correlation.py`, task/explainability scripts |
| Retrieval and top-k | MRR, Hit@K, Precision@K, Recall@K, NDCG@K, Lift@K, top-k Jaccard | `src/imas/evaluation/ranking.py`, stage-wise and task-evaluation scripts |
| Calibration and robustness | Brier, ECE, bootstrap confidence/stability | `src/imas/evaluation/calibration.py`, `stability.py` |

## Boundary of completion

The repository now contains the complete benchmark code actually used in the
project. It does not contain official package-native runs for CPA, scGPT, or
GEARS. Any manuscript or README statement must retain the `-style` label unless
official implementations are added later.
