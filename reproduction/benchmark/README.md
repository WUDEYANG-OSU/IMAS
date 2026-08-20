# Benchmark reproduction scripts

This directory contains the post-processing and figure workflows associated
with the benchmark suite.

```text
data_regimes/       Multiomic-only, mixed, and scRNA-only comparisons
mltd_on_off/         RNA-TF-RE bridge, CCI, and GNN-LTC on/off comparisons
stageb_components/   Component-ablation summary figures
stagewise_ablation/  Unified endpoint main and supplementary metrics
task_evaluation/     RNA, TF, and RE task recovery and retrieval analyses
explainability/      Counterfactual deletion and biological-comparator analyses
```

These scripts preserve the final analysis logic and file expectations from the
source workflow documents. Most are standalone scripts with explicit project
paths rather than generic command-line wrappers.
