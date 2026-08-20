# Benchmark runners

| Script | Purpose |
|---|---|
| `01_run_virtual_ko_style_baselines.py` | Generate scGPT-style, CPA-style, and GEARS-style virtual-KO controls. |
| `02_evaluate_virtual_ko_trend_consistency.py` | Compare control-to-predicted shifts with observed DEG trends. |
| `03_run_stageb_component_ablation.py` | Retrain the Stage-B component-ablation modes. |
| `04_evaluate_stageb_component_ablation.py` | Evaluate existing ablation checkpoints without retraining. |
| `05_run_data_regime_comparison.py` | Run multiomic-only, mixed, and scRNA-only Stage-B regimes. |
| `06_run_global_stagewise_ablation.py` | Evaluate null controls and progressive stages on unified endpoints. |

The three named virtual-KO controls are **style baselines**, not official package
reproductions. The exact source scripts retain their original path settings and
must be edited or wrapped with site-specific configuration before execution.
