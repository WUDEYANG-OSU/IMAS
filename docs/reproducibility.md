# Reproducibility scope

## Included

- Reusable `src/imas` algorithms.
- Complete legacy/main workflow scripts extracted from the supplied code.
- Benchmark and ablation workflows.
- Portable path configuration.
- Synthetic tests for package imports, data/prior utilities, perturbation and
  benchmark metrics.
- Syntax compilation and repository integrity reports.

## Not bundled

- Raw single-cell matrices and human subject data.
- Pan-cancer prior caches and projected feature matrices.
- Foundation and target-adaptation checkpoints.
- Full generated figures and intermediate run outputs.
- Official CPA, scGPT or GEARS package reproductions; the current controls are
  accurately labelled `*-style`.

## Validation boundary

This release can validate package installation and lightweight synthetic
operations. It does not claim that every large GPU workflow was rerun in this
container. Full reproduction requires the external artifacts and compute
resources named by the corresponding script/configuration.
