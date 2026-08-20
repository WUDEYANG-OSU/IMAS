# Checkpoints and large artifacts

The Git repository excludes model weights, projected matrices, prior caches and
full analysis outputs. Store them below the configured roots or publish them in
an external archive.

Recommended release manifest columns:

```text
artifact_id, dataset, role, relative_path, size_bytes, sha256, download_url
```

Before publishing a checkpoint bundle, record the exact commit, configuration,
training seed, input-manifest checksum and software environment. Dataset-specific
checkpoint paths can be supplied through `configs/datasets/*.yaml` or
`IMAS_CKPT_<DATASET>` environment variables.
