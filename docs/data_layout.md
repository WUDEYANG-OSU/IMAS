# Expected data layout

Large data files are intentionally excluded from Git. The main scripts expect a
workspace similar to:

```text
LLM-WorldWord/<dataset>/
├── assets/
│   ├── genes.tsv
│   ├── peaks.bed
│   └── node_index.json
└── processed/
    ├── RNA/
    │   ├── manifest_rna.tsv
    │   ├── <sample>.rna_meta.npz
    │   └── <sample>.X_rna_counts_dense.npy
    ├── ATAC/
    │   ├── manifest_atac.tsv
    │   ├── <sample>.atac_meta.npz
    │   └── <sample>.X_atac_peaks_csr.npz
    └── ATAC_geneActivity_ALLGENES/
```

Global resources are expected below `networks/qc_global_projection/`:

```text
maps/global_gene_vocab.txt
maps/global_tf_vocab.txt
maps/global_peak_vocab.txt
priors_cache/
projected_inputs/
```

Checkpoint and output roots are configurable. See `docs/configuration.md`.
Input schemas differ between legacy reproduction scripts; the exact required
files are documented in each script header.
