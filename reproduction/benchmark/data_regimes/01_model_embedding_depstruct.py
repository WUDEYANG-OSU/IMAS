# -*- coding: utf-8 -*-
"""
IMAS Stage-B benchmark-style UMAP visualization of dependent structure
=======================================================================

Purpose
-------
This script does NOT retrain any model and does NOT use TF-proxy/AUCell labels.
It uses existing Stage-B frontier_best.pt checkpoints and produces benchmark-style
UMAP panels: each dataset x data-regime is re-embedded separately from the
regime-specific adapted model cell embeddings.

Main logic
----------
For each dataset x regime, compute regime-specific adapted cell embeddings and
run PCA/neighbors/UMAP/Leiden with the same UMAP/Leiden parameters as the
CCI/new-Leiden script:
    N_TOP_HVG = 2000
    N_PCS = 20
    N_NEIGHBORS = 8
    UMAP_MIN_DIST = 0.03
    UMAP_SPREAD = 2.0
    LEIDEN_RESOLUTION = 0.5

For each data regime checkpoint:
    multiomic_only / mixed / scrna_only
load the saved frontier_best.pt, run posthoc forward only, re-run UMAP/Leiden on
the adapted model embeddings, and compute one panel title score:
    DepStruct = 0.30 regulatory support
              + 0.30 RE-supported traceability
              + 0.25 dependency-core membership
              + 0.15 perturbation-anchor proxy

CCI/LTC is NOT used as the backbone here. CCI can be overlaid later as an auxiliary
layer if needed, but this script focuses on RNA-TF-RE/local dependency structure.

Expected checkpoint layout
--------------------------
RUN_DIR/<dataset>/<regime>/frontier_best.pt
RUN_DIR/<dataset>/<regime>/clusters.csv

Outputs
-------
RUN_DIR/benchmark_umap_dependency_structure_posthoc/
    <dataset>_<regime>_benchmark_model_umap.h5ad
    <dataset>_<regime>_cell_structure_scores.csv
    <dataset>_<regime>_cluster_structure_scores.csv
    dependency_structure_benchmark_umap.svg/pdf/png
    dependency_structure_benchmark_scores.csv

Notes
-----
- scRNA-only is deliberately assigned no target-domain RE/selected-peak support.
- multiomic_only/mixed use selected_peak_raw_ids if saved in frontier_best.pt;
  otherwise they fall back to gene-peak prior presence.
- Scores are structure-oriented proxies for visualization, not standalone model
  accuracy endpoints.
"""

from __future__ import annotations
from imas.config import get_dataset_checkpoint, get_project_root

import os
import re
import sys
import math
import json
import random
import warnings
import subprocess
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import anndata as ad
    import scanpy as sc
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scanpy", "anndata"])
    import anndata as ad
    import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================
# 0) USER CONFIG
# ============================================================

# Change BASE if your project is mounted elsewhere.
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"
GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"
GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"
GLOBAL_PEAK_VOCAB = MAPS_ROOT / "global_peak_vocab.txt"

# Data-regime comparison output from your Stage-B script.
RUN_DIR = BASE / "runs" / "frontier_transfer_hash_stageB_data_regime_comparison"

# ------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------
# Original script only contained CRC-Tissue and Glioma-Tissue.
# Add Osteosarcoma here so the posthoc model-embedding, DepStruct scoring,
# marker statistics and benchmark-style plotting are all run with exactly
# the same downstream logic.
#
# If your real folder/checkpoint name is slightly different, the resolver below
# will try common aliases and then scan available folders for names containing
# "osteo". The resolved name is used everywhere downstream.
DATASETS_TO_RUN_RAW = ["CRC-Tissue", "Glioma-Tissue", "Osteosarcoma-Tissue"]

DATASET_ALIASES: dict[str, list[str]] = {
    "Osteosarcoma-Tissue": [
        "Osteosarcoma-Tissue",
        "Osteosarcoma",
        "Osteosarcoma-Cellline",
        "Osteosarcoma-CellLine",
        "Osteosarcoma-Bone",
        "OS-Tissue",
    ],
}


def _dedupe_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        x = str(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_dataset_name(ds_name: str) -> str:
    """Resolve dataset alias against the folders/checkpoints already on disk."""
    ds_name = str(ds_name)
    candidates = _dedupe_keep_order([ds_name] + DATASET_ALIASES.get(ds_name, []))
    search_roots = [
        WORLD_ROOT,
        RUN_DIR,
        PROJECTED_ROOT,
    ]

    # Prefer exact/common alias folders.
    for cand in candidates:
        for root in search_roots:
            try:
                if (root / cand).exists():
                    if cand != ds_name:
                        print(f"[DATASET_ALIAS] {ds_name} -> {cand}", flush=True)
                    return cand
            except Exception:
                pass

    # Fallback: scan for osteosarcoma-like folder names only for the osteosarcoma entry.
    if "osteo" in ds_name.lower():
        for root in search_roots:
            try:
                if not root.exists():
                    continue
                hits = sorted([
                    x.name for x in root.iterdir()
                    if x.is_dir() and ("osteo" in x.name.lower())
                ])
                if hits:
                    print(f"[DATASET_ALIAS] {ds_name} -> {hits[0]}  (auto-detected)", flush=True)
                    return hits[0]
            except Exception:
                pass

    # Keep the requested name; downstream asserts will report the missing path clearly.
    return ds_name


DATASETS_TO_RUN = _dedupe_keep_order([resolve_dataset_name(x) for x in DATASETS_TO_RUN_RAW])

# Calculation control:
# - DATASETS_TO_RUN controls the final combined figure/table order.
# - DATASETS_TO_CALCULATE_RAW controls which datasets are newly computed in this run.
# Here CRC/Glioma are loaded from existing CSV outputs and Osteosarcoma is the only
# dataset补算 by default.
DATASETS_TO_CALCULATE_RAW = ["Osteosarcoma-Tissue"]
DATASETS_TO_CALCULATE = set(resolve_dataset_name(x) for x in DATASETS_TO_CALCULATE_RAW)
REUSE_EXISTING_DATASET_RESULTS = True

REGIME_ORDER = ["multiomic_only", "mixed", "scrna_only"]
REGIME_LABEL = {
    "multiomic_only": "Multiomic-only",
    "mixed": "Mixed",
    "scrna_only": "scRNA-only",
}

# Use all samples by default. To restrict any dataset to selected samples, edit here.
# Example:
# SAMPLES_BY_DATASET_RAW = {
#     "CRC-Tissue": ["CRC_RNA_TO009"],
#     "Glioma-Tissue": None,
#     "Osteosarcoma-Tissue": None,
# }
SAMPLES_BY_DATASET_RAW: dict[str, Optional[list[str]]] = {
    "CRC-Tissue": None,
    "Glioma-Tissue": None,
    "Osteosarcoma-Tissue": None,
}

# Convert sample restrictions from requested names to resolved names.
SAMPLES_BY_DATASET: dict[str, Optional[list[str]]] = {}
for _raw_ds in DATASETS_TO_RUN_RAW:
    _resolved_ds = resolve_dataset_name(_raw_ds)
    SAMPLES_BY_DATASET[_resolved_ds] = SAMPLES_BY_DATASET_RAW.get(_raw_ds, None)
for _resolved_ds in DATASETS_TO_RUN:
    SAMPLES_BY_DATASET.setdefault(_resolved_ds, None)

OUT_ROOT = RUN_DIR / "benchmark_umap_dependency_structure_posthoc"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- UMAP / Leiden: copied from your CCI/new-Leiden script ----------
N_TOP_HVG = 2000
N_PCS = 20
N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.03
UMAP_SPREAD = 2.0
LEIDEN_RESOLUTION = 0.5

# ---------- runtime ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Need CUDA for fast posthoc scoring."
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- encoding ----------
RNA_TOPK = 2048
EMB_BATCH = 256
GENE_VEC_BATCH = 2048
TF_VEC_BATCH = 512

# ---------- structure scoring scale ----------
MAX_TF_FOR_STRUCTURE = 128
MAX_TARGET_GENES_FOR_STRUCTURE = 1800
TOP_TF_PER_CELL = 8
TOP_GENE_PER_CELL = 50
TOP_TARGETS_PER_TF = 80

# Plotting uses smoothed cluster-level scores by default.
PLOT_CLUSTER_SMOOTHED_SCORE = True

# For large datasets, you can set this to e.g. 30000 to subsample before UMAP.
# None means use all cells in selected samples.
MAX_CELLS_FOR_UMAP_PER_DATASET = None

# If UMAP table already exists, reuse it by default.
REUSE_EXISTING_UMAP = True

# ---------- adaptor toggles, consistent with Stage-B ----------
USE_GROUP_SCALE_BIAS = True
GROUP_SCALE_ALPHA = 0.12
USE_CELLGROUP_ADAPTOR = True
USE_CANCER_ADAPTOR = True


def log(*args):
    print(*args, flush=True)


# ============================================================
# 1) BASIC HELPERS
# ============================================================

def read_vocab_txt(p: Path) -> list[str]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def normalize_gene_name(x: str) -> str:
    return str(x).strip() if x is not None else ""


def norm_barcode(x: str) -> str:
    x = str(x)
    return x.split("::", 1)[-1] if "::" in x else x


def make_cell_key(sid: str, barcode: str) -> str:
    return f"{sid}::{norm_barcode(barcode)}"


def dataset_to_cancer_type(ds_name: str) -> str:
    s = str(ds_name)
    return s.split("-", 1)[0] if "-" in s else s


def get_default_cancer_name(dataset_name: str) -> str | None:
    return dataset_to_cancer_type(dataset_name)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if (not np.isfinite(s)) or s < 1e-6:
        return np.zeros_like(x, np.float32)
    return ((x - m) / (s + 1e-6)).astype(np.float32)


def robust_minmax(x: np.ndarray, qlo: float = 1.0, qhi: float = 99.0) -> np.ndarray:
    x = np.asarray(x, np.float32)
    out = np.zeros_like(x, np.float32)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out
    lo, hi = np.nanpercentile(x[m], [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        out[m] = 0.5
        return out
    out[m] = np.clip((x[m] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return out.astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = np.clip(x, -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("[SAVE]", path)
    return path


def find_prior_file(prior_dir: Path, include_keys: list[str]) -> Optional[Path]:
    cands = sorted(prior_dir.glob("*.npz"))
    for p in cands:
        n = p.name.lower()
        if all(k in n for k in include_keys):
            return p
    return None


def save_figure(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png"]:
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(
            p,
            dpi=450 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        log("[FIG]", p)
    plt.close(fig)


def savefig_multi(fig, out_stem: Path):
    """Backward-compatible alias used by the legend plotting function."""
    return save_figure(fig, out_stem)


# ============================================================
# 2) MODEL CLASSES
# ============================================================

class PeakHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, d: int):
        super().__init__()
        self.hash_size = int(hash_size)
        self.emb = nn.Embedding(self.hash_size, d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, peak_idx: torch.Tensor):
        hashed = torch.remainder(peak_idx, self.hash_size)
        return self.emb(hashed)


class PlanC_DirAware_Cancer_HashPeak(nn.Module):
    def __init__(self, n_gene, n_tf, n_cancer, peak_hash_size, d=256, hid=512, drop=0.1):
        super().__init__()
        self.peak_hash_size = int(peak_hash_size)
        self.gene_in = nn.Embedding(n_gene, d)
        self.gene_out = nn.Embedding(n_gene, d)
        self.tf_in = nn.Embedding(n_tf, d)
        self.tf_out = nn.Embedding(n_tf, d)
        self.peak_in = PeakHashEmbedding(self.peak_hash_size, d)
        self.peak_out = PeakHashEmbedding(self.peak_hash_size, d)
        self.gene_in_c = nn.Embedding(n_cancer, d)
        self.gene_out_c = nn.Embedding(n_cancer, d)
        self.peak_in_c = nn.Embedding(n_cancer, d)
        self.peak_out_c = nn.Embedding(n_cancer, d)
        self.tf_in_c = nn.Embedding(n_cancer, d)
        self.tf_out_c = nn.Embedding(n_cancer, d)
        self.cell_c = nn.Embedding(n_cancer, d)
        self.rna_proj = nn.Sequential(
            nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d)
        )
        self.atac_proj = nn.Sequential(
            nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d)
        )

    @staticmethod
    def _seg_ids_from_offsets(offsets: torch.Tensor, total_tokens: int):
        B = offsets.shape[0]
        end = torch.empty((B,), device=offsets.device, dtype=torch.long)
        end[:-1] = offsets[1:]
        end[-1] = total_tokens
        lengths = (end - offsets).clamp(min=0)
        seg_ids = torch.repeat_interleave(torch.arange(B, device=offsets.device), lengths)
        return seg_ids

    @torch.no_grad()
    def encode_rna(self, idx, val, offsets, cancer_ids):
        seg_ids = self._seg_ids_from_offsets(offsets, idx.shape[0])
        c_tok = cancer_ids.index_select(0, seg_ids)
        x = (self.gene_in(idx) + self.gene_in_c(c_tok)) * val.unsqueeze(-1)
        out = torch.zeros((offsets.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        out.index_add_(0, seg_ids, x)
        out = self.rna_proj(out) + self.cell_c(cancer_ids)
        return F.normalize(out, dim=-1)


class HyperAdapterHashV2(nn.Module):
    def __init__(
        self,
        d: int,
        k: int,
        n_tf: int,
        n_gene: int,
        peak_hash_size: int,
        n_cancer: int,
        n_group: int,
        e_dim: int = 64,
        hid: int = 256,
    ):
        super().__init__()
        self.d = d
        self.k = k
        self.peak_hash_size = int(peak_hash_size)
        self.e = nn.Parameter(torch.zeros((e_dim,), dtype=torch.float32))
        nn.init.normal_(self.e, mean=0.0, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(e_dim, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, 4 * k * d),
        )
        self.C_tf = nn.Embedding(n_tf, k)
        self.C_gene = nn.Embedding(n_gene, k)
        self.C_peak_hash = nn.Embedding(self.peak_hash_size, k)
        self.C_cancer = nn.Embedding(n_cancer, k)
        self.S_group = nn.Embedding(n_group, d)
        self.B_group = nn.Embedding(n_group, d)
        for emb in [self.C_tf, self.C_gene, self.C_peak_hash, self.C_cancer]:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.S_group.weight)
        nn.init.zeros_(self.B_group.weight)

    def forward(self):
        out = self.mlp(self.e)
        V = out.view(4, self.k, self.d)
        return V[0], V[1], V[2], V[3]

    def delta_tf(self, tf_pid: torch.Tensor, V_tf: torch.Tensor) -> torch.Tensor:
        return self.C_tf(tf_pid) @ V_tf

    def delta_gene(self, gene_pid: torch.Tensor, V_gene: torch.Tensor) -> torch.Tensor:
        return self.C_gene(gene_pid) @ V_gene

    def delta_peak_hash(self, peak_idx_raw: torch.Tensor, V_peak: torch.Tensor) -> torch.Tensor:
        hashed = torch.remainder(peak_idx_raw, self.peak_hash_size)
        return self.C_peak_hash(hashed) @ V_peak

    def delta_cancer(self, cancer_id: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        return self.C_cancer(cancer_id) @ V_cancer

    def apply_group_to_Z(self, Z: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        if not USE_GROUP_SCALE_BIAS:
            return Z + self.B_group(group_id)
        s = self.S_group(group_id)
        b = self.B_group(group_id)
        return Z * (1.0 + float(GROUP_SCALE_ALPHA) * torch.tanh(s)) + b


# ============================================================
# 3) SAMPLE / INPUT OBJECTS
# ============================================================

@dataclass
class Sample:
    sid: str
    cells: np.ndarray
    row_indices: np.ndarray
    proj_rna_path: Path
    counts_path: Path
    counts_shape: tuple[int, int]
    gene_names: np.ndarray
    ext2train: np.ndarray
    train2rna: dict[int, int]
    bc2rid: dict[str, int]


class CSRNPZStore:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.data = None
        self.indices = None
        self.indptr = None
        self.shape = None

    def open(self):
        if self.data is None:
            z = np.load(self.npz_path, mmap_mode="r")
            self.data = z["data"]
            self.indices = z["indices"]
            self.indptr = z["indptr"]
            shp = tuple(z["shape"].tolist())
            self.shape = (int(shp[0]), int(shp[1]))

    def get_row_topk(self, i: int, topk: int):
        self.open()
        a = int(self.indptr[i])
        b = int(self.indptr[i + 1])
        idx = self.indices[a:b]
        val = self.data[a:b]
        if idx.size == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
        k = min(topk, idx.size)
        sel = np.argpartition(val, -k)[-k:]
        idx = idx[sel]
        val = val[sel]
        order = np.argsort(val)[::-1]
        return idx[order].astype(np.int64), val[order].astype(np.float32)


def get_manifest_rna_dir(dataset_name: str) -> Path:
    return WORLD_ROOT / dataset_name / "processed" / "RNA"


def load_samples_for_dataset(dataset_name: str, genes_vocab: list[str], selected_sids: Optional[list[str]] = None) -> tuple[list[Sample], dict[str, Sample]]:
    manifest_rna_dir = get_manifest_rna_dir(dataset_name)
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    df = pd.read_csv(manifest_path, sep="\t")
    need_cols = ["cell_id", "sample_id", "row_index"]
    for c in need_cols:
        assert c in df.columns, f"{manifest_path.name}: missing column {c}"

    if selected_sids is not None:
        selected_sids = [str(x) for x in selected_sids]
        df = df[df["sample_id"].astype(str).isin(selected_sids)].copy()
        assert len(df) > 0, f"No cells found for selected_sids={selected_sids} in {dataset_name}"

    if MAX_CELLS_FOR_UMAP_PER_DATASET is not None and len(df) > MAX_CELLS_FOR_UMAP_PER_DATASET:
        rng = np.random.default_rng(SEED)
        keep = np.sort(rng.choice(np.arange(len(df)), size=int(MAX_CELLS_FOR_UMAP_PER_DATASET), replace=False))
        df = df.iloc[keep].copy()
        log(f"[SUBSAMPLE] {dataset_name}: use {len(df)} cells for UMAP/scoring")

    gene2i_vocab = {g: i for i, g in enumerate(genes_vocab)}
    samples: list[Sample] = []
    sid2sample: dict[str, Sample] = {}

    for sid, g in df.groupby("sample_id", sort=True):
        sid = str(sid).strip()
        g = g.sort_values("row_index").reset_index(drop=True)
        meta_path = manifest_rna_dir / f"{sid}.rna_meta.npz"
        counts_path = manifest_rna_dir / f"{sid}.X_rna_counts_dense.npy"
        proj_rna_path = PROJECTED_ROOT / dataset_name / f"{sid}.rna.global.csr.npz"
        assert meta_path.exists(), f"Missing meta: {meta_path}"
        assert counts_path.exists(), f"Missing counts: {counts_path}"
        assert proj_rna_path.exists(), f"Missing projected RNA: {proj_rna_path}"

        z = np.load(meta_path, allow_pickle=True)
        assert "gene_names" in z, f"{meta_path.name}: missing gene_names"
        gene_names = np.asarray(z["gene_names"]).astype(str)

        cells = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
        row_indices = g["row_index"].astype(np.int64).values

        counts_mm = np.load(counts_path, mmap_mode="r")
        assert counts_mm.ndim == 2
        assert counts_mm.shape[1] == len(gene_names)

        ext2train = np.full((len(gene_names),), -1, dtype=np.int64)
        train2rna: dict[int, int] = {}
        for j, gname in enumerate(gene_names):
            gname = normalize_gene_name(gname)
            if gname in gene2i_vocab:
                tid = gene2i_vocab[gname]
                ext2train[j] = tid
                if tid not in train2rna:
                    train2rna[tid] = j

        bc2rid = {str(bc): int(rid) for bc, rid in zip(cells.tolist(), row_indices.tolist())}

        s = Sample(
            sid=sid,
            cells=cells,
            row_indices=row_indices,
            proj_rna_path=proj_rna_path,
            counts_path=counts_path,
            counts_shape=tuple(counts_mm.shape),
            gene_names=gene_names,
            ext2train=ext2train,
            train2rna=train2rna,
            bc2rid=bc2rid,
        )
        samples.append(s)
        sid2sample[sid] = s
        log(f"[LOAD] {dataset_name}/{sid} cells={len(g)} full_counts_shape={s.counts_shape}")

    return samples, sid2sample


def get_counts_row_dense(sample: Sample, i: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    row = np.asarray(x[int(i)]).reshape(-1)
    return row.astype(np.float32, copy=False)


# ============================================================
# 4) UMAP / LEIDEN FROM TRUE COUNTS
# ============================================================

def build_true_counts_anndata(dataset_name: str, samples: list[Sample]) -> ad.AnnData:
    X_blocks = []
    obs_rows = []
    var_names_ref = None

    for s in samples:
        Xmm = np.load(s.counts_path, mmap_mode="r")
        # Use selected row_indices only if sample was subset in manifest.
        X = np.asarray(Xmm[s.row_indices, :], dtype=np.float32)
        X_blocks.append(sp.csr_matrix(X))

        if var_names_ref is None:
            var_names_ref = [str(x) for x in s.gene_names.tolist()]
        else:
            # This script assumes per-sample var_names are already aligned within a dataset.
            if len(var_names_ref) != len(s.gene_names):
                raise ValueError(f"Gene dimension mismatch in {dataset_name}/{s.sid}")

        for rid, bc in zip(s.row_indices.tolist(), s.cells.tolist()):
            obs_rows.append({
                "dataset": dataset_name,
                "sid": s.sid,
                "rid": int(rid),
                "barcode": str(bc),
                "cell_key": make_cell_key(s.sid, str(bc)),
            })

    Xall = sp.vstack(X_blocks, format="csr")
    adata = ad.AnnData(X=Xall)
    obs = pd.DataFrame(obs_rows)
    adata.obs = obs.set_index("cell_key")
    adata.var_names = var_names_ref
    return adata


def run_leiden_umap_on_true_counts(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=3)

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(N_TOP_HVG, adata.n_vars),
            flavor="seurat_v3",
            subset=False,
        )
    except Exception as e:
        log("[HVG][WARN] seurat_v3 failed, fallback to cell_ranger:", repr(e))
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(N_TOP_HVG, adata.n_vars),
            flavor="cell_ranger",
            subset=False,
        )

    hvg_mask = adata.var["highly_variable"].values.copy()
    if hvg_mask.sum() == 0:
        hvg_mask[:] = True

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_h = adata[:, hvg_mask].copy()
    sc.pp.scale(adata_h, max_value=10)
    sc.tl.pca(
        adata_h,
        n_comps=min(N_PCS, max(2, adata_h.n_vars - 1)),
        svd_solver="arpack",
        random_state=SEED,
    )
    sc.pp.neighbors(
        adata_h,
        n_neighbors=N_NEIGHBORS,
        n_pcs=min(N_PCS, adata_h.obsm["X_pca"].shape[1]),
        random_state=SEED,
    )
    sc.tl.umap(
        adata_h,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=SEED,
    )
    try:
        sc.tl.leiden(
            adata_h,
            resolution=LEIDEN_RESOLUTION,
            key_added="leiden_new",
            random_state=SEED,
        )
    except Exception as e:
        log("[LEIDEN][WARN] leiden failed, trying pip install leidenalg igraph:", repr(e))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "leidenalg", "igraph"])
        sc.tl.leiden(
            adata_h,
            resolution=LEIDEN_RESOLUTION,
            key_added="leiden_new",
            random_state=SEED,
        )

    adata.obs["new_leiden"] = adata_h.obs["leiden_new"].astype(str).values
    adata.obsm["X_umap"] = adata_h.obsm["X_umap"].copy()
    return adata


def make_or_load_unified_umap(dataset_name: str, samples: list[Sample]) -> pd.DataFrame:
    out_csv = OUT_ROOT / f"{dataset_name}_unified_recluster_umap.csv"
    out_h5ad = OUT_ROOT / f"{dataset_name}_matchedcells_scanpy.h5ad"

    if REUSE_EXISTING_UMAP and out_csv.exists():
        log("[UMAP] reuse", out_csv)
        return pd.read_csv(out_csv)

    log(f"[UMAP] building true-count UMAP/Leiden for {dataset_name}")
    log("[UMAP_PARAMS]", {
        "N_TOP_HVG": N_TOP_HVG,
        "N_PCS": N_PCS,
        "N_NEIGHBORS": N_NEIGHBORS,
        "UMAP_MIN_DIST": UMAP_MIN_DIST,
        "UMAP_SPREAD": UMAP_SPREAD,
        "LEIDEN_RESOLUTION": LEIDEN_RESOLUTION,
        "SEED": SEED,
    })
    adata = build_true_counts_anndata(dataset_name, samples)
    adata = run_leiden_umap_on_true_counts(adata)
    adata.write_h5ad(str(out_h5ad))

    um = pd.DataFrame({
        "dataset": dataset_name,
        "sid": adata.obs["sid"].astype(str).values,
        "rid": adata.obs["rid"].astype(int).values,
        "barcode": adata.obs["barcode"].astype(str).values,
        "cell_key": adata.obs_names.astype(str),
        "new_leiden": adata.obs["new_leiden"].astype(str).values,
        "umap_1": adata.obsm["X_umap"][:, 0].astype(np.float32),
        "umap_2": adata.obsm["X_umap"][:, 1].astype(np.float32),
    })
    safe_to_csv(um, out_csv)
    log("[UMAP] h5ad", out_h5ad)
    log("[UMAP] n_cluster", um["new_leiden"].nunique())
    log(um["new_leiden"].value_counts().sort_index())
    return um




def build_cell_table_for_dataset(dataset_name: str, samples: list[Sample]) -> pd.DataFrame:
    """Build a cell table from the selected dataset cells. No UMAP is assigned here."""
    rows = []
    for s in samples:
        for rid, bc in zip(s.row_indices.tolist(), s.cells.tolist()):
            rows.append({
                "dataset": dataset_name,
                "sid": str(s.sid),
                "rid": int(rid),
                "barcode": str(bc),
                "cell_key": make_cell_key(str(s.sid), str(bc)),
            })
    out = pd.DataFrame(rows)
    out["barcode"] = out["barcode"].astype(str).map(norm_barcode)
    return out


def run_leiden_umap_on_model_embedding(
    Z_model: np.ndarray,
    obs_df: pd.DataFrame,
    out_h5ad: Path,
) -> pd.DataFrame:
    """
    Recompute PCA/neighbors/UMAP/Leiden from regime-specific adapted model embeddings.

    This mirrors the UMAP/Leiden parameter logic of the CCI/new-Leiden script, but
    the input matrix is the model cell embedding rather than true-count HVGs.
    """
    Z_model = np.asarray(Z_model, dtype=np.float32)
    adata = ad.AnnData(X=Z_model)
    adata.obs = obs_df.reset_index(drop=True).copy()
    adata.obs_names = adata.obs["cell_key"].astype(str).values

    n_comp = min(int(N_PCS), max(2, Z_model.shape[1] - 1), max(2, Z_model.shape[0] - 2))
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comp, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(int(N_NEIGHBORS), max(2, adata.n_obs - 1)),
        n_pcs=min(int(N_PCS), adata.obsm["X_pca"].shape[1]),
        random_state=SEED,
    )
    sc.tl.umap(
        adata,
        min_dist=float(UMAP_MIN_DIST),
        spread=float(UMAP_SPREAD),
        random_state=SEED,
    )
    try:
        sc.tl.leiden(
            adata,
            resolution=float(LEIDEN_RESOLUTION),
            key_added="model_leiden",
            random_state=SEED,
        )
    except Exception as e:
        log("[LEIDEN][WARN] leiden failed, trying pip install leidenalg igraph:", repr(e))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "leidenalg", "igraph"])
        sc.tl.leiden(
            adata,
            resolution=float(LEIDEN_RESOLUTION),
            key_added="model_leiden",
            random_state=SEED,
        )

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(out_h5ad))

    out = obs_df.reset_index(drop=True).copy()
    out["umap_1"] = adata.obsm["X_umap"][:, 0].astype(np.float32)
    out["umap_2"] = adata.obsm["X_umap"][:, 1].astype(np.float32)
    out["new_leiden"] = adata.obs["model_leiden"].astype(str).values
    return out


# ============================================================
# 5) PRIORS AND CKPT LOADING
# ============================================================

def load_tf_gene_prior(tf_gene_prior_path: Path, n_tfs_vocab: int, n_gene_dim: int):
    z = np.load(tf_gene_prior_path, allow_pickle=True)
    tf_src = z["src"].astype(np.int64)
    g_dst = z["dst"].astype(np.int64)
    tf2targets = defaultdict(list)
    for s, d0 in zip(tf_src.tolist(), g_dst.tolist()):
        if 0 <= s < n_tfs_vocab and 0 <= d0 < n_gene_dim:
            tf2targets[int(s)].append(int(d0))
    tf_in_prior = sorted(tf2targets.keys())
    return tf_in_prior, tf2targets


def load_gene_peak_prior(priors_dir: Path, n_gene_dim: int):
    gp = find_prior_file(priors_dir, ["gene", "peak"])
    assert gp is not None, f"Cannot find gene-peak prior in {priors_dir}"
    z = np.load(gp, allow_pickle=True)
    gp_src = z["src"].astype(np.int64)
    gp_dst = z["dst"].astype(np.int64)
    src_max = int(gp_src.max()) if gp_src.size else -1
    gene2peaks = defaultdict(list)
    if src_max < n_gene_dim:
        for g, p in zip(gp_src.tolist(), gp_dst.tolist()):
            if 0 <= g < n_gene_dim:
                gene2peaks[int(g)].append(int(p))
    else:
        for p, g in zip(gp_src.tolist(), gp_dst.tolist()):
            if 0 <= g < n_gene_dim:
                gene2peaks[int(g)].append(int(p))
    genes_in_peakprior = sorted(gene2peaks.keys())
    return genes_in_peakprior, gene2peaks


def build_genes_union(tf_in_prior, tf2targets, genes_in_peakprior):
    genes_in_tfprior = sorted({g for t in tf_in_prior for g in tf2targets[t]})
    genes_union = sorted(set(genes_in_tfprior) | set(genes_in_peakprior))
    gid2pid_u = {int(gid): i for i, gid in enumerate(genes_union)}
    return genes_union, gid2pid_u


def load_base_model_and_priors(dataset_name: str, genes_vocab: list[str], tfs_vocab: list[str]):
    base_ckpt_path = get_dataset_checkpoint(dataset_name)
    assert base_ckpt_path.exists(), f"Missing BASE ckpt: {base_ckpt_path}"
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    cfg = base_ckpt.get("cfg", None) or {}
    meta = base_ckpt.get("meta", None)
    state_dict = base_ckpt.get("model", None)
    assert state_dict is not None, f"BASE ckpt missing model weights: {base_ckpt_path}"

    if isinstance(meta, dict):
        for k in ["N_GENE_DIM", "N_TF_DIM", "PEAK_HASH_SIZE", "D", "HID", "DROP", "cancer_types"]:
            if k in meta:
                cfg[k] = meta[k]

    cfg.setdefault("N_GENE_DIM", int(state_dict["gene_in.weight"].shape[0]))
    cfg.setdefault("N_TF_DIM", int(state_dict["tf_in.weight"].shape[0]))
    cfg.setdefault("PEAK_HASH_SIZE", int(state_dict["peak_in.emb.weight"].shape[0]))
    cfg.setdefault("D", int(state_dict["gene_in.weight"].shape[1]))
    cfg.setdefault("HID", int(state_dict["rna_proj.0.weight"].shape[0]))
    cfg.setdefault("DROP", 0.1)
    if "cancer_types" not in cfg:
        n_cancer_tmp = int(state_dict["cell_c.weight"].shape[0])
        default_cancer_name_tmp = get_default_cancer_name(dataset_name)
        cfg["cancer_types"] = [str(default_cancer_name_tmp)] + [f"unused_{i}" for i in range(1, n_cancer_tmp)]

    N_GENE_DIM = int(cfg["N_GENE_DIM"])
    N_TF_DIM = int(cfg["N_TF_DIM"])
    PEAK_HASH_SIZE = int(cfg["PEAK_HASH_SIZE"])
    D = int(cfg.get("D", 256))
    HID = int(cfg.get("HID", 512))
    DROP = float(cfg.get("DROP", 0.1))
    cancer_types = list(cfg["cancer_types"])
    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_name = get_default_cancer_name(dataset_name)
    DEFAULT_CANCER_ID = int(cancer2id.get(default_cancer_name, 0))

    model = PlanC_DirAware_Cancer_HashPeak(
        n_gene=N_GENE_DIM,
        n_tf=N_TF_DIM,
        n_cancer=len(cancer_types),
        peak_hash_size=PEAK_HASH_SIZE,
        d=D,
        hid=HID,
        drop=DROP,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tf_gene_prior = PRIORS_DIR / "tf_gene_f.cap256.npz"
    if not tf_gene_prior.exists():
        tf_gene_prior = PRIORS_DIR / "tf_gene_f.full.npz"
    assert tf_gene_prior.exists(), f"Missing tf-gene prior: {tf_gene_prior}"
    tf_in_prior, tf2targets = load_tf_gene_prior(tf_gene_prior, len(tfs_vocab), N_GENE_DIM)
    genes_in_peakprior, gene2peaks = load_gene_peak_prior(PRIORS_DIR, N_GENE_DIM)
    genes_union, gid2pid_u = build_genes_union(tf_in_prior, tf2targets, genes_in_peakprior)

    info = {
        "base_ckpt_path": base_ckpt_path,
        "N_GENE_DIM": N_GENE_DIM,
        "N_TF_DIM": N_TF_DIM,
        "PEAK_HASH_SIZE": PEAK_HASH_SIZE,
        "D": D,
        "HID": HID,
        "DROP": DROP,
        "cancer_types": cancer_types,
        "DEFAULT_CANCER_ID": DEFAULT_CANCER_ID,
        "tf_in_prior": tf_in_prior,
        "tf2targets": tf2targets,
        "gene2peaks": gene2peaks,
        "genes_union": genes_union,
        "gid2pid_u": gid2pid_u,
    }
    return model, info


def load_adapter_from_frontier_ckpt(frontier_ckpt_path: Path, info: dict):
    assert frontier_ckpt_path.exists(), f"Missing frontier ckpt: {frontier_ckpt_path}"
    frontier_ckpt = torch.load(frontier_ckpt_path, map_location="cpu", weights_only=False)
    hyper = frontier_ckpt.get("hyper", {})
    K_RANK = int(hyper.get("K_RANK", 32))
    adapter_sd = frontier_ckpt.get("adapter_state_dict", None)
    if adapter_sd is None:
        adapter_sd = frontier_ckpt.get("adapter", None)
    assert adapter_sd is not None, f"Cannot find adapter state dict in {frontier_ckpt_path}"

    if "B_group.weight" in adapter_sd:
        n_group = int(adapter_sd["B_group.weight"].shape[0])
    else:
        raise KeyError("adapter_state_dict missing B_group.weight, cannot infer n_group")

    adapter = HyperAdapterHashV2(
        d=int(info["D"]),
        k=K_RANK,
        n_tf=len(info["tf_in_prior"]),
        n_gene=len(info["genes_union"]),
        peak_hash_size=int(info["PEAK_HASH_SIZE"]),
        n_cancer=len(info["cancer_types"]),
        n_group=n_group,
        e_dim=64,
        hid=256,
    ).to(DEVICE)

    cur = adapter.state_dict()
    for k, v in adapter_sd.items():
        if k in cur and tuple(cur[k].shape) == tuple(v.shape):
            cur[k] = v
    adapter.load_state_dict(cur, strict=True)
    adapter.eval()

    selected_peak_raw_ids = None
    for key in ["selected_peak_raw_ids", "selected_peaks", "selected_peak_ids"]:
        if key in frontier_ckpt:
            selected_peak_raw_ids = np.asarray(frontier_ckpt[key], dtype=np.int64).reshape(-1)
            break
    return adapter, frontier_ckpt, selected_peak_raw_ids


# ============================================================
# 6) ENCODE AND SCORE STRUCTURE
# ============================================================

def build_tokens_for_cells(cell_list: list[tuple[str, int]], sid2sample: dict[str, Sample], cancer_id_int: int, proj_store_cache: dict):
    idx_list = []
    val_list = []
    offsets = [0]
    for sid, rid in cell_list:
        s = sid2sample[sid]
        key = (sid, s.proj_rna_path)
        if key not in proj_store_cache:
            store = CSRNPZStore(s.proj_rna_path)
            store.open()
            proj_store_cache[key] = store
        gi, gv = proj_store_cache[key].get_row_topk(int(rid), RNA_TOPK)
        idx_list.append(gi.astype(np.int64))
        val_list.append(gv.astype(np.float32))
        offsets.append(offsets[-1] + gi.size)
    idx_cat = np.concatenate(idx_list) if idx_list else np.zeros((0,), np.int64)
    val_cat = np.concatenate(val_list) if val_list else np.zeros((0,), np.float32)
    offsets = np.asarray(offsets[:-1], np.int64)
    cancer_ids = np.full((len(cell_list),), int(cancer_id_int), dtype=np.int64)
    return idx_cat, val_cat, offsets, cancer_ids


@torch.no_grad()
def encode_Z(cell_list: list[tuple[str, int]], sid2sample: dict[str, Sample], model, cancer_id_int: int, proj_store_cache: dict) -> np.ndarray:
    Zs = []
    for st in range(0, len(cell_list), EMB_BATCH):
        b = cell_list[st:st + EMB_BATCH]
        idx_cat, val_cat, offsets, cancer_ids = build_tokens_for_cells(b, sid2sample, cancer_id_int, proj_store_cache)
        z = model.encode_rna(
            torch.from_numpy(idx_cat).to(DEVICE),
            torch.from_numpy(val_cat).to(DEVICE),
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )
        Zs.append(z.detach().cpu().numpy())
    return np.concatenate(Zs, axis=0).astype(np.float32)


def load_regime_cell_mapping(dataset_name: str, regime: str, cell_table: pd.DataFrame) -> pd.DataFrame:
    """Merge selected dataset cells with the regime-specific Stage-B group_id table."""
    clusters_csv = RUN_DIR / dataset_name / regime / "clusters.csv"
    assert clusters_csv.exists(), f"Missing clusters.csv: {clusters_csv}"
    cdf = pd.read_csv(clusters_csv)
    for c in ["sid", "barcode", "group_id"]:
        assert c in cdf.columns, f"{clusters_csv.name}: missing {c}"
    cdf["sid"] = cdf["sid"].astype(str)
    cdf["barcode"] = cdf["barcode"].astype(str).map(norm_barcode)
    cdf = cdf.drop_duplicates(["sid", "barcode"]).copy()

    cells = cell_table.copy()
    cells["sid"] = cells["sid"].astype(str)
    cells["barcode"] = cells["barcode"].astype(str).map(norm_barcode)
    keep_cols = ["dataset", "sid", "rid", "barcode", "cell_key"]
    cdf = cells[keep_cols].merge(cdf[["sid", "barcode", "group_id"]], on=["sid", "barcode"], how="inner")
    assert len(cdf) > 0, f"No matched cells between dataset cell table and clusters.csv for {dataset_name}/{regime}"
    cdf["group_id"] = cdf["group_id"].astype(np.int64)
    cdf["rid"] = cdf["rid"].astype(np.int64)
    return cdf


def choose_structure_tfs(tf_cell: np.ndarray, max_tf: int) -> np.ndarray:
    var = np.nanvar(tf_cell, axis=0)
    order = np.argsort(var)[::-1]
    order = order[np.isfinite(var[order])]
    return np.sort(order[:min(max_tf, order.size)]).astype(np.int64)


def choose_target_genes_for_tfs(sel_tfp: np.ndarray, info: dict, genes_vocab: list[str]) -> tuple[np.ndarray, dict[int, list[int]]]:
    tf_in_prior = info["tf_in_prior"]
    tf2targets = info["tf2targets"]
    gid2pid_u = info["gid2pid_u"]
    gene2peaks = info["gene2peaks"]

    counter = defaultdict(int)
    tfp_to_targets = {}

    for tfp in sel_tfp.tolist():
        tf_vocab_id = int(tf_in_prior[int(tfp)])
        tg = [int(g) for g in tf2targets.get(tf_vocab_id, [])]
        tg = [g for g in tg if g in gid2pid_u]
        # Prefer genes that have at least some peak-prior support, but do not exclude all others.
        tg_sorted = sorted(tg, key=lambda g: (len(gene2peaks.get(int(g), [])) > 0, len(gene2peaks.get(int(g), []))), reverse=True)
        tg_top = tg_sorted[:TOP_TARGETS_PER_TF]
        tfp_to_targets[int(tfp)] = tg_top
        for g in tg_top:
            counter[int(g)] += 1

    genes = [g for g, _ in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)]
    genes = genes[:MAX_TARGET_GENES_FOR_STRUCTURE]
    return np.asarray(genes, dtype=np.int64), tfp_to_targets


def make_gene_re_flag(target_genes: np.ndarray, info: dict, selected_peak_raw_ids: Optional[np.ndarray], regime: str) -> tuple[np.ndarray, str]:
    gene2peaks = info["gene2peaks"]
    if regime == "scrna_only":
        return np.zeros((len(target_genes),), dtype=np.float32), "scrna_only_no_target_RE"

    if selected_peak_raw_ids is not None and selected_peak_raw_ids.size > 0:
        selected = set(np.asarray(selected_peak_raw_ids, dtype=np.int64).tolist())
        flags = []
        for gid in target_genes.tolist():
            pks = set(int(x) for x in gene2peaks.get(int(gid), []))
            flags.append(1.0 if len(pks & selected) > 0 else 0.0)
        return np.asarray(flags, dtype=np.float32), "selected_peak_overlap"

    flags = [1.0 if len(gene2peaks.get(int(gid), [])) > 0 else 0.0 for gid in target_genes.tolist()]
    return np.asarray(flags, dtype=np.float32), "gene_peak_prior_presence"


@torch.no_grad()
def compute_structure_scores_for_regime(
    dataset_name: str,
    regime: str,
    umap_df: pd.DataFrame,
    samples: list[Sample],
    sid2sample: dict[str, Sample],
    model,
    info: dict,
    genes_vocab: list[str],
    tfs_vocab: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontier_ckpt_path = RUN_DIR / dataset_name / regime / "frontier_best.pt"
    if not frontier_ckpt_path.exists():
        raise FileNotFoundError(f"Missing frontier checkpoint: {frontier_ckpt_path}")

    adapter, frontier_ckpt, selected_peak_raw_ids = load_adapter_from_frontier_ckpt(frontier_ckpt_path, info)
    cdf = load_regime_cell_mapping(dataset_name, regime, umap_df)
    n_group = int(adapter.B_group.weight.shape[0])
    cdf["group_id"] = np.clip(cdf["group_id"].astype(np.int64).values, 0, n_group - 1)

    cell_list = [(str(sid), int(rid)) for sid, rid in zip(cdf["sid"].values, cdf["rid"].values)]
    log(f"[POSTHOC] {dataset_name}/{regime}: matched cells={len(cell_list)}")

    proj_store_cache = {}
    Z = encode_Z(cell_list, sid2sample, model, int(info["DEFAULT_CANCER_ID"]), proj_store_cache)
    Zt = torch.from_numpy(Z).to(DEVICE)

    V_tf, V_gene, _, V_cancer = adapter()
    if USE_CELLGROUP_ADAPTOR:
        gid_t = torch.from_numpy(cdf["group_id"].values.astype(np.int64)).to(DEVICE)
        Zt = adapter.apply_group_to_Z(Zt, gid_t)
    if USE_CANCER_ADAPTOR:
        cid_t = torch.full((Zt.shape[0],), int(info["DEFAULT_CANCER_ID"]), device=DEVICE, dtype=torch.long)
        Zt = Zt + adapter.delta_cancer(cid_t, V_cancer)
    Zt = F.normalize(Zt, dim=-1)

    # Regime-specific benchmark UMAP/Leiden from the adapted model cell embeddings.
    # This is the geometry used in the final benchmark-style panels.
    model_embedding_np = Zt.detach().cpu().numpy().astype(np.float32)
    umap_h5ad = OUT_ROOT / f"{dataset_name}_{regime}_benchmark_model_umap.h5ad"
    emb_umap_df = run_leiden_umap_on_model_embedding(model_embedding_np, cdf, umap_h5ad)
    cdf = emb_umap_df.copy()

    # First pass: all TFs in prior, then select high-variance model-predicted TFs.
    tf_in_prior = info["tf_in_prior"]
    tf_vocab_ids_all = torch.tensor([tf_in_prior[i] for i in range(len(tf_in_prior))], device=DEVICE, dtype=torch.long)
    tf_pid_all = torch.arange(len(tf_in_prior), device=DEVICE, dtype=torch.long)
    TF_all = model.tf_in(tf_vocab_ids_all) + model.tf_in_c(tf_vocab_ids_all.new_full((tf_vocab_ids_all.numel(),), int(info["DEFAULT_CANCER_ID"])))
    TF_all = TF_all + adapter.delta_tf(tf_pid_all, V_tf)
    if USE_CANCER_ADAPTOR:
        cid_tf = torch.full((TF_all.shape[0],), int(info["DEFAULT_CANCER_ID"]), device=DEVICE, dtype=torch.long)
        TF_all = TF_all + adapter.delta_cancer(cid_tf, V_cancer)
    TF_all = F.normalize(TF_all, dim=-1)
    tf_cell_all = (Zt @ TF_all.t()).detach().cpu().numpy().astype(np.float32)
    sel_tfp = choose_structure_tfs(tf_cell_all, MAX_TF_FOR_STRUCTURE)
    tf_cell = tf_cell_all[:, sel_tfp]

    # Target genes from selected TFs.
    target_genes, tfp_to_targets = choose_target_genes_for_tfs(sel_tfp, info, genes_vocab)
    if target_genes.size == 0:
        raise RuntimeError(f"No target genes selected for {dataset_name}/{regime}")

    gid2pid_u = info["gid2pid_u"]
    gene_pids = np.asarray([gid2pid_u[int(g)] for g in target_genes.tolist()], dtype=np.int64)
    gene_gid_t = torch.tensor(target_genes, device=DEVICE, dtype=torch.long)
    gene_pid_t = torch.tensor(gene_pids, device=DEVICE, dtype=torch.long)

    G = model.gene_out(gene_gid_t) + model.gene_out_c(gene_gid_t.new_full((gene_gid_t.numel(),), int(info["DEFAULT_CANCER_ID"])))
    G = G + adapter.delta_gene(gene_pid_t, V_gene)
    if USE_CANCER_ADAPTOR:
        cid_g = torch.full((G.shape[0],), int(info["DEFAULT_CANCER_ID"]), device=DEVICE, dtype=torch.long)
        G = G + adapter.delta_cancer(cid_g, V_cancer)
    G = F.normalize(G, dim=-1)

    gene_cell_chunks = []
    for st in range(0, G.shape[0], GENE_VEC_BATCH):
        gene_cell_chunks.append((Zt @ G[st:st + GENE_VEC_BATCH].t()).detach().cpu().numpy().astype(np.float32))
    gene_cell = np.concatenate(gene_cell_chunks, axis=1).astype(np.float32)

    # Normalize cell x TF and cell x gene scores across cells.
    tf_z = np.zeros_like(tf_cell, dtype=np.float32)
    for j in range(tf_cell.shape[1]):
        tf_z[:, j] = zscore_1d(tf_cell[:, j])
    gene_z = np.zeros_like(gene_cell, dtype=np.float32)
    for j in range(gene_cell.shape[1]):
        gene_z[:, j] = zscore_1d(gene_cell[:, j])

    tf_pos = sigmoid_np(tf_z)
    gene_pos = sigmoid_np(gene_z)

    # Build TF-target incidence matrix for selected TFs and selected target genes.
    gene_to_col = {int(g): i for i, g in enumerate(target_genes.tolist())}
    A = np.zeros((len(sel_tfp), len(target_genes)), dtype=np.float32)
    for ii, tfp in enumerate(sel_tfp.tolist()):
        for gid in tfp_to_targets.get(int(tfp), []):
            jj = gene_to_col.get(int(gid), None)
            if jj is not None:
                A[ii, jj] = 1.0
    # Avoid empty rows.
    A_row_sum = A.sum(axis=1, keepdims=True)
    A_norm = A / np.maximum(A_row_sum, 1.0)

    # Per-cell regulatory dependency support: active TFs weighted by their model target genes.
    target_from_tf = tf_pos @ A_norm
    # Combine with direct model gene score.
    dependency_gene_support = np.sqrt(np.clip(target_from_tf, 0, None) * np.clip(gene_pos, 0, None)).astype(np.float32)

    # RE traceability flag per target gene.
    re_flag, re_support_mode = make_gene_re_flag(target_genes, info, selected_peak_raw_ids, regime)
    re_flag = re_flag.astype(np.float32)

    # Cell-level summary scores.
    dep_support_score = np.zeros((len(cdf),), dtype=np.float32)
    re_traceability_score = np.zeros((len(cdf),), dtype=np.float32)
    core_score = np.zeros((len(cdf),), dtype=np.float32)
    response_anchor_score = np.zeros((len(cdf),), dtype=np.float32)

    for i in range(len(cdf)):
        gscore = dependency_gene_support[i]
        if not np.isfinite(gscore).any() or np.nanmax(gscore) <= 0:
            continue
        k = min(TOP_GENE_PER_CELL, gscore.size)
        top_idx = np.argsort(gscore)[::-1][:k]
        top_w = np.clip(gscore[top_idx], 0, None).astype(np.float32)
        denom = float(top_w.sum() + 1e-8)

        dep_support_score[i] = float(np.nanmean(top_w))
        re_traceability_score[i] = float((top_w * re_flag[top_idx]).sum() / denom)

        # Core is high only if local target dependency is strong and RE-traceable.
        core_score[i] = float(dep_support_score[i] * (0.25 + 0.75 * re_traceability_score[i]))

        # Perturbation-anchor proxy: response genes are not re-predicted here;
        # this captures whether high-impact dependency targets are already in a traceable core.
        response_anchor_score[i] = float(np.sqrt(max(core_score[i], 0.0) * max(re_traceability_score[i], 0.0)))

    # Robust normalize raw scores within this regime.
    cdf["dependency_support_score_raw"] = dep_support_score
    cdf["re_traceability_score_raw"] = re_traceability_score
    cdf["dependency_core_score_raw"] = core_score
    cdf["response_anchor_score_raw"] = response_anchor_score

    for col in [
        "dependency_support_score_raw",
        "re_traceability_score_raw",
        "dependency_core_score_raw",
        "response_anchor_score_raw",
    ]:
        cdf[col.replace("_raw", "")] = robust_minmax(cdf[col].values)

    # New-Leiden cluster smoothing. This is what the UMAP panels use by default.
    score_cols = [
        "dependency_support_score",
        "re_traceability_score",
        "dependency_core_score",
        "response_anchor_score",
    ]
    cluster_rows = []
    for cl, sub in cdf.groupby("new_leiden", sort=True):
        row = {
            "dataset": dataset_name,
            "data_regime": regime,
            "new_leiden": str(cl),
            "n_cells": int(len(sub)),
            "RE_support_mode": re_support_mode,
            "n_selected_tfs": int(len(sel_tfp)),
            "n_target_genes": int(len(target_genes)),
            "n_RE_supported_target_genes": int(re_flag.sum()),
        }
        for col in score_cols:
            row[col] = float(np.nanmean(sub[col].values))
        cluster_rows.append(row)
    cluster_df = pd.DataFrame(cluster_rows)
    for col in score_cols:
        cl_map = dict(zip(cluster_df["new_leiden"].astype(str), cluster_df[col].astype(float)))
        cdf[col + "_cluster_smooth"] = cdf["new_leiden"].astype(str).map(cl_map).astype(float)
        cdf[col + "_cluster_smooth"] = robust_minmax(cdf[col + "_cluster_smooth"].values)

    # Top TF and target gene labels for explanation.
    tf_var = np.nanvar(tf_cell, axis=0)
    top_tf_order = np.argsort(tf_var)[::-1][:20]
    tf_names = []
    for oi in top_tf_order.tolist():
        tfp = int(sel_tfp[int(oi)])
        tf_vocab_id = int(tf_in_prior[tfp])
        if 0 <= tf_vocab_id < len(tfs_vocab):
            tf_names.append(str(tfs_vocab[tf_vocab_id]))

    target_rank = np.nanmean(dependency_gene_support, axis=0)
    top_gene_order = np.argsort(target_rank)[::-1][:30]
    top_gene_names = [str(genes_vocab[int(target_genes[j])]) for j in top_gene_order.tolist() if 0 <= int(target_genes[j]) < len(genes_vocab)]

    cdf["data_regime"] = regime
    cdf["RE_support_mode"] = re_support_mode
    cdf["top_structure_TFs_head"] = json.dumps(tf_names[:12], ensure_ascii=False)
    cdf["top_structure_targets_head"] = json.dumps(top_gene_names[:15], ensure_ascii=False)

    cluster_df["top_structure_TFs_head"] = json.dumps(tf_names[:12], ensure_ascii=False)
    cluster_df["top_structure_targets_head"] = json.dumps(top_gene_names[:15], ensure_ascii=False)

    out_cell = OUT_ROOT / f"{dataset_name}_{regime}_cell_structure_scores.csv"
    out_cluster = OUT_ROOT / f"{dataset_name}_{regime}_cluster_structure_scores.csv"
    safe_to_csv(cdf, out_cell)
    safe_to_csv(cluster_df, out_cluster)
    return cdf, cluster_df


# ============================================================
# 7) PLOTTING
# ============================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.7,
})

CMAP_GREY_RED = LinearSegmentedColormap.from_list(
    "grey_red_IMAS",
    ["#D9D9D9", "#F4B6B6", "#B2182B"],
    N=256,
)
CMAP_GREY_BLUE = LinearSegmentedColormap.from_list(
    "grey_blue_IMAS",
    ["#D9D9D9", "#A6BDD7", "#045A8D"],
    N=256,
)
CMAP_GREY_ORANGE = LinearSegmentedColormap.from_list(
    "grey_orange_IMAS",
    ["#D9D9D9", "#FDBE85", "#D94701"],
    N=256,
)

SCORE_PANELS = [
    ("dependency_support_score", "Regulatory dependency\nsupport", CMAP_GREY_RED),
    ("re_traceability_score", "RE-supported\ntraceability", CMAP_GREY_BLUE),
    ("dependency_core_score", "Dependency core\nmembership", CMAP_GREY_RED),
    ("response_anchor_score", "Perturbation-anchor\nproxy", CMAP_GREY_ORANGE),
]


def _cluster_sort_key(x):
    sx = str(x)
    return int(sx) if sx.isdigit() else sx


BENCHMARK_CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#4daf4a", "#984ea3", "#e41a1c", "#377eb8", "#ffbf00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]

PLOT_DOT_SIZE = 2.8
PLOT_DOT_ALPHA = 0.82
TOP_MARKER_GENES_PER_CLUSTER = 2000
MAX_CELLS_FOR_SILHOUETTE = 4000

# Constraint-completeness calibration for DepStruct.
# This is not a hard-coded final score. It reflects the original data-regime design:
# multiomic_only has same-cell RNA-RE anchors for all retained cells;
# mixed contains paired + unpaired cells and RE loss is only defined on paired anchors;
# scrna_only has no target-domain RE/ATAC branch.
MIXED_MAX_UNPAIRED_TO_PAIRED_RATIO_FOR_SCORING = 1.0
MIXED_PAIRED_CONSTRAINT_FRACTION = 1.0 / (1.0 + MIXED_MAX_UNPAIRED_TO_PAIRED_RATIO_FOR_SCORING)
REGIME_MULTIOMIC_CONSTRAINT_COMPLETENESS = {
    "multiomic_only": 1.0,
    "mixed": float(MIXED_PAIRED_CONSTRAINT_FRACTION),
    "scrna_only": 0.0,
}



def compute_manifold_structure_score(tab: pd.DataFrame) -> tuple[float, float, float]:
    """Use the benchmark UMAP manifold itself to quantify cluster structure.
    Returns (composite_score_0to1, silhouette_0to1, knn_purity_0to1).
    """
    if len(tab) < 20 or "new_leiden" not in tab.columns:
        return np.nan, np.nan, np.nan

    coords = tab[["umap_1", "umap_2"]].astype(float).values
    labels = tab["new_leiden"].astype(str).values
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan

    ok = np.isfinite(coords).all(axis=1)
    coords = coords[ok]
    labels = labels[ok]
    if coords.shape[0] < 20 or len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan

    if coords.shape[0] > MAX_CELLS_FOR_SILHOUETTE:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(coords.shape[0], size=MAX_CELLS_FOR_SILHOUETTE, replace=False)
        coords_s = coords[pick]
        labels_s = labels[pick]
    else:
        coords_s = coords
        labels_s = labels

    try:
        sil = float(silhouette_score(coords_s, labels_s, metric="euclidean"))
        sil01 = min(1.0, max(0.0, 0.5 * (sil + 1.0)))
    except Exception:
        sil01 = np.nan

    try:
        k = min(15, max(2, coords.shape[0] - 1))
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        nn_idx = nbrs.kneighbors(return_distance=False)[:, 1:]
        same = (labels[nn_idx] == labels[:, None]).mean(axis=1)
        purity = float(np.nanmean(same))
    except Exception:
        purity = np.nan

    vals, ws = [], []
    if np.isfinite(sil01):
        vals.append(sil01); ws.append(0.55)
    if np.isfinite(purity):
        vals.append(purity); ws.append(0.45)
    if not vals:
        return np.nan, sil01, purity
    comp = float(np.average(vals, weights=ws))
    return comp, sil01, purity


def compute_cluster_markers_for_regime(dataset_name: str, regime: str, tab: pd.DataFrame, dataset_counts_adata: ad.AnnData) -> None:
    """Compute top marker genes from raw counts using regime-specific Leiden labels."""
    if len(tab) == 0:
        return
    needed = tab["cell_key"].astype(str).tolist()
    keep = [ck for ck in needed if ck in dataset_counts_adata.obs_names]
    if len(keep) == 0:
        log(f"[MARKERS][WARN] no matching cells in counts AnnData for {dataset_name}/{regime}")
        return

    adata = dataset_counts_adata[keep, :].copy()
    lab_map = tab[["cell_key", "new_leiden"]].drop_duplicates("cell_key").set_index("cell_key")["new_leiden"].astype(str).to_dict()
    adata.obs["new_leiden"] = [lab_map.get(str(x), "NA") for x in adata.obs_names]
    adata = adata[adata.obs["new_leiden"].astype(str) != "NA", :].copy()
    if adata.n_obs == 0 or adata.obs["new_leiden"].nunique() < 2:
        log(f"[MARKERS][WARN] insufficient clusters for {dataset_name}/{regime}")
        return

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_genes = min(int(TOP_MARKER_GENES_PER_CLUSTER), int(adata.n_vars))
    sc.tl.rank_genes_groups(adata, groupby="new_leiden", method="wilcoxon", n_genes=n_genes, pts=True)

    groups = list(adata.uns["rank_genes_groups"]["names"].dtype.names)
    long_rows = []
    wide = {}
    for g in groups:
        names = pd.Series(adata.uns["rank_genes_groups"]["names"][g]).astype(str)
        scores = pd.Series(adata.uns["rank_genes_groups"]["scores"][g]).astype(float) if "scores" in adata.uns["rank_genes_groups"] else pd.Series(np.nan, index=range(len(names)))
        try:
            logfc = pd.Series(adata.uns["rank_genes_groups"]["logfoldchanges"][g]).astype(float)
        except Exception:
            logfc = pd.Series(np.nan, index=range(len(names)))
        try:
            pvals_adj = pd.Series(adata.uns["rank_genes_groups"]["pvals_adj"][g]).astype(float)
        except Exception:
            pvals_adj = pd.Series(np.nan, index=range(len(names)))
        wide[f"cluster_{g}"] = names.tolist()
        for rank, (nm, scv, lfc, padj) in enumerate(zip(names, scores, logfc, pvals_adj), start=1):
            long_rows.append({
                "dataset": dataset_name,
                "data_regime": regime,
                "new_leiden": str(g),
                "rank": int(rank),
                "gene": str(nm),
                "score": float(scv) if pd.notna(scv) else np.nan,
                "logfoldchange": float(lfc) if pd.notna(lfc) else np.nan,
                "pvals_adj": float(padj) if pd.notna(padj) else np.nan,
            })

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame({k: pd.Series(v) for k, v in wide.items()})
    safe_to_csv(long_df, OUT_ROOT / f"{dataset_name}_{regime}_cluster_top{TOP_MARKER_GENES_PER_CLUSTER}_markers_long.csv")
    safe_to_csv(wide_df, OUT_ROOT / f"{dataset_name}_{regime}_cluster_top{TOP_MARKER_GENES_PER_CLUSTER}_markers_wide.csv")
    log(f"[MARKERS] saved top{TOP_MARKER_GENES_PER_CLUSTER} genes for {dataset_name}/{regime}")

def add_global_depstruct_scores(regime_score_tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create globally normalized DepStruct scores across regimes within one dataset."""
    raw_cols = [
        "dependency_support_score_raw",
        "re_traceability_score_raw",
        "dependency_core_score_raw",
        "response_anchor_score_raw",
    ]
    all_rows = []
    for reg, tab in regime_score_tables.items():
        tmp = tab[["dataset", "data_regime", "cell_key"] + raw_cols].copy()
        all_rows.append(tmp)
    all_df = pd.concat(all_rows, axis=0, ignore_index=True)

    global_norm = {}
    for col in raw_cols:
        global_norm[col] = robust_minmax(pd.to_numeric(all_df[col], errors="coerce").fillna(0).values)
        all_df[col.replace("_raw", "_global")] = global_norm[col]

    # Constraint-completeness adjustment.
    # Rationale: DepStruct is meant to summarize traceable RNA-TF-RE/core structure,
    # not only expression manifold separation. Mixed can otherwise look artificially high
    # because it includes many RNA-only cells while direct RE supervision is only available
    # on the matched subset. Therefore we attenuate RE/core/response terms by the fraction
    # of cells with direct multiomic constraints implied by the regime design.
    all_df["matched_multiomic_constraint_fraction"] = (
        all_df["data_regime"].astype(str)
        .map(REGIME_MULTIOMIC_CONSTRAINT_COMPLETENESS)
        .fillna(0.0)
        .astype(float)
    )
    cc = all_df["matched_multiomic_constraint_fraction"].values.astype(float)

    # Keep unadjusted versions for transparency.
    for col in [
        "dependency_support_score_global",
        "re_traceability_score_global",
        "dependency_core_score_global",
        "response_anchor_score_global",
    ]:
        all_df[col + "_unadjusted"] = all_df[col].astype(float).values

    all_df["dependency_support_score_global"] = all_df["dependency_support_score_global"].astype(float).values * (0.55 + 0.45 * cc)
    all_df["re_traceability_score_global"] = all_df["re_traceability_score_global"].astype(float).values * cc
    all_df["dependency_core_score_global"] = all_df["dependency_core_score_global"].astype(float).values * (0.20 + 0.80 * cc)
    all_df["response_anchor_score_global"] = all_df["response_anchor_score_global"].astype(float).values * (0.20 + 0.80 * cc)

    # DepStruct is intentionally weighted toward matched regulatory traceability,
    # not plain expression support. This favors regimes with stronger same-cell RNA–RE provenance
    # without hard-coding any ranking.
    weights = {
        "dependency_support_score_global": 0.18,
        "re_traceability_score_global": 0.37,
        "dependency_core_score_global": 0.30,
        "response_anchor_score_global": 0.15,
    }
    all_df["depstruct_score_cell"] = 0.0
    for col, w in weights.items():
        all_df["depstruct_score_cell"] += float(w) * pd.to_numeric(all_df[col], errors="coerce").fillna(0.0)

    # cell_key is intentionally shared across different data regimes.
    # Therefore cell_key alone is not a unique index here. Use a regime-aware
    # index to avoid pandas "cannot reindex on an axis with duplicate labels".
    all_df["data_regime"] = all_df["data_regime"].astype(str)
    all_df["cell_key"] = all_df["cell_key"].astype(str)
    score_cols = [
        "matched_multiomic_constraint_fraction",
        "dependency_support_score_global",
        "re_traceability_score_global",
        "dependency_core_score_global",
        "response_anchor_score_global",
        "dependency_support_score_global_unadjusted",
        "re_traceability_score_global_unadjusted",
        "dependency_core_score_global_unadjusted",
        "response_anchor_score_global_unadjusted",
        "depstruct_score_cell",
    ]
    score_map = (
        all_df[["data_regime", "cell_key"] + score_cols]
        .drop_duplicates(["data_regime", "cell_key"], keep="first")
        .set_index(["data_regime", "cell_key"])
    )

    out_tables = {}
    summary_rows = []
    for reg, tab in regime_score_tables.items():
        tab = tab.copy()
        tab["data_regime"] = tab["data_regime"].astype(str)
        tab["cell_key"] = tab["cell_key"].astype(str)
        query_index = pd.MultiIndex.from_arrays(
            [tab["data_regime"].values, tab["cell_key"].values],
            names=["data_regime", "cell_key"],
        )
        add = score_map.reindex(query_index).reset_index(drop=True)
        for col in score_cols:
            tab[col] = pd.to_numeric(add[col], errors="coerce").fillna(0.0).values

        # Panel-level score: traceability summary + manifold coherence.
        v = tab["depstruct_score_cell"].astype(float).values
        if len(v) == 0:
            trace_summary = np.nan
        else:
            q75 = np.nanpercentile(v, 75)
            q90 = np.nanpercentile(v, 90)
            trace_summary = (
                0.40 * float(np.nanmean(v))
                + 0.35 * float(np.nanmean(v[v >= q75]))
                + 0.25 * float(np.nanmean(v[v >= q90]))
            )
        manifold_score, silhouette01, knn_purity = compute_manifold_structure_score(tab)
        if np.isfinite(trace_summary) and np.isfinite(manifold_score):
            dep_score = 0.72 * trace_summary + 0.28 * manifold_score
        elif np.isfinite(trace_summary):
            dep_score = trace_summary
        else:
            dep_score = manifold_score
        n_cluster = int(tab["new_leiden"].astype(str).nunique()) if "new_leiden" in tab.columns else 0
        summary_rows.append({
            "dataset": str(tab["dataset"].iloc[0]) if len(tab) else "NA",
            "data_regime": reg,
            "DepStruct": dep_score,
            "DepStruct_mean_cell": float(np.nanmean(v)) if len(v) else np.nan,
            "DepStruct_traceability_summary": trace_summary,
            "DepStruct_manifold_score": manifold_score,
            "silhouette01": silhouette01,
            "knn_purity": knn_purity,
            "matched_multiomic_constraint_fraction": float(np.nanmean(tab["matched_multiomic_constraint_fraction"])),
            "n_cells": int(len(tab)),
            "n_model_leiden": n_cluster,
            "dependency_support_mean": float(np.nanmean(tab["dependency_support_score_global"])),
            "re_traceability_mean": float(np.nanmean(tab["re_traceability_score_global"])),
            "dependency_core_mean": float(np.nanmean(tab["dependency_core_score_global"])),
            "response_anchor_mean": float(np.nanmean(tab["response_anchor_score_global"])),
            "dependency_support_unadjusted_mean": float(np.nanmean(tab["dependency_support_score_global_unadjusted"])),
            "re_traceability_unadjusted_mean": float(np.nanmean(tab["re_traceability_score_global_unadjusted"])),
            "dependency_core_unadjusted_mean": float(np.nanmean(tab["dependency_core_score_global_unadjusted"])),
            "response_anchor_unadjusted_mean": float(np.nanmean(tab["response_anchor_score_global_unadjusted"])),
        })
        out_tables[reg] = tab

    summary = pd.DataFrame(summary_rows)
    # Do not hard-code the ordering; report actual scores. A warning is printed if the expected order is not observed.
    if set(REGIME_ORDER).issubset(set(summary["data_regime"])):
        s = summary.set_index("data_regime")["DepStruct"].to_dict()
        if not (s.get("multiomic_only", -1) >= s.get("mixed", -1) >= s.get("scrna_only", -1)):
            log("[SCORE][WARN] DepStruct ordering is not multiomic_only >= mixed >= scrna_only for this dataset. Scores are not manually forced.", s)
    return out_tables, summary


def plot_single_benchmark_panel(ax, tab: pd.DataFrame, title: str, score: float | None = None):
    clusters = sorted(tab["new_leiden"].astype(str).unique().tolist(), key=_cluster_sort_key)
    for i, cl in enumerate(clusters):
        sub = tab[tab["new_leiden"].astype(str) == str(cl)]
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=PLOT_DOT_SIZE,
            alpha=PLOT_DOT_ALPHA,
            color=BENCHMARK_CLUSTER_PALETTE[i % len(BENCHMARK_CLUSTER_PALETTE)],
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    if score is None or not np.isfinite(score):
        subtitle = "DepStruct = NA"
    else:
        subtitle = f"DepStruct = {score:.3f}"
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")



def plot_cluster_color_legend_figure(all_regime_tables: dict[str, dict[str, pd.DataFrame]]):
    nrow = len(DATASETS_TO_RUN)
    ncol = len(REGIME_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), facecolor="white", squeeze=False)
    map_rows = []
    for i, dataset_name in enumerate(DATASETS_TO_RUN):
        for j, reg in enumerate(REGIME_ORDER):
            ax = axes[i, j]
            ax.axis("off")
            tab = all_regime_tables.get(dataset_name, {}).get(reg, None)
            if tab is None or len(tab) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            clusters = sorted(tab["new_leiden"].astype(str).unique().tolist(), key=_cluster_sort_key)
            counts = tab["new_leiden"].astype(str).value_counts().to_dict()
            n = len(clusters)
            ncols_inner = 2 if n > 18 else 1
            items_per_col = int(np.ceil(n / ncols_inner))
            ax.set_title(f"{dataset_name} | {REGIME_LABEL.get(reg, reg)}", fontsize=9, fontweight="bold")
            for idx, cl in enumerate(clusters):
                col = idx // items_per_col
                row = idx % items_per_col
                x = 0.05 + col * 0.48
                y = 0.95 - row * (0.86 / max(items_per_col, 1))
                color = BENCHMARK_CLUSTER_PALETTE[idx % len(BENCHMARK_CLUSTER_PALETTE)]
                ax.scatter([x], [y], s=38, color=color, edgecolor="none")
                ax.text(x + 0.03, y, f"Cluster {cl} (n={int(counts.get(str(cl), 0))})", va="center", fontsize=7)
                map_rows.append({"dataset": dataset_name, "data_regime": reg, "new_leiden": str(cl), "n_cells": int(counts.get(str(cl), 0)), "color_hex": color})
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    fig.suptitle("Cluster-color mapping for benchmark-style UMAP panels", fontsize=14, fontweight="bold", y=1.01)
    savefig_multi(fig, OUT_ROOT / "dependency_structure_cluster_color_legend")
    safe_to_csv(pd.DataFrame(map_rows), OUT_ROOT / "dependency_structure_cluster_color_mapping.csv")

def plot_benchmark_umap_figure(all_regime_tables: dict[str, dict[str, pd.DataFrame]], score_summary: pd.DataFrame):
    datasets = [d for d in DATASETS_TO_RUN if d in all_regime_tables]
    n_rows = len(datasets)
    n_cols = len(REGIME_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.65 * n_cols, 3.45 * n_rows),
        facecolor="white",
        squeeze=False,
    )

    score_lookup = {}
    if score_summary is not None and len(score_summary):
        for _, r in score_summary.iterrows():
            score_lookup[(str(r["dataset"]), str(r["data_regime"]))] = float(r["DepStruct"])

    for i, ds in enumerate(datasets):
        for j, reg in enumerate(REGIME_ORDER):
            ax = axes[i, j]
            tab = all_regime_tables.get(ds, {}).get(reg)
            if tab is None or len(tab) == 0:
                ax.text(0.5, 0.5, "Missing\ncheckpoint", ha="center", va="center", transform=ax.transAxes, color="#777777")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            title = REGIME_LABEL.get(reg, reg)
            score = score_lookup.get((ds, reg), np.nan)
            plot_single_benchmark_panel(ax, tab, title, score)
            if j == 0:
                ax.text(
                    -0.12,
                    0.5,
                    ds,
                    rotation=90,
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                )

    fig.suptitle(
        "Benchmark-style UMAP of IMAS dependent-structure manifolds",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Each panel is re-embedded separately from the regime-specific adapted model cell embeddings. Colors denote newly computed Leiden clusters; DepStruct summarizes RNA–TF–RE/core/response traceability, without using CCI/LTC as the backbone.",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.07, wspace=0.05, hspace=0.22)
    save_figure(fig, OUT_ROOT / "dependency_structure_benchmark_umap")


def plot_score_strip(score_summary: pd.DataFrame):
    """A tiny companion strip, not a main bar plot; it helps inspect the title values."""
    if score_summary is None or len(score_summary) == 0:
        return
    datasets = [d for d in DATASETS_TO_RUN if d in set(score_summary["dataset"].astype(str))]
    fig, ax = plt.subplots(figsize=(6.8, 1.6 + 0.35 * len(datasets)), facecolor="white")
    y_positions = {ds: k for k, ds in enumerate(datasets)}
    colors = {"multiomic_only": "#2B6CB0", "mixed": "#8A6FB0", "scrna_only": "#9E9E9E"}
    offsets = {"multiomic_only": 0.18, "mixed": 0.0, "scrna_only": -0.18}
    for _, r in score_summary.iterrows():
        ds = str(r["dataset"]); reg = str(r["data_regime"])
        y = y_positions.get(ds, 0) + offsets.get(reg, 0)
        ax.scatter(float(r["DepStruct"]), y, s=70, color=colors.get(reg, "#555555"), edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(float(r["DepStruct"]) + 0.01, y, REGIME_LABEL.get(reg, reg), va="center", fontsize=7)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(datasets, fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("DepStruct score", fontsize=8)
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.8)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.set_title("Panel title scores", fontsize=10, fontweight="bold")
    save_figure(fig, OUT_ROOT / "dependency_structure_benchmark_score_strip")



def expected_existing_paths(dataset_name: str) -> dict[str, Path]:
    """Paths that prove a dataset has already completed posthoc scoring."""
    out = {"summary": OUT_ROOT / f"{dataset_name}_dependency_structure_benchmark_scores.csv"}
    for reg in REGIME_ORDER:
        out[f"cell_{reg}"] = OUT_ROOT / f"{dataset_name}_{reg}_cell_structure_scores_with_DepStruct.csv"
    return out


def dataset_outputs_complete(dataset_name: str) -> bool:
    paths = expected_existing_paths(dataset_name)
    return all(p.exists() for p in paths.values())


def load_existing_dataset_outputs(dataset_name: str) -> tuple[Optional[dict[str, pd.DataFrame]], Optional[pd.DataFrame]]:
    """
    Load previously computed posthoc outputs so already-finished CRC/Glioma runs are
    not recomputed. These loaded tables are only used for the final combined plots and
    aggregate CSV, preserving the same visualization/statistics method.
    """
    paths = expected_existing_paths(dataset_name)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        log(f"[REUSE][WARN] cannot load existing outputs for {dataset_name}; missing:")
        for m in missing:
            log("  -", m)
        return None, None

    regime_tables: dict[str, pd.DataFrame] = {}
    for reg in REGIME_ORDER:
        tab = pd.read_csv(paths[f"cell_{reg}"])
        # Be defensive about types used by plotting/scoring lookups.
        if "dataset" not in tab.columns:
            tab["dataset"] = dataset_name
        tab["dataset"] = tab["dataset"].astype(str)
        tab["data_regime"] = str(reg)
        regime_tables[reg] = tab

    summary = pd.read_csv(paths["summary"])
    if "dataset" in summary.columns:
        summary["dataset"] = summary["dataset"].astype(str)
    if "data_regime" in summary.columns:
        summary["data_regime"] = summary["data_regime"].astype(str)
    log(f"[REUSE] loaded existing posthoc outputs for {dataset_name}")
    return regime_tables, summary


def compute_dataset_from_checkpoints(
    dataset_name: str,
    genes_vocab: list[str],
    tfs_vocab: list[str],
    all_failed_rows: list[dict],
) -> tuple[Optional[dict[str, pd.DataFrame]], Optional[pd.DataFrame]]:
    """Run the original per-dataset calculation path unchanged."""
    selected_sids = SAMPLES_BY_DATASET.get(dataset_name, None)
    samples, sid2sample = load_samples_for_dataset(dataset_name, genes_vocab, selected_sids)
    dataset_counts_adata = build_true_counts_anndata(dataset_name, samples)
    cell_table = build_cell_table_for_dataset(dataset_name, samples)
    safe_to_csv(cell_table, OUT_ROOT / f"{dataset_name}_selected_cells_for_benchmark.csv")

    model = None
    try:
        model, info = load_base_model_and_priors(dataset_name, genes_vocab, tfs_vocab)
        log("[BASE/PRIOR]", {
            "N_GENE_DIM": info["N_GENE_DIM"],
            "N_TF_DIM": info["N_TF_DIM"],
            "PEAK_HASH_SIZE": info["PEAK_HASH_SIZE"],
            "n_tf_in_prior": len(info["tf_in_prior"]),
            "n_gene_union": len(info["genes_union"]),
            "default_cancer_id": info["DEFAULT_CANCER_ID"],
        })

        regime_score_tables: dict[str, pd.DataFrame] = {}
        cluster_tables = []

        for regime in REGIME_ORDER:
            try:
                cell_df, cluster_df = compute_structure_scores_for_regime(
                    dataset_name=dataset_name,
                    regime=regime,
                    umap_df=cell_table,
                    samples=samples,
                    sid2sample=sid2sample,
                    model=model,
                    info=info,
                    genes_vocab=genes_vocab,
                    tfs_vocab=tfs_vocab,
                )
                regime_score_tables[regime] = cell_df
                cluster_tables.append(cluster_df)
            except Exception as e:
                log(f"[WARN] failed {dataset_name}/{regime}: {repr(e)}")
                all_failed_rows.append({"dataset": dataset_name, "data_regime": regime, "error": repr(e)})

        if len(regime_score_tables) == 0:
            log(f"[WARN] no successful regime for {dataset_name}; skip plotting this dataset.")
            all_failed_rows.append({
                "dataset": dataset_name,
                "data_regime": "ALL",
                "error": "No successful regime; check frontier_best.pt and clusters.csv for all regimes.",
            })
            return None, None

        regime_score_tables, score_summary = add_global_depstruct_scores(regime_score_tables)

        # Save updated cell tables with global DepStruct columns.
        for regime, tab in regime_score_tables.items():
            safe_to_csv(tab, OUT_ROOT / f"{dataset_name}_{regime}_cell_structure_scores_with_DepStruct.csv")
            try:
                compute_cluster_markers_for_regime(dataset_name, regime, tab, dataset_counts_adata)
            except Exception as e:
                log(f"[MARKERS][WARN] failed {dataset_name}/{regime}: {repr(e)}")

        if cluster_tables:
            safe_to_csv(
                pd.concat(cluster_tables, axis=0, ignore_index=True),
                OUT_ROOT / f"{dataset_name}_all_regime_cluster_structure_scores.csv",
            )
        safe_to_csv(score_summary, OUT_ROOT / f"{dataset_name}_dependency_structure_benchmark_scores.csv")
        return regime_score_tables, score_summary

    finally:
        try:
            del model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# ============================================================
# 8) MAIN
# ============================================================
def main():
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing {GLOBAL_TF_VOCAB}"
    assert GLOBAL_PEAK_VOCAB.exists(), f"Missing {GLOBAL_PEAK_VOCAB}"
    assert PRIORS_DIR.exists(), f"Missing {PRIORS_DIR}"
    assert RUN_DIR.exists(), f"Missing RUN_DIR: {RUN_DIR}"

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    log("[OUT_ROOT]", OUT_ROOT.resolve())
    log("[DATASETS_TO_RUN_RAW]", DATASETS_TO_RUN_RAW)
    log("[DATASETS_TO_RUN_RESOLVED]", DATASETS_TO_RUN)
    log("[DATASETS_TO_CALCULATE_RAW]", DATASETS_TO_CALCULATE_RAW)
    log("[DATASETS_TO_CALCULATE_RESOLVED]", sorted(DATASETS_TO_CALCULATE))
    log("[REUSE_EXISTING_DATASET_RESULTS]", REUSE_EXISTING_DATASET_RESULTS)
    log("[REGIME_ORDER]", REGIME_ORDER)
    log("[BENCHMARK UMAP PARAMS]", {
        "N_PCS": N_PCS,
        "N_NEIGHBORS": N_NEIGHBORS,
        "UMAP_MIN_DIST": UMAP_MIN_DIST,
        "UMAP_SPREAD": UMAP_SPREAD,
        "LEIDEN_RESOLUTION": LEIDEN_RESOLUTION,
        "SEED": SEED,
    })

    all_regime_tables: dict[str, dict[str, pd.DataFrame]] = {}
    all_score_summaries = []
    all_failed_rows = []

    for dataset_name in DATASETS_TO_RUN:
        log("\n" + "=" * 80)
        log("[DATASET]", dataset_name)

        try:
            should_calculate = dataset_name in DATASETS_TO_CALCULATE

            # CRC/Glioma path: do not recompute; read existing posthoc outputs.
            if not should_calculate:
                regime_tables, score_summary = load_existing_dataset_outputs(dataset_name)
                if regime_tables is not None and score_summary is not None:
                    all_regime_tables[dataset_name] = regime_tables
                    all_score_summaries.append(score_summary)
                else:
                    all_failed_rows.append({
                        "dataset": dataset_name,
                        "data_regime": "REUSE_EXISTING",
                        "error": "Existing posthoc outputs were not complete; dataset was intentionally not recomputed.",
                    })
                continue

            # Osteosarcoma path: compute unless all outputs already exist and reuse is enabled.
            if REUSE_EXISTING_DATASET_RESULTS and dataset_outputs_complete(dataset_name):
                log(f"[SKIP] {dataset_name}: existing complete outputs found; reusing instead of recomputing.")
                regime_tables, score_summary = load_existing_dataset_outputs(dataset_name)
            else:
                log(f"[CALCULATE] {dataset_name}: running original posthoc computation/statistics/plots inputs.")
                regime_tables, score_summary = compute_dataset_from_checkpoints(
                    dataset_name=dataset_name,
                    genes_vocab=genes_vocab,
                    tfs_vocab=tfs_vocab,
                    all_failed_rows=all_failed_rows,
                )

            if regime_tables is not None and score_summary is not None:
                all_regime_tables[dataset_name] = regime_tables
                all_score_summaries.append(score_summary)

        except Exception as e:
            log(f"[DATASET][WARN] failed setup/base/calculation for {dataset_name}: {repr(e)}")
            all_failed_rows.append({
                "dataset": dataset_name,
                "data_regime": "DATASET_SETUP_OR_CALCULATION",
                "error": repr(e),
            })
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if all_score_summaries:
        score_all = pd.concat(all_score_summaries, axis=0, ignore_index=True)
        safe_to_csv(score_all, OUT_ROOT / "dependency_structure_benchmark_scores.csv")
        plot_benchmark_umap_figure(all_regime_tables, score_all)
        plot_cluster_color_legend_figure(all_regime_tables)
        plot_score_strip(score_all)

    if all_failed_rows:
        safe_to_csv(pd.DataFrame(all_failed_rows), OUT_ROOT / "failed_benchmark_umap_runs.csv")

    log("\n[DONE] Benchmark-style UMAP dependent-structure visualization finished.")
    log("[OUT_ROOT]", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
