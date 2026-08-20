from __future__ import annotations
from imas.config import get_checkpoint_root, get_project_root


# ============================================================
# SOD2-SPECIFIC COUNTERFACTUAL REVISION
# ============================================================
#
# This standalone version retains the model/data loader from the user's
# original counterfactual analysis, but replaces the generic cross-TF
# Figure 9 workflow with a single target-gene case:
#
#   Target: SOD2
#
# For dependency-sensitive SOD2 hotspot cells:
#
#   1. rank cell-specific RNA support genes for the adapted SOD2 target score;
#   2. sequentially remove the highest-ranked support genes;
#   3. compare the resulting SOD2 collapse with abundance- and
#      state-matched random gene deletion in the same cells;
#   4. calculate a flip-independent fragility-core score;
#   5. record the actual top SOD2-associated TF before and after deletion.
#
# Main figure:
#
#   A. SOD2 support-retention deletion curves
#   B. matched-null distribution of mean SOD2 fragility
#   C. actual SOD2-associated TF-rank transitions
#
# The null comparison preserves:
#
#   - the same cells;
#   - the same number of deleted genes;
#   - approximately matched projected-RNA abundance;
#   - approximately matched baseline cell-state support.
#
# SOD2 itself is excluded from both rationale and null deletion sets.
# ============================================================


# ============================================================
# frontier_explain_fig9_matrix_atlas_v1_linux.py
# Standalone explainability plotting script for:
#   frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection.py
#
# Re-designed Figure A:
#   Matrix-atlas counterfactual deletion overview
#
# Generates:
#   fig9_matrix_atlas.png
#   fig9_matrix_atlas.svg
#   fig9_matrix_atlas_cases.csv
#   run_summary_fig9_matrix_atlas.json
# ============================================================

import math
import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp  # kept for compatibility
import scipy.ndimage as ndi
import scipy.stats as st
import anndata as ad
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D


# ============================================================
# 0) PATHS / CONFIG
# ============================================================
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"

GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_PEAK_VOCAB = MAPS_ROOT / "global_peak_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"

RUN_DIR = BASE / "runs" / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
DATASET_NAME = "CRC-Tissue"
DATASET_DIR = RUN_DIR / DATASET_NAME

BASE_CKPT = (
    get_checkpoint_root()
    / "qc_lodo_planC_v7_global_hashpeak"
    / "qc_lodo_planC_v7_global_hashpeak"
    / "CRC-Tissue"
    / "ckpt_last_BASE_global_hashpeak.pt"
)
FRONTIER_CKPT = DATASET_DIR / "frontier_best.pt"

FIG2A_DIR = DATASET_DIR / "fig2a_mltd_support_umap"
OUT_DIR = FIG2A_DIR / "SOD2_counterfactual_support_deletion_matched_null"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CANCER_NAME = None
VAL_FRAC_FROM_TRAIN = 0.10
ANCHOR_DTYPE = np.float16
RNA_TOPK = 2048
EMB_BATCH = 256
STORE_TOP_G = 512
AUCELL_TOPK = 500
TF_MIX_ALPHA = 0.6

POS_Q = 0.80
NEG_Q = 0.20

GROUP_SCALE_ALPHA = 0.12

USE_GROUP_SCALE_BIAS = True
USE_CANCER_ADAPTOR = True
USE_CELLGROUP_ADAPTOR = True
USE_CLUSTER_CORRECTION = True
CLUSTER_CORRECTION_ALPHA_TF = 0.25
CLUSTER_CORRECTION_ALPHA_RNA = 0.25
CLUSTER_CORRECTION_ALPHA_RE = 0.25
CLUSTER_CORRECTION_HID = 256

COUNTERFACTUAL_MAX_REMOVE = 12
COUNTERFACTUAL_TARGET_DROP = 0.35

CIRCULAR_OVERVIEW_MAX_CASES = 128
CIRCULAR_TRAJ_BINS = 8
CIRCULAR_USE_ALL_TEST = True

# --------------------------
# informative TF-cell pair selection
# --------------------------
# Instead of using only the top-1 TF per cell, evaluate several high-scoring TFs
# and select TF-cell pairs with informative counterfactual deletion trajectories.
USE_INFORMATIVE_TF_SELECTION = False
PRESELECT_TF_PER_CELL = 12           # cheap stage: from full cell x TF score matrix
RUN_DELETE_TF_PER_CELL = 12          # expensive stage: run deletion for all preselected TFs per cell
MAX_COUNTERFACTUAL_CELLS_FOR_TF_SCAN = 64
MAX_COUNTERFACTUAL_PAIRS_FOR_DELETE = 768
MIN_BASELINE_TF_SCORE_QUANTILE = 0.40
MIN_DISPLAY_MAX_DROP = 0.05
MIN_DISPLAY_FINAL_DROP = 0.03

# More discriminative fragility score: emphasize early collapse.
EARLY_DROP_K = 3
FRAG_W_EARLY_DROP = 0.45
FRAG_W_AUC_LOSS = 0.25
FRAG_W_MAX_DROP = 0.15
FRAG_W_FINAL_LOSS = 0.10
FRAG_W_EARLY_EVENT = 0.05
FRAG_FLIP_BONUS = 0.08

# --------------------------
# figure reclustering config
# --------------------------
RECLUSTER_FOR_FIG = True
FIG_RECLUSTER_SAMPLE_ID = "CRC_RNA_TO009"
FIG_FILTER_TO_RECLUSTER_SAMPLE = True
RECLUSTER_MAX_CELLS = 12000
N_TOP_HVG = 2000
N_PCS = 20
N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.03
UMAP_SPREAD = 2.0
LEIDEN_RESOLUTION = 0.5

# --------------------------
# heatmap rendering config
# --------------------------
# "relative_score" plots row-scaled relative TF score trajectory.
# This gives red-white-blue transition like the reference figure while retaining part of global scale.
HEATMAP_VALUE_MODE = "score_envelope"
HEATMAP_DROP_QMAX = 0.98
HEATMAP_DROP_MIN_VMAX = 0.20
HEATMAP_DROP_MAX_VMAX = 0.80
HEATMAP_ROW_SCALE_ALPHA = 0.75  # unused for score_envelope

# Prefer cases whose deletion curves are visually informative.
# This helps the heatmap show red/orange -> white -> blue transition instead of mostly stable flat rows.
PREFER_INFORMATIVE_CASES = True
INFORMATIVE_MIN_DROP = 0.03
INFORMATIVE_STABLE_FRACTION = 0.20

# Visualization-only enhancement. These do not change statistics or exported values.
HEATMAP_VISUAL_TRANSFORM = "linear"  # keep raw heatmap values; no visual nonlinear enhancement
HEATMAP_INTERPOLATE_X = 1            # no interpolation; keep discrete deletion steps
HEATMAP_SMOOTH_SIGMA_X = 0.0         # no smoothing along deletion axis
HEATMAP_SMOOTH_SIGMA_Y = 0.0         # no smoothing across rows
SORT_WITHIN_CLUSTER_BY = "trajectory_auc"  # options: "trajectory_auc", "k_event", "max_drop"


# ============================================================
# 1) HELPERS
# ============================================================
def log(*args):
    print(*args, flush=True)


def read_vocab_txt(p: Path):
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
    return f"{sid}::{barcode}"


def split_5050(key: str) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 1


def dataset_to_cancer_type(ds_name: str) -> str:
    s = str(ds_name)
    return s.split("-", 1)[0] if "-" in s else s


def get_default_cancer_name(dataset_name: str) -> str | None:
    if DEFAULT_CANCER_NAME is not None:
        return DEFAULT_CANCER_NAME
    return dataset_to_cancer_type(dataset_name)


def find_prior_file(prior_dir: Path, include_keys: list[str]):
    cands = sorted(prior_dir.glob("*.npz"))
    hits = []
    for p in cands:
        n = p.name.lower()
        ok = all(k in n for k in include_keys)
        if ok:
            hits.append(p)
    return hits[0] if hits else None


# ============================================================
# 2) MODEL DEFINITIONS
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
    def __init__(self, d: int, k: int, n_tf: int, n_gene: int, peak_hash_size: int, n_cancer: int, n_group: int, e_dim: int = 64, hid: int = 256):
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


class SelectedPeakResidual(nn.Module):
    def __init__(self, selected_peak_raw_ids: np.ndarray, d: int):
        super().__init__()
        ids = np.asarray(selected_peak_raw_ids, dtype=np.int64)
        self.d = int(d)
        self.register_buffer("selected_peak_raw_ids", torch.from_numpy(ids), persistent=True)
        self.emb = nn.Embedding(max(1, ids.size), self.d)
        nn.init.zeros_(self.emb.weight)
        self.raw2local = {int(pk): i for i, pk in enumerate(ids.tolist())}

    @property
    def n_selected(self) -> int:
        return int(self.selected_peak_raw_ids.numel())

    def lookup(self, peak_idx_raw: torch.Tensor) -> torch.Tensor:
        if self.n_selected <= 0 or peak_idx_raw.numel() == 0:
            return torch.zeros((peak_idx_raw.numel(), self.d), device=peak_idx_raw.device, dtype=torch.float32)
        raw_np = peak_idx_raw.detach().cpu().numpy().astype(np.int64)
        local_np = np.full((raw_np.size,), -1, dtype=np.int64)
        for i, pk in enumerate(raw_np.tolist()):
            local_np[i] = self.raw2local.get(int(pk), -1)
        out = torch.zeros((raw_np.size, self.d), device=peak_idx_raw.device, dtype=self.emb.weight.dtype)
        m = local_np >= 0
        if m.any():
            pos_t = torch.from_numpy(np.where(m)[0]).to(peak_idx_raw.device)
            idx_t = torch.from_numpy(local_np[m]).to(peak_idx_raw.device)
            out[pos_t] = self.emb(idx_t)
        return out


class ClusterCellCorrectionHead(nn.Module):
    def __init__(self, z_dim: int, summary_dim: int, hid: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + summary_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, z_dim),
        )

    def forward(self, z_cell: torch.Tensor, cluster_summary: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_cell, cluster_summary], dim=-1)
        return self.mlp(x)


class ClusterSummaryTable(nn.Module):
    def __init__(self, cluster_summary_np: dict[int, np.ndarray]):
        super().__init__()
        keys = sorted(cluster_summary_np.keys())
        arr = np.stack([cluster_summary_np[k] for k in keys], axis=0).astype(np.float32)
        self.register_buffer("summary_table", torch.from_numpy(arr), persistent=True)
        self.cid2row = {int(cid): i for i, cid in enumerate(keys)}

    @property
    def summary_dim(self) -> int:
        return int(self.summary_table.shape[1])

    def lookup(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        idx = []
        for cid in cluster_ids.detach().cpu().numpy().tolist():
            idx.append(self.cid2row.get(int(cid), 0))
        idx_t = torch.tensor(idx, device=cluster_ids.device, dtype=torch.long)
        return self.summary_table[idx_t]


# ============================================================
# 3) DATA
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
    anchor_gene2col: dict[str, int]
    anchor_cells: np.ndarray
    anchor_X: np.ndarray
    rid2arow: np.ndarray
    depth_rna: np.ndarray | None = None


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


def get_anchor_dir(dataset_name: str) -> Path:
    return WORLD_ROOT / dataset_name / "processed" / "ATAC_geneActivity_ALLGENES"


def resolve_anchor_path(anchor_dir: Path, sid: str) -> Path:
    candidates = [anchor_dir / f"{sid}_ATAC_geneActivity_ALLGENES.npz"]
    sid_alt = None
    if "_RNA_" in sid:
        sid_alt = sid.replace("_RNA_", "_ATAC_")
    elif "_ATAC_" in sid:
        sid_alt = sid.replace("_ATAC_", "_RNA_")

    if sid_alt is not None and sid_alt != sid:
        candidates.append(anchor_dir / f"{sid_alt}_ATAC_geneActivity_ALLGENES.npz")

    for p in candidates:
        if p.exists():
            return p

    raise AssertionError(
        f"Missing anchor for sid={sid}. Tried:\n" +
        "\n".join(str(x) for x in candidates)
    )


def load_anchor_all(anchor_file: Path):
    assert anchor_file.exists(), f"Missing anchor: {anchor_file}"
    z = np.load(str(anchor_file), allow_pickle=True)
    if "genes" in z:
        genes = z["genes"].astype(object)
    elif "var_names" in z:
        genes = z["var_names"].astype(object)
    else:
        raise KeyError(f"{anchor_file.name}: cannot find genes key")
    gene2col = {str(g): i for i, g in enumerate(genes.tolist())}
    if "cells" in z:
        cells_raw = z["cells"].astype(object)
    elif "barcodes" in z:
        cells_raw = z["barcodes"].astype(object)
    else:
        raise KeyError(f"{anchor_file.name}: cannot find cells key")
    cells = np.asarray([norm_barcode(c) for c in cells_raw.tolist()], dtype=object)
    X = None
    for k in ["X_geneActivity", "X_geneactivity", "X_atac", "X", "X_ga"]:
        if k in z:
            X = z[k]
            break
    if X is None:
        raise KeyError(f"{anchor_file.name}: cannot find geneActivity matrix key")
    X = np.asarray(X)
    if X.dtype != ANCHOR_DTYPE:
        X = X.astype(ANCHOR_DTYPE, copy=False)
    return gene2col, cells, X


def load_samples(dataset_name: str, genes_vocab: list[str], only_sample_id: str | None = None):
    manifest_rna_dir = get_manifest_rna_dir(dataset_name)
    anchor_dir = get_anchor_dir(dataset_name)
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"

    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    assert anchor_dir.exists(), f"Missing anchor dir: {anchor_dir}"

    df = pd.read_csv(manifest_path, sep="	")
    need_cols = ["cell_id", "sample_id", "row_index", "counts_npy"]
    for c in need_cols:
        assert c in df.columns, f"{manifest_path.name}: missing column {c}"

    gene2i_vocab = {g: i for i, g in enumerate(genes_vocab)}
    samples = []
    sid2sample = {}

    if only_sample_id is not None:
        df = df[df["sample_id"].astype(str) == str(only_sample_id)].copy()
        assert len(df) > 0, f"No rows found for only_sample_id={only_sample_id} in {manifest_path}"

    for sid, g in df.groupby("sample_id", sort=True):
        sid = str(sid).strip()
        g = g.sort_values("row_index").reset_index(drop=True)

        meta_path = manifest_rna_dir / f"{sid}.rna_meta.npz"
        counts_path = manifest_rna_dir / f"{sid}.X_rna_counts_dense.npy"
        proj_rna_path = PROJECTED_ROOT / DATASET_NAME / f"{sid}.rna.global.csr.npz"
        anchor_path = resolve_anchor_path(anchor_dir, sid)

        assert meta_path.exists(), f"Missing meta: {meta_path}"
        assert counts_path.exists(), f"Missing counts: {counts_path}"
        assert proj_rna_path.exists(), f"Missing projected RNA: {proj_rna_path}"
        assert anchor_path.exists(), f"Missing anchor: {anchor_path}"

        z = np.load(meta_path, allow_pickle=True)
        assert "gene_names" in z, f"{meta_path.name}: missing gene_names"
        gene_names = np.asarray(z["gene_names"]).astype(str)

        cells = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
        row_indices = g["row_index"].astype(np.int64).values
        counts_mm = np.load(counts_path, mmap_mode="r")
        assert counts_mm.ndim == 2
        assert counts_mm.shape[0] == len(row_indices)
        assert counts_mm.shape[1] == len(gene_names)

        depth_rna = None
        if "depth_rna" in z:
            depth_rna = np.asarray(z["depth_rna"], dtype=np.float32)

        ext2train = np.full((len(gene_names),), -1, dtype=np.int64)
        train2rna = {}
        for j, gname in enumerate(gene_names):
            gname = normalize_gene_name(gname)
            if gname in gene2i_vocab:
                tid = gene2i_vocab[gname]
                ext2train[j] = tid
                if tid not in train2rna:
                    train2rna[tid] = j
        matched = int((ext2train >= 0).sum())
        assert matched > 1000, f"{DATASET_NAME}/{sid}: too few genes matched to global vocab: {matched}"

        anchor_gene2col, anchor_cells, anchor_X = load_anchor_all(anchor_path)
        m = {str(c): i for i, c in enumerate(anchor_cells.tolist())}
        rid2arow = np.full((len(cells),), -1, dtype=np.int64)
        for i, bc in enumerate(cells.tolist()):
            rid2arow[i] = int(m.get(str(bc), -1))

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
            anchor_gene2col=anchor_gene2col,
            anchor_cells=anchor_cells,
            anchor_X=anchor_X,
            rid2arow=rid2arow,
            depth_rna=depth_rna,
        )
        samples.append(s)
        sid2sample[sid] = s
        log(f"[LOAD] {DATASET_NAME}/{sid} | cells={s.counts_shape[0]} genes={s.counts_shape[1]} matched_genes={matched}")

    return samples, sid2sample


def build_full_cell_lists(samples: list[Sample], limit_per_sample=None):
    all_cells = []
    for s in samples:
        n = int(s.counts_shape[0])
        idx = np.arange(n, dtype=np.int64)
        if limit_per_sample is not None and n > int(limit_per_sample):
            idx = np.random.choice(idx, size=int(limit_per_sample), replace=False)
            idx = np.sort(idx)
        for rid in idx.tolist():
            if int(s.rid2arow[int(rid)]) < 0:
                continue
            key = make_cell_key(s.sid, str(s.cells[int(rid)]))
            all_cells.append((s.sid, int(rid), key))
    return all_cells


def split_cells_5050(all_cells):
    train = []
    test = []
    for sid, rid, key in all_cells:
        if split_5050(key) == 0:
            train.append((sid, rid))
        else:
            test.append((sid, rid))
    return train, test


def split_train_val(train_cells: list[tuple[str, int]], val_frac: float):
    n = len(train_cells)
    if n < 200:
        return train_cells, []
    idx = np.arange(n)
    np.random.shuffle(idx)
    nval = max(1, int(n * float(val_frac)))
    val = [train_cells[i] for i in idx[:nval]]
    tr = [train_cells[i] for i in idx[nval:]]
    return tr, val


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
        gi, gv = proj_store_cache[key].get_row_topk(rid, RNA_TOPK)
        idx_list.append(gi.astype(np.int64))
        val_list.append(gv.astype(np.float32))
        offsets.append(offsets[-1] + gi.size)
    idx_cat = np.concatenate(idx_list) if idx_list else np.zeros((0,), np.int64)
    val_cat = np.concatenate(val_list) if val_list else np.zeros((0,), np.float32)
    offsets = np.asarray(offsets[:-1], np.int64)
    cancer_ids = np.full((len(cell_list),), int(cancer_id_int), dtype=np.int64)
    return idx_cat, val_cat, offsets, cancer_ids


# ============================================================
# 4) EXPLAINER
# ============================================================
class FrontierExplainerStandalone:
    def __init__(self):
        assert BASE_CKPT.exists(), f"Missing BASE_CKPT: {BASE_CKPT}"
        assert FRONTIER_CKPT.exists(), f"Missing FRONTIER_CKPT: {FRONTIER_CKPT}"
        assert GLOBAL_GENE_VOCAB.exists(), f"Missing {GLOBAL_GENE_VOCAB}"
        assert GLOBAL_TF_VOCAB.exists(), f"Missing {GLOBAL_TF_VOCAB}"
        assert GLOBAL_PEAK_VOCAB.exists(), f"Missing {GLOBAL_PEAK_VOCAB}"

        self.genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
        self.tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)
        self.peaks_vocab = read_vocab_txt(GLOBAL_PEAK_VOCAB)

        self.base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
        self.frontier_ckpt = torch.load(FRONTIER_CKPT, map_location="cpu", weights_only=False)

        self.cfg = self._build_cfg()
        self.N_GENE_DIM = int(self.cfg["N_GENE_DIM"])
        self.N_TF_DIM = int(self.cfg["N_TF_DIM"])
        self.PEAK_HASH_SIZE = int(self.cfg["PEAK_HASH_SIZE"])
        self.D = int(self.cfg["D"])
        self.HID = int(self.cfg["HID"])
        self.DROP = float(self.cfg["DROP"])
        self.cancer_types = list(self.cfg["cancer_types"])

        self.cancer2id = {c: i for i, c in enumerate(self.cancer_types)}
        self.default_cancer_name = get_default_cancer_name(DATASET_NAME)
        self.default_cancer_id = int(self.cancer2id.get(self.default_cancer_name, 0))
        self.cid = torch.tensor([self.default_cancer_id], device=DEVICE, dtype=torch.long)

        self.base_model = PlanC_DirAware_Cancer_HashPeak(
            n_gene=self.N_GENE_DIM,
            n_tf=self.N_TF_DIM,
            n_cancer=len(self.cancer_types),
            peak_hash_size=self.PEAK_HASH_SIZE,
            d=self.D, hid=self.HID, drop=self.DROP,
        )
        self.base_model.load_state_dict(self.base_ckpt["model"], strict=True)
        self.base_model.to(DEVICE).eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self._load_priors()
        self._load_data()
        self._load_frontier_modules()

    def _build_cfg(self):
        cfg = self.base_ckpt.get("cfg", None)
        meta = self.base_ckpt.get("meta", None)
        state_dict = self.base_ckpt.get("model", None)
        assert state_dict is not None

        if cfg is None:
            cfg = {}

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
            n_cancer = int(state_dict["cell_c.weight"].shape[0])
            c0 = get_default_cancer_name(DATASET_NAME)
            cfg["cancer_types"] = [str(c0)] + [f"unused_{i}" for i in range(1, n_cancer)]
        return cfg

    def _load_priors(self):
        tf_gene_prior = PRIORS_DIR / "tf_gene_f.cap256.npz"
        if not tf_gene_prior.exists():
            tf_gene_prior = PRIORS_DIR / "tf_gene_f.full.npz"
        assert tf_gene_prior.exists()

        z = np.load(tf_gene_prior, allow_pickle=True)
        tf_src = z["src"].astype(np.int64)
        g_dst = z["dst"].astype(np.int64)
        self.tf2targets = defaultdict(list)
        for s, d0 in zip(tf_src.tolist(), g_dst.tolist()):
            if 0 <= s < len(self.tfs_vocab) and 0 <= d0 < self.N_GENE_DIM:
                self.tf2targets[int(s)].append(int(d0))
        self.tf_in_prior = sorted(self.tf2targets.keys())
        self.tf_global2local = {int(tfid): i for i, tfid in enumerate(self.tf_in_prior)}

        gene_peak_prior = find_prior_file(PRIORS_DIR, ["gene", "peak"])
        assert gene_peak_prior is not None, f"Cannot find gene-peak prior in {PRIORS_DIR}"
        z2 = np.load(gene_peak_prior, allow_pickle=True)
        gp_src = z2["src"].astype(np.int64)
        gp_dst = z2["dst"].astype(np.int64)
        src_max = int(gp_src.max()) if gp_src.size else -1
        self.gene2peaks = defaultdict(list)
        if src_max < self.N_GENE_DIM:
            for g, p in zip(gp_src.tolist(), gp_dst.tolist()):
                if 0 <= g < self.N_GENE_DIM:
                    self.gene2peaks[int(g)].append(int(p))
        else:
            for p, g in zip(gp_src.tolist(), gp_dst.tolist()):
                if 0 <= g < self.N_GENE_DIM:
                    self.gene2peaks[int(g)].append(int(p))

        for g in list(self.gene2peaks.keys()):
            if len(self.gene2peaks[g]) > 64:
                self.gene2peaks[g] = self.gene2peaks[g][:64]

        genes_in_peakprior = sorted(self.gene2peaks.keys())
        genes_in_tfprior = sorted({g for t in self.tf_in_prior for g in self.tf2targets[t]})
        self.genes_union = sorted(set(genes_in_tfprior) | set(genes_in_peakprior))
        self.gid2pid_u = {int(gid): i for i, gid in enumerate(self.genes_union)}

    def _load_data(self):
        self.samples, self.sid2sample = load_samples(
            DATASET_NAME,
            self.genes_vocab,
            only_sample_id=FIG_RECLUSTER_SAMPLE_ID if FIG_FILTER_TO_RECLUSTER_SAMPLE else None,
        )
        self.all_cells = build_full_cell_lists(self.samples)
        tr_full, te = split_cells_5050(self.all_cells)
        self.tr, self.va = split_train_val(tr_full, VAL_FRAC_FROM_TRAIN)
        self.te = te

        self.proj_store_cache = {}
        self.cluster_df = pd.read_csv(DATASET_DIR / "clusters.csv")
        self.cluster_df["cell_key"] = self.cluster_df["cell_key"].astype(str)

        self.cellkey2meta = {}
        for _, r in self.cluster_df.iterrows():
            self.cellkey2meta[str(r["cell_key"])] = {
                "split": str(r["split"]),
                "sid": str(r["sid"]),
                "rid": int(r["rid"]),
                "barcode": str(r["barcode"]),
                "cluster_id": int(r["cluster_id"]),
                "cluster_label": str(r["cluster_label"]),
                "group_id": int(r["group_id"]),
            }

    def _load_frontier_modules(self):
        hyper = self.frontier_ckpt["hyper"]
        n_group = int(self.cluster_df["group_id"].max()) + 1

        self.adapter_after = HyperAdapterHashV2(
            d=self.D,
            k=int(hyper["K_RANK"]),
            n_tf=len(self.tf_in_prior),
            n_gene=len(self.genes_union),
            peak_hash_size=self.PEAK_HASH_SIZE,
            n_cancer=len(self.cancer_types),
            n_group=n_group,
            e_dim=64, hid=256,
        ).to(DEVICE)
        self.adapter_after.load_state_dict(self.frontier_ckpt["adapter_state_dict"], strict=True)
        self.adapter_after.eval()

        selected_peak_raw_ids = np.asarray(self.frontier_ckpt.get("selected_peak_raw_ids", []), dtype=np.int64)
        self.peak_res_after = SelectedPeakResidual(selected_peak_raw_ids, d=self.D).to(DEVICE)
        if self.frontier_ckpt.get("peak_residual_state_dict", None) is not None:
            self.peak_res_after.load_state_dict(self.frontier_ckpt["peak_residual_state_dict"], strict=True)
        self.peak_res_after.eval()

        cluster_summary_vecs = self.frontier_ckpt["cluster_summary_vecs"]
        cluster_summary_vecs = {int(k): np.asarray(v, dtype=np.float32) for k, v in cluster_summary_vecs.items()}
        self.cluster_summary_table = ClusterSummaryTable(cluster_summary_vecs).to(DEVICE)
        sumdim = self.cluster_summary_table.summary_dim

        self.tf_corr_after = ClusterCellCorrectionHead(self.D, sumdim, hid=CLUSTER_CORRECTION_HID).to(DEVICE)
        self.rna_corr_after = ClusterCellCorrectionHead(self.D, sumdim, hid=CLUSTER_CORRECTION_HID).to(DEVICE)
        self.re_corr_after = ClusterCellCorrectionHead(self.D, sumdim, hid=CLUSTER_CORRECTION_HID).to(DEVICE)

        if self.frontier_ckpt.get("tf_corr_head_state_dict", None) is not None:
            self.tf_corr_after.load_state_dict(self.frontier_ckpt["tf_corr_head_state_dict"], strict=True)
        if self.frontier_ckpt.get("rna_corr_head_state_dict", None) is not None:
            self.rna_corr_after.load_state_dict(self.frontier_ckpt["rna_corr_head_state_dict"], strict=True)
        if self.frontier_ckpt.get("re_corr_head_state_dict", None) is not None:
            self.re_corr_after.load_state_dict(self.frontier_ckpt["re_corr_head_state_dict"], strict=True)

        self.tf_corr_after.eval()
        self.rna_corr_after.eval()
        self.re_corr_after.eval()

    def get_cell_meta(self, sid: str, rid: int):
        s = self.sid2sample[sid]
        bc = str(s.cells[int(rid)])
        key = make_cell_key(sid, bc)
        return self.cellkey2meta[key]

    def get_cell_tokens(self, sid: str, rid: int):
        return build_tokens_for_cells([(sid, rid)], self.sid2sample, self.default_cancer_id, self.proj_store_cache)

    @torch.no_grad()
    def encode_cell_base(self, sid: str, rid: int):
        idx_cat, val_cat, offsets, cancer_ids = self.get_cell_tokens(sid, rid)
        z = self.base_model.encode_rna(
            torch.from_numpy(idx_cat).to(DEVICE),
            torch.from_numpy(val_cat).to(DEVICE),
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )[0]
        return z

    def tf_base(self, tf_local_pid_t: torch.Tensor) -> torch.Tensor:
        tf_ids = torch.tensor([self.tf_in_prior[i] for i in tf_local_pid_t.detach().cpu().numpy().tolist()], device=DEVICE, dtype=torch.long)
        return self.base_model.tf_in(tf_ids) + self.base_model.tf_in_c(self.cid).expand(tf_ids.numel(), -1)

    def gene_base_from_pid(self, gene_pid_t: torch.Tensor) -> torch.Tensor:
        gids = torch.tensor([self.genes_union[i] for i in gene_pid_t.detach().cpu().numpy().tolist()], device=DEVICE, dtype=torch.long)
        return self.base_model.gene_out(gids) + self.base_model.gene_out_c(self.cid).expand(gids.numel(), -1)

    def _module_pack(self, mode: str):
        if mode == "after":
            return {
                "adapter": self.adapter_after,
                "peak_res": self.peak_res_after,
                "tf_corr": self.tf_corr_after,
                "rna_corr": self.rna_corr_after,
                "re_corr": self.re_corr_after,
            }
        raise ValueError(mode)

    def cluster_ids_to_summary(self, cluster_ids_np: np.ndarray):
        cid_t = torch.from_numpy(cluster_ids_np.astype(np.int64)).to(DEVICE)
        return self.cluster_summary_table.lookup(cid_t)

    def apply_cluster_correction(self, z_in: torch.Tensor, cluster_ids_np: np.ndarray, head, alpha: float):
        if head is None:
            return z_in
        summary_t = self.cluster_ids_to_summary(cluster_ids_np)
        delta = head(z_in, summary_t)
        return F.normalize(z_in + float(alpha) * delta, dim=-1)

    @torch.no_grad()
    def adjust_Z(self, z0: torch.Tensor, sid: str, rid: int, mode: str, task: str = "tf"):
        pack = self._module_pack(mode)
        meta = self.get_cell_meta(sid, rid)
        group_id = np.asarray([meta["group_id"]], dtype=np.int64)
        cluster_id = np.asarray([meta["cluster_id"]], dtype=np.int64)
        cancer_id = np.asarray([self.default_cancer_id], dtype=np.int64)

        adapter = pack["adapter"]
        _, _, _, V_cancer = adapter()
        z = z0.view(1, -1)

        if USE_CELLGROUP_ADAPTOR:
            gid_t = torch.from_numpy(group_id).to(DEVICE)
            z = adapter.apply_group_to_Z(z, gid_t)

        if USE_CANCER_ADAPTOR:
            cid_t = torch.from_numpy(cancer_id).to(DEVICE)
            z = z + adapter.delta_cancer(cid_t, V_cancer)

        z = F.normalize(z, dim=-1)

        if task == "tf":
            z = self.apply_cluster_correction(z, cluster_id, pack["tf_corr"], CLUSTER_CORRECTION_ALPHA_TF)
        elif task == "rna":
            z = self.apply_cluster_correction(z, cluster_id, pack["rna_corr"], CLUSTER_CORRECTION_ALPHA_RNA)
        elif task == "re":
            z = self.apply_cluster_correction(z, cluster_id, pack["re_corr"], CLUSTER_CORRECTION_ALPHA_RE)

        return z[0]

    @torch.no_grad()
    def score_tf_vec(self, tf_local_pid_t: torch.Tensor, mode: str) -> torch.Tensor:
        pack = self._module_pack(mode)
        TF = self.tf_base(tf_local_pid_t)
        adapter = pack["adapter"]
        V_tf, _, _, V_cancer = adapter()
        TF = TF + adapter.delta_tf(tf_local_pid_t, V_tf)
        c_tok = torch.full((tf_local_pid_t.numel(),), int(self.default_cancer_id), device=DEVICE, dtype=torch.long)
        TF = TF + adapter.delta_cancer(c_tok, V_cancer)
        return TF

    @torch.no_grad()
    def score_gene_vec(self, gene_local_pid_t: torch.Tensor, mode: str) -> torch.Tensor:
        pack = self._module_pack(mode)
        G = self.gene_base_from_pid(gene_local_pid_t)
        adapter = pack["adapter"]
        _, V_gene, _, V_cancer = adapter()
        G = G + adapter.delta_gene(gene_local_pid_t, V_gene)
        c_tok = torch.full((gene_local_pid_t.numel(),), int(self.default_cancer_id), device=DEVICE, dtype=torch.long)
        G = G + adapter.delta_cancer(c_tok, V_cancer)
        return G

    @torch.no_grad()
    def cell_tf_score(self, sid: str, rid: int, tf_global_id: int, mode: str) -> float:
        z0 = self.encode_cell_base(sid, rid)
        z = self.adjust_Z(z0, sid, rid, mode, task="tf")
        tf_pid = self.tf_global2local[int(tf_global_id)]
        tf_vec = self.score_tf_vec(torch.tensor([tf_pid], device=DEVICE), mode)[0]
        return float((z * tf_vec).sum().detach().cpu())

    def top_rationale_gene_ids(self, sid: str, rid: int, tf_global_id: int, mode: str, k: int = 20):
        z0 = self.encode_cell_base(sid, rid)
        z_tf = self.adjust_Z(z0, sid, rid, mode, task="tf")

        idx_cat, val_cat, _, _ = self.get_cell_tokens(sid, rid)
        token_gene_ids = np.asarray(idx_cat, dtype=np.int64)
        token_vals = np.asarray(val_cat, dtype=np.float32)

        tf_targets = set(self.tf2targets[int(tf_global_id)])
        keep = np.array([int(g) in tf_targets for g in token_gene_ids], dtype=bool)
        cand_gene_ids = token_gene_ids[keep]
        cand_vals = token_vals[keep]

        if cand_gene_ids.size == 0:
            keep2 = np.array([int(g) in self.gid2pid_u for g in token_gene_ids], dtype=bool)
            cand_gene_ids = token_gene_ids[keep2][:k]
            cand_vals = token_vals[keep2][:k]

        rows = []
        tf_pid = self.tf_global2local[int(tf_global_id)]
        tf_vec = self.score_tf_vec(torch.tensor([tf_pid], device=DEVICE), mode)[0]

        for gid, gval in zip(cand_gene_ids.tolist(), cand_vals.tolist()):
            if int(gid) not in self.gid2pid_u:
                continue
            gene_pid = self.gid2pid_u[int(gid)]
            g_vec = self.score_gene_vec(torch.tensor([gene_pid], device=DEVICE), mode)[0]
            gene_score = float((z_tf * g_vec).sum().detach().cpu())
            tf_align = float((g_vec * tf_vec).sum().detach().cpu())
            contrib = float(max(gval, 1e-8) * gene_score * max(tf_align, 0.0))
            rows.append((int(gid), contrib))

        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        return [gid for gid, _ in rows[:k]]

    @torch.no_grad()
    def encode_cell_from_masked_tokens(self, sid: str, rid: int, keep_gene_ids: set[int] | None = None, remove_gene_ids: set[int] | None = None):
        idx_cat, val_cat, _, _ = self.get_cell_tokens(sid, rid)
        idx_cat = np.asarray(idx_cat, dtype=np.int64)
        val_cat = np.asarray(val_cat, dtype=np.float32)

        m = np.ones(idx_cat.shape[0], dtype=bool)
        if keep_gene_ids is not None:
            m &= np.array([int(x) in keep_gene_ids for x in idx_cat], dtype=bool)
        if remove_gene_ids is not None:
            m &= np.array([int(x) not in remove_gene_ids for x in idx_cat], dtype=bool)

        idx2 = idx_cat[m]
        val2 = val_cat[m]

        if idx2.size == 0:
            idx2 = np.asarray([idx_cat[0]], dtype=np.int64)
            val2 = np.asarray([0.0], dtype=np.float32)

        offsets = np.asarray([0], dtype=np.int64)
        cancer_ids = np.asarray([self.default_cancer_id], dtype=np.int64)

        z = self.base_model.encode_rna(
            torch.from_numpy(idx2).to(DEVICE),
            torch.from_numpy(val2).to(DEVICE),
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )[0]
        return z

    @torch.no_grad()
    def masked_tf_score(self, sid: str, rid: int, tf_global_id: int, mode: str, keep_gene_ids: set[int] | None = None, remove_gene_ids: set[int] | None = None) -> float:
        z0 = self.encode_cell_from_masked_tokens(sid, rid, keep_gene_ids=keep_gene_ids, remove_gene_ids=remove_gene_ids)
        z = self.adjust_Z(z0, sid, rid, mode, task="tf")
        tf_pid = self.tf_global2local[int(tf_global_id)]
        tf_vec = self.score_tf_vec(torch.tensor([tf_pid], device=DEVICE), mode)[0]
        return float((z * tf_vec).sum().detach().cpu())

    def counterfactual_delete(self, sid: str, rid: int, tf_global_id: int, mode: str, max_remove: int = COUNTERFACTUAL_MAX_REMOVE):
        top_genes = self.top_rationale_gene_ids(sid, rid, tf_global_id, mode, k=max_remove + 5)
        orig_score = self.cell_tf_score(sid, rid, tf_global_id, mode)

        removed = []
        trajectory = [orig_score]

        z0 = self.encode_cell_base(sid, rid)
        z = self.adjust_Z(z0, sid, rid, mode, task="tf")
        tf_local_all = np.arange(len(self.tf_in_prior), dtype=np.int64)
        tf_vecs = self.score_tf_vec(torch.from_numpy(tf_local_all).to(DEVICE), mode)
        scores_all = (z.view(1, -1) @ tf_vecs.t()).view(-1).detach().cpu().numpy()
        target_local = self.tf_global2local[int(tf_global_id)]
        target_rank = np.argsort(scores_all)[::-1]
        alt_local = int(target_rank[0]) if int(target_rank[0]) != target_local else int(target_rank[1])
        alt_tf_global = int(self.tf_in_prior[alt_local])

        flipped = False
        for gid in top_genes[:max_remove]:
            removed.append(int(gid))
            s_now = self.masked_tf_score(sid, rid, tf_global_id, mode, remove_gene_ids=set(removed))
            trajectory.append(s_now)

            zcf0 = self.encode_cell_from_masked_tokens(sid, rid, remove_gene_ids=set(removed))
            zcf = self.adjust_Z(zcf0, sid, rid, mode, task="tf")
            scores_cf = (zcf.view(1, -1) @ tf_vecs.t()).view(-1).detach().cpu().numpy()
            new_top = int(np.argmax(scores_cf))

            if (new_top != target_local) or (s_now <= orig_score * (1.0 - COUNTERFACTUAL_TARGET_DROP)):
                flipped = True
                break

        return {
            "orig_score": float(orig_score),
            "trajectory": trajectory,
            "removed_gene_ids": removed,
            "removed_gene_names": [self.genes_vocab[g] for g in removed],
            "target_tf": self.tfs_vocab[int(tf_global_id)],
            "alt_tf": self.tfs_vocab[int(alt_tf_global)],
            "flipped": bool(flipped),
        }

# ============================================================
# 5) SOD2-SPECIFIC CONFIG
# ============================================================

SOD2_TARGET_GENE = "SOD2"

SOD2_HOTSPOT_CELL_TABLE = (
    FIG2A_DIR
    / "fig2a_SOD2_exact_MLTD_contribution_hotspot_cell_table.csv"
)

# The current main figure retains the four strongest positive-waveform
# SOD2 source clusters.
SOD2_TOP_SOURCE_CLUSTER_COUNT = 4

# Balanced cell selection.
SOD2_MAX_CELLS_PER_CLUSTER = 20
SOD2_MAX_TOTAL_CELLS = 80
SOD2_MIN_TOTAL_CELLS = 24

# Sequential deletion.
SOD2_DELETE_STEPS = 10
SOD2_MIN_POSITIVE_RATIONALE_GENES = SOD2_DELETE_STEPS
SOD2_TARGET_DROP_THRESHOLD = 0.35
SOD2_EARLY_DROP_STEP = 3

# Matched-null control.
#
# Publication run: 100.
# A quick test may temporarily use 10-20.
SOD2_N_NULL_REPEATS = 100
SOD2_NULL_NEAREST_POOL = 24

# Candidate support genes.
SOD2_MAX_SUPPORT_CANDIDATES = 512
SOD2_EXCLUDE_TARGET_FROM_DELETION = True

# Associated TF transition.
SOD2_MIN_ASSOCIATED_TFS = 8
SOD2_MAX_ASSOCIATED_TFS = 20
SOD2_MAX_TRANSITION_EDGES = 14
SOD2_MAX_TRANSITION_NODES = 10

# Fragility core: deliberately excludes any flip bonus.
SOD2_FRAG_W_EARLY = 0.45
SOD2_FRAG_W_AUC = 0.25
SOD2_FRAG_W_MAX = 0.15
SOD2_FRAG_W_FINAL = 0.10
SOD2_FRAG_W_EVENT = 0.05

# Bootstrap / plotting.
SOD2_N_BOOTSTRAP = 1000
SOD2_DPI = 450

SOD2_COLOR_RATIONALE = "#C5486B"
SOD2_COLOR_RATIONALE_LIGHT = "#E8AFC0"
SOD2_COLOR_NULL = "#7E8792"
SOD2_COLOR_NULL_LIGHT = "#D8DCE1"
SOD2_COLOR_GOLD = "#E5B94F"
SOD2_COLOR_TEXT = "#222222"
SOD2_COLOR_GRID = "#E9E9E9"

SOD2_CACHE_DIR = OUT_DIR / "cell_cache"
SOD2_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6) GENERAL HELPERS
# ============================================================

def sod2_normalize_name(value) -> str:
    return str(value).strip().upper()


def sod2_safe_filename(value: str) -> str:
    text = str(value)

    for character in [
        "/",
        "\\",
        ":",
        " ",
        "|",
        ";",
        ",",
    ]:
        text = text.replace(
            character,
            "_",
        )

    return text


def sod2_save_figure(
    figure: plt.Figure,
    stem: Path,
) -> None:
    for extension in [
        "png",
        "pdf",
        "svg",
    ]:
        path = stem.with_suffix(
            f".{extension}"
        )

        kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }

        if extension == "png":
            kwargs[
                "dpi"
            ] = int(
                SOD2_DPI
            )

        figure.savefig(
            path,
            **kwargs,
        )

        log(
            "[SAVE]",
            path,
        )


def sod2_style_axis(
    axis: plt.Axes,
    grid_axis: str | None = "y",
) -> None:
    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.spines[
        "left"
    ].set_linewidth(
        0.9
    )

    axis.spines[
        "bottom"
    ].set_linewidth(
        0.9
    )

    axis.tick_params(
        width=0.8,
        length=3.2,
        labelsize=8.0,
    )

    if grid_axis is not None:
        axis.grid(
            axis=grid_axis,
            color=SOD2_COLOR_GRID,
            linewidth=0.7,
            zorder=0,
        )

        axis.set_axisbelow(
            True
        )


def sod2_empirical_upper_p(
    observed: float,
    null_values: np.ndarray,
) -> float:
    null_values = np.asarray(
        null_values,
        dtype=np.float64,
    )

    null_values = null_values[
        np.isfinite(
            null_values
        )
    ]

    if null_values.size == 0:
        return np.nan

    return float(
        (
            1
            + np.sum(
                null_values
                >= float(
                    observed
                )
            )
        )
        / (
            null_values.size
            + 1
        )
    )


def sod2_bootstrap_mean_interval(
    matrix: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Bootstrap input must be cells × deletion steps."
        )

    number_of_rows = int(
        matrix.shape[
            0
        ]
    )

    mean_curve = np.nanmean(
        matrix,
        axis=0,
    )

    if number_of_rows < 2:
        return (
            mean_curve,
            mean_curve,
            mean_curve,
        )

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    bootstrap_curves = np.zeros(
        (
            int(
                n_bootstrap
            ),
            matrix.shape[
                1
            ],
        ),
        dtype=np.float64,
    )

    for bootstrap_index in range(
        int(
            n_bootstrap
        )
    ):
        sampled_rows = rng.integers(
            0,
            number_of_rows,
            size=number_of_rows,
        )

        bootstrap_curves[
            bootstrap_index
        ] = np.nanmean(
            matrix[
                sampled_rows
            ],
            axis=0,
        )

    lower = np.nanpercentile(
        bootstrap_curves,
        2.5,
        axis=0,
    )

    upper = np.nanpercentile(
        bootstrap_curves,
        97.5,
        axis=0,
    )

    return (
        mean_curve,
        lower,
        upper,
    )


# ============================================================
# 7) TARGET GENE AND CELL SELECTION
# ============================================================

def sod2_find_target_gene_ids(
    explainer: FrontierExplainerStandalone,
) -> tuple[
    int,
    int,
    str,
]:
    """
    Resolve SOD2 against the adapted gene union rather than taking the
    first symbol match in the global vocabulary.

    The global vocabulary may contain duplicated gene symbols. The first
    global SOD2 entry can therefore be absent from the prior-constrained
    adapted union even though another SOD2 entry is present and was used by
    the Fig. 2a heatmap workflow.

    Selection rule:
      1. collect every exact SOD2 symbol match in the global vocabulary;
      2. retain only matches present in gid2pid_u;
      3. if several remain, select the entry with the largest structural
         prior degree (TF-target degree + gene-peak degree);
      4. use the adapted-union local PID for all downstream scoring.
    """
    target = sod2_normalize_name(
        SOD2_TARGET_GENE
    )

    all_symbol_matches = [
        int(
            index
        )
        for index, gene_name in enumerate(
            explainer.genes_vocab
        )
        if sod2_normalize_name(
            gene_name
        )
        == target
    ]

    if len(
        all_symbol_matches
    ) == 0:
        partial_matches = [
            (
                int(
                    index
                ),
                str(
                    gene_name
                ),
            )
            for index, gene_name in enumerate(
                explainer.genes_vocab
            )
            if target
            in sod2_normalize_name(
                gene_name
            )
        ][
            : 30
        ]

        raise KeyError(
            (
                f"{SOD2_TARGET_GENE} is absent from the global gene "
                f"vocabulary. Partial matches: {partial_matches}"
            )
        )

    adapted_matches = [
        int(
            global_id
        )
        for global_id in all_symbol_matches
        if int(
            global_id
        )
        in explainer.gid2pid_u
    ]

    diagnostic_rows = []

    for global_id in all_symbol_matches:
        tf_target_degree = int(
            sum(
                int(
                    global_id
                )
                in set(
                    map(
                        int,
                        target_ids,
                    )
                )
                for target_ids in explainer.tf2targets.values()
            )
        )

        peak_degree = int(
            len(
                explainer.gene2peaks.get(
                    int(
                        global_id
                    ),
                    [],
                )
            )
        )

        diagnostic_rows.append(
            {
                "gene_symbol": str(
                    explainer.genes_vocab[
                        int(
                            global_id
                        )
                    ]
                ),
                "global_gene_id": int(
                    global_id
                ),
                "present_in_adapted_union": int(
                    int(
                        global_id
                    )
                    in explainer.gid2pid_u
                ),
                "adapted_local_pid": (
                    int(
                        explainer.gid2pid_u[
                            int(
                                global_id
                            )
                        ]
                    )
                    if int(
                        global_id
                    )
                    in explainer.gid2pid_u
                    else -1
                ),
                "TF_target_degree": (
                    tf_target_degree
                ),
                "gene_peak_degree": (
                    peak_degree
                ),
                "combined_prior_degree": int(
                    tf_target_degree
                    + peak_degree
                ),
            }
        )

    diagnostic_table = pd.DataFrame(
        diagnostic_rows
    ).sort_values(
        [
            "present_in_adapted_union",
            "combined_prior_degree",
            "global_gene_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    diagnostic_path = (
        OUT_DIR
        / "SOD2_global_symbol_resolution_diagnostics.csv"
    )

    diagnostic_table.to_csv(
        diagnostic_path,
        index=False,
    )

    log(
        "[SAVE]",
        diagnostic_path,
    )

    log(
        "[SOD2 SYMBOL MATCHES]",
        diagnostic_table.to_dict(
            orient="records"
        ),
    )

    if len(
        adapted_matches
    ) == 0:
        union_partial_matches = [
            {
                "local_pid": int(
                    local_pid
                ),
                "global_gene_id": int(
                    global_id
                ),
                "gene_name": str(
                    explainer.genes_vocab[
                        int(
                            global_id
                        )
                    ]
                ),
            }
            for global_id, local_pid in explainer.gid2pid_u.items()
            if target
            in sod2_normalize_name(
                explainer.genes_vocab[
                    int(
                        global_id
                    )
                ]
            )
        ][
            : 30
        ]

        raise KeyError(
            (
                f"None of the exact {SOD2_TARGET_GENE} symbol entries are "
                "present in the adapted gene union. "
                f"Global exact IDs={all_symbol_matches}; "
                f"union partial matches={union_partial_matches}. "
                "This means the loaded checkpoint/prior vocabulary does not "
                "contain an adapted SOD2 output vector and should not be "
                "silently replaced by an unadapted base-only vector."
            )
        )

    degree_lookup = {
        int(
            row[
                "global_gene_id"
            ]
        ): int(
            row[
                "combined_prior_degree"
            ]
        )
        for _, row in diagnostic_table.iterrows()
    }

    target_global_id = sorted(
        adapted_matches,
        key=lambda global_id: (
            -int(
                degree_lookup.get(
                    int(
                        global_id
                    ),
                    0,
                )
            ),
            int(
                global_id
            ),
        ),
    )[
        0
    ]

    target_local_pid = int(
        explainer.gid2pid_u[
            int(
                target_global_id
            )
        ]
    )

    resolved_name = str(
        explainer.genes_vocab[
            int(
                target_global_id
            )
        ]
    )

    log(
        "[TARGET RESOLUTION]",
        (
            f"symbol={resolved_name}, "
            f"global_id={target_global_id}, "
            f"adapted_local_pid={target_local_pid}, "
            f"candidate_exact_ids={all_symbol_matches}"
        ),
    )

    return (
        int(
            target_global_id
        ),
        int(
            target_local_pid
        ),
        resolved_name,
    )


def sod2_load_selected_cells(
    explainer: FrontierExplainerStandalone,
) -> tuple[
    pd.DataFrame,
    list[int],
]:
    if not SOD2_HOTSPOT_CELL_TABLE.exists():
        raise FileNotFoundError(
            (
                "Missing exact SOD2 hotspot cell table:\n"
                f"{SOD2_HOTSPOT_CELL_TABLE}"
            )
        )

    table = pd.read_csv(
        SOD2_HOTSPOT_CELL_TABLE
    )

    required_columns = [
        "sid",
        "rid",
        "CCI_cluster_position",
        "CCI_cluster_name",
        "saved_cluster_waveform_zscore",
        "exact_oriented_response_intensity",
        "within_cluster_response_percentile",
        "is_hotspot_cell",
    ]

    missing = [
        column
        for column in required_columns
        if column not in table.columns
    ]

    if missing:
        raise KeyError(
            (
                f"{SOD2_HOTSPOT_CELL_TABLE.name} is missing columns: "
                f"{missing}"
            )
        )

    table[
        "sid"
    ] = table[
        "sid"
    ].astype(
        str
    )

    table[
        "rid"
    ] = table[
        "rid"
    ].astype(
        np.int64
    )

    table[
        "CCI_cluster_position"
    ] = table[
        "CCI_cluster_position"
    ].astype(
        np.int64
    )

    # The explainer is currently restricted to CRC_RNA_TO009.
    table = table[
        table[
            "sid"
        ].isin(
            set(
                explainer.sid2sample.keys()
            )
        )
    ].copy()

    cluster_summary = (
        table[
            [
                "CCI_cluster_position",
                "saved_cluster_waveform_zscore",
            ]
        ]
        .drop_duplicates(
            "CCI_cluster_position"
        )
        .sort_values(
            "saved_cluster_waveform_zscore",
            ascending=False,
        )
    )

    positive_clusters = cluster_summary[
        cluster_summary[
            "saved_cluster_waveform_zscore"
        ]
        > 0
    ]

    retained_clusters = (
        positive_clusters.head(
            int(
                SOD2_TOP_SOURCE_CLUSTER_COUNT
            )
        )[
            "CCI_cluster_position"
        ]
        .astype(
            int
        )
        .tolist()
    )

    if len(
        retained_clusters
    ) < int(
        SOD2_TOP_SOURCE_CLUSTER_COUNT
    ):
        raise RuntimeError(
            (
                "Fewer than four positive SOD2 waveform clusters were "
                "available."
            )
        )

    candidate = table[
        table[
            "CCI_cluster_position"
        ].isin(
            retained_clusters
        )
        & (
            table[
                "is_hotspot_cell"
            ]
            == 1
        )
    ].copy()

    # A fallback is provided for any retained cluster whose contour contains
    # too few cells.
    selected_parts = []

    for cluster_position in retained_clusters:
        group = candidate[
            candidate[
                "CCI_cluster_position"
            ]
            == int(
                cluster_position
            )
        ].copy()

        if group.shape[
            0
        ] < 5:
            group = table[
                (
                    table[
                        "CCI_cluster_position"
                    ]
                    == int(
                        cluster_position
                    )
                )
                & (
                    table[
                        "within_cluster_response_percentile"
                    ]
                    >= 0.80
                )
            ].copy()

        group = group.sort_values(
            "exact_oriented_response_intensity",
            ascending=False,
        ).head(
            int(
                SOD2_MAX_CELLS_PER_CLUSTER
            )
        )

        selected_parts.append(
            group
        )

        log(
            "[SOD2 CELLS]",
            (
                f"cluster={cluster_position}, "
                f"selected={group.shape[0]}"
            ),
        )

    selected = pd.concat(
        selected_parts,
        axis=0,
        ignore_index=True,
    )

    selected = selected.sort_values(
        [
            "CCI_cluster_position",
            "exact_oriented_response_intensity",
        ],
        ascending=[
            True,
            False,
        ],
    ).head(
        int(
            SOD2_MAX_TOTAL_CELLS
        )
    ).reset_index(
        drop=True
    )

    valid_rows = []

    for index, row in selected.iterrows():
        sid = str(
            row[
                "sid"
            ]
        )

        rid = int(
            row[
                "rid"
            ]
        )

        if sid not in explainer.sid2sample:
            continue

        if (
            rid < 0
            or rid
            >= explainer.sid2sample[
                sid
            ].counts_shape[
                0
            ]
        ):
            continue

        valid_rows.append(
            int(
                index
            )
        )

    selected = selected.loc[
        valid_rows
    ].reset_index(
        drop=True
    )

    if selected.shape[
        0
    ] < int(
        SOD2_MIN_TOTAL_CELLS
    ):
        raise RuntimeError(
            (
                f"Only {selected.shape[0]} usable SOD2 hotspot cells were "
                f"found; minimum required is {SOD2_MIN_TOTAL_CELLS}."
            )
        )

    output_path = (
        OUT_DIR
        / "SOD2_counterfactual_selected_hotspot_cells.csv"
    )

    selected.to_csv(
        output_path,
        index=False,
    )

    log(
        "[SAVE]",
        output_path,
    )

    return (
        selected,
        retained_clusters,
    )


# ============================================================
# 8) SOD2 TARGET SCORE AND SUPPORT-GENE RANKING
# ============================================================

@torch.no_grad()
def sod2_target_gene_score(
    explainer: FrontierExplainerStandalone,
    sid: str,
    rid: int,
    target_local_pid: int,
    remove_gene_ids: set[int] | None = None,
) -> float:
    if remove_gene_ids is None:
        z0 = explainer.encode_cell_base(
            sid,
            rid,
        )
    else:
        z0 = explainer.encode_cell_from_masked_tokens(
            sid,
            rid,
            remove_gene_ids=remove_gene_ids,
        )

    z = explainer.adjust_Z(
        z0,
        sid,
        rid,
        mode="after",
        task="rna",
    )

    target_vector = explainer.score_gene_vec(
        torch.tensor(
            [
                int(
                    target_local_pid
                )
            ],
            device=DEVICE,
            dtype=torch.long,
        ),
        mode="after",
    )[
        0
    ]

    return float(
        (
            z
            * target_vector
        ).sum().detach().cpu()
    )


@torch.no_grad()
def sod2_rank_support_genes(
    explainer: FrontierExplainerStandalone,
    sid: str,
    rid: int,
    target_global_id: int,
    target_local_pid: int,
) -> pd.DataFrame:
    z0 = explainer.encode_cell_base(
        sid,
        rid,
    )

    z = explainer.adjust_Z(
        z0,
        sid,
        rid,
        mode="after",
        task="rna",
    )

    target_vector = explainer.score_gene_vec(
        torch.tensor(
            [
                int(
                    target_local_pid
                )
            ],
            device=DEVICE,
            dtype=torch.long,
        ),
        mode="after",
    )[
        0
    ]

    (
        token_gene_ids,
        token_values,
        _,
        _,
    ) = explainer.get_cell_tokens(
        sid,
        rid,
    )

    token_gene_ids = np.asarray(
        token_gene_ids,
        dtype=np.int64,
    )

    token_values = np.asarray(
        token_values,
        dtype=np.float32,
    )

    keep = np.asarray(
        [
            (
                int(
                    gene_id
                )
                in explainer.gid2pid_u
            )
            and (
                not SOD2_EXCLUDE_TARGET_FROM_DELETION
                or int(
                    gene_id
                )
                != int(
                    target_global_id
                )
            )
            and float(
                value
            )
            > 0
            for gene_id, value in zip(
                token_gene_ids.tolist(),
                token_values.tolist(),
            )
        ],
        dtype=bool,
    )

    candidate_gene_ids = token_gene_ids[
        keep
    ]

    candidate_values = token_values[
        keep
    ]

    if candidate_gene_ids.size == 0:
        return pd.DataFrame()

    candidate_pids = np.asarray(
        [
            explainer.gid2pid_u[
                int(
                    gene_id
                )
            ]
            for gene_id in candidate_gene_ids.tolist()
        ],
        dtype=np.int64,
    )

    gene_vectors = explainer.score_gene_vec(
        torch.from_numpy(
            candidate_pids
        ).to(
            DEVICE
        ),
        mode="after",
    )

    cell_support = (
        gene_vectors
        @ z
    ).detach().cpu().numpy().astype(
        np.float64
    )

    target_alignment = (
        gene_vectors
        @ target_vector
    ).detach().cpu().numpy().astype(
        np.float64
    )

    positive_cell_support = np.clip(
        cell_support,
        0.0,
        None,
    )

    positive_target_alignment = np.clip(
        target_alignment,
        0.0,
        None,
    )

    contribution = (
        np.maximum(
            candidate_values.astype(
                np.float64
            ),
            1e-12,
        )
        * positive_cell_support
        * positive_target_alignment
    )

    result = pd.DataFrame(
        {
            "gene_global_id": (
                candidate_gene_ids.astype(
                    np.int64
                )
            ),
            "gene_name": [
                str(
                    explainer.genes_vocab[
                        int(
                            gene_id
                        )
                    ]
                )
                for gene_id in candidate_gene_ids.tolist()
            ],
            "projected_RNA_value": (
                candidate_values.astype(
                    np.float64
                )
            ),
            "cell_state_support": (
                cell_support
            ),
            "target_alignment": (
                target_alignment
            ),
            "SOD2_support_contribution": (
                contribution
            ),
        }
    )

    result = result.sort_values(
        [
            "SOD2_support_contribution",
            "projected_RNA_value",
        ],
        ascending=[
            False,
            False,
        ],
    ).drop_duplicates(
        "gene_global_id"
    ).head(
        int(
            SOD2_MAX_SUPPORT_CANDIDATES
        )
    ).reset_index(
        drop=True
    )

    return result


def sod2_choose_matched_null_order(
    support_table: pd.DataFrame,
    rationale_gene_ids: list[int],
    rng: np.random.Generator,
) -> list[int]:
    rationale_set = set(
        map(
            int,
            rationale_gene_ids
        )
    )

    pool = support_table[
        ~support_table[
            "gene_global_id"
        ].isin(
            rationale_set
        )
    ].copy()

    if pool.shape[
        0
    ] < len(
        rationale_gene_ids
    ):
        raise RuntimeError(
            "Too few non-rationale genes were available for matched null."
        )

    features = pool[
        [
            "projected_RNA_value",
            "cell_state_support",
        ]
    ].to_numpy(
        dtype=np.float64,
    )

    features[
        :,
        0
    ] = np.log1p(
        np.maximum(
            features[
                :,
                0
            ],
            0.0,
        )
    )

    median = np.nanmedian(
        features,
        axis=0,
    )

    mad = np.nanmedian(
        np.abs(
            features
            - median[
                None,
                :
            ]
        ),
        axis=0,
    )

    scale = np.maximum(
        1.4826
        * mad,
        1e-5,
    )

    used_pool_indices: set[int] = set()
    selected_gene_ids = []

    support_lookup = support_table.set_index(
        "gene_global_id"
    )

    pool_gene_ids = pool[
        "gene_global_id"
    ].to_numpy(
        dtype=np.int64,
    )

    for rationale_gene_id in rationale_gene_ids:
        rationale_row = support_lookup.loc[
            int(
                rationale_gene_id
            )
        ]

        target_feature = np.asarray(
            [
                np.log1p(
                    max(
                        float(
                            rationale_row[
                                "projected_RNA_value"
                            ]
                        ),
                        0.0,
                    )
                ),
                float(
                    rationale_row[
                        "cell_state_support"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        standardized_distance = np.sqrt(
            np.sum(
                (
                    (
                        features
                        - target_feature[
                            None,
                            :
                        ]
                    )
                    / scale[
                        None,
                        :
                    ]
                )
                ** 2,
                axis=1,
            )
        )

        available_indices = np.asarray(
            [
                index
                for index in range(
                    pool.shape[
                        0
                    ]
                )
                if index
                not in used_pool_indices
            ],
            dtype=np.int64,
        )

        if available_indices.size == 0:
            raise RuntimeError(
                "Matched-null candidate pool was exhausted."
            )

        available_order = available_indices[
            np.argsort(
                standardized_distance[
                    available_indices
                ]
            )
        ]

        nearest = available_order[
            : min(
                int(
                    SOD2_NULL_NEAREST_POOL
                ),
                available_order.size,
            )
        ]

        chosen_pool_index = int(
            rng.choice(
                nearest
            )
        )

        used_pool_indices.add(
            chosen_pool_index
        )

        selected_gene_ids.append(
            int(
                pool_gene_ids[
                    chosen_pool_index
                ]
            )
        )

    return selected_gene_ids


# ============================================================
# 9) TRAJECTORY AND FRAGILITY
# ============================================================

def sod2_run_deletion_trajectory(
    explainer: FrontierExplainerStandalone,
    sid: str,
    rid: int,
    target_local_pid: int,
    deletion_order: list[int],
) -> np.ndarray:
    scores = [
        sod2_target_gene_score(
            explainer,
            sid,
            rid,
            target_local_pid,
            remove_gene_ids=None,
        )
    ]

    removed: set[int] = set()

    for gene_id in deletion_order:
        removed.add(
            int(
                gene_id
            )
        )

        scores.append(
            sod2_target_gene_score(
                explainer,
                sid,
                rid,
                target_local_pid,
                remove_gene_ids=removed,
            )
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


def sod2_score_to_retention(
    scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    baseline = float(
        scores[
            0
        ]
    )

    if (
        not np.isfinite(
            baseline
        )
        or baseline
        <= 1e-6
    ):
        raise ValueError(
            (
                "The baseline SOD2 target score is non-positive; "
                "the cell cannot be used for ratio-based retention."
            )
        )

    return (
        scores
        / baseline
    )


def sod2_fragility_metrics(
    retention: np.ndarray,
) -> dict[str, float | int]:
    retention = np.asarray(
        retention,
        dtype=np.float64,
    )

    # The deletion path may show small rebounds. The cumulative minimum
    # measures the achieved collapse without allowing later numerical
    # rebounds to erase it.
    envelope = np.minimum.accumulate(
        retention
    )

    loss = np.clip(
        1.0
        - envelope,
        0.0,
        2.0,
    )

    early_index = min(
        int(
            SOD2_EARLY_DROP_STEP
        ),
        loss.size
        - 1,
    )

    early_drop = float(
        loss[
            early_index
        ]
    )

    if loss.size > 1:
        auc_loss = float(
            np.trapezoid(
                loss
            )
            / (
                loss.size
                - 1
            )
        )
    else:
        auc_loss = 0.0

    maximum_drop = float(
        np.max(
            loss
        )
    )

    final_loss = float(
        loss[
            -1
        ]
    )

    event_indices = np.where(
        loss
        >= float(
            SOD2_TARGET_DROP_THRESHOLD
        )
    )[
        0
    ]

    if event_indices.size > 0:
        event_step = int(
            event_indices[
                0
            ]
        )
    else:
        event_step = int(
            SOD2_DELETE_STEPS
            + 1
        )

    if event_step > int(
        SOD2_DELETE_STEPS
    ):
        early_event_score = 0.0
    else:
        early_event_score = float(
            1.0
            - np.clip(
                (
                    event_step
                    - 1
                )
                / max(
                    int(
                        SOD2_DELETE_STEPS
                    ),
                    1,
                ),
                0.0,
                1.0,
            )
        )

    fragility_core = float(
        float(
            SOD2_FRAG_W_EARLY
        )
        * early_drop
        + float(
            SOD2_FRAG_W_AUC
        )
        * auc_loss
        + float(
            SOD2_FRAG_W_MAX
        )
        * maximum_drop
        + float(
            SOD2_FRAG_W_FINAL
        )
        * final_loss
        + float(
            SOD2_FRAG_W_EVENT
        )
        * early_event_score
    )

    return {
        "early_drop": early_drop,
        "auc_loss": auc_loss,
        "maximum_drop": maximum_drop,
        "final_loss": final_loss,
        "event_step": event_step,
        "early_event_score": early_event_score,
        "fragility_core": fragility_core,
    }


# ============================================================
# 10) SOD2-ASSOCIATED TF TRANSITIONS
# ============================================================

@torch.no_grad()
def sod2_get_associated_tf_local_ids(
    explainer: FrontierExplainerStandalone,
    target_global_id: int,
    target_local_pid: int,
) -> np.ndarray:
    prior_tf_global_ids = [
        int(
            tf_global_id
        )
        for tf_global_id in explainer.tf_in_prior
        if int(
            target_global_id
        )
        in set(
            explainer.tf2targets[
                int(
                    tf_global_id
                )
            ]
        )
    ]

    if len(
        prior_tf_global_ids
    ) >= int(
        SOD2_MIN_ASSOCIATED_TFS
    ):
        prior_tf_global_ids = prior_tf_global_ids[
            : int(
                SOD2_MAX_ASSOCIATED_TFS
            )
        ]

        return np.asarray(
            [
                explainer.tf_global2local[
                    int(
                        tf_global_id
                    )
                ]
                for tf_global_id in prior_tf_global_ids
            ],
            dtype=np.int64,
        )

    # Fallback: choose TF embeddings most aligned with the adapted SOD2
    # target-gene vector.
    all_tf_local = np.arange(
        len(
            explainer.tf_in_prior
        ),
        dtype=np.int64,
    )

    tf_vectors = explainer.score_tf_vec(
        torch.from_numpy(
            all_tf_local
        ).to(
            DEVICE
        ),
        mode="after",
    )

    target_vector = explainer.score_gene_vec(
        torch.tensor(
            [
                int(
                    target_local_pid
                )
            ],
            device=DEVICE,
            dtype=torch.long,
        ),
        mode="after",
    )[
        0
    ]

    alignment = (
        tf_vectors
        @ target_vector
    ).detach().cpu().numpy()

    top_local = np.argsort(
        alignment
    )[
        ::-1
    ][
        : int(
            SOD2_MAX_ASSOCIATED_TFS
        )
    ]

    return np.asarray(
        top_local,
        dtype=np.int64,
    )


@torch.no_grad()
def sod2_top_associated_tf(
    explainer: FrontierExplainerStandalone,
    sid: str,
    rid: int,
    associated_tf_local_ids: np.ndarray,
    remove_gene_ids: set[int] | None = None,
) -> dict[str, float | int | str]:
    if remove_gene_ids is None:
        z0 = explainer.encode_cell_base(
            sid,
            rid,
        )
    else:
        z0 = explainer.encode_cell_from_masked_tokens(
            sid,
            rid,
            remove_gene_ids=remove_gene_ids,
        )

    z = explainer.adjust_Z(
        z0,
        sid,
        rid,
        mode="after",
        task="tf",
    )

    tf_vectors = explainer.score_tf_vec(
        torch.from_numpy(
            associated_tf_local_ids.astype(
                np.int64
            )
        ).to(
            DEVICE
        ),
        mode="after",
    )

    scores = (
        z.view(
            1,
            -1,
        )
        @ tf_vectors.t()
    ).view(
        -1
    ).detach().cpu().numpy()

    order = np.argsort(
        scores
    )[
        ::-1
    ]

    top_index = int(
        order[
            0
        ]
    )

    top_local_pid = int(
        associated_tf_local_ids[
            top_index
        ]
    )

    top_global_id = int(
        explainer.tf_in_prior[
            top_local_pid
        ]
    )

    return {
        "tf_local_pid": top_local_pid,
        "tf_global_id": top_global_id,
        "tf_name": str(
            explainer.tfs_vocab[
                top_global_id
            ]
        ),
        "tf_score": float(
            scores[
                top_index
            ]
        ),
    }


# ============================================================
# 11) PER-CELL COMPUTATION AND CACHE
# ============================================================

def sod2_cell_cache_paths(
    sid: str,
    rid: int,
) -> tuple[
    Path,
    Path,
]:
    stem = (
        f"{sod2_safe_filename(sid)}_"
        f"rid{int(rid)}"
    )

    return (
        SOD2_CACHE_DIR
        / f"{stem}.npz",
        SOD2_CACHE_DIR
        / f"{stem}.json",
    )


def sod2_compute_one_cell(
    explainer: FrontierExplainerStandalone,
    row: pd.Series,
    cell_index: int,
    target_global_id: int,
    target_local_pid: int,
    associated_tf_local_ids: np.ndarray,
) -> dict:
    sid = str(
        row[
            "sid"
        ]
    )

    rid = int(
        row[
            "rid"
        ]
    )

    npz_path, json_path = sod2_cell_cache_paths(
        sid,
        rid,
    )

    if (
        npz_path.exists()
        and json_path.exists()
    ):
        arrays = np.load(
            npz_path,
            allow_pickle=True,
        )

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(
                handle
            )

        log(
            "[CACHE]",
            sid,
            rid,
        )

        return {
            "metadata": metadata,
            "rationale_scores": np.asarray(
                arrays[
                    "rationale_scores"
                ],
                dtype=np.float64,
            ),
            "rationale_retention": np.asarray(
                arrays[
                    "rationale_retention"
                ],
                dtype=np.float64,
            ),
            "null_scores": np.asarray(
                arrays[
                    "null_scores"
                ],
                dtype=np.float64,
            ),
            "null_retention": np.asarray(
                arrays[
                    "null_retention"
                ],
                dtype=np.float64,
            ),
            "null_fragility": np.asarray(
                arrays[
                    "null_fragility"
                ],
                dtype=np.float64,
            ),
            "rationale_gene_ids": np.asarray(
                arrays[
                    "rationale_gene_ids"
                ],
                dtype=np.int64,
            ),
            "null_gene_ids": np.asarray(
                arrays[
                    "null_gene_ids"
                ],
                dtype=np.int64,
            ),
        }

    support_table = sod2_rank_support_genes(
        explainer,
        sid,
        rid,
        target_global_id,
        target_local_pid,
    )

    positive_support = support_table[
        support_table[
            "SOD2_support_contribution"
        ]
        > 0
    ].copy()

    if positive_support.shape[
        0
    ] < int(
        SOD2_MIN_POSITIVE_RATIONALE_GENES
    ):
        raise RuntimeError(
            (
                f"{sid}/{rid}: only {positive_support.shape[0]} positive "
                "SOD2 support genes were available."
            )
        )

    rationale_gene_ids = (
        positive_support.head(
            int(
                SOD2_DELETE_STEPS
            )
        )[
            "gene_global_id"
        ]
        .astype(
            int
        )
        .tolist()
    )

    rationale_gene_names = (
        positive_support.head(
            int(
                SOD2_DELETE_STEPS
            )
        )[
            "gene_name"
        ]
        .astype(
            str
        )
        .tolist()
    )

    rationale_scores = sod2_run_deletion_trajectory(
        explainer,
        sid,
        rid,
        target_local_pid,
        rationale_gene_ids,
    )

    rationale_retention = sod2_score_to_retention(
        rationale_scores
    )

    rationale_metrics = sod2_fragility_metrics(
        rationale_retention
    )

    tf_before = sod2_top_associated_tf(
        explainer,
        sid,
        rid,
        associated_tf_local_ids,
        remove_gene_ids=None,
    )

    tf_after = sod2_top_associated_tf(
        explainer,
        sid,
        rid,
        associated_tf_local_ids,
        remove_gene_ids=set(
            rationale_gene_ids
        ),
    )

    null_scores = np.zeros(
        (
            int(
                SOD2_N_NULL_REPEATS
            ),
            int(
                SOD2_DELETE_STEPS
            )
            + 1,
        ),
        dtype=np.float64,
    )

    null_retention = np.zeros_like(
        null_scores
    )

    null_fragility = np.zeros(
        int(
            SOD2_N_NULL_REPEATS
        ),
        dtype=np.float64,
    )

    null_gene_ids = np.zeros(
        (
            int(
                SOD2_N_NULL_REPEATS
            ),
            int(
                SOD2_DELETE_STEPS
            ),
        ),
        dtype=np.int64,
    )

    for null_repeat in range(
        int(
            SOD2_N_NULL_REPEATS
        )
    ):
        rng = np.random.default_rng(
            int(
                SEED
            )
            + int(
                cell_index
            )
            * 100003
            + int(
                null_repeat
            )
            * 101
        )

        null_order = sod2_choose_matched_null_order(
            support_table,
            rationale_gene_ids,
            rng,
        )

        null_gene_ids[
            null_repeat
        ] = np.asarray(
            null_order,
            dtype=np.int64,
        )

        repeat_scores = sod2_run_deletion_trajectory(
            explainer,
            sid,
            rid,
            target_local_pid,
            null_order,
        )

        repeat_retention = sod2_score_to_retention(
            repeat_scores
        )

        repeat_metrics = sod2_fragility_metrics(
            repeat_retention
        )

        null_scores[
            null_repeat
        ] = repeat_scores

        null_retention[
            null_repeat
        ] = repeat_retention

        null_fragility[
            null_repeat
        ] = float(
            repeat_metrics[
                "fragility_core"
            ]
        )

        if (
            null_repeat
            + 1
        ) % 10 == 0:
            log(
                "[NULL]",
                (
                    f"cell={cell_index + 1}, "
                    f"repeat={null_repeat + 1}/"
                    f"{SOD2_N_NULL_REPEATS}"
                ),
            )

    metadata = {
        "sid": sid,
        "rid": rid,
        "barcode": str(
            row.get(
                "barcode",
                "",
            )
        ),
        "CCI_cluster_position": int(
            row[
                "CCI_cluster_position"
            ]
        ),
        "CCI_cluster_name": str(
            row[
                "CCI_cluster_name"
            ]
        ),
        "saved_cluster_waveform_zscore": float(
            row[
                "saved_cluster_waveform_zscore"
            ]
        ),
        "exact_oriented_response_intensity": float(
            row[
                "exact_oriented_response_intensity"
            ]
        ),
        "baseline_SOD2_target_score": float(
            rationale_scores[
                0
            ]
        ),
        "rationale_gene_names": (
            rationale_gene_names
        ),
        "rationale_fragility_core": float(
            rationale_metrics[
                "fragility_core"
            ]
        ),
        "rationale_early_drop": float(
            rationale_metrics[
                "early_drop"
            ]
        ),
        "rationale_auc_loss": float(
            rationale_metrics[
                "auc_loss"
            ]
        ),
        "rationale_maximum_drop": float(
            rationale_metrics[
                "maximum_drop"
            ]
        ),
        "rationale_final_loss": float(
            rationale_metrics[
                "final_loss"
            ]
        ),
        "rationale_event_step": int(
            rationale_metrics[
                "event_step"
            ]
        ),
        "null_fragility_mean": float(
            np.mean(
                null_fragility
            )
        ),
        "null_fragility_median": float(
            np.median(
                null_fragility
            )
        ),
        "delta_fragility_vs_null_mean": float(
            rationale_metrics[
                "fragility_core"
            ]
            - np.mean(
                null_fragility
            )
        ),
        "empirical_cell_level_p": float(
            sod2_empirical_upper_p(
                rationale_metrics[
                    "fragility_core"
                ],
                null_fragility,
            )
        ),
        "associated_tf_before": str(
            tf_before[
                "tf_name"
            ]
        ),
        "associated_tf_after": str(
            tf_after[
                "tf_name"
            ]
        ),
        "associated_tf_before_score": float(
            tf_before[
                "tf_score"
            ]
        ),
        "associated_tf_after_score": float(
            tf_after[
                "tf_score"
            ]
        ),
        "associated_tf_switched": int(
            str(
                tf_before[
                    "tf_name"
                ]
            )
            != str(
                tf_after[
                    "tf_name"
                ]
            )
        ),
    }

    np.savez_compressed(
        npz_path,
        rationale_scores=(
            rationale_scores.astype(
                np.float32
            )
        ),
        rationale_retention=(
            rationale_retention.astype(
                np.float32
            )
        ),
        null_scores=(
            null_scores.astype(
                np.float32
            )
        ),
        null_retention=(
            null_retention.astype(
                np.float32
            )
        ),
        null_fragility=(
            null_fragility.astype(
                np.float32
            )
        ),
        rationale_gene_ids=np.asarray(
            rationale_gene_ids,
            dtype=np.int64,
        ),
        null_gene_ids=(
            null_gene_ids.astype(
                np.int64
            )
        ),
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    log(
        "[CACHE SAVE]",
        sid,
        rid,
    )

    return {
        "metadata": metadata,
        "rationale_scores": rationale_scores,
        "rationale_retention": (
            rationale_retention
        ),
        "null_scores": null_scores,
        "null_retention": (
            null_retention
        ),
        "null_fragility": (
            null_fragility
        ),
        "rationale_gene_ids": np.asarray(
            rationale_gene_ids,
            dtype=np.int64,
        ),
        "null_gene_ids": null_gene_ids,
    }


# ============================================================
# 12) AGGREGATION
# ============================================================

def sod2_aggregate_results(
    cell_results: list[dict],
) -> dict:
    metadata_table = pd.DataFrame(
        [
            result[
                "metadata"
            ]
            for result in cell_results
        ]
    )

    rationale_retention = np.stack(
        [
            result[
                "rationale_retention"
            ]
            for result in cell_results
        ],
        axis=0,
    )

    null_retention = np.stack(
        [
            result[
                "null_retention"
            ]
            for result in cell_results
        ],
        axis=0,
    )

    null_fragility = np.stack(
        [
            result[
                "null_fragility"
            ]
            for result in cell_results
        ],
        axis=0,
    )

    rationale_fragility = metadata_table[
        "rationale_fragility_core"
    ].to_numpy(
        dtype=np.float64,
    )

    null_mean_per_cell = np.nanmean(
        null_fragility,
        axis=1,
    )

    delta_fragility = (
        rationale_fragility
        - null_mean_per_cell
    )

    true_mean_fragility = float(
        np.mean(
            rationale_fragility
        )
    )

    # Each null repeat is summarized across the exact same selected cells.
    null_repeat_mean_fragility = np.mean(
        null_fragility,
        axis=0,
    )

    global_empirical_p = sod2_empirical_upper_p(
        true_mean_fragility,
        null_repeat_mean_fragility,
    )

    metadata_table[
        "paired_delta_fragility"
    ] = delta_fragility

    metadata_table_path = (
        OUT_DIR
        / "SOD2_counterfactual_cell_level_summary.csv"
    )

    metadata_table.to_csv(
        metadata_table_path,
        index=False,
    )

    log(
        "[SAVE]",
        metadata_table_path,
    )

    curve_rows = []

    for cell_index, metadata_row in metadata_table.iterrows():
        for deletion_step in range(
            int(
                SOD2_DELETE_STEPS
            )
            + 1
        ):
            curve_rows.append(
                {
                    "sid": str(
                        metadata_row[
                            "sid"
                        ]
                    ),
                    "rid": int(
                        metadata_row[
                            "rid"
                        ]
                    ),
                    "CCI_cluster_position": int(
                        metadata_row[
                            "CCI_cluster_position"
                        ]
                    ),
                    "mode": "rationale",
                    "null_repeat": -1,
                    "deletion_step": int(
                        deletion_step
                    ),
                    "retention": float(
                        rationale_retention[
                            cell_index,
                            deletion_step
                        ]
                    ),
                }
            )

            for null_repeat in range(
                int(
                    SOD2_N_NULL_REPEATS
                )
            ):
                curve_rows.append(
                    {
                        "sid": str(
                            metadata_row[
                                "sid"
                            ]
                        ),
                        "rid": int(
                            metadata_row[
                                "rid"
                            ]
                        ),
                        "CCI_cluster_position": int(
                            metadata_row[
                                "CCI_cluster_position"
                            ]
                        ),
                        "mode": "matched_null",
                        "null_repeat": int(
                            null_repeat
                        ),
                        "deletion_step": int(
                            deletion_step
                        ),
                        "retention": float(
                            null_retention[
                                cell_index,
                                null_repeat,
                                deletion_step
                            ]
                        ),
                    }
                )

    curve_table_path = (
        OUT_DIR
        / "SOD2_counterfactual_retention_curves_long.csv"
    )

    pd.DataFrame(
        curve_rows
    ).to_csv(
        curve_table_path,
        index=False,
    )

    log(
        "[SAVE]",
        curve_table_path,
    )

    null_distribution_path = (
        OUT_DIR
        / "SOD2_counterfactual_matched_null_mean_fragility_distribution.csv"
    )

    pd.DataFrame(
        {
            "null_repeat": np.arange(
                int(
                    SOD2_N_NULL_REPEATS
                ),
                dtype=np.int64,
            ),
            "mean_fragility_across_cells": (
                null_repeat_mean_fragility
            ),
        }
    ).to_csv(
        null_distribution_path,
        index=False,
    )

    log(
        "[SAVE]",
        null_distribution_path,
    )

    transition_table = (
        metadata_table.groupby(
            [
                "associated_tf_before",
                "associated_tf_after",
            ],
            sort=True,
        )
        .agg(
            n_cells=(
                "rid",
                "size",
            ),
            mean_delta_fragility=(
                "paired_delta_fragility",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "n_cells",
                "mean_delta_fragility",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    transition_path = (
        OUT_DIR
        / "SOD2_counterfactual_associated_TF_transitions.csv"
    )

    transition_table.to_csv(
        transition_path,
        index=False,
    )

    log(
        "[SAVE]",
        transition_path,
    )

    summary = {
        "metadata_table": (
            metadata_table
        ),
        "rationale_retention": (
            rationale_retention
        ),
        "null_retention": (
            null_retention
        ),
        "rationale_fragility": (
            rationale_fragility
        ),
        "null_fragility": (
            null_fragility
        ),
        "null_repeat_mean_fragility": (
            null_repeat_mean_fragility
        ),
        "true_mean_fragility": (
            true_mean_fragility
        ),
        "global_empirical_p": (
            global_empirical_p
        ),
        "transition_table": (
            transition_table
        ),
    }

    return summary


# ============================================================
# 13) PLOTTING
# ============================================================

def sod2_plot_retention_curves(
    axis: plt.Axes,
    aggregate: dict,
) -> None:
    rationale_retention = aggregate[
        "rationale_retention"
    ]

    null_retention = aggregate[
        "null_retention"
    ]

    (
        rationale_mean,
        rationale_lower,
        rationale_upper,
    ) = sod2_bootstrap_mean_interval(
        rationale_retention,
        n_bootstrap=(
            SOD2_N_BOOTSTRAP
        ),
        seed=(
            SEED
            + 8001
        ),
    )

    # Null uncertainty is represented by the distribution of repeat-wise
    # mean curves across the same selected cells.
    null_repeat_mean_curves = np.nanmean(
        null_retention,
        axis=0,
    )

    null_mean = np.nanmean(
        null_repeat_mean_curves,
        axis=0,
    )

    null_lower = np.nanpercentile(
        null_repeat_mean_curves,
        2.5,
        axis=0,
    )

    null_upper = np.nanpercentile(
        null_repeat_mean_curves,
        97.5,
        axis=0,
    )

    x = np.arange(
        int(
            SOD2_DELETE_STEPS
        )
        + 1
    )

    axis.fill_between(
        x,
        null_lower,
        null_upper,
        color=SOD2_COLOR_NULL_LIGHT,
        alpha=0.72,
        linewidth=0,
        zorder=1,
    )

    axis.plot(
        x,
        null_mean,
        color=SOD2_COLOR_NULL,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Expression/state-matched null",
        zorder=2,
    )

    axis.fill_between(
        x,
        rationale_lower,
        rationale_upper,
        color=SOD2_COLOR_RATIONALE_LIGHT,
        alpha=0.68,
        linewidth=0,
        zorder=3,
    )

    axis.plot(
        x,
        rationale_mean,
        color=SOD2_COLOR_RATIONALE,
        linewidth=2.7,
        marker="o",
        markersize=4.0,
        label="MLTD-prioritized support genes",
        zorder=4,
    )

    axis.axhline(
        1.0
        - float(
            SOD2_TARGET_DROP_THRESHOLD
        ),
        color="#888888",
        linestyle=":",
        linewidth=0.9,
    )

    axis.set_xlabel(
        "Sequentially removed RNA support genes"
    )

    axis.set_ylabel(
        "Retained SOD2 target score"
    )

    axis.set_title(
        "A  SOD2 dependency-structure collapse",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
    )

    axis.set_xticks(
        x
    )

    axis.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper right",
    )

    sod2_style_axis(
        axis
    )


def sod2_plot_null_distribution(
    axis: plt.Axes,
    aggregate: dict,
) -> None:
    null_values = np.asarray(
        aggregate[
            "null_repeat_mean_fragility"
        ],
        dtype=np.float64,
    )

    observed = float(
        aggregate[
            "true_mean_fragility"
        ]
    )

    empirical_p = float(
        aggregate[
            "global_empirical_p"
        ]
    )

    axis.hist(
        null_values,
        bins=min(
            28,
            max(
                10,
                int(
                    np.sqrt(
                        null_values.size
                    )
                    * 2
                ),
            ),
        ),
        color=SOD2_COLOR_NULL_LIGHT,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.95,
    )

    axis.axvline(
        np.mean(
            null_values
        ),
        color=SOD2_COLOR_NULL,
        linestyle="--",
        linewidth=1.4,
        label="Null mean",
    )

    axis.axvline(
        observed,
        color=SOD2_COLOR_RATIONALE,
        linestyle="--",
        linewidth=2.2,
        label="Rationale deletion",
    )

    axis.text(
        0.03,
        0.96,
        (
            f"Observed mean={observed:.3f}\n"
            f"Empirical P={empirical_p:.3g}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
    )

    axis.set_xlabel(
        "Mean SOD2 fragility core score"
    )

    axis.set_ylabel(
        "Matched-null repetitions"
    )

    axis.set_title(
        "B  Rationale-specific SOD2 fragility",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
    )

    axis.legend(
        frameon=False,
        fontsize=7.4,
        loc="upper right",
    )

    sod2_style_axis(
        axis
    )


def sod2_transition_node_positions(
    labels: list[str],
) -> dict[str, np.ndarray]:
    angles = np.linspace(
        np.pi
        / 2.0,
        np.pi
        / 2.0
        - 2.0
        * np.pi,
        len(
            labels
        ),
        endpoint=False,
    )

    return {
        label: np.asarray(
            [
                np.cos(
                    angle
                ),
                np.sin(
                    angle
                ),
            ],
            dtype=np.float64,
        )
        for label, angle in zip(
            labels,
            angles,
        )
    }


def sod2_plot_tf_transitions(
    axis: plt.Axes,
    aggregate: dict,
) -> None:
    transition_table = aggregate[
        "transition_table"
    ].copy()

    metadata_table = aggregate[
        "metadata_table"
    ]

    switch_rate = float(
        metadata_table[
            "associated_tf_switched"
        ].mean()
    )

    if transition_table.shape[
        0
    ] == 0:
        axis.text(
            0.5,
            0.5,
            "No SOD2-associated TF transitions",
            ha="center",
            va="center",
        )

        axis.axis(
            "off"
        )

        return

    transition_table[
        "is_switch"
    ] = (
        transition_table[
            "associated_tf_before"
        ]
        != transition_table[
            "associated_tf_after"
        ]
    ).astype(
        int
    )

    node_weight: dict[str, float] = defaultdict(
        float
    )

    for _, row in transition_table.iterrows():
        node_weight[
            str(
                row[
                    "associated_tf_before"
                ]
            )
        ] += float(
            row[
                "n_cells"
            ]
        )

        node_weight[
            str(
                row[
                    "associated_tf_after"
                ]
            )
        ] += float(
            row[
                "n_cells"
            ]
        )

    labels = [
        label
        for label, _ in sorted(
            node_weight.items(),
            key=lambda item: item[
                1
            ],
            reverse=True,
        )[
            : int(
                SOD2_MAX_TRANSITION_NODES
            )
        ]
    ]

    transition_table = transition_table[
        transition_table[
            "associated_tf_before"
        ].isin(
            labels
        )
        & transition_table[
            "associated_tf_after"
        ].isin(
            labels
        )
    ].sort_values(
        [
            "is_switch",
            "n_cells",
            "mean_delta_fragility",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).head(
        int(
            SOD2_MAX_TRANSITION_EDGES
        )
    )

    positions = sod2_transition_node_positions(
        labels
    )

    maximum_edge_weight = max(
        float(
            transition_table[
                "n_cells"
            ].max()
        ),
        1.0,
    )

    maximum_node_weight = max(
        [
            node_weight[
                label
            ]
            for label in labels
        ]
        + [
            1.0
        ]
    )

    for _, row in transition_table.iterrows():
        source = str(
            row[
                "associated_tf_before"
            ]
        )

        target = str(
            row[
                "associated_tf_after"
            ]
        )

        number_of_cells = float(
            row[
                "n_cells"
            ]
        )

        source_position = positions[
            source
        ]

        target_position = positions[
            target
        ]

        if source == target:
            loop = Circle(
                source_position
                * 1.05,
                radius=0.12,
                fill=False,
                linewidth=(
                    0.8
                    + 2.5
                    * number_of_cells
                    / maximum_edge_weight
                ),
                edgecolor="#8B8B8B",
                alpha=0.55,
                zorder=1,
            )

            axis.add_patch(
                loop
            )

            continue

        angle_source = np.arctan2(
            source_position[
                1
            ],
            source_position[
                0
            ],
        )

        angle_target = np.arctan2(
            target_position[
                1
            ],
            target_position[
                0
            ],
        )

        angle_difference = (
            angle_target
            - angle_source
        )

        while angle_difference > np.pi:
            angle_difference -= (
                2.0
                * np.pi
            )

        while angle_difference < -np.pi:
            angle_difference += (
                2.0
                * np.pi
            )

        curvature = (
            0.20
            if angle_difference
            > 0
            else -0.20
        )

        arrow = FancyArrowPatch(
            source_position
            * 0.90,
            target_position
            * 0.90,
            connectionstyle=(
                f"arc3,rad={curvature}"
            ),
            arrowstyle="-|>",
            mutation_scale=(
                7.0
                + 5.0
                * number_of_cells
                / maximum_edge_weight
            ),
            linewidth=(
                0.9
                + 4.0
                * number_of_cells
                / maximum_edge_weight
            ),
            color=SOD2_COLOR_RATIONALE,
            alpha=(
                0.35
                + 0.50
                * number_of_cells
                / maximum_edge_weight
            ),
            zorder=1,
        )

        axis.add_patch(
            arrow
        )

    for label in labels:
        position = positions[
            label
        ]

        radius = (
            0.055
            + 0.035
            * node_weight[
                label
            ]
            / maximum_node_weight
        )

        node = Circle(
            position,
            radius=radius,
            facecolor=SOD2_COLOR_GOLD,
            edgecolor=SOD2_COLOR_TEXT,
            linewidth=0.9,
            zorder=3,
        )

        axis.add_patch(
            node
        )

        axis.text(
            position[
                0
            ]
            * 1.18,
            position[
                1
            ]
            * 1.18,
            label,
            ha="center",
            va="center",
            fontsize=7.7,
        )

    axis.text(
        0.02,
        0.97,
        (
            f"Actual switch rate={switch_rate:.1%}\n"
            f"N cells={metadata_table.shape[0]}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
    )

    axis.set_xlim(
        -1.40,
        1.40,
    )

    axis.set_ylim(
        -1.32,
        1.32,
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_title(
        "C  SOD2-associated TF-rank transitions",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
    )

    axis.axis(
        "off"
    )


def sod2_plot_triptych(
    aggregate: dict,
) -> None:
    figure = plt.figure(
        figsize=(
            13.4,
            4.3,
        )
    )

    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[
            1.15,
            1.0,
            1.0,
        ],
        wspace=0.34,
    )

    axis_a = figure.add_subplot(
        grid[
            0,
            0
        ]
    )

    axis_b = figure.add_subplot(
        grid[
            0,
            1
        ]
    )

    axis_c = figure.add_subplot(
        grid[
            0,
            2
        ]
    )

    sod2_plot_retention_curves(
        axis_a,
        aggregate,
    )

    sod2_plot_null_distribution(
        axis_b,
        aggregate,
    )

    sod2_plot_tf_transitions(
        axis_c,
        aggregate,
    )

    figure.suptitle(
        (
            "Counterfactual validation of the SOD2 MLTD dependency structure"
        ),
        fontsize=13.0,
        fontweight="bold",
        y=1.02,
    )

    figure.tight_layout()

    sod2_save_figure(
        figure,
        OUT_DIR
        / "SOD2_counterfactual_triptych",
    )

    plt.close(
        figure
    )

    for panel_name, plot_function in [
        (
            "SOD2_counterfactual_retention_curves",
            sod2_plot_retention_curves,
        ),
        (
            "SOD2_counterfactual_matched_null_fragility",
            sod2_plot_null_distribution,
        ),
        (
            "SOD2_counterfactual_associated_TF_transitions",
            sod2_plot_tf_transitions,
        ),
    ]:
        panel_figure, panel_axis = plt.subplots(
            figsize=(
                4.7,
                3.8,
            )
        )

        plot_function(
            panel_axis,
            aggregate,
        )

        panel_figure.tight_layout()

        sod2_save_figure(
            panel_figure,
            OUT_DIR
            / panel_name,
        )

        plt.close(
            panel_figure
        )


# ============================================================
# 14) MAIN
# ============================================================

def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    log(
        "[START]",
        "SOD2-specific counterfactual support-deletion analysis",
    )

    log(
        "[DEVICE]",
        DEVICE,
    )

    log(
        "[OUT DIR]",
        OUT_DIR,
    )

    explainer = FrontierExplainerStandalone()

    (
        target_global_id,
        target_local_pid,
        resolved_target_name,
    ) = sod2_find_target_gene_ids(
        explainer
    )

    log(
        "[TARGET]",
        (
            f"{resolved_target_name}: "
            f"global_id={target_global_id}, "
            f"local_pid={target_local_pid}"
        ),
    )

    selected_cells, retained_clusters = (
        sod2_load_selected_cells(
            explainer
        )
    )

    associated_tf_local_ids = (
        sod2_get_associated_tf_local_ids(
            explainer,
            target_global_id,
            target_local_pid,
        )
    )

    associated_tf_names = [
        str(
            explainer.tfs_vocab[
                int(
                    explainer.tf_in_prior[
                        int(
                            local_pid
                        )
                    ]
                )
            ]
        )
        for local_pid in associated_tf_local_ids.tolist()
    ]

    associated_tf_path = (
        OUT_DIR
        / "SOD2_associated_TF_candidates.csv"
    )

    pd.DataFrame(
        {
            "tf_local_pid": (
                associated_tf_local_ids
            ),
            "tf_global_id": [
                int(
                    explainer.tf_in_prior[
                        int(
                            local_pid
                        )
                    ]
                )
                for local_pid in associated_tf_local_ids.tolist()
            ],
            "tf_name": associated_tf_names,
        }
    ).to_csv(
        associated_tf_path,
        index=False,
    )

    log(
        "[SAVE]",
        associated_tf_path,
    )

    cell_results = []
    skipped_rows = []

    for cell_index, row in selected_cells.iterrows():
        try:
            result = sod2_compute_one_cell(
                explainer=explainer,
                row=row,
                cell_index=int(
                    cell_index
                ),
                target_global_id=(
                    target_global_id
                ),
                target_local_pid=(
                    target_local_pid
                ),
                associated_tf_local_ids=(
                    associated_tf_local_ids
                ),
            )

            cell_results.append(
                result
            )

            log(
                "[CELL DONE]",
                (
                    f"{cell_index + 1}/"
                    f"{selected_cells.shape[0]} "
                    f"{row['sid']}/{int(row['rid'])}"
                ),
            )

        except Exception as error:
            skipped_rows.append(
                {
                    "sid": str(
                        row[
                            "sid"
                        ]
                    ),
                    "rid": int(
                        row[
                            "rid"
                        ]
                    ),
                    "CCI_cluster_position": int(
                        row[
                            "CCI_cluster_position"
                        ]
                    ),
                    "error": str(
                        error
                    ),
                }
            )

            log(
                "[CELL SKIP]",
                (
                    f"{row['sid']}/{int(row['rid'])}: "
                    f"{error}"
                ),
            )

    if len(
        skipped_rows
    ) > 0:
        skipped_path = (
            OUT_DIR
            / "SOD2_counterfactual_skipped_cells.csv"
        )

        pd.DataFrame(
            skipped_rows
        ).to_csv(
            skipped_path,
            index=False,
        )

        log(
            "[SAVE]",
            skipped_path,
        )

    if len(
        cell_results
    ) < int(
        SOD2_MIN_TOTAL_CELLS
    ):
        raise RuntimeError(
            (
                f"Only {len(cell_results)} cells completed the SOD2 "
                f"counterfactual analysis; minimum required is "
                f"{SOD2_MIN_TOTAL_CELLS}."
            )
        )

    aggregate = sod2_aggregate_results(
        cell_results
    )

    sod2_plot_triptych(
        aggregate
    )

    manifest = {
        "dataset": DATASET_NAME,
        "target_gene": (
            resolved_target_name
        ),
        "base_checkpoint": str(
            BASE_CKPT
        ),
        "frontier_checkpoint": str(
            FRONTIER_CKPT
        ),
        "hotspot_cell_table": str(
            SOD2_HOTSPOT_CELL_TABLE
        ),
        "retained_source_clusters": [
            int(
                value
            )
            for value in retained_clusters
        ],
        "number_of_completed_cells": int(
            len(
                cell_results
            )
        ),
        "number_of_skipped_cells": int(
            len(
                skipped_rows
            )
        ),
        "deletion_steps": int(
            SOD2_DELETE_STEPS
        ),
        "null_repeats": int(
            SOD2_N_NULL_REPEATS
        ),
        "matched_null_features": [
            "projected_RNA_value",
            "cell_state_support",
        ],
        "fragility_core_weights": {
            "early_drop": float(
                SOD2_FRAG_W_EARLY
            ),
            "auc_loss": float(
                SOD2_FRAG_W_AUC
            ),
            "maximum_drop": float(
                SOD2_FRAG_W_MAX
            ),
            "final_loss": float(
                SOD2_FRAG_W_FINAL
            ),
            "early_event": float(
                SOD2_FRAG_W_EVENT
            ),
            "flip_bonus": 0.0,
        },
        "true_mean_fragility": float(
            aggregate[
                "true_mean_fragility"
            ]
        ),
        "global_empirical_p": float(
            aggregate[
                "global_empirical_p"
            ]
        ),
        "associated_tf_candidates": (
            associated_tf_names
        ),
        "interpretation": (
            "Deleting MLTD-prioritized RNA support genes is compared with "
            "abundance- and cell-state-matched random deletion in the same "
            "SOD2 dependency-sensitive cells. The analysis provides "
            "counterfactual structural-necessity evidence, not direct "
            "experimental causality."
        ),
    }

    manifest_path = (
        OUT_DIR
        / "SOD2_counterfactual_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    log(
        "[SAVE]",
        manifest_path,
    )

    log(
        "[DONE]",
        (
            "SOD2-specific counterfactual support-deletion analysis "
            "completed."
        ),
    )


if __name__ == "__main__":
    main()
