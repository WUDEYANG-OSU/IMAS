from __future__ import annotations

import hashlib
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0) CONFIG
# ============================================================
DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"

GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"

RUN_BASE = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)

FRONTIER_CKPT = RUN_BASE / "frontier_best.pt"
CENTROIDS_NPY = RUN_BASE / "centroids.npy"
CLUSTERS_CSV = RUN_BASE / "clusters.csv"
BASE_CKPT = get_dataset_checkpoint("CRC-Tissue")

OUT_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_base_frontier_linear_roc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- seeds ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Need CUDA"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- encoding ----------
RNA_TOPK = 2048
EMB_BATCH = 256
AUCELL_TOPK = 500

# ---------- TF proxy / ROC ----------
TF_SAMPLE = 128
TF_MIX_ALPHA = 0.6
POS_Q = 0.80
NEG_Q = 0.20

OVERALL_MIN_KEEP = 256
OVERALL_MIN_POS = 64
OVERALL_MIN_NEG = 64

CLUSTER_MIN_KEEP_ABS = 60
CLUSTER_MIN_KEEP_FRAC = 0.20
CLUSTER_MIN_POS = 12
CLUSTER_MIN_NEG = 12

FPR_GRID_N = 400
SMOOTH_SIGMA = 1.6

# ---------- linear ridge ----------
RIDGE_LAMBDA = 1e-2

# ---------- new clustering: aligned to true RNA UMAP ----------
N_TOP_HVG = 2000
N_PCS = 20
N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.03
UMAP_SPREAD = 2.0
LEIDEN_RESOLUTION = 0.5

# ---------- plot ----------
FIG_DPI = 220
POINT_SIZE_UMAP = 6
MAX_CLUSTERS_TO_PLOT = 20

# ---------- outputs ----------
UMAP_NEW_CLUSTER_PNG = OUT_DIR / f"{SAMPLE_ID}_umap_new_leiden_res0p5.png"
NEW_CLUSTER_ASSIGN_CSV = OUT_DIR / f"{SAMPLE_ID}_new_leiden_cluster_assignments.csv"
NEW_CLUSTER_SUMMARY_CSV = OUT_DIR / f"{SAMPLE_ID}_new_leiden_cluster_summary.csv"

ROC_OVERALL_COMPARE_PNG = OUT_DIR / f"{SAMPLE_ID}_roc_overall_base_frontier_linear.png"
ROC_CLUSTER_BASE_PNG = OUT_DIR / f"{SAMPLE_ID}_roc_newclusters_overlay_base.png"
ROC_CLUSTER_FRONTIER_PNG = OUT_DIR / f"{SAMPLE_ID}_roc_newclusters_overlay_frontier.png"
ROC_CLUSTER_LINEAR_PNG = OUT_DIR / f"{SAMPLE_ID}_roc_newclusters_overlay_linear.png"

TF_AUC_BASE_CSV = OUT_DIR / f"{SAMPLE_ID}_tf_auc_overall_base.csv"
TF_AUC_FRONTIER_CSV = OUT_DIR / f"{SAMPLE_ID}_tf_auc_overall_frontier.csv"
TF_AUC_LINEAR_CSV = OUT_DIR / f"{SAMPLE_ID}_tf_auc_overall_linear.csv"


def log(*args):
    print(*args, flush=True)


# ============================================================
# 1) HELPERS
# ============================================================
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


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (s + 1e-6)


def topk_from_dense_row(row: np.ndarray, topk: int):
    n = row.size
    k = min(topk, n)
    if k <= 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
    idx = np.argpartition(row, -k)[-k:]
    vals = row[idx]
    order = np.argsort(vals)[::-1]
    return idx[order].astype(np.int64), vals[order].astype(np.float32)


def make_pos_neg_labels(scores: np.ndarray, pos_q: float, neg_q: float):
    scores = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(scores)
    s = scores[finite]
    if s.size < 10:
        y = np.full(scores.shape, -1, dtype=np.int8)
        keep = np.zeros(scores.shape, dtype=bool)
        return y, keep
    hi = np.quantile(s, pos_q)
    lo = np.quantile(s, neg_q)
    y = np.full(scores.shape, -1, dtype=np.int8)
    y[(scores >= hi) & finite] = 1
    y[(scores <= lo) & finite] = 0
    keep = (y >= 0)
    return y, keep


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores).reshape(-1)
    neg = np.asarray(neg_scores).reshape(-1)
    if pos.size < 1 or neg.size < 1:
        return np.nan
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    r_pos = ranks[:pos.size]
    auc = (r_pos.sum() - pos.size * (pos.size - 1) / 2) / (pos.size * neg.size + 1e-12)
    return float(auc)


def roc_curve_from_scores(y_true: np.ndarray, scores: np.ndarray):
    y = np.asarray(y_true, np.int8).reshape(-1)
    s = np.asarray(scores, np.float64).reshape(-1)
    m = np.isfinite(s) & (y >= 0)
    y = y[m]
    s = s[m]
    if y.size < 50:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-s)
    y = y[order]
    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P < 1 or N < 1:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(y == 1).astype(np.float64)
    fps = np.cumsum(y == 0).astype(np.float64)
    tpr = tps / P
    fpr = fps / N
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr


def interp_to_grid(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray):
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    return np.interp(grid, fpr, tpr)


def smooth_1d(y: np.ndarray, sigma: float = 1.6):
    if sigma <= 0:
        return y
    radius = int(max(3, round(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def fix_roc_endpoints_and_monotonic(tpr: np.ndarray):
    t = np.asarray(tpr, np.float64).copy()
    t[0] = 0.0
    t[-1] = 1.0
    t = np.clip(t, 0.0, 1.0)
    t = np.maximum.accumulate(t)
    t[0] = 0.0
    t[-1] = 1.0
    return t


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


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
    def __init__(self, d: int, k: int, n_tf: int, n_gene: int, peak_hash_size: int, n_cancer: int, n_group: int,
                 e_dim: int = 64, hid: int = 256):
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

    def delta_cancer(self, cancer_id: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        return self.C_cancer(cancer_id) @ V_cancer

    def apply_group_to_Z(self, Z: torch.Tensor, group_id: torch.Tensor, group_scale_alpha: float) -> torch.Tensor:
        s = self.S_group(group_id)
        b = self.B_group(group_id)
        return Z * (1.0 + float(group_scale_alpha) * torch.tanh(s)) + b


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


# ============================================================
# 3) DATA OBJECTS
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


def load_single_sample(dataset_name: str, sample_id: str, genes_vocab: list[str]) -> Sample:
    manifest_rna_dir = WORLD_ROOT / dataset_name / "processed" / "RNA"
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    df = pd.read_csv(manifest_path, sep="\t")
    g = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    assert len(g) > 0, f"No rows found for sample_id={sample_id}"
    g = g.sort_values("row_index").reset_index(drop=True)

    meta_path = manifest_rna_dir / f"{sample_id}.rna_meta.npz"
    counts_path = manifest_rna_dir / f"{sample_id}.X_rna_counts_dense.npy"
    proj_rna_path = PROJECTED_ROOT / dataset_name / f"{sample_id}.rna.global.csr.npz"

    z = np.load(meta_path, allow_pickle=True)
    gene_names = np.asarray(z["gene_names"]).astype(str)

    cells = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
    row_indices = g["row_index"].astype(np.int64).values

    counts_mm = np.load(counts_path, mmap_mode="r")

    gene2i_vocab = {g: i for i, g in enumerate(genes_vocab)}
    ext2train = np.full((len(gene_names),), -1, dtype=np.int64)
    train2rna = {}
    for j, gname in enumerate(gene_names):
        gname = normalize_gene_name(gname)
        if gname in gene2i_vocab:
            tid = gene2i_vocab[gname]
            ext2train[j] = tid
            if tid not in train2rna:
                train2rna[tid] = j

    return Sample(
        sid=sample_id,
        cells=cells,
        row_indices=row_indices,
        proj_rna_path=proj_rna_path,
        counts_path=counts_path,
        counts_shape=tuple(counts_mm.shape),
        gene_names=gene_names,
        ext2train=ext2train,
        train2rna=train2rna,
    )


def get_counts_row_dense(sample: Sample, i: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    row = np.asarray(x[i]).reshape(-1)
    return row.astype(np.float32, copy=False)


# ============================================================
# 4) TF PROXY / ENCODING
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
        gi, gv = proj_store_cache[key].get_row_topk(rid, RNA_TOPK)
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


def regproxy_aucell_lite_from_top(cell_list, sid2sample, tf2targets, tf_in_prior, topk, N_GENE_DIM):
    n_tf = len(tf_in_prior)
    out = np.full((len(cell_list), n_tf), np.nan, dtype=np.float32)
    tf_targets = [set(int(g) for g in tf2targets[int(tf_id)]) for tf_id in tf_in_prior]
    for i, (sid, rid) in enumerate(cell_list):
        s = sid2sample[sid]
        row = get_counts_row_dense(s, rid)
        gi, _ = topk_from_dense_row(row, topk)
        train_idx = s.ext2train[gi]
        m = (train_idx >= 0) & (train_idx < N_GENE_DIM)
        top_train = set(int(x) for x in train_idx[m].tolist())
        if len(top_train) == 0:
            continue
        for tfp, tgtset in enumerate(tf_targets):
            out[i, tfp] = float(len(top_train & tgtset)) / float(topk)
    return out


def regproxy_mean_log1p(cell_list, sid2sample, tf2targets, tf_in_prior):
    n_tf = len(tf_in_prior)
    out = np.full((len(cell_list), n_tf), np.nan, dtype=np.float32)
    sample2items = defaultdict(list)
    for i, (sid, rid) in enumerate(cell_list):
        sample2items[sid].append((i, rid))

    for sid, items in sample2items.items():
        s = sid2sample[sid]
        rows = []
        cols = []
        data = []
        for tf_pid, tf_id in enumerate(tf_in_prior):
            tgs = tf2targets[int(tf_id)]
            idxs = [s.train2rna[g] for g in tgs if g in s.train2rna]
            if not idxs:
                continue
            w = 1.0 / float(len(idxs))
            rows.extend(idxs)
            cols.extend([tf_pid] * len(idxs))
            data.extend([w] * len(idxs))

        if not data:
            continue

        W = sp.coo_matrix(
            (np.asarray(data, np.float32), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
            shape=(int(s.counts_shape[1]), int(n_tf))
        ).tocsr()

        rids = [rid for _, rid in items]
        Xmm = np.load(s.counts_path, mmap_mode="r")
        X = np.asarray(Xmm[rids, :], dtype=np.float32)
        X = sp.csr_matrix(X)
        X = X.copy()
        if X.nnz > 0:
            X.data = np.clip(X.data, 0.0, None)
            X.data = np.log1p(X.data)

        REG = X @ W
        REG = REG.toarray().astype(np.float32) if sp.issparse(REG) else np.asarray(REG, np.float32)
        grow = [gi for gi, _ in items]
        out[np.asarray(grow, np.int64), :] = REG

    return out


def tfproxy_mix(aucell: np.ndarray, meanlog: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(aucell, np.nan, dtype=np.float32)
    for j in range(aucell.shape[1]):
        out[:, j] = float(alpha) * zscore_safe(aucell[:, j]) + (1.0 - float(alpha)) * zscore_safe(meanlog[:, j])
    return out


def spherical_kmeans_assign(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    ZN = _l2norm_rows(Z)
    CN = _l2norm_rows(C)
    sim = ZN @ CN.T
    return sim.argmax(axis=1).astype(np.int64)


# ============================================================
# 5) TRUE-UMAP-ALIGNED CLUSTERING
# ============================================================
def build_adata_for_scanpy(sample: Sample) -> ad.AnnData:
    Xmm = np.load(sample.counts_path, mmap_mode="r")
    X = np.asarray(Xmm, dtype=np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.var_names = np.asarray(sample.gene_names, dtype=str)
    adata.obs_names = np.asarray([make_cell_key(sample.sid, bc) for bc in sample.cells.tolist()], dtype=object)
    adata.obs["sid"] = sample.sid
    adata.obs["barcode"] = sample.cells.astype(str)
    return adata


def run_single_sample_leiden(adata: ad.AnnData):
    adata = adata.copy()

    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(N_TOP_HVG, adata.n_vars),
        flavor="seurat_v3",
        subset=False,
    )
    hvg_mask = adata.var["highly_variable"].values.copy()
    if hvg_mask.sum() == 0:
        hvg_mask[:] = True

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata = adata[:, hvg_mask].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(
        adata,
        n_comps=min(N_PCS, max(2, adata.n_vars - 1)),
        svd_solver="arpack",
        random_state=SEED,
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        n_pcs=min(N_PCS, adata.obsm["X_pca"].shape[1]),
        random_state=SEED,
    )
    sc.tl.umap(
        adata,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=SEED,
    )
    sc.tl.leiden(
        adata,
        resolution=LEIDEN_RESOLUTION,
        key_added="new_cluster",
        random_state=SEED,
    )

    n_clusters = adata.obs["new_cluster"].astype(str).nunique()
    log(f"[LEIDEN] fixed resolution={LEIDEN_RESOLUTION}, n_clusters={n_clusters}")
    return adata, LEIDEN_RESOLUTION, int(n_clusters)


def save_newcluster_umap(adata: ad.AnnData, out_png: Path):
    xy = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    lab = adata.obs["new_cluster"].astype(str).values
    uniq = sorted(pd.unique(lab).tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    fig = plt.figure(figsize=(7.2, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)

    for c in uniq:
        m = (lab == c)
        ax.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE_UMAP, alpha=0.85, label=f"C{c}")

    for c in uniq:
        m = (lab == c)
        if int(m.sum()) == 0:
            continue
        cx = float(np.median(xy[m, 0]))
        cy = float(np.median(xy[m, 1]))
        ax.text(
            cx, cy, str(c),
            fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax.set_title(f"True RNA UMAP | {SAMPLE_ID} | Leiden(res={LEIDEN_RESOLUTION})", fontsize=13)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", out_png)


# ============================================================
# 6) ROC CORE
# ============================================================
def compute_mean_roc_across_tfs(
    pred: np.ndarray,
    reg: np.ndarray,
    tf_ids: np.ndarray,
    pos_q: float,
    neg_q: float,
    min_keep: int,
    min_pos: int,
    min_neg: int,
):
    grid = np.linspace(0, 1, FPR_GRID_N, dtype=np.float64)
    aucs = []
    tpr_stack = []
    used_tf = 0
    per_tf_rows = []

    for j, tfp in enumerate(tf_ids.tolist()):
        regj = reg[:, int(tfp)]
        y, keep = make_pos_neg_labels(regj, pos_q, neg_q)
        n_keep = int(keep.sum())
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_keep < min_keep or n_pos < min_pos or n_neg < min_neg:
            continue

        scores = pred[:, j]
        auc0 = auc_from_scores(scores[y == 1], scores[y == 0])
        if not np.isfinite(auc0):
            continue

        fpr, tpr = roc_curve_from_scores(y, scores)
        tpr_stack.append(interp_to_grid(fpr, tpr, grid))
        aucs.append(float(auc0))
        used_tf += 1
        per_tf_rows.append({
            "tf_pid": int(tfp),
            "auc": float(auc0),
            "n_keep": n_keep,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })

    if used_tf < 1:
        return None

    tpr_mean = np.mean(np.stack(tpr_stack, axis=0), axis=0)
    tpr_mean = smooth_1d(tpr_mean, sigma=SMOOTH_SIGMA)
    tpr_mean = fix_roc_endpoints_and_monotonic(tpr_mean)
    mean_auc = float(np.mean(np.asarray(aucs, np.float32)))

    return {
        "grid": grid,
        "tpr_mean": tpr_mean,
        "mean_auc": mean_auc,
        "used_tf": int(used_tf),
        "per_tf": pd.DataFrame(per_tf_rows),
    }


# ============================================================
# 7) RIDGE FIT
# ============================================================
@torch.no_grad()
def fit_and_predict_linear_ridge(
    Z_train: np.ndarray,
    REG_train_tf: np.ndarray,
    Z_eval: np.ndarray,
    tf_ids: np.ndarray,
    ridge_lambda: float,
):
    Xtr = torch.from_numpy(Z_train).to(DEVICE, dtype=torch.float32)
    Xev = torch.from_numpy(Z_eval).to(DEVICE, dtype=torch.float32)

    pred_eval = np.full((Z_eval.shape[0], len(tf_ids)), np.nan, dtype=np.float32)

    I = None
    d = Xtr.shape[1]

    for j_out, tfp in enumerate(tf_ids.tolist()):
        ytr = REG_train_tf[:, int(tfp)].astype(np.float32)
        mtr = np.isfinite(ytr)
        if int(mtr.sum()) < 200:
            continue

        Xsub = Xtr[mtr]
        ysub = torch.from_numpy(ytr[mtr]).to(DEVICE, dtype=torch.float32)

        if I is None:
            I = torch.eye(d, device=DEVICE, dtype=torch.float32)

        XtX = Xsub.t() @ Xsub
        Xty = Xsub.t() @ ysub
        w = torch.linalg.solve(XtX + float(ridge_lambda) * I, Xty)
        pred_eval[:, j_out] = (Xev @ w).detach().cpu().numpy().astype(np.float32)

    return pred_eval


# ============================================================
# 8) PRIOR READERS
# ============================================================
def read_tf_prior(prior_dir: Path, tfs_vocab: list[str], N_GENE_DIM: int):
    tf_gene_prior = prior_dir / "tf_gene_f.cap256.npz"
    if not tf_gene_prior.exists():
        tf_gene_prior = prior_dir / "tf_gene_f.full.npz"
    assert tf_gene_prior.exists(), f"Missing tf-gene prior: {tf_gene_prior}"

    z = np.load(tf_gene_prior, allow_pickle=True)
    tf_src = z["src"].astype(np.int64)
    g_dst = z["dst"].astype(np.int64)

    tf2targets = defaultdict(list)
    for s, d0 in zip(tf_src.tolist(), g_dst.tolist()):
        if 0 <= s < len(tfs_vocab) and 0 <= d0 < N_GENE_DIM:
            tf2targets[int(s)].append(int(d0))
    tf_in_prior = sorted(tf2targets.keys())
    log("[PRIOR] tf_in_prior =", len(tf_in_prior))
    return tf2targets, tf_in_prior


def read_gene_peak_prior(prior_dir: Path, N_GENE_DIM: int):
    gene_peak_prior = find_prior_file(prior_dir, ["gene", "peak"])
    assert gene_peak_prior is not None, f"Cannot find gene-peak prior in {prior_dir}"
    z2 = np.load(gene_peak_prior, allow_pickle=True)
    gp_src = z2["src"].astype(np.int64)
    gp_dst = z2["dst"].astype(np.int64)

    src_max = int(gp_src.max()) if gp_src.size else -1
    gene2peaks = defaultdict(list)
    if src_max < N_GENE_DIM:
        for g, p in zip(gp_src.tolist(), gp_dst.tolist()):
            if 0 <= g < N_GENE_DIM:
                gene2peaks[int(g)].append(int(p))
    else:
        for p, g in zip(gp_src.tolist(), gp_dst.tolist()):
            if 0 <= g < N_GENE_DIM:
                gene2peaks[int(g)].append(int(p))

    for g in list(gene2peaks.keys()):
        if len(gene2peaks[g]) > 64:
            gene2peaks[g] = gene2peaks[g][:64]

    genes_in_peakprior = sorted(gene2peaks.keys())
    genes_union = sorted(set(genes_in_peakprior))
    return gene2peaks, genes_union


# ============================================================
# 9) MAIN
# ============================================================
def main():
    assert GLOBAL_GENE_VOCAB.exists()
    assert GLOBAL_TF_VOCAB.exists()
    assert FRONTIER_CKPT.exists()
    assert CENTROIDS_NPY.exists()
    assert CLUSTERS_CSV.exists()
    assert BASE_CKPT.exists()

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    # ---------- base ckpt ----------
    base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = dict(base_ckpt.get("cfg", {}) or {})
    meta = base_ckpt.get("meta", {})
    state_dict = base_ckpt["model"]

    if isinstance(meta, dict):
        if "N_GENE_DIM" in meta:
            cfg["N_GENE_DIM"] = meta["N_GENE_DIM"]
        if "N_TF_DIM" in meta:
            cfg["N_TF_DIM"] = meta["N_TF_DIM"]
        if "PEAK_HASH_SIZE" in meta:
            cfg["PEAK_HASH_SIZE"] = meta["PEAK_HASH_SIZE"]
        if "D" in meta:
            cfg["D"] = meta["D"]
        if "HID" in meta:
            cfg["HID"] = meta["HID"]
        if "DROP" in meta:
            cfg["DROP"] = meta["DROP"]
        if "cancer_types" in meta:
            cfg["cancer_types"] = meta["cancer_types"]

    cfg.setdefault("N_GENE_DIM", int(state_dict["gene_in.weight"].shape[0]))
    cfg.setdefault("N_TF_DIM", int(state_dict["tf_in.weight"].shape[0]))
    cfg.setdefault("PEAK_HASH_SIZE", int(state_dict["peak_in.emb.weight"].shape[0]))
    cfg.setdefault("D", int(state_dict["gene_in.weight"].shape[1]))
    cfg.setdefault("HID", int(state_dict["rna_proj.0.weight"].shape[0]))
    cfg.setdefault("DROP", 0.1)

    if "cancer_types" not in cfg:
        n_cancer = int(state_dict["cell_c.weight"].shape[0])
        cfg["cancer_types"] = ["CRC"] + [f"unused_{i}" for i in range(1, n_cancer)]

    N_GENE_DIM = int(cfg["N_GENE_DIM"])
    N_TF_DIM = int(cfg["N_TF_DIM"])
    PEAK_HASH_SIZE = int(cfg["PEAK_HASH_SIZE"])
    D = int(cfg["D"])
    HID = int(cfg["HID"])
    DROP = float(cfg["DROP"])
    cancer_types = list(cfg["cancer_types"])

    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_id = int(cancer2id.get("CRC", 0))
    cid = torch.tensor([default_cancer_id], device=DEVICE, dtype=torch.long)

    model = PlanC_DirAware_Cancer_HashPeak(
        n_gene=N_GENE_DIM,
        n_tf=N_TF_DIM,
        n_cancer=len(cancer_types),
        peak_hash_size=PEAK_HASH_SIZE,
        d=D, hid=HID, drop=DROP,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---------- frontier ckpt ----------
    ckpt = torch.load(FRONTIER_CKPT, map_location="cpu", weights_only=False)
    hyper = ckpt.get("hyper", {})
    K_RANK = int(hyper.get("K_RANK", 32))
    GROUP_SCALE_ALPHA = float(hyper.get("GROUP_SCALE_ALPHA", 0.12))
    TF_MIX_ALPHA = float(hyper.get("TF_MIX_ALPHA", 0.6))
    USE_CLUSTER_CORRECTION = bool(hyper.get("USE_CLUSTER_CORRECTION", True))
    CLUSTER_CORRECTION_USE_RAW_CLUSTER = bool(hyper.get("CLUSTER_CORRECTION_USE_RAW_CLUSTER", True))

    cluster_summary_vecs = {int(k): np.asarray(v, dtype=np.float32) for k, v in ckpt["cluster_summary_vecs"].items()}
    cluster_summary_table = ClusterSummaryTable(cluster_summary_vecs).to(DEVICE)

    # ---------- infer n_group ----------
    df_clusters = pd.read_csv(CLUSTERS_CSV)
    cl2gid_counter = defaultdict(Counter)
    for _, r in df_clusters.iterrows():
        cl2gid_counter[int(r["cluster_id"])][int(r["group_id"])] += 1
    cl2gid = {int(cl): cnt.most_common(1)[0][0] for cl, cnt in cl2gid_counter.items()}
    n_group = int(max(cl2gid.values()) + 1) if len(cl2gid) else 1

    # ---------- priors ----------
    tf2targets, tf_in_prior = read_tf_prior(PRIORS_DIR, tfs_vocab, N_GENE_DIM)
    _, genes_union = read_gene_peak_prior(PRIORS_DIR, N_GENE_DIM)

    # ---------- adapter ----------
    adapter = HyperAdapterHashV2(
        d=D,
        k=K_RANK,
        n_tf=len(tf_in_prior),
        n_gene=len(genes_union),
        peak_hash_size=PEAK_HASH_SIZE,
        n_cancer=len(cancer_types),
        n_group=n_group,
        e_dim=64,
        hid=256,
    ).to(DEVICE)
    adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
    adapter.eval()

    tf_corr_head = None
    if USE_CLUSTER_CORRECTION and ckpt.get("tf_corr_head_state_dict", None) is not None:
        tf_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=256).to(DEVICE)
        tf_corr_head.load_state_dict(ckpt["tf_corr_head_state_dict"], strict=True)
        tf_corr_head.eval()

    @torch.no_grad()
    def tf_base(tf_pid: torch.Tensor) -> torch.Tensor:
        tf_ids = torch.tensor([tf_in_prior[i] for i in tf_pid.tolist()], device=DEVICE, dtype=torch.long)
        return model.tf_in(tf_ids) + model.tf_in_c(cid).expand(tf_ids.numel(), -1)

    @torch.no_grad()
    def score_tf_vec(tf_pid_t: torch.Tensor, V_tf: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        TF = tf_base(tf_pid_t) + adapter.delta_tf(tf_pid_t, V_tf)
        c_tok = torch.full((tf_pid_t.numel(),), int(default_cancer_id), device=DEVICE, dtype=torch.long)
        TF = TF + adapter.delta_cancer(c_tok, V_cancer)
        return TF

    def cluster_ids_to_summary(cluster_ids_np: np.ndarray):
        cid_t = torch.from_numpy(cluster_ids_np.astype(np.int64)).to(DEVICE)
        return cluster_summary_table.lookup(cid_t)

    def apply_cluster_correction(z_in: torch.Tensor, cluster_ids_np: np.ndarray, alpha: float = 0.25):
        if tf_corr_head is None:
            return z_in
        summary_t = cluster_ids_to_summary(cluster_ids_np)
        delta = tf_corr_head(z_in, summary_t)
        return F.normalize(z_in + float(alpha) * delta, dim=-1)

    # ---------- sample ----------
    sample = load_single_sample(DATASET_NAME, SAMPLE_ID, genes_vocab)
    sid2sample = {sample.sid: sample}
    cell_list = [(sample.sid, int(i)) for i in range(sample.counts_shape[0])]

    # ---------- true-UMAP-aligned clusters ----------
    log("[SCANPY] building new single-sample clustering ...")
    adata = build_adata_for_scanpy(sample)
    adata, _, _ = run_single_sample_leiden(adata)
    save_newcluster_umap(adata, UMAP_NEW_CLUSTER_PNG)

    new_cluster_vals = adata.obs["new_cluster"].astype(str).values
    key_order_cluster = np.asarray(adata.obs_names.astype(str), dtype=object)

    # ---------- encode Z ----------
    log("[ENC] encoding Z ...")
    proj_store_cache = {}
    Z = encode_Z(cell_list, sid2sample, model, default_cancer_id, proj_store_cache)
    Zt = torch.from_numpy(Z).to(DEVICE)

    # ---------- train/val split inside this single sample ----------
    all_with_key = []
    for rid in range(sample.counts_shape[0]):
        key = make_cell_key(sample.sid, str(sample.cells[rid]))
        all_with_key.append((sample.sid, int(rid), key))

    tr_full, te_dummy = [], []
    for sid, rid, key in all_with_key:
        if int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) & 1 == 0:
            tr_full.append((sid, rid))
        else:
            te_dummy.append((sid, rid))
    train_cells, val_cells = [], []
    if len(tr_full) < 200:
        train_cells = [(sid, rid) for sid, rid, _ in tr_full]
        val_cells = []
    else:
        idx = np.arange(len(tr_full))
        rng_local_split = np.random.RandomState(SEED)
        rng_local_split.shuffle(idx)
        nval = max(1, int(len(tr_full) * 0.10))
        val_cells = [(tr_full[i][0], tr_full[i][1]) for i in idx[:nval]]
        train_cells = [(tr_full[i][0], tr_full[i][1]) for i in idx[nval:]]

    assert len(train_cells) > 0 and len(val_cells) > 0, "train/val split failed"

    log(f"[SPLIT] train={len(train_cells)} val={len(val_cells)}")

    # ---------- TF mixed proxy on all cells, then slice ----------
    log("[PROXY] TF mixed proxy on full sample ...")
    REG_au_all = regproxy_aucell_lite_from_top(cell_list, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM)
    REG_mn_all = regproxy_mean_log1p(cell_list, sid2sample, tf2targets, tf_in_prior)
    REG_tf_all = tfproxy_mix(REG_au_all, REG_mn_all, TF_MIX_ALPHA)

    # ---------- deterministic TF subset ----------
    tf_ids_all = np.arange(len(tf_in_prior), dtype=np.int64)
    h = int(hashlib.md5(str(SAMPLE_ID).encode("utf-8")).hexdigest()[:8], 16)
    rng_local = np.random.RandomState(SEED ^ h)
    if tf_ids_all.size > TF_SAMPLE:
        tf_ids = rng_local.choice(tf_ids_all, size=TF_SAMPLE, replace=False)
        tf_ids = np.sort(tf_ids)
    else:
        tf_ids = tf_ids_all.copy()

    # ---------- internal cluster/group for frontier inference ----------
    C = np.load(CENTROIDS_NPY).astype(np.float32)
    raw_cluster = spherical_kmeans_assign(Z, C)
    group_ids = np.asarray([cl2gid.get(int(c), 0) for c in raw_cluster.tolist()], dtype=np.int64)

    # ---------- align cluster table to model order ----------
    key_order_model = np.asarray([make_cell_key(sample.sid, sample.cells[i]) for _, i in cell_list], dtype=object)
    df_obs = pd.DataFrame({
        "cell_key": key_order_cluster.tolist(),
        "new_cluster": new_cluster_vals.tolist(),
        "umap1": np.asarray(adata.obsm["X_umap"][:, 0], dtype=np.float32),
        "umap2": np.asarray(adata.obsm["X_umap"][:, 1], dtype=np.float32),
    })
    df_model = pd.DataFrame({
        "cell_key": key_order_model.tolist(),
        "sid": sample.sid,
        "rid": np.arange(len(cell_list), dtype=np.int64),
        "barcode": sample.cells.astype(str),
        "internal_raw_cluster": raw_cluster.astype(np.int64),
        "internal_group_id": group_ids.astype(np.int64),
    })
    df_join = df_model.merge(df_obs, on="cell_key", how="left")
    assert df_join["new_cluster"].notna().all(), "new_cluster alignment failed"
    df_join.to_csv(NEW_CLUSTER_ASSIGN_CSV, index=False)

    # ---------- index helpers ----------
    rid_to_pos = {int(rid): i for i, (_, rid) in enumerate(cell_list)}
    train_idx = np.asarray([rid_to_pos[int(rid)] for _, rid in train_cells], dtype=np.int64)
    val_idx = np.asarray([rid_to_pos[int(rid)] for _, rid in val_cells], dtype=np.int64)

    Z_train = Z[train_idx]
    Z_val = Z[val_idx]
    Zt_val = torch.from_numpy(Z_val).to(DEVICE)

    REG_train_tf = REG_tf_all[train_idx]
    REG_val_tf = REG_tf_all[val_idx]

    val_new_cluster = df_join.iloc[val_idx]["new_cluster"].astype(str).values

    # ---------- base prediction on VAL ----------
    with torch.no_grad():
        tf_ids_real = torch.tensor([tf_in_prior[i] for i in tf_ids.tolist()], device=DEVICE, dtype=torch.long)
        TF_vec_base = model.tf_in(tf_ids_real) + model.tf_in_c(cid).expand(tf_ids_real.numel(), -1)
        pred_base_val = (Zt_val @ TF_vec_base.t()).detach().cpu().numpy().astype(np.float32)

    # ---------- frontier prediction on VAL ----------
    with torch.no_grad():
        V_tf, _, _, V_cancer = adapter()
        tf_pid_t = torch.from_numpy(tf_ids).to(DEVICE)
        TF_vec_frontier = score_tf_vec(tf_pid_t, V_tf, V_cancer)

        gid_t = torch.from_numpy(group_ids[val_idx].astype(np.int64)).to(DEVICE)
        cid_t = torch.full((len(val_idx),), int(default_cancer_id), device=DEVICE, dtype=torch.long)

        Zadj = Zt_val
        Zadj = adapter.apply_group_to_Z(Zadj, gid_t, GROUP_SCALE_ALPHA)
        Zadj = Zadj + adapter.delta_cancer(cid_t, V_cancer)
        Zadj = F.normalize(Zadj, dim=-1)

        cluster_for_corr = raw_cluster[val_idx] if CLUSTER_CORRECTION_USE_RAW_CLUSTER else group_ids[val_idx]
        Zadj = apply_cluster_correction(Zadj, cluster_for_corr, alpha=0.25)

        pred_frontier_val = (Zadj @ TF_vec_frontier.t()).detach().cpu().numpy().astype(np.float32)

    # ---------- linear ridge prediction on VAL ----------
    log(f"[LINEAR] fit ridge on train and predict val, lambda={RIDGE_LAMBDA}")
    pred_linear_val = fit_and_predict_linear_ridge(
        Z_train=Z_train,
        REG_train_tf=REG_train_tf,
        Z_eval=Z_val,
        tf_ids=tf_ids,
        ridge_lambda=RIDGE_LAMBDA,
    )

    # ---------- overall ROC ----------
    overall_base = compute_mean_roc_across_tfs(
        pred=pred_base_val,
        reg=REG_val_tf,
        tf_ids=tf_ids,
        pos_q=POS_Q,
        neg_q=NEG_Q,
        min_keep=OVERALL_MIN_KEEP,
        min_pos=OVERALL_MIN_POS,
        min_neg=OVERALL_MIN_NEG,
    )
    overall_frontier = compute_mean_roc_across_tfs(
        pred=pred_frontier_val,
        reg=REG_val_tf,
        tf_ids=tf_ids,
        pos_q=POS_Q,
        neg_q=NEG_Q,
        min_keep=OVERALL_MIN_KEEP,
        min_pos=OVERALL_MIN_POS,
        min_neg=OVERALL_MIN_NEG,
    )
    overall_linear = compute_mean_roc_across_tfs(
        pred=pred_linear_val,
        reg=REG_val_tf,
        tf_ids=tf_ids,
        pos_q=POS_Q,
        neg_q=NEG_Q,
        min_keep=OVERALL_MIN_KEEP,
        min_pos=OVERALL_MIN_POS,
        min_neg=OVERALL_MIN_NEG,
    )

    assert overall_base is not None, "Base: no TF passed overall filters."
    assert overall_frontier is not None, "Frontier: no TF passed overall filters."
    assert overall_linear is not None, "Linear ridge: no TF passed overall filters."

    overall_base["per_tf"].sort_values("auc", ascending=False).to_csv(TF_AUC_BASE_CSV, index=False)
    overall_frontier["per_tf"].sort_values("auc", ascending=False).to_csv(TF_AUC_FRONTIER_CSV, index=False)
    overall_linear["per_tf"].sort_values("auc", ascending=False).to_csv(TF_AUC_LINEAR_CSV, index=False)

    fig = plt.figure(figsize=(6.5, 6.1), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.plot(
        overall_base["grid"], overall_base["tpr_mean"],
        linewidth=2.4,
        label=f"Base AUC={overall_base['mean_auc']:.3f} (TF={overall_base['used_tf']})"
    )
    ax.plot(
        overall_frontier["grid"], overall_frontier["tpr_mean"],
        linewidth=2.4,
        label=f"Frontier AUC={overall_frontier['mean_auc']:.3f} (TF={overall_frontier['used_tf']})"
    )
    ax.plot(
        overall_linear["grid"], overall_linear["tpr_mean"],
        linewidth=2.4,
        label=f"Linear ridge AUC={overall_linear['mean_auc']:.3f} (TF={overall_linear['used_tf']})"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.65, label="Random")
    ax.set_title(f"ROC (RNA→TF, {SAMPLE_ID}, VAL-like split)", fontsize=13)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROC_OVERALL_COMPARE_PNG, bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", ROC_OVERALL_COMPARE_PNG)

    # ---------- per-cluster ROC ----------
    rows_summary = []
    curves_base = []
    curves_frontier = []
    curves_linear = []

    uniq_clusters = sorted(pd.unique(val_new_cluster).tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    for cl in uniq_clusters:
        m = (val_new_cluster == cl)
        ncell = int(m.sum())
        min_keep = max(CLUSTER_MIN_KEEP_ABS, int(CLUSTER_MIN_KEEP_FRAC * ncell))

        res_base = compute_mean_roc_across_tfs(
            pred=pred_base_val[m],
            reg=REG_val_tf[m],
            tf_ids=tf_ids,
            pos_q=POS_Q,
            neg_q=NEG_Q,
            min_keep=min_keep,
            min_pos=CLUSTER_MIN_POS,
            min_neg=CLUSTER_MIN_NEG,
        )
        res_frontier = compute_mean_roc_across_tfs(
            pred=pred_frontier_val[m],
            reg=REG_val_tf[m],
            tf_ids=tf_ids,
            pos_q=POS_Q,
            neg_q=NEG_Q,
            min_keep=min_keep,
            min_pos=CLUSTER_MIN_POS,
            min_neg=CLUSTER_MIN_NEG,
        )
        res_linear = compute_mean_roc_across_tfs(
            pred=pred_linear_val[m],
            reg=REG_val_tf[m],
            tf_ids=tf_ids,
            pos_q=POS_Q,
            neg_q=NEG_Q,
            min_keep=min_keep,
            min_pos=CLUSTER_MIN_POS,
            min_neg=CLUSTER_MIN_NEG,
        )

        rows_summary.append({
            "new_cluster": str(cl),
            "n_cells": ncell,
            "base_mean_auc": np.nan if res_base is None else float(res_base["mean_auc"]),
            "base_tf_used": 0 if res_base is None else int(res_base["used_tf"]),
            "frontier_mean_auc": np.nan if res_frontier is None else float(res_frontier["mean_auc"]),
            "frontier_tf_used": 0 if res_frontier is None else int(res_frontier["used_tf"]),
            "linear_mean_auc": np.nan if res_linear is None else float(res_linear["mean_auc"]),
            "linear_tf_used": 0 if res_linear is None else int(res_linear["used_tf"]),
        })

        if res_base is not None:
            curves_base.append((str(cl), ncell, float(res_base["mean_auc"]), res_base["grid"], res_base["tpr_mean"]))
        if res_frontier is not None:
            curves_frontier.append((str(cl), ncell, float(res_frontier["mean_auc"]), res_frontier["grid"], res_frontier["tpr_mean"]))
        if res_linear is not None:
            curves_linear.append((str(cl), ncell, float(res_linear["mean_auc"]), res_linear["grid"], res_linear["tpr_mean"]))

    df_sum = pd.DataFrame(rows_summary)
    df_sum.to_csv(NEW_CLUSTER_SUMMARY_CSV, index=False)
    log("[SAVE]", NEW_CLUSTER_SUMMARY_CSV)

    curves_base = sorted(curves_base, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])[:MAX_CLUSTERS_TO_PLOT]
    curves_frontier = sorted(curves_frontier, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])[:MAX_CLUSTERS_TO_PLOT]
    curves_linear = sorted(curves_linear, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])[:MAX_CLUSTERS_TO_PLOT]

    # base
    fig = plt.figure(figsize=(6.5, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    for cl, ncell, auc_m, g, tpr_m in curves_base:
        ax.plot(g, tpr_m, linewidth=2.0, label=f"C{cl} (n={ncell}) AUC={auc_m:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.65, label="Random")
    ax.set_title(f"Base ROC by new Leiden cluster ({SAMPLE_ID})", fontsize=13)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROC_CLUSTER_BASE_PNG, bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", ROC_CLUSTER_BASE_PNG)

    # frontier
    fig = plt.figure(figsize=(6.5, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    for cl, ncell, auc_m, g, tpr_m in curves_frontier:
        ax.plot(g, tpr_m, linewidth=2.0, label=f"C{cl} (n={ncell}) AUC={auc_m:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.65, label="Random")
    ax.set_title(f"Frontier ROC by new Leiden cluster ({SAMPLE_ID})", fontsize=13)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROC_CLUSTER_FRONTIER_PNG, bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", ROC_CLUSTER_FRONTIER_PNG)

    # linear
    fig = plt.figure(figsize=(6.5, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    for cl, ncell, auc_m, g, tpr_m in curves_linear:
        ax.plot(g, tpr_m, linewidth=2.0, label=f"C{cl} (n={ncell}) AUC={auc_m:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.65, label="Random")
    ax.set_title(f"Linear ridge ROC by new Leiden cluster ({SAMPLE_ID})", fontsize=13)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROC_CLUSTER_LINEAR_PNG, bbox_inches="tight")
    plt.close(fig)
    log("[SAVE]", ROC_CLUSTER_LINEAR_PNG)

    log("[DONE]")


if __name__ == "__main__":
    main()
