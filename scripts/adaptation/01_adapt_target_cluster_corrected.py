# Source: 5.建模代码3-动态选边后对选择边显示构建embedding-辅助功能改善优化-加上cluster聚类矫正.docx
# Extracted from the user-provided Word code; scientific logic preserved.
from __future__ import annotations

# ============================================================
# frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection.py
# ============================================================
# Stage-B frontier transfer with:
#   - HASH-compatible peak side
#   - frozen Stage-A backbone
#   - 3 transfer tasks:
#       1) RNA -> TF   (mixed proxy)
#       2) RNA -> ATAC (geneActivity proxy via peak-pool)
#       3) RNA -> RNA  (gene retrieval / recon)
#   - clustering + group adaptor + selector
#   - selected-peak explicit residual on top of hash peak representation
#   - optional cluster-specific correction heads:
#       * TF score correction
#       * RNA retrieval correction
#       * RE/ATAC correction
#   - marker prior placeholder for later annotation integration
#   - validation monitor + best checkpoint by validation TF AUC
#   - final QC / Spearman / retrieval evaluation on val/test
#
# Design notes
# ------------
# - Main model remains DATASET-LEVEL. We do NOT train a separate model per cluster.
# - Cluster correction is OPTIONAL and acts as a residual on cell representation:
#       Z_task = normalize(Z_dataset + alpha * Delta_cluster(Z_dataset, cluster_summary))
# - Different task heads can use different cluster-corrected cell residuals:
#       TF / RNA / ATAC each get their own head.
# - Marker list is not required now; marker loader is a placeholder and can be
#   filled later once marker priors are uploaded.
# ============================================================

import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp


# ============================================================
# 0) CONFIG
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
RUN_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_TSV = RUN_DIR / "summary_all_runs.tsv"
FAILED_TSV = RUN_DIR / "failed_runs.tsv"

BASE_CKPT_MAP: dict[str, Path] = {
    # "BC-Tissue": (get_project_root() / "runs/qc_lodo_planC_v7_global_hashpeak/BC-Tissue/ckpt_last_BASE_global_hashpeak.pt"),
}

RUN_ONLY_DATASET = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Need CUDA"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_CANCER_NAME = None

# ---------- data ----------
MAX_CELLS_PER_SAMPLE = None
VAL_FRAC_FROM_TRAIN = 0.10
ANCHOR_DTYPE = np.float16
RNA_TOPK = 2048
EMB_BATCH = 256
STORE_TOP_G = 512

# ---------- TF proxies ----------
AUCELL_TOPK = 500
TF_MIX_ALPHA = 0.6

# ---------- transfer training ----------
STEPS = 10_000
LR = 7e-4
K_RANK = 32
CELL_BATCH = 2000
TF_BATCH = 96
RNA_POS_PER_CELL = 32
RNA_NEG_PER_CELL = 32
RE_CELL_BATCH = 768
RE_GENE_BATCH = 32
RE_POOL_MAX = 192
RE_TOPK_FOR_POOL = 32
W_TF_BASE = 1.0
W_RNA_BASE = 0.40
W_RE_BASE = 1.0
MARGIN_TF = 0.2
MARGIN_RE = 0.2
LAMBDA_L2 = 5e-5
POS_Q = 0.80
NEG_Q = 0.20

# ---------- val monitor ----------
SAVE_BEST_BY_VAL = True
VAL_MONITOR_EVERY = 300
VAL_MONITOR_TF_SAMPLE = 128
VAL_CLUSTER_MIN_CELLS = 250
VAL_CLUSTER_MIN_POSNEG = 16
VAL_CLUSTER_TOPK_PRINT = 5
VAL_CLUSTER_BOTTOMK_PRINT = 3

# ---------- selector ----------
USE_SELECTOR = True
SELECTOR_SUMMARY_NCELLS = 3000
SELECTOR_L2 = 5e-3
SELECTOR_INIT_SCALE = 0.0
SELECTOR_SOFTPLUS_BIAS = 1e-6

# ---------- ATAC filter ----------
ATAC_FILTER_NCELLS = 8000
ATAC_FILTER_MIN_NNZ_FRAC = 0.05
ATAC_FILTER_MIN_STD_LOG1P = 0.15
ATAC_FILTER_MAX_GENES = None

# ---------- clustering ----------
N_CLUSTERS = 16
CLUSTER_TRAIN_MAX = 60000
CLUSTER_ITERS = 25
CLUSTER_NMIN_TRAIN = 100

# ---------- group adaptor ----------
USE_GROUP_SCALE_BIAS = True
GROUP_SCALE_ALPHA = 0.12

# ---------- stratified batching ----------
USE_STRATIFIED_BATCH = True
STRAT_MIN_GROUP_SIZE = 200
STRAT_PER_GROUP_MIN = 128

# ---------- toggles ----------
USE_CANCER_ADAPTOR = True
USE_CELLGROUP_ADAPTOR = True

# ---------- selected peak residual ----------
USE_SELECTED_PEAK_RESIDUAL = True
SELECTED_PEAK_MAX = 50_000
SELECTED_PEAK_MIN_EDGE_FREQ = 2
SELECTED_PEAK_RESIDUAL_L2 = 1e-5
SELECTED_PEAK_ONLY_FROM_FILTERED_GENES = True

# ---------- cluster correction ----------
USE_CLUSTER_CORRECTION = True
CLUSTER_CORRECTION_ALPHA_TF = 0.25
CLUSTER_CORRECTION_ALPHA_RNA = 0.25
CLUSTER_CORRECTION_ALPHA_RE = 0.25
CLUSTER_CORRECTION_HID = 256
CLUSTER_CORRECTION_L2 = 1e-5
CLUSTER_CORRECTION_USE_RAW_CLUSTER = True  # True=raw cluster_id, False=group_id

# ---------- marker prior placeholder ----------
USE_CLUSTER_ANNOTATION = True
MARKER_PRIOR_PATH = None  # later fill with uploaded marker file path
ANNOT_SUMMARY_MAX_GENES = 256

# ---------- final QC / eval ----------
QC_TF_SAMPLE = None
QC_RE_GENE_EVAL = 256
QC_RE_MIN_FINITE = 256
QC_GENE_CELL_EVAL = 1200
QC_GENE_HARD_NEG = True


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


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (s + 1e-6)


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


def make_pos_neg_labels_grouped(scores: np.ndarray, groups: np.ndarray, pos_q: float, neg_q: float, min_n: int = 30):
    scores = np.asarray(scores, dtype=np.float32)
    groups = np.asarray(groups)
    y = np.full(scores.shape, -1, dtype=np.int8)
    finite = np.isfinite(scores)
    for g in np.unique(groups):
        m = (groups == g) & finite
        if m.sum() < min_n:
            continue
        s = scores[m]
        hi = np.quantile(s, pos_q)
        lo = np.quantile(s, neg_q)
        y[m & (scores >= hi)] = 1
        y[m & (scores <= lo)] = 0
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


def spearmanr_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 3 or y.size < 3:
        return np.nan
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum()) + 1e-12
    return float((rx * ry).sum() / den)


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


def find_prior_file(prior_dir: Path, include_keys: list[str]):
    cands = sorted(prior_dir.glob("*.npz"))
    hits = []
    for p in cands:
        n = p.name.lower()
        ok = all(k in n for k in include_keys)
        if ok:
            hits.append(p)
    return hits[0] if hits else None


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def load_marker_prior(marker_prior_path: Path | None):
    """
    Placeholder loader for future marker prior upload.
    Expected later formats:
      - JSON dict: {"T_cell": ["CD3D","CD3E"], "B_cell": ["MS4A1"]}
      - Or custom parser you can replace here.

    Right now:
      - if path is None -> returns {}
      - if file exists and is json -> tries to read dict[str, list[str]]
      - else -> returns {} with warning
    """
    if marker_prior_path is None:
        return {}
    p = Path(marker_prior_path)
    if not p.exists():
        log(f"[MARKER][WARN] marker prior file not found: {p}")
        return {}
    try:
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            out = {}
            for k, v in obj.items():
                if isinstance(v, (list, tuple)):
                    out[str(k)] = [str(x).strip() for x in v if str(x).strip()]
            log(f"[MARKER] loaded labels={len(out)} from {p}")
            return out
    except Exception as e:
        log(f"[MARKER][WARN] failed loading {p}: {repr(e)}")
    log(f"[MARKER][WARN] unsupported marker prior format for now: {p}")
    return {}


# ============================================================
# 2) BACKBONE (HASH COMPATIBLE)
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


# ============================================================
# 3) INPUT OBJECTS
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


def get_counts_row_dense(sample: Sample, i: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    row = np.asarray(x[i]).reshape(-1)
    return row.astype(np.float32, copy=False)


def topk_from_dense_row(row: np.ndarray, topk: int):
    n = row.size
    k = min(topk, n)
    if k <= 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
    idx = np.argpartition(row, -k)[-k:]
    vals = row[idx]
    order = np.argsort(vals)[::-1]
    return idx[order].astype(np.int64), vals[order].astype(np.float32)


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


def load_samples(dataset_name: str, genes_vocab: list[str]):
    manifest_rna_dir = get_manifest_rna_dir(dataset_name)
    anchor_dir = get_anchor_dir(dataset_name)
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"

    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    assert anchor_dir.exists(), f"Missing anchor dir: {anchor_dir}"

    df = pd.read_csv(manifest_path, sep="\t")
    need_cols = ["cell_id", "sample_id", "row_index", "counts_npy"]
    for c in need_cols:
        assert c in df.columns, f"{manifest_path.name}: missing column {c}"

    gene2i_vocab = {g: i for i, g in enumerate(genes_vocab)}
    samples = []
    sid2sample = {}

    for sid, g in df.groupby("sample_id", sort=True):
        sid = str(sid).strip()
        g = g.sort_values("row_index").reset_index(drop=True)

        meta_path = manifest_rna_dir / f"{sid}.rna_meta.npz"
        counts_path = manifest_rna_dir / f"{sid}.X_rna_counts_dense.npy"
        proj_rna_path = PROJECTED_ROOT / dataset_name / f"{sid}.rna.global.csr.npz"
        anchor_path = anchor_dir / f"{sid}_ATAC_geneActivity_ALLGENES.npz"

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
        assert matched > 1000, f"{dataset_name}/{sid}: too few genes matched to global vocab: {matched}"

        anchor_gene2col, anchor_cells, anchor_X = load_anchor_all(anchor_path)
        m = {str(c): i for i, c in enumerate(anchor_cells.tolist())}
        rid2arow = np.full((len(cells),), -1, dtype=np.int64)
        for i, bc in enumerate(cells.tolist()):
            rid2arow[i] = int(m.get(str(bc), -1))
        n_matched_cells = int((rid2arow >= 0).sum())
        if n_matched_cells < len(cells):
            log(f"[WARN] {dataset_name}/{sid}: RNA cells={len(cells)} matched_to_anchor={n_matched_cells}")

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
        log(f"[LOAD] {dataset_name}/{sid} | cells={s.counts_shape[0]} genes={s.counts_shape[1]} matched_genes={matched} anchor_shape={anchor_X.shape}")

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


# ============================================================
# 4) TF / GENE / CLUSTER HELPERS
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


def precompute_top_genes(cell_list: list[tuple[str, int]], sid2sample: dict[str, Sample], store_top_g: int, N_GENE_DIM: int):
    G_top = np.full((len(cell_list), store_top_g), -1, dtype=np.int64)
    V_top = np.zeros((len(cell_list), store_top_g), dtype=np.float32)
    for i, (sid, rid) in enumerate(cell_list):
        s = sid2sample[sid]
        row = get_counts_row_dense(s, rid)
        gi, gv = topk_from_dense_row(row, store_top_g)
        train_idx = s.ext2train[gi]
        m = (train_idx >= 0) & (train_idx < N_GENE_DIM)
        train_idx = train_idx[m]
        gv = gv[m]
        k = min(store_top_g, train_idx.size)
        if k > 0:
            G_top[i, :k] = train_idx[:k]
            V_top[i, :k] = gv[:k]
    return G_top, V_top


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


def tfproxy_mix(aucell: np.ndarray, meanlog: np.ndarray) -> np.ndarray:
    out = np.full_like(aucell, np.nan, dtype=np.float32)
    for j in range(aucell.shape[1]):
        out[:, j] = TF_MIX_ALPHA * zscore_safe(aucell[:, j]) + (1.0 - TF_MIX_ALPHA) * zscore_safe(meanlog[:, j])
    return out


def spherical_kmeans_fit(Z_train: np.ndarray, k: int, iters: int = 25, max_train: int = 60000):
    Z = _l2norm_rows(Z_train)
    n = Z.shape[0]
    k = int(min(k, max(2, n)))
    if n > int(max_train):
        idx = np.random.choice(np.arange(n), size=int(max_train), replace=False)
        Zfit = Z[idx]
    else:
        Zfit = Z
    init_idx = np.random.choice(np.arange(Zfit.shape[0]), size=k, replace=False)
    C = Zfit[init_idx].copy()
    for _ in range(int(iters)):
        sim = Zfit @ C.T
        y = sim.argmax(axis=1).astype(np.int64)
        for j in range(k):
            m = (y == j)
            if m.sum() < 1:
                C[j] = Zfit[np.random.randint(0, Zfit.shape[0])]
            else:
                C[j] = Zfit[m].mean(axis=0)
        C = _l2norm_rows(C)
    return C


def spherical_kmeans_assign(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    ZN = _l2norm_rows(Z)
    CN = _l2norm_rows(C)
    sim = ZN @ CN.T
    return sim.argmax(axis=1).astype(np.int64)


def build_group_ids_from_clusters(train_cluster: np.ndarray, all_cluster: np.ndarray, nmin: int):
    cnt = np.bincount(train_cluster, minlength=int(train_cluster.max()) + 1)
    keep = np.where(cnt >= int(nmin))[0].tolist()
    keep = [int(x) for x in keep]
    cl2gid = {cl: i + 1 for i, cl in enumerate(sorted(keep))}
    group_ids = np.zeros_like(all_cluster, dtype=np.int64)
    for i, cl in enumerate(all_cluster.tolist()):
        group_ids[i] = int(cl2gid.get(int(cl), 0))
    return group_ids, cl2gid, cnt


# ============================================================
# 5) CLUSTER SUMMARY / ANNOTATION
# ============================================================
def build_cluster_summaries(
    cell_list: list[tuple[str, int]],
    sid2sample: dict[str, Sample],
    Z: np.ndarray,
    cluster_ids: np.ndarray,
    genes_vocab: list[str],
    marker_prior: dict[str, list[str]] | None = None,
):
    marker_prior = marker_prior or {}
    marker_labels = sorted(marker_prior.keys())
    gene2id = {g: i for i, g in enumerate(genes_vocab)}
    summaries = {}

    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue

        z_mean = Z[idx].mean(axis=0).astype(np.float32)
        rna_nnz = []
        rna_logmean = []

        if len(marker_labels) > 0:
            marker_scores = np.zeros((len(marker_labels),), dtype=np.float32)
        else:
            marker_scores = np.zeros((0,), dtype=np.float32)

        for ii in idx.tolist():
            sid, rid = cell_list[ii]
            s = sid2sample[sid]
            row = get_counts_row_dense(s, rid)
            rna_nnz.append(float((row > 0).mean()))
            nz = row[row > 0]
            rna_logmean.append(float(np.log1p(nz).mean()) if nz.size else 0.0)

            if len(marker_labels) > 0:
                for j, label in enumerate(marker_labels):
                    vals = []
                    for g in marker_prior[label][:ANNOT_SUMMARY_MAX_GENES]:
                        gid = gene2id.get(str(g), None)
                        if gid is None:
                            continue
                        if gid in s.train2rna:
                            vals.append(float(row[s.train2rna[gid]]))
                    if len(vals) > 0:
                        marker_scores[j] += float(np.mean(vals))

        if idx.size > 0 and len(marker_scores) > 0:
            marker_scores /= float(idx.size)

        summaries[int(cid)] = {
            "z_mean": z_mean,
            "rna_nnz_mean": float(np.mean(rna_nnz)) if len(rna_nnz) else 0.0,
            "rna_logmean": float(np.mean(rna_logmean)) if len(rna_logmean) else 0.0,
            "size": int(idx.size),
            "marker_scores": marker_scores.astype(np.float32),
            "marker_labels": marker_labels,
        }
    return summaries


def cluster_summary_to_vector(summary: dict) -> np.ndarray:
    z_mean = np.asarray(summary["z_mean"], dtype=np.float32).reshape(-1)
    meta = np.asarray([
        float(summary["rna_nnz_mean"]),
        float(summary["rna_logmean"]),
        float(summary["size"]),
    ], dtype=np.float32)
    marker_scores = np.asarray(summary.get("marker_scores", []), dtype=np.float32).reshape(-1)
    return np.concatenate([z_mean, meta, marker_scores], axis=0).astype(np.float32)


class ClusterSummaryTable(nn.Module):
    def __init__(self, cluster_summary_np: dict[int, np.ndarray]):
        super().__init__()
        keys = sorted(cluster_summary_np.keys())
        self.cluster_ids = keys
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
# 6) ADAPTER + SELECTOR + RESIDUAL + CORRECTION HEADS
# ============================================================
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


class PriorSelectorV1(nn.Module):
    def __init__(self, summary_dim: int):
        super().__init__()
        self.e = nn.Parameter(torch.zeros((32,), dtype=torch.float32))
        if SELECTOR_INIT_SCALE > 0:
            nn.init.normal_(self.e, mean=0.0, std=SELECTOR_INIT_SCALE)
        self.mlp = nn.Sequential(
            nn.Linear(32 + summary_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 6),
        )
        self.register_buffer("summary", torch.zeros((summary_dim,), dtype=torch.float32), persistent=True)

    def set_summary(self, summary_np: np.ndarray):
        s = np.asarray(summary_np, dtype=np.float32).reshape(-1)
        assert s.size == self.summary.numel()
        self.summary.copy_(torch.from_numpy(s))

    def forward(self) -> torch.Tensor:
        x = torch.cat([self.e, self.summary], dim=0)
        logits = self.mlp(x)
        w = F.softplus(logits) + float(SELECTOR_SOFTPLUS_BIAS)
        w = w / (w.mean() + 1e-12)
        return w


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
    """
    Dataset-level model + cluster-specific residual.
    Output is a D-dimensional delta on cell representation.
    """
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
# 7) SUMMARIES / FILTERS / MINING
# ============================================================
def compute_dataset_summary(train_cells: list[tuple[str, int]], sid2sample: dict[str, Sample], tf2targets: dict, tf_in_prior: list[int], N_GENE_DIM: int) -> np.ndarray:
    if len(train_cells) == 0:
        return np.zeros((8,), np.float32)
    n = min(SELECTOR_SUMMARY_NCELLS, len(train_cells))
    idx = np.random.choice(np.arange(len(train_cells)), size=n, replace=False)
    sub = [train_cells[i] for i in idx.tolist()]
    r_means = []
    r_nnz = []
    for sid, rid in sub:
        s = sid2sample[sid]
        row = get_counts_row_dense(s, rid)
        r_nnz.append(float((row > 0).mean()))
        nz = row[row > 0]
        r_means.append(float(np.log1p(nz).mean()) if nz.size else 0.0)
    reg = regproxy_aucell_lite_from_top(sub, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM)
    tf_cols = np.arange(reg.shape[1])
    if tf_cols.size > 256:
        tf_cols = np.random.choice(tf_cols, size=256, replace=False)
    tf_vals = reg[:, tf_cols]
    tf_mean = float(np.nanmean(tf_vals))
    tf_std = float(np.nanstd(tf_vals))
    anchor_genes = list(next(iter(sid2sample.values())).anchor_gene2col.keys())
    if len(anchor_genes) > 512:
        g_pick = np.random.choice(np.arange(len(anchor_genes)), size=512, replace=False)
        gnames = [anchor_genes[i] for i in g_pick.tolist()]
    else:
        gnames = anchor_genes
    a_means = []
    a_nnz = []
    for sid, rid in sub:
        s = sid2sample[sid]
        arow = int(s.rid2arow[int(rid)])
        if arow < 0:
            continue
        cols = [s.anchor_gene2col.get(g, -1) for g in gnames]
        cols = [c for c in cols if c >= 0]
        if not cols:
            continue
        v = s.anchor_X[arow, np.asarray(cols, np.int64)].astype(np.float32, copy=False)
        a_nnz.append(float((v > 0).mean()))
        nz = v[v > 0]
        a_means.append(float(np.log1p(nz).mean()) if nz.size else 0.0)
    return np.asarray([
        float(np.mean(r_means)), float(np.std(r_means) + 1e-6),
        float(np.mean(r_nnz)), float(np.std(r_nnz) + 1e-6),
        tf_mean, tf_std + 1e-6,
        float(np.mean(a_means)) if a_means else 0.0,
        float(np.mean(a_nnz)) if a_nnz else 0.0,
    ], dtype=np.float32)


def build_filtered_gene_pool_from_train(train_cells: list[tuple[str, int]], sid2sample: dict[str, Sample], gene_pool_full: np.ndarray, genes_vocab: list[str]) -> np.ndarray:
    if gene_pool_full.size == 0 or len(train_cells) == 0:
        return gene_pool_full
    n = min(ATAC_FILTER_NCELLS, len(train_cells))
    idx = np.random.choice(np.arange(len(train_cells)), size=n, replace=False)
    sub = [train_cells[i] for i in idx.tolist()]
    sid2arows = defaultdict(list)
    for sid, rid in sub:
        s = sid2sample[sid]
        arow = int(s.rid2arow[int(rid)])
        if arow >= 0:
            sid2arows[sid].append(arow)
    for sid in list(sid2arows.keys()):
        sid2arows[sid] = np.asarray(sid2arows[sid], dtype=np.int64)
    gene_ids = gene_pool_full.copy()
    if ATAC_FILTER_MAX_GENES is not None and gene_ids.size > int(ATAC_FILTER_MAX_GENES):
        gene_ids = np.random.choice(gene_ids, size=int(ATAC_FILTER_MAX_GENES), replace=False)
        gene_ids = np.asarray(sorted(gene_ids.tolist()), dtype=np.int64)
    gid2name = {int(g): genes_vocab[int(g)] for g in gene_pool_full.tolist() if 0 <= int(g) < len(genes_vocab)}
    keep = []
    nnz_fracs = []
    stds = []
    for gid in gene_ids.tolist():
        gname = gid2name.get(int(gid), "")
        if not gname:
            continue
        vals = []
        for sid, arows in sid2arows.items():
            s = sid2sample[sid]
            col = s.anchor_gene2col.get(gname, None)
            if col is None:
                continue
            vals.append(s.anchor_X[arows, int(col)].astype(np.float32, copy=False))
        if not vals:
            continue
        y = np.concatenate(vals, axis=0).astype(np.float32, copy=False)
        nnz = float((y > 0).mean())
        sd = float(np.std(np.log1p(y)))
        nnz_fracs.append(nnz)
        stds.append(sd)
        if (nnz >= ATAC_FILTER_MIN_NNZ_FRAC) and (sd >= ATAC_FILTER_MIN_STD_LOG1P):
            keep.append(int(gid))
    keep = np.asarray(keep, dtype=np.int64)
    log(f"[ATAC_FILTER] sampled_cells={n} genes_in={gene_pool_full.size} genes_checked={gene_ids.size} kept={keep.size} nnz_frac_med={float(np.median(nnz_fracs)) if nnz_fracs else float('nan'):.3f} std_med={float(np.median(stds)) if stds else float('nan'):.3f}")
    if keep.size < 1024:
        log("[ATAC_FILTER][WARN] kept genes too few; fallback full pool")
        return gene_pool_full
    return keep


def mine_selected_peaks(gene_pool: np.ndarray, gene2peaks: dict[int, list[int]]) -> np.ndarray:
    if gene_pool.size == 0:
        return np.zeros((0,), dtype=np.int64)
    counter = Counter()
    for gid in gene_pool.tolist():
        for pk in gene2peaks.get(int(gid), []):
            counter[int(pk)] += 1
    if len(counter) == 0:
        return np.zeros((0,), dtype=np.int64)
    items = [(pk, cnt) for pk, cnt in counter.items() if cnt >= SELECTED_PEAK_MIN_EDGE_FREQ]
    if len(items) == 0:
        items = list(counter.items())
    items = sorted(items, key=lambda x: (-x[1], x[0]))
    if len(items) > SELECTED_PEAK_MAX:
        items = items[:SELECTED_PEAK_MAX]
    out = np.asarray([int(pk) for pk, _ in items], dtype=np.int64)
    log(f"[SELECTED_PEAK] n_selected={out.size} from_gene_pool={gene_pool.size}")
    return out


# ============================================================
# 8) MAIN TRAIN/EVAL
# ============================================================
def run_one_dataset(dataset_name: str, base_ckpt_path: Path):
    assert base_ckpt_path.exists(), f"Missing BASE ckpt: {base_ckpt_path}"

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)
    marker_prior = load_marker_prior(MARKER_PRIOR_PATH) if USE_CLUSTER_ANNOTATION else {}

    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    cfg = base_ckpt.get("cfg", None)
    assert cfg is not None, f"BASE ckpt missing cfg: {base_ckpt_path}"

    N_GENE_DIM = int(cfg["N_GENE_DIM"])
    N_TF_DIM = int(cfg["N_TF_DIM"])
    peak_hash_size = int(cfg.get("PEAK_HASH_SIZE", 2_000_000))
    D = int(cfg.get("D", 256))
    HID = int(cfg.get("HID", 512))
    DROP = float(cfg.get("DROP", 0.1))
    cancer_types = cfg["cancer_types"]
    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_name = get_default_cancer_name(dataset_name)
    default_cancer_id = int(cancer2id.get(default_cancer_name, 0))
    cid = torch.tensor([default_cancer_id], device=DEVICE, dtype=torch.long)

    log("=" * 100)
    log(f"[RUN] dataset={dataset_name}")
    log(f"[BASE] {base_ckpt_path}")
    log(f"[CANCER] default_name={default_cancer_name} default_id={default_cancer_id}")

    model = PlanC_DirAware_Cancer_HashPeak(
        n_gene=N_GENE_DIM,
        n_tf=N_TF_DIM,
        n_cancer=len(cancer_types),
        peak_hash_size=peak_hash_size,
        d=D, hid=HID, drop=DROP,
    )
    model.load_state_dict(base_ckpt["model"], strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tf_gene_prior = PRIORS_DIR / "tf_gene_f.cap256.npz"
    if not tf_gene_prior.exists():
        tf_gene_prior = PRIORS_DIR / "tf_gene_f.full.npz"
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

    gene_peak_prior = find_prior_file(PRIORS_DIR, ["gene", "peak"])
    assert gene_peak_prior is not None, f"Cannot find gene-peak prior in {PRIORS_DIR}"
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
    genes_in_tfprior = sorted({g for t in tf_in_prior for g in tf2targets[t]})
    genes_union = sorted(set(genes_in_tfprior) | set(genes_in_peakprior))
    gid2pid_u = {int(gid): i for i, gid in enumerate(genes_union)}

    samples, sid2sample = load_samples(dataset_name, genes_vocab)
    all_cells = build_full_cell_lists(samples, MAX_CELLS_PER_SAMPLE)
    assert len(all_cells) > 0, f"No matched cells for {dataset_name}"
    log(f"[ALL] total matched cells used = {len(all_cells)}")

    tr_full, te = split_cells_5050(all_cells)
    tr, va = split_train_val(tr_full, VAL_FRAC_FROM_TRAIN)
    log(f"[CELLS] train={len(tr)} val={len(va)} test={len(te)}")

    proj_store_cache = {}

    log("[ENC] encoding train/val/test Z ...")
    Z_train = encode_Z(tr, sid2sample, model, default_cancer_id, proj_store_cache)
    Zt_train = torch.from_numpy(Z_train).to(DEVICE)
    Z_val = encode_Z(va, sid2sample, model, default_cancer_id, proj_store_cache) if len(va) > 0 else None
    Zt_val = torch.from_numpy(Z_val).to(DEVICE) if Z_val is not None else None
    Z_test = encode_Z(te, sid2sample, model, default_cancer_id, proj_store_cache)
    Zt_test = torch.from_numpy(Z_test).to(DEVICE)

    ds_out = RUN_DIR / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)
    centroid_path = ds_out / "centroids.npy"
    if centroid_path.exists():
        C = np.load(centroid_path).astype(np.float32)
        log(f"[CLUSTER] load fixed centroids: {centroid_path} shape={tuple(C.shape)}")
    else:
        log(f"[CLUSTER] fitting fixed centroids k={N_CLUSTERS} -> {centroid_path}")
        C = spherical_kmeans_fit(Z_train, k=N_CLUSTERS, iters=CLUSTER_ITERS, max_train=CLUSTER_TRAIN_MAX).astype(np.float32)
        np.save(centroid_path, C)
    cl_train = spherical_kmeans_assign(Z_train, C)
    cl_val = spherical_kmeans_assign(Z_val, C) if Z_val is not None else None
    cl_test = spherical_kmeans_assign(Z_test, C)
    cl_all = np.concatenate([cl_train, cl_val, cl_test], axis=0) if cl_val is not None else np.concatenate([cl_train, cl_test], axis=0)
    group_all, cl2gid, cnt = build_group_ids_from_clusters(cl_train, cl_all, nmin=CLUSTER_NMIN_TRAIN)
    ntr = len(tr)
    nva = len(va)
    train_group_ids = group_all[:ntr]
    val_group_ids = group_all[ntr:ntr + nva] if cl_val is not None else None
    test_group_ids = group_all[ntr + nva:] if cl_val is not None else group_all[ntr:]
    train_cancer_ids = np.full((len(tr),), default_cancer_id, np.int64)
    val_cancer_ids = np.full((len(va),), default_cancer_id, np.int64) if len(va) > 0 else None
    test_cancer_ids = np.full((len(te),), default_cancer_id, np.int64)
    log(f"[CLUSTER] raw cluster counts top5={sorted(cnt.tolist(), reverse=True)[:5]}")
    log(f"[CLUSTER] kept clusters={len(cl2gid)} ; n_group={int(train_group_ids.max()+1)} (0=other)")

    train_cluster_for_corr = cl_train if CLUSTER_CORRECTION_USE_RAW_CLUSTER else train_group_ids
    val_cluster_for_corr = cl_val if (CLUSTER_CORRECTION_USE_RAW_CLUSTER and cl_val is not None) else val_group_ids
    test_cluster_for_corr = cl_test if CLUSTER_CORRECTION_USE_RAW_CLUSTER else test_group_ids

    cluster_rows = []
    for split, cells, cl_raw, gid in [
        ("train", tr, cl_train, train_group_ids),
        ("val", va, cl_val, val_group_ids),
        ("test", te, cl_test, test_group_ids),
    ]:
        if cells is None or cl_raw is None:
            continue
        for i, (sid, rid) in enumerate(cells):
            bc = str(sid2sample[sid].cells[int(rid)])
            cluster_rows.append({
                "split": split,
                "sid": sid,
                "rid": int(rid),
                "barcode": bc,
                "cell_key": make_cell_key(sid, bc),
                "cluster_id": int(cl_raw[i]),
                "group_id": int(gid[i]),
            })
    pd.DataFrame(cluster_rows).to_csv(ds_out / "clusters.csv", index=False)

    log("[ENC] precompute top genes ...")
    G_top_tr, V_top_tr = precompute_top_genes(tr, sid2sample, STORE_TOP_G, N_GENE_DIM)
    G_top_te, V_top_te = precompute_top_genes(te, sid2sample, STORE_TOP_G, N_GENE_DIM)
    G_top_va, V_top_va = precompute_top_genes(va, sid2sample, STORE_TOP_G, N_GENE_DIM) if len(va) > 0 else (None, None)

    log("[PROXY] TF proxies ...")
    REG_train_au = regproxy_aucell_lite_from_top(tr, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM)
    REG_test_au = regproxy_aucell_lite_from_top(te, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM)
    REG_val_au = regproxy_aucell_lite_from_top(va, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM) if len(va) > 0 else None
    REG_train_mn = regproxy_mean_log1p(tr, sid2sample, tf2targets, tf_in_prior)
    REG_test_mn = regproxy_mean_log1p(te, sid2sample, tf2targets, tf_in_prior)
    REG_val_mn = regproxy_mean_log1p(va, sid2sample, tf2targets, tf_in_prior) if len(va) > 0 else None
    REG_train_tf = tfproxy_mix(REG_train_au, REG_train_mn)
    REG_test_tf = tfproxy_mix(REG_test_au, REG_test_mn)
    REG_val_tf = tfproxy_mix(REG_val_au, REG_val_mn) if len(va) > 0 else None

    anchor_gene_union = set()
    for s in samples:
        anchor_gene_union.update(s.anchor_gene2col.keys())
    gene_pool_full = []
    for gid in genes_in_peakprior:
        if 0 <= int(gid) < len(genes_vocab):
            gname = genes_vocab[int(gid)]
            if gname in anchor_gene_union:
                gene_pool_full.append(int(gid))
    gene_pool_full = np.asarray(gene_pool_full, dtype=np.int64)
    log("[ATAC] gene_pool_full size =", int(gene_pool_full.size))
    gene_pool_train = build_filtered_gene_pool_from_train(tr, sid2sample, gene_pool_full, genes_vocab)
    log("[ATAC] gene_pool_train(filtered) size =", int(gene_pool_train.size))

    selected_source = gene_pool_train if SELECTED_PEAK_ONLY_FROM_FILTERED_GENES else gene_pool_full
    selected_peak_raw_ids = mine_selected_peaks(selected_source, gene2peaks) if USE_SELECTED_PEAK_RESIDUAL else np.zeros((0,), dtype=np.int64)

    log("[CLUSTER_SUMMARY] build train cluster summaries ...")
    train_cluster_summaries = build_cluster_summaries(
        cell_list=tr,
        sid2sample=sid2sample,
        Z=Z_train,
        cluster_ids=train_cluster_for_corr,
        genes_vocab=genes_vocab,
        marker_prior=marker_prior,
    )
    train_cluster_summary_vecs = {cid_: cluster_summary_to_vector(v) for cid_, v in train_cluster_summaries.items()}
    cluster_summary_table = ClusterSummaryTable(train_cluster_summary_vecs).to(DEVICE)
    pd.DataFrame([
        {
            "cluster_id": int(cid_),
            "size": int(train_cluster_summaries[cid_]["size"]),
            "rna_nnz_mean": float(train_cluster_summaries[cid_]["rna_nnz_mean"]),
            "rna_logmean": float(train_cluster_summaries[cid_]["rna_logmean"]),
        }
        for cid_ in sorted(train_cluster_summaries.keys())
    ]).to_csv(ds_out / "cluster_summary_stats.csv", index=False)

    n_group = int(max(1, train_group_ids.max() + 1))
    adapter = HyperAdapterHashV2(
        d=D, k=K_RANK,
        n_tf=len(tf_in_prior),
        n_gene=len(genes_union),
        peak_hash_size=peak_hash_size,
        n_cancer=len(cancer_types),
        n_group=n_group,
        e_dim=64, hid=256,
    ).to(DEVICE)
    peak_residual = SelectedPeakResidual(selected_peak_raw_ids, d=D).to(DEVICE) if USE_SELECTED_PEAK_RESIDUAL else None
    selector = None
    if USE_SELECTOR:
        summary = compute_dataset_summary(tr, sid2sample, tf2targets, tf_in_prior, N_GENE_DIM)
        selector = PriorSelectorV1(summary_dim=int(summary.size)).to(DEVICE)
        selector.set_summary(summary)

    tf_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=CLUSTER_CORRECTION_HID).to(DEVICE) if USE_CLUSTER_CORRECTION else None
    rna_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=CLUSTER_CORRECTION_HID).to(DEVICE) if USE_CLUSTER_CORRECTION else None
    re_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=CLUSTER_CORRECTION_HID).to(DEVICE) if USE_CLUSTER_CORRECTION else None

    params = list(adapter.parameters())
    if peak_residual is not None:
        params += list(peak_residual.parameters())
    if selector is not None:
        params += list(selector.parameters())
    if tf_corr_head is not None:
        params += list(tf_corr_head.parameters())
    if rna_corr_head is not None:
        params += list(rna_corr_head.parameters())
    if re_corr_head is not None:
        params += list(re_corr_head.parameters())

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    train_groups_sid = np.asarray([sid for sid, _ in tr], dtype=object)
    val_groups_sid = np.asarray([sid for sid, _ in va], dtype=object) if len(va) else None
    test_groups_sid = np.asarray([sid for sid, _ in te], dtype=object)

    gid2idx = defaultdict(list)
    for i, g in enumerate(train_group_ids.tolist()):
        gid2idx[int(g)].append(i)
    gid2idx = {g: np.asarray(v, np.int64) for g, v in gid2idx.items()}
    big_groups = sorted([g for g, v in gid2idx.items() if v.size >= STRAT_MIN_GROUP_SIZE])
    tf_prior_ids_all = np.arange(len(tf_in_prior), dtype=np.int64)

    @torch.no_grad()
    def tf_base(tf_pid: torch.Tensor) -> torch.Tensor:
        tf_ids = torch.tensor([tf_in_prior[i] for i in tf_pid.tolist()], device=DEVICE, dtype=torch.long)
        return model.tf_in(tf_ids) + model.tf_in_c(cid).expand(tf_ids.numel(), -1)

    @torch.no_grad()
    def gene_base_from_pid(gene_pid: torch.Tensor) -> torch.Tensor:
        gene_ids = torch.tensor([genes_union[i] for i in gene_pid.tolist()], device=DEVICE, dtype=torch.long)
        return model.gene_out(gene_ids) + model.gene_out_c(cid).expand(gene_ids.numel(), -1)

    @torch.no_grad()
    def peak_base_from_raw(peak_idx_raw: torch.Tensor) -> torch.Tensor:
        return model.peak_out(peak_idx_raw) + model.peak_out_c(cid).expand(peak_idx_raw.numel(), -1)

    def score_tf_vec(tf_pid_t: torch.Tensor, V_tf: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        TF = tf_base(tf_pid_t) + adapter.delta_tf(tf_pid_t, V_tf)
        if USE_CANCER_ADAPTOR:
            c_tok = torch.full((tf_pid_t.numel(),), int(default_cancer_id), device=DEVICE, dtype=torch.long)
            TF = TF + adapter.delta_cancer(c_tok, V_cancer)
        return TF

    def score_gene_vec(gene_pid_t: torch.Tensor, V_gene: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        G = gene_base_from_pid(gene_pid_t) + adapter.delta_gene(gene_pid_t, V_gene)
        if USE_CANCER_ADAPTOR:
            c_tok = torch.full((gene_pid_t.numel(),), int(default_cancer_id), device=DEVICE, dtype=torch.long)
            G = G + adapter.delta_cancer(c_tok, V_cancer)
        return G

    def score_peak_vec(pool_peak_raw_np: np.ndarray, V_peak: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        peak_raw_t = torch.from_numpy(pool_peak_raw_np.astype(np.int64)).to(DEVICE)
        P = peak_base_from_raw(peak_raw_t) + adapter.delta_peak_hash(peak_raw_t, V_peak)
        if peak_residual is not None and peak_residual.n_selected > 0:
            P = P + peak_residual.lookup(peak_raw_t)
        if USE_CANCER_ADAPTOR:
            c_tok = torch.full((peak_raw_t.numel(),), int(default_cancer_id), device=DEVICE, dtype=torch.long)
            P = P + adapter.delta_cancer(c_tok, V_cancer)
        return P

    def cluster_ids_to_summary(cluster_ids_np: np.ndarray):
        cid_t = torch.from_numpy(cluster_ids_np.astype(np.int64)).to(DEVICE)
        return cluster_summary_table.lookup(cid_t)

    def apply_cluster_correction(z_in: torch.Tensor, cluster_ids_np: np.ndarray, head: ClusterCellCorrectionHead | None, alpha: float):
        if (not USE_CLUSTER_CORRECTION) or head is None:
            return z_in
        summary_t = cluster_ids_to_summary(cluster_ids_np)
        delta = head(z_in, summary_t)
        return F.normalize(z_in + float(alpha) * delta, dim=-1)

    def adjust_Z(Zt, group_ids_np, cancer_ids_np, cluster_ids_np, V_cancer, task: str):
        Zadj = Zt
        if USE_CELLGROUP_ADAPTOR:
            gid_t = torch.from_numpy(group_ids_np.astype(np.int64)).to(DEVICE)
            Zadj = adapter.apply_group_to_Z(Zadj, gid_t)
        if USE_CANCER_ADAPTOR:
            cid_t = torch.from_numpy(cancer_ids_np.astype(np.int64)).to(DEVICE)
            Zadj = Zadj + adapter.delta_cancer(cid_t, V_cancer)
        Zadj = F.normalize(Zadj, dim=-1)
        if task == "tf":
            Zadj = apply_cluster_correction(Zadj, cluster_ids_np, tf_corr_head, CLUSTER_CORRECTION_ALPHA_TF)
        elif task == "rna":
            Zadj = apply_cluster_correction(Zadj, cluster_ids_np, rna_corr_head, CLUSTER_CORRECTION_ALPHA_RNA)
        elif task == "re":
            Zadj = apply_cluster_correction(Zadj, cluster_ids_np, re_corr_head, CLUSTER_CORRECTION_ALPHA_RE)
        return Zadj

    def peaks_pool_for_gene(gid_train: int) -> np.ndarray:
        pks = gene2peaks.get(int(gid_train), [])
        pool = np.asarray(list(dict.fromkeys(pks)), dtype=np.int64)
        if pool.size > RE_POOL_MAX:
            pool = np.random.choice(pool, size=RE_POOL_MAX, replace=False)
        return pool

    def sample_genes_pos_neg_from_top(G_top: np.ndarray, V_top: np.ndarray, cell_ids: np.ndarray):
        B = cell_ids.size
        pos = np.full((B, RNA_POS_PER_CELL), -1, dtype=np.int64)
        neg = np.full((B, RNA_NEG_PER_CELL), -1, dtype=np.int64)
        for i, c in enumerate(cell_ids.tolist()):
            g_all = G_top[c]
            v_all = V_top[c]
            valid = (g_all >= 0)
            if valid.sum() < 4:
                continue
            g = g_all[valid].astype(np.int64, copy=False)
            v = v_all[valid].astype(np.float32, copy=False)
            pids = np.array([gid2pid_u.get(int(x), -1) for x in g.tolist()], dtype=np.int64)
            ok = (pids >= 0)
            if ok.sum() < 4:
                continue
            pids = pids[ok]
            vv = v[ok]
            vv = np.clip(np.nan_to_num(vv, nan=0.0), 0.0, None)
            k = min(RNA_POS_PER_CELL, pids.size)
            if k <= 0:
                continue
            nz = np.where(vv > 0)[0]
            if nz.size >= k:
                p = vv[nz] / (vv[nz].sum() + 1e-12)
                choice = np.random.choice(nz, size=k, replace=False, p=p)
            else:
                choice = np.random.choice(np.arange(pids.size), size=k, replace=(pids.size < k))
            pos[i, :k] = pids[choice[:k]]
            neg_take = min(RNA_NEG_PER_CELL, len(genes_union))
            if QC_GENE_HARD_NEG and pids.size > 0:
                pool = np.setdiff1d(np.arange(len(genes_union)), np.unique(pids), assume_unique=False)
                if pool.size >= neg_take:
                    neg[i, :neg_take] = np.random.choice(pool, size=neg_take, replace=False)
                else:
                    neg[i, :neg_take] = np.random.choice(np.arange(len(genes_union)), size=neg_take, replace=False)
            else:
                neg[i, :neg_take] = np.random.choice(np.arange(len(genes_union)), size=neg_take, replace=False)
        return pos, neg

    def re_anchor_loss(cell_idx: np.ndarray, Zb: torch.Tensor, cluster_idx_np: np.ndarray, V_peak: torch.Tensor, V_cancer: torch.Tensor):
        if gene_pool_train.size == 0:
            return torch.zeros((), device=DEVICE), 0
        B = cell_idx.size
        idx = np.arange(B, dtype=np.int64)
        if idx.size > RE_CELL_BATCH:
            idx = np.random.choice(idx, size=RE_CELL_BATCH, replace=False)
        Zp0 = Zb[torch.from_numpy(idx).to(DEVICE)]
        cluster_sub = cluster_idx_np[idx]
        Zp = apply_cluster_correction(Zp0, cluster_sub, re_corr_head, CLUSTER_CORRECTION_ALPHA_RE)

        sids = []
        arows = []
        for bi in idx.tolist():
            sid, rid = tr[int(cell_idx[bi])]
            s = sid2sample[sid]
            sids.append(sid)
            arows.append(int(s.rid2arow[int(rid)]))
        g_sel = np.random.choice(gene_pool_train, size=min(RE_GENE_BATCH, gene_pool_train.size), replace=False)
        loss = torch.zeros((), device=DEVICE)
        used = 0
        for gid in g_sel.tolist():
            gname = genes_vocab[int(gid)] if 0 <= int(gid) < len(genes_vocab) else ""
            if not gname:
                continue
            y = np.full((len(idx),), np.nan, dtype=np.float32)
            for i0, (sid, arow) in enumerate(zip(sids, arows)):
                if arow < 0:
                    continue
                s = sid2sample[sid]
                col = s.anchor_gene2col.get(gname, None)
                if col is None:
                    continue
                y[i0] = float(s.anchor_X[int(arow), int(col)])
            finite = np.isfinite(y)
            if finite.sum() < 128:
                continue
            pool = peaks_pool_for_gene(int(gid))
            if pool.size < 8:
                continue
            P = score_peak_vec(pool, V_peak, V_cancer)
            S = Zp @ P.t()
            k2 = min(RE_TOPK_FOR_POOL, S.shape[1])
            topv = torch.topk(S, k=k2, dim=1, largest=True).values
            topv = torch.clamp(topv, min=0.0)
            pred = torch.logsumexp(topv, dim=1)
            fi = np.where(finite)[0].astype(np.int64)
            pred_f = pred[torch.from_numpy(fi).to(DEVICE)]
            y_f = torch.from_numpy(y[finite].astype(np.float32)).to(DEVICE)
            pred_z = (pred_f - pred_f.mean()) / (pred_f.std() + 1e-6)
            y_z = (y_f - y_f.mean()) / (y_f.std() + 1e-6)
            reg = F.smooth_l1_loss(pred_z, y_z)
            y_np = y[finite]
            ybin, _ = make_pos_neg_labels(y_np, POS_Q, NEG_Q)
            pos = np.where(ybin == 1)[0]
            neg = np.where(ybin == 0)[0]
            if pos.size < 24 or neg.size < 24:
                loss = loss + 0.3 * reg
                used += 1
                continue
            m0 = min(pos.size, neg.size, 128)
            pos_take = np.random.choice(pos, size=m0, replace=False)
            neg_take = np.random.choice(neg, size=m0, replace=False)
            ap = pred_z[torch.from_numpy(pos_take).to(DEVICE)]
            an = pred_z[torch.from_numpy(neg_take).to(DEVICE)]
            diff = ap.view(-1, 1) - an.view(1, -1)
            rank = F.softplus(MARGIN_RE - diff).mean()
            loss = loss + 0.3 * reg + 0.7 * rank
            used += 1
        if used > 0:
            loss = loss / used
        return loss, used

    def eval_tf_auc(Zt, group_ids_np, cancer_ids_np, cluster_ids_np, REG_tf_np, groups_sid_np, raw_cluster_np, max_tf_sample):
        if Zt is None or REG_tf_np is None or groups_sid_np is None:
            return {"auc_mean": np.nan, "n_tf_eval": 0, "cluster_rows": []}
        adapter.eval()
        if selector is not None:
            selector.eval()
        if peak_residual is not None:
            peak_residual.eval()
        if tf_corr_head is not None:
            tf_corr_head.eval()
        with torch.no_grad():
            V_tf_eval, _, _, V_cancer_eval = adapter()
            tf_ids = np.arange(len(tf_in_prior), dtype=np.int64)
            if max_tf_sample is not None and tf_ids.size > int(max_tf_sample):
                tf_ids = np.random.choice(tf_ids, size=int(max_tf_sample), replace=False)
            tf_ids = np.sort(tf_ids)
            tf_pid_t = torch.from_numpy(tf_ids).to(DEVICE)
            TF_vec = score_tf_vec(tf_pid_t, V_tf_eval, V_cancer_eval)
            Zadj = adjust_Z(Zt, group_ids_np, cancer_ids_np, cluster_ids_np, V_cancer_eval, task="tf")
            pred = (Zadj @ TF_vec.t()).detach().cpu().numpy().astype(np.float32)
            aucs = []
            for j, tfp in enumerate(tf_ids.tolist()):
                regj = REG_tf_np[:, int(tfp)]
                y, keep = make_pos_neg_labels_grouped(regj, groups_sid_np, POS_Q, NEG_Q, min_n=30)
                if keep.sum() < 128 or (y == 1).sum() < 32 or (y == 0).sum() < 32:
                    continue
                aucs.append(auc_from_scores(pred[y == 1, j], pred[y == 0, j]))
            auc_mean = float(np.nanmean(np.asarray(aucs, np.float32))) if aucs else float("nan")
            cluster_rows = []
            if raw_cluster_np is not None:
                for cid_raw in np.unique(raw_cluster_np):
                    m = (raw_cluster_np == cid_raw)
                    ncell = int(m.sum())
                    if ncell < VAL_CLUSTER_MIN_CELLS:
                        continue
                    aucs_c = []
                    for j, tfp in enumerate(tf_ids.tolist()):
                        regj = REG_tf_np[m, int(tfp)]
                        y, keep = make_pos_neg_labels(regj, POS_Q, NEG_Q)
                        if keep.sum() < max(40, int(0.2 * ncell)):
                            continue
                        if (y == 1).sum() < VAL_CLUSTER_MIN_POSNEG or (y == 0).sum() < VAL_CLUSTER_MIN_POSNEG:
                            continue
                        aucs_c.append(auc_from_scores(pred[m, j][y == 1], pred[m, j][y == 0]))
                    if len(aucs_c) >= 8:
                        cluster_rows.append((int(cid_raw), ncell, float(np.nanmean(np.asarray(aucs_c, np.float32)))))
                cluster_rows = sorted(cluster_rows, key=lambda x: x[2], reverse=True)
        adapter.train()
        if selector is not None:
            selector.train()
        if peak_residual is not None:
            peak_residual.train()
        if tf_corr_head is not None:
            tf_corr_head.train()
        return {"auc_mean": auc_mean, "n_tf_eval": int(len(aucs)), "cluster_rows": cluster_rows}

    def eval_re_proxy(cell_list, Zt, group_ids_np, cancer_ids_np, cluster_ids_np, gene_pool_eval):
        if Zt is None or len(cell_list) == 0 or gene_pool_eval.size == 0:
            return {"spearman_mean": np.nan, "rank_auc_mean": np.nan, "n_genes_eval": 0}
        adapter.eval()
        if selector is not None:
            selector.eval()
        if peak_residual is not None:
            peak_residual.eval()
        if re_corr_head is not None:
            re_corr_head.eval()
        with torch.no_grad():
            _, _, V_peak_eval, V_cancer_eval = adapter()
            Zadj = adjust_Z(Zt, group_ids_np, cancer_ids_np, cluster_ids_np, V_cancer_eval, task="re")
            n_eval = min(QC_RE_GENE_EVAL, gene_pool_eval.size)
            g_sel = np.random.choice(gene_pool_eval, size=n_eval, replace=False)
            spears = []
            aucs = []
            for gid in g_sel.tolist():
                gname = genes_vocab[int(gid)] if 0 <= int(gid) < len(genes_vocab) else ""
                if not gname:
                    continue
                y = np.full((len(cell_list),), np.nan, dtype=np.float32)
                for i0, (sid, rid) in enumerate(cell_list):
                    s = sid2sample[sid]
                    arow = int(s.rid2arow[int(rid)])
                    if arow < 0:
                        continue
                    col = s.anchor_gene2col.get(gname, None)
                    if col is None:
                        continue
                    y[i0] = float(s.anchor_X[int(arow), int(col)])
                finite = np.isfinite(y)
                if finite.sum() < QC_RE_MIN_FINITE:
                    continue
                pool = peaks_pool_for_gene(int(gid))
                if pool.size < 8:
                    continue
                P = score_peak_vec(pool, V_peak_eval, V_cancer_eval)
                S = Zadj @ P.t()
                k2 = min(RE_TOPK_FOR_POOL, S.shape[1])
                topv = torch.topk(S, k=k2, dim=1, largest=True).values
                topv = torch.clamp(topv, min=0.0)
                pred = torch.logsumexp(topv, dim=1).detach().cpu().numpy().astype(np.float32)
                pred_f = pred[finite]
                y_f = y[finite]
                spears.append(spearmanr_fast(pred_f, y_f))
                ybin, keep = make_pos_neg_labels(y_f, POS_Q, NEG_Q)
                if keep.sum() >= 64 and (ybin == 1).sum() >= 16 and (ybin == 0).sum() >= 16:
                    aucs.append(auc_from_scores(pred_f[ybin == 1], pred_f[ybin == 0]))
        adapter.train()
        if selector is not None:
            selector.train()
        if peak_residual is not None:
            peak_residual.train()
        if re_corr_head is not None:
            re_corr_head.train()
        return {
            "spearman_mean": float(np.nanmean(np.asarray(spears, np.float32))) if len(spears) else float("nan"),
            "rank_auc_mean": float(np.nanmean(np.asarray(aucs, np.float32))) if len(aucs) else float("nan"),
            "n_genes_eval": int(len(spears)),
        }

    def eval_rna_retrieval(cell_list, Zt, group_ids_np, cancer_ids_np, cluster_ids_np, G_top, V_top, max_cells):
        if Zt is None or G_top is None or len(cell_list) == 0:
            return {"retrieval_auc": np.nan, "n_scored": 0}
        adapter.eval()
        if selector is not None:
            selector.eval()
        if peak_residual is not None:
            peak_residual.eval()
        if rna_corr_head is not None:
            rna_corr_head.eval()
        with torch.no_grad():
            _, V_gene_eval, _, V_cancer_eval = adapter()
            n_eval = min(max_cells, len(cell_list))
            idx = np.random.choice(np.arange(len(cell_list)), size=n_eval, replace=False)
            Zadj = adjust_Z(
                Zt[torch.from_numpy(idx).to(DEVICE)],
                group_ids_np[idx],
                cancer_ids_np[idx],
                cluster_ids_np[idx],
                V_cancer_eval,
                task="rna",
            )
            pos_gene_pid, neg_gene_pid = sample_genes_pos_neg_from_top(G_top, V_top, idx)
            pos_flat = pos_gene_pid.reshape(-1)
            neg_flat = neg_gene_pid.reshape(-1)
            pos_mask = pos_flat >= 0
            neg_mask = neg_flat >= 0
            if pos_mask.sum() < 32 or neg_mask.sum() < 32:
                return {"retrieval_auc": np.nan, "n_scored": 0}
            pos_pid_t = torch.from_numpy(pos_flat[pos_mask]).to(DEVICE)
            neg_pid_t = torch.from_numpy(neg_flat[neg_mask]).to(DEVICE)
            G_pos = score_gene_vec(pos_pid_t, V_gene_eval, V_cancer_eval)
            G_neg = score_gene_vec(neg_pid_t, V_gene_eval, V_cancer_eval)
            B = idx.size
            cell_rep_pos = np.repeat(np.arange(B), RNA_POS_PER_CELL)[pos_mask]
            cell_rep_neg = np.repeat(np.arange(B), RNA_NEG_PER_CELL)[neg_mask]
            Z_pos = Zadj[torch.from_numpy(cell_rep_pos).to(DEVICE)]
            Z_neg = Zadj[torch.from_numpy(cell_rep_neg).to(DEVICE)]
            s_pos = (Z_pos * G_pos).sum(-1).detach().cpu().numpy().astype(np.float32)
            s_neg = (Z_neg * G_neg).sum(-1).detach().cpu().numpy().astype(np.float32)
            auc = auc_from_scores(s_pos, s_neg)
        adapter.train()
        if selector is not None:
            selector.train()
        if peak_residual is not None:
            peak_residual.train()
        if rna_corr_head is not None:
            rna_corr_head.train()
        return {"retrieval_auc": float(auc), "n_scored": int(min(s_pos.size, s_neg.size))}

    best_auc = -1.0
    best_ckpt_path = ds_out / "frontier_best.pt"
    best_state_cpu = None

    adapter.train()
    if selector is not None:
        selector.train()
    if peak_residual is not None:
        peak_residual.train()
    if tf_corr_head is not None:
        tf_corr_head.train()
    if rna_corr_head is not None:
        rna_corr_head.train()
    if re_corr_head is not None:
        re_corr_head.train()

    val_cluster_id = cl_val.copy() if cl_val is not None else None

    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        V_tf, V_gene, V_peak, V_cancer = adapter()
        if selector is not None:
            w_rel = selector()
            w_tf_gate = float(w_rel[0].detach().cpu())
            w_re_gate = float(w_rel[3].detach().cpu())
            sel_pen = ((w_rel - 1.0).pow(2).mean()) * float(SELECTOR_L2)
        else:
            w_tf_gate = 1.0
            w_re_gate = 1.0
            sel_pen = torch.zeros((), device=DEVICE)

        if USE_STRATIFIED_BATCH and len(big_groups) > 0:
            per_g = max(STRAT_PER_GROUP_MIN, int(CELL_BATCH // max(1, len(big_groups))))
            parts = []
            for g in big_groups:
                idxs = gid2idx[g]
                take = min(per_g, idxs.size)
                parts.append(np.random.choice(idxs, size=take, replace=(idxs.size < take)))
            cell_idx = np.concatenate(parts).astype(np.int64)
            if cell_idx.size > CELL_BATCH:
                cell_idx = np.random.choice(cell_idx, size=CELL_BATCH, replace=False)
            elif cell_idx.size < CELL_BATCH:
                extra = np.random.choice(np.arange(len(tr)), size=(CELL_BATCH - cell_idx.size), replace=False)
                cell_idx = np.concatenate([cell_idx, extra]).astype(np.int64)
        else:
            cell_idx = np.random.choice(np.arange(len(tr)), size=min(CELL_BATCH, len(tr)), replace=(len(tr) < CELL_BATCH)).astype(np.int64)

        Zb0 = Zt_train[torch.from_numpy(cell_idx).to(DEVICE)]
        gid_b_np = train_group_ids[cell_idx].astype(np.int64)
        cid_b_np = train_cancer_ids[cell_idx].astype(np.int64)
        clcorr_b_np = train_cluster_for_corr[cell_idx].astype(np.int64)

        if USE_CELLGROUP_ADAPTOR:
            gid_b_t = torch.from_numpy(gid_b_np).to(DEVICE)
            Zb0 = adapter.apply_group_to_Z(Zb0, gid_b_t)
        if USE_CANCER_ADAPTOR:
            cid_b_t = torch.from_numpy(cid_b_np).to(DEVICE)
            Zb0 = Zb0 + adapter.delta_cancer(cid_b_t, V_cancer)
        Zb0 = F.normalize(Zb0, dim=-1)

        Zb_tf = apply_cluster_correction(Zb0, clcorr_b_np, tf_corr_head, CLUSTER_CORRECTION_ALPHA_TF)
        Zb_rna = apply_cluster_correction(Zb0, clcorr_b_np, rna_corr_head, CLUSTER_CORRECTION_ALPHA_RNA)
        groups_batch = train_groups_sid[cell_idx]

        tf_pid = np.random.choice(tf_prior_ids_all, size=min(TF_BATCH, len(tf_prior_ids_all)), replace=False)
        tf_pid_t = torch.from_numpy(tf_pid).to(DEVICE)
        TF_vec = score_tf_vec(tf_pid_t, V_tf, V_cancer)
        pred_tf = Zb_tf @ TF_vec.t()
        loss_tf = torch.zeros((), device=DEVICE)
        used_tf = 0
        for j, tfp in enumerate(tf_pid.tolist()):
            reg = REG_train_tf[cell_idx, int(tfp)]
            yb, keep = make_pos_neg_labels_grouped(reg, groups_batch, POS_Q, NEG_Q, min_n=30)
            if keep.sum() < 128:
                continue
            kk = np.where(keep)[0]
            yk = yb[kk].astype(np.float32)
            pj = pred_tf[torch.from_numpy(kk).to(DEVICE), j]
            bce = F.binary_cross_entropy_with_logits(pj, torch.from_numpy(yk).to(DEVICE))
            pos_idx = kk[yk > 0.5]
            neg_idx = kk[yk < 0.5]
            if pos_idx.size > 0 and neg_idx.size > 0:
                ppos = pred_tf[torch.from_numpy(pos_idx).to(DEVICE), j]
                pneg = pred_tf[torch.from_numpy(neg_idx).to(DEVICE), j]
                mm = min(ppos.numel(), pneg.numel(), 128)
                if mm > 0:
                    ppos = ppos[torch.randperm(ppos.numel(), device=DEVICE)[:mm]]
                    pneg = pneg[torch.randperm(pneg.numel(), device=DEVICE)[:mm]]
                    rank = F.softplus(MARGIN_TF - (ppos - pneg)).mean()
                else:
                    rank = 0.0 * bce
            else:
                rank = 0.0 * bce
            loss_tf = loss_tf + bce + 0.5 * rank
            used_tf += 1
        if used_tf > 0:
            loss_tf = loss_tf / used_tf

        pos_gene_pid, neg_gene_pid = sample_genes_pos_neg_from_top(G_top_tr, V_top_tr, cell_idx)
        pos_flat = pos_gene_pid.reshape(-1)
        neg_flat = neg_gene_pid.reshape(-1)
        pos_mask = pos_flat >= 0
        neg_mask = neg_flat >= 0
        loss_rna = torch.zeros((), device=DEVICE)
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_pid_t = torch.from_numpy(pos_flat[pos_mask]).to(DEVICE)
            neg_pid_t = torch.from_numpy(neg_flat[neg_mask]).to(DEVICE)
            G_pos = score_gene_vec(pos_pid_t, V_gene, V_cancer)
            G_neg = score_gene_vec(neg_pid_t, V_gene, V_cancer)
            B = cell_idx.size
            cell_rep_pos = np.repeat(np.arange(B), RNA_POS_PER_CELL)[pos_mask]
            cell_rep_neg = np.repeat(np.arange(B), RNA_NEG_PER_CELL)[neg_mask]
            Z_pos = Zb_rna[torch.from_numpy(cell_rep_pos).to(DEVICE)]
            Z_neg = Zb_rna[torch.from_numpy(cell_rep_neg).to(DEVICE)]
            s_pos = (Z_pos * G_pos).sum(-1)
            s_neg = (Z_neg * G_neg).sum(-1)
            loss_rna = 0.5 * (
                F.binary_cross_entropy_with_logits(s_pos, torch.ones_like(s_pos)) +
                F.binary_cross_entropy_with_logits(s_neg, torch.zeros_like(s_neg))
            )

        loss_re, used_re = re_anchor_loss(cell_idx, Zb0, clcorr_b_np, V_peak, V_cancer)

        total_loss = (W_TF_BASE * w_tf_gate) * loss_tf + W_RNA_BASE * loss_rna + (W_RE_BASE * w_re_gate) * loss_re + sel_pen

        l2 = (
            adapter.C_tf.weight.pow(2).mean() +
            adapter.C_gene.weight.pow(2).mean() +
            adapter.C_peak_hash.weight.pow(2).mean() +
            adapter.C_cancer.weight.pow(2).mean() +
            adapter.S_group.weight.pow(2).mean() +
            adapter.B_group.weight.pow(2).mean() +
            V_tf.pow(2).mean() + V_gene.pow(2).mean() + V_peak.pow(2).mean() + V_cancer.pow(2).mean()
        )
        if peak_residual is not None and peak_residual.n_selected > 0:
            l2 = l2 + float(SELECTED_PEAK_RESIDUAL_L2) * peak_residual.emb.weight.pow(2).mean()
        if USE_CLUSTER_CORRECTION:
            if tf_corr_head is not None:
                l2 = l2 + float(CLUSTER_CORRECTION_L2) * sum(p.pow(2).mean() for p in tf_corr_head.parameters())
            if rna_corr_head is not None:
                l2 = l2 + float(CLUSTER_CORRECTION_L2) * sum(p.pow(2).mean() for p in rna_corr_head.parameters())
            if re_corr_head is not None:
                l2 = l2 + float(CLUSTER_CORRECTION_L2) * sum(p.pow(2).mean() for p in re_corr_head.parameters())
        total_loss = total_loss + LAMBDA_L2 * l2

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()

        if step % 300 == 0 or step == 1:
            lr_now = float(opt.param_groups[0]["lr"])
            w_show = "off"
            if selector is not None:
                w_show = np.array2string(selector().detach().cpu().numpy(), precision=2)
            n_peak_res = int(peak_residual.n_selected) if peak_residual is not None else 0
            log(
                f"[{dataset_name}][step {step:4d}] "
                f"total={float(total_loss.detach().cpu()):.4f} "
                f"tf={float(loss_tf.detach().cpu()):.3f} "
                f"re={float(loss_re.detach().cpu()):.3f} "
                f"rna={float(loss_rna.detach().cpu()):.3f} "
                f"used_tf={used_tf} used_re={used_re} "
                f"lr={lr_now:.2e} w_rel={w_show} "
                f"n_peak_res={n_peak_res}"
            )

        if SAVE_BEST_BY_VAL and (Zt_val is not None) and (step % VAL_MONITOR_EVERY == 0 or step == STEPS):
            out = eval_tf_auc(
                Zt_val,
                val_group_ids,
                val_cancer_ids,
                val_cluster_for_corr,
                REG_val_tf,
                val_groups_sid,
                cl_val,
                VAL_MONITOR_TF_SAMPLE,
            )
            auc_now = out["auc_mean"]
            cluster_rows = out["cluster_rows"]
            log(f"[{dataset_name}][VAL@{step}] TF(mixed) AUC(sample={VAL_MONITOR_TF_SAMPLE}) = {auc_now:.4f}")
            if cluster_rows:
                top = cluster_rows[:VAL_CLUSTER_TOPK_PRINT]
                bot = cluster_rows[-VAL_CLUSTER_BOTTOMK_PRINT:] if len(cluster_rows) >= VAL_CLUSTER_BOTTOMK_PRINT else cluster_rows
                log(f"[{dataset_name}][VAL@{step}] per-cluster top: " + ", ".join([f"cl{c}(n={n}):{a:.3f}" for c, n, a in top]))
                log(f"[{dataset_name}][VAL@{step}] per-cluster bottom: " + ", ".join([f"cl{c}(n={n}):{a:.3f}" for c, n, a in bot]))
            else:
                log(f"[{dataset_name}][VAL@{step}] per-cluster AUC: none passed thresholds")
            if np.isfinite(auc_now) and auc_now > best_auc:
                best_auc = auc_now
                best_state_cpu = {
                    "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                    "peak_residual": {k: v.detach().cpu() for k, v in peak_residual.state_dict().items()} if peak_residual is not None else None,
                    "selector": {k: v.detach().cpu() for k, v in selector.state_dict().items()} if selector is not None else None,
                    "tf_corr_head": {k: v.detach().cpu() for k, v in tf_corr_head.state_dict().items()} if tf_corr_head is not None else None,
                    "rna_corr_head": {k: v.detach().cpu() for k, v in rna_corr_head.state_dict().items()} if rna_corr_head is not None else None,
                    "re_corr_head": {k: v.detach().cpu() for k, v in re_corr_head.state_dict().items()} if re_corr_head is not None else None,
                }
                torch.save({
                    "dataset": dataset_name,
                    "step": int(step),
                    "base_ckpt": str(base_ckpt_path),
                    "best_val_tf_auc_mixed": float(best_auc),
                    "adapter_state_dict": adapter.state_dict(),
                    "peak_residual_state_dict": peak_residual.state_dict() if peak_residual is not None else None,
                    "selector_state_dict": selector.state_dict() if selector is not None else None,
                    "tf_corr_head_state_dict": tf_corr_head.state_dict() if tf_corr_head is not None else None,
                    "rna_corr_head_state_dict": rna_corr_head.state_dict() if rna_corr_head is not None else None,
                    "re_corr_head_state_dict": re_corr_head.state_dict() if re_corr_head is not None else None,
                    "selected_peak_raw_ids": selected_peak_raw_ids,
                    "cluster_summary_vecs": train_cluster_summary_vecs,
                    "cfg_base": cfg,
                    "hyper": {
                        "STEPS": int(STEPS),
                        "LR": float(LR),
                        "K_RANK": int(K_RANK),
                        "TF_MIX_ALPHA": float(TF_MIX_ALPHA),
                        "N_CLUSTERS": int(N_CLUSTERS),
                        "CLUSTER_NMIN_TRAIN": int(CLUSTER_NMIN_TRAIN),
                        "GROUP_SCALE_ALPHA": float(GROUP_SCALE_ALPHA),
                        "USE_STRATIFIED_BATCH": bool(USE_STRATIFIED_BATCH),
                        "PEAK_HASH_SIZE": int(peak_hash_size),
                        "USE_SELECTED_PEAK_RESIDUAL": bool(USE_SELECTED_PEAK_RESIDUAL),
                        "N_SELECTED_PEAKS": int(selected_peak_raw_ids.size),
                        "USE_CLUSTER_CORRECTION": bool(USE_CLUSTER_CORRECTION),
                        "CLUSTER_CORRECTION_USE_RAW_CLUSTER": bool(CLUSTER_CORRECTION_USE_RAW_CLUSTER),
                    },
                }, best_ckpt_path)
                log(f"[{dataset_name}] [SAVE] BEST ckpt -> {best_ckpt_path} (val_auc={best_auc:.4f} @step={step})")

    if best_state_cpu is not None:
        adapter.load_state_dict(best_state_cpu["adapter"], strict=True)
        if peak_residual is not None and best_state_cpu["peak_residual"] is not None:
            peak_residual.load_state_dict(best_state_cpu["peak_residual"], strict=True)
        if selector is not None and best_state_cpu["selector"] is not None:
            selector.load_state_dict(best_state_cpu["selector"], strict=True)
        if tf_corr_head is not None and best_state_cpu["tf_corr_head"] is not None:
            tf_corr_head.load_state_dict(best_state_cpu["tf_corr_head"], strict=True)
        if rna_corr_head is not None and best_state_cpu["rna_corr_head"] is not None:
            rna_corr_head.load_state_dict(best_state_cpu["rna_corr_head"], strict=True)
        if re_corr_head is not None and best_state_cpu["re_corr_head"] is not None:
            re_corr_head.load_state_dict(best_state_cpu["re_corr_head"], strict=True)
        log(f"[{dataset_name}] loaded BEST by VAL TF(mixed) AUC = {best_auc:.4f}")

    val_tf = eval_tf_auc(
        Zt_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, REG_val_tf, val_groups_sid, cl_val, QC_TF_SAMPLE
    ) if len(va) > 0 else {"auc_mean": np.nan, "n_tf_eval": 0, "cluster_rows": []}
    test_tf = eval_tf_auc(
        Zt_test, test_group_ids, test_cancer_ids, test_cluster_for_corr, REG_test_tf, test_groups_sid, cl_test, QC_TF_SAMPLE
    )

    val_re = eval_re_proxy(va, Zt_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, gene_pool_train) if len(va) > 0 else {"spearman_mean": np.nan, "rank_auc_mean": np.nan, "n_genes_eval": 0}
    test_re = eval_re_proxy(te, Zt_test, test_group_ids, test_cancer_ids, test_cluster_for_corr, gene_pool_train)

    val_rna = eval_rna_retrieval(va, Zt_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, G_top_va, V_top_va, QC_GENE_CELL_EVAL) if len(va) > 0 else {"retrieval_auc": np.nan, "n_scored": 0}
    test_rna = eval_rna_retrieval(te, Zt_test, test_group_ids, test_cancer_ids, test_cluster_for_corr, G_top_te, V_top_te, QC_GENE_CELL_EVAL)

    cluster_eval_rows = []
    for split_name, rows_in in [("val", val_tf["cluster_rows"]), ("test", test_tf["cluster_rows"])]:
        for cid_raw, ncell, auc in rows_in:
            cluster_eval_rows.append({
                "split": split_name,
                "cluster_id": int(cid_raw),
                "n_cells": int(ncell),
                "tf_auc_mixed": float(auc),
            })
    pd.DataFrame(cluster_eval_rows).to_csv(ds_out / "cluster_eval.csv", index=False)

    final_row = {
        "dataset": dataset_name,
        "base_ckpt": str(base_ckpt_path),
        "frontier_ckpt": str(best_ckpt_path),
        "best_val_tf_auc_mixed": float(best_auc),
        "val_tf_auc_mixed": float(val_tf["auc_mean"]),
        "test_tf_auc_mixed": float(test_tf["auc_mean"]),
        "val_tf_n_eval": int(val_tf["n_tf_eval"]),
        "test_tf_n_eval": int(test_tf["n_tf_eval"]),
        "val_re_spearman": float(val_re["spearman_mean"]),
        "test_re_spearman": float(test_re["spearman_mean"]),
        "val_re_rank_auc": float(val_re["rank_auc_mean"]),
        "test_re_rank_auc": float(test_re["rank_auc_mean"]),
        "val_re_n_eval": int(val_re["n_genes_eval"]),
        "test_re_n_eval": int(test_re["n_genes_eval"]),
        "val_rna_retrieval_auc": float(val_rna["retrieval_auc"]),
        "test_rna_retrieval_auc": float(test_rna["retrieval_auc"]),
        "val_rna_n_scored": int(val_rna["n_scored"]),
        "test_rna_n_scored": int(test_rna["n_scored"]),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_group": int(max(1, train_group_ids.max() + 1)),
        "clusters_k": int(C.shape[0]),
        "n_selected_peaks": int(selected_peak_raw_ids.size),
        "use_cluster_correction": int(USE_CLUSTER_CORRECTION),
        "cluster_corr_raw_cluster": int(CLUSTER_CORRECTION_USE_RAW_CLUSTER),
        "n_marker_labels": int(len(marker_prior)),
    }
    pd.DataFrame([final_row]).to_csv(ds_out / "final_metrics.csv", index=False)
    return final_row


# ============================================================
# 9) MAIN
# ============================================================
def main():
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing {GLOBAL_TF_VOCAB}"
    assert GLOBAL_PEAK_VOCAB.exists(), f"Missing {GLOBAL_PEAK_VOCAB}"
    assert PRIORS_DIR.exists(), f"Missing {PRIORS_DIR}"
    assert PROJECTED_ROOT.exists(), f"Missing {PROJECTED_ROOT}"
    assert len(BASE_CKPT_MAP) > 0, "Please fill BASE_CKPT_MAP first."

    datasets = sorted(BASE_CKPT_MAP.keys())
    if RUN_ONLY_DATASET is not None:
        datasets = [x for x in datasets if x == RUN_ONLY_DATASET]
    assert len(datasets) > 0, "No datasets selected to run."

    log("[DATASETS]", datasets)

    all_rows = []
    failed_rows = []

    for ds in datasets:
        try:
            row = run_one_dataset(ds, Path(BASE_CKPT_MAP[ds]))
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(SUMMARY_TSV, sep="\t", index=False)
            log("[SAVE]", SUMMARY_TSV)
        except Exception as e:
            log(f"[ERROR] dataset={ds} err={repr(e)}")
            failed_rows.append({
                "dataset": ds,
                "base_ckpt": str(BASE_CKPT_MAP[ds]),
                "error": repr(e),
            })
            pd.DataFrame(failed_rows).to_csv(FAILED_TSV, sep="\t", index=False)
            log("[SAVE]", FAILED_TSV)

    if len(all_rows) > 0:
        pd.DataFrame(all_rows).to_csv(SUMMARY_TSV, sep="\t", index=False)
        log("[FINAL SAVE]", SUMMARY_TSV)
    if len(failed_rows) > 0:
        pd.DataFrame(failed_rows).to_csv(FAILED_TSV, sep="\t", index=False)
        log("[FINAL SAVE]", FAILED_TSV)
    log("[DONE]")


if __name__ == "__main__":
    main()
