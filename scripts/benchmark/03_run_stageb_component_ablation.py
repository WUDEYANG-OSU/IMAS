from __future__ import annotations

# ============================================================
# stageB_component_ablation_integrated.py
# ============================================================
# 一段式完整 Stage-B component-wise ablation 脚本
#
# 目的：回应审稿人关于 Stage-B 适配组件归因的问题：
#   1) low-rank hyper-adapters
#   2) cluster-aware correction modules
#   3) selected-peak residual branch
#   4) simple fine-tuning control
#
# 输出：
#   $IMAS_PROJECT_ROOT/runs/stageB_component_ablation_integrated/<mode>/CRC-Tissue/
#       best_ckpt.pt
#       clusters.csv
#       cluster_top500_genes.csv
#       cluster_eval.csv
#       final_metrics.csv
#   $IMAS_PROJECT_ROOT/runs/stageB_component_ablation_integrated/stageB_ablation_summary.csv
#
# 运行：
#   python stageB_component_ablation_integrated.py
#
# 只跑一个模式：
#   ABLATION_MODE=no_cluster_correction python stageB_component_ablation_integrated.py
#
# 快速调试：
#   FAST_ABLATION_DEBUG=1 ABLATION_MODE=full_stageB python stageB_component_ablation_integrated.py
# ============================================================

import os
import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


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

BASE_CKPT_MAP: dict[str, Path] = {
    "CRC-Tissue": get_dataset_checkpoint("CRC-Tissue"),
}
RUN_ONLY_DATASET = "CRC-Tissue"
DEFAULT_CANCER_NAME = None

ROOT_RUN_DIR = BASE / "runs" / "stageB_component_ablation_integrated"
ROOT_RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Need CUDA"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

# ---------- selector ----------
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
CLUSTER_TOP_GENE_N = 500

# ---------- group adaptor ----------
GROUP_SCALE_ALPHA = 0.12

# ---------- stratified batching ----------
USE_STRATIFIED_BATCH = True
STRAT_MIN_GROUP_SIZE = 200
STRAT_PER_GROUP_MIN = 128

# ---------- selected peak residual ----------
SELECTED_PEAK_MAX = 50_000
SELECTED_PEAK_MIN_EDGE_FREQ = 2
SELECTED_PEAK_RESIDUAL_L2 = 1e-5
SELECTED_PEAK_ONLY_FROM_FILTERED_GENES = True

# ---------- cluster correction ----------
CLUSTER_CORRECTION_ALPHA_TF = 0.25
CLUSTER_CORRECTION_ALPHA_RNA = 0.25
CLUSTER_CORRECTION_ALPHA_RE = 0.25
CLUSTER_CORRECTION_HID = 256
CLUSTER_CORRECTION_L2 = 1e-5
CLUSTER_CORRECTION_USE_RAW_CLUSTER = True

# ---------- marker prior ----------
USE_CLUSTER_ANNOTATION = False
MARKER_PRIOR_PATH = None
ANNOT_SUMMARY_MAX_GENES = 256

# ---------- final QC / eval ----------
QC_TF_SAMPLE = None
QC_RE_GENE_EVAL = 256
QC_RE_MIN_FINITE = 256
QC_GENE_CELL_EVAL = 1200
QC_GENE_HARD_NEG = True

FAST_ABLATION_DEBUG = bool(int(os.environ.get("FAST_ABLATION_DEBUG", "0")))
if FAST_ABLATION_DEBUG:
    STEPS = min(STEPS, 1200)

ALL_ABLATION_MODES = [
    "full_stageB",
    "simple_finetune",
    "no_lowrank_adapter",
    "no_cluster_correction",
    "no_selected_peak_residual",
    "no_group_scale_bias",
    "no_cellgroup_adapter",
]
ENV_MODE = os.environ.get("ABLATION_MODE", "").strip()
MODES_TO_RUN = [ENV_MODE] if ENV_MODE else ALL_ABLATION_MODES


def log(*args):
    print(*args, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AblationConfig:
    mode: str
    use_selector: bool = True
    use_lowrank_delta: bool = True
    use_cancer_adaptor: bool = True
    use_cellgroup_adaptor: bool = True
    use_group_scale_bias: bool = True
    use_selected_peak_residual: bool = True
    use_cluster_correction: bool = True


def make_ablation_config(mode: str) -> AblationConfig:
    cfg = AblationConfig(mode=mode)
    if mode == "full_stageB":
        pass
    elif mode == "simple_finetune":
        # 简单 fine-tuning 对照：去掉 cluster/group/peak residual/selector，
        # 仅保留全局 low-rank target delta，避免没有可训练参数。
        cfg.use_selector = False
        cfg.use_lowrank_delta = True
        cfg.use_cancer_adaptor = True
        cfg.use_cellgroup_adaptor = False
        cfg.use_group_scale_bias = False
        cfg.use_selected_peak_residual = False
        cfg.use_cluster_correction = False
    elif mode == "no_lowrank_adapter":
        # 去掉 low-rank delta；保留 cluster correction 和 selected-peak residual。
        cfg.use_lowrank_delta = False
        cfg.use_cancer_adaptor = False
        cfg.use_cellgroup_adaptor = False
    elif mode == "no_cluster_correction":
        cfg.use_cluster_correction = False
    elif mode == "no_selected_peak_residual":
        cfg.use_selected_peak_residual = False
    elif mode == "no_group_scale_bias":
        cfg.use_group_scale_bias = False
    elif mode == "no_cellgroup_adapter":
        cfg.use_cellgroup_adaptor = False
        cfg.use_group_scale_bias = False
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")
    return cfg


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
    keep = y >= 0
    return y, keep


def auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores).reshape(-1)
    neg = np.asarray(neg_scores).reshape(-1)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size < 1 or neg.size < 1:
        return np.nan
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    r_pos = ranks[: pos.size]
    auc = (r_pos.sum() - pos.size * (pos.size - 1) / 2) / (pos.size * neg.size + 1e-12)
    return float(auc)


def spearmanr_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
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
        if all(k in n for k in include_keys):
            hits.append(p)
    return hits[0] if hits else None


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def load_marker_prior(marker_prior_path: Path | None):
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
    return {}


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
            if p.name != f"{sid}_ATAC_geneActivity_ALLGENES.npz":
                log(f"[ANCHOR_MAP] RNA sid={sid} -> anchor file={p.name}")
            return p
    raise AssertionError("Missing anchor for sid=" + sid + "\n" + "\n".join(str(x) for x in candidates))


# ============================================================
# 2) BACKBONE
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
        self.rna_proj = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d))
        self.atac_proj = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d))

    @staticmethod
    def _seg_ids_from_offsets(offsets: torch.Tensor, total_tokens: int):
        B = offsets.shape[0]
        end = torch.empty((B,), device=offsets.device, dtype=torch.long)
        end[:-1] = offsets[1:]
        end[-1] = total_tokens
        lengths = (end - offsets).clamp(min=0)
        return torch.repeat_interleave(torch.arange(B, device=offsets.device), lengths)

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
        anchor_path = resolve_anchor_path(anchor_dir, sid)
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
        assert counts_mm.shape[0] == len(row_indices)
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
        assert matched > 1000, f"{dataset_name}/{sid}: too few matched genes: {matched}"
        anchor_gene2col, anchor_cells, anchor_X = load_anchor_all(anchor_path)
        m = {str(c): i for i, c in enumerate(anchor_cells.tolist())}
        rid2arow = np.full((len(cells),), -1, dtype=np.int64)
        for i, bc in enumerate(cells.tolist()):
            rid2arow[i] = int(m.get(str(bc), -1))
        depth_rna = np.asarray(z["depth_rna"], dtype=np.float32) if "depth_rna" in z else None
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
        log(f"[LOAD] {dataset_name}/{sid} cells={s.counts_shape[0]} genes={s.counts_shape[1]} matched_genes={matched} anchor_shape={anchor_X.shape}")
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
    train, test = [], []
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
    idx_list, val_list, offsets = [], [], [0]
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
        b = cell_list[st : st + EMB_BATCH]
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
        rows, cols, data = [], [], []
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
        W = sp.coo_matrix((np.asarray(data, np.float32), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))), shape=(int(s.counts_shape[1]), int(n_tf))).tocsr()
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
            m = y == j
            if m.sum() < 1:
                C[j] = Zfit[np.random.randint(0, Zfit.shape[0])]
            else:
                C[j] = Zfit[m].mean(axis=0)
        C = _l2norm_rows(C)
    return C.astype(np.float32)


def spherical_kmeans_assign(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    return (_l2norm_rows(Z) @ _l2norm_rows(C).T).argmax(axis=1).astype(np.int64)


def build_group_ids_from_clusters(train_cluster: np.ndarray, all_cluster: np.ndarray, nmin: int):
    cnt = np.bincount(train_cluster, minlength=int(train_cluster.max()) + 1)
    keep = np.where(cnt >= int(nmin))[0].tolist()
    cl2gid = {int(cl): i + 1 for i, cl in enumerate(sorted(keep))}
    group_ids = np.zeros_like(all_cluster, dtype=np.int64)
    for i, cl in enumerate(all_cluster.tolist()):
        group_ids[i] = int(cl2gid.get(int(cl), 0))
    return group_ids, cl2gid, cnt


# ============================================================
# 5) CLUSTER SUMMARY / ANNOTATION
# ============================================================

def build_cluster_summaries(cell_list, sid2sample, Z, cluster_ids, genes_vocab, marker_prior=None):
    marker_prior = marker_prior or {}
    marker_labels = sorted(marker_prior.keys())
    gene2id = {g: i for i, g in enumerate(genes_vocab)}
    summaries = {}
    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        z_mean = Z[idx].mean(axis=0).astype(np.float32)
        rna_nnz, rna_logmean = [], []
        marker_scores = np.zeros((len(marker_labels),), dtype=np.float32)
        for ii in idx.tolist():
            sid, rid = cell_list[ii]
            s = sid2sample[sid]
            row = get_counts_row_dense(s, rid)
            rna_nnz.append(float((row > 0).mean()))
            nz = row[row > 0]
            rna_logmean.append(float(np.log1p(nz).mean()) if nz.size else 0.0)
            for j, label in enumerate(marker_labels):
                vals = []
                for g in marker_prior[label][:ANNOT_SUMMARY_MAX_GENES]:
                    gid = gene2id.get(str(g), None)
                    if gid is not None and gid in s.train2rna:
                        vals.append(float(row[s.train2rna[gid]]))
                if vals:
                    marker_scores[j] += float(np.mean(vals))
        if idx.size > 0 and len(marker_scores) > 0:
            marker_scores /= float(idx.size)
        summaries[int(cid)] = {
            "z_mean": z_mean,
            "rna_nnz_mean": float(np.mean(rna_nnz)) if rna_nnz else 0.0,
            "rna_logmean": float(np.mean(rna_logmean)) if rna_logmean else 0.0,
            "size": int(idx.size),
            "marker_scores": marker_scores.astype(np.float32),
            "marker_labels": marker_labels,
        }
    return summaries


def cluster_summary_to_vector(summary: dict) -> np.ndarray:
    z_mean = np.asarray(summary["z_mean"], dtype=np.float32).reshape(-1)
    meta = np.asarray([float(summary["rna_nnz_mean"]), float(summary["rna_logmean"]), float(summary["size"])], dtype=np.float32)
    marker_scores = np.asarray(summary.get("marker_scores", []), dtype=np.float32).reshape(-1)
    return np.concatenate([z_mean, meta, marker_scores], axis=0).astype(np.float32)


def export_cluster_top_genes(cell_list, sid2sample, cluster_ids, genes_vocab, out_csv: Path, out_label_csv: Path, top_n: int = 500):
    rows, label_rows = [], []
    uniq = sorted(np.unique(cluster_ids).tolist())
    cluster_label_map = {int(cid): f"C{i + 1}" for i, cid in enumerate(uniq)}
    for cid in uniq:
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        sum_expr, n_cells = None, 0
        for ii in idx.tolist():
            sid, rid = cell_list[ii]
            row = get_counts_row_dense(sid2sample[sid], rid).astype(np.float32, copy=False)
            row = np.log1p(np.clip(row, 0.0, None))
            if sum_expr is None:
                sum_expr = np.zeros_like(row, dtype=np.float64)
            sum_expr += row
            n_cells += 1
        if sum_expr is None or n_cells == 0:
            continue
        mean_expr = (sum_expr / float(n_cells)).astype(np.float32)
        k = min(int(top_n), mean_expr.size)
        top_idx = np.argpartition(mean_expr, -k)[-k:]
        top_vals = mean_expr[top_idx]
        order = np.argsort(top_vals)[::-1]
        top_idx = top_idx[order]
        top_vals = top_vals[order]
        cluster_label = cluster_label_map[int(cid)]
        label_rows.append({"cluster_id": int(cid), "cluster_label": cluster_label, "n_cells": int(n_cells), "manual_annotation": ""})
        for rank, (gidx, score) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), start=1):
            gene = str(genes_vocab[int(gidx)]) if int(gidx) < len(genes_vocab) else str(gidx)
            rows.append({"cluster_id": int(cid), "cluster_label": cluster_label, "rank": rank, "gene": gene, "mean_log1p_expr": float(score), "n_cells": int(n_cells)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(label_rows).to_csv(out_label_csv, index=False)


class ClusterSummaryTable(nn.Module):
    def __init__(self, cluster_summary_np: dict[int, np.ndarray]):
        super().__init__()
        keys = sorted(cluster_summary_np.keys())
        arr = np.stack([cluster_summary_np[k] for k in keys], axis=0).astype(np.float32)
        self.cluster_ids = keys
        self.register_buffer("summary_table", torch.from_numpy(arr), persistent=True)
        self.cid2row = {int(cid): i for i, cid in enumerate(keys)}

    @property
    def summary_dim(self) -> int:
        return int(self.summary_table.shape[1])

    def lookup(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        idx = [self.cid2row.get(int(cid), 0) for cid in cluster_ids.detach().cpu().numpy().tolist()]
        idx_t = torch.tensor(idx, device=cluster_ids.device, dtype=torch.long)
        return self.summary_table[idx_t]


# ============================================================
# 6) ADAPTER + SELECTOR + RESIDUAL + CORRECTION
# ============================================================

class HyperAdapterHashV2(nn.Module):
    def __init__(self, d: int, k: int, n_tf: int, n_gene: int, peak_hash_size: int, n_cancer: int, n_group: int, e_dim: int = 64, hid: int = 256, ablation_cfg: AblationConfig | None = None):
        super().__init__()
        self.d = d
        self.k = k
        self.peak_hash_size = int(peak_hash_size)
        self.ablation_cfg = ablation_cfg or AblationConfig(mode="full_stageB")
        self.e = nn.Parameter(torch.zeros((e_dim,), dtype=torch.float32))
        nn.init.normal_(self.e, mean=0.0, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(e_dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 4 * k * d))
        self.C_tf = nn.Embedding(n_tf, k)
        self.C_gene = nn.Embedding(n_gene, k)
        self.C_peak_hash = nn.Embedding(self.peak_hash_size, k)
        self.C_cancer = nn.Embedding(n_cancer, k)
        self.S_group = nn.Embedding(n_group, d)
        self.B_group = nn.Embedding(n_group, d)
        for emb in [self.C_tf, self.C_gene, self.C_peak_hash, self.C_cancer, self.S_group, self.B_group]:
            nn.init.zeros_(emb.weight)

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
        if not self.ablation_cfg.use_group_scale_bias:
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
        self.mlp = nn.Sequential(nn.Linear(32 + summary_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 6))
        self.register_buffer("summary", torch.zeros((summary_dim,), dtype=torch.float32), persistent=True)

    def set_summary(self, summary_np: np.ndarray):
        s = np.asarray(summary_np, dtype=np.float32).reshape(-1)
        assert s.size == self.summary.numel()
        self.summary.copy_(torch.from_numpy(s))

    def forward(self) -> torch.Tensor:
        x = torch.cat([self.e, self.summary], dim=0)
        logits = self.mlp(x)
        w = F.softplus(logits) + float(SELECTOR_SOFTPLUS_BIAS)
        return w / (w.mean() + 1e-12)


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
        self.mlp = nn.Sequential(nn.Linear(z_dim + summary_dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, z_dim))

    def forward(self, z_cell: torch.Tensor, cluster_summary: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z_cell, cluster_summary], dim=-1))


# ============================================================
# 7) SUMMARIES / FILTERS / MINING
# ============================================================

def compute_dataset_summary(train_cells, sid2sample, tf2targets, tf_in_prior, N_GENE_DIM) -> np.ndarray:
    if len(train_cells) == 0:
        return np.zeros((8,), np.float32)
    n = min(SELECTOR_SUMMARY_NCELLS, len(train_cells))
    idx = np.random.choice(np.arange(len(train_cells)), size=n, replace=False)
    sub = [train_cells[i] for i in idx.tolist()]
    r_means, r_nnz = [], []
    for sid, rid in sub:
        row = get_counts_row_dense(sid2sample[sid], rid)
        r_nnz.append(float((row > 0).mean()))
        nz = row[row > 0]
        r_means.append(float(np.log1p(nz).mean()) if nz.size else 0.0)
    reg = regproxy_aucell_lite_from_top(sub, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM)
    tf_cols = np.arange(reg.shape[1])
    if tf_cols.size > 256:
        tf_cols = np.random.choice(tf_cols, size=256, replace=False)
    tf_vals = reg[:, tf_cols]
    return np.asarray([
        float(np.mean(r_means)), float(np.std(r_means) + 1e-6),
        float(np.mean(r_nnz)), float(np.std(r_nnz) + 1e-6),
        float(np.nanmean(tf_vals)), float(np.nanstd(tf_vals) + 1e-6),
        0.0, 0.0,
    ], dtype=np.float32)


def build_filtered_gene_pool_from_train(train_cells, sid2sample, gene_pool_full: np.ndarray, genes_vocab: list[str]) -> np.ndarray:
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
    sid2arows = {sid: np.asarray(v, dtype=np.int64) for sid, v in sid2arows.items()}
    gene_ids = gene_pool_full.copy()
    if ATAC_FILTER_MAX_GENES is not None and gene_ids.size > int(ATAC_FILTER_MAX_GENES):
        gene_ids = np.random.choice(gene_ids, size=int(ATAC_FILTER_MAX_GENES), replace=False)
    gene_ids = np.asarray(sorted(gene_ids.tolist()), dtype=np.int64)
    keep, nnz_fracs, stds = [], [], []
    for gid in gene_ids.tolist():
        if not (0 <= int(gid) < len(genes_vocab)):
            continue
        gname = genes_vocab[int(gid)]
        vals = []
        for sid, arows in sid2arows.items():
            s = sid2sample[sid]
            col = s.anchor_gene2col.get(gname, None)
            if col is not None:
                vals.append(s.anchor_X[arows, int(col)].astype(np.float32, copy=False))
        if not vals:
            continue
        y = np.concatenate(vals, axis=0).astype(np.float32, copy=False)
        nnz = float((y > 0).mean())
        sd = float(np.std(np.log1p(y)))
        nnz_fracs.append(nnz)
        stds.append(sd)
        if nnz >= ATAC_FILTER_MIN_NNZ_FRAC and sd >= ATAC_FILTER_MIN_STD_LOG1P:
            keep.append(int(gid))
    keep = np.asarray(keep, dtype=np.int64)
    log(f"[ATAC_FILTER] sampled_cells={n} genes_in={gene_pool_full.size} checked={gene_ids.size} kept={keep.size}")
    if keep.size < 1024:
        log("[ATAC_FILTER][WARN] kept genes too few; fallback full pool")
        return gene_pool_full
    return keep


def mine_selected_peaks(gene_pool: np.ndarray, gene2peaks: dict[int, list[int]]) -> np.ndarray:
    counter = Counter()
    for gid in gene_pool.tolist():
        for pk in gene2peaks.get(int(gid), []):
            counter[int(pk)] += 1
    if len(counter) == 0:
        return np.zeros((0,), dtype=np.int64)
    items = [(pk, cnt) for pk, cnt in counter.items() if cnt >= SELECTED_PEAK_MIN_EDGE_FREQ]
    if len(items) == 0:
        items = list(counter.items())
    items = sorted(items, key=lambda x: (-x[1], x[0]))[:SELECTED_PEAK_MAX]
    out = np.asarray([int(pk) for pk, _ in items], dtype=np.int64)
    log(f"[SELECTED_PEAK] n_selected={out.size} from_gene_pool={gene_pool.size}")
    return out


# ============================================================
# 8) TRAIN/EVAL CORE
# ============================================================

@dataclass
class RunContext:
    cfg: AblationConfig
    dataset_name: str
    ds_out: Path
    model: PlanC_DirAware_Cancer_HashPeak
    adapter: HyperAdapterHashV2
    selector: PriorSelectorV1 | None
    peak_residual: SelectedPeakResidual | None
    tf_corr_head: ClusterCellCorrectionHead | None
    rna_corr_head: ClusterCellCorrectionHead | None
    re_corr_head: ClusterCellCorrectionHead | None
    cluster_summary_table: ClusterSummaryTable
    default_cancer_id: int
    cid: torch.Tensor
    tf_in_prior: list[int]
    genes_union: list[int]
    gene2peaks: dict[int, list[int]]
    genes_vocab: list[str]


def apply_cluster_correction(z: torch.Tensor, cluster_ids_np: np.ndarray, corr_head, alpha: float, ctx: RunContext) -> torch.Tensor:
    if (not ctx.cfg.use_cluster_correction) or corr_head is None:
        return z
    cl_t = torch.from_numpy(cluster_ids_np.astype(np.int64)).to(DEVICE)
    summary = ctx.cluster_summary_table.lookup(cl_t)
    dz = corr_head(z, summary)
    return F.normalize(z + float(alpha) * dz, dim=-1)


def adjust_Z(Zt, group_ids_np, cancer_ids_np, cluster_ids_np, V_cancer, task: str, ctx: RunContext):
    Zadj = Zt
    if ctx.cfg.use_cellgroup_adaptor:
        gid_t = torch.from_numpy(group_ids_np.astype(np.int64)).to(DEVICE)
        Zadj = ctx.adapter.apply_group_to_Z(Zadj, gid_t)
    if ctx.cfg.use_cancer_adaptor and ctx.cfg.use_lowrank_delta:
        cid_t = torch.from_numpy(cancer_ids_np.astype(np.int64)).to(DEVICE)
        Zadj = Zadj + ctx.adapter.delta_cancer(cid_t, V_cancer)
    Zadj = F.normalize(Zadj, dim=-1)
    if task == "tf":
        return apply_cluster_correction(Zadj, cluster_ids_np, ctx.tf_corr_head, CLUSTER_CORRECTION_ALPHA_TF, ctx)
    if task == "rna":
        return apply_cluster_correction(Zadj, cluster_ids_np, ctx.rna_corr_head, CLUSTER_CORRECTION_ALPHA_RNA, ctx)
    if task == "re":
        return apply_cluster_correction(Zadj, cluster_ids_np, ctx.re_corr_head, CLUSTER_CORRECTION_ALPHA_RE, ctx)
    return Zadj


def tf_base(tf_pid: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    tf_ids = torch.tensor([ctx.tf_in_prior[i] for i in tf_pid.tolist()], device=DEVICE, dtype=torch.long)
    return ctx.model.tf_in(tf_ids) + ctx.model.tf_in_c(ctx.cid).expand(tf_ids.numel(), -1)


def gene_base_from_pid(gene_pid: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    gene_ids = torch.tensor([ctx.genes_union[i] for i in gene_pid.tolist()], device=DEVICE, dtype=torch.long)
    return ctx.model.gene_out(gene_ids) + ctx.model.gene_out_c(ctx.cid).expand(gene_ids.numel(), -1)


def peak_base_from_raw(peak_raw_t: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    return ctx.model.peak_out(peak_raw_t) + ctx.model.peak_out_c(ctx.cid).expand(peak_raw_t.numel(), -1)


def score_tf_vec(tf_pid_t: torch.Tensor, V_tf: torch.Tensor, V_cancer: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    TF = tf_base(tf_pid_t, ctx)
    if ctx.cfg.use_lowrank_delta:
        TF = TF + ctx.adapter.delta_tf(tf_pid_t, V_tf)
    if ctx.cfg.use_cancer_adaptor and ctx.cfg.use_lowrank_delta:
        c_tok = torch.full((tf_pid_t.numel(),), int(ctx.default_cancer_id), device=DEVICE, dtype=torch.long)
        TF = TF + ctx.adapter.delta_cancer(c_tok, V_cancer)
    return F.normalize(TF, dim=-1)


def score_gene_vec(gene_pid_t: torch.Tensor, V_gene: torch.Tensor, V_cancer: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    G = gene_base_from_pid(gene_pid_t, ctx)
    if ctx.cfg.use_lowrank_delta:
        G = G + ctx.adapter.delta_gene(gene_pid_t, V_gene)
    if ctx.cfg.use_cancer_adaptor and ctx.cfg.use_lowrank_delta:
        c_tok = torch.full((gene_pid_t.numel(),), int(ctx.default_cancer_id), device=DEVICE, dtype=torch.long)
        G = G + ctx.adapter.delta_cancer(c_tok, V_cancer)
    return F.normalize(G, dim=-1)


def score_peak_vec(pool_peak_raw_np: np.ndarray, V_peak: torch.Tensor, V_cancer: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    peak_raw_t = torch.from_numpy(pool_peak_raw_np.astype(np.int64)).to(DEVICE)
    P = peak_base_from_raw(peak_raw_t, ctx)
    if ctx.cfg.use_lowrank_delta:
        P = P + ctx.adapter.delta_peak_hash(peak_raw_t, V_peak)
    if ctx.peak_residual is not None and ctx.peak_residual.n_selected > 0:
        P = P + ctx.peak_residual.lookup(peak_raw_t)
    if ctx.cfg.use_cancer_adaptor and ctx.cfg.use_lowrank_delta:
        c_tok = torch.full((peak_raw_t.numel(),), int(ctx.default_cancer_id), device=DEVICE, dtype=torch.long)
        P = P + ctx.adapter.delta_cancer(c_tok, V_cancer)
    return F.normalize(P, dim=-1)


def gene_peak_vec_for_gene_pid(gene_pid: int, V_peak, V_cancer, ctx: RunContext):
    gid = int(ctx.genes_union[int(gene_pid)])
    peaks = ctx.gene2peaks.get(gid, [])
    if not peaks:
        return None
    if len(peaks) > RE_POOL_MAX:
        peaks = peaks[:RE_POOL_MAX]
    P = score_peak_vec(np.asarray(peaks, dtype=np.int64), V_peak, V_cancer, ctx)
    return F.normalize(P.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


def sample_train_indices(train_group_ids: np.ndarray, big_groups: list[int], n: int) -> np.ndarray:
    if not USE_STRATIFIED_BATCH or len(big_groups) == 0:
        return np.random.choice(np.arange(train_group_ids.size), size=min(n, train_group_ids.size), replace=False)
    gid2idx = defaultdict(list)
    for i, g in enumerate(train_group_ids.tolist()):
        gid2idx[int(g)].append(i)
    chunks = []
    per = max(1, min(STRAT_PER_GROUP_MIN, n // max(1, len(big_groups))))
    for g in big_groups:
        arr = np.asarray(gid2idx[g], dtype=np.int64)
        if arr.size == 0:
            continue
        take = min(per, arr.size)
        chunks.append(np.random.choice(arr, size=take, replace=False))
    if chunks:
        idx = np.concatenate(chunks)
        if idx.size < n:
            rest = np.random.choice(np.arange(train_group_ids.size), size=min(n - idx.size, train_group_ids.size), replace=False)
            idx = np.unique(np.concatenate([idx, rest]))
        return idx[: min(n, idx.size)]
    return np.random.choice(np.arange(train_group_ids.size), size=min(n, train_group_ids.size), replace=False)


def build_anchor_targets_for_cells(cell_indices: np.ndarray, cell_list, sid2sample, gene_global_ids: list[int], genes_vocab: list[str]) -> np.ndarray:
    Y = np.full((cell_indices.size, len(gene_global_ids)), np.nan, dtype=np.float32)
    for ii, ci in enumerate(cell_indices.tolist()):
        sid, rid = cell_list[int(ci)]
        s = sid2sample[sid]
        arow = int(s.rid2arow[int(rid)])
        if arow < 0:
            continue
        for j, gid in enumerate(gene_global_ids):
            if 0 <= int(gid) < len(genes_vocab):
                col = s.anchor_gene2col.get(genes_vocab[int(gid)], None)
                if col is not None:
                    Y[ii, j] = float(s.anchor_X[arow, int(col)])
    return Y


def loss_l2_module(m: nn.Module | None) -> torch.Tensor:
    if m is None:
        return torch.tensor(0.0, device=DEVICE)
    out = torch.tensor(0.0, device=DEVICE)
    for p in m.parameters():
        out = out + (p.float() ** 2).mean()
    return out


@torch.no_grad()
def eval_tf_auc(Z_np, group_ids, cancer_ids, cluster_ids, REG_tf, ctx: RunContext, max_tf=None):
    if Z_np is None or len(Z_np) == 0 or REG_tf is None:
        return {"auc_mean": np.nan, "n_tf_eval": 0, "cluster_rows": []}
    V_tf, V_gene, V_peak, V_cancer = ctx.adapter()
    Zt = torch.from_numpy(Z_np).to(DEVICE)
    Zadj = adjust_Z(Zt, group_ids, cancer_ids, cluster_ids, V_cancer, "tf", ctx)
    tf_pids = np.arange(len(ctx.tf_in_prior), dtype=np.int64)
    if max_tf is not None and tf_pids.size > int(max_tf):
        tf_pids = np.random.choice(tf_pids, size=int(max_tf), replace=False)
        tf_pids = np.sort(tf_pids)
    aucs = []
    scores_all = []
    for st in range(0, tf_pids.size, 256):
        pid_np = tf_pids[st: st + 256]
        TF = score_tf_vec(torch.from_numpy(pid_np).to(DEVICE), V_tf, V_cancer, ctx)
        S = (Zadj @ TF.T).detach().cpu().numpy()
        scores_all.append(S)
    pred = np.concatenate(scores_all, axis=1) if scores_all else np.zeros((Z_np.shape[0], 0), np.float32)
    truth = REG_tf[:, tf_pids]
    for j in range(pred.shape[1]):
        y, keep = make_pos_neg_labels(truth[:, j], POS_Q, NEG_Q)
        if keep.sum() < 20 or (y[keep] == 1).sum() < 5 or (y[keep] == 0).sum() < 5:
            continue
        aucs.append(auc_from_scores(pred[keep, j][y[keep] == 1], pred[keep, j][y[keep] == 0]))
    cluster_rows = []
    for cid0 in sorted(np.unique(cluster_ids).tolist()):
        m = cluster_ids == cid0
        if m.sum() < VAL_CLUSTER_MIN_CELLS:
            continue
        caucs = []
        for j in range(pred.shape[1]):
            y, keep = make_pos_neg_labels(truth[m, j], POS_Q, NEG_Q)
            if keep.sum() < VAL_CLUSTER_MIN_POSNEG or (y[keep] == 1).sum() < 4 or (y[keep] == 0).sum() < 4:
                continue
            spred = pred[m, j][keep]
            caucs.append(auc_from_scores(spred[y[keep] == 1], spred[y[keep] == 0]))
        if caucs:
            cluster_rows.append((int(cid0), int(m.sum()), float(np.nanmean(caucs))))
    return {"auc_mean": float(np.nanmean(aucs)) if aucs else np.nan, "n_tf_eval": int(len(aucs)), "cluster_rows": cluster_rows}


@torch.no_grad()
def eval_rna_retrieval_auc(Z_np, group_ids, cancer_ids, cluster_ids, G_top, V_top, ctx: RunContext, n_cells_eval=1200):
    if Z_np is None or len(Z_np) == 0:
        return {"retrieval_auc": np.nan, "n_scored": 0}
    V_tf, V_gene, V_peak, V_cancer = ctx.adapter()
    n = Z_np.shape[0]
    idx = np.arange(n)
    if n > int(n_cells_eval):
        idx = np.random.choice(idx, size=int(n_cells_eval), replace=False)
    Zt = torch.from_numpy(Z_np[idx]).to(DEVICE)
    Zadj = adjust_Z(Zt, group_ids[idx], cancer_ids[idx], cluster_ids[idx], V_cancer, "rna", ctx)
    aucs = []
    n_gene_pid = len(ctx.genes_union)
    for bi, ci in enumerate(idx.tolist()):
        pos_global = G_top[ci]
        pos_global = pos_global[pos_global >= 0]
        if pos_global.size == 0:
            continue
        # global gene id -> pid in genes_union
        g2pid = {int(g): p for p, g in enumerate(ctx.genes_union)}
        pos_pid = np.asarray([g2pid[int(g)] for g in pos_global if int(g) in g2pid], dtype=np.int64)
        if pos_pid.size < 4:
            continue
        if pos_pid.size > RNA_POS_PER_CELL:
            pos_pid = np.random.choice(pos_pid, size=RNA_POS_PER_CELL, replace=False)
        neg_pid = np.random.choice(np.arange(n_gene_pid), size=min(RNA_NEG_PER_CELL, n_gene_pid), replace=False)
        Gp = score_gene_vec(torch.from_numpy(pos_pid).to(DEVICE), V_gene, V_cancer, ctx)
        Gn = score_gene_vec(torch.from_numpy(neg_pid).to(DEVICE), V_gene, V_cancer, ctx)
        zp = Zadj[bi: bi + 1]
        sp = (zp @ Gp.T).detach().cpu().numpy().ravel()
        sn = (zp @ Gn.T).detach().cpu().numpy().ravel()
        aucs.append(auc_from_scores(sp, sn))
    return {"retrieval_auc": float(np.nanmean(aucs)) if aucs else np.nan, "n_scored": int(len(aucs))}


@torch.no_grad()
def eval_re_anchor(Z_np, group_ids, cancer_ids, cluster_ids, cell_list, sid2sample, gene_pool, ctx: RunContext, n_genes_eval=256):
    if Z_np is None or len(Z_np) == 0 or gene_pool.size == 0:
        return {"spearman_mean": np.nan, "rank_auc_mean": np.nan, "n_genes_eval": 0}
    V_tf, V_gene, V_peak, V_cancer = ctx.adapter()
    Zt = torch.from_numpy(Z_np).to(DEVICE)
    Zadj = adjust_Z(Zt, group_ids, cancer_ids, cluster_ids, V_cancer, "re", ctx)
    gene_ids = gene_pool.copy()
    if gene_ids.size > n_genes_eval:
        gene_ids = np.random.choice(gene_ids, size=int(n_genes_eval), replace=False)
    gid2pid = {int(g): p for p, g in enumerate(ctx.genes_union)}
    spears, aucs = [], []
    cell_idx = np.arange(Z_np.shape[0], dtype=np.int64)
    for gid in gene_ids.tolist():
        if int(gid) not in gid2pid:
            continue
        pid = gid2pid[int(gid)]
        pv = gene_peak_vec_for_gene_pid(pid, V_peak, V_cancer, ctx)
        if pv is None:
            continue
        pred = (Zadj @ pv.view(-1, 1)).detach().cpu().numpy().ravel()
        y = build_anchor_targets_for_cells(cell_idx, cell_list, sid2sample, [int(gid)], ctx.genes_vocab).ravel()
        m = np.isfinite(y)
        if m.sum() < QC_RE_MIN_FINITE:
            continue
        spears.append(spearmanr_fast(pred[m], y[m]))
        lab, keep = make_pos_neg_labels(y[m], POS_Q, NEG_Q)
        if keep.sum() >= 20 and (lab[keep] == 1).sum() >= 5 and (lab[keep] == 0).sum() >= 5:
            aucs.append(auc_from_scores(pred[m][keep][lab[keep] == 1], pred[m][keep][lab[keep] == 0]))
    return {"spearman_mean": float(np.nanmean(spears)) if spears else np.nan, "rank_auc_mean": float(np.nanmean(aucs)) if aucs else np.nan, "n_genes_eval": int(len(spears))}


def save_ckpt(path: Path, ctx: RunContext, best_auc: float, step: int, extra: dict):
    obj = {
        "ablation_cfg": asdict(ctx.cfg),
        "adapter": ctx.adapter.state_dict(),
        "selector": ctx.selector.state_dict() if ctx.selector is not None else None,
        "peak_residual": ctx.peak_residual.state_dict() if ctx.peak_residual is not None else None,
        "tf_corr_head": ctx.tf_corr_head.state_dict() if ctx.tf_corr_head is not None else None,
        "rna_corr_head": ctx.rna_corr_head.state_dict() if ctx.rna_corr_head is not None else None,
        "re_corr_head": ctx.re_corr_head.state_dict() if ctx.re_corr_head is not None else None,
        "best_auc": float(best_auc),
        "step": int(step),
        "extra": extra,
    }
    torch.save(obj, path)


def run_one_dataset_one_mode(dataset_name: str, base_ckpt_path: Path, ablation_cfg: AblationConfig):
    set_seed(SEED)
    mode = ablation_cfg.mode
    mode_root = ROOT_RUN_DIR / mode
    ds_out = mode_root / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)
    log("=" * 100)
    log(f"[RUN] dataset={dataset_name} mode={mode}")
    log(f"[ABLATION_CFG] {asdict(ablation_cfg)}")

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)
    marker_prior = load_marker_prior(MARKER_PRIOR_PATH) if USE_CLUSTER_ANNOTATION else {}

    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = base_ckpt.get("model", None)
    assert state_dict is not None, f"BASE ckpt missing model weights: {base_ckpt_path}"
    cfg0 = base_ckpt.get("cfg", {}) or {}
    meta = base_ckpt.get("meta", {}) or {}
    for k in ["N_GENE_DIM", "N_TF_DIM", "PEAK_HASH_SIZE", "D", "HID", "DROP", "cancer_types"]:
        if k in meta:
            cfg0[k] = meta[k]
    cfg0.setdefault("N_GENE_DIM", int(state_dict["gene_in.weight"].shape[0]))
    cfg0.setdefault("N_TF_DIM", int(state_dict["tf_in.weight"].shape[0]))
    cfg0.setdefault("PEAK_HASH_SIZE", int(state_dict["peak_in.emb.weight"].shape[0]))
    cfg0.setdefault("D", int(state_dict["gene_in.weight"].shape[1]))
    cfg0.setdefault("HID", int(state_dict["rna_proj.0.weight"].shape[0]))
    cfg0.setdefault("DROP", 0.1)
    if "cancer_types" not in cfg0:
        n_cancer = int(state_dict["cell_c.weight"].shape[0])
        cfg0["cancer_types"] = [str(get_default_cancer_name(dataset_name))] + [f"unused_{i}" for i in range(1, n_cancer)]

    N_GENE_DIM = int(cfg0["N_GENE_DIM"])
    N_TF_DIM = int(cfg0["N_TF_DIM"])
    peak_hash_size = int(cfg0["PEAK_HASH_SIZE"])
    D = int(cfg0["D"])
    HID = int(cfg0["HID"])
    DROP = float(cfg0["DROP"])
    cancer_types = list(cfg0["cancer_types"])
    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_name = get_default_cancer_name(dataset_name)
    default_cancer_id = int(cancer2id.get(default_cancer_name, 0))
    cid = torch.tensor([default_cancer_id], device=DEVICE, dtype=torch.long)

    model = PlanC_DirAware_Cancer_HashPeak(N_GENE_DIM, N_TF_DIM, len(cancer_types), peak_hash_size, d=D, hid=HID, drop=DROP)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # priors
    tf_gene_prior = PRIORS_DIR / "tf_gene_f.cap256.npz"
    if not tf_gene_prior.exists():
        tf_gene_prior = PRIORS_DIR / "tf_gene_f.full.npz"
    assert tf_gene_prior.exists(), f"Missing tf-gene prior: {tf_gene_prior}"
    z = np.load(tf_gene_prior, allow_pickle=True)
    tf_src = z["src"].astype(np.int64)
    g_dst = z["dst"].astype(np.int64)
    tf2targets = defaultdict(list)
    for s0, d0 in zip(tf_src.tolist(), g_dst.tolist()):
        if 0 <= s0 < len(tfs_vocab) and 0 <= d0 < N_GENE_DIM:
            tf2targets[int(s0)].append(int(d0))
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
    tr_full, te = split_cells_5050(all_cells)
    tr, va = split_train_val(tr_full, VAL_FRAC_FROM_TRAIN)
    log(f"[CELLS] train={len(tr)} val={len(va)} test={len(te)}")

    proj_store_cache = {}
    log("[ENC] encode train/val/test Z ...")
    Z_train = encode_Z(tr, sid2sample, model, default_cancer_id, proj_store_cache)
    Z_val = encode_Z(va, sid2sample, model, default_cancer_id, proj_store_cache) if len(va) else None
    Z_test = encode_Z(te, sid2sample, model, default_cancer_id, proj_store_cache)
    Zt_train = torch.from_numpy(Z_train).to(DEVICE)

    # fixed clustering per mode directory; for full comparability, save/load centroids in root mode-independent path
    centroid_path = ROOT_RUN_DIR / dataset_name / "shared_centroids.npy"
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    if centroid_path.exists():
        C = np.load(centroid_path).astype(np.float32)
    else:
        C = spherical_kmeans_fit(Z_train, k=N_CLUSTERS, iters=CLUSTER_ITERS, max_train=CLUSTER_TRAIN_MAX)
        np.save(centroid_path, C)
    cl_train = spherical_kmeans_assign(Z_train, C)
    cl_val = spherical_kmeans_assign(Z_val, C) if Z_val is not None else None
    cl_test = spherical_kmeans_assign(Z_test, C)
    cl_all = np.concatenate([cl_train, cl_val, cl_test], axis=0) if cl_val is not None else np.concatenate([cl_train, cl_test], axis=0)
    group_all, cl2gid, cnt = build_group_ids_from_clusters(cl_train, cl_all, CLUSTER_NMIN_TRAIN)
    ntr, nva = len(tr), len(va)
    train_group_ids = group_all[:ntr]
    val_group_ids = group_all[ntr: ntr + nva] if cl_val is not None else None
    test_group_ids = group_all[ntr + nva:] if cl_val is not None else group_all[ntr:]
    train_cancer_ids = np.full((len(tr),), default_cancer_id, np.int64)
    val_cancer_ids = np.full((len(va),), default_cancer_id, np.int64) if len(va) else None
    test_cancer_ids = np.full((len(te),), default_cancer_id, np.int64)
    train_cluster_for_corr = cl_train if CLUSTER_CORRECTION_USE_RAW_CLUSTER else train_group_ids
    val_cluster_for_corr = cl_val if (CLUSTER_CORRECTION_USE_RAW_CLUSTER and cl_val is not None) else val_group_ids
    test_cluster_for_corr = cl_test if CLUSTER_CORRECTION_USE_RAW_CLUSTER else test_group_ids
    cluster_label_map = {int(cid0): f"C{i + 1}" for i, cid0 in enumerate(sorted(np.unique(cl_train).tolist()))}

    cluster_rows = []
    for split, cells, cl_raw, gid in [("train", tr, cl_train, train_group_ids), ("val", va, cl_val, val_group_ids), ("test", te, cl_test, test_group_ids)]:
        if cells is None or cl_raw is None:
            continue
        for i, (sid, rid) in enumerate(cells):
            bc = str(sid2sample[sid].cells[int(rid)])
            cluster_rows.append({"split": split, "sid": sid, "rid": int(rid), "barcode": bc, "cell_key": make_cell_key(sid, bc), "cluster_id": int(cl_raw[i]), "cluster_label": cluster_label_map.get(int(cl_raw[i]), "C?"), "group_id": int(gid[i])})
    pd.DataFrame(cluster_rows).to_csv(ds_out / "clusters.csv", index=False)

    log("[ENC] precompute top genes ...")
    G_top_tr, V_top_tr = precompute_top_genes(tr, sid2sample, STORE_TOP_G, N_GENE_DIM)
    G_top_te, V_top_te = precompute_top_genes(te, sid2sample, STORE_TOP_G, N_GENE_DIM)
    G_top_va, V_top_va = precompute_top_genes(va, sid2sample, STORE_TOP_G, N_GENE_DIM) if len(va) else (None, None)

    log("[PROXY] TF proxies ...")
    REG_train_tf = tfproxy_mix(regproxy_aucell_lite_from_top(tr, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM), regproxy_mean_log1p(tr, sid2sample, tf2targets, tf_in_prior))
    REG_test_tf = tfproxy_mix(regproxy_aucell_lite_from_top(te, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM), regproxy_mean_log1p(te, sid2sample, tf2targets, tf_in_prior))
    REG_val_tf = tfproxy_mix(regproxy_aucell_lite_from_top(va, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM), regproxy_mean_log1p(va, sid2sample, tf2targets, tf_in_prior)) if len(va) else None

    anchor_gene_union = set()
    for s in samples:
        anchor_gene_union.update(s.anchor_gene2col.keys())
    gene_pool_full = []
    for gid in genes_in_peakprior:
        if 0 <= int(gid) < len(genes_vocab) and genes_vocab[int(gid)] in anchor_gene_union:
            gene_pool_full.append(int(gid))
    gene_pool_full = np.asarray(gene_pool_full, dtype=np.int64)
    gene_pool_train = build_filtered_gene_pool_from_train(tr, sid2sample, gene_pool_full, genes_vocab)
    selected_source = gene_pool_train if SELECTED_PEAK_ONLY_FROM_FILTERED_GENES else gene_pool_full
    selected_peak_raw_ids = mine_selected_peaks(selected_source, gene2peaks) if ablation_cfg.use_selected_peak_residual else np.zeros((0,), dtype=np.int64)

    train_cluster_summaries = build_cluster_summaries(tr, sid2sample, Z_train, train_cluster_for_corr, genes_vocab, marker_prior)
    train_cluster_summary_vecs = {cid_: cluster_summary_to_vector(v) for cid_, v in train_cluster_summaries.items()}
    cluster_summary_table = ClusterSummaryTable(train_cluster_summary_vecs).to(DEVICE)
    pd.DataFrame([{"cluster_id": int(cid_), "size": int(train_cluster_summaries[cid_]["size"]), "rna_nnz_mean": float(train_cluster_summaries[cid_]["rna_nnz_mean"]), "rna_logmean": float(train_cluster_summaries[cid_]["rna_logmean"])} for cid_ in sorted(train_cluster_summaries.keys())]).to_csv(ds_out / "cluster_summary_stats.csv", index=False)
    export_cluster_top_genes(tr, sid2sample, cl_train, genes_vocab, ds_out / "cluster_top500_genes.csv", ds_out / "cluster_manual_labels.csv", top_n=CLUSTER_TOP_GENE_N)

    n_group = int(max(1, train_group_ids.max() + 1))
    adapter = HyperAdapterHashV2(D, K_RANK, len(tf_in_prior), len(genes_union), peak_hash_size, len(cancer_types), n_group, ablation_cfg=ablation_cfg).to(DEVICE)
    peak_residual = SelectedPeakResidual(selected_peak_raw_ids, d=D).to(DEVICE) if ablation_cfg.use_selected_peak_residual else None
    selector = None
    if ablation_cfg.use_selector:
        summary = compute_dataset_summary(tr, sid2sample, tf2targets, tf_in_prior, N_GENE_DIM)
        selector = PriorSelectorV1(summary_dim=int(summary.size)).to(DEVICE)
        selector.set_summary(summary)
    tf_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=CLUSTER_CORRECTION_HID).to(DEVICE) if ablation_cfg.use_cluster_correction else None
    rna_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=CLUSTER_CORRECTION_HID).to(DEVICE) if ablation_cfg.use_cluster_correction else None
    re_corr_head = ClusterCellCorrectionHead(D, cluster_summary_table.summary_dim, hid=CLUSTER_CORRECTION_HID).to(DEVICE) if ablation_cfg.use_cluster_correction else None

    ctx = RunContext(ablation_cfg, dataset_name, ds_out, model, adapter, selector, peak_residual, tf_corr_head, rna_corr_head, re_corr_head, cluster_summary_table, default_cancer_id, cid, tf_in_prior, genes_union, gene2peaks, genes_vocab)

    params = []
    if ablation_cfg.use_lowrank_delta or ablation_cfg.use_cellgroup_adaptor:
        params += list(adapter.parameters())
    if peak_residual is not None:
        params += list(peak_residual.parameters())
    if selector is not None:
        params += list(selector.parameters())
    for head in [tf_corr_head, rna_corr_head, re_corr_head]:
        if head is not None:
            params += list(head.parameters())
    assert len(params) > 0, f"No trainable parameters for mode={mode}"
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    gid2idx = defaultdict(list)
    for i, g in enumerate(train_group_ids.tolist()):
        gid2idx[int(g)].append(i)
    big_groups = sorted([g for g, v in gid2idx.items() if len(v) >= STRAT_MIN_GROUP_SIZE])
    tf_prior_ids_all = np.arange(len(tf_in_prior), dtype=np.int64)
    gene_pid_all = np.arange(len(genes_union), dtype=np.int64)

    best_auc = -np.inf
    best_ckpt_path = ds_out / "best_ckpt.pt"

    log("[TRAIN] start")
    for step in range(1, STEPS + 1):
        adapter.train()
        for m in [selector, peak_residual, tf_corr_head, rna_corr_head, re_corr_head]:
            if m is not None:
                m.train()
        V_tf, V_gene, V_peak, V_cancer = adapter()
        sel_w = selector() if selector is not None else torch.ones((6,), device=DEVICE)

        idx_np = sample_train_indices(train_group_ids, big_groups, CELL_BATCH)
        Zb = Zt_train[idx_np]
        gb = train_group_ids[idx_np]
        cb = train_cancer_ids[idx_np]
        clb = train_cluster_for_corr[idx_np]
        loss_total = torch.tensor(0.0, device=DEVICE)
        loss_tf = torch.tensor(0.0, device=DEVICE)
        loss_rna = torch.tensor(0.0, device=DEVICE)
        loss_re = torch.tensor(0.0, device=DEVICE)

        # RNA -> TF BCE ranking proxy
        tf_pid_np = np.random.choice(tf_prior_ids_all, size=min(TF_BATCH, tf_prior_ids_all.size), replace=False)
        y_tf = REG_train_tf[idx_np[:, None], tf_pid_np[None, :]]
        y_lab = np.full(y_tf.shape, -1, dtype=np.int8)
        for j in range(y_tf.shape[1]):
            lab, keep = make_pos_neg_labels(y_tf[:, j], POS_Q, NEG_Q)
            y_lab[:, j] = lab
        keep = y_lab >= 0
        if keep.sum() > 0:
            Ztf = adjust_Z(Zb, gb, cb, clb, V_cancer, "tf", ctx)
            TF = score_tf_vec(torch.from_numpy(tf_pid_np).to(DEVICE), V_tf, V_cancer, ctx)
            logits = Ztf @ TF.T
            target = torch.from_numpy(np.maximum(y_lab, 0).astype(np.float32)).to(DEVICE)
            mask = torch.from_numpy(keep.astype(np.float32)).to(DEVICE)
            loss_tf = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / (mask.sum() + 1e-6)
            loss_total = loss_total + W_TF_BASE * sel_w[0] * loss_tf

        # RNA -> RNA retrieval margin
        n_rna_cells = min(idx_np.size, 512)
        sub_local = np.random.choice(np.arange(idx_np.size), size=n_rna_cells, replace=False)
        pos_pairs, neg_pairs, cell_rows = [], [], []
        g2pid = {int(g): p for p, g in enumerate(genes_union)}
        for loc in sub_local.tolist():
            ci = int(idx_np[loc])
            pos_g = G_top_tr[ci]
            pos_g = [g2pid[int(g)] for g in pos_g.tolist() if int(g) in g2pid and int(g) >= 0]
            if len(pos_g) == 0:
                continue
            pos_g = random.sample(pos_g, min(RNA_POS_PER_CELL, len(pos_g)))
            neg_g = np.random.choice(gene_pid_all, size=min(RNA_NEG_PER_CELL, gene_pid_all.size), replace=False).tolist()
            for ppid, npid in zip(pos_g, neg_g[: len(pos_g)]):
                cell_rows.append(loc)
                pos_pairs.append(ppid)
                neg_pairs.append(npid)
        if len(pos_pairs) > 0:
            Zrna = adjust_Z(Zb, gb, cb, clb, V_cancer, "rna", ctx)
            zp = Zrna[torch.tensor(cell_rows, device=DEVICE, dtype=torch.long)]
            Gp = score_gene_vec(torch.tensor(pos_pairs, device=DEVICE, dtype=torch.long), V_gene, V_cancer, ctx)
            Gn = score_gene_vec(torch.tensor(neg_pairs, device=DEVICE, dtype=torch.long), V_gene, V_cancer, ctx)
            sp = (zp * Gp).sum(dim=-1)
            sn = (zp * Gn).sum(dim=-1)
            loss_rna = F.relu(float(MARGIN_TF) - sp + sn).mean()
            loss_total = loss_total + W_RNA_BASE * sel_w[1] * loss_rna

        # RNA -> ATAC geneActivity via gene-linked peak pool
        re_idx_np = np.random.choice(idx_np, size=min(RE_CELL_BATCH, idx_np.size), replace=False)
        re_local_map = {int(cidx): ii for ii, cidx in enumerate(re_idx_np.tolist())}
        re_gene_global = gene_pool_train.copy()
        if re_gene_global.size > RE_GENE_BATCH:
            re_gene_global = np.random.choice(re_gene_global, size=RE_GENE_BATCH, replace=False)
        re_gene_global = [int(g) for g in re_gene_global.tolist() if int(g) in gid2pid_u and len(gene2peaks.get(int(g), [])) > 0]
        if len(re_gene_global) > 0:
            Zre_all = adjust_Z(Zt_train[re_idx_np], train_group_ids[re_idx_np], train_cancer_ids[re_idx_np], train_cluster_for_corr[re_idx_np], V_cancer, "re", ctx)
            Y = build_anchor_targets_for_cells(np.arange(len(re_idx_np)), [tr[i] for i in re_idx_np.tolist()], sid2sample, re_gene_global, genes_vocab)
            losses = []
            for gj, gid in enumerate(re_gene_global):
                y = Y[:, gj]
                lab, keep_np = make_pos_neg_labels(y, POS_Q, NEG_Q)
                if keep_np.sum() < 20 or (lab[keep_np] == 1).sum() < 5 or (lab[keep_np] == 0).sum() < 5:
                    continue
                pid = gid2pid_u[int(gid)]
                pv = gene_peak_vec_for_gene_pid(pid, V_peak, V_cancer, ctx)
                if pv is None:
                    continue
                logits = Zre_all @ pv
                target = torch.from_numpy(np.maximum(lab, 0).astype(np.float32)).to(DEVICE)
                mask = torch.from_numpy(keep_np.astype(np.float32)).to(DEVICE)
                losses.append((F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / (mask.sum() + 1e-6))
            if losses:
                loss_re = torch.stack(losses).mean()
                loss_total = loss_total + W_RE_BASE * sel_w[2] * loss_re

        reg_loss = LAMBDA_L2 * loss_l2_module(adapter)
        if peak_residual is not None:
            reg_loss = reg_loss + SELECTED_PEAK_RESIDUAL_L2 * loss_l2_module(peak_residual)
        if selector is not None:
            reg_loss = reg_loss + SELECTOR_L2 * loss_l2_module(selector)
        for head in [tf_corr_head, rna_corr_head, re_corr_head]:
            if head is not None:
                reg_loss = reg_loss + CLUSTER_CORRECTION_L2 * loss_l2_module(head)
        loss_total = loss_total + reg_loss

        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()

        if step == 1 or step % 50 == 0:
            log(f"[STEP {step:05d}] loss={float(loss_total):.4f} tf={float(loss_tf):.4f} rna={float(loss_rna):.4f} re={float(loss_re):.4f}")

        if len(va) > 0 and (step == 1 or step % VAL_MONITOR_EVERY == 0 or step == STEPS):
            adapter.eval()
            for m in [selector, peak_residual, tf_corr_head, rna_corr_head, re_corr_head]:
                if m is not None:
                    m.eval()
            val_tf = eval_tf_auc(Z_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, REG_val_tf, ctx, max_tf=VAL_MONITOR_TF_SAMPLE)
            val_auc = float(val_tf["auc_mean"])
            log(f"[VAL {step:05d}] tf_auc={val_auc:.5f} n_tf={val_tf['n_tf_eval']} best={best_auc:.5f}")
            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                save_ckpt(best_ckpt_path, ctx, best_auc, step, {"val_tf": val_tf})
                log(f"[SAVE_BEST] {best_ckpt_path}")

    if not best_ckpt_path.exists():
        save_ckpt(best_ckpt_path, ctx, best_auc, STEPS, {})

    # final evaluation uses current model state; best checkpoint already saved for downstream use.
    for m in [adapter, selector, peak_residual, tf_corr_head, rna_corr_head, re_corr_head]:
        if m is not None:
            m.eval()
    val_tf = eval_tf_auc(Z_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, REG_val_tf, ctx, max_tf=QC_TF_SAMPLE) if len(va) else {"auc_mean": np.nan, "n_tf_eval": 0, "cluster_rows": []}
    test_tf = eval_tf_auc(Z_test, test_group_ids, test_cancer_ids, test_cluster_for_corr, REG_test_tf, ctx, max_tf=QC_TF_SAMPLE)
    val_re = eval_re_anchor(Z_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, va, sid2sample, gene_pool_train, ctx, QC_RE_GENE_EVAL) if len(va) else {"spearman_mean": np.nan, "rank_auc_mean": np.nan, "n_genes_eval": 0}
    test_re = eval_re_anchor(Z_test, test_group_ids, test_cancer_ids, test_cluster_for_corr, te, sid2sample, gene_pool_train, ctx, QC_RE_GENE_EVAL)
    val_rna = eval_rna_retrieval_auc(Z_val, val_group_ids, val_cancer_ids, val_cluster_for_corr, G_top_va, V_top_va, ctx, QC_GENE_CELL_EVAL) if len(va) else {"retrieval_auc": np.nan, "n_scored": 0}
    test_rna = eval_rna_retrieval_auc(Z_test, test_group_ids, test_cancer_ids, test_cluster_for_corr, G_top_te, V_top_te, ctx, QC_GENE_CELL_EVAL)

    cluster_eval_rows = []
    for split_name, rows_in in [("val", val_tf["cluster_rows"]), ("test", test_tf["cluster_rows"])] :
        for cid_raw, ncell, auc in rows_in:
            cluster_eval_rows.append({"split": split_name, "cluster_id": int(cid_raw), "cluster_label": cluster_label_map.get(int(cid_raw), "C?"), "n_cells": int(ncell), "tf_auc_mixed": float(auc)})
    pd.DataFrame(cluster_eval_rows).to_csv(ds_out / "cluster_eval.csv", index=False)

    final_row = {
        "ablation_mode": mode,
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
        "use_lowrank_delta": int(ablation_cfg.use_lowrank_delta),
        "use_selector": int(ablation_cfg.use_selector),
        "use_cancer_adaptor": int(ablation_cfg.use_cancer_adaptor),
        "use_cellgroup_adaptor": int(ablation_cfg.use_cellgroup_adaptor),
        "use_group_scale_bias": int(ablation_cfg.use_group_scale_bias),
        "use_selected_peak_residual": int(ablation_cfg.use_selected_peak_residual),
        "use_cluster_correction": int(ablation_cfg.use_cluster_correction),
        "cluster_corr_raw_cluster": int(CLUSTER_CORRECTION_USE_RAW_CLUSTER),
        "n_marker_labels": int(len(marker_prior)),
        "steps": int(STEPS),
    }
    pd.DataFrame([final_row]).to_csv(ds_out / "final_metrics.csv", index=False)
    log(f"[SAVE] {ds_out / 'final_metrics.csv'}")
    return final_row


# ============================================================
# 9) SUMMARY PLOTS
# ============================================================

def plot_summary(summary_csv: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[PLOT][WARN] matplotlib unavailable: {e}")
        return
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    metrics = [
        "best_val_tf_auc_mixed",
        "val_tf_auc_mixed",
        "test_tf_auc_mixed",
        "val_rna_retrieval_auc",
        "test_rna_retrieval_auc",
        "val_re_spearman",
        "test_re_spearman",
        "val_re_rank_auc",
        "test_re_rank_auc",
    ]
    metrics = [m for m in metrics if m in df.columns]
    for m in metrics:
        sub = df[["ablation_mode", m]].copy()
        sub["ablation_mode"] = pd.Categorical(sub["ablation_mode"], categories=ALL_ABLATION_MODES, ordered=True)
        sub = sub.sort_values("ablation_mode")
        plt.figure(figsize=(8.2, 3.8))
        plt.bar(sub["ablation_mode"].astype(str), sub[m].astype(float))
        plt.ylabel(m)
        plt.xticks(rotation=35, ha="right")
        plt.title(f"Stage-B component ablation: {m}")
        plt.tight_layout()
        for ext in ["png", "pdf", "svg"]:
            out = ROOT_RUN_DIR / f"stageB_ablation_{m}.{ext}"
            plt.savefig(out, dpi=300 if ext == "png" else None)
            log(f"[SAVE] {out}")
        plt.close()


# ============================================================
# 10) MAIN
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

    all_rows, failed_rows = [], []
    for mode in MODES_TO_RUN:
        ablation_cfg = make_ablation_config(mode)
        for ds in datasets:
            try:
                row = run_one_dataset_one_mode(ds, Path(BASE_CKPT_MAP[ds]), ablation_cfg)
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(ROOT_RUN_DIR / "stageB_ablation_summary.partial.csv", index=False)
            except Exception as e:
                log(f"[FAILED] mode={mode} dataset={ds}: {repr(e)}")
                failed_rows.append({"ablation_mode": mode, "dataset": ds, "error": repr(e)})
                pd.DataFrame(failed_rows).to_csv(ROOT_RUN_DIR / "stageB_ablation_failed.csv", index=False)
                raise

    summary_path = ROOT_RUN_DIR / "stageB_ablation_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    log(f"[SAVE] {summary_path}")
    plot_summary(summary_path)
    log("[ALL DONE]")


if __name__ == "__main__":
    main()
