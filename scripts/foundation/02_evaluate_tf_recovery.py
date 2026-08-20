# Source: LODO真实建模代码-基准模型及普通迁移后ROC检测-TF侧(2).docx
# Extracted from the user-provided Word code; scientific logic preserved.
from __future__ import annotations

import hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp
import matplotlib.pyplot as plt


# =========================
# Global paths / config
# =========================
BASE = get_project_root()
RUN_DIR = BASE / "runs/qc_lodo_planC_v7_global_hashpeak"

GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"

TF_VOCAB_TXT = MAPS_ROOT / "global_tf_vocab.txt"
GENES_VOCAB_TXT = MAPS_ROOT / "global_gene_vocab.txt"

OUT_ROOT = RUN_DIR / "qc_roc_batch_summary_v3_ckpt_cancer_skipbad"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.RandomState(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Need CUDA"

# ---- match training defaults ----
RNA_TOPK = 2048
EMB_BATCH = 256
AUCELL_TOPK = 500
PEAK_HASH_SIZE_DEFAULT = 2_000_000

# ---- TF monitor ----
VAL_MONITOR_TF_SAMPLE = 128
POS_Q = 0.80
NEG_Q = 0.20

# ---- ROC params ----
FPR_GRID_N = 400
SMOOTH_SIGMA = 1.6

# ---- per-sample ----
MIN_CELLS_PER_SAMPLE = 150
MAX_SAMPLES_TO_PLOT = 10
PLOT_MODE = "overlay"   # "overlay" or "separate"

# ---- label strictness ----
OVERALL_MIN_KEEP = 256
OVERALL_MIN_POS = 64
OVERALL_MIN_NEG = 64

SAMPLE_MIN_KEEP_FRAC = 0.20
SAMPLE_MIN_KEEP_ABS = 80
SAMPLE_MIN_POS = 16
SAMPLE_MIN_NEG = 16

SAVE_PLOTS = True

# 评估时统一用固定 cancer id，不再从外部推断癌种空间
DEFAULT_CANCER_ID = 0

# None 表示全跑
# RUN_ONLY = {"dataset": "BC-Tissue", "model_type": "baseline"}
RUN_ONLY = None


def log(*args):
    print(*args, flush=True)


# =========================
# Helpers
# =========================
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


def topk_from_dense_row(row: np.ndarray, topk: int):
    n = row.size
    k = min(topk, n)
    if k <= 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
    idx = np.argpartition(row, -k)[-k:]
    vals = row[idx]
    order = np.argsort(vals)[::-1]
    return idx[order].astype(np.int64), vals[order].astype(np.float32)


def should_run(dataset_name: str, model_type: str) -> bool:
    if RUN_ONLY is None:
        return True
    return (
        RUN_ONLY.get("dataset") == dataset_name and
        RUN_ONLY.get("model_type") == model_type
    )


def resolve_model_ckpt(dataset_name: str, model_type: str) -> Path:
    ds_dir = RUN_DIR / dataset_name
    assert ds_dir.exists(), f"Missing dataset dir: {ds_dir}"

    if model_type == "baseline":
        direct = ds_dir / "ckpt_last_BASE_global_hashpeak.pt"
        if direct.exists():
            return direct
        cands = sorted(ds_dir.rglob("ckpt_last_BASE_global_hashpeak.pt"))
        if cands:
            return cands[0]
        cands = sorted([p for p in ds_dir.rglob("*.pt") if "base" in p.name.lower()])
        if cands:
            return cands[0]
        raise FileNotFoundError(f"Cannot find baseline ckpt for {dataset_name} under {ds_dir}")

    if model_type == "transfer":
        direct = ds_dir / "ckpt_last_TRANSFER_global_hashpeak.pt"
        if direct.exists():
            return direct
        cands = sorted(ds_dir.rglob("ckpt_last_TRANSFER_global_hashpeak.pt"))
        if cands:
            return cands[0]
        cands = sorted([p for p in ds_dir.rglob("*.pt") if "transfer" in p.name.lower()])
        if cands:
            return cands[0]
        raise FileNotFoundError(f"Cannot find transfer ckpt for {dataset_name} under {ds_dir}")

    raise ValueError(f"Unknown model_type: {model_type}")


def load_tf_gene_prior_full(priors_dir: Path, n_tf_vocab: int, n_gene_dim: int):
    tf_gene_prior = priors_dir / "tf_gene_f.full.npz"
    assert tf_gene_prior.exists(), f"Missing tf_gene_prior: {tf_gene_prior}"

    z = np.load(tf_gene_prior, allow_pickle=True)
    tf_src = z["src"].astype(np.int64)
    g_dst = z["dst"].astype(np.int64)

    tf2targets = defaultdict(list)
    for s, d0 in zip(tf_src.tolist(), g_dst.tolist()):
        if 0 <= s < n_tf_vocab and 0 <= d0 < n_gene_dim:
            tf2targets[int(s)].append(int(d0))

    tf_in_prior = sorted(tf2targets.keys())
    return tf2targets, tf_in_prior


def looks_like_gene_name(x: str) -> bool:
    x = str(x).strip()
    if not x:
        return False
    if x == "0":
        return False
    if x.startswith("0-"):
        return False
    if x.isdigit():
        return False
    return True


def is_gene_name_vector_valid(gene_names: np.ndarray, min_good_frac: float = 0.50):
    g = np.asarray(gene_names).astype(str)
    if g.size == 0:
        return False, {"reason": "empty_gene_names", "good_frac": 0.0, "preview": []}
    good = np.array([looks_like_gene_name(x) for x in g], dtype=bool)
    frac = float(good.mean())
    return frac >= float(min_good_frac), {
        "reason": "ok" if frac >= float(min_good_frac) else "suspicious_gene_names",
        "good_frac": frac,
        "preview": g[:10].tolist(),
        "n_genes": int(g.size),
    }


def get_manifest_rna_dir(dataset_name: str) -> Path:
    return BASE / "LLM-WorldWord" / dataset_name / "processed" / "RNA"


# =========================
# Models
# =========================
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
    def __init__(self, n_gene, n_tf, n_cancer, peak_hash_size=PEAK_HASH_SIZE_DEFAULT, d=256, hid=512, drop=0.1):
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


# =========================
# Data stores / samples
# =========================
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


@dataclass
class Sample:
    sid: str
    cells: np.ndarray
    row_indices: np.ndarray
    proj_rna_path: Path
    proj_rna_store: CSRNPZStore
    counts_path: Path
    counts_shape: tuple[int, int]
    gene_names: np.ndarray
    ext2train: np.ndarray
    train2rna: dict[int, int]
    depth_rna: np.ndarray | None = None


def get_counts_row_dense(sample: Sample, i: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    row = np.asarray(x[i]).reshape(-1)
    return row.astype(np.float32, copy=False)


def check_dataset_rna_quality(dataset_name: str, manifest_rna_dir: Path):
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"
    if not manifest_path.exists():
        return False, f"missing manifest: {manifest_path}"

    meta_files = sorted(manifest_rna_dir.glob("*.rna_meta.npz"))
    if len(meta_files) == 0:
        return False, f"no rna_meta.npz under {manifest_rna_dir}"

    any_good = False
    msgs = []

    for mp in meta_files:
        try:
            z = np.load(mp, allow_pickle=True)
            if "gene_names" not in z:
                msgs.append(f"{mp.name}: missing gene_names")
                continue
            ok, info = is_gene_name_vector_valid(z["gene_names"])
            if ok:
                any_good = True
            else:
                msgs.append(
                    f"{mp.name}: suspicious gene_names good_frac={info['good_frac']:.3f} preview={info['preview'][:5]}"
                )
        except Exception as e:
            msgs.append(f"{mp.name}: {repr(e)}")

    proj_dir = PROJECTED_ROOT / dataset_name
    proj_files = sorted(proj_dir.glob("*.rna.global.csr.npz"))
    if len(proj_files) == 0:
        msgs.append(f"no projected RNA csr under {proj_dir}")

    if not any_good:
        return False, " | ".join(msgs[:3]) if msgs else "no valid rna_meta found"

    return True, "ok"


def load_samples(dataset_name: str, genes_vocab: list[str], manifest_rna_dir: Path) -> tuple[list[Sample], dict[str, Sample], dict[str, int]]:
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

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

        assert meta_path.exists(), f"Missing meta: {meta_path}"
        assert counts_path.exists(), f"Missing counts: {counts_path}"
        assert proj_rna_path.exists(), f"Missing projected RNA: {proj_rna_path}"

        z = np.load(meta_path, allow_pickle=True)
        assert "gene_names" in z, f"{meta_path.name}: missing gene_names"
        gene_names = np.asarray(z["gene_names"]).astype(str)

        ok_gn, info_gn = is_gene_name_vector_valid(gene_names)
        assert ok_gn, f"{meta_path.name}: invalid gene_names good_frac={info_gn['good_frac']:.3f} preview={info_gn['preview'][:5]}"

        cells = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
        row_indices = g["row_index"].astype(np.int64).values
        counts_mm = np.load(counts_path, mmap_mode="r")
        assert counts_mm.ndim == 2, f"{counts_path.name}: expected 2D matrix, got {counts_mm.shape}"
        assert counts_mm.shape[0] == len(row_indices), f"{sid}: counts rows mismatch"
        assert counts_mm.shape[1] == len(gene_names), f"{sid}: counts genes mismatch"

        depth_rna = None
        if "depth_rna" in z:
            depth_rna = np.asarray(z["depth_rna"], dtype=np.float32)

        proj_store = CSRNPZStore(proj_rna_path)
        proj_store.open()
        assert proj_store.shape[0] == len(row_indices), f"{sid}: projected rows mismatch"
        assert proj_store.shape[1] == len(genes_vocab), f"{sid}: projected dim mismatch vs global gene vocab"

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
        assert matched > 1000, f"{sid}: too few genes matched to global vocab: {matched}"

        s = Sample(
            sid=sid,
            cells=cells,
            row_indices=row_indices,
            proj_rna_path=proj_rna_path,
            proj_rna_store=proj_store,
            counts_path=counts_path,
            counts_shape=tuple(counts_mm.shape),
            gene_names=gene_names,
            ext2train=ext2train,
            train2rna=train2rna,
            depth_rna=depth_rna,
        )
        samples.append(s)
        sid2sample[sid] = s

        log(f"[LOAD] {dataset_name}/{sid} | cells={s.counts_shape[0]} genes={s.counts_shape[1]} matched_genes={matched} proj_shape={proj_store.shape}")

    return samples, sid2sample, gene2i_vocab


def collect_all_cells(samples: list[Sample]):
    all_cells = []
    sid_of_cell = []
    for s in samples:
        n_obs = int(s.counts_shape[0])
        for rid in range(n_obs):
            all_cells.append((s.sid, int(rid)))
            sid_of_cell.append(s.sid)
    return all_cells, np.asarray(sid_of_cell, dtype=object)


# =========================
# TF proxy
# =========================
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
            (
                np.asarray(data, np.float32),
                (np.asarray(rows, np.int64), np.asarray(cols, np.int64))
            ),
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

        REG = (X @ W)
        REG = REG.toarray().astype(np.float32) if sp.issparse(REG) else np.asarray(REG, np.float32)

        grow = [gi for gi, _ in items]
        out[np.asarray(grow, np.int64), :] = REG

    return out


def tfproxy_mix(aucell: np.ndarray, meanlog: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(aucell, np.nan, dtype=np.float32)
    for j in range(aucell.shape[1]):
        az = zscore_safe(aucell[:, j])
        mz = zscore_safe(meanlog[:, j])
        out[:, j] = float(alpha) * az + (1.0 - float(alpha)) * mz
    return out


# =========================
# Encode Z / score TF
# =========================
def build_tokens_for_cells(cell_list, sid2sample, cancer_id_int):
    idx_list = []
    val_list = []
    offsets = [0]
    for sid, rid in cell_list:
        s = sid2sample[sid]
        gi, gv = s.proj_rna_store.get_row_topk(rid, RNA_TOPK)
        idx_list.append(gi.astype(np.int64))
        val_list.append(gv.astype(np.float32))
        offsets.append(offsets[-1] + gi.size)
    idx_cat = np.concatenate(idx_list) if idx_list else np.zeros((0,), np.int64)
    val_cat = np.concatenate(val_list) if val_list else np.zeros((0,), np.float32)
    offsets = np.asarray(offsets[:-1], np.int64)
    cancer_ids = np.full((len(cell_list),), int(cancer_id_int), dtype=np.int64)
    return idx_cat, val_cat, offsets, cancer_ids


@torch.no_grad()
def encode_Z(cell_list, sid2sample, model, cancer_id_int):
    Zs = []
    for st in range(0, len(cell_list), EMB_BATCH):
        b = cell_list[st:st + EMB_BATCH]
        idx_cat, val_cat, offsets, cancer_ids = build_tokens_for_cells(b, sid2sample, cancer_id_int)
        z = model.encode_rna(
            torch.from_numpy(idx_cat).to(DEVICE),
            torch.from_numpy(val_cat).to(DEVICE),
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )
        Zs.append(z.detach().cpu().numpy())
    return np.concatenate(Zs, axis=0).astype(np.float32)


@torch.no_grad()
def score_tf_vectors(model, tf_ids: np.ndarray, cancer_id_int: int):
    tf_ids_t = torch.from_numpy(tf_ids.astype(np.int64)).to(DEVICE)
    c = torch.full((tf_ids_t.numel(),), int(cancer_id_int), device=DEVICE, dtype=torch.long)
    TF = model.tf_in(tf_ids_t) + model.tf_in_c(c)
    TF = F.normalize(TF, dim=-1)
    return TF


# =========================
# ROC aggregation
# =========================
def compute_mean_roc_across_tfs(pred, reg, tf_ids,
                                pos_q, neg_q,
                                min_keep, min_pos, min_neg):

    grid = np.linspace(0, 1, FPR_GRID_N, dtype=np.float64)
    aucs = []
    tpr_stack = []
    used_tf = 0

    for j, tfp in enumerate(tf_ids.tolist()):
        regj = reg[:, int(tfp)]
        y, keep = make_pos_neg_labels(regj, pos_q, neg_q)

        if keep.sum() < min_keep or (y == 1).sum() < min_pos or (y == 0).sum() < min_neg:
            continue

        scores = pred[:, j]
        auc0 = auc_from_scores(scores[y == 1], scores[y == 0])
        if not np.isfinite(auc0):
            continue

        fpr, tpr = roc_curve_from_scores(y, scores)
        tpr_stack.append(interp_to_grid(fpr, tpr, grid))
        aucs.append(auc0)
        used_tf += 1

    if used_tf < 1:
        return None

    mean_auc = float(np.mean(np.asarray(aucs, np.float32)))
    tpr_mean = np.mean(np.stack(tpr_stack, 0), axis=0)
    tpr_mean = smooth_1d(tpr_mean, sigma=SMOOTH_SIGMA)
    tpr_mean = fix_roc_endpoints_and_monotonic(tpr_mean)
    return grid, tpr_mean, mean_auc, used_tf, np.asarray(aucs, np.float32)


# =========================
# Core: evaluate one full model ckpt
# =========================
def evaluate_one_model(
    dataset_name: str,
    model_type: str,
    ckpt_path: Path,
    manifest_rna_dir: Path,
    out_dir: Path,
    genes_vocab: list[str],
    tfs_vocab: list[str],
):
    assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"
    assert PRIORS_DIR.exists(), f"Missing PRIORS_DIR: {PRIORS_DIR}"
    assert manifest_rna_dir.exists(), f"Missing manifest_rna_dir: {manifest_rna_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log(f"[START] dataset={dataset_name} | model_type={model_type}")
    log(f"[CKPT] {ckpt_path}")
    log(f"[RNA_DIR] {manifest_rna_dir}")
    log(f"[PROJECTED_DIR] {PROJECTED_ROOT / dataset_name}")
    log(f"[PRIORS_DIR] {PRIORS_DIR}")
    log(f"[OUT_DIR] {out_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_sd = ckpt["model"]
    meta = ckpt.get("meta", {})

    N_GENE_DIM = int(model_sd["gene_in.weight"].shape[0])
    N_TF_DIM = int(model_sd["tf_in.weight"].shape[0])
    D = int(model_sd["gene_in.weight"].shape[1])
    HID = int(model_sd["rna_proj.0.weight"].shape[0])
    N_CANCER = int(model_sd["cell_c.weight"].shape[0])
    peak_hash_size = int(meta.get("PEAK_HASH_SIZE", PEAK_HASH_SIZE_DEFAULT))

    assert N_GENE_DIM == len(genes_vocab), f"N_GENE_DIM mismatch: ckpt={N_GENE_DIM}, vocab={len(genes_vocab)}"
    assert N_TF_DIM == len(tfs_vocab), f"N_TF_DIM mismatch: ckpt={N_TF_DIM}, vocab={len(tfs_vocab)}"
    assert 0 <= DEFAULT_CANCER_ID < N_CANCER, f"DEFAULT_CANCER_ID={DEFAULT_CANCER_ID} out of range for ckpt N_CANCER={N_CANCER}"

    log(f"[CKPT_META] N_GENE_DIM={N_GENE_DIM} N_TF_DIM={N_TF_DIM} N_CANCER={N_CANCER} D={D} HID={HID} peak_hash_size={peak_hash_size}")
    log(f"[EVAL_CFG] using fixed cancer_id={DEFAULT_CANCER_ID}")

    model = PlanC_DirAware_Cancer_HashPeak(
        n_gene=N_GENE_DIM,
        n_tf=N_TF_DIM,
        n_cancer=N_CANCER,
        peak_hash_size=peak_hash_size,
        d=D,
        hid=HID,
        drop=0.1,
    )
    model.load_state_dict(model_sd, strict=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    samples, sid2sample, _ = load_samples(dataset_name, genes_vocab, manifest_rna_dir)
    all_cells, groups_sid = collect_all_cells(samples)
    assert len(all_cells) > 0, f"No cells found in {manifest_rna_dir}"
    log("[CELLS] total =", len(all_cells), "| samples =", len(samples))

    tf2targets, tf_in_prior = load_tf_gene_prior_full(
        PRIORS_DIR,
        n_tf_vocab=len(tfs_vocab),
        n_gene_dim=N_GENE_DIM,
    )
    log("[PRIOR] tf_in_prior =", len(tf_in_prior))

    log("[ENC] encoding all RNA cells from projected CSR ...")
    Z = encode_Z(all_cells, sid2sample, model, DEFAULT_CANCER_ID)
    Zt = torch.from_numpy(Z).to(DEVICE)
    log("[ENC] Z shape =", Z.shape)

    log("[PROXY] TF mixed proxy from dense RNA counts ...")
    REG_au = regproxy_aucell_lite_from_top(
        all_cells, sid2sample, tf2targets, tf_in_prior, AUCELL_TOPK, N_GENE_DIM
    )
    REG_mn = regproxy_mean_log1p(all_cells, sid2sample, tf2targets, tf_in_prior)

    tf_mix_alpha = 0.6
    REG_tf = tfproxy_mix(REG_au, REG_mn, tf_mix_alpha)

    tf_ids_all = np.arange(len(tf_in_prior), dtype=np.int64)
    h = int(hashlib.md5(f"{dataset_name}::{model_type}".encode("utf-8")).hexdigest()[:8], 16)
    rng_local = np.random.RandomState(SEED ^ h)
    if tf_ids_all.size > VAL_MONITOR_TF_SAMPLE:
        tf_ids = rng_local.choice(tf_ids_all, size=VAL_MONITOR_TF_SAMPLE, replace=False)
        tf_ids = np.sort(tf_ids)
    else:
        tf_ids = tf_ids_all.copy()

    with torch.no_grad():
        tf_vocab_ids = np.asarray([tf_in_prior[i] for i in tf_ids.tolist()], dtype=np.int64)
        TF_vec = score_tf_vectors(model, tf_vocab_ids, DEFAULT_CANCER_ID)
        pred = (F.normalize(Zt, dim=-1) @ TF_vec.t()).detach().cpu().numpy().astype(np.float32)

    res = compute_mean_roc_across_tfs(
        pred, REG_tf, tf_ids,
        POS_Q, NEG_Q,
        min_keep=OVERALL_MIN_KEEP, min_pos=OVERALL_MIN_POS, min_neg=OVERALL_MIN_NEG,
    )
    assert res is not None, f"No TF passed overall filters: {dataset_name}-{model_type}"
    grid, tpr_mean, auc_mean, used_tf, per_tf_aucs = res
    log(f"[OVERALL] mean AUC={auc_mean:.4f}, tf_used={used_tf}")

    overall_png = None
    if SAVE_PLOTS:
        fig = plt.figure(figsize=(6.2, 6.2), dpi=220)
        ax = fig.add_subplot(111)
        ax.plot(grid, tpr_mean, linewidth=2.6, label=f"{model_type} AUC={auc_mean:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.6, label="Random")
        ax.set_title(f"ROC (RNA→TF): {dataset_name} | {model_type}", fontsize=13)
        ax.set_xlabel("False positive rate", fontsize=11)
        ax.set_ylabel("True positive rate", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.tight_layout()
        overall_png = out_dir / f"{dataset_name}_{model_type}_roc_overall.png"
        fig.savefig(overall_png, bbox_inches="tight")
        plt.close(fig)
        log("[SAVE]", overall_png)

    per_tf_rows = []
    for j, tfp in enumerate(tf_ids.tolist()):
        regj = REG_tf[:, int(tfp)]
        y, keep = make_pos_neg_labels(regj, POS_Q, NEG_Q)
        if keep.sum() < OVERALL_MIN_KEEP or (y == 1).sum() < OVERALL_MIN_POS or (y == 0).sum() < OVERALL_MIN_NEG:
            continue
        scores = pred[:, j]
        auc0 = auc_from_scores(scores[y == 1], scores[y == 0])
        per_tf_rows.append({
            "tf_pid_in_prior": int(tfp),
            "tf_vocab_id": int(tf_in_prior[int(tfp)]),
            "auc": float(auc0),
        })

    tf_auc_csv = out_dir / f"{dataset_name}_{model_type}_tf_aucs.csv"
    df_tf = pd.DataFrame(per_tf_rows)
    if len(df_tf) > 0:
        df_tf = df_tf.sort_values("auc", ascending=False).reset_index(drop=True)
    df_tf.to_csv(tf_auc_csv, index=False)
    log("[SAVE]", tf_auc_csv)

    sid_unique, sid_counts = np.unique(groups_sid, return_counts=True)
    order = np.argsort(-sid_counts)
    sid_unique = sid_unique[order]
    sid_counts = sid_counts[order]

    rows_summary = []
    curves = []

    for sid, ncell in zip(sid_unique.tolist(), sid_counts.tolist()):
        if ncell < MIN_CELLS_PER_SAMPLE:
            continue

        m = (groups_sid == sid)
        min_keep = max(SAMPLE_MIN_KEEP_ABS, int(SAMPLE_MIN_KEEP_FRAC * ncell))

        res_sid = compute_mean_roc_across_tfs(
            pred[m], REG_tf[m], tf_ids,
            POS_Q, NEG_Q,
            min_keep=min_keep, min_pos=SAMPLE_MIN_POS, min_neg=SAMPLE_MIN_NEG,
        )

        if res_sid is None:
            continue

        g, tpr_m, auc_m, used, _ = res_sid
        rows_summary.append({
            "sample_id": str(sid),
            "n_cells": int(ncell),
            "mean_auc": float(auc_m),
            "tf_used": int(used),
        })
        curves.append((str(sid), int(ncell), float(auc_m), g, tpr_m))

        if SAVE_PLOTS and PLOT_MODE == "separate":
            fig = plt.figure(figsize=(6.2, 6.2), dpi=220)
            ax = fig.add_subplot(111)
            ax.plot(g, tpr_m, linewidth=2.6, label=f"{sid} (n={int(ncell)}) AUC={auc_m:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.6, label="Random")
            ax.set_title(f"ROC (RNA→TF, sample {sid})", fontsize=13)
            ax.set_xlabel("False positive rate", fontsize=11)
            ax.set_ylabel("True positive rate", fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="lower right", frameon=False, fontsize=9)
            fig.tight_layout()
            p2 = out_dir / f"{dataset_name}_{model_type}_roc_sample_{sid}.png"
            fig.savefig(p2, bbox_inches="tight")
            plt.close(fig)

    sample_summary_csv = out_dir / f"{dataset_name}_{model_type}_sample_summary.csv"
    df_sum = pd.DataFrame(rows_summary)
    if len(df_sum) > 0:
        df_sum = df_sum.sort_values("n_cells", ascending=False).reset_index(drop=True)
    df_sum.to_csv(sample_summary_csv, index=False)
    log("[SAVE]", sample_summary_csv)

    sample_overlay_png = None
    if SAVE_PLOTS and PLOT_MODE == "overlay" and len(curves) > 0:
        curves = sorted(curves, key=lambda x: x[1], reverse=True)[:MAX_SAMPLES_TO_PLOT]
        fig = plt.figure(figsize=(6.2, 6.2), dpi=220)
        ax = fig.add_subplot(111)
        for sid, ncell, auc_m, g, tpr_m in curves:
            ax.plot(g, tpr_m, linewidth=2.0, label=f"{sid} (n={ncell}) AUC={auc_m:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.6, label="Random")
        ax.set_title(f"ROC (RNA→TF, per sample): {dataset_name} | {model_type}", fontsize=13)
        ax.set_xlabel("False positive rate", fontsize=11)
        ax.set_ylabel("True positive rate", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        fig.tight_layout()
        sample_overlay_png = out_dir / f"{dataset_name}_{model_type}_roc_samples_overlay.png"
        fig.savefig(sample_overlay_png, bbox_inches="tight")
        plt.close(fig)
        log("[SAVE]", sample_overlay_png)

    result = {
        "dataset": dataset_name,
        "model_type": model_type,
        "ckpt_path": str(ckpt_path),
        "manifest_rna_dir": str(manifest_rna_dir),
        "projected_rna_dir": str(PROJECTED_ROOT / dataset_name),
        "n_cells": int(len(all_cells)),
        "n_samples": int(len(samples)),
        "n_cancer_ckpt": int(N_CANCER),
        "eval_cancer_id": int(DEFAULT_CANCER_ID),
        "overall_auc": float(auc_mean),
        "tf_used": int(used_tf),
        "overall_png": str(overall_png) if overall_png is not None else "",
        "tf_auc_csv": str(tf_auc_csv),
        "sample_summary_csv": str(sample_summary_csv),
        "sample_overlay_png": str(sample_overlay_png) if sample_overlay_png is not None else "",
    }

    log(f"[DONE] dataset={dataset_name} | model_type={model_type} | overall_auc={auc_mean:.4f}")
    return result


# =========================
# Summary helpers
# =========================
def build_dataset_summary(model_level_df: pd.DataFrame) -> pd.DataFrame:
    if len(model_level_df) == 0:
        return pd.DataFrame(columns=["dataset", "baseline_roc", "transfer_roc", "delta"])

    x = model_level_df.copy()
    x["roc"] = x["overall_auc"]

    baseline_df = (
        x[x["model_type"] == "baseline"][["dataset", "roc"]]
        .rename(columns={"roc": "baseline_roc"})
        .drop_duplicates(subset=["dataset"], keep="last")
    )
    transfer_df = (
        x[x["model_type"] == "transfer"][["dataset", "roc"]]
        .rename(columns={"roc": "transfer_roc"})
        .drop_duplicates(subset=["dataset"], keep="last")
    )

    merged = pd.merge(baseline_df, transfer_df, on="dataset", how="outer")
    merged["delta"] = merged["transfer_roc"] - merged["baseline_roc"]
    merged = merged.sort_values("dataset").reset_index(drop=True)
    return merged


# =========================
# Main
# =========================
def main():
    assert BASE.exists(), f"Missing BASE: {BASE}"
    assert RUN_DIR.exists(), f"Missing RUN_DIR: {RUN_DIR}"
    assert GLOBAL_ROOT.exists(), f"Missing GLOBAL_ROOT: {GLOBAL_ROOT}"
    assert PRIORS_DIR.exists(), f"Missing PRIORS_DIR: {PRIORS_DIR}"
    assert MAPS_ROOT.exists(), f"Missing MAPS_ROOT: {MAPS_ROOT}"
    assert PROJECTED_ROOT.exists(), f"Missing PROJECTED_ROOT: {PROJECTED_ROOT}"
    assert GENES_VOCAB_TXT.exists(), f"Missing GENES_VOCAB_TXT: {GENES_VOCAB_TXT}"
    assert TF_VOCAB_TXT.exists(), f"Missing TF_VOCAB_TXT: {TF_VOCAB_TXT}"

    genes_vocab = read_vocab_txt(GENES_VOCAB_TXT)
    tfs_vocab = read_vocab_txt(TF_VOCAB_TXT)

    dataset_names = sorted([p.name for p in RUN_DIR.iterdir() if p.is_dir()])
    assert len(dataset_names) > 0, f"No dataset dirs under {RUN_DIR}"

    log("[RUN_DATASETS]", dataset_names)

    all_results = []
    failed = []
    skipped_quality = []

    usable_datasets = []
    for dataset_name in dataset_names:
        manifest_rna_dir = get_manifest_rna_dir(dataset_name)
        ok, msg = check_dataset_rna_quality(dataset_name, manifest_rna_dir)
        if not ok:
            log(f"[SKIP][RNA_QUALITY] {dataset_name} -> {msg}")
            skipped_quality.append({
                "dataset": dataset_name,
                "reason": msg,
            })
            continue
        usable_datasets.append(dataset_name)

    log("[DATASETS] usable =", usable_datasets)
    log("[DATASETS] skipped_quality =", [x["dataset"] for x in skipped_quality])

    if len(skipped_quality) > 0:
        skipped_csv = OUT_ROOT / "skipped_low_quality_datasets.csv"
        pd.DataFrame(skipped_quality).to_csv(skipped_csv, index=False)
        log("[SAVE]", skipped_csv)

    for dataset_name in usable_datasets:
        manifest_rna_dir = get_manifest_rna_dir(dataset_name)

        try:
            baseline_ckpt = resolve_model_ckpt(dataset_name, "baseline")
            log(f"[FOUND][{dataset_name}][baseline] {baseline_ckpt}")
        except Exception as e:
            failed.append({
                "dataset": dataset_name,
                "model_type": "baseline",
                "ckpt_path": "",
                "error": repr(e),
            })
            baseline_ckpt = None
            log(f"[ERROR][find baseline] dataset={dataset_name}, err={repr(e)}")

        try:
            transfer_ckpt = resolve_model_ckpt(dataset_name, "transfer")
            log(f"[FOUND][{dataset_name}][transfer] {transfer_ckpt}")
        except Exception as e:
            failed.append({
                "dataset": dataset_name,
                "model_type": "transfer",
                "ckpt_path": "",
                "error": repr(e),
            })
            transfer_ckpt = None
            log(f"[ERROR][find transfer] dataset={dataset_name}, err={repr(e)}")

        for model_type, ckpt in [("baseline", baseline_ckpt), ("transfer", transfer_ckpt)]:
            if ckpt is None:
                continue
            if not should_run(dataset_name, model_type):
                continue

            out_dir = OUT_ROOT / dataset_name / model_type
            try:
                res = evaluate_one_model(
                    dataset_name=dataset_name,
                    model_type=model_type,
                    ckpt_path=ckpt,
                    manifest_rna_dir=manifest_rna_dir,
                    out_dir=out_dir,
                    genes_vocab=genes_vocab,
                    tfs_vocab=tfs_vocab,
                )
                all_results.append(res)
            except Exception as e:
                log(f"[ERROR] dataset={dataset_name}, model_type={model_type}, err={repr(e)}")
                failed.append({
                    "dataset": dataset_name,
                    "model_type": model_type,
                    "ckpt_path": str(ckpt),
                    "error": repr(e),
                })

            df_model = pd.DataFrame(all_results)
            model_csv = OUT_ROOT / "model_level_results.csv"
            df_model.to_csv(model_csv, index=False)

            df_summary = build_dataset_summary(df_model)
            summary_csv = OUT_ROOT / "dataset_roc_summary.csv"
            df_summary.to_csv(summary_csv, index=False)

            if len(failed) > 0:
                pd.DataFrame(failed).to_csv(OUT_ROOT / "failed_models.csv", index=False)

            log("[REFRESH SAVE]", model_csv)
            log("[REFRESH SAVE]", summary_csv)

    df_model = pd.DataFrame(all_results)
    model_csv = OUT_ROOT / "model_level_results.csv"
    df_model.to_csv(model_csv, index=False)

    df_summary = build_dataset_summary(df_model)
    summary_csv = OUT_ROOT / "dataset_roc_summary.csv"
    df_summary.to_csv(summary_csv, index=False)

    if len(failed) > 0:
        failed_csv = OUT_ROOT / "failed_models.csv"
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        log("[SAVE]", failed_csv)

    log("=" * 100)
    log("[FINAL SAVE]", model_csv)
    log("[FINAL SAVE]", summary_csv)
    log("[ALL DONE]")


if __name__ == "__main__":
    main()
