from __future__ import annotations

# ============================================================
# eval_midas_like_baseline_sample9_harder_disjoint.py
# ============================================================
# 只做评估，不重新训练
#
# 只评估单样本：
#   CRC_RNA_TO009
#
# 模型：
#   MIDAS-like baseline standalone
#
# 评估内容：
#   1) RNA AUROC
#   2) TF ROC/AUC（更难版）
#
# 更难版 TF 评估逻辑：
#   - TF->target genes 从 prior cache 读取
#   - 每个 TF 的 target genes 先截断，再做 disjoint split：
#       * targets_label: 只用于 true label
#       * targets_pred : 只用于 pred score
#   - true proxy: 用真实 RNA 的 AUCELL-lite（仅 targets_label）
#   - pred proxy: 用重建 RNA_HAT 的 mixed proxy（仅 targets_pred）
#   - 单样本内细胞做 2-fold cross-cell split
#       * 用 fold A 估计 quantile 阈值
#       * 在 fold B 上打标签并评估
#       * 再交换 A/B
#   - 对每个 TF 聚合两个 fold 的 ROC/AUC
#
# 适配说明：
#   - 输入是 RNA + ATAC_geneActivity_ALLGENES
#   - 预处理方式严格读取 checkpoint 中保存的：
#       * feature_selection
#       * normalization
#   - 仅加载 checkpoint 做评估，不训练
# ============================================================

import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


# ============================================================
# 0) CONFIG
# ============================================================
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"

DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

RNA_DIR = WORLD_ROOT / DATASET_NAME / "processed" / "RNA"
ANCHOR_DIR = WORLD_ROOT / DATASET_NAME / "processed" / "ATAC_geneActivity_ALLGENES"
RNA_MANIFEST = RNA_DIR / "manifest_rna.tsv"

RUN_DIR = BASE / "runs" / "midas_like_baseline_sample9_standalone" / DATASET_NAME / SAMPLE_ID
MODEL_PATH = RUN_DIR / "model.pt"

OUT_DIR = RUN_DIR / f"eval_tfproxy_{SAMPLE_ID}_harder_disjoint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 512

# -------- RNA AUROC --------
RNA_POS_THRESHOLD = 0.0

# -------- TF prior resources --------
GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"

# -------- harder TF settings --------
TF_MIN_TARGETS_EVAL = 16         # before split
TF_MAX_TFS = None                # None = all
TF_MAX_TARGETS_PER_TF = 40       # truncate before disjoint split
TF_MIN_EACH_SIDE = 8             # each side after disjoint split should have >=8

AUCELL_TOPK = 50
TF_PROXY_MIX_ALPHA = 0.6

POS_Q = 0.95
NEG_Q = 0.05

OVERALL_MIN_KEEP = 32
OVERALL_MIN_POS = 8
OVERALL_MIN_NEG = 8

FPR_GRID_N = 400
N_FOLDS = 2


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


def norm_barcode(x: str) -> str:
    x = str(x)
    return x.split("::", 1)[-1] if "::" in x else x


def make_cell_key(sample_id: str, barcode: str) -> str:
    return f"{sample_id}::{barcode}"


def canonical_sid(x: str) -> str:
    s = str(x).strip()
    s = s.replace("_RNA_", "___")
    s = s.replace("_ATAC_", "___")
    return s


def build_sid_match_map(rna_sids: list[str], atac_sids: list[str]):
    atac_by_canon = {}
    for s in atac_sids:
        atac_by_canon.setdefault(canonical_sid(s), []).append(s)

    matched_pairs = []
    rna2atac = {}
    atac2rna = {}

    for rs in sorted(rna_sids):
        c = canonical_sid(rs)
        hits = atac_by_canon.get(c, [])

        if rs in hits:
            asid = rs
        elif len(hits) == 1:
            asid = hits[0]
        elif len(hits) > 1:
            atac_hits = [x for x in hits if "_ATAC_" in x]
            if len(atac_hits) == 1:
                asid = atac_hits[0]
            else:
                asid = sorted(hits)[0]
        else:
            asid = None

        if asid is not None:
            matched_pairs.append((rs, asid))
            rna2atac[rs] = asid
            atac2rna[asid] = rs

    return matched_pairs, rna2atac, atac2rna


def pick_first_key(z, candidates: list[str], context: str):
    for k in candidates:
        if k in z:
            return k
    raise KeyError(f"{context}: none of keys found: {candidates}")


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

    raise FileNotFoundError(
        f"Missing anchor for sid={sid}. Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def load_anchor_all(anchor_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(str(anchor_file), allow_pickle=True)

    if "genes" in z:
        genes = np.asarray(z["genes"]).astype(str)
    elif "var_names" in z:
        genes = np.asarray(z["var_names"]).astype(str)
    else:
        raise KeyError(f"{anchor_file.name}: cannot find genes key")

    if "cells" in z:
        cells_raw = np.asarray(z["cells"]).astype(str)
    elif "barcodes" in z:
        cells_raw = np.asarray(z["barcodes"]).astype(str)
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

    X = np.asarray(X, dtype=np.float32)
    return cells, genes, X


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s < 1e-6:
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
    if y.size < 2:
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


def find_tf_gene_prior_file(prior_dir: Path):
    p1 = prior_dir / "tf_gene_f.cap256.npz"
    p2 = prior_dir / "tf_gene_f.full.npz"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def hash_fold(cell_key: str, n_folds: int = 2) -> int:
    h = hashlib.md5(cell_key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(n_folds)


def threshold_labels_from_reference(
    scores_ref: np.ndarray,
    scores_eval: np.ndarray,
    pos_q: float,
    neg_q: float,
):
    scores_ref = np.asarray(scores_ref, dtype=np.float32)
    scores_eval = np.asarray(scores_eval, dtype=np.float32)

    finite_ref = np.isfinite(scores_ref)
    ref = scores_ref[finite_ref]
    if ref.size < 10:
        y = np.full(scores_eval.shape, -1, dtype=np.int8)
        keep = np.zeros(scores_eval.shape, dtype=bool)
        return y, keep, np.nan, np.nan

    hi = np.quantile(ref, pos_q)
    lo = np.quantile(ref, neg_q)

    y = np.full(scores_eval.shape, -1, dtype=np.int8)
    finite_eval = np.isfinite(scores_eval)
    y[(scores_eval >= hi) & finite_eval] = 1
    y[(scores_eval <= lo) & finite_eval] = 0
    keep = (y >= 0)
    return y, keep, float(lo), float(hi)


def split_tf_targets_disjoint(
    tf2targets: list[np.ndarray],
    seed: int = 42,
    min_each_side: int = 4,
):
    rng = np.random.default_rng(seed)
    tf2targets_label = []
    tf2targets_pred = []

    for arr in tf2targets:
        arr = np.asarray(arr, dtype=np.int64)
        arr = np.unique(arr)

        if arr.size < 2 * int(min_each_side):
            tf2targets_label.append(np.zeros((0,), dtype=np.int64))
            tf2targets_pred.append(np.zeros((0,), dtype=np.int64))
            continue

        perm = rng.permutation(arr.size)
        arrp = arr[perm]

        mid = arr.size // 2
        left = np.sort(arrp[:mid])
        right = np.sort(arrp[mid:])

        if left.size < int(min_each_side) or right.size < int(min_each_side):
            tf2targets_label.append(np.zeros((0,), dtype=np.int64))
            tf2targets_pred.append(np.zeros((0,), dtype=np.int64))
            continue

        tf2targets_label.append(left.astype(np.int64))
        tf2targets_pred.append(right.astype(np.int64))

    return tf2targets_label, tf2targets_pred


# ============================================================
# 2) DATA OBJECTS
# ============================================================
@dataclass
class SampleData:
    sid: str
    barcodes: np.ndarray
    X_rna: np.ndarray
    X_atac: np.ndarray
    gene_names_rna: np.ndarray
    gene_names_atac: np.ndarray


# ============================================================
# 3) LOAD SINGLE SAMPLE PAIRED DATA
# ============================================================
def load_single_sample_paired_data(sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    assert RNA_MANIFEST.exists(), f"Missing RNA manifest: {RNA_MANIFEST}"
    df = pd.read_csv(RNA_MANIFEST, sep="\t")
    assert "sample_id" in df.columns and "cell_id" in df.columns and "row_index" in df.columns, "RNA manifest columns missing"

    df = df[df["sample_id"].astype(str) == sample_id].copy()
    assert len(df) > 0, f"No RNA rows for {sample_id}"
    df = df.sort_values("row_index").reset_index(drop=True)

    sid = sample_id
    meta_path = RNA_DIR / f"{sid}.rna_meta.npz"
    counts_path = RNA_DIR / f"{sid}.X_rna_counts_dense.npy"
    anchor_path = resolve_anchor_path(ANCHOR_DIR, sid)

    assert meta_path.exists(), f"Missing RNA meta: {meta_path}"
    assert counts_path.exists(), f"Missing RNA counts: {counts_path}"
    assert anchor_path.exists(), f"Missing anchor: {anchor_path}"

    meta = np.load(meta_path, allow_pickle=True)
    gene_key = pick_first_key(meta, ["gene_names", "genes", "var_names"], f"RNA meta {meta_path.name}")
    gene_names_rna = np.asarray(meta[gene_key]).astype(str)

    X_rna_all = np.asarray(np.load(counts_path, mmap_mode="r"), dtype=np.float32)
    assert X_rna_all.ndim == 2, f"RNA counts must be 2D: {counts_path}"
    assert X_rna_all.shape[0] == len(df), f"RNA row mismatch for {sample_id}"

    rna_cells = np.asarray([norm_barcode(x) for x in df["cell_id"].astype(str).tolist()], dtype=object)

    anchor_cells, gene_names_atac, X_atac_all = load_anchor_all(anchor_path)
    assert X_atac_all.ndim == 2, f"ATAC geneActivity must be 2D: {anchor_path}"

    anchor_map = {str(c): i for i, c in enumerate(anchor_cells.tolist())}
    matched_rna_idx = []
    matched_atac_idx = []

    for i, bc in enumerate(rna_cells.tolist()):
        j = anchor_map.get(str(bc), None)
        if j is not None:
            matched_rna_idx.append(i)
            matched_atac_idx.append(j)

    assert len(matched_rna_idx) > 0, f"No matched cells for {sample_id}"

    matched_rna_idx = np.asarray(matched_rna_idx, dtype=np.int64)
    matched_atac_idx = np.asarray(matched_atac_idx, dtype=np.int64)

    X_rna = X_rna_all[matched_rna_idx].astype(np.float32, copy=False)
    X_atac = X_atac_all[matched_atac_idx].astype(np.float32, copy=False)
    barcodes = rna_cells[matched_rna_idx]

    obs_rows = []
    for bc in barcodes.tolist():
        obs_rows.append({
            "sample_id": sid,
            "barcode": str(bc),
            "cell_key": make_cell_key(sid, str(bc)),
            "split": "all",
        })
    obs = pd.DataFrame(obs_rows)

    log(f"[LOAD] sid={sid} paired_cells={len(barcodes)} RNA={X_rna.shape} ATAC(geneActivity)={X_atac.shape}")
    return X_rna, X_atac, gene_names_rna.astype(str), gene_names_atac.astype(str), obs


# ============================================================
# 4) PREPROCESS USING SAVED FEATURE SELECTION + NORMALIZATION
# ============================================================
def preprocess_with_checkpoint(
    X_rna_raw: np.ndarray,
    X_atac_raw: np.ndarray,
    gene_names_rna: np.ndarray,
    gene_names_atac: np.ndarray,
    ckpt: dict,
):
    feat = ckpt["feature_selection"]
    norm = ckpt["normalization"]

    keep_rna = np.asarray(feat["keep_rna_idx"], dtype=np.int64)
    keep_atac = np.asarray(feat["keep_atac_idx"], dtype=np.int64)

    saved_rna_names = np.asarray(feat["rna_feature_names"]).astype(str)
    saved_atac_names = np.asarray(feat["atac_feature_names"]).astype(str)

    cur_rna_names = gene_names_rna[keep_rna].astype(str)
    cur_atac_names = gene_names_atac[keep_atac].astype(str)

    assert np.array_equal(cur_rna_names, saved_rna_names), "RNA feature names mismatch with checkpoint"
    assert np.array_equal(cur_atac_names, saved_atac_names), "ATAC feature names mismatch with checkpoint"

    X_rna = np.log1p(np.clip(X_rna_raw[:, keep_rna], 0.0, None)).astype(np.float32)
    X_atac = np.log1p(np.clip(X_atac_raw[:, keep_atac], 0.0, None)).astype(np.float32)

    rna_mu = np.asarray(norm["rna_mu"], dtype=np.float32)
    rna_sd = np.asarray(norm["rna_sd"], dtype=np.float32)
    atac_mu = np.asarray(norm["atac_mu"], dtype=np.float32)
    atac_sd = np.asarray(norm["atac_sd"], dtype=np.float32)

    assert rna_mu.shape[0] == X_rna.shape[1], "rna_mu dim mismatch"
    assert rna_sd.shape[0] == X_rna.shape[1], "rna_sd dim mismatch"
    assert atac_mu.shape[0] == X_atac.shape[1], "atac_mu dim mismatch"
    assert atac_sd.shape[0] == X_atac.shape[1], "atac_sd dim mismatch"

    rna_sd = np.where(rna_sd < 1e-6, 1.0, rna_sd).astype(np.float32)
    atac_sd = np.where(atac_sd < 1e-6, 1.0, atac_sd).astype(np.float32)

    X_rna = ((X_rna - rna_mu[None, :]) / rna_sd[None, :]).astype(np.float32, copy=False)
    X_atac = ((X_atac - atac_mu[None, :]) / atac_sd[None, :]).astype(np.float32, copy=False)

    log(f"[FILTER][LOAD] RNA selected = {len(cur_rna_names)}")
    log(f"[FILTER][LOAD] ATAC selected = {len(cur_atac_names)}")

    return X_rna, X_atac, cur_rna_names, cur_atac_names, keep_rna, keep_atac


# ============================================================
# 5) TF TARGETS FROM PRIOR
# ============================================================
def read_tf_prior(prior_dir: Path, tfs_vocab: list[str], n_gene_dim: int):
    tf_gene_prior = find_tf_gene_prior_file(prior_dir)
    assert tf_gene_prior is not None, f"Cannot find tf-gene prior in {prior_dir}"

    z = np.load(tf_gene_prior, allow_pickle=True)
    assert "src" in z and "dst" in z, f"Invalid tf-gene prior format: {tf_gene_prior}"

    tf_src = z["src"].astype(np.int64)
    g_dst = z["dst"].astype(np.int64)

    tf2targets = defaultdict(list)
    for s, d0 in zip(tf_src.tolist(), g_dst.tolist()):
        if 0 <= s < len(tfs_vocab) and 0 <= d0 < n_gene_dim:
            tf2targets[int(s)].append(int(d0))

    tf_in_prior = sorted(tf2targets.keys())
    log(f"[PRIOR] tf-gene file = {tf_gene_prior}")
    log(f"[PRIOR] tf_in_prior = {len(tf_in_prior)}")
    return tf2targets, tf_in_prior


def tf_targets_from_prior_to_selected_genes(
    tf2targets_prior: dict[int, list[int]],
    tf_in_prior: list[int],
    global_gene_vocab: list[str],
    global_tf_vocab: list[str],
    gene_sel: np.ndarray,
    min_targets: int = 10,
    max_tfs: int | None = None,
    max_targets_per_tf: int | None = None,
):
    gene_sel = np.asarray(gene_sel).astype(str)
    sel_gene_to_idx = {g: i for i, g in enumerate(gene_sel.tolist())}

    tf_names = []
    tf2targets_sel = []

    for tf_id in tf_in_prior:
        prior_gene_ids = tf2targets_prior[int(tf_id)]
        mapped = []

        for gid in prior_gene_ids:
            gid = int(gid)
            if 0 <= gid < len(global_gene_vocab):
                gname = str(global_gene_vocab[gid])
                if gname in sel_gene_to_idx:
                    mapped.append(sel_gene_to_idx[gname])

        mapped = sorted(set(mapped))
        if max_targets_per_tf is not None and len(mapped) > int(max_targets_per_tf):
            mapped = mapped[: int(max_targets_per_tf)]

        if len(mapped) >= int(min_targets):
            if 0 <= int(tf_id) < len(global_tf_vocab):
                tf_name = str(global_tf_vocab[int(tf_id)])
            else:
                tf_name = str(tf_id)
            tf_names.append(tf_name)
            tf2targets_sel.append(np.asarray(mapped, dtype=np.int64))

    if max_tfs is not None:
        tf_names = tf_names[: int(max_tfs)]
        tf2targets_sel = tf2targets_sel[: int(max_tfs)]

    return tf_names, tf2targets_sel


# ============================================================
# 6) MODEL DEFINITION (same as training script)
# ============================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int, z_dim: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.mu = nn.Linear(hid, z_dim)
        self.logvar = nn.Linear(hid, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, z_dim: int, hid: int, out_dim: int, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, out_dim),
        )

    def forward(self, z):
        return self.net(z)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def poe_fusion(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    prec1 = 1.0 / (var1 + 1e-8)
    prec2 = 1.0 / (var2 + 1e-8)
    var = 1.0 / (prec1 + prec2)
    mu = var * (mu1 * prec1 + mu2 * prec2)
    logvar = torch.log(var + 1e-8)
    return mu, logvar


class MIDASLikeVAE(nn.Module):
    def __init__(self, rna_dim: int, atac_dim: int, hid: int, z_dim: int, drop: float = 0.1):
        super().__init__()
        self.enc_rna = MLPEncoder(rna_dim, hid, z_dim, drop)
        self.enc_atac = MLPEncoder(atac_dim, hid, z_dim, drop)
        self.dec_rna = MLPDecoder(z_dim, hid, rna_dim, drop)
        self.dec_atac = MLPDecoder(z_dim, hid, atac_dim, drop)

    def forward(self, x_rna, x_atac):
        mu_rna, lv_rna = self.enc_rna(x_rna)
        mu_atac, lv_atac = self.enc_atac(x_atac)
        mu_joint, lv_joint = poe_fusion(mu_rna, lv_rna, mu_atac, lv_atac)

        z_joint = reparameterize(mu_joint, lv_joint)
        z_rna = reparameterize(mu_rna, lv_rna)
        z_atac = reparameterize(mu_atac, lv_atac)

        out = {
            "mu_rna": mu_rna,
            "lv_rna": lv_rna,
            "mu_atac": mu_atac,
            "lv_atac": lv_atac,
            "mu_joint": mu_joint,
            "lv_joint": lv_joint,
            "z_joint": z_joint,
            "z_rna": z_rna,
            "z_atac": z_atac,
            "rna_from_joint": self.dec_rna(z_joint),
            "atac_from_joint": self.dec_atac(z_joint),
            "rna_from_atac": self.dec_rna(z_atac),
            "atac_from_rna": self.dec_atac(z_rna),
        }
        return out

    @torch.no_grad()
    def encode_joint_mean(self, x_rna, x_atac):
        mu_rna, lv_rna = self.enc_rna(x_rna)
        mu_atac, lv_atac = self.enc_atac(x_atac)
        mu_joint, _ = poe_fusion(mu_rna, lv_rna, mu_atac, lv_atac)
        return mu_joint


# ============================================================
# 7) RECONSTRUCTION
# ============================================================
@torch.no_grad()
def reconstruct_all(model: nn.Module, X_rna: np.ndarray, X_atac: np.ndarray, batch_size: int = 512):
    model.eval()
    n = X_rna.shape[0]

    mus = []
    rna_hat = []
    atac_hat = []

    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        xr = torch.from_numpy(X_rna[st:ed].astype(np.float32, copy=False)).to(DEVICE)
        xa = torch.from_numpy(X_atac[st:ed].astype(np.float32, copy=False)).to(DEVICE)

        mu_joint = model.encode_joint_mean(xr, xa)
        xr_hat = model.dec_rna(mu_joint)
        xa_hat = model.dec_atac(mu_joint)

        mus.append(mu_joint.detach().cpu().numpy().astype(np.float32))
        rna_hat.append(xr_hat.detach().cpu().numpy().astype(np.float32))
        atac_hat.append(xa_hat.detach().cpu().numpy().astype(np.float32))

    return (
        np.concatenate(rna_hat, axis=0),
        np.concatenate(atac_hat, axis=0),
        np.concatenate(mus, axis=0),
    )


# ============================================================
# 8) RNA AUROC
# ============================================================
def compute_rna_auroc(
    X_rna_raw_bin_source: np.ndarray,
    rna_scores: np.ndarray,
    cell_indices: np.ndarray,
    pos_threshold: float = 0.0,
):
    X_true = (X_rna_raw_bin_source[cell_indices] > float(pos_threshold)).astype(np.uint8)
    X_score = np.asarray(rna_scores[cell_indices], dtype=np.float32)

    y_true = X_true.reshape(-1)
    y_score = X_score.reshape(-1)

    if np.unique(y_true).size < 2:
        micro_auroc = np.nan
    else:
        micro_auroc = float(roc_auc_score(y_true, y_score))

    cell_aurocs = []
    for i in range(X_true.shape[0]):
        yt = X_true[i]
        ys = X_score[i]
        if np.unique(yt).size < 2:
            continue
        try:
            cell_aurocs.append(float(roc_auc_score(yt, ys)))
        except Exception:
            pass

    macro_cell_auroc = float(np.mean(cell_aurocs)) if len(cell_aurocs) > 0 else np.nan

    return {
        "rna_micro_auroc": micro_auroc,
        "rna_macro_cell_auroc": macro_cell_auroc,
        "n_eval_cells": int(X_true.shape[0]),
        "n_eval_points": int(X_true.size),
        "n_valid_cells_for_macro": int(len(cell_aurocs)),
    }


# ============================================================
# 9) TF PROXY
# ============================================================
def regproxy_aucell_lite_from_matrix(X: np.ndarray, tf2targets: list[np.ndarray], topk: int):
    n_cells = X.shape[0]
    n_tf = len(tf2targets)
    out = np.full((n_cells, n_tf), np.nan, dtype=np.float32)

    tf_targets = [set(int(g) for g in arr.tolist()) for arr in tf2targets]

    for i in range(n_cells):
        row = np.asarray(X[i], dtype=np.float32)
        gi, _ = topk_from_dense_row(row, topk)
        top_set = set(int(x) for x in gi.tolist())
        if len(top_set) == 0:
            continue
        denom = float(min(topk, row.size))
        for j, tgtset in enumerate(tf_targets):
            out[i, j] = float(len(top_set & tgtset)) / denom
    return out


def regproxy_mean_log1p_from_matrix(X: np.ndarray, tf2targets: list[np.ndarray]):
    X = np.asarray(X, dtype=np.float32)
    X = np.log1p(np.clip(X, 0.0, None))
    n_cells = X.shape[0]
    n_tf = len(tf2targets)
    out = np.full((n_cells, n_tf), np.nan, dtype=np.float32)

    for j, idx in enumerate(tf2targets):
        if idx.size == 0:
            continue
        out[:, j] = X[:, idx].mean(axis=1).astype(np.float32)
    return out


def tfproxy_mix(aucell: np.ndarray, meanlog: np.ndarray, alpha: float):
    out = np.full_like(aucell, np.nan, dtype=np.float32)
    for j in range(aucell.shape[1]):
        out[:, j] = float(alpha) * zscore_safe(aucell[:, j]) + (1.0 - float(alpha)) * zscore_safe(meanlog[:, j])
    return out


# ============================================================
# 10) HARDER TF ROC EVALUATION
# ============================================================
def compute_tf_harder_crossfold_auc(
    pred: np.ndarray,
    reg_true_for_labels: np.ndarray,
    tf_names: list[str],
    cell_keys: np.ndarray,
    pos_q: float,
    neg_q: float,
    min_keep: int,
    min_pos: int,
    min_neg: int,
    n_folds: int = 2,
):
    pred = np.asarray(pred, dtype=np.float32)
    reg_true_for_labels = np.asarray(reg_true_for_labels, dtype=np.float32)
    cell_keys = np.asarray(cell_keys).astype(str)

    assert pred.shape == reg_true_for_labels.shape
    assert pred.shape[0] == cell_keys.shape[0]
    assert pred.shape[1] == len(tf_names)

    fold_ids = np.asarray([hash_fold(k, n_folds=n_folds) for k in cell_keys.tolist()], dtype=np.int64)

    grid = np.linspace(0, 1, FPR_GRID_N, dtype=np.float64)
    tpr_stack = []
    aucs = []
    per_tf_rows = []

    for j, tf_name in enumerate(tf_names):
        tf_scores_pred = pred[:, j]
        tf_scores_true = reg_true_for_labels[:, j]

        y_all = np.full(tf_scores_true.shape, -1, dtype=np.int8)

        for holdout_fold in range(n_folds):
            ref_mask = (fold_ids != holdout_fold)
            eval_mask = (fold_ids == holdout_fold)

            if int(ref_mask.sum()) < 10 or int(eval_mask.sum()) < 10:
                continue

            y_eval, _, _, _ = threshold_labels_from_reference(
                scores_ref=tf_scores_true[ref_mask],
                scores_eval=tf_scores_true[eval_mask],
                pos_q=pos_q,
                neg_q=neg_q,
            )
            y_all[eval_mask] = y_eval

        keep = (y_all >= 0)
        n_keep = int(keep.sum())
        n_pos = int((y_all == 1).sum())
        n_neg = int((y_all == 0).sum())

        if n_keep < min_keep or n_pos < min_pos or n_neg < min_neg:
            per_tf_rows.append({
                "tf": str(tf_name),
                "auc": np.nan,
                "n_keep": n_keep,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "used": 0,
            })
            continue

        auc0 = auc_from_scores(tf_scores_pred[y_all == 1], tf_scores_pred[y_all == 0])
        if not np.isfinite(auc0):
            per_tf_rows.append({
                "tf": str(tf_name),
                "auc": np.nan,
                "n_keep": n_keep,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "used": 0,
            })
            continue

        fpr, tpr = roc_curve_from_scores(y_all, tf_scores_pred)
        tpr_stack.append(interp_to_grid(fpr, tpr, grid))
        aucs.append(float(auc0))

        per_tf_rows.append({
            "tf": str(tf_name),
            "auc": float(auc0),
            "n_keep": n_keep,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "used": 1,
        })

    if len(aucs) < 1:
        return None, fold_ids

    tpr_mean = np.mean(np.stack(tpr_stack, axis=0), axis=0)
    mean_auc = float(np.mean(np.asarray(aucs, np.float32)))

    out = {
        "grid": grid,
        "tpr_mean": tpr_mean,
        "mean_auc": mean_auc,
        "used_tf": int(len(aucs)),
        "per_tf": pd.DataFrame(per_tf_rows),
    }
    return out, fold_ids


# ============================================================
# 11) MAIN
# ============================================================
def main():
    log(f"[DEVICE] {DEVICE}")
    log(f"[SAMPLE] {SAMPLE_ID}")
    log("[MODE] HARDER TF EVAL | DISJOINT TARGETS | MIDAS-LIKE BASELINE")

    assert MODEL_PATH.exists(), f"Missing model.pt: {MODEL_PATH}"
    assert PRIORS_DIR.exists(), f"Missing PRIORS_DIR: {PRIORS_DIR}"
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing GLOBAL_GENE_VOCAB: {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing GLOBAL_TF_VOCAB: {GLOBAL_TF_VOCAB}"

    log("[RUN] loading checkpoint ...")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model_cfg = ckpt["config"]

    log("[RUN] loading single-sample paired data ...")
    X_rna_raw, X_atac_raw, gene_names_rna, gene_names_atac, obs = load_single_sample_paired_data(SAMPLE_ID)

    log("[RUN] preprocessing with saved feature selection + normalization ...")
    X_rna, X_atac, gene_sel, atac_sel, keep_rna, keep_atac = preprocess_with_checkpoint(
        X_rna_raw=X_rna_raw,
        X_atac_raw=X_atac_raw,
        gene_names_rna=gene_names_rna,
        gene_names_atac=gene_names_atac,
        ckpt=ckpt,
    )

    eval_idx = np.arange(obs.shape[0], dtype=np.int64)
    log(f"[EVAL] single-sample all cells n={len(eval_idx)}")

    log("[RUN] rebuilding model ...")
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("HIDDEN_DIM", 512)))
    latent_dim = int(model_cfg.get("latent_dim", model_cfg.get("LATENT_DIM", 64)))
    dropout = float(model_cfg.get("dropout", model_cfg.get("DROP", 0.1)))

    model = MIDASLikeVAE(
        rna_dim=X_rna.shape[1],
        atac_dim=X_atac.shape[1],
        hid=hidden_dim,
        z_dim=latent_dim,
        drop=dropout,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    log("[RUN] reconstructing ...")
    RNA_HAT, ATAC_HAT, MU = reconstruct_all(model, X_rna, X_atac, batch_size=BATCH_SIZE)

    # ---------------- RNA AUROC ----------------
    log("[EVAL] computing RNA AUROC ...")
    rna_auc = compute_rna_auroc(
        X_rna_raw_bin_source=X_rna_raw[:, keep_rna],
        rna_scores=RNA_HAT,
        cell_indices=eval_idx,
        pos_threshold=RNA_POS_THRESHOLD,
    )
    log(
        f"[RNA AUROC] micro={rna_auc['rna_micro_auroc']:.6f} | "
        f"macro_cell={rna_auc['rna_macro_cell_auroc']:.6f}"
    )

    # ---------------- TF ROC/AUC ----------------
    tf_summary = None
    tf_per_tf = None

    log("[TF] loading global vocab + prior ...")
    global_gene_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    global_tf_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    log(f"[TF DEBUG] len(global_gene_vocab) = {len(global_gene_vocab)}")
    log(f"[TF DEBUG] len(global_tf_vocab) = {len(global_tf_vocab)}")
    log(f"[TF DEBUG] len(gene_sel) = {len(gene_sel)}")

    tf2targets_prior, tf_in_prior = read_tf_prior(
        prior_dir=PRIORS_DIR,
        tfs_vocab=global_tf_vocab,
        n_gene_dim=len(global_gene_vocab),
    )

    tf_names, tf2targets = tf_targets_from_prior_to_selected_genes(
        tf2targets_prior=tf2targets_prior,
        tf_in_prior=tf_in_prior,
        global_gene_vocab=global_gene_vocab,
        global_tf_vocab=global_tf_vocab,
        gene_sel=gene_sel,
        min_targets=TF_MIN_TARGETS_EVAL,
        max_tfs=TF_MAX_TFS,
        max_targets_per_tf=TF_MAX_TARGETS_PER_TF,
    )

    log(f"[TF] usable TFs after intersecting prior targets with selected genes = {len(tf_names)}")
    if len(tf2targets) > 0:
        lens = np.asarray([len(x) for x in tf2targets], dtype=np.int64)
        log(f"[TF DEBUG] targets per TF before disjoint split: min={lens.min()} median={np.median(lens)} max={lens.max()}")

    # -------- disjoint target split --------
    tf2targets_label, tf2targets_pred = split_tf_targets_disjoint(
        tf2targets=tf2targets,
        seed=SEED,
        min_each_side=TF_MIN_EACH_SIDE,
    )

    valid_mask = np.asarray(
        [(a.size >= TF_MIN_EACH_SIDE and b.size >= TF_MIN_EACH_SIDE)
         for a, b in zip(tf2targets_label, tf2targets_pred)],
        dtype=bool,
    )

    tf_names = [x for x, m in zip(tf_names, valid_mask.tolist()) if m]
    tf2targets_label = [x for x, m in zip(tf2targets_label, valid_mask.tolist()) if m]
    tf2targets_pred = [x for x, m in zip(tf2targets_pred, valid_mask.tolist()) if m]

    log(f"[TF HARDER] usable TFs after disjoint target split = {len(tf_names)}")
    if len(tf2targets_label) > 0:
        lens_a = np.asarray([x.size for x in tf2targets_label], dtype=np.int64)
        lens_b = np.asarray([x.size for x in tf2targets_pred], dtype=np.int64)
        log(f"[TF HARDER] label-targets per TF: min={lens_a.min()} median={np.median(lens_a)} max={lens_a.max()}")
        log(f"[TF HARDER] pred-targets per TF: min={lens_b.min()} median={np.median(lens_b)} max={lens_b.max()}")

    if len(tf_names) == 0:
        log("[TF][WARN] no TF left after disjoint target split")
        fold_ids = np.asarray([hash_fold(k, n_folds=N_FOLDS) for k in obs["cell_key"].astype(str).tolist()], dtype=np.int64)
    else:
        X_true_eval = np.asarray(X_rna_raw[:, keep_rna], dtype=np.float32)
        X_pred_eval = np.asarray(RNA_HAT, dtype=np.float32)

        log("[TF] building true proxy from raw RNA (AUCELL only, label targets) ...")
        REG_true_for_labels = regproxy_aucell_lite_from_matrix(
            X_true_eval,
            tf2targets_label,
            topk=AUCELL_TOPK,
        )

        log("[TF] building pred proxy from reconstructed RNA (mixed, pred targets) ...")
        pred_au = regproxy_aucell_lite_from_matrix(
            X_pred_eval,
            tf2targets_pred,
            topk=AUCELL_TOPK,
        )
        pred_mn = regproxy_mean_log1p_from_matrix(
            X_pred_eval,
            tf2targets_pred,
        )
        REG_pred = tfproxy_mix(pred_au, pred_mn, TF_PROXY_MIX_ALPHA)

        log("[TF] computing harder cross-fold ROC/AUC ...")
        overall, fold_ids = compute_tf_harder_crossfold_auc(
            pred=REG_pred[eval_idx],
            reg_true_for_labels=REG_true_for_labels[eval_idx],
            tf_names=tf_names,
            cell_keys=obs["cell_key"].astype(str).values[eval_idx],
            pos_q=POS_Q,
            neg_q=NEG_Q,
            min_keep=OVERALL_MIN_KEEP,
            min_pos=OVERALL_MIN_POS,
            min_neg=OVERALL_MIN_NEG,
            n_folds=N_FOLDS,
        )

        if overall is None:
            log("[TF][WARN] no TF passed harder cross-fold filters for ROC")
            tf_summary = None
            tf_per_tf = None
        else:
            tf_summary = {
                "tf_mean_auc": float(overall["mean_auc"]),
                "tf_used": int(overall["used_tf"]),
                "fpr_grid": overall["grid"].astype(float).tolist(),
                "tpr_mean": overall["tpr_mean"].astype(float).tolist(),
                "mode": "harder_crossfold_disjoint_targets",
                "n_folds": int(N_FOLDS),
                "pos_q": float(POS_Q),
                "neg_q": float(NEG_Q),
                "aucell_topk": int(AUCELL_TOPK),
                "tf_max_targets_per_tf": None if TF_MAX_TARGETS_PER_TF is None else int(TF_MAX_TARGETS_PER_TF),
                "tf_min_each_side": int(TF_MIN_EACH_SIDE),
            }
            tf_per_tf = overall["per_tf"].sort_values(["used", "auc"], ascending=[False, False]).reset_index(drop=True)

            log(f"[TF ROC HARDER] mean_auc={tf_summary['tf_mean_auc']:.6f} | tf_used={tf_summary['tf_used']}")

    # ---------------- Save ----------------
    np.save(OUT_DIR / "latent_mu_eval.npy", MU.astype(np.float32))
    np.save(OUT_DIR / "rna_recon_eval.npy", RNA_HAT.astype(np.float32))
    np.save(OUT_DIR / "atac_recon_eval.npy", ATAC_HAT.astype(np.float32))

    obs_out = obs.copy()
    obs_out["latent_row"] = np.arange(obs_out.shape[0], dtype=np.int64)
    obs_out["fold_id"] = fold_ids.astype(np.int64)
    obs_out.to_csv(OUT_DIR / "obs_eval.csv", index=False)

    with open(OUT_DIR / "rna_auroc.json", "w", encoding="utf-8") as f:
        json.dump(rna_auc, f, ensure_ascii=False, indent=2)

    if tf_per_tf is not None:
        tf_per_tf.to_csv(OUT_DIR / "tf_proxy_auc_per_tf_harder.csv", index=False)

    if tf_summary is not None:
        with open(OUT_DIR / "tf_proxy_auc_summary_harder.json", "w", encoding="utf-8") as f:
            json.dump(tf_summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "cell_key": obs["cell_key"].astype(str).tolist(),
        "fold_id": fold_ids.astype(int).tolist(),
    }).to_csv(OUT_DIR / "cell_fold_assignment.csv", index=False)

    summary = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "device": DEVICE,
        "model_path": str(MODEL_PATH),
        "mode": "harder_crossfold_disjoint_targets",
        "n_cells": int(obs.shape[0]),
        "rna_shape_after_filter": [int(X_rna.shape[0]), int(X_rna.shape[1])],
        "atac_shape_after_filter": [int(X_atac.shape[0]), int(X_atac.shape[1])],
        "latent_dim": int(latent_dim),
        "seed": int(model_cfg.get("seed", SEED)),
        "pos_q": float(POS_Q),
        "neg_q": float(NEG_Q),
        "aucell_topk": int(AUCELL_TOPK),
        "tf_min_targets_eval": int(TF_MIN_TARGETS_EVAL),
        "tf_max_targets_per_tf": None if TF_MAX_TARGETS_PER_TF is None else int(TF_MAX_TARGETS_PER_TF),
        "tf_min_each_side": int(TF_MIN_EACH_SIDE),
        "n_folds": int(N_FOLDS),
        "rna_micro_auroc": None if not np.isfinite(rna_auc["rna_micro_auroc"]) else float(rna_auc["rna_micro_auroc"]),
        "rna_macro_cell_auroc": None if not np.isfinite(rna_auc["rna_macro_cell_auroc"]) else float(rna_auc["rna_macro_cell_auroc"]),
        "tf_mean_auc": None if (tf_summary is None) else float(tf_summary["tf_mean_auc"]),
        "tf_used": None if (tf_summary is None) else int(tf_summary["tf_used"]),
    }
    with open(OUT_DIR / "summary_eval_harder.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"[SAVE] latent_mu_eval.npy -> {OUT_DIR / 'latent_mu_eval.npy'}")
    log(f"[SAVE] rna_recon_eval.npy -> {OUT_DIR / 'rna_recon_eval.npy'}")
    log(f"[SAVE] atac_recon_eval.npy -> {OUT_DIR / 'atac_recon_eval.npy'}")
    log(f"[SAVE] obs_eval.csv -> {OUT_DIR / 'obs_eval.csv'}")
    log(f"[SAVE] {OUT_DIR / 'rna_auroc.json'}")
    log(f"[SAVE] {OUT_DIR / 'cell_fold_assignment.csv'}")
    if tf_per_tf is not None:
        log(f"[SAVE] {OUT_DIR / 'tf_proxy_auc_per_tf_harder.csv'}")
    if tf_summary is not None:
        log(f"[SAVE] {OUT_DIR / 'tf_proxy_auc_summary_harder.json'}")
    log(f"[SAVE] {OUT_DIR / 'summary_eval_harder.json'}")
    log("[DONE]")


if __name__ == "__main__":
    main()
