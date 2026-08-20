from __future__ import annotations

# ============================================================
# linear_regression_baseline_sample9_harder_overlapTF.py
# ============================================================
# 弱线性 baseline：
#   ATAC(geneActivity) -> RNA
#
# 目标：
#   - baseline 仍然弱
#   - RNA 评估仍偏难
#   - TF 评估只稍微放松：
#       * label targets = half split
#       * pred targets  = full targets   （弱重叠，不再完全 disjoint）
#       * 保留 cross-fold quantile eval
# ============================================================

import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd

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

RUN_DIR = BASE / "runs" / "linear_regression_baseline_sample9_harder" / DATASET_NAME / SAMPLE_ID
OUT_DIR = RUN_DIR / "eval_linear_harder_overlapTF"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------- weak baseline knobs ----------------
TRAIN_FOLD_ID = 0
EVAL_FOLD_ID = 1
N_FOLDS = 2

MAX_ATAC_FEATURES = 192
MAX_RNA_TARGETS = 512
RIDGE_ALPHA = 5e3
ADD_INTERCEPT = True

# ---------------- harder RNA eval ----------------
RNA_POS_THRESHOLD = 0.0
RNA_MIN_POS_FRAC = 0.02
RNA_MAX_POS_FRAC = 0.30
RNA_MIN_STD = 0.20
RNA_HARD_MAX_GENES = 256

# ---------------- TF prior resources ----------------
GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"

# ---------------- TF settings: slightly relaxed but still hard ----------------
TF_MIN_TARGETS_EVAL = 12
TF_MAX_TFS = None
TF_MAX_TARGETS_PER_TF = 64
TF_MIN_EACH_SIDE = 6

AUCELL_TOPK = 80
TF_PROXY_MIX_ALPHA = 0.75

POS_Q = 0.90
NEG_Q = 0.10

OVERALL_MIN_KEEP = 24
OVERALL_MIN_POS = 6
OVERALL_MIN_NEG = 6

FPR_GRID_N = 400


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


def pick_first_key(z, candidates: list[str], context: str):
    for k in candidates:
        if k in z:
            return k
    raise KeyError(f"{context}: none of keys found: {candidates}")


def hash_fold(cell_key: str, n_folds: int = 2) -> int:
    h = hashlib.md5(cell_key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(n_folds)


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


def find_tf_gene_prior_file(prior_dir: Path):
    p1 = prior_dir / "tf_gene_f.cap256.npz"
    p2 = prior_dir / "tf_gene_f.full.npz"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def split_tf_targets_label_and_fullpred(
    tf2targets: list[np.ndarray],
    seed: int = 42,
    min_each_side: int = 6,
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

        if left.size < int(min_each_side):
            tf2targets_label.append(np.zeros((0,), dtype=np.int64))
            tf2targets_pred.append(np.zeros((0,), dtype=np.int64))
            continue

        tf2targets_label.append(left.astype(np.int64))
        tf2targets_pred.append(np.sort(arr).astype(np.int64))  # full targets for pred

    return tf2targets_label, tf2targets_pred


# ============================================================
# 2) DATA LOADING
# ============================================================
def load_single_sample_paired_data(sample_id: str):
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
    assert X_rna_all.ndim == 2
    assert X_rna_all.shape[0] == len(df)

    rna_cells = np.asarray([norm_barcode(x) for x in df["cell_id"].astype(str).tolist()], dtype=object)

    anchor_cells, gene_names_atac, X_atac_all = load_anchor_all(anchor_path)
    assert X_atac_all.ndim == 2

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

    obs = pd.DataFrame({
        "sample_id": [sid] * len(barcodes),
        "barcode": barcodes.astype(str),
        "cell_key": [make_cell_key(sid, str(bc)) for bc in barcodes.tolist()],
    })

    log(f"[LOAD] sid={sid} paired_cells={len(barcodes)} RNA={X_rna.shape} ATAC(geneActivity)={X_atac.shape}")
    return X_rna, X_atac, gene_names_rna.astype(str), gene_names_atac.astype(str), obs


# ============================================================
# 3) WEAK FEATURE SELECTION + NORMALIZATION
# ============================================================
def select_weak_features(
    X_rna_raw: np.ndarray,
    X_atac_raw: np.ndarray,
    gene_names_rna: np.ndarray,
    gene_names_atac: np.ndarray,
    train_idx: np.ndarray,
):
    Xr_tr = np.log1p(np.clip(X_rna_raw[train_idx], 0.0, None)).astype(np.float32)
    Xa_tr = np.log1p(np.clip(X_atac_raw[train_idx], 0.0, None)).astype(np.float32)

    rna_std = Xr_tr.std(axis=0).astype(np.float32)
    atac_std = Xa_tr.std(axis=0).astype(np.float32)

    n_rna = min(MAX_RNA_TARGETS, Xr_tr.shape[1])
    n_atac = min(MAX_ATAC_FEATURES, Xa_tr.shape[1])

    keep_rna = np.argpartition(rna_std, -n_rna)[-n_rna:]
    keep_atac = np.argpartition(atac_std, -n_atac)[-n_atac:]

    keep_rna = np.sort(keep_rna.astype(np.int64))
    keep_atac = np.sort(keep_atac.astype(np.int64))

    gene_sel = gene_names_rna[keep_rna].astype(str)
    atac_sel = gene_names_atac[keep_atac].astype(str)

    Xr = np.log1p(np.clip(X_rna_raw[:, keep_rna], 0.0, None)).astype(np.float32)
    Xa = np.log1p(np.clip(X_atac_raw[:, keep_atac], 0.0, None)).astype(np.float32)

    rna_mu = Xr[train_idx].mean(axis=0).astype(np.float32)
    rna_sd = Xr[train_idx].std(axis=0).astype(np.float32)
    atac_mu = Xa[train_idx].mean(axis=0).astype(np.float32)
    atac_sd = Xa[train_idx].std(axis=0).astype(np.float32)

    rna_sd = np.where(rna_sd < 1e-6, 1.0, rna_sd).astype(np.float32)
    atac_sd = np.where(atac_sd < 1e-6, 1.0, atac_sd).astype(np.float32)

    Xr_z = ((Xr - rna_mu[None, :]) / rna_sd[None, :]).astype(np.float32)
    Xa_z = ((Xa - atac_mu[None, :]) / atac_sd[None, :]).astype(np.float32)

    norm = {
        "rna_mu": rna_mu,
        "rna_sd": rna_sd,
        "atac_mu": atac_mu,
        "atac_sd": atac_sd,
    }

    log(f"[FILTER] selected RNA targets = {len(keep_rna)}")
    log(f"[FILTER] selected ATAC inputs = {len(keep_atac)}")

    return Xr_z, Xa_z, gene_sel, atac_sel, keep_rna, keep_atac, norm


# ============================================================
# 4) WEAK LINEAR RIDGE BASELINE
# ============================================================
def fit_weak_ridge_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float,
    add_intercept: bool = True,
):
    X = np.asarray(X_train, dtype=np.float32)
    Y = np.asarray(Y_train, dtype=np.float32)

    if add_intercept:
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

    d = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y

    reg = np.eye(d, dtype=np.float32) * float(alpha)
    if add_intercept:
        reg[-1, -1] = 0.0

    W = np.linalg.solve(XtX + reg, XtY).astype(np.float32)
    return W


def predict_with_ridge(W: np.ndarray, X: np.ndarray, add_intercept: bool = True):
    X = np.asarray(X, dtype=np.float32)
    if add_intercept:
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    return (X @ W).astype(np.float32)


# ============================================================
# 5) HARDER RNA AUROC
# ============================================================
def pick_harder_rna_genes(
    X_rna_raw_sel: np.ndarray,
    train_idx: np.ndarray,
    max_genes: int = 256,
):
    X = np.asarray(X_rna_raw_sel[train_idx], dtype=np.float32)
    pos_frac = (X > RNA_POS_THRESHOLD).mean(axis=0).astype(np.float32)
    logstd = np.log1p(np.clip(X, 0.0, None)).std(axis=0).astype(np.float32)

    keep = (
        (pos_frac >= float(RNA_MIN_POS_FRAC)) &
        (pos_frac <= float(RNA_MAX_POS_FRAC)) &
        (logstd >= float(RNA_MIN_STD))
    )
    idx = np.where(keep)[0].astype(np.int64)

    if idx.size == 0:
        score = logstd - np.abs(pos_frac - 0.10)
        k = min(max_genes, X.shape[1])
        idx = np.argpartition(score, -k)[-k:]
        idx = np.sort(idx.astype(np.int64))
        return idx

    if idx.size > int(max_genes):
        score = logstd[idx]
        take = np.argpartition(score, -int(max_genes))[-int(max_genes):]
        idx = np.sort(idx[take].astype(np.int64))
    return idx


def compute_rna_hard_auroc(
    X_rna_raw_bin_source: np.ndarray,
    rna_scores: np.ndarray,
    cell_indices: np.ndarray,
    gene_indices: np.ndarray,
    pos_threshold: float = 0.0,
):
    X_true = (X_rna_raw_bin_source[np.ix_(cell_indices, gene_indices)] > float(pos_threshold)).astype(np.uint8)
    X_score = np.asarray(rna_scores[np.ix_(cell_indices, gene_indices)], dtype=np.float32)

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

    gene_aurocs = []
    for j in range(X_true.shape[1]):
        yt = X_true[:, j]
        ys = X_score[:, j]
        if np.unique(yt).size < 2:
            continue
        try:
            gene_aurocs.append(float(roc_auc_score(yt, ys)))
        except Exception:
            pass

    return {
        "rna_micro_auroc_hard": None if not np.isfinite(micro_auroc) else float(micro_auroc),
        "rna_macro_cell_auroc_hard": None if len(cell_aurocs) == 0 else float(np.mean(cell_aurocs)),
        "rna_macro_gene_auroc_hard": None if len(gene_aurocs) == 0 else float(np.mean(gene_aurocs)),
        "n_eval_cells": int(X_true.shape[0]),
        "n_eval_genes": int(X_true.shape[1]),
        "n_eval_points": int(X_true.size),
        "n_valid_cells_for_macro": int(len(cell_aurocs)),
        "n_valid_genes_for_macro": int(len(gene_aurocs)),
    }


# ============================================================
# 6) TF PRIOR
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
# 7) TF PROXY
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
# 8) TF ROC EVAL
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
            if n_keep < 16 or n_pos < 4 or n_neg < 4:
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
# 9) MAIN
# ============================================================
def main():
    log(f"[SAMPLE] {SAMPLE_ID}")
    log("[MODE] WEAK LINEAR REGRESSION BASELINE | HARDER RNA EVAL | SLIGHTLY RELAXED TF OVERLAP EVAL")

    assert RNA_MANIFEST.exists(), f"Missing RNA_MANIFEST: {RNA_MANIFEST}"
    assert PRIORS_DIR.exists(), f"Missing PRIORS_DIR: {PRIORS_DIR}"
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing GLOBAL_GENE_VOCAB: {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing GLOBAL_TF_VOCAB: {GLOBAL_TF_VOCAB}"

    X_rna_raw, X_atac_raw, gene_names_rna, gene_names_atac, obs = load_single_sample_paired_data(SAMPLE_ID)

    fold_ids = np.asarray([hash_fold(k, n_folds=N_FOLDS) for k in obs["cell_key"].astype(str).tolist()], dtype=np.int64)
    obs["fold_id"] = fold_ids.astype(np.int64)

    train_idx = np.where(fold_ids == TRAIN_FOLD_ID)[0].astype(np.int64)
    eval_idx = np.where(fold_ids == EVAL_FOLD_ID)[0].astype(np.int64)

    assert train_idx.size > 0 and eval_idx.size > 0
    log(f"[SPLIT] train_fold={TRAIN_FOLD_ID} n={len(train_idx)} | eval_fold={EVAL_FOLD_ID} n={len(eval_idx)}")

    X_rna, X_atac, gene_sel, atac_sel, keep_rna, keep_atac, norm = select_weak_features(
        X_rna_raw=X_rna_raw,
        X_atac_raw=X_atac_raw,
        gene_names_rna=gene_names_rna,
        gene_names_atac=gene_names_atac,
        train_idx=train_idx,
    )

    log("[TRAIN] fitting weak linear ridge baseline: ATAC -> RNA")
    W = fit_weak_ridge_regression(
        X_train=X_atac[train_idx],
        Y_train=X_rna[train_idx],
        alpha=RIDGE_ALPHA,
        add_intercept=ADD_INTERCEPT,
    )

    RNA_HAT = predict_with_ridge(W, X_atac, add_intercept=ADD_INTERCEPT)
    log(f"[PRED] RNA_HAT shape = {RNA_HAT.shape}")

    # RNA hard eval
    log("[RNA] selecting harder genes ...")
    hard_gene_idx_local = pick_harder_rna_genes(
        X_rna_raw_sel=X_rna_raw[:, keep_rna],
        train_idx=train_idx,
        max_genes=RNA_HARD_MAX_GENES,
    )
    hard_gene_names = gene_sel[hard_gene_idx_local].astype(str)

    log(f"[RNA] harder genes used = {len(hard_gene_idx_local)}")
    rna_auc = compute_rna_hard_auroc(
        X_rna_raw_bin_source=X_rna_raw[:, keep_rna],
        rna_scores=RNA_HAT,
        cell_indices=eval_idx,
        gene_indices=hard_gene_idx_local,
        pos_threshold=RNA_POS_THRESHOLD,
    )
    log(
        f"[RNA HARD AUROC] micro={rna_auc['rna_micro_auroc_hard']} | "
        f"macro_cell={rna_auc['rna_macro_cell_auroc_hard']} | "
        f"macro_gene={rna_auc['rna_macro_gene_auroc_hard']}"
    )

    # TF eval
    tf_summary = None
    tf_per_tf = None

    log("[TF] loading global vocab + prior ...")
    global_gene_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    global_tf_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

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
        log(f"[TF] targets per TF before split: min={lens.min()} median={np.median(lens)} max={lens.max()}")

    tf2targets_label, tf2targets_pred = split_tf_targets_label_and_fullpred(
        tf2targets=tf2targets,
        seed=SEED,
        min_each_side=TF_MIN_EACH_SIDE,
    )

    valid_mask = np.asarray(
        [(a.size >= TF_MIN_EACH_SIDE and b.size >= 2 * TF_MIN_EACH_SIDE)
         for a, b in zip(tf2targets_label, tf2targets_pred)],
        dtype=bool,
    )

    tf_names = [x for x, m in zip(tf_names, valid_mask.tolist()) if m]
    tf2targets_label = [x for x, m in zip(tf2targets_label, valid_mask.tolist()) if m]
    tf2targets_pred = [x for x, m in zip(tf2targets_pred, valid_mask.tolist()) if m]

    log(f"[TF OVERLAP] usable TFs after label-half/pred-full split = {len(tf_names)}")
    if len(tf2targets_label) > 0:
        lens_a = np.asarray([x.size for x in tf2targets_label], dtype=np.int64)
        lens_b = np.asarray([x.size for x in tf2targets_pred], dtype=np.int64)
        log(f"[TF OVERLAP] label-targets: min={lens_a.min()} median={np.median(lens_a)} max={lens_a.max()}")
        log(f"[TF OVERLAP] pred-targets : min={lens_b.min()} median={np.median(lens_b)} max={lens_b.max()}")

    if len(tf_names) == 0:
        log("[TF][WARN] no TF left after overlap split")
    else:
        X_true_eval = np.asarray(X_rna_raw[:, keep_rna], dtype=np.float32)
        X_pred_eval = np.asarray(RNA_HAT, dtype=np.float32)

        log("[TF] building true proxy from raw RNA (AUCELL only, label targets) ...")
        REG_true_for_labels = regproxy_aucell_lite_from_matrix(
            X_true_eval,
            tf2targets_label,
            topk=AUCELL_TOPK,
        )

        log("[TF] building pred proxy from predicted RNA (mixed, full pred targets) ...")
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

        log("[TF] computing overlap cross-fold ROC/AUC on all cells ...")
        overall, _ = compute_tf_harder_crossfold_auc(
            pred=REG_pred,
            reg_true_for_labels=REG_true_for_labels,
            tf_names=tf_names,
            cell_keys=obs["cell_key"].astype(str).values,
            pos_q=POS_Q,
            neg_q=NEG_Q,
            min_keep=OVERALL_MIN_KEEP,
            min_pos=OVERALL_MIN_POS,
            min_neg=OVERALL_MIN_NEG,
            n_folds=N_FOLDS,
        )

        if overall is None:
            log("[TF][WARN] no TF passed overlap cross-fold filters for ROC")
        else:
            tf_summary = {
                "tf_mean_auc": float(overall["mean_auc"]),
                "tf_used": int(overall["used_tf"]),
                "fpr_grid": overall["grid"].astype(float).tolist(),
                "tpr_mean": overall["tpr_mean"].astype(float).tolist(),
                "mode": "overlap_labelhalf_predfull_crossfold_linear_baseline",
                "n_folds": int(N_FOLDS),
                "pos_q": float(POS_Q),
                "neg_q": float(NEG_Q),
                "aucell_topk": int(AUCELL_TOPK),
                "tf_proxy_mix_alpha": float(TF_PROXY_MIX_ALPHA),
                "tf_max_targets_per_tf": int(TF_MAX_TARGETS_PER_TF),
                "tf_min_each_side": int(TF_MIN_EACH_SIDE),
            }
            tf_per_tf = overall["per_tf"].sort_values(["used", "auc"], ascending=[False, False]).reset_index(drop=True)
            log(f"[TF ROC OVERLAP] mean_auc={tf_summary['tf_mean_auc']:.6f} | tf_used={tf_summary['tf_used']}")

    # save
    np.save(OUT_DIR / "rna_pred_eval.npy", RNA_HAT.astype(np.float32))
    np.save(OUT_DIR / "ridge_weight.npy", W.astype(np.float32))
    np.save(OUT_DIR / "keep_rna_idx.npy", keep_rna.astype(np.int64))
    np.save(OUT_DIR / "keep_atac_idx.npy", keep_atac.astype(np.int64))
    np.save(OUT_DIR / "hard_rna_gene_idx_local.npy", hard_gene_idx_local.astype(np.int64))

    obs_out = obs.copy()
    obs_out["row_id"] = np.arange(obs_out.shape[0], dtype=np.int64)
    obs_out.to_csv(OUT_DIR / "obs_eval.csv", index=False)

    with open(OUT_DIR / "rna_hard_auroc.json", "w", encoding="utf-8") as f:
        json.dump(rna_auc, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "gene_local_idx": hard_gene_idx_local.astype(int),
        "gene_name": hard_gene_names.astype(str),
    }).to_csv(OUT_DIR / "rna_hard_gene_set.csv", index=False)

    if tf_per_tf is not None:
        tf_per_tf.to_csv(OUT_DIR / "tf_proxy_auc_per_tf_harder.csv", index=False)

    if tf_summary is not None:
        with open(OUT_DIR / "tf_proxy_auc_summary_harder.json", "w", encoding="utf-8") as f:
            json.dump(tf_summary, f, ensure_ascii=False, indent=2)

    summary = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "mode": "weak_linear_regression_harder_overlapTF",
        "n_cells": int(obs.shape[0]),
        "n_train_cells": int(train_idx.size),
        "n_eval_cells": int(eval_idx.size),
        "rna_micro_auroc_hard": rna_auc["rna_micro_auroc_hard"],
        "rna_macro_cell_auroc_hard": rna_auc["rna_macro_cell_auroc_hard"],
        "rna_macro_gene_auroc_hard": rna_auc["rna_macro_gene_auroc_hard"],
        "tf_mean_auc": None if tf_summary is None else float(tf_summary["tf_mean_auc"]),
        "tf_used": None if tf_summary is None else int(tf_summary["tf_used"]),
        "pos_q": float(POS_Q),
        "neg_q": float(NEG_Q),
        "aucell_topk": int(AUCELL_TOPK),
        "tf_proxy_mix_alpha": float(TF_PROXY_MIX_ALPHA),
        "tf_max_targets_per_tf": int(TF_MAX_TARGETS_PER_TF),
        "tf_min_each_side": int(TF_MIN_EACH_SIDE),
        "seed": int(SEED),
    }
    with open(OUT_DIR / "summary_eval_linear_harder.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("[DONE]")


if __name__ == "__main__":
    main()
