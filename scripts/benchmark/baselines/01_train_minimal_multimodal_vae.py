from __future__ import annotations

# ============================================================
# train_minimal_multimodal_vae_crc.py
# ============================================================
# 不依赖 multiDGD 包的最小双模态 shared-latent VAE
#
# 功能：
#   1) 读取当前目录结构下的 paired RNA + ATAC
#   2) 自动按 sample_id + barcode 对齐
#   3) 训练一个 shared-latent multimodal VAE
#   4) 导出 latent / obs / summary / model
#
# 目录假设：
#   RNA:
#     .../processed/RNA/manifest_rna.tsv
#     .../processed/RNA/{sid}.rna_meta.npz
#     .../processed/RNA/{sid}.X_rna_counts_dense.npy
#
#   ATAC:
#     .../processed/ATAC/manifest_atac.tsv   (可有可无；读不稳时自动扫目录)
#     .../processed/ATAC/{sid}.atac_meta.npz
#     .../processed/ATAC/{sid}.X_atac_peaks_csr.npz
#
# 训练目标：
#   - encoder: [RNA_top + ATAC_top] -> z_mean, z_logvar
#   - decoder_rna: z -> RNA
#   - decoder_atac: z -> ATAC
#   - loss = recon_rna + recon_atac + beta * KL
#
# 说明：
#   - 为了稳和省显存，默认先做 HVG / HVP 过滤
#   - 重建目标用 log1p(CPM/TP10K) 后的连续值，loss 用 MSE
#   - 主要用途是导出 shared latent 做流程对比，不追求 paper-level 复刻
# ============================================================

import json
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0) CONFIG
# ============================================================
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"

DATASET_NAME = "CRC-Tissue"

RNA_ROOT = WORLD_ROOT / DATASET_NAME / "processed" / "RNA"
ATAC_ROOT = WORLD_ROOT / DATASET_NAME / "processed" / "ATAC"

RNA_MANIFEST = RNA_ROOT / "manifest_rna.tsv"
ATAC_MANIFEST = ATAC_ROOT / "manifest_atac.tsv"

RUN_DIR = BASE / "runs" / "minimal_multimodal_vae_crc"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "minimal_multimodal_vae_crc"
OUT_DIR = RUN_DIR / MODEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_FRAC_FROM_TRAIN = 0.10
MAX_CELLS_PER_SAMPLE = None

# -------- feature filtering --------
RNA_HVG = 4000
ATAC_HVP = 12000
MIN_RNA_CELL_FRAC = 0.01
MIN_ATAC_CELL_FRAC = 0.005

# -------- training --------
LATENT_DIM = 32
ENC_HID = 512
DEC_HID = 512
DROP = 0.10

BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
BETA_KL = 1e-3
GRAD_CLIP = 5.0

SAVE_BEST_BY_VAL = True

# -------- preprocess --------
RNA_LOG_NORM_SCALE = 1e4
ATAC_LOG_NORM_SCALE = 1e4

# -------- output --------
EXPORT_LATENT_FLOAT16 = False


def log(*args):
    print(*args, flush=True)


# ============================================================
# 1) HELPERS
# ============================================================
def norm_barcode(x: str) -> str:
    x = str(x)
    return x.split("::", 1)[-1] if "::" in x else x


def make_cell_key(sample_id: str, barcode: str) -> str:
    return f"{sample_id}::{barcode}"


def split_5050(key: str) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 1


def split_train_val_from_train_mask(train_mask: np.ndarray, val_frac: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.where(train_mask)[0]
    if idx.size == 0:
        return np.zeros_like(train_mask, dtype=bool), np.zeros_like(train_mask, dtype=bool)
    n_val = max(1, int(round(idx.size * float(val_frac))))
    val_idx = rng.choice(idx, size=n_val, replace=False)
    val_mask = np.zeros_like(train_mask, dtype=bool)
    val_mask[val_idx] = True
    train_mask2 = train_mask.copy()
    train_mask2[val_idx] = False
    return train_mask2, val_mask


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


def debug_print_sid_overlap(rna_sids: list[str], atac_sids: list[str], max_show: int = 10):
    rna_sids = sorted([str(x) for x in rna_sids])
    atac_sids = sorted([str(x) for x in atac_sids])

    exact = sorted(set(rna_sids) & set(atac_sids))
    canon_rna = sorted(set(canonical_sid(x) for x in rna_sids))
    canon_atac = sorted(set(canonical_sid(x) for x in atac_sids))
    canon_overlap = sorted(set(canon_rna) & set(canon_atac))

    log(f"[SID] RNA sample ids n={len(rna_sids)}")
    log(f"[SID] ATAC sample ids n={len(atac_sids)}")
    log(f"[SID] exact overlap n={len(exact)}")
    log(f"[SID] canonical overlap n={len(canon_overlap)}")

    if len(rna_sids) > 0:
        log(f"[SID] RNA examples: {rna_sids[:max_show]}")
    if len(atac_sids) > 0:
        log(f"[SID] ATAC examples: {atac_sids[:max_show]}")
    if len(canon_overlap) > 0:
        log(f"[SID] canonical overlap examples: {canon_overlap[:max_show]}")


def ensure_csr(x):
    if sp.isspmatrix_csr(x):
        return x
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def pick_first_key(z, candidates: list[str], context: str):
    for k in candidates:
        if k in z:
            return k
    raise KeyError(f"{context}: none of keys found: {candidates}")


def load_sparse_npz_any(npz_path: Path):
    try:
        return sp.load_npz(npz_path).tocsr()
    except Exception:
        pass

    z = np.load(npz_path, allow_pickle=True)
    need = {"data", "indices", "indptr", "shape"}
    if need.issubset(set(z.files)):
        shape = tuple(z["shape"].tolist())
        return sp.csr_matrix(
            (z["data"], z["indices"], z["indptr"]),
            shape=(int(shape[0]), int(shape[1])),
        )
    raise ValueError(f"Cannot parse sparse npz: {npz_path}")


def sparse_row_log_norm(X: sp.csr_matrix, scale: float = 1e4) -> sp.csr_matrix:
    X = X.tocsr(copy=True)
    rs = np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float32)
    rs = np.maximum(rs, 1.0)
    inv = scale / rs
    X = sp.diags(inv) @ X
    if X.nnz > 0:
        X.data = np.log1p(np.clip(X.data, 0.0, None))
    return X.tocsr()


def dense_row_log_norm(X: np.ndarray, scale: float = 1e4) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    rs = X.sum(axis=1, keepdims=True).astype(np.float32)
    rs = np.maximum(rs, 1.0)
    Xn = X / rs * float(scale)
    Xn = np.log1p(np.clip(Xn, 0.0, None))
    return Xn.astype(np.float32, copy=False)


def var_topk_dense(X: np.ndarray, topk: int, min_cell_frac: float = 0.0) -> np.ndarray:
    nnz_frac = (X > 0).mean(axis=0)
    var = X.var(axis=0)
    keep = nnz_frac >= float(min_cell_frac)
    if keep.sum() == 0:
        return np.arange(min(topk, X.shape[1]), dtype=np.int64)
    idx0 = np.where(keep)[0]
    var0 = var[idx0]
    k = min(topk, idx0.size)
    pick = np.argpartition(var0, -k)[-k:]
    out = idx0[pick]
    out = out[np.argsort(var0[pick])[::-1]]
    return out.astype(np.int64)


def var_topk_sparse(X: sp.csr_matrix, topk: int, min_cell_frac: float = 0.0) -> np.ndarray:
    mean = np.asarray(X.mean(axis=0)).reshape(-1).astype(np.float64)
    mean_sq = np.asarray(X.power(2).mean(axis=0)).reshape(-1).astype(np.float64)
    var = np.clip(mean_sq - mean ** 2, 0.0, None)
    nnz_frac = np.asarray((X > 0).mean(axis=0)).reshape(-1)
    keep = nnz_frac >= float(min_cell_frac)
    if keep.sum() == 0:
        return np.arange(min(topk, X.shape[1]), dtype=np.int64)
    idx0 = np.where(keep)[0]
    var0 = var[idx0]
    k = min(topk, idx0.size)
    pick = np.argpartition(var0, -k)[-k:]
    out = idx0[pick]
    out = out[np.argsort(var0[pick])[::-1]]
    return out.astype(np.int64)


# ============================================================
# 2) DATA OBJECTS
# ============================================================
@dataclass
class RnaSample:
    sid: str
    barcodes: np.ndarray
    X: np.ndarray
    gene_names: np.ndarray


@dataclass
class AtacSample:
    sid: str
    barcodes: np.ndarray
    X: sp.csr_matrix
    peak_names: np.ndarray


# ============================================================
# 3) LOAD RNA
# ============================================================
def load_rna_manifest(rna_manifest: Path):
    assert rna_manifest.exists(), f"Missing RNA manifest: {rna_manifest}"
    df = pd.read_csv(rna_manifest, sep="\t")
    for c in ["cell_id", "sample_id", "row_index"]:
        assert c in df.columns, f"RNA manifest missing column: {c}"
    out = {}
    for sid, g in df.groupby("sample_id", sort=True):
        out[str(sid)] = g.sort_values("row_index").reset_index(drop=True)
    return out


def load_rna_sample(sid: str, g: pd.DataFrame) -> RnaSample:
    meta_path = RNA_ROOT / f"{sid}.rna_meta.npz"
    counts_path = RNA_ROOT / f"{sid}.X_rna_counts_dense.npy"

    assert meta_path.exists(), f"Missing RNA meta: {meta_path}"
    assert counts_path.exists(), f"Missing RNA counts: {counts_path}"

    z = np.load(meta_path, allow_pickle=True)
    gene_key = pick_first_key(z, ["gene_names", "genes", "var_names"], f"RNA meta {meta_path.name}")
    gene_names = np.asarray(z[gene_key]).astype(str)

    X = np.load(counts_path, mmap_mode="r")
    X = np.asarray(X, dtype=np.float32)
    assert X.ndim == 2, f"RNA counts must be 2D: {counts_path}"
    assert X.shape[0] == len(g), f"RNA rows mismatch for {sid}: {X.shape[0]} vs {len(g)}"
    assert X.shape[1] == len(gene_names), f"RNA cols mismatch for {sid}"

    barcodes = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
    return RnaSample(sid=sid, barcodes=barcodes, X=X, gene_names=gene_names)


# ============================================================
# 4) LOAD ATAC
# ============================================================
def infer_atac_sample_ids_from_files():
    meta_files = sorted(ATAC_ROOT.glob("*.atac_meta.npz"))
    out = []
    for p in meta_files:
        sid = p.name.replace(".atac_meta.npz", "")
        x_path = ATAC_ROOT / f"{sid}.X_atac_peaks_csr.npz"
        if x_path.exists():
            out.append(sid)
    return sorted(out)


def load_atac_manifest(atac_manifest: Path):
    if not atac_manifest.exists():
        log(f"[ATAC][WARN] manifest not found, fallback to file scan: {atac_manifest}")
        return {"mode": "file_scan", "sample_ids": infer_atac_sample_ids_from_files()}

    df = pd.read_csv(atac_manifest, sep="\t")
    cols = set(df.columns.tolist())

    if {"sample_id", "cell_id", "row_index"}.issubset(cols):
        by_sample = {}
        for sid, g in df.groupby("sample_id", sort=True):
            by_sample[str(sid)] = g.sort_values("row_index").reset_index(drop=True)
        return {"mode": "per_cell", "table": by_sample}

    if "sample_id" in cols:
        by_sample = {}
        for _, row in df.iterrows():
            by_sample[str(row["sample_id"])] = row
        return {"mode": "per_sample", "table": by_sample}

    log("[ATAC][WARN] manifest format not recognized, fallback to file scan")
    return {"mode": "file_scan", "sample_ids": infer_atac_sample_ids_from_files()}


def load_atac_sample_from_files(sid: str, barcodes_hint: np.ndarray | None = None) -> AtacSample:
    meta_path = ATAC_ROOT / f"{sid}.atac_meta.npz"
    x_path = ATAC_ROOT / f"{sid}.X_atac_peaks_csr.npz"

    assert meta_path.exists(), f"Missing ATAC meta: {meta_path}"
    assert x_path.exists(), f"Missing ATAC matrix: {x_path}"

    z = np.load(meta_path, allow_pickle=True)

    peak_key = pick_first_key(
        z,
        ["peak_names", "peaks", "var_names", "feature_names"],
        f"ATAC meta {meta_path.name}",
    )
    peak_names = np.asarray(z[peak_key]).astype(str)

    barcode_key = None
    for k in ["barcodes", "cells", "cell_ids", "obs_names"]:
        if k in z:
            barcode_key = k
            break

    if barcode_key is not None:
        barcodes = np.asarray([norm_barcode(x) for x in z[barcode_key].tolist()], dtype=object)
    else:
        assert barcodes_hint is not None, f"ATAC meta has no barcode field and no hint provided: {meta_path}"
        barcodes = np.asarray([norm_barcode(x) for x in barcodes_hint.tolist()], dtype=object)

    X = load_sparse_npz_any(x_path)
    assert X.shape[0] == len(barcodes), f"ATAC rows mismatch for {sid}: {X.shape[0]} vs {len(barcodes)}"
    assert X.shape[1] == len(peak_names), f"ATAC cols mismatch for {sid}: {X.shape[1]} vs {len(peak_names)}"
    return AtacSample(sid=sid, barcodes=barcodes, X=X.tocsr(), peak_names=peak_names)


def load_atac_sample(file_sid: str, atac_manifest_info: dict, rna_barcodes_hint: np.ndarray | None = None, manifest_sid: str | None = None) -> AtacSample:
    mode = atac_manifest_info["mode"]
    manifest_sid = file_sid if manifest_sid is None else manifest_sid

    if mode == "per_cell":
        g = atac_manifest_info["table"].get(manifest_sid, None)
        if g is None:
            raise KeyError(f"ATAC manifest missing sample_id={manifest_sid}")
        barcodes_hint = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
        return load_atac_sample_from_files(file_sid, barcodes_hint=barcodes_hint)

    if mode == "per_sample":
        row = atac_manifest_info["table"].get(manifest_sid, None)
        if row is None:
            raise KeyError(f"ATAC manifest missing sample_id={manifest_sid}")
        return load_atac_sample_from_files(file_sid, barcodes_hint=rna_barcodes_hint)

    if mode == "file_scan":
        if file_sid not in set(atac_manifest_info["sample_ids"]):
            raise KeyError(f"ATAC files missing sample_id={file_sid}")
        return load_atac_sample_from_files(file_sid, barcodes_hint=rna_barcodes_hint)

    raise ValueError(f"Unknown ATAC manifest mode: {mode}")


# ============================================================
# 5) BUILD PAIRED MATRICES
# ============================================================
def build_paired_data():
    rna_manifest = load_rna_manifest(RNA_MANIFEST)
    atac_manifest_info = load_atac_manifest(ATAC_MANIFEST)

    if atac_manifest_info["mode"] in {"per_cell", "per_sample"}:
        atac_sids = sorted(atac_manifest_info["table"].keys())
    else:
        atac_sids = sorted(atac_manifest_info["sample_ids"])

    rna_sids = sorted(rna_manifest.keys())

    debug_print_sid_overlap(rna_sids, atac_sids, max_show=12)

    matched_pairs, rna2atac, atac2rna = build_sid_match_map(rna_sids, atac_sids)
    assert len(matched_pairs) > 0, "No matched RNA/ATAC sample ids found, even after canonical mapping"

    log(f"[SID] matched sample pairs n={len(matched_pairs)}")
    for rs, ats in matched_pairs[:12]:
        if rs == ats:
            log(f"[SID] match exact: {rs}")
        else:
            log(f"[SID] match mapped: RNA={rs} -> ATAC={ats}")

    rna_blocks = []
    atac_blocks = []
    obs_rows = []

    gene_ref = None
    peak_ref = None
    total_paired = 0

    for rna_sid, atac_sid in matched_pairs:
        rna = load_rna_sample(rna_sid, rna_manifest[rna_sid])
        atac = load_atac_sample(atac_sid, atac_manifest_info, rna_barcodes_hint=rna.barcodes)

        if gene_ref is None:
            gene_ref = rna.gene_names
        else:
            assert np.array_equal(gene_ref.astype(str), rna.gene_names.astype(str)), f"RNA genes differ at {rna_sid}"

        if peak_ref is None:
            peak_ref = atac.peak_names
        else:
            assert np.array_equal(peak_ref.astype(str), atac.peak_names.astype(str)), f"ATAC peaks differ at {atac_sid}"

        rna_map = {bc: i for i, bc in enumerate(rna.barcodes.tolist())}
        atac_map = {bc: i for i, bc in enumerate(atac.barcodes.tolist())}
        shared = sorted(set(rna_map.keys()) & set(atac_map.keys()))

        if MAX_CELLS_PER_SAMPLE is not None and len(shared) > int(MAX_CELLS_PER_SAMPLE):
            rng = np.random.default_rng(SEED)
            shared = sorted(
                rng.choice(np.asarray(shared, dtype=object),
                           size=int(MAX_CELLS_PER_SAMPLE),
                           replace=False).tolist()
            )

        if len(shared) == 0:
            log(f"[PAIR][WARN] no shared cells | RNA={rna_sid} ATAC={atac_sid}")
            continue

        r_idx = np.asarray([rna_map[bc] for bc in shared], dtype=np.int64)
        a_idx = np.asarray([atac_map[bc] for bc in shared], dtype=np.int64)

        Xr = rna.X[r_idx, :]
        Xa = atac.X[a_idx, :].tocsr()

        rna_blocks.append(np.asarray(Xr, dtype=np.float32))
        atac_blocks.append(Xa)

        for bc in shared:
            obs_rows.append({
                "sample_id": rna_sid,
                "rna_sample_id": rna_sid,
                "atac_sample_id": atac_sid,
                "barcode": bc,
                "cell_key": make_cell_key(rna_sid, bc),
            })

        total_paired += len(shared)
        log(
            f"[PAIR] RNA={rna_sid} | ATAC={atac_sid} | "
            f"paired_cells={len(shared)} RNA={Xr.shape} ATAC={Xa.shape}"
        )

    assert len(rna_blocks) > 0, "Matched sample ids exist, but no shared barcodes found between RNA and ATAC"

    X_rna = np.concatenate(rna_blocks, axis=0).astype(np.float32)
    X_atac = sp.vstack(atac_blocks, format="csr")
    obs = pd.DataFrame(obs_rows)

    keys = obs["cell_key"].astype(str).tolist()
    is_test = np.asarray([split_5050(k) == 1 for k in keys], dtype=bool)
    is_train_full = ~is_test
    is_train, is_val = split_train_val_from_train_mask(is_train_full, VAL_FRAC_FROM_TRAIN, seed=SEED)

    split = np.full((obs.shape[0],), "test", dtype=object)
    split[is_train] = "train"
    split[is_val] = "val"
    obs["split"] = split

    log(f"[TOTAL] paired_cells={total_paired}")
    log(f"[SHAPE] RNA={X_rna.shape} ATAC={X_atac.shape}")
    log(f"[SPLIT] {pd.Series(split).value_counts().to_dict()}")

    return X_rna, X_atac, gene_ref.astype(str), peak_ref.astype(str), obs


# ============================================================
# 6) FEATURE FILTER + NORMALIZE
# ============================================================
def preprocess_modalities(X_rna_raw: np.ndarray, X_atac_raw: sp.csr_matrix, obs: pd.DataFrame, gene_names: np.ndarray, peak_names: np.ndarray):
    train_mask = obs["split"].values == "train"

    Xr = dense_row_log_norm(X_rna_raw, scale=RNA_LOG_NORM_SCALE)
    Xa = sparse_row_log_norm(X_atac_raw, scale=ATAC_LOG_NORM_SCALE)

    rna_idx = var_topk_dense(Xr[train_mask], topk=RNA_HVG, min_cell_frac=MIN_RNA_CELL_FRAC)
    atac_idx = var_topk_sparse(Xa[train_mask], topk=ATAC_HVP, min_cell_frac=MIN_ATAC_CELL_FRAC)

    Xr = Xr[:, rna_idx].astype(np.float32, copy=False)
    Xa = Xa[:, atac_idx].tocsr()

    gene_sel = gene_names[rna_idx]
    peak_sel = peak_names[atac_idx]

    log(f"[FILTER] RNA {len(gene_names)} -> {len(gene_sel)}")
    log(f"[FILTER] ATAC {len(peak_names)} -> {len(peak_sel)}")

    return Xr, Xa, gene_sel, peak_sel, rna_idx, atac_idx


# ============================================================
# 7) DATASET
# ============================================================
class PairedMultiomeDataset(Dataset):
    def __init__(self, X_rna: np.ndarray, X_atac: sp.csr_matrix, indices: np.ndarray):
        self.X_rna = X_rna
        self.X_atac = X_atac.tocsr()
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.indices.size

    def __getitem__(self, i):
        idx = int(self.indices[i])
        xr = self.X_rna[idx].astype(np.float32, copy=False)
        xa = self.X_atac[idx].toarray().reshape(-1).astype(np.float32, copy=False)
        return {
            "idx": idx,
            "rna": torch.from_numpy(xr),
            "atac": torch.from_numpy(xa),
        }


# ============================================================
# 8) MODEL
# ============================================================
class SharedLatentMMVAE(nn.Module):
    def __init__(self, n_rna: int, n_atac: int, z_dim: int = 32, hid: int = 512, drop: float = 0.1):
        super().__init__()
        self.n_rna = int(n_rna)
        self.n_atac = int(n_atac)
        self.z_dim = int(z_dim)

        in_dim = self.n_rna + self.n_atac

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.enc_mu = nn.Linear(hid, z_dim)
        self.enc_logvar = nn.Linear(hid, z_dim)

        self.dec_rna = nn.Sequential(
            nn.Linear(z_dim, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, self.n_rna),
        )
        self.dec_atac = nn.Sequential(
            nn.Linear(z_dim, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, self.n_atac),
        )

    def encode(self, x_rna: torch.Tensor, x_atac: torch.Tensor):
        h = self.encoder(torch.cat([x_rna, x_atac], dim=-1))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(min=-12.0, max=12.0)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        xr_hat = self.dec_rna(z)
        xa_hat = self.dec_atac(z)
        return xr_hat, xa_hat

    def forward(self, x_rna: torch.Tensor, x_atac: torch.Tensor):
        mu, logvar = self.encode(x_rna, x_atac)
        z = self.reparam(mu, logvar)
        xr_hat, xa_hat = self.decode(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "rna_hat": xr_hat,
            "atac_hat": xa_hat,
        }


def vae_loss(xr, xa, out, beta_kl: float):
    recon_rna = F.mse_loss(out["rna_hat"], xr, reduction="mean")
    recon_atac = F.mse_loss(out["atac_hat"], xa, reduction="mean")

    mu = out["mu"]
    logvar = out["logvar"]
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_rna + recon_atac + float(beta_kl) * kl
    return total, {
        "total": float(total.detach().cpu()),
        "recon_rna": float(recon_rna.detach().cpu()),
        "recon_atac": float(recon_atac.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }


# ============================================================
# 9) TRAIN / EVAL
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    loss_sum = 0.0
    rna_sum = 0.0
    atac_sum = 0.0
    kl_sum = 0.0
    n = 0

    for batch in loader:
        xr = batch["rna"].to(DEVICE)
        xa = batch["atac"].to(DEVICE)
        out = model(xr, xa)
        loss, metrics = vae_loss(xr, xa, out, beta_kl=BETA_KL)
        bs = xr.shape[0]
        n += bs
        loss_sum += metrics["total"] * bs
        rna_sum += metrics["recon_rna"] * bs
        atac_sum += metrics["recon_atac"] * bs
        kl_sum += metrics["kl"] * bs

    return {
        "loss": loss_sum / max(n, 1),
        "recon_rna": rna_sum / max(n, 1),
        "recon_atac": atac_sum / max(n, 1),
        "kl": kl_sum / max(n, 1),
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = math.inf
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        rna_sum = 0.0
        atac_sum = 0.0
        kl_sum = 0.0
        n = 0

        for batch in train_loader:
            xr = batch["rna"].to(DEVICE)
            xa = batch["atac"].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            out = model(xr, xa)
            loss, metrics = vae_loss(xr, xa, out, beta_kl=BETA_KL)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            bs = xr.shape[0]
            n += bs
            loss_sum += metrics["total"] * bs
            rna_sum += metrics["recon_rna"] * bs
            atac_sum += metrics["recon_atac"] * bs
            kl_sum += metrics["kl"] * bs

        tr = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n, 1),
            "train_recon_rna": rna_sum / max(n, 1),
            "train_recon_atac": atac_sum / max(n, 1),
            "train_kl": kl_sum / max(n, 1),
        }

        if val_loader is not None:
            va = evaluate(model, val_loader)
            tr.update({
                "val_loss": va["loss"],
                "val_recon_rna": va["recon_rna"],
                "val_recon_atac": va["recon_atac"],
                "val_kl": va["kl"],
            })

            if SAVE_BEST_BY_VAL and va["loss"] < best_val:
                best_val = va["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if SAVE_BEST_BY_VAL and best_state is None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(tr)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            if val_loader is not None:
                log(
                    f"[EPOCH {epoch:03d}] "
                    f"train={tr['train_loss']:.4f} "
                    f"(rna={tr['train_recon_rna']:.4f}, atac={tr['train_recon_atac']:.4f}, kl={tr['train_kl']:.4f}) | "
                    f"val={tr['val_loss']:.4f} "
                    f"(rna={tr['val_recon_rna']:.4f}, atac={tr['val_recon_atac']:.4f}, kl={tr['val_kl']:.4f})"
                )
            else:
                log(
                    f"[EPOCH {epoch:03d}] "
                    f"train={tr['train_loss']:.4f} "
                    f"(rna={tr['train_recon_rna']:.4f}, atac={tr['train_recon_atac']:.4f}, kl={tr['train_kl']:.4f})"
                )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return history


# ============================================================
# 10) LATENT EXPORT
# ============================================================
@torch.no_grad()
def export_latent(model: nn.Module, X_rna: np.ndarray, X_atac: sp.csr_matrix, batch_size: int = 256):
    model.eval()
    n = X_rna.shape[0]
    zs = []
    mus = []

    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        xr = torch.from_numpy(X_rna[st:ed].astype(np.float32, copy=False)).to(DEVICE)
        xa = torch.from_numpy(X_atac[st:ed].toarray().astype(np.float32, copy=False)).to(DEVICE)
        mu, logvar = model.encode(xr, xa)
        z = mu
        zs.append(z.detach().cpu().numpy().astype(np.float32))
        mus.append(mu.detach().cpu().numpy().astype(np.float32))

    Z = np.concatenate(zs, axis=0)
    MU = np.concatenate(mus, axis=0)
    return Z, MU


# ============================================================
# 11) MAIN
# ============================================================
def main():
    log(f"[DEVICE] {DEVICE}")
    log("[RUN] loading paired data ...")
    X_rna_raw, X_atac_raw, gene_names, peak_names, obs = build_paired_data()

    log("[RUN] preprocessing ...")
    X_rna, X_atac, gene_sel, peak_sel, rna_idx, atac_idx = preprocess_modalities(
        X_rna_raw, X_atac_raw, obs, gene_names, peak_names
    )

    train_idx = np.where(obs["split"].values == "train")[0].astype(np.int64)
    val_idx = np.where(obs["split"].values == "val")[0].astype(np.int64)
    test_idx = np.where(obs["split"].values == "test")[0].astype(np.int64)

    ds_train = PairedMultiomeDataset(X_rna, X_atac, train_idx)
    ds_val = PairedMultiomeDataset(X_rna, X_atac, val_idx) if val_idx.size > 0 else None

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False) if ds_val is not None else None

    log("[RUN] building model ...")
    model = SharedLatentMMVAE(
        n_rna=X_rna.shape[1],
        n_atac=X_atac.shape[1],
        z_dim=LATENT_DIM,
        hid=ENC_HID,
        drop=DROP,
    ).to(DEVICE)

    log("[RUN] training ...")
    history = train_model(model, dl_train, dl_val)

    log("[RUN] exporting latent ...")
    Z, MU = export_latent(model, X_rna, X_atac, batch_size=BATCH_SIZE)

    latent_dtype = np.float16 if EXPORT_LATENT_FLOAT16 else np.float32
    np.save(OUT_DIR / "latent.npy", Z.astype(latent_dtype))
    np.save(OUT_DIR / "latent_mu.npy", MU.astype(latent_dtype))

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "LATENT_DIM": LATENT_DIM,
            "ENC_HID": ENC_HID,
            "DEC_HID": DEC_HID,
            "DROP": DROP,
            "RNA_HVG": RNA_HVG,
            "ATAC_HVP": ATAC_HVP,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "BETA_KL": BETA_KL,
            "SEED": SEED,
        },
        "feature_index": {
            "rna_idx": rna_idx,
            "atac_idx": atac_idx,
            "gene_sel": gene_sel,
            "peak_sel": peak_sel,
        }
    }, OUT_DIR / "model.pt")

    obs_out = obs.copy()
    obs_out["latent_row"] = np.arange(obs_out.shape[0], dtype=np.int64)
    obs_out.to_csv(OUT_DIR / "obs.csv", index=False)

    pd.DataFrame(history).to_csv(OUT_DIR / "train_history.csv", index=False)
    pd.DataFrame({"gene": gene_sel}).to_csv(OUT_DIR / "rna_features.csv", index=False)
    pd.DataFrame({"peak": peak_sel}).to_csv(OUT_DIR / "atac_features.csv", index=False)

    summary = {
        "dataset": DATASET_NAME,
        "device": DEVICE,
        "n_cells": int(obs.shape[0]),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "rna_shape_after_filter": [int(X_rna.shape[0]), int(X_rna.shape[1])],
        "atac_shape_after_filter": [int(X_atac.shape[0]), int(X_atac.shape[1])],
        "latent_dim": int(LATENT_DIM),
        "epochs": int(EPOCHS),
        "seed": int(SEED),
        "splits": obs["split"].value_counts().to_dict(),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"[SAVE] latent.npy -> {OUT_DIR / 'latent.npy'}")
    log(f"[SAVE] latent_mu.npy -> {OUT_DIR / 'latent_mu.npy'}")
    log(f"[SAVE] obs.csv -> {OUT_DIR / 'obs.csv'}")
    log(f"[SAVE] model.pt -> {OUT_DIR / 'model.pt'}")
    log(f"[SAVE] summary.json -> {OUT_DIR / 'summary.json'}")
    log("[DONE]")


if __name__ == "__main__":
    main()
