from __future__ import annotations

# ============================================================
# crc_sample9_cluster_communication_pseudotime.py
# ------------------------------------------------------------
# 目标：
# - 基于 new Leiden cluster-level CCI 网络
# - 构建 communication pseudotime
# - 核心分析：sender -> receiver -> TF response
# - pathway 不进入核心建模，只保留后续解释接口
#
# 输入依赖（来自你现有流程）：
# 1) interactions_cluster_level_highconf_TFkept.csv
# 2) cluster_profiles.csv
# 3) cluster_sizes.csv
# 4) cell_level_mapping_with_umap_and_internal16.csv
# 5) clusters.csv (仅为 group adaptor 使用)
# 6) projected RNA + base/frontier checkpoint
#
# 输出：
# 1) cluster_dynamic_features.csv
# 2) cluster_pseudotime.csv
# 3) edge_pseudotime_summary.csv
# 4) cell_projected_pseudotime.csv
# 5) qc_*.png / .svg
#
# 核心思想：
# - 节点 = new Leiden cluster
# - 边 = cluster->cluster CCI (TF-kept/highconf)
# - 节点特征：
#     sender_strength
#     receiver_strength
#     net_flow
#     io_ratio
#     receiver_tf_response
#     cluster centroid latent Z
# - teacher:
#     + receiver_strength
#     + receiver_tf_response
#     - sender_strength
# - GNN + LTC 风格 rollout 学习平滑、连续的 communication pseudotime
# ============================================================

import os
import re
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt


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

RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)

FRONTIER_CKPT = RUN_STAGEB_DIR / "frontier_best.pt"
BASE_CKPT = get_dataset_checkpoint("CRC-Tissue")

CCI_DIR = RUN_STAGEB_DIR / f"{SAMPLE_ID}_cci_newLeiden"
RECLUSTER_DIR = RUN_STAGEB_DIR / f"single_sample_{SAMPLE_ID}_recluster_for_cci"

INT_TFKEEP_CSV = CCI_DIR / "interactions_cluster_level_highconf_TFkept.csv"
INT_HIGHCONF_CSV = CCI_DIR / "interactions_cluster_level_highconf.csv"
INT_ALL_CSV = CCI_DIR / "interactions_cluster_level.csv"
CLUSTER_PROF_CSV = CCI_DIR / "cluster_profiles.csv"
CLUSTER_SIZE_CSV = CCI_DIR / "cluster_sizes.csv"
CELL_MAP_CSV = RECLUSTER_DIR / f"{SAMPLE_ID}_cell_level_mapping_with_umap_and_internal16.csv"
OLD_CLUSTER_CSV = RUN_STAGEB_DIR / "clusters.csv"

RNA_DIR = WORLD_ROOT / DATASET_NAME / "processed" / "RNA"
PROJECTED_RNA = PROJECTED_ROOT / DATASET_NAME / f"{SAMPLE_ID}.rna.global.csr.npz"

OUT_DIR = CCI_DIR / "communication_pseudotime_clusterGNN_LTC"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- runtime ----------
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

# ---------- graph ----------
USE_DIRECTED_CCI_EDGE = True
USE_SIMILARITY_GRAPH = True
SIM_K = 3                      # cluster centroid KNN
EDGE_SCORE_COL_CANDIDATES = ["score", "expr_only", "confidence"]
EDGE_AGG = "sum"               # sum / mean / max
MIN_EDGE_SCORE = 0.0

# ---------- node feature weights for weak teacher ----------
W_RECEIVER = 1.00
W_TFRESP = 0.80
W_SENDER = 0.70
W_NET = 0.25
W_RATIO = 0.15

# ---------- model ----------
H = 128
ROLLOUT_STEPS = 12
DT = 0.20
TAU_MIN = 0.15

EPOCHS = 300
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

W_TEACH = 1.0
W_RANK = 0.7
W_SMOOTH = 0.35
W_DIR_EDGE = 0.55
W_RECON = 0.20

N_ENSEMBLE = 5
ENSEMBLE_SEEDS = list(range(N_ENSEMBLE))

# ---------- training samples ----------
RANK_PAIRS = 8000
SMOOTH_EDGES = 20000
DIR_EDGES = 20000

# ---------- output ----------
DPI = 350
N_BINS = 12

# optional cell projection smoothing
CELL_KNN = 25


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


def make_cell_key(sid: str, barcode: str) -> str:
    return f"{sid}::{barcode}"


def dataset_to_cancer_type(ds_name: str) -> str:
    s = str(ds_name)
    return s.split("-", 1)[0] if "-" in s else s


def get_default_cancer_name(dataset_name: str) -> str | None:
    return dataset_to_cancer_type(dataset_name)


def normalize_cluster_str(x: str) -> str:
    s = str(x).strip()
    if s.startswith("C") and s[1:].isdigit():
        return s[1:]
    return s


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if (not np.isfinite(s)) or s < 1e-6:
        return np.zeros_like(x, np.float32)
    return (x - m) / (s + 1e-6)


def soft_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    order = np.argsort(x)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    r = r / max(len(x) - 1, 1)
    return r.astype(np.float32)


def winsorize(x: np.ndarray, qlo=0.01, qhi=0.99) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return x
    lo = np.quantile(x, qlo)
    hi = np.quantile(x, qhi)
    return np.clip(x, lo, hi)


def bin_by_quantile(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    b = np.digitize(x, edges[1:-1], right=False)
    return b.astype(np.int64)


def safe_name(x: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x))[:n]


def parse_json_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    try:
        return json.loads(str(x))
    except Exception:
        return []


def find_first_existing_col(df: pd.DataFrame, cand_cols: list[str]) -> str | None:
    for c in cand_cols:
        if c in df.columns:
            return c
    return None


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


# ============================================================
# 2) BASE MODEL / ADAPTER / STORE
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

    def delta_cancer(self, cancer_id: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        return self.C_cancer(cancer_id) @ V_cancer

    def apply_group_to_Z(self, Z: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        s = self.S_group(group_id)
        b = self.B_group(group_id)
        return Z * (1.0 + 0.12 * torch.tanh(s)) + b


class CSRNPZStore:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.data = None
        self.indices = None
        self.indptr = None

    def open(self):
        if self.data is None:
            z = np.load(self.npz_path, mmap_mode="r")
            self.data = z["data"]
            self.indices = z["indices"]
            self.indptr = z["indptr"]

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


# ============================================================
# 3) CLUSTER GRAPH MODEL
# ============================================================
class ClusterCommLTC(nn.Module):
    def __init__(self, xdim: int, hdim: int, out_dim: int):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, hdim),
        )

        self.edge_msg = nn.Linear(hdim, hdim, bias=False)

        # edge_feat:
        # [edge_weight, sim_edge, src_sender, tgt_receiver, src_tf, tgt_tf]
        self.att = nn.Sequential(
            nn.Linear(hdim * 2 + 6, hdim),
            nn.GELU(),
            nn.Linear(hdim, 1),
        )

        self.core = nn.Sequential(
            nn.Linear(hdim + xdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, hdim),
        )
        self.tau_head = nn.Sequential(
            nn.Linear(hdim + xdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, 1),
        )
        self.t_head = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, 1),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, out_dim),
        )

    def forward(self, x, nbr_idx, nbr_mask, edge_feat):
        # x: [N, D]
        # nbr_idx: [N, K]
        # nbr_mask: [N, K]
        # edge_feat: [N, K, 6]
        N, K = nbr_idx.shape
        h = self.inp(x)
        tau_out = None

        for _ in range(ROLLOUT_STEPS):
            hn = h[nbr_idx]                              # [N,K,H]
            hi = h.unsqueeze(1).expand(-1, K, -1)       # [N,K,H]

            a_in = torch.cat([hi, hn, edge_feat], dim=-1)
            logits = self.att(a_in).squeeze(-1)         # [N,K]
            logits = logits.masked_fill(~nbr_mask, -1e9)
            att = torch.softmax(logits, dim=1)
            att = att * nbr_mask.float()
            att = att / (att.sum(dim=1, keepdim=True) + 1e-8)

            m = torch.sum(att[..., None] * self.edge_msg(hn), dim=1)
            cat = torch.cat([h + m, x], dim=-1)

            drive = torch.tanh(self.core(cat))
            tau_out = F.softplus(self.tau_head(cat)) + TAU_MIN
            h = h + DT * ((-h + drive) / tau_out)

        t_hat = torch.sigmoid(self.t_head(h)).squeeze(-1)
        recon = self.recon_head(h)
        tau = tau_out.squeeze(-1)
        return t_hat, recon, tau


# ============================================================
# 4) LOAD SAMPLE / ENCODE CELL Z
# ============================================================
@dataclass
class Sample:
    sid: str
    cells: np.ndarray
    gene_names: np.ndarray
    bc2rid: dict[str, int]
    counts_path: Path
    proj_rna_path: Path


def load_single_sample(dataset_name: str, sample_id: str) -> Sample:
    manifest_path = RNA_DIR / "manifest_rna.tsv"
    assert manifest_path.exists()

    df = pd.read_csv(manifest_path, sep="\t")
    g = df[df["sample_id"].astype(str) == str(sample_id)].copy().sort_values("row_index").reset_index(drop=True)
    assert len(g) > 0

    meta_path = RNA_DIR / f"{sample_id}.rna_meta.npz"
    counts_path = RNA_DIR / f"{sample_id}.X_rna_counts_dense.npy"
    proj_rna_path = PROJECTED_ROOT / dataset_name / f"{sample_id}.rna.global.csr.npz"

    assert meta_path.exists()
    assert counts_path.exists()
    assert proj_rna_path.exists()

    z = np.load(meta_path, allow_pickle=True)
    gene_names = np.asarray(z["gene_names"]).astype(str)
    cells = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
    bc2rid = {str(bc): i for i, bc in enumerate(cells.tolist())}

    return Sample(
        sid=sample_id,
        cells=cells,
        gene_names=gene_names,
        bc2rid=bc2rid,
        counts_path=counts_path,
        proj_rna_path=proj_rna_path,
    )


@torch.no_grad()
def encode_cells_Z(
    model,
    adapter,
    sample: Sample,
    cell_df: pd.DataFrame,
    default_cancer_id: int,
):
    proj_store = CSRNPZStore(sample.proj_rna_path)
    proj_store.open()

    cell_list = []
    for r in cell_df.itertuples(index=False):
        rid = sample.bc2rid.get(str(r.barcode), None)
        if rid is None:
            continue
        cell_list.append({
            "barcode": str(r.barcode),
            "rid": int(rid),
            "cluster_norm": str(r.cluster_norm),
            "group_id": int(r.group_id),
        })

    Zs = []
    kept_rows = []
    for st in range(0, len(cell_list), EMB_BATCH):
        batch = cell_list[st:st + EMB_BATCH]
        idx_list = []
        val_list = []
        offsets = [0]
        gid = []

        for rec in batch:
            gi, gv = proj_store.get_row_topk(int(rec["rid"]), RNA_TOPK)
            idx_list.append(gi.astype(np.int64))
            val_list.append(gv.astype(np.float32))
            offsets.append(offsets[-1] + gi.size)
            gid.append(int(rec["group_id"]))

        idx_cat = np.concatenate(idx_list) if idx_list else np.zeros((0,), np.int64)
        val_cat = np.concatenate(val_list) if val_list else np.zeros((0,), np.float32)
        offsets_np = np.asarray(offsets[:-1], np.int64)
        cancer_ids = np.full((len(batch),), int(default_cancer_id), np.int64)

        z = model.encode_rna(
            torch.from_numpy(idx_cat).to(DEVICE),
            torch.from_numpy(val_cat).to(DEVICE),
            torch.from_numpy(offsets_np).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )

        gid_t = torch.from_numpy(np.asarray(gid, np.int64)).to(DEVICE)
        _, _, _, V_cancer = adapter()
        z = adapter.apply_group_to_Z(z, gid_t)

        cid_t = torch.full((z.shape[0],), int(default_cancer_id), device=DEVICE, dtype=torch.long)
        z = z + adapter.delta_cancer(cid_t, V_cancer)
        z = F.normalize(z, dim=-1)

        Zs.append(z.detach().cpu().numpy().astype(np.float32))
        kept_rows.extend(batch)

    Z = np.concatenate(Zs, axis=0).astype(np.float32)
    kept_df = pd.DataFrame(kept_rows)
    return kept_df, Z


# ============================================================
# 5) BUILD CLUSTER FEATURES FROM CCI
# ============================================================
def load_cluster_profiles_tf_response(cluster_prof_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cluster_prof_csv)
    assert "cluster_label" in df.columns

    rows = []
    for r in df.itertuples(index=False):
        cl = str(r.cluster_label)
        fallback = float(getattr(r, "fallback_tf_resp_z", 0.0))

        top_tfs = parse_json_list(getattr(r, "top_tfs", "[]"))
        vals = []
        for item in top_tfs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    vals.append(float(item[1]))
                except Exception:
                    pass
            elif isinstance(item, dict) and "tf_z" in item:
                try:
                    vals.append(float(item["tf_z"]))
                except Exception:
                    pass

        tf_top_mean = float(np.mean(vals[:5])) if len(vals) > 0 else float(fallback)

        rows.append({
            "cluster_norm": normalize_cluster_str(cl),
            "fallback_tf_resp_z": float(fallback),
            "tf_top_mean": tf_top_mean,
        })
    out = pd.DataFrame(rows).drop_duplicates("cluster_norm")
    out["receiver_tf_response"] = zscore_1d(out["tf_top_mean"].values)
    return out[["cluster_norm", "fallback_tf_resp_z", "tf_top_mean", "receiver_tf_response"]]


def aggregate_cluster_cci(int_csv: Path, cluster_size_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(int_csv)
    assert "sender_cluster" in df.columns
    assert "receiver_cluster" in df.columns

    score_col = find_first_existing_col(df, EDGE_SCORE_COL_CANDIDATES)
    assert score_col is not None, f"Cannot find score column in {int_csv}"

    df["sender_cluster"] = df["sender_cluster"].astype(str).map(normalize_cluster_str)
    df["receiver_cluster"] = df["receiver_cluster"].astype(str).map(normalize_cluster_str)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0)
    else:
        df["confidence"] = 1.0

    if "kept_by_tf" in df.columns:
        df["kept_by_tf"] = pd.to_numeric(df["kept_by_tf"], errors="coerce").fillna(1).astype(int)
    else:
        df["kept_by_tf"] = 1

    # weighted score
    df["edge_score"] = df[score_col].astype(np.float32) * df["confidence"].astype(np.float32)

    if MIN_EDGE_SCORE > 0:
        df = df[df["edge_score"] >= float(MIN_EDGE_SCORE)].copy()

    if EDGE_AGG == "sum":
        edge_df = df.groupby(["sender_cluster", "receiver_cluster"], as_index=False).agg(
            edge_score=("edge_score", "sum"),
            expr_only=("expr_only", "mean") if "expr_only" in df.columns else ("edge_score", "mean"),
            confidence=("confidence", "mean"),
            n_pairs=("ligand", "count"),
        )
    elif EDGE_AGG == "mean":
        edge_df = df.groupby(["sender_cluster", "receiver_cluster"], as_index=False).agg(
            edge_score=("edge_score", "mean"),
            expr_only=("expr_only", "mean") if "expr_only" in df.columns else ("edge_score", "mean"),
            confidence=("confidence", "mean"),
            n_pairs=("ligand", "count"),
        )
    else:
        edge_df = df.groupby(["sender_cluster", "receiver_cluster"], as_index=False).agg(
            edge_score=("edge_score", "max"),
            expr_only=("expr_only", "mean") if "expr_only" in df.columns else ("edge_score", "mean"),
            confidence=("confidence", "mean"),
            n_pairs=("ligand", "count"),
        )

    size_df = pd.read_csv(cluster_size_csv)
    ccol = "cluster_label" if "cluster_label" in size_df.columns else "cluster_norm"
    ncol = "n_cells"
    size_df[ccol] = size_df[ccol].astype(str).map(normalize_cluster_str)

    sender_sum = edge_df.groupby("sender_cluster", as_index=False)["edge_score"].sum().rename(
        columns={"sender_cluster": "cluster_norm", "edge_score": "sender_strength_raw"}
    )
    receiver_sum = edge_df.groupby("receiver_cluster", as_index=False)["edge_score"].sum().rename(
        columns={"receiver_cluster": "cluster_norm", "edge_score": "receiver_strength_raw"}
    )

    clusters = sorted(
        set(size_df[ccol].astype(str).tolist()) |
        set(edge_df["sender_cluster"].astype(str).tolist()) |
        set(edge_df["receiver_cluster"].astype(str).tolist()),
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    node_df = pd.DataFrame({"cluster_norm": clusters})
    node_df = node_df.merge(size_df[[ccol, ncol]].rename(columns={ccol: "cluster_norm"}), on="cluster_norm", how="left")
    node_df = node_df.merge(sender_sum, on="cluster_norm", how="left")
    node_df = node_df.merge(receiver_sum, on="cluster_norm", how="left")
    node_df["n_cells"] = node_df["n_cells"].fillna(0).astype(np.int64)
    node_df["sender_strength_raw"] = node_df["sender_strength_raw"].fillna(0.0).astype(np.float32)
    node_df["receiver_strength_raw"] = node_df["receiver_strength_raw"].fillna(0.0).astype(np.float32)

    node_df["sender_strength"] = zscore_1d(np.log1p(node_df["sender_strength_raw"].values))
    node_df["receiver_strength"] = zscore_1d(np.log1p(node_df["receiver_strength_raw"].values))
    node_df["net_flow_raw"] = node_df["receiver_strength_raw"].values - node_df["sender_strength_raw"].values
    node_df["net_flow"] = zscore_1d(node_df["net_flow_raw"].values)
    node_df["io_ratio_raw"] = np.log1p(node_df["receiver_strength_raw"].values) - np.log1p(node_df["sender_strength_raw"].values)
    node_df["io_ratio"] = zscore_1d(node_df["io_ratio_raw"].values)

    return node_df, edge_df


def build_cluster_centroids(cell_df: pd.DataFrame, Z: np.ndarray) -> pd.DataFrame:
    tmp = cell_df.copy().reset_index(drop=True)
    tmp["cluster_norm"] = tmp["cluster_norm"].astype(str).map(normalize_cluster_str)

    rows = []
    for cl in sorted(tmp["cluster_norm"].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else x):
        idx = np.where(tmp["cluster_norm"].values.astype(str) == str(cl))[0]
        if idx.size == 0:
            continue
        zmean = Z[idx].mean(axis=0).astype(np.float32)
        rows.append({
            "cluster_norm": str(cl),
            "n_cells_z": int(idx.size),
            "Z_centroid": zmean,
        })
    return pd.DataFrame(rows)


# ============================================================
# 6) BUILD GRAPH
# ============================================================
def build_cluster_graph(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    centroid_df: pd.DataFrame,
):
    node_df = node_df.copy()
    centroid_df = centroid_df.copy()

    node_df = node_df.merge(
        centroid_df[["cluster_norm", "Z_centroid"]],
        on="cluster_norm",
        how="left",
    )

    clusters = node_df["cluster_norm"].astype(str).tolist()
    c2i = {c: i for i, c in enumerate(clusters)}
    N = len(clusters)

    Zc = np.stack(node_df["Z_centroid"].tolist(), axis=0).astype(np.float32)
    sender = node_df["sender_strength"].values.astype(np.float32)
    receiver = node_df["receiver_strength"].values.astype(np.float32)
    tfresp = node_df["receiver_tf_response"].values.astype(np.float32)

    edge_rows = []

    # 1) directed CCI edges
    if USE_DIRECTED_CCI_EDGE:
        e = edge_df.copy()
        e["sender_cluster"] = e["sender_cluster"].astype(str).map(normalize_cluster_str)
        e["receiver_cluster"] = e["receiver_cluster"].astype(str).map(normalize_cluster_str)

        es = np.log1p(e["edge_score"].values.astype(np.float32))
        es = zscore_1d(es)
        e["edge_score_z"] = es

        for r in e.itertuples(index=False):
            s = str(r.sender_cluster)
            t = str(r.receiver_cluster)
            if s not in c2i or t not in c2i:
                continue
            i = c2i[s]
            j = c2i[t]
            edge_rows.append({
                "src": i,
                "dst": j,
                "edge_weight": float(r.edge_score_z),
                "sim_edge": 0.0,
                "src_sender": float(sender[i]),
                "tgt_receiver": float(receiver[j]),
                "src_tf": float(tfresp[i]),
                "tgt_tf": float(tfresp[j]),
                "edge_type": "cci",
            })

    # 2) similarity graph on centroid Z
    if USE_SIMILARITY_GRAPH and N > 1:
        nnbrs = NearestNeighbors(n_neighbors=min(SIM_K + 1, N), metric="cosine")
        nnbrs.fit(Zc)
        dist, ind = nnbrs.kneighbors(Zc)

        for i in range(N):
            for jj in range(1, ind.shape[1]):
                j = int(ind[i, jj])
                sim = float(1.0 - dist[i, jj])
                edge_rows.append({
                    "src": i,
                    "dst": j,
                    "edge_weight": 0.0,
                    "sim_edge": sim,
                    "src_sender": float(sender[i]),
                    "tgt_receiver": float(receiver[j]),
                    "src_tf": float(tfresp[i]),
                    "tgt_tf": float(tfresp[j]),
                    "edge_type": "sim",
                })

    edge_all = pd.DataFrame(edge_rows)
    if len(edge_all) == 0:
        raise RuntimeError("No graph edges built.")

    # combine same src-dst if repeated
    agg = edge_all.groupby(["src", "dst"], as_index=False).agg(
        edge_weight=("edge_weight", "sum"),
        sim_edge=("sim_edge", "max"),
        src_sender=("src_sender", "mean"),
        tgt_receiver=("tgt_receiver", "mean"),
        src_tf=("src_tf", "mean"),
        tgt_tf=("tgt_tf", "mean"),
    )

    # normalize edge channels
    agg["edge_weight"] = zscore_1d(agg["edge_weight"].values.astype(np.float32))
    agg["sim_edge"] = zscore_1d(agg["sim_edge"].values.astype(np.float32))

    # build fixed neighbor tensor
    src_groups = defaultdict(list)
    for r in agg.itertuples(index=False):
        src_groups[int(r.src)].append(r)

    K = max(len(v) for v in src_groups.values())
    nbr_idx = np.zeros((N, K), np.int64)
    nbr_mask = np.zeros((N, K), np.bool_)
    edge_feat = np.zeros((N, K, 6), np.float32)

    for i in range(N):
        rows = src_groups.get(i, [])
        if len(rows) == 0:
            nbr_idx[i, 0] = i
            nbr_mask[i, 0] = True
            edge_feat[i, 0, :] = np.array([
                0.0, 0.0, sender[i], receiver[i], tfresp[i], tfresp[i]
            ], np.float32)
            continue

        for k, r in enumerate(rows):
            nbr_idx[i, k] = int(r.dst)
            nbr_mask[i, k] = True
            edge_feat[i, k, :] = np.array([
                float(r.edge_weight),
                float(r.sim_edge),
                float(r.src_sender),
                float(r.tgt_receiver),
                float(r.src_tf),
                float(r.tgt_tf),
            ], np.float32)

    return clusters, Zc, agg, nbr_idx, nbr_mask, edge_feat


# ============================================================
# 7) TRAIN COMMUNICATION PSEUDOTIME
# ============================================================
def train_comm_pseudotime(
    feature_mat: np.ndarray,
    weak_teacher: np.ndarray,
    recon_target: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_mask: np.ndarray,
    edge_feat: np.ndarray,
    edge_df_agg: pd.DataFrame,
):
    rng = np.random.RandomState(SEED)
    N = feature_mat.shape[0]

    x_t = torch.from_numpy(feature_mat).to(DEVICE)
    teacher_t = torch.from_numpy(weak_teacher.astype(np.float32)).to(DEVICE)
    recon_t = torch.from_numpy(recon_target.astype(np.float32)).to(DEVICE)
    nbr_idx_t = torch.from_numpy(nbr_idx).to(DEVICE)
    nbr_mask_t = torch.from_numpy(nbr_mask).to(DEVICE)
    edge_feat_t = torch.from_numpy(edge_feat).to(DEVICE)

    # edge arrays for losses
    edge_src = edge_df_agg["src"].values.astype(np.int64)
    edge_dst = edge_df_agg["dst"].values.astype(np.int64)
    edge_w = edge_df_agg["edge_weight"].values.astype(np.float32)
    sim_w = edge_df_agg["sim_edge"].values.astype(np.float32)

    # only directional CCI edges for direction loss
    dir_mask = np.abs(edge_w) > 1e-8
    dir_src = edge_src[dir_mask]
    dir_dst = edge_dst[dir_mask]
    dir_w = edge_w[dir_mask]

    # smooth on all edges
    sm_src = edge_src
    sm_dst = edge_dst
    sm_w = np.maximum(np.abs(edge_w) + np.maximum(sim_w, 0.0), 1e-6)

    def sample_rank_pairs(n_pairs):
        i = rng.randint(0, N, size=n_pairs)
        j = rng.randint(0, N, size=n_pairs)
        m = weak_teacher[j] > weak_teacher[i]
        i = i[m]
        j = j[m]
        if i.size < max(20, n_pairs // 5):
            order = np.argsort(weak_teacher)
            take = min(len(order) // 2, max(20, n_pairs // 2))
            i = order[:take]
            j = order[-take:]
        return i.astype(np.int64), j.astype(np.int64)

    def sample_edges(src, dst, w, n_take):
        if len(src) == 0:
            return (
                np.zeros((0,), np.int64),
                np.zeros((0,), np.int64),
                np.zeros((0,), np.float32),
            )
        idx = rng.choice(np.arange(len(src)), size=min(n_take, len(src)), replace=False)
        return src[idx].astype(np.int64), dst[idx].astype(np.int64), w[idx].astype(np.float32)

    T_runs = []
    Tau_runs = []

    for run_seed in ENSEMBLE_SEEDS:
        log(f"\n[ENSEMBLE] seed={run_seed}")
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        model = ClusterCommLTC(
            xdim=feature_mat.shape[1],
            hdim=H,
            out_dim=recon_target.shape[1],
        ).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for ep in range(1, EPOCHS + 1):
            model.train()
            opt.zero_grad(set_to_none=True)

            t_hat, recon, tau = model(x_t, nbr_idx_t, nbr_mask_t, edge_feat_t)

            # weak teacher fit
            loss_teach = F.mse_loss(t_hat, teacher_t)

            # rank loss
            pi, pj = sample_rank_pairs(RANK_PAIRS)
            pi_t = torch.from_numpy(pi).to(DEVICE)
            pj_t = torch.from_numpy(pj).to(DEVICE)
            loss_rank = F.softplus(-(t_hat[pj_t] - t_hat[pi_t])).mean()

            # smoothness loss
            ei, ej, ew = sample_edges(sm_src, sm_dst, sm_w, SMOOTH_EDGES)
            ei_t = torch.from_numpy(ei).to(DEVICE)
            ej_t = torch.from_numpy(ej).to(DEVICE)
            ew_t = torch.from_numpy(ew).to(DEVICE)
            if ei_t.numel() > 0:
                loss_smooth = (ew_t * (t_hat[ei_t] - t_hat[ej_t]).pow(2)).mean()
            else:
                loss_smooth = torch.tensor(0.0, device=DEVICE)

            # directed edge consistency: src earlier than dst on positive CCI flow
            di, dj, dw = sample_edges(dir_src, dir_dst, np.maximum(dir_w, 1e-6), DIR_EDGES)
            di_t = torch.from_numpy(di).to(DEVICE)
            dj_t = torch.from_numpy(dj).to(DEVICE)
            dw_t = torch.from_numpy(dw).to(DEVICE)
            if di_t.numel() > 0:
                loss_dir = (dw_t * F.softplus(-(t_hat[dj_t] - t_hat[di_t]))).mean()
            else:
                loss_dir = torch.tensor(0.0, device=DEVICE)

            # reconstruction helps latent keep node semantics
            loss_recon = F.mse_loss(recon, recon_t)

            loss = (
                W_TEACH * loss_teach
                + W_RANK * loss_rank
                + W_SMOOTH * loss_smooth
                + W_DIR_EDGE * loss_dir
                + W_RECON * loss_recon
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            if ep == 1 or ep % 50 == 0:
                corr = float(np.corrcoef(t_hat.detach().cpu().numpy(), weak_teacher)[0, 1])
                log(
                    f"  ep {ep:03d} "
                    f"loss={loss.item():.4f} "
                    f"teach={loss_teach.item():.4f} "
                    f"rank={loss_rank.item():.4f} "
                    f"smooth={loss_smooth.item():.4f} "
                    f"dir={loss_dir.item():.4f} "
                    f"recon={loss_recon.item():.4f} "
                    f"corr={corr:.3f}"
                )

        model.eval()
        with torch.no_grad():
            t_hat, recon, tau = model(x_t, nbr_idx_t, nbr_mask_t, edge_feat_t)

        T_runs.append(t_hat.detach().cpu().numpy().astype(np.float32))
        Tau_runs.append(tau.detach().cpu().numpy().astype(np.float32))

    T = np.stack(T_runs, axis=0)
    Tau = np.stack(Tau_runs, axis=0)

    Tn = (T - T.min(axis=1, keepdims=True)) / (T.max(axis=1, keepdims=True) - T.min(axis=1, keepdims=True) + 1e-12)
    t_hat_norm = Tn.mean(axis=0)
    sigma_t = Tn.std(axis=0)
    tau_mean = Tau.mean(axis=0)

    return t_hat_norm.astype(np.float32), sigma_t.astype(np.float32), tau_mean.astype(np.float32)


# ============================================================
# 8) CELL-LEVEL PROJECTION
# ============================================================
def project_cell_pseudotime(cell_df: pd.DataFrame, Z: np.ndarray, cluster_time_df: pd.DataFrame):
    tmp = cell_df.copy().reset_index(drop=True)
    m = cluster_time_df[["cluster_norm", "t_hat_norm_mean"]].copy()
    tmp = tmp.merge(m, on="cluster_norm", how="left")
    base_t = tmp["t_hat_norm_mean"].fillna(tmp["t_hat_norm_mean"].mean()).values.astype(np.float32)

    if len(tmp) <= 1:
        tmp["cell_t_proj"] = base_t
        return tmp

    nnbrs = NearestNeighbors(n_neighbors=min(CELL_KNN + 1, len(tmp)), metric="cosine")
    nnbrs.fit(Z)
    dist, ind = nnbrs.kneighbors(Z)

    smooth_t = np.zeros_like(base_t, np.float32)
    for i in range(len(tmp)):
        nb = ind[i, 1:]
        dd = dist[i, 1:]
        w = np.exp(-dd.astype(np.float32))
        smooth_t[i] = (0.6 * base_t[i] + 0.4 * np.sum(w * base_t[nb]) / (np.sum(w) + 1e-8))

    smooth_t = (smooth_t - smooth_t.min()) / (smooth_t.max() - smooth_t.min() + 1e-12)
    tmp["cell_t_proj"] = smooth_t.astype(np.float32)
    return tmp


# ============================================================
# 9) MAIN
# ============================================================
def main():
    # ---------- checks ----------
    for p in [
        BASE_CKPT,
        FRONTIER_CKPT,
        INT_TFKEEP_CSV,
        CLUSTER_PROF_CSV,
        CLUSTER_SIZE_CSV,
        CELL_MAP_CSV,
        OLD_CLUSTER_CSV,
        PROJECTED_RNA,
        GLOBAL_GENE_VOCAB,
    ]:
        assert p.exists(), f"Missing: {p}"

    log("[OUT]", OUT_DIR.resolve())

    # ---------- load base model ----------
    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)

    base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = base_ckpt.get("cfg", {}) or {}
    meta = base_ckpt.get("meta", None)
    state_dict = base_ckpt.get("model", None)
    assert state_dict is not None

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
        default_cancer_name_tmp = get_default_cancer_name(DATASET_NAME)
        cfg["cancer_types"] = [str(default_cancer_name_tmp)] + [f"unused_{i}" for i in range(1, n_cancer_tmp)]

    N_GENE_DIM = int(cfg["N_GENE_DIM"])
    N_TF_DIM = int(cfg["N_TF_DIM"])
    PEAK_HASH_SIZE = int(cfg["PEAK_HASH_SIZE"])
    D = int(cfg["D"])
    HID = int(cfg["HID"])
    DROP = float(cfg["DROP"])
    cancer_types = list(cfg["cancer_types"])

    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    DEFAULT_CANCER_ID = int(cancer2id.get(get_default_cancer_name(DATASET_NAME), 0))

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

    # ---------- adapter ----------
    frontier_ckpt = torch.load(FRONTIER_CKPT, map_location="cpu", weights_only=False)
    adapter_sd = frontier_ckpt.get("adapter_state_dict", None)
    if adapter_sd is None:
        adapter_sd = frontier_ckpt.get("adapter", None)
    assert adapter_sd is not None

    hyper = frontier_ckpt.get("hyper", {})
    K_RANK = int(hyper.get("K_RANK", 32))
    n_group = int(adapter_sd["B_group.weight"].shape[0])

    # minimal prior dims only for shape match
    adapter = HyperAdapterHashV2(
        d=D,
        k=K_RANK,
        n_tf=1,
        n_gene=1,
        peak_hash_size=PEAK_HASH_SIZE,
        n_cancer=len(cancer_types),
        n_group=n_group,
        e_dim=64,
        hid=256,
    ).to(DEVICE)

    cur = adapter.state_dict()
    for k, v in adapter_sd.items():
        if k in cur and tuple(cur[k].shape) == tuple(v.shape):
            cur[k] = v
    adapter.load_state_dict(cur, strict=False)
    adapter.eval()

    # ---------- cell map ----------
    sample = load_single_sample(DATASET_NAME, SAMPLE_ID)

    cell_map = pd.read_csv(CELL_MAP_CSV)
    assert "sid" in cell_map.columns
    assert "barcode" in cell_map.columns
    assert "new_leiden" in cell_map.columns

    cell_map["sid"] = cell_map["sid"].astype(str)
    cell_map["barcode"] = cell_map["barcode"].astype(str).map(norm_barcode)
    cell_map = cell_map[cell_map["sid"] == SAMPLE_ID].copy().reset_index(drop=True)
    cell_map["cluster_norm"] = cell_map["new_leiden"].astype(str).map(normalize_cluster_str)

    olddf = pd.read_csv(OLD_CLUSTER_CSV)
    olddf["sid"] = olddf["sid"].astype(str)
    olddf["barcode"] = olddf["barcode"].astype(str).map(norm_barcode)
    olddf = olddf[olddf["sid"] == SAMPLE_ID][["barcode", "group_id"]].drop_duplicates("barcode")

    cell_map = cell_map.merge(olddf, on="barcode", how="left")
    cell_map["group_id"] = cell_map["group_id"].fillna(0).astype(np.int64)

    # only keep barcodes present in sample
    cell_map = cell_map[cell_map["barcode"].astype(str).isin(sample.bc2rid.keys())].copy().reset_index(drop=True)
    assert len(cell_map) > 0

    log("[CELLS] matched =", len(cell_map))

    # ---------- encode cell Z ----------
    kept_cell_df, Z = encode_cells_Z(
        model=model,
        adapter=adapter,
        sample=sample,
        cell_df=cell_map[["barcode", "cluster_norm", "group_id"]].copy(),
        default_cancer_id=DEFAULT_CANCER_ID,
    )
    assert len(kept_cell_df) == Z.shape[0]

    # ---------- cluster features ----------
    node_df, edge_df = aggregate_cluster_cci(INT_TFKEEP_CSV, CLUSTER_SIZE_CSV)
    tf_df = load_cluster_profiles_tf_response(CLUSTER_PROF_CSV)
    centroid_df = build_cluster_centroids(kept_cell_df, Z)

    node_df = node_df.merge(tf_df, on="cluster_norm", how="left")
    node_df["receiver_tf_response"] = node_df["receiver_tf_response"].fillna(0.0).astype(np.float32)
    node_df["tf_top_mean"] = node_df["tf_top_mean"].fillna(0.0).astype(np.float32)
    node_df["fallback_tf_resp_z"] = node_df["fallback_tf_resp_z"].fillna(0.0).astype(np.float32)

    # weak teacher
    weak_raw = (
        W_RECEIVER * node_df["receiver_strength"].values.astype(np.float32)
        + W_TFRESP * node_df["receiver_tf_response"].values.astype(np.float32)
        - W_SENDER * node_df["sender_strength"].values.astype(np.float32)
        + W_NET * node_df["net_flow"].values.astype(np.float32)
        + W_RATIO * node_df["io_ratio"].values.astype(np.float32)
    ).astype(np.float32)
    weak_raw = winsorize(weak_raw, 0.01, 0.99)
    weak_teacher = soft_rank(weak_raw)

    # ---------- graph ----------
    clusters, Zc, edge_df_agg, nbr_idx, nbr_mask, edge_feat = build_cluster_graph(
        node_df=node_df,
        edge_df=edge_df,
        centroid_df=centroid_df,
    )

    node_df = node_df.set_index("cluster_norm").loc[clusters].reset_index()

    # ---------- feature matrix ----------
    pca = PCA(n_components=min(8, Zc.shape[1], len(clusters)), random_state=SEED)
    Zc_pca = pca.fit_transform(Zc).astype(np.float32)

    feat_df = pd.DataFrame({
        "cluster_norm": clusters,
        "sender_strength": node_df["sender_strength"].values.astype(np.float32),
        "receiver_strength": node_df["receiver_strength"].values.astype(np.float32),
        "net_flow": node_df["net_flow"].values.astype(np.float32),
        "io_ratio": node_df["io_ratio"].values.astype(np.float32),
        "receiver_tf_response": node_df["receiver_tf_response"].values.astype(np.float32),
        "n_cells_log1p": np.log1p(node_df["n_cells"].values.astype(np.float32)),
    })

    X_basic = feat_df.drop(columns=["cluster_norm"]).values.astype(np.float32)
    X_basic = StandardScaler().fit_transform(X_basic).astype(np.float32)
    X_all = np.concatenate([X_basic, Zc_pca], axis=1).astype(np.float32)

    recon_target = np.concatenate([
        node_df["sender_strength"].values[:, None].astype(np.float32),
        node_df["receiver_strength"].values[:, None].astype(np.float32),
        node_df["net_flow"].values[:, None].astype(np.float32),
        node_df["io_ratio"].values[:, None].astype(np.float32),
        node_df["receiver_tf_response"].values[:, None].astype(np.float32),
        Zc_pca.astype(np.float32),
    ], axis=1).astype(np.float32)

    # save node feature table
    feat_out = node_df.copy()
    feat_out["weak_teacher_raw"] = weak_raw.astype(np.float32)
    feat_out["weak_teacher_rank"] = weak_teacher.astype(np.float32)
    feat_out.to_csv(OUT_DIR / "cluster_dynamic_features.csv", index=False)

    # ---------- train ----------
    t_hat_norm, sigma_t, tau_mean = train_comm_pseudotime(
        feature_mat=X_all,
        weak_teacher=weak_teacher,
        recon_target=recon_target,
        nbr_idx=nbr_idx,
        nbr_mask=nbr_mask,
        edge_feat=edge_feat,
        edge_df_agg=edge_df_agg,
    )

    cluster_pt = node_df.copy()
    cluster_pt["weak_teacher_raw"] = weak_raw.astype(np.float32)
    cluster_pt["weak_teacher_rank"] = weak_teacher.astype(np.float32)
    cluster_pt["t_hat_norm_mean"] = t_hat_norm.astype(np.float32)
    cluster_pt["sigma_t"] = sigma_t.astype(np.float32)
    cluster_pt["tau_mean"] = tau_mean.astype(np.float32)
    cluster_pt = cluster_pt.sort_values("t_hat_norm_mean", ascending=True).reset_index(drop=True)
    cluster_pt.to_csv(OUT_DIR / "cluster_pseudotime.csv", index=False)

    # ---------- edge summary ----------
    c2t = dict(zip(cluster_pt["cluster_norm"].astype(str), cluster_pt["t_hat_norm_mean"].astype(np.float32)))
    es = edge_df.copy()
    es["sender_cluster"] = es["sender_cluster"].astype(str).map(normalize_cluster_str)
    es["receiver_cluster"] = es["receiver_cluster"].astype(str).map(normalize_cluster_str)
    es["t_sender"] = es["sender_cluster"].map(c2t).astype(np.float32)
    es["t_receiver"] = es["receiver_cluster"].map(c2t).astype(np.float32)
    es["delta_t_receiver_minus_sender"] = (es["t_receiver"] - es["t_sender"]).astype(np.float32)
    es = es.sort_values(["edge_score", "delta_t_receiver_minus_sender"], ascending=[False, False]).reset_index(drop=True)
    es.to_csv(OUT_DIR / "edge_pseudotime_summary.csv", index=False)

    # ---------- cell projection ----------
    cell_proj = project_cell_pseudotime(
        cell_df=kept_cell_df.copy(),
        Z=Z,
        cluster_time_df=cluster_pt[["cluster_norm", "t_hat_norm_mean"]].copy(),
    )
    cell_proj["sid"] = SAMPLE_ID
    cell_proj["cell_key"] = [make_cell_key(SAMPLE_ID, x) for x in cell_proj["barcode"].astype(str).tolist()]
    cell_proj.to_csv(OUT_DIR / "cell_projected_pseudotime.csv", index=False)

    # ========================================================
    # QC PLOTS
    # ========================================================
    t_map = dict(zip(cluster_pt["cluster_norm"].astype(str), cluster_pt["t_hat_norm_mean"].values.astype(np.float32)))

    # 1) weak teacher vs learned t
    plt.figure(figsize=(6, 5))
    plt.scatter(cluster_pt["weak_teacher_rank"].values, cluster_pt["t_hat_norm_mean"].values, s=35, alpha=0.9)
    for _, r in cluster_pt.iterrows():
        plt.text(float(r["weak_teacher_rank"]), float(r["t_hat_norm_mean"]), str(r["cluster_norm"]), fontsize=8)
    plt.xlabel("weak_teacher_rank")
    plt.ylabel("t_hat_norm_mean")
    plt.title("Weak teacher vs learned pseudotime")
    save_fig(OUT_DIR / "qc_teacher_vs_t.png")

    # 2) sender/receiver along t
    df_plot = cluster_pt.sort_values("t_hat_norm_mean").reset_index(drop=True)
    plt.figure(figsize=(7, 4))
    plt.plot(df_plot["t_hat_norm_mean"], df_plot["sender_strength"], marker="o", label="sender")
    plt.plot(df_plot["t_hat_norm_mean"], df_plot["receiver_strength"], marker="o", label="receiver")
    plt.plot(df_plot["t_hat_norm_mean"], df_plot["receiver_tf_response"], marker="o", label="receiver_tf")
    plt.legend()
    plt.xlabel("t_hat_norm_mean")
    plt.ylabel("z-scored feature")
    plt.title("Sender / receiver / TF along pseudotime")
    save_fig(OUT_DIR / "qc_sender_receiver_tf_along_t.png")

    # 3) sigma
    plt.figure(figsize=(6, 4))
    plt.scatter(cluster_pt["t_hat_norm_mean"], cluster_pt["sigma_t"], s=35, alpha=0.9)
    for _, r in cluster_pt.iterrows():
        plt.text(float(r["t_hat_norm_mean"]), float(r["sigma_t"]), str(r["cluster_norm"]), fontsize=8)
    plt.xlabel("t_hat_norm_mean")
    plt.ylabel("sigma_t")
    plt.title("Uncertainty along pseudotime")
    save_fig(OUT_DIR / "qc_sigma_vs_t.png")

    # 4) tau
    plt.figure(figsize=(6, 4))
    plt.scatter(cluster_pt["t_hat_norm_mean"], cluster_pt["tau_mean"], s=35, alpha=0.9)
    for _, r in cluster_pt.iterrows():
        plt.text(float(r["t_hat_norm_mean"]), float(r["tau_mean"]), str(r["cluster_norm"]), fontsize=8)
    plt.xlabel("t_hat_norm_mean")
    plt.ylabel("tau_mean")
    plt.title("LTC tau along pseudotime")
    save_fig(OUT_DIR / "qc_tau_vs_t.png")

    # 5) edge delta t histogram
    plt.figure(figsize=(6, 4))
    plt.hist(es["delta_t_receiver_minus_sender"].values.astype(np.float32), bins=20)
    plt.xlabel("t(receiver) - t(sender)")
    plt.ylabel("count")
    plt.title("Edge temporal shift")
    save_fig(OUT_DIR / "qc_edge_delta_t_hist.png")

    # 6) bin summary
    x = cluster_pt["t_hat_norm_mean"].values.astype(np.float32)
    b = bin_by_quantile(x, N_BINS)

    xs, ys_sender, ys_receiver, ys_tf = [], [], [], []
    for bi in range(N_BINS):
        idx = np.where(b == bi)[0]
        if idx.size == 0:
            continue
        xs.append(float((bi + 0.5) / N_BINS))
        ys_sender.append(float(cluster_pt["sender_strength"].values[idx].mean()))
        ys_receiver.append(float(cluster_pt["receiver_strength"].values[idx].mean()))
        ys_tf.append(float(cluster_pt["receiver_tf_response"].values[idx].mean()))

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys_sender, marker="o", label="sender")
    plt.plot(xs, ys_receiver, marker="o", label="receiver")
    plt.plot(xs, ys_tf, marker="o", label="receiver_tf")
    plt.legend()
    plt.xlabel("pseudotime bin")
    plt.ylabel("mean feature")
    plt.title("Binned dynamics along pseudotime")
    save_fig(OUT_DIR / "qc_binned_dynamics.png")

    log("[SAVE] cluster_dynamic_features.csv")
    log("[SAVE] cluster_pseudotime.csv")
    log("[SAVE] edge_pseudotime_summary.csv")
    log("[SAVE] cell_projected_pseudotime.csv")
    log("[DONE] outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()