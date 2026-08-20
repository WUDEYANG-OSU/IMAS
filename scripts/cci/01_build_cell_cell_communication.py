from __future__ import annotations

# ============================================================
# crc_sample9_cci_newLeiden_from_frontierHash.py
# ------------------------------------------------------------
# 目标：
# - 仅分析 CRC-Tissue 的第九个样本：CRC_RNA_TO009
# - 通讯分析 cluster 使用“该样本重新降维聚类得到的新 Leiden cluster”
# - 不再使用旧的 internal 16-cluster 作为通讯单元
# - 仍复用已经训练好的 frontier hash Stage-B 模型进行细胞表征 / gene-TF 打分
#
# 输出：
# 1) receptor_tf_signature.csv
# 2) interactions_cluster_level.csv
# 3) interactions_cluster_level_highconf.csv
# 4) interactions_cluster_level_highconf_TFkept.csv
# 5) cluster_profiles.csv
# 6) cluster_sizes.csv
# 7) cell_level_mapping_with_umap_and_internal16.csv
# 8) matchedcells_scanpy.h5ad
#
# 说明：
# - 这里保留 cell-level 的 group adaptor（来自旧模型），因为它作用在每个细胞上
# - 但不使用 cluster correction head，因为它绑定旧训练 cluster summary，不适合新 Leiden cluster
# - new Leiden 现在由本脚本直接重跑，参数与 single-sample trendmap 脚本保持一致
# ============================================================

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
import anndata as ad
import scanpy as sc

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
GLOBAL_PEAK_VOCAB = MAPS_ROOT / "global_peak_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"

RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)
FRONTIER_CKPT = RUN_STAGEB_DIR / "frontier_best.pt"
BASE_CKPT = get_dataset_checkpoint("CRC-Tissue")
CLUSTERS_CSV = RUN_STAGEB_DIR / "clusters.csv"

# 这次改成：直接在本脚本里重跑聚类，并把结果写到这个目录
RECLUSTER_RUN_DIR = RUN_STAGEB_DIR / f"single_sample_{SAMPLE_ID}_recluster_for_cci"
RECLUSTER_RUN_DIR.mkdir(parents=True, exist_ok=True)

ADATA_RECLUSTERED = RECLUSTER_RUN_DIR / f"{SAMPLE_ID}_matchedcells_scanpy.h5ad"
CELL_LEVEL_TABLE = RECLUSTER_RUN_DIR / f"{SAMPLE_ID}_cell_level_mapping_with_umap_and_internal16.csv"

# 原始 RNA / projected RNA / manifest
RNA_DIR = WORLD_ROOT / DATASET_NAME / "processed" / "RNA"

# LR prior
LR_UNION_CSV = Path("./lr_union_out/lr_union_human_WITH_CELLCHAT.csv")

OUT_DIR = RUN_STAGEB_DIR / f"{SAMPLE_ID}_cci_newLeiden"
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

# ---------- scanpy / UMAP / Leiden ----------
# 与你给的 single-sample trendmap 脚本保持一致
N_TOP_HVG = 2000
N_PCS = 20
N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.03
UMAP_SPREAD = 2.0
LEIDEN_RESOLUTION = 0.5

# ---------- encoding ----------
RNA_TOPK = 2048
EMB_BATCH = 256

# ---------- score mixing ----------
COUNTS_WEIGHT = 0.50
HIGHCONF_MIN_SOURCES = 2
GENE_VEC_BATCH = 2048

# ---------- CCI v3 sharp TF signature ----------
TF_SIG_TOPM = 10
TF_SIG_MIN_CORR = 0.18
TF_FALLBACK_TOPK = 5

# TF post-filter gating
TF_KEEP_Z = 0.0
TF_PENALTY = 1.2

# ---------- adaptor ----------
USE_GROUP_SCALE_BIAS = True
GROUP_SCALE_ALPHA = 0.12
USE_CELLGROUP_ADAPTOR = True
USE_CANCER_ADAPTOR = True

# ---------- output ----------
SIG_CSV = OUT_DIR / "receptor_tf_signature.csv"
INT_CSV = OUT_DIR / "interactions_cluster_level.csv"
INT_HI_CSV = OUT_DIR / "interactions_cluster_level_highconf.csv"
INT_HI_TF_CSV = OUT_DIR / "interactions_cluster_level_highconf_TFkept.csv"
PROF_CSV = OUT_DIR / "cluster_profiles.csv"
CLUSTER_SIZE_CSV = OUT_DIR / "cluster_sizes.csv"


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


def dataset_to_cancer_type(ds_name: str) -> str:
    s = str(ds_name)
    return s.split("-", 1)[0] if "-" in s else s


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if (not np.isfinite(s)) or s < 1e-6:
        return np.zeros_like(x, np.float32)
    return (x - m) / (s + 1e-6)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    x = np.clip(x, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-x))


def tf_gate(tfz: np.ndarray) -> np.ndarray:
    tfz = np.asarray(tfz, np.float32)
    out = np.ones_like(tfz, np.float32)
    m = tfz < float(TF_KEEP_Z)
    out[m] = np.exp(-float(TF_PENALTY) * (float(TF_KEEP_Z) - tfz[m]))
    return out


def looks_like_gene_symbol(x: str) -> bool:
    x = str(x).strip()
    if not x or " " in x:
        return False
    if len(x) < 2 or len(x) > 40:
        return False
    return re.fullmatch(r"[A-Za-z0-9\-\.\:]+", x) is not None


def safe_to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def topk_from_dense_row(row: np.ndarray, topk: int):
    n = row.size
    k = min(int(topk), int(n))
    if k <= 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
    idx = np.argpartition(row, -k)[-k:]
    vals = row[idx]
    order = np.argsort(vals)[::-1]
    return idx[order].astype(np.int64), vals[order].astype(np.float32)


def get_default_cancer_name(dataset_name: str) -> str | None:
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


def find_first_existing_col(df: pd.DataFrame, cand_cols: list[str]) -> str | None:
    for c in cand_cols:
        if c in df.columns:
            return c
    return None


# ============================================================
# 2) BASE MODEL (HASH PEAK VERSION)
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
# 3) TRANSFER ADAPTER (hash version)
# ============================================================
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
        return V[0], V[1], V[2], V[3]  # tf, gene, peak, cancer

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
# 4) SAMPLE OBJECT / PROJECTED RNA STORE
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


def load_single_sample(dataset_name: str, sample_id: str, genes_vocab: list[str]) -> Sample:
    manifest_rna_dir = get_manifest_rna_dir(dataset_name)
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    df = pd.read_csv(manifest_path, sep="\t")
    need_cols = ["cell_id", "sample_id", "row_index"]
    for c in need_cols:
        assert c in df.columns, f"{manifest_path.name}: missing column {c}"

    g = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    assert len(g) > 0, f"No rows found in manifest for sample={sample_id}"
    g = g.sort_values("row_index").reset_index(drop=True)

    meta_path = manifest_rna_dir / f"{sample_id}.rna_meta.npz"
    counts_path = manifest_rna_dir / f"{sample_id}.X_rna_counts_dense.npy"
    proj_rna_path = PROJECTED_ROOT / dataset_name / f"{sample_id}.rna.global.csr.npz"

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
    assert counts_mm.shape[1] == len(gene_names)

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

    bc2rid = {str(bc): i for i, bc in enumerate(cells.tolist())}

    s = Sample(
        sid=sample_id,
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
    return s


def get_counts_row_dense(sample: Sample, i: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    row = np.asarray(x[i]).reshape(-1)
    return row.astype(np.float32, copy=False)


def build_true_counts_anndata(sample: Sample) -> ad.AnnData:
    X = np.load(sample.counts_path, mmap_mode="r")
    X = np.asarray(X, dtype=np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.obs_names = [make_cell_key(sample.sid, bc) for bc in sample.cells.tolist()]
    adata.obs["sid"] = sample.sid
    adata.obs["barcode"] = sample.cells.astype(str)
    adata.obs["rid"] = np.arange(len(sample.cells), dtype=np.int64)
    adata.var_names = [str(x) for x in sample.gene_names.tolist()]
    return adata


def run_leiden_umap_on_true_counts(adata: ad.AnnData):
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

    adata_h = adata[:, hvg_mask].copy()

    sc.pp.scale(adata_h, max_value=10)
    sc.tl.pca(adata_h, n_comps=min(N_PCS, max(2, adata_h.n_vars - 1)), svd_solver="arpack")
    sc.pp.neighbors(adata_h, n_neighbors=N_NEIGHBORS, n_pcs=min(N_PCS, adata_h.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata_h, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD)
    sc.tl.leiden(adata_h, resolution=LEIDEN_RESOLUTION, key_added="leiden_new")

    adata.obs["new_cluster"] = adata_h.obs["leiden_new"].astype(str).values
    adata.obsm["X_umap"] = adata_h.obsm["X_umap"].copy()
    return adata


# ============================================================
# 5) LOADING PRIORS
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


# ============================================================
# 6) DIRECT RECLUSTER WITH SAME PARAMS AS TRENDMAP SCRIPT
# ============================================================
def run_and_save_new_leiden_mapping(sample: Sample) -> pd.DataFrame:
    log("[LEIDEN] re-running clustering directly from true counts ...")
    log("[LEIDEN] params =", {
        "N_TOP_HVG": N_TOP_HVG,
        "N_PCS": N_PCS,
        "N_NEIGHBORS": N_NEIGHBORS,
        "UMAP_MIN_DIST": UMAP_MIN_DIST,
        "UMAP_SPREAD": UMAP_SPREAD,
        "LEIDEN_RESOLUTION": LEIDEN_RESOLUTION,
        "SEED": SEED,
    })

    adata_true = build_true_counts_anndata(sample)
    adata_true = run_leiden_umap_on_true_counts(adata_true)

    # save h5ad
    adata_true.write_h5ad(str(ADATA_RECLUSTERED))

    out = pd.DataFrame({
        "sid": SAMPLE_ID,
        "rid": np.arange(adata_true.n_obs, dtype=np.int64),
        "barcode": adata_true.obs["barcode"].astype(str).values,
        "cell_key": adata_true.obs_names.astype(str),
        "new_leiden": adata_true.obs["new_cluster"].astype(str).values,
        "umap_1": adata_true.obsm["X_umap"][:, 0].astype(np.float32),
        "umap_2": adata_true.obsm["X_umap"][:, 1].astype(np.float32),
    })

    out.to_csv(CELL_LEVEL_TABLE, index=False)

    log("[LEIDEN] saved h5ad:", ADATA_RECLUSTERED)
    log("[LEIDEN] saved cell-level csv:", CELL_LEVEL_TABLE)
    log("[LEIDEN] n_cluster =", out["new_leiden"].astype(str).nunique())
    log("[LEIDEN] cluster counts:")
    log(out["new_leiden"].astype(str).value_counts().sort_index())

    ret = out[["barcode", "new_leiden"]].copy()
    ret = ret.rename(columns={"new_leiden": "new_cluster"})
    ret["barcode"] = ret["barcode"].astype(str).map(norm_barcode)
    ret["new_cluster"] = ret["new_cluster"].astype(str)
    ret = ret.drop_duplicates("barcode").reset_index(drop=True)
    return ret


# ============================================================
# 7) ENCODE RNA -> Z
# ============================================================
def build_tokens_for_cells(
    cell_list: list[tuple[str, int]],
    sid2sample: dict[str, Sample],
    cancer_id_int: int,
    proj_store_cache: dict,
):
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
def encode_Z(
    cell_list: list[tuple[str, int]],
    sid2sample: dict[str, Sample],
    model,
    cancer_id_int: int,
    proj_store_cache: dict,
) -> np.ndarray:
    Zs = []
    for st in range(0, len(cell_list), EMB_BATCH):
        b = cell_list[st:st + EMB_BATCH]
        idx_cat, val_cat, offsets, cancer_ids = build_tokens_for_cells(
            b, sid2sample, cancer_id_int, proj_store_cache
        )
        z = model.encode_rna(
            torch.from_numpy(idx_cat).to(DEVICE),
            torch.from_numpy(val_cat).to(DEVICE),
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )
        Zs.append(z.detach().cpu().numpy())
    return np.concatenate(Zs, axis=0).astype(np.float32)


# ============================================================
# 8) MAIN
# ============================================================
def main():
    # ---------- checks ----------
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing {GLOBAL_TF_VOCAB}"
    assert GLOBAL_PEAK_VOCAB.exists(), f"Missing {GLOBAL_PEAK_VOCAB}"
    assert PRIORS_DIR.exists(), f"Missing {PRIORS_DIR}"
    assert FRONTIER_CKPT.exists(), f"Missing {FRONTIER_CKPT}"
    assert BASE_CKPT.exists(), f"Missing {BASE_CKPT}"
    assert CLUSTERS_CSV.exists(), f"Missing {CLUSTERS_CSV}"
    assert LR_UNION_CSV.exists(), f"Missing {LR_UNION_CSV}"

    log("[OUT]", OUT_DIR.resolve())

    # ---------- vocab ----------
    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    # ---------- load base ckpt ----------
    base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = base_ckpt.get("cfg", None)
    meta = base_ckpt.get("meta", None)
    state_dict = base_ckpt.get("model", None)
    assert state_dict is not None, f"BASE ckpt missing model weights: {BASE_CKPT}"

    if cfg is None:
        cfg = {}

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
        n_cancer_tmp = int(state_dict["cell_c.weight"].shape[0])
        default_cancer_name_tmp = get_default_cancer_name(DATASET_NAME)
        cfg["cancer_types"] = [str(default_cancer_name_tmp)] + [f"unused_{i}" for i in range(1, n_cancer_tmp)]

    N_GENE_DIM = int(cfg["N_GENE_DIM"])
    N_TF_DIM = int(cfg["N_TF_DIM"])
    PEAK_HASH_SIZE = int(cfg["PEAK_HASH_SIZE"])
    D = int(cfg.get("D", 256))
    HID = int(cfg.get("HID", 512))
    DROP = float(cfg.get("DROP", 0.1))
    cancer_types = list(cfg["cancer_types"])

    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_name = get_default_cancer_name(DATASET_NAME)
    DEFAULT_CANCER_ID = int(cancer2id.get(default_cancer_name, 0))
    cid_scalar = torch.tensor([DEFAULT_CANCER_ID], device=DEVICE, dtype=torch.long)

    log("[BASE] loaded")
    log("[BASE_CFG]", {
        "N_GENE_DIM": N_GENE_DIM,
        "N_TF_DIM": N_TF_DIM,
        "PEAK_HASH_SIZE": PEAK_HASH_SIZE,
        "D": D,
        "HID": HID,
        "DROP": DROP,
        "n_cancer": len(cancer_types),
        "default_cancer_name": default_cancer_name,
        "default_cancer_id": DEFAULT_CANCER_ID,
    })

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

    # ---------- priors ----------
    tf_gene_prior = PRIORS_DIR / "tf_gene_f.cap256.npz"
    if not tf_gene_prior.exists():
        tf_gene_prior = PRIORS_DIR / "tf_gene_f.full.npz"
    assert tf_gene_prior.exists(), f"Missing tf-gene prior: {tf_gene_prior}"

    tf_in_prior, tf2targets = load_tf_gene_prior(tf_gene_prior, len(tfs_vocab), N_GENE_DIM)
    genes_in_peakprior, gene2peaks = load_gene_peak_prior(PRIORS_DIR, N_GENE_DIM)
    genes_union, gid2pid_u = build_genes_union(tf_in_prior, tf2targets, genes_in_peakprior)
    log("[PRIOR] tf_in_prior =", len(tf_in_prior), "genes_union =", len(genes_union))

    # ---------- load adapter ----------
    frontier_ckpt = torch.load(FRONTIER_CKPT, map_location="cpu", weights_only=False)
    hyper = frontier_ckpt.get("hyper", {})
    K_RANK = int(hyper.get("K_RANK", 32))

    adapter_sd = frontier_ckpt.get("adapter_state_dict", None)
    if adapter_sd is None:
        adapter_sd = frontier_ckpt.get("adapter", None)
    assert adapter_sd is not None, f"Cannot find adapter state dict in {FRONTIER_CKPT}"

    if "B_group.weight" in adapter_sd:
        n_group = int(adapter_sd["B_group.weight"].shape[0])
    else:
        raise KeyError("adapter_state_dict missing B_group.weight, cannot infer n_group")

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

    cur = adapter.state_dict()
    for k, v in adapter_sd.items():
        if k in cur and tuple(cur[k].shape) == tuple(v.shape):
            cur[k] = v
    adapter.load_state_dict(cur, strict=True)
    adapter.eval()
    log("[ADAPTER] loaded K_RANK =", K_RANK, "n_group =", n_group)

    # ---------- sample ----------
    sample = load_single_sample(DATASET_NAME, SAMPLE_ID, genes_vocab)
    sid2sample = {sample.sid: sample}
    log("[SAMPLE]", sample.sid, "cells =", sample.counts_shape[0], "genes =", sample.counts_shape[1])

    # ---------- old cell-level group ids (仅作 cell adaptor 用) ----------
    cdf_old = pd.read_csv(CLUSTERS_CSV)
    cdf_old["sid"] = cdf_old["sid"].astype(str)
    cdf_old["barcode"] = cdf_old["barcode"].astype(str).map(norm_barcode)

    cdf_old = cdf_old[cdf_old["sid"] == SAMPLE_ID].copy()
    assert len(cdf_old) > 0, f"No old cluster rows found for sample={SAMPLE_ID} in {CLUSTERS_CSV}"

    keep_cols = ["sid", "barcode", "cluster_id", "group_id"]
    for c in keep_cols:
        assert c in cdf_old.columns, f"{CLUSTERS_CSV.name}: missing {c}"

    cdf_old = cdf_old[keep_cols].drop_duplicates(["sid", "barcode"]).reset_index(drop=True)

    # ---------- new Leiden clusters: now rerun directly ----------
    cdf_new = run_and_save_new_leiden_mapping(sample)
    assert len(cdf_new) > 0, f"No new Leiden mapping rows found for {SAMPLE_ID}"

    # merge
    cdf = cdf_old.merge(cdf_new, on="barcode", how="inner").copy()
    assert len(cdf) > 0, "No matched cells between old cell mapping and new Leiden mapping"

    # map to rid
    rid_list = []
    miss = 0
    for bc in cdf["barcode"].astype(str).tolist():
        rid = sample.bc2rid.get(str(bc), None)
        if rid is None:
            rid_list.append(-1)
            miss += 1
        else:
            rid_list.append(int(rid))
    cdf["rid"] = np.asarray(rid_list, np.int64)
    cdf = cdf[cdf["rid"] >= 0].copy().reset_index(drop=True)
    assert len(cdf) > 0, f"No matched cells survived after barcode->rid mapping for {SAMPLE_ID}"

    log("[MATCH] matched cells =", len(cdf), "barcode->rid missing =", miss)

    # new cluster labels
    cluster_labels = sorted(cdf["new_cluster"].astype(str).unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    cl2pos = {c: i for i, c in enumerate(cluster_labels)}
    cdf["new_cluster_pos"] = cdf["new_cluster"].astype(str).map(cl2pos).astype(np.int64)
    n_cluster = len(cluster_labels)

    # cluster sizes
    n_cells_cluster = np.bincount(cdf["new_cluster_pos"].values.astype(np.int64), minlength=n_cluster).astype(np.int64)
    cluster_size_df = pd.DataFrame({
        "cluster_label": cluster_labels,
        "cluster_pos": np.arange(n_cluster, dtype=np.int64),
        "n_cells": n_cells_cluster,
    })
    safe_to_csv(cluster_size_df, CLUSTER_SIZE_CSV)
    log("[CLUSTERS] new Leiden n_cluster =", n_cluster)
    log("[SAVE]", CLUSTER_SIZE_CSV)

    # ---------- encode Z ----------
    cell_list = [(SAMPLE_ID, int(rid)) for rid in cdf["rid"].values.astype(np.int64).tolist()]
    proj_store_cache = {}
    log("[ENC] encoding cell Z ...")
    Z = encode_Z(cell_list, sid2sample, model, DEFAULT_CANCER_ID, proj_store_cache)
    Zt = torch.from_numpy(Z).to(DEVICE)

    # ---------- apply transfer adaptor on each cell ----------
    with torch.no_grad():
        V_tf, V_gene, _, V_cancer = adapter()

        if USE_CELLGROUP_ADAPTOR:
            gid_t = torch.from_numpy(cdf["group_id"].values.astype(np.int64)).to(DEVICE)
            Zt = adapter.apply_group_to_Z(Zt, gid_t)

        if USE_CANCER_ADAPTOR:
            cid_t = torch.full((Zt.shape[0],), int(DEFAULT_CANCER_ID), device=DEVICE, dtype=torch.long)
            Zt = Zt + adapter.delta_cancer(cid_t, V_cancer)

        Zt = F.normalize(Zt, dim=-1)

    # ========================================================
    # 9) LR PAIRS
    # ========================================================
    lr = pd.read_csv(LR_UNION_CSV)
    lr["ligand"] = lr["ligand"].astype(str).str.strip()
    lr["receptor"] = lr["receptor"].astype(str).str.strip()
    if "n_sources" not in lr.columns:
        lr["n_sources"] = 1
    lr["n_sources"] = lr["n_sources"].astype(int)
    if "sources" not in lr.columns:
        lr["sources"] = ""

    lr = lr[lr["ligand"].map(looks_like_gene_symbol) & lr["receptor"].map(looks_like_gene_symbol)].copy()
    lr = lr.drop_duplicates(["ligand", "receptor"]).reset_index(drop=True)

    genes_need = sorted(set(lr["ligand"].tolist()) | set(lr["receptor"].tolist()))
    g2col = {g: i for i, g in enumerate(genes_need)}
    log("[LR] pairs =", len(lr), "genes_need =", len(genes_need))

    # ========================================================
    # 10) counts mean per new Leiden cluster
    # ========================================================
    log("[COUNTS] cluster mean log1p(counts) ...")
    counts_sum = np.zeros((n_cluster, len(genes_need)), np.float64)
    counts_n = np.zeros((n_cluster,), np.int64)

    var_names = [str(x).strip() for x in sample.gene_names.tolist()]
    name2j = {g: j for j, g in enumerate(var_names)}
    cols = np.asarray([name2j.get(g, -1) for g in genes_need], dtype=np.int64)
    ok = np.where(cols >= 0)[0]
    log("[COUNTS] genes_need matched in sample var =", int(ok.size), "/", int(len(genes_need)))

    if ok.size > 0:
        counts_mm = np.load(sample.counts_path, mmap_mode="r")
        rid_arr = cdf["rid"].values.astype(np.int64)
        clpos_arr = cdf["new_cluster_pos"].values.astype(np.int64)

        X = np.asarray(counts_mm[rid_arr[:, None], cols[ok][None, :]], dtype=np.float32)
        X = np.log1p(np.clip(X, 0.0, None))

        for cpos in np.unique(clpos_arr):
            m = (clpos_arr == cpos)
            if m.sum() == 0:
                continue
            counts_sum[cpos, ok] += X[m].sum(axis=0)
            counts_n[cpos] += int(m.sum())

    counts_mean = np.zeros((n_cluster, len(genes_need)), np.float32)
    for cpos in range(n_cluster):
        denom = max(1, int(counts_n[cpos]))
        counts_mean[cpos, :] = (counts_sum[cpos, :] / float(denom)).astype(np.float32)

    # ========================================================
    # 11) model gene score mean per new Leiden cluster
    # ========================================================
    log("[MODEL-GENE] cluster mean Z·G ...")

    gene_name2train_gid = {g: i for i, g in enumerate(genes_vocab)}
    gene_pid_list = []
    gene_train_gid_list = []
    gene_need_idx = []

    baseonly_need_idx = []
    baseonly_train_gid = []

    for j, g in enumerate(genes_need):
        gid = gene_name2train_gid.get(g, None)
        if gid is None:
            continue
        pid = gid2pid_u.get(int(gid), None)
        if pid is not None:
            gene_need_idx.append(j)
            gene_pid_list.append(int(pid))
            gene_train_gid_list.append(int(genes_union[int(pid)]))
        else:
            baseonly_need_idx.append(j)
            baseonly_train_gid.append(int(gid))

    model_sum = np.zeros((n_cluster, len(genes_need)), np.float32)
    clpos_arr = cdf["new_cluster_pos"].values.astype(np.int64)

    with torch.no_grad():
        if gene_pid_list:
            pid_t = torch.tensor(gene_pid_list, device=DEVICE, dtype=torch.long)
            gid_t = torch.tensor(gene_train_gid_list, device=DEVICE, dtype=torch.long)

            G = model.gene_out(gid_t) + model.gene_out_c(gid_t.new_full((gid_t.numel(),), int(DEFAULT_CANCER_ID)))
            G = G + adapter.delta_gene(pid_t, V_gene)

            if USE_CANCER_ADAPTOR:
                cid_g = torch.full((G.shape[0],), int(DEFAULT_CANCER_ID), device=DEVICE, dtype=torch.long)
                G = G + adapter.delta_cancer(cid_g, V_cancer)

            G = F.normalize(G, dim=-1)

            B = int(GENE_VEC_BATCH)
            for st in range(0, G.shape[0], B):
                Gc = G[st:st + B]
                Sc = (Zt @ Gc.t()).detach().cpu().numpy().astype(np.float32)
                cols_need = np.asarray(gene_need_idx[st:st + Gc.shape[0]], np.int64)
                for cpos in range(n_cluster):
                    m = (clpos_arr == cpos)
                    if m.sum() == 0:
                        continue
                    model_sum[cpos, cols_need] += Sc[m].sum(axis=0)

        if baseonly_train_gid:
            gid2_t = torch.tensor(baseonly_train_gid, device=DEVICE, dtype=torch.long)
            G2 = model.gene_out(gid2_t) + model.gene_out_c(gid2_t.new_full((gid2_t.numel(),), int(DEFAULT_CANCER_ID)))

            if USE_CANCER_ADAPTOR:
                cid_g2 = torch.full((G2.shape[0],), int(DEFAULT_CANCER_ID), device=DEVICE, dtype=torch.long)
                G2 = G2 + adapter.delta_cancer(cid_g2, V_cancer)

            G2 = F.normalize(G2, dim=-1)

            B = int(GENE_VEC_BATCH)
            for st in range(0, G2.shape[0], B):
                Gc = G2[st:st + B]
                Sc = (Zt @ Gc.t()).detach().cpu().numpy().astype(np.float32)
                cols_need = np.asarray(baseonly_need_idx[st:st + Gc.shape[0]], np.int64)
                for cpos in range(n_cluster):
                    m = (clpos_arr == cpos)
                    if m.sum() == 0:
                        continue
                    model_sum[cpos, cols_need] += Sc[m].sum(axis=0)

    model_mean = np.zeros_like(model_sum, np.float32)
    for cpos in range(n_cluster):
        denom = max(1, int(n_cells_cluster[cpos]))
        model_mean[cpos, :] = model_sum[cpos, :] / float(denom)

    model_z = np.zeros_like(model_mean, np.float32)
    counts_z = np.zeros_like(counts_mean, np.float32)
    for j in range(len(genes_need)):
        model_z[:, j] = zscore_1d(model_mean[:, j])
        counts_z[:, j] = zscore_1d(counts_mean[:, j])

    gene_ev = ((1.0 - COUNTS_WEIGHT) * model_z + COUNTS_WEIGHT * counts_z).astype(np.float32)

    # ========================================================
    # 12) TF activity per new Leiden cluster
    # ========================================================
    log("[TF] cluster TF activity ...")
    with torch.no_grad():
        tf_vocab_ids = torch.tensor([tf_in_prior[i] for i in range(len(tf_in_prior))], device=DEVICE, dtype=torch.long)
        tf_pid_t = torch.arange(len(tf_in_prior), device=DEVICE, dtype=torch.long)

        TF = model.tf_in(tf_vocab_ids) + model.tf_in_c(tf_vocab_ids.new_full((tf_vocab_ids.numel(),), int(DEFAULT_CANCER_ID)))
        TF = TF + adapter.delta_tf(tf_pid_t, V_tf)

        if USE_CANCER_ADAPTOR:
            cid_tf = torch.full((TF.shape[0],), int(DEFAULT_CANCER_ID), device=DEVICE, dtype=torch.long)
            TF = TF + adapter.delta_cancer(cid_tf, V_cancer)

        TF = F.normalize(TF, dim=-1)
        tf_cell = (Zt @ TF.t()).detach().cpu().numpy().astype(np.float32)

    tf_mean = np.zeros((n_cluster, tf_cell.shape[1]), np.float32)
    for cpos in range(n_cluster):
        m = (clpos_arr == cpos)
        if m.sum() == 0:
            continue
        tf_mean[cpos, :] = tf_cell[m].mean(axis=0)

    tf_z = np.zeros_like(tf_mean, np.float32)
    for j in range(tf_mean.shape[1]):
        tf_z[:, j] = zscore_1d(tf_mean[:, j])

    # ========================================================
    # 13) build sharper receptor -> TF signature (基于新 Leiden)
    # ========================================================
    log("[TF-SIGNATURE] build receptor -> TF signature on new Leiden clusters ...")

    receptors_unique = sorted(set(lr["receptor"].tolist()))
    receptor2tf = {}
    sig_rows = []

    for rec in receptors_unique:
        j = g2col.get(rec, None)
        if j is None:
            continue

        xz = zscore_1d(gene_ev[:, j])
        corr = (tf_z * xz[:, None]).mean(axis=0).astype(np.float32)
        order = np.argsort(corr)[::-1]

        pick = []
        for idx in order[:max(TF_SIG_TOPM * 3, TF_SIG_TOPM)]:
            if corr[idx] >= TF_SIG_MIN_CORR:
                pick.append(int(idx))
            if len(pick) >= TF_SIG_TOPM:
                break

        if len(pick) == 0:
            continue

        receptor2tf[rec] = np.asarray(pick, np.int64)
        for rank, tidx in enumerate(pick, start=1):
            sig_rows.append({
                "receptor": rec,
                "rank": int(rank),
                "tf_idx_in_prior": int(tidx),
                "tf_name": str(tfs_vocab[int(tf_in_prior[int(tidx)])]),
                "corr": float(corr[tidx]),
            })

    df_sig = pd.DataFrame(sig_rows)
    safe_to_csv(df_sig, SIG_CSV)
    log("[TF-SIGNATURE] receptors with signature:", len(receptor2tf), "/", len(receptors_unique))
    log("[SAVE]", SIG_CSV)

    tf_fallback = np.sort(tf_z, axis=1)[:, -min(TF_FALLBACK_TOPK, tf_z.shape[1]):].mean(axis=1).astype(np.float32)
    tf_fallback_z = zscore_1d(tf_fallback)

    # ========================================================
    # 14) CCI scoring on new Leiden clusters
    # ========================================================
    log("[CCI] scoring interactions on new Leiden clusters ...")

    cluster_list = cluster_labels[:]
    Lsig_cache = {}
    Rsig_cache = {}

    def expr_sig(gname: str) -> np.ndarray:
        j = g2col.get(gname, None)
        if j is None:
            return np.zeros((n_cluster,), np.float32)
        return sigmoid_np(gene_ev[:, j])

    def conf_size(ns: int, nr: int, nsrc: int) -> float:
        x = 0.55 * (math.log1p(ns) + math.log1p(nr)) + 0.9 * max(0, nsrc - 1)
        return float(sigmoid_np(np.array([x - 6.0], np.float32))[0])

    out_rows = []

    for _, r in lr.iterrows():
        lig = str(r["ligand"])
        rec = str(r["receptor"])
        nsrc = int(r["n_sources"])
        sources = str(r["sources"])

        if lig not in Lsig_cache:
            Lsig_cache[lig] = expr_sig(lig)
        if rec not in Rsig_cache:
            Rsig_cache[rec] = expr_sig(rec)

        Ls = Lsig_cache[lig]
        Rs = Rsig_cache[rec]

        tf_idx = receptor2tf.get(rec, None)
        if tf_idx is None:
            resp_z = tf_fallback_z
            tf_mode = "fallback_topTF"
            tf_set_size = 0
        else:
            resp = tf_z[:, tf_idx].mean(axis=1).astype(np.float32)
            resp_z = zscore_1d(resp)
            tf_mode = "receptor_signature"
            tf_set_size = int(tf_idx.size)

        gate = tf_gate(resp_z)
        kept = (resp_z >= float(TF_KEEP_Z)).astype(np.int8)

        for isrc, csrc in enumerate(cluster_list):
            for itgt, ctgt in enumerate(cluster_list):
                expr = float(Ls[isrc] * Rs[itgt])
                score = float(expr * gate[itgt])
                conf = conf_size(int(n_cells_cluster[isrc]), int(n_cells_cluster[itgt]), nsrc)

                out_rows.append({
                    "ligand": lig,
                    "receptor": rec,
                    "sources": sources,
                    "n_sources": nsrc,

                    "sender_cluster": str(csrc),
                    "receiver_cluster": str(ctgt),
                    "sender_cluster_pos": int(isrc),
                    "receiver_cluster_pos": int(itgt),

                    "expr_only": expr,
                    "receiver_tf_resp_z": float(resp_z[itgt]),
                    "tf_gate": float(gate[itgt]),
                    "kept_by_tf": int(kept[itgt]),
                    "tf_mode": tf_mode,
                    "tf_set_size": tf_set_size,

                    "score": score,
                    "confidence": conf,

                    "sender_cells": int(n_cells_cluster[isrc]),
                    "receiver_cells": int(n_cells_cluster[itgt]),

                    "ligand_ev_z": float(gene_ev[isrc, g2col[lig]]) if lig in g2col else float("nan"),
                    "receptor_ev_z": float(gene_ev[itgt, g2col[rec]]) if rec in g2col else float("nan"),
                })

    df_int = pd.DataFrame(out_rows).sort_values(["score", "confidence"], ascending=[False, False]).reset_index(drop=True)
    safe_to_csv(df_int, INT_CSV)
    log("[SAVE]", INT_CSV, "rows =", len(df_int))

    df_hi = df_int[df_int["n_sources"] >= HIGHCONF_MIN_SOURCES].copy()
    safe_to_csv(df_hi, INT_HI_CSV)
    log("[SAVE]", INT_HI_CSV, "rows =", len(df_hi))

    df_hi_tf = df_hi[df_hi["kept_by_tf"] == 1].copy()
    safe_to_csv(df_hi_tf, INT_HI_TF_CSV)
    log("[SAVE]", INT_HI_TF_CSV, "rows =", len(df_hi_tf))

    # ========================================================
    # 15) cluster profiles
    # ========================================================
    tf_names = np.array([tfs_vocab[int(tf_in_prior[i])] for i in range(len(tf_in_prior))], dtype=object)

    prof = []
    for i, clab in enumerate(cluster_list):
        tv = tf_z[i, :]
        topk = min(10, tv.size)
        idx = np.argsort(tv)[::-1][:topk]

        top_tfs = [(str(tf_names[j]), float(tv[j])) for j in idx.tolist()]

        tf_explain = []
        for j in idx[:5]:
            tf_vocab_id = int(tf_in_prior[int(j)])
            tg = tf2targets.get(tf_vocab_id, [])[:12]
            tg_names = [genes_vocab[g] for g in tg if 0 <= int(g) < len(genes_vocab)]
            tf_explain.append({
                "tf": str(tf_names[int(j)]),
                "tf_z": float(tv[int(j)]),
                "targets_head": tg_names,
            })

        gv = gene_ev[i, :]
        gidx = np.argsort(gv)[::-1][:15]
        top_lr_genes = [(genes_need[j], float(gv[j])) for j in gidx.tolist()]

        prof.append({
            "cluster_label": str(clab),
            "cluster_pos": int(i),
            "n_cells": int(n_cells_cluster[i]),
            "fallback_tf_resp_z": float(tf_fallback_z[i]),
            "top_tfs": json.dumps(top_tfs, ensure_ascii=False),
            "top_tf_targets": json.dumps(tf_explain, ensure_ascii=False),
            "top_lr_genes": json.dumps(top_lr_genes, ensure_ascii=False),
        })

    df_prof = pd.DataFrame(prof)
    safe_to_csv(df_prof, PROF_CSV)
    log("[SAVE]", PROF_CSV)

    log("[DONE] out_dir =", OUT_DIR.resolve())


if __name__ == "__main__":
    main()