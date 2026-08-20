from __future__ import annotations

# ============================================================
# crc_sample9_newcluster_roc_rna2rna_aligned_to_true_umap.py
# ------------------------------------------------------------
# 只评估单样本：CRC_RNA_TO009
#
# 目标：
# 1) 用 frontier_best.pt 做 RNA embedding 推断
# 2) 只看 sample9（CRC_RNA_TO009）
# 3) 重聚类流程严格对齐 true RNA UMAP 脚本：
#    - filter_genes(min_cells=3)
#    - HVG: seurat_v3, subset=False
#    - normalize_total + log1p
#    - 只保留 HVG
#    - scale / PCA(arpack) / neighbors / UMAP / Leiden(res=0.5)
# 4) 画两张图：
#    - overall ROC（RNA→RNA gene retrieval）
#    - per-new-cluster ROC (overlay)
# 5) 额外要求：
#    - 路径统一改成 E 盘，并使用 Linux 格式路径：/mnt/e/...
#    - 除了 PNG，还同时输出 SVG
#
# 说明：
# - 模型推理仍然使用训练期 internal cluster/group/correction
# - 展示 cluster ROC 时，使用 sample9 重新聚类后的 new Leiden cluster
# - RNA→RNA 评估方式参考 qc_plot_roc_rna2rna_val_like_monitor.py：
#     * 每个 cell 作为一个 retrieval task
#     * positives = 该细胞 top expressed genes
#     * negatives = expressed-but-not-positive 的 hard negatives（不足则随机补）
#     * score = dot(Z_cell_adj, GeneVec)
# ============================================================

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

OUT_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_newcluster_roc_rna2rna_aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

RNA_TOPK = 2048
EMB_BATCH = 256

STORE_TOP_G = 512
QC_GENE_CELL_EVAL = 1200
QC_GENE_POS = 32
QC_GENE_NEG = 128
QC_GENE_HARD_NEG = True

OVERALL_MIN_VALID_CELLS = 50
CLUSTER_MIN_VALID_CELLS = 12

N_TOP_HVG = 2000
N_PCS = 20
N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.03
UMAP_SPREAD = 2.0
LEIDEN_RESOLUTION = 0.5

FPR_GRID_N = 400
SMOOTH_SIGMA = 1.6

FIG_DPI = 220
POINT_SIZE_UMAP = 6
MAX_CLUSTERS_TO_PLOT = 20

NEW_CLUSTER_ASSIGN_CSV = OUT_DIR / f"{SAMPLE_ID}_new_leiden_cluster_assignments.csv"
NEW_CLUSTER_SUMMARY_CSV = OUT_DIR / f"{SAMPLE_ID}_new_leiden_cluster_summary_rna2rna.csv"

UMAP_NEW_CLUSTER_PNG = OUT_DIR / f"{SAMPLE_ID}_umap_new_leiden_res0p5.png"
UMAP_NEW_CLUSTER_SVG = OUT_DIR / f"{SAMPLE_ID}_umap_new_leiden_res0p5.svg"

ROC_OVERALL_PNG = OUT_DIR / f"{SAMPLE_ID}_roc_overall_rna2rna.png"
ROC_OVERALL_SVG = OUT_DIR / f"{SAMPLE_ID}_roc_overall_rna2rna.svg"

ROC_CLUSTER_OVERLAY_PNG = OUT_DIR / f"{SAMPLE_ID}_roc_newclusters_overlay_rna2rna.png"
ROC_CLUSTER_OVERLAY_SVG = OUT_DIR / f"{SAMPLE_ID}_roc_newclusters_overlay_rna2rna.svg"

CELL_AUC_CSV = OUT_DIR / f"{SAMPLE_ID}_rna2rna_cell_aucs_overall.csv"
CELL_AUC_WITH_CLUSTER_CSV = OUT_DIR / f"{SAMPLE_ID}_rna2rna_cell_aucs_with_newcluster.csv"


def log(*args):
    print(*args, flush=True)


def save_figure_both(fig, png_path: Path, svg_path: Path):
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    log("[SAVE]", png_path)
    log("[SAVE]", svg_path)


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


def roc_curve_binary(y01: np.ndarray, scores: np.ndarray):
    y = np.asarray(y01, np.int8).reshape(-1)
    s = np.asarray(scores, np.float64).reshape(-1)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if y.size < 20:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P < 1 or N < 1:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-s)
    y = y[order]
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

    def delta_gene(self, gene_pid: torch.Tensor, V_gene: torch.Tensor) -> torch.Tensor:
        return self.C_gene(gene_pid) @ V_gene

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
    for c in ["cell_id", "sample_id", "row_index"]:
        assert c in df.columns, f"manifest missing {c}"

    g = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    assert len(g) > 0, f"No rows found for sample_id={sample_id} in {manifest_path}"
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

    matched = int((ext2train >= 0).sum())
    assert matched > 1000, f"{sample_id}: too few genes matched to global vocab: {matched}"

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
    )
    log(f"[LOAD] {sample_id} | cells={s.counts_shape[0]} genes={s.counts_shape[1]} matched_genes={matched}")
    return s


def get_counts_row_dense(sample: Sample, i: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    row = np.asarray(x[i]).reshape(-1)
    return row.astype(np.float32, copy=False)


# ============================================================
# 4) ENCODING / INTERNAL CLUSTER
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


def spherical_kmeans_assign(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    ZN = _l2norm_rows(Z)
    CN = _l2norm_rows(C)
    sim = ZN @ CN.T
    return sim.argmax(axis=1).astype(np.int64)


# ============================================================
# 5) NEW SINGLE-SAMPLE CLUSTERING
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


def save_newcluster_umap(adata: ad.AnnData, out_png: Path, out_svg: Path):
    xy = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    lab = adata.obs["new_cluster"].astype(str).values
    uniq = sorted(pd.unique(lab).tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    fig = plt.figure(figsize=(7.2, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)

    for c in uniq:
        m = (lab == c)
        ax.scatter(
            xy[m, 0], xy[m, 1],
            s=POINT_SIZE_UMAP,
            alpha=0.85,
            label=f"C{c}"
        )

    for c in uniq:
        m = (lab == c)
        if int(m.sum()) == 0:
            continue
        cx = float(np.median(xy[m, 0]))
        cy = float(np.median(xy[m, 1]))
        ax.text(
            cx, cy, str(c),
            fontsize=8,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax.set_title(f"True RNA UMAP | {SAMPLE_ID} | Leiden(res={LEIDEN_RESOLUTION})", fontsize=13)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()

    save_figure_both(fig, out_png, out_svg)
    plt.close(fig)


# ============================================================
# 6) RNA->RNA EVAL HELPERS
# ============================================================
def precompute_top_genes_single_sample(sample: Sample, cell_list: list[tuple[str, int]], n_gene_dim: int, store_top_g: int = 512):
    G_top = np.full((len(cell_list), store_top_g), -1, dtype=np.int64)
    V_top = np.zeros((len(cell_list), store_top_g), dtype=np.float32)

    for i, (_sid, rid) in enumerate(cell_list):
        row = get_counts_row_dense(sample, rid)
        gi, gv = topk_from_dense_row(row, store_top_g)
        train_idx = sample.ext2train[gi]
        m = (train_idx >= 0) & (train_idx < n_gene_dim)
        train_idx = train_idx[m]
        gv = gv[m]
        k = min(store_top_g, train_idx.size)
        if k > 0:
            G_top[i, :k] = train_idx[:k]
            V_top[i, :k] = gv[:k]
    return G_top, V_top


def aggregate_mean_roc_from_cell_tasks(pred_pos_list, pred_neg_list, min_valid_cells: int):
    grid = np.linspace(0, 1, FPR_GRID_N, dtype=np.float64)
    tpr_stack = []
    cell_aucs = []

    for spv, snv in zip(pred_pos_list, pred_neg_list):
        spv = np.asarray(spv, dtype=np.float32).reshape(-1)
        snv = np.asarray(snv, dtype=np.float32).reshape(-1)

        if spv.size < 8 or snv.size < 32:
            continue

        auc0 = auc_from_scores(spv, snv)
        if not np.isfinite(auc0):
            continue

        y = np.concatenate([np.ones(spv.size, np.int8), np.zeros(snv.size, np.int8)])
        scv = np.concatenate([spv.astype(np.float64), snv.astype(np.float64)])
        fpr, tpr = roc_curve_binary(y, scv)
        tpr_stack.append(interp_to_grid(fpr, tpr, grid))
        cell_aucs.append(float(auc0))

    if len(cell_aucs) < int(min_valid_cells):
        return None

    cell_aucs = np.asarray(cell_aucs, np.float32)
    tpr_mean = np.mean(np.stack(tpr_stack, axis=0), axis=0)
    tpr_mean = smooth_1d(tpr_mean, sigma=SMOOTH_SIGMA)
    tpr_mean = fix_roc_endpoints_and_monotonic(tpr_mean)

    return {
        "grid": grid,
        "tpr_mean": tpr_mean,
        "mean_auc": float(np.nanmean(cell_aucs)),
        "median_auc": float(np.nanmedian(cell_aucs)),
        "used_cells": int(cell_aucs.size),
        "cell_aucs": cell_aucs,
    }


# ============================================================
# 7) PRIOR READERS
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
    """
    这里严格对齐原 frontier 脚本：
    - 只检查 gene id 是否合法
    - 不要过滤 peak id 是否 < PEAK_HASH_SIZE
    因为模型里是 hash peak，原始 peak id 会在 forward 时做取模。
    """
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
# 8) MAIN
# ============================================================
def main():
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing {GLOBAL_TF_VOCAB}"
    assert FRONTIER_CKPT.exists(), f"Missing {FRONTIER_CKPT}"
    assert CENTROIDS_NPY.exists(), f"Missing {CENTROIDS_NPY}"
    assert CLUSTERS_CSV.exists(), f"Missing {CLUSTERS_CSV}"
    assert BASE_CKPT.exists(), f"Missing {BASE_CKPT}"

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = dict(base_ckpt.get("cfg", {}) or {})
    meta = base_ckpt.get("meta", {})
    state_dict = base_ckpt["model"]

    log("[BASE_CKPT] cfg keys =", list((base_ckpt.get("cfg", {}) or {}).keys()))
    log("[BASE_CKPT] meta keys =", list(meta.keys()) if isinstance(meta, dict) else None)

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
    default_cancer_name = "CRC"
    default_cancer_id = int(cancer2id.get(default_cancer_name, 0))
    cid = torch.tensor([default_cancer_id], device=DEVICE, dtype=torch.long)

    log(
        "[BASE_CFG]",
        f"N_GENE_DIM={N_GENE_DIM}",
        f"N_TF_DIM={N_TF_DIM}",
        f"PEAK_HASH_SIZE={PEAK_HASH_SIZE}",
        f"D={D}",
        f"HID={HID}",
        f"DROP={DROP}",
        f"n_cancer={len(cancer_types)}",
        f"default_cancer_name={default_cancer_name}",
        f"default_cancer_id={default_cancer_id}",
    )

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

    ckpt = torch.load(FRONTIER_CKPT, map_location="cpu", weights_only=False)
    hyper = ckpt.get("hyper", {})

    K_RANK = int(hyper.get("K_RANK", 32))
    GROUP_SCALE_ALPHA = float(hyper.get("GROUP_SCALE_ALPHA", 0.12))
    USE_CLUSTER_CORRECTION = bool(hyper.get("USE_CLUSTER_CORRECTION", True))
    CLUSTER_CORRECTION_USE_RAW_CLUSTER = bool(hyper.get("CLUSTER_CORRECTION_USE_RAW_CLUSTER", True))

    cluster_summary_vecs = ckpt["cluster_summary_vecs"]
    cluster_summary_vecs = {int(k): np.asarray(v, dtype=np.float32) for k, v in cluster_summary_vecs.items()}
    cluster_summary_table = ClusterSummaryTable(cluster_summary_vecs).to(DEVICE)

    df_clusters = pd.read_csv(CLUSTERS_CSV)
    need_cols = {"cluster_id", "group_id"}
    assert need_cols.issubset(df_clusters.columns), f"{CLUSTERS_CSV} missing {need_cols - set(df_clusters.columns)}"

    cl2gid_counter = defaultdict(Counter)
    for _, r in df_clusters.iterrows():
        cl2gid_counter[int(r["cluster_id"])][int(r["group_id"])] += 1

    cl2gid = {}
    for cl, cnt in cl2gid_counter.items():
        cl2gid[int(cl)] = cnt.most_common(1)[0][0]

    n_group = int(max(cl2gid.values()) + 1) if len(cl2gid) else 1
    log(f"[GROUP] inferred n_group={n_group}")

    tf2targets, tf_in_prior = read_tf_prior(PRIORS_DIR, tfs_vocab, N_GENE_DIM)

    gene2peaks, genes_union = read_gene_peak_prior(PRIORS_DIR, N_GENE_DIM)
    gid2pid_u = {int(gid): i for i, gid in enumerate(genes_union)}
    log("[PRIOR] genes_union =", len(genes_union))

    ck_gene_rows = int(ckpt["adapter_state_dict"]["C_gene.weight"].shape[0])
    log("[CHECK] len(genes_union) =", len(genes_union), "| ckpt C_gene =", ck_gene_rows)
    assert len(genes_union) == ck_gene_rows, \
        f"gene vocab mismatch: len(genes_union)={len(genes_union)} vs ckpt C_gene={ck_gene_rows}"

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

    def cluster_ids_to_summary(cluster_ids_np: np.ndarray):
        cid_t = torch.from_numpy(cluster_ids_np.astype(np.int64)).to(DEVICE)
        return cluster_summary_table.lookup(cid_t)

    def apply_cluster_correction(z_in: torch.Tensor, cluster_ids_np: np.ndarray, alpha: float = 0.25):
        if tf_corr_head is None:
            return z_in
        summary_t = cluster_ids_to_summary(cluster_ids_np)
        delta = tf_corr_head(z_in, summary_t)
        return F.normalize(z_in + float(alpha) * delta, dim=-1)

    @torch.no_grad()
    def score_gene_vec(gene_pid_t: torch.Tensor, V_gene: torch.Tensor, V_cancer: torch.Tensor) -> torch.Tensor:
        gene_ids = torch.tensor([genes_union[i] for i in gene_pid_t.tolist()], device=DEVICE, dtype=torch.long)
        G = model.gene_out(gene_ids) + model.gene_out_c(cid).expand(gene_ids.numel(), -1)
        G = G + adapter.delta_gene(gene_pid_t, V_gene)
        c_tok = torch.full((gene_pid_t.numel(),), int(default_cancer_id), device=DEVICE, dtype=torch.long)
        G = G + adapter.delta_cancer(c_tok, V_cancer)
        return G

    sample = load_single_sample(DATASET_NAME, SAMPLE_ID, genes_vocab)
    sid2sample = {sample.sid: sample}
    cell_list = [(sample.sid, int(i)) for i in range(sample.counts_shape[0])]
    log(f"[CELLS] sample={SAMPLE_ID} n_cells={len(cell_list)}")

    proj_store_cache = {}
    log("[ENC] encoding Z ...")
    Z = encode_Z(cell_list, sid2sample, model, default_cancer_id, proj_store_cache)
    Zt = torch.from_numpy(Z).to(DEVICE)

    C = np.load(CENTROIDS_NPY).astype(np.float32)
    raw_cluster = spherical_kmeans_assign(Z, C)
    group_ids = np.asarray([cl2gid.get(int(c), 0) for c in raw_cluster.tolist()], dtype=np.int64)

    with torch.no_grad():
        _, V_gene, _, V_cancer = adapter()

        gid_t = torch.from_numpy(group_ids.astype(np.int64)).to(DEVICE)
        cid_t = torch.full((len(cell_list),), int(default_cancer_id), device=DEVICE, dtype=torch.long)

        Zadj = Zt
        Zadj = adapter.apply_group_to_Z(Zadj, gid_t, GROUP_SCALE_ALPHA)
        Zadj = Zadj + adapter.delta_cancer(cid_t, V_cancer)
        Zadj = F.normalize(Zadj, dim=-1)

        cluster_for_corr = raw_cluster if CLUSTER_CORRECTION_USE_RAW_CLUSTER else group_ids
        Zadj = apply_cluster_correction(Zadj, cluster_for_corr, alpha=0.25)

    log("[SCANPY] building new single-sample clustering aligned to true UMAP ...")
    adata = build_adata_for_scanpy(sample)
    adata, _best_res, _best_k = run_single_sample_leiden(adata)
    save_newcluster_umap(adata, UMAP_NEW_CLUSTER_PNG, UMAP_NEW_CLUSTER_SVG)

    key_order_model = np.asarray([make_cell_key(sample.sid, sample.cells[i]) for _, i in cell_list], dtype=object)
    df_obs = pd.DataFrame({
        "cell_key": adata.obs_names.astype(str).tolist(),
        "new_cluster": adata.obs["new_cluster"].astype(str).tolist(),
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
    log("[SAVE]", NEW_CLUSTER_ASSIGN_CSV)

    log("[RNA2RNA] precompute top genes ...")
    G_top, _ = precompute_top_genes_single_sample(sample, cell_list, N_GENE_DIM, store_top_g=STORE_TOP_G)

    n_cells = len(cell_list)
    n_eval = min(QC_GENE_CELL_EVAL, n_cells)

    h = int(hashlib.md5(str(SAMPLE_ID).encode("utf-8")).hexdigest()[:8], 16)
    rng_local = np.random.RandomState(SEED ^ h)
    idx_cells = rng_local.choice(np.arange(n_cells), size=n_eval, replace=False)
    idx_cells = np.sort(idx_cells)

    cell_pos_scores = []
    cell_neg_scores = []
    kept_rows = []

    log(f"[RNA2RNA] building cell tasks ... n_eval={n_eval}")

    for ci in idx_cells.tolist():
        _sid, rid = cell_list[ci]
        row = get_counts_row_dense(sample, rid)
        expressed = np.where(row > 0)[0].astype(np.int64)

        g_all = G_top[ci]
        valid = g_all >= 0
        if int(valid.sum()) < 16:
            continue
        g = g_all[valid].astype(np.int64, copy=False)

        ppos_gid = g[:QC_GENE_POS]
        ppos = np.array([gid2pid_u.get(int(x), -1) for x in ppos_gid.tolist()], dtype=np.int64)
        ppos = ppos[ppos >= 0]
        if ppos.size < 8:
            continue

        if QC_GENE_HARD_NEG:
            train_idx = sample.ext2train[expressed]
            train_idx = train_idx[(train_idx >= 0) & (train_idx < N_GENE_DIM)]
            pids_expr = np.array([gid2pid_u.get(int(x), -1) for x in train_idx.tolist()], dtype=np.int64)
            pids_expr = pids_expr[pids_expr >= 0]
            pids_expr = np.setdiff1d(pids_expr, np.unique(ppos), assume_unique=False)
            if pids_expr.size < QC_GENE_NEG:
                pneg = rng_local.choice(np.arange(len(genes_union)), size=QC_GENE_NEG, replace=False)
            else:
                pneg = rng_local.choice(pids_expr, size=QC_GENE_NEG, replace=False)
        else:
            pneg = rng_local.choice(np.arange(len(genes_union)), size=QC_GENE_NEG, replace=False)

        pos_t = torch.from_numpy(ppos.astype(np.int64)).to(DEVICE)
        neg_t = torch.from_numpy(pneg.astype(np.int64)).to(DEVICE)

        with torch.no_grad():
            Gpos = score_gene_vec(pos_t, V_gene, V_cancer)
            Gneg = score_gene_vec(neg_t, V_gene, V_cancer)
            zc = Zadj[ci:ci + 1]

            s_pos = (zc @ Gpos.t()).reshape(-1).detach().cpu().numpy().astype(np.float32)
            s_neg = (zc @ Gneg.t()).reshape(-1).detach().cpu().numpy().astype(np.float32)

        if s_pos.size < 8 or s_neg.size < 32:
            continue

        auc0 = auc_from_scores(s_pos, s_neg)
        if not np.isfinite(auc0):
            continue

        cell_pos_scores.append(s_pos)
        cell_neg_scores.append(s_neg)
        kept_rows.append({
            "cell_index": int(ci),
            "sid": sample.sid,
            "rid": int(rid),
            "barcode": str(sample.cells[rid]),
            "cell_key": make_cell_key(sample.sid, str(sample.cells[rid])),
            "n_pos": int(s_pos.size),
            "n_neg": int(s_neg.size),
            "cell_auc": float(auc0),
        })

    assert len(cell_pos_scores) >= OVERALL_MIN_VALID_CELLS, \
        f"Too few valid cells for overall ROC: {len(cell_pos_scores)}"

    df_cell_auc = pd.DataFrame(kept_rows)
    df_cell_auc.to_csv(CELL_AUC_CSV, index=False)
    log("[SAVE]", CELL_AUC_CSV)

    overall = aggregate_mean_roc_from_cell_tasks(
        pred_pos_list=cell_pos_scores,
        pred_neg_list=cell_neg_scores,
        min_valid_cells=OVERALL_MIN_VALID_CELLS,
    )
    assert overall is not None, "No valid cells passed overall RNA->RNA ROC filters."

    log(
        f"[OVERALL][RNA2RNA] mean_auc={overall['mean_auc']:.4f}, "
        f"median_auc={overall['median_auc']:.4f}, used_cells={overall['used_cells']}"
    )

    fig = plt.figure(figsize=(6.2, 6.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.plot(
        overall["grid"], overall["tpr_mean"],
        linewidth=2.8,
        label=f"Overall AUC={overall['mean_auc']:.3f} (cells used={overall['used_cells']})"
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.65, label="Random")
    ax.set_title(f"ROC (RNA→RNA, {SAMPLE_ID})", fontsize=13)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure_both(fig, ROC_OVERALL_PNG, ROC_OVERALL_SVG)
    plt.close(fig)

    df_cell_auc = df_cell_auc.merge(
        df_join[["cell_key", "new_cluster", "umap1", "umap2"]],
        on="cell_key",
        how="left",
    )
    assert df_cell_auc["new_cluster"].notna().all(), "cell auc cluster alignment failed"
    df_cell_auc.to_csv(CELL_AUC_WITH_CLUSTER_CSV, index=False)
    log("[SAVE]", CELL_AUC_WITH_CLUSTER_CSV)

    rows_summary = []
    curves = []

    uniq_clusters = sorted(
        pd.unique(df_join["new_cluster"].astype(str).values).tolist(),
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    grouped = defaultdict(list)
    for i, rr in df_cell_auc.reset_index(drop=True).iterrows():
        grouped[str(rr["new_cluster"])].append(i)

    for cl in uniq_clusters:
        idxs = grouped.get(str(cl), [])
        ncell = len(idxs)

        pos_list = [cell_pos_scores[i] for i in idxs]
        neg_list = [cell_neg_scores[i] for i in idxs]

        res = aggregate_mean_roc_from_cell_tasks(
            pred_pos_list=pos_list,
            pred_neg_list=neg_list,
            min_valid_cells=CLUSTER_MIN_VALID_CELLS,
        )

        if res is None:
            rows_summary.append({
                "new_cluster": str(cl),
                "n_valid_cells": int(ncell),
                "mean_auc": np.nan,
                "median_auc": np.nan,
                "used_cells": 0,
                "kept_for_plot": 0,
            })
            continue

        rows_summary.append({
            "new_cluster": str(cl),
            "n_valid_cells": int(ncell),
            "mean_auc": float(res["mean_auc"]),
            "median_auc": float(res["median_auc"]),
            "used_cells": int(res["used_cells"]),
            "kept_for_plot": 1,
        })
        curves.append((str(cl), int(res["used_cells"]), float(res["mean_auc"]), res["grid"], res["tpr_mean"]))

    df_sum = pd.DataFrame(rows_summary)
    df_sum.to_csv(NEW_CLUSTER_SUMMARY_CSV, index=False)
    log("[SAVE]", NEW_CLUSTER_SUMMARY_CSV)

    curves = sorted(curves, key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    if len(curves) > MAX_CLUSTERS_TO_PLOT:
        curves = curves[:MAX_CLUSTERS_TO_PLOT]

    fig = plt.figure(figsize=(6.5, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    for cl, used_cells, auc_m, g, tpr_m in curves:
        ax.plot(g, tpr_m, linewidth=2.0, label=f"C{cl} (cells={used_cells}) AUC={auc_m:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, alpha=0.65, label="Random")
    ax.set_title(f"ROC (RNA→RNA, {SAMPLE_ID}, new Leiden clusters)", fontsize=13)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()
    save_figure_both(fig, ROC_CLUSTER_OVERLAY_PNG, ROC_CLUSTER_OVERLAY_SVG)
    plt.close(fig)

    log("[DONE]")


if __name__ == "__main__":
    main()
