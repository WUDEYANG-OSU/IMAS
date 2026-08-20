from __future__ import annotations

# ============================================================
# crc_sample9_rna_tf_top10_network_heatmap_with_qc.py
# ------------------------------------------------------------
# 单样本：CRC_RNA_TO009
#
# 目标：
# 1) 读取已经算好的 Top10 RNA 与 Top10 TF 名单
# 2) 对每个 RNA-TF pair 计算 network-aware score
# 3) score 不采用单纯相关性，而是结合：
#    - RNA expression vs predicted TF score 的 Spearman
#    - tf-gene prior 支持
#    - 模型内部 gene-TF latent edge weight（embedding cosine）
# 4) 对行列做整体高分导向的重排
# 5) 输出热图（类似 cluster-style matrix）
# 6) 追加 4 个轻量 QC：
#    A. score decomposition
#    B. raw vs reordered heatmap
#    C. raw score vs minmax score rank consistency
#    D. signed corr vs abs(corr)
#
# 输入：
#   RUN_BASE/single_sample_<SAMPLE_ID>_network_aware_top_rna/<SAMPLE_ID>_final_top10_rnas.csv
#   RUN_BASE/single_sample_<SAMPLE_ID>_network_aware_top_tf/<SAMPLE_ID>_final_top10_tfs.csv
#
# 输出：
#   OUT_DIR/
#     <SAMPLE_ID>_top10_rna_tf_network_score_matrix.csv
#     <SAMPLE_ID>_top10_rna_tf_network_score_heatmap.png
#     <SAMPLE_ID>_top10_rna_tf_network_score_heatmap.svg
#
#   OUT_DIR/qc/
#     <SAMPLE_ID>_qc_score_decomposition.*
#     <SAMPLE_ID>_qc_raw_heatmap.*
#     <SAMPLE_ID>_qc_raw_vs_reordered_heatmap.*
#     <SAMPLE_ID>_qc_raw_vs_minmax_rank.*
#     <SAMPLE_ID>_qc_signed_vs_abs_corr.*
#     <SAMPLE_ID>_qc_signed_corr_heatmap.*
# ============================================================

import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

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

RNA_TOP_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_network_aware_top_rna"
TF_TOP_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_network_aware_top_tf"

FINAL_TOP10_RNA_CSV = RNA_TOP_DIR / f"{SAMPLE_ID}_final_top10_rnas.csv"
FINAL_TOP10_TF_CSV = TF_TOP_DIR / f"{SAMPLE_ID}_final_top10_tfs.csv"

OUT_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_rna_tf_top10_network_heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATRIX_CSV = OUT_DIR / f"{SAMPLE_ID}_top10_rna_tf_network_score_matrix.csv"
OUT_HEATMAP_PNG = OUT_DIR / f"{SAMPLE_ID}_top10_rna_tf_network_score_heatmap.png"
OUT_HEATMAP_SVG = OUT_DIR / f"{SAMPLE_ID}_top10_rna_tf_network_score_heatmap.svg"

# ---------------- QC outputs ----------------
QC_DIR = OUT_DIR / "qc"
QC_DIR.mkdir(parents=True, exist_ok=True)

QC_SCORE_DECOMP_PNG = QC_DIR / f"{SAMPLE_ID}_qc_score_decomposition.png"
QC_SCORE_DECOMP_SVG = QC_DIR / f"{SAMPLE_ID}_qc_score_decomposition.svg"
QC_SCORE_DECOMP_CSV = QC_DIR / f"{SAMPLE_ID}_qc_score_decomposition_table.csv"

QC_RAW_HEATMAP_PNG = QC_DIR / f"{SAMPLE_ID}_qc_raw_heatmap.png"
QC_RAW_HEATMAP_SVG = QC_DIR / f"{SAMPLE_ID}_qc_raw_heatmap.svg"
QC_RAW_MATRIX_CSV = QC_DIR / f"{SAMPLE_ID}_qc_raw_matrix.csv"

QC_REORDER_COMPARE_PNG = QC_DIR / f"{SAMPLE_ID}_qc_raw_vs_reordered_heatmap.png"
QC_REORDER_COMPARE_SVG = QC_DIR / f"{SAMPLE_ID}_qc_raw_vs_reordered_heatmap.svg"

QC_MINMAX_COMPARE_PNG = QC_DIR / f"{SAMPLE_ID}_qc_raw_vs_minmax_rank.png"
QC_MINMAX_COMPARE_SVG = QC_DIR / f"{SAMPLE_ID}_qc_raw_vs_minmax_rank.svg"
QC_MINMAX_COMPARE_TXT = QC_DIR / f"{SAMPLE_ID}_qc_raw_vs_minmax_rank_summary.txt"

QC_SIGNED_COMPARE_PNG = QC_DIR / f"{SAMPLE_ID}_qc_signed_vs_abs_corr.png"
QC_SIGNED_COMPARE_SVG = QC_DIR / f"{SAMPLE_ID}_qc_signed_vs_abs_corr.svg"
QC_SIGNED_COMPARE_CSV = QC_DIR / f"{SAMPLE_ID}_qc_signed_vs_abs_corr_pairs.csv"
QC_SIGNED_HEATMAP_PNG = QC_DIR / f"{SAMPLE_ID}_qc_signed_corr_heatmap.png"
QC_SIGNED_HEATMAP_SVG = QC_DIR / f"{SAMPLE_ID}_qc_signed_corr_heatmap.svg"

RNA_TOPK = 2048
EMB_BATCH = 256

# pair score 权重
W_CORR = 0.45
W_EDGE = 0.40
W_PRIOR = 0.15

# 排序时对整体高分的偏好
REORDER_LAMBDA = 0.65
N_REORDER_ITERS = 12

FIG_DPI = 220

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


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo + 1e-12)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    xs = x[order]

    i = 0
    n = x.shape[0]
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    assert x.shape[0] == y.shape[0]
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum()) + 1e-12
    r = float((rx * ry).sum() / den)
    return float(np.clip(r, -1.0, 1.0))


def find_prior_file(prior_dir: Path, include_keys: list[str]):
    cands = sorted(prior_dir.glob("*.npz"))
    hits = []
    for p in cands:
        n = p.name.lower()
        ok = all(k in n for k in include_keys)
        if ok:
            hits.append(p)
    return hits[0] if hits else None


def topk_overlap(a_idx: np.ndarray, b_idx: np.ndarray, k: int) -> float:
    a_idx = np.asarray(a_idx).reshape(-1)
    b_idx = np.asarray(b_idx).reshape(-1)
    k = int(min(k, a_idx.size, b_idx.size))
    if k <= 0:
        return 0.0
    sa = set(a_idx[:k].tolist())
    sb = set(b_idx[:k].tolist())
    return float(len(sa & sb) / max(1, k))


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


# ============================================================
# 4) ENCODING / INFERENCE
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
# 5) PRIOR READERS
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
    gene2tfs = defaultdict(list)
    for s, d0 in zip(tf_src.tolist(), g_dst.tolist()):
        if 0 <= s < len(tfs_vocab) and 0 <= d0 < N_GENE_DIM:
            tf2targets[int(s)].append(int(d0))
            gene2tfs[int(d0)].append(int(s))

    tf_in_prior = sorted(tf2targets.keys())
    log("[PRIOR] tf_in_prior =", len(tf_in_prior))
    return tf2targets, gene2tfs, tf_in_prior


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
# 6) VECTOR BUILDERS
# ============================================================
def build_gene_vectors(
    model,
    adapter,
    V_gene: torch.Tensor,
    V_cancer: torch.Tensor,
    default_cancer_id: int,
    N_GENE_DIM: int,
    genes_union: list[int],
):
    gene_union_to_pid = {g: i for i, g in enumerate(genes_union)}

    with torch.no_grad():
        g_ids = torch.arange(N_GENE_DIM, device=DEVICE, dtype=torch.long)
        c_tok = torch.full((N_GENE_DIM,), int(default_cancer_id), device=DEVICE, dtype=torch.long)

        G = model.gene_out(g_ids) + model.gene_out_c(c_tok)
        G = G + adapter.delta_cancer(c_tok, V_cancer)

        covered_gene_ids = []
        covered_gene_pids = []
        for g in range(N_GENE_DIM):
            pid = gene_union_to_pid.get(g, None)
            if pid is not None:
                covered_gene_ids.append(g)
                covered_gene_pids.append(pid)

        if len(covered_gene_ids) > 0:
            gid_t = torch.tensor(covered_gene_ids, device=DEVICE, dtype=torch.long)
            gpid_t = torch.tensor(covered_gene_pids, device=DEVICE, dtype=torch.long)
            G[gid_t] = G[gid_t] + adapter.delta_gene(gpid_t, V_gene)

        G = F.normalize(G, dim=-1)
        return G.detach().cpu().numpy().astype(np.float32)


def build_tf_vectors(
    model,
    adapter,
    tf_in_prior: list[int],
    V_tf: torch.Tensor,
    V_cancer: torch.Tensor,
    default_cancer_id: int,
    cid: torch.Tensor,
):
    with torch.no_grad():
        tf_pid_t = torch.arange(len(tf_in_prior), device=DEVICE, dtype=torch.long)
        tf_ids = torch.tensor(tf_in_prior, device=DEVICE, dtype=torch.long)

        TF = model.tf_in(tf_ids) + model.tf_in_c(cid).expand(tf_ids.numel(), -1)
        TF = TF + adapter.delta_tf(tf_pid_t, V_tf)

        c_tok = torch.full((tf_ids.numel(),), int(default_cancer_id), device=DEVICE, dtype=torch.long)
        TF = TF + adapter.delta_cancer(c_tok, V_cancer)
        TF = F.normalize(TF, dim=-1)

        return TF.detach().cpu().numpy().astype(np.float32)


# ============================================================
# 7) REORDER
# ============================================================
def reorder_matrix_for_global_high_score(M: np.ndarray, row_names: list[str], col_names: list[str]):
    """
    目标不是严格聚类，而是让整体高分更集中、视觉上更紧凑。
    用交替重排：
      row_score = row_sum + lambda * weighted_center_similarity
      col_score = col_sum + lambda * weighted_center_similarity
    """
    M = np.asarray(M, dtype=np.float64)
    nr, nc = M.shape

    row_idx = np.arange(nr)
    col_idx = np.arange(nc)

    row_order = row_idx.copy()
    col_order = col_idx.copy()

    target_row_center = (nr - 1) / 2.0
    target_col_center = (nc - 1) / 2.0

    for _ in range(N_REORDER_ITERS):
        M0 = M[np.ix_(row_order, col_order)]

        col_weights = M0.sum(axis=0)
        if np.sum(col_weights) > 0:
            col_center = float(np.sum(np.arange(nc) * col_weights) / (np.sum(col_weights) + 1e-12))
        else:
            col_center = target_col_center

        row_scores = []
        for i in range(nr):
            vals = M0[i]
            s = float(vals.sum())
            if s > 0:
                local_center = float(np.sum(np.arange(nc) * vals) / (s + 1e-12))
            else:
                local_center = target_col_center
            compact = -abs(local_center - col_center)
            row_scores.append(s + REORDER_LAMBDA * compact)
        row_order = row_order[np.argsort(np.asarray(row_scores))[::-1]]

        M1 = M[np.ix_(row_order, col_order)]
        row_weights = M1.sum(axis=1)
        if np.sum(row_weights) > 0:
            row_center = float(np.sum(np.arange(nr) * row_weights) / (np.sum(row_weights) + 1e-12))
        else:
            row_center = target_row_center

        col_scores = []
        for j in range(nc):
            vals = M1[:, j]
            s = float(vals.sum())
            if s > 0:
                local_center = float(np.sum(np.arange(nr) * vals) / (s + 1e-12))
            else:
                local_center = target_row_center
            compact = -abs(local_center - row_center)
            col_scores.append(s + REORDER_LAMBDA * compact)
        col_order = col_order[np.argsort(np.asarray(col_scores))[::-1]]

    M2 = M[np.ix_(row_order, col_order)]
    row_names2 = [row_names[i] for i in row_order.tolist()]
    col_names2 = [col_names[j] for j in col_order.tolist()]
    return M2, row_names2, col_names2, row_order, col_order


# ============================================================
# 8) PLOT
# ============================================================
def plot_heatmap(M: np.ndarray, row_names: list[str], col_names: list[str]):
    nr, nc = M.shape

    fig = plt.figure(figsize=(1.25 * nc + 2.2, 0.9 * nr + 2.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111)

    im = ax.imshow(M, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(nc))
    ax.set_yticks(np.arange(nr))
    ax.set_xticklabels(col_names, rotation=45, ha="left", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(row_names, fontsize=10)

    for i in range(nr):
        for j in range(nc):
            v = float(M[i, j])
            txt_color = "white" if v >= 0.55 else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_title(f"Network-aware RNA-TF score heatmap | {SAMPLE_ID}", fontsize=13)
    ax.set_xlabel("Top TF")
    ax.set_ylabel("Top RNA")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Network-aware score", fontsize=10)

    fig.tight_layout()
    save_figure_both(fig, OUT_HEATMAP_PNG, OUT_HEATMAP_SVG)
    plt.close(fig)


def plot_heatmap_simple(M: np.ndarray, row_names: list[str], col_names: list[str], title: str,
                        png_path: Path, svg_path: Path, annotate: bool = True):
    nr, nc = M.shape
    fig = plt.figure(figsize=(1.25 * nc + 2.2, 0.9 * nr + 2.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111)

    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(nc))
    ax.set_yticks(np.arange(nr))
    ax.set_xticklabels(col_names, rotation=45, ha="left", rotation_mode="anchor", fontsize=10)
    ax.set_yticklabels(row_names, fontsize=10)

    if annotate:
        med = float(np.nanmedian(M)) if np.isfinite(np.nanmedian(M)) else 0.5
        for i in range(nr):
            for j in range(nc):
                v = float(M[i, j])
                txt_color = "white" if v >= med else "#222222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Top TF")
    ax.set_ylabel("Top RNA")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score", fontsize=10)

    fig.tight_layout()
    save_figure_both(fig, png_path, svg_path)
    plt.close(fig)


# ============================================================
# 9) QC HELPERS
# ============================================================
def qc_score_decomposition(
    M_corr_abs: np.ndarray,
    M_edge: np.ndarray,
    M_prior: np.ndarray,
    M_final_raw_abs: np.ndarray,
    row_names: list[str],
    col_names: list[str],
):
    corr_flat = M_corr_abs.reshape(-1)
    edge_flat = M_edge.reshape(-1)
    prior_flat = M_prior.reshape(-1)
    final_flat = M_final_raw_abs.reshape(-1)

    den = final_flat + 1e-12
    corr_contrib = (W_CORR * corr_flat) / den
    edge_contrib = (W_EDGE * edge_flat) / den
    prior_contrib = (W_PRIOR * prior_flat) / den

    df = pd.DataFrame({
        "rna": np.repeat(np.asarray(row_names, dtype=object), len(col_names)),
        "tf": np.tile(np.asarray(col_names, dtype=object), len(row_names)),
        "corr_abs": corr_flat,
        "edge": edge_flat,
        "prior": prior_flat,
        "final_raw_abs": final_flat,
        "corr_contrib_frac": corr_contrib,
        "edge_contrib_frac": edge_contrib,
        "prior_contrib_frac": prior_contrib,
    })
    df.to_csv(QC_SCORE_DECOMP_CSV, index=False)
    log("[SAVE]", QC_SCORE_DECOMP_CSV)

    r_corr = spearman_corr_1d(corr_flat, final_flat)
    r_edge = spearman_corr_1d(edge_flat, final_flat)
    r_prior = spearman_corr_1d(prior_flat, final_flat)

    top_idx = np.argsort(final_flat)[::-1][:min(20, final_flat.size)]
    top_df = df.iloc[top_idx].copy().reset_index(drop=True)

    fig = plt.figure(figsize=(15, 10), dpi=FIG_DPI)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(corr_flat, final_flat, s=24, alpha=0.8)
    ax1.set_xlabel("corr_abs")
    ax1.set_ylabel("final_raw_abs")
    ax1.set_title(f"corr_abs vs final | Spearman={r_corr:.3f}")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(edge_flat, final_flat, s=24, alpha=0.8)
    ax2.set_xlabel("edge")
    ax2.set_ylabel("final_raw_abs")
    ax2.set_title(f"edge vs final | Spearman={r_edge:.3f}")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(prior_flat, final_flat, s=24, alpha=0.8)
    ax3.set_xlabel("prior")
    ax3.set_ylabel("final_raw_abs")
    ax3.set_title(f"prior vs final | Spearman={r_prior:.3f}")

    ax4 = fig.add_subplot(2, 2, 4)
    x = np.arange(len(top_df))
    ax4.bar(x, top_df["corr_contrib_frac"].values, label="corr")
    ax4.bar(x, top_df["edge_contrib_frac"].values, bottom=top_df["corr_contrib_frac"].values, label="edge")
    ax4.bar(
        x,
        top_df["prior_contrib_frac"].values,
        bottom=(top_df["corr_contrib_frac"].values + top_df["edge_contrib_frac"].values),
        label="prior",
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [f"{r}/{t}" for r, t in zip(top_df["rna"].tolist(), top_df["tf"].tolist())],
        rotation=75, ha="right", fontsize=8
    )
    ax4.set_ylabel("fraction of final score")
    ax4.set_title("Top pairs contribution decomposition")
    ax4.legend()

    fig.suptitle(f"QC A: score decomposition | {SAMPLE_ID}", fontsize=14)
    fig.tight_layout()
    save_figure_both(fig, QC_SCORE_DECOMP_PNG, QC_SCORE_DECOMP_SVG)
    plt.close(fig)


def qc_raw_vs_reordered_heatmap(
    M_raw_vis: np.ndarray,
    M_reordered_vis: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    row_names_re: list[str],
    col_names_re: list[str],
):
    df_raw = pd.DataFrame(M_raw_vis, index=row_names, columns=col_names)
    df_raw.to_csv(QC_RAW_MATRIX_CSV)
    log("[SAVE]", QC_RAW_MATRIX_CSV)

    plot_heatmap_simple(
        M_raw_vis,
        row_names,
        col_names,
        title=f"QC B1: Raw score heatmap | {SAMPLE_ID}",
        png_path=QC_RAW_HEATMAP_PNG,
        svg_path=QC_RAW_HEATMAP_SVG,
        annotate=True,
    )

    fig = plt.figure(figsize=(16, 7), dpi=FIG_DPI)

    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(M_raw_vis, aspect="auto", interpolation="nearest")
    ax1.set_xticks(np.arange(len(col_names)))
    ax1.set_yticks(np.arange(len(row_names)))
    ax1.set_xticklabels(col_names, rotation=45, ha="left", rotation_mode="anchor", fontsize=9)
    ax1.set_yticklabels(row_names, fontsize=9)
    ax1.set_title("Raw order")

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(M_reordered_vis, aspect="auto", interpolation="nearest")
    ax2.set_xticks(np.arange(len(col_names_re)))
    ax2.set_yticks(np.arange(len(row_names_re)))
    ax2.set_xticklabels(col_names_re, rotation=45, ha="left", rotation_mode="anchor", fontsize=9)
    ax2.set_yticklabels(row_names_re, fontsize=9)
    ax2.set_title("Reordered for display")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label("Score")

    fig.suptitle(f"QC B: raw vs reordered heatmap | {SAMPLE_ID}", fontsize=14)
    fig.tight_layout()
    save_figure_both(fig, QC_REORDER_COMPARE_PNG, QC_REORDER_COMPARE_SVG)
    plt.close(fig)


def qc_raw_vs_minmax_rank(M_final_raw_abs: np.ndarray, M_final_minmax_abs: np.ndarray):
    raw_flat = M_final_raw_abs.reshape(-1)
    mm_flat = M_final_minmax_abs.reshape(-1)

    rank_spr = spearman_corr_1d(raw_flat, mm_flat)

    raw_order = np.argsort(raw_flat)[::-1]
    mm_order = np.argsort(mm_flat)[::-1]

    overlap_5 = topk_overlap(raw_order, mm_order, 5)
    overlap_10 = topk_overlap(raw_order, mm_order, 10)
    overlap_20 = topk_overlap(raw_order, mm_order, 20)

    with open(QC_MINMAX_COMPARE_TXT, "w", encoding="utf-8") as f:
        f.write(f"SAMPLE_ID\t{SAMPLE_ID}\n")
        f.write(f"spearman_rank_raw_vs_minmax\t{rank_spr:.6f}\n")
        f.write(f"top5_overlap\t{overlap_5:.6f}\n")
        f.write(f"top10_overlap\t{overlap_10:.6f}\n")
        f.write(f"top20_overlap\t{overlap_20:.6f}\n")
    log("[SAVE]", QC_MINMAX_COMPARE_TXT)

    fig = plt.figure(figsize=(7.5, 6.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.scatter(raw_flat, mm_flat, s=28, alpha=0.8)
    ax.set_xlabel("Raw final score")
    ax.set_ylabel("Minmax-scaled final score")
    ax.set_title(
        f"QC C: raw vs minmax\n"
        f"Spearman={rank_spr:.3f} | top5={overlap_5:.2f} | top10={overlap_10:.2f}"
    )
    fig.tight_layout()
    save_figure_both(fig, QC_MINMAX_COMPARE_PNG, QC_MINMAX_COMPARE_SVG)
    plt.close(fig)


def qc_signed_vs_abs_corr(
    M_corr_signed: np.ndarray,
    M_corr_abs: np.ndarray,
    M_edge: np.ndarray,
    M_prior: np.ndarray,
    row_names: list[str],
    col_names: list[str],
):
    # signed version: map corr [-1,1] -> [0,1]
    M_corr_signed01 = np.clip((M_corr_signed + 1.0) * 0.5, 0.0, 1.0)

    M_final_raw_signed = (
        W_CORR * M_corr_signed01
        + W_EDGE * M_edge
        + W_PRIOR * M_prior
    )
    M_final_signed = minmax01(M_final_raw_signed.reshape(-1)).reshape(M_final_raw_signed.shape)

    M_final_raw_abs = (
        W_CORR * M_corr_abs
        + W_EDGE * M_edge
        + W_PRIOR * M_prior
    )
    M_final_abs = minmax01(M_final_raw_abs.reshape(-1)).reshape(M_final_raw_abs.shape)

    abs_flat = M_final_abs.reshape(-1)
    signed_flat = M_final_signed.reshape(-1)

    abs_order = np.argsort(abs_flat)[::-1]
    signed_order = np.argsort(signed_flat)[::-1]

    overlap_5 = topk_overlap(abs_order, signed_order, 5)
    overlap_10 = topk_overlap(abs_order, signed_order, 10)
    overlap_20 = topk_overlap(abs_order, signed_order, 20)

    df = pd.DataFrame({
        "rna": np.repeat(np.asarray(row_names, dtype=object), len(col_names)),
        "tf": np.tile(np.asarray(col_names, dtype=object), len(row_names)),
        "corr_signed": M_corr_signed.reshape(-1),
        "corr_abs": M_corr_abs.reshape(-1),
        "edge": M_edge.reshape(-1),
        "prior": M_prior.reshape(-1),
        "final_abs_minmax": abs_flat,
        "final_signed_minmax": signed_flat,
        "delta_signed_minus_abs": signed_flat - abs_flat,
    }).sort_values("delta_signed_minus_abs", ascending=False)
    df.to_csv(QC_SIGNED_COMPARE_CSV, index=False)
    log("[SAVE]", QC_SIGNED_COMPARE_CSV)

    fig = plt.figure(figsize=(15, 10), dpi=FIG_DPI)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(M_corr_signed.reshape(-1), M_corr_abs.reshape(-1), s=24, alpha=0.8)
    ax1.set_xlabel("signed corr")
    ax1.set_ylabel("abs corr")
    ax1.set_title("signed corr vs abs corr")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(abs_flat, signed_flat, s=24, alpha=0.8)
    ax2.set_xlabel("final_abs_minmax")
    ax2.set_ylabel("final_signed_minmax")
    ax2.set_title(
        f"final score comparison\n"
        f"top5={overlap_5:.2f} | top10={overlap_10:.2f} | top20={overlap_20:.2f}"
    )

    ax3 = fig.add_subplot(2, 2, 3)
    vals = df["delta_signed_minus_abs"].values
    ax3.hist(vals, bins=20)
    ax3.set_xlabel("signed - abs final score")
    ax3.set_ylabel("count")
    ax3.set_title("distribution of score change")

    ax4 = fig.add_subplot(2, 2, 4)
    top_show = min(15, len(df))
    sub = df.iloc[:top_show].copy()
    x = np.arange(top_show)
    ax4.bar(x - 0.2, sub["final_abs_minmax"].values, width=0.4, label="abs")
    ax4.bar(x + 0.2, sub["final_signed_minmax"].values, width=0.4, label="signed")
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [f"{r}/{t}" for r, t in zip(sub["rna"].tolist(), sub["tf"].tolist())],
        rotation=75, ha="right", fontsize=8
    )
    ax4.set_ylabel("minmax score")
    ax4.set_title("pairs most affected by signed-vs-abs choice")
    ax4.legend()

    fig.suptitle(f"QC D: signed corr vs abs(corr) | {SAMPLE_ID}", fontsize=14)
    fig.tight_layout()
    save_figure_both(fig, QC_SIGNED_COMPARE_PNG, QC_SIGNED_COMPARE_SVG)
    plt.close(fig)

    plot_heatmap_simple(
        M_final_signed,
        row_names,
        col_names,
        title=f"QC D2: signed-corr final heatmap | {SAMPLE_ID}",
        png_path=QC_SIGNED_HEATMAP_PNG,
        svg_path=QC_SIGNED_HEATMAP_SVG,
        annotate=True,
    )


# ============================================================
# 10) MAIN
# ============================================================
def main():
    assert GLOBAL_GENE_VOCAB.exists(), f"Missing {GLOBAL_GENE_VOCAB}"
    assert GLOBAL_TF_VOCAB.exists(), f"Missing {GLOBAL_TF_VOCAB}"
    assert FRONTIER_CKPT.exists(), f"Missing {FRONTIER_CKPT}"
    assert CENTROIDS_NPY.exists(), f"Missing {CENTROIDS_NPY}"
    assert CLUSTERS_CSV.exists(), f"Missing {CLUSTERS_CSV}"
    assert BASE_CKPT.exists(), f"Missing {BASE_CKPT}"
    assert FINAL_TOP10_RNA_CSV.exists(), f"Missing {FINAL_TOP10_RNA_CSV}"
    assert FINAL_TOP10_TF_CSV.exists(), f"Missing {FINAL_TOP10_TF_CSV}"

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    # ---------- top lists ----------
    df_rna_top = pd.read_csv(FINAL_TOP10_RNA_CSV)
    df_tf_top = pd.read_csv(FINAL_TOP10_TF_CSV)

    assert {"train_gid", "gene"}.issubset(df_rna_top.columns), "RNA top10 csv missing train_gid/gene"
    assert {"tf_global_id", "tf"}.issubset(df_tf_top.columns), "TF top10 csv missing tf_global_id/tf"

    top_rna_gids = df_rna_top["train_gid"].astype(int).tolist()
    top_rna_names = df_rna_top["gene"].astype(str).tolist()

    top_tf_global_ids = df_tf_top["tf_global_id"].astype(int).tolist()
    top_tf_names = df_tf_top["tf"].astype(str).tolist()

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
    D = int(cfg["D"])
    HID = int(cfg["HID"])
    DROP = float(cfg["DROP"])
    PEAK_HASH_SIZE = int(cfg["PEAK_HASH_SIZE"])
    cancer_types = list(cfg["cancer_types"])

    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_id = int(cancer2id.get("CRC", 0))
    cid = torch.tensor([default_cancer_id], device=DEVICE, dtype=torch.long)

    model = PlanC_DirAware_Cancer_HashPeak(
        n_gene=N_GENE_DIM,
        n_tf=int(cfg["N_TF_DIM"]),
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

    # ---------- frontier ckpt ----------
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

    tf2targets, gene2tfs_prior, tf_in_prior = read_tf_prior(PRIORS_DIR, tfs_vocab, N_GENE_DIM)
    _, genes_union = read_gene_peak_prior(PRIORS_DIR, N_GENE_DIM)
    gene2tfs_prior_set = {g: set(v) for g, v in gene2tfs_prior.items()}

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

    # ---------- sample ----------
    sample = load_single_sample(DATASET_NAME, SAMPLE_ID, genes_vocab)
    sid2sample = {sample.sid: sample}
    cell_list = [(sample.sid, int(i)) for i in range(sample.counts_shape[0])]
    n_cells = len(cell_list)
    log("[CELLS]", n_cells)

    # ---------- encode sample ----------
    proj_store_cache = {}
    Z = encode_Z(cell_list, sid2sample, model, default_cancer_id, proj_store_cache)
    Zt = torch.from_numpy(Z).to(DEVICE)

    C = np.load(CENTROIDS_NPY).astype(np.float32)
    raw_cluster = spherical_kmeans_assign(Z, C)
    group_ids = np.asarray([cl2gid.get(int(c), 0) for c in raw_cluster.tolist()], dtype=np.int64)

    with torch.no_grad():
        V_tf, V_gene, _, V_cancer = adapter()

        gid_t = torch.from_numpy(group_ids.astype(np.int64)).to(DEVICE)
        cid_t = torch.full((n_cells,), int(default_cancer_id), device=DEVICE, dtype=torch.long)

        Zadj = Zt
        Zadj = adapter.apply_group_to_Z(Zadj, gid_t, GROUP_SCALE_ALPHA)
        Zadj = Zadj + adapter.delta_cancer(cid_t, V_cancer)
        Zadj = F.normalize(Zadj, dim=-1)

        cluster_for_corr = raw_cluster if CLUSTER_CORRECTION_USE_RAW_CLUSTER else group_ids
        Zadj = apply_cluster_correction(Zadj, cluster_for_corr, alpha=0.25)

        TF_vec = build_tf_vectors(
            model=model,
            adapter=adapter,
            tf_in_prior=tf_in_prior,
            V_tf=V_tf,
            V_cancer=V_cancer,
            default_cancer_id=default_cancer_id,
            cid=cid,
        )
        pred_tf = (Zadj @ torch.from_numpy(TF_vec).to(DEVICE).t()).detach().cpu().numpy().astype(np.float32)

        G_vec = build_gene_vectors(
            model=model,
            adapter=adapter,
            V_gene=V_gene,
            V_cancer=V_cancer,
            default_cancer_id=default_cancer_id,
            N_GENE_DIM=N_GENE_DIM,
            genes_union=genes_union,
        )

    # ---------- mappings ----------
    tfid_to_pid = {tf_id: i for i, tf_id in enumerate(tf_in_prior)}

    rna_rows = []
    for gid, gname in zip(top_rna_gids, top_rna_names):
        if gid < 0 or gid >= N_GENE_DIM:
            raise ValueError(f"Invalid top RNA train_gid: {gid}")
        if gid not in sample.train2rna:
            raise ValueError(f"Top RNA not found in sample train2rna: gid={gid}, gene={gname}")
        rna_rows.append((int(gid), str(gname), int(sample.train2rna[gid])))

    tf_rows = []
    for tfid, tfname in zip(top_tf_global_ids, top_tf_names):
        if tfid not in tfid_to_pid:
            raise ValueError(f"Top TF global id not covered by tf_in_prior: tfid={tfid}, tf={tfname}")
        tf_rows.append((int(tfid), str(tfname), int(tfid_to_pid[tfid])))

    # ---------- pair score matrix ----------
    Xmm = np.load(sample.counts_path, mmap_mode="r")

    M_corr_signed = np.zeros((len(rna_rows), len(tf_rows)), dtype=np.float64)
    M_corr_abs = np.zeros((len(rna_rows), len(tf_rows)), dtype=np.float64)
    M_edge = np.zeros((len(rna_rows), len(tf_rows)), dtype=np.float64)
    M_prior = np.zeros((len(rna_rows), len(tf_rows)), dtype=np.float64)
    M_final_raw_abs = np.zeros((len(rna_rows), len(tf_rows)), dtype=np.float64)

    for i, (gid, gname, rna_col) in enumerate(rna_rows):
        x = np.asarray(Xmm[:, rna_col], dtype=np.float32).reshape(-1)
        x = np.log1p(np.clip(x, 0.0, None)).astype(np.float64)

        gvec = G_vec[gid].astype(np.float32)

        for j, (tfid, tfname, tf_pid) in enumerate(tf_rows):
            y = pred_tf[:, tf_pid].astype(np.float64)

            corr = spearman_corr_1d(x, y)
            corr_abs01 = float(np.clip(abs(corr), 0.0, 1.0))

            edge_cos = float(TF_vec[tf_pid] @ gvec)
            edge01 = float(np.clip((edge_cos + 1.0) * 0.5, 0.0, 1.0))

            prior01 = 1.0 if tfid in gene2tfs_prior_set.get(gid, set()) else 0.0

            score_raw_abs = (
                W_CORR * corr_abs01
                + W_EDGE * edge01
                + W_PRIOR * prior01
            )

            M_corr_signed[i, j] = corr
            M_corr_abs[i, j] = corr_abs01
            M_edge[i, j] = edge01
            M_prior[i, j] = prior01
            M_final_raw_abs[i, j] = score_raw_abs

    # ---------- final score for display ----------
    M_final = minmax01(M_final_raw_abs.reshape(-1)).reshape(M_final_raw_abs.shape)

    # ---------- reorder ----------
    row_names = [x[1] for x in rna_rows]
    col_names = [x[1] for x in tf_rows]
    M_re, row_names_re, col_names_re, row_order, col_order = reorder_matrix_for_global_high_score(
        M_final, row_names, col_names
    )

    # ---------- save matrix ----------
    df_mat = pd.DataFrame(M_re, index=row_names_re, columns=col_names_re)
    df_mat.to_csv(OUT_MATRIX_CSV)
    log("[SAVE]", OUT_MATRIX_CSV)

    # ---------- main plot ----------
    plot_heatmap(M_re, row_names_re, col_names_re)

    # ============================================================
    # QC A-D
    # ============================================================
    qc_score_decomposition(
        M_corr_abs=M_corr_abs,
        M_edge=M_edge,
        M_prior=M_prior,
        M_final_raw_abs=M_final_raw_abs,
        row_names=row_names,
        col_names=col_names,
    )

    qc_raw_vs_reordered_heatmap(
        M_raw_vis=M_final,
        M_reordered_vis=M_re,
        row_names=row_names,
        col_names=col_names,
        row_names_re=row_names_re,
        col_names_re=col_names_re,
    )

    qc_raw_vs_minmax_rank(
        M_final_raw_abs=M_final_raw_abs,
        M_final_minmax_abs=M_final,
    )

    qc_signed_vs_abs_corr(
        M_corr_signed=M_corr_signed,
        M_corr_abs=M_corr_abs,
        M_edge=M_edge,
        M_prior=M_prior,
        row_names=row_names,
        col_names=col_names,
    )

    log("[DONE]")


if __name__ == "__main__":
    main()
