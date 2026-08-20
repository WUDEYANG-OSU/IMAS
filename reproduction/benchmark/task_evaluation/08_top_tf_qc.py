from __future__ import annotations

# ============================================================
# crc_sample9_network_aware_top_tf_qc_allinone.py
# ------------------------------------------------------------
# 单文件完整版本：
# 1) 重新跑单样本 RNA->TF network-aware ranking
# 2) 生成 10 个 QC
#
# QC 列表：
# 01 最终 Top TF 分数分解
# 02 TF-target gene 一致性检验
# 03 bootstrap 稳定性
# 04 消融实验
# 05 subgroup / pseudo-bulk consistency
# 06 正负方向一致性 QC
# 07 prior degree bias 检查
# 08 latent edge 正负样本分布
# 09 permutation / null test
# 10 每个 TF 的证据卡片整理
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

try:
    from scipy.stats import mannwhitneyu, spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

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

OUT_DIR = RUN_BASE / f"single_sample_{SAMPLE_ID}_network_aware_top_tf_qc_allinone"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 主流程参数
GENE_TOPK = 10
FINAL_TOPK = 10
MIN_CELLS_EXPRESSED = 10
MIN_GENE_STD = 0.08
RNA_TOPK = 2048
EMB_BATCH = 256

EDGE_WEIGHT_POWER = 1.0
CORR_WEIGHT_POWER = 1.0
GENE_WEIGHT_POWER = 1.0
ACTIVITY_ALPHA = 0.72
CONTINUITY_ALPHA = 0.28

# QC 参数
BOOT_N = 50
BOOT_FRAC = 0.80
SUBGROUP_MIN_CELLS = 80
NULL_N = 20
LATENT_NEG_N = 50000

# plotting
FIG_DPI = 220
BAR_COLOR = "#f28e2b"
BLUE = "#4e79a7"
ORANGE = "#f28e2b"
RED = "#e15759"
GREEN = "#59a14f"

# reproducibility
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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure_both(fig, png_path: Path, svg_path: Path):
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    log("[SAVE]", png_path)
    log("[SAVE]", svg_path)
    plt.close(fig)


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


def find_prior_file(prior_dir: Path, include_keys: list[str]):
    cands = sorted(prior_dir.glob("*.npz"))
    hits = []
    for p in cands:
        n = p.name.lower()
        ok = all(k in n for k in include_keys)
        if ok:
            hits.append(p)
    return hits[0] if hits else None


def rank_center_norm_matrix(Y: np.ndarray):
    n = Y.shape[0]
    order = np.argsort(Y, axis=0)
    ranks = np.empty_like(order, dtype=np.int32)
    for j in range(order.shape[1]):
        ranks[order[:, j], j] = np.arange(n, dtype=np.int32)
    ry = ranks.astype(np.float64)
    ry -= ry.mean(axis=0, keepdims=True)
    ry_norm = np.sqrt((ry * ry).sum(axis=0)) + 1e-12
    return ry, ry_norm


def spearman_against_fixed_ranked_Y(x: np.ndarray, ry: np.ndarray, ry_norm: np.ndarray) -> np.ndarray:
    rx = x.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    rx_norm = np.sqrt((rx * rx).sum()) + 1e-12
    num = (rx[:, None] * ry).sum(axis=0)
    r = num / (rx_norm * ry_norm)
    return np.clip(r, -1.0, 1.0)


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo + 1e-12)


def _safe_mwu(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    if SCIPY_OK:
        try:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            return float(stat), float(p)
        except Exception:
            return np.nan, np.nan
    return np.nan, np.nan


def _safe_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or len(y) < 3:
        return np.nan
    if SCIPY_OK:
        try:
            r, _ = spearmanr(x, y)
            return float(r)
        except Exception:
            return np.nan
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum()) + 1e-12
    return float((rx * ry).sum() / den)


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
# 6) NETWORK-AWARE HELPERS
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


def greedy_select_top_tfs(activity_score: np.ndarray, continuity: np.ndarray, final_topk: int):
    n_tf = activity_score.shape[0]
    activity_n = minmax01(activity_score)
    if continuity.size > 0:
        cont_n = continuity.copy()
        cont_n = cont_n / (np.nanmax(cont_n) + 1e-12)
    else:
        cont_n = continuity

    selected = []
    remaining = set(range(n_tf))

    while len(selected) < min(final_topk, n_tf) and len(remaining) > 0:
        best_tf = None
        best_score = -1e18

        for tf in remaining:
            act = ACTIVITY_ALPHA * activity_n[tf]
            if len(selected) == 0:
                coh = 0.0
            else:
                coh = CONTINUITY_ALPHA * float(np.mean(cont_n[tf, selected]))
            score = act + coh

            if score > best_score:
                best_score = score
                best_tf = tf

        selected.append(best_tf)
        remaining.remove(best_tf)

    return selected


# ============================================================
# 7) CONTEXT OBJECT
# ============================================================
@dataclass
class QCContext:
    sample: Sample
    genes_vocab: list[str]
    tfs_vocab: list[str]
    gene_names_norm: list[str]
    X_log1p: np.ndarray
    pred: np.ndarray
    TF_vec: np.ndarray
    G_vec: np.ndarray
    tf_in_prior: list[int]
    tf_names: list[str]
    tf2targets: dict[int, list[int]]
    gene2tfs_prior: dict[int, list[int]]
    tfid_to_pid: dict[int, int]
    raw_cluster: np.ndarray
    group_ids: np.ndarray
    N_GENE_DIM: int
    full_edges: pd.DataFrame
    full_tf: pd.DataFrame
    full_final: pd.DataFrame
    continuity_full: np.ndarray


# ============================================================
# 8) 核心 ranking 引擎
# ============================================================
def rank_from_arrays(
    ctx: QCContext,
    pred: np.ndarray,
    X_log1p: np.ndarray,
    gene_topk: int = GENE_TOPK,
    final_topk: int = FINAL_TOPK,
    use_prior: bool = True,
    use_edge: bool = True,
    use_continuity: bool = True,
    random_gene_to_cands: dict[int, np.ndarray] | None = None,
    tag: str = "full",
):
    ry, ry_norm = rank_center_norm_matrix(pred.astype(np.float64))

    n_tf = pred.shape[1]
    tf_activity = np.zeros((n_tf,), dtype=np.float64)
    continuity = np.zeros((n_tf, n_tf), dtype=np.float64)
    edge_rows = []

    n_cells, n_genes_total = X_log1p.shape
    all_pids = np.arange(n_tf, dtype=np.int64)

    for j in range(n_genes_total):
        gene_name = ctx.gene_names_norm[j]
        train_gid = int(ctx.sample.ext2train[j])
        if train_gid < 0 or train_gid >= ctx.N_GENE_DIM:
            continue

        if random_gene_to_cands is not None:
            cand_pids = np.asarray(random_gene_to_cands.get(train_gid, []), dtype=np.int64)
        else:
            if use_prior:
                prior_tf_ids = ctx.gene2tfs_prior.get(train_gid, [])
                cand_pids = [ctx.tfid_to_pid[tf_id] for tf_id in prior_tf_ids if tf_id in ctx.tfid_to_pid]
                cand_pids = np.asarray(sorted(set(cand_pids)), dtype=np.int64)
            else:
                cand_pids = all_pids

        if len(cand_pids) == 0:
            continue

        y = np.asarray(X_log1p[:, j], dtype=np.float32).reshape(-1).astype(np.float64)
        n_expr = int((y > 0).sum())
        y_std = float(np.std(y))
        if n_expr < MIN_CELLS_EXPRESSED:
            continue
        if y_std < MIN_GENE_STD:
            continue

        r_all = spearman_against_fixed_ranked_Y(y, ry, ry_norm)
        r_cand = r_all[cand_pids]
        abs_r_cand = np.abs(r_cand)

        if use_edge:
            gvec = ctx.G_vec[train_gid]
            edge_cos = ctx.TF_vec[cand_pids] @ gvec.astype(np.float32)
            edge_strength = np.clip((edge_cos + 1.0) * 0.5, 0.0, 1.0)
        else:
            edge_strength = np.ones_like(abs_r_cand, dtype=np.float64)

        combined = (
            np.power(abs_r_cand, CORR_WEIGHT_POWER)
            * np.power(edge_strength, EDGE_WEIGHT_POWER)
        )

        if combined.size == 0 or np.nanmax(combined) <= 0:
            continue

        topk = min(gene_topk, combined.size)
        sel_local = np.argpartition(combined, -topk)[-topk:]
        sel_local = sel_local[np.argsort(combined[sel_local])[::-1]]

        frac_expr = float(n_expr) / float(n_cells + 1e-12)
        gene_weight = np.power(y_std * np.sqrt(max(frac_expr, 1e-12)), GENE_WEIGHT_POWER)

        chosen_pids = cand_pids[sel_local]
        chosen_scores = combined[sel_local]
        chosen_r = r_cand[sel_local]
        chosen_abs_r = abs_r_cand[sel_local]
        chosen_edge = edge_strength[sel_local]

        for pid, score in zip(chosen_pids.tolist(), chosen_scores.tolist()):
            tf_activity[pid] += gene_weight * float(score)

        for a in range(len(chosen_pids)):
            pa = int(chosen_pids[a])
            sa = float(chosen_scores[a])
            for b in range(a + 1, len(chosen_pids)):
                pb = int(chosen_pids[b])
                sb = float(chosen_scores[b])
                c_add = gene_weight * min(sa, sb)
                continuity[pa, pb] += c_add
                continuity[pb, pa] += c_add

        for rank0, (pid, score, rr, arr, ew) in enumerate(zip(
            chosen_pids.tolist(),
            chosen_scores.tolist(),
            chosen_r.tolist(),
            chosen_abs_r.tolist(),
            chosen_edge.tolist(),
        ), start=1):
            tf_global_id = int(ctx.tf_in_prior[pid])
            edge_rows.append({
                "mode": tag,
                "gene_name": gene_name,
                "gene_col": int(j),
                "train_gid": int(train_gid),
                "gene_weight": float(gene_weight),
                "n_expr_cells": int(n_expr),
                "gene_std_log1p": float(y_std),
                "tf_pid": int(pid),
                "tf_global_id": tf_global_id,
                "tf_name": ctx.tf_names[pid],
                "rank_within_gene": int(rank0),
                "spearman": float(rr),
                "abs_spearman": float(arr),
                "model_edge_weight": float(ew),
                "combined_edge_score": float(score),
            })

    assert len(edge_rows) > 0, f"[{tag}] no edges found"

    df_edges = pd.DataFrame(edge_rows).sort_values(
        ["combined_edge_score", "gene_weight"], ascending=[False, False]
    ).reset_index(drop=True)

    continuity_sum = continuity.sum(axis=1)
    df_tf = pd.DataFrame({
        "mode": tag,
        "tf_pid": np.arange(len(ctx.tf_in_prior), dtype=np.int64),
        "tf_global_id": np.asarray(ctx.tf_in_prior, dtype=np.int64),
        "tf": ctx.tf_names,
        "activity_score": tf_activity.astype(np.float64),
        "continuity_sum": continuity_sum.astype(np.float64),
    })
    df_tf["activity_norm"] = minmax01(df_tf["activity_score"].values)
    df_tf["continuity_norm"] = minmax01(df_tf["continuity_sum"].values)
    df_tf["pre_score"] = (
        ACTIVITY_ALPHA * df_tf["activity_norm"].values
        + CONTINUITY_ALPHA * df_tf["continuity_norm"].values
    )
    df_tf = df_tf.sort_values("pre_score", ascending=False).reset_index(drop=True)

    if use_continuity:
        selected = greedy_select_top_tfs(
            activity_score=tf_activity,
            continuity=continuity,
            final_topk=final_topk,
        )
    else:
        selected = np.argsort(tf_activity)[::-1][:final_topk].tolist()

    final_rows = []
    act_all_n = minmax01(tf_activity)
    mx_cont = float(np.nanmax(continuity)) if np.nanmax(continuity) > 0 else 0.0

    for rank0, pid in enumerate(selected, start=1):
        others = [x for x in selected if x != pid]
        local_coh = float(np.mean(continuity[pid, others])) if len(others) > 0 else 0.0
        act_n = float(act_all_n[pid])
        coh_n = float(local_coh / (mx_cont + 1e-12)) if mx_cont > 0 else 0.0
        final_score = ACTIVITY_ALPHA * act_n + (CONTINUITY_ALPHA * coh_n if use_continuity else 0.0)

        final_rows.append({
            "mode": tag,
            "rank": int(rank0),
            "tf_pid": int(pid),
            "tf_global_id": int(ctx.tf_in_prior[pid]),
            "tf": ctx.tf_names[pid],
            "activity_score": float(tf_activity[pid]),
            "selected_set_mean_continuity": float(local_coh),
            "activity_norm": act_n,
            "cohesion_norm": coh_n,
            "final_score": float(final_score),
        })

    df_final = pd.DataFrame(final_rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    return df_edges, df_tf, df_final, continuity


def build_random_candidate_map(ctx: QCContext, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_tf = len(ctx.tf_in_prior)
    gene_to_cands = {}
    for train_gid, tf_ids in ctx.gene2tfs_prior.items():
        deg = len([x for x in tf_ids if x in ctx.tfid_to_pid])
        if deg <= 0:
            continue
        deg = min(deg, n_tf)
        gene_to_cands[int(train_gid)] = rng.choice(np.arange(n_tf), size=deg, replace=False).astype(np.int64)
    return gene_to_cands


# ============================================================
# 9) 主运行：重建 full 结果
# ============================================================
def build_context() -> QCContext:
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

    sample = load_single_sample(DATASET_NAME, SAMPLE_ID, genes_vocab)
    sid2sample = {sample.sid: sample}
    cell_list = [(sample.sid, int(i)) for i in range(sample.counts_shape[0])]
    n_cells = len(cell_list)
    log("[CELLS]", n_cells)

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
        TF_vec_t = torch.from_numpy(TF_vec).to(DEVICE)
        pred = (Zadj @ TF_vec_t.t()).detach().cpu().numpy().astype(np.float32)

        G_vec = build_gene_vectors(
            model=model,
            adapter=adapter,
            V_gene=V_gene,
            V_cancer=V_cancer,
            default_cancer_id=default_cancer_id,
            N_GENE_DIM=N_GENE_DIM,
            genes_union=genes_union,
        )

    Xmm = np.load(sample.counts_path, mmap_mode="r")
    X_log1p = np.log1p(np.clip(np.asarray(Xmm, dtype=np.float32), 0.0, None)).astype(np.float32)
    gene_names_norm = [normalize_gene_name(g) for g in sample.gene_names.tolist()]

    tf_names = [tfs_vocab[int(tf_id)] for tf_id in tf_in_prior]
    tfid_to_pid = {tf_id: i for i, tf_id in enumerate(tf_in_prior)}

    # full ranking
    dummy_ctx = QCContext(
        sample=sample,
        genes_vocab=genes_vocab,
        tfs_vocab=tfs_vocab,
        gene_names_norm=gene_names_norm,
        X_log1p=X_log1p,
        pred=pred,
        TF_vec=TF_vec,
        G_vec=G_vec,
        tf_in_prior=tf_in_prior,
        tf_names=tf_names,
        tf2targets=tf2targets,
        gene2tfs_prior=gene2tfs_prior,
        tfid_to_pid=tfid_to_pid,
        raw_cluster=raw_cluster,
        group_ids=group_ids,
        N_GENE_DIM=N_GENE_DIM,
        full_edges=pd.DataFrame(),
        full_tf=pd.DataFrame(),
        full_final=pd.DataFrame(),
        continuity_full=np.zeros((len(tf_in_prior), len(tf_in_prior)), dtype=np.float64),
    )

    full_edges, full_tf, full_final, continuity_full = rank_from_arrays(
        dummy_ctx, pred, X_log1p,
        use_prior=True, use_edge=True, use_continuity=True,
        tag="full"
    )

    full_edges.to_csv(OUT_DIR / f"{SAMPLE_ID}_gene_topk_tf_edges.csv", index=False)
    full_tf.to_csv(OUT_DIR / f"{SAMPLE_ID}_tf_activity_summary.csv", index=False)
    pd.DataFrame(continuity_full, index=tf_names, columns=tf_names).to_csv(OUT_DIR / f"{SAMPLE_ID}_tf_continuity_matrix.csv")
    full_final.to_csv(OUT_DIR / f"{SAMPLE_ID}_final_top10_tfs.csv", index=False)

    log("[SAVE]", OUT_DIR / f"{SAMPLE_ID}_gene_topk_tf_edges.csv")
    log("[SAVE]", OUT_DIR / f"{SAMPLE_ID}_tf_activity_summary.csv")
    log("[SAVE]", OUT_DIR / f"{SAMPLE_ID}_tf_continuity_matrix.csv")
    log("[SAVE]", OUT_DIR / f"{SAMPLE_ID}_final_top10_tfs.csv")

    ctx = QCContext(
        sample=sample,
        genes_vocab=genes_vocab,
        tfs_vocab=tfs_vocab,
        gene_names_norm=gene_names_norm,
        X_log1p=X_log1p,
        pred=pred,
        TF_vec=TF_vec,
        G_vec=G_vec,
        tf_in_prior=tf_in_prior,
        tf_names=tf_names,
        tf2targets=tf2targets,
        gene2tfs_prior=gene2tfs_prior,
        tfid_to_pid=tfid_to_pid,
        raw_cluster=raw_cluster,
        group_ids=group_ids,
        N_GENE_DIM=N_GENE_DIM,
        full_edges=full_edges,
        full_tf=full_tf,
        full_final=full_final,
        continuity_full=continuity_full,
    )
    return ctx


# ============================================================
# 10) 主结果柱状图
# ============================================================
def plot_final_bar(df_final: pd.DataFrame, png_path: Path, svg_path: Path, title_suffix: str):
    top = df_final.copy().sort_values("final_score", ascending=True).reset_index(drop=True)
    labels = top["tf"].astype(str).tolist()
    vals = top["final_score"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(9.2, max(5.0, 0.48 * len(labels))), dpi=FIG_DPI)
    ax = fig.add_subplot(111)

    y = np.arange(len(labels))
    ax.barh(y, vals, color=BAR_COLOR, edgecolor="none", alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Final network-aware TF score", fontsize=11)
    ax.set_title(title_suffix, fontsize=12)

    xmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    for i, v in enumerate(vals):
        ax.text(v + 0.02 * max(1e-6, xmax), i, f"{v:.3f}", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.20)
    fig.tight_layout()
    save_figure_both(fig, png_path, svg_path)


# ============================================================
# 11) QC1：最终 Top TF 分数分解
# ============================================================
def qc1_score_decomposition(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "01_score_decomposition")
    rows = []

    for _, r in ctx.full_final.iterrows():
        tf = str(r["tf"])
        pid = int(r["tf_pid"])
        sub = ctx.full_edges[ctx.full_edges["tf_pid"] == pid].copy()

        if len(sub) == 0:
            continue

        w = sub["gene_weight"].values.astype(float)
        rows.append({
            "tf": tf,
            "tf_pid": pid,
            "final_rank": int(r["rank"]),
            "final_score": float(r["final_score"]),
            "activity_score": float(r["activity_score"]),
            "selected_set_mean_continuity": float(r["selected_set_mean_continuity"]),
            "support_gene_count": int(len(sub)),
            "median_abs_spearman": float(sub["abs_spearman"].median()),
            "median_edge_weight": float(sub["model_edge_weight"].median()),
            "weighted_mean_signed_spearman": float((sub["spearman"].values * w).sum() / (w.sum() + 1e-12)),
            "weighted_mean_abs_spearman": float((sub["abs_spearman"].values * w).sum() / (w.sum() + 1e-12)),
            "weighted_mean_edge_weight": float((sub["model_edge_weight"].values * w).sum() / (w.sum() + 1e-12)),
            "weighted_mean_combined_score": float((sub["combined_edge_score"].values * w).sum() / (w.sum() + 1e-12)),
            "top5_gene_contribution_fraction": float(
                sub.sort_values("combined_edge_score", ascending=False)["combined_edge_score"].head(5).sum()
                / (sub["combined_edge_score"].sum() + 1e-12)
            ),
        })

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    df.to_csv(out / "top_tf_score_decomposition.csv", index=False)
    log("[SAVE]", out / "top_tf_score_decomposition.csv")

    show_cols = [
        "activity_score",
        "selected_set_mean_continuity",
        "support_gene_count",
        "median_abs_spearman",
        "median_edge_weight",
        "top5_gene_contribution_fraction",
    ]
    M = df[show_cols].values.astype(float)
    M = np.column_stack([minmax01(M[:, i]) for i in range(M.shape[1])])

    fig = plt.figure(figsize=(10, max(4.5, 0.45 * len(df))), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["tf"].tolist(), fontsize=9)
    ax.set_xticks(np.arange(len(show_cols)))
    ax.set_xticklabels(show_cols, rotation=35, ha="right")
    ax.set_title("QC1 | Top TF score decomposition (normalized)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save_figure_both(fig, out / "top_tf_score_decomposition.png", out / "top_tf_score_decomposition.svg")


# ============================================================
# 12) QC2：TF-target gene 一致性
# ============================================================
def qc2_target_consistency(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "02_target_consistency")

    valid_gene_rows = []
    for j in range(ctx.X_log1p.shape[1]):
        train_gid = int(ctx.sample.ext2train[j])
        if train_gid < 0 or train_gid >= ctx.N_GENE_DIM:
            continue
        y = ctx.X_log1p[:, j]
        n_expr = int((y > 0).sum())
        y_std = float(np.std(y))
        if n_expr < MIN_CELLS_EXPRESSED:
            continue
        if y_std < MIN_GENE_STD:
            continue
        valid_gene_rows.append((j, train_gid, ctx.gene_names_norm[j]))

    valid_df = pd.DataFrame(valid_gene_rows, columns=["gene_col", "train_gid", "gene_name"])
    valid_gid_set = set(valid_df["train_gid"].tolist())

    rows = []
    for _, r in ctx.full_final.iterrows():
        pid = int(r["tf_pid"])
        tf_global_id = int(r["tf_global_id"])
        tf = str(r["tf"])
        tf_score = ctx.pred[:, pid]

        corr_all = []
        for _, gr in valid_df.iterrows():
            j = int(gr["gene_col"])
            corr_all.append(_safe_spearman(ctx.X_log1p[:, j], tf_score))

        vv = valid_df.copy()
        vv["corr_to_tf"] = np.asarray(corr_all, dtype=float)

        target_gids = [g for g in ctx.tf2targets.get(tf_global_id, []) if g in valid_gid_set]
        target_mask = vv["train_gid"].isin(target_gids).values
        bg_mask = ~target_mask

        target_corr = vv.loc[target_mask, "corr_to_tf"].dropna().values
        bg_corr = vv.loc[bg_mask, "corr_to_tf"].dropna().values
        stat, p = _safe_mwu(target_corr, bg_corr)

        rows.append({
            "tf": tf,
            "tf_pid": pid,
            "tf_global_id": tf_global_id,
            "n_target_genes_in_sample": int(target_mask.sum()),
            "n_bg_genes": int(bg_mask.sum()),
            "target_corr_mean": float(np.nanmean(target_corr)) if len(target_corr) else np.nan,
            "target_corr_median": float(np.nanmedian(target_corr)) if len(target_corr) else np.nan,
            "bg_corr_mean": float(np.nanmean(bg_corr)) if len(bg_corr) else np.nan,
            "bg_corr_median": float(np.nanmedian(bg_corr)) if len(bg_corr) else np.nan,
            "delta_median_target_minus_bg": float(np.nanmedian(target_corr) - np.nanmedian(bg_corr)) if len(target_corr) and len(bg_corr) else np.nan,
            "mwu_stat": stat,
            "mwu_pvalue": p,
        })

        fig = plt.figure(figsize=(5.2, 4.3), dpi=FIG_DPI)
        ax = fig.add_subplot(111)
        ax.boxplot([target_corr, bg_corr], labels=["targets", "background"], showfliers=False)
        ax.set_title(f"QC2 | {tf}\ntarget-vs-background correlation")
        ax.set_ylabel("Spearman(gene, TF score)")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        save_figure_both(fig, out / f"{tf}_target_consistency.png", out / f"{tf}_target_consistency.svg")

    df = pd.DataFrame(rows).sort_values("delta_median_target_minus_bg", ascending=False).reset_index(drop=True)
    df.to_csv(out / "target_consistency_summary.csv", index=False)
    log("[SAVE]", out / "target_consistency_summary.csv")


# ============================================================
# 13) QC3：bootstrap 稳定性
# ============================================================
def qc3_bootstrap_stability(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "03_bootstrap_stability")
    rng = np.random.default_rng(SEED)

    rank_rows = []
    score_rows = []

    n_cells = ctx.pred.shape[0]
    take = max(20, int(round(n_cells * BOOT_FRAC)))

    for b in range(BOOT_N):
        idx = np.sort(rng.choice(np.arange(n_cells), size=take, replace=False))
        pred_sub = ctx.pred[idx]
        X_sub = ctx.X_log1p[idx]

        _, _, df_final_b, _ = rank_from_arrays(
            ctx, pred_sub, X_sub,
            use_prior=True, use_edge=True, use_continuity=True,
            tag=f"boot_{b:03d}"
        )

        rank_map = {str(x["tf"]): int(x["rank"]) for _, x in df_final_b.iterrows()}
        score_map = {str(x["tf"]): float(x["final_score"]) for _, x in df_final_b.iterrows()}

        for tf in ctx.full_final["tf"].tolist():
            rank_rows.append({
                "bootstrap_id": b,
                "tf": tf,
                "selected": int(tf in rank_map),
                "rank": int(rank_map[tf]) if tf in rank_map else np.nan,
            })
            score_rows.append({
                "bootstrap_id": b,
                "tf": tf,
                "final_score": float(score_map[tf]) if tf in score_map else np.nan,
            })

        if (b + 1) % 10 == 0:
            log(f"[BOOT] {b+1}/{BOOT_N}")

    df_rank = pd.DataFrame(rank_rows)
    df_score = pd.DataFrame(score_rows)

    rows = []
    for tf in ctx.full_final["tf"].tolist():
        subr = df_rank[df_rank["tf"] == tf]
        subs = df_score[df_score["tf"] == tf]
        rows.append({
            "tf": tf,
            "selection_frequency": float(subr["selected"].mean()),
            "rank_mean_when_selected": float(subr["rank"].dropna().mean()) if subr["rank"].notna().any() else np.nan,
            "rank_std_when_selected": float(subr["rank"].dropna().std()) if subr["rank"].notna().sum() > 1 else np.nan,
            "final_score_mean": float(subs["final_score"].dropna().mean()) if subs["final_score"].notna().any() else np.nan,
            "final_score_std": float(subs["final_score"].dropna().std()) if subs["final_score"].notna().sum() > 1 else np.nan,
        })

    df_summary = pd.DataFrame(rows).sort_values("selection_frequency", ascending=False).reset_index(drop=True)
    df_rank.to_csv(out / "bootstrap_rank_details.csv", index=False)
    df_score.to_csv(out / "bootstrap_score_details.csv", index=False)
    df_summary.to_csv(out / "bootstrap_stability_summary.csv", index=False)
    log("[SAVE]", out / "bootstrap_stability_summary.csv")

    fig = plt.figure(figsize=(7.0, max(4.2, 0.42 * len(df_summary))), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    y = np.arange(len(df_summary))
    ax.barh(y, df_summary["selection_frequency"].values, color=BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels(df_summary["tf"].tolist(), fontsize=9)
    ax.set_xlabel("Bootstrap selection frequency")
    ax.set_title("QC3 | Bootstrap stability")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure_both(fig, out / "bootstrap_selection_frequency.png", out / "bootstrap_selection_frequency.svg")


# ============================================================
# 14) QC4：消融实验
# ============================================================
def qc4_ablation(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "04_ablation")

    modes = {
        "corr_only": dict(use_prior=False, use_edge=False, use_continuity=False),
        "corr_plus_prior": dict(use_prior=True, use_edge=False, use_continuity=False),
        "corr_plus_latent": dict(use_prior=False, use_edge=True, use_continuity=False),
        "full_no_continuity": dict(use_prior=True, use_edge=True, use_continuity=False),
        "full": dict(use_prior=True, use_edge=True, use_continuity=True),
    }

    all_final = []
    compare_rows = []
    full_top = set(ctx.full_final["tf"].tolist())

    for name, kw in modes.items():
        _, df_tf, df_final, _ = rank_from_arrays(ctx, ctx.pred, ctx.X_log1p, tag=name, **kw)
        df_tf.to_csv(out / f"{name}_tf_summary.csv", index=False)
        df_final.to_csv(out / f"{name}_final_top_tfs.csv", index=False)
        all_final.append(df_final)

        top_set = set(df_final["tf"].tolist())
        jacc = float(len(full_top & top_set) / len(full_top | top_set)) if len(full_top | top_set) else np.nan

        compare_rows.append({
            "mode": name,
            "n_top_tf": int(len(top_set)),
            "jaccard_vs_full": jacc,
            "mean_activity_score_top": float(df_final["activity_score"].mean()),
            "mean_final_score_top": float(df_final["final_score"].mean()),
        })

    df_compare = pd.DataFrame(compare_rows).sort_values("jaccard_vs_full", ascending=False).reset_index(drop=True)
    df_compare.to_csv(out / "ablation_compare_summary.csv", index=False)
    log("[SAVE]", out / "ablation_compare_summary.csv")

    all_final_cat = pd.concat(all_final, axis=0, ignore_index=True)
    pivot = all_final_cat.pivot_table(index="tf", columns="mode", values="rank", aggfunc="first")
    pivot = pivot.reindex(ctx.full_final["tf"].tolist())

    fig = plt.figure(figsize=(8.2, max(4.5, 0.4 * pivot.shape[0])), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    arr = pivot.values.astype(float)
    arr2 = np.where(np.isnan(arr), np.nanmax(np.nan_to_num(arr, nan=0.0)) + 1.0, arr)
    im = ax.imshow(arr2, aspect="auto")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right")
    ax.set_title("QC4 | Top TF rank across ablations")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save_figure_both(fig, out / "ablation_rank_heatmap.png", out / "ablation_rank_heatmap.svg")


# ============================================================
# 15) QC5：subgroup / pseudo-bulk consistency
# ============================================================
def qc5_subgroup_consistency(ctx: QCContext, use_group_ids: bool = False):
    out = ensure_dir(OUT_DIR / "05_subgroup_consistency")

    groups = ctx.group_ids if use_group_ids else ctx.raw_cluster
    label_name = "group_id" if use_group_ids else "raw_cluster"

    uniq, cnt = np.unique(groups, return_counts=True)
    keep = uniq[cnt >= SUBGROUP_MIN_CELLS]

    rows = []
    rank_rows = []

    for g in keep.tolist():
        idx = np.where(groups == g)[0]
        _, _, df_final_g, _ = rank_from_arrays(
            ctx, ctx.pred[idx], ctx.X_log1p[idx],
            use_prior=True, use_edge=True, use_continuity=True,
            tag=f"{label_name}_{g}"
        )

        top_set = set(df_final_g["tf"].tolist())
        full_set = set(ctx.full_final["tf"].tolist())
        jacc = float(len(top_set & full_set) / len(top_set | full_set)) if len(top_set | full_set) else np.nan

        rows.append({
            label_name: int(g),
            "n_cells": int(len(idx)),
            "jaccard_vs_full": jacc,
            "mean_final_score_top": float(df_final_g["final_score"].mean()),
        })

        for _, r in df_final_g.iterrows():
            rank_rows.append({
                label_name: int(g),
                "n_cells": int(len(idx)),
                "tf": str(r["tf"]),
                "rank": int(r["rank"]),
                "final_score": float(r["final_score"]),
            })

    df_sum = pd.DataFrame(rows).sort_values("jaccard_vs_full", ascending=False).reset_index(drop=True)
    df_rank = pd.DataFrame(rank_rows)

    df_sum.to_csv(out / f"{label_name}_consistency_summary.csv", index=False)
    df_rank.to_csv(out / f"{label_name}_top_tf_ranks.csv", index=False)
    log("[SAVE]", out / f"{label_name}_consistency_summary.csv")

    if len(df_rank) > 0:
        pivot = df_rank.pivot_table(index="tf", columns=label_name, values="rank", aggfunc="first")
        fig = plt.figure(figsize=(10, max(4.5, 0.4 * pivot.shape[0])), dpi=FIG_DPI)
        ax = fig.add_subplot(111)
        arr = pivot.values.astype(float)
        arr2 = np.where(np.isnan(arr), np.nanmax(np.nan_to_num(arr, nan=0.0)) + 1.0, arr)
        im = ax.imshow(arr2, aspect="auto")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(x) for x in pivot.columns.tolist()], rotation=35, ha="right")
        ax.set_title(f"QC5 | rank consistency across {label_name}")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        save_figure_both(fig, out / f"{label_name}_rank_heatmap.png", out / f"{label_name}_rank_heatmap.svg")


# ============================================================
# 16) QC6：正负方向一致性
# ============================================================
def qc6_sign_consistency(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "06_sign_consistency")

    rows = []
    for _, r in ctx.full_final.iterrows():
        pid = int(r["tf_pid"])
        tf = str(r["tf"])
        sub = ctx.full_edges[ctx.full_edges["tf_pid"] == pid].copy()

        if len(sub) == 0:
            continue

        pos = sub[sub["spearman"] > 0].copy()
        neg = sub[sub["spearman"] < 0].copy()

        pos_w = float((pos["gene_weight"] * pos["combined_edge_score"]).sum()) if len(pos) else 0.0
        neg_w = float((neg["gene_weight"] * neg["combined_edge_score"]).sum()) if len(neg) else 0.0
        total_w = pos_w + neg_w + 1e-12

        rows.append({
            "tf": tf,
            "tf_pid": pid,
            "n_support_genes": int(len(sub)),
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "positive_ratio_count": float(len(pos) / max(1, len(sub))),
            "negative_ratio_count": float(len(neg) / max(1, len(sub))),
            "positive_ratio_weighted": float(pos_w / total_w),
            "negative_ratio_weighted": float(neg_w / total_w),
            "weighted_mean_spearman": float((sub["spearman"] * sub["gene_weight"]).sum() / (sub["gene_weight"].sum() + 1e-12)),
        })

    df = pd.DataFrame(rows).sort_values("positive_ratio_weighted", ascending=False).reset_index(drop=True)
    df.to_csv(out / "sign_consistency_summary.csv", index=False)
    log("[SAVE]", out / "sign_consistency_summary.csv")

    fig = plt.figure(figsize=(7.2, max(4.5, 0.42 * len(df))), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    y = np.arange(len(df))
    ax.barh(y, df["positive_ratio_weighted"].values, color=GREEN, label="positive")
    ax.barh(y, -df["negative_ratio_weighted"].values, color=RED, label="negative")
    ax.set_yticks(y)
    ax.set_yticklabels(df["tf"].tolist(), fontsize=9)
    ax.set_xlabel("Weighted sign ratio (+ / -)")
    ax.set_title("QC6 | Sign consistency of supporting genes")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure_both(fig, out / "sign_consistency.png", out / "sign_consistency.svg")


# ============================================================
# 17) QC7：prior degree bias
# ============================================================
def qc7_prior_degree_bias(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "07_prior_degree_bias")

    degree_rows = []
    for pid, tf_global_id in enumerate(ctx.tf_in_prior):
        degree_rows.append({
            "tf_pid": pid,
            "tf_global_id": int(tf_global_id),
            "tf": ctx.tf_names[pid],
            "prior_target_degree": int(len(ctx.tf2targets.get(int(tf_global_id), []))),
        })
    df_deg = pd.DataFrame(degree_rows)

    df = ctx.full_tf.merge(df_deg, on=["tf_pid", "tf_global_id", "tf"], how="left")
    corr_degree_vs_activity = _safe_spearman(df["prior_target_degree"].values, df["activity_score"].values)
    corr_degree_vs_prescore = _safe_spearman(df["prior_target_degree"].values, df["pre_score"].values)

    df["corr_degree_vs_activity"] = corr_degree_vs_activity
    df["corr_degree_vs_prescore"] = corr_degree_vs_prescore
    df.to_csv(out / "prior_degree_bias_summary.csv", index=False)
    log("[SAVE]", out / "prior_degree_bias_summary.csv")

    fig = plt.figure(figsize=(5.2, 4.3), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.scatter(df["prior_target_degree"].values, df["activity_score"].values, s=18, alpha=0.75)
    for _, r in df.nlargest(10, "activity_score").iterrows():
        ax.text(r["prior_target_degree"], r["activity_score"], str(r["tf"]), fontsize=7)
    ax.set_xlabel("Prior target degree")
    ax.set_ylabel("Activity score")
    ax.set_title(f"QC7 | Prior degree bias\nrho(activity)={corr_degree_vs_activity:.3f}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_figure_both(fig, out / "prior_degree_bias.png", out / "prior_degree_bias.svg")


# ============================================================
# 18) QC8：latent edge 正负样本分布
# ============================================================
def qc8_latent_edge_pos_neg(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "08_latent_edge_pos_neg")
    rng = np.random.default_rng(SEED)

    pos_scores = []
    for tf_global_id, gene_list in ctx.tf2targets.items():
        if tf_global_id not in ctx.tfid_to_pid:
            continue
        pid = ctx.tfid_to_pid[tf_global_id]
        tvec = ctx.TF_vec[pid]
        for gid in gene_list:
            if 0 <= gid < ctx.N_GENE_DIM:
                pos_scores.append(float(np.dot(tvec, ctx.G_vec[gid])))
    pos_scores = np.asarray(pos_scores, dtype=np.float64)

    neg_scores = []
    n_tf = len(ctx.tf_in_prior)
    for _ in range(LATENT_NEG_N):
        pid = int(rng.integers(0, n_tf))
        gid = int(rng.integers(0, ctx.N_GENE_DIM))
        neg_scores.append(float(np.dot(ctx.TF_vec[pid], ctx.G_vec[gid])))
    neg_scores = np.asarray(neg_scores, dtype=np.float64)

    stat, p = _safe_mwu(pos_scores, neg_scores)
    df = pd.DataFrame({
        "set": ["prior_positive", "random_negative"],
        "mean": [float(np.nanmean(pos_scores)), float(np.nanmean(neg_scores))],
        "median": [float(np.nanmedian(pos_scores)), float(np.nanmedian(neg_scores))],
        "n": [int(len(pos_scores)), int(len(neg_scores))],
        "mwu_stat": [stat, stat],
        "mwu_pvalue": [p, p],
    })
    df.to_csv(out / "latent_edge_pos_neg_summary.csv", index=False)
    log("[SAVE]", out / "latent_edge_pos_neg_summary.csv")

    fig = plt.figure(figsize=(5.4, 4.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.hist(pos_scores, bins=80, alpha=0.6, density=True, label="prior positive")
    ax.hist(neg_scores, bins=80, alpha=0.6, density=True, label="random negative")
    ax.set_xlabel("gene-TF latent cosine")
    ax.set_ylabel("Density")
    ax.set_title("QC8 | Latent edge positive vs negative")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_figure_both(fig, out / "latent_edge_pos_neg.png", out / "latent_edge_pos_neg.svg")


# ============================================================
# 19) QC9：permutation / null test
# ============================================================
def qc9_permutation_null(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "09_permutation_null")
    rng = np.random.default_rng(SEED)

    rows = []
    rows.append({
        "null_type": "true",
        "rep": -1,
        "mean_top_final_score": float(ctx.full_final["final_score"].mean()),
        "max_top_final_score": float(ctx.full_final["final_score"].max()),
    })

    for rep in range(NULL_N):
        pred_perm = ctx.pred.copy()
        for k in range(pred_perm.shape[1]):
            pred_perm[:, k] = pred_perm[rng.permutation(pred_perm.shape[0]), k]

        _, _, df_final_perm, _ = rank_from_arrays(
            ctx, pred_perm, ctx.X_log1p,
            use_prior=True, use_edge=True, use_continuity=True,
            tag=f"perm_scores_{rep}"
        )
        rows.append({
            "null_type": "permute_tf_scores",
            "rep": rep,
            "mean_top_final_score": float(df_final_perm["final_score"].mean()),
            "max_top_final_score": float(df_final_perm["final_score"].max()),
        })

        rand_map = build_random_candidate_map(ctx, seed=SEED + rep + 1000)
        _, _, df_final_randprior, _ = rank_from_arrays(
            ctx, ctx.pred, ctx.X_log1p,
            use_prior=True, use_edge=True, use_continuity=True,
            random_gene_to_cands=rand_map,
            tag=f"perm_prior_{rep}"
        )
        rows.append({
            "null_type": "randomized_prior",
            "rep": rep,
            "mean_top_final_score": float(df_final_randprior["final_score"].mean()),
            "max_top_final_score": float(df_final_randprior["final_score"].max()),
        })

        if (rep + 1) % 5 == 0:
            log(f"[NULL] {rep+1}/{NULL_N}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "null_test_summary.csv", index=False)
    log("[SAVE]", out / "null_test_summary.csv")

    fig = plt.figure(figsize=(6.0, 4.2), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    groups = ["true", "permute_tf_scores", "randomized_prior"]
    data = [df[df["null_type"] == g]["mean_top_final_score"].values for g in groups]
    ax.boxplot(data, labels=groups, showfliers=False)
    ax.set_ylabel("Mean final score of Top TF set")
    ax.set_title("QC9 | Null tests")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure_both(fig, out / "null_test_boxplot.png", out / "null_test_boxplot.svg")


# ============================================================
# 20) QC10：证据卡片
# ============================================================
def qc10_evidence_cards(ctx: QCContext):
    out = ensure_dir(OUT_DIR / "10_evidence_cards")

    df_decomp = pd.read_csv(OUT_DIR / "01_score_decomposition" / "top_tf_score_decomposition.csv")
    df_sign = pd.read_csv(OUT_DIR / "06_sign_consistency" / "sign_consistency_summary.csv")
    df_target = pd.read_csv(OUT_DIR / "02_target_consistency" / "target_consistency_summary.csv")
    df_boot = pd.read_csv(OUT_DIR / "03_bootstrap_stability" / "bootstrap_stability_summary.csv")

    df_all = (
        ctx.full_final
        .merge(df_decomp, on=["tf", "tf_pid"], how="left", suffixes=("", "_decomp"))
        .merge(df_sign, on=["tf", "tf_pid"], how="left", suffixes=("", "_sign"))
        .merge(df_target, on=["tf", "tf_pid", "tf_global_id"], how="left", suffixes=("", "_target"))
        .merge(df_boot, on=["tf"], how="left", suffixes=("", "_boot"))
    )

    df_all.to_csv(out / "evidence_cards_summary.csv", index=False)
    log("[SAVE]", out / "evidence_cards_summary.csv")

    for _, r in df_all.iterrows():
        tf = str(r["tf"])
        pid = int(r["tf_pid"])
        sub = ctx.full_edges[ctx.full_edges["tf_pid"] == pid].copy()
        sub = sub.sort_values("combined_edge_score", ascending=False).head(20).copy()

        lines = []
        lines.append(f"# TF evidence card: {tf}")
        lines.append("")
        lines.append(f"- final rank: {int(r['rank'])}")
        lines.append(f"- final score: {float(r['final_score']):.4f}")
        lines.append(f"- activity score: {float(r['activity_score']):.4f}")
        lines.append(f"- selected-set mean continuity: {float(r['selected_set_mean_continuity']):.4f}")
        lines.append(f"- support gene count: {int(r.get('support_gene_count', 0))}")
        lines.append(f"- median abs spearman: {float(r.get('median_abs_spearman', np.nan)):.4f}")
        lines.append(f"- median edge weight: {float(r.get('median_edge_weight', np.nan)):.4f}")
        lines.append(f"- positive weighted ratio: {float(r.get('positive_ratio_weighted', np.nan)):.4f}")
        lines.append(f"- negative weighted ratio: {float(r.get('negative_ratio_weighted', np.nan)):.4f}")
        lines.append(f"- target consistency delta median: {float(r.get('delta_median_target_minus_bg', np.nan)):.4f}")
        lines.append(f"- bootstrap selection frequency: {float(r.get('selection_frequency', np.nan)):.4f}")
        lines.append("")
        lines.append("## Top supporting genes")
        if len(sub) == 0:
            lines.append("- none")
        else:
            for _, rr in sub.iterrows():
                lines.append(
                    f"- {rr['gene_name']}: score={rr['combined_edge_score']:.4f}, "
                    f"rho={rr['spearman']:.4f}, edge={rr['model_edge_weight']:.4f}, "
                    f"gene_weight={rr['gene_weight']:.4f}"
                )

        md = out / f"{tf}_evidence_card.md"
        with open(md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log("[SAVE]", md)


# ============================================================
# 21) 额外：full top10 bar
# ============================================================
def save_full_bar(ctx: QCContext):
    plot_final_bar(
        ctx.full_final,
        OUT_DIR / f"{SAMPLE_ID}_final_top10_tfs.png",
        OUT_DIR / f"{SAMPLE_ID}_final_top10_tfs.svg",
        f"Top{len(ctx.full_final)} active TFs | {SAMPLE_ID}\nactivity + network continuity",
    )


# ============================================================
# 22) main
# ============================================================
def main():
    ctx = build_context()
    save_full_bar(ctx)

    qc1_score_decomposition(ctx)
    qc2_target_consistency(ctx)
    qc3_bootstrap_stability(ctx)
    qc4_ablation(ctx)
    qc5_subgroup_consistency(ctx, use_group_ids=False)
    qc5_subgroup_consistency(ctx, use_group_ids=True)
    qc6_sign_consistency(ctx)
    qc7_prior_degree_bias(ctx)
    qc8_latent_edge_pos_neg(ctx)
    qc9_permutation_null(ctx)
    qc10_evidence_cards(ctx)

    log("[DONE] all QC finished")


if __name__ == "__main__":
    main()
