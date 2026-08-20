from __future__ import annotations

# ============================================================
# eval_stageB_fig2lm_standalone_strict.py
# ============================================================
# 一段式完整评估代码：
# - 不 import 本地 stageB_component_ablation_integrated.py
# - 不读取旧 python 脚本
# - 不重新训练
# - 只读取已有数据矩阵、base checkpoint、ablation best_ckpt.pt
#
# 新增 strict Fig.2M-like retention：
#   retained cell =
#       1) own cluster is best matched true cluster
#       2) own_sim > 0
#       3) own_sim - best_other_sim > STRICT_RETENTION_MARGIN
#       4) cell-level predicted-vs-true TF profile Spearman > 0
#
# 输出：
#   $IMAS_PROJECT_ROOT/runs/stageB_component_ablation_integrated/eval_fig2lm_standalone_strict_summary.csv
#   $IMAS_PROJECT_ROOT/runs/stageB_component_ablation_integrated/eval_fig2lm_standalone_strict_plots/
# ============================================================

import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIG
# ============================================================

BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"
GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"
GLOBAL_PEAK_VOCAB = MAPS_ROOT / "global_peak_vocab.txt"

RUN_ROOT = BASE / "runs" / "stageB_component_ablation_integrated"
DATASET = "CRC-Tissue"

BASE_CKPT_MAP = {
    "CRC-Tissue": get_dataset_checkpoint("CRC-Tissue"),
}

MODES = [
    "full_stageB",
    "simple_finetune",
    "no_lowrank_adapter",
    "no_cluster_correction",
    "no_selected_peak_residual",
    "no_group_scale_bias",
    "no_cellgroup_adapter",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Need CUDA"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_CANCER_NAME = None

ANCHOR_DTYPE = np.float16
RNA_TOPK = 2048
EMB_BATCH = 256

AUCELL_TOPK = 500
TF_MIX_ALPHA = 0.6

K_RANK = 32
GROUP_SCALE_ALPHA = 0.12
CLUSTER_CORRECTION_HID = 256
CLUSTER_CORRECTION_USE_RAW_CLUSTER = True

N_CLUSTERS = 16
CLUSTER_ITERS = 25
CLUSTER_TRAIN_MAX = 60000
CLUSTER_NMIN_TRAIN = 100

VAL_FRAC_FROM_TRAIN = 0.10
MAX_CELLS_PER_SAMPLE = None

MAX_TF_FOR_FIG2LM = 512
MIN_CELLS_PER_CLUSTER = 30
STRICT_RETENTION_MARGIN = 0.05

POS_Q = 0.80
NEG_Q = 0.20


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def log(*args):
    print(*args, flush=True)


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
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (sd + 1e-6)


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


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


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


# ============================================================
# 2. ABLATION CONFIG
# ============================================================

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
        cfg.use_selector = False
        cfg.use_lowrank_delta = True
        cfg.use_cancer_adaptor = True
        cfg.use_cellgroup_adaptor = False
        cfg.use_group_scale_bias = False
        cfg.use_selected_peak_residual = False
        cfg.use_cluster_correction = False

    elif mode == "no_lowrank_adapter":
        cfg.use_lowrank_delta = False
        cfg.use_cancer_adaptor = False
        cfg.use_cellgroup_adaptor = True
        cfg.use_group_scale_bias = True

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
        raise ValueError(f"Unknown mode: {mode}")

    return cfg


def load_ablation_cfg_from_ckpt(ckpt_obj: dict, mode: str) -> AblationConfig:
    if "ablation_cfg" in ckpt_obj and isinstance(ckpt_obj["ablation_cfg"], dict):
        d = ckpt_obj["ablation_cfg"]
        try:
            return AblationConfig(**d)
        except TypeError:
            return make_ablation_config(mode)

    return make_ablation_config(mode)


# ============================================================
# 3. MODEL CLASSES
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
            nn.Linear(d, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, d),
        )
        self.atac_proj = nn.Sequential(
            nn.Linear(d, hid),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hid, d),
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

        out = torch.zeros(
            (offsets.shape[0], x.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
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
        ablation_cfg: AblationConfig | None = None,
    ):
        super().__init__()
        self.d = int(d)
        self.k = int(k)
        self.peak_hash_size = int(peak_hash_size)
        self.ablation_cfg = ablation_cfg or AblationConfig(mode="full_stageB")

        self.e = nn.Parameter(torch.zeros((e_dim,), dtype=torch.float32))
        nn.init.normal_(self.e, mean=0.0, std=0.02)

        self.mlp = nn.Sequential(
            nn.Linear(e_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, 4 * k * d),
        )

        self.C_tf = nn.Embedding(n_tf, k)
        self.C_gene = nn.Embedding(n_gene, k)
        self.C_peak_hash = nn.Embedding(self.peak_hash_size, k)
        self.C_cancer = nn.Embedding(n_cancer, k)

        self.S_group = nn.Embedding(n_group, d)
        self.B_group = nn.Embedding(n_group, d)

        for emb in [
            self.C_tf,
            self.C_gene,
            self.C_peak_hash,
            self.C_cancer,
            self.S_group,
            self.B_group,
        ]:
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
        self.mlp = nn.Sequential(
            nn.Linear(32 + summary_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        self.register_buffer(
            "summary",
            torch.zeros((summary_dim,), dtype=torch.float32),
            persistent=True,
        )

    def forward(self) -> torch.Tensor:
        x = torch.cat([self.e, self.summary], dim=0)
        logits = self.mlp(x)
        w = F.softplus(logits) + 1e-6
        return w / (w.mean() + 1e-12)


class SelectedPeakResidual(nn.Module):
    def __init__(self, selected_peak_raw_ids: np.ndarray, d: int):
        super().__init__()
        ids = np.asarray(selected_peak_raw_ids, dtype=np.int64)
        self.d = int(d)

        self.register_buffer(
            "selected_peak_raw_ids",
            torch.from_numpy(ids),
            persistent=True,
        )
        self.emb = nn.Embedding(max(1, ids.size), self.d)
        nn.init.zeros_(self.emb.weight)
        self.raw2local = {int(pk): i for i, pk in enumerate(ids.tolist())}

    @property
    def n_selected(self) -> int:
        return int(self.selected_peak_raw_ids.numel())

    def lookup(self, peak_idx_raw: torch.Tensor) -> torch.Tensor:
        if self.n_selected <= 0 or peak_idx_raw.numel() == 0:
            return torch.zeros(
                (peak_idx_raw.numel(), self.d),
                device=peak_idx_raw.device,
                dtype=torch.float32,
            )

        raw_np = peak_idx_raw.detach().cpu().numpy().astype(np.int64)
        local_np = np.full((raw_np.size,), -1, dtype=np.int64)

        for i, pk in enumerate(raw_np.tolist()):
            local_np[i] = self.raw2local.get(int(pk), -1)

        out = torch.zeros(
            (raw_np.size, self.d),
            device=peak_idx_raw.device,
            dtype=self.emb.weight.dtype,
        )

        m = local_np >= 0
        if m.any():
            pos_t = torch.from_numpy(np.where(m)[0]).to(peak_idx_raw.device)
            idx_t = torch.from_numpy(local_np[m]).to(peak_idx_raw.device)
            out[pos_t] = self.emb(idx_t)

        return out


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
        return self.mlp(torch.cat([z_cell, cluster_summary], dim=-1))


# ============================================================
# 4. DATA LOADING
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
            return p

    raise AssertionError(
        f"Missing anchor for sid={sid}. Tried:\n"
        + "\n".join(str(x) for x in candidates)
    )


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
        assert anchor_path.exists(), f"Missing anchor: {anchor_path}"

        z = np.load(meta_path, allow_pickle=True)
        assert "gene_names" in z, f"{meta_path.name}: missing gene_names"

        gene_names = np.asarray(z["gene_names"]).astype(str)
        cells = np.asarray(
            [norm_barcode(x) for x in g["cell_id"].astype(str).tolist()],
            dtype=object,
        )
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

        log(
            f"[LOAD] {dataset_name}/{sid} "
            f"cells={s.counts_shape[0]} genes={s.counts_shape[1]} "
            f"matched_genes={matched} anchor_shape={anchor_X.shape}"
        )

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
    train = []
    test = []

    for sid, rid, key in all_cells:
        if split_5050(key) == 0:
            train.append((sid, rid))
        else:
            test.append((sid, rid))

    return train, test


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


# ============================================================
# 5. ENCODING / PROXIES / CLUSTERING
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
        b = cell_list[st: st + EMB_BATCH]

        idx_cat, val_cat, offsets, cancer_ids = build_tokens_for_cells(
            b,
            sid2sample,
            cancer_id_int,
            proj_store_cache,
        )

        z = model.encode_rna(
            torch.from_numpy(idx_cat).to(DEVICE),
            torch.from_numpy(val_cat).to(DEVICE),
            torch.from_numpy(offsets).to(DEVICE),
            torch.from_numpy(cancer_ids).to(DEVICE),
        )

        Zs.append(z.detach().cpu().numpy())

    return np.concatenate(Zs, axis=0).astype(np.float32)


def regproxy_aucell_lite_from_top(
    cell_list,
    sid2sample,
    tf2targets,
    tf_in_prior,
    topk,
    N_GENE_DIM,
):
    n_tf = len(tf_in_prior)
    out = np.full((len(cell_list), n_tf), np.nan, dtype=np.float32)

    tf_targets = [
        set(int(g) for g in tf2targets[int(tf_id)])
        for tf_id in tf_in_prior
    ]

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
                (np.asarray(rows, np.int64), np.asarray(cols, np.int64)),
            ),
            shape=(int(s.counts_shape[1]), int(n_tf)),
        ).tocsr()

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
        out[:, j] = (
            TF_MIX_ALPHA * zscore_safe(aucell[:, j])
            + (1.0 - TF_MIX_ALPHA) * zscore_safe(meanlog[:, j])
        )

    return out


def spherical_kmeans_fit(
    Z_train: np.ndarray,
    k: int,
    iters: int = 25,
    max_train: int = 60000,
):
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


def build_group_ids_from_clusters(
    train_cluster: np.ndarray,
    all_cluster: np.ndarray,
    nmin: int,
):
    cnt = np.bincount(train_cluster, minlength=int(train_cluster.max()) + 1)

    keep = np.where(cnt >= int(nmin))[0].tolist()
    cl2gid = {int(cl): i + 1 for i, cl in enumerate(sorted(keep))}

    group_ids = np.zeros_like(all_cluster, dtype=np.int64)

    for i, cl in enumerate(all_cluster.tolist()):
        group_ids[i] = int(cl2gid.get(int(cl), 0))

    return group_ids, cl2gid, cnt


# ============================================================
# 6. CLUSTER SUMMARY
# ============================================================

def build_cluster_summaries(
    cell_list,
    sid2sample,
    Z,
    cluster_ids,
    genes_vocab,
):
    summaries = {}

    for cid in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cid)[0]

        if idx.size == 0:
            continue

        z_mean = Z[idx].mean(axis=0).astype(np.float32)

        rna_nnz = []
        rna_logmean = []

        for ii in idx.tolist():
            sid, rid = cell_list[ii]
            s = sid2sample[sid]
            row = get_counts_row_dense(s, rid)

            rna_nnz.append(float((row > 0).mean()))

            nz = row[row > 0]
            rna_logmean.append(float(np.log1p(nz).mean()) if nz.size else 0.0)

        summaries[int(cid)] = {
            "z_mean": z_mean,
            "rna_nnz_mean": float(np.mean(rna_nnz)) if rna_nnz else 0.0,
            "rna_logmean": float(np.mean(rna_logmean)) if rna_logmean else 0.0,
            "size": int(idx.size),
        }

    return summaries


def cluster_summary_to_vector(summary: dict) -> np.ndarray:
    z_mean = np.asarray(summary["z_mean"], dtype=np.float32).reshape(-1)

    meta = np.asarray(
        [
            float(summary["rna_nnz_mean"]),
            float(summary["rna_logmean"]),
            float(summary["size"]),
        ],
        dtype=np.float32,
    )

    return np.concatenate([z_mean, meta], axis=0).astype(np.float32)


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
        idx = []
        for cid in cluster_ids.detach().cpu().numpy().tolist():
            idx.append(self.cid2row.get(int(cid), 0))

        idx_t = torch.tensor(idx, device=cluster_ids.device, dtype=torch.long)

        return self.summary_table[idx_t]


# ============================================================
# 7. PRIORS
# ============================================================

def build_gene_priors(tfs_vocab, N_GENE_DIM):
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

    return tf2targets, tf_in_prior


def build_gene_peak_priors(N_GENE_DIM):
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

    return gene2peaks


# ============================================================
# 8. CONTEXT AND SCORING
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


def apply_cluster_correction(
    z: torch.Tensor,
    cluster_ids_np: np.ndarray,
    corr_head,
    alpha: float,
    ctx: RunContext,
) -> torch.Tensor:
    if (not ctx.cfg.use_cluster_correction) or corr_head is None:
        return z

    cl_t = torch.from_numpy(cluster_ids_np.astype(np.int64)).to(DEVICE)
    summary = ctx.cluster_summary_table.lookup(cl_t)
    dz = corr_head(z, summary)

    return F.normalize(z + float(alpha) * dz, dim=-1)


def adjust_Z(
    Zt,
    group_ids_np,
    cancer_ids_np,
    cluster_ids_np,
    V_cancer,
    task: str,
    ctx: RunContext,
):
    Zadj = Zt

    if ctx.cfg.use_cellgroup_adaptor:
        gid_t = torch.from_numpy(group_ids_np.astype(np.int64)).to(DEVICE)
        Zadj = ctx.adapter.apply_group_to_Z(Zadj, gid_t)

    if ctx.cfg.use_cancer_adaptor and ctx.cfg.use_lowrank_delta:
        cid_t = torch.from_numpy(cancer_ids_np.astype(np.int64)).to(DEVICE)
        Zadj = Zadj + ctx.adapter.delta_cancer(cid_t, V_cancer)

    Zadj = F.normalize(Zadj, dim=-1)

    if task == "tf":
        return apply_cluster_correction(Zadj, cluster_ids_np, ctx.tf_corr_head, 0.25, ctx)

    if task == "rna":
        return apply_cluster_correction(Zadj, cluster_ids_np, ctx.rna_corr_head, 0.25, ctx)

    if task == "re":
        return apply_cluster_correction(Zadj, cluster_ids_np, ctx.re_corr_head, 0.25, ctx)

    return Zadj


def tf_base(tf_pid: torch.Tensor, ctx: RunContext) -> torch.Tensor:
    tf_ids = torch.tensor(
        [ctx.tf_in_prior[i] for i in tf_pid.tolist()],
        device=DEVICE,
        dtype=torch.long,
    )

    return ctx.model.tf_in(tf_ids) + ctx.model.tf_in_c(ctx.cid).expand(tf_ids.numel(), -1)


def score_tf_vec(
    tf_pid_t: torch.Tensor,
    V_tf: torch.Tensor,
    V_cancer: torch.Tensor,
    ctx: RunContext,
) -> torch.Tensor:
    TF = tf_base(tf_pid_t, ctx)

    if ctx.cfg.use_lowrank_delta:
        TF = TF + ctx.adapter.delta_tf(tf_pid_t, V_tf)

    if ctx.cfg.use_cancer_adaptor and ctx.cfg.use_lowrank_delta:
        c_tok = torch.full(
            (tf_pid_t.numel(),),
            int(ctx.default_cancer_id),
            device=DEVICE,
            dtype=torch.long,
        )
        TF = TF + ctx.adapter.delta_cancer(c_tok, V_cancer)

    return F.normalize(TF, dim=-1)


def infer_selected_peak_ids_from_ckpt(ckpt_obj: dict) -> np.ndarray:
    st = ckpt_obj.get("peak_residual", None)

    if st is None:
        return np.zeros((0,), dtype=np.int64)

    if "selected_peak_raw_ids" in st:
        x = st["selected_peak_raw_ids"]

        if torch.is_tensor(x):
            return x.detach().cpu().numpy().astype(np.int64)

        return np.asarray(x, dtype=np.int64)

    return np.zeros((0,), dtype=np.int64)


# ============================================================
# 9. STRICT FIG2L-M EVALUATION
# ============================================================

@torch.no_grad()
def compute_tf_pred_truth_matrix(
    Z_np: np.ndarray,
    group_ids: np.ndarray,
    cancer_ids: np.ndarray,
    cluster_ids: np.ndarray,
    REG_tf: np.ndarray,
    ctx: RunContext,
    max_tf: int | None = 512,
):
    if Z_np is None or len(Z_np) == 0 or REG_tf is None:
        return None

    ctx.adapter.eval()

    for m in [
        ctx.selector,
        ctx.peak_residual,
        ctx.tf_corr_head,
        ctx.rna_corr_head,
        ctx.re_corr_head,
    ]:
        if m is not None:
            m.eval()

    V_tf, V_gene, V_peak, V_cancer = ctx.adapter()

    tf_pids = np.arange(len(ctx.tf_in_prior), dtype=np.int64)

    finite_frac = np.isfinite(REG_tf).mean(axis=0)
    sd = np.nanstd(REG_tf, axis=0)

    keep_tf = np.where((finite_frac > 0.8) & np.isfinite(sd) & (sd > 1e-6))[0]

    if keep_tf.size == 0:
        return None

    tf_pids = tf_pids[keep_tf]

    if max_tf is not None and tf_pids.size > int(max_tf):
        sd2 = sd[tf_pids]
        order = np.argsort(sd2)[::-1]
        tf_pids = tf_pids[order[: int(max_tf)]]

    Zt = torch.from_numpy(Z_np).to(DEVICE)

    Zadj = adjust_Z(
        Zt,
        group_ids,
        cancer_ids,
        cluster_ids,
        V_cancer,
        "tf",
        ctx,
    )

    pred_chunks = []

    for st in range(0, tf_pids.size, 256):
        pid_np = tf_pids[st: st + 256]

        TF = score_tf_vec(
            torch.from_numpy(pid_np).to(DEVICE),
            V_tf,
            V_cancer,
            ctx,
        )

        S_pred = (Zadj @ TF.T).detach().cpu().numpy().astype(np.float32)
        pred_chunks.append(S_pred)

    pred = np.concatenate(pred_chunks, axis=1) if pred_chunks else np.zeros((Z_np.shape[0], 0), np.float32)
    truth = REG_tf[:, tf_pids].astype(np.float32)

    pred_z = np.zeros_like(pred, dtype=np.float32)
    truth_z = np.zeros_like(truth, dtype=np.float32)

    for j in range(pred.shape[1]):
        pred_z[:, j] = zscore_safe(pred[:, j])
        truth_z[:, j] = zscore_safe(truth[:, j])

    return {
        "pred": pred_z,
        "truth": truth_z,
        "tf_pids": tf_pids,
    }


def evaluate_cluster_correspondence_and_retention(
    pred: np.ndarray,
    truth: np.ndarray,
    cluster_ids: np.ndarray,
    min_cells_per_cluster: int = 30,
    strict_margin: float = STRICT_RETENTION_MARGIN,
):
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    cluster_ids = np.asarray(cluster_ids)

    uniq = []
    for c in sorted(np.unique(cluster_ids).tolist()):
        if int((cluster_ids == c).sum()) >= int(min_cells_per_cluster):
            uniq.append(c)

    if len(uniq) < 2:
        return {
            "n_clusters_eval": int(len(uniq)),
            "diag_score": np.nan,
            "offdiag_score": np.nan,
            "diag_offdiag_delta": np.nan,
            "diag_offdiag_ratio": np.nan,
            "match_accuracy": np.nan,
            "overall_retained": np.nan,
            "cluster_coverage": np.nan,
            "mean_cluster_retention": np.nan,
            "distribution_preservation": np.nan,
            "strict_overall_retained": np.nan,
            "strict_cluster_coverage": np.nan,
            "strict_mean_cluster_retention": np.nan,
            "strict_distribution_preservation": np.nan,
            "mean_own_margin": np.nan,
            "mean_cell_profile_spearman": np.nan,
            "correspondence_matrix": None,
            "cluster_ids_eval": uniq,
        }

    pred_mu = []
    truth_mu = []
    cluster_sizes = []

    for c in uniq:
        m = cluster_ids == c
        cluster_sizes.append(int(m.sum()))
        pred_mu.append(np.nanmean(pred[m], axis=0))
        truth_mu.append(np.nanmean(truth[m], axis=0))

    pred_mu = np.vstack(pred_mu).astype(np.float32)
    truth_mu = np.vstack(truth_mu).astype(np.float32)
    cluster_sizes = np.asarray(cluster_sizes, dtype=np.float32)

    pred_mu_z = pred_mu.copy()
    truth_mu_z = truth_mu.copy()

    for j in range(pred_mu.shape[1]):
        pred_mu_z[:, j] = zscore_safe(pred_mu[:, j])
        truth_mu_z[:, j] = zscore_safe(truth_mu[:, j])

    corr_mat = cosine_matrix(pred_mu_z, truth_mu_z)

    diag = np.diag(corr_mat)
    off = corr_mat[~np.eye(corr_mat.shape[0], dtype=bool)]

    diag_score = float(np.nanmean(diag))
    offdiag_score = float(np.nanmean(off))
    diag_offdiag_delta = float(diag_score - offdiag_score)
    diag_offdiag_ratio = float((diag_score + 1.0) / (offdiag_score + 1.0 + 1e-12))

    pred_match = np.nanargmax(corr_mat, axis=1)
    match_accuracy = float(np.mean(pred_match == np.arange(len(uniq))))

    truth_cluster_norm = truth_mu_z / (np.linalg.norm(truth_mu_z, axis=1, keepdims=True) + 1e-12)

    pred_cell = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred_cell_norm = pred_cell / (np.linalg.norm(pred_cell, axis=1, keepdims=True) + 1e-12)

    sim_cell_to_true_cluster = pred_cell_norm @ truth_cluster_norm.T

    cell_to_cluster_pos = {c: i for i, c in enumerate(uniq)}

    loose_retained = np.zeros((cluster_ids.shape[0],), dtype=bool)
    strict_retained = np.zeros((cluster_ids.shape[0],), dtype=bool)
    valid_cell = np.zeros((cluster_ids.shape[0],), dtype=bool)

    own_margins = []
    cell_profile_spears = []

    for i, c in enumerate(cluster_ids.tolist()):
        if c not in cell_to_cluster_pos:
            continue

        valid_cell[i] = True
        own_pos = cell_to_cluster_pos[c]

        sims = sim_cell_to_true_cluster[i]
        best_pos = int(np.nanargmax(sims))
        own_sim = float(sims[own_pos])

        if len(sims) > 1:
            other_sims = np.delete(sims, own_pos)
            best_other_sim = float(np.nanmax(other_sims))
        else:
            best_other_sim = -np.inf

        own_margin = own_sim - best_other_sim
        own_margins.append(own_margin)

        cell_spear = spearmanr_fast(pred[i], truth[i])
        cell_profile_spears.append(cell_spear)

        own_is_best = best_pos == own_pos

        loose_retained[i] = own_is_best and (own_sim > 0.0)

        strict_retained[i] = (
            own_is_best
            and (own_sim > 0.0)
            and (own_margin > float(strict_margin))
            and np.isfinite(cell_spear)
            and (cell_spear > 0.0)
        )

    def summarize_retained(retained_vec: np.ndarray):
        if valid_cell.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan

        overall_retained = float(retained_vec[valid_cell].mean())

        cluster_ret = []
        retained_counts = []
        original_counts = []

        for c in uniq:
            m = cluster_ids == c
            original_counts.append(float(m.sum()))
            retained_counts.append(float((retained_vec & m).sum()))
            cluster_ret.append(float(retained_vec[m].mean()) if m.sum() > 0 else np.nan)

        cluster_ret = np.asarray(cluster_ret, dtype=np.float32)
        retained_counts = np.asarray(retained_counts, dtype=np.float32)
        original_counts = np.asarray(original_counts, dtype=np.float32)

        cluster_coverage = float(np.mean(retained_counts > 0))
        mean_cluster_retention = float(np.nanmean(cluster_ret))

        p0 = original_counts / (original_counts.sum() + 1e-12)

        if retained_counts.sum() <= 0:
            distribution_preservation = 0.0
        else:
            p1 = retained_counts / (retained_counts.sum() + 1e-12)
            tvd = 0.5 * np.abs(p0 - p1).sum()
            distribution_preservation = float(max(0.0, 1.0 - tvd))

        return (
            overall_retained,
            cluster_coverage,
            mean_cluster_retention,
            distribution_preservation,
        )

    (
        overall_retained,
        cluster_coverage,
        mean_cluster_retention,
        distribution_preservation,
    ) = summarize_retained(loose_retained)

    (
        strict_overall_retained,
        strict_cluster_coverage,
        strict_mean_cluster_retention,
        strict_distribution_preservation,
    ) = summarize_retained(strict_retained)

    own_margins = np.asarray(own_margins, dtype=np.float32)
    cell_profile_spears = np.asarray(cell_profile_spears, dtype=np.float32)

    return {
        "n_clusters_eval": int(len(uniq)),
        "diag_score": diag_score,
        "offdiag_score": offdiag_score,
        "diag_offdiag_delta": diag_offdiag_delta,
        "diag_offdiag_ratio": diag_offdiag_ratio,
        "match_accuracy": match_accuracy,

        "overall_retained": overall_retained,
        "cluster_coverage": cluster_coverage,
        "mean_cluster_retention": mean_cluster_retention,
        "distribution_preservation": distribution_preservation,

        "strict_overall_retained": strict_overall_retained,
        "strict_cluster_coverage": strict_cluster_coverage,
        "strict_mean_cluster_retention": strict_mean_cluster_retention,
        "strict_distribution_preservation": strict_distribution_preservation,

        "mean_own_margin": float(np.nanmean(own_margins)) if own_margins.size else np.nan,
        "mean_cell_profile_spearman": float(np.nanmean(cell_profile_spears)) if cell_profile_spears.size else np.nan,

        "correspondence_matrix": corr_mat,
        "cluster_ids_eval": uniq,
    }


def evaluate_tf_trend_consistency_from_cluster_profiles(
    pred: np.ndarray,
    truth: np.ndarray,
    cluster_ids: np.ndarray,
    min_cells_per_cluster: int = 30,
):
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    cluster_ids = np.asarray(cluster_ids)

    uniq = []

    for c in sorted(np.unique(cluster_ids).tolist()):
        if int((cluster_ids == c).sum()) >= int(min_cells_per_cluster):
            uniq.append(c)

    if len(uniq) < 3:
        return {
            "tf_trend_mean_spearman": np.nan,
            "tf_trend_positive_fraction": np.nan,
            "tf_trend_highdev_mean_spearman": np.nan,
            "tf_trend_lowdev_mean_spearman": np.nan,
            "n_tf_trend_eval": 0,
        }

    pred_mu = []
    truth_mu = []

    for c in uniq:
        m = cluster_ids == c
        pred_mu.append(np.nanmean(pred[m], axis=0))
        truth_mu.append(np.nanmean(truth[m], axis=0))

    pred_mu = np.vstack(pred_mu).astype(np.float32)
    truth_mu = np.vstack(truth_mu).astype(np.float32)

    cors = []
    devs = []

    for j in range(pred_mu.shape[1]):
        x = pred_mu[:, j]
        y = truth_mu[:, j]

        if np.nanstd(y) < 1e-6:
            continue

        r = spearmanr_fast(x, y)

        if np.isfinite(r):
            cors.append(float(r))
            devs.append(float(np.nanstd(y)))

    if len(cors) == 0:
        return {
            "tf_trend_mean_spearman": np.nan,
            "tf_trend_positive_fraction": np.nan,
            "tf_trend_highdev_mean_spearman": np.nan,
            "tf_trend_lowdev_mean_spearman": np.nan,
            "n_tf_trend_eval": 0,
        }

    cors = np.asarray(cors, dtype=np.float32)
    devs = np.asarray(devs, dtype=np.float32)

    q = np.nanmedian(devs)
    high = devs >= q
    low = devs < q

    return {
        "tf_trend_mean_spearman": float(np.nanmean(cors)),
        "tf_trend_positive_fraction": float(np.mean(cors > 0)),
        "tf_trend_highdev_mean_spearman": float(np.nanmean(cors[high])) if high.sum() else np.nan,
        "tf_trend_lowdev_mean_spearman": float(np.nanmean(cors[low])) if low.sum() else np.nan,
        "n_tf_trend_eval": int(cors.size),
    }


@torch.no_grad()
def eval_fig2lm_like_metrics(
    Z_np: np.ndarray,
    group_ids: np.ndarray,
    cancer_ids: np.ndarray,
    cluster_ids: np.ndarray,
    REG_tf: np.ndarray,
    ctx: RunContext,
    split_name: str,
):
    mat = compute_tf_pred_truth_matrix(
        Z_np=Z_np,
        group_ids=group_ids,
        cancer_ids=cancer_ids,
        cluster_ids=cluster_ids,
        REG_tf=REG_tf,
        ctx=ctx,
        max_tf=MAX_TF_FOR_FIG2LM,
    )

    if mat is None:
        return {
            f"{split_name}_fig2lm_n_clusters_eval": 0,

            f"{split_name}_fig2l_diag_score": np.nan,
            f"{split_name}_fig2l_offdiag_score": np.nan,
            f"{split_name}_fig2l_diag_offdiag_delta": np.nan,
            f"{split_name}_fig2l_diag_offdiag_ratio": np.nan,
            f"{split_name}_fig2l_match_accuracy": np.nan,

            f"{split_name}_fig2m_overall_retained": np.nan,
            f"{split_name}_fig2m_cluster_coverage": np.nan,
            f"{split_name}_fig2m_mean_cluster_retention": np.nan,
            f"{split_name}_fig2m_distribution_preservation": np.nan,

            f"{split_name}_fig2m_strict_overall_retained": np.nan,
            f"{split_name}_fig2m_strict_cluster_coverage": np.nan,
            f"{split_name}_fig2m_strict_mean_cluster_retention": np.nan,
            f"{split_name}_fig2m_strict_distribution_preservation": np.nan,

            f"{split_name}_fig2m_mean_own_margin": np.nan,
            f"{split_name}_fig2m_mean_cell_profile_spearman": np.nan,

            f"{split_name}_fig2j_tf_trend_mean_spearman": np.nan,
            f"{split_name}_fig2j_tf_trend_positive_fraction": np.nan,
            f"{split_name}_fig2j_tf_trend_highdev_mean_spearman": np.nan,
            f"{split_name}_fig2j_tf_trend_lowdev_mean_spearman": np.nan,
            f"{split_name}_fig2j_n_tf_trend_eval": 0,
        }

    corr_ret = evaluate_cluster_correspondence_and_retention(
        pred=mat["pred"],
        truth=mat["truth"],
        cluster_ids=cluster_ids,
        min_cells_per_cluster=MIN_CELLS_PER_CLUSTER,
        strict_margin=STRICT_RETENTION_MARGIN,
    )

    trend = evaluate_tf_trend_consistency_from_cluster_profiles(
        pred=mat["pred"],
        truth=mat["truth"],
        cluster_ids=cluster_ids,
        min_cells_per_cluster=MIN_CELLS_PER_CLUSTER,
    )

    if corr_ret.get("correspondence_matrix", None) is not None:
        corr_df = pd.DataFrame(
            corr_ret["correspondence_matrix"],
            index=[f"pred_C{int(x)}" for x in corr_ret["cluster_ids_eval"]],
            columns=[f"true_C{int(x)}" for x in corr_ret["cluster_ids_eval"]],
        )
        corr_df.to_csv(ctx.ds_out / f"{split_name}_fig2l_correspondence_matrix_strict.csv")

    return {
        f"{split_name}_fig2lm_n_clusters_eval": int(corr_ret["n_clusters_eval"]),

        f"{split_name}_fig2l_diag_score": float(corr_ret["diag_score"]),
        f"{split_name}_fig2l_offdiag_score": float(corr_ret["offdiag_score"]),
        f"{split_name}_fig2l_diag_offdiag_delta": float(corr_ret["diag_offdiag_delta"]),
        f"{split_name}_fig2l_diag_offdiag_ratio": float(corr_ret["diag_offdiag_ratio"]),
        f"{split_name}_fig2l_match_accuracy": float(corr_ret["match_accuracy"]),

        f"{split_name}_fig2m_overall_retained": float(corr_ret["overall_retained"]),
        f"{split_name}_fig2m_cluster_coverage": float(corr_ret["cluster_coverage"]),
        f"{split_name}_fig2m_mean_cluster_retention": float(corr_ret["mean_cluster_retention"]),
        f"{split_name}_fig2m_distribution_preservation": float(corr_ret["distribution_preservation"]),

        f"{split_name}_fig2m_strict_overall_retained": float(corr_ret["strict_overall_retained"]),
        f"{split_name}_fig2m_strict_cluster_coverage": float(corr_ret["strict_cluster_coverage"]),
        f"{split_name}_fig2m_strict_mean_cluster_retention": float(corr_ret["strict_mean_cluster_retention"]),
        f"{split_name}_fig2m_strict_distribution_preservation": float(corr_ret["strict_distribution_preservation"]),

        f"{split_name}_fig2m_mean_own_margin": float(corr_ret["mean_own_margin"]),
        f"{split_name}_fig2m_mean_cell_profile_spearman": float(corr_ret["mean_cell_profile_spearman"]),

        f"{split_name}_fig2j_tf_trend_mean_spearman": float(trend["tf_trend_mean_spearman"]),
        f"{split_name}_fig2j_tf_trend_positive_fraction": float(trend["tf_trend_positive_fraction"]),
        f"{split_name}_fig2j_tf_trend_highdev_mean_spearman": float(trend["tf_trend_highdev_mean_spearman"]),
        f"{split_name}_fig2j_tf_trend_lowdev_mean_spearman": float(trend["tf_trend_lowdev_mean_spearman"]),
        f"{split_name}_fig2j_n_tf_trend_eval": int(trend["n_tf_trend_eval"]),
    }


# ============================================================
# 10. BUILD SHARED DATA CACHE
# ============================================================

def build_shared_data_cache():
    log("=" * 100)
    log("[BUILD] shared data cache")
    log("=" * 100)

    base_ckpt_path = BASE_CKPT_MAP[DATASET]
    assert base_ckpt_path.exists(), f"Missing base checkpoint: {base_ckpt_path}"

    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)

    state_dict = base_ckpt.get("model", None)
    assert state_dict is not None, f"BASE ckpt missing model weights: {base_ckpt_path}"

    base_cfg = base_ckpt.get("cfg", {}) or {}
    meta = base_ckpt.get("meta", {}) or {}

    for k in [
        "N_GENE_DIM",
        "N_TF_DIM",
        "PEAK_HASH_SIZE",
        "D",
        "HID",
        "DROP",
        "cancer_types",
    ]:
        if k in meta:
            base_cfg[k] = meta[k]

    base_cfg.setdefault("N_GENE_DIM", int(state_dict["gene_in.weight"].shape[0]))
    base_cfg.setdefault("N_TF_DIM", int(state_dict["tf_in.weight"].shape[0]))
    base_cfg.setdefault("PEAK_HASH_SIZE", int(state_dict["peak_in.emb.weight"].shape[0]))
    base_cfg.setdefault("D", int(state_dict["gene_in.weight"].shape[1]))
    base_cfg.setdefault("HID", int(state_dict["rna_proj.0.weight"].shape[0]))
    base_cfg.setdefault("DROP", 0.1)

    if "cancer_types" not in base_cfg:
        n_cancer = int(state_dict["cell_c.weight"].shape[0])
        base_cfg["cancer_types"] = [
            str(get_default_cancer_name(DATASET))
        ] + [f"unused_{i}" for i in range(1, n_cancer)]

    N_GENE_DIM = int(base_cfg["N_GENE_DIM"])
    N_TF_DIM = int(base_cfg["N_TF_DIM"])
    PEAK_HASH_SIZE = int(base_cfg["PEAK_HASH_SIZE"])
    D = int(base_cfg["D"])
    HID = int(base_cfg["HID"])
    DROP = float(base_cfg["DROP"])
    cancer_types = list(base_cfg["cancer_types"])

    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    default_cancer_name = get_default_cancer_name(DATASET)
    default_cancer_id = int(cancer2id.get(default_cancer_name, 0))
    cid = torch.tensor([default_cancer_id], device=DEVICE, dtype=torch.long)

    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    tfs_vocab = read_vocab_txt(GLOBAL_TF_VOCAB)

    model = PlanC_DirAware_Cancer_HashPeak(
        N_GENE_DIM,
        N_TF_DIM,
        len(cancer_types),
        PEAK_HASH_SIZE,
        d=D,
        hid=HID,
        drop=DROP,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    tf2targets, tf_in_prior = build_gene_priors(tfs_vocab, N_GENE_DIM)
    gene2peaks = build_gene_peak_priors(N_GENE_DIM)

    genes_in_peakprior = sorted(gene2peaks.keys())
    genes_in_tfprior = sorted({g for t in tf_in_prior for g in tf2targets[t]})
    genes_union = sorted(set(genes_in_tfprior) | set(genes_in_peakprior))

    samples, sid2sample = load_samples(DATASET, genes_vocab)

    random.seed(SEED)
    np.random.seed(SEED)

    all_cells = build_full_cell_lists(samples, MAX_CELLS_PER_SAMPLE)
    tr_full, te = split_cells_5050(all_cells)
    tr, va = split_train_val(tr_full, VAL_FRAC_FROM_TRAIN)

    log(f"[CELLS] train={len(tr)} val={len(va)} test={len(te)}")

    proj_store_cache = {}

    log("[ENC] encode train")
    Z_train = encode_Z(tr, sid2sample, model, default_cancer_id, proj_store_cache)

    log("[ENC] encode val")
    Z_val = encode_Z(va, sid2sample, model, default_cancer_id, proj_store_cache) if len(va) else None

    log("[ENC] encode test")
    Z_test = encode_Z(te, sid2sample, model, default_cancer_id, proj_store_cache)

    centroid_path = RUN_ROOT / DATASET / "shared_centroids.npy"

    if centroid_path.exists():
        C = np.load(centroid_path).astype(np.float32)
        log(f"[CLUSTER] load shared centroids: {centroid_path}")
    else:
        log(f"[CLUSTER][WARN] shared centroids not found, refitting: {centroid_path}")
        centroid_path.parent.mkdir(parents=True, exist_ok=True)
        C = spherical_kmeans_fit(
            Z_train,
            k=N_CLUSTERS,
            iters=CLUSTER_ITERS,
            max_train=CLUSTER_TRAIN_MAX,
        ).astype(np.float32)
        np.save(centroid_path, C)

    cl_train = spherical_kmeans_assign(Z_train, C)
    cl_val = spherical_kmeans_assign(Z_val, C) if Z_val is not None else None
    cl_test = spherical_kmeans_assign(Z_test, C)

    cl_all = (
        np.concatenate([cl_train, cl_val, cl_test], axis=0)
        if cl_val is not None
        else np.concatenate([cl_train, cl_test], axis=0)
    )

    group_all, cl2gid, cnt = build_group_ids_from_clusters(
        cl_train,
        cl_all,
        CLUSTER_NMIN_TRAIN,
    )

    ntr = len(tr)
    nva = len(va)

    train_group_ids = group_all[:ntr]
    val_group_ids = group_all[ntr: ntr + nva] if cl_val is not None else None
    test_group_ids = group_all[ntr + nva:] if cl_val is not None else group_all[ntr:]

    train_cancer_ids = np.full((len(tr),), default_cancer_id, np.int64)
    val_cancer_ids = np.full((len(va),), default_cancer_id, np.int64) if len(va) else None
    test_cancer_ids = np.full((len(te),), default_cancer_id, np.int64)

    train_cluster_for_corr = cl_train if CLUSTER_CORRECTION_USE_RAW_CLUSTER else train_group_ids
    val_cluster_for_corr = (
        cl_val
        if (CLUSTER_CORRECTION_USE_RAW_CLUSTER and cl_val is not None)
        else val_group_ids
    )
    test_cluster_for_corr = cl_test if CLUSTER_CORRECTION_USE_RAW_CLUSTER else test_group_ids

    log("[PROXY] val TF proxy")
    REG_val_tf = None
    if len(va):
        REG_val_tf = tfproxy_mix(
            regproxy_aucell_lite_from_top(
                va,
                sid2sample,
                tf2targets,
                tf_in_prior,
                AUCELL_TOPK,
                N_GENE_DIM,
            ),
            regproxy_mean_log1p(
                va,
                sid2sample,
                tf2targets,
                tf_in_prior,
            ),
        )

    log("[PROXY] test TF proxy")
    REG_test_tf = tfproxy_mix(
        regproxy_aucell_lite_from_top(
            te,
            sid2sample,
            tf2targets,
            tf_in_prior,
            AUCELL_TOPK,
            N_GENE_DIM,
        ),
        regproxy_mean_log1p(
            te,
            sid2sample,
            tf2targets,
            tf_in_prior,
        ),
    )

    train_cluster_summaries = build_cluster_summaries(
        cell_list=tr,
        sid2sample=sid2sample,
        Z=Z_train,
        cluster_ids=train_cluster_for_corr,
        genes_vocab=genes_vocab,
    )

    train_cluster_summary_vecs = {
        cid_: cluster_summary_to_vector(v)
        for cid_, v in train_cluster_summaries.items()
    }

    cluster_summary_table = ClusterSummaryTable(train_cluster_summary_vecs).to(DEVICE)

    n_group = int(max(1, train_group_ids.max() + 1))

    return {
        "base_cfg": base_cfg,
        "state_dict": state_dict,
        "N_GENE_DIM": N_GENE_DIM,
        "N_TF_DIM": N_TF_DIM,
        "PEAK_HASH_SIZE": PEAK_HASH_SIZE,
        "D": D,
        "HID": HID,
        "DROP": DROP,
        "cancer_types": cancer_types,
        "default_cancer_id": default_cancer_id,
        "cid": cid,
        "genes_vocab": genes_vocab,
        "tfs_vocab": tfs_vocab,
        "tf2targets": tf2targets,
        "tf_in_prior": tf_in_prior,
        "gene2peaks": gene2peaks,
        "genes_union": genes_union,
        "model": model,
        "samples": samples,
        "sid2sample": sid2sample,
        "tr": tr,
        "va": va,
        "te": te,
        "Z_train": Z_train,
        "Z_val": Z_val,
        "Z_test": Z_test,
        "train_group_ids": train_group_ids,
        "val_group_ids": val_group_ids,
        "test_group_ids": test_group_ids,
        "train_cancer_ids": train_cancer_ids,
        "val_cancer_ids": val_cancer_ids,
        "test_cancer_ids": test_cancer_ids,
        "train_cluster_for_corr": train_cluster_for_corr,
        "val_cluster_for_corr": val_cluster_for_corr,
        "test_cluster_for_corr": test_cluster_for_corr,
        "REG_val_tf": REG_val_tf,
        "REG_test_tf": REG_test_tf,
        "cluster_summary_table": cluster_summary_table,
        "n_group": n_group,
    }


# ============================================================
# 11. LOAD MODE CONTEXT
# ============================================================

def build_context_for_mode(mode: str, shared: dict):
    ds_out = RUN_ROOT / mode / DATASET
    ckpt_path = ds_out / "best_ckpt.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {mode}: {ckpt_path}")

    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ablation_cfg = load_ablation_cfg_from_ckpt(ckpt_obj, mode)

    D = shared["D"]
    PEAK_HASH_SIZE = shared["PEAK_HASH_SIZE"]
    cancer_types = shared["cancer_types"]
    tf_in_prior = shared["tf_in_prior"]
    genes_union = shared["genes_union"]
    n_group = shared["n_group"]

    adapter = HyperAdapterHashV2(
        d=D,
        k=K_RANK,
        n_tf=len(tf_in_prior),
        n_gene=len(genes_union),
        peak_hash_size=PEAK_HASH_SIZE,
        n_cancer=len(cancer_types),
        n_group=n_group,
        ablation_cfg=ablation_cfg,
    ).to(DEVICE)

    selected_peak_raw_ids = infer_selected_peak_ids_from_ckpt(ckpt_obj)

    peak_residual = None
    if ckpt_obj.get("peak_residual", None) is not None:
        peak_residual = SelectedPeakResidual(selected_peak_raw_ids, d=D).to(DEVICE)

    selector = None
    if ckpt_obj.get("selector", None) is not None:
        selector_state = ckpt_obj["selector"]
        summary_dim = int(selector_state["summary"].numel())
        selector = PriorSelectorV1(summary_dim=summary_dim).to(DEVICE)

    tf_corr_head = None
    rna_corr_head = None
    re_corr_head = None

    cluster_summary_table = shared["cluster_summary_table"]

    if ckpt_obj.get("tf_corr_head", None) is not None:
        tf_corr_head = ClusterCellCorrectionHead(
            D,
            cluster_summary_table.summary_dim,
            hid=CLUSTER_CORRECTION_HID,
        ).to(DEVICE)

    if ckpt_obj.get("rna_corr_head", None) is not None:
        rna_corr_head = ClusterCellCorrectionHead(
            D,
            cluster_summary_table.summary_dim,
            hid=CLUSTER_CORRECTION_HID,
        ).to(DEVICE)

    if ckpt_obj.get("re_corr_head", None) is not None:
        re_corr_head = ClusterCellCorrectionHead(
            D,
            cluster_summary_table.summary_dim,
            hid=CLUSTER_CORRECTION_HID,
        ).to(DEVICE)

    adapter.load_state_dict(ckpt_obj["adapter"], strict=True)

    if selector is not None:
        selector.load_state_dict(ckpt_obj["selector"], strict=True)

    if peak_residual is not None:
        peak_residual.load_state_dict(ckpt_obj["peak_residual"], strict=True)

    if tf_corr_head is not None:
        tf_corr_head.load_state_dict(ckpt_obj["tf_corr_head"], strict=True)

    if rna_corr_head is not None:
        rna_corr_head.load_state_dict(ckpt_obj["rna_corr_head"], strict=True)

    if re_corr_head is not None:
        re_corr_head.load_state_dict(ckpt_obj["re_corr_head"], strict=True)

    for m in [
        adapter,
        selector,
        peak_residual,
        tf_corr_head,
        rna_corr_head,
        re_corr_head,
    ]:
        if m is not None:
            m.eval()

    ctx = RunContext(
        cfg=ablation_cfg,
        dataset_name=DATASET,
        ds_out=ds_out,
        model=shared["model"],
        adapter=adapter,
        selector=selector,
        peak_residual=peak_residual,
        tf_corr_head=tf_corr_head,
        rna_corr_head=rna_corr_head,
        re_corr_head=re_corr_head,
        cluster_summary_table=cluster_summary_table,
        default_cancer_id=shared["default_cancer_id"],
        cid=shared["cid"],
        tf_in_prior=shared["tf_in_prior"],
        genes_union=shared["genes_union"],
        gene2peaks=shared["gene2peaks"],
        genes_vocab=shared["genes_vocab"],
    )

    return ctx, ckpt_path, ablation_cfg


# ============================================================
# 12. PLOTTING
# ============================================================

def plot_metric_bars(summary_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "test_fig2l_diag_score",
        "test_fig2l_offdiag_score",
        "test_fig2l_diag_offdiag_delta",
        "test_fig2l_diag_offdiag_ratio",

        "test_fig2m_overall_retained",
        "test_fig2m_cluster_coverage",
        "test_fig2m_mean_cluster_retention",
        "test_fig2m_distribution_preservation",

        "test_fig2m_strict_overall_retained",
        "test_fig2m_strict_cluster_coverage",
        "test_fig2m_strict_mean_cluster_retention",
        "test_fig2m_strict_distribution_preservation",

        "test_fig2m_mean_own_margin",
        "test_fig2m_mean_cell_profile_spearman",

        "test_fig2j_tf_trend_mean_spearman",
        "test_fig2j_tf_trend_highdev_mean_spearman",
        "test_fig2j_tf_trend_lowdev_mean_spearman",
    ]

    metrics = [m for m in metrics if m in summary_df.columns]

    for metric in metrics:
        df = summary_df.copy()
        df["ablation_mode"] = pd.Categorical(df["ablation_mode"], categories=MODES, ordered=True)
        df = df.sort_values("ablation_mode")

        plt.figure(figsize=(8.6, 3.8))
        plt.bar(df["ablation_mode"].astype(str), df[metric].astype(float))
        plt.ylabel(metric)
        plt.xlabel("")
        plt.xticks(rotation=35, ha="right")
        plt.title(metric)
        plt.tight_layout()

        for ext in ["png", "pdf", "svg"]:
            out = out_dir / f"{metric}.{ext}"
            plt.savefig(out, dpi=300 if ext == "png" else None)

        plt.close()
        log(f"[SAVE] {out_dir / (metric + '.png')}")


def plot_compact_summary(summary_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "test_fig2l_diag_offdiag_delta",
        "test_fig2m_strict_overall_retained",
        "test_fig2m_strict_distribution_preservation",
        "test_fig2j_tf_trend_mean_spearman",
        "test_fig2j_tf_trend_highdev_mean_spearman",
    ]

    metrics = [m for m in metrics if m in summary_df.columns]

    if not metrics:
        return

    df = summary_df.copy()
    df["ablation_mode"] = pd.Categorical(df["ablation_mode"], categories=MODES, ordered=True)
    df = df.sort_values("ablation_mode")

    x = np.arange(len(df))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(10.8, 4.5))

    for i, m in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        plt.bar(x + offset, df[m].astype(float), width=width, label=m)

    plt.xticks(x, df["ablation_mode"].astype(str), rotation=35, ha="right")
    plt.ylabel("score")
    plt.title("Stage-B component ablation: strict Fig.2L-M-like metrics")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    for ext in ["png", "pdf", "svg"]:
        out = out_dir / f"compact_fig2lm_strict_summary.{ext}"
        plt.savefig(out, dpi=300 if ext == "png" else None)

    plt.close()
    log(f"[SAVE] {out_dir / 'compact_fig2lm_strict_summary.png'}")


# ============================================================
# 13. MAIN
# ============================================================

def main():
    assert GLOBAL_GENE_VOCAB.exists(), GLOBAL_GENE_VOCAB
    assert GLOBAL_TF_VOCAB.exists(), GLOBAL_TF_VOCAB
    assert BASE_CKPT_MAP[DATASET].exists(), BASE_CKPT_MAP[DATASET]
    assert RUN_ROOT.exists(), RUN_ROOT

    shared = build_shared_data_cache()

    all_rows = []

    for mode in MODES:
        log("=" * 100)
        log(f"[EVAL] mode={mode}")
        log("=" * 100)

        mode_dir = RUN_ROOT / mode / DATASET
        ckpt_path = mode_dir / "best_ckpt.pt"

        if not ckpt_path.exists():
            log(f"[MISS] skip {mode}: {ckpt_path}")
            continue

        ctx, ckpt_path, ablation_cfg = build_context_for_mode(mode, shared)

        row = {
            "ablation_mode": mode,
            "dataset": DATASET,
            "ckpt_path": str(ckpt_path),
            "n_val": int(len(shared["va"])),
            "n_test": int(len(shared["te"])),
            "strict_retention_margin": float(STRICT_RETENTION_MARGIN),
            "use_lowrank_delta": int(ablation_cfg.use_lowrank_delta),
            "use_selector": int(ablation_cfg.use_selector),
            "use_cancer_adaptor": int(ablation_cfg.use_cancer_adaptor),
            "use_cellgroup_adaptor": int(ablation_cfg.use_cellgroup_adaptor),
            "use_group_scale_bias": int(ablation_cfg.use_group_scale_bias),
            "use_selected_peak_residual": int(ablation_cfg.use_selected_peak_residual),
            "use_cluster_correction": int(ablation_cfg.use_cluster_correction),
        }

        if len(shared["va"]) > 0:
            val_metrics = eval_fig2lm_like_metrics(
                Z_np=shared["Z_val"],
                group_ids=shared["val_group_ids"],
                cancer_ids=shared["val_cancer_ids"],
                cluster_ids=shared["val_cluster_for_corr"],
                REG_tf=shared["REG_val_tf"],
                ctx=ctx,
                split_name="val",
            )
            row.update(val_metrics)

        test_metrics = eval_fig2lm_like_metrics(
            Z_np=shared["Z_test"],
            group_ids=shared["test_group_ids"],
            cancer_ids=shared["test_cancer_ids"],
            cluster_ids=shared["test_cluster_for_corr"],
            REG_tf=shared["REG_test_tf"],
            ctx=ctx,
            split_name="test",
        )
        row.update(test_metrics)

        old_metrics_path = mode_dir / "final_metrics.csv"

        if old_metrics_path.exists():
            old_df = pd.read_csv(old_metrics_path)
            if len(old_df) > 0:
                for c in old_df.columns:
                    if c not in row:
                        row[f"old_{c}"] = old_df.iloc[0][c]

        out_one = mode_dir / "fig2lm_eval_standalone_strict.csv"
        pd.DataFrame([row]).to_csv(out_one, index=False)
        log(f"[SAVE] {out_one}")

        all_rows.append(row)
        torch.cuda.empty_cache()

    if not all_rows:
        raise RuntimeError("No modes evaluated. Please check checkpoint paths.")

    summary = pd.DataFrame(all_rows)

    out_summary = RUN_ROOT / "eval_fig2lm_standalone_strict_summary.csv"
    summary.to_csv(out_summary, index=False)
    log(f"[SAVE] {out_summary}")

    plot_dir = RUN_ROOT / "eval_fig2lm_standalone_strict_plots"
    plot_metric_bars(summary, plot_dir)
    plot_compact_summary(summary, plot_dir)

    log("=" * 100)
    log("[DONE] standalone strict Fig.2L-M-like evaluation")
    log(f"[SUMMARY] {out_summary}")
    log(f"[PLOTS] {plot_dir}")
    log("=" * 100)

    show_cols = [
        "ablation_mode",
        "test_fig2l_diag_score",
        "test_fig2l_diag_offdiag_delta",
        "test_fig2l_match_accuracy",

        "test_fig2m_overall_retained",
        "test_fig2m_distribution_preservation",

        "test_fig2m_strict_overall_retained",
        "test_fig2m_strict_distribution_preservation",

        "test_fig2m_mean_own_margin",
        "test_fig2m_mean_cell_profile_spearman",

        "test_fig2j_tf_trend_mean_spearman",
        "test_fig2j_tf_trend_highdev_mean_spearman",
    ]

    show_cols = [c for c in show_cols if c in summary.columns]
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
