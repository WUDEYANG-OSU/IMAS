from __future__ import annotations

# ============================================================
# Build matched-reference RE evaluation files from scratch
# ============================================================
# This script prepares:
#
#   cluster_profile_full_test_re_baseline.npz
#   cluster_profile_full_test_re_frontier.npz
#
# for the exact CRC_RNA_TO009 CCI 10-cluster scheme.
#
# Truth/reference:
#   observed projected ATAC accessibility for held-out paired cells,
#   binarized as accessible / inaccessible.
#
# Prediction:
#   RNA -> RE scores from the frozen foundation model and from the
#   target-adapted model, including the RE correction head and selected-
#   peak residual branch.
#
# Feature selection:
#   checkpoint-selected RE candidates are ranked only by accessibility
#   prevalence/variance in the TRAIN half; the TEST half is not used to
#   choose REs.
#
# Important:
#   - no inferred RE matrix is used as its own truth
#   - no UMAP is fitted
#   - CCI labels are loaded from the exact existing C0-C9 mapping
#   - adaptation still uses the internal model cluster IDs from clusters.csv
# ============================================================

import gc
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 0) PATHS / CONFIG
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

DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

RUN_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
)
DATASET_DIR = RUN_DIR / DATASET_NAME

BASE_CKPT_CANDIDATES = [
    get_checkpoint_root() / "qc_lodo_planC_v7_global_hashpeak" / "qc_lodo_planC_v7_global_hashpeak" / "CRC-Tissue" / "ckpt_last_BASE_global_hashpeak.pt",
    get_dataset_checkpoint("CRC-Tissue"),
]
BASE_CKPT = next(
    (path for path in BASE_CKPT_CANDIDATES if path.exists()),
    BASE_CKPT_CANDIDATES[0],
)
FRONTIER_CKPT = DATASET_DIR / "frontier_best.pt"

CLUSTERS_CSV = DATASET_DIR / "clusters.csv"
CCI_CELL_LEVEL_TABLE = (
    DATASET_DIR
    / f"single_sample_{SAMPLE_ID}_recluster_for_cci"
    / f"{SAMPLE_ID}_cell_level_mapping_with_umap_and_internal16.csv"
)

# Save beside the existing RNA/TF profile files so the plotting code can
# immediately discover task="re".
OUT_DIR = (
    DATASET_DIR
    / "structure_stability_full_eval_frontier_vs_baseline_rna_tf"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REQUIRE_CUDA = True

# Model settings copied from the target-adaptation implementation.
DEFAULT_CANCER_NAME = None
RNA_TOPK = 2048
GROUP_SCALE_ALPHA = 0.12
USE_GROUP_SCALE_BIAS = True
USE_CANCER_ADAPTOR = True
USE_CELLGROUP_ADAPTOR = True
USE_CLUSTER_CORRECTION = True
CLUSTER_CORRECTION_ALPHA_TF = 0.25
CLUSTER_CORRECTION_ALPHA_RNA = 0.25
CLUSTER_CORRECTION_ALPHA_RE = 0.25
CLUSTER_CORRECTION_HID = 256

# Defaults needed by reusable helper definitions.
COLOR_Q_LOW = 0.01
COLOR_Q_HIGH = 0.99

# RE evaluation settings.
RE_EVAL_N = 1024
RE_MIN_FEATURES = 128
RE_MIN_TRAIN_ACCESSIBLE_CELLS = 5
RE_MIN_TRAIN_PREVALENCE = 0.002
RE_MAX_TRAIN_PREVALENCE = 0.80
CELL_BATCH_SIZE = 128
PEAK_VECTOR_BATCH_SIZE = 2048
SAVE_CELL_LEVEL_MATRICES = False

# Exact held-out split used by the previous evaluation:
# md5("sid::barcode") parity, 0=train and 1=test.
SPLIT_NAME = "test"

# ATAC discovery is deliberately strict by default.
ALLOW_POSITIONAL_ATAC_FALLBACK = False
MIN_PAIRED_CELL_FRACTION = 0.70

CCI_EXPECTED_LABELS = [str(index) for index in range(10)]


def log(*args: Any) -> None:
    print(*args, flush=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_vocab_txt(path: Path) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                out.append(value)
    return out


def normalize_gene_name(value: str) -> str:
    return str(value).strip() if value is not None else ""


def norm_barcode(value: str) -> str:
    value = str(value)
    return value.split("::", 1)[-1] if "::" in value else value


def dataset_to_cancer_type(dataset_name: str) -> str:
    value = str(dataset_name)
    return value.split("-", 1)[0] if "-" in value else value


def get_default_cancer_name(dataset_name: str) -> str | None:
    if DEFAULT_CANCER_NAME is not None:
        return DEFAULT_CANCER_NAME
    return dataset_to_cancer_type(dataset_name)


def find_prior_file(prior_dir: Path, include_keys: list[str]) -> Path | None:
    candidates = sorted(prior_dir.glob("*.npz"))
    hits: list[Path] = []
    for path in candidates:
        name = path.name.lower()
        if all(key.lower() in name for key in include_keys):
            hits.append(path)
    return hits[0] if hits else None

class PeakHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, d: int):
        super().__init__()
        self.hash_size = int(hash_size)
        self.emb = nn.Embedding(self.hash_size, d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, peak_idx: torch.Tensor) -> torch.Tensor:
        hashed = torch.remainder(peak_idx, self.hash_size)
        return self.emb(hashed)


class PlanC_DirAware_Cancer_HashPeak(nn.Module):
    def __init__(
        self,
        n_gene: int,
        n_tf: int,
        n_cancer: int,
        peak_hash_size: int,
        d: int = 256,
        hid: int = 512,
        drop: float = 0.1,
    ):
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
    def _seg_ids_from_offsets(
        offsets: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        batch_size = offsets.shape[0]
        end = torch.empty(
            (batch_size,),
            device=offsets.device,
            dtype=torch.long,
        )
        end[:-1] = offsets[1:]
        end[-1] = total_tokens
        lengths = (end - offsets).clamp(min=0)
        return torch.repeat_interleave(
            torch.arange(batch_size, device=offsets.device),
            lengths,
        )

    @torch.no_grad()
    def encode_rna(
        self,
        idx: torch.Tensor,
        val: torch.Tensor,
        offsets: torch.Tensor,
        cancer_ids: torch.Tensor,
    ) -> torch.Tensor:
        seg_ids = self._seg_ids_from_offsets(offsets, idx.shape[0])
        cancer_token_ids = cancer_ids.index_select(0, seg_ids)

        x = (
            self.gene_in(idx)
            + self.gene_in_c(cancer_token_ids)
        ) * val.unsqueeze(-1)

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
    ):
        super().__init__()

        self.d = int(d)
        self.k = int(k)
        self.peak_hash_size = int(peak_hash_size)

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

        for embedding in [
            self.C_tf,
            self.C_gene,
            self.C_peak_hash,
            self.C_cancer,
        ]:
            nn.init.zeros_(embedding.weight)

        nn.init.zeros_(self.S_group.weight)
        nn.init.zeros_(self.B_group.weight)

    def forward(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.mlp(self.e)
        vectors = out.view(4, self.k, self.d)
        return vectors[0], vectors[1], vectors[2], vectors[3]

    def delta_tf(
        self,
        tf_pid: torch.Tensor,
        vector_tf: torch.Tensor,
    ) -> torch.Tensor:
        return self.C_tf(tf_pid) @ vector_tf

    def delta_gene(
        self,
        gene_pid: torch.Tensor,
        vector_gene: torch.Tensor,
    ) -> torch.Tensor:
        return self.C_gene(gene_pid) @ vector_gene

    def delta_peak_hash(
        self,
        peak_idx_raw: torch.Tensor,
        vector_peak: torch.Tensor,
    ) -> torch.Tensor:
        hashed = torch.remainder(peak_idx_raw, self.peak_hash_size)
        return self.C_peak_hash(hashed) @ vector_peak

    def delta_cancer(
        self,
        cancer_id: torch.Tensor,
        vector_cancer: torch.Tensor,
    ) -> torch.Tensor:
        return self.C_cancer(cancer_id) @ vector_cancer

    def apply_group_to_Z(
        self,
        z: torch.Tensor,
        group_id: torch.Tensor,
    ) -> torch.Tensor:
        if not USE_GROUP_SCALE_BIAS:
            return z + self.B_group(group_id)

        scale = self.S_group(group_id)
        bias = self.B_group(group_id)
        return (
            z
            * (1.0 + float(GROUP_SCALE_ALPHA) * torch.tanh(scale))
            + bias
        )


class SelectedPeakResidual(nn.Module):
    def __init__(
        self,
        selected_peak_raw_ids: np.ndarray,
        d: int,
    ):
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

        self.raw2local = {
            int(peak_id): index
            for index, peak_id in enumerate(ids.tolist())
        }

    @property
    def n_selected(self) -> int:
        return int(self.selected_peak_raw_ids.numel())

    def lookup(
        self,
        peak_idx_raw: torch.Tensor,
    ) -> torch.Tensor:
        if self.n_selected <= 0 or peak_idx_raw.numel() == 0:
            return torch.zeros(
                (peak_idx_raw.numel(), self.d),
                device=peak_idx_raw.device,
                dtype=torch.float32,
            )

        raw_np = (
            peak_idx_raw
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        local_np = np.full(
            (raw_np.size,),
            -1,
            dtype=np.int64,
        )

        for index, peak_id in enumerate(raw_np.tolist()):
            local_np[index] = self.raw2local.get(int(peak_id), -1)

        out = torch.zeros(
            (raw_np.size, self.d),
            device=peak_idx_raw.device,
            dtype=self.emb.weight.dtype,
        )

        matched = local_np >= 0
        if matched.any():
            position_tensor = torch.from_numpy(
                np.where(matched)[0]
            ).to(peak_idx_raw.device)
            local_tensor = torch.from_numpy(
                local_np[matched]
            ).to(peak_idx_raw.device)
            out[position_tensor] = self.emb(local_tensor)

        return out


class ClusterCellCorrectionHead(nn.Module):
    def __init__(
        self,
        z_dim: int,
        summary_dim: int,
        hid: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + summary_dim, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, z_dim),
        )

    def forward(
        self,
        z_cell: torch.Tensor,
        cluster_summary: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [z_cell, cluster_summary],
            dim=-1,
        )
        return self.mlp(x)


class ClusterSummaryTable(nn.Module):
    def __init__(
        self,
        cluster_summary_np: dict[int, np.ndarray],
    ):
        super().__init__()

        keys = sorted(cluster_summary_np.keys())
        if not keys:
            raise ValueError(
                "cluster_summary_vecs is empty in frontier checkpoint."
            )

        array = np.stack(
            [cluster_summary_np[key] for key in keys],
            axis=0,
        ).astype(np.float32)

        self.register_buffer(
            "summary_table",
            torch.from_numpy(array),
            persistent=True,
        )
        self.cid2row = {
            int(cluster_id): row
            for row, cluster_id in enumerate(keys)
        }

    @property
    def summary_dim(self) -> int:
        return int(self.summary_table.shape[1])

    def lookup(
        self,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        rows = [
            self.cid2row.get(int(cluster_id), 0)
            for cluster_id
            in cluster_ids.detach().cpu().numpy().tolist()
        ]
        row_tensor = torch.tensor(
            rows,
            device=cluster_ids.device,
            dtype=torch.long,
        )
        return self.summary_table[row_tensor]


# ============================================================
# 3) DATA LOADING
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
    depth_rna: np.ndarray | None = None


class CSRNPZStore:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.data = None
        self.indices = None
        self.indptr = None
        self.shape = None

    def open(self) -> None:
        if self.data is not None:
            return

        z = np.load(self.npz_path, mmap_mode="r")
        self.data = z["data"]
        self.indices = z["indices"]
        self.indptr = z["indptr"]
        shape = tuple(z["shape"].tolist())
        self.shape = (int(shape[0]), int(shape[1]))

    def get_row_topk(
        self,
        row_index: int,
        topk: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.open()

        start = int(self.indptr[row_index])
        end = int(self.indptr[row_index + 1])

        idx = self.indices[start:end]
        val = self.data[start:end]

        if idx.size == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
            )

        k = min(int(topk), idx.size)
        selected = np.argpartition(val, -k)[-k:]
        idx = idx[selected]
        val = val[selected]

        order = np.argsort(val)[::-1]
        return (
            idx[order].astype(np.int64),
            val[order].astype(np.float32),
        )


def get_manifest_rna_dir(dataset_name: str) -> Path:
    return (
        WORLD_ROOT
        / dataset_name
        / "processed"
        / "RNA"
    )


def resolve_counts_path(
    manifest_rna_dir: Path,
    sid: str,
    group: pd.DataFrame,
) -> Path:
    conventional = (
        manifest_rna_dir
        / f"{sid}.X_rna_counts_dense.npy"
    )
    if conventional.exists():
        return conventional

    if "counts_npy" in group.columns:
        raw = str(group["counts_npy"].iloc[0]).strip()
        candidate = Path(raw)

        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend(
                [
                    manifest_rna_dir / candidate,
                    manifest_rna_dir / candidate.name,
                ]
            )

        for path in candidates:
            if path.exists():
                return path

    raise FileNotFoundError(
        f"Cannot resolve RNA count matrix for sid={sid}."
    )


def load_samples(
    dataset_name: str,
    genes_vocab: list[str],
) -> tuple[list[Sample], dict[str, Sample]]:
    manifest_rna_dir = get_manifest_rna_dir(dataset_name)
    manifest_path = manifest_rna_dir / "manifest_rna.tsv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing RNA manifest: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        sep="\t",
    )

    required_columns = [
        "cell_id",
        "sample_id",
        "row_index",
    ]
    for column in required_columns:
        if column not in manifest.columns:
            raise KeyError(
                f"{manifest_path.name} is missing column: {column}"
            )

    gene2global = {
        gene: index
        for index, gene in enumerate(genes_vocab)
    }

    samples: list[Sample] = []
    sid2sample: dict[str, Sample] = {}

    for sid, group in manifest.groupby(
        "sample_id",
        sort=True,
    ):
        sid = str(sid).strip()
        group = (
            group
            .sort_values("row_index")
            .reset_index(drop=True)
        )

        meta_path = (
            manifest_rna_dir
            / f"{sid}.rna_meta.npz"
        )
        counts_path = resolve_counts_path(
            manifest_rna_dir,
            sid,
            group,
        )
        projected_path = (
            PROJECTED_ROOT
            / dataset_name
            / f"{sid}.rna.global.csr.npz"
        )

        for path in [
            meta_path,
            counts_path,
            projected_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required file: {path}"
                )

        metadata = np.load(
            meta_path,
            allow_pickle=True,
        )
        if "gene_names" not in metadata:
            raise KeyError(
                f"{meta_path.name} has no gene_names."
            )

        gene_names = np.asarray(
            metadata["gene_names"]
        ).astype(str)
        cells = np.asarray(
            [
                norm_barcode(value)
                for value
                in group["cell_id"].astype(str).tolist()
            ],
            dtype=object,
        )
        row_indices = (
            group["row_index"]
            .astype(np.int64)
            .to_numpy()
        )

        counts_memmap = np.load(
            counts_path,
            mmap_mode="r",
        )
        if counts_memmap.ndim != 2:
            raise ValueError(
                f"Count matrix must be 2D: {counts_path}"
            )
        if counts_memmap.shape[0] != len(row_indices):
            raise ValueError(
                f"Cell count mismatch for {sid}: "
                f"{counts_memmap.shape[0]} vs {len(row_indices)}"
            )
        if counts_memmap.shape[1] != len(gene_names):
            raise ValueError(
                f"Gene count mismatch for {sid}: "
                f"{counts_memmap.shape[1]} vs {len(gene_names)}"
            )

        ext2train = np.full(
            (len(gene_names),),
            -1,
            dtype=np.int64,
        )
        train2rna: dict[int, int] = {}

        for column_index, gene_name in enumerate(gene_names):
            gene_name = normalize_gene_name(gene_name)
            if gene_name in gene2global:
                global_id = gene2global[gene_name]
                ext2train[column_index] = global_id
                train2rna.setdefault(
                    global_id,
                    column_index,
                )

        matched = int((ext2train >= 0).sum())
        if matched <= 1000:
            raise ValueError(
                f"{dataset_name}/{sid}: "
                f"only {matched} genes matched global vocabulary."
            )

        depth_rna = None
        if "depth_rna" in metadata:
            depth_rna = np.asarray(
                metadata["depth_rna"],
                dtype=np.float32,
            )

        sample = Sample(
            sid=sid,
            cells=cells,
            row_indices=row_indices,
            proj_rna_path=projected_path,
            counts_path=counts_path,
            counts_shape=(
                int(counts_memmap.shape[0]),
                int(counts_memmap.shape[1]),
            ),
            gene_names=gene_names,
            ext2train=ext2train,
            train2rna=train2rna,
            depth_rna=depth_rna,
        )

        samples.append(sample)
        sid2sample[sid] = sample

        log(
            f"[LOAD] {dataset_name}/{sid}",
            f"| cells={sample.counts_shape[0]}",
            f"| genes={sample.counts_shape[1]}",
            f"| matched_genes={matched}",
        )

    return samples, sid2sample


def build_tokens_for_cells(
    cell_list: list[tuple[str, int]],
    sid2sample: dict[str, Sample],
    cancer_id_int: int,
    projected_store_cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    offsets = [0]

    for sid, rid in cell_list:
        sample = sid2sample[sid]
        key = (sid, sample.proj_rna_path)

        if key not in projected_store_cache:
            store = CSRNPZStore(sample.proj_rna_path)
            store.open()
            projected_store_cache[key] = store

        gene_idx, gene_val = (
            projected_store_cache[key]
            .get_row_topk(rid, RNA_TOPK)
        )

        # 避免极少数空行导致 encode_rna 无 token。
        if gene_idx.size == 0:
            gene_idx = np.asarray([0], dtype=np.int64)
            gene_val = np.asarray([0.0], dtype=np.float32)

        idx_list.append(gene_idx.astype(np.int64))
        val_list.append(gene_val.astype(np.float32))
        offsets.append(offsets[-1] + gene_idx.size)

    idx_cat = np.concatenate(idx_list)
    val_cat = np.concatenate(val_list)

    offsets_np = np.asarray(
        offsets[:-1],
        dtype=np.int64,
    )
    cancer_ids = np.full(
        (len(cell_list),),
        int(cancer_id_int),
        dtype=np.int64,
    )

    return (
        idx_cat,
        val_cat,
        offsets_np,
        cancer_ids,
    )


# ============================================================
# 4) STANDALONE EXPLAINER
# ============================================================

class FrontierExplainerStandalone:
    def __init__(self):
        required_paths = [
            BASE_CKPT,
            FRONTIER_CKPT,
            GLOBAL_GENE_VOCAB,
            GLOBAL_TF_VOCAB,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required path: {path}"
                )

        self.genes_vocab = read_vocab_txt(
            GLOBAL_GENE_VOCAB
        )
        self.tfs_vocab = read_vocab_txt(
            GLOBAL_TF_VOCAB
        )
        # The global peak vocabulary can contain tens of millions of entries.
        # RE evaluation uses raw global peak IDs directly and therefore does not
        # load the full vocabulary into memory.
        self.peaks_vocab = None

        self.base_ckpt = safe_torch_load(
            BASE_CKPT
        )
        self.frontier_ckpt = safe_torch_load(
            FRONTIER_CKPT
        )

        self.cfg = self._build_cfg()

        self.N_GENE_DIM = int(
            self.cfg["N_GENE_DIM"]
        )
        self.N_TF_DIM = int(
            self.cfg["N_TF_DIM"]
        )
        self.PEAK_HASH_SIZE = int(
            self.cfg["PEAK_HASH_SIZE"]
        )
        self.D = int(self.cfg["D"])
        self.HID = int(self.cfg["HID"])
        self.DROP = float(self.cfg["DROP"])

        self.cancer_types = list(
            self.cfg["cancer_types"]
        )
        self.cancer2id = {
            cancer: index
            for index, cancer
            in enumerate(self.cancer_types)
        }

        self.default_cancer_name = (
            get_default_cancer_name(DATASET_NAME)
        )
        self.default_cancer_id = int(
            self.cancer2id.get(
                self.default_cancer_name,
                0,
            )
        )
        self.cid = torch.tensor(
            [self.default_cancer_id],
            device=DEVICE,
            dtype=torch.long,
        )

        self.base_model = (
            PlanC_DirAware_Cancer_HashPeak(
                n_gene=self.N_GENE_DIM,
                n_tf=self.N_TF_DIM,
                n_cancer=len(self.cancer_types),
                peak_hash_size=self.PEAK_HASH_SIZE,
                d=self.D,
                hid=self.HID,
                drop=self.DROP,
            )
        )
        self.base_model.load_state_dict(
            self.base_ckpt["model"],
            strict=True,
        )
        self.base_model.to(DEVICE).eval()

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

        self._load_priors()
        self._load_data()
        self._load_frontier_modules()

    def _build_cfg(self) -> dict:
        config = self.base_ckpt.get("cfg", None)
        metadata = self.base_ckpt.get("meta", None)
        state_dict = self.base_ckpt.get("model", None)

        if state_dict is None:
            raise KeyError(
                "BASE_CKPT has no 'model' state_dict."
            )

        if config is None:
            config = {}
        else:
            config = dict(config)

        if isinstance(metadata, dict):
            for key in [
                "N_GENE_DIM",
                "N_TF_DIM",
                "PEAK_HASH_SIZE",
                "D",
                "HID",
                "DROP",
                "cancer_types",
            ]:
                if key in metadata:
                    config[key] = metadata[key]

        config.setdefault(
            "N_GENE_DIM",
            int(state_dict["gene_in.weight"].shape[0]),
        )
        config.setdefault(
            "N_TF_DIM",
            int(state_dict["tf_in.weight"].shape[0]),
        )
        config.setdefault(
            "PEAK_HASH_SIZE",
            int(state_dict["peak_in.emb.weight"].shape[0]),
        )
        config.setdefault(
            "D",
            int(state_dict["gene_in.weight"].shape[1]),
        )
        config.setdefault(
            "HID",
            int(state_dict["rna_proj.0.weight"].shape[0]),
        )
        config.setdefault("DROP", 0.1)

        if "cancer_types" not in config:
            n_cancer = int(
                state_dict["cell_c.weight"].shape[0]
            )
            first_name = get_default_cancer_name(
                DATASET_NAME
            )
            config["cancer_types"] = [
                str(first_name)
            ] + [
                f"unused_{index}"
                for index in range(1, n_cancer)
            ]

        return config

    def _load_priors(self) -> None:
        tf_gene_prior = (
            PRIORS_DIR
            / "tf_gene_f.cap256.npz"
        )
        if not tf_gene_prior.exists():
            tf_gene_prior = (
                PRIORS_DIR
                / "tf_gene_f.full.npz"
            )
        if not tf_gene_prior.exists():
            raise FileNotFoundError(
                "Cannot find TF-gene prior."
            )

        tf_gene_npz = np.load(
            tf_gene_prior,
            allow_pickle=True,
        )
        tf_source = tf_gene_npz["src"].astype(
            np.int64
        )
        gene_destination = tf_gene_npz[
            "dst"
        ].astype(np.int64)

        self.tf2targets: defaultdict[int, list[int]] = (
            defaultdict(list)
        )

        for tf_id, gene_id in zip(
            tf_source.tolist(),
            gene_destination.tolist(),
        ):
            if (
                0 <= tf_id < len(self.tfs_vocab)
                and 0 <= gene_id < self.N_GENE_DIM
            ):
                self.tf2targets[int(tf_id)].append(
                    int(gene_id)
                )

        self.tf_in_prior = sorted(
            self.tf2targets.keys()
        )
        self.tf_global2local = {
            int(tf_id): local_id
            for local_id, tf_id
            in enumerate(self.tf_in_prior)
        }

        gene_peak_prior = find_prior_file(
            PRIORS_DIR,
            ["gene", "peak"],
        )
        if gene_peak_prior is None:
            raise FileNotFoundError(
                f"Cannot find gene-peak prior in {PRIORS_DIR}"
            )

        gene_peak_npz = np.load(
            gene_peak_prior,
            allow_pickle=True,
        )
        source = gene_peak_npz["src"].astype(
            np.int64
        )
        destination = gene_peak_npz["dst"].astype(
            np.int64
        )

        source_max = (
            int(source.max())
            if source.size
            else -1
        )

        self.gene2peaks: defaultdict[int, list[int]] = (
            defaultdict(list)
        )

        if source_max < self.N_GENE_DIM:
            # src = gene, dst = peak
            for gene_id, peak_id in zip(
                source.tolist(),
                destination.tolist(),
            ):
                if 0 <= gene_id < self.N_GENE_DIM:
                    self.gene2peaks[
                        int(gene_id)
                    ].append(int(peak_id))
        else:
            # src = peak, dst = gene
            for peak_id, gene_id in zip(
                source.tolist(),
                destination.tolist(),
            ):
                if 0 <= gene_id < self.N_GENE_DIM:
                    self.gene2peaks[
                        int(gene_id)
                    ].append(int(peak_id))

        genes_in_peak_prior = sorted(
            self.gene2peaks.keys()
        )
        genes_in_tf_prior = sorted(
            {
                gene_id
                for tf_id in self.tf_in_prior
                for gene_id
                in self.tf2targets[tf_id]
            }
        )

        self.genes_union = sorted(
            set(genes_in_tf_prior)
            | set(genes_in_peak_prior)
        )
        self.gid2pid_u = {
            int(gene_id): local_id
            for local_id, gene_id
            in enumerate(self.genes_union)
        }
        self.pid_u2gid = {
            local_id: gene_id
            for local_id, gene_id
            in enumerate(self.genes_union)
        }

        log(
            "[PRIOR]",
            f"TF={len(self.tf_in_prior)}",
            f"gene_union={len(self.genes_union)}",
            f"gene_with_peaks={len(self.gene2peaks)}",
        )

    def _load_data(self) -> None:
        (
            self.samples,
            self.sid2sample,
        ) = load_samples(
            DATASET_NAME,
            self.genes_vocab,
        )

        self.projected_store_cache: dict = {}

        cluster_path = DATASET_DIR / "clusters.csv"
        if not cluster_path.exists():
            raise FileNotFoundError(
                f"Missing cluster metadata: {cluster_path}"
            )

        self.cluster_df = pd.read_csv(cluster_path)

        required_columns = [
            "sid",
            "rid",
            "cluster_id",
            "cluster_label",
            "group_id",
        ]
        for column in required_columns:
            if column not in self.cluster_df.columns:
                raise KeyError(
                    f"clusters.csv is missing: {column}"
                )

        if "split" not in self.cluster_df.columns:
            self.cluster_df["split"] = "unknown"

        self.sidrid2meta: dict[tuple[str, int], dict] = {}

        for _, row in self.cluster_df.iterrows():
            key = (
                str(row["sid"]),
                int(row["rid"]),
            )
            self.sidrid2meta[key] = {
                "split": str(row["split"]),
                "sid": str(row["sid"]),
                "rid": int(row["rid"]),
                "cluster_id": int(row["cluster_id"]),
                "cluster_label": str(row["cluster_label"]),
                "group_id": int(row["group_id"]),
            }

    def _load_frontier_modules(self) -> None:
        if "hyper" not in self.frontier_ckpt:
            raise KeyError(
                "FRONTIER_CKPT has no 'hyper'."
            )

        hyper = self.frontier_ckpt["hyper"]
        n_group = int(
            self.cluster_df["group_id"].max()
        ) + 1

        self.adapter_after = HyperAdapterHashV2(
            d=self.D,
            k=int(hyper["K_RANK"]),
            n_tf=len(self.tf_in_prior),
            n_gene=len(self.genes_union),
            peak_hash_size=self.PEAK_HASH_SIZE,
            n_cancer=len(self.cancer_types),
            n_group=n_group,
            e_dim=64,
            hid=256,
        ).to(DEVICE)

        self.adapter_after.load_state_dict(
            self.frontier_ckpt[
                "adapter_state_dict"
            ],
            strict=True,
        )
        self.adapter_after.eval()

        selected_peak_raw_ids = np.asarray(
            self.frontier_ckpt.get(
                "selected_peak_raw_ids",
                [],
            ),
            dtype=np.int64,
        )

        self.peak_res_after = SelectedPeakResidual(
            selected_peak_raw_ids,
            d=self.D,
        ).to(DEVICE)

        peak_residual_state = self.frontier_ckpt.get(
            "peak_residual_state_dict",
            None,
        )
        if peak_residual_state is not None:
            self.peak_res_after.load_state_dict(
                peak_residual_state,
                strict=True,
            )
        self.peak_res_after.eval()

        cluster_summary_vecs = self.frontier_ckpt.get(
            "cluster_summary_vecs",
            None,
        )
        if cluster_summary_vecs is None:
            raise KeyError(
                "FRONTIER_CKPT has no cluster_summary_vecs."
            )

        cluster_summary_vecs = {
            int(key): np.asarray(
                value,
                dtype=np.float32,
            )
            for key, value
            in cluster_summary_vecs.items()
        }

        self.cluster_summary_table = (
            ClusterSummaryTable(
                cluster_summary_vecs
            ).to(DEVICE)
        )
        summary_dim = (
            self.cluster_summary_table.summary_dim
        )

        self.tf_corr_after = (
            ClusterCellCorrectionHead(
                self.D,
                summary_dim,
                hid=CLUSTER_CORRECTION_HID,
            ).to(DEVICE)
        )
        self.rna_corr_after = (
            ClusterCellCorrectionHead(
                self.D,
                summary_dim,
                hid=CLUSTER_CORRECTION_HID,
            ).to(DEVICE)
        )
        self.re_corr_after = (
            ClusterCellCorrectionHead(
                self.D,
                summary_dim,
                hid=CLUSTER_CORRECTION_HID,
            ).to(DEVICE)
        )

        state_mapping = [
            (
                "tf_corr_head_state_dict",
                self.tf_corr_after,
            ),
            (
                "rna_corr_head_state_dict",
                self.rna_corr_after,
            ),
            (
                "re_corr_head_state_dict",
                self.re_corr_after,
            ),
        ]

        for state_key, module in state_mapping:
            state = self.frontier_ckpt.get(
                state_key,
                None,
            )
            if state is not None:
                module.load_state_dict(
                    state,
                    strict=True,
                )
            module.eval()

    def get_cell_meta(
        self,
        sid: str,
        rid: int,
    ) -> dict:
        key = (str(sid), int(rid))
        if key not in self.sidrid2meta:
            raise KeyError(
                f"No cluster metadata for cell: {key}"
            )
        return self.sidrid2meta[key]

    def get_cells_meta_arrays(
        self,
        cell_list: list[tuple[str, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        group_ids: list[int] = []
        cluster_ids: list[int] = []

        for sid, rid in cell_list:
            metadata = self.get_cell_meta(
                sid,
                rid,
            )
            group_ids.append(
                int(metadata["group_id"])
            )
            cluster_ids.append(
                int(metadata["cluster_id"])
            )

        return (
            np.asarray(group_ids, dtype=np.int64),
            np.asarray(cluster_ids, dtype=np.int64),
        )

    @torch.no_grad()
    def encode_cells_base(
        self,
        cell_list: list[tuple[str, int]],
    ) -> torch.Tensor:
        (
            idx_cat,
            val_cat,
            offsets,
            cancer_ids,
        ) = build_tokens_for_cells(
            cell_list,
            self.sid2sample,
            self.default_cancer_id,
            self.projected_store_cache,
        )

        return self.base_model.encode_rna(
            torch.from_numpy(idx_cat).to(
                DEVICE
            ),
            torch.from_numpy(val_cat).to(
                DEVICE
            ),
            torch.from_numpy(offsets).to(
                DEVICE
            ),
            torch.from_numpy(cancer_ids).to(
                DEVICE
            ),
        )

    def cluster_ids_to_summary(
        self,
        cluster_ids_np: np.ndarray,
    ) -> torch.Tensor:
        cluster_tensor = torch.from_numpy(
            cluster_ids_np.astype(np.int64)
        ).to(DEVICE)
        return self.cluster_summary_table.lookup(
            cluster_tensor
        )

    def apply_cluster_correction(
        self,
        z_in: torch.Tensor,
        cluster_ids_np: np.ndarray,
        head: nn.Module | None,
        alpha: float,
    ) -> torch.Tensor:
        if head is None:
            return z_in

        summary = self.cluster_ids_to_summary(
            cluster_ids_np
        )
        delta = head(
            z_in,
            summary,
        )
        return F.normalize(
            z_in + float(alpha) * delta,
            dim=-1,
        )

    @torch.no_grad()
    def adjust_Z_batch(
        self,
        z0: torch.Tensor,
        group_ids_np: np.ndarray,
        cluster_ids_np: np.ndarray,
        mode: str,
        task: str,
    ) -> torch.Tensor:
        if mode == "pretrain":
            return F.normalize(z0, dim=-1)

        if mode != "after":
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        adapter = self.adapter_after
        _, _, _, vector_cancer = adapter()

        z = z0

        if USE_CELLGROUP_ADAPTOR:
            group_tensor = torch.from_numpy(
                group_ids_np.astype(np.int64)
            ).to(DEVICE)
            z = adapter.apply_group_to_Z(
                z,
                group_tensor,
            )

        if USE_CANCER_ADAPTOR:
            cancer_tensor = torch.full(
                (z.shape[0],),
                int(self.default_cancer_id),
                device=DEVICE,
                dtype=torch.long,
            )
            z = (
                z
                + adapter.delta_cancer(
                    cancer_tensor,
                    vector_cancer,
                )
            )

        z = F.normalize(z, dim=-1)

        if not USE_CLUSTER_CORRECTION:
            return z

        if task == "tf":
            return self.apply_cluster_correction(
                z,
                cluster_ids_np,
                self.tf_corr_after,
                CLUSTER_CORRECTION_ALPHA_TF,
            )
        if task == "rna":
            return self.apply_cluster_correction(
                z,
                cluster_ids_np,
                self.rna_corr_after,
                CLUSTER_CORRECTION_ALPHA_RNA,
            )
        if task == "re":
            return self.apply_cluster_correction(
                z,
                cluster_ids_np,
                self.re_corr_after,
                CLUSTER_CORRECTION_ALPHA_RE,
            )

        raise ValueError(
            f"Unknown task: {task}"
        )

    def tf_base(
        self,
        tf_local_pid_t: torch.Tensor,
    ) -> torch.Tensor:
        tf_global_ids = torch.tensor(
            [
                self.tf_in_prior[index]
                for index
                in tf_local_pid_t.detach().cpu().numpy().tolist()
            ],
            device=DEVICE,
            dtype=torch.long,
        )

        return (
            self.base_model.tf_in(tf_global_ids)
            + self.base_model.tf_in_c(
                self.cid
            ).expand(tf_global_ids.numel(), -1)
        )

    def gene_base_from_pid(
        self,
        gene_local_pid_t: torch.Tensor,
    ) -> torch.Tensor:
        gene_global_ids = torch.tensor(
            [
                self.genes_union[index]
                for index
                in gene_local_pid_t.detach().cpu().numpy().tolist()
            ],
            device=DEVICE,
            dtype=torch.long,
        )

        return (
            self.base_model.gene_out(
                gene_global_ids
            )
            + self.base_model.gene_out_c(
                self.cid
            ).expand(gene_global_ids.numel(), -1)
        )

    def peak_base_from_raw(
        self,
        peak_raw_t: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.base_model.peak_out(
                peak_raw_t
            )
            + self.base_model.peak_out_c(
                self.cid
            ).expand(peak_raw_t.numel(), -1)
        )

    @torch.no_grad()
    def score_tf_vec(
        self,
        tf_local_pid_t: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        tf_vectors = self.tf_base(
            tf_local_pid_t
        )

        if mode == "pretrain":
            return tf_vectors
        if mode != "after":
            raise ValueError(
                f"Unsupported TF mode: {mode}"
            )

        (
            vector_tf,
            _,
            _,
            vector_cancer,
        ) = self.adapter_after()

        tf_vectors = (
            tf_vectors
            + self.adapter_after.delta_tf(
                tf_local_pid_t,
                vector_tf,
            )
        )

        cancer_tensor = torch.full(
            (tf_local_pid_t.numel(),),
            int(self.default_cancer_id),
            device=DEVICE,
            dtype=torch.long,
        )

        return (
            tf_vectors
            + self.adapter_after.delta_cancer(
                cancer_tensor,
                vector_cancer,
            )
        )

    @torch.no_grad()
    def score_gene_vec(
        self,
        gene_local_pid_t: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        gene_vectors = self.gene_base_from_pid(
            gene_local_pid_t
        )

        if mode == "pretrain":
            return gene_vectors
        if mode != "after":
            raise ValueError(
                f"Unsupported gene mode: {mode}"
            )

        (
            _,
            vector_gene,
            _,
            vector_cancer,
        ) = self.adapter_after()

        gene_vectors = (
            gene_vectors
            + self.adapter_after.delta_gene(
                gene_local_pid_t,
                vector_gene,
            )
        )

        cancer_tensor = torch.full(
            (gene_local_pid_t.numel(),),
            int(self.default_cancer_id),
            device=DEVICE,
            dtype=torch.long,
        )

        return (
            gene_vectors
            + self.adapter_after.delta_cancer(
                cancer_tensor,
                vector_cancer,
            )
        )

    @torch.no_grad()
    def score_peak_vec(
        self,
        peak_raw_np: np.ndarray,
        mode: str,
    ) -> torch.Tensor:
        peak_raw_np = np.asarray(
            peak_raw_np,
            dtype=np.int64,
        )
        peak_raw_t = torch.from_numpy(
            peak_raw_np
        ).to(DEVICE)

        peak_vectors = self.peak_base_from_raw(
            peak_raw_t
        )

        if mode == "pretrain":
            return peak_vectors
        if mode != "after":
            raise ValueError(
                f"Unsupported peak mode: {mode}"
            )

        (
            _,
            _,
            vector_peak,
            vector_cancer,
        ) = self.adapter_after()

        peak_vectors = (
            peak_vectors
            + self.adapter_after.delta_peak_hash(
                peak_raw_t,
                vector_peak,
            )
        )

        if (
            self.peak_res_after is not None
            and self.peak_res_after.n_selected > 0
        ):
            peak_vectors = (
                peak_vectors
                + self.peak_res_after.lookup(
                    peak_raw_t
                )
            )

        cancer_tensor = torch.full(
            (peak_raw_t.numel(),),
            int(self.default_cancer_id),
            device=DEVICE,
            dtype=torch.long,
        )

        return (
            peak_vectors
            + self.adapter_after.delta_cancer(
                cancer_tensor,
                vector_cancer,
            )
        )


# ============================================================
# 5) RE-SPECIFIC DATA DISCOVERY
# ============================================================

def make_cell_key(sid: str, barcode: str) -> str:
    return f"{sid}::{norm_barcode(barcode)}"


def split_5050(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 1


def normalize_pair_sample_id(value: str) -> str:
    value = str(value).strip()
    replacements = [
        ("_RNA_", "_MOD_"),
        ("_ATAC_", "_MOD_"),
        ("-RNA-", "-MOD-"),
        ("-ATAC-", "-MOD-"),
        (".RNA.", ".MOD."),
        (".ATAC.", ".MOD."),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    if value.endswith("_RNA"):
        value = value[:-4] + "_MOD"
    if value.endswith("_ATAC"):
        value = value[:-5] + "_MOD"
    return value


def sample_id_candidates(rna_sid: str) -> list[str]:
    candidates = [
        str(rna_sid),
        str(rna_sid).replace("_RNA_", "_ATAC_"),
        str(rna_sid).replace("-RNA-", "-ATAC-"),
        str(rna_sid).replace(".RNA.", ".ATAC."),
    ]
    if str(rna_sid).endswith("_RNA"):
        candidates.append(str(rna_sid)[:-4] + "_ATAC")
    out: list[str] = []
    for value in candidates:
        if value not in out:
            out.append(value)
    return out


class SparseCSRStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = None
        self.indices = None
        self.indptr = None
        self.shape = None

    def open(self) -> None:
        if self.data is not None:
            return
        z = np.load(self.path, mmap_mode="r")
        required = ["data", "indices", "indptr", "shape"]
        missing = [key for key in required if key not in z]
        if missing:
            raise KeyError(f"{self.path.name}: missing CSR keys {missing}")
        self.data = z["data"]
        self.indices = z["indices"]
        self.indptr = z["indptr"]
        shape = tuple(np.asarray(z["shape"]).tolist())
        self.shape = (int(shape[0]), int(shape[1]))

    def row_indices_values(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        self.open()
        if row < 0 or row >= self.shape[0]:
            raise IndexError(f"ATAC row {row} outside matrix with {self.shape[0]} rows")
        start = int(self.indptr[row])
        end = int(self.indptr[row + 1])
        return (
            np.asarray(self.indices[start:end], dtype=np.int64),
            np.asarray(self.data[start:end], dtype=np.float32),
        )


@dataclass(frozen=True)
class ATACReference:
    projected_path: Path
    manifest_path: Optional[Path]
    atac_sample_id: str
    store: SparseCSRStore
    barcode_to_row: dict[str, int]


def discover_atac_manifests() -> list[Path]:
    processed_root = WORLD_ROOT / DATASET_NAME / "processed"
    candidates: list[Path] = []
    if processed_root.exists():
        patterns = [
            "**/manifest_atac.tsv",
            "**/manifest_ATAC.tsv",
            "**/*atac*manifest*.tsv",
            "**/*ATAC*manifest*.tsv",
        ]
        for pattern in patterns:
            candidates.extend(processed_root.glob(pattern))
    # De-duplicate while preserving deterministic order.
    return sorted(set(Path(path) for path in candidates))


def choose_column(columns: list[str], aliases: list[str]) -> Optional[str]:
    lower_to_original = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def resolve_atac_manifest_rows(rna_sid: str) -> tuple[Optional[Path], Optional[pd.DataFrame], Optional[str]]:
    normalized_target = normalize_pair_sample_id(rna_sid)

    for manifest_path in discover_atac_manifests():
        try:
            dataframe = pd.read_csv(manifest_path, sep="\t")
        except Exception:
            continue

        sample_column = choose_column(
            dataframe.columns.tolist(),
            ["sample_id", "sid", "sample", "library_id"],
        )
        cell_column = choose_column(
            dataframe.columns.tolist(),
            ["cell_id", "barcode", "barcodes", "cell", "cell_barcode"],
        )
        row_column = choose_column(
            dataframe.columns.tolist(),
            ["row_index", "rid", "row", "matrix_row"],
        )
        if sample_column is None or cell_column is None or row_column is None:
            continue

        sample_values = dataframe[sample_column].astype(str)
        exact_candidates = sample_id_candidates(rna_sid)

        selected = dataframe[sample_values.isin(exact_candidates)].copy()
        selected_sid = None
        if selected.empty:
            normalized = sample_values.map(normalize_pair_sample_id)
            selected = dataframe[normalized == normalized_target].copy()
        if selected.empty:
            continue

        unique_sample_ids = selected[sample_column].astype(str).unique().tolist()
        if len(unique_sample_ids) > 1:
            # Prefer the explicit ATAC-looking sample ID.
            atac_like = [value for value in unique_sample_ids if "ATAC" in value.upper()]
            selected_sid = atac_like[0] if atac_like else unique_sample_ids[0]
            selected = selected[selected[sample_column].astype(str) == selected_sid].copy()
        else:
            selected_sid = unique_sample_ids[0]

        selected = selected.sort_values(row_column).reset_index(drop=True)
        selected["_barcode_norm"] = selected[cell_column].astype(str).map(norm_barcode)
        selected["_atac_row"] = pd.to_numeric(
            selected[row_column],
            errors="raise",
        ).astype(np.int64)

        if selected["_barcode_norm"].duplicated().any():
            raise ValueError(
                f"Duplicate ATAC barcodes in {manifest_path} for sample {selected_sid}"
            )
        if selected["_atac_row"].duplicated().any():
            raise ValueError(
                f"Duplicate ATAC rows in {manifest_path} for sample {selected_sid}"
            )

        return manifest_path, selected, str(selected_sid)

    return None, None, None


def discover_projected_atac_path(
    rna_sid: str,
    manifest_rows: Optional[pd.DataFrame],
    atac_sid: Optional[str],
) -> Path:
    projected_dir = PROJECTED_ROOT / DATASET_NAME
    if not projected_dir.exists():
        raise FileNotFoundError(f"Projected-input directory does not exist: {projected_dir}")

    sid_values: list[str] = []
    for value in [atac_sid, *sample_id_candidates(rna_sid)]:
        if value is not None and str(value) not in sid_values:
            sid_values.append(str(value))

    candidates: list[Path] = []
    for sid in sid_values:
        candidates.extend(
            [
                projected_dir / f"{sid}.atac.global.csr.npz",
                projected_dir / f"{sid}.ATAC.global.csr.npz",
                projected_dir / f"{sid}.peak.global.csr.npz",
                projected_dir / f"{sid}.peaks.global.csr.npz",
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    all_candidates = sorted(
        set(
            list(projected_dir.glob("*.atac.global.csr.npz"))
            + list(projected_dir.glob("*.ATAC.global.csr.npz"))
            + list(projected_dir.glob("*.peak.global.csr.npz"))
            + list(projected_dir.glob("*.peaks.global.csr.npz"))
        )
    )

    normalized_target = normalize_pair_sample_id(rna_sid)
    normalized_hits = [
        path
        for path in all_candidates
        if normalize_pair_sample_id(path.name.split(".atac.global", 1)[0])
        == normalized_target
        or normalize_pair_sample_id(path.stem)
        == normalized_target
    ]
    if len(normalized_hits) == 1:
        return normalized_hits[0]

    if len(all_candidates) == 1:
        return all_candidates[0]

    raise FileNotFoundError(
        "Could not resolve the projected ATAC CSR matrix.\n"
        f"RNA sample: {rna_sid}\n"
        f"Tried explicit paths: {[str(path) for path in candidates]}\n"
        f"Available projected ATAC-like files: {[str(path) for path in all_candidates[:30]]}"
    )


def load_sidecar_atac_barcodes(projected_path: Path, expected_rows: int) -> Optional[np.ndarray]:
    parent_candidates = [
        projected_path.parent,
        WORLD_ROOT / DATASET_NAME / "processed" / "ATAC",
        WORLD_ROOT / DATASET_NAME / "processed" / "ATAC_peak",
        WORLD_ROOT / DATASET_NAME / "processed" / "ATAC_peaks",
    ]
    stem_prefixes = [
        projected_path.name.split(".atac.global.csr.npz")[0],
        projected_path.name.split(".ATAC.global.csr.npz")[0],
        projected_path.name.split(".peak.global.csr.npz")[0],
        projected_path.name.split(".peaks.global.csr.npz")[0],
    ]
    metadata_paths: list[Path] = []
    for parent in parent_candidates:
        for stem in stem_prefixes:
            metadata_paths.extend(
                [
                    parent / f"{stem}.atac_meta.npz",
                    parent / f"{stem}.ATAC_meta.npz",
                    parent / f"{stem}.peak_meta.npz",
                    parent / f"{stem}.meta.npz",
                ]
            )

    for path in metadata_paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as z:
            for key in ["cells", "barcodes", "cell_ids", "cell_id"]:
                if key in z:
                    barcodes = np.asarray(
                        [norm_barcode(value) for value in z[key].astype(object).tolist()],
                        dtype=object,
                    )
                    if barcodes.size == expected_rows:
                        log("[ATAC] barcode sidecar:", path)
                        return barcodes
    return None


def build_atac_reference(explainer: FrontierExplainerStandalone, rna_sid: str) -> ATACReference:
    manifest_path, manifest_rows, atac_sid = resolve_atac_manifest_rows(rna_sid)
    projected_path = discover_projected_atac_path(
        rna_sid,
        manifest_rows,
        atac_sid,
    )
    store = SparseCSRStore(projected_path)
    store.open()

    barcode_to_row: dict[str, int] = {}

    if manifest_rows is not None:
        max_row = int(manifest_rows["_atac_row"].max())
        if max_row >= store.shape[0]:
            raise ValueError(
                f"ATAC manifest row max {max_row} exceeds matrix rows {store.shape[0]}"
            )
        barcode_to_row = {
            str(row["_barcode_norm"]): int(row["_atac_row"])
            for _, row in manifest_rows.iterrows()
        }
    else:
        sidecar_barcodes = load_sidecar_atac_barcodes(
            projected_path,
            store.shape[0],
        )
        if sidecar_barcodes is not None:
            if len(set(sidecar_barcodes.tolist())) != sidecar_barcodes.size:
                raise ValueError("Duplicate barcodes in ATAC sidecar metadata")
            barcode_to_row = {
                str(barcode): int(index)
                for index, barcode in enumerate(sidecar_barcodes.tolist())
            }
        elif ALLOW_POSITIONAL_ATAC_FALLBACK:
            sample = explainer.sid2sample[rna_sid]
            if store.shape[0] != sample.counts_shape[0]:
                raise RuntimeError(
                    "Positional fallback requested, but RNA and ATAC row counts differ."
                )
            barcode_to_row = {
                str(norm_barcode(barcode)): int(index)
                for index, barcode in enumerate(sample.cells.tolist())
            }
            log("[ATAC][WARNING] using positional RNA/ATAC row matching")
        else:
            raise FileNotFoundError(
                "Projected ATAC matrix was found, but no trustworthy barcode metadata "
                "could be found. Positional matching is disabled."
            )

    log("[ATAC] projected matrix:", projected_path)
    log("[ATAC] shape:", store.shape)
    log("[ATAC] manifest:", manifest_path)
    log("[ATAC] barcode rows:", len(barcode_to_row))

    return ATACReference(
        projected_path=projected_path,
        manifest_path=manifest_path,
        atac_sample_id=str(atac_sid or rna_sid),
        store=store,
        barcode_to_row=barcode_to_row,
    )


# ============================================================
# 6) EXACT CCI10 AND MODEL CELL MATCHING
# ============================================================

def canonicalize_cci_label(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            text = str(int(number))
    except Exception:
        pass
    return text


def load_exact_cci_mapping(explainer: FrontierExplainerStandalone) -> pd.DataFrame:
    if not CCI_CELL_LEVEL_TABLE.exists():
        raise FileNotFoundError(
            f"Missing exact CCI cell mapping: {CCI_CELL_LEVEL_TABLE}"
        )

    mapping = pd.read_csv(CCI_CELL_LEVEL_TABLE)
    required = ["sid", "rid", "barcode", "new_leiden"]
    missing = [column for column in required if column not in mapping.columns]
    if missing:
        raise KeyError(
            f"{CCI_CELL_LEVEL_TABLE.name}: missing columns {missing}"
        )

    mapping = mapping[
        mapping["sid"].astype(str) == str(SAMPLE_ID)
    ].copy()
    mapping["rid"] = pd.to_numeric(
        mapping["rid"],
        errors="raise",
    ).astype(np.int64)
    mapping["barcode"] = mapping["barcode"].astype(str).map(norm_barcode)
    mapping["cci_cluster"] = mapping["new_leiden"].map(canonicalize_cci_label)

    found_labels = sorted(
        mapping["cci_cluster"].astype(str).unique().tolist(),
        key=lambda value: int(value),
    )
    if found_labels != CCI_EXPECTED_LABELS:
        raise RuntimeError(
            f"Expected exact CCI labels {CCI_EXPECTED_LABELS}, found {found_labels}"
        )

    sample = explainer.sid2sample[SAMPLE_ID]
    sample_barcodes = np.asarray(
        [norm_barcode(value) for value in sample.cells.tolist()],
        dtype=object,
    )

    mapping = mapping.sort_values("rid").reset_index(drop=True)
    if len(mapping) != sample_barcodes.size:
        raise RuntimeError(
            f"CCI mapping rows {len(mapping)} do not match RNA cells {sample_barcodes.size}"
        )
    if not np.array_equal(
        mapping["rid"].to_numpy(dtype=np.int64),
        np.arange(sample_barcodes.size, dtype=np.int64),
    ):
        raise RuntimeError("CCI mapping does not contain a complete ordered rid axis")
    if not np.array_equal(
        mapping["barcode"].to_numpy(dtype=object),
        sample_barcodes,
    ):
        mismatch = np.where(
            mapping["barcode"].to_numpy(dtype=object) != sample_barcodes
        )[0]
        raise RuntimeError(
            "CCI rid/barcode identity mismatch. "
            f"First mismatches: {mismatch[:10].tolist()}"
        )

    mapping["cci_cluster_id"] = mapping["cci_cluster"].astype(int)
    return mapping


def build_paired_cell_table(
    explainer: FrontierExplainerStandalone,
    atac: ATACReference,
    cci_mapping: pd.DataFrame,
) -> pd.DataFrame:
    model_table = explainer.cluster_df.copy()
    model_table = model_table[
        model_table["sid"].astype(str) == str(SAMPLE_ID)
    ].copy()
    if model_table.empty:
        raise ValueError(f"No model cells for {SAMPLE_ID} in {CLUSTERS_CSV}")

    model_table["rid"] = pd.to_numeric(
        model_table["rid"],
        errors="raise",
    ).astype(np.int64)
    model_table = model_table.sort_values("rid").drop_duplicates(
        subset=["sid", "rid"],
        keep="first",
    )

    sample = explainer.sid2sample[SAMPLE_ID]
    rows: list[dict[str, Any]] = []
    for _, model_row in model_table.iterrows():
        rid = int(model_row["rid"])
        if rid < 0 or rid >= sample.counts_shape[0]:
            continue
        barcode = norm_barcode(sample.cells[rid])
        atac_row = atac.barcode_to_row.get(str(barcode), -1)
        cci_row = cci_mapping.iloc[rid]
        split_value = split_5050(make_cell_key(SAMPLE_ID, barcode))

        rows.append(
            {
                "sid": SAMPLE_ID,
                "rid": rid,
                "barcode": barcode,
                "atac_row": int(atac_row),
                "split_bit": int(split_value),
                "split": "train" if split_value == 0 else "test",
                "cci_cluster_id": int(cci_row["cci_cluster_id"]),
                "internal_cluster_id": int(model_row["cluster_id"]),
                "group_id": int(model_row["group_id"]),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No model cells were assembled")

    paired_mask = table["atac_row"] >= 0
    paired_fraction = float(paired_mask.mean())
    log(
        "[PAIR]",
        f"model cells={len(table)}",
        f"paired ATAC={int(paired_mask.sum())}",
        f"fraction={paired_fraction:.4f}",
    )
    if paired_fraction < MIN_PAIRED_CELL_FRACTION:
        raise RuntimeError(
            f"Only {paired_fraction:.1%} of model cells matched ATAC; "
            f"required at least {MIN_PAIRED_CELL_FRACTION:.1%}."
        )

    table = table[paired_mask].copy().reset_index(drop=True)
    return table


# ============================================================
# 7) TRAIN-ONLY RE PANEL SELECTION
# ============================================================

def count_candidate_accessibility(
    store: SparseCSRStore,
    atac_rows: np.ndarray,
    candidate_peak_ids: np.ndarray,
) -> np.ndarray:
    candidates = np.unique(
        np.asarray(candidate_peak_ids, dtype=np.int64)
    )
    candidates = candidates[
        (candidates >= 0) & (candidates < int(store.shape[1]))
    ]
    if candidates.size == 0:
        raise RuntimeError("No candidate checkpoint peaks overlap the ATAC matrix")

    counts = np.zeros(candidates.size, dtype=np.int64)

    for position, atac_row in enumerate(atac_rows.tolist()):
        row_indices, row_values = store.row_indices_values(int(atac_row))
        if row_indices.size == 0:
            continue
        positive = row_values > 0
        row_indices = row_indices[positive]
        if row_indices.size == 0:
            continue

        candidate_pos = np.searchsorted(candidates, row_indices)
        valid = candidate_pos < candidates.size
        candidate_pos = candidate_pos[valid]
        row_indices_valid = row_indices[valid]
        matched = candidates[candidate_pos] == row_indices_valid
        if matched.any():
            counts[np.unique(candidate_pos[matched])] += 1

        if (position + 1) % 500 == 0:
            log(
                f"[RE SELECT] scanned {position + 1}/{len(atac_rows)} train cells"
            )

    return candidates, counts


def select_re_evaluation_panel(
    explainer: FrontierExplainerStandalone,
    atac: ATACReference,
    paired_table: pd.DataFrame,
) -> pd.DataFrame:
    candidate_ids = np.asarray(
        explainer.frontier_ckpt.get("selected_peak_raw_ids", []),
        dtype=np.int64,
    )
    if candidate_ids.size == 0:
        raise RuntimeError(
            "frontier_best.pt has no selected_peak_raw_ids. "
            "A direct, model-defined RE evaluation panel cannot be constructed."
        )

    train_rows = paired_table.loc[
        paired_table["split"] == "train",
        "atac_row",
    ].to_numpy(dtype=np.int64)
    if train_rows.size == 0:
        raise RuntimeError("No paired train cells for RE feature selection")

    candidates, counts = count_candidate_accessibility(
        atac.store,
        train_rows,
        candidate_ids,
    )
    prevalence = counts.astype(np.float64) / float(train_rows.size)
    variance = prevalence * (1.0 - prevalence)

    panel = pd.DataFrame(
        {
            "peak_raw_id": candidates,
            "train_accessible_cells": counts,
            "train_prevalence": prevalence,
            "train_bernoulli_variance": variance,
        }
    )

    strict = panel[
        (panel["train_accessible_cells"] >= RE_MIN_TRAIN_ACCESSIBLE_CELLS)
        & (panel["train_prevalence"] >= RE_MIN_TRAIN_PREVALENCE)
        & (panel["train_prevalence"] <= RE_MAX_TRAIN_PREVALENCE)
    ].copy()

    if len(strict) < RE_MIN_FEATURES:
        log(
            "[RE SELECT][WARNING]",
            f"strict panel has only {len(strict)} peaks; relaxing prevalence thresholds",
        )
        strict = panel[
            panel["train_accessible_cells"] >= max(
                2,
                RE_MIN_TRAIN_ACCESSIBLE_CELLS // 2,
            )
        ].copy()

    strict = strict.sort_values(
        ["train_bernoulli_variance", "train_accessible_cells", "peak_raw_id"],
        ascending=[False, False, True],
    ).head(RE_EVAL_N).reset_index(drop=True)

    if len(strict) < RE_MIN_FEATURES:
        raise RuntimeError(
            f"Only {len(strict)} evaluable REs remain; need at least {RE_MIN_FEATURES}."
        )

    strict["feature_index"] = np.arange(len(strict), dtype=np.int64)
    strict["peak_name"] = strict["peak_raw_id"].map(
        lambda value: f"peak_{int(value)}"
    )

    panel_path = OUT_DIR / "re_evaluation_peak_panel.csv"
    strict.to_csv(panel_path, index=False)
    log("[SAVE]", panel_path)
    return strict


def extract_binary_selected_matrix(
    store: SparseCSRStore,
    atac_rows: np.ndarray,
    selected_peak_ids: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(selected_peak_ids, dtype=np.int64)
    if selected.size == 0:
        raise ValueError("selected_peak_ids is empty")
    order = np.argsort(selected)
    selected_sorted = selected[order]
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)

    matrix_sorted = np.zeros(
        (len(atac_rows), selected_sorted.size),
        dtype=np.float32,
    )

    for cell_index, atac_row in enumerate(atac_rows.tolist()):
        row_indices, row_values = store.row_indices_values(int(atac_row))
        positive = row_values > 0
        row_indices = row_indices[positive]
        if row_indices.size > 0:
            positions = np.searchsorted(selected_sorted, row_indices)
            valid = positions < selected_sorted.size
            positions = positions[valid]
            row_indices_valid = row_indices[valid]
            matched = selected_sorted[positions] == row_indices_valid
            if matched.any():
                matrix_sorted[cell_index, np.unique(positions[matched])] = 1.0

        if (cell_index + 1) % 500 == 0:
            log(
                f"[TRUE RE] extracted {cell_index + 1}/{len(atac_rows)} test cells"
            )

    return matrix_sorted[:, inverse_order]


# ============================================================
# 8) RNA -> RE PREDICTION
# ============================================================

@torch.no_grad()
def build_peak_vectors(
    explainer: FrontierExplainerStandalone,
    peak_ids: np.ndarray,
    mode: str,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(peak_ids), PEAK_VECTOR_BATCH_SIZE):
        end = min(start + PEAK_VECTOR_BATCH_SIZE, len(peak_ids))
        vectors = explainer.score_peak_vec(
            np.asarray(peak_ids[start:end], dtype=np.int64),
            mode=mode,
        )
        chunks.append(vectors.detach())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def predict_re_for_cells(
    explainer: FrontierExplainerStandalone,
    cell_table: pd.DataFrame,
    peak_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = len(cell_table)
    n_peaks = len(peak_ids)

    pred_baseline = np.empty((n_cells, n_peaks), dtype=np.float32)
    pred_frontier = np.empty((n_cells, n_peaks), dtype=np.float32)

    peak_vectors_baseline = build_peak_vectors(
        explainer,
        peak_ids,
        mode="pretrain",
    )
    peak_vectors_frontier = build_peak_vectors(
        explainer,
        peak_ids,
        mode="after",
    )

    for start in range(0, n_cells, CELL_BATCH_SIZE):
        end = min(start + CELL_BATCH_SIZE, n_cells)
        batch = cell_table.iloc[start:end]
        cell_list = [
            (str(row["sid"]), int(row["rid"]))
            for _, row in batch.iterrows()
        ]

        z0 = explainer.encode_cells_base(cell_list)
        group_ids = batch["group_id"].to_numpy(dtype=np.int64)
        internal_cluster_ids = batch[
            "internal_cluster_id"
        ].to_numpy(dtype=np.int64)

        z_baseline = explainer.adjust_Z_batch(
            z0,
            group_ids,
            internal_cluster_ids,
            mode="pretrain",
            task="re",
        )
        z_frontier = explainer.adjust_Z_batch(
            z0,
            group_ids,
            internal_cluster_ids,
            mode="after",
            task="re",
        )

        pred_baseline[start:end] = (
            z_baseline @ peak_vectors_baseline.t()
        ).detach().cpu().numpy().astype(np.float32)
        pred_frontier[start:end] = (
            z_frontier @ peak_vectors_frontier.t()
        ).detach().cpu().numpy().astype(np.float32)

        del z0, z_baseline, z_frontier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[PRED RE] cells {end}/{n_cells}")

    return pred_baseline, pred_frontier


# ============================================================
# 9) CLUSTER PROFILES AND OUTPUT
# ============================================================

def cluster_mean_matrix(
    matrix: np.ndarray,
    cci_cluster_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cluster_ids = np.arange(10, dtype=np.int64)
    means = np.full(
        (10, matrix.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    counts = np.zeros(10, dtype=np.int64)

    for cluster_id in cluster_ids.tolist():
        mask = cci_cluster_ids == cluster_id
        counts[cluster_id] = int(mask.sum())
        if mask.any():
            means[cluster_id] = np.nanmean(
                matrix[mask],
                axis=0,
            ).astype(np.float32)

    missing = cluster_ids[counts == 0]
    if missing.size:
        raise RuntimeError(
            f"The held-out RE evaluation has no cells in CCI clusters "
            f"{missing.tolist()}."
        )
    return cluster_ids, means, counts


def profile_correlation_rows(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for cluster_id in range(10):
        true_values = np.asarray(truth[cluster_id], dtype=np.float64)
        pred_values = np.asarray(prediction[cluster_id], dtype=np.float64)
        valid = np.isfinite(true_values) & np.isfinite(pred_values)
        if valid.sum() >= 3 and np.std(true_values[valid]) > 1e-12 and np.std(
            pred_values[valid]
        ) > 1e-12:
            pearson = float(
                np.corrcoef(true_values[valid], pred_values[valid])[0, 1]
            )
        else:
            pearson = np.nan
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_features": int(valid.sum()),
                "pearson": pearson,
            }
        )
    return pd.DataFrame(rows)


def save_profile_npz(
    model_name: str,
    cluster_ids: np.ndarray,
    cluster_counts: np.ndarray,
    true_cluster: np.ndarray,
    pred_cluster: np.ndarray,
    peak_panel: pd.DataFrame,
    test_table: pd.DataFrame,
) -> Path:
    output_path = (
        OUT_DIR
        / f"cluster_profile_full_{SPLIT_NAME}_re_{model_name}.npz"
    )
    residual_cluster = pred_cluster - true_cluster

    np.savez_compressed(
        output_path,
        cluster_ids=cluster_ids.astype(np.int64),
        cluster_counts=cluster_counts.astype(np.int64),
        true_cluster=true_cluster.astype(np.float32),
        pred_cluster=pred_cluster.astype(np.float32),
        residual_cluster=residual_cluster.astype(np.float32),
        feature_ids=peak_panel["peak_raw_id"].to_numpy(dtype=np.int64),
        feature_names=peak_panel["peak_name"].astype(str).to_numpy(dtype=object),
        test_rids=test_table["rid"].to_numpy(dtype=np.int64),
        test_barcodes=test_table["barcode"].astype(str).to_numpy(dtype=object),
        test_cci_cluster_ids=test_table[
            "cci_cluster_id"
        ].to_numpy(dtype=np.int64),
    )
    log("[SAVE]", output_path)
    return output_path


def main() -> None:
    set_seed(SEED)
    if REQUIRE_CUDA and DEVICE != "cuda":
        raise RuntimeError("CUDA is required for RE prediction")

    log("[START] matched-reference RE profile preparation")
    log("[DEVICE]", DEVICE)
    log("[BASE_CKPT]", BASE_CKPT)
    log("[FRONTIER_CKPT]", FRONTIER_CKPT)
    log("[OUT_DIR]", OUT_DIR)

    explainer = FrontierExplainerStandalone()
    if SAMPLE_ID not in explainer.sid2sample:
        raise KeyError(
            f"{SAMPLE_ID} not found. Available samples: "
            f"{sorted(explainer.sid2sample.keys())}"
        )

    atac = build_atac_reference(explainer, SAMPLE_ID)
    cci_mapping = load_exact_cci_mapping(explainer)
    paired_table = build_paired_cell_table(
        explainer,
        atac,
        cci_mapping,
    )
    paired_table.to_csv(
        OUT_DIR / "re_paired_cell_mapping_all.csv",
        index=False,
    )

    peak_panel = select_re_evaluation_panel(
        explainer,
        atac,
        paired_table,
    )
    peak_ids = peak_panel["peak_raw_id"].to_numpy(dtype=np.int64)

    test_table = paired_table[
        paired_table["split"] == SPLIT_NAME
    ].copy().reset_index(drop=True)
    if test_table.empty:
        raise RuntimeError("No held-out paired test cells")

    test_cluster_counts = (
        test_table["cci_cluster_id"]
        .value_counts()
        .reindex(range(10), fill_value=0)
        .sort_index()
    )
    log("[TEST CCI COUNTS]", test_cluster_counts.tolist())
    if (test_cluster_counts == 0).any():
        raise RuntimeError(
            "At least one CCI10 cluster has zero paired test cells: "
            f"{test_cluster_counts[test_cluster_counts == 0].index.tolist()}"
        )

    true_re_cell = extract_binary_selected_matrix(
        atac.store,
        test_table["atac_row"].to_numpy(dtype=np.int64),
        peak_ids,
    )
    pred_re_baseline_cell, pred_re_frontier_cell = predict_re_for_cells(
        explainer,
        test_table,
        peak_ids,
    )

    cluster_ids, true_cluster, cluster_counts = cluster_mean_matrix(
        true_re_cell,
        test_table["cci_cluster_id"].to_numpy(dtype=np.int64),
    )
    _, pred_baseline_cluster, _ = cluster_mean_matrix(
        pred_re_baseline_cell,
        test_table["cci_cluster_id"].to_numpy(dtype=np.int64),
    )
    _, pred_frontier_cluster, _ = cluster_mean_matrix(
        pred_re_frontier_cell,
        test_table["cci_cluster_id"].to_numpy(dtype=np.int64),
    )

    baseline_path = save_profile_npz(
        "baseline",
        cluster_ids,
        cluster_counts,
        true_cluster,
        pred_baseline_cluster,
        peak_panel,
        test_table,
    )
    frontier_path = save_profile_npz(
        "frontier",
        cluster_ids,
        cluster_counts,
        true_cluster,
        pred_frontier_cluster,
        peak_panel,
        test_table,
    )

    baseline_corr = profile_correlation_rows(
        true_cluster,
        pred_baseline_cluster,
    ).rename(columns={"pearson": "baseline_pearson"})
    frontier_corr = profile_correlation_rows(
        true_cluster,
        pred_frontier_cluster,
    ).rename(columns={"pearson": "frontier_pearson"})

    qc = baseline_corr.merge(
        frontier_corr[["cluster_id", "frontier_pearson"]],
        on="cluster_id",
        how="outer",
    )
    qc["delta_pearson"] = (
        qc["frontier_pearson"] - qc["baseline_pearson"]
    )
    qc["n_cells"] = cluster_counts
    qc.to_csv(
        OUT_DIR / "re_cluster_profile_correlation_qc.csv",
        index=False,
    )

    feature_qc = peak_panel.copy()
    feature_qc["test_prevalence"] = np.nanmean(
        true_re_cell,
        axis=0,
    )
    feature_qc.to_csv(
        OUT_DIR / "re_evaluation_peak_panel_with_test_qc.csv",
        index=False,
    )

    if SAVE_CELL_LEVEL_MATRICES:
        np.savez_compressed(
            OUT_DIR / "re_cell_level_test_matrices.npz",
            true_re=true_re_cell.astype(np.float32),
            pred_re_baseline=pred_re_baseline_cell.astype(np.float32),
            pred_re_frontier=pred_re_frontier_cell.astype(np.float32),
            peak_raw_ids=peak_ids.astype(np.int64),
            cci_cluster_ids=test_table[
                "cci_cluster_id"
            ].to_numpy(dtype=np.int64),
            rids=test_table["rid"].to_numpy(dtype=np.int64),
        )

    summary = {
        "dataset": DATASET_NAME,
        "sample_id": SAMPLE_ID,
        "split": SPLIT_NAME,
        "base_checkpoint": str(BASE_CKPT),
        "frontier_checkpoint": str(FRONTIER_CKPT),
        "projected_atac_path": str(atac.projected_path),
        "atac_manifest_path": (
            str(atac.manifest_path)
            if atac.manifest_path is not None
            else None
        ),
        "cci_mapping_path": str(CCI_CELL_LEVEL_TABLE),
        "n_all_model_paired_cells": int(len(paired_table)),
        "n_test_cells": int(len(test_table)),
        "test_cluster_counts": {
            str(index): int(value)
            for index, value in enumerate(cluster_counts.tolist())
        },
        "n_re_features": int(len(peak_panel)),
        "baseline_output": str(baseline_path),
        "frontier_output": str(frontier_path),
        "mean_baseline_cluster_pearson": float(
            np.nanmean(qc["baseline_pearson"])
        ),
        "mean_frontier_cluster_pearson": float(
            np.nanmean(qc["frontier_pearson"])
        ),
        "mean_delta_cluster_pearson": float(
            np.nanmean(qc["delta_pearson"])
        ),
        "truth_definition": (
            "Observed projected ATAC accessibility, binarized per cell."
        ),
        "feature_selection": (
            "Checkpoint-selected RE candidates ranked by train-only observed "
            "accessibility variance; held-out test cells were not used."
        ),
    }
    with open(
        OUT_DIR / "re_profile_generation_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    log("[QC]", qc.to_string(index=False))
    log("[DONE] RE profile files are ready")


if __name__ == "__main__":
    main()
