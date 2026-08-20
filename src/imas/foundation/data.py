from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def topk_from_csr_row(indices: np.ndarray, data: np.ndarray, topk: int):
    if indices.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    k = min(int(topk), indices.size)
    selected = np.argpartition(data, -k)[-k:]
    order = np.argsort(data[selected])[::-1]
    selected = selected[order]
    return indices[selected].astype(np.int64), data[selected].astype(np.float32)


class CSRNPZStore:
    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        self.data = self.indices = self.indptr = self.shape = None

    def open(self):
        if self.data is None:
            z = np.load(self.path, mmap_mode="r")
            self.data, self.indices, self.indptr = z["data"], z["indices"], z["indptr"]
            self.shape = tuple(map(int, z["shape"].tolist()))
        return self

    def get_row_topk(self, row: int, topk: int):
        self.open()
        start, end = int(self.indptr[row]), int(self.indptr[row + 1])
        return topk_from_csr_row(self.indices[start:end], self.data[start:end], topk)


class PairedCellsDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cancer_to_id: dict[str, int]):
        self.frame = frame.reset_index(drop=True).copy()
        self.cancer_to_id = cancer_to_id

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return (
            str(row["dataset"]), str(row["sample_id_rna"]), str(row["sample_id_atac"]),
            int(row["row_rna"]), int(row["row_atac"]),
            int(self.cancer_to_id[str(row["cancer_type"])]),
        )


def make_sparse_collate(
    rna_store_getter: Callable[[str, str], CSRNPZStore],
    atac_store_getter: Callable[[str, str], CSRNPZStore],
    n_gene: int,
    n_peak: int,
    rna_topk: int = 2048,
    atac_topk: int = 1024,
):
    def collate(batch):
        ri, rv, ro = [], [], [0]
        ai, av, ao = [], [], [0]
        cancers = []
        for dataset, sid_rna, sid_atac, row_rna, row_atac, cancer_id in batch:
            cancers.append(cancer_id)
            idx, val = rna_store_getter(dataset, sid_rna).get_row_topk(row_rna, rna_topk)
            mask = (idx >= 0) & (idx < n_gene)
            ri.append(idx[mask]); rv.append(val[mask]); ro.append(ro[-1] + int(mask.sum()))
            idx, val = atac_store_getter(dataset, sid_atac).get_row_topk(row_atac, atac_topk)
            mask = (idx >= 0) & (idx < n_peak)
            ai.append(idx[mask]); av.append(val[mask]); ao.append(ao[-1] + int(mask.sum()))
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)
        return (
            torch.from_numpy(cat(ri, np.int64)), torch.from_numpy(cat(rv, np.float32)), torch.tensor(ro[:-1]),
            torch.from_numpy(cat(ai, np.int64)), torch.from_numpy(cat(av, np.float32)), torch.tensor(ao[:-1]),
            torch.tensor(cancers, dtype=torch.long),
        )
    return collate
