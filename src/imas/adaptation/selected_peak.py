from __future__ import annotations
from collections import Counter
import numpy as np
import torch
import torch.nn as nn


class SelectedPeakResidual(nn.Module):
    def __init__(self, selected_peak_ids: np.ndarray, embedding_dim: int):
        super().__init__(); ids = np.asarray(selected_peak_ids, np.int64)
        self.register_buffer("selected_peak_raw_ids", torch.from_numpy(ids), persistent=True)
        self.embedding_dim = int(embedding_dim)
        self.emb = nn.Embedding(max(1, ids.size), embedding_dim); nn.init.zeros_(self.emb.weight)
        self.raw_to_local = {int(value): index for index, value in enumerate(ids.tolist())}

    @property
    def n_selected(self): return int(self.selected_peak_raw_ids.numel())

    def lookup(self, raw_ids: torch.Tensor):
        out = torch.zeros((raw_ids.numel(), self.embedding_dim), device=raw_ids.device, dtype=self.emb.weight.dtype)
        if self.n_selected == 0 or raw_ids.numel() == 0: return out
        raw = raw_ids.detach().cpu().numpy().astype(np.int64)
        local = np.asarray([self.raw_to_local.get(int(value), -1) for value in raw], np.int64)
        mask = local >= 0
        if mask.any():
            out[torch.from_numpy(np.where(mask)[0]).to(raw_ids.device)] = self.emb(torch.from_numpy(local[mask]).to(raw_ids.device))
        return out


def mine_selected_peaks(gene_pool: np.ndarray, gene_to_peaks: dict[int, list[int]],
                        max_peaks: int = 50_000, min_edge_frequency: int = 2) -> np.ndarray:
    counts = Counter()
    for gene in np.asarray(gene_pool, np.int64).tolist():
        counts.update(map(int, gene_to_peaks.get(int(gene), [])))
    items = [(peak, count) for peak, count in counts.items() if count >= min_edge_frequency] or list(counts.items())
    items.sort(key=lambda item: (-item[1], item[0]))
    return np.asarray([peak for peak, _ in items[:max_peaks]], np.int64)
