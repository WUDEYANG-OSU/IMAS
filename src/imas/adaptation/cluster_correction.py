from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ClusterSummaryTable(nn.Module):
    def __init__(self, summaries: dict[int, np.ndarray]):
        super().__init__(); keys = sorted(summaries)
        self.cluster_ids = keys; self.cluster_to_row = {cluster: i for i, cluster in enumerate(keys)}
        self.register_buffer("summary_table", torch.from_numpy(np.stack([summaries[k] for k in keys]).astype(np.float32)), persistent=True)

    @property
    def summary_dim(self): return int(self.summary_table.shape[1])

    def lookup(self, cluster_ids: torch.Tensor):
        rows = [self.cluster_to_row.get(int(value), 0) for value in cluster_ids.detach().cpu().tolist()]
        return self.summary_table[torch.tensor(rows, device=cluster_ids.device)]


class ClusterCellCorrectionHead(nn.Module):
    def __init__(self, embedding_dim: int, summary_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim + summary_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
    def forward(self, z_cell, cluster_summary): return self.mlp(torch.cat([z_cell, cluster_summary], dim=-1))
