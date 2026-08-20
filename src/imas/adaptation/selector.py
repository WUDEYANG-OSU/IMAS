from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorSelector(nn.Module):
    def __init__(self, summary_dim: int, n_relations: int = 6, bias: float = 1e-6):
        super().__init__(); self.bias = float(bias)
        self.context = nn.Parameter(torch.zeros(32))
        self.mlp = nn.Sequential(nn.Linear(32 + summary_dim, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU(), nn.Linear(128, n_relations))
        self.register_buffer("summary", torch.zeros(summary_dim), persistent=True)

    def set_summary(self, values: np.ndarray):
        values = np.asarray(values, np.float32).reshape(-1)
        if values.size != self.summary.numel(): raise ValueError("Summary dimension mismatch")
        self.summary.copy_(torch.from_numpy(values))

    def forward(self):
        weights = F.softplus(self.mlp(torch.cat([self.context, self.summary]))) + self.bias
        return weights / (weights.mean() + 1e-12)

PriorSelectorV1 = PriorSelector
