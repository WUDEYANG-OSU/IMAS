from __future__ import annotations
import torch
import torch.nn as nn


class HyperAdapterHashV2(nn.Module):
    def __init__(self, embedding_dim: int, rank: int, n_tf: int, n_gene: int,
                 peak_hash_size: int, n_cancer: int, n_group: int,
                 context_dim: int = 64, hidden_dim: int = 256,
                 group_scale_alpha: float = 0.12, use_group_scale_bias: bool = True):
        super().__init__()
        self.d, self.rank, self.peak_hash_size = embedding_dim, rank, int(peak_hash_size)
        self.group_scale_alpha, self.use_group_scale_bias = group_scale_alpha, use_group_scale_bias
        self.context = nn.Parameter(torch.zeros(context_dim)); nn.init.normal_(self.context, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 4 * rank * embedding_dim))
        self.C_tf, self.C_gene = nn.Embedding(n_tf, rank), nn.Embedding(n_gene, rank)
        self.C_peak_hash, self.C_cancer = nn.Embedding(peak_hash_size, rank), nn.Embedding(n_cancer, rank)
        self.S_group, self.B_group = nn.Embedding(n_group, embedding_dim), nn.Embedding(n_group, embedding_dim)
        for emb in [self.C_tf, self.C_gene, self.C_peak_hash, self.C_cancer, self.S_group, self.B_group]:
            nn.init.zeros_(emb.weight)

    def forward(self):
        return self.mlp(self.context).view(4, self.rank, self.d).unbind(0)
    def delta_tf(self, ids, basis): return self.C_tf(ids) @ basis
    def delta_gene(self, ids, basis): return self.C_gene(ids) @ basis
    def delta_peak_hash(self, raw_ids, basis): return self.C_peak_hash(torch.remainder(raw_ids, self.peak_hash_size)) @ basis
    def delta_cancer(self, ids, basis): return self.C_cancer(ids) @ basis
    def apply_group(self, z, group_ids):
        bias = self.B_group(group_ids)
        if not self.use_group_scale_bias: return z + bias
        return z * (1.0 + self.group_scale_alpha * torch.tanh(self.S_group(group_ids))) + bias
