from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterCommLTC(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, reconstruction_dim: int,
                 rollout_steps: int = 12, dt: float = 0.20, tau_min: float = 0.15):
        super().__init__(); self.rollout_steps, self.dt, self.tau_min = rollout_steps, dt, tau_min
        self.input_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention = nn.Sequential(nn.Linear(hidden_dim*2 + 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.core = nn.Sequential(nn.Linear(hidden_dim + input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.tau_head = nn.Sequential(nn.Linear(hidden_dim + input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.reconstruction_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, reconstruction_dim))

    def forward(self, x, neighbor_index, neighbor_mask, edge_features):
        h = self.input_net(x); tau = None; _, k = neighbor_index.shape
        for _ in range(self.rollout_steps):
            hn = h[neighbor_index]; hi = h.unsqueeze(1).expand(-1, k, -1)
            logits = self.attention(torch.cat([hi, hn, edge_features], -1)).squeeze(-1).masked_fill(~neighbor_mask, -1e9)
            attention = torch.softmax(logits, 1) * neighbor_mask.float(); attention /= attention.sum(1, keepdim=True) + 1e-8
            message = torch.sum(attention[..., None] * self.edge_message(hn), 1)
            combined = torch.cat([h + message, x], -1)
            drive = torch.tanh(self.core(combined)); tau = F.softplus(self.tau_head(combined)) + self.tau_min
            h = h + self.dt * ((-h + drive) / tau)
        return torch.sigmoid(self.time_head(h)).squeeze(-1), self.reconstruction_head(h), tau.squeeze(-1)
