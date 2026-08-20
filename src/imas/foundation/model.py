from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, embedding_dim: int):
        super().__init__()
        self.hash_size = int(hash_size)
        self.emb = nn.Embedding(self.hash_size, embedding_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, peak_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(torch.remainder(peak_idx, self.hash_size))


class PlanCDirectionAwareModel(nn.Module):
    """Direction-aware RNA/ATAC foundation model with cancer-conditioned embeddings."""

    def __init__(
        self,
        n_gene: int,
        n_tf: int,
        n_cancer: int,
        peak_hash_size: int = 2_000_000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        d = embedding_dim
        self.peak_hash_size = int(peak_hash_size)
        self.gene_in, self.gene_out = nn.Embedding(n_gene, d), nn.Embedding(n_gene, d)
        self.tf_in, self.tf_out = nn.Embedding(n_tf, d), nn.Embedding(n_tf, d)
        self.peak_in = PeakHashEmbedding(peak_hash_size, d)
        self.peak_out = PeakHashEmbedding(peak_hash_size, d)
        self.gene_in_c, self.gene_out_c = nn.Embedding(n_cancer, d), nn.Embedding(n_cancer, d)
        self.peak_in_c, self.peak_out_c = nn.Embedding(n_cancer, d), nn.Embedding(n_cancer, d)
        self.tf_in_c, self.tf_out_c = nn.Embedding(n_cancer, d), nn.Embedding(n_cancer, d)
        self.cell_c = nn.Embedding(n_cancer, d)
        self.rna_proj = nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d))
        self.atac_proj = nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, d))
        for emb in [self.gene_in, self.gene_out, self.tf_in, self.tf_out,
                    self.gene_in_c, self.gene_out_c, self.peak_in_c, self.peak_out_c,
                    self.tf_in_c, self.tf_out_c, self.cell_c]:
            nn.init.normal_(emb.weight, std=0.02)

    @staticmethod
    def _segment_ids(offsets: torch.Tensor, total_tokens: int) -> torch.Tensor:
        batch = offsets.shape[0]
        end = torch.empty((batch,), device=offsets.device, dtype=torch.long)
        end[:-1], end[-1] = offsets[1:], total_tokens
        lengths = (end - offsets).clamp(min=0)
        return torch.repeat_interleave(torch.arange(batch, device=offsets.device), lengths)

    def encode_rna(self, idx, value, offsets, cancer_ids):
        seg = self._segment_ids(offsets, idx.shape[0])
        c_tok = cancer_ids.index_select(0, seg)
        token = (self.gene_in(idx) + self.gene_in_c(c_tok)) * value.unsqueeze(-1)
        out = torch.zeros((offsets.shape[0], token.shape[1]), device=token.device, dtype=token.dtype)
        out.index_add_(0, seg, token)
        return F.normalize(self.rna_proj(out) + self.cell_c(cancer_ids), dim=-1)

    def encode_atac(self, idx, value, offsets, cancer_ids):
        seg = self._segment_ids(offsets, idx.shape[0])
        c_tok = cancer_ids.index_select(0, seg)
        token = (self.peak_in(idx) + self.peak_in_c(c_tok)) * value.unsqueeze(-1)
        out = torch.zeros((offsets.shape[0], token.shape[1]), device=token.device, dtype=token.dtype)
        out.index_add_(0, seg, token)
        return F.normalize(self.atac_proj(out) + self.cell_c(cancer_ids), dim=-1)

    def _score(self, left, right):
        return (left * right).sum(-1)

    def score_tf_gene_f(self, tf_i, gene_i, c):
        return self._score(self.tf_in(tf_i)+self.tf_in_c(c), self.gene_out(gene_i)+self.gene_out_c(c))
    def score_gene_tf_r(self, gene_i, tf_i, c):
        return self._score(self.gene_in(gene_i)+self.gene_in_c(c), self.tf_out(tf_i)+self.tf_out_c(c))
    def score_peak_gene_f(self, peak_i, gene_i, c):
        return self._score(self.peak_in(peak_i)+self.peak_in_c(c), self.gene_out(gene_i)+self.gene_out_c(c))
    def score_gene_peak_r(self, gene_i, peak_i, c):
        return self._score(self.gene_in(gene_i)+self.gene_in_c(c), self.peak_out(peak_i)+self.peak_out_c(c))
    def score_tf_peak_f(self, tf_i, peak_i, c):
        return self._score(self.tf_in(tf_i)+self.tf_in_c(c), self.peak_out(peak_i)+self.peak_out_c(c))
    def score_peak_tf_r(self, peak_i, tf_i, c):
        return self._score(self.peak_in(peak_i)+self.peak_in_c(c), self.tf_out(tf_i)+self.tf_out_c(c))

    def cancer_l2(self):
        weights = [self.gene_in_c.weight, self.gene_out_c.weight, self.peak_in_c.weight,
                   self.peak_out_c.weight, self.tf_in_c.weight, self.tf_out_c.weight,
                   self.cell_c.weight]
        return sum((weight * weight).mean() for weight in weights)

# Backwards-compatible source name.
PlanC_DirAware_Cancer_HashPeak = PlanCDirectionAwareModel
