from __future__ import annotations
import torch
import torch.nn.functional as F


def binary_graph_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    return (
        F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    )


def symmetric_alignment_loss(z_rna: torch.Tensor, z_atac: torch.Tensor, temperature: float = 0.07):
    logits = (z_rna @ z_atac.T) / float(temperature)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
