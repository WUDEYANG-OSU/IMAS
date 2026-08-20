from __future__ import annotations
import torch


def alignment_top1(model, loader, device: str | torch.device = "cuda") -> float:
    model.eval(); hits = total = 0
    with torch.no_grad():
        for batch in loader:
            rna_i, rna_v, rna_o, atac_i, atac_v, atac_o, cancer = [x.to(device) for x in batch]
            zr = model.encode_rna(rna_i, rna_v, rna_o, cancer)
            za = model.encode_atac(atac_i, atac_v, atac_o, cancer)
            prediction = (zr @ za.T).argmax(dim=1)
            target = torch.arange(prediction.numel(), device=prediction.device)
            hits += int((prediction == target).sum())
            total += prediction.numel()
    model.train()
    return hits / max(1, total)
