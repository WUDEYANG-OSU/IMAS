from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from .losses import symmetric_alignment_loss


@dataclass
class FoundationTrainConfig:
    steps: int = 30_000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    temperature: float = 0.07
    grad_clip: float = 1.0
    cancer_l2: float = 1e-4
    log_every: int = 200
    amp_dtype: torch.dtype = torch.bfloat16


class FoundationTrainer:
    """Reusable alignment trainer. Relation-specific graph losses can be injected."""
    def __init__(self, model, config: FoundationTrainConfig, device="cuda"):
        self.model, self.config, self.device = model.to(device), config, torch.device(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def fit(self, loader, graph_loss_fn=None, output_checkpoint: str | Path | None = None):
        iterator = iter(loader)
        for step in range(1, self.config.steps + 1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            batch = [x.to(self.device, non_blocking=True) for x in batch]
            rna_i, rna_v, rna_o, atac_i, atac_v, atac_o, cancer = batch
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=self.config.amp_dtype, enabled=self.device.type == "cuda"):
                zr = self.model.encode_rna(rna_i, rna_v, rna_o, cancer)
                za = self.model.encode_atac(atac_i, atac_v, atac_o, cancer)
                loss = symmetric_alignment_loss(zr, za, self.config.temperature)
                if graph_loss_fn is not None: loss = loss + graph_loss_fn(self.model, cancer)
                loss = loss + self.config.cancer_l2 * self.model.cancer_l2()
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip); self.optimizer.step()
            if step == 1 or step % self.config.log_every == 0:
                print(f"[foundation] step={step} loss={float(loss):.6f}", flush=True)
        if output_checkpoint is not None:
            path = Path(output_checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": self.model.state_dict(), "config": self.config.__dict__}, path)
        return self.model
