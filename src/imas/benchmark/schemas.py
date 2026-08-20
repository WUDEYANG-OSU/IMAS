"""Common exchange schema for virtual-perturbation benchmark responses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class PerturbationResponse:
    method: str
    target_gene: str
    gene_ids: np.ndarray
    cell_ids: np.ndarray
    baseline_state: np.ndarray
    perturbed_state: np.ndarray

    @property
    def response_delta(self) -> np.ndarray:
        return np.asarray(self.perturbed_state) - np.asarray(self.baseline_state)

    def validate(self) -> None:
        baseline = np.asarray(self.baseline_state)
        perturbed = np.asarray(self.perturbed_state)
        if baseline.shape != perturbed.shape:
            raise ValueError("baseline_state and perturbed_state must have identical shape")
        if baseline.ndim != 2:
            raise ValueError("state matrices must be cells x genes")
        if baseline.shape != (len(self.cell_ids), len(self.gene_ids)):
            raise ValueError("state matrix shape does not match cell_ids/gene_ids")

    def save_npz(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            method=np.asarray(self.method),
            target_gene=np.asarray(self.target_gene),
            gene_ids=np.asarray(self.gene_ids).astype(str),
            cell_ids=np.asarray(self.cell_ids).astype(str),
            baseline_state=np.asarray(self.baseline_state, dtype=np.float32),
            perturbed_state=np.asarray(self.perturbed_state, dtype=np.float32),
            response_delta=np.asarray(self.response_delta, dtype=np.float32),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "PerturbationResponse":
        z = np.load(Path(path), allow_pickle=True)
        return cls(
            method=str(z["method"].item()),
            target_gene=str(z["target_gene"].item()),
            gene_ids=np.asarray(z["gene_ids"]).astype(str),
            cell_ids=np.asarray(z["cell_ids"]).astype(str),
            baseline_state=np.asarray(z["baseline_state"], dtype=np.float32),
            perturbed_state=np.asarray(z["perturbed_state"], dtype=np.float32),
        )
