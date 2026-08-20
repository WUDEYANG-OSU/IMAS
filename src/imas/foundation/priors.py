from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

PRIOR_FILES = {
    "tf_gene_f": "tf_gene_f.full.npz", "gene_tf_r": "gene_tf_r.full.npz",
    "peak_gene_f": "peak_gene_f.full.npz", "gene_peak_r": "gene_peak_r.full.npz",
    "tf_peak_f": "tf_peak_f.full.npz", "peak_tf_r": "peak_tf_r.full.npz",
}


def load_global_priors(root: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = Path(root)
    cache = {}
    for relation, filename in PRIOR_FILES.items():
        path = root / filename
        if not path.exists(): raise FileNotFoundError(path)
        z = np.load(path, mmap_mode="r")
        src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
        if src.size == 0: raise ValueError(f"Empty prior: {path}")
        cache[relation] = (src, dst)
    return cache


def sample_edges(src: torch.Tensor, dst: torch.Tensor, n: int):
    indices = torch.randint(0, src.numel(), (int(n),), device=src.device)
    return src.index_select(0, indices), dst.index_select(0, indices)
