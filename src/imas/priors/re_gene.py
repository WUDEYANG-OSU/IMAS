from __future__ import annotations

import numpy as np
import pandas as pd


_CANONICAL_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}


def canonicalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    text = text.upper()
    return text


def is_canonical_chromosome(value: object) -> bool:
    return canonicalize_chromosome(value) in _CANONICAL_CHROMS


def build_re_gene_edges(
    peaks: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    max_distance: int = 200_000,
    topk_per_peak: int = 3,
    tau: float = 50_000.0,
    peak_chrom: str = "chrom",
    peak_start: str = "start",
    peak_end: str = "end",
    peak_name: str = "peak",
    gene_chrom: str = "chrom",
    gene_tss: str = "tss",
    gene_name: str = "gene",
) -> pd.DataFrame:
    """Build distance-decayed regulatory-element-to-gene edges."""
    p = peaks.copy()
    g = genes.copy()
    required_p = [peak_chrom, peak_start, peak_end, peak_name]
    required_g = [gene_chrom, gene_tss, gene_name]
    if missing := [c for c in required_p if c not in p.columns]:
        raise KeyError(f"peaks missing columns: {missing}")
    if missing := [c for c in required_g if c not in g.columns]:
        raise KeyError(f"genes missing columns: {missing}")
    p["_chrom"] = p[peak_chrom].map(canonicalize_chromosome)
    g["_chrom"] = g[gene_chrom].map(canonicalize_chromosome)
    p = p[p["_chrom"].isin(_CANONICAL_CHROMS)].copy()
    g = g[g["_chrom"].isin(_CANONICAL_CHROMS)].copy()
    p["_center"] = (
        pd.to_numeric(p[peak_start], errors="raise")
        + pd.to_numeric(p[peak_end], errors="raise")
    ) / 2.0
    g["_tss"] = pd.to_numeric(g[gene_tss], errors="raise")
    records: list[pd.DataFrame] = []
    for chrom, peak_block in p.groupby("_chrom", sort=False):
        gene_block = g[g["_chrom"] == chrom].sort_values("_tss")
        if gene_block.empty:
            continue
        tss = gene_block["_tss"].to_numpy(dtype=float)
        gene_names = gene_block[gene_name].astype(str).to_numpy()
        for _, row in peak_block.iterrows():
            center = float(row["_center"])
            lo = np.searchsorted(tss, center - max_distance, side="left")
            hi = np.searchsorted(tss, center + max_distance, side="right")
            if hi <= lo:
                continue
            distances = np.abs(tss[lo:hi] - center)
            order = np.lexsort((gene_names[lo:hi], distances))[: int(topk_per_peak)]
            selected_dist = distances[order]
            records.append(
                pd.DataFrame(
                    {
                        "src_name": f"RE:{row[peak_name]}",
                        "dst_name": [f"Gene:{x}" for x in gene_names[lo:hi][order]],
                        "distance": selected_dist.astype(np.float64),
                        "weight": np.exp(-selected_dist / max(float(tau), 1e-12)),
                    }
                )
            )
    if not records:
        return pd.DataFrame(columns=["src_name", "dst_name", "distance", "weight"])
    out = pd.concat(records, ignore_index=True)
    return out.sort_values(["src_name", "distance", "dst_name"]).reset_index(drop=True)
