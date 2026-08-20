from __future__ import annotations

import numpy as np
import pandas as pd


def cap_by_tf(frame: pd.DataFrame, topm: int | None) -> pd.DataFrame:
    if topm is None:
        return frame.copy()
    return (
        frame.sort_values(["tf", "weight", "re"], ascending=[True, False, True])
        .groupby("tf", sort=False, group_keys=False)
        .head(int(topm))
        .reset_index(drop=True)
    )


def cap_by_re(frame: pd.DataFrame, topn: int | None) -> pd.DataFrame:
    if topn is None:
        return frame.copy()
    return (
        frame.sort_values(["re", "weight", "tf"], ascending=[True, False, True])
        .groupby("re", sort=False, group_keys=False)
        .head(int(topn))
        .reset_index(drop=True)
    )


def mix_rank_evidence(
    frame: pd.DataFrame,
    *,
    alpha: float = 0.5,
    beta: float = 1.0,
    epsilon: float = 1e-12,
    normalize_per_tf: bool = True,
) -> pd.DataFrame:
    """Combine motif evidence with within-TF rank support."""
    required = {"tf", "re", "weight"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"TF-RE table missing columns: {sorted(missing)}")
    out = frame.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out[np.isfinite(out["weight"]) & (out["weight"] >= 0)].copy()
    out["rank"] = (
        out.groupby("tf")["weight"].rank(method="first", ascending=False).astype(np.float64)
    )
    out["weight_original"] = out["weight"]
    out["weight"] = np.power(out["weight"] + epsilon, beta) * np.power(out["rank"], -alpha)
    if normalize_per_tf:
        denom = out.groupby("tf")["weight"].transform("sum")
        out["weight"] = out["weight"] / np.maximum(denom, epsilon)
    return out.reset_index(drop=True)


def build_tf_re_edges(
    fimo_hits: pd.DataFrame,
    *,
    tf_col: str = "motif_alt_id",
    re_col: str = "sequence_name",
    pvalue_col: str = "p-value",
    topm_per_tf: int | None = None,
    topn_per_re: int | None = None,
    rank_alpha: float = 0.5,
    evidence_beta: float = 1.0,
) -> pd.DataFrame:
    """Convert FIMO hits to weighted TF-to-regulatory-element edges."""
    required = [tf_col, re_col, pvalue_col]
    if missing := [c for c in required if c not in fimo_hits.columns]:
        raise KeyError(f"FIMO table missing columns: {missing}")
    pvalues = pd.to_numeric(fimo_hits[pvalue_col], errors="coerce").clip(lower=1e-300)
    out = pd.DataFrame(
        {
            "tf": fimo_hits[tf_col].astype(str).str.strip(),
            "re": fimo_hits[re_col].astype(str).str.strip(),
            "weight": -np.log10(pvalues),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.groupby(["tf", "re"], as_index=False)["weight"].max()
    out = mix_rank_evidence(out, alpha=rank_alpha, beta=evidence_beta)
    out = cap_by_tf(out, topm_per_tf)
    out = cap_by_re(out, topn_per_re)
    return pd.DataFrame(
        {
            "src_name": "TF:" + out["tf"].astype(str),
            "dst_name": "RE:" + out["re"].astype(str),
            "weight": out["weight"].astype(float),
        }
    ).reset_index(drop=True)
