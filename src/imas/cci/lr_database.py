from __future__ import annotations
import pandas as pd


def _find(frame, candidates):
    lower = {str(c).lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns: return candidate
        if candidate.lower() in lower: return lower[candidate.lower()]
    return None


def standardize_lr_table(frame: pd.DataFrame) -> pd.DataFrame:
    ligand = _find(frame, ["ligand", "ligand_symbol", "source"])
    receptor = _find(frame, ["receptor", "receptor_symbol", "target"])
    if ligand is None or receptor is None: raise KeyError("Ligand/receptor columns not found")
    out = frame.rename(columns={ligand: "ligand", receptor: "receptor"}).copy()
    out["ligand"] = out["ligand"].astype(str).str.strip(); out["receptor"] = out["receptor"].astype(str).str.strip()
    return out[(out["ligand"] != "") & (out["receptor"] != "")].drop_duplicates(["ligand", "receptor"])
