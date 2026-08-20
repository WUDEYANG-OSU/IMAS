from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairingResult:
    pairs: pd.DataFrame
    strategy: str
    n_rna: int
    n_atac: int
    n_pairs: int
    n_ambiguous_rna: int = 0
    n_ambiguous_atac: int = 0

    @property
    def rna_coverage(self) -> float:
        return self.n_pairs / max(1, self.n_rna)

    @property
    def atac_coverage(self) -> float:
        return self.n_pairs / max(1, self.n_atac)


def _normalise_barcode(value: object) -> str:
    text = str(value)
    return text.split("::", 1)[-1] if "::" in text else text


def _prepare(frame: pd.DataFrame, sample_col: str, cell_col: str, row_col: str, prefix: str) -> pd.DataFrame:
    required = [cell_col, row_col]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"{prefix} table missing columns: {missing}")
    out = frame.copy()
    if sample_col not in out.columns:
        out[sample_col] = ""
    out[sample_col] = out[sample_col].astype(str).str.strip()
    out[cell_col] = out[cell_col].map(_normalise_barcode)
    out[row_col] = pd.to_numeric(out[row_col], errors="raise").astype(np.int64)
    return out[[sample_col, cell_col, row_col]].rename(
        columns={
            sample_col: f"{prefix}_sample_id",
            cell_col: f"{prefix}_cell_id",
            row_col: f"{prefix}_row",
        }
    )


def pair_cells(
    rna: pd.DataFrame,
    atac: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    cell_col: str = "cell_id",
    row_col: str = "row_index",
    strategy: Literal["sample_barcode", "barcode", "row_index", "auto"] = "auto",
    strict: bool = True,
) -> PairingResult:
    """Pair RNA and ATAC cells with explicit ambiguity checks.

    ``auto`` tries sample+barcode first, then barcode-only, and never falls back to
    row-index unless both tables have equal length and row indices are unique.
    """
    rna_p = _prepare(rna, sample_col, cell_col, row_col, "rna")
    atac_p = _prepare(atac, sample_col, cell_col, row_col, "atac")

    strategies = [strategy] if strategy != "auto" else ["sample_barcode", "barcode", "row_index"]
    last_error: Exception | None = None
    for current in strategies:
        try:
            if current == "sample_barcode":
                left_keys = ["rna_sample_id", "rna_cell_id"]
                right_keys = ["atac_sample_id", "atac_cell_id"]
            elif current == "barcode":
                left_keys = ["rna_cell_id"]
                right_keys = ["atac_cell_id"]
            elif current == "row_index":
                if len(rna_p) != len(atac_p):
                    raise ValueError("row-index pairing requires equal RNA and ATAC row counts")
                pairs = pd.DataFrame(
                    {
                        "rna_sample_id": rna_p["rna_sample_id"].to_numpy(),
                        "atac_sample_id": atac_p["atac_sample_id"].to_numpy(),
                        "rna_cell_id": rna_p["rna_cell_id"].to_numpy(),
                        "atac_cell_id": atac_p["atac_cell_id"].to_numpy(),
                        "rna_row": rna_p["rna_row"].to_numpy(),
                        "atac_row": atac_p["atac_row"].to_numpy(),
                    }
                )
                pairs["pairing_strategy"] = current
                return PairingResult(pairs, current, len(rna_p), len(atac_p), len(pairs))
            else:
                raise ValueError(f"Unknown pairing strategy: {current}")

            n_amb_rna = int(rna_p.duplicated(left_keys, keep=False).sum())
            n_amb_atac = int(atac_p.duplicated(right_keys, keep=False).sum())
            if strict and (n_amb_rna or n_amb_atac):
                raise ValueError(
                    f"Ambiguous {current} keys: RNA={n_amb_rna}, ATAC={n_amb_atac}"
                )
            left = rna_p.drop_duplicates(left_keys, keep="first")
            right = atac_p.drop_duplicates(right_keys, keep="first")
            pairs = left.merge(
                right,
                left_on=left_keys,
                right_on=right_keys,
                how="inner",
                validate="one_to_one",
            )
            if pairs.empty:
                raise ValueError(f"No cell pairs found using {current}")
            pairs["pairing_strategy"] = current
            return PairingResult(
                pairs.reset_index(drop=True), current, len(rna_p), len(atac_p), len(pairs),
                n_ambiguous_rna=n_amb_rna, n_ambiguous_atac=n_amb_atac,
            )
        except Exception as exc:  # try next strategy in auto mode
            last_error = exc
    raise ValueError(f"Unable to pair RNA and ATAC cells: {last_error}")
