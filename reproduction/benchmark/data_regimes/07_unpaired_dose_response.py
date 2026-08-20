# -*- coding: utf-8 -*-
"""
Force-generate FIG1 for CRC, Glioma, and Osteosarcoma.

This version does NOT depend on whether mixed can be split into paired/unpaired.
It always uses a counterfactual dilution design:

    paired backbone = multiomic-only
    unpaired pool   = scRNA-only

Then it progressively adds RNA-only cells to the multiomic backbone.
So FIG1 will be generated as long as each dataset has:
    - <dataset>_multiomic_only_cell_structure_scores_with_DepStruct.csv
    - <dataset>_scrna_only_cell_structure_scores_with_DepStruct.csv

Outputs:
    FIG1_unpaired_dose_response_3datasets_force.png/pdf/svg
    FIG1_unpaired_dose_response_3datasets_force_summary.csv
    FIG1_unpaired_dose_response_3datasets_force_diagnostics.csv
"""

from __future__ import annotations
from pathlib import Path
import warnings
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from scipy.stats import spearmanr
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.stats import spearmanr

OUT_ROOT_CANDIDATES = [
    get_runs_root() / "frontier_transfer_hash_stageB_data_regime_comparison" / "benchmark_umap_dependency_structure_posthoc",
    Path.cwd(),
]

REQUESTED_DATASETS = ["CRC-Tissue", "Glioma-Tissue", "Osteosarcoma"]
DATASET_ALIASES = {
    "CRC-Tissue": ["CRC-Tissue", "CRC"],
    "Glioma-Tissue": ["Glioma-Tissue", "Glioma"],
    "Osteosarcoma": [
        "Osteosarcoma", "Osteosarcoma-Tissue", "Osteosarcoma-Cellline",
        "Osteosarcoma-CellLine", "Osteosarcoma_Cellline", "Osteosarcoma_Tissue"
    ],
}
DATASET_DISPLAY = {
    "CRC-Tissue": "CRC",
    "Glioma-Tissue": "Glioma",
    "Osteosarcoma": "Osteosarcoma",
}
METRICS = [
    ("depstruct_score_cell", "DepStruct", "#222222"),
    ("dependency_support_score_global", "Dependency support", "#2ca25f"),
    ("re_traceability_score_global", "RE traceability", "#3182bd"),
    ("dependency_core_score_global", "Dependency core", "#756bb1"),
    ("response_anchor_score_global", "Response anchor", "#d95f0e"),
]
METRIC_ALIASES = {
    "depstruct_score_cell": ["depstruct_score_cell", "DepStruct", "depstruct_score"],
    "dependency_support_score_global": ["dependency_support_score_global", "dependency_support_score"],
    "re_traceability_score_global": ["re_traceability_score_global", "re_traceability_score"],
    "dependency_core_score_global": ["dependency_core_score_global", "dependency_core_score"],
    "response_anchor_score_global": ["response_anchor_score_global", "response_anchor_score"],
}
DOSE_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
N_BOOT = 80
SEED = 42

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def detect_out_root() -> Path:
    for p in OUT_ROOT_CANDIDATES:
        if p.exists() and len(list(p.glob("*_cell_structure_scores_with_DepStruct.csv"))) > 0:
            print("[OUT_ROOT]", p)
            return p
    raise FileNotFoundError("Cannot detect OUT_ROOT automatically.")


def savefig_multi(fig, stem: Path):
    for ext in ["png", "pdf", "svg"]:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=400 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("[SAVE]", out)
    plt.close(fig)


def pick_col(df: pd.DataFrame, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in low:
            return low[a.lower()]
    return None


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def get_available_prefixes(out_root: Path, regime: str):
    suffix = f"_{regime}_cell_structure_scores_with_DepStruct.csv"
    return sorted([f.name[:-len(suffix)] for f in out_root.glob(f"*{suffix}")])


def resolve_dataset_prefixes(out_root: Path):
    multi_prefixes = set(get_available_prefixes(out_root, "multiomic_only"))
    scrna_prefixes = set(get_available_prefixes(out_root, "scrna_only"))
    common = sorted(multi_prefixes & scrna_prefixes)
    print("[AVAILABLE PREFIXES WITH multiomic_only + scrna_only]")
    for x in common:
        print("  -", x)

    resolved, missing = [], []
    for req in REQUESTED_DATASETS:
        aliases = DATASET_ALIASES.get(req, [req])
        hit = None
        for a in aliases:
            if a in common:
                hit = a
                break
        if hit is None:
            req_low = req.lower().replace("-", "").replace("_", "")
            for p in common:
                p_low = p.lower().replace("-", "").replace("_", "")
                if req_low in p_low or p_low in req_low:
                    hit = p
                    break
        if hit is None and "osteo" in req.lower():
            for p in common:
                if "osteo" in p.lower():
                    hit = p
                    break
        if hit is None:
            missing.append(req)
        else:
            resolved.append((req, hit))

    if missing:
        raise FileNotFoundError(
            "Missing dataset(s) for FIG1: " + ", ".join(missing) +
            "\nNeed at least multiomic_only + scrna_only score files.\nAvailable prefixes:\n  " +
            "\n  ".join(common)
        )
    return resolved


def load_table(out_root: Path, prefix: str, requested_name: str, regime: str) -> pd.DataFrame:
    f = out_root / f"{prefix}_{regime}_cell_structure_scores_with_DepStruct.csv"
    df = pd.read_csv(f)
    out = pd.DataFrame()
    out["dataset"] = requested_name
    out["dataset_prefix"] = prefix
    out["data_regime"] = regime
    for metric, _, _ in METRICS:
        c = pick_col(df, METRIC_ALIASES[metric])
        if c is None:
            raise ValueError(f"Cannot find {metric} in {f.name}")
        out[metric] = pd.to_numeric(df[c], errors="coerce").clip(lower=0.0, upper=1.0)
    return out.replace([np.inf, -np.inf], np.nan)


OUT_ROOT = detect_out_root()
RESOLVED = resolve_dataset_prefixes(OUT_ROOT)
dataset_order = [x[0] for x in RESOLVED]

rng = np.random.default_rng(SEED)
summary_rows = []
diag_rows = []

for requested_name, prefix in RESOLVED:
    multi_df = load_table(OUT_ROOT, prefix, requested_name, "multiomic_only")
    scrna_df = load_table(OUT_ROOT, prefix, requested_name, "scrna_only")

    paired_pool = multi_df.copy().reset_index(drop=True)
    unpaired_pool = scrna_df.copy().reset_index(drop=True)

    diag_rows.append({
        "dataset": requested_name,
        "dataset_display": DATASET_DISPLAY.get(requested_name, requested_name),
        "dataset_prefix": prefix,
        "source_mode": "forced_counterfactual_multiomic_backbone_plus_scrna_unpaired_pool",
        "n_paired_pool": len(paired_pool),
        "n_unpaired_pool": len(unpaired_pool),
    })
    print(f"[SOURCE] {requested_name}: paired_pool={len(paired_pool)}; unpaired_pool={len(unpaired_pool)}")

    n_p = len(paired_pool)
    if n_p == 0 or len(unpaired_pool) == 0:
        raise RuntimeError(f"Empty paired or unpaired pool for {requested_name}")

    for ratio in DOSE_RATIOS:
        n_u = min(len(unpaired_pool), int(round(ratio * n_p)))
        for b in range(N_BOOT):
            if n_u == 0:
                boot = paired_pool.copy()
            else:
                pick = rng.choice(len(unpaired_pool), size=n_u, replace=(n_u > len(unpaired_pool)))
                boot = pd.concat([paired_pool, unpaired_pool.iloc[pick].copy()], axis=0, ignore_index=True)
            for metric, label, _ in METRICS:
                vals = pd.to_numeric(boot[metric], errors="coerce").dropna().values
                if len(vals) == 0:
                    continue
                summary_rows.append({
                    "dataset": requested_name,
                    "dataset_display": DATASET_DISPLAY.get(requested_name, requested_name),
                    "dataset_prefix": prefix,
                    "source_mode": "forced_counterfactual_multiomic_backbone_plus_scrna_unpaired_pool",
                    "unpaired_to_paired_ratio": ratio,
                    "bootstrap": b,
                    "metric": metric,
                    "metric_label": label,
                    "mean_value": float(np.mean(vals)),
                    "high_fraction_075": float(np.mean(vals >= 0.75)),
                    "n_cells": int(len(boot)),
                    "n_paired_backbone": int(n_p),
                    "n_unpaired_added": int(n_u),
                })

diag_df = pd.DataFrame(diag_rows)
diag_out = OUT_ROOT / "FIG1_unpaired_dose_response_3datasets_force_diagnostics.csv"
diag_df.to_csv(diag_out, index=False)
print("[SAVE]", diag_out)

boot_df = pd.DataFrame(summary_rows)
agg_rows = []
for (ds, metric, ratio), g in boot_df.groupby(["dataset", "metric", "unpaired_to_paired_ratio"], as_index=False):
    m, lo, hi = ci95(g["mean_value"].values)
    hf, hflo, hfhi = ci95(g["high_fraction_075"].values)
    agg_rows.append({
        "dataset": ds,
        "dataset_display": g["dataset_display"].iloc[0],
        "dataset_prefix": g["dataset_prefix"].iloc[0],
        "metric": metric,
        "metric_label": g["metric_label"].iloc[0],
        "source_mode": g["source_mode"].iloc[0],
        "unpaired_to_paired_ratio": ratio,
        "mean": m,
        "mean_lo": lo,
        "mean_hi": hi,
        "high_fraction_075": hf,
        "high_fraction_075_lo": hflo,
        "high_fraction_075_hi": hfhi,
        "n_boot": len(g),
    })
agg_df = pd.DataFrame(agg_rows)
agg_out = OUT_ROOT / "FIG1_unpaired_dose_response_3datasets_force_summary.csv"
agg_df.to_csv(agg_out, index=False)
print("[SAVE]", agg_out)

fig, axes = plt.subplots(len(dataset_order), 2, figsize=(12.8, 3.0 * len(dataset_order)), facecolor="white", squeeze=False)
for i, ds in enumerate(dataset_order):
    sub = agg_df[agg_df["dataset"] == ds].copy()

    ax = axes[i, 0]
    for metric, label, color in METRICS:
        ss = sub[sub["metric"] == metric].sort_values("unpaired_to_paired_ratio")
        if len(ss) == 0:
            continue
        x = ss["unpaired_to_paired_ratio"].values
        y = ss["mean"].values
        lo = ss["mean_lo"].values
        hi = ss["mean_hi"].values
        ax.plot(x, y, color=color, lw=2.0, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    if i == 0:
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    ax.set_title(f"{DATASET_DISPLAY.get(ds, ds)} | component means", fontweight="bold", fontsize=10)
    ax.set_xlabel("RNA-only dilution ratio")
    ax.set_ylabel("Mean component value")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.text(0.02, 0.98, "forced counterfactual", transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax = axes[i, 1]
    ss = sub[sub["metric"] == "depstruct_score_cell"].sort_values("unpaired_to_paired_ratio")
    x = ss["unpaired_to_paired_ratio"].values
    y = ss["high_fraction_075"].values
    lo = ss["high_fraction_075_lo"].values
    hi = ss["high_fraction_075_hi"].values
    ax.plot(x, y, color="#222222", lw=2.2)
    ax.fill_between(x, lo, hi, color="#777777", alpha=0.22)
    try:
        rho, p = spearmanr(x, y)
        ptxt = "p < 1e-16" if p < 1e-16 else f"p={p:.2e}"
        ax.text(0.98, 0.98, f"Spearman r={rho:.2f}\n{ptxt}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.95))
    except Exception:
        pass
    ax.set_title(f"{DATASET_DISPLAY.get(ds, ds)} | high-support fraction", fontweight="bold", fontsize=10)
    ax.set_xlabel("RNA-only dilution ratio")
    ax.set_ylabel("Fraction with DepStruct ≥ 0.75")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.5)

fig.suptitle("Figure 1 | Addition of RNA-only unpaired evidence weakens dependent-structure support", fontweight="bold", y=1.02)
savefig_multi(fig, OUT_ROOT / "FIG1_unpaired_dose_response_3datasets_force")

print("\n[DONE]")
print("[FIG1] FIG1_unpaired_dose_response_3datasets_force.png/pdf/svg")
print("[SUMMARY]", agg_out)
print("[DIAGNOSTICS]", diag_out)
