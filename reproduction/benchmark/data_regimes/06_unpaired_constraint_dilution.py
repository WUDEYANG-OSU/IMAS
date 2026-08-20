# -*- coding: utf-8 -*-
"""
Mechanistic explanation of WHY RNA-only unpaired information weakens dependent structure.

Design philosophy
-----------------
This script does NOT only show the phenomenon (DepStruct goes down).
Instead, it explicitly visualizes the mechanism:

1) As RNA-only cells are added to a multiomic backbone, the fraction of cells carrying
   same-cell RE/ATAC constraints decreases.
2) Therefore, the density of complete RNA–TF–RE closure opportunities decreases.
3) The decrease in closure opportunity explains the fall in RE traceability,
   dependency-core support, and final DepStruct support.

The script force-runs a counterfactual dilution analysis using:
    paired backbone = multiomic-only
    unpaired pool   = scRNA-only

This guarantees that it can run from the posthoc score files alone.

Expected inputs:
    <OUT_ROOT>/<dataset>_multiomic_only_cell_structure_scores_with_DepStruct.csv
    <OUT_ROOT>/<dataset>_scrna_only_cell_structure_scores_with_DepStruct.csv
    <OUT_ROOT>/<dataset>_mixed_cell_structure_scores_with_DepStruct.csv   (optional but recommended for endpoint overlay)

Main outputs:
    WHY1_constraint_density_dilution_3datasets.png/pdf/svg
    WHY2_closure_component_dilution_3datasets.png/pdf/svg
    WHY3_closure_explains_depstruct_3datasets.png/pdf/svg
    why_unpaired_mechanism_summary_3datasets.csv
    why_unpaired_mechanism_endpoints_3datasets.csv
"""

from __future__ import annotations

from pathlib import Path
import warnings
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

try:
    from scipy.stats import spearmanr, linregress
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.stats import spearmanr, linregress

# ============================================================
# 0. CONFIG
# ============================================================
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
REGIMES = ["multiomic_only", "mixed", "scrna_only"]
REGIME_LABEL = {
    "multiomic_only": "Multiomic-only",
    "mixed": "Mixed",
    "scrna_only": "scRNA-only",
}
REQUIRE_ALL_DATASETS = True

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

DOSE_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
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

# ============================================================
# 1. HELPERS
# ============================================================
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
    # require multiomic_only + scrna_only; mixed optional for endpoint overlay
    multi_prefixes = set(get_available_prefixes(out_root, "multiomic_only"))
    scrna_prefixes = set(get_available_prefixes(out_root, "scrna_only"))
    mixed_prefixes = set(get_available_prefixes(out_root, "mixed"))
    common_min = sorted(multi_prefixes & scrna_prefixes)
    print("[AVAILABLE PREFIXES WITH multiomic_only + scrna_only]")
    for x in common_min:
        print("  -", x)

    resolved, missing = [], []
    for req in REQUESTED_DATASETS:
        aliases = DATASET_ALIASES.get(req, [req])
        hit = None
        for a in aliases:
            if a in common_min:
                hit = a
                break
        if hit is None:
            req_low = req.lower().replace("-", "").replace("_", "")
            for p in common_min:
                p_low = p.lower().replace("-", "").replace("_", "")
                if req_low in p_low or p_low in req_low:
                    hit = p
                    break
        if hit is None and "osteo" in req.lower():
            for p in common_min:
                if "osteo" in p.lower():
                    hit = p
                    break
        if hit is None:
            missing.append(req)
        else:
            resolved.append((req, hit, hit in mixed_prefixes))

    if missing and REQUIRE_ALL_DATASETS:
        raise FileNotFoundError(
            "Missing dataset(s): " + ", ".join(missing) +
            "\nNeed at least multiomic_only + scrna_only score files."
        )
    if missing:
        print("[WARN] Missing datasets skipped:", missing)
    return resolved


def load_table(out_root: Path, prefix: str, requested_name: str, regime: str) -> pd.DataFrame:
    f = out_root / f"{prefix}_{regime}_cell_structure_scores_with_DepStruct.csv"
    if not f.exists():
        raise FileNotFoundError(f)
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

# ============================================================
# 2. LOAD DATA
# ============================================================
OUT_ROOT = detect_out_root()
RESOLVED = resolve_dataset_prefixes(OUT_ROOT)
dataset_order = [x[0] for x in RESOLVED]

rng = np.random.default_rng(SEED)
dose_rows = []
endpoint_rows = []

for requested_name, prefix, has_mixed in RESOLVED:
    multi_df = load_table(OUT_ROOT, prefix, requested_name, "multiomic_only")
    scrna_df = load_table(OUT_ROOT, prefix, requested_name, "scrna_only")
    mixed_df = load_table(OUT_ROOT, prefix, requested_name, "mixed") if has_mixed else None

    # endpoint summaries for observed regimes
    for reg, df in [("multiomic_only", multi_df), ("scrna_only", scrna_df)] + ([("mixed", mixed_df)] if mixed_df is not None else []):
        for metric, label, _ in METRICS:
            vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
            endpoint_rows.append({
                "dataset": requested_name,
                "dataset_display": DATASET_DISPLAY.get(requested_name, requested_name),
                "dataset_prefix": prefix,
                "data_regime": reg,
                "metric": metric,
                "metric_label": label,
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "n_cells": int(len(vals)),
            })

    # forced counterfactual mechanism analysis
    paired_pool = multi_df.copy().reset_index(drop=True)
    unpaired_pool = scrna_df.copy().reset_index(drop=True)

    n_p = len(paired_pool)
    n_u_max = len(unpaired_pool)
    print(f"[COUNTERFACTUAL] {requested_name}: paired={n_p}, unpaired_pool={n_u_max}")

    for ratio in DOSE_RATIOS:
        n_u = min(n_u_max, int(round(ratio * n_p)))
        for b in range(N_BOOT):
            if n_u == 0:
                boot = paired_pool.copy()
            else:
                pick = rng.choice(len(unpaired_pool), size=n_u, replace=(n_u > len(unpaired_pool)))
                boot = pd.concat([paired_pool, unpaired_pool.iloc[pick].copy()], axis=0, ignore_index=True)

            # "why" variables: not just the final scores
            total_n = len(boot)
            paired_fraction = n_p / total_n
            unpaired_fraction = n_u / total_n
            rna_observed_fraction = 1.0                     # every cell has RNA
            tf_proxy_fraction = 1.0                         # every cell contributes RNA/TF proxy side
            samecell_re_constraint_fraction = paired_fraction   # only paired cells carry same-cell RE/ATAC evidence
            open_path_burden = unpaired_fraction                # fraction of cells lacking same-cell RE closure
            closure_density_theoretical = paired_fraction       # theoretical closure opportunity density

            dep_vals = pd.to_numeric(boot["depstruct_score_cell"], errors="coerce").dropna().values
            re_vals = pd.to_numeric(boot["re_traceability_score_global"], errors="coerce").dropna().values
            core_vals = pd.to_numeric(boot["dependency_core_score_global"], errors="coerce").dropna().values
            supp_vals = pd.to_numeric(boot["dependency_support_score_global"], errors="coerce").dropna().values
            anch_vals = pd.to_numeric(boot["response_anchor_score_global"], errors="coerce").dropna().values

            # empirically estimated closure proxy: the average of the specific closure-related components
            closure_proxy_empirical = float(np.nanmean([
                np.mean(re_vals) if len(re_vals) else np.nan,
                np.mean(core_vals) if len(core_vals) else np.nan,
                np.mean(supp_vals) if len(supp_vals) else np.nan,
            ]))

            dose_rows.append({
                "dataset": requested_name,
                "dataset_display": DATASET_DISPLAY.get(requested_name, requested_name),
                "dataset_prefix": prefix,
                "unpaired_to_paired_ratio": ratio,
                "bootstrap": b,
                "n_paired_backbone": int(n_p),
                "n_unpaired_added": int(n_u),
                "n_total": int(total_n),
                "rna_observed_fraction": rna_observed_fraction,
                "tf_proxy_fraction": tf_proxy_fraction,
                "samecell_re_constraint_fraction": samecell_re_constraint_fraction,
                "open_path_burden": open_path_burden,
                "closure_density_theoretical": closure_density_theoretical,
                "closure_proxy_empirical": closure_proxy_empirical,
                "depstruct_mean": float(np.mean(dep_vals)) if len(dep_vals) else np.nan,
                "depstruct_high_fraction_075": float(np.mean(dep_vals >= 0.75)) if len(dep_vals) else np.nan,
                "re_traceability_mean": float(np.mean(re_vals)) if len(re_vals) else np.nan,
                "dependency_core_mean": float(np.mean(core_vals)) if len(core_vals) else np.nan,
                "dependency_support_mean": float(np.mean(supp_vals)) if len(supp_vals) else np.nan,
                "response_anchor_mean": float(np.mean(anch_vals)) if len(anch_vals) else np.nan,
            })

boot_df = pd.DataFrame(dose_rows)
endpoint_df = pd.DataFrame(endpoint_rows)
endpoint_out = OUT_ROOT / "why_unpaired_mechanism_endpoints_3datasets.csv"
endpoint_df.to_csv(endpoint_out, index=False)
print("[SAVE]", endpoint_out)

# summarize bootstraps
agg_rows = []
for (ds, ratio), g in boot_df.groupby(["dataset", "unpaired_to_paired_ratio"], as_index=False):
    row = {
        "dataset": ds,
        "dataset_display": g["dataset_display"].iloc[0],
        "dataset_prefix": g["dataset_prefix"].iloc[0],
        "unpaired_to_paired_ratio": ratio,
        "n_boot": int(len(g)),
    }
    for col in [
        "rna_observed_fraction", "tf_proxy_fraction", "samecell_re_constraint_fraction",
        "open_path_burden", "closure_density_theoretical", "closure_proxy_empirical",
        "depstruct_mean", "depstruct_high_fraction_075", "re_traceability_mean",
        "dependency_core_mean", "dependency_support_mean", "response_anchor_mean"
    ]:
        m, lo, hi = ci95(g[col].values)
        row[col] = m
        row[col + "_lo"] = lo
        row[col + "_hi"] = hi
    agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows)
agg_out = OUT_ROOT / "why_unpaired_mechanism_summary_3datasets.csv"
agg_df.to_csv(agg_out, index=False)
print("[SAVE]", agg_out)

# ============================================================
# 3. FIGURE WHY1: constraint density dilution
# ============================================================
fig, axes = plt.subplots(len(dataset_order), 1, figsize=(8.2, 2.7 * len(dataset_order)), facecolor="white", squeeze=False)
axes = axes[:, 0]
for ax, ds in zip(axes, dataset_order):
    sub = agg_df[agg_df["dataset"] == ds].copy().sort_values("unpaired_to_paired_ratio")
    x = sub["unpaired_to_paired_ratio"].values
    ax.plot(x, sub["rna_observed_fraction"].values, color="#636363", lw=1.8, label="RNA-observed fraction")
    ax.plot(x, sub["tf_proxy_fraction"].values, color="#969696", lw=1.8, ls="--", label="TF-proxy-supported fraction")
    ax.plot(x, sub["samecell_re_constraint_fraction"].values, color="#3182bd", lw=2.3, label="Same-cell RE/ATAC-constrained fraction")
    ax.plot(x, sub["open_path_burden"].values, color="#de2d26", lw=2.0, label="Open-path burden (RNA-only fraction)")
    ax.fill_between(x, sub["samecell_re_constraint_fraction_lo"].values, sub["samecell_re_constraint_fraction_hi"].values,
                    color="#9ecae1", alpha=0.18)
    ax.fill_between(x, sub["open_path_burden_lo"].values, sub["open_path_burden_hi"].values,
                    color="#fcbba1", alpha=0.18)
    ax.set_title(f"{DATASET_DISPLAY.get(ds, ds)} | why RNA-only dilution weakens closure density", fontweight="bold", fontsize=10)
    ax.set_xlabel("RNA-only dilution ratio")
    ax.set_ylabel("Fraction / burden")
    ax.set_xlim(min(DOSE_RATIOS) - 0.02, max(DOSE_RATIOS) + 0.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.5)

handles = [
    Line2D([0], [0], color="#636363", lw=1.8, label="RNA-observed fraction"),
    Line2D([0], [0], color="#969696", lw=1.8, ls="--", label="TF-proxy-supported fraction"),
    Line2D([0], [0], color="#3182bd", lw=2.3, label="Same-cell RE/ATAC-constrained fraction"),
    Line2D([0], [0], color="#de2d26", lw=2.0, label="Open-path burden (RNA-only fraction)"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("WHY1 | Unpaired RNA-only cells dilute the density of same-cell multiomic constraints", fontweight="bold", y=1.05)
savefig_multi(fig, OUT_ROOT / "WHY1_constraint_density_dilution_3datasets")

# ============================================================
# 4. FIGURE WHY2: closure components fall with dilution
# ============================================================
fig, axes = plt.subplots(len(dataset_order), 2, figsize=(12.6, 3.0 * len(dataset_order)), facecolor="white", squeeze=False)
for i, ds in enumerate(dataset_order):
    sub = agg_df[agg_df["dataset"] == ds].copy().sort_values("unpaired_to_paired_ratio")
    x = sub["unpaired_to_paired_ratio"].values

    ax = axes[i, 0]
    series = [
        ("closure_proxy_empirical", "Empirical closure proxy", "#1b9e77"),
        ("re_traceability_mean", "RE traceability", "#3182bd"),
        ("dependency_core_mean", "Dependency core", "#756bb1"),
        ("dependency_support_mean", "Dependency support", "#2ca25f"),
        ("response_anchor_mean", "Response anchor", "#d95f0e"),
    ]
    for col, label, color in series:
        ax.plot(x, sub[col].values, color=color, lw=2.0, label=label)
        ax.fill_between(x, sub[col + "_lo"].values, sub[col + "_hi"].values, color=color, alpha=0.12)
    if i == 0:
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.set_title(f"{DATASET_DISPLAY.get(ds, ds)} | closure-related components", fontweight="bold", fontsize=10)
    ax.set_xlabel("RNA-only dilution ratio")
    ax.set_ylabel("Mean value")
    ax.set_xlim(min(DOSE_RATIOS) - 0.02, max(DOSE_RATIOS) + 0.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.5)

    ax = axes[i, 1]
    ax.plot(x, sub["depstruct_mean"].values, color="#222222", lw=2.2, label="Mean DepStruct")
    ax.fill_between(x, sub["depstruct_mean_lo"].values, sub["depstruct_mean_hi"].values, color="#999999", alpha=0.2)
    ax.plot(x, sub["depstruct_high_fraction_075"].values, color="#000000", lw=1.7, ls="--", label="High-support cell fraction")
    ax.fill_between(x, sub["depstruct_high_fraction_075_lo"].values, sub["depstruct_high_fraction_075_hi"].values, color="#bdbdbd", alpha=0.18)
    try:
        rho, p = spearmanr(x, sub["depstruct_mean"].values)
        ptxt = "p < 1e-16" if np.isfinite(p) and p < 1e-16 else (f"p={p:.2e}" if np.isfinite(p) else "p=n.a.")
        ax.text(0.98, 0.98, f"Spearman r={rho:.2f}\n{ptxt}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.95))
    except Exception:
        pass
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{DATASET_DISPLAY.get(ds, ds)} | final DepStruct support", fontweight="bold", fontsize=10)
    ax.set_xlabel("RNA-only dilution ratio")
    ax.set_ylabel("Support")
    ax.set_xlim(min(DOSE_RATIOS) - 0.02, max(DOSE_RATIOS) + 0.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.5)

fig.suptitle("WHY2 | Reduced closure density selectively weakens RE traceability, dependency core and final DepStruct support", fontweight="bold", y=1.03)
savefig_multi(fig, OUT_ROOT / "WHY2_closure_component_dilution_3datasets")

# ============================================================
# 5. FIGURE WHY3: closure explains DepStruct + observed endpoints
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), facecolor="white")

# left: closure proxy vs depstruct under dilution
ax = axes[0]
color_map = {"CRC-Tissue": "#1f77b4", "Glioma-Tissue": "#2ca02c", "Osteosarcoma": "#d62728"}
for ds in dataset_order:
    sub = agg_df[agg_df["dataset"] == ds].copy().sort_values("unpaired_to_paired_ratio")
    x = sub["closure_proxy_empirical"].values
    y = sub["depstruct_mean"].values
    ax.plot(x, y, marker="o", lw=1.8, ms=4.5, color=color_map.get(ds, "#333333"), label=DATASET_DISPLAY.get(ds, ds))
    for xi, yi, rr in zip(x, y, sub["unpaired_to_paired_ratio"].values):
        ax.text(xi + 0.008, yi, f"{rr:g}", fontsize=7, alpha=0.75)
# pooled regression
x_all = agg_df["closure_proxy_empirical"].values
y_all = agg_df["depstruct_mean"].values
mask = np.isfinite(x_all) & np.isfinite(y_all)
if mask.sum() >= 2:
    lr = linregress(x_all[mask], y_all[mask])
    xs = np.linspace(np.nanmin(x_all[mask]), np.nanmax(x_all[mask]), 100)
    ys = lr.intercept + lr.slope * xs
    ax.plot(xs, ys, color="#444444", ls="--", lw=1.5)
    rho, p = spearmanr(x_all[mask], y_all[mask])
    ptxt = "p < 1e-16" if np.isfinite(p) and p < 1e-16 else (f"p={p:.2e}" if np.isfinite(p) else "p=n.a.")
    ax.text(0.03, 0.97, f"Pooled Spearman r={rho:.2f}\n{ptxt}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.95))
ax.set_xlabel("Empirical closure proxy")
ax.set_ylabel("Mean DepStruct")
ax.set_title("Counterfactual dilution trajectory", fontweight="bold")
ax.grid(alpha=0.18, linewidth=0.5)
ax.legend(frameon=False, loc="lower right")

# right: observed endpoint overlay using closure-relevant metrics
ax = axes[1]
for ds in dataset_order:
    sub = endpoint_df[(endpoint_df["dataset"] == ds) & (endpoint_df["metric"].isin(["depstruct_score_cell", "re_traceability_score_global", "dependency_core_score_global"]))].copy()
    if len(sub) == 0:
        continue
    # build a simple triangle summary: x=observed RE traceability, y=observed DepStruct, marker size by dependency core
    tmp = sub.pivot_table(index="data_regime", columns="metric", values="mean", aggfunc="mean")
    for reg in [r for r in ["multiomic_only", "mixed", "scrna_only"] if r in tmp.index]:
        xv = float(tmp.loc[reg, "re_traceability_score_global"]) if "re_traceability_score_global" in tmp.columns else np.nan
        yv = float(tmp.loc[reg, "depstruct_score_cell"]) if "depstruct_score_cell" in tmp.columns else np.nan
        sv = float(tmp.loc[reg, "dependency_core_score_global"]) if "dependency_core_score_global" in tmp.columns else np.nan
        if np.isfinite(xv) and np.isfinite(yv):
            ax.scatter(xv, yv, s=50 + 250 * max(0, sv), alpha=0.85, label=f"{DATASET_DISPLAY.get(ds, ds)} | {REGIME_LABEL.get(reg, reg)}")
            ax.text(xv + 0.008, yv + 0.008, f"{DATASET_DISPLAY.get(ds, ds)}\n{REGIME_LABEL.get(reg, reg)}", fontsize=7)
ax.set_xlabel("Observed RE traceability")
ax.set_ylabel("Observed DepStruct")
ax.set_title("Observed regime endpoints", fontweight="bold")
ax.grid(alpha=0.18, linewidth=0.5)

fig.suptitle("WHY3 | DepStruct falls because closure opportunity falls", fontweight="bold", y=1.03)
savefig_multi(fig, OUT_ROOT / "WHY3_closure_explains_depstruct_3datasets")

print("\n[DONE]")
print("[FIG WHY1] WHY1_constraint_density_dilution_3datasets.png/pdf/svg")
print("[FIG WHY2] WHY2_closure_component_dilution_3datasets.png/pdf/svg")
print("[FIG WHY3] WHY3_closure_explains_depstruct_3datasets.png/pdf/svg")
print("[SUMMARY]", agg_out)
print("[ENDPOINTS]", endpoint_out)
