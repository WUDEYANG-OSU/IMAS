# -*- coding: utf-8 -*-
"""
HNSCC LN_all virtual KO baseline comparison

Purpose
-------
Compare the Stage5 / Full-IMAS virtual perturbation gene scores with simple
baselines computed directly from the HNSCC LN_all expression matrix:

1) Expression-only target-low vs target-high contrast
2) Correlation-only target-gene correlation baseline
3) Residualized linear baseline, gene ~ target + covariates

Important
---------
This script does NOT claim external perturbation validation, because the HNSCC
LN_all dataset is observational and does not contain real KO/treatment labels.
It is intended to test whether Stage5 response scores are trivially reproduced
by expression/correlation/linear baselines.

Outputs
-------
All outputs are saved to:
    OUT_DIR = PRED_DIR / "linear_correlation_baseline_comparison"

Main files:
    *_gene_level_baseline_comparison.tsv
    all_targets_model_level_summary.tsv
    all_targets_topN_overlap_summary.tsv
    baseline_score_correlation_bar.svg/png
    baseline_topN_overlap_bar.svg/png
"""

from __future__ import annotations

from pathlib import Path
import re
import json
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================================================
# 0) ABSOLUTE PATH CONFIG
# =========================================================

BASE = get_project_root()

EXPLOIT_HNSCC_ROOT = BASE / "LLM-WorldWord" / "Exploit-HNSCC"
RNA_DIR = EXPLOIT_HNSCC_ROOT / "processed_LN" / "RNA_LN_all"
GENE_FILE = EXPLOIT_HNSCC_ROOT / "assets_LN" / "genes.tsv"

SAMPLE_ID = "LN_all"
RNA_COUNTS = RNA_DIR / f"{SAMPLE_ID}.X_rna_counts_dense.npy"
RNA_META = RNA_DIR / f"{SAMPLE_ID}.rna_meta.npz"
RNA_MANIFEST = RNA_DIR / "manifest_rna.tsv"

RUN_STAGEB_DIR = BASE / "runs" / "frontier_transfer_rna_only_stageB_clusterCorrection_HNSCC_LNall_only"
RUN_STAGEB_DATASET_DIR = RUN_STAGEB_DIR / "HNSCC-LN_all"

# Stage5 code in the uploaded Word used a relative OUT_DIR:
#     Path("./five_stage_network_viz_out_hnscc_lnall")
# Here we convert it into absolute candidates.
PRED_DIR_CANDIDATES = [
    BASE / "five_stage_network_viz_out_hnscc_lnall",
    EXPLOIT_HNSCC_ROOT / "five_stage_network_viz_out_hnscc_lnall",
    RUN_STAGEB_DATASET_DIR / "five_stage_network_viz_out_hnscc_lnall",
    Path.cwd() / "five_stage_network_viz_out_hnscc_lnall",
]

# If you know the exact Stage5 output folder, set it here, for example:
# PRED_DIR_MANUAL = (get_project_root() / "five_stage_network_viz_out_hnscc_lnall")
PRED_DIR_MANUAL = None

# If None, all detected *_stage5_ko_node_state.csv files are used.
# For example: TARGET_GENES = ["SOX2"] or ["TCF7", "TREM2"]
TARGET_GENES = None

# If target name cannot be parsed from file name, use this fallback.
DEFAULT_TARGET_GENE = "SOX2"


# =========================================================
# 1) ANALYSIS PARAMS
# =========================================================

TARGET_SUM = 1e4
GENE_CHUNK = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Target-high/low cells used by expression-only baseline
TARGET_HIGH_Q = 0.75
TARGET_LOW_Q = 0.25
MIN_TARGET_HIGH_CELLS = 20
MIN_TARGET_LOW_CELLS = 20

# If the dataset is very large, subsample cells for baseline computation.
# HNSCC LN_all is usually small enough; keep None by default.
MAX_CELLS_FOR_BASELINE = None

# Stage5 score threshold for directional concordance.
# If your Stage5 deltas are small, reduce this to 0.01.
FULL_IMAS_DELTA_THRESHOLD = 0.03
BASELINE_DELTA_THRESHOLD = 0.0

# Top-N overlap settings
TOPN_LIST = [25, 50, 100, 200, 300, 500]
N_SHUFFLE = 5000

# Residualized linear baseline covariates
USE_COVARIATES = True
MAX_LEVELS_SAMPLE = 30
MAX_LEVELS_CELLTYPE = 30
RIDGE_FOR_COVARIATE_RESIDUAL = 1e-4

# Plotting
DPI = 350


# =========================================================
# 2) SMALL UTILS
# =========================================================

def log(*args):
    print(*args, flush=True)


def safe_name(x: str, n: int = 100) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n].strip("_")


def read_gene_file(gene_file: Path) -> np.ndarray:
    df = pd.read_csv(gene_file, sep="\t")
    if "gene" in df.columns:
        genes = df["gene"].astype(str).to_numpy()
    elif "gene_symbol" in df.columns:
        genes = df["gene_symbol"].astype(str).to_numpy()
    else:
        genes = df.iloc[:, 0].astype(str).to_numpy()
    return genes


def calc_depth(X: np.ndarray) -> np.ndarray:
    depth = np.asarray(X.sum(axis=1), dtype=np.float32)
    depth = np.maximum(depth, 1.0)
    return depth


def log1p_norm_block(block: np.ndarray, depth: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    out = (block / depth[:, None]) * float(TARGET_SUM)
    return np.log1p(out).astype(np.float32)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / (s + 1e-8)).astype(np.float32)


def safe_pearson_vec(Xlog: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xlog = np.asarray(Xlog, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y0 = y - np.nanmean(y)
    yden = np.sqrt(np.nansum(y0 * y0)) + 1e-12
    Xm = Xlog - np.nanmean(Xlog, axis=0, keepdims=True)
    xden = np.sqrt(np.nansum(Xm * Xm, axis=0)) + 1e-12
    corr = np.nansum(Xm * y0[:, None], axis=0) / (xden * yden)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def score_to_sign(x: np.ndarray, thr: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.zeros(x.shape, dtype=np.int8)
    s[x >= float(thr)] = 1
    s[x <= -float(thr)] = -1
    return s


def spearman_fast(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ar = pd.Series(a[m]).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b[m]).rank(method="average").to_numpy(dtype=float)
    ar = ar - ar.mean()
    br = br - br.mean()
    den = np.sqrt((ar * ar).sum()) * np.sqrt((br * br).sum()) + 1e-12
    return float((ar * br).sum() / den)


def pearson_fast(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    den = np.sqrt((aa * aa).sum()) * np.sqrt((bb * bb).sum()) + 1e-12
    return float((aa * bb).sum() / den)


# =========================================================
# 3) PATH / FILE DISCOVERY
# =========================================================

def find_pred_dir_and_files() -> tuple[Path, list[Path]]:
    if PRED_DIR_MANUAL is not None:
        pred_dir = Path(PRED_DIR_MANUAL)
        if not pred_dir.exists():
            raise FileNotFoundError(f"PRED_DIR_MANUAL does not exist: {pred_dir}")
        files = sorted(pred_dir.glob("*_stage5_ko_node_state.csv"))
        if len(files) == 0:
            files = sorted(pred_dir.rglob("*_stage5_ko_node_state.csv"))
        if len(files) == 0:
            raise FileNotFoundError(f"No *_stage5_ko_node_state.csv under: {pred_dir}")
        return pred_dir, files

    for d in PRED_DIR_CANDIDATES:
        if not d.exists():
            continue
        files = sorted(d.glob("*_stage5_ko_node_state.csv"))
        if len(files) == 0:
            files = sorted(d.rglob("*_stage5_ko_node_state.csv"))
        if len(files) > 0:
            return d, files

    # last-resort targeted search, still using absolute roots
    search_roots = [
        EXPLOIT_HNSCC_ROOT,
        RUN_STAGEB_DATASET_DIR,
        BASE,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        log(f"[SEARCH] Looking for *_stage5_ko_node_state.csv under {root}")
        files = sorted(root.rglob("*_stage5_ko_node_state.csv"))
        if len(files) > 0:
            return files[0].parent, files

    msg = ["Cannot find Stage5 KO node_state files.", "Searched candidates:"]
    for d in PRED_DIR_CANDIDATES:
        msg.append(f"  {d} | exists={d.exists()}")
    raise FileNotFoundError("\n".join(msg))


def parse_target_from_stage5_file(fp: Path) -> str:
    stem = fp.stem
    # expected: TCF7_stage5_ko_node_state
    m = re.match(r"^(.+?)_stage5_ko_node_state$", stem)
    if m:
        return m.group(1)
    m = re.match(r"^(.+?)_stage5_.+?_node_state$", stem)
    if m:
        return m.group(1)
    return DEFAULT_TARGET_GENE


def matching_baseline_file(ko_file: Path, target: str) -> Path:
    return ko_file.parent / f"{safe_name(target)}_stage5_baseline_node_state.csv"


# =========================================================
# 4) LOAD EXPRESSION AND METADATA
# =========================================================

def load_meta_table(meta_npz: Path, n_cells: int) -> pd.DataFrame:
    if not meta_npz.exists():
        log(f"[WARN] Missing meta npz: {meta_npz}")
        return pd.DataFrame({"cell_index": np.arange(n_cells)})

    z = np.load(meta_npz, allow_pickle=True)
    if "metadata_columns" in z.files and "metadata_table" in z.files:
        cols = [str(x) for x in z["metadata_columns"].tolist()]
        tab = z["metadata_table"]
        meta = pd.DataFrame(tab, columns=cols)
    else:
        meta = pd.DataFrame({"cell_index": np.arange(n_cells)})

    if len(meta) != n_cells:
        log(f"[WARN] metadata rows={len(meta)} != n_cells={n_cells}; resetting index only")
        meta = meta.iloc[:min(len(meta), n_cells)].copy()

    meta["cell_index"] = np.arange(len(meta))
    return meta


def subset_cells_if_needed(X: np.ndarray, meta: pd.DataFrame):
    n = X.shape[0]
    if MAX_CELLS_FOR_BASELINE is None or n <= int(MAX_CELLS_FOR_BASELINE):
        return np.arange(n, dtype=np.int64), X, meta

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.sort(rng.choice(np.arange(n), size=int(MAX_CELLS_FOR_BASELINE), replace=False))
    return idx, X[idx, :], meta.iloc[idx].reset_index(drop=True).copy()


def make_covariate_matrix(meta: pd.DataFrame, depth: np.ndarray) -> np.ndarray:
    n = len(meta)
    cols = [np.ones((n, 1), dtype=np.float32)]

    log_depth = np.log1p(np.asarray(depth, dtype=np.float32))
    cols.append(zscore_1d(log_depth).reshape(-1, 1))

    if not USE_COVARIATES or len(meta) == 0:
        return np.hstack(cols).astype(np.float32)

    cat_candidates = []
    if "sample.id" in meta.columns:
        cat_candidates.append(("sample.id", MAX_LEVELS_SAMPLE))
    if "cell.type" in meta.columns:
        cat_candidates.append(("cell.type", MAX_LEVELS_CELLTYPE))
    elif "integrated_label" in meta.columns:
        cat_candidates.append(("integrated_label", MAX_LEVELS_CELLTYPE))

    for cname, max_levels in cat_candidates:
        s = meta[cname].astype(str).fillna("NA")
        vc = s.value_counts()
        keep_levels = vc.index.tolist()[:int(max_levels)]
        # drop first level to avoid full collinearity with intercept
        keep_levels = keep_levels[1:]
        for lev in keep_levels:
            cols.append((s.values == lev).astype(np.float32).reshape(-1, 1))

    C = np.hstack(cols).astype(np.float32)
    log(f"[COVARIATES] matrix shape = {C.shape}")
    return C


def residualize_y_and_prepare(C: np.ndarray, y: np.ndarray):
    y = np.asarray(y, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)
    k = C.shape[1]
    CtC = C.T @ C
    CtC = CtC + np.eye(k, dtype=np.float32) * float(RIDGE_FOR_COVARIATE_RESIDUAL)
    inv = np.linalg.pinv(CtC).astype(np.float32)
    beta_y = inv @ C.T @ y
    y_res = y - C @ beta_y
    y_res = y_res.astype(np.float32)
    denom = float(np.dot(y_res, y_res)) + 1e-12
    return inv, y_res, denom


def residualized_linear_slope_chunk(Xlog: np.ndarray, C: np.ndarray, inv: np.ndarray, y_res: np.ndarray, denom: float) -> np.ndarray:
    beta = inv @ C.T @ Xlog
    X_res = Xlog - C @ beta
    slopes = (y_res.reshape(1, -1) @ X_res).reshape(-1) / float(denom)
    return np.nan_to_num(slopes, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =========================================================
# 5) STAGE5 SCORES AND BASELINES
# =========================================================

def load_full_imas_stage5_score(ko_file: Path, target: str) -> pd.DataFrame:
    pert = pd.read_csv(ko_file)
    if "node" not in pert.columns:
        raise KeyError(f"{ko_file} lacks column: node")

    base_file = matching_baseline_file(ko_file, target)
    if base_file.exists():
        base = pd.read_csv(base_file)
        if "node" not in base.columns or "state_final" not in base.columns:
            raise KeyError(f"{base_file} lacks node/state_final")
        if "state_final" not in pert.columns:
            raise KeyError(f"{ko_file} lacks state_final")
        tmp = pert[["node", "state_final"]].copy()
        tmp = tmp.rename(columns={"state_final": "state_final_perturbed"})
        bb = base[["node", "state_final"]].copy()
        bb = bb.rename(columns={"state_final": "state_final_baseline"})
        out = tmp.merge(bb, on="node", how="inner")
        out["gene"] = out["node"].astype(str)
        out["score_full_imas"] = out["state_final_perturbed"] - out["state_final_baseline"]
        out["score_source"] = "perturbed_state_final_minus_baseline_state_final"
        return out[["gene", "score_full_imas", "score_source", "state_final_perturbed", "state_final_baseline"]]

    # fallback if baseline file is unavailable
    out = pert.copy()
    out["gene"] = out["node"].astype(str)
    if "delta_state" in out.columns:
        out["score_full_imas"] = pd.to_numeric(out["delta_state"], errors="coerce")
        out["score_source"] = "delta_state_in_ko_csv"
    elif "state_final" in out.columns and "state_init" in out.columns:
        out["score_full_imas"] = pd.to_numeric(out["state_final"], errors="coerce") - pd.to_numeric(out["state_init"], errors="coerce")
        out["score_source"] = "state_final_minus_state_init_in_ko_csv"
    else:
        raise KeyError(f"Cannot compute Stage5 score from {ko_file}")
    return out[["gene", "score_full_imas", "score_source"]].dropna()


def compute_simple_baselines_for_target(X_counts: np.ndarray, meta: pd.DataFrame, gene_names: np.ndarray, target_gene: str) -> pd.DataFrame:
    gene_names = np.asarray(gene_names).astype(str)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    if target_gene not in gene2idx:
        raise KeyError(f"target gene {target_gene} not found in gene_names")

    target_idx = int(gene2idx[target_gene])
    depth = calc_depth(X_counts)

    target_raw = np.asarray(X_counts[:, target_idx], dtype=np.float32).reshape(-1, 1)
    target_log = log1p_norm_block(target_raw, depth).reshape(-1)
    target_log = np.nan_to_num(target_log, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    lo = float(np.quantile(target_log, TARGET_LOW_Q))
    hi = float(np.quantile(target_log, TARGET_HIGH_Q))
    low_mask = target_log <= lo
    high_mask = target_log >= hi

    if int(low_mask.sum()) < MIN_TARGET_LOW_CELLS or int(high_mask.sum()) < MIN_TARGET_HIGH_CELLS:
        log(f"[WARN] {target_gene}: too few target-low/high cells; using 30/70 quantiles")
        lo = float(np.quantile(target_log, 0.30))
        hi = float(np.quantile(target_log, 0.70))
        low_mask = target_log <= lo
        high_mask = target_log >= hi

    C = make_covariate_matrix(meta, depth)
    inv, target_res, target_res_denom = residualize_y_and_prepare(C, target_log)

    rows = []
    n_genes = len(gene_names)

    log("=" * 100)
    log(f"[BASELINES] target = {target_gene}")
    log(f"[BASELINES] cells = {X_counts.shape[0]} | genes = {n_genes}")
    log(f"[BASELINES] target-low cells = {int(low_mask.sum())} | target-high cells = {int(high_mask.sum())}")

    for st in range(0, n_genes, GENE_CHUNK):
        ed = min(st + GENE_CHUNK, n_genes)
        raw = np.asarray(X_counts[:, st:ed], dtype=np.float32)
        Xlog = log1p_norm_block(raw, depth)

        baseline_mean = np.nanmean(Xlog, axis=0)
        baseline_var = np.nanvar(Xlog, axis=0)
        baseline_pct = (raw > 0).mean(axis=0)

        mean_low = np.nanmean(Xlog[low_mask, :], axis=0)
        mean_high = np.nanmean(Xlog[high_mask, :], axis=0)

        # KO/suppression means moving from target-high toward target-low.
        score_expr_only = mean_low - mean_high

        corr = safe_pearson_vec(Xlog, target_log)
        # If target is knocked down, positively correlated genes are expected to decrease.
        score_corr_only = -corr

        slope = residualized_linear_slope_chunk(Xlog, C, inv, target_res, target_res_denom)
        score_linear_resid = -slope

        for j in range(ed - st):
            gi = st + j
            rows.append({
                "gene": str(gene_names[gi]),
                "baseline_mean_log1p": float(baseline_mean[j]),
                "baseline_var_log1p": float(baseline_var[j]),
                "baseline_pct_expr": float(baseline_pct[j]),
                "target_corr": float(corr[j]),
                "score_expr_only": float(score_expr_only[j]),
                "score_corr_only": float(score_corr_only[j]),
                "score_linear_residualized": float(score_linear_resid[j]),
            })

        log(f"  [BASELINES] genes {st}:{ed} done")

    return pd.DataFrame(rows)


# =========================================================
# 6) COMPARISON METRICS
# =========================================================

def model_level_metrics(df: pd.DataFrame, target_gene: str, model_name: str, score_col: str) -> dict:
    sub = df[["gene", "score_full_imas", score_col]].dropna().copy()
    full = sub["score_full_imas"].to_numpy(float)
    pred = sub[score_col].to_numpy(float)

    full_sign = score_to_sign(full, FULL_IMAS_DELTA_THRESHOLD)
    pred_sign = score_to_sign(pred, BASELINE_DELTA_THRESHOLD)
    m_dir = full_sign != 0

    return {
        "target_gene": target_gene,
        "model": model_name,
        "score_col": score_col,
        "n_genes": int(len(sub)),
        "n_full_directional": int(m_dir.sum()),
        "pearson_vs_full": pearson_fast(full, pred),
        "spearman_vs_full": spearman_fast(full, pred),
        "direction_concordance_on_full_directional": float(np.mean(full_sign[m_dir] == pred_sign[m_dir])) if m_dir.sum() > 0 else np.nan,
        "full_delta_threshold": float(FULL_IMAS_DELTA_THRESHOLD),
        "baseline_delta_threshold": float(BASELINE_DELTA_THRESHOLD),
    }


def topn_overlap_with_null(df: pd.DataFrame, target_gene: str, model_name: str, score_col: str) -> pd.DataFrame:
    sub = df[["gene", "score_full_imas", score_col]].dropna().copy()
    if len(sub) == 0:
        return pd.DataFrame()

    sub["abs_full"] = sub["score_full_imas"].abs()
    sub["abs_model"] = sub[score_col].abs()
    sub["full_sign"] = score_to_sign(sub["score_full_imas"].to_numpy(float), 0.0)
    sub["model_sign"] = score_to_sign(sub[score_col].to_numpy(float), 0.0)

    genes = sub["gene"].astype(str).to_numpy()
    full_order = np.argsort(-sub["abs_full"].to_numpy(float))
    model_order = np.argsort(-sub["abs_model"].to_numpy(float))

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for topn in TOPN_LIST:
        n = min(int(topn), len(sub))
        full_idx = set(full_order[:n].tolist())
        model_idx = set(model_order[:n].tolist())
        inter = sorted(full_idx & model_idx)
        overlap = len(inter)
        jaccard = overlap / max(1, len(full_idx | model_idx))

        if overlap > 0:
            same_dir = float(np.mean(sub.iloc[inter]["full_sign"].to_numpy(np.int8) == sub.iloc[inter]["model_sign"].to_numpy(np.int8)))
        else:
            same_dir = np.nan

        null_overlap = np.zeros(N_SHUFFLE, dtype=np.float32)
        null_same = np.zeros(N_SHUFFLE, dtype=np.float32)
        model_idx_arr = np.array(sorted(model_idx), dtype=np.int64)
        model_sign = sub["model_sign"].to_numpy(np.int8)
        full_sign = sub["full_sign"].to_numpy(np.int8)

        for i in range(N_SHUFFLE):
            perm_model = set(rng.choice(np.arange(len(sub)), size=n, replace=False).tolist())
            inter_null = sorted(full_idx & perm_model)
            null_overlap[i] = len(inter_null)
            if len(inter_null) > 0:
                # Shuffle model signs among genes to preserve sign balance approximately.
                shuffled_sign = rng.permutation(model_sign)
                null_same[i] = float(np.mean(full_sign[inter_null] == shuffled_sign[inter_null]))
            else:
                null_same[i] = np.nan

        null_overlap_mean = float(np.nanmean(null_overlap))
        null_overlap_sd = float(np.nanstd(null_overlap, ddof=1))
        overlap_z = (overlap - null_overlap_mean) / (null_overlap_sd + 1e-12)
        overlap_emp_p = float((1 + np.sum(null_overlap >= overlap)) / (N_SHUFFLE + 1))

        null_same_mean = float(np.nanmean(null_same))
        null_same_sd = float(np.nanstd(null_same, ddof=1))
        same_z = (same_dir - null_same_mean) / (null_same_sd + 1e-12) if np.isfinite(same_dir) else np.nan

        rows.append({
            "target_gene": target_gene,
            "model": model_name,
            "score_col": score_col,
            "topN": int(n),
            "n_intersection": int(overlap),
            "jaccard": float(jaccard),
            "same_direction_in_intersection": float(same_dir) if np.isfinite(same_dir) else np.nan,
            "null_overlap_mean": null_overlap_mean,
            "null_overlap_sd": null_overlap_sd,
            "overlap_z": float(overlap_z),
            "overlap_empirical_p": overlap_emp_p,
            "null_same_direction_mean": null_same_mean,
            "null_same_direction_sd": null_same_sd,
            "same_direction_z": float(same_z) if np.isfinite(same_z) else np.nan,
        })

    return pd.DataFrame(rows)


# =========================================================
# 7) PLOTS
# =========================================================

def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def plot_summary(model_summary: pd.DataFrame, topn_summary: pd.DataFrame, out_dir: Path):
    if len(model_summary) > 0:
        metric = "spearman_vs_full"
        order = ["Expression_only", "Correlation_only", "Linear_residualized"]
        order = [x for x in order if x in set(model_summary["model"])]
        rows = []
        for m in order:
            sub = model_summary[model_summary["model"] == m]
            rows.append({"model": m, "mean": sub[metric].mean(), "sem": sem(sub[metric])})
        stat = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        x = np.arange(len(stat))
        ax.bar(x, stat["mean"].to_numpy(float), yerr=stat["sem"].to_numpy(float), capsize=4, edgecolor="black", linewidth=0.8)
        for i, m in enumerate(stat["model"]):
            vals = model_summary.loc[model_summary["model"] == m, metric].to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=28, edgecolor="black", linewidth=0.3)
        ax.axhline(0, linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(stat["model"], rotation=25, ha="right")
        ax.set_ylabel("Spearman correlation with Full IMAS score")
        ax.set_title("Simple baselines vs Full IMAS virtual perturbation")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_score_correlation_bar.png", dpi=DPI)
        fig.savefig(out_dir / "baseline_score_correlation_bar.svg")
        plt.close(fig)

    if len(topn_summary) > 0:
        use_topn = 100 if 100 in set(topn_summary["topN"]) else int(topn_summary["topN"].iloc[0])
        df = topn_summary[topn_summary["topN"] == use_topn].copy()
        order = ["Expression_only", "Correlation_only", "Linear_residualized"]
        order = [x for x in order if x in set(df["model"])]
        rows = []
        for m in order:
            sub = df[df["model"] == m]
            rows.append({"model": m, "mean": sub["jaccard"].mean(), "sem": sem(sub["jaccard"])})
        stat = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        x = np.arange(len(stat))
        ax.bar(x, stat["mean"].to_numpy(float), yerr=stat["sem"].to_numpy(float), capsize=4, edgecolor="black", linewidth=0.8)
        for i, m in enumerate(stat["model"]):
            vals = df.loc[df["model"] == m, "jaccard"].to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=28, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(stat["model"], rotation=25, ha="right")
        ax.set_ylabel(f"Top-{use_topn} Jaccard overlap")
        ax.set_title("Top response genes: Full IMAS vs simple baselines")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_topN_overlap_bar.png", dpi=DPI)
        fig.savefig(out_dir / "baseline_topN_overlap_bar.svg")
        plt.close(fig)


# =========================================================
# 8) MAIN
# =========================================================

def main():
    log("=" * 100)
    log("[PATHS]")
    log("BASE =", BASE)
    log("RNA_DIR =", RNA_DIR)
    log("RNA_COUNTS =", RNA_COUNTS)
    log("RNA_META =", RNA_META)
    log("GENE_FILE =", GENE_FILE)

    for p in [BASE, RNA_DIR, RNA_COUNTS, RNA_META, GENE_FILE]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    pred_dir, ko_files = find_pred_dir_and_files()
    if TARGET_GENES is not None:
        allowed = {str(x).upper() for x in TARGET_GENES}
        ko_files = [fp for fp in ko_files if parse_target_from_stage5_file(fp).upper() in allowed]
    if len(ko_files) == 0:
        raise FileNotFoundError("No Stage5 KO files left after TARGET_GENES filtering")

    out_dir = pred_dir / "linear_correlation_baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("[FOUND] PRED_DIR =", pred_dir)
    log("[FOUND] KO files:")
    for fp in ko_files:
        log(" ", fp, "| target =", parse_target_from_stage5_file(fp))
    log("[OUT_DIR]", out_dir)

    gene_names = read_gene_file(GENE_FILE)
    X_full = np.load(RNA_COUNTS, mmap_mode="r")
    meta_full = load_meta_table(RNA_META, X_full.shape[0])

    if X_full.shape[1] != len(gene_names):
        raise ValueError(f"n_genes mismatch: X={X_full.shape[1]} vs genes={len(gene_names)}")

    cell_idx, X_use, meta_use = subset_cells_if_needed(X_full, meta_full)
    log("=" * 100)
    log(f"[DATA] X_full shape = {X_full.shape}")
    log(f"[DATA] X_use shape = {X_use.shape}")
    log(f"[DATA] meta columns = {meta_use.columns.tolist()}")

    model_summary_rows = []
    topn_rows = []

    for ko_file in ko_files:
        target = parse_target_from_stage5_file(ko_file)
        target_out = out_dir / f"target_{safe_name(target)}"
        target_out.mkdir(parents=True, exist_ok=True)

        log("\n" + "#" * 120)
        log(f"[TARGET] {target}")
        log(f"[KO_FILE] {ko_file}")

        full_score = load_full_imas_stage5_score(ko_file, target)
        simple = compute_simple_baselines_for_target(X_use, meta_use, gene_names, target)

        merged = full_score.merge(simple, on="gene", how="inner")
        merged = merged.dropna(subset=["score_full_imas"]).copy()
        merged = merged.sort_values("score_full_imas", key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

        merged_path = target_out / f"{safe_name(target)}_gene_level_baseline_comparison.tsv"
        merged.to_csv(merged_path, sep="\t", index=False)
        log(f"[SAVE] {merged_path} | n_genes={len(merged)}")

        model_defs = [
            ("Expression_only", "score_expr_only"),
            ("Correlation_only", "score_corr_only"),
            ("Linear_residualized", "score_linear_residualized"),
        ]

        for model_name, score_col in model_defs:
            ms = model_level_metrics(merged, target, model_name, score_col)
            model_summary_rows.append(ms)
            tn = topn_overlap_with_null(merged, target, model_name, score_col)
            if len(tn) > 0:
                topn_rows.append(tn)

        # Per-target scatter plots for quick QC
        for model_name, score_col in model_defs:
            fig, ax = plt.subplots(figsize=(4.2, 4.0))
            sub = merged[["score_full_imas", score_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) > 5000:
                sub = sub.sample(5000, random_state=RANDOM_SEED)
            ax.scatter(sub[score_col], sub["score_full_imas"], s=6, alpha=0.35)
            ax.axhline(0, linestyle="--", linewidth=0.8)
            ax.axvline(0, linestyle="--", linewidth=0.8)
            ax.set_xlabel(model_name)
            ax.set_ylabel("Full IMAS Stage5 score")
            ax.set_title(f"{target}: Full IMAS vs {model_name}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(target_out / f"{safe_name(target)}_scatter_full_vs_{model_name}.png", dpi=DPI)
            fig.savefig(target_out / f"{safe_name(target)}_scatter_full_vs_{model_name}.svg")
            plt.close(fig)

    model_summary = pd.DataFrame(model_summary_rows)
    topn_summary = pd.concat(topn_rows, axis=0, ignore_index=True) if len(topn_rows) > 0 else pd.DataFrame()

    model_summary_path = out_dir / "all_targets_model_level_summary.tsv"
    model_summary.to_csv(model_summary_path, sep="\t", index=False)
    log("=" * 100)
    log(f"[SAVE] {model_summary_path}")

    topn_summary_path = out_dir / "all_targets_topN_overlap_summary.tsv"
    topn_summary.to_csv(topn_summary_path, sep="\t", index=False)
    log(f"[SAVE] {topn_summary_path}")

    plot_summary(model_summary, topn_summary, out_dir)

    config = {
        "BASE": str(BASE),
        "EXPLOIT_HNSCC_ROOT": str(EXPLOIT_HNSCC_ROOT),
        "RNA_DIR": str(RNA_DIR),
        "RNA_COUNTS": str(RNA_COUNTS),
        "RNA_META": str(RNA_META),
        "GENE_FILE": str(GENE_FILE),
        "PRED_DIR": str(pred_dir),
        "OUT_DIR": str(out_dir),
        "TARGET_GENES": TARGET_GENES,
        "TARGET_SUM": TARGET_SUM,
        "FULL_IMAS_DELTA_THRESHOLD": FULL_IMAS_DELTA_THRESHOLD,
        "BASELINE_DELTA_THRESHOLD": BASELINE_DELTA_THRESHOLD,
        "TARGET_HIGH_Q": TARGET_HIGH_Q,
        "TARGET_LOW_Q": TARGET_LOW_Q,
        "N_SHUFFLE": N_SHUFFLE,
        "USE_COVARIATES": USE_COVARIATES,
        "MAX_CELLS_FOR_BASELINE": MAX_CELLS_FOR_BASELINE,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log("=" * 100)
    log("[DONE]")
    log("Output dir:", out_dir)


if __name__ == "__main__":
    main()
