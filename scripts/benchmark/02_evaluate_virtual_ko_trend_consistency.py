# -*- coding: utf-8 -*-
"""
trend_consistency_check_virtual_ko_allgenes.py

功能：
1) 读取真实 DEG: processed/DEG_0_vs_others/*.deg_all.tsv
2) 读取 control reference 与 virtual KO 结果
3) 用 control vs virtual KO 计算预测变化:
      pred_score = mean(virtual_KO) - mean(control)
4) 直接在所有可对齐基因上做趋势一致性检测
   - truth_trend = sign3(real_logFC, TRUTH_TREND_THRESHOLD)
   - pred_trend  = sign3(pred_score, PRED_TREND_THRESHOLD)
5) 输出：
   - 每个方法每个 DEG 对比的 trend 明细
   - summary 统计
   - confusion matrix
   - heatmap / trend summary / top inconsistent genes

说明：
- 不再保留 marker gene 子集
- 不再输出 marker summary
- 直接基于 all-gene overlap 比较真值与 KO 预测值
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) CONFIG
# ============================================================

BASE = Path(get_verify_pc_root())

# ===== 真值 DEG 目录 =====
DEG_DIR = BASE / "processed" / "DEG_0_vs_others"

# ===== virtual KO 目录 =====
# 如需 noleak 版本，改成:
# VKO_DIR = BASE / "processed" / "virtual_ko_baselines_noleak"
VKO_DIR = BASE / "processed" / "virtual_ko_baselines"

# ===== 输出目录 =====
OUT_DIR = BASE / "processed" / "trend_consistency_check_virtual_ko_allgenes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 可选：只分析指定 DEG 文件；留空则扫描全部 =====
ONLY_DEG_FILES = [
    # "0_vs_1.deg_all.tsv",
]

# ===== 可选：只分析指定方法 =====
ONLY_METHODS = [
    # "scGPT_style",
    # "CPA_style",
    # "GEARS_style",
    # "scGPT_style_noleak",
    # "CPA_style_noleak",
    # "GEARS_style_noleak",
]

# ===== 真实 DEG 列 =====
DEG_GENE_COL = "gene"
DEG_FC_COL = "log2FC_group2_vs_group1"
DEG_PADJ_COL = "padj"

# ===== 真实 DEG 过滤 =====
USE_ONLY_SIGNIFICANT_DEG = True
PADJ_THRESHOLD = 0.00001
ABS_LOGFC_THRESHOLD = 0.00025

# ===== 趋势判定阈值 =====
TRUTH_TREND_THRESHOLD = 0.00025
PRED_TREND_THRESHOLD = 0.00001

# ===== 可选：只统计 top 强变化基因 =====
USE_TOPK_STRONG_GENES = False
TOPK_STRONG_GENES = 500

# ===== 作图 =====
DPI = 260
COLOR_MATCH = "#4daf4a"
COLOR_MISMATCH = "#e41a1c"
COLOR_UP = "#d95f02"
COLOR_DOWN = "#1b9e77"
COLOR_FLAT = "#999999"

DEFAULT_METHODS = ["scGPT_style", "CPA_style", "GEARS_style"]


# ============================================================
# 1) HELPERS
# ============================================================

def normalize_gene(x: str) -> str:
    return str(x).strip().upper()


def safe_name(x: str, n: int = 120) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n]


def sign3(x: float, thr: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > thr:
        return 1
    if x < -thr:
        return -1
    return 0


def trend_label(v: int) -> str:
    if v > 0:
        return "UP"
    if v < 0:
        return "DOWN"
    return "FLAT"


def mean_shift(control_mat: np.ndarray, pred_mat: np.ndarray) -> np.ndarray:
    if control_mat.shape[1] != pred_mat.shape[1]:
        raise ValueError("control_mat 与 pred_mat 的基因维度不一致")
    control_mean = np.asarray(control_mat, dtype=np.float32).mean(axis=0)
    pred_mean = np.asarray(pred_mat, dtype=np.float32).mean(axis=0)
    return pred_mean - control_mean


def find_first_file(dir_path: Path, pattern: str) -> Path:
    files = sorted(dir_path.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"{dir_path} 下找不到文件: {pattern}")
    return files[0]


def parse_deg_compare_name(fp: Path) -> str:
    return fp.name.replace(".deg_all.tsv", "")


def print_pred_score_distribution(pred_df: pd.DataFrame, method_name: str):
    x = pred_df["pred_score"].abs().dropna()
    if len(x) == 0:
        print(f"[PRED DIST] {method_name}: empty")
        return
    qs = x.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    print(
        f"[PRED DIST] {method_name}: "
        f"median={qs.get(0.5, np.nan):.6f}, "
        f"q75={qs.get(0.75, np.nan):.6f}, "
        f"q90={qs.get(0.9, np.nan):.6f}, "
        f"q95={qs.get(0.95, np.nan):.6f}, "
        f"q99={qs.get(0.99, np.nan):.6f}, "
        f"max={x.max():.6f}"
    )


# ============================================================
# 2) FILE COLLECTION
# ============================================================

def collect_deg_files(deg_dir: Path, only_deg_files: List[str]) -> List[Path]:
    if not deg_dir.exists():
        raise FileNotFoundError(f"DEG_DIR 不存在: {deg_dir}")

    if only_deg_files:
        out = []
        for f in only_deg_files:
            p = deg_dir / f
            if p.exists():
                out.append(p)
            else:
                print(f"[WARN] DEG file not found: {p}")
        return out

    return sorted(deg_dir.glob("*.deg_all.tsv"))


def collect_method_dirs(vko_dir: Path, only_methods: List[str]) -> List[Tuple[str, Path]]:
    methods = only_methods if only_methods else DEFAULT_METHODS
    out = []
    for m in methods:
        mp = vko_dir / m
        if mp.exists():
            out.append((m, mp))
        else:
            print(f"[WARN] method dir not found: {mp}")
    return out


# ============================================================
# 3) LOADERS
# ============================================================

def load_deg_table(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep="\t")

    if DEG_GENE_COL not in df.columns:
        raise ValueError(f"{fp} 缺少列: {DEG_GENE_COL}")
    if DEG_FC_COL not in df.columns:
        raise ValueError(f"{fp} 缺少列: {DEG_FC_COL}")

    x = df.copy()
    x[DEG_GENE_COL] = x[DEG_GENE_COL].astype(str)
    x["_gene_norm"] = x[DEG_GENE_COL].map(normalize_gene)
    x[DEG_FC_COL] = pd.to_numeric(x[DEG_FC_COL], errors="coerce")

    if DEG_PADJ_COL in x.columns:
        x[DEG_PADJ_COL] = pd.to_numeric(x[DEG_PADJ_COL], errors="coerce")

    if USE_ONLY_SIGNIFICANT_DEG and DEG_PADJ_COL in x.columns:
        x = x[
            (x[DEG_PADJ_COL] < PADJ_THRESHOLD) &
            (x[DEG_FC_COL].abs() >= ABS_LOGFC_THRESHOLD)
        ].copy()

    x = x.dropna(subset=[DEG_FC_COL])
    x = x.drop_duplicates(subset=["_gene_norm"], keep="first").copy()

    if USE_TOPK_STRONG_GENES and len(x) > TOPK_STRONG_GENES:
        x["_abs_fc"] = x[DEG_FC_COL].abs()
        x = x.sort_values("_abs_fc", ascending=False).head(TOPK_STRONG_GENES).copy()
        x = x.drop(columns=["_abs_fc"])

    return x


def load_control_reference(vko_dir: Path) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
    ref_dir = vko_dir / "reference_control"
    control_mat = np.load(ref_dir / "PC9_day0.log1p.npy")
    genes = pd.read_csv(ref_dir / "genes.tsv", sep="\t")["gene"].astype(str).tolist()
    cells = pd.read_csv(ref_dir / "cell_ids.tsv", sep="\t")["cell_id"].astype(str).tolist()
    meta = pd.read_csv(ref_dir / "metadata.tsv", sep="\t")
    return control_mat, genes, cells, meta


def load_virtual_ko_method(method_dir: Path) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame]:
    pred_path = find_first_file(method_dir, "*.virtual_ko.log1p.npy")
    meta_path = find_first_file(method_dir, "*.virtual_ko.meta.tsv")
    gene_path = method_dir / "genes.tsv"
    cell_path = method_dir / "cell_ids.tsv"

    pred_mat = np.load(pred_path)
    genes = pd.read_csv(gene_path, sep="\t")["gene"].astype(str).tolist()
    cells = pd.read_csv(cell_path, sep="\t")["cell_id"].astype(str).tolist()
    meta = pd.read_csv(meta_path, sep="\t")

    return pred_mat, genes, cells, meta


def build_pred_table_from_virtual_ko(
    control_mat: np.ndarray,
    control_genes: List[str],
    pred_mat: np.ndarray,
    pred_genes: List[str],
    method_name: str
) -> pd.DataFrame:
    c_gene_norm = [normalize_gene(g) for g in control_genes]
    p_gene_norm = [normalize_gene(g) for g in pred_genes]

    if c_gene_norm != p_gene_norm:
        raise ValueError(f"{method_name}: control genes 与 pred genes 不一致，请先统一基因顺序。")

    delta = mean_shift(control_mat, pred_mat)
    out = pd.DataFrame({
        "gene": control_genes,
        "_gene_norm": c_gene_norm,
        "pred_score": delta.astype(np.float32),
    })
    out = out.drop_duplicates(subset=["_gene_norm"], keep="first").copy()
    return out


# ============================================================
# 4) TREND CONSISTENCY CORE
# ============================================================

def build_trend_compare_table(
    deg_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    deg_name: str,
    method_name: str,
) -> pd.DataFrame:
    merged = deg_df[["_gene_norm", DEG_GENE_COL, DEG_FC_COL]].merge(
        pred_df[["_gene_norm", "gene", "pred_score"]],
        on="_gene_norm",
        how="inner",
        suffixes=("_deg", "_pred")
    ).copy()

    if len(merged) == 0:
        return pd.DataFrame()

    merged["gene_show"] = merged["_gene_norm"]
    merged["truth_raw"] = pd.to_numeric(merged[DEG_FC_COL], errors="coerce")
    merged["pred_raw"] = pd.to_numeric(merged["pred_score"], errors="coerce")

    merged["truth_trend"] = merged["truth_raw"].map(lambda x: sign3(x, TRUTH_TREND_THRESHOLD))
    merged["pred_trend"] = merged["pred_raw"].map(lambda x: sign3(x, PRED_TREND_THRESHOLD))

    merged["truth_label"] = merged["truth_trend"].map(trend_label)
    merged["pred_label"] = merged["pred_trend"].map(trend_label)

    merged["same_trend_3class"] = merged["truth_trend"] == merged["pred_trend"]

    nonflat_mask = (merged["truth_trend"] != 0) & (merged["pred_trend"] != 0)
    merged["same_trend_nonzero_only"] = False
    merged.loc[nonflat_mask, "same_trend_nonzero_only"] = (
        merged.loc[nonflat_mask, "truth_trend"] == merged.loc[nonflat_mask, "pred_trend"]
    )

    merged["truth_nonflat"] = merged["truth_trend"] != 0
    merged["pred_nonflat"] = merged["pred_trend"] != 0

    merged["abs_truth"] = merged["truth_raw"].abs()
    merged["abs_pred"] = merged["pred_raw"].abs()
    merged["trend_margin"] = merged["abs_truth"] + merged["abs_pred"]

    merged["deg_compare"] = deg_name
    merged["method_name"] = method_name

    cols = [
        "deg_compare", "method_name",
        "gene_show", "_gene_norm",
        "truth_raw", "pred_raw",
        "truth_trend", "pred_trend",
        "truth_label", "pred_label",
        "same_trend_3class", "same_trend_nonzero_only",
        "truth_nonflat", "pred_nonflat",
        "abs_truth", "abs_pred", "trend_margin",
    ]
    return merged[cols].sort_values(["abs_truth", "abs_pred"], ascending=False).reset_index(drop=True)


def summarize_trend_compare(df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return {
            "n_overlap": 0,
            "direction_acc_3class": np.nan,
            "direction_acc_nonzero_only": np.nan,
            "n_nonzero_overlap": 0,
            "truth_up_recall": np.nan,
            "truth_down_recall": np.nan,
            "balanced_acc_ud": np.nan,
            "pred_nonflat_rate": np.nan,
            "truth_nonflat_rate": np.nan,
        }

    out = {"n_overlap": int(len(df))}
    out["direction_acc_3class"] = float(df["same_trend_3class"].mean())

    nonzero_df = df[(df["truth_trend"] != 0) & (df["pred_trend"] != 0)].copy()
    out["n_nonzero_overlap"] = int(len(nonzero_df))
    out["direction_acc_nonzero_only"] = (
        float(nonzero_df["same_trend_nonzero_only"].mean()) if len(nonzero_df) > 0 else np.nan
    )

    truth_up = df[df["truth_trend"] == 1]
    truth_down = df[df["truth_trend"] == -1]

    out["truth_up_recall"] = (
        float((truth_up["pred_trend"] == 1).mean()) if len(truth_up) > 0 else np.nan
    )
    out["truth_down_recall"] = (
        float((truth_down["pred_trend"] == -1).mean()) if len(truth_down) > 0 else np.nan
    )

    vals = [out["truth_up_recall"], out["truth_down_recall"]]
    vals = [v for v in vals if pd.notna(v)]
    out["balanced_acc_ud"] = float(np.mean(vals)) if len(vals) > 0 else np.nan

    out["pred_nonflat_rate"] = float((df["pred_trend"] != 0).mean())
    out["truth_nonflat_rate"] = float((df["truth_trend"] != 0).mean())

    return out


def build_confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["truth_label", "pred_label", "n"])

    order = ["UP", "FLAT", "DOWN"]

    ct = pd.crosstab(df["truth_label"], df["pred_label"], dropna=False)
    for r in order:
        if r not in ct.index:
            ct.loc[r] = 0
    for c in order:
        if c not in ct.columns:
            ct[c] = 0

    ct = ct.loc[order, order].copy()
    ct_long = ct.stack().reset_index()
    ct_long.columns = ["truth_label", "pred_label", "n"]
    return ct_long


# ============================================================
# 5) PLOTTING
# ============================================================

def plot_trend_confusion_heatmap(conf_df: pd.DataFrame, out_png: Path, out_svg: Path, title: str):
    if len(conf_df) == 0:
        return

    order = ["UP", "FLAT", "DOWN"]
    mat = conf_df.pivot(index="truth_label", columns="pred_label", values="n").fillna(0)
    mat = mat.reindex(index=order, columns=order, fill_value=0)

    arr = mat.values.astype(float)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=DPI)
    im = ax.imshow(arr, aspect="auto")

    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticklabels(order)
    ax.set_xlabel("pred trend")
    ax.set_ylabel("truth trend")
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("count")

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_svg), bbox_inches="tight")
    plt.close(fig)


def plot_trend_agreement_bar(summary_df: pd.DataFrame, out_png: Path, out_svg: Path):
    if len(summary_df) == 0:
        return

    x = summary_df.copy().sort_values(
        ["balanced_acc_ud", "direction_acc_nonzero_only", "direction_acc_3class"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    x["pair_label"] = x["deg_compare"].astype(str) + "\n" + x["method_name"].astype(str)

    ypos = np.arange(len(x))
    fig_h = max(6, 0.36 * len(x))
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=DPI)

    ax.barh(ypos, x["direction_acc_3class"].fillna(0), alpha=0.80, label="direction_acc_3class")
    ax.barh(ypos, x["direction_acc_nonzero_only"].fillna(0), alpha=0.55, label="direction_acc_nonzero_only")
    ax.barh(ypos, x["balanced_acc_ud"].fillna(0), alpha=0.40, label="balanced_acc_ud")

    ax.set_yticks(ypos)
    ax.set_yticklabels(x["pair_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title("Trend consistency summary (all genes)")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_svg), bbox_inches="tight")
    plt.close(fig)


def plot_top_inconsistent_genes(df: pd.DataFrame, out_png: Path, out_svg: Path, title: str, topn: int = 20):
    if len(df) == 0:
        return

    x = df.copy()
    x = x[x["truth_trend"] != 0].copy()
    x = x[x["truth_trend"] != x["pred_trend"]].copy()

    if len(x) == 0:
        return

    x = x.sort_values(["abs_truth", "abs_pred"], ascending=False).head(topn).copy()
    x = x.sort_values("truth_raw", ascending=True).reset_index(drop=True)

    ypos = np.arange(len(x))
    fig_h = max(4.5, 0.42 * len(x))
    fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=DPI)

    colors = x["truth_trend"].map(lambda z: COLOR_UP if z == 1 else COLOR_DOWN)

    ax.barh(ypos, x["truth_raw"].values, alpha=0.70, color=colors, label="truth logFC")
    ax.scatter(x["pred_raw"].values, ypos, s=48, color=COLOR_MISMATCH, edgecolors="black", linewidths=0.35, label="pred shift")

    ax.axvline(0, color="#666666", lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(x["gene_show"].tolist())
    ax.set_xlabel("effect size")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    fig.savefig(str(out_png), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_svg), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6) MAIN
# ============================================================

def main():
    print(f"[INFO] DEG_DIR = {DEG_DIR}")
    print(f"[INFO] VKO_DIR = {VKO_DIR}")
    print(f"[INFO] OUT_DIR = {OUT_DIR}")

    deg_files = collect_deg_files(DEG_DIR, ONLY_DEG_FILES)
    method_dirs = collect_method_dirs(VKO_DIR, ONLY_METHODS)

    if len(deg_files) == 0:
        raise FileNotFoundError(f"没有找到 DEG 文件: {DEG_DIR}")
    if len(method_dirs) == 0:
        raise FileNotFoundError(f"没有找到任何方法目录: {VKO_DIR}")

    control_mat, control_genes, control_cells, control_meta = load_control_reference(VKO_DIR)
    print(f"[INFO] control_mat shape = {control_mat.shape}")

    method_pred_tables = {}
    for method_name, method_dir in method_dirs:
        print(f"[LOAD METHOD] {method_name} <- {method_dir}")
        pred_mat, pred_genes, pred_cells, pred_meta = load_virtual_ko_method(method_dir)
        print(f"    pred_mat shape = {pred_mat.shape}")

        if len(control_cells) != len(pred_cells):
            print(f"[WARN] {method_name}: cell 数不一致，请确认输入是否来自同一 control 集。")

        pred_df = build_pred_table_from_virtual_ko(
            control_mat=control_mat,
            control_genes=control_genes,
            pred_mat=pred_mat,
            pred_genes=pred_genes,
            method_name=method_name,
        )
        method_pred_tables[method_name] = pred_df
        print_pred_score_distribution(pred_df, method_name)

    summary_rows = []
    all_detail_rows = []
    confusion_rows = []

    details_dir = OUT_DIR / "details"
    confusion_dir = OUT_DIR / "confusion"
    plots_dir = OUT_DIR / "plots"

    details_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for deg_fp in deg_files:
        deg_name = parse_deg_compare_name(deg_fp)
        print("\n" + "=" * 100)
        print(f"[DEG] {deg_name}")

        try:
            deg_df = load_deg_table(deg_fp)
        except Exception as e:
            print(f"[SKIP] load DEG failed: {deg_fp.name} | {e}")
            continue

        for method_name, _ in method_dirs:
            print(f"  [METHOD] {method_name}")
            pred_df = method_pred_tables[method_name]

            try:
                trend_df = build_trend_compare_table(
                    deg_df=deg_df,
                    pred_df=pred_df,
                    deg_name=deg_name,
                    method_name=method_name,
                )
            except Exception as e:
                print(f"  [SKIP] compare failed: {method_name} | {e}")
                continue

            if len(trend_df) == 0:
                continue

            stat_all = summarize_trend_compare(trend_df)
            conf_df = build_confusion_table(trend_df)

            detail_file = details_dir / f"{safe_name(deg_name)}__vs__{safe_name(method_name)}.trend_detail.tsv"
            trend_df.to_csv(detail_file, sep="\t", index=False)

            conf_file = confusion_dir / f"{safe_name(deg_name)}__vs__{safe_name(method_name)}.confusion.tsv"
            conf_df.to_csv(conf_file, sep="\t", index=False)

            tmp = trend_df.copy()
            tmp["pair_id"] = f"{deg_name}__vs__{method_name}"
            all_detail_rows.append(tmp)

            conf_tmp = conf_df.copy()
            conf_tmp["deg_compare"] = deg_name
            conf_tmp["method_name"] = method_name
            confusion_rows.append(conf_tmp)

            summary_rows.append({
                "deg_compare": deg_name,
                "method_name": method_name,
                "n_overlap": stat_all["n_overlap"],
                "direction_acc_3class": stat_all["direction_acc_3class"],
                "direction_acc_nonzero_only": stat_all["direction_acc_nonzero_only"],
                "n_nonzero_overlap": stat_all["n_nonzero_overlap"],
                "truth_up_recall": stat_all["truth_up_recall"],
                "truth_down_recall": stat_all["truth_down_recall"],
                "balanced_acc_ud": stat_all["balanced_acc_ud"],
                "pred_nonflat_rate": stat_all["pred_nonflat_rate"],
                "truth_nonflat_rate": stat_all["truth_nonflat_rate"],
                "detail_file": str(detail_file),
                "confusion_file": str(conf_file),
            })

            plot_base = plots_dir / f"{safe_name(deg_name)}__vs__{safe_name(method_name)}"

            plot_trend_confusion_heatmap(
                conf_df=conf_df,
                out_png=plot_base.with_name(plot_base.name + "_confusion").with_suffix(".png"),
                out_svg=plot_base.with_name(plot_base.name + "_confusion").with_suffix(".svg"),
                title=f"{deg_name} vs {method_name} trend confusion",
            )

            plot_top_inconsistent_genes(
                df=trend_df,
                out_png=plot_base.with_name(plot_base.name + "_top_inconsistent").with_suffix(".png"),
                out_svg=plot_base.with_name(plot_base.name + "_top_inconsistent").with_suffix(".svg"),
                title=f"{deg_name} vs {method_name} top inconsistent genes",
                topn=20,
            )

            msg = (
                f"    n={stat_all['n_overlap']}, "
                f"acc3={stat_all['direction_acc_3class']:.6f}, "
                f"acc_nz={stat_all['direction_acc_nonzero_only']:.6f}, "
                f"nz_n={stat_all['n_nonzero_overlap']}, "
                f"pred_nonflat_rate={stat_all['pred_nonflat_rate']:.6f}"
            )
            print(msg)

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) == 0:
        print("[WARN] no valid comparisons")
        return

    summary_df = summary_df.sort_values(
        ["balanced_acc_ud", "direction_acc_nonzero_only", "direction_acc_3class", "n_overlap"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    summary_df.to_csv(OUT_DIR / "summary.tsv", sep="\t", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    if len(all_detail_rows) > 0:
        all_detail_df = pd.concat(all_detail_rows, axis=0, ignore_index=True)
        all_detail_df.to_csv(OUT_DIR / "all_trend_details.tsv", sep="\t", index=False)

    if len(confusion_rows) > 0:
        all_conf_df = pd.concat(confusion_rows, axis=0, ignore_index=True)
        all_conf_df.to_csv(OUT_DIR / "all_confusion_tables.tsv", sep="\t", index=False)

    plot_trend_agreement_bar(
        summary_df=summary_df,
        out_png=OUT_DIR / "trend_summary_bar.png",
        out_svg=OUT_DIR / "trend_summary_bar.svg",
    )

    print("\n" + "=" * 100)
    print("[DONE] summary ->", OUT_DIR / "summary.tsv")
    print("[DONE] plots   ->", plots_dir)
    print("[DONE] details ->", details_dir)
    print("[DONE] confusion ->", confusion_dir)

    print("\nTop results:")
    show_cols = [
        "deg_compare",
        "method_name",
        "n_overlap",
        "direction_acc_3class",
        "direction_acc_nonzero_only",
        "n_nonzero_overlap",
        "balanced_acc_ud",
        "pred_nonflat_rate",
    ]
    print(summary_df[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
