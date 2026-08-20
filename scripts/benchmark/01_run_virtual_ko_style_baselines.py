# REPOSITORY NOTE
# The scGPT-style, CPA-style, and GEARS-style methods in this script are
# task-aligned architectural controls. They are not executions of the official
# scGPT, CPA, or GEARS packages/checkpoints. Preserve the “-style” labels.

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# =========================
# 0) 路径配置
# =========================
BASE = get_verify_pc_root()

ASSET_DIR = os.path.join(BASE, "assets")
PROC_DIR = os.path.join(BASE, "processed")
OUT_DIR = os.path.join(PROC_DIR, "virtual_ko_baselines_noleak")
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

COUNT_MATRIX = os.path.join(BASE, "GSE150949_pc9_count_matrix.csv.gz")
META_FILE = os.path.join(BASE, "GSE150949_metaData_with_lineage.txt.gz")

# 只用 control 组做虚拟敲除
CONTROL_SAMPLE_NAME = "0"
TARGET_GENE = "EGFR"

# 输出配置
SAVE_FLOAT_DTYPE = np.float32
COUNTS_DTYPE = np.int32
TARGET_SUM = 1e4

# baseline 超参
N_HVG = 2000
PCA_DIM = 64

# GEARS-style
GEARS_NEIGHBOR_GENES = 50
GEARS_PROPAGATION_ALPHA = 0.20  # 比原先更保守
GEARS_STRENGTH_QUANTILE = 0.75  # 用 control 群体统计量确定全局 KO 强度

# CPA-style
CPA_EFFECT_SCALE = 0.20         # 全局 effect 缩放，越大越强
CPA_EFFECT_CLIP_QUANTILE = 0.95 # 防止 effect 向量过大

# scGPT-style
SCGPT_MASK_TO_ZERO = True

# sample_name -> 更清晰的样本名
SAMPLE_NAME_MAP = {
    "0": "PC9_day0",
    "3_rep1": "PC9_day3_rep1",
    "3_rep2": "PC9_day3_rep2",
    "7_rep1": "PC9_day7_rep1",
    "7_rep2": "PC9_day7_rep2",
    "14_rep1_high": "PC9_day14_rep1_high",
    "14_rep2_high": "PC9_day14_rep2_high",
    "14_rep1_med": "PC9_day14_rep1_med",
    "14_rep2_med": "PC9_day14_rep2_med",
    "14_rep1_low": "PC9_day14_rep1_low",
    "14_rep2_low": "PC9_day14_rep2_low",
}

print("[INFO] COUNT_MATRIX =", COUNT_MATRIX)
print("[INFO] META_FILE    =", META_FILE)
print("[INFO] OUT_DIR      =", OUT_DIR)
print("[INFO] CONTROL_SAMPLE_NAME =", CONTROL_SAMPLE_NAME)
print("[INFO] TARGET_GENE  =", TARGET_GENE)
print("=" * 100)

if not os.path.exists(COUNT_MATRIX):
    raise FileNotFoundError(f"COUNT_MATRIX 不存在: {COUNT_MATRIX}")
if not os.path.exists(META_FILE):
    raise FileNotFoundError(f"META_FILE 不存在: {META_FILE}")

# =========================
# 1) 工具函数
# =========================
def sanitize_name(x: str) -> str:
    x = str(x)
    for a, b in [
        (" ", "_"), ("/", "_"), ("\\", "_"), (":", "_"), (";", "_"),
        (",", "_"), ("|", "_"), ("(", "_"), (")", "_"), ("[", "_"), ("]", "_")
    ]:
        x = x.replace(a, b)
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_")


def looks_like_barcode(values, n_check=100):
    vals = [str(x).strip() for x in values[:n_check]]
    if len(vals) == 0:
        return 0.0
    score = 0.0
    for v in vals:
        if v.endswith(("-1", "-2", "-3", "-4")):
            score += 2.0
        if len(v) >= 8:
            score += 0.5
        if any(ch.isalpha() for ch in v) and any(ch.isdigit() for ch in v):
            score += 0.5
        if "_" not in v and " " not in v:
            score += 0.2
    return score / len(vals)


def looks_like_gene(values, n_check=100):
    vals = [str(x).strip() for x in values[:n_check]]
    if len(vals) == 0:
        return 0.0
    score = 0.0
    for v in vals:
        if len(v) <= 20:
            score += 0.5
        if any(ch.isalpha() for ch in v):
            score += 0.5
        if not v.endswith(("-1", "-2", "-3", "-4")):
            score += 0.5
        if " " not in v:
            score += 0.2
    return score / len(vals)


def detect_matrix_orientation(df):
    row_names = df.index.astype(str).tolist()
    col_names = df.columns.astype(str).tolist()

    row_bar = looks_like_barcode(row_names)
    col_bar = looks_like_barcode(col_names)
    row_gene = looks_like_gene(row_names)
    col_gene = looks_like_gene(col_names)

    print("[MATRIX] row barcode score =", round(row_bar, 3))
    print("[MATRIX] col barcode score =", round(col_bar, 3))
    print("[MATRIX] row gene score    =", round(row_gene, 3))
    print("[MATRIX] col gene score    =", round(col_gene, 3))

    if row_gene > col_gene and col_bar > row_bar:
        return "genes_by_cells"
    if col_gene > row_gene and row_bar > col_bar:
        return "cells_by_genes"

    return "genes_by_cells" if df.shape[0] > df.shape[1] else "cells_by_genes"


def load_rna_from_csv_gz(csv_path):
    print(f"[LOAD] reading matrix: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, index_col=0, compression="gzip")

    orientation = detect_matrix_orientation(df)
    print(f"[LOAD] detected orientation = {orientation}")
    print(f"[LOAD] raw shape = {df.shape}")

    if orientation == "genes_by_cells":
        gene_names = df.index.astype(str).to_numpy()
        cell_ids = df.columns.astype(str).to_numpy()
        X = df.to_numpy(dtype=np.float32).T
    else:
        cell_ids = df.index.astype(str).to_numpy()
        gene_names = df.columns.astype(str).to_numpy()
        X = df.to_numpy(dtype=np.float32)

    # gene 去重
    s = pd.Series(gene_names, dtype="string")
    if s.duplicated(keep=False).any():
        counts = {}
        new = []
        for g in gene_names:
            if g not in counts:
                counts[g] = 0
                new.append(g)
            else:
                counts[g] += 1
                new.append(f"{g}-{counts[g]}")
        gene_names = np.array(new, dtype=str)

    X = np.rint(X).astype(COUNTS_DTYPE, copy=False)

    print(f"[LOAD] final shape = {X.shape} (cells × genes)")
    print(f"[LOAD] first 5 cell_ids = {cell_ids[:5].tolist()}")
    print(f"[LOAD] first 5 genes    = {gene_names[:5].tolist()}")

    return X, gene_names, cell_ids


def load_metadata(meta_path):
    print(f"[META] reading metadata: {os.path.basename(meta_path)}")
    meta = pd.read_csv(meta_path, sep="\t", compression="gzip", index_col=0)
    print(f"[META] shape = {meta.shape}")
    print(f"[META] columns = {meta.columns.tolist()}")
    print("[META] head:")
    print(meta.head(3))

    meta = meta.copy()
    meta["__cell_id__"] = meta.index.astype(str).str.strip()

    sample_col = "sample_name"
    if sample_col not in meta.columns:
        raise ValueError(f"metadata 中找不到 sample 列: {sample_col}")

    meta[sample_col] = meta[sample_col].astype(str).str.strip()

    print("[META] sample_name groups:")
    print(meta[sample_col].value_counts())

    return meta, "__cell_id__", sample_col


def normalize_log1p(X_counts, target_sum=1e4):
    depth = X_counts.sum(axis=1).astype(np.float32)
    denom = np.maximum(depth, 1.0)
    X_norm = (X_counts.astype(np.float32) / denom[:, None]) * float(target_sum)
    X_log1p = np.log1p(X_norm).astype(np.float32)
    return X_log1p, depth


def get_gene_index(gene_names, target_gene):
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    if target_gene not in gene_to_idx:
        upper_map = {g.upper(): i for i, g in enumerate(gene_names)}
        if target_gene.upper() in upper_map:
            return upper_map[target_gene.upper()]
        raise ValueError(f"目标基因 {target_gene} 不在 gene_names 中。")
    return gene_to_idx[target_gene]


def select_hvg(X_log1p, gene_names, n_hvg=2000, force_include=None):
    var = X_log1p.var(axis=0)
    n_hvg = min(n_hvg, X_log1p.shape[1])
    idx = np.argsort(-var)[:n_hvg]

    if force_include is not None and force_include not in idx:
        idx = np.unique(np.concatenate([idx, np.array([force_include], dtype=int)]))

    idx = np.sort(idx)
    return idx, gene_names[idx]


def clip_nonnegative(X):
    X = X.copy()
    X[X < 0] = 0
    return X


def robust_l2_normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.sqrt((v ** 2).sum()) + eps)
    return v / n


def compute_global_target_association(X_log1p, gidx):
    """
    只用 control 群体统计，估计 target gene 与其他基因的全局关联方向。
    不返回任何 cell-specific target activity。
    """
    xg = X_log1p[:, gidx].astype(np.float32)
    xg_center = xg - xg.mean()

    X_center = X_log1p - X_log1p.mean(axis=0, keepdims=True)
    denom_gene = np.sqrt((X_center ** 2).sum(axis=0)) + 1e-6
    denom_target = np.sqrt((xg_center ** 2).sum()) + 1e-6

    corr = (xg_center[:, None] * X_center).sum(axis=0) / (denom_target * denom_gene)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    corr[gidx] = 0.0
    return corr


def save_method_output(
    method_name,
    X_pred_log1p,
    gene_names,
    cell_ids,
    meta_df,
    base_out_dir,
    target_gene,
    source_sample_name
):
    method_dir = os.path.join(base_out_dir, sanitize_name(method_name))
    os.makedirs(method_dir, exist_ok=True)

    pred_path = os.path.join(method_dir, f"{method_name}.{target_gene}.virtual_ko.log1p.npy")
    meta_path = os.path.join(method_dir, f"{method_name}.{target_gene}.virtual_ko.meta.tsv")
    info_path = os.path.join(method_dir, f"{method_name}.{target_gene}.run_info.json")
    gene_path = os.path.join(method_dir, "genes.tsv")
    cell_path = os.path.join(method_dir, "cell_ids.tsv")

    np.save(pred_path, X_pred_log1p.astype(SAVE_FLOAT_DTYPE))
    pd.DataFrame({"gene": gene_names}).to_csv(gene_path, sep="\t", index=False)
    pd.DataFrame({"cell_id": cell_ids}).to_csv(cell_path, sep="\t", index=False)

    meta_out = meta_df.copy()
    meta_out["virtual_ko_method"] = method_name
    meta_out["virtual_ko_target_gene"] = target_gene
    meta_out["virtual_ko_source_sample"] = source_sample_name
    meta_out.to_csv(meta_path, sep="\t", index=False)

    info = {
        "method_name": method_name,
        "target_gene": target_gene,
        "source_sample_name": source_sample_name,
        "n_cells": int(X_pred_log1p.shape[0]),
        "n_genes": int(X_pred_log1p.shape[1]),
        "pred_path": pred_path,
        "meta_path": meta_path,
        "gene_path": gene_path,
        "cell_path": cell_path,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {method_name} done -> {method_dir}")
    return {
        "method_name": method_name,
        "method_dir": method_dir,
        "pred_path": pred_path,
        "meta_path": meta_path,
        "info_path": info_path,
    }


# =========================
# 2) 三种 baseline（去掉 cell-wise target leakage）
# =========================
def virtual_ko_scgpt_style(X_log1p, gene_names, target_gene):
    """
    scGPT-style baseline:
    - 把 cell expression 当成 token-like gene expression 向量
    - 对 target gene 做 mask/zero
    - 用 PCA 做一个轻量“上下文重建”
    - 不引入任何监督
    """
    print("[scGPT-style] running...")

    X = X_log1p.copy()
    gidx = get_gene_index(gene_names, target_gene)

    X_masked = X.copy()
    if SCGPT_MASK_TO_ZERO:
        X_masked[:, gidx] = 0.0
    else:
        X_masked[:, gidx] = float(X_masked[:, gidx].mean())

    pca_dim = min(PCA_DIM, max(2, min(X_masked.shape[0] - 1, X_masked.shape[1] - 1)))
    pca = PCA(n_components=pca_dim, random_state=0)
    Z = pca.fit_transform(X_masked)
    X_rec = pca.inverse_transform(Z).astype(np.float32)

    # target gene 强制 KO
    X_rec[:, gidx] = 0.0
    X_rec = clip_nonnegative(X_rec)
    return X_rec


def virtual_ko_cpa_style(X_log1p, gene_names, target_gene):
    """
    CPA-style baseline（去泄露版）:
    - basal state: 在 target mask 后的 control manifold 上做 PCA
    - perturbation effect: 从 control 群体统计估一个“全局共享”的 KO effect 向量
    - 不再使用每个细胞自己的 EGFR 原始表达做 gate
    """
    print("[CPA-style] running...")

    X = X_log1p.copy()
    gidx = get_gene_index(gene_names, target_gene)

    # 1) 先 mask target，只在 non-target 空间估 basal state
    X_masked = X.copy()
    X_masked[:, gidx] = 0.0

    pca_dim = min(PCA_DIM, max(2, min(X_masked.shape[0] - 1, X_masked.shape[1] - 1)))
    pca = PCA(n_components=pca_dim, random_state=0)
    Z = pca.fit_transform(X_masked)
    X_basal = pca.inverse_transform(Z).astype(np.float32)

    # 2) 用 control 群体统计估 target 对其他基因的全局关联方向
    corr = compute_global_target_association(X, gidx)

    # 用更稳健的方式控制 effect 大小
    abs_corr = np.abs(corr)
    clip_thr = np.quantile(abs_corr[abs_corr > 0], CPA_EFFECT_CLIP_QUANTILE) if np.any(abs_corr > 0) else 1.0
    corr_clipped = np.clip(corr, -clip_thr, clip_thr)

    effect_dir = robust_l2_normalize(corr_clipped)
    effect_dir[gidx] = 0.0

    # 3) 用 control 群体中 target 的稳健统计量作为全局 KO 强度
    #    注意：这是“全局共享强度”，不是 cell-wise gate
    target_strength = float(np.median(X[:, gidx]))
    global_effect = (CPA_EFFECT_SCALE * target_strength) * effect_dir[None, :]

    # KO：沿正相关方向做负向推
    X_pred = X_basal - global_effect

    # target gene 强制 KO
    X_pred[:, gidx] = 0.0
    X_pred = clip_nonnegative(X_pred)
    return X_pred.astype(np.float32)


def virtual_ko_gears_style(X_log1p, gene_names, target_gene):
    """
    GEARS-style baseline（去泄露版）:
    - 建一个简化版 gene-gene 图（只基于 control 群体统计）
    - 找到 EGFR 的邻居基因
    - 用“全局共享强度”做图传播，不再用每个细胞自己的 EGFR activity
    """
    print("[GEARS-style] running...")

    X = X_log1p.copy()
    gidx = get_gene_index(gene_names, target_gene)

    # 1) 只基于 control 群体统计计算 EGFR 对其他基因的相关性
    corr = compute_global_target_association(X, gidx)

    abs_corr = np.abs(corr)
    nbr_idx = np.argsort(-abs_corr)[:min(GEARS_NEIGHBOR_GENES, len(abs_corr))]
    nbr_idx = np.unique(np.concatenate([nbr_idx, np.array([gidx], dtype=int)]))

    X_pred = X.copy()

    # 2) target gene 直接 knockout
    X_pred[:, gidx] = 0.0

    # 3) 用群体统计确定全局传播强度，而不是 cell-wise target activity
    global_strength = float(np.quantile(X[:, gidx], GEARS_STRENGTH_QUANTILE))

    # 4) 对邻居基因做共享传播
    for j in nbr_idx:
        if j == gidx:
            continue
        weight = float(corr[j]) * GEARS_PROPAGATION_ALPHA
        X_pred[:, j] = X_pred[:, j] - weight * global_strength

    X_pred = clip_nonnegative(X_pred)
    return X_pred.astype(np.float32)


# =========================
# 3) 主流程
# =========================
def main():
    # ---------- 读入 ----------
    X_counts, gene_names, cell_ids = load_rna_from_csv_gz(COUNT_MATRIX)
    meta, cell_col, sample_col = load_metadata(META_FILE)

    # ---------- 对齐 ----------
    meta2 = meta.drop_duplicates(subset=[cell_col]).copy().set_index(cell_col)
    common_cells = pd.Index(cell_ids).intersection(meta2.index)

    print(f"[ALIGN] matrix cells = {len(cell_ids)}")
    print(f"[ALIGN] meta cells   = {meta.shape[0]}")
    print(f"[ALIGN] common cells = {len(common_cells)}")

    if len(common_cells) == 0:
        raise ValueError("matrix 和 metadata 没有对齐上的 cell。")

    keep_mask = pd.Index(cell_ids).isin(common_cells)
    X_counts = X_counts[keep_mask, :]
    cell_ids = cell_ids[keep_mask]
    meta_aligned = meta2.loc[cell_ids].reset_index()
    assert np.all(meta_aligned[cell_col].astype(str).to_numpy() == cell_ids)

    # ---------- 只取 control ----------
    sample_values = meta_aligned[sample_col].astype(str).to_numpy()
    control_idx = np.where(sample_values == str(CONTROL_SAMPLE_NAME))[0]
    if len(control_idx) == 0:
        raise ValueError(f"找不到 control sample_name={CONTROL_SAMPLE_NAME}")

    X_control_counts = X_counts[control_idx, :]
    meta_control = meta_aligned.iloc[control_idx].copy().reset_index(drop=True)
    cell_ids_control = cell_ids[control_idx]

    control_label = SAMPLE_NAME_MAP.get(CONTROL_SAMPLE_NAME, f"PC9_{CONTROL_SAMPLE_NAME}")
    control_label = sanitize_name(control_label)

    print(f"[CONTROL] selected sample = {CONTROL_SAMPLE_NAME} -> {control_label}")
    print(f"[CONTROL] n_cells = {X_control_counts.shape[0]}, n_genes = {X_control_counts.shape[1]}")

    # ---------- log1p norm ----------
    X_control_log1p, depth = normalize_log1p(X_control_counts, target_sum=TARGET_SUM)

    # ---------- HVG ----------
    gidx = get_gene_index(gene_names, TARGET_GENE)
    hvg_idx, hvg_gene_names = select_hvg(X_control_log1p, gene_names, n_hvg=N_HVG, force_include=gidx)
    X_hvg = X_control_log1p[:, hvg_idx]

    # 重置 target gene index 到 HVG 空间
    target_gene_hvg_idx = get_gene_index(hvg_gene_names, TARGET_GENE)
    print(f"[HVG] selected genes = {len(hvg_gene_names)}")
    print(f"[HVG] target gene idx in HVG = {target_gene_hvg_idx}")

    # ---------- 保存 control reference ----------
    ref_dir = os.path.join(OUT_DIR, "reference_control")
    os.makedirs(ref_dir, exist_ok=True)
    np.save(os.path.join(ref_dir, f"{control_label}.log1p.npy"), X_hvg.astype(np.float32))
    pd.DataFrame({"gene": hvg_gene_names}).to_csv(os.path.join(ref_dir, "genes.tsv"), sep="\t", index=False)
    pd.DataFrame({"cell_id": cell_ids_control}).to_csv(os.path.join(ref_dir, "cell_ids.tsv"), sep="\t", index=False)
    meta_control.to_csv(os.path.join(ref_dir, "metadata.tsv"), sep="\t", index=False)
    np.save(os.path.join(ref_dir, "depth.npy"), depth.astype(np.float32))

    # ---------- 三种方法 ----------
    results = []

    # 1) scGPT-style
    X_scgpt = virtual_ko_scgpt_style(X_hvg, hvg_gene_names, TARGET_GENE)
    results.append(
        save_method_output(
            method_name="scGPT_style_noleak",
            X_pred_log1p=X_scgpt,
            gene_names=hvg_gene_names,
            cell_ids=cell_ids_control,
            meta_df=meta_control,
            base_out_dir=OUT_DIR,
            target_gene=TARGET_GENE,
            source_sample_name=control_label
        )
    )

    # 2) CPA-style
    X_cpa = virtual_ko_cpa_style(X_hvg, hvg_gene_names, TARGET_GENE)
    results.append(
        save_method_output(
            method_name="CPA_style_noleak",
            X_pred_log1p=X_cpa,
            gene_names=hvg_gene_names,
            cell_ids=cell_ids_control,
            meta_df=meta_control,
            base_out_dir=OUT_DIR,
            target_gene=TARGET_GENE,
            source_sample_name=control_label
        )
    )

    # 3) GEARS-style
    X_gears = virtual_ko_gears_style(X_hvg, hvg_gene_names, TARGET_GENE)
    results.append(
        save_method_output(
            method_name="GEARS_style_noleak",
            X_pred_log1p=X_gears,
            gene_names=hvg_gene_names,
            cell_ids=cell_ids_control,
            meta_df=meta_control,
            base_out_dir=OUT_DIR,
            target_gene=TARGET_GENE,
            source_sample_name=control_label
        )
    )

    # ---------- 汇总 ----------
    summary = pd.DataFrame(results)
    summary_path = os.path.join(OUT_DIR, f"virtual_ko_{TARGET_GENE}_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)

    config = {
        "COUNT_MATRIX": COUNT_MATRIX,
        "META_FILE": META_FILE,
        "CONTROL_SAMPLE_NAME": CONTROL_SAMPLE_NAME,
        "CONTROL_LABEL": control_label,
        "TARGET_GENE": TARGET_GENE,
        "TARGET_SUM": TARGET_SUM,
        "N_HVG": N_HVG,
        "PCA_DIM": PCA_DIM,
        "GEARS_NEIGHBOR_GENES": GEARS_NEIGHBOR_GENES,
        "GEARS_PROPAGATION_ALPHA": GEARS_PROPAGATION_ALPHA,
        "GEARS_STRENGTH_QUANTILE": GEARS_STRENGTH_QUANTILE,
        "CPA_EFFECT_SCALE": CPA_EFFECT_SCALE,
        "CPA_EFFECT_CLIP_QUANTILE": CPA_EFFECT_CLIP_QUANTILE,
        "SCGPT_MASK_TO_ZERO": SCGPT_MASK_TO_ZERO,
        "n_control_cells": int(X_control_counts.shape[0]),
        "n_hvg_genes": int(len(hvg_gene_names)),
        "summary_path": summary_path,
        "note": (
            "Only control sample_name==0 is used. "
            "No supervised labels are used. "
            "CPA/GEARS are modified to avoid cell-wise target leakage by using only global target statistics."
        ),
    }
    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[DONE] virtual KO baseline finished.")
    print("[DONE] summary ->", summary_path)
    print("[DONE] methods ->", [x["method_name"] for x in results])
    print("=" * 100)


if __name__ == "__main__":
    main()
