# Source: 1.ATAC锚点构建.docx
# Extracted from the user-provided Word code; scientific logic preserved.
from __future__ import annotations

# ============================================================
# build_atac_geneactivity_anchors.py
# ============================================================
# 作用：
#   从每个数据集 processed/ATAC 下的：
#       - *.X_atac_peaks_csr.npz
#       - *.atac_meta.npz
#       - manifest_atac.tsv
#   结合全局：
#       - global_peak_vocab.txt
#       - global_gene_vocab.txt
#       - peak_gene_f.full.npz
#   生成第二段迁移代码需要的每个 sample 一个 geneActivity anchor:
#       processed/ATAC_geneActivity_ALLGENES/<sid>_ATAC_geneActivity_ALLGENES.npz
#
# 输出 npz 至少包含：
#   - genes: object[str], shape=(n_gene_kept,)
#   - cells: object[str], shape=(n_cells,)
#   - X_geneActivity: float32, shape=(n_cells, n_gene_kept)
#
# 说明：
#   1) 这里按 peak->gene prior 做 sum 聚合
#   2) 默认输出“只保留该数据集可达的 gene 列”，不是 18万全基因全展开
#      这和你第二段代码兼容，因为它是按 gene 名查列，不要求全基因都在
#   3) 如果你硬要输出 full global genes，也可以改 KEEP_ONLY_REACHABLE_GENES = False
# ============================================================

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ============================================================
# 0) CONFIG
# ============================================================
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"

GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
MAPS_ROOT = GLOBAL_ROOT / "maps"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"

GLOBAL_PEAK_VOCAB = MAPS_ROOT / "global_peak_vocab.txt"
GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
PEAK_GENE_F = PRIORS_DIR / "peak_gene_f.full.npz"

# None = 跑所有数据集
RUN_ONLY_DATASET = None
# RUN_ONLY_DATASET = "BC-Tissue"

# True: 只输出该数据集本地 peak 可达的 genes（推荐）
# False: 输出 full global genes（非常大，不推荐）
KEEP_ONLY_REACHABLE_GENES = True

# ATAC 是二值时 sum 很自然；也可改成 log1p(sum) / clipped
APPLY_LOG1P_TO_OUTPUT = False

# 稀疏转 dense 时用 float32
OUT_DTYPE = np.float32

# 进度打印
PRINT_EVERY_SAMPLE = True


def log(*args):
    print(*args, flush=True)


# ============================================================
# 1) HELPERS
# ============================================================
def read_vocab_txt(p: Path):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def norm_barcode(x: str) -> str:
    x = str(x)
    return x.split("::", 1)[-1] if "::" in x else x


def load_csr_npz(npz_path: Path) -> sp.csr_matrix:
    z = np.load(npz_path, allow_pickle=True)
    for k in ["indptr", "indices", "data", "shape"]:
        if k not in z:
            raise KeyError(f"{npz_path.name}: missing key {k}")
    X = sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]),
        shape=tuple(z["shape"].tolist())
    )
    return X


def find_datasets():
    ds_list = []
    for p in WORLD_ROOT.iterdir():
        if not p.is_dir():
            continue
        atac_dir = p / "processed" / "ATAC"
        if atac_dir.exists() and (atac_dir / "manifest_atac.tsv").exists():
            ds_list.append(p.name)
    ds_list = sorted(ds_list)
    if RUN_ONLY_DATASET is not None:
        ds_list = [x for x in ds_list if x == RUN_ONLY_DATASET]
    return ds_list


# ============================================================
# 2) LOAD GLOBAL PRIORS
# ============================================================
assert GLOBAL_PEAK_VOCAB.exists(), f"missing: {GLOBAL_PEAK_VOCAB}"
assert GLOBAL_GENE_VOCAB.exists(), f"missing: {GLOBAL_GENE_VOCAB}"
assert PEAK_GENE_F.exists(), f"missing: {PEAK_GENE_F}"

global_peaks = read_vocab_txt(GLOBAL_PEAK_VOCAB)
global_genes = read_vocab_txt(GLOBAL_GENE_VOCAB)

peak2gid = {p: i for i, p in enumerate(global_peaks)}

z_pg = np.load(PEAK_GENE_F, allow_pickle=True)
peak_src = z_pg["src"].astype(np.int64)
gene_dst = z_pg["dst"].astype(np.int64)

assert peak_src.shape == gene_dst.shape
n_edges = len(peak_src)

log("[GLOBAL] n_global_peaks =", len(global_peaks))
log("[GLOBAL] n_global_genes =", len(global_genes))
log("[PRIOR] peak_gene_f edges =", n_edges)
log("[PRIOR] unique peaks in prior =", len(np.unique(peak_src)))
log("[PRIOR] unique genes in prior =", len(np.unique(gene_dst)))
log()

# 建一个 global peak -> list(global gene) 的映射
peak_to_genes = defaultdict(list)
for p, g in zip(peak_src.tolist(), gene_dst.tolist()):
    peak_to_genes[int(p)].append(int(g))


# ============================================================
# 3) BUILD ONE SAMPLE
# ============================================================
def build_one_sample_geneactivity(
    dataset_name: str,
    sid: str,
    atac_dir: Path,
    out_dir: Path,
):
    meta_path = atac_dir / f"{sid}.atac_meta.npz"
    csr_path = atac_dir / f"{sid}.X_atac_peaks_csr.npz"

    assert meta_path.exists(), f"missing: {meta_path}"
    assert csr_path.exists(), f"missing: {csr_path}"

    zm = np.load(meta_path, allow_pickle=True)
    required_keys = ["peak_names", "cell_ids"]
    for k in required_keys:
        assert k in zm, f"{meta_path.name}: missing key {k}"

    peak_names = np.asarray(zm["peak_names"]).astype(str)
    cell_ids = np.asarray(zm["cell_ids"]).astype(str)
    cell_ids = np.asarray([norm_barcode(x) for x in cell_ids.tolist()], dtype=object)

    X_peak = load_csr_npz(csr_path)

    assert X_peak.shape[0] == len(cell_ids), (
        f"{sid}: csr rows {X_peak.shape[0]} != n_cells {len(cell_ids)}"
    )
    assert X_peak.shape[1] == len(peak_names), (
        f"{sid}: csr cols {X_peak.shape[1]} != n_peak_names {len(peak_names)}"
    )

    # local peak -> global peak id
    local2global_peak = np.array([peak2gid.get(p, -1) for p in peak_names], dtype=np.int64)
    matched = (local2global_peak >= 0)
    n_matched = int(matched.sum())

    if n_matched == 0:
        return {
            "dataset": dataset_name,
            "sample_id": sid,
            "status": "skipped_no_global_peak_match",
            "reason": "no local peaks matched global peak vocab",
            "n_cells": int(X_peak.shape[0]),
            "n_local_peaks": int(X_peak.shape[1]),
            "n_local_peaks_used": 0,
            "frac_local_peaks_used": 0.0,
            "n_out_genes": 0,
            "out_path": "",
        }

    # 只保留在 prior 中有 gene 映射的 local peaks
    local_peak_keep = []
    local_peak_keep_gene_lists = []

    for local_col, gpeak in enumerate(local2global_peak.tolist()):
        if gpeak < 0:
            continue
        glist = peak_to_genes.get(int(gpeak), None)
        if glist is None or len(glist) == 0:
            continue
        local_peak_keep.append(local_col)
        local_peak_keep_gene_lists.append(glist)

    if len(local_peak_keep) == 0:
        return {
            "dataset": dataset_name,
            "sample_id": sid,
            "status": "skipped_no_peak_gene_prior_match",
            "reason": "no local peaks mapped to peak->gene prior",
            "n_cells": int(X_peak.shape[0]),
            "n_local_peaks": int(X_peak.shape[1]),
            "n_local_peaks_used": 0,
            "frac_local_peaks_used": 0.0,
            "n_out_genes": 0,
            "out_path": "",
        }

    local_peak_keep = np.asarray(local_peak_keep, dtype=np.int64)

    # 该 sample 可达 genes
    reachable_gene_set = set()
    for glist in local_peak_keep_gene_lists:
        reachable_gene_set.update(glist)

    if KEEP_ONLY_REACHABLE_GENES:
        out_gene_ids = np.asarray(sorted(reachable_gene_set), dtype=np.int64)
    else:
        out_gene_ids = np.arange(len(global_genes), dtype=np.int64)

    out_genes = np.asarray([global_genes[g] for g in out_gene_ids.tolist()], dtype=object)
    gene_global_to_out = {int(g): i for i, g in enumerate(out_gene_ids.tolist())}

    # 构造 peak(local filtered) -> gene(out) 的稀疏映射矩阵 A
    # X_gene = X_peak_filtered @ A
    rows = []
    cols = []
    data = []

    for r_local_filtered, glist in enumerate(local_peak_keep_gene_lists):
        w = 1.0
        for g in glist:
            c = gene_global_to_out.get(int(g), None)
            if c is not None:
                rows.append(r_local_filtered)
                cols.append(c)
                data.append(w)

    A = sp.coo_matrix(
        (np.asarray(data, dtype=np.float32),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(len(local_peak_keep), len(out_gene_ids))
    ).tocsr()

    # 过滤 peak 列
    Xp = X_peak[:, local_peak_keep].tocsr().astype(np.float32, copy=False)

    # 聚合成 gene activity
    Xg = (Xp @ A).astype(np.float32, copy=False)

    if APPLY_LOG1P_TO_OUTPUT:
        Xg = Xg.tocsr(copy=True)
        if Xg.nnz > 0:
            Xg.data = np.log1p(Xg.data)

    Xg_dense = Xg.toarray().astype(OUT_DTYPE, copy=False)

    out_path = out_dir / f"{sid}_ATAC_geneActivity_ALLGENES.npz"
    np.savez_compressed(
        out_path,
        genes=out_genes,
        cells=cell_ids,
        X_geneActivity=Xg_dense,
        gene_ids_global=out_gene_ids,
        source_dataset=np.asarray([dataset_name], dtype=object),
        source_sample_id=np.asarray([sid], dtype=object),
    )

    return {
        "dataset": dataset_name,
        "sample_id": sid,
        "status": "ok",
        "reason": "",
        "n_cells": int(X_peak.shape[0]),
        "n_local_peaks": int(X_peak.shape[1]),
        "n_local_peaks_used": int(len(local_peak_keep)),
        "frac_local_peaks_used": float(len(local_peak_keep) / max(1, X_peak.shape[1])),
        "n_out_genes": int(len(out_gene_ids)),
        "out_path": str(out_path),
    }


# ============================================================
# 4) MAIN
# ============================================================
def main():
    datasets = find_datasets()
    assert len(datasets) > 0, "no datasets found"

    all_rows = []
    skipped_rows = []

    log("[DATASETS]", datasets)
    log()

    for ds in datasets:
        atac_dir = WORLD_ROOT / ds / "processed" / "ATAC"
        manifest = atac_dir / "manifest_atac.tsv"
        out_dir = WORLD_ROOT / ds / "processed" / "ATAC_geneActivity_ALLGENES"
        out_dir.mkdir(parents=True, exist_ok=True)

        assert atac_dir.exists(), f"missing: {atac_dir}"
        assert manifest.exists(), f"missing: {manifest}"

        df = pd.read_csv(manifest, sep="\t")
        assert "sample_id" in df.columns, f"{manifest.name}: missing sample_id"
        sids = sorted(df["sample_id"].astype(str).unique().tolist())

        log("=" * 100)
        log(f"[DATASET] {ds}")
        log(f"[ATAC_DIR] {atac_dir}")
        log(f"[OUT_DIR]  {out_dir}")
        log(f"[SAMPLES]  {sids}")
        log()

        for sid in sids:
            row = build_one_sample_geneactivity(
                dataset_name=ds,
                sid=sid,
                atac_dir=atac_dir,
                out_dir=out_dir,
            )
            all_rows.append(row)

            if row["status"] == "ok":
                if PRINT_EVERY_SAMPLE:
                    log(
                        f"[DONE] {ds}/{sid} | "
                        f"cells={row['n_cells']} "
                        f"peaks_used={row['n_local_peaks_used']}/{row['n_local_peaks']} "
                        f"({row['frac_local_peaks_used']:.4f}) "
                        f"genes={row['n_out_genes']}"
                    )
            else:
                skipped_rows.append(row)
                log(f"[SKIP] {ds}/{sid} | reason={row['reason']}")

        ds_df = pd.DataFrame([r for r in all_rows if r["dataset"] == ds])
        ds_csv = out_dir / "geneactivity_build_summary.tsv"
        ds_df.to_csv(ds_csv, sep="\t", index=False)
        log(f"[SAVE] {ds_csv}")
        log()

    final_df = pd.DataFrame(all_rows)
    final_csv = BASE / "runs" / "geneactivity_anchor_build_summary.tsv"
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_csv, sep="\t", index=False)

    skipped_csv = BASE / "runs" / "geneactivity_anchor_skipped_samples.tsv"
    pd.DataFrame(skipped_rows).to_csv(skipped_csv, sep="\t", index=False)

    log("=" * 100)
    log("[FINAL SAVE]", final_csv)
    log("[SKIPPED SAVE]", skipped_csv)

    if len(skipped_rows) > 0:
        log()
        log("[SKIPPED SAMPLES]")
        for r in skipped_rows:
            log(f"  - {r['dataset']}/{r['sample_id']} | {r['reason']}")
    else:
        log("[SKIPPED SAMPLES] none")

    log("[ALL DONE]")


if __name__ == "__main__":
    main()
