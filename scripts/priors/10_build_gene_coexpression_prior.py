# Source: RNA共表达边构建(3).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: build per-dataset Gene-Gene coexpression edges from WORLD dense RNA (AUTO gene key mapping)
#       + signed corr weight (w0=corr) + all cells + E drive
#       + gene filtering by detection rate (MIN_DETECT_RATE) + variance (MIN_VAR)
#       + chunked multi-thread parallel correlation (tunable chunk/threads/BLAS threads)
#       + dataset list by scanning WORLD_ROOT
#       + output to assets/G0_full49_v1
#       + NEW: skip datasets that already have output in assets/G0_full49_v1 =====

import os, json, gc, re
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# ============================================================
# 0) Paths  (C盘：数据集与输出都在这里)
# ============================================================
BASE = get_world_root()     # ✅ E盘 -> C盘
WORLD_ROOT = BASE               # ✅ 这里下面直接是 PC-Tissue2/PC-Tissue3/PDAC-Cellline 等数据集文件夹

OUT_REPORT = BASE / "networks" / "build_gene_gene_coexpr_report.tsv"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1) Output config
# ============================================================
G0_SUBDIR = "G0_full49_v1"

OUT_FWD_NAME = "edges_gene_gene_coexpr.tsv"
REL_TYPE_FWD = "gene_gene_coexpr"

WRITE_REV = False
OUT_REV_NAME = "edges_gene_gene_coexpr_rev.tsv"
REL_TYPE_REV = "gene_gene_coexpr_rev"

# ✅ 只要目标文件已存在就跳过
SKIP_IF_EXISTS = True
OVERWRITE = False  # 保持 False 更安全（反正存在就跳过）

# ============================================================
# 2) Where to find WORLD dense RNA (rglob to support nested)
# ============================================================
RNA_FILE_PATTERNS = [
    "*X_rna_log1p_norm_dense.npy",
    "*X_rna_counts_dense.npy",
]
META_PATTERNS = [
    "*.rna_meta.npz",
]

# ============================================================
# 3) Coexpression hyperparams
# ============================================================
MAX_GENES = None
MIN_VAR = 0.0

MIN_DETECT_RATE = 0.01
DETECT_ON = 0.0

TOPK_PER_GENE = 50
MIN_CORR = 0.2
ABS_CORR = True

MAX_CELLS = None
SEED = 7

SKIP_DATASET_NAMES = {"新建文件夹"}

# ============================================================
# 3.5) Parallel / chunk config
# ============================================================
PARALLEL = True
N_WORKERS = 32
CHUNK_COLS = 512
BLAS_THREADS = 8

NODE_INDEX_NAME = "node_index.json"
BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp)$", re.I)

# ============================================================
# 4) Helpers
# ============================================================
def list_datasets(world_root: Path):
    ds = []
    for p in sorted(world_root.iterdir(), key=lambda x: x.name):
        if p.is_dir() and (not BAD_DS_RE.match(p.name)) and (p.name not in SKIP_DATASET_NAMES):
            ds.append(p.name)
    return ds

def pick_assets_dir(dataset_root: Path):
    need = ["genes.tsv", "node_index.json"]
    direct = dataset_root / "assets"
    if direct.is_dir() and all((direct / x).exists() for x in need):
        return direct, "direct"
    cands = []
    for a in dataset_root.rglob("assets"):
        if not a.is_dir():
            continue
        if all((a / x).exists() for x in need):
            peaks = a / "peaks.bed"
            sz = peaks.stat().st_size if peaks.exists() else -1
            cands.append((sz, a))
    if not cands:
        return None, "missing"
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1], "recursive_best_peaks"

def find_first_by_patterns(root: Path, patterns):
    hits = []
    for pat in patterns:
        hits.extend(root.rglob(pat))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_size if p.exists() else -1, reverse=True)
    return hits[0]

def load_world_matrix(npypath: Path):
    X = np.load(npypath, mmap_mode="r")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape} from {npypath}")
    return X

def load_gene_names_from_meta(metapath: Path):
    z = np.load(metapath, allow_pickle=True)
    if "gene_names" not in z:
        raise ValueError(f"meta missing gene_names: {metapath}")
    return z["gene_names"].astype(str)

def load_node_index(assets_dir: Path):
    p = assets_dir / NODE_INDEX_NAME
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _canon_gene(g: str):
    g = str(g).strip()
    out = [g]
    if "." in g:
        out.append(g.split(".", 1)[0])
    out.append(g.upper())
    if "." in g:
        out.append(g.split(".", 1)[0].upper())
    out = [re.sub(r"\s+", "", x) for x in out]
    seen, res = set(), []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res

def infer_gene_key_template(node_index: dict, gene_names: np.ndarray, sample_n=800, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(gene_names)
    take = min(sample_n, n)
    idx = rng.choice(n, size=take, replace=False)
    sample = gene_names[idx]
    prefixes = ["", "GENE:", "Gene:", "gene:", "G:", "g:", "HGNC:", "hgnc:"]
    best_pref, best_hits = "", -1
    for pref in prefixes:
        hits = 0
        for g in sample:
            ok = False
            for gg in _canon_gene(g):
                if (pref + gg) in node_index:
                    ok = True
                    break
            hits += int(ok)
        if hits > best_hits:
            best_hits = hits
            best_pref = pref
    hit_rate = best_hits / float(take)
    return best_pref, hit_rate, {"sample_n": take, "hits": best_hits}

def map_gene_names_to_ids_auto(gene_names: np.ndarray, node_index: dict, min_hit_rate=0.05):
    pref, hit_rate, stat = infer_gene_key_template(node_index, gene_names)
    print(f"  [MAP] inferred gene key prefix='{pref}'  hit_rate≈{hit_rate:.3f}  (hits={stat['hits']}/{stat['sample_n']})")
    ids = np.full((len(gene_names),), -1, dtype=np.int64)
    miss = 0
    for i, g in enumerate(gene_names):
        found = None
        for gg in _canon_gene(g):
            v = node_index.get(pref + gg, None)
            if v is not None:
                found = int(v)
                break
        if found is None:
            miss += 1
        else:
            ids[i] = found
    return ids, miss, pref, hit_rate

def maybe_downsample_cells(X, max_cells=MAX_CELLS, seed=SEED):
    n = X.shape[0]
    if max_cells is None or n <= max_cells:
        return X, {"cells_used": int(n), "cells_total": int(n), "downsample": False}
    rng = np.random.default_rng(seed)
    take = rng.choice(n, size=int(max_cells), replace=False)
    take.sort()
    return X[take, :], {"cells_used": int(len(take)), "cells_total": int(n), "downsample": True}

def select_genes_by_variance(
    X,
    gene_ids,
    max_genes=MAX_GENES,
    min_var=MIN_VAR,
    min_detect_rate=MIN_DETECT_RATE,
    detect_on=DETECT_ON,
):
    Xf = X.astype(np.float32, copy=False)
    det = (Xf > float(detect_on)).mean(axis=0)
    var = Xf.var(axis=0)
    keep = (det >= float(min_detect_rate)) & (var >= float(min_var))
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return None, None, {"kept": 0}
    if (max_genes is not None) and (max_genes > 0) and (len(idx) > max_genes):
        pick = idx[np.argsort(-var[idx])[:max_genes]]
        idx = np.sort(pick)
    return X[:, idx], gene_ids[idx], {
        "kept": int(len(idx)),
        "det_p50": float(np.quantile(det[idx], 0.5)),
        "var_p50": float(np.quantile(var[idx], 0.5)),
    }

def compute_topk_corr_edges_blockwise(
    X,
    gene_ids,
    topk=TOPK_PER_GENE,
    min_corr=MIN_CORR,
    abs_corr=ABS_CORR,
    parallel=PARALLEL,
    n_workers=N_WORKERS,
    chunk_cols=CHUNK_COLS,
    blas_threads=BLAS_THREADS,
):
    X = X.astype(np.float32, copy=False)
    n_cells, n_genes = X.shape

    mu = X.mean(axis=0)
    Xc = X - mu
    var = (Xc * Xc).mean(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    Z = Xc / std

    top_idx = np.empty((n_genes, topk), dtype=np.int32)
    top_val = np.full((n_genes, topk), -np.inf, dtype=np.float32)

    def _topk_one_col(col: np.ndarray, j: int):
        col = col.astype(np.float32, copy=False)
        col[j] = -np.inf
        kk = min(topk, n_genes - 1)
        if abs_corr:
            key = np.abs(col)
            idx = np.argpartition(-key, kth=kk - 1)[:kk]
            vals = col[idx]
            order = np.argsort(-np.abs(vals))
        else:
            idx = np.argpartition(-col, kth=kk - 1)[:kk]
            vals = col[idx]
            order = np.argsort(-vals)
        idx = idx[order]
        vals = vals[order]
        return idx.astype(np.int32, copy=False), vals.astype(np.float32, copy=False)

    def _process_chunk(j0: int, j1: int):
        B = Z[:, j0:j1]
        C = (Z.T @ B) / float(n_cells)

        local_idx = np.empty((j1 - j0, topk), dtype=np.int32)
        local_val = np.full((j1 - j0, topk), -np.inf, dtype=np.float32)

        for bi, j in enumerate(range(j0, j1)):
            col = C[:, bi]
            idx, vals = _topk_one_col(col, j)
            local_idx[bi, :len(idx)] = idx
            local_val[bi, :len(vals)] = vals

        del B, C
        return j0, j1, local_idx, local_val

    ranges = [(j0, min(n_genes, j0 + int(chunk_cols))) for j0 in range(0, n_genes, int(chunk_cols))]
    n_chunks = (n_genes + int(chunk_cols) - 1) // int(chunk_cols)
    print(f"  [PAR] genes={n_genes} chunk_cols={chunk_cols} -> chunks={n_chunks} workers={n_workers} blas_threads={blas_threads}")

    ctx = threadpool_limits(limits=int(blas_threads)) if threadpool_limits else None
    if ctx:
        ctx.__enter__()
    try:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
            futs = [ex.submit(_process_chunk, j0, j1) for (j0, j1) in ranges]
            for fut in as_completed(futs):
                j0_, j1_, li, lv = fut.result()
                top_idx[j0_:j1_, :] = li
                top_val[j0_:j1_, :] = lv
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    src_ids, dst_ids, corr_vals = [], [], []
    for j in range(n_genes):
        sj = int(gene_ids[j])
        nbrs = top_idx[j]
        vals = top_val[j]
        for k in range(topk):
            v = float(vals[k])
            if not np.isfinite(v):
                continue
            if abs_corr:
                if abs(v) < float(min_corr):
                    continue
            else:
                if v < float(min_corr):
                    continue
            src_ids.append(sj)
            dst_ids.append(int(gene_ids[int(nbrs[k])]))
            corr_vals.append(v)

    return pd.DataFrame({
        "src_id": np.asarray(src_ids, dtype=np.int64),
        "dst_id": np.asarray(dst_ids, dtype=np.int64),
        "rel_type": REL_TYPE_FWD,
        "w0": np.asarray(corr_vals, dtype=np.float32),
    })

# ============================================================
# 5) Run per dataset (scan + skip if exists)
# ============================================================
assert WORLD_ROOT.exists(), f"WORLD_ROOT not found: {WORLD_ROOT}"

datasets = list_datasets(WORLD_ROOT)
rows = []

for ds in datasets:
    ds_root = WORLD_ROOT / ds

    assets_dir, mode = pick_assets_dir(ds_root)
    if assets_dir is None:
        rows.append({"dataset": ds, "status": "skip_no_assets", "mode": mode})
        continue

    out_g0 = assets_dir / G0_SUBDIR
    out_fwd = out_g0 / OUT_FWD_NAME

    # ✅ 只检测 G0_full49_v1 里有没有这个输出文件；有就跳过
    if SKIP_IF_EXISTS and out_fwd.exists():
        rows.append({
            "dataset": ds,
            "status": "skip_already_built",
            "assets_dir": str(assets_dir),
            "mode": mode,
            "g0_dir": str(out_g0),
            "out_fwd": str(out_fwd),
        })
        print(f"[skip] {ds:<22} already exists -> {out_fwd}")
        continue

    proc_rna_root = ds_root / "processed" / "RNA"
    if not proc_rna_root.exists():
        proc_rna_root = ds_root

    mat_path = find_first_by_patterns(proc_rna_root, RNA_FILE_PATTERNS)
    meta_path = find_first_by_patterns(proc_rna_root, META_PATTERNS)

    if mat_path is None:
        rows.append({"dataset": ds, "status": "fail_no_world_matrix", "search_root": str(proc_rna_root)})
        print(f"[fail] {ds}: cannot find world RNA matrix under {proc_rna_root}")
        continue
    if meta_path is None:
        rows.append({"dataset": ds, "status": "fail_no_rna_meta", "search_root": str(proc_rna_root)})
        print(f"[fail] {ds}: cannot find rna_meta.npz under {proc_rna_root}")
        continue

    out_g0.mkdir(parents=True, exist_ok=True)

    # load world matrix + meta
    try:
        X = load_world_matrix(mat_path)
        gene_names = load_gene_names_from_meta(meta_path)
    except Exception as e:
        rows.append({"dataset": ds, "status": "fail_load_world", "matrix": str(mat_path), "meta": str(meta_path), "error": str(e)})
        print(f"[fail] {ds}: load failed: {e}")
        continue

    if X.shape[1] != len(gene_names):
        rows.append({"dataset": ds, "status": "fail_vocab_mismatch", "X_shape": str(tuple(X.shape)), "n_gene_names": int(len(gene_names))})
        print(f"[fail] {ds}: X.shape[1]={X.shape[1]} != len(gene_names)={len(gene_names)}")
        continue

    # node_index + map genes
    try:
        node_index = load_node_index(assets_dir)
    except Exception as e:
        rows.append({"dataset": ds, "status": "fail_load_node_index", "assets_dir": str(assets_dir), "error": str(e)})
        print(f"[fail] {ds}: load node_index failed: {e}")
        continue

    gene_ids, miss, pref, hit_rate = map_gene_names_to_ids_auto(gene_names, node_index, min_hit_rate=0.01)
    mapped_mask = gene_ids >= 0

    if mapped_mask.sum() < 100:
        rows.append({"dataset": ds, "status": "fail_too_few_mapped_genes", "mapped": int(mapped_mask.sum()), "miss": int(miss), "prefix": pref, "hit_rate": float(hit_rate)})
        print(f"[fail] {ds}: too few mapped genes: {mapped_mask.sum()} (miss={miss}) prefix='{pref}' hit_rate≈{hit_rate:.3f}")
        continue

    X = X[:, mapped_mask]
    gene_ids = gene_ids[mapped_mask]

    X, cell_stat = maybe_downsample_cells(X, max_cells=MAX_CELLS, seed=SEED)

    Xs, gene_ids_s, gstat = select_genes_by_variance(
        X, gene_ids,
        max_genes=MAX_GENES, min_var=MIN_VAR,
        min_detect_rate=MIN_DETECT_RATE, detect_on=DETECT_ON
    )
    if Xs is None or Xs.shape[1] < 50:
        rows.append({"dataset": ds, "status": "fail_too_few_genes_after_filter", "genes_after": 0})
        print(f"[fail] {ds}: too few genes after filter (MIN_DETECT_RATE={MIN_DETECT_RATE}, MIN_VAR={MIN_VAR})")
        continue

    # compute corr edges
    try:
        edges = compute_topk_corr_edges_blockwise(Xs, gene_ids_s)
    except Exception as e:
        rows.append({"dataset": ds, "status": "fail_compute", "error": str(e)})
        print(f"[fail] {ds}: compute failed: {e}")
        continue

    edges.to_csv(out_fwd, sep="\t", index=False)

    if WRITE_REV:
        out_rev = out_g0 / OUT_REV_NAME
        rev = edges.rename(columns={"src_id": "dst_id", "dst_id": "src_id"}).copy()
        rev["rel_type"] = REL_TYPE_REV
        rev.to_csv(out_rev, sep="\t", index=False)

    w = edges["w0"].to_numpy(np.float32)
    rows.append({
        "dataset": ds,
        "status": "ok",
        "assets_dir": str(assets_dir),
        "mode": mode,
        "g0_dir": str(out_g0),
        "dataset_root": str(ds_root),
        "matrix_path": str(mat_path),
        "meta_path": str(meta_path),
        "gene_key_prefix": pref,
        "gene_key_hit_rate": float(hit_rate),
        "cells_total": int(cell_stat["cells_total"]),
        "cells_used": int(cell_stat["cells_used"]),
        "genes_global": int(len(gene_names)),
        "genes_mapped": int(mapped_mask.sum()),
        "genes_used": int(Xs.shape[1]),
        "det_rate_min": float(MIN_DETECT_RATE),
        "min_var": float(MIN_VAR),
        "max_genes": -1 if MAX_GENES is None else int(MAX_GENES),
        "topk_per_gene": int(TOPK_PER_GENE),
        "min_corr": float(MIN_CORR),
        "abs_corr": bool(ABS_CORR),
        "parallel": bool(PARALLEL),
        "n_workers": int(N_WORKERS),
        "chunk_cols": int(CHUNK_COLS),
        "blas_threads": int(BLAS_THREADS),
        "edges_out": int(len(edges)),
        "w_p50": float(np.quantile(w, 0.5)),
        "w_p90": float(np.quantile(w, 0.9)),
        "w_p99": float(np.quantile(w, 0.99)),
        "w_min": float(np.min(w)),
        "w_max": float(np.max(w)),
        "out_fwd": str(out_fwd),
        "det_p50": float(gstat.get("det_p50", np.nan)),
        "var_p50": float(gstat.get("var_p50", np.nan)),
    })

    print(
        f"[ok] {ds:<22} prefix='{pref}' hit≈{hit_rate:.2f}  "
        f"cells={cell_stat['cells_used']:>6}/{cell_stat['cells_total']:<6} "
        f"genes_used={Xs.shape[1]:>5} (det>={MIN_DETECT_RATE}) "
        f"edges={len(edges):,} w_min={np.min(w):.3f} w_p50={np.quantile(w,0.5):.3f} w_max={np.max(w):.3f} -> {out_fwd.name}"
    )

    del X, Xs, edges
    gc.collect()

rep_out = pd.DataFrame(rows)
rep_out.to_csv(OUT_REPORT, sep="\t", index=False)
print("\nSTATUS COUNTS:\n", rep_out["status"].value_counts(dropna=False))
print("\nSaved report:", OUT_REPORT)
print(rep_out.head(30))