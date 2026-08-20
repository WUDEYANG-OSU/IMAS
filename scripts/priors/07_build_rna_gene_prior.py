# Source: 诛仙剑阵4(1).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: build RNA(dataset) -> GENE edges from world processed RNA dense inputs (all samples, concat-cells) =====
import os, json, re
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 0) CONFIG  （保持你原来的不变；只新增 OUT_SUBDIR 并改输出目录）
# ============================================================
BASE = get_project_root()
ROOT = BASE / "LLM-WorldWord"

OUT_REPORT = BASE / "networks" / "build_rna_gene_expr_report.tsv"
REL_TYPE = "rna_gene_expr"
AGG_INPUT = "log1p"     # "log1p" or "counts"
AGG_STAT  = "mean"      # "mean" or "sum"
CHUNK_CELLS = 4096
NORMALIZE_TO_01 = False

# ✅ 新增：每个数据集 assets 下的输出子目录
OUT_SUBDIR = "G0_full49_v1"

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp)$", re.I)

# ============================================================
# 1) helpers  （只在“找 RNA/manifest”这块增强；其余保持原逻辑）
# ============================================================
def pick_assets_dir(dataset_root: Path):
    """优先 dataset_root/assets；否则递归找 **/assets；选 peaks.bed 最大的一套（与你们工程一致）"""
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

def load_node_index(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_node_index(p: Path, node_index: dict):
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(node_index, f, ensure_ascii=False, indent=2)

def read_genes_df(genes_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(genes_tsv, sep="\t", low_memory=False)
    if "symbol" not in df.columns:
        raise ValueError(f"{genes_tsv} missing column 'symbol'. got={list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str)
    return df

def _fast_count_lines(p: Path, max_bytes=200_000_000):
    """
    快速估算行数：优先按 size 做粗筛；再做真正行计数。
    """
    try:
        sz = p.stat().st_size
    except Exception:
        return -1
    # 太大就不精确算也行，但这里你们 manifest 一般几十 MB，可接受
    if sz > max_bytes:
        return int(sz)  # 用 size 当 proxy
    try:
        with open(p, "rb") as f:
            n = 0
            for _ in f:
                n += 1
        return max(n - 1, 0)  # 去掉表头
    except Exception:
        return -1

def find_manifest_in_rna_dir(rna_dir: Path):
    """
    支持 manifest_rna.tsv / manifest.tsv / 以及 manifest*.tsv
    """
    if rna_dir is None or (not rna_dir.is_dir()):
        return None
    for name in ["manifest_rna.tsv", "manifest.tsv", "manifest.txt", "manifest_rna.txt"]:
        p = rna_dir / name
        if p.exists():
            return p
    ms = sorted(rna_dir.glob("manifest*.tsv"))
    return ms[0] if ms else None

def find_best_rna_and_manifest(ds_root: Path, assets_dir: Path):
    """
    ✅ 关键增强：支持二级目录 + 支持 manifest.tsv
    规则（尽量“全样本”）：
      1) ds_root/processed/RNA
      2) assets_dir.parent/processed/RNA （例如 CRC-Cellline/COLO320/... 以及 Glioblastoma/tumor_only/...）
      3) ds_root/**/processed/RNA 全递归
    多个候选时：选 manifest “行数最多”的那个（最代表 all samples）
    """
    cands = []

    # (1) direct dataset root
    r1 = ds_root / "processed" / "RNA"
    m1 = find_manifest_in_rna_dir(r1)
    if m1 is not None:
        cands.append(("direct_ds_root", r1, m1, _fast_count_lines(m1)))

    # (2) assets parent (instance root)
    inst_root = assets_dir.parent if assets_dir is not None else None
    if inst_root is not None:
        r2 = inst_root / "processed" / "RNA"
        m2 = find_manifest_in_rna_dir(r2)
        if m2 is not None:
            cands.append(("assets_parent", r2, m2, _fast_count_lines(m2)))

    # (3) recursive
    for rna_dir in ds_root.rglob("processed"):
        rna = rna_dir / "RNA"
        m = find_manifest_in_rna_dir(rna)
        if m is not None:
            cands.append(("recursive", rna, m, _fast_count_lines(m)))

    if not cands:
        return None, None, "missing"

    # 选行数最大的（proxy: 最多 cell 行）
    cands.sort(key=lambda x: x[3], reverse=True)
    mode, rna_dir, manifest, nlines = cands[0]
    return rna_dir, manifest, mode

def iter_samples_from_manifest(manifest_path: Path):
    """
    manifest 里按 sample_id 去重，返回 sample->(counts, log1p)
    兼容：manifest.tsv / manifest_rna.tsv
    """
    m = pd.read_csv(manifest_path, sep="\t")

    # 兼容列名（尽量不改变你原来的行为：优先用标准列）
    def _pick_col(df, keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    sample_col = _pick_col(m, ["sample_id", "sample", "sid", "sampleID"])
    counts_col = _pick_col(m, ["counts_npy", "counts_path", "counts"])
    log1p_col  = _pick_col(m, ["log1p_npy", "log1p_path", "log1p"])

    if sample_col is None or counts_col is None:
        raise ValueError(f"{manifest_path} missing required cols. got={list(m.columns)}; need sample_id & counts_npy")

    # 你原来就是 first 去重（同一个 sample_id 对应很多 cell 行）
    agg_dict = {counts_col: "first"}
    if log1p_col is not None:
        agg_dict[log1p_col] = "first"

    g = (m.groupby(sample_col, as_index=False).agg(agg_dict))

    out = []
    for _, r in g.iterrows():
        sample = str(r[sample_col])
        counts = Path(str(r[counts_col]))
        log1p = None
        if log1p_col is not None:
            v = r.get(log1p_col, "")
            v = "" if (pd.isna(v)) else str(v)
            if v and v.lower() != "nan":
                log1p = Path(v)
        out.append((sample, counts, log1p))
    return out

def sum_by_gene_dense(npy_path: Path, chunk_cells: int):
    X = np.load(npy_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"{npy_path} not 2D. shape={X.shape}")
    n_cells, n_genes = X.shape
    acc = np.zeros((n_genes,), dtype=np.float64)
    for s in range(0, n_cells, chunk_cells):
        e = min(s + chunk_cells, n_cells)
        blk = np.asarray(X[s:e, :], dtype=np.float32)
        acc += blk.sum(axis=0, dtype=np.float64)
    return acc, int(n_cells), int(n_genes)

def find_or_create_rna_node(node_index: dict, dataset: str, node_index_path: Path):
    """
    ✅ 保持你原来的逻辑：找不到就创建 RNA:{dataset}
    """
    cands = [
        f"RNA:{dataset}", f"rna:{dataset}",
        f"OBS_RNA:{dataset}", f"obs_rna:{dataset}",
        f"RNA|{dataset}", f"rna|{dataset}",
        f"RNA_{dataset}", f"rna_{dataset}",
        f"{dataset}:RNA", f"{dataset}:rna",
    ]
    for k in cands:
        if k in node_index:
            return int(node_index[k]), k, False

    ds_l = dataset.lower()
    for k in node_index.keys():
        kl = str(k).lower()
        if ds_l in kl and "rna" in kl:
            return int(node_index[k]), k, False

    max_id = max(int(v) for v in node_index.values()) if node_index else -1
    new_id = int(max_id + 1)
    new_key = f"RNA:{dataset}"
    node_index[new_key] = new_id
    save_node_index(node_index_path, node_index)
    return new_id, new_key, True

# ============================================================
# 2) main
# ============================================================
assert ROOT.exists(), f"ROOT not found: {ROOT}"

datasets = sorted([d for d in ROOT.iterdir() if d.is_dir()], key=lambda x: x.name)
datasets = [d for d in datasets if not BAD_DS_RE.match(d.name)]

rows = []
for ds_root in datasets:
    ds = ds_root.name

    assets_dir, mode = pick_assets_dir(ds_root)
    if assets_dir is None:
        rows.append({"dataset": ds, "status": "skip_no_assets", "mode": mode})
        continue

    genes_tsv = assets_dir / "genes.tsv"
    node_index_path = assets_dir / "node_index.json"
    if not genes_tsv.exists() or not node_index_path.exists():
        rows.append({"dataset": ds, "status": "skip_missing_genes_or_node_index", "assets_dir": str(assets_dir)})
        continue

    # ✅ 关键：改成“递归找最好的 processed/RNA + manifest（支持 manifest.tsv）”
    rna_dir, manifest, manifest_mode = find_best_rna_and_manifest(ds_root, assets_dir)
    if manifest is None or (not manifest.exists()):
        rows.append({
            "dataset": ds,
            "status": "skip_no_manifest_rna",
            "assets_dir": str(assets_dir),
            "manifest_mode": manifest_mode,
            "rna_dir_used": str(rna_dir) if rna_dir else None
        })
        continue

    node_index = load_node_index(node_index_path)
    genes_df = read_genes_df(genes_tsv)
    symbols = genes_df["symbol"].astype(str).to_numpy()

    # 2.1 RNA(dataset) 节点：找不到就创建（保持原样）
    rna_id, rna_key, created = find_or_create_rna_node(node_index, ds, node_index_path)

    # 2.2 gene 映射：严格 GENE:{symbol}（保持原样）
    gene_ids = np.full((len(symbols),), -1, dtype=np.int64)
    mapped = 0
    for i, sym in enumerate(symbols):
        k = f"GENE:{sym}"
        v = node_index.get(k)
        if v is not None:
            gene_ids[i] = int(v)
            mapped += 1

    if mapped == 0:
        rows.append({
            "dataset": ds,
            "status": "skip_no_gene_mapped_to_node_index",
            "assets_dir": str(assets_dir),
            "gene_total": int(len(symbols)),
            "gene_mapped": 0,
            "hint": "Expected gene keys like 'GENE:{symbol}' in node_index.json (as in RE->Gene(dist) builder)."
        })
        continue

    # 2.3 聚合所有 sample dense（全拼最真实）（保持原样）
    try:
        samples = iter_samples_from_manifest(manifest)
    except Exception as e:
        rows.append({
            "dataset": ds,
            "status": "fail_read_manifest",
            "assets_dir": str(assets_dir),
            "manifest": str(manifest),
            "manifest_mode": manifest_mode,
            "rna_dir_used": str(rna_dir),
            "err": str(e)
        })
        continue

    if len(samples) == 0:
        rows.append({"dataset": ds, "status": "skip_manifest_empty", "manifest": str(manifest), "rna_dir_used": str(rna_dir)})
        continue

    sum_vec = None
    total_cells = 0
    n_used = 0
    used_mode = None
    n_genes_ref = None

    for sample, counts_npy, log1p_npy in samples:
        if AGG_INPUT == "log1p" and (log1p_npy is not None) and log1p_npy.exists():
            src = log1p_npy
            used_mode = "log1p_norm"
        else:
            src = counts_npy
            used_mode = "counts"

        if not src.exists():
            continue

        svec, n_cells, n_genes = sum_by_gene_dense(src, CHUNK_CELLS)
        if n_genes_ref is None:
            n_genes_ref = n_genes
        elif n_genes != n_genes_ref:
            continue

        sum_vec = svec if sum_vec is None else (sum_vec + svec)
        total_cells += n_cells
        n_used += 1

    if sum_vec is None or total_cells == 0 or n_used == 0:
        rows.append({"dataset": ds, "status": "skip_no_valid_sample_npy", "manifest": str(manifest), "rna_dir_used": str(rna_dir)})
        continue

    if n_genes_ref != len(symbols):
        rows.append({
            "dataset": ds,
            "status": "skip_vocab_len_mismatch",
            "genes_tsv_n": int(len(symbols)),
            "dense_n": int(n_genes_ref),
            "hint": "genes.tsv length should match dense matrix gene dimension."
        })
        continue

    if AGG_STAT == "mean":
        agg_vec = (sum_vec / float(total_cells)).astype(np.float32)
    elif AGG_STAT == "sum":
        agg_vec = sum_vec.astype(np.float32)
    else:
        raise ValueError(f"Unknown AGG_STAT={AGG_STAT}")

    if NORMALIZE_TO_01:
        mx = float(np.max(agg_vec))
        if mx > 0:
            agg_vec = agg_vec / mx

    # 2.4 写 edges：RNA(dataset)->Gene
    keep = gene_ids >= 0
    edges = pd.DataFrame({
        "src_id": np.full((int(keep.sum()),), int(rna_id), dtype=np.int64),
        "dst_id": gene_ids[keep].astype(np.int64),
        "rel_type": REL_TYPE,
        "w0": agg_vec[keep].astype(np.float32),
    })

    # ✅ 改输出目录：assets/G0_full49_v1/
    g0 = assets_dir / OUT_SUBDIR
    g0.mkdir(parents=True, exist_ok=True)
    out_edges = g0 / "edges_rna_gene_expr.tsv"
    edges.to_csv(out_edges, sep="\t", index=False)

    w = edges["w0"].to_numpy()
    rows.append({
        "dataset": ds,
        "status": "ok",
        "assets_dir": str(assets_dir),
        "mode": mode,
        "rna_key": rna_key,
        "rna_id": int(rna_id),
        "rna_node_created": bool(created),
        "n_samples_in_manifest": int(len(samples)),
        "n_samples_used": int(n_used),
        "total_cells": int(total_cells),
        "agg_input_used": used_mode,
        "gene_total": int(len(symbols)),
        "gene_mapped": int(mapped),
        "gene_mapped_ratio": float(mapped / max(len(symbols), 1)),
        "edges_out": int(len(edges)),
        "w_p50": float(np.quantile(w, 0.50)),
        "w_p90": float(np.quantile(w, 0.90)),
        "w_p99": float(np.quantile(w, 0.99)),
        "w_max": float(np.max(w)),
        "out_edges": str(out_edges),
        "manifest": str(manifest),
        "manifest_mode": manifest_mode,
        "rna_dir_used": str(rna_dir),
    })

    print(f"[ok] {ds:22s}  RNA={rna_key}{'(+created)' if created else ''}  "
          f"samples_used={n_used:3d}  cells={total_cells:8d}  gene_map={mapped/len(symbols):.3f}  edges={len(edges):7d}")

rep = pd.DataFrame(rows)
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
rep.to_csv(OUT_REPORT, sep="\t", index=False)

print("\nSTATUS COUNTS:\n", rep["status"].value_counts(dropna=False))
print("\nSaved report:", OUT_REPORT)
print(rep.head(30))