# Source: 诛仙剑阵5(1).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: build RNA(dataset) -> RE edges from (RNA->GENE) + (RE->GENE(dist)) =====
import os, re, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 0) CONFIG
# ============================================================
BASE = get_project_root()
ROOT = BASE / "LLM-WorldWord"

REL_TYPE = "rna_re_expr"          # 输出 rel_type
TOPM_PER_DATASET = 250000         # 每个 dataset 最多保留多少条 RNA->RE（按 weight 降序截断）
NORMALIZE_PER_RNA = True          # 是否让每个 dataset 的 RNA->RE 权重和=1（推荐 True）
EPS = 1e-12

OUT_REPORT = BASE / "networks" / "build_rna_re_expr_report.tsv"

# ✅ 改：统一用这个输出/输入子目录
G0_SUBDIR = "G0_full49_v1"
OUT_EDGES_NAME = "edges_rna_re_expr.tsv"

# ✅ 特别关照的数据集
SPECIAL_DS = {"BC-Tissue2"}

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp)$", re.I)

# ============================================================
# 1) helpers: assets selection (same style as your other edges)
# ============================================================
def pick_assets_dir(dataset_root: Path):
    """
    优先 dataset_root/assets；否则递归找 **/assets；选 peaks.bed 最大的一套（与你们工程一致）
    要求：genes.tsv + node_index.json + peaks.bed
    """
    need = ["genes.tsv", "node_index.json", "peaks.bed"]
    direct = dataset_root / "assets"
    if direct.is_dir() and all((direct / x).exists() for x in need):
        return direct, "direct"

    cands = []
    for a in dataset_root.rglob("assets"):
        if not a.is_dir():
            continue
        if all((a / x).exists() for x in need):
            peaks = a / "peaks.bed"
            try:
                sz = peaks.stat().st_size
            except Exception:
                sz = -1
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

def ensure_node(node_index: dict, key: str, node_index_path: Path):
    """
    找不到就创建。这里用于 RNA:{dataset}（通常你们之前已创建过）
    """
    if key in node_index:
        return int(node_index[key]), False
    mx = max(int(v) for v in node_index.values()) if node_index else -1
    nid = int(mx + 1)
    node_index[key] = nid
    save_node_index(node_index_path, node_index)
    return nid, True

# ============================================================
# 2) locate edge files (dir-robust + BC-Tissue2 special fallback)
# ============================================================
def _pick_largest_file(paths):
    if not paths:
        return None
    best = None
    best_sz = -1
    for fp in paths:
        try:
            sz = os.path.getsize(fp)
        except Exception:
            sz = -1
        if sz > best_sz:
            best_sz = sz
            best = fp
    return Path(best) if best else None

def find_edge_file_in_dir(dir_path: Path, patterns, must_have_cols=None):
    """
    在 dir_path 下按 pattern 找候选，优先选“文件最大”的（通常主文件）
    可选：校验包含列 must_have_cols
    """
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(str(dir_path / pat)))
    hits = sorted(set(hits))
    if not hits:
        return None

    # sort by size desc
    hits2 = []
    for fp in hits:
        try:
            sz = os.path.getsize(fp)
        except Exception:
            sz = -1
        hits2.append((sz, fp))
    hits2.sort(reverse=True)

    if must_have_cols is None:
        return Path(hits2[0][1])

    # 如果要校验列，就从大到小试
    for _, fp in hits2:
        try:
            df = pd.read_csv(fp, sep="\t", nrows=5)
            cols = set(df.columns)
            if all(c in cols for c in must_have_cols):
                return Path(fp)
        except Exception:
            continue

    return Path(hits2[0][1])

def find_edge_file_with_fallback(ds_root: Path, g0_dir: Path, patterns, must_have_cols=None, special=False):
    """
    先在 g0_dir 找；若 special=True 且找不到，则在整个 ds_root 下递归找（按文件名 pattern 的“末端名字”近似）
    """
    fp = find_edge_file_in_dir(g0_dir, patterns, must_have_cols=must_have_cols)
    if fp is not None:
        return fp, "in_g0_subdir"

    if not special:
        return None, "missing"

    # special fallback: recursive search within dataset
    # 只用 pattern 的“具体文件名/关键字”做 rglob，避免全盘扫描太慢
    # 这里直接用你们最常见的固定名优先
    fixed_names = []
    for pat in patterns:
        if "*" not in pat and "?" not in pat:
            fixed_names.append(pat)
    cands = []
    for name in fixed_names:
        cands.extend([str(p) for p in ds_root.rglob(name)])

    # 如果 fixed_names 没搜到，用更宽松的：挑最关键的几个名
    if not cands:
        # 常见：edges_rna_gene_expr.tsv / edges_re_gene_dist.tsv
        # 或者包含关键字 rna gene expr / re gene dist
        # 用 rglob 先粗搜所有 edges_*.tsv 再做字符串过滤（只在一个 dataset 内，可接受）
        all_edges = list(ds_root.rglob("edges_*.tsv"))
        for p in all_edges:
            s = p.name.lower()
            ok = True
            # 用 patterns 里抽取的关键 token
            # 简单策略：如果 patterns 里出现 rna/gene/expr/dist/re 等关键词，就要求命中
            want_tokens = []
            pats_join = " ".join(patterns).lower()
            for tok in ["rna", "re", "gene", "expr", "dist"]:
                if tok in pats_join:
                    want_tokens.append(tok)
            for tok in want_tokens:
                if tok not in s:
                    ok = False
                    break
            if ok:
                cands.append(str(p))

    fp2 = _pick_largest_file(cands)
    return fp2, ("recursive_ds_root" if fp2 is not None else "missing")

# ============================================================
# 3) core: compute RNA->RE = sum_g (RNA->g) * (RE->g)
# ============================================================
def load_rna_gene_edges(fp: Path):
    """
    期望列：
      - dst_id, w0/weight
    """
    df = pd.read_csv(fp, sep="\t")
    cols = set(df.columns)
    wcol = "w0" if "w0" in cols else ("weight" if "weight" in cols else None)
    if wcol is None:
        raise ValueError(f"RNA->GENE edges missing w0/weight. fp={fp}, cols={list(df.columns)}")
    if "dst_id" not in cols:
        raise ValueError(f"RNA->GENE edges missing dst_id. fp={fp}, cols={list(df.columns)}")

    out = df[["dst_id", wcol]].copy()
    out = out.rename(columns={wcol: "w_rna_gene"})
    out["dst_id"] = out["dst_id"].astype(np.int64)
    out["w_rna_gene"] = out["w_rna_gene"].astype(np.float64)
    return out

def load_re_gene_edges(fp: Path):
    """
    期望列：
      - src_id(=RE id), dst_id(=GENE id), w0/weight
    """
    df = pd.read_csv(fp, sep="\t")
    cols = set(df.columns)
    wcol = "w0" if "w0" in cols else ("weight" if "weight" in cols else None)
    if wcol is None:
        raise ValueError(f"RE->GENE edges missing w0/weight. fp={fp}, cols={list(df.columns)}")
    need = {"src_id", "dst_id"}
    if not need.issubset(cols):
        raise ValueError(f"RE->GENE edges missing {need}. fp={fp}, cols={list(df.columns)}")

    out = df[["src_id", "dst_id", wcol]].copy()
    out = out.rename(columns={wcol: "w_re_gene"})
    out["src_id"] = out["src_id"].astype(np.int64)
    out["dst_id"] = out["dst_id"].astype(np.int64)
    out["w_re_gene"] = out["w_re_gene"].astype(np.float64)
    return out

def compute_rna_re(rna_gene: pd.DataFrame, re_gene: pd.DataFrame):
    """
    join on gene dst_id，然后对每个 RE(src_id) 求和：
      w(RNA->RE) = Σ_g w(RNA->g) * w(RE->g)
    """
    m = re_gene.merge(rna_gene, on="dst_id", how="inner")
    if len(m) == 0:
        return pd.DataFrame({"re_id": [], "weight": []}), 0
    m["contrib"] = m["w_re_gene"] * m["w_rna_gene"]
    agg = m.groupby("src_id", as_index=False)["contrib"].sum()
    agg = agg.rename(columns={"src_id": "re_id", "contrib": "weight"})
    return agg, int(len(m))

def normalize_topm(df: pd.DataFrame, topm: int, do_norm: bool):
    if len(df) == 0:
        return df
    df = df.sort_values("weight", ascending=False)
    if topm is not None and topm > 0 and len(df) > topm:
        df = df.head(int(topm)).copy()
    if do_norm:
        s = float(df["weight"].sum())
        if s > 0:
            df["weight"] = df["weight"] / (s + EPS)
    return df

def quantiles(x: np.ndarray, qs=(0.01,0.05,0.25,0.5,0.75,0.95,0.99)):
    if x.size == 0:
        return {f"w_p{int(q*100):02d}": np.nan for q in qs} | {"w_max": np.nan, "w_sum": 0.0}
    out = {f"w_p{int(q*100):02d}": float(np.quantile(x, q)) for q in qs}
    out["w_max"] = float(np.max(x))
    out["w_sum"] = float(np.sum(x))
    return out

# ============================================================
# 4) RUN ALL DATASETS (scan)
# ============================================================
assert ROOT.exists(), f"ROOT not found: {ROOT}"

datasets = sorted([d for d in ROOT.iterdir() if d.is_dir()], key=lambda x: x.name)
datasets = [d for d in datasets if not BAD_DS_RE.match(d.name)]

rows = []
for ds_root in datasets:
    ds = ds_root.name
    special = (ds in SPECIAL_DS)

    assets_dir, mode = pick_assets_dir(ds_root)
    if assets_dir is None:
        rows.append({"dataset": ds, "status": "skip_no_assets", "mode": mode})
        continue

    node_index_path = assets_dir / "node_index.json"
    if not node_index_path.exists():
        rows.append({"dataset": ds, "status": "skip_no_node_index", "assets_dir": str(assets_dir)})
        continue

    # ✅ 改：用 G0_full49_v1 作为输入/输出目录
    g0 = assets_dir / G0_SUBDIR
    g0.mkdir(parents=True, exist_ok=True)

    # ---- locate inputs ----
    rna_gene_fp, rna_gene_loc = find_edge_file_with_fallback(
        ds_root=ds_root,
        g0_dir=g0,
        patterns=[
            "edges_rna_gene_expr.tsv",
            "edges_*rna*gene*expr*.tsv",
            "edges_*rna*gene*.tsv",
        ],
        must_have_cols=["dst_id"],
        special=special
    )
    re_gene_fp, re_gene_loc = find_edge_file_with_fallback(
        ds_root=ds_root,
        g0_dir=g0,
        patterns=[
            "edges_re_gene_dist.tsv",
            "edges_*re*gene*dist*.tsv",
            "edges_*re*gene*.tsv",
        ],
        must_have_cols=["src_id", "dst_id"],
        special=special
    )

    if rna_gene_fp is None:
        rows.append({
            "dataset": ds, "status": "skip_no_rna_gene_edges",
            "assets_dir": str(assets_dir), "g0_dir": str(g0),
            "loc": rna_gene_loc, "special": bool(special)
        })
        continue
    if re_gene_fp is None:
        rows.append({
            "dataset": ds, "status": "skip_no_re_gene_edges",
            "assets_dir": str(assets_dir), "g0_dir": str(g0),
            "loc": re_gene_loc, "special": bool(special)
        })
        continue

    # ---- load node_index + ensure RNA node ----
    node_index = load_node_index(node_index_path)
    rna_key = f"RNA:{ds}"
    rna_id, created = ensure_node(node_index, rna_key, node_index_path)

    # ---- load edges ----
    try:
        rna_gene = load_rna_gene_edges(rna_gene_fp)
        re_gene  = load_re_gene_edges(re_gene_fp)
    except Exception as e:
        rows.append({
            "dataset": ds,
            "status": "fail_load_inputs",
            "assets_dir": str(assets_dir),
            "g0_dir": str(g0),
            "rna_gene_fp": str(rna_gene_fp),
            "re_gene_fp": str(re_gene_fp),
            "rna_gene_loc": rna_gene_loc,
            "re_gene_loc": re_gene_loc,
            "err": str(e)
        })
        continue

    # ---- compute RNA->RE ----
    agg, n_join = compute_rna_re(rna_gene, re_gene)
    agg = normalize_topm(agg, TOPM_PER_DATASET, NORMALIZE_PER_RNA)

    # write edges: src_id=RNA(dataset), dst_id=RE_id
    out_edges = pd.DataFrame({
        "src_id": np.full((len(agg),), int(rna_id), dtype=np.int64),
        "dst_id": agg["re_id"].astype(np.int64),
        "rel_type": REL_TYPE,
        "w0": agg["weight"].astype(np.float32),
    })

    out_fp = g0 / OUT_EDGES_NAME
    out_edges.to_csv(out_fp, sep="\t", index=False)

    w = out_edges["w0"].to_numpy(dtype=np.float32)
    rep = {
        "dataset": ds,
        "status": "ok",
        "assets_dir": str(assets_dir),
        "mode": mode,
        "g0_dir": str(g0),
        "rna_key": rna_key,
        "rna_id": int(rna_id),
        "rna_node_created": bool(created),

        "in_rna_gene": str(rna_gene_fp),
        "in_re_gene": str(re_gene_fp),
        "rna_gene_loc": rna_gene_loc,
        "re_gene_loc": re_gene_loc,

        "rna_gene_n": int(len(rna_gene)),
        "re_gene_n": int(len(re_gene)),
        "join_rows_n": int(n_join),

        "rna_re_edges_n": int(len(out_edges)),
        "topm": int(TOPM_PER_DATASET),
        "normalized_per_rna": bool(NORMALIZE_PER_RNA),

        **quantiles(w),
        "out_edges": str(out_fp),
        "special": bool(special),
    }
    rows.append(rep)

    print(f"[ok] {ds:<22} RNA={rna_key:<26} "
          f"RNA->GENE={len(rna_gene):>6}  RE->GENE={len(re_gene):>7}  join={n_join:>8}  "
          f"RNA->RE={len(out_edges):>7}  w_p50={rep['w_p50']:.6g}  "
          f"(rna_gene:{rna_gene_loc}, re_gene:{re_gene_loc})")

df = pd.DataFrame(rows)
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_REPORT, sep="\t", index=False)

print("\nSTATUS COUNTS:\n", df["status"].value_counts(dropna=False))
print("\nSaved report:", OUT_REPORT)
print(df.head(30))