# Source: 诛仙剑阵6(1).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: build RNA(dataset) -> TF edges using (RNA->RE) ⊙ (TF->RE v2_full OR local motif fallback) =====
# RNA->TF(tf) = sum_{RE} P(RE|RNA) * P(RE|TF)
# where:
#   - P(RE|RNA) comes from assets/G0_full49_v1/edges_rna_re_expr.tsv (w0, already normalized to sum=1 in your build)
#   - P(RE|TF) prefers v2_full/tf_re (cap + per-TF normalize), and FALLBACK to dataset local assets/G0_full49_v1/edges_tf_re_motif.tsv
#
# Also:
#   - scan WORLD datasets
#   - output to assets/G0_full49_v1/edges_rna_tf_expr.tsv
#   - BC-Tissue2 special: if RNA->RE missing in G0_full49_v1, try recursive search under dataset

import os, glob, json, re
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 0) CONFIG
# ============================================================
BASE = get_project_root()
WORLD = BASE / "LLM-WorldWord"

# TF->RE (prefer v2_full as requested)
TFRE_ROOT = BASE / "networks_fixweight_20251231_v2_full" / "tf_re"

REL_TYPE = "rna_tf_expr"

# 输出报告（集中）
OUT_REPORT_DIR = BASE / "networks" / "rna_tf_expr"
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = OUT_REPORT_DIR / "build_rna_tf_expr_report.tsv"

# ✅ 输入/输出都走这个目录
G0_SUBDIR = "G0_full49_v1"
OUT_EDGES_NAME = "edges_rna_tf_expr.tsv"

# ---- TF->RE 侧裁剪/归一化超参（对齐你 train 逻辑：alpha=0, beta=1）----
TOPM_PER_TF = 20000     # 每个 TF 最多保留多少 RE（按原始 weight 降序）
TOPN_PER_RE = 50        # 每个 RE 最多保留多少 TF（按 weight 降序）
EPS = 1e-12

# ---- RNA->TF 输出裁剪（可选）----
TOPK_TF_OUT = 250000    # 输出最多保留多少 TF（按 RNA->TF 权重降序）
FILTER_ZERO = True      # 是否丢弃 0 权重

# ---- 特例：dataset -> TFRE 文件名 override（你这个 CRC-Cellline -> edges_COLO320.tsv）----
TFRE_FILE_OVERRIDE = {
    "CRC-Cellline": "edges_COLO320.tsv",
}

# ✅ 特别关照的数据集（RNA->RE 找不到时递归搜）
SPECIAL_DS = {"BC-Tissue2"}

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp)$", re.I)

# ============================================================
# 1) helpers
# ============================================================
def pick_assets_dir(dataset_root: Path):
    """优先 dataset_root/assets；否则递归找 **/assets；选 peaks.bed 最大的一套"""
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

def invert_node_index(node_index: dict) -> dict:
    inv = {}
    for k, v in node_index.items():
        try:
            inv[int(v)] = str(k)
        except Exception:
            pass
    return inv

def ensure_node(node_index: dict, key: str, node_index_path: Path):
    if key in node_index:
        return int(node_index[key]), False
    mx = max(int(v) for v in node_index.values()) if node_index else -1
    nid = int(mx + 1)
    node_index[key] = nid
    save_node_index(node_index_path, node_index)
    return nid, True

def find_or_create_rna_node(node_index: dict, dataset: str, node_index_path: Path):
    cands = [
        f"RNA:{dataset}", f"RNA|{dataset}", f"RNA_{dataset}",
        f"{dataset}:RNA",
        f"rna:{dataset}", f"rna|{dataset}", f"rna_{dataset}", f"{dataset}:rna",
    ]
    for k in cands:
        if k in node_index:
            return int(node_index[k]), k, False

    ds_l = dataset.lower()
    for k in node_index.keys():
        kl = str(k).lower()
        if ds_l in kl and "rna" in kl:
            return int(node_index[k]), str(k), False

    new_key = f"RNA:{dataset}"
    nid, created = ensure_node(node_index, new_key, node_index_path)
    return nid, new_key, created

def build_tfre_dataset_index(tfre_dir: Path):
    """
    扫描 edges_*.tsv，读第一行 dataset 建索引：dataset -> filepath
    """
    idx = {}
    files = sorted(glob.glob(str(tfre_dir / "edges_*.tsv")))
    for fp in files:
        try:
            head = pd.read_csv(fp, sep="\t", usecols=["dataset"], nrows=1)
            ds = str(head["dataset"].iloc[0])
            idx[ds] = fp
        except Exception:
            continue
    return idx

def cap_by_tf(df, topm):
    d = df.sort_values(["tf_node", "weight"], ascending=[True, False])
    return d.groupby("tf_node", as_index=False, group_keys=False).head(topm)

def cap_by_re(df, topn):
    d = df.sort_values(["re_node", "weight"], ascending=[True, False])
    return d.groupby("re_node", as_index=False, group_keys=False).head(topn)

def per_tf_normalize(df):
    s = df.groupby("tf_node")["weight"].transform("sum")
    df = df.copy()
    df["weight"] = df["weight"] / (s + EPS)
    return df

def load_rna_re_edges(fp: Path):
    df = pd.read_csv(fp, sep="\t")
    if "w0" not in df.columns:
        raise ValueError(f"RNA->RE missing w0 column. cols={list(df.columns)} file={fp}")
    if "dst_id" not in df.columns:
        raise ValueError(f"RNA->RE missing dst_id column. cols={list(df.columns)} file={fp}")
    df = df[["dst_id", "w0"]].copy()
    df.rename(columns={"dst_id": "re_id", "w0": "w_rna_re"}, inplace=True)
    return df

def load_tf_re_v2(fp: str):
    # v2_full 和 local motif 都是这四列（你已验证本地 motif 也是）
    df = pd.read_csv(fp, sep="\t", usecols=["tf_node", "re_node", "weight", "dataset"])
    df["tf_node"] = df["tf_node"].astype(str)
    df["re_node"] = df["re_node"].astype(str)
    return df

def resolve_tf_re_file_for_dataset(ds: str, tfre_index: dict):
    # 1) override
    if ds in TFRE_FILE_OVERRIDE:
        fp = TFRE_ROOT / TFRE_FILE_OVERRIDE[ds]
        if fp.exists():
            return str(fp), "override"
        else:
            raise FileNotFoundError(f"Override TF->RE file not found for {ds}: {fp}")

    # 2) dataset_field exact
    if ds in tfre_index:
        return tfre_index[ds], "by_dataset_field"

    # 3) ✅ 去尾号回退：BC-Tissue2 -> BC-Tissue
    ds_strip = re.sub(r"\d+$", "", ds)
    if ds_strip != ds and ds_strip in tfre_index:
        return tfre_index[ds_strip], f"by_dataset_field_strip_digits({ds_strip})"

    # 4) filename fuzzy
    hits = sorted(glob.glob(str(TFRE_ROOT / f"edges_*{ds}*.tsv")))
    if hits:
        return hits[0], "by_filename_fuzzy"

    # 5) filename fuzzy with stripped name
    if ds_strip != ds:
        hits = sorted(glob.glob(str(TFRE_ROOT / f"edges_*{ds_strip}*.tsv")))
        if hits:
            return hits[0], f"by_filename_fuzzy_strip_digits({ds_strip})"

    raise FileNotFoundError(f"Cannot find TF->RE file for dataset={ds} under {TFRE_ROOT}")

# --- BC-Tissue2 special: find RNA->RE with recursive fallback ---
def _pick_largest(paths):
    if not paths:
        return None
    best = None
    best_sz = -1
    for p in paths:
        try:
            sz = os.path.getsize(p)
        except Exception:
            sz = -1
        if sz > best_sz:
            best = p
            best_sz = sz
    return Path(best) if best else None

def find_rna_re_file(assets_dir: Path, ds_root: Path, special: bool):
    """
    先找 assets/G0_full49_v1/edges_rna_re_expr.tsv
    special=True 时，若不存在则在 ds_root 内递归找 edges_*rna*re*expr*.tsv（选最大）
    """
    g0 = assets_dir / G0_SUBDIR
    fp = g0 / "edges_rna_re_expr.tsv"
    if fp.exists():
        return fp, "in_g0_subdir"

    if not special:
        return None, "missing"

    hits = [str(p) for p in ds_root.rglob("edges_*rna*re*expr*.tsv")]
    if not hits:
        cand = []
        for p in ds_root.rglob("edges_*.tsv"):
            n = p.name.lower()
            if ("rna" in n) and ("re" in n) and ("expr" in n):
                cand.append(str(p))
        hits = cand

    fp2 = _pick_largest(hits)
    return fp2, ("recursive_ds_root" if fp2 is not None else "missing")

def find_local_tf_re_motif(assets_dir: Path):
    """本地 TF->RE motif 文件（已验证 schema= tf_node,re_node,weight,dataset）"""
    fp = assets_dir / G0_SUBDIR / "edges_tf_re_motif.tsv"
    return fp

# ============================================================
# 2) MAIN
# ============================================================
assert WORLD.exists(), f"WORLD not found: {WORLD}"
assert TFRE_ROOT.exists(), f"TFRE_ROOT not found: {TFRE_ROOT}"

tfre_index = build_tfre_dataset_index(TFRE_ROOT)

datasets = sorted([d for d in WORLD.iterdir() if d.is_dir()], key=lambda x: x.name)
datasets = [d for d in datasets if not BAD_DS_RE.match(d.name)]

rows = []
for ds_root in datasets:
    ds = ds_root.name
    special = (ds in SPECIAL_DS)

    try:
        assets_dir, mode = pick_assets_dir(ds_root)
        if assets_dir is None:
            rows.append({"dataset": ds, "status": "skip_no_assets", "mode": mode})
            continue

        node_index_path = assets_dir / "node_index.json"
        if not node_index_path.exists():
            rows.append({"dataset": ds, "status": "skip_no_node_index", "assets_dir": str(assets_dir), "mode": mode})
            continue

        node_index = load_node_index(node_index_path)
        id2key = invert_node_index(node_index)

        # RNA node
        rna_id, rna_key, rna_created = find_or_create_rna_node(node_index, ds, node_index_path)

        # RNA->RE edges (G0_full49_v1, with BC-Tissue2 fallback)
        rna_re_fp, rna_re_loc = find_rna_re_file(assets_dir, ds_root, special=special)
        if rna_re_fp is None or (not Path(rna_re_fp).exists()):
            rows.append({
                "dataset": ds, "status": "skip_no_rna_re_edges",
                "assets_dir": str(assets_dir), "mode": mode,
                "in_rna_re": str(rna_re_fp) if rna_re_fp else None,
                "rna_re_loc": rna_re_loc,
                "special": bool(special),
            })
            continue
        rna_re_fp = Path(rna_re_fp)

        # ---- load RNA->RE ----
        rr = load_rna_re_edges(rna_re_fp)
        rr["re_node"] = rr["re_id"].map(id2key).astype(str)
        rr = rr[rr["re_node"].str.startswith("RE:")].copy()

        if FILTER_ZERO:
            rr = rr[rr["w_rna_re"] > 0].copy()

        if len(rr) == 0:
            rows.append({
                "dataset": ds, "status": "fail_rna_re_empty_after_filter",
                "assets_dir": str(assets_dir), "mode": mode,
                "in_rna_re": str(rna_re_fp), "rna_re_loc": rna_re_loc
            })
            print(f"[fail] {ds}: RNA->RE has no edges after filter. file={rna_re_fp}")
            continue

        rna_re_w = rr.set_index("re_node")["w_rna_re"]

        # ---- load TF->RE (prefer v2_full; fallback local motif) ----
        tf_re_fp = None
        tfre_pick = None
        tr0 = None

        try:
            tf_re_fp, tfre_pick = resolve_tf_re_file_for_dataset(ds, tfre_index)
            tr0 = load_tf_re_v2(tf_re_fp)
        except Exception as _e:
            local_fp = find_local_tf_re_motif(assets_dir)
            if local_fp.exists():
                tf_re_fp, tfre_pick = str(local_fp), "local_motif_fallback"
                tr0 = load_tf_re_v2(tf_re_fp)
            else:
                raise FileNotFoundError(
                    f"Cannot find TF->RE in v2_full and no local motif file: {local_fp}. "
                    f"original_error={_e}"
                )

        tr0 = tr0[tr0["tf_node"].str.startswith("TF:") & tr0["re_node"].str.startswith("RE:")].copy()
        if len(tr0) == 0:
            rows.append({
                "dataset": ds, "status": "fail_tf_re_empty",
                "assets_dir": str(assets_dir), "mode": mode,
                "in_tf_re": str(tf_re_fp), "tf_re_pick": tfre_pick
            })
            print(f"[fail] {ds}: TF->RE empty after prefix filter. file={tf_re_fp}")
            continue

        # ---- cap + per-TF normalize ----
        tr = cap_by_tf(tr0, TOPM_PER_TF)
        tr = cap_by_re(tr, TOPN_PER_RE)
        tr = per_tf_normalize(tr)

        # ---- RNA->TF = sum_re P(re|rna)*P(re|tf) ----
        tr["w_rna_re"] = tr["re_node"].map(rna_re_w).fillna(0.0).astype(np.float64)
        tr["contrib"] = tr["weight"].astype(np.float64) * tr["w_rna_re"]
        tf_score = (tr.groupby("tf_node", as_index=True)["contrib"].sum()).sort_values(ascending=False)

        if FILTER_ZERO:
            tf_score = tf_score[tf_score > 0]

        if len(tf_score) == 0:
            rows.append({
                "dataset": ds, "status": "fail_rna_tf_all_zero",
                "assets_dir": str(assets_dir), "mode": mode,
                "in_rna_re": str(rna_re_fp), "rna_re_loc": rna_re_loc,
                "in_tf_re": str(tf_re_fp), "tf_re_pick": tfre_pick
            })
            print(f"[fail] {ds}: RNA->TF all zero after join.")
            continue

        if TOPK_TF_OUT is not None and len(tf_score) > int(TOPK_TF_OUT):
            tf_score = tf_score.iloc[:int(TOPK_TF_OUT)]

        w = tf_score.to_numpy(dtype=np.float64)
        wsum = float(w.sum())
        if wsum <= 0:
            rows.append({"dataset": ds, "status": "fail_rna_tf_sum_nonpos", "assets_dir": str(assets_dir), "mode": mode})
            print(f"[fail] {ds}: RNA->TF sum<=0.")
            continue
        w = (w / wsum).astype(np.float32)

        tf_nodes = tf_score.index.astype(str).to_list()

        # map TF nodes to ids (create if missing)
        tf_ids = []
        tf_created_n = 0
        for k in tf_nodes:
            if k in node_index:
                tf_ids.append(int(node_index[k]))
            else:
                nid, created = ensure_node(node_index, k, node_index_path)
                tf_ids.append(int(nid))
                tf_created_n += int(created)

        # ---- write edges ----
        out_g0 = assets_dir / G0_SUBDIR
        out_g0.mkdir(parents=True, exist_ok=True)
        out_edges = out_g0 / OUT_EDGES_NAME

        edges = pd.DataFrame({
            "src_id": np.full((len(tf_ids),), int(rna_id), dtype=np.int64),
            "dst_id": np.array(tf_ids, dtype=np.int64),
            "rel_type": REL_TYPE,
            "w0": w,
        })
        edges.to_csv(out_edges, sep="\t", index=False)

        ww = edges["w0"].to_numpy(dtype=np.float64)

        rows.append({
            "dataset": ds,
            "status": "ok",
            "assets_dir": str(assets_dir),
            "mode": mode,
            "g0_dir": str(out_g0),
            "rna_key": rna_key,
            "rna_id": int(rna_id),
            "rna_node_created": bool(rna_created),
            "tf_nodes_created": int(tf_created_n),
            "in_rna_re": str(rna_re_fp),
            "rna_re_loc": rna_re_loc,
            "in_tf_re": str(tf_re_fp),
            "tf_re_pick": tfre_pick,
            "rna_re_n_pos": int(len(rr)),
            "tf_re_n_raw": int(len(tr0)),
            "tf_re_n_used": int(len(tr)),
            "rna_tf_edges": int(len(edges)),
            "w_sum": float(ww.sum()),
            "w_p50": float(np.quantile(ww, 0.50)),
            "w_p90": float(np.quantile(ww, 0.90)),
            "w_p99": float(np.quantile(ww, 0.99)),
            "w_max": float(np.max(ww)),
            "out_edges": str(out_edges),
            "special": bool(special),
        })

        print(f"[ok] {ds:<22s} TFRE={tfre_pick:<26s} RNA={rna_key:<22s}  "
              f"RNA->RE(pos)={len(rr):7d}({rna_re_loc})  TF->RE(used)={len(tr):7d}  "
              f"RNA->TF={len(edges):7d}  w_p50={np.quantile(ww,0.5):.3e}")

    except Exception as e:
        rows.append({"dataset": ds, "status": "fail", "error": str(e)})
        print(f"[fail] {ds}: {e}")

rep = pd.DataFrame(rows)
rep.to_csv(OUT_REPORT, sep="\t", index=False)

print("\nSTATUS COUNTS:\n", rep["status"].value_counts(dropna=False))
print("\nSaved report:", OUT_REPORT)
print(rep.sort_values(["status","dataset"]).head(60))