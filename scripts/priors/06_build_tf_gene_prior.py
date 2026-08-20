# Source: 诛仙剑阵3(3).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: same as your ORIGINAL TF->Gene prior build, but add an explain branch =====
import os, json, re
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 0) Paths / Config (match your current layout)
# ============================================================
WORLD = get_world_root()

OUT_TAG = "G0_full49_v1"  # your isolated G0 folder
OUT_REPORT = WORLD.parent / "networks" / f"tf_gene_prior_build_report_{OUT_TAG}_with_explain.tsv"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

SKIP_DATASET_NAMES = {"新建文件夹"}
BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp)$", re.I)

# files in assets/<OUT_TAG> (from your backfill script)
TF_RE_TRAIN_NAME = "edges_tf_re_motif.tsv"
TF_RE_EXPL_NAME  = "edges_tf_re_motif_explain.tsv"

RE_GENE_NAME      = "edges_re_gene_dist.tsv"

# outputs (keep train name same, add explain output)
TF_GENE_TRAIN_OUT = "edges_tf_gene_prior.tsv"
TF_GENE_EXPL_OUT  = "edges_tf_gene_prior_explain.tsv"

# keep EXACT same params as your original code
AGG = "sum"
TOPK_PER_TF = 2000
NORMALIZE_PER_TF = False

# ============================================================
# 1) assets locating (same strategy as original)
# ============================================================
def is_valid_dataset_dir(p: Path) -> bool:
    return p.is_dir() and p.name.strip() not in SKIP_DATASET_NAMES and (not BAD_DS_RE.match(p.name))

def pick_assets_dir(dataset_root: Path):
    need = ["genes.tsv", "peaks.bed", "node_index.json"]
    direct = dataset_root / "assets"
    if direct.is_dir() and all((direct / x).exists() for x in need):
        return direct
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
        return None
    cands.sort(reverse=True)
    return cands[0][1]

# ============================================================
# 2) TF->RE mapping (same as original)
# ============================================================
def map_tf_re_to_ids(edges_tf_re_path: Path, node_index: dict):
    df = pd.read_csv(edges_tf_re_path, sep="\t")
    if "tf_node" not in df.columns or "re_node" not in df.columns:
        raise ValueError(f"{edges_tf_re_path} missing tf_node/re_node columns, got: {list(df.columns)}")

    wcol = "weight" if "weight" in df.columns else ("w0" if "w0" in df.columns else None)
    if wcol is None:
        raise ValueError(f"{edges_tf_re_path} missing weight (or w0) column, got: {list(df.columns)}")

    src = df["tf_node"].map(node_index)
    dst = df["re_node"].map(node_index)
    keep = src.notna() & dst.notna()
    kept = df.loc[keep, ["tf_node","re_node",wcol]].copy()

    kept["src_id"] = src[keep].astype(np.int64).to_numpy()
    kept["dst_id"] = dst[keep].astype(np.int64).to_numpy()
    kept["w0"] = kept[wcol].astype(np.float32)

    out = (kept.groupby(["src_id","dst_id"], as_index=False)
                .agg(w0=("w0","mean")))
    out["rel_type"] = "tf_re_motif"
    return out[["src_id","dst_id","rel_type","w0"]], {
        "rows_in": int(len(df)),
        "rows_mapped": int(len(kept)),
        "edges_out": int(len(out)),
        "schema": "name",
    }

def load_tf_re_as_ids(tf_re_path: Path, node_index: dict):
    """
    To be robust with backfilled files:
      - If already has src_id/dst_id: treat as ids, aggregate duplicates by mean (same idea as original)
      - Else if has tf_node/re_node: map like original
    """
    df = pd.read_csv(tf_re_path, sep="\t")
    cols = set(df.columns)

    if {"src_id","dst_id"}.issubset(cols):
        wcol = "w0" if "w0" in df.columns else ("weight" if "weight" in df.columns else None)
        if wcol is None:
            raise ValueError(f"{tf_re_path} has src_id/dst_id but no w0/weight, got: {list(df.columns)}")
        out = df[["src_id","dst_id",wcol]].copy()
        out["src_id"] = out["src_id"].astype(np.int64)
        out["dst_id"] = out["dst_id"].astype(np.int64)
        out["w0"] = out[wcol].astype(np.float32)

        out = (out.groupby(["src_id","dst_id"], as_index=False)
                 .agg(w0=("w0","mean")))
        out["rel_type"] = "tf_re_motif"
        stat = {"rows_in": int(len(df)), "rows_mapped": int(len(df)), "edges_out": int(len(out)), "schema": "id"}
        return out[["src_id","dst_id","rel_type","w0"]], stat

    if {"tf_node","re_node"}.issubset(cols):
        return map_tf_re_to_ids(tf_re_path, node_index)

    raise ValueError(f"{tf_re_path} unsupported schema, columns: {list(df.columns)}")

# ============================================================
# 3) TF->Gene prior (IDENTICAL to original)
# ============================================================
def build_tf_gene_prior(tf_re_id_df: pd.DataFrame, re_gene_df: pd.DataFrame,
                        agg="sum", topk_per_tf=2000, normalize_per_tf=False):
    tf_re = tf_re_id_df.rename(columns={"src_id":"tf_id","dst_id":"re_id","w0":"w_tf_re"})[["tf_id","re_id","w_tf_re"]]
    re_gene = re_gene_df.rename(columns={"src_id":"re_id","dst_id":"gene_id","w0":"w_re_gene"})[["re_id","gene_id","w_re_gene"]]

    tf_re["tf_id"] = tf_re["tf_id"].astype(np.int64)
    tf_re["re_id"] = tf_re["re_id"].astype(np.int64)
    tf_re["w_tf_re"] = tf_re["w_tf_re"].astype(np.float32)

    re_gene["re_id"] = re_gene["re_id"].astype(np.int64)
    re_gene["gene_id"] = re_gene["gene_id"].astype(np.int64)
    re_gene["w_re_gene"] = re_gene["w_re_gene"].astype(np.float32)

    j = tf_re.merge(re_gene, on="re_id", how="inner")
    j["w"] = j["w_tf_re"] * j["w_re_gene"]

    if agg == "sum":
        out = j.groupby(["tf_id","gene_id"], as_index=False)["w"].sum()
    else:
        out = j.groupby(["tf_id","gene_id"], as_index=False)["w"].max()

    out = out.rename(columns={"tf_id":"src_id","gene_id":"dst_id","w":"w0"})
    out["rel_type"] = "tf_gene_prior"

    if normalize_per_tf:
        mx = out.groupby("src_id")["w0"].transform("max")
        out["w0"] = out["w0"] / (mx + 1e-12)

    if topk_per_tf and topk_per_tf > 0:
        out = out.sort_values(["src_id","w0"], ascending=[True, False])
        out = out.groupby("src_id", as_index=False, group_keys=False).head(topk_per_tf)

    return out[["src_id","dst_id","rel_type","w0"]]

# ============================================================
# 4) batch run (train + explain)
# ============================================================
datasets = sorted([d for d in WORLD.iterdir() if is_valid_dataset_dir(d)], key=lambda x: x.name)

report = []
for ds_root in datasets:
    dataset = ds_root.name

    assets_dir = pick_assets_dir(ds_root)
    if assets_dir is None:
        report.append({"dataset": dataset, "variant": "-", "status": "skip_no_assets"})
        continue

    g0_dir = assets_dir / OUT_TAG
    if not g0_dir.is_dir():
        report.append({"dataset": dataset, "variant": "-", "status": "skip_no_out_tag_dir", "assets_dir": str(assets_dir)})
        continue

    node_index_path = assets_dir / "node_index.json"
    if not node_index_path.exists():
        report.append({"dataset": dataset, "variant": "-", "status": "skip_no_node_index", "assets_dir": str(assets_dir)})
        continue
    with open(node_index_path, "r", encoding="utf-8") as f:
        node_index = json.load(f)

    re_gene_path = g0_dir / RE_GENE_NAME
    if not re_gene_path.exists():
        report.append({"dataset": dataset, "variant": "-", "status": "skip_no_re_gene_dist", "re_gene_path": str(re_gene_path)})
        continue

    re_gene = pd.read_csv(re_gene_path, sep="\t")
    need_cols = {"src_id","dst_id","w0"}
    if not need_cols.issubset(re_gene.columns):
        report.append({"dataset": dataset, "variant": "-", "status": "fail_re_gene_cols", "re_gene_path": str(re_gene_path)})
        continue

    # ---------- TRAIN ----------
    tf_re_train_path = g0_dir / TF_RE_TRAIN_NAME
    if tf_re_train_path.exists():
        try:
            tf_re_id, stat = load_tf_re_as_ids(tf_re_train_path, node_index)
            # keep original behavior: write mapped/standardized TF->RE file into G0 dir
            tf_re_id.to_csv(tf_re_train_path, sep="\t", index=False)

            tf_gene = build_tf_gene_prior(tf_re_id, re_gene, agg=AGG, topk_per_tf=TOPK_PER_TF, normalize_per_tf=NORMALIZE_PER_TF)
            out_tf_gene = g0_dir / TF_GENE_TRAIN_OUT
            tf_gene.to_csv(out_tf_gene, sep="\t", index=False)

            report.append({
                "dataset": dataset, "variant": "train", "status": "ok",
                "assets_dir": str(assets_dir),
                "tf_re_in": str(tf_re_train_path),
                "tf_re_schema": stat.get("schema"),
                "tf_re_rows_in": stat.get("rows_in"),
                "tf_re_rows_mapped": stat.get("rows_mapped"),
                "tf_re_edges_out": stat.get("edges_out"),
                "tf_gene_edges_out": int(len(tf_gene)),
                "tf_gene_mean_genes_per_tf": float(tf_gene.groupby("src_id").size().mean()) if len(tf_gene) else 0.0,
                "tf_gene_mean_tfs_per_gene": float(tf_gene.groupby("dst_id").size().mean()) if len(tf_gene) else 0.0,
                "out_tf_gene": str(out_tf_gene),
            })
            print(f"[ok][train] {dataset}  TF->RE={stat.get('edges_out'):,}  TF->Gene={len(tf_gene):,}")
        except Exception as e:
            report.append({"dataset": dataset, "variant": "train", "status": "fail", "err": str(e), "tf_re_in": str(tf_re_train_path)})
    else:
        report.append({"dataset": dataset, "variant": "train", "status": "skip_no_tf_re", "tf_re_in": str(tf_re_train_path)})

    # ---------- EXPLAIN (only addition; same method) ----------
    tf_re_expl_path = g0_dir / TF_RE_EXPL_NAME
    if tf_re_expl_path.exists():
        try:
            tf_re_id, stat = load_tf_re_as_ids(tf_re_expl_path, node_index)
            # keep original behavior: write mapped/standardized TF->RE file into G0 dir
            tf_re_id.to_csv(tf_re_expl_path, sep="\t", index=False)

            tf_gene = build_tf_gene_prior(tf_re_id, re_gene, agg=AGG, topk_per_tf=TOPK_PER_TF, normalize_per_tf=NORMALIZE_PER_TF)
            out_tf_gene = g0_dir / TF_GENE_EXPL_OUT
            tf_gene.to_csv(out_tf_gene, sep="\t", index=False)

            report.append({
                "dataset": dataset, "variant": "explain", "status": "ok",
                "assets_dir": str(assets_dir),
                "tf_re_in": str(tf_re_expl_path),
                "tf_re_schema": stat.get("schema"),
                "tf_re_rows_in": stat.get("rows_in"),
                "tf_re_rows_mapped": stat.get("rows_mapped"),
                "tf_re_edges_out": stat.get("edges_out"),
                "tf_gene_edges_out": int(len(tf_gene)),
                "tf_gene_mean_genes_per_tf": float(tf_gene.groupby("src_id").size().mean()) if len(tf_gene) else 0.0,
                "tf_gene_mean_tfs_per_gene": float(tf_gene.groupby("dst_id").size().mean()) if len(tf_gene) else 0.0,
                "out_tf_gene": str(out_tf_gene),
            })
            print(f"[ok][explain] {dataset}  TF->RE={stat.get('edges_out'):,}  TF->Gene={len(tf_gene):,}")
        except Exception as e:
            report.append({"dataset": dataset, "variant": "explain", "status": "fail", "err": str(e), "tf_re_in": str(tf_re_expl_path)})
    else:
        report.append({"dataset": dataset, "variant": "explain", "status": "skip_no_tf_re_explain", "tf_re_in": str(tf_re_expl_path)})

rep_df = pd.DataFrame(report)
print("\nSTATUS COUNTS:\n", rep_df["status"].value_counts(dropna=False))
if "variant" in rep_df.columns:
    print("\nSTATUS COUNTS (by variant):")
    print(rep_df.pivot_table(index="variant", columns="status", values="dataset", aggfunc="count", fill_value=0))
rep_df.to_csv(OUT_REPORT, sep="\t", index=False)
print("\nSaved report:", OUT_REPORT)

bad = rep_df[rep_df["status"].isin(["fail","skip_no_tf_re","skip_no_tf_re_explain","skip_no_re_gene_dist","skip_no_assets","skip_no_out_tag_dir","skip_no_node_index"])]
if len(bad):
    print("\n[SKIP/FAIL] first 50:")
    print(bad.head(50))