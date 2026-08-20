# Source: 反边构建1-3(3).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: build reverse edges (3 relations) inside each dataset's assets/_MERGE_MATERIALS =====
# Targets under assets/_MERGE_MATERIALS:
#   1) TFRE/edges_tf_re_motif.tsv                    -> TFRE/edges_tf_re_motif_rev.tsv            (rel_type = re_tf_motif)
#   2) TFRE/edges_tf_re_motif_explain.tsv            -> TFRE/edges_tf_re_motif_explain_rev.tsv    (rel_type = re_tf_motif_explain)
#   3) REGENE/edges_re_gene_dist*.tsv(.gz optional)  -> REGENE/edges_re_gene_dist_rev.tsv         (rel_type = gene_re_dist)
#   4) STD_ONECELL_TF_GENE/edges_tf_gene_prior*.tsv  -> STD_ONECELL_TF_GENE/..._rev.tsv           (rel_type = gene_tf_prior / gene_tf_prior_explain)
#
# TFRE supports name-schema (tf_node,re_node,weight,...) and will map name->id using node_index.json
# node_index.json location: prefer assets/node_index.json, fallback assets/_MERGE_MATERIALS/node_index.json
#
# Output is always ID schema: src_id,dst_id,rel_type,w0
#
import re, json, gzip
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# 0) Paths / config
# =========================
BASE = get_project_root()
WORLD = BASE / "LLM-WorldWord"
MATERIALS = "_MERGE_MATERIALS"

SUF = "_rev"
CHUNK = 2_000_000
OVERWRITE = True

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp|_QC|_WORLD_GLOBAL|世界模型构建|矫正模型相关图|魔剑-真世界)$", re.I)

# =========================
# 1) helpers
# =========================
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _open_text_maybe_gz(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def _reverse_edges_stream(in_path: Path, out_path: Path, new_rel_type: str,
                          chunksize: int = CHUNK, overwrite: bool = OVERWRITE):
    """
    For ID-schema edges: swap src/dst, set rel_type, keep w0.
    Accepts extra cols. Requires src_id,dst_id and (w0 or weight).
    Works for .tsv or .tsv.gz.
    """
    if out_path.exists() and not overwrite:
        return {"rows_in": None, "rows_out": None, "skipped": True, "out": str(out_path)}

    _ensure_parent(out_path)
    if out_path.exists() and overwrite:
        out_path.unlink()

    rows_in = 0
    rows_out = 0
    first = True

    with _open_text_maybe_gz(in_path) as f:
        for df in pd.read_csv(f, sep="\t", chunksize=chunksize, low_memory=True, dtype=str, on_bad_lines="skip"):
            cols = set(df.columns)
            need = {"src_id", "dst_id"}
            if not need.issubset(cols):
                raise ValueError(f"[reverse] {in_path} missing {need - cols}, got={list(df.columns)[:30]}")

            wcol = "w0" if "w0" in cols else ("weight" if "weight" in cols else None)
            if wcol is None:
                raise ValueError(f"[reverse] {in_path} missing w0/weight, got={list(df.columns)[:30]}")

            part = df[["src_id", "dst_id", wcol]].copy()
            part = part.dropna(subset=["src_id", "dst_id", wcol])

            part["src_id"] = pd.to_numeric(part["src_id"], errors="coerce").astype("Int64")
            part["dst_id"] = pd.to_numeric(part["dst_id"], errors="coerce").astype("Int64")
            part["w0"] = pd.to_numeric(part[wcol], errors="coerce").astype(np.float32)
            part = part.dropna(subset=["src_id", "dst_id", "w0"])
            if len(part) == 0:
                continue

            rows_in += len(part)

            out = pd.DataFrame({
                "src_id": part["dst_id"].astype(np.int64).to_numpy(),
                "dst_id": part["src_id"].astype(np.int64).to_numpy(),
                "rel_type": new_rel_type,
                "w0": part["w0"].to_numpy(np.float32),
            })

            out.to_csv(out_path, sep="\t", index=False, mode="w" if first else "a", header=first)
            first = False
            rows_out += len(out)

    return {"rows_in": int(rows_in), "rows_out": int(rows_out), "skipped": False, "out": str(out_path)}

def _load_node_index(node_index_path: Path) -> dict:
    obj = json.loads(node_index_path.read_text(encoding="utf-8"))
    # tolerate {name2id: {...}} or {id2name: {...}}
    if isinstance(obj, dict) and "name2id" in obj and isinstance(obj["name2id"], dict):
        obj = obj["name2id"]
    if isinstance(obj, dict) and "id2name" in obj and isinstance(obj["id2name"], dict):
        # id2name -> invert
        inv = {}
        for k, v in obj.items():
            inv[str(v)] = int(k) if str(k).lstrip("+-").isdigit() else k
        # above is messy; better just map names->ids from assets/node_index.json typically name->id already
        # if this happens, we fall back to building name->id by scanning keys/vals
        # We'll do a robust conversion below.
        obj = obj

    # robust: decide orientation by sampling
    if not isinstance(obj, dict) or not obj:
        return {}

    def is_int_like(x):
        return isinstance(x, int) or (isinstance(x, str) and re.fullmatch(r"[+-]?\d+", x.strip() or "") is not None)

    items = list(obj.items())[:200]
    key_int = sum(1 for k, _ in items if is_int_like(k))
    val_int = sum(1 for _, v in items if is_int_like(v))

    if val_int >= key_int:
        # name -> id
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            sv = str(v).strip()
            if not re.fullmatch(r"[+-]?\d+", sv or ""):
                continue
            out[str(k)] = int(sv)
        return out
    else:
        # id -> name  => invert to name -> id
        out = {}
        for k, v in obj.items():
            sk = str(k).strip()
            if not re.fullmatch(r"[+-]?\d+", sk or ""):
                continue
            out[str(v)] = int(sk)
        return out

def _reverse_tf_re_with_mapping(tf_re_path: Path, out_path: Path, new_rel_type: str, node_index: dict,
                                chunksize: int = CHUNK, overwrite: bool = OVERWRITE):
    """
    TF-RE can be:
      - id schema: src_id,dst_id,w0/weight
      - name schema: tf_node,re_node,weight/w0 (+ extra cols like dataset)
    Writes reversed edges in id schema: RE->TF
    Works for .tsv or .tsv.gz.
    """
    # peek header
    with _open_text_maybe_gz(tf_re_path) as f:
        head = pd.read_csv(f, sep="\t", nrows=5, dtype=str, on_bad_lines="skip")
    cols = set(head.columns)

    if {"src_id", "dst_id"}.issubset(cols):
        return _reverse_edges_stream(tf_re_path, out_path, new_rel_type, chunksize=chunksize, overwrite=overwrite)

    if not {"tf_node", "re_node"}.issubset(cols):
        raise ValueError(f"[tf_re_reverse] unsupported schema: {tf_re_path}, cols={list(head.columns)}")

    wcol = "weight" if "weight" in cols else ("w0" if "w0" in cols else None)
    if wcol is None:
        raise ValueError(f"[tf_re_reverse] missing weight/w0: {tf_re_path}, cols={list(head.columns)}")

    if out_path.exists() and not overwrite:
        return {"rows_in": None, "rows_out": None, "skipped": True, "out": str(out_path), "schema": "name"}

    _ensure_parent(out_path)
    if out_path.exists() and overwrite:
        out_path.unlink()

    rows_in = 0
    rows_out = 0
    first_write = True

    with _open_text_maybe_gz(tf_re_path) as f:
        for df in pd.read_csv(f, sep="\t", chunksize=chunksize, low_memory=True, dtype=str, on_bad_lines="skip"):
            if "tf_node" not in df.columns or "re_node" not in df.columns:
                raise ValueError(f"[tf_re_reverse] chunk missing tf_node/re_node in {tf_re_path}")

            wcol_use = wcol if wcol in df.columns else ("weight" if "weight" in df.columns else ("w0" if "w0" in df.columns else None))
            if wcol_use is None:
                raise ValueError(f"[tf_re_reverse] chunk missing weight/w0 in {tf_re_path}")

            part = df[["tf_node", "re_node", wcol_use]].copy()
            part = part.dropna(subset=["tf_node", "re_node", wcol_use])
            if len(part) == 0:
                continue

            # map -> ids
            src = part["tf_node"].map(node_index)
            dst = part["re_node"].map(node_index)
            keep = src.notna() & dst.notna()
            part = part.loc[keep].copy()
            if len(part) == 0:
                continue

            w = pd.to_numeric(part[wcol_use], errors="coerce").astype(np.float32)
            ok = w.notna()
            part = part.loc[ok].copy()
            w = w.loc[ok]
            if len(part) == 0:
                continue

            rows_in += len(part)

            # reverse TF->RE to RE->TF
            out = pd.DataFrame({
                "src_id": dst.loc[keep].loc[part.index].astype(np.int64).to_numpy(),
                "dst_id": src.loc[keep].loc[part.index].astype(np.int64).to_numpy(),
                "rel_type": new_rel_type,
                "w0": w.to_numpy(np.float32),
            })

            out.to_csv(out_path, sep="\t", index=False, mode="w" if first_write else "a", header=first_write)
            first_write = False
            rows_out += len(out)

    return {"rows_in": int(rows_in), "rows_out": int(rows_out), "skipped": False, "out": str(out_path), "schema": "name"}

def _first_existing(*paths: Path):
    for p in paths:
        if p is not None and p.exists():
            return p
    return None

# =========================
# 2) scan datasets + build reverses
# =========================
assert WORLD.exists(), f"WORLD not found: {WORLD}"

rows = []
datasets = sorted([d for d in WORLD.iterdir() if d.is_dir() and not BAD_DS_RE.match(d.name)], key=lambda x: x.name)

for ds_root in datasets:
    ds = ds_root.name
    assets = ds_root / "assets"
    mats = assets / MATERIALS

    if not mats.is_dir():
        rows.append({"dataset": ds, "status": "skip_no_materials", "assets": str(assets)})
        continue

    # node_index.json: prefer assets/, fallback _MERGE_MATERIALS/
    node_index_path = _first_existing(assets / "node_index.json", mats / "node_index.json")
    if node_index_path is None:
        rows.append({"dataset": ds, "status": "skip_no_node_index", "assets": str(assets)})
        continue

    node_index = _load_node_index(node_index_path)
    if not node_index:
        rows.append({"dataset": ds, "status": "skip_bad_node_index", "node_index": str(node_index_path)})
        continue

    # ----- input paths (materials) -----
    # TFRE
    tfre_dir = mats / "TFRE"
    tfre_train = tfre_dir / "edges_tf_re_motif.tsv"
    tfre_expl  = tfre_dir / "edges_tf_re_motif_explain.tsv"
    tfre_train_gz = tfre_dir / "edges_tf_re_motif.tsv.gz"
    tfre_expl_gz  = tfre_dir / "edges_tf_re_motif_explain.tsv.gz"

    # REGENE
    reg_dir = mats / "REGENE"
    reg = _first_existing(reg_dir / "edges_re_gene_dist.tsv",
                          reg_dir / "edges_re_gene_dist.std.tsv",
                          reg_dir / "edges_re_gene_dist.std.tsv.gz",
                          reg_dir / "edges_re_gene_dist.tsv.gz")

    # STD_ONECELL_TF_GENE
    tfg_dir = mats / "STD_ONECELL_TF_GENE"
    tfg_train = _first_existing(tfg_dir / "edges_tf_gene_prior.std.fixed.tsv",
                                tfg_dir / "edges_tf_gene_prior.std.fixed.tsv.gz",
                                tfg_dir / "edges_tf_gene_prior.std.tsv",
                                tfg_dir / "edges_tf_gene_prior.std.tsv.gz",
                                tfg_dir / "edges_tf_gene_prior.tsv",
                                tfg_dir / "edges_tf_gene_prior.tsv.gz")

    tfg_expl = _first_existing(tfg_dir / "edges_tf_gene_prior_explain.std.fixed.tsv",
                               tfg_dir / "edges_tf_gene_prior_explain.std.fixed.tsv.gz",
                               tfg_dir / "edges_tf_gene_prior_explain.std.tsv",
                               tfg_dir / "edges_tf_gene_prior_explain.std.tsv.gz",
                               tfg_dir / "edges_tf_gene_prior_explain.tsv",
                               tfg_dir / "edges_tf_gene_prior_explain.tsv.gz")

    # decide tfre actual file (prefer .tsv, fallback .gz)
    tfre_train_use = tfre_train if tfre_train.exists() else (tfre_train_gz if tfre_train_gz.exists() else None)
    tfre_expl_use  = tfre_expl  if tfre_expl.exists()  else (tfre_expl_gz  if tfre_expl_gz.exists()  else None)

    # ----- reverse 3 relations (train/explain variants where applicable) -----
    st_train = st_expl = st_reg = st_tfg_train = st_tfg_expl = None

    try:
        # TFRE train -> RE->TF
        if tfre_train_use is not None:
            out_re_tf = Path(str(tfre_train_use).replace(".tsv.gz", ".tsv").replace(".tsv", f"{SUF}.tsv"))
            st_train = _reverse_tf_re_with_mapping(
                tfre_train_use, out_re_tf, "re_tf_motif", node_index,
                chunksize=CHUNK, overwrite=OVERWRITE
            )

        # TFRE explain -> RE->TF
        if tfre_expl_use is not None:
            out_re_tf_expl = Path(str(tfre_expl_use).replace(".tsv.gz", ".tsv").replace(".tsv", f"{SUF}.tsv"))
            st_expl = _reverse_tf_re_with_mapping(
                tfre_expl_use, out_re_tf_expl, "re_tf_motif_explain", node_index,
                chunksize=CHUNK, overwrite=OVERWRITE
            )

        # REGENE -> Gene<-RE  (reverse RE->Gene to Gene->RE)
        if reg is not None:
            out_gene_re = reg.with_name(reg.name.replace(".tsv.gz", ".tsv").replace(".tsv", f"{SUF}.tsv"))
            st_reg = _reverse_edges_stream(reg, out_gene_re, "gene_re_dist", chunksize=CHUNK, overwrite=OVERWRITE)

        # TF-GENE train -> Gene<-TF
        if tfg_train is not None:
            out_gene_tf = tfg_train.with_name(tfg_train.name.replace(".tsv.gz", ".tsv").replace(".tsv", f"{SUF}.tsv"))
            st_tfg_train = _reverse_edges_stream(tfg_train, out_gene_tf, "gene_tf_prior", chunksize=CHUNK, overwrite=OVERWRITE)

        # TF-GENE explain -> Gene<-TF
        if tfg_expl is not None:
            out_gene_tf_expl = tfg_expl.with_name(tfg_expl.name.replace(".tsv.gz", ".tsv").replace(".tsv", f"{SUF}.tsv"))
            st_tfg_expl = _reverse_edges_stream(tfg_expl, out_gene_tf_expl, "gene_tf_prior_explain", chunksize=CHUNK, overwrite=OVERWRITE)

        rows.append({
            "dataset": ds,
            "status": "ok",
            "node_index": str(node_index_path),

            "tf_re_train_in": str(tfre_train_use) if tfre_train_use else None,
            "tf_re_train_rev": (st_train["out"] if st_train else None),
            "rows_tf_re_train": (st_train["rows_out"] if st_train else None),

            "tf_re_expl_in": str(tfre_expl_use) if tfre_expl_use else None,
            "tf_re_expl_rev": (st_expl["out"] if st_expl else None),
            "rows_tf_re_expl": (st_expl["rows_out"] if st_expl else None),

            "re_gene_in": str(reg) if reg else None,
            "re_gene_rev": (st_reg["out"] if st_reg else None),
            "rows_re_gene": (st_reg["rows_out"] if st_reg else None),

            "tf_gene_train_in": str(tfg_train) if tfg_train else None,
            "tf_gene_train_rev": (st_tfg_train["out"] if st_tfg_train else None),
            "rows_tf_gene_train": (st_tfg_train["rows_out"] if st_tfg_train else None),

            "tf_gene_expl_in": str(tfg_expl) if tfg_expl else None,
            "tf_gene_expl_rev": (st_tfg_expl["out"] if st_tfg_expl else None),
            "rows_tf_gene_expl": (st_tfg_expl["rows_out"] if st_tfg_expl else None),
        })

        print(f"[ok] {ds:<24s} "
              f"TFRE_train_rev={st_train['rows_out'] if st_train else 'NA'}  "
              f"TFRE_expl_rev={st_expl['rows_out'] if st_expl else 'NA'}  "
              f"REGENE_rev={st_reg['rows_out'] if st_reg else 'NA'}  "
              f"TFGENE_train_rev={st_tfg_train['rows_out'] if st_tfg_train else 'NA'}  "
              f"TFGENE_expl_rev={st_tfg_expl['rows_out'] if st_tfg_expl else 'NA'}")

    except Exception as e:
        rows.append({
            "dataset": ds,
            "status": "fail",
            "error": str(e),
            "node_index": str(node_index_path) if node_index_path else None,
            "tf_re_train_in": str(tfre_train_use) if tfre_train_use else None,
            "tf_re_expl_in": str(tfre_expl_use) if tfre_expl_use else None,
            "re_gene_in": str(reg) if reg else None,
            "tf_gene_train_in": str(tfg_train) if tfg_train else None,
            "tf_gene_expl_in": str(tfg_expl) if tfg_expl else None,
        })
        print(f"[fail] {ds}: {e}")

# =========================
# 3) Save report
# =========================
OUT_REPORT = (WORLD / "_QC" / "MERGE_PRECHECK" / "reverse_3_edges_in_materials_report.tsv")
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_REPORT, sep="\t", index=False)
print("\n[DONE] report ->", OUT_REPORT)
print(pd.DataFrame(rows)["status"].value_counts(dropna=False))
