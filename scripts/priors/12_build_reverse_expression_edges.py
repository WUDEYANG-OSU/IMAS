# Source: 反边构建4-6(2).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: reverse RNA_*_EXPR edges in _MERGE_MATERIALS (per-dataset, streaming) =====
# Like your TFRE/REGENE/TFGENE reverse script: one dataset at a time, just swap src/dst + set rel_type.
#
# Inputs (per dataset):
#   assets/_MERGE_MATERIALS/RNA_GENE_EXPR/edges_rna_gene_expr.tsv(.gz)
#   assets/_MERGE_MATERIALS/RNA_RE_EXPR/edges_rna_re_expr.tsv(.gz)
#   assets/_MERGE_MATERIALS/RNA_TF_EXPR/edges_rna_tf_expr.tsv(.gz)
#
# Outputs (same folders):
#   edges_gene_rna_expr_rev.tsv
#   edges_re_rna_expr_rev.tsv
#   edges_tf_rna_expr_rev.tsv
#
# Output schema always: src_id,dst_id,rel_type,w0

import re, gzip
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# 0) Paths / config
# =========================
BASE = get_project_root()
WORLD = BASE / "LLM-WorldWord"
MATERIALS = "_MERGE_MATERIALS"

CHUNK = 2_000_000
OVERWRITE = True
REV_SUFFIX = "_rev"

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp|_QC|_WORLD_GLOBAL|世界模型构建|矫正模型相关图|魔剑-真世界)$", re.I)

REL_SPECS = {
    "rna_gene_expr": {
        "subdir": "RNA_GENE_EXPR",
        "candidates": ["edges_rna_gene_expr.tsv", "edges_rna_gene_expr.tsv.gz"],
        "out_base": "edges_gene_rna_expr.tsv",
        "rel_type": "gene_rna_expr",
    },
    "rna_re_expr": {
        "subdir": "RNA_RE_EXPR",
        "candidates": ["edges_rna_re_expr.tsv", "edges_rna_re_expr.tsv.gz"],
        "out_base": "edges_re_rna_expr.tsv",
        "rel_type": "re_rna_expr",
    },
    "rna_tf_expr": {
        "subdir": "RNA_TF_EXPR",
        "candidates": ["edges_rna_tf_expr.tsv", "edges_rna_tf_expr.tsv.gz"],
        "out_base": "edges_tf_rna_expr.tsv",
        "rel_type": "tf_rna_expr",
    },
}

# =========================
# 1) helpers
# =========================
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _open_text_maybe_gz(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def _first_existing(base: Path, names: list[str]) -> Path | None:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None

def _rev_name(out_base: str) -> str:
    p = Path(out_base)
    return f"{p.stem}{REV_SUFFIX}{p.suffix}"  # edges_xxx.tsv -> edges_xxx_rev.tsv

def _reverse_edges_stream(in_path: Path, out_path: Path, new_rel_type: str,
                          chunksize: int = CHUNK, overwrite: bool = OVERWRITE):
    """
    Read TSV in chunks, swap src/dst, set rel_type, write.
    Requires src_id,dst_id and (w0 or weight). Accepts extra cols.
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
            if len(part) == 0:
                continue

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

# =========================
# 2) scan datasets + reverse per dataset
# =========================
assert WORLD.exists(), f"WORLD not found: {WORLD}"

datasets = sorted([d for d in WORLD.iterdir() if d.is_dir() and not BAD_DS_RE.match(d.name)], key=lambda x: x.name)

rows = []
for ds_root in datasets:
    ds = ds_root.name
    assets = ds_root / "assets"
    mats = assets / MATERIALS

    if not mats.is_dir():
        rows.append({"dataset": ds, "status": "skip_no_materials"})
        continue

    ds_row = {"dataset": ds, "status": "ok"}

    any_done = False
    try:
        for key, spec in REL_SPECS.items():
            subdir = mats / spec["subdir"]
            if not subdir.is_dir():
                ds_row[f"{key}_status"] = "missing_dir"
                continue

            in_fp = _first_existing(subdir, spec["candidates"])
            if in_fp is None:
                ds_row[f"{key}_status"] = "missing_in"
                continue

            out_fp = subdir / _rev_name(spec["out_base"])
            st = _reverse_edges_stream(in_fp, out_fp, spec["rel_type"], chunksize=CHUNK, overwrite=OVERWRITE)

            ds_row[f"{key}_status"] = "ok" if not st.get("skipped") else "skipped"
            ds_row[f"{key}_in"] = str(in_fp)
            ds_row[f"{key}_out"] = st["out"]
            ds_row[f"{key}_rows_out"] = st["rows_out"]

            any_done = any_done or (st.get("rows_out") is not None and st["rows_out"] > 0)

        if not any_done and ds_row["status"] == "ok":
            ds_row["status"] = "ok_but_nothing_done"

        rows.append(ds_row)

        print(f"[{ds_row['status']}] {ds:<24s} "
              f"rna_gene={ds_row.get('rna_gene_expr_status','-'):<12s} "
              f"rna_re={ds_row.get('rna_re_expr_status','-'):<12s} "
              f"rna_tf={ds_row.get('rna_tf_expr_status','-'):<12s}")

    except Exception as e:
        ds_row["status"] = "fail"
        ds_row["error"] = str(e)
        rows.append(ds_row)
        print(f"[fail] {ds}: {e}")

# =========================
# 3) Save report
# =========================
OUT_REPORT = WORLD / "_QC" / "MERGE_PRECHECK" / "reverse_rna_expr3_in_materials_report.tsv"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
rep = pd.DataFrame(rows)
rep.to_csv(OUT_REPORT, sep="\t", index=False)

print("\n[DONE] report ->", OUT_REPORT)
print(rep["status"].value_counts(dropna=False))
print(rep.head(60))