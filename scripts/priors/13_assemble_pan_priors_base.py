# Source: 先验组装(6).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# ===== Jupyter one-cell: fast assemble priors with dataset + split(train/explain) info (THREADS, shard-first) =====
import os, re, json, gzip, shutil, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# ============================================================
# 0) CONFIG
# ============================================================
BASE  = get_project_root()
WORLD = BASE / "LLM-WorldWord"
OUT_ROOT = BASE / "networks" / "pan_priors_assembled_fast"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MATERIALS = "_MERGE_MATERIALS"

# 并行：线程通常 8~32 都可以（看磁盘/解压/CPU），先保守点
MAX_WORKERS = min(16, (os.cpu_count() or 8) * 2)

# 如果你的文件确实不大，直接整读更快；如果某些很大再调小
READ_CHUNK = None   # None=整读；也可以设 2_000_000 / 500_000 等

OVERWRITE = True

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp|_QC|_WORLD_GLOBAL|世界模型构建|矫正模型相关图|魔剑-真世界)$", re.I)

# ============================================================
# 1) What to assemble (edit here)
#   - base_dir: subfolder under _MERGE_MATERIALS
#   - files: candidate filenames (train/explain + fwd/rev)
#   - schema: "id" needs node_index id->name; "name" uses columns directly
#   - name_cols: for schema="name"
# ============================================================
REL_SPECS = [
    # ---------- RNA expr (id schema) ----------
    dict(rel="rna_gene_expr", base_dir="RNA_GENE_EXPR",
         files=[
             ("train","fwd","edges_rna_gene_expr.tsv"),
             ("train","fwd","edges_rna_gene_expr.tsv.gz"),
             ("train","rev","edges_gene_rna_expr_rev.tsv"),
             ("train","rev","edges_gene_rna_expr_rev.tsv.gz"),
         ],
         schema="id", usecols=("src_id","dst_id","rel_type","w0","weight")),

    dict(rel="rna_re_expr", base_dir="RNA_RE_EXPR",
         files=[
             ("train","fwd","edges_rna_re_expr.tsv"),
             ("train","fwd","edges_rna_re_expr.tsv.gz"),
             ("train","rev","edges_re_rna_expr_rev.tsv"),
             ("train","rev","edges_re_rna_expr_rev.tsv.gz"),
         ],
         schema="id", usecols=("src_id","dst_id","rel_type","w0","weight")),

    dict(rel="rna_tf_expr", base_dir="RNA_TF_EXPR",
         files=[
             ("train","fwd","edges_rna_tf_expr.tsv"),
             ("train","fwd","edges_rna_tf_expr.tsv.gz"),
             ("train","rev","edges_tf_rna_expr_rev.tsv"),
             ("train","rev","edges_tf_rna_expr_rev.tsv.gz"),
         ],
         schema="id", usecols=("src_id","dst_id","rel_type","w0","weight")),

    # ---------- TFRE motif (name schema; train + explain; fwd + rev) ----------
    dict(rel="tf_re_motif", base_dir="TFRE",
         files=[
             ("train","fwd","edges_tf_re_motif.tsv"),
             ("train","fwd","edges_tf_re_motif.tsv.gz"),
             ("explain","fwd","edges_tf_re_motif_explain.tsv"),
             ("explain","fwd","edges_tf_re_motif_explain.tsv.gz"),

             ("train","rev","edges_tf_re_motif_rev.tsv"),
             ("explain","rev","edges_tf_re_motif_explain_rev.tsv"),
         ],
         schema="name", name_cols=("tf_node","re_node"), wcols=("weight","w0")),

    # ---------- TF-GENE prior (id schema; train + explain; fwd + rev) ----------
    dict(rel="tf_gene_prior", base_dir="STD_ONECELL_TF_GENE",
         files=[
             ("train","fwd","edges_tf_gene_prior.std.fixed.tsv"),
             ("train","fwd","edges_tf_gene_prior.std.fixed.tsv.gz"),
             ("train","fwd","edges_tf_gene_prior.std.tsv"),
             ("train","fwd","edges_tf_gene_prior.std.tsv.gz"),
             ("train","fwd","edges_tf_gene_prior.tsv"),
             ("train","fwd","edges_tf_gene_prior.tsv.gz"),

             ("explain","fwd","edges_tf_gene_prior_explain.std.fixed.tsv"),
             ("explain","fwd","edges_tf_gene_prior_explain.std.fixed.tsv.gz"),
             ("explain","fwd","edges_tf_gene_prior_explain.std.tsv"),
             ("explain","fwd","edges_tf_gene_prior_explain.std.tsv.gz"),
             ("explain","fwd","edges_tf_gene_prior_explain.tsv"),
             ("explain","fwd","edges_tf_gene_prior_explain.tsv.gz"),

             ("train","rev","edges_tf_gene_prior_rev.tsv"),
             ("explain","rev","edges_tf_gene_prior_explain_rev.tsv"),
         ],
         schema="id", usecols=("src_id","dst_id","rel_type","w0","weight")),

    # ---------- RE-GENE distance (id schema; fwd + rev) ----------
    dict(rel="re_gene_dist", base_dir="REGENE",
         files=[
             ("train","fwd","edges_re_gene_dist.tsv"),
             ("train","fwd","edges_re_gene_dist.tsv.gz"),
             ("train","fwd","edges_re_gene_dist.std.tsv"),
             ("train","fwd","edges_re_gene_dist.std.tsv.gz"),

             ("train","rev","edges_re_gene_dist_rev.tsv"),
             ("train","rev","edges_re_gene_dist_rev.tsv.gz"),
         ],
         schema="id", usecols=("src_id","dst_id","rel_type","w0","weight")),
]

# 输出：每个 relation 一个文件（最终合并）
OUT_PER_REL = OUT_ROOT / "per_relation"
OUT_PER_REL.mkdir(parents=True, exist_ok=True)

# 临时分片目录：每个 relation 下按 dataset 写 shard
SHARD_ROOT = OUT_ROOT / "_shards"
if OVERWRITE and SHARD_ROOT.exists():
    shutil.rmtree(SHARD_ROOT)
SHARD_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2) Utilities (thread-safe)
# ============================================================
def _open_text(p: Path):
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")

def _load_node_index_inv(assets_dir: Path):
    """
    return inv: id(int)->name(str)
    node_index.json assumed name->id (your usual), but we handle id->name too.
    """
    p1 = assets_dir / "node_index.json"
    p2 = assets_dir / MATERIALS / "node_index.json"
    p = p1 if p1.exists() else (p2 if p2.exists() else None)
    if p is None:
        return None

    obj = json.loads(p.read_text("utf-8"))
    if isinstance(obj, dict) and "name2id" in obj and isinstance(obj["name2id"], dict):
        obj = obj["name2id"]
    elif isinstance(obj, dict) and "id2name" in obj and isinstance(obj["id2name"], dict):
        obj = obj["id2name"]

    if not isinstance(obj, dict) or not obj:
        return None

    # detect orientation by sampling
    items = list(obj.items())[:200]
    key_int = sum(1 for k,_ in items if str(k).lstrip("+-").isdigit())
    val_int = sum(1 for _,v in items if str(v).lstrip("+-").isdigit())

    if val_int >= key_int:
        # name->id
        return {int(v): str(k) for k,v in obj.items() if str(v).lstrip("+-").isdigit()}
    else:
        # id->name
        return {int(k): str(v) for k,v in obj.items() if str(k).lstrip("+-").isdigit()}

def _pick_first_existing(base: Path, file_triplets):
    """
    file_triplets: list of (split, direction, fname)
    returns list of matched files (keeping split/direction)
    """
    found = []
    for split, direction, fname in file_triplets:
        p = base / fname
        if p.exists():
            found.append((split, direction, p))
    return found

def _read_edges_id_schema(p: Path, inv: dict, usecols):
    """
    Read id schema file, return dataframe with src_name,dst_name,rel_type,w0
    """
    with _open_text(p) as f:
        df = pd.read_csv(
            f, sep="\t", low_memory=False,
            usecols=lambda c: c in usecols,
            dtype=str, chunksize=READ_CHUNK, on_bad_lines="skip"
        )

        def _iter_parts(reader):
            if isinstance(reader, pd.io.parsers.TextFileReader):
                yield from reader
            else:
                yield reader

        outs = []
        for part in _iter_parts(df):
            cols = set(part.columns)
            wcol = "w0" if "w0" in cols else ("weight" if "weight" in cols else None)
            if wcol is None:
                continue
            part = part.dropna(subset=["src_id","dst_id",wcol])
            if part.empty:
                continue
            src_i = pd.to_numeric(part["src_id"], errors="coerce").astype("Int64")
            dst_i = pd.to_numeric(part["dst_id"], errors="coerce").astype("Int64")
            w0    = pd.to_numeric(part[wcol], errors="coerce").astype(np.float32)

            src_name = src_i.map(inv)
            dst_name = dst_i.map(inv)
            ok = src_name.notna() & dst_name.notna() & w0.notna()
            if not ok.any():
                continue

            rt = part["rel_type"].astype(str) if "rel_type" in part.columns else None
            if rt is None:
                rel_type = np.array([""] * int(ok.sum()), dtype=object)
            else:
                rel_type = rt[ok].to_numpy()

            outs.append(pd.DataFrame({
                "src_name": src_name[ok].astype(str).to_numpy(),
                "dst_name": dst_name[ok].astype(str).to_numpy(),
                "rel_type": rel_type,
                "w0": w0[ok].to_numpy(np.float32),
            }))
        if not outs:
            return None
        return pd.concat(outs, ignore_index=True)

def _read_edges_name_schema(p: Path, name_cols, wcols):
    """
    Read name schema file (tf_node/re_node...), return dataframe with src_name,dst_name,rel_type,w0
    """
    a,b = name_cols
    with _open_text(p) as f:
        df = pd.read_csv(
            f, sep="\t", low_memory=False, dtype=str,
            chunksize=READ_CHUNK, on_bad_lines="skip"
        )

        def _iter_parts(reader):
            if isinstance(reader, pd.io.parsers.TextFileReader):
                yield from reader
            else:
                yield reader

        outs = []
        for part in _iter_parts(df):
            cols = set(part.columns)
            wcol = None
            for c in wcols:
                if c in cols:
                    wcol = c
                    break
            if wcol is None or a not in cols or b not in cols:
                continue
            part = part.dropna(subset=[a,b,wcol])
            if part.empty:
                continue
            w0 = pd.to_numeric(part[wcol], errors="coerce").astype(np.float32)
            ok = w0.notna()
            if not ok.any():
                continue
            rt = part["rel_type"].astype(str) if "rel_type" in cols else None
            rel_type = (rt[ok].to_numpy() if rt is not None else np.array([""] * int(ok.sum()), dtype=object))
            outs.append(pd.DataFrame({
                "src_name": part.loc[ok, a].astype(str).to_numpy(),
                "dst_name": part.loc[ok, b].astype(str).to_numpy(),
                "rel_type": rel_type,
                "w0": w0[ok].to_numpy(np.float32),
            }))
        if not outs:
            return None
        return pd.concat(outs, ignore_index=True)

def _atomic_write_tsv(df: pd.DataFrame, out_path: Path):
    """
    Thread-safe-ish: write to temp then replace.
    Avoid partially-written files being concatenated.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and OVERWRITE:
        try:
            out_path.unlink()
        except Exception:
            pass

    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=out_path.stem + ".", suffix=".tmp", dir=str(out_path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        df.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, out_path)  # atomic on same volume
    finally:
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

def _write_shard(rel: str, split: str, direction: str, dataset: str, df: pd.DataFrame):
    # shard path: _shards/<rel>/<split>/<direction>/<dataset>.tsv
    out_dir = SHARD_ROOT / rel / split / direction
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.tsv"
    _atomic_write_tsv(df, out_path)
    return out_path

def _process_one_dataset(ds: str):
    """
    Worker (thread): read this dataset's materials once, emit shards for each rel/split/direction.
    """
    ds_root = WORLD / ds
    assets  = ds_root / "assets"
    mats    = assets / MATERIALS
    if not mats.is_dir():
        return {"dataset": ds, "status": "skip_no_materials", "n_shards": 0}

    inv = _load_node_index_inv(assets)
    # inv can be None if dataset lacks node_index; name-schema rels can still work
    n_shards = 0
    errors = []

    for spec in REL_SPECS:
        rel = spec["rel"]
        base = mats / spec["base_dir"]
        if not base.is_dir():
            continue

        found = _pick_first_existing(base, spec["files"])
        if not found:
            continue

        for split, direction, fp in found:
            try:
                if spec["schema"] == "id":
                    if inv is None:
                        continue
                    df = _read_edges_id_schema(fp, inv, spec["usecols"])
                else:
                    df = _read_edges_name_schema(fp, spec["name_cols"], spec["wcols"])

                if df is None or df.empty:
                    continue

                # add meta cols
                df.insert(0, "dataset", ds)
                df.insert(1, "split", split)
                df.insert(2, "direction", direction)

                _write_shard(rel, split, direction, ds, df)
                n_shards += 1
            except Exception as e:
                errors.append(f"{rel}:{fp.name}:{type(e).__name__}:{e}")

    return {
        "dataset": ds,
        "status": "ok" if n_shards > 0 else "ok_but_no_files",
        "n_shards": int(n_shards),
        "errors": (" | ".join(errors[:5]) if errors else ""),
    }

# ============================================================
# 3) Run parallel extraction (THREADS)
# ============================================================
datasets = []
for p in WORLD.iterdir():
    if p.is_dir() and not BAD_DS_RE.match(p.name):
        datasets.append(p.name)
datasets = sorted(datasets)

print(f"[INFO] datasets={len(datasets)}  MAX_WORKERS={MAX_WORKERS}")

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(_process_one_dataset, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        ds = futs[fut]
        try:
            r = fut.result()
        except Exception as e:
            r = {"dataset": ds, "status": "thread_crash", "n_shards": 0, "errors": f"{type(e).__name__}:{e}"}
        results.append(r)

        if r["status"] == "ok":
            print(f"[ok]   {r['dataset']:<24s} shards={r['n_shards']}")
        elif r["status"].startswith("skip"):
            print(f"[skip] {r['dataset']:<24s} {r['status']}")
        else:
            print(f"[warn] {r['dataset']:<24s} {r['status']} shards={r['n_shards']}  {r.get('errors','')[:180]}")

rep = pd.DataFrame(results).sort_values(["status","dataset"])
rep_path = OUT_ROOT / "assemble_shards_report.tsv"
rep.to_csv(rep_path, sep="\t", index=False)
print(f"[OK] shard report -> {rep_path}")

# ============================================================
# 4) Concatenate shards into per-relation outputs (very fast)
# ============================================================
def concat_tsvs(in_files, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and OVERWRITE:
        out_file.unlink()

    first = True
    with open(out_file, "wb") as w:
        for fp in in_files:
            with open(fp, "rb") as r:
                if first:
                    w.write(r.read())
                    first = False
                else:
                    # skip header line
                    _ = r.readline()
                    w.write(r.read())

manifest_rows = []
for spec in REL_SPECS:
    rel = spec["rel"]
    rel_dir = SHARD_ROOT / rel
    if not rel_dir.exists():
        continue

    # we output one file per (rel, split, direction) to keep train/explain明确
    for split_dir in sorted([p for p in rel_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
        split = split_dir.name
        for dir_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
            direction = dir_dir.name
            shards = sorted(dir_dir.glob("*.tsv"), key=lambda x: x.name)
            if not shards:
                continue
            out_path = OUT_PER_REL / rel / f"{rel}.{split}.{direction}.tsv"
            concat_tsvs(shards, out_path)

            manifest_rows.append({
                "rel": rel,
                "split": split,
                "direction": direction,
                "out_path": str(out_path),
                "n_shards": len(shards),
            })
            print(f"[MERGED] {rel:<18s} {split:<7s} {direction:<3s}  shards={len(shards):4d}  -> {out_path}")

manifest = pd.DataFrame(manifest_rows).sort_values(["rel","split","direction"])
man_path = OUT_ROOT / "manifest.tsv"
manifest.to_csv(man_path, sep="\t", index=False)
print(f"\n[OK] manifest -> {man_path}")
print(manifest.head(60))