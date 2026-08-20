# Source: 先验组装-补充1(6).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, re, json, gzip, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

BASE  = get_project_root()
WORLD = BASE / "LLM-WorldWord"
OUT_ROOT    = BASE / "networks" / "pan_priors_assembled_fast"
SHARD_ROOT  = OUT_ROOT / "_shards"
OUT_PER_REL = OUT_ROOT / "per_relation"
MAN_PATH    = OUT_ROOT / "manifest.tsv"
MATERIALS = "_MERGE_MATERIALS"

MAX_WORKERS = min(16, (os.cpu_count() or 8) * 2)
READ_CHUNK = None
BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp|_QC|_WORLD_GLOBAL|世界模型构建|矫正模型相关图|魔剑-真世界)$", re.I)

REL_DIRS = {
    "tf_re_motif": "TFRE",
    "tf_gene_prior": "STD_ONECELL_TF_GENE",
}

def _open_text(p: Path):
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")

def _atomic_write_tsv(df: pd.DataFrame, out_path: Path, overwrite=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return False
    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=out_path.stem + ".", suffix=".tmp", dir=str(out_path.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        df.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, out_path)
        return True
    finally:
        if tmp is not None and tmp.exists():
            try: tmp.unlink()
            except Exception: pass

def _load_node_index_inv(assets_dir: Path):
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

    items = list(obj.items())[:200]
    key_int = sum(1 for k,_ in items if str(k).lstrip("+-").isdigit())
    val_int = sum(1 for _,v in items if str(v).lstrip("+-").isdigit())
    if val_int >= key_int:
        return {int(v): str(k) for k,v in obj.items() if str(v).lstrip("+-").isdigit()}
    else:
        return {int(k): str(v) for k,v in obj.items() if str(k).lstrip("+-").isdigit()}

def _pick_weight_col(cols):
    for c in ("w0","weight","w","score","value"):
        if c in cols:
            return c
    return None

def _normalize_id_cols(part: pd.DataFrame):
    colmap = {}
    if "src_id" not in part.columns:
        for c in ("src","source"):
            if c in part.columns:
                colmap[c] = "src_id"; break
    if "dst_id" not in part.columns:
        for c in ("dst","target"):
            if c in part.columns:
                colmap[c] = "dst_id"; break
    if colmap:
        part = part.rename(columns=colmap)
    return part

def _read_edges_id_to_name(fp: Path, inv: dict):
    with _open_text(fp) as f:
        reader = pd.read_csv(
            f, sep="\t", low_memory=False, dtype=str,
            chunksize=READ_CHUNK, on_bad_lines="skip"
        )

        def _iter(r):
            if isinstance(r, pd.io.parsers.TextFileReader):
                yield from r
            else:
                yield r

        outs = []
        for part in _iter(reader):
            part = _normalize_id_cols(part)
            cols = set(part.columns)
            if "src_id" not in cols or "dst_id" not in cols:
                continue
            wcol = _pick_weight_col(cols)
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

            rt = part["rel_type"].astype(str) if "rel_type" in cols else None
            rel_type = (rt[ok].to_numpy() if rt is not None else np.array([""] * int(ok.sum()), dtype=object))

            outs.append(pd.DataFrame({
                "src_name": src_name[ok].astype(str).to_numpy(),
                "dst_name": dst_name[ok].astype(str).to_numpy(),
                "rel_type": rel_type,
                "w0": w0[ok].to_numpy(np.float32),
            }))

        if not outs:
            return None
        return pd.concat(outs, ignore_index=True)

def _find_rev_files(base_dir: Path):
    """
    返回 [(split, fp), ...]，split 根据文件名是否含 explain 判断
    """
    fps = sorted(base_dir.glob("*rev*.tsv*"))
    out = []
    for fp in fps:
        low = fp.name.lower()
        split = "explain" if "explain" in low else "train"
        out.append((split, fp))
    return out

def _write_rev_shard(rel: str, split: str, dataset: str, df: pd.DataFrame):
    out_dir = SHARD_ROOT / rel / split / "rev"
    out_path = out_dir / f"{dataset}.tsv"
    _atomic_write_tsv(df, out_path, overwrite=True)  # 覆盖写：修复/补齐
    return out_path

def _concat_tsvs(in_files, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()
    first = True
    with open(out_file, "wb") as w:
        for fp in in_files:
            with open(fp, "rb") as r:
                if first:
                    w.write(r.read()); first = False
                else:
                    _ = r.readline()
                    w.write(r.read())

def _process_one_dataset(ds: str):
    ds_root = WORLD / ds
    assets  = ds_root / "assets"
    mats    = assets / MATERIALS
    if not mats.is_dir():
        return {"dataset": ds, "status": "skip_no_materials", "n": 0}

    inv = _load_node_index_inv(assets)
    if inv is None:
        return {"dataset": ds, "status": "skip_no_node_index", "n": 0}

    n = 0
    errs = []
    for rel, dirname in REL_DIRS.items():
        base = mats / dirname
        if not base.is_dir():
            continue

        found = _find_rev_files(base)
        if not found:
            continue

        for split, fp in found:
            try:
                df = _read_edges_id_to_name(fp, inv)
                if df is None or df.empty:
                    continue

                # ===== 语义A：rev 文件保持“反向边”，不做 swap =====
                # （删除/取消之前的 src/dst 互换逻辑）

                df.insert(0, "dataset", ds)
                df.insert(1, "split", split)
                df.insert(2, "direction", "rev")

                _write_rev_shard(rel, split, ds, df)
                n += 1
            except Exception as e:
                errs.append(f"{rel}:{fp.name}:{type(e).__name__}:{e}")

    return {"dataset": ds, "status": "ok" if n else "ok_no_rev", "n": int(n), "errors": " | ".join(errs[:2])}

# ===== run =====
datasets = sorted([p.name for p in WORLD.iterdir() if p.is_dir() and not BAD_DS_RE.match(p.name)])
print(f"[INFO] build REV shards (glob) for tf_re_motif + tf_gene_prior: datasets={len(datasets)} workers={MAX_WORKERS}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(_process_one_dataset, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        r = fut.result()
        if r["status"] == "ok":
            print(f"[ok] {r['dataset']:<24s} rev_written={r['n']}")
        elif r.get("errors"):
            print(f"[warn] {r['dataset']:<24s} {r['status']} {r['errors'][:160]}")

# ===== merge per_relation rev outputs =====
for rel in ["tf_re_motif", "tf_gene_prior"]:
    rel_dir = SHARD_ROOT / rel
    if not rel_dir.exists():
        continue
    for split_dir in sorted([p for p in rel_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
        split = split_dir.name
        rev_dir = split_dir / "rev"
        if not rev_dir.is_dir():
            continue
        shards = sorted(rev_dir.glob("*.tsv"), key=lambda x: x.name)
        if not shards:
            continue
        out_path = OUT_PER_REL / rel / f"{rel}.{split}.rev.tsv"
        _concat_tsvs(shards, out_path)
        print(f"[MERGED-REV] {rel:<12s} {split:<7s} rev shards={len(shards):4d} -> {out_path}")

# ===== rebuild manifest from per_relation =====
manifest_rows = []
for rd in sorted([p for p in OUT_PER_REL.iterdir() if p.is_dir()], key=lambda x: x.name):
    rel = rd.name
    for fp in sorted(rd.glob("*.tsv"), key=lambda x: x.name):
        parts = fp.name.split(".")
        split = parts[-3] if len(parts) >= 4 else ""
        direction = parts[-2] if len(parts) >= 4 else ""
        manifest_rows.append({"rel": rel, "split": split, "direction": direction, "out_path": str(fp)})

manifest = pd.DataFrame(manifest_rows).sort_values(["rel","split","direction","out_path"])
manifest.to_csv(MAN_PATH, sep="\t", index=False)
print(f"[OK] manifest updated -> {MAN_PATH}")
print(manifest[manifest["rel"].isin(["tf_re_motif","tf_gene_prior"])].head(50))