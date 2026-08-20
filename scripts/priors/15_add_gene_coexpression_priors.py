# Source: 先验组装-补充2(6).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, re, json, gzip, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# =========================
# 路径：跟你 assemble 那套保持一致
# =========================
BASE  = get_project_root()          # assemble 输出在 G 盘这套
WORLD = BASE / "LLM-WorldWord"
OUT_ROOT   = BASE / "networks" / "pan_priors_assembled_fast"
SHARD_ROOT = OUT_ROOT / "_shards"
OUT_PER_REL = OUT_ROOT / "per_relation"
MAN_PATH   = OUT_ROOT / "manifest.tsv"
MATERIALS = "_MERGE_MATERIALS"

OUT_PER_REL.mkdir(parents=True, exist_ok=True)
SHARD_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# 新增关系：Gene-Gene coexpr（来自 assets/G0_full49_v1）
# =========================
COEXPR_SUBDIR = "G0_full49_v1"
COEXPR_FILES = [
    ("train","fwd","edges_gene_gene_coexpr.tsv"),
    ("train","fwd","edges_gene_gene_coexpr.tsv.gz"),
    ("train","rev","edges_gene_gene_coexpr_rev.tsv"),
    ("train","rev","edges_gene_gene_coexpr_rev.tsv.gz"),
]

REL_NAME = "gene_gene_coexpr"
MAX_WORKERS = min(16, (os.cpu_count() or 8) * 2)
READ_CHUNK = None  # 大文件可改 500_000

BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp|_QC|_WORLD_GLOBAL|世界模型构建|矫正模型相关图|魔剑-真世界)$", re.I)

# =========================
# utils
# =========================
def _open_text(p: Path):
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")

def _load_node_index_inv(assets_dir: Path):
    # 你的 node_index.json 常见是 name->id；这里返回 id->name
    p = assets_dir / "node_index.json"
    if not p.exists():
        # 兼容有些放在 _MERGE_MATERIALS 里
        p2 = assets_dir / MATERIALS / "node_index.json"
        p = p2 if p2.exists() else None
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
        # name->id
        return {int(v): str(k) for k,v in obj.items() if str(v).lstrip("+-").isdigit()}
    else:
        # id->name
        return {int(k): str(v) for k,v in obj.items() if str(k).lstrip("+-").isdigit()}

def _pick_existing(paths):
    return [x for x in paths if x.exists()]

def _atomic_write_tsv(df: pd.DataFrame, out_path: Path, overwrite=False):
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

def _read_coexpr_edges_id_schema(fp: Path, inv: dict):
    # 兼容列名：src_id/dst_id/w0 or weight
    usecols = {"src_id","dst_id","w0","weight","rel_type","w"}
    with _open_text(fp) as f:
        reader = pd.read_csv(
            f, sep="\t", low_memory=False,
            usecols=lambda c: c in usecols,
            dtype=str, chunksize=READ_CHUNK, on_bad_lines="skip"
        )

        def _iter(r):
            if isinstance(r, pd.io.parsers.TextFileReader):
                yield from r
            else:
                yield r

        outs = []
        for part in _iter(reader):
            cols = set(part.columns)
            if "src_id" not in cols or "dst_id" not in cols:
                continue
            wcol = "w0" if "w0" in cols else ("weight" if "weight" in cols else ("w" if "w" in cols else None))
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
            rel_type = (rt[ok].to_numpy() if rt is not None else np.array(["gene_gene_coexpr"] * int(ok.sum()), dtype=object))

            outs.append(pd.DataFrame({
                "src_name": src_name[ok].astype(str).to_numpy(),
                "dst_name": dst_name[ok].astype(str).to_numpy(),
                "rel_type": rel_type,
                "w0": w0[ok].to_numpy(np.float32),
            }))
        if not outs:
            return None
        return pd.concat(outs, ignore_index=True)

def _write_shard(rel: str, split: str, direction: str, dataset: str, df: pd.DataFrame):
    out_dir = SHARD_ROOT / rel / split / direction
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.tsv"
    # 增量：默认不覆盖单 dataset shard；想强制重建某些 dataset 再手动删 shard
    _atomic_write_tsv(df, out_path, overwrite=False)
    return out_path

def _process_one_dataset(ds: str):
    ds_root = WORLD / ds
    assets = ds_root / "assets"
    if not assets.is_dir():
        return {"dataset": ds, "status": "skip_no_assets", "n_shards": 0}

    inv = _load_node_index_inv(assets)
    if inv is None:
        return {"dataset": ds, "status": "skip_no_node_index", "n_shards": 0}

    g0 = assets / COEXPR_SUBDIR
    if not g0.is_dir():
        return {"dataset": ds, "status": "skip_no_G0", "n_shards": 0}

    found = []
    for split, direction, fname in COEXPR_FILES:
        p = g0 / fname
        if p.exists():
            found.append((split, direction, p))

    if not found:
        return {"dataset": ds, "status": "skip_no_files", "n_shards": 0}

    n_shards = 0
    errs = []
    for split, direction, fp in found:
        try:
            df = _read_coexpr_edges_id_schema(fp, inv)
            if df is None or df.empty:
                continue
            df.insert(0, "dataset", ds)
            df.insert(1, "split", split)
            df.insert(2, "direction", direction)
            _write_shard(REL_NAME, split, direction, ds, df)
            n_shards += 1
        except Exception as e:
            errs.append(f"{fp.name}:{type(e).__name__}:{e}")

    return {"dataset": ds, "status": "ok" if n_shards else "ok_but_empty", "n_shards": int(n_shards), "errors": " | ".join(errs[:3])}

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

# =========================
# 1) 生成（或补齐）shards
# =========================
datasets = sorted([p.name for p in WORLD.iterdir() if p.is_dir() and not BAD_DS_RE.match(p.name)])
print(f"[INFO] integrate {REL_NAME}: datasets={len(datasets)}  workers={MAX_WORKERS}")

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = {ex.submit(_process_one_dataset, ds): ds for ds in datasets}
    for fut in as_completed(futs):
        r = fut.result()
        results.append(r)
        if r["status"] == "ok":
            print(f"[ok]   {r['dataset']:<24s} shards={r['n_shards']}")
        elif r["status"].startswith("skip"):
            pass
        else:
            print(f"[warn] {r['dataset']:<24s} {r['status']}  {r.get('errors','')[:160]}")

# =========================
# 2) 合并 per_relation 输出（只针对 gene_gene_coexpr）
# =========================
rel_dir = SHARD_ROOT / REL_NAME
if rel_dir.exists():
    for split_dir in sorted([p for p in rel_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
        split = split_dir.name
        for dir_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda x: x.name):
            direction = dir_dir.name
            shards = sorted(dir_dir.glob("*.tsv"), key=lambda x: x.name)
            if not shards:
                continue
            out_path = OUT_PER_REL / REL_NAME / f"{REL_NAME}.{split}.{direction}.tsv"
            _concat_tsvs(shards, out_path)
            print(f"[MERGED] {REL_NAME:<18s} {split:<7s} {direction:<3s} shards={len(shards):4d} -> {out_path}")

# =========================
# 3) 更新 manifest（以 per_relation 为准重建）
# =========================
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
print(manifest[manifest["rel"] == REL_NAME].head(50))