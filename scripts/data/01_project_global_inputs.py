# Source: 输入文件统一空间映射(6).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, json, re, gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

# ============================================================
# 0) Config
# ============================================================
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"
NETWORK_ROOT = BASE / "networks"

PAN_NODE_INDEX_JSON = NETWORK_ROOT / "pan_tetrahedron_merged" / "node_index.json"
assert PAN_NODE_INDEX_JSON.exists(), f"missing: {PAN_NODE_INDEX_JSON}"

OUT_ROOT = NETWORK_ROOT / "qc_global_projection"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_MAPS = OUT_ROOT / "maps"
OUT_MAPS.mkdir(parents=True, exist_ok=True)

OUT_PROJECTED = OUT_ROOT / "projected_inputs"
OUT_PROJECTED.mkdir(parents=True, exist_ok=True)

OUT_PRIORS = OUT_ROOT / "priors_cache"
OUT_PRIORS.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_ROOT / "dataset_feature_projection_summary.tsv"

OVERWRITE_PROJECTED = False
OVERWRITE_PRIORS = False

RNA_TOPK_SAVE = None   # 可设为整数，比如 4096；None=保留全部非零
ATAC_TOPK_SAVE = None  # 可设为整数，比如 4096；None=保留全部非零

CHUNK_EDGES = 2_000_000

# ============================================================
# 1) Helpers
# ============================================================
def find_manifest(ds: str, kind: str):
    p = WORLD_ROOT / ds / "processed" / kind / f"manifest_{kind.lower()}.tsv"
    return p if p.exists() else None

def infer_col(df: pd.DataFrame, cands):
    cols = set(df.columns)
    for c in cands:
        if c in cols:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def normalize_manifest_path(p_str: str, manifest_dir: Path) -> Path:
    if p_str is None:
        return None
    s0 = str(p_str).strip().strip('"').strip("'")
    if re.match(r"^[A-Za-z]:[\\/].*", s0):
        return Path(s0)

    s = s0.replace("\\", "/")
    mm = re.match(r"^/mnt/([a-zA-Z])/+(.*)$", s)
    if mm:
        drive = mm.group(1).upper()
        rest = mm.group(2).replace("/", "\\")
        return Path(f"{drive}:\\{rest}")

    return (manifest_dir / s0).resolve()

def get_npy_shape(npy_path: Path):
    arr = np.load(npy_path, mmap_mode="r")
    return tuple(arr.shape)

def get_npz_csr_shape(npz_path: Path):
    z = np.load(npz_path)
    if "shape" in z.files:
        shp = tuple(z["shape"].tolist())
        return int(shp[0]), int(shp[1])
    n_rows = len(z["indptr"]) - 1
    n_cols = int(z["indices"].max()) + 1 if z["indices"].size else 0
    return int(n_rows), int(n_cols)

def save_csr_npz(path: Path, mat: sparse.csr_matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=mat.data.astype(np.float32),
        indices=mat.indices.astype(np.int32),
        indptr=mat.indptr.astype(np.int64),
        shape=np.asarray(mat.shape, dtype=np.int64),
    )

def load_csr_npz(path: Path) -> sparse.csr_matrix:
    z = np.load(path)
    shp = tuple(z["shape"].tolist())
    return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shp)

def topk_csr_per_row(mat: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    if k is None:
        return mat
    mat = mat.tocsr()
    data_out = []
    indices_out = []
    indptr_out = [0]
    for i in range(mat.shape[0]):
        a, b = mat.indptr[i], mat.indptr[i + 1]
        idx = mat.indices[a:b]
        val = mat.data[a:b]
        if len(val) > k:
            sel = np.argpartition(val, -k)[-k:]
            idx = idx[sel]
            val = val[sel]
            order = np.argsort(val)[::-1]
            idx = idx[order]
            val = val[order]
        data_out.append(val)
        indices_out.append(idx)
        indptr_out.append(indptr_out[-1] + len(val))
    data_out = np.concatenate(data_out) if data_out else np.zeros((0,), dtype=np.float32)
    indices_out = np.concatenate(indices_out) if indices_out else np.zeros((0,), dtype=np.int32)
    indptr_out = np.asarray(indptr_out, dtype=np.int64)
    return sparse.csr_matrix((data_out, indices_out, indptr_out), shape=mat.shape)

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
            cands.append((sz, str(a), a))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][2]

def read_gene_list(genes_tsv: Path):
    df = pd.read_csv(genes_tsv, sep="\t", low_memory=False)
    gene_col = None
    for c in ["symbol", "gene", "gene_name", "name"]:
        if c in df.columns:
            gene_col = c
            break
    if gene_col is None:
        raise ValueError(f"no gene symbol col in {genes_tsv}, cols={list(df.columns)}")
    return df[gene_col].astype(str).tolist()

def read_peak_list(peaks_bed: Path):
    arr = []
    with open(peaks_bed, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                arr.append(parts[3])
            elif len(parts) >= 3:
                arr.append(f"{parts[0]}:{parts[1]}-{parts[2]}")
            else:
                arr.append(line.strip())
    return arr

def read_vocab_from_pan(node_index: dict, prefix: str):
    items = [(k, int(v)) for k, v in node_index.items() if k.startswith(prefix)]
    items.sort(key=lambda x: x[1])
    return [k[len(prefix):] for k, _ in items]

def write_txt(path: Path, arr):
    with open(path, "w", encoding="utf-8") as f:
        for x in arr:
            f.write(str(x).strip() + "\n")

def build_local_to_global_index(local_names, global_name2i, prefix=""):
    out = np.full((len(local_names),), -1, dtype=np.int64)
    miss = 0
    for i, name in enumerate(local_names):
        key = prefix + str(name)
        if key in global_name2i:
            out[i] = global_name2i[key]
        else:
            miss += 1
    return out, miss

def dense_row_to_sparse_global(row: np.ndarray, local2global: np.ndarray):
    nz = np.flatnonzero(row)
    if nz.size == 0:
        return np.zeros((0,), np.int32), np.zeros((0,), np.float32)
    vals = row[nz].astype(np.float32)
    gidx = local2global[nz]
    keep = gidx >= 0
    return gidx[keep].astype(np.int32), vals[keep]

def project_dense_npy_to_global_csr(npy_path: Path, local2global: np.ndarray, global_dim: int, topk=None):
    arr = np.load(npy_path, mmap_mode="r")
    data_all = []
    indices_all = []
    indptr = [0]

    for i in range(arr.shape[0]):
        gidx, vals = dense_row_to_sparse_global(arr[i], local2global)

        if topk is not None and len(vals) > topk:
            sel = np.argpartition(vals, -topk)[-topk:]
            gidx = gidx[sel]
            vals = vals[sel]
            order = np.argsort(vals)[::-1]
            gidx = gidx[order]
            vals = vals[order]

        data_all.append(vals)
        indices_all.append(gidx)
        indptr.append(indptr[-1] + len(vals))

    data = np.concatenate(data_all) if data_all else np.zeros((0,), dtype=np.float32)
    indices = np.concatenate(indices_all) if indices_all else np.zeros((0,), dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=(arr.shape[0], global_dim))

def project_atac_csr_to_global_csr(npz_path: Path, local2global: np.ndarray, global_dim: int, topk=None):
    z = np.load(npz_path)
    shp = tuple(z["shape"].tolist()) if "shape" in z.files else (len(z["indptr"]) - 1, int(z["indices"].max()) + 1)
    mat = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shp)

    data_all = []
    indices_all = []
    indptr = [0]

    for i in range(mat.shape[0]):
        a, b = mat.indptr[i], mat.indptr[i + 1]
        loc_idx = mat.indices[a:b]
        vals = mat.data[a:b].astype(np.float32)
        gidx = local2global[loc_idx]
        keep = gidx >= 0
        gidx = gidx[keep].astype(np.int32)
        vals = vals[keep]

        if topk is not None and len(vals) > topk:
            sel = np.argpartition(vals, -topk)[-topk:]
            gidx = gidx[sel]
            vals = vals[sel]
            order = np.argsort(vals)[::-1]
            gidx = gidx[order]
            vals = vals[order]

        data_all.append(vals)
        indices_all.append(gidx)
        indptr.append(indptr[-1] + len(vals))

    data = np.concatenate(data_all) if data_all else np.zeros((0,), dtype=np.float32)
    indices = np.concatenate(indices_all) if indices_all else np.zeros((0,), dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=(mat.shape[0], global_dim))

# ============================================================
# 2) Discover datasets
# ============================================================
all_ds = []
manifest_map = {}
for p in WORLD_ROOT.iterdir():
    if not p.is_dir():
        continue
    ds = p.name
    mr = find_manifest(ds, "RNA")
    ma = find_manifest(ds, "ATAC")
    if mr is not None and ma is not None:
        all_ds.append(ds)
        manifest_map[ds] = (mr, ma)

all_ds = sorted(all_ds)
assert all_ds, "no datasets found"
print("[DATASETS]", len(all_ds))

# ============================================================
# 3) Load global pan vocab from node_index
# ============================================================
pan_node_index = json.loads(PAN_NODE_INDEX_JSON.read_text("utf-8"))

global_genes = read_vocab_from_pan(pan_node_index, "GENE:")
global_peaks = read_vocab_from_pan(pan_node_index, "RE:")
global_tfs   = read_vocab_from_pan(pan_node_index, "TF:")

global_gene_name2i = {f"GENE:{g}": i for i, g in enumerate(global_genes)}
global_peak_name2i = {f"RE:{p}": i for i, p in enumerate(global_peaks)}
global_tf_name2i   = {f"TF:{t}": i for i, t in enumerate(global_tfs)}

write_txt(OUT_MAPS / "global_gene_vocab.txt", global_genes)
write_txt(OUT_MAPS / "global_peak_vocab.txt", global_peaks)
write_txt(OUT_MAPS / "global_tf_vocab.txt", global_tfs)

print("[GLOBAL VOCAB]", "genes=", len(global_genes), "peaks=", len(global_peaks), "tfs=", len(global_tfs))

# ============================================================
# 4) Build dataset local->global projection maps
# ============================================================
projection_rows = []
dataset_maps = {}

for ds in all_ds:
    dataset_root = WORLD_ROOT / ds
    assets = pick_assets_dir(dataset_root)
    if assets is None:
        raise FileNotFoundError(f"assets not found for {ds}")

    genes_tsv = assets / "genes.tsv"
    peaks_bed = assets / "peaks.bed"

    local_genes = read_gene_list(genes_tsv)
    local_peaks = read_peak_list(peaks_bed)

    gene_l2g, gene_miss = build_local_to_global_index(local_genes, global_gene_name2i, prefix="GENE:")
    peak_l2g, peak_miss = build_local_to_global_index(local_peaks, global_peak_name2i, prefix="RE:")

    dataset_maps[ds] = {
        "assets": assets,
        "gene_l2g": gene_l2g,
        "peak_l2g": peak_l2g,
        "local_genes": local_genes,
        "local_peaks": local_peaks,
    }

    projection_rows.append({
        "dataset": ds,
        "n_local_genes": len(local_genes),
        "n_local_peaks": len(local_peaks),
        "gene_miss_in_global": int(gene_miss),
        "peak_miss_in_global": int(peak_miss),
        "gene_hit_rate": float((len(local_genes) - gene_miss) / max(1, len(local_genes))),
        "peak_hit_rate": float((len(local_peaks) - peak_miss) / max(1, len(local_peaks))),
        "assets_dir": str(assets),
    })

projection_df = pd.DataFrame(projection_rows).sort_values("dataset")
projection_df.to_csv(OUT_SUMMARY, sep="\t", index=False)
print("[MAP SUMMARY] ->", OUT_SUMMARY)

# ============================================================
# 5) Project RNA / ATAC matrices to global spaces
# ============================================================
for ds in all_ds:
    print(f"\n[PROJECT] {ds}")
    mr, ma = manifest_map[ds]
    assets = dataset_maps[ds]["assets"]
    gene_l2g = dataset_maps[ds]["gene_l2g"]
    peak_l2g = dataset_maps[ds]["peak_l2g"]

    rna = pd.read_csv(mr, sep="\t", dtype=str, low_memory=False)
    atac = pd.read_csv(ma, sep="\t", dtype=str, low_memory=False)

    rna_mat_col = infer_col(rna, ["log1p_npy", "counts_npy"])
    atac_mat_col = infer_col(atac, ["atac_npz", "atac_csr_npz", "csr_npz", "X_csr_npz", "X_npz", "npz"])

    if rna_mat_col is None:
        raise ValueError(f"{ds} RNA manifest missing matrix column")
    if atac_mat_col is None:
        raise ValueError(f"{ds} ATAC manifest missing matrix column")

    ds_out = OUT_PROJECTED / ds
    ds_out.mkdir(parents=True, exist_ok=True)

    # sample-wise unique matrix paths
    for sid, g in rna.groupby("sample_id"):
        raw = g.iloc[0][rna_mat_col]
        npy_path = normalize_manifest_path(raw, mr.parent)
        out_path = ds_out / f"{sid}.rna.global.csr.npz"
        if out_path.exists() and not OVERWRITE_PROJECTED:
            print("  [SKIP RNA]", sid)
        else:
            print("  [RNA]", sid, npy_path.name)
            mat = project_dense_npy_to_global_csr(
                npy_path=npy_path,
                local2global=gene_l2g,
                global_dim=len(global_genes),
                topk=RNA_TOPK_SAVE,
            )
            save_csr_npz(out_path, mat)

    for sid, g in atac.groupby("sample_id"):
        raw = g.iloc[0][atac_mat_col]
        npz_path = normalize_manifest_path(raw, ma.parent)
        out_path = ds_out / f"{sid}.atac.global.csr.npz"
        if out_path.exists() and not OVERWRITE_PROJECTED:
            print("  [SKIP ATAC]", sid)
        else:
            print("  [ATAC]", sid, npz_path.name)
            mat = project_atac_csr_to_global_csr(
                npz_path=npz_path,
                local2global=peak_l2g,
                global_dim=len(global_peaks),
                topk=ATAC_TOPK_SAVE,
            )
            save_csr_npz(out_path, mat)

# ============================================================
# 6) Build global priors from dataset assets/_MERGE_MATERIALS
# ============================================================
REL_FILES = {
    "tf_gene_f":   ("STD_ONECELL_TF_GENE", "edges_tf_gene_prior.std.fixed.tsv",       "TF",   "GENE"),
    "gene_tf_r":   ("STD_ONECELL_TF_GENE", "edges_tf_gene_prior.std.fixed_rev.tsv",   "GENE", "TF"),
    "peak_gene_f": ("REGENE",              "edges_re_gene_dist.tsv",                   "RE",   "GENE"),
    "gene_peak_r": ("REGENE",              "edges_re_gene_dist_rev.tsv",               "GENE", "RE"),
    "tf_peak_f":   ("TFRE",                "edges_tf_re_motif.tsv",                    "TF",   "RE"),
    "peak_tf_r":   ("TFRE",                "edges_tf_re_motif_rev.tsv",                "RE",   "TF"),
}

def read_dataset_node_index(path: Path):
    obj = json.loads(path.read_text("utf-8"))
    id2name = {int(v): k for k, v in obj.items()}
    return obj, id2name

def get_local_name_to_local_col(ds: str, mode: str):
    if mode == "GENE":
        return {f"GENE:{g}": i for i, g in enumerate(dataset_maps[ds]["local_genes"])}
    if mode == "RE":
        return {f"RE:{p}": i for i, p in enumerate(dataset_maps[ds]["local_peaks"])}
    if mode == "TF":
        # TF 不依赖矩阵列；直接走全局 tf 空间
        return None
    raise ValueError(mode)

def map_name_to_global_idx(name: str, mode: str):
    if mode == "GENE":
        return global_gene_name2i.get(name, -1)
    if mode == "RE":
        return global_peak_name2i.get(name, -1)
    if mode == "TF":
        return global_tf_name2i.get(name, -1)
    raise ValueError(mode)

def iter_edge_chunks(path: Path, chunksize=CHUNK_EDGES):
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False):
        need = {"src_id", "dst_id"}
        if not need.issubset(set(chunk.columns)):
            raise ValueError(f"{path} missing src_id/dst_id")
        chunk["src_id"] = pd.to_numeric(chunk["src_id"], errors="coerce")
        chunk["dst_id"] = pd.to_numeric(chunk["dst_id"], errors="coerce")
        chunk = chunk.dropna(subset=["src_id", "dst_id"])
        if len(chunk) == 0:
            continue
        yield chunk["src_id"].astype(np.int64).to_numpy(), chunk["dst_id"].astype(np.int64).to_numpy()

for rel_name, (subdir, fname, src_mode, dst_mode) in REL_FILES.items():
    out_path = OUT_PRIORS / f"{rel_name}.full.npz"
    if out_path.exists() and not OVERWRITE_PRIORS:
        print("[SKIP PRIOR]", rel_name)
        continue

    print(f"\n[BUILD PRIOR] {rel_name}")

    src_all = []
    dst_all = []
    n_raw = 0
    n_keep = 0

    for ds in all_ds:
        assets = dataset_maps[ds]["assets"]
        edge_path = assets / "_MERGE_MATERIALS" / subdir / fname
        node_index_path = assets / "node_index.json"

        if not edge_path.exists():
            print("  [MISS]", ds, edge_path.name)
            continue

        _, id2name = read_dataset_node_index(node_index_path)

        for src_ids, dst_ids in iter_edge_chunks(edge_path):
            n_raw += len(src_ids)

            src_names = [id2name.get(int(x), None) for x in src_ids]
            dst_names = [id2name.get(int(x), None) for x in dst_ids]

            src_idx = np.asarray([map_name_to_global_idx(x, src_mode) if x is not None else -1 for x in src_names], dtype=np.int64)
            dst_idx = np.asarray([map_name_to_global_idx(x, dst_mode) if x is not None else -1 for x in dst_names], dtype=np.int64)

            keep = (src_idx >= 0) & (dst_idx >= 0)
            if keep.any():
                src_all.append(src_idx[keep])
                dst_all.append(dst_idx[keep])
                n_keep += int(keep.sum())

    if not src_all:
        raise RuntimeError(f"no edges kept for {rel_name}")

    src_arr = np.concatenate(src_all).astype(np.int64)
    dst_arr = np.concatenate(dst_all).astype(np.int64)

    np.savez_compressed(out_path, src=src_arr, dst=dst_arr)
    print(f"  raw={n_raw:,} kept={n_keep:,} saved={out_path}")

print("\n[DONE] Global projection + global priors are ready.")
print("Projected inputs:", OUT_PROJECTED)
print("Global maps     :", OUT_MAPS)
print("Priors cache    :", OUT_PRIORS)
print("Summary         :", OUT_SUMMARY)
