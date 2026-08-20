# Source: LODO真实建模代码-改1(2).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, json, time, math, random, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================
# 0) Config (global projected + hashed peak embedding)
# ============================================================
BASE = get_project_root()
WORLD_ROOT = BASE / "LLM-WorldWord"

GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"
PRIORS_ROOT = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_PEAK_VOCAB = MAPS_ROOT / "global_peak_vocab.txt"
GLOBAL_TF_VOCAB   = MAPS_ROOT / "global_tf_vocab.txt"

NPZ_INPUTS = BASE / "npz_inputs"
PAIRS_TSV_ALL = NPZ_INPUTS / "cell_pairs_all.tsv"

QC_RUNS = BASE / "runs" / "qc_lodo_planC_v7_global_hashpeak"
QC_RUNS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 0.1) Runtime behavior controls
# ============================================================
SKIP_COMPLETED_DATASETS = True
VERBOSE_COLLATE = False
RUN_DEBUG_BATCH_CHECK = False

SUMMARY_PARTIAL = QC_RUNS / "qc_lodo_summary.partial.tsv"
SUMMARY_FINAL   = QC_RUNS / "qc_lodo_summary.tsv"

# ============================================================
# 0.2) Path sanitation / G->E residual checks
# ============================================================
G_PATH_PAT = re.compile(r"(?:G:\\|G:/|/mnt/g/)", re.IGNORECASE)

def has_old_g_path(x) -> bool:
    return bool(G_PATH_PAT.search(str(x)))

def assert_not_g_drive(*paths):
    for p in paths:
        s = str(p)
        if has_old_g_path(s):
            raise ValueError(f"Found old G-drive path in config/path variable: {s}")

def scan_df_for_g_paths(df: pd.DataFrame, name: str, warn_only: bool = True, max_show: int = 3):
    hits = []
    for col in df.columns:
        try:
            if df[col].dtype == object:
                ss = df[col].astype(str)
                bad = ss.str.contains(G_PATH_PAT, regex=True, na=False)
                n = int(bad.sum())
                if n > 0:
                    example = ss[bad].iloc[0]
                    hits.append((col, n, example))
        except Exception:
            pass

    if hits:
        msg = [f"[WARN] {name} contains old G-drive strings ({len(hits)} columns hit):"]
        for col, n, ex in hits[:max_show]:
            msg.append(f"    - column={col!r}, rows={n}, example={ex}")
        if len(hits) > max_show:
            msg.append(f"    - ... and {len(hits) - max_show} more columns")
        text = "\n".join(msg)
        if warn_only:
            print(text, flush=True)
        else:
            raise ValueError(text)

def assert_mapping_paths_clean(mapping: dict, mapping_name: str):
    for k, p in mapping.items():
        s = str(p)
        if has_old_g_path(s):
            raise ValueError(f"[{mapping_name}] key={k} still points to old G-drive path: {s}")

assert_not_g_drive(
    BASE,
    WORLD_ROOT,
    GLOBAL_ROOT,
    PROJECTED_ROOT,
    PRIORS_ROOT,
    MAPS_ROOT,
    NPZ_INPUTS,
    PAIRS_TSV_ALL,
    QC_RUNS,
    GLOBAL_GENE_VOCAB,
    GLOBAL_PEAK_VOCAB,
    GLOBAL_TF_VOCAB,
    SUMMARY_PARTIAL,
    SUMMARY_FINAL,
)

assert PROJECTED_ROOT.exists(), f"missing projected_inputs: {PROJECTED_ROOT}"
assert PRIORS_ROOT.exists(), f"missing priors_cache: {PRIORS_ROOT}"
assert MAPS_ROOT.exists(), f"missing maps: {MAPS_ROOT}"

assert GLOBAL_GENE_VOCAB.exists(), f"missing: {GLOBAL_GENE_VOCAB}"
assert GLOBAL_PEAK_VOCAB.exists(), f"missing: {GLOBAL_PEAK_VOCAB}"
assert GLOBAL_TF_VOCAB.exists(), f"missing: {GLOBAL_TF_VOCAB}"

# ============================================================
# 1) Hyperparams
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "CUDA not available"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BATCH_SIZE = 32
NUM_WORKERS = 0
PREFETCH_FACTOR = None
PIN_MEMORY = True
PERSISTENT_WORKERS = False

CAP_RNA_TOKENS  = 2048
CAP_ATAC_TOKENS = 1024

D = 256
HID = 512
DROP = 0.1

PEAK_HASH_SIZE = 2_000_000

CL_TEMP = 0.07
W_ALIGN = 1.0

EDGE_BATCH = 4096
NEG_K = 4
W_TF_GENE_F   = 0.25
W_GENE_TF_R   = 0.25
W_PEAK_GENE_F = 0.20
W_GENE_PEAK_R = 0.20
W_TF_PEAK_F   = 0.10
W_PEAK_TF_R   = 0.10

LR = 2e-4
WD = 1e-2
CLIP = 1.0
USE_AMP = True
AMP_DTYPE = torch.bfloat16

LAMBDA_CANCER_L2 = 1e-4

BASE_STEPS = 30_000
TRANSFER_STEPS = 10_000
LOG_EVERY = 50
SAVE_EVERY = 5000

print("[CFG] BASE =", BASE, flush=True)
print("[CFG] QC_RUNS =", QC_RUNS, flush=True)

# ============================================================
# 2) Helpers
# ============================================================
BARCODE_RE = re.compile(r"([ACGT]{12,})(-\d+)?", re.IGNORECASE)

def extract_barcode(cell_id: str) -> str:
    if cell_id is None:
        return ""
    s = str(cell_id).strip()
    m = BARCODE_RE.search(s)
    if not m:
        tok = re.split(r"[\s#/|,:;]+", s)[-1]
        m = BARCODE_RE.search(tok)
        if not m:
            return s
    return m.group(1).upper()

def dataset_to_cancer_type(ds_name: str) -> str:
    s = str(ds_name)
    return s.split("-", 1)[0] if "-" in s else s

def read_vocab_txt(p: Path):
    arr = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                arr.append(s)
    return arr

def topk_from_csr_row(indices: np.ndarray, data: np.ndarray, topk: int):
    if indices.size == 0:
        return np.empty((0,), np.int64), np.empty((0,), np.float32)
    k = min(topk, indices.size)
    sel = np.argpartition(data, -k)[-k:]
    idx = indices[sel]
    vals = data[sel]
    order = np.argsort(vals)[::-1]
    return idx[order].astype(np.int64), vals[order].astype(np.float32)

def load_existing_qc_summary():
    rows_by_holdout = {}
    loaded_from = None

    for p in [SUMMARY_FINAL, SUMMARY_PARTIAL]:
        if p.exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if "holdout_ds" not in df.columns:
                    continue
                scan_df_for_g_paths(df, f"existing_summary({p.name})", warn_only=True, max_show=2)

                for _, row in df.iterrows():
                    ds = str(row["holdout_ds"])
                    rows_by_holdout[ds] = row.to_dict()
                loaded_from = p
                break
            except Exception as e:
                print(f"[WARN] failed to load existing summary from {p}: {e}", flush=True)

    return rows_by_holdout, loaded_from

def row_checkpoint_exists(row_dict):
    ckpt_transfer = str(row_dict.get("ckpt_transfer", "")).strip()
    if not ckpt_transfer:
        return False
    if has_old_g_path(ckpt_transfer):
        return False
    return Path(ckpt_transfer).exists()

def default_transfer_ckpt_for_holdout(holdout_ds: str):
    return QC_RUNS / holdout_ds / "ckpt_last_TRANSFER_global_hashpeak.pt"

def holdout_is_completed(holdout_ds: str, existing_rows_by_holdout: dict):
    row = existing_rows_by_holdout.get(holdout_ds, None)
    if row is not None and row_checkpoint_exists(row):
        return True
    ckpt_default = default_transfer_ckpt_for_holdout(holdout_ds)
    if ckpt_default.exists():
        return True
    return False

def upsert_qc_row(rows_by_holdout: dict, row: dict):
    rows_by_holdout[str(row["holdout_ds"])] = row

# ============================================================
# 2.1) Optimized pairing rules
# ============================================================
FORCE_ROW_INDEX_DATASETS = {
    "CRC-Cellline",
}

def normalize_manifest_for_pairing(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["sample_id_norm"] = x["sample_id"].astype(str).str.strip()
    x["cell_id_norm"] = x["cell_id"].astype(str).str.strip()
    x["barcode_norm"] = x["cell_id_norm"].map(extract_barcode)
    x["row_index_norm"] = x["row_index"].astype(str).str.strip()
    return x

def pair_manifests_optimized(ds: str, rna: pd.DataFrame, atac: pd.DataFrame):
    need = ["cell_id", "sample_id", "row_index"]
    for c in need:
        if c not in rna.columns:
            raise ValueError(f"{ds} RNA manifest missing {c}")
        if c not in atac.columns:
            raise ValueError(f"{ds} ATAC manifest missing {c}")

    rna = normalize_manifest_for_pairing(rna)
    atac = normalize_manifest_for_pairing(atac)

    def pair_by_key(key_rna: str, key_atac: str, strategy_name: str):
        rna_u = rna.drop_duplicates(key_rna, keep="first").copy()
        atac_u = atac.drop_duplicates(key_atac, keep="first").copy()
        paired_ = rna_u.merge(
            atac_u,
            left_on=key_rna,
            right_on=key_atac,
            how="inner",
            suffixes=("_rna", "_atac"),
        ).copy()
        paired_["pair_strategy"] = strategy_name
        return paired_

    if ds in FORCE_ROW_INDEX_DATASETS:
        paired = pair_by_key("row_index_norm", "row_index_norm", "row_index")
        return paired, "row_index"

    rna["join_sample_barcode"] = rna["sample_id_norm"] + "||" + rna["barcode_norm"]
    atac["join_sample_barcode"] = atac["sample_id_norm"] + "||" + atac["barcode_norm"]
    paired_sb = pair_by_key("join_sample_barcode", "join_sample_barcode", "sample_barcode")
    if len(paired_sb) > 0:
        return paired_sb, "sample_barcode"

    paired_bc = pair_by_key("barcode_norm", "barcode_norm", "barcode_only")
    if len(paired_bc) > 0:
        return paired_bc, "barcode_only"

    paired_ri = pair_by_key("row_index_norm", "row_index_norm", "row_index")
    return paired_ri, "row_index"

# ============================================================
# 3) Discover all datasets
# ============================================================
def find_manifest(ds: str, kind: str):
    base = WORLD_ROOT / ds / "processed" / kind
    if not base.exists():
        return None

    preferred = [
        base / f"manifest_{kind.lower()}.tsv",
        base / "manifest.tsv",
        base / f"manifest_{kind.upper()}.tsv",
    ]

    for p in preferred:
        if p.exists():
            return p

    extra = sorted(base.glob("manifest*.tsv"))
    for p in extra:
        if p.exists():
            return p

    return None

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
assert len(all_ds) > 0, "No datasets found"
print("[DATASETS] found:", len(all_ds), flush=True)

for ds, (mr, ma) in manifest_map.items():
    assert_not_g_drive(mr, ma)

# ============================================================
# 4) Global vocab / dims
# ============================================================
global_genes = read_vocab_txt(GLOBAL_GENE_VOCAB)
global_peaks = read_vocab_txt(GLOBAL_PEAK_VOCAB)
global_tfs   = read_vocab_txt(GLOBAL_TF_VOCAB)

N_GENE_DIM = len(global_genes)
N_PEAK_DIM = len(global_peaks)
N_TF_DIM   = len(global_tfs)

print("[GLOBAL DIM] gene_dim =", N_GENE_DIM, " peak_dim =", N_PEAK_DIM, " tf_dim =", N_TF_DIM, flush=True)

# ============================================================
# 5) Build / load ALL pairs
# ============================================================
def build_pairs_all(out_tsv: Path):
    rows = []
    pairing_audit_rows = []

    for ds, (mr, ma) in manifest_map.items():
        rna = pd.read_csv(mr, sep="\t", dtype=str)
        atac = pd.read_csv(ma, sep="\t", dtype=str)

        scan_df_for_g_paths(rna, f"{ds} RNA manifest", warn_only=True, max_show=2)
        scan_df_for_g_paths(atac, f"{ds} ATAC manifest", warn_only=True, max_show=2)

        paired, strategy = pair_manifests_optimized(ds, rna, atac)

        print(
            f"[PAIRING] {ds:<30s} strategy={strategy:<15s} "
            f"rna={len(rna):>8d} atac={len(atac):>8d} paired={len(paired):>8d}",
            flush=True
        )

        pairing_audit_rows.append({
            "dataset": ds,
            "cancer_type": dataset_to_cancer_type(ds),
            "rna_rows": int(len(rna)),
            "atac_rows": int(len(atac)),
            "paired_rows": int(len(paired)),
            "pair_strategy": strategy,
        })

        if len(paired) == 0:
            print(f"[WARN] {ds} produced 0 pairs even after optimized pairing.", flush=True)
            continue

        sample_id_rna_out = paired["sample_id_norm_rna"].astype(str).str.strip()
        sample_id_atac_col = "sample_id_norm_atac" if "sample_id_norm_atac" in paired.columns else "sample_id_norm"
        sample_id_atac_out = paired[sample_id_atac_col].astype(str).str.strip()

        if strategy in ("sample_barcode", "barcode_only"):
            barcode_col = "barcode_norm_rna" if "barcode_norm_rna" in paired.columns else "barcode_norm"
            cell_id_unified = sample_id_rna_out + ":" + paired[barcode_col].astype(str)
        else:
            rowidx_col = "row_index_norm_rna" if "row_index_norm_rna" in paired.columns else "row_index_norm"
            cell_id_unified = sample_id_rna_out + ":ROWIDX_" + paired[rowidx_col].astype(str)

        out = pd.DataFrame({
            "dataset": ds,
            "cancer_type": dataset_to_cancer_type(ds),
            "sample_id_rna": sample_id_rna_out,
            "sample_id_atac": sample_id_atac_out,
            "cell_id_unified": cell_id_unified,
            "cell_id_rna": paired["cell_id_rna"].astype(str),
            "row_rna": paired["row_index_rna"].astype(int),
            "cell_id_atac": paired["cell_id_atac"].astype(str),
            "row_atac": paired["row_index_atac"].astype(int),
            "pair_strategy": strategy,
        })

        rows.append(out)

    if len(rows) == 0:
        raise RuntimeError("No paired cells produced by optimized pairing logic")

    pairs_df = pd.concat(rows, ignore_index=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(out_tsv, sep="\t", index=False)

    audit_df = pd.DataFrame(pairing_audit_rows).sort_values(["dataset"]).reset_index(drop=True)
    audit_path = out_tsv.parent / "cell_pairs_all.pairing_audit.tsv"
    audit_df.to_csv(audit_path, sep="\t", index=False)

    print(f"[PAIRING] saved pairs -> {out_tsv}", flush=True)
    print(f"[PAIRING] saved audit -> {audit_path}", flush=True)
    print("[PAIRING] paired input datasets =", pairs_df["dataset"].nunique(), flush=True)
    print("[PAIRING] paired input cells =", len(pairs_df), flush=True)

    by_strategy = pairs_df.groupby("pair_strategy", as_index=False).agg(
        n_datasets=("dataset", "nunique"),
        n_cells=("dataset", "size"),
    )
    print("[PAIRING] summary by strategy:", flush=True)
    print(by_strategy.sort_values(["n_cells", "n_datasets"], ascending=False).to_string(index=False), flush=True)

    return pairs_df

regen = True
if PAIRS_TSV_ALL.exists():
    try:
        tmp = pd.read_csv(PAIRS_TSV_ALL, sep="\t")
        scan_df_for_g_paths(tmp, "cell_pairs_all.tsv", warn_only=True, max_show=2)
        need_cols = {"dataset", "row_rna", "row_atac", "sample_id_rna", "sample_id_atac", "pair_strategy"}
        regen = (
            (not set(all_ds).issubset(set(tmp["dataset"].astype(str).unique())))
            or (not need_cols.issubset(set(tmp.columns)))
        )
    except Exception:
        regen = True

pairs_all = build_pairs_all(PAIRS_TSV_ALL) if regen else pd.read_csv(PAIRS_TSV_ALL, sep="\t")
pairs_all = pairs_all[pairs_all["dataset"].isin(all_ds)].reset_index(drop=True)
scan_df_for_g_paths(pairs_all, "pairs_all(active)", warn_only=True, max_show=2)

print("[PAIRS] total rows =", len(pairs_all), flush=True)
print("[PAIRS] input datasets after pairing =", pairs_all["dataset"].nunique(), flush=True)
print("[PAIRS] input cells after pairing =", len(pairs_all), flush=True)

if "pair_strategy" in pairs_all.columns:
    print("[PAIRS] strategy counts:", flush=True)
    print(pairs_all["pair_strategy"].value_counts(dropna=False).to_string(), flush=True)

# ============================================================
# 6) sample -> projected matrix paths
# ============================================================
rna_sample2mat = {}
atac_sample2mat = {}

valid_ds = []
invalid_ds_reasons = {}

for ds, (mr, ma) in manifest_map.items():
    rna = pd.read_csv(mr, sep="\t", dtype=str)
    atac = pd.read_csv(ma, sep="\t", dtype=str)

    bad_reason = None
    tmp_rna = {}
    tmp_atac = {}

    for sid, g in rna.groupby("sample_id"):
        sid = str(sid).strip()
        p = PROJECTED_ROOT / ds / f"{sid}.rna.global.csr.npz"
        if has_old_g_path(p):
            bad_reason = f"RNA projected path still points to G-drive: {p}"
            break
        if not p.exists():
            bad_reason = f"missing projected RNA for {ds}/{sid}: {p}"
            break
        tmp_rna[(ds, sid)] = p

    if bad_reason is None:
        for sid, g in atac.groupby("sample_id"):
            sid = str(sid).strip()
            p = PROJECTED_ROOT / ds / f"{sid}.atac.global.csr.npz"
            if has_old_g_path(p):
                bad_reason = f"ATAC projected path still points to G-drive: {p}"
                break
            if not p.exists():
                bad_reason = f"missing projected ATAC for {ds}/{sid}: {p}"
                break
            tmp_atac[(ds, sid)] = p

    if bad_reason is not None:
        invalid_ds_reasons[ds] = bad_reason
        print(f"[SKIP DATASET] {ds} -> {bad_reason}", flush=True)
        continue

    rna_sample2mat.update(tmp_rna)
    atac_sample2mat.update(tmp_atac)
    valid_ds.append(ds)

assert len(valid_ds) > 0, "No valid datasets with complete projected inputs found"

all_ds = sorted(valid_ds)
manifest_map = {ds: manifest_map[ds] for ds in all_ds}
pairs_all = pairs_all[pairs_all["dataset"].isin(all_ds)].reset_index(drop=True)

assert_mapping_paths_clean(rna_sample2mat, "rna_sample2mat")
assert_mapping_paths_clean(atac_sample2mat, "atac_sample2mat")

print("[PROJECTED MATRICES] RNA samples =", len(rna_sample2mat), " ATAC samples =", len(atac_sample2mat), flush=True)
print("[DATASETS] valid after projected check:", len(all_ds), flush=True)
print("[DATASETS] paired datasets after projected check:", pairs_all["dataset"].nunique(), flush=True)
print("[DATASETS] paired cells after projected check:", len(pairs_all), flush=True)
if invalid_ds_reasons:
    print("[DATASETS] skipped due to missing projected inputs:", len(invalid_ds_reasons), flush=True)

# ============================================================
# 7) CSR store
# ============================================================
class CSRNPZStore:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.data = None
        self.indices = None
        self.indptr = None
        self.shape = None

        if has_old_g_path(self.npz_path):
            raise ValueError(f"CSRNPZStore got old G-drive path: {self.npz_path}")

    def open(self):
        if self.data is None:
            z = np.load(self.npz_path, mmap_mode="r")
            self.data = z["data"]
            self.indices = z["indices"]
            self.indptr = z["indptr"]
            shp = tuple(z["shape"].tolist())
            self.shape = (int(shp[0]), int(shp[1]))

    def get_row_topk(self, i: int, topk: int):
        self.open()
        a = int(self.indptr[i])
        b = int(self.indptr[i + 1])
        return topk_from_csr_row(self.indices[a:b], self.data[a:b], topk)

_RNA_STORE_CACHE = {}
_ATAC_STORE_CACHE = {}

def get_rna_store(ds: str, sid_rna: str):
    key = (ds, sid_rna)
    if key not in _RNA_STORE_CACHE:
        _RNA_STORE_CACHE[key] = CSRNPZStore(rna_sample2mat[key])
    return _RNA_STORE_CACHE[key]

def get_atac_store(ds: str, sid_atac: str):
    key = (ds, sid_atac)
    if key not in _ATAC_STORE_CACHE:
        _ATAC_STORE_CACHE[key] = CSRNPZStore(atac_sample2mat[key])
    return _ATAC_STORE_CACHE[key]

# ============================================================
# 8) Dataset + sampler
# ============================================================
def make_cancer_maps(pairs_df: pd.DataFrame):
    cancer_types = sorted(pairs_df["cancer_type"].astype(str).unique().tolist())
    cancer2id = {c: i for i, c in enumerate(cancer_types)}
    return cancer_types, cancer2id

class PairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cancer2id):
        self.ds = df["dataset"].astype(str).tolist()
        self.sid_rna = df["sample_id_rna"].astype(str).tolist()
        self.sid_atac = df["sample_id_atac"].astype(str).tolist()
        self.r = df["row_rna"].astype(int).tolist()
        self.a = df["row_atac"].astype(int).tolist()
        self.ct_id = [cancer2id[str(c)] for c in df["cancer_type"].astype(str).tolist()]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i], self.sid_rna[i], self.sid_atac[i], self.r[i], self.a[i], self.ct_id[i]

def collate_batch(batch):
    t0 = time.time()

    rna_idx_all, rna_val_all, rna_off = [], [], [0]
    atac_idx_all, atac_val_all, atac_off = [], [], [0]
    cancer_ids = []

    for (ds, sid_rna, sid_atac, rr, ra, cid) in batch:
        cancer_ids.append(cid)

        rs = get_rna_store(ds, sid_rna)
        gi, gv = rs.get_row_topk(rr, CAP_RNA_TOKENS)
        mask = (gi >= 0) & (gi < N_GENE_DIM)
        gi = gi[mask]
        gv = gv[mask]
        rna_idx_all.append(gi)
        rna_val_all.append(gv)
        rna_off.append(rna_off[-1] + len(gi))

        ats = get_atac_store(ds, sid_atac)
        pi, pv = ats.get_row_topk(ra, CAP_ATAC_TOKENS)
        mask = (pi >= 0) & (pi < N_PEAK_DIM)
        pi = pi[mask]
        pv = pv[mask]
        atac_idx_all.append(pi)
        atac_val_all.append(pv)
        atac_off.append(atac_off[-1] + len(pi))

    rna_idx = torch.from_numpy(np.concatenate(rna_idx_all) if rna_idx_all else np.zeros((0,), np.int64))
    rna_val = torch.from_numpy(np.concatenate(rna_val_all) if rna_val_all else np.zeros((0,), np.float32))
    rna_off = torch.tensor(rna_off[:-1], dtype=torch.long)

    atac_idx = torch.from_numpy(np.concatenate(atac_idx_all) if atac_idx_all else np.zeros((0,), np.int64))
    atac_val = torch.from_numpy(np.concatenate(atac_val_all) if atac_val_all else np.zeros((0,), np.float32))
    atac_off = torch.tensor(atac_off[:-1], dtype=torch.long)

    cancer_ids = torch.tensor(cancer_ids, dtype=torch.long)

    if VERBOSE_COLLATE:
        dt = time.time() - t0
        print(
            f"[collate_batch] B={len(batch)} "
            f"rna_tokens={rna_idx.numel()} atac_tokens={atac_idx.numel()} "
            f"time={dt:.2f}s",
            flush=True
        )

    return rna_idx, rna_val, rna_off, atac_idx, atac_val, atac_off, cancer_ids

def make_train_loader(pairs_df: pd.DataFrame, cancer2id):
    ds = PairsDataset(pairs_df, cancer2id)
    ct_counts = np.bincount(np.array(ds.ct_id, dtype=np.int64), minlength=len(cancer2id)).astype(np.float64)
    w = np.array([1.0 / max(1.0, ct_counts[c]) for c in ds.ct_id], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(ds), replacement=True)

    kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_batch,
        drop_last=True,
    )

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        if PREFETCH_FACTOR is not None:
            kwargs["prefetch_factor"] = PREFETCH_FACTOR

    loader = DataLoader(**kwargs)
    return loader

def make_eval_loader(pairs_df: pd.DataFrame, cancer2id):
    ds = PairsDataset(pairs_df, cancer2id)

    kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_batch,
        drop_last=False,
    )

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        if PREFETCH_FACTOR is not None:
            kwargs["prefetch_factor"] = PREFETCH_FACTOR

    loader = DataLoader(**kwargs)
    return loader

# ============================================================
# 9) Load priors
# ============================================================
def load_global_priors(priors_root: Path):
    need = {
        "tf_gene_f":   priors_root / "tf_gene_f.full.npz",
        "gene_tf_r":   priors_root / "gene_tf_r.full.npz",
        "peak_gene_f": priors_root / "peak_gene_f.full.npz",
        "gene_peak_r": priors_root / "gene_peak_r.full.npz",
        "tf_peak_f":   priors_root / "tf_peak_f.full.npz",
        "peak_tf_r":   priors_root / "peak_tf_r.full.npz",
    }

    cache = {}
    for k, p in need.items():
        if has_old_g_path(p):
            raise ValueError(f"Prior path still points to old G-drive: {p}")
        if not p.exists():
            raise FileNotFoundError(f"missing prior file: {p}")
        z = np.load(p, mmap_mode="r")
        src = z["src"].astype(np.int64)
        dst = z["dst"].astype(np.int64)
        if len(src) == 0:
            raise RuntimeError(f"empty prior file: {p}")
        cache[k] = (src, dst)
        print(f"[PRIORS] {k}: edges={len(src):,}", flush=True)

    return cache

FULL_PRIORS = load_global_priors(PRIORS_ROOT)
print("[PRIORS] global priors ready", flush=True)

# ============================================================
# 10) Model (hash peak embedding)
# ============================================================
class PeakHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, d: int):
        super().__init__()
        self.hash_size = int(hash_size)
        self.emb = nn.Embedding(self.hash_size, d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, peak_idx: torch.Tensor):
        hashed = torch.remainder(peak_idx, self.hash_size)
        return self.emb(hashed)

class PlanC_DirAware_Cancer_HashPeak(nn.Module):
    def __init__(self, n_gene, n_tf, n_cancer, peak_hash_size=PEAK_HASH_SIZE, d=D, hid=HID, drop=DROP):
        super().__init__()
        self.peak_hash_size = peak_hash_size

        self.gene_in  = nn.Embedding(n_gene, d)
        self.gene_out = nn.Embedding(n_gene, d)

        self.tf_in    = nn.Embedding(n_tf, d)
        self.tf_out   = nn.Embedding(n_tf, d)

        self.peak_in  = PeakHashEmbedding(peak_hash_size, d)
        self.peak_out = PeakHashEmbedding(peak_hash_size, d)

        self.gene_in_c  = nn.Embedding(n_cancer, d)
        self.gene_out_c = nn.Embedding(n_cancer, d)
        self.peak_in_c  = nn.Embedding(n_cancer, d)
        self.peak_out_c = nn.Embedding(n_cancer, d)
        self.tf_in_c    = nn.Embedding(n_cancer, d)
        self.tf_out_c   = nn.Embedding(n_cancer, d)
        self.cell_c     = nn.Embedding(n_cancer, d)

        self.rna_proj  = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d))
        self.atac_proj = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, d))

        for emb in [
            self.gene_in, self.gene_out,
            self.tf_in, self.tf_out,
            self.gene_in_c, self.gene_out_c,
            self.peak_in_c, self.peak_out_c,
            self.tf_in_c, self.tf_out_c,
            self.cell_c,
        ]:
            nn.init.normal_(emb.weight, std=0.02)

    @staticmethod
    def _seg_ids_from_offsets(offsets: torch.Tensor, total_tokens: int):
        B = offsets.shape[0]
        end = torch.empty((B,), device=offsets.device, dtype=torch.long)
        end[:-1] = offsets[1:]
        end[-1] = total_tokens
        lengths = (end - offsets).clamp(min=0)
        seg_ids = torch.repeat_interleave(torch.arange(B, device=offsets.device), lengths)
        return seg_ids

    def encode_rna(self, idx, val, offsets, cancer_ids):
        seg_ids = self._seg_ids_from_offsets(offsets, idx.shape[0])
        c_tok = cancer_ids.index_select(0, seg_ids)
        x = (self.gene_in(idx) + self.gene_in_c(c_tok)) * val.unsqueeze(-1)
        out = torch.zeros((offsets.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        out.index_add_(0, seg_ids, x)
        out = self.rna_proj(out) + self.cell_c(cancer_ids)
        return F.normalize(out, dim=-1)

    def encode_atac(self, idx, val, offsets, cancer_ids):
        seg_ids = self._seg_ids_from_offsets(offsets, idx.shape[0])
        c_tok = cancer_ids.index_select(0, seg_ids)
        x = (self.peak_in(idx) + self.peak_in_c(c_tok)) * val.unsqueeze(-1)
        out = torch.zeros((offsets.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        out.index_add_(0, seg_ids, x)
        out = self.atac_proj(out) + self.cell_c(cancer_ids)
        return F.normalize(out, dim=-1)

    def score_tf_gene_f(self, tf_i, gene_i, c):
        return ((self.tf_in(tf_i) + self.tf_in_c(c)) * (self.gene_out(gene_i) + self.gene_out_c(c))).sum(-1)

    def score_gene_tf_r(self, gene_i, tf_i, c):
        return ((self.gene_in(gene_i) + self.gene_in_c(c)) * (self.tf_out(tf_i) + self.tf_out_c(c))).sum(-1)

    def score_peak_gene_f(self, peak_i, gene_i, c):
        return ((self.peak_in(peak_i) + self.peak_in_c(c)) * (self.gene_out(gene_i) + self.gene_out_c(c))).sum(-1)

    def score_gene_peak_r(self, gene_i, peak_i, c):
        return ((self.gene_in(gene_i) + self.gene_in_c(c)) * (self.peak_out(peak_i) + self.peak_out_c(c))).sum(-1)

    def score_tf_peak_f(self, tf_i, peak_i, c):
        return ((self.tf_in(tf_i) + self.tf_in_c(c)) * (self.peak_out(peak_i) + self.peak_out_c(c))).sum(-1)

    def score_peak_tf_r(self, peak_i, tf_i, c):
        return ((self.peak_in(peak_i) + self.peak_in_c(c)) * (self.tf_out(tf_i) + self.tf_out_c(c))).sum(-1)

    def cancer_l2(self):
        regs = [
            self.gene_in_c.weight, self.gene_out_c.weight,
            self.peak_in_c.weight, self.peak_out_c.weight,
            self.tf_in_c.weight, self.tf_out_c.weight,
            self.cell_c.weight,
        ]
        return sum((w * w).mean() for w in regs)

# ============================================================
# 11) Train / eval
# ============================================================
@torch.no_grad()
def sample_edges(src_all: torch.Tensor, dst_all: torch.Tensor, batch: int):
    n = src_all.shape[0]
    if n == 0:
        raise RuntimeError("empty prior edges")
    idx = torch.randint(0, n, (batch,), device=src_all.device)
    return src_all[idx], dst_all[idx]

def bce_graph_loss(pos_logits, neg_logits):
    pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return pos + neg

def sample_c_from_batch(batch_c: torch.Tensor, n: int):
    ridx = torch.randint(0, batch_c.shape[0], (n,), device=batch_c.device)
    return batch_c[ridx]

def eval_alignment_top1(model, loader):
    model.eval()
    tot = 0
    hit = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        for rna_idx, rna_val, rna_off, atac_idx, atac_val, atac_off, cancer_ids in loader:
            rna_idx = rna_idx.to(DEVICE, non_blocking=True)
            rna_val = rna_val.to(DEVICE, non_blocking=True)
            rna_off = rna_off.to(DEVICE, non_blocking=True)
            atac_idx = atac_idx.to(DEVICE, non_blocking=True)
            atac_val = atac_val.to(DEVICE, non_blocking=True)
            atac_off = atac_off.to(DEVICE, non_blocking=True)
            cancer_ids = cancer_ids.to(DEVICE, non_blocking=True)

            z_rna = model.encode_rna(rna_idx, rna_val, rna_off, cancer_ids)
            z_atac = model.encode_atac(atac_idx, atac_val, atac_off, cancer_ids)
            sim = (z_rna @ z_atac.t())
            pred = sim.argmax(dim=1)
            target = torch.arange(sim.shape[0], device=DEVICE)
            hit += (pred == target).sum().item()
            tot += sim.shape[0]

    model.train()
    return hit / max(1, tot)

def train_loop(model, opt, train_loader, priors, steps, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = out_dir / f"ckpt_last_{tag}.pt"

    if has_old_g_path(out_dir) or has_old_g_path(ckpt_last):
        raise ValueError(f"[{tag}] output path still points to G-drive: out_dir={out_dir}, ckpt={ckpt_last}")

    print(f"[{tag}] loading priors to GPU ...", flush=True)
    t_pr = time.time()

    tf_gene_f_src   = torch.from_numpy(priors["tf_gene_f"][0]).to(DEVICE)
    tf_gene_f_dst   = torch.from_numpy(priors["tf_gene_f"][1]).to(DEVICE)
    gene_tf_r_src   = torch.from_numpy(priors["gene_tf_r"][0]).to(DEVICE)
    gene_tf_r_dst   = torch.from_numpy(priors["gene_tf_r"][1]).to(DEVICE)
    peak_gene_f_src = torch.from_numpy(priors["peak_gene_f"][0]).to(DEVICE)
    peak_gene_f_dst = torch.from_numpy(priors["peak_gene_f"][1]).to(DEVICE)
    gene_peak_r_src = torch.from_numpy(priors["gene_peak_r"][0]).to(DEVICE)
    gene_peak_r_dst = torch.from_numpy(priors["gene_peak_r"][1]).to(DEVICE)
    tf_peak_f_src   = torch.from_numpy(priors["tf_peak_f"][0]).to(DEVICE)
    tf_peak_f_dst   = torch.from_numpy(priors["tf_peak_f"][1]).to(DEVICE)
    peak_tf_r_src   = torch.from_numpy(priors["peak_tf_r"][0]).to(DEVICE)
    peak_tf_r_dst   = torch.from_numpy(priors["peak_tf_r"][1]).to(DEVICE)

    print(f"[{tag}] priors ready in {(time.time()-t_pr)/60:.2f} min", flush=True)

    autocast_ctx = torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP)

    def save_ckpt(step):
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "meta": {
                    "PEAK_HASH_SIZE": PEAK_HASH_SIZE,
                    "N_GENE_DIM": N_GENE_DIM,
                    "N_PEAK_DIM": N_PEAK_DIM,
                    "N_TF_DIM": N_TF_DIM,
                },
            },
            ckpt_last,
        )

    def graph_step_tf_gene_f(batch_c):
        tf_i, g_pos = sample_edges(tf_gene_f_src, tf_gene_f_dst, EDGE_BATCH)
        c = sample_c_from_batch(batch_c, EDGE_BATCH)
        g_neg = torch.randint(0, N_GENE_DIM, (EDGE_BATCH * NEG_K,), device=DEVICE)
        tf_rep = tf_i.repeat_interleave(NEG_K)
        c_rep = c.repeat_interleave(NEG_K)
        return bce_graph_loss(model.score_tf_gene_f(tf_i, g_pos, c), model.score_tf_gene_f(tf_rep, g_neg, c_rep))

    def graph_step_gene_tf_r(batch_c):
        g_i, tf_pos = sample_edges(gene_tf_r_src, gene_tf_r_dst, EDGE_BATCH)
        c = sample_c_from_batch(batch_c, EDGE_BATCH)
        tf_neg = torch.randint(0, N_TF_DIM, (EDGE_BATCH * NEG_K,), device=DEVICE)
        g_rep = g_i.repeat_interleave(NEG_K)
        c_rep = c.repeat_interleave(NEG_K)
        return bce_graph_loss(model.score_gene_tf_r(g_i, tf_pos, c), model.score_gene_tf_r(g_rep, tf_neg, c_rep))

    def graph_step_peak_gene_f(batch_c):
        p_i, g_pos = sample_edges(peak_gene_f_src, peak_gene_f_dst, EDGE_BATCH)
        c = sample_c_from_batch(batch_c, EDGE_BATCH)
        g_neg = torch.randint(0, N_GENE_DIM, (EDGE_BATCH * NEG_K,), device=DEVICE)
        p_rep = p_i.repeat_interleave(NEG_K)
        c_rep = c.repeat_interleave(NEG_K)
        return bce_graph_loss(model.score_peak_gene_f(p_i, g_pos, c), model.score_peak_gene_f(p_rep, g_neg, c_rep))

    def graph_step_gene_peak_r(batch_c):
        g_i, p_pos = sample_edges(gene_peak_r_src, gene_peak_r_dst, EDGE_BATCH)
        c = sample_c_from_batch(batch_c, EDGE_BATCH)
        p_neg = torch.randint(0, N_PEAK_DIM, (EDGE_BATCH * NEG_K,), device=DEVICE)
        g_rep = g_i.repeat_interleave(NEG_K)
        c_rep = c.repeat_interleave(NEG_K)
        return bce_graph_loss(model.score_gene_peak_r(g_i, p_pos, c), model.score_gene_peak_r(g_rep, p_neg, c_rep))

    def graph_step_tf_peak_f(batch_c):
        tf_i, p_pos = sample_edges(tf_peak_f_src, tf_peak_f_dst, EDGE_BATCH)
        c = sample_c_from_batch(batch_c, EDGE_BATCH)
        p_neg = torch.randint(0, N_PEAK_DIM, (EDGE_BATCH * NEG_K,), device=DEVICE)
        tf_rep = tf_i.repeat_interleave(NEG_K)
        c_rep = c.repeat_interleave(NEG_K)
        return bce_graph_loss(model.score_tf_peak_f(tf_i, p_pos, c), model.score_tf_peak_f(tf_rep, p_neg, c_rep))

    def graph_step_peak_tf_r(batch_c):
        p_i, tf_pos = sample_edges(peak_tf_r_src, peak_tf_r_dst, EDGE_BATCH)
        c = sample_c_from_batch(batch_c, EDGE_BATCH)
        tf_neg = torch.randint(0, N_TF_DIM, (EDGE_BATCH * NEG_K,), device=DEVICE)
        p_rep = p_i.repeat_interleave(NEG_K)
        c_rep = c.repeat_interleave(NEG_K)
        return bce_graph_loss(model.score_peak_tf_r(p_i, tf_pos, c), model.score_peak_tf_r(p_rep, tf_neg, c_rep))

    model.train()
    step = 0
    t0 = time.time()

    print(f"[{tag}] building dataloader iterator ...", flush=True)
    t_it = time.time()
    it = iter(train_loader)
    print(f"[{tag}] iterator ready in {(time.time()-t_it):.1f}s", flush=True)

    print(f"[{tag}] fetching first batch ...", flush=True)
    t_fb = time.time()
    try:
        first_batch = next(it)
    except StopIteration:
        it = iter(train_loader)
        first_batch = next(it)
    print(f"[{tag}] first batch ready in {(time.time()-t_fb):.1f}s", flush=True)

    batch = first_batch

    while step < steps:
        step += 1

        rna_idx, rna_val, rna_off, atac_idx, atac_val, atac_off, cancer_ids = batch
        rna_idx = rna_idx.to(DEVICE, non_blocking=True)
        rna_val = rna_val.to(DEVICE, non_blocking=True)
        rna_off = rna_off.to(DEVICE, non_blocking=True)
        atac_idx = atac_idx.to(DEVICE, non_blocking=True)
        atac_val = atac_val.to(DEVICE, non_blocking=True)
        atac_off = atac_off.to(DEVICE, non_blocking=True)
        cancer_ids = cancer_ids.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with autocast_ctx:
            z_rna = model.encode_rna(rna_idx, rna_val, rna_off, cancer_ids)
            z_atac = model.encode_atac(atac_idx, atac_val, atac_off, cancer_ids)

            logits = (z_rna @ z_atac.t()) / CL_TEMP
            target = torch.arange(logits.shape[0], device=DEVICE)
            loss_align = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

            loss_tf_gene_f   = graph_step_tf_gene_f(cancer_ids)
            loss_gene_tf_r   = graph_step_gene_tf_r(cancer_ids)
            loss_peak_gene_f = graph_step_peak_gene_f(cancer_ids)
            loss_gene_peak_r = graph_step_gene_peak_r(cancer_ids)
            loss_tf_peak_f   = graph_step_tf_peak_f(cancer_ids)
            loss_peak_tf_r   = graph_step_peak_tf_r(cancer_ids)

            loss_graph = (
                W_TF_GENE_F   * loss_tf_gene_f +
                W_GENE_TF_R   * loss_gene_tf_r +
                W_PEAK_GENE_F * loss_peak_gene_f +
                W_GENE_PEAK_R * loss_gene_peak_r +
                W_TF_PEAK_F   * loss_tf_peak_f +
                W_PEAK_TF_R   * loss_peak_tf_r
            )

            loss_reg = LAMBDA_CANCER_L2 * model.cancer_l2()
            loss = W_ALIGN * loss_align + loss_graph + loss_reg

        loss.backward()
        if CLIP and CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            dt = (time.time() - t0) / 60
            print(
                f"[{tag}] step={step:6d}/{steps} "
                f"loss={loss.item():.4f} "
                f"align={loss_align.item():.4f} "
                f"graph={loss_graph.item():.4f} "
                f"reg={loss_reg.item():.2e} "
                f"| tf->g={loss_tf_gene_f.item():.4f} "
                f"g->tf={loss_gene_tf_r.item():.4f} "
                f"p->g={loss_peak_gene_f.item():.4f} "
                f"g->p={loss_gene_peak_r.item():.4f} "
                f"tf->p={loss_tf_peak_f.item():.4f} "
                f"p->tf={loss_peak_tf_r.item():.4f} "
                f"| {dt:.1f} min",
                flush=True
            )

        if step % SAVE_EVERY == 0:
            save_ckpt(step)
            print(f"[{tag}] ckpt saved -> {ckpt_last}", flush=True)

        if step < steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

    save_ckpt(step)
    print(f"[{tag}] finished. ckpt saved -> {ckpt_last}", flush=True)
    return ckpt_last

# ============================================================
# 12) LODO main loop
# ============================================================
existing_rows_by_holdout, loaded_summary_from = load_existing_qc_summary()
if loaded_summary_from is not None:
    print(f"[QC] existing summary loaded from: {loaded_summary_from}", flush=True)
    print(f"[QC] existing completed/recorded holdouts: {len(existing_rows_by_holdout)}", flush=True)
else:
    print("[QC] no existing summary found, start fresh.", flush=True)

qc_rows_by_holdout = dict(existing_rows_by_holdout)

for holdout_ds in all_ds:
    print("\n" + "=" * 100, flush=True)
    print("[QC] HOLDOUT =", holdout_ds, flush=True)

    if SKIP_COMPLETED_DATASETS and holdout_is_completed(holdout_ds, qc_rows_by_holdout):
        print(f"[QC] skip already completed holdout: {holdout_ds}", flush=True)
        continue

    pairs_train = pairs_all[pairs_all["dataset"] != holdout_ds].reset_index(drop=True)
    pairs_hold = pairs_all[pairs_all["dataset"] == holdout_ds].reset_index(drop=True)

    if len(pairs_hold) == 0:
        print("[QC] skip (no holdout pairs):", holdout_ds, flush=True)
        continue

    cancer_types, cancer2id = make_cancer_maps(pairs_train)

    train_loader = make_train_loader(pairs_train, cancer2id)
    eval_loader = make_eval_loader(
        pairs_hold.assign(
            cancer_type=pairs_hold["cancer_type"].where(
                pairs_hold["cancer_type"].isin(cancer2id),
                cancer_types[0]
            )
        ),
        cancer2id
    )

    if RUN_DEBUG_BATCH_CHECK:
        print("[DEBUG] trying one batch manually...", flush=True)
        t_dbg = time.time()
        it_dbg = iter(train_loader)
        b0 = next(it_dbg)
        print(f"[DEBUG] one batch ok in {time.time()-t_dbg:.1f}s", flush=True)
        for i, x in enumerate(b0):
            if torch.is_tensor(x):
                print(f"[DEBUG] batch[{i}] shape={tuple(x.shape)} dtype={x.dtype}", flush=True)
            else:
                print(f"[DEBUG] batch[{i}] type={type(x)}", flush=True)

    run_dir = QC_RUNS / holdout_ds
    run_dir.mkdir(parents=True, exist_ok=True)

    if has_old_g_path(run_dir):
        raise ValueError(f"run_dir still points to old G-drive: {run_dir}")

    model = PlanC_DirAware_Cancer_HashPeak(
        n_gene=N_GENE_DIM,
        n_tf=N_TF_DIM,
        n_cancer=len(cancer_types),
        peak_hash_size=PEAK_HASH_SIZE,
        d=D,
        hid=HID,
        drop=DROP,
    ).to(DEVICE)

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    print(f"[QC] baseline training start for holdout={holdout_ds}", flush=True)
    ckpt_base = train_loop(
        model=model,
        opt=opt,
        train_loader=train_loader,
        priors=FULL_PRIORS,
        steps=BASE_STEPS,
        out_dir=run_dir,
        tag="BASE_global_hashpeak",
    )

    acc0 = eval_alignment_top1(model, eval_loader)
    print("[QC] holdout top1 (before transfer) =", acc0, flush=True)

    pairs_hold2 = pairs_hold.copy()
    pairs_hold2["cancer_type"] = pairs_hold2["cancer_type"].where(
        pairs_hold2["cancer_type"].isin(cancer2id),
        cancer_types[0]
    )
    hold_loader = make_train_loader(pairs_hold2, cancer2id)

    print(f"[QC] transfer training start for holdout={holdout_ds}", flush=True)
    ckpt_tr = train_loop(
        model=model,
        opt=opt,
        train_loader=hold_loader,
        priors=FULL_PRIORS,
        steps=TRANSFER_STEPS,
        out_dir=run_dir,
        tag="TRANSFER_global_hashpeak",
    )

    acc1 = eval_alignment_top1(model, eval_loader)
    print("[QC] holdout top1 (after transfer)  =", acc1, flush=True)

    row = {
        "holdout_ds": holdout_ds,
        "n_train_pairs": int(len(pairs_train)),
        "n_hold_pairs": int(len(pairs_hold)),
        "top1_before_transfer": float(acc0),
        "top1_after_transfer": float(acc1),
        "ckpt_base": str(ckpt_base),
        "ckpt_transfer": str(ckpt_tr),
        "priors_mode_baseline": "GLOBAL_FULL_HASH_PEAK",
        "priors_mode_transfer": "GLOBAL_FULL_HASH_PEAK",
        "peak_hash_size": int(PEAK_HASH_SIZE),
        "train_pairs_exclude_holdout": True,
    }

    upsert_qc_row(qc_rows_by_holdout, row)

    qc_df_mid = pd.DataFrame(list(qc_rows_by_holdout.values()))
    qc_df_mid = qc_df_mid.sort_values("holdout_ds").reset_index(drop=True)
    qc_df_mid.to_csv(SUMMARY_PARTIAL, sep="\t", index=False)
    print("[QC] partial summary updated.", flush=True)

qc_df = pd.DataFrame(list(qc_rows_by_holdout.values()))
if len(qc_df) > 0:
    qc_df = qc_df.sort_values("holdout_ds").reset_index(drop=True)

out_sum = SUMMARY_FINAL
qc_df.to_csv(out_sum, sep="\t", index=False)
print("\n[DONE] QC summary ->", out_sum, flush=True)

try:
    print(qc_df.sort_values("top1_after_transfer", ascending=False).head(30))
except Exception:
    print(qc_df.sort_values("top1_after_transfer", ascending=False).head(30))
