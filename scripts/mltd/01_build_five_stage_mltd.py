# -*- coding: utf-8 -*-
"""
integrated_five_stage_lamb1_network_viz_v7_3d_fixed.py

功能：
1. 保留 Stage1/2/3/4 的原有构图逻辑
2. 新增 Stage5 物理启发网络：
   - 节点场强 ~ 表达量
   - 边传播能力 ~ prior + representation + communication + temporal
   - 支持 virtual KO / OE
3. 新增 Stage5 三维可视化：
   - baseline 3D field network
   - perturbation response 3D network

修正：
- 修复 compute_edge_flux_from_state 中重复列 rename 导致的
  "TypeError: cannot convert the series to float"
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform


# ============================================================
# 0) CONFIG
# ============================================================

TARGET_GENE = "LAMB1"

BASE = get_project_root()
DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

WORLD_ROOT = BASE / "LLM-WorldWord"
GLOBAL_ROOT = BASE / "networks" / "qc_global_projection"
PROJECTED_ROOT = GLOBAL_ROOT / "projected_inputs"
PRIORS_DIR = GLOBAL_ROOT / "priors_cache"
MAPS_ROOT = GLOBAL_ROOT / "maps"

GLOBAL_GENE_VOCAB = MAPS_ROOT / "global_gene_vocab.txt"
GLOBAL_TF_VOCAB = MAPS_ROOT / "global_tf_vocab.txt"

RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)

CLUSTERS_CSV = RUN_STAGEB_DIR / "clusters.csv"

CCI_DIR = RUN_STAGEB_DIR / f"{SAMPLE_ID}_cci_newLeiden"
PT_DIR = CCI_DIR / "communication_pseudotime_clusterGNN_LTC"

RNA_DIR = WORLD_ROOT / DATASET_NAME / "processed" / "RNA"
PROJECTED_RNA = PROJECTED_ROOT / DATASET_NAME / f"{SAMPLE_ID}.rna.global.csr.npz"
RNA_META = RNA_DIR / f"{SAMPLE_ID}.rna_meta.npz"
RNA_COUNTS = RNA_DIR / f"{SAMPLE_ID}.X_rna_counts_dense.npy"
RNA_MANIFEST = RNA_DIR / "manifest_rna.tsv"

CCI_TFKEPT_CSV = CCI_DIR / "interactions_cluster_level_highconf_TFkept.csv"
RECEPTOR_TF_SIG_CSV = CCI_DIR / "receptor_tf_signature.csv"

CLUSTER_PT_CSV = PT_DIR / "cluster_pseudotime.csv"
EDGE_PT_CSV = PT_DIR / "edge_pseudotime_summary.csv"

OUT_DIR = Path("./five_stage_network_viz_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREFER_FULL_WITH_DECAY = True
DECAY_AFTER_256 = 0.35
DECAY_MODE = "constant"

TOPK_STAGE1_TF = 8
TOPK_STAGE1_GENES = 12

TOPK_STAGE2_GENES = 12
STAGE2_TARGET_CELL_Q = 0.85
STAGE2_MIN_TARGET_CELLS = 40
STAGE2_PROJECTED_ROW_TOPK = 2048

TOPK_STAGE3_ANNOT = 16
TOPK_STAGE4_ANNOT = 16

FIGSIZE = (13, 10)
DPI = 260

# ============================================================
# 0b) STAGE5 CONFIG
# ============================================================

STAGE5_MODE = "oe"                  # "ko" or "oe"
STAGE5_PERTURB_STRENGTH = 0.75      # ko: 压低比例; oe: 增强比例
STAGE5_ALPHA = 1.6                  # field = expr^alpha
STAGE5_BETA = 1.25                  # susc = (1-expr)^beta + eps
STAGE5_EPS = 0.05

STAGE5_A1 = 1.0                     # prior/base
STAGE5_A2 = 0.8                     # representation
STAGE5_A3 = 1.15                    # communication
STAGE5_A4 = 1.0                     # temporal

STAGE5_KAPPA = 0.55                 # propagation strength
STAGE5_MU = 0.35                    # update step
STAGE5_MAX_ITER = 20
STAGE5_TOL = 1e-4

STAGE5_TOPK_HIGHLIGHT_EDGES = 18
STAGE5_TOPK_HIGHLIGHT_NODES = 14
STAGE5_USE_TWO_HOP_PROPAGATION = True


# ============================================================
# 1) HELPERS
# ============================================================

def read_vocab_txt(path: Path) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def safe_name(x: str, n: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n]


def normalize_cluster_str(x: str) -> str:
    s = str(x).strip()
    if s.startswith("C") and s[1:].isdigit():
        return s[1:]
    return s


def normalize_gene_name(x: str) -> str:
    return str(x).strip() if x is not None else ""


def norm_barcode(x: str) -> str:
    x = str(x)
    return x.split("::", 1)[-1] if "::" in x else x


def find_first_existing_col(df: pd.DataFrame, cand_cols: List[str]) -> Optional[str]:
    for c in cand_cols:
        if c in df.columns:
            return c
    return None


def save_figure_both(fig: plt.Figure, out_base: Path):
    fig.savefig(str(out_base.with_suffix(".png")), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".svg")), bbox_inches="tight")


def edge_width(w: float, low=1.0, high=4.2) -> float:
    if not np.isfinite(w):
        return low
    x = math.log1p(max(0.0, float(w)))
    return low + (high - low) * min(1.0, x / math.log1p(10.0))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _safe_minmax_scale(values: Dict, floor: float = 0.0, ceil: float = 1.0) -> Dict:
    if len(values) == 0:
        return {}
    xs = np.array(list(values.values()), dtype=float)
    finite = np.isfinite(xs)
    if finite.sum() == 0:
        return {k: floor for k in values}
    lo = float(np.nanmin(xs[finite]))
    hi = float(np.nanmax(xs[finite]))
    if hi - lo < 1e-12:
        mid = 0.5 * (floor + ceil)
        return {k: mid for k in values}
    out = {}
    for k, v in values.items():
        if not np.isfinite(v):
            out[k] = floor
        else:
            z = (float(v) - lo) / (hi - lo)
            out[k] = floor + (ceil - floor) * z
    return out


def _safe_log1p_scale(values: Dict) -> Dict:
    tmp = {}
    for k, v in values.items():
        vv = 0.0 if (not np.isfinite(v)) else max(0.0, float(v))
        tmp[k] = math.log1p(vv)
    return _safe_minmax_scale(tmp)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


# ============================================================
# 2) SAMPLE / STORE
# ============================================================

@dataclass
class Sample:
    sid: str
    cells: np.ndarray
    row_indices: np.ndarray
    proj_rna_path: Path
    counts_path: Path
    counts_shape: tuple[int, int]
    gene_names: np.ndarray
    ext2train: np.ndarray
    train2rna: dict[int, int]
    bc2rid: dict[str, int]


class CSRNPZStore:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)
        self.data = None
        self.indices = None
        self.indptr = None
        self.shape = None

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
        idx = self.indices[a:b]
        val = self.data[a:b]
        if idx.size == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
        k = min(topk, idx.size)
        sel = np.argpartition(val, -k)[-k:]
        idx = idx[sel]
        val = val[sel]
        order = np.argsort(val)[::-1]
        return idx[order].astype(np.int64), val[order].astype(np.float32)


def load_single_sample(dataset_name: str, sample_id: str, genes_vocab: list[str]) -> Sample:
    manifest_path = RNA_MANIFEST
    df = pd.read_csv(manifest_path, sep="\t")
    g = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    g = g.sort_values("row_index").reset_index(drop=True)

    z = np.load(RNA_META, allow_pickle=True)
    gene_names = np.asarray(z["gene_names"]).astype(str)

    cells = np.asarray([norm_barcode(x) for x in g["cell_id"].astype(str).tolist()], dtype=object)
    row_indices = g["row_index"].astype(np.int64).values

    counts_mm = np.load(RNA_COUNTS, mmap_mode="r")

    gene2i_vocab = {gg: i for i, gg in enumerate(genes_vocab)}
    ext2train = np.full((len(gene_names),), -1, dtype=np.int64)
    train2rna = {}
    for j, gname in enumerate(gene_names):
        gname = normalize_gene_name(gname)
        if gname in gene2i_vocab:
            tid = gene2i_vocab[gname]
            ext2train[j] = tid
            if tid not in train2rna:
                train2rna[tid] = j

    bc2rid = {str(bc): i for i, bc in enumerate(cells.tolist())}

    return Sample(
        sid=sample_id,
        cells=cells,
        row_indices=row_indices,
        proj_rna_path=PROJECTED_RNA,
        counts_path=RNA_COUNTS,
        counts_shape=tuple(counts_mm.shape),
        gene_names=gene_names,
        ext2train=ext2train,
        train2rna=train2rna,
        bc2rid=bc2rid,
    )


def get_counts_col_dense(sample: Sample, gene_idx_in_sample: int) -> np.ndarray:
    x = np.load(sample.counts_path, mmap_mode="r")
    col = np.asarray(x[:, gene_idx_in_sample]).reshape(-1)
    return col.astype(np.float32, copy=False)


# ============================================================
# 3) STAGE1
# ============================================================

def resolve_tf_gene_prior_file(priors_dir: Path) -> Path:
    p1 = priors_dir / "tf_gene_f.cap256.npz"
    if p1.exists():
        return p1
    p2 = priors_dir / "tf_gene_f.full.npz"
    if p2.exists():
        return p2
    cands = sorted(priors_dir.glob("*tf*gene*.npz"))
    if len(cands) > 0:
        return cands[0]
    raise FileNotFoundError(f"在 {priors_dir} 下找不到 tf-gene prior 文件")


def load_tf_gene_prior_edges(
    priors_dir: Path,
    tf_vocab_txt: Path,
    gene_vocab_txt: Path,
    prefer_full_with_decay: bool = True,
    decay_after_256: float = 0.35,
    decay_mode: str = "constant",
) -> pd.DataFrame:
    tf_vocab = read_vocab_txt(tf_vocab_txt)
    gene_vocab = read_vocab_txt(gene_vocab_txt)

    p_cap = priors_dir / "tf_gene_f.cap256.npz"
    p_full = priors_dir / "tf_gene_f.full.npz"

    if prefer_full_with_decay and p_full.exists():
        prior_path = p_full
        use_decay = True
    else:
        if p_cap.exists():
            prior_path = p_cap
            use_decay = False
        elif p_full.exists():
            prior_path = p_full
            use_decay = True
        else:
            prior_path = resolve_tf_gene_prior_file(priors_dir)
            use_decay = ("full" in prior_path.name.lower())

    z = np.load(prior_path, allow_pickle=True)
    tf_src = z["src"].astype(np.int64)
    g_dst = z["dst"].astype(np.int64)

    rows = []
    for s, d in zip(tf_src.tolist(), g_dst.tolist()):
        if not (0 <= s < len(tf_vocab) and 0 <= d < len(gene_vocab)):
            continue

        w = 1.0
        after_256 = int(d >= 256)

        if use_decay and d >= 256:
            if decay_mode == "constant":
                w = float(decay_after_256)
            elif decay_mode == "sqrt":
                w = float(decay_after_256 / np.sqrt(1.0 + (d - 255)))
            elif decay_mode == "linear":
                w = float(max(0.1, decay_after_256 - 0.0005 * (d - 255)))
            else:
                w = float(decay_after_256)

        rows.append({
            "tf": str(tf_vocab[s]),
            "gene": str(gene_vocab[d]),
            "weight": float(w),
            "is_after_256": after_256,
        })

    return pd.DataFrame(rows)


def build_stage1_core_graph(target_gene: str, prior_df: pd.DataFrame) -> nx.DiGraph:
    hit = prior_df[prior_df["gene"].astype(str) == target_gene].copy()

    tf_rank = (
        hit.groupby("tf", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .head(TOPK_STAGE1_TF)
    )
    keep_tfs = set(tf_rank["tf"].astype(str).tolist())

    nbr = prior_df[prior_df["tf"].astype(str).isin(keep_tfs)].copy()
    gene_rank = (
        nbr.groupby("gene", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
    )

    keep_genes = [target_gene]
    for g in gene_rank["gene"].astype(str).tolist():
        if g != target_gene:
            keep_genes.append(g)
        if len(keep_genes) >= TOPK_STAGE1_GENES:
            break

    sub = nbr[nbr["gene"].astype(str).isin(keep_genes)].copy()

    G = nx.DiGraph()
    G.add_node(target_gene, kind="target_gene")

    for _, r in sub.iterrows():
        tf = str(r["tf"])
        ge = str(r["gene"])
        if tf not in G:
            G.add_node(tf, kind="tf")
        if ge not in G:
            G.add_node(ge, kind="target_gene" if ge == target_gene else "gene")
        G.add_edge(
            tf, ge,
            weight=float(r["weight"]),
            edge_type="prior_tf_gene",
            is_after_256=int(r.get("is_after_256", 0)),
        )
    return G


# ============================================================
# 4) STAGE2
# ============================================================

def auto_make_stage2_edge_table(
    target_gene: str,
    sample: Sample,
    gene_vocab_txt: Path,
    stage1_nodes: Set[str],
    topk_genes: int = 8,
    target_cell_q: float = 0.85,
    min_target_cells: int = 40,
    projected_row_topk: int = 2048,
) -> pd.DataFrame:
    genes_vocab = read_vocab_txt(gene_vocab_txt)
    target_tid = genes_vocab.index(target_gene)
    target_rna_col = sample.train2rna[target_tid]
    target_expr = get_counts_col_dense(sample, target_rna_col)

    finite_mask = np.isfinite(target_expr)
    x = target_expr[finite_mask]
    thr = float(np.quantile(x, target_cell_q))
    target_rows = np.where(target_expr >= thr)[0]
    if target_rows.size < min_target_cells:
        order = np.argsort(target_expr)[::-1]
        target_rows = order[:min(min_target_cells, len(order))]

    store = CSRNPZStore(sample.proj_rna_path)
    token_weight_sum = defaultdict(float)
    token_hit_count = defaultdict(int)

    for rid in target_rows.tolist():
        gi, gv = store.get_row_topk(int(rid), int(projected_row_topk))
        for gidx, gval in zip(gi.tolist(), gv.tolist()):
            if 0 <= int(gidx) < len(genes_vocab):
                token_weight_sum[int(gidx)] += float(gval)
                token_hit_count[int(gidx)] += 1

    items = []
    for gid, wsum in token_weight_sum.items():
        if gid == target_tid:
            continue
        gene_name = str(genes_vocab[gid])
        hits = token_hit_count.get(gid, 0)
        score = float(wsum) * (1.0 + math.log1p(hits))
        items.append((gene_name, score))

    items_sorted = sorted(items, key=lambda x: (0 if x[0] in stage1_nodes else 1, -x[1]))[:topk_genes]

    rows = []
    for gene_name, score in items_sorted:
        rows.append({
            "src": target_gene,
            "dst": gene_name,
            "weight": float(score),
        })
    return pd.DataFrame(rows)


def build_stage2_core_graph_from_stage1(G1: nx.DiGraph, stage2_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph(G1)
    for _, r in stage2_df.iterrows():
        s = str(r["src"])
        d = str(r["dst"])
        w = float(r["weight"])

        if d not in G:
            G.add_node(d, kind="gene")

        if not G.has_edge(s, d):
            G.add_edge(s, d, weight=max(0.8, min(5.0, np.log1p(w))), edge_type="representation_similarity")
        else:
            G[s][d]["stage2_boost"] = float(max(0.8, min(5.0, np.log1p(w))))
    return G


# ============================================================
# 5) STAGE3 / STAGE4 anchor diversification
# ============================================================

def guess_columns_stage3_cci(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    out = {"ligand": None, "receptor": None, "sender_cluster": None, "receiver_cluster": None, "score": None}
    for c in df.columns:
        lc = c.lower()
        if out["ligand"] is None and "ligand" in lc:
            out["ligand"] = c
        if out["receptor"] is None and "receptor" in lc:
            out["receptor"] = c
        if out["sender_cluster"] is None and ("sender_cluster" in lc or "source_cluster" in lc):
            out["sender_cluster"] = c
        if out["receiver_cluster"] is None and ("receiver_cluster" in lc or "target_cluster" in lc):
            out["receiver_cluster"] = c
    out["score"] = find_first_existing_col(df, ["edge_score", "expr_only", "confidence", "n_pairs", "score", "weight"])
    return out


def guess_columns_sig(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    out = {"receptor": None, "tf": None, "score": None}
    for c in df.columns:
        lc = c.lower()
        if out["receptor"] is None and "receptor" in lc:
            out["receptor"] = c
        if out["tf"] is None and lc == "tf":
            out["tf"] = c
    out["score"] = find_first_existing_col(df, ["corr", "score", "weight", "tf_score", "tf_corr"])
    return out


def get_target_adjacent_edges(G: nx.DiGraph, target_gene: str) -> List[Tuple[str, str]]:
    edges = []
    for u, v in G.edges():
        if u == target_gene or v == target_gene:
            edges.append((u, v))
    return edges


def choose_diverse_anchor_edge(
    core_graph: nx.DiGraph,
    target_gene: str,
    record_idx: int,
    used_counter: Counter,
    ligand: str,
    receptor: str,
    tf_candidates: List[str],
) -> Optional[Tuple[str, str]]:
    candidates = []
    for tf in tf_candidates:
        if core_graph.has_edge(tf, target_gene):
            candidates.append((tf, target_gene))

    for u, v in core_graph.edges():
        if u == target_gene and v not in {target_gene, receptor}:
            candidates.append((u, v))
        if v == target_gene and u not in {target_gene, ligand}:
            candidates.append((u, v))

    for u, v in core_graph.edges():
        if ligand in {u, v} or receptor in {u, v}:
            candidates.append((u, v))

    seen = set()
    uniq = []
    for e in candidates:
        if e not in seen:
            uniq.append(e)
            seen.add(e)

    if len(uniq) == 0:
        uniq = get_target_adjacent_edges(core_graph, target_gene)

    if len(uniq) == 0:
        edges = list(core_graph.edges())
        if len(edges) == 0:
            return None
        uniq = edges

    uniq = sorted(uniq, key=lambda e: (used_counter[e], e[0], e[1]))
    return uniq[record_idx % len(uniq)] if len(uniq) > 0 else None


def build_stage3_annotations(target_gene: str, core_graph: nx.DiGraph, cci_df: pd.DataFrame, sig_df: pd.DataFrame) -> List[dict]:
    ccols = guess_columns_stage3_cci(cci_df)
    scols = guess_columns_sig(sig_df)

    if ccols["ligand"] is None or ccols["receptor"] is None:
        return []

    if ccols["score"] is None:
        cci_df = cci_df.copy()
        cci_df["_score"] = 1.0
        ccols["score"] = "_score"

    cci_df[ccols["score"]] = pd.to_numeric(cci_df[ccols["score"]], errors="coerce").fillna(0.0)
    hit = cci_df[
        (cci_df[ccols["ligand"]].astype(str) == target_gene) |
        (cci_df[ccols["receptor"]].astype(str) == target_gene)
    ].copy()

    if len(hit) == 0:
        return []

    receptor_to_tfs = defaultdict(list)
    if scols["receptor"] is not None and scols["tf"] is not None:
        sig2 = sig_df.copy()
        if scols["score"] is None:
            sig2["_score"] = 1.0
            scols["score"] = "_score"
        sig2[scols["score"]] = pd.to_numeric(sig2[scols["score"]], errors="coerce").fillna(0.0)
        for _, r in sig2.iterrows():
            receptor_to_tfs[str(r[scols["receptor"]])].append((str(r[scols["tf"]]), float(r[scols["score"]])))

    hit = hit.sort_values(ccols["score"], ascending=False).head(TOPK_STAGE3_ANNOT)
    used_counter = Counter()
    ann = []

    for i, (_, r) in enumerate(hit.iterrows()):
        ligand = str(r[ccols["ligand"]])
        receptor = str(r[ccols["receptor"]])
        score = float(r[ccols["score"]])
        sender_cluster = str(r[ccols["sender_cluster"]]) if ccols["sender_cluster"] else ""
        receiver_cluster = str(r[ccols["receiver_cluster"]]) if ccols["receiver_cluster"] else ""

        tf_candidates = [x[0] for x in sorted(receptor_to_tfs.get(receptor, []), key=lambda z: z[1], reverse=True)]
        anchor = choose_diverse_anchor_edge(core_graph, target_gene, i, used_counter, ligand, receptor, tf_candidates)
        if anchor is None:
            continue
        used_counter[anchor] += 1

        ann.append({
            "anchor_u": anchor[0],
            "anchor_v": anchor[1],
            "lr": f"{ligand}-{receptor}",
            "receptor": receptor,
            "cluster": receiver_cluster if receiver_cluster else sender_cluster,
            "tf": tf_candidates[0] if len(tf_candidates) > 0 else None,
            "score": score,
        })

    return ann


def build_stage4_annotations(stage3_ann: List[dict], cluster_df: pd.DataFrame, edge_df: pd.DataFrame) -> List[dict]:
    if len(stage3_ann) == 0:
        return []

    cl = cluster_df.copy()
    ed = edge_df.copy()

    if "cluster_norm" in cl.columns:
        cl["cluster_norm"] = cl["cluster_norm"].astype(str).map(normalize_cluster_str)
    if "sender_cluster" in ed.columns:
        ed["sender_cluster"] = ed["sender_cluster"].astype(str).map(normalize_cluster_str)
    if "receiver_cluster" in ed.columns:
        ed["receiver_cluster"] = ed["receiver_cluster"].astype(str).map(normalize_cluster_str)

    score_col = find_first_existing_col(ed, ["edge_score", "expr_only", "confidence", "n_pairs", "score", "weight"])
    if score_col is None:
        ed["_score"] = 1.0
        score_col = "_score"

    ed[score_col] = pd.to_numeric(ed[score_col], errors="coerce").fillna(0.0)
    for c in ["t_sender", "t_receiver", "delta_t_receiver_minus_sender"]:
        if c in ed.columns:
            ed[c] = pd.to_numeric(ed[c], errors="coerce")

    cluster_map = cl.set_index("cluster_norm").to_dict(orient="index") if "cluster_norm" in cl.columns else {}

    out = []
    for a in stage3_ann:
        aa = dict(a)
        cluster = normalize_cluster_str(a.get("cluster", ""))
        sub = ed[
            (ed["sender_cluster"].astype(str) == cluster) |
            (ed["receiver_cluster"].astype(str) == cluster)
        ].copy()

        if len(sub) > 0:
            sub = sub.sort_values(score_col, ascending=False).head(1)
            rr = sub.iloc[0]
            aa["delta_t"] = float(rr["delta_t_receiver_minus_sender"]) if "delta_t_receiver_minus_sender" in rr and pd.notna(rr["delta_t_receiver_minus_sender"]) else np.nan
            aa["t_sender"] = float(rr["t_sender"]) if "t_sender" in rr and pd.notna(rr["t_sender"]) else np.nan
            aa["t_receiver"] = float(rr["t_receiver"]) if "t_receiver" in rr and pd.notna(rr["t_receiver"]) else np.nan
        else:
            aa["delta_t"] = np.nan
            aa["t_sender"] = np.nan
            aa["t_receiver"] = np.nan

        aa["receiver_tf_response"] = float(cluster_map.get(cluster, {}).get("receiver_tf_response", np.nan))
        out.append(aa)

    return out[:TOPK_STAGE4_ANNOT]


# ============================================================
# 5b) STAGE5 PHYSICS-INSPIRED NETWORK
# ============================================================

def collect_stage5_node_expression(
    G: nx.DiGraph,
    sample: Sample,
    gene_vocab_txt: Path,
    fallback_value: float = 0.05,
) -> pd.DataFrame:
    genes_vocab = read_vocab_txt(gene_vocab_txt)
    gene2tid = {g: i for i, g in enumerate(genes_vocab)}

    node_expr_raw = {}
    node_kind = {}

    for n, attrs in G.nodes(data=True):
        kind = str(attrs.get("kind", "gene"))
        node_kind[n] = kind

        if str(n) in gene2tid and gene2tid[str(n)] in sample.train2rna:
            ridx = sample.train2rna[gene2tid[str(n)]]
            col = get_counts_col_dense(sample, ridx)
            x = np.asarray(col, dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                node_expr_raw[n] = float(fallback_value)
            else:
                node_expr_raw[n] = float(np.mean(x))
        else:
            node_expr_raw[n] = float(fallback_value)

    node_expr_log = {k: math.log1p(max(0.0, v)) for k, v in node_expr_raw.items()}
    node_expr_norm = _safe_minmax_scale(node_expr_log, floor=0.02, ceil=1.0)

    rows = []
    for n in G.nodes():
        rows.append({
            "node": str(n),
            "kind": str(node_kind.get(n, "gene")),
            "expr_raw": float(node_expr_raw.get(n, fallback_value)),
            "expr_log1p": float(node_expr_log.get(n, math.log1p(fallback_value))),
            "expr_norm": float(node_expr_norm.get(n, 0.02)),
        })
    return pd.DataFrame(rows)


def summarize_stage3_edge_support(stage3_ann: List[dict]) -> Dict[Tuple[str, str], float]:
    agg = defaultdict(float)
    for a in stage3_ann:
        e = (str(a["anchor_u"]), str(a["anchor_v"]))
        s = float(a.get("score", 0.0)) if np.isfinite(a.get("score", np.nan)) else 0.0
        agg[e] += max(0.0, s)
    return _safe_log1p_scale(agg)


def summarize_stage4_edge_support(stage4_ann: List[dict]) -> Dict[Tuple[str, str], float]:
    raw = {}
    for a in stage4_ann:
        e = (str(a["anchor_u"]), str(a["anchor_v"]))
        dt = a.get("delta_t", np.nan)
        tfresp = a.get("receiver_tf_response", np.nan)

        dt_score = 0.0
        if np.isfinite(dt):
            dt_score = math.exp(-abs(float(dt) - 0.15) / 0.45)

        tf_score = 0.0
        if np.isfinite(tfresp):
            tf_score = max(0.0, float(tfresp))

        raw[e] = max(raw.get(e, 0.0), 0.65 * dt_score + 0.35 * tf_score)

    return _safe_log1p_scale(raw)


def build_stage5_field_edge_table(
    G2: nx.DiGraph,
    stage3_ann: List[dict],
    stage4_ann: List[dict],
    node_expr_df: pd.DataFrame,
    alpha1: float = STAGE5_A1,
    alpha2: float = STAGE5_A2,
    alpha3: float = STAGE5_A3,
    alpha4: float = STAGE5_A4,
) -> pd.DataFrame:
    node_expr_map = {
        str(r["node"]): float(r["expr_norm"])
        for _, r in node_expr_df.iterrows()
    }

    stage3_support = summarize_stage3_edge_support(stage3_ann)
    stage4_support = summarize_stage4_edge_support(stage4_ann)

    base_raw = {}
    rep_raw = {}

    for u, v, attrs in G2.edges(data=True):
        e = (str(u), str(v))
        et = str(attrs.get("edge_type", ""))

        if et == "prior_tf_gene":
            base_raw[e] = float(attrs.get("weight", 1.0))
            rep_raw[e] = float(attrs.get("stage2_boost", 0.0))
        elif et == "representation_similarity":
            base_raw[e] = 0.4
            rep_raw[e] = float(attrs.get("weight", 1.0))
        else:
            base_raw[e] = float(attrs.get("weight", 0.5))
            rep_raw[e] = float(attrs.get("stage2_boost", 0.0))

    base_norm = _safe_log1p_scale(base_raw)
    rep_norm = _safe_log1p_scale(rep_raw)

    rows = []
    for u, v, attrs in G2.edges(data=True):
        e = (str(u), str(v))
        p = float(base_norm.get(e, 0.0))
        r = float(rep_norm.get(e, 0.0))
        c = float(stage3_support.get(e, 0.0))
        t = float(stage4_support.get(e, 0.0))

        log_weff = (
            alpha1 * math.log1p(max(0.0, p)) +
            alpha2 * math.log1p(max(0.0, r)) +
            alpha3 * math.log1p(max(0.0, c)) +
            alpha4 * math.log1p(max(0.0, t))
        )
        w_eff = float(math.exp(log_weff) - 1.0)

        src_expr = float(node_expr_map.get(str(u), 0.05))
        dst_expr = float(node_expr_map.get(str(v), 0.05))

        rows.append({
            "src": str(u),
            "dst": str(v),
            "edge_type": str(attrs.get("edge_type", "")),
            "base_prior": p,
            "repr_support": r,
            "comm_support": c,
            "temporal_support": t,
            "w_eff": w_eff,
            "src_expr_norm": src_expr,
            "dst_expr_norm": dst_expr,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["w_eff_norm"] = pd.Series(_safe_log1p_scale({
        (str(r["src"]), str(r["dst"])): float(r["w_eff"])
        for _, r in df.iterrows()
    }).values(), index=df.index)

    return df


def build_stage5_node_state(
    node_expr_df: pd.DataFrame,
    alpha: float = STAGE5_ALPHA,
    beta: float = STAGE5_BETA,
    eps: float = STAGE5_EPS,
) -> pd.DataFrame:
    df = node_expr_df.copy()
    e = np.asarray(df["expr_norm"].values, dtype=float)
    field = np.power(np.clip(e, 0.0, 1.0), alpha)
    susc = np.power(np.clip(1.0 - e, 0.0, 1.0), beta) + float(eps)

    df["field_strength"] = field
    df["susceptibility"] = susc
    df["state_init"] = e.copy()
    return df


def apply_stage5_perturbation(
    node_state_df: pd.DataFrame,
    target_node: str,
    mode: str = "oe",
    strength: float = STAGE5_PERTURB_STRENGTH,
) -> pd.DataFrame:
    df = node_state_df.copy()
    if target_node not in set(df["node"].astype(str).tolist()):
        return df

    m = df["node"].astype(str) == str(target_node)
    x = float(df.loc[m, "state_init"].iloc[0])

    if mode == "ko":
        newx = x * max(0.0, 1.0 - float(strength))
    elif mode == "oe":
        newx = min(1.0, x + float(strength) * (1.0 - x))
    else:
        newx = x

    df.loc[m, "state_init"] = newx
    e = np.asarray(df["state_init"].values, dtype=float)
    df["field_strength"] = np.power(np.clip(e, 0.0, 1.0), STAGE5_ALPHA)
    df["susceptibility"] = np.power(np.clip(1.0 - e, 0.0, 1.0), STAGE5_BETA) + STAGE5_EPS
    return df


def compute_edge_flux_from_state(
    edge_df: pd.DataFrame,
    node_state_df: pd.DataFrame,
    state_col: str = "state_final",
    field_col: str = "field_strength_final",
    susc_col: str = "susceptibility_final",
) -> pd.DataFrame:
    ns = {}
    for _, r in node_state_df.iterrows():
        node = str(r["node"])

        a = r[state_col] if state_col in node_state_df.columns else np.nan
        f = r[field_col] if field_col in node_state_df.columns else np.nan
        s = r[susc_col] if susc_col in node_state_df.columns else np.nan

        ns[node] = {
            "A": float(a) if pd.notna(a) else 0.0,
            "F": float(f) if pd.notna(f) else 0.0,
            "S": float(s) if pd.notna(s) else 0.0,
        }

    rows = []
    for _, r in edge_df.iterrows():
        u = str(r["src"])
        v = str(r["dst"])
        we = float(r["w_eff_norm"])
        Fu = float(ns.get(u, {}).get("F", 0.0))
        Sv = float(ns.get(v, {}).get("S", 0.0))
        flux = float(we * Fu * Sv)

        rows.append({
            **r.to_dict(),
            "src_field": Fu,
            "dst_susc": Sv,
            "edge_flux": flux,
        })

    return pd.DataFrame(rows)


def run_stage5_field_propagation(
    edge_df: pd.DataFrame,
    node_state_df: pd.DataFrame,
    kappa: float = STAGE5_KAPPA,
    mu: float = STAGE5_MU,
    max_iter: int = STAGE5_MAX_ITER,
    tol: float = STAGE5_TOL,
    use_two_hop: bool = STAGE5_USE_TWO_HOP_PROPAGATION,
) -> pd.DataFrame:
    df = node_state_df.copy()

    node_list = df["node"].astype(str).tolist()
    node2idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    A0 = np.asarray(df["state_init"].values, dtype=float)
    A = A0.copy()

    W = np.zeros((n, n), dtype=float)
    for _, r in edge_df.iterrows():
        u = str(r["src"])
        v = str(r["dst"])
        if u in node2idx and v in node2idx:
            W[node2idx[u], node2idx[v]] = max(W[node2idx[u], node2idx[v]], float(r["w_eff_norm"]))

    for _ in range(int(max_iter)):
        F = np.power(np.clip(A, 0.0, 1.0), STAGE5_ALPHA)
        S = np.power(np.clip(1.0 - A, 0.0, 1.0), STAGE5_BETA) + STAGE5_EPS

        incoming = (W.T @ F) * S

        if use_two_hop:
            incoming2 = (W.T @ (W.T @ F)) * 0.35 * S
            incoming = incoming + incoming2

        target = _clip01(A0 + float(kappa) * incoming)
        A_new = _clip01((1.0 - float(mu)) * A + float(mu) * target)

        delta = float(np.max(np.abs(A_new - A)))
        A = A_new
        if delta < tol:
            break

    df["state_final"] = A
    df["field_strength_final"] = np.power(np.clip(A, 0.0, 1.0), STAGE5_ALPHA)
    df["susceptibility_final"] = np.power(np.clip(1.0 - A, 0.0, 1.0), STAGE5_BETA) + STAGE5_EPS
    df["delta_state"] = df["state_final"] - df["state_init"]
    return df


def compute_stage5_delta_flux(
    edge_df: pd.DataFrame,
    baseline_node_state_final: pd.DataFrame,
    perturb_node_state_final: pd.DataFrame,
) -> pd.DataFrame:
    base_ns = {
        str(r["node"]): {
            "F": float(r["field_strength_final"]),
            "S": float(r["susceptibility_final"]),
        }
        for _, r in baseline_node_state_final.iterrows()
    }
    pert_ns = {
        str(r["node"]): {
            "F": float(r["field_strength_final"]),
            "S": float(r["susceptibility_final"]),
        }
        for _, r in perturb_node_state_final.iterrows()
    }

    rows = []
    for _, r in edge_df.iterrows():
        u = str(r["src"])
        v = str(r["dst"])
        we = float(r["w_eff_norm"])

        fb = float(base_ns.get(u, {}).get("F", 0.0))
        sb = float(base_ns.get(v, {}).get("S", 0.0))
        fp = float(pert_ns.get(u, {}).get("F", 0.0))
        sp = float(pert_ns.get(v, {}).get("S", 0.0))

        flux_b = we * fb * sb
        flux_p = we * fp * sp
        dflux = flux_p - flux_b

        rows.append({
            **r.to_dict(),
            "flux_baseline": flux_b,
            "flux_perturbed": flux_p,
            "delta_flux": dflux,
            "abs_delta_flux": abs(dflux),
        })
    return pd.DataFrame(rows)


def build_stage5_graph_from_edge_table(
    base_graph: nx.DiGraph,
    edge_df: pd.DataFrame,
    node_state_final_df: pd.DataFrame,
) -> nx.DiGraph:
    G = nx.DiGraph()
    node_info = {
        str(r["node"]): r.to_dict()
        for _, r in node_state_final_df.iterrows()
    }

    for n, attrs in base_graph.nodes(data=True):
        dd = dict(attrs)
        if n in node_info:
            dd.update({
                "expr_norm": float(node_info[n].get("expr_norm", 0.05)),
                "field_strength_final": float(node_info[n].get("field_strength_final", node_info[n].get("field_strength", 0.0))),
                "susceptibility_final": float(node_info[n].get("susceptibility_final", node_info[n].get("susceptibility", 0.0))),
                "state_final": float(node_info[n].get("state_final", node_info[n].get("state_init", 0.0))),
                "delta_state": float(node_info[n].get("delta_state", 0.0)),
            })
        G.add_node(n, **dd)

    for _, r in edge_df.iterrows():
        u = str(r["src"])
        v = str(r["dst"])
        if u not in G:
            G.add_node(u, kind="gene")
        if v not in G:
            G.add_node(v, kind="gene")
        G.add_edge(
            u, v,
            edge_type="field_effective_coupling",
            w_eff=float(r["w_eff"]),
            w_eff_norm=float(r["w_eff_norm"]),
            base_prior=float(r["base_prior"]),
            repr_support=float(r["repr_support"]),
            comm_support=float(r["comm_support"]),
            temporal_support=float(r["temporal_support"]),
        )
    return G


# ============================================================
# 6) LAYOUT / DRAW
# ============================================================

def build_master_graph(graphs):
    U = nx.Graph()
    for G in graphs:
        U.add_nodes_from(G.nodes(data=True))
        U.add_edges_from(G.edges(data=True))
    return U


def build_incremental_layout(G1, G2, G3, G4, center_node="LAMB1", seed=42):
    U = build_master_graph([G1, G2, G3, G4])
    pos = nx.spring_layout(nx.Graph(G1), seed=seed, k=0.9, iterations=300)

    rng = np.random.default_rng(seed)
    for n in U.nodes():
        if n not in pos:
            pos[n] = rng.normal(scale=0.2, size=2)

    pos = nx.spring_layout(U, pos=pos, fixed=list(G1.nodes()), seed=seed, k=0.9, iterations=120)

    if center_node in pos:
        shift = np.array(pos[center_node])
        for n in pos:
            pos[n] = np.array(pos[n]) - shift
    return pos


def build_stage5_3d_layout(
    pos2d: Dict[str, np.ndarray],
    node_state_df: pd.DataFrame,
    z_mode: str = "field_strength_final",
) -> Dict[str, np.ndarray]:
    node_info = {
        str(r["node"]): r.to_dict()
        for _, r in node_state_df.iterrows()
    }
    z_raw = {}
    for n in pos2d:
        if n in node_info:
            z_raw[n] = float(node_info[n].get(z_mode, node_info[n].get("field_strength", 0.0)))
        else:
            z_raw[n] = 0.0

    z_scaled = _safe_minmax_scale(z_raw, floor=-0.65, ceil=0.95)

    pos3d = {}
    for n, p in pos2d.items():
        pos3d[n] = np.array([float(p[0]), float(p[1]), float(z_scaled.get(n, 0.0))], dtype=float)
    return pos3d


def build_stage5_3d_layout_from_delta(
    pos2d: Dict[str, np.ndarray],
    node_state_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    node_info = {
        str(r["node"]): r.to_dict()
        for _, r in node_state_df.iterrows()
    }
    z_raw = {}
    for n in pos2d:
        if n in node_info:
            z_raw[n] = float(node_info[n].get("delta_state", 0.0))
        else:
            z_raw[n] = 0.0

    z_scaled = _safe_minmax_scale(z_raw, floor=-0.85, ceil=0.85)

    pos3d = {}
    for n, p in pos2d.items():
        pos3d[n] = np.array([float(p[0]), float(p[1]), float(z_scaled.get(n, 0.0))], dtype=float)
    return pos3d


def node_style(kind: str):
    if kind == "target_gene":
        return {"size": 1900, "shape": "o"}
    if kind == "tf":
        return {"size": 1100, "shape": "s"}
    if kind == "gene":
        return {"size": 820, "shape": "o"}
    return {"size": 820, "shape": "o"}


def node_color(kind: str, attrs: dict):
    if kind == "target_gene":
        return "#d62728"
    if kind == "tf":
        return "#1f77b4"
    if kind == "gene":
        return "#9ecae1"
    return "#bdbdbd"


def base_edge_color(edge_type: str, attrs: dict):
    if edge_type == "prior_tf_gene":
        return "#7f7f7f" if int(attrs.get("is_after_256", 0)) == 0 else "#c7c7c7"
    if edge_type == "representation_similarity":
        return "#cfd8e6"
    return "#d7d7d7"


def build_stage2_highlight_edges(G2: nx.DiGraph):
    out = {}
    for u, v, attrs in G2.edges(data=True):
        if attrs.get("edge_type") == "representation_similarity" or "stage2_boost" in attrs:
            out[(u, v)] = {"color": "#4c78a8", "width": 2.4}
    return out


def build_stage3_highlight_edges(stage3_ann: List[dict]):
    out = {}
    for a in stage3_ann:
        k = (a["anchor_u"], a["anchor_v"])
        if k not in out:
            out[k] = {"color": "#2ca02c", "width": 2.9}
    return out


def build_stage4_highlight_edges(stage4_ann: List[dict]):
    out = {}
    for a in stage4_ann:
        k = (a["anchor_u"], a["anchor_v"])
        dt = a.get("delta_t", np.nan)
        color = "#2ca02c" if not np.isfinite(dt) else plt.get_cmap("plasma")(min(1, max(0, (float(dt)+0.2)/1.2)))
        out[k] = {"color": color, "width": 3.2}
    return out


def draw_full_core_network(ax, G, pos):
    for u, v, attrs in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=1.0,
            edge_color=[base_edge_color(str(attrs.get("edge_type", "")), attrs)],
            alpha=0.35,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.05",
        )

    kind_map = defaultdict(list)
    for n, attrs in G.nodes(data=True):
        kind_map[str(attrs.get("kind", "gene"))].append(n)

    for kind, nodes in kind_map.items():
        sty = node_style(kind)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=nodes,
            node_size=sty["size"],
            node_shape=sty["shape"],
            node_color=[node_color(kind, G.nodes[n]) for n in nodes],
            alpha=0.95 if kind == "target_gene" else 0.88,
            linewidths=1.2,
            edgecolors="black",
        )

    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: str(n) for n in G.nodes()}, font_size=9, font_weight="bold")


def draw_highlight_edges(ax, G, pos, highlight_edges):
    for (u, v), info in highlight_edges.items():
        if not G.has_edge(u, v):
            continue
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=info["width"],
            edge_color=[info["color"]],
            alpha=0.95,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            connectionstyle="arc3,rad=0.05",
        )


def unit_normal(p0: np.ndarray, p1: np.ndarray):
    v = p1 - p0
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return np.array([0.0, 1.0])
    v = v / nv
    return np.array([-v[1], v[0]])


def draw_short_pointer(ax, text_point, edge_mid, color="#666666"):
    arr = FancyArrowPatch(
        (text_point[0], text_point[1]), (edge_mid[0], edge_mid[1]),
        arrowstyle='-|>',
        mutation_scale=8,
        linewidth=1.0,
        color=color,
        alpha=0.8,
        zorder=9
    )
    ax.add_patch(arr)


def draw_edge_annotations(ax, pos, annotations: List[dict], stage: int):
    edge_count = defaultdict(int)

    for ann in annotations:
        u, v = ann["anchor_u"], ann["anchor_v"]
        if u not in pos or v not in pos:
            continue

        p0 = np.array(pos[u], dtype=float)
        p1 = np.array(pos[v], dtype=float)
        mid = 0.5 * (p0 + p1)
        nvec = unit_normal(p0, p1)

        ek = tuple(sorted((u, v)))
        rank = edge_count[ek]
        edge_count[ek] += 1

        base = mid + (0.12 + 0.09 * rank) * nvec
        edge_target = mid + 0.02 * nvec

        cluster = ann.get("cluster", "")
        lr = ann.get("lr", "")

        ax.scatter([base[0]], [base[1]], s=50, marker="D", c=["#ff7f0e"], edgecolors="black", linewidths=0.6, zorder=10)
        if cluster:
            cpt = base + np.array([0.05, 0.0])
            ax.scatter([cpt[0]], [cpt[1]], s=60, marker="h", c=["#9467bd"], edgecolors="black", linewidths=0.6, zorder=10)

        tpt = base + np.array([0.06, 0.03])
        ax.text(
            tpt[0], tpt[1], str(lr),
            fontsize=6.8, color="#2ca02c", ha="left", va="bottom", zorder=11,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.8)
        )
        draw_short_pointer(ax, tpt + np.array([-0.01, -0.01]), edge_target, color="#7a7a7a")

        if stage >= 4:
            dt = ann.get("delta_t", np.nan)
            tfresp = ann.get("receiver_tf_response", np.nan)
            if np.isfinite(dt):
                t2 = base - 0.08 * nvec + np.array([-0.03, -0.02])
                ax.text(
                    t2[0], t2[1], f"Δt={dt:.2f}",
                    fontsize=6.2, color="#444444", ha="right", va="center", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75)
                )
            if np.isfinite(tfresp):
                t3 = base - 0.08 * nvec + np.array([0.02, -0.05])
                ax.text(
                    t3[0], t3[1], f"TF={tfresp:.2f}",
                    fontsize=6.2, color="#444444", ha="left", va="center", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75)
                )


def draw_stage_figure(G, pos, title, subtitle, out_base, highlight_edges=None, annotations=None, stage=1):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_title(title, fontsize=17, pad=14)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)

    draw_full_core_network(ax, G, pos)
    if highlight_edges:
        draw_highlight_edges(ax, G, pos, highlight_edges)
    if annotations:
        draw_edge_annotations(ax, pos, annotations, stage=stage)

    ax.axis("off")
    plt.tight_layout()
    save_figure_both(fig, out_base)
    plt.close(fig)


# ============================================================
# 6b) STAGE5 3D DRAW
# ============================================================

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def stage5_node_size(attrs: dict) -> float:
    f = float(attrs.get("field_strength_final", attrs.get("field_strength", 0.0)))
    return 220 + 2200 * max(0.0, min(1.0, f))


def stage5_node_color_baseline(attrs: dict):
    e = float(attrs.get("expr_norm", 0.0))
    return plt.get_cmap("OrRd")(0.25 + 0.65 * e)


def stage5_node_color_perturb(attrs: dict):
    ds = float(attrs.get("delta_state", 0.0))
    if ds > 1e-6:
        c = min(1.0, abs(ds) / 0.35)
        return plt.get_cmap("Reds")(0.35 + 0.55 * c)
    elif ds < -1e-6:
        c = min(1.0, abs(ds) / 0.35)
        return plt.get_cmap("Blues")(0.35 + 0.55 * c)
    else:
        e = float(attrs.get("expr_norm", 0.0))
        return plt.get_cmap("Greys")(0.35 + 0.35 * e)


def draw_stage5_3d_baseline(
    ax,
    G: nx.DiGraph,
    pos3d: Dict[str, np.ndarray],
    edge_df: pd.DataFrame,
    title: str,
):
    ax.set_title(title, fontsize=16, pad=16)

    for u, v in G.edges():
        if u not in pos3d or v not in pos3d:
            continue
        p0 = pos3d[u]
        p1 = pos3d[v]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#d7d7d7",
            linewidth=0.8,
            alpha=0.22,
            zorder=1,
        )

    if len(edge_df) > 0:
        sub = edge_df.sort_values("w_eff_norm", ascending=False).head(STAGE5_TOPK_HIGHLIGHT_EDGES).copy()
        for _, r in sub.iterrows():
            u, v = str(r["src"]), str(r["dst"])
            if u not in pos3d or v not in pos3d:
                continue
            p0 = pos3d[u]
            p1 = pos3d[v]

            ww = 1.2 + 4.0 * float(r["w_eff_norm"])
            comm = float(r.get("comm_support", 0.0))
            temp = float(r.get("temporal_support", 0.0))
            cval = min(1.0, 0.55 * comm + 0.45 * temp)
            color = plt.get_cmap("viridis")(0.20 + 0.75 * cval)

            arrow = Arrow3D(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                mutation_scale=12, lw=ww, arrowstyle="-|>", color=color, alpha=0.90
            )
            ax.add_artist(arrow)

    for n, attrs in G.nodes(data=True):
        if n not in pos3d:
            continue
        p = pos3d[n]
        f = float(attrs.get("field_strength_final", attrs.get("field_strength", 0.0)))
        halo_size = 300 + 2600 * f
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            s=halo_size,
            c=["#fdae6b"],
            alpha=0.08 + 0.15 * f,
            linewidths=0,
            zorder=2,
        )

    for n, attrs in G.nodes(data=True):
        if n not in pos3d:
            continue
        p = pos3d[n]
        size = stage5_node_size(attrs)
        color = stage5_node_color_baseline(attrs)
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            s=size,
            c=[color],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.94,
            zorder=4,
        )
        ax.text(
            p[0], p[1], p[2] + 0.03, str(n),
            fontsize=8, ha="center", va="bottom", zorder=5
        )

    ax.set_axis_off()
    ax.view_init(elev=20, azim=-58)


def draw_stage5_3d_perturbation(
    ax,
    G: nx.DiGraph,
    pos3d: Dict[str, np.ndarray],
    delta_flux_df: pd.DataFrame,
    title: str,
):
    ax.set_title(title, fontsize=16, pad=16)

    for u, v in G.edges():
        if u not in pos3d or v not in pos3d:
            continue
        p0 = pos3d[u]
        p1 = pos3d[v]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="#d7d7d7",
            linewidth=0.8,
            alpha=0.18,
            zorder=1,
        )

    if len(delta_flux_df) > 0:
        sub = delta_flux_df.sort_values("abs_delta_flux", ascending=False).head(STAGE5_TOPK_HIGHLIGHT_EDGES).copy()
        max_abs = float(sub["abs_delta_flux"].max()) if len(sub) > 0 else 1.0
        max_abs = max(max_abs, 1e-8)

        for _, r in sub.iterrows():
            u, v = str(r["src"]), str(r["dst"])
            if u not in pos3d or v not in pos3d:
                continue
            p0 = pos3d[u]
            p1 = pos3d[v]
            d = float(r["delta_flux"])
            mag = abs(d) / max_abs
            ww = 1.4 + 4.8 * mag
            color = plt.get_cmap("Reds")(0.35 + 0.55 * mag) if d >= 0 else plt.get_cmap("Blues")(0.35 + 0.55 * mag)

            arrow = Arrow3D(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                mutation_scale=12, lw=ww, arrowstyle="-|>", color=color, alpha=0.94
            )
            ax.add_artist(arrow)

    rank_nodes = sorted(
        list(G.nodes(data=True)),
        key=lambda kv: abs(float(kv[1].get("delta_state", 0.0))),
        reverse=True
    )
    top_show = set([n for n, _ in rank_nodes[:STAGE5_TOPK_HIGHLIGHT_NODES]])

    for n, attrs in G.nodes(data=True):
        if n not in pos3d:
            continue
        p = pos3d[n]
        size = stage5_node_size(attrs)
        color = stage5_node_color_perturb(attrs)
        ax.scatter(
            [p[0]], [p[1]], [p[2]],
            s=size,
            c=[color],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.95,
            zorder=4,
        )

        if n in top_show:
            ds = float(attrs.get("delta_state", 0.0))
            ax.text(
                p[0], p[1], p[2] + 0.05,
                f"{n}\n{ds:+.2f}",
                fontsize=7.4, ha="center", va="bottom", zorder=5
            )
        else:
            ax.text(
                p[0], p[1], p[2] + 0.03, str(n),
                fontsize=7.5, ha="center", va="bottom", zorder=5
            )

    ax.set_axis_off()
    ax.view_init(elev=24, azim=-52)


def draw_stage5_3d_figures(
    G_baseline: nx.DiGraph,
    G_perturbed: nx.DiGraph,
    pos3d_baseline: Dict[str, np.ndarray],
    pos3d_perturb: Dict[str, np.ndarray],
    edge_df_baseline: pd.DataFrame,
    delta_flux_df: pd.DataFrame,
    out_base_baseline: Path,
    out_base_compare: Path,
    target_gene: str,
    mode: str,
):
    fig1 = plt.figure(figsize=(14, 11), dpi=DPI)
    ax1 = fig1.add_subplot(111, projection="3d")
    draw_stage5_3d_baseline(
        ax1, G_baseline, pos3d_baseline, edge_df_baseline,
        title=f"Stage 5 — Physics-inspired 3D Field Network around {target_gene}",
    )
    plt.tight_layout()
    save_figure_both(fig1, out_base_baseline)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(22, 10), dpi=DPI)
    ax2 = fig2.add_subplot(121, projection="3d")
    ax3 = fig2.add_subplot(122, projection="3d")

    draw_stage5_3d_baseline(
        ax2, G_baseline, pos3d_baseline, edge_df_baseline,
        title="Baseline field network (3D)",
    )
    draw_stage5_3d_perturbation(
        ax3, G_perturbed, pos3d_perturb, delta_flux_df,
        title=f"Perturbation response (3D, {mode.upper()} of {target_gene})",
    )

    plt.tight_layout()
    save_figure_both(fig2, out_base_compare)
    plt.close(fig2)


# ============================================================
# 7) MAIN
# ============================================================

def main():
    genes_vocab = read_vocab_txt(GLOBAL_GENE_VOCAB)
    sample = load_single_sample(DATASET_NAME, SAMPLE_ID, genes_vocab)

    # ---------------------------
    # Stage1
    # ---------------------------
    prior_df = load_tf_gene_prior_edges(
        priors_dir=PRIORS_DIR,
        tf_vocab_txt=GLOBAL_TF_VOCAB,
        gene_vocab_txt=GLOBAL_GENE_VOCAB,
        prefer_full_with_decay=PREFER_FULL_WITH_DECAY,
        decay_after_256=DECAY_AFTER_256,
        decay_mode=DECAY_MODE,
    )
    prior_df.to_csv(OUT_DIR / f"{safe_name(TARGET_GENE)}_stage1_prior_edges_auto.csv", index=False)
    G1 = build_stage1_core_graph(TARGET_GENE, prior_df)

    # ---------------------------
    # Stage2
    # ---------------------------
    stage2_table = auto_make_stage2_edge_table(
        target_gene=TARGET_GENE,
        sample=sample,
        gene_vocab_txt=GLOBAL_GENE_VOCAB,
        stage1_nodes=set(G1.nodes()),
        topk_genes=TOPK_STAGE2_GENES,
        target_cell_q=STAGE2_TARGET_CELL_Q,
        min_target_cells=STAGE2_MIN_TARGET_CELLS,
        projected_row_topk=STAGE2_PROJECTED_ROW_TOPK,
    )
    stage2_table.to_csv(OUT_DIR / f"{safe_name(TARGET_GENE)}_stage2_representation_edges_auto.csv", index=False)
    G2 = build_stage2_core_graph_from_stage1(G1, stage2_table)
    stage2_highlight = build_stage2_highlight_edges(G2)

    # ---------------------------
    # Stage3
    # ---------------------------
    cci_df = pd.read_csv(CCI_TFKEPT_CSV)
    sig_df = pd.read_csv(RECEPTOR_TF_SIG_CSV)
    G3 = nx.DiGraph(G2)
    stage3_ann = build_stage3_annotations(TARGET_GENE, G3, cci_df, sig_df)
    pd.DataFrame(stage3_ann).to_csv(OUT_DIR / f"{safe_name(TARGET_GENE)}_stage3_edge_annotations_auto.csv", index=False)
    stage3_highlight = build_stage3_highlight_edges(stage3_ann)

    # ---------------------------
    # Stage4
    # ---------------------------
    cl_df = pd.read_csv(CLUSTER_PT_CSV)
    ed_df = pd.read_csv(EDGE_PT_CSV)
    G4 = nx.DiGraph(G3)
    stage4_ann = build_stage4_annotations(stage3_ann, cl_df, ed_df)
    pd.DataFrame(stage4_ann).to_csv(OUT_DIR / f"{safe_name(TARGET_GENE)}_stage4_edge_annotations_auto.csv", index=False)
    stage4_highlight = build_stage4_highlight_edges(stage4_ann)

    pos = build_incremental_layout(G1, G2, G3, G4, center_node=TARGET_GENE, seed=42)

    # ---------------------------
    # Draw Stage1~4
    # ---------------------------
    draw_stage_figure(
        G1, pos,
        title=f"Stage 1 — Prior Injection around {TARGET_GENE}",
        subtitle="Initial RNA-TF prior skeleton",
        out_base=OUT_DIR / f"{safe_name(TARGET_GENE)}_stage1_prior_injection",
        highlight_edges=None,
        annotations=None,
        stage=1,
    )

    draw_stage_figure(
        G2, pos,
        title=f"Stage 2 — Representation Reshaping around {TARGET_GENE}",
        subtitle="Representation-enhanced edges are thickened on top of the same backbone",
        out_base=OUT_DIR / f"{safe_name(TARGET_GENE)}_stage2_representation_reshape",
        highlight_edges=stage2_highlight,
        annotations=None,
        stage=2,
    )

    draw_stage_figure(
        G3, pos,
        title=f"Stage 3 — Communication Filtering around {TARGET_GENE}",
        subtitle="Communication-supported edges are highlighted and annotated beside the edge",
        out_base=OUT_DIR / f"{safe_name(TARGET_GENE)}_stage3_communication_filter",
        highlight_edges=stage3_highlight,
        annotations=stage3_ann,
        stage=3,
    )

    draw_stage_figure(
        G4, pos,
        title=f"Stage 4 — Temporal Enhancement around {TARGET_GENE}",
        subtitle="Temporal support is attached to the already highlighted edges",
        out_base=OUT_DIR / f"{safe_name(TARGET_GENE)}_stage4_temporal_enhancement",
        highlight_edges=stage4_highlight,
        annotations=stage4_ann,
        stage=4,
    )

    # ========================================================
    # Stage 5 — Physics-inspired integration
    # ========================================================
    node_expr_df = collect_stage5_node_expression(
        G=G4,
        sample=sample,
        gene_vocab_txt=GLOBAL_GENE_VOCAB,
        fallback_value=0.05,
    )
    node_expr_df.to_csv(OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_node_expression.csv", index=False)

    stage5_edge_df = build_stage5_field_edge_table(
        G2=G4,
        stage3_ann=stage3_ann,
        stage4_ann=stage4_ann,
        node_expr_df=node_expr_df,
        alpha1=STAGE5_A1,
        alpha2=STAGE5_A2,
        alpha3=STAGE5_A3,
        alpha4=STAGE5_A4,
    )
    stage5_edge_df.to_csv(OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_field_edges.csv", index=False)

    baseline_node_state = build_stage5_node_state(
        node_expr_df=node_expr_df,
        alpha=STAGE5_ALPHA,
        beta=STAGE5_BETA,
        eps=STAGE5_EPS,
    )
    baseline_node_state_final = run_stage5_field_propagation(
        edge_df=stage5_edge_df,
        node_state_df=baseline_node_state,
        kappa=STAGE5_KAPPA,
        mu=STAGE5_MU,
        max_iter=STAGE5_MAX_ITER,
        tol=STAGE5_TOL,
        use_two_hop=STAGE5_USE_TWO_HOP_PROPAGATION,
    )
    baseline_node_state_final.to_csv(
        OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_baseline_node_state.csv",
        index=False
    )

    perturbed_node_state = apply_stage5_perturbation(
        node_state_df=baseline_node_state,
        target_node=TARGET_GENE,
        mode=STAGE5_MODE,
        strength=STAGE5_PERTURB_STRENGTH,
    )
    perturbed_node_state_final = run_stage5_field_propagation(
        edge_df=stage5_edge_df,
        node_state_df=perturbed_node_state,
        kappa=STAGE5_KAPPA,
        mu=STAGE5_MU,
        max_iter=STAGE5_MAX_ITER,
        tol=STAGE5_TOL,
        use_two_hop=STAGE5_USE_TWO_HOP_PROPAGATION,
    )
    perturbed_node_state_final.to_csv(
        OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_{STAGE5_MODE}_node_state.csv",
        index=False
    )

    stage5_baseline_flux_df = compute_edge_flux_from_state(
        edge_df=stage5_edge_df,
        node_state_df=baseline_node_state_final,
        state_col="state_final",
        field_col="field_strength_final",
        susc_col="susceptibility_final",
    )
    stage5_baseline_flux_df.to_csv(
        OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_baseline_edge_flux.csv",
        index=False
    )

    stage5_delta_flux_df = compute_stage5_delta_flux(
        edge_df=stage5_edge_df,
        baseline_node_state_final=baseline_node_state_final,
        perturb_node_state_final=perturbed_node_state_final,
    )
    stage5_delta_flux_df.to_csv(
        OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_{STAGE5_MODE}_delta_flux.csv",
        index=False
    )

    G5_baseline = build_stage5_graph_from_edge_table(
        base_graph=G4,
        edge_df=stage5_edge_df,
        node_state_final_df=baseline_node_state_final,
    )

    G5_perturbed = nx.DiGraph(G5_baseline)
    pert_map = {
        str(r["node"]): r.to_dict()
        for _, r in perturbed_node_state_final.iterrows()
    }
    base_map = {
        str(r["node"]): r.to_dict()
        for _, r in baseline_node_state_final.iterrows()
    }

    for n in G5_perturbed.nodes():
        if n in pert_map:
            G5_perturbed.nodes[n]["expr_norm"] = float(pert_map[n].get("expr_norm", G5_perturbed.nodes[n].get("expr_norm", 0.05)))
            G5_perturbed.nodes[n]["field_strength_final"] = float(pert_map[n].get("field_strength_final", 0.0))
            G5_perturbed.nodes[n]["susceptibility_final"] = float(pert_map[n].get("susceptibility_final", 0.0))
            G5_perturbed.nodes[n]["state_final"] = float(pert_map[n].get("state_final", 0.0))
            G5_perturbed.nodes[n]["delta_state"] = float(
                pert_map[n].get("state_final", 0.0) - base_map.get(n, {}).get("state_final", 0.0)
            )

    pos3d_baseline = build_stage5_3d_layout(
        pos2d=pos,
        node_state_df=baseline_node_state_final,
        z_mode="field_strength_final",
    )
    pos3d_perturb = build_stage5_3d_layout_from_delta(
        pos2d=pos,
        node_state_df=perturbed_node_state_final.assign(
            delta_state=perturbed_node_state_final["state_final"].values - baseline_node_state_final["state_final"].values
        ),
    )

    draw_stage5_3d_figures(
        G_baseline=G5_baseline,
        G_perturbed=G5_perturbed,
        pos3d_baseline=pos3d_baseline,
        pos3d_perturb=pos3d_perturb,
        edge_df_baseline=stage5_edge_df,
        delta_flux_df=stage5_delta_flux_df,
        out_base_baseline=OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_physics_integration_3d",
        out_base_compare=OUT_DIR / f"{safe_name(TARGET_GENE)}_stage5_{STAGE5_MODE}_response_compare_3d",
        target_gene=TARGET_GENE,
        mode=STAGE5_MODE,
    )

    print("[DONE] Outputs saved to:", OUT_DIR.resolve())
    print("[DONE] Stage1~4 + Stage5(3D) PNG + SVG generated.")


if __name__ == "__main__":
    main()