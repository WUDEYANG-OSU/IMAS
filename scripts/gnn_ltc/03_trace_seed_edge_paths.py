from __future__ import annotations

# ============================================================
# seed_edge_clusterpath_summary.py
# ------------------------------------------------------------
# 目标：
#   - 以一条 seed edge 为起点
#   - 在全部 sender->receiver edges 中寻找与其轨迹/状态最相似的 topK edges
#   - 输出 edge-level cluster-path detail / summary / flow plot
#
# 输入：
#   1) cluster_pseudotime.csv
#   2) edge_pseudotime_summary.csv
#
# 输出（在 OUT_DIR）：
#   1) edge_enriched_table.csv
#   2) seed_<SEED_EDGE>_topK_edges.csv
#   3) edge_sender_presence_curves.csv
#   4) edge_receiver_presence_curves.csv
#   5) seed_<SEED_EDGE>_topDETAIL_clusterpath_detail.csv
#   6) seed_<SEED_EDGE>_topK_clusterpath_summary.csv
#   7) flow_seededge_<SEED_EDGE_LABEL>.png/svg for topDETAIL
#
# 注：
#   - 不修改旧文件
#   - 不使用 pathway / RNA expression
#   - 完全基于 cluster communication pseudotime 输出
# ============================================================

import re
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


# ============================================================
# CONFIG
# ============================================================
DATASET_NAME = "CRC-Tissue"
SAMPLE_ID = "CRC_RNA_TO009"

BASE = get_project_root()
RUN_STAGEB_DIR = (
    BASE
    / "runs"
    / "frontier_transfer_hash_stageB_selectedPeakResidual_clusterCorrection"
    / DATASET_NAME
)
CCI_DIR = RUN_STAGEB_DIR / f"{SAMPLE_ID}_cci_newLeiden"
PT_DIR = CCI_DIR / "communication_pseudotime_clusterGNN_LTC"

CLUSTER_PT_CSV = PT_DIR / "cluster_pseudotime.csv"
EDGE_PT_CSV = PT_DIR / "edge_pseudotime_summary.csv"

OUT_DIR = PT_DIR / "seed_edge_clusterpath"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Seed edge
# 可直接写成 "3->8" 或 ("3","8")
# 如果不存在，会自动回退到排序第一的强边
# -------------------------
SEED_SENDER = "3"
SEED_RECEIVER = "8"

TOPK = 20
TOPK_DETAIL = 10

# -------------------------
# Similarity / ranking
# -------------------------
USE_COMPOSITE_SCORE = True

LAMBDA_TSENDER = 5.0
LAMBDA_TRECEIVER = 5.0
LAMBDA_DT = 4.0
LAMBDA_TF = 1.5
LAMBDA_STRENGTH = 1.0

# 主 profile 的相关性用哪些维度
PROFILE_COLS = [
    "t_sender",
    "t_receiver",
    "delta_t_receiver_minus_sender",
    "edge_strength_log1p",
    "receiver_cluster_tf_response",
    "sender_cluster_sender_strength",
    "receiver_cluster_receiver_strength",
    "receiver_cluster_net_flow",
    "receiver_cluster_io_ratio",
]

# 过滤边
FILTER_EDGES = True
MIN_EDGE_SCORE = 0.0
MIN_DELTA_T = -0.20
MAX_DELTA_T = 1.00

# strict cluster-path gates
MIN_EDGES_PER_CLUSTER = 2
PASS_CORR_THR = 0.20
FIT_CORR_THR = 0.20
MU_CORR_THR = 0.15
MIN_SCORE_CLUSTER = 0.05

# flow plot
FLOW_ALPHA = 0.18
FLOW_LW_MIN, FLOW_LW_MAX = 1.0, 16.0
FLOW_GAMMA = 0.65
FLOW_COLOR_PASS = "#2b6cb0"
FLOW_COLOR_FAIL = "#bdbdbd"

# misc
DPI = 320
SEED = 42
np.random.seed(SEED)

# outputs
def seed_edge_label(sender: str, receiver: str) -> str:
    return f"C{sender}_to_C{receiver}"

OUT_EDGE_ENRICHED = OUT_DIR / "edge_enriched_table.csv"
OUT_TOP = OUT_DIR / f"seed_{seed_edge_label(SEED_SENDER, SEED_RECEIVER)}_top{TOPK}_edges.csv"
OUT_SENDER_PCURVE = OUT_DIR / "edge_sender_presence_curves.csv"
OUT_RECEIVER_PCURVE = OUT_DIR / "edge_receiver_presence_curves.csv"
OUT_DETAIL = OUT_DIR / f"seed_{seed_edge_label(SEED_SENDER, SEED_RECEIVER)}_top{TOPK_DETAIL}_clusterpath_detail.csv"
OUT_SUMMARY = OUT_DIR / f"seed_{seed_edge_label(SEED_SENDER, SEED_RECEIVER)}_top{TOPK}_clusterpath_summary.csv"


# ============================================================
# Helpers
# ============================================================
def log(*args):
    print(*args, flush=True)


def normalize_cluster_str(x: str) -> str:
    s = str(x).strip()
    if s.startswith("C") and s[1:].isdigit():
        return s[1:]
    return s


def safe_name(x: str, n: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(x))[:n]


def pearson_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m]
    b = b[m]
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-9 or sb < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def zscore_1d(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return x.astype(float)
    m = np.nanmean(x)
    s = np.nanstd(x) + 1e-12
    return (x - m) / s


def normalize01(w):
    w = np.asarray(w, float)
    m = np.isfinite(w)
    if m.sum() == 0:
        return np.zeros_like(w)
    a = float(np.min(w[m]))
    b = float(np.max(w[m]))
    if b <= a + 1e-12:
        y = np.zeros_like(w)
        y[m] = 0.5
        return y
    y = (w - a) / (b - a)
    y[~m] = 0.0
    return np.clip(y, 0.0, 1.0)


def weight_to_lw(w01):
    w01 = float(np.clip(w01, 0, 1)) ** FLOW_GAMMA
    return FLOW_LW_MIN + (FLOW_LW_MAX - FLOW_LW_MIN) * w01


def draw_bezier(ax, x0, y0, x1, y1, lw, color, alpha, z=1):
    dx = (x1 - x0)
    c1x = x0 + 0.35 * dx
    c2x = x1 - 0.35 * dx
    verts = [(x0, y0), (c1x, y0), (c2x, y1), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor="none",
            edgecolor=color,
            lw=lw,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
            zorder=z,
        )
    )


def stage_of_t(x):
    if not np.isfinite(x):
        return "NA"
    if x < 0.33:
        return "early"
    elif x < 0.66:
        return "mid"
    return "late"


def edge_name(sender: str, receiver: str) -> str:
    return f"C{sender}→C{receiver}"


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


# ============================================================
# 1) Load and enrich edges
# ============================================================
def load_inputs():
    assert CLUSTER_PT_CSV.exists(), f"Missing: {CLUSTER_PT_CSV}"
    assert EDGE_PT_CSV.exists(), f"Missing: {EDGE_PT_CSV}"

    cl = pd.read_csv(CLUSTER_PT_CSV)
    ed = pd.read_csv(EDGE_PT_CSV)

    cl["cluster_norm"] = cl["cluster_norm"].astype(str).map(normalize_cluster_str)
    ed["sender_cluster"] = ed["sender_cluster"].astype(str).map(normalize_cluster_str)
    ed["receiver_cluster"] = ed["receiver_cluster"].astype(str).map(normalize_cluster_str)

    for c in [
        "t_hat_norm_mean",
        "sender_strength",
        "receiver_strength",
        "receiver_tf_response",
        "net_flow",
        "io_ratio",
        "sigma_t",
        "tau_mean",
    ]:
        if c in cl.columns:
            cl[c] = pd.to_numeric(cl[c], errors="coerce")

    for c in [
        "edge_score",
        "expr_only",
        "confidence",
        "n_pairs",
        "t_sender",
        "t_receiver",
        "delta_t_receiver_minus_sender",
    ]:
        if c in ed.columns:
            ed[c] = pd.to_numeric(ed[c], errors="coerce")

    cl = cl[np.isfinite(cl["t_hat_norm_mean"].values)].copy().reset_index(drop=True)
    ed = ed[
        np.isfinite(ed["t_sender"].values)
        & np.isfinite(ed["t_receiver"].values)
        & np.isfinite(ed["delta_t_receiver_minus_sender"].values)
    ].copy().reset_index(drop=True)

    return cl, ed


def enrich_edges(cluster_df: pd.DataFrame, edge_df: pd.DataFrame):
    cl = cluster_df.copy().reset_index(drop=True)
    ed = edge_df.copy().reset_index(drop=True)

    merge_cols = [
        "cluster_norm",
        "t_hat_norm_mean",
        "sender_strength",
        "receiver_strength",
        "receiver_tf_response",
        "net_flow",
        "io_ratio",
        "sigma_t",
        "tau_mean",
    ]
    cl_small = cl[merge_cols].copy()

    ed = ed.merge(
        cl_small.rename(columns={
            "cluster_norm": "sender_cluster",
            "t_hat_norm_mean": "sender_cluster_t_global",
            "sender_strength": "sender_cluster_sender_strength",
            "receiver_strength": "sender_cluster_receiver_strength",
            "receiver_tf_response": "sender_cluster_tf_response",
            "net_flow": "sender_cluster_net_flow",
            "io_ratio": "sender_cluster_io_ratio",
            "sigma_t": "sender_cluster_sigma_t",
            "tau_mean": "sender_cluster_tau_mean",
        }),
        on="sender_cluster",
        how="left",
    )

    ed = ed.merge(
        cl_small.rename(columns={
            "cluster_norm": "receiver_cluster",
            "t_hat_norm_mean": "receiver_cluster_t_global",
            "sender_strength": "receiver_cluster_sender_strength",
            "receiver_strength": "receiver_cluster_receiver_strength",
            "receiver_tf_response": "receiver_cluster_tf_response",
            "net_flow": "receiver_cluster_net_flow",
            "io_ratio": "receiver_cluster_io_ratio",
            "sigma_t": "receiver_cluster_sigma_t",
            "tau_mean": "receiver_cluster_tau_mean",
        }),
        on="receiver_cluster",
        how="left",
    )

    # pick main edge score
    score_col = None
    for c in ["edge_score", "expr_only", "confidence", "n_pairs"]:
        if c in ed.columns:
            score_col = c
            break
    if score_col is None:
        raise RuntimeError("No usable score column found in edge_pseudotime_summary.csv")

    ed["edge_score_main"] = pd.to_numeric(ed[score_col], errors="coerce").fillna(0.0)
    ed["edge_strength_log1p"] = np.log1p(np.maximum(ed["edge_score_main"].values.astype(float), 0.0))
    ed["edge_label"] = [
        edge_name(s, r)
        for s, r in zip(ed["sender_cluster"].astype(str), ed["receiver_cluster"].astype(str))
    ]
    ed["seed_key"] = [
        f"{s}->{r}"
        for s, r in zip(ed["sender_cluster"].astype(str), ed["receiver_cluster"].astype(str))
    ]
    ed["sender_stage"] = ed["t_sender"].map(stage_of_t)
    ed["receiver_stage"] = ed["t_receiver"].map(stage_of_t)

    ed["timing_span"] = np.where(
        ed["delta_t_receiver_minus_sender"].values < 0.10,
        "tight",
        np.where(ed["delta_t_receiver_minus_sender"].values < 0.30, "moderate", "long"),
    )

    if FILTER_EDGES:
        ed = ed[
            (ed["edge_score_main"] >= MIN_EDGE_SCORE)
            & (ed["delta_t_receiver_minus_sender"] >= MIN_DELTA_T)
            & (ed["delta_t_receiver_minus_sender"] <= MAX_DELTA_T)
        ].copy().reset_index(drop=True)

    ed.to_csv(OUT_EDGE_ENRICHED, index=False)
    log("[SAVE]", OUT_EDGE_ENRICHED)
    return ed


# ============================================================
# 2) Seed matching
# ============================================================
def choose_seed_edge(ed: pd.DataFrame):
    seed_key = f"{normalize_cluster_str(SEED_SENDER)}->{normalize_cluster_str(SEED_RECEIVER)}"
    hit = ed[ed["seed_key"] == seed_key].copy()

    if len(hit) > 0:
        row = hit.sort_values(["edge_score_main", "delta_t_receiver_minus_sender"], ascending=[False, False]).iloc[0]
        log("[SEED] using requested seed:", row["edge_label"])
        return row

    # fallback
    row = ed.sort_values(
        ["edge_score_main", "delta_t_receiver_minus_sender", "receiver_cluster_tf_response"],
        ascending=[False, False, False],
    ).iloc[0]
    log("[SEED] requested seed not found; fallback to:", row["edge_label"])
    return row


def build_profile_vector(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    vals = []
    for c in cols:
        vals.append(pd.to_numeric(df[c], errors="coerce").values.astype(float))
    x = np.vstack(vals).T.squeeze()
    return np.asarray(x, float)


def rank_edges_against_seed(ed: pd.DataFrame, seed_row: pd.Series):
    seed_vec = np.asarray([float(seed_row[c]) if pd.notnull(seed_row[c]) else np.nan for c in PROFILE_COLS], float)
    seed_vec_z = zscore_1d(seed_vec)

    rows = []
    for _, r in ed.iterrows():
        if str(r["seed_key"]) == str(seed_row["seed_key"]):
            continue

        vec = np.asarray([float(r[c]) if pd.notnull(r[c]) else np.nan for c in PROFILE_COLS], float)
        corr = pearson_corr(seed_vec_z, zscore_1d(vec))
        if not np.isfinite(corr):
            continue

        if USE_COMPOSITE_SCORE:
            d_ts = abs(float(r["t_sender"]) - float(seed_row["t_sender"]))
            d_tr = abs(float(r["t_receiver"]) - float(seed_row["t_receiver"]))
            d_dt = abs(float(r["delta_t_receiver_minus_sender"]) - float(seed_row["delta_t_receiver_minus_sender"]))
            d_tf = abs(
                float(pd.to_numeric(r["receiver_cluster_tf_response"], errors="coerce"))
                - float(pd.to_numeric(seed_row["receiver_cluster_tf_response"], errors="coerce"))
            )
            d_strength = abs(float(r["edge_strength_log1p"]) - float(seed_row["edge_strength_log1p"]))

            score = float(
                corr
                * math.exp(-LAMBDA_TSENDER * d_ts)
                * math.exp(-LAMBDA_TRECEIVER * d_tr)
                * math.exp(-LAMBDA_DT * d_dt)
                * math.exp(-LAMBDA_TF * d_tf)
                * math.exp(-LAMBDA_STRENGTH * d_strength)
            )
        else:
            score = float(corr)

        rows.append({
            "seed_edge": str(seed_row["edge_label"]),
            "edge_label": str(r["edge_label"]),
            "seed_key": str(r["seed_key"]),
            "score": score,
            "corr": float(corr),
            "sender_cluster": str(r["sender_cluster"]),
            "receiver_cluster": str(r["receiver_cluster"]),
            "t_sender": float(r["t_sender"]),
            "t_receiver": float(r["t_receiver"]),
            "delta_t_receiver_minus_sender": float(r["delta_t_receiver_minus_sender"]),
            "edge_score_main": float(r["edge_score_main"]),
            "receiver_cluster_tf_response": float(pd.to_numeric(r["receiver_cluster_tf_response"], errors="coerce")),
            "sender_cluster_sender_strength": float(pd.to_numeric(r["sender_cluster_sender_strength"], errors="coerce")),
            "receiver_cluster_receiver_strength": float(pd.to_numeric(r["receiver_cluster_receiver_strength"], errors="coerce")),
            "receiver_cluster_net_flow": float(pd.to_numeric(r["receiver_cluster_net_flow"], errors="coerce")),
            "receiver_cluster_io_ratio": float(pd.to_numeric(r["receiver_cluster_io_ratio"], errors="coerce")),
        })

    df_rank = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df_top = df_rank.head(TOPK).copy()
    df_top.to_csv(OUT_TOP, index=False)
    log("[SAVE]", OUT_TOP)
    return df_rank, df_top


# ============================================================
# 3) Presence curves over clusters
# ============================================================
def build_presence_curves(cluster_df: pd.DataFrame, edge_df: pd.DataFrame):
    cl = cluster_df.copy().sort_values("t_hat_norm_mean").reset_index(drop=True)
    clusters = cl["cluster_norm"].astype(str).tolist()

    sender_counts = []
    receiver_counts = []
    total_edges = float(len(edge_df)) if len(edge_df) > 0 else 1.0

    for c in clusters:
        sender_counts.append(float(np.mean(edge_df["sender_cluster"].astype(str).values == str(c))))
        receiver_counts.append(float(np.mean(edge_df["receiver_cluster"].astype(str).values == str(c))))

    df_sender = pd.DataFrame({
        "cluster_norm": clusters,
        "cluster_label": [f"C{x}" for x in clusters],
        "t_hat_norm_mean": cl["t_hat_norm_mean"].values.astype(float),
        "sender_presence": np.asarray(sender_counts, float),
    })
    df_receiver = pd.DataFrame({
        "cluster_norm": clusters,
        "cluster_label": [f"C{x}" for x in clusters],
        "t_hat_norm_mean": cl["t_hat_norm_mean"].values.astype(float),
        "receiver_presence": np.asarray(receiver_counts, float),
    })

    df_sender.to_csv(OUT_SENDER_PCURVE, index=False)
    df_receiver.to_csv(OUT_RECEIVER_PCURVE, index=False)
    log("[SAVE]", OUT_SENDER_PCURVE)
    log("[SAVE]", OUT_RECEIVER_PCURVE)

    return df_sender, df_receiver


# ============================================================
# 4) Cluster-path summary for matched edges
# ============================================================
def cluster_path_for_edges(cluster_df: pd.DataFrame, all_edges: pd.DataFrame, seed_row: pd.Series, top_df: pd.DataFrame):
    cl = cluster_df.copy().sort_values("t_hat_norm_mean").reset_index(drop=True)
    clusters = cl["cluster_norm"].astype(str).tolist()

    # global presence curves for all edges
    global_sender_presence = {
        c: float(np.mean(all_edges["sender_cluster"].astype(str).values == str(c)))
        for c in clusters
    }
    global_receiver_presence = {
        c: float(np.mean(all_edges["receiver_cluster"].astype(str).values == str(c)))
        for c in clusters
    }

    # seed edge reference vectors
    seed_sender_vec = np.asarray([
        1.0 if str(c) == str(seed_row["sender_cluster"]) else 0.0
        for c in clusters
    ], float)
    seed_receiver_vec = np.asarray([
        1.0 if str(c) == str(seed_row["receiver_cluster"]) else 0.0
        for c in clusters
    ], float)

    top_edges_with_seed = pd.concat([
        pd.DataFrame([{
            "seed_edge": str(seed_row["edge_label"]),
            "edge_label": str(seed_row["edge_label"]),
            "seed_key": str(seed_row["seed_key"]),
            "score": 1.0,
            "corr": 1.0,
            "sender_cluster": str(seed_row["sender_cluster"]),
            "receiver_cluster": str(seed_row["receiver_cluster"]),
            "t_sender": float(seed_row["t_sender"]),
            "t_receiver": float(seed_row["t_receiver"]),
            "delta_t_receiver_minus_sender": float(seed_row["delta_t_receiver_minus_sender"]),
            "edge_score_main": float(seed_row["edge_score_main"]),
            "receiver_cluster_tf_response": float(pd.to_numeric(seed_row["receiver_cluster_tf_response"], errors="coerce")),
            "sender_cluster_sender_strength": float(pd.to_numeric(seed_row["sender_cluster_sender_strength"], errors="coerce")),
            "receiver_cluster_receiver_strength": float(pd.to_numeric(seed_row["receiver_cluster_receiver_strength"], errors="coerce")),
            "receiver_cluster_net_flow": float(pd.to_numeric(seed_row["receiver_cluster_net_flow"], errors="coerce")),
            "receiver_cluster_io_ratio": float(pd.to_numeric(seed_row["receiver_cluster_io_ratio"], errors="coerce")),
        }]),
        top_df.copy()
    ], axis=0, ignore_index=True)

    top_detail_edges = top_edges_with_seed.head(min(1 + TOPK_DETAIL, len(top_edges_with_seed))).copy()

    detail_rows = []
    summary_rows = []

    for _, er in top_detail_edges.iterrows():
        s = str(er["sender_cluster"])
        r = str(er["receiver_cluster"])

        # 这个 edge 的 sender/receiver one-hot path
        edge_sender_vec = np.asarray([1.0 if str(c) == s else 0.0 for c in clusters], float)
        edge_receiver_vec = np.asarray([1.0 if str(c) == r else 0.0 for c in clusters], float)

        fit_corr_sender = pearson_corr(seed_sender_vec, edge_sender_vec)
        fit_corr_receiver = pearson_corr(seed_receiver_vec, edge_receiver_vec)
        fit_corr_mean = np.nanmean([fit_corr_sender, fit_corr_receiver])

        pass_corr_sender = pearson_corr(
            np.asarray([global_sender_presence[c] for c in clusters], float),
            edge_sender_vec
        )
        pass_corr_receiver = pearson_corr(
            np.asarray([global_receiver_presence[c] for c in clusters], float),
            edge_receiver_vec
        )
        pass_corr_mean = np.nanmean([pass_corr_sender, pass_corr_receiver])

        mu_seed = np.asarray([
            float(seed_row["t_sender"]),
            float(seed_row["t_receiver"]),
            float(seed_row["delta_t_receiver_minus_sender"]),
            float(pd.to_numeric(seed_row["receiver_cluster_tf_response"], errors="coerce")),
        ], float)
        mu_edge = np.asarray([
            float(er["t_sender"]),
            float(er["t_receiver"]),
            float(er["delta_t_receiver_minus_sender"]),
            float(pd.to_numeric(er["receiver_cluster_tf_response"], errors="coerce")),
        ], float)
        expr_corr_mu = pearson_corr(zscore_1d(mu_seed), zscore_1d(mu_edge))

        path_score = 0.0
        if np.isfinite(fit_corr_mean) and np.isfinite(pass_corr_mean) and np.isfinite(expr_corr_mu):
            path_score = max(0.0, fit_corr_mean) * max(0.0, pass_corr_mean) * max(0.0, expr_corr_mu)
            path_score *= max(0.0, float(er["score"]))

        sender_edge_count = int(np.sum(top_df["sender_cluster"].astype(str).values == s))
        receiver_edge_count = int(np.sum(top_df["receiver_cluster"].astype(str).values == r))
        n_edges_cluster = int(sender_edge_count + receiver_edge_count)

        passed_flag = (
            (n_edges_cluster >= MIN_EDGES_PER_CLUSTER)
            and np.isfinite(fit_corr_mean)
            and np.isfinite(pass_corr_mean)
            and np.isfinite(expr_corr_mu)
            and (fit_corr_mean >= FIT_CORR_THR)
            and (pass_corr_mean >= PASS_CORR_THR)
            and (expr_corr_mu >= MU_CORR_THR)
            and (float(er["score"]) >= MIN_SCORE_CLUSTER)
        )

        detail_rows.append({
            "seed_edge": str(seed_row["edge_label"]),
            "edge_label": str(er["edge_label"]),
            "sender_cluster": s,
            "receiver_cluster": r,
            "sender_stage": stage_of_t(float(er["t_sender"])),
            "receiver_stage": stage_of_t(float(er["t_receiver"])),
            "n_edges_cluster": n_edges_cluster,
            "fit_corr_sender": fit_corr_sender,
            "fit_corr_receiver": fit_corr_receiver,
            "fit_corr_mean": fit_corr_mean,
            "pass_corr_sender": pass_corr_sender,
            "pass_corr_receiver": pass_corr_receiver,
            "pass_corr_mean": pass_corr_mean,
            "expr_corr_mu": expr_corr_mu,
            "path_score": path_score,
            "global_score": float(er["score"]),
            "global_corr": float(er["corr"]),
            "t_sender": float(er["t_sender"]),
            "t_receiver": float(er["t_receiver"]),
            "delta_t_receiver_minus_sender": float(er["delta_t_receiver_minus_sender"]),
            "receiver_cluster_tf_response": float(pd.to_numeric(er["receiver_cluster_tf_response"], errors="coerce")),
            "passed_flag_strict": int(passed_flag),
        })

    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(OUT_DETAIL, index=False)
    log("[SAVE]", OUT_DETAIL)

    # per edge summary
    for _, sub in df_detail.groupby("edge_label", sort=False):
        rr = sub.iloc[0]
        summary_rows.append({
            "seed_edge": str(seed_row["edge_label"]),
            "edge_label": str(rr["edge_label"]),
            "global_rank": int(np.where(top_detail_edges["edge_label"].astype(str).values == str(rr["edge_label"]))[0][0]),
            "global_corr": float(rr["global_corr"]),
            "global_score": float(rr["global_score"]),
            "fit_corr_mean": float(rr["fit_corr_mean"]) if pd.notnull(rr["fit_corr_mean"]) else np.nan,
            "pass_corr_mean": float(rr["pass_corr_mean"]) if pd.notnull(rr["pass_corr_mean"]) else np.nan,
            "expr_corr_mu": float(rr["expr_corr_mu"]) if pd.notnull(rr["expr_corr_mu"]) else np.nan,
            "n_passed_clusters_strict": int(rr["passed_flag_strict"]),
            "passed_clusters_strict": f"C{rr['sender_cluster']}→C{rr['receiver_cluster']}" if int(rr["passed_flag_strict"]) == 1 else "",
            "path_score_sum_passed": float(rr["path_score"]) if int(rr["passed_flag_strict"]) == 1 else 0.0,
            "path_score_max_any": float(rr["path_score"]),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SUMMARY, index=False)
    log("[SAVE]", OUT_SUMMARY)

    return df_detail, df_summary, top_detail_edges


# ============================================================
# 5) Flow plot
# ============================================================
def make_flow_plot_for_edge(seed_row: pd.Series, edge_row: pd.Series, detail_row: pd.Series):
    fig = plt.figure(figsize=(9.6, 6.6), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xL, xR = 0.08, 0.94
    y_seed_s, y_seed_r = 0.72, 0.42
    y_edge_s, y_edge_r = 0.72, 0.42

    w01 = normalize01(np.asarray([float(detail_row["path_score"])], float))[0]
    lw = weight_to_lw(1.0 if not np.isfinite(w01) else w01)
    color = FLOW_COLOR_PASS if int(detail_row["passed_flag_strict"]) == 1 else FLOW_COLOR_FAIL

    ax.text(
        xL, 0.95, f"Seed edge: {seed_row['edge_label']}",
        ha="left", va="top", fontsize=14, weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.2),
    )
    ax.text(
        xR, 0.95, f"Matched edge: {edge_row['edge_label']}",
        ha="right", va="top", fontsize=14, weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.2),
    )

    ax.text(
        xL, y_seed_s, f"Sender {seed_row['sender_cluster']}  t={float(seed_row['t_sender']):.3f}",
        ha="left", va="center", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.15),
    )
    ax.text(
        xL, y_seed_r, f"Receiver {seed_row['receiver_cluster']}  t={float(seed_row['t_receiver']):.3f}",
        ha="left", va="center", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.15),
    )

    ax.text(
        xR, y_edge_s, f"Sender {edge_row['sender_cluster']}  t={float(edge_row['t_sender']):.3f}",
        ha="right", va="center", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.15),
    )
    ax.text(
        xR, y_edge_r, f"Receiver {edge_row['receiver_cluster']}  t={float(edge_row['t_receiver']):.3f}",
        ha="right", va="center", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.15),
    )

    draw_bezier(ax, xL + 0.18, y_seed_s, xR - 0.18, y_edge_s, lw=lw, color=color, alpha=FLOW_ALPHA, z=1)
    draw_bezier(ax, xL + 0.18, y_seed_r, xR - 0.18, y_edge_r, lw=lw, color=color, alpha=FLOW_ALPHA, z=1)

    info = (
        f"score={float(detail_row['global_score']):.3f}\n"
        f"corr={float(detail_row['global_corr']):.3f}\n"
        f"fit={float(detail_row['fit_corr_mean']):.3f}\n"
        f"pass={float(detail_row['pass_corr_mean']):.3f}\n"
        f"mu={float(detail_row['expr_corr_mu']):.3f}\n"
        f"path={float(detail_row['path_score']):.3f}\n"
        f"dt={float(detail_row['delta_t_receiver_minus_sender']):.3f}\n"
        f"TF={float(detail_row['receiver_cluster_tf_response']):.3f}"
    )
    ax.text(
        0.50, 0.16, info,
        ha="center", va="center", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.25),
    )

    out_png = OUT_DIR / f"flow_seededge_{safe_name(str(seed_row['edge_label']))}_match_{safe_name(str(edge_row['edge_label']))}.png"
    save_fig(out_png)
    log("[SAVE]", out_png)


# ============================================================
# 6) Main
# ============================================================
def main():
    log("[LOAD]", CLUSTER_PT_CSV)
    log("[LOAD]", EDGE_PT_CSV)

    cluster_df, edge_df_raw = load_inputs()
    edge_df = enrich_edges(cluster_df, edge_df_raw)

    log("[INFO] n_clusters =", len(cluster_df))
    log("[INFO] n_edges =", len(edge_df))

    seed_row = choose_seed_edge(edge_df)
    rank_df, top_df = rank_edges_against_seed(edge_df, seed_row)
    sender_presence_df, receiver_presence_df = build_presence_curves(cluster_df, edge_df)
    detail_df, summary_df, top_detail_edges = cluster_path_for_edges(cluster_df, edge_df, seed_row, top_df)

    # flow plots for top detail
    for _, er in top_detail_edges.iterrows():
        if str(er["edge_label"]) == str(seed_row["edge_label"]):
            continue
        dr = detail_df[detail_df["edge_label"].astype(str) == str(er["edge_label"])].copy()
        if len(dr) == 0:
            continue
        make_flow_plot_for_edge(seed_row, er, dr.iloc[0])

    # extra scatter
    plt.figure(figsize=(7, 6), dpi=DPI)
    x = top_df["t_sender"].values.astype(float)
    y = top_df["t_receiver"].values.astype(float)
    s = np.clip(40 + 160 * normalize01(top_df["score"].values.astype(float)), 20, 220)

    plt.scatter(x, y, s=s, alpha=0.85)
    for _, r in top_df.iterrows():
        plt.text(float(r["t_sender"]), float(r["t_receiver"]), str(r["edge_label"]), fontsize=8)

    plt.scatter(
        [float(seed_row["t_sender"])],
        [float(seed_row["t_receiver"])],
        s=[220],
        marker="*",
        alpha=0.95,
    )
    plt.text(float(seed_row["t_sender"]), float(seed_row["t_receiver"]), f"SEED:{seed_row['edge_label']}", fontsize=9)

    lim0 = min(np.nanmin(x), np.nanmin(y), float(seed_row["t_sender"]), float(seed_row["t_receiver"]), 0.0)
    lim1 = max(np.nanmax(x), np.nanmax(y), float(seed_row["t_sender"]), float(seed_row["t_receiver"]), 1.0)
    plt.plot([lim0, lim1], [lim0, lim1], linestyle="--")
    plt.xlabel("Sender pseudotime")
    plt.ylabel("Receiver pseudotime")
    plt.title(f"Seed-edge matched edges: {seed_row['edge_label']}")
    save_fig(OUT_DIR / f"seed_{safe_name(str(seed_row['edge_label']))}_matched_edges_scatter.png")

    print("\n[DONE] All outputs in:", OUT_DIR)
    print("Note: NO original file modified.")


if __name__ == "__main__":
    main()