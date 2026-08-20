from __future__ import annotations

# ============================================================
# cluster_comm_stageA_plus_edgeC_summary.py
# ------------------------------------------------------------
# 目标：
# A) cluster-level communication stage summary
#    - 沿 cluster communication pseudotime 分析:
#        sender_strength
#        receiver_strength
#        receiver_tf_response
#        net_flow
#        io_ratio
#    - 计算类似 StageA 的指标:
#        amp_raw / coverage / max_speed_raw / t50
#
# C) key-edge timing summary
#    - 基于 edge_pseudotime_summary.csv
#    - 识别关键 sender->receiver 边
#    - 输出 edge 时序指标和热图
#
# 输入（来自前一步 cluster communication pseudotime 脚本）：
# 1) cluster_pseudotime.csv
# 2) edge_pseudotime_summary.csv
#
# 输出：
# A:
#   - A_cluster_feature_metrics.csv
#   - A_cluster_feature_curve_points.csv
#   - A_heatmap_amp_raw.png/.svg
#   - A_heatmap_coverage.png/.svg
#   - A_heatmap_max_speed_raw.png/.svg
#   - A_heatmap_t50.png/.svg
#   - A_feature_curves.png/.svg
#
# C:
#   - C_edge_metrics.csv
#   - C_top_edges_selected.csv
#   - C_heatmap_edge_weight.png/.svg
#   - C_heatmap_edge_delta_t.png/.svg
#   - C_heatmap_edge_receiver_tf.png/.svg
#   - C_edge_timing_scatter.png/.svg
#
# 注：
# - 不做 cell-level 虚假分辨率扩展
# - A 以 cluster 为基本状态单元
# - C 以 edge 为机制补充层
# ============================================================

import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) CONFIG
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

OUT_DIR = PT_DIR / "stageA_cluster_plus_edgeC"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- A settings ----------
A_FEATURE_COLS = [
    "sender_strength",
    "receiver_strength",
    "receiver_tf_response",
    "net_flow",
    "io_ratio",
]

A_FEATURE_NAME_MAP = {
    "sender_strength": "Sender strength",
    "receiver_strength": "Receiver strength",
    "receiver_tf_response": "Receiver TF response",
    "net_flow": "Net flow",
    "io_ratio": "IO ratio",
}

A_N_BINS = 8              # cluster 数通常不多，不宜太大
A_SMOOTH_K = 3
A_Q_BASE = 0.10
A_Q_LO = 0.10
A_Q_HI = 0.90
A_COV_THR = 0.50

# ---------- C settings ----------
TOP_EDGE_N = 25           # 展示的关键边数量
EDGE_RANK_SCORE_WEIGHTS = {
    "edge_weight_z": 0.45,
    "delta_t_z": 0.20,
    "receiver_tf_z": 0.20,
    "edge_score_z": 0.15,
}

# 如果 edge_pseudotime_summary.csv 很大，先取前若干高分边再打分
EDGE_PRESELECT_TOPK = 500

# ---------- plotting ----------
DPI = 320
FIG_W = 11
FIG_H = 7

# ---------- misc ----------
SEED = 42
np.random.seed(SEED)


# ============================================================
# 1) HELPERS
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


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    if (not np.isfinite(s)) or s < 1e-8:
        return np.zeros_like(x, np.float32)
    return ((x - m) / (s + 1e-8)).astype(np.float32)


def smooth_1d(y, k=3):
    y = np.asarray(y, float)
    if k is None or k <= 1 or len(y) == 0:
        return y
    k = int(k)
    if k % 2 == 0:
        k += 1
    if len(y) < k:
        return y
    pad = k // 2
    yy = np.pad(y, (pad, pad), mode="edge")
    w = np.ones((k,), float) / float(k)
    return np.convolve(yy, w, mode="valid")


def make_bins_quantile(t, n_bins):
    t = np.asarray(t, float)
    m = np.isfinite(t)
    tt = t[m]
    if tt.size == 0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        bid = np.zeros_like(t, dtype=np.int64)
        return edges, mids, bid

    edges = np.quantile(tt, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges + np.linspace(0, 1e-10, len(edges)))
    mids = 0.5 * (edges[:-1] + edges[1:])
    bid = np.digitize(t, edges[1:-1], right=True)
    return edges, mids, bid.astype(np.int64)


def baseline_amp_norm(y, q_base=0.10, q_lo=0.10, q_hi=0.90, eps=1e-8):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 3:
        return y * np.nan, np.nan, np.nan
    base = float(np.nanquantile(y[m], q_base))
    lo = float(np.nanquantile(y[m], q_lo))
    hi = float(np.nanquantile(y[m], q_hi))
    amp = max(hi - lo, eps)
    yn = np.clip((y - base) / amp, 0.0, 1.0)
    return yn, base, amp


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def plot_heatmap(mat, row_labels, col_labels, title, out_png):
    plt.figure(figsize=(max(8, 0.55 * len(col_labels)), max(4, 0.42 * len(row_labels))), dpi=DPI)
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Feature / Edge", fontsize=12)
    ax.set_ylabel("Cluster / Metric", fontsize=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    save_fig(out_png)


def feature_display_name(c: str) -> str:
    return A_FEATURE_NAME_MAP.get(str(c), str(c))


def edge_display_name(sender: str, receiver: str) -> str:
    return f"C{sender}→C{receiver}"


# ============================================================
# 2) LOAD
# ============================================================
def load_inputs():
    assert CLUSTER_PT_CSV.exists(), f"Missing: {CLUSTER_PT_CSV}"
    assert EDGE_PT_CSV.exists(), f"Missing: {EDGE_PT_CSV}"

    cl = pd.read_csv(CLUSTER_PT_CSV)
    ed = pd.read_csv(EDGE_PT_CSV)

    assert "cluster_norm" in cl.columns
    assert "t_hat_norm_mean" in cl.columns
    for c in A_FEATURE_COLS:
        assert c in cl.columns, f"Missing column in cluster_pseudotime.csv: {c}"

    assert "sender_cluster" in ed.columns
    assert "receiver_cluster" in ed.columns
    assert "t_sender" in ed.columns
    assert "t_receiver" in ed.columns
    assert "delta_t_receiver_minus_sender" in ed.columns

    cl["cluster_norm"] = cl["cluster_norm"].astype(str).map(normalize_cluster_str)
    ed["sender_cluster"] = ed["sender_cluster"].astype(str).map(normalize_cluster_str)
    ed["receiver_cluster"] = ed["receiver_cluster"].astype(str).map(normalize_cluster_str)

    # numeric cleanup
    for c in ["t_hat_norm_mean", "sigma_t", "tau_mean", "weak_teacher_rank", "weak_teacher_raw"] + A_FEATURE_COLS:
        if c in cl.columns:
            cl[c] = pd.to_numeric(cl[c], errors="coerce")

    for c in [
        "edge_score", "expr_only", "confidence", "n_pairs",
        "t_sender", "t_receiver", "delta_t_receiver_minus_sender"
    ]:
        if c in ed.columns:
            ed[c] = pd.to_numeric(ed[c], errors="coerce")

    cl = cl[np.isfinite(cl["t_hat_norm_mean"].values)].copy().reset_index(drop=True)
    ed = ed[
        np.isfinite(ed["t_sender"].values) &
        np.isfinite(ed["t_receiver"].values) &
        np.isfinite(ed["delta_t_receiver_minus_sender"].values)
    ].copy().reset_index(drop=True)

    return cl, ed


# ============================================================
# 3) A: CLUSTER-LEVEL STAGE SUMMARY
# ============================================================
def build_A_cluster_stage_summary(cluster_df: pd.DataFrame):
    df = cluster_df.copy().reset_index(drop=True)
    df = df.sort_values("t_hat_norm_mean", ascending=True).reset_index(drop=True)

    t = df["t_hat_norm_mean"].values.astype(float)
    n = len(df)

    # bins over clusters
    n_bins = min(A_N_BINS, max(3, n))
    edges, mids, bid = make_bins_quantile(t, n_bins)

    # 因为这里是 cluster-level，不再套用 cell-level min_bin 逻辑
    # 每个 bin 至少要有 1 个 cluster；若太 sparse，后续自然会缺失
    feature_names = [feature_display_name(c) for c in A_FEATURE_COLS]

    AMP = np.full((1, len(A_FEATURE_COLS)), np.nan, float)
    COV = np.full((1, len(A_FEATURE_COLS)), np.nan, float)
    SPEED = np.full((1, len(A_FEATURE_COLS)), np.nan, float)
    T50 = np.full((1, len(A_FEATURE_COLS)), np.nan, float)

    metrics_rows = []
    curve_rows = []

    for fj, col in enumerate(A_FEATURE_COLS):
        v = pd.to_numeric(df[col], errors="coerce").values.astype(float)

        mu = np.full((n_bins,), np.nan, float)
        n_per_bin = np.zeros((n_bins,), int)
        for b in range(n_bins):
            idx = np.where((bid == b) & np.isfinite(v))[0]
            n_per_bin[b] = int(idx.size)
            if idx.size < 1:
                continue
            mu[b] = float(np.nanmean(v[idx]))

        mu_s = smooth_1d(mu, A_SMOOTH_K)

        # 保存曲线点
        for b in range(n_bins):
            curve_rows.append({
                "feature_col": col,
                "feature": feature_display_name(col),
                "bin_id": int(b),
                "bin_mid_t": float(mids[b]),
                "n_clusters_in_bin": int(n_per_bin[b]),
                "mean_raw": float(mu[b]) if np.isfinite(mu[b]) else np.nan,
                "mean_smooth": float(mu_s[b]) if np.isfinite(mu_s[b]) else np.nan,
            })

        if np.sum(np.isfinite(mu_s)) < 3:
            continue

        amp_raw = float(np.nanquantile(mu_s, 0.90) - np.nanquantile(mu_s, 0.10))
        grad = np.gradient(mu_s)
        max_speed_raw = float(np.nanmax(grad))

        yn, base, amp = baseline_amp_norm(mu_s, A_Q_BASE, A_Q_LO, A_Q_HI)
        if np.isfinite(base) and np.isfinite(amp) and amp > 1e-9:
            v_norm = np.clip((v - base) / amp, 0.0, 1.0)
            cov = float(np.mean(v_norm >= A_COV_THR))
            mask = np.isfinite(v_norm) & (v_norm >= A_COV_THR) & np.isfinite(t)
            t50 = float(np.nanmedian(t[mask])) if mask.sum() >= 1 else np.nan
        else:
            cov = np.nan
            t50 = np.nan

        AMP[0, fj] = amp_raw
        COV[0, fj] = cov
        SPEED[0, fj] = max_speed_raw
        T50[0, fj] = t50

        metrics_rows.append({
            "feature_col": col,
            "feature": feature_display_name(col),
            "n_clusters_total": int(n),
            "n_bins": int(n_bins),
            "amp_raw": amp_raw,
            "coverage": cov,
            "max_speed_raw": max_speed_raw,
            "t50": t50,
            "mean_value": float(np.nanmean(v)),
            "sd_value": float(np.nanstd(v)),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    curve_df = pd.DataFrame(curve_rows)

    # 热图：这里 row 只有一行，即 "global cluster trajectory"
    row_labels = ["Global cluster trajectory"]

    plot_heatmap(
        AMP, row_labels, feature_names,
        "A — amp_raw across cluster communication pseudotime",
        OUT_DIR / "A_heatmap_amp_raw.png"
    )
    plot_heatmap(
        COV, row_labels, feature_names,
        "A — coverage across cluster communication pseudotime",
        OUT_DIR / "A_heatmap_coverage.png"
    )
    plot_heatmap(
        SPEED, row_labels, feature_names,
        "A — max_speed_raw across cluster communication pseudotime",
        OUT_DIR / "A_heatmap_max_speed_raw.png"
    )
    plot_heatmap(
        T50, row_labels, feature_names,
        "A — t50 across cluster communication pseudotime",
        OUT_DIR / "A_heatmap_t50.png"
    )

    # feature curves
    plt.figure(figsize=(11, 6), dpi=DPI)
    for col in A_FEATURE_COLS:
        sub = curve_df[curve_df["feature_col"] == col].copy()
        if len(sub) == 0:
            continue
        plt.plot(
            sub["bin_mid_t"].values,
            sub["mean_smooth"].values,
            marker="o",
            label=feature_display_name(col),
        )

    plt.xlabel("Cluster communication pseudotime")
    plt.ylabel("Binned smoothed mean feature")
    plt.title("A — Feature dynamics along cluster communication pseudotime")
    plt.legend()
    save_fig(OUT_DIR / "A_feature_curves.png")

    # cluster ordered table
    cluster_out = df.copy()
    cluster_out["cluster_label"] = ["C" + str(x) for x in cluster_out["cluster_norm"].astype(str)]
    cluster_out.to_csv(OUT_DIR / "A_cluster_ordered_table.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "A_cluster_feature_metrics.csv", index=False)
    curve_df.to_csv(OUT_DIR / "A_cluster_feature_curve_points.csv", index=False)

    log("[SAVE]", OUT_DIR / "A_cluster_feature_metrics.csv")
    log("[SAVE]", OUT_DIR / "A_cluster_feature_curve_points.csv")
    log("[SAVE]", OUT_DIR / "A_cluster_ordered_table.csv")

    return metrics_df, curve_df, cluster_out


# ============================================================
# 4) C: EDGE-LEVEL KEY MECHANISM SUMMARY
# ============================================================
def build_C_edge_summary(cluster_df: pd.DataFrame, edge_df: pd.DataFrame):
    cl = cluster_df.copy().reset_index(drop=True)
    ed = edge_df.copy().reset_index(drop=True)

    # merge receiver TF / sender/receiver state info for interpretation
    merge_cols = [
        "cluster_norm",
        "t_hat_norm_mean",
        "sender_strength",
        "receiver_strength",
        "receiver_tf_response",
        "net_flow",
        "io_ratio",
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
        }),
        on="receiver_cluster",
        how="left",
    )

    # score column
    edge_score_col = None
    for c in ["edge_score", "expr_only", "confidence", "n_pairs"]:
        if c in ed.columns:
            edge_score_col = c
            break

    if edge_score_col is None:
        raise RuntimeError("Cannot find a usable edge score column in edge_pseudotime_summary.csv")

    # preselect
    ed["_pre_rank_score"] = pd.to_numeric(ed[edge_score_col], errors="coerce").fillna(0.0)
    ed = ed.sort_values("_pre_rank_score", ascending=False).head(EDGE_PRESELECT_TOPK).copy().reset_index(drop=True)

    # ranking signals
    ed["edge_score_main"] = pd.to_numeric(ed[edge_score_col], errors="coerce").fillna(0.0)
    ed["edge_weight_z"] = zscore_1d(np.log1p(np.maximum(ed["edge_score_main"].values.astype(np.float32), 0.0)))
    ed["delta_t_z"] = zscore_1d(ed["delta_t_receiver_minus_sender"].values.astype(np.float32))
    ed["receiver_tf_z"] = zscore_1d(
        pd.to_numeric(ed["receiver_cluster_tf_response"], errors="coerce").fillna(0.0).values.astype(np.float32)
    )
    ed["edge_score_z"] = zscore_1d(ed["edge_score_main"].values.astype(np.float32))

    ed["edge_rank_score"] = (
        EDGE_RANK_SCORE_WEIGHTS["edge_weight_z"] * ed["edge_weight_z"].values
        + EDGE_RANK_SCORE_WEIGHTS["delta_t_z"] * ed["delta_t_z"].values
        + EDGE_RANK_SCORE_WEIGHTS["receiver_tf_z"] * ed["receiver_tf_z"].values
        + EDGE_RANK_SCORE_WEIGHTS["edge_score_z"] * ed["edge_score_z"].values
    ).astype(np.float32)

    # interpretation labels
    ed["edge_label"] = [
        edge_display_name(s, r)
        for s, r in zip(ed["sender_cluster"].astype(str), ed["receiver_cluster"].astype(str))
    ]

    # coarse stage label
    def stage_of_t(x):
        if not np.isfinite(x):
            return "NA"
        if x < 0.33:
            return "early"
        elif x < 0.66:
            return "mid"
        return "late"

    ed["sender_stage"] = ed["t_sender"].map(stage_of_t)
    ed["receiver_stage"] = ed["t_receiver"].map(stage_of_t)

    # timing type
    def edge_timing_type(dt):
        if not np.isfinite(dt):
            return "NA"
        if dt < 0.10:
            return "tight-coupled"
        elif dt < 0.30:
            return "moderate-shift"
        return "long-shift"

    ed["timing_type"] = ed["delta_t_receiver_minus_sender"].map(edge_timing_type)

    # top edges
    top_ed = ed.sort_values(
        ["edge_rank_score", "edge_score_main", "delta_t_receiver_minus_sender"],
        ascending=[False, False, False]
    ).head(TOP_EDGE_N).copy().reset_index(drop=True)

    # outputs
    out_cols = [
        "edge_label",
        "sender_cluster",
        "receiver_cluster",
        "edge_rank_score",
        "edge_score_main",
        "t_sender",
        "t_receiver",
        "delta_t_receiver_minus_sender",
        "sender_stage",
        "receiver_stage",
        "timing_type",
        "receiver_cluster_tf_response",
        "sender_cluster_sender_strength",
        "receiver_cluster_receiver_strength",
        "receiver_cluster_net_flow",
        "receiver_cluster_io_ratio",
    ]
    for c in out_cols:
        if c not in top_ed.columns:
            top_ed[c] = np.nan

    top_ed[out_cols].to_csv(OUT_DIR / "C_top_edges_selected.csv", index=False)
    ed.to_csv(OUT_DIR / "C_edge_metrics.csv", index=False)

    log("[SAVE]", OUT_DIR / "C_top_edges_selected.csv")
    log("[SAVE]", OUT_DIR / "C_edge_metrics.csv")

    # heatmaps for top edges
    edge_labels = top_ed["edge_label"].astype(str).tolist()

    mat_weight = top_ed["edge_score_main"].values.astype(float)[None, :]
    mat_dt = top_ed["delta_t_receiver_minus_sender"].values.astype(float)[None, :]
    mat_rtf = pd.to_numeric(top_ed["receiver_cluster_tf_response"], errors="coerce").fillna(0.0).values.astype(float)[None, :]

    plot_heatmap(
        mat_weight,
        ["Top edges"],
        edge_labels,
        "C — edge communication strength (top selected edges)",
        OUT_DIR / "C_heatmap_edge_weight.png"
    )
    plot_heatmap(
        mat_dt,
        ["Top edges"],
        edge_labels,
        "C — edge temporal shift: t(receiver)-t(sender)",
        OUT_DIR / "C_heatmap_edge_delta_t.png"
    )
    plot_heatmap(
        mat_rtf,
        ["Top edges"],
        edge_labels,
        "C — receiver cluster TF response (top selected edges)",
        OUT_DIR / "C_heatmap_edge_receiver_tf.png"
    )

    # scatter
    plt.figure(figsize=(8, 6), dpi=DPI)
    x = top_ed["t_sender"].values.astype(float)
    y = top_ed["t_receiver"].values.astype(float)
    s = np.clip(
        50.0 + 120.0 * zscore_1d(np.log1p(np.maximum(top_ed["edge_score_main"].values.astype(np.float32), 0.0))),
        20.0, 220.0
    )

    plt.scatter(x, y, s=s, alpha=0.85)
    for _, r in top_ed.iterrows():
        plt.text(float(r["t_sender"]), float(r["t_receiver"]), str(r["edge_label"]), fontsize=8)

    lim0 = min(np.nanmin(x), np.nanmin(y), 0.0)
    lim1 = max(np.nanmax(x), np.nanmax(y), 1.0)
    plt.plot([lim0, lim1], [lim0, lim1], linestyle="--")
    plt.xlabel("Sender pseudotime")
    plt.ylabel("Receiver pseudotime")
    plt.title("C — Key edge timing map")
    save_fig(OUT_DIR / "C_edge_timing_scatter.png")

    return ed, top_ed


# ============================================================
# 5) SUMMARY TABLE
# ============================================================
def build_joint_summary(A_metrics: pd.DataFrame, C_top_edges: pd.DataFrame):
    rows = []

    # A summary
    if len(A_metrics) > 0:
        a1 = A_metrics.sort_values("amp_raw", ascending=False).head(3).copy()
        a2 = A_metrics.sort_values("t50", ascending=True, na_position="last").head(3).copy()
        a3 = A_metrics.sort_values("max_speed_raw", ascending=False).head(3).copy()

        for _, r in a1.iterrows():
            rows.append({
                "panel": "A",
                "category": "top_amplitude",
                "item": r["feature"],
                "score": r["amp_raw"],
                "detail": f"coverage={r['coverage']:.3f}; t50={r['t50']:.3f}" if np.isfinite(r["t50"]) else f"coverage={r['coverage']:.3f}; t50=NA",
            })

        for _, r in a2.iterrows():
            rows.append({
                "panel": "A",
                "category": "earliest_features",
                "item": r["feature"],
                "score": r["t50"],
                "detail": f"amp={r['amp_raw']:.3f}; speed={r['max_speed_raw']:.3f}",
            })

        for _, r in a3.iterrows():
            rows.append({
                "panel": "A",
                "category": "fastest_features",
                "item": r["feature"],
                "score": r["max_speed_raw"],
                "detail": f"amp={r['amp_raw']:.3f}; t50={r['t50']:.3f}" if np.isfinite(r["t50"]) else f"amp={r['amp_raw']:.3f}; t50=NA",
            })

    # C summary
    if len(C_top_edges) > 0:
        c1 = C_top_edges.sort_values("edge_rank_score", ascending=False).head(5).copy()
        c2 = C_top_edges.sort_values("delta_t_receiver_minus_sender", ascending=False).head(5).copy()
        c3 = C_top_edges.sort_values("t_sender", ascending=True).head(5).copy()

        for _, r in c1.iterrows():
            rows.append({
                "panel": "C",
                "category": "top_ranked_edges",
                "item": r["edge_label"],
                "score": r["edge_rank_score"],
                "detail": f"dt={r['delta_t_receiver_minus_sender']:.3f}; receiver_tf={pd.to_numeric(r['receiver_cluster_tf_response'], errors='coerce')}",
            })

        for _, r in c2.iterrows():
            rows.append({
                "panel": "C",
                "category": "largest_temporal_shift_edges",
                "item": r["edge_label"],
                "score": r["delta_t_receiver_minus_sender"],
                "detail": f"sender_t={r['t_sender']:.3f}; receiver_t={r['t_receiver']:.3f}",
            })

        for _, r in c3.iterrows():
            rows.append({
                "panel": "C",
                "category": "earliest_sender_edges",
                "item": r["edge_label"],
                "score": r["t_sender"],
                "detail": f"receiver_t={r['t_receiver']:.3f}; dt={r['delta_t_receiver_minus_sender']:.3f}",
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "AC_joint_summary.csv", index=False)
    log("[SAVE]", OUT_DIR / "AC_joint_summary.csv")
    return out


# ============================================================
# 6) MAIN
# ============================================================
def main():
    log("[LOAD]", CLUSTER_PT_CSV)
    log("[LOAD]", EDGE_PT_CSV)

    cluster_df, edge_df = load_inputs()

    log("[INFO] n_clusters =", len(cluster_df))
    log("[INFO] n_edges =", len(edge_df))

    A_metrics, A_curves, A_cluster_out = build_A_cluster_stage_summary(cluster_df)
    C_edge_metrics, C_top_edges = build_C_edge_summary(cluster_df, edge_df)
    AC_summary = build_joint_summary(A_metrics, C_top_edges)

    log("[DONE] outputs in:", OUT_DIR.resolve())

    print("\n=== Suggested interpretation order ===")
    print("1) A_cluster_feature_metrics.csv")
    print("2) A_feature_curves.png")
    print("3) C_top_edges_selected.csv")
    print("4) C_edge_timing_scatter.png")
    print("5) AC_joint_summary.csv")


if __name__ == "__main__":
    main()