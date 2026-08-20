# Source: 诛仙剑阵2-F-改（解释）-Shrap-改-加权(4).docx
# Extracted from the user-provided Word code; scientific logic preserved.
# build_tf_re_rank_mix_model.py
import os, glob
import numpy as np
import pandas as pd

# =========================
# 0) PATHS（只改这里）
# =========================
IN_ROOT = str(get_project_root() / "networks_fixweight_20260228_v1_worldword_raw")
IN_PER  = os.path.join(IN_ROOT, "tf_re")

OUT_ROOT = str(get_project_root() / "networks_fixweight_20260228_v2_rank_mix_model")
OUT_TRAIN = os.path.join(OUT_ROOT, "tf_re_train")
OUT_EXPL  = os.path.join(OUT_ROOT, "tf_re_explain")
os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_EXPL, exist_ok=True)

QC_OUT = os.path.join(OUT_ROOT, "qc_v2_rank_mix_model.tsv")

# =========================
# 1) HYPERPARAMS（demon 同款）
# =========================
TOPN_PER_RE = 50

TOPM_TRAIN = 20000
TOPM_EXPL  = 2000

ALPHA_TRAIN = 0.0
BETA_TRAIN  = 1.0

ALPHA_EXPL = 1.0
BETA_EXPL  = 1.0

EPS = 1e-12

# =========================
# 2) Helpers
# =========================
def cap_by_tf(df, topm):
    df = df.sort_values(["tf_node", "weight"], ascending=[True, False])
    return df.groupby("tf_node", as_index=False, group_keys=False).head(topm)

def cap_by_re(df, topn):
    df = df.sort_values(["re_node", "weight"], ascending=[True, False])
    return df.groupby("re_node", as_index=False, group_keys=False).head(topn)

def mix_rank_evidence(df, alpha, beta):
    """
    w ∝ (orig_weight+eps)^beta * rank^{-alpha}, then per-TF normalize
    """
    d = df.sort_values(["tf_node", "weight"], ascending=[True, False]).copy()
    d["rank"] = d.groupby("tf_node").cumcount() + 1

    ev = np.power(d["weight"].to_numpy(dtype=float) + EPS, beta)

    if alpha <= 0:
        rk = 1.0
    else:
        rk = np.power(d["rank"].to_numpy(dtype=float), -alpha)

    w = ev * rk
    d["weight"] = w

    s = d.groupby("tf_node")["weight"].transform("sum")
    d["weight"] = d["weight"] / (s + EPS)

    return d.drop(columns=["rank"])

def eff_out_quantiles(df, qs=(0.1, 0.5, 0.9)):
    sumsq = df.groupby("tf_node")["weight"].apply(lambda x: float(np.sum(np.square(x.values))))
    deff = 1.0 / (sumsq + EPS)
    return {f"deff_p{int(q*100)}": float(deff.quantile(q)) for q in qs}

def wmax_quantiles(df, qs=(0.5, 0.9, 0.99)):
    wmax = df.groupby("tf_node")["weight"].max()
    return {f"wmax_p{int(q*100)}": float(wmax.quantile(q)) for q in qs}

# ===== [ADD BACK] evidence保真度QC：Spearman median（每TF topk）=====
def spearman_per_tf_median(orig_df, new_df, topk=2000):
    """
    看“权重排序是否保留 evidence 的相对差异”
    仅在每个 TF 的 topk 边上算 Spearman（对齐 re_node）
    返回所有 TF spearman 的中位数
    """
    from scipy.stats import spearmanr

    # 取每个 TF topk（按 orig weight）
    o = orig_df.sort_values(["tf_node", "weight"], ascending=[True, False])
    o = o.groupby("tf_node", as_index=False, group_keys=False).head(topk)

    # new 对齐同一批 (tf, re)
    key = ["tf_node", "re_node"]
    m = o[key + ["weight"]].merge(
        new_df[key + ["weight"]],
        on=key, how="inner", suffixes=("_orig", "_new")
    )

    sp = []
    for tf, g in m.groupby("tf_node"):
        if len(g) < 5:
            continue
        r = spearmanr(g["weight_orig"].values, g["weight_new"].values).correlation
        if np.isfinite(r):
            sp.append(float(r))
    if not sp:
        return np.nan
    return float(np.median(sp))

# =========================
# 3) Main
# =========================
per_files = sorted(glob.glob(os.path.join(IN_PER, "edges_*.tsv")))
assert per_files, f"No per-ds files found under: {IN_PER}"

qc_rows = []
for fp in per_files:
    df0 = pd.read_csv(fp, sep="\t", usecols=["tf_node", "re_node", "weight", "dataset"])
    ds = str(df0["dataset"].iloc[0])

    # ---------- TRAIN ----------
    d = cap_by_tf(df0, TOPM_TRAIN)
    d = cap_by_re(d, TOPN_PER_RE)
    d = mix_rank_evidence(d, alpha=ALPHA_TRAIN, beta=BETA_TRAIN)
    out_train = os.path.join(OUT_TRAIN, f"edges_{ds}.tsv")
    d.to_csv(out_train, sep="\t", index=False)

    # ---------- EXPLAIN ----------
    e = cap_by_tf(df0, TOPM_EXPL)
    e = cap_by_re(e, TOPN_PER_RE)
    e = mix_rank_evidence(e, alpha=ALPHA_EXPL, beta=BETA_EXPL)
    out_expl = os.path.join(OUT_EXPL, f"edges_{ds}.tsv")
    e.to_csv(out_expl, sep="\t", index=False)

    # QC
    qd = eff_out_quantiles(d); wd = wmax_quantiles(d)
    qe = eff_out_quantiles(e); we = wmax_quantiles(e)

    # ===== [ADD BACK] explain: Spearman median vs orig (每TF top2k以内) =====
    sp_expl = spearman_per_tf_median(df0, e, topk=min(TOPM_EXPL, 2000))

    qc_rows.append({
        "dataset": ds,

        "train_edges": int(len(d)),
        "train_n_tf": int(d["tf_node"].nunique()),
        "train_n_re": int(d["re_node"].nunique()),
        "train_mean_RE_per_TF": float(len(d) / max(d["tf_node"].nunique(), 1)),
        "train_mean_TF_per_RE": float(len(d) / max(d["re_node"].nunique(), 1)),
        **{f"train_{k}": v for k, v in qd.items()},
        **{f"train_{k}": v for k, v in wd.items()},

        "expl_edges": int(len(e)),
        "expl_n_tf": int(e["tf_node"].nunique()),
        "expl_n_re": int(e["re_node"].nunique()),
        "expl_mean_RE_per_TF": float(len(e) / max(e["tf_node"].nunique(), 1)),
        "expl_mean_TF_per_RE": float(len(e) / max(e["re_node"].nunique(), 1)),
        **{f"expl_{k}": v for k, v in qe.items()},
        **{f"expl_{k}": v for k, v in we.items()},
        "expl_spearman_median_vs_orig": float(sp_expl),

        "out_train": out_train,
        "out_explain": out_expl,
    })

qc = pd.DataFrame(qc_rows).sort_values(["expl_deff_p50", "dataset"], ascending=[False, True])
qc.to_csv(QC_OUT, sep="\t", index=False)

print("[DONE]")
print("IN_ROOT :", IN_ROOT)
print("OUT_ROOT:", OUT_ROOT)
print("QC TSV  :", QC_OUT)
print("train_dir:", OUT_TRAIN, "files=", len(glob.glob(os.path.join(OUT_TRAIN, 'edges_*.tsv'))))
print("expl_dir :", OUT_EXPL,  "files=", len(glob.glob(os.path.join(OUT_EXPL,  'edges_*.tsv'))))
