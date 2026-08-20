# ============================================================
# Stage-B component ablation: primary endpoint line plot
# - 只展示第一列主指标：Diag-offdiag Δ
# - 不重跑模型
# - 读取已有 eval_fig2lm_standalone_strict_summary.csv
# - 输出 png / pdf / svg
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0. PATHS
# ============================================================

SUMMARY_CSV = get_runs_root() / "stageB_component_ablation_integrated" / "eval_fig2lm_standalone_strict_summary.csv"

OUT_DIR = get_runs_root() / "stageB_component_ablation_integrated" / "eval_fig2lm_standalone_strict_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUT_DIR / "stageB_component_ablation_primary_diagoffdiag_line"

# ============================================================
# 1. SETTINGS
# ============================================================

PRIMARY_METRIC = "test_fig2l_diag_offdiag_delta"
PRIMARY_LABEL = "Diag–offdiag correspondence Δ"

MODE_ORDER = [
    "full_stageB",
    "simple_finetune",
    "no_lowrank_adapter",
    "no_cluster_correction",
    "no_selected_peak_residual",
    "no_group_scale_bias",
    "no_cellgroup_adapter",
]

MODE_LABELS = {
    "full_stageB": "Full\nStage-B",
    "simple_finetune": "Simple\nfine-tune",
    "no_lowrank_adapter": "– Low-rank\nadapter",
    "no_cluster_correction": "– Cluster\ncorrection",
    "no_selected_peak_residual": "– Selected-peak\nresidual",
    "no_group_scale_bias": "– Group\nscale/bias",
    "no_cellgroup_adapter": "– Cell-group\nadapter",
}

# 主图小 panel 建议尺寸
FIGSIZE = (5.4, 2.25)

# ============================================================
# 2. HELPERS
# ============================================================

def find_mode_col(df):
    candidates = ["mode", "ablation_mode", "ablation", "name", "model", "setting"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Cannot find mode column. Available columns:\n"
        + "\n".join(df.columns.tolist())
    )

def reorder_df(df, mode_col):
    df = df.copy()
    df[mode_col] = df[mode_col].astype(str)

    present = [m for m in MODE_ORDER if m in df[mode_col].tolist()]
    if len(present) == 0:
        raise ValueError("None of MODE_ORDER entries found in summary CSV.")

    df = df[df[mode_col].isin(present)].copy()
    df["__order__"] = df[mode_col].map({m: i for i, m in enumerate(MODE_ORDER)})
    df = df.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)
    return df

# ============================================================
# 3. LOAD DATA
# ============================================================

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(f"Summary CSV not found:\n{SUMMARY_CSV}")

df = pd.read_csv(SUMMARY_CSV)
mode_col = find_mode_col(df)

if PRIMARY_METRIC not in df.columns:
    raise ValueError(
        f"Missing primary metric column: {PRIMARY_METRIC}\n\n"
        "Available columns:\n" + "\n".join(df.columns.tolist())
    )

df = reorder_df(df, mode_col)

modes = df[mode_col].astype(str).tolist()
labels = [MODE_LABELS.get(m, m) for m in modes]
values = df[PRIMARY_METRIC].astype(float).values

# 找 full_stageB 和 simple_finetune
full_idx = modes.index("full_stageB") if "full_stageB" in modes else None
simple_idx = modes.index("simple_finetune") if "simple_finetune" in modes else None

# 相对 full Stage-B 的 loss，用于辅助标注
if full_idx is not None:
    full_value = values[full_idx]
    loss_vs_full = full_value - values
else:
    full_value = np.nan
    loss_vs_full = np.full_like(values, np.nan)

# ============================================================
# 4. PLOT
# ============================================================

x = np.arange(len(values))

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)

# 主折线
ax.plot(
    x,
    values,
    linewidth=1.8,
    color="#555555",
    zorder=2
)

# 普通点
ax.scatter(
    x,
    values,
    s=42,
    color="#888888",
    edgecolor="white",
    linewidth=0.8,
    zorder=3
)

# Full Stage-B 高亮
if full_idx is not None:
    ax.scatter(
        [full_idx],
        [values[full_idx]],
        s=78,
        color="#B2182B",
        edgecolor="black",
        linewidth=0.8,
        zorder=4
    )

# Simple fine-tune 次级高亮
if simple_idx is not None:
    ax.scatter(
        [simple_idx],
        [values[simple_idx]],
        s=64,
        color="#2166AC",
        edgecolor="black",
        linewidth=0.7,
        zorder=4
    )

# 数值标注
for i, v in enumerate(values):
    if i == full_idx:
        color = "#B2182B"
        weight = "bold"
    elif i == simple_idx:
        color = "#2166AC"
        weight = "bold"
    else:
        color = "#333333"
        weight = "normal"

    ax.text(
        i,
        v + 0.018,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
        fontweight=weight
    )

# 给 no_lowrank_adapter / no_cellgroup_adapter 加 loss 标注，突出组件贡献
for mode_key in ["no_lowrank_adapter", "no_cellgroup_adapter"]:
    if mode_key in modes and full_idx is not None:
        i = modes.index(mode_key)
        ax.text(
            i,
            values[i] - 0.035,
            f"loss {loss_vs_full[i]:.2f}",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#B2182B"
        )

# 横轴
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.3)

# 纵轴
ymin = max(0.0, np.nanmin(values) - 0.10)
ymax = min(1.05, np.nanmax(values) + 0.10)
ax.set_ylim(ymin, ymax)

ax.set_ylabel(PRIMARY_LABEL, fontsize=9.5, fontweight="bold")

# 标题
ax.set_title(
    "Primary Stage-B ablation endpoint",
    fontsize=11,
    fontweight="bold",
    pad=8
)

# 参考线：simple fine-tune
if simple_idx is not None:
    ax.axhline(
        values[simple_idx],
        color="#2166AC",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        zorder=1
    )
    ax.text(
        len(values) - 0.45,
        values[simple_idx] + 0.006,
        "simple fine-tune",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#2166AC"
    )

# 参考线：full Stage-B
if full_idx is not None:
    ax.axhline(
        values[full_idx],
        color="#B2182B",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
        zorder=1
    )
    ax.text(
        len(values) - 0.45,
        values[full_idx] + 0.006,
        "full Stage-B",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#B2182B"
    )

# 风格：去掉多余边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="y", labelsize=8.5, width=0.8, length=3)
ax.tick_params(axis="x", width=0.8, length=3)

# 轻微 y 网格，方便读数
ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.22)
ax.grid(axis="x", visible=False)

# 简短说明，适合小 panel
fig.text(
    0.01,
    0.005,
    "Higher values indicate stronger diagonal–offdiagonal separation in the predicted-vs-proxy cluster correspondence matrix.",
    fontsize=7.4,
    ha="left"
)

plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])

# ============================================================
# 5. SAVE
# ============================================================

for ext in ["png", "pdf", "svg"]:
    out = f"{OUT_PREFIX}.{ext}"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {out}")

plt.close(fig)

print("[DONE] Primary diag-offdiag ablation line plot finished.")
