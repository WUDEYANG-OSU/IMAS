# Source: 诛仙剑阵1-过滤质控1(3).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, json, math, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

# =========================
# CONFIG (改这里)
# =========================
BASE = get_project_root()
ROOT = BASE / "LLM-WorldWord"

# 原先的 prior tag（可不填，仅用于记录）
OLD_OUT_TAG = "G0_full49_v1"

# 新 tag：canonical peaks 过滤后的 prior 输出目录（不会碰旧输出）
NEW_OUT_TAG = "G0_full49_v1_canonPeaks"

# prior 参数（与 build 一致）
MAX_DIST = 200_000
TOPK = 3
TAU = 50_000.0

# 直接剔除的“坏数据集”
EXCLUDE = {"Osteosarcoma-Tissue2", "Medulloblastoma-Tissue2"}

# dataset 目录过滤
BAD_DS_RE = re.compile(r"^(新建文件夹|\.|__|temp|tmp|_QC|_WORLD_GLOBAL|世界模型构建|矫正模型相关图|魔剑-真世界)$", re.I)

# 输出目录
QC_OUTDIR = ROOT / "_QC" / NEW_OUT_TAG
QC_OUTDIR.mkdir(parents=True, exist_ok=True)

# 报警阈值（你可改）
ALERT_SRC_COV_LT = 0.90
ALERT_EDGE_COV_LT = 0.70
ALERT_NONCANON_FRAC_GT = 0.30         # peaks 落在 non-canon contig 的比例太高
ALERT_FRAC_NEAR_MAXDIST_GT = 0.20

# =========================
# CHR / CONTIG HELPERS
# =========================
def canonize_chr(c):
    c = str(c).strip()
    c = c.replace("Chr", "chr").replace("CHR", "chr")
    # mitochondria
    if c in {"MT", "chrMT"}: c = "chrM"
    if c == "M": c = "chrM"
    # add chr prefix if missing
    if c.isdigit(): c = "chr" + c
    if c in {"X", "Y"}: c = "chr" + c
    return c

CANON = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])

def is_canon_chr(c):
    return canonize_chr(c) in CANON

# =========================
# ASSETS PICKER (与你原逻辑一致)
# =========================
def pick_assets_dir(dataset_root: Path):
    need = ["genes.tsv", "peaks.bed", "node_index.json"]
    direct = dataset_root / "assets"
    if direct.is_dir() and all((direct / x).exists() for x in need):
        return direct, "direct"

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
        return None, "missing"
    cands.sort(reverse=True)
    return cands[0][2], "recursive_best_peaks"

# =========================
# IO HELPERS
# =========================
def iter_peaks_bed(bed_path: Path):
    with open(bed_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line[0] == "#":
                continue
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1]); end = int(parts[2])
            except Exception:
                continue
            name = parts[3]
            center = (start + end) // 2
            yield chrom, center, name, parts

def robust_quantile(arr, q):
    if arr is None or len(arr) == 0:
        return None
    return int(np.quantile(np.asarray(arr, dtype=np.float64), q))

def estimate_tau(distances, w0s):
    d = np.asarray(distances, dtype=np.float64)
    w = np.asarray(w0s, dtype=np.float64)
    mask = (d >= 0) & np.isfinite(d) & (w > 0) & (w < 1) & np.isfinite(w)
    d = d[mask]; w = w[mask]
    if len(d) < 50:
        return None, "too_few_points"
    y = -np.log(w)  # y ~ d/tau
    denom = float(np.dot(d, d))
    if denom <= 0:
        return None, "bad_denom"
    a = float(np.dot(d, y)) / denom
    if a <= 0:
        return None, "nonpos_slope"
    return 1.0 / a, "ok"

# =========================
# STEP 1: contig stats + write canonical peaks
# =========================
def write_canonical_peaks(in_bed: Path, out_bed: Path):
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    chr_cnt = Counter()

    with open(in_bed, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_bed, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip() or line[0] == "#":
                continue
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 3:
                continue
            chrom0 = parts[0]
            chrom = canonize_chr(chrom0)
            n_in += 1
            chr_cnt[chrom] += 1
            if chrom in CANON:
                parts[0] = chrom  # 规范化
                fout.write("\t".join(parts) + "\n")
                n_out += 1

    n_can = sum(v for k, v in chr_cnt.items() if k in CANON)
    n_non = n_in - n_can
    noncanon_frac = (n_non / n_in) if n_in else None
    top_noncanon = [k for k, _ in chr_cnt.most_common(30) if k not in CANON][:10]

    return {
        "peaks_in": n_in,
        "peaks_canon": n_can,
        "peaks_noncanon": n_non,
        "noncanon_frac": noncanon_frac,
        "chr_unique_total": len(chr_cnt),
        "chr_unique_canon": len([k for k in chr_cnt if k in CANON]),
        "top_noncanon_chr": ",".join(top_noncanon),
    }

# =========================
# STEP 2: build prior using canonical peaks
# =========================
def build_prior_one(dataset_name: str, assets_dir: Path, peaks_bed: Path, out_tag: str):
    genes_tsv = assets_dir / "genes.tsv"
    node_index_json = assets_dir / "node_index.json"

    out_dir = assets_dir / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "edges_re_gene_dist.tsv"
    rep_path = out_dir / "edges_re_gene_dist.report.json"

    # node_index
    with open(node_index_json, "r", encoding="utf-8") as f:
        node_index = json.load(f)

    # genes
    genes = pd.read_csv(
        genes_tsv, sep="\t",
        dtype={"gene_id": str, "symbol": str, "chr": str, "tss": "Int64", "strand": str},
        low_memory=False
    )
    if not {"chr", "tss", "symbol"}.issubset(set(genes.columns)):
        return {"dataset": dataset_name, "status": "fail_genes_cols", "assets_dir": str(assets_dir)}

    genes = genes[genes["chr"].notna() & genes["tss"].notna()]
    genes = genes[genes["tss"].astype("int64") >= 0]
    if len(genes) == 0:
        return {"dataset": dataset_name, "status": "fail_genes_empty", "assets_dir": str(assets_dir)}

    # normalize gene chr for robustness
    genes["chr"] = genes["chr"].astype(str).map(canonize_chr)

    chr2_tss, chr2_sym = {}, {}
    for chrom, sub in genes.groupby("chr"):
        sub = sub.sort_values("tss")
        chr2_tss[chrom] = sub["tss"].to_numpy(dtype=np.int64)
        chr2_sym[chrom] = sub["symbol"].astype(str).to_numpy()

    n_edges = 0
    dist_list = []
    per_peak_deg = []
    peaks_total = 0
    peaks_has_src = 0
    peaks_has_edge = 0

    with open(out_path, "w", encoding="utf-8") as w:
        w.write("src_id\tdst_id\trel_type\tw0\tdistance\n")
        for chrom, center, peak_name, _ in iter_peaks_bed(peaks_bed):
            peaks_total += 1
            chrom = canonize_chr(chrom)

            tss = chr2_tss.get(chrom)
            syms = chr2_sym.get(chrom)
            if tss is None or len(tss) == 0:
                per_peak_deg.append(0)
                continue

            src_id = node_index.get(f"RE:{peak_name}")
            if src_id is None:
                per_peak_deg.append(0)
                continue
            peaks_has_src += 1

            pos = int(np.searchsorted(tss, center))
            cand = []
            L, R = pos - 1, pos
            while (L >= 0 or R < len(tss)) and len(cand) < TOPK:
                pick_left = False
                if L >= 0 and R < len(tss):
                    if abs(int(tss[L]) - int(center)) <= abs(int(tss[R]) - int(center)):
                        pick_left = True
                elif L >= 0:
                    pick_left = True

                if pick_left:
                    d = abs(int(tss[L]) - int(center))
                    sym = str(syms[L]); L -= 1
                else:
                    d = abs(int(tss[R]) - int(center))
                    sym = str(syms[R]); R += 1

                if d <= MAX_DIST:
                    cand.append((d, sym))

            deg = 0
            for d, sym in cand:
                dst_id = node_index.get(f"GENE:{sym}")
                if dst_id is None:
                    continue
                w0 = math.exp(-float(d) / float(TAU))
                w.write(f"{src_id}\t{dst_id}\tre_gene_dist\t{w0:.6g}\t{d}\n")
                n_edges += 1
                deg += 1
                dist_list.append(int(d))

            if deg > 0:
                peaks_has_edge += 1
            per_peak_deg.append(deg)

    rep = {
        "dataset": dataset_name,
        "assets_dir": str(assets_dir),
        "max_dist": MAX_DIST,
        "topk": TOPK,
        "tau": TAU,
        "peaks_total": int(peaks_total),
        "peaks_has_src_in_node_index": int(peaks_has_src),
        "peaks_has_at_least_1_edge": int(peaks_has_edge),
        "peak_src_coverage": (peaks_has_src / peaks_total) if peaks_total else None,
        "peak_edge_coverage": (peaks_has_edge / peaks_total) if peaks_total else None,
        "n_edges": int(n_edges),
        "distance_p50": int(np.median(dist_list)) if dist_list else None,
        "distance_p95": int(np.quantile(dist_list, 0.95)) if dist_list else None,
        "per_peak_degree_mean": float(np.mean(per_peak_deg)) if per_peak_deg else None,
        "status": "ok",
        "out": str(out_path),
        "report": str(rep_path),
        "out_tag": out_tag,
        "peaks_used": str(peaks_bed),
    }
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep

# =========================
# STEP 3: QC from edges + report
# =========================
def qc_from_outputs(dataset_name: str, out_dir: Path, rep: dict, contig_stats: dict):
    edges_path = out_dir / "edges_re_gene_dist.tsv"

    # edges 可能为空（0行），用 try 防护
    try:
        df = pd.read_csv(edges_path, sep="\t")
    except Exception:
        df = pd.DataFrame(columns=["src_id","dst_id","rel_type","w0","distance"])

    n_edges_qc = int(len(df)) if len(df) else 0
    n_src_unique = int(df["src_id"].astype(str).nunique()) if n_edges_qc else 0
    n_dst_unique = int(df["dst_id"].astype(str).nunique()) if n_edges_qc else 0

    # dist / w0
    if n_edges_qc:
        ok_mask = pd.to_numeric(df["distance"], errors="coerce").notna() & pd.to_numeric(df["w0"], errors="coerce").notna()
        dist = pd.to_numeric(df.loc[ok_mask, "distance"], errors="coerce").astype(np.int64).to_numpy()
        w0 = pd.to_numeric(df.loc[ok_mask, "w0"], errors="coerce").astype(np.float64).to_numpy()
    else:
        dist = np.array([], dtype=np.int64)
        w0 = np.array([], dtype=np.float64)

    distance_p50_qc = robust_quantile(dist, 0.50)
    distance_p95_qc = robust_quantile(dist, 0.95)
    distance_max_qc = int(np.max(dist)) if len(dist) else None
    frac_near_maxdist = float(np.mean(dist >= int(0.95 * MAX_DIST))) if len(dist) else None

    # degree by src
    if n_edges_qc:
        deg = df["src_id"].astype(str).value_counts()
        per_src_deg_mean = float(deg.mean()) if len(deg) else None
        per_src_deg_p50 = int(deg.median()) if len(deg) else None
        per_src_deg_p95 = robust_quantile(deg.to_numpy(dtype=np.int64), 0.95) if len(deg) else None
        frac_src_deg_eq_topk = float(np.mean(deg.to_numpy() == TOPK)) if len(deg) else None
        frac_src_deg_lt_topk = float(np.mean(deg.to_numpy() < TOPK)) if len(deg) else None
        rel_counts = df["rel_type"].astype(str).value_counts()
        rel_type_mode = rel_counts.index[0] if len(rel_counts) else None
        rel_type_purity = float(rel_counts.iloc[0] / rel_counts.sum()) if len(rel_counts) else None
        w0_valid_frac = float(np.mean((w0 > 0) & (w0 <= 1))) if len(w0) else None
        tau_hat, tau_status = estimate_tau(dist, w0)
    else:
        per_src_deg_mean = per_src_deg_p50 = per_src_deg_p95 = None
        frac_src_deg_eq_topk = frac_src_deg_lt_topk = None
        rel_type_mode = None
        rel_type_purity = None
        w0_valid_frac = None
        tau_hat, tau_status = None, "too_few_points"

    # alerts
    alerts = []
    src_cov = rep.get("peak_src_coverage")
    edge_cov = rep.get("peak_edge_coverage")
    noncanon_frac = contig_stats.get("noncanon_frac")

    if src_cov is not None and src_cov < ALERT_SRC_COV_LT:
        alerts.append("ALERT_SRC_LOW")
    if edge_cov is not None and edge_cov < ALERT_EDGE_COV_LT:
        alerts.append("ALERT_EDGE_LOW")
    if noncanon_frac is not None and noncanon_frac > ALERT_NONCANON_FRAC_GT:
        alerts.append("ALERT_NONCANON_HIGH")
    if frac_near_maxdist is not None and frac_near_maxdist > ALERT_FRAC_NEAR_MAXDIST_GT:
        alerts.append("ALERT_MANY_NEAR_MAXDIST")
    if rel_type_purity is not None and rel_type_purity < 0.99:
        alerts.append("ALERT_RELTYPE_MIXED")
    if tau_hat is not None and not (0.8 * TAU <= tau_hat <= 1.2 * TAU):
        alerts.append("ALERT_TAU_DRIFT")

    status = "ok" if not alerts else "warn"

    return {
        "dataset": dataset_name,
        "status": status,
        "alerts": "|".join(alerts),

        "assets_dir": rep.get("assets_dir"),
        "out_dir": str(out_dir),
        "old_out_tag": OLD_OUT_TAG,
        "new_out_tag": NEW_OUT_TAG,

        # contig stats (原 peaks 的统计)
        **contig_stats,

        # report (canonical peaks 构建出来的)
        "peaks_total_report": rep.get("peaks_total"),
        "peaks_has_src_report": rep.get("peaks_has_src_in_node_index"),
        "peaks_has_edge_report": rep.get("peaks_has_at_least_1_edge"),
        "peak_src_coverage_report": rep.get("peak_src_coverage"),
        "peak_edge_coverage_report": rep.get("peak_edge_coverage"),
        "n_edges_report": rep.get("n_edges"),
        "per_peak_degree_mean_report": rep.get("per_peak_degree_mean"),
        "distance_p50_report": rep.get("distance_p50"),
        "distance_p95_report": rep.get("distance_p95"),

        # qc from edges
        "n_edges_qc": n_edges_qc,
        "n_src_unique_qc": n_src_unique,
        "n_dst_unique_qc": n_dst_unique,
        "distance_p50_qc": distance_p50_qc,
        "distance_p95_qc": distance_p95_qc,
        "distance_max_qc": distance_max_qc,
        "frac_near_maxdist_qc": frac_near_maxdist,
        "per_src_deg_mean_qc": per_src_deg_mean,
        "per_src_deg_p50_qc": per_src_deg_p50,
        "per_src_deg_p95_qc": per_src_deg_p95,
        "frac_src_deg_eq_topk_qc": frac_src_deg_eq_topk,
        "frac_src_deg_lt_topk_qc": frac_src_deg_lt_topk,
        "rel_type_mode_qc": rel_type_mode,
        "rel_type_purity_qc": rel_type_purity,
        "w0_valid_frac_qc": w0_valid_frac,
        "tau_hat_qc": float(tau_hat) if tau_hat is not None else None,
        "tau_hat_status_qc": tau_status,
    }

# =========================
# MAIN: run all datasets
# =========================
datasets = sorted([d.name for d in ROOT.iterdir() if d.is_dir()])
datasets = [d for d in datasets if not BAD_DS_RE.match(d)]
datasets = [d for d in datasets if d not in EXCLUDE]

qc_rows = []
contig_rows = []

for ds in datasets:
    dataset_root = ROOT / ds
    assets_dir, pick_mode = pick_assets_dir(dataset_root)
    if assets_dir is None:
        print(f"[skip_missing_assets] {ds}")
        continue

    peaks_in = assets_dir / "peaks.bed"
    genes_tsv = assets_dir / "genes.tsv"
    node_index_json = assets_dir / "node_index.json"
    if not (peaks_in.exists() and genes_tsv.exists() and node_index_json.exists()):
        print(f"[skip_missing_files] {ds}")
        continue

    # 1) contig stats (based on original peaks)
    #    + write canonical peaks into assets/{NEW_TAG}/peaks.canon.bed
    canon_peaks = assets_dir / NEW_OUT_TAG / "peaks.canon.bed"
    contig_stats = write_canonical_peaks(peaks_in, canon_peaks)
    contig_stats.update({"dataset": ds, "assets_dir": str(assets_dir), "pick_mode": pick_mode, "canon_peaks_path": str(canon_peaks)})
    contig_rows.append(contig_stats)

    # 2) build prior using canonical peaks (NEW_TAG)
    rep = build_prior_one(ds, assets_dir, canon_peaks, NEW_OUT_TAG)

    if rep.get("status") != "ok":
        print(f"[{rep.get('status')}] {ds} (build failed)")
        # 仍写一条 qc，方便汇总定位
        qc_rows.append({
            "dataset": ds, "status": rep.get("status"), "alerts": "BUILD_FAIL",
            "assets_dir": str(assets_dir), "out_dir": str(assets_dir / NEW_OUT_TAG),
            **contig_stats
        })
        continue

    # 3) qc
    out_dir = assets_dir / NEW_OUT_TAG
    qc_row = qc_from_outputs(ds, out_dir, rep, contig_stats)
    qc_rows.append(qc_row)

    print(f"[{qc_row['status']}] {ds} noncanon={qc_row.get('noncanon_frac')} edge_cov={qc_row.get('peak_edge_coverage_report')} alerts={qc_row.get('alerts')}")

# =========================
# SAVE outputs
# =========================
qc_df = pd.DataFrame(qc_rows)
contig_df = pd.DataFrame(contig_rows)

# 排序：先 warn，再 ok
status_rank = {"warn": 0, "ok": 1}
qc_df["_rank"] = qc_df["status"].map(lambda x: status_rank.get(str(x), 9))
qc_df = qc_df.sort_values(
    ["_rank", "peak_edge_coverage_report", "peak_src_coverage_report", "noncanon_frac", "dataset"],
    ascending=[True, True, True, False, True],
    na_position="last"
).drop(columns=["_rank"])

qc_tsv = QC_OUTDIR / "qc_summary.tsv"
qc_xlsx = QC_OUTDIR / "qc_summary.xlsx"
contig_tsv = QC_OUTDIR / "contig_stats.tsv"

qc_df.to_csv(qc_tsv, sep="\t", index=False)
contig_df.to_csv(contig_tsv, sep="\t", index=False)
try:
    qc_df.to_excel(qc_xlsx, index=False)
except Exception as e:
    print("[WARN] to_excel failed:", e)

print("\n[DONE]")
print("QC summary TSV :", qc_tsv)
print("QC summary XLSX:", qc_xlsx)
print("Contig TSV     :", contig_tsv)

# 可选：快速打印 non-canon 占比 top
if len(contig_df):
    show = contig_df.sort_values("noncanon_frac", ascending=False).head(15)[
        ["dataset", "peaks_in", "peaks_noncanon", "noncanon_frac", "chr_unique_total", "top_noncanon_chr"]
    ]
    print("\n[TOP non-canonical fraction]")
    print(show.to_string(index=False))
