# Source: 先验组装后最终词典QC(5).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, gzip, math, random
from pathlib import Path
from collections import Counter
import pandas as pd

# =========================
# 0) 自动定位 OUT_ROOT
# =========================
random.seed(0)

# Word 里写死的默认 BASE/OUT_ROOT（Windows）
BASE_WIN = get_project_root()
OUT_ROOT = BASE_WIN / "networks" / "pan_priors_assembled_fast"

def find_out_root_fallback():
    """
    如果默认路径不存在：
    1) 先从当前工作目录开始往下找名为 pan_priors_assembled_fast 的目录
    2) 再从当前工作目录的父目录找（更广一些）
    """
    targets = []
    cwd = Path.cwd()

    # 先在 cwd 下找
    targets += list(cwd.rglob("pan_priors_assembled_fast"))
    # 再在父目录下找（避免你在子目录里启动）
    if cwd.parent != cwd:
        targets += list(cwd.parent.rglob("pan_priors_assembled_fast"))

    # 只保留目录
    targets = [p for p in targets if p.is_dir()]
    # 更像我们需要的：包含 per_relation/ 或 manifest.tsv
    scored = []
    for p in targets:
        score = 0
        if (p / "per_relation").exists():
            score += 10
        if (p / "manifest.tsv").exists():
            score += 10
        scored.append((score, p))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in scored]

if not OUT_ROOT.exists():
    cands = find_out_root_fallback()
    if not cands:
        raise FileNotFoundError(
            "找不到 OUT_ROOT。\n"
            "你 Word 里默认是：G:\\Project\\Core-LLM-Frag\\networks\\pan_priors_assembled_fast\n"
            "但你当前环境里没有这个路径。\n"
            "请把 OUT_ROOT 改成你机器上的真实输出目录，或把该目录放到当前工程下再跑。"
        )
    OUT_ROOT = cands[0]

print(f"[INFO] Using OUT_ROOT = {OUT_ROOT}")

# =========================
# 1) QC 参数
# =========================
SAMPLE_PER_FILE = 200_000
MAX_PROBLEMS_PRINT = 200

REQ_COLS = ["dataset","split","direction","src_name","dst_name","rel_type","w0"]
ALLOWED_DIR = {"fwd","rev"}
ALLOWED_SPLIT = {"train","explain"}

def open_maybe_gz(p: Path, mode="rt"):
    if str(p).endswith(".gz"):
        return gzip.open(p, mode)
    return open(p, mode, encoding="utf-8", errors="replace")

def read_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            r = {header[i]: (parts[i] if i < len(parts) else "") for i in range(len(header))}
            rows.append(r)
        return rows

def read_header(p: Path):
    with open_maybe_gz(p, "rt") as f:
        line = f.readline()
    if not line:
        return None
    return line.rstrip("\n").split("\t")

def iter_rows_sampled(p: Path, sample_n: int):
    with open_maybe_gz(p, "rt") as f:
        header = f.readline()
        if not header:
            return None, []
        cols = header.rstrip("\n").split("\t")
        buf, seen = [], 0
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            seen += 1
            if len(buf) < sample_n:
                buf.append(line)
            else:
                j = random.randint(1, seen)
                if j <= sample_n:
                    buf[j-1] = line
        return cols, buf

def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None

def pct_quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = int(q * (len(sorted_vals)-1))
    return sorted_vals[k]

def qc_file(p: Path, sample_n: int):
    res = {
        "file": str(p),
        "exists": p.exists(),
        "header_ok": False,
        "missing_cols": "",
        "extra_cols": "",
        "n_sample": 0,
        "empty_src_pct": None,
        "empty_dst_pct": None,
        "bad_w0_pct": None,
        "w0_min": None,
        "w0_p50": None,
        "w0_p99": None,
        "w0_max": None,
        "dup_edge_pct": None,
        "split_values_top": "",
        "direction_values_top": "",
        "dataset_values_top": "",
    }
    if not p.exists():
        return res

    cols = read_header(p)
    if cols is None:
        res["missing_cols"] = ",".join(REQ_COLS)
        return res

    colset = set(cols)
    missing = [c for c in REQ_COLS if c not in colset]
    extra = [c for c in cols if c not in REQ_COLS]
    res["missing_cols"] = ",".join(missing)
    res["extra_cols"] = ",".join(extra)
    res["header_ok"] = (len(missing) == 0)

    cols2, lines = iter_rows_sampled(p, sample_n)
    if cols2 is None:
        return res

    idx = {c:i for i,c in enumerate(cols2)}
    if not all(c in idx for c in ["src_name","dst_name","w0","split","direction","dataset"]):
        return res

    n = len(lines)
    res["n_sample"] = n
    if n == 0:
        return res

    empty_src = empty_dst = bad_w0 = 0
    w0s = []
    edge_counter = Counter()
    split_vals = Counter()
    dir_vals = Counter()
    ds_vals = Counter()

    for line in lines:
        parts = line.split("\t")
        def get(c):
            i = idx.get(c, None)
            if i is None or i >= len(parts):
                return ""
            return parts[i]

        src = get("src_name").strip()
        dst = get("dst_name").strip()
        if src == "":
            empty_src += 1
        if dst == "":
            empty_dst += 1

        w = safe_float(get("w0").strip())
        if w is None:
            bad_w0 += 1
        else:
            w0s.append(w)

        edge_counter[(src, dst)] += 1
        split_vals[get("split").strip()] += 1
        dir_vals[get("direction").strip()] += 1
        ds_vals[get("dataset").strip()] += 1

    res["empty_src_pct"] = empty_src / n
    res["empty_dst_pct"] = empty_dst / n
    res["bad_w0_pct"] = bad_w0 / n

    if w0s:
        w0s.sort()
        res["w0_min"] = w0s[0]
        res["w0_p50"] = pct_quantile(w0s, 0.50)
        res["w0_p99"] = pct_quantile(w0s, 0.99)
        res["w0_max"] = w0s[-1]

    dup_keys = sum(1 for _,c in edge_counter.items() if c > 1)
    res["dup_edge_pct"] = (dup_keys / len(edge_counter)) if edge_counter else 0.0

    res["split_values_top"] = ",".join([f"{k}:{v}" for k,v in split_vals.most_common(5)])
    res["direction_values_top"] = ",".join([f"{k}:{v}" for k,v in dir_vals.most_common(5)])
    res["dataset_values_top"] = ",".join([f"{k}:{v}" for k,v in ds_vals.most_common(5)])

    return res

# =========================
# 2) 收集文件列表 & 跑 QC
# =========================
manifest_path = OUT_ROOT / "manifest.tsv"
per_rel_root = OUT_ROOT / "per_relation"

problems = []
report_rows = []

manifest_rows = read_manifest(manifest_path)
files = []
if manifest_rows and ("path" in manifest_rows[0]):
    files = [OUT_ROOT / r["path"] for r in manifest_rows]
else:
    files = list(per_rel_root.rglob("*.tsv")) + list(per_rel_root.rglob("*.tsv.gz"))

files = sorted(set(files))

print(f"[INFO] manifest exists: {manifest_path.exists()} | per_relation exists: {per_rel_root.exists()}")
print(f"[INFO] files to QC: {len(files)}")

if not manifest_path.exists():
    problems.append(f"[MISS] manifest.tsv not found at {manifest_path}")
elif not manifest_rows:
    problems.append(f"[BAD] manifest.tsv empty or unreadable: {manifest_path}")

for p in files:
    res = qc_file(p, SAMPLE_PER_FILE)
    report_rows.append(res)

    if not res["exists"]:
        problems.append(f"[MISS] {p}")
        continue
    if not res["header_ok"]:
        problems.append(f"[SCHEMA] {p} missing={res['missing_cols']} extra={res['extra_cols']}")
    if res["n_sample"] == 0:
        problems.append(f"[EMPTY] {p}")

    if res["bad_w0_pct"] is not None and res["bad_w0_pct"] > 0.001:
        problems.append(f"[W0] {p} bad_w0_pct={res['bad_w0_pct']:.4f}")
    if res["empty_src_pct"] is not None and res["empty_src_pct"] > 0.001:
        problems.append(f"[SRC] {p} empty_src_pct={res['empty_src_pct']:.4f}")
    if res["empty_dst_pct"] is not None and res["empty_dst_pct"] > 0.001:
        problems.append(f"[DST] {p} empty_dst_pct={res['empty_dst_pct']:.4f}")
    if res["dup_edge_pct"] is not None and res["dup_edge_pct"] > 0.05:
        problems.append(f"[DUP] {p} dup_edge_pct={res['dup_edge_pct']:.3f}")

    if res["direction_values_top"]:
        dirs = {kv.split(":")[0] for kv in res["direction_values_top"].split(",") if ":" in kv}
        if any(d not in ALLOWED_DIR and d != "" for d in dirs):
            problems.append(f"[DIRVAL] {p} {res['direction_values_top']}")
    if res["split_values_top"]:
        splits = {kv.split(":")[0] for kv in res["split_values_top"].split(",") if ":" in kv}
        if any(s not in ALLOWED_SPLIT and s != "" for s in splits):
            problems.append(f"[SPLITVAL] {p} {res['split_values_top']}")

df = pd.DataFrame(report_rows)

# =========================
# 3) 摘要展示
# =========================
print(df.head(10))
def top_by(col, n=15):
    d = df.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.sort_values(col, ascending=False).head(n)[
        ["file", col, "n_sample", "missing_cols", "extra_cols", "split_values_top", "direction_values_top"]
    ]

print("\n[SUMMARY] worst bad_w0_pct")
print(top_by("bad_w0_pct"))
print("\n[SUMMARY] worst empty_src_pct")
print(top_by("empty_src_pct"))
print("\n[SUMMARY] worst empty_dst_pct")
print(top_by("empty_dst_pct"))
print("\n[SUMMARY] worst dup_edge_pct")
print(top_by("dup_edge_pct"))
report_path = OUT_ROOT / "qc_report.tsv"
df.to_csv(report_path, sep="\t", index=False)
print(f"\n[REPORT] written: {report_path}")

if problems:
    print(f"\n[WARN] problems found: {len(problems)} (showing first {min(len(problems), MAX_PROBLEMS_PRINT)})")
    for x in problems[:MAX_PROBLEMS_PRINT]:
        print(" ", x)
else:
    print("\n[OK] no blocking problems detected")
