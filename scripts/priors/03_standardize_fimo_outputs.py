# Source: 诛仙剑阵2服务器Fimo构建后格式统一(1).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os
import re
import gzip
from pathlib import Path

WORLD = str(get_world_root())
OUT_REPORT = os.path.join(WORLD, "_fimo_unify_report.tsv")

# 你可以改：是否覆盖已经存在的 dataset/fimo.tsv
OVERWRITE = True

# 选择优先级（从高到低）
CANDIDATES = [
    ("final_full_gz",   r"final/fimo.full.tsv.gz"),
    ("final_full",      r"final/fimo.full.tsv"),
    ("root_full_gz",    r"fimo.full.tsv.gz"),
    ("root_full",       r"fimo.full.tsv"),
    ("final_plain_gz",  r"final/fimo.tsv.gz"),
    ("final_plain",     r"final/fimo.tsv"),
    ("root_plain_gz",   r"fimo.tsv.gz"),
    ("root_plain",      r"fimo.tsv"),
]

def is_gzip_file(p: Path) -> bool:
    """用 magic bytes 判断是否 gzip，避免 .gz 伪装"""
    try:
        with open(p, "rb") as f:
            head = f.read(2)
        return head == b"\x1f\x8b"
    except Exception:
        return False

def open_maybe_gzip(p: Path):
    """返回一个可迭代的 text 行流（utf-8 容错）"""
    if is_gzip_file(p):
        return gzip.open(p, "rt", encoding="utf-8", errors="ignore", newline="")
    return open(p, "rt", encoding="utf-8", errors="ignore", newline="")

def pick_best_fimo(ds_dir: Path):
    found = []
    for tag, rel in CANDIDATES:
        p = ds_dir / rel
        if p.is_file() and p.stat().st_size > 0:
            found.append((tag, p))
    return found[0] if found else (None, None)

def count_hits_quick(p: Path, max_lines=200000):
    """
    统计 hit 行数（非注释且非表头 motif_id 开头）
    为了快，只扫前 max_lines 行；如果你想全量计数可以改成 None（会很慢）
    """
    hits = 0
    scanned = 0
    with open_maybe_gzip(p) as f:
        for line in f:
            scanned += 1
            if max_lines and scanned > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("motif_id\t") or line == "motif_id":
                continue
            hits += 1
    return hits, scanned

def write_plain_tsv(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tsv.tmp")
    written = 0
    with open_maybe_gzip(src) as fin, open(tmp, "wt", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            # 统一换行 / 去掉 \r
            if line.endswith("\r\n"):
                line = line[:-2] + "\n"
            else:
                line = line.replace("\r", "")
            fout.write(line)
            written += 1
    os.replace(tmp, dst)
    return written

def main():
    world = Path(WORLD)
    rows = []

    ds_dirs = [p for p in world.iterdir() if p.is_dir() and not p.name.startswith("_")]
    ds_dirs.sort(key=lambda x: x.name.lower())

    for ds_dir in ds_dirs:
        ds = ds_dir.name
        dst = ds_dir / "fimo.tsv"

        if dst.exists() and (not OVERWRITE):
            rows.append({
                "dataset": ds,
                "status": "SKIP_EXISTS",
                "chosen_tag": "",
                "src": "",
                "dst": str(dst),
                "dst_size": dst.stat().st_size,
                "hits_head": "",
                "note": "keep existing",
            })
            continue

        tag, src = pick_best_fimo(ds_dir)
        if src is None:
            rows.append({
                "dataset": ds,
                "status": "MISSING",
                "chosen_tag": "",
                "src": "",
                "dst": str(dst),
                "dst_size": "",
                "hits_head": "",
                "note": "no fimo found",
            })
            continue

        try:
            hits, scanned = count_hits_quick(src, max_lines=200000)
            written_lines = write_plain_tsv(src, dst)
            rows.append({
                "dataset": ds,
                "status": "OK",
                "chosen_tag": tag,
                "src": str(src),
                "dst": str(dst),
                "dst_size": dst.stat().st_size,
                "hits_head": hits,
                "note": f"scanned_head_lines={scanned}, written_lines={written_lines}",
            })
            print(f"[OK] {ds:<28} {tag:<14} -> fimo.tsv  size={dst.stat().st_size}  hits(head)={hits}")
        except Exception as e:
            rows.append({
                "dataset": ds,
                "status": "ERROR",
                "chosen_tag": tag,
                "src": str(src),
                "dst": str(dst),
                "dst_size": "",
                "hits_head": "",
                "note": repr(e),
            })
            print(f"[ERR] {ds} {src} :: {e}")

    # 写报告
    import pandas as pd
    df = pd.DataFrame(rows, columns=["dataset","status","chosen_tag","src","dst","dst_size","hits_head","note"])
    df.to_csv(OUT_REPORT, sep="\t", index=False)
    print("\n[DONE]")
    print("WORLD     :", WORLD)
    print("REPORT TSV:", OUT_REPORT)
    print(df["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
