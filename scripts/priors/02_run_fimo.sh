# Source: 诛仙剑阵2服务器Fimo构建(3).docx
#!/usr/bin/env bash
set -euo pipefail
trap 'echo -e "\n[FATAL] line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

# -------------------- params --------------------
N_PART="${N_PART:-16}"         # 外层 16 份
N_SUB="${N_SUB:-128}"          # 每份最多再拆 128
N_PROC="${N_PROC:-128}"        # 并发跑 fimo
THRESH="${THRESH:-1e-5}"
COMPRESS="${COMPRESS:-1}"      # 1: 输出 .tsv.gz  0: 输出 .tsv

# 避免 fimo/blas 自己开多线程导致爆线程
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE="/data/Core-LLM-Frag"
WORLD="${BASE}/LLM-WorldWord"
HG38_FA="${BASE}/refdata-cellranger-arc-GRCh38-2024-A/refdata-cellranger-arc-GRCh38-2024-A/fasta/genome.fa"
MOTIFS="${BASE}/ref/motifs.meme"
CHAIN="${BASE}/ref/hg19ToHg38.over.chain.gz"

# -------------------- checks --------------------
for c in bedtools fimo python3 awk find sort uniq gzip; do
  command -v "$c" >/dev/null || { echo "[FATAL] missing cmd: $c"; exit 1; }
done

CROSSMAP=""
if command -v CrossMap.py >/dev/null 2>&1; then CROSSMAP="CrossMap.py"; fi
if [[ -z "$CROSSMAP" ]] && command -v CrossMap >/dev/null 2>&1; then CROSSMAP="CrossMap"; fi

[[ -f "$HG38_FA" ]] || { echo "[FATAL] missing fasta: $HG38_FA"; exit 1; }
[[ -f "${HG38_FA}.fai" ]] || { echo "[FATAL] missing fai: ${HG38_FA}.fai (run: samtools faidx $HG38_FA)"; exit 1; }
[[ -f "$MOTIFS" ]] || { echo "[FATAL] missing motifs: $MOTIFS"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
WORKROOT="/data/tmp/fimo_fullrun_16x128_CANON_${TS}"
mkdir -p "$WORKROOT"
SUMMARY="${WORKROOT}/summary.tsv"
INDEX="${WORKROOT}/assets_peaks_index.tsv"
printf "dataset\tpeaks_path\tneed_liftover\tclean_lines\thg38_canon_lines\tverify_status\tfimo_out\n" > "$SUMMARY"

echo "[INFO] workroot=$WORKROOT"
echo "[INFO] summary=$SUMMARY"
echo "[INFO] index=$INDEX"
echo "[INFO] params: N_PART=$N_PART N_SUB=$N_SUB N_PROC=$N_PROC THRESH=$THRESH COMPRESS=$COMPRESS"
echo "[INFO] CrossMap entry: ${CROSSMAP:-NONE}"

# -------------------- helpers --------------------

# 0) 建索引：扫描 WORLD 下所有 assets 目录里的 peaks.bed
index_assets() {
  echo "[STEP] indexing peaks.bed under: $WORLD (this may take a bit) ..."
  python3 - "$WORLD" "$INDEX" <<'PY'
import os, sys
root, outp = sys.argv[1], sys.argv[2]
rows=[]
for dirpath, dirnames, filenames in os.walk(root):
    if os.path.basename(dirpath) != "assets":
        continue
    peaks = os.path.join(dirpath, "peaks.bed")
    if os.path.isfile(peaks):
        ds = os.path.basename(os.path.dirname(dirpath))  # parent of assets
        rows.append((ds, dirpath, peaks))
rows = sorted(set(rows))
with open(outp, "w") as out:
    out.write("dataset\tassets_dir\tpeaks_path\tfiletype\n")
    for ds, a, p in rows:
        out.write(f"{ds}\t{a}\t{p}\tbed\n")
print(len(rows))
PY
}

# 1) 清洗 bed：兼容空格/Tab，去 CRLF，跳过 track/browser/#，只输出 BED3
clean_bed_py() {
  local inbed="$1" outbed="$2"
  python3 - "$inbed" "$outbed" <<'PY'
import sys
inp, outp = sys.argv[1], sys.argv[2]
n_ok = 0
with open(inp, "r", errors="ignore") as f, open(outp, "w") as out:
    for line in f:
        line = line.rstrip("\n").rstrip("\r")
        if not line: 
            continue
        if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        chrom = parts[0]
        try:
            s = int(parts[1]); e = int(parts[2])
        except:
            continue
        if s < 0 or e <= s:
            continue
        out.write(f"{chrom}\t{s}\t{e}\n")
        n_ok += 1
print(n_ok)
PY
}

# 2) 用 hg38 fai 过滤：只保留 fasta 存在的 contig
filter_by_fai() {
  local inbed="$1" outbed="$2"
  awk 'NR==FNR{a[$1]=1; next} ($1 in a) && $2>=0 && $3>$2 {print $1"\t"$2"\t"$3}' \
    "${HG38_FA}.fai" "$inbed" > "$outbed"
}

# 3) 丢掉非 canonical：只留 chr1-22,X,Y,M
filter_canon() {
  local inbed="$1" outbed="$2"
  awk '$1 ~ /^chr([0-9]+|X|Y|M)$/ {print $0}' "$inbed" > "$outbed"
}

# 4) 判定是否需要 liftover：看 end 是否超过 hg38 染色体长度（抽 20w 行）
detect_need_liftover_py() {
  local bed3="$1"
  python3 - "$bed3" "${HG38_FA}.fai" <<'PY'
import sys
bed, fai = sys.argv[1], sys.argv[2]
lens={}
with open(fai) as f:
    for line in f:
        c, L = line.split("\t", 2)[:2]
        lens[c]=int(L)

tot=0
in_fai=0
beyond=0
miss=0

with open(bed) as f:
    for line in f:
        if not line.strip():
            continue
        chrom,s,e = line.rstrip("\n").split("\t")[:3]
        tot += 1
        s=int(s); e=int(e)
        if chrom in lens:
            in_fai += 1
            if e > lens[chrom]:
                beyond += 1
        else:
            miss += 1
        if tot >= 200000:
            break

need = False
if beyond >= 10:
    need = True
elif in_fai >= 1000 and (beyond / max(1,in_fai)) > 0.001:
    need = True

print(f"tot={tot} in_fai={in_fai} miss_chr={miss} beyond_len={beyond} need={('YES' if need else 'NO')}")
sys.exit(0 if need else 1)
PY
}

# 5) 给 BED3 补 safeID 第4列
add_safe_id() {
  local inbed="$1" outbed="$2"
  awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$1"_"$2"_"$3}' "$inbed" > "$outbed"
}

# 6) 按“行数”把 BED3 拆成 N 份（输出路径列表）
split_bed_py() {
  local inbed="$1" outdir="$2" nparts="$3" prefix="$4"
  python3 - "$inbed" "$outdir" "$nparts" "$prefix" <<'PY'
import sys, os, math
inbed, outdir, nparts, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)

# count lines
n=0
with open(inbed) as f:
    for _ in f: n += 1
if n==0: sys.exit(2)

nparts = min(nparts, n)
per = max(1, math.ceil(n / nparts))

paths=[]
idx=0
count=0
out_path=os.path.join(outdir, f"{prefix}_{idx:03d}.bed")
out=open(out_path,"w")
paths.append(out_path)

with open(inbed) as f:
    for line in f:
        if count>=per and idx < nparts-1:
            out.close()
            idx+=1
            count=0
            out_path=os.path.join(outdir, f"{prefix}_{idx:03d}.bed")
            out=open(out_path,"w")
            paths.append(out_path)
        out.write(line)
        count += 1
out.close()

for p in paths: print(p)
PY
}

# 7) 合并多个 fimo text tsv：comments/header 只写一次
merge_fimo_py() {
  python3 - "$@" <<'PY'
import sys, os
ins = sys.argv[1:-1]
out_path = sys.argv[-1]

def first_header_and_comments(p):
    comments=[]
    header=None
    with open(p) as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line)
                continue
            header=line
            break
    return comments, header

wrote_c=False
wrote_h=False
with open(out_path, "w") as out:
    for p in ins:
        if not os.path.exists(p) or os.path.getsize(p)==0:
            continue
        comments, header = first_header_and_comments(p)
        if header is None:
            continue
        if not wrote_c:
            out.writelines(comments); wrote_c=True
        if not wrote_h:
            out.write(header); wrote_h=True
        with open(p) as f:
            for line in f:
                if line.startswith("#"): 
                    continue
                if wrote_h and line == header:
                    continue
                if not line.strip():
                    continue
                out.write(line)
print(out_path)
PY
}

# 8) 强检查：sequence_name 必须是 safeID（chr*_start_end），不能退化成 chr-only
verify_full_tsv() {
  local tsv="$1" report="$2"

  # 支持 .gz
  local reader="cat"
  if [[ "$tsv" == *.gz ]]; then reader="gzip -cd"; fi

  # 统计前 2000 行：safeID比例、chr-only比例、unique(前1M)用于防退化
  bash -lc "$reader \"$tsv\" | awk -F'\t' -v MAX=1000000 -v H=2000 '
    BEGIN{n=0; nunq=0; headn=0; chr_only=0; safeid=0}
    /^#/ {next}
    \$1==\"motif_id\" {next}
    {
      n++
      sn=\$3
      if (!(sn in seen)) { seen[sn]=1; nunq++ }
      if (headn < H) {
        headn++
        if (sn ~ /^chr([0-9]+|X|Y|M)$/) chr_only++
        if (sn ~ /^chr([0-9]+|X|Y|M)_[0-9]+_[0-9]+$/) safeid++
      }
      if (n>=MAX) exit
    }
    END{
      chr_ratio = (headn>0)? chr_only/headn : 0
      safe_ratio = (headn>0)? safeid/headn : 0
      printf(\"nunique_1M=%d chr_only_ratio_head2000=%.6f safeid_ratio_head2000=%.6f\\n\", nunq, chr_ratio, safe_ratio)
      # PASS 条件：chr-only 比例必须为0；safeid比例必须接近1（>=0.99）
      if (chr_ratio==0 && safe_ratio>=0.99) exit 0
      exit 1
    }
  ' > \"$report\""
}

# -------------------- run --------------------

# 0) index
idx_n="$(index_assets)"
echo "[INFO] indexed_datasets=$idx_n"

# 1) 读索引到 map（dataset -> peaks_path）
declare -A PEAK_MAP
while IFS=$'\t' read -r ds assets peaks ftype; do
  [[ "$ds" == "dataset" ]] && continue
  PEAK_MAP["$ds"]="$peaks"
done < "$INDEX"

# 2) datasets list：如果传参就跑传参；不传参就跑索引里全部
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  mapfile -t DATASETS < <(cut -f1 "$INDEX" | tail -n +2)
fi

for ds in "${DATASETS[@]}"; do
  echo "=============================="
  echo "[DATASET] $ds"

  PEAKS="${PEAK_MAP[$ds]:-}"
  if [[ -z "$PEAKS" || ! -f "$PEAKS" ]]; then
    echo "[MISS] peaks not found for $ds"
    printf "%s\tNA\tNA\t0\t0\tFAIL(no_peaks)\tNA\n" "$ds" >> "$SUMMARY"
    continue
  fi
  echo "[INFO] peaks=$PEAKS"

  WDIR="${WORKROOT}/${ds}"
  mkdir -p "$WDIR"/{tmp,logs,bed_parts1,bed_parts2,fimo_txt,merged,final}

  # A) clean
  echo "[STEP] clean peaks ..."
  CLEAN="$WDIR/tmp/peaks.clean.bed"
  clean_lines="$(clean_bed_py "$PEAKS" "$CLEAN")"
  echo "[INFO] clean_lines=$clean_lines"
  if [[ "$clean_lines" -eq 0 ]]; then
    printf "%s\t%s\tNA\t0\t0\tFAIL(clean_zero)\tNA\n" "$ds" "$PEAKS" >> "$SUMMARY"
    continue
  fi

  # B) detect liftover
  echo "[STEP] detect hg19/hg38 ..."
  need_liftover="NO"
  if detect_need_liftover_py "$CLEAN" > "$WDIR/logs/detect.log" 2>&1; then
    need_liftover="YES"
  fi
  echo "[INFO] $(cat "$WDIR/logs/detect.log")"
  echo "[INFO] need_liftover=$need_liftover"

  # C) to hg38 + filter by fai
  HG38_FILT="$WDIR/tmp/peaks.hg38.filt.bed"
  if [[ "$need_liftover" == "YES" ]]; then
    [[ -n "$CROSSMAP" ]] || { echo "[FAIL] need liftover but no CrossMap"; printf "%s\t%s\tYES\t%s\t0\tFAIL(no_crossmap)\tNA\n" "$ds" "$PEAKS" "$clean_lines" >> "$SUMMARY"; continue; }
    [[ -f "$CHAIN" ]] || { echo "[FAIL] missing chain: $CHAIN"; printf "%s\t%s\tYES\t%s\t0\tFAIL(no_chain)\tNA\n" "$ds" "$PEAKS" "$clean_lines" >> "$SUMMARY"; continue; }

    echo "[STEP] CrossMap hg19->hg38 ..."
    RAW="$WDIR/tmp/peaks.hg38.raw.bed"
    "$CROSSMAP" bed "$CHAIN" "$CLEAN" "$RAW" > "$WDIR/logs/crossmap.log" 2>&1 || true
    filter_by_fai "$RAW" "$HG38_FILT"
  else
    filter_by_fai "$CLEAN" "$HG38_FILT"
  fi

  # D) drop non-canonical contigs
  HG38_CANON="$WDIR/tmp/peaks.hg38.canon.bed"
  filter_canon "$HG38_FILT" "$HG38_CANON"
  canon_lines="$(wc -l < "$HG38_CANON" | tr -d ' ')"
  echo "[INFO] hg38_canon_lines=$canon_lines"
  if [[ "$canon_lines" -lt 10 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tFAIL(too_few_canon)\tNA\n" "$ds" "$PEAKS" "$need_liftover" "$clean_lines" "$canon_lines" >> "$SUMMARY"
    continue
  fi

  # E) split BED into N_PART parts
  echo "[STEP] split BED L1 (N_PART=$N_PART) ..."
  mapfile -t PARTBEDS < <(split_bed_py "$HG38_CANON" "$WDIR/bed_parts1" "$N_PART" "part")

  # export to xargs subprocs
  export HG38_FA MOTIFS THRESH WDIR COMPRESS

  # F) for each part -> split into sub beds -> bedtools getfasta -nameOnly -> fimo --text -> part merge
  pidx=0
  for pbed in "${PARTBEDS[@]}"; do
    pidx=$((pidx+1))
    pnm="$(basename "$pbed" .bed)"
    echo "[PART] $pnm ($pidx/${#PARTBEDS[@]}) split L2 (N_SUB up to $N_SUB) ..."

    # decide subN (<= N_SUB and <= numlines)
    plines="$(wc -l < "$pbed" | tr -d ' ')"
    subN="$N_SUB"; [[ "$plines" -lt "$subN" ]] && subN="$plines"
    subdir="$WDIR/bed_parts2/$pnm"
    mkdir -p "$subdir"

    mapfile -t SUBBEDS < <(split_bed_py "$pbed" "$subdir" "$subN" "${pnm}.sub")

    echo "[PART] $pnm -> run fimo --text with -P $N_PROC (jobs=$subN)"
    outdir="$WDIR/fimo_txt/$pnm"
    mkdir -p "$outdir"

    # 每个 sub：生成 safe bed4 -> getfasta -nameOnly -> fimo --text -> (可选 gzip) -> 删除 fasta/bed4
    printf "%s\n" "${SUBBEDS[@]}" | xargs -P "$N_PROC" -I{} bash -lc '
      set -euo pipefail
      bed="{}"
      b=$(basename "$bed" .bed)
      bed4="$WDIR/tmp/${b}.safe.bed"
      fa="$WDIR/tmp/${b}.fa"
      tsv="$WDIR/fimo_txt/'"$pnm"'/${b}.tsv"

      awk '"'"'BEGIN{OFS="\t"}{print $1,$2,$3,$1"_"$2"_"$3}'"'"' "$bed" > "$bed4"
      bedtools getfasta -fi "$HG38_FA" -bed "$bed4" -fo "$fa" -nameOnly > "$WDIR/logs/getfasta_'$pnm'_${b}.log" 2>&1
      fimo --text --thresh "$THRESH" "$MOTIFS" "$fa" > "$tsv" 2> "$WDIR/logs/fimo_'$pnm'_${b}.stderr.log" || true

      rm -f "$fa" "$bed4" || true

      if [[ "$COMPRESS" == "1" ]]; then
        gzip -1 -f "$tsv"
      fi
    '

    # merge sub tsvs -> one part tsv
    echo "[PART] $pnm merge ..."
    if [[ "$COMPRESS" == "1" ]]; then
      mapfile -t TLIST < <(ls -1 "$WDIR/fimo_txt/$pnm"/*.tsv.gz 2>/dev/null | sort || true)
      [[ "${#TLIST[@]}" -gt 0 ]] || { echo "[WARN] $pnm no tsv.gz"; continue; }
      # 解压合并：先解到临时再合（避免 python 读 gzip）
      tmpmerge="$WDIR/tmp/${pnm}.merge.list"
      rm -f "$tmpmerge"
      for g in "${TLIST[@]}"; do
        u="${g%.gz}"
        gzip -cd "$g" > "$u"
        echo "$u" >> "$tmpmerge"
      done
      merge_fimo_py $(cat "$tmpmerge") "$WDIR/merged/${pnm}.fimo.tsv" >/dev/null
      # 清理解压文件
      while read -r u; do rm -f "$u"; done < "$tmpmerge"
      rm -f "$tmpmerge"
    else
      mapfile -t TLIST < <(ls -1 "$WDIR/fimo_txt/$pnm"/*.tsv 2>/dev/null | sort || true)
      [[ "${#TLIST[@]}" -gt 0 ]] || { echo "[WARN] $pnm no tsv"; continue; }
      merge_fimo_py "${TLIST[@]}" "$WDIR/merged/${pnm}.fimo.tsv" >/dev/null
    fi

    # 可选：清理子输出（省空间）
    rm -rf "$WDIR/fimo_txt/$pnm" "$WDIR/bed_parts2/$pnm" || true
  done

  # G) merge parts -> final
  mapfile -t PM < <(ls -1 "$WDIR/merged"/part_*.fimo.tsv 2>/dev/null | sort || true)
  if [[ "${#PM[@]}" -lt 1 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\tFAIL(no_parts)\tNA\n" "$ds" "$PEAKS" "$need_liftover" "$clean_lines" "$canon_lines" >> "$SUMMARY"
    continue
  fi

  FINAL="$WDIR/final/fimo.full.tsv"
  merge_fimo_py "${PM[@]}" "$FINAL" >/dev/null

  if [[ "$COMPRESS" == "1" ]]; then
    gzip -1 -f "$FINAL"
    FINAL="${FINAL}.gz"
  fi

  # H) verify (防 chr-only 退化 + 防 header 不一致)
  echo "[STEP] verify full ..."
  report="$WDIR/final/verify.txt"
  if verify_full_tsv "$FINAL" "$report"; then
    echo "[OK] $ds fullrun PASS -> $FINAL"
    status="PASS"
  else
    echo "[BAD] $ds fullrun FAIL(header) -> $FINAL"
    status="FAIL(header)"
  fi
  echo "[VERIFY] $(cat "$report")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$ds" "$PEAKS" "$need_liftover" "$clean_lines" "$canon_lines" "$status" "$FINAL" >> "$SUMMARY"
done

echo "=============================="
echo "[DONE] summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

# 如果全 PASS，给一句提示
if awk 'NR>1 && $6!="PASS"{bad=1} END{exit(bad?1:0)}' "$SUMMARY"; then
  echo "[OK] All datasets PASS. Outputs under: $WORKROOT"
else
  echo "[FAIL] Some datasets failed. Check per-dataset logs under: $WORKROOT/<ds>/logs"
fi
