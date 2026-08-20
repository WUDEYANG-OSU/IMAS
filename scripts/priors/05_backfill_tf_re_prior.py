# Source: 诛仙剑阵2-F-改（解释）-Shrap-改-文件回填(5).docx
# Extracted from the user-provided Word code; scientific logic preserved.
import os, glob, shutil
from pathlib import Path

WORLD = str(get_world_root())

SRC_ROOT = str(get_project_root() / "networks_fixweight_20260228_v2_rank_mix_model")
SRC_TRAIN = os.path.join(SRC_ROOT, "tf_re_train")
SRC_EXPL  = os.path.join(SRC_ROOT, "tf_re_explain")

ASSET_SUBDIR = os.path.join("assets", "G0_full49_v1")

# demon风格目标文件名（你截图里叫 edges_tf_re_motif.tsv）
DST_TRAIN_NAME = "edges_tf_re_motif.tsv"
DST_EXPL_NAME  = "edges_tf_re_motif_explain.tsv"  # 避免覆盖；如需覆盖改成同名

def ds_from_edges_file(fp: str) -> str:
    # edges_<dataset>.tsv
    name = Path(fp).name
    if not (name.startswith("edges_") and name.endswith(".tsv")):
        raise ValueError(f"Unexpected file name: {name}")
    return name[len("edges_"):-len(".tsv")]

train_files = sorted(glob.glob(os.path.join(SRC_TRAIN, "edges_*.tsv")))
expl_files  = sorted(glob.glob(os.path.join(SRC_EXPL,  "edges_*.tsv")))

train_map = {ds_from_edges_file(f): f for f in train_files}
expl_map  = {ds_from_edges_file(f): f for f in expl_files}
all_ds = sorted(set(train_map) | set(expl_map))

copied_train = 0
copied_expl  = 0
missing_ds_dir = []
missing_asset_dir = []
missing_train = []
missing_expl = []

for ds in all_ds:
    ds_dir = os.path.join(WORLD, ds)
    if not os.path.isdir(ds_dir):
        missing_ds_dir.append(ds)
        continue

    asset_dir = os.path.join(ds_dir, ASSET_SUBDIR)
    if not os.path.isdir(asset_dir):
        missing_asset_dir.append(ds)
        continue

    # train -> edges_tf_re_motif.tsv
    if ds in train_map:
        dst = os.path.join(asset_dir, DST_TRAIN_NAME)
        shutil.copy2(train_map[ds], dst)
        copied_train += 1
    else:
        missing_train.append(ds)

    # explain -> edges_tf_re_motif_explain.tsv
    if ds in expl_map:
        dst = os.path.join(asset_dir, DST_EXPL_NAME)
        shutil.copy2(expl_map[ds], dst)
        copied_expl += 1
    else:
        missing_expl.append(ds)

print("[DONE] backfill to", ASSET_SUBDIR)
print("datasets_total:", len(all_ds))
print("copied_train :", copied_train)
print("copied_expl  :", copied_expl)
print("missing_dataset_dir :", len(missing_ds_dir))
print("missing_asset_dir   :", len(missing_asset_dir))
print("missing_train_edges :", len(missing_train))
print("missing_expl_edges  :", len(missing_expl))

if missing_asset_dir:
    print("\n[WARN] asset dir missing (first 20):", missing_asset_dir[:20])
if missing_ds_dir:
    print("\n[WARN] dataset dir missing (first 20):", missing_ds_dir[:20])
