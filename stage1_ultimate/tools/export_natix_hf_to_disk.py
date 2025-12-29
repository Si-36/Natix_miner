#!/usr/bin/env python3
"""
Export NATIX HuggingFace dataset to disk format for training pipeline

Exports:
- Images to /data/natix_hf_export/images/
- splits.json with train/val_select/val_calib/val_test splits
- Splits by video_id to avoid leakage (frames from same video stay together)
"""
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import json
import random
from collections import defaultdict
from tqdm import tqdm

# Output directory
OUT = Path("/data/natix_hf_export")
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset from HF cache...")
ds = load_dataset("natix-network-org/roadwork")

IMAGE_COL = "image"
LABEL_COL = "label"
GROUP_COL = "video_info.vid_id"  # Group by video to avoid leakage

def export_grouped(hf_split, prefix):
    """Export split grouped by video_id to avoid leakage"""
    print(f"\nProcessing {prefix} split...")

    # Group examples by video_id
    groups = defaultdict(list)
    for ex in tqdm(hf_split, desc=f"Grouping {prefix}"):
        vid_id = str(ex[GROUP_COL]) if ex[GROUP_COL] is not None else "unknown"
        groups[vid_id].append(ex)

    vid_ids = list(groups.keys())
    random.Random(42).shuffle(vid_ids)

    # Split: 80% train, 10% val_select, 10% val_calib
    n = len(vid_ids)
    n_train = int(0.80 * n)
    n_val_select = int(0.10 * n)

    train_vids = set(vid_ids[:n_train])
    val_select_vids = set(vid_ids[n_train:n_train + n_val_select])
    val_calib_vids = set(vid_ids[n_train + n_val_select:])

    def dump_examples(examples, name):
        """Save images and return metadata"""
        out_items = []
        for i, ex in enumerate(tqdm(examples, desc=f"Exporting {name}")):
            img = ex[IMAGE_COL]
            if not isinstance(img, Image.Image):
                img = Image.open(img) if hasattr(img, 'read') else Image.fromarray(img)

            label = int(ex[LABEL_COL])
            rel = f"images/{name}_{i:08d}.png"
            img.save(OUT / rel)
            out_items.append({"filename": rel, "label": label})
        return out_items

    # Collect examples for each split
    train_ex = [ex for vid in train_vids for ex in groups[vid]]
    val_select_ex = [ex for vid in val_select_vids for ex in groups[vid]]
    val_calib_ex = [ex for vid in val_calib_vids for ex in groups[vid]]

    print(f"  Train videos: {len(train_vids)}, examples: {len(train_ex)}")
    print(f"  Val_select videos: {len(val_select_vids)}, examples: {len(val_select_ex)}")
    print(f"  Val_calib videos: {len(val_calib_vids)}, examples: {len(val_calib_ex)}")

    return (
        dump_examples(train_ex, f"{prefix}_train"),
        dump_examples(val_select_ex, f"{prefix}_vsel"),
        dump_examples(val_calib_ex, f"{prefix}_vcal"),
    )

# Export train split (grouped by video)
train_items, vsel_items, vcal_items = export_grouped(ds["train"], "hf")

def export_plain(hf_split, name):
    """Export test split (no splitting needed)"""
    print(f"\nExporting {name}...")
    out_items = []
    for i, ex in enumerate(tqdm(hf_split, desc=f"Exporting {name}")):
        img = ex[IMAGE_COL]
        if not isinstance(img, Image.Image):
            img = Image.open(img) if hasattr(img, 'read') else Image.fromarray(img)

        label = int(ex[LABEL_COL])
        rel = f"images/{name}_{i:08d}.png"
        img.save(OUT / rel)
        out_items.append({"filename": rel, "label": label})
    return out_items

# Export test split
test_items = export_plain(ds["test"], "hf_test")

# Create splits.json
splits = {
    "train": train_items,
    "val_select": vsel_items,
    "val_calib": vcal_items,
    "val_test": test_items,
}

splits_file = OUT / "splits.json"
print(f"\nWriting splits.json to {splits_file}...")
with open(splits_file, "w") as f:
    json.dump(splits, f, indent=2)

print("\n" + "="*70)
print("✅ EXPORT COMPLETE!")
print("="*70)
print(f"Root: {OUT}")
print(f"Images: {IMG_DIR}")
print(f"Splits: {splits_file}")
print("\nSplit sizes:")
for k, v in splits.items():
    print(f"  {k}: {len(v)} examples")
print("\nReady for training!")
print(f"Use: data.data_root={OUT}")
