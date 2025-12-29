#!/usr/bin/env python3
"""
Memory-efficient NATIX export (no grouping, faster, simpler)

Exports HF dataset to disk with train/val splits
"""
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

# Output directory
OUT = Path.home() / "data" / "natix"
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset from HF cache...")
ds = load_dataset("natix-network-org/roadwork")

print(f"\nDataset loaded:")
print(f"  Train: {len(ds['train'])} examples")
print(f"  Test: {len(ds['test'])} examples")

# Simple split: use first 80% for train, 10% for val_select, 10% for val_calib
n_train_total = len(ds['train'])
n_train = int(0.80 * n_train_total)
n_val_select = int(0.10 * n_train_total)

train_indices = list(range(0, n_train))
val_select_indices = list(range(n_train, n_train + n_val_select))
val_calib_indices = list(range(n_train + n_val_select, n_train_total))

print(f"\nSplit sizes:")
print(f"  Train: {len(train_indices)}")
print(f"  Val_select: {len(val_select_indices)}")
print(f"  Val_calib: {len(val_calib_indices)}")
print(f"  Val_test: {len(ds['test'])}")

def export_indices(split_name, dataset, indices):
    """Export specific indices from dataset"""
    print(f"\nExporting {split_name}...")
    items = []

    for i in tqdm(indices, desc=split_name):
        ex = dataset[i]
        img = ex['image']
        if not isinstance(img, Image.Image):
            img = Image.open(img) if hasattr(img, 'read') else Image.fromarray(img)

        label = int(ex['label'])
        filename = f"images/{split_name}_{i:08d}.png"
        img.save(OUT / filename)
        items.append({"filename": filename, "label": label})

    return items

def export_full_split(split_name, dataset):
    """Export entire split"""
    print(f"\nExporting {split_name}...")
    items = []

    for i in tqdm(range(len(dataset)), desc=split_name):
        ex = dataset[i]
        img = ex['image']
        if not isinstance(img, Image.Image):
            img = Image.open(img) if hasattr(img, 'read') else Image.fromarray(img)

        label = int(ex['label'])
        filename = f"images/{split_name}_{i:08d}.png"
        img.save(OUT / filename)
        items.append({"filename": filename, "label": label})

    return items

# Export all splits
train_items = export_indices("train", ds['train'], train_indices)
val_select_items = export_indices("val_select", ds['train'], val_select_indices)
val_calib_items = export_indices("val_calib", ds['train'], val_calib_indices)
val_test_items = export_full_split("val_test", ds['test'])

# Create splits.json
splits = {
    "train": train_items,
    "val_select": val_select_items,
    "val_calib": val_calib_items,
    "val_test": val_test_items,
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
print("\nFinal split sizes:")
for k, v in splits.items():
    print(f"  {k}: {len(v)} examples")
print(f"\nReady for training! Use: data.data_root={OUT}")
