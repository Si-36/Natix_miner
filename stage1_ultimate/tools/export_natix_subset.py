#!/usr/bin/env python3
"""
Export small subset for smoke testing (fast!)
"""
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm

OUT = Path.home() / "data" / "natix_subset"
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
ds = load_dataset("natix-network-org/roadwork")

# Small subset for smoke test
N_TRAIN = 300
N_VAL_SELECT = 50
N_VAL_CALIB = 50
N_VAL_TEST = 100

def export_range(split_name, dataset, start, end):
    """Export range of examples"""
    print(f"\nExporting {split_name} ({end-start} examples)...")
    items = []
    for i in tqdm(range(start, end), desc=split_name):
        ex = dataset[i]
        img = ex['image']
        label = int(ex['label'])
        filename = f"images/{split_name}_{i:05d}.png"
        img.save(OUT / filename)
        items.append({"filename": filename, "label": label})
    return items

# Export subsets
train_items = export_range("train", ds['train'], 0, N_TRAIN)
val_select_items = export_range("val_select", ds['train'], N_TRAIN, N_TRAIN + N_VAL_SELECT)
val_calib_items = export_range("val_calib", ds['train'], N_TRAIN + N_VAL_SELECT, N_TRAIN + N_VAL_SELECT + N_VAL_CALIB)
val_test_items = export_range("val_test", ds['test'], 0, N_VAL_TEST)

splits = {
    "train": train_items,
    "val_select": val_select_items,
    "val_calib": val_calib_items,
    "val_test": val_test_items,
}

with open(OUT / "splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print("\n" + "="*70)
print("✅ SUBSET READY FOR SMOKE TEST!")
print("="*70)
print(f"Root: {OUT}")
print("Sizes:", {k: len(v) for k, v in splits.items()})
print(f"\nUse: data.data_root={OUT}")
print("\nNOTE: This is a SUBSET (500 total images) for smoke testing.")
print("Run full export later if needed.")
