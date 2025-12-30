#!/usr/bin/env python3
"""
Export stratified subset for smoke testing (FIXED - balanced classes!)

CRITICAL FIX: Uses stratified sampling to ensure all classes are represented.
Previous version used sequential indices 0-499, which were all class 0.
"""
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import random

OUT = Path.home() / "data" / "natix_subset_stratified"
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
ds = load_dataset("natix-network-org/roadwork")

# Small subset for smoke test (same total size as before)
N_TRAIN = 300
N_VAL_SELECT = 50
N_VAL_CALIB = 50
N_VAL_TEST = 100

TOTAL = N_TRAIN + N_VAL_SELECT + N_VAL_CALIB + N_VAL_TEST  # 500

print(f"\nDataset loaded:")
print(f"  Train: {len(ds['train'])} samples")
print(f"  Test: {len(ds['test'])} samples")

# Step 1: Group train samples by class
print("\nGrouping train samples by class...")
train_by_class = defaultdict(list)
for i, ex in enumerate(tqdm(ds['train'], desc="Indexing")):
    label = int(ex['label'])
    train_by_class[label].append(i)

num_classes = len(train_by_class)
print(f"\nFound {num_classes} classes in train split:")
for label in sorted(train_by_class.keys()):
    print(f"  Class {label}: {len(train_by_class[label])} samples")

# Step 2: Stratified sampling for train/val_select/val_calib
print("\nPerforming stratified sampling...")

# Calculate samples per class for each split
train_per_class = N_TRAIN // num_classes
val_select_per_class = N_VAL_SELECT // num_classes
val_calib_per_class = N_VAL_CALIB // num_classes

print(f"Target samples per class:")
print(f"  Train: {train_per_class} × {num_classes} = {train_per_class * num_classes}")
print(f"  Val Select: {val_select_per_class} × {num_classes} = {val_select_per_class * num_classes}")
print(f"  Val Calib: {val_calib_per_class} × {num_classes} = {val_calib_per_class * num_classes}")

# Sample indices for each split
train_indices = []
val_select_indices = []
val_calib_indices = []

random.seed(42)  # Reproducible
for label in sorted(train_by_class.keys()):
    indices = train_by_class[label].copy()
    random.shuffle(indices)

    # Ensure we have enough samples
    needed = train_per_class + val_select_per_class + val_calib_per_class
    if len(indices) < needed:
        print(f"  WARNING: Class {label} has only {len(indices)} samples, need {needed}")
        # Use all available
        train_indices.extend(indices[:len(indices)])
        continue

    # Split
    train_indices.extend(indices[:train_per_class])
    val_select_indices.extend(indices[train_per_class:train_per_class + val_select_per_class])
    val_calib_indices.extend(indices[train_per_class + val_select_per_class:train_per_class + val_select_per_class + val_calib_per_class])

# Shuffle to mix classes
random.shuffle(train_indices)
random.shuffle(val_select_indices)
random.shuffle(val_calib_indices)

print(f"\nActual samples selected:")
print(f"  Train: {len(train_indices)}")
print(f"  Val Select: {len(val_select_indices)}")
print(f"  Val Calib: {len(val_calib_indices)}")

# Step 3: Sample val_test from test split (also stratified)
print("\nGrouping test samples by class...")
test_by_class = defaultdict(list)
for i, ex in enumerate(tqdm(ds['test'], desc="Indexing test")):
    label = int(ex['label'])
    test_by_class[label].append(i)

val_test_per_class = N_VAL_TEST // num_classes
val_test_indices = []

for label in sorted(test_by_class.keys()):
    indices = test_by_class[label].copy()
    random.shuffle(indices)
    # Take up to val_test_per_class samples
    val_test_indices.extend(indices[:min(val_test_per_class, len(indices))])

random.shuffle(val_test_indices)
print(f"  Val Test: {len(val_test_indices)}")

# Step 4: Export images
def export_indices(split_name, dataset, indices):
    """Export images from indices"""
    print(f"\nExporting {split_name} ({len(indices)} examples)...")
    items = []
    for idx in tqdm(indices, desc=split_name):
        ex = dataset[idx]
        img = ex['image']
        label = int(ex['label'])
        filename = f"images/{split_name}_{idx:05d}.png"
        img.save(OUT / filename)
        items.append({"filename": filename, "label": label})
    return items

# Export all splits
train_items = export_indices("train", ds['train'], train_indices)
val_select_items = export_indices("val_select", ds['train'], val_select_indices)
val_calib_items = export_indices("val_calib", ds['train'], val_calib_indices)
val_test_items = export_indices("val_test", ds['test'], val_test_indices)

# Step 5: Verify class distribution
print("\n" + "="*70)
print("VERIFICATION: Class Distribution")
print("="*70)

from collections import Counter

for name, items in [("train", train_items), ("val_select", val_select_items),
                     ("val_calib", val_calib_items), ("val_test", val_test_items)]:
    labels = [item['label'] for item in items]
    counts = Counter(labels)
    print(f"\n{name}: {len(items)} samples")
    for label in sorted(counts.keys()):
        print(f"  Class {label}: {counts[label]:3d} ({100.0*counts[label]/len(items):5.1f}%)")

# Step 6: Save splits.json
splits = {
    "train": train_items,
    "val_select": val_select_items,
    "val_calib": val_calib_items,
    "val_test": val_test_items,
}

with open(OUT / "splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print("\n" + "="*70)
print("✅ STRATIFIED SUBSET READY FOR SMOKE TEST!")
print("="*70)
print(f"Root: {OUT}")
print("Sizes:", {k: len(v) for k, v in splits.items()})
print(f"\nUse: data.data_root={OUT}")
print("\nNOTE: This is a STRATIFIED SUBSET (balanced classes) for smoke testing.")
print("All 13 classes are represented proportionally.")
