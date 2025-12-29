#!/usr/bin/env python3
"""
Fast NATIX export using batch processing
"""
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm
import concurrent.futures
from functools import partial

# Output directory
OUT = Path.home() / "data" / "natix"
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
ds = load_dataset("natix-network-org/roadwork")

print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

# Split indices
n_train_total = len(ds['train'])
n_train = int(0.80 * n_train_total)
n_val_select = int(0.10 * n_train_total)

def save_example(idx, ex, split_name, out_dir):
    """Save single example (for parallel processing)"""
    try:
        img = ex['image']
        label = int(ex['label'])
        filename = f"images/{split_name}_{idx:08d}.png"
        img.save(out_dir / filename)
        return {"filename": filename, "label": label}
    except Exception as e:
        print(f"Error saving {split_name}_{idx}: {e}")
        return None

def export_split_parallel(split_name, dataset, indices, max_workers=4):
    """Export split using parallel processing"""
    print(f"\nExporting {split_name} ({len(indices)} examples)...")
    items = []

    # Create partial function with fixed args
    save_fn = partial(save_example, split_name=split_name, out_dir=OUT)

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in indices:
            ex = dataset[idx]
            future = executor.submit(save_fn, idx, ex)
            futures.append(future)

        # Collect results with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=split_name):
            result = future.result()
            if result:
                items.append(result)

    # Sort by filename to maintain order
    items.sort(key=lambda x: x['filename'])
    return items

print("\n" + "="*70)
print("Exporting with 8 parallel workers...")
print("="*70)

# Export splits in parallel
train_indices = list(range(0, n_train))
val_select_indices = list(range(n_train, n_train + n_val_select))
val_calib_indices = list(range(n_train + n_val_select, n_train_total))
test_indices = list(range(len(ds['test'])))

train_items = export_split_parallel("train", ds['train'], train_indices, max_workers=8)
val_select_items = export_split_parallel("val_select", ds['train'], val_select_indices, max_workers=8)
val_calib_items = export_split_parallel("val_calib", ds['train'], val_calib_indices, max_workers=8)
val_test_items = export_split_parallel("val_test", ds['test'], test_indices, max_workers=8)

# Create splits.json
splits = {
    "train": train_items,
    "val_select": val_select_items,
    "val_calib": val_calib_items,
    "val_test": val_test_items,
}

splits_file = OUT / "splits.json"
print(f"\nWriting {splits_file}...")
with open(splits_file, "w") as f:
    json.dump(splits, f, indent=2)

print("\n" + "="*70)
print("✅ DONE!")
print("="*70)
print(f"Root: {OUT}")
print("Split sizes:")
for k, v in splits.items():
    print(f"  {k}: {len(v)}")
print(f"\nUse: data.data_root={OUT}")
