#!/bin/bash
# Script 3: Create Stratified Splits
# Creates splits.json with 70/10/10/10 split
# Usage: bash scripts/3_create_splits.sh

set -e
echo "==================================================================="
echo "Creating Stratified Splits"
echo "==================================================================="
echo ""

source .venv/bin/activate

python << 'EOF'
import json
import random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Paths
data_root = Path.home() / "natix" / "data" / "natix_full"
output_path = data_root / "splits.json"

print(f"Data root: {data_root}")
print("")

# Find all images
class_0_dir = data_root / "not_roadwork"
class_1_dir = data_root / "roadwork"

if not class_0_dir.exists() or not class_1_dir.exists():
    print("❌ Error: Class folders not found!")
    print(f"  Expected: {class_0_dir}")
    print(f"  Expected: {class_1_dir}")
    print("")
    print("Make sure you ran script 2_download_data.sh first!")
    exit(1)

class_0_images = sorted(list(class_0_dir.glob("*.jpg")) + list(class_0_dir.glob("*.png")))
class_1_images = sorted(list(class_1_dir.glob("*.jpg")) + list(class_1_dir.glob("*.png")))

print(f"Class 0 (not_roadwork): {len(class_0_images)} images")
print(f"Class 1 (roadwork): {len(class_1_images)} images")
print(f"Total: {len(class_0_images) + len(class_1_images)} images")
print("")

if len(class_0_images) == 0 or len(class_1_images) == 0:
    print("❌ Error: No images found in class folders!")
    exit(1)

# Shuffle
random.shuffle(class_0_images)
random.shuffle(class_1_images)

# Stratified split function (70/10/10/10)
def stratified_split(images, ratios=[0.7, 0.1, 0.1, 0.1]):
    n = len(images)
    split_sizes = [int(n * r) for r in ratios]
    # Fix rounding error
    split_sizes[-1] = n - sum(split_sizes[:-1])

    splits = []
    start = 0
    for size in split_sizes:
        splits.append(images[start:start+size])
        start += size
    return splits

# Split each class separately (stratification)
splits_0 = stratified_split(class_0_images)
splits_1 = stratified_split(class_1_images)

# Combine classes for each split
splits = {
    "train": [str(p.relative_to(data_root)) for p in splits_0[0] + splits_1[0]],
    "val_select": [str(p.relative_to(data_root)) for p in splits_0[1] + splits_1[1]],
    "val_calib": [str(p.relative_to(data_root)) for p in splits_0[2] + splits_1[2]],
    "val_test": [str(p.relative_to(data_root)) for p in splits_0[3] + splits_1[3]],
}

# Shuffle each split (mix classes)
for split_files in splits.values():
    random.shuffle(split_files)

# Save splits.json
with open(output_path, 'w') as f:
    json.dump(splits, f, indent=2)

print("Splits created:")
for name, files in splits.items():
    class_0_count = sum(1 for f in files if 'not_roadwork' in f)
    class_1_count = sum(1 for f in files if 'roadwork' in f)
    print(f"  {name:12s}: {len(files):5d} images (class_0: {class_0_count}, class_1: {class_1_count})")

print("")
print(f"✅ Splits saved to: {output_path}")

# Verify no overlap
all_files = []
for files in splits.values():
    all_files.extend(files)

if len(all_files) != len(set(all_files)):
    duplicates = len(all_files) - len(set(all_files))
    print(f"❌ ERROR: {duplicates} duplicate files across splits!")
    exit(1)
else:
    print("✅ No overlap between splits (clean split)")

print("")
print("Next: bash scripts/4_run_phase4.sh")

EOF
