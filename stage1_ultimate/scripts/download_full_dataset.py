#!/usr/bin/env python3
"""
Download FULL NATIX Dataset from HuggingFace
Creates stratified splits (70/10/10/10)
Based on NATIX official download code
"""
import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME = "natix-network-org/roadwork"
OUTPUT_DIR = Path.home() / "natix" / "data" / "natix_full"
SEED = 42

random.seed(SEED)

def main():
    print("=" * 70)
    print("NATIX Full Dataset Download")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print("")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    # Download dataset
    print("Downloading from HuggingFace...")
    try:
        dataset = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("")
        print("MANUAL FIX:")
        print(f"1. Go to https://huggingface.co/datasets/{DATASET_NAME}")
        print("2. Check if dataset exists and is public")
        print("3. If private, run: huggingface-cli login")
        return

    print(f"Dataset splits: {list(dataset.keys())}")
    print("")

    # Combine all splits
    all_data = []
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"Processing {split_name}: {len(split_data)} images...")

        for i, example in enumerate(tqdm(split_data, desc=f"  {split_name}")):
            # Save image
            image = example['image']  # PIL Image
            label = example['label']  # 0 or 1

            image_filename = f"{split_name}_{i:05d}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)

            all_data.append({
                "filename": f"images/{image_filename}",
                "label": int(label)
            })

    print("")
    print(f"Total images: {len(all_data)}")

    # Split by class
    class_0 = [d for d in all_data if d['label'] == 0]
    class_1 = [d for d in all_data if d['label'] == 1]

    print(f"Class 0 (not roadwork): {len(class_0)}")
    print(f"Class 1 (roadwork): {len(class_1)}")
    print("")

    # Shuffle
    random.shuffle(class_0)
    random.shuffle(class_1)

    # Stratified split (70/10/10/10)
    def stratified_split(items, ratios=[0.7, 0.1, 0.1, 0.1]):
        n = len(items)
        sizes = [int(n * r) for r in ratios]
        sizes[-1] = n - sum(sizes[:-1])  # Fix rounding

        splits = []
        start = 0
        for size in sizes:
            splits.append(items[start:start+size])
            start += size
        return splits

    splits_0 = stratified_split(class_0)
    splits_1 = stratified_split(class_1)

    # Combine classes
    splits = {
        "train": splits_0[0] + splits_1[0],
        "val_select": splits_0[1] + splits_1[1],
        "val_calib": splits_0[2] + splits_1[2],
        "val_test": splits_0[3] + splits_1[3],
    }

    # Shuffle each split
    for split_items in splits.values():
        random.shuffle(split_items)

    # Save splits.json
    splits_json_path = OUTPUT_DIR / "splits.json"
    with open(splits_json_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print("Splits created:")
    for name, items in splits.items():
        c0 = sum(1 for d in items if d['label'] == 0)
        c1 = sum(1 for d in items if d['label'] == 1)
        print(f"  {name:12s}: {len(items):5d} ({c0} class_0, {c1} class_1)")

    print("")
    print(f"✅ Dataset saved to: {OUTPUT_DIR}")
    print(f"✅ Splits saved to: {splits_json_path}")

    # Verify no overlap
    all_files = []
    for items in splits.values():
        all_files.extend([d['filename'] for d in items])

    if len(all_files) == len(set(all_files)):
        print("✅ No overlap between splits")
    else:
        print(f"❌ WARNING: {len(all_files) - len(set(all_files))} duplicates!")

    print("")
    print("Next: Copy splits.json to run directory")
    print(f"  cp {splits_json_path} ~/natix/runs/run_001/splits.json")

if __name__ == "__main__":
    main()
