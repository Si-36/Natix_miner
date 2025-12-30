#!/bin/bash
# Script 2: Download NATIX Dataset from HuggingFace
# Usage: bash scripts/2_download_data.sh

set -e
echo "==================================================================="
echo "Downloading NATIX Dataset from HuggingFace"
echo "==================================================================="
echo ""

# Activate venv
source .venv/bin/activate

# Download using datasets library
python << 'EOF'
from datasets import load_dataset
from pathlib import Path
import shutil

print("Downloading NATIX dataset from HuggingFace...")

# ADJUST THIS: Replace with actual HuggingFace dataset path
# Common options:
#   - "username/natix-roadwork"
#   - "natix/roadwork-detection"
# If you don't know it, search on https://huggingface.co/datasets

DATASET_NAME = "sina/natix-roadwork"  # CHANGE THIS TO YOUR DATASET

print(f"Dataset: {DATASET_NAME}")
print("")

try:
    # Load dataset
    dataset = load_dataset(DATASET_NAME)

    # Create output directory
    output_dir = Path.home() / "natix" / "data" / "natix_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_dir}")
    print("")

    # Save images to disk
    # Assuming dataset has 'image' and 'label' columns
    # Adjust if structure is different

    splits_to_download = ['train', 'validation', 'test'] if 'train' in dataset else list(dataset.keys())

    all_images = []
    all_labels = []

    for split_name in splits_to_download:
        if split_name not in dataset:
            continue

        print(f"Processing {split_name}...")
        split_data = dataset[split_name]

        # Create class folders
        class_0_dir = output_dir / "not_roadwork"
        class_1_dir = output_dir / "roadwork"
        class_0_dir.mkdir(exist_ok=True)
        class_1_dir.mkdir(exist_ok=True)

        for i, example in enumerate(split_data):
            image = example['image']  # PIL Image
            label = example['label']  # 0 or 1

            # Save image
            if label == 0:
                image_path = class_0_dir / f"{split_name}_{i:05d}.jpg"
            else:
                image_path = class_1_dir / f"{split_name}_{i:05d}.jpg"

            image.save(image_path)

            all_images.append(str(image_path.relative_to(output_dir)))
            all_labels.append(label)

        print(f"  Saved {len(split_data)} images")

    print("")
    print(f"✅ Downloaded {len(all_images)} total images")
    print(f"   Class 0 (not_roadwork): {sum(1 for l in all_labels if l == 0)}")
    print(f"   Class 1 (roadwork): {sum(1 for l in all_labels if l == 1)}")
    print("")
    print(f"Dataset saved to: {output_dir}")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("")
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Go to https://huggingface.co/datasets")
    print("2. Search for 'NATIX roadwork' or your dataset name")
    print("3. Download using 'datasets' library or manual download")
    print("4. Place images in ~/natix/data/natix_full/")
    print("   Structure:")
    print("     natix_full/")
    print("       roadwork/")
    print("         image_001.jpg")
    print("         ...")
    print("       not_roadwork/")
    print("         image_001.jpg")
    print("         ...")
    exit(1)

EOF

echo ""
echo "✅ Data download complete!"
echo ""
echo "Next: bash scripts/3_create_splits.sh"
