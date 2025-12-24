#!/usr/bin/env python3
"""
Open Images V7 Download (NO MONGODB VERSION)

This script processes already-downloaded Open Images V7 images
without requiring MongoDB. It creates the binary labels CSV directly.

The images should already be downloaded to:
/home/sina/fiftyone/open-images-v7/train/data/
"""
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Source directory (where FiftyOne downloaded images)
source_dir = Path("/home/sina/fiftyone/open-images-v7/train/data")

# Target directories
target_image_dir = Path("data/open_images/images")
target_coco_dir = Path("data/open_images/coco/data")
target_labels_csv = Path("data/open_images/train_labels.csv")

print("="*80)
print("OPEN IMAGES V7 PROCESSING (NO MONGODB)")
print("="*80)

# Check if source directory exists
if not source_dir.exists():
    print(f"❌ Error: Source directory not found: {source_dir}")
    print(f"   Please run the original download script first to download images.")
    print(f"   (It will fail at MongoDB step, but images will be downloaded)")
    exit(1)

# Get all image files
image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
print(f"\n📁 Found {len(image_files)} images in {source_dir}")

if len(image_files) == 0:
    print(f"❌ No images found! Check if download completed.")
    exit(1)

# Create target directories
target_image_dir.mkdir(parents=True, exist_ok=True)
target_coco_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📋 Copying images to target directory...")
print(f"   Source: {source_dir}")
print(f"   Target: {target_image_dir}")

# Copy images and create labels
samples = []
for img_file in tqdm(image_files, desc="Copying images"):
    # Copy to both locations
    target_path = target_image_dir / img_file.name
    coco_path = target_coco_dir / img_file.name
    
    shutil.copy2(img_file, target_path)
    shutil.copy2(img_file, coco_path)
    
    # ALL images are labeled as 1 (positives)
    # Format: image_path,label
    rel_path = f"data/open_images/images/{img_file.name}"
    samples.append((rel_path, 1))

print(f"\n✅ Copied {len(samples)} images")

# Save labels CSV
print(f"\n📝 Creating binary labels CSV...")
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv(target_labels_csv, index=False, header=False)

print(f"✅ Created labels: {target_labels_csv}")
print(f"   Total samples: {len(samples)}")
print(f"   ALL labeled as 1 (positives)")

# Create a minimal COCO format JSON (for compatibility)
print(f"\n📦 Creating minimal COCO format...")
coco_data = {
    "images": [
        {
            "id": idx,
            "file_name": img_file.name,
            "width": 0,  # Unknown, but not needed for binary classification
            "height": 0
        }
        for idx, img_file in enumerate(image_files)
    ],
    "annotations": [],  # No bounding boxes needed for binary classification
    "categories": [
        {"id": 1, "name": "roadwork", "supercategory": "none"}
    ]
}

import json
coco_json_path = Path("data/open_images/coco/labels.json")
with open(coco_json_path, 'w') as f:
    json.dump(coco_data, f, indent=2)

print(f"✅ Created COCO format: {coco_json_path}")

print(f"\n" + "="*80)
print("✅ COMPLETE!")
print("="*80)
print(f"📁 Images: {target_image_dir} ({len(samples)} files)")
print(f"📁 COCO: {target_coco_dir} ({len(samples)} files)")
print(f"📄 Labels: {target_labels_csv}")
print(f"\n⚠️  Remember: These are ALL positives (label=1)")
print(f"   Negatives come from NATIX dataset!")
print("="*80)

