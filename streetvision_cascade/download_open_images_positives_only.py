#!/usr/bin/env python3
"""
Open Images V7 Download (2025 CORRECTED)

CRITICAL: When downloading by classes, FiftyOne only loads samples that
contain those classes. This means you're downloading POSITIVES, not a
balanced binary dataset.

This script correctly treats all downloaded images as label=1 (roadwork).
Your negatives should come from NATIX and normal driving data.
"""
import fiftyone as fo
import fiftyone.zoo as foz
import os
import json
import pandas as pd

# Target classes for roadwork-related objects
# Use EXACT class names from Open Images V7 documentation
target_classes = [
    "Traffic sign",
    "Traffic cone",
    "Barrel",
    "Person",  # For workers
]

print("="*80)
print("OPEN IMAGES V7 DOWNLOAD (POSITIVES BOOSTER)")
print("="*80)
print(f"\nTarget classes: {target_classes}")
print(f"⚠️  IMPORTANT: These will be POSITIVES ONLY (label=1)")
print(f"   Negatives come from NATIX, not this dataset.\n")

# Download train split with targeted classes
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=target_classes,
    max_samples=2000,  # Limit to avoid huge download
    dataset_name="open_images_roadwork_positives",
)

print(f"\n✅ Downloaded {len(dataset)} images")
print(f"   (These are images that contain at least one target class)")

# Export to COCO format (preserves bounding boxes for future use)
export_dir = "data/open_images/coco"
os.makedirs(export_dir, exist_ok=True)

print(f"\nExporting to COCO format...")
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",  # CORRECT field name
)

print(f"✅ Exported COCO to: {export_dir}")

# Create binary labels (ALL as positives)
print(f"\nCreating binary labels...")

coco_path = os.path.join(export_dir, "labels.json")
with open(coco_path, 'r') as f:
    coco = json.load(f)

# ALL images are labeled as 1 (positives)
samples = []
for img in coco['images']:
    img_path = os.path.join(export_dir, "data", img['file_name'])
    label = 1  # ALL positives (we filtered by classes)
    samples.append((img_path, label))

# Save to CSV
os.makedirs("data/open_images", exist_ok=True)
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv("data/open_images/train_labels.csv", index=False, header=False)

print(f"\n✅ Created binary labels: data/open_images/train_labels.csv")
print(f"   Total samples: {len(samples)}")
print(f"   ALL labeled as 1 (positives)")
print(f"\n⚠️  Remember: Your negatives come from NATIX, not this dataset!")
print("="*80)
