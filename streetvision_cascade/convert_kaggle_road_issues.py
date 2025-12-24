#!/usr/bin/env python3
"""
Kaggle Road Issues Converter (2025 PRO)

This dataset contains road problems (potholes, broken signs, etc.)
but NOT active roadwork/construction zones.

All images are labeled as 0 (no roadwork) to help the model learn
what road problems look like WITHOUT active construction.

This is valuable negative data to reduce false positives.
"""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

print("="*80)
print("KAGGLE ROAD ISSUES CONVERTER (NEGATIVES DATASET)")
print("="*80)

raw_dir = Path("data/kaggle_road_issues/raw/data/Road Issues")
output_dir = Path("data/kaggle_road_issues/images")
output_dir.mkdir(parents=True, exist_ok=True)

if not raw_dir.exists():
    print(f"❌ Error: {raw_dir} not found!")
    exit(1)

# Categories in the dataset
categories = [
    "Broken Road Sign Issues",
    "Damaged Road issues",
    "Illegal Parking Issues",
    "Mixed Issues",
    "Pothole Issues",
]

print(f"\n📊 Dataset categories:")
for cat in categories:
    cat_dir = raw_dir / cat
    if cat_dir.exists():
        img_count = len(list(cat_dir.glob("*.jpg"))) + len(list(cat_dir.glob("*.png")))
        print(f"   {cat}: {img_count} images")

samples = []
total_copied = 0

print(f"\n⚠️  IMPORTANT: Labeling ALL as 0 (no roadwork)")
print(f"   These are road PROBLEMS, not active CONSTRUCTION")
print(f"   This helps reduce false positives!\n")

# Process all categories
for cat in categories:
    cat_dir = raw_dir / cat
    
    if not cat_dir.exists():
        print(f"⚠️  Skipping {cat} (not found)")
        continue
    
    # Find all images
    image_files = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg"))
    
    print(f"Processing {cat}: {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc=f"  {cat[:30]}"):
        try:
            # Create unique filename with category prefix
            cat_prefix = cat.replace(" ", "_").replace("/", "_")
            new_filename = f"{cat_prefix}_{img_path.name}"
            new_path = output_dir / new_filename
            
            # Copy image
            shutil.copy2(img_path, new_path)
            
            # All labeled as 0 (no roadwork - these are just road problems)
            rel_path = f"data/kaggle_road_issues/images/{new_filename}"
            samples.append((rel_path, 0))
            total_copied += 1
            
        except Exception as e:
            print(f"⚠️  Failed to process {img_path.name}: {e}")

# Save labels
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv("data/kaggle_road_issues/train_labels.csv", index=False, header=False)

print(f"\n✅ Processed {total_copied} images")
print(f"✅ Created labels: data/kaggle_road_issues/train_labels.csv")
print(f"   ALL labeled as 0 (no roadwork - road problems only)")

print(f"\n📊 Label distribution:")
label_counts = df['label'].value_counts()
print(f"   Class 0 (no roadwork): {label_counts.get(0, 0)} (100%)")
print(f"   Class 1 (roadwork):    {label_counts.get(1, 0)} (0%)")

print(f"\n💡 This is VALUABLE negative data:")
print(f"   Helps model learn: potholes ≠ roadwork")
print(f"   Reduces false positives on damaged roads")
print("="*80)
