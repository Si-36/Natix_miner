#!/usr/bin/env python3
"""
GTSRB Class 25 Converter (2025 CORRECTED)

Properly converts .ppm images to .png using PIL.
Class 25 = "Road work" signs in German traffic sign benchmark.

Use this as EU sign booster (all labeled as 1).
"""
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

print("="*80)
print("GTSRB CLASS 25 CONVERTER (EU ROADWORK SIGNS)")
print("="*80)

# Try both uppercase and lowercase "Train"
raw_dir_upper = Path("data/gtsrb_class25/raw/Train")
raw_dir_lower = Path("data/gtsrb_class25/raw/train")

if raw_dir_upper.exists():
    raw_dir = raw_dir_upper
elif raw_dir_lower.exists():
    raw_dir = raw_dir_lower
else:
    print(f"❌ Error: Neither {raw_dir_upper} nor {raw_dir_lower} found!")
    print(f"   Make sure you downloaded and extracted GTSRB dataset.")
    print(f"   Expected structure: data/gtsrb_class25/raw/Train/25/ or data/gtsrb_class25/raw/train/25/")
    exit(1)

output_dir = Path("data/gtsrb_class25/train_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Class 25 is "Road work" in GTSRB
class25_dir = raw_dir / "25"

if not class25_dir.exists():
    print(f"❌ Error: {class25_dir} not found!")
    print(f"   Available classes: {sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])}")
    print(f"   Make sure Class 25 directory exists.")
    exit(1)

# Find all .ppm and .png images (some datasets already have PNG)
ppm_files = list(class25_dir.glob("*.ppm"))
png_files = list(class25_dir.glob("*.png"))

if not ppm_files and not png_files:
    print(f"❌ No .ppm or .png files found in {class25_dir}")
    print(f"   Found files: {list(class25_dir.glob('*'))[:5]}")
    exit(1)

# Use PPM if available, otherwise PNG
if ppm_files:
    image_files = ppm_files
    convert_needed = True
    print(f"\n✅ Found {len(ppm_files)} Class 25 (roadwork) sign images (.ppm)")
    print(f"   Converting .ppm → .png using PIL...")
else:
    image_files = png_files
    convert_needed = False
    print(f"\n✅ Found {len(png_files)} Class 25 (roadwork) sign images (.png)")
    print(f"   Files are already PNG format, copying...")

samples = []

for img_path in tqdm(image_files, desc="Processing"):
    try:
        if convert_needed:
            # Open .ppm and convert to .png
            img = Image.open(img_path)
            png_filename = img_path.stem + ".png"
            png_path = output_dir / png_filename
            img.save(png_path, "PNG")
        else:
            # Already PNG, just copy
            import shutil
            png_path = output_dir / img_path.name
            shutil.copy2(img_path, png_path)

        # All are roadwork signs (label=1)
        # Use relative path for CSV
        rel_path = f"data/gtsrb_class25/train_images/{png_path.name}"
        samples.append((rel_path, 1))
    except Exception as e:
        print(f"⚠️  Failed to process {img_path.name}: {e}")

# Save labels
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv("data/gtsrb_class25/train_labels.csv", index=False, header=False)

print(f"\n✅ Converted {len(samples)} images to PNG")
print(f"✅ Created labels: data/gtsrb_class25/train_labels.csv")
print(f"   ALL labeled as 1 (EU roadwork signs)")
print("="*80)
