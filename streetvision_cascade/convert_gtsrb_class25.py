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

raw_dir = "data/gtsrb_class25/raw/Train"
output_dir = "data/gtsrb_class25/train_images"
os.makedirs(output_dir, exist_ok=True)

# Class 25 is "Road work" in GTSRB
class25_dir = os.path.join(raw_dir, "25")

if not os.path.exists(class25_dir):
    print(f"❌ Error: {class25_dir} not found!")
    print(f"   Make sure you downloaded and extracted GTSRB dataset.")
    print(f"   Expected structure: data/gtsrb_class25/raw/Train/25/")
    exit(1)

# Find all .ppm images
ppm_files = list(Path(class25_dir).glob("*.ppm"))

if not ppm_files:
    print(f"❌ No .ppm files found in {class25_dir}")
    exit(1)

print(f"\n✅ Found {len(ppm_files)} Class 25 (roadwork) sign images")
print(f"Converting .ppm → .png using PIL...")

samples = []

for ppm_path in tqdm(ppm_files, desc="Converting"):
    try:
        # Open .ppm and convert to .png
        img = Image.open(ppm_path)

        # Save as .png
        png_filename = ppm_path.stem + ".png"
        png_path = os.path.join(output_dir, png_filename)
        img.save(png_path, "PNG")

        # All are roadwork signs (label=1)
        samples.append((png_path, 1))
    except Exception as e:
        print(f"⚠️  Failed to convert {ppm_path.name}: {e}")

# Save labels
df = pd.DataFrame(samples, columns=['image_path', 'label'])
df.to_csv("data/gtsrb_class25/train_labels.csv", index=False, header=False)

print(f"\n✅ Converted {len(samples)} images to PNG")
print(f"✅ Created labels: data/gtsrb_class25/train_labels.csv")
print(f"   ALL labeled as 1 (EU roadwork signs)")
print("="*80)
