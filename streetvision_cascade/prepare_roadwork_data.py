#!/usr/bin/env python3
"""
ROADWork Dataset Preparation Script (2025 Production-Grade)

Downloads and prepares ROADWork dataset + additional roadwork sources
for training with NATIX data.

Usage:
    python prepare_roadwork_data.py --download_roadwork --download_extra
"""

import os
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import shutil


def download_roadwork_iccv(output_dir="data/roadwork_iccv"):
    """
    Download ROADWork dataset from official GitHub/source.

    ROADWork is an ICCV 2025 dataset with ~5000 work zones in 18 US cities.
    Paper: https://arxiv.org/abs/2406.07661
    GitHub: https://github.com/anuragxel/roadwork-dataset
    """
    print("\n" + "="*80)
    print("DOWNLOADING ROADWORK DATASET (ICCV 2025)")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # GitHub repo for dataset access
    github_repo = "https://github.com/anuragxel/roadwork-dataset"

    print(f"\n📦 ROADWork Dataset Info:")
    print(f"   Paper: https://arxiv.org/abs/2406.07661")
    print(f"   GitHub: {github_repo}")
    print(f"   Output: {output_dir}")

    print(f"\n⚠️  MANUAL STEP REQUIRED:")
    print(f"   1. Visit: {github_repo}")
    print(f"   2. Follow their download instructions (likely Google Drive or similar)")
    print(f"   3. Download the dataset to: {output_dir}/raw/")
    print(f"   4. Extract all images and annotations")
    print(f"   5. Re-run this script with --process_roadwork to convert to our format")

    # Create directory structure
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/train_images", exist_ok=True)
    os.makedirs(f"{output_dir}/val_images", exist_ok=True)

    print(f"\n✅ Created directory structure at {output_dir}")
    print(f"   Waiting for you to download raw data to {output_dir}/raw/")


def process_roadwork_iccv(raw_dir="data/roadwork_iccv/raw", output_dir="data/roadwork_iccv"):
    """
    Process downloaded ROADWork dataset into our binary format.

    ROADWork uses COCO-like format with categories for:
    - Cones, barriers, arrow boards, signs, workers, etc.

    Converts to binary labels:
    - label=1: work zone present (any work zone object detected)
    - label=0: clean road (no work zone objects)

    NOTE: This parser assumes COCO format. Adjust if actual format differs.
    """
    print("\n" + "="*80)
    print("PROCESSING ROADWORK DATASET")
    print("="*80)

    if not os.path.exists(raw_dir):
        print(f"❌ Error: {raw_dir} not found!")
        print(f"   Please download ROADWork ZIPs from CMU KiltHub first.")
        print(f"   Visit: https://github.com/anuragxel/roadwork-dataset")
        return

    # Look for annotation files (COCO format: instances_train.json, instances_val.json)
    annotation_files = list(Path(raw_dir).rglob("instances_*.json")) + \
                       list(Path(raw_dir).rglob("annotations_*.json"))

    if not annotation_files:
        print(f"❌ No COCO annotation files found in {raw_dir}")
        print(f"   Expected: instances_train.json, instances_val.json")
        print(f"   Check if you unzipped annotations.zip correctly")
        return

    print(f"✅ Found {len(annotation_files)} annotation files")

    # Process each annotation file (train/val)
    for ann_file in annotation_files:
        print(f"\nProcessing: {ann_file.name}")

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # COCO format: images list + annotations list
        images_info = {img['id']: img for img in coco_data['images']}

        # Build image_id -> has_annotation map
        image_has_workzone = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            # Any annotation means work zone present
            image_has_workzone[img_id] = True

        # Create samples
        samples = []
        for img_id, img_info in images_info.items():
            # Construct image path relative to raw_dir
            img_filename = img_info['file_name']
            img_path = os.path.join(raw_dir, img_filename)

            # Check if image exists, else try common variations
            if not os.path.exists(img_path):
                # Try under scene/images/ subdirectory
                alt_path = os.path.join(raw_dir, "scene", "images", img_filename)
                if os.path.exists(alt_path):
                    img_path = alt_path

            label = 1 if img_id in image_has_workzone else 0
            samples.append((img_path, label))

        # Determine split from filename
        if 'train' in ann_file.name.lower():
            split = 'train'
        elif 'val' in ann_file.name.lower():
            split = 'val'
        else:
            split = 'train'  # Default to train

        # Save to CSV
        df = pd.DataFrame(samples, columns=['image_path', 'label'])
        csv_path = os.path.join(output_dir, f'{split}_labels.csv')
        df.to_csv(csv_path, index=False, header=False)

        print(f"✅ {split.upper()}: {len(samples)} samples -> {csv_path}")
        print(f"   Work zones: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
        print(f"   Clean roads: {(df['label']==0).sum()} ({100*(1-df['label'].mean()):.1f}%)")


def download_roboflow_roadwork(output_dir="data/roadwork_extra"):
    """
    Download small Roboflow roadwork datasets.

    Source: https://universe.roboflow.com/workzone/roadwork
    """
    print("\n" + "="*80)
    print("DOWNLOADING ROBOFLOW ROADWORK DATASETS")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 Roboflow Roadwork Info:")
    print(f"   URL: https://universe.roboflow.com/workzone/roadwork")
    print(f"   Output: {output_dir}")

    print(f"\n⚠️  MANUAL STEP REQUIRED:")
    print(f"   1. Visit: https://universe.roboflow.com/workzone/roadwork")
    print(f"   2. Download dataset (COCO format or Pascal VOC)")
    print(f"   3. Extract to: {output_dir}/raw/")
    print(f"   4. Re-run with --process_extra to convert")

    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/train_images", exist_ok=True)

    print(f"\n✅ Created directory structure at {output_dir}")


def process_roboflow_roadwork(raw_dir="data/roadwork_extra/raw", output_dir="data/roadwork_extra"):
    """
    Process Roboflow datasets into binary format.
    """
    print("\n" + "="*80)
    print("PROCESSING ROBOFLOW ROADWORK DATASETS")
    print("="*80)

    if not os.path.exists(raw_dir):
        print(f"❌ Error: {raw_dir} not found!")
        return

    # Look for images and annotations
    image_files = list(Path(raw_dir).rglob("*.jpg")) + list(Path(raw_dir).rglob("*.png"))

    if not image_files:
        print(f"❌ No images found in {raw_dir}")
        return

    print(f"✅ Found {len(image_files)} images")

    # Simple heuristic: all images from Roboflow "roadwork" dataset are positive examples
    # (They're specifically collected for roadwork detection)
    train_samples = []

    for img_path in image_files:
        # All images are roadwork examples (label=1)
        train_samples.append((str(img_path), 1))

    # Save to CSV
    train_df = pd.DataFrame(train_samples, columns=['image_path', 'label'])
    train_csv = os.path.join(output_dir, 'train_labels.csv')
    train_df.to_csv(train_csv, index=False, header=False)

    print(f"\n✅ Processed Roboflow roadwork:")
    print(f"   Train: {len(train_samples)} samples -> {train_csv}")
    print(f"   All labeled as roadwork=1 (focused dataset)")


def create_combined_dataset_info():
    """
    Create DATA_SOURCES.md documenting all datasets.
    """
    content = """# Roadwork Training Data Sources

## Overview

This document describes all datasets used for training the Stage 1 DINOv3 roadwork classifier.

---

## 1. NATIX Official Roadwork Dataset (Primary)

**Source**: NATIX StreetVision Subnet 72 official data

**Path**:
- `data/natix_official/train/` + `train_labels.csv`
- `data/natix_official/val/` + `val_labels.csv`

**Description**:
- Crowdsourced roadwork images from NATIX network
- European-centric distribution
- Binary labels: 0 = no roadwork, 1 = roadwork

**Statistics** (update after running):
- Train samples: ~10,000
- Val samples: ~2,500
- Positive rate: ~25%

**Use**: Primary training and validation set. NATIX val is always used as the main validation metric.

---

## 2. ROADWork Dataset (ICCV 2025) - External Source

**Source**: CMU ROADWork Dataset (ICCV 2025)

**Paper**: https://arxiv.org/abs/2406.07661
**GitHub**: https://github.com/anuragxel/roadwork-dataset
**Website**: https://cs.cmu.edu/~roadwork/

**Path**:
- `data/roadwork_iccv/train_images/` + `train_labels.csv`
- `data/roadwork_iccv/val_images/` + `val_labels.csv` (optional)

**Description**:
- Nearly 5,000 work zones in 18 US cities
- Annotations for:
  - Work zone presence
  - Cones, barriers, arrow boards, signs
  - Workers, lane shifts, drivable paths
- Video sequences with temporal context

**Label Mapping**:
- `label = 1` if:
  - work_zone_present = true, OR
  - Any of: cones, barriers, arrow_boards, work_zone_signs, workers present
- `label = 0` otherwise (clean road)

**Statistics** (update after processing):
- Train samples: ~4,000-5,000
- Positive rate: ~80-90% (focused on work zones)

**Why This Dataset**:
- ROADWork paper shows models fine-tuned on this data improve work zone detection by 32.5% precision
- Adds US-centric work zones, different sign styles, lighting conditions
- Covers edge cases: night work, heavy rain, unusual layouts

**Use**: Combined with NATIX train when `--use_extra_roadwork=True`

---

## 3. Roboflow Roadwork Datasets (Small Extras)

**Source**: Roboflow Universe public datasets

**URL**: https://universe.roboflow.com/workzone/roadwork

**Path**:
- `data/roadwork_extra/train_images/` + `train_labels.csv`

**Description**:
- Small curated roadwork image collections
- Various camera types, road markings, countries
- Additional edge cases

**Label Mapping**:
- All images labeled as `1` (focused roadwork dataset)

**Statistics** (update after downloading):
- Train samples: ~500-1,000
- Positive rate: 100% (all roadwork examples)

**Use**: Combined with NATIX + ROADWork when `--use_extra_roadwork=True`

---

## Combined "Super-Train" Dataset

When training with `--use_extra_roadwork=True`:

**Total Training Samples**: ~14,000-16,000
- NATIX train: ~10,000
- ROADWork train: ~4,000-5,000
- Roboflow extra: ~500-1,000

**Positive/Negative Balance**:
- Will be recomputed at training time
- Class weights automatically adjusted for imbalance

**Validation**:
- Always uses NATIX val only (~2,500 samples)
- This ensures metrics remain tied to deployment distribution

---

## Data Pipeline

### Without Extra Data (Default)
```bash
python train_stage1_head.py --mode train
# Uses: NATIX train + NATIX val only
```

### With Extra Data (Aggressive Mode)
```bash
python train_stage1_head.py --mode train --use_extra_roadwork
# Uses: NATIX train + ROADWork train + Roboflow train
# Val: NATIX val only
```

---

## Expected Impact

Adding ROADWork + extras should provide:

1. **Higher Recall**: Fewer missed work zones on unusual layouts
2. **Better F1**: More balanced precision/recall
3. **Lower ECE**: Better calibration from diverse positive examples
4. **Stronger Exit Accuracy**: More confident high-threshold decisions (0.88-0.95)
5. **Geographical Robustness**: US + Europe coverage

ROADWork paper shows 32.5% precision improvement and 12.8× higher discovery rate on global imagery after fine-tuning on their data.

---

## Download Instructions

### 1. NATIX (Already Have)
- You already have this data locally

### 2. ROADWork
```bash
# Download from official source
python prepare_roadwork_data.py --download_roadwork

# Follow manual steps to download from GitHub/Google Drive
# Then process:
python prepare_roadwork_data.py --process_roadwork
```

### 3. Roboflow
```bash
# Download from Roboflow Universe
python prepare_roadwork_data.py --download_extra

# Follow manual steps to download
# Then process:
python prepare_roadwork_data.py --process_extra
```

---

## Quality Control

Before training with combined data:

1. **Visual Inspection**: Sample 100 random images from each dataset
2. **Label Validation**: Check that binary labels are correct
3. **Class Balance**: Verify class weights are reasonable (not >10:1 ratio)
4. **Val Set Purity**: Ensure NATIX val has no leakage from other datasets

---

## References

- **ROADWork Paper**: Ghosh et al. "ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones" ICCV 2025
- **NATIX**: https://github.com/natixnetwork/streetvision-subnet
- **Roboflow**: https://universe.roboflow.com/workzone/roadwork

---

Last updated: 2025-12-23
"""

    with open("DATA_SOURCES.md", 'w') as f:
        f.write(content)

    print("\n✅ Created DATA_SOURCES.md")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ROADWork and extra roadwork datasets for training"
    )

    parser.add_argument("--download_roadwork", action="store_true", help="Download ROADWork dataset")
    parser.add_argument("--process_roadwork", action="store_true", help="Process ROADWork to binary labels")
    parser.add_argument("--download_extra", action="store_true", help="Download Roboflow extras")
    parser.add_argument("--process_extra", action="store_true", help="Process Roboflow extras")
    parser.add_argument("--create_docs", action="store_true", help="Create DATA_SOURCES.md")

    args = parser.parse_args()

    if args.download_roadwork:
        download_roadwork_iccv()

    if args.process_roadwork:
        process_roadwork_iccv()

    if args.download_extra:
        download_roboflow_roadwork()

    if args.process_extra:
        process_roboflow_roadwork()

    if args.create_docs:
        create_combined_dataset_info()

    if not any([args.download_roadwork, args.process_roadwork, args.download_extra, args.process_extra, args.create_docs]):
        print("\n⚠️  No action specified. Use --help to see options.")
        print("\nQuick start:")
        print("  1. python prepare_roadwork_data.py --download_roadwork --download_extra")
        print("  2. [Manual download steps shown above]")
        print("  3. python prepare_roadwork_data.py --process_roadwork --process_extra --create_docs")


if __name__ == "__main__":
    main()
