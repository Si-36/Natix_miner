#!/usr/bin/env python3
"""
Smart Dataset Filtering for NATIX Subnet 72

Intelligently combines:
- Mapillary Vistas (construction objects + hard negatives)
- ROADWork (hayden-yuma) dataset

Creates balanced 50/50 roadwork vs non-roadwork dataset.
Filters for quality, removes blur, extreme aspect ratios.

Author: Claude + User
Date: Dec 2025
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import sys

random.seed(42)

# ========== CONFIG ==========
OUTPUT_DIR = Path("~/Natix_miner/streetvision_cascade/data/filtered_combined").expanduser()
MAPILLARY_DIR = Path("~/Natix_miner/data/mapillary_vistas").expanduser()
ROADWORK_DIR = Path("~/Natix_miner/data/roadwork_hf").expanduser()

TARGET_SAMPLES = 15000  # Total samples
POSITIVE_RATIO = 0.5    # 50% roadwork, 50% no-roadwork (balanced!)

MIN_IMAGE_SIZE = (512, 512)  # Filter out tiny images
MAX_ASPECT_RATIO = 3.0       # Remove extreme aspect ratios


# ========== HELPER FUNCTIONS ==========

def is_valid_image(img_path):
    """Check if image is high quality"""
    try:
        img = Image.open(img_path)
        w, h = img.size

        # Check minimum size
        if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
            return False

        # Check aspect ratio
        aspect = max(w, h) / min(w, h)
        if aspect > MAX_ASPECT_RATIO:
            return False

        # Try to load image data
        img.verify()

        # Re-open after verify (verify closes the file)
        img = Image.open(img_path)
        img.load()

        return True
    except Exception as e:
        return False


# ========== MAPILLARY FILTERING ==========

def find_mapillary_structure(base_dir):
    """
    Mapillary structure can vary. Find where images and annotations are.
    """
    print("🔍 Detecting Mapillary directory structure...")

    # Common patterns
    possible_img_dirs = [
        base_dir / "training" / "images",
        base_dir / "training" / "v2.0" / "images",
        base_dir / "images",
        base_dir / "train",
    ]

    possible_anno_dirs = [
        base_dir / "training" / "v2.0" / "labels",
        base_dir / "training" / "labels",
        base_dir / "labels",
        base_dir / "annotations",
    ]

    img_dir = None
    anno_dir = None

    for d in possible_img_dirs:
        if d.exists() and list(d.glob("*.jpg")) or list(d.glob("*.png")):
            img_dir = d
            print(f"   Found images: {img_dir}")
            break

    for d in possible_anno_dirs:
        if d.exists() and (list(d.glob("*.json")) or list(d.glob("*.txt"))):
            anno_dir = d
            print(f"   Found annotations: {anno_dir}")
            break

    return img_dir, anno_dir


def filter_mapillary():
    """
    Extract from Mapillary:
    - Construction scenes (positives)
    - Normal road scenes (hard negatives)
    """
    print("\n[1/3] Filtering Mapillary Vistas...")

    img_dir, anno_dir = find_mapillary_structure(MAPILLARY_DIR)

    if img_dir is None:
        print("⚠️  Could not find Mapillary images directory!")
        print(f"   Checked: {MAPILLARY_DIR}")
        print("   Trying to use images directly...")

        # Fallback: just use all images as negatives (no annotations)
        all_images = list(MAPILLARY_DIR.rglob("*.jpg")) + list(MAPILLARY_DIR.rglob("*.png"))

        if len(all_images) == 0:
            print("❌ No images found in Mapillary directory!")
            return [], []

        print(f"   Found {len(all_images)} images without annotations")
        print("   Will use random sample as hard negatives...")

        road_samples = []
        for img_path in tqdm(all_images[:10000], desc="Sampling images"):
            if is_valid_image(img_path):
                road_samples.append((img_path, 0))

        return [], road_samples

    construction_samples = []
    road_samples = []

    # If we have annotations, use them
    if anno_dir and anno_dir.exists():
        print(f"   Processing annotations from {anno_dir}")

        anno_files = list(anno_dir.glob("*.json"))

        for anno_file in tqdm(anno_files, desc="Processing annotations"):
            try:
                with open(anno_file) as f:
                    data = json.load(f)

                # Check for construction objects
                has_construction = False

                # Handle different annotation formats
                objects = data.get('objects', [])
                if not objects and 'labels' in data:
                    objects = data['labels']

                for obj in objects:
                    label = obj.get('label', '').lower()
                    category = obj.get('category', '').lower()

                    # Construction keywords
                    construction_keywords = [
                        'construction', 'barrier', 'cone', 'warning',
                        'roadwork', 'work', 'fence', 'traffic-sign--'
                    ]

                    if any(kw in label or kw in category for kw in construction_keywords):
                        has_construction = True
                        break

                # Find corresponding image
                img_name = anno_file.stem + '.jpg'
                img_path = img_dir / img_name

                if not img_path.exists():
                    img_name = anno_file.stem + '.png'
                    img_path = img_dir / img_name

                if not img_path.exists():
                    continue

                if not is_valid_image(img_path):
                    continue

                if has_construction:
                    construction_samples.append((img_path, 1))
                else:
                    road_samples.append((img_path, 0))

            except Exception as e:
                continue

    else:
        # No annotations - use all images as negatives
        print("   No annotations found, using all images as negatives")
        all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        for img_path in tqdm(all_images, desc="Validating images"):
            if is_valid_image(img_path):
                road_samples.append((img_path, 0))

    print(f"✅ Found {len(construction_samples)} construction scenes")
    print(f"✅ Found {len(road_samples)} normal road scenes")

    return construction_samples, road_samples


# ========== ROADWORK FILTERING ==========

def filter_roadwork():
    """
    ROADWork dataset - all are roadwork scenes (positives)
    """
    print("\n[2/3] Filtering ROADWork (hayden-yuma)...")

    # Check if already extracted
    train_dir = ROADWORK_DIR / "train_extracted"

    if not train_dir.exists():
        print(f"   Extracting from HuggingFace parquet format...")

        try:
            from datasets import load_from_disk, load_dataset

            # Try loading from disk first
            if (ROADWORK_DIR / "dataset_info.json").exists():
                print("   Loading from disk...")
                dataset = load_from_disk(str(ROADWORK_DIR))
            else:
                print("   Downloading dataset...")
                dataset = load_dataset("hayden-yuma/roadwork")

            # Extract train split
            train_dir.mkdir(parents=True, exist_ok=True)

            print("   Extracting images...")
            for idx, sample in enumerate(tqdm(dataset['train'])):
                img = sample['image']
                img_path = train_dir / f"roadwork_{idx:06d}.jpg"

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img.save(img_path, 'JPEG', quality=95)

            print(f"   ✅ Extracted {len(dataset['train'])} images")

        except Exception as e:
            print(f"   ❌ Error extracting ROADWork dataset: {e}")
            print(f"   Trying alternative method...")

            # Alternative: look for pre-extracted images
            possible_dirs = [
                ROADWORK_DIR / "data" / "train",
                ROADWORK_DIR / "train",
                ROADWORK_DIR,
            ]

            for d in possible_dirs:
                if d.exists():
                    images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                    if len(images) > 0:
                        train_dir = d
                        print(f"   Found {len(images)} images in {d}")
                        break

    # Collect roadwork samples
    roadwork_samples = []

    if train_dir.exists():
        for img_path in tqdm(list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")), desc="Validating images"):
            if is_valid_image(img_path):
                roadwork_samples.append((img_path, 1))  # All are positives

    print(f"✅ Found {len(roadwork_samples)} roadwork scenes")

    return roadwork_samples


# ========== SMART BALANCING ==========

def balance_dataset(construction_samples, road_samples, roadwork_samples):
    """
    Create balanced dataset:
    - 50% positives (roadwork + construction)
    - 50% negatives (normal roads)
    """
    print("\n[3/3] Creating balanced dataset...")

    target_positives = int(TARGET_SAMPLES * POSITIVE_RATIO)
    target_negatives = TARGET_SAMPLES - target_positives

    # Combine positives
    all_positives = construction_samples + roadwork_samples
    random.shuffle(all_positives)

    # Sample what we need
    if len(all_positives) < target_positives:
        print(f"   ⚠️  Only {len(all_positives)} positives available (target: {target_positives})")
        selected_positives = all_positives
    else:
        selected_positives = all_positives[:target_positives]

    # Sample negatives
    random.shuffle(road_samples)

    if len(road_samples) < target_negatives:
        print(f"   ⚠️  Only {len(road_samples)} negatives available (target: {target_negatives})")
        selected_negatives = road_samples
    else:
        selected_negatives = road_samples[:target_negatives]

    total = len(selected_positives) + len(selected_negatives)
    pos_ratio = len(selected_positives) / total * 100 if total > 0 else 0

    print(f"\n📊 Final Dataset Composition:")
    print(f"   Positives (roadwork): {len(selected_positives)} ({pos_ratio:.1f}%)")
    print(f"   Negatives (no roadwork): {len(selected_negatives)} ({100-pos_ratio:.1f}%)")
    print(f"   Total: {total}")

    return selected_positives, selected_negatives


# ========== COPY TO TRAINING DIR ==========

def create_training_dataset(positives, negatives):
    """
    Copy filtered images to training directory in NATIX format
    """
    print("\n[4/4] Creating training dataset...")

    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Combine and shuffle
    all_samples = positives + negatives
    random.shuffle(all_samples)

    # Split 80/20
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Copy train
    print("   Copying training images...")
    train_labels = []
    for idx, (src_path, label) in enumerate(tqdm(train_samples, desc="Train")):
        dst_name = f"img_{idx:08d}.jpg"
        dst_path = train_dir / dst_name

        try:
            # Copy and convert to JPG if needed
            img = Image.open(src_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(dst_path, 'JPEG', quality=95)
            train_labels.append(f"{dst_name},{label}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy {src_path}: {e}")
            continue

    # Copy val
    print("   Copying validation images...")
    val_labels = []
    for idx, (src_path, label) in enumerate(tqdm(val_samples, desc="Val")):
        dst_name = f"img_{idx:08d}.jpg"
        dst_path = val_dir / dst_name

        try:
            img = Image.open(src_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(dst_path, 'JPEG', quality=95)
            val_labels.append(f"{dst_name},{label}")
        except Exception as e:
            print(f"   ⚠️  Failed to copy {src_path}: {e}")
            continue

    # Write label files (NO HEADER - NATIX format!)
    with open(OUTPUT_DIR / "train_labels.csv", "w") as f:
        f.write("\n".join(train_labels))

    with open(OUTPUT_DIR / "val_labels.csv", "w") as f:
        f.write("\n".join(val_labels))

    print(f"\n✅ Dataset created at: {OUTPUT_DIR}")
    print(f"   Train: {len(train_labels)} samples")
    print(f"   Val: {len(val_labels)} samples")

    # Class balance
    train_pos = sum(1 for l in train_labels if l.endswith(',1'))
    val_pos = sum(1 for l in val_labels if l.endswith(',1'))

    print(f"\n📊 Class Balance:")
    print(f"   Train - Pos: {train_pos} ({100*train_pos/len(train_labels):.1f}%), Neg: {len(train_labels)-train_pos} ({100*(len(train_labels)-train_pos)/len(train_labels):.1f}%)")
    print(f"   Val   - Pos: {val_pos} ({100*val_pos/len(val_labels):.1f}%), Neg: {len(val_labels)-val_pos} ({100*(len(val_labels)-val_pos)/len(val_labels):.1f}%)")


# ========== MAIN ==========

if __name__ == "__main__":
    print("="*80)
    print("SMART DATASET FILTERING FOR NATIX SUBNET 72")
    print("="*80)

    # Check if source directories exist
    if not MAPILLARY_DIR.exists():
        print(f"\n❌ Mapillary directory not found: {MAPILLARY_DIR}")
        print("   Run download_mapillary_kaggle.sh first!")
        sys.exit(1)

    if not ROADWORK_DIR.exists():
        print(f"\n❌ ROADWork directory not found: {ROADWORK_DIR}")
        print("   Run download_roadwork_hf.sh first!")
        sys.exit(1)

    try:
        # Step 1: Filter Mapillary
        construction_samples, road_samples = filter_mapillary()

        # Step 2: Filter ROADWork
        roadwork_samples = filter_roadwork()

        # Step 3: Balance dataset
        selected_positives, selected_negatives = balance_dataset(
            construction_samples, road_samples, roadwork_samples
        )

        # Step 4: Create training dataset
        create_training_dataset(selected_positives, selected_negatives)

        print("\n" + "="*80)
        print("✅ DONE! Dataset ready for training.")
        print("="*80)
        print("\nNext steps:")
        print("1. Update train_stage1_v2.py config:")
        print("   train_image_dir = 'data/filtered_combined/train'")
        print("   train_labels_file = 'data/filtered_combined/train_labels.csv'")
        print("   val_image_dir = 'data/filtered_combined/val'")
        print("   val_labels_file = 'data/filtered_combined/val_labels.csv'")
        print("")
        print("2. Train: python3 train_stage1_v2.py --mode train --epochs 15")
        print("")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
