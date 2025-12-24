#!/usr/bin/env python3
"""
Dataset Verification Script (2025 Pro)

Verifies all dataset paths and structures BEFORE training.
Catches path mismatches, missing files, and incorrect CSV formats early.

Usage:
    python3 verify_datasets.py --check_all
    python3 verify_datasets.py --check_natix
    python3 verify_datasets.py --check_extra
"""
import os
import argparse
import pandas as pd
from pathlib import Path

def check_dataset(name, image_dir, labels_csv, expected_min_samples=0):
    """
    Verify a single dataset's structure and files.

    Returns: (success: bool, num_samples: int, errors: list)
    """
    errors = []

    print(f"\n{'='*80}")
    print(f"CHECKING: {name}")
    print(f"{'='*80}")

    # Check labels CSV exists
    if not os.path.exists(labels_csv):
        errors.append(f"❌ Labels file not found: {labels_csv}")
        return False, 0, errors

    print(f"✅ Labels CSV exists: {labels_csv}")

    # Load CSV (no header, columns: image_path, label)
    try:
        df = pd.read_csv(labels_csv, header=None, names=['image_path', 'label'])
    except Exception as e:
        errors.append(f"❌ Failed to read CSV: {e}")
        return False, 0, errors

    print(f"✅ Loaded {len(df)} samples from CSV")

    # Check image directory exists
    if not os.path.exists(image_dir):
        errors.append(f"❌ Image directory not found: {image_dir}")
        return False, len(df), errors

    print(f"✅ Image directory exists: {image_dir}")

    # Check first 5 image paths exist
    print(f"\nVerifying image paths (first 5)...")
    missing_count = 0

    for idx, row in df.head(5).iterrows():
        img_path = row['image_path']

        # Try multiple path variations
        possible_paths = [
            img_path,  # Absolute path
            os.path.join(image_dir, img_path),  # Relative to image_dir
            os.path.join(image_dir, os.path.basename(img_path)),  # Just filename
        ]

        found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  ✅ {os.path.basename(path)}")
                found = True
                break

        if not found:
            print(f"  ❌ NOT FOUND: {img_path}")
            print(f"     Tried: {possible_paths}")
            missing_count += 1

    if missing_count > 0:
        errors.append(f"❌ {missing_count}/5 sample images not found - check path format in CSV")

    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"\n📊 Label distribution:")
    print(f"   Class 0 (no roadwork): {label_counts.get(0, 0)} ({100*label_counts.get(0, 0)/len(df):.1f}%)")
    print(f"   Class 1 (roadwork):    {label_counts.get(1, 0)} ({100*label_counts.get(1, 0)/len(df):.1f}%)")

    # Warn if all positives or all negatives
    if label_counts.get(0, 0) == 0:
        print(f"   ⚠️  WARNING: All samples are positives (label=1)")
        print(f"      This is expected for Open Images/Roboflow/GTSRB/ROADWork (positives boosters)")

    if label_counts.get(1, 0) == 0:
        # Kaggle Road Issues is intentionally all negatives (road problems, not roadwork)
        if "kaggle" in name.lower() or "road issues" in name.lower():
            print(f"   ⚠️  WARNING: All samples are negatives (label=0)")
            print(f"      This is CORRECT for Kaggle Road Issues (negatives booster)")
        else:
            errors.append(f"❌ All samples are negatives (label=0) - likely wrong labels")

    # Check minimum sample count
    if expected_min_samples > 0 and len(df) < expected_min_samples:
        errors.append(f"⚠️  Expected at least {expected_min_samples} samples, got {len(df)}")

    success = len(errors) == 0
    return success, len(df), errors


def check_natix():
    """Verify NATIX dataset (train + val)"""
    print("\n" + "="*80)
    print("NATIX DATASET VERIFICATION")
    print("="*80)

    # Check train
    train_success, train_count, train_errors = check_dataset(
        "NATIX Train",
        "data/natix_official/train",
        "data/natix_official/train_labels.csv",
        expected_min_samples=6000
    )

    # Check val
    val_success, val_count, val_errors = check_dataset(
        "NATIX Val",
        "data/natix_official/val",
        "data/natix_official/val_labels.csv",
        expected_min_samples=2000
    )

    # Summary
    print("\n" + "="*80)
    print("NATIX SUMMARY")
    print("="*80)

    if train_success and val_success:
        print(f"✅ NATIX dataset OK!")
        print(f"   Train: {train_count} samples")
        print(f"   Val:   {val_count} samples")
        return True
    else:
        print(f"❌ NATIX dataset has errors:")
        for err in train_errors + val_errors:
            print(f"   {err}")
        return False


def check_extra_datasets():
    """Verify external datasets (ROADWork, Open Images, Roboflow, GTSRB)"""
    print("\n" + "="*80)
    print("EXTERNAL DATASETS VERIFICATION")
    print("="*80)

    datasets = [
        ("ROADWork", "data/roadwork_iccv/raw/images", "data/roadwork_iccv/train_labels.csv", 2000),
        ("Open Images V7", "data/open_images/coco/data", "data/open_images/train_labels.csv", 1000),
        ("Roboflow", "data/roadwork_extra/raw/train", "data/roadwork_extra/train_labels.csv", 300),
        ("GTSRB Class 25", "data/gtsrb_class25/train_images", "data/gtsrb_class25/train_labels.csv", 400),
        ("Kaggle Road Issues", "data/kaggle_road_issues/images", "data/kaggle_road_issues/train_labels.csv", 5000),
    ]

    results = {}
    total_samples = 0

    for name, img_dir, labels_csv, min_samples in datasets:
        if os.path.exists(labels_csv):
            success, count, errors = check_dataset(name, img_dir, labels_csv, min_samples)
            results[name] = (success, count, errors)
            total_samples += count
        else:
            print(f"\n⚠️  {name} not found (skipping): {labels_csv}")
            results[name] = (None, 0, [])

    # Summary
    print("\n" + "="*80)
    print("EXTERNAL DATASETS SUMMARY")
    print("="*80)

    for name, (success, count, errors) in results.items():
        if success is None:
            print(f"⚠️  {name}: Not downloaded")
        elif success:
            print(f"✅ {name}: {count} samples")
        else:
            print(f"❌ {name}: {count} samples (HAS ERRORS)")
            for err in errors:
                print(f"     {err}")

    print(f"\n📊 Total external samples: {total_samples}")

    return all(s is None or s for s, _, _ in results.values())


def main():
    parser = argparse.ArgumentParser(description="Verify dataset structures before training")
    parser.add_argument("--check_all", action="store_true", help="Check all datasets")
    parser.add_argument("--check_natix", action="store_true", help="Check NATIX only")
    parser.add_argument("--check_extra", action="store_true", help="Check external datasets only")

    args = parser.parse_args()

    if not any([args.check_all, args.check_natix, args.check_extra]):
        args.check_all = True  # Default to check all

    natix_ok = True
    extra_ok = True

    if args.check_all or args.check_natix:
        natix_ok = check_natix()

    if args.check_all or args.check_extra:
        extra_ok = check_extra_datasets()

    # Final summary
    print("\n" + "="*80)
    print("FINAL VERIFICATION SUMMARY")
    print("="*80)

    if natix_ok and extra_ok:
        print("✅ All datasets verified successfully!")
        print("\nYou can now run training:")
        print("  Baseline:   python3 train_stage1_head.py --mode train --epochs 10")
        print("  Aggressive: python3 train_stage1_head.py --mode train --epochs 15 --use_extra_roadwork")
    elif natix_ok and not extra_ok:
        print("✅ NATIX OK, but some external datasets have errors")
        print("\nYou can run baseline training (NATIX only):")
        print("  python3 train_stage1_head.py --mode train --epochs 10")
        print("\n⚠️  Fix external dataset errors before using --use_extra_roadwork")
    else:
        print("❌ Critical errors found - fix before training!")
        print("\nRe-run this script after fixing issues:")
        print("  python3 verify_datasets.py --check_all")


if __name__ == "__main__":
    main()
