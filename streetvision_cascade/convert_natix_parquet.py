#!/usr/bin/env python3
"""
Convert NATIX Hugging Face parquet dataset to individual image files + CSV labels.

Expected input: data/natix_official/data/ (parquet files from Hugging Face download)
Expected output:
- data/natix_official/train/ (image files)
- data/natix_official/train_labels.csv
- data/natix_official/val/ (image files)
- data/natix_official/val_labels.csv
"""

import os
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
import io
from tqdm import tqdm

def convert_natix_dataset():
    """Convert parquet NATIX dataset to image files + CSV labels."""

    base_dir = Path("data/natix_official")
    data_dir = base_dir / "data"

    print("="*80)
    print("CONVERTING NATIX PARQUET DATASET")
    print("="*80)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False

    # Create output directories
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print(f"Created directories: {train_dir}, {val_dir}")

    # Process train files
    print("\nProcessing TRAIN files...")
    train_files = sorted(glob.glob(str(data_dir / "train-*.parquet")))
    train_labels = process_parquet_files(train_files, train_dir, "train")

    # Process test files as validation
    print("\nProcessing TEST files (as validation)...")
    test_files = sorted(glob.glob(str(data_dir / "test-*.parquet")))
    val_labels = process_parquet_files(test_files, val_dir, "val")

    # Save CSV files
    print("\nSaving label files...")
    train_df = pd.DataFrame(train_labels)
    val_df = pd.DataFrame(val_labels)

    train_csv = base_dir / "train_labels.csv"
    val_csv = base_dir / "val_labels.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"✅ Saved {len(train_df)} train labels to {train_csv}")
    print(f"✅ Saved {len(val_df)} val labels to {val_csv}")

    # Show statistics
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)

    train_pos = train_df['label'].sum()
    train_neg = len(train_df) - train_pos
    val_pos = val_df['label'].sum()
    val_neg = len(val_df) - val_pos

    print(f"Train: {len(train_df)} images ({train_pos} positive, {train_neg} negative)")
    print(f"Val:   {len(val_df)} images ({val_pos} positive, {val_neg} negative)")

    return True

def process_parquet_files(parquet_files, output_dir, split_name):
    """Process parquet files and save images + collect labels."""

    labels = []
    global_idx = 0

    for file_path in tqdm(parquet_files, desc=f"Processing {split_name} files"):
        try:
            # Read parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()

            for _, row in df.iterrows():
                try:
                    # Get image bytes and convert to PIL Image
                    image_bytes = row['image']['bytes']
                    image = Image.open(io.BytesIO(image_bytes))

                    # Get label
                    label = row['label']
                    image_id = row['id']

                    # Create filename (zero-padded for sorting)
                    filename = "06d"

                    # Save image
                    image_path = output_dir / filename
                    image.save(image_path)

                    # Collect label info
                    labels.append({
                        'filename': filename,
                        'label': label
                    })

                    global_idx += 1

                except Exception as e:
                    print(f"⚠️  Error processing row in {file_path}: {e}")
                    continue

        except Exception as e:
            print(f"⚠️  Error processing file {file_path}: {e}")
            continue

    return labels

if __name__ == "__main__":
    success = convert_natix_dataset()
    if success:
        print("\n🎉 NATIX dataset conversion complete!")
        print("You can now run: python3 verify_datasets.py --check_natix")
    else:
        print("\n❌ Conversion failed!")
        exit(1)
