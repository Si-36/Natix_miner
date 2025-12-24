#!/usr/bin/env python3
"""
Process NATIX Parquet dataset to image files + CSV labels
"""
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm

print("="*80)
print("PROCESSING NATIX PARQUET DATASET")
print("="*80)

# Directories
natix_dir = "data/natix_official"
data_dir = os.path.join(natix_dir, "data")
train_dir = os.path.join(natix_dir, "train")
val_dir = os.path.join(natix_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def process_split(split_name, output_dir):
    """Process train or test split"""
    print(f"\n{'='*80}")
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print(f"{'='*80}")

    # Find all parquet files for this split
    parquet_files = sorted(Path(data_dir).glob(f"{split_name}-*.parquet"))

    if not parquet_files:
        print(f"❌ No {split_name} parquet files found!")
        return

    print(f"Found {len(parquet_files)} parquet files")

    all_samples = []
    skipped_count = 0

    for parquet_file in tqdm(parquet_files, desc=f"Processing {split_name} parquets"):
        # Read parquet
        df = pd.read_parquet(parquet_file)

        for idx, row in df.iterrows():
            try:
                # Extract image
                if 'image' in row and row['image'] is not None:
                    # Image might be in 'bytes' or 'path' field
                    if isinstance(row['image'], dict) and 'bytes' in row['image']:
                        img_bytes = row['image']['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                    elif isinstance(row['image'], bytes):
                        img = Image.open(io.BytesIO(row['image']))
                    else:
                        skipped_count += 1
                        continue

                    # Generate filename using row.name for consistent indexing
                    img_filename = f"{split_name}_{row.name:09d}.jpg"
                    img_path = os.path.join(output_dir, img_filename)

                    # Save image
                    img.convert('RGB').save(img_path, 'JPEG', quality=95)

                    # Get label (roadwork=1, no roadwork=0)
                    label = int(row.get('label', row.get('roadwork', 0)))

                    all_samples.append((img_filename, label))

            except Exception as e:
                skipped_count += 1
                continue

    # Save labels CSV
    csv_path = os.path.join(natix_dir, f"{split_name}_labels.csv")
    df_labels = pd.DataFrame(all_samples, columns=['image_path', 'label'])
    df_labels.to_csv(csv_path, index=False, header=False)

    print(f"\n✅ Processed {len(all_samples)} {split_name} samples")
    if skipped_count > 0:
        print(f"   ⚠️  Skipped {skipped_count} corrupt/invalid images")
    print(f"   Images: {output_dir}")
    print(f"   Labels: {csv_path}")
    print(f"   Positive (roadwork): {df_labels['label'].sum()} ({100*df_labels['label'].mean():.1f}%)")
    print(f"   Negative (no roadwork): {(df_labels['label']==0).sum()} ({100*(1-df_labels['label'].mean()):.1f}%)")

# Process train split
process_split("train", train_dir)

# Process test split (use as validation)
process_split("test", val_dir)

# Rename test_labels.csv to val_labels.csv
if os.path.exists(os.path.join(natix_dir, "test_labels.csv")):
    os.rename(
        os.path.join(natix_dir, "test_labels.csv"),
        os.path.join(natix_dir, "val_labels.csv")
    )

print(f"\n{'='*80}")
print("✅ NATIX PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"\nNext step: python3 verify_datasets.py --check_natix")
