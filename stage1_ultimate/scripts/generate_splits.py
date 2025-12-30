#!/usr/bin/env python3
"""
Generate splits.json for NATIX dataset

Creates 4-way stratified split:
- train: 60%
- val_select: 15% (early stopping)
- val_calib: 15% (threshold/calibration)
- val_test: 10% (final eval)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.split_generator import SplitGenerator, SplitConfig
from data.label_schema import LabelSchema

def main():
    # Config paths
    data_root = Path.home() / "data" / "natix_subset"
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_root}")
    if not data_root.exists():
        print(f"ERROR: Data not found at {data_root}")
        print("Please update data_root in this script or download data first")
        sys.exit(1)

    # Load dataset
    label_schema = LabelSchema.from_dataset(str(data_root))
    total = len(label_schema.all_indices)
    print(f"Total samples: {total}")

    if total == 0:
        print("ERROR: No samples found in dataset")
        sys.exit(1)

    # Generate splits
    config = SplitConfig(
        train_ratio=0.60,
        val_select_ratio=0.15,
        val_calib_ratio=0.15,
        val_test_ratio=0.10,
        random_seed=42
    )

    print("\nGenerating stratified splits...")
    generator = SplitGenerator(label_schema, config)
    splits = generator.generate()

    # Save
    output_file = output_dir / "splits.json"
    splits.save(str(output_file))

    print(f"\n✅ Splits saved to: {output_file}")
    print(f"\nSplit Statistics:")
    print(f"  train:      {len(splits.train):>6} samples ({len(splits.train)/total*100:.1f}%)")
    print(f"  val_select: {len(splits.val_select):>6} samples ({len(splits.val_select)/total*100:.1f}%)")
    print(f"  val_calib:  {len(splits.val_calib):>6} samples ({len(splits.val_calib)/total*100:.1f}%)")
    print(f"  val_test:   {len(splits.val_test):>6} samples ({len(splits.val_test)/total*100:.1f}%)")
    print(f"  TOTAL:      {total:>6} samples")

    # Verify no overlap
    all_indices = set(splits.train) | set(splits.val_select) | set(splits.val_calib) | set(splits.val_test)
    assert len(all_indices) == total, "ERROR: Splits have overlaps!"
    print("\n✅ Verified: No overlaps between splits")

    return output_file

if __name__ == "__main__":
    main()
