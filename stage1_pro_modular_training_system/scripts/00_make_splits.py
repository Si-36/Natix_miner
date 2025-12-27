"""
Script to create deterministic 4-way splits (train/val_select/val_calib/val_test)

CRITICAL: Document usage rules (val_select=model selection, val_calib=calibration, val_test=evaluation).
"""

import argparse
import json
from pathlib import Path

from ..config import Stage1ProConfig
from ..data.datasets import NATIXDataset
from ..data.splits import create_val_splits, save_splits, compute_split_metadata
from transformers import AutoImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Create deterministic 4-way splits")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--val_labels_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_select_ratio", type=float, default=0.33)
    parser.add_argument("--val_calib_ratio", type=float, default=0.33)
    parser.add_argument("--val_test_ratio", type=float, default=0.34)
    
    args = parser.parse_args()
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(args.model_path)
    
    # Load val dataset
    val_dataset = NATIXDataset(
        image_dir=args.val_image_dir,
        labels_file=args.val_labels_file,
        processor=processor,
        augment=False
    )
    
    print(f"Loaded validation dataset: {len(val_dataset)} samples")
    
    # Create splits
    splits = create_val_splits(
        val_dataset,
        val_select_ratio=args.val_select_ratio,
        val_calib_ratio=args.val_calib_ratio,
        val_test_ratio=args.val_test_ratio,
        seed=args.seed
    )
    
    print(f"\nSplit sizes:")
    print(f"  val_select: {len(splits['val_select'])} samples")
    print(f"  val_calib: {len(splits['val_calib'])} samples")
    print(f"  val_test: {len(splits['val_test'])} samples")
    
    # Compute metadata
    metadata = compute_split_metadata(val_dataset, splits)
    metadata['seed'] = args.seed
    metadata['val_select_ratio'] = args.val_select_ratio
    metadata['val_calib_ratio'] = args.val_calib_ratio
    metadata['val_test_ratio'] = args.val_test_ratio
    
    # Save splits
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    splits_path = Path(args.output_dir) / "splits.json"
    save_splits(splits, str(splits_path), metadata)
    
    print(f"\n✅ Splits saved to {splits_path}")
    print(f"\nUsage rules:")
    print(f"  - val_select: Model selection/early stopping ONLY")
    print(f"  - val_calib: Fitting calibrators/policies ONLY")
    print(f"  - val_test: Final evaluation ONLY")


if __name__ == "__main__":
    main()
