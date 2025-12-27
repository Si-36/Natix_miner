#!/usr/bin/env python3
"""
Phase 1: Create val_select/val_calib splits deterministically.
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

from stage1_pro.stage1_pro.data import create_val_splits, save_splits, NATIXDataset
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natix_val_dir", type=str, required=True)
    parser.add_argument("--val_select_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="splits.json")
    args = parser.parse_args()

    print(f"Loading NATIX validation data from {args.natix_val_dir}...")

    # Load validation set
    from pathlib import Path

    labels_path = Path(args.natix_val_dir) / "labels.json"

    if not labels_path.exists():
        print(f"WARNING: labels.json not found at {labels_path}")
        return

    val_dataset = NATIXDataset(root=args.natix_val_dir, labels_path=str(labels_path))

    # Create splits
    indices = np.arange(len(val_dataset))
    train_idx, val_select_idx, val_calib_idx = create_val_splits(
        indices, val_select_ratio=args.val_select_ratio, seed=args.seed
    )

    # Note: In Phase 1, we only need val splits
    splits_dict = {
        "val_select": val_select_idx,
        "val_calib": val_calib_idx,
        "train": train_idx,
        "seed": args.seed,
    }

    save_splits(splits_dict, args.output)
    print(f"\nSplits created:")
    print(f"  Val Select: {len(val_select_idx)} samples")
    print(f"  Val Calib: {len(val_calib_idx)} samples")
    print(f"  Total Val: {len(val_select_idx) + len(val_calib_idx)} samples")


if __name__ == "__main__":
    main()
