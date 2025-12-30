#!/usr/bin/env python3
"""Quick script to inspect val_calib labels distribution."""

import torch
import sys
from pathlib import Path
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python inspect_labels.py <phase1_dir>")
    sys.exit(1)

phase1_dir = Path(sys.argv[1])
labels_path = phase1_dir / "val_calib_labels.pt"

print(f"Loading labels from: {labels_path}")
labels = torch.load(labels_path, map_location='cpu')
print(f"Labels shape: {labels.shape}")
print(f"Labels dtype: {labels.dtype}")

# Count label distribution
label_counts = Counter(labels.tolist())
print(f"\nLabel distribution:")
for label in sorted(label_counts.keys()):
    count = label_counts[label]
    pct = 100.0 * count / len(labels)
    print(f"  Class {label}: {count:3d} / {len(labels)} ({pct:5.1f}%)")

print(f"\nAll labels the same? {len(set(labels.tolist())) == 1}")
