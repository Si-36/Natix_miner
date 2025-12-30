#!/usr/bin/env python3
"""Check NATIX dataset label distribution."""

from datasets import load_dataset
from collections import Counter

print("Loading NATIX dataset...")
ds = load_dataset("natix-network-org/roadwork")

print("\n" + "="*70)
print("NATIX Dataset Overview")
print("="*70)

for split_name in ['train', 'test']:
    split = ds[split_name]
    print(f"\n{split_name.upper()}: {len(split)} samples")

    # Count label distribution
    labels = [ex['label'] for ex in split]
    label_counts = Counter(labels)

    print(f"  Classes found: {sorted(label_counts.keys())}")
    print(f"  Distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100.0 * count / len(split)
        print(f"    Class {label}: {count:5d} ({pct:5.1f}%)")

    # Check if sorted by class
    is_sorted = all(labels[i] <= labels[i+1] for i in range(len(labels)-1))
    print(f"  Sorted by class: {is_sorted}")

    # Show first 20 labels
    print(f"  First 20 labels: {labels[:20]}")

print("\n" + "="*70)
