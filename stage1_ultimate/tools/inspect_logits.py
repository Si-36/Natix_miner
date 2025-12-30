#!/usr/bin/env python3
"""Quick script to inspect val_calib logits and understand threshold sweep results."""

import torch
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_logits.py <phase1_dir>")
    sys.exit(1)

phase1_dir = Path(sys.argv[1])
logits_path = phase1_dir / "val_calib_logits.pt"
labels_path = phase1_dir / "val_calib_labels.pt"

print(f"Loading logits from: {logits_path}")
logits = torch.load(logits_path, map_location='cpu')
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"\nLogits stats:")
print(f"  Min: {logits.min().item():.4f}")
print(f"  Max: {logits.max().item():.4f}")
print(f"  Mean: {logits.mean().item():.4f}")
print(f"  Contains NaN: {torch.isnan(logits).any().item()}")
print(f"  Contains Inf: {torch.isinf(logits).any().item()}")

# Apply softmax to get probabilities
probs = torch.softmax(logits, dim=1)
print(f"\nProbabilities (after softmax):")
print(f"  Min: {probs.min().item():.4f}")
print(f"  Max: {probs.max().item():.4f}")
print(f"  Mean: {probs.mean().item():.4f}")

# Get max probabilities (confidence scores)
max_probs, pred_classes = probs.max(dim=1)
print(f"\nMax probabilities (confidence scores):")
print(f"  Min: {max_probs.min().item():.4f}")
print(f"  Max: {max_probs.max().item():.4f}")
print(f"  Mean: {max_probs.mean().item():.4f}")
print(f"  Median: {max_probs.median().item():.4f}")

print(f"\nDistribution of max probs:")
for threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    count = (max_probs >= threshold).sum().item()
    pct = 100.0 * count / len(max_probs)
    print(f"  >= {threshold:.2f}: {count:3d} / {len(max_probs)} ({pct:5.1f}%)")

# Check accuracy
labels = torch.load(labels_path, map_location='cpu')
correct = (pred_classes == labels).sum().item()
total = len(labels)
acc = 100.0 * correct / total
print(f"\nAccuracy on val_calib: {correct}/{total} = {acc:.1f}%")

print(f"\nFirst 5 examples:")
for i in range(min(5, len(logits))):
    print(f"  Example {i}: pred={pred_classes[i].item()} (conf={max_probs[i].item():.4f}), true={labels[i].item()}")
