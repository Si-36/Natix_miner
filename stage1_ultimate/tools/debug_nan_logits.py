#!/usr/bin/env python3
"""Debug NaN logits issue in validation."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.module import StreetVisionModule
from data.datamodule import StreetVisionDataModule
from omegaconf import OmegaConf

print("="*70)
print("Debugging NaN Logits Issue")
print("="*70)

# Load config
config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
cfg = OmegaConf.load(config_path)

# Override with test settings
cfg.data.data_root = "/home/sina/data/natix_subset"
cfg.data.dataloader.batch_size = 2
cfg.data.dataloader.val_batch_size = 2
cfg.model.use_multiview = True
cfg.model.use_ema = True
cfg.training.mixed_precision.enabled = True

print(f"\nTest Configuration:")
print(f"  Data root: {cfg.data.data_root}")
print(f"  Multi-view: {cfg.model.use_multiview}")
print(f"  EMA: {cfg.model.use_ema}")
print(f"  FP16: {cfg.training.mixed_precision.enabled}")

# Create datamodule
print("\nCreating datamodule...")
datamodule = StreetVisionDataModule(cfg)
datamodule.setup(stage='fit')

# Create model
print("\nCreating model...")
model = StreetVisionModule(cfg)
model = model.cuda()
model.eval()

# Test 1: Single-view forward pass (no multi-view, no EMA)
print("\n" + "="*70)
print("TEST 1: Single-view forward (no multi-view, no EMA)")
print("="*70)

val_calib_loader = datamodule.val_calib_dataloader()
batch = next(iter(val_calib_loader))

if len(batch) == 3:
    images, labels, content_boxes = batch
else:
    images, labels = batch
    content_boxes = None

images = images.cuda()
labels = labels.cuda()
if content_boxes is not None:
    content_boxes = content_boxes.cuda()

print(f"Batch: images={images.shape}, labels={labels.shape}")
if content_boxes is not None:
    print(f"       content_boxes={content_boxes.shape}")

# Forward pass (single-view, no EMA)
with torch.no_grad():
    logits = model.forward(images, use_multiview=False, content_boxes=None)

print(f"\nSingle-view logits:")
print(f"  Shape: {logits.shape}")
print(f"  Min: {logits.min().item():.4f}")
print(f"  Max: {logits.max().item():.4f}")
print(f"  Mean: {logits.mean().item():.4f}")
print(f"  Contains NaN: {torch.isnan(logits).any().item()}")
print(f"  Contains Inf: {torch.isinf(logits).any().item()}")

# Test 2: Multi-view forward pass (with multi-view, no EMA)
print("\n" + "="*70)
print("TEST 2: Multi-view forward (with multi-view, no EMA)")
print("="*70)

if model.multiview is None:
    print("  SKIPPED: Multi-view not enabled")
else:
    with torch.no_grad():
        logits_mv = model.forward(images, use_multiview=True, content_boxes=content_boxes)

    print(f"\nMulti-view logits:")
    print(f"  Shape: {logits_mv.shape}")
    print(f"  Min: {logits_mv.min().item():.4f}")
    print(f"  Max: {logits_mv.max().item():.4f}")
    print(f"  Mean: {logits_mv.mean().item():.4f}")
    print(f"  Contains NaN: {torch.isnan(logits_mv).any().item()}")
    print(f"  Contains Inf: {torch.isinf(logits_mv).any().item()}")

# Test 3: Multi-view with EMA
print("\n" + "="*70)
print("TEST 3: Multi-view forward (with multi-view, with EMA)")
print("="*70)

if model.ema is None:
    print("  SKIPPED: EMA not enabled")
elif model.multiview is None:
    print("  SKIPPED: Multi-view not enabled")
else:
    with torch.no_grad():
        with model.ema.average_parameters():
            logits_mv_ema = model.forward(images, use_multiview=True, content_boxes=content_boxes)

    print(f"\nMulti-view + EMA logits:")
    print(f"  Shape: {logits_mv_ema.shape}")
    print(f"  Min: {logits_mv_ema.min().item():.4f}")
    print(f"  Max: {logits_mv_ema.max().item():.4f}")
    print(f"  Mean: {logits_mv_ema.mean().item():.4f}")
    print(f"  Contains NaN: {torch.isnan(logits_mv_ema).any().item()}")
    print(f"  Contains Inf: {torch.isinf(logits_mv_ema).any().item()}")

# Test 4: Inspect multi-view components
if model.multiview is not None:
    print("\n" + "="*70)
    print("TEST 4: Multi-view component diagnostics")
    print("="*70)

    with torch.no_grad():
        # Generate crops
        print("\nGenerating crops...")
        crops = model.multiview.generator(images, content_boxes=content_boxes)
        print(f"  Crops shape: {crops.shape}")
        print(f"  Crops min: {crops.min().item():.4f}")
        print(f"  Crops max: {crops.max().item():.4f}")
        print(f"  Contains NaN: {torch.isnan(crops).any().item()}")
        print(f"  Contains Inf: {torch.isinf(crops).any().item()}")

        # Flatten crops
        B, num_crops, C, H, W = crops.shape
        crops_flat = crops.view(B * num_crops, C, H, W)
        print(f"\n  Flattened crops: {crops_flat.shape}")

        # Forward through backbone
        print("\nBackbone forward pass...")
        features = model.multiview.backbone(crops_flat)
        print(f"  Features shape: {features.shape}")
        print(f"  Features min: {features.min().item():.4f}")
        print(f"  Features max: {features.max().item():.4f}")
        print(f"  Contains NaN: {torch.isnan(features).any().item()}")
        print(f"  Contains Inf: {torch.isinf(features).any().item()}")

        # Forward through head
        print("\nHead forward pass...")
        logits_per_crop = model.multiview.head(features)
        print(f"  Logits shape: {logits_per_crop.shape}")
        print(f"  Logits min: {logits_per_crop.min().item():.4f}")
        print(f"  Logits max: {logits_per_crop.max().item():.4f}")
        print(f"  Contains NaN: {torch.isnan(logits_per_crop).any().item()}")
        print(f"  Contains Inf: {torch.isinf(logits_per_crop).any().item()}")

        # Reshape and aggregate
        print("\nAggregation...")
        num_classes = logits_per_crop.size(-1)
        logits_per_crop = logits_per_crop.view(B, num_crops, num_classes)
        print(f"  Reshaped logits: {logits_per_crop.shape}")

        aggregated = model.multiview.aggregator(logits_per_crop)
        print(f"  Aggregated logits: {aggregated.shape}")
        print(f"  Aggregated min: {aggregated.min().item():.4f}")
        print(f"  Aggregated max: {aggregated.max().item():.4f}")
        print(f"  Contains NaN: {torch.isnan(aggregated).any().item()}")
        print(f"  Contains Inf: {torch.isinf(aggregated).any().item()}")

print("\n" + "="*70)
print("Diagnostic complete!")
print("="*70)
