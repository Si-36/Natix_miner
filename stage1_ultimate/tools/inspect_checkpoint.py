#!/usr/bin/env python3
"""Inspect checkpoint to understand NaN issue."""

import torch
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

ckpt_path = Path(sys.argv[1])
print(f"Loading checkpoint: {ckpt_path}")

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\nCheckpoint keys: {list(ckpt.keys())}")

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    print(f"\nState dict has {len(state_dict)} keys")

    # Check for NaN in state dict
    nan_params = []
    for name, param in state_dict.items():
        if torch.isnan(param).any():
            nan_params.append(name)

    if nan_params:
        print(f"\n❌ Found NaN in {len(nan_params)} parameters:")
        for name in nan_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(nan_params) > 10:
            print(f"  ... and {len(nan_params) - 10} more")
    else:
        print(f"\n✅ No NaN found in state_dict")

# Check EMA state
if 'ema_state' in ckpt:
    ema_state = ckpt['ema_state']
    print(f"\nEMA state keys: {list(ema_state.keys())}")

    if 'shadow' in ema_state:
        shadow = ema_state['shadow']
        print(f"EMA shadow has {len(shadow)} keys")

        nan_shadow = []
        for name, param in shadow.items():
            if torch.isnan(param).any():
                nan_shadow.append(name)

        if nan_shadow:
            print(f"\n❌ Found NaN in {len(nan_shadow)} EMA shadow parameters:")
            for name in nan_shadow[:10]:
                print(f"  - {name}")
            if len(nan_shadow) > 10:
                print(f"  ... and {len(nan_shadow) - 10} more")
        else:
            print(f"\n✅ No NaN found in EMA shadow")

# Check optimizer state
if 'optimizer_states' in ckpt:
    print(f"\n✅ Checkpoint has optimizer states")

if 'epoch' in ckpt:
    print(f"\nEpoch: {ckpt['epoch']}")

if 'global_step' in ckpt:
    print(f"Global step: {ckpt['global_step']}")

print("\n" + "="*70)
