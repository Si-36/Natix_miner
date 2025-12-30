#!/bin/bash
# Local smoke test - run this BEFORE pushing to SSH
# Tests: imports, splits generation, tiny pipeline run

set -e  # Exit on error

cd "$(dirname "$0")/.."

echo "=================================================="
echo "LOCAL SMOKE TEST - Stage 1 Ultimate"
echo "=================================================="

# Step 1: Verify Python syntax
echo ""
echo "[1/6] Verifying Python syntax..."
python3 -m compileall src/ -q
echo "✅ All Python files compile"

# Step 2: Verify imports
echo ""
echo "[2/6] Verifying imports..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

# Old system imports
from contracts.artifact_schema import create_artifact_schema
from pipeline.dag_engine import DAGEngine
from pipeline.phase_spec import PhaseType
from data.datamodule import NATIXDataModule

# New system imports
from streetvision.io import write_json_atomic, create_step_manifest
from streetvision.eval import compute_mcc, compute_all_metrics
from streetvision.pipeline.steps import (
    run_phase1_baseline,
    run_phase2_threshold_sweep,
    run_phase6_bundle_export,
)

print('✅ All imports successful')
"

# Step 3: Generate splits.json
echo ""
echo "[3/6] Generating splits.json..."
python3 scripts/generate_splits.py

# Step 4: Verify splits.json exists
echo ""
echo "[4/6] Verifying splits.json..."
if [ ! -f "outputs/splits.json" ]; then
    echo "❌ ERROR: splits.json not found"
    exit 1
fi
echo "✅ splits.json exists"

# Step 5: Run tiny Phase-1 (CPU, 100 samples, 1 epoch)
echo ""
echo "[5/6] Running tiny Phase-1 smoke test (CPU, 100 samples)..."
echo "This will take ~5 minutes..."
python3 scripts/train_cli_v2.py \
    pipeline.phases=[phase1] \
    data.max_samples=100 \
    training.epochs=1 \
    hardware.num_gpus=0 \
    data.splits_json=outputs/splits.json

# Step 6: Verify outputs exist
echo ""
echo "[6/6] Verifying smoke test outputs..."

# Find latest run
LATEST_RUN=$(ls -td outputs/stage1_ultimate/runs/* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ ERROR: No run directory found"
    exit 1
fi

echo "Latest run: $LATEST_RUN"

# Check manifest
if [ ! -f "$LATEST_RUN/phase1_baseline/manifest.json" ]; then
    echo "❌ ERROR: manifest.json not found"
    exit 1
fi
echo "✅ manifest.json exists"

# Check checkpoint
if [ ! -f "$LATEST_RUN/phase1_baseline/model_best.pth" ]; then
    echo "❌ ERROR: model_best.pth not found"
    exit 1
fi
echo "✅ model_best.pth exists"

# Check logits/labels
if [ ! -f "$LATEST_RUN/phase1_baseline/val_calib_logits.pt" ]; then
    echo "❌ ERROR: val_calib_logits.pt not found"
    exit 1
fi
echo "✅ val_calib_logits.pt exists"

if [ ! -f "$LATEST_RUN/phase1_baseline/val_calib_labels.pt" ]; then
    echo "❌ ERROR: val_calib_labels.pt not found"
    exit 1
fi
echo "✅ val_calib_labels.pt exists"

# Print manifest metrics
echo ""
echo "Manifest metrics:"
python3 -c "
import json
with open('$LATEST_RUN/phase1_baseline/manifest.json') as f:
    m = json.load(f)
    print(f\"  MCC: {m['metrics']['mcc']:.4f}\")
    print(f\"  Accuracy: {m['metrics']['accuracy']:.4f}\")
    print(f\"  Duration: {m['duration_seconds']:.1f}s\")
"

echo ""
echo "=================================================="
echo "✅ LOCAL SMOKE TEST PASSED"
echo "=================================================="
echo ""
echo "You can now:"
echo "1. Clean outputs: rm -rf outputs/stage1_ultimate/runs/*"
echo "2. Push to GitHub: git add . && git commit -m '...' && git push"
echo "3. Deploy to SSH GPU server"
echo ""
