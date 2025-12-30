#!/bin/bash
# Local smoke test - run this BEFORE pushing to SSH
# Tests: imports, splits generation, tiny pipeline run

set -e  # Exit on error

cd "$(dirname "$0")/.."

echo "=================================================="
echo "LOCAL SMOKE TEST - Stage 1 Ultimate"
echo "=================================================="

# Step 0: Ensure environment is installed (venv + deps)
echo ""
echo "[0/6] Ensuring Python environment..."

# Pick a compatible Python (Hydra is not stable on some Python 3.14 builds)
PY_CANDIDATES=(
  ".venv_py313/bin/python"
  "python3.13"
  "python3.12"
  "python3.11"
  "python3.10"
  "python3"
)

PY_BIN=""
for c in "${PY_CANDIDATES[@]}"; do
  if [ -x "$c" ]; then
    PY_BIN="$c"
    break
  fi
  command -v "$c" >/dev/null 2>&1 && { PY_BIN="$c"; break; }
done

if [ -z "$PY_BIN" ]; then
  echo "❌ ERROR: Could not find a usable python (need 3.11-3.13 recommended)"
  exit 1
fi

# Recreate venv if it was made with Python 3.14 (known Hydra/argparse issue)
if [ -f ".venv/pyvenv.cfg" ] && grep -q "version = 3.14" ".venv/pyvenv.cfg"; then
  echo "⚠️  Detected .venv built with Python 3.14; recreating with: $PY_BIN"
  rm -rf .venv
fi

if [ ! -d ".venv" ]; then
  "$PY_BIN" -m venv .venv
fi

source .venv/bin/activate

# Some Linux images create venv without pip; bootstrap it if needed
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade >/dev/null
python -m pip install --upgrade pip >/dev/null

# Install editable package deps if missing (fast check)
python -c "import omegaconf" 2>/dev/null || python -m pip install -e . >/dev/null
echo "✅ Environment ready"

# Step 1: Verify Python syntax
echo ""
echo "[1/6] Verifying Python syntax..."
python -m compileall src/ -q
echo "✅ All Python files compile"

# Step 2: Verify imports
echo ""
echo "[2/6] Verifying imports..."
python -c "
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

# Prefer real local dataset if present, otherwise create a tiny synthetic dataset
REAL_DATA_ROOT="$HOME/data/natix_subset"
SMOKE_DATA_ROOT="outputs/smoke_data/natix_subset"

if [ -f "$REAL_DATA_ROOT/labels.json" ]; then
  DATA_ROOT="$REAL_DATA_ROOT"
else
  DATA_ROOT="$SMOKE_DATA_ROOT"
  echo "labels.json not found at $REAL_DATA_ROOT - creating tiny synthetic dataset at $SMOKE_DATA_ROOT"
  mkdir -p "$SMOKE_DATA_ROOT/images"
  python - <<'PY'
import json
import random
from pathlib import Path

from PIL import Image

root = Path("outputs/smoke_data/natix_subset")
images = root / "images"
images.mkdir(parents=True, exist_ok=True)

random.seed(42)
labels = []
num = 200
for i in range(num):
    label = 0 if i < num // 2 else 1
    # deterministic pseudo-random pixels
    rnd = random.Random(i)
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(0, 256, 8):
        for x in range(0, 256, 8):
            px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
    name = f"smoke_{i:04d}.jpg"
    img.save(images / name, quality=85)
    labels.append({"filename": f"images/{name}", "label": int(label)})

with open(root / "labels.json", "w") as f:
    json.dump(labels, f, indent=2)
print(f"Wrote synthetic dataset: {root} ({len(labels)} images)")
PY
fi

python scripts/generate_splits.py --data-root "$DATA_ROOT"

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
python scripts/train_cli_v2.py \
    pipeline.phases=[phase1] \
    data.max_samples=100 \
    training.epochs=1 \
    hardware.num_gpus=0 \
    data.data_root=$DATA_ROOT \
    data.splits_json=outputs/splits.json \
    model.backbone_id=facebook/deit-tiny-patch16-224 \
    model.use_multiview=false \
    data.dataloader.num_workers=0 \
    data.dataloader.batch_size=8 \
    data.dataloader.val_batch_size=8

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
if [ ! -f "$LATEST_RUN/phase1/manifest.json" ]; then
    echo "❌ ERROR: manifest.json not found"
    exit 1
fi
echo "✅ manifest.json exists"

# Check checkpoint
if [ ! -f "$LATEST_RUN/phase1/model_best.pth" ]; then
    echo "❌ ERROR: model_best.pth not found"
    exit 1
fi
echo "✅ model_best.pth exists"

# Check logits/labels
if [ ! -f "$LATEST_RUN/phase1/val_calib_logits.pt" ]; then
    echo "❌ ERROR: val_calib_logits.pt not found"
    exit 1
fi
echo "✅ val_calib_logits.pt exists"

if [ ! -f "$LATEST_RUN/phase1/val_calib_labels.pt" ]; then
    echo "❌ ERROR: val_calib_labels.pt not found"
    exit 1
fi
echo "✅ val_calib_labels.pt exists"

# Print manifest metrics
echo ""
echo "Manifest metrics:"
python3 -c "
import json
with open('$LATEST_RUN/phase1/manifest.json') as f:
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
