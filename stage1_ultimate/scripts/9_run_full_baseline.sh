#!/bin/bash
# IF STOP: Run full baseline pipeline (NO ExPLoRA)
# Use this if ExPLoRA test showed NO improvement

set -e
source .venv/bin/activate

echo "========================================================================"
echo "BASELINE PIPELINE (no ExPLoRA)"
echo "========================================================================"
echo ""
echo "Phase-1 baseline: 100 epochs (~12-16 hours)"
echo "Phase-2 thresholds: ~5 minutes"
echo "Phase-6 bundle: ~1 minute"
echo ""
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

# Phase-1 baseline
echo ""
echo "Running Phase-1..."
python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  data.data_root=~/natix/data/natix_full \
  output_dir=~/natix/runs/run_baseline_full \
  experiment_name=run_baseline_full \
  model.init_from_explora=false \
  training.epochs=100 \
  hardware.num_gpus=2 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  model.use_multiview=true \
  2>&1 | tee ~/natix/logs/phase1_baseline_full.log

# Phase-2 thresholds
echo ""
echo "Running Phase-2..."
python scripts/train_cli.py \
  pipeline.phases=[phase2] \
  output_dir=~/natix/runs/run_baseline_full \
  experiment_name=run_baseline_full \
  training.target_fnr_exit=0.05 \
  2>&1 | tee ~/natix/logs/phase2_baseline.log

# Phase-6 bundle
echo ""
echo "Running Phase-6..."
python scripts/train_cli.py \
  pipeline.phases=[phase6] \
  output_dir=~/natix/runs/run_baseline_full \
  experiment_name=run_baseline_full \
  2>&1 | tee ~/natix/logs/phase6_baseline.log

echo ""
echo "========================================================================"
echo "🎉 BASELINE PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Final artifacts:"
echo "  - Checkpoint: ~/natix/runs/run_baseline_full/phase1/model_best.pth"
echo "  - Policy: ~/natix/runs/run_baseline_full/phase2/thresholds.json"
echo "  - Bundle: ~/natix/runs/run_baseline_full/export/bundle.json"
echo ""
echo "Expected accuracy: 75-85% (without ExPLoRA)"
echo ""
