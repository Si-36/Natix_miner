#!/bin/bash
# Run Phase-1 WITHOUT ExPLoRA (baseline, 10 epochs test)
# Compare to WITH ExPLoRA to decide if it helps

set -e
source .venv/bin/activate

echo "========================================================================"
echo "Phase-1 BASELINE (no ExPLoRA): 10 epochs test"
echo "========================================================================"

python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  data.data_root=~/natix/data/natix_full \
  output_dir=~/natix/runs/run_test_baseline \
  experiment_name=run_test_baseline \
  model.init_from_explora=false \
  training.epochs=10 \
  hardware.num_gpus=2 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  2>&1 | tee ~/natix/logs/phase1_baseline_test.log

echo ""
echo "✅ Phase-1 BASELINE complete!"
echo ""
echo "Check accuracy:"
echo "  tail -50 ~/natix/logs/phase1_baseline_test.log | grep val_select/acc"
echo ""
echo "Next: bash scripts/7_compare_results.sh"
