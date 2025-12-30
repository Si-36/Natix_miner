#!/bin/bash
# Run Phase-1 WITH ExPLoRA checkpoint (10 epochs test)
# Tests if ExPLoRA improves accuracy

set -e
source .venv/bin/activate

echo "========================================================================"
echo "Phase-1 WITH ExPLoRA: 10 epochs test"
echo "========================================================================"

python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  data.data_root=~/natix/data/natix_full \
  output_dir=~/natix/runs/run_test_with_explora \
  experiment_name=run_test_with_explora \
  model.init_from_explora=true \
  model.explora_checkpoint_path=~/natix/runs/run_test/phase4_explora/explora_backbone.pth \
  training.epochs=10 \
  hardware.num_gpus=2 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  2>&1 | tee ~/natix/logs/phase1_with_explora_test.log

echo ""
echo "✅ Phase-1 WITH ExPLoRA complete!"
echo ""
echo "Check accuracy:"
echo "  tail -50 ~/natix/logs/phase1_with_explora_test.log | grep val_select/acc"
echo ""
echo "Next: bash scripts/6_run_phase1_baseline.sh"
