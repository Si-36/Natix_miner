#!/bin/bash
# Run Phase-4 Test (20 epochs, 8-12 hours)
# Tests if ExPLoRA helps before committing to 100 epochs

set -e
source .venv/bin/activate

echo "========================================================================"
echo "Phase-4 Test: 20 epochs (~8-12 hours)"
echo "========================================================================"

python scripts/train_cli.py \
  pipeline.phases=[phase4] \
  data.data_root=~/natix/data/natix_full \
  output_dir=~/natix/runs/run_test \
  experiment_name=run_test \
  hardware.num_gpus=2 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  training.max_epochs=20 \
  2>&1 | tee ~/natix/logs/phase4_test_20epochs.log

echo ""
echo "✅ Phase-4 test complete!"
echo "Checkpoint: ~/natix/runs/run_test/phase4_explora/explora_backbone.pth"
echo ""
echo "Next: bash scripts/5_run_phase1_test.sh"
