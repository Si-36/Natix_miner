#!/bin/bash
# IF GO: Run FULL Phase-4 (100 epochs, ~36-48 hours)
# Only run this if ExPLoRA test showed improvement!

set -e
source .venv/bin/activate

echo "========================================================================"
echo "Phase-4 FULL: 100 epochs (~36-48 hours on 2×A6000)"
echo "========================================================================"
echo ""
echo "WARNING: This will take 36-48 hours!"
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

python scripts/train_cli.py \
  pipeline.phases=[phase4] \
  data.data_root=~/natix/data/natix_full \
  output_dir=~/natix/runs/run_full \
  experiment_name=run_full \
  hardware.num_gpus=2 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  training.max_epochs=100 \
  2>&1 | tee ~/natix/logs/phase4_full_100epochs.log

echo ""
echo "✅ Phase-4 FULL complete!"
echo "Checkpoint: ~/natix/runs/run_full/phase4_explora/explora_backbone.pth"
echo ""
echo "Next: bash scripts/10_run_phase1_full.sh"
