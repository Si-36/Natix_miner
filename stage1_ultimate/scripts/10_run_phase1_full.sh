#!/bin/bash
# Run FULL Phase-1 with ExPLoRA (100 epochs, ~12-16 hours)

set -e
source .venv/bin/activate

echo "========================================================================"
echo "Phase-1 FULL with ExPLoRA: 100 epochs (~12-16 hours)"
echo "========================================================================"

python scripts/train_cli.py \
  pipeline.phases=[phase1] \
  data.data_root=~/natix/data/natix_full \
  output_dir=~/natix/runs/run_full \
  experiment_name=run_full \
  model.init_from_explora=true \
  model.explora_checkpoint_path=~/natix/runs/run_full/phase4_explora/explora_backbone.pth \
  training.epochs=100 \
  hardware.num_gpus=2 \
  training.mixed_precision.enabled=true \
  training.mixed_precision.dtype=bfloat16 \
  model.use_multiview=true \
  2>&1 | tee ~/natix/logs/phase1_full_100epochs.log

echo ""
echo "✅ Phase-1 FULL complete!"
echo "Expected accuracy: 85-92%"
echo ""
echo "Check final accuracy:"
echo "  tail -50 ~/natix/logs/phase1_full_100epochs.log | grep val_select/acc"
echo ""
echo "Next: bash scripts/11_run_phase2_and_6.sh"
