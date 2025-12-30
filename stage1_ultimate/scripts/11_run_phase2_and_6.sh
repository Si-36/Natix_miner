#!/bin/bash
# Run Phase-2 (thresholds) and Phase-6 (bundle export)
# Final step: creates deployable bundle for NATIX mining

set -e
source .venv/bin/activate

echo "========================================================================"
echo "Phase-2: Threshold Policy (~5 minutes)"
echo "========================================================================"

python scripts/train_cli.py \
  pipeline.phases=[phase2] \
  output_dir=~/natix/runs/run_full \
  experiment_name=run_full \
  training.target_fnr_exit=0.05 \
  2>&1 | tee ~/natix/logs/phase2.log

echo ""
echo "✅ Phase-2 complete!"
echo "Policy: ~/natix/runs/run_full/phase2/thresholds.json"
echo ""

echo "========================================================================"
echo "Phase-6: Bundle Export (~1 minute)"
echo "========================================================================"

python scripts/train_cli.py \
  pipeline.phases=[phase6] \
  output_dir=~/natix/runs/run_full \
  experiment_name=run_full \
  2>&1 | tee ~/natix/logs/phase6.log

echo ""
echo "✅ Phase-6 complete!"
echo "Bundle: ~/natix/runs/run_full/export/bundle.json"
echo ""
echo "========================================================================"
echo "🎉 COMPLETE PIPELINE FINISHED!"
echo "========================================================================"
echo ""
echo "Final artifacts:"
echo "  - Checkpoint: ~/natix/runs/run_full/phase1/model_best.pth"
echo "  - Policy: ~/natix/runs/run_full/phase2/thresholds.json"
echo "  - Bundle: ~/natix/runs/run_full/export/bundle.json"
echo ""
echo "Download to local machine:"
echo "  scp -r user@ssh-ip:~/natix/runs/run_full/export/ ~/natix_results/"
echo ""
