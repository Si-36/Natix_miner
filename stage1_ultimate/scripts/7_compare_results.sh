#!/bin/bash
# Compare Phase-1 WITH vs WITHOUT ExPLoRA
# Decide: GO (continue Phase-4 to 100 epochs) or STOP (skip ExPLoRA)

echo "========================================================================"
echo "COMPARISON: ExPLoRA vs Baseline"
echo "========================================================================"
echo ""

echo "WITH ExPLoRA accuracy:"
tail -100 ~/natix/logs/phase1_with_explora_test.log | grep "val_select/acc" | tail -5

echo ""
echo "BASELINE (no ExPLoRA) accuracy:"
tail -100 ~/natix/logs/phase1_baseline_test.log | grep "val_select/acc" | tail -5

echo ""
echo "========================================================================"
echo "DECISION RULES:"
echo "========================================================================"
echo "GO (continue Phase-4 to 100 epochs):"
echo "  - WITH ExPLoRA accuracy > BASELINE accuracy + 3%"
echo "  - No NaN/errors in Phase-4 or Phase-1"
echo "  - Next: bash scripts/8_run_phase4_full.sh"
echo ""
echo "STOP (skip ExPLoRA, use baseline):"
echo "  - WITH ExPLoRA accuracy ≤ BASELINE accuracy"
echo "  - Next: bash scripts/9_run_full_baseline.sh"
echo ""
echo "========================================================================"
