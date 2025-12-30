#!/bin/bash
# STAGE-1 FULL PIPELINE FOR SSH RENTAL GPU
# Run this on rental GPU after setup is complete
#
# Prerequisites:
# 1. Dataset at ~/natix/data/natix_full/
# 2. splits.json at ~/natix/runs/run_stage1_explora_001/splits.json
# 3. Virtual environment activated
#
# Usage:
#   chmod +x scripts/run_full_pipeline_ssh.sh
#   ./scripts/run_full_pipeline_ssh.sh 2>&1 | tee ~/natix/logs/full_pipeline.log

set -e  # Exit on error

echo "================================================================================"
echo " STAGE-1 FULL PIPELINE: Phase-4 → Phase-1 → Phase-2 → Phase-6"
echo "================================================================================"
echo ""

# Configuration
DATA_ROOT=~/natix/data/natix_full
RUN_DIR=~/natix/runs/run_stage1_explora_001
LOG_DIR=~/natix/logs
NUM_GPUS=2  # Change to 4 for H100
EXPERIMENT_NAME=stage1_explora_001

# Verify prerequisites
echo "Checking prerequisites..."
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi

if [ ! -f "$RUN_DIR/splits.json" ]; then
    echo "ERROR: splits.json not found at $RUN_DIR/splits.json"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --list-gpus
echo ""

# Activate venv (adjust if using conda)
if [ -d ".venv/bin" ]; then
    source .venv/bin/activate
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
else
    echo "WARNING: Virtual environment not found. Continuing with system Python..."
fi

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# ============================================================================
# PHASE 4: ExPLoRA Domain Adaptation
# ============================================================================

echo "================================================================================"
echo " PHASE 4: ExPLoRA Domain Adaptation (ETA: 24h on 2×A6000, 12-16h on 4×H100)"
echo "================================================================================"
echo ""

python scripts/train_cli.py \
    pipeline.phases=[phase4] \
    data.data_root=$DATA_ROOT \
    experiment_name=$EXPERIMENT_NAME \
    output_dir=$RUN_DIR \
    hardware.num_gpus=$NUM_GPUS \
    training.mixed_precision.enabled=true \
    training.mixed_precision.dtype=bfloat16 \
    2>&1 | tee $LOG_DIR/phase4_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Phase 4 Complete!"
echo "Checkpoint: $RUN_DIR/phase4_explora/explora_backbone.pth"
echo ""

# Verify Phase-4 output
if [ ! -f "$RUN_DIR/phase4_explora/explora_backbone.pth" ]; then
    echo "ERROR: Phase-4 checkpoint not found!"
    exit 1
fi

# ============================================================================
# PHASE 1: Real Training (Init from ExPLoRA)
# ============================================================================

echo "================================================================================"
echo " PHASE 1: Real Training (ETA: 12-16h on 2×A6000, 6-8h on 4×H100)"
echo "================================================================================"
echo ""

python scripts/train_cli.py \
    pipeline.phases=[phase1] \
    data.data_root=$DATA_ROOT \
    experiment_name=$EXPERIMENT_NAME \
    output_dir=$RUN_DIR \
    model.init_from_explora=true \
    model.explora_checkpoint_path=$RUN_DIR/phase4_explora/explora_backbone.pth \
    training.epochs=100 \
    training.batch_size=32 \
    hardware.num_gpus=$NUM_GPUS \
    training.mixed_precision.enabled=true \
    training.mixed_precision.dtype=bfloat16 \
    model.use_multiview=true \
    2>&1 | tee $LOG_DIR/phase1_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Phase 1 Complete!"
echo "Checkpoint: $RUN_DIR/phase1/model_best.pth"
echo "Val_calib logits: $RUN_DIR/phase1/val_calib_logits.pt"
echo ""

# Verify Phase-1 outputs
if [ ! -f "$RUN_DIR/phase1/model_best.pth" ]; then
    echo "ERROR: Phase-1 checkpoint not found!"
    exit 1
fi

if [ ! -f "$RUN_DIR/phase1/val_calib_logits.pt" ]; then
    echo "ERROR: Phase-1 val_calib_logits not found!"
    exit 1
fi

# Quick validation check
echo "Validating Phase-1 outputs..."
python << 'EOF'
import torch
logits = torch.load('$RUN_DIR/phase1/val_calib_logits.pt')
labels = torch.load('$RUN_DIR/phase1/val_calib_labels.pt')
print(f"Logits: {logits.shape}, dtype={logits.dtype}")
print(f"Has NaN: {torch.isnan(logits).any()}, Has Inf: {torch.isinf(logits).any()}")
if torch.isnan(logits).any() or torch.isinf(logits).any():
    print("❌ ERROR: NaN/Inf in logits!")
    exit(1)
print("✅ Logits valid!")
EOF

# ============================================================================
# PHASE 2: Threshold Policy
# ============================================================================

echo "================================================================================"
echo " PHASE 2: Threshold Policy (ETA: ~5 minutes)"
echo "================================================================================"
echo ""

python scripts/train_cli.py \
    pipeline.phases=[phase2] \
    experiment_name=$EXPERIMENT_NAME \
    output_dir=$RUN_DIR \
    training.target_fnr_exit=0.05 \
    2>&1 | tee $LOG_DIR/phase2_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Phase 2 Complete!"
echo "Policy: $RUN_DIR/phase2/thresholds.json"
echo ""

# Verify Phase-2 output
if [ ! -f "$RUN_DIR/phase2/thresholds.json" ]; then
    echo "ERROR: Phase-2 thresholds not found!"
    exit 1
fi

# ============================================================================
# PHASE 6: Bundle Export
# ============================================================================

echo "================================================================================"
echo " PHASE 6: Bundle Export (ETA: ~1 minute)"
echo "================================================================================"
echo ""

python scripts/train_cli.py \
    pipeline.phases=[phase6] \
    experiment_name=$EXPERIMENT_NAME \
    output_dir=$RUN_DIR \
    2>&1 | tee $LOG_DIR/phase6_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Phase 6 Complete!"
echo "Bundle: $RUN_DIR/export/bundle.json"
echo ""

# Verify Phase-6 output
if [ ! -f "$RUN_DIR/export/bundle.json" ]; then
    echo "ERROR: Phase-6 bundle not found!"
    exit 1
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "================================================================================"
echo " 🎉 STAGE-1 PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Artifacts created:"
echo "  Phase-4: $RUN_DIR/phase4_explora/explora_backbone.pth"
echo "  Phase-1: $RUN_DIR/phase1/model_best.pth"
echo "  Phase-1: $RUN_DIR/phase1/val_calib_logits.pt"
echo "  Phase-2: $RUN_DIR/phase2/thresholds.json"
echo "  Phase-6: $RUN_DIR/export/bundle.json"
echo ""
echo "Next steps:"
echo "  1. Run final evaluation on val_test split (ONLY ONCE!)"
echo "  2. Transfer artifacts to local machine"
echo "  3. Deploy bundle for NATIX mining"
echo ""
echo "To evaluate on val_test:"
echo "  python scripts/evaluate_bundle.py --bundle $RUN_DIR/export/bundle.json --split val_test"
echo ""
echo "================================================================================"
