#!/bin/bash
# ALL PHASES END-TO-END TEST SCRIPT
# Tests all 11 phases in sequence (Dec 2025)
#
# Usage:
#   bash scripts/99_test_all_phases.sh [--skip_explora]
#
# Arguments:
#   --skip_explora: Skip Phase 4.1 (ExPLoRA requires unlabeled data)

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default arguments
SKIP_EXPLORA=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_explora)
            SKIP_EXPLORA=true
            echo -e "${YELLOW}⚠️  Skipping Phase 4.1 (ExPLoRA) due to --skip_explora flag${NC}"
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL PHASES END-TO-END TEST (Dec 2025)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create results directory
RESULTS_DIR="test_all_phases_results"
mkdir -p "$RESULTS_DIR"

# Test results file
RESULTS_FILE="$RESULTS_DIR/test_results.csv"
echo "Phase,SubPhase,Status,ExitCode,Duration" > "$RESULTS_FILE"

# Track start time
START_TIME=$(date +%s)

echo -e "${GREEN}PHASE 1: BASELINE TEST${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PHASE1_START=$(date +%s)

# Test Phase 1.1-1.10: Train + evaluate
echo -e "${YELLOW}Running: scripts/20_train.py (Phase 1)${NC}"
if python scripts/20_train.py --config config.yaml > "$RESULTS_DIR/phase1_train.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 1 Training PASSED${NC}"
    echo "Phase 1,Train,PASSED,0,0" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 1 Training FAILED${NC}"
    echo "Phase 1,Train,FAILED,1,$(($(date +%s) - PHASE1_START))" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: scripts/25_threshold_sweep.py (Phase 1)${NC}"
if python scripts/25_threshold_sweep.py --config config.yaml > "$RESULTS_DIR/phase1_threshold.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 1 Threshold Sweep PASSED${NC}"
    echo "Phase 1,Threshold,PASSED,0,0" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 1 Threshold Sweep FAILED${NC}"
    echo "Phase 1,Threshold,FAILED,1,$(($(date +%s) - PHASE1_START))" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: scripts/40_eval_selective.py (Phase 1)${NC}"
if python scripts/40_eval_selective.py --checkpoint model_best.pth --config config.yaml > "$RESULTS_DIR/phase1_eval.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 1 Evaluation PASSED${NC}"
    echo "Phase 1,Eval,PASSED,0,$(($(date +%s) - PHASE1_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 1 Evaluation FAILED${NC}"
    echo "Phase 1,Eval,FAILED,1,$(($(date +%s) - PHASE1_START))" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: scripts/50_export_bundle.py (Phase 1)${NC}"
if python scripts/50_export_bundle.py --config config.yaml > "$RESULTS_DIR/phase1_bundle.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 1 Bundle Export PASSED${NC}"
    echo "Phase 1,Bundle,PASSED,0,$(($(date +%s) - PHASE1_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 1 Bundle Export FAILED${NC}"
    echo "Phase 1,Bundle,FAILED,1,$(($(date +%s) - PHASE1_START))" >> "$RESULTS_FILE"
    exit 1
fi

PHASE1_DURATION=$(($(date +%s) - PHASE1_START))
echo -e "${GREEN}✅ Phase 1 COMPLETE (${PHASE1_DURATION}s)${NC}"
echo ""
echo -e "${GREEN}PHASE 2: SELECTIVE METRICS TEST${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PHASE2_START=$(date +%s)

# Test Phase 2.1-2.19: Selective metrics
echo -e "${YELLOW}Running: Phase 2 Evaluation (AUGRC)${NC}"
# Note: Uses model trained with Phase 2.5 checkpoint selection
if python scripts/40_eval_selective.py --checkpoint model_best.pth --config config.yaml > "$RESULTS_DIR/phase2_eval.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 2 Evaluation PASSED${NC}"
    echo "Phase 2,Eval,PASSED,0,$(($(date +%s) - PHASE2_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 2 Evaluation FAILED${NC}"
    echo "Phase 2,Eval,FAILED,1,$(($(date +%s) - PHASE2_START))" >> "$RESULTS_FILE"
    exit 1
fi

PHASE2_DURATION=$(($(date +%s) - PHASE2_START))
echo -e "${GREEN}✅ Phase 2 COMPLETE (${PHASE2_DURATION}s)${NC}"
echo ""
echo -e "${GREEN}PHASE 3: GATE CASCADE TEST${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PHASE3_START=$(date +%s)

# Test Phase 3.1-3.16: Gate cascade
echo -e "${YELLOW}Running: Phase 3 Training (Gate)${NC}"
if python scripts/20_train.py --config config.yaml --exit_policy gate > "$RESULTS_DIR/phase3_train.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 3 Training PASSED${NC}"
    echo "Phase 3,Train,PASSED,0,0" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 3 Training FAILED${NC}"
    echo "Phase 3,Train,FAILED,1,0" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: scripts/33_calibrate_gate.py (Phase 3)${NC}"
if python scripts/33_calibrate_gate.py --checkpoint gate_best.pth --config config.yaml > "$RESULTS_DIR/phase3_calib.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 3 Calibration PASSED${NC}"
    echo "Phase 3,Calib,PASSED,0,$(($(date +%s) - PHASE3_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 3 Calibration FAILED${NC}"
    echo "Phase 3,Calib,FAILED,1,$(($(date +%s) - PHASE3_START))" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: scripts/41_infer_gate.py (Phase 3)${NC}"
if python scripts/41_infer_gate.py --checkpoint gate_best.pth --gateparams gateparams.json --config config.yaml > "$RESULTS_DIR/phase3_infer.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 3 Inference PASSED${NC}"
    echo "Phase 3,Infer,PASSED,0,$(($(date +%s) - PHASE3_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 3 Inference FAILED${NC}"
    echo "Phase 3,Infer,FAILED,1,$(($(date +%s) - PHASE3_START))" >> "$RESULTS_FILE"
    exit 1
fi

PHASE3_DURATION=$(($(date +%s) - PHASE3_START))
echo -e "${GREEN}✅ Phase 3 COMPLETE (${PHASE3_DURATION}s)${NC}"
echo ""
echo -e "${GREEN}PHASE 4: PEFT + EXPLORA TEST${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PHASE4_START=$(date +%s)

# Test Phase 4.7.6-4.7.8: PEFT acceptance tests
echo -e "${YELLOW}Running: Phase 4.7 Acceptance Tests${NC}"
if python tests/test_peft_47_acceptance.py > "$RESULTS_DIR/phase4_acceptance.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 4.7 Acceptance Tests PASSED${NC}"
    echo "Phase 4,Acceptance,PASSED,0,$(($(date +%s) - PHASE4_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 4.7 Acceptance Tests FAILED${NC}"
    echo "Phase 4,Acceptance,FAILED,1,$(($(date +%s) - PHASE4_START))" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: Phase 4.7.5 A/B Test${NC}"
if python scripts/43_ab_test_peft.py --config config.yaml --output_dir "$RESULTS_DIR/ab_results" > "$RESULTS_DIR/phase4_ab_test.log" 2>&1; then
    echo -e "${GREEN}✅ Phase 4.7.5 A/B Test PASSED${NC}"
    echo "Phase 4,ABTest,PASSED,0,$(($(date +%s) - PHASE4_START))" >> "$RESULTS_FILE"
else
    echo -e "${RED}❌ Phase 4.7.5 A/B Test FAILED${NC}"
    echo "Phase 4,ABTest,FAILED,1,$(($(date +%s) - PHASE4_START))" >> "$RESULTS_FILE"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running: Phase 4.1 ExPLoRA (if not skipped)${NC}"
if [ "$SKIP_EXPLORA" = false ]; then
    EXPLORA_START=$(date +%s)
    
    # Test Phase 4.1.2: ExPLoRA pretraining
    echo -e "${YELLOW}Running: Phase 4.1.2 ExPLoRA Pretraining${NC}"
    if python scripts/44_explora_pretrain.py \
        --config config.yaml \
        --backbone facebook/dinov3-vith14 \
        --unfreeze_blocks 2 \
        --peft_r 16 \
        --epochs 2 > "$RESULTS_DIR/phase4_explora_pretrain.log" 2>&1; then
        echo -e "${GREEN}✅ Phase 4.1.2 ExPLoRA Pretraining PASSED${NC}"
        echo "Phase 4,ExploraPretrain,PASSED,0,$(($(date +%s) - EXPLORA_START))" >> "$RESULTS_FILE"
    else
        echo -e "${RED}❌ Phase 4.1.2 ExPLoRA Pretraining FAILED${NC}"
        echo "Phase 4,ExploraPretrain,FAILED,1,$(($(date +%s) - EXPLORA_START))" >> "$RESULTS_FILE"
        exit 1
    fi
    
    echo ""
    
    # Test Phase 4.1.3: Supervised training with ExPLoRA
    echo -e "${YELLOW}Running: Phase 4.1.3 Supervised Training (ExPLoRA)${NC}"
    if python scripts/45_train_supervised_explora.py \
        --config config.yaml \
        --backbone output_explora/backbone_explora.pth \
        --peft_type lora \
        --peft_r 8 > "$RESULTS_DIR/phase4_explora_supervised.log" 2>&1; then
        echo -e "${GREEN}✅ Phase 4.1.3 Supervised Training PASSED${NC}"
        echo "Phase 4,ExploraSupervised,PASSED,0,$(($(date +%s) - EXPLORA_START))" >> "$RESULTS_FILE"
    else
        echo -e "${RED}❌ Phase 4.1.3 Supervised Training FAILED${NC}"
        echo "Phase 4,ExploraSupervised,FAILED,1,$(($(date +%s) - EXPLORA_START))" >> "$RESULTS_FILE"
        exit 1
    fi
    
    EXPLORA_DURATION=$(($(date +%s) - EXPLORA_START))
    echo -e "${GREEN}✅ Phase 4.1 COMPLETE (${EXPLORA_DURATION}s)${NC}"
else
    echo -e "${YELLOW}⏭  Skipping Phase 4.1 (ExPLoRA)${NC}"
    echo "Phase 4,Explora,SKIPPED,0,0" >> "$RESULTS_FILE"
fi

PHASE4_DURATION=$(($(date +%s) - PHASE4_START))
echo -e "${GREEN}✅ Phase 4 COMPLETE (${PHASE4_DURATION}s)${NC}"
echo ""
echo -e "${GREEN}PHASE 5: SCRC TEST${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PHASE5_START=$(date +%s)

# Test Phase 5: SCRC (if implemented)
echo -e "${YELLOW}Running: Phase 5 SCRC Tests${NC}"
if [ -f scripts/46_calibrate_scrc.py ]; then
    if python tests/test_scrc_51_acceptance.py > "$RESULTS_DIR/phase5_acceptance.log" 2>&1; then
        echo -e "${GREEN}✅ Phase 5 SCRC Tests PASSED${NC}"
        echo "Phase 5,Acceptance,PASSED,0,$(($(date +%s) - PHASE5_START))" >> "$RESULTS_FILE"
    else
        echo -e "${RED}❌ Phase 5 SCRC Tests FAILED${NC}"
        echo "Phase 5,Acceptance,FAILED,1,$(($(date +%s) - PHASE5_START))" >> "$RESULTS_FILE"
        exit 1
    fi
    
    PHASE5_DURATION=$(($(date +%s) - PHASE5_START))
    echo -e "${GREEN}✅ Phase 5 COMPLETE (${PHASE5_DURATION}s)${NC}"
else
    echo -e "${RED}❌ Phase 5 NOT IMPLEMENTED${NC}"
    echo "Phase 5,NotImplemented,FAILED,1,0" >> "$RESULTS_FILE"
    PHASE5_DURATION=0
fi

echo ""
echo -e "${GREEN}PHASE 6: PRODUCTION TEST${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

PHASE6_START=$(date +%s)

# Test Phase 6: Production (if implemented)
echo -e "${YELLOW}Running: Phase 6 Production Tests${NC}"
if [ -f scripts/48_deploy_production.py ]; then
    if python tests/test_production_61_acceptance.py > "$RESULTS_DIR/phase6_acceptance.log" 2>&1; then
        echo -e "${GREEN}✅ Phase 6 Production Tests PASSED${NC}"
        echo "Phase 6,Acceptance,PASSED,0,$(($(date +%s) - PHASE6_START))" >> "$RESULTS_FILE"
    else
        echo -e "${RED}❌ Phase 6 Production Tests FAILED${NC}"
        echo "Phase 6,Acceptance,FAILED,1,$(($(date +%s) - PHASE6_START))" >> "$RESULTS_FILE"
        exit 1
    fi
    
    PHASE6_DURATION=$(($(date +%s) - PHASE6_START))
    echo -e "${GREEN}✅ Phase 6 COMPLETE (${PHASE6_DURATION}s)${NC}"
else
    echo -e "${RED}❌ Phase 6 NOT IMPLEMENTED${NC}"
    echo "Phase 6,NotImplemented,FAILED,1,0" >> "$RESULTS_FILE"
    PHASE6_DURATION=0
fi

# Total time
TOTAL_TIME=$(($(date +%s) - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FINAL RESULTS SUMMARY${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${NC}Total Time: ${TOTAL_TIME}s (${TOTAL_MINUTES} minutes)${NC}"
echo ""

# Show results table
echo -e "${GREEN}Phase Results:${NC}"
cat "$RESULTS_FILE"
echo ""

# Count passed/failed
PASSED=$(grep -c ",PASSED," "$RESULTS_FILE" || true)
FAILED=$(grep -c ",FAILED," "$RESULTS_FILE" || true)
SKIPPED=$(grep -c ",SKIPPED," "$RESULTS_FILE" || true)
TOTAL=$(PASSED + FAILED + SKIPPED)

echo -e "${GREEN}Summary:${NC}"
echo -e "${GREEN}  Passed: ${PASSED}${NC}"
echo -e "${GREEN}  Failed: ${FAILED}${NC}"
echo -e "${YELLOW}  Skipped: ${SKIPPED}${NC}"
echo -e "${GREEN}  Total: ${TOTAL}${NC}"
echo ""

# Exit code
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}❌ SOME TESTS FAILED${NC}"
    exit 1
elif [ $PASSED -eq $TOTAL ] || [ $PASSED -eq $((TOTAL - SKIPPED)) ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  SOME TESTS SKIPPED (Phases 5-6 not implemented)${NC}"
    exit 0
fi

