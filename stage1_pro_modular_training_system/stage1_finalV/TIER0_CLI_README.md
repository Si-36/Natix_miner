# Tier 0 CLI - Training Pipeline Entry Point

## Overview
The Tier 0 CLI (`scripts/train.py`) provides a command-line interface to run the modular DAG-based training pipeline.

## Quick Start

### Run full pipeline (smoke test - 1 epoch, synthetic data):
```bash
source .venv/bin/activate
python scripts/train.py --synthetic --epochs 1
```

### Run specific step:
```bash
# Run only training step
python scripts/train.py --target_step train_baseline_head --synthetic --epochs 5

# Run only export step (requires completed training)
python scripts/train.py --target_step export_calib_logits --synthetic

# Run only threshold sweep (requires completed export)
python scripts/train.py --target_step sweep_thresholds --synthetic
```

### Override training parameters:
```bash
python scripts/train.py --synthetic --epochs 10 --batch_size 64 --learning_rate 5e-4
```

## Pipeline Steps
The CLI runs steps in dependency order:
1. **train_baseline_head** - Trains baseline head on frozen DINOv3 backbone
2. **export_calib_logits** - Runs inference on VAL_CALIB split to get logits
3. **sweep_thresholds** - Finds optimal threshold using calibration logits

## Artifacts Generated
All artifacts saved to `runs/<run_id>/`:
- `phase1/model_best.pth` - Trained model checkpoint
- `phase1/val_calib_logits.pt` - Calibration logits
- `phase1/val_calib_labels.pt` - Calibration labels
- `phase2/thresholds.json` - Optimal thresholds
- `phase2/thresholds_metrics.csv` - Metrics for all thresholds

## Command Line Arguments

| Argument | Type | Default | Description |
|-----------|--------|----------|-------------|
| `--target_step` | str | sweep_thresholds | Target step to run (train_baseline_head, export_calib_logits, sweep_thresholds) |
| `--epochs` | int | 1 | Number of training epochs |
| `--batch_size` | int | 32 | Training batch size |
| `--learning_rate` | float | 1e-4 | Learning rate |
| `--synthetic` | flag | False | Use synthetic/mock data instead of real dataset |
| `--run_id` | str | None (auto) | Override run ID |
| `--artifact_root` | str | runs | Root directory for artifacts |

## Testing

### Run smoke tests:
```bash
source .venv/bin/activate
python tests/integration/test_dag_smoke.py
```

Expected output:
- ✅ Export calib logits (VAL_CALIB ONLY!)
- ✅ Sweep thresholds (correct metrics!)
- ✅ DAG resolution (correct ordering!)
- ✅ Manifest tracking (all hashes recorded!)
- ✅ Split contracts (NO LEAKAGE!)
- ✅ All artifacts exist (4 artifacts!)

### Run E2E test with real checkpoint:
```bash
source .venv/bin/activate
python tests/integration/test_e2e_real_checkpoint.py
```

## Status
- ✅ CLI implemented and working
- ✅ All 3 steps execute correctly
- ✅ Artifacts saved to correct paths
- ✅ Split contracts enforced (no VAL_CALIB leakage)
- ✅ DAG resolution works
- ✅ Manifest tracking enabled
