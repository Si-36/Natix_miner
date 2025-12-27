# Stage-1 Pro: Production-Ready Roadwork Detection System

**Version:** 2.0
**Date:** December 25, 2025
**Context:** Refactored `train_stage1_v2.py` into a modular, production-ready system.

## Architecture

### Core Modules

1. **Data** (`data/`)
   - `datasets.py`: Multi-source dataset loaders (NATIX, Roadwork, Kaggle)
   - `splits.py`: Deterministic val_select/val_calib splits (no leakage)
   - `transforms.py`: Timm-style augmentation (aggressive + moderate modes)

2. **Model** (`model/`)
   - `backbone.py`: DINOv3 wrapper (frozen, PEFT hooks)
   - `head.py`: 3-head architecture (cls/gate/aux), compiled
   - `peft.py`: DoRAN implementation with DoRA fallback

3. **Training** (`training/`)
   - `trainer.py`: Main trainer (extract_features, train_cached, train, train_crc)
   - `optimizers.py`: AdamW/F-SAM with per-layer LR
   - `ema.py`: EMA shadowing
   - `losses.py`: CE + Selective/Risk/Auxiliary losses

4. **Calibration** (`calibration/`)
   - `scrc.py`: SCRC-I calibrator (prediction sets)
   - `dirichlet.py`: Dirichlet probability calibration
   - `gate.py`: Platt scaling for gate scores

5. **Metrics** (`metrics/`)
   - `selective.py`: Risk@Coverage, Coverage@Risk, FNR_on_exits, AvgSetSize

6. **Utils** (`utils/`)
   - `feature_cache.py`: Save/Load train_features.pt files
   - `checkpointing.py`: Save/Load model dict, ema shadow
   - `logging.py`: CSV logging with selective metrics

## Usage

### Quick Start

```bash
# 1. Create validation splits
python scripts/00_make_splits.py

# 2. Train model (cached features for speed)
python scripts/20_train_riskaware.py --mode train_cached

# 3. Calibrate SCRC
python scripts/30_calibrate_scrc.py

# 4. Evaluate selective performance
python scripts/40_eval_selective.py

# 5. Export deployment bundle
python scripts/50_export_bundle.py
```

### Full Pipeline

```bash
# Sequential execution
python scripts/00_make_splits.py && \
python scripts/20_train_riskaware.py --mode train && \
python scripts/30_calibrate_scrc.py && \
python scripts/40_eval_selective.py && \
python scripts/50_export_bundle.py
```

## Configuration

Edit `configs/stage1_pro.yaml` to set:
- Paths (datasets, checkpoints, outputs)
- Hyperparameters (batch size, learning rate, epochs)
- SCRC targets (FNR ≤ 2%, Coverage ≥ 80%)
- PEFT settings (DoRAN rank, alpha)
- Mode (train_cached, train, train_crc)

## Deployment

The deployment bundle includes:
- `model_best.pth`: Best model checkpoint
- `scrc_params.json`: SCRC calibration parameters
- `splits.json`: Train/val indices
- `metrics.csv`: Selective metrics

Load in inference:
```python
from stage1_pro.utils.checkpointing import load_checkpoint
from stage1_pro.calibration.scrc import SCRCCalibrator

model, config = load_checkpoint("artifacts/runs/<run_id>/model_best.pth")
scrc = SCRCCalibrator.load("artifacts/runs/<run_id>/scrc_params.json")
```

## Migration from Baseline

The baseline `train_stage1_v2.py` is preserved in `artifacts/baseline/`. Key changes:
- Modular architecture (easier to test and extend)
- SCRC-I calibrator replaces hardcoded thresholds
- Proper validation splits (val_select/val_calib)
- Comprehensive selective metrics
- EMA shadowing for stable performance
- PEFT (DoRAN) for efficient fine-tuning

## Testing

```bash
# Unit tests
python -m pytest scripts/unit/test_config.py
python -m pytest scripts/unit/test_data.py

# Integration test (smoke test)
python scripts/integration/test_full_pipeline.py
```
