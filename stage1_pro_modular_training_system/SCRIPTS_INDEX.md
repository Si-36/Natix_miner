# Scripts Index - Stage-1 Pro Modular Training System

## Status: ANALYSIS COMPLETE ✅

---

## ✅ WORKING SCRIPTS (Proven to execute)

### 1. `cli.py` (Main Entry Point)
- **Path**: `/home/sina/projects/miner_b/stage1_pro_modular_training_system/cli.py`
- **Status**: ✅ WORKING - Help text displays correctly
- **Usage**:
  ```bash
  cd /home/sina/projects/miner_b/stage1_pro_modular_training_system
  python3 cli.py --help
  python3 cli.py --mode train --phase 1 --exit_policy softmax --epochs 1 --max_batch_size 4
  ```
- **Purpose**: Unified CLI for all training phases and modes
- **Imports Fixed**: ✅ Changed from relative imports to absolute imports (`from config import...` instead of `from .config import...`)

---

## ⚠️ SCRIPTS WITH ISSUES (Need fixes)

### 2. `scripts/20_train.py`
- **Path**: `scripts/20_train.py`
- **Status**: ⚠️ PARTIALLY FIXED
- **Issues Fixed**:
  - ✅ Changed from `from stage1_pro_modular_training_system.config` to `from config` (line 16)
  - ✅ Changed from `Stage1Trainer` to `Stage1ProTrainer` (line 17)
  - ✅ Added `sys.path.insert(0, str(Path(__file__).parent.parent))` (line 14)
  - ✅ Skip `phase` attribute when setting config properties (read-only)
  - ✅ Removed call to `config.validate_phase_compatibility()` (method doesn't exist)
  - ✅ Changed `Stage1Trainer(..., phase=args.phase)` to `Stage1ProTrainer(...)` (phase arg not supported)
- **Remaining Issues**: UNTESTED - Hasn't been run yet
- **Usage**:
  ```bash
  cd /home/sina/projects/miner_b/stage1_pro_modular_training_system/scripts
  python3 20_train.py --mode train --phase 1 --exit_policy softmax --config ../config.yaml --output_dir ../outputs/baseline --epochs 1 --max_batch_size 4
  ```

---

## 🔍 IMPORT PATTERNS FOUND

### Pattern A: Scripts with Parent-Directory Path Insertion (Correct Pattern)
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Stage1ProConfig
from training.trainer import Stage1ProTrainer
```
**Used in**: `scripts/20_train.py` (after my fix)

### Pattern B: Relative Imports from Parent Directory (Also Works)
```python
from ..config import Stage1ProConfig
from ..data.datasets import NATIXDataset
from ..data.splits import create_val_splits, save_splits, compute_split_metadata
```
**Used in**: `scripts/00_make_splits.py`

### Pattern C: Direct Absolute Imports (Works when running from project root)
```python
from config import Stage1ProConfig
from model.backbone import DINOv3Backbone
from model.head import Stage1Head
```
**Used in**: 
- `scripts/25_threshold_sweep.py`
- `scripts/40_eval_selective.py`
- `scripts/43_ab_test_peft.py`
- `scripts/45_train_supervised_explora.py`

---

## 📋 ALL SCRIPTS WITH `def main()`

1. ✅ `cli.py` - Main CLI entry point (FIXED)
2. ⚠️ `scripts/20_train.py` - Unified training script (PARTIALLY FIXED, UNTESTED)
3. `scripts/00_make_splits.py` - Create deterministic 4-way splits
4. `scripts/25_threshold_sweep.py` - Threshold sweep on val_calib
5. `scripts/33_calibrate_gate.py` - Gate calibration with Platt scaling
6. `scripts/40_eval_selective.py` - Selective evaluation on val_test
7. `scripts/41_infer_gate.py` - Gate exit inference with Stage A/B cascade
8. `scripts/43_ab_test_peft.py` - A/B testing for PEFT
9. `scripts/44_explora_pretrain.py` - ExPLoRA pretraining (MAE)
10. `scripts/45_train_supervised_explora.py` - Supervised training with ExPLoRA backbone
11. `scripts/50_export_bundle.py` - Export model bundle

---

## 🚀 NEXT STEPS TO TEST PIPELINE

### Step 1: Run a tiny baseline training test
```bash
cd /home/sina/projects/miner_b/stage1_pro_modular_training_system
timeout 120 python3 cli.py \
  --mode train \
  --phase 1 \
  --exit_policy softmax \
  --epochs 1 \
  --max_batch_size 4 \
  --fallback_batch_size 4 \
  2>&1 | tee training_test.log
```

### Step 2: Verify outputs are created
Check for:
- `model_best.pth`
- `val_calib_logits.pt`
- `val_calib_labels.pt`
- `thresholds.json`
- `bundle.json`

---

## 📊 SUMMARY OF MY MISTAKES

1. **Blindly changing imports** without understanding the project structure
2. **Using absolute imports with package names** (`from stage1_pro_modular_training_system.config`) when relative or simple absolute imports were expected
3. **Not testing incrementally** - should have verified imports worked before making bulk changes
4. **Ignoring existing patterns** in the codebase - other scripts already had working import patterns
5. **Not running simple tests** like `python3 -c "from config import Stage1ProConfig"` before making changes

---

## ✅ CORRECT APPROACH (Going Forward)

1. First, examine existing working scripts to understand patterns
2. Test imports individually before integrating
3. Make minimal changes to match project conventions
4. Run incremental tests after each fix
5. Document what works and what doesn't

---

Generated: 2025-12-27
Project: stage1_pro_modular_training_system
Status: Scripts analyzed, patterns documented, ready for testing

