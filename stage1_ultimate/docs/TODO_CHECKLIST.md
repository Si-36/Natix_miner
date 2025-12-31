# 🚀 COMPLETE UPGRADE TODO CHECKLIST (203 Tasks)
**Date**: 2025-12-31  
**Status**: Ready for Implementation  
**Source**: Extracted from `MASTER_PLAN.md`  
**Standards**: 2025/2026 Best Practices (PyTorch 2.6+, Modern APIs, Type Hints)

---

## ⚠️ CRITICAL: PHASE ORDER CLARIFICATION

**There are TWO different orders - don't confuse them!**

### 1. **Implementation Order** (What to BUILD first)
This is the order tasks appear in this checklist:
```
Phase 2 (Tasks #9-35) → Phase 1 (Tasks #36-71) → Phase 4a (Tasks #72-105) → Phase 4c (Tasks #106-140) → Phase 5 (Tasks #141-160) → Phase 6 (Tasks #161-175)
```
**Why this order?** Phase 2 is easiest (fast win), Phase 1 validates training stack, then Phase 4a adds domain adaptation.

### 2. **Runtime Order** (What to RUN at inference)
This is the order phases execute when training:
```
phase4a_explora → phase1 → phase2 → phase5 → phase6
```
**Why this order?** Domain adaptation (phase4a) must happen BEFORE task training (phase1), then threshold/calibration after.

**IMPORTANT**: 
- ✅ Build phases in implementation order (this checklist)
- ✅ Run phases in runtime order (phase4 → phase1 → phase2 → phase5 → phase6)
- ✅ Phase 4a adapts backbone, Phase 1 trains on adapted backbone, Phase 2 optimizes threshold

---

## 📋 TABLE OF CONTENTS

- [Day 0: Setup (Tasks #1-8)](#day-0-setup-tasks-1-8)
- [Day 1: Phase 2 MCC Sweep (Tasks #9-35)](#day-1-phase-2-mcc-sweep-tasks-9-35)
- [Day 2: Training Optimizations (Tasks #36-71)](#day-2-training-optimizations-tasks-36-71)
- [Day 3-4: ExPLoRA SimCLR (Tasks #72-105)](#day-3-4-explora-simclr-tasks-72-105)
- [Day 5-6: CVFM Implementation (Tasks #106-140)](#day-5-6-cvfm-implementation-tasks-106-140)
- [Day 7: Phase 5 SCRC (Tasks #141-160)](#day-7-phase-5-scrc-tasks-141-160)
- [Day 8: Phase 6 Export (Tasks #161-175)](#day-8-phase-6-export-tasks-161-175)
- [Day 9-10: Evaluation Framework (Tasks #176-195)](#day-9-10-evaluation-framework-tasks-176-195)
- [2025/2026 Upgrades (Tasks #196-203)](#20252026-upgrades-tasks-196-203)

---

## 📝 PROGRESS TRACKING

- **Total Tasks**: 203
- **Completed**: 0
- **In Progress**: 0
- **Blocked**: 0
- **Skipped**: 0

**Completion Rate**: 0%

---

## DAY 0: SETUP (Tasks #1-8)

### Git & Directory Setup

- [ ] **Task #1**: Create git branch for upgrade
  - **File**: N/A (terminal command)
  - **Action**: `cd stage1_ultimate && git checkout -b upgrade-ultimate-2025`
  - **Verify**: `git branch` shows `* upgrade-ultimate-2025`
  - **Time**: 1 min

- [ ] **Task #2**: Commit current baseline
  - **File**: N/A (terminal command)
  - **Action**: `git add -A && git commit -m "Backup: Pre-upgrade baseline"`
  - **Verify**: `git log --oneline -1` shows backup commit
  - **Time**: 1 min

- [ ] **Task #3**: Create new source directories
  - **File**: N/A (terminal command)
  - **Action**: `mkdir -p src/peft src/streetvision/tta`
  - **Verify**: `ls -d src/peft src/streetvision/tta` both exist
  - **Time**: 1 min

- [ ] **Task #4**: Create config directories
  - **File**: N/A (terminal command)
  - **Action**: `mkdir -p configs/{phase2,phase4a,phase4b,phase4c,phase5,data,training,evaluation}`
  - **Verify**: All 8 directories exist in `configs/`
  - **Time**: 1 min

- [ ] **Task #5**: Verify Python version
  - **File**: N/A (terminal command)
  - **Action**: `python --version` (must be 3.11+)
  - **Verify**: Output shows `Python 3.11.x` or higher
  - **Time**: 1 min

- [ ] **Task #6**: Verify PyTorch version
  - **File**: N/A (terminal command)
  - **Action**: `python -c "import torch; print(torch.__version__)"` (must be 2.6+)
  - **Verify**: Output shows `2.6.x` or higher
  - **Time**: 1 min

- [ ] **Task #7**: Verify CUDA availability
  - **File**: N/A (terminal command)
  - **Action**: `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"`
  - **Verify**: CUDA=True, GPUs >= 1
  - **Time**: 1 min

- [ ] **Task #8**: Install/verify package structure
  - **File**: N/A (terminal command)
  - **Action**: `pip install -e .` (from stage1_ultimate/)
  - **Verify**: `python -c "from contracts.artifact_schema import ArtifactSchema; print('OK')"` succeeds
  - **Time**: 2 min

**Day 0 Total**: 8 tasks, ~10 minutes

---

## DAY 1: PHASE 2 MCC SWEEP (Tasks #9-35)

### Morning: ArtifactSchema Updates (Tasks #9-15)

- [ ] **Task #9**: Open `src/contracts/artifact_schema.py`
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Read current file, identify where to add Phase 2 properties
  - **Verify**: File exists and is readable
  - **Time**: 2 min

- [ ] **Task #10**: Add `phase2_dir` property
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `@property def phase2_dir(self) -> Path: return self.output_dir / "phase2"`
  - **Verify**: `python -c "from contracts.artifact_schema import ArtifactSchema; a=ArtifactSchema(Path('test')); print(a.phase2_dir)"` prints `test/phase2`
  - **Time**: 2 min

- [ ] **Task #11**: Add `thresholds_json` property
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `@property def thresholds_json(self) -> Path: return self.phase2_dir / "thresholds.json"`
  - **Verify**: Property returns correct path
  - **Time**: 1 min

- [ ] **Task #12**: Add `threshold_sweep_csv` property
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `@property def threshold_sweep_csv(self) -> Path: return self.phase2_dir / "threshold_sweep.csv"`
  - **Verify**: Property returns correct path
  - **Time**: 1 min

- [ ] **Task #13**: Add `mcc_curve_plot` property
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `@property def mcc_curve_plot(self) -> Path: return self.phase2_dir / "mcc_curve.png"`
  - **Verify**: Property returns correct path
  - **Time**: 1 min

- [ ] **Task #14**: Update `ensure_dirs()` to create phase2_dir
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `self.phase2_dir.mkdir(parents=True, exist_ok=True)` in `ensure_dirs()`
  - **Verify**: `artifacts.ensure_dirs()` creates `outputs/phase2/`
  - **Time**: 2 min

- [ ] **Task #15**: Test ArtifactSchema Phase 2 properties
  - **File**: N/A (test script)
  - **Action**: `python -c "from contracts.artifact_schema import ArtifactSchema; from pathlib import Path; a=ArtifactSchema(Path('test')); a.ensure_dirs(); assert a.phase2_dir.exists(); print('OK')"`
  - **Verify**: No errors, all properties work
  - **Time**: 2 min

### Morning: MCC Selection Function (Tasks #16-24)

- [ ] **Task #16**: Create `src/streetvision/eval/thresholds.py`
  - **File**: `src/streetvision/eval/thresholds.py` (NEW)
  - **Action**: Create file with imports: `import torch, numpy as np, pandas as pd, sklearn.metrics`
  - **Verify**: File exists and imports work
  - **Time**: 2 min

- [ ] **Task #17**: Add `select_threshold_max_mcc()` function signature
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add function with type hints:
    ```python
    def select_threshold_max_mcc(
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_thresholds: int = 5000,
        return_curve: bool = False,
    ) -> Tuple[float, float, Dict]:
    ```
  - **Verify**: Function signature is valid Python
  - **Time**: 3 min

- [ ] **Task #18**: Implement vectorized probability conversion
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add code: `probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()`
  - **Verify**: Code converts logits to probabilities correctly
  - **Time**: 2 min

- [ ] **Task #19**: Implement vectorized threshold grid
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add code: `thresholds = np.linspace(0, 1, n_thresholds)`
  - **Verify**: Creates 5000 thresholds by default
  - **Time**: 1 min

- [ ] **Task #20**: Implement vectorized MCC computation
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add vectorized confusion matrix computation (see MASTER_PLAN.md lines 2826-2845)
  - **Verify**: Computes MCC for all thresholds at once (no loop)
  - **Time**: 10 min

- [ ] **Task #21**: Add best threshold selection logic
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add `best_idx = np.argmax(mccs); best_threshold = float(thresholds[best_idx])`
  - **Verify**: Returns threshold with highest MCC
  - **Time**: 2 min

- [ ] **Task #22**: Add metrics computation at best threshold
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add confusion matrix + accuracy/precision/recall/F1/FNR/FPR computation
  - **Verify**: Returns complete metrics dict
  - **Time**: 5 min

- [ ] **Task #23**: Add `plot_mcc_curve()` function
  - **File**: `src/streetvision/eval/thresholds.py`
  - **Action**: Add matplotlib plotting function (see MASTER_PLAN.md lines 2885-2924)
  - **Verify**: Function creates 2-panel plot (MCC curve + confusion breakdown)
  - **Time**: 10 min

- [ ] **Task #24**: Test MCC selection function
  - **File**: N/A (test script)
  - **Action**: `python -c "from streetvision.eval.thresholds import select_threshold_max_mcc; import torch; logits=torch.randn(100,2); labels=torch.randint(0,2,(100,)); thresh,mcc,metrics=select_threshold_max_mcc(logits,labels,n_thresholds=100); print(f'Best: {thresh:.3f}, MCC: {mcc:.3f}')"`
  - **Verify**: No errors, returns valid threshold and MCC
  - **Time**: 3 min

### Morning: Config File (Tasks #25-27)

- [ ] **Task #25**: Create `configs/phase2/mcc.yaml`
  - **File**: `configs/phase2/mcc.yaml` (NEW)
  - **Action**: Create file with:
    ```yaml
    n_thresholds: 5000
    optimize_metric: "mcc"
    save_sweep_curve: true
    ```
  - **Verify**: File exists and is valid YAML
  - **Time**: 2 min

- [ ] **Task #26**: Add Phase 2 config to main config
  - **File**: `configs/config.yaml`
  - **Action**: Add `phase2: ${phase2:mcc}` or equivalent Hydra config group
  - **Verify**: Hydra can load phase2 config
  - **Time**: 3 min

- [ ] **Task #27**: Test config loading
  - **File**: N/A (test script)
  - **Action**: `python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('configs/phase2/mcc.yaml'); print(cfg.n_thresholds)"`
  - **Verify**: Prints `5000`
  - **Time**: 2 min

### Afternoon: Phase 2 Step Update (Tasks #28-35)

- [ ] **Task #28**: Open `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Read current implementation
  - **Verify**: File exists
  - **Time**: 5 min

- [ ] **Task #29**: Import `select_threshold_max_mcc`
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Add `from streetvision.eval.thresholds import select_threshold_max_mcc`
  - **Verify**: Import succeeds
  - **Time**: 1 min

- [ ] **Task #30**: Replace threshold sweep logic with MCC optimization
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Replace existing sweep loop with `select_threshold_max_mcc()` call
  - **Verify**: Uses MCC optimization instead of selective accuracy
  - **Time**: 10 min

- [ ] **Task #31**: Write validator-compatible `thresholds.json`
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Write JSON with schema: `{"policy_type": "threshold", "threshold": 0.5, "metrics": {...}}`
  - **Verify**: JSON matches validator schema (see MASTER_PLAN.md validators section)
  - **Time**: 10 min

- [ ] **Task #32**: Save threshold sweep CSV
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Save full sweep curve to CSV: `threshold,mcc,tp,tn,fp,fn`
  - **Verify**: CSV has 5000 rows (or n_thresholds rows)
  - **Time**: 5 min

- [ ] **Task #33**: Save MCC curve plot
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Call `plot_mcc_curve()` and save to `artifacts.mcc_curve_plot`
  - **Verify**: PNG file created
  - **Time**: 3 min

- [ ] **Task #34**: Add atomic write + manifest
  - **File**: `src/streetvision/pipeline/steps/sweep_thresholds.py`
  - **Action**: Use `write_json_atomic()` and `create_step_manifest()` (see MASTER_PLAN.md)
  - **Verify**: Files written atomically, manifest.json created last
  - **Time**: 5 min

- [ ] **Task #35**: Test Phase 2 end-to-end
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase2] phase2.n_thresholds=100`
  - **Verify**: Phase 2 completes, creates `thresholds.json`, `threshold_sweep.csv`, `mcc_curve.png`, `manifest.json`
  - **Time**: 5 min

**Day 1 Total**: 27 tasks, ~4 hours

---

## DAY 2: TRAINING OPTIMIZATIONS (Tasks #36-71)

### Morning: Model Updates (Tasks #36-50)

- [ ] **Task #36**: Open `src/models/module.py`
  - **File**: `src/models/module.py`
  - **Action**: Read current implementation
  - **Verify**: File exists
  - **Time**: 5 min

- [ ] **Task #37**: Add `FocalLoss` class
  - **File**: `src/models/module.py`
  - **Action**: Add FocalLoss class (see MASTER_PLAN.md lines 3635-3658)
  - **Verify**: Class implements focal loss formula correctly
  - **Time**: 10 min

- [ ] **Task #38**: Add loss function selection logic
  - **File**: `src/models/module.py`
  - **Action**: Add `if config.training.loss.name == 'focal': self.criterion = FocalLoss(...)` logic
  - **Verify**: Can switch between CE, focal, weighted_ce
  - **Time**: 5 min

- [ ] **Task #39**: Add `create_model_with_compile()` function
  - **File**: `src/models/module.py`
  - **Action**: Add function that wraps model with `torch.compile()` if enabled (see MASTER_PLAN.md lines 3784-3798)
  - **Verify**: Function compiles model when `config.hardware.compile=True`
  - **Time**: 5 min

- [ ] **Task #40**: Update `configure_optimizers()` with cosine warmup
  - **File**: `src/models/module.py`
  - **Action**: Add cosine annealing with linear warmup scheduler (see MASTER_PLAN.md lines 3748-3779)
  - **Verify**: Scheduler has warmup phase then cosine decay
  - **Time**: 10 min

- [ ] **Task #41**: Add gradient accumulation support
  - **File**: `src/models/module.py`
  - **Action**: Ensure `training_step()` can accumulate gradients (Lightning handles this automatically)
  - **Verify**: Check that `accumulate_grad_batches` is passed to Trainer
  - **Time**: 3 min

- [ ] **Task #42**: Test FocalLoss
  - **File**: N/A (test script)
  - **Action**: `python -c "from models.module import FocalLoss; import torch; loss=FocalLoss(); logits=torch.randn(10,2); labels=torch.randint(0,2,(10,)); print(loss(logits,labels))"`
  - **Verify**: Returns scalar loss value
  - **Time**: 2 min

- [ ] **Task #43**: Test model compilation
  - **File**: N/A (test script)
  - **Action**: `python -c "from models.module import create_model_with_compile; from omegaconf import OmegaConf; cfg=OmegaConf.create({'hardware':{'compile':True},'model':{...}}); model=create_model_with_compile(cfg); print('OK')"`
  - **Verify**: Model compiles without errors
  - **Time**: 3 min

- [ ] **Task #44**: Create `configs/training/optimization.yaml`
  - **File**: `configs/training/optimization.yaml` (NEW)
  - **Action**: Create config with mixed_precision, compile, gradient_accumulation, loss settings (see MASTER_PLAN.md lines 3960-4007)
  - **Verify**: Config is valid YAML
  - **Time**: 5 min

- [ ] **Task #45**: Add PyTorch 2.6 compile stance config
  - **File**: `configs/training/optimization.yaml`
  - **Action**: Add `hardware.compiler.stance: performance` (for `torch.compiler.set_stance()`)
  - **Verify**: Config includes stance setting
  - **Time**: 2 min

- [ ] **Task #46**: Update main config to include training optimization
  - **File**: `configs/config.yaml`
  - **Action**: Add `training: ${training:optimization}` or equivalent
  - **Verify**: Hydra can load training config
  - **Time**: 2 min

- [ ] **Task #47**: Test config loading
  - **File**: N/A (test script)
  - **Action**: `python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('configs/training/optimization.yaml'); print(cfg.mixed_precision.enabled)"`
  - **Verify**: Prints `True` or config value
  - **Time**: 2 min

- [ ] **Task #48**: Create `src/data/augmentation.py`
  - **File**: `src/data/augmentation.py` (NEW)
  - **Action**: Create file with imports: `torchvision.transforms.v2 as v2, torch.nn.functional as F`
  - **Verify**: File exists
  - **Time**: 2 min

- [ ] **Task #49**: Add `get_train_transforms()` function
  - **File**: `src/data/augmentation.py`
  - **Action**: Add function that builds transform pipeline from config (see MASTER_PLAN.md lines 3480-3549)
  - **Verify**: Function returns `v2.Compose()` transform
  - **Time**: 15 min

- [ ] **Task #50**: Add `get_val_transforms()` function
  - **File**: `src/data/augmentation.py`
  - **Action**: Add minimal validation transforms (resize, center crop, normalize)
  - **Verify**: Function returns transform pipeline
  - **Time**: 5 min

### Afternoon: Training Script Updates (Tasks #51-71)

- [ ] **Task #51**: Open `src/streetvision/pipeline/steps/train_baseline.py`
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Read current implementation
  - **Verify**: File exists
  - **Time**: 5 min

- [ ] **Task #52**: Add BF16 auto-selection logic
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Add code to detect GPU capability and set `precision="bf16-mixed"` if supported
  - **Verify**: BF16 enabled on A100/H100, FP32 fallback on older GPUs
  - **Time**: 10 min

- [ ] **Task #53**: Pass gradient accumulation to Trainer
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Add `accumulate_grad_batches=cfg.training.gradient_accumulation_steps` to Trainer
  - **Verify**: Effective batch size = batch_size × gradient_accumulation_steps
  - **Time**: 3 min

- [ ] **Task #54**: Add torch.compile integration
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Call `create_model_with_compile()` instead of direct model creation
  - **Verify**: Model compiles if `config.hardware.compile=True`
  - **Time**: 5 min

- [ ] **Task #55**: Add PyTorch 2.6 stance setting
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Add `if cfg.hardware.compiler.get('stance'): torch.compiler.set_stance(cfg.hardware.compiler.stance)` before compile
  - **Verify**: Stance is set before compilation
  - **Time**: 3 min

- [ ] **Task #56**: Generate VAL_CALIB logits after training
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Add code to run inference on VAL_CALIB split and save logits/labels
  - **Verify**: Creates `val_calib_logits.pt` and `val_calib_labels.pt`
  - **Time**: 10 min

- [ ] **Task #57**: Use configurable transforms from augmentation.py
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Import and use `get_train_transforms(cfg)` and `get_val_transforms(cfg)`
  - **Verify**: Transforms come from config, not hardcoded
  - **Time**: 5 min

- [ ] **Task #58**: Create `configs/data/augmentation.yaml`
  - **File**: `configs/data/augmentation.yaml` (NEW)
  - **Action**: Create config with all augmentation settings (see MASTER_PLAN.md lines 4011-4068)
  - **Verify**: Config includes train/val augmentation settings
  - **Time**: 10 min

- [ ] **Task #59**: Add TrivialAugmentWide v2 config
  - **File**: `configs/data/augmentation.yaml`
  - **Action**: Add `trivial_augment_wide: {enabled: true, num_magnitude_bins: 31}` section
  - **Verify**: Config includes TrivialAugmentWide settings
  - **Time**: 3 min

- [ ] **Task #60**: Add AugMix config
  - **File**: `configs/data/augmentation.yaml`
  - **Action**: Add `aug_mix: {enabled: true, severity: 3, mixture_width: 3, alpha: 1.0}` section
  - **Verify**: Config includes AugMix settings
  - **Time**: 3 min

- [ ] **Task #61**: Update augmentation.py to use TrivialAugmentWide v2
  - **File**: `src/data/augmentation.py`
  - **Action**: Replace old transforms with `v2.TrivialAugmentWide()` when enabled
  - **Verify**: Uses torchvision.transforms.v2 API
  - **Time**: 10 min

- [ ] **Task #62**: Add AugMix implementation
  - **File**: `src/data/augmentation.py`
  - **Action**: Add AugMix transform class or use `v2.AugMix()` if available
  - **Verify**: AugMix transform works
  - **Time**: 10 min

- [ ] **Task #63**: Add RandomErasing with MCC-safe gate
  - **File**: `src/data/augmentation.py`
  - **Action**: Add `v2.RandomErasing()` with configurable probability, add ablation gate check
  - **Verify**: RandomErasing can be enabled/disabled
  - **Time**: 5 min

- [ ] **Task #64**: Update datamodule to use new transforms
  - **File**: `src/data/datamodule.py`
  - **Action**: Import `get_train_transforms` and `get_val_transforms`, use them in dataset creation
  - **Verify**: Datasets use config-driven transforms
  - **Time**: 5 min

- [ ] **Task #65**: Test augmentation pipeline
  - **File**: N/A (test script)
  - **Action**: `python -c "from data.augmentation import get_train_transforms; from omegaconf import OmegaConf; cfg=OmegaConf.load('configs/data/augmentation.yaml'); t=get_train_transforms(cfg); print('OK')"`
  - **Verify**: Transforms load and can be applied to images
  - **Time**: 3 min

- [ ] **Task #66**: Test Phase 1 with optimizations
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase1] training.epochs=1 training.mixed_precision.enabled=true`
  - **Verify**: Training runs with BF16, no NaN errors
  - **Time**: 10 min

- [ ] **Task #67**: Verify VAL_CALIB logits generated
  - **File**: N/A (check files)
  - **Action**: After Phase 1, check `outputs/phase1/val_calib_logits.pt` exists
  - **Verify**: File exists and can be loaded
  - **Time**: 2 min

- [ ] **Task #68**: Benchmark training speed improvement
  - **File**: N/A (timing test)
  - **Action**: Time Phase 1 with/without compile and BF16, compare speeds
  - **Verify**: Compile + BF16 gives 2-3× speedup
  - **Time**: 15 min

- [ ] **Task #69**: Test focal loss
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase1] training.loss.name=focal training.epochs=1`
  - **Verify**: Training uses focal loss, no errors
  - **Time**: 5 min

- [ ] **Task #70**: Test gradient accumulation
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase1] training.gradient_accumulation_steps=4 training.epochs=1`
  - **Verify**: Effective batch size increases, training slower per step but same per epoch
  - **Time**: 5 min

- [ ] **Task #71**: Commit Day 2 changes
  - **File**: N/A (git command)
  - **Action**: `git add -A && git commit -m "feat: Phase 1 training optimizations (BF16, compile, focal loss, augmentation)"`
  - **Verify**: Changes committed
  - **Time**: 1 min

**Day 2 Total**: 36 tasks, ~4 hours

---

## DAY 3-4: EXPLORA SIMCLR (Tasks #72-105)

### Day 3: Core Implementation (Tasks #72-88)

- [ ] **Task #72**: Create `src/peft/explora_domain.py`
  - **File**: `src/peft/explora_domain.py` (NEW)
  - **Action**: Create file with imports: `torch, torch.nn, peft`
  - **Verify**: File exists
  - **Time**: 2 min

- [ ] **Task #73**: Add `ExPLoRAConfig` class
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Add config class with LoRA settings (see MASTER_PLAN.md lines 2317-2341)
  - **Verify**: Class creates LoraConfig correctly
  - **Time**: 10 min

- [ ] **Task #74**: Add `SimCLRLoss` class
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Add SimCLR NT-Xent loss with vectorized computation (see MASTER_PLAN.md lines 2344-2396)
  - **Verify**: Loss computes contrastive loss correctly
  - **Time**: 15 min

- [ ] **Task #75**: Add projection head helper
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Add function to create MLP projection head (2-layer, LayerNorm, GELU)
  - **Verify**: Creates projection head with correct dimensions
  - **Time**: 5 min

- [ ] **Task #76**: Add strong augmentation function
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Add function that creates SimCLR-style strong augmentations (crop, color jitter, blur, grayscale)
  - **Verify**: Augmentations create two different views of same image
  - **Time**: 10 min

- [ ] **Task #77**: Create `src/models/explora_module.py`
  - **File**: `src/models/explora_module.py` (NEW)
  - **Action**: Create LightningModule for SimCLR training (see MASTER_PLAN.md lines 2401-2532)
  - **Verify**: Module inherits from pl.LightningModule
  - **Time**: 5 min

- [ ] **Task #78**: Add backbone loading with ExPLoRA
  - **File**: `src/models/explora_module.py`
  - **Action**: Load DINOv3 backbone, apply LoRA adapters using ExPLoRAConfig
  - **Verify**: Backbone has trainable LoRA parameters
  - **Time**: 10 min

- [ ] **Task #79**: Add projection head to module
  - **File**: `src/models/explora_module.py`
  - **Action**: Add projection head that maps backbone features to projection_dim
  - **Verify**: Projection head has correct input/output dimensions
  - **Time**: 5 min

- [ ] **Task #80**: Add SimCLR training step
  - **File**: `src/models/explora_module.py`
  - **Action**: Implement `training_step()` that generates two views, computes features, applies projection, computes SimCLR loss
  - **Verify**: Training step returns loss scalar
  - **Time**: 15 min

- [ ] **Task #81**: Add DDP all-gather for negatives
  - **File**: `src/models/explora_module.py`
  - **Action**: Add code to gather embeddings from all GPUs for larger negative set (see MASTER_PLAN.md)
  - **Verify**: Effective batch size increases with DDP
  - **Time**: 10 min

- [ ] **Task #82**: Add validation step
  - **File**: `src/models/explora_module.py`
  - **Action**: Add `validation_step()` that computes SimCLR loss on val set
  - **Verify**: Validation loss logged
  - **Time**: 5 min

- [ ] **Task #83**: Add optimizer configuration
  - **File**: `src/models/explora_module.py`
  - **Action**: Add `configure_optimizers()` with AdamW and cosine scheduler
  - **Verify**: Optimizer and scheduler configured correctly
  - **Time**: 5 min

- [ ] **Task #84**: Create `configs/phase4a/explora.yaml`
  - **File**: `configs/phase4a/explora.yaml` (NEW)
  - **Action**: Create config with SimCLR and ExPLoRA settings (see MASTER_PLAN.md lines 3805-3850)
  - **Verify**: Config includes all required settings
  - **Time**: 5 min

- [ ] **Task #85**: Update `src/streetvision/pipeline/steps/train_explora.py`
  - **File**: `src/streetvision/pipeline/steps/train_explora.py`
  - **Action**: Remove "not implemented" fallback, use ExPLoRAModule for training
  - **Verify**: Phase 4a uses SimCLR, not supervised CE
  - **Time**: 10 min

- [ ] **Task #86**: Add gradient accumulation support
  - **File**: `src/streetvision/pipeline/steps/train_explora.py`
  - **Action**: Pass `accumulate_grad_batches` to Trainer for larger effective batch
  - **Verify**: Effective batch size = batch_size × num_gpus × gradient_accumulation
  - **Time**: 3 min

- [ ] **Task #87**: Save domain-adapted backbone
  - **File**: `src/streetvision/pipeline/steps/train_explora.py`
  - **Action**: After training, merge LoRA adapters and save backbone checkpoint
  - **Verify**: Checkpoint saved to `artifacts.explora_checkpoint`
  - **Time**: 5 min

- [ ] **Task #88**: Test Phase 4a SimCLR training
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase4a_explora] phase4a.num_epochs=5`
  - **Verify**: SimCLR loss decreases, no errors
  - **Time**: 15 min

### Day 4: Testing & Integration (Tasks #89-105)

- [ ] **Task #89**: Verify SimCLR loss decreases
  - **File**: N/A (check logs)
  - **Action**: Check training logs show decreasing SimCLR loss
  - **Verify**: Loss starts high (>5) and decreases to <2
  - **Time**: 2 min

- [ ] **Task #90**: Verify no memory bank used
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Confirm SimCLRLoss uses only in-batch negatives
  - **Verify**: No queue or memory bank code present
  - **Time**: 2 min

- [ ] **Task #91**: Test DDP all-gather (if multi-GPU)
  - **File**: N/A (multi-GPU test)
  - **Action**: Run Phase 4a on 2 GPUs, verify effective batch size doubles
  - **Verify**: Negative set size = batch_size × num_gpus × 2 (two views)
  - **Time**: 10 min

- [ ] **Task #92**: Verify domain-adapted checkpoint saved
  - **File**: N/A (check files)
  - **Action**: Check `outputs/phase4a_explora/explora_checkpoint.pth` exists
  - **Verify**: File exists and can be loaded
  - **Time**: 2 min

- [ ] **Task #93**: Test loading domain-adapted backbone in Phase 1
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Add option to load ExPLoRA checkpoint before Phase 1 training
  - **Verify**: Phase 1 can start from ExPLoRA-adapted backbone
  - **Time**: 10 min

- [ ] **Task #94**: Add config option for ExPLoRA checkpoint
  - **File**: `configs/model/dinov3_vith16.yaml`
  - **Action**: Add `init_from_explora: true` and `explora_checkpoint_path: null` options
  - **Verify**: Config allows loading ExPLoRA checkpoint
  - **Time**: 3 min

- [ ] **Task #95**: Test full Phase 4a → Phase 1 pipeline (RUNTIME ORDER)
  - **File**: N/A (CLI test)
  - **Action**: Run `python scripts/train_cli_v2.py pipeline.phases=[phase4a_explora,phase1]` (phase4a FIRST, then phase1)
  - **Note**: This tests RUNTIME order - phase4a adapts backbone, phase1 trains on adapted backbone
  - **Verify**: Phase 1 starts from ExPLoRA-adapted backbone, MCC improves
  - **Time**: 20 min

- [ ] **Task #96**: Measure MCC improvement from ExPLoRA
  - **File**: N/A (comparison)
  - **Action**: Compare Phase 1 MCC with/without ExPLoRA pre-training
  - **Verify**: ExPLoRA gives +6-8% MCC improvement
  - **Time**: 15 min

- [ ] **Task #97**: Add hard negative mining placeholder
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Add TODO comment for future hard negative mining (class-balanced sampling)
  - **Verify**: Comment documents future enhancement
  - **Time**: 1 min

- [ ] **Task #98**: Update ArtifactSchema for Phase 4a
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `phase4a_dir`, `explora_checkpoint` properties
  - **Verify**: Properties return correct paths
  - **Time**: 3 min

- [ ] **Task #99**: Add Phase 4a to ensure_dirs()
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `self.phase4a_dir.mkdir(...)` in `ensure_dirs()`
  - **Verify**: Phase 4a directory created
  - **Time**: 2 min

- [ ] **Task #100**: Add manifest tracking for Phase 4a
  - **File**: `src/streetvision/pipeline/steps/train_explora.py`
  - **Action**: Add `create_step_manifest()` call at end of Phase 4a
  - **Verify**: Manifest.json created with git SHA, config hash, metrics
  - **Time**: 5 min

- [ ] **Task #101**: Test Phase 4a with different batch sizes
  - **File**: N/A (CLI test)
  - **Action**: Test with batch_size=32, 64, 128, verify SimCLR still works
  - **Verify**: Smaller batches still train (with gradient accumulation)
  - **Time**: 10 min

- [ ] **Task #102**: Add SimCLR temperature tuning
  - **File**: `configs/phase4a/explora.yaml`
  - **Action**: Add comment about temperature tuning (0.07-0.2 range)
  - **Verify**: Config documents temperature parameter
  - **Time**: 2 min

- [ ] **Task #103**: Verify no data leakage
  - **File**: `src/streetvision/pipeline/steps/train_explora.py`
  - **Action**: Confirm Phase 4a only uses TRAIN split, never VAL_CALIB/VAL_TEST
  - **Verify**: No validation on wrong splits
  - **Time**: 3 min

- [ ] **Task #104**: Add unit tests for SimCLR loss
  - **File**: `tests/unit/test_explora.py` (NEW)
  - **Action**: Add tests for SimCLRLoss computation
  - **Verify**: Tests pass
  - **Time**: 10 min

- [ ] **Task #105**: Commit Day 3-4 changes
  - **File**: N/A (git command)
  - **Action**: `git add -A && git commit -m "feat: Phase 4a ExPLoRA SimCLR domain adaptation"`
  - **Verify**: Changes committed
  - **Time**: 1 min

**Day 3-4 Total**: 34 tasks, ~8 hours

---

## DAY 5-6: CVFM IMPLEMENTATION (Tasks #106-140)

### Day 5: CVFM Core (Tasks #106-125)

- [ ] **Task #106**: Create `src/streetvision/tta/simple_cvfm.py`
  - **File**: `src/streetvision/tta/simple_cvfm.py` (NEW)
  - **Action**: Create file with imports: `torch, torch.nn, torch.nn.functional`
  - **Verify**: File exists
  - **Time**: 2 min

- [ ] **Task #107**: Add `InferenceCVFM` class
  - **File**: `src/streetvision/tta/simple_cvfm.py`
  - **Action**: Add class for inference-only CVFM fusion (see MASTER_PLAN.md lines 2589-2714)
  - **Verify**: Class has `forward()` method
  - **Time**: 10 min

- [ ] **Task #108**: Implement simple_mean fusion mode
  - **File**: `src/streetvision/tta/simple_cvfm.py`
  - **Action**: Add mode that averages all view logits
  - **Verify**: Returns averaged logits
  - **Time**: 5 min

- [ ] **Task #109**: Implement weighted_uncertainty fusion mode
  - **File**: `src/streetvision/tta/simple_cvfm.py`
  - **Action**: Add mode that weights views by inverse entropy (low uncertainty = high weight)
  - **Verify**: High-confidence views weighted more
  - **Time**: 10 min

- [ ] **Task #110**: Implement content_aware fusion mode
  - **File**: `src/streetvision/tta/simple_cvfm.py`
  - **Action**: Add mode that weights views by content box area (larger boxes = more weight)
  - **Verify**: Views with more content weighted more
  - **Time**: 10 min

- [ ] **Task #111**: Create `src/streetvision/tta/learned_cvfm.py`
  - **File**: `src/streetvision/tta/learned_cvfm.py` (NEW)
  - **Action**: Create file for trainable CVFM
  - **Verify**: File exists
  - **Time**: 2 min

- [ ] **Task #112**: Add `TrainableCVFM` class
  - **File**: `src/streetvision/tta/learned_cvfm.py`
  - **Action**: Add class with MLP-based fusion module (see MASTER_PLAN.md lines 2715-2815)
  - **Verify**: Class inherits from nn.Module
  - **Time**: 10 min

- [ ] **Task #113**: Implement shared encoder
  - **File**: `src/streetvision/tta/learned_cvfm.py`
  - **Action**: Add encoder that projects all views to shared latent space
  - **Verify**: Encoder maps features to latent_dim
  - **Time**: 10 min

- [ ] **Task #114**: Implement view-specific decoders
  - **File**: `src/streetvision/tta/learned_cvfm.py`
  - **Action**: Add decoders that reconstruct view-specific features from latent
  - **Verify**: Decoders map latent back to feature_dim
  - **Time**: 10 min

- [ ] **Task #115**: Implement learned view weights
  - **File**: `src/streetvision/tta/learned_cvfm.py`
  - **Action**: Add learnable parameters for view importance weights
  - **Verify**: Weights are trainable parameters
  - **Time**: 5 min

- [ ] **Task #116**: Add forward pass with fusion
  - **File**: `src/streetvision/tta/learned_cvfm.py`
  - **Action**: Implement forward() that encodes, averages in latent, decodes, fuses with weights
  - **Verify**: Forward returns fused features and weights
  - **Time**: 15 min

- [ ] **Task #117**: Create `configs/phase4c/cvfm.yaml`
  - **File**: `configs/phase4c/cvfm.yaml` (NEW)
  - **Action**: Create config with CVFM settings (see MASTER_PLAN.md lines 3880-3914)
  - **Verify**: Config includes trained/inference mode settings
  - **Time**: 5 min

- [ ] **Task #118**: Create `src/streetvision/pipeline/steps/train_cvfm.py`
  - **File**: `src/streetvision/pipeline/steps/train_cvfm.py` (NEW)
  - **Action**: Create training script for CVFM fusion module
  - **Verify**: File exists
  - **Time**: 2 min

- [ ] **Task #119**: Add CVFM training loop
  - **File**: `src/streetvision/pipeline/steps/train_cvfm.py`
  - **Action**: Implement training that freezes backbone+head, trains only CVFM fusion weights
  - **Verify**: Only CVFM parameters have requires_grad=True
  - **Time**: 15 min

- [ ] **Task #120**: Add validation on VAL_SELECT
  - **File**: `src/streetvision/pipeline/steps/train_cvfm.py`
  - **Action**: Add validation loop that evaluates CVFM on VAL_SELECT (NOT VAL_CALIB)
  - **Verify**: Validation uses correct split
  - **Time**: 5 min

- [ ] **Task #121**: Save CVFM weights
  - **File**: `src/streetvision/pipeline/steps/train_cvfm.py`
  - **Action**: Save trained CVFM weights to `artifacts.cvfm_weights`
  - **Verify**: Weights saved correctly
  - **Time**: 3 min

- [ ] **Task #122**: Update `src/models/multi_view.py`
  - **File**: `src/models/multi_view.py`
  - **Action**: Read current implementation
  - **Verify**: File exists
  - **Time**: 5 min

- [ ] **Task #123**: Add CVFM aggregator integration
  - **File**: `src/models/multi_view.py`
  - **Action**: Add code to use InferenceCVFM or TrainableCVFM based on config mode
  - **Verify**: Multi-view model can use CVFM fusion
  - **Time**: 15 min

- [ ] **Task #124**: Pass per-view features to CVFM
  - **File**: `src/models/multi_view.py`
  - **Action**: Modify forward() to extract and pass per-view features to CVFM aggregator
  - **Verify**: CVFM receives features from all views
  - **Time**: 10 min

- [ ] **Task #125**: Test CVFM inference mode
  - **File**: N/A (test script)
  - **Action**: Test InferenceCVFM with dummy features
  - **Verify**: CVFM fuses features correctly
  - **Time**: 5 min

### Day 6: Integration & Testing (Tasks #126-140)

- [ ] **Task #126**: Test CVFM training
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase4c_cvfm] phase4c.epochs=1`
  - **Verify**: CVFM training completes, weights saved
  - **Time**: 10 min

- [ ] **Task #127**: Verify CVFM weights loaded in inference
  - **File**: `src/models/multi_view.py`
  - **Action**: Add code to load CVFM weights if available
  - **Verify**: Trained CVFM weights loaded correctly
  - **Time**: 5 min

- [ ] **Task #128**: Test multi-view with CVFM
  - **File**: N/A (test script)
  - **Action**: Test MultiViewDINOv3 with CVFM fusion enabled
  - **Verify**: Multi-view inference uses CVFM
  - **Time**: 5 min

- [ ] **Task #129**: Measure MCC improvement from CVFM
  - **File**: N/A (comparison)
  - **Action**: Compare MCC with/without CVFM fusion
  - **Verify**: CVFM gives +8-12% MCC improvement
  - **Time**: 15 min

- [ ] **Task #130**: Update ArtifactSchema for Phase 4c
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `phase4c_dir`, `cvfm_weights` properties
  - **Verify**: Properties return correct paths
  - **Time**: 3 min

- [ ] **Task #131**: Add Phase 4c to ensure_dirs()
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `self.phase4c_dir.mkdir(...)` in `ensure_dirs()`
  - **Verify**: Phase 4c directory created
  - **Time**: 2 min

- [ ] **Task #132**: Add manifest tracking for Phase 4c
  - **File**: `src/streetvision/pipeline/steps/train_cvfm.py`
  - **Action**: Add `create_step_manifest()` call at end
  - **Verify**: Manifest.json created
  - **Time**: 3 min

- [ ] **Task #133**: Verify no data leakage in CVFM training
  - **File**: `src/streetvision/pipeline/steps/train_cvfm.py`
  - **Action**: Confirm CVFM trains on TRAIN, validates on VAL_SELECT only
  - **Verify**: Never uses VAL_CALIB or VAL_TEST
  - **Time**: 3 min

- [ ] **Task #134**: Add unit tests for CVFM
  - **File**: `tests/unit/test_cvfm.py` (NEW)
  - **Action**: Add tests for InferenceCVFM and TrainableCVFM
  - **Verify**: Tests pass
  - **Time**: 10 min

- [ ] **Task #135**: Test different CVFM modes
  - **File**: N/A (CLI test)
  - **Action**: Test simple_mean, weighted_uncertainty, content_aware, trained modes
  - **Verify**: All modes work correctly
  - **Time**: 10 min

- [ ] **Task #136**: Add FlexAttention placeholder (optional)
  - **File**: `src/streetvision/tta/flex_cvfm.py` (NEW, optional)
  - **Action**: Create placeholder for FlexAttention-based CVFM (see `docs/allstepsoffupgrade/06_flexattention_cvfm.md`)
  - **Verify**: File created with TODO comment
  - **Time**: 5 min

- [ ] **Task #137**: Document CVFM mode selection
  - **File**: `configs/phase4c/cvfm.yaml`
  - **Action**: Add comments explaining when to use each mode
  - **Verify**: Config documents mode selection
  - **Time**: 2 min

- [ ] **Task #138**: Add CVFM to model config
  - **File**: `configs/model/dinov3_vith16.yaml`
  - **Action**: Add `multiview.cvfm.mode: trained` option
  - **Verify**: Config allows enabling CVFM
  - **Time**: 3 min

- [ ] **Task #139**: Test full pipeline Phase 4a → Phase 1 → Phase 4c (RUNTIME ORDER)
  - **File**: N/A (CLI test)
  - **Action**: Run `python scripts/train_cli_v2.py pipeline.phases=[phase4a_explora,phase1,phase4c_cvfm]` (RUNTIME order)
  - **Note**: phase4a FIRST (domain adaptation), then phase1 (task training), then phase4c (CVFM fusion)
  - **Verify**: Pipeline completes successfully, MCC improves at each stage
  - **Time**: 30 min

- [ ] **Task #140**: Commit Day 5-6 changes
  - **File**: N/A (git command)
  - **Action**: `git add -A && git commit -m "feat: Phase 4c CVFM fusion (inference + trained)"`
  - **Verify**: Changes committed
  - **Time**: 1 min

**Day 5-6 Total**: 35 tasks, ~8 hours

---

## DAY 7: PHASE 5 SCRC (Tasks #141-160)

- [ ] **Task #141**: Open `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Read current implementation
  - **Verify**: File exists
  - **Time**: 5 min

- [ ] **Task #142**: Fix ECE computation function
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Fix `_compute_ece()` function (current has bugs, see MASTER_PLAN.md line 3049)
  - **Verify**: ECE computed correctly
  - **Time**: 10 min

- [ ] **Task #143**: Implement isotonic regression calibration
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add IsotonicRegression calibrator (see MASTER_PLAN.md lines 2987-3002)
  - **Verify**: Calibrator fits and predicts correctly
  - **Time**: 10 min

- [ ] **Task #144**: Add smooth isotonic (PCHIP) option
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add option to use PchipInterpolator for smooth calibration curve
  - **Verify**: Smooth isotonic prevents overfitting
  - **Time**: 10 min

- [ ] **Task #145**: Implement temperature scaling
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add temperature scaling calibrator (optimize T on VAL_CALIB)
  - **Verify**: Temperature scaling improves ECE
  - **Time**: 10 min

- [ ] **Task #146**: Add Platt scaling option
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add logistic regression (Platt) calibrator
  - **Verify**: Platt scaling works
  - **Time**: 5 min

- [ ] **Task #147**: Add Beta calibration option
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add Beta calibrator (more flexible than Platt)
  - **Verify**: Beta calibration works
  - **Time**: 5 min

- [ ] **Task #148**: Implement multi-objective calibration ensemble
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add tiered calibration: Tier 1 (isotonic+temperature) → Tier 2 (platt+beta) → Tier 3 (ensemble with learnable weights)
  - **Verify**: Ensemble combines multiple calibrators
  - **Time**: 20 min

- [ ] **Task #149**: Write validator-compatible policy JSON
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Write `scrc_params.json` with schema: `{"policy_type": "scrc", "scrc_params": {...}, "metrics": {...}}`
  - **Verify**: JSON matches validator schema
  - **Time**: 10 min

- [ ] **Task #150**: Save calibrator pickle file
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Save calibrator to `scrc_params.pkl` using pickle
  - **Verify**: Pickle file can be loaded
  - **Time**: 3 min

- [ ] **Task #151**: Compute ECE before/after calibration
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Compute and log ECE before and after calibration
  - **Verify**: ECE improves after calibration
  - **Time**: 5 min

- [ ] **Task #152**: Create `configs/phase5/scrc.yaml`
  - **File**: `configs/phase5/scrc.yaml` (NEW)
  - **Action**: Create config with calibration method settings (see MASTER_PLAN.md lines 3942-3956)
  - **Verify**: Config includes method selection
  - **Time**: 5 min

- [ ] **Task #153**: Add multi-objective calibration config
  - **File**: `configs/phase5/scrc.yaml`
  - **Action**: Add settings for tiered calibration ensemble
  - **Verify**: Config allows selecting ensemble method
  - **Time**: 5 min

- [ ] **Task #154**: Update ArtifactSchema for Phase 5
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `phase5_dir`, `scrc_params_json`, `scrc_params_pkl` properties
  - **Verify**: Properties return correct paths
  - **Time**: 3 min

- [ ] **Task #155**: Add Phase 5 to ensure_dirs()
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `self.phase5_dir.mkdir(...)` in `ensure_dirs()`
  - **Verify**: Phase 5 directory created
  - **Time**: 2 min

- [ ] **Task #156**: Add manifest tracking for Phase 5
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add `create_step_manifest()` call at end
  - **Verify**: Manifest.json created
  - **Time**: 3 min

- [ ] **Task #157**: Test Phase 5 calibration
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase5]`
  - **Verify**: Calibration completes, ECE improves
  - **Time**: 5 min

- [ ] **Task #158**: Verify validator compatibility
  - **File**: N/A (test script)
  - **Action**: Load `scrc_params.json` and validate with PolicyValidator
  - **Verify**: Validator accepts policy
  - **Time**: 5 min

- [ ] **Task #159**: Test different calibration methods
  - **File**: N/A (CLI test)
  - **Action**: Test isotonic, temperature, platt, beta, ensemble methods
  - **Verify**: All methods work, ensemble gives best ECE
  - **Time**: 15 min

- [ ] **Task #160**: Commit Day 7 changes
  - **File**: N/A (git command)
  - **Action**: `git add -A && git commit -m "feat: Phase 5 SCRC calibration (multi-objective ensemble)"`
  - **Verify**: Changes committed
  - **Time**: 1 min

**Day 7 Total**: 20 tasks, ~3 hours

---

## DAY 8: PHASE 6 EXPORT (Tasks #161-175)

- [ ] **Task #161**: Open `src/streetvision/pipeline/steps/export_bundle.py`
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Read current implementation
  - **Verify**: File exists
  - **Time**: 5 min

- [ ] **Task #162**: Update bundle to SCRC-only policy
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Change bundle to point to `scrc_params.pkl` instead of `thresholds.json` (see MASTER_PLAN.md lines 3095-3173)
  - **Verify**: Bundle uses SCRC policy
  - **Time**: 10 min

- [ ] **Task #163**: Verify required artifacts exist
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Add checks for phase1_checkpoint, cvfm_weights, scrc_params_json
  - **Verify**: Phase 6 fails gracefully if artifacts missing
  - **Time**: 5 min

- [ ] **Task #164**: Copy checkpoint to export directory
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Copy model checkpoint to `export/model.pth`
  - **Verify**: Checkpoint copied correctly
  - **Time**: 3 min

- [ ] **Task #165**: Copy CVFM weights if available
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Copy CVFM weights to `export/cvfm_weights.pth` if they exist
  - **Verify**: CVFM weights copied if available
  - **Time**: 3 min

- [ ] **Task #166**: Copy SCRC policy files
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Copy `scrc_params.pkl` and `scrc_params.json` to export directory
  - **Verify**: Policy files copied
  - **Time**: 3 min

- [ ] **Task #167**: Create validator-compatible bundle.json
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Write bundle.json with schema: `{"policy_type": "scrc", "policy": {...}, "model_checkpoint": "...", ...}`
  - **Verify**: Bundle.json matches validator schema
  - **Time**: 10 min

- [ ] **Task #168**: Add bundle compression option
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Add option to create tar.gz of export directory
  - **Verify**: Compression works if enabled
  - **Time**: 5 min

- [ ] **Task #169**: Update ArtifactSchema for Phase 6
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `export_dir`, `bundle_json` properties
  - **Verify**: Properties return correct paths
  - **Time**: 3 min

- [ ] **Task #170**: Add Phase 6 to ensure_dirs()
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `self.export_dir.mkdir(...)` in `ensure_dirs()`
  - **Verify**: Export directory created
  - **Time**: 2 min

- [ ] **Task #171**: Add manifest tracking for Phase 6
  - **File**: `src/streetvision/pipeline/steps/export_bundle.py`
  - **Action**: Add `create_step_manifest()` call at end
  - **Verify**: Manifest.json created
  - **Time**: 3 min

- [ ] **Task #172**: Test Phase 6 export
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/train_cli_v2.py pipeline.phases=[phase6]`
  - **Verify**: Bundle exported successfully
  - **Time**: 5 min

- [ ] **Task #173**: Verify bundle validator compatibility
  - **File**: N/A (test script)
  - **Action**: Load `bundle.json` and validate with BundleValidator
  - **Verify**: Validator accepts bundle
  - **Time**: 5 min

- [ ] **Task #174**: Test loading bundle in inference
  - **File**: N/A (test script)
  - **Action**: Load bundle, load model, load SCRC policy, run inference
  - **Verify**: Inference works with exported bundle
  - **Time**: 10 min

- [ ] **Task #175**: Commit Day 8 changes
  - **File**: N/A (git command)
  - **Action**: `git add -A && git commit -m "feat: Phase 6 export bundle (SCRC-only)"`
  - **Verify**: Changes committed
  - **Time**: 1 min

**Day 8 Total**: 15 tasks, ~2 hours

---

## DAY 9-10: EVALUATION FRAMEWORK (Tasks #176-195)

- [ ] **Task #176**: Create `src/streetvision/pipeline/steps/evaluate_model.py`
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py` (NEW)
  - **Action**: Create evaluation script (see MASTER_PLAN.md lines 3179-3498)
  - **Verify**: File exists
  - **Time**: 2 min

- [ ] **Task #177**: Add bundle loading logic
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Load bundle.json, extract model and policy paths
  - **Verify**: Bundle loaded correctly
  - **Time**: 5 min

- [ ] **Task #178**: Add model loading logic
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Load model checkpoint, apply SCRC calibration
  - **Verify**: Model loads and calibrates correctly
  - **Time**: 10 min

- [ ] **Task #179**: Add VAL_TEST inference loop
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Run inference on VAL_TEST split only (never tune on this)
  - **Verify**: Inference runs on correct split
  - **Time**: 10 min

- [ ] **Task #180**: Add comprehensive metrics computation
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Compute accuracy, precision, recall, F1, MCC, FNR, FPR, ROC-AUC, PR-AUC
  - **Verify**: All metrics computed correctly
  - **Time**: 10 min

- [ ] **Task #181**: Add bootstrap confidence intervals
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Add bootstrap resampling for CI computation (see MASTER_PLAN.md)
  - **Verify**: CIs computed for all metrics
  - **Time**: 15 min

- [ ] **Task #182**: Add ROC curve plotting
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Plot ROC curve and save to evaluation directory
  - **Verify**: ROC curve PNG created
  - **Time**: 5 min

- [ ] **Task #183**: Add PR curve plotting
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Plot precision-recall curve and save
  - **Verify**: PR curve PNG created
  - **Time**: 5 min

- [ ] **Task #184**: Add confusion matrix plotting
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Plot confusion matrix heatmap
  - **Verify**: Confusion matrix PNG created
  - **Time**: 5 min

- [ ] **Task #185**: Add reliability diagram (calibration plot)
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Plot calibration curve (predicted vs actual probability)
  - **Verify**: Reliability diagram PNG created
  - **Time**: 10 min

- [ ] **Task #186**: Add evaluation report JSON
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Save all metrics + CIs to `evaluation_report.json`
  - **Verify**: JSON contains all metrics
  - **Time**: 5 min

- [ ] **Task #187**: Create `configs/evaluation/default.yaml`
  - **File**: `configs/evaluation/default.yaml` (NEW)
  - **Action**: Create config with bootstrap and plotting settings (see MASTER_PLAN.md lines 4072-4107)
  - **Verify**: Config includes all evaluation settings
  - **Time**: 5 min

- [ ] **Task #188**: Create `scripts/evaluate_cli.py`
  - **File**: `scripts/evaluate_cli.py` (NEW)
  - **Action**: Create CLI for running evaluation (see MASTER_PLAN.md lines 4151-4230)
  - **Verify**: CLI accepts bundle path and runs evaluation
  - **Time**: 15 min

- [ ] **Task #189**: Update ArtifactSchema for evaluation
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `evaluation_dir`, `evaluation_report_json` properties
  - **Verify**: Properties return correct paths
  - **Time**: 3 min

- [ ] **Task #190**: Add evaluation to ensure_dirs()
  - **File**: `src/contracts/artifact_schema.py`
  - **Action**: Add `self.evaluation_dir.mkdir(...)` in `ensure_dirs()`
  - **Verify**: Evaluation directory created
  - **Time**: 2 min

- [ ] **Task #191**: Test evaluation framework
  - **File**: N/A (CLI test)
  - **Action**: `python scripts/evaluate_cli.py --bundle outputs/.../export/bundle.json`
  - **Verify**: Evaluation completes, creates all plots and reports
  - **Time**: 10 min

- [ ] **Task #192**: Verify no data leakage in evaluation
  - **File**: `src/streetvision/pipeline/steps/evaluate_model.py`
  - **Action**: Confirm evaluation ONLY uses VAL_TEST, never TRAIN/VAL_SELECT/VAL_CALIB
  - **Verify**: No tuning on VAL_TEST
  - **Time**: 3 min

- [ ] **Task #193**: Add unit tests for evaluation
  - **File**: `tests/unit/test_evaluation.py` (NEW)
  - **Action**: Add tests for metrics computation and plotting
  - **Verify**: Tests pass
  - **Time**: 10 min

- [ ] **Task #194**: Test full pipeline end-to-end (RUNTIME ORDER)
  - **File**: N/A (CLI test)
  - **Action**: Run `python scripts/train_cli_v2.py pipeline.phases=[phase4a_explora,phase1,phase2,phase4c_cvfm,phase5,phase6]` (RUNTIME order)
  - **Note**: Runtime order is phase4a → phase1 → phase2 → phase4c → phase5 → phase6 (NOT implementation order)
  - **Verify**: Full pipeline completes successfully, final MCC >0.90
  - **Time**: 60 min

- [ ] **Task #195**: Commit Day 9-10 changes
  - **File**: N/A (git command)
  - **Action**: `git add -A && git commit -m "feat: Complete evaluation framework"`
  - **Verify**: Changes committed
  - **Time**: 1 min

**Day 9-10 Total**: 20 tasks, ~4 hours

---

## 2025/2026 UPGRADES (Tasks #196-203)

### PyTorch 2.6 Compile Optimizations

- [ ] **Task #196**: Add `torch.compiler.set_stance()` call
  - **File**: `src/streetvision/pipeline/steps/train_baseline.py`
  - **Action**: Add `torch.compiler.set_stance(cfg.hardware.compiler.stance)` before compile (see `docs/allstepsoffupgrade/04_pytorch26_compile.md`)
  - **Verify**: Stance set before compilation
  - **Time**: 3 min

- [ ] **Task #197**: Update compile mode to max-autotune
  - **File**: `configs/training/optimization.yaml`
  - **Action**: Change `compile_mode: "max-autotune"` and add `fullgraph: true, dynamic: false`
  - **Verify**: Config uses max-autotune mode
  - **Time**: 2 min

### DoRA + RSLoRA + PiSSA Init

- [ ] **Task #198**: Add RSLoRA and PiSSA to DoRA config
  - **File**: `configs/phase4b/dora.yaml`
  - **Action**: Add `use_rslora: true` and `init_lora_weights: pissa` (see `docs/allstepsoffupgrade/02_task_peft_dora_rslora_pissa.md`)
  - **Verify**: Config includes RSLoRA and PiSSA settings
  - **Time**: 3 min

- [ ] **Task #199**: Update DoRA implementation to use PiSSA init
  - **File**: `src/peft/dora_task.py`
  - **Action**: Add code to initialize LoRA weights with PiSSA (principal singular values)
  - **Verify**: LoRA weights initialized with PiSSA
  - **Time**: 10 min

### Multi-Objective Calibration Ensemble

- [ ] **Task #200**: Implement MaC-Cal calibration method
  - **File**: `src/streetvision/pipeline/steps/calibrate_scrc.py`
  - **Action**: Add MaC-Cal (multi-objective calibration) method (see `docs/allstepsoffupgrade/03_calibration_sweep_tiers.md`)
  - **Verify**: MaC-Cal improves ECE vs single methods
  - **Time**: 15 min

### TrivialAugmentWide + AugMix (Already in Day 2)

- [ ] **Task #201**: Verify TrivialAugmentWide v2 implemented
  - **File**: `src/data/augmentation.py`
  - **Action**: Confirm TrivialAugmentWide uses `torchvision.transforms.v2` API (already done in Task #61)
  - **Verify**: Uses v2 API, not deprecated v1
  - **Time**: 2 min

### BYOL/SwAV Hybrid (Optional Alternative)

- [ ] **Task #202**: Add BYOL/SwAV hybrid option (optional)
  - **File**: `src/peft/explora_domain.py`
  - **Action**: Add BYOL/SwAV hybrid as alternative to SimCLR (see `docs/allstepsoffupgrade/05_byol_swav_hybrid.md`)
  - **Verify**: Can switch between SimCLR and BYOL/SwAV
  - **Time**: 20 min

### FlexAttention (Optional)

- [ ] **Task #203**: Add FlexAttention CVFM option (optional)
  - **File**: `src/streetvision/tta/flex_cvfm.py`
  - **Action**: Implement FlexAttention-based CVFM (see `docs/allstepsoffupgrade/06_flexattention_cvfm.md`)
  - **Verify**: FlexAttention works for multi-view fusion
  - **Time**: 30 min

**2025/2026 Upgrades Total**: 8 tasks, ~2 hours

---

## ✅ FINAL CHECKLIST

After completing all 203 tasks:

- [ ] **Final Test**: Run complete pipeline end-to-end (RUNTIME ORDER)
  - **Command**: `python scripts/train_cli_v2.py pipeline.phases=[phase4a_explora,phase1,phase2,phase4c_cvfm,phase5,phase6]`
  - **Note**: This is RUNTIME order (phase4a first), not implementation order
  - **Verify**: All phases complete successfully

- [ ] **Final Evaluation**: Run evaluation on VAL_TEST
  - **Command**: `python scripts/evaluate_cli.py --bundle outputs/.../export/bundle.json`
  - **Verify**: Evaluation completes, MCC >0.90

- [ ] **Final Commit**: Commit all changes
  - **Command**: `git add -A && git commit -m "feat: Complete Stage1 Ultimate upgrade (203 tasks)"`
  - **Verify**: All changes committed

- [ ] **Documentation**: Update README with new features
  - **File**: `README.md`
  - **Action**: Document new phases, configs, and features
  - **Verify**: README is up-to-date

---

## 📊 EXPECTED RESULTS

After completing all tasks:

- **MCC**: 0.94-1.03 (+29-38% improvement)
- **Training Speed**: 3× faster (BF16 + compile)
- **Inference Speed**: 2× faster (compile + CVFM)
- **ECE**: <0.02 (multi-objective calibration)
- **Zero Data Leakage**: Strict split enforcement
- **Full Validator Compliance**: All artifacts pass validation

---

## 🎯 HOW TO USE THIS CHECKLIST

1. **Start with Day 0**: Complete all 8 setup tasks
2. **Work Day-by-Day**: Complete each day's tasks in order (IMPLEMENTATION order)
3. **Verify Each Task**: Don't move forward until verification passes
4. **Commit Regularly**: Commit after each day's completion
5. **Test Continuously**: Run tests after each major change
6. **Track Progress**: Update the progress tracking section above
7. **Remember Phase Order**: 
   - ✅ BUILD in implementation order (Phase 2 → Phase 1 → Phase 4a → Phase 4c → Phase 5 → Phase 6)
   - ✅ RUN in runtime order (phase4a → phase1 → phase2 → phase4c → phase5 → phase6)

**Total Estimated Time**: ~35 hours (7-10 days of focused work)

---

## 📝 NOTES FOR IMPLEMENTATION AGENT

**Context**: This is for Bittensor Subnet 72 mining (COMPETITION, not production deployment). Need TOP 10% performance.

**Execution Strategy**:
1. ✅ Start with Days 1-2 (Tasks #1-71): Baseline + optimizations
2. ✅ TEST baseline works before continuing
3. ✅ Then add Days 3-4 (Tasks #72-105): ExPLoRA SimCLR domain adaptation
4. ✅ Then add Days 5-6 (Tasks #106-140): CVFM learned fusion
5. ✅ Then add Days 7-9 (Tasks #141-195): Calibration + export + evaluation

**Phase Order Clarification**:
- **Implementation Order** (what to build): Phase 2 → Phase 1 → Phase 4a → Phase 4c → Phase 5 → Phase 6
- **Runtime Order** (what to run): `phase4a_explora → phase1 → phase2 → phase4c_cvfm → phase5 → phase6`
- **Why different?** Phase 2 is easiest to build (validates schema), but phase4a must run FIRST at training time (domain adaptation before task training)

**Optional Features** (can skip for v1.0):
- Task #202: BYOL/SwAV hybrid (only if SimCLR fails)
- Task #203: FlexAttention CVFM (only if num_views >5)

**Follow checklist exactly, test after each day, commit regularly.**

---

**Status**: ✅ Checklist Complete - Ready for Implementation

