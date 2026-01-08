# 🚀 WEEK 0 EXECUTION PLAN - CRITICAL GAP CLOSURE
**DO THIS FIRST - BLOCKS EVERYTHING!**

**Duration**: 18.5 hours (was 16h - added 2.5h for critical verification)
**Status**: 🔴 BLOCKING - Must complete before Week 1
**Goal**: Close Stage1 gaps to enable 99%+ MCC training

**⚠️ UPDATED**: Added 5 critical verification steps to prevent silent failures!

---

## 📋 PRE-FLIGHT CHECKLIST

### ✅ Verify Environment
```bash
# 1. Check repo structure
cd /home/sina/projects/miner_b/stage1_ultimate
ls -la src/data/samplers/gps_weighted_sampler.py  # Should exist
ls -la src/data/augmentation/heavy_aug_kornia.py  # Should exist
ls -la src/training/callbacks/  # Should be empty
ls -la src/training/lora/  # Should NOT exist yet

# 2. Check Python environment
python3 --version  # Should be 3.11+
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 🆕 ✅ Choose Dataset Mode (CRITICAL - DECIDE NOW!)
**From TRAINING_PLAN_2026_CLEAN.md lines 163-184**

**Mode A: Local images + splits.json** (if you have local data)
```bash
ls /workspace/data/natix_subset/  # Verify images exist
ls outputs/splits.json  # Verify splits exist
```

**Mode B: HuggingFace dataset** (if using remote dataset)
```bash
python3 -c "from datasets import load_dataset; ds = load_dataset('natix-network-org/roadwork', split='train'); print(f'{len(ds)} samples')"
```

**DECISION**: [ ] Mode A or [ ] Mode B?
**⚠️ DO NOT proceed until you choose one!**

### 🆕 ✅ Verify GPS Schema (CRITICAL - #1 SILENT FAILURE!)
**From TRAINING_PLAN_2026_CLEAN.md lines 187-193**

```bash
# CRITICAL: Verify GPS schema BEFORE wiring sampler
python3 << 'EOF'
from datasets import load_dataset

ds = load_dataset("natix-network-org/roadwork", split="train")
sample = ds[0]

print("Available keys:", sample.keys())
print("\nGPS format check:")
print("  latitude:", type(sample.get("latitude")), sample.get("latitude"))
print("  longitude:", type(sample.get("longitude")), sample.get("longitude"))
print("  gps (string):", sample.get("gps"))  # Some datasets have "gps" as "(lat, lon)" string

# Check if GPS needs normalization
if "gps" in sample and isinstance(sample["gps"], str):
    print("\n⚠️  WARNING: GPS is a string! Need to parse before sampling.")
    print(f"  Example: {sample['gps']}")

# Check GPS coverage
has_gps = sum(1 for ex in ds if ex.get("latitude") and ex.get("longitude"))
print(f"\n✅ GPS coverage: {has_gps}/{len(ds)} ({100*has_gps/len(ds):.1f}%)")

if has_gps / len(ds) < 0.8:
    print("⚠️  WARNING: GPS coverage <80%! Training may have location bias.")
EOF
```

**Expected output**:
- ✅ `latitude` and `longitude` are floats (GOOD!)
- ❌ `gps` is a string like "(lat, lon)" (NEEDS PARSING!)

**If GPS is a string → Create normalization function BEFORE Day 0.1!**

### 🆕 ✅ Verify Split Ratios (CRITICAL!)
**From TRAINING_PLAN_2026_CLEAN.md lines 195-199**

```bash
# Check current split ratios
grep -A 5 "splits:" configs/data/natix.yaml

# Expected output:
#   train: 0.60
#   val_select: 0.15
#   val_calib: 0.15
#   val_test: 0.10

# If different → Update to match TRAINING_PLAN spec
```

---

## 🎯 WEEK 0 TASKS

### **Day 0.1: Wire GPS-aware sampling (4.5 hours)** 🔥 CRITICAL

**Problem**: GPS sampler EXISTS but NOT wired into training
**Impact**: Without this, training has location bias (silent failure!)
**Action**: Wire `GPSWeightedSampler` into dataloader config

**⚠️ UPDATED**: Added Step 0 (GPS schema verification) and Step 5.5 (split ratio verification)

#### Step 0: Verify GPS metadata schema (15 min) 🚨 CRITICAL - NEW!
**From TRAINING_PLAN_2026_CLEAN.md lines 187-193**

```bash
# Run GPS schema verification (from Pre-Flight Checklist)
python3 << 'EOF'
from datasets import load_dataset

ds = load_dataset("natix-network-org/roadwork", split="train")
sample = ds[0]

print("Available keys:", sample.keys())
print("\nGPS format check:")
print("  latitude:", type(sample.get("latitude")), sample.get("latitude"))
print("  longitude:", type(sample.get("longitude")), sample.get("longitude"))
print("  gps (string):", sample.get("gps"))

# Check if GPS needs normalization
if "gps" in sample and isinstance(sample["gps"], str):
    print("\n⚠️  WARNING: GPS is a string! Need to parse before sampling.")
    print(f"  Example: {sample['gps']}")
    print("\n🔧 ACTION REQUIRED: Create GPS normalization function!")
    print("   Add to src/data/utils/gps_utils.py:")
    print("   def parse_gps_string(gps_str):")
    print("       # Parse '(lat, lon)' string to floats")
    print("       lat, lon = gps_str.strip('()').split(',')")
    print("       return float(lat), float(lon)")
EOF
```

**If GPS is a string**:
1. Create `src/data/utils/gps_utils.py` with parsing function
2. Update dataset to normalize GPS on load
3. **STOP and verify** before proceeding to Step 1

**If GPS is latitude/longitude floats**: ✅ Proceed to Step 1

#### Step 1: Verify GPS sampler exists (5 min)
```bash
cat src/data/samplers/gps_weighted_sampler.py | head -30
# Should see: class GPSWeightedSampler(Sampler)
```

#### Step 2: Test import (5 min)
```python
python3 << 'EOF'
from src.data.samplers.gps_weighted_sampler import GPSWeightedSampler
print("✅ GPS sampler imports successfully")
EOF
```

#### Step 3: Update dataloader config (2 hours)
**File**: `configs/data/natix.yaml`

**Add GPS sampler config**:
```yaml
# GPS-aware sampling (NATIX-critical - prevents location bias)
gps_sampling:
  enabled: true  # Set to true to enable GPS-weighted sampling
  n_clusters: 5  # Number of K-Means clusters for test GPS
  weight_brackets:  # Distance (km) → weight multiplier
    - [0, 50, 5.0]      # < 50 km: 5× weight
    - [50, 200, 2.5]    # 50-200 km: 2.5× weight
    - [200, 500, 1.0]   # 200-500 km: 1× weight
    - [500, inf, 0.3]   # > 500 km: 0.3× weight
  random_seed: 42
```

#### Step 4: Wire into datamodule (1.5 hours)
**File**: `src/data/datamodule.py`

**Add GPS sampler logic** (pseudocode - adapt to your datamodule):
```python
def setup_train_dataloader(self, cfg):
    dataset = self.train_dataset
    
    # GPS-weighted sampling (if enabled)
    if cfg.data.gps_sampling.enabled:
        from src.data.samplers.gps_weighted_sampler import GPSWeightedSampler
        
        # Get test GPS coordinates
        test_gps = self._get_test_gps_coords()  # Implement this
        
        # Create GPS-weighted sampler
        sampler = GPSWeightedSampler(
            data_source=dataset,
            test_gps=test_gps,
            n_clusters=cfg.data.gps_sampling.n_clusters,
            random_seed=cfg.data.gps_sampling.random_seed
        )
        
        return DataLoader(dataset, batch_size=cfg.data.dataloader.batch_size, sampler=sampler)
    else:
        # Standard random sampling
        return DataLoader(dataset, batch_size=cfg.data.dataloader.batch_size, shuffle=True)
```

#### Step 5: Verify GPS sampling works (30 min)
```bash
# Test GPS sampling
python3 << 'EOF'
from src.data.datamodule import NATIXDataModule
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
cfg.data.gps_sampling.enabled = True

dm = NATIXDataModule(cfg)
dm.setup("fit")

train_loader = dm.train_dataloader()
print(f"✅ GPS-weighted sampler created: {len(train_loader)} batches")
EOF
```

#### Step 5.5: Verify split ratios (10 min) 🆕 CRITICAL!
**From TRAINING_PLAN_2026_CLEAN.md lines 195-199**

```bash
# Check current split ratios
grep -A 5 "splits:" configs/data/natix.yaml

# Expected output (MUST MATCH):
#   train: 0.60
#   val_select: 0.15
#   val_calib: 0.15
#   val_test: 0.10

# If different → Update configs/data/natix.yaml to match spec
```

**Why this matters**: Wrong split ratios will cause MCC calculation errors!

**✅ Success Criteria**:
- [ ] GPS schema verified (latitude/longitude floats OR normalized from string)
- [ ] GPS sampler config added to `configs/data/natix.yaml`
- [ ] GPS sampler wired into `src/data/datamodule.py`
- [ ] Test script runs without errors
- [ ] GPS coverage verified (>80% samples have GPS)
- [ ] Split ratios verified (match TRAINING_PLAN spec)

---

### **Day 0.2: Create latest augmentations (5 hours)** 🎨

**Problem**: Missing TrivialAugment + CutMix + MixUp (CVPR 2025 SOTA)
**Impact**: +3-5% detection accuracy
**Action**: Create `src/data/augmentation/latest_aug_2025.py`

**⚠️ UPDATED**: Added Step 5 (wire into datamodule) - CRITICAL!

#### Step 1: Create latest augmentation file (2 hours)
**File**: `src/data/augmentation/latest_aug_2025.py`

See TRAINING_PLAN_2026_CLEAN.md lines 318-323 for full implementation.

#### Step 2: Verify syntax (5 min)
```bash
python3 -m py_compile src/data/augmentation/latest_aug_2025.py
echo "✅ Syntax OK"
```

#### Step 3: Test import (5 min)
```python
python3 << 'EOF'
from src.data.augmentation.latest_aug_2025 import LatestAugmentation2025
aug = LatestAugmentation2025(img_size=640)
print("✅ Latest augmentation imports successfully")
EOF
```

#### Step 4: Make selectable from config (1.5 hours)
**File**: `configs/data/natix.yaml`

**Add augmentation selector**:
```yaml
augmentation:
  train:
    mode: latest_2025  # Options: 'heavy_kornia', 'latest_2025'
    # ... existing config ...
```

#### Step 5: Wire into datamodule (1 hour) 🆕 CRITICAL!
**File**: `src/data/datamodule.py`

**Add augmentation selector logic**:
```python
def _get_train_transforms(self, cfg):
    """Get training augmentations based on config"""

    mode = cfg.data.augmentation.train.mode

    if mode == "latest_2025":
        from src.data.augmentation.latest_aug_2025 import LatestAugmentation2025
        return LatestAugmentation2025(img_size=cfg.data.img_size)

    elif mode == "heavy_kornia":
        from src.data.augmentation.heavy_aug_kornia import HeavyAugmentationKornia
        return HeavyAugmentationKornia(img_size=cfg.data.img_size)

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")
```

**Verify integration**:
```bash
python3 << 'EOF'
from src.data.datamodule import NATIXDataModule
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
cfg.data.augmentation.train.mode = "latest_2025"

dm = NATIXDataModule(cfg)
dm.setup("fit")
print("✅ Latest augmentation integrated!")
EOF
```

**✅ Success Criteria**:
- [ ] `src/data/augmentation/latest_aug_2025.py` created
- [ ] Syntax check passes
- [ ] Import test passes
- [ ] Config selector added
- [ ] Augmentation wired into datamodule
- [ ] Integration test passes

---

### **Day 0.3: Implement callbacks (5 hours)** 📊

**Problem**: `src/training/callbacks/` is EMPTY
**Impact**: Can't track MCC during training, no EMA for stability
**Action**: Create MCC callback + EMA callback

**⚠️ UPDATED**: Added Step 3 (register callbacks in trainer) - CRITICAL!

#### Step 1: Create MCC callback (1.5 hours)
**File**: `src/training/callbacks/mcc_callback.py`

```python
"""MCC Callback for Roadwork Detection"""
import logging
from sklearn.metrics import matthews_corrcoef
import torch

logger = logging.getLogger(__name__)


class MCCCallback:
    """Track Matthews Correlation Coefficient during training"""

    def __init__(self):
        self.best_mcc = -1.0
        self.y_true_buffer = []
        self.y_pred_buffer = []
        logger.info("✅ MCC callback initialized")

    def on_batch_end(self, outputs, labels):
        """Accumulate predictions during epoch"""
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        self.y_true_buffer.extend(labels)
        self.y_pred_buffer.extend(preds)

    def on_epoch_end(self, epoch):
        """Compute MCC at end of epoch"""
        if len(self.y_true_buffer) == 0:
            logger.warning("No predictions to compute MCC")
            return {}

        mcc = matthews_corrcoef(self.y_true_buffer, self.y_pred_buffer)
        logger.info(f"Epoch {epoch}: MCC = {mcc:.4f}")

        if mcc > self.best_mcc:
            self.best_mcc = mcc
            logger.info(f"🎉 New best MCC: {mcc:.4f}")

        # Clear buffers
        self.y_true_buffer = []
        self.y_pred_buffer = []

        return {'mcc': mcc, 'best_mcc': self.best_mcc}
```

#### Step 2: Create EMA callback (1.5 hours)
**File**: `src/training/callbacks/ema_callback.py`

```python
"""EMA Callback for Model Stability"""
import torch
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class EMACallback:
    """Exponential Moving Average for model weights"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
        logger.info(f"✅ EMA callback initialized (decay={decay})")

    def _register(self):
        """Register EMA shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights after each training step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights (for validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
```

#### Step 3: Register callbacks in trainer (1 hour) 🆕 CRITICAL!
**File**: `src/training/trainers/base_trainer.py` (or wherever your train loop is)

**Add callback registration**:
```python
class BaseTrainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

        # Initialize callbacks
        self.callbacks = []

        # MCC callback (always enabled)
        from src.training.callbacks.mcc_callback import MCCCallback
        self.mcc_callback = MCCCallback()
        self.callbacks.append(self.mcc_callback)

        # EMA callback (if enabled)
        if cfg.training.use_ema:
            from src.training.callbacks.ema_callback import EMACallback
            self.ema_callback = EMACallback(model, decay=cfg.training.ema_decay)
            self.callbacks.append(self.ema_callback)

    def train_epoch(self, epoch):
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update EMA (if enabled)
            if hasattr(self, 'ema_callback'):
                self.ema_callback.update()

            # Accumulate predictions for MCC
            self.mcc_callback.on_batch_end(outputs, labels)

        # Compute MCC at epoch end
        metrics = self.mcc_callback.on_epoch_end(epoch)
        logger.info(f"Epoch {epoch} metrics: {metrics}")
```

**Add to config** (`configs/training/default.yaml`):
```yaml
training:
  use_ema: true
  ema_decay: 0.999
```

**Verify registration**:
```bash
# Test that callbacks are called during training
python3 scripts/train.py --config configs/config.yaml --dry-run
# Should see: "✅ MCC callback initialized" and "✅ EMA callback initialized"
```

#### Step 4: Verify syntax (5 min)
```bash
python3 -m py_compile src/training/callbacks/mcc_callback.py
python3 -m py_compile src/training/callbacks/ema_callback.py
echo "✅ Syntax OK"
```

#### Step 5: Test import (5 min)
```bash
# Test MCC callback
python3 << 'EOF'
from src.training.callbacks.mcc_callback import MCCCallback
callback = MCCCallback()
print("✅ MCC callback imports successfully")
EOF

# Test EMA callback
python3 << 'EOF'
import torch
from src.training.callbacks.ema_callback import EMACallback
model = torch.nn.Linear(10, 2)
callback = EMACallback(model, decay=0.999)
print("✅ EMA callback imports successfully")
EOF
```

**✅ Success Criteria**:
- [ ] `src/training/callbacks/mcc_callback.py` created
- [ ] `src/training/callbacks/ema_callback.py` created
- [ ] Both callbacks pass syntax check
- [ ] Both callbacks pass import test
- [ ] Callbacks registered in trainer (or documented for later)
- [ ] Config updated with `use_ema` and `ema_decay`
- [ ] Dry-run test shows callbacks initialized

---

### **Day 0.4: Create PEFT config stubs (4 hours)** 🧩

**Problem**: No PEFT configs for Week 1.5 (AdaLoRA, VeRA, IA³)
**Impact**: Blocks Week 1.5 advanced PEFT techniques
**Action**: Create `src/training/lora/` with config stubs

#### Step 1: Create lora directory (5 min)
```bash
mkdir -p src/training/lora
touch src/training/lora/__init__.py
```

#### Step 2-4: Create PEFT configs (3 hours)
See TRAINING_PLAN_2026_CLEAN.md for full implementations:
- AdaLoRA config (lines 750-838)
- VeRA config (lines 847-933)
- IA³ config (lines 942-1027)

**✅ Success Criteria**:
- [ ] `src/training/lora/` directory created
- [ ] `adalora_config.py` created
- [ ] `vera_config.py` created
- [ ] `ia3_config.py` created
- [ ] All configs pass syntax check

---

## 🎯 WEEK 0 COMPLETION CHECKLIST

### ✅ All Tasks Complete
- [ ] Day 0.1: GPS-aware sampling wired (4.5h)
- [ ] Day 0.2: Latest augmentations created + integrated (5h)
- [ ] Day 0.3: MCC + EMA callbacks created + registered (5h)
- [ ] Day 0.4: PEFT config stubs created (4h)

### 🆕 ✅ Critical Verifications (NEW!)
- [ ] GPS schema verified (latitude/longitude floats OR normalized)
- [ ] Dataset mode chosen (Mode A or Mode B)
- [ ] Split ratios verified (train=0.60, val_select=0.15, val_calib=0.15, val_test=0.10)
- [ ] GPS coverage >80%
- [ ] Augmentation integrated into datamodule
- [ ] Callbacks registered in trainer

### ✅ Final Verification
```bash
# Test all imports
python3 << 'EOF'
from src.data.samplers.gps_weighted_sampler import GPSWeightedSampler
from src.data.augmentation.latest_aug_2025 import LatestAugmentation2025
from src.training.callbacks.mcc_callback import MCCCallback
from src.training.callbacks.ema_callback import EMACallback
from src.training.lora.adalora_config import create_adalora_config
from src.training.lora.vera_config import create_vera_config
from src.training.lora.ia3_config import create_ia3_config
print("✅ All Week 0 modules import successfully!")
EOF

# Test GPS sampling integration
python3 << 'EOF'
from src.data.datamodule import NATIXDataModule
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
cfg.data.gps_sampling.enabled = True
cfg.data.augmentation.train.mode = "latest_2025"

dm = NATIXDataModule(cfg)
dm.setup("fit")
print("✅ GPS sampling + latest augmentation integrated!")
EOF

# Test callbacks (dry-run)
python3 scripts/train.py --config configs/config.yaml --dry-run
# Should see: "✅ MCC callback initialized" and "✅ EMA callback initialized"
```

**⚠️ DO NOT proceed to Week 1 until ALL checkboxes are ✅!**

---

## 🆕 UPDATED TIMELINE

**Original Week 0**: 16 hours
**Updated Week 0**: **18.5 hours** (added 2.5 hours for verification)

**Breakdown**:
- Day 0.1: 4.5 hours (was 4h) - Added GPS schema + split ratio verification
- Day 0.2: 5 hours (was 4h) - Added augmentation integration
- Day 0.3: 5 hours (was 4h) - Added callback registration
- Day 0.4: 4 hours (unchanged)

---

## 🚀 NEXT STEPS

**After Week 0**: Proceed to Week 1 (Core Training Infrastructure)

**Expected Impact**: +10-15% MCC from Week 0 alone!

---

## 🚨 CRITICAL FIXES APPLIED

This plan was updated based on feedback from another agent who identified **5 critical gaps**:

1. ✅ **GPS Schema Verification** (Day 0.1, Step 0) - Prevents string vs float mismatch
2. ✅ **Dataset Mode Selection** (Pre-Flight) - Mode A vs Mode B decision
3. ✅ **Split Ratio Verification** (Day 0.1, Step 5.5) - Ensures correct MCC calculation
4. ✅ **Augmentation Integration** (Day 0.2, Step 5) - Actually uses the new augmentation
5. ✅ **Callback Registration** (Day 0.3, Step 3) - Wires callbacks into training loop

**Without these fixes**: Silent failures, augmentation not used, callbacks never called!
**With these fixes**: Production-ready Week 0 ✅


