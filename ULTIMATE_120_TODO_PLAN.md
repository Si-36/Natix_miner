# ULTIMATE STAGE 1 COMPLETE IMPLEMENTATION PLAN
## Production-Grade Multi-View DINOv3 Classifier - Dec 28, 2025

**Status**: NEW clean implementation using PyTorch Lightning template
**Target**: 90%+ validation accuracy, ECE <0.05, 6× faster inference
**Estimated Time**: ~40 hours (~4 days)
**Implementation**: PyTorch Lightning + Hydra + Multi-view BUILT-IN + Complete Calibration

---

## EXECUTIVE SUMMARY - COMPLETE STAGE 1

**NEW CLEAN IMPLEMENTATION** in `stage1_ultimate/` using official PyTorch Lightning template:

### Core Features (BUILT-IN from Day 1)

1. ✅ **Multi-View Inference** (1 global + 3×3 tiles)
   - Batched forward pass (all 10 crops in one call - 5-10× faster)
   - Top-K mean aggregation (K=2 or 3) - more robust than max
   - 10-15% tile overlap for boundary cases
   - Expected: +3-8% accuracy over single-view

2. ✅ **ExPLoRA PEFT** (+8.2% - BIGGEST GAIN)
   - Continue SSL pretraining on roadwork domain
   - 8-10× cheaper than full pretraining
   - Only 0.1-10% trainable parameters
   - Expected: +8.2% accuracy (arXiv 2406.10973)

3. ✅ **SOTA 2025 Features**
   - DoRAN head (+1-3% over LoRA) - Dec 2024
   - Flash Attention 3 (1.5-2× faster on H100)
   - Safe hyperparams (dropout 0.3, WD 0.01, LS 0.1 - NOT 0.45/0.05/0.15)
   - torch.compile for additional speedup

4. ✅ **Production Architecture**
   - PyTorch Lightning + Hydra configuration
   - DAG pipeline (artifact registry, split contracts, validators)
   - Leakage prevention enforced as code
   - Type-safe configs with Pydantic

5. ✅ **Complete Calibration & Metrics**
   - Temperature scaling, Dirichlet, SCRC
   - AUROC, AUPRC, ECE, AUGRC, bootstrap CI
   - Slice-based evaluation (day/night/weather)
   - Conformal prediction (APS, RAPS)

6. ✅ **Multi-Dataset Fusion**
   - NATIX + Mapillary Vistas (50/50 or 30/70)
   - Hard negative mining (orange objects)
   - Class balancing strategies

**Expected Results**:
- **Accuracy**: 69% → 90% (+21 percentage points)
- **Speed**: 6× faster inference (batched multi-view + TensorRT)
- **Cost**: 10× cheaper training ($120 → $12 with ExPLoRA)
- **Calibration**: ECE 0.29 → 0.05 (-83% error)
- **Quality**: Production-ready with zero data leakage

---

## SETUP: PyTorch Lightning Template

### Step 1: Download Official Lightning Template

```bash
# Option 1: Using GitHub CLI (recommended)
gh repo clone Lightning-AI/lightning-template stage1_ultimate
cd stage1_ultimate

# Option 2: Using git
git clone https://github.com/Lightning-AI/lightning-template.git stage1_ultimate
cd stage1_ultimate

# Option 3: Create from Lightning CLI
pip install lightning
lightning init stage1_ultimate
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install lightning hydra-core transformers peft
pip install timm scikit-learn pandas numpy
pip install tensorboard wandb

# Calibration & metrics
pip install netcal # Temperature scaling, ECE
pip install uncertainty-toolbox # Calibration metrics

# Optional: Flash Attention 3 (requires PyTorch 2.5+)
pip install torch==2.5.0 --upgrade

# Optional: TensorRT export
pip install tensorrt onnx onnxruntime
```

### Lightning Template Structure (We'll Adapt This)

```
stage1_ultimate/
├── configs/                    # Hydra configs (Lightning standard)
│   ├── model/
│   ├── data/
│   ├── trainer/
│   └── config.yaml             # Main config
│
├── src/
│   ├── models/                 # LightningModule
│   │   └── module.py
│   ├── data/                   # LightningDataModule
│   │   └── datamodule.py
│   └── utils/
│
├── train.py                    # Main entry point
└── requirements.txt
```

---

## COMPLETE FOLDER STRUCTURE (Final)

```
stage1_ultimate/
├── configs/                    # Hydra configs (all hyperparams)
│   ├── model/
│   │   ├── dinov2_multiview.yaml       # Multi-view config
│   │   ├── explora.yaml                # ExPLoRA config
│   │   └── doran.yaml                  # DoRAN config
│   ├── training/
│   │   ├── baseline.yaml               # Phase 1 config
│   │   ├── explora_pretrain.yaml       # Phase 1b config
│   │   └── optimizer_ablation.yaml     # SAM2, Sophia, Muon, etc.
│   ├── data/
│   │   ├── natix.yaml                  # NATIX dataset
│   │   ├── mapillary.yaml              # Mapillary Vistas
│   │   └── fusion.yaml                 # Multi-dataset fusion
│   ├── calibration/
│   │   ├── temperature.yaml
│   │   ├── dirichlet.yaml
│   │   ├── scrc.yaml                   # Split Conformal Risk Control
│   │   └── conformal.yaml              # APS, RAPS
│   ├── evaluation/
│   │   ├── metrics.yaml                # AUROC, AUPRC, ECE, etc.
│   │   ├── slicing.yaml                # Day/night/weather slices
│   │   └── bootstrap.yaml              # Confidence intervals
│   └── config.yaml                     # Main config
│
├── src/
│   ├── contracts/              # Leakage prevention (code-enforced)
│   │   ├── artifact_schema.py  # ⭐ CRITICAL: All file paths
│   │   ├── split_contracts.py  # Split usage rules
│   │   └── validators.py       # Fail-fast checking
│   │
│   ├── pipeline/               # DAG orchestrator
│   │   ├── phase_spec.py       # Phase contracts
│   │   └── dag_engine.py       # ⭐ CRITICAL: Dependency resolution
│   │
│   ├── models/                 # All model components
│   │   ├── module.py           # ⭐ CRITICAL: LightningModule (main)
│   │   ├── multi_view.py       # ⭐ CRITICAL: Multi-view inference
│   │   ├── explora.py          # ⭐ CRITICAL: ExPLoRA PEFT (+8.2%)
│   │   ├── doran.py            # DoRAN head (+1-3%)
│   │   ├── flash_attention.py  # Flash Attention 3 (1.5-2× speed)
│   │   ├── backbone.py         # DINOv3 wrapper
│   │   └── head.py             # Classification head
│   │
│   ├── data/                   # Data loading
│   │   ├── datamodule.py       # ⭐ CRITICAL: LightningDataModule
│   │   ├── datasets.py         # NATIX, Mapillary
│   │   ├── splits.py           # 4-way split generation
│   │   ├── transforms.py       # Augmentation
│   │   └── multi_view_loader.py # Multi-crop dataloader
│   │
│   ├── training/               # Training logic
│   │   ├── ema.py              # EMA implementation
│   │   └── explora_pretrain.py # ExPLoRA pretraining
│   │
│   ├── metrics/                # Evaluation
│   │   ├── calibration.py      # ECE, MCE, NLL, Brier, ACE
│   │   ├── selective.py        # Risk-coverage, AUGRC
│   │   ├── bootstrap.py        # Confidence intervals
│   │   ├── classification.py   # AUROC, AUPRC, F1
│   │   └── slicing.py          # Slice-based evaluation
│   │
│   ├── calibration/            # Post-hoc calibration
│   │   ├── temperature.py      # Temperature scaling
│   │   ├── dirichlet.py        # Dirichlet calibration
│   │   ├── scrc.py             # Split Conformal Risk Control
│   │   └── conformal.py        # APS, RAPS, CRCP
│   │
│   └── utils/
│       ├── monitoring.py       # Drift detection (PSI, KS test)
│       └── logging.py
│
├── scripts/                    # CLI entry points
│   ├── 10_train_baseline.py    # Phase 1: Baseline
│   ├── 15_train_explora.py     # Phase 1b: ExPLoRA
│   ├── 20_calibrate.py         # Phase 2: Calibration
│   ├── 30_evaluate.py          # Phase 3: Evaluation
│   └── 40_export.py            # Phase 4: TensorRT export
│
├── train.py                    # Main entry point (Lightning CLI)
├── requirements.txt
└── README.md
```

---

## IMPLEMENTATION PHASES (4 Days, ~40h)

### Phase 1: Baseline Multi-View (Day 1-2, ~16h)

**Goal**: Train frozen DINOv3 + multi-view inference with safe hyperparams

#### Tasks:
1. ✅ Setup Lightning template structure
2. ✅ Implement contracts (artifact_schema.py, split_contracts.py, validators.py)
3. ✅ Implement multi-view inference:
   - MultiViewGenerator (1 global + 3×3 tiles, 10-15% overlap)
   - TopKMeanAggregator (K=2 or 3)
   - Batched forward pass (CRITICAL for speed)
4. ✅ Implement LightningModule with:
   - Multi-view forward pass
   - EMA (decay=0.9999)
   - Early stopping on val_select
   - Save logits on val_calib
   - Safe hyperparams (dropout=0.3, WD=0.01, LS=0.1)
5. ✅ Implement LightningDataModule
6. ✅ Train and validate

**Expected**: 72-77% accuracy (+3-8% from multi-view)

**Critical Files**:
- `src/models/module.py` - LightningModule (main training logic)
- `src/models/multi_view.py` - Multi-view inference
- `src/data/datamodule.py` - LightningDataModule
- `src/contracts/artifact_schema.py` - File paths

---

### Phase 1b: ExPLoRA (Day 2-3, ~8h) **BIGGEST GAIN**

**Goal**: Continue SSL pretraining on roadwork domain (+8.2% accuracy)

#### Tasks:
1. ✅ Implement ExPLoRA wrapper:
   - Freeze backbone
   - Add LoRA adapters (r=8, alpha=16)
   - Unfreeze last N blocks (N=2)
2. ✅ Continue DINOv2 SSL objective on unlabeled roadwork images
3. ✅ Merge LoRA weights to checkpoint
4. ✅ Fine-tune on labeled data

**Expected**: 77% → 85% accuracy (+8.2% BIGGEST GAIN)
**Cost**: 8-10× cheaper than full pretraining
**Reference**: arXiv 2406.10973

**Critical Files**:
- `src/models/explora.py` - ExPLoRA PEFT implementation
- `src/training/explora_pretrain.py` - SSL pretraining loop
- `scripts/15_train_explora.py` - Entry point

---

### Phase 2: SOTA 2025 Features (Day 3, ~8h)

**Goal**: Add DoRAN + Flash Attention 3 + torch.compile

#### Tasks:
1. ✅ Implement DoRAN head (weight-decomposed LoRA + noise injection)
2. ✅ Integrate Flash Attention 3 (PyTorch 2.5+ SDPA)
3. ✅ Add torch.compile for additional 1.3-2× speedup
4. ✅ Train with DoRAN head
5. ✅ Benchmark speed improvements

**Expected**: 85% → 88% accuracy (+3%), 1.5-2× faster inference
**Reference**: DoRAN (Dec 2024), Flash Attention 3 (PyTorch Blog)

**Critical Files**:
- `src/models/doran.py` - DoRAN implementation
- `src/models/flash_attention.py` - Flash Attention 3 wrapper

---

### Phase 3: Multi-Dataset + Complete Calibration (Day 4, ~8h)

**Goal**: Production-ready system with complete calibration & metrics

#### Tasks:

**Multi-Dataset Fusion:**
1. ✅ Integrate Mapillary Vistas dataset
2. ✅ Implement fusion strategies (50/50 or 30/70 NATIX/Mapillary)
3. ✅ Hard negative mining (orange objects, construction signs)
4. ✅ Class balancing with WeightedRandomSampler

**Complete Calibration:**
5. ✅ Temperature scaling (global + per-class)
6. ✅ Dirichlet calibration
7. ✅ SCRC (Split Conformal Risk Control)
8. ✅ Conformal prediction (APS, RAPS)

**Complete Metrics:**
9. ✅ Classification metrics (AUROC, AUPRC, F1)
10. ✅ Calibration metrics (ECE, MCE, ACE, Brier, NLL)
11. ✅ Selective prediction (AUGRC, risk-coverage curves)
12. ✅ Bootstrap confidence intervals (95% CI)
13. ✅ Slice-based evaluation (day/night/weather)

**Export:**
14. ✅ TensorRT export for 3-5× inference speedup
15. ✅ ONNX export for compatibility

**Expected**: 88% → 90% accuracy, ECE 0.05-0.10

**Critical Files**:
- `src/data/datasets.py` - Mapillary integration
- `src/calibration/scrc.py` - SCRC calibration
- `src/metrics/calibration.py` - All calibration metrics
- `src/metrics/slicing.py` - Slice-based evaluation
- `scripts/40_export.py` - TensorRT/ONNX export

---

## CRITICAL IMPLEMENTATION DETAILS

### 1. Multi-View Implementation (MOST CRITICAL)

**Multi-view MUST be built into forward pass from day 1 (not bolted on later):**

```python
# src/models/multi_view.py
class MultiViewDINOv3(nn.Module):
    """
    Multi-view inference with batched forward pass

    Expected gain: +3-8% accuracy
    Speed: 1.5× slower than single-view (but 5-10× faster than sequential)
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        num_crops: int = 10,  # 1 global + 9 tiles
        tile_size: int = 224,
        overlap: float = 0.15,  # 15% overlap (CRITICAL for boundaries)
        aggregation: str = "topk_mean",  # topk_mean | attention | max
        topk: int = 2,  # K for top-K mean
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.generator = MultiViewGenerator(tile_size, overlap)

        if aggregation == "topk_mean":
            self.aggregator = TopKMeanAggregator(K=topk)
        elif aggregation == "attention":
            self.aggregator = AttentionAggregator(hidden_size=768, num_views=num_crops)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] batch of images

        Returns:
            final_probs: [B, num_classes] aggregated predictions
            debug_info: dict with intermediate outputs
        """
        B = images.size(0)

        # Step 1: Generate 10 crops per image
        all_crops = []
        for i in range(B):
            crops = self.generator.generate_views(images[i])  # [10, 3, 224, 224]
            all_crops.append(crops)
        all_crops = torch.stack(all_crops)  # [B, 10, 3, 224, 224]

        # Step 2: Flatten for batched forward pass (CRITICAL FOR SPEED!)
        crops_flat = all_crops.view(B * 10, 3, 224, 224)  # [B*10, 3, 224, 224]

        # Step 3: Single batched forward pass (5-10× faster than looping!)
        with torch.no_grad() if not self.backbone.training else torch.enable_grad():
            features = self.backbone(crops_flat)  # [B*10, 768]

        logits = self.head(features)  # [B*10, 2]

        # Step 4: Reshape to [B, 10, 2]
        logits = logits.view(B, 10, 2)
        probs = torch.softmax(logits, dim=-1)  # [B, 10, 2]

        # Step 5: Aggregate
        if isinstance(self.aggregator, TopKMeanAggregator):
            # Top-K mean on positive class probabilities
            pos_probs = probs[:, :, 1]  # [B, 10]
            final_prob = self.aggregator(pos_probs)  # [B]
            final_probs = torch.stack([1 - final_prob, final_prob], dim=-1)  # [B, 2]
            attention_weights = None
        else:
            # Attention aggregation
            final_probs, attention_weights = self.aggregator(probs)

        debug_info = {
            'view_probs': probs,  # [B, 10, 2]
            'attention_weights': attention_weights,  # [B, 10] or None
            'crops': all_crops,  # [B, 10, 3, 224, 224]
        }

        return final_probs, debug_info


class MultiViewGenerator:
    """Generate 1 global + 3×3 tiles with overlap"""

    def __init__(self, tile_size: int = 224, overlap: float = 0.15):
        self.tile_size = tile_size
        self.overlap = overlap

    def generate_views(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generate 10 crops: 1 global + 9 tiles

        Args:
            image: [C, H, W] single image

        Returns:
            crops: [10, C, 224, 224] tensor
        """
        C, H, W = image.shape
        crops = []

        # 1. Global view (resize full image)
        global_crop = F.interpolate(
            image.unsqueeze(0),
            size=(self.tile_size, self.tile_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        crops.append(global_crop)

        # 2. 3×3 tiles with 15% overlap
        step_size = int(self.tile_size * (1 - self.overlap))  # 224 * 0.85 = 190

        for i in range(3):
            for j in range(3):
                # Calculate crop position
                y_start = i * step_size
                x_start = j * step_size
                y_end = min(y_start + self.tile_size, H)
                x_end = min(x_start + self.tile_size, W)

                # Extract and resize crop
                crop = image[:, y_start:y_end, x_start:x_end]
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(self.tile_size, self.tile_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                crops.append(crop)

        return torch.stack(crops)  # [10, C, 224, 224]


class TopKMeanAggregator(nn.Module):
    """
    Top-K mean aggregation (simpler and effective)

    Takes top-K most confident predictions and averages them.
    More robust than pure max, simpler than attention.

    Recommended: K=2 or K=3
    """

    def __init__(self, K: int = 2):
        super().__init__()
        self.K = K

    def forward(self, view_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_probs: [B, N_views] probabilities for positive class

        Returns:
            aggregated_prob: [B] final probabilities
        """
        # Get top-K probabilities
        topk_probs, _ = torch.topk(view_probs, self.K, dim=1)

        # Average
        aggregated_prob = topk_probs.mean(dim=1)

        return aggregated_prob


class AttentionAggregator(nn.Module):
    """
    Attention-based aggregation (learns which views to trust)

    More powerful than top-K mean but requires more data.
    """

    def __init__(self, hidden_size: int = 768, num_views: int = 10):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, view_features: torch.Tensor):
        """
        Args:
            view_features: [B, N_views, hidden_size] features from each view

        Returns:
            aggregated: [B, hidden_size] aggregated features
            attention_weights: [B, N_views] attention weights
        """
        # Compute attention scores
        attention_logits = self.attention(view_features)  # [B, N_views, 1]
        attention_weights = F.softmax(attention_logits, dim=1)  # [B, N_views, 1]

        # Weighted sum
        aggregated = (view_features * attention_weights).sum(dim=1)  # [B, hidden_size]

        return aggregated, attention_weights.squeeze(-1)
```

**Why this matters**: Batched forward pass is 5-10× faster than looping over crops.

---

### 2. Lightning Module (Main Training Logic)

```python
# src/models/module.py
import lightning as L
import torch
import torch.nn as nn
from typing import Any, Dict
from src.models.multi_view import MultiViewDINOv3
from src.models.backbone import DINOv3Backbone
from src.models.head import ClassificationHead
from src.training.ema import EMA

class Stage1LightningModule(L.LightningModule):
    """
    Main LightningModule for Stage 1 training

    Features:
    - Multi-view inference (built-in)
    - EMA (exponential moving average)
    - Safe hyperparameters
    - Split-based logging (val_select vs val_calib)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        # Multi-view
        use_multiview: bool = True,
        num_crops: int = 10,
        tile_overlap: float = 0.15,
        aggregation: str = "topk_mean",
        topk: int = 2,
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.01,  # SAFE (not 0.05)
        # Regularization
        dropout: float = 0.3,  # SAFE (not 0.45)
        label_smoothing: float = 0.1,  # SAFE (not 0.15)
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        # Training
        warmup_epochs: int = 1,
        max_epochs: int = 15,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model
        backbone = DINOv3Backbone(model_name, freeze=freeze_backbone)
        head = ClassificationHead(
            hidden_size=768,  # DINOv3-base
            num_classes=num_classes,
            dropout=dropout,
        )

        if use_multiview:
            self.model = MultiViewDINOv3(
                backbone=backbone,
                head=head,
                num_crops=num_crops,
                overlap=tile_overlap,
                aggregation=aggregation,
                topk=topk,
            )
        else:
            self.model = nn.Sequential(backbone, head)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # EMA
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = None

        # Logits storage (for calibration)
        self.val_calib_logits = []
        self.val_calib_labels = []

    def forward(self, x):
        if isinstance(self.model, MultiViewDINOv3):
            probs, debug_info = self.model(x)
            return probs
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # Log
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA
        if self.ema:
            self.ema.update()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch

        # Use EMA model if available
        if self.ema:
            with self.ema.average_parameters():
                logits = self(images)
        else:
            logits = self(images)

        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        # Split-based logging
        if dataloader_idx == 0:
            # val_select (for early stopping)
            prefix = "val_select"
        elif dataloader_idx == 1:
            # val_calib (save logits for calibration)
            prefix = "val_calib"
            self.val_calib_logits.append(logits.detach().cpu())
            self.val_calib_labels.append(labels.detach().cpu())
        else:
            # val_test (final evaluation only)
            prefix = "val_test"

        self.log(f"{prefix}/loss", loss, add_dataloader_idx=False)
        self.log(f"{prefix}/acc", acc, add_dataloader_idx=False)

        return {f"{prefix}_loss": loss, f"{prefix}_acc": acc}

    def on_validation_epoch_end(self):
        # Save val_calib logits to disk
        if len(self.val_calib_logits) > 0:
            logits = torch.cat(self.val_calib_logits, dim=0)
            labels = torch.cat(self.val_calib_labels, dim=0)

            # Save for calibration
            torch.save(logits, "outputs/val_calib_logits.pt")
            torch.save(labels, "outputs/val_calib_labels.pt")

            # Clear for next epoch
            self.val_calib_logits = []
            self.val_calib_labels = []

    def configure_optimizers(self):
        # AdamW with safe hyperparameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine annealing with warmup
        warmup_steps = self.hparams.warmup_epochs * self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
```

---

### 3. Split Contracts (PREVENTS DATA LEAKAGE)

**CRITICAL**: Enforce split usage as code, not developer discipline:

```python
# src/contracts/split_contracts.py
from enum import Enum
from typing import Set

class Split(Enum):
    TRAIN = "train"
    VAL_SELECT = "val_select"  # For model selection ONLY (early stopping)
    VAL_CALIB = "val_calib"    # For policy fitting ONLY (threshold sweep, calibration)
    VAL_TEST = "val_test"      # For final evaluation ONLY (NEVER touch during training)

class SplitPolicy:
    """
    Enforces split usage rules - prevents leakage by construction

    CRITICAL CONTRACTS:
    1. Model selection (early stopping, checkpoint selection) → ONLY val_select
    2. Policy fitting (threshold sweep, gate calibration, SCRC) → ONLY val_calib
    3. Final evaluation → ONLY val_test

    These rules are ENFORCED AS CODE - system cannot run if violated.
    """

    MODEL_SELECTION_SPLITS = {Split.VAL_SELECT}
    POLICY_FITTING_SPLITS = {Split.VAL_CALIB}
    FINAL_EVAL_SPLITS = {Split.VAL_TEST}

    @staticmethod
    def validate_model_selection(splits_used: Set[Split]) -> bool:
        """CRITICAL: Model selection must NEVER use val_calib or val_test"""
        forbidden = SplitPolicy.POLICY_FITTING_SPLITS | SplitPolicy.FINAL_EVAL_SPLITS
        if splits_used & forbidden:
            raise ValueError(
                f"❌ LEAKAGE VIOLATION: Model selection used {splits_used & forbidden}. "
                f"ONLY {SplitPolicy.MODEL_SELECTION_SPLITS} allowed."
            )
        return True

    @staticmethod
    def validate_policy_fitting(splits_used: Set[Split]) -> bool:
        """CRITICAL: Policy fitting must ONLY use val_calib"""
        if splits_used != SplitPolicy.POLICY_FITTING_SPLITS:
            raise ValueError(
                f"❌ LEAKAGE VIOLATION: Policy fitting used {splits_used}. "
                f"ONLY {SplitPolicy.POLICY_FITTING_SPLITS} allowed."
            )
        return True

    @staticmethod
    def validate_final_eval(splits_used: Set[Split]) -> bool:
        """CRITICAL: Final evaluation must ONLY use val_test"""
        if splits_used != SplitPolicy.FINAL_EVAL_SPLITS:
            raise ValueError(
                f"❌ LEAKAGE VIOLATION: Final evaluation used {splits_used}. "
                f"ONLY {SplitPolicy.FINAL_EVAL_SPLITS} allowed."
            )
        return True
```

---

### 4. Safe Hyperparameters (FROM BESTOFFALL.MD)

**DO NOT use aggressive values** (they cause underfitting):

```yaml
# configs/training/baseline.yaml

# SAFE hyperparameters (use these)
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 0.01  # NOT 0.05 (causes underfitting)
  betas: [0.9, 0.999]

model:
  dropout: 0.3  # NOT 0.45 (causes underfitting)
  label_smoothing: 0.1  # NOT 0.15 (no benefit)

data:
  augmentation:
    scale: [0.8, 1.0]  # NOT [0.5, 1.0] (too aggressive)
    random_erasing_p: 0.25  # NOT 0.5
    color_jitter: null  # NO ColorJitter (hurts roadwork detection)

# AGGRESSIVE values (DO NOT USE):
# dropout: 0.45  # Too high - underfits
# weight_decay: 0.05  # Too high - underfits
# label_smoothing: 0.15  # Too high - no benefit
```

**Why**: "Weight decay is not 'higher is always better'; optimal WD depends on dataset size." (bestoffall.md)

---

### 5. ExPLoRA (BIGGEST ACCURACY GAIN)

**Key insight**: Continue SSL pretraining on unlabeled roadwork images with LoRA adapters BEFORE supervised fine-tuning.

```python
# src/models/explora.py
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

class ExPLoRAWrapper(nn.Module):
    """
    ExPLoRA: Extended Pre-training with LoRA

    Two-phase training:
    1. Phase 1a: ExPLoRA pretraining (5-10 epochs, unlabeled roadwork data, DINOv2 objective)
    2. Phase 1b: Supervised fine-tuning (15 epochs, labeled data, CE loss)

    Expected gain: +8.2% accuracy vs standard fine-tuning
    Cost: 8-10× cheaper than full pretraining
    Reference: arXiv 2406.10973
    """

    def __init__(
        self,
        backbone: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list = None,
        unfreeze_last_n_blocks: int = 2,
    ):
        super().__init__()

        # Freeze entire backbone first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks (full fine-tuning for these layers)
        if unfreeze_last_n_blocks > 0:
            blocks = backbone.blocks[-unfreeze_last_n_blocks:]
            for block in blocks:
                for param in block.parameters():
                    param.requires_grad = True

        # Add LoRA adapters to remaining layers
        if target_modules is None:
            target_modules = ["qkv", "mlp.fc1", "mlp.fc2"]  # DINOv3 modules

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.backbone = get_peft_model(backbone, lora_config)

        # Print trainable params
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"ExPLoRA: {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)")

    def forward(self, x):
        return self.backbone(x)

    def save_merged(self, save_path: str):
        """Save backbone with LoRA weights merged for inference"""
        self.backbone = self.backbone.merge_and_unload()
        torch.save(self.backbone.state_dict(), save_path)
        print(f"✅ Merged ExPLoRA weights saved to {save_path}")

# Usage:
# backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# explora = ExPLoRAWrapper(backbone, r=8, lora_alpha=16)
# explora_pretrain(explora, unlabeled_roadwork_data)  # Continue DINOv2 objective
# supervised_train(explora, labeled_data)  # Standard classification
```

---

### 6. Complete Calibration Methods (SCRC, Temperature, Dirichlet)

```python
# src/calibration/scrc.py
import numpy as np
import torch
from typing import Tuple

class SCRCCalibrator:
    """
    Split Conformal Risk Control

    Handles label noise and distributional shift better than standard conformal prediction.

    Reference: "Split Conformal Prediction under Data Contamination" (2023)
    """

    def __init__(
        self,
        alpha: float = 0.1,  # Coverage level (1 - alpha = 90% coverage)
        contamination_rate: float = 0.05,  # Assume 5% label noise
        method: str = "bonferroni",  # bonferroni | bootstrap
    ):
        self.alpha = alpha
        self.contamination_rate = contamination_rate
        self.method = method
        self.fitted = False

    def fit(
        self,
        class_logits: np.ndarray,  # [N, num_classes]
        gate_logits: np.ndarray,   # [N, 1] optional
        labels: np.ndarray,         # [N]
    ) -> dict:
        """
        Fit SCRC thresholds on validation calibration set

        Returns:
            params: dict with lambda1, lambda2 thresholds
        """
        # Convert logits to probabilities
        class_probs = torch.softmax(torch.from_numpy(class_logits), dim=1).numpy()
        max_probs = class_probs.max(axis=1)
        pred_labels = class_probs.argmax(axis=1)

        # Compute correctness scores
        is_correct = (pred_labels == labels).astype(float)

        # Adjust alpha for contamination (Bonferroni correction)
        if self.method == "bonferroni":
            alpha_adjusted = self.alpha / (1 - self.contamination_rate)
        else:
            alpha_adjusted = self.alpha

        # Compute lambda1 (class confidence threshold)
        correct_scores = max_probs[is_correct == 1]
        if len(correct_scores) > 0:
            lambda1 = np.percentile(correct_scores, 100 * (alpha_adjusted))
        else:
            lambda1 = 0.5

        # Compute lambda2 (gate threshold if available)
        if gate_logits is not None:
            gate_scores = torch.sigmoid(torch.from_numpy(gate_logits)).numpy().flatten()
            lambda2 = np.percentile(gate_scores[is_correct == 1], 100 * (alpha_adjusted))
        else:
            lambda2 = None

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.fitted = True

        return {
            "lambda1": float(lambda1),
            "lambda2": float(lambda2) if lambda2 is not None else None,
            "alpha": self.alpha,
            "contamination_rate": self.contamination_rate,
        }

    def predict(
        self,
        class_probs: np.ndarray,  # [N, num_classes]
        gate_score: np.ndarray = None,  # [N] optional
    ) -> np.ndarray:
        """
        Predict with SCRC

        Returns:
            prediction_sets: list of sets (prediction sets per sample)
        """
        if not self.fitted:
            raise ValueError("Must call fit() before predict()")

        max_probs = class_probs.max(axis=1)
        pred_labels = class_probs.argmax(axis=1)

        prediction_sets = []
        for i in range(len(class_probs)):
            # Check confidence threshold
            if max_probs[i] >= self.lambda1:
                if gate_score is not None and self.lambda2 is not None:
                    # Check gate threshold
                    if gate_score[i] >= self.lambda2:
                        # High confidence - return singleton set
                        prediction_sets.append({pred_labels[i]})
                    else:
                        # Low gate score - return full set
                        prediction_sets.append(set(range(class_probs.shape[1])))
                else:
                    # No gate - just check class confidence
                    prediction_sets.append({pred_labels[i]})
            else:
                # Low confidence - return full set (defer)
                prediction_sets.append(set(range(class_probs.shape[1])))

        return prediction_sets


# src/calibration/temperature.py
import torch
import torch.nn as nn
from torch.optim import LBFGS

class TemperatureScaling(nn.Module):
    """
    Temperature scaling calibration

    Expected ECE reduction: 50-60%
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50):
        """
        Fit temperature on validation calibration set

        Args:
            logits: [N, num_classes] logits from model
            labels: [N] ground truth labels
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        print(f"✅ Temperature fitted: T = {self.temperature.item():.3f}")
        return self.temperature.item()
```

---

### 7. Complete Metrics (AUROC, AUPRC, ECE, AUGRC, Bootstrap CI)

```python
# src/metrics/calibration.py
import numpy as np
from typing import Tuple

def expected_calibration_error(
    probs: np.ndarray,  # [N, num_classes]
    labels: np.ndarray,  # [N]
    num_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE)

    Excellent: <0.05
    Good: 0.05-0.10
    Poor: >0.10
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def adaptive_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """
    Adaptive Calibration Error (ACE)

    Uses equal-mass bins instead of equal-width bins.
    Better for skewed confidence distributions.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    # Create equal-mass bins
    bin_boundaries = np.percentile(confidences, np.linspace(0, 100, num_bins + 1))

    ace = 0.0
    for i in range(num_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if i == num_bins - 1:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ace += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ace


# src/metrics/selective.py
import numpy as np
from sklearn.metrics import auc

def area_under_risk_coverage(
    risks: np.ndarray,  # [N] predicted risks (e.g., 1 - max_prob)
    correctness: np.ndarray,  # [N] binary correctness
) -> float:
    """
    Area Under Risk-Coverage curve (AUGRC)

    Lower is better.
    Excellent: <0.05
    Good: 0.05-0.10
    Poor: >0.10
    """
    # Sort by risk (ascending)
    sorted_indices = np.argsort(risks)
    sorted_correctness = correctness[sorted_indices]

    # Compute coverage and risk at each threshold
    coverages = []
    cumulative_risks = []

    for i in range(1, len(sorted_correctness) + 1):
        coverage = i / len(sorted_correctness)
        risk = 1 - sorted_correctness[:i].mean()  # Error rate

        coverages.append(coverage)
        cumulative_risks.append(risk)

    # Compute AUC
    augrc = auc(coverages, cumulative_risks)

    return augrc


# src/metrics/bootstrap.py
import numpy as np
from typing import Tuple, Callable

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    num_samples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any metric

    Args:
        y_true: [N] ground truth labels
        y_pred: [N] predictions
        metric_fn: function(y_true, y_pred) -> metric_value
        num_samples: number of bootstrap resamples
        confidence_level: confidence level (0.95 = 95% CI)

    Returns:
        mean, lower_bound, upper_bound
    """
    np.random.seed(random_seed)

    N = len(y_true)
    bootstrap_metrics = []

    for _ in range(num_samples):
        # Sample with replacement
        indices = np.random.choice(N, size=N, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metric
        metric_value = metric_fn(y_true_boot, y_pred_boot)
        bootstrap_metrics.append(metric_value)

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Compute percentiles
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    mean = bootstrap_metrics.mean()
    lower = np.percentile(bootstrap_metrics, lower_percentile)
    upper = np.percentile(bootstrap_metrics, upper_percentile)

    return mean, lower, upper

# Example usage:
# mean, lower, upper = bootstrap_confidence_interval(
#     y_true=labels,
#     y_pred=predictions,
#     metric_fn=lambda yt, yp: (yt == yp).mean(),  # Accuracy
#     num_samples=1000,
#     confidence_level=0.95,
# )
# print(f"Accuracy: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
```

---

### 8. Slice-Based Evaluation (Day/Night/Weather)

```python
# src/metrics/slicing.py
import numpy as np
import pandas as pd
from typing import Dict, List, Callable

class SliceEvaluator:
    """
    Slice-based evaluation for fairness and robustness

    Evaluate model performance on different data slices:
    - Time of day (day/night/dawn/dusk)
    - Weather (clear/rain/snow/fog)
    - Camera source (natix/roadwork)
    - Confidence bins (very_low/low/medium/high/very_high)
    """

    def __init__(self, slice_definitions: Dict):
        self.slice_definitions = slice_definitions

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        metadata: pd.DataFrame,
        metrics: Dict[str, Callable],
    ) -> pd.DataFrame:
        """
        Evaluate all metrics on all slices

        Args:
            y_true: [N] ground truth labels
            y_pred: [N] predicted labels
            y_probs: [N, num_classes] predicted probabilities
            metadata: DataFrame with columns like 'hour', 'weather', etc.
            metrics: dict of metric functions

        Returns:
            DataFrame with metrics per slice
        """
        results = []

        for slice_name, slice_config in self.slice_definitions.items():
            attribute = slice_config['attribute']

            for slice_value, slice_filter in slice_config['slices'].items():
                # Get samples in this slice
                if 'range' in slice_filter:
                    # Time range
                    low, high = slice_filter['range']
                    if low < high:
                        mask = (metadata[attribute] >= low) & (metadata[attribute] < high)
                    else:
                        # Wrap around (e.g., night: 20-6)
                        mask = (metadata[attribute] >= low) | (metadata[attribute] < high)
                elif 'values' in slice_filter:
                    # Categorical values
                    mask = metadata[attribute].isin(slice_filter['values'])
                else:
                    continue

                # Skip if too few samples
                if mask.sum() < 10:
                    continue

                # Compute metrics on slice
                slice_metrics = {
                    'slice_name': slice_name,
                    'slice_value': slice_value,
                    'sample_count': mask.sum(),
                }

                for metric_name, metric_fn in metrics.items():
                    try:
                        value = metric_fn(
                            y_true[mask],
                            y_pred[mask],
                            y_probs[mask] if y_probs is not None else None,
                        )
                        slice_metrics[metric_name] = value
                    except Exception as e:
                        slice_metrics[metric_name] = np.nan

                results.append(slice_metrics)

        return pd.DataFrame(results)

# Example slice definitions (from olanzapin.md)
SLICE_DEFINITIONS = {
    'time_of_day': {
        'attribute': 'hour',
        'slices': {
            'day': {'range': [8, 18]},      # 8am-6pm
            'night': {'range': [20, 6]},    # 8pm-6am
            'dawn': {'range': [6, 8]},
            'dusk': {'range': [18, 20]},
        }
    },
    'weather': {
        'attribute': 'weather',
        'slices': {
            'clear': {'values': ['clear', 'sunny']},
            'rain': {'values': ['rain', 'drizzle']},
            'snow': {'values': ['snow', 'sleet']},
            'fog': {'values': ['fog', 'mist']},
        }
    },
}
```

---

## EXPECTED RESULTS SUMMARY

| Phase | Features | Accuracy | ECE | Speed | Cost |
|-------|----------|----------|-----|-------|------|
| **Baseline** | Single-view DINOv3 | 69% | 0.29 | 1× | $120 |
| **Phase 1** | + Multi-view (1+9 crops) | 72-77% | 0.25 | 1.1× | $120 |
| **Phase 1b** | + ExPLoRA (+8.2%) | 85% | 0.20 | 1.1× | $12 |
| **Phase 2** | + DoRAN + Flash Attn 3 | 88% | 0.18 | 2× | $12 |
| **Phase 3** | + Calibration + Fusion | 90% | 0.05 | 6× | $12 |

**Total Improvement**:
- Accuracy: 69% → 90% (+21 percentage points)
- Calibration: ECE 0.29 → 0.05 (-83% error)
- Speed: 6× faster inference (batched multi-view + Flash Attn 3 + TensorRT)
- Cost: 10× cheaper training ($120 → $12 with ExPLoRA)

---

## COMPLETE FILE CHECKLIST (18 Priority Files)

### Priority 1 - Core Architecture (Day 1)
- [ ] `src/contracts/artifact_schema.py` - All file paths
- [ ] `src/contracts/split_contracts.py` - Leakage prevention
- [ ] `src/contracts/validators.py` - Fail-fast checking
- [ ] `src/models/multi_view.py` - Multi-view inference (CRITICAL)
- [ ] `src/data/multi_view_loader.py` - Multi-crop dataloader

### Priority 2 - Training (Day 1-2)
- [ ] `src/models/module.py` - LightningModule (main training logic)
- [ ] `src/data/datamodule.py` - LightningDataModule
- [ ] `src/training/ema.py` - EMA implementation
- [ ] `src/models/backbone.py` - DINOv3 wrapper
- [ ] `src/models/head.py` - Classification head
- [ ] `src/data/datasets.py` - NATIX dataset

### Priority 3 - ExPLoRA (Day 2-3)
- [ ] `src/models/explora.py` - ExPLoRA PEFT (+8.2%)
- [ ] `src/training/explora_pretrain.py` - SSL pretraining
- [ ] `src/pipeline/dag_engine.py` - DAG orchestrator

### Priority 4 - SOTA Features (Day 3)
- [ ] `src/models/doran.py` - DoRAN head (+1-3%)
- [ ] `src/models/flash_attention.py` - Flash Attention 3

### Priority 5 - Production (Day 4)
- [ ] `src/calibration/scrc.py` - SCRC calibration
- [ ] `src/metrics/calibration.py` - ECE, MCE, ACE, Brier
- [ ] `src/metrics/selective.py` - AUGRC, risk-coverage
- [ ] `src/metrics/bootstrap.py` - Bootstrap CI
- [ ] `src/metrics/slicing.py` - Slice-based evaluation
- [ ] `scripts/40_export.py` - TensorRT export

---

## SUCCESS CRITERIA

**Must achieve ALL of these to consider implementation complete**:

1. ✅ **Accuracy**: ≥88% on validation set (target: 90%)
2. ✅ **Calibration**: ECE ≤0.10 (target: 0.05)
3. ✅ **Speed**: ≥30 FPS inference on single GPU
4. ✅ **Cost**: Training cost ≤$15 (with ExPLoRA)
5. ✅ **Leakage**: Zero data leakage (enforced by split contracts)
6. ✅ **Multi-view**: Batched forward pass working correctly
7. ✅ **Architecture**: PyTorch Lightning + Hydra structure
8. ✅ **Metrics**: Complete evaluation (AUROC, ECE, bootstrap CI, slicing)
9. ✅ **Calibration**: SCRC + temperature + Dirichlet working
10. ✅ **Tests**: Integration tests pass for all phases

---

## NEXT STEPS

1. ✅ Download PyTorch Lightning template
2. ✅ Setup folder structure
3. ✅ Implement Priority 1 files (core architecture)
4. ✅ Implement Priority 2 files (training)
5. ✅ Train Phase 1 baseline (multi-view)
6. ✅ Implement Priority 3 files (ExPLoRA)
7. ✅ Train Phase 1b (ExPLoRA pretraining)
8. ✅ Implement Priority 4 files (SOTA features)
9. ✅ Train Phase 2 (DoRAN + Flash Attention)
10. ✅ Implement Priority 5 files (production)
11. ✅ Train Phase 3 (multi-dataset + calibration)
12. ✅ Final validation and deployment

**Ready to start implementation!**
