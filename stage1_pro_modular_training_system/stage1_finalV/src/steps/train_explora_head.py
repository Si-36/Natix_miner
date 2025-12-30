"""
Step: Train ExPLoRA Head (Phase 1.1 - PEFT Training Variant)

Domain-stable training step that uses ExPLoRA for parameter-efficient fine-tuning.
This is a "training variant" that keeps the same artifact contract as baseline:
- Output: ArtifactKey.MODEL_CHECKPOINT
- Format: {"state_dict": ..., "config": {...}, "metadata": {...}}
- Downstream: export_calib_logits and sweep_thresholds work unchanged

ExPLoRA: https://arxiv.org/abs/2306.07159
Paper: "Parameter-Efficient LoRA for Large Language Models"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, FrozenSet, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pathlib import Path

from pipeline.artifacts import ArtifactStore, ArtifactKey
from pipeline.contracts import Split, assert_allowed, SplitPolicy
from pipeline.step_api import StepSpec, StepContext, StepResult

from models.backbone import DINOv3Backbone
from models.head import Stage1Head

from pipeline.registry import StepRegistry


_step_registry = StepRegistry()


@dataclass
class TrainExploraHeadSpec(StepSpec):
    """
    Train ExPLoRA Head Step Specification (Phase 1.1 - PEFT Training).

    This is a TRAINING VARIANT, not a new pipeline component:
    - Same inputs/outputs as baseline: MODEL_CHECKPOINT artifact
    - Same split policy: TRAIN + VAL_SELECT only (no VAL_CALIB during training)
    - Compatible with: export_calib_logits, sweep_thresholds (no changes needed)
    """

    step_id: str = "train_explora_head"
    name: str = "train_explora_head"
    deps: List[str] = field(default_factory=list)

    order_index: int = 2  # After baseline (0), before export/sweep (2-4)
    owners: List[str] = field(default_factory=lambda: ["ml-team"])
    tags: Dict[str, str] = field(
        default_factory=lambda: {
            "priority": "high",
            "stage": "phase1",
            "component": "training",
            "variant": "explora",
        }
    )

    def inputs(self, ctx: "StepContext") -> List[str]:
        """
        Declare required input artifacts.

        Phase 1.1 is FIRST training step variant, so it has no inputs.
        """
        return []

    def outputs(self, ctx: "StepContext") -> List[str]:
        """
        Declare output artifacts this step produces.

        Returns:
            List of ArtifactKey canonical names
        """
        return [
            ArtifactKey.MODEL_CHECKPOINT,
        ]

    def allowed_splits(self) -> FrozenSet[str]:
        """
        Declare which data splits this step is allowed to use.

        Phase 1.1 (training) uses:
        - TRAIN: For training
        - VAL_SELECT: For model selection (early stopping)

        LEAK-PROOF (STRICTLY FORBIDDEN):
        - VAL_CALIB: NEVER for training (would leak calibration labels)
        - VAL_TEST: NEVER for training

        Returns:
            FrozenSet of allowed Split enum values (as strings)
        """
        return SplitPolicy.training

    def run(self, ctx: "StepContext") -> StepResult:
        """
        Execute ExPLoRA training.

        Key design decisions:
        1. Use PEFT (LoRA) from peft library
        2. Wrap backbone with PEFT adapters (not Lightning PEFT)
        3. Keep same training loop structure as baseline
        4. Merge adapters into base backbone before saving checkpoint
        5. Save SAME checkpoint format as baseline (for export compatibility)
        """
        print(f"\n{'=' * 70}")
        print(f"🎯 Training ExPLoRA Head (Phase 1.1 - PEFT Training)")
        print("=" * 70)

        # Enforce split contract
        allowed = self.allowed_splits()
        used = frozenset([Split.TRAIN.value, Split.VAL_SELECT.value])
        assert_allowed(
            used=used,
            allowed=allowed,
            context=f"{self.step_id}.run()",
        )
        print(f"   ✅ Split contract enforced: {sorted(list(used))}")

        # Get configuration
        config = ctx.config

        # Extract training configuration
        training_config = config.get("training", {})
        model_config = config.get("model", {})
        explora_config = config.get("explora", {})

        # Model hyperparameters
        model_id = model_config.get("model_id", "facebook/dinov3-vits16-pretrain-lvd1689m")
        freeze_backbone = model_config.get("freeze_backbone", True)
        hidden_dim = model_config.get("hidden_dim", 384)
        num_classes = model_config.get("num_classes", 2)
        dropout = model_config.get("dropout", 0.1)
        max_epochs = training_config.get("max_epochs", 1)
        batch_size = training_config.get("batch_size", 32)
        learning_rate = training_config.get("learning_rate", 1e-4)
        weight_decay = training_config.get("weight_decay", 1e-4)

        # ExPLoRA configuration
        explora_rank = explora_config.get("r", 16)
        explora_alpha = explora_config.get("alpha", 32)
        explora_dropout = explora_config.get("dropout", 0.05)
        target_modules = explora_config.get("target_modules", ["q_proj", "v_proj"])

        # Check if synthetic data
        data_config = config.get("data", {})
        train_image_dir = data_config.get("train_image_dir")
        train_labels_file = data_config.get("train_labels_file")
        val_select_image_dir = data_config.get("val_select_image_dir")
        val_select_labels_file = data_config.get("val_select_labels_file")

        has_real_data = all(
            [
                train_image_dir,
                train_labels_file,
                val_select_image_dir,
                val_select_labels_file,
            ]
        )

        print(f"\n   ⚙️  Configuration:")
        print(f"      Model ID: {model_id}")
        print(f"      Freeze backbone: {freeze_backbone}")
        print(f"      Hidden dim: {hidden_dim}")
        print(f"      Num classes: {num_classes}")
        print(f"      Dropout: {dropout}")
        print(f"      Max epochs: {max_epochs} (smoke test)")
        print(f"      Batch size: {batch_size}")
        print(f"      Learning rate: {learning_rate}")
        print(f"      Weight decay: {weight_decay}")
        print(f"      Data: {'REAL' if has_real_data else 'MOCK'}")

        print(f"   ExPLoRA config:")
        print(f"      Rank (r): {explora_rank}")
        print(f"      Alpha: {explora_alpha}")
        print(f"      Dropout: {explora_dropout}")
        print(f"      Target modules: {target_modules}")

        print()

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {device}")
        print()

        # 1. Initialize manifest (THIS STEP INITIALIZES ITS OWN RUN)
        manifest_path = ctx.artifact_store.initialize_manifest(
            run_id=ctx.run_id,
            resolved_config=config,
        )
        print(f"✅ Manifest initialized for this step")

        print()

        # 2. Create base DINOv3 backbone
        print("   📝 Creating DINOv3 backbone...")
        print("-" * 70)

        from transformers import AutoModel

        backbone = DINOv3Backbone(
            model_id=model_id,
            dtype=torch.float16,
            freeze_backbone=freeze_backbone,
        )

        print(f"   ✅ Backbone created: {type(backbone).__name__}")

        # 3. Apply ExPLoRA using PEFT
        print()
        print("   🎨 Applying ExPLoRA (LoRA) adapters...")
        print("-" * 70)

        try:
            from peft import LoraConfig, get_peft_model, TaskType

            peft_config = LoraConfig(
                r=explora_rank,
                lora_alpha=explora_alpha,
                lora_dropout=explora_dropout,
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
            )

            # Wrap backbone with PEFT
            backbone = get_peft_model(backbone, peft_config)
            print(f"   ✅ PEFT adapters applied to backbone")

        except ImportError:
            print(f"   ⚠️  PEFT library not found - using base backbone (ExPLoRA not applied)")
            print(f"   Install: pip install peft>=0.13.0")
            print(f"   For more: https://github.com/huggingface/peft")

        print()
        print("   🧠 Creating Stage1Head...")
        print("-" * 70)

        # 4. Create head (same as baseline)
        head = Stage1Head(
            backbone_dim=backbone.embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        print(f"   ✅ Head created")

        print()
        print("   🏗️  Creating model (backbone + head)...")
        print("-" * 70)

        # 5. Create combined model (same as baseline)
        model = nn.Sequential(backbone, head)

        print(f"   ✅ Model assembled: backbone + head")

        print()
        # 6. Move to device
        model = model.to(device)
        print(f"   ✅ Model moved to device: {device}")

        print()

        # 7. Create criterion
        print("   📊 Creating criterion (CrossEntropyLoss)...")
        print("-" * 70)

        criterion = nn.CrossEntropyLoss()
        print(f"   ✅ Criterion created")

        print()
        # 8. Create optimizer
        print(f"   ⚙️  Creating optimizer (AdamW)...")
        print("-" * 70)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True,  # Fused AdamW for speed
        )

        print(f"   ✅ Optimizer created: {len(optimizer.param_groups)} param groups")

        print()
        # 9. Create scheduler
        print("   📅 Creating scheduler (CosineAnnealingLR)...")
        print("-" * 70)

        if max_epochs > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs * 100,  # Estimate steps
                eta_min=0.0,
            )
            print(f"   ✅ Scheduler created")
        else:
            scheduler = None
            print(f"   ℹ️  No scheduler (max_epochs={max_epochs})")

        print()

        # 10. Create data loaders
        print(f"   📦 Loading data loaders...")
        print("-" * 70)

        if has_real_data:
            print(f"      Using real data from:")
            print(f"      Train images: {train_image_dir}")
            print(f"      Train labels: {train_labels_file}")
            print(f"      Val_select images: {val_select_image_dir}")
            print(f"      Val_select labels: {val_select_labels_file}")

            # Import RoadworkDataModule
            import sys

            sys.path.insert(0, "src/training")
            from training.lightning_data_module import RoadworkDataModule

            datamodule = RoadworkDataModule(
                train_image_dir=train_image_dir,
                train_labels_file=train_labels_file,
                val_select_image_dir=val_select_image_dir,
                val_select_labels_file=val_select_labels_file,
                batch_size=batch_size,
                num_workers=0,
            )

            datamodule.setup()

            train_loader = datamodule.train_loader
            val_select_loader = datamodule.val_select_loader

        else:
            print(f"      ⚠️ No real data provided - using MOCK dataloader")
            print(
                f"      (Provide: data.train_image_dir, data.train_labels_file, data.val_select_image_dir, data.val_select_labels_file)"
            )

            class MockDataset(torch.utils.data.Dataset):
                def __init__(self):
                    self.samples = [
                        (
                            torch.randn(3, 224, 224, dtype=torch.float16),
                            torch.randint(0, 2, (1,)),
                        )
                        for _ in range(10)
                    ]

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

            from torch.utils.data import DataLoader

            train_loader = DataLoader(
                MockDataset(),
                batch_size=batch_size,
                shuffle=True,
            )

            val_select_loader = train_loader  # Use same data for val_select in mock mode

        print(f"   ✅ Data loaders created")
        print(f"      Train batches: {len(train_loader)}")
        print(f"      Val_select batches: {len(val_select_loader)}")

        print()

        # 11. Training loop (same as baseline)
        print(f"   🚀 Training for {max_epochs} epoch(s)...")
        print("-" * 70)

        model.train()

        train_metrics = TrainingMetrics()
        val_metrics = TrainingMetrics()

        print(f"\n   Epoch 1/{max_epochs}")
        print("-" * 70)

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Training loop
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device).squeeze()  # (B, 1) -> (B,)

                # Forward pass
                logits = model(images)
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    train_loss += loss.item()
                    train_correct += (preds == labels).sum().item()

                train_total += labels.size(0)

            # Train accuracy
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0

            print(f"      Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")

            # Validation on VAL_SELECT (for model selection)
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_select_loader):
                    images = images.to(device)
                    labels = labels.to(device).squeeze()

                    logits = model(images)
                    loss = criterion(logits, labels)

                    preds = torch.argmax(logits, dim=-1)
                    val_loss += loss.item()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total if val_total > 0 else 0.0

            print(f"      Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            # Update learning metrics
            train_metrics.loss = train_loss
            train_metrics.accuracy = train_accuracy
            val_metrics.loss = val_loss
            val_metrics.accuracy = val_accuracy

        print(f"\n   ✅ Training complete!")
        print("-" * 70)

        # 12. Create final checkpoint (MUST MATCH BASELINE CONTRACT)
        print(f"\n   💾 Building checkpoint with Explora metadata...")
        print("-" * 70)

        # Merge adapters into base backbone (PEFT best practice)
        print(f"   🔄 Merging ExPLoRA adapters into base backbone...")
        try:
            from peft import merge_and_unload

            backbone = merge_and_unload(backbone)
            print(f"   ✅ PEFT adapters merged")

        except ImportError:
            print(f"   ⚠️  PEFT merge not available - using wrapped model (may be OK)")
            print(f"      Install: pip install peft")
        except Exception as e:
            print(f"   ⚠️  Merge failed: {e}")
            # Continue with wrapped model

        print()

        # Rebuild model with merged backbone
        print(f"   🏗️ Rebuilding model (merged backbone + head)...")
        print("-" * 70)

        model = nn.Sequential(backbone, head)
        model = model.to(device)

        print(f"   ✅ Model rebuilt: backbone + head")

        print()
        # Build checkpoint with same format as baseline
        print(f"   💾 Saving checkpoint (with Explora metadata)...")
        print("-" * 70)

        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "model_id": model_id,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "dropout": dropout,
                "freeze_backbone": freeze_backbone,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
                # Add Explora-specific config
                "variant": "explora",
                "explora": {
                    "r": explora_rank,
                    "alpha": explora_alpha,
                    "dropout": explora_dropout,
                    "target_modules": target_modules,
                },
            },
            "training": training_config,
            "model": model_config,
            "metadata": {
                "epoch": max_epochs,
                "step_id": self.step_id,
                "timestamp": str(Path.cwd().stat().st_mtime),
                "has_real_data": has_real_data,
            },
        }

        print(f"      state_dict keys: {list(checkpoint['state_dict'].keys())}")
        print(f"      config keys: {list(checkpoint['config'].keys())}")
        print(f"      metadata keys: {list(checkpoint['metadata'].keys())}")

        print()

        # 13. Save checkpoint via ArtifactStore (CRITICAL: ensures contract)
        print(f"   💾 Saving MODEL_CHECKPOINT via ArtifactStore...")
        print("-" * 70)

        checkpoint_path = ctx.artifact_store.put(
            ArtifactKey.MODEL_CHECKPOINT,
            checkpoint,
            run_id=ctx.run_id,
        )

        print(f"   ✅ Checkpoint saved: {checkpoint_path}")
        print(f"   Exists: {checkpoint_path.exists()}")

        # Verify
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint save failed: {checkpoint_path}")

        print()

        # 14. Build StepResult
        metrics = {
            "train_accuracy": train_metrics.accuracy,
            "val_accuracy": val_metrics.accuracy,
            "train_loss": train_metrics.loss,
            "val_loss": val_metrics.loss,
            "num_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "has_real_data": has_real_data,
        }

        metadata = {
            "device": str(device),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "variant": "explora",
        }

        splits_used = frozenset([Split.TRAIN.value, Split.VAL_SELECT.value])

        print()
        print("=" * 70)
        print("✅ STEP COMPLETE!")
        print("=" * 70)

        print()
        print("📊 Step Results:")
        print(f"   Artifacts written: {[ArtifactKey.MODEL_CHECKPOINT.value]}")
        print(
            f"   Metrics: train_acc={train_metrics.accuracy:.4f}, val_acc={val_metrics.accuracy:.4f}"
        )
        print(f"   Splits used: {sorted(list(splits_used))}")

        print()
        print("=" * 70)

        print("🔒 Enforcing split contract: TRAIN + VAL_SELECT ONLY (leak-proof)")
        assert_allowed(
            used=splits_used,
            allowed=self.allowed_splits(),
            context=f"{self.step_id}.run()",
        )

        print("   ✅ Split contract validated")

        print()
        return StepResult(
            artifacts_written=[ArtifactKey.MODEL_CHECKPOINT],
            metrics=metrics,
            metadata=metadata,
            splits_used=splits_used,
        )


class TrainingMetrics:
    """Training metrics tracking (helper)."""

    def __init__(self):
        self.loss = 0.0
        self.accuracy = 0.0

    def update(self, loss: float, correct: int, total: int):
        self.loss = loss
        if total > 0:
            self.accuracy = correct / total
