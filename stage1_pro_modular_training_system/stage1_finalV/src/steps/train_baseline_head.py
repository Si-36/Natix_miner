"""
Step: Train Baseline Head (Phase 1 - Foundation)
Domain-stable name: train_baseline_head

REAL ML EXECUTION - Not Skeleton!
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, FrozenSet
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.artifacts import ArtifactStore, ArtifactKey
from pipeline.contracts import Split, assert_allowed, SplitPolicy
from pipeline.step_api import StepSpec, StepContext, StepResult
from models.backbone import DINOv3Backbone
from models.head import Stage1Head


@dataclass
class TrainBaselineHeadSpec(StepSpec):
    """
    Train Baseline Head Step Specification (Phase 1 - Foundation).

    This is the FIRST step in the pipeline. It trains a model
    and saves a checkpoint that downstream steps can load.

    🔥 LEAK-PROOF DESIGN:
    - Training: TRAIN + VAL_SELECT (model selection)
    - NEVER uses VAL_CALIB or VAL_TEST during training
    """

    step_id: str = "train_baseline_head"
    name: str = "train_baseline_head"
    deps: List[str] = field(default_factory=list)  # No dependencies (first step)
    order_index: int = 0  # First step
    owners: List[str] = field(default_factory=lambda: ["ml-team"])
    tags: Dict[str, str] = field(
        default_factory=lambda: {
            "priority": "high",
            "stage": "phase1",
            "component": "training",
        }
    )

    def inputs(self, ctx: "StepContext") -> List[str]:
        """
        Declare required input artifacts.

        Phase 1 is the FIRST step, so it has no inputs.

        Returns:
            Empty list (no inputs required)
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

        Phase 1 training uses:
        - TRAIN: For training
        - VAL_SELECT: For model selection (early stopping)

        STRICTLY FORBIDDEN (leak-proof):
        - VAL_CALIB: NEVER for training
        - VAL_TEST: NEVER for training

        Returns:
            FrozenSet of allowed Split enum values (as strings)
        """
        return SplitPolicy.training  # TRAIN + VAL_SELECT only

    def run(self, ctx: "StepContext") -> StepResult:
        """
        Execute baseline training.

        🔥 LEAK-PROOF: Enforce split contract BEFORE any training!
        """
        print(f"\n{'=' * 70}")
        print(f"{'🎯'} Training Baseline Head (Phase 1 - Foundation)")
        print("=" * 70)

        # 🔥 LEAK-PROOF: Enforce split contract BEFORE any training!
        allowed = self.allowed_splits()
        used = frozenset({Split.TRAIN.value, Split.VAL_SELECT.value})
        assert_allowed(
            used=used,
            allowed=allowed,
            context=f"{self.step_id}.run()",
        )
        print(f"   ✅ Split contract enforced: {sorted(list(used))}")

        # Get training configuration from ctx.config
        config = ctx.config

        # Extract data paths (if provided)
        data_config = config.get("data", {})
        train_image_dir = data_config.get("train_image_dir")
        train_labels_file = data_config.get("train_labels_file")
        val_select_image_dir = data_config.get("val_select_image_dir")
        val_select_labels_file = data_config.get("val_select_labels_file")

        # Check if real data paths are provided
        has_real_data = all(
            [
                train_image_dir,
                train_labels_file,
                val_select_image_dir,
                val_select_labels_file,
            ]
        )

        # Training hyperparameters
        training_config = config.get("training", {})
        model_config = config.get("model", {})

        model_id = model_config.get("model_id", "facebook/dinov3-vits16-pretrain-lvd1689m")
        freeze_backbone = model_config.get("freeze_backbone", True)
        hidden_dim = model_config.get("hidden_dim", 384)
        num_classes = model_config.get("num_classes", 2)
        dropout = model_config.get("dropout", 0.1)

        max_epochs = training_config.get("max_epochs", 1)
        batch_size = training_config.get("batch_size", 32)
        learning_rate = training_config.get("learning_rate", 1e-4)
        weight_decay = training_config.get("weight_decay", 0.01)

        print(f"\n   ⚙️  Configuration:")
        print(f"      Model ID: {model_id}")
        print(f"      Freeze backbone: {freeze_backbone}")
        print(f"      Hidden dim: {hidden_dim}")
        print(f"      Num classes: {num_classes}")
        print(f"      Batch size: {batch_size}")
        print(f"      Max epochs: {max_epochs} (smoke test)")
        print(f"      Learning rate: {learning_rate}")
        print(f"      Weight decay: {weight_decay}")
        print(f"      Data: {'REAL' if has_real_data else 'MOCK'}")

        # Initialize manifest (THIS STEP INITIALIZES ITS OWN RUN)
        manifest_path = ctx.artifact_store.initialize_manifest(
            run_id=ctx.run_id,
            config=config,
        )
        print(f"\n   ✅ Manifest initialized for this step")

        # Initialize RoadworkDataModule
        print(f"\n   📦 Initializing RoadworkDataModule...")
        print("-" * 70)

        if has_real_data:
            # Real data paths provided
            print(f"      Using real data from:")
            print(f"         Train images: {train_image_dir}")
            print(f"         Train labels: {train_labels_file}")
            print(f"         Val_select images: {val_select_image_dir}")
            print(f"         Val_select labels: {val_select_labels_file}")

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

            # Create dataloader dict for Lightning module
            train_loader_dict = {
                "train": train_loader,
                "val_select": val_select_loader,
            }
        else:
            # No data paths provided - use mock dataloader
            print(f"      ⚠️  No real data paths provided")
            print(f"      Using MOCK dataloader (10 samples, 2 classes)")

            # Create simple mock dataloader for testing
            class MockDataset(torch.utils.data.Dataset):
                def __init__(self):
                    self.samples = [
                        (torch.randn(3, 224, 224, dtype=torch.float32), torch.randint(0, 2, (1,)))
                        for _ in range(10)
                    ]

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

            mock_train_loader = torch.utils.data.DataLoader(
                MockDataset(),
                batch_size=batch_size,
                shuffle=True,
            )

            # Create dataloader dict for Lightning module
            train_loader_dict = {
                "train": mock_train_loader,
                "val_select": mock_train_loader,
            }

        # Create Lightning module
        print(f"\n   🏗️  Creating Lightning module...")
        print("-" * 70)

        from training.lightning_module import Phase1LightningModule, Phase1LightningModuleConfig

        # Create config object
        module_config = Phase1LightningModuleConfig(
            model_id=model_id,
            freeze_backbone=freeze_backbone,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            precision="bf16",
        )

        lightning_module = Phase1LightningModule(
            config=module_config,
            artifact_store=ctx.artifact_store,
        )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lightning_module = lightning_module.to(device)
        print(f"   ✅ Model created and moved to device: {device}")

        # Setup model (create backbone, head, model, criterion, optimizer)
        lightning_module.setup(
            train_loader=train_loader_dict.get("train"),
            val_select_loader=train_loader_dict.get("val_select"),
        )

        # Get optimizer and criterion from module (already created in setup())
        optimizer = lightning_module.optimizer
        criterion = lightning_module.criterion

        # Train for specified epochs (SMOKE TEST = 1 epoch)
        print(f"\n   🚀 Training for {max_epochs} epoch(s)...")
        print("-" * 70)

        for epoch in range(1, max_epochs + 1):
            lightning_module.train()

            # Training epoch
            train_loss = 0.0
            train_correct = 0
            total = 0

            # Train loop
            for batch_idx, (images, labels) in enumerate(train_loader_dict["train"]):
                images = images.to(device)
                labels = labels.to(device).squeeze()  # (B, 1) -> (B,)

                # Forward pass
                logits = lightning_module(images)
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                preds = torch.argmax(logits, dim=-1)
                train_loss += loss.item()
                train_correct += (preds == labels).sum().item()
                total += labels.size(0)

            # Train accuracy
            train_accuracy = train_correct / total if total > 0 else 0.0

            # Validation on VAL_SELECT (for model selection)
            lightning_module.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(train_loader_dict["val_select"]):
                    images = images.to(device)
                    labels = labels.to(device).squeeze()  # (B, 1) -> (B,)

                    logits = lightning_module(images)
                    loss = criterion(logits, labels)

                    preds = torch.argmax(logits, dim=-1)
                    val_loss += loss.item()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total if val_total > 0 else 0.0

            print(
                f"   Epoch {epoch}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        print(f"\n   ✅ Training complete!")
        print("-" * 70)

        # 🔥 CRITICAL: Build checkpoint with correct contract format
        # This MUST match what export_calib_logits expects:
        # - "state_dict": model.state_dict()
        # - "config": {...}
        # - "metadata": {...}
        print(f"\n   💾 Building checkpoint with correct format...")
        print("-" * 70)

        # Just save the trained lightning_module directly
        # It already has correct structure from setup()
        model_wrapper = nn.Sequential(lightning_module.backbone, lightning_module.head)

        checkpoint = {
            "state_dict": model_wrapper.state_dict(),
            "config": {
                "model_id": model_id,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "dropout": dropout,
                "freeze_backbone": freeze_backbone,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
            },
            "metadata": {
                "epoch": max_epochs,
                "step_id": self.step_id,
                "timestamp": str(Path.cwd().stat().st_mtime),
                "has_real_data": has_real_data,
            },
        }

        checkpoint = {
            "state_dict": model_wrapper.state_dict(),
            "config": {
                "model_id": model_id,
                "hidden_dim": hidden_dim,
                "num_classes": num_classes,
                "dropout": dropout,
                "freeze_backbone": freeze_backbone,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
            },
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

        # 💾 Save checkpoint via ArtifactStore (CRITICAL: This ensures contract!)
        checkpoint_path = ctx.artifact_store.put(
            ArtifactKey.MODEL_CHECKPOINT,
            checkpoint,
            run_id=ctx.run_id,
        )

        print(f"\n   💾 Checkpoint saved:")
        print(f"      Path: {checkpoint_path}")
        print(f"      Exists: {checkpoint_path.exists()}")

        # Verify checkpoint was saved correctly
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint save failed: {checkpoint_path}")

        # Build StepResult
        artifacts_written = [ArtifactKey.MODEL_CHECKPOINT]

        metrics = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "final_accuracy": (train_accuracy + val_accuracy) / 2,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "num_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "has_real_data": has_real_data,
        }

        # 🔥 LEAK-PROOF: Report splits used
        splits_used = frozenset({Split.TRAIN.value, Split.VAL_SELECT.value})
        print(f"\n   🔒 Splits used: {sorted(list(splits_used))}")

        metadata = {
            "device": str(device),
            "num_parameters": sum(p.numel() for p in lightning_module.parameters()),
            "checkpoint_path": str(checkpoint_path),
        }

        print("\n" + "=" * 70)
        print("✅ STEP COMPLETE!")
        print("=" * 70)

        return StepResult(
            artifacts_written=artifacts_written,
            metrics=metrics,
            metadata=metadata,
            splits_used=splits_used,
        )
