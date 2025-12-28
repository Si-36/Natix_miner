"""
Example: Train DINOv3 Baseline Classifier

This example shows how to train a basic DINOv3 classifier using the foundation
components we built in TODO 1-20.

Usage:
    python examples/train_baseline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from data import NATIXDataModule
from models import DINOv3Classifier


def main():
    """Train baseline DINOv3 classifier"""

    # ========== Configuration ==========
    DATA_ROOT = "/data/natix"  # Update this to your data path
    SPLITS_JSON = "outputs/data_splits/splits.json"
    DINOV3_CHECKPOINT = (
        "../streetvision_cascade/models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m"
    )

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # ========== Create DataModule ==========
    print("Creating DataModule...")
    datamodule = NATIXDataModule(
        data_root=DATA_ROOT,
        splits_json=SPLITS_JSON,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # ========== Create Model ==========
    print("Creating Model...")
    model = DINOv3Classifier(
        backbone_name="vit_huge",  # ViT-H/16+ (1280-dim)
        num_classes=13,  # NATIX roadwork classes
        pretrained_path=DINOV3_CHECKPOINT,
        freeze_backbone=True,  # Freeze DINOv3, only train head
        head_type="linear",  # Use linear head (DoRAN in TODO 141-160)
        dropout_rate=0.3,  # Recommended for overfitting prevention
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        use_ema=True,  # EMA for better convergence (+0.5-1.5%)
        ema_decay=0.9999,
    )

    print(f"\n{model}")
    print(f"\nTotal parameters: {model.num_parameters:,}")
    print(f"Trainable parameters: {model.num_trainable_parameters:,}")

    # ========== Callbacks ==========
    callbacks = [
        # Model checkpoint (save best model based on val_acc)
        ModelCheckpoint(
            dirpath="outputs/phase1/checkpoints",
            filename="dinov3-baseline-{epoch:02d}-{val/acc:.4f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        # Early stopping (stop if val_acc doesn't improve for 10 epochs)
        EarlyStopping(
            monitor="val/acc",
            mode="max",
            patience=10,
            verbose=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ========== Logger ==========
    logger = TensorBoardLogger(
        save_dir="outputs/phase1/logs",
        name="dinov3_baseline",
    )

    # ========== Trainer ==========
    print("\nCreating Trainer...")
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Use GPU if available
        devices=1,
        precision="16-mixed",  # Mixed precision training (2× memory, 1.5× speed)
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        gradient_clip_val=1.0,  # Gradient clipping for stability
        deterministic=True,  # Reproducible results
    )

    # ========== Train ==========
    print("\nStarting training...")
    print("=" * 80)

    trainer.fit(model, datamodule=datamodule)

    print("=" * 80)
    print("Training complete!")

    # ========== Evaluate on Test Set ==========
    print("\nEvaluating on test set (val_test)...")
    trainer.test(model, datamodule=datamodule)

    print("\n✅ Done!")
    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val_acc: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    # Check if paths exist
    data_root = Path("/data/natix")
    splits_json = Path("outputs/data_splits/splits.json")

    if not data_root.exists():
        print(f"⚠️  Data root not found: {data_root}")
        print("Please update DATA_ROOT in this script to point to your NATIX dataset")
        sys.exit(1)

    if not splits_json.exists():
        print(f"⚠️  Splits file not found: {splits_json}")
        print("Please run split generation first:")
        print("  python -m src.data.split_generator /data/natix outputs/data_splits/splits.json")
        sys.exit(1)

    # Run training
    main()
