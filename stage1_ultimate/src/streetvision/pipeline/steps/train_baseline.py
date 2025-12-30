"""
Phase 1: Baseline Training Step (Production-Grade 2025-12-30)

Improvements over old implementation:
- ✅ Atomic checkpoint writes (crash-safe)
- ✅ Manifest-last commit (lineage tracking)
- ✅ Centralized metrics (no MCC drift)
- ✅ Validation metrics computed on val_calib
- ✅ Duration tracking
- ✅ Type-safe with proper error handling
- ✅ Fully Hydra-driven (zero hardcoding)

Contract:
- Inputs: splits.json
- Outputs:
  - phase1/model_best.pth (best checkpoint)
  - phase1/val_calib_logits.pt (logits on val_calib)
  - phase1/val_calib_labels.pt (labels on val_calib)
  - phase1/metrics.csv (training metrics)
  - phase1/config.json (hyperparameters)
  - phase1/manifest.json (lineage + checksums) ◄── LAST
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

# Import old modules (gradual migration)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from callbacks import ValCalibArtifactSaver
from contracts.artifact_schema import ArtifactSchema
from data import NATIXDataModule
from models import DINOv3Classifier

# Import new foundation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from streetvision.eval import compute_mcc, compute_all_metrics
from streetvision.io import (
    create_step_manifest,
    write_checkpoint_atomic,
    write_json_atomic,
    write_torch_artifact_atomic,
)

logger = logging.getLogger(__name__)


def run_phase1_baseline(
    artifacts: ArtifactSchema,
    cfg: DictConfig,
) -> None:
    """
    Run Phase 1: Baseline Training with production-grade practices

    Args:
        artifacts: Artifact schema (all file paths)
        cfg: Hydra configuration

    Outputs:
        - model_best.pth: Best model checkpoint (atomic write + SHA256)
        - val_calib_logits.pt: Logits on val_calib split
        - val_calib_labels.pt: Labels on val_calib split
        - metrics.csv: Training metrics from Lightning
        - config.json: Hyperparameters used
        - manifest.json: Lineage tracking (git SHA, config hash, checksums) ◄── LAST

    2025 Improvements:
        - Atomic writes prevent corrupted checkpoints
        - Manifest-last ensures all artifacts exist if manifest exists
        - Centralized metrics prevent MCC drift
        - Validation metrics computed on val_calib using centralized functions
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("PHASE 1: Baseline Training (Production-Grade 2025-12-30)")
    logger.info("=" * 80)

    # Read from Hydra config (ZERO hardcoding)
    data_root = cfg.data.data_root
    backbone_id = cfg.model.backbone_id
    num_classes = cfg.model.num_classes
    batch_size = cfg.data.dataloader.batch_size
    num_workers = cfg.data.dataloader.num_workers
    max_samples = getattr(cfg.data, "max_samples", None)
    max_epochs = cfg.training.epochs
    learning_rate = cfg.training.optimizer.lr
    weight_decay = cfg.training.optimizer.weight_decay
    num_gpus = cfg.hardware.num_gpus
    precision = "16-mixed" if cfg.training.mixed_precision.enabled else "32"

    # Early stopping config
    monitor_metric = cfg.training.early_stopping.monitor  # "val_select/acc"
    monitor_mode = cfg.training.early_stopping.mode  # "max"
    patience = cfg.training.early_stopping.patience

    logger.info(f"Data root: {data_root}")
    logger.info(f"Backbone: {backbone_id}")
    logger.info(f"Batch size: {batch_size}, Epochs: {max_epochs}")
    logger.info(f"Early stopping: {monitor_metric} ({monitor_mode}, patience={patience})")

    # Ensure phase1 directory exists
    artifacts.phase1_dir.mkdir(parents=True, exist_ok=True)

    # Create datamodule
    datamodule = NATIXDataModule(
        data_root=data_root,
        splits_json=str(artifacts.splits_json),
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )

    # Create model (config-driven, NOT hardcoded)
    model = DINOv3Classifier(
        backbone_name=backbone_id,
        num_classes=num_classes,
        freeze_backbone=cfg.model.freeze_backbone,
        head_type=cfg.model.head_type,
        dropout_rate=cfg.model.dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_ema=cfg.model.use_ema,
        ema_decay=cfg.checkpointing.ema_decay,
        use_multiview=cfg.model.use_multiview,
        multiview_aggregation=cfg.model.multiview_aggregation,
        multiview_topk=cfg.model.multiview_topk,
    )

    # CRITICAL: Load ExPLoRA checkpoint if requested
    if cfg.model.init_from_explora and artifacts.explora_checkpoint.exists():
        logger.info(f"Loading ExPLoRA checkpoint: {artifacts.explora_checkpoint}")
        explora_state = torch.load(artifacts.explora_checkpoint, map_location="cpu")
        model.net["backbone"].model.load_state_dict(explora_state, strict=False)
        logger.info("✅ Loaded ExPLoRA-adapted backbone (Phase 4 → Phase 1)")
    elif cfg.model.init_from_explora:
        logger.warning(
            f"ExPLoRA requested but checkpoint not found: {artifacts.explora_checkpoint}"
        )

    logger.info(f"Model trainable params: {model.net['head'].num_parameters:,}")

    # Callbacks (config-driven)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(artifacts.phase1_dir),
            filename="best_model",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=patience,
            verbose=True,
        ),
        ValCalibArtifactSaver(
            logits_path=artifacts.val_calib_logits,
            labels_path=artifacts.val_calib_labels,
            dataloader_idx=1,  # val_calib is index 1
        ),
    ]

    # Trainer (config-driven)
    # Lightning expects devices>=1. For CPU runs we use devices=1.
    accelerator = "cpu" if num_gpus == 0 else "gpu"
    devices = 1 if num_gpus == 0 else num_gpus
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        default_root_dir=str(artifacts.phase1_dir),
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Training complete!")

    # ========== PRODUCTION-GRADE OUTPUT HANDLING (2025) ==========

    # 1. Copy best checkpoint with ATOMIC WRITE
    best_ckpt_path = callbacks[0].best_model_path
    if best_ckpt_path and Path(best_ckpt_path).exists():
        logger.info(f"Saving best checkpoint atomically...")
        # Load checkpoint and re-save atomically
        ckpt_data = torch.load(best_ckpt_path, map_location="cpu")
        ckpt_checksum = write_checkpoint_atomic(artifacts.phase1_checkpoint, ckpt_data)
        logger.info(
            f"✅ Checkpoint saved: {artifacts.phase1_checkpoint} "
            f"(SHA256: {ckpt_checksum[:12]}...)"
        )
    else:
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

    # 2. Copy metrics.csv (Lightning CSVLogger output)
    if trainer.logger and hasattr(trainer.logger, "log_dir"):
        lightning_metrics_csv = Path(trainer.logger.log_dir) / "metrics.csv"
        if lightning_metrics_csv.exists():
            shutil.copy2(lightning_metrics_csv, artifacts.metrics_csv)
            logger.info(f"✅ Metrics saved: {artifacts.metrics_csv}")
        else:
            logger.warning(f"Lightning metrics.csv not found at {lightning_metrics_csv}")
    else:
        logger.warning("No logger with log_dir found")

    # 3. Compute validation metrics on val_calib using CENTRALIZED FUNCTIONS
    logger.info("Computing validation metrics on val_calib...")
    val_metrics = compute_validation_metrics(
        logits_path=artifacts.val_calib_logits,
        labels_path=artifacts.val_calib_labels,
    )
    logger.info(
        f"Val Calib Metrics: MCC={val_metrics['mcc']:.3f}, "
        f"Acc={val_metrics['accuracy']:.3f}, FNR={val_metrics['fnr']:.3f}"
    )

    # 4. Save config (atomic JSON write)
    config_data = {
        "backbone_id": backbone_id,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "monitor_metric": monitor_metric,
        "best_checkpoint": str(artifacts.phase1_checkpoint),
        "init_from_explora": cfg.model.init_from_explora,
        # Add validation metrics
        "val_calib_mcc": val_metrics["mcc"],
        "val_calib_accuracy": val_metrics["accuracy"],
        "val_calib_fnr": val_metrics["fnr"],
    }
    config_checksum = write_json_atomic(artifacts.config_json, config_data)
    logger.info(f"✅ Config saved: {artifacts.config_json} (SHA256: {config_checksum[:12]}...)")

    # 5. Create and save MANIFEST (LAST STEP - ensures all artifacts exist)
    duration_seconds = time.time() - start_time
    logger.info("Creating manifest (lineage tracking)...")

    manifest = create_step_manifest(
        step_name="phase1_baseline",
        input_paths=[artifacts.splits_json],
        output_paths=[
            artifacts.phase1_checkpoint,
            artifacts.val_calib_logits,
            artifacts.val_calib_labels,
            artifacts.metrics_csv,
            artifacts.config_json,
        ],
        output_dir=artifacts.output_dir,
        metrics=val_metrics,
        duration_seconds=duration_seconds,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    manifest_checksum = manifest.save(artifacts.phase1_dir / "manifest.json")
    logger.info(
        f"✅ Manifest saved: {artifacts.phase1_dir / 'manifest.json'} "
        f"(SHA256: {manifest_checksum[:12]}...)"
    )

    # Summary
    logger.info("=" * 80)
    logger.info("✅ Phase 1 Complete (Production-Grade)")
    logger.info(f"Duration: {duration_seconds / 60:.1f} minutes")
    logger.info(f"Checkpoint: {artifacts.phase1_checkpoint}")
    logger.info(f"Val calib logits: {artifacts.val_calib_logits}")
    logger.info(f"Val calib labels: {artifacts.val_calib_labels}")
    logger.info(f"Metrics CSV: {artifacts.metrics_csv}")
    logger.info(f"Config JSON: {artifacts.config_json}")
    logger.info(f"Manifest: {artifacts.phase1_dir / 'manifest.json'}")
    logger.info(f"Val Calib MCC: {val_metrics['mcc']:.3f}")
    logger.info("=" * 80)


def compute_validation_metrics(
    logits_path: Path,
    labels_path: Path,
) -> Dict[str, float]:
    """
    Compute validation metrics using centralized functions

    Args:
        logits_path: Path to logits tensor (.pt file)
        labels_path: Path to labels tensor (.pt file)

    Returns:
        Dict with metrics: mcc, accuracy, precision, recall, f1, fnr, fpr

    Why centralized:
        - Prevents metric drift across phases
        - Ensures all phases compute MCC identically
        - Single source of truth
    """
    # Load artifacts
    logits = torch.load(logits_path)
    labels = torch.load(labels_path)

    # Get predictions
    if len(logits.shape) == 2:
        # Two-class logits: take argmax
        preds = torch.argmax(logits, dim=1)
    else:
        # Already predictions
        preds = logits

    # Compute all metrics using CENTRALIZED FUNCTIONS
    metrics = compute_all_metrics(labels.cpu().numpy(), preds.cpu().numpy())

    return metrics
