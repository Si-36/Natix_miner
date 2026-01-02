"""
Phase 4a: ExPLoRA SimCLR Training Step (2025 Best Practices)
=============================================================

ExPLoRA domain adaptation with SimCLR contrastive learning.

Contract:
- Inputs: NONE (truly unsupervised - uses all images from data_root)
- Outputs:
  - phase4_explora/explora_backbone.pth (merged backbone with LoRA)
  - phase4_explora/metrics.json (training metrics)
  - phase4_explora/manifest.json (lineage + checksums) ◄── LAST

Expected Gain: +6-8% MCC on downstream task

2025 Best Practices:
- SimCLR contrastive learning (unsupervised)
- Standard LoRA (NOT DoRA) for speed
- DDP all-gather with sync_grads=True
- Strong augmentations
- BF16 mixed precision
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from contracts.artifact_schema import ArtifactSchema
from data import NATIXDataModule

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.explora_module import ExPLoRAModule
from streetvision.io import (
    create_step_manifest,
    write_checkpoint_atomic,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


def run_phase4a_explora_simclr(
    artifacts: ArtifactSchema,
    cfg: DictConfig,
) -> None:
    """
    Run Phase 4a: ExPLoRA SimCLR Domain Adaptation
    
    Args:
        artifacts: Artifact schema (all file paths)
        cfg: Hydra configuration (config.phase4a)
    
    Outputs:
        - explora_backbone.pth: Merged backbone with LoRA adapters
        - metrics.json: Training metrics
        - manifest.json: Lineage tracking ◄── LAST
    
    2025 Best Practices:
        - SimCLR contrastive learning (unsupervised)
        - DDP all-gather for large negative pool
        - Standard LoRA (NOT DoRA) for speed
        - Strong augmentations
    """
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("PHASE 4a: ExPLoRA SimCLR Domain Adaptation (2025 Best Practices)")
    logger.info("=" * 80)
    
    # Read config
    phase4a_cfg = cfg.phase4a
    data_root = cfg.data.data_root
    batch_size = phase4a_cfg.training.batch_size
    num_workers = cfg.data.dataloader.num_workers
    num_epochs = phase4a_cfg.training.num_epochs
    num_gpus = phase4a_cfg.hardware.get("num_gpus", cfg.hardware.num_gpus)
    
    # BF16 auto-detection
    if phase4a_cfg.hardware.get("mixed_precision", True):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            precision = "bf16-mixed"
            logger.info("✅ Using BF16 mixed precision (GPU supports it)")
        else:
            precision = "32"
            logger.warning("BF16 requested but GPU doesn't support it, using FP32")
    else:
        precision = "32"
    
    # Gradient accumulation
    gradient_accumulation_steps = phase4a_cfg.training.get("gradient_accumulation_steps", 1)
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    logger.info(f"Effective batch: {batch_size} × {gradient_accumulation_steps} × {num_gpus} × 2 (views)")
    
    # Ensure phase4 directory exists
    artifacts.phase4_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datamodule (unsupervised - uses all images, no labels needed)
    # For SimCLR, we only need images (no labels)
    datamodule = NATIXDataModule(
        data_root=data_root,
        splits_json=str(artifacts.splits_json) if artifacts.splits_json.exists() else None,
        batch_size=batch_size,
        num_workers=num_workers,
        # Use train split only (or all images if no splits)
        # SimCLR is unsupervised, so labels don't matter
    )
    
    # Create ExPLoRA module
    logger.info("Creating ExPLoRA SimCLR module...")
    module = ExPLoRAModule(config=phase4a_cfg)
    
    # Apply torch.compile if enabled
    if phase4a_cfg.hardware.get("compile", False):
        compile_mode = phase4a_cfg.hardware.get("compile_mode", "reduce-overhead")
        compiler_stance = phase4a_cfg.hardware.get("compiler", {}).get("stance", None)
        
        if compiler_stance is not None:
            try:
                import torch.compiler
                torch.compiler.set_stance(compiler_stance)
                logger.info(f"✅ Set compiler stance: {compiler_stance}")
            except AttributeError:
                logger.warning("torch.compiler.set_stance not available (requires PyTorch 2.6+)")
        
        logger.info(f"🔥 Compiling model with torch.compile (mode={compile_mode})...")
        from models.module import create_model_with_compile
        module = create_model_with_compile(
            model=module,
            compile_enabled=True,
            compile_mode=compile_mode,
            compiler_stance=compiler_stance,
        )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(artifacts.phase4_dir),
            filename="explora-{epoch:03d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=phase4a_cfg.checkpoint.get("save_every_n_epochs", 10),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Trainer
    accelerator = "cpu" if num_gpus == 0 else "gpu"
    devices = "auto" if num_gpus == 0 else num_gpus
    strategy = "ddp" if num_gpus > 1 else "auto"
    
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=gradient_accumulation_steps,
        callbacks=callbacks,
        default_root_dir=str(artifacts.phase4_dir),
        log_every_n_steps=10,
        deterministic=False,  # Allow non-deterministic for speed
    )
    
    # Train
    logger.info("Starting ExPLoRA SimCLR training...")
    trainer.fit(module, datamodule=datamodule)
    logger.info("Training complete!")
    
    # ========== PRODUCTION-GRADE OUTPUT HANDLING ==========
    
    # 1. Merge LoRA weights into backbone and save checkpoint
    logger.info("Merging LoRA weights into backbone...")
    best_ckpt_path = callbacks[0].best_model_path
    if best_ckpt_path and Path(best_ckpt_path).exists():
        # Load best checkpoint
        ckpt_data = torch.load(best_ckpt_path, map_location="cpu")
        module.load_state_dict(ckpt_data["state_dict"])
        
        # Merge LoRA weights (if using PEFT)
        if hasattr(module.backbone, "merge_and_unload"):
            logger.info("Merging LoRA adapters into backbone...")
            merged_backbone = module.backbone.merge_and_unload()
        else:
            # Already merged or not using PEFT
            merged_backbone = module.backbone
        
        # Save merged backbone
        backbone_state = merged_backbone.state_dict()
        ckpt_checksum = write_checkpoint_atomic(artifacts.explora_checkpoint, backbone_state)
        logger.info(
            f"✅ Merged backbone saved: {artifacts.explora_checkpoint} "
            f"(SHA256: {ckpt_checksum[:12]}...)"
        )
    else:
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")
    
    # 2. Save metrics
    if trainer.logger and hasattr(trainer.logger, "log_dir"):
        lightning_metrics_csv = Path(trainer.logger.log_dir) / "metrics.csv"
        if lightning_metrics_csv.exists():
            import shutil
            shutil.copy2(lightning_metrics_csv, artifacts.phase4_dir / "metrics.csv")
            logger.info(f"✅ Metrics saved: {artifacts.phase4_dir / 'metrics.csv'}")
    
    # Extract final metrics
    metrics = {
        "final_train_loss": float(trainer.callback_metrics.get("train_loss", 0.0)),
        "num_epochs": num_epochs,
        "effective_batch_size": batch_size * gradient_accumulation_steps * max(num_gpus, 1) * 2,
    }
    
    metrics_checksum = write_json_atomic(artifacts.explora_metrics_json, metrics)
    logger.info(f"✅ Metrics saved: {artifacts.explora_metrics_json}")
    
    # 3. Create manifest
    duration_seconds = time.time() - start_time
    logger.info("Creating manifest (lineage tracking)...")
    
    manifest = create_step_manifest(
        step_name="phase4a_explora_simclr",
        input_paths=[],  # No inputs (unsupervised)
        output_paths=[
            artifacts.explora_checkpoint,
            artifacts.explora_metrics_json,
        ],
        output_dir=artifacts.output_dir,
        metrics=metrics,
        duration_seconds=duration_seconds,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    manifest_checksum = manifest.save(artifacts.phase4_dir / "manifest.json")
    logger.info(
        f"✅ Manifest saved: {artifacts.phase4_dir / 'manifest.json'} "
        f"(SHA256: {manifest_checksum[:12]}...)"
    )
    
    logger.info("=" * 80)
    logger.info("✅ PHASE 4a COMPLETE: ExPLoRA SimCLR Domain Adaptation")
    logger.info(f"   Duration: {duration_seconds / 60:.1f} minutes")
    logger.info(f"   Checkpoint: {artifacts.explora_checkpoint}")
    logger.info("=" * 80)

