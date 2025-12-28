#!/usr/bin/env python3
"""
Training CLI - Hydra-Driven Command-Line Interface

Usage:
    # Run all phases
    python scripts/train_cli.py pipeline.phases=[phase1,phase2,phase6]

    # Run specific phases with config overrides
    python scripts/train_cli.py pipeline.phases=[phase1] model.lr=0.001 data.batch_size=32

    # Dry run
    python scripts/train_cli.py pipeline.phases=[all] pipeline.dry_run=true

Latest 2025-2026 practices:
- Python 3.14+ with Hydra configuration
- Proper logging
- Type-safe configs
"""

import sys
import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path (temporary until proper package install)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contracts.artifact_schema import create_artifact_schema
from pipeline.phase_spec import PhaseType
from pipeline.dag_engine import DAGEngine


def resolve_phases(phase_names: List[str]) -> List[PhaseType]:
    """Resolve phase names to PhaseType enum"""
    if "all" in phase_names:
        return [
            PhaseType.PHASE1_BASELINE,
            PhaseType.PHASE2_THRESHOLD,
            PhaseType.PHASE3_GATE,
            PhaseType.PHASE4_EXPLORA,
            PhaseType.PHASE5_SCRC,
            PhaseType.PHASE6_BUNDLE,
        ]

    phase_map = {
        "phase1": PhaseType.PHASE1_BASELINE,
        "phase2": PhaseType.PHASE2_THRESHOLD,
        "phase3": PhaseType.PHASE3_GATE,
        "phase4": PhaseType.PHASE4_EXPLORA,
        "phase5": PhaseType.PHASE5_SCRC,
        "phase6": PhaseType.PHASE6_BUNDLE,
    }

    return [phase_map[name] for name in phase_names if name in phase_map]


def register_phase_executors(engine: DAGEngine) -> None:
    """
    Register executor functions for all phases

    CRITICAL FIX: phase1_executor is now REAL (not stub)!
    """

    def phase1_executor(artifacts):
        """
        Phase 1: Baseline Training

        Train DINOv3 classifier and save:
        - Best checkpoint
        - Val calib logits/labels
        - Metrics
        """
        import numpy as np
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from data import NATIXDataModule
        from models import DINOv3Classifier

        logger.info("=" * 80)
        logger.info("PHASE 1: Baseline Training")
        logger.info("=" * 80)

        # TODO: Get these from Hydra config (for now hardcoded)
        DATA_ROOT = "/data/natix"  # Update to your data path
        DINOV3_MODEL = "facebook/dinov3-vith16-pretrain-lvd1689m"  # Or local path
        BATCH_SIZE = 32
        MAX_EPOCHS = 50
        NUM_WORKERS = 4

        logger.info(f"Data root: {DATA_ROOT}")
        logger.info(f"DINOv3 model: {DINOV3_MODEL}")

        # Create datamodule
        datamodule = NATIXDataModule(
            data_root=DATA_ROOT,
            splits_json=str(artifacts.splits_json),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        # Create model
        model = DINOv3Classifier(
            backbone_name="vit_huge",
            num_classes=13,
            pretrained_path=DINOV3_MODEL,
            freeze_backbone=True,  # Freeze DINOv3, only train head
            head_type="linear",
            dropout_rate=0.3,
            learning_rate=1e-4,
            weight_decay=0.01,
            use_ema=True,
        )

        logger.info(f"Model: {model.num_trainable_parameters:,} trainable params")

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=str(artifacts.phase1_dir),
                filename="best_model",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
            ),
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=10,
                verbose=True,
            ),
        ]

        # Trainer
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            callbacks=callbacks,
            default_root_dir=str(artifacts.phase1_dir),
            log_every_n_steps=10,
            deterministic=True,
        )

        # Train
        logger.info("Starting training...")
        trainer.fit(model, datamodule=datamodule)

        # Save val_calib logits/labels
        if hasattr(model, "latest_val_calib_logits"):
            logger.info("Saving val_calib logits/labels...")
            np.save(artifacts.val_calib_logits, model.latest_val_calib_logits)
            np.save(artifacts.val_calib_labels, model.latest_val_calib_labels)
            logger.info(f"Saved to {artifacts.val_calib_logits}")
        else:
            logger.warning("No val_calib logits found! Check validation loop.")

        # Save config
        import json
        config = {
            "backbone": "vit_huge",
            "num_classes": 13,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "best_checkpoint": str(callbacks[0].best_model_path),
        }
        with open(artifacts.config_json, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"✅ Phase 1 complete!")
        logger.info(f"Best checkpoint: {callbacks[0].best_model_path}")
        logger.info("=" * 80)

    def phase2_executor(artifacts):
        logger.warning("Phase 2 executor not implemented yet")
        raise NotImplementedError("TODO: Implement Phase 2 threshold sweep")

    def phase3_executor(artifacts):
        logger.warning("Phase 3 executor not implemented yet")
        raise NotImplementedError("TODO: Implement Phase 3 gate training")

    def phase4_executor(artifacts):
        """
        Phase 4: ExPLoRA (Extended Pretraining with LoRA)

        Domain adaptation: Fine-tune DINOv3 with LoRA adapters on roadwork images.
        Expected gain: +8.2% accuracy (69% → 77.2%)

        Saves:
        - Merged backbone checkpoint (explora_backbone.pth)
        - LoRA adapter weights (explora_lora.pth)
        - Metrics JSON (metrics.json)
        """
        import json
        import numpy as np
        import torch
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from transformers import AutoModel
        from data import NATIXDataModule
        from models.explora_module import ExPLoRAModule
        from models.explora_config import ExPLoRAConfig

        logger.info("=" * 80)
        logger.info("PHASE 4: ExPLoRA (Extended Pretraining with LoRA)")
        logger.info("=" * 80)

        # Configuration (TODO: Get from Hydra config)
        DATA_ROOT = "/data/natix"  # Update to your data path
        DINOV3_MODEL = "facebook/dinov2-giant"  # DINOv2 Giant (ViT-g/14)
        BATCH_SIZE = 16  # Smaller batch for 4 GPUs (total effective: 64)
        MAX_EPOCHS = 100  # Extended pretraining
        NUM_WORKERS = 4
        NUM_GPUS = 4  # 4 GPUs for distributed training
        LORA_RANK = 16
        LORA_ALPHA = 32

        logger.info(f"Data root: {DATA_ROOT}")
        logger.info(f"DINOv3 model: {DINOV3_MODEL}")
        logger.info(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
        logger.info(f"Training on {NUM_GPUS} GPUs for {MAX_EPOCHS} epochs")

        # Load frozen DINOv3 backbone
        logger.info("Loading DINOv3 backbone...")
        backbone = AutoModel.from_pretrained(
            DINOV3_MODEL,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        )
        backbone.requires_grad_(False)  # Freeze completely

        # Create LoRA configuration
        lora_config = ExPLoRAConfig(
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj"],
            use_rslora=True,  # Rank-Stabilized LoRA
        )

        # Create ExPLoRA module
        model = ExPLoRAModule(
            backbone=backbone,
            num_classes=13,
            lora_config=lora_config,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_epochs=2,
            max_epochs=MAX_EPOCHS,
            use_gradient_checkpointing=True,
        )

        # Print trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model parameters:\n"
            f"  Total:     {total_params:,}\n"
            f"  Trainable: {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)"
        )

        # Create datamodule
        datamodule = NATIXDataModule(
            data_root=DATA_ROOT,
            splits_json=str(artifacts.splits_json),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            # ExPLoRA only uses TRAIN + VAL_SELECT (no val_calib)
            use_val_calib=False,
        )

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=str(artifacts.phase4_dir),
                filename="best_model",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=15,  # More patience for long training
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Trainer (4 GPUs, DDP strategy)
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu",
            devices=NUM_GPUS,
            strategy="ddp",  # Distributed Data Parallel
            precision="bf16-mixed",  # bfloat16 mixed precision
            callbacks=callbacks,
            default_root_dir=str(artifacts.phase4_dir),
            log_every_n_steps=10,
            deterministic=True,
            gradient_clip_val=1.0,  # Gradient clipping for stability
        )

        # Train
        logger.info("Starting ExPLoRA training...")
        logger.info(f"Expected time: ~24 hours on 4× A100 GPUs")
        trainer.fit(model, datamodule=datamodule)

        # Merge LoRA adapters and save (only on rank 0)
        if trainer.global_rank == 0:
            logger.info("Merging LoRA adapters and saving checkpoints...")

            # Load best checkpoint
            best_ckpt_path = callbacks[0].best_model_path
            logger.info(f"Loading best checkpoint: {best_ckpt_path}")
            checkpoint = torch.load(best_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

            # Merge and save
            model.merge_and_save(
                output_path=artifacts.explora_checkpoint,
                save_lora_separately=True,
                lora_path=artifacts.explora_lora_checkpoint,
            )

            # Save metrics
            metrics = model.get_metrics_summary()
            metrics["config"] = {
                "backbone": DINOV3_MODEL,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "num_gpus": NUM_GPUS,
                "best_checkpoint": str(best_ckpt_path),
            }

            with open(artifacts.explora_metrics_json, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"✅ Phase 4 complete!")
            logger.info(f"Merged backbone: {artifacts.explora_checkpoint}")
            logger.info(f"LoRA adapters:   {artifacts.explora_lora_checkpoint}")
            logger.info(f"Metrics:         {artifacts.explora_metrics_json}")
            logger.info("=" * 80)

    def phase5_executor(artifacts):
        logger.warning("Phase 5 executor not implemented yet")
        raise NotImplementedError("TODO: Implement SCRC calibration")

    def phase6_executor(artifacts):
        logger.warning("Phase 6 executor not implemented yet")
        raise NotImplementedError("TODO: Implement bundle export")

    engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
    engine.register_executor(PhaseType.PHASE2_THRESHOLD, phase2_executor)
    engine.register_executor(PhaseType.PHASE3_GATE, phase3_executor)
    engine.register_executor(PhaseType.PHASE4_EXPLORA, phase4_executor)
    engine.register_executor(PhaseType.PHASE5_SCRC, phase5_executor)
    engine.register_executor(PhaseType.PHASE6_BUNDLE, phase6_executor)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main Hydra entry point

    Args:
        cfg: Hydra configuration (automatically loaded from configs/)
    """
    logger.info("=" * 80)
    logger.info("StreetVision Training Pipeline - Stage 1 Ultimate")
    logger.info("Python 3.14+ | Hydra Configuration")
    logger.info("=" * 80)

    # Print resolved config (helpful for debugging)
    logger.info("\nResolved Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Create artifact schema
    output_dir = cfg.get("output_dir", "outputs")
    artifacts = create_artifact_schema(output_dir)
    artifacts.ensure_dirs()  # Create all directories

    # Create DAG engine
    engine = DAGEngine(artifacts=artifacts)

    # Register phase executors
    register_phase_executors(engine)

    # Get phases to run
    phases_config = cfg.pipeline.get("phases", [])
    if not phases_config:
        logger.error("No phases specified! Use: pipeline.phases=[phase1,phase2]")
        return

    phases_to_run = resolve_phases(phases_config)

    if not phases_to_run:
        logger.error(f"No valid phases found in: {phases_config}")
        return

    # Dry run mode
    if cfg.pipeline.get("dry_run", False):
        logger.info("\n🔍 DRY RUN MODE - No execution, just showing plan\n")
        execution_order = engine.registry.get_execution_order(phases_to_run)

        logger.info(f"Phases to run: {len(execution_order)}")
        for i, phase_type in enumerate(execution_order, 1):
            phase = engine.registry.get_phase(phase_type)
            logger.info(f"\n{i}. {phase.name}")
            logger.info(f"   Dependencies: {[d.value for d in phase.dependencies] or 'None'}")
            logger.info(f"   Estimated time: {phase.resources.estimated_time_minutes:.0f} min")
        return

    # Run pipeline
    try:
        engine.run(phases_to_run)

        # Save state if requested
        if cfg.pipeline.get("save_state", False):
            state_path = Path(cfg.pipeline.get("state_path", "outputs/pipeline_state.json"))
            engine.save_state(state_path)
            logger.info(f"Saved pipeline state to: {state_path}")

        logger.info("\n✅ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)

        # Save state on failure
        if cfg.pipeline.get("save_state", True):
            state_path = Path("outputs/pipeline_state_failed.json")
            engine.save_state(state_path)
            logger.info(f"Saved failed state to: {state_path}")

        raise


if __name__ == "__main__":
    main()
