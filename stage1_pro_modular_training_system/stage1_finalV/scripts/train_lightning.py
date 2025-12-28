#!/usr/bin/env python3
"""
🔥 **PyTorch Lightning 2.4 CLI (2025 Best Practices)**
Complete production-grade training script
"""

import sys
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main training function using Lightning 2.4 CLI"""
    
    # Create CLI (2025 best practice)
    cli = LightningCLI(
        run=True,  # Run training
        save_config_kwargs=True,  # Save config
        parser_kwargs={
            "parser_mode": "omegaconf",  # Use OmegaConf for parsing
        "config_path": "configs/hydra",
        "config_name": "config",
        "version_base": "1.0",
            "version_base": "1.0",
            "version_base": "1.0",
        },
    )
    
    print(f"\n{'='*80}")
    print(f"🔥 Training Started")
    print(f"  Phase: {cli.config.get('phase', 1)}")
    print(f"  Epochs: {cli.config.get('num_epochs', 50)}")
    print(f"  Batch size: {cli.config.get('batch_size', 32)}")
    print(f"  Learning rate: {cli.config.get('learning_rate', 1e-4)}")
    print(f"  Device: {cli.trainer.accelerator}")
    print(f"  Precision: {cli.trainer.precision}")
    print(f"{'='*80}\n")
    
    # Train
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"  Best model: {cli.trainer.checkpoint_callback.best_model_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
