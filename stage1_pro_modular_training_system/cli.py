"""
CLI interface preserving ALL arguments from train_stage1_head.py

Plus phase-specific options for 2025 features.
"""

import argparse
import os
import sys
from pathlib import Path

from config import Stage1ProConfig
from training.trainer import Stage1ProTrainer


def create_parser():
    """Create comprehensive argument parser"""
    parser = argparse.ArgumentParser(
        description="Stage-1 Pro Modular Training System (2025 Production-Grade)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Baseline training
  python -m stage1_pro_modular_training_system.cli --mode train --phase 1 --exit_policy softmax
  
  # Extract features
  python -m stage1_pro_modular_training_system.cli --mode extract_features
  
  # Train on cached features (10x faster)
  python -m stage1_pro_modular_training_system.cli --mode train_cached --phase 1
"""
    )
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["extract_features", "train_cached", "train"],
                       default="train", help="Training mode")
    
    # Phase and exit policy
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
                       help="Phase number (1-6)")
    parser.add_argument("--exit_policy", type=str, choices=["softmax", "gate", "scrc"],
                       default="softmax", help="Exit policy")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    # All baseline arguments (preserve from train_stage1_head.py)
    parser.add_argument("--model_path", type=str,
                       default="models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m")
    parser.add_argument("--train_image_dir", type=str, default="data/natix_official/train")
    parser.add_argument("--train_labels_file", type=str, default="data/natix_official/train_labels.csv")
    parser.add_argument("--val_image_dir", type=str, default="data/natix_official/val")
    parser.add_argument("--val_labels_file", type=str, default="data/natix_official/val_labels.csv")
    parser.add_argument("--cached_features_dir", type=str, default="cached_features")
    parser.add_argument("--use_extra_roadwork", action="store_true")
    parser.add_argument("--roadwork_iccv_dir", type=str, default="data/roadwork_iccv")
    parser.add_argument("--roadwork_extra_dir", type=str, default="data/roadwork_extra")
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument("--fallback_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--legacy_exit_threshold_for_logging", type=float, default=0.88,
                       help="Monitoring only, NOT used for inference")
    parser.add_argument("--resume_checkpoint", type=str)
    parser.add_argument("--output_dir", type=str, default="models/stage1_dinov3")
    parser.add_argument("--log_file", type=str, default="training.log")
    
    # Phase 1 arguments
    parser.add_argument("--target_fnr_exit", type=float, default=0.02,
                       help="Target FNR on exited samples (single constraint, maximize coverage)")
    parser.add_argument("--val_select_ratio", type=float, default=0.33)
    parser.add_argument("--val_calib_ratio", type=float, default=0.33)
    parser.add_argument("--val_test_ratio", type=float, default=0.34)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_confidence", type=float, default=0.95)
    
    # Phase 3+ arguments (disabled in Phase 1)
    parser.add_argument("--gate_loss_weight", type=float, default=0.0)
    parser.add_argument("--aux_weight", type=float, default=0.5)
    
    # Phase 4+ arguments
    parser.add_argument("--peft_type", type=str, choices=["none", "doran", "dora", "lora"],
                       default="none")
    parser.add_argument("--peft_r", type=int, default=16)
    parser.add_argument("--peft_blocks", type=int, default=6)
    
    # Phase 5+ arguments
    parser.add_argument("--optimizer", type=str, choices=["adamw", "fsam"], default="adamw")
    
    # Phase 6+ arguments
    parser.add_argument("--use_dirichlet", action="store_true")
    parser.add_argument("--calibration_iters", type=int, default=300)
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create config from args
    if args.config and os.path.exists(args.config):
        config = Stage1ProConfig.load(args.config)
    else:
        config = Stage1ProConfig()
        # Update with provided args (skip phase - it's a read-only property)
        for key, value in vars(args).items():
            if key == 'phase':
                continue  # Skip - phase is read-only property
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
    
    # Set exit policy
    config.exit_policy = args.exit_policy
    
    # Create trainer and route to mode
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Note: Stage1ProTrainer doesn't take phase arg - config.phase is read-only property
    trainer = Stage1ProTrainer(config, device=device)
    
    if args.mode == "extract_features":
        trainer.extract_features()
    elif args.mode == "train_cached":
        trainer.train_cached()
    elif args.mode == "train":
        trainer.train()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
