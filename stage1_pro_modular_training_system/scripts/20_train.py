"""
Unified training script supporting ALL phases and ALL modes

Single entry point with --phase {1..6} and --exit_policy {softmax,gate,scrc} arguments.
Routes to appropriate training mode based on phase and mode.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Stage1ProConfig
from training.trainer import Stage1ProTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Stage-1 Pro Modular Training System - Unified Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["extract_features", "train_cached", "train"],
                       default="train", help="Training mode")
    
    # Phase and exit policy
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
                       help="Phase number (1-6)")
    parser.add_argument("--exit_policy", type=str, choices=["softmax", "gate", "scrc"],
                       default="softmax", help="Exit policy")
    
    # Config file (optional - can also use individual args)
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    # All baseline arguments (preserve from train_stage1_head.py)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--train_image_dir", type=str)
    parser.add_argument("--train_labels_file", type=str)
    parser.add_argument("--val_image_dir", type=str)
    parser.add_argument("--val_labels_file", type=str)
    parser.add_argument("--cached_features_dir", type=str)
    parser.add_argument("--use_extra_roadwork", action="store_true")
    parser.add_argument("--roadwork_iccv_dir", type=str)
    parser.add_argument("--roadwork_extra_dir", type=str)
    parser.add_argument("--max_batch_size", type=int)
    parser.add_argument("--fallback_batch_size", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--lr_head", type=float)
    parser.add_argument("--lr_backbone", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float)
    parser.add_argument("--early_stop_patience", type=int)
    parser.add_argument("--legacy_exit_threshold_for_logging", type=float)
    parser.add_argument("--resume_checkpoint", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--log_file", type=str)
    
    # Phase-specific arguments
    parser.add_argument("--target_fnr_exit", type=float, default=0.02,
                       help="Target FNR on exited samples (single constraint)")
    parser.add_argument("--val_select_ratio", type=float, default=0.33)
    parser.add_argument("--val_calib_ratio", type=float, default=0.33)
    parser.add_argument("--val_test_ratio", type=float, default=0.34)
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = Stage1ProConfig.load(args.config)
    else:
        # Create config from args
        config = Stage1ProConfig()
        # Update with provided args (skip phase - it's a read-only property)
        for key, value in vars(args).items():
            if key == 'phase':
                continue  # Skip - phase is read-only property
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
    
    # Set exit policy
    config.exit_policy = args.exit_policy
    
    # Create splits if needed
    from data.splits import create_val_splits, load_splits, save_splits
    from data.datasets import NATIXDataset
    
    splits_path = Path(args.output_dir if hasattr(args, 'output_dir') else config.output_dir) / "splits.json"
    if not splits_path.exists():
        print(f"Creating splits...")
        # Load val dataset for split creation
        from data.loaders import create_data_loaders
        val_dataset = NATIXDataset(
            image_dir=config.val_image_dir,
            labels_file=config.val_labels_file,
            processor=None,  # Will load from backbone
            augment=False
        )
        splits = create_val_splits(
            val_dataset,
            val_select_ratio=config.val_select_ratio,
            val_calib_ratio=config.val_calib_ratio,
            val_test_ratio=config.val_test_ratio,
            seed=42
        )
        save_splits(splits, str(splits_path))
    else:
        print(f"Loading splits from {splits_path}...")
        splits = load_splits(str(splits_path))
    
    # Create dataloaders
    from data.loaders import create_data_loaders
    train_dataset = NATIXDataset(
        image_dir=config.train_image_dir,
        labels_file=config.train_labels_file,
        processor=None,  # Will load from backbone
        augment=True
    )
    val_dataset = NATIXDataset(
        image_dir=config.val_image_dir,
        labels_file=config.val_labels_file,
        processor=None,
        augment=False
    )
    train_loader, val_select_loader, val_calib_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        splits,
        config
    )
    
    # Create backbone
    from model.backbone import DINOv3Backbone
    backbone = DINOv3Backbone(config.model_path)
    backbone.load(freeze=(config.phase == 1))
    backbone = backbone.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create head based on phase
    if config.phase == 1:
        from model.head import Stage1Head
        head = Stage1Head(num_classes=2, hidden_size=768, phase=config.phase)
    else:
        from model.gate_head import GateHead
        head = GateHead(backbone_dim=768, num_classes=2, gate_hidden_dim=128)
    head = head.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create trainer with ALL required components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Stage1ProTrainer(
        model=head,
        backbone=backbone,
        train_loader=train_loader,
        val_select_loader=val_select_loader,
        val_calib_loader=val_calib_loader,
        config=config,
        device=device
    )
    
    # Route to appropriate mode
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
