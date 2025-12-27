#!/usr/bin/env python3
"""
Phase 4.1.3: Supervised Fine-Tuning with ExPLoRA-Adapted Backbone

Train classifier+gate+aux on ExPLoRA-adapted backbone (Phase 4.1.3).

Usage:
    python scripts/45_train_supervised_explora.py --config config.yaml --backbone output_explora/backbone_explora.pth

This is downstream supervised training AFTER ExPLoRA pretraining.
Uses REAL HuggingFace PEFT library (LoRA/DoRA) for efficiency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime

from config import Stage1ProConfig
from model.peft_integration import apply_lora_to_backbone, apply_dora_to_backbone, count_peft_parameters
from model.gate_head import GateHead
from training.peft_real_trainer import RealPEFTTrainer, create_real_peft_trainer
from data.datasets import MultiRoadworkDataset
from data.splits import load_splits
from data.loaders import create_data_loaders
from utils.checkpointing import save_checkpoint
from utils.reproducibility import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4.1.3: Supervised Fine-Tuning with ExPLoRA-Adapted Backbone")
    
    # Config
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    
    # Backbone (ExPLoRA-adapted)
    parser.add_argument("--backbone_checkpoint", type=str, required=True,
                       help="Path to ExPLoRA-adapted backbone checkpoint")
    
    # PEFT (additional PEFT on top of ExPLoRA)
    parser.add_argument("--peft_type", type=str, default="lora", choices=["lora", "dora", "none"],
                       help="Additional PEFT type (default: lora)")
    parser.add_argument("--peft_r", type=int, default=8,
                       help="Additional PEFT rank (default: 8, smaller for ExPLoRA)")
    parser.add_argument("--peft_alpha", type=int, default=16,
                       help="Additional PEFT alpha (default: 16)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output_supervised_explora",
                       help="Output directory (default: output_supervised_explora)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_config(config_path: str) -> Stage1ProConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Stage1ProConfig(**config_dict)


def load_explora_backbone(checkpoint_path: str, device: str = "cuda"):
    """
    Load ExPLoRA-adapted backbone from checkpoint.
    
    Args:
        checkpoint_path: Path to ExPLoRA checkpoint
        device: Device
    
    Returns:
        Backbone with ExPLoRA-adapted weights
    """
    print(f"\n📦 Loading ExPLoRA-adapted backbone from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load backbone state dict
    backbone_state_dict = checkpoint['backbone_state_dict']
    
    # Get ExPLoRA config
    explora_config = checkpoint.get('explora_config', {})
    
    print(f"✅ ExPLoRA config:")
    print(f"   Unfreeze blocks: {explora_config.get('unfreeze_blocks', 'unknown')}")
    print(f"   PEFT rank: {explora_config.get('peft_r', 'unknown')}")
    print(f"   PEFT alpha: {explora_config.get('peft_alpha', 'unknown')}")
    print(f"   Mask ratio: {explora_config.get('mask_ratio', 'unknown')}")
    
    # Create fresh backbone from HuggingFace
    backbone_path = explora_config.get('backbone_path', 'facebook/dinov3-vith14')
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(backbone_path)
    backbone.load_state_dict(backbone_state_dict)
    backbone = backbone.to(device)
    
    print(f"✅ ExPLoRA-adapted backbone loaded successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return backbone


def create_gate_model(num_classes: int, hidden_size: int, device: str = "cuda"):
    """Create gate head model (3-head architecture)."""
    print(f"\n🧠 Creating GateHead model (3-head architecture)")
    
    model = GateHead(
        backbone_dim=hidden_size,
        num_classes=num_classes,
        gate_hidden_dim=128,
        device=device,
        verbose=True
    )
    
    print(f"✅ Model created successfully")
    
    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {model_params:,} ({model_params/1e6:.2f}M)")
    
    return model


def apply_additional_peft(
    backbone: torch.nn.Module,
    peft_type: str,
    peft_r: int,
    peft_alpha: int,
    verbose: bool = True
):
    """
    Apply additional PEFT on top of ExPLoRA-adapted backbone.
    
    This is optional - ExPLoRA already has LoRA.
    Additional PEFT can further fine-tune specific layers.
    
    Args:
        backbone: ExPLoRA-adapted backbone
        peft_type: "lora" or "dora"
        peft_r: Rank
        peft_alpha: Alpha
        verbose: Print status
    
    Returns:
        PEFT-adapted backbone
    """
    if peft_type == "none":
        print(f"\n⚠️  No additional PEFT (using ExPLoRA-adapted backbone as-is)")
        return backbone
    
    print(f"\n🔄 Applying additional {peft_type.upper()} to ExPLoRA-adapted backbone...")
    
    if peft_type.lower() == "dora":
        adapted_backbone = apply_dora_to_backbone(backbone, r=peft_r, lora_alpha=peft_alpha)
    else:
        adapted_backbone = apply_lora_to_backbone(backbone, r=peft_r, lora_alpha=peft_alpha)
    
    # Count parameters
    peft_params = count_peft_parameters(adapted_backbone)
    
    if verbose:
        print(f"✅ Additional {peft_type.upper()} applied")
        print(f"   PEFT trainable: {peft_params['trainable']:,} ({peft_params['trainable']/1e6:.2f}M)")
        print(f"   Trainable ratio: {100*peft_params['trainable_ratio']:.2f}%")
    
    return adapted_backbone


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"PHASE 4.1.3: SUPERVISED FINE-TUNING WITH EXPLORA-ADAPTED BACKBONE")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"ExPLoRA Backbone: {args.backbone_checkpoint}")
    print(f"Additional PEFT: {args.peft_type.upper()}, r={args.peft_r}")
    print(f"Output Dir: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}\n")
    
    # Step 1: Load ExPLoRA-adapted backbone
    explora_backbone = load_explora_backbone(args.backbone_checkpoint, device)
    
    # Step 2: Create gate head model
    num_classes = config.num_classes
    hidden_size = explora_backbone.config.hidden_size
    model = create_gate_model(num_classes, hidden_size, device)
    
    # Step 3: Apply additional PEFT (optional)
    backbone_with_peft = apply_additional_peft(
        backbone=explora_backbone,
        peft_type=args.peft_type,
        peft_r=args.peft_r,
        peft_alpha=args.peft_alpha,
        verbose=True
    )
    
    # Step 4: Create data loaders
    print(f"\n📊 Creating data loaders...")
    train_loader, val_select_loader, val_calib_loader = create_data_loaders(
        config=config,
        split_type="4way",  # train/val_select/val_calib/val_test
        batch_size=config.max_batch_size,
        num_workers=4
    )
    print(f"✅ Data loaders created successfully")
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Val Select: {len(val_select_loader.dataset)} samples")
    print(f"   Val Calib: {len(val_calib_loader.dataset)} samples\n")
    
    # Step 5: Create REAL PEFT trainer
    print(f"\n🎯 Creating REAL PEFT trainer...")
    
    # Create base trainer configuration
    base_trainer_config = {
        "config": config,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "num_epochs": config.num_epochs,
        "early_stop_patience": config.early_stop_patience,
        "use_ema": config.use_ema,
        "ema_decay": config.ema_decay
    }
    
    # Create trainer
    trainer = RealPEFTTrainer(
        backbone=backbone_with_peft,
        model=model,
        train_loader=train_loader,
        val_select_loader=val_select_loader,
        val_calib_loader=val_calib_loader,
        config=base_trainer_config,
        device=device,
        verbose=True
    )
    
    # Print PEFT info
    from peft import PeftModel
    if isinstance(backbone_with_peft, PeftModel):
        peft_info = count_peft_parameters(backbone_with_peft)
        print(f"📊 PEFT Model Info:")
        print(f"   Trainable params: {peft_info['trainable']:,} ({peft_info['trainable']/1e6:.2f}M)")
        print(f"   Total params: {peft_info['total']:,} ({peft_info['total']/1e6:.2f}M)")
        print(f"   Trainable ratio: {100*peft_info['trainable_ratio']:.2f}%\n")
    
    # Step 6: Train
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")
    
    results = trainer.train()
    
    # Step 7: Save checkpoints
    print(f"\n💾 Saving checkpoints...")
    
    # Save adapter checkpoint (if PEFT)
    if isinstance(backbone_with_peft, PeftModel):
        adapter_dir = output_dir / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_adapters(str(adapter_dir))
        print(f"✅ Saved adapters to {adapter_dir}")
        
        # Export merged checkpoint
        merged_checkpoint_path = output_dir / "backbone_merged.pth"
        merged_backbone = trainer.merge_and_unload()
        torch.save({
            'backbone_state_dict': merged_backbone.state_dict(),
            'head_state_dict': model.state_dict(),
            'supervised_config': {
                'explora_backbone': args.backbone_checkpoint,
                'additional_peft_type': args.peft_type,
                'additional_peft_r': args.peft_r,
                'additional_peft_alpha': args.peft_alpha
            },
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        }, merged_checkpoint_path)
        print(f"✅ Saved merged checkpoint to {merged_checkpoint_path}")
    else:
        # Regular model
        checkpoint_path = output_dir / "checkpoint_final.pth"
        torch.save({
            'backbone_state_dict': backbone_with_peft.state_dict(),
            'head_state_dict': model.state_dict(),
            'config': base_trainer_config,
            'results': results,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        print(f"✅ Saved checkpoint to {checkpoint_path}")
    
    # Step 8: Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Metric: {results.get('best_metric', 0.0):.4f}")
    print(f"Best Epoch: {results.get('best_epoch', 0)}")
    print(f"\nArtifacts:")
    print(f"   Checkpoint: {output_dir / 'checkpoint_final.pth' if not isinstance(backbone_with_peft, PeftModel) else output_dir / 'backbone_merged.pth'}")
    print(f"{'='*80}\n")
    
    # Instructions for downstream evaluation
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Evaluate on val_test:")
    print(f"   python scripts/40_eval_selective.py \\")
    print(f"       --checkpoint {output_dir / ('backbone_merged.pth' if isinstance(backbone_with_peft, PeftModel) else 'checkpoint_final.pth')} \\")
    print(f"       --config {args.config}")
    print(f"\n2. Compare vs no-ExPLoRA baseline (Acceptance Test 4.1.4)")
    print()


if __name__ == "__main__":
    main()

