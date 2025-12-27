#!/usr/bin/env python3
"""
Phase 4.1: ExPLoRA Pretraining Script (Dec 2025 Best Practice)

Extended Parameter-Efficient Low-Rank Adaptation for Domain Adaptation:
- Continue unsupervised pretraining on target-domain images (NATIX + synthetic)
- Unfreeze only 1-2 transformer blocks
- Apply LoRA to remaining blocks
- MAE-style masked image modeling objective

Usage:
    python scripts/44_explora_pretrain.py --config config.yaml --backbone facebook/dinov3-vith14

Acceptance Test 4.1.4:
- Show downstream improvement vs no-ExPLoRA on val_test (accuracy, MCC)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime

from transformers import AutoModel, AutoImageProcessor
from model.peft_integration import apply_lora_to_backbone, count_peft_parameters
from domain_adaptation.explora import ExPLoRATrainer
from domain_adaptation.data import UnlabeledRoadDataset
from utils.reproducibility import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4.1: ExPLoRA Pretraining")
    
    # Config
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    
    # Backbone
    parser.add_argument("--backbone_path", type=str, default="facebook/dinov3-vith14",
                       help="HuggingFace model path (default: facebook/dinov3-vith14)")
    
    # ExPLoRA settings
    parser.add_argument("--unfreeze_blocks", type=int, default=2,
                       help="Number of last transformer blocks to unfreeze (default: 2)")
    parser.add_argument("--peft_r", type=int, default=16,
                       help="PEFT rank for LoRA (default: 16)")
    parser.add_argument("--peft_alpha", type=int, default=32,
                       help="PEFT alpha for LoRA (default: 32)")
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                       help="Mask ratio for MAE (default: 0.75)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of pretraining epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    
    # Data
    parser.add_argument("--natix_extras_dir", type=str, default=None,
                       help="Path to NATIX extras directory")
    parser.add_argument("--sdxl_synthetics_dir", type=str, default=None,
                       help="Path to SDXL synthetics directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output_explora",
                       help="Output directory (default: output_explora)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_backbone(model_path: str, device: str = "cuda"):
    """Create DINOv3 backbone."""
    print(f"\n📦 Creating DINOv3 backbone from {model_path}")
    
    backbone = AutoModel.from_pretrained(model_path)
    backbone = backbone.to(device)
    
    print(f"✅ Backbone created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    return backbone


def create_unlabeled_dataset(
    natix_extras_dir: str,
    sdxl_synthetics_dir: str,
    augment: bool = True
) -> UnlabeledRoadDataset:
    """
    Create unlabeled dataset (Phase 4.1.1).
    
    Args:
        natix_extras_dir: Path to NATIX extras
        sdxl_synthetics_dir: Path to SDXL synthetics
        augment: Apply augmentation
    
    Returns:
        UnlabeledRoadDataset
    """
    print(f"\n📊 Creating unlabeled dataset...")
    
    dataset = UnlabeledRoadDataset(
        natix_extras_dir=natix_extras_dir,
        sdxl_synthetics_dir=sdxl_synthetics_dir,
        augment=augment
    )
    
    print(f"✅ Unlabeled dataset created")
    print(f"   Total images: {len(dataset)}")
    print(f"   NATIX extras: {natix_extras_dir}")
    print(f"   SDXL synthetics: {sdxl_synthetics_dir}")
    print(f"   Augmentation: {augment}")
    
    return dataset


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
    print(f"PHASE 4.1: EXPLORA PRETRAINING (Dec 2025 Best Practice)")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Backbone: {args.backbone_path}")
    print(f"Unfreeze blocks: {args.unfreeze_blocks}")
    print(f"PEFT Rank: {args.peft_r}")
    print(f"Mask ratio: {args.mask_ratio:.1%}")
    print(f"Output Dir: {output_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}\n")
    
    # Step 1: Create backbone
    backbone = create_backbone(args.backbone_path, device)
    
    # Step 2: Create unlabeled dataset (Phase 4.1.1)
    dataset = create_unlabeled_dataset(
        args.natix_extras_dir,
        args.sdxl_synthetics_dir,
        augment=True  # Augmentation for pretraining
    )
    
    # Step 3: Create ExPLoRA trainer
    print(f"\n🎯 Creating ExPLoRA trainer...")
    explora_trainer = ExPLoRATrainer(
        backbone=backbone,
        device=device,
        unfreeze_blocks=args.unfreeze_blocks,
        peft_type="lora",
        peft_r=args.peft_r,
        peft_alpha=args.peft_alpha,
        mask_ratio=args.mask_ratio
    )
    
    # Step 4: Apply LoRA to backbone (Phase 4.1.3)
    print(f"\n🔄 Applying LoRA to frozen blocks (Phase 4.1.3)...")
    adapted_backbone = apply_lora_to_backbone(
        backbone=backbone,
        r=args.peft_r,
        lora_alpha=args.peft_alpha
    )
    explora_trainer.backbone = adapted_backbone
    
    # Count PEFT parameters
    peft_params = count_peft_parameters(adapted_backbone)
    print(f"✅ PEFT applied")
    print(f"   PEFT trainable: {peft_params['trainable']:,} ({peft_params['trainable']/1e6:.2f}M)")
    print(f"   Total trainable: {sum(p.numel() for p in adapted_backbone.parameters() if p.requires_grad):,} ({sum(p.numel() for p in adapted_backbone.parameters() if p.requires_grad)/1e6:.2f}M)")
    print(f"   Trainable ratio: {100*peft_params['trainable_ratio']:.2f}%")
    
    # Step 5: Save initial checkpoint
    initial_checkpoint_path = output_dir / "backbone_initial.pth"
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'explora_config': {
            'unfreeze_blocks': args.unfreeze_blocks,
            'peft_r': args.peft_r,
            'peft_alpha': args.peft_alpha,
            'mask_ratio': args.mask_ratio
        },
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }, initial_checkpoint_path)
    
    print(f"\n💾 Saved initial checkpoint to {initial_checkpoint_path}")
    
    # Step 6: Run ExPLoRA pretraining (Phase 4.1.2)
    print(f"\n{'='*80}")
    print(f"STARTING EXPLORA PRETRAINING")
    print(f"{'='*80}\n")
    
    adapted_backbone = explora_trainer.pretrain(
        unlabeled_dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=1
    )
    
    # Step 7: Save adapted backbone
    adapted_checkpoint_path = output_dir / "backbone_explora.pth"
    torch.save({
        'backbone_state_dict': adapted_backbone.state_dict(),
        'explora_config': {
            'unfreeze_blocks': args.unfreeze_blocks,
            'peft_r': args.peft_r,
            'peft_alpha': args.peft_alpha,
            'mask_ratio': args.mask_ratio
        },
        'pretrain_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr
        },
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }, adapted_checkpoint_path)
    
    print(f"\n💾 Saved ExPLoRA-adapted backbone to {adapted_checkpoint_path}")
    
    # Step 8: Save ExPLoRA decoder checkpoint
    decoder_checkpoint_path = output_dir / "decoder_explora.pth"
    explora_trainer.save_checkpoint(str(decoder_checkpoint_path))
    
    # Step 9: Create metadata
    metadata = {
        'backbone_path': args.backbone_path,
        'output_dir': str(output_dir),
        'explora_config': {
            'unfreeze_blocks': args.unfreeze_blocks,
            'peft_r': args.peft_r,
            'peft_alpha': args.peft_alpha,
            'mask_ratio': args.mask_ratio
        },
        'pretrain_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'natix_extras_dir': args.natix_extras_dir,
            'sdxl_synthetics_dir': args.sdxl_synthetics_dir
        },
        'checkpoint_paths': {
            'initial': str(initial_checkpoint_path),
            'adapted': str(adapted_checkpoint_path),
            'decoder': str(decoder_checkpoint_path)
        },
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / "explora_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Saved metadata to {metadata_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"EXPLORA PRETRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Output Artifacts:")
    print(f"   Initial backbone: {initial_checkpoint_path}")
    print(f"   Adapted backbone: {adapted_checkpoint_path}")
    print(f"   Decoder weights: {decoder_checkpoint_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"{'='*80}\n")
    
    # Instructions for downstream training
    print(f"\n📋 NEXT STEPS (Phase 4.1.3):")
    print(f"1. Use adapted backbone for supervised fine-tuning:")
    print(f"   python scripts/45_train_supervised_explora.py \\")
    print(f"       --backbone backbone_explora.pth \\")
    print(f"       --config config.yaml")
    print(f"\n2. Compare vs no-ExPLoRA baseline on val_test (Acceptance Test 4.1.4)")
    print()


if __name__ == "__main__":
    main()

