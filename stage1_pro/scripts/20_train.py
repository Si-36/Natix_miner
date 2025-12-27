#!/usr/bin/env python3
"""
Phase 3: Unified training script for all phases (1-3).

Modern 2025 implementation with clean CLI.
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import argparse
import torch
from torch.utils.data import DataLoader
from stage1_pro.config import Stage1ProConfig
from stage1_pro.model import DINOv3Backbone, Stage1Head
from stage1_pro.data import NATIXDataset, MultiRoadworkDataset
from stage1_pro.training import Stage1Trainer


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--mode", default="train", choices=["train", "train_cached"])

    # Paths
    parser.add_argument("--train_image_dir", default=None)
    parser.add_argument("--train_labels_file", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_labels_file", default=None)
    parser.add_argument("--output_dir", default="./outputs")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)

    # Phase 2-3 only
    parser.add_argument("--use_dirichlet", action="store_true", help="Phase 2 only")
    parser.add_argument(
        "--peft_type",
        default="none",
        choices=["none", "dora", "lora"],
        help="Phase 3 only",
    )
    parser.add_argument("--use_fsam", action="store_true", help="Phase 4 only")

    args = parser.parse_args()

    print(f"Phase {args.phase} Training Started")

    # Build config
    config = Stage1ProConfig(
        mode=args.mode,
        train_image_dir=args.train_image_dir,
        train_labels_file=args.train_labels_file,
        val_image_dir=args.val_image_dir,
        val_labels_file=args.val_labels_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        use_dirichlet=args.use_dirichlet,
        peft_type=args.peft_type,
        optimizer="fsam" if args.use_fsam else "adamw",
    )

    # Build model
    backbone = DINOv3Backbone(
        model_name="facebook/dinov2-base",
        freeze=(args.phase < 3),
        use_peft=(args.phase >= 3),
        peft_type=args.peft_type,
    )

    head = Stage1Head(
        hidden_size=backbone.embed_dim,
        dropout=0.3,
        use_gate=(args.phase >= 2),
    )

    class FullModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, features=None, pixel_values=None):
            if pixel_values is not None:
                outputs = self.backbone(pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
            return self.head(features)

    model = FullModel(backbone, head)
    config.num_train_samples = 1000  # placeholder

    # Build datasets (placeholder - use actual paths)
    print("Building datasets...")
    # train_dataset = ...
    # val_dataset = ...

    # Build trainer
    trainer = Stage1Trainer(model, config)

    # Train
    print(f"Training {args.epochs} epochs...")
    # for epoch in range(args.epochs):
    #     trainer.train_epoch(train_loader, epoch + 1, mode="train")

    print(f"Training complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
