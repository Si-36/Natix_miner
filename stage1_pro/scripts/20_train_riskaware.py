#!/usr/bin/env python3
"""
Phase 1: Training script matching train_stage1_head.py exactly.

Usage:
    python scripts/20_train_riskaware.py --mode train --epochs 10
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from stage1_pro.config import Stage1ProConfig
from stage1_pro.model import DINOv3Backbone, Stage1Head
from stage1_pro.data import NATIXDataset, MultiRoadworkDataset
from stage1_pro.training import Stage1Trainer


def build_model(config: Stage1ProConfig):
    """Build model with frozen backbone and trainable head."""
    # Load DINOv3 backbone
    backbone = DINOv3Backbone(
        model_name=config.model_path,
        freeze=True,
        use_peft=False,  # Phase 1: no PEFT
    )

    # Create classifier head
    head = Stage1Head(hidden_size=backbone.embed_dim, dropout=config.dropout)

    # Compile for speed (2025 SOTA)
    head = torch.compile(head, mode="default")

    # Combine
    class Stage1Model(torch.nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, features=None, pixel_values=None):
            if pixel_values is not None:
                outputs = self.backbone(pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
            return self.head(features)

    model = Stage1Model(backbone, head)
    return model


def main():
    parser = argparse.ArgumentParser()
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "train_cached", "extract_features"],
    )

    # Paths
    parser.add_argument("--model_path", type=str, default="facebook/dinov2-large")
    parser.add_argument("--train_image_dir", type=str, default=None)
    parser.add_argument("--train_labels_file", type=str, default=None)
    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument("--val_labels_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument("--fallback_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    # Features
    parser.add_argument("--cached_features_dir", type=str, default="cached_features")

    # Phase 1 settings
    parser.add_argument("--target_fnr_exit", type=float, default=0.02)
    parser.add_argument("--val_select_ratio", type=float, default=0.5)

    # Advanced
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--early_stop_patience", type=int, default=3)

    # Multi-dataset
    parser.add_argument("--use_extra_roadwork", action="store_true")
    parser.add_argument("--roadwork_iccv_dir", type=str, default=None)
    parser.add_argument("--roadwork_extra_dir", type=str, default=None)

    # Resume
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    args = parser.parse_args()

    # Create config
    config = Stage1ProConfig(
        mode=args.mode,
        model_path=args.model_path,
        train_image_dir=args.train_image_dir,
        train_labels_file=args.train_labels_file,
        val_image_dir=args.val_image_dir,
        val_labels_file=args.val_labels_file,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        fallback_batch_size=args.fallback_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        max_grad_norm=args.max_grad_norm,
        cached_features_dir=args.cached_features_dir,
        target_fnr_exit=args.target_fnr_exit,
        val_select_ratio=args.val_select_ratio,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        early_stop_patience=args.early_stop_patience,
        use_extra_roadwork=args.use_extra_roadwork,
        roadwork_iccv_dir=args.roadwork_iccv_dir,
        roadwork_extra_dir=args.roadwork_extra_dir,
        resume_checkpoint=args.resume_checkpoint,
    )

    print("\n" + "=" * 80)
    print("STAGE-1 PRO TRAINING - PHASE 1: BASELINE")
    print("=" * 80)
    print(f"Mode: {config.mode}")
    print(f"Model: {config.model_path}")
    print(f"Output: {config.output_dir}")

    # Build model
    print("\n[1/5] Building model...")
    model = build_model(config)
    config.num_train_samples = 0

    # Build datasets
    print("\n[2/5] Building datasets...")
    if config.use_extra_roadwork:
        dataset_configs = [
            (config.train_image_dir, config.train_labels_file),
        ]
        # Add extra datasets if available
        if config.roadwork_iccv_dir:
            dataset_configs.append(
                (
                    os.path.join(config.roadwork_iccv_dir, "train_images"),
                    os.path.join(config.roadwork_iccv_dir, "train_labels.csv"),
                )
            )

        train_dataset = MultiRoadworkDataset(
            dataset_configs=dataset_configs, augment=True
        )
    else:
        train_dataset = NATIXDataset(
            root=config.train_image_dir,
            labels_path=config.train_labels_file,
            augment=True,
        )

    config.num_train_samples = len(train_dataset)

    val_dataset = NATIXDataset(
        root=config.val_image_dir, labels_path=config.val_labels_file, augment=False
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Build dataloaders
    print("\n[3/5] Building dataloaders...")
    batch_size = config.max_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # For torch.compile stability
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build trainer
    print("\n[4/5] Building trainer...")
    trainer = Stage1Trainer(model, config)

    # Train
    print("\n[5/5] Starting training...")
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)
        val_results = trainer.validate(val_loader)

        print(f"\nEpoch {epoch + 1}/{config.epochs}:")
        print(f"  Train: {train_loss:.4f} loss, {train_acc:.2f}% acc")
        print(
            f"  Val:   {val_results['loss']:.4f} loss, {val_results['accuracy']:.2f}% acc"
        )
        print(f"  ECE:    {val_results['ece']:.4f}")

        # Save best checkpoint
        if val_results["accuracy"] > best_acc:
            best_acc = val_results["accuracy"]
            patience_counter = 0

            os.makedirs(config.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(config.output_dir, "model_best.pth")

            if trainer.ema:
                trainer.ema.apply_shadow()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "scheduler_state_dict": trainer.scheduler.state_dict(),
                        "best_acc": best_acc,
                    },
                    checkpoint_path,
                )
                trainer.ema.restore()
            else:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": trainer.optimizer.state_dict(),
                        "scheduler_state_dict": trainer.scheduler.state_dict(),
                        "best_acc": best_acc,
                    },
                    checkpoint_path,
                )

            print(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stop_patience})")

            if patience_counter >= config.early_stop_patience:
                print(f"\nEarly stopping triggered")
                break

    print(f"\nTraining complete! Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
