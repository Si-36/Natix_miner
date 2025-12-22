#!/usr/bin/env python3
"""
DINOv3 Classifier Head Training Script
Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025

Strategy:
- Freeze DINOv3-Large backbone (1.3B params frozen)
- Train only MLP classifier head (300K params)
- 20× faster training vs full fine-tuning
- Use Focal Loss for hard negatives

Training Data:
- NATIX official dataset: 8,000 images
- SDXL synthetic: 1,000 images (FREE)
- Hard cases from FiftyOne: 200-400/week
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")


class FocalLoss(nn.Module):
    """
    Focal Loss for hard negative mining (per plan)
    
    Standard cross-entropy treats all errors equally.
    Focal loss focuses on HARD examples (low confidence).
    Expected: +0.3-0.5% accuracy on hard cases.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class DINOv3Classifier(nn.Module):
    """
    DINOv3-Large with frozen backbone + trainable MLP head
    
    Architecture:
    - DINOv3-Large: 1.3B params (FROZEN)
    - MLP Head: 300K params (TRAINABLE)
        - Linear(1536, 768)
        - ReLU
        - Dropout(0.3)
        - Linear(768, 2)
    """
    
    def __init__(self, backbone_path: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        from transformers import AutoModel
        
        # Load backbone and FREEZE it
        # DINOv3 HF repos can use custom code; enabling trust_remote_code makes it robust.
        self.backbone = AutoModel.from_pretrained(backbone_path, trust_remote_code=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get hidden size robustly.
        # Some DINOv3 checkpoints (timm-backed) don't expose config.hidden_size.
        hidden_size = getattr(getattr(self.backbone, "config", None), "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            # Infer from a dummy forward pass (CPU-safe).
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                try:
                    out = self.backbone(pixel_values=dummy)
                except TypeError:
                    out = self.backbone(dummy)
                if hasattr(out, "last_hidden_state"):
                    hidden_size = int(out.last_hidden_state.shape[-1])
                elif hasattr(out, "pooler_output"):
                    hidden_size = int(out.pooler_output.shape[-1])
                elif isinstance(out, torch.Tensor):
                    hidden_size = int(out.shape[-1])
                else:
                    raise RuntimeError(f"Cannot infer backbone hidden size from output type: {type(out)}")
        
        # Trainable classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )
        
        # Count parameters
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        logger.info(f"Backbone params: {backbone_params:,} (FROZEN)")
        logger.info(f"Classifier params: {classifier_params:,} (TRAINABLE)")
        logger.info(f"Trainable ratio: {classifier_params / backbone_params * 100:.4f}%")
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Get backbone features (no grad)
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0]  # CLS token
            
        # Classify (with grad)
        logits = self.classifier(features)
        return logits


class RoadworkDataset(Dataset):
    """
    Dataset for roadwork classification
    
    Sources:
    - NATIX official: 8,000 images
    - SDXL synthetic: 1,000 images
    - Hard cases: 200-400/week
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Default transform (Validator-aligned per plan)
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform
            
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> list:
        """Load image paths and labels from data directory"""
        samples = []
        
        # Structure: data_dir/positive/*.jpg, data_dir/negative/*.jpg
        positive_dir = self.data_dir / "positive"
        negative_dir = self.data_dir / "negative"
        
        if positive_dir.exists():
            for img_path in positive_dir.glob("*.jpg"):
                samples.append((img_path, 1))
            for img_path in positive_dir.glob("*.png"):
                samples.append((img_path, 1))
                
        if negative_dir.exists():
            for img_path in negative_dir.glob("*.jpg"):
                samples.append((img_path, 0))
            for img_path in negative_dir.glob("*.png"):
                samples.append((img_path, 0))
                
        # Shuffle
        np.random.shuffle(samples)
        
        # Split
        split_idx = int(len(samples) * 0.8)
        if self.split == "train":
            samples = samples[:split_idx]
        else:
            samples = samples[split_idx:]
            
        logger.info(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        return image, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch"""
    model.train()
    model.classifier.train()  # Only classifier is trainable
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total*100:.2f}%"
        })
        
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 Classifier Head")
    parser.add_argument("--backbone-path", type=str, required=True,
                        help="Path to DINOv3/DINOv2 backbone")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--use-focal-loss", action="store_true",
                        help="Use Focal Loss for hard negatives")
    args = parser.parse_args()
    
    # Setup
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DINOv3 Classifier Head Training")
    logger.info("Per REALISTIC_DEPLOYMENT_PLAN.md - December 20, 2025")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading backbone from: {args.backbone_path}")
    model = DINOv3Classifier(
        backbone_path=args.backbone_path,
        num_classes=2,
        dropout=0.3
    ).to(device)
    
    # Load data
    logger.info(f"Loading data from: {args.data_dir}")
    train_dataset = RoadworkDataset(args.data_dir, split="train")
    val_dataset = RoadworkDataset(args.data_dir, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss function
    if args.use_focal_loss:
        logger.info("Using Focal Loss (per plan: +0.3-0.5% on hard cases)")
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
        
    # Optimizer (only classifier params)
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = output_dir / f"dinov3_classifier_best.pth"
            torch.save({
                "epoch": epoch,
                "classifier_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss
            }, checkpoint_path)
            logger.info(f"✅ New best model saved: {checkpoint_path}")
            
    # Save final model
    final_path = output_dir / f"dinov3_classifier_final.pth"
    torch.save({
        "epoch": args.epochs,
        "classifier_state_dict": model.classifier.state_dict(),
        "val_acc": val_acc,
    }, final_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

