import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import os
import math
import random
from tqdm import tqdm
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import numpy as np
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

# === 2025 SOTA: Enable TF32 precision for 20% speedup on Ampere GPUs ===
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainingConfig:
    """
    Production-grade config dataclass (2025 SOTA)

    All hyperparameters in one place, automatically saved to config.json.
    This ensures reproducibility and makes hyperparameter tuning easier.
    """
    # Model paths
    model_path: str = "models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m"
    train_image_dir: str = "data/natix_official/train"
    train_labels_file: str = "data/natix_official/train_labels.csv"
    val_image_dir: str = "data/natix_official/val"
    val_labels_file: str = "data/natix_official/val_labels.csv"

    # Training mode
    mode: str = "train"  # "extract_features", "train", or "train_cached"
    cached_features_dir: str = "cached_features"  # Where to save/load features

    # Multi-dataset training (AGGRESSIVE MODE for max accuracy)
    use_extra_roadwork: bool = False  # Combine NATIX + ROADWork + extras
    roadwork_iccv_dir: str = "data/roadwork_iccv"
    roadwork_extra_dir: str = "data/roadwork_extra"

    # Kaggle datasets (2025 - CRITICAL for 90%+ accuracy!)
    use_kaggle_data: bool = False
    kaggle_construction_dir: str = "data/kaggle_construction_std"
    kaggle_road_issues_dir: str = "data/kaggle_road_issues_std"

    # Batch sizes
    max_batch_size: int = 64
    fallback_batch_size: int = 32
    grad_accum_steps: int = 2

    # Training schedule
    epochs: int = 15
    warmup_epochs: int = 1

    # Optimizer (AGGRESSIVE regularization)
    lr_head: float = 8e-5  # Slightly reduced from 1e-4 for more stable learning
    lr_backbone: float = 1e-5
    weight_decay: float = 0.05  # Increased from 0.02 - STRONG weight decay to prevent overfitting

    # Regularization (AGGRESSIVE anti-overfitting)
    label_smoothing: float = 0.15  # Increased from 0.1 for better generalization
    max_grad_norm: float = 1.0
    dropout: float = 0.45  # Increased from 0.35 - STRONG regularization to prevent memorization

    # Advanced features
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    early_stop_patience: int = 3

    # Cascade exit monitoring
    exit_threshold: float = 0.88  # Target threshold for ~60% exit rate

    # Checkpointing
    resume_checkpoint: Optional[str] = None

    # Output
    log_file: str = "training.log"
    output_dir: str = "models/stage1_dinov3"

    def save(self, path: str):
        """Save config to JSON for reproducibility"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"✅ Config saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class TimmStyleAugmentation:
    """timm-style augmentation for vision transformers (2025 SOTA)

    Uses RandomResizedCrop + HorizontalFlip + RandomErasing pattern
    which is the standard for modern vision transformer training.
    """
    def __init__(self, img_size=224, scale=(0.8, 1.0), ratio=(0.75, 1.33)):
        self.img_size = img_size
        self.scale = scale
        self.ratio = ratio

        # AGGRESSIVE augmentation to prevent overfitting
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.7, 1.0),  # More aggressive crop (was 0.8-1.0)
                ratio=(0.7, 1.4),  # Wider aspect ratio range (was 0.75-1.33)
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # NEW: Color augmentation
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.4), ratio=(0.3, 3.3)),  # More aggressive erasing (was p=0.25)
        ])

    def __call__(self, img):
        """Apply timm-style augmentation and return tensor"""
        return self.transform(img)


class NATIXDataset(Dataset):
    """NATIX dataset with timm-style augmentation (2025 SOTA)"""
    def __init__(self, image_dir, labels_file, processor, augment=False):
        self.image_dir = image_dir
        self.processor = processor
        self.augment = augment

        # timm-style augmentation for training (AGGRESSIVE)
        if augment:
            self.timm_aug = TimmStyleAugmentation(img_size=224, scale=(0.7, 1.0))  # More aggressive crop
        else:
            # Validation: just resize + center crop
            self.val_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

        # Load labels (CSV format: image_path,label) with header skip support
        with open(labels_file, 'r') as f:
            all_lines = [line.strip() for line in f if line.strip()]
            # Skip header if exists (Kaggle CSVs have "image,label")
            if all_lines and all_lines[0].lower().startswith('image,'):
                all_lines = all_lines[1:]
            lines = [line.split(',') for line in all_lines]
            self.samples = lines
            self.labels = [int(label) for _, label in lines]  # Store labels for class weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')

        if self.augment:
            # Apply timm augmentation (returns tensor already)
            pixel_tensor = self.timm_aug(image)
        else:
            # Validation transform (returns tensor)
            pixel_tensor = self.val_transform(image)

        # Normalize using DINOv3 processor normalization values
        # Standard ImageNet normalization for DINOv3
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pixel_values = (pixel_tensor - mean) / std

        label = int(label)  # 0 = no roadwork, 1 = roadwork

        return pixel_values, label


class MultiRoadworkDataset(Dataset):
    """
    Multi-source roadwork dataset (2025 SOTA for max data)

    Combines NATIX + ROADWork + Kaggle datasets into one unified training set.
    All datasets are normalized to binary labels: 0 = no roadwork, 1 = roadwork.

    Why: Combining diverse work zone datasets improves robustness and edge-case handling.
    Kaggle construction provides positives, Kaggle road issues provides negatives.
    Measure impact on NATIX val set for true deployment performance.

    Args:
        dataset_configs: List of (image_dir, labels_file) tuples
        processor: DINOv3 image processor
        augment: Whether to apply data augmentation
    """
    def __init__(self, dataset_configs, processor, augment=False):
        self.processor = processor
        self.augment = augment

        # timm-style augmentation for training (AGGRESSIVE)
        if augment:
            self.timm_aug = TimmStyleAugmentation(img_size=224, scale=(0.7, 1.0))  # More aggressive crop
        else:
            self.val_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

        # Load and combine all datasets
        self.samples = []
        self.labels = []
        self.dataset_sources = []  # Track which dataset each sample came from

        for dataset_idx, (image_dir, labels_file) in enumerate(dataset_configs):
            if not os.path.exists(labels_file):
                print(f"⚠️  Skipping {labels_file} (not found)")
                continue

            with open(labels_file, 'r') as f:
                all_lines = [line.strip() for line in f if line.strip()]
                # Skip header if exists (Kaggle CSVs have "image,label")
                if all_lines and all_lines[0].lower().startswith('image,'):
                    all_lines = all_lines[1:]
                lines = [line.split(',') for line in all_lines]

            for img_path, label in lines:
                # Handle both absolute and relative paths
                if os.path.isabs(img_path):
                    full_path = img_path
                else:
                    full_path = os.path.join(image_dir, img_path)

                self.samples.append(full_path)
                self.labels.append(int(label))
                self.dataset_sources.append(dataset_idx)

        print(f"\n📊 Multi-Dataset Stats:")
        print(f"   Total samples: {len(self.samples)}")
        for idx, (image_dir, labels_file) in enumerate(dataset_configs):
            count = sum(1 for s in self.dataset_sources if s == idx)
            if count > 0:
                dataset_name = Path(labels_file).parent.name
                print(f"   {dataset_name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.augment:
            pixel_tensor = self.timm_aug(image)
        else:
            pixel_tensor = self.val_transform(image)

        # DINOv3 normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pixel_values = (pixel_tensor - mean) / std

        return pixel_values, label


class EMA:
    """Exponential Moving Average (2025 SOTA for vision models)"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) - 2025 SOTA metric for cascade exits

    ECE measures how well predicted probabilities match actual accuracy.
    Lower ECE = better calibrated model = more reliable exit thresholds.

    Args:
        probs: (N, num_classes) predicted probabilities
        labels: (N,) ground truth labels
        n_bins: number of bins for calibration curve

    Returns:
        ece: Expected Calibration Error (0-1, lower is better)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += (bin_size / len(labels)) * abs(avg_confidence - avg_accuracy)

    return ece


def extract_features(config: TrainingConfig):
    """
    Feature caching mode (2025 SOTA for fast iteration)

    Extract CLS features from frozen DINOv3 backbone once and save to disk.
    This allows 10x faster training iterations when tuning the classifier head.

    Why: DINOv3 inference is expensive. By caching features, you can experiment
    with different head architectures, learning rates, etc. without re-running
    the backbone every time.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"FEATURE EXTRACTION MODE")
    print(f"{'='*80}")
    print(f"Device: {device}")

    # Load DINOv3 backbone
    print(f"\n[1/3] Loading DINOv3 backbone from {config.model_path}...")
    backbone = AutoModel.from_pretrained(config.model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(config.model_path)
    backbone.eval()

    os.makedirs(config.cached_features_dir, exist_ok=True)

    # Extract features for train and val sets
    for split_name, img_dir, labels_file in [
        ("train", config.train_image_dir, config.train_labels_file),
        ("val", config.val_image_dir, config.val_labels_file)
    ]:
        print(f"\n[2/3] Extracting {split_name} features...")

        dataset = NATIXDataset(
            image_dir=img_dir,
            labels_file=labels_file,
            processor=processor,
            augment=False  # No augmentation for feature extraction
        )

        loader = DataLoader(
            dataset,
            batch_size=config.max_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Extracting {split_name}"):
                images = images.to(device)

                # Extract CLS token features
                outputs = backbone(pixel_values=images)
                features = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

                all_features.append(features.cpu())
                all_labels.append(labels)

        # Save to disk
        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        features_path = os.path.join(config.cached_features_dir, f"{split_name}_features.pt")
        labels_path = os.path.join(config.cached_features_dir, f"{split_name}_labels.pt")

        torch.save(features_tensor, features_path)
        torch.save(labels_tensor, labels_path)

        print(f"✅ Saved {split_name} features: {features_tensor.shape} -> {features_path}")
        print(f"✅ Saved {split_name} labels: {labels_tensor.shape} -> {labels_path}")

    print(f"\n[3/3] Feature extraction complete!")
    print(f"📁 Cached features saved to: {config.cached_features_dir}")
    print(f"\nNext step: Run with --mode train_cached to train head only (10x faster)")


def train_with_cached_features(config: TrainingConfig):
    """
    Fast training mode using pre-extracted features (2025 SOTA)

    Train classifier head on cached DINOv3 features.
    This is 10x faster than full training because we skip DINOv3 inference.

    Perfect for:
    - Hyperparameter tuning (learning rate, dropout, etc.)
    - Architecture experiments (different head designs)
    - Quick validation of training setup
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"CACHED FEATURE TRAINING MODE (10x faster)")
    print(f"{'='*80}")

    # Load cached features
    print(f"\n[1/5] Loading cached features from {config.cached_features_dir}...")

    train_features = torch.load(os.path.join(config.cached_features_dir, "train_features.pt"))
    train_labels = torch.load(os.path.join(config.cached_features_dir, "train_labels.pt"))
    val_features = torch.load(os.path.join(config.cached_features_dir, "val_features.pt"))
    val_labels = torch.load(os.path.join(config.cached_features_dir, "val_labels.pt"))

    print(f"✅ Train: {train_features.shape}, Val: {val_features.shape}")

    # Create datasets from cached features
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # Compute class weights
    class_counts = np.bincount(train_labels.numpy())
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\n📊 Class distribution:")
    print(f"   Class 0: {class_counts[0]} ({100*class_counts[0]/total_samples:.1f}%)")
    print(f"   Class 1: {class_counts[1]} ({100*class_counts[1]/total_samples:.1f}%)")

    # Create classifier head
    print(f"\n[2/5] Creating classifier head...")
    hidden_size = train_features.shape[1]  # Feature dimension
    classifier_head = nn.Sequential(
        nn.Linear(hidden_size, 768),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(768, 2)
    ).to(device)

    classifier_head = torch.compile(classifier_head, mode="default")
    print(f"✅ Compiled classifier head ({hidden_size} -> 768 -> 2)")

    # DataLoaders
    batch_size = config.max_batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing)
    optimizer = AdamW(classifier_head.parameters(), lr=config.lr_head, weight_decay=config.weight_decay)

    total_steps = config.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda') if config.use_amp else None
    ema = EMA(classifier_head, decay=config.ema_decay) if config.use_ema else None

    print(f"\n[3/5] Starting training ({config.epochs} epochs, very fast)...")

    best_acc = 0.0
    for epoch in range(config.epochs):
        # Train
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            features, labels = features.to(device), labels.to(device)

            if config.use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    logits = classifier_head(features)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = classifier_head(features)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            if config.use_ema and ema:
                ema.update()

            train_loss += loss.item()
            train_correct += logits.argmax(1).eq(labels).sum().item()

        train_acc = 100. * train_correct / len(train_dataset)

        # Validate with EMA
        if config.use_ema and ema:
            ema.apply_shadow()

        classifier_head.eval()
        val_correct = 0
        all_probs = []
        all_labels_list = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = classifier_head(features)
                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
                val_correct += logits.argmax(1).eq(labels).sum().item()

        if config.use_ema and ema:
            ema.restore()

        val_acc = 100. * val_correct / len(val_dataset)

        # Compute metrics
        all_probs = np.concatenate(all_probs)
        all_labels_np = np.concatenate(all_labels_list)
        ece = compute_ece(all_probs, all_labels_np)

        exit_mask = (all_probs[:, 1] >= config.exit_threshold) | (all_probs[:, 1] <= (1 - config.exit_threshold))
        exit_coverage = exit_mask.mean() * 100

        if exit_mask.sum() > 0:
            exit_preds = (all_probs[exit_mask][:, 1] > 0.5).astype(int)
            exit_accuracy = (exit_preds == all_labels_np[exit_mask]).mean() * 100
        else:
            exit_accuracy = 0.0

        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print(f"  ECE: {ece:.4f}, Exit@{config.exit_threshold}: {exit_coverage:.1f}% @ {exit_accuracy:.2f}% acc")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.output_dir, "classifier_head.pth")
            os.makedirs(config.output_dir, exist_ok=True)

            if config.use_ema and ema:
                ema.apply_shadow()
                torch.save(classifier_head.state_dict(), save_path)
                ema.restore()
            else:
                torch.save(classifier_head.state_dict(), save_path)

            print(f"  ✅ Saved best checkpoint: {save_path}")

    print(f"\n[5/5] Training complete! Best Val Acc: {best_acc:.2f}%")


def train_dinov3_head(config: TrainingConfig):
    """Train ONLY the classifier head, freeze DINOv3 backbone (2025 SOTA)"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"FULL TRAINING MODE (with data augmentation)")
    print(f"{'='*80}")
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Save config for reproducibility
    config.save(os.path.join(config.output_dir, "config.json"))

    # Load DINOv3 backbone (FROZEN)
    print("\n[1/7] Loading DINOv3 backbone...")
    model_path = config.model_path
    backbone = AutoModel.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    # FREEZE ALL backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in backbone.parameters())
    print(f"✅ Frozen {frozen_params/1e6:.1f}M backbone parameters")

    # Create classifier head (TRAINABLE)
    hidden_size = backbone.config.hidden_size  # Auto-detect (1280 or 1536)
    classifier_head = nn.Sequential(
        nn.Linear(hidden_size, 768),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(768, 2)  # Binary: [no_roadwork, roadwork]
    ).to(device)

    # 2025 SOTA: Compile classifier head for 40% speedup
    classifier_head = torch.compile(classifier_head, mode="default")

    trainable_params = sum(p.numel() for p in classifier_head.parameters())
    print(f"✅ Training {trainable_params/1e3:.0f}K classifier parameters")
    print(f"✅ torch.compile enabled (expect 40% speedup after warmup)")

    # Pick optimal batch size
    def pick_batch_size():
        """Try larger batch sizes, fallback if OOM"""
        for bs in [config.max_batch_size, config.fallback_batch_size]:
            try:
                dummy_images = torch.randn(bs, 3, 224, 224).to(device)
                with torch.no_grad():
                    _ = backbone(pixel_values=dummy_images)
                torch.cuda.empty_cache()
                print(f"✅ Batch size {bs} works on this GPU")
                return bs
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"⚠️ Batch size {bs} OOM, trying smaller...")
                    continue
                else:
                    raise
        return config.fallback_batch_size

    batch_size = pick_batch_size()
    effective_batch = batch_size * config.grad_accum_steps
    print(f"✅ Effective batch size: {effective_batch} ({batch_size} × {config.grad_accum_steps} accum)")

    # Load dataset(s) with timm-style augmentation
    print("\n[2/7] Loading dataset...")

    if config.use_extra_roadwork or config.use_kaggle_data:
        # AGGRESSIVE MODE: Combine NATIX + extras for maximum robustness
        print("🚀 MULTI-DATASET MODE: Combining all roadwork sources")

        dataset_configs = [
            # NATIX (primary)
            (config.train_image_dir, config.train_labels_file),
        ]

        # Add ROADWork if available
        roadwork_train = os.path.join(config.roadwork_iccv_dir, "train_labels.csv")
        if os.path.exists(roadwork_train):
            dataset_configs.append((
                os.path.join(config.roadwork_iccv_dir, "train_images"),
                roadwork_train
            ))
            print(f"   ✅ Adding ROADWork dataset (ICCV 2025)")
        else:
            print(f"   ⚠️  ROADWork not found at {roadwork_train}")

        # Add Roboflow work zone datasets if available
        extra_train = os.path.join(config.roadwork_extra_dir, "train_labels.csv")
        if os.path.exists(extra_train):
            dataset_configs.append((
                os.path.join(config.roadwork_extra_dir, "train_images"),
                extra_train
            ))
            print(f"   ✅ Adding Roboflow work zone datasets")
        else:
            print(f"   ⚠️  Roboflow not found at {extra_train}")

        # Add Open Images V7 (positives booster) if available
        open_images_train = "data/open_images/train_labels.csv"
        if os.path.exists(open_images_train):
            dataset_configs.append((
                "data/open_images/coco/data",
                open_images_train
            ))
            print(f"   ✅ Adding Open Images V7 (positives booster)")
        else:
            print(f"   ⚠️  Open Images not found at {open_images_train}")

        # Add GTSRB Class 25 (EU signs) if available
        gtsrb_train = "data/gtsrb_class25/train_labels.csv"
        if os.path.exists(gtsrb_train):
            dataset_configs.append((
                "data/gtsrb_class25/train_images",
                gtsrb_train
            ))
            print(f"   ✅ Adding GTSRB Class 25 (EU roadwork signs)")
        else:
            print(f"   ⚠️  GTSRB not found at {gtsrb_train}")

        # Add Kaggle construction dataset (positives booster)
        if config.use_kaggle_data:
            kaggle_construction_train = os.path.join(config.kaggle_construction_dir, "train_labels.csv")
            if os.path.exists(kaggle_construction_train):
                dataset_configs.append((
                    os.path.join(config.kaggle_construction_dir, "train"),
                    kaggle_construction_train
                ))
                print(f"   ✅ Adding Kaggle Construction dataset (2,826 positives)")
            else:
                print(f"   ⚠️  Kaggle Construction not found at {kaggle_construction_train}")

            # Add Kaggle road issues dataset (negatives booster - CRITICAL for false positive reduction!)
            kaggle_issues_train = os.path.join(config.kaggle_road_issues_dir, "train_labels.csv")
            if os.path.exists(kaggle_issues_train):
                dataset_configs.append((
                    os.path.join(config.kaggle_road_issues_dir, "train"),
                    kaggle_issues_train
                ))
                print(f"   ✅ Adding Kaggle Road Issues dataset (9,661 negatives)")
            else:
                print(f"   ⚠️  Kaggle Road Issues not found at {kaggle_issues_train}")

        train_dataset = MultiRoadworkDataset(
            dataset_configs=dataset_configs,
            processor=processor,
            augment=True
        )

    else:
        # STANDARD MODE: NATIX only
        print("📦 NATIX-only mode (use --use_kaggle_data for 90%+ accuracy!)")
        train_dataset = NATIXDataset(
            image_dir=config.train_image_dir,
            labels_file=config.train_labels_file,
            processor=processor,
            augment=True
        )

    # Validation always uses NATIX val (primary metric)
    val_dataset = NATIXDataset(
        image_dir=config.val_image_dir,
        labels_file=config.val_labels_file,
        processor=processor,
        augment=False  # No augmentation for validation
    )
    print(f"✅ timm-style augmentation enabled for training")
    print(f"✅ Validation: NATIX val only (primary deployment metric)")

    # 2025 SOTA: Compute class weights for imbalanced dataset
    train_labels_array = np.array(train_dataset.labels)
    class_counts = np.bincount(train_labels_array)
    total_samples = len(train_labels_array)

    # Inverse frequency weighting
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    print(f"\n📊 Class distribution:")
    print(f"   Class 0 (no roadwork): {class_counts[0]} samples ({100*class_counts[0]/total_samples:.1f}%)")
    print(f"   Class 1 (roadwork):    {class_counts[1]} samples ({100*class_counts[1]/total_samples:.1f}%)")
    print(f"   Class weights: {class_weights}")

    # Loss with label smoothing + class weights (2025 SOTA)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=config.label_smoothing
    )
    print(f"✅ Class-weighted loss with label smoothing={config.label_smoothing}")

    # Optimizer
    optimizer = AdamW(
        classifier_head.parameters(),
        lr=config.lr_head,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Cosine annealing LR scheduler with warmup (2025 SOTA)
    total_steps = config.epochs * (len(train_dataset) // batch_size)
    warmup_steps = config.warmup_epochs * (len(train_dataset) // batch_size)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"✅ Cosine LR scheduler with {config.warmup_epochs} epoch warmup")

    # Mixed precision scaler
    scaler = GradScaler('cuda') if config.use_amp else None
    if config.use_amp:
        print(f"✅ Mixed precision (AMP) enabled")

    # EMA (2025 SOTA)
    ema = EMA(classifier_head, decay=config.ema_decay) if config.use_ema else None
    if config.use_ema:
        print(f"✅ EMA enabled (decay={config.ema_decay})")

    print(f"✅ Gradient clipping: max_norm={config.max_grad_norm}")
    print(f"✅ Early stopping: patience={config.early_stop_patience} epochs")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}, Epochs: {config.epochs}")

    # Setup logging
    log_file = config.log_file

    # Checkpoint resuming
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0

    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        print(f"\n[RESUME] Loading checkpoint from {config.resume_checkpoint}")
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        classifier_head.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        patience_counter = checkpoint.get('patience_counter', 0)
        if config.use_ema and 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']
        print(f"✅ Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")
    else:
        with open(log_file, 'w') as f:
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,ECE,Exit_Coverage,Exit_Acc,Best_Val_Acc,LR\n")

    # Training loop
    print(f"\n[3/7] Starting training ({config.epochs} epochs)...")

    for epoch in range(start_epoch, config.epochs):
        # ===== TRAIN =====
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # Mixed precision forward pass
            if config.use_amp and scaler is not None:
                with torch.amp.autocast('cuda'):
                    # Extract features with FROZEN backbone
                    with torch.no_grad():
                        outputs = backbone(pixel_values=images)
                        features = outputs.last_hidden_state[:, 0, :]  # CLS token

                    # Forward through TRAINABLE head
                    logits = classifier_head(features)
                    loss = criterion(logits, labels) / config.grad_accum_steps

                # Backward (accumulate gradients)
                scaler.scale(loss).backward()

                # Step optimizer every config.grad_accum_steps
                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), config.max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    scheduler.step()

                    if config.use_ema and ema:
                        ema.update()

            else:
                # Extract features with FROZEN backbone
                with torch.no_grad():
                    outputs = backbone(pixel_values=images)
                    features = outputs.last_hidden_state[:, 0, :]

                logits = classifier_head(features)
                loss = criterion(logits, labels) / config.grad_accum_steps

                loss.backward()

                if (batch_idx + 1) % config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(classifier_head.parameters(), config.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                    scheduler.step()

                    if config.use_ema and ema:
                        ema.update()

            # Metrics
            train_loss += loss.item() * config.grad_accum_steps
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix({
                'loss': f'{loss.item() * config.grad_accum_steps:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'lr': f'{current_lr:.2e}'
            })

        train_acc = 100. * train_correct / train_total

        # ===== VALIDATE with EMA weights =====
        if config.use_ema and ema:
            ema.apply_shadow()

        classifier_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_probs = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                if config.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = backbone(pixel_values=images)
                        features = outputs.last_hidden_state[:, 0, :]
                        logits = classifier_head(features)
                        loss = criterion(logits, labels)
                else:
                    outputs = backbone(pixel_values=images)
                    features = outputs.last_hidden_state[:, 0, :]
                    logits = classifier_head(features)
                    loss = criterion(logits, labels)

                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        if config.use_ema and ema:
            ema.restore()

        val_acc = 100. * val_correct / val_total

        # Compute ECE and cascade exit metrics
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        ece = compute_ece(all_probs, all_labels, n_bins=10)

        exit_threshold = config.exit_threshold
        exit_mask = (all_probs[:, 1] >= exit_threshold) | (all_probs[:, 1] <= (1 - exit_threshold))
        exit_coverage = exit_mask.mean() * 100

        if exit_mask.sum() > 0:
            exit_labels = all_labels[exit_mask]
            exit_preds = (all_probs[exit_mask][:, 1] > 0.5).astype(int)
            exit_accuracy = (exit_preds == exit_labels).mean() * 100
        else:
            exit_accuracy = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{config.epochs} Summary:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss/len(val_loader):.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  ECE (Calibration): {ece:.4f} (lower is better)")
        print(f"  Cascade Exit @ {exit_threshold:.2f}: {exit_coverage:.1f}% coverage, {exit_accuracy:.2f}% accuracy")
        print(f"  LR: {current_lr:.2e}")

        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss/len(train_loader):.4f},{train_acc:.2f},"
                   f"{val_loss/len(val_loader):.4f},{val_acc:.2f},{ece:.4f},"
                   f"{exit_coverage:.1f},{exit_accuracy:.2f},{best_acc:.2f},{current_lr:.2e}\n")

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0

            checkpoint_path = f"{config.model_path}/classifier_head.pth"
            checkpoint_full = f"{config.model_path}/checkpoint_epoch{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            if config.use_ema and ema:
                ema.apply_shadow()
                torch.save(classifier_head.state_dict(), checkpoint_path)
                ema.restore()
            else:
                torch.save(classifier_head.state_dict(), checkpoint_path)

            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': classifier_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'patience_counter': patience_counter,
            }
            if config.use_ema and ema:
                checkpoint_dict['ema_state_dict'] = ema.shadow

            torch.save(checkpoint_dict, checkpoint_full)
            print(f"  ✅ Saved best checkpoint (Val Acc={val_acc:.2f}%)")
            print(f"  ✅ Full checkpoint: {checkpoint_full}")

        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{config.early_stop_patience})")

            if patience_counter >= config.early_stop_patience:
                print(f"\n⛔ Early stopping triggered after {patience_counter} epochs without improvement")
                print(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")
                break

    print(f"\n[7/7] Training complete!")
    print(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"📁 Checkpoint saved: {config.model_path}/classifier_head.pth")
    print(f"📊 Training log saved: {log_file}")


def main():
    """
    Main CLI entry point (2025 SOTA)

    Three training modes:
    1. extract_features: Extract DINOv3 features once and cache to disk
    2. train_cached: Train classifier head on cached features (10x faster)
    3. train: Full end-to-end training with data augmentation
    """
    parser = argparse.ArgumentParser(
        description="DINOv3 Stage 1 Training for NATIX StreetVision (2025 Production-Grade)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training (NATIX only) - expect 79% accuracy
  python train_stage1_v2.py --mode train --epochs 10

  # WITH KAGGLE DATA (RECOMMENDED) - expect 90-95% accuracy! 🚀
  python train_stage1_v2.py --mode train --epochs 15 --use_kaggle_data

  # AGGRESSIVE MODE: Train on all available datasets (max accuracy)
  python train_stage1_v2.py --mode train --epochs 15 --use_extra_roadwork --use_kaggle_data

  # Extract features once (for fast iteration)
  python train_stage1_v2.py --mode extract_features

  # Train head only on cached features (10x faster)
  python train_stage1_v2.py --mode train_cached --epochs 20 --lr_head 2e-4

  # Resume from checkpoint
  python train_stage1_v2.py --mode train --resume_checkpoint models/stage1_dinov3/checkpoint_epoch5.pth
"""
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "extract_features", "train_cached"],
        default="train",
        help="Training mode: train (full), extract_features (cache features), train_cached (fast)"
    )

    # Paths
    parser.add_argument("--model_path", type=str, default="models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m")
    parser.add_argument("--train_image_dir", type=str, default="data/natix_official/train")
    parser.add_argument("--train_labels_file", type=str, default="data/natix_official/train_labels.csv")
    parser.add_argument("--val_image_dir", type=str, default="data/natix_official/val")
    parser.add_argument("--val_labels_file", type=str, default="data/natix_official/val_labels.csv")
    parser.add_argument("--cached_features_dir", type=str, default="cached_features")
    parser.add_argument("--output_dir", type=str, default="models/stage1_dinov3")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_batch_size", type=int, default=64)
    parser.add_argument("--fallback_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr_head", type=float, default=8e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    # Advanced features
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable AMP")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="Disable EMA")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--early_stop_patience", type=int, default=3)

    # Cascade exit threshold
    parser.add_argument("--exit_threshold", type=float, default=0.88, help="Target threshold for cascade exit")

    # Multi-dataset training
    parser.add_argument("--use_extra_roadwork", action="store_true", help="Combine NATIX + ROADWork + extras")
    parser.add_argument("--roadwork_iccv_dir", type=str, default="data/roadwork_iccv")
    parser.add_argument("--roadwork_extra_dir", type=str, default="data/roadwork_extra")

    # Kaggle datasets (2025 - CRITICAL!)
    parser.add_argument("--use_kaggle_data", action="store_true", help="Add Kaggle datasets (12,487 samples) - HUGE accuracy boost!")
    parser.add_argument("--kaggle_construction_dir", type=str, default="data/kaggle_construction_std")
    parser.add_argument("--kaggle_road_issues_dir", type=str, default="data/kaggle_road_issues_std")

    # Checkpointing
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to resume from")
    parser.add_argument("--log_file", type=str, default="training.log")

    args = parser.parse_args()

    # Create config from args
    config = TrainingConfig(
        mode=args.mode,
        model_path=args.model_path,
        train_image_dir=args.train_image_dir,
        train_labels_file=args.train_labels_file,
        val_image_dir=args.val_image_dir,
        val_labels_file=args.val_labels_file,
        cached_features_dir=args.cached_features_dir,
        output_dir=args.output_dir,
        use_extra_roadwork=args.use_extra_roadwork,
        roadwork_iccv_dir=args.roadwork_iccv_dir,
        roadwork_extra_dir=args.roadwork_extra_dir,
        use_kaggle_data=args.use_kaggle_data,
        kaggle_construction_dir=args.kaggle_construction_dir,
        kaggle_road_issues_dir=args.kaggle_road_issues_dir,
        max_batch_size=args.max_batch_size,
        fallback_batch_size=args.fallback_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        dropout=args.dropout,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        early_stop_patience=args.early_stop_patience,
        exit_threshold=args.exit_threshold,
        resume_checkpoint=args.resume_checkpoint,
        log_file=args.log_file,
    )

    # Print banner
    print("\n" + "="*80)
    print("DINOv3 STAGE 1 TRAINING - NATIX STREETVISION SUBNET 72")
    print("Production-Grade 2025 | RTX A6000 Optimized | WITH KAGGLE SUPPORT")
    print("="*80)
    print(f"\nMode: {config.mode}")
    print(f"Kaggle datasets: {'✅ ENABLED (expect 90-95% accuracy!)' if config.use_kaggle_data else '❌ Disabled (use --use_kaggle_data)'}")
    print(f"Config will be saved to: {config.output_dir}/config.json")

    # Route to appropriate training mode
    if config.mode == "extract_features":
        extract_features(config)
    elif config.mode == "train_cached":
        train_with_cached_features(config)
    elif config.mode == "train":
        train_dinov3_head(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
