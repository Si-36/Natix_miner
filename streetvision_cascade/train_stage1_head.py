import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

class NATIXDataset(Dataset):
    """NATIX dataset loader for training Stage 1"""
    def __init__(self, image_dir, labels_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        # Load labels (CSV format: image_path,label)
        with open(labels_file, 'r') as f:
            self.samples = [line.strip().split(',') for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')

        # Process image for DINOv3
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        label = int(label)  # 0 = no roadwork, 1 = roadwork

        return pixel_values, label

def train_dinov3_head():
    """Train ONLY the classifier head, freeze DINOv3 backbone"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load DINOv3 backbone (FROZEN)
    print("\n[1/7] Loading DINOv3-vith16plus backbone...")
    model_path = "models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m"
    backbone = AutoModel.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    # FREEZE ALL backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    frozen_params = sum(p.numel() for p in backbone.parameters())
    print(f"✅ Frozen {frozen_params/1e6:.1f}M backbone parameters")

    # Create classifier head (TRAINABLE)
    hidden_size = backbone.config.hidden_size  # 1536 for vith16plus
    classifier_head = nn.Sequential(
        nn.Linear(hidden_size, 768),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(768, 2)  # Binary: [no_roadwork, roadwork]
    ).to(device)

    trainable_params = sum(p.numel() for p in classifier_head.parameters())
    print(f"✅ Training {trainable_params/1e3:.0f}K classifier parameters (0.02% of full model)")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        classifier_head.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # Load NATIX dataset
    print("\n[2/7] Loading NATIX dataset...")
    train_dataset = NATIXDataset(
        image_dir="data/natix_official/train",
        labels_file="data/natix_official/train_labels.csv",
        processor=processor
    )

    val_dataset = NATIXDataset(
        image_dir="data/natix_official/val",
        labels_file="data/natix_official/val_labels.csv",
        processor=processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Training loop (10 epochs)
    print("\n[3/7] Starting training (10 epochs, ~2-3 hours)...")
    best_acc = 0.0

    for epoch in range(10):
        # ===== TRAIN =====
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10 [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Extract features with FROZEN backbone
            with torch.no_grad():
                outputs = backbone(pixel_values=images)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token

            # Forward through TRAINABLE head
            logits = classifier_head(features)
            loss = criterion(logits, labels)

            # Backward (only head gradients)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        train_acc = 100. * train_correct / train_total

        # ===== VALIDATE =====
        classifier_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/10 [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = backbone(pixel_values=images)
                features = outputs.last_hidden_state[:, 0, :]
                logits = classifier_head(features)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        val_acc = 100. * val_correct / val_total

        print(f"\nEpoch {epoch+1}/10 Summary:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss/len(val_loader):.4f}, Val Acc:   {val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = f"{model_path}/classifier_head.pth"
            torch.save(classifier_head.state_dict(), checkpoint_path)
            print(f"  ✅ Saved best checkpoint (Val Acc={val_acc:.2f}%)")

    print(f"\n[7/7] Training complete!")
    print(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"📁 Checkpoint saved: {model_path}/classifier_head.pth")

if __name__ == "__main__":
    train_dinov3_head()
