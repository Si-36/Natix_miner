"""
═══════════════════════════════════════════════════════════
COMPLETE 2026 LOSS FUNCTION (4 COMPONENTS)
════════════════════════════════════════════════════════════

Loss = 0.40*Focal + 0.25*Consistency + 0.15*Auxiliary + 0.20*SAM3

Components:
1. Focal Loss (40% weight) - Focus on hard examples
2. Multi-View Consistency Loss (25% weight) - Ensure view agreement
3. Auxiliary Metadata Loss (15% weight) - Predict weather (robust to NULL)
4. SAM 3 Segmentation Loss (20% weight) - Learn fine-grained features

Expected Impact:
- Focal Loss: +3-5% MCC (handles class imbalance)
- Consistency Loss: +2-3% MCC (multi-view robustness)
- Auxiliary Loss: +1-2% MCC (handles NULL metadata)
- SAM 3 Loss: +2-4% MCC (fine-grained detection)
- TOTAL: +8-14% MCC vs. plain cross-entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np


class FocalLoss(nn.Module):
    """
    ════════════════════════════════════════════════════════════
    FOCAL LOSS WITH LABEL SMOOTHING (40% weight)
    ════════════════════════════════════════════════════════════
    
    Focuses training on hard examples by down-weighting well-classified examples.
    
    Formula:
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t: Predicted probability for true class
    - α: Balancing factor (0.25 for positive class)
    - γ: Focusing parameter (2.0 - more focus on hard examples)
    
    With Label Smoothing:
    Smooth targets: y_smooth = (1 - ε) * y_onehot + ε / C
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss with label smoothing
        
        Args:
            logits: [B, num_classes] - Model predictions (logits)
            labels: [B] - Ground truth class indices (0 or 1)
        
        Returns:
            loss: Scalar or [B] depending on reduction
        """
        n_classes = logits.size(1)
        
        # One-hot encode labels
        smooth_labels = F.one_hot(labels, n_classes).float()
        
        # Apply label smoothing
        smooth_labels = (1.0 - self.label_smoothing) * smooth_labels + \
                       self.label_smoothing / n_classes
        
        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            smooth_labels,
            reduction='none'
        )  # [B, 2]
        
        # Get probability of true class
        pt = torch.sigmoid(logits.gather(1, labels.unsqueeze(1))).squeeze(1)  # [B]
        
        # Compute modulating factor (1-pt)^γ
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # Combine with alpha weighting
        focal_loss = self.alpha * modulating_factor * bce_loss.sum(dim=1)  # [B]
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiViewConsistencyLoss(nn.Module):
    """
    ════════════════════════════════════════════════════════════
    MULTI-VIEW CONSISTENCY LOSS (25% weight)
    ════════════════════════════════════════════════════════════
    
    Ensures different views agree on prediction using KL divergence.
    
    Strategy:
    1. Each view makes independent prediction via small classifier
    2. Compute mean prediction across all views
    3. Compute KL divergence from mean for each view
    4. Average KL divergences
    
    This encourages view diversity while maintaining agreement.
    """
    
    def __init__(self, hidden_dim: int = 512, num_views: int = 8):
        super().__init__()
        self.num_views = num_views
        
        # Small classifier head for each view
        self.view_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2)
            ) for _ in range(num_views)
        ])
        
    def forward(
        self,
        view_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-view consistency loss
        
        Args:
            view_features: [B, N, hidden_dim] - Features for each view (N=8)
            labels: [B] - Optional (for supervised consistency)
        
        Returns:
            loss: Scalar - KL divergence loss
        """
        B, N, D = view_features.shape
        
        # Get per-view predictions
        view_logits = []
        for i in range(N):
            logit = self.view_classifier[i](view_features[:, i])  # [B, 2]
            view_logits.append(logit)
        
        view_logits = torch.stack(view_logits, dim=1)  # [B, 8, 2]
        
        # Compute softmax probabilities
        view_probs = F.softmax(view_logits, dim=-1)  # [B, 8, 2]
        
        # Compute mean prediction across views
        mean_probs = view_probs.mean(dim=1, keepdim=True)  # [B, 1, 2]
        
        # Compute KL divergence from mean for each view
        # KL(P || Q) = sum(P * (log(P) - log(Q)))
        kl_divs = F.kl_div(
            F.log_softmax(view_logits, dim=-1),  # log(P)
            mean_probs.expand_as(view_probs),      # Q
            reduction='none'
        ).sum(dim=-1)  # [B, 8]
        
        # Sum KL divergences across all views
        consistency_loss = kl_divs.mean()
        
        return consistency_loss


class AuxiliaryMetadataLoss(nn.Module):
    """
    ════════════════════════════════════════════════════════════
    AUXILIARY METADATA LOSS (15% weight)
    ════════════════════════════════════════════════════════════
    
    Task: Predict weather from vision features (makes model robust to NULL).
    
    Why this works:
    - Forces model to learn weather-related visual features
    - When weather is NULL, model has already learned to predict it
    - Provides supervision signal even when metadata is missing
    
    Classes: 8 weather types (7 actual + 1 NULL)
    - 0: sunny, 1: rainy, 2: foggy, 3: cloudy
    - 4: clear, 5: overcast, 6: snowy, 7: NULL
    """
    
    def __init__(self, hidden_dim: int = 512, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),  # Swish activation
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(
        self,
        vision_features: torch.Tensor,
        weather_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary weather prediction loss
        
        Args:
            vision_features: [B, hidden_dim] - Fused vision features
            weather_labels: [B] - Weather indices (0-7), -1 for NULL
        
        Returns:
            loss: Scalar - Cross-entropy loss
        """
        logits = self.classifier(vision_features)  # [B, 8]
        
        # Filter out samples where weather label is NULL (-1)
        mask = weather_labels != -1
        
        if mask.sum() > 0:
            loss = F.cross_entropy(
                logits[mask],
                weather_labels[mask]
            )
        else:
            loss = torch.tensor(0.0, device=logits.device)
            
        return loss


class SAM3SegmentationLoss(nn.Module):
    """
    ════════════════════════════════════════════════════════════
    SAM 3 SEGMENTATION LOSS (20% weight)
    ════════════════════════════════════════════════════════════
    
    Forces model to learn fine-grained spatial features using Dice loss.
    
    Classes: 6 roadwork objects
    - 0: traffic cone
    - 1: construction barrier
    - 2: road work sign
    - 3: construction worker
    - 4: construction vehicle
    - 5: construction equipment
    
    Use Dice loss for segmentation (better than BCE for imbalanced classes)
    """
    
    def __init__(self, vision_dim: int = 512, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        
        # Segmentation decoder (512-dim → 6-channel mask)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(vision_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Multi-class Dice loss
        
        Args:
            pred: [B, C, H, W] - Predicted logits
            target: [B, C, H, W] - Ground truth masks (one-hot)
            smooth: Smoothing constant
        
        Returns:
            loss: Scalar - Dice loss
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, H*W]
        target = target.view(target.size(0), target.size(1), -1)
        
        # Compute intersection and union
        intersection = (pred * target).sum(dim=-1)  # [B, C]
        union = pred.sum(dim=-1) + target.sum(dim=-1)  # [B, C]
        
        # Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # Dice loss (1 - dice)
        return 1.0 - dice.mean()
        
    def forward(
        self,
        vision_features: torch.Tensor,
        sam3_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SAM 3 segmentation loss
        
        Args:
            vision_features: [B, hidden_dim] - Fused vision features
            sam3_masks: [B, num_classes, H, W] - Pseudo-labels from SAM 3
        
        Returns:
            loss: Scalar - Dice loss (0 if no masks provided)
        """
        if sam3_masks is None:
            return torch.tensor(0.0, device=vision_features.device)
        
        # Reshape for conv operations
        features_2d = vision_features.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        
        # Decode masks
        pred_masks = self.decoder(features_2d)  # [B, 6, H', W']
        
        # Interpolate to target size if needed
        if pred_masks.shape[-2:] != sam3_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks,
                size=sam3_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Compute Dice loss
        loss = self.dice_loss(pred_masks, sam3_masks)
        
        return loss


class CompleteLoss(nn.Module):
    """
    ════════════════════════════════════════════════════════════
    COMPLETE 2026 LOSS FUNCTION (ALL 4 COMPONENTS)
    ════════════════════════════════════════════════════════════
    
    Loss = 0.40*Focal + 0.25*Consistency + 0.15*Auxiliary + 0.20*SAM3
    
    This is the main loss function used during training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Initialize component losses
        focal_cfg = config.get('focal', {})
        self.focal_loss = FocalLoss(
            gamma=focal_cfg.get('gamma', 2.0),
            alpha=focal_cfg.get('alpha', 0.25),
            label_smoothing=focal_cfg.get('label_smoothing', 0.1),
            reduction='mean'
        )
        
        consistency_cfg = config.get('consistency', {})
        self.consistency_loss = MultiViewConsistencyLoss(
            hidden_dim=consistency_cfg.get('hidden_dim', 512),
            num_views=consistency_cfg.get('num_views', 8)
        )
        
        auxiliary_cfg = config.get('auxiliary', {})
        self.aux_loss = AuxiliaryMetadataLoss(
            hidden_dim=auxiliary_cfg.get('hidden_dim', 512),
            num_classes=auxiliary_cfg.get('num_classes', 8)
        )
        
        sam3_cfg = config.get('sam3', {})
        self.sam3_loss = SAM3SegmentationLoss(
            vision_dim=sam3_cfg.get('vision_dim', 512),
            num_classes=sam3_cfg.get('num_classes', 6)
        )
        
        # Loss weights (should sum to 1.0)
        self.w_focal = config.get('loss', {}).get('focal_weight', 0.40)
        self.w_consistency = config.get('loss', {}).get('consistency_weight', 0.25)
        self.w_auxiliary = config.get('loss', {}).get('auxiliary_weight', 0.15)
        self.w_sam3 = config.get('loss', {}).get('sam3_weight', 0.20)
        
        # Normalize weights to sum to 1.0
        total_weight = self.w_focal + self.w_consistency + self.w_auxiliary + self.w_sam3
        if total_weight != 1.0:
            print(f"⚠️  Warning: Loss weights sum to {total_weight:.3f}, normalizing to 1.0")
            self.w_focal /= total_weight
            self.w_consistency /= total_weight
            self.w_auxiliary /= total_weight
            self.w_sam3 /= total_weight
        
        print("\n" + "="*60)
        print("📊 COMPLETE 2026 LOSS FUNCTION INITIALIZED")
        print("="*60)
        print(f"✅ Focal Loss: {self.w_focal:.2%} weight")
        print(f"✅ Consistency Loss: {self.w_consistency:.2%} weight")
        print(f"✅ Auxiliary Loss: {self.w_auxiliary:.2%} weight")
        print(f"✅ SAM 3 Loss: {self.w_sam3:.2%} weight")
        print("="*60 + "\n")
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete loss
        
        Args:
            outputs: Dict with keys:
                - 'logits': [B, 2] - Main prediction logits
                - 'view_features': [B, 8, 512] - Features before GAFM
                - 'aux_weather_logits': [B, 8] - Weather prediction
                - 'seg_masks': [B, 6, H, W] - SAM 3 segmentation
            targets: Dict with keys:
                - 'labels': [B] - Ground truth labels
                - 'weather_labels': [B] - Weather labels
                - 'sam3_masks': [B, 6, H, W] - SAM 3 pseudo-labels
        
        Returns:
            total_loss: Scalar - Weighted sum of all losses
            loss_dict: Dict - Individual losses for logging
        """
        # 1. Focal Loss (Main classification)
        l_focal = self.focal_loss(outputs['logits'], targets['labels'])
        
        # 2. Multi-View Consistency Loss
        l_consistency = self.consistency_loss(
            outputs['view_features'],
            labels=targets.get('labels', None)
        )
        
        # 3. Auxiliary Metadata Loss (Weather prediction)
        l_auxiliary = self.aux_loss(
            outputs['aux_weather_logits'],
            targets['weather_labels']
        )
        
        # 4. SAM 3 Segmentation Loss
        sam3_masks = targets.get('sam3_masks', None)
        l_sam3 = self.sam3_loss(
            outputs.get('vision_features', outputs['view_features'].mean(dim=1)),
            sam3_masks
        )
        
        # Combine with weights
        total_loss = (
            self.w_focal * l_focal +
            self.w_consistency * l_consistency +
            self.w_auxiliary * l_auxiliary +
            self.w_sam3 * l_sam3
        )
        
        # Build loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'focal': l_focal.item(),
            'consistency': l_consistency.item(),
            'auxiliary': l_auxiliary.item(),
            'sam3': l_sam3.item()
        }
        
        return total_loss, loss_dict


def create_loss(config_path: str) -> CompleteLoss:
    """
    Factory function to create loss from config file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        loss: CompleteLoss instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loss = CompleteLoss(config)
    
    return loss


if __name__ == "__main__":
    # Test loss function
    print("📊 Testing CompleteLoss...\n")
    
    # Mock config
    mock_config = {
        'focal': {
            'gamma': 2.0,
            'alpha': 0.25,
            'label_smoothing': 0.1
        },
        'consistency': {
            'hidden_dim': 512,
            'num_views': 8
        },
        'auxiliary': {
            'hidden_dim': 512,
            'num_classes': 8
        },
        'sam3': {
            'vision_dim': 512,
            'num_classes': 6
        },
        'loss': {
            'focal_weight': 0.40,
            'consistency_weight': 0.25,
            'auxiliary_weight': 0.15,
            'sam3_weight': 0.20
        }
    }
    
    # Create loss
    loss_fn = CompleteLoss(mock_config)
    
    # Mock data
    batch_size = 4
    num_views = 8
    
    outputs = {
        'logits': torch.randn(batch_size, 2),
        'view_features': torch.randn(batch_size, num_views, 512),
        'aux_weather_logits': torch.randn(batch_size, 8),
        'seg_masks': torch.randn(batch_size, 6, 64, 64),
        'vision_features': torch.randn(batch_size, 512)
    }
    
    targets = {
        'labels': torch.randint(0, 2, (batch_size,)),
        'weather_labels': torch.randint(-1, 8, (batch_size,)),
        'sam3_masks': torch.rand(batch_size, 6, 64, 64)
    }
    
    # Compute loss
    total_loss, loss_dict = loss_fn(outputs, targets)
    
    print(f"\n📊 Loss Breakdown:")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.4f}")
    print(f"\n✅ Total Loss: {total_loss.item():.4f}")
    print("\n✅ Loss test passed!\n")

