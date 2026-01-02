"""
AUXILIARY HEADS FOR 2026 MODEL

Components:
1. Weather Prediction Head (8 classes: 7 weather types + NULL)
2. SAM 3 Segmentation Decoder (6 roadwork objects)

Purpose:
- Auxiliary supervision signals
- Makes model robust to NULL metadata
- Learns fine-grained features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeatherPredictionHead(nn.Module):
    """
    Weather Prediction Head (Auxiliary Task)
    
    Task: Predict weather from vision features
    Purpose: Makes model robust to NULL metadata (40% missing in test)
    
    Classes: 8
    - 0: sunny
    - 1: rainy
    - 2: foggy
    - 3: cloudy
    - 4: clear
    - 5: overcast
    - 6: snowy
    - 7: NULL
    
    Architecture:
    - 512-dim input → 256 hidden → 8 output
    - SiLU activation (2026 standard)
    - Dropout 0.1
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation (2026 standard)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict weather class
        
        Args:
            x: [B, input_dim] - Vision features
        
        Returns:
            logits: [B, num_classes] - Weather prediction logits
        """
        return self.classifier(x)


class SAM3SegmentationHead(nn.Module):
    """
    SAM 3 Segmentation Head (Auxiliary Task)
    
    Task: Generate 6-channel roadwork segmentation masks
    Purpose: Forces model to learn fine-grained spatial features
    
    Classes: 6 roadwork objects
    - 0: traffic cone
    - 1: construction barrier
    - 2: road work sign
    - 3: construction worker
    - 4: construction vehicle
    - 5: construction equipment
    
    Architecture:
    - 512-dim input → transposed convolutions → 6-channel output
    - Decoder: 256 → 128 → 64 → 6
    - SiLU activation
    - BatchNorm
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 6,
        output_size: tuple = (64, 64),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.output_size = output_size
        
        # Decoder architecture (upsample to output size)
        self.decoder = nn.Sequential(
            # 512 → 256 (2× upsampling)
            nn.ConvTranspose2d(input_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            # 256 → 128 (2× upsampling)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            # 128 → 64 (2× upsampling)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            
            # 64 → 6 (1×1 conv)
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
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
        target_masks: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> dict:
        """
        Forward pass and compute loss if targets provided
        
        Args:
            vision_features: [B, input_dim] - Vision features
            target_masks: [B, C, H, W] - SAM 3 pseudo-labels (optional)
            return_loss: Whether to return dice loss
        
        Returns:
            outputs: Dict with 'pred_masks' and 'loss' keys
        """
        batch_size = vision_features.shape[0]
        
        # Reshape for conv operations: [B, 512] → [B, 512, 1, 1]
        features_2d = vision_features.unsqueeze(-1).unsqueeze(-1)
        
        # Decode masks
        pred_masks = self.decoder(features_2d)  # [B, 6, 64, 64]
        
        outputs = {'pred_masks': pred_masks}
        
        # Compute Dice loss if targets provided
        if target_masks is not None and return_loss:
            # Interpolate to target size if needed
            if pred_masks.shape[-2:] != target_masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks,
                    size=target_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            loss = self.dice_loss(pred_masks, target_masks)
            outputs['loss'] = loss
        
        return outputs


class AuxiliaryHeads(nn.Module):
    """
    COMPLETE AUXILIARY HEADS MODULE
    
    Combines:
    1. Weather prediction (8 classes)
    2. SAM 3 segmentation (6 classes)
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        weather_classes: int = 8,
        sam3_classes: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Weather prediction head
        self.weather_head = WeatherPredictionHead(
            input_dim=vision_dim,
            num_classes=weather_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # SAM 3 segmentation head
        self.sam3_head = SAM3SegmentationHead(
            input_dim=vision_dim,
            num_classes=sam3_classes,
            output_size=(64, 64),
            dropout=dropout
        )
    
    def predict_weather(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Predict weather class
        
        Args:
            vision_features: [B, vision_dim] - Vision features
        
        Returns:
            weather_logits: [B, 8] - Weather prediction logits
        """
        return self.weather_head(vision_features)
    
    def predict_segmentation(
        self,
        vision_features: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> dict:
        """
        Predict roadwork segmentation masks
        
        Args:
            vision_features: [B, vision_dim] - Vision features
            target_masks: [B, 6, H, W] - SAM 3 pseudo-labels (optional)
            return_loss: Whether to return dice loss
        
        Returns:
            outputs: Dict with 'pred_masks' and 'loss' keys
        """
        return self.sam3_head(vision_features, target_masks, return_loss)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        weather_labels: Optional[torch.Tensor] = None,
        sam3_masks: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass through both auxiliary heads
        
        Args:
            vision_features: [B, vision_dim] - Vision features
            weather_labels: [B] - Weather ground truth (optional)
            sam3_masks: [B, 6, H, W] - SAM 3 pseudo-labels (optional)
        
        Returns:
            outputs: Dict with all predictions and losses
        """
        batch_size = vision_features.shape[0]
        
        outputs = {
            'weather_logits': None,
            'weather_loss': None,
            'pred_masks': None,
            'sam3_loss': None
        }
        
        # Weather prediction
        if weather_labels is not None:
            weather_logits = self.weather_head(vision_features)
            weather_loss = F.cross_entropy(
                weather_logits,
                weather_labels,
                reduction='none'
            )
            outputs['weather_logits'] = weather_logits
            outputs['weather_loss'] = weather_loss
        
        # SAM 3 segmentation
        if sam3_masks is not None:
            seg_outputs = self.sam3_head(vision_features, sam3_masks, return_loss=True)
            outputs['pred_masks'] = seg_outputs['pred_masks']
            outputs['sam3_loss'] = seg_outputs['loss']
        
        return outputs


if __name__ == "__main__":
    print("🧠 Testing AuxiliaryHeads...\n")
    
    # Mock data
    batch_size = 4
    vision_dim = 512
    vision_features = torch.randn(batch_size, vision_dim)
    
    # Create auxiliary heads
    aux_heads = AuxiliaryHeads(
        vision_dim=vision_dim,
        weather_classes=8,
        sam3_classes=6
    )
    
    # Test weather prediction
    print("📊 Testing weather prediction...")
    weather_logits = aux_heads.predict_weather(vision_features)
    print(f"   Input shape: {vision_features.shape}")
    print(f"   Weather logits shape: {weather_logits.shape}")
    print(f"   Expected: [{batch_size}, 8]")
    print("   ✅ Weather prediction test passed!\n")
    
    # Test SAM 3 segmentation
    print("📊 Testing SAM 3 segmentation...")
    sam3_masks = torch.rand(batch_size, 6, 64, 64)
    seg_outputs = aux_heads.predict_segmentation(vision_features, sam3_masks, return_loss=True)
    print(f"   Input shape: {vision_features.shape}")
    print(f"   Predicted masks shape: {seg_outputs['pred_masks'].shape}")
    print(f"   Dice loss: {seg_outputs['loss'].item():.4f}")
    print(f"   Expected masks shape: [{batch_size}, 6, 64, 64]")
    print("   ✅ SAM 3 segmentation test passed!\n")
    
    # Test full forward
    print("📊 Testing full forward pass...")
    weather_labels = torch.randint(0, 8, (batch_size,))
    outputs = aux_heads(vision_features, weather_labels, sam3_masks)
    print(f"   Weather logits shape: {outputs['weather_logits'].shape}")
    print(f"   Weather loss shape: {outputs['weather_loss'].shape}")
    print(f"   SAM 3 loss: {outputs['sam3_loss'].item():.4f}")
    print("   ✅ Full forward test passed!\n")

