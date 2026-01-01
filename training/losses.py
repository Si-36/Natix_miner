"""
Complete Loss Function with 4 Components
Focal Loss + Multi-View Consistency + Metadata Prediction + SAM 3 Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (γ=2.0)"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiViewConsistencyLoss(nn.Module):
    """Encourage consistency across selected top-8 views"""
    def __init__(self):
        super().__init__()
    
    def forward(self, view_features, view_indices):
        """
        Args:
            view_features: [B, 8, D] features from top-8 views
            view_indices: [B, 8] indices of selected views
        """
        B, N, D = view_features.shape
        
        # Compute pairwise similarities
        view_features_norm = F.normalize(view_features, dim=-1)
        similarity_matrix = torch.bmm(view_features_norm, view_features_norm.transpose(1, 2))
        
        # Target: high similarity between spatially adjacent views
        # Adjacent views should have indices close to each other
        adjacency_target = torch.exp(-torch.abs(view_indices.unsqueeze(2) - view_indices.unsqueeze(1)).float() / 2.0)
        adjacency_target = adjacency_target / adjacency_target.sum(dim=2, keepdim=True)
        
        # KL divergence loss
        consistency_loss = F.kl_div(
            F.log_softmax(similarity_matrix, dim=2),
            adjacency_target,
            reduction='batchmean'
        )
        
        return consistency_loss


class MetadataPredictionLoss(nn.Module):
    """Auxiliary loss for metadata prediction (GPS, weather, daytime, scene)"""
    def __init__(self, metadata_dims):
        super().__init__()
        self.losses = nn.ModuleDict({
            'gps': nn.MSELoss(),
            'weather': nn.CrossEntropyLoss(),
            'daytime': nn.CrossEntropyLoss(),
            'scene': nn.CrossEntropyLoss()
        })
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'gps', 'weather', 'daytime', 'scene'
            targets: dict with 'gps', 'weather', 'daytime', 'scene'
        """
        total_loss = 0.0
        
        # GPS regression
        if predictions['gps'] is not None and targets['gps'] is not None:
            gps_loss = self.losses['gps'](predictions['gps'], targets['gps'])
            total_loss += gps_loss * 0.1  # Lower weight for GPS
        
        # Weather classification
        if predictions['weather'] is not None and targets['weather'] is not None:
            weather_loss = self.losses['weather'](predictions['weather'], targets['weather'])
            total_loss += weather_loss * 0.3
        
        # Daytime classification
        if predictions['daytime'] is not None and targets['daytime'] is not None:
            daytime_loss = self.losses['daytime'](predictions['daytime'], targets['daytime'])
            total_loss += daytime_loss * 0.2
        
        # Scene classification
        if predictions['scene'] is not None and targets['scene'] is not None:
            scene_loss = self.losses['scene'](predictions['scene'], targets['scene'])
            total_loss += scene_loss * 0.2
        
        return total_loss


class SAM3SegmentationLoss(nn.Module):
    """Dice loss for SAM 3 semantic segmentation (6 classes)"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice coefficient loss"""
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def forward(self, pred_masks, target_masks):
        """
        Args:
            pred_masks: [B, 6, H, W] predicted segmentation masks
            target_masks: [B, H, W] target class indices
        """
        return self.dice_loss(pred_masks, target_masks)


class CompleteLoss(nn.Module):
    """Complete composite loss with all 4 components"""
    def __init__(self, num_classes=2, metadata_dims=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=2.0)
        self.view_consistency_loss = MultiViewConsistencyLoss()
        
        if metadata_dims:
            self.metadata_loss = MetadataPredictionLoss(metadata_dims)
        else:
            self.metadata_loss = None
        
        self.sam_loss = SAM3SegmentationLoss(num_classes=6)
        
        # Loss weights (tuned for roadwork detection)
        self.weights = {
            'focal': 1.0,           # Main classification
            'view_consistency': 0.2, # Multi-view alignment
            'metadata': 0.15,       # Auxiliary prediction
            'sam': 0.3              # Segmentation supervision
        }
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'logits', 'view_features', 'view_indices', 
                     'metadata_pred', 'sam_masks'
            targets: dict with 'labels', 'metadata', 'sam_masks', 'view_indices'
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. Focal Loss (main classification)
        focal_loss = self.focal_loss(outputs['logits'], targets['labels'])
        total_loss += self.weights['focal'] * focal_loss
        loss_dict['focal'] = focal_loss.item()
        
        # 2. Multi-View Consistency Loss
        if 'view_features' in outputs and 'view_indices' in outputs:
            view_loss = self.view_consistency_loss(
                outputs['view_features'],
                outputs['view_indices']
            )
            total_loss += self.weights['view_consistency'] * view_loss
            loss_dict['view_consistency'] = view_loss.item()
        
        # 3. Metadata Prediction Loss (auxiliary)
        if self.metadata_loss and 'metadata_pred' in outputs:
            metadata_loss = self.metadata_loss(
                outputs['metadata_pred'],
                targets['metadata']
            )
            total_loss += self.weights['metadata'] * metadata_loss
            loss_dict['metadata'] = metadata_loss.item()
        
        # 4. SAM 3 Segmentation Loss
        if 'sam_masks' in outputs and targets['sam_masks'] is not None:
            sam_loss = self.sam_loss(outputs['sam_masks'], targets['sam_masks'])
            total_loss += self.weights['sam'] * sam_loss
            loss_dict['sam'] = sam_loss.item()
        
        return total_loss, loss_dict


# ============== USAGE EXAMPLE ==============
"""
# Initialize complete loss
complete_loss = CompleteLoss(
    num_classes=2,
    metadata_dims={
        'gps': 2,          # lat, lon
        'weather': 5,     # 5 weather conditions
        'daytime': 4,      # 4 time periods
        'scene': 6         # 6 scene types
    }
)

# Forward pass
outputs = {
    'logits': classification_logits,      # [B, 2]
    'view_features': top8_features,        # [B, 8, D]
    'view_indices': selected_indices,     # [B, 8]
    'metadata_pred': metadata_predictions, # dict
    'sam_masks': sam_segmentation_masks     # [B, 6, H, W]
}

targets = {
    'labels': ground_truth_labels,         # [B]
    'metadata': metadata_targets,          # dict
    'sam_masks': sam_ground_truth,         # [B, H, W]
    'view_indices': None
}

# Compute loss
total_loss, loss_dict = complete_loss(outputs, targets)

# loss_dict contains individual component losses:
# {'focal': 0.234, 'view_consistency': 0.123, 'metadata': 0.089, 'sam': 0.156}
print(f"Total Loss: {total_loss.item():.4f}")
print(f"Loss Breakdown: {loss_dict}")
"""

