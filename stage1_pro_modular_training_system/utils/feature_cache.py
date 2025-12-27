"""
Feature caching utilities preserving exact logic from train_stage1_head.py

Extract and cache DINOv3 features for 10x faster training iterations.
"""

import os
import torch
from typing import Tuple


def extract_features(backbone, dataset, cached_features_dir: str, split_name: str, device: str = "cuda"):
    """
    Extract features preserving exact logic from train_stage1_head.py.
    
    Args:
        backbone: DINOv3 backbone model
        dataset: Dataset to extract features from
        cached_features_dir: Directory to save features
        split_name: Split name ("train" or "val")
        device: Device
    
    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=64,  # Default batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            # Extract CLS token features
            outputs = backbone(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    # Concatenate and save
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    features_path = os.path.join(cached_features_dir, f"{split_name}_features.pt")
    labels_path = os.path.join(cached_features_dir, f"{split_name}_labels.pt")
    
    torch.save(features_tensor, features_path)
    torch.save(labels_tensor, labels_path)
    
    return features_tensor, labels_tensor


def load_cached_features(cached_features_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load cached features preserving exact logic.
    
    Args:
        cached_features_dir: Directory containing cached features
    
    Returns:
        Tuple of (train_features, train_labels, val_features, val_labels)
    """
    train_features = torch.load(os.path.join(cached_features_dir, "train_features.pt"))
    train_labels = torch.load(os.path.join(cached_features_dir, "train_labels.pt"))
    val_features = torch.load(os.path.join(cached_features_dir, "val_features.pt"))
    val_labels = torch.load(os.path.join(cached_features_dir, "val_labels.pt"))
    
    return train_features, train_labels, val_features, val_labels
