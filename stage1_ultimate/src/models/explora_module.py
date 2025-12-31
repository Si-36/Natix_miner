"""
ExPLoRA SimCLR Lightning Module (2025 Best Practices)
=======================================================

PyTorch Lightning module for ExPLoRA domain adaptation with SimCLR.

Features:
- SimCLR contrastive learning (unsupervised)
- DDP all-gather with sync_grads=True (multi-GPU scaling)
- Strong augmentations (color jitter, blur, grayscale)
- BF16 mixed precision support
- torch.compile compatible

2025 Best Practices:
- Vectorized SimCLR loss (no Python loops)
- DDP all-gather for large negative pool
- Effective batch size = batch_size × num_gpus × 2 (views)
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision.transforms.v2 as v2
from omegaconf import DictConfig

from explora_peft.domain import SimCLRLoss, create_projection_head
from models.backbone import create_dinov3_backbone

logger = logging.getLogger(__name__)


class ExPLoRAModule(L.LightningModule):
    """
    ExPLoRA SimCLR Lightning Module (2025 Best Practices)
    
    Domain adaptation via contrastive learning:
    - Unsupervised pretraining on NATIX domain
    - SimCLR loss with strong augmentations
    - LoRA adapters (rank-32) on Q,V only
    - Unfreeze last block + all LayerNorms
    
    Args:
        config: Hydra configuration (config.phase4a)
        backbone: Optional pre-loaded backbone (if None, loads from config)
    
    Example:
        >>> module = ExPLoRAModule(config)
        >>> trainer = L.Trainer(devices=2, strategy="ddp")
        >>> trainer.fit(module, datamodule)
    """
    
    def __init__(
        self,
        config: DictConfig,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Load backbone (or use provided)
        if backbone is None:
            backbone_name = config.model.backbone_id
            self.backbone = create_dinov3_backbone(
                model_name=backbone_name,
                freeze=False,  # Will be frozen by ExPLoRA config
            )
        else:
            self.backbone = backbone
        
        # Apply ExPLoRA configuration (LoRA + unfrozen blocks)
        from explora_peft.domain import (
            create_explora_config,
            apply_explora_to_backbone,
        )
        
        lora_config, unfrozen_blocks = create_explora_config(
            backbone_name=config.model.backbone_id,
            lora_rank=config.explora.r,
            lora_alpha=config.explora.lora_alpha,
            target_modules=config.explora.get("target_modules", None),
            lora_dropout=config.explora.get("lora_dropout", 0.05),
            unfrozen_blocks=config.explora.get("unfrozen_blocks", None),
        )
        
        self.backbone = apply_explora_to_backbone(
            backbone=self.backbone,
            lora_config=lora_config,
            unfrozen_blocks=unfrozen_blocks,
        )
        
        # Get backbone output dimension
        if hasattr(self.backbone, "hidden_size"):
            hidden_dim = self.backbone.hidden_size
        elif hasattr(self.backbone, "config"):
            hidden_dim = self.backbone.config.hidden_size
        else:
            # Default for DINOv3 ViT-H
            hidden_dim = 1536
            logger.warning(f"Could not detect hidden_dim, using default {hidden_dim}")
        
        # Projection head (SimCLR standard: 2-layer MLP)
        projection_dim = config.simclr.get("projection_dim", 128)
        self.projection_head = create_projection_head(
            input_dim=hidden_dim,
            hidden_dim=config.simclr.get("hidden_dim", 2048),
            output_dim=projection_dim,
            dropout=config.simclr.get("dropout", 0.1),
        )
        
        # SimCLR loss
        temperature = config.simclr.get("temperature", 0.1)
        self.criterion = SimCLRLoss(temperature=temperature)
        
        # Augmentation transforms (will be created in training_step)
        self._augment = None
        
        logger.info(
            f"✅ ExPLoRA module initialized: "
            f"hidden_dim={hidden_dim}, projection_dim={projection_dim}, "
            f"temperature={temperature}"
        )
    
    def _get_augmentation(self) -> v2.Compose:
        """Get SimCLR strong augmentation pipeline"""
        if self._augment is None:
            aug_cfg = self.config.get("augmentation", {})
            
            self._augment = v2.Compose([
                v2.RandomResizedCrop(
                    224,
                    scale=tuple(aug_cfg.get("crop_scale", [0.2, 1.0])),
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([
                    v2.ColorJitter(
                        brightness=aug_cfg.get("color_jitter_strength", 0.8),
                        contrast=aug_cfg.get("color_jitter_strength", 0.8),
                        saturation=aug_cfg.get("color_jitter_strength", 0.8),
                        hue=0.2,
                    )
                ], p=0.8),
                v2.RandomGrayscale(p=aug_cfg.get("grayscale_prob", 0.2)),
            ])
            
            # Add Gaussian blur if enabled
            if aug_cfg.get("gaussian_blur", True):
                blur_kernel = aug_cfg.get("blur_kernel_size", 23)
                blur_sigma = tuple(aug_cfg.get("blur_sigma", [0.1, 2.0]))
                self._augment.transforms.append(
                    v2.RandomApply([
                        v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
                    ], p=0.5)
                )
            
            # Normalization (always last)
            self._augment.transforms.append(
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        
        return self._augment
    
    def augment_views(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views for SimCLR.
        
        Args:
            images: [B, C, H, W] input images
        
        Returns:
            Tuple of (view1, view2) augmented images
        """
        augment = self._get_augmentation()
        
        # Apply augmentation twice (different random seeds)
        view1 = augment(images)
        view2 = augment(images)
        
        return view1, view2
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone → projection head.
        
        Args:
            images: [B, C, H, W] input images
        
        Returns:
            [B, projection_dim] embeddings
        """
        # Extract features from backbone
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(images)
        else:
            # HuggingFace DINOv3
            outputs = self.backbone(images)
            if hasattr(outputs, "last_hidden_state"):
                # Use CLS token
                features = outputs.last_hidden_state[:, 0]
            else:
                features = outputs
        
        # Project to embedding space
        embeddings = self.projection_head(features)
        
        return embeddings
    
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step with SimCLR contrastive learning.
        
        CRITICAL: Uses DDP all-gather with sync_grads=True for multi-GPU scaling.
        
        Args:
            batch: Dictionary with 'image' key
            batch_idx: Batch index
        
        Returns:
            Contrastive loss
        """
        images = batch["image"]  # [B, C, H, W]
        
        # Generate two augmented views
        x_i, x_j = self.augment_views(images)
        
        # Forward pass: backbone → projection head
        z_i = self.forward(x_i)  # [B, projection_dim]
        z_j = self.forward(x_j)  # [B, projection_dim]
        
        # DDP: All-gather embeddings from all GPUs (CRITICAL for large negatives)
        if self.trainer.world_size > 1:
            # CRITICAL: sync_grads=True enables gradient flow through all_gather!
            z_i_gathered = self.all_gather(z_i, sync_grads=True)  # [world_size, B, D]
            z_j_gathered = self.all_gather(z_j, sync_grads=True)
            
            # Flatten batch dimension: [world_size*B, D]
            z_i = z_i_gathered.flatten(0, 1)
            z_j = z_j_gathered.flatten(0, 1)
            
            # Log effective batch size
            effective_batch = z_i.shape[0] * 2  # ×2 for two views
            self.log("effective_batch_size", effective_batch, prog_bar=True)
        
        # Compute contrastive loss
        loss = self.criterion(z_i, z_j)
        
        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        
        return loss
    
    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and scheduler (cosine with linear warmup).
        
        Returns:
            Dictionary with optimizer and scheduler config
        """
        # Optimizer: AdamW (best for vision transformers)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.get("weight_decay", 0.05),
            betas=(0.9, 0.999),
        )
        
        # Scheduler: Cosine annealing with linear warmup
        max_epochs = self.config.training.get("num_epochs", 30)
        warmup_epochs = int(max_epochs * 0.1)  # 10% warmup
        
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-6,
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
