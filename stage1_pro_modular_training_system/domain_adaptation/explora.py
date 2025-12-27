"""
ExPLoRA (Extended Parameter-Efficient Low-Rank Adaptation) - PHASE 4 ONLY

Parameter-efficient extended pretraining for domain adaptation.
Unfreeze last 1-2 blocks, PEFT on rest.
MAE-style masked modeling on unlabeled road images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm


class MAEDecoder(nn.Module):
    """
    MAE decoder for predicting masked patches.
    
    Phase 4.2: Simple decoder that takes encoded visible patches
    and predicts masked patch features.
    """
    
    def __init__(self, hidden_size: int, decoder_dim: int = 512, decoder_depth: int = 2):
        """
        Initialize MAE decoder.
        
        Args:
            hidden_size: Input hidden size (from backbone)
            decoder_dim: Decoder hidden dimension
            decoder_depth: Number of decoder layers
        """
        super().__init__()
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        
        # Projection from backbone hidden_size to decoder_dim
        self.proj = nn.Linear(hidden_size, decoder_dim)
        
        # Decoder layers (simple MLP)
        layers = []
        for _ in range(decoder_depth):
            layers.extend([
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.LayerNorm(decoder_dim)
            ])
        self.decoder = nn.Sequential(*layers)
        
        # Prediction head (predict patch features)
        self.pred_head = nn.Linear(decoder_dim, hidden_size)
    
    def forward(
        self,
        encoded_patches: torch.Tensor,
        mask_indices: torch.Tensor,
        visible_indices: torch.Tensor,
        num_patches: int
    ) -> torch.Tensor:
        """
        Decode masked patches.
        
        Args:
            encoded_patches: Encoded visible patches [B, num_visible, hidden_size]
            mask_indices: Indices of masked patches [B, num_masked]
            visible_indices: Indices of visible patches [B, num_visible]
            num_patches: Total number of patches per image
        
        Returns:
            Predicted patch features [B, num_masked, hidden_size]
        """
        B = encoded_patches.shape[0]
        device = encoded_patches.device
        
        # Project encoded patches to decoder dimension
        x = self.proj(encoded_patches)  # [B, num_visible, decoder_dim]
        
        # Create full sequence with mask tokens
        # Initialize with mask tokens
        full_seq = self.mask_token.expand(B, num_patches, -1)  # [B, num_patches, decoder_dim]
        
        # Fill in visible patches using scatter
        # Use advanced indexing to place visible patches
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, visible_indices.shape[1])
        full_seq[batch_indices, visible_indices] = x
        
        # Decode
        decoded = self.decoder(full_seq)  # [B, num_patches, decoder_dim]
        
        # Extract masked patches only
        batch_indices_mask = torch.arange(B, device=device).unsqueeze(1).expand(-1, mask_indices.shape[1])
        masked_decoded = decoded[batch_indices_mask, mask_indices]  # [B, num_masked, decoder_dim]
        
        # Predict patch features
        pred_patches = self.pred_head(masked_decoded)  # [B, num_masked, hidden_size]
        
        return pred_patches


class ExPLoRATrainer:
    """
    ExPLoRA trainer for Phase 4.
    
    Parameter-efficient extended pretraining:
    - Unfreeze last 1-2 transformer blocks
    - Apply PEFT (DoRAN/LoRA) to remaining blocks
    - MAE-style masked image modeling objective
    - Train on unlabeled road images (5-10 epochs)
    """
    
    def __init__(
        self,
        backbone,
        device: str = "cuda",
        unfreeze_blocks: int = 2,
        peft_type: str = "doran",
        peft_r: int = 16,
        mask_ratio: float = 0.75
    ):
        """
        Initialize ExPLoRA trainer.
        
        Args:
            backbone: DINOv3 backbone
            device: Device
            unfreeze_blocks: Number of last blocks to unfreeze
            peft_type: PEFT type ("doran", "dora", "lora")
            peft_r: PEFT rank
            mask_ratio: Ratio of patches to mask (default: 0.75)
        """
        self.backbone = backbone
        self.device = device
        self.unfreeze_blocks = unfreeze_blocks
        self.peft_type = peft_type
        self.peft_r = peft_r
        self.mask_ratio = mask_ratio
        
        # Phase 4.2: Create MAE decoder
        hidden_size = backbone.config.hidden_size
        self.decoder = MAEDecoder(hidden_size=hidden_size).to(device)
        
        # Phase 4.3: Setup unfreezing
        # Phase 4.4-4.6: Setup PEFT
        self._setup_backbone()
    
    def _setup_backbone(self):
        """
        Phase 4.3: Setup backbone unfreezing and PEFT.
        
        Unfreezes last N transformer blocks, keeps rest frozen.
        PEFT will be applied to frozen blocks in Phase 4.4-4.6.
        """
        if self.backbone is None:
            return
        
        # Phase 4.3: Find transformer encoder blocks
        # DINOv3 structure: backbone.encoder.layer (list of blocks)
        encoder = getattr(self.backbone, 'encoder', None)
        if encoder is None:
            print("⚠️  Could not find encoder in backbone. Skipping block unfreezing.")
            return
        
        # Get transformer blocks
        # Try different possible attribute names
        blocks = None
        if hasattr(encoder, 'layer'):
            blocks = encoder.layer
        elif hasattr(encoder, 'blocks'):
            blocks = encoder.blocks
        elif hasattr(encoder, 'layers'):
            blocks = encoder.layers
        else:
            print("⚠️  Could not find transformer blocks. Skipping block unfreezing.")
            return
        
        if not isinstance(blocks, torch.nn.ModuleList):
            print("⚠️  Transformer blocks not in ModuleList format. Skipping block unfreezing.")
            return
        
        total_blocks = len(blocks)
        print(f"\n📊 DINOv3 Backbone Structure:")
        print(f"   Total transformer blocks: {total_blocks}")
        print(f"   Unfreezing last {self.unfreeze_blocks} blocks")
        print(f"   Keeping first {total_blocks - self.unfreeze_blocks} blocks frozen")
        
        # Phase 4.3: Freeze all blocks first
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # Phase 4.3: Unfreeze last N blocks
        if self.unfreeze_blocks > 0:
            unfrozen_blocks = blocks[-self.unfreeze_blocks:]
            for block_idx, block in enumerate(unfrozen_blocks):
                block_num = total_blocks - self.unfreeze_blocks + block_idx
                for param in block.parameters():
                    param.requires_grad = True
                
                # Count trainable parameters in this block
                trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
                print(f"   ✅ Unfrozen block {block_num}: {trainable_params/1e6:.2f}M trainable params")
        
        # Count total trainable parameters
        total_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        
        print(f"\n📊 Parameter Summary:")
        print(f"   Trainable: {total_trainable/1e6:.2f}M params")
        print(f"   Frozen: {total_frozen/1e6:.2f}M params")
        print(f"   Trainable ratio: {100*total_trainable/(total_trainable+total_frozen):.2f}%")
        
        # Set backbone to train mode for unfrozen blocks
        # But we'll control this during training
        self.backbone.train()  # Will be set to eval() for frozen blocks during forward
    
    def _random_masking(
        self,
        num_patches: int,
        batch_size: int,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 4.2: Randomly mask patches.
        
        Args:
            num_patches: Number of patches per image
            batch_size: Batch size
            device: Device
        
        Returns:
            Tuple of (mask_indices [B, num_masked], visible_indices [B, num_visible])
        """
        num_masked = int(num_patches * self.mask_ratio)
        num_visible = num_patches - num_masked
        
        mask_indices_list = []
        visible_indices_list = []
        
        for _ in range(batch_size):
            # Random permutation
            indices = torch.randperm(num_patches, device=device)
            mask_indices = indices[:num_masked]
            visible_indices = indices[num_masked:]
            
            mask_indices_list.append(mask_indices)
            visible_indices_list.append(visible_indices)
        
        mask_indices = torch.stack(mask_indices_list, dim=0)  # [B, num_masked]
        visible_indices = torch.stack(visible_indices_list, dim=0)  # [B, num_visible]
        
        return mask_indices, visible_indices
    
    def _extract_patch_features(
        self,
        pixel_values: torch.Tensor,
        visible_indices: torch.Tensor,
        training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 4.2: Extract patch features from visible patches.
        
        Phase 4.3: Supports training mode for unfrozen blocks.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            visible_indices: Indices of visible patches [B, num_visible]
            training: Whether in training mode (affects unfrozen blocks)
        
        Returns:
            Tuple of (encoded_patches [B, num_visible, hidden_size], 
                     all_patch_features [B, num_patches, hidden_size] for target)
        """
        # Phase 4.3: Set backbone mode appropriately
        # Unfrozen blocks need training mode, frozen blocks stay in eval
        if training and self.unfreeze_blocks > 0:
            # Set backbone to train mode (unfrozen blocks will train)
            self.backbone.train()
        else:
            # All frozen, use eval mode
            self.backbone.eval()
        
        # Forward through backbone
        # Phase 4.3: Unfrozen blocks will compute gradients if training=True
        if training and self.unfreeze_blocks > 0:
            outputs = self.backbone(pixel_values=pixel_values)
        else:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        
        # Get all patch embeddings (exclude CLS token at index 0)
        # DINOv3 outputs: last_hidden_state [B, num_patches+1, hidden_size]
        # Index 0 is CLS token, indices 1: are patch tokens
        all_patch_features = outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_size]
        
        # Extract visible patches only
        B = all_patch_features.shape[0]
        encoded_patches = []
        for i in range(B):
            encoded_patches.append(all_patch_features[i, visible_indices[i]])
        encoded_patches = torch.stack(encoded_patches, dim=0)  # [B, num_visible, hidden_size]
        
        return encoded_patches, all_patch_features
    
    def pretrain(
        self,
        unlabeled_dataset,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-4,
        grad_accum_steps: int = 1
    ):
        """
        Phase 4.2: Run ExPLoRA pretraining with MAE objective.
        
        Args:
            unlabeled_dataset: Unlabeled road image dataset
            epochs: Number of pretraining epochs
            batch_size: Batch size
            lr: Learning rate
            grad_accum_steps: Gradient accumulation steps
        
        Returns:
            Adapted backbone
        """
        # Create dataloader
        dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup optimizer (only decoder for Phase 4.2)
        optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=lr,
            weight_decay=0.05
        )
        
        # Loss function (MSE for patch reconstruction)
        criterion = nn.MSELoss()
        
        # Compute number of patches (DINOv3 ViT-H/16: 224/16 = 14 patches per side)
        # Total: 14 * 14 = 196 patches
        img_size = 224
        patch_size = 16  # DINOv3 ViT-H/16
        num_patches = (img_size // patch_size) ** 2
        
        print(f"\n{'='*60}")
        print(f"ExPLoRA Pretraining (Phase 4.2)")
        print(f"{'='*60}")
        print(f"Mask ratio: {self.mask_ratio:.1%}")
        print(f"Num patches per image: {num_patches}")
        print(f"Num masked patches: {int(num_patches * self.mask_ratio)}")
        print(f"Num visible patches: {int(num_patches * (1 - self.mask_ratio))}")
        print(f"Training decoder only (backbone frozen)")
        print(f"{'='*60}\n")
        
        # Phase 4.3: Setup training mode
        # Decoder always trains
        self.decoder.train()
        
        # Phase 4.3: Setup optimizer for decoder + unfrozen blocks
        trainable_params = list(self.decoder.parameters())
        if self.unfreeze_blocks > 0:
            # Add unfrozen backbone parameters to optimizer
            encoder = getattr(self.backbone, 'encoder', None)
            if encoder:
                blocks = getattr(encoder, 'layer', None) or getattr(encoder, 'blocks', None) or getattr(encoder, 'layers', None)
                if blocks and isinstance(blocks, torch.nn.ModuleList):
                    unfrozen_blocks = blocks[-self.unfreeze_blocks:]
                    for block in unfrozen_blocks:
                        trainable_params.extend(block.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=0.05
        )
        
        print(f"✅ Optimizer setup:")
        print(f"   Decoder parameters: {sum(p.numel() for p in self.decoder.parameters())/1e6:.2f}M")
        if self.unfreeze_blocks > 0:
            unfrozen_backbone_params = sum(p.numel() for p in trainable_params if p not in self.decoder.parameters())
            print(f"   Unfrozen backbone parameters: {unfrozen_backbone_params/1e6:.2f}M")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, pixel_values in enumerate(pbar):
                pixel_values = pixel_values.to(self.device)
                B = pixel_values.shape[0]
                
                # Phase 4.2: Random masking
                mask_indices, visible_indices = self._random_masking(
                    num_patches=num_patches,
                    batch_size=B,
                    device=self.device
                )
                
                # Phase 4.2: Extract patch features
                # Phase 4.3: Pass training=True to enable gradients for unfrozen blocks
                encoded_patches, all_patch_features = self._extract_patch_features(
                    pixel_values=pixel_values,
                    visible_indices=visible_indices,
                    training=True  # Enable gradients for unfrozen blocks
                )
                
                # Phase 4.2: Decode masked patches
                pred_patches = self.decoder(
                    encoded_patches=encoded_patches,
                    mask_indices=mask_indices,
                    visible_indices=visible_indices,
                    num_patches=num_patches
                )  # [B, num_masked, hidden_size]
                
                # Phase 4.2: Extract target masked patches
                target_patches = []
                for i in range(B):
                    target_patches.append(all_patch_features[i, mask_indices[i]])
                target_patches = torch.stack(target_patches, dim=0)  # [B, num_masked, hidden_size]
                
                # Phase 4.2: Compute reconstruction loss (MSE)
                loss = criterion(pred_patches, target_patches)
                loss = loss / grad_accum_steps
                
                # Backward
                loss.backward()
                
                # Update every grad_accum_steps
                if (batch_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * grad_accum_steps
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item() * grad_accum_steps:.4f}',
                    'avg_loss': f'{epoch_loss / num_batches:.4f}'
                })
            
            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Average MAE Loss: {avg_loss:.4f}")
        
        print(f"\n✅ ExPLoRA pretraining complete!")
        
        # Set decoder to eval mode
        self.decoder.eval()
        
        return self.backbone
    
    def save_checkpoint(self, path: str):
        """
        Save ExPLoRA checkpoint (decoder weights).
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'mask_ratio': self.mask_ratio,
            'unfreeze_blocks': self.unfreeze_blocks,
            'peft_type': self.peft_type,
            'peft_r': self.peft_r
        }, path)
        print(f"✅ Saved ExPLoRA checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load ExPLoRA checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"✅ Loaded ExPLoRA checkpoint from {path}")
