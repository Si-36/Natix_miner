"""
REAL PEFT Training - Dec 2025 Production Implementation

ACTUAL HuggingFace PEFT library usage in training loop:
- Use PEFTBackboneAdapter for adapter application
- Train only PEFT parameters (efficient)
- Save/load adapters properly
- Merge for inference (zero overhead)

This is NOT a wrapper - it's real PEFT integration.

Example Usage:
    from transformers import AutoModel
    from model.peft_integration import apply_lora_to_backbone
    from training.peft_real_trainer import RealPEFTTrainer
    
    # Load backbone
    backbone = AutoModel.from_pretrained("facebook/dinov3-vith14")
    
    # Apply LoRA
    adapted_backbone = apply_lora_to_backbone(backbone, r=16)
    
    # Create trainer with PEFT backbone
    trainer = RealPEFTTrainer(
        backbone=adapted_backbone,
        model=gate_head,
        train_loader=train_loader,
        val_select_loader=val_select_loader,
        val_calib_loader=val_calib_loader,
        config=config,
        device="cuda"
    )
    
    # Train
    results = trainer.train()
    
    # Save adapters
    trainer.save_adapters("output/adapters")
    
    # Merge for inference
    merged_backbone = trainer.merge_and_unload()
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any, Union
from pathlib import Path
from tqdm import tqdm

from model.peft_integration import PEFTBackboneAdapter, count_peft_parameters
from training.ema import EMA
from utils.logging import CSVLogger
from utils.checkpointing import save_checkpoint, load_checkpoint


class RealPEFTTrainer:
    """
    REAL PEFT Trainer with ACTUAL HuggingFace library usage.
    
    Training loop that:
    1. Uses PEFT-adapted backbone (LoRA/DoRA)
    2. Only optimizes PEFT parameters (efficient)
    3. Saves adapter-only checkpoints
    4. Supports merge_and_unload for inference
    
    Key differences from regular trainer:
    - Uses PeftModel backbone (with adapters)
    - Optimizer only gets PEFT params (not frozen backbone)
    - Checkpoints save adapters (small files)
    - Supports merge_and_unload() for zero-overhead inference
    """
    
    def __init__(
        self,
        backbone: Union[nn.Module, "PeftModel"],
        model: nn.Module,
        train_loader,
        val_select_loader,
        val_calib_loader,
        config: Any,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Initialize REAL PEFT trainer.
        
        Args:
            backbone: PeftModel (adapted with LoRA/DoRA) or regular nn.Module
            model: Gate head (classifier + gate + aux)
            train_loader: Training data loader
            val_select_loader: Validation data loader (model selection)
            val_calib_loader: Calibration data loader (threshold sweep)
            config: Training configuration
            device: Device (cuda/cpu)
            verbose: Print status messages
        """
        self.backbone = backbone.to(device)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_select_loader = val_select_loader
        self.val_calib_loader = val_calib_loader
        self.config = config
        self.device = device
        self.verbose = verbose
        
        # Check if backbone is PEFT model
        from peft import PeftModel
        self.is_peft_model = isinstance(backbone, PeftModel)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"REAL PEFT TRAINER (Dec 2025 Production)")
            print(f"{'='*80}")
            print(f"PEFT Model: {self.is_peft_model}")
            if self.is_peft_model:
                # Count PEFT parameters
                peft_params = count_peft_parameters(backbone)
                print(f"Trainable Params: {peft_params['trainable']:,} ({peft_params['trainable']/1e6:.2f}M)")
                print(f"Total Params: {peft_params['total']:,} ({peft_params['total']/1e6:.2f}M)")
                print(f"Trainable Ratio: {100*peft_params['trainable_ratio']:.2f}%")
            print(f"{'='*80}\n")
        
        # Create optimizer with PEFT parameters only
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=getattr(config, 'learning_rate_min', 1e-6)
        )
        
        # EMA
        self.ema = None
        if hasattr(config, 'use_ema') and config.use_ema:
            self.ema = EMA(
                self.model,
                decay=getattr(config, 'ema_decay', 0.9999),
                device=device
            )
            if self.verbose:
                print(f"EMA enabled: decay={config.ema_decay}")
    
    def _create_optimizer(self) -> AdamW:
        """
        Create optimizer with PEFT parameters only.
        
        If backbone is PEFT model:
        - Only optimize PEFT adapter parameters
        - Frozen backbone parameters are NOT included
        
        If backbone is regular model:
        - Optimize all parameters (full fine-tuning)
        
        Returns:
            AdamW optimizer
        """
        if self.is_peft_model:
            # REAL PEFT: Only optimize adapter parameters
            # This is the key efficiency gain of PEFT
            trainable_params = [
                p for p in self.backbone.parameters()
                if p.requires_grad
            ]
            
            if self.verbose:
                print(f"\n📊 Creating optimizer with PEFT parameters only:")
                print(f"   PEFT trainable params: {sum(p.numel() for p in trainable_params):,}")
            
            # Add model parameters (always trainable)
            model_params = list(self.model.parameters())
            
            # Combine PEFT + model parameters
            all_trainable_params = trainable_params + model_params
            
            optimizer = AdamW(
                all_trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            # Regular model: optimize all parameters
            if self.verbose:
                print(f"\n📊 Creating optimizer with ALL parameters:")
                print(f"   Total params: {sum(p.numel() for p in self.backbone.parameters()):,}")
            
            all_params = list(self.backbone.parameters()) + list(self.model.parameters())
            
            optimizer = AdamW(
                all_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        return optimizer
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train with PEFT (real training loop).
        
        Args:
            num_epochs: Number of epochs (overrides config)
        
        Returns:
            Training results dictionary
        """
        epochs = num_epochs or self.config.num_epochs
        
        best_metric = 0.0
        best_epoch = 0
        patience_counter = 0
        
        print(f"\n{'='*80}")
        print(f"TRAINING: {epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 80)
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Checkpointing
            if val_acc > best_metric:
                best_metric = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                if self.verbose:
                    print(f"✅ New best metric: {best_metric:.4f} at epoch {best_epoch+1}")
                
                # Save checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    best_metric=best_metric
                )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= getattr(self.config, 'early_stop_patience', 10):
                print(f"\n⏸️  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best Metric: {best_metric:.4f} at epoch {best_epoch+1}")
        print(f"{'='*80}\n")
        
        return {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "final_epoch": epoch + 1
        }
    
    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (train_loss, train_acc)
        """
        self.backbone.train()
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            if isinstance(batch, dict):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
            else:
                pixel_values, labels = batch
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)
            
            # Forward
            # Backbone with PEFT adapters
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Model (gate head)
            model_outputs = self.model(features)
            logits = model_outputs['classifier_logits']
            
            # Loss
            from torch.nn.functional import cross_entropy
            loss = cross_entropy(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config, 'max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            # Step
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        acc = 100.0 * correct / total
        
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%")
        
        return avg_loss, acc
    
    def validate(self, epoch: int) -> tuple:
        """
        Validate on val_select.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (val_loss, val_acc)
        """
        self.backbone.eval()
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_select_loader, desc=f"Val Epoch {epoch+1}")
            for batch in pbar:
                # Move to device
                if isinstance(batch, dict):
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    pixel_values, labels = batch
                    pixel_values = pixel_values.to(self.device)
                    labels = labels.to(self.device)
                
                # Forward
                outputs = self.backbone(pixel_values=pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
                
                model_outputs = self.model(features)
                logits = model_outputs['classifier_logits']
                
                # Loss
                from torch.nn.functional import cross_entropy
                loss = cross_entropy(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_select_loader)
        acc = 100.0 * correct / total
        
        print(f"Val Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")
        
        return avg_loss, acc
    
    def save_checkpoint(self, epoch: int, best_metric: float):
        """
        Save checkpoint (adapter-only if PEFT model).
        
        Args:
            epoch: Current epoch
            best_metric: Best metric value
        """
        output_dir = Path(getattr(self.config, 'output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_dir / "checkpoint_best.pth"
        
        if self.is_peft_model:
            # PEFT model: Save adapter-only checkpoint
            # This is much smaller than full model checkpoint
            adapter_dir = output_dir / "adapters"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            
            # REAL HuggingFace library call
            self.backbone.save_pretrained(str(adapter_dir))
            
            # Save head state dict
            head_state_dict = self.model.state_dict()
            
            checkpoint = {
                "head_state_dict": head_state_dict,
                "epoch": epoch,
                "best_metric": best_metric,
                "is_peft": True
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            if self.verbose:
                print(f"✅ Saved PEFT checkpoint:")
                print(f"   Adapters: {adapter_dir}")
                print(f"   Head state dict: {checkpoint_path}")
        else:
            # Regular model: Save full checkpoint
            checkpoint = {
                "backbone_state_dict": self.backbone.state_dict(),
                "head_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "is_peft": False
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            if self.verbose:
                print(f"✅ Saved full checkpoint: {checkpoint_path}")
    
    def save_adapters(self, save_directory: str):
        """
        Save PEFT adapters (REAL HuggingFace library call).
        
        Args:
            save_directory: Directory to save adapters
        """
        if not self.is_peft_model:
            warnings.warn("Backbone is not a PEFT model. Cannot save adapters.")
            return
        
        # REAL HuggingFace library call
        self.backbone.save_pretrained(save_directory)
        
        if self.verbose:
            print(f"✅ Saved PEFT adapters to {save_directory}")
    
    def merge_and_unload(self) -> nn.Module:
        """
        Merge adapters and unload PEFT wrapper (REAL library call).
        
        This eliminates adapter overhead for inference.
        
        Returns:
            Merged model (regular nn.Module)
        """
        if not self.is_peft_model:
            warnings.warn("Backbone is not a PEFT model. Nothing to merge.")
            return self.backbone
        
        # REAL HuggingFace library call
        merged_backbone = self.backbone.merge_and_unload()
        
        if self.verbose:
            print(f"✅ Merged adapters and unloaded PEFT wrapper")
            print(f"   Model is now regular nn.Module (zero inference overhead)")
        
        return merged_backbone


def create_real_peft_trainer(
    backbone_path: str,
    model: nn.Module,
    train_loader,
    val_select_loader,
    val_calib_loader,
    config: Any,
    peft_type: str = "lora",
    peft_r: int = 16,
    device: str = "cuda"
) -> RealPEFTTrainer:
    """
    Convenience function to create REAL PEFT trainer.
    
    Usage:
        trainer = create_real_peft_trainer(
            backbone_path="facebook/dinov3-vith14",
            model=gate_head,
            train_loader=train_loader,
            val_select_loader=val_select_loader,
            val_calib_loader=val_calib_loader,
            config=config,
            peft_type="lora",
            peft_r=16
        )
    
    Args:
        backbone_path: HuggingFace model path
        model: Gate head model
        train_loader: Training data loader
        val_select_loader: Validation data loader
        val_calib_loader: Calibration data loader
        config: Training configuration
        peft_type: "lora" or "dora"
        peft_r: PEFT rank
        device: Device
    
    Returns:
        RealPEFTTrainer instance
    """
    # Load backbone
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(backbone_path)
    
    # Apply PEFT
    from model.peft_integration import (
        PEFTBackboneAdapter,
        apply_lora_to_backbone,
        apply_dora_to_backbone
    )
    
    if peft_type.lower() == "dora":
        # Apply DoRA (Weight-Decomposed LoRA)
        adapted_backbone = apply_dora_to_backbone(backbone, r=peft_r)
    elif peft_type.lower() == "lora":
        # Apply LoRA
        adapted_backbone = apply_lora_to_backbone(backbone, r=peft_r)
    else:
        # No PEFT (full fine-tuning)
        adapted_backbone = backbone
    
    # Create trainer
    trainer = RealPEFTTrainer(
        backbone=adapted_backbone,
        model=model,
        train_loader=train_loader,
        val_select_loader=val_select_loader,
        val_calib_loader=val_calib_loader,
        config=config,
        device=device
    )
    
    return trainer

