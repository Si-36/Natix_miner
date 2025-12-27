"""
Trainer Module - Phase 1.6: Strict val_select Usage

Preserves exact logic from train_stage1_head.py while enforcing
strict split separation: val_select for model selection ONLY.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any
import os

from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.logging import CSVLogger
from data.loaders import pick_batch_size
from metrics.selective import compute_risk_coverage, compute_augrc, compute_selective_metrics
from model.gate_head import compute_selective_loss, compute_auxiliary_loss


class Stage1ProTrainer:
    """
    Stage-1 Professional Trainer (Phase 1.6: Strict val_select Usage)
    
    Enforces strict split separation:
    - val_select: Model selection/early stopping ONLY
    - val_calib: Fitting calibrators/policies ONLY (Phase 1.8+)
    - val_test: Final evaluation ONLY (Phase 2+)
    
    CRITICAL: Validation loop uses val_select_loader ONLY.
    """
    
    def __init__(
        self,
        model: nn.Module,
        backbone: nn.Module,
        train_loader,
        val_select_loader,
        val_calib_loader,
        config,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Initialize trainer (preserving exact structure from train_stage1_head.py).
        
        Args:
            model: Head model (classifier)
            backbone: DINOv3 backbone
            train_loader: Training data loader
            val_select_loader: Validation data loader (val_select - model selection ONLY!)
            val_calib_loader: Calibration data loader (val_calib - threshold sweep ONLY!)
            config: Training configuration
            device: Device (cuda/cpu)
            verbose: Print status messages
        """
        self.model = model.to(device)
        self.backbone = backbone.to(device)
        self.train_loader = train_loader
        self.val_select_loader = val_select_loader
        self.val_calib_loader = val_calib_loader  # CRITICAL FIX: Separate loader for calibration
        self.config = config
        self.device = device
        self.verbose = verbose
        
        # Phase 1.6: Enforce strict split usage (CRITICAL FIX)
        if verbose:
            print(f"\n{'='*80}")
            print(f"PHASE 1.6: STRICT SPLIT USAGE")
            print(f"{'='*80}")
            print(f"Validation: val_select_loader (model selection/early stopping)")
            print(f"Calibration: val_calib_loader (threshold sweep/policy fitting)")
            print(f"CRITICAL: val_select and val_calib are SEPARATE!")
            print(f"{'='*80}")
        
        # Optimizer (preserve exact setup from train_stage1_head.py)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler (preserve exact setup from train_stage1_head.py)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate_min
        )
        
        # EMA (if enabled in config)
        self.ema = None
        if config.use_ema:
            self.ema = EMA(
                self.model,
                decay=config.ema_decay,
                device=device
            )
            if verbose:
                print(f"EMA enabled: decay={config.ema_decay}")
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.patience_counter = 0
        self.best_val_acc = 0.0
        
        # CSV Logger (Phase 2: selective metrics + bootstrap CIs)
        self.logger = CSVLogger(
            log_file=os.path.join(config.output_dir, "training_log.csv"),
            phase=config.phase,
            include_selective_metrics=config.use_selective_metrics,
            include_gate_metrics=config.exit_policy == "gate",
            verbose=verbose
        )
        
        # Phase 2: Selective metrics state
        self.best_augrc = float('inf')  # Lower is better for AUGRC
        self.best_risk_at_coverage = float('inf')
        
        # Phase 3: Gate head state
        self.exit_policy = config.exit_policy  # 'softmax', 'gate', 'scrc'
        self.gate_threshold = config.gate_threshold if hasattr(config, 'gate_threshold') else 0.5
        self.gate_lower_threshold = config.gate_lower_threshold if hasattr(config, 'gate_lower_threshold') else None
        
        # Output directories
        self.checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if verbose:
            print(f"✅ Trainer initialized")
            print(f"   Optimizer: AdamW (lr={config.learning_rate})")
            print(f"   Scheduler: CosineAnnealingLR (T_max={config.num_epochs})")
            print(f"   EMA: {self.ema is not None}")
            print(f"   Checkpoint dir: {self.checkpoint_dir}")
    
    def train(self):
        """
        Main training loop (preserving exact logic from train_stage1_head.py).
        
        Phase 1.6: Uses val_select for validation/early stopping ONLY.
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TRAINING START")
            print(f"{'='*80}")
        
        # Phase 1.3: Pick batch size dynamically
        # Note: pick_batch_size will be called in training loop
        # after model is compiled and backbone is loaded
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            if self.verbose:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate on val_select (Phase 1.6: val_select ONLY)
            val_loss, val_acc, ece, exit_coverage, exit_acc, augrc_metrics = self.validate_epoch()
            
            # Log metrics (Phase 2.7: Include selective metrics + bootstrap CIs)
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                ece=ece,
                exit_coverage=exit_coverage,
                exit_acc=exit_acc,
                augrc=augrc_metrics if augrc_metrics else None,
                best_acc=self.best_val_acc,
                lr=self.scheduler.get_last_lr()[0]
            )
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_epoch{epoch}.pth"
            )
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_acc=self.best_val_acc,
                patience_counter=self.patience_counter,
                ema=self.ema,
                checkpoint_reason="end_of_epoch",
                verbose=self.verbose
            )
            
            # Checkpoint selection (Phase 2.5-2.6: based on AUGRC or val_acc)
            # Phase 1/2: Use val_acc if selective metrics disabled
            # Phase 2: Use AUGRC if selective metrics enabled
            
            checkpoint_reason = ""
            save_best = False
            
            if self.config.use_selective_metrics and augrc_metrics is not None:
                # Phase 2.5-2.6: Checkpoint selection by AUGRC (lower is better)
                current_augrc = augrc_metrics['augrc']
                
                if current_augrc < self.best_augrc:
                    self.best_augrc = current_augrc
                    self.patience_counter = 0
                    save_best = True
                    checkpoint_reason = "best_augrc"
                    
                    if self.verbose:
                        print(f"✅ Best AUGRC: {current_augrc:.6f} (saved model_best.pth)")
            else:
                # Phase 1: Checkpoint selection by val_acc
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.patience_counter = 0
                    save_best = True
                    checkpoint_reason = "best_val_acc"
                    
                    if self.verbose:
                        print(f"✅ Best val_acc: {val_acc:.4f} (saved model_best.pth)")
            
            if save_best:
                # Update best_val_acc for consistency
                if checkpoint_reason == "best_val_acc":
                    self.best_val_acc = val_acc
                
                # Save best checkpoint
                best_checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    "model_best.pth"
                )
                save_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_acc=self.best_val_acc,
                    patience_counter=self.patience_counter,
                    ema=self.ema,
                    checkpoint_reason=checkpoint_reason,
                    verbose=self.verbose
                )
            else:
                self.patience_counter += 1
            
            # Early stopping (preserve exact logic from train_stage1_head.py)
            if self.patience_counter >= self.config.patience:
                if self.verbose:
                    print(f"\n⏹️  Early stopping triggered (patience={self.config.patience})")
                break
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TRAINING COMPLETE")
            print(f"Best val_acc: {self.best_val_acc:.4f}")
            print(f"{'='*80}")
    
    def train_epoch(self) -> tuple:
        """
        Train one epoch (Phase 3.5: Gate Head Support).
        
        Returns:
            (train_loss, train_acc)
        """
        self.model.train()
        self.backbone.eval()  # Phase 1: backbone is frozen
        
        total_loss = 0.0
        total_classifier_loss = 0.0
        total_gate_loss = 0.0
        total_auxiliary_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.no_grad():
                features = self.backbone.extract_features(images)
            
            # Phase 3.5: Gate head training (3-head architecture) - CORRECTED
            if self.exit_policy == 'gate':
                # Get outputs from gate head
                outputs = self.model(features)  # Dict: classifier_logits, gate_logits, head_logits
                classifier_logits = outputs['classifier_logits']
                gate_logits = outputs['gate_logits']
                head_logits = outputs['head_logits']
                
                # CORRECTION: Use gate exit mask from config, NOT from gate's own output (circular)
                # Compute exit mask using configured gate threshold
                gate_prob = torch.sigmoid(gate_logits)
                if self.gate_lower_threshold is not None:
                    # Two-sided exit: gate >= upper OR gate <= lower
                    exit_mask = (gate_prob >= self.gate_threshold) | (gate_prob <= self.gate_lower_threshold)
                else:
                    # One-sided exit: gate >= upper
                    exit_mask = gate_prob >= self.gate_threshold
                
                # Compute selective loss (Phase 3.3) - CORRECTED
                selective_loss = compute_selective_loss(
                    classifier_logits,
                    gate_logits.unsqueeze(1),  # [N] -> [N, 1]
                    labels,
                    exit_mask,  # Used for coverage regularizer only
                    self.gate_threshold,
                    coverage_weight=1.0,
                    verbose=False
                )
                
                # Compute auxiliary loss (Phase 3.4)
                aux_loss = compute_auxiliary_loss(
                    head_logits,
                    labels,
                    verbose=False
                )
                
                # Total loss: selective + auxiliary
                loss = selective_loss + 0.1 * aux_loss  # Weight auxiliary loss lower
                
                # Track individual losses
                total_classifier_loss += nn.functional.cross_entropy(classifier_logits, labels).item()
                total_gate_loss += nn.functional.binary_cross_entropy(
                    gate_prob, exit_mask.float(), reduction='mean'
                ).item()
                total_auxiliary_loss += aux_loss.item()
                
                # Metrics
                _, predicted = classifier_logits.max(1)
            else:
                # Phase 1/2: Standard classifier (not gate head)
                outputs = self.model(features)
                loss = nn.functional.cross_entropy(outputs, labels)
                _, predicted = outputs.max(1)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)
            
            # Metrics
            total_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if self.verbose and batch_idx % 100 == 0:
                if self.exit_policy == 'gate':
                    print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.6f}, Sel={selective_loss.item():.6f}, Aux={aux_loss.item():.6f}")
                else:
                    print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.6f}")
        
        train_loss = total_loss / len(self.train_loader)
        train_acc = correct / total
        
        if self.verbose and self.exit_policy == 'gate':
            print(f"   Classifier Loss: {total_classifier_loss / len(self.train_loader):.6f}")
            print(f"   Gate Loss: {total_gate_loss / len(self.train_loader):.6f}")
            print(f"   Auxiliary Loss: {total_auxiliary_loss / len(self.train_loader):.6f}")
        
        return train_loss, train_acc
    
    def validate_epoch(self) -> tuple:
        """
        Validate one epoch (preserving exact logic from train_stage1_head.py).
        
        Phase 1.6 CRITICAL FIX:
        - val_select_loader: Model selection/early stopping
        - val_calib_loader: Threshold sweep/logits saving
        
        Phase 2: Selective metrics (AUGRC, risk-coverage, bootstrap CIs)
        
        Returns:
            (val_loss, val_acc, ece, exit_coverage, exit_acc, augrc_metrics)
        """
        self.model.eval()
        self.backbone.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Exit metrics (Phase 1: softmax threshold, Phase 3: gate head)
        exit_correct = 0
        exit_total = 0
        
        all_probs = []
        all_labels = []
        
        # Phase 1.7 CRITICAL FIX: Save calibration logits from val_calib_loader
        # Phase 3.6: Save gate logits for calibration
        # val_calib_loader is ALREADY indexed by val_calib_indices (via IndexedDataset)
        # Save WITHOUT double-indexing - threshold sweep will use tensors directly
        calib_logits = []
        calib_labels = []
        calib_gate_logits = []  # Phase 3.6
        
        # CRITICAL FIX: Validate on val_select_loader for model selection
        with torch.no_grad():
            for images, labels in self.val_select_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                features = self.backbone.extract_features(images)
                
                # Phase 3.5: Gate head support - CORRECTED
                if self.exit_policy == 'gate':
                    outputs = self.model(features)
                    classifier_logits = outputs['classifier_logits']
                    gate_logits = outputs['gate_logits']
                    
                    # Loss: Cross-entropy on classifier logits (correct - NOT softmax)
                    loss = nn.functional.cross_entropy(classifier_logits, labels)
                    total_loss += loss.item()
                    
                    # Metrics
                    _, predicted = classifier_logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Exit metrics (Phase 3: gate threshold)
                    gate_prob = torch.sigmoid(gate_logits)
                    if self.gate_lower_threshold is not None:
                        # Two-sided exit: gate >= upper OR gate <= lower
                        exit_mask = (gate_prob >= self.gate_threshold) | (gate_prob <= self.gate_lower_threshold)
                    else:
                        # One-sided exit: gate >= upper
                        exit_mask = gate_prob >= self.gate_threshold
                    
                    if exit_mask.any():
                        exit_correct += predicted[exit_mask].eq(labels[exit_mask]).sum().item()
                        exit_total += exit_mask.sum().item()
                    
                    # Store probs for selective metrics (Phase 2)
                    probs = torch.softmax(classifier_logits, dim=1)
                    all_probs.append(probs.cpu())
                    all_labels.append(labels.cpu())
                else:
                    # Phase 1/2: Standard classifier
                    outputs = self.model(features)
                    
                    # Loss
                    loss = nn.functional.cross_entropy(outputs, labels)
                    total_loss += loss.item()
                    
                    # Metrics
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # Exit metrics (Phase 1: softmax threshold)
                    probs = torch.softmax(outputs, dim=1)
                    max_probs, _ = probs.max(1)
                    
                    # Exit condition: max_prob >= 0.88 OR max_prob <= 0.12
                    exit_mask = (max_probs >= 0.88) | (max_probs <= 0.12)
                    
                    if exit_mask.any():
                        exit_correct += predicted[exit_mask].eq(labels[exit_mask]).sum().item()
                        exit_total += exit_mask.sum().item()
                    
                    # Store probs for selective metrics (Phase 2)
                    all_probs.append(probs.cpu())
                    all_labels.append(labels.cpu())
        
        # CRITICAL FIX: Save calibration logits from val_calib_loader
        # Phase 3.6: Save gate logits for calibration
        with torch.no_grad():
            for images, labels in self.val_calib_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                features = self.backbone.extract_features(images)
                
                # Phase 3.5: Gate head support
                if self.exit_policy == 'gate':
                    outputs = self.model(features)
                    classifier_logits = outputs['classifier_logits']
                    gate_logits = outputs['gate_logits']
                    
                    # Store logits/labels (Phase 1.7 - CRITICAL: from val_calib)
                    calib_logits.append(classifier_logits.cpu())
                    calib_labels.append(labels.cpu())
                    # Phase 3.6: Save gate logits for calibration
                    calib_gate_logits.append(gate_logits.cpu())
                else:
                    # Phase 1/2: Standard classifier
                    outputs = self.model(features)
                    calib_logits.append(outputs.cpu())
                    calib_labels.append(labels.cpu())
        
        # Compute metrics
        val_loss = total_loss / len(self.val_select_loader)
        val_acc = correct / total
        
        # Exit metrics
        exit_acc = exit_correct / exit_total if exit_total > 0 else 0.0
        exit_coverage = exit_total / total
        
        # ECE (Expected Calibration Error)
        ece = compute_ece(all_probs, all_labels)
        
        # Phase 2.1-2.4: Selective metrics (AUGRC, risk-coverage)
        augrc_metrics = None
        if self.config.use_selective_metrics and len(all_probs) > 0:
            import numpy as np
            from metrics.selective import compute_risk_coverage, compute_augrc, compute_bootstrap_cis
            
            # Concatenate probs/labels
            all_probs_cat = torch.cat(all_probs)
            all_labels_cat = torch.cat(all_labels)
            
            # Risk-coverage curve (Phase 2.1)
            thresholds = np.linspace(0.0, 1.0, 100)
            coverage_array, risk_array, _ = compute_risk_coverage(
                all_probs_cat, all_labels_cat, thresholds, self.device
            )
            
            # AUGRC (Phase 2.2)
            augrc_result = compute_augrc(coverage_array, risk_array, target_coverage=0.9)
            
            # Bootstrap CIs for AUGRC (Phase 2.4)
            # Resample to get distribution
            n_bootstrap = self.config.bootstrap_samples if hasattr(self.config, 'bootstrap_samples') else 1000
            augrc_samples = []
            
            for _ in range(n_bootstrap):
                # Resample indices with replacement
                indices = np.random.choice(len(all_labels_cat), size=len(all_labels_cat), replace=True)
                resampled_probs = all_probs_cat[indices]
                resampled_labels = all_labels_cat[indices]
                
                # Compute AUGRC for resampled data
                resampled_cov, resampled_risk, _ = compute_risk_coverage(
                    resampled_probs, resampled_labels, thresholds, self.device
                )
                resampled_augrc = compute_augrc(resampled_cov, resampled_risk, target_coverage=0.9)
                augrc_samples.append(resampled_augrc['augrc'])
            
            # Compute bootstrap CI for AUGRC
            augrc_bootstrap = np.array(augrc_samples)
            augrc_ci = compute_bootstrap_cis(
                augrc_bootstrap,
                n_bootstrap=len(augrc_samples),
                confidence=self.config.bootstrap_confidence if hasattr(self.config, 'bootstrap_confidence') else 0.95
            )
            
            augrc_metrics = {
                **augrc_result,
                **augrc_ci
            }
            
            if self.verbose:
                print(f"\n   Phase 2: Selective Metrics (val_select)")
                print(f"   AUGRC: {augrc_result['augrc']:.6f}")
                print(f"   Risk@90% Coverage: {augrc_result['risk_at_coverage_90']:.6f}")
                print(f"   AUGRC Bootstrap CI: [{augrc_ci['ci_lower']:.6f}, {augrc_ci['ci_upper']:.6f}]")
        else:
            augrc_metrics = {}
        
        # Phase 1.7 CRITICAL FIX: Save calibration logits from val_calib
        # Phase 3.6: Save gate logits for calibration
        if self.verbose:
            print(f"\n   Phase 1.7/3.6: Saving calibration logits (from val_calib)...")
            print(f"   Total calib logits: {sum(logits.size(0) for logits in calib_logits)}")
            if self.exit_policy == 'gate':
                print(f"   Total calib gate logits: {sum(logits.size(0) for logits in calib_gate_logits)}")
        
        # Save to file (Phase 1.7 - CRITICAL: val_calib for threshold sweep)
        calib_logits_path = os.path.join(self.config.output_dir, "val_calib_logits.pt")
        calib_labels_path = os.path.join(self.config.output_dir, "val_calib_labels.pt")
        
        torch.save(torch.cat(calib_logits), calib_logits_path)
        torch.save(torch.cat(calib_labels), calib_labels_path)
        
        if self.verbose:
            print(f"   Saved to: {calib_logits_path}")
            print(f"   Saved to: {calib_labels_path}")
            print(f"   ✅ Calibration logits saved from val_calib (Phase 1.7)")
        
        # Phase 3.6: Save gate logits for calibration
        if self.exit_policy == 'gate':
            calib_gate_logits_path = os.path.join(self.config.output_dir, "val_calib_gate_logits.pt")
            torch.save(torch.cat(calib_gate_logits), calib_gate_logits_path)
            if self.verbose:
                print(f"   Saved to: {calib_gate_logits_path}")
                print(f"   ✅ Gate logits saved from val_calib (Phase 3.6)")
        
        if self.verbose:
            print(f"   Val Loss (val_select): {val_loss:.6f}")
            print(f"   Val Acc (val_select): {val_acc:.4f}")
            print(f"   ECE: {ece:.6f}")
            print(f"   Exit Coverage: {exit_coverage:.4f}")
            print(f"   Exit Acc: {exit_acc:.4f}")
        
        return val_loss, val_acc, ece, exit_coverage, exit_acc, augrc_metrics


class EMA:
    """
    Exponential Moving Average (EMA) for model weights.
    
    Preserves exact EMA implementation from train_stage1_head.py.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cuda"):
        """
        Initialize EMA.
        
        Args:
            model: Model to apply EMA to
            decay: EMA decay rate (default 0.999)
            device: Device to store shadow weights
        """
        self.decay = decay
        self.device = device
        
        # Initialize shadow weights as ParameterDict (Phase 1.5 fix)
        self.shadow = torch.nn.ParameterDict({
            name: param.clone().detach().to(device)
            for name, param in model.named_parameters()
        })
        
        self.backup = {}
    
    def update(self, model: nn.Module):
        """
        Update EMA shadow weights.
        
        Args:
            model: Model to update shadow from
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """
        Apply shadow weights to model (for inference).
        
        Args:
            model: Model to apply shadow to
        """
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data.clone()
    
    def restore(self, model: nn.Module):
        """
        Restore original weights from backup.
        
        Args:
            model: Model to restore weights to
        """
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def compute_ece(probs, labels, n_bins: int = 15):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    # Simple ECE implementation
    if len(probs) == 0 or len(labels) == 0:
        return 0.0
    
    # TODO: Implement proper ECE computation
    return 0.0
