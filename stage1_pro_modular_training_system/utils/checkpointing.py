"""
Checkpointing Utilities - Dec 2025 Best Practices

Comprehensive checkpoint validation, saving, and loading with graceful error handling.
Preserves exact format from train_stage1_head.py with Dec 2025 enhancements.
"""

import torch
import os
from typing import Optional, Dict
from collections import OrderedDict


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc: float,
    patience_counter: int,
    ema: Optional[object],
    checkpoint_reason: str = "",
    verbose: bool = True
) -> None:
    """
    Save checkpoint with comprehensive state preservation (Dec 2025 best practice).
    
    Preserves exact format from train_stage1_head.py with enhancements:
    - model_state_dict
    - optimizer_state_dict
    - scheduler_state_dict
    - epoch
    - best_acc
    - patience_counter
    - ema_state_dict (if EMA enabled) - FIXED: Save as OrderedDict
    - timestamp (for versioning)
    - git_commit (for versioning)
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save state dict from
        optimizer: Optimizer to save state dict from
        scheduler: Scheduler to save state dict from
        epoch: Current epoch
        best_acc: Best accuracy so far
        patience_counter: Early stopping counter
        ema: EMA object (optional)
        checkpoint_reason: Reason for checkpoint save (Dec 2025 enhancement)
        verbose: Print status messages
    """
    # Create checkpoint directory if needed
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Build checkpoint dictionary (preserve exact format)
    checkpoint = {
        'epoch': epoch,
        'best_acc': best_acc,
        'patience_counter': patience_counter,
        'checkpoint_reason': checkpoint_reason,
    }
    
    # Save model state dict
    checkpoint['model_state_dict'] = model.state_dict()
    if verbose:
        print(f"   Model state dict: {len(checkpoint['model_state_dict'])} tensors")
    
    # Save optimizer state dict
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if verbose:
        print(f"   Optimizer state dict: {len(checkpoint['optimizer_state_dict'])} tensors")
    
    # Save scheduler state dict
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if verbose:
            print(f"   Scheduler state dict: {len(checkpoint['scheduler_state_dict'])} tensors")
    
    # FIX: Save EMA shadow state dict as OrderedDict (Dec 2025 best practice)
    if ema is not None:
        if hasattr(ema, 'shadow'):
            # EMA.shadow should be an OrderedDict or ParameterDict
            # Convert to OrderedDict for serialization
            if isinstance(ema.shadow, (dict, OrderedDict, torch.nn.ParameterDict)):
                checkpoint['ema_state_dict'] = OrderedDict(
                    (k, v.clone().cpu()) for k, v in ema.shadow.items()
                )
            else:
                # Fallback: treat as state dict
                checkpoint['ema_state_dict'] = ema.state_dict()
            if verbose:
                print(f"   EMA shadow state dict: {len(checkpoint['ema_state_dict'])} tensors")
        else:
            if verbose:
                print("   EMA object has no shadow attribute")
    
    # Add timestamp (Dec 2025 enhancement for versioning)
    from datetime import datetime
    checkpoint['timestamp'] = datetime.now().isoformat()
    if verbose:
        print(f"   Timestamp: {checkpoint['timestamp']}")
    
    # Add git commit (Dec 2025 enhancement for reproducibility)
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        if result.returncode == 0:
            checkpoint['git_commit'] = result.stdout.strip()
            if verbose:
                print(f"   Git commit: {checkpoint['git_commit']}")
        else:
            checkpoint['git_commit'] = "unknown"
            if verbose:
                print("   Git commit unknown (not in git repository)")
    except Exception as e:
        checkpoint['git_commit'] = "unknown"
        if verbose:
            print(f"   Git commit error: {e}")
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    if verbose:
        print(f"✅ Checkpoint saved to {checkpoint_path}")
        print(f"   Reason: {checkpoint_reason}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema: Optional[object] = None,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """
    Load checkpoint with comprehensive validation and graceful degradation (Dec 2025 best practice).
    
    Handles missing keys gracefully, provides recovery options.
    Validates required keys exist.
    Loads EMA shadow state (if EMA was used).
    
    Returns:
        dict: Checkpoint data with epoch, best_acc, patience_counter
    """
    # Validate checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if verbose:
        print(f"\n[1/3] Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Validate required keys (preserve exact format from train_stage1_head.py)
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_acc', 'patience_counter']
    missing_keys = []
    for key in required_keys:
        if key not in checkpoint:
            missing_keys.append(key)
            if verbose:
                print(f"   ⚠️  Missing required key: {key}")
    
    if missing_keys:
        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
    
    # Load model state (handle missing gracefully)
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            if verbose:
                print(f"✅ Model state dict loaded")
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load model state dict: {e}")
            raise ValueError(f"Failed to load model state dict: {e}")
    else:
        if verbose:
            print("⚠️  No model_state_dict in checkpoint (starting from scratch)")
    
    # Load optimizer state (handle missing gracefully)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if verbose:
                print(f"✅ Optimizer state dict loaded")
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load optimizer state dict: {e}")
            # Continue with random optimizer
    elif optimizer is not None:
        if verbose:
            print("⚠️  No optimizer state dict in checkpoint (using provided optimizer)")
    
    # Load scheduler state (handle missing gracefully)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if verbose:
                print(f"✅ Scheduler state dict loaded")
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load scheduler state dict: {e}")
            # Continue with fresh scheduler
    elif scheduler is not None:
        if verbose:
            print("⚠️  No scheduler state dict in checkpoint (using provided scheduler)")
    
    # FIX: Load EMA shadow state (handle missing gracefully)
    if ema is not None and 'ema_state_dict' in checkpoint:
        try:
            ema_shadow = checkpoint['ema_state_dict']
            if hasattr(ema, 'shadow'):
                if isinstance(ema.shadow, torch.nn.ParameterDict):
                    # Restore ParameterDict
                    for k, v in ema_shadow.items():
                        if k in ema.shadow:
                            ema.shadow[k] = v.to(device)
                elif isinstance(ema.shadow, (dict, OrderedDict)):
                    # Restore dict/OrderedDict
                    ema.shadow = OrderedDict(
                        (k, v.to(device)) for k, v in ema_shadow.items()
                    )
                else:
                    if verbose:
                        print(f"⚠️  Unknown EMA shadow format: {type(ema.shadow)}")
            else:
                if verbose:
                    print("⚠️  EMA object has no shadow attribute")
            if verbose:
                print(f"✅ EMA shadow state dict loaded")
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load EMA shadow: {e}")
            # Continue without EMA shadow
    elif ema is not None:
        if verbose:
            print("⚠️  No EMA shadow state dict in checkpoint (EMA not available)")
    
    # Extract training state
    training_state = {
        'epoch': checkpoint.get('epoch', 0),
        'best_acc': checkpoint.get('best_acc', 0.0),
        'patience_counter': checkpoint.get('patience_counter', 0),
        'checkpoint_reason': checkpoint.get('checkpoint_reason', ''),
        'timestamp': checkpoint.get('timestamp', ''),
        'git_commit': checkpoint.get('git_commit', '')
    }
    
    if verbose:
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Epoch: {training_state['epoch']}")
        print(f"   Best Acc: {training_state['best_acc']:.4f}")
        print(f"   Patience Counter: {training_state['patience_counter']}")
        print(f"   Reason: {training_state['checkpoint_reason']}")
        print(f"   Timestamp: {training_state['timestamp']}")
        print(f"   Git Commit: {training_state['git_commit']}")
        print(f"{'='*80}")
    
    return training_state


def validate_checkpoint(checkpoint_path: str, verbose: bool = True) -> Dict[str, bool]:
    """
    Validate checkpoint file without loading (Dec 2025 best practice).
    
    Args:
        checkpoint_path: Path to checkpoint file
        verbose: Print status messages
    
    Returns:
        dict: Validation results (exists, valid_format, file_size)
    """
    result = {
        'exists': False,
        'valid_format': False,
        'file_size_mb': 0.0
    }
    
    if not os.path.exists(checkpoint_path):
        result['exists'] = False
        if verbose:
            print(f"❌ Checkpoint does not exist: {checkpoint_path}")
        return result
    
    result['exists'] = True
    
    # Try to load checkpoint (to check format)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result['file_size_mb'] = os.path.getsize(checkpoint_path) / (1024 * 1024)
        
        # Check required keys (preserve exact format from train_stage1_head.py)
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_acc', 'patience_counter']
        missing_keys = []
        for key in required_keys:
            if key not in checkpoint:
                missing_keys.append(key)
        
        if missing_keys:
            result['valid_format'] = False
            if verbose:
                print(f"❌ Invalid format: Missing required keys: {missing_keys}")
        else:
            result['valid_format'] = True
            if verbose:
                print(f"✅ Valid format: Contains all required keys")
                print(f"   File size: {result['file_size_mb']:.2f} MB")
    
    except Exception as e:
        if verbose:
            print(f"❌ Invalid format: Failed to load checkpoint: {e}")
        result['valid_format'] = False
    
    return result


def get_latest_checkpoint(checkpoint_dir: str, verbose: bool = True) -> Optional[str]:
    """
    Find latest checkpoint in directory (Dec 2025 enhancement).
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        verbose: Print status messages
    
    Returns:
        Path to latest checkpoint (based on epoch number)
    """
    if not os.path.exists(checkpoint_dir):
        if verbose:
            print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # List all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
    
    if not checkpoint_files:
        if verbose:
            print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Sort by epoch number (extract from filename)
    checkpoint_files_with_epochs = []
    for f in checkpoint_files:
        # Extract epoch number (e.g., "checkpoint_epoch5.pth" -> 5)
        try:
            epoch_str = f.replace('checkpoint_epoch', '').replace('.pth', '')
            epoch = int(epoch_str)
            checkpoint_files_with_epochs.append((epoch, f))
        except ValueError:
            # Skip files that don't match expected pattern
            if verbose:
                print(f"⚠️  Skipping invalid checkpoint filename: {f}")
    
    # Sort by epoch (highest first)
    checkpoint_files_with_epochs.sort(key=lambda x: x[0], reverse=True)
    
    if not checkpoint_files_with_epochs:
        if verbose:
            print(f"❌ No valid checkpoint files found")
        return None
    
    # Return latest checkpoint
    latest_checkpoint = checkpoint_files_with_epochs[0][1]
    
    if verbose:
        print(f"✅ Latest checkpoint: {os.path.basename(latest_checkpoint)} (epoch {checkpoint_files_with_epochs[0][0]})")
    
    return os.path.join(checkpoint_dir, latest_checkpoint)
