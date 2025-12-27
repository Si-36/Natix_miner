"""
Reproducibility Utilities - Dec 2025 Best Practices

Deterministic training with seed setting and TF32 precision.
Preserves exact reproducibility logic from train_stage1_head.py.
"""

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Dict, Optional


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility (Dec 2025 best practice).
    
    Dec 2025 Best Practices:
    - Set seeds BEFORE any other imports for consistent RNG state
    - Use random.seed(), np.random.seed(), torch.manual_seed(), torch.cuda.manual_seed_all()
    - Enable deterministic operations (slower but reproducible)
    
    Preserves exact seed setting logic from train_stage1_head.py.
    
    Args:
        seed: Random seed for reproducibility
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed (CPU)
    torch.manual_seed(seed)
    
    # Set PyTorch random seed (all GPUs)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic operations (Dec 2025 best practice)
    cudnn.deterministic = True
    cudnn.benchmark = False  # Important for reproducibility
    
    print(f"✅ Seed set to {seed} (deterministic mode enabled)")


def enable_tf32_precision():
    """
    Enable TF32 precision for faster training without significant precision loss.
    
    Dec 2025 Best Practice (PyTorch 3.0+):
    - torch.set_float32_matmul_precision('high') for faster training
    - torch.backends.cuda.matmul.allow_tf32 = True
    - torch.backends.cudnn.allow_tf32 = True
    
    Preserves exact TF32 precision logic from train_stage1_head.py.
    """
    # Set TF32 precision (Dec 2025 best practice)
    torch.set_float32_matmul_precision('high')
    
    # Enable TF32 in backends
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ TF32 precision enabled (faster training without significant precision loss)")


def save_seeds_to_config(config, seed: int, data_split_seed: Optional[int] = None) -> None:
    """
    Save all seeds to config.json for full reproducibility.
    
    Args:
        config: Configuration object
        seed: Training seed
        data_split_seed: Data split generation seed
    """
    # Add seeds to config
    config.seed = seed
    
    if data_split_seed is not None:
        config.data_split_seed = data_split_seed
    
    print(f"✅ Seeds saved to config:")
    print(f"   Training seed: {seed}")
    if data_split_seed:
        print(f"   Data split seed: {data_split_seed}")


def get_random_seed() -> int:
    """
    Generate random seed for current run.
    
    Returns:
        Random seed (for reproducibility documentation)
    """
    import time
    seed = int(time.time())  # Use timestamp as seed
    return seed


def get_seed_from_config(config) -> int:
    """
    Get training seed from config.
    
    Args:
        config: Configuration object
    
    Returns:
        Training seed
    """
    return getattr(config, 'seed', get_random_seed())


def enable_deterministic_mode():
    """
    Enable fully deterministic mode (slower but reproducible).
    
    Dec 2025 Best Practice:
    - Use torch.use_deterministic_algorithms(True, warn_only=True)
    - Only use if strict reproducibility is required
    """
    try:
        # Try to use deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("✅ Deterministic algorithms enabled")
    except AttributeError:
        # Older PyTorch versions may not have this
        print("⚠️  torch.use_deterministic_algorithms not available (older PyTorch version)")


def log_reproducibility_info(config) -> None:
    """
    Log reproducibility information for debugging.
    
    Args:
        config: Configuration object
    """
    print(f"\n{'='*80}")
    print(f"REPRODUCIBILITY INFO")
    print(f"{'='*80}")
    print(f"{'='*80}")
    print(f"Training seed: {config.seed}")
    if hasattr(config, 'data_split_seed'):
        print(f"Data split seed: {config.data_split_seed}")
    print(f"CUDNN deterministic: {cudnn.deterministic}")
    print(f"CUDNN benchmark: {cudnn.benchmark}")
    print(f"TF32 precision: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"{'='*80}")
