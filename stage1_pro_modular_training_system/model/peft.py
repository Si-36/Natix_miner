"""
PEFT (Parameter-Efficient Fine-Tuning) module - PHASE 4 ONLY

Uses HuggingFace PEFT library (Dec 2025 latest) for DoRA, LoRA, and other adapters.
Falls back to custom implementation only if library not available.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import warnings

# Try to import HuggingFace PEFT library (Dec 2025 latest)
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        PeftConfig
    )
    from peft.tuners.lora import LoraLayer
    # Check if DoRA is available (newer versions)
    try:
        from peft import DoRAConfig
        DORA_AVAILABLE = True
    except ImportError:
        DORA_AVAILABLE = False
        warnings.warn("DoRA not available in PEFT library. Using LoRA fallback.")
    
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn(
        "HuggingFace PEFT library not found. "
        "Install with: pip install peft>=0.10.0\n"
        "Falling back to custom implementation."
    )


def apply_peft_with_hf(
    model: nn.Module,
    peft_type: str = "lora",
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION"
) -> nn.Module:
    """
    Apply PEFT using HuggingFace PEFT library (Dec 2025 best practice).
    
    Args:
        model: PyTorch model to apply PEFT to
        peft_type: "dora", "lora", or other PEFT types
        r: Rank for low-rank adaptation
        lora_alpha: Scaling factor (alpha)
        target_modules: List of module names to target (e.g., ["qkv", "dense"])
        lora_dropout: Dropout rate
        bias: Bias handling ("none", "all", "lora_only")
        task_type: Task type for PEFT
    
    Returns:
        Model with PEFT adapters applied
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "HuggingFace PEFT library required. Install with: pip install peft>=0.10.0"
        )
    
    # Default target modules for Vision Transformer (DINOv3)
    if target_modules is None:
        target_modules = [
            "query",
            "key",
            "value",
            "dense",  # Attention output
            "fc1",   # MLP input
            "fc2",   # MLP output
        ]
    
    # Map task type
    task_type_map = {
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
        "CLASSIFICATION": TaskType.SEQ_CLS,
    }
    peft_task_type = task_type_map.get(task_type, TaskType.FEATURE_EXTRACTION)
    
    # Create PEFT config
    if peft_type.lower() == "dora" and DORA_AVAILABLE:
        # Use DoRA if available
        peft_config = DoRAConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=peft_task_type,
        )
    else:
        # Fallback to LoRA
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=peft_task_type,
        )
    
    # Apply PEFT to model
    model = get_peft_model(model, peft_config)
    
    return model


def apply_peft_to_blocks(
    model: nn.Module,
    peft_type: str = "lora",
    r: int = 16,
    target_blocks: int = 6,
    block_prefix: str = "encoder.layer"
) -> nn.Module:
    """
    Apply PEFT to specific transformer blocks using HuggingFace PEFT.
    
    Args:
        model: Model with transformer blocks
        peft_type: "dora" or "lora"
        r: Rank for adapters
        target_blocks: Number of blocks to apply PEFT to
        block_prefix: Prefix for block modules (e.g., "encoder.layer")
    
    Returns:
        Model with PEFT applied to specified blocks
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "HuggingFace PEFT library required. Install with: pip install peft>=0.10.0"
        )
    
    # Find transformer blocks
    blocks = []
    for name, module in model.named_modules():
        if block_prefix in name and isinstance(module, nn.Module):
            blocks.append((name, module))
    
    if not blocks:
        warnings.warn(f"Could not find blocks with prefix '{block_prefix}'. Applying PEFT to all modules.")
        return apply_peft_with_hf(model, peft_type=peft_type, r=r)
    
    # Select target blocks (last N blocks)
    total_blocks = len(blocks)
    target_start = max(0, total_blocks - target_blocks)
    target_block_names = [blocks[i][0] for i in range(target_start, total_blocks)]
    
    # Build target modules list
    target_modules = []
    for block_name in target_block_names:
        target_modules.extend([
            f"{block_name}.attention.self.query",
            f"{block_name}.attention.self.key",
            f"{block_name}.attention.self.value",
            f"{block_name}.attention.output.dense",
            f"{block_name}.intermediate.dense",  # MLP fc1
            f"{block_name}.output.dense",        # MLP fc2
        ])
    
    return apply_peft_with_hf(
        model,
        peft_type=peft_type,
        r=r,
        target_modules=target_modules
    )


def apply_peft(
    backbone: nn.Module,
    peft_type: str = "lora",
    r: int = 16,
    target_blocks: int = 6
) -> nn.Module:
    """
    Apply PEFT to backbone - PHASE 4.6 (Dec 2025 best practice).
    
    Uses HuggingFace PEFT library if available, falls back to custom implementation.
    
    Args:
        backbone: DINOv3 backbone model
        peft_type: "dora", "lora", or "none"
        r: Rank for adapters
        target_blocks: Number of blocks to apply PEFT to
    
    Returns:
        Modified backbone with PEFT adapters
    """
    if peft_type == "none":
        return backbone
    
    if PEFT_AVAILABLE:
        # Use HuggingFace PEFT library (Dec 2025 best practice)
        print(f"\n📊 Using HuggingFace PEFT library ({peft_type.upper()})")
        try:
            return apply_peft_to_blocks(
                backbone,
                peft_type=peft_type,
                r=r,
                target_blocks=target_blocks
            )
        except Exception as e:
            warnings.warn(
                f"Failed to apply PEFT with HuggingFace library: {e}\n"
                "Falling back to custom implementation."
            )
            # Fall through to custom implementation
    
    # Fallback: Custom implementation (kept for compatibility)
    warnings.warn(
        "Using custom PEFT implementation. "
        "For best results, install HuggingFace PEFT: pip install peft>=0.10.0"
    )
    return _apply_peft_custom(backbone, peft_type=peft_type, r=r, target_blocks=target_blocks)


def _apply_peft_custom(
    backbone: nn.Module,
    peft_type: str = "lora",
    r: int = 16,
    target_blocks: int = 6
) -> nn.Module:
    """
    Custom PEFT implementation (fallback only).
    
    This is kept for backward compatibility but HuggingFace PEFT is preferred.
    """
    # Import custom adapters only if needed
    from .peft_custom import apply_dora_custom, apply_lora_custom
    
    if peft_type == "dora":
        return apply_dora_custom(backbone, r=r, target_blocks=target_blocks)
    elif peft_type == "lora":
        return apply_lora_custom(backbone, r=r, target_blocks=target_blocks)
    else:
        raise ValueError(f"Unknown peft_type: {peft_type}")


def get_peft_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about PEFT adapters in model.
    
    Args:
        model: Model with PEFT adapters
    
    Returns:
        Dictionary with PEFT information
    """
    if not PEFT_AVAILABLE:
        return {"peft_available": False}
    
    try:
        if isinstance(model, PeftModel):
            config = model.peft_config
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            return {
                "peft_available": True,
                "is_peft_model": True,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
                "peft_config": str(config),
            }
        else:
            # Check if model has PEFT adapters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            return {
                "peft_available": True,
                "is_peft_model": False,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
            }
    except Exception as e:
        return {
            "peft_available": True,
            "error": str(e)
        }
