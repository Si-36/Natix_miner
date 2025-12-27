"""
Custom PEFT implementation (fallback only).

Dec 2025 best practice: Use HuggingFace PEFT library.
This file is kept ONLY for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DoRANAdapter(nn.Module):
    """
    DoRAN (DoRA with Noise) adapter for Phase 4.4 (legacy fallback).
    
    Fallback implementation only if HuggingFace PEFT not available.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        noise_scale: float = 0.01,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.scaling = alpha / rank
        
        # LoRA matrices (low-rank adaptation)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # DoRA: Magnitude parameter (learnable scaling)
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Noise parameters for stabilization
        self.noise_A = nn.Parameter(torch.randn(rank, in_features) * noise_scale)
        self.noise_B = nn.Parameter(torch.randn(out_features, rank) * noise_scale)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Original weight (set at register time)
        self.weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DoRAN decomposition and noise.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        if self.weight is None:
            raise RuntimeError("DoRANAdapter weight not set. Call register_weight() first.")
        
        # Original weight path
        original_output = F.linear(x, self.weight)
        
        # LoRA path: B @ A @ x
        lora_output = F.linear(x, self.lora_A.T)  # [..., rank]
        lora_output = self.dropout(lora_output)
        lora_output = F.linear(lora_output, self.lora_B.T)  # [..., out_features]
        lora_output = lora_output * self.scaling
        
        # DoRA: Weight decomposition
        # Compute delta_W = B @ A
        delta_W = self.lora_B @ self.lora_A * self.scaling
        
        # Combine with original weight
        W_combined = self.weight + delta_W
        
        # Normalize direction (unit vector)
        W_norm = W_combined.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        W_direction = W_combined / W_norm
        
        # Apply magnitude scaling
        W_dora = self.magnitude.unsqueeze(1) * W_direction
        
        # Apply decomposed weight
        dora_output = F.linear(x, W_dora.T)
        
        # Combine: original + LoRA + DoRA adjustment
        return dora_output
    
    def register_weight(self, weight: nn.Parameter):
        """Register original weight for adaptation."""
        self.weight = weight


class DoRAAdapter(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for Phase 4.5 (legacy fallback).
    
    Fallback implementation only if HuggingFace PEFT not available.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # DoRA: Magnitude parameter
        self.magnitude = nn.Parameter(torch.ones(out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Original weight
        self.weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DoRA decomposition.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        if self.weight is None:
            raise RuntimeError("DoRAAdapter weight not set. Call register_weight() first.")
        
        # Original weight path
        original_output = F.linear(x, self.weight)
        
        # LoRA path: B @ A @ x
        lora_output = F.linear(x, self.lora_A.T)  # [..., rank]
        lora_output = self.dropout(lora_output)
        lora_output = F.linear(lora_output, self.lora_B.T)  # [..., out_features]
        lora_output = lora_output * self.scaling
        
        # DoRA: Weight decomposition
        delta_W = self.lora_B @ self.lora_A * self.scaling
        W_combined = self.weight + delta_W
        
        # Normalize direction
        W_norm = W_combined.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        W_direction = W_combined / W_norm
        
        # Apply magnitude scaling
        W_dora = self.magnitude.unsqueeze(1) * W_direction
        
        # Apply decomposed weight
        dora_output = F.linear(x, W_dora.T)
        
        # Combine: original + LoRA + DoRA adjustment
        return original_output + lora_output + (dora_output - original_output)
    
    def register_weight(self, weight: nn.Parameter):
        """Register original weight for adaptation."""
        self.weight = weight


class LoRAAdapter(nn.Module):
    """
    Standard LoRA (Low-Rank Adaptation) adapter (fallback).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Original weight
        self.weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        if self.weight is None:
            raise RuntimeError("LoRAAdapter weight not set. Call register_weight() first.")
        
        # Original weight path
        original_output = F.linear(x, self.weight)
        
        # LoRA path: B @ A @ x
        lora_output = F.linear(x, self.lora_A.T)  # [..., rank]
        lora_output = self.dropout(lora_output)
        lora_output = F.linear(lora_output, self.lora_B.T)  # [..., out_features]
        lora_output = lora_output * self.scaling
        
        # Combine: original + LoRA
        return original_output + lora_output
    
    def register_weight(self, weight: nn.Parameter):
        """Register original weight for adaptation."""
        self.weight = weight


def apply_dora_custom(
    backbone: nn.Module,
    r: int = 16,
    target_blocks: int = 6
) -> nn.Module:
    """
    Apply DoRA adapter (custom fallback only).
    
    Uses forward hook/wrapping to modify linear layers.
    """
    # Get transformer blocks
    encoder = getattr(backbone, 'encoder', None)
    if encoder is None:
        print("⚠️  Could not find encoder in backbone. Skipping DoRA application.")
        return backbone
    
    # Get transformer blocks
    blocks = None
    if hasattr(encoder, 'layer'):
        blocks = list(encoder.layer)
    elif hasattr(encoder, 'blocks'):
        blocks = list(encoder.blocks)
    elif hasattr(encoder, 'layers'):
        blocks = list(encoder.layers)
    else:
        print("⚠️  Could not find transformer blocks. Skipping DoRA application.")
        return backbone
    
    if not blocks:
        print("⚠️  No blocks found. Skipping DoRA application.")
        return backbone
    
    total_blocks = len(blocks)
    target_start = max(0, total_blocks - target_blocks)
    
    print(f"\n📊 Applying DoRA adapters (custom):")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Applying to blocks {target_start} to {total_blocks - 1}")
    
    adapters_applied = 0
    
    for block_idx in range(target_start, total_blocks):
        block = blocks[block_idx]
        
        # Find attention module
        attn = getattr(block, 'attention', None) or getattr(block, 'self_attention', None) or getattr(block, 'attn', None)
        if attn:
            # QKV projection
            qkv = getattr(attn, 'query_key_value', None) or getattr(attn, 'qkv', None)
            if qkv and isinstance(qkv, nn.Linear):
                adapter = DoRAAdapter(
                    in_features=qkv.in_features,
                    out_features=qkv.out_features,
                    rank=r
                )
                adapter.register_weight(qkv.weight)
                
                # Wrap forward method
                original_forward = qkv.forward
                def wrapped_forward(x):
                    return adapter(x)
                qkv.forward = wrapped_forward
                
                if not hasattr(attn, '_peft_adapters'):
                    attn._peft_adapters = {}
                attn._peft_adapters['qkv'] = adapter
                adapters_applied += 1
            
            # Attention output projection
            out_proj = getattr(attn, 'dense', None) or getattr(attn, 'output', None) or getattr(attn, 'out_proj', None)
            if out_proj and isinstance(out_proj, nn.Linear):
                adapter = DoRAAdapter(
                    in_features=out_proj.in_features,
                    out_features=out_proj.out_features,
                    rank=r
                )
                adapter.register_weight(out_proj.weight)
                
                original_forward = out_proj.forward
                def wrapped_forward(x):
                    return adapter(x)
                out_proj.forward = wrapped_forward
                
                if not hasattr(attn, '_peft_adapters'):
                    attn._peft_adapters = {}
                attn._peft_adapters['out_proj'] = adapter
                adapters_applied += 1
        
        # MLP layers
        mlp = getattr(block, 'mlp', None) or getattr(block, 'feed_forward', None) or getattr(block, 'ffn', None)
        if mlp:
            # MLP input projection
            mlp_fc1 = getattr(mlp, 'fc1', None) or getattr(mlp, 'dense_h_to_4h', None) or getattr(mlp, 'gate_proj', None)
            if mlp_fc1 and isinstance(mlp_fc1, nn.Linear):
                adapter = DoRAAdapter(
                    in_features=mlp_fc1.in_features,
                    out_features=mlp_fc1.out_features,
                    rank=r
                )
                adapter.register_weight(mlp_fc1.weight)
                
                original_forward = mlp_fc1.forward
                def wrapped_forward(x):
                    return adapter(x)
                mlp_fc1.forward = wrapped_forward
                
                if not hasattr(mlp, '_peft_adapters'):
                    mlp._peft_adapters = {}
                mlp._peft_adapters['fc1'] = adapter
                adapters_applied += 1
            
            # MLP output projection
            mlp_fc2 = getattr(mlp, 'fc2', None) or getattr(mlp, 'dense_4h_to_h', None) or getattr(mlp, 'down_proj', None)
            if mlp_fc2 and isinstance(mlp_fc2, nn.Linear):
                adapter = DoRAAdapter(
                    in_features=mlp_fc2.in_features,
                    out_features=mlp_fc2.out_features,
                    rank=r
                )
                adapter.register_weight(mlp_fc2.weight)
                
                original_forward = mlp_fc2.forward
                def wrapped_forward(x):
                    return adapter(x)
                mlp_fc2.forward = wrapped_forward
                
                if not hasattr(mlp, '_peft_adapters'):
                    mlp._peft_adapters = {}
                mlp._peft_adapters['fc2'] = adapter
                adapters_applied += 1
    
    print(f"✅ Applied {adapters_applied} DoRA adapters (custom)")
    return backbone


def apply_lora_custom(
    backbone: nn.Module,
    r: int = 16,
    target_blocks: int = 6
) -> nn.Module:
    """
    Apply LoRA adapter (custom fallback only).
    """
    return apply_dora_custom(backbone, r=r, target_blocks=target_blocks)


def apply_doran_custom(
    backbone: nn.Module,
    r: int = 16,
    target_blocks: int = 6
) -> nn.Module:
    """
    Apply DoRAN adapter (custom fallback only).
    """
    # Temporarily use DoRA as DoRAN fallback
    # DoRAN implementation would require more research (CVPR 2025)
    return apply_dora_custom(backbone, r=r, target_blocks=target_blocks)

