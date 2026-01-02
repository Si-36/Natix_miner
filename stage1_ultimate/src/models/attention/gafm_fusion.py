"""
GAFM (Gated Attention Fusion Module)
Latest 2026 implementation

Achievements:
- 95% MCC in medical imaging (original paper)
- Cross-view attention fusion
- View importance gating
- Multi-head attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class GAFM(nn.Module):
    """
    Gated Attention Fusion Module
    
    Fuses 8 view features into single representation
    Input: [B, 8, D] - 8 pruned views
    Output: [B, D] - Single fused feature
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        use_gated_fusion: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_gated_fusion = use_gated_fusion
        
        # Cross-view attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # View importance gates
        if use_gated_fusion:
            self.importance_gate = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        logger.info(f"GAFM initialized: {num_heads} heads, dim={hidden_dim}")
    
    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: [B, 8, D]
        Returns:
            fused: [B, D]
        """
        batch_size, num_views, dim = view_features.shape
        
        # Cross-view self-attention
        attn_output, attn_weights = self.cross_attn(
            view_features, view_features, view_features,
            need_weights=True, average_attn_weights=True
        )
        
        # Gated fusion
        if self.use_gated_fusion:
            # Compute importance for each view
            importance = self.importance_gate(attn_output)  # [B, 8, 1]
            
            # Weighted sum
            fused = (attn_output * importance).sum(dim=1)  # [B, D]
        else:
            # Simple mean pooling
            fused = attn_output.mean(dim=1)  # [B, D]
        
        # Output projection
        fused = self.output_proj(fused)
        
        return fused


if __name__ == "__main__":
    print("Testing GAFM...")
    gafm = GAFM(hidden_dim=512, num_heads=8)
    x = torch.randn(2, 8, 512)
    out = gafm(x)
    print(f"✅ Input: {x.shape}, Output: {out.shape}")
    print("🎉 GAFM test passed!")
