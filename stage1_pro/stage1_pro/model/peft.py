import torch
import torch.nn as nn
from typing import List, Optional


class DoRAAdapter(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) for Phase 3.

    Fallback to LoRA if DoRA not supported.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        # DoRA-specific: weight decomposition
        self.magnitude = nn.Parameter(torch.ones(out_features, in_features))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Original weight (set at register time)
        self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DoRA decomposition."""
        if self.weight is None:
            raise RuntimeError(
                "DoRAAdapter weight not set. Call register_weight() first."
            )

        # LoRA path: B * A * x
        lora_result = self.lora_B @ self.lora_A @ x.T
        lora_result = lora_result.T

        # Apply dropout
        lora_result = self.dropout(lora_result)

        # DoRA: Combine magnitude and direction
        # W = magnitude * (W_0 + delta_W / ||W_0 + delta_W||)
        delta_W = self.lora_B @ self.lora_A * self.scaling

        # Weight norm for normalization
        W_combined = self.weight + delta_W
        W_norm = W_combined.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)

        # Decomposed weight
        W_dora = self.magnitude * (W_combined / W_norm)

        # Apply weight
        output = x @ W_dora.T

        return output

    def register_weight(self, weight: nn.Parameter):
        """Register the original weight for adaptation."""
        self.weight = weight

    def get_lora_weights(self) -> torch.Tensor:
        """Get LoRA component for saving."""
        return torch.cat([self.lora_A.flatten(), self.lora_B.flatten()])

    def set_lora_weights(self, weights: torch.Tensor):
        """Set LoRA component from saved weights."""
        split = self.in_features * self.rank
        self.lora_A.data = weights[:split].view(self.rank, self.in_features)
        self.lora_B.data = weights[split:].view(self.out_features, self.rank)


class LoRAAdapter(nn.Module):
    """
    Standard LoRA (Low-Rank Adaptation) as fallback.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Original weight
        self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA."""
        if self.weight is None:
            raise RuntimeError(
                "LoRAAdapter weight not set. Call register_weight() first."
            )

        # LoRA path: B * A * x
        lora_result = self.lora_B @ self.lora_A @ x.T
        lora_result = lora_result.T

        # Apply dropout
        lora_result = self.dropout(lora_result)

        # Apply scaling
        lora_result = lora_result * self.scaling

        # Original weight path
        original_result = x @ self.weight.T

        return original_result + lora_result

    def register_weight(self, weight: nn.Parameter):
        """Register the original weight for adaptation."""
        self.weight = weight

    def get_lora_weights(self) -> torch.Tensor:
        """Get LoRA component for saving."""
        return torch.cat([self.lora_A.flatten(), self.lora_B.flatten()])

    def set_lora_weights(self, weights: torch.Tensor):
        """Set LoRA component from saved weights."""
        split = self.in_features * self.rank
        self.lora_A.data = weights[:split].view(self.rank, self.in_features)
        self.lora_B.data = weights[split:].view(self.out_features, self.rank)
