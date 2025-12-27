import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class Stage1Head(nn.Module):
    """
    Multi-phase classifier head:

    Phase 1: Single-head (cls only)
    Phase 2+: 3-head architecture (cls, gate, aux)

    Architecture: hidden_size -> 768 -> [cls, gate, aux]
    """

    def __init__(
        self,
        hidden_size: int = 1024,  # DINOv3 hidden size
        dropout: float = 0.1,
        use_gate: bool = False,  # Phase 2+
    ):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Phase 1: Single-head (binary classification)
        self.cls_head = nn.Linear(768, 2)  # [no_roadwork, roadwork]

        # Phase 2+: Gate and auxiliary heads
        self.use_gate = use_gate
        if use_gate:
            self.gate_head = nn.Linear(768, 1)  # Exit decision: [stay, exit]
            self.aux_head = nn.Linear(768, 2)  # Auxiliary task

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning dictionary of outputs.

        Args:
            x: CLS token features [batch, hidden_size]

        Returns:
            Dictionary with:
            - cls_logits: [batch, 2] - classification logits
            - gate_logits: [batch, 1] - gate logits (Phase 2+)
            - aux_logits: [batch, 2] - auxiliary logits (Phase 2+)
        """
        features = self.shared(x)

        outputs = {"cls_logits": self.cls_head(features)}

        if self.use_gate:
            outputs["gate_logits"] = self.gate_head(features).squeeze(-1)
            outputs["aux_logits"] = self.aux_head(features)

        return outputs

    def get_probs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get probabilities from logits.

        Returns:
            Dictionary with:
            - cls_probs: [batch, 2] - classification probabilities
            - gate_probs: [batch] - gate probabilities (Phase 2+)
            - aux_probs: [batch, 2] - auxiliary probabilities (Phase 2+)
            - max_cls_probs: [batch] - max class probabilities
            - cls_preds: [batch] - class predictions
        """
        logits_dict = self.forward(x)

        cls_logits = logits_dict["cls_logits"]
        cls_probs = F.softmax(cls_logits, dim=-1)
        max_cls_probs, cls_preds = cls_probs.max(dim=-1)

        outputs = {
            "cls_logits": cls_logits,
            "cls_probs": cls_probs,
            "max_cls_probs": max_cls_probs,
            "cls_preds": cls_preds,
        }

        if self.use_gate:
            gate_logits = logits_dict["gate_logits"]
            aux_logits = logits_dict["aux_logits"]

            outputs["gate_logits"] = gate_logits
            outputs["gate_probs"] = torch.sigmoid(gate_logits)
            outputs["aux_logits"] = aux_logits
            outputs["aux_probs"] = F.softmax(aux_logits, dim=-1)

        return outputs
