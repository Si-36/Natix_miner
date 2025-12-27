import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any


def build_optimizer(
    model: nn.Module,
    config: Any,
    use_fsam: bool = False,
):
    """
    Build AdamW optimizer with per-layer learning rates.

    Exact match to baseline train_stage1_head.py:
    - lr_head for classifier head
    - lr_backbone for unfrozen backbone params (Phase 1: frozen)
    - Phase 3: PEFT parameters use lr_head
    """
    from ..config import Stage1ProConfig

    param_groups = []

    # Head parameters (always trainable)
    if hasattr(model, "head"):
        param_groups.append({"params": model.head.parameters(), "lr": config.lr_head})

    # Backbone parameters (frozen in Phase 1-2, unfrozen in Phase 3+ with PEFT)
    if hasattr(model, "backbone"):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if len(backbone_params) > 0:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": config.lr_backbone
                    if config.peft_type == "none"
                    else config.lr_head,
                }
            )

    # PEFT parameters (disabled in Phase 1-2)
    if hasattr(model, "backbone"):
        peft_params = (
            model.backbone.get_peft_parameters()
            if hasattr(model.backbone, "get_peft_parameters")
            else []
        )
        if len(peft_params) > 0:
            param_groups.append({"params": peft_params, "lr": config.lr_head})

    if use_fsam:
        # Phase 5+ F-SAM optimizer
        from .fsam import FSAM

        base_optimizer = optim.SGD(
            param_groups,
            lr=config.lr_head,
            momentum=0.9,
        )
        optimizer = FSAM(model, base_optimizer, rho=getattr(config, "fsam_rho", 0.5))
    else:
        # Phase 1-4 AdamW
        optimizer = optim.AdamW(
            param_groups,
            lr=config.lr_head,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    return optimizer
