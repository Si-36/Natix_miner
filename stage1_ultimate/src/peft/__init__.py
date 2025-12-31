"""PEFT (Parameter-Efficient Fine-Tuning) modules for ExPLoRA"""

# Use relative imports to avoid conflict with HuggingFace peft library
from .explora_domain import (
    SimCLRLoss,
    create_projection_head,
    create_explora_config,
    apply_explora_to_backbone,
)

__all__ = [
    "SimCLRLoss",
    "create_projection_head",
    "create_explora_config",
    "apply_explora_to_backbone",
]

