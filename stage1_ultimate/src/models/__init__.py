"""Model architectures and components"""

from models.backbone import DINOv3Backbone, create_dinov3_backbone
from models.head import ClassificationHead, DoRANHead, create_classification_head
from models.module import DINOv3Classifier, EMA
from models.multi_view import (
    MultiViewGenerator,
    TopKMeanAggregator,
    AttentionAggregator,
    MultiViewDINOv3,
    create_multiview_model,
)
from models.explora_config import ExPLoRAConfig, EXPLORA_PRESETS
from models.explora_module import ExPLoRAModule

__all__ = [
    # Backbone
    "DINOv3Backbone",
    "create_dinov3_backbone",
    # Head
    "ClassificationHead",
    "DoRANHead",
    "create_classification_head",
    # Module
    "DINOv3Classifier",
    "EMA",
    # Multi-view
    "MultiViewGenerator",
    "TopKMeanAggregator",
    "AttentionAggregator",
    "MultiViewDINOv3",
    "create_multiview_model",
    # ExPLoRA
    "ExPLoRAConfig",
    "EXPLORA_PRESETS",
    "ExPLoRAModule",
]
