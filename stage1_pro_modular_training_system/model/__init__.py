"""
Model module for Stage-1 Pro Modular Training System

Provides backbone, head, and PEFT components.
"""

# Make imports optional for offline testing
try:
    from .backbone import DINOv3Backbone
    _has_backbone = True
except ImportError:
    _has_backbone = False
    DINOv3Backbone = None

try:
    from .head import Stage1Head
    _has_head = True
except ImportError:
    _has_head = False
    Stage1Head = None

try:
    from .gate_head import GateHead
    _has_gate_head = True
except ImportError:
    _has_gate_head = False
    GateHead = None

__all__ = ["DINOv3Backbone", "Stage1Head", "GateHead"]
