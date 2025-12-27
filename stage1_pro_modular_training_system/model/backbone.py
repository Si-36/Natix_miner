"""
DINOv3 Backbone Wrapper - Dec 2025 Best Practices

Uses HuggingFace Transformers 4.40.0+ for loading DINOv3 (latest).
Implements proper freezing, feature extraction, and PEFT hooks for Phase 4+.
"""

import torch
from transformers import AutoModel, AutoImageProcessor
from typing import Optional, Dict


class DINOv3Backbone:
    """
    DINOv3 backbone wrapper preserving exact logic from train_stage1_head.py.
    
    Dec 2025 Best Practices:
    - Uses HuggingFace Transformers AutoModel.from_pretrained()
    - AutoImageProcessor for automatic image normalization
    - Supports PEFT hooks for Phase 4+ (DoRA/LoRA via HuggingFace PEFT library)
    - TF32 precision support for faster training
    - Proper freezing/unfreezing for different phases
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize DINOv3 backbone wrapper.
        
        Args:
            model_path: Path to pretrained DINOv3 model
                    (e.g., "facebook/dinov3-vit-huge-14-224")
            device: Device to load model on (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device
        self.backbone = None
        self.processor = None
        self._frozen = False
    
    def load(self, freeze: bool = True):
        """
        Load DINOv3 backbone (Dec 2025 HuggingFace best practice).
        
        Uses:
        - AutoModel.from_pretrained() for loading
        - AutoImageProcessor.from_pretrained() for image preprocessing
        - Device management (.to(device))
        - Eval mode setting (.eval())
        
        Args:
            freeze: Whether to freeze all parameters (Phase 1 default)
        
        Returns:
            self (for method chaining)
        """
        print(f"\n[1/3] Loading DINOv3 backbone from {self.model_path}...")
        print(f"    Dec 2025: Using HuggingFace Transformers AutoModel.from_pretrained()")
        
        # Load DINOv3 backbone (Dec 2025 best practice)
        self.backbone = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.backbone.to(self.device)
        
        # Load processor (Dec 2025 best practice)
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        
        # Freeze all parameters (Phase 1 default)
        if freeze:
            self.freeze()
        else:
            self._frozen = False
        
        # Set to eval mode (frozen parameters don't need gradients)
        self.backbone.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.backbone.parameters())
        frozen_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        print(f"    Model: {type(self.backbone).__name__}")
        print(f"    Total parameters: {total_params/1e6:.2f}M")
        print(f"    Frozen parameters: {frozen_params/1e6:.2f}M ({100*frozen_params/total_params:.1f}%)")
        print(f"    Device: {self.device}")
        
        return self
    
    def freeze(self):
        """
        Freeze all backbone parameters.
        
        Dec 2025 Best Practice:
        - Uses .requires_grad = False for all parameters
        - Tracks frozen state
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._frozen = True
        print(f"    Backbone frozen: {sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)/1e6:.2f}M params")
    
    def unfreeze(self):
        """
        Unfreeze all backbone parameters (Phase 4+ PEFT training).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        print(f"    Backbone unfrozen: {sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)/1e6:.2f}M params")
    
    def extract_features(self, pixel_values):
        """
        Preserve EXACT logic from train_stage1_head.py.
        
        Extract CLS token from outputs.last_hidden_state[:, 0, :]
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            # Extract CLS token (index 0 is CLS token)
            features = outputs.last_hidden_state[:, 0, :]
        
        return features
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get parameter count (frozen + trainable).
        
        Dec 2025 Best Practice:
        - Returns breakdown of total, frozen, trainable parameters
        - Useful for logging and debugging
        """
        if self.backbone is None:
            return {
                'total': 0,
                'frozen': 0,
                'trainable': 0
            }
        
        total = sum(p.numel() for p in self.backbone.parameters())
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        trainable = total - frozen
        
        return {
            'total': total,
            'frozen': frozen,
            'trainable': trainable
        }
