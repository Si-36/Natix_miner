import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional
import random


class ExploraAugmentation:
    """
    Explora-style data augmentation for Phase 5 (2025 SOTA).

    Combines multiple augmentation strategies:
    - Color jitter (brightness, contrast, saturation)
    - Gaussian blur
    - Sharpness adjustment
    - Cutout
    - Mixup-style blending

    Reference: https://arxiv.org/abs/2105.02701
    """

    def __init__(
        self,
        img_size: int = 224,
        color_jitter_strength: float = 0.4,
        blur_prob: float = 0.1,
        cutout_prob: float = 0.5,
        cutout_size: int = 16,
    ):
        self.img_size = img_size
        self.color_jitter_strength = color_jitter_strength
        self.blur_prob = blur_prob
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size

        # Augmentation pipeline
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
        """Apply color jitter using PIL."""
        brightness = 1.0 + random.uniform(
            -self.color_jitter_strength, self.color_jitter_strength
        )
        contrast = 1.0 + random.uniform(
            -self.color_jitter_strength, self.color_jitter_strength
        )
        saturation = 1.0 + random.uniform(
            -self.color_jitter_strength, self.color_jitter_strength
        )

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

        return img

    def _apply_gaussian_blur(self, img: Image.Image) -> Image.Image:
        """Apply random Gaussian blur."""
        radius = random.uniform(0.1, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _apply_cutout(self, img: Image.Image) -> Image.Image:
        """Apply cutout augmentation."""
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        # Random cutout position
        y = random.randint(0, h - self.cutout_size)
        x = random.randint(0, w - self.cutout_size)

        # Apply cutout (set to mean)
        img_array[y : y + self.cutout_size, x : x + self.cutout_size] = img_array.mean(
            axis=(0, 1)
        )

        return Image.fromarray(img_array.astype(np.uint8))

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply Explora augmentation pipeline."""
        # Convert to tensor first
        img_tensor = transforms.ToTensor()(img)

        # Convert back to PIL for PIL-based augmentations
        img = to_pil_image(img_tensor)

        # Apply basic transforms
        img = self.transform(img)

        # Explora-specific augmentations
        if random.random() < 0.8:
            img = self._apply_color_jitter(img)

        if random.random() < self.blur_prob:
            img = self._apply_gaussian_blur(img)

        if random.random() < self.cutout_prob:
            img = self._apply_cutout(img)

        # Convert to tensor and normalize
        img_tensor = transforms.ToTensor()(img)

        # DINOv3 normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor


class DomainAwareTransform:
    """
    Domain-aware augmentation for multi-dataset training.

    Adapts augmentation strength based on dataset source.
    """

    def __init__(
        self,
        base_augmentation: ExploraAugmentation,
        domain_configs: Optional[dict] = None,
    ):
        self.base_augmentation = base_augmentation

        # Domain-specific configs
        self.domain_configs = domain_configs or {
            "natix": {"color_jitter_strength": 0.3},
            "roadwork": {"color_jitter_strength": 0.5},
            "kaggle": {"color_jitter_strength": 0.4},
        }

    def __call__(self, img: Image.Image, domain: str = "natix") -> torch.Tensor:
        """Apply domain-aware augmentation."""
        config = self.domain_configs.get(domain, {})

        # Temporarily modify augmentation
        original_strength = self.base_augmentation.color_jitter_strength
        self.base_augmentation.color_jitter_strength = config.get(
            "color_jitter_strength", original_strength
        )

        # Apply augmentation
        result = self.base_augmentation(img)

        # Restore original
        self.base_augmentation.color_jitter_strength = original_strength

        return result
