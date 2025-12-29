"""
High-Resolution Transforms for Maximum Accuracy (2025-12-29)

Implements letterbox transform to preserve 4K detail while allowing batching.

Key features:
- Letterbox: Resize with aspect ratio, pad to fixed canvas
- Content-aware boxing: Track content region for ROI generation
- Multi-scale support: Multiple canvas sizes for ensemble
- Batching-friendly: Fixed-shape outputs [3, canvas, canvas]
"""

from typing import Tuple, Optional, List
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


class LetterboxTransform:
    """
    Letterbox transform for high-res eval (preserves aspect ratio, no cropping).

    Resizes image to fit inside a square canvas, pads remainder with ImageNet mean.
    Returns both the transformed tensor and content box for content-aware tiling.

    Args:
        canvas_size: Square canvas size (e.g., 896, 1024, 1280)
        interpolation: PIL interpolation mode
        fill_rgb: RGB fill color for padding (ImageNet mean)
        mean: Normalization mean (ImageNet)
        std: Normalization std (ImageNet)

    Returns:
        tuple: (tensor [3, canvas_size, canvas_size], content_box (x1, y1, x2, y2))
    """

    def __init__(
        self,
        canvas_size: int = 896,
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
        fill_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet mean
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.canvas_size = canvas_size
        self.interpolation = interpolation
        self.fill_rgb = fill_rgb
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Apply letterbox transform.

        Args:
            image: PIL Image

        Returns:
            tensor: [3, canvas_size, canvas_size]
            content_box: (x1, y1, x2, y2) of content region (for ROI generation)
        """
        orig_w, orig_h = image.size

        # Compute scale to fit inside canvas (preserve aspect ratio)
        scale = min(self.canvas_size / orig_w, self.canvas_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize with aspect ratio
        resized_image = image.resize((new_w, new_h), self.interpolation)

        # Create canvas filled with ImageNet mean (so after normalization it's ~0)
        canvas = Image.new('RGB', (self.canvas_size, self.canvas_size),
                          tuple(int(c * 255) for c in self.fill_rgb))

        # Compute padding to center the image
        pad_left = (self.canvas_size - new_w) // 2
        pad_top = (self.canvas_size - new_h) // 2

        # Paste resized image onto canvas
        canvas.paste(resized_image, (pad_left, pad_top))

        # Convert to tensor and normalize
        tensor = TF.to_tensor(canvas)  # [3, canvas_size, canvas_size], range [0, 1]
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)

        # Content box (x1, y1, x2, y2) in canvas coordinates
        content_box = (
            pad_left,
            pad_top,
            pad_left + new_w,
            pad_top + new_h,
        )

        return tensor, content_box

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"canvas_size={self.canvas_size}, "
            f"interpolation={self.interpolation.name})"
        )


class MultiScaleLetterbox:
    """
    Multi-scale letterbox for ensemble evaluation.

    Applies letterbox at multiple canvas sizes (e.g., [1024, 1280])
    and returns a list of (tensor, content_box) tuples.

    Args:
        canvas_sizes: List of canvas sizes for multi-scale
        interpolation: PIL interpolation mode
        fill_rgb: RGB fill color for padding
        mean: Normalization mean
        std: Normalization std
    """

    def __init__(
        self,
        canvas_sizes: List[int] = [1024, 1280],
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
        fill_rgb: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.transforms = [
            LetterboxTransform(
                canvas_size=size,
                interpolation=interpolation,
                fill_rgb=fill_rgb,
                mean=mean,
                std=std,
            )
            for size in canvas_sizes
        ]

    def __call__(self, image: Image.Image) -> List[Tuple[torch.Tensor, Tuple[int, int, int, int]]]:
        """
        Apply multi-scale letterbox.

        Args:
            image: PIL Image

        Returns:
            List of (tensor, content_box) for each scale
        """
        return [transform(image) for transform in self.transforms]

    def __repr__(self) -> str:
        sizes = [t.canvas_size for t in self.transforms]
        return f"{self.__class__.__name__}(canvas_sizes={sizes})"


def letterbox_collate_fn(batch):
    """
    Custom collate function for letterbox transforms (2025-12-29).

    Handles batches where each sample is (image_tensor, label, content_box).
    Stacks tensors into batched format for vectorized multi-view processing.

    Args:
        batch: List of (image_tensor, label, content_box) tuples
               where content_box is already Tensor[4] from dataset

    Returns:
        images: Tensor[B, 3, H, W]
        labels: Tensor[B] (long)
        content_boxes: Tensor[B, 4] (float32) in (x1, y1, x2, y2) format
    """
    images = []
    labels = []
    content_boxes = []

    for item in batch:
        if len(item) == 3:
            # Letterbox mode: (image_tensor, label, content_box)
            image, label, content_box = item
            images.append(image)
            labels.append(label)
            content_boxes.append(content_box)
        elif len(item) == 2:
            # Standard mode: (image_tensor, label) - shouldn't happen with letterbox
            # But handle gracefully for backward compatibility
            image, label = item
            images.append(image)
            labels.append(label)
            # No content box, will return None

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    if content_boxes:
        # Stack content boxes into Tensor[B, 4] for vectorized ops
        content_boxes = torch.stack(content_boxes, dim=0)
        return images, labels, content_boxes
    else:
        # No content boxes (backward compat)
        return images, labels


# Pre-configured transforms for common use cases
def get_letterbox_transform(
    canvas_size: int = 896,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> LetterboxTransform:
    """Get letterbox transform with ImageNet normalization."""
    return LetterboxTransform(
        canvas_size=canvas_size,
        interpolation=Image.Resampling.BILINEAR,
        fill_rgb=mean,  # Fill with mean so after normalization it's ~0
        mean=mean,
        std=std,
    )


def get_multiscale_letterbox(
    canvas_sizes: List[int] = [1024, 1280],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> MultiScaleLetterbox:
    """Get multi-scale letterbox transform."""
    return MultiScaleLetterbox(
        canvas_sizes=canvas_sizes,
        interpolation=Image.Resampling.BILINEAR,
        fill_rgb=mean,
        mean=mean,
        std=std,
    )


if __name__ == "__main__":
    # Test letterbox transform
    print("Testing Letterbox Transform...")

    # Create test image (4K resolution, common in NATIX)
    test_image = Image.new('RGB', (3840, 2160), color='red')

    # Test single-scale
    transform = get_letterbox_transform(canvas_size=1024)
    tensor, content_box = transform(test_image)

    print(f"Input shape: {test_image.size}")
    print(f"Output tensor shape: {tensor.shape}")
    print(f"Content box: {content_box}")
    print(f"✓ Single-scale letterbox works!\n")

    # Test multi-scale
    multi_transform = get_multiscale_letterbox(canvas_sizes=[896, 1024, 1280])
    results = multi_transform(test_image)

    print(f"Multi-scale results:")
    for i, (tensor, content_box) in enumerate(results):
        print(f"  Scale {i}: tensor shape {tensor.shape}, content box {content_box}")
    print(f"✓ Multi-scale letterbox works!")
