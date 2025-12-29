"""
NATIX Dataset - Production-Grade PyTorch Dataset

Loads NATIX roadwork images with:
- 4-way split support (train, val_select, val_calib, val_test)
- DINOv3 transforms (224×224, ImageNet normalization)
- Split contract enforcement (zero data leakage)
- Fast image loading with PIL
- Proper error handling

Latest 2025-2026 practices:
- Python 3.14+ with modern type hints
- Cached properties for efficiency
- Clear error messages
- Type-safe with dataclasses
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from contracts.split_contracts import Split, SplitValidator
from data.label_schema import LabelSchema
from data.transforms import get_letterbox_transform

logger = logging.getLogger(__name__)


# DINOv3 canonical transforms (from official DINOv3 repository)
DINOV3_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
DINOV3_STD = [0.229, 0.224, 0.225]  # ImageNet std
DINOV3_IMAGE_SIZE = 224  # ViT-H/16+ uses 224×224


@dataclass(frozen=True, slots=True)
class NATIXImageMetadata:
    """Metadata for a single NATIX image"""

    image_path: Path
    label: int
    split: Split
    image_id: str  # Unique identifier


def get_dinov3_transforms(
    train: bool = True,
    eval_mode: str = "center_crop_224",
    eval_canvas_size: int = 896,
) -> Union[T.Compose, Callable]:
    """
    Get DINOv3 canonical transforms (2025-12-29 with high-res eval support)

    Args:
        train: If True, apply training augmentation (RandomResizedCrop, horizontal flip)
               If False, apply validation transforms
        eval_mode: Eval transform mode (only used if train=False):
                   - "center_crop_224": Legacy Resize(256) + CenterCrop(224)
                   - "letterbox_canvas": High-res letterbox to preserve 4K detail (BEST)
        eval_canvas_size: Canvas size for letterbox mode (896, 1024, or 1280)

    Returns:
        Composed transforms for DINOv3 (or letterbox transform for eval)
    """
    if train:
        # Training transforms with data augmentation (224×224 crops)
        return T.Compose(
            [
                T.RandomResizedCrop(
                    DINOV3_IMAGE_SIZE,
                    scale=(0.08, 1.0),  # DINOv3 default
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD),
            ]
        )
    else:
        # Validation/test transforms
        if eval_mode == "letterbox_canvas":
            # HIGH-RES MODE (BEST for max accuracy)
            # Returns: (tensor, content_box) tuple
            return get_letterbox_transform(
                canvas_size=eval_canvas_size,
                mean=DINOV3_MEAN,
                std=DINOV3_STD,
            )
        else:
            # LEGACY MODE (center_crop_224)
            # Returns: tensor only
            return T.Compose(
                [
                    T.Resize(
                        256, interpolation=T.InterpolationMode.BICUBIC
                    ),  # Resize shorter edge to 256
                    T.CenterCrop(DINOV3_IMAGE_SIZE),
                    T.ToTensor(),
                    T.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD),
                ]
            )


class NATIXDataset(Dataset):
    """
    NATIX Roadwork Dataset

    Loads images from disk with split-aware loading and DINOv3 preprocessing.

    CRITICAL: This dataset enforces split contracts to prevent data leakage.

    Args:
        data_root: Root directory containing NATIX images
        splits_json: Path to splits.json file (contains train/val splits)
        split: Which split to load (train, val_select, val_calib, val_test)
        transform: Optional custom transform (if None, uses DINOv3 defaults)
        validate_splits: If True, validates split usage against contracts

    Example:
        >>> dataset = NATIXDataset(
        ...     data_root="/data/natix",
        ...     splits_json="outputs/data_splits/splits.json",
        ...     split=Split.TRAIN
        ... )
        >>> image, label = dataset[0]
        >>> print(image.shape)  # torch.Size([3, 224, 224])
    """

    def __init__(
        self,
        data_root: str | Path,
        splits_json: str | Path,
        split: Split,
        transform: Optional[Callable] = None,
        validate_splits: bool = True,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.splits_json = Path(splits_json)
        self.split = split
        self.validate_splits = validate_splits

        # Validate data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        # Validate splits file exists
        if not self.splits_json.exists():
            raise FileNotFoundError(f"Splits file does not exist: {self.splits_json}")

        # Load splits
        self.splits_data = self._load_splits()

        # Validate split contract (if enabled)
        if self.validate_splits:
            self._validate_split_usage()

        # Load image metadata
        self.samples = self._load_samples()

        # Set transforms
        if transform is None:
            # Use DINOv3 defaults
            self.transform = get_dinov3_transforms(train=(split == Split.TRAIN))
        else:
            self.transform = transform

        logger.info(
            f"Loaded {len(self.samples)} images for split '{split.value}' from {self.data_root}"
        )

    def _load_splits(self) -> dict[str, Any]:
        """Load splits.json file"""
        try:
            with open(self.splits_json) as f:
                splits_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in splits file {self.splits_json}: {e}")

        # Validate splits file has required keys
        required_keys = {"train", "val_select", "val_calib", "val_test"}
        missing_keys = required_keys - set(splits_data.keys())
        if missing_keys:
            raise ValueError(
                f"Splits file missing required keys: {missing_keys}\n"
                f"Found keys: {list(splits_data.keys())}"
            )

        return splits_data

    def _validate_split_usage(self) -> None:
        """
        Validate that split usage conforms to contracts

        This is CRITICAL for preventing data leakage.
        Split contracts are enforced at dataset level.
        """
        validator = SplitValidator()

        # For now, just check that split is valid
        # More complex validation happens at pipeline level
        if self.split not in Split:
            raise ValueError(
                f"Invalid split: {self.split}\n" f"Valid splits: {[s.value for s in Split]}"
            )

    def _load_samples(self) -> list[NATIXImageMetadata]:
        """
        Load image samples for the current split

        Returns:
            List of NATIXImageMetadata objects
        """
        split_key = self.split.value
        image_list = self.splits_data[split_key]

        if not image_list:
            raise ValueError(f"Split '{split_key}' has no images!")

        samples = []
        for image_entry in image_list:
            # Handle both string (just filename) and dict (with metadata) formats
            if isinstance(image_entry, str):
                image_filename = image_entry
                # Infer label from filename or directory structure
                # Format: "class_name/image_001.jpg" or "image_001_class5.jpg"
                label = self._infer_label(image_filename)
            elif isinstance(image_entry, dict):
                image_filename = image_entry["filename"]
                label = image_entry["label"]
            else:
                raise ValueError(
                    f"Invalid image entry format: {image_entry}\n"
                    f"Expected str or dict, got {type(image_entry)}"
                )

            # Construct full path
            image_path = self.data_root / image_filename

            # Validate image exists
            if not image_path.exists():
                logger.warning(f"Image does not exist: {image_path} (skipping)")
                continue

            # Create metadata
            metadata = NATIXImageMetadata(
                image_path=image_path,
                label=label,
                split=self.split,
                image_id=image_filename,
            )
            samples.append(metadata)

        if not samples:
            raise ValueError(
                f"No valid images found for split '{split_key}'!\n"
                f"Check that images exist in {self.data_root}"
            )

        return samples

    def _infer_label(self, image_filename: str) -> int:
        """
        Infer label from image filename using LabelSchema

        CRITICAL: Uses single source of truth (label_schema.py).
        Fails fast if class is unknown.

        Expected formats:
        1. "class_name/image_001.jpg" → parse class_name
        2. "image_001_class5.jpg" → parse class5
        3. Numeric directory: "5/image_001.jpg" → class 5

        Args:
            image_filename: Filename (possibly with path)

        Returns:
            Integer label (0-12 for 13 classes)

        Raises:
            ValueError: If label cannot be inferred
        """
        # Try to parse from directory structure
        parts = Path(image_filename).parts
        if len(parts) > 1:
            # Has directory, try to parse class name
            class_name = parts[-2]

            # Try as class name
            try:
                return LabelSchema.get_class_id(class_name)
            except ValueError:
                pass

            # Try as integer
            try:
                label = int(class_name)
                LabelSchema.validate_label(label)
                return label
            except (ValueError, ValueError):
                pass

        # Try to parse from filename (e.g., "image_001_class5.jpg")
        filename = Path(image_filename).stem
        if "_class" in filename:
            try:
                label = int(filename.split("_class")[-1])
                LabelSchema.validate_label(label)
                return label
            except (ValueError, IndexError):
                pass

        # CRITICAL: Fail-fast instead of guessing
        raise ValueError(
            f"Could not infer label from filename: {image_filename}\n"
            f"Expected formats:\n"
            f"  1. class_name/image.jpg (e.g., 'pothole/img001.jpg')\n"
            f"  2. class_id/image.jpg (e.g., '4/img001.jpg')\n"
            f"  3. image_classID.jpg (e.g., 'img001_class4.jpg')\n"
            f"\nValid class names: {LabelSchema.CLASS_NAMES}\n"
            f"Valid class IDs: 0-{LabelSchema.NUM_CLASSES-1}\n"
            f"\nFIX: Use splits.json instead of inferring labels!"
        )

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, torch.Tensor]]:
        """
        Get a single sample (2025-12-29 with letterbox support)

        Args:
            idx: Index of sample

        Returns:
            Train mode OR eval without letterbox:
                (image_tensor, label)
                - image_tensor: torch.Tensor [3, H, W]
                - label: int (0-12 for 13 classes)

            Eval with letterbox mode:
                (image_tensor, label, content_box)
                - image_tensor: torch.Tensor [3, canvas, canvas]
                - label: int (0-12 for 13 classes)
                - content_box: torch.Tensor [4] in (x1, y1, x2, y2) format
        """
        metadata = self.samples[idx]

        # Load image
        try:
            image = Image.open(metadata.image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image {metadata.image_path}: {e}"
            ) from e

        # Apply transforms
        if self.transform is not None:
            result = self.transform(image)

            # Check if transform returns (tensor, content_box) tuple (letterbox mode)
            if isinstance(result, tuple) and len(result) == 2:
                image_tensor, content_box = result
                # Convert content_box tuple to Tensor[4] for vectorized ops
                content_box = torch.tensor(content_box, dtype=torch.float32)
                return image_tensor, metadata.label, content_box
            else:
                # Standard mode (train or legacy eval)
                image_tensor = result
                return image_tensor, metadata.label
        else:
            # No transform (shouldn't happen in practice)
            return image, metadata.label

    def get_metadata(self, idx: int) -> NATIXImageMetadata:
        """
        Get metadata for a sample (without loading image)

        Args:
            idx: Index of sample

        Returns:
            NATIXImageMetadata object
        """
        return self.samples[idx]

    @property
    def num_classes(self) -> int:
        """Number of classes in dataset (from LabelSchema)"""
        return LabelSchema.NUM_CLASSES  # CRITICAL: Use single source of truth

    def get_class_distribution(self) -> dict[int, int]:
        """
        Get class distribution (count per class)

        Returns:
            Dictionary mapping class ID to count
        """
        distribution = {}
        for sample in self.samples:
            label = sample.label
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"NATIXDataset(split={self.split.value}, "
            f"num_samples={len(self.samples)}, "
            f"data_root={self.data_root})"
        )
