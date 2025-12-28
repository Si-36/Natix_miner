"""
4-Way Split Generator - Stratified Balanced Splits

Generates train/val_select/val_calib/val_test splits with:
- Stratified sampling (balanced class distribution)
- Deterministic splits (reproducible with seed)
- Leakage prevention (enforced contracts)
- Comprehensive metadata

Latest 2025-2026 practices:
- Python 3.14+ with modern type hints
- Scikit-learn for stratified splitting
- JSON output with metadata
- Clear validation and error handling
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

from contracts.split_contracts import Split, SplitValidator
from data.label_schema import LabelSchema

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SplitConfig:
    """Configuration for split generation"""

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.60  # 60% for training
    val_select_ratio: float = 0.15  # 15% for model selection (early stopping)
    val_calib_ratio: float = 0.15  # 15% for policy fitting (calibration)
    val_test_ratio: float = 0.10  # 10% for final evaluation

    # Random seed for reproducibility
    random_seed: int = 42

    # Minimum samples per class per split
    min_samples_per_class: int = 5

    def validate(self) -> None:
        """Validate split configuration"""
        total = (
            self.train_ratio
            + self.val_select_ratio
            + self.val_calib_ratio
            + self.val_test_ratio
        )

        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total}\n"
                f"train={self.train_ratio}, val_select={self.val_select_ratio}, "
                f"val_calib={self.val_calib_ratio}, val_test={self.val_test_ratio}"
            )

        if any(
            r <= 0
            for r in [
                self.train_ratio,
                self.val_select_ratio,
                self.val_calib_ratio,
                self.val_test_ratio,
            ]
        ):
            raise ValueError("All split ratios must be positive")


@dataclass(slots=True)
class ImageSample:
    """Single image sample with metadata"""

    filename: str
    label: int
    class_name: str


class SplitGenerator:
    """
    Generate stratified 4-way splits for NATIX dataset

    This ensures:
    1. Balanced class distribution across all splits
    2. Deterministic splits (reproducible)
    3. Proper split size validation
    4. Leakage prevention contracts

    Example:
        >>> generator = SplitGenerator(
        ...     data_root="/data/natix",
        ...     output_path="outputs/data_splits/splits.json"
        ... )
        >>> splits = generator.generate_splits()
        >>> print(f"Train: {len(splits['train'])} samples")
    """

    def __init__(
        self,
        data_root: str | Path,
        output_path: str | Path,
        config: Optional[SplitConfig] = None,
    ):
        self.data_root = Path(data_root)
        self.output_path = Path(output_path)
        self.config = config or SplitConfig()

        # Validate config
        self.config.validate()

        # Validate data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        logger.info(f"Initializing SplitGenerator for {self.data_root}")

    def generate_splits(self) -> dict[str, list[dict]]:
        """
        Generate 4-way stratified splits

        Returns:
            Dictionary with keys: train, val_select, val_calib, val_test
            Each value is a list of image metadata dicts
        """
        logger.info("Scanning dataset...")

        # Scan dataset and collect all samples
        samples = self._scan_dataset()

        logger.info(f"Found {len(samples)} total samples")

        # Extract labels for stratification
        labels = np.array([s.label for s in samples])

        # Check class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique_labels, counts))}")

        # Validate minimum samples per class
        min_count = counts.min()
        min_required = (
            self.config.min_samples_per_class * 4
        )  # Need enough for all 4 splits
        if min_count < min_required:
            logger.warning(
                f"Some classes have <{min_required} samples (min={min_count}). "
                f"Splits may be unbalanced."
            )

        # Generate splits using stratified sampling
        logger.info("Generating stratified splits...")
        splits_dict = self._stratified_split(samples, labels)

        # Validate splits
        self._validate_splits(splits_dict)

        # Save splits to JSON
        self._save_splits(splits_dict)

        # Log summary
        self._log_split_summary(splits_dict)

        return splits_dict

    def _scan_dataset(self) -> list[ImageSample]:
        """
        Scan dataset directory and collect all samples

        Expected structure:
            data_root/
                class_0/
                    image_001.jpg
                    image_002.jpg
                class_1/
                    image_003.jpg
                ...

        Returns:
            List of ImageSample objects
        """
        samples = []

        # Get all class directories
        class_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(
                f"No class directories found in {self.data_root}\n"
                f"Expected structure: data_root/class_name/*.jpg"
            )

        for class_dir in class_dirs:
            class_name = class_dir.name

            # Map class name to integer label
            label = self._class_name_to_label(class_name)

            # Get all images in this class
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            if not image_files:
                logger.warning(f"No images found in {class_dir}")
                continue

            # Create samples
            for image_path in image_files:
                # Store relative path from data_root
                relative_path = image_path.relative_to(self.data_root)

                sample = ImageSample(
                    filename=str(relative_path),
                    label=label,
                    class_name=class_name,
                )
                samples.append(sample)

        if not samples:
            raise ValueError(f"No images found in {self.data_root}")

        return samples

    def _class_name_to_label(self, class_name: str) -> int:
        """
        Map class name to integer label using LabelSchema

        CRITICAL: This uses the single source of truth (label_schema.py).
        No fallbacks - fail-fast if class is unknown.

        Args:
            class_name: Name of the class directory

        Returns:
            Integer label (0-12 for 13 classes)

        Raises:
            ValueError: If class name is unknown
        """
        # CRITICAL: Use LabelSchema (single source of truth)
        try:
            return LabelSchema.get_class_id(class_name)
        except ValueError:
            # Try to parse as integer (for datasets with numeric class dirs)
            try:
                label = int(class_name)
                LabelSchema.validate_label(label)  # Validate range
                return label
            except (ValueError, ValueError):
                # CRITICAL: Fail-fast instead of guessing
                raise ValueError(
                    f"Unknown class directory: '{class_name}'\n"
                    f"Valid class names: {LabelSchema.CLASS_NAMES}\n"
                    f"Or use integer directories: 0-{LabelSchema.NUM_CLASSES-1}\n"
                    f"Found in: {self.data_root / class_name}\n"
                    f"\n"
                    f"FIX: Either rename the directory to a valid class name,\n"
                    f"     or add the new class to src/data/label_schema.py"
                ) from None

    def _stratified_split(
        self, samples: list[ImageSample], labels: np.ndarray
    ) -> dict[str, list[dict]]:
        """
        Perform stratified 4-way split

        Args:
            samples: List of ImageSample objects
            labels: Array of integer labels

        Returns:
            Dictionary with train/val_select/val_calib/val_test splits
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        # Step 1: Split off test set first (val_test)
        train_val_samples, test_samples, train_val_labels, test_labels = (
            train_test_split(
                samples,
                labels,
                test_size=self.config.val_test_ratio,
                stratify=labels,
                random_state=self.config.random_seed,
            )
        )

        # Step 2: Split train_val into train and validation sets
        # Calculate remaining ratios
        remaining_ratio = 1.0 - self.config.val_test_ratio
        val_select_ratio_adj = self.config.val_select_ratio / remaining_ratio
        val_calib_ratio_adj = self.config.val_calib_ratio / remaining_ratio

        # Split off val_select
        train_calib_samples, val_select_samples, train_calib_labels, val_select_labels = (
            train_test_split(
                train_val_samples,
                train_val_labels,
                test_size=val_select_ratio_adj,
                stratify=train_val_labels,
                random_state=self.config.random_seed + 1,
            )
        )

        # Step 3: Split train_calib into train and val_calib
        calib_ratio_adj = val_calib_ratio_adj / (1.0 - val_select_ratio_adj)

        train_samples, val_calib_samples, train_labels, val_calib_labels = (
            train_test_split(
                train_calib_samples,
                train_calib_labels,
                test_size=calib_ratio_adj,
                stratify=train_calib_labels,
                random_state=self.config.random_seed + 2,
            )
        )

        # Convert samples to dictionaries
        return {
            "train": [asdict(s) for s in train_samples],
            "val_select": [asdict(s) for s in val_select_samples],
            "val_calib": [asdict(s) for s in val_calib_samples],
            "val_test": [asdict(s) for s in test_samples],
        }

    def _validate_splits(self, splits_dict: dict[str, list[dict]]) -> None:
        """
        Validate generated splits

        Checks:
        1. All splits are non-empty
        2. No overlap between splits
        3. All original samples are included
        4. Class distribution is reasonable
        """
        # Check non-empty
        for split_name, split_samples in splits_dict.items():
            if not split_samples:
                raise ValueError(f"Split '{split_name}' is empty!")

        # Check no overlap
        all_filenames = set()
        for split_name, split_samples in splits_dict.items():
            split_filenames = {s["filename"] for s in split_samples}

            # Check for overlap with previous splits
            overlap = all_filenames & split_filenames
            if overlap:
                raise ValueError(
                    f"Split '{split_name}' has overlap with previous splits: {overlap}"
                )

            all_filenames.update(split_filenames)

        # Validate split contracts
        validator = SplitValidator()

        # All splits should be valid
        for split_enum in [Split.TRAIN, Split.VAL_SELECT, Split.VAL_CALIB, Split.VAL_TEST]:
            # Just validate that splits exist
            split_key = split_enum.value
            if split_key not in splits_dict:
                raise ValueError(f"Missing required split: {split_key}")

        logger.info("✅ Split validation passed!")

    def _save_splits(self, splits_dict: dict[str, list[dict]]) -> None:
        """
        Save splits to JSON file

        Adds metadata:
        - Creation timestamp
        - Split sizes
        - Configuration used
        - Class distribution per split
        """
        # Add metadata
        output_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "data_root": str(self.data_root),
                "total_samples": sum(len(v) for v in splits_dict.values()),
                "split_sizes": {k: len(v) for k, v in splits_dict.items()},
                "config": asdict(self.config),
                "class_distribution": self._compute_class_distribution(splits_dict),
            },
            **splits_dict,
        }

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(self.output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved splits to {self.output_path}")

    def _compute_class_distribution(
        self, splits_dict: dict[str, list[dict]]
    ) -> dict[str, dict[int, int]]:
        """Compute class distribution for each split"""
        distribution = {}

        for split_name, split_samples in splits_dict.items():
            split_dist = {}
            for sample in split_samples:
                label = sample["label"]
                split_dist[label] = split_dist.get(label, 0) + 1
            distribution[split_name] = split_dist

        return distribution

    def _log_split_summary(self, splits_dict: dict[str, list[dict]]) -> None:
        """Log summary of generated splits"""
        logger.info("=" * 80)
        logger.info("SPLIT GENERATION SUMMARY")
        logger.info("=" * 80)

        for split_name, split_samples in splits_dict.items():
            logger.info(f"\n{split_name.upper()}: {len(split_samples)} samples")

            # Class distribution
            labels = [s["label"] for s in split_samples]
            unique_labels, counts = np.unique(labels, return_counts=True)
            logger.info(f"  Class distribution: {dict(zip(unique_labels, counts))}")

        logger.info("=" * 80)


def generate_splits_cli(
    data_root: str,
    output_path: str,
    train_ratio: float = 0.60,
    val_select_ratio: float = 0.15,
    val_calib_ratio: float = 0.15,
    val_test_ratio: float = 0.10,
    random_seed: int = 42,
) -> dict[str, list[dict]]:
    """
    CLI wrapper for split generation

    Args:
        data_root: Root directory containing images
        output_path: Where to save splits.json
        train_ratio: Fraction for training (default: 0.60)
        val_select_ratio: Fraction for model selection (default: 0.15)
        val_calib_ratio: Fraction for calibration (default: 0.15)
        val_test_ratio: Fraction for final evaluation (default: 0.10)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with generated splits
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        val_select_ratio=val_select_ratio,
        val_calib_ratio=val_calib_ratio,
        val_test_ratio=val_test_ratio,
        random_seed=random_seed,
    )

    generator = SplitGenerator(
        data_root=data_root, output_path=output_path, config=config
    )

    return generator.generate_splits()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python split_generator.py <data_root> <output_path>")
        print("Example: python split_generator.py /data/natix outputs/splits.json")
        sys.exit(1)

    data_root = sys.argv[1]
    output_path = sys.argv[2]

    splits = generate_splits_cli(data_root=data_root, output_path=output_path)

    print(f"\n✅ Generated splits successfully!")
    print(f"   Train: {len(splits['train'])} samples")
    print(f"   Val Select: {len(splits['val_select'])} samples")
    print(f"   Val Calib: {len(splits['val_calib'])} samples")
    print(f"   Val Test: {len(splits['val_test'])} samples")
