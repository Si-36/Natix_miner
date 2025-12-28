"""
Lightning DataModule - Production-Grade Data Loading

PyTorch Lightning DataModule with:
- Split-aware data loading (train/val_select/val_calib/val_test)
- Multi-worker loading for efficiency
- Proper batch collation
- DINOv3 transforms built-in

Latest 2025-2026 practices:
- Python 3.14+ with modern type hints
- Lightning 2.4+ patterns
- Clean separation of concerns
- Reproducible data loading
"""

import logging
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from data.natix_dataset import NATIXDataset
from contracts.split_contracts import Split

logger = logging.getLogger(__name__)


class NATIXDataModule(L.LightningDataModule):
    """
    Lightning DataModule for NATIX dataset

    Handles all data loading logic with:
    - Automatic dataset creation for each split
    - Configurable batch size and workers
    - Proper train/val/test split handling
    - DINOv3 transforms applied automatically

    Args:
        data_root: Root directory containing NATIX images
        splits_json: Path to splits.json file
        batch_size: Batch size for dataloaders (default: 32)
        num_workers: Number of worker processes (default: 4)
        pin_memory: If True, pin memory for faster GPU transfer (default: True)
        persistent_workers: If True, keep workers alive between epochs (default: True)

    Example:
        >>> datamodule = NATIXDataModule(
        ...     data_root="/data/natix",
        ...     splits_json="outputs/data_splits/splits.json",
        ...     batch_size=32,
        ...     num_workers=4
        ... )
        >>> datamodule.setup("fit")
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        data_root: str | Path,
        splits_json: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()

        # Save hyperparameters (Lightning feature)
        self.save_hyperparameters()

        self.data_root = Path(data_root)
        self.splits_json = Path(splits_json)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # Datasets (created in setup())
        self.train_dataset: Optional[NATIXDataset] = None
        self.val_select_dataset: Optional[NATIXDataset] = None
        self.val_calib_dataset: Optional[NATIXDataset] = None
        self.val_test_dataset: Optional[NATIXDataset] = None

        logger.info(
            f"Initialized NATIXDataModule: batch_size={batch_size}, "
            f"num_workers={num_workers}"
        )

    def prepare_data(self) -> None:
        """
        Download and prepare data (called once on main process)

        For NATIX, we don't need to download anything.
        But we validate that data exists.
        """
        # Validate data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        # Validate splits file exists
        if not self.splits_json.exists():
            raise FileNotFoundError(f"Splits file does not exist: {self.splits_json}")

        logger.info("Data preparation complete (no download needed)")

    def setup(self, stage: str) -> None:
        """
        Setup datasets for each split

        Args:
            stage: One of "fit", "validate", "test", or "predict"
                - "fit": Create train + val_select datasets
                - "validate": Create val_select dataset
                - "test": Create val_test dataset
                - "predict": Create prediction dataset (same as test)
        """
        if stage == "fit":
            # Training: need train + val_select (for early stopping)
            logger.info("Setting up datasets for training...")

            self.train_dataset = NATIXDataset(
                data_root=self.data_root,
                splits_json=self.splits_json,
                split=Split.TRAIN,
                transform=None,  # Use DINOv3 defaults
            )

            self.val_select_dataset = NATIXDataset(
                data_root=self.data_root,
                splits_json=self.splits_json,
                split=Split.VAL_SELECT,
                transform=None,  # Use DINOv3 defaults
            )

            logger.info(f"Train: {len(self.train_dataset)} samples")
            logger.info(f"Val Select: {len(self.val_select_dataset)} samples")

        elif stage == "validate":
            # Validation: need val_select
            logger.info("Setting up datasets for validation...")

            self.val_select_dataset = NATIXDataset(
                data_root=self.data_root,
                splits_json=self.splits_json,
                split=Split.VAL_SELECT,
                transform=None,
            )

            logger.info(f"Val Select: {len(self.val_select_dataset)} samples")

        elif stage == "test":
            # Testing: need val_test (final evaluation)
            logger.info("Setting up datasets for testing...")

            self.val_test_dataset = NATIXDataset(
                data_root=self.data_root,
                splits_json=self.splits_json,
                split=Split.VAL_TEST,
                transform=None,
            )

            logger.info(f"Val Test: {len(self.val_test_dataset)} samples")

        elif stage == "predict":
            # Prediction: same as test
            logger.info("Setting up datasets for prediction...")

            self.val_test_dataset = NATIXDataset(
                data_root=self.data_root,
                splits_json=self.splits_json,
                split=Split.VAL_TEST,
                transform=None,
            )

            logger.info(f"Prediction: {len(self.val_test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader

        Returns:
            DataLoader for training set
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,  # Drop incomplete batches for stable training
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader (val_select)

        This is used for:
        - Early stopping
        - Model selection
        - Hyperparameter tuning

        CRITICAL: This should NEVER be used for policy fitting or calibration!
                  Use val_calib_dataloader() for that.

        Returns:
            DataLoader for validation set (val_select)
        """
        if self.val_select_dataset is None:
            raise RuntimeError("val_select_dataset is None. Call setup('fit') first.")

        return DataLoader(
            self.val_select_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,  # Keep all validation samples
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader (val_test)

        This is used for:
        - Final evaluation ONLY
        - NEVER touch during training/tuning!

        Returns:
            DataLoader for test set (val_test)
        """
        if self.val_test_dataset is None:
            raise RuntimeError("val_test_dataset is None. Call setup('test') first.")

        return DataLoader(
            self.val_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create prediction dataloader

        Same as test_dataloader but can be used for inference.

        Returns:
            DataLoader for prediction
        """
        return self.test_dataloader()

    def val_calib_dataloader(self) -> DataLoader:
        """
        Create calibration dataloader (val_calib)

        This is used for:
        - Policy fitting (threshold sweep)
        - Calibration (temperature scaling, SCRC)
        - NEVER for model selection!

        CRITICAL: This is a separate split from val_select to prevent leakage.

        Returns:
            DataLoader for calibration set (val_calib)
        """
        # Create dataset if not already created
        if self.val_calib_dataset is None:
            self.val_calib_dataset = NATIXDataset(
                data_root=self.data_root,
                splits_json=self.splits_json,
                split=Split.VAL_CALIB,
                transform=None,
            )

        return DataLoader(
            self.val_calib_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    @property
    def num_classes(self) -> int:
        """Number of classes in dataset"""
        return 13  # NATIX roadwork has 13 classes

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"NATIXDataModule(\n"
            f"  data_root={self.data_root},\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_workers={self.num_workers},\n"
            f"  train_samples={len(self.train_dataset) if self.train_dataset else None},\n"
            f"  val_select_samples={len(self.val_select_dataset) if self.val_select_dataset else None},\n"
            f"  val_calib_samples={len(self.val_calib_dataset) if self.val_calib_dataset else None},\n"
            f"  val_test_samples={len(self.val_test_dataset) if self.val_test_dataset else None}\n"
            f")"
        )


if __name__ == "__main__":
    # Test datamodule
    print("Testing NATIXDataModule...")

    # Example paths (update these to your actual paths)
    data_root = "../../data/natix"
    splits_json = "../../outputs/data_splits/splits.json"

    if not Path(data_root).exists() or not Path(splits_json).exists():
        print(f"\n⚠️  Test data not found:")
        print(f"  data_root: {data_root}")
        print(f"  splits_json: {splits_json}")
        print("Skipping tests (need real data)")
    else:
        # Create datamodule
        datamodule = NATIXDataModule(
            data_root=data_root,
            splits_json=splits_json,
            batch_size=4,
            num_workers=0,  # 0 for testing (no multiprocessing)
        )

        print(f"\nDataModule: {datamodule}")

        # Setup for training
        datamodule.setup("fit")

        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        calib_loader = datamodule.val_calib_dataloader()

        print(f"\nTrain loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        print(f"Calib loader: {len(calib_loader)} batches")

        # Test batch loading
        batch = next(iter(train_loader))
        images, labels = batch
        print(f"\nBatch shapes:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")

        print("\n✅ All tests passed!")
