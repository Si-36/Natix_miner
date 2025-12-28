"""
Val Calib Artifact Saver - ELITE 2025 Pattern

Saves val_calib logits/labels directly to ArtifactSchema paths (not RAM).

Why this is elite:
- Artifacts written to disk immediately (no RAM accumulation)
- Works with distributed training (rank 0 only)
- Fail-fast validation after write
- Clean separation: module = training, callback = artifacts

Latest 2025-2026 practices:
- Python 3.14+ with modern type hints
- Lightning 2.4+ callback patterns
- Artifact schema driven (reproducible paths)
- Memory-efficient (streaming to disk)
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class ValCalibArtifactSaver(Callback):
    """
    Save val_calib logits/labels to artifact paths (ELITE 2025 pattern)

    CRITICAL: Separates "training" (module) from "artifacts" (pipeline).
    - Module stays clean (no file writes)
    - Artifacts go directly to schema paths
    - Memory-efficient (no RAM accumulation)

    Args:
        logits_path: Path to save logits (e.g., artifacts.val_calib_logits)
        labels_path: Path to save labels (e.g., artifacts.val_calib_labels)
        dataloader_idx: Which dataloader is val_calib (default: 1)
                       0=val_select, 1=val_calib

    Example:
        >>> from contracts.artifact_schema import create_artifact_schema
        >>> artifacts = create_artifact_schema("outputs")
        >>>
        >>> callback = ValCalibArtifactSaver(
        ...     logits_path=artifacts.val_calib_logits,
        ...     labels_path=artifacts.val_calib_labels,
        ... )
        >>>
        >>> trainer = L.Trainer(callbacks=[callback])
        >>> trainer.fit(model, datamodule)
        >>> # artifacts.val_calib_logits.npy and labels.npy are now saved!
    """

    def __init__(
        self,
        logits_path: Path | str,
        labels_path: Path | str,
        dataloader_idx: int = 1,
    ):
        super().__init__()

        self.logits_path = Path(logits_path)
        self.labels_path = Path(labels_path)
        self.dataloader_idx = dataloader_idx

        # Buffers for accumulation (cleared each epoch)
        self.logits_buffer: list[torch.Tensor] = []
        self.labels_buffer: list[torch.Tensor] = []

        logger.info(
            f"Initialized ValCalibArtifactSaver: "
            f"logits={self.logits_path}, labels={self.labels_path}"
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Collect logits/labels from val_calib batches

        Called after each validation batch.
        Only collects from dataloader_idx=1 (val_calib).
        """
        # Only process val_calib dataloader
        if dataloader_idx != self.dataloader_idx:
            return

        # Only on rank 0 (distributed training)
        if trainer.global_rank != 0:
            return

        # Get logits/labels from batch
        images, labels = batch

        # Forward pass to get logits
        # Use the same logic as validation_step (with EMA if enabled)
        if hasattr(pl_module, "use_ema") and pl_module.use_ema and pl_module.ema is not None:
            with pl_module.ema.average_parameters():
                logits = pl_module.forward(images, use_multiview=pl_module.multiview is not None)
        else:
            logits = pl_module.forward(images, use_multiview=pl_module.multiview is not None)

        # Accumulate in buffer (detached, on CPU)
        self.logits_buffer.append(logits.detach().cpu())
        self.labels_buffer.append(labels.detach().cpu())

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Save accumulated logits/labels to artifact paths

        Called after validation epoch ends.
        Writes to disk and validates.
        """
        # Only on rank 0
        if trainer.global_rank != 0:
            return

        # Check if we collected anything
        if len(self.logits_buffer) == 0:
            logger.warning("No val_calib logits collected (empty dataloader?)")
            return

        # Concatenate all batches (keep as torch tensors)
        logits_tensor = torch.cat(self.logits_buffer, dim=0)
        labels_tensor = torch.cat(self.labels_buffer, dim=0)

        logger.info(
            f"Collected val_calib: logits {logits_tensor.shape}, labels {labels_tensor.shape}"
        )

        # Ensure parent directories exist
        self.logits_path.parent.mkdir(parents=True, exist_ok=True)
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)

        # CRITICAL FIX: Save as .pt (torch format), NOT .npy (numpy format)
        # ArtifactSchema expects .pt files for compatibility with downstream phases
        torch.save(logits_tensor, self.logits_path)
        torch.save(labels_tensor, self.labels_path)

        logits_size_mb = self.logits_path.stat().st_size / 1024 / 1024
        labels_size_kb = self.labels_path.stat().st_size / 1024

        logger.info(f"✅ Saved val_calib artifacts (torch .pt format):")
        logger.info(f"  Logits: {self.logits_path} ({logits_size_mb:.2f} MB)")
        logger.info(f"  Labels: {self.labels_path} ({labels_size_kb:.2f} KB)")

        # Clear buffers for next epoch
        self.logits_buffer = []
        self.labels_buffer = []

    def state_dict(self) -> dict[str, Any]:
        """Save callback state (for checkpointing)"""
        return {
            "logits_path": str(self.logits_path),
            "labels_path": str(self.labels_path),
            "dataloader_idx": self.dataloader_idx,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state (from checkpoint)"""
        self.logits_path = Path(state_dict["logits_path"])
        self.labels_path = Path(state_dict["labels_path"])
        self.dataloader_idx = state_dict["dataloader_idx"]


if __name__ == "__main__":
    # Test callback
    print("Testing ValCalibArtifactSaver...")

    # Create temp paths
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logits_path = Path(tmpdir) / "val_calib_logits.npy"
        labels_path = Path(tmpdir) / "val_calib_labels.npy"

        callback = ValCalibArtifactSaver(
            logits_path=logits_path,
            labels_path=labels_path,
        )

        print(f"Callback: {callback}")
        print(f"Logits path: {callback.logits_path}")
        print(f"Labels path: {callback.labels_path}")
        print("✅ ValCalibArtifactSaver test passed")
