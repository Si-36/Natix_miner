"""
Lightning Module - Production-Grade Training Module

Complete training module with:
- DINOv3 backbone + classification head
- Training step with cross-entropy loss
- Validation step with accuracy metrics
- EMA (Exponential Moving Average)
- Multi-view inference ready (extensible)
- Proper logging and checkpointing

Latest 2025-2026 practices:
- Python 3.14+ with modern type hints
- Lightning 2.4+ patterns
- Modular architecture
- Production-ready training
"""

import logging
from typing import Optional, Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, MetricCollection

from models.backbone import DINOv3Backbone, create_dinov3_backbone
from models.head import ClassificationHead, create_classification_head
from models.multi_view import (
    MultiViewDINOv3,
    TopKMeanAggregator,
    AttentionAggregator,
    create_multiview_model,
)

logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average for model weights

    EMA maintains a moving average of model parameters:
        ema_param = decay * ema_param + (1 - decay) * model_param

    Benefits:
    - Smoother convergence
    - Better generalization (+0.5-1.5% accuracy)
    - More stable predictions

    Args:
        model: Model to track
        decay: EMA decay rate (default: 0.9999)

    Example:
        >>> ema = EMA(model, decay=0.9999)
        >>> # After each training step:
        >>> ema.update(model)
        >>> # Use EMA weights for validation:
        >>> with ema.average_parameters():
        ...     val_loss = validate(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay

        # Store shadow parameters (EMA weights)
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        logger.info(f"Initialized EMA with decay={decay}")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA parameters

        Call this after each training step.

        Args:
            model: Model with updated weights
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    @torch.no_grad()
    def apply_shadow(self) -> None:
        """Apply EMA weights to model (for validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self) -> None:
        """Restore original weights (after validation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def average_parameters(self):
        """
        Context manager to temporarily use EMA weights

        Example:
            >>> with ema.average_parameters():
            ...     val_loss = validate(model)
        """
        return EMAContext(self)


class EMAContext:
    """Context manager for EMA weights"""

    def __init__(self, ema: EMA):
        self.ema = ema

    def __enter__(self):
        self.ema.apply_shadow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ema.restore()
        return False


class DINOv3Classifier(L.LightningModule):
    """
    DINOv3-based Image Classifier

    Complete training module combining:
    - DINOv3 backbone (frozen or trainable)
    - Classification head
    - Cross-entropy loss
    - AdamW optimizer
    - Cosine annealing LR scheduler
    - EMA for better convergence
    - Comprehensive metrics

    Args:
        backbone_name: DINOv3 HuggingFace model ID or local checkpoint path
                      (e.g., "facebook/dinov3-vith16-pretrain-lvd1689m")
        num_classes: Number of output classes (13 for NATIX)
        freeze_backbone: If True, freeze backbone weights
        head_type: Type of classification head ("linear" or "doran")
        dropout_rate: Dropout probability (0.3 recommended)
        learning_rate: Initial learning rate (1e-4 recommended)
        weight_decay: AdamW weight decay (0.01 recommended)
        use_ema: If True, use EMA (recommended)
        ema_decay: EMA decay rate (0.9999 recommended)
        use_multiview: If True, use multi-view inference (for validation/test only)
        multiview_aggregation: Aggregation strategy ("topk_mean" or "attention")
        multiview_topk: K for top-k aggregation (only used if aggregation="topk_mean")
        multiview_grid_size: Grid dimensions for tiles (default: (3, 3))
        multiview_overlap: Overlap ratio between tiles (default: 0.15)

    Example:
        >>> model = DINOv3Classifier(
        ...     backbone_name="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        ...     num_classes=13,
        ...     freeze_backbone=True,
        ...     learning_rate=1e-4
        ... )
        >>> trainer = L.Trainer(max_epochs=10)
        >>> trainer.fit(model, datamodule=datamodule)
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        num_classes: int = 13,
        freeze_backbone: bool = True,
        head_type: str = "linear",
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_multiview: bool = False,
        multiview_aggregation: str = "topk_mean",
        multiview_topk: int = 2,
        multiview_grid_size: tuple[int, int] = (3, 3),
        multiview_overlap: float = 0.15,
    ):
        super().__init__()

        # Save hyperparameters (Lightning feature)
        self.save_hyperparameters()

        # Model architecture - CRITICAL: Use ModuleDict for clean EMA scope
        backbone = create_dinov3_backbone(
            model_name=backbone_name,
            freeze=freeze_backbone,
        )

        head = create_classification_head(
            hidden_size=backbone.hidden_size,
            num_classes=num_classes,
            head_type=head_type,
            dropout_rate=dropout_rate,
        )

        # CRITICAL FIX: Wrap in ModuleDict so EMA tracks ONLY model params
        # (not metrics, optimizers, or other LightningModule state)
        self.net = nn.ModuleDict({
            "backbone": backbone,
            "head": head,
        })

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
            },
            prefix="train/",
        )

        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
            },
            prefix="val/",
        )

        # EMA
        self.use_ema = use_ema
        self.ema = None  # Created in configure_model()

        # Multi-view wrapper (if enabled)
        self.use_multiview = use_multiview
        self.multiview = None  # Created here if use_multiview=True

        if use_multiview:
            # Create aggregator based on strategy
            if multiview_aggregation == "topk_mean":
                aggregator = TopKMeanAggregator(topk=multiview_topk)
            elif multiview_aggregation == "attention":
                aggregator = AttentionAggregator(num_classes=num_classes)
            else:
                raise ValueError(
                    f"Unknown aggregation strategy: {multiview_aggregation}. "
                    f"Valid options: 'topk_mean', 'attention'"
                )

            # Create multi-view wrapper
            self.multiview = MultiViewDINOv3(
                backbone=self.net["backbone"],
                head=self.net["head"],
                aggregator=aggregator,
                grid_size=multiview_grid_size,
                overlap=multiview_overlap,
            )
            logger.info(
                f"Multi-view enabled: {multiview_aggregation} aggregation, "
                f"grid_size={multiview_grid_size}, overlap={multiview_overlap:.1%}"
            )

        logger.info(
            f"Initialized DINOv3Classifier: {backbone_name} + {head_type} head "
            f"({num_classes} classes)"
        )

    def configure_model(self) -> None:
        """
        Configure model (called after model is moved to device)

        This is where we initialize EMA (needs model on correct device).
        """
        if self.use_ema and self.ema is None:
            # CRITICAL FIX: EMA tracks self.net (ModuleDict), not entire LightningModule
            self.ema = EMA(self.net, decay=self.hparams.ema_decay)
            logger.info("EMA initialized (tracking only model params)")

    def forward(
        self,
        images: torch.Tensor,
        use_multiview: bool = False,
        content_boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (2025-12-29 with content-aware multi-view support)

        CRITICAL: Multi-view is ONLY used during validation/test, NOT training!
        Training always uses single-view (multi-view is too slow).

        Args:
            images: Input images [B, 3, H, W]
            use_multiview: If True and multiview is enabled, use multi-view inference
            content_boxes: Optional content boxes [B, 4] in (x1, y1, x2, y2) format
                          Only used when use_multiview=True (letterbox mode).

        Returns:
            logits: Class logits [B, num_classes]
        """
        # Multi-view forward (if requested and available)
        if use_multiview and self.multiview is not None:
            return self.multiview(images, content_boxes=content_boxes)  # [B, num_classes]

        # Single-view forward (default)
        features = self.net["backbone"](images)  # [B, hidden_size]
        logits = self.net["head"](features)  # [B, num_classes]

        return logits

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step

        CRITICAL: Always uses single-view (multi-view too slow for training)

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            loss: Training loss
        """
        images, labels = batch

        # Forward pass (ALWAYS single-view during training)
        logits = self.forward(images, use_multiview=False)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute metrics
        self.train_metrics.update(logits, labels)

        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """
        Called after each training batch

        This is where we update EMA.
        """
        if self.use_ema and self.ema is not None:
            # CRITICAL FIX: Update EMA with self.net (not self)
            self.ema.update(self.net)

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of training epoch

        Log aggregated metrics.
        """
        # Compute and log metrics
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # Reset metrics
        self.train_metrics.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Validation step (2025-12-29 with content-aware multi-view support)

        CRITICAL:
        - dataloader_idx=0: val_select (for early stopping)
        - dataloader_idx=1: val_calib (for calibration - save logits!)
        - Uses multi-view if enabled (only for validation, not training!)
        - Uses EMA weights if enabled
        - Supports both batch formats: (images, labels) and (images, labels, content_boxes)

        Args:
            batch: Tuple of (images, labels) OR (images, labels, content_boxes)
            batch_idx: Batch index
            dataloader_idx: Which dataloader (0=val_select, 1=val_calib)

        Returns:
            loss: Validation loss
        """
        # Unpack batch (support both formats)
        if len(batch) == 3:
            # Letterbox mode: (images, labels, content_boxes)
            images, labels, content_boxes = batch
        elif len(batch) == 2:
            # Standard mode: (images, labels)
            images, labels = batch
            content_boxes = None
        else:
            raise ValueError(f"Expected batch with 2 or 3 elements, got {len(batch)}")

        # Use multi-view if available (for better validation accuracy)
        use_mv = self.multiview is not None

        # Forward pass (with EMA if enabled, with multi-view if enabled)
        if self.use_ema and self.ema is not None:
            with self.ema.average_parameters():
                logits = self.forward(images, use_multiview=use_mv, content_boxes=content_boxes)
        else:
            logits = self.forward(images, use_multiview=use_mv, content_boxes=content_boxes)

        # Compute loss
        loss = self.criterion(logits, labels)

        # CRITICAL: Handle different dataloaders
        if dataloader_idx == 0:
            # val_select: For model selection (early stopping)
            self.val_metrics.update(logits, labels)

            # CRITICAL FIX: Explicitly log val_select/acc for early stopping
            # (MetricCollection logs as "val/acc", but config monitors "val_select/acc")
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()

            self.log("val_select/loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("val_select/acc", acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

            # Also log without prefix for compatibility
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        else:
            # val_calib: For calibration (save logits!)
            self.log("val_calib/loss", loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)

            # CRITICAL FIX: Return logits/labels so callback can reuse them
            # (no extra forward pass needed in callback)
            return {
                "loss": loss,
                "logits": logits.detach(),
                "labels": labels.detach(),
            }

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch

        Log aggregated metrics (for val_select only).

        NOTE: val_calib logits/labels are now handled by ValCalibArtifactSaver callback
        (reuses validation_step outputs, no extra forward pass)
        """
        # Compute and log metrics (for val_select)
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

        # Reset metrics
        self.val_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Configure optimizer and LR scheduler

        Uses:
        - AdamW optimizer (best for vision transformers)
        - Cosine annealing LR scheduler

        Returns:
            Dictionary with optimizer and scheduler config
        """
        # Get trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if not trainable_params:
            raise RuntimeError("No trainable parameters found!")

        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        # Cosine annealing LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Prediction step (for inference)

        Uses multi-view if enabled (recommended for best accuracy).

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Batch index

        Returns:
            Dictionary with predictions and metadata
        """
        images, labels = batch

        # Use multi-view if available (for best inference accuracy)
        use_mv = self.multiview is not None

        # Forward pass (with EMA if enabled, with multi-view if enabled)
        if self.use_ema and self.ema is not None:
            with self.ema.average_parameters():
                logits = self.forward(images, use_multiview=use_mv)
        else:
            logits = self.forward(images, use_multiview=use_mv)

        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "labels": labels,
        }

    @property
    def num_parameters(self) -> int:
        """Total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DINOv3Classifier(\n"
            f"  backbone={self.hparams.backbone_name},\n"
            f"  head={self.hparams.head_type},\n"
            f"  num_classes={self.hparams.num_classes},\n"
            f"  freeze_backbone={self.hparams.freeze_backbone},\n"
            f"  use_ema={self.use_ema},\n"
            f"  params={self.num_parameters:,},\n"
            f"  trainable_params={self.num_trainable_parameters:,}\n"
            f")"
        )


if __name__ == "__main__":
    # Test Lightning Module
    print("Testing DINOv3Classifier...")

    # Create model (will load from HuggingFace)
    model = DINOv3Classifier(
        backbone_name="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        num_classes=13,
        freeze_backbone=True,
        head_type="linear",
        learning_rate=1e-4,
        use_ema=True,
    )

    print(f"\n{model}")

    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.randint(0, 13, (2,))

    print(f"\nInput shape: {dummy_images.shape}")

    try:
        logits = model(dummy_images)
        print(f"Output shape: {logits.shape}")
        print(f"Expected: [2, 13]")

        assert logits.shape == (2, 13), "Output shape mismatch!"

        # Test training step
        loss = model.training_step((dummy_images, dummy_labels), 0)
        print(f"\nTraining loss: {loss.item():.4f}")

        # Test validation step
        val_loss = model.validation_step((dummy_images, dummy_labels), 0)
        print(f"Validation loss: {val_loss.item():.4f}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print("(This is expected if you don't have pretrained weights or internet)")
