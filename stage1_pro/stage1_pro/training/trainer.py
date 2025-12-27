import torch
import torch.nn.functional as F
from typing import Dict, Any
from tqdm import tqdm
import numpy as np
from .optimizers import build_optimizer
from .schedulers import build_scheduler
from .losses import CrossEntropyLoss, SelectiveLoss, RiskLoss, AuxiliaryLoss
from .ema import EMAModel


class Stage1Trainer:
    """
    Multi-phase trainer supporting Phase 1-2 training modes.

    Phase 1: Single-head baseline (CE loss only)
    Phase 2: Risk-aware training (CE + selective + aux losses)
    """

    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Build losses based on phase
        self.ce_loss = CrossEntropyLoss(label_smoothing=config.label_smoothing)

        if hasattr(config, "use_dirichlet") and config.use_dirichlet:
            self.selective_loss = SelectiveLoss(lambda_cov=1.0)
            self.risk_loss = RiskLoss(lambda_risk=1.0)
            self.aux_loss = AuxiliaryLoss(alpha_aux=0.5)
        else:
            self.selective_loss = None
            self.risk_loss = None
            self.aux_loss = None

        # Build optimizer
        self.optimizer = build_optimizer(model, config)

        # Build scheduler
        num_training_steps = config.epochs * (
            config.num_train_samples // config.max_batch_size
        )
        self.scheduler = build_scheduler(self.optimizer, config, num_training_steps)

        # EMA
        self.ema = EMAModel(model, config.ema_decay) if config.use_ema else None

        # Mixed precision
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if config.use_amp and torch.cuda.is_available()
            else None
        )

    def train_epoch(self, train_loader, epoch: int, mode: str = "train"):
        """Train one epoch supporting Phase 1 and Phase 2 modes."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if mode == "train_cached":
                features = images
                outputs = self.model(features)
            else:
                with torch.no_grad():
                    backbone_outputs = self.model.backbone(pixel_values=images)
                    features = backbone_outputs.last_hidden_state[:, 0, :]

                outputs = self.model(features)

            cls_logits = outputs["cls_logits"]

            if mode == "train_risk" and self.selective_loss:
                # Phase 2: Risk-aware training
                gate_logits = outputs.get("gate_logits", None)
                aux_logits = outputs.get("aux_logits", None)

                # Compute losses
                loss = self.ce_loss(cls_logits, labels)

                if gate_logits is not None:
                    loss = loss + self.selective_loss(cls_logits, gate_logits, labels)

                if aux_logits is not None:
                    loss = loss + self.aux_loss(cls_logits, aux_logits, labels)

                loss = loss / self.config.grad_accum_steps
            else:
                # Phase 1: Standard training
                loss = self.ce_loss(cls_logits, labels) / self.config.grad_accum_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    if self.ema:
                        self.ema.update()
            else:
                loss.backward()

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    if self.ema:
                        self.ema.update()

            total_loss += loss.item() * self.config.grad_accum_steps

            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * self.config.grad_accum_steps:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader, mode: str = "train"):
        """Validate supporting Phase 1 and Phase 2 modes."""
        if self.ema:
            self.ema.apply_shadow()

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_gate_probs = []

        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(self.device), labels.to(self.device)

            if mode == "train_cached":
                features = images
                outputs = self.model(features)
            else:
                backbone_outputs = self.model.backbone(pixel_values=images)
                features = backbone_outputs.last_hidden_state[:, 0, :]
                outputs = self.model(features)

            cls_logits = outputs["cls_logits"]

            loss = self.ce_loss(cls_logits, labels)
            total_loss += loss.item()

            probs = F.softmax(cls_logits, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            if "gate_probs" in outputs:
                all_gate_probs.append(outputs["gate_probs"].cpu())

            preds = probs.argmax(dim=-1)
            all_preds.append(preds.cpu())

        if self.ema:
            self.ema.restore()

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        accuracy = (all_preds == all_labels).float().mean().item()

        result = {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy,
            "preds": all_preds,
            "labels": all_labels,
            "probs": all_probs.numpy(),
            "ece": self._compute_ece(all_probs.numpy(), all_labels.numpy()),
        }

        if len(all_gate_probs) > 0:
            result["gate_probs"] = torch.cat(all_gate_probs).numpy()

        return result

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
        """Compute Expected Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_size = in_bin.sum()

            if bin_size > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += (bin_size / len(labels)) * abs(avg_confidence - avg_accuracy)

        return ece
