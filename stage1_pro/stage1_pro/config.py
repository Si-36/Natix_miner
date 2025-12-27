import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
import json
import hashlib
import os


class Stage1ProConfig:
    """
    Configuration for Stage-1 Pro training system.
    Phase 1: Baseline training + threshold sweep + deploy bundle
    """

    def __init__(self, **kwargs):
        # Preserve EXACT fields from train_stage1_head.py
        self.model_path = kwargs.get("model_path", "facebook/dinov2-large")
        self.train_image_dir = kwargs.get("train_image_dir", None)
        self.train_labels_file = kwargs.get("train_labels_file", None)
        self.val_image_dir = kwargs.get("val_image_dir", None)
        self.val_labels_file = kwargs.get("val_labels_file", None)
        self.mode = kwargs.get("mode", "train")  # extract_features, train_cached, train
        self.cached_features_dir = kwargs.get("cached_features_dir", "cache")
        self.use_extra_roadwork = kwargs.get("use_extra_roadwork", False)
        self.roadwork_iccv_dir = kwargs.get("roadwork_iccv_dir", None)
        self.roadwork_extra_dir = kwargs.get("roadwork_extra_dir", None)
        self.max_batch_size = kwargs.get("max_batch_size", 32)
        self.fallback_batch_size = kwargs.get("fallback_batch_size", 16)
        self.grad_accum_steps = kwargs.get("grad_accum_steps", 1)
        self.epochs = kwargs.get("epochs", 50)
        self.warmup_epochs = kwargs.get("warmup_epochs", 3)
        self.lr_head = kwargs.get("lr_head", 1e-4)
        self.lr_backbone = kwargs.get("lr_backbone", 1e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.dropout = kwargs.get("dropout", 0.1)
        self.label_smoothing = kwargs.get("label_smoothing", 0.1)
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.use_amp = kwargs.get("use_amp", True)
        self.use_ema = kwargs.get("use_ema", True)
        self.ema_decay = kwargs.get("ema_decay", 0.9999)
        self.early_stop_patience = kwargs.get("early_stop_patience", 10)
        self.legacy_exit_threshold_for_logging = kwargs.get(
            "legacy_exit_threshold_for_logging", 0.88
        )  # Monitoring ONLY, not inference
        self.resume_checkpoint = kwargs.get("resume_checkpoint", None)
        self.output_dir = kwargs.get("output_dir", "./outputs")
        self.log_file = kwargs.get("log_file", "training.csv")

        # Phase 1 ONLY fields
        self.val_select_ratio = kwargs.get("val_select_ratio", 0.5)
        self.target_fnr_exit = kwargs.get(
            "target_fnr_exit", 0.02
        )  # 2% FNR constraint ONLY
        # CRITICAL: NO target_coverage parameter - maximize coverage subject to FNR constraint
        self.exit_policy = kwargs.get("exit_policy", "softmax")  # Phase 1: only softmax

        # Phase 2+ fields (disabled in Phase 1)
        self.use_dirichlet = kwargs.get("use_dirichlet", False)
        self.calibration_iters = kwargs.get("calibration_iters", 300)

        # Phase 3+ fields (disabled in Phase 1-2)
        self.gate_loss_weight = kwargs.get("gate_loss_weight", 0.0)
        self.aux_weight = kwargs.get("aux_weight", 0.0)

        # Phase 4+ fields (disabled in Phase 1-3)
        self.peft_type = kwargs.get("peft_type", "none")
        self.peft_r = kwargs.get("peft_r", 16)
        self.peft_blocks = kwargs.get("peft_blocks", 6)

        # Phase 5+ fields (disabled in Phase 1-4)
        self.optimizer = kwargs.get("optimizer", "adamw")  # fsam for Phase 5+

        # Seed for reproducibility
        self.seed = kwargs.get("seed", 42)
        self.timestamp = self._get_timestamp()
        self.git_commit = self._get_git_commit()

        self._validate()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_git_commit(self) -> Optional[str]:
        """Get git commit hash if available."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return None

    def _validate(self):
        """Validate configuration."""
        self.validate_paths()
        self.validate_hyperparameters()
        self.validate_mode_compatibility()
        self.validate_phase_compatibility()

    def validate_paths(self):
        """Check that all required paths exist."""
        missing_paths = []

        if self.train_image_dir and not os.path.exists(self.train_image_dir):
            missing_paths.append(self.train_image_dir)
        if self.train_labels_file and not os.path.exists(self.train_labels_file):
            missing_paths.append(self.train_labels_file)
        if self.val_image_dir and not os.path.exists(self.val_image_dir):
            missing_paths.append(self.val_image_dir)
        if self.val_labels_file and not os.path.exists(self.val_labels_file):
            missing_paths.append(self.val_labels_file)

        if missing_paths:
            raise ValueError(f"Missing required paths: {missing_paths}")

    def validate_hyperparameters(self):
        """Validate hyperparameter ranges."""
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.lr_head <= 0:
            raise ValueError(f"lr_head must be > 0, got {self.lr_head}")
        if self.lr_backbone < 0:
            raise ValueError(f"lr_backbone must be >= 0, got {self.lr_backbone}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if not (0 < self.warmup_epochs < self.epochs):
            raise ValueError(
                f"warmup_epochs must be in (0, epochs), got {self.warmup_epochs}"
            )
        if not (0 < self.target_fnr_exit <= 0.1):
            raise ValueError(
                f"target_fnr_exit must be in (0, 0.1], got {self.target_fnr_exit}"
            )

    def validate_mode_compatibility(self):
        """Validate mode compatibility with other settings."""
        if self.mode == "extract_features":
            # extract_features doesn't need cached_features_dir to exist
            pass
        elif self.mode == "train_cached":
            if self.cached_features_dir == "cache":
                raise ValueError(
                    "train_cached mode requires cached_features_dir to be set"
                )
        elif self.mode == "train":
            if self.train_image_dir is None:
                raise ValueError("train mode requires train_image_dir to be set")
            if self.train_labels_file is None:
                raise ValueError("train mode requires train_labels_file to be set")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def validate_phase_compatibility(self):
        """Validate that settings match current phase (Phase 1 only)."""
        # Phase 1: Only exit_policy='softmax' allowed
        if self.exit_policy != "softmax":
            raise ValueError(
                f"Phase 1 only supports exit_policy='softmax', got {self.exit_policy}"
            )

        # Phase 1: PEFT should be disabled
        if self.peft_type != "none":
            raise ValueError(f"Phase 1 requires peft_type='none', got {self.peft_type}")

        # Phase 1: Optimizer should be adamw (no fsam)
        if self.optimizer != "adamw":
            raise ValueError(
                f"Phase 1 requires optimizer='adamw', got {self.optimizer}"
            )

        # Phase 1: Gate loss should be disabled
        if self.gate_loss_weight > 0:
            raise ValueError(
                f"Phase 1 requires gate_loss_weight=0, got {self.gate_loss_weight}"
            )

        # Phase 1: Dirichlet should be disabled
        if self.use_dirichlet:
            raise ValueError(f"Phase 1 requires use_dirichlet=False")

    def save(self, path: str):
        """Save configuration to JSON file."""
        config_dict = {
            "model_path": self.model_path,
            "train_image_dir": self.train_image_dir,
            "train_labels_file": self.train_labels_file,
            "val_image_dir": self.val_image_dir,
            "val_labels_file": self.val_labels_file,
            "mode": self.mode,
            "cached_features_dir": self.cached_features_dir,
            "use_extra_roadwork": self.use_extra_roadwork,
            "roadwork_iccv_dir": self.roadwork_iccv_dir,
            "roadwork_extra_dir": self.roadwork_extra_dir,
            "max_batch_size": self.max_batch_size,
            "fallback_batch_size": self.fallback_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "lr_head": self.lr_head,
            "lr_backbone": self.lr_backbone,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "max_grad_norm": self.max_grad_norm,
            "use_amp": self.use_amp,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "early_stop_patience": self.early_stop_patience,
            "legacy_exit_threshold_for_logging": self.legacy_exit_threshold_for_logging,
            "resume_checkpoint": self.resume_checkpoint,
            "output_dir": self.output_dir,
            "log_file": self.log_file,
            "val_select_ratio": self.val_select_ratio,
            "target_fnr_exit": self.target_fnr_exit,
            "exit_policy": self.exit_policy,
            "use_dirichlet": self.use_dirichlet,
            "calibration_iters": self.calibration_iters,
            "gate_loss_weight": self.gate_loss_weight,
            "aux_weight": self.aux_weight,
            "peft_type": self.peft_type,
            "peft_r": self.peft_r,
            "peft_blocks": self.peft_blocks,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Stage1ProConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
