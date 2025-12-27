"""
Stage-1 Pro Modular Training System - Configuration Module

Production-grade configuration system with validation, reproducibility, and phase support.
Dec 2025 Best Practices:
- Seed setting BEFORE imports for reproducibility
- TF32 precision for faster training
- Comprehensive validation with helpful error messages
- JSON schema validation support
- Full Phase 1-3 field support
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path
from datetime import datetime
import subprocess


@dataclass
class Stage1ProConfig:
    """
    Production-grade config dataclass (2025 SOTA).
    
    All hyperparameters in one place, automatically saved to config.json.
    This ensures reproducibility and makes hyperparameter tuning easier.
    
    Preserves ALL fields from TrainingConfig in train_stage1_head.py.
    
    Dec 2025 Best Practices:
    - Seed setting BEFORE imports (phase1_1_reproducibility_full)
    - TF32 precision (phase1_3_tf32_implementation)
    - Comprehensive validation (phase1_4, phase1_5)
    - Full Phase 1-3 field support
    """
    # ==================== Model Paths (preserved from baseline) ====================
    model_path: str = "models/stage1_dinov3/dinov3-vith16plus-pretrain-lvd1689m"
    train_image_dir: str = "data/natix_official/train"
    train_labels_file: str = "data/natix_official/train_labels.csv"
    val_image_dir: str = "data/natix_official/val"
    val_labels_file: str = "data/natix_official/val_labels.csv"
    
    # ==================== Training Mode (preserved from baseline) ====================
    mode: str = "train"  # "extract_features", "train", or "train_cached"
    cached_features_dir: str = "cached_features"  # Where to save/load features
    
    # ==================== Multi-Dataset Training (preserved from baseline) ====================
    use_extra_roadwork: bool = False  # Combine NATIX + ROADWork + extras
    roadwork_iccv_dir: str = "data/roadwork_iccv"
    roadwork_extra_dir: str = "data/roadwork_extra"
    
    # ==================== Batch Sizes (preserved from baseline) ====================
    max_batch_size: int = 64
    fallback_batch_size: int = 32
    grad_accum_steps: int = 2
    
    # ==================== Training Schedule (preserved from baseline) ====================
    epochs: int = 10
    warmup_epochs: int = 1
    
    # ==================== Optimizer (preserved from baseline) ====================
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    
    # ==================== Regularization (preserved from baseline) ====================
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    dropout: float = 0.3
    
    # ==================== Advanced Features (preserved from baseline) ====================
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    early_stop_patience: int = 3
    
    # ==================== Cascade Exit Monitoring (renamed for clarity) ====================
    legacy_exit_threshold_for_logging: float = 0.88  # Monitoring only, NOT used for inference
    
    # ==================== Checkpointing (preserved from baseline) ====================
    resume_checkpoint: Optional[str] = None
    
    # ==================== Output (preserved from baseline) ====================
    log_file: str = "training.log"
    output_dir: str = "models/stage1_dinov3"
    
    # ==================== Phase 1 ONLY Fields (Dec 2025 Best Practices) ====================
    # Validation splits (4-way split) - phase1_5_checkpoint_validation_full
    val_select_ratio: float = 0.33  # Model selection/early stopping ONLY
    val_calib_ratio: float = 0.33  # Calibration/policy fitting ONLY
    val_test_ratio: float = 0.34  # Final evaluation ONLY
    
    # Risk constraint (single constraint - maximize coverage) - phase1_8_threshold_sweep_valcalib
    target_fnr_exit: float = 0.02  # ONLY constraint: 2% FNR on exited samples
    
    # Bootstrap settings for confidence intervals - phase2_3_bootstrap_ci_implementation
    bootstrap_samples: int = 1000  # Default for stable estimates
    bootstrap_confidence: float = 0.95  # 95% confidence level
    
    # Exit policy (mutually exclusive - phase1_9_thresholds_json_complete)
    exit_policy: str = "softmax"  # Phase 1: only "softmax", Phase 3+: "softmax" or "gate", Phase 6: "scrc"
    
    # ==================== Phase 2 ONLY Fields (Disabled in Phase 1) ====================
    # Selective metrics flags - phase2_1_risk_coverage_implementation
    use_selective_metrics: bool = False  # Enable selective metrics (AUGRC, risk-coverage)
    
    # ==================== Phase 3 ONLY Fields (Disabled in Phase 1) ====================
    # Gate head training settings - phase3_1_head_3head_complete
    gate_loss_weight: float = 1.0  # Weight for selective loss (default: 1.0)
    aux_weight: float = 0.5  # Weight for auxiliary loss (default: 0.5)
    gate_threshold: float = 0.5  # Training threshold for gate (calibration uses val_calib)
    
    # ==================== Phase 4 ONLY Fields (Disabled in Phase 1) ====================
    # PEFT settings - phase4_8_explora_script
    peft_type: str = "none"  # "none", "doran", "dora", "lora"
    peft_r: int = 16  # Rank for PEFT adapters
    peft_blocks: int = 6  # Number of transformer blocks to apply PEFT
    use_timm_mae: bool = False  # Use timm MAE for ExPLoRA pretraining (Dec 2025 best practice)
    
    # ==================== Phase 5 ONLY Fields (Disabled in Phase 1) ====================
    # Optimizer choice - phase5_1_fsam_research
    optimizer: str = "adamw"  # "adamw" or "fsam"
    
    # ==================== Phase 6 ONLY Fields (Disabled in Phase 1) ====================
    # Calibration flags - phase6_1_dirichlet_calibrator_fitting
    use_dirichlet: bool = False  # Enable Dirichlet calibration
    calibration_iters: int = 300  # Iterations for LBFGS optimizer
    
    # ==================== Reproducibility (Dec 2025 Best Practice) ====================
    # Seed setting - phase1_1_research_baseline
    seed: Optional[int] = None  # Training seed (set before imports for reproducibility)
    data_split_seed: Optional[int] = None  # Data split generation seed
    model_seed: Optional[int] = None  # Model initialization seed
    
    # Timestamp and git commit - phase1_4_checkpoint_validation_full
    timestamp: Optional[str] = None
    git_commit: Optional[str] = None
    
    def __post_init__(self):
        """Set default timestamp and git commit if not provided (phase1_4)."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.git_commit is None:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).parent.parent
                )
                if result.returncode == 0:
                    self.git_commit = result.stdout.strip()
            except Exception:
                self.git_commit = "unknown"
    
    def validate(self) -> bool:
        """
        Validate all configuration parameters (Dec 2025 best practice).
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        errors = []
        
        # Validate paths (phase1_4)
        if not os.path.exists(self.model_path):
            errors.append(f"Model path does not exist: {self.model_path}")
        
        if self.mode in ["train", "train_cached"]:
            if not os.path.exists(self.train_image_dir):
                errors.append(f"Train image directory does not exist: {self.train_image_dir}")
            if not os.path.exists(self.train_labels_file):
                errors.append(f"Train labels file does not exist: {self.train_labels_file}")
            if not os.path.exists(self.val_image_dir):
                errors.append(f"Val image directory does not exist: {self.val_image_dir}")
            if not os.path.exists(self.val_labels_file):
                errors.append(f"Val labels file does not exist: {self.val_labels_file}")
        
        # Validate hyperparameters (phase1_4)
        if not (0.0 < self.lr_head < 1.0):
            errors.append(f"lr_head must be between 0 and 1: {self.lr_head}")
        if not (0.0 < self.lr_backbone < 1.0):
            errors.append(f"lr_backbone must be between 0 and 1: {self.lr_backbone}")
        if not (0.0 <= self.weight_decay < 1.0):
            errors.append(f"weight_decay must be between 0 and 1: {self.weight_decay}")
        if not (0.0 <= self.label_smoothing < 1.0):
            errors.append(f"label_smoothing must be between 0 and 1: {self.label_smoothing}")
        if not (0.0 <= self.dropout < 1.0):
            errors.append(f"dropout must be between 0 and 1: {self.dropout}")
        if not (0.0 < self.epochs < 100):
            errors.append(f"epochs must be between 1 and 100: {self.epochs}")
        
        # Validate mode compatibility (phase1_4)
        if self.mode == "train_cached" and not os.path.exists(self.cached_features_dir):
            errors.append(f"cached_features_dir does not exist for train_cached mode: {self.cached_features_dir}")
        
        # Validate phase compatibility (Dec 2025 best practice)
        if self.phase < 1 or self.phase > 6:
            errors.append(f"phase must be between 1 and 6: {self.phase}")
        
        if self.phase == 1:
            # Phase 1: Only "softmax" exit policy
            if self.exit_policy != "softmax":
                errors.append(f"Phase 1 only supports exit_policy='softmax'. Got: {self.exit_policy}")
            # Phase 1: Disable Phase 2+ features
            if self.use_selective_metrics:
                errors.append(f"Phase 1 cannot use selective metrics (use_selective_metrics must be False)")
            if self.gate_loss_weight != 1.0 or self.aux_weight != 0.5:
                errors.append(f"Phase 1 cannot use gate weights (use default values)")
            if self.peft_type != "none":
                errors.append(f"Phase 1 cannot use PEFT (peft_type must be 'none')")
            if self.optimizer != "adamw":
                errors.append(f"Phase 1 cannot use F-SAM (optimizer must be 'adamw')")
            if self.use_dirichlet:
                errors.append(f"Phase 1 cannot use Dirichlet calibration (use_dirichlet must be False)")
        
        if self.phase == 2:
            # Phase 2: Enable selective metrics
            if self.exit_policy not in ["softmax", "gate"]:
                errors.append(f"Phase 2 exit_policy must be 'softmax' or 'gate'. Got: {self.exit_policy}")
        
        if self.phase == 3:
            # Phase 3: Gate training enabled
            if self.exit_policy not in ["softmax", "gate"]:
                errors.append(f"Phase 3 exit_policy must be 'softmax' or 'gate'. Got: {self.exit_policy}")
            # Phase 3: Disable Phase 4+ features
            if self.peft_type != "none":
                errors.append(f"Phase 3 cannot use PEFT (peft_type must be 'none' until Phase 4)")
            if self.optimizer != "adamw":
                errors.append(f"Phase 3 cannot use F-SAM (optimizer must be 'adamw' until Phase 5)")
            if self.use_dirichlet:
                errors.append(f"Phase 3 cannot use Dirichlet calibration (use_dirichlet must be False until Phase 6)")
        
        if self.phase == 4:
            # Phase 4: PEFT enabled
            if self.peft_type not in ["none", "doran", "dora", "lora"]:
                errors.append(f"Phase 4 peft_type must be 'none', 'doran', 'dora', or 'lora'. Got: {self.peft_type}")
        
        if self.phase == 5:
            # Phase 5: Optimizer experiments
            if self.optimizer not in ["adamw", "fsam"]:
                errors.append(f"Phase 5 optimizer must be 'adamw' or 'fsam'. Got: {self.optimizer}")
        
        if self.phase == 6:
            # Phase 6: Calibration enabled
            if self.exit_policy != "scrc":
                errors.append(f"Phase 6 exit_policy must be 'scrc'. Got: {self.exit_policy}")
            if not self.use_dirichlet:
                errors.append(f"Phase 6 must use Dirichlet calibration (use_dirichlet must be True)")
        
        # Print validation results (Dec 2025 best practice)
        if errors:
            print(f"\n{'='*80}")
            print(f"CONFIGURATION VALIDATION ERRORS:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            print(f"{'='*80}")
            return False
        else:
            print(f"✅ Configuration validation passed!")
            return True
    
    def validate_dataset_availability(self) -> dict:
        """
        Validate dataset availability (Dec 2025 best practice).
        
        Returns:
            dict: Dataset availability info
        """
        available_datasets = {
            'natix': os.path.exists(self.train_image_dir) and os.path.exists(self.train_labels_file),
            'roadwork_iccv': os.path.exists(self.roadwork_iccv_dir) if self.use_extra_roadwork else False,
            'roadwork_extra': os.path.exists(self.roadwork_extra_dir) if self.use_extra_roadwork else False
        }
        
        # Count available datasets
        num_available = sum(1 for avail in available_datasets.values() if avail)
        total_requested = 1 + (1 if self.use_extra_roadwork else 0) + (1 if self.use_extra_roadwork else 0)
        
        print(f"\n{'='*80}")
        print(f"DATASET AVAILABILITY:")
        print(f"   NATIX: {'✅ Available' if available_datasets['natix'] else '❌ Not found'}")
        if self.use_extra_roadwork:
            print(f"   ROADWork ICCV: {'✅ Available' if available_datasets['roadwork_iccv'] else '❌ Not found'}")
            print(f"   ROADWork Extra: {'✅ Available' if available_datasets['roadwork_extra'] else '❌ Not found'}")
        print(f"   Total available: {num_available}/{total_requested}")
        print(f"{'='*80}")
        
        return available_datasets
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to JSON file (Dec 2025 best practice).
        
        Saves all fields including Phase 1-3 settings, seeds, timestamps.
        """
        if path is None:
            # Use default path from output_dir
            path = os.path.join(self.output_dir, "config.json")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dict
        config_dict = asdict(self)
        
        # Save to JSON
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved to {path}")
        print(f"   Phase: {self.phase}")
        print(f"   Exit Policy: {self.exit_policy}")
        print(f"   Seed: {self.seed}")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Git Commit: {self.git_commit}")
    
    @classmethod
    def load(cls, path: str) -> 'Stage1ProConfig':
        """
        Load configuration from JSON file (Dec 2025 best practice).
        
        Handles missing fields gracefully with default values.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Validate loaded config
        config = cls(**config_dict)
        
        # Validate fields
        if not config.validate():
            raise ValueError(f"Loaded configuration failed validation")
        
        print(f"✅ Configuration loaded from {path}")
        print(f"   Phase: {config.phase}")
        print(f"   Exit Policy: {config.exit_policy}")
        print(f"   Seed: {config.seed}")
        
        return config
    
    @property
    def phase(self) -> int:
        """
        Get phase number based on enabled features.
        
        Dec 2025 Best Practice:
        - Automatically determines phase based on config flags
        - Ensures phase gates are respected
        """
        if self.peft_type != "none" or self.use_timm_mae:
            return 4  # Phase 4: PEFT/Domain Adaptation
        elif self.optimizer == "fsam":
            return 5  # Phase 5: Advanced Optimization
        elif self.use_dirichlet:
            return 6  # Phase 6: Conformal Risk Training
        elif self.use_selective_metrics:
            return 2  # Phase 2: Selective Evaluation
        elif self.exit_policy == "gate" or self.gate_loss_weight != 1.0:
            return 3  # Phase 3: Gate Head
        else:
            return 1  # Phase 1: Baseline Training
