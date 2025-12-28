"""
Pydantic 2.9 + Hydra-Zen Configurations (2025 Best Practices)
Type-safe, validated configurations with zero boilerplate
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================================
# DATA CONFIG
# ============================================================================

class DataConfig(BaseModel):
    """Dataset configuration with validation"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    # Paths
    data_dir: Path = Field(default=Path("data/roadwork"), description="Root data directory")
    train_image_dir: Path = Field(default=Path("data/roadwork/train"), description="Training images")
    train_labels_file: Path = Field(default=Path("data/roadwork/train_labels.csv"), description="Training labels")
    
    # Dataset selection
    use_extra_roadwork: bool = Field(default=False, description="Use extra ROADWork dataset")
    roadwork_iccv_dir: Optional[Path] = Field(default=None, description="ICCV ROADWork directory")
    roadwork_extra_dir: Optional[Path] = Field(default=None, description="Extra ROADWork directory")
    
    # Dataloader
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size")
    num_workers: int = Field(default=4, ge=0, le=16, description="Number of dataloader workers")
    pin_memory: bool = Field(default=True, description="Pin GPU memory")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")
    
    # Augmentation
    augmentation_strength: float = Field(default=0.8, ge=0.0, le=1.0, description="Augmentation magnitude")
    use_tta: bool = Field(default=False, description="Test-time augmentation")
    tta_flips: List[str] = Field(default=["horizontal"], description="TTA flips")
    
    # Splits
    splits_file: Path = Field(default=Path("splits.json"), description="Train/val splits file")
    train_split_ratio: float = Field(default=0.7, ge=0.5, le=0.9, description="Training split ratio")
    val_select_ratio: float = Field(default=0.15, ge=0.1, le=0.25, description="Validation select ratio")
    val_calib_ratio: float = Field(default=0.15, ge=0.1, le=0.25, description="Validation calibration ratio")


# ============================================================================
# MODEL CONFIG
# ============================================================================

class BackboneConfig(BaseModel):
    """DINOv3 backbone configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    model_name: str = Field(default="dinov2_vitl14", description="DINOv3 model name")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    freeze_backbone: bool = Field(default=True, description="Freeze backbone parameters")
    
    # PEFT
    use_peft: bool = Field(default=False, description="Use PEFT (LoRA/DoRA)")
    peft_type: Literal["lora", "dora", "doran", "explora"] = Field(
        default="lora", description="PEFT type"
    )
    peft_r: int = Field(default=8, ge=1, le=128, description="PEFT rank")
    peft_alpha: int = Field(default=16, ge=1, le=256, description="PEFT alpha")
    peft_dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="PEFT dropout")
    target_modules: List[str] = Field(
        default=["qkv", "mlp.fc1", "mlp.fc2"], description="Target modules for PEFT"
    )
    
    # Flash Attention
    use_flash_attn: bool = Field(default=False, description="Use Flash Attention 3")
    
    # Compile
    compile_model: bool = Field(default=True, description="Use torch.compile (30-50% speedup)")
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="max-autotune", description="torch.compile mode"
    )


class HeadConfig(BaseModel):
    """Classification head configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    hidden_dim: int = Field(default=1024, ge=128, le=4096, description="Hidden dimension")
    num_classes: int = Field(default=2, ge=2, le=100, description="Number of classes")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout probability")
    
    # Multi-head (for gated classification)
    use_multi_head: bool = Field(default=False, description="Use multi-head architecture")
    num_heads: int = Field(default=3, ge=1, le=5, description="Number of heads")


class MultiViewConfig(BaseModel):
    """Multi-view inference configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    use_multi_view: bool = Field(default=False, description="Use multi-view inference")
    tile_size: int = Field(default=224, ge=128, le=384, description="Tile size")
    overlap: float = Field(default=0.125, ge=0.0, le=0.5, description="Tile overlap ratio")
    use_adaptive_tiling: bool = Field(default=False, description="Use attention-based tiling")
    min_tiles: int = Field(default=4, ge=1, le=16, description="Minimum tiles (adaptive)")
    max_tiles: int = Field(default=16, ge=4, le=64, description="Maximum tiles (adaptive)")
    
    # Aggregation
    aggregation_method: Literal["max", "topk", "attention"] = Field(
        default="attention", description="View aggregation method"
    )
    topk_k: int = Field(default=3, ge=1, le=10, description="Top-K for mean aggregation")
    
    # Attention aggregator
    attention_hidden_dim: int = Field(default=512, ge=64, le=1024, description="Attention hidden dim")
    attention_num_heads: int = Field(default=4, ge=1, le=8, description="Attention heads")


class ModelConfig(BaseModel):
    """Complete model configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    backbone: BackboneConfig = Field(default_factory=BackboneConfig, description="Backbone config")
    head: HeadConfig = Field(default_factory=HeadConfig, description="Head config")
    multi_view: MultiViewConfig = Field(default_factory=MultiViewConfig, description="Multi-view config")


# ============================================================================
# TRAINING CONFIG
# ============================================================================

class OptimizerConfig(BaseModel):
    """Optimizer configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    name: Literal["adamw", "sam2", "sophia", "schedule_free", "ademamix", "muon"] = Field(
        default="adamw", description="Optimizer name"
    )
    lr: float = Field(default=1e-4, ge=1e-6, le=1e-1, description="Learning rate")
    betas: List[float] = Field(default=[0.9, 0.999], description="Adam betas")
    eps: float = Field(default=1e-8, ge=1e-10, le=1e-6, description="Adam epsilon")
    weight_decay: float = Field(default=0.05, ge=0.0, le=1.0, description="Weight decay")
    
    # SAM-specific
    sam_rho: float = Field(default=0.05, ge=0.01, le=0.2, description="SAM rho")
    sam_adaptive: bool = Field(default=True, description="SAM adaptive mode")
    
    # Sophia-specific
    sophia_rho: float = Field(default=0.04, ge=0.01, le=0.1, description="Sophia rho")
    sophia_update_period: int = Field(default=10, ge=1, le=50, description="Sophia Hessian update period")


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    name: Literal["cosine", "linear", "step", "one_cycle", "none"] = Field(
        default="cosine", description="Scheduler name"
    )
    warmup_epochs: int = Field(default=5, ge=0, le=20, description="Warmup epochs")
    min_lr: float = Field(default=1e-6, ge=1e-8, le=1e-4, description="Minimum LR")


class LossConfig(BaseModel):
    """Loss function configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    name: Literal["cross_entropy", "focal", "lcron", "supcon", "koleo"] = Field(
        default="cross_entropy", description="Loss name"
    )
    
    # CrossEntropy
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.3, description="Label smoothing")
    class_weights: Optional[List[float]] = Field(default=None, description="Class weights")
    
    # Focal Loss
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0, description="Focal alpha")
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0, description="Focal gamma")
    
    # LCRON
    lcron_lambda_rank: float = Field(default=0.5, ge=0.0, le=1.0, description="LCRON ranking weight")
    lcron_lambda_cost: float = Field(default=0.3, ge=0.0, le=1.0, description="LCRON cost weight")
    lcron_lambda_acc: float = Field(default=0.2, ge=0.0, le=1.0, description="LCRON accuracy weight")


class TrainingConfig(BaseModel):
    """Training configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    num_epochs: int = Field(default=50, ge=1, le=200, description="Number of epochs")
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=10.0, description="Gradient clipping")
    accumulate_grad_batches: int = Field(default=1, ge=1, le=8, description="Gradient accumulation")
    
    # Early stopping
    early_stopping: bool = Field(default=True, description="Use early stopping")
    early_stopping_patience: int = Field(default=10, ge=1, le=50, description="Early stopping patience")
    early_stopping_monitor: str = Field(default="val_select/accuracy", description="Metric to monitor")
    
    # Checkpointing
    save_top_k: int = Field(default=1, ge=1, le=5, description="Save top K checkpoints")
    save_last: bool = Field(default=True, description="Save last checkpoint")
    checkpoint_every_n_epochs: int = Field(default=5, ge=1, le=50, description="Checkpoint frequency")
    
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer config")
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig, description="Scheduler config")
    loss: LossConfig = Field(default_factory=LossConfig, description="Loss config")


# ============================================================================
# VALIDATION CONFIG
# ============================================================================

class ValidationConfig(BaseModel):
    """Validation configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    target_fnr_exit: float = Field(default=0.02, ge=0.0, le=0.1, description="Target FNR for exit")
    min_coverage: float = Field(default=0.70, ge=0.5, le=1.0, description="Minimum coverage")
    
    # Calibration
    calibration_method: Literal[
        "temperature", "beta", "isotonic", "dirichlet", "none"
    ] = Field(default="temperature", description="Calibration method")
    calibration_by_slice: bool = Field(default=False, description="Calibrate per slice")
    
    # Conformal
    use_conformal: bool = Field(default=False, description="Use conformal prediction")
    conformal_method: Literal["split", "scrc", "crcp", "aps", "raps"] = Field(
        default="split", description="Conformal method"
    )
    conformal_alpha: float = Field(default=0.1, ge=0.01, le=0.5, description="Conformal alpha")


# ============================================================================
# OUTPUT CONFIG
# ============================================================================

class OutputConfig(BaseModel):
    """Output and logging configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    output_dir: Path = Field(default=Path("outputs"), description="Output directory")
    experiment_name: str = Field(default="roadwork_v1", description="Experiment name")
    run_name: Optional[str] = Field(default=None, description="Specific run name")
    
    # Logging
    use_wandb: bool = Field(default=True, description="Use Weights & Biases")
    wandb_project: str = Field(default="roadwork", description="W&B project name")
    use_mlflow: bool = Field(default=False, description="Use MLflow")
    mlflow_experiment: str = Field(default="roadwork", description="MLflow experiment")
    
    # Checkpoints
    checkpoint_dir: Path = Field(default=Path("outputs/checkpoints"), description="Checkpoint directory")
    log_every_n_steps: int = Field(default=10, ge=1, le=100, description="Log frequency")


# ============================================================================
# REPRODUCIBILITY CONFIG
# ============================================================================

class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    seed: int = Field(default=42, ge=0, le=2**32, description="Random seed")
    deterministic: bool = Field(default=False, description="Deterministic algorithms (slower)")
    benchmark: bool = Field(default=True, description="Enable cuDNN benchmark")


# ============================================================================
# COMPLETE CONFIG
# ============================================================================

class Config(BaseModel):
    """Complete configuration (2025 best practices)"""
    
    config_dict = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
    
    data: DataConfig = Field(default_factory=DataConfig, description="Data config")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model config")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training config")
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation config")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output config")
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReprocibilityConfig, description="Reproducibility config"
    )
    
    @field_validator("output.output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ============================================================================
# HYDRA-ZEN FACTORY
# ============================================================================

from hydra_zen import make_config, instantiate

# Create Hydra-compatible config
def make_hydra_config(**kwargs) -> Config:
    """Create a validated Pydantic config from Hydra kwargs"""
    return Config(**kwargs)


# Default configs for different phases
def phase1_config() -> Config:
    """Phase 1: Baseline training"""
    return Config(
        model=ModelConfig(
            backbone=BackboneConfig(freeze_backbone=True, use_peft=False),
            head=HeadConfig(use_multi_head=False),
            multi_view=MultiViewConfig(use_multi_view=False),
        ),
        training=TrainingConfig(
            num_epochs=50,
            optimizer=OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.05),
            loss=LossConfig(name="cross_entropy"),
        ),
    )


def phase3_config() -> Config:
    """Phase 3: Gate training"""
    return Config(
        model=ModelConfig(
            backbone=BackboneConfig(freeze_backbone=True, use_peft=False),
            head=HeadConfig(use_multi_head=True, num_heads=3),
            multi_view=MultiViewConfig(use_multi_view=False),
        ),
        training=TrainingConfig(
            num_epochs=50,
            optimizer=OptimizerConfig(name="adamw", lr=1e-4, weight_decay=0.05),
            loss=LossConfig(name="lcron"),
        ),
        validation=ValidationConfig(
            target_fnr_exit=0.02,
            min_coverage=0.70,
        ),
    )


def phase4_config() -> Config:
    """Phase 4: PEFT fine-tuning"""
    return Config(
        model=ModelConfig(
            backbone=BackboneConfig(
                freeze_backbone=False,
                use_peft=True,
                peft_type="explora",
                peft_r=8,
                peft_alpha=16,
            ),
            head=HeadConfig(use_multi_head=False),
            multi_view=MultiViewConfig(use_multi_view=True, aggregation_method="attention"),
        ),
        training=TrainingConfig(
            num_epochs=50,
            optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.01),
            loss=LossConfig(name="cross_entropy", label_smoothing=0.1),
        ),
    )

