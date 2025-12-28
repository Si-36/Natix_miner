"""
ExPLoRA Configuration - ELITE 2025 Pattern

Defines LoRA adapter hyperparameters for extended pretraining on DINOv3.

Why ExPLoRA works:
- Extended Pretraining with LoRA adapters (+8.2% accuracy)
- Domain adaptation: General vision → Roadwork detection
- Parameter-efficient: Only train 0.1% of backbone parameters
- Merge adapters after training for zero inference overhead

Latest 2025-2026 practices:
- Python 3.14+ with dataclasses
- PEFT library (>= 0.13.0)
- Optimal LoRA hyperparameters for ViT-H
- Task-specific target module selection
"""

from dataclasses import dataclass, field
from typing import Literal

try:
    from peft import TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    # Fallback: define enum locally if PEFT not installed
    from enum import Enum
    class TaskType(str, Enum):
        FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
        SEQ_CLS = "SEQ_CLS"
        CAUSAL_LM = "CAUSAL_LM"


@dataclass
class ExPLoRAConfig:
    """
    ExPLoRA (Extended Pretraining with LoRA) Configuration

    Hyperparameters optimized for DINOv3 ViT-H fine-tuning on roadwork images.

    Attributes:
        rank: LoRA rank (r). Higher = more capacity, more parameters.
              Default: 16 (optimal for ViT-H, ~2M params)
        alpha: LoRA scaling factor. Controls adapter strength.
               Default: 32 (2× rank is common practice)
        dropout: Dropout rate for LoRA layers. Prevents overfitting.
                 Default: 0.05 (conservative for extended pretraining)
        target_modules: Which attention modules to adapt.
                        Default: ["q_proj", "v_proj", "k_proj"] (query, value, key)
        bias: Whether to adapt bias terms.
              Default: "none" (adapting weights is usually sufficient)
        task_type: PEFT task type.
                   Default: "FEATURE_EXTRACTION" (extended pretraining, not classification)
        init_lora_weights: LoRA weight initialization strategy.
                          Default: "gaussian" (stable, recommended for ViT)
        use_rslora: Whether to use Rank-Stabilized LoRA (2024 improvement).
                    Default: True (better scaling across ranks)
        use_dora: Whether to use DoRA (Weight-Decomposed LoRA, 2024).
                  Default: False (ExPLoRA is simpler, more stable)
    """

    # Core LoRA hyperparameters
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05

    # Target modules (attention layers only for ViT)
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj"]
    )

    # Bias adaptation
    bias: Literal["none", "all", "lora_only"] = "none"

    # Task type (FIXED: use TaskType enum)
    task_type: TaskType | str = TaskType.FEATURE_EXTRACTION

    # Initialization
    init_lora_weights: Literal["gaussian", "pissa", "loftq"] = "gaussian"

    # Advanced features (2024-2025)
    use_rslora: bool = True  # Rank-Stabilized LoRA (better scaling)
    use_dora: bool = False   # DoRA (more complex, not needed for ExPLoRA)

    def to_peft_config(self) -> dict:
        """
        Convert to PEFT LoraConfig dictionary

        FIXED: Properly handles TaskType enum (PEFT expects enum, not string)

        Returns:
            dict: Parameters for peft.LoraConfig(**config)
        """
        # Ensure task_type is TaskType enum for PEFT compatibility
        task_type_value = self.task_type
        if isinstance(task_type_value, str) and PEFT_AVAILABLE:
            # Convert string to enum if PEFT is available
            task_type_value = TaskType(task_type_value)

        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": task_type_value,  # FIXED: Use enum
            "init_lora_weights": self.init_lora_weights,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
        }

    def __post_init__(self):
        """Validate configuration"""
        # Validate rank
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.rank > 64:
            raise ValueError(f"rank > 64 is usually overkill, got {self.rank}")

        # Validate alpha
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

        # Validate dropout
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        # Validate target_modules
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")

        # Common ViT attention modules
        valid_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "qkv"]
        for module in self.target_modules:
            if module not in valid_modules:
                raise ValueError(
                    f"Unknown target module '{module}'. "
                    f"Valid modules: {valid_modules}"
                )


# Preset configurations for common use cases
EXPLORA_PRESETS = {
    "tiny": ExPLoRAConfig(
        rank=4,
        alpha=8,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Only Q/V
    ),
    "small": ExPLoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj"],
    ),
    "base": ExPLoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj"],
    ),
    "large": ExPLoRAConfig(
        rank=32,
        alpha=64,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    ),
}


if __name__ == "__main__":
    # Test configuration
    print("Testing ExPLoRAConfig...")

    # Default config
    config = ExPLoRAConfig()
    print(f"\nDefault config: {config}")
    print(f"PEFT config: {config.to_peft_config()}")

    # Presets
    print("\nPresets:")
    for name, preset in EXPLORA_PRESETS.items():
        print(f"  {name}: rank={preset.rank}, alpha={preset.alpha}")

    # Validation
    try:
        bad_config = ExPLoRAConfig(rank=-1)
    except ValueError as e:
        print(f"\n✅ Validation works: {e}")

    print("\n✅ ExPLoRAConfig test passed")
