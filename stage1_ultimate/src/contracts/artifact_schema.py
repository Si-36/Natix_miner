"""
Artifact Registry - Single Source of Truth for ALL File Paths

Benefits:
- Zero hardcoded paths in codebase
- Cannot "forget" to save required artifacts
- Clear phase input/output contracts
- Type-safe path construction
- Explicit directory creation via ensure_dirs()

Latest 2025-2026 practices:
- Python 3.11+ type hints
- Dataclass with slots for memory efficiency
- Pathlib for type-safe paths
- Immutable (frozen=True) where possible
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from enum import Enum


class ArtifactType(Enum):
    """All artifact types in the system"""

    CHECKPOINT = "checkpoint"
    LOGITS = "logits"
    LABELS = "labels"
    POLICY = "policy"
    GATEPARAMS = "gateparams"
    SCRCPARAMS = "scrcparams"
    BUNDLE = "bundle"
    METRICS = "metrics"
    CONFIG = "config"
    SPLITS = "splits"


@dataclass(slots=True, frozen=True)
class ArtifactSchema:
    """
    Artifact Registry - Single Source of Truth for ALL file paths

    CRITICAL RULES:
    - NO hardcoded paths anywhere in codebase
    - ALL file I/O goes through this schema
    - Each phase declares inputs/outputs explicitly
    - Automatic validation of file existence

    Usage:
        artifacts = ArtifactSchema(output_dir=Path("outputs"))
        checkpoint_path = artifacts.phase1_checkpoint
        torch.save(model.state_dict(), checkpoint_path)
    """

    output_dir: Path

    def __post_init__(self):
        """Convert output_dir to Path (no side-effects)"""
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def ensure_dirs(self) -> None:
        """
        Create all required output directories

        Call this ONCE at pipeline initialization, not in properties.
        This makes directory creation explicit and predictable.
        """
        # Create base output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create phase directories
        self.phase1_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_dir.mkdir(parents=True, exist_ok=True)
        self.phase3_dir.mkdir(parents=True, exist_ok=True)
        self.phase4_dir.mkdir(parents=True, exist_ok=True)
        self.phase5_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Create auxiliary directories
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        self.tuning_dir.mkdir(parents=True, exist_ok=True)
        self.mlops_dir.mkdir(parents=True, exist_ok=True)

    # ============= PHASE 1: Baseline Training =============

    @property
    def phase1_dir(self) -> Path:
        """Phase 1 output directory"""
        return self.output_dir / "phase1"

    @property
    def phase1_checkpoint(self) -> Path:
        """Best model checkpoint from Phase 1"""
        return self.phase1_dir / "model_best.pth"

    @property
    def phase1_last_checkpoint(self) -> Path:
        """Last model checkpoint from Phase 1"""
        return self.phase1_dir / "model_last.pth"

    @property
    def phase1_ema_checkpoint(self) -> Path:
        """EMA model checkpoint from Phase 1"""
        return self.phase1_dir / "model_ema.pth"

    @property
    def val_select_logits(self) -> Path:
        """Validation logits on val_select (for model selection)"""
        return self.phase1_dir / "val_select_logits.pt"

    @property
    def val_calib_logits(self) -> Path:
        """Validation logits on val_calib (for policy fitting)"""
        return self.phase1_dir / "val_calib_logits.pt"

    @property
    def val_calib_labels(self) -> Path:
        """Validation labels on val_calib"""
        return self.phase1_dir / "val_calib_labels.pt"

    @property
    def val_calib_features(self) -> Path:
        """Validation features on val_calib (for uncertainty)"""
        return self.phase1_dir / "val_calib_features.pt"

    @property
    def metrics_csv(self) -> Path:
        """Training metrics CSV"""
        return self.phase1_dir / "metrics.csv"

    @property
    def config_json(self) -> Path:
        """Training configuration JSON"""
        return self.phase1_dir / "config.json"

    @property
    def tensorboard_dir(self) -> Path:
        """TensorBoard logs directory"""
        path = self.phase1_dir / "tensorboard"
        return path

    # ============= PHASE 2: Threshold Sweep =============

    @property
    def phase2_dir(self) -> Path:
        """Phase 2 output directory"""
        path = self.output_dir / "phase2"
        return path

    @property
    def thresholds_json(self) -> Path:
        """Threshold sweep results (softmax policy)"""
        return self.phase2_dir / "thresholds.json"

    @property
    def threshold_sweep_csv(self) -> Path:
        """Detailed threshold sweep results"""
        return self.phase2_dir / "threshold_sweep.csv"

    # ============= PHASE 3: Gate Head Training =============

    @property
    def phase3_dir(self) -> Path:
        """Phase 3 output directory"""
        path = self.output_dir / "phase3"
        return path

    @property
    def phase3_checkpoint(self) -> Path:
        """Best model checkpoint from Phase 3 (with gate)"""
        return self.phase3_dir / "model_best.pth"

    @property
    def val_calib_gate_logits(self) -> Path:
        """Gate logits on val_calib"""
        return self.phase3_dir / "val_calib_gate_logits.pt"

    @property
    def gateparams_json(self) -> Path:
        """Gate calibration parameters"""
        return self.phase3_dir / "gateparams.json"

    # ============= PHASE 4: ExPLoRA Pretraining =============

    @property
    def phase4_dir(self) -> Path:
        """Phase 4 output directory (ExPLoRA)"""
        path = self.output_dir / "phase4_explora"
        return path

    @property
    def explora_checkpoint(self) -> Path:
        """ExPLoRA adapted backbone checkpoint"""
        return self.phase4_dir / "explora_backbone.pth"

    @property
    def explora_lora_checkpoint(self) -> Path:
        """ExPLoRA LoRA adapters (before merging)"""
        return self.phase4_dir / "explora_lora.pth"

    @property
    def explora_metrics_json(self) -> Path:
        """ExPLoRA pretraining metrics"""
        return self.phase4_dir / "metrics.json"

    # ============= PHASE 5: SCRC Calibration =============

    @property
    def phase5_dir(self) -> Path:
        """Phase 5 output directory (SCRC)"""
        path = self.output_dir / "phase5_scrc"
        return path

    @property
    def scrcparams_json(self) -> Path:
        """SCRC calibration parameters"""
        return self.phase5_dir / "scrcparams.json"

    @property
    def conformal_params_json(self) -> Path:
        """Conformal prediction parameters (APS, RAPS, CRCP)"""
        return self.phase5_dir / "conformal_params.json"

    # ============= PHASE 6: Bundle Export =============

    @property
    def export_dir(self) -> Path:
        """Export directory"""
        path = self.output_dir / "export"
        return path

    @property
    def bundle_json(self) -> Path:
        """Final deployment bundle manifest"""
        return self.export_dir / "bundle.json"

    @property
    def onnx_model(self) -> Path:
        """ONNX exported model"""
        return self.export_dir / "model.onnx"

    @property
    def tensorrt_engine(self) -> Path:
        """TensorRT engine"""
        return self.export_dir / "model.trt"

    @property
    def triton_model_dir(self) -> Path:
        """Triton model repository"""
        path = self.export_dir / "triton_models"
        return path

    # ============= Calibration Outputs =============

    @property
    def calibration_dir(self) -> Path:
        """Calibration methods output directory"""
        path = self.output_dir / "calibration"
        return path

    @property
    def temperature_params_json(self) -> Path:
        """Temperature scaling parameters"""
        return self.calibration_dir / "temperature.json"

    @property
    def dirichlet_params_json(self) -> Path:
        """Dirichlet calibration parameters"""
        return self.calibration_dir / "dirichlet.json"

    @property
    def beta_params_json(self) -> Path:
        """Beta calibration parameters"""
        return self.calibration_dir / "beta.json"

    @property
    def platt_params_json(self) -> Path:
        """Platt scaling parameters"""
        return self.calibration_dir / "platt.json"

    @property
    def isotonic_params_json(self) -> Path:
        """Isotonic regression parameters"""
        return self.calibration_dir / "isotonic.json"

    @property
    def ensemble_params_json(self) -> Path:
        """Ensemble temperature parameters"""
        return self.calibration_dir / "ensemble.json"

    @property
    def calibration_summary_json(self) -> Path:
        """Calibration methods comparison summary"""
        return self.calibration_dir / "summary.json"

    # ============= Evaluation Outputs =============

    @property
    def evaluation_dir(self) -> Path:
        """Evaluation output directory"""
        path = self.output_dir / "evaluation"
        return path

    @property
    def metrics_summary_json(self) -> Path:
        """Complete metrics summary"""
        return self.evaluation_dir / "metrics_summary.json"

    @property
    def slice_evaluation_csv(self) -> Path:
        """Slice-based evaluation results"""
        return self.evaluation_dir / "slice_evaluation.csv"

    @property
    def bootstrap_results_json(self) -> Path:
        """Bootstrap confidence interval results"""
        return self.evaluation_dir / "bootstrap_ci.json"

    @property
    def reliability_diagram_png(self) -> Path:
        """Reliability diagram visualization"""
        return self.evaluation_dir / "reliability_diagram.png"

    @property
    def calibration_plot_png(self) -> Path:
        """Calibration plot visualization"""
        return self.evaluation_dir / "calibration_plot.png"

    @property
    def roc_curve_png(self) -> Path:
        """ROC curve visualization"""
        return self.evaluation_dir / "roc_curve.png"

    @property
    def pr_curve_png(self) -> Path:
        """Precision-Recall curve visualization"""
        return self.evaluation_dir / "pr_curve.png"

    # ============= Data Splits =============

    @property
    def splits_json(self) -> Path:
        """4-way data splits (train, val_select, val_calib, val_test)"""
        return self.output_dir / "splits.json"

    # ============= Drift Detection =============

    @property
    def drift_dir(self) -> Path:
        """Drift detection output directory"""
        path = self.output_dir / "drift"
        return path

    @property
    def drift_report_json(self) -> Path:
        """Drift detection report (PSI, KS test, MMD)"""
        return self.drift_dir / "drift_report.json"

    # ============= Hyperparameter Tuning =============

    @property
    def tuning_dir(self) -> Path:
        """Hyperparameter tuning output directory"""
        path = self.output_dir / "hyperparameter_tuning"
        return path

    @property
    def optuna_db(self) -> Path:
        """Optuna study database"""
        return self.tuning_dir / "optuna_study.db"

    @property
    def best_hyperparams_json(self) -> Path:
        """Best hyperparameters from tuning"""
        return self.tuning_dir / "best_hyperparams.json"

    # ============= MLOps =============

    @property
    def mlops_dir(self) -> Path:
        """MLOps output directory"""
        path = self.output_dir / "mlops"
        return path

    @property
    def dvc_dir(self) -> Path:
        """DVC cache directory"""
        path = self.mlops_dir / "dvc"
        return path

    @property
    def mlflow_dir(self) -> Path:
        """MLflow tracking directory"""
        path = self.mlops_dir / "mlflow"
        return path

    @property
    def model_registry_json(self) -> Path:
        """Model registry manifest"""
        return self.mlops_dir / "model_registry.json"

    # ============= Phase Contracts =============

    def get_required_inputs(self, phase: int) -> List[Path]:
        """
        Returns required input artifacts for a phase

        Args:
            phase: Phase number (1-6)

        Returns:
            List of required input file paths
        """
        if phase == 1:
            return [self.splits_json]  # Need splits first
        elif phase == 2:
            return [self.val_calib_logits, self.val_calib_labels]
        elif phase == 3:
            return [self.phase1_checkpoint]
        elif phase == 4:
            return []  # ExPLoRA pretraining needs no inputs
        elif phase == 5:
            # FIXED (2025-12-29): SCRC only needs val_calib logits/labels (no checkpoint)
            return [self.val_calib_logits, self.val_calib_labels]
        elif phase == 6:
            # Bundle needs EXACTLY ONE policy file (mutual exclusivity)
            return [self.phase1_checkpoint, self.splits_json]
        return []

    def get_expected_outputs(self, phase: int) -> List[Path]:
        """
        Returns expected output artifacts for a phase

        Args:
            phase: Phase number (1-6)

        Returns:
            List of expected output file paths
        """
        if phase == 1:
            return [
                self.phase1_checkpoint,
                self.val_calib_logits,
                self.val_calib_labels,
                self.metrics_csv,
                self.config_json,
            ]
        elif phase == 2:
            return [self.thresholds_json]
        elif phase == 3:
            return [self.phase3_checkpoint, self.gateparams_json]
        elif phase == 4:
            return [self.explora_checkpoint]
        elif phase == 5:
            return [self.scrcparams_json]
        elif phase == 6:
            return [self.bundle_json]
        return []

    def validate_phase_inputs(self, phase: int) -> bool:
        """
        Validate that all required inputs exist for a phase

        Args:
            phase: Phase number (1-6)

        Returns:
            True if all inputs exist

        Raises:
            FileNotFoundError: If any required input is missing
        """
        required_inputs = self.get_required_inputs(phase)
        missing = [p for p in required_inputs if not p.exists()]

        if missing:
            raise FileNotFoundError(
                f"Phase {phase} missing required inputs: {[str(p) for p in missing]}"
            )

        return True

    def cleanup_phase(self, phase: int) -> None:
        """
        Clean up output files from a phase (for re-running)

        Args:
            phase: Phase number (1-6)
        """
        import shutil

        phase_dirs = {
            1: self.phase1_dir,
            2: self.phase2_dir,
            3: self.phase3_dir,
            4: self.phase4_dir,
            5: self.phase5_dir,
            6: self.export_dir,
        }

        phase_dir = phase_dirs.get(phase)
        if phase_dir and phase_dir.exists():
            shutil.rmtree(phase_dir)
            phase_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Cleaned up Phase {phase} outputs")


# Convenience function for creating artifact schema
def create_artifact_schema(output_dir: str | Path = "outputs") -> ArtifactSchema:
    """
    Create an ArtifactSchema instance

    Args:
        output_dir: Output directory path

    Returns:
        ArtifactSchema instance

    Example:
        >>> artifacts = create_artifact_schema("outputs")
        >>> checkpoint_path = artifacts.phase1_checkpoint
        >>> print(checkpoint_path)
        outputs/phase1/model_best.pth
    """
    return ArtifactSchema(output_dir=Path(output_dir))


if __name__ == "__main__":
    # Test artifact schema
    artifacts = create_artifact_schema("test_outputs")

    print("📋 Artifact Schema Test")
    print(f"Phase 1 checkpoint: {artifacts.phase1_checkpoint}")
    print(f"Val calib logits: {artifacts.val_calib_logits}")
    print(f"Splits JSON: {artifacts.splits_json}")
    print(f"Bundle JSON: {artifacts.bundle_json}")
    print(f"ONNX model: {artifacts.onnx_model}")
    print(f"TensorRT engine: {artifacts.tensorrt_engine}")

    print("\n✅ Artifact schema test passed!")

    # Cleanup test outputs
    import shutil

    if Path("test_outputs").exists():
        shutil.rmtree("test_outputs")
