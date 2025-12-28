"""
🔥 **Artifact Registry (2025 Best Practices)**
Single source of truth for all file paths - Pydantic validated
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import field_validator, BaseModel, ConfigDict, Field
import json


class ArtifactPaths(BaseModel):
    """
    Complete artifact path registry with Pydantic validation
    Prevents "forgot to save X" bugs
    """
    
    # Output directory
    output_dir: Path = Field(default=Path("outputs"), description="Root output directory")
    
    # Phase 1 artifacts
    phase1_checkpoint: Path = Field(default=Path("outputs/phase1/best.ckpt"), description="Phase 1 best checkpoint")
    phase1_last_checkpoint: Path = Field(default=Path("outputs/phase1/last.ckpt"), description="Phase 1 last checkpoint")
    val_select_logits: Path = Field(default=Path("outputs/phase1/val_select_logits.pt"), description="Val_select logits for calibration")
    val_select_labels: Path = Field(default=Path("outputs/phase1/val_select_labels.pt"), description="Val_select labels")
    
    # Phase 2 artifacts
    val_calib_logits: Path = Field(default=Path("outputs/phase2/val_calib_logits.pt"), description="Val_calib logits for threshold sweep")
    thresholds_json: Path = Field(default=Path("outputs/phase2/thresholds.json"), description="Thresholds from phase 2")
    
    # Phase 3 artifacts
    phase3_checkpoint: Path = Field(default=Path("outputs/phase3/gate.ckpt"), description="Phase 3 gate checkpoint")
    val_calib_gate_logits: Path = Field(default=Path("outputs/phase3/val_calib_gate_logits.pt"), description="Val_calib gate logits")
    gateparams_json: Path = Field(default=Path("outputs/phase3/gateparams.json"), description="Gate parameters from phase 3")
    
    # Phase 4 artifacts (ExPLoRA)
    phase4_checkpoint: Path = Field(default=Path("outputs/phase4/explora.ckpt"), description="Phase 4 ExPLoRA checkpoint")
    
    # Phase 6 artifacts (Bundle export)
    bundle_json: Path = Field(default=Path("outputs/phase6/bundle.json"), description="Production bundle")
    
    # Data splits
    splits_json: Path = Field(default=Path("data/splits.json"), description="Data splits file")
    
    # Logs
    training_log: Path = Field(default=Path("logs/training.csv"), description="Training metrics log")
    
    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def ensure_dirs(self):
        """Ensure all parent directories exist"""
        paths = [
            self.phase1_checkpoint.parent,
            self.val_select_logits.parent,
            self.val_calib_logits.parent,
            self.gateparams_json.parent,
            self.splits_json.parent,
            self.training_log.parent,
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
    
    def phase1_artifacts_exist(self) -> bool:
        """Check if Phase 1 artifacts exist"""
        return self.phase1_checkpoint.exists() and self.val_select_logits.exists()
    
    def phase2_artifacts_exist(self) -> bool:
        """Check if Phase 2 artifacts exist"""
        return self.thresholds_json.exists() and self.val_calib_logits.exists()
    
    def phase3_artifacts_exist(self) -> bool:
        """Check if Phase 3 artifacts exist"""
        return self.phase3_checkpoint.exists() and self.gateparams_json.exists()
    
    def all_required_for_phase1(self) -> bool:
        """Check all required inputs for Phase 1"""
        return self.splits_json.exists()
    
    def all_required_for_phase2(self) -> bool:
        """Check all required inputs for Phase 2"""
        return self.phase1_artifacts_exist()
    
    def all_required_for_phase3(self) -> bool:
        """Check all required inputs for Phase 3"""
        return self.phase1_artifacts_exist()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access"""
        return {
            "phase1_checkpoint": str(self.phase1_checkpoint),
            "val_select_logits": str(self.val_select_logits),
            "val_select_labels": str(self.val_select_labels),
            "thresholds_json": str(self.thresholds_json),
            "gateparams_json": str(self.gateparams_json),
            "bundle_json": str(self.bundle_json),
        }


# Global artifact registry (singleton pattern)
_artifact_registry: Optional[ArtifactPaths] = None


def get_artifact_registry(output_dir: Optional[str] = None) -> ArtifactPaths:
    """
    Get global artifact registry (singleton pattern)
    
    Args:
        output_dir: Optional output directory override
    
    Returns:
        ArtifactPaths instance
    """
    global _artifact_registry
    
    if _artifact_registry is None or output_dir is not None:
        if output_dir:
            _artifact_registry = ArtifactPaths(output_dir=Path(output_dir))
        else:
            _artifact_registry = ArtifactPaths()
        
        _artifact_registry.ensure_dirs()
    
    return _artifact_registry


# Export for easy imports
__all__ = ["ArtifactPaths", "get_artifact_registry"]
