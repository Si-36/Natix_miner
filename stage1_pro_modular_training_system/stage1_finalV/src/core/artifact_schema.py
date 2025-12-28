"""
🔒️ **Artifact Schema** - Single Source of Truth for All Paths
Implements TODO 121: Artifact registry for every phase output
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSchema:
    """
    Single source of truth for all artifact paths across all phases.
    
    Prevents "forgot to save X" bugs and provides deterministic paths.
    Every phase reads inputs only from ArtifactSchema and writes outputs only to ArtifactSchema.
    
    Args:
        output_dir: Base output directory
        run_id: Unique run identifier (e.g., YYYYMMDD-HHMMSS)
    """
    output_dir: Path
    run_id: str  # e.g. YYYYMMDD-HHMMSS
    
    # -----------------------------------------------------------------
    # Run-level artifacts (every phase produces these)
    # -----------------------------------------------------------------
    
    @property
    def run_dir(self) -> Path:
        """Main run directory"""
        return self.output_dir / "runs" / self.run_id
    
    @property
    def config_resolved_yaml(self) -> Path:
        """Resolved Hydra config (full config dump)"""
        return self.run_dir / "config_resolved.yaml"
    
    @property
    def config_schema_json(self) -> Path:
        """Pydantic JSON schema for this run"""
        return self.run_dir / "config_schema.json"
    
    @property
    def manifest_json(self) -> Path:
        """Run manifest (git commit, timestamp, seeds, artifact hashes)"""
        return self.run_dir / "manifest.json"
    
    # -----------------------------------------------------------------
    # Phase 1: Training (train → validate → calibrate → export)
    # -----------------------------------------------------------------
    
    @property
    def phase1_dir(self) -> Path:
        """Phase 1 directory"""
        return self.run_dir / "phase1"
    
    @property
    def phase1_checkpoint(self) -> Path:
        """Best model checkpoint (Phase 1)"""
        return self.phase1_dir / "model_best.pth"
    
    @property
    def phase1_val_select_logits_pt(self) -> Path:
        """Validation logits for model selection (Phase 1)"""
        return self.phase1_dir / "val_select_logits.pt"
    
    @property
    def phase1_val_select_labels_pt(self) -> Path:
        """Validation labels for model selection (Phase 1)"""
        return self.phase1_dir / "val_select_labels.pt"
    
    @property
    def phase1_metrics_csv(self) -> Path:
        """Metrics CSV (Phase 1)"""
        return self.phase1_dir / "metrics.csv"
    
    # -----------------------------------------------------------------
    # Phase 2: Threshold Sweep / Policy Fitting
    # -----------------------------------------------------------------
    
    @property
    def phase2_dir(self) -> Path:
        """Phase 2 directory"""
        return self.run_dir / "phase2"
    
    @property
    def phase2_thresholds_json(self) -> Path:
        """Optimal thresholds (Phase 2)"""
        return self.phase2_dir / "thresholds.json"
    
    @property
    def phase2_policy_metrics_csv(self) -> Path:
        """Policy metrics (Phase 2)"""
        return self.phase2_dir / "policy_metrics.csv"
    
    # -----------------------------------------------------------------
    # Phase 3: Gate Training
    # -----------------------------------------------------------------
    
    @property
    def phase3_dir(self) -> Path:
        """Phase 3 directory"""
        return self.run_dir / "phase3"
    
    @property
    def phase3_checkpoint(self) -> Path:
        """Best gate checkpoint (Phase 3)"""
        return self.phase3_dir / "gate_best.pth"
    
    @property
    def phase3_gate_params_json(self) -> Path:
        """Gate parameters (Phase 3)"""
        return self.phase3_dir / "gate_params.json"
    
    # -----------------------------------------------------------------
    # Phase 6: Export Bundle
    # -----------------------------------------------------------------
    
    @property
    def phase6_dir(self) -> Path:
        """Phase 6 directory"""
        return self.run_dir / "phase6"
    
    @property
    def export_dir(self) -> Path:
        """Export directory (Phase 6)"""
        return self.run_dir / "export"
    
    @property
    def bundle_json(self) -> Path:
        """Production bundle (Phase 6)"""
        return self.export_dir / "bundle.json"
    
    @property
    def bundle_readme_md(self) -> Path:
        """Bundle README (Phase 6)"""
        return self.export_dir / "README.md"


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "ArtifactSchema",
]

