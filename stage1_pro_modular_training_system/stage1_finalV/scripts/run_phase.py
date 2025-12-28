"""
🔒 **Run Phase CLI** - Single Entrypoint for All Phases
Implements TODO 126: Single CLI entrypoint with Hydra + DAG engine
"""

from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize, version_base

from src.core.artifact_schema import ArtifactSchema
from src.core.validators import ArtifactValidator
from src.core.dag_engine import DAGEngine
from src.core.split_contracts import SplitPolicy, Split


def new_run_id() -> str:
    """
    Generate new run ID.
    
    Returns:
        Run ID in YYYYMMDD-HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@hydra.main(
    config_path="../configs", 
    config_name="config.yaml", 
    version_base=None
)
def main(cfg):
    """
    Main CLI entrypoint for phase execution.
    
    Usage:
        python scripts/run_phase.py --config-path=configs/config.yaml target_phase=1
        
    Or with CLI overrides:
        python scripts/run_phase.py target_phase=1 training.mode=frozen model.model_id=...
    
    Args:
        cfg: Validated Hydra config
    """
    print("=" * 70)
    print("🚀 Stage-1 Pro System - Phase Runner")
    print("=" * 70)
    print(f"Target Phase: {cfg.run.target_phase}")
    print(f"Skip Existing: {cfg.run.skip_existing}")
    print(f"Output Dir: {cfg.paths.output_dir}")
    print("=" * 70)
    
    # Setup artifact schema
    output_dir = Path(cfg.paths.output_dir)
    run_id = cfg.run.run_id or new_run_id()
    artifacts = ArtifactSchema(output_dir=output_dir, run_id=run_id)
    
    # Create output directory
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create DAG engine with phases
    # (Phase specs will be loaded in Phase 131-140)
    print(f"\n📂 Run Directory: {artifacts.run_dir}")
    print(f"📝 Run ID: {run_id}")
    print("=" * 70)
    
    # Save resolved config (for reproducibility)
    OmegaConf.save(cfg, artifacts.config_resolved_yaml)
    print(f"✅ Config saved: {artifacts.config_resolved_yaml}")
    
    # Initialize DAG engine with validator
    validator = ArtifactValidator()
    
    # TODO: Load actual phase specs here (Phase 131-140)
    # For now, we'll skip phase execution since specs aren't implemented yet
    print("\n⚠️  Phase specs not yet implemented (TODO 131-140)")
    print("   Skipping actual phase execution")
    print("=" * 70)
    print("✅ DAG engine infrastructure ready for Phase 1-6")
    print("=" * 70)


if __name__ == "__main__":
    main()

