"""
🔒️ **DAG Engine** - Dependency Resolver + Phase Executor
Implements TODO 125: Dependency resolution + ordered phase execution
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from .artifact_schema import ArtifactSchema
from .split_contracts import assert_allowed
from .validators import ArtifactValidator
from .phase_spec import PhaseSpec, PhaseResult


@dataclass
class DAGEngine:
    """
    Dependency-aware DAG engine for phase execution.
    
    Features (2025/2026 best practices):
    - Automatic dependency resolution (topological sort)
    - Fail-fast validation before GPU work
    - Artifact schema validation
    - Split contract enforcement
    - Skip-existing support (for CI)
    
    Minimal explicit dependency graph from your plan:
    Phase 1 (train): []
    Phase 2 (thresholds): [1]
    Phase 3 (gate): [1]
    Phase 6 (export): [1, 2]
    """
    phases: Dict[int, PhaseSpec]
    validator: ArtifactValidator
    
    # Minimal explicit dependency graph
    # Adjust as needed for your actual phases
    _deps = {
        1: [],          # Phase 1: No dependencies
        2: [1],         # Phase 2: Depends on trained model + splits
        3: [1],         # Phase 3: Depends on trained model
        6: [1, 2],     # Phase 6: Depends on model + thresholds
    }
    
    def __post_init__(self):
        """Validate all phase specs after initialization"""
        for pid, spec in self.phases.items():
            assert isinstance(spec, PhaseSpec), f"Phase {pid} must be PhaseSpec"
    
    def _deps_for(self, target: int) -> List[int]:
        """Get explicit dependencies for a target phase"""
        return self._deps.get(target, [])
    
    def resolve(self, target: int) -> List[int]:
        """
        Resolve execution order using topological sort.
        
        Args:
            target: Target phase ID
        
        Returns:
            Ordered list of phase IDs to execute
        """
        order: List[int] = []
        visited: Set[int] = set()
        
        def dfs(n: int):
            if n in visited:
                return
            visited.add(n)
            for dep in self._deps_for(n):
                dfs(dep)
            order.append(n)
        
        dfs(target)
        return order
    
    def run(self, cfg, artifacts: ArtifactSchema, target_phase: int, skip_existing: bool = True) -> None:
        """
        Execute DAG in dependency order.
        
        Args:
            cfg: Hydra config (validated)
            artifacts: ArtifactSchema for path resolution
            target_phase: Target phase ID to run
            skip_existing: Skip phases if outputs already exist
        
        Raises:
            ValueError: If split contracts violated
        """
        execution_order = self.resolve(target_phase)
        
        print("=" * 70)
        print(f"🔒️ DAG Engine: Executing phases in order: {execution_order}")
        print("=" * 70)
        
        for pid in execution_order:
            spec = self.phases[pid]
            print(f"\n📦 Phase {pid}: {spec.name}")
            print("-" * 70)
            
            # Validate inputs exist
            in_paths = spec.inputs(artifacts)
            self.validator.validate_required_files([(f"phase{pid}_input", p) for p in in_paths])
            
            # Check if we should skip existing outputs
            out_paths = spec.outputs(artifacts)
            if skip_existing and out_paths and all(p.exists() and p.stat().st_size > 0 for p in out_paths):
                print(f"⏭️  Skipping {spec.name} (outputs already exist)")
                print(f"   Existing outputs: {[p.name for p in out_paths if p.exists()]}")
                continue
            
            # Execute phase
            result = spec.run(cfg, artifacts)
            
            # Validate allowed splits
            allowed = spec.allowed_splits()
            used = result.splits_used
            assert_allowed(used, allowed, context=f"Phase {pid} {spec.name}")
            
            # Validate outputs exist
            self.validator.validate_required_files([(f"phase{pid}_output", p) for p in out_paths])
            
            print(f"✅ Phase {pid} ({spec.name}) completed")
            print(f"   Outputs produced: {[p.name for p in out_paths]}")
            print(f"   Splits used: {[s.value for s in used]}")
            print("-" * 70)
        
        print("=" * 70)
        print("✅ All phases completed successfully")
        print("=" * 70)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "DAGEngine",
]

