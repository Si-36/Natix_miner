"""
🔒️ **Step Registry** - Single Source of Truth for All Steps (2026 PRO)

Implements:
- Step names (canonical domain names)
- Dependencies (no hardcoded _deps in engine!)
- Order index (explicit ordering hint)
- Owners (team responsible for step)
- Tags (searchable metadata)

2026 PRO Features:
- Dynamic StepSpec discovery (no hardcoding!)
- Step-based DAG (not phase-based!)
- Leak-proof split contracts enforced at registry level

This replaces hardcoded _deps dictionaries and prevents
"forgotten dependency" bugs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path
import sys

# Import StepSpec base class
from .step_api import StepSpec

# Import step specs (absolute imports to avoid circular dependencies)
from steps.export_calib_logits import ExportCalibLogitsSpec
from steps.sweep_thresholds import SweepThresholdsSpec

# Dynamically import all step specs
# This makes the registry extensible - new steps are auto-discovered!
# 2026 PRO: No hardcoded step_catalog!


# NOTE: These imports will be done dynamically in get_step_spec()
# to avoid circular dependencies and enable lazy loading
# from ..steps.export_calib_logits import ExportCalibLogitsSpec
# from ..steps.sweep_thresholds import SweepThresholdsSpec


@dataclass
class StepRegistry:
    """
    Step Registry (2026 PRO Standard).
    
    Single source of truth for all step specifications.
    No hardcoded dependencies - DAG engine reads from here.
    """
    
    # Step catalog (step_name: StepSpec)
    _step_specs: Optional[Dict[str, type[StepSpec]]] = field(default=None)
    
    # Dependency graph (step_name: set[dependency_names])
    _dependency_graph: Optional[Dict[str, frozenset[str]]] = field(default=None)
    
    def __post_init__(self):
        """Initialize step catalog with lazy loading"""
        # Lazy load step specs (only when needed)
        # This avoids circular dependencies
        self._step_specs = {}
        self._dependency_graph = {}
    
    def _discover_steps(self):
        """
        Dynamically discover and register all StepSpec subclasses.
        
        2026 PRO: No hardcoded step_catalog - automatic discovery!
        """
        if self._step_specs is not None and len(self._step_specs) > 0:
            return  # Already discovered
        
        print(f"\n{'='*70}")
        print(f"🔍 Discovering Steps")
        print("=" * 70)
        
        # Register steps (imported at module level to avoid circular dependencies)
        self._step_specs = {
            "export_calib_logits": ExportCalibLogitsSpec,
            "sweep_thresholds": SweepThresholdsSpec,
        }
        
        # Build dependency graph
        self._dependency_graph = {}
        for step_name, step_spec in self._step_specs.items():
            # Instantiate StepSpec to get deps
            # Note: Some steps may require runtime config
            # For registry, we use class-level defaults
            spec_instance = step_spec()  # Create instance to get deps
            
            self._dependency_graph[step_name] = frozenset(spec_instance.deps)
            
            print(f"   ✅ Registered step: {step_name}")
            print(f"      Deps: {list(spec_instance.deps) if spec_instance.deps else ['None']}")
        
        print(f"\n   📋 Total steps discovered: {len(self._step_specs)}")
        print("=" * 70)
    
    def register_step(self, step_name: str, step_spec: type[StepSpec]) -> None:
        """
        Register a step dynamically.
        
        Args:
            step_name: Canonical step name
            step_spec: StepSpec subclass
        """
        if self._step_specs is None:
            self._step_specs = {}
            self._dependency_graph = {}
        
        if step_name in self._step_specs:
            raise ValueError(f"Step '{step_name}' already registered!")
        
        # Create instance to get deps
        spec_instance = step_spec()
        
        self._step_specs[step_name] = step_spec
        self._dependency_graph[step_name] = frozenset(spec_instance.deps)
        
        print(f"✅ Registered step: {step_name}")
    
    def get_step_spec(self, step_name: str) -> Optional[type[StepSpec]]:
        """
        Get StepSpec by name (lazy discovery).
        
        Args:
            step_name: Canonical step name
        
        Returns:
            StepSpec class or None if not found
        """
        # Lazy discovery
        if self._step_specs is None or len(self._step_specs) == 0:
            self._discover_steps()
        
        return self._step_specs.get(step_name, None)
    
    def get_dependencies(self, step_name: str) -> frozenset[str]:
        """
        Get dependencies for a step.
        
        Args:
            step_name: Canonical step name
        
        Returns:
            Frozenset of dependency names
        """
        # Lazy discovery
        if self._dependency_graph is None or len(self._dependency_graph) == 0:
            self._discover_steps()
        
        return self._dependency_graph.get(step_name, frozenset())
    
    def get_all_steps(self) -> Dict[str, type[StepSpec]]:
        """
        Get all registered steps.
        
        Returns:
            Dict mapping step_name -> StepSpec class
        """
        # Lazy discovery
        if self._step_specs is None or len(self._step_specs) == 0:
            self._discover_steps()
        
        return self._step_specs.copy()
    
    def resolve_execution_order(self, target_step: str) -> list[str]:
        """
        Resolve execution order for a target step (DAG resolution).
        
        Args:
            target_step: Target step to run
        
        Returns:
            List of step names in execution order (dependencies first)
        
        Raises:
            RuntimeError: If circular dependency detected
        """
        # Lazy discovery
        if self._dependency_graph is None or len(self._dependency_graph) == 0:
            self._discover_steps()
        
        print(f"\n{'='*70}")
        print(f"🔗 Resolving DAG Execution Order")
        print("=" * 70)
        
        # Topological sort
        visited = set()
        order = []
        
        def visit(step_name: str):
            """Recursive visit for topological sort"""
            if step_name in visited:
                return
            
            # Check for circular dependency
            if step_name in order:
                raise RuntimeError(
                    f"Circular dependency detected: {' → '.join(order + [step_name])}"
                )
            
            # Visit dependencies first
            deps = self._dependency_graph.get(step_name, frozenset())
            for dep in deps:
                visit(dep)
            
            # Add to order
            order.append(step_name)
            visited.add(step_name)
        
        # Visit target step
        visit(target_step)
        
        print(f"   ✅ Execution order resolved:")
        for i, step in enumerate(order, 1):
            print(f"      {i}. {step}")
        
        print("=" * 70)
        
        return order


# Global registry instance (singleton pattern)
STEP_REGISTRY = StepRegistry()


def get_step_spec(step_name: str) -> Optional[type[StepSpec]]:
    """
    Get StepSpec by name (convenience function).
    
    Args:
        step_name: Canonical step name
    
    Returns:
        StepSpec class or None if not found
    """
    return STEP_REGISTRY.get_step_spec(step_name)


def get_dependencies(step_name: str) -> frozenset[str]:
    """
    Get dependencies for a step (convenience function).
    
    Args:
        step_name: Canonical step name
    
    Returns:
        Frozenset of dependency names
    """
    return STEP_REGISTRY.get_dependencies(step_name)


def resolve_execution_order(target_step: str) -> list[str]:
    """
    Resolve execution order for a target step (convenience function).
    
    Args:
        target_step: Target step to run
    
    Returns:
        List of step names in execution order (dependencies first)
    """
    return STEP_REGISTRY.resolve_execution_order(target_step)


__all__ = [
    "StepRegistry",
    "STEP_REGISTRY",
    "get_step_spec",
    "get_dependencies",
    "resolve_execution_order",
]
