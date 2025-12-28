#!/usr/bin/env python3
"""
Training CLI - Clean Command-Line Interface

Usage:
    # Run all phases
    python scripts/train_cli.py --phases all

    # Run specific phases
    python scripts/train_cli.py --phases phase1 phase2 phase6

    # Resume from saved state
    python scripts/train_cli.py --resume outputs/pipeline_state.json

    # Specify output directory
    python scripts/train_cli.py --output-dir my_experiment --phases phase1

    # Override config
    python scripts/train_cli.py --phases phase1 --config-overrides model.lr=0.001

Benefits:
- Clear command-line interface
- Type-safe argument parsing
- Integration with DAG engine
- Automatic validation
- Progress reporting

Latest 2025-2026 practices:
- Python 3.11+ type hints
- argparse for CLI (standard library)
- Rich for beautiful output
- Structured logging
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contracts.artifact_schema import create_artifact_schema, ArtifactSchema
from pipeline.phase_spec import PhaseType, get_phase_registry
from pipeline.dag_engine import DAGEngine


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="StreetVision Training Pipeline - Stage 1 Ultimate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases
  %(prog)s --phases all

  # Run specific phases
  %(prog)s --phases phase1 phase2 phase6

  # Resume from saved state
  %(prog)s --resume outputs/pipeline_state.json

  # Specify output directory
  %(prog)s --output-dir my_experiment --phases phase1

  # Override config (with Hydra)
  %(prog)s --phases phase1 model.lr=0.001 data.batch_size=32

Available phases:
  phase1  - Baseline Training (DINOv2 + Multi-view)
  phase2  - Threshold Sweep (Softmax policy)
  phase3  - Gate Head Training (Learned gate)
  phase4  - ExPLoRA Pretraining (+8.2% accuracy)
  phase5  - SCRC Calibration
  phase6  - Bundle Export
  all     - Run all phases in order
        """,
    )

    # Phase selection
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["all", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6"],
        help="Phases to run (space-separated)",
    )

    # Resume from state
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from saved pipeline state JSON file",
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for artifacts (default: outputs)",
    )

    # Verbose mode
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without running (useful for debugging)",
    )

    # Save state
    parser.add_argument(
        "--save-state",
        type=Path,
        help="Save pipeline state to JSON file after execution",
    )

    # Config overrides (for Hydra integration)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides (e.g., model.lr=0.001 data.batch_size=32)",
    )

    args = parser.parse_args()

    # Validation
    if not args.phases and not args.resume:
        parser.error("Must specify either --phases or --resume")

    return args


def resolve_phases(phase_names: List[str]) -> List[PhaseType]:
    """
    Resolve phase names to PhaseType enum

    Args:
        phase_names: List of phase names (e.g., ["phase1", "phase2"])

    Returns:
        List of PhaseType enums
    """
    if "all" in phase_names:
        return [
            PhaseType.PHASE1_BASELINE,
            PhaseType.PHASE2_THRESHOLD,
            PhaseType.PHASE3_GATE,
            PhaseType.PHASE4_EXPLORA,
            PhaseType.PHASE5_SCRC,
            PhaseType.PHASE6_BUNDLE,
        ]

    phase_map = {
        "phase1": PhaseType.PHASE1_BASELINE,
        "phase2": PhaseType.PHASE2_THRESHOLD,
        "phase3": PhaseType.PHASE3_GATE,
        "phase4": PhaseType.PHASE4_EXPLORA,
        "phase5": PhaseType.PHASE5_SCRC,
        "phase6": PhaseType.PHASE6_BUNDLE,
    }

    return [phase_map[name] for name in phase_names]


def print_execution_plan(
    engine: DAGEngine,
    phases_to_run: List[PhaseType],
) -> None:
    """
    Print execution plan (for dry-run mode)

    Args:
        engine: DAG engine instance
        phases_to_run: List of phases to run
    """
    print("\n" + "=" * 80)
    print("EXECUTION PLAN (DRY RUN)")
    print("=" * 80)

    # Get execution order
    execution_order = engine.registry.get_execution_order(phases_to_run)

    print(f"\nPhases to run: {len(execution_order)}")
    print("-" * 80)

    total_time_minutes = 0.0

    for i, phase_type in enumerate(execution_order, 1):
        phase = engine.registry.get_phase(phase_type)

        print(f"\n{i}. {phase.name}")
        print(f"   Type: {phase.phase_type.value}")
        print(f"   Dependencies: {[d.value for d in phase.dependencies] or 'None'}")
        print(f"   Inputs: {phase.input_artifacts or 'None'}")
        print(f"   Outputs: {phase.output_artifacts}")
        print(f"   Execution mode: {phase.execution_mode.value}")
        print(f"   GPU required: {phase.resources.requires_gpu}")
        print(f"   Estimated time: {phase.resources.estimated_time_minutes:.0f} min")

        total_time_minutes += phase.resources.estimated_time_minutes

    print("\n" + "-" * 80)
    print(f"Total estimated time: {total_time_minutes:.0f} min ({total_time_minutes/60:.1f} hours)")
    print("=" * 80 + "\n")


def register_phase_executors(engine: DAGEngine) -> None:
    """
    Register executor functions for all phases

    Args:
        engine: DAG engine instance

    Note:
        In a complete implementation, these would import and call
        the actual training/calibration/export modules.
        For now, we register placeholder executors.
    """
    def phase1_executor(artifacts: ArtifactSchema) -> None:
        """Execute Phase 1: Baseline Training"""
        print("\n⚠️  Phase 1 executor not implemented yet")
        print("   TODO: Import and call training/train_baseline.py")
        raise NotImplementedError(
            "Phase 1 executor not implemented.\n"
            "This will be implemented in later TODOs."
        )

    def phase2_executor(artifacts: ArtifactSchema) -> None:
        """Execute Phase 2: Threshold Sweep"""
        print("\n⚠️  Phase 2 executor not implemented yet")
        print("   TODO: Import and call calibration/threshold_sweep.py")
        raise NotImplementedError(
            "Phase 2 executor not implemented.\n"
            "This will be implemented in later TODOs."
        )

    def phase3_executor(artifacts: ArtifactSchema) -> None:
        """Execute Phase 3: Gate Head Training"""
        print("\n⚠️  Phase 3 executor not implemented yet")
        print("   TODO: Import and call training/train_gate.py")
        raise NotImplementedError(
            "Phase 3 executor not implemented.\n"
            "This will be implemented in later TODOs."
        )

    def phase4_executor(artifacts: ArtifactSchema) -> None:
        """Execute Phase 4: ExPLoRA Pretraining"""
        print("\n⚠️  Phase 4 executor not implemented yet")
        print("   TODO: Import and call training/train_explora.py")
        raise NotImplementedError(
            "Phase 4 executor not implemented.\n"
            "This will be implemented in later TODOs."
        )

    def phase5_executor(artifacts: ArtifactSchema) -> None:
        """Execute Phase 5: SCRC Calibration"""
        print("\n⚠️  Phase 5 executor not implemented yet")
        print("   TODO: Import and call calibration/scrc_calibration.py")
        raise NotImplementedError(
            "Phase 5 executor not implemented.\n"
            "This will be implemented in later TODOs."
        )

    def phase6_executor(artifacts: ArtifactSchema) -> None:
        """Execute Phase 6: Bundle Export"""
        print("\n⚠️  Phase 6 executor not implemented yet")
        print("   TODO: Import and call export/bundle_export.py")
        raise NotImplementedError(
            "Phase 6 executor not implemented.\n"
            "This will be implemented in later TODOs."
        )

    # Register executors
    engine.register_executor(PhaseType.PHASE1_BASELINE, phase1_executor)
    engine.register_executor(PhaseType.PHASE2_THRESHOLD, phase2_executor)
    engine.register_executor(PhaseType.PHASE3_GATE, phase3_executor)
    engine.register_executor(PhaseType.PHASE4_EXPLORA, phase4_executor)
    engine.register_executor(PhaseType.PHASE5_SCRC, phase5_executor)
    engine.register_executor(PhaseType.PHASE6_BUNDLE, phase6_executor)


def main() -> int:
    """
    Main CLI entry point

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    args = parse_args()

    # Print header
    print("\n" + "=" * 80)
    print("StreetVision Training Pipeline - Stage 1 Ultimate")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Verbose: {args.verbose}")
    print("=" * 80 + "\n")

    # Create artifact schema
    artifacts = create_artifact_schema(args.output_dir)

    # Create DAG engine
    engine = DAGEngine(artifacts=artifacts, verbose=args.verbose)

    # Register phase executors
    register_phase_executors(engine)

    # Handle resume mode
    if args.resume:
        print(f"Resuming from state: {args.resume}\n")
        try:
            engine.resume(args.resume)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return 1

        # No phases to run in resume mode (state tells us what to run)
        print("⚠️  Resume mode not fully implemented yet")
        print("   TODO: Determine which phases still need to run from state")
        return 1

    # Resolve phases to run
    phases_to_run = resolve_phases(args.phases)

    # Dry run mode
    if args.dry_run:
        print_execution_plan(engine, phases_to_run)
        return 0

    # Run pipeline
    try:
        engine.run(phases_to_run)

        # Save state if requested
        if args.save_state:
            engine.save_state(args.save_state)

        print("\n✅ Pipeline completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")

        # Save state on failure (for debugging)
        if args.save_state:
            engine.save_state(args.save_state)
            print(f"   State saved to: {args.save_state}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
