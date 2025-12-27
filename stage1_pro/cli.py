#!/usr/bin/env python3
"""
Unified CLI for all Phases (1-6+).

Modern 2025 CLI with clear interface and auto-detection.
"""

import sys

sys.path.insert(0, "/home/sina/projects/miner_b")

import argparse
from pathlib import Path


def run_tests(phase=None):
    """Run acceptance tests for specified phase or all."""
    test_dir = Path(__file__).parent.parent / "tests"

    if phase:
        test_files = [test_dir / f"test_phase{phase}_acceptance.py"]
    else:
        test_files = [test_dir / f"test_phase{i}_acceptance.py" for i in range(1, 7)]

    for test_file in test_files:
        if test_file.exists():
            import subprocess

            result = subprocess.run(
                ["python3", str(test_file)],
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Stage-1 Pro CLI - Unified Interface for Phases 1-6+

Examples:
  # Run all acceptance tests
  python cli.py test --all
  
  # Run Phase 3 tests
  python cli.py test --phase 3
  
  # Train Phase 3 with PEFT
  python cli.py train --phase 3 --peft_type dora
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run acceptance tests")
    test_parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6])
    test_parser.add_argument("--all", action="store_true")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=1)
    train_parser.add_argument(
        "--peft_type", choices=["none", "dora", "lora"], default="none"
    )
    train_parser.add_argument("--use_fsam", action="store_true")

    args = parser.parse_args()

    if args.command == "test":
        if args.all:
            run_tests()
        else:
            run_tests(phase=args.phase)
    elif args.command == "train":
        print(f"Training Phase {args.phase}...")
        from scripts.train import main as train_main

        sys.argv = ["train", "--phase", str(args.phase)]
        if args.peft_type != "none":
            sys.argv.extend(["--peft_type", args.peft_type])
        if args.use_fsam:
            sys.argv.append("--use_fsam")
        train_main()


if __name__ == "__main__":
    main()
