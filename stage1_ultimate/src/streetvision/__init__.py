"""
StreetVision Stage-1 Training Pipeline
Production-grade implementation (2025-12-30)

Package structure:
- io/: Atomic file operations and manifest tracking
- eval/: Centralized metric computation (MCC, accuracy, FNR)
- cli/: Command-line interface
- pipeline/: DAG orchestration and step execution
"""

__version__ = "1.0.0"
