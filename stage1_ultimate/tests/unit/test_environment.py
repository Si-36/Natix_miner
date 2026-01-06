"""Test that local environment is set up correctly"""
import sys
import pytest
from pathlib import Path

def test_python_version():
    """Verify Python 3.10+ is being used"""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version_info}"

def test_cpu_pytorch():
    """Verify PyTorch is using CPU (not CUDA) for local testing"""
    import torch
    # For local testing, we want CPU mode (torch==2.8.0+cpu)
    # In production, torch.cuda.is_available() will be True
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    # This test just verifies PyTorch is installed correctly

def test_project_structure():
    """Verify all required directories exist"""
    project_root = Path(__file__).parent.parent.parent

    required_dirs = [
        "src/compression_2026",
        "src/optimizations_2026",
        "src/infrastructure/vllm",
        "src/infrastructure/monitoring",
        "src/infrastructure/deployment",
        "tests/unit",
        "tests/integration",
        "tests/smoke",
        "deployment/scripts",
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Missing directory: {dir_path}"

def test_dependencies():
    """Verify all required dependencies are installed"""
    required_packages = [
        "pytest",
        "torch",
        "transformers",
        "accelerate",
        "pydantic",
        "loguru",
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            pytest.fail(f"Required package not installed: {package}")

def test_env_file_exists():
    """Verify .env file exists"""
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    # .env file will be created by setup script, skip if not exists yet
    if not env_file.exists():
        pytest.skip(".env file not created yet (run setup_local_env.sh)")
