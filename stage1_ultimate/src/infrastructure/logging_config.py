"""Centralized logging configuration for all components"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logging(
    level: str = "DEBUG",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """
    Configure loguru logger for the project

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        rotation: When to rotate log file (e.g., "10 MB", "1 day")
        retention: How long to keep old log files
    """
    # Remove default handler
    logger.remove()

    # Add console handler with rich formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured at {level} level")
    if log_file:
        logger.info(f"Logging to file: {log_file}")

# Example usage
if __name__ == "__main__":
    setup_logging(
        level="DEBUG",
        log_file=Path("outputs/logs/stage1_ultimate.log")
    )

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
