"""
Centralized logging using Loguru.
Provides structured, rotating, colored logs across all modules.
"""

import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove Loguru's default handler
logger.remove()

# Console — colored, human-readable
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<white>{message}</white>"
    ),
    level="INFO",
    colorize=True,
)

# Daily rotating file — full debug
logger.add(
    LOG_DIR / "color_detector_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    level="DEBUG",
    rotation="00:00",
    retention="7 days",
    compression="zip",
)

# Separate error-only log
logger.add(
    LOG_DIR / "errors.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}\n{exception}",
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
)

__all__ = ["logger"]
