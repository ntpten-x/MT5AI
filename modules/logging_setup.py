from __future__ import annotations

import sys

from loguru import logger

from config import Settings, ensure_runtime_directories


def configure_logging(settings: Settings):
    ensure_runtime_directories(settings)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.bot.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        settings.log_dir / "{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="21 days",
        enqueue=True,
        level=settings.bot.log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    return logger
