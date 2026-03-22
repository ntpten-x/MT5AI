from __future__ import annotations

from typing import Any

from loguru import logger


def log_event(event: str, /, level: str = "info", **fields: Any) -> None:
    """Emit a structured log event with normalized event name."""

    normalized_event = event.strip() or "event"
    bound = logger.bind(event=normalized_event, **fields)
    log_method = getattr(bound, level.lower(), bound.info)
    log_method(normalized_event)
