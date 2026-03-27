"""Telegram bot application layer."""

from typing import Any


def create_application(*args: Any, **kwargs: Any):  # noqa: ANN401
    from .telegram_app import create_application as _create_application

    return _create_application(*args, **kwargs)

__all__ = ["create_application"]
