from __future__ import annotations

from modules.notifications import TelegramNotifier


def test_notifier_formats_severity_prefix():
    notifier = TelegramNotifier(enabled=False, bot_token="", chat_id="")

    assert notifier._format_message("cycle failed", level="error") == "[ERROR] cycle failed"
    assert notifier._format_message("panic", level="CRITICAL") == "[CRITICAL] panic"
