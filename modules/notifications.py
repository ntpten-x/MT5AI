from __future__ import annotations

import httpx
from loguru import logger


class TelegramNotifier:
    def __init__(self, enabled: bool, bot_token: str, chat_id: str, timeout: float = 10.0):
        self.enabled = enabled
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram is enabled but bot token or chat id is missing")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            response = httpx.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: {}", exc)
            return False
