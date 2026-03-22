from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from invest_advisor_bot.config import get_settings
from invest_advisor_bot.providers.market_data_client import MarketDataClient


async def main() -> int:
    settings = get_settings()
    print("Loaded settings")
    print(f"Telegram token configured: {bool(settings.telegram_token.strip())}")
    print(f"LLM configured: {settings.llm_available()}")

    client = MarketDataClient(cache_ttl_seconds=60)
    macro = await client.get_macro_context()
    print(f"Macro context: {macro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
