import sys
import asyncio
from pathlib import Path

# Add src to PYTHONPATH
# Script being in /tmp/verify_deploy.py meansparents[1] might not work if /tmp is top-level.
# Let's use absolute path for workspace.
workspace = Path("e:/AI Trade/MT5AI")
sys.path.append(str(workspace / "src"))

print(f"Python Path: {sys.path[-1]}")

try:
    from invest_advisor_bot.config import get_settings
    from invest_advisor_bot.providers.market_data_client import MarketDataClient
    from invest_advisor_bot.services.recommendation_service import RecommendationService
    print("✅ Core modules imported successfully.")
except ImportError as e:
    print(f"❌ Core modules import failed: {e}")
    sys.exit(1)


async def test_macro_fetch():
    print("\n🔍 Testing Macro Context Fetch...")
    client = MarketDataClient()
    try:
        macro = await client.get_macro_context()
        print(f"✅ Macro context fetched: {macro}")
    except Exception as e:
        print(f"❌ Failed to fetch macro context: {e}")

def verify():
    print("\n⚙️ Loading Settings...")
    try:
        settings = get_settings()
        if not settings.telegram_token.strip():
            print("⚠️ Warning: TELEGRAM_TOKEN is missing or empty.")
        else:
            print("✅ Settings loaded successfully.")
    except Exception as e:
        print(f"❌ Failed loading settings: {e}")

    asyncio.run(test_macro_fetch())


if __name__ == "__main__":
    verify()
