from __future__ import annotations

from telegram.ext import Application, ApplicationBuilder

from invest_advisor_bot.bot.handlers import (
    BOT_SERVICES_KEY,
    BotServices,
    register_handlers,
    set_bot_commands,
)
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.services.recommendation_service import RecommendationService


def create_application(
    *,
    bot_token: str,
    recommendation_service: RecommendationService,
    market_data_client: MarketDataClient,
    news_client: NewsClient,
    market_news_limit: int = 8,
    market_history_period: str = "6mo",
    market_history_interval: str = "1d",
    market_history_limit: int = 180,
    telegram_report_chat_id: str = "",
) -> Application:
    """Create the base Telegram application and register handlers."""

    normalized_token = bot_token.strip()
    if not normalized_token:
        raise ValueError("bot_token must not be empty")

    application = (
        ApplicationBuilder()
        .token(normalized_token)
        .post_init(_post_init)
        .build()
    )
    application.bot_data[BOT_SERVICES_KEY] = BotServices(
        recommendation_service=recommendation_service,
        market_data_client=market_data_client,
        news_client=news_client,
        market_news_limit=market_news_limit,
        market_history_period=market_history_period,
        market_history_interval=market_history_interval,
        market_history_limit=market_history_limit,
        telegram_report_chat_id=telegram_report_chat_id,
    )
    
    from invest_advisor_bot.bot.jobs import register_jobs
    register_jobs(application)
    
    register_handlers(application)
    return application


async def _post_init(application: Application) -> None:
    await set_bot_commands(application)
