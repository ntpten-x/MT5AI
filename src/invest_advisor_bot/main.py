from __future__ import annotations

import sys
from contextlib import suppress

from loguru import logger

from invest_advisor_bot.bot.telegram_app import create_application
from invest_advisor_bot.config import Settings, get_settings
from invest_advisor_bot.providers.llm_client import OpenAILLMClient
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.services.recommendation_service import RecommendationService


def configure_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        settings.logs_dir / "invest_advisor_bot.log",
        level=settings.log_level.upper(),
        rotation="10 MB",
        retention=10,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
    )


def build_application(settings: Settings):
    market_data_client = MarketDataClient()
    news_client = NewsClient(timeout=settings.news_timeout_seconds)
    llm_client = OpenAILLMClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
        organization=settings.llm_organization,
        project=settings.llm_project,
    )
    recommendation_service = RecommendationService(
        llm_client=llm_client,
        system_prompt_path=settings.system_prompt_path,
    )
    return create_application(
        bot_token=settings.telegram_token,
        recommendation_service=recommendation_service,
        market_data_client=market_data_client,
        news_client=news_client,
        market_news_limit=settings.market_news_limit,
        market_history_period=settings.market_history_period,
        market_history_interval=settings.market_history_interval,
        market_history_limit=settings.market_history_limit,
    )


def main() -> int:
    try:
        settings = get_settings()
        settings.validate_runtime()
    except Exception as exc:
        logger.error("Configuration error: {}", exc)
        return 2

    configure_logging(settings)
    if not settings.llm_available():
        logger.warning("LLM_API_KEY is not configured; the bot will use fallback summaries only")

    logger.info("Starting Telegram investment advisor bot")
    application = build_application(settings)

    try:
        application.run_polling(drop_pending_updates=False)
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped by user")
    except Exception as exc:
        logger.exception("Telegram bot crashed: {}", exc)
        return 1
    finally:
        with suppress(Exception):
            logger.info("Telegram bot shutdown complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
