from __future__ import annotations

import sys
import asyncio
from contextlib import suppress

from loguru import logger

from invest_advisor_bot.bot.alert_state import AlertStateStore
from invest_advisor_bot.bot.backup_manager import BackupManager
from invest_advisor_bot.bot.health_check import start_health_check_server, stop_health_check_server
from invest_advisor_bot.bot.portfolio_state import PortfolioStateStore
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.bot.telegram_app import create_application
from invest_advisor_bot.bot.user_state import UserStateStore
from invest_advisor_bot.config import Settings, get_settings
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.providers.llm_client import build_default_llm_client
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.providers.research_client import ResearchClient
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.services.recommendation_service import RecommendationService


def configure_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        settings.logs_dir / "invest_advisor_bot.jsonl",
        level=settings.log_level,
        rotation=settings.log_rotation_size,
        retention=settings.log_retention,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
        serialize=True,
    )
    logger.add(
        settings.logs_dir / "invest_advisor_bot.log",
        level=settings.log_level,
        rotation=settings.log_rotation_size,
        retention=settings.log_retention,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
    )


def build_application(settings: Settings):
    alert_state_store = AlertStateStore(
        path=settings.alert_state_path,
        suppression_minutes=settings.alert_suppression_minutes,
        database_url=settings.database_url,
    )
    user_state_store = UserStateStore(path=settings.user_state_path, database_url=settings.database_url)
    portfolio_state_store = PortfolioStateStore(
        path=settings.portfolio_state_path,
        database_url=settings.database_url,
    )
    sector_rotation_state_store = SectorRotationStateStore(
        path=settings.sector_rotation_state_path,
        database_url=settings.database_url,
    )
    report_memory_store = ReportMemoryStore(path=settings.report_memory_path, database_url=settings.database_url)
    runtime_history_store = RuntimeHistoryStore(
        path=settings.runtime_history_path,
        database_url=settings.database_url,
        retention_days=settings.runtime_history_retention_days,
    )
    backup_manager = BackupManager(
        backup_dir=settings.backup_dir,
        database_url=settings.database_url,
        retention_days=settings.backup_retention_days,
    )
    diagnostics.attach_history_store(runtime_history_store)
    market_data_client = MarketDataClient(
        cache_ttl_seconds=settings.market_cache_ttl_seconds,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        provider_order=[item.strip() for item in settings.market_data_provider_order.split(",") if item.strip()],
        http_timeout_seconds=settings.market_data_http_timeout_seconds,
    )
    news_client = NewsClient(
        timeout=settings.news_timeout_seconds,
        cache_ttl_seconds=settings.news_cache_ttl_seconds,
    )
    llm_client = build_default_llm_client(
        llm_api_key=settings.llm_api_key,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        llm_max_output_tokens=settings.llm_max_output_tokens,
        llm_organization=settings.llm_organization,
        llm_project=settings.llm_project,
        llm_provider=settings.llm_provider,
        llm_provider_order=[item.strip() for item in settings.llm_provider_order.split(",") if item.strip()],
        llm_model_fallbacks=[item.strip() for item in settings.llm_model_fallbacks.split(",") if item.strip()],
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_models=[item.strip() for item in settings.openrouter_models.split(",") if item.strip()],
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_app_title=settings.openrouter_app_title,
        gemini_api_key=settings.gemini_api_key,
        gemini_models=[item.strip() for item in settings.gemini_models.split(",") if item.strip()],
        gemini_base_url=settings.gemini_base_url,
        groq_api_key=settings.groq_api_key,
        groq_models=[item.strip() for item in settings.groq_models.split(",") if item.strip()],
        groq_base_url=settings.groq_base_url,
        github_models_api_key=settings.github_models_api_key,
        github_models=[item.strip() for item in settings.github_models.split(",") if item.strip()],
        github_models_base_url=settings.github_models_base_url,
        github_models_api_version=settings.github_models_api_version,
    )
    research_client = ResearchClient(
        tavily_api_key=settings.tavily_api_key,
        exa_api_key=settings.exa_api_key,
        provider_order=[item.strip() for item in settings.research_provider_order.split(",") if item.strip()],
        timeout=settings.news_timeout_seconds,
        cache_ttl_seconds=settings.news_cache_ttl_seconds,
    )
    recommendation_service = RecommendationService(
        llm_client=llm_client,
        system_prompt_path=settings.system_prompt_path,
        default_investor_profile=settings.default_investor_profile,  # type: ignore[arg-type]
    )
    return create_application(
        bot_token=settings.telegram_token,
        recommendation_service=recommendation_service,
        market_data_client=market_data_client,
        news_client=news_client,
        research_client=research_client,
        market_news_limit=settings.market_news_limit,
        market_history_period=settings.market_history_period,
        market_history_interval=settings.market_history_interval,
        market_history_limit=settings.market_history_limit,
        telegram_report_chat_id=settings.telegram_report_chat_id,
        min_request_interval_seconds=settings.bot_min_request_interval_seconds,
        risk_vix_alert_threshold=settings.risk_vix_alert_threshold,
        risk_score_alert_threshold=settings.risk_score_alert_threshold,
        opportunity_score_alert_threshold=settings.opportunity_score_alert_threshold,
        news_impact_alert_threshold=settings.news_impact_alert_threshold,
        alert_state_store=alert_state_store,
        sector_rotation_state_store=sector_rotation_state_store,
        report_memory_store=report_memory_store,
        user_state_store=user_state_store,
        portfolio_state_store=portfolio_state_store,
        runtime_history_store=runtime_history_store,
        backup_manager=backup_manager,
        logs_dir=settings.logs_dir,
        log_retention=settings.log_retention,
        burn_in_target_days=settings.burn_in_target_days,
        jobs_enabled=settings.jobs_enabled,
        daily_digest_hour_utc=settings.daily_digest_hour_utc,
        daily_digest_minute_utc=settings.daily_digest_minute_utc,
        morning_report_hour_utc=settings.morning_report_hour_utc,
        morning_report_minute_utc=settings.morning_report_minute_utc,
        midday_report_hour_utc=settings.midday_report_hour_utc,
        midday_report_minute_utc=settings.midday_report_minute_utc,
        closing_report_hour_utc=settings.closing_report_hour_utc,
        closing_report_minute_utc=settings.closing_report_minute_utc,
        risk_check_interval_minutes=settings.risk_check_interval_minutes,
        maintenance_cleanup_interval_minutes=settings.maintenance_cleanup_interval_minutes,
        earnings_alert_days_ahead=settings.earnings_alert_days_ahead,
        earnings_result_lookback_days=settings.earnings_result_lookback_days,
        sector_rotation_min_streak=settings.sector_rotation_min_streak,
        stock_pick_evaluation_horizon_days=settings.stock_pick_evaluation_horizon_days,
        database_url=settings.database_url,
        stock_pick_evaluation_interval_minutes=settings.stock_pick_evaluation_interval_minutes,
        backup_interval_hours=settings.backup_interval_hours,
    )


def main() -> int:
    try:
        settings = get_settings()
        settings.validate_runtime()
    except Exception as exc:
        logger.error("Configuration error: {}", exc)
        return 2

    configure_logging(settings)
    log_event(
        "runtime_boot",
        database_backend="postgres" if settings.database_url.strip() else "file",
        jobs_enabled=settings.jobs_enabled,
        health_check_enabled=settings.health_check_enabled,
    )
    if not settings.llm_available():
        logger.warning("No LLM provider key is configured; the bot will use fallback summaries")
    if not settings.research_available():
        logger.warning("No research API key is configured; the bot will rely on RSS-only news context")
    if any(
        secret.strip()
        for secret in (
            settings.telegram_token,
            settings.llm_api_key,
            settings.gemini_api_key,
            settings.groq_api_key,
            settings.github_models_api_key,
            settings.tavily_api_key,
            settings.exa_api_key,
        )
    ):
        logger.warning("Rotate any token/API key that was previously shared or used for testing before production deploy")

    application = build_application(settings)
    logger.info("Starting Telegram investment advisor bot")

    if settings.health_check_enabled:
        start_health_check_server(
            host=settings.health_check_host,
            port=settings.health_check_port,
        )

    try:
        application.run_polling(drop_pending_updates=False)
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped by user")
    except Exception as exc:
        logger.exception("Telegram bot crashed: {}", exc)
        return 1
    finally:
        with suppress(Exception):
            asyncio.run(_shutdown_clients(application))
        with suppress(Exception):
            stop_health_check_server()
        with suppress(Exception):
            logger.info("Telegram bot shutdown complete")

    return 0


async def _shutdown_clients(application: object) -> None:
    bot_data = getattr(application, "bot_data", {})
    if not isinstance(bot_data, dict):
        return
    for key in ("bot_services",):
        services = bot_data.get(key)
        if services is None:
            continue
        for attr_name in ("market_data_client", "news_client", "research_client", "recommendation_service"):
            instance = getattr(services, attr_name, None)
            if attr_name == "recommendation_service":
                instance = getattr(instance, "llm_client", None)
            aclose = getattr(instance, "aclose", None)
            if callable(aclose):
                await aclose()


if __name__ == "__main__":
    raise SystemExit(main())
