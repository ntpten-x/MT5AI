from __future__ import annotations

from pathlib import Path

from telegram.ext import Application, ApplicationBuilder

from invest_advisor_bot.bot.alert_state import AlertStateStore
from invest_advisor_bot.bot.backup_manager import BackupManager
from invest_advisor_bot.bot.portfolio_state import PortfolioStateStore
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.bot.user_state import UserStateStore
from invest_advisor_bot.bot.handlers import (
    BOT_SERVICES_KEY,
    BotServices,
    register_handlers,
    set_bot_commands,
)
from invest_advisor_bot.bot.jobs import register_jobs
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.providers.research_client import ResearchClient
from invest_advisor_bot.services.recommendation_service import RecommendationService


def create_application(
    *,
    bot_token: str,
    recommendation_service: RecommendationService,
    market_data_client: MarketDataClient,
    news_client: NewsClient,
    research_client: ResearchClient | None = None,
    market_news_limit: int = 5,
    market_history_period: str = "6mo",
    market_history_interval: str = "1d",
    market_history_limit: int = 180,
    telegram_report_chat_id: str = "",
    min_request_interval_seconds: float = 2.0,
    risk_vix_alert_threshold: float = 30.0,
    risk_score_alert_threshold: float = 6.5,
    opportunity_score_alert_threshold: float = 2.8,
    news_impact_alert_threshold: float = 2.0,
    earnings_alert_days_ahead: int = 7,
    earnings_result_lookback_days: int = 14,
    sector_rotation_min_streak: int = 3,
    stock_pick_evaluation_horizon_days: int = 5,
    database_url: str = "",
    backup_manager: BackupManager | None = None,
    alert_state_store: AlertStateStore | None = None,
    sector_rotation_state_store: SectorRotationStateStore | None = None,
    report_memory_store: ReportMemoryStore | None = None,
    user_state_store: UserStateStore | None = None,
    portfolio_state_store: PortfolioStateStore | None = None,
    runtime_history_store: RuntimeHistoryStore | None = None,
    logs_dir: Path | None = None,
    log_retention: str = "14 days",
    burn_in_target_days: int = 14,
    jobs_enabled: bool = False,
    daily_digest_hour_utc: int = 1,
    daily_digest_minute_utc: int = 0,
    morning_report_hour_utc: int = 12,
    morning_report_minute_utc: int = 30,
    midday_report_hour_utc: int = 17,
    midday_report_minute_utc: int = 0,
    closing_report_hour_utc: int = 21,
    closing_report_minute_utc: int = 15,
    risk_check_interval_minutes: int = 30,
    maintenance_cleanup_interval_minutes: int = 360,
    stock_pick_evaluation_interval_minutes: int = 360,
    backup_interval_hours: int = 24,
) -> Application:
    """Create the Telegram application and register handlers and optional jobs."""

    normalized_token = bot_token.strip()
    if not normalized_token:
        raise ValueError("bot_token must not be empty")

    application = ApplicationBuilder().token(normalized_token).post_init(_post_init).build()
    application.bot_data[BOT_SERVICES_KEY] = BotServices(
        recommendation_service=recommendation_service,
        market_data_client=market_data_client,
        news_client=news_client,
        research_client=research_client,
        market_news_limit=market_news_limit,
        market_history_period=market_history_period,
        market_history_interval=market_history_interval,
        market_history_limit=market_history_limit,
        telegram_report_chat_id=telegram_report_chat_id,
        min_request_interval_seconds=min_request_interval_seconds,
        risk_vix_alert_threshold=risk_vix_alert_threshold,
        risk_score_alert_threshold=risk_score_alert_threshold,
        opportunity_score_alert_threshold=opportunity_score_alert_threshold,
        news_impact_alert_threshold=news_impact_alert_threshold,
        earnings_alert_days_ahead=earnings_alert_days_ahead,
        earnings_result_lookback_days=earnings_result_lookback_days,
        sector_rotation_min_streak=sector_rotation_min_streak,
        stock_pick_evaluation_horizon_days=stock_pick_evaluation_horizon_days,
        database_url=database_url,
        alert_state_store=alert_state_store,
        sector_rotation_state_store=sector_rotation_state_store,
        report_memory_store=report_memory_store,
        user_state_store=user_state_store,
        portfolio_state_store=portfolio_state_store,
        runtime_history_store=runtime_history_store,
        backup_manager=backup_manager,
        logs_dir=logs_dir,
        log_retention=log_retention,
        burn_in_target_days=burn_in_target_days,
    )

    if jobs_enabled:
        register_jobs(
            application,
            daily_digest_hour_utc=daily_digest_hour_utc,
            daily_digest_minute_utc=daily_digest_minute_utc,
            morning_report_hour_utc=morning_report_hour_utc,
            morning_report_minute_utc=morning_report_minute_utc,
            midday_report_hour_utc=midday_report_hour_utc,
            midday_report_minute_utc=midday_report_minute_utc,
            closing_report_hour_utc=closing_report_hour_utc,
            closing_report_minute_utc=closing_report_minute_utc,
            risk_check_interval_minutes=risk_check_interval_minutes,
            earnings_alert_days_ahead=earnings_alert_days_ahead,
            maintenance_cleanup_interval_minutes=maintenance_cleanup_interval_minutes,
            stock_pick_evaluation_interval_minutes=stock_pick_evaluation_interval_minutes,
            backup_interval_hours=backup_interval_hours,
        )

    register_handlers(application)
    return application


async def _post_init(application: Application) -> None:
    await set_bot_commands(application)
