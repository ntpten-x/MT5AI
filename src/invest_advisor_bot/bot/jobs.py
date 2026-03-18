from datetime import time, timezone, timedelta
from loguru import logger
from telegram.ext import Application, ContextTypes

from invest_advisor_bot.bot.handlers import BOT_SERVICES_KEY


async def send_daily_wealth_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduled task to send daily intelligence report."""
    services = context.application.bot_data.get(BOT_SERVICES_KEY)
    if not services or not services.telegram_report_chat_id:
        return

    chat_id = services.telegram_report_chat_id
    try:
        result = await services.recommendation_service.generate_daily_wealth_analysis(
            news_client=services.news_client,
            market_data_client=services.market_data_client,
            news_limit=services.market_news_limit,
        )
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"📊 *รายงานสรุปความมั่งคั่งประจําวัน*\n\n{result.recommendation_text}",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.exception("Failed to send daily wealth report: {}", e)


async def monitor_black_swan_events(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic task to detect extreme risk items."""
    services = context.application.bot_data.get(BOT_SERVICES_KEY)
    if not services or not services.telegram_report_chat_id:
        return

    chat_id = services.telegram_report_chat_id
    try:
        # Fetch data for checks
        news = await services.news_client.fetch_latest_macro_news(limit=5)
        macro = await services.market_data_client.get_macro_context()
        
        triggered, alert_msg = services.recommendation_service.check_black_swan(news, macro)
        if triggered:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🚨 *EMERGENCY BLACK SWAN ALERT*\n\n{alert_msg}",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.exception("Black Swan monitor failed: {}", e)


def register_jobs(application: Application) -> None:
    """Initialize scheduled tasks."""
    jq = application.job_queue
    if jq is None:
        logger.warning("JobQueue is not initialized! Ensure standard installation settings.")
        return

    # 1. Daily Report at 08:00 AM Bangkok Time (UTC+7 -> 01:00 AM UTC)
    bangkok_8am = time(hour=1, minute=0, tzinfo=timezone.utc)
    jq.run_daily(send_daily_wealth_report, time=bangkok_8am)
    logger.info("Registered daily wealth report for 08:00 AM Bangkok")

    # 2. Monitor every 30 minutes for emergency triggers
    jq.run_repeating(monitor_black_swan_events, interval=timedelta(minutes=30), first=10)
    logger.info("Registered repeating Black Swan monitor (every 30 mins)")
