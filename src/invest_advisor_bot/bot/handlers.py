from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Mapping

from loguru import logger
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from invest_advisor_bot.bot.alert_state import AlertStateStore
from invest_advisor_bot.bot.ai_simulated_portfolio_state import AISimulatedPortfolioStateStore
from invest_advisor_bot.bot.backup_manager import BackupManager
from invest_advisor_bot.bot.portfolio_state import PortfolioHolding, PortfolioStateStore
from invest_advisor_bot.bot.postgres_state import PostgresStateBackend
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.bot.user_state import UserPreferences, UserStateStore
from invest_advisor_bot.orchestration.prefect_flows import WorkflowOrchestrator
from invest_advisor_bot.providers.broker_client import ExecutionSandboxClient
from invest_advisor_bot.providers.live_market_stream import LiveMarketStreamClient
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.microstructure_client import MicrostructureClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.providers.research_client import ResearchClient
from invest_advisor_bot.providers.transcript_client import EarningsTranscriptClient
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.analysis.portfolio_profile import normalize_profile_name
from invest_advisor_bot.services.recommendation_service import AssetScope, FallbackVerbosity, RecommendationService
from invest_advisor_bot.services.ai_simulated_portfolio import AISimulatedPortfolioService

BOT_SERVICES_KEY: Final[str] = "bot_services"
MAX_TELEGRAM_MESSAGE_LENGTH: Final[int] = 4000

CALLBACK_QUICK_SUMMARY: Final[str] = "quick:summary"
CALLBACK_GOLD: Final[str] = "quick:gold"
CALLBACK_US_STOCKS: Final[str] = "quick:us_stocks"
CALLBACK_ETF: Final[str] = "quick:etf"
CALLBACK_GLOBAL_TREND: Final[str] = "quick:global_trend"
CALLBACK_STOCK_IDEAS: Final[str] = "quick:stock_ideas"
CALLBACK_MENU_HELP: Final[str] = "menu:help"
CALLBACK_MENU_PROFILE: Final[str] = "menu:profile"
CALLBACK_MENU_PORTFOLIO: Final[str] = "menu:portfolio"
CALLBACK_MENU_WATCHLIST: Final[str] = "menu:watchlist"
CALLBACK_MENU_PREFS: Final[str] = "menu:prefs"
CALLBACK_MENU_REPORT_NOW: Final[str] = "menu:report_now"
CALLBACK_MENU_MARKET_UPDATE: Final[str] = "menu:market_update"
CALLBACK_MENU_SCORECARD: Final[str] = "menu:scorecard"

QuickAction = tuple[str, AssetScope, FallbackVerbosity | None]

QUICK_ACTIONS: Final[dict[str, QuickAction]] = {
    CALLBACK_QUICK_SUMMARY: ("สรุปด่วนภาพรวมตลาดโลกตอนนี้", "all", "short"),
    CALLBACK_GOLD: ("ช่วยวิเคราะห์ทองคำตอนนี้พร้อมคำแนะนำแบบกระชับ", "gold-only", None),
    CALLBACK_US_STOCKS: ("ช่วยวิเคราะห์หุ้นสหรัฐและดัชนีหลักตอนนี้พร้อมคำแนะนำแบบกระชับ", "us-stocks", None),
    CALLBACK_ETF: ("ช่วยวิเคราะห์ ETF หลักตอนนี้พร้อมคำแนะนำแบบกระชับ", "etf-only", None),
    CALLBACK_GLOBAL_TREND: ("สรุปเทรนด์ตลาดโลกตอนนี้พร้อมสินทรัพย์ที่ควรจับตา", "all", None),
    CALLBACK_STOCK_IDEAS: ("ตอนนี้ควรซื้อหุ้นอะไร 5 ตัวในกลุ่ม S&P 500 และ Nasdaq 100 พร้อมเหตุผลสั้น ๆ", "us-stocks", None),
}

_LAST_REQUEST_BY_CHAT: dict[str, float] = {}


@dataclass(slots=True, frozen=True)
class BotServices:
    recommendation_service: RecommendationService
    market_data_client: MarketDataClient
    news_client: NewsClient
    research_client: ResearchClient | None
    market_news_limit: int
    market_history_period: str
    market_history_interval: str
    market_history_limit: int
    broker_client: ExecutionSandboxClient | None = None
    transcript_client: EarningsTranscriptClient | None = None
    microstructure_client: MicrostructureClient | None = None
    live_market_stream_client: LiveMarketStreamClient | None = None
    workflow_orchestrator: WorkflowOrchestrator | None = None
    telegram_report_chat_id: str = ""
    min_request_interval_seconds: float = 2.0
    macro_event_refresh_interval_minutes: int = 5
    macro_event_pre_window_minutes: int = 20
    macro_event_post_window_minutes: int = 90
    macro_event_lookahead_hours: int = 12
    risk_vix_alert_threshold: float = 30.0
    risk_score_alert_threshold: float = 6.5
    opportunity_score_alert_threshold: float = 2.8
    news_impact_alert_threshold: float = 2.0
    earnings_alert_days_ahead: int = 7
    earnings_result_lookback_days: int = 14
    sector_rotation_min_streak: int = 3
    stock_pick_evaluation_horizon_days: int = 5
    database_url: str = ""
    alert_state_store: AlertStateStore | None = None
    sector_rotation_state_store: SectorRotationStateStore | None = None
    report_memory_store: ReportMemoryStore | None = None
    user_state_store: UserStateStore | None = None
    portfolio_state_store: PortfolioStateStore | None = None
    ai_simulated_portfolio_state_store: AISimulatedPortfolioStateStore | None = None
    ai_simulated_portfolio_service: AISimulatedPortfolioService | None = None
    runtime_history_store: RuntimeHistoryStore | None = None
    backup_manager: BackupManager | None = None
    logs_dir: Path | None = None
    log_retention: str = "14 days"
    burn_in_target_days: int = 14
    health_alert_webhook_url: str = ""
    health_alert_webhook_secret: str = ""
    health_alert_interval_minutes: int = 5
    health_alert_cooldown_minutes: int = 30
    health_alert_timeout_seconds: float = 8.0
    health_alert_retry_count: int = 3
    health_alert_retry_backoff_seconds: float = 1.5
    live_stream_symbols: tuple[str, ...] = ()
    live_stream_poll_interval_seconds: int = 60
    live_stream_max_events: int = 25
    live_stream_spread_alert_bps: float = 25.0
    telegram_transport: str = "polling"
    telegram_webhook_url: str = ""
    telegram_webhook_path: str = ""
    telegram_webhook_port: int = 0


def register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("broker", broker_command))
    application.add_handler(CommandHandler("paperbuy", paperbuy_command))
    application.add_handler(CommandHandler("papersell", papersell_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("ai_portfolio", ai_portfolio_command))
    application.add_handler(CommandHandler("ai_trades", ai_trades_command))
    application.add_handler(CommandHandler("ai_performance", ai_performance_command))
    application.add_handler(CommandHandler("ai_rebalance", ai_rebalance_command))
    application.add_handler(CommandHandler("ai_reset", ai_reset_command))
    application.add_handler(CommandHandler("holdadd", holdadd_command))
    application.add_handler(CommandHandler("holdremove", holdremove_command))
    application.add_handler(CommandHandler("watchlist", watchlist_command))
    application.add_handler(CommandHandler("watchadd", watchadd_command))
    application.add_handler(CommandHandler("watchremove", watchremove_command))
    application.add_handler(CommandHandler("prefs", prefs_command))
    application.add_handler(CommandHandler("report_now", report_now_command))
    application.add_handler(CommandHandler("scorecard", scorecard_command))
    application.add_handler(CommandHandler("dashboard", dashboard_command))
    application.add_handler(CommandHandler("backup_now", backup_now_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("analyst", analyst_command))
    application.add_handler(CommandHandler("reviewqueue", review_queue_command))
    application.add_handler(CommandHandler("reviewdone", review_done_command))
    application.add_handler(CommandHandler("market_update", market_update_command))
    application.add_handler(CallbackQueryHandler(handle_quick_action, pattern=r"^quick:"))
    application.add_handler(CallbackQueryHandler(handle_menu_action, pattern=r"^menu:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(handle_error)


async def set_bot_commands(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("start", "เริ่มใช้งานและเปิดเมนูหลัก"),
            BotCommand("help", "ดูวิธีใช้งานแบบย่อ"),
            BotCommand("broker", "ดู paper account และ positions"),
            BotCommand("paperbuy", "ส่งคำสั่ง paper buy"),
            BotCommand("papersell", "ส่งคำสั่ง paper sell"),
            BotCommand("profile", "ตั้งหรือดูโปรไฟล์นักลงทุน"),
            BotCommand("portfolio", "ดูพอร์ตที่บันทึกไว้"),
            BotCommand("ai_portfolio", "ดูพอร์ตจำลองของ AI"),
            BotCommand("ai_trades", "ดูประวัติซื้อขายของ AI"),
            BotCommand("ai_performance", "ดูผลตอบแทนพอร์ต AI"),
            BotCommand("ai_rebalance", "สั่งให้ AI ทบทวนพอร์ตตัวอย่าง"),
            BotCommand("ai_reset", "รีเซ็ตพอร์ตจำลอง AI"),
            BotCommand("holdadd", "เพิ่มหรืออัปเดต holding"),
            BotCommand("holdremove", "ลบ holding ออกจากพอร์ต"),
            BotCommand("watchlist", "ดูรายการหุ้นที่ติดตาม"),
            BotCommand("watchadd", "เพิ่มหุ้นเข้า watchlist"),
            BotCommand("watchremove", "ลบหุ้นออกจาก watchlist"),
            BotCommand("prefs", "ดูหรือปรับ alert preferences"),
            BotCommand("report_now", "ขอรายงาน morning/midday/closing ตอนนี้"),
            BotCommand("scorecard", "ดูผลหุ้นที่ระบบเคยแนะนำ"),
            BotCommand("dashboard", "ดู burn-in และ evaluation dashboard"),
            BotCommand("backup_now", "สร้าง backup ของ state/runtime history"),
            BotCommand("status", "ดูสถานะ runtime ของระบบ"),
            BotCommand("analyst", "ถามข้อมูล runtime/analytics แบบภาษาคน"),
            BotCommand("reviewqueue", "ดู human review queue"),
            BotCommand("reviewdone", "ปิด human review"),
            BotCommand("market_update", "สรุปตลาดโลกและแนวทางจัดพอร์ตล่าสุด"),
        ]
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    current_profile = services.recommendation_service.get_investor_profile(_conversation_key(update))
    text = (
        "สวัสดีครับ ผมคือ Invest Advisor Bot บน Telegram\n"
        "ผมช่วยวิเคราะห์หุ้นสหรัฐ ETF ทองคำ ข่าวเศรษฐกิจ และภาวะมหภาค เพื่อสรุปแนวทางจัดพอร์ตแบบนักลงทุนที่เน้นรักษาและเติบโตทรัพย์สิน\n\n"
        f"โปรไฟล์ปัจจุบันของคุณ: {current_profile.title_th}\n"
        "ถ้าต้องการเปลี่ยนโปรไฟล์ ใช้ /profile conservative, /profile balanced, /profile growth หรือ /profile aggressive\n\n"
        "ระบบสามารถสแกนตลาดอัตโนมัติและส่ง Morning / Midday / Closing reports,"
        " หุ้นเด่น, sector rotation, earnings และข่าวสำคัญเข้า report chat ได้เอง\n\n"
        "ตัวอย่างคำถาม\n"
        "- ตอนนี้หุ้นเมกาน่าเข้าหรือยัง\n"
        "- ตอนนี้ควรซื้อหุ้นอะไร 5 ตัว\n"
        "- ช่วยวิเคราะห์ ETF แบบละเอียดพร้อมเหตุผล\n"
        "- ขอแนวทางจัดพอร์ตแบบรักษาเงินต้น\n\n"
        "คำสั่งเสริม\n"
        "- /broker\n"
        "- /paperbuy AAPL 1\n"
        "- /papersell NVDA 1 850\n"
        "- /holdadd VOO 10 470\n"
        "- /portfolio\n"
        "- /ai_portfolio\n"
        "- /ai_trades\n"
        "- /ai_performance\n"
        "- /ai_rebalance balanced\n"
        "- /watchadd AAPL\n"
        "- /watchlist\n"
        "- /prefs sectors=Technology,Healthcare threshold=2.1\n"
        "- /report_now closing\n"
        "- /scorecard\n"
        "- /dashboard\n"
        "- /analyst recommendation fallback trends\n"
        "- /reviewqueue\n"
        "- /reviewdone review-abc accepted 0.82 note=good rationale"
    )
    await _reply_text(message, text, reply_markup=_build_main_menu())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    await _reply_text(
        message,
        (
            "คำสั่งหลัก\n"
            "/start เปิดเมนูหลัก\n/help ดูวิธีใช้งาน\n/broker ดู paper account และ positions\n/paperbuy ส่งคำสั่ง paper buy\n/papersell ส่งคำสั่ง paper sell\n/profile ดูหรือตั้งโปรไฟล์นักลงทุน\n/portfolio ดูพอร์ตที่บันทึกไว้\n/ai_portfolio ดูพอร์ตตัวอย่างของ AI\n/ai_trades ดูประวัติซื้อขาย AI\n/ai_performance ดูผลตอบแทนพอร์ต AI\n/ai_rebalance ให้ AI ทบทวนพอร์ตอีกครั้ง\n/ai_reset รีเซ็ตพอร์ต AI\n/holdadd เพิ่มหรืออัปเดต holding\n/holdremove ลบ holding\n/watchlist ดู watchlist\n/watchadd เพิ่มหุ้นเข้ารายการติดตาม\n/watchremove ลบหุ้นออกจาก watchlist\n/prefs ดูหรือปรับ preferences\n/report_now ขอรายงานล่าสุดทันที\n/scorecard ดูผลหุ้นที่ระบบเคยแนะนำ\n/dashboard ดู burn-in และ evaluation dashboard\n/backup_now สร้าง backup ทันที\n/status ดูสถานะ runtime ของระบบ\n/analyst ถามข้อมูล analytics/runtime\n/reviewqueue ดูรายการรอ human review\n/reviewdone ปิด human review พร้อม decision/score\n/market_update สรุปตลาดและแนวทางจัดพอร์ตล่าสุด\n\n"
            "ระบบอัตโนมัติ\n- Morning / Midday / Closing reports\n- Stock pick alerts\n- Sector rotation alerts\n- Earnings calendar alerts\n\n"
            "โปรไฟล์ที่รองรับ\n- conservative: รักษาเงินต้น\n- balanced: สมดุล\n- growth: เติบโต\n- aggressive: alias ของ growth สำหรับสายรุก\n\n"
            "ตัวอย่างคำถาม\n- วิเคราะห์ทองคำแบบรักษาเงินต้น\n- ขอพอร์ต ETF สำหรับคนรับความเสี่ยงปานกลาง\n- ช่วยสรุปหุ้นสหรัฐแบบสั้นมาก"
            "\n- ตอนนี้ควรซื้อหุ้นอะไร 5 ตัว\n- วิเคราะห์ AAPL และ MSFT ให้หน่อย\n- /ai_rebalance balanced\n- /ai_reset 1000 conservative\n- /analyst recommendation fallback trends\n- /reviewdone review-abc accepted 0.82 note=clear thesis"
        ),
        reply_markup=_build_main_menu(),
    )


async def broker_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    broker_client = services.broker_client
    if broker_client is None or not broker_client.enabled():
        await _reply_text(message, "broker sandbox ยังไม่พร้อมใช้งาน ตรวจ BROKER_SANDBOX_ENABLED และ Alpaca paper credentials")
        return

    account = await broker_client.get_account()
    positions = await broker_client.list_positions()
    status = broker_client.status()
    lines = [
        "Broker Sandbox",
        f"provider: {status.get('provider')}",
        f"configured: {status.get('configured')} | enabled={status.get('enabled')}",
    ]
    if account is not None:
        lines.append(
            "account: "
            f"status={account.status or '-'} | equity={account.equity if account.equity is not None else '-'} "
            f"| buying_power={account.buying_power if account.buying_power is not None else '-'} "
            f"| cash={account.cash if account.cash is not None else '-'}"
        )
    else:
        lines.append("account: unavailable")
    if positions:
        lines.append("positions:")
        for item in positions[:8]:
            lines.append(
                f"- {item.symbol}: qty={item.qty} | mv={item.market_value if item.market_value is not None else '-'} | upl={item.unrealized_pl if item.unrealized_pl is not None else '-'}"
            )
    else:
        lines.append("positions: none")
    if status.get("warning"):
        lines.append(f"warning: {status.get('warning')}")
    await _reply_text(message, "\n".join(lines))


async def paperbuy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_paper_trade(update, context, side="buy")


async def papersell_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_paper_trade(update, context, side="sell")


async def _handle_paper_trade(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    side: str,
) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    broker_client = services.broker_client
    if broker_client is None or not broker_client.enabled():
        await _reply_text(message, "broker sandbox ยังไม่พร้อมใช้งาน")
        return
    parsed = _parse_paper_trade_args(context.args)
    if parsed is None:
        await _reply_text(
            message,
            f"รูปแบบคำสั่ง: /paper{side} SYMBOL QTY [LIMIT_PRICE]\nตัวอย่าง: /paper{side} AAPL 1 หรือ /paper{side} NVDA 1 850",
        )
        return
    symbol, qty, limit_price = parsed
    order_type = "limit" if limit_price is not None else "market"
    result = await broker_client.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        order_type=order_type,
        limit_price=limit_price,
    )
    if result is None:
        await _reply_text(message, f"paper {side} ไม่สำเร็จ: {broker_client.status().get('warning') or 'unknown error'}")
        return
    rendered = (
        f"paper {side} submitted\n"
        f"symbol: {result.symbol}\n"
        f"qty: {result.qty}\n"
        f"type: {result.type}\n"
        f"status: {result.status or '-'}\n"
        f"order_id: {result.order_id or '-'}\n"
        f"limit_price: {result.limit_price if result.limit_price is not None else '-'}"
    )
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind=f"paper_{side}",
        question=f"/paper{side} {symbol} {qty}",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    conversation_key = _conversation_key(update)
    if conversation_key is None:
        return

    if not context.args:
        profile = services.recommendation_service.get_investor_profile(conversation_key)
        await _reply_text(
            message,
            (
                f"โปรไฟล์ปัจจุบัน: {profile.title_th}\n"
                f"เป้าหมาย: {profile.objective}\n"
                f"ความเสี่ยง: {profile.risk_summary}\n\n"
                "ตั้งค่าใหม่ได้ด้วย\n- /profile conservative\n- /profile balanced\n- /profile growth\n- /profile aggressive"
            ),
        )
        return

    profile_arg = normalize_profile_name(context.args[0].strip(), default=services.recommendation_service.default_investor_profile)
    if profile_arg not in {"conservative", "balanced", "growth"}:
        await _reply_text(message, "โปรไฟล์ที่รองรับคือ conservative, balanced, growth และ aggressive")
        return

    profile = services.recommendation_service.set_investor_profile(conversation_key=conversation_key, profile_name=profile_arg)
    await _reply_text(message, f"ตั้งค่าโปรไฟล์เป็น {profile.title_th} แล้ว\nเป้าหมาย: {profile.objective}\nแนวทาง: {profile.rebalance_hint}")


async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    holdings = _get_portfolio_holdings(update, services)
    if not holdings:
        await _reply_text(
            message,
            "ยังไม่มีพอร์ตที่บันทึกไว้\nใช้ /holdadd VOO 10 470 หรือ /holdadd CASH 20000 เพื่อเริ่มบันทึกพอร์ต",
        )
        return
    lines = ["Portfolio Holdings"]
    for holding in holdings:
        avg_cost = f" | avg cost {holding.avg_cost:.2f}" if holding.avg_cost is not None else ""
        note = f" | {holding.note}" if holding.note else ""
        lines.append(f"- {holding.normalized_ticker}: qty {holding.quantity:.4f}{avg_cost}{note}")
    await _reply_text(message, "\n".join(lines))


async def ai_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        await _reply_text(message, "ระบบ AI simulated portfolio ยังไม่พร้อมใช้งาน")
        return
    refresh = any(str(arg).strip().casefold() in {"refresh", "rebalance"} for arg in context.args)
    rendered = await ai_service.render_portfolio_text(conversation_key=services.telegram_report_chat_id, refresh=refresh)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="ai_portfolio",
        question="/ai_portfolio",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def ai_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        await _reply_text(message, "ระบบ AI simulated portfolio ยังไม่พร้อมใช้งาน")
        return
    rendered = await ai_service.render_trades_text(conversation_key=services.telegram_report_chat_id, limit=10)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="ai_trades",
        question="/ai_trades",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def ai_performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        await _reply_text(message, "ระบบ AI simulated portfolio ยังไม่พร้อมใช้งาน")
        return
    rendered = await ai_service.render_performance_text(conversation_key=services.telegram_report_chat_id)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="ai_performance",
        question="/ai_performance",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def ai_rebalance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        await _reply_text(message, "ระบบ AI simulated portfolio ยังไม่พร้อมใช้งาน")
        return
    profile_name = None
    if context.args:
        profile_name = normalize_profile_name(context.args[0], default="growth")
        ai_service.set_profile(conversation_key=services.telegram_report_chat_id, profile_name=profile_name)
    await message.reply_text("กำลังให้ AI ทบทวนพอร์ตจำลอง ...")
    result = await ai_service.maybe_rebalance(
        conversation_key=services.telegram_report_chat_id,
        reason="manual_command",
        force=True,
    )
    rendered = result.rendered_summary
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="ai_rebalance",
        question=f"/ai_rebalance {profile_name}" if profile_name else "/ai_rebalance",
        response_text=rendered,
        fallback_used=False,
        model=None,
        detail={"action_count": result.action_count, "skipped_reason": result.skipped_reason, "profile_name": profile_name},
    )
    await _reply_text(message, rendered)


async def ai_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        await _reply_text(message, "ระบบ AI simulated portfolio ยังไม่พร้อมใช้งาน")
        return
    starting_cash = ai_service.starting_cash_usd
    profile_name = None
    args = list(context.args or [])
    if args:
        try:
            starting_cash = max(100.0, float(args[0]))
            args = args[1:]
        except ValueError:
            profile_name = normalize_profile_name(args[0], default="growth")
            args = args[1:]
    if args and profile_name is None:
        profile_name = normalize_profile_name(args[0], default="growth")
    await ai_service.reset_portfolio(
        conversation_key=services.telegram_report_chat_id,
        starting_cash=starting_cash,
        profile_name=profile_name,
    )
    rendered = await ai_service.render_portfolio_text(conversation_key=services.telegram_report_chat_id, refresh=True)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="ai_reset",
        question=f"/ai_reset {starting_cash:.2f} {profile_name}".strip(),
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def holdadd_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.portfolio_state_store is None:
        await _reply_text(message, "ระบบ portfolio tracking ยังไม่พร้อมใช้งาน")
        return
    if len(context.args) < 2:
        await _reply_text(message, "ใช้รูปแบบ /holdadd VOO 10 470 หรือ /holdadd CASH 20000")
        return
    ticker = context.args[0].strip().upper()
    try:
        quantity = float(context.args[1])
    except ValueError:
        await _reply_text(message, "จำนวนต้องเป็นตัวเลข เช่น /holdadd VOO 10 470")
        return
    avg_cost: float | None = None
    note_parts: list[str] = []
    if len(context.args) >= 3:
        try:
            avg_cost = float(context.args[2])
            note_parts = context.args[3:]
        except ValueError:
            note_parts = context.args[2:]
    note = " ".join(note_parts).strip() or None
    holdings = services.portfolio_state_store.upsert_holding(
        conversation_key,
        ticker=ticker,
        quantity=quantity,
        avg_cost=avg_cost,
        note=note,
    )
    await _reply_text(
        message,
        f"บันทึก holding {ticker} แล้ว\nจำนวน holdings ตอนนี้: {len(holdings)}\nดูพอร์ตด้วย /portfolio",
    )


async def holdremove_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.portfolio_state_store is None:
        await _reply_text(message, "ระบบ portfolio tracking ยังไม่พร้อมใช้งาน")
        return
    if not context.args:
        await _reply_text(message, "ใช้รูปแบบ /holdremove VOO")
        return
    ticker = context.args[0].strip().upper()
    holdings = services.portfolio_state_store.remove_holding(conversation_key, ticker=ticker)
    latest = ", ".join(item.normalized_ticker for item in holdings) if holdings else "ยังไม่มี"
    await _reply_text(message, f"ลบ {ticker} ออกจากพอร์ตแล้ว\nรายการล่าสุด: {latest}")


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    prefs = _get_user_preferences(update, services)
    watchlist_text = ", ".join(prefs.watchlist) if prefs.watchlist else "ยังไม่มี"
    sectors_text = ", ".join(prefs.preferred_sectors) if prefs.preferred_sectors else "ทุก sector"
    await _reply_text(
        message,
        f"Watchlist: {watchlist_text}\nPreferred sectors: {sectors_text}\nStock alert threshold: {prefs.stock_alert_threshold:.1f}\nDaily pick enabled: {prefs.daily_pick_enabled}\nDashboard execution filter: {prefs.dashboard_execution_filter or 'all'}\nApproval mode: {prefs.approval_mode}\nMax position size: {prefs.max_position_size_pct if prefs.max_position_size_pct is not None else 'default'}",
    )


async def watchadd_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.user_state_store is None:
        await _reply_text(message, "ระบบ watchlist ยังไม่พร้อมใช้งาน")
        return
    if not context.args:
        await _reply_text(message, "ใช้รูปแบบ /watchadd AAPL")
        return
    prefs = services.user_state_store.add_watchlist(conversation_key, context.args[0])
    await _reply_text(message, f"เพิ่ม {context.args[0].upper()} เข้า watchlist แล้ว\nรายการล่าสุด: {', '.join(prefs.watchlist)}")


async def watchremove_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.user_state_store is None:
        await _reply_text(message, "ระบบ watchlist ยังไม่พร้อมใช้งาน")
        return
    if not context.args:
        await _reply_text(message, "ใช้รูปแบบ /watchremove AAPL")
        return
    prefs = services.user_state_store.remove_watchlist(conversation_key, context.args[0])
    latest = ", ".join(prefs.watchlist) if prefs.watchlist else "ยังไม่มี"
    await _reply_text(message, f"ลบ {context.args[0].upper()} ออกจาก watchlist แล้ว\nรายการล่าสุด: {latest}")


async def prefs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.user_state_store is None:
        await _reply_text(message, "ระบบ preferences ยังไม่พร้อมใช้งาน")
        return
    if not context.args:
        prefs = services.user_state_store.get(conversation_key)
        await _reply_text(
            message,
            f"Preferred sectors: {', '.join(prefs.preferred_sectors) if prefs.preferred_sectors else 'ทุก sector'}\nStock alert threshold: {prefs.stock_alert_threshold:.1f}\nDaily pick enabled: {prefs.daily_pick_enabled}\nDashboard execution filter: {prefs.dashboard_execution_filter or 'all'}\nApproval mode: {prefs.approval_mode}\nMax position size: {prefs.max_position_size_pct if prefs.max_position_size_pct is not None else 'default'}\n\nตัวอย่าง: /prefs sectors=Technology,Healthcare threshold=2.0 daily=true approval=review maxsize=2.5",
        )
        return

    sectors: list[str] | None = None
    threshold: float | None = None
    daily_pick_enabled: bool | None = None
    approval_mode: str | None = None
    max_position_size_pct: float | None = None
    for argument in context.args:
        key, _, value = argument.partition("=")
        normalized_key = key.strip().casefold()
        if normalized_key == "sectors":
            sectors = [item.strip() for item in value.split(",") if item.strip()]
        elif normalized_key == "threshold":
            try:
                threshold = float(value)
            except ValueError:
                threshold = None
        elif normalized_key == "daily":
            daily_pick_enabled = value.strip().casefold() in {"true", "1", "yes", "on"}
        elif normalized_key == "approval":
            approval_mode = value.strip()
        elif normalized_key == "maxsize":
            try:
                max_position_size_pct = float(value)
            except ValueError:
                max_position_size_pct = None
    prefs = services.user_state_store.update_preferences(
        conversation_key,
        preferred_sectors=sectors,
        stock_alert_threshold=threshold,
        daily_pick_enabled=daily_pick_enabled,
        approval_mode=approval_mode,
        max_position_size_pct=max_position_size_pct,
    )
    await _reply_text(
        message,
        f"อัปเดต preferences แล้ว\nPreferred sectors: {', '.join(prefs.preferred_sectors) if prefs.preferred_sectors else 'ทุก sector'}\nStock alert threshold: {prefs.stock_alert_threshold:.1f}\nDaily pick enabled: {prefs.daily_pick_enabled}\nDashboard execution filter: {prefs.dashboard_execution_filter or 'all'}\nApproval mode: {prefs.approval_mode}\nMax position size: {prefs.max_position_size_pct if prefs.max_position_size_pct is not None else 'default'}",
    )


async def report_now_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if await _is_rate_limited(update, services, message):
        return

    requested = context.args[0].strip().casefold() if context.args else ""
    report_kind = requested if requested in {"morning", "midday", "closing"} else _infer_report_kind_now()
    await _send_typing(update, context)
    await message.reply_text(f"กำลังสร้างรายงาน {report_kind} ...")
    result = await services.recommendation_service.generate_periodic_report(
        report_kind=report_kind,  # type: ignore[arg-type]
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        research_client=services.research_client,
        sector_rotation_state_store=services.sector_rotation_state_store,
        report_memory_store=services.report_memory_store,
        sector_rotation_min_streak=services.sector_rotation_min_streak,
        earnings_result_lookback_days=services.earnings_result_lookback_days,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
        portfolio_holdings=_get_portfolio_holdings(update, services),
    )
    rendered_text = await _append_ai_portfolio_summary(services, result.recommendation_text)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="report_now",
        question=f"/report_now {report_kind}",
        response_text=rendered_text,
        fallback_used=result.fallback_used,
        model=result.model,
        detail=_build_result_history_detail(source="report_now", result=result),
    )
    if services.runtime_history_store is not None and update.effective_chat is not None:
        services.runtime_history_store.record_sent_report(
            report_kind=report_kind,
            chat_id=str(update.effective_chat.id),
            fallback_used=result.fallback_used,
            model=result.model,
            summary=rendered_text[:240],
            detail=_build_result_history_detail(source="manual_command", result=result),
        )
    await _reply_text(message, rendered_text)


async def scorecard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if services.runtime_history_store is None or update.effective_chat is None:
        await _reply_text(message, "ระบบ scorecard ยังไม่พร้อมใช้งาน")
        return

    rows = services.runtime_history_store.recent_stock_pick_scorecard(chat_id=str(update.effective_chat.id), limit=8)
    if not rows:
        await _reply_text(message, "ยังไม่มี scorecard ของหุ้นที่ระบบเคยแนะนำในห้องนี้")
        return

    lines = ["Stock Pick Scorecard"]
    for row in rows:
        ticker = str(row.get("ticker") or "-")
        status = str(row.get("status") or "-")
        confidence_label = str(row.get("confidence_label") or "-")
        confidence_score = row.get("confidence_score")
        composite_score = row.get("composite_score")
        return_pct = row.get("return_pct")
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        return_after_cost_pct = detail.get("return_after_cost_pct")
        alpha_after_cost_pct = detail.get("alpha_after_cost_pct")
        if return_pct is not None:
            after_cost_text = ""
            try:
                if return_after_cost_pct is not None:
                    after_cost_text = f" | after cost {float(return_after_cost_pct):+.1%}"
            except (TypeError, ValueError):
                after_cost_text = ""
            alpha_text = ""
            try:
                if alpha_after_cost_pct is not None:
                    benchmark_label = str(detail.get("benchmark_ticker") or detail.get("benchmark") or "benchmark")
                    alpha_text = f" | alpha vs {benchmark_label} {float(alpha_after_cost_pct):+.1%}"
            except (TypeError, ValueError):
                alpha_text = ""
            lines.append(
                f"- {ticker}: {status} | raw {float(return_pct):+.1%}{after_cost_text}{alpha_text} | confidence {confidence_label} ({confidence_score}) | score {composite_score}"
            )
        else:
            due_at = str(detail.get("due_at") or row.get("created_at") or "-")
            lines.append(
                f"- {ticker}: {status} | confidence {confidence_label} ({confidence_score}) | score {composite_score} | due {due_at}"
            )
        thesis_label = _extract_primary_scorecard_thesis(detail)
        if thesis_label:
            lines.append(f"  thesis: {thesis_label}")
        postmortem_action = str(detail.get("postmortem_action") or "").strip()
        decay_label = str(detail.get("signal_decay_label") or "").strip()
        if postmortem_action or decay_label:
            rendered_postmortem = postmortem_action or "-"
            rendered_decay = decay_label or "-"
            lines.append(f"  postmortem: {rendered_postmortem} | decay: {rendered_decay}")
    rendered = "\n".join(lines)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="scorecard",
        question="/scorecard",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    if services.runtime_history_store is None:
        await _reply_text(message, "ยังไม่มี runtime history store สำหรับสร้าง dashboard")
        return

    execution_alert_kind, execution_sort = _resolve_dashboard_execution_view(update, services, context.args)

    history_snapshot = services.runtime_history_store.build_evaluation_dashboard(
        chat_id=str(update.effective_chat.id) if update.effective_chat is not None else None,
        lookback_days=max(7, services.burn_in_target_days),
        burn_in_target_days=services.burn_in_target_days,
        execution_alert_kind=execution_alert_kind,
    )
    _get_mlflow_status(services)
    live_snapshot = diagnostics.snapshot()
    burn_in = history_snapshot.get("burn_in") or {}
    scorecard = history_snapshot.get("scorecard") or {}
    reports = history_snapshot.get("reports") or {}
    interactions = history_snapshot.get("interactions") or {}
    alerts = history_snapshot.get("alerts") or {}
    providers = history_snapshot.get("providers") or []
    jobs = history_snapshot.get("jobs") or []
    source_ranking = history_snapshot.get("source_ranking") or []
    thesis_ranking = history_snapshot.get("thesis_ranking") or []
    decision_quality = history_snapshot.get("decision_quality") or {}
    thesis_lifecycle = history_snapshot.get("thesis_lifecycle") or {}
    walk_forward_eval = history_snapshot.get("walk_forward_eval") or {}
    execution_panel = history_snapshot.get("execution_panel") or {}
    mlflow_status = live_snapshot.get("mlflow") or _get_mlflow_status(services)
    live_latest = live_snapshot.get("latest_provider_success") or {}
    live_response = live_snapshot.get("response_stats") or {}
    broker_status = services.broker_client.status() if services.broker_client is not None else None
    transcript_status = services.transcript_client.status() if services.transcript_client is not None else None
    microstructure_status = services.microstructure_client.status() if services.microstructure_client is not None else None
    live_stream_status = services.live_market_stream_client.status() if services.live_market_stream_client is not None else None
    workflow_status = services.workflow_orchestrator.status() if services.workflow_orchestrator is not None else None
    braintrust_observer = getattr(services.recommendation_service, "braintrust_observer", None)
    braintrust_status = braintrust_observer.status() if braintrust_observer is not None else None
    recommendation_status_getter = getattr(services.recommendation_service, "status", None)
    recommendation_status = recommendation_status_getter() if callable(recommendation_status_getter) else {}
    thesis_vector_status = recommendation_status.get("thesis_vector_store") if isinstance(recommendation_status, Mapping) else None
    feature_store_status = recommendation_status.get("feature_store") if isinstance(recommendation_status, Mapping) else None
    backtesting_status = recommendation_status.get("backtesting") if isinstance(recommendation_status, Mapping) else None
    analytics_warehouse_status = recommendation_status.get("analytics_warehouse") if isinstance(recommendation_status, Mapping) else None
    event_bus_status = recommendation_status.get("event_bus") if isinstance(recommendation_status, Mapping) else None
    event_bus_consumer_status = recommendation_status.get("event_bus_consumer") if isinstance(recommendation_status, Mapping) else None
    hot_path_cache_status = recommendation_status.get("hot_path_cache") if isinstance(recommendation_status, Mapping) else None
    semantic_analyst_status = recommendation_status.get("semantic_analyst") if isinstance(recommendation_status, Mapping) else None
    ownership_status = recommendation_status.get("ownership") if isinstance(recommendation_status, Mapping) else None
    order_flow_status = recommendation_status.get("order_flow") if isinstance(recommendation_status, Mapping) else None
    policy_feed_status = recommendation_status.get("policy_feed") if isinstance(recommendation_status, Mapping) else None
    dbt_semantic_status = recommendation_status.get("dbt_semantic_layer") if isinstance(recommendation_status, Mapping) else None
    langfuse_status = recommendation_status.get("langfuse") if isinstance(recommendation_status, Mapping) else None
    human_review_status = recommendation_status.get("human_review") if isinstance(recommendation_status, Mapping) else None
    average_return = scorecard.get("avg_return_pct")
    average_return_text = f"{average_return}%" if average_return is not None else "-"
    average_return_after_cost = scorecard.get("avg_return_after_cost_pct")
    average_return_after_cost_text = f"{average_return_after_cost}%" if average_return_after_cost is not None else "-"
    average_alpha = scorecard.get("avg_alpha_pct")
    average_alpha_text = f"{average_alpha}%" if average_alpha is not None else "-"
    average_alpha_after_cost = scorecard.get("avg_alpha_after_cost_pct")
    average_alpha_after_cost_text = f"{average_alpha_after_cost}%" if average_alpha_after_cost is not None else "-"
    source_health_panel = decision_quality.get("source_health") or {}
    no_trade_panel = decision_quality.get("no_trade") or {}
    alert_summary = ", ".join(
        f"{item.get('category')}={item.get('total')}"
        for item in alerts.get("by_category", [])[:5]
        if isinstance(item, dict)
    ) or "-"

    lines = [
        "Evaluation Dashboard",
        (
            "Burn-in: "
            f"{burn_in.get('elapsed_days', 0)} / {burn_in.get('target_days', services.burn_in_target_days)} days "
            f"({burn_in.get('progress_pct', 0)}%) | started={burn_in.get('started_at') or '-'}"
        ),
        (
            "Scorecard: "
            f"closed={scorecard.get('closed_count', 0)} | open={scorecard.get('open_count', 0)} | "
            f"hit rate={scorecard.get('hit_rate_pct', 0)}% | avg raw={average_return_text} | "
            f"avg after cost={average_return_after_cost_text} | avg alpha={average_alpha_text} | "
            f"avg alpha after cost={average_alpha_after_cost_text}"
        ),
        (
            "Reports: "
            f"total={reports.get('total', 0)} | fallback={reports.get('fallback_total', 0)} | "
            f"last={reports.get('last_sent_at') or '-'}"
        ),
        (
            "Alerts: "
            f"total={alerts.get('total', 0)} | "
            f"{alert_summary}"
        ),
        f"Interactions: total={interactions.get('total', 0)} | last={interactions.get('last_at') or '-'}",
        (
            "MLflow: "
            f"enabled={mlflow_status.get('enabled')} | tracking={mlflow_status.get('tracking_configured')} | "
            f"experiment={mlflow_status.get('experiment_name') or '-'} | "
            f"last_run_id={mlflow_status.get('last_run_id') or '-'} | "
            f"last_kind={mlflow_status.get('last_run_kind') or '-'}"
        ),
    ]
    if isinstance(broker_status, Mapping):
        lines.append(
            f"Broker: enabled={broker_status.get('enabled')} | available={broker_status.get('available')} | last_order={broker_status.get('last_order_id') or '-'}"
        )
    if isinstance(transcript_status, Mapping):
        lines.append(
            f"Transcripts: available={transcript_status.get('available')} | cache={transcript_status.get('cache_entries') or 0}"
        )
    if isinstance(microstructure_status, Mapping):
        lines.append(
            f"Microstructure: available={microstructure_status.get('available')} | equities={microstructure_status.get('equities_dataset') or '-'} | options={microstructure_status.get('options_dataset') or '-'}"
        )
    if isinstance(live_stream_status, Mapping):
        lines.append(
            f"Live Stream: available={live_stream_status.get('available')} | dataset={live_stream_status.get('dataset') or '-'} | events={live_stream_status.get('last_event_count') or 0}"
        )
    if isinstance(braintrust_status, Mapping):
        lines.append(
            f"Braintrust: enabled={braintrust_status.get('enabled')} | configured={braintrust_status.get('configured')} | project={braintrust_status.get('project_name') or '-'}"
        )
    if isinstance(thesis_vector_status, Mapping):
        lines.append(
            f"Thesis Vector: available={thesis_vector_status.get('available')} | backend={thesis_vector_status.get('backend') or '-'} | points={thesis_vector_status.get('point_count') or 0}"
        )
    if isinstance(feature_store_status, Mapping):
        lines.append(
            f"Feature Store: available={feature_store_status.get('available')} | backend={feature_store_status.get('backend') or '-'} | recs={(feature_store_status.get('feature_counts') or {}).get('recommendation', 0) if isinstance(feature_store_status.get('feature_counts'), Mapping) else 0}"
        )
    if isinstance(backtesting_status, Mapping):
        lines.append(
            f"Backtesting: available={backtesting_status.get('available')} | backend={backtesting_status.get('backend') or '-'} | benchmark={backtesting_status.get('benchmark_ticker') or '-'}"
        )
    if isinstance(analytics_warehouse_status, Mapping):
        table_counts = analytics_warehouse_status.get("table_counts")
        lines.append(
            f"Analytics Warehouse: available={analytics_warehouse_status.get('available')} | backend={analytics_warehouse_status.get('backend') or '-'} | tables={len(table_counts) if isinstance(table_counts, Mapping) else 0}"
        )
    if isinstance(event_bus_status, Mapping):
        lines.append(
            f"Event Bus: available={event_bus_status.get('available')} | backend={event_bus_status.get('backend') or '-'} | published={event_bus_status.get('published_count') or 0}"
        )
    if isinstance(event_bus_consumer_status, Mapping):
        lines.append(
            f"Event Bus Consumer: available={event_bus_consumer_status.get('available')} | backend={event_bus_consumer_status.get('backend') or '-'} | processed={event_bus_consumer_status.get('processed_count') or 0}"
        )
    if isinstance(hot_path_cache_status, Mapping):
        lines.append(
            f"Hot Cache: available={hot_path_cache_status.get('available')} | backend={hot_path_cache_status.get('backend') or '-'} | keys={hot_path_cache_status.get('cache_keys') or 0} | stream_events={hot_path_cache_status.get('stream_event_count') or 0}"
        )
    if isinstance(semantic_analyst_status, Mapping):
        lines.append(
            f"Semantic Analyst: available={semantic_analyst_status.get('available')} | backend={semantic_analyst_status.get('backend') or '-'} | model={semantic_analyst_status.get('model_name') or '-'}"
        )
    if isinstance(workflow_status, Mapping):
        lines.append(
            f"Workflow: enabled={workflow_status.get('enabled')} | prefect={workflow_status.get('prefect_available')} | flows={len(workflow_status.get('flows') or []) if isinstance(workflow_status.get('flows'), list) else 0}"
        )
    if mlflow_status.get("warning"):
        lines.append(f"MLflow warning: {mlflow_status.get('warning')}")
    if live_latest:
        lines.append(
            "Latest live provider: "
            f"{live_latest.get('provider')} | {live_latest.get('model')} | {live_latest.get('service')}"
        )
    if isinstance(live_response, dict) and live_response:
        lines.append("Live fallback rate")
        for service_name, stats in live_response.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- {service_name}: {stats.get('fallback_rate')}% ({stats.get('fallback')}/{stats.get('total')})"
            )
    if providers:
        lines.append("Provider history")
        for item in providers[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('provider')}: ok={item.get('success_count')} fail={item.get('failure_count')} last={item.get('last_event_at') or '-'}"
            )
    if jobs:
        lines.append("Job history")
        for item in jobs[:6]:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"- {item.get('job_name')}: ok={item.get('success_count')} fail={item.get('failure_count')} last={item.get('last_event_at') or '-'}"
            )
    if reports.get("by_kind"):
        lines.append(
            "Report mix: "
            + ", ".join(
                f"{item.get('report_kind')}={item.get('total')}" for item in reports.get("by_kind", [])[:6] if isinstance(item, dict)
            )
        )
    if source_ranking:
        lines.append("Source ranking")
        for item in source_ranking[:6]:
            if not isinstance(item, dict):
                continue
            avg_return = item.get("avg_return_pct")
            avg_return_text = f"{avg_return}%" if avg_return is not None else "-"
            hit_rate = item.get("hit_rate_pct")
            hit_rate_text = f"{hit_rate}%" if hit_rate is not None else "-"
            weighted_score = item.get("weighted_score")
            weighted_text = f"{weighted_score}" if weighted_score is not None else "-"
            ttl_fit_score = item.get("ttl_fit_score")
            ttl_fit_text = f"{ttl_fit_score}" if ttl_fit_score is not None else "-"
            ttl_hit = item.get("ttl_hit_rate_pct")
            ttl_hit_text = f"{ttl_hit}%" if ttl_hit is not None else "-"
            fast_decay = item.get("fast_decay_rate_pct")
            fast_decay_text = f"{fast_decay}%" if fast_decay is not None else "-"
            source_health = item.get("source_health_score")
            source_health_text = f"{source_health}" if source_health is not None else "-"
            source_freshness = item.get("source_freshness_score")
            source_freshness_text = f"{source_freshness}" if source_freshness is not None else "-"
            best_ttl_bucket = str(item.get("best_ttl_bucket") or "-")
            thesis_alignment = str(item.get("thesis_alignment") or "-")
            thesis_alignment_pct = item.get("thesis_alignment_pct")
            thesis_alignment_text = f"{thesis_alignment}/{thesis_alignment_pct}%" if thesis_alignment_pct is not None else "-"
            lines.append(
                f"- {item.get('source')}: reports={item.get('report_mentions', 0)} chats={item.get('interaction_mentions', 0)} "
                f"picks={item.get('stock_pick_total', 0)} closed={item.get('closed_count', 0)} hit={hit_rate_text} "
                f"avg={avg_return_text} thesis={thesis_alignment_text} ttl_hit={ttl_hit_text} fast_decay={fast_decay_text} "
                f"health={source_health_text} freshness={source_freshness_text} "
                f"best_ttl={best_ttl_bucket} ttl_fit={ttl_fit_text} weighted={weighted_text}"
            )
    if source_health_panel:
        lines.append("Source health")
        lines.append(
            f"- samples={source_health_panel.get('sample_count', 0)} "
            f"avg_score={source_health_panel.get('avg_score') if source_health_panel.get('avg_score') is not None else '-'} "
            f"avg_freshness={source_health_panel.get('avg_freshness_pct') if source_health_panel.get('avg_freshness_pct') is not None else '-'}% "
            f"strong={source_health_panel.get('strong_count', 0)} mixed={source_health_panel.get('mixed_count', 0)} fragile={source_health_panel.get('fragile_count', 0)} "
            f"degraded_sla={source_health_panel.get('degraded_sla_count', 0)} outage={source_health_panel.get('outage_count', 0)}"
        )
    if no_trade_panel:
        lines.append("No-trade framework")
        lines.append(
            f"- decisions={no_trade_panel.get('decision_count', 0)} abstain={no_trade_panel.get('abstain_count', 0)} "
            f"abstain_rate={no_trade_panel.get('abstain_rate_pct') if no_trade_panel.get('abstain_rate_pct') is not None else '-'}%"
        )
        top_reasons = no_trade_panel.get("top_reasons")
        if isinstance(top_reasons, list) and top_reasons:
            reason_summary = "; ".join(
                f"{item.get('reason')} ({item.get('count')})"
                for item in top_reasons[:4]
                if isinstance(item, dict) and item.get("reason")
            )
            if reason_summary:
                lines.append(f"- top reasons: {reason_summary}")
    if thesis_ranking:
        lines.append("Thesis ranking")
        for item in thesis_ranking[:5]:
            if not isinstance(item, dict):
                continue
            avg_return = item.get("avg_return_pct")
            avg_return_text = f"{avg_return}%" if avg_return is not None else "-"
            hit_rate = item.get("hit_rate_pct")
            hit_rate_text = f"{hit_rate}%" if hit_rate is not None else "-"
            reliability = item.get("reliability_score")
            reliability_text = f"{reliability}" if reliability is not None else "-"
            lines.append(
                f"- {item.get('thesis')}: picks={item.get('stock_pick_total', 0)} closed={item.get('closed_count', 0)} "
                f"hit={hit_rate_text} avg={avg_return_text} reliability={reliability_text}"
            )
    if isinstance(thesis_lifecycle, dict) and (thesis_lifecycle.get("counts") or thesis_lifecycle.get("top_invalidations")):
        lines.append("Thesis lifecycle")
        for item in (thesis_lifecycle.get("counts") or [])[:5]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('stage')}: {item.get('count')}")
        for item in (thesis_lifecycle.get("top_invalidations") or [])[:2]:
            if isinstance(item, dict):
                lines.append(
                    f"- invalidation: {item.get('summary')} | score={item.get('score') if item.get('score') is not None else '-'} | severity={item.get('severity') or '-'}"
                )
    if isinstance(walk_forward_eval, dict) and walk_forward_eval.get("window_count"):
        lines.append("Walk-forward eval")
        lines.append(
            f"- windows={walk_forward_eval.get('window_count')} size={walk_forward_eval.get('window_size')} "
            f"avg_hit={walk_forward_eval.get('avg_hit_rate_pct')}% avg_after_cost={walk_forward_eval.get('avg_return_after_cost_pct')}%"
        )
    if execution_panel:
        lines.append("Execution panel")
        execution_filter = execution_panel.get("alert_kind_filter")
        if execution_filter:
            lines.append(f"Execution filter: {execution_filter}")
        if execution_sort:
            lines.append(f"Execution sort: {execution_sort}")
        by_alert_kind = _sort_dashboard_execution_rows(execution_panel.get("by_alert_kind"), sort_mode=execution_sort)
        if isinstance(by_alert_kind, list) and by_alert_kind:
            lines.append("Execution by alert kind")
            for item in by_alert_kind[:5]:
                if not isinstance(item, dict):
                    continue
                ttl_hit = item.get("ttl_hit_rate_pct")
                ttl_hit_text = f"{ttl_hit}%" if ttl_hit is not None else "-"
                fast_decay = item.get("fast_decay_rate_pct")
                fast_decay_text = f"{fast_decay}%" if fast_decay is not None else "-"
                hold_rate = item.get("hold_after_expiry_rate_pct")
                hold_rate_text = f"{hold_rate}%" if hold_rate is not None else "-"
                lines.append(
                    f"- {item.get('alert_kind')}: closed={item.get('closed_postmortems', 0)} ttl_hit={ttl_hit_text} "
                    f"fast_decay={fast_decay_text} hold_after_expiry={hold_rate_text} best_ttl={item.get('best_ttl_bucket') or '-'}"
                )
        lines.append(
            "Execution summary: "
            f"closed={execution_panel.get('closed_postmortems', 0)} | "
            f"ttl_hit={execution_panel.get('ttl_hit_rate_pct') if execution_panel.get('ttl_hit_rate_pct') is not None else '-'}% | "
            f"fast_decay={execution_panel.get('fast_decay_rate_pct') if execution_panel.get('fast_decay_rate_pct') is not None else '-'}% | "
            f"hold_after_expiry={execution_panel.get('hold_after_expiry_rate_pct') if execution_panel.get('hold_after_expiry_rate_pct') is not None else '-'}% | "
            f"discard_after_expiry={execution_panel.get('discard_after_expiry_rate_pct') if execution_panel.get('discard_after_expiry_rate_pct') is not None else '-'}%"
        )
        best_ttl = execution_panel.get("best_ttl_by_alert_kind")
        if isinstance(best_ttl, list) and best_ttl:
            lines.append("Best TTL by alert kind")
            for item in best_ttl[:5]:
                if not isinstance(item, dict):
                    continue
                avg_return = item.get("avg_return_pct")
                avg_return_text = f"{avg_return}%" if avg_return is not None else "-"
                hit_rate = item.get("hit_rate_pct")
                hit_rate_text = f"{hit_rate}%" if hit_rate is not None else "-"
                hold_rate = item.get("hold_rate_pct")
                hold_rate_text = f"{hold_rate}%" if hold_rate is not None else "-"
                lines.append(
                    f"- {item.get('alert_kind')}: ttl={item.get('best_ttl_bucket')} sample={item.get('sample_count', 0)} "
                    f"hit={hit_rate_text} hold={hold_rate_text} avg={avg_return_text}"
                )
        heatmap = _sort_dashboard_execution_rows(execution_panel.get("source_ttl_heatmap"), sort_mode=execution_sort)
        if isinstance(heatmap, list) and heatmap:
            lines.append("Source TTL heatmap")
            for item in heatmap[:8]:
                if not isinstance(item, dict):
                    continue
                ttl_hit = item.get("ttl_hit_rate_pct")
                ttl_hit_text = f"{ttl_hit}%" if ttl_hit is not None else "-"
                hold_rate = item.get("hold_rate_pct")
                hold_rate_text = f"{hold_rate}%" if hold_rate is not None else "-"
                avg_return = item.get("avg_return_pct")
                avg_return_text = f"{avg_return}%" if avg_return is not None else "-"
                lines.append(
                    f"- {item.get('source')} -> {item.get('alert_kind')} -> {item.get('ttl_bucket')}: "
                    f"sample={item.get('sample_count', 0)} ttl_hit={ttl_hit_text} hold={hold_rate_text} avg={avg_return_text}"
                )
    lines.append("หมายเหตุ: burn-in 7-14 วันต้องใช้เวลาจริงของระบบ ข้อมูลในหน้าจอนี้คือความคืบหน้าปัจจุบัน")
    rendered = "\n".join(lines)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="dashboard",
        question="/dashboard",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


def _resolve_dashboard_execution_view(
    update: Update,
    services: BotServices,
    args: list[str] | None,
) -> tuple[str | None, str | None]:
    explicit_filter = _parse_dashboard_execution_filter(args)
    explicit_sort = _parse_dashboard_execution_sort(args)
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.user_state_store is None:
        return explicit_filter, explicit_sort
    if args:
        normalized_args = [str(item or "").strip().casefold().replace("-", "_") for item in args if str(item or "").strip()]
        if any(item in {"all", "clear", "reset"} for item in normalized_args):
            services.user_state_store.update_preferences(conversation_key, dashboard_execution_filter="all")
            return None, explicit_sort
        if explicit_filter is not None:
            services.user_state_store.update_preferences(conversation_key, dashboard_execution_filter=explicit_filter)
            return explicit_filter, explicit_sort
    prefs = services.user_state_store.get(conversation_key)
    return prefs.dashboard_execution_filter, explicit_sort


def _parse_dashboard_execution_filter(args: list[str] | None) -> str | None:
    if not args:
        return None
    for item in args:
        raw = str(item or "").strip().casefold().replace("-", "_")
        if raw in {"stock_pick", "macro_playbook", "macro_surprise"}:
            return raw
    return None


def _parse_dashboard_execution_sort(args: list[str] | None) -> str | None:
    if not args:
        return None
    valid = {"score", "worst", "ttl_hit", "hold", "avg_return", "sample"}
    aliases = {
        "ttl": "ttl_hit",
        "ttlhit": "ttl_hit",
        "return": "avg_return",
        "avg": "avg_return",
        "samples": "sample",
    }
    for item in args:
        raw = str(item or "").strip().casefold().replace("-", "_")
        normalized = aliases.get(raw, raw)
        if normalized in valid:
            return normalized
    return None


def _sort_dashboard_execution_rows(rows: Any, *, sort_mode: str | None) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    items = [dict(item) for item in rows if isinstance(item, Mapping)]
    if not items:
        return []
    mode = (sort_mode or "score").strip().casefold()
    if mode == "ttl_hit":
        items.sort(
            key=lambda item: (
                float(item.get("ttl_hit_rate_pct") if item.get("ttl_hit_rate_pct") is not None else -9999.0),
                int(item.get("sample_count") or item.get("closed_postmortems") or 0),
                float(item.get("score") if item.get("score") is not None else item.get("best_ttl_score") or -9999.0),
            ),
            reverse=True,
        )
    elif mode == "hold":
        items.sort(
            key=lambda item: (
                float(item.get("hold_rate_pct") if item.get("hold_rate_pct") is not None else item.get("hold_after_expiry_rate_pct") if item.get("hold_after_expiry_rate_pct") is not None else -9999.0),
                int(item.get("sample_count") or item.get("closed_postmortems") or 0),
                float(item.get("score") if item.get("score") is not None else item.get("best_ttl_score") or -9999.0),
            ),
            reverse=True,
        )
    elif mode == "avg_return":
        items.sort(
            key=lambda item: (
                float(item.get("avg_return_pct") if item.get("avg_return_pct") is not None else -9999.0),
                int(item.get("sample_count") or item.get("closed_postmortems") or 0),
                float(item.get("score") if item.get("score") is not None else item.get("best_ttl_score") or -9999.0),
            ),
            reverse=True,
        )
    elif mode == "sample":
        items.sort(
            key=lambda item: (
                int(item.get("sample_count") or item.get("closed_postmortems") or 0),
                float(item.get("score") if item.get("score") is not None else item.get("best_ttl_score") or -9999.0),
            ),
            reverse=True,
        )
    elif mode == "worst":
        items.sort(
            key=lambda item: (
                float(item.get("score") if item.get("score") is not None else item.get("best_ttl_score") or 9999.0),
                float(item.get("fast_decay_rate_pct") if item.get("fast_decay_rate_pct") is not None else -9999.0) * -1.0,
                int(item.get("sample_count") or item.get("closed_postmortems") or 0) * -1,
            ),
        )
    else:
        items.sort(
            key=lambda item: (
                float(item.get("score") if item.get("score") is not None else item.get("best_ttl_score") or -9999.0),
                int(item.get("sample_count") or item.get("closed_postmortems") or 0),
                float(item.get("avg_return_pct") if item.get("avg_return_pct") is not None else -9999.0),
            ),
            reverse=True,
        )
    return items


async def backup_now_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    if services.backup_manager is None or not services.backup_manager.available():
        await _reply_text(message, "ยังไม่มี Postgres backup manager พร้อมใช้งาน")
        return

    await _reply_text(message, "กำลังสร้าง backup ของ state และ runtime history...")
    try:
        manifest = await asyncio.to_thread(services.backup_manager.create_backup, reason="manual_telegram")
    except Exception as exc:
        logger.exception("Manual backup failed: {}", exc)
        diagnostics.record_job_run(job="manual_backup", status="error", duration_ms=0, error=str(exc), detail={})
        await _reply_text(message, f"สร้าง backup ไม่สำเร็จ: {exc}")
        return

    summary = ", ".join(f"{name}={count}" for name, count in sorted(manifest.row_counts.items())[:6])
    rendered = (
        "Backup Complete\n"
        f"path: {manifest.path}\n"
        f"created_at: {manifest.created_at.isoformat()}\n"
        f"tables: {summary}"
    )
    diagnostics.record_job_run(
        job="manual_backup",
        status="ok",
        duration_ms=0,
        detail={"path": str(manifest.path), "table_count": len(manifest.row_counts)},
    )
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="backup_now",
        question="/backup_now",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return

    db_backend = "postgres" if services.database_url.strip() else "file"
    db_healthy: bool | None = None
    db_error: str | None = None
    if services.database_url.strip():
        try:
            db_healthy = PostgresStateBackend.ping_database_url(services.database_url)
        except Exception as exc:
            db_healthy = False
            db_error = str(exc)
    diagnostics.record_db_state(backend=db_backend, healthy=db_healthy, error=db_error)
    _get_mlflow_status(services)
    snapshot = diagnostics.snapshot()
    mlflow_status = snapshot.get("mlflow") or _get_mlflow_status(services)
    latest = snapshot.get("latest_provider_success") or {}
    response_stats = snapshot.get("response_stats") or {}
    jobs = snapshot.get("jobs") or {}
    alerts_today = snapshot.get("alerts_today") or {}
    db_state = snapshot.get("db_state") or {}
    circuit = snapshot.get("provider_circuit") or {}
    market_data_status = services.market_data_client.status()

    lines = [
        "Runtime Status",
        f"DB: {_render_health_label(bool(db_state.get('healthy'))) if db_state.get('healthy') is not None else 'unknown'} | backend={db_state.get('backend')} | checked_at={db_state.get('checked_at') or '-'}",
        (
            "MLflow: "
            f"enabled={mlflow_status.get('enabled')} | tracking={mlflow_status.get('tracking_configured')} | "
            f"experiment={mlflow_status.get('experiment_name') or '-'} | "
            f"last_run_id={mlflow_status.get('last_run_id') or '-'} | "
            f"last_kind={mlflow_status.get('last_run_kind') or '-'}"
        ),
    ]
    if db_state.get("error"):
        lines.append(f"DB error: {db_state.get('error')}")
    if mlflow_status.get("warning"):
        lines.append(f"MLflow warning: {mlflow_status.get('warning')}")
    if latest:
        lines.append(
            "LLM ล่าสุด: "
            f"{latest.get('provider')} | {latest.get('model')} | service={latest.get('service')} | at={latest.get('succeeded_at')}"
        )
    else:
        lines.append("LLM ล่าสุด: ยังไม่มี successful response")

    if isinstance(response_stats, dict) and response_stats:
        lines.append("Fallback Rate")
        for service_name, stats in response_stats.items():
            if not isinstance(stats, dict):
                continue
            lines.append(
                f"- {service_name}: {stats.get('fallback_rate')}% ({stats.get('fallback')}/{stats.get('total')})"
            )
    if isinstance(jobs, dict) and jobs:
        lines.append("Jobs")
        for job_name, job in sorted(jobs.items()):
            if not isinstance(job, dict):
                continue
            lines.append(
                f"- {job_name}: {job.get('last_status')} | at={job.get('last_run_at') or '-'} | ms={job.get('duration_ms') or '-'} | ok={job.get('success_count')} fail={job.get('failure_count')}"
            )
    lines.append(
        f"Alerts Today: {alerts_today.get('total', 0)} | {', '.join(f'{k}={v}' for k, v in sorted((alerts_today.get('by_category') or {}).items())) or '-'}"
    )
    if isinstance(circuit, dict) and circuit:
        lines.append("Provider Circuit")
        for provider_name, state in sorted(circuit.items()):
            if not isinstance(state, dict):
                continue
            lines.append(
                f"- {provider_name}: open={state.get('is_open')} | failures={state.get('failure_count')} | until={state.get('open_until') or '-'}"
            )
    broker_status = services.broker_client.status() if services.broker_client is not None else None
    transcript_status = services.transcript_client.status() if services.transcript_client is not None else None
    microstructure_status = services.microstructure_client.status() if services.microstructure_client is not None else None
    live_stream_status = services.live_market_stream_client.status() if services.live_market_stream_client is not None else None
    workflow_status = services.workflow_orchestrator.status() if services.workflow_orchestrator is not None else None
    braintrust_observer = getattr(services.recommendation_service, "braintrust_observer", None)
    braintrust_status = braintrust_observer.status() if braintrust_observer is not None else None
    recommendation_status_getter = getattr(services.recommendation_service, "status", None)
    recommendation_status = recommendation_status_getter() if callable(recommendation_status_getter) else {}
    ai_portfolio_status = services.ai_simulated_portfolio_service.status(services.telegram_report_chat_id) if services.ai_simulated_portfolio_service is not None else None
    ai_portfolio_snapshot = (
        await services.ai_simulated_portfolio_service.build_snapshot(conversation_key=services.telegram_report_chat_id)
        if services.ai_simulated_portfolio_service is not None
        else None
    )
    llm_status = recommendation_status.get("llm") if isinstance(recommendation_status, Mapping) else None
    thesis_vector_status = recommendation_status.get("thesis_vector_store") if isinstance(recommendation_status, Mapping) else None
    feature_store_status = recommendation_status.get("feature_store") if isinstance(recommendation_status, Mapping) else None
    backtesting_status = recommendation_status.get("backtesting") if isinstance(recommendation_status, Mapping) else None
    analytics_warehouse_status = recommendation_status.get("analytics_warehouse") if isinstance(recommendation_status, Mapping) else None
    event_bus_status = recommendation_status.get("event_bus") if isinstance(recommendation_status, Mapping) else None
    event_bus_consumer_status = recommendation_status.get("event_bus_consumer") if isinstance(recommendation_status, Mapping) else None
    hot_path_cache_status = recommendation_status.get("hot_path_cache") if isinstance(recommendation_status, Mapping) else None
    semantic_analyst_status = recommendation_status.get("semantic_analyst") if isinstance(recommendation_status, Mapping) else None
    ownership_status = recommendation_status.get("ownership") if isinstance(recommendation_status, Mapping) else None
    order_flow_status = recommendation_status.get("order_flow") if isinstance(recommendation_status, Mapping) else None
    policy_feed_status = recommendation_status.get("policy_feed") if isinstance(recommendation_status, Mapping) else None
    dbt_semantic_status = recommendation_status.get("dbt_semantic_layer") if isinstance(recommendation_status, Mapping) else None
    langfuse_status = recommendation_status.get("langfuse") if isinstance(recommendation_status, Mapping) else None
    human_review_status = recommendation_status.get("human_review") if isinstance(recommendation_status, Mapping) else None
    if isinstance(llm_status, Mapping):
        lines.append("LLM Providers")
        for item in _render_llm_provider_status_lines(llm_status):
            lines.append(item)
    lines.append("Key Providers")
    lines.append(
        "Telegram: "
        f"transport={services.telegram_transport or 'polling'} | "
        f"webhook_path={services.telegram_webhook_path or '-'} | "
        f"webhook_port={services.telegram_webhook_port or '-'}"
    )
    if services.telegram_transport == "webhook" and services.telegram_webhook_url:
        lines.append(f"Telegram webhook: {services.telegram_webhook_url}{services.telegram_webhook_path}")
    if isinstance(ai_portfolio_status, Mapping):
        ai_profile = str(ai_portfolio_status.get("profile_name") or "-").strip() or "-"
        ai_universe = ",".join(str(item) for item in (ai_portfolio_status.get("allowed_asset_types") or []) if str(item).strip())
        lines.append(
            "AI Portfolio: "
            f"enabled={ai_portfolio_status.get('enabled')} | backend={ai_portfolio_status.get('backend') or '-'} | "
            f"value={float((ai_portfolio_snapshot or {}).get('total_value') or 0.0):.2f} | "
            f"cash={float((ai_portfolio_snapshot or {}).get('cash') or 0.0):.2f} | "
            f"holdings={ai_portfolio_status.get('holding_count') or 0} | "
            f"profile={ai_profile} | universe={ai_universe or '-'} | "
            f"last={ai_portfolio_status.get('last_rebalanced_at') or '-'}"
        )
        if ai_portfolio_status.get("last_action_summary"):
            lines.append(f"AI Portfolio action: {ai_portfolio_status.get('last_action_summary')}")
    research_status = services.research_client.status() if services.research_client is not None else None
    transcript_mode = _render_transcript_fallback_label(research_status)
    if isinstance(transcript_status, Mapping):
        lines.append(
            "Transcript: "
            f"{_classify_component_status(transcript_status)} | backend={transcript_status.get('backend') or '-'} | "
            f"fallback={transcript_mode} | cache={transcript_status.get('cache_entries') or 0}"
        )
        if transcript_status.get("warning"):
            lines.append(f"Transcript warning: {transcript_status.get('warning')}")
    elif services.research_client is not None and services.research_client.available():
        lines.append(f"Transcript: degraded | backend=- | fallback={transcript_mode} | cache=0")
    lines.append(
        "Research: "
        f"{_classify_component_status(research_status)}"
        f" | providers={', '.join((research_status.get('provider_order') or [])) if isinstance(research_status, Mapping) else '-'}"
        f" | pages={research_status.get('cached_pages') if isinstance(research_status, Mapping) else 0}"
    )
    lines.append(
        "Market Data: "
        f"{_classify_component_status(market_data_status)} | order={','.join(market_data_status.get('provider_order') or []) or '-'}"
    )
    lines.append(
        "Alpha Vantage: "
        f"{_classify_alpha_vantage_status(market_data_status)}"
    )
    if market_data_status.get("alpha_vantage_warning"):
        lines.append(f"Alpha Vantage warning: {market_data_status.get('alpha_vantage_warning')}")
    lines.append(
        "Nasdaq Data Link: "
        f"{_classify_nasdaq_status(market_data_status)} | datasets={len(market_data_status.get('nasdaq_data_link_datasets') or [])}"
    )
    if market_data_status.get("nasdaq_data_link_warning"):
        lines.append(f"Nasdaq warning: {market_data_status.get('nasdaq_data_link_warning')}")
    if isinstance(broker_status, Mapping):
        lines.append(
            "Broker: "
            f"available={broker_status.get('available')} | provider={broker_status.get('provider') or '-'} | "
            f"last_order={broker_status.get('last_order_id') or '-'}"
        )
        if broker_status.get("warning"):
            lines.append(f"Broker warning: {broker_status.get('warning')}")
    if isinstance(microstructure_status, Mapping):
        lines.append(
            "Microstructure: "
            f"available={microstructure_status.get('available')} | equities={microstructure_status.get('equities_dataset') or '-'} | "
            f"options={microstructure_status.get('options_dataset') or '-'}"
        )
        if microstructure_status.get("warning"):
            lines.append(f"Microstructure warning: {microstructure_status.get('warning')}")
    if isinstance(live_stream_status, Mapping):
        lines.append(
            "Live Stream: "
            f"available={live_stream_status.get('available')} | dataset={live_stream_status.get('dataset') or '-'} | "
            f"schema={live_stream_status.get('schema') or '-'} | last_events={live_stream_status.get('last_event_count') or 0}"
        )
        if live_stream_status.get("warning"):
            lines.append(f"Live Stream warning: {live_stream_status.get('warning')}")
    if isinstance(braintrust_status, Mapping):
        lines.append(
            "Braintrust: "
            f"enabled={braintrust_status.get('enabled')} | configured={braintrust_status.get('configured')} | "
            f"project={braintrust_status.get('project_name') or '-'} | experiment={braintrust_status.get('experiment_name') or '-'}"
        )
        if braintrust_status.get("warning"):
            lines.append(f"Braintrust warning: {braintrust_status.get('warning')}")
    if isinstance(thesis_vector_status, Mapping):
        lines.append(
            "Thesis Vector: "
            f"available={thesis_vector_status.get('available')} | configured={thesis_vector_status.get('configured')} | "
            f"backend={thesis_vector_status.get('backend') or '-'} | points={thesis_vector_status.get('point_count') or 0}"
        )
        if thesis_vector_status.get("warning"):
            lines.append(f"Thesis Vector warning: {thesis_vector_status.get('warning')}")
    if isinstance(feature_store_status, Mapping):
        feature_counts = feature_store_status.get("feature_counts")
        recommendation_count = feature_counts.get("recommendation", 0) if isinstance(feature_counts, Mapping) else 0
        outcome_count = feature_counts.get("outcome", 0) if isinstance(feature_counts, Mapping) else 0
        lines.append(
            "Feature Store: "
            f"available={feature_store_status.get('available')} | configured={feature_store_status.get('configured')} | "
            f"backend={feature_store_status.get('backend') or '-'} | recs={recommendation_count} | outcomes={outcome_count}"
        )
        if feature_store_status.get("warning"):
            lines.append(f"Feature Store warning: {feature_store_status.get('warning')}")
    if isinstance(backtesting_status, Mapping):
        lines.append(
            "Backtesting: "
            f"available={backtesting_status.get('available')} | backend={backtesting_status.get('backend') or '-'} | "
            f"benchmark={backtesting_status.get('benchmark_ticker') or '-'} | lookback={backtesting_status.get('lookback_period') or '-'}"
        )
        if backtesting_status.get("warning"):
            lines.append(f"Backtesting warning: {backtesting_status.get('warning')}")
    if isinstance(analytics_warehouse_status, Mapping):
        table_counts = analytics_warehouse_status.get("table_counts")
        lines.append(
            "Analytics Warehouse: "
            f"available={analytics_warehouse_status.get('available')} | configured={analytics_warehouse_status.get('configured')} | "
            f"backend={analytics_warehouse_status.get('backend') or '-'} | tables={len(table_counts) if isinstance(table_counts, Mapping) else 0}"
        )
        if analytics_warehouse_status.get("warning"):
            lines.append(f"Analytics Warehouse warning: {analytics_warehouse_status.get('warning')}")
    if isinstance(event_bus_status, Mapping):
        lines.append(
            "Event Bus: "
            f"available={event_bus_status.get('available')} | configured={event_bus_status.get('configured')} | "
            f"backend={event_bus_status.get('backend') or '-'} | published={event_bus_status.get('published_count') or 0}"
        )
        if event_bus_status.get("warning"):
            lines.append(f"Event Bus warning: {event_bus_status.get('warning')}")
    if isinstance(event_bus_consumer_status, Mapping):
        lines.append(
            "Event Bus Consumer: "
            f"available={event_bus_consumer_status.get('available')} | configured={event_bus_consumer_status.get('configured')} | "
            f"backend={event_bus_consumer_status.get('backend') or '-'} | processed={event_bus_consumer_status.get('processed_count') or 0}"
        )
        if event_bus_consumer_status.get("warning"):
            lines.append(f"Event Bus Consumer warning: {event_bus_consumer_status.get('warning')}")
    if isinstance(hot_path_cache_status, Mapping):
        lines.append(
            "Hot Cache: "
            f"available={hot_path_cache_status.get('available')} | configured={hot_path_cache_status.get('configured')} | "
            f"backend={hot_path_cache_status.get('backend') or '-'} | keys={hot_path_cache_status.get('cache_keys') or 0} | stream_events={hot_path_cache_status.get('stream_event_count') or 0}"
        )
        if hot_path_cache_status.get("warning"):
            lines.append(f"Hot Cache warning: {hot_path_cache_status.get('warning')}")
    if isinstance(semantic_analyst_status, Mapping):
        lines.append(
            "Semantic Analyst: "
            f"available={semantic_analyst_status.get('available')} | configured={semantic_analyst_status.get('configured')} | "
            f"backend={semantic_analyst_status.get('backend') or '-'} | model={semantic_analyst_status.get('model_name') or '-'}"
        )
        if semantic_analyst_status.get("warning"):
            lines.append(f"Semantic Analyst warning: {semantic_analyst_status.get('warning')}")
    if isinstance(ownership_status, Mapping):
        lines.append(
            "Ownership: "
            f"configured={ownership_status.get('configured')} | manager_ciks={ownership_status.get('manager_cik_count') or 0} | "
            f"cache={ownership_status.get('cache_entries') or 0}"
        )
        if ownership_status.get("warning"):
            lines.append(f"Ownership warning: {ownership_status.get('warning')}")
    if isinstance(order_flow_status, Mapping):
        lines.append(
            "Order Flow: "
            f"enabled={order_flow_status.get('enabled')} | configured={order_flow_status.get('configured')} | "
            f"cache={order_flow_status.get('cache_entries') or 0}"
        )
        if order_flow_status.get("warning"):
            lines.append(f"Order Flow warning: {order_flow_status.get('warning')}")
    if isinstance(policy_feed_status, Mapping):
        lines.append(
            "Policy Feed: "
            f"enabled={policy_feed_status.get('enabled')} | cache={policy_feed_status.get('cache_entries') or 0}"
        )
        if policy_feed_status.get("warning"):
            lines.append(f"Policy Feed warning: {policy_feed_status.get('warning')}")
    if isinstance(dbt_semantic_status, Mapping):
        lines.append(
            "dbt Semantic: "
            f"enabled={dbt_semantic_status.get('enabled')} | project={dbt_semantic_status.get('project_name') or '-'} | "
            f"schema={dbt_semantic_status.get('target_schema') or '-'}"
        )
        if dbt_semantic_status.get("warning"):
            lines.append(f"dbt Semantic warning: {dbt_semantic_status.get('warning')}")
    if isinstance(langfuse_status, Mapping):
        lines.append(
            "Langfuse: "
            f"enabled={langfuse_status.get('enabled')} | configured={langfuse_status.get('configured')} | "
            f"events={langfuse_status.get('event_count') or 0}"
        )
        if langfuse_status.get("warning"):
            lines.append(f"Langfuse warning: {langfuse_status.get('warning')}")
    if isinstance(human_review_status, Mapping):
        lines.append(
            "Human Review: "
            f"enabled={human_review_status.get('enabled')} | pending={human_review_status.get('pending_count') or 0} | "
            f"enqueued={human_review_status.get('enqueued_count') or 0}"
        )
        if human_review_status.get("warning"):
            lines.append(f"Human Review warning: {human_review_status.get('warning')}")
    if isinstance(workflow_status, Mapping):
        flow_count = len(workflow_status.get("flows") or []) if isinstance(workflow_status.get("flows"), list) else 0
        lines.append(
            "Workflow: "
            f"available={workflow_status.get('available')} | prefect={workflow_status.get('prefect_available')} | flows={flow_count}"
        )
        if workflow_status.get("warning"):
            lines.append(f"Workflow warning: {workflow_status.get('warning')}")
    lines.append("Secret Rotation: แนะนำให้ rotate key/token ที่เคยแชร์หรือใช้ทดสอบก่อน deploy production")
    rendered = "\n".join(lines)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="status",
        question="/status",
        response_text=rendered,
        fallback_used=False,
        model=None,
    )
    await _reply_text(message, rendered)


def _render_health_label(value: bool) -> str:
    return "ok" if value else "error"


def _classify_component_status(status: Mapping[str, Any] | None) -> str:
    if not isinstance(status, Mapping):
        return "disabled"
    warning_text = str(status.get("warning") or "").casefold()
    if any(token in warning_text for token in ("429", "rate limit", "quota")):
        return "rate_limited"
    if any(token in warning_text for token in ("401", "403", "forbidden", "restricted")):
        return "restricted"
    if status.get("disabled") is True:
        return "disabled"
    if status.get("available") is True or status.get("enabled") is True:
        return "ok"
    if status.get("configured") is True:
        return "degraded"
    return "disabled"


def _classify_nasdaq_status(status: Mapping[str, Any] | None) -> str:
    if not isinstance(status, Mapping):
        return "disabled"
    if status.get("nasdaq_data_link_disabled") is True:
        return "disabled"
    configured_sources = status.get("configured_sources")
    if isinstance(configured_sources, Mapping) and configured_sources.get("nasdaq_data_link"):
        return "ok"
    if status.get("nasdaq_data_link_datasets"):
        return "restricted"
    return "disabled"


def _classify_alpha_vantage_status(status: Mapping[str, Any] | None) -> str:
    if not isinstance(status, Mapping):
        return "disabled"
    if status.get("alpha_vantage_disabled") is True:
        return "disabled"
    configured_sources = status.get("configured_sources")
    if isinstance(configured_sources, Mapping) and configured_sources.get("alpha_vantage"):
        warning = str(status.get("alpha_vantage_warning") or "").casefold()
        if "rate limited" in warning:
            return "rate_limited"
        return "ok"
    if status.get("alpha_vantage_warning"):
        return "rate_limited"
    return "disabled"


def _render_transcript_fallback_label(research_status: Mapping[str, Any] | None) -> str:
    if not isinstance(research_status, Mapping) or not research_status.get("available"):
        return "disabled"
    providers = [str(item).strip() for item in (research_status.get("provider_order") or []) if str(item).strip()]
    return f"research_proxy[{'+'.join(providers)}]" if providers else "research_proxy"


def _render_llm_provider_status_lines(status: Mapping[str, Any]) -> list[str]:
    provider_statuses = status.get("provider_statuses")
    if not isinstance(provider_statuses, list) or not provider_statuses:
        return ["- none configured"]
    rendered: list[str] = []
    for item in provider_statuses:
        if not isinstance(item, Mapping):
            continue
        state = str(item.get("state") or "idle").strip() or "idle"
        provider = str(item.get("provider") or "-").strip() or "-"
        model = str(item.get("last_model") or "").strip()
        status_code = item.get("last_status_code")
        failure_count = item.get("failure_count")
        cooldown_until = str(item.get("cooldown_until") or "").strip()
        extras: list[str] = []
        if model:
            extras.append(f"model={model}")
        if status_code is not None:
            extras.append(f"code={status_code}")
        if failure_count:
            extras.append(f"failures={failure_count}")
        if cooldown_until:
            extras.append(f"until={cooldown_until}")
        rendered.append(f"- {provider}: {state}" + (f" | {' | '.join(extras)}" if extras else ""))
    return rendered or ["- none configured"]


async def analyst_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    question = " ".join(context.args or []).strip()
    if not question:
        await _reply_text(message, "ตัวอย่าง: /analyst recommendation fallback trends")
        return
    await _send_typing(update, context)
    answer = await services.recommendation_service.answer_analytics_question(question=question)
    rendered = f"Analytics Analyst\nQuestion: {question}\n\n{answer}"
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="analytics_analyst",
        question=f"/analyst {question}",
        response_text=rendered,
        fallback_used=False,
        model="semantic-analyst",
    )
    await _reply_text(message, rendered)


async def review_queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    pending = services.recommendation_service.list_pending_reviews(limit=10)
    if not pending:
        await _reply_text(message, "Human Review Queue\n- ไม่มีรายการรอตรวจ")
        return
    lines = ["Human Review Queue"]
    for item in pending[:10]:
        review_id = str(item.get("review_id") or "-").strip()
        artifact_key = str(item.get("artifact_key") or "-").strip()
        confidence = item.get("confidence_score")
        model = str(item.get("model") or "-").strip()
        question = str(item.get("question") or "").strip()
        lines.append(
            f"- {review_id} | artifact={artifact_key} | confidence={confidence if confidence is not None else '-'} | model={model} | question={question[:80] or '-'}"
        )
    await _reply_text(message, "\n".join(lines))


async def review_done_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    services = _get_services(context)
    if not _is_admin_chat(update, services):
        await _reply_text(message, "คำสั่งนี้อนุญาตเฉพาะ admin/report chat")
        return
    args = list(context.args or [])
    if len(args) < 2:
        await _reply_text(message, "ตัวอย่าง: /reviewdone review-abc accepted 0.82 note=clear thesis")
        return
    review_id = str(args[0]).strip()
    decision = str(args[1]).strip() or "accepted"
    score: float | None = None
    note_parts: list[str] = []
    for token in args[2:]:
        if token.startswith("note="):
            note_parts.append(token.split("=", 1)[1])
            continue
        if score is None:
            try:
                score = float(token)
                continue
            except ValueError:
                pass
        note_parts.append(token)
    completed = services.recommendation_service.complete_human_review(
        review_id=review_id,
        decision=decision,
        score=score,
        note=" ".join(part for part in note_parts if part).strip() or None,
    )
    if not completed:
        await _reply_text(message, f"ไม่พบ review id: {review_id}")
        return
    await _reply_text(
        message,
        (
            "Human Review Completed\n"
            f"- review_id: {completed.get('review_id')}\n"
            f"- artifact_key: {completed.get('artifact_key')}\n"
            f"- decision: {completed.get('decision')}\n"
            f"- score: {completed.get('score') if completed.get('score') is not None else '-'}"
        ),
    )


async def market_update_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    services = _get_services(context)
    if await _is_rate_limited(update, services, message):
        return

    await _send_typing(update, context)
    await message.reply_text("กำลังสรุปภาพรวมตลาดและแนวทางจัดพอร์ตล่าสุด...")
    result = await services.recommendation_service.generate_market_update(
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        research_client=services.research_client,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
        conversation_key=_conversation_key(update),
        portfolio_holdings=_get_portfolio_holdings(update, services),
    )
    rendered_text = await _append_ai_portfolio_summary(services, result.recommendation_text)
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="market_update",
        question="/market_update",
        response_text=rendered_text,
        fallback_used=result.fallback_used,
        model=result.model,
        detail=_build_result_history_detail(source="market_update", result=result),
    )
    await _reply_text(message, rendered_text)


async def handle_quick_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return

    services = _get_services(context)
    if await _is_rate_limited(update, services, query.message):
        await query.answer("กรุณารอสักครู่แล้วลองอีกครั้ง", show_alert=False)
        return

    quick_action = QUICK_ACTIONS.get(query.data or "")
    await query.answer()
    if quick_action is None:
        await query.message.reply_text("ไม่พบเมนูที่เลือก กรุณาลองใหม่อีกครั้ง")
        return

    question, asset_scope, fallback_verbosity = quick_action
    await _send_typing(update, context)
    await query.message.reply_text("กำลังวิเคราะห์ข้อมูลล่าสุด...")

    result = await services.recommendation_service.answer_user_question(
        question=question,
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        research_client=services.research_client,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
        conversation_key=_conversation_key(update),
        asset_scope=asset_scope,
        fallback_verbosity_override=fallback_verbosity,
        portfolio_holdings=_get_portfolio_holdings(update, services),
    )
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="quick_action",
        question=question,
        response_text=result.recommendation_text,
        fallback_used=result.fallback_used,
        model=result.model,
        detail=_build_result_history_detail(source="quick_action", result=result),
    )
    _record_stock_pick_scorecards_for_result(
        services,
        chat_id=str(update.effective_chat.id) if update.effective_chat is not None else None,
        source_kind="quick_action",
        result_payload=result.input_payload,
    )
    await _reply_text(query.message, result.recommendation_text, reply_markup=_build_main_menu())


async def handle_menu_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return

    menu_actions = {
        CALLBACK_MENU_HELP: help_command,
        CALLBACK_MENU_PROFILE: profile_command,
        CALLBACK_MENU_PORTFOLIO: portfolio_command,
        CALLBACK_MENU_WATCHLIST: watchlist_command,
        CALLBACK_MENU_PREFS: prefs_command,
        CALLBACK_MENU_REPORT_NOW: report_now_command,
        CALLBACK_MENU_MARKET_UPDATE: market_update_command,
        CALLBACK_MENU_SCORECARD: scorecard_command,
    }
    action_handler = menu_actions.get(query.data or "")
    await query.answer()
    if action_handler is None:
        await query.message.reply_text("ไม่พบเมนูที่เลือก กรุณาลองใหม่อีกครั้ง", reply_markup=_build_main_menu())
        return
    await action_handler(update, context)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.text is None:
        return

    services = _get_services(context)
    if await _is_rate_limited(update, services, message):
        return

    user_question = message.text.strip()
    if not user_question:
        return

    await _send_typing(update, context)
    await message.reply_text("กำลังวิเคราะห์คำถามของคุณ...")

    result = await services.recommendation_service.answer_user_question(
        question=user_question,
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        research_client=services.research_client,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
        conversation_key=_conversation_key(update),
        portfolio_holdings=_get_portfolio_holdings(update, services),
    )
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="chat",
        question=user_question,
        response_text=result.recommendation_text,
        fallback_used=result.fallback_used,
        model=result.model,
        detail=_build_result_history_detail(source="chat", result=result),
    )
    _record_stock_pick_scorecards_for_result(
        services,
        chat_id=str(update.effective_chat.id) if update.effective_chat is not None else None,
        source_kind="chat",
        result_payload=result.input_payload,
    )
    await _reply_text(message, result.recommendation_text, reply_markup=_build_main_menu())


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram handler error: {}", context.error)
    log_event("telegram_handler_error", level="error", error=str(context.error))
    if isinstance(update, Update) and update.effective_message is not None:
        await update.effective_message.reply_text("เกิดข้อผิดพลาดระหว่างประมวลผลคำขอ กรุณาลองใหม่อีกครั้ง")


def _get_services(context: ContextTypes.DEFAULT_TYPE) -> BotServices:
    services = context.application.bot_data.get(BOT_SERVICES_KEY)
    if not isinstance(services, BotServices):
        raise RuntimeError("Telegram bot services are not configured")
    return services


def _get_mlflow_status(services: BotServices) -> dict[str, Any]:
    observer = getattr(services.recommendation_service, "mlflow_observer", None)
    status_getter = getattr(observer, "status", None)
    status = status_getter() if callable(status_getter) else None
    if not isinstance(status, Mapping):
        status = {
            "enabled": False,
            "tracking_configured": False,
            "experiment_name": None,
            "warning": None,
            "last_run_id": None,
            "last_run_kind": None,
            "last_run_name": None,
        }
    normalized = dict(status)
    diagnostics.record_mlflow_state(normalized)
    return normalized


def _get_user_preferences(update: Update, services: BotServices) -> UserPreferences:
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.user_state_store is None:
        return UserPreferences()
    return services.user_state_store.get(conversation_key)


def _get_portfolio_holdings(update: Update, services: BotServices) -> tuple[PortfolioHolding, ...]:
    conversation_key = _conversation_key(update)
    if conversation_key is None or services.portfolio_state_store is None:
        return ()
    return services.portfolio_state_store.list_holdings(conversation_key)


def _parse_paper_trade_args(args: list[str] | None) -> tuple[str, float, float | None] | None:
    if not args or len(args) < 2:
        return None
    symbol = str(args[0] or "").strip().upper()
    if not symbol:
        return None
    try:
        qty = float(args[1])
    except (TypeError, ValueError):
        return None
    if qty <= 0:
        return None
    limit_price: float | None = None
    if len(args) >= 3:
        try:
            limit_price = float(args[2])
        except (TypeError, ValueError):
            return None
        if limit_price <= 0:
            return None
    return symbol, qty, limit_price


def _build_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("⚡ สรุปด่วน", callback_data=CALLBACK_QUICK_SUMMARY),
                InlineKeyboardButton("📰 สรุปเทรนด์ตลาดโลก", callback_data=CALLBACK_GLOBAL_TREND),
            ],
            [
                InlineKeyboardButton("🥇 วิเคราะห์ทองคำ", callback_data=CALLBACK_GOLD),
                InlineKeyboardButton("📈 วิเคราะห์หุ้นสหรัฐ", callback_data=CALLBACK_US_STOCKS),
            ],
            [
                InlineKeyboardButton("📊 วิเคราะห์ ETF", callback_data=CALLBACK_ETF),
                InlineKeyboardButton("🏆 หุ้นเด่น 5 ตัว", callback_data=CALLBACK_STOCK_IDEAS),
            ],
            [
                InlineKeyboardButton("👤 โปรไฟล์", callback_data=CALLBACK_MENU_PROFILE),
                InlineKeyboardButton("💼 พอร์ต", callback_data=CALLBACK_MENU_PORTFOLIO),
            ],
            [
                InlineKeyboardButton("👀 Watchlist", callback_data=CALLBACK_MENU_WATCHLIST),
                InlineKeyboardButton("⚙️ ตั้งค่า", callback_data=CALLBACK_MENU_PREFS),
            ],
            [
                InlineKeyboardButton("🗞 รายงานล่าสุด", callback_data=CALLBACK_MENU_REPORT_NOW),
                InlineKeyboardButton("📈 Market Update", callback_data=CALLBACK_MENU_MARKET_UPDATE),
            ],
            [
                InlineKeyboardButton("📚 วิธีใช้", callback_data=CALLBACK_MENU_HELP),
                InlineKeyboardButton("🧪 Scorecard", callback_data=CALLBACK_MENU_SCORECARD),
            ],
        ]
    )


def _conversation_key(update: Update) -> str | None:
    chat = update.effective_chat
    if chat is None:
        return None
    return str(chat.id)


def _is_admin_chat(update: Update, services: BotServices) -> bool:
    chat = update.effective_chat
    if chat is None:
        return False
    report_chat_id = (services.telegram_report_chat_id or "").strip()
    if not report_chat_id:
        return True
    return str(chat.id) == report_chat_id


def _infer_report_kind_now() -> str:
    bangkok_now = datetime.now(timezone.utc) + timedelta(hours=7)
    hour = bangkok_now.hour
    if hour < 12:
        return "morning"
    if hour < 18:
        return "midday"
    return "closing"


async def _append_ai_portfolio_summary(services: BotServices, text: str) -> str:
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        return text
    snapshot = await ai_service.build_snapshot(conversation_key=services.telegram_report_chat_id)
    summary = ai_service.render_report_summary_text(snapshot)
    return f"{text}\n\n{summary}" if summary.strip() else text


def _build_result_history_detail(*, source: str, result: object) -> dict[str, Any]:
    detail: dict[str, Any] = {"source": source}
    input_payload = getattr(result, "input_payload", None)
    if isinstance(input_payload, Mapping):
        coverage = RecommendationService.summarize_source_coverage(input_payload)
        if coverage.get("used_sources"):
            detail["source_coverage"] = coverage
        if isinstance(input_payload.get("source_health"), Mapping):
            detail["source_health"] = dict(input_payload.get("source_health") or {})
        if isinstance(input_payload.get("champion_challenger"), Mapping):
            detail["champion_challenger"] = dict(input_payload.get("champion_challenger") or {})
        if isinstance(input_payload.get("no_trade_decision"), Mapping):
            detail["no_trade_decision"] = dict(input_payload.get("no_trade_decision") or {})
    return detail


def _record_interaction(
    services: BotServices,
    *,
    conversation_key: str | None,
    interaction_kind: str,
    question: str | None,
    response_text: str,
    fallback_used: bool,
    model: str | None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    if services.runtime_history_store is None:
        return
    services.runtime_history_store.record_user_interaction(
        conversation_key=conversation_key,
        interaction_kind=interaction_kind,
        question=question,
        response_excerpt=response_text[:400],
        fallback_used=fallback_used,
        model=model,
        detail=detail or {},
    )


def _record_stock_pick_scorecards_for_result(
    services: BotServices,
    *,
    chat_id: str | None,
    source_kind: str,
    result_payload: object,
) -> None:
    if services.runtime_history_store is None or not isinstance(result_payload, dict):
        return
    coverage = RecommendationService.summarize_source_coverage(result_payload)
    picks = result_payload.get("stock_picks")
    if not isinstance(picks, list) or not picks:
        return

    due_at = datetime.now(timezone.utc) + timedelta(days=max(1, services.stock_pick_evaluation_horizon_days))
    day_key = datetime.now(timezone.utc).date().isoformat()
    for item in picks[:5]:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        entry_price = item.get("price")
        if not ticker:
            continue
        try:
            price_value = float(entry_price)
        except (TypeError, ValueError):
            continue
        if price_value <= 0:
            continue

        key_payload = f"{chat_id or 'unknown'}|{source_kind}|{ticker}|{day_key}|{item.get('score')}|{item.get('confidence_score')}"
        recommendation_key = f"manual:{hashlib.sha1(key_payload.encode('utf-8')).hexdigest()[:16]}"
        confidence_score = None
        composite_score = None
        try:
            if item.get("confidence_score") is not None:
                confidence_score = float(item.get("confidence_score"))
        except (TypeError, ValueError):
            confidence_score = None
        try:
            if item.get("score") is not None:
                composite_score = float(item.get("score"))
        except (TypeError, ValueError):
            composite_score = None
        services.runtime_history_store.record_stock_pick_candidate(
            recommendation_key=recommendation_key,
            due_at=due_at,
            source_kind=source_kind,
            chat_id=chat_id,
            ticker=ticker,
            company_name=str(item.get("company_name") or ticker),
            stance=str(item.get("stance") or "watch"),
            confidence_score=confidence_score,
            confidence_label=str(item.get("confidence_label") or "") or None,
            composite_score=composite_score,
            entry_price=price_value,
            detail={
                "due_at": due_at.isoformat(),
                "source_coverage": coverage,
                "thesis_memory": result_payload.get("thesis_memory")[:2] if isinstance(result_payload.get("thesis_memory"), list) else [],
                "thesis_summary": _build_stock_pick_thesis_summary(item),
                "macro_headline": (
                    result_payload.get("macro_intelligence", {}).get("headline")
                    if isinstance(result_payload.get("macro_intelligence"), dict)
                    else None
                ),
                "macro_signals": (
                    result_payload.get("macro_intelligence", {}).get("signals")
                    if isinstance(result_payload.get("macro_intelligence"), dict)
                    else []
                ),
                "position_size_pct": item.get("suggested_position_size_pct"),
                "ttl_minutes": item.get("signal_ttl_minutes"),
                "benchmark": item.get("benchmark"),
                "benchmark_ticker": item.get("benchmark_ticker"),
                "source_health": result_payload.get("source_health") if isinstance(result_payload.get("source_health"), dict) else {},
                "no_trade_decision": (
                    result_payload.get("no_trade_decision") if isinstance(result_payload.get("no_trade_decision"), dict) else {}
                ),
                "portfolio_constraints": (
                    result_payload.get("portfolio_constraints")
                    if isinstance(result_payload.get("portfolio_constraints"), dict)
                    else {}
                ),
                "execution_realism": item.get("execution_realism") if isinstance(item.get("execution_realism"), dict) else {},
            },
        )


def _build_stock_pick_thesis_summary(item: Mapping[str, Any]) -> str | None:
    rationale = item.get("rationale")
    if not isinstance(rationale, list):
        return None
    snippets = [str(part).strip() for part in rationale[:2] if str(part).strip()]
    if not snippets:
        return None
    return "; ".join(snippets)


def _extract_primary_scorecard_thesis(detail: Mapping[str, Any]) -> str | None:
    thesis_summary = str(detail.get("thesis_summary") or "").strip()
    if thesis_summary:
        return thesis_summary
    thesis_memory = detail.get("thesis_memory")
    if isinstance(thesis_memory, list):
        for item in thesis_memory:
            if not isinstance(item, Mapping):
                continue
            thesis_text = str(item.get("thesis_text") or "").strip()
            if thesis_text:
                return thesis_text
    macro_headline = str(detail.get("macro_headline") or "").strip()
    return macro_headline or None


async def _is_rate_limited(update: Update, services: BotServices, message: Message) -> bool:
    chat_key = _conversation_key(update)
    if chat_key is None:
        return False

    now = time.monotonic()
    last_request_at = _LAST_REQUEST_BY_CHAT.get(chat_key)
    min_interval = max(0.0, services.min_request_interval_seconds)
    if last_request_at is not None and now - last_request_at < min_interval:
        remaining = max(0.0, min_interval - (now - last_request_at))
        await message.reply_text(f"คำขอถูกส่งถี่เกินไป กรุณารอประมาณ {remaining:.1f} วินาที")
        return True
    _LAST_REQUEST_BY_CHAT[chat_key] = now
    return False


async def _send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None:
        return
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)


async def _reply_text(message: Message, text: str, *, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    chunks = _chunk_text(text, limit=MAX_TELEGRAM_MESSAGE_LENGTH)
    for index, chunk in enumerate(chunks):
        await message.reply_text(chunk, reply_markup=reply_markup if index == 0 else None)


def _chunk_text(text: str, *, limit: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return ["ขออภัย ระบบไม่พบข้อความสำหรับตอบกลับ"]
    if len(normalized) <= limit:
        return [normalized]

    chunks: list[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return [chunk for chunk in chunks if chunk]
