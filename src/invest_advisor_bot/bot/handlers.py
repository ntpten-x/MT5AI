from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

from loguru import logger
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from invest_advisor_bot.bot.alert_state import AlertStateStore
from invest_advisor_bot.bot.backup_manager import BackupManager
from invest_advisor_bot.bot.portfolio_state import PortfolioHolding, PortfolioStateStore
from invest_advisor_bot.bot.postgres_state import PostgresStateBackend
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.bot.user_state import UserPreferences, UserStateStore
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.providers.research_client import ResearchClient
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.services.recommendation_service import AssetScope, FallbackVerbosity, RecommendationService

BOT_SERVICES_KEY: Final[str] = "bot_services"
MAX_TELEGRAM_MESSAGE_LENGTH: Final[int] = 4000

CALLBACK_QUICK_SUMMARY: Final[str] = "quick:summary"
CALLBACK_GOLD: Final[str] = "quick:gold"
CALLBACK_US_STOCKS: Final[str] = "quick:us_stocks"
CALLBACK_ETF: Final[str] = "quick:etf"
CALLBACK_GLOBAL_TREND: Final[str] = "quick:global_trend"
CALLBACK_STOCK_IDEAS: Final[str] = "quick:stock_ideas"

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
    telegram_report_chat_id: str = ""
    min_request_interval_seconds: float = 2.0
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
    runtime_history_store: RuntimeHistoryStore | None = None
    backup_manager: BackupManager | None = None
    logs_dir: Path | None = None
    log_retention: str = "14 days"
    burn_in_target_days: int = 14


def register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("profile", profile_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
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
    application.add_handler(CommandHandler("market_update", market_update_command))
    application.add_handler(CallbackQueryHandler(handle_quick_action, pattern=r"^quick:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(handle_error)


async def set_bot_commands(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("start", "เริ่มใช้งานและเปิดเมนูหลัก"),
            BotCommand("help", "ดูวิธีใช้งานแบบย่อ"),
            BotCommand("profile", "ตั้งหรือดูโปรไฟล์นักลงทุน"),
            BotCommand("portfolio", "ดูพอร์ตที่บันทึกไว้"),
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
        "ถ้าต้องการเปลี่ยนโปรไฟล์ ใช้ /profile conservative, /profile balanced หรือ /profile growth\n\n"
        "ระบบสามารถสแกนตลาดอัตโนมัติและส่ง Morning / Midday / Closing reports,"
        " หุ้นเด่น, sector rotation, earnings และข่าวสำคัญเข้า report chat ได้เอง\n\n"
        "ตัวอย่างคำถาม\n"
        "- ตอนนี้หุ้นเมกาน่าเข้าหรือยัง\n"
        "- ตอนนี้ควรซื้อหุ้นอะไร 5 ตัว\n"
        "- ช่วยวิเคราะห์ ETF แบบละเอียดพร้อมเหตุผล\n"
        "- ขอแนวทางจัดพอร์ตแบบรักษาเงินต้น\n\n"
        "คำสั่งเสริม\n"
        "- /holdadd VOO 10 470\n"
        "- /portfolio\n"
        "- /watchadd AAPL\n"
        "- /watchlist\n"
        "- /prefs sectors=Technology,Healthcare threshold=2.1\n"
        "- /report_now closing\n"
        "- /scorecard\n"
        "- /dashboard"
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
            "/start เปิดเมนูหลัก\n/help ดูวิธีใช้งาน\n/profile ดูหรือตั้งโปรไฟล์นักลงทุน\n/portfolio ดูพอร์ตที่บันทึกไว้\n/holdadd เพิ่มหรืออัปเดต holding\n/holdremove ลบ holding\n/watchlist ดู watchlist\n/watchadd เพิ่มหุ้นเข้ารายการติดตาม\n/watchremove ลบหุ้นออกจาก watchlist\n/prefs ดูหรือปรับ preferences\n/report_now ขอรายงานล่าสุดทันที\n/scorecard ดูผลหุ้นที่ระบบเคยแนะนำ\n/dashboard ดู burn-in และ evaluation dashboard\n/backup_now สร้าง backup ทันที\n/market_update สรุปตลาดและแนวทางจัดพอร์ตล่าสุด\n\n"
            "ระบบอัตโนมัติ\n- Morning / Midday / Closing reports\n- Stock pick alerts\n- Sector rotation alerts\n- Earnings calendar alerts\n\n"
            "โปรไฟล์ที่รองรับ\n- conservative: รักษาเงินต้น\n- balanced: สมดุล\n- growth: เติบโต\n\n"
            "ตัวอย่างคำถาม\n- วิเคราะห์ทองคำแบบรักษาเงินต้น\n- ขอพอร์ต ETF สำหรับคนรับความเสี่ยงปานกลาง\n- ช่วยสรุปหุ้นสหรัฐแบบสั้นมาก"
            "\n- ตอนนี้ควรซื้อหุ้นอะไร 5 ตัว\n- วิเคราะห์ AAPL และ MSFT ให้หน่อย"
        ),
    )


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
                "ตั้งค่าใหม่ได้ด้วย\n- /profile conservative\n- /profile balanced\n- /profile growth"
            ),
        )
        return

    profile_arg = context.args[0].strip().casefold()
    if profile_arg not in {"conservative", "balanced", "growth"}:
        await _reply_text(message, "โปรไฟล์ที่รองรับคือ conservative, balanced และ growth")
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
        f"Watchlist: {watchlist_text}\nPreferred sectors: {sectors_text}\nStock alert threshold: {prefs.stock_alert_threshold:.1f}\nDaily pick enabled: {prefs.daily_pick_enabled}",
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
            f"Preferred sectors: {', '.join(prefs.preferred_sectors) if prefs.preferred_sectors else 'ทุก sector'}\nStock alert threshold: {prefs.stock_alert_threshold:.1f}\nDaily pick enabled: {prefs.daily_pick_enabled}\n\nตัวอย่าง: /prefs sectors=Technology,Healthcare threshold=2.0 daily=true",
        )
        return

    sectors: list[str] | None = None
    threshold: float | None = None
    daily_pick_enabled: bool | None = None
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
    prefs = services.user_state_store.update_preferences(
        conversation_key,
        preferred_sectors=sectors,
        stock_alert_threshold=threshold,
        daily_pick_enabled=daily_pick_enabled,
    )
    await _reply_text(
        message,
        f"อัปเดต preferences แล้ว\nPreferred sectors: {', '.join(prefs.preferred_sectors) if prefs.preferred_sectors else 'ทุก sector'}\nStock alert threshold: {prefs.stock_alert_threshold:.1f}\nDaily pick enabled: {prefs.daily_pick_enabled}",
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
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="report_now",
        question=f"/report_now {report_kind}",
        response_text=result.recommendation_text,
        fallback_used=result.fallback_used,
        model=result.model,
    )
    if services.runtime_history_store is not None and update.effective_chat is not None:
        services.runtime_history_store.record_sent_report(
            report_kind=report_kind,
            chat_id=str(update.effective_chat.id),
            fallback_used=result.fallback_used,
            model=result.model,
            summary=result.recommendation_text[:240],
            detail={"source": "manual_command"},
        )
    await _reply_text(message, result.recommendation_text)


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
        if return_pct is not None:
            lines.append(
                f"- {ticker}: {status} | return {float(return_pct):+.1%} | confidence {confidence_label} ({confidence_score}) | score {composite_score}"
            )
        else:
            due_at = str(detail.get("due_at") or row.get("created_at") or "-")
            lines.append(
                f"- {ticker}: {status} | confidence {confidence_label} ({confidence_score}) | score {composite_score} | due {due_at}"
            )
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

    history_snapshot = services.runtime_history_store.build_evaluation_dashboard(
        chat_id=str(update.effective_chat.id) if update.effective_chat is not None else None,
        lookback_days=max(7, services.burn_in_target_days),
        burn_in_target_days=services.burn_in_target_days,
    )
    live_snapshot = diagnostics.snapshot()
    burn_in = history_snapshot.get("burn_in") or {}
    scorecard = history_snapshot.get("scorecard") or {}
    reports = history_snapshot.get("reports") or {}
    interactions = history_snapshot.get("interactions") or {}
    alerts = history_snapshot.get("alerts") or {}
    providers = history_snapshot.get("providers") or []
    jobs = history_snapshot.get("jobs") or []
    live_latest = live_snapshot.get("latest_provider_success") or {}
    live_response = live_snapshot.get("response_stats") or {}
    average_return = scorecard.get("avg_return_pct")
    average_return_text = f"{average_return}%" if average_return is not None else "-"
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
            f"hit rate={scorecard.get('hit_rate_pct', 0)}% | avg return={average_return_text}"
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
    ]
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
    snapshot = diagnostics.snapshot()
    latest = snapshot.get("latest_provider_success") or {}
    response_stats = snapshot.get("response_stats") or {}
    jobs = snapshot.get("jobs") or {}
    alerts_today = snapshot.get("alerts_today") or {}
    db_state = snapshot.get("db_state") or {}
    circuit = snapshot.get("provider_circuit") or {}

    lines = [
        "Runtime Status",
        f"DB: {db_state.get('backend')} | healthy={db_state.get('healthy')} | checked_at={db_state.get('checked_at') or '-'}",
    ]
    if db_state.get("error"):
        lines.append(f"DB error: {db_state.get('error')}")
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
    _record_interaction(
        services,
        conversation_key=_conversation_key(update),
        interaction_kind="market_update",
        question="/market_update",
        response_text=result.recommendation_text,
        fallback_used=result.fallback_used,
        model=result.model,
    )
    await _reply_text(message, result.recommendation_text)


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
    )
    _record_stock_pick_scorecards_for_result(
        services,
        chat_id=str(update.effective_chat.id) if update.effective_chat is not None else None,
        source_kind="quick_action",
        result_payload=result.input_payload,
    )
    await _reply_text(query.message, result.recommendation_text)


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
    )
    _record_stock_pick_scorecards_for_result(
        services,
        chat_id=str(update.effective_chat.id) if update.effective_chat is not None else None,
        source_kind="chat",
        result_payload=result.input_payload,
    )
    await _reply_text(message, result.recommendation_text)


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


def _record_interaction(
    services: BotServices,
    *,
    conversation_key: str | None,
    interaction_kind: str,
    question: str | None,
    response_text: str,
    fallback_used: bool,
    model: str | None,
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
        detail={},
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
            detail={"due_at": due_at.isoformat()},
        )


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
