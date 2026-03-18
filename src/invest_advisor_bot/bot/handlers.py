from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from loguru import logger
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.services.recommendation_service import (
    AssetScope,
    FallbackVerbosity,
    RecommendationService,
)

BOT_SERVICES_KEY: Final[str] = "bot_services"
MAX_TELEGRAM_MESSAGE_LENGTH: Final[int] = 4000
CALLBACK_MACRO: Final[str] = "quick:macro"
CALLBACK_ALLOCATION: Final[str] = "quick:allocation"
CALLBACK_RISK: Final[str] = "quick:risk"
CALLBACK_OUTLOOK: Final[str] = "quick:outlook"

QuickAction = tuple[str, AssetScope, FallbackVerbosity | None]

QUICK_ACTION_QUESTIONS: Final[dict[str, QuickAction]] = {
    CALLBACK_MACRO: ("ช่วยสรุปภาพรวมเศรษฐกิจมหภาค (Macro Economic Update) และข่าวกระทบการลงทุน", "all", None),
    CALLBACK_ALLOCATION: ("ช่วยแนะนำกลยุทธ์การจัดสรรสินทรัพย์ (Global Asset Allocation Strategy) ในภาพรวม", "all", None),
    CALLBACK_RISK: ("ช่วยประเมินความเสี่ยงและจุดที่ต้องระวังเพื่อรักษาเงินต้น (Risk Assessment)", "all", None),
    CALLBACK_OUTLOOK: ("ช่วยสรุปมุมมองทองคำและดัชนีหุ้นต่างประเทศระยะยาว (Gold & Equity Outlook)", "all", None),
}


@dataclass(slots=True, frozen=True)
class BotServices:
    recommendation_service: RecommendationService
    market_data_client: MarketDataClient
    news_client: NewsClient
    market_news_limit: int
    market_history_period: str
    market_history_interval: str
    market_history_limit: int
    telegram_report_chat_id: str = ""


def register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("market_update", market_update_command))
    application.add_handler(CallbackQueryHandler(handle_quick_action, pattern=r"^quick:"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(handle_error)


async def set_bot_commands(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("start", "แนะนำการใช้งานบอท"),
            BotCommand("market_update", "สรุปตลาดโลกและคำแนะนำล่าสุด"),
        ]
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    text = (
        "สวัสดีครับ ผมคือ AI Wealth Manager บน Telegram\n"
        "ผมช่วยวิเคราะห์เศรษฐกิจมหภาค การจัดสรรสินทรัพย์ และการประเมินความเสี่ยงเพื่อพอร์ตเติบโตระยะยาว\n\n"
        "เลือกเมนูด้านล่าง หรือพิมพ์คำถามได้โดยตรง เช่น\n"
        "\"มุมมองการจัดพอร์ตราสารหนี้ช่วงนี้เป็นอย่างไร?\"\n"
        "\"ความเสี่ยงที่ต้องกังวลสูงสุดในครึ่งปีนี้คืออะไร?\""
    )
    await _reply_text(message, text, reply_markup=_build_main_menu())


async def market_update_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    services = _get_services(context)
    await _send_typing(update, context)
    await message.reply_text("กำลังสรุปภาพรวมตลาดและคำแนะนำล่าสุด...")

    result = await services.recommendation_service.generate_market_update(
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
    )
    await _reply_text(message, result.recommendation_text)


async def handle_quick_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return

    callback_data = query.data or ""
    quick_action = QUICK_ACTION_QUESTIONS.get(callback_data)
    await query.answer()
    if not quick_action:
        await query.message.reply_text("ไม่พบเมนูที่เลือก กรุณาลองใหม่อีกครั้ง")
        return
    question, asset_scope, fallback_verbosity = quick_action

    services = _get_services(context)
    await _send_typing(update, context)
    await query.message.reply_text("กำลังวิเคราะห์ข้อมูลล่าสุด...")

    result = await services.recommendation_service.answer_user_question(
        question=question,
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
        conversation_key=_conversation_key(update),
        asset_scope=asset_scope,
        fallback_verbosity_override=fallback_verbosity,
    )
    await _reply_text(query.message, result.recommendation_text)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or message.text is None:
        return

    user_question = message.text.strip()
    if not user_question:
        return

    services = _get_services(context)
    await _send_typing(update, context)
    await message.reply_text("กำลังวิเคราะห์คำถามของคุณ...")

    result = await services.recommendation_service.answer_user_question(
        question=user_question,
        news_client=services.news_client,
        market_data_client=services.market_data_client,
        news_limit=services.market_news_limit,
        history_period=services.market_history_period,
        history_interval=services.market_history_interval,
        history_limit=services.market_history_limit,
        conversation_key=_conversation_key(update),
    )
    await _reply_text(message, result.recommendation_text)


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram handler error: {}", context.error)
    if isinstance(update, Update) and update.effective_message is not None:
        await update.effective_message.reply_text(
            "เกิดข้อผิดพลาดระหว่างประมวลผลคำขอ กรุณาลองใหม่อีกครั้ง"
        )


def _get_services(context: ContextTypes.DEFAULT_TYPE) -> BotServices:
    services = context.application.bot_data.get(BOT_SERVICES_KEY)
    if not isinstance(services, BotServices):
        raise RuntimeError("Telegram bot services are not configured")
    return services


def _build_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🌍 Macro Update", callback_data=CALLBACK_MACRO),
                InlineKeyboardButton("⚖️ Asset Allocation", callback_data=CALLBACK_ALLOCATION),
            ],
            [
                InlineKeyboardButton("🛡️ Risk Assessment", callback_data=CALLBACK_RISK),
                InlineKeyboardButton("📈 Gold & Equity Outlook", callback_data=CALLBACK_OUTLOOK),
            ],
        ]
    )


def _conversation_key(update: Update) -> str | None:
    chat = update.effective_chat
    if chat is None:
        return None
    return str(chat.id)


async def _send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None:
        return
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)


async def _reply_text(
    message: Message,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
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
