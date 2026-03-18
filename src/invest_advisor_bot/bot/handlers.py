from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from loguru import logger
from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.services.recommendation_service import RecommendationService

BOT_SERVICES_KEY: Final[str] = "bot_services"
MAX_TELEGRAM_MESSAGE_LENGTH: Final[int] = 4000


@dataclass(slots=True, frozen=True)
class BotServices:
    recommendation_service: RecommendationService
    market_data_client: MarketDataClient
    news_client: NewsClient
    market_news_limit: int
    market_history_period: str
    market_history_interval: str
    market_history_limit: int


def register_handlers(application: Application) -> None:
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("market_update", market_update_command))
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
        "สวัสดีครับ ผมคือ AI Investment Advisor บน Telegram\n"
        "ผมช่วยสรุปภาพรวมตลาดโลก วิเคราะห์ทองคำ หุ้นสหรัฐฯ และ ETF จากข่าวมหภาค ข้อมูลตลาด และเทรนด์ทางเทคนิค\n\n"
        "คำสั่งที่ใช้ได้:\n"
        "/start - แนะนำการใช้งาน\n"
        "/market_update - สรุปตลาดและคำแนะนำล่าสุดทันที\n\n"
        "คุณยังสามารถพิมพ์คำถามธรรมดาได้ เช่น\n"
        "\"ตอนนี้หุ้นเมกาน่าเข้าไหม?\"\n"
        "\"ทองคำวันนี้ควรซื้อหรือรอก่อน?\""
    )
    await _reply_text(message, text)


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


async def _send_typing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is None:
        return
    await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)


async def _reply_text(message, text: str) -> None:
    chunks = _chunk_text(text, limit=MAX_TELEGRAM_MESSAGE_LENGTH)
    for chunk in chunks:
        await message.reply_text(chunk)


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
