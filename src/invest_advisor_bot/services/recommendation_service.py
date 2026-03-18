from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from loguru import logger

from invest_advisor_bot.analysis.trend_engine import TrendAssessment, evaluate_trend
from invest_advisor_bot.providers.llm_client import OpenAILLMClient
from invest_advisor_bot.providers.market_data_client import AssetQuote, MarketDataClient, OhlcvBar
from invest_advisor_bot.providers.news_client import NewsArticle, NewsClient

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "system_investment_advisor.txt"


@dataclass(slots=True, frozen=True)
class RecommendationResult:
    recommendation_text: str
    model: str | None
    system_prompt_path: str
    input_payload: dict[str, Any]
    response_id: str | None = None
    fallback_used: bool = False


class RecommendationService:
    """Combines news, market data, and trend analysis into an LLM recommendation."""

    def __init__(
        self,
        llm_client: OpenAILLMClient,
        *,
        system_prompt_path: Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt_path = Path(system_prompt_path or DEFAULT_PROMPT_PATH)

    async def generate_recommendation(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
    ) -> RecommendationResult:
        system_prompt = self._load_system_prompt()
        payload = self._build_payload(news=news, market_data=market_data, trends=trends)
        user_prompt = self._build_user_prompt(payload)

        llm_response = await self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={"service": "recommendation_service", "language": "th"},
        )

        if llm_response is None:
            logger.warning("LLM recommendation generation failed; using fallback summary")
            return RecommendationResult(
                recommendation_text=self._build_fallback_summary(payload),
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload=payload,
                response_id=None,
                fallback_used=True,
            )

        return RecommendationResult(
            recommendation_text=llm_response.text,
            model=llm_response.model,
            system_prompt_path=str(self.system_prompt_path),
            input_payload=payload,
            response_id=llm_response.response_id,
            fallback_used=False,
        )

    async def generate_market_update(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        news_limit: int = 8,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
    ) -> RecommendationResult:
        news, market_data, trends = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
        )
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
        )

    async def answer_user_question(
        self,
        *,
        question: str,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        news_limit: int = 8,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
    ) -> RecommendationResult:
        normalized_question = question.strip()
        if not normalized_question:
            return RecommendationResult(
                recommendation_text="กรุณาพิมพ์คำถามเกี่ยวกับการลงทุนที่ต้องการให้วิเคราะห์",
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload={},
                response_id=None,
                fallback_used=True,
            )

        system_prompt = self._load_system_prompt()
        news, market_data, trends = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
        )
        payload = self._build_payload(news=news, market_data=market_data, trends=trends)
        user_prompt = self._build_question_prompt(question=normalized_question, payload=payload)

        llm_response = await self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={"service": "recommendation_service", "mode": "chat", "language": "th"},
        )
        if llm_response is None:
            return RecommendationResult(
                recommendation_text=self._build_fallback_question_answer(
                    question=normalized_question,
                    payload=payload,
                ),
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload=payload,
                response_id=None,
                fallback_used=True,
            )

        return RecommendationResult(
            recommendation_text=llm_response.text,
            model=llm_response.model,
            system_prompt_path=str(self.system_prompt_path),
            input_payload=payload,
            response_id=llm_response.response_id,
            fallback_used=False,
        )

    def _load_system_prompt(self) -> str:
        try:
            return self.system_prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to load system prompt from {}: {}", self.system_prompt_path, exc)
            return (
                "คุณคือที่ปรึกษาการลงทุนระดับโลก วิเคราะห์ข้อมูลข่าว ตลาด และเทรนด์ทางเทคนิค "
                "แล้วสรุปคำแนะนำเป็นภาษาไทยอย่างชัดเจน ระบุ ซื้อ, ขาย หรือ 관망 (Wait and see) "
                "พร้อมเหตุผลสำคัญ ความเสี่ยง และย้ำว่าไม่ใช่การรับประกันผลตอบแทน"
            )

    def _build_payload(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
    ) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "news": [self._serialize_news_article(article) for article in news[:8]],
            "market_data": {
                asset_name: self._serialize_market_quote(asset_name, quote)
                for asset_name, quote in market_data.items()
            },
            "trends": {
                asset_name: self._serialize_trend(trend)
                for asset_name, trend in trends.items()
            },
        }

    def _build_user_prompt(self, payload: Mapping[str, Any]) -> str:
        data_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            "วิเคราะห์ข้อมูลการลงทุนต่อไปนี้ แล้วสรุปผลเป็นภาษาไทยสำหรับผู้ใช้งาน Telegram\n\n"
            "รูปแบบคำตอบที่ต้องการ:\n"
            "1. ภาพรวมตลาดสั้น ๆ\n"
            "2. คำแนะนำรายสินทรัพย์ โดยใช้คำว่า ซื้อ, ขาย, หรือ 관망 (Wait and see)\n"
            "3. เหตุผลหลักที่สนับสนุนคำแนะนำ\n"
            "4. ความเสี่ยงหรือประเด็นที่ต้องติดตาม\n\n"
            "เงื่อนไขสำคัญ:\n"
            "- ใช้เฉพาะข้อมูลที่ให้มา\n"
            "- ถ้าสัญญาณขัดแย้งกัน ให้เอนเอียงไปทาง 관망 (Wait and see)\n"
            "- ห้ามเขียนเกินจริงหรือรับประกันผลตอบแทน\n"
            "- เขียนให้อ่านง่าย กระชับ แต่มีเหตุผล\n\n"
            f"ข้อมูลสำหรับวิเคราะห์:\n{data_json}"
        )

    def _build_question_prompt(self, *, question: str, payload: Mapping[str, Any]) -> str:
        data_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return (
            "ผู้ใช้ถามคำถามเกี่ยวกับการลงทุนดังนี้:\n"
            f"{question}\n\n"
            "จงตอบเป็นภาษาไทยโดยอ้างอิงจากข่าว ข้อมูลตลาด และแนวโน้มทางเทคนิคที่ให้มาเท่านั้น\n"
            "ถ้าคำตอบยังไม่ชัดเจน ให้ตอบแบบระมัดระวังและเสนอ 관망 (Wait and see)\n"
            "ให้สรุปแบบอ่านง่ายสำหรับการตอบกลับใน Telegram\n\n"
            f"บริบทตลาดปัจจุบัน:\n{data_json}"
        )

    def _build_fallback_summary(self, payload: Mapping[str, Any]) -> str:
        trend_items = payload.get("trends", {})
        if not isinstance(trend_items, Mapping) or not trend_items:
            return (
                "ยังไม่สามารถสร้างคำแนะนำจาก AI ได้ในขณะนี้ และข้อมูลแนวโน้มยังไม่เพียงพอ "
                "จึงแนะนำให้ 관망 (Wait and see) ไปก่อน"
            )

        lines = [
            "สรุปเบื้องต้นจากระบบสำรอง",
            "ภาพรวม: ใช้ผลวิเคราะห์เชิงเทคนิคและข่าวที่มีอยู่เพื่อให้คำแนะนำเบื้องต้นเท่านั้น",
        ]
        for asset_name, trend_data in trend_items.items():
            if not isinstance(trend_data, Mapping):
                continue
            direction = str(trend_data.get("direction") or "sideways")
            recommendation = {
                "uptrend": "ซื้อ",
                "downtrend": "ขาย",
                "sideways": "관망 (Wait and see)",
            }.get(direction, "관망 (Wait and see)")
            reasons = trend_data.get("reasons") or []
            reason_text = ", ".join(str(reason) for reason in reasons[:3]) if isinstance(reasons, list) else "-"
            lines.append(f"- {asset_name}: {recommendation} | แนวโน้ม {direction} | เหตุผล: {reason_text}")
        lines.append("หมายเหตุ: นี่เป็นคำแนะนำสำรองเมื่อ LLM ไม่พร้อมใช้งาน และไม่ใช่การรับประกันผลตอบแทน")
        return "\n".join(lines)

    def _build_fallback_question_answer(self, *, question: str, payload: Mapping[str, Any]) -> str:
        summary = self._build_fallback_summary(payload)
        return (
            f"คำถาม: {question}\n\n"
            "ขณะนี้ไม่สามารถเรียกใช้ LLM ได้ จึงสรุปจากข้อมูลแนวโน้มที่มีอยู่แทน:\n"
            f"{summary}"
        )

    @staticmethod
    def _serialize_news_article(article: NewsArticle) -> dict[str, Any]:
        return {
            "title": article.title,
            "source": article.source,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "summary": article.summary,
            "link": article.link,
        }

    @staticmethod
    def _serialize_market_quote(asset_name: str, quote: AssetQuote | None) -> dict[str, Any]:
        if quote is None:
            return {"asset_name": asset_name, "available": False}

        day_change_pct: float | None = None
        if quote.previous_close:
            day_change_pct = ((quote.price - quote.previous_close) / quote.previous_close) * 100.0

        return {
            "asset_name": asset_name,
            "available": True,
            "ticker": quote.ticker,
            "name": quote.name,
            "exchange": quote.exchange,
            "currency": quote.currency,
            "price": quote.price,
            "previous_close": quote.previous_close,
            "day_change_pct": day_change_pct,
            "open_price": quote.open_price,
            "day_high": quote.day_high,
            "day_low": quote.day_low,
            "volume": quote.volume,
            "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
        }

    def _serialize_trend(self, trend: TrendAssessment) -> dict[str, Any]:
        raw = self._serialize_value(trend)
        if not isinstance(raw, dict):
            return {"direction": "sideways"}
        return raw

    async def _gather_context(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        news_limit: int,
        history_period: str,
        history_interval: str,
        history_limit: int,
    ) -> tuple[list[NewsArticle], Mapping[str, AssetQuote | None], dict[str, TrendAssessment]]:
        news_task = news_client.fetch_latest_macro_news(limit=news_limit)
        market_snapshot_task = market_data_client.get_core_market_snapshot()
        market_history_task = market_data_client.get_core_market_history(
            period=history_period,
            interval=history_interval,
            limit=history_limit,
        )
        news, market_data, market_history = await asyncio.gather(
            news_task,
            market_snapshot_task,
            market_history_task,
        )
        trends = self._build_trends_from_history(market_history)
        return list(news), market_data, trends

    def _build_trends_from_history(
        self,
        history_by_asset: Mapping[str, Sequence[OhlcvBar]],
    ) -> dict[str, TrendAssessment]:
        trends: dict[str, TrendAssessment] = {}
        for asset_name, bars in history_by_asset.items():
            if not bars:
                continue
            frame = self._bars_to_frame(bars)
            if frame.empty:
                continue
            try:
                trends[asset_name] = evaluate_trend(
                    frame,
                    ticker=bars[0].ticker,
                )
            except Exception as exc:
                logger.warning("Failed to evaluate trend for {}: {}", asset_name, exc)
        return trends

    @staticmethod
    def _bars_to_frame(bars: Sequence[OhlcvBar]) -> pd.DataFrame:
        rows = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
        return frame.reset_index(drop=True)

    def _serialize_value(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._serialize_value(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): self._serialize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        return value
