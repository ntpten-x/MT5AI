from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
from loguru import logger

from invest_advisor_bot.analysis.trend_engine import TrendAssessment, evaluate_trend
from invest_advisor_bot.providers.llm_client import OpenAILLMClient
from invest_advisor_bot.providers.market_data_client import AssetQuote, MarketDataClient, OhlcvBar
from invest_advisor_bot.providers.news_client import NewsArticle, NewsClient

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "system_investment_advisor.txt"
DEFAULT_CHAT_HISTORY_LIMIT = 3
DEFAULT_NEWS_CONTEXT_LIMIT = 5
DEFAULT_REASON_LIMIT = 2
DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT = 3
AssetScope = Literal["all", "gold-only", "us-stocks", "etf-only", "bonds"]
FallbackVerbosity = Literal["short", "medium", "detailed"]

ASSET_SCOPE_MEMBERS: dict[AssetScope, tuple[str, ...]] = {
    "all": (),
    "gold-only": ("gold_futures", "gld_etf", "iau_etf"),
    "us-stocks": ("sp500_index", "nasdaq_index", "spy_etf", "qqq_etf", "vti_etf", "xlf_etf", "xle_etf", "xlk_etf", "voo_etf"),
    "etf-only": ("spy_etf", "qqq_etf", "gld_etf", "iau_etf", "vti_etf", "xlf_etf", "xle_etf", "xlk_etf", "tlt_etf", "voo_etf"),
    "bonds": ("tlt_etf",),
}


@dataclass(slots=True, frozen=True)
class RecommendationResult:
    recommendation_text: str
    model: str | None
    system_prompt_path: str
    input_payload: dict[str, Any]
    response_id: str | None = None
    fallback_used: bool = False


class RecommendationService:
    """Combines cached news, market data, and trend analysis into a compact LLM prompt."""

    def __init__(
        self,
        llm_client: OpenAILLMClient,
        *,
        system_prompt_path: Path | None = None,
        chat_history_limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt_path = Path(system_prompt_path or DEFAULT_PROMPT_PATH)
        self.chat_history_limit = max(1, int(chat_history_limit))
        self._conversation_history: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.chat_history_limit)
        )

    async def generate_recommendation(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        macro_context: dict[str, float | None] | None = None,
        question: str | None = None,
        conversation_key: str | None = None,
        asset_scope: AssetScope = "all",
        fallback_verbosity_override: FallbackVerbosity | None = None,
    ) -> RecommendationResult:
        system_prompt = self._load_system_prompt()
        payload = self._build_payload(
            news=news,
            market_data=market_data,
            trends=trends,
            asset_scope=asset_scope,
            question=question,
            macro_context=macro_context,
        )
        history_lines = self._get_history_lines(conversation_key)
        user_prompt = self._build_prompt(
            payload=payload,
            question=question,
            history_lines=history_lines,
        )

        llm_response = await self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={"service": "recommendation_service", "language": "th"},
        )

        if llm_response is None:
            logger.warning("LLM recommendation generation failed; using fallback summary")
            fallback_verbosity = fallback_verbosity_override or self._determine_fallback_verbosity(
                question=question,
                asset_scope=asset_scope,
            )
            fallback_text = (
                self._build_fallback_question_answer(
                    question=question,
                    payload=payload,
                    verbosity=fallback_verbosity,
                )
                if question
                else self._build_fallback_summary(payload, verbosity=fallback_verbosity)
            )
            self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=fallback_text)
            return RecommendationResult(
                recommendation_text=fallback_text,
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload=payload,
                response_id=None,
                fallback_used=True,
            )

        self._remember_turns(
            conversation_key=conversation_key,
            user_text=question,
            assistant_text=llm_response.text,
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
        news_limit: int = DEFAULT_NEWS_CONTEXT_LIMIT,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
        asset_scope: AssetScope = "all",
    ) -> RecommendationResult:
        news, market_data, trends, macro_context = await self._gather_context(
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
            macro_context=macro_context,
            question="สรุปภาพรวมตลาดโลกและคำแนะนำล่าสุดแบบกระชับ",
            asset_scope=asset_scope,
        )

    async def answer_user_question(
        self,
        *,
        question: str,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        news_limit: int = DEFAULT_NEWS_CONTEXT_LIMIT,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
        conversation_key: str | None = None,
        asset_scope: AssetScope | None = None,
        fallback_verbosity_override: FallbackVerbosity | None = None,
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

        effective_scope = asset_scope or self._detect_asset_scope(normalized_question)
        news, market_data, trends, macro_context = await self._gather_context(
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
            macro_context=macro_context,
            question=normalized_question,
            conversation_key=conversation_key,
            asset_scope=effective_scope,
            fallback_verbosity_override=fallback_verbosity_override,
        )

    def _load_system_prompt(self) -> str:
        try:
            return self.system_prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to load system prompt from {}: {}", self.system_prompt_path, exc)
            return (
                "คุณคือที่ปรึกษาการลงทุนระดับโลก วิเคราะห์ข่าวมหภาค ภาพรวมตลาด และสัญญาณทางเทคนิค "
                "แล้วสรุปคำแนะนำเป็นภาษาไทยอย่างกระชับ ชัดเจน และไม่รับประกันผลตอบแทน"
            )

    def _build_payload(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        asset_scope: AssetScope,
        question: str | None,
        macro_context: dict[str, float | None] | None = None,
    ) -> dict[str, Any]:
        filtered_market_data, filtered_trends = self._filter_asset_context(
            market_data=market_data,
            trends=trends,
            asset_scope=asset_scope,
        )
        scoped_news_limit = DEFAULT_NEWS_CONTEXT_LIMIT if asset_scope == "all" else DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT
        compact_news = [
            self._serialize_news_article(article)
            for article in list(news)[:scoped_news_limit]
        ]
        compact_assets = [
            self._serialize_asset_context(
                asset_name=asset_name,
                quote=filtered_market_data.get(asset_name),
                trend=filtered_trends.get(asset_name),
            )
            for asset_name in filtered_market_data.keys()
        ]
        compact_assets = [asset for asset in compact_assets if asset is not None]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": asset_scope,
            "question": question,
            "news_headlines": compact_news,
            "asset_snapshots": compact_assets,
            "macro_context": macro_context or {},
        }

    def _build_prompt(
        self,
        *,
        payload: Mapping[str, Any],
        question: str | None,
        history_lines: Sequence[str],
    ) -> str:
        news_lines = self._format_news_lines(payload.get("news_headlines"))
        asset_lines = self._format_asset_lines(payload.get("asset_snapshots"))
        macro_context = payload.get("macro_context") or {}
        macro_text = (
            f"- VIX Index (ความกลัวตลาด): {macro_context.get('vix', 'N/A')}\n"
            f"- ผลตอบแทนพันธบัตร 10 ปี (TNX): {macro_context.get('tnx', 'N/A')}%\n"
            f"- เงินเฟ้อสหรัฐฯ (CPI YoY): {macro_context.get('cpi_yoy', 'N/A')}%"
        )
        history_text = "\n".join(history_lines) if history_lines else "ไม่มี"

        intro = (
            f"คำถามผู้ใช้: {question}\n"
            if question
            else "คำสั่งผู้ใช้: สรุปภาพรวมตลาดโลกและคำแนะนำล่าสุด\n"
        )
        return (
            f"{intro}"
            "บริบทบทสนทนาล่าสุด:\n"
            f"{history_text}\n\n"
            "ดัชนีชี้วัดเศรษฐกิจมหภาค (Macro Indicators):\n"
            f"{macro_text}\n\n"
            "ข่าวสำคัญล่าสุด:\n"
            f"{news_lines}\n\n"
            "สรุปสินทรัพย์สำคัญ:\n"
            f"{asset_lines}\n\n"
            "ตอบเป็นภาษาไทยแบบจัดสรรพอร์ตความมั่งคั่งสำหรับ Telegram\n"
            "ต้องมี:\n"
            "1. ภาพรวมเศรษฐกิจและตลาด\n"
            "2. มุมมองจัดสรรน้ำหนักการลงทุน โดยใช้ Overweight / Neutral / Underweight\n"
            "3. [Worst-case Scenario] คาดการณ์ % Drawdown กรณีเกิดวิกฤต/สงคราม ของพอร์ตอ้างอิง (เช่น VOO+Gold 50/50) พร้อมเงื่อนไขจุดถอยหนีความเสี่ยง (Trigger to Cash)\n"
            "4. เหตุผลหลักที่อ้างอิงจากข่าว Macro และแนวโน้มระยะยาว\n"
            "5. ความเสี่ยงสูงสุดหรือปัจจัยที่ต้องเฝ้าระวัง\n"
            "ถ้าแนวโน้มมีความก้ำกึ่ง ไม่ชัดเจนให้เอนเอียงไปทาง Neutral"
        )

    def _build_fallback_summary(
        self,
        payload: Mapping[str, Any],
        *,
        verbosity: FallbackVerbosity = "medium",
    ) -> str:
        asset_snapshots = payload.get("asset_snapshots", [])
        scope = str(payload.get("scope") or "all")
        if not isinstance(asset_snapshots, list) or not asset_snapshots:
            return "ยังมีข้อมูลไม่พอสำหรับสรุปคำแนะนำ จึงแนะนำให้ 관망 (Wait and see) ไปก่อน"

        market_view = self._build_market_overview(asset_snapshots, scope)
        if verbosity == "short":
            lines = ["สรุปย่อจากระบบสำรอง", market_view, ""]
            for asset in asset_snapshots:
                if not isinstance(asset, Mapping):
                    continue
                recommendation = self._direction_to_recommendation(str(asset.get("trend") or "sideways"))
                lines.append(
                    f"{self._asset_display_name(asset.get('asset'))}: {recommendation} | "
                    f"RSI {asset.get('rsi')} | แนวรับ {asset.get('support')} | แนวต้าน {asset.get('resistance')}"
                )
            lines.append("หมายเหตุ: ใช้สรุปสำรองเมื่อ LLM ไม่พร้อมใช้งาน")
            return "\n".join(lines)
        lines = ["สรุปเบื้องต้นจากระบบสำรอง", market_view, ""]
        if verbosity == "detailed":
            news_lines = self._format_fallback_news(payload.get("news_headlines"))
            if news_lines:
                lines.extend(["ข่าวที่ต้องติดตาม:", news_lines, ""])
        for asset in asset_snapshots:
            if not isinstance(asset, Mapping):
                continue
            recommendation = self._direction_to_recommendation(str(asset.get("trend") or "sideways"))
            reasons = self._humanize_signals(asset.get("signals"))
            risk_note = self._build_risk_note(asset)
            if verbosity == "detailed":
                lines.append(
                    f"{self._asset_display_name(asset.get('asset'))}: {recommendation}\n"
                    f"ราคา/การเปลี่ยนแปลง: {asset.get('price')} | {asset.get('day_change_pct')}%\n"
                    f"โมเมนตัม: trend={asset.get('trend')} | score={asset.get('trend_score')} | "
                    f"RSI={asset.get('rsi')} | MACD_hist={asset.get('macd_hist')}\n"
                    f"เหตุผลหลัก: {reasons}\n"
                    f"ระดับสำคัญ: แนวรับ {asset.get('support')} | แนวต้าน {asset.get('resistance')}\n"
                    f"สิ่งที่ต้องติดตาม: {risk_note}\n"
                )
                continue
            lines.append(
                f"{self._asset_display_name(asset.get('asset'))}: {recommendation}\n"
                f"เหตุผลหลัก: {reasons}\n"
                f"ระดับสำคัญ: แนวรับ {asset.get('support')} | แนวต้าน {asset.get('resistance')}\n"
                f"สิ่งที่ต้องติดตาม: {risk_note}\n"
            )
        lines.append("หมายเหตุ: เป็นสรุปสำรองเมื่อ LLM ไม่พร้อมใช้งาน และใช้เพื่อประกอบการตัดสินใจเท่านั้น")
        return "\n".join(lines)

    def _build_fallback_question_answer(
        self,
        *,
        question: str | None,
        payload: Mapping[str, Any],
        verbosity: FallbackVerbosity,
    ) -> str:
        summary = self._build_fallback_summary(payload, verbosity=verbosity)
        if not question:
            return summary
        if verbosity == "short":
            return f"คำถาม: {question}\n\n{summary}"
        return (
            f"คำถาม: {question}\n\n"
            "ขณะนี้ LLM ไม่พร้อมใช้งาน จึงสรุปจากข้อมูลล่าสุดที่มีอยู่ในระบบแทน:\n\n"
            f"{summary}"
        )

    def _serialize_news_article(self, article: NewsArticle) -> dict[str, Any]:
        return {
            "title": self._truncate_text(article.title, 110),
            "source": article.source,
            "published_at": article.published_at.isoformat() if article.published_at else None,
        }

    def _serialize_asset_context(
        self,
        *,
        asset_name: str,
        quote: AssetQuote | None,
        trend: TrendAssessment | None,
    ) -> dict[str, Any] | None:
        if quote is None and trend is None:
            return None

        day_change_pct: float | None = None
        if quote is not None and quote.previous_close:
            day_change_pct = round(((quote.price - quote.previous_close) / quote.previous_close) * 100.0, 2)

        support = None
        resistance = None
        trend_direction = "sideways"
        trend_score = None
        rsi = None
        ema_gap_pct = None
        macd_hist = None
        signals: list[str] = []
        if trend is not None:
            support = self._round_optional(trend.support_resistance.nearest_support)
            resistance = self._round_optional(trend.support_resistance.nearest_resistance)
            trend_direction = trend.direction
            trend_score = round(trend.score, 2)
            rsi = self._round_optional(trend.rsi)
            ema_gap_pct = self._round_optional((trend.ema_gap_pct or 0.0) * 100.0, digits=2)
            macd_hist = self._round_optional(trend.macd_hist)
            signals = list(trend.reasons[:DEFAULT_REASON_LIMIT])

        return {
            "asset": asset_name,
            "ticker": quote.ticker if quote else trend.ticker if trend else None,
            "price": self._round_optional(quote.price if quote else None),
            "day_change_pct": day_change_pct,
            "trend": trend_direction,
            "trend_score": trend_score,
            "rsi": rsi,
            "ema_gap_pct": ema_gap_pct,
            "macd_hist": macd_hist,
            "support": support,
            "resistance": resistance,
            "signals": signals,
        }

    def _filter_asset_context(
        self,
        *,
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        asset_scope: AssetScope,
    ) -> tuple[dict[str, AssetQuote | None], dict[str, TrendAssessment]]:
        scope_members = ASSET_SCOPE_MEMBERS.get(asset_scope, ())
        if not scope_members:
            return dict(market_data), dict(trends)
        filtered_market = {asset: market_data.get(asset) for asset in scope_members if asset in market_data}
        filtered_trends = {asset: trends.get(asset) for asset in scope_members if asset in trends}
        if filtered_market:
            return filtered_market, filtered_trends
        return dict(market_data), dict(trends)

    async def _gather_context(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        news_limit: int,
        history_period: str,
        history_interval: str,
        history_limit: int,
    ) -> tuple[list[NewsArticle], Mapping[str, AssetQuote | None], dict[str, TrendAssessment], dict[str, float | None]]:
        news_task = news_client.fetch_latest_macro_news(limit=min(news_limit, DEFAULT_NEWS_CONTEXT_LIMIT))
        market_snapshot_task = market_data_client.get_core_market_snapshot()
        market_history_task = market_data_client.get_core_market_history(
            period=history_period,
            interval=history_interval,
            limit=history_limit,
        )
        macro_task = market_data_client.get_macro_context()
        news, market_data, market_history, macro_context = await asyncio.gather(
            news_task,
            market_snapshot_task,
            market_history_task,
            macro_task,
        )
        trends = self._build_trends_from_history(market_history)
        return list(news), market_data, trends, macro_context

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
                trends[asset_name] = evaluate_trend(frame, ticker=bars[0].ticker)
            except Exception as exc:
                logger.warning("Failed to evaluate trend for {}: {}", asset_name, exc)
        return trends

    def _get_history_lines(self, conversation_key: str | None) -> list[str]:
        if not conversation_key:
            return []
        history = self._conversation_history.get(conversation_key)
        if not history:
            return []
        return [f"{item['role']}: {item['text']}" for item in history]

    def _remember_turns(
        self,
        *,
        conversation_key: str | None,
        user_text: str | None,
        assistant_text: str | None,
    ) -> None:
        if not conversation_key:
            return
        history = self._conversation_history[conversation_key]
        if user_text:
            history.append({"role": "user", "text": self._truncate_text(user_text, 180)})
        if assistant_text:
            history.append({"role": "assistant", "text": self._truncate_text(assistant_text, 220)})

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

    def _format_news_lines(self, news_items: Any) -> str:
        if not isinstance(news_items, list) or not news_items:
            return "- ไม่มี headline สำคัญ"
        lines: list[str] = []
        for item in news_items[:DEFAULT_NEWS_CONTEXT_LIMIT]:
            if not isinstance(item, Mapping):
                continue
            title = item.get("title") or "-"
            source = item.get("source") or "Unknown"
            lines.append(f"- {title} ({source})")
        return "\n".join(lines) if lines else "- ไม่มี headline สำคัญ"

    def _format_asset_lines(self, asset_items: Any) -> str:
        if not isinstance(asset_items, list) or not asset_items:
            return "- ไม่มี snapshot สินทรัพย์"
        lines: list[str] = []
        for item in asset_items:
            if not isinstance(item, Mapping):
                continue
            line = (
                f"- {item.get('asset')} [{item.get('ticker')}]: "
                f"price={item.get('price')}, change={item.get('day_change_pct')}%, "
                f"trend={item.get('trend')}, score={item.get('trend_score')}, "
                f"RSI={item.get('rsi')}, MACD_hist={item.get('macd_hist')}, "
                f"support={item.get('support')}, resistance={item.get('resistance')}, "
                f"signals={','.join(item.get('signals') or []) or '-'}"
            )
            lines.append(line)
        return "\n".join(lines) if lines else "- ไม่มี snapshot สินทรัพย์"

    @staticmethod
    def _direction_to_recommendation(direction: str) -> str:
        return {
            "uptrend": "เพิ่มน้ำหนักการลงทุน (Overweight)",
            "downtrend": "ลดน้ำหนักการลงทุน (Underweight)",
            "sideways": "คงน้ำหนักการลงทุน (Neutral)",
        }.get(direction, "คงน้ำหนักการลงทุน (Neutral)")

    @staticmethod
    def _detect_asset_scope(question: str) -> AssetScope:
        normalized = question.lower()
        thai_gold = "\u0e17\u0e2d\u0e07"
        thai_us_stock = "\u0e2b\u0e38\u0e49\u0e19\u0e2a\u0e2b\u0e23\u0e31\u0e10"
        thai_us_stock_alt = "\u0e2b\u0e38\u0e49\u0e19\u0e40\u0e21\u0e01\u0e32"
        if any(keyword in normalized for keyword in (thai_gold, "gold", "xau", "gld", "iau")):
            return "gold-only"
        if "etf" in normalized:
            return "etf-only"
        if any(
            keyword in normalized
            for keyword in (thai_us_stock, thai_us_stock_alt, "us stock", "nasdaq", "s&p", "spy", "qqq")
        ):
            return "us-stocks"
        return "all"

    @staticmethod
    def _asset_display_name(asset_name: object) -> str:
        return {
            "gold_futures": "ทองคำ",
            "sp500_index": "ดัชนี S&P 500",
            "nasdaq_index": "ดัชนี NASDAQ",
            "spy_etf": "ETF SPY (S&P 500)",
            "qqq_etf": "ETF QQQ (NASDAQ)",
            "gld_etf": "ETF GLD (Gold)",
            "iau_etf": "ETF IAU (Gold)",
            "vti_etf": "ETF VTI (Total Stock)",
            "xlf_etf": "ETF XLF (Financial)",
            "xle_etf": "ETF XLE (Energy)",
            "xlk_etf": "ETF XLK (Tech)",
            "tlt_etf": "ETF TLT (20+ Yr Bond)",
            "voo_etf": "ETF VOO (S&P 500)",
        }.get(str(asset_name), str(asset_name))

    def _humanize_signals(self, signals: object) -> str:
        if not isinstance(signals, list) or not signals:
            return "สัญญาณยังไม่เด่นชัด จึงควรรอจังหวะที่ชัดขึ้น"
        phrase_map = {
            "price_above_fast_and_slow_ema": "ราคายืนเหนือเส้น EMA หลัก",
            "price_below_fast_and_slow_ema": "ราคายังอยู่ใต้เส้น EMA หลัก",
            "price_mixed_vs_ema": "ราคาและเส้น EMA ยังไม่เรียงตัวชัดเจน",
            "ema_spread_bullish": "โครงสร้าง EMA ยังเอนเอียงเชิงบวก",
            "ema_spread_bearish": "โครงสร้าง EMA ยังเอนเอียงเชิงลบ",
            "ema_spread_flat": "แรงส่งของ EMA ยังไม่เด่น",
            "rsi_above_55": "RSI อยู่ฝั่งบวก",
            "rsi_below_45": "RSI อยู่ฝั่งอ่อนแรง",
            "rsi_neutral": "RSI ยังเป็นกลาง",
            "macd_bullish": "MACD ยังสนับสนุนฝั่งบวก",
            "macd_bearish": "MACD ยังสนับสนุนฝั่งลบ",
            "macd_neutral": "MACD ยังไม่ยืนยันทางใดทางหนึ่ง",
        }
        translated = [phrase_map.get(str(signal), str(signal)) for signal in signals[:DEFAULT_REASON_LIMIT]]
        return " และ ".join(translated)

    def _build_market_overview(self, asset_snapshots: list[Any], scope: str) -> str:
        trend_counts = {"uptrend": 0, "downtrend": 0, "sideways": 0}
        for asset in asset_snapshots:
            if not isinstance(asset, Mapping):
                continue
            trend_counts[str(asset.get("trend") or "sideways")] = trend_counts.get(
                str(asset.get("trend") or "sideways"),
                0,
            ) + 1
        if trend_counts["downtrend"] > trend_counts["uptrend"]:
            tone = "ภาพรวมยังค่อนข้างระวังความเสี่ยง เพราะสินทรัพย์ส่วนใหญ่ยังอยู่ในโหมดอ่อนแรง"
        elif trend_counts["uptrend"] > trend_counts["downtrend"]:
            tone = "ภาพรวมยังมีแรงบวกพอสมควร เพราะสินทรัพย์ส่วนใหญ่ยังรักษาแนวโน้มขาขึ้นได้"
        else:
            tone = "ภาพรวมยังค่อนข้างผสมและยังไม่มีฝั่งใดชนะอย่างชัดเจน"

        scope_text = {
            "gold-only": "โฟกัสเฉพาะกลุ่มทองคำ",
            "us-stocks": "โฟกัสเฉพาะหุ้นสหรัฐและดัชนีหลัก",
            "etf-only": "โฟกัสเฉพาะกลุ่ม ETF",
            "all": "โฟกัสภาพรวมตลาดหลัก",
        }.get(scope, "โฟกัสภาพรวมตลาด")
        return f"ภาพรวม: {scope_text} | {tone}"

    def _build_risk_note(self, asset: Mapping[str, Any]) -> str:
        trend = str(asset.get("trend") or "sideways")
        rsi = asset.get("rsi")
        if trend == "sideways":
            return "แนวโน้มยังไม่ชัด หากราคาไม่ผ่านแนวต้านหรือหลุดแนวรับอาจแกว่งออกข้างต่อ"
        if trend == "uptrend" and isinstance(rsi, (float, int)) and float(rsi) >= 68:
            return "RSI เริ่มสูง ควรระวังแรงขายทำกำไรระยะสั้น"
        if trend == "downtrend" and isinstance(rsi, (float, int)) and float(rsi) <= 35:
            return "แม้แนวโน้มยังลบ แต่มีโอกาสรีบาวด์ทางเทคนิคได้"
        return "ติดตามการยืนเหนือแนวรับและการผ่านแนวต้านสำคัญเพื่อยืนยันรอบถัดไป"

    @staticmethod
    def _format_fallback_news(news_items: Any) -> str:
        if not isinstance(news_items, list) or not news_items:
            return ""
        lines: list[str] = []
        for item in news_items[:DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT]:
            if not isinstance(item, Mapping):
                continue
            title = item.get("title") or "-"
            source = item.get("source") or "Unknown"
            lines.append(f"- {title} ({source})")
        return "\n".join(lines)

    @staticmethod
    def _determine_fallback_verbosity(
        *,
        question: str | None,
        asset_scope: AssetScope,
    ) -> FallbackVerbosity:
        if not question:
            return "medium"
        normalized = question.casefold()
        short_check_keywords = (
            "น่าเข้าไหม",
            "เข้าไหม",
            "ซื้อไหม",
            "ถือไหม",
            "รอไหม",
            "ดีไหม",
            "ยังไหวไหม",
            "ตอนนี้",
            "quick take",
            "quick check",
        )
        short_keywords = (
            "สั้นมาก",
            "สั้น",
            "ย่อ",
            "สรุปเร็ว",
            "เร็ว",
            "quick",
            "brief",
            "short",
        )
        medium_keywords = (
            "ภาพรวม",
            "สรุปตลาด",
            "market update",
            "trend",
            "เทรนด์",
            "overview",
        )
        detailed_keywords = (
            "ละเอียด",
            "detail",
            "detailed",
            "deep",
            "ลึก",
            "เพราะอะไร",
            "เหตุผล",
            "why",
            "เปรียบเทียบ",
            "เทียบ",
            "จัดพอร์ต",
            "พอร์ต",
            "allocation",
            "strategy",
            "entry",
            "exit",
            "แนวรับ",
            "แนวต้าน",
        )
        gold_keywords = ("ทอง", "gold", "xau", "gld", "iau")
        equity_keywords = ("หุ้น", "stock", "nasdaq", "s&p", "spy", "qqq")
        etf_keywords = ("etf", "vti", "xlf", "xle", "xlk")
        asset_specific_short = any(keyword in normalized for keyword in short_check_keywords)
        asks_for_detail = any(keyword in normalized for keyword in detailed_keywords)
        asks_for_overview = any(keyword in normalized for keyword in medium_keywords)
        asks_for_short = any(keyword in normalized for keyword in short_keywords)
        mentions_gold = any(keyword in normalized for keyword in gold_keywords)
        mentions_equity = any(keyword in normalized for keyword in equity_keywords)
        mentions_etf = any(keyword in normalized for keyword in etf_keywords)

        if asks_for_detail:
            return "detailed"
        if asks_for_short:
            return "short"
        if asks_for_overview and asset_scope == "all":
            return "medium"
        if asset_specific_short and (mentions_gold or mentions_equity or mentions_etf or asset_scope != "all"):
            return "short"
        if any(keyword in normalized for keyword in short_keywords):
            return "short"
        if asks_for_overview:
            return "medium"
        if (
            any(keyword in normalized for keyword in ("เปรียบ", "compare", "เทียบ", "ตัวไหนดีกว่า"))
            or normalized.count("?") >= 2
            or normalized.count("หรือ") >= 2
        ):
            return "detailed"
        if asset_scope != "all" and len(normalized) <= 80:
            return "short"
        if any(keyword in normalized for keyword in ("ไหม", "มั้ย", "หรือ", "ควร")):
            return "short"
        return "medium"

    @staticmethod
    def _truncate_text(value: str | None, limit: int) -> str:
        text = (value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 3)].rstrip()}..."

    @staticmethod
    def _round_optional(value: float | None, digits: int = 2) -> float | None:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)

    async def generate_daily_wealth_analysis(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        news_limit: int = 8,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
    ) -> RecommendationResult:
        """Generate a structured daily wealth report."""
        news, market_data, trends, macro_context = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
        )

        question = "โปรดจัดทำรายงาน 'รายงานสรุปความมั่งคั่งประจำวัน (Daily Intelligence Report)' ตามรูปแบบที่กำหนด"
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            question=question,
            asset_scope="all",
        )

    def check_black_swan(self, news: list[NewsArticle], macro_context: dict[str, float | None]) -> tuple[bool, str]:
        """Detect extreme market danger trigger events."""
        vix = macro_context.get("vix")
        if vix and vix > 30.0:
            return True, f"📌 VIX Index พุ่งสูงถึง {vix} สะท้อนความกลัวในตลาดระดับวิกฤต"
            
        keywords = ["market crash", "war", "สงคราม", "วิกฤต", "ทรุดหนัก"]
        for article in news:
            title = article.title.lower()
            if any(kw in title for kw in keywords):
                return True, f"🚨 ตรวจพบข่าววิกฤตสำคัญ: {article.title}"
        return False, ""
