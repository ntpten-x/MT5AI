from __future__ import annotations

import asyncio
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
from loguru import logger

from invest_advisor_bot.analysis.portfolio_profile import (
    AllocationBucket,
    InvestorProfile,
    InvestorProfileName,
    PortfolioPlan,
    build_portfolio_plan,
    detect_investor_profile,
    get_investor_profile,
    normalize_profile_name,
)
from invest_advisor_bot.analysis.macro_regime import MacroRegimeAssessment, assess_macro_regime
from invest_advisor_bot.analysis.portfolio_allocation import (
    PortfolioHoldingReview,
    PortfolioRebalanceReview,
    PortfolioAllocationPlan,
    build_portfolio_allocation_plan,
    build_portfolio_rebalance_review,
    infer_allocation_mix_category,
)
from invest_advisor_bot.analysis.confidence_scoring import (
    ConfidenceAssessment,
    assess_market_recommendation_confidence,
    assess_stock_candidate_confidence,
)
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore, StoredSectorRotationSnapshot
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.portfolio_state import PortfolioHolding
from invest_advisor_bot.analysis.asset_ranking import RankedAsset, rank_asset_snapshots
from invest_advisor_bot.analysis.news_impact import NewsImpact, score_news_impacts
from invest_advisor_bot.analysis.risk_score import RiskScoreAssessment, calculate_risk_score
from invest_advisor_bot.analysis.stock_screener import StockCandidate, rank_stock_universe
from invest_advisor_bot.analysis.trend_engine import TrendAssessment, evaluate_trend
from invest_advisor_bot.providers.llm_client import OpenAICompatibleLLMClient
from invest_advisor_bot.providers.market_data_client import (
    AnalystExpectationProfile,
    AssetQuote,
    EarningsEvent,
    MarketDataClient,
    OhlcvBar,
    RecentEarningsResult,
    StockFundamentals,
)
from invest_advisor_bot.providers.news_client import NewsArticle, NewsClient
from invest_advisor_bot.providers.research_client import ResearchClient, ResearchFinding
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.universe import StockUniverseMember, US_LARGE_CAP_STOCK_UNIVERSE, find_stock_candidates_from_text

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "system_investment_advisor.txt"
DEFAULT_CHAT_HISTORY_LIMIT = 3
DEFAULT_NEWS_CONTEXT_LIMIT = 5
DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT = 3
DEFAULT_REASON_LIMIT = 2

SECTOR_ETF_LABELS: dict[str, str] = {
    "xlk_etf": "Technology",
    "xlf_etf": "Financials",
    "xle_etf": "Energy",
    "xly_etf": "Consumer Discretionary",
    "xlp_etf": "Consumer Staples",
    "xlv_etf": "Healthcare",
    "xli_etf": "Industrials",
    "xlb_etf": "Materials",
    "xlu_etf": "Utilities",
    "xlc_etf": "Communication Services",
    "xlre_etf": "Real Estate",
}

AssetScope = Literal["all", "gold-only", "us-stocks", "etf-only", "bonds"]
FallbackVerbosity = Literal["short", "medium", "detailed"]

ASSET_SCOPE_MEMBERS: dict[AssetScope, tuple[str, ...]] = {
    "all": (),
    "gold-only": ("gold_futures", "gld_etf", "iau_etf"),
    "us-stocks": (
        "sp500_index", "nasdaq_index", "spy_etf", "qqq_etf", "vti_etf", "xlf_etf", "xle_etf", "xlk_etf", "voo_etf",
    ),
    "etf-only": (
        "spy_etf", "qqq_etf", "gld_etf", "iau_etf", "vti_etf", "xlf_etf", "xle_etf", "xlk_etf", "tlt_etf", "voo_etf",
    ),
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


@dataclass(slots=True, frozen=True)
class InterestingAlert:
    key: str
    text: str
    severity: str = "info"
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class StockScreeningResult:
    recommendation_text: str
    picks: tuple[StockCandidate, ...]
    question: str
    fallback_used: bool
    model: str | None = None


@dataclass(slots=True, frozen=True)
class SectorRotationSignal:
    sector: str
    asset: str
    ticker: str | None
    trend_direction: str
    trend_score: float
    rsi: float | None
    stance: str
    rationale: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SectorPersistenceInsight:
    sector: str
    ticker: str | None
    regime: str
    stance: str
    streak: int
    average_score: float
    score_delta: float
    changed_from: str | None


@dataclass(slots=True, frozen=True)
class EarningsInterpretation:
    ticker: str
    earnings_at: datetime
    eps_estimate: float | None
    reported_eps: float | None
    surprise_pct: float | None
    revenue_signal: str
    revenue_qoq_change: float | None
    revenue_qoq_trend: str
    revenue_expectation_gap_pct: float | None
    revenue_expectation_source: str | None
    revenue_vs_sector_median: float | None
    revenue_relative_signal: str
    margin_signal: str
    margin_qoq_change: float | None
    margin_qoq_trend: str
    margin_vs_sector_median: float | None
    margin_relative_signal: str
    guidance_signal: str
    guidance_score: int
    management_tone: str
    fcf_quality: str
    fcf_qoq_change: float | None
    fcf_qoq_trend: str
    fcf_vs_sector_median: float | None
    fcf_relative_signal: str
    earnings_quality_score: float
    earnings_quality_label: str
    one_off_risk: str
    stance: str
    rationale: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SectorBreadthInsight:
    sector: str
    ticker: str | None
    constituent_count: int
    advancers: int
    decliners: int
    participation_ratio: float
    average_trend_score: float
    equal_weight_confirmed: bool
    breadth_label: str


@dataclass(slots=True, frozen=True)
class SectorBreadthTrend:
    sector: str
    ticker: str | None
    regime: str
    current_participation_ratio: float
    score_delta: float
    sparkline: str
    history_scores: tuple[float, ...]
    trend_label: str


@dataclass(slots=True, frozen=True)
class MarketBreadthDiffusion:
    participant_count: int
    advancers: int
    decliners: int
    neutral_count: int
    advancing_ratio: float
    declining_ratio: float
    diffusion_score: float
    average_trend_score: float
    breadth_label: str
    rally_confirmed: bool


@dataclass(slots=True, frozen=True)
class MarketBreadthTrend:
    regime: str
    current_diffusion_score: float
    score_delta: float
    sparkline: str
    history_scores: tuple[float, ...]
    trend_label: str
    breadth_label: str


@dataclass(slots=True, frozen=True)
class IndexLeadershipDivergence:
    label: str
    cap_weight_ticker: str
    equal_weight_ticker: str
    short_return_spread: float
    medium_return_spread: float
    cap_weight_score: float
    equal_weight_score: float
    divergence_label: str
    severity: str


@dataclass(slots=True, frozen=True)
class PreEarningsRiskSignal:
    ticker: str
    earnings_at: datetime
    risk_score: float
    forward_pe: float | None
    revenue_growth_estimate: float | None
    eps_growth_estimate: float | None
    market_breadth_label: str | None
    divergence_label: str | None
    rationale: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class EarningsSetupCandidate:
    ticker: str
    company_name: str
    earnings_at: datetime
    setup_score: float
    setup_label: str
    valuation_signal: str
    expectation_signal: str
    trend_signal: str
    rationale: tuple[str, ...]


class RecommendationService:
    """Advisor service focused on wealth preservation, allocation, and explainable guidance."""

    def __init__(
        self,
        llm_client: OpenAICompatibleLLMClient,
        *,
        system_prompt_path: Path | None = None,
        chat_history_limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
        default_investor_profile: InvestorProfileName = "balanced",
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt_path = Path(system_prompt_path or DEFAULT_PROMPT_PATH)
        self.chat_history_limit = max(1, int(chat_history_limit))
        self.default_investor_profile = normalize_profile_name(default_investor_profile)
        self._conversation_history: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=self.chat_history_limit)
        )
        self._conversation_profiles: dict[str, InvestorProfileName] = {}

    def set_investor_profile(self, *, conversation_key: str, profile_name: InvestorProfileName | str) -> InvestorProfile:
        normalized = normalize_profile_name(str(profile_name), default=self.default_investor_profile)
        self._conversation_profiles[conversation_key] = normalized
        return get_investor_profile(normalized)

    def get_investor_profile(self, conversation_key: str | None = None) -> InvestorProfile:
        if conversation_key and conversation_key in self._conversation_profiles:
            return get_investor_profile(self._conversation_profiles[conversation_key])
        return get_investor_profile(self.default_investor_profile)

    async def generate_recommendation(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        macro_context: Mapping[str, float | None] | None = None,
        research_findings: Sequence[ResearchFinding] | None = None,
        portfolio_snapshot: Mapping[str, Any] | None = None,
        question: str | None = None,
        conversation_key: str | None = None,
        asset_scope: AssetScope = "all",
        fallback_verbosity_override: FallbackVerbosity | None = None,
        investor_profile_name: InvestorProfileName | None = None,
    ) -> RecommendationResult:
        effective_profile = self._resolve_investor_profile(
            question=question,
            conversation_key=conversation_key,
            investor_profile_name=investor_profile_name,
        )
        system_prompt = self._load_system_prompt()
        payload = self._build_payload(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            asset_scope=asset_scope,
            question=question,
            investor_profile=effective_profile,
        )
        user_prompt = self._build_prompt(
            payload=payload,
            question=question,
            history_lines=self._get_history_lines(conversation_key),
        )

        llm_response = await self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={"service": "recommendation_service", "language": "th", "profile": effective_profile.name, "scope": asset_scope},
        )
        if llm_response is None:
            verbosity = fallback_verbosity_override or self._determine_fallback_verbosity(question=question, asset_scope=asset_scope)
            fallback_text = self._build_fallback_question_answer(question=question, payload=payload, verbosity=verbosity) if question else self._build_fallback_summary(payload, verbosity=verbosity)
            log_event(
                "llm_fallback_used",
                service="recommendation_service",
                scope=asset_scope,
                profile=effective_profile.name,
            )
            diagnostics.record_response(service="recommendation_service", fallback_used=True)
            self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=fallback_text)
            return RecommendationResult(
                recommendation_text=fallback_text,
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload=payload,
                fallback_used=True,
        )

        self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=llm_response.text)
        log_event(
            "llm_response_used",
            service="recommendation_service",
            model=llm_response.model,
            scope=asset_scope,
            profile=effective_profile.name,
        )
        diagnostics.record_response(service="recommendation_service", fallback_used=False)
        return RecommendationResult(
            recommendation_text=llm_response.text,
            model=llm_response.model,
            system_prompt_path=str(self.system_prompt_path),
            input_payload=payload,
            response_id=llm_response.response_id,
        )

    async def generate_market_update(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        news_limit: int = DEFAULT_NEWS_CONTEXT_LIMIT,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
        asset_scope: AssetScope = "all",
        conversation_key: str | None = None,
        portfolio_holdings: Sequence[PortfolioHolding] = (),
    ) -> RecommendationResult:
        news, market_data, trends, macro_context, research_findings = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
            research_query="latest macro outlook Federal Reserve US stocks ETF gold inflation Treasury yields",
        )
        portfolio_snapshot = await self._gather_portfolio_snapshot(
            market_data_client=market_data_client,
            holdings=portfolio_holdings,
        )
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            question="สรุปภาพรวมตลาดโลกและแนวทางจัดพอร์ตล่าสุดแบบอ่านง่าย",
            conversation_key=conversation_key,
            asset_scope=asset_scope,
        )

    async def answer_user_question(
        self,
        *,
        question: str,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        news_limit: int = DEFAULT_NEWS_CONTEXT_LIMIT,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
        conversation_key: str | None = None,
        asset_scope: AssetScope | None = None,
        fallback_verbosity_override: FallbackVerbosity | None = None,
        investor_profile_name: InvestorProfileName | None = None,
        portfolio_holdings: Sequence[PortfolioHolding] = (),
    ) -> RecommendationResult:
        normalized_question = question.strip()
        if not normalized_question:
            return RecommendationResult(
                recommendation_text="กรุณาพิมพ์คำถามเกี่ยวกับการลงทุนที่ต้องการวิเคราะห์",
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload={},
                fallback_used=True,
            )

        if self._is_stock_screener_question(normalized_question):
            screening = await self.screen_us_stocks(
                question=normalized_question,
                news_client=news_client,
                market_data_client=market_data_client,
                research_client=research_client,
                limit=self._extract_requested_pick_count(normalized_question),
                conversation_key=conversation_key,
                investor_profile_name=investor_profile_name,
            )
            return RecommendationResult(
                recommendation_text=screening.recommendation_text,
                model=screening.model,
                system_prompt_path=str(self.system_prompt_path),
                input_payload={
                    "question": normalized_question,
                    "stock_picks": [self._serialize_stock_candidate(candidate) for candidate in screening.picks],
                },
                fallback_used=screening.fallback_used,
            )

        mentioned_stocks = self._match_stock_mentions(normalized_question)
        if mentioned_stocks:
            screening = await self.analyze_specific_stocks(
                question=normalized_question,
                stock_members=mentioned_stocks,
                news_client=news_client,
                market_data_client=market_data_client,
                research_client=research_client,
                conversation_key=conversation_key,
                investor_profile_name=investor_profile_name,
            )
            return RecommendationResult(
                recommendation_text=screening.recommendation_text,
                model=screening.model,
                system_prompt_path=str(self.system_prompt_path),
                input_payload={
                    "question": normalized_question,
                    "stock_picks": [self._serialize_stock_candidate(candidate) for candidate in screening.picks],
                },
                fallback_used=screening.fallback_used,
            )

        effective_scope = asset_scope or self._detect_asset_scope(normalized_question)
        news, market_data, trends, macro_context, research_findings = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
            research_query=normalized_question,
        )
        portfolio_snapshot = await self._gather_portfolio_snapshot(
            market_data_client=market_data_client,
            holdings=portfolio_holdings,
        )
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            question=normalized_question,
            conversation_key=conversation_key,
            asset_scope=effective_scope,
            fallback_verbosity_override=fallback_verbosity_override,
            investor_profile_name=investor_profile_name,
        )

    async def screen_us_stocks(
        self,
        *,
        question: str,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        limit: int = 5,
        conversation_key: str | None = None,
        investor_profile_name: InvestorProfileName | None = None,
    ) -> StockScreeningResult:
        effective_profile = self._resolve_investor_profile(
            question=question,
            conversation_key=conversation_key,
            investor_profile_name=investor_profile_name,
        )
        picks = await self._screen_stock_universe(
            market_data_client=market_data_client,
            top_k=limit,
        )
        stock_news = await self._fetch_news_for_candidates(news_client=news_client, candidates=picks, limit_per_stock=2)
        research_findings = await self._gather_research_findings(
            research_client=research_client,
            research_query=question,
            limit=3,
        )
        system_prompt = self._load_system_prompt()
        prompt = self._build_stock_screener_prompt(
            question=question,
            profile=effective_profile,
            picks=picks,
            stock_news=stock_news,
            research_findings=research_findings,
        )
        llm_response = await self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            metadata={"service": "stock_screener", "language": "th", "profile": effective_profile.name, "scope": "us-stocks"},
        )
        if llm_response is None:
            fallback = self._build_stock_screener_fallback(
                question=question,
                picks=picks,
                profile=effective_profile,
                stock_news=stock_news,
            )
            log_event("llm_fallback_used", service="stock_screener", profile=effective_profile.name, scope="us-stocks")
            diagnostics.record_response(service="stock_screener", fallback_used=True)
            self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=fallback)
            return StockScreeningResult(
                recommendation_text=fallback,
                picks=tuple(picks),
                question=question,
                fallback_used=True,
            )
        self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=llm_response.text)
        log_event("llm_response_used", service="stock_screener", model=llm_response.model, profile=effective_profile.name, scope="us-stocks")
        diagnostics.record_response(service="stock_screener", fallback_used=False)
        return StockScreeningResult(
            recommendation_text=llm_response.text,
            picks=tuple(picks),
            question=question,
            fallback_used=False,
            model=llm_response.model,
        )

    async def analyze_specific_stocks(
        self,
        *,
        question: str,
        stock_members: Sequence[StockUniverseMember],
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        conversation_key: str | None = None,
        investor_profile_name: InvestorProfileName | None = None,
    ) -> StockScreeningResult:
        effective_profile = self._resolve_investor_profile(
            question=question,
            conversation_key=conversation_key,
            investor_profile_name=investor_profile_name,
        )
        stock_universe = {
            member.ticker.casefold().replace("-", "_"): member
            for member in stock_members
        }
        picks = await self._screen_stock_universe(
            market_data_client=market_data_client,
            stock_universe=stock_universe,
            top_k=max(1, len(stock_members)),
        )
        stock_news = await self._fetch_news_for_candidates(news_client=news_client, candidates=picks, limit_per_stock=3)
        research_findings = await self._gather_research_findings(
            research_client=research_client,
            research_query=question,
            limit=3,
        )
        fallback = self._build_stock_screener_fallback(
            question=question,
            picks=picks,
            profile=effective_profile,
            stock_news=stock_news,
        )
        if research_findings:
            fallback += "\n\nข้อมูลวิจัยเว็บล่าสุด\n" + "\n".join(
                f"- {item.title} ({item.source})" for item in research_findings[:3]
            )
        self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=fallback)
        return StockScreeningResult(
            recommendation_text=fallback,
            picks=tuple(picks),
            question=question,
            fallback_used=True,
        )

    async def generate_daily_digest(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        news_limit: int = 6,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
        portfolio_holdings: Sequence[PortfolioHolding] = (),
    ) -> RecommendationResult:
        news, market_data, trends, macro_context, research_findings = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
            research_query="daily market intelligence for US stocks ETF gold macro risk",
        )
        portfolio_snapshot = await self._gather_portfolio_snapshot(
            market_data_client=market_data_client,
            holdings=portfolio_holdings,
        )
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            question="สรุป Daily Intelligence Report สำหรับนักลงทุนที่ต้องการรักษาและเติบโตทรัพย์สิน",
            fallback_verbosity_override="medium",
            investor_profile_name=self.default_investor_profile,
        )

    async def generate_periodic_report(
        self,
        *,
        report_kind: Literal["morning", "midday", "closing"],
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        sector_rotation_state_store: SectorRotationStateStore | None = None,
        report_memory_store: ReportMemoryStore | None = None,
        sector_rotation_min_streak: int = 3,
        earnings_result_lookback_days: int = 14,
        news_limit: int = 6,
        history_period: str = "6mo",
        history_interval: str = "1d",
        history_limit: int = 180,
        portfolio_holdings: Sequence[PortfolioHolding] = (),
    ) -> RecommendationResult:
        news, market_data, trends, macro_context, research_findings = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
            research_query=f"{report_kind} report sector rotation earnings calendar us stocks etf gold",
        )
        sector_rotation = self._analyze_sector_rotation(market_data=market_data, trends=trends)
        sector_persistence_daily = self._analyze_sector_rotation_persistence(
            sector_rotation,
            state_store=sector_rotation_state_store,
            min_streak=sector_rotation_min_streak,
            regime="daily",
            record_snapshot=False,
        )
        sector_persistence_intraday = self._analyze_sector_rotation_persistence(
            sector_rotation,
            state_store=sector_rotation_state_store,
            min_streak=sector_rotation_min_streak,
            regime="intraday",
            record_snapshot=False,
        )
        sector_breadth = await self._analyze_sector_breadth(
            market_data_client=market_data_client,
            rotation=sector_rotation,
        )
        market_breadth = await self._analyze_market_breadth_diffusion(
            market_data_client=market_data_client,
        )
        leadership_divergence = await self._analyze_index_leadership_divergence(
            market_data_client=market_data_client,
        )
        if sector_rotation_state_store is not None and report_kind == "closing":
            self._record_sector_regime_snapshot(
                rotation=sector_rotation,
                breadth=sector_breadth,
                market_breadth=market_breadth,
                state_store=sector_rotation_state_store,
                regime="daily",
            )
        market_breadth_trend = self._analyze_market_breadth_trend(
            state_store=sector_rotation_state_store,
            current_breadth=market_breadth,
            regime="daily",
        )
        sector_breadth_trend = self._analyze_sector_breadth_trend(
            state_store=sector_rotation_state_store,
            current_breadth=sector_breadth,
            regime="daily",
        )
        stock_picks = await self._screen_stock_universe(market_data_client=market_data_client, top_k=5)
        report_memory_context = self._build_report_memory_context(
            report_kind=report_kind,
            report_memory_store=report_memory_store,
        )
        earnings = await market_data_client.get_earnings_calendar(
            [pick.ticker for pick in stock_picks],
            days_ahead=7 if report_kind != "closing" else 3,
        )
        earnings_setups = await self._rank_earnings_setups(
            market_data_client=market_data_client,
            candidates=stock_picks,
            days_ahead=10,
            market_breadth=market_breadth,
            divergence_signals=leadership_divergence,
        )
        earnings_surprises = await self._gather_post_earnings_interpretations(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            tickers=[pick.ticker for pick in stock_picks],
            lookback_days=earnings_result_lookback_days,
        )
        portfolio_snapshot = await self._gather_portfolio_snapshot(
            market_data_client=market_data_client,
            holdings=portfolio_holdings,
        )
        fundamentals_map = await market_data_client.get_stock_universe_fundamentals(
            {
                candidate.asset: StockUniverseMember(
                    ticker=candidate.ticker,
                    company_name=candidate.company_name,
                    sector=candidate.sector,
                    benchmark=candidate.benchmark,
                )
                for candidate in stock_picks
            }
        )
        question = {
            "morning": "สรุปรายงานก่อนเปิดตลาด เน้น sector rotation หุ้นเด่น และ earnings ที่ใกล้เข้ามา",
            "midday": "สรุปรายงานระหว่างวัน เน้นการเปลี่ยนแรงนำของ sector หุ้นเด่น และสิ่งที่ต้องจับตาช่วงครึ่งหลัง",
            "closing": "สรุปรายงานหลังปิดตลาด เน้น sector rotation หุ้นเด่น และ earnings ที่ต้องติดตามรอบถัดไป",
        }[report_kind]
        report_profile = get_investor_profile(self.default_investor_profile)
        payload = self._build_payload(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            asset_scope="all",
            question=question,
            investor_profile=report_profile,
        )
        payload["sector_rotation"] = [self._serialize_sector_rotation(item) for item in sector_rotation[:5]]
        payload["sector_rotation_persistence_daily"] = [
            self._serialize_sector_persistence(item) for item in sector_persistence_daily[:5]
        ]
        payload["sector_rotation_persistence_intraday"] = [
            self._serialize_sector_persistence(item) for item in sector_persistence_intraday[:5]
        ]
        payload["market_breadth_diffusion"] = self._serialize_market_breadth_diffusion(market_breadth)
        payload["market_breadth_trend"] = self._serialize_market_breadth_trend(market_breadth_trend)
        payload["index_leadership_divergence"] = [self._serialize_index_leadership_divergence(item) for item in leadership_divergence]
        payload["sector_breadth"] = [self._serialize_sector_breadth(item) for item in sector_breadth[:5]]
        payload["sector_breadth_trend"] = [self._serialize_sector_breadth_trend(item) for item in sector_breadth_trend[:5]]
        payload["stock_picks"] = [self._serialize_stock_candidate(item) for item in stock_picks]
        payload["report_memory_context"] = report_memory_context
        payload["earnings_calendar"] = [self._serialize_earnings_event(item) for item in earnings.values()]
        payload["earnings_setups"] = [self._serialize_earnings_setup(item) for item in earnings_setups[:5]]
        payload["earnings_surprises"] = [self._serialize_earnings_interpretation(item) for item in earnings_surprises[:4]]
        payload["earnings_quality_focus"] = [
            {
                "ticker": candidate.ticker,
                "forward_pe": candidate.forward_pe,
                "revenue_growth": candidate.revenue_growth,
                "earnings_growth": candidate.earnings_growth,
                "profit_margin": candidate.profit_margin,
            }
            for candidate in stock_picks[:4]
            if candidate.asset in fundamentals_map
        ]
        if portfolio_snapshot:
            payload["portfolio_snapshot"] = dict(portfolio_snapshot)

        system_prompt = self._load_system_prompt()
        user_prompt = self._build_periodic_report_prompt(report_kind=report_kind, payload=payload)
        llm_response = await self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={"service": "periodic_report", "language": "th", "profile": report_profile.name, "scope": "all"},
        )
        if llm_response is None:
            fallback = self._build_periodic_report_fallback(report_kind=report_kind, payload=payload)
            log_event("llm_fallback_used", service="periodic_report", report_kind=report_kind, profile=report_profile.name)
            diagnostics.record_response(service="periodic_report", fallback_used=True)
            if report_memory_store is not None:
                report_memory_store.remember(
                    report_kind=report_kind,
                    summary=self._build_report_memory_summary(report_kind=report_kind, payload=payload),
                )
            return RecommendationResult(
                recommendation_text=fallback,
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload=payload,
                fallback_used=True,
            )
        if report_memory_store is not None:
            report_memory_store.remember(
                report_kind=report_kind,
                summary=self._build_report_memory_summary(report_kind=report_kind, payload=payload),
            )
        log_event("llm_response_used", service="periodic_report", report_kind=report_kind, model=llm_response.model, profile=report_profile.name)
        diagnostics.record_response(service="periodic_report", fallback_used=False)
        return RecommendationResult(
            recommendation_text=llm_response.text,
            model=llm_response.model,
            system_prompt_path=str(self.system_prompt_path),
            input_payload=payload,
            response_id=llm_response.response_id,
        )

    async def generate_risk_alerts(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        news_limit: int = 5,
        history_period: str = "3mo",
        history_interval: str = "1d",
        history_limit: int = 90,
        vix_threshold: float = 30.0,
    ) -> list[str]:
        alerts = await self.generate_interest_alerts(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
            vix_threshold=vix_threshold,
            risk_score_threshold=max(6.0, vix_threshold / 5.0),
            opportunity_score_threshold=3.0,
            news_impact_threshold=2.0,
        )
        return [alert.text for alert in alerts]

    async def generate_interest_alerts(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        news_limit: int = 5,
        history_period: str = "3mo",
        history_interval: str = "1d",
        history_limit: int = 90,
        vix_threshold: float = 30.0,
        risk_score_threshold: float = 6.5,
        opportunity_score_threshold: float = 2.8,
        news_impact_threshold: float = 2.0,
    ) -> list[InterestingAlert]:
        news, market_data, trends, macro_context, research_findings = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=news_limit,
            history_period=history_period,
            history_interval=history_interval,
            history_limit=history_limit,
            research_query="market risk alert latest macro news VIX US stocks gold ETF",
        )
        portfolio_snapshot = await self._gather_portfolio_snapshot(
            market_data_client=market_data_client,
            holdings=(),
        )

        alert_profile = get_investor_profile(self.default_investor_profile)
        payload = self._build_payload(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            asset_scope="all",
            question="continuous market scan",
            investor_profile=alert_profile,
        )
        asset_snapshots = payload.get("asset_snapshots", [])
        if not isinstance(asset_snapshots, list):
            asset_snapshots = []

        news_impacts = score_news_impacts(news, limit=min(news_limit, 5), min_abs_score=news_impact_threshold)
        rankings = rank_asset_snapshots(asset_snapshots, top_k=4)
        risk_score = calculate_risk_score(
            macro_context=macro_context,
            trends=trends,
            news_impacts=news_impacts,
        )
        alerts = self._compose_interesting_alerts(
            payload=payload,
            rankings=rankings,
            news_impacts=news_impacts,
            risk_score=risk_score,
            vix_threshold=vix_threshold,
            risk_score_threshold=risk_score_threshold,
            opportunity_score_threshold=opportunity_score_threshold,
            news_impact_threshold=news_impact_threshold,
        )
        return alerts

    async def generate_stock_pick_alerts(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        watchlist: Sequence[str] = (),
        preferred_sectors: Sequence[str] = (),
        score_threshold: float = 1.8,
        daily_pick_enabled: bool = True,
        limit: int = 5,
    ) -> list[InterestingAlert]:
        picks = await self._screen_stock_universe(
            market_data_client=market_data_client,
            top_k=max(limit, 8),
        )
        if preferred_sectors:
            normalized_sectors = {sector.strip().casefold() for sector in preferred_sectors if sector.strip()}
            picks = [pick for pick in picks if pick.sector.casefold() in normalized_sectors]
        picks = [pick for pick in picks if pick.composite_score >= score_threshold and pick.stance in {"buy", "watch"}][:limit]
        if not picks:
            return []
        stock_news = await self._fetch_news_for_candidates(news_client=news_client, candidates=picks, limit_per_stock=1)
        research_findings = await self._gather_research_findings(
            research_client=research_client,
            research_query="best US stocks to buy now earnings momentum guidance",
            limit=2,
        )
        alerts: list[InterestingAlert] = []
        today_key = datetime.now(timezone.utc).date().isoformat()
        if daily_pick_enabled:
            top_pick = picks[0]
            top_news = stock_news.get(top_pick.asset, [])
            top_confidence = assess_stock_candidate_confidence(top_pick)
            top_news_line = f"\n- ข่าวประกอบ: {top_news[0].title}" if top_news else ""
            alerts.append(
                InterestingAlert(
                    key=f"stock:daily:{today_key}:{top_pick.ticker}",
                    severity="info",
                    text=(
                        f"{self._format_badged_title('✅ ยืนยัน', 'หุ้นเด่นวันนี้')}\n"
                        f"- หุ้น: {top_pick.company_name} ({top_pick.ticker}) | sector {top_pick.sector}\n"
                        f"- ภาพรวม: คะแนน {top_pick.composite_score:.2f} | ความมั่นใจ {self._humanize_confidence_label(top_confidence.label)} ({top_confidence.score:.2f}) | มุมมอง {self._humanize_stock_stance(top_pick.stance)}\n"
                        f"- เหตุผล: {'; '.join(top_pick.rationale[:3])}"
                        f"{top_news_line}\n"
                        f"- Action: {self._build_stock_candidate_action(top_pick, top_confidence.score, mode='daily')}"
                    ),
                    metadata={
                        "stock_pick": True,
                        "source_kind": "daily_pick",
                        "ticker": top_pick.ticker,
                        "company_name": top_pick.company_name,
                        "stance": top_pick.stance,
                        "entry_price": top_pick.price,
                        "composite_score": top_pick.composite_score,
                        "confidence_score": top_confidence.score,
                        "confidence_label": top_confidence.label,
                    },
                )
            )
        for candidate in picks[1:]:
            if candidate.composite_score < score_threshold + 0.3:
                continue
            candidate_news = stock_news.get(candidate.asset, [])
            candidate_confidence = assess_stock_candidate_confidence(candidate)
            candidate_news_line = f"\n- ข่าวประกอบ: {candidate_news[0].title}" if candidate_news else ""
            research_line = f"\n- วิจัยประกอบ: {research_findings[0].title}" if research_findings else ""
            alerts.append(
                InterestingAlert(
                    key=f"stock:opportunity:{candidate.ticker}:{int(round(candidate.composite_score * 10))}",
                    severity="info",
                    text=(
                        f"{self._format_badged_title('🔎 จับตา', 'หุ้นเด่นเพิ่ม')}\n"
                        f"- หุ้น: {candidate.company_name} ({candidate.ticker}) | sector {candidate.sector}\n"
                        f"- ภาพรวม: คะแนน {candidate.composite_score:.2f} | ความมั่นใจ {self._humanize_confidence_label(candidate_confidence.label)} ({candidate_confidence.score:.2f}) | มุมมอง {self._humanize_stock_stance(candidate.stance)}\n"
                        f"- เหตุผล: {'; '.join(candidate.rationale[:3])}"
                        f"{candidate_news_line}"
                        f"{research_line}\n"
                        f"- Action: {self._build_stock_candidate_action(candidate, candidate_confidence.score, mode='opportunity')}"
                    ),
                    metadata={
                        "stock_pick": True,
                        "source_kind": "opportunity_pick",
                        "ticker": candidate.ticker,
                        "company_name": candidate.company_name,
                        "stance": candidate.stance,
                        "entry_price": candidate.price,
                        "composite_score": candidate.composite_score,
                        "confidence_score": candidate_confidence.score,
                        "confidence_label": candidate_confidence.label,
                    },
                )
            )
        if watchlist:
            watchlist_universe = {
                ticker.casefold().replace("-", "_"): StockUniverseMember(
                    ticker=ticker.upper(),
                    company_name=ticker.upper(),
                    sector="Unknown",
                    benchmark="watchlist",
                )
                for ticker in watchlist
                if ticker.strip()
            }
            watchlist_candidates = await self._screen_stock_universe(
                market_data_client=market_data_client,
                stock_universe=watchlist_universe,
                top_k=len(watchlist_universe),
            )
            for candidate in watchlist_candidates:
                if candidate.composite_score < score_threshold:
                    continue
                candidate_confidence = assess_stock_candidate_confidence(candidate)
                alerts.append(
                    InterestingAlert(
                        key=f"watchlist:{candidate.ticker}:{int(round(candidate.composite_score * 10))}",
                        severity="info",
                        text=(
                            f"{self._format_badged_title('🔎 จับตา', 'หุ้นใน Watchlist')}\n"
                            f"- หุ้น: {candidate.company_name} ({candidate.ticker})\n"
                            f"- ภาพรวม: คะแนน {candidate.composite_score:.2f} | ความมั่นใจ {self._humanize_confidence_label(candidate_confidence.label)} ({candidate_confidence.score:.2f}) | มุมมอง {self._humanize_stock_stance(candidate.stance)}\n"
                            f"- เหตุผล: {'; '.join(candidate.rationale[:3])}\n"
                            f"- Action: {self._build_stock_candidate_action(candidate, candidate_confidence.score, mode='watchlist')}"
                        ),
                        metadata={
                            "stock_pick": True,
                            "source_kind": "watchlist_pick",
                            "ticker": candidate.ticker,
                            "company_name": candidate.company_name,
                            "stance": candidate.stance,
                            "entry_price": candidate.price,
                            "composite_score": candidate.composite_score,
                            "confidence_score": candidate_confidence.score,
                            "confidence_label": candidate_confidence.label,
                        },
                    )
                )
        return alerts

    async def generate_sector_rotation_alerts(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        sector_rotation_state_store: SectorRotationStateStore | None = None,
        min_streak: int = 3,
    ) -> list[InterestingAlert]:
        news, market_data, trends, _, _ = await self._gather_context(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            news_limit=4,
            history_period="6mo",
            history_interval="1d",
            history_limit=180,
            research_query="sector rotation leadership us market",
        )
        rotation = self._analyze_sector_rotation(market_data=market_data, trends=trends)
        breadth = await self._analyze_sector_breadth(
            market_data_client=market_data_client,
            rotation=rotation,
        )
        market_breadth = await self._analyze_market_breadth_diffusion(
            market_data_client=market_data_client,
        )
        leadership_divergence = await self._analyze_index_leadership_divergence(
            market_data_client=market_data_client,
        )
        persistence = self._analyze_sector_rotation_persistence(
            rotation,
            state_store=sector_rotation_state_store,
            min_streak=min_streak,
            regime="intraday",
            record_snapshot=False,
        )
        if sector_rotation_state_store is not None:
            self._record_sector_regime_snapshot(
                rotation=rotation,
                breadth=breadth,
                market_breadth=market_breadth,
                state_store=sector_rotation_state_store,
                regime="intraday",
            )
        market_breadth_trend = self._analyze_market_breadth_trend(
            state_store=sector_rotation_state_store,
            current_breadth=market_breadth,
            regime="intraday",
        )
        sector_breadth_trend = self._analyze_sector_breadth_trend(
            state_store=sector_rotation_state_store,
            current_breadth=breadth,
            regime="intraday",
        )
        if len(rotation) < 2:
            return []
        leaders = [item for item in rotation if item.stance == "overweight"][:2]
        laggards = [item for item in rotation if item.stance == "underweight"][:2]
        alerts: list[InterestingAlert] = []
        news_bias = score_news_impacts(news, limit=2, min_abs_score=1.5) if news else []
        snapshot_alert = self._build_sector_rotation_snapshot_alert(
            leaders=leaders,
            laggards=laggards,
            persistence=persistence,
            breadth=breadth,
            sector_breadth_trend=sector_breadth_trend,
            market_breadth=market_breadth,
            market_breadth_trend=market_breadth_trend,
            leadership_divergence=leadership_divergence,
            news_impact=news_bias[0] if news_bias else None,
        )
        if snapshot_alert is not None:
            alerts.append(snapshot_alert)
        if market_breadth is not None:
            sp500_trend = trends.get("sp500_index")
            nasdaq_trend = trends.get("nasdaq_index")
            major_indexes_holding = any(
                trend is not None and (trend.direction == "uptrend" or trend.score >= 1.5)
                for trend in (sp500_trend, nasdaq_trend)
            )
            internals_break = (
                major_indexes_holding
                and market_breadth.breadth_label in {"fragile rally", "mixed tape", "risk-off tape", "broad selloff"}
                and (
                    market_breadth.diffusion_score <= 0.05
                    or (market_breadth_trend is not None and market_breadth_trend.trend_label == "weakening")
                )
            )
            if internals_break:
                alerts.append(
                    InterestingAlert(
                        key=(
                            "market:internals-break:"
                            f"{int(round(market_breadth.diffusion_score * 100))}:"
                            f"{market_breadth_trend.trend_label if market_breadth_trend is not None else 'static'}"
                        ),
                        severity="warning",
                        text=self._build_market_internals_break_text(
                            market_breadth=market_breadth,
                            market_breadth_trend=market_breadth_trend,
                            sp500_trend=sp500_trend,
                            nasdaq_trend=nasdaq_trend,
                        ),
                    )
                )
        return alerts

    async def generate_post_earnings_alerts(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        watchlist: Sequence[str] = (),
        top_candidates: Sequence[StockCandidate] = (),
        lookback_days: int = 14,
    ) -> list[InterestingAlert]:
        tickers = list(dict.fromkeys([*watchlist, *[candidate.ticker for candidate in top_candidates]]))
        interpretations = await self._gather_post_earnings_interpretations(
            news_client=news_client,
            market_data_client=market_data_client,
            research_client=research_client,
            tickers=tickers,
            lookback_days=lookback_days,
        )
        alerts: list[InterestingAlert] = []
        for item in interpretations:
            if item.stance == "neutral":
                continue
            severity = "info" if item.stance == "bullish" else "warning"
            alerts.append(
                InterestingAlert(
                    key=f"earnings:post:{item.ticker}:{item.earnings_at.date().isoformat()}:{item.stance}",
                    severity=severity,
                    text=(
                        f"{self._format_badged_title('✅ ยืนยัน' if item.stance == 'bullish' else '🟠 ระวัง', 'สรุปหลังประกาศงบ')}\n"
                        f"- หุ้น: {item.ticker} | surprise {item.surprise_pct if item.surprise_pct is not None else 'N/A'}% | EPS est {item.eps_estimate if item.eps_estimate is not None else 'N/A'} | EPS reported {item.reported_eps if item.reported_eps is not None else 'N/A'}\n"
                        f"- ภาพรวม: guidance {self._humanize_earnings_direction(item.guidance_signal)} | น้ำเสียงผู้บริหาร {self._humanize_earnings_direction(item.management_tone)} | มุมมอง {'เชิงบวก' if item.stance == 'bullish' else 'เชิงลบ'}\n"
                        f"- คุณภาพงบ: รายได้ {self._humanize_earnings_signal(item.revenue_signal)} ({self._humanize_trend_description(item.revenue_qoq_trend)} QoQ {self._format_percent_delta(item.revenue_qoq_change)}, {self._humanize_relative_signal(item.revenue_relative_signal)} {self._format_percent_delta(item.revenue_vs_sector_median)} vs sector, rev gap {self._format_percent_delta(item.revenue_expectation_gap_pct)}"
                        f"{f' via {item.revenue_expectation_source}' if item.revenue_expectation_source else ''})"
                        f" | margin {self._humanize_earnings_signal(item.margin_signal)} ({self._humanize_trend_description(item.margin_qoq_trend)} QoQ {self._format_percent_delta(item.margin_qoq_change)}, {self._humanize_relative_signal(item.margin_relative_signal)} {self._format_percent_delta(item.margin_vs_sector_median)} vs sector)"
                        f" | FCF {self._humanize_fcf_quality(item.fcf_quality)} ({self._humanize_trend_description(item.fcf_qoq_trend)} QoQ {self._format_percent_delta(item.fcf_qoq_change)}, {self._humanize_relative_signal(item.fcf_relative_signal)} {self._format_percent_delta(item.fcf_vs_sector_median)} vs sector)\n"
                        f"- คุณภาพกำไร: {self._humanize_earnings_quality_label(item.earnings_quality_label)} ({item.earnings_quality_score:.2f}) | one-off risk {self._humanize_one_off_risk(item.one_off_risk)}\n"
                        f"- เหตุผล: {'; '.join(item.rationale[:3])}\n"
                        f"- Action: {self._build_post_earnings_action(item)}"
                    ),
                )
            )
        return alerts

    async def generate_earnings_calendar_alerts(
        self,
        *,
        market_data_client: MarketDataClient,
        watchlist: Sequence[str] = (),
        top_candidates: Sequence[StockCandidate] = (),
        days_ahead: int = 7,
    ) -> list[InterestingAlert]:
        tickers = list(dict.fromkeys([*watchlist, *[candidate.ticker for candidate in top_candidates]]))
        if not tickers:
            return []
        earnings = await market_data_client.get_earnings_calendar(tickers, days_ahead=days_ahead)
        alerts: list[InterestingAlert] = []
        now = datetime.now(timezone.utc)
        for ticker, event in earnings.items():
            days_left = (event.earnings_at.date() - now.date()).days
            urgency = "วันนี้" if days_left <= 0 else f"อีก {days_left} วัน"
            badge = "🟠 ระวัง" if days_left <= 1 else "🔎 จับตา"
            alerts.append(
                InterestingAlert(
                    key=f"earnings:{ticker}:{event.earnings_at.date().isoformat()}",
                    severity="warning" if days_left <= 1 else "info",
                    text=(
                        f"{self._format_badged_title(badge, 'ปฏิทินงบ')}\n"
                        f"- หุ้น: {ticker} | จะประกาศงบ {urgency}\n"
                        f"- ภาพรวม: เวลา {event.earnings_at.astimezone(timezone.utc).isoformat(timespec='minutes')} | EPS estimate {event.eps_estimate if event.eps_estimate is not None else 'N/A'}\n"
                        f"- เหตุผล: ยิ่งใกล้วันประกาศงบ ความผันผวนและ gap risk มักเพิ่มขึ้น\n"
                        f"- Action: {self._build_earnings_calendar_action(days_left)}"
                    ),
                )
            )
        return alerts

    async def generate_pre_earnings_risk_alerts(
        self,
        *,
        market_data_client: MarketDataClient,
        watchlist: Sequence[str] = (),
        top_candidates: Sequence[StockCandidate] = (),
        days_ahead: int = 7,
    ) -> list[InterestingAlert]:
        tickers = list(dict.fromkeys([*watchlist, *[candidate.ticker for candidate in top_candidates]]))
        if not tickers:
            return []
        earnings = await market_data_client.get_earnings_calendar(tickers, days_ahead=days_ahead)
        if not earnings:
            return []
        expectation_profiles = await market_data_client.get_analyst_expectation_profiles(earnings.keys())
        fundamentals_results = await asyncio.gather(
            *[market_data_client.get_fundamentals(ticker) for ticker in earnings.keys()],
            return_exceptions=True,
        )
        fundamentals_map: dict[str, StockFundamentals | None] = {}
        for ticker, result in zip(earnings.keys(), fundamentals_results, strict=False):
            fundamentals_map[ticker] = None if isinstance(result, Exception) else result
        market_breadth = await self._analyze_market_breadth_diffusion(market_data_client=market_data_client)
        divergence_signals = await self._analyze_index_leadership_divergence(market_data_client=market_data_client)
        signals: list[PreEarningsRiskSignal] = []
        for ticker, event in earnings.items():
            expectation = expectation_profiles.get(ticker)
            fundamentals = fundamentals_map.get(ticker)
            if expectation is None and fundamentals is None:
                continue
            signal = self._evaluate_pre_earnings_risk(
                ticker=ticker,
                event=event,
                expectation=expectation,
                fundamentals=fundamentals,
                market_breadth=market_breadth,
                divergence_signals=divergence_signals,
            )
            if signal is not None:
                signals.append(signal)
        signals.sort(key=lambda item: item.risk_score, reverse=True)
        alerts: list[InterestingAlert] = []
        for item in signals:
            alerts.append(
                InterestingAlert(
                    key=f"earnings:pre-risk:{item.ticker}:{item.earnings_at.date().isoformat()}:{int(round(item.risk_score * 10))}",
                    severity="warning",
                    text=(
                        f"{self._format_badged_title('🟠 ระวัง', 'ความเสี่ยงก่อนประกาศงบ')}\n"
                        f"- หุ้น: {item.ticker} | วันประกาศงบ {item.earnings_at.date().isoformat()}\n"
                        f"- ภาพรวม: risk score {item.risk_score:.1f} | forward PE {item.forward_pe if item.forward_pe is not None else 'N/A'} | revenue est growth {self._format_percent_delta(item.revenue_growth_estimate)} | EPS est growth {self._format_percent_delta(item.eps_growth_estimate)}\n"
                        f"- บริบทตลาด: breadth {self._translate_market_breadth_label(item.market_breadth_label) if item.market_breadth_label else 'n/a'}"
                        f"{f' | divergence {item.divergence_label}' if item.divergence_label else ''}\n"
                        f"- เหตุผล: {'; '.join(item.rationale[:4])}\n"
                        f"- Action: {self._build_pre_earnings_action(item)}"
                    ),
                )
            )
        return alerts

    async def generate_earnings_setup_alerts(
        self,
        *,
        market_data_client: MarketDataClient,
        top_candidates: Sequence[StockCandidate] = (),
        days_ahead: int = 10,
    ) -> list[InterestingAlert]:
        setups = await self._rank_earnings_setups(
            market_data_client=market_data_client,
            candidates=top_candidates,
            days_ahead=days_ahead,
            market_breadth=await self._analyze_market_breadth_diffusion(market_data_client=market_data_client),
            divergence_signals=await self._analyze_index_leadership_divergence(market_data_client=market_data_client),
        )
        alerts: list[InterestingAlert] = []
        for item in setups[:2]:
            if item.setup_score < 1.8:
                continue
            badge = "✅ ยืนยัน" if item.setup_score >= 2.4 else "🔎 จับตา"
            alerts.append(
                InterestingAlert(
                    key=f"earnings:setup:{item.ticker}:{item.earnings_at.date().isoformat()}:{int(round(item.setup_score*10))}",
                    severity="info",
                    text=(
                        f"{self._format_badged_title(badge, 'Setup ก่อนงบ')}\n"
                        f"- หุ้น: {item.company_name} ({item.ticker}) | วันประกาศงบ {item.earnings_at.date().isoformat()}\n"
                        f"- ภาพรวม: score {item.setup_score:.2f} | {self._humanize_setup_label(item.setup_label)}\n"
                        f"- ปัจจัย: trend {self._humanize_trend_description(item.trend_signal)} | valuation {self._humanize_valuation_signal(item.valuation_signal)} | expectations {self._humanize_expectation_signal(item.expectation_signal)}\n"
                        f"- เหตุผล: {'; '.join(item.rationale[:3])}\n"
                        f"- Action: {self._build_earnings_setup_action(item)}"
                    ),
                )
            )
        return alerts

    async def _screen_stock_universe(
        self,
        *,
        market_data_client: MarketDataClient,
        stock_universe: Mapping[str, StockUniverseMember] | None = None,
        top_k: int,
    ) -> list[StockCandidate]:
        effective_universe = (
            dict(stock_universe)
            if stock_universe is not None
            else await market_data_client.get_dynamic_stock_universe(indexes=("sp500", "nasdaq100"), max_members=200)
        )
        quotes_task = market_data_client.get_stock_universe_snapshot(effective_universe)
        histories_task = market_data_client.get_stock_universe_history(
            stock_universe=effective_universe,
            period="6mo",
            interval="1d",
            limit=180,
        )
        fundamentals_task = market_data_client.get_stock_universe_fundamentals(effective_universe)
        quotes, histories, fundamentals = await asyncio.gather(
            self._safe_async_call(quotes_task, default={}, source_name="stock_universe_snapshot"),
            self._safe_async_call(histories_task, default={}, source_name="stock_universe_history"),
            self._safe_async_call(fundamentals_task, default={}, source_name="stock_universe_fundamentals"),
        )

        trends: dict[str, TrendAssessment] = {}
        for asset_name, bars in histories.items():
            frame = self._bars_to_frame(bars)
            if frame.empty:
                continue
            quote = quotes.get(asset_name)
            try:
                trends[asset_name] = evaluate_trend(frame, ticker=quote.ticker if quote else effective_universe[asset_name].ticker)
            except Exception as exc:
                logger.warning("Failed to evaluate stock trend for {}: {}", asset_name, exc)
        return rank_stock_universe(
            stock_universe=effective_universe,
            quotes=quotes,
            trends=trends,
            fundamentals=fundamentals,
            top_k=top_k,
        )

    async def _fetch_news_for_candidates(
        self,
        *,
        news_client: NewsClient,
        candidates: Sequence[StockCandidate],
        limit_per_stock: int,
    ) -> dict[str, list[NewsArticle]]:
        tasks = {
            candidate.asset: news_client.fetch_stock_news(
                candidate.ticker,
                company_name=candidate.company_name,
                limit=limit_per_stock,
                when="7d",
            )
            for candidate in candidates
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        payload: dict[str, list[NewsArticle]] = {}
        for asset, result in zip(tasks.keys(), results, strict=False):
            payload[asset] = [] if isinstance(result, Exception) else result
        return payload

    def _build_stock_screener_prompt(
        self,
        *,
        question: str,
        profile: InvestorProfile,
        picks: Sequence[StockCandidate],
        stock_news: Mapping[str, Sequence[NewsArticle]],
        research_findings: Sequence[ResearchFinding],
    ) -> str:
        pick_lines = []
        for candidate in picks:
            confidence = assess_stock_candidate_confidence(candidate)
            pick_lines.append(
                f"- {candidate.company_name} ({candidate.ticker}) | sector={candidate.sector} | score={candidate.composite_score} | confidence={confidence.label}:{confidence.score} | stance={candidate.stance} | trend={candidate.trend_direction} | RSI={candidate.rsi} | fwdPE={candidate.forward_pe} | revGrowth={candidate.revenue_growth} | reasons={'; '.join(candidate.rationale[:3])}"
            )
        news_lines = []
        for candidate in picks[:3]:
            articles = stock_news.get(candidate.asset, [])
            for article in articles[:2]:
                news_lines.append(f"- {candidate.ticker}: {article.title} ({article.source or 'Unknown'})")
        research_lines = [f"- {item.title} ({item.source})" for item in research_findings[:3]]
        return (
            f"คำถามผู้ใช้: {question}\n"
            f"โปรไฟล์ผู้ลงทุน: {profile.title_th} | เป้าหมาย: {profile.objective}\n\n"
            "ผล stock screener ล่าสุด:\n"
            + ("\n".join(pick_lines) or "- ไม่มีหุ้นที่ผ่านเงื่อนไข")
            + "\n\nข่าวเฉพาะหุ้น:\n"
            + ("\n".join(news_lines) or "- ไม่มีข่าวเด่นเฉพาะหุ้น")
            + "\n\nข้อมูลวิจัยเว็บ:\n"
            + ("\n".join(research_lines) or "- ไม่มีข้อมูลวิจัยเว็บเพิ่ม")
            + "\n\nตอบเป็นภาษาไทยแบบสั้น อ่านง่าย และบอกว่า 5 ตัวไหนน่าสนใจที่สุดตอนนี้ พร้อมเหตุผล, ความเสี่ยง, และลำดับความน่าสนใจ"
        )

    def _build_stock_screener_fallback(
        self,
        *,
        question: str,
        picks: Sequence[StockCandidate],
        profile: InvestorProfile,
        stock_news: Mapping[str, Sequence[NewsArticle]],
    ) -> str:
        if not picks:
            return "ยังไม่พบหุ้นที่มีสัญญาณเชิงคุณภาพและโมเมนตัมเด่นพอสำหรับแนะนำในตอนนี้ ควรรอจังหวะตลาดหรือถือเงินสดบางส่วน"
        lines = [
            f"คำถาม: {question}",
            f"โปรไฟล์ผู้ลงทุน: {profile.title_th}",
            "หุ้นเด่นจาก stock screener ตอนนี้",
        ]
        for index, candidate in enumerate(picks, start=1):
            confidence = assess_stock_candidate_confidence(candidate)
            news_snippet = ""
            candidate_news = stock_news.get(candidate.asset, [])
            if candidate_news:
                news_snippet = f" | ข่าว: {candidate_news[0].title}"
            lines.append(
                f"{index}. {candidate.company_name} ({candidate.ticker}) | คะแนน {candidate.composite_score} | confidence {confidence.label} ({confidence.score}) | มุมมอง {self._humanize_stock_stance(candidate.stance)} | เหตุผล: {'; '.join(candidate.rationale[:3])}{news_snippet}"
            )
        lines.append("หมายเหตุ: รายชื่อหุ้นนี้เป็นการคัดกรองเชิงระบบจาก trend + fundamentals + valuation ไม่ใช่การรับประกันผลตอบแทน")
        return "\n".join(lines)

    def _resolve_investor_profile(
        self,
        *,
        question: str | None,
        conversation_key: str | None,
        investor_profile_name: InvestorProfileName | None,
    ) -> InvestorProfile:
        if investor_profile_name is not None:
            if conversation_key:
                self._conversation_profiles[conversation_key] = investor_profile_name
            return get_investor_profile(investor_profile_name)

        detected = detect_investor_profile(question)
        if detected is not None:
            if conversation_key:
                self._conversation_profiles[conversation_key] = detected
            return get_investor_profile(detected)

        if conversation_key and conversation_key in self._conversation_profiles:
            return get_investor_profile(self._conversation_profiles[conversation_key])
        return get_investor_profile(self.default_investor_profile)

    def _load_system_prompt(self) -> str:
        try:
            return self.system_prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Failed to load system prompt from {}: {}", self.system_prompt_path, exc)
            return (
                "คุณคือ AI Investment Advisor ที่เน้นการรักษาเงินต้น จัดพอร์ตแบบมืออาชีพ "
                "และตอบเป็นภาษาไทยที่กระชับ มีเหตุผล และควบคุมความเสี่ยง"
            )

    def _build_payload(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        macro_context: Mapping[str, float | None] | None,
        research_findings: Sequence[ResearchFinding] | None,
        portfolio_snapshot: Mapping[str, Any] | None,
        asset_scope: AssetScope,
        question: str | None,
        investor_profile: InvestorProfile,
    ) -> dict[str, Any]:
        filtered_market_data, filtered_trends = self._filter_asset_context(
            market_data=market_data,
            trends=trends,
            asset_scope=asset_scope,
        )
        news_limit = DEFAULT_NEWS_CONTEXT_LIMIT if asset_scope == "all" else DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT
        asset_snapshots = [
            self._serialize_asset_context(
                asset_name=name,
                quote=filtered_market_data.get(name),
                trend=filtered_trends.get(name),
            )
            for name in filtered_market_data.keys()
        ]
        asset_snapshot_list = [item for item in asset_snapshots if item is not None]
        macro_regime = assess_macro_regime(
            macro_context=macro_context,
            asset_snapshots=asset_snapshot_list,
        )
        portfolio_plan = build_portfolio_plan(
            asset_snapshots=asset_snapshot_list,
            macro_context=macro_context,
            profile_name=investor_profile.name,
            asset_scope=asset_scope,
        )
        allocation_plan = build_portfolio_allocation_plan(
            investor_profile=investor_profile,
            macro_regime=macro_regime,
            asset_snapshots=asset_snapshot_list,
        )
        portfolio_review = self._build_portfolio_review_data(
            allocation_plan=allocation_plan,
            portfolio_snapshot=portfolio_snapshot,
        )
        market_confidence = assess_market_recommendation_confidence(
            asset_snapshots=asset_snapshot_list,
            macro_regime=self._serialize_macro_regime(macro_regime),
            news_items=[self._serialize_news_article(item) for item in list(news)[:news_limit]],
            research_items=[self._serialize_research_finding(item) for item in list(research_findings or [])[:3]],
            portfolio_review=portfolio_review,
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": asset_scope,
            "question": question,
            "investor_profile": self._serialize_investor_profile(investor_profile),
            "macro_context": {
                "vix": self._round_optional((macro_context or {}).get("vix")),
                "tnx": self._round_optional((macro_context or {}).get("tnx")),
                "cpi_yoy": self._round_optional((macro_context or {}).get("cpi_yoy")),
            },
            "news_headlines": [self._serialize_news_article(item) for item in list(news)[:news_limit]],
            "research_highlights": [self._serialize_research_finding(item) for item in list(research_findings or [])[:3]],
            "asset_snapshots": asset_snapshot_list,
            "macro_regime": self._serialize_macro_regime(macro_regime),
            "market_confidence": self._serialize_confidence_assessment(market_confidence),
            "portfolio_plan": self._serialize_portfolio_plan(portfolio_plan),
            "allocation_plan": self._serialize_allocation_plan(allocation_plan),
            "portfolio_snapshot": dict(portfolio_snapshot or {}),
            "portfolio_review": portfolio_review,
        }

    def _build_prompt(self, *, payload: Mapping[str, Any], question: str | None, history_lines: Sequence[str]) -> str:
        history_text = "\n".join(history_lines) if history_lines else "- ไม่มีบทสนทนาก่อนหน้า"
        intro = f"คำถามผู้ใช้: {question}\n" if question else "คำสั่งผู้ใช้: สรุปภาพรวมตลาดและแนวทางจัดพอร์ตล่าสุด\n"
        return (
            f"{intro}"
            "บริบทบทสนทนาล่าสุด:\n"
            f"{history_text}\n\n"
            "โปรไฟล์ผู้ลงทุน:\n"
            f"{self._format_profile_lines(payload.get('investor_profile'))}\n\n"
            "ตัวชี้วัดมหภาค:\n"
            f"{self._format_macro_lines(payload.get('macro_context'))}\n\n"
            "Macro regime:\n"
            f"{self._format_macro_regime_lines(payload.get('macro_regime'))}\n\n"
            "Recommendation confidence:\n"
            f"{self._format_confidence_lines(payload.get('market_confidence'))}\n\n"
            "ข่าวสำคัญล่าสุด:\n"
            f"{self._format_news_lines(payload.get('news_headlines'))}\n\n"
            "ข้อมูลวิจัยเว็บล่าสุด:\n"
            f"{self._format_research_lines(payload.get('research_highlights'))}\n\n"
            "สรุปสินทรัพย์สำคัญ:\n"
            f"{self._format_asset_lines(payload.get('asset_snapshots'))}\n\n"
            "แผนจัดพอร์ต:\n"
            f"{self._format_portfolio_lines(payload.get('portfolio_plan'))}\n\n"
            "Allocation mix:\n"
            f"{self._format_allocation_mix_lines(payload.get('allocation_plan'))}\n\n"
            "Current portfolio:\n"
            f"{self._format_portfolio_snapshot_lines(payload.get('portfolio_snapshot'))}\n\n"
            "Rebalance review:\n"
            f"{self._format_portfolio_review_lines(payload.get('portfolio_review'))}\n\n"
            "ตอบเป็นภาษาไทยแบบอ่านง่ายสำหรับ Telegram โดยมีหัวข้อดังนี้:\n"
            "1. มุมมองตลาด\n2. โปรไฟล์ผู้ลงทุน\n3. แผนจัดพอร์ต\n4. พอร์ตปัจจุบันและการรีบาลานซ์\n5. สินทรัพย์ที่ควรเพิ่มน้ำหนัก/คงน้ำหนัก/ลดน้ำหนัก\n"
            "6. ความเสี่ยงที่ต้องติดตาม\n7. แผนปฏิบัติการวันนี้\n8. หมายเหตุว่าเป็นข้อมูลเพื่อการศึกษาและการวางแผนพอร์ต ไม่ใช่การรับประกันผลตอบแทน\n"
            "หลีกเลี่ยงการชี้นำแบบเทรดสั้นและให้เหตุผลจาก macro + trend เท่านั้น"
        )

    def _build_fallback_question_answer(self, *, question: str, payload: Mapping[str, Any], verbosity: FallbackVerbosity) -> str:
        return f"คำถาม: {question}\n\n{self._build_fallback_summary(payload, verbosity=verbosity)}"

    def _build_fallback_summary(self, payload: Mapping[str, Any], *, verbosity: FallbackVerbosity = "medium") -> str:
        asset_snapshots = payload.get("asset_snapshots", [])
        if not isinstance(asset_snapshots, list) or not asset_snapshots:
            return "ข้อมูลตลาดยังไม่เพียงพอสำหรับสรุปคำแนะนำเชิงพอร์ตในตอนนี้ จึงควรรักษาเงินสดและรอข้อมูลเพิ่มก่อน"

        portfolio_plan = payload.get("portfolio_plan", {})
        market_view = self._build_market_overview(asset_snapshots, str(payload.get("scope") or "all"), portfolio_plan)
        profile_line = self._format_profile_one_line(payload.get("investor_profile"))
        macro_line = self._format_macro_one_line(payload.get("macro_context"))
        confidence_line = self._format_confidence_one_line(payload.get("market_confidence"))
        allocations = self._format_allocation_summary(portfolio_plan)
        portfolio_snapshot_lines = self._format_portfolio_snapshot_lines(payload.get("portfolio_snapshot"))
        portfolio_review_lines = self._format_portfolio_review_lines(payload.get("portfolio_review"))
        portfolio_review_brief = portfolio_review_lines.splitlines()[0].lstrip("- ").strip() if portfolio_review_lines else ""
        focus_assets = self._select_focus_assets(asset_snapshots, limit=4 if verbosity == "detailed" else 3)
        asset_lines = [self._render_asset_focus_line(asset) for asset in focus_assets]
        risk_line = self._extract_portfolio_value(portfolio_plan, "risk_watch") or "ติดตาม VIX, bond yield และแนวรับของดัชนีหลัก"
        action_line = self._extract_portfolio_value(portfolio_plan, "action_plan") or "ค่อย ๆ จัดพอร์ตตาม profile และหลีกเลี่ยงการไล่ราคา"
        news_lines = self._format_fallback_news(payload.get("news_headlines"))
        research_lines = self._format_fallback_research(payload.get("research_highlights"))

        if verbosity == "short":
            extra_line = f"ข้อมูลเสริม: {research_lines.splitlines()[0]}\n" if research_lines else ""
            return (
                "สรุปย่อจากระบบสำรอง\n"
                f"{market_view}\n{profile_line}\n{confidence_line}\nพอร์ตแนะนำ: {allocations}\nแนวทางวันนี้: {action_line}\n"
                f"รีบาลานซ์พอร์ตจริง: {portfolio_review_brief or 'ยังไม่มีสัญญาณรีบาลานซ์จากพอร์ตจริง'}\n"
                f"ตัวที่น่าจับตา: {' | '.join(asset_lines[:2])}\nความเสี่ยงหลัก: {risk_line}\n"
                f"{extra_line}"
                "หมายเหตุ: ใช้เพื่อวางแผนพอร์ต ไม่ใช่การรับประกันผลตอบแทน"
            )

        sections = [
            "สรุปจากระบบสำรอง",
            f"มุมมองตลาด\n{market_view}\n{macro_line}\n{confidence_line}",
            f"โปรไฟล์ผู้ลงทุน\n{profile_line}",
            f"แผนจัดพอร์ต\n{allocations}",
            f"Current Portfolio\n{portfolio_snapshot_lines}",
            f"Rebalance Review\n{portfolio_review_lines}",
            "สินทรัพย์ที่ควรโฟกัส\n" + "\n".join(f"- {line}" for line in asset_lines),
            f"ความเสี่ยงที่ต้องติดตาม\n- {risk_line}",
            f"แผนปฏิบัติการวันนี้\n- {action_line}",
        ]
        if news_lines and verbosity == "detailed":
            sections.insert(4, f"ข่าวที่ต้องติดตาม\n{news_lines}")
        if research_lines:
            insert_at = 5 if verbosity == "detailed" and news_lines else 4
            sections.insert(insert_at, f"ข้อมูลวิจัยเว็บล่าสุด\n{research_lines}")
        sections.append("หมายเหตุ: ข้อมูลนี้ใช้เพื่อการศึกษาและการจัดพอร์ต ไม่ใช่การรับประกันผลตอบแทน")
        return "\n\n".join(sections)

    def _compose_interesting_alerts(
        self,
        *,
        payload: Mapping[str, Any],
        rankings: Sequence[RankedAsset],
        news_impacts: Sequence[NewsImpact],
        risk_score: RiskScoreAssessment,
        vix_threshold: float,
        risk_score_threshold: float,
        opportunity_score_threshold: float,
        news_impact_threshold: float,
    ) -> list[InterestingAlert]:
        alerts: list[InterestingAlert] = []
        macro_context = payload.get("macro_context", {})
        portfolio_plan = payload.get("portfolio_plan", {})

        vix = self._as_float(macro_context.get("vix") if isinstance(macro_context, Mapping) else None)
        if (vix is not None and vix >= vix_threshold) or risk_score.score >= risk_score_threshold:
            reason_text = " | ".join(risk_score.reasons[:3]) if risk_score.reasons else "ความผันผวนเพิ่มขึ้น"
            alerts.append(
                InterestingAlert(
                    key=f"risk:{risk_score.level}:{int(round(risk_score.score))}",
                    severity="warning" if risk_score.level in {"elevated", "high"} else "critical",
                    text=(
                        f"{self._format_badged_title('🟠 ระวัง', 'ความเสี่ยงตลาด')}\n"
                        f"- ภาพรวม: คะแนนความเสี่ยง {risk_score.score:.1f}/10 | ระดับ {self._humanize_risk_level(risk_score.level)}\n"
                        f"- เหตุผล: {self._extract_portfolio_value(portfolio_plan, 'risk_watch') or reason_text}\n"
                        f"- Action: {self._extract_portfolio_value(portfolio_plan, 'action_plan') or 'เพิ่มเงินสดและลดสินทรัพย์เสี่ยงบางส่วน'}"
                    ),
                )
            )

        for impact in news_impacts:
            if impact.impact_score < news_impact_threshold:
                continue
            if impact.sentiment == "negative":
                prefix = "Macro News Alert"
                action = "ติดตามผลต่อพอร์ตและหลีกเลี่ยงการไล่ความเสี่ยงเพิ่ม"
            else:
                prefix = "Opportunity News Alert"
                action = "ติดตามว่าสัญญาณเชิงบวกนี้ถูกยืนยันต่อด้วย trend และ volume หรือไม่"
            alerts.append(
                InterestingAlert(
                    key=f"news:{impact.sentiment}:{self._slugify(impact.title)}",
                    severity="info" if impact.sentiment == "positive" else "warning",
                    text=(
                        f"{self._format_badged_title('✅ ยืนยัน' if impact.sentiment == 'positive' else '🟠 ระวัง', 'ข่าวหนุนโอกาส' if impact.sentiment == 'positive' else 'ข่าวมหภาค')}\n"
                        f"- ภาพรวม: {impact.title}\n"
                        f"- ผลกระทบ: score {impact.impact_score:.1f} | หมวด {self._humanize_related_bucket(impact.related_bucket)}\n"
                        f"- เหตุผล: {impact.rationale}\n"
                        f"- Action: {action}"
                    ),
                )
            )

        for ranked in rankings[:3]:
            if ranked.score >= opportunity_score_threshold and ranked.stance == "watch-long":
                alerts.append(
                    InterestingAlert(
                        key=f"rank:long:{ranked.asset}:{int(round(ranked.score * 10))}",
                        severity="info",
                        text=(
                            f"{self._format_badged_title('🔎 จับตา', 'สินทรัพย์เด่น')}\n"
                            f"- ภาพรวม: {ranked.label} | score {ranked.score:.2f}\n"
                            f"- เหตุผล: {ranked.rationale}\n"
                            f"- Action: ทยอยเพิ่มน้ำหนักได้ถ้าสอดคล้องกับโปรไฟล์และ market regime"
                        ),
                    )
                )
            elif ranked.score <= -opportunity_score_threshold and ranked.stance == "avoid":
                alerts.append(
                    InterestingAlert(
                        key=f"rank:avoid:{ranked.asset}:{int(round(abs(ranked.score) * 10))}",
                        severity="warning",
                        text=(
                            f"{self._format_badged_title('🟠 ระวัง', 'สินทรัพย์อ่อนแรง')}\n"
                            f"- ภาพรวม: {ranked.label} | score {ranked.score:.2f}\n"
                            f"- เหตุผล: {ranked.rationale}\n"
                            f"- Action: ชะลอการเพิ่มพอร์ตหรือรอให้สัญญาณกลับตัวชัดขึ้น"
                        ),
                    )
                )

        unique: dict[str, InterestingAlert] = {}
        for alert in alerts:
            unique.setdefault(alert.key, alert)
        return list(unique.values())

    async def _gather_context(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None,
        news_limit: int,
        history_period: str,
        history_interval: str,
        history_limit: int,
        research_query: str | None = None,
    ) -> tuple[
        list[NewsArticle],
        dict[str, AssetQuote | None],
        dict[str, TrendAssessment],
        dict[str, float | None],
        list[ResearchFinding],
    ]:
        news_task = news_client.fetch_latest_macro_news(limit=news_limit, when="1d")
        snapshot_task = market_data_client.get_core_market_snapshot()
        history_task = market_data_client.get_core_market_history(period=history_period, interval=history_interval, limit=history_limit)
        macro_task = market_data_client.get_macro_context()
        research_task = self._gather_research_findings(
            research_client=research_client,
            research_query=research_query,
            limit=min(news_limit, 4),
        )
        news, market_data, market_history, macro_context, research_findings = await asyncio.gather(
            self._safe_async_call(news_task, default=[], source_name="news"),
            self._safe_async_call(snapshot_task, default={}, source_name="market_snapshot"),
            self._safe_async_call(history_task, default={}, source_name="market_history"),
            self._safe_async_call(
                macro_task,
                default={"vix": None, "tnx": None, "cpi_yoy": None},
                source_name="macro_context",
            ),
            self._safe_async_call(research_task, default=[], source_name="research"),
        )

        news = self._guard_news_articles(news)
        market_data = self._guard_market_snapshot(market_data)
        macro_context = self._guard_macro_context(macro_context)
        research_findings = self._guard_research_findings(research_findings)

        trends: dict[str, TrendAssessment] = {}
        for asset_name, bars in market_history.items():
            frame = self._bars_to_frame(bars)
            if frame.empty:
                continue
            try:
                quote = market_data.get(asset_name)
                trends[asset_name] = evaluate_trend(frame, ticker=quote.ticker if quote else None)
            except Exception as exc:
                logger.warning("Failed to evaluate trend for {}: {}", asset_name, exc)
        return list(news), dict(market_data), trends, dict(macro_context), list(research_findings)

    async def _safe_async_call(self, awaitable: Any, *, default: Any, source_name: str) -> Any:
        try:
            return await awaitable
        except Exception as exc:
            logger.warning("External provider degraded for {}: {}", source_name, exc)
            log_event("external_provider_degraded", source=source_name, error=str(exc))
            return default

    async def _gather_portfolio_snapshot(
        self,
        *,
        market_data_client: MarketDataClient,
        holdings: Sequence[PortfolioHolding],
    ) -> dict[str, Any]:
        normalized_holdings = [holding for holding in holdings if holding.normalized_ticker and holding.quantity > 0]
        if not normalized_holdings:
            return {}

        live_tickers = [holding.normalized_ticker for holding in normalized_holdings if not holding.is_cash]
        quotes = await self._safe_async_call(
            market_data_client.get_latest_prices(live_tickers),
            default={},
            source_name="portfolio_quotes",
        )
        fundamentals_results = await asyncio.gather(
            *[market_data_client.get_fundamentals(ticker) for ticker in live_tickers],
            return_exceptions=True,
        )
        fundamentals_map: dict[str, StockFundamentals | None] = {}
        for ticker, result in zip(live_tickers, fundamentals_results, strict=False):
            fundamentals_map[ticker] = None if isinstance(result, Exception) else result

        holding_reviews: list[PortfolioHoldingReview] = []
        provisional_values: list[tuple[PortfolioHolding, float, AssetQuote | None, StockFundamentals | None]] = []
        total_market_value = 0.0
        for holding in normalized_holdings:
            quote = quotes.get(holding.normalized_ticker) if isinstance(quotes, Mapping) else None
            fundamentals = fundamentals_map.get(holding.normalized_ticker)
            if holding.is_cash:
                current_price = 1.0
            else:
                current_price = quote.price if isinstance(quote, AssetQuote) else None
                if current_price is None and holding.avg_cost is not None:
                    current_price = holding.avg_cost
            market_value = float(holding.quantity) * float(current_price or 0.0)
            total_market_value += max(0.0, market_value)
            provisional_values.append((holding, market_value, quote if isinstance(quote, AssetQuote) else None, fundamentals))

        for holding, market_value, _quote, fundamentals in provisional_values:
            sector = fundamentals.sector if isinstance(fundamentals, StockFundamentals) else None
            category = infer_allocation_mix_category(symbol=holding.normalized_ticker, sector=sector)
            cost_basis = float(holding.quantity) * float(holding.avg_cost) if holding.avg_cost is not None else None
            pnl_pct = None
            if cost_basis is not None and cost_basis > 0:
                pnl_pct = (market_value - cost_basis) / cost_basis
            current_weight_pct = (market_value / total_market_value) * 100.0 if total_market_value > 0 else 0.0
            holding_reviews.append(
                PortfolioHoldingReview(
                    ticker=holding.normalized_ticker,
                    category=category,
                    market_value=round(market_value, 2),
                    cost_basis=round(cost_basis, 2) if cost_basis is not None else None,
                    unrealized_pnl_pct=self._as_float(pnl_pct),
                    current_weight_pct=round(current_weight_pct, 1),
                    note=holding.note,
                )
            )

        return {
            "total_market_value": round(total_market_value, 2),
            "holdings": [
                {
                    "ticker": item.ticker,
                    "category": item.category,
                    "market_value": item.market_value,
                    "cost_basis": item.cost_basis,
                    "unrealized_pnl_pct": self._round_optional(item.unrealized_pnl_pct, 4),
                    "current_weight_pct": item.current_weight_pct,
                    "note": item.note,
                }
                for item in holding_reviews
            ],
        }

    def _build_portfolio_review_data(
        self,
        *,
        allocation_plan: PortfolioAllocationPlan,
        portfolio_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(portfolio_snapshot, Mapping):
            return {}
        holdings_raw = portfolio_snapshot.get("holdings")
        if not isinstance(holdings_raw, list):
            return {}
        holdings: list[PortfolioHoldingReview] = []
        for item in holdings_raw:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "").strip().upper()
            category = str(item.get("category") or "").strip()
            market_value = self._as_float(item.get("market_value"))
            current_weight_pct = self._as_float(item.get("current_weight_pct"))
            if not ticker or category not in {"cash", "gold", "core_etf", "growth", "defensive"} or market_value is None or current_weight_pct is None:
                continue
            holdings.append(
                PortfolioHoldingReview(
                    ticker=ticker,
                    category=category,  # type: ignore[arg-type]
                    market_value=market_value,
                    cost_basis=self._as_float(item.get("cost_basis")),
                    unrealized_pnl_pct=self._as_float(item.get("unrealized_pnl_pct")),
                    current_weight_pct=current_weight_pct,
                    note=str(item.get("note")).strip() if item.get("note") is not None else None,
                )
            )
        review = build_portfolio_rebalance_review(allocation_plan=allocation_plan, holdings=holdings)
        if review is None:
            return {}
        return {
            "total_market_value": review.total_market_value,
            "holdings_count": review.holdings_count,
            "action_summary": review.action_summary,
            "buckets": [
                {
                    "category": bucket.category,
                    "target_pct": bucket.target_pct,
                    "current_pct": bucket.current_pct,
                    "drift_pct": bucket.drift_pct,
                    "action": bucket.action,
                    "top_holdings": list(bucket.top_holdings),
                }
                for bucket in review.buckets
            ],
        }

    async def _gather_research_findings(
        self,
        *,
        research_client: ResearchClient | None,
        research_query: str | None,
        limit: int,
    ) -> list[ResearchFinding]:
        if research_client is None or not research_client.available():
            return []
        query = (research_query or "").strip()
        if not query:
            return []
        try:
            return await research_client.search_market_context(query=query, limit=limit)
        except Exception as exc:
            logger.warning("Failed to gather research findings: {}", exc)
            return []

    def _analyze_sector_rotation(
        self,
        *,
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
    ) -> list[SectorRotationSignal]:
        signals: list[SectorRotationSignal] = []
        for asset, sector in SECTOR_ETF_LABELS.items():
            trend = trends.get(asset)
            quote = market_data.get(asset)
            if trend is None:
                continue
            stance = "neutral"
            if trend.direction == "uptrend" and trend.score >= 2.0:
                stance = "overweight"
            elif trend.direction == "downtrend" or trend.score <= -1.5:
                stance = "underweight"
            signals.append(
                SectorRotationSignal(
                    sector=sector,
                    asset=asset,
                    ticker=quote.ticker if quote else trend.ticker,
                    trend_direction=trend.direction,
                    trend_score=round(trend.score, 2),
                    rsi=round(trend.rsi, 2) if trend.rsi is not None else None,
                    stance=stance,
                    rationale=tuple(trend.reasons[:3]),
                )
            )
        signals.sort(key=lambda item: item.trend_score, reverse=True)
        return signals

    async def _analyze_sector_breadth(
        self,
        *,
        market_data_client: MarketDataClient,
        rotation: Sequence[SectorRotationSignal],
    ) -> list[SectorBreadthInsight]:
        if not rotation:
            return []
        universe = await market_data_client.get_dynamic_stock_universe(indexes=("sp500", "nasdaq100"), max_members=220)
        sector_names = {signal.sector for signal in rotation}
        sector_universe = {
            alias: member
            for alias, member in universe.items()
            if member.sector in sector_names
        }
        if not sector_universe:
            return []
        histories = await market_data_client.get_stock_universe_history(
            stock_universe=sector_universe,
            period="3mo",
            interval="1d",
            limit=90,
        )
        constituent_trends: dict[str, TrendAssessment] = {}
        for asset_name, bars in histories.items():
            frame = self._bars_to_frame(bars)
            if frame.empty:
                continue
            try:
                constituent_trends[asset_name] = evaluate_trend(frame, ticker=sector_universe[asset_name].ticker)
            except Exception as exc:
                logger.warning("Failed to evaluate constituent trend for {}: {}", asset_name, exc)

        insights: list[SectorBreadthInsight] = []
        for signal in rotation:
            members = [alias for alias, member in sector_universe.items() if member.sector == signal.sector]
            sector_trends = [constituent_trends[alias] for alias in members if alias in constituent_trends]
            if not sector_trends:
                continue
            advancers = sum(1 for trend in sector_trends if trend.direction == "uptrend" and trend.score >= 1.0)
            decliners = sum(1 for trend in sector_trends if trend.direction == "downtrend" or trend.score <= -1.0)
            constituent_count = len(sector_trends)
            participation_ratio = advancers / constituent_count if constituent_count else 0.0
            average_trend_score = sum(trend.score for trend in sector_trends) / constituent_count if constituent_count else 0.0
            equal_weight_confirmed = False
            breadth_label = "mixed"
            if signal.stance == "overweight":
                equal_weight_confirmed = participation_ratio >= 0.5 and average_trend_score >= 0.8
                breadth_label = "confirmed" if equal_weight_confirmed else "narrow leadership"
            elif signal.stance == "underweight":
                equal_weight_confirmed = (decliners / constituent_count) >= 0.45 and average_trend_score <= -0.6
                breadth_label = "confirmed weakness" if equal_weight_confirmed else "false weakness"
            else:
                equal_weight_confirmed = participation_ratio >= 0.4 and average_trend_score >= 0.2
                breadth_label = "balanced"
            insights.append(
                SectorBreadthInsight(
                    sector=signal.sector,
                    ticker=signal.ticker,
                    constituent_count=constituent_count,
                    advancers=advancers,
                    decliners=decliners,
                    participation_ratio=round(participation_ratio, 2),
                    average_trend_score=round(average_trend_score, 2),
                    equal_weight_confirmed=equal_weight_confirmed,
                    breadth_label=breadth_label,
                )
            )
        insights.sort(key=lambda item: (item.equal_weight_confirmed, item.participation_ratio, item.average_trend_score), reverse=True)
        return insights

    async def _analyze_market_breadth_diffusion(
        self,
        *,
        market_data_client: MarketDataClient,
    ) -> MarketBreadthDiffusion | None:
        stock_universe = await market_data_client.get_dynamic_stock_universe(
            indexes=("sp500", "nasdaq100"),
            max_members=220,
        )
        if not stock_universe:
            return None
        histories = await market_data_client.get_stock_universe_history(
            stock_universe=stock_universe,
            period="3mo",
            interval="1d",
            limit=90,
        )
        advancers = 0
        decliners = 0
        neutral_count = 0
        trend_scores: list[float] = []
        for alias, bars in histories.items():
            frame = self._bars_to_frame(bars)
            if frame.empty:
                continue
            member = stock_universe.get(alias)
            try:
                trend = evaluate_trend(frame, ticker=member.ticker if member else alias.upper())
            except Exception as exc:
                logger.warning("Failed to evaluate breadth trend for {}: {}", alias, exc)
                continue
            trend_scores.append(trend.score)
            if trend.direction == "uptrend" and trend.score >= 1.0:
                advancers += 1
            elif trend.direction == "downtrend" or trend.score <= -1.0:
                decliners += 1
            else:
                neutral_count += 1
        participant_count = advancers + decliners + neutral_count
        if participant_count <= 0:
            return None
        advancing_ratio = advancers / participant_count
        declining_ratio = decliners / participant_count
        diffusion_score = (advancers - decliners) / participant_count
        average_trend_score = (sum(trend_scores) / len(trend_scores)) if trend_scores else 0.0
        breadth_label = "mixed tape"
        rally_confirmed = False
        if diffusion_score >= 0.2 and advancing_ratio >= 0.55 and average_trend_score >= 0.4:
            breadth_label = "broad rally"
            rally_confirmed = True
        elif diffusion_score >= 0.05 and advancing_ratio >= 0.45 and average_trend_score >= 0.1:
            breadth_label = "constructive but selective"
            rally_confirmed = True
        elif diffusion_score <= -0.2 and declining_ratio >= 0.5 and average_trend_score <= -0.4:
            breadth_label = "broad selloff"
        elif diffusion_score < 0.05 and average_trend_score > 0:
            breadth_label = "fragile rally"
        elif diffusion_score < -0.05 and average_trend_score < 0:
            breadth_label = "risk-off tape"
        return MarketBreadthDiffusion(
            participant_count=participant_count,
            advancers=advancers,
            decliners=decliners,
            neutral_count=neutral_count,
            advancing_ratio=round(advancing_ratio, 2),
            declining_ratio=round(declining_ratio, 2),
            diffusion_score=round(diffusion_score, 2),
            average_trend_score=round(average_trend_score, 2),
            breadth_label=breadth_label,
            rally_confirmed=rally_confirmed,
        )

    async def _analyze_index_leadership_divergence(
        self,
        *,
        market_data_client: MarketDataClient,
    ) -> list[IndexLeadershipDivergence]:
        pairs = (
            ("S&P 500 breadth", "SPY", "RSP"),
            ("Technology breadth", "QQQ", "RYT"),
        )
        tickers = [ticker for _, cap_weight, equal_weight in pairs for ticker in (cap_weight, equal_weight)]
        histories = await asyncio.gather(
            *[market_data_client.get_history(ticker, period="6mo", interval="1d", limit=180) for ticker in tickers],
            return_exceptions=True,
        )
        history_map: dict[str, list[OhlcvBar]] = {}
        for ticker, result in zip(tickers, histories, strict=False):
            history_map[ticker] = [] if isinstance(result, Exception) else result
        signals: list[IndexLeadershipDivergence] = []
        for label, cap_weight_ticker, equal_weight_ticker in pairs:
            cap_frame = self._bars_to_frame(history_map.get(cap_weight_ticker, []))
            equal_frame = self._bars_to_frame(history_map.get(equal_weight_ticker, []))
            if cap_frame.empty or equal_frame.empty:
                continue
            try:
                cap_trend = evaluate_trend(cap_frame, ticker=cap_weight_ticker)
                equal_trend = evaluate_trend(equal_frame, ticker=equal_weight_ticker)
            except Exception as exc:
                logger.warning("Failed to evaluate divergence pair {} / {}: {}", cap_weight_ticker, equal_weight_ticker, exc)
                continue
            short_return_spread = self._compute_trailing_return(cap_frame, window=21) - self._compute_trailing_return(equal_frame, window=21)
            medium_return_spread = self._compute_trailing_return(cap_frame, window=63) - self._compute_trailing_return(equal_frame, window=63)
            divergence_label = "balanced"
            severity = "info"
            if short_return_spread >= 0.04 and medium_return_spread >= 0.015 and (cap_trend.score - equal_trend.score) >= 0.5:
                divergence_label = "narrow leadership"
                severity = "warning"
            elif short_return_spread <= -0.03 and medium_return_spread <= -0.03:
                divergence_label = "equal-weight leadership"
            signals.append(
                IndexLeadershipDivergence(
                    label=label,
                    cap_weight_ticker=cap_weight_ticker,
                    equal_weight_ticker=equal_weight_ticker,
                    short_return_spread=round(short_return_spread, 4),
                    medium_return_spread=round(medium_return_spread, 4),
                    cap_weight_score=round(cap_trend.score, 2),
                    equal_weight_score=round(equal_trend.score, 2),
                    divergence_label=divergence_label,
                    severity=severity,
                )
            )
        return signals

    def _evaluate_pre_earnings_risk(
        self,
        *,
        ticker: str,
        event: EarningsEvent,
        expectation: AnalystExpectationProfile | None,
        fundamentals: StockFundamentals | None,
        market_breadth: MarketBreadthDiffusion | None,
        divergence_signals: Sequence[IndexLeadershipDivergence],
    ) -> PreEarningsRiskSignal | None:
        risk_score = 0.0
        rationale: list[str] = []
        forward_pe = fundamentals.forward_pe if fundamentals is not None else None
        revenue_growth_estimate = expectation.revenue_growth_estimate_current_q if expectation is not None else None
        eps_growth_estimate = expectation.eps_growth_estimate_current_q if expectation is not None else None
        if forward_pe is not None:
            if forward_pe >= 30:
                risk_score += 1.4
                rationale.append(f"forward PE สูง {forward_pe:.1f}")
            elif forward_pe >= 24:
                risk_score += 0.7
                rationale.append(f"valuation ตึง {forward_pe:.1f}")
        if revenue_growth_estimate is not None and revenue_growth_estimate >= 0.12:
            risk_score += 0.9
            rationale.append(f"ตลาดคาด revenue growth สูง {revenue_growth_estimate:+.1%}")
        if eps_growth_estimate is not None and eps_growth_estimate >= 0.18:
            risk_score += 0.9
            rationale.append(f"ตลาดคาด EPS growth สูง {eps_growth_estimate:+.1%}")
        divergence_label = None
        sector_name = (fundamentals.sector or "").casefold() if fundamentals is not None and fundamentals.sector else ""
        for signal in divergence_signals:
            if signal.divergence_label != "narrow leadership":
                continue
            if sector_name == "technology" and signal.cap_weight_ticker == "QQQ":
                divergence_label = signal.divergence_label
                risk_score += 0.8
                rationale.append("tech leadership แคบเมื่อเทียบ equal-weight")
            elif signal.cap_weight_ticker == "SPY":
                divergence_label = signal.divergence_label
                risk_score += 0.5
                rationale.append("ตลาดนำโดยหุ้นใหญ่ไม่กี่ตัว")
        market_breadth_label = market_breadth.breadth_label if market_breadth is not None else None
        if market_breadth is not None and not market_breadth.rally_confirmed:
            risk_score += 0.8
            rationale.append(f"market breadth ไม่หนุน ({market_breadth.breadth_label})")
        if risk_score < 2.0:
            return None
        return PreEarningsRiskSignal(
            ticker=ticker,
            earnings_at=event.earnings_at,
            risk_score=round(risk_score, 2),
            forward_pe=forward_pe,
            revenue_growth_estimate=revenue_growth_estimate,
            eps_growth_estimate=eps_growth_estimate,
            market_breadth_label=market_breadth_label,
            divergence_label=divergence_label,
            rationale=tuple(dict.fromkeys(rationale)),
        )

    async def _rank_earnings_setups(
        self,
        *,
        market_data_client: MarketDataClient,
        candidates: Sequence[StockCandidate],
        days_ahead: int,
        market_breadth: MarketBreadthDiffusion | None,
        divergence_signals: Sequence[IndexLeadershipDivergence],
    ) -> list[EarningsSetupCandidate]:
        if not candidates:
            return []
        candidate_map = {candidate.ticker: candidate for candidate in candidates}
        earnings = await market_data_client.get_earnings_calendar(candidate_map.keys(), days_ahead=days_ahead)
        if not earnings:
            return []
        expectation_profiles = await market_data_client.get_analyst_expectation_profiles(earnings.keys())
        setups: list[EarningsSetupCandidate] = []
        for ticker, event in earnings.items():
            candidate = candidate_map.get(ticker)
            if candidate is None:
                continue
            expectation = expectation_profiles.get(ticker)
            setups.append(
                self._evaluate_earnings_setup_candidate(
                    candidate=candidate,
                    event=event,
                    expectation=expectation,
                    market_breadth=market_breadth,
                    divergence_signals=divergence_signals,
                )
            )
        setups.sort(key=lambda item: item.setup_score, reverse=True)
        return setups

    def _evaluate_earnings_setup_candidate(
        self,
        *,
        candidate: StockCandidate,
        event: EarningsEvent,
        expectation: AnalystExpectationProfile | None,
        market_breadth: MarketBreadthDiffusion | None,
        divergence_signals: Sequence[IndexLeadershipDivergence],
    ) -> EarningsSetupCandidate:
        setup_score = 0.0
        rationale: list[str] = []
        trend_signal = candidate.trend_direction
        valuation_signal = "neutral"
        expectation_signal = "moderate"

        if candidate.trend_direction == "uptrend":
            setup_score += 1.1
            rationale.append("trend ก่อนงบยังเป็นบวก")
        elif candidate.trend_direction == "downtrend":
            setup_score -= 0.8
            rationale.append("trend ก่อนงบยังอ่อน")

        if candidate.stance == "buy":
            setup_score += 0.4
        elif candidate.stance == "avoid":
            setup_score -= 0.4

        if candidate.composite_score >= 2.3:
            setup_score += 0.9
            rationale.append("คุณภาพรวมก่อนงบเด่นกว่าค่าเฉลี่ย")
        elif candidate.composite_score >= 1.5:
            setup_score += 0.5

        if candidate.revenue_growth is not None and candidate.revenue_growth >= 0.08:
            setup_score += 0.3
        elif candidate.revenue_growth is not None and candidate.revenue_growth <= 0:
            setup_score -= 0.3

        if candidate.earnings_growth is not None and candidate.earnings_growth >= 0.10:
            setup_score += 0.3
        elif candidate.earnings_growth is not None and candidate.earnings_growth <= 0:
            setup_score -= 0.3

        if candidate.forward_pe is not None:
            if candidate.forward_pe <= 24:
                valuation_signal = "reasonable"
                setup_score += 0.5
            elif candidate.forward_pe >= 32:
                valuation_signal = "stretched"
                setup_score -= 0.7

        if expectation is not None:
            rev_growth = expectation.revenue_growth_estimate_current_q
            eps_growth = expectation.eps_growth_estimate_current_q
            if rev_growth is not None and rev_growth >= 0.12:
                expectation_signal = "high"
                setup_score -= 0.3
                rationale.append(f"ตลาดคาด revenue สูง {rev_growth:+.1%}")
            if eps_growth is not None and eps_growth >= 0.18:
                expectation_signal = "high"
                setup_score -= 0.3
                rationale.append(f"ตลาดคาด EPS สูง {eps_growth:+.1%}")
            if (rev_growth is not None and rev_growth <= 0.08) and (eps_growth is not None and eps_growth <= 0.12):
                expectation_signal = "contained"
                setup_score += 0.3

        if market_breadth is not None and market_breadth.rally_confirmed:
            setup_score += 0.4
        elif market_breadth is not None:
            setup_score -= 0.4
            rationale.append(f"breadth ตลาดไม่แข็งแรง ({market_breadth.breadth_label})")

        if candidate.sector.casefold() == "technology":
            for signal in divergence_signals:
                if signal.cap_weight_ticker == "QQQ" and signal.divergence_label == "narrow leadership":
                    setup_score -= 0.4
                    rationale.append("tech leadership ยังแคบ")
                    break

        setup_label = "balanced setup"
        if setup_score >= 1.8:
            setup_label = "favorable setup"
        elif setup_score <= 0.3:
            setup_label = "fragile setup"
        return EarningsSetupCandidate(
            ticker=candidate.ticker,
            company_name=candidate.company_name,
            earnings_at=event.earnings_at,
            setup_score=round(setup_score, 2),
            setup_label=setup_label,
            valuation_signal=valuation_signal,
            expectation_signal=expectation_signal,
            trend_signal=trend_signal,
            rationale=tuple(dict.fromkeys(rationale))[:5] or ("setup ก่อนงบยังสมดุล",),
        )

    def _analyze_sector_rotation_persistence(
        self,
        rotation: Sequence[SectorRotationSignal],
        *,
        state_store: SectorRotationStateStore | None,
        min_streak: int,
        regime: Literal["daily", "intraday"],
        record_snapshot: bool,
    ) -> list[SectorPersistenceInsight]:
        if state_store is None:
            return []
        previous_snapshots = state_store.recent_snapshots(limit=max(4, min_streak + 2), regime=regime)
        insights: list[SectorPersistenceInsight] = []
        for signal in rotation:
            streak = 1
            prior_scores: list[float] = []
            changed_from: str | None = None
            for snapshot in reversed(previous_snapshots):
                previous = snapshot.sectors.get(signal.sector)
                if not isinstance(previous, Mapping):
                    break
                previous_stance = str(previous.get("stance") or "").strip() or None
                previous_score = self._as_float(previous.get("trend_score"))
                if previous_score is not None:
                    prior_scores.append(previous_score)
                if changed_from is None and previous_stance and previous_stance != signal.stance:
                    changed_from = previous_stance
                if previous_stance == signal.stance:
                    streak += 1
                else:
                    break
            average_score = sum(prior_scores) / len(prior_scores) if prior_scores else signal.trend_score
            score_delta = round(signal.trend_score - average_score, 2)
            if streak < max(2, min_streak) and changed_from is None:
                continue
            insights.append(
                SectorPersistenceInsight(
                    sector=signal.sector,
                    ticker=signal.ticker,
                    regime=regime,
                    stance=signal.stance,
                    streak=streak,
                    average_score=round(average_score, 2),
                    score_delta=score_delta,
                    changed_from=changed_from,
                )
            )
        if record_snapshot:
            state_store.append_snapshot([self._serialize_sector_rotation(item) for item in rotation], regime=regime)
        insights.sort(key=lambda item: (item.streak, abs(item.score_delta)), reverse=True)
        return insights

    def _record_sector_regime_snapshot(
        self,
        *,
        rotation: Sequence[SectorRotationSignal],
        breadth: Sequence[SectorBreadthInsight],
        market_breadth: MarketBreadthDiffusion | None,
        state_store: SectorRotationStateStore,
        regime: Literal["daily", "intraday"],
    ) -> None:
        breadth_map = {item.sector: item for item in breadth}
        payload: list[dict[str, Any]] = []
        for signal in rotation:
            item = self._serialize_sector_rotation(signal)
            breadth_item = breadth_map.get(signal.sector)
            if breadth_item is not None:
                item.update(self._serialize_sector_breadth(breadth_item))
            payload.append(item)
        state_store.append_snapshot(
            payload,
            regime=regime,
            market_breadth=self._serialize_market_breadth_diffusion(market_breadth),
        )

    def _analyze_sector_breadth_trend(
        self,
        *,
        state_store: SectorRotationStateStore | None,
        current_breadth: Sequence[SectorBreadthInsight],
        regime: Literal["daily", "intraday"],
    ) -> list[SectorBreadthTrend]:
        if state_store is None or not current_breadth:
            return []
        snapshots = state_store.recent_snapshots(limit=5, regime=regime)
        trends: list[SectorBreadthTrend] = []
        for item in current_breadth:
            history_scores: list[float] = []
            for snapshot in snapshots:
                sector_data = snapshot.sectors.get(item.sector)
                if not isinstance(sector_data, Mapping):
                    continue
                participation = self._as_float(sector_data.get("participation_ratio"))
                if participation is not None:
                    history_scores.append(participation)
            if history_scores and abs(history_scores[-1] - item.participation_ratio) < 1e-9:
                series = history_scores
            else:
                series = [*history_scores, item.participation_ratio]
            prior = series[:-1]
            average_prior = (sum(prior) / len(prior)) if prior else item.participation_ratio
            delta = round(item.participation_ratio - average_prior, 2)
            trend_label = "stable"
            if delta >= 0.08:
                trend_label = "broadening"
            elif delta <= -0.08:
                trend_label = "narrowing"
            trends.append(
                SectorBreadthTrend(
                    sector=item.sector,
                    ticker=item.ticker,
                    regime=regime,
                    current_participation_ratio=item.participation_ratio,
                    score_delta=delta,
                    sparkline=self._build_ascii_sparkline(series),
                    history_scores=tuple(round(score, 2) for score in series),
                    trend_label=trend_label,
                )
            )
        trends.sort(key=lambda item: abs(item.score_delta), reverse=True)
        return trends

    def _analyze_market_breadth_trend(
        self,
        *,
        state_store: SectorRotationStateStore | None,
        current_breadth: MarketBreadthDiffusion | None,
        regime: Literal["daily", "intraday"],
    ) -> MarketBreadthTrend | None:
        if state_store is None or current_breadth is None:
            return None
        snapshots = state_store.recent_snapshots(limit=6, regime=regime)
        history_scores: list[float] = []
        for snapshot in snapshots:
            breadth = snapshot.market_breadth
            if not isinstance(breadth, Mapping):
                continue
            score = self._as_float(breadth.get("diffusion_score"))
            if score is not None:
                history_scores.append(score)
        if history_scores and abs(history_scores[-1] - current_breadth.diffusion_score) < 1e-9:
            series = history_scores
        else:
            series = [*history_scores, current_breadth.diffusion_score]
        prior = series[:-1]
        average_prior = (sum(prior) / len(prior)) if prior else current_breadth.diffusion_score
        delta = round(current_breadth.diffusion_score - average_prior, 2)
        trend_label = "stable"
        if delta >= 0.12:
            trend_label = "broadening"
        elif delta <= -0.12:
            trend_label = "weakening"
        return MarketBreadthTrend(
            regime=regime,
            current_diffusion_score=current_breadth.diffusion_score,
            score_delta=delta,
            sparkline=self._build_ascii_sparkline([(score + 1.0) / 2.0 for score in series]),
            history_scores=tuple(round(score, 2) for score in series),
            trend_label=trend_label,
            breadth_label=current_breadth.breadth_label,
        )

    async def _build_sector_quality_benchmarks(
        self,
        *,
        market_data_client: MarketDataClient,
        fundamentals_map: Mapping[str, StockFundamentals | None],
    ) -> dict[str, dict[str, float]]:
        sectors = {
            (fundamentals.sector or "").strip()
            for fundamentals in fundamentals_map.values()
            if fundamentals is not None and (fundamentals.sector or "").strip()
        }
        if not sectors:
            return {}
        stock_universe = await market_data_client.get_dynamic_stock_universe(
            indexes=("sp500", "nasdaq100"),
            max_members=220,
        )
        if not stock_universe:
            return {}
        peer_universe = {
            alias: member
            for alias, member in stock_universe.items()
            if (member.sector or "").strip() in sectors
        }
        if not peer_universe:
            return {}
        peer_fundamentals = await market_data_client.get_stock_universe_fundamentals(peer_universe)
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for fundamentals in peer_fundamentals.values():
            if fundamentals is None or not (fundamentals.sector or "").strip():
                continue
            sector_key = fundamentals.sector.strip().casefold()
            for metric_name, metric_value in (
                ("revenue_qoq_change", fundamentals.revenue_qoq_change),
                ("operating_margin_qoq_change", fundamentals.operating_margin_qoq_change),
                ("free_cash_flow_qoq_change", fundamentals.free_cash_flow_qoq_change),
            ):
                numeric = self._as_float(metric_value)
                if numeric is not None:
                    grouped[sector_key][metric_name].append(numeric)
        benchmarks: dict[str, dict[str, float]] = {}
        for sector_key, metrics in grouped.items():
            sector_payload: dict[str, float] = {}
            for metric_name, values in metrics.items():
                if not values:
                    continue
                series = pd.Series(values, dtype="float64").dropna()
                if series.empty:
                    continue
                sector_payload[metric_name] = round(float(series.median()), 4)
            if sector_payload:
                benchmarks[sector_key] = sector_payload
        return benchmarks

    async def _gather_post_earnings_interpretations(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None,
        tickers: Sequence[str],
        lookback_days: int,
    ) -> list[EarningsInterpretation]:
        unique_tickers = [ticker.upper() for ticker in dict.fromkeys(tickers) if ticker.strip()]
        if not unique_tickers:
            return []
        results = await market_data_client.get_recent_earnings_results(unique_tickers, lookback_days=max(1, lookback_days))
        if not results:
            return []
        fundamentals_map = await asyncio.gather(
            *[market_data_client.get_fundamentals(ticker) for ticker in results.keys()],
            return_exceptions=True,
        )
        earnings_fundamentals: dict[str, StockFundamentals | None] = {}
        for ticker, result in zip(results.keys(), fundamentals_map, strict=False):
            earnings_fundamentals[ticker] = None if isinstance(result, Exception) else result
        sector_benchmarks = await self._build_sector_quality_benchmarks(
            market_data_client=market_data_client,
            fundamentals_map=earnings_fundamentals,
        )
        news_tasks = {
            ticker: news_client.fetch_stock_news(ticker, company_name=None, limit=3, when=f"{max(lookback_days, 7)}d")
            for ticker in results.keys()
        }
        news_results = await asyncio.gather(*news_tasks.values(), return_exceptions=True)
        news_map: dict[str, list[NewsArticle]] = {}
        for ticker, result in zip(news_tasks.keys(), news_results, strict=False):
            news_map[ticker] = [] if isinstance(result, Exception) else result

        research_map: dict[str, list[ResearchFinding]] = {}
        if research_client is not None and research_client.available():
            research_tasks = {
                ticker: research_client.search_earnings_call_context(ticker=ticker, company_name=None, limit=2)
                for ticker in results.keys()
            }
            research_results = await asyncio.gather(*research_tasks.values(), return_exceptions=True)
            for ticker, result in zip(research_tasks.keys(), research_results, strict=False):
                research_map[ticker] = [] if isinstance(result, Exception) else result
        else:
            research_map = {ticker: [] for ticker in results.keys()}

        interpretations = [
            self._interpret_recent_earnings_result(
                result,
                news_map.get(result.ticker, []),
                research_map.get(result.ticker, []),
                earnings_fundamentals.get(result.ticker),
                sector_benchmarks.get((earnings_fundamentals.get(result.ticker).sector or "").casefold())
                if earnings_fundamentals.get(result.ticker) is not None
                else None,
            )
            for result in results.values()
        ]
        filtered = [item for item in interpretations if item is not None]
        filtered.sort(key=lambda item: (abs(item.guidance_score), abs(item.surprise_pct or 0.0)), reverse=True)
        return filtered

    def _interpret_recent_earnings_result(
        self,
        result: RecentEarningsResult,
        related_news: Sequence[NewsArticle],
        research_findings: Sequence[ResearchFinding],
        fundamentals: StockFundamentals | None,
        sector_benchmark: Mapping[str, float] | None,
    ) -> EarningsInterpretation | None:
        guidance_score = 0
        management_tone_score = 0
        quality_score = 0.0
        positive_keywords = (
            "raises guidance",
            "boosts outlook",
            "strong guidance",
            "reaffirms guidance",
            "better-than-expected",
            "beats estimates",
            "record revenue",
            "upside",
        )
        negative_keywords = (
            "cuts guidance",
            "weak guidance",
            "soft outlook",
            "misses estimates",
            "warns",
            "lowered forecast",
            "slower demand",
            "margin pressure",
        )
        tone_positive_keywords = (
            "strong demand",
            "healthy pipeline",
            "margin expansion",
            "bookings strength",
            "confidence",
            "durable growth",
            "improving trends",
        )
        tone_negative_keywords = (
            "macro uncertainty",
            "demand pressure",
            "inventory correction",
            "weaker spending",
            "headwinds",
            "visibility remains limited",
            "margin compression",
        )
        one_off_negative_keywords = (
            "tax benefit",
            "asset sale",
            "one-time",
            "one off",
            "restructuring gain",
            "accounting benefit",
            "litigation gain",
            "fx benefit",
        )
        core_business_keywords = (
            "strong demand",
            "healthy pipeline",
            "backlog growth",
            "subscription growth",
            "pricing power",
            "volume growth",
            "market share gains",
            "margin expansion",
        )
        revenue_positive_keywords = (
            "revenue beat",
            "sales beat",
            "top-line beat",
            "better-than-expected revenue",
            "revenue above estimates",
        )
        revenue_negative_keywords = (
            "revenue miss",
            "sales miss",
            "top-line miss",
            "revenue below estimates",
        )
        margin_positive_keywords = (
            "margin expansion",
            "gross margin beat",
            "operating margin beat",
            "better margins",
        )
        margin_negative_keywords = (
            "margin compression",
            "gross margin miss",
            "operating margin miss",
            "weaker margins",
        )
        rationale: list[str] = []
        one_off_hits = 0
        core_hits = 0
        revenue_signal_score = 0
        margin_signal_score = 0
        revenue_expectation_gap_pct: float | None = None
        revenue_expectation_source: str | None = None
        for article in related_news[:3]:
            text = f"{article.title} {article.summary or ''}".casefold()
            if revenue_expectation_gap_pct is None:
                parsed_gap = self._extract_revenue_expectation_gap(text)
                if parsed_gap is not None:
                    revenue_expectation_gap_pct = parsed_gap
                    revenue_expectation_source = "news"
            for keyword in positive_keywords:
                if keyword in text:
                    guidance_score += 1
                    rationale.append(f"headline บวก: {keyword}")
                    break
            for keyword in negative_keywords:
                if keyword in text:
                    guidance_score -= 1
                    rationale.append(f"headline ลบ: {keyword}")
                    break
            for keyword in one_off_negative_keywords:
                if keyword in text:
                    one_off_hits += 1
                    rationale.append(f"one-off risk: {keyword}")
                    break
            for keyword in revenue_positive_keywords:
                if keyword in text:
                    revenue_signal_score += 1
                    rationale.append(f"revenue signal บวก: {keyword}")
                    break
            for keyword in revenue_negative_keywords:
                if keyword in text:
                    revenue_signal_score -= 1
                    rationale.append(f"revenue signal ลบ: {keyword}")
                    break
            for keyword in margin_positive_keywords:
                if keyword in text:
                    margin_signal_score += 1
                    rationale.append(f"margin signal บวก: {keyword}")
                    break
            for keyword in margin_negative_keywords:
                if keyword in text:
                    margin_signal_score -= 1
                    rationale.append(f"margin signal ลบ: {keyword}")
                    break
            for keyword in core_business_keywords:
                if keyword in text:
                    core_hits += 1
                    rationale.append(f"core business: {keyword}")
                    break
        for finding in research_findings[:2]:
            text = f"{finding.title} {finding.snippet}".casefold()
            if revenue_expectation_gap_pct is None:
                parsed_gap = self._extract_revenue_expectation_gap(text)
                if parsed_gap is not None:
                    revenue_expectation_gap_pct = parsed_gap
                    revenue_expectation_source = finding.provider or "research"
            for keyword in positive_keywords:
                if keyword in text:
                    guidance_score += 1
                    rationale.append(f"call summary บวก: {keyword}")
                    break
            for keyword in negative_keywords:
                if keyword in text:
                    guidance_score -= 1
                    rationale.append(f"call summary ลบ: {keyword}")
                    break
            for keyword in tone_positive_keywords:
                if keyword in text:
                    management_tone_score += 1
                    rationale.append(f"management tone บวก: {keyword}")
                    break
            for keyword in tone_negative_keywords:
                if keyword in text:
                    management_tone_score -= 1
                    rationale.append(f"management tone ลบ: {keyword}")
                    break
            for keyword in one_off_negative_keywords:
                if keyword in text:
                    one_off_hits += 1
                    rationale.append(f"one-off risk: {keyword}")
                    break
            for keyword in revenue_positive_keywords:
                if keyword in text:
                    revenue_signal_score += 1
                    rationale.append(f"revenue signal บวก: {keyword}")
                    break
            for keyword in revenue_negative_keywords:
                if keyword in text:
                    revenue_signal_score -= 1
                    rationale.append(f"revenue signal ลบ: {keyword}")
                    break
            for keyword in margin_positive_keywords:
                if keyword in text:
                    margin_signal_score += 1
                    rationale.append(f"margin signal บวก: {keyword}")
                    break
            for keyword in margin_negative_keywords:
                if keyword in text:
                    margin_signal_score -= 1
                    rationale.append(f"margin signal ลบ: {keyword}")
                    break
            for keyword in core_business_keywords:
                if keyword in text:
                    core_hits += 1
                    rationale.append(f"core business: {keyword}")
                    break

        surprise = result.surprise_pct
        if surprise is not None:
            if surprise >= 5.0:
                rationale.append(f"EPS beat {surprise:.1f}%")
                quality_score += min(2.0, surprise / 10.0)
            elif surprise <= -5.0:
                rationale.append(f"EPS miss {surprise:.1f}%")
                quality_score -= min(2.0, abs(surprise) / 10.0)

        if fundamentals is not None:
            if fundamentals.revenue_growth is not None:
                if fundamentals.revenue_growth >= 0.08:
                    quality_score += 0.9
                    rationale.append("revenue growth ยังหนุนจากธุรกิจหลัก")
                elif fundamentals.revenue_growth <= 0:
                    quality_score -= 0.8
                    rationale.append("รายได้ไม่ยืนยันคุณภาพของ earnings beat")
            if fundamentals.earnings_growth is not None:
                if fundamentals.earnings_growth >= 0.08:
                    quality_score += 0.8
                elif fundamentals.earnings_growth <= 0:
                    quality_score -= 0.6
            if fundamentals.profit_margin is not None and fundamentals.operating_margin is not None:
                if fundamentals.profit_margin >= 0.12 and fundamentals.operating_margin >= 0.12:
                    quality_score += 0.5
                elif fundamentals.profit_margin <= 0.05:
                    quality_score -= 0.5
        quality_score += min(1.5, core_hits * 0.4)
        quality_score -= min(2.0, one_off_hits * 0.8)

        guidance_signal = "neutral"
        stance = "neutral"
        management_tone = "neutral"
        revenue_signal = "neutral"
        revenue_qoq_change = None
        revenue_qoq_trend = "neutral"
        revenue_vs_sector_median = None
        revenue_relative_signal = "in-line"
        margin_signal = "neutral"
        margin_qoq_change = None
        margin_qoq_trend = "neutral"
        margin_vs_sector_median = None
        margin_relative_signal = "in-line"
        fcf_quality = "neutral"
        fcf_qoq_change = None
        fcf_qoq_trend = "neutral"
        fcf_vs_sector_median = None
        fcf_relative_signal = "in-line"
        earnings_quality_label = "mixed"
        one_off_risk = "low"
        if guidance_score > 0:
            guidance_signal = "positive"
        elif guidance_score < 0:
            guidance_signal = "negative"
        if management_tone_score > 0:
            management_tone = "positive"
        elif management_tone_score < 0:
            management_tone = "negative"
        if revenue_signal_score > 0:
            revenue_signal = "beat"
        elif revenue_signal_score < 0:
            revenue_signal = "miss"
        elif fundamentals is not None and fundamentals.revenue_growth is not None:
            revenue_signal = "beat" if fundamentals.revenue_growth >= 0.08 else "neutral" if fundamentals.revenue_growth > 0 else "miss"
        if margin_signal_score > 0:
            margin_signal = "beat"
        elif margin_signal_score < 0:
            margin_signal = "miss"
        elif fundamentals is not None and fundamentals.operating_margin is not None:
            margin_signal = "beat" if fundamentals.operating_margin >= 0.15 else "neutral" if fundamentals.operating_margin > 0.08 else "miss"
        if one_off_hits >= 2:
            one_off_risk = "high"
        elif one_off_hits == 1:
            one_off_risk = "medium"
        if fundamentals is not None and fundamentals.free_cash_flow_margin is not None:
            if fundamentals.free_cash_flow_margin >= 0.08:
                fcf_quality = "strong"
                quality_score += 0.7
            elif fundamentals.free_cash_flow_margin <= 0:
                fcf_quality = "weak"
                quality_score -= 0.8
        if fundamentals is not None:
            revenue_qoq_change = fundamentals.revenue_qoq_change
            revenue_qoq_trend = fundamentals.revenue_quality_trend or "neutral"
            margin_qoq_change = fundamentals.operating_margin_qoq_change
            margin_qoq_trend = fundamentals.margin_quality_trend or "neutral"
            fcf_qoq_change = fundamentals.free_cash_flow_qoq_change
            fcf_qoq_trend = fundamentals.fcf_quality_trend or "neutral"
            if revenue_qoq_change is not None:
                if revenue_qoq_change >= 0.03:
                    quality_score += 0.6
                    rationale.append(f"รายได้ QoQ เร่งขึ้น {revenue_qoq_change:+.1%}")
                elif revenue_qoq_change <= -0.03:
                    quality_score -= 0.6
                    rationale.append(f"รายได้ QoQ ชะลอลง {revenue_qoq_change:+.1%}")
            if margin_qoq_change is not None:
                if margin_qoq_change >= 0.01:
                    quality_score += 0.5
                    rationale.append(f"operating margin QoQ ดีขึ้น {margin_qoq_change:+.1%}")
                elif margin_qoq_change <= -0.01:
                    quality_score -= 0.5
                    rationale.append(f"operating margin QoQ แผ่วลง {margin_qoq_change:+.1%}")
            if fcf_qoq_change is not None:
                if fcf_qoq_change >= 0.08:
                    quality_score += 0.6
                    rationale.append(f"FCF QoQ เร่งขึ้น {fcf_qoq_change:+.1%}")
                elif fcf_qoq_change <= -0.08:
                    quality_score -= 0.6
                    rationale.append(f"FCF QoQ อ่อนลง {fcf_qoq_change:+.1%}")
        if revenue_expectation_gap_pct is not None:
            if revenue_expectation_gap_pct >= 0.02:
                quality_score += 0.6
                rationale.append(f"revenue beat analyst estimate {revenue_expectation_gap_pct:+.1%}")
            elif revenue_expectation_gap_pct <= -0.02:
                quality_score -= 0.6
                rationale.append(f"revenue missed analyst estimate {revenue_expectation_gap_pct:+.1%}")
        if sector_benchmark is not None:
            revenue_vs_sector_median = self._relative_to_sector_median(
                revenue_qoq_change,
                sector_benchmark.get("revenue_qoq_change"),
            )
            margin_vs_sector_median = self._relative_to_sector_median(
                margin_qoq_change,
                sector_benchmark.get("operating_margin_qoq_change"),
            )
            fcf_vs_sector_median = self._relative_to_sector_median(
                fcf_qoq_change,
                sector_benchmark.get("free_cash_flow_qoq_change"),
            )
            revenue_relative_signal = self._classify_relative_signal(revenue_vs_sector_median, threshold=0.03)
            margin_relative_signal = self._classify_relative_signal(margin_vs_sector_median, threshold=0.01)
            fcf_relative_signal = self._classify_relative_signal(fcf_vs_sector_median, threshold=0.08)
            if revenue_relative_signal == "above_sector":
                quality_score += 0.5
                rationale.append("รายได้ QoQ ดีกว่า sector median")
            elif revenue_relative_signal == "below_sector":
                quality_score -= 0.5
                rationale.append("รายได้ QoQ แย่กว่า sector median")
            if margin_relative_signal == "above_sector":
                quality_score += 0.4
                rationale.append("margin QoQ ดีกว่า sector median")
            elif margin_relative_signal == "below_sector":
                quality_score -= 0.4
                rationale.append("margin QoQ แย่กว่า sector median")
            if fcf_relative_signal == "above_sector":
                quality_score += 0.5
                rationale.append("FCF QoQ ดีกว่า sector median")
            elif fcf_relative_signal == "below_sector":
                quality_score -= 0.5
                rationale.append("FCF QoQ แย่กว่า sector median")

        if quality_score >= 2.2:
            earnings_quality_label = "core_business_strong"
        elif quality_score <= -1.0:
            earnings_quality_label = "one_off_or_low_quality"

        if (surprise is not None and surprise >= 5.0) or (
            (guidance_score >= 1 or management_tone_score >= 1) and (surprise or 0.0) >= 0.0
        ):
            stance = "bullish"
        elif (surprise is not None and surprise <= -5.0) or (
            (guidance_score <= -1 or management_tone_score <= -1) and (surprise or 0.0) <= 0.0
        ):
            stance = "bearish"

        if stance == "neutral" and abs(guidance_score) < 1 and abs(management_tone_score) < 1 and abs(surprise or 0.0) < 4.0:
            return None

        if result.reported_eps is not None and result.eps_estimate is not None:
            rationale.insert(0, f"reported EPS {result.reported_eps} vs est {result.eps_estimate}")
        return EarningsInterpretation(
            ticker=result.ticker,
            earnings_at=result.earnings_at,
            eps_estimate=result.eps_estimate,
            reported_eps=result.reported_eps,
            surprise_pct=result.surprise_pct,
            revenue_signal=revenue_signal,
            revenue_qoq_change=self._as_float(revenue_qoq_change),
            revenue_qoq_trend=revenue_qoq_trend,
            revenue_expectation_gap_pct=self._as_float(revenue_expectation_gap_pct),
            revenue_expectation_source=revenue_expectation_source,
            revenue_vs_sector_median=self._as_float(revenue_vs_sector_median),
            revenue_relative_signal=revenue_relative_signal,
            margin_signal=margin_signal,
            margin_qoq_change=self._as_float(margin_qoq_change),
            margin_qoq_trend=margin_qoq_trend,
            margin_vs_sector_median=self._as_float(margin_vs_sector_median),
            margin_relative_signal=margin_relative_signal,
            guidance_signal=guidance_signal,
            guidance_score=guidance_score,
            management_tone=management_tone,
            fcf_quality=fcf_quality,
            fcf_qoq_change=self._as_float(fcf_qoq_change),
            fcf_qoq_trend=fcf_qoq_trend,
            fcf_vs_sector_median=self._as_float(fcf_vs_sector_median),
            fcf_relative_signal=fcf_relative_signal,
            earnings_quality_score=round(quality_score, 2),
            earnings_quality_label=earnings_quality_label,
            one_off_risk=one_off_risk,
            stance=stance,
            rationale=tuple(dict.fromkeys(rationale))[:4] or ("ไม่มี headline guidance ชัดเจน",),
        )

    def _build_sector_rotation_snapshot_alert(
        self,
        *,
        leaders: Sequence[SectorRotationSignal],
        laggards: Sequence[SectorRotationSignal],
        persistence: Sequence[SectorPersistenceInsight],
        breadth: Sequence[SectorBreadthInsight],
        sector_breadth_trend: Sequence[SectorBreadthTrend],
        market_breadth: MarketBreadthDiffusion | None,
        market_breadth_trend: MarketBreadthTrend | None,
        leadership_divergence: Sequence[IndexLeadershipDivergence],
        news_impact: Any,
    ) -> InterestingAlert | None:
        if not leaders and not laggards and market_breadth is None:
            return None

        severity = "warning" if (market_breadth is not None and not market_breadth.rally_confirmed) or laggards else "info"
        breadth_map = {item.sector: item for item in breadth}
        persistence_map = {item.sector: item for item in persistence}
        breadth_trend_map = {item.sector: item for item in sector_breadth_trend}
        risk_lines = self._build_sector_rotation_risk_lines(
            breadth=breadth,
            sector_breadth_trend=sector_breadth_trend,
            leadership_divergence=leadership_divergence,
            persistence=persistence,
        )
        macro_line = self._build_sector_macro_note(
            news_impact=news_impact,
            leaders=leaders,
            laggards=laggards,
            market_breadth=market_breadth,
        )
        action_lines = self._build_sector_rotation_action_lines(
            leaders=leaders,
            laggards=laggards,
            market_breadth=market_breadth,
        )

        lines = ["สรุปการหมุนกลุ่มหุ้น"]
        if market_breadth is not None:
            lines.append(f"- ภาพรวมตลาด: {self._format_market_breadth_summary(market_breadth)}")
        if market_breadth_trend is not None:
            trend_line = self._format_market_breadth_trend_summary(market_breadth_trend)
            if trend_line:
                lines.append(f"- เทียบรอบก่อน: {trend_line}")

        if leaders:
            lines.extend(
                [
                    "",
                    "ผู้นำตลาด",
                    *[
                        f"- {self._format_sector_rotation_snapshot_signal(item, persistence_map, breadth_map, breadth_trend_map)}"
                        for item in leaders
                    ],
                ]
            )
        if laggards:
            lines.extend(
                [
                    "",
                    "กลุ่มอ่อนแรง",
                    *[
                        f"- {self._format_sector_rotation_snapshot_signal(item, persistence_map, breadth_map, breadth_trend_map)}"
                        for item in laggards
                    ],
                ]
            )
        if risk_lines:
            lines.extend(["", "จุดที่ต้องระวัง", *[f"- {line}" for line in risk_lines]])
        if macro_line:
            lines.extend(["", "ธีมข่าวที่เกี่ยวข้อง", f"- {macro_line}"])
        if action_lines:
            lines.extend(["", "Action", *[f"- {line}" for line in action_lines]])

        leaders_key = "-".join(item.ticker or item.asset for item in leaders) or "none"
        laggards_key = "-".join(item.ticker or item.asset for item in laggards) or "none"
        return InterestingAlert(
            key=f"sector:snapshot:{leaders_key}:{laggards_key}",
            severity=severity,
            text="\n".join(lines),
        )

    def _build_sector_rotation_risk_lines(
        self,
        *,
        breadth: Sequence[SectorBreadthInsight],
        sector_breadth_trend: Sequence[SectorBreadthTrend],
        leadership_divergence: Sequence[IndexLeadershipDivergence],
        persistence: Sequence[SectorPersistenceInsight],
    ) -> list[str]:
        lines: list[str] = []
        breadth_trend_map = {item.sector: item for item in sector_breadth_trend}
        for item in breadth:
            if item.equal_weight_confirmed or item.constituent_count < 3:
                continue
            trend = breadth_trend_map.get(item.sector)
            suffix = ""
            if trend is not None and abs(trend.score_delta) >= 0.08:
                suffix = f" | เทียบรอบก่อน{self._translate_trend_label(trend.trend_label)} {trend.score_delta:+.0%}"
            lines.append(
                f"{item.sector} ({item.ticker}) การยืนยันในกลุ่มยังแคบ | แรงกระจาย {item.advancers}/{item.constituent_count} ({item.participation_ratio:.0%}){suffix}"
            )
        for signal in leadership_divergence:
            if signal.divergence_label != "narrow leadership":
                continue
            lines.append(
                f"{signal.label}: {signal.cap_weight_ticker} นำ {signal.equal_weight_ticker} แบบกระจุกตัว | spread 1 เดือน {signal.short_return_spread:+.1%}"
            )
        for insight in persistence:
            if insight.changed_from is None:
                continue
            lines.append(
                f"{insight.sector} ({insight.ticker}) เปลี่ยนจาก {self._translate_sector_stance(insight.changed_from)} เป็น {self._translate_sector_stance(insight.stance)} | delta {insight.score_delta:+.2f}"
            )
        return list(dict.fromkeys(lines))[:3]

    def _build_sector_macro_note(
        self,
        *,
        news_impact: Any,
        leaders: Sequence[SectorRotationSignal],
        laggards: Sequence[SectorRotationSignal],
        market_breadth: MarketBreadthDiffusion | None,
    ) -> str | None:
        if news_impact is None:
            return None
        title = str(getattr(news_impact, "title", "") or "").strip()
        rationale = str(getattr(news_impact, "rationale", "") or "").strip()
        if not title and not rationale:
            return None

        normalized = f"{title} {rationale}".casefold()
        bias_parts: list[str] = []
        leader_sectors = {item.sector.casefold() for item in leaders}
        laggard_sectors = {item.sector.casefold() for item in laggards}
        if any(keyword in normalized for keyword in ("war", "oil", "energy", "commodity")) and "energy" in leader_sectors:
            bias_parts.append("หนุนกลุ่มพลังงานมากกว่าตลาดส่วนอื่น")
        if any(keyword in normalized for keyword in ("inflation", "yield", "rate")) and "technology" in laggard_sectors:
            bias_parts.append("กดดันหุ้นเติบโตและ valuation สูง")
        if any(keyword in normalized for keyword in ("inflation", "rate", "bank", "credit")) and "financials" in laggard_sectors:
            bias_parts.append("กด sentiment ต่อกลุ่มการเงิน")
        if market_breadth is not None and not market_breadth.rally_confirmed:
            bias_parts.append("สอดคล้องกับภาวะรับความเสี่ยงต่ำของตลาด")

        source = title or rationale
        source = source[:160].rstrip()
        if bias_parts:
            return f"{source} | ผลต่อ sector: {' / '.join(dict.fromkeys(bias_parts))}"
        return source

    def _build_sector_rotation_action_lines(
        self,
        *,
        leaders: Sequence[SectorRotationSignal],
        laggards: Sequence[SectorRotationSignal],
        market_breadth: MarketBreadthDiffusion | None,
    ) -> list[str]:
        actions: list[str] = []
        weak_market = (
            market_breadth is not None
            and market_breadth.breadth_label in {"fragile rally", "mixed tape", "risk-off tape", "broad selloff"}
        )
        if weak_market:
            actions.append("ลดน้ำหนักหุ้น beta สูง และรอให้แรงซื้อกระจายกลับมาก่อนเพิ่มความเสี่ยง")
        if laggards:
            laggard_names = ", ".join(item.ticker or item.sector for item in laggards[:2])
            actions.append(f"ยังไม่ไล่ซื้อ {laggard_names} จนกว่าการยืนยันในกลุ่มจะฟื้น")
        if leaders:
            leader_name = leaders[0].ticker or leaders[0].sector
            if weak_market:
                actions.append(f"ถ้าจะเปิดสถานะใหม่ ให้เน้นผู้นำสัมพัทธ์อย่าง {leader_name} มากกว่ากลุ่มที่ breadth อ่อน")
            else:
                actions.append(f"เฝ้าดู {leader_name} เป็นตัวนำต่อ หากแรงซื้อในกลุ่มยังยืนยันได้")
        if not actions:
            actions.append("ถือพอร์ตตามเดิมและรอดูการยืนยันจาก breadth รอบถัดไป")
        return list(dict.fromkeys(actions))[:3]

    def _format_market_breadth_summary(self, item: MarketBreadthDiffusion) -> str:
        return (
            f"{self._translate_market_breadth_label(item.breadth_label)}"
            f" | หุ้นบวก {item.advancers}/{item.participant_count}"
            f" | หุ้นลบ {item.decliners}"
            f" | ดุลแรงซื้อ/ขาย {item.diffusion_score:+.2f}"
            f" | คะแนนแนวโน้มเฉลี่ย {item.average_trend_score:.2f}"
        )

    def _format_market_breadth_trend_summary(self, item: MarketBreadthTrend) -> str:
        return f"{self._translate_trend_label(item.trend_label)} | delta {item.score_delta:+.2f}"

    def _format_sector_rotation_snapshot_signal(
        self,
        signal: SectorRotationSignal,
        persistence_map: Mapping[str, SectorPersistenceInsight],
        breadth_map: Mapping[str, SectorBreadthInsight],
        breadth_trend_map: Mapping[str, SectorBreadthTrend],
    ) -> str:
        parts = [f"{signal.sector} ({signal.ticker})", f"score {signal.trend_score:.2f}"]
        breadth = breadth_map.get(signal.sector)
        if breadth is not None:
            parts.append(f"แรงกระจาย {breadth.advancers}/{breadth.constituent_count} ({breadth.participation_ratio:.0%})")
            if not breadth.equal_weight_confirmed and breadth.constituent_count >= 3:
                parts.append("การนำยังแคบ")
        insight = persistence_map.get(signal.sector)
        if insight is not None:
            unit = "วัน" if insight.regime == "daily" else "รอบ"
            parts.append(f"ต่อเนื่อง {insight.streak} {unit}")
            if abs(insight.score_delta) >= 0.2:
                parts.append(f"delta {insight.score_delta:+.2f}")
        breadth_trend = breadth_trend_map.get(signal.sector)
        if breadth_trend is not None and abs(breadth_trend.score_delta) >= 0.08:
            parts.append(f"แรงกระจาย{self._translate_trend_label(breadth_trend.trend_label)} {breadth_trend.score_delta:+.0%}")
        return " | ".join(parts)

    def _build_market_internals_break_text(
        self,
        *,
        market_breadth: MarketBreadthDiffusion,
        market_breadth_trend: MarketBreadthTrend | None,
        sp500_trend: TrendAssessment | None,
        nasdaq_trend: TrendAssessment | None,
    ) -> str:
        lines = [
            "สัญญาณโครงสร้างตลาดเริ่มอ่อน",
            "- ดัชนีหลักยังไม่เสียทรง แต่แรงซื้อภายในเริ่มแผ่วลง",
            (
                f"- ภาพรวมตลาด: {self._translate_market_breadth_label(market_breadth.breadth_label)}"
                f" | ดุลแรงซื้อ/ขาย {market_breadth.diffusion_score:+.2f}"
                f" | คะแนนแนวโน้มเฉลี่ย {market_breadth.average_trend_score:.2f}"
            ),
            (
                f"- S&P 500 score {sp500_trend.score if sp500_trend is not None else 'n/a'}"
                f" | NASDAQ score {nasdaq_trend.score if nasdaq_trend is not None else 'n/a'}"
            ),
        ]
        if market_breadth_trend is not None:
            lines.append(
                f"- เทียบรอบก่อน: {self._translate_trend_label(market_breadth_trend.trend_label)} | delta {market_breadth_trend.score_delta:+.2f}"
            )
        lines.extend(
            [
                "",
                "Action",
                "- อย่าไล่ซื้อดัชนีตาม headline จนกว่า breadth ตลาดจะฟื้น",
                "- เน้นลดความเสี่ยงในกลุ่มที่ breadth อ่อนก่อน",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _translate_market_breadth_label(label: str) -> str:
        mapping = {
            "broad rally": "ตลาดขึ้นเป็นวงกว้าง",
            "constructive but selective": "ตลาดบวกแต่เลือกกลุ่ม",
            "broad selloff": "ตลาดลงเป็นวงกว้าง",
            "fragile rally": "รีบาวด์เปราะบาง",
            "risk-off tape": "ตลาดปิดรับความเสี่ยง",
            "mixed tape": "ตลาดผสม",
        }
        normalized = str(label or "").strip().casefold()
        return mapping.get(normalized, str(label or "").strip() or "ตลาดผสม")

    @staticmethod
    def _translate_trend_label(label: str) -> str:
        mapping = {
            "stable": "ทรงตัว",
            "broadening": "ดีขึ้น",
            "weakening": "อ่อนลง",
            "narrowing": "แคบลง",
        }
        normalized = str(label or "").strip().casefold()
        return mapping.get(normalized, str(label or "").strip() or "ทรงตัว")

    @staticmethod
    def _translate_sector_stance(label: str) -> str:
        mapping = {
            "overweight": "นำตลาด",
            "underweight": "อ่อนแรง",
            "neutral": "กลางตลาด",
        }
        normalized = str(label or "").strip().casefold()
        return mapping.get(normalized, str(label or "").strip() or "กลางตลาด")

    def _build_periodic_report_prompt(self, *, report_kind: str, payload: Mapping[str, Any]) -> str:
        sector_lines = self._format_sector_rotation_lines(payload.get("sector_rotation"))
        sector_persistence_daily_lines = self._format_sector_persistence_lines(payload.get("sector_rotation_persistence_daily"))
        sector_persistence_intraday_lines = self._format_sector_persistence_lines(payload.get("sector_rotation_persistence_intraday"))
        market_breadth_lines = self._format_market_breadth_lines(payload.get("market_breadth_diffusion"))
        market_breadth_trend_lines = self._format_market_breadth_trend_lines(payload.get("market_breadth_trend"))
        divergence_lines = self._format_index_divergence_lines(payload.get("index_leadership_divergence"))
        sector_breadth_lines = self._format_sector_breadth_lines(payload.get("sector_breadth"))
        sector_breadth_trend_lines = self._format_sector_breadth_trend_lines(payload.get("sector_breadth_trend"))
        stock_lines = self._format_stock_pick_lines(payload.get("stock_picks"))
        earnings_lines = self._format_earnings_lines(payload.get("earnings_calendar"))
        earnings_setup_lines = self._format_earnings_setup_lines(payload.get("earnings_setups"))
        earnings_surprise_lines = self._format_earnings_surprise_lines(payload.get("earnings_surprises"))
        return (
            f"รายงานชนิด: {report_kind}\n"
            "มหภาค:\n"
            f"{self._format_macro_lines(payload.get('macro_context'))}\n\n"
            "Macro regime:\n"
            f"{self._format_macro_regime_lines(payload.get('macro_regime'))}\n\n"
            "Recommendation confidence:\n"
            f"{self._format_confidence_lines(payload.get('market_confidence'))}\n\n"
            "Allocation mix:\n"
            f"{self._format_allocation_mix_lines(payload.get('allocation_plan'))}\n\n"
            "Current portfolio:\n"
            f"{self._format_portfolio_snapshot_lines(payload.get('portfolio_snapshot'))}\n\n"
            "Rebalance review:\n"
            f"{self._format_portfolio_review_lines(payload.get('portfolio_review'))}\n\n"
            "Narrative memory ของวันเดียวกัน:\n"
            f"{self._format_report_memory_lines(payload.get('report_memory_context'))}\n\n"
            "ข่าวหลัก:\n"
            f"{self._format_news_lines(payload.get('news_headlines'))}\n\n"
            "Sector rotation:\n"
            f"{sector_lines}\n\n"
            "Sector persistence daily:\n"
            f"{sector_persistence_daily_lines}\n\n"
            "Sector persistence intraday:\n"
            f"{sector_persistence_intraday_lines}\n\n"
            "Market breadth diffusion:\n"
            f"{market_breadth_lines}\n\n"
            "Market breadth trend over time:\n"
            f"{market_breadth_trend_lines}\n\n"
            "Index leadership divergence:\n"
            f"{divergence_lines}\n\n"
            "Sector breadth and equal-weight confirmation:\n"
            f"{sector_breadth_lines}\n\n"
            "Sector breadth trend over time:\n"
            f"{sector_breadth_trend_lines}\n\n"
            "หุ้นเด่น:\n"
            f"{stock_lines}\n\n"
            "Earnings calendar:\n"
            f"{earnings_lines}\n\n"
            "Pre-earnings setup ranking:\n"
            f"{earnings_setup_lines}\n\n"
            "Post-earnings interpretation:\n"
            f"{earnings_surprise_lines}\n\n"
            "ตอบเป็นภาษาไทยในสไตล์ desk analyst สำหรับ Telegram โดยมีหัวข้อ: ภาพรวม, macro regime, allocation mix, narrative continuity, sector rotation, daily regime, intraday regime, breadth confirmation, หุ้นเด่น, earnings setup, post-earnings read-through, action plan"
        )

    def _build_periodic_report_fallback(self, *, report_kind: str, payload: Mapping[str, Any]) -> str:
        report_label = {
            "morning": "Morning Report",
            "midday": "Midday Report",
            "closing": "Closing Report",
        }.get(report_kind, "Market Report")
        return (
            f"{report_label}\n"
            f"{self._build_market_overview(payload.get('asset_snapshots', []), str(payload.get('scope') or 'all'), payload.get('portfolio_plan', {}))}\n\n"
            f"Macro Regime\n{self._format_macro_regime_lines(payload.get('macro_regime'))}\n\n"
            f"Recommendation Confidence\n{self._format_confidence_lines(payload.get('market_confidence'))}\n\n"
            f"Allocation Mix\n{self._format_allocation_mix_lines(payload.get('allocation_plan'))}\n\n"
            f"Current Portfolio\n{self._format_portfolio_snapshot_lines(payload.get('portfolio_snapshot'))}\n\n"
            f"Rebalance Review\n{self._format_portfolio_review_lines(payload.get('portfolio_review'))}\n\n"
            f"Daily Narrative Memory\n{self._format_report_memory_lines(payload.get('report_memory_context'))}\n\n"
            f"Sector Rotation\n{self._format_sector_rotation_lines(payload.get('sector_rotation'))}\n\n"
            f"Sector Persistence Daily\n{self._format_sector_persistence_lines(payload.get('sector_rotation_persistence_daily'))}\n\n"
            f"Sector Persistence Intraday\n{self._format_sector_persistence_lines(payload.get('sector_rotation_persistence_intraday'))}\n\n"
            f"Market Breadth Diffusion\n{self._format_market_breadth_lines(payload.get('market_breadth_diffusion'))}\n\n"
            f"Market Breadth Trend\n{self._format_market_breadth_trend_lines(payload.get('market_breadth_trend'))}\n\n"
            f"Index Leadership Divergence\n{self._format_index_divergence_lines(payload.get('index_leadership_divergence'))}\n\n"
            f"Sector Breadth\n{self._format_sector_breadth_lines(payload.get('sector_breadth'))}\n\n"
            f"Sector Breadth Trend\n{self._format_sector_breadth_trend_lines(payload.get('sector_breadth_trend'))}\n\n"
            f"หุ้นเด่น\n{self._format_stock_pick_lines(payload.get('stock_picks'))}\n\n"
            f"Earnings ที่ต้องจับตา\n{self._format_earnings_lines(payload.get('earnings_calendar'))}\n\n"
            f"Pre-Earnings Setup Ranking\n{self._format_earnings_setup_lines(payload.get('earnings_setups'))}\n\n"
            f"Post-Earnings Read-Through\n{self._format_earnings_surprise_lines(payload.get('earnings_surprises'))}\n\n"
            f"ข่าวสำคัญ\n{self._format_fallback_news(payload.get('news_headlines')) or '- ไม่มี headline เด่น'}"
        )

    def _filter_asset_context(
        self,
        *,
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        asset_scope: AssetScope,
    ) -> tuple[dict[str, AssetQuote | None], dict[str, TrendAssessment]]:
        members = ASSET_SCOPE_MEMBERS.get(asset_scope, ())
        if not members:
            return dict(market_data), dict(trends)
        return (
            {name: market_data.get(name) for name in members if name in market_data},
            {name: trends.get(name) for name in members if name in trends},
        )

    def _serialize_investor_profile(self, profile: InvestorProfile) -> dict[str, Any]:
        return {
            "name": profile.name,
            "title": profile.title_th,
            "objective": profile.objective,
            "risk_summary": profile.risk_summary,
            "rebalance_hint": profile.rebalance_hint,
        }

    def _serialize_portfolio_plan(self, plan: PortfolioPlan) -> dict[str, Any]:
        return {
            "profile_name": plan.profile.name,
            "market_regime": plan.market_regime,
            "regime_summary": plan.regime_summary,
            "allocations": [self._serialize_allocation_bucket(bucket) for bucket in plan.buckets],
            "action_plan": plan.action_plan,
            "risk_watch": plan.risk_watch,
        }

    def _serialize_macro_regime(self, assessment: MacroRegimeAssessment) -> dict[str, Any]:
        return {
            "regime": assessment.regime,
            "confidence": assessment.confidence,
            "growth_signal": assessment.growth_signal,
            "inflation_signal": assessment.inflation_signal,
            "market_signal": assessment.market_signal,
            "headline": assessment.headline,
            "rationale": list(assessment.rationale),
        }

    @staticmethod
    def _serialize_confidence_assessment(assessment: ConfidenceAssessment) -> dict[str, Any]:
        return {
            "score": assessment.score,
            "label": assessment.label,
            "rationale": list(assessment.rationale),
        }

    def _serialize_allocation_plan(self, plan: PortfolioAllocationPlan) -> dict[str, Any]:
        return {
            "profile_name": plan.profile_name,
            "macro_regime": plan.macro_regime,
            "narrative": plan.narrative,
            "buckets": [
                {
                    "category": bucket.category,
                    "target_pct": bucket.target_pct,
                    "stance": bucket.stance,
                    "rationale": bucket.rationale,
                    "preferred_assets": list(bucket.preferred_assets),
                }
                for bucket in plan.buckets
            ],
            "rebalance_note": plan.rebalance_note,
        }

    def _serialize_allocation_bucket(self, bucket: AllocationBucket) -> dict[str, Any]:
        return {
            "category": bucket.category,
            "label": bucket.label,
            "target_pct": bucket.target_pct,
            "stance": bucket.stance,
            "preferred_assets": list(bucket.preferred_assets),
            "rationale": bucket.rationale,
        }

    def _serialize_news_article(self, article: NewsArticle) -> dict[str, Any]:
        return {
            "title": self._truncate_text(article.title, 110),
            "source": article.source,
            "published_at": article.published_at.isoformat() if article.published_at else None,
        }

    def _serialize_research_finding(self, finding: ResearchFinding) -> dict[str, Any]:
        return {
            "title": self._truncate_text(finding.title, 110),
            "source": self._truncate_text(finding.source, 40),
            "provider": finding.provider,
            "url": finding.url,
            "snippet": self._truncate_text(finding.snippet, 180),
            "published_at": finding.published_at.isoformat() if finding.published_at else None,
        }

    def _serialize_stock_candidate(self, candidate: StockCandidate) -> dict[str, Any]:
        confidence = assess_stock_candidate_confidence(candidate)
        return {
            "ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "sector": candidate.sector,
            "price": candidate.price,
            "score": candidate.composite_score,
            "confidence_score": confidence.score,
            "confidence_label": confidence.label,
            "stance": candidate.stance,
            "trend_direction": candidate.trend_direction,
            "rationale": list(candidate.rationale),
            "forward_pe": candidate.forward_pe,
            "revenue_growth": candidate.revenue_growth,
            "earnings_growth": candidate.earnings_growth,
        }

    def _serialize_sector_rotation(self, signal: SectorRotationSignal) -> dict[str, Any]:
        return {
            "sector": signal.sector,
            "ticker": signal.ticker,
            "trend_direction": signal.trend_direction,
            "trend_score": signal.trend_score,
            "rsi": signal.rsi,
            "stance": signal.stance,
            "rationale": list(signal.rationale),
        }

    def _serialize_sector_persistence(self, insight: SectorPersistenceInsight) -> dict[str, Any]:
        return {
            "sector": insight.sector,
            "ticker": insight.ticker,
            "regime": insight.regime,
            "stance": insight.stance,
            "streak": insight.streak,
            "average_score": round(insight.average_score, 2),
            "score_delta": round(insight.score_delta, 2),
            "changed_from": insight.changed_from,
        }

    def _serialize_sector_breadth(self, insight: SectorBreadthInsight) -> dict[str, Any]:
        return {
            "sector": insight.sector,
            "ticker": insight.ticker,
            "constituent_count": insight.constituent_count,
            "advancers": insight.advancers,
            "decliners": insight.decliners,
            "participation_ratio": round(insight.participation_ratio, 2),
            "average_trend_score": round(insight.average_trend_score, 2),
            "equal_weight_confirmed": insight.equal_weight_confirmed,
            "breadth_label": insight.breadth_label,
        }

    def _serialize_sector_breadth_trend(self, item: SectorBreadthTrend) -> dict[str, Any]:
        return {
            "sector": item.sector,
            "ticker": item.ticker,
            "regime": item.regime,
            "current_participation_ratio": round(item.current_participation_ratio, 2),
            "score_delta": round(item.score_delta, 2),
            "sparkline": item.sparkline,
            "history_scores": list(item.history_scores),
            "trend_label": item.trend_label,
        }

    def _serialize_market_breadth_diffusion(self, item: MarketBreadthDiffusion | None) -> dict[str, Any] | None:
        if item is None:
            return None
        return {
            "participant_count": item.participant_count,
            "advancers": item.advancers,
            "decliners": item.decliners,
            "neutral_count": item.neutral_count,
            "advancing_ratio": item.advancing_ratio,
            "declining_ratio": item.declining_ratio,
            "diffusion_score": item.diffusion_score,
            "average_trend_score": item.average_trend_score,
            "breadth_label": item.breadth_label,
            "rally_confirmed": item.rally_confirmed,
        }

    def _serialize_market_breadth_trend(self, item: MarketBreadthTrend | None) -> dict[str, Any] | None:
        if item is None:
            return None
        return {
            "regime": item.regime,
            "current_diffusion_score": item.current_diffusion_score,
            "score_delta": item.score_delta,
            "sparkline": item.sparkline,
            "history_scores": list(item.history_scores),
            "trend_label": item.trend_label,
            "breadth_label": item.breadth_label,
        }

    def _serialize_index_leadership_divergence(self, item: IndexLeadershipDivergence) -> dict[str, Any]:
        return {
            "label": item.label,
            "cap_weight_ticker": item.cap_weight_ticker,
            "equal_weight_ticker": item.equal_weight_ticker,
            "short_return_spread": item.short_return_spread,
            "medium_return_spread": item.medium_return_spread,
            "cap_weight_score": item.cap_weight_score,
            "equal_weight_score": item.equal_weight_score,
            "divergence_label": item.divergence_label,
            "severity": item.severity,
        }

    def _serialize_earnings_event(self, event: EarningsEvent) -> dict[str, Any]:
        return {
            "ticker": event.ticker,
            "earnings_at": event.earnings_at.isoformat(),
            "eps_estimate": event.eps_estimate,
            "reported_eps": event.reported_eps,
            "surprise_pct": event.surprise_pct,
        }

    def _serialize_earnings_interpretation(self, item: EarningsInterpretation) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "earnings_at": item.earnings_at.isoformat(),
            "eps_estimate": item.eps_estimate,
            "reported_eps": item.reported_eps,
            "surprise_pct": item.surprise_pct,
            "revenue_signal": item.revenue_signal,
            "revenue_qoq_change": item.revenue_qoq_change,
            "revenue_qoq_trend": item.revenue_qoq_trend,
            "revenue_expectation_gap_pct": item.revenue_expectation_gap_pct,
            "revenue_expectation_source": item.revenue_expectation_source,
            "revenue_vs_sector_median": item.revenue_vs_sector_median,
            "revenue_relative_signal": item.revenue_relative_signal,
            "margin_signal": item.margin_signal,
            "margin_qoq_change": item.margin_qoq_change,
            "margin_qoq_trend": item.margin_qoq_trend,
            "margin_vs_sector_median": item.margin_vs_sector_median,
            "margin_relative_signal": item.margin_relative_signal,
            "guidance_signal": item.guidance_signal,
            "guidance_score": item.guidance_score,
            "management_tone": item.management_tone,
            "fcf_quality": item.fcf_quality,
            "fcf_qoq_change": item.fcf_qoq_change,
            "fcf_qoq_trend": item.fcf_qoq_trend,
            "fcf_vs_sector_median": item.fcf_vs_sector_median,
            "fcf_relative_signal": item.fcf_relative_signal,
            "earnings_quality_score": item.earnings_quality_score,
            "earnings_quality_label": item.earnings_quality_label,
            "one_off_risk": item.one_off_risk,
            "stance": item.stance,
            "rationale": list(item.rationale),
        }

    def _serialize_earnings_setup(self, item: EarningsSetupCandidate) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "company_name": item.company_name,
            "earnings_at": item.earnings_at.isoformat(),
            "setup_score": item.setup_score,
            "setup_label": item.setup_label,
            "valuation_signal": item.valuation_signal,
            "expectation_signal": item.expectation_signal,
            "trend_signal": item.trend_signal,
            "rationale": list(item.rationale),
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
        return {
            "asset": asset_name,
            "label": self._asset_display_name(asset_name),
            "ticker": quote.ticker if quote else trend.ticker if trend else None,
            "price": self._round_optional(quote.price if quote else None),
            "day_change_pct": day_change_pct,
            "trend": trend.direction if trend else "sideways",
            "trend_score": round(trend.score, 2) if trend else None,
            "rsi": self._round_optional(trend.rsi if trend else None),
            "macd_hist": self._round_optional(trend.macd_hist if trend else None),
            "support": self._round_optional(trend.support_resistance.nearest_support if trend else None),
            "resistance": self._round_optional(trend.support_resistance.nearest_resistance if trend else None),
            "signals": list((trend.reasons if trend else [])[:DEFAULT_REASON_LIMIT]),
        }

    def _get_history_lines(self, conversation_key: str | None) -> list[str]:
        if conversation_key is None:
            return []
        history = self._conversation_history.get(conversation_key)
        if not history:
            return []
        return [f"- {entry['role']}: {self._truncate_text(entry['text'], 160)}" for entry in history]

    def _build_report_memory_context(
        self,
        *,
        report_kind: str,
        report_memory_store: ReportMemoryStore | None,
    ) -> dict[str, str]:
        if report_memory_store is None:
            return {}
        day_entries = report_memory_store.get_day_entries()
        allowed_prior = {
            "morning": (),
            "midday": ("morning",),
            "closing": ("morning", "midday"),
        }.get(report_kind, ("morning", "midday"))
        context: dict[str, str] = {}
        for key in allowed_prior:
            entry = day_entries.get(key)
            if entry is not None and entry.summary.strip():
                context[key] = entry.summary.strip()
        return context

    def _build_report_memory_summary(self, *, report_kind: str, payload: Mapping[str, Any]) -> str:
        macro_regime = payload.get("macro_regime")
        market_breadth = payload.get("market_breadth_diffusion")
        stock_picks = payload.get("stock_picks")
        top_pick = None
        if isinstance(stock_picks, list):
            top_pick = next((item for item in stock_picks if isinstance(item, Mapping)), None)
        sector_rotation = payload.get("sector_rotation")
        leading_sector = None
        if isinstance(sector_rotation, list):
            leading_sector = next((item for item in sector_rotation if isinstance(item, Mapping)), None)
        parts = [report_kind.capitalize()]
        if isinstance(macro_regime, Mapping):
            headline = str(macro_regime.get("headline") or "").strip()
            if headline:
                parts.append(headline)
        if isinstance(market_breadth, Mapping):
            label = str(market_breadth.get("breadth_label") or "").strip()
            if label:
                parts.append(f"breadth {label}")
        if isinstance(leading_sector, Mapping):
            sector = str(leading_sector.get("sector") or "").strip()
            stance = str(leading_sector.get("stance") or "").strip()
            if sector:
                parts.append(f"sector นำ {sector} ({stance or 'n/a'})")
        if isinstance(top_pick, Mapping):
            ticker = str(top_pick.get("ticker") or "").strip()
            stance = str(top_pick.get("stance") or "").strip()
            if ticker:
                parts.append(f"หุ้นเด่น {ticker} ({stance or 'watch'})")
        return " | ".join(parts[:5])

    def _remember_turns(self, *, conversation_key: str | None, user_text: str | None, assistant_text: str) -> None:
        if conversation_key is None:
            return
        history = self._conversation_history[conversation_key]
        if user_text:
            history.append({"role": "user", "text": user_text})
        history.append({"role": "assistant", "text": assistant_text})

    @staticmethod
    def _format_profile_lines(profile: Any) -> str:
        if not isinstance(profile, Mapping):
            return "- โปรไฟล์มาตรฐาน"
        return (
            f"- โปรไฟล์: {profile.get('title')}\n"
            f"- เป้าหมาย: {profile.get('objective')}\n"
            f"- ความเสี่ยง: {profile.get('risk_summary')}\n"
            f"- วินัยพอร์ต: {profile.get('rebalance_hint')}"
        )

    @staticmethod
    def _format_profile_one_line(profile: Any) -> str:
        if not isinstance(profile, Mapping):
            return "โปรไฟล์ผู้ลงทุน: สมดุล"
        return f"โปรไฟล์ผู้ลงทุน: {profile.get('title')} | เป้าหมาย: {profile.get('objective')}"

    @staticmethod
    def _format_macro_lines(macro_context: Any) -> str:
        if not isinstance(macro_context, Mapping):
            return "- ไม่มีข้อมูลมหภาค"
        return (
            f"- VIX: {macro_context.get('vix', 'N/A')}\n"
            f"- US 10Y Yield: {macro_context.get('tnx', 'N/A')}\n"
            f"- US CPI YoY: {macro_context.get('cpi_yoy', 'N/A')}"
        )

    @staticmethod
    def _format_macro_one_line(macro_context: Any) -> str:
        if not isinstance(macro_context, Mapping):
            return "มหภาค: ไม่มีข้อมูล"
        return (
            f"มหภาค: VIX {macro_context.get('vix', 'N/A')} | "
            f"US10Y {macro_context.get('tnx', 'N/A')} | CPI YoY {macro_context.get('cpi_yoy', 'N/A')}"
        )

    @staticmethod
    def _format_confidence_one_line(confidence: Any) -> str:
        if not isinstance(confidence, Mapping):
            return "ระดับความมั่นใจ: n/a"
        return f"ระดับความมั่นใจ: {confidence.get('label') or 'n/a'} ({confidence.get('score') or 'n/a'})"

    @staticmethod
    def _format_macro_regime_lines(macro_regime: Any) -> str:
        if not isinstance(macro_regime, Mapping):
            return "- ยังไม่มี macro regime"
        rationale = ", ".join(str(item) for item in (macro_regime.get("rationale") or []) if item)
        return (
            f"- Regime: {macro_regime.get('regime')} | confidence={macro_regime.get('confidence')}\n"
            f"- Headline: {macro_regime.get('headline')}\n"
            f"- Growth={macro_regime.get('growth_signal')} | Inflation={macro_regime.get('inflation_signal')} | Market={macro_regime.get('market_signal')}\n"
            f"- เหตุผล: {rationale or 'ไม่มีเหตุผลเพิ่มเติม'}"
        )

    @staticmethod
    def _format_confidence_lines(confidence: Any) -> str:
        if not isinstance(confidence, Mapping):
            return "- score=n/a | label=n/a"
        rationale = ", ".join(str(item) for item in (confidence.get("rationale") or []) if item)
        return (
            f"- score={confidence.get('score')} | label={confidence.get('label')}\n"
            f"- rationale: {rationale or 'ไม่มีเหตุผลเพิ่มเติม'}"
        )

    @staticmethod
    def _format_news_lines(news_items: Any) -> str:
        if not isinstance(news_items, list) or not news_items:
            return "- ไม่มี headline สำคัญ"
        return "\n".join(
            f"- {item.get('title') or '-'} ({item.get('source') or 'Unknown'})"
            for item in news_items
            if isinstance(item, Mapping)
        ) or "- ไม่มี headline สำคัญ"

    @staticmethod
    def _format_research_lines(research_items: Any) -> str:
        if not isinstance(research_items, list) or not research_items:
            return "- ไม่มีข้อมูลวิจัยเว็บเพิ่มเติม"
        return "\n".join(
            f"- {item.get('title') or '-'} ({item.get('source') or item.get('provider') or 'Unknown'})"
            for item in research_items
            if isinstance(item, Mapping)
        ) or "- ไม่มีข้อมูลวิจัยเว็บเพิ่มเติม"

    @staticmethod
    def _format_sector_rotation_lines(rotation_items: Any) -> str:
        if not isinstance(rotation_items, list) or not rotation_items:
            return "- ไม่มีสัญญาณ sector rotation"
        return "\n".join(
            f"- {item.get('sector')} ({item.get('ticker')}): {item.get('stance')} | score={item.get('trend_score')} | RSI={item.get('rsi')}"
            for item in rotation_items
            if isinstance(item, Mapping)
        ) or "- ไม่มีสัญญาณ sector rotation"

    @staticmethod
    def _format_sector_persistence_lines(persistence_items: Any) -> str:
        if not isinstance(persistence_items, list) or not persistence_items:
            return "- ยังไม่มีข้อมูลต่อเนื่องหลายรอบ"
        return "\n".join(
            f"- {item.get('sector')} ({item.get('ticker')}): {item.get('stance')} [{item.get('regime')}] ต่อเนื่อง {item.get('streak')} {'วัน' if item.get('regime') == 'daily' else 'รอบ'} | avg={item.get('average_score')} | delta={item.get('score_delta')} | changed_from={item.get('changed_from') or '-'}"
            for item in persistence_items
            if isinstance(item, Mapping)
        ) or "- ยังไม่มีข้อมูลต่อเนื่องหลายรอบ"

    @staticmethod
    def _format_market_breadth_lines(breadth_item: Any) -> str:
        if not isinstance(breadth_item, Mapping):
            return "- ยังไม่มี market breadth diffusion"
        return (
            f"- {breadth_item.get('breadth_label')} | advancers={breadth_item.get('advancers')}/{breadth_item.get('participant_count')}"
            f" | decliners={breadth_item.get('decliners')} | neutral={breadth_item.get('neutral_count')}"
            f" | diffusion={breadth_item.get('diffusion_score')} | avg trend={breadth_item.get('average_trend_score')}"
            f" | rally_confirmed={breadth_item.get('rally_confirmed')}"
        )

    @staticmethod
    def _format_market_breadth_trend_lines(breadth_item: Any) -> str:
        if not isinstance(breadth_item, Mapping):
            return "- ยังไม่มี market breadth history"
        history_scores = breadth_item.get("history_scores") or []
        return (
            f"- {breadth_item.get('breadth_label')} | {breadth_item.get('trend_label')}"
            f" | sparkline={breadth_item.get('sparkline')}"
            f" | series={' -> '.join(str(score) for score in history_scores)}"
            f" | delta={breadth_item.get('score_delta')}"
        )

    @staticmethod
    def _format_index_divergence_lines(divergence_items: Any) -> str:
        if not isinstance(divergence_items, list) or not divergence_items:
            return "- ยังไม่มี index leadership divergence เด่น"
        return "\n".join(
            f"- {item.get('label')}: {item.get('divergence_label')} | {item.get('cap_weight_ticker')} vs {item.get('equal_weight_ticker')} | 1m spread={RecommendationService._format_percent_delta(item.get('short_return_spread'))} | 3m spread={RecommendationService._format_percent_delta(item.get('medium_return_spread'))}"
            for item in divergence_items
            if isinstance(item, Mapping)
        ) or "- ยังไม่มี index leadership divergence เด่น"

    @staticmethod
    def _format_sector_breadth_lines(breadth_items: Any) -> str:
        if not isinstance(breadth_items, list) or not breadth_items:
            return "- ยังไม่มี breadth confirmation"
        return "\n".join(
            f"- {item.get('sector')} ({item.get('ticker')}): {item.get('breadth_label')} | breadth {item.get('advancers')}/{item.get('constituent_count')} | participation={item.get('participation_ratio')} | equal_weight_confirmed={item.get('equal_weight_confirmed')}"
            for item in breadth_items
            if isinstance(item, Mapping)
        ) or "- ยังไม่มี breadth confirmation"

    @staticmethod
    def _format_sector_breadth_trend_lines(breadth_items: Any) -> str:
        if not isinstance(breadth_items, list) or not breadth_items:
            return "- ยังไม่มี breadth trend history"
        return "\n".join(
            f"- {item.get('sector')} ({item.get('ticker')}): {item.get('trend_label')} | sparkline={item.get('sparkline')} | series={' -> '.join(str(score) for score in (item.get('history_scores') or []))} | delta={item.get('score_delta')}"
            for item in breadth_items
            if isinstance(item, Mapping)
        ) or "- ยังไม่มี breadth trend history"

    @staticmethod
    def _format_stock_pick_lines(stock_items: Any) -> str:
        if not isinstance(stock_items, list) or not stock_items:
            return "- ไม่มีหุ้นเด่น"
        return "\n".join(
            f"- {item.get('company_name')} ({item.get('ticker')}): {item.get('stance')} | score={item.get('score')} | confidence={item.get('confidence_label')} ({item.get('confidence_score')}) | {', '.join(item.get('rationale') or [])}"
            for item in stock_items
            if isinstance(item, Mapping)
        ) or "- ไม่มีหุ้นเด่น"

    @staticmethod
    def _format_earnings_lines(earnings_items: Any) -> str:
        if not isinstance(earnings_items, list) or not earnings_items:
            return "- ไม่มี earnings ใกล้เข้ามา"
        return "\n".join(
            f"- {item.get('ticker')}: {item.get('earnings_at')} | EPS est={item.get('eps_estimate')}"
            for item in earnings_items
            if isinstance(item, Mapping)
        ) or "- ไม่มี earnings ใกล้เข้ามา"

    @staticmethod
    def _format_earnings_surprise_lines(earnings_items: Any) -> str:
        if not isinstance(earnings_items, list) or not earnings_items:
            return "- ไม่มี post-earnings signal เด่น"
        lines: list[str] = []
        for item in earnings_items:
            if not isinstance(item, Mapping):
                continue
            source = item.get("revenue_expectation_source")
            source_suffix = f" via {source}" if source else ""
            lines.append(
                f"- {item.get('ticker')}: stance={item.get('stance')} | surprise={item.get('surprise_pct')} "
                f"| revenue={item.get('revenue_signal')} ({item.get('revenue_qoq_trend')} {RecommendationService._format_percent_delta(item.get('revenue_qoq_change'))}, "
                f"{item.get('revenue_relative_signal')} {RecommendationService._format_percent_delta(item.get('revenue_vs_sector_median'))} vs sector, "
                f"rev gap {RecommendationService._format_percent_delta(item.get('revenue_expectation_gap_pct'))}{source_suffix}) "
                f"| margin={item.get('margin_signal')} ({item.get('margin_qoq_trend')} {RecommendationService._format_percent_delta(item.get('margin_qoq_change'))}, "
                f"{item.get('margin_relative_signal')} {RecommendationService._format_percent_delta(item.get('margin_vs_sector_median'))} vs sector) "
                f"| guidance={item.get('guidance_signal')} | tone={item.get('management_tone')} "
                f"| FCF={item.get('fcf_quality')} ({item.get('fcf_qoq_trend')} {RecommendationService._format_percent_delta(item.get('fcf_qoq_change'))}, "
                f"{item.get('fcf_relative_signal')} {RecommendationService._format_percent_delta(item.get('fcf_vs_sector_median'))} vs sector) "
                f"| quality={item.get('earnings_quality_label')} ({item.get('earnings_quality_score')}) "
                f"| one_off={item.get('one_off_risk')} | {', '.join(item.get('rationale') or [])}"
            )
        return "\n".join(lines) or "- ไม่มี post-earnings signal เด่น"

    @staticmethod
    def _format_earnings_setup_lines(earnings_items: Any) -> str:
        if not isinstance(earnings_items, list) or not earnings_items:
            return "- ไม่มี pre-earnings setup เด่น"
        return "\n".join(
            f"- {item.get('company_name')} ({item.get('ticker')}): score={item.get('setup_score')} | {item.get('setup_label')} | trend={item.get('trend_signal')} | valuation={item.get('valuation_signal')} | expectations={item.get('expectation_signal')} | {', '.join(item.get('rationale') or [])}"
            for item in earnings_items
            if isinstance(item, Mapping)
        ) or "- ไม่มี pre-earnings setup เด่น"

    @staticmethod
    def _format_report_memory_lines(memory_context: Any) -> str:
        if not isinstance(memory_context, Mapping) or not memory_context:
            return "- ไม่มี narrative ก่อนหน้าของวันเดียวกัน"
        order = ("morning", "midday", "closing")
        lines: list[str] = []
        for key in order:
            value = memory_context.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"- {key}: {value}")
        return "\n".join(lines) or "- ไม่มี narrative ก่อนหน้าของวันเดียวกัน"

    @staticmethod
    def _format_asset_lines(asset_items: Any) -> str:
        if not isinstance(asset_items, list) or not asset_items:
            return "- ไม่มี snapshot สินทรัพย์"
        lines: list[str] = []
        for item in asset_items:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- {item.get('label')} [{item.get('ticker')}]: price={item.get('price')}, trend={item.get('trend')}, score={item.get('trend_score')}, RSI={item.get('rsi')}, support={item.get('support')}, resistance={item.get('resistance')}"
            )
        return "\n".join(lines) or "- ไม่มี snapshot สินทรัพย์"

    def _format_portfolio_lines(self, portfolio_plan: Any) -> str:
        if not isinstance(portfolio_plan, Mapping):
            return "- ไม่มีแผนจัดพอร์ต"
        allocations = portfolio_plan.get("allocations")
        lines = [f"- ภาวะตลาด: {portfolio_plan.get('market_regime')} | {portfolio_plan.get('regime_summary')}"]
        if isinstance(allocations, list):
            for bucket in allocations:
                if not isinstance(bucket, Mapping):
                    continue
                preferred_assets = ", ".join(bucket.get("preferred_assets") or [])
                lines.append(f"- {bucket.get('label')}: {bucket.get('target_pct')}% ({bucket.get('stance')}) | {preferred_assets or '-'}")
        action_plan = portfolio_plan.get("action_plan")
        if action_plan:
            lines.append(f"- Action: {action_plan}")
        return "\n".join(lines)

    @staticmethod
    def _format_allocation_mix_lines(allocation_plan: Any) -> str:
        if not isinstance(allocation_plan, Mapping):
            return "- ยังไม่มี allocation mix"
        lines = [f"- Narrative: {allocation_plan.get('narrative') or '-'}"]
        buckets = allocation_plan.get("buckets")
        if isinstance(buckets, list):
            for bucket in buckets:
                if not isinstance(bucket, Mapping):
                    continue
                preferred_assets = ", ".join(str(item) for item in (bucket.get("preferred_assets") or []) if item) or "-"
                lines.append(
                    f"- {bucket.get('category')}: {bucket.get('target_pct')}% ({bucket.get('stance')}) | {preferred_assets}"
                )
        rebalance_note = allocation_plan.get("rebalance_note")
        if rebalance_note:
            lines.append(f"- Rebalance: {rebalance_note}")
        return "\n".join(lines)

    @staticmethod
    def _format_portfolio_snapshot_lines(portfolio_snapshot: Any) -> str:
        if not isinstance(portfolio_snapshot, Mapping):
            return "- ยังไม่มีพอร์ตจริงที่บันทึกไว้"
        holdings = portfolio_snapshot.get("holdings")
        if not isinstance(holdings, list) or not holdings:
            return "- ยังไม่มีพอร์ตจริงที่บันทึกไว้"

        total_market_value = RecommendationService._as_float(portfolio_snapshot.get("total_market_value"))
        lines: list[str] = []
        if total_market_value is not None:
            lines.append(f"- Total market value: {total_market_value:,.2f} USD")
        for item in holdings[:8]:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "").strip().upper()
            category = str(item.get("category") or "").strip().replace("_", " ")
            market_value = RecommendationService._as_float(item.get("market_value"))
            current_weight_pct = RecommendationService._as_float(item.get("current_weight_pct"))
            unrealized_pnl_pct = RecommendationService._as_float(item.get("unrealized_pnl_pct"))
            note = str(item.get("note") or "").strip()
            if not ticker:
                continue
            line = f"- {ticker}: {category or 'unclassified'}"
            if current_weight_pct is not None:
                line += f" | weight {current_weight_pct:.1f}%"
            if market_value is not None:
                line += f" | value {market_value:,.2f} USD"
            if unrealized_pnl_pct is not None:
                line += f" | pnl {unrealized_pnl_pct:+.1%}"
            if note:
                line += f" | note: {note}"
            lines.append(line)
        return "\n".join(lines) or "- ยังไม่มีพอร์ตจริงที่บันทึกไว้"

    @staticmethod
    def _format_portfolio_review_lines(portfolio_review: Any) -> str:
        if not isinstance(portfolio_review, Mapping):
            return "- ยังไม่มีสัญญาณรีบาลานซ์จากพอร์ตจริง"
        buckets = portfolio_review.get("buckets")
        if not isinstance(buckets, list) or not buckets:
            return "- ยังไม่มีสัญญาณรีบาลานซ์จากพอร์ตจริง"

        lines: list[str] = []
        action_summary = str(portfolio_review.get("action_summary") or "").strip()
        holdings_count = RecommendationService._as_float(portfolio_review.get("holdings_count"))
        total_market_value = RecommendationService._as_float(portfolio_review.get("total_market_value"))
        summary_parts: list[str] = []
        if action_summary:
            summary_parts.append(action_summary)
        if holdings_count is not None:
            summary_parts.append(f"holdings={int(holdings_count)}")
        if total_market_value is not None:
            summary_parts.append(f"value={total_market_value:,.2f} USD")
        if summary_parts:
            lines.append(f"- Summary: {' | '.join(summary_parts)}")

        for bucket in buckets[:5]:
            if not isinstance(bucket, Mapping):
                continue
            category = str(bucket.get("category") or "").strip().replace("_", " ")
            target_pct = RecommendationService._as_float(bucket.get("target_pct"))
            current_pct = RecommendationService._as_float(bucket.get("current_pct"))
            drift_pct = RecommendationService._as_float(bucket.get("drift_pct"))
            action = str(bucket.get("action") or "").strip()
            top_holdings_raw = bucket.get("top_holdings")
            top_holdings = ", ".join(str(item).upper() for item in top_holdings_raw if item) if isinstance(top_holdings_raw, list) else ""
            line = f"- {category or 'unclassified'}"
            if current_pct is not None and target_pct is not None:
                line += f": current {current_pct:.1f}% / target {target_pct:.1f}%"
            elif current_pct is not None:
                line += f": current {current_pct:.1f}%"
            elif target_pct is not None:
                line += f": target {target_pct:.1f}%"
            if drift_pct is not None:
                line += f" | drift {drift_pct:+.1f}%"
            if action:
                line += f" | {action}"
            if top_holdings:
                line += f" | top holdings: {top_holdings}"
            lines.append(line)
        return "\n".join(lines) or "- ยังไม่มีสัญญาณรีบาลานซ์จากพอร์ตจริง"

    @staticmethod
    def _format_fallback_news(news_items: Any) -> str:
        if not isinstance(news_items, list) or not news_items:
            return ""
        return "\n".join(
            f"- {item.get('title') or '-'} ({item.get('source') or 'Unknown'})"
            for item in news_items[:DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT]
            if isinstance(item, Mapping)
        )

    @staticmethod
    def _format_fallback_research(research_items: Any) -> str:
        if not isinstance(research_items, list) or not research_items:
            return ""
        return "\n".join(
            f"- {item.get('title') or '-'} ({item.get('source') or item.get('provider') or 'Unknown'})"
            for item in research_items[:DEFAULT_SPECIFIC_SCOPE_NEWS_LIMIT]
            if isinstance(item, Mapping)
        )

    def _render_asset_focus_line(self, asset: Mapping[str, Any]) -> str:
        stance = self._direction_to_allocation_stance(str(asset.get("trend") or "sideways"))
        support = asset.get("support")
        resistance = asset.get("resistance")
        technical = self._humanize_signals(asset.get("signals"))
        level_text = f"แนวรับ {support} | แนวต้าน {resistance}" if support or resistance else "รอแนวราคาชัดเจน"
        return f"{asset.get('label')}: {stance} | {technical} | {level_text}"

    @staticmethod
    def _direction_to_allocation_stance(direction: str) -> str:
        return {
            "uptrend": "เพิ่มน้ำหนักได้แบบทยอยสะสม",
            "downtrend": "ลดน้ำหนักหรือชะลอการเพิ่มพอร์ต",
            "sideways": "คงน้ำหนักและรอจังหวะยืนยัน",
        }.get(direction, "คงน้ำหนักและรอจังหวะยืนยัน")

    @staticmethod
    def _humanize_stock_stance(stance: str) -> str:
        return {
            "buy": "น่าสนใจสำหรับทยอยสะสม",
            "watch": "ควรเฝ้าดูและรอจังหวะเพิ่มน้ำหนัก",
            "avoid": "ยังไม่ใช่จังหวะที่ดี",
        }.get(stance, stance)

    @staticmethod
    def _format_badged_title(badge: str, title: str) -> str:
        return f"{badge} | {title}"

    @staticmethod
    def _humanize_confidence_label(label: str) -> str:
        return {
            "very_high": "สูงมาก",
            "high": "สูง",
            "medium": "ปานกลาง",
            "low": "ต่ำ",
        }.get(str(label or "").strip().casefold(), str(label or "").strip() or "n/a")

    @staticmethod
    def _humanize_risk_level(level: str) -> str:
        return {
            "low": "ต่ำ",
            "moderate": "ปานกลาง",
            "elevated": "ยกระดับ",
            "high": "สูง",
        }.get(str(level or "").strip().casefold(), str(level or "").strip() or "ไม่ระบุ")

    @staticmethod
    def _humanize_related_bucket(bucket: str) -> str:
        return {
            "equities": "หุ้น",
            "gold": "ทองคำ",
            "bonds": "ตราสารหนี้",
            "macro": "มหภาค",
            "fx": "อัตราแลกเปลี่ยน",
            "commodities": "สินค้าโภคภัณฑ์",
        }.get(str(bucket or "").strip().casefold(), str(bucket or "").strip() or "ตลาดรวม")

    @staticmethod
    def _humanize_earnings_direction(value: str) -> str:
        return {
            "positive": "บวก",
            "negative": "ลบ",
            "neutral": "ทรงตัว",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_earnings_signal(value: str) -> str:
        return {
            "beat": "ดีกว่าคาด",
            "miss": "ต่ำกว่าคาด",
            "neutral": "ใกล้เคียงคาด",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_relative_signal(value: str) -> str:
        return {
            "above_sector": "ดีกว่า sector",
            "below_sector": "แย่กว่า sector",
            "in-line": "ใกล้เคียง sector",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_trend_description(value: str) -> str:
        return {
            "improving": "ดีขึ้น",
            "deteriorating": "อ่อนลง",
            "neutral": "ทรงตัว",
            "uptrend": "ขาขึ้น",
            "downtrend": "ขาลง",
            "sideways": "แกว่งตัว",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_fcf_quality(value: str) -> str:
        return {
            "strong": "แข็งแรง",
            "weak": "อ่อนแอ",
            "neutral": "ทรงตัว",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_earnings_quality_label(value: str) -> str:
        return {
            "core_business_strong": "กำไรหลักแข็งแรง",
            "one_off_or_low_quality": "คุณภาพกำไรต่ำหรือมีรายการพิเศษ",
            "mixed": "คุณภาพกำไรผสม",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_one_off_risk(value: str) -> str:
        return {
            "low": "ต่ำ",
            "medium": "ปานกลาง",
            "high": "สูง",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_setup_label(value: str) -> str:
        return {
            "favorable setup": "setup ค่อนข้างพร้อม",
            "balanced setup": "setup สมดุล",
            "fragile setup": "setup เปราะบาง",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_valuation_signal(value: str) -> str:
        return {
            "reasonable": "มูลค่าไม่ตึงเกินไป",
            "stretched": "มูลค่าค่อนข้างตึง",
            "neutral": "มูลค่ากลาง ๆ",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    @staticmethod
    def _humanize_expectation_signal(value: str) -> str:
        return {
            "high": "ตลาดคาดหวังสูง",
            "contained": "ตลาดคาดหวังไม่สูงมาก",
            "moderate": "ตลาดคาดหวังปานกลาง",
        }.get(str(value or "").strip().casefold(), str(value or "").strip() or "n/a")

    def _build_post_earnings_action(self, item: EarningsInterpretation) -> str:
        if item.stance == "bullish":
            if item.one_off_risk == "low" and item.earnings_quality_label == "core_business_strong":
                return "ติดตามแรงซื้อหลังงบและทยอยเพิ่มน้ำหนักได้ถ้าราคาไม่ไล่ไกลเกินไป"
            return "มองบวกได้ แต่ควรรอ price action ยืนยันก่อนเพิ่มน้ำหนักมาก"
        if item.one_off_risk == "high":
            return "หลีกเลี่ยงการรีบรับมีด และรอให้ตลาดแยกผลบวกจริงออกจากรายการพิเศษก่อน"
        return "ชะลอการเปิดสถานะใหม่ และรอให้ guidance กับราคาเริ่มนิ่งก่อนประเมินซ้ำ"

    def _build_pre_earnings_action(self, item: PreEarningsRiskSignal) -> str:
        if item.risk_score >= 7.5:
            return "ลดขนาดสถานะก่อนงบหรือรอหลังประกาศ เพื่อลดความเสี่ยงจากความคาดหวังที่สูงเกินไป"
        if item.risk_score >= 6.0:
            return "ยังไม่ควรไล่ซื้อก่อนงบ และควรกำหนดขนาดสถานะให้เล็กกว่าปกติ"
        return "ติดตามต่อได้ แต่ควรวางแผนรับความผันผวนในวันประกาศงบ"

    @staticmethod
    def _build_earnings_calendar_action(days_left: int) -> str:
        if days_left <= 0:
            return "ถ้ามีสถานะอยู่แล้ว ให้ทบทวนขนาดความเสี่ยงก่อนตลาดเปิดหรือก่อนประกาศ"
        if days_left <= 1:
            return "หลีกเลี่ยงการเพิ่มสถานะใหญ่ก่อนงบ เว้นแต่ยอมรับความผันผวนได้"
        return "จดวันประกาศไว้ล่วงหน้าและเตรียมแผนรับผลบวกหรือลบหลังงบ"

    def _build_earnings_setup_action(self, item: EarningsSetupCandidate) -> str:
        if item.setup_score >= 2.4 and item.valuation_signal == "reasonable":
            return "ติดตามเป็นตัวเลือกก่อนงบได้ แต่ยังควรแบ่งไม้และไม่เปิดสถานะเต็มขนาด"
        if item.expectation_signal == "high" or item.setup_label == "fragile setup":
            return "เฝ้าดูต่อมากกว่าลุยก่อนงบ เพราะ downside จากความคาดหวังยังสูง"
        return "เก็บไว้ใน watchlist ก่อนงบ และรอราคา/volume ยืนยันเพิ่ม"

    def _build_stock_candidate_action(self, candidate: StockCandidate, confidence_score: float, *, mode: str) -> str:
        if candidate.stance == "buy":
            if confidence_score >= 0.7:
                return "ทยอยสะสมได้เป็นไม้เล็ก หากน้ำหนักในพอร์ตยังไม่มากเกินไป"
            return "เริ่มติดตามจุดเข้าได้ แต่ควรรอราคาและแรงซื้อยืนยันก่อนเพิ่มน้ำหนัก"
        if mode == "watchlist":
            return "คงไว้ใน watchlist และรอสัญญาณราคา/ข่าวยืนยันก่อนเพิ่มน้ำหนัก"
        return "เฝ้าดูต่อและรอจังหวะที่ risk/reward ชัดขึ้นก่อนเปิดสถานะ"

    @staticmethod
    def _detect_asset_scope(question: str) -> AssetScope:
        normalized = question.casefold()
        if any(keyword in normalized for keyword in ("ทอง", "gold", "xau", "gld", "iau")):
            return "gold-only"
        if any(keyword in normalized for keyword in ("bond", "พันธบัตร", "tlt", "yield")):
            return "bonds"
        if "etf" in normalized:
            return "etf-only"
        if any(keyword in normalized for keyword in ("หุ้นสหรัฐ", "หุ้นเมกา", "us stock", "nasdaq", "s&p", "spy", "qqq")):
            return "us-stocks"
        return "all"

    @staticmethod
    def _is_stock_screener_question(question: str) -> bool:
        normalized = question.casefold()
        pick_keywords = (
            "หุ้นอะไร",
            "หุ้นตัวไหน",
            "ตัวไหนน่าซื้อ",
            "top 5",
            "5 ตัว",
            "หุ้นเด่น",
            "stock pick",
            "stock screener",
            "nasdaq 100",
            "s&p 500",
            "หุ้นรายตัว",
        )
        return any(keyword in normalized for keyword in pick_keywords)

    @staticmethod
    def _extract_requested_pick_count(question: str) -> int:
        normalized = question.casefold()
        for value in range(10, 2, -1):
            if str(value) in normalized:
                return value
        return 5

    @staticmethod
    def _match_stock_mentions(question: str) -> list[StockUniverseMember]:
        deduped: dict[str, StockUniverseMember] = {}
        for member in find_stock_candidates_from_text(question):
            deduped.setdefault(member.ticker, member)
        for token in re.findall(r"\b[A-Z]{1,5}(?:-[A-Z])?\b", question):
            normalized = token.strip().upper()
            if normalized in {"ETF", "VIX", "CPI"}:
                continue
            deduped.setdefault(
                normalized,
                StockUniverseMember(
                    ticker=normalized,
                    company_name=normalized,
                    sector="Unknown",
                    benchmark="custom",
                ),
            )
        return list(deduped.values())

    @staticmethod
    def _asset_display_name(asset_name: object) -> str:
        return {
            "gold_futures": "ทองคำ",
            "sp500_index": "ดัชนี S&P 500",
            "nasdaq_index": "ดัชนี NASDAQ",
            "spy_etf": "ETF SPY",
            "qqq_etf": "ETF QQQ",
            "gld_etf": "ETF GLD",
            "iau_etf": "ETF IAU",
            "vti_etf": "ETF VTI",
            "xlf_etf": "ETF XLF",
            "xle_etf": "ETF XLE",
            "xlk_etf": "ETF XLK",
            "xly_etf": "ETF XLY",
            "xlp_etf": "ETF XLP",
            "xlv_etf": "ETF XLV",
            "xli_etf": "ETF XLI",
            "xlb_etf": "ETF XLB",
            "xlu_etf": "ETF XLU",
            "xlc_etf": "ETF XLC",
            "xlre_etf": "ETF XLRE",
            "tlt_etf": "ETF TLT",
            "voo_etf": "ETF VOO",
        }.get(str(asset_name), str(asset_name))

    def _humanize_signals(self, signals: object) -> str:
        if not isinstance(signals, list) or not signals:
            return "สัญญาณยังไม่เด่นชัด"
        phrase_map = {
            "price_above_fast_and_slow_ema": "ราคาอยู่เหนือ EMA หลัก",
            "price_below_fast_and_slow_ema": "ราคาอยู่ใต้ EMA หลัก",
            "price_mixed_vs_ema": "ราคาและ EMA ยังไม่เรียงตัวชัด",
            "ema_spread_bullish": "โครงสร้าง EMA ยังหนุนฝั่งบวก",
            "ema_spread_bearish": "โครงสร้าง EMA ยังหนุนฝั่งลบ",
            "ema_spread_flat": "EMA ยังไม่ให้แรงส่งชัด",
            "rsi_above_55": "RSI อยู่ฝั่งบวก",
            "rsi_below_45": "RSI ยังอ่อนแรง",
            "rsi_neutral": "RSI ยังเป็นกลาง",
            "macd_bullish": "MACD ยังสนับสนุนบวก",
            "macd_bearish": "MACD ยังสนับสนุนลบ",
            "macd_neutral": "MACD ยังไม่ยืนยันทิศทาง",
        }
        return " และ ".join(phrase_map.get(str(item), str(item)) for item in signals[:DEFAULT_REASON_LIMIT])

    def _build_market_overview(self, asset_snapshots: list[Any], scope: str, portfolio_plan: Any) -> str:
        trend_counts = {"uptrend": 0, "downtrend": 0, "sideways": 0}
        for asset in asset_snapshots:
            if isinstance(asset, Mapping):
                trend = str(asset.get("trend") or "sideways")
                trend_counts[trend] = trend_counts.get(trend, 0) + 1
        scope_text = {
            "gold-only": "โฟกัสเฉพาะทองคำ",
            "us-stocks": "โฟกัสหุ้นสหรัฐและดัชนีหลัก",
            "etf-only": "โฟกัส ETF ที่เป็นแกนของพอร์ต",
            "bonds": "โฟกัสสินทรัพย์ป้องกันความเสี่ยงฝั่งตราสารหนี้",
            "all": "โฟกัสภาพรวมตลาดหลัก",
        }.get(scope, "โฟกัสภาพรวมตลาด")
        regime_summary = self._extract_portfolio_value(portfolio_plan, "regime_summary")
        if trend_counts["downtrend"] > trend_counts["uptrend"]:
            tone = "สินทรัพย์ส่วนใหญ่ยังอ่อนแรงกว่าฝั่งบวก"
        elif trend_counts["uptrend"] > trend_counts["downtrend"]:
            tone = "สินทรัพย์ส่วนใหญ่ยังรักษาโมเมนตัมเชิงบวกได้"
        else:
            tone = "โมเมนตัมยังผสมกันและต้องเลือกตัวอย่างระมัดระวัง"
        return f"มุมมองตลาด: {scope_text} | {tone}" + (f" | {regime_summary}" if regime_summary else "")

    @staticmethod
    def _extract_portfolio_value(portfolio_plan: Any, key: str) -> str | None:
        if not isinstance(portfolio_plan, Mapping):
            return None
        value = portfolio_plan.get(key)
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _format_allocation_summary(portfolio_plan: Any) -> str:
        if not isinstance(portfolio_plan, Mapping):
            return "ยังไม่มีแผนจัดพอร์ต"
        allocations = portfolio_plan.get("allocations")
        if not isinstance(allocations, list):
            return "ยังไม่มีแผนจัดพอร์ต"
        parts = [f"{bucket.get('label')} {bucket.get('target_pct')}%" for bucket in allocations if isinstance(bucket, Mapping)]
        return " | ".join(parts) if parts else "ยังไม่มีแผนจัดพอร์ต"

    @staticmethod
    def _select_focus_assets(asset_snapshots: Sequence[Any], *, limit: int) -> list[Mapping[str, Any]]:
        ranked: list[tuple[float, Mapping[str, Any]]] = []
        for asset in asset_snapshots:
            if not isinstance(asset, Mapping):
                continue
            trend_score = RecommendationService._as_float(asset.get("trend_score")) or 0.0
            ranked.append((abs(trend_score), asset))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [asset for _, asset in ranked[:limit]]

    @staticmethod
    def _determine_fallback_verbosity(*, question: str | None, asset_scope: AssetScope) -> FallbackVerbosity:
        if not question:
            return "medium"
        normalized = question.casefold()
        short_checks = ("น่าเข้าไหม", "เข้าดีไหม", "ควรถือไหม", "ตอนนี้", "quick take", "quick check")
        short_keywords = ("สั้นมาก", "สั้น", "ย่อ", "สรุปเร็ว", "เร็ว", "quick", "brief", "short")
        medium_keywords = ("ภาพรวม", "สรุปตลาด", "market update", "trend", "overview")
        detailed_keywords = ("ละเอียด", "detail", "detailed", "deep", "ลึก", "เพราะอะไร", "เหตุผล", "why", "เปรียบเทียบ", "จัดพอร์ต", "พอร์ต", "allocation", "strategy", "entry", "exit", "แนวรับ", "แนวต้าน")
        if any(keyword in normalized for keyword in detailed_keywords):
            return "detailed"
        if any(keyword in normalized for keyword in short_keywords):
            return "short"
        if any(keyword in normalized for keyword in medium_keywords) and asset_scope == "all":
            return "medium"
        if any(keyword in normalized for keyword in short_checks) and asset_scope != "all":
            return "short"
        if asset_scope != "all" and len(normalized) <= 80:
            return "short"
        if any(keyword in normalized for keyword in ("ไหม", "มั้ย", "หรือ", "ควร")):
            return "short"
        return "medium"

    def _guard_news_articles(self, news: Sequence[NewsArticle]) -> list[NewsArticle]:
        cleaned: list[NewsArticle] = []
        seen: set[tuple[str, str]] = set()
        for item in news:
            title = (item.title or "").strip()
            link = (item.link or "").strip()
            if not title or not link:
                continue
            dedupe_key = (title.casefold(), link)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cleaned.append(item)
        if len(cleaned) < len(news):
            log_event("data_quality_guard", source="news", kept=len(cleaned), dropped=max(0, len(news) - len(cleaned)))
        return cleaned

    def _guard_market_snapshot(self, market_data: Mapping[str, AssetQuote | None]) -> dict[str, AssetQuote | None]:
        cleaned: dict[str, AssetQuote | None] = {}
        dropped = 0
        for asset_name, quote in market_data.items():
            if quote is None:
                cleaned[asset_name] = None
                continue
            if quote.price <= 0:
                cleaned[asset_name] = None
                dropped += 1
                continue
            cleaned[asset_name] = quote
        if dropped:
            log_event("data_quality_guard", source="market_snapshot", dropped=dropped, kept=len(cleaned) - dropped)
        return cleaned

    def _guard_macro_context(self, macro_context: Mapping[str, float | None]) -> dict[str, float | None]:
        cleaned = dict(macro_context)
        sanitized = 0
        checks = {
            "vix": (0.0, 120.0),
            "tnx": (-5.0, 20.0),
            "cpi_yoy": (-10.0, 25.0),
        }
        for key, (lower, upper) in checks.items():
            value = self._as_float(cleaned.get(key))
            if value is None:
                cleaned[key] = None
                continue
            if value < lower or value > upper:
                cleaned[key] = None
                sanitized += 1
            else:
                cleaned[key] = round(value, 3)
        if sanitized:
            log_event("data_quality_guard", source="macro_context", sanitized=sanitized)
        return cleaned

    def _guard_research_findings(self, findings: Sequence[ResearchFinding]) -> list[ResearchFinding]:
        cleaned: list[ResearchFinding] = []
        dropped = 0
        for item in findings:
            if not (item.title or "").strip() and not (item.snippet or "").strip():
                dropped += 1
                continue
            cleaned.append(item)
        if dropped:
            log_event("data_quality_guard", source="research", dropped=dropped, kept=len(cleaned))
        return cleaned

    @staticmethod
    def _bars_to_frame(bars: Sequence[OhlcvBar]) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
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
        )
        if frame.empty:
            return frame
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
        return frame.reset_index(drop=True)

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

    @staticmethod
    def _as_float(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_percent_delta(value: object) -> str:
        numeric = RecommendationService._as_float(value)
        if numeric is None:
            return "n/a"
        return f"{numeric:+.1%}"

    @staticmethod
    def _relative_to_sector_median(value: object, median: object) -> float | None:
        numeric_value = RecommendationService._as_float(value)
        numeric_median = RecommendationService._as_float(median)
        if numeric_value is None or numeric_median is None:
            return None
        return numeric_value - numeric_median

    @staticmethod
    def _classify_relative_signal(delta: object, *, threshold: float) -> str:
        numeric_delta = RecommendationService._as_float(delta)
        if numeric_delta is None:
            return "in-line"
        if numeric_delta >= threshold:
            return "above_sector"
        if numeric_delta <= -threshold:
            return "below_sector"
        return "in-line"

    @staticmethod
    def _compute_trailing_return(frame: pd.DataFrame, *, window: int) -> float:
        if frame.empty or "close" not in frame.columns:
            return 0.0
        closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
        if closes.empty:
            return 0.0
        if len(closes) <= window:
            start = float(closes.iloc[0])
        else:
            start = float(closes.iloc[-(window + 1)])
        end = float(closes.iloc[-1])
        if start == 0:
            return 0.0
        return (end - start) / abs(start)

    @staticmethod
    def _extract_revenue_expectation_gap(text: str) -> float | None:
        normalized = text.casefold()
        explicit_gap = re.search(
            r"(?:revenue|sales|top-line)[^.%]{0,80}?(?:beat|beats|above|miss|missed|below)[^.%]{0,30}?([+-]?\d+(?:\.\d+)?)\s*%",
            normalized,
        )
        if explicit_gap:
            value = RecommendationService._as_float(explicit_gap.group(1))
            if value is None:
                return None
            if any(keyword in normalized for keyword in ("miss", "below")) and value > 0:
                return -value / 100.0
            return value / 100.0
        by_gap = re.search(
            r"(?:revenue|sales|top-line)[^.%]{0,80}?(?:beat|beats|above|miss|missed|below)[^.%]{0,20}?by\s+([+-]?\d+(?:\.\d+)?)\s*%",
            normalized,
        )
        if by_gap:
            value = RecommendationService._as_float(by_gap.group(1))
            if value is None:
                return None
            if any(keyword in normalized for keyword in ("miss", "below")):
                return -value / 100.0
            return value / 100.0
        return None

    @staticmethod
    def _build_ascii_sparkline(values: Sequence[float]) -> str:
        if not values:
            return ""
        chars = "._-^#"
        normalized = [max(0.0, min(1.0, float(value))) for value in values]
        return "".join(chars[min(len(chars) - 1, int(round(value * (len(chars) - 1))))] for value in normalized)

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:80]
