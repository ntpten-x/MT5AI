from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
from loguru import logger

from invest_advisor_bot.analytics_store import AnalyticsStore
from invest_advisor_bot.analytics_warehouse import AnalyticsWarehouse
from invest_advisor_bot.backtesting import BacktestingEngine
from invest_advisor_bot.braintrust_observer import BraintrustObserver
from invest_advisor_bot.dbt_semantic_layer import DbtSemanticLayer
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
from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore
from invest_advisor_bot.data_quality import ReasoningDataQualityGate
from invest_advisor_bot.evidently_observer import EvidentlyObserver
from invest_advisor_bot.event_bus import EventBus
from invest_advisor_bot.event_bus_worker import EventBusConsumerWorker
from invest_advisor_bot.feature_store import FeatureStoreBridge
from invest_advisor_bot.hot_path_cache import HotPathCache
from invest_advisor_bot.human_review_store import HumanReviewStore
from invest_advisor_bot.langfuse_observer import LangfuseObserver
from invest_advisor_bot.analysis.asset_ranking import RankedAsset, rank_asset_snapshots
from invest_advisor_bot.analysis.news_impact import NewsImpact, score_news_impacts
from invest_advisor_bot.analysis.risk_score import RiskScoreAssessment, calculate_risk_score
from invest_advisor_bot.analysis.stock_screener import StockCandidate, rank_stock_universe
from invest_advisor_bot.analysis.trend_engine import TrendAssessment, evaluate_trend
from invest_advisor_bot.providers.llm_client import OpenAICompatibleLLMClient
from invest_advisor_bot.providers.broker_client import ExecutionSandboxClient
from invest_advisor_bot.providers.market_data_client import (
    AnalystExpectationProfile,
    AssetQuote,
    CompanyIntelligence,
    ETFExposureProfile,
    EarningsEvent,
    MacroEvent,
    MacroMarketReaction,
    MacroReactionAssetMove,
    MacroSurpriseSignal,
    MarketDataClient,
    OhlcvBar,
    RecentEarningsResult,
    StockFundamentals,
)
from invest_advisor_bot.providers.microstructure_client import MicrostructureClient, MicrostructureSnapshot
from invest_advisor_bot.providers.news_client import NewsArticle, NewsClient
from invest_advisor_bot.providers.order_flow_client import OrderFlowClient, OrderFlowSnapshot
from invest_advisor_bot.providers.ownership_client import OwnershipIntelligence, OwnershipIntelligenceClient
from invest_advisor_bot.providers.policy_feed_client import PolicyFeedClient, PolicyFeedEvent
from invest_advisor_bot.providers.research_client import ResearchClient, ResearchFinding
from invest_advisor_bot.providers.transcript_client import EarningsTranscriptClient, TranscriptInsight
from invest_advisor_bot.mlflow_observer import MLflowObserver
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.semantic_analyst import SemanticAnalyst
from invest_advisor_bot.thesis_vector_store import ThesisVectorStore
from invest_advisor_bot.universe import (
    StockUniverseMember,
    US_LARGE_CAP_STOCK_UNIVERSE,
    filter_stock_universe_members,
    find_stock_candidates_from_text,
)

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

MACRO_CONTEXT_LABELS: dict[str, str] = {
    "vix": "VIX",
    "tnx": "US 10Y Yield",
    "cpi_yoy": "US CPI YoY",
    "core_cpi_yoy": "Core CPI YoY",
    "ppi_yoy": "PPI YoY",
    "yield_spread_10y_2y": "2s10s Spread",
    "high_yield_spread": "HY Spread",
    "mortgage_30y": "30Y Mortgage",
    "financial_conditions_index": "Financial Conditions",
    "unemployment_rate": "Unemployment Rate",
    "payrolls_mom_k": "Payrolls MoM (k)",
    "payrolls_revision_k": "Payroll Revision (k)",
    "avg_interest_rate_pct": "Treasury Avg Rate",
    "operating_cash_balance_b": "TGA Balance (B)",
    "public_debt_total_t": "Public Debt (T)",
    "wti_usd": "WTI",
    "brent_usd": "Brent",
    "gasoline_usd_gal": "Gasoline",
    "natgas_usd_mmbtu": "Nat Gas",
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
class MacroPostEventPlaybook:
    playbook_key: str
    title: str
    trigger: str
    action: str
    risk_watch: str
    confidence: str
    confidence_score: float
    learning_note: str
    event_key: str


@dataclass(slots=True, frozen=True)
class RegimeSpecificPlaybook:
    playbook_key: str
    regime: str
    title: str
    action: str
    risk_watch: str
    conviction: str
    sizing_bias: str
    ttl_bias: str
    rationale: tuple[str, ...]


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
        runtime_history_store: RuntimeHistoryStore | None = None,
        thesis_memory_top_k: int = 3,
        data_quality_gate: ReasoningDataQualityGate | None = None,
        analytics_store: AnalyticsStore | None = None,
        analytics_warehouse: AnalyticsWarehouse | None = None,
        event_bus: EventBus | None = None,
        event_bus_consumer: EventBusConsumerWorker | None = None,
        hot_path_cache: HotPathCache | None = None,
        thesis_vector_store: ThesisVectorStore | None = None,
        feature_store: FeatureStoreBridge | None = None,
        backtesting_engine: BacktestingEngine | None = None,
        semantic_analyst: SemanticAnalyst | None = None,
        evidently_observer: EvidentlyObserver | None = None,
        braintrust_observer: BraintrustObserver | None = None,
        broker_client: ExecutionSandboxClient | None = None,
        transcript_client: EarningsTranscriptClient | None = None,
        microstructure_client: MicrostructureClient | None = None,
        ownership_client: OwnershipIntelligenceClient | None = None,
        order_flow_client: OrderFlowClient | None = None,
        policy_feed_client: PolicyFeedClient | None = None,
        dbt_semantic_layer: DbtSemanticLayer | None = None,
        langfuse_observer: LangfuseObserver | None = None,
        human_review_store: HumanReviewStore | None = None,
        mlflow_observer: MLflowObserver | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt_path = Path(system_prompt_path or DEFAULT_PROMPT_PATH)
        self.chat_history_limit = max(1, int(chat_history_limit))
        self.default_investor_profile = normalize_profile_name(default_investor_profile)
        self.runtime_history_store = runtime_history_store
        self.thesis_memory_top_k = max(1, int(thesis_memory_top_k))
        self.data_quality_gate = data_quality_gate
        self.analytics_store = analytics_store
        self.analytics_warehouse = analytics_warehouse
        self.event_bus = event_bus
        self.event_bus_consumer = event_bus_consumer
        self.hot_path_cache = hot_path_cache
        self.thesis_vector_store = thesis_vector_store
        self.feature_store = feature_store
        self.backtesting_engine = backtesting_engine
        self.semantic_analyst = semantic_analyst
        self.evidently_observer = evidently_observer
        self.braintrust_observer = braintrust_observer
        self.broker_client = broker_client
        self.transcript_client = transcript_client
        self.microstructure_client = microstructure_client
        self.ownership_client = ownership_client
        self.order_flow_client = order_flow_client
        self.policy_feed_client = policy_feed_client
        self.dbt_semantic_layer = dbt_semantic_layer
        self.langfuse_observer = langfuse_observer
        self.human_review_store = human_review_store
        self.mlflow_observer = mlflow_observer
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

    async def answer_analytics_question(self, *, question: str) -> str:
        if self.semantic_analyst is None:
            return "semantic analyst ยังไม่ได้ถูกตั้งค่าในระบบ"
        return await self.semantic_analyst.analyze(question)

    def status(self) -> dict[str, Any]:
        mlflow_status = self.mlflow_observer.status() if self.mlflow_observer is not None else None
        data_quality_status = self.data_quality_gate.status() if self.data_quality_gate is not None else None
        analytics_status = self.analytics_store.status() if self.analytics_store is not None else None
        analytics_warehouse_status = self.analytics_warehouse.status() if self.analytics_warehouse is not None else None
        event_bus_status = self.event_bus.status() if self.event_bus is not None else None
        event_bus_consumer_status = self.event_bus_consumer.status() if self.event_bus_consumer is not None else None
        hot_path_cache_status = self.hot_path_cache.status() if self.hot_path_cache is not None else None
        thesis_vector_store_status = self.thesis_vector_store.status() if self.thesis_vector_store is not None else None
        feature_store_status = self.feature_store.status() if self.feature_store is not None else None
        backtesting_status = self.backtesting_engine.status() if self.backtesting_engine is not None else None
        semantic_analyst_status = self.semantic_analyst.status() if self.semantic_analyst is not None else None
        evidently_status = self.evidently_observer.status() if self.evidently_observer is not None else None
        braintrust_status = self.braintrust_observer.status() if self.braintrust_observer is not None else None
        broker_status = self.broker_client.status() if self.broker_client is not None else None
        transcript_status = self.transcript_client.status() if self.transcript_client is not None else None
        microstructure_status = self.microstructure_client.status() if self.microstructure_client is not None else None
        ownership_status = self.ownership_client.status() if self.ownership_client is not None else None
        order_flow_status = self.order_flow_client.status() if self.order_flow_client is not None else None
        policy_feed_status = self.policy_feed_client.status() if self.policy_feed_client is not None else None
        dbt_semantic_status = self.dbt_semantic_layer.status() if self.dbt_semantic_layer is not None else None
        langfuse_status = self.langfuse_observer.status() if self.langfuse_observer is not None else None
        human_review_status = self.human_review_store.status() if self.human_review_store is not None else None
        return {
            "available": True,
            "llm": self.llm_client.status(),
            "system_prompt_path": str(self.system_prompt_path),
            "system_prompt_exists": self.system_prompt_path.exists(),
            "chat_history_limit": self.chat_history_limit,
            "default_investor_profile": self.default_investor_profile,
            "thesis_memory_top_k": self.thesis_memory_top_k,
            "runtime_history_enabled": self.runtime_history_store is not None,
            "active_conversations": len(self._conversation_history),
            "profile_overrides": len(self._conversation_profiles),
            "data_quality": dict(data_quality_status) if isinstance(data_quality_status, Mapping) else None,
            "analytics_store": dict(analytics_status) if isinstance(analytics_status, Mapping) else None,
            "analytics_warehouse": dict(analytics_warehouse_status) if isinstance(analytics_warehouse_status, Mapping) else None,
            "event_bus": dict(event_bus_status) if isinstance(event_bus_status, Mapping) else None,
            "event_bus_consumer": dict(event_bus_consumer_status) if isinstance(event_bus_consumer_status, Mapping) else None,
            "hot_path_cache": dict(hot_path_cache_status) if isinstance(hot_path_cache_status, Mapping) else None,
            "thesis_vector_store": dict(thesis_vector_store_status) if isinstance(thesis_vector_store_status, Mapping) else None,
            "feature_store": dict(feature_store_status) if isinstance(feature_store_status, Mapping) else None,
            "backtesting": dict(backtesting_status) if isinstance(backtesting_status, Mapping) else None,
            "semantic_analyst": dict(semantic_analyst_status) if isinstance(semantic_analyst_status, Mapping) else None,
            "evidently": dict(evidently_status) if isinstance(evidently_status, Mapping) else None,
            "braintrust": dict(braintrust_status) if isinstance(braintrust_status, Mapping) else None,
            "broker": dict(broker_status) if isinstance(broker_status, Mapping) else None,
            "transcripts": dict(transcript_status) if isinstance(transcript_status, Mapping) else None,
            "microstructure": dict(microstructure_status) if isinstance(microstructure_status, Mapping) else None,
            "ownership": dict(ownership_status) if isinstance(ownership_status, Mapping) else None,
            "order_flow": dict(order_flow_status) if isinstance(order_flow_status, Mapping) else None,
            "policy_feed": dict(policy_feed_status) if isinstance(policy_feed_status, Mapping) else None,
            "dbt_semantic_layer": dict(dbt_semantic_status) if isinstance(dbt_semantic_status, Mapping) else None,
            "langfuse": dict(langfuse_status) if isinstance(langfuse_status, Mapping) else None,
            "human_review": dict(human_review_status) if isinstance(human_review_status, Mapping) else None,
            "mlflow": dict(mlflow_status) if isinstance(mlflow_status, Mapping) else None,
        }

    def list_pending_reviews(self, *, limit: int = 10) -> list[dict[str, Any]]:
        if self.human_review_store is None:
            return []
        try:
            return self.human_review_store.list_pending(limit=limit)
        except Exception as exc:
            logger.warning("Failed to list pending reviews: {}", exc)
            return []

    def complete_human_review(
        self,
        *,
        review_id: str,
        decision: str,
        score: float | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        if self.human_review_store is None:
            return None
        try:
            completed = self.human_review_store.complete_review(
                review_id=review_id,
                decision=decision,
                score=score,
                note=note,
            )
        except Exception as exc:
            logger.warning("Failed to complete human review {}: {}", review_id, exc)
            return None
        if completed and self.langfuse_observer is not None:
            try:
                self.langfuse_observer.log_human_review(
                    review_id=str(completed.get("review_id") or review_id),
                    artifact_key=str(completed.get("artifact_key") or ""),
                    decision=str(completed.get("decision") or decision),
                    score=self._as_float(completed.get("score")),
                    note=str(completed.get("note") or note or "").strip() or None,
                    metadata=completed,
                )
            except Exception as exc:
                logger.warning("Failed to log Langfuse human review: {}", exc)
        return completed

    async def generate_recommendation(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        trends: Mapping[str, TrendAssessment],
        macro_context: Mapping[str, float | None] | None = None,
        macro_intelligence: Mapping[str, Any] | None = None,
        macro_event_calendar: Sequence[MacroEvent] | None = None,
        macro_surprise_signals: Sequence[MacroSurpriseSignal] | None = None,
        macro_market_reactions: Sequence[MacroMarketReaction] | None = None,
        research_findings: Sequence[ResearchFinding] | None = None,
        portfolio_snapshot: Mapping[str, Any] | None = None,
        question: str | None = None,
        conversation_key: str | None = None,
        asset_scope: AssetScope = "all",
        fallback_verbosity_override: FallbackVerbosity | None = None,
        investor_profile_name: InvestorProfileName | None = None,
        etf_exposures: Sequence[ETFExposureProfile] | None = None,
        policy_events: Sequence[PolicyFeedEvent] | None = None,
        ownership_intelligence: Sequence[OwnershipIntelligence] | None = None,
        order_flow: Sequence[OrderFlowSnapshot] | None = None,
    ) -> RecommendationResult:
        effective_profile = self._resolve_investor_profile(
            question=question,
            conversation_key=conversation_key,
            investor_profile_name=investor_profile_name,
        )
        system_prompt = self._load_system_prompt()
        thesis_memory = self._load_thesis_memory(question=question, conversation_key=conversation_key)
        payload = self._build_payload(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            asset_scope=asset_scope,
            question=question,
            investor_profile=effective_profile,
            thesis_memory=thesis_memory,
            etf_exposures=etf_exposures,
            policy_events=policy_events,
            ownership_intelligence=ownership_intelligence,
            order_flow=order_flow,
        )
        data_quality_report = self._evaluate_data_quality(
            news=news,
            market_data=market_data,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            research_findings=research_findings,
        )
        payload["data_quality"] = data_quality_report
        user_prompt = self._build_prompt(
            payload=payload,
            question=question,
            history_lines=self._get_history_lines(conversation_key),
        )

        if bool((data_quality_report or {}).get("blocking")):
            fallback_text = self._build_fallback_question_answer(
                question=question,
                payload=payload,
                verbosity="detailed",
            ) if question else self._build_fallback_summary(payload, verbosity="detailed")
            diagnostics.record_response(service="recommendation_service", fallback_used=True)
            self._remember_turns(conversation_key=conversation_key, user_text=question, assistant_text=fallback_text)
            self._record_learning_artifacts(
                payload=payload,
                question=question,
                conversation_key=conversation_key,
                recommendation_text=fallback_text,
                model=None,
                response_id=None,
                fallback_used=True,
                service_name="recommendation_service",
            )
            return RecommendationResult(
                recommendation_text=fallback_text,
                model=None,
                system_prompt_path=str(self.system_prompt_path),
                input_payload=payload,
                fallback_used=True,
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
            self._record_learning_artifacts(
                payload=payload,
                question=question,
                conversation_key=conversation_key,
                recommendation_text=fallback_text,
                model=None,
                response_id=None,
                fallback_used=True,
                service_name="recommendation_service",
            )
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
        self._record_learning_artifacts(
            payload=payload,
            question=question,
            conversation_key=conversation_key,
            recommendation_text=llm_response.text,
            model=llm_response.model,
            response_id=llm_response.response_id,
            fallback_used=False,
            service_name="recommendation_service",
        )
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
        news, market_data, trends, macro_context, research_findings, macro_intelligence, macro_event_calendar, macro_surprise_signals, macro_market_reactions = await self._gather_context(
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
        etf_exposures = await self._fetch_etf_exposures_for_market_data(
            market_data_client=market_data_client,
            market_data=market_data,
        )
        policy_events = await self._fetch_policy_feed_events(limit=6)
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            question="สรุปภาพรวมตลาดโลกและแนวทางจัดพอร์ตล่าสุดแบบอ่านง่าย",
            conversation_key=conversation_key,
            asset_scope=asset_scope,
            etf_exposures=etf_exposures,
            policy_events=policy_events,
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
        news, market_data, trends, macro_context, research_findings, macro_intelligence, macro_event_calendar, macro_surprise_signals, macro_market_reactions = await self._gather_context(
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
        etf_exposures = await self._fetch_etf_exposures_for_market_data(
            market_data_client=market_data_client,
            market_data=market_data,
        )
        policy_events = await self._fetch_policy_feed_events(limit=6)
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            question=normalized_question,
            conversation_key=conversation_key,
            asset_scope=effective_scope,
            fallback_verbosity_override=fallback_verbosity_override,
            investor_profile_name=investor_profile_name,
            etf_exposures=etf_exposures,
            policy_events=policy_events,
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
        transcript_insights = await self._fetch_transcript_insights_for_candidates(
            picks,
            limit=min(4, len(picks)),
            research_client=research_client,
        )
        microstructure_snapshots = await self._fetch_microstructure_for_candidates(picks, limit=min(4, len(picks)))
        ownership_intelligence = await self._fetch_ownership_intelligence_for_candidates(picks, limit=min(4, len(picks)))
        order_flow_snapshots = await self._fetch_order_flow_for_candidates(picks, limit=min(4, len(picks)))
        backtest_summary = await self._build_candidate_backtest_summary(
            market_data_client=market_data_client,
            picks=picks,
        )
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
            transcript_insights=transcript_insights,
            microstructure_snapshots=microstructure_snapshots,
            ownership_intelligence=ownership_intelligence,
            order_flow_snapshots=order_flow_snapshots,
            backtest_summary=backtest_summary,
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
                transcript_insights=transcript_insights,
                microstructure_snapshots=microstructure_snapshots,
                ownership_intelligence=ownership_intelligence,
                order_flow_snapshots=order_flow_snapshots,
                backtest_summary=backtest_summary,
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
        transcript_insights = await self._fetch_transcript_insights_for_candidates(
            picks,
            limit=min(4, len(picks)),
            research_client=research_client,
        )
        microstructure_snapshots = await self._fetch_microstructure_for_candidates(picks, limit=min(4, len(picks)))
        ownership_intelligence = await self._fetch_ownership_intelligence_for_candidates(picks, limit=min(4, len(picks)))
        order_flow_snapshots = await self._fetch_order_flow_for_candidates(picks, limit=min(4, len(picks)))
        backtest_summary = await self._build_candidate_backtest_summary(
            market_data_client=market_data_client,
            picks=picks,
        )
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
            transcript_insights=transcript_insights,
            microstructure_snapshots=microstructure_snapshots,
            ownership_intelligence=ownership_intelligence,
            order_flow_snapshots=order_flow_snapshots,
            backtest_summary=backtest_summary,
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
        news, market_data, trends, macro_context, research_findings, macro_intelligence, macro_event_calendar, macro_surprise_signals, macro_market_reactions = await self._gather_context(
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
        etf_exposures = await self._fetch_etf_exposures_for_market_data(
            market_data_client=market_data_client,
            market_data=market_data,
        )
        policy_events = await self._fetch_policy_feed_events(limit=6)
        return await self.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            question="สรุป Daily Intelligence Report สำหรับนักลงทุนที่ต้องการรักษาและเติบโตทรัพย์สิน",
            fallback_verbosity_override="medium",
            investor_profile_name=self.default_investor_profile,
            etf_exposures=etf_exposures,
            policy_events=policy_events,
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
        news, market_data, trends, macro_context, research_findings, macro_intelligence, macro_event_calendar, macro_surprise_signals, macro_market_reactions = await self._gather_context(
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
        etf_exposures = await self._fetch_etf_exposures_for_market_data(
            market_data_client=market_data_client,
            market_data=market_data,
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
        company_intelligence_map = await self._safe_async_call(
            market_data_client.get_company_intelligence_batch(
                [candidate.ticker for candidate in stock_picks[:4]],
                company_names={candidate.ticker.upper(): candidate.company_name for candidate in stock_picks[:4]},
            ),
            default={},
            source_name="company_intelligence",
        )
        transcript_insights = await self._fetch_transcript_insights_for_candidates(
            stock_picks,
            limit=4,
            research_client=research_client,
        )
        microstructure_snapshots = await self._fetch_microstructure_for_candidates(stock_picks, limit=4)
        ownership_intelligence_map = await self._fetch_ownership_intelligence_for_candidates(stock_picks, limit=4)
        order_flow_snapshots = await self._fetch_order_flow_for_candidates(stock_picks, limit=4)
        policy_events = await self._fetch_policy_feed_events(limit=6)
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
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            asset_scope="all",
            question=question,
            investor_profile=report_profile,
            company_intelligence=[item for item in company_intelligence_map.values() if item is not None],
            etf_exposures=etf_exposures,
            policy_events=policy_events,
            ownership_intelligence=[item for item in ownership_intelligence_map.values() if item is not None],
            order_flow=list(order_flow_snapshots.values()),
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
        payload["company_intelligence"] = [
            self._serialize_company_intelligence(item)
            for item in company_intelligence_map.values()
            if item is not None
        ]
        payload["etf_exposures"] = [self._serialize_etf_exposure_profile(item) for item in etf_exposures]
        payload["management_commentary"] = [
            self._serialize_transcript_insight(item)
            for item in transcript_insights.values()
        ]
        payload["microstructure"] = [
            self._serialize_microstructure_snapshot(item)
            for item in microstructure_snapshots.values()
        ]
        payload["ownership_intelligence"] = [
            self._serialize_ownership_intelligence(item)
            for item in ownership_intelligence_map.values()
            if item is not None
        ]
        payload["order_flow"] = [
            self._serialize_order_flow_snapshot(item)
            for item in order_flow_snapshots.values()
        ]
        payload["policy_events"] = [self._serialize_policy_feed_event(item) for item in policy_events]
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
        news, market_data, trends, macro_context, research_findings, macro_intelligence, macro_event_calendar, macro_surprise_signals, macro_market_reactions = await self._gather_context(
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
        etf_exposures = await self._fetch_etf_exposures_for_market_data(
            market_data_client=market_data_client,
            market_data=market_data,
        )

        alert_profile = get_investor_profile(self.default_investor_profile)
        payload = self._build_payload(
            news=news,
            market_data=market_data,
            trends=trends,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            research_findings=research_findings,
            portfolio_snapshot=portfolio_snapshot,
            asset_scope="all",
            question="continuous market scan",
            investor_profile=alert_profile,
            etf_exposures=etf_exposures,
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

    async def generate_macro_event_driven_alerts(
        self,
        *,
        market_data_client: MarketDataClient,
        pre_window_minutes: int = 20,
        post_window_minutes: int = 90,
        lookahead_hours: int = 12,
    ) -> list[InterestingAlert]:
        now = datetime.now(timezone.utc)
        event_calendar = await self._safe_async_call(
            market_data_client.get_macro_event_calendar(days_ahead=max(1, lookahead_hours // 24 + 2)),
            default=[],
            source_name="macro_event_calendar",
        )
        surprise_signals = await self._safe_async_call(
            market_data_client.get_macro_surprise_signals(),
            default=[],
            source_name="macro_surprise_signals",
        )
        market_reactions = await self._safe_async_call(
            market_data_client.get_macro_market_reactions(),
            default=[],
            source_name="macro_market_reactions",
        )

        event_items = [item for item in event_calendar if isinstance(item, MacroEvent)]
        surprise_items = [item for item in surprise_signals if isinstance(item, MacroSurpriseSignal)]
        reaction_items = [item for item in market_reactions if isinstance(item, MacroMarketReaction)]
        playbooks = self._build_macro_post_event_playbooks(
            macro_surprise_signals=surprise_items,
            macro_market_reactions=reaction_items,
        )
        source_learning_map = self._load_source_learning_map()

        alerts: list[InterestingAlert] = []
        horizon = now + timedelta(hours=max(1, lookahead_hours))
        for event in event_items:
            time_to_event = event.scheduled_at - now
            if timedelta(0) <= time_to_event <= timedelta(minutes=max(1, pre_window_minutes)):
                severity = "info"
                alerts.append(
                    InterestingAlert(
                        key=f"macro_event_watch:{event.event_key}:{event.scheduled_at.isoformat()}",
                        severity=severity,
                        text=(
                            f"{self._format_badged_title('🔎 จับตา', 'macro event window')}\n"
                            f"- ภาพรวม: {event.event_name} จะออกเวลา {event.scheduled_at.isoformat()}\n"
                            f"- เหตุผล: เข้าสู่ pre-event window {int(time_to_event.total_seconds() // 60)} นาที | importance {event.importance}\n"
                            f"- Action: เตรียมดู surprise และ cross-asset confirmation หลังเลขออก"
                        ),
                        metadata=self._build_alert_metadata(alert_kind="macro_event", severity=severity),
                    )
                )
            elif now <= event.scheduled_at <= horizon:
                severity = "info"
                alerts.append(
                    InterestingAlert(
                        key=f"macro_event_upcoming:{event.event_key}:{event.scheduled_at.isoformat()}",
                        severity=severity,
                        text=(
                            f"{self._format_badged_title('📅 ติดตาม', 'upcoming macro event')}\n"
                            f"- ภาพรวม: {event.event_name} | {event.scheduled_at.isoformat()} | importance {event.importance}\n"
                            f"- Action: เตรียม scenario ถ้าเลขออกแรงกว่าคาดและรอตลาด confirm"
                        ),
                        metadata=self._build_alert_metadata(alert_kind="macro_event", severity=severity),
                    )
                )

        reaction_map = {item.event_key: item for item in reaction_items}
        for signal in surprise_items:
            released_at = signal.released_at
            if released_at is None:
                continue
            elapsed = now - released_at
            if elapsed < timedelta(0) or elapsed > timedelta(minutes=max(10, post_window_minutes)):
                continue
            reaction = reaction_map.get(signal.event_key)
            reaction_label = reaction.confirmation_label if reaction is not None else "insufficient_data"
            evidence_score = self._average_source_learning_score(
                source_learning_map,
                (signal.source, "macro_surprise_engine", "macro_market_reaction"),
            )
            severity = self._adjust_alert_severity(
                base_severity="warning" if signal.market_bias and ("risk_off" in signal.market_bias or "defensive" in signal.market_bias or "rates_up" in signal.market_bias) else "info",
                evidence_score=evidence_score,
            )
            alerts.append(
                InterestingAlert(
                    key=f"macro_event_refresh:{signal.event_key}:{released_at.isoformat()}",
                    severity=severity,
                    text=(
                        f"{self._format_badged_title('✅ ยืนยัน' if reaction_label == 'confirmed' else '🟠 ระวัง', 'macro refresh')}\n"
                        f"- ภาพรวม: {signal.event_name} | surprise {signal.surprise_label}\n"
                        f"- เหตุผล: consensus={signal.consensus_surprise_label or 'n/a'} | baseline={signal.baseline_surprise_label or 'n/a'} | reaction={reaction_label}\n"
                        f"- Action: ใช้ price confirmation ร่วมกับ playbook ก่อนปรับพอร์ต"
                    ),
                    metadata=self._build_alert_metadata(
                        alert_kind="macro_surprise",
                        severity=severity,
                        evidence_score=evidence_score,
                        preferred_sources=tuple(name for name in (signal.source, "macro_surprise_engine", "macro_market_reaction") if name),
                    ),
                )
            )

        for playbook in playbooks:
            evidence_score = self._average_source_learning_score(source_learning_map, ("macro_post_event_playbooks",))
            severity = self._adjust_alert_severity(
                base_severity="warning",
                evidence_score=evidence_score,
                confidence_score=playbook.confidence_score,
            )
            alerts.append(
                InterestingAlert(
                    key=f"macro_playbook_window:{playbook.playbook_key}",
                    severity=severity,
                    text=(
                        f"{self._format_badged_title('🔎 จับตา', 'event-driven playbook')}\n"
                        f"- ภาพรวม: {playbook.title}\n"
                        f"- Trigger: {playbook.trigger}\n"
                        f"- Confidence: {playbook.confidence} ({playbook.confidence_score:.2f}) | {playbook.learning_note}\n"
                        f"- Action: {playbook.action}"
                    ),
                    metadata=self._build_alert_metadata(
                        alert_kind="macro_playbook",
                        severity=severity,
                        confidence_score=playbook.confidence_score,
                        evidence_score=evidence_score,
                        preferred_sources=("macro_post_event_playbooks",),
                    ),
                )
            )

        unique: dict[str, InterestingAlert] = {}
        for alert in alerts:
            unique.setdefault(alert.key, alert)
        return list(unique.values())

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
        portfolio_holdings: Sequence[PortfolioHolding] = (),
        approval_mode: str = "auto",
        max_position_size_pct: float | None = None,
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
        portfolio_snapshot = await self._gather_portfolio_snapshot(
            market_data_client=market_data_client,
            holdings=portfolio_holdings,
        )
        factor_exposures = self._build_factor_exposure_summary(portfolio_snapshot=portfolio_snapshot)
        portfolio_constraints = self._build_portfolio_constraint_summary(
            portfolio_snapshot=portfolio_snapshot,
            factor_exposures=factor_exposures,
        )
        alerts: list[InterestingAlert] = []
        today_key = datetime.now(timezone.utc).date().isoformat()
        stock_pick_source_coverage = self._build_stock_pick_alert_source_coverage()
        stock_pick_source_health = self._build_stock_pick_flow_health(
            picks=picks,
            stock_news=stock_news,
            research_findings=research_findings,
            portfolio_constraints=portfolio_constraints,
        )
        stock_pick_no_trade = self._build_stock_pick_no_trade_decision(
            picks=picks,
            portfolio_constraints=portfolio_constraints,
            source_health=stock_pick_source_health,
        )
        if bool(stock_pick_no_trade.get("should_abstain")):
            severity = self._adjust_alert_severity(
                base_severity="warning",
                evidence_score=self._as_float(stock_pick_source_health.get("score")),
            )
            metadata = self._build_alert_metadata(
                alert_kind="stock_pick",
                severity=severity,
                evidence_score=self._as_float(stock_pick_source_health.get("score")),
                preferred_sources=tuple(stock_pick_source_coverage.get("used_sources") or ()),
                extra={
                    "stock_pick": False,
                    "source_kind": "stock_pick_abstain",
                    "source_coverage": stock_pick_source_coverage,
                    "source_health": stock_pick_source_health,
                    "portfolio_constraints": portfolio_constraints,
                    "no_trade_decision": stock_pick_no_trade,
                },
            )
            return [
                InterestingAlert(
                    key=f"stock:abstain:{today_key}",
                    severity=severity,
                    text=(
                        f"{self._format_badged_title('🟠 ระวัง', 'งดเปิด stock pick ตอนนี้')}\n"
                        f"- ภาพรวม: {stock_pick_no_trade.get('summary') or 'execution / portfolio conditions are not clean enough'}\n"
                        f"- เหตุผล: {'; '.join(stock_pick_no_trade.get('reasons') or ['risk budget is constrained'])}\n"
                        f"- Action: {stock_pick_no_trade.get('action') or 'รอคุณภาพสัญญาณและ risk budget ดีขึ้นก่อน'}"
                    ),
                    metadata=metadata,
                )
            ]
        if daily_pick_enabled:
            top_pick = picks[0]
            top_news = stock_news.get(top_pick.asset, [])
            top_confidence = assess_stock_candidate_confidence(top_pick)
            top_plan = self._determine_stock_pick_position_plan(
                source_kind="daily_pick",
                confidence_score=top_confidence.score,
                stance=top_pick.stance,
                candidate=top_pick,
                portfolio_constraints=portfolio_constraints,
                approval_mode=approval_mode,
                max_position_size_pct=max_position_size_pct,
            )
            top_news_line = f"\n- ข่าวประกอบ: {top_news[0].title}" if top_news else ""
            top_severity = self._adjust_alert_severity(
                base_severity="info",
                confidence_score=max(top_confidence.score, top_plan["learning_score"] or 0.0),
            )
            alerts.append(
                InterestingAlert(
                    key=f"stock:daily:{today_key}:{top_pick.ticker}",
                    severity=top_severity,
                    text=(
                        f"{self._format_badged_title('✅ ยืนยัน', 'หุ้นเด่นวันนี้')}\n"
                        f"- หุ้น: {top_pick.company_name} ({top_pick.ticker}) | sector {top_pick.sector}\n"
                        f"- ภาพรวม: คะแนน {top_pick.composite_score:.2f} | ความมั่นใจ {self._humanize_confidence_label(top_confidence.label)} ({top_confidence.score:.2f}) | มุมมอง {self._humanize_stock_stance(top_pick.stance)}\n"
                        f"- เหตุผล: {'; '.join(top_pick.rationale[:3])}"
                        f"{top_news_line}\n"
                        f"- Action: {self._build_stock_candidate_action(top_pick, top_confidence.score, mode='daily', portfolio_constraints=portfolio_constraints, approval_mode=approval_mode, max_position_size_pct=max_position_size_pct)}"
                    ),
                    metadata=self._build_alert_metadata(
                        alert_kind="stock_pick",
                        severity=top_severity,
                        confidence_score=max(top_confidence.score, top_plan["learning_score"] or 0.0),
                        execution_feedback=top_plan.get("execution_feedback"),
                        preferred_sources=tuple(stock_pick_source_coverage.get("used_sources") or ()),
                        extra={
                            "stock_pick": True,
                            "source_kind": "daily_pick",
                            "ticker": top_pick.ticker,
                            "company_name": top_pick.company_name,
                            "sector": top_pick.sector,
                            "benchmark": top_pick.benchmark,
                            "benchmark_ticker": self._resolve_benchmark_ticker(top_pick.benchmark),
                            "peer_benchmark_ticker": top_pick.peer_benchmark_ticker or self._resolve_sector_benchmark_ticker(top_pick.sector),
                            "stance": top_pick.stance,
                            "entry_price": top_pick.price,
                            "composite_score": top_pick.composite_score,
                            "confidence_score": top_confidence.score,
                            "confidence_label": top_confidence.label,
                            "thesis_summary": "; ".join(top_pick.rationale[:2]),
                            "thesis_memory": [{"thesis_text": "; ".join(top_pick.rationale[:2]), "tags": list(top_pick.macro_drivers[:2])}],
                            "macro_drivers": list(top_pick.macro_drivers),
                            "source_coverage": stock_pick_source_coverage,
                            "source_health": stock_pick_source_health,
                            "portfolio_constraints": portfolio_constraints,
                            "no_trade_decision": stock_pick_no_trade,
                            "position_size_pct": top_plan["position_size_pct"],
                            "position_size_tier": top_plan["size_tier"],
                            "execution_realism": top_plan.get("execution_realism"),
                            "approval_state": top_plan.get("approval_state"),
                        },
                    ),
                )
            )
        for candidate in picks[1:]:
            if candidate.composite_score < score_threshold + 0.3:
                continue
            candidate_news = stock_news.get(candidate.asset, [])
            candidate_confidence = assess_stock_candidate_confidence(candidate)
            candidate_plan = self._determine_stock_pick_position_plan(
                source_kind="opportunity_pick",
                confidence_score=candidate_confidence.score,
                stance=candidate.stance,
                candidate=candidate,
                portfolio_constraints=portfolio_constraints,
                approval_mode=approval_mode,
                max_position_size_pct=max_position_size_pct,
            )
            candidate_news_line = f"\n- ข่าวประกอบ: {candidate_news[0].title}" if candidate_news else ""
            research_line = f"\n- วิจัยประกอบ: {research_findings[0].title}" if research_findings else ""
            candidate_severity = self._adjust_alert_severity(
                base_severity="info",
                confidence_score=max(candidate_confidence.score, candidate_plan["learning_score"] or 0.0),
            )
            alerts.append(
                InterestingAlert(
                    key=f"stock:opportunity:{candidate.ticker}:{int(round(candidate.composite_score * 10))}",
                    severity=candidate_severity,
                    text=(
                        f"{self._format_badged_title('🔎 จับตา', 'หุ้นเด่นเพิ่ม')}\n"
                        f"- หุ้น: {candidate.company_name} ({candidate.ticker}) | sector {candidate.sector}\n"
                        f"- ภาพรวม: คะแนน {candidate.composite_score:.2f} | ความมั่นใจ {self._humanize_confidence_label(candidate_confidence.label)} ({candidate_confidence.score:.2f}) | มุมมอง {self._humanize_stock_stance(candidate.stance)}\n"
                        f"- เหตุผล: {'; '.join(candidate.rationale[:3])}"
                        f"{candidate_news_line}"
                        f"{research_line}\n"
                        f"- Action: {self._build_stock_candidate_action(candidate, candidate_confidence.score, mode='opportunity', portfolio_constraints=portfolio_constraints, approval_mode=approval_mode, max_position_size_pct=max_position_size_pct)}"
                    ),
                    metadata=self._build_alert_metadata(
                        alert_kind="stock_pick",
                        severity=candidate_severity,
                        confidence_score=max(candidate_confidence.score, candidate_plan["learning_score"] or 0.0),
                        execution_feedback=candidate_plan.get("execution_feedback"),
                        preferred_sources=tuple(stock_pick_source_coverage.get("used_sources") or ()),
                        extra={
                            "stock_pick": True,
                            "source_kind": "opportunity_pick",
                            "ticker": candidate.ticker,
                            "company_name": candidate.company_name,
                            "sector": candidate.sector,
                            "benchmark": candidate.benchmark,
                            "benchmark_ticker": self._resolve_benchmark_ticker(candidate.benchmark),
                            "peer_benchmark_ticker": candidate.peer_benchmark_ticker or self._resolve_sector_benchmark_ticker(candidate.sector),
                            "stance": candidate.stance,
                            "entry_price": candidate.price,
                            "composite_score": candidate.composite_score,
                            "confidence_score": candidate_confidence.score,
                            "confidence_label": candidate_confidence.label,
                            "thesis_summary": "; ".join(candidate.rationale[:2]),
                            "thesis_memory": [{"thesis_text": "; ".join(candidate.rationale[:2]), "tags": list(candidate.macro_drivers[:2])}],
                            "macro_drivers": list(candidate.macro_drivers),
                            "source_coverage": stock_pick_source_coverage,
                            "source_health": stock_pick_source_health,
                            "portfolio_constraints": portfolio_constraints,
                            "no_trade_decision": stock_pick_no_trade,
                            "position_size_pct": candidate_plan["position_size_pct"],
                            "position_size_tier": candidate_plan["size_tier"],
                            "execution_realism": candidate_plan.get("execution_realism"),
                            "approval_state": candidate_plan.get("approval_state"),
                        },
                    ),
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
                candidate_plan = self._determine_stock_pick_position_plan(
                    source_kind="watchlist_pick",
                    confidence_score=candidate_confidence.score,
                    stance=candidate.stance,
                    candidate=candidate,
                    portfolio_constraints=portfolio_constraints,
                    approval_mode=approval_mode,
                    max_position_size_pct=max_position_size_pct,
                )
                candidate_severity = self._adjust_alert_severity(
                    base_severity="info",
                    confidence_score=max(candidate_confidence.score, candidate_plan["learning_score"] or 0.0),
                )
                alerts.append(
                    InterestingAlert(
                        key=f"watchlist:{candidate.ticker}:{int(round(candidate.composite_score * 10))}",
                        severity=candidate_severity,
                        text=(
                            f"{self._format_badged_title('🔎 จับตา', 'หุ้นใน Watchlist')}\n"
                            f"- หุ้น: {candidate.company_name} ({candidate.ticker})\n"
                            f"- ภาพรวม: คะแนน {candidate.composite_score:.2f} | ความมั่นใจ {self._humanize_confidence_label(candidate_confidence.label)} ({candidate_confidence.score:.2f}) | มุมมอง {self._humanize_stock_stance(candidate.stance)}\n"
                            f"- เหตุผล: {'; '.join(candidate.rationale[:3])}\n"
                            f"- Action: {self._build_stock_candidate_action(candidate, candidate_confidence.score, mode='watchlist', portfolio_constraints=portfolio_constraints, approval_mode=approval_mode, max_position_size_pct=max_position_size_pct)}"
                        ),
                        metadata=self._build_alert_metadata(
                            alert_kind="stock_pick",
                            severity=candidate_severity,
                            confidence_score=max(candidate_confidence.score, candidate_plan["learning_score"] or 0.0),
                            execution_feedback=candidate_plan.get("execution_feedback"),
                            preferred_sources=tuple(stock_pick_source_coverage.get("used_sources") or ()),
                            extra={
                            "stock_pick": True,
                            "source_kind": "watchlist_pick",
                            "ticker": candidate.ticker,
                            "company_name": candidate.company_name,
                            "sector": candidate.sector,
                            "benchmark": candidate.benchmark,
                            "benchmark_ticker": self._resolve_benchmark_ticker(candidate.benchmark),
                            "peer_benchmark_ticker": candidate.peer_benchmark_ticker or self._resolve_sector_benchmark_ticker(candidate.sector),
                            "stance": candidate.stance,
                            "entry_price": candidate.price,
                            "composite_score": candidate.composite_score,
                                "confidence_score": candidate_confidence.score,
                                "confidence_label": candidate_confidence.label,
                                "thesis_summary": "; ".join(candidate.rationale[:2]),
                                "thesis_memory": [{"thesis_text": "; ".join(candidate.rationale[:2]), "tags": list(candidate.macro_drivers[:2])}],
                                "macro_drivers": list(candidate.macro_drivers),
                                "source_coverage": stock_pick_source_coverage,
                                "source_health": stock_pick_source_health,
                                "portfolio_constraints": portfolio_constraints,
                                "no_trade_decision": stock_pick_no_trade,
                            "position_size_pct": candidate_plan["position_size_pct"],
                            "position_size_tier": candidate_plan["size_tier"],
                            "execution_realism": candidate_plan.get("execution_realism"),
                            "approval_state": candidate_plan.get("approval_state"),
                        },
                    ),
                )
                )
        return alerts

    @staticmethod
    def _build_stock_pick_alert_source_coverage() -> dict[str, Any]:
        used_sources = [
            "stock_screener",
            "stock_universe_history",
            "stock_universe_fundamentals",
            "macro_context",
            "factor_risk_engine",
        ]
        return {
            "used_sources": used_sources,
            "counts": {
                "stock_screener": 1,
                "stock_universe_history": 1,
                "stock_universe_fundamentals": 1,
                "macro_context": 1,
                "factor_risk_engine": 1,
            },
        }

    async def generate_sector_rotation_alerts(
        self,
        *,
        news_client: NewsClient,
        market_data_client: MarketDataClient,
        research_client: ResearchClient | None = None,
        sector_rotation_state_store: SectorRotationStateStore | None = None,
        min_streak: int = 3,
    ) -> list[InterestingAlert]:
        news, market_data, trends, _, _, _, _, _, _ = await self._gather_context(
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
        filtered_universe, _rejected_universe = filter_stock_universe_members(dict(effective_universe))
        if filtered_universe:
            effective_universe = filtered_universe
        quotes_task = market_data_client.get_stock_universe_snapshot(effective_universe)
        histories_task = market_data_client.get_stock_universe_history(
            stock_universe=effective_universe,
            period="6mo",
            interval="1d",
            limit=180,
        )
        fundamentals_task = market_data_client.get_stock_universe_fundamentals(effective_universe)
        macro_context_task = market_data_client.get_macro_context()
        core_market_history_task = market_data_client.get_core_market_history(period="6mo", interval="1d", limit=180)
        quotes, histories, fundamentals, macro_context, core_market_history = await asyncio.gather(
            self._safe_async_call(quotes_task, default={}, source_name="stock_universe_snapshot"),
            self._safe_async_call(histories_task, default={}, source_name="stock_universe_history"),
            self._safe_async_call(fundamentals_task, default={}, source_name="stock_universe_fundamentals"),
            self._safe_async_call(macro_context_task, default={}, source_name="macro_context"),
            self._safe_async_call(core_market_history_task, default={}, source_name="core_market_history"),
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
        macro_regime = assess_macro_regime(macro_context=macro_context, asset_snapshots=[])
        market_histories: dict[str, list[object]] = {}
        for asset_name, bars in histories.items():
            if isinstance(bars, list):
                market_histories[asset_name] = list(bars)
        if isinstance(core_market_history, Mapping):
            core_alias_tickers = {
                "sp500_index": "SPY",
                "nasdaq_index": "QQQ",
                "xlk_etf": "XLK",
                "xlf_etf": "XLF",
                "xle_etf": "XLE",
                "xly_etf": "XLY",
                "xlp_etf": "XLP",
                "xlv_etf": "XLV",
                "xli_etf": "XLI",
                "xlb_etf": "XLB",
                "xlu_etf": "XLU",
                "xlc_etf": "XLC",
                "xlre_etf": "XLRE",
            }
            for asset_name, bars in core_market_history.items():
                if not isinstance(asset_name, str) or not isinstance(bars, list):
                    continue
                market_histories[asset_name] = list(bars)
                quote = quotes.get(asset_name)
                ticker = quote.ticker if quote is not None and getattr(quote, "ticker", None) else None
                if isinstance(ticker, str) and ticker.strip():
                    market_histories[ticker.strip().upper()] = list(bars)
                alias_ticker = core_alias_tickers.get(asset_name)
                if alias_ticker:
                    market_histories[alias_ticker] = list(bars)
        return rank_stock_universe(
            stock_universe=effective_universe,
            quotes=quotes,
            trends=trends,
            fundamentals=fundamentals,
            macro_context=macro_context,
            market_histories=market_histories,
            macro_regime=macro_regime.regime,
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

    async def _fetch_transcript_insights_for_candidates(
        self,
        candidates: Sequence[StockCandidate],
        *,
        limit: int = 4,
        research_client: ResearchClient | None = None,
    ) -> dict[str, TranscriptInsight]:
        direct_payload: dict[str, TranscriptInsight] = {}
        if self.transcript_client is not None and self.transcript_client.available():
            direct_payload = await self.transcript_client.get_latest_management_commentary_batch(
                [candidate.ticker for candidate in candidates],
                limit=limit,
            )
            if direct_payload:
                return direct_payload
        if research_client is None or not research_client.available():
            return direct_payload
        return await self._build_research_proxy_transcript_insights(
            candidates=candidates,
            research_client=research_client,
            limit=limit,
        )

    async def _build_research_proxy_transcript_insights(
        self,
        *,
        candidates: Sequence[StockCandidate],
        research_client: ResearchClient,
        limit: int,
    ) -> dict[str, TranscriptInsight]:
        limited_candidates = list(candidates)[: max(1, limit)]
        if not limited_candidates:
            return {}
        research_tasks = {
            candidate.ticker.upper(): research_client.search_earnings_call_context(
                ticker=candidate.ticker,
                company_name=candidate.company_name,
                limit=2,
            )
            for candidate in limited_candidates
        }
        research_results = await asyncio.gather(*research_tasks.values(), return_exceptions=True)
        payload: dict[str, TranscriptInsight] = {}
        for ticker, result in zip(research_tasks.keys(), research_results, strict=False):
            if isinstance(result, Exception) or not result:
                continue
            proxy_builder = self.transcript_client.build_research_proxy_insight if self.transcript_client is not None else EarningsTranscriptClient.build_research_proxy_insight
            proxy_insight = proxy_builder(ticker=ticker, findings=result)
            if proxy_insight is None:
                continue
            payload[ticker] = proxy_insight
        return payload

    async def _fetch_microstructure_for_candidates(
        self,
        candidates: Sequence[StockCandidate],
        *,
        limit: int = 4,
    ) -> dict[str, MicrostructureSnapshot]:
        if self.microstructure_client is None or not self.microstructure_client.available():
            return {}
        return await self.microstructure_client.get_equity_snapshot_batch(
            [candidate.ticker for candidate in candidates],
            limit=limit,
        )

    async def _fetch_ownership_intelligence_for_candidates(
        self,
        candidates: Sequence[StockCandidate],
        *,
        limit: int = 4,
    ) -> dict[str, OwnershipIntelligence]:
        if self.ownership_client is None:
            return {}
        company_names = {
            candidate.ticker.upper(): candidate.company_name
            for candidate in list(candidates)[: max(1, limit)]
        }
        try:
            payload = await self.ownership_client.get_company_ownership_batch(
                list(company_names.keys()),
                company_names=company_names,
            )
        except Exception as exc:
            logger.warning("Failed to fetch ownership intelligence: {}", exc)
            return {}
        return {
            ticker: item
            for ticker, item in payload.items()
            if isinstance(item, OwnershipIntelligence)
        }

    async def _fetch_order_flow_for_candidates(
        self,
        candidates: Sequence[StockCandidate],
        *,
        limit: int = 4,
    ) -> dict[str, OrderFlowSnapshot]:
        if self.order_flow_client is None or not self.order_flow_client.enabled_and_configured():
            return {}
        try:
            return await self.order_flow_client.get_order_flow(
                [candidate.ticker for candidate in list(candidates)[: max(1, limit)]]
            )
        except Exception as exc:
            logger.warning("Failed to fetch order-flow analytics: {}", exc)
            return {}

    async def _fetch_policy_feed_events(self, *, limit: int = 6) -> list[PolicyFeedEvent]:
        if self.policy_feed_client is None:
            return []
        try:
            return await self.policy_feed_client.fetch_recent_policy_events(limit=limit)
        except Exception as exc:
            logger.warning("Failed to fetch policy feed events: {}", exc)
            return []

    async def _build_candidate_backtest_summary(
        self,
        *,
        market_data_client: MarketDataClient,
        picks: Sequence[StockCandidate],
    ) -> dict[str, Any]:
        if self.backtesting_engine is None or not getattr(self.backtesting_engine, "enabled", False):
            return {}
        benchmark_ticker = getattr(self.backtesting_engine, "benchmark_ticker", "SPY")
        lookback_period = getattr(self.backtesting_engine, "lookback_period", "6mo")
        history_limit = getattr(self.backtesting_engine, "history_limit", 126)
        candidate_histories: dict[str, list[OhlcvBar]] = {}
        for candidate in list(picks)[:3]:
            bars = await market_data_client.get_history(
                candidate.ticker,
                period=lookback_period,
                interval="1d",
                limit=history_limit,
            )
            if bars:
                candidate_histories[candidate.ticker.upper()] = bars
        benchmark_history = await market_data_client.get_history(
            benchmark_ticker,
            period=lookback_period,
            interval="1d",
            limit=history_limit,
        )
        try:
            return self.backtesting_engine.evaluate_candidate_histories(
                candidate_histories=candidate_histories,
                benchmark_history=benchmark_history,
            )
        except Exception as exc:
            logger.warning("Failed to evaluate candidate backtest summary: {}", exc)
            return {}

    @staticmethod
    def _format_candidate_backtest_lines(payload: Mapping[str, Any] | None) -> str:
        if not isinstance(payload, Mapping):
            return "- ไม่มี backtest snapshot"
        benchmark = str(payload.get("benchmark_ticker") or "").strip() or "benchmark"
        benchmark_return = payload.get("benchmark_return_pct")
        benchmark_text = f"{benchmark_return:+.1f}%" if isinstance(benchmark_return, (float, int)) else "-"
        lines = [f"- benchmark {benchmark}: return {benchmark_text}"]
        for item in list(payload.get("candidates") or [])[:3]:
            if not isinstance(item, Mapping):
                continue
            total_return = item.get("total_return_pct")
            prefix = (
                f"- {item.get('ticker')}: return {total_return:+.1f}%"
                if isinstance(total_return, (float, int))
                else f"- {item.get('ticker')}: return -"
            )
            suffix: list[str] = []
            alpha_pct = item.get("alpha_pct")
            max_drawdown = item.get("max_drawdown_pct")
            sharpe_like = item.get("sharpe_like")
            if isinstance(alpha_pct, (float, int)):
                suffix.append(f"alpha {alpha_pct:+.1f}%")
            if isinstance(max_drawdown, (float, int)):
                suffix.append(f"maxDD {max_drawdown:+.1f}%")
            if isinstance(sharpe_like, (float, int)):
                suffix.append(f"sharpe_like {sharpe_like:+.2f}")
            lines.append(prefix if not suffix else prefix + " | " + " | ".join(suffix))
        return "\n".join(lines) if lines else "- ไม่มี backtest snapshot"

    def _build_stock_screener_prompt(
        self,
        *,
        question: str,
        profile: InvestorProfile,
        picks: Sequence[StockCandidate],
        stock_news: Mapping[str, Sequence[NewsArticle]],
        research_findings: Sequence[ResearchFinding],
        transcript_insights: Mapping[str, TranscriptInsight] | None = None,
        microstructure_snapshots: Mapping[str, MicrostructureSnapshot] | None = None,
        ownership_intelligence: Mapping[str, OwnershipIntelligence] | None = None,
        order_flow_snapshots: Mapping[str, OrderFlowSnapshot] | None = None,
        backtest_summary: Mapping[str, Any] | None = None,
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
        management_lines = self._format_management_commentary_lines(transcript_insights)
        microstructure_lines = self._format_microstructure_lines(microstructure_snapshots)
        ownership_lines = self._format_ownership_intelligence_lines(ownership_intelligence)
        order_flow_lines = self._format_order_flow_lines(order_flow_snapshots)
        backtest_lines = self._format_candidate_backtest_lines(backtest_summary)
        return (
            f"คำถามผู้ใช้: {question}\n"
            f"โปรไฟล์ผู้ลงทุน: {profile.title_th} | เป้าหมาย: {profile.objective}\n\n"
            "ผล stock screener ล่าสุด:\n"
            + ("\n".join(pick_lines) or "- ไม่มีหุ้นที่ผ่านเงื่อนไข")
            + "\n\nข่าวเฉพาะหุ้น:\n"
            + ("\n".join(news_lines) or "- ไม่มีข่าวเด่นเฉพาะหุ้น")
            + "\n\nManagement commentary ล่าสุด:\n"
            + management_lines
            + "\n\nMicrostructure ล่าสุด:\n"
            + microstructure_lines
            + "\n\nOwnership / 13F / 13D-G:\n"
            + ownership_lines
            + "\n\nOptions order-flow ล่าสุด:\n"
            + order_flow_lines
            + "\n\nBacktest snapshot:\n"
            + backtest_lines
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
        transcript_insights: Mapping[str, TranscriptInsight] | None = None,
        microstructure_snapshots: Mapping[str, MicrostructureSnapshot] | None = None,
        ownership_intelligence: Mapping[str, OwnershipIntelligence] | None = None,
        order_flow_snapshots: Mapping[str, OrderFlowSnapshot] | None = None,
        backtest_summary: Mapping[str, Any] | None = None,
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
            transcript_item = (transcript_insights or {}).get(candidate.ticker.upper())
            transcript_snippet = ""
            if transcript_item is not None:
                transcript_tone = transcript_item.get("tone") if isinstance(transcript_item, Mapping) else transcript_item.tone
                transcript_guidance = transcript_item.get("guidance_signal") if isinstance(transcript_item, Mapping) else transcript_item.guidance_signal
                transcript_snippet = (
                    f" | ผู้บริหาร: tone {transcript_tone}, guidance {transcript_guidance}"
                )
            micro_item = (microstructure_snapshots or {}).get(candidate.ticker.upper())
            micro_snippet = ""
            if micro_item is not None:
                micro_spread = micro_item.get("spread_bps") if isinstance(micro_item, Mapping) else micro_item.spread_bps
                micro_snippet = (
                    f" | flow: spread {micro_spread if micro_spread is not None else '-'} bps"
                )
            ownership_item = (ownership_intelligence or {}).get(candidate.ticker.upper())
            ownership_snippet = ""
            if ownership_item is not None:
                ownership_signal = (
                    ownership_item.get("ownership_signal")
                    if isinstance(ownership_item, Mapping)
                    else ownership_item.ownership_signal
                )
                ownership_snippet = f" | ownership: {ownership_signal or '-'}"
            order_flow_item = (order_flow_snapshots or {}).get(candidate.ticker.upper())
            order_flow_snippet = ""
            if order_flow_item is not None:
                flow_sentiment = (
                    order_flow_item.get("sentiment")
                    if isinstance(order_flow_item, Mapping)
                    else order_flow_item.sentiment
                )
                order_flow_snippet = f" | options flow: {flow_sentiment or '-'}"
            lines.append(
                f"{index}. {candidate.company_name} ({candidate.ticker}) | คะแนน {candidate.composite_score} | confidence {confidence.label} ({confidence.score}) | มุมมอง {self._humanize_stock_stance(candidate.stance)} | เหตุผล: {'; '.join(candidate.rationale[:3])}{news_snippet}{transcript_snippet}{micro_snippet}{ownership_snippet}{order_flow_snippet}"
            )
        backtest_lines = self._format_candidate_backtest_lines(backtest_summary)
        if backtest_lines and backtest_lines != "- ไม่มี backtest snapshot":
            lines.append("Backtest snapshot")
            lines.extend(backtest_lines.splitlines())
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
        macro_intelligence: Mapping[str, Any] | None,
        macro_event_calendar: Sequence[MacroEvent] | None,
        macro_surprise_signals: Sequence[MacroSurpriseSignal] | None,
        macro_market_reactions: Sequence[MacroMarketReaction] | None,
        research_findings: Sequence[ResearchFinding] | None,
        portfolio_snapshot: Mapping[str, Any] | None,
        asset_scope: AssetScope,
        question: str | None,
        investor_profile: InvestorProfile,
        company_intelligence: Sequence[CompanyIntelligence] | None = None,
        thesis_memory: Sequence[Mapping[str, Any]] | None = None,
        etf_exposures: Sequence[ETFExposureProfile] | None = None,
        policy_events: Sequence[PolicyFeedEvent] | None = None,
        ownership_intelligence: Sequence[OwnershipIntelligence] | None = None,
        order_flow: Sequence[OrderFlowSnapshot] | None = None,
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
        macro_playbooks = self._build_macro_post_event_playbooks(
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
        )
        source_learning_map = self._load_source_learning_map()
        learning_multiplier = self._determine_allocation_learning_multiplier(
            payload_sources=self._collect_payload_source_names(
                macro_intelligence=macro_intelligence,
                macro_event_calendar=macro_event_calendar,
                macro_surprise_signals=macro_surprise_signals,
                macro_market_reactions=macro_market_reactions,
            ),
            source_learning_map=source_learning_map,
            playbooks=macro_playbooks,
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
            macro_context=macro_context,
            learning_multiplier=learning_multiplier,
        )
        portfolio_review = self._build_portfolio_review_data(
            allocation_plan=allocation_plan,
            portfolio_snapshot=portfolio_snapshot,
        )
        factor_exposures = self._build_factor_exposure_summary(portfolio_snapshot=portfolio_snapshot)
        portfolio_constraints = self._build_portfolio_constraint_summary(
            portfolio_snapshot=portfolio_snapshot,
            factor_exposures=factor_exposures,
        )
        source_health = self._build_source_health_summary(
            macro_intelligence=macro_intelligence,
            macro_event_calendar=macro_event_calendar,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            news_items=news,
            research_items=research_findings,
            company_intelligence=company_intelligence,
        )
        thesis_invalidation = self._build_thesis_invalidation_summary(
            thesis_memory=thesis_memory,
            macro_regime=macro_regime,
            macro_surprise_signals=macro_surprise_signals,
            macro_market_reactions=macro_market_reactions,
            company_intelligence=company_intelligence,
        )
        regime_specific_playbooks = self._build_regime_specific_playbooks(
            macro_regime=macro_regime,
            macro_context=macro_context,
            portfolio_constraints=portfolio_constraints,
        )
        base_market_confidence = assess_market_recommendation_confidence(
            asset_snapshots=asset_snapshot_list,
            macro_regime=self._serialize_macro_regime(macro_regime),
            news_items=[self._serialize_news_article(item) for item in list(news)[:news_limit]],
            research_items=[self._serialize_research_finding(item) for item in list(research_findings or [])[:3]],
            portfolio_review=portfolio_review,
        )
        market_confidence = self._blend_confidence_with_source_health(
            assessment=base_market_confidence,
            source_health=source_health,
            portfolio_constraints=portfolio_constraints,
        )
        thesis_lifecycle = self._build_thesis_lifecycle_summary(
            thesis_memory=thesis_memory,
            source_health=source_health,
            market_confidence=market_confidence,
            thesis_invalidation=thesis_invalidation,
        )
        no_trade_decision = self._build_no_trade_decision(
            market_confidence=market_confidence,
            source_health=source_health,
            portfolio_constraints=portfolio_constraints,
            thesis_invalidation=thesis_invalidation,
            asset_scope=asset_scope,
            question=question,
        )
        champion_challenger = self._build_champion_challenger_view(
            market_confidence=market_confidence,
            source_health=source_health,
            portfolio_constraints=portfolio_constraints,
            no_trade_decision=no_trade_decision,
            source_learning_map=source_learning_map,
            regime_playbooks=regime_specific_playbooks,
            factor_exposures=factor_exposures,
            thesis_invalidation=thesis_invalidation,
        )
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": asset_scope,
            "question": question,
            "investor_profile": self._serialize_investor_profile(investor_profile),
            "macro_context": {
                str(key): self._round_optional(value)
                for key, value in (macro_context or {}).items()
                if isinstance(key, str)
            },
            "macro_intelligence": self._serialize_macro_intelligence(macro_intelligence),
            "macro_event_calendar": [self._serialize_macro_event(item) for item in list(macro_event_calendar or [])[:6]],
            "macro_surprise_signals": [self._serialize_macro_surprise(item) for item in list(macro_surprise_signals or [])[:6]],
            "macro_market_reactions": [self._serialize_macro_market_reaction(item) for item in list(macro_market_reactions or [])[:4]],
            "macro_post_event_playbooks": [self._serialize_macro_playbook(item) for item in macro_playbooks[:4]],
            "regime_specific_playbooks": [self._serialize_regime_playbook(item) for item in regime_specific_playbooks[:4]],
            "news_headlines": [self._serialize_news_article(item) for item in list(news)[:news_limit]],
            "research_highlights": [self._serialize_research_finding(item) for item in list(research_findings or [])[:3]],
            "asset_snapshots": asset_snapshot_list,
            "macro_regime": self._serialize_macro_regime(macro_regime),
            "market_confidence": self._serialize_confidence_assessment(market_confidence),
            "source_health": source_health,
            "portfolio_plan": self._serialize_portfolio_plan(portfolio_plan),
            "allocation_plan": self._serialize_allocation_plan(allocation_plan),
            "portfolio_snapshot": dict(portfolio_snapshot or {}),
            "portfolio_review": portfolio_review,
            "factor_exposures": factor_exposures,
            "portfolio_constraints": portfolio_constraints,
            "no_trade_decision": no_trade_decision,
            "champion_challenger": champion_challenger,
            "thesis_invalidation": thesis_invalidation,
            "thesis_lifecycle": thesis_lifecycle,
            "company_intelligence": [self._serialize_company_intelligence(item) for item in list(company_intelligence or [])[:6]],
            "etf_exposures": [self._serialize_etf_exposure_profile(item) for item in list(etf_exposures or [])[:6]],
            "policy_events": [self._serialize_policy_feed_event(item) for item in list(policy_events or [])[:6]],
            "ownership_intelligence": [
                self._serialize_ownership_intelligence(item)
                for item in list(ownership_intelligence or [])[:6]
            ],
            "order_flow": [self._serialize_order_flow_snapshot(item) for item in list(order_flow or [])[:6]],
            "thesis_memory": [self._serialize_thesis_memory_item(item) for item in list(thesis_memory or [])[: self.thesis_memory_top_k]],
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
            "Macro intelligence:\n"
            f"{self._format_macro_intelligence_lines(payload.get('macro_intelligence'))}\n\n"
            "Source health:\n"
            f"{self._format_source_health_lines(payload.get('source_health'))}\n\n"
            "Data quality:\n"
            f"{self._format_data_quality_lines(payload.get('data_quality'))}\n\n"
            "Upcoming macro events:\n"
            f"{self._format_macro_event_calendar_lines(payload.get('macro_event_calendar'))}\n\n"
            "Macro surprise engine:\n"
            f"{self._format_macro_surprise_lines(payload.get('macro_surprise_signals'))}\n\n"
            "Macro market reaction:\n"
            f"{self._format_macro_market_reaction_lines(payload.get('macro_market_reactions'))}\n\n"
            "Post-event playbooks:\n"
            f"{self._format_macro_playbook_lines(payload.get('macro_post_event_playbooks'))}\n\n"
            "Regime playbooks:\n"
            f"{self._format_regime_playbook_lines(payload.get('regime_specific_playbooks'))}\n\n"
            "Thesis memory:\n"
            f"{self._format_thesis_memory_lines(payload.get('thesis_memory'))}\n\n"
            "Macro regime:\n"
            f"{self._format_macro_regime_lines(payload.get('macro_regime'))}\n\n"
            "Recommendation confidence:\n"
            f"{self._format_confidence_lines(payload.get('market_confidence'))}\n\n"
            "ข่าวสำคัญล่าสุด:\n"
            f"{self._format_news_lines(payload.get('news_headlines'))}\n\n"
            "ข้อมูลวิจัยเว็บล่าสุด:\n"
            f"{self._format_research_lines(payload.get('research_highlights'))}\n\n"
            "Fed / ECB policy feed:\n"
            f"{self._format_policy_feed_lines(payload.get('policy_events'))}\n\n"
            "Company / filing intelligence:\n"
            f"{self._format_company_intelligence_lines(payload.get('company_intelligence'))}\n\n"
            "Ownership / 13F / 13D-G:\n"
            f"{self._format_ownership_intelligence_lines(payload.get('ownership_intelligence'))}\n\n"
            "Options order-flow:\n"
            f"{self._format_order_flow_lines(payload.get('order_flow'))}\n\n"
            "ETF holdings / exposure:\n"
            f"{self._format_etf_exposure_lines(payload.get('etf_exposures'))}\n\n"
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
            "Factor exposures:\n"
            f"{self._format_factor_exposure_lines(payload.get('factor_exposures'))}\n\n"
            "Portfolio constraints:\n"
            f"{self._format_portfolio_constraints_lines(payload.get('portfolio_constraints'))}\n\n"
            "No-trade framework:\n"
            f"{self._format_no_trade_lines(payload.get('no_trade_decision'))}\n\n"
            "Champion / challenger:\n"
            f"{self._format_champion_challenger_lines(payload.get('champion_challenger'))}\n\n"
            "Thesis invalidation:\n"
            f"{self._format_thesis_invalidation_lines(payload.get('thesis_invalidation'))}\n\n"
            "Thesis lifecycle:\n"
            f"{self._format_thesis_lifecycle_lines(payload.get('thesis_lifecycle'))}\n\n"
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
        macro_event_line = self._format_macro_event_calendar_lines(payload.get("macro_event_calendar"), limit=2)
        macro_surprise_line = self._format_macro_surprise_lines(payload.get("macro_surprise_signals"), limit=2)
        macro_reaction_line = self._format_macro_market_reaction_lines(payload.get("macro_market_reactions"), limit=2)
        macro_playbook_line = self._format_macro_playbook_lines(payload.get("macro_post_event_playbooks"), limit=2)
        regime_playbook_line = self._format_regime_playbook_lines(payload.get("regime_specific_playbooks"), limit=2)
        thesis_memory_line = self._format_thesis_memory_lines(payload.get("thesis_memory"), limit=2)
        confidence_line = self._format_confidence_one_line(payload.get("market_confidence"))
        source_health_line = self._format_source_health_lines(payload.get("source_health"))
        data_quality_line = self._format_data_quality_lines(payload.get("data_quality"))
        etf_exposure_line = self._format_etf_exposure_lines(payload.get("etf_exposures"), limit=2)
        allocations = self._format_allocation_summary(portfolio_plan)
        portfolio_snapshot_lines = self._format_portfolio_snapshot_lines(payload.get("portfolio_snapshot"))
        portfolio_review_lines = self._format_portfolio_review_lines(payload.get("portfolio_review"))
        portfolio_review_brief = portfolio_review_lines.splitlines()[0].lstrip("- ").strip() if portfolio_review_lines else ""
        factor_exposure_line = self._format_factor_exposure_lines(payload.get("factor_exposures"), limit=3)
        no_trade_line = self._format_no_trade_lines(payload.get("no_trade_decision"))
        thesis_invalidation_line = self._format_thesis_invalidation_lines(payload.get("thesis_invalidation"), limit=2)
        thesis_lifecycle_line = self._format_thesis_lifecycle_lines(payload.get("thesis_lifecycle"))
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
                f"{market_view}\n{profile_line}\n{confidence_line}\nsource health: {source_health_line.lstrip('- ').strip()}\ndata quality: {data_quality_line.lstrip('- ').strip()}\nอีเวนต์มหภาค: {macro_event_line.lstrip('- ').strip()}\nmacro surprise: {macro_surprise_line.lstrip('- ').strip()}\nmarket reaction: {macro_reaction_line.lstrip('- ').strip()}\nplaybook: {macro_playbook_line.lstrip('- ').strip()}\nregime playbook: {regime_playbook_line.lstrip('- ').strip()}\netf exposure: {etf_exposure_line.lstrip('- ').strip()}\nfactor exposure: {factor_exposure_line.lstrip('- ').strip()}\nพอร์ตแนะนำ: {allocations}\nno-trade: {no_trade_line.lstrip('- ').strip()}\nthesis invalidation: {thesis_invalidation_line.lstrip('- ').strip()}\nthesis lifecycle: {thesis_lifecycle_line.lstrip('- ').strip()}\nแนวทางวันนี้: {action_line}\n"
                f"thesis memory: {thesis_memory_line.lstrip('- ').strip()}\n"
                f"รีบาลานซ์พอร์ตจริง: {portfolio_review_brief or 'ยังไม่มีสัญญาณรีบาลานซ์จากพอร์ตจริง'}\n"
                f"ตัวที่น่าจับตา: {' | '.join(asset_lines[:2])}\nความเสี่ยงหลัก: {risk_line}\n"
                f"{extra_line}"
                "หมายเหตุ: ใช้เพื่อวางแผนพอร์ต ไม่ใช่การรับประกันผลตอบแทน"
            )

        sections = [
            "สรุปจากระบบสำรอง",
            f"มุมมองตลาด\n{market_view}\n{macro_line}\nSource health\n{source_health_line}\nData quality\n{data_quality_line}\nMacro events\n{macro_event_line}\nMacro surprise\n{macro_surprise_line}\nMarket reaction\n{macro_reaction_line}\nPlaybooks\n{macro_playbook_line}\nRegime playbooks\n{regime_playbook_line}\nETF exposure\n{etf_exposure_line}\nThesis memory\n{thesis_memory_line}\n{confidence_line}",
            f"โปรไฟล์ผู้ลงทุน\n{profile_line}",
            f"แผนจัดพอร์ต\n{allocations}",
            f"Current Portfolio\n{portfolio_snapshot_lines}",
            f"Rebalance Review\n{portfolio_review_lines}",
            f"Factor Exposures\n{factor_exposure_line}",
            f"No-trade framework\n{no_trade_line}",
            f"Thesis invalidation\n{thesis_invalidation_line}",
            f"Thesis lifecycle\n{thesis_lifecycle_line}",
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

    def _load_thesis_memory(self, *, question: str | None, conversation_key: str | None) -> list[dict[str, Any]]:
        query = (question or "").strip()
        if not query:
            return []
        if self.thesis_vector_store is not None:
            try:
                rows = self.thesis_vector_store.search(
                    query_text=query,
                    conversation_key=conversation_key,
                    limit=self.thesis_memory_top_k,
                )
            except Exception as exc:
                logger.warning("Failed to load thesis vector memory: {}", exc)
            else:
                if rows:
                    return [self._serialize_thesis_memory_item(item) for item in rows if isinstance(item, Mapping)]
        if self.runtime_history_store is None:
            return []
        try:
            rows = self.runtime_history_store.search_thesis_memory(
                query_text=query,
                conversation_key=conversation_key,
                limit=self.thesis_memory_top_k,
            )
        except Exception as exc:
            logger.warning("Failed to load thesis memory: {}", exc)
            return []
        return [self._serialize_thesis_memory_item(item) for item in rows if isinstance(item, Mapping)]

    def _record_learning_artifacts(
        self,
        *,
        payload: Mapping[str, Any],
        question: str | None,
        conversation_key: str | None,
        recommendation_text: str,
        model: str | None,
        response_id: str | None,
        fallback_used: bool,
        service_name: str,
    ) -> None:
        source_coverage = self.summarize_source_coverage(payload)
        data_quality = dict(payload.get("data_quality") or {}) if isinstance(payload.get("data_quality"), Mapping) else {}
        learning_snapshot = self._load_learning_snapshot()
        execution_panel = learning_snapshot.get("execution_panel") if isinstance(learning_snapshot, Mapping) else None
        source_ranking = learning_snapshot.get("source_ranking") if isinstance(learning_snapshot, Mapping) else None
        walk_forward_eval = learning_snapshot.get("walk_forward_eval") if isinstance(learning_snapshot, Mapping) else None
        artifact_key = self._build_eval_artifact_key(
            conversation_key=conversation_key,
            question=question,
            recommendation_text=recommendation_text,
            service_name=service_name,
        )
        conversation_key_hash = hashlib.sha1(str(conversation_key or "").encode("utf-8")).hexdigest()[:16] if conversation_key else None
        thesis_record = self._build_thesis_memory_record(
            payload=payload,
            question=question,
            recommendation_text=recommendation_text,
            conversation_key=conversation_key,
        )
        if thesis_record is not None and self.thesis_vector_store is not None:
            try:
                self.thesis_vector_store.record_thesis(**thesis_record)
            except Exception as exc:
                logger.warning("Failed to record thesis vector memory: {}", exc)
        if self.runtime_history_store is not None:
            try:
                if thesis_record is not None:
                    self.runtime_history_store.record_thesis_memory(**thesis_record)
                self.runtime_history_store.record_evaluation_artifact(
                    artifact_key=artifact_key,
                    artifact_kind=service_name,
                    conversation_key=conversation_key,
                    model=model,
                    fallback_used=fallback_used,
                    metrics={
                        "source_count": len(source_coverage.get("used_sources") or []),
                        "fallback_used": fallback_used,
                        "source_health_score": self._as_float((payload.get("source_health") or {}).get("score") if isinstance(payload.get("source_health"), Mapping) else None),
                        "champion_delta_vs_baseline": self._as_float((payload.get("champion_challenger") or {}).get("delta_vs_baseline") if isinstance(payload.get("champion_challenger"), Mapping) else None),
                        "factor_exposure_concentration_pct": self._as_float((payload.get("factor_exposures") or {}).get("top_exposure_weight_pct") if isinstance(payload.get("factor_exposures"), Mapping) else None),
                        "thesis_invalidation_score": self._as_float((payload.get("thesis_invalidation") or {}).get("score") if isinstance(payload.get("thesis_invalidation"), Mapping) else None),
                        "thesis_lifecycle_stage": str((payload.get("thesis_lifecycle") or {}).get("stage") or "") if isinstance(payload.get("thesis_lifecycle"), Mapping) else None,
                        "execution_ttl_hit_rate_pct": self._as_float((execution_panel or {}).get("ttl_hit_rate_pct") if isinstance(execution_panel, Mapping) else None),
                        "execution_fast_decay_rate_pct": self._as_float((execution_panel or {}).get("fast_decay_rate_pct") if isinstance(execution_panel, Mapping) else None),
                    },
                    detail={
                        "question": question,
                        "source_coverage": source_coverage,
                        "data_quality": data_quality,
                        "source_health": dict(payload.get("source_health") or {}) if isinstance(payload.get("source_health"), Mapping) else {},
                        "factor_exposures": dict(payload.get("factor_exposures") or {}) if isinstance(payload.get("factor_exposures"), Mapping) else {},
                        "no_trade_decision": dict(payload.get("no_trade_decision") or {}) if isinstance(payload.get("no_trade_decision"), Mapping) else {},
                        "champion_challenger": dict(payload.get("champion_challenger") or {}) if isinstance(payload.get("champion_challenger"), Mapping) else {},
                        "thesis_invalidation": dict(payload.get("thesis_invalidation") or {}) if isinstance(payload.get("thesis_invalidation"), Mapping) else {},
                        "thesis_lifecycle": dict(payload.get("thesis_lifecycle") or {}) if isinstance(payload.get("thesis_lifecycle"), Mapping) else {},
                        "execution_panel": dict(execution_panel) if isinstance(execution_panel, Mapping) else {},
                        "source_ranking": list(source_ranking[:8]) if isinstance(source_ranking, list) else [],
                    },
                )
            except Exception as exc:
                logger.warning("Failed to record learning artifacts: {}", exc)
        if self.analytics_store is not None:
            try:
                self.analytics_store.record_recommendation_event(
                    artifact_key=artifact_key,
                    conversation_key_hash=conversation_key_hash,
                    question=question,
                    model=model,
                    fallback_used=fallback_used,
                    response_text=recommendation_text,
                    payload=payload,
                    source_coverage=source_coverage,
                    data_quality=data_quality,
                )
                self.analytics_store.record_runtime_snapshot({"runtime": diagnostics.snapshot()})
            except Exception as exc:
                logger.warning("Failed to record analytics store recommendation event: {}", exc)
        if self.analytics_warehouse is not None:
            try:
                self.analytics_warehouse.record_recommendation_event(
                    artifact_key=artifact_key,
                    conversation_key_hash=conversation_key_hash,
                    question=question,
                    model=model,
                    fallback_used=fallback_used,
                    response_text=recommendation_text,
                    payload=payload,
                    source_coverage=source_coverage,
                    data_quality=data_quality,
                )
                self.analytics_warehouse.record_runtime_snapshot({"runtime": diagnostics.snapshot()})
            except Exception as exc:
                logger.warning("Failed to record analytics warehouse recommendation event: {}", exc)
        if self.feature_store is not None:
            try:
                self.feature_store.record_recommendation_features(
                    artifact_key=artifact_key,
                    question=question,
                    model=model,
                    payload=payload,
                    source_coverage=source_coverage,
                    data_quality=data_quality,
                    fallback_used=fallback_used,
                    service_name=service_name,
                )
            except Exception as exc:
                logger.warning("Failed to record feature store recommendation event: {}", exc)
        if self.hot_path_cache is not None:
            try:
                cached_payload = {
                    "artifact_key": artifact_key,
                    "service_name": service_name,
                    "model": model,
                    "fallback_used": fallback_used,
                    "source_coverage": source_coverage,
                    "source_health": dict(payload.get("source_health") or {}) if isinstance(payload.get("source_health"), Mapping) else {},
                }
                self.hot_path_cache.set_json(
                    namespace="recommendation",
                    key=artifact_key,
                    payload=cached_payload,
                    ttl_seconds=900,
                )
                self.hot_path_cache.append_stream(
                    stream="recommendations",
                    payload=cached_payload,
                )
            except Exception as exc:
                logger.warning("Failed to update hot-path cache: {}", exc)
        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    topic="recommendation_event",
                    key=artifact_key,
                    payload={
                        "artifact_key": artifact_key,
                        "service_name": service_name,
                        "model": model,
                        "fallback_used": fallback_used,
                        "source_coverage": source_coverage,
                    },
                )
            except Exception as exc:
                logger.warning("Failed to publish recommendation event: {}", exc)
        if self.evidently_observer is not None:
            try:
                self.evidently_observer.log_recommendation(
                    artifact_key=artifact_key,
                    question=question,
                    response_text=recommendation_text,
                    model=model,
                    fallback_used=fallback_used,
                    payload=payload,
                    data_quality=data_quality,
                )
            except Exception as exc:
                logger.warning("Failed to record Evidently recommendation event: {}", exc)
        if self.braintrust_observer is not None:
            try:
                self.braintrust_observer.log_recommendation(
                    artifact_key=artifact_key,
                    question=question,
                    response_text=recommendation_text,
                    model=model,
                    fallback_used=fallback_used,
                    payload=payload,
                    data_quality=data_quality,
                    source_coverage=source_coverage,
                )
            except Exception as exc:
                logger.warning("Failed to record Braintrust recommendation event: {}", exc)
        if self.langfuse_observer is not None:
            try:
                self.langfuse_observer.log_recommendation(
                    artifact_key=artifact_key,
                    question=question,
                    response_text=recommendation_text,
                    model=model,
                    fallback_used=fallback_used,
                    payload=payload,
                    data_quality=data_quality,
                )
            except Exception as exc:
                logger.warning("Failed to record Langfuse recommendation event: {}", exc)
        if self.human_review_store is not None:
            confidence_score = self._as_float(
                (payload.get("market_confidence") or {}).get("score")
                if isinstance(payload.get("market_confidence"), Mapping)
                else None
            )
            try:
                if self.human_review_store.should_enqueue(
                    fallback_used=fallback_used,
                    confidence_score=confidence_score,
                ):
                    self.human_review_store.enqueue(
                        artifact_key=artifact_key,
                        question=question,
                        recommendation_text=recommendation_text,
                        model=model,
                        fallback_used=fallback_used,
                        confidence_score=confidence_score,
                        metadata={
                            "service_name": service_name,
                            "source_coverage": source_coverage,
                            "data_quality": data_quality,
                            "response_id": response_id,
                        },
                    )
            except Exception as exc:
                logger.warning("Failed to enqueue human review: {}", exc)
        mlflow_run_id: str | None = None
        if self.mlflow_observer is not None:
            mlflow_run_id = self.mlflow_observer.log_recommendation(
                service_name=service_name,
                question=question,
                conversation_key=conversation_key,
                model=model,
                fallback_used=fallback_used,
                payload=payload,
                response_text=recommendation_text,
                source_coverage=source_coverage,
                artifact_key=artifact_key,
                response_id=response_id,
                data_quality=data_quality,
                execution_panel=execution_panel if isinstance(execution_panel, Mapping) else None,
                source_ranking=[item for item in source_ranking[:8] if isinstance(item, Mapping)] if isinstance(source_ranking, list) else None,
                source_health=payload.get("source_health") if isinstance(payload.get("source_health"), Mapping) else None,
                champion_challenger=payload.get("champion_challenger") if isinstance(payload.get("champion_challenger"), Mapping) else None,
                factor_exposures=payload.get("factor_exposures") if isinstance(payload.get("factor_exposures"), Mapping) else None,
                thesis_invalidation=payload.get("thesis_invalidation") if isinstance(payload.get("thesis_invalidation"), Mapping) else None,
                walk_forward_eval=walk_forward_eval if isinstance(walk_forward_eval, Mapping) else None,
            )
        if mlflow_run_id and self.runtime_history_store is not None:
            try:
                self.runtime_history_store.record_evaluation_artifact(
                    artifact_key=artifact_key,
                    artifact_kind=service_name,
                    conversation_key=conversation_key,
                    model=model,
                    fallback_used=fallback_used,
                    detail={"mlflow_run_id": mlflow_run_id},
                )
            except Exception as exc:
                logger.warning("Failed to persist MLflow run correlation: {}", exc)

    def _load_learning_snapshot(self) -> dict[str, Any]:
        dashboard_getter = getattr(self.runtime_history_store, "build_evaluation_dashboard", None)
        if not callable(dashboard_getter):
            return {}
        try:
            snapshot = dashboard_getter(lookback_days=30, burn_in_target_days=14)
        except Exception:
            return {}
        return dict(snapshot) if isinstance(snapshot, Mapping) else {}

    def _build_thesis_memory_record(
        self,
        *,
        payload: Mapping[str, Any],
        question: str | None,
        recommendation_text: str,
        conversation_key: str | None,
    ) -> dict[str, Any] | None:
        summary = self._truncate_text(recommendation_text, 240)
        if not summary:
            return None
        thesis_text = self._truncate_text(
            " | ".join(
                part
                for part in (
                    str(question or "").strip(),
                    str((payload.get("macro_intelligence") or {}).get("headline") if isinstance(payload.get("macro_intelligence"), Mapping) else "").strip(),
                    summary,
                )
                if part
            ),
            420,
        )
        if not thesis_text:
            return None
        thesis_key = self._build_eval_artifact_key(
            conversation_key=conversation_key,
            question=question,
            recommendation_text=thesis_text,
            service_name="thesis",
        )
        return {
            "thesis_key": thesis_key,
            "conversation_key": conversation_key,
            "query_text": question,
            "thesis_text": thesis_text,
            "source_kind": "recommendation",
            "tags": self._extract_top_payload_tags(payload),
            "confidence_score": self._as_float((payload.get("market_confidence") or {}).get("score") if isinstance(payload.get("market_confidence"), Mapping) else None),
            "detail": {
                "scope": payload.get("scope"),
                "macro_headline": (payload.get("macro_intelligence") or {}).get("headline") if isinstance(payload.get("macro_intelligence"), Mapping) else None,
                "thesis_lifecycle": dict(payload.get("thesis_lifecycle") or {}) if isinstance(payload.get("thesis_lifecycle"), Mapping) else {},
            },
        }

    def _extract_top_payload_tags(self, payload: Mapping[str, Any]) -> list[str]:
        tags: list[str] = []
        macro_signals = (payload.get("macro_intelligence") or {}).get("signals") if isinstance(payload.get("macro_intelligence"), Mapping) else []
        if isinstance(macro_signals, list):
            tags.extend(str(item).strip() for item in macro_signals[:4] if str(item).strip())
        stock_picks = payload.get("stock_picks")
        if isinstance(stock_picks, list):
            tags.extend(str(item.get("ticker") or "").strip() for item in stock_picks[:3] if isinstance(item, Mapping) and str(item.get("ticker") or "").strip())
        return list(dict.fromkeys(tag for tag in tags if tag))

    @staticmethod
    def _build_eval_artifact_key(
        *,
        conversation_key: str | None,
        question: str | None,
        recommendation_text: str,
        service_name: str,
    ) -> str:
        seed = "|".join(
            [
                service_name,
                str(conversation_key or ""),
                str(question or ""),
                RecommendationService._truncate_text(recommendation_text, 160),
            ]
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]

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
        source_learning_map = self._load_source_learning_map()
        payload_sources = self._collect_payload_source_names(
            macro_intelligence=payload.get("macro_intelligence"),
            macro_event_calendar=payload.get("macro_event_calendar"),
            macro_surprise_signals=payload.get("macro_surprise_signals"),
            macro_market_reactions=payload.get("macro_market_reactions"),
        )
        payload_evidence_score = self._average_source_learning_score(source_learning_map, payload_sources)

        vix = self._as_float(macro_context.get("vix") if isinstance(macro_context, Mapping) else None)
        if (vix is not None and vix >= vix_threshold) or risk_score.score >= risk_score_threshold:
            reason_text = " | ".join(risk_score.reasons[:3]) if risk_score.reasons else "ความผันผวนเพิ่มขึ้น"
            severity = self._adjust_alert_severity(
                base_severity="warning" if risk_score.level in {"elevated", "high"} else "critical",
                evidence_score=payload_evidence_score,
            )
            alerts.append(
                InterestingAlert(
                    key=f"risk:{risk_score.level}:{int(round(risk_score.score))}",
                    severity=severity,
                    text=(
                        f"{self._format_badged_title('🟠 ระวัง', 'ความเสี่ยงตลาด')}\n"
                        f"- ภาพรวม: คะแนนความเสี่ยง {risk_score.score:.1f}/10 | ระดับ {self._humanize_risk_level(risk_score.level)}\n"
                        f"- เหตุผล: {self._extract_portfolio_value(portfolio_plan, 'risk_watch') or reason_text}\n"
                        f"- Action: {self._extract_portfolio_value(portfolio_plan, 'action_plan') or 'เพิ่มเงินสดและลดสินทรัพย์เสี่ยงบางส่วน'}"
                    ),
                    metadata=self._build_alert_metadata(
                        alert_kind="risk",
                        severity=severity,
                        evidence_score=payload_evidence_score,
                    ),
                )
            )

        macro_surprises = payload.get("macro_surprise_signals")
        if isinstance(macro_surprises, list):
            for item in macro_surprises[:3]:
                if not isinstance(item, Mapping):
                    continue
                label = str(item.get("surprise_label") or "").strip()
                if label in {"", "in_line", "steady", "insufficient_data"}:
                    continue
                event_name = str(item.get("event_name") or item.get("event_key") or "macro").strip()
                bias = str(item.get("market_bias") or "balanced").strip()
                rationale = ", ".join(str(part) for part in (item.get("rationale") or [])[:3] if part) or label
                action = (
                    "ลดการไล่ risk asset และติดตาม bond yield / USD"
                    if "risk_off" in bias or "defensive" in bias or "rates_up" in bias
                    else "ติดตามแรงหนุนต่อ duration, quality growth และ sector ที่อ่อนไหวต่ออัตราดอกเบี้ย"
                )
                source_names = [str(item.get("source") or "").strip(), "macro_surprise_engine"]
                evidence_score = self._average_source_learning_score(source_learning_map, source_names)
                severity = self._adjust_alert_severity(
                    base_severity="warning" if "risk_off" in bias or "defensive" in bias or "rates_up" in bias else "info",
                    evidence_score=evidence_score,
                )
                alerts.append(
                    InterestingAlert(
                        key=f"macro_surprise:{event_name.casefold()}:{label}",
                        severity=severity,
                        text=(
                            f"{self._format_badged_title('🟠 ระวัง' if 'risk_off' in bias or 'defensive' in bias or 'rates_up' in bias else '✅ ยืนยัน', 'macro surprise')}\n"
                            f"- ภาพรวม: {event_name} | {label}\n"
                            f"- เหตุผล: {rationale}\n"
                            f"- Action: {action}"
                        ),
                        metadata=self._build_alert_metadata(
                            alert_kind="macro_surprise",
                            severity=severity,
                            evidence_score=evidence_score,
                            preferred_sources=tuple(name for name in source_names if name),
                        ),
                    )
                )

        macro_reactions = payload.get("macro_market_reactions")
        if isinstance(macro_reactions, list):
            for item in macro_reactions[:3]:
                if not isinstance(item, Mapping):
                    continue
                confirmation_label = str(item.get("confirmation_label") or "").strip()
                if confirmation_label != "not_confirmed":
                    continue
                event_name = str(item.get("event_name") or item.get("event_key") or "macro").strip()
                bias = str(item.get("market_bias") or "balanced").strip()
                rationale = ", ".join(str(part) for part in (item.get("rationale") or [])[:3] if part) or confirmation_label
                severity = self._adjust_alert_severity(
                    base_severity="warning",
                    evidence_score=self._average_source_learning_score(source_learning_map, ("macro_market_reaction",)),
                )
                alerts.append(
                    InterestingAlert(
                        key=f"macro_reaction:{event_name.casefold()}:{confirmation_label}",
                        severity=severity,
                        text=(
                            f"{self._format_badged_title('🟠 ระวัง', 'market reaction divergence')}\n"
                            f"- ภาพรวม: {event_name} | surprise bias {bias} แต่ตลาดไม่ confirm\n"
                            f"- เหตุผล: {rationale}\n"
                            f"- Action: อย่ารีบ chase narrative จากเลขข่าวอย่างเดียว รอ price confirmation เพิ่ม"
                        ),
                        metadata=self._build_alert_metadata(
                            alert_kind="macro_reaction",
                            severity=severity,
                            evidence_score=self._average_source_learning_score(source_learning_map, ("macro_market_reaction",)),
                            preferred_sources=("macro_market_reaction",),
                        ),
                    )
                )

        playbooks = payload.get("macro_post_event_playbooks")
        if isinstance(playbooks, list):
            for item in playbooks[:3]:
                if not isinstance(item, Mapping):
                    continue
                title = str(item.get("title") or item.get("playbook_key") or "").strip()
                action = str(item.get("action") or "").strip()
                trigger = str(item.get("trigger") or "").strip()
                confidence = str(item.get("confidence") or "").strip()
                confidence_score = item.get("confidence_score")
                learning_note = str(item.get("learning_note") or "").strip()
                if not title or not action:
                    continue
                severity = self._adjust_alert_severity(
                    base_severity="warning",
                    evidence_score=self._average_source_learning_score(source_learning_map, ("macro_post_event_playbooks",)),
                    confidence_score=float(confidence_score) if isinstance(confidence_score, (float, int)) else None,
                )
                alerts.append(
                    InterestingAlert(
                        key=f"macro_playbook:{str(item.get('playbook_key') or title).strip()}",
                        severity=severity,
                        text=(
                            f"{self._format_badged_title('🔎 จับตา', 'post-event playbook')}\n"
                            f"- ภาพรวม: {title}\n"
                            f"- Trigger: {trigger or '-'}\n"
                            f"- Confidence: {confidence or '-'}"
                            + (f" ({float(confidence_score):.2f})" if isinstance(confidence_score, (float, int)) else "")
                            + (f" | {learning_note}" if learning_note else "")
                            + "\n"
                            f"- Action: {action}"
                        ),
                        metadata=self._build_alert_metadata(
                            alert_kind="macro_playbook",
                            severity=severity,
                            confidence_score=float(confidence_score) if isinstance(confidence_score, (float, int)) else None,
                            evidence_score=self._average_source_learning_score(source_learning_map, ("macro_post_event_playbooks",)),
                            preferred_sources=("macro_post_event_playbooks",),
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
                    metadata=self._build_alert_metadata(
                        alert_kind="news",
                        severity="info" if impact.sentiment == "positive" else "warning",
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
                        metadata=self._build_alert_metadata(alert_kind="ranking", severity="info"),
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
                        metadata=self._build_alert_metadata(alert_kind="ranking", severity="warning"),
                    )
                )

        unique: dict[str, InterestingAlert] = {}
        for alert in alerts:
            unique.setdefault(alert.key, alert)
        return list(unique.values())

    def _load_source_learning_map(self) -> dict[str, float]:
        dashboard_getter = getattr(self.runtime_history_store, "build_evaluation_dashboard", None)
        if not callable(dashboard_getter):
            return {}
        try:
            snapshot = dashboard_getter(lookback_days=30, burn_in_target_days=14)
        except Exception:
            return {}
        if not isinstance(snapshot, Mapping):
            return {}
        ranking = snapshot.get("source_ranking")
        if not isinstance(ranking, list):
            return {}
        scores: dict[str, float] = {}
        for item in ranking:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "").strip()
            if not source:
                continue
            try:
                weighted_score = float(item.get("weighted_score"))
            except (TypeError, ValueError):
                continue
            scores[source] = weighted_score
        return scores

    @staticmethod
    def _average_source_learning_score(source_learning_map: Mapping[str, float], source_names: Sequence[str]) -> float | None:
        matches = [
            float(source_learning_map[name])
            for name in dict.fromkeys(str(item).strip() for item in source_names if str(item).strip())
            if name in source_learning_map
        ]
        if not matches:
            return None
        return round(sum(matches) / len(matches), 2)

    @staticmethod
    def _adjust_alert_severity(
        *,
        base_severity: str,
        evidence_score: float | None = None,
        confidence_score: float | None = None,
    ) -> str:
        severity_levels = ("info", "warning", "critical")
        try:
            current_index = severity_levels.index(base_severity)
        except ValueError:
            current_index = 1
        signal_strength = max(
            (float(evidence_score) / 100.0) if evidence_score is not None else 0.0,
            float(confidence_score) if confidence_score is not None else 0.0,
        )
        if signal_strength >= 0.78 and current_index < 2:
            return severity_levels[current_index + 1]
        if signal_strength <= 0.48 and current_index > 0:
            return severity_levels[current_index - 1]
        return severity_levels[current_index]

    @staticmethod
    def _collect_payload_source_names(
        *,
        macro_intelligence: Any,
        macro_event_calendar: Any,
        macro_surprise_signals: Any,
        macro_market_reactions: Any,
    ) -> list[str]:
        names: list[str] = []
        if isinstance(macro_intelligence, Mapping):
            sources_used = macro_intelligence.get("sources_used")
            if isinstance(sources_used, list):
                names.extend(str(item).strip() for item in sources_used if str(item).strip())
        if isinstance(macro_event_calendar, list) and macro_event_calendar:
            names.append("macro_event_calendar")
            names.extend(str(item.get("source") or "").strip() for item in macro_event_calendar if isinstance(item, Mapping))
        if isinstance(macro_surprise_signals, list) and macro_surprise_signals:
            names.append("macro_surprise_engine")
            names.extend(str(item.get("source") or "").strip() for item in macro_surprise_signals if isinstance(item, Mapping))
        if isinstance(macro_market_reactions, list) and macro_market_reactions:
            names.append("macro_market_reaction")
        return [name for name in dict.fromkeys(names) if name]

    def _determine_allocation_learning_multiplier(
        self,
        *,
        payload_sources: Sequence[str],
        source_learning_map: Mapping[str, float],
        playbooks: Sequence[MacroPostEventPlaybook],
    ) -> float:
        multiplier = 1.0
        source_score = self._average_source_learning_score(source_learning_map, payload_sources)
        if source_score is not None:
            if source_score >= 72:
                multiplier += 0.15
            elif source_score <= 50:
                multiplier -= 0.15
        playbook_scores = [float(item.confidence_score) for item in playbooks if item.confidence_score > 0]
        if playbook_scores:
            average_playbook_score = sum(playbook_scores) / len(playbook_scores)
            if average_playbook_score >= 0.72:
                multiplier += 0.1
            elif average_playbook_score <= 0.55:
                multiplier -= 0.1
        return max(0.75, min(1.35, round(multiplier, 2)))

    def _compute_alert_ttl_minutes(
        self,
        *,
        alert_kind: str,
        severity: str,
        evidence_score: float | None = None,
        confidence_score: float | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
        execution_prior: Mapping[str, Any] | None = None,
    ) -> int:
        base_map = {
            "critical": 360,
            "warning": 180,
            "info": 90,
        }
        base_minutes = base_map.get(severity, 120)
        kind_bonus = {
            "stock_pick": 90,
            "macro_playbook": 45,
            "macro_surprise": 15,
            "macro_reaction": 15,
            "risk": 60,
        }.get(alert_kind, 0)
        signal_strength = max(
            (float(evidence_score) / 100.0) if evidence_score is not None else 0.0,
            float(confidence_score) if confidence_score is not None else 0.0,
        )
        if signal_strength >= 0.8:
            base_minutes += 60
        elif signal_strength <= 0.45:
            base_minutes -= 30
        if isinstance(execution_feedback, Mapping):
            try:
                ttl_hit_rate_pct = float(execution_feedback.get("ttl_hit_rate_pct"))
            except (TypeError, ValueError):
                ttl_hit_rate_pct = None
            try:
                fast_decay_rate_pct = float(execution_feedback.get("fast_decay_rate_pct"))
            except (TypeError, ValueError):
                fast_decay_rate_pct = None
            try:
                hold_rate_pct = float(execution_feedback.get("hold_rate_pct"))
            except (TypeError, ValueError):
                hold_rate_pct = None
            if fast_decay_rate_pct is not None:
                if fast_decay_rate_pct >= 35:
                    base_minutes -= 45
                elif fast_decay_rate_pct >= 20:
                    base_minutes -= 20
            if hold_rate_pct is not None and hold_rate_pct >= 65:
                base_minutes += 30
            elif ttl_hit_rate_pct is not None and ttl_hit_rate_pct <= 40:
                base_minutes -= 15
        ttl_minutes = max(30, int(base_minutes + kind_bonus))
        return self._apply_execution_ttl_prior(
            ttl_minutes=ttl_minutes,
            alert_kind=alert_kind,
            execution_prior=execution_prior,
        )

    @staticmethod
    def _compute_realert_cadence_minutes(
        *,
        ttl_minutes: int,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> int:
        cadence = float(ttl_minutes) * 0.7
        if isinstance(execution_feedback, Mapping):
            try:
                ttl_hit_rate_pct = float(execution_feedback.get("ttl_hit_rate_pct"))
            except (TypeError, ValueError):
                ttl_hit_rate_pct = None
            try:
                fast_decay_rate_pct = float(execution_feedback.get("fast_decay_rate_pct"))
            except (TypeError, ValueError):
                fast_decay_rate_pct = None
            try:
                hold_rate_pct = float(execution_feedback.get("hold_rate_pct"))
            except (TypeError, ValueError):
                hold_rate_pct = None
            if fast_decay_rate_pct is not None and fast_decay_rate_pct >= 35:
                cadence *= 0.55
            elif fast_decay_rate_pct is not None and fast_decay_rate_pct >= 20:
                cadence *= 0.75
            elif hold_rate_pct is not None and hold_rate_pct >= 65:
                cadence *= 1.15
            elif ttl_hit_rate_pct is not None and ttl_hit_rate_pct <= 40:
                cadence *= 0.75
        return max(20, min(720, int(round(cadence))))

    def _build_alert_metadata(
        self,
        *,
        alert_kind: str,
        severity: str,
        confidence_score: float | None = None,
        evidence_score: float | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
        preferred_sources: Sequence[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        effective_feedback = execution_feedback or self._load_alert_execution_feedback(alert_kind=alert_kind)
        execution_prior = self._load_execution_heatmap_prior(
            alert_kind=alert_kind,
            preferred_sources=preferred_sources,
        )
        ttl_minutes = self._compute_alert_ttl_minutes(
            alert_kind=alert_kind,
            severity=severity,
            evidence_score=evidence_score,
            confidence_score=confidence_score,
            execution_feedback=effective_feedback,
            execution_prior=execution_prior,
        )
        realert_after_minutes = self._compute_realert_cadence_minutes(
            ttl_minutes=ttl_minutes,
            execution_feedback=effective_feedback,
        )
        metadata = {
            "alert_kind": alert_kind,
            "ttl_minutes": ttl_minutes,
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat(),
            "realert_after_minutes": realert_after_minutes,
        }
        if confidence_score is not None:
            metadata["confidence_score"] = round(float(confidence_score), 2)
        if evidence_score is not None:
            metadata["evidence_score"] = round(float(evidence_score), 2)
        if extra:
            metadata.update(dict(extra))
        if isinstance(execution_prior, Mapping):
            metadata["execution_prior"] = {
                "source": execution_prior.get("source"),
                "ttl_bucket": execution_prior.get("ttl_bucket"),
                "score": execution_prior.get("score"),
                "sample_count": execution_prior.get("sample_count"),
            }
        return metadata

    @staticmethod
    def _apply_execution_ttl_prior(
        *,
        ttl_minutes: int,
        alert_kind: str,
        execution_prior: Mapping[str, Any] | None,
    ) -> int:
        if not isinstance(execution_prior, Mapping):
            return ttl_minutes
        best_bucket = str(execution_prior.get("ttl_bucket") or "").strip()
        sample_count = int(execution_prior.get("sample_count") or 0)
        try:
            prior_score = float(execution_prior.get("score"))
        except (TypeError, ValueError):
            prior_score = None
        bucket_targets = {
            "stock_pick": {"short": 120, "medium": 210, "long": 330},
            "macro_playbook": {"short": 90, "medium": 150, "long": 240},
            "macro_surprise": {"short": 45, "medium": 90, "long": 150},
            "macro_reaction": {"short": 45, "medium": 90, "long": 150},
        }
        target = bucket_targets.get(alert_kind, {"short": 90, "medium": 180, "long": 300}).get(best_bucket)
        if target is None:
            return ttl_minutes
        blend = 0.45 if sample_count >= 3 else 0.25
        if prior_score is not None and prior_score >= 75:
            blend += 0.1
        blended = round((ttl_minutes * (1.0 - blend)) + (target * blend))
        return max(30, min(720, int(blended)))

    def _build_stock_pick_flow_health(
        self,
        *,
        picks: Sequence[StockCandidate],
        stock_news: Mapping[str, Sequence[NewsArticle]],
        research_findings: Sequence[ResearchFinding],
        portfolio_constraints: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        top_picks = list(picks[:3])
        avg_universe_quality = (
            round(sum(item.universe_quality_score for item in top_picks) / len(top_picks), 2)
            if top_picks
            else None
        )
        news_coverage_count = sum(1 for item in top_picks if stock_news.get(item.asset))
        score = 45.0
        rationale: list[str] = []
        if avg_universe_quality is not None:
            score += (avg_universe_quality - 0.5) * 28.0
            if avg_universe_quality >= 0.72:
                rationale.append("top ideas still trade in liquid large-cap names")
            elif avg_universe_quality <= 0.46:
                rationale.append("candidate execution quality is mixed")
        if news_coverage_count >= max(1, len(top_picks) - 1):
            score += 12.0
            rationale.append("most top names have live news context")
        elif news_coverage_count == 0:
            score -= 10.0
            rationale.append("top names are missing fresh news context")
        if len(research_findings) >= 2:
            score += 8.0
            rationale.append("research cross-check is available")
        if isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True)):
            score -= 14.0
            rationale.append("portfolio risk budget is already tight")
        clipped = round(max(5.0, min(100.0, score)), 1)
        label = "strong" if clipped >= 72 else "mixed" if clipped >= 52 else "fragile"
        return {
            "score": clipped,
            "label": label,
            "avg_universe_quality_score": avg_universe_quality,
            "news_coverage_count": news_coverage_count,
            "research_count": len(research_findings),
            "rationale": rationale[:4],
        }

    def _build_stock_pick_no_trade_decision(
        self,
        *,
        picks: Sequence[StockCandidate],
        portfolio_constraints: Mapping[str, Any] | None,
        source_health: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True)):
            reasons.append("portfolio concentration already exceeds the current risk budget")
        top_picks = list(picks[:3])
        if top_picks:
            avg_quality = sum(item.universe_quality_score for item in top_picks) / len(top_picks)
            if avg_quality <= 0.42:
                reasons.append("top candidates are not liquid or robust enough for clean execution")
            coverage_scores = [self._assess_stock_candidate_coverage(item)["score"] for item in top_picks]
            avg_coverage = sum(coverage_scores) / len(coverage_scores)
            if avg_coverage <= 0.56:
                reasons.append("top candidates still have thin per-asset coverage")
        source_health_score = self._as_float((source_health or {}).get("score"))
        if source_health_score is not None and source_health_score < 48:
            reasons.append("supporting data coverage is still too thin")
        should_abstain = bool(reasons)
        return {
            "should_abstain": should_abstain,
            "summary": "hold fire on new stock-pick risk" if should_abstain else "risk budget allows selective entries",
            "reasons": reasons[:4],
            "action": (
                "ลดการหาไอเดียใหม่ เหลือแค่ watchlist / quality names จนกว่า portfolio risk และ data coverage จะดีขึ้น"
                if should_abstain
                else "เปิดได้เฉพาะไม้เล็กและเลือกชื่อที่ liquidity / quality สูงก่อน"
            ),
        }

    def _estimate_execution_realism(
        self,
        *,
        candidate: StockCandidate | None,
        desired_position_size_pct: float,
        portfolio_constraints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        liquidity_tier = str(candidate.liquidity_tier if candidate is not None else "high").strip().casefold()
        spread_bps = {"very_high": 3.0, "high": 5.0, "medium": 10.0, "low": 18.0}.get(liquidity_tier, 7.0)
        quality_score = float(candidate.universe_quality_score) if candidate is not None else 0.65
        size_pressure_bps = max(1.5, float(desired_position_size_pct) * 2.2)
        slippage_bps = round(size_pressure_bps + max(0.0, (0.65 - quality_score) * 18.0), 1)
        execution_cost_bps = round(spread_bps + slippage_bps, 1)
        base_cap = {"very_high": 7.0, "high": 5.5, "medium": 3.5, "low": 2.0}.get(liquidity_tier, 4.0)
        if isinstance(portfolio_constraints, Mapping):
            base_cap = min(base_cap, float(portfolio_constraints.get("position_size_cap_pct") or base_cap))
        cost_haircut = 1.0
        if execution_cost_bps >= 32:
            cost_haircut -= 0.35
        elif execution_cost_bps >= 22:
            cost_haircut -= 0.18
        adjusted_position_size_pct = max(0.4, min(base_cap, round(float(desired_position_size_pct) * cost_haircut, 1)))
        label = "efficient" if execution_cost_bps <= 12 else "acceptable" if execution_cost_bps <= 24 else "fragile"
        return {
            "spread_bps": round(spread_bps, 1),
            "slippage_bps": slippage_bps,
            "execution_cost_bps": execution_cost_bps,
            "position_size_cap_pct": round(base_cap, 1),
            "adjusted_position_size_pct": adjusted_position_size_pct,
            "execution_label": label,
        }

    def _build_macro_post_event_playbooks(
        self,
        *,
        macro_surprise_signals: Sequence[MacroSurpriseSignal] | None,
        macro_market_reactions: Sequence[MacroMarketReaction] | None,
    ) -> list[MacroPostEventPlaybook]:
        surprise_map = {
            item.event_key: item
            for item in list(macro_surprise_signals or [])
        }
        reaction_map = {
            item.event_key: item
            for item in list(macro_market_reactions or [])
        }
        playbooks: list[MacroPostEventPlaybook] = []

        cpi_signal = surprise_map.get("cpi")
        cpi_reaction = reaction_map.get("cpi")
        if cpi_signal is not None and cpi_reaction is not None:
            hot_cpi = self._signal_matches_labels(cpi_signal, {"hotter_than_baseline"})
            no_confirm = cpi_reaction.confirmation_label == "not_confirmed"
            if hot_cpi and no_confirm:
                confidence_label, confidence_score, learning_note = self._score_playbook_confidence(
                    base_confidence="medium",
                    keywords=("inflation", "cpi", "sticky inflation", "duration-sensitive growth", "rates"),
                )
                playbooks.append(
                    MacroPostEventPlaybook(
                        playbook_key="hot_cpi_no_market_confirm",
                        title="Hot CPI + No Market Confirm",
                        trigger="inflation hotter than expected/baseline แต่ cross-asset reaction ไม่ยืนยัน",
                        action="อย่ารีบไล่ risk-off เพิ่มทันที รอ US10Y, DXY, VIX หรือ broad equity weakness confirm ก่อนค่อยลด beta",
                        risk_watch="US10Y, DXY, VIX, SPY breadth",
                        confidence=confidence_label,
                        confidence_score=confidence_score,
                        learning_note=learning_note,
                        event_key="cpi",
                    )
                )

        nfp_signal = surprise_map.get("nfp")
        nfp_reaction = reaction_map.get("nfp")
        if nfp_signal is not None and nfp_reaction is not None:
            weak_nfp = self._signal_matches_labels(nfp_signal, {"weaker_than_baseline"})
            tlt_reaction = self._find_macro_reaction_asset(nfp_reaction, "TLT")
            tlt_not_bid = tlt_reaction is not None and (
                tlt_reaction.confirmed_1h is False
                or ((tlt_reaction.move_1h_pct or 0.0) <= 0.05)
            )
            if weak_nfp and tlt_not_bid:
                confidence_label, confidence_score, learning_note = self._score_playbook_confidence(
                    base_confidence="high",
                    keywords=("labor", "nfp", "payroll", "duration", "bond", "growth slowing"),
                )
                playbooks.append(
                    MacroPostEventPlaybook(
                        playbook_key="weak_nfp_tlt_not_bid",
                        title="Weak NFP + TLT Not Bid",
                        trigger="labor data อ่อน แต่ bond duration ไม่รับข่าวดี",
                        action="อย่ารีบ assume ว่าตลาดจะ price-in rate cuts เร็วขึ้น ถือ barbell/quality และรอ TLT confirm ก่อน overweight duration",
                        risk_watch="TLT, US10Y, QQQ relative strength",
                        confidence=confidence_label,
                        confidence_score=confidence_score,
                        learning_note=learning_note,
                        event_key="nfp",
                    )
                )

        fomc_signal = surprise_map.get("fomc")
        fomc_reaction = reaction_map.get("fomc")
        if fomc_signal is not None and fomc_reaction is not None:
            hawkish = self._signal_matches_labels(fomc_signal, {"hawkish_shift"})
            qqq_reaction = self._find_macro_reaction_asset(fomc_reaction, "QQQ")
            qqq_resilient = qqq_reaction is not None and ((qqq_reaction.move_1h_pct or 0.0) >= 0.1)
            if hawkish and qqq_resilient:
                confidence_label, confidence_score, learning_note = self._score_playbook_confidence(
                    base_confidence="medium",
                    keywords=("fed", "fomc", "hawkish", "tech", "qqq", "growth leadership", "breadth"),
                )
                playbooks.append(
                    MacroPostEventPlaybook(
                        playbook_key="hawkish_fomc_qqq_resilient",
                        title="Hawkish FOMC + QQQ Still Resilient",
                        trigger="Fed hawkish แต่ growth leadership ยังไม่ยอมลง",
                        action="หลีกเลี่ยงการ short broad tech แบบรีบเร่ง เน้น relative strength, ลดเฉพาะ weak beta และรอ breadth เสียก่อนค่อย hedge เพิ่ม",
                        risk_watch="QQQ vs SPY, XLK breadth, US10Y follow-through",
                        confidence=confidence_label,
                        confidence_score=confidence_score,
                        learning_note=learning_note,
                        event_key="fomc",
                    )
                )

        return playbooks

    def _score_playbook_confidence(
        self,
        *,
        base_confidence: str,
        keywords: Sequence[str],
    ) -> tuple[str, float, str]:
        base_score_map = {"low": 0.45, "medium": 0.6, "high": 0.74}
        base_score = base_score_map.get(base_confidence, 0.6)
        rows_getter = getattr(self.runtime_history_store, "recent_stock_pick_scorecard", None)
        if not callable(rows_getter):
            return base_confidence, round(base_score, 2), "no local scorecard history yet"
        try:
            rows = rows_getter(limit=160)
        except Exception:
            return base_confidence, round(base_score, 2), "local scorecard history unavailable"
        matched_rows = [row for row in rows if self._row_matches_playbook_keywords(row, keywords=keywords)]
        closed_rows = [row for row in matched_rows if str(row.get("status") or "").strip().casefold() == "closed"]
        if not closed_rows:
            return base_confidence, round(base_score, 2), "no closed local analogs yet"
        win_count = 0
        return_values: list[float] = []
        for row in closed_rows:
            try:
                return_value = float(row.get("return_pct")) if row.get("return_pct") is not None else None
            except (TypeError, ValueError):
                return_value = None
            if return_value is None:
                continue
            return_values.append(return_value)
            if return_value > 0:
                win_count += 1
        closed_count = len(closed_rows)
        hit_rate_pct = round((win_count / closed_count) * 100.0, 1) if closed_count else None
        avg_return_pct = round((sum(return_values) / len(return_values)) * 100.0, 2) if return_values else None
        learned_score = self._compute_learning_confidence_score(
            closed_count=closed_count,
            hit_rate_pct=hit_rate_pct,
            avg_return_pct=avg_return_pct,
        )
        final_score = round((base_score * 0.6) + (learned_score * 0.4), 2)
        final_label = "high" if final_score >= 0.72 else "medium" if final_score >= 0.58 else "low"
        return (
            final_label,
            final_score,
            f"learned from {closed_count} closed analogs | hit {hit_rate_pct or 0:.1f}% | avg {avg_return_pct or 0:.2f}%",
        )

    def _row_matches_playbook_keywords(self, row: Mapping[str, Any], *, keywords: Sequence[str]) -> bool:
        detail = row.get("detail") if isinstance(row.get("detail"), Mapping) else {}
        haystack_parts = [
            str(row.get("source_kind") or ""),
            str(detail.get("thesis_summary") or ""),
            str(detail.get("macro_headline") or ""),
        ]
        macro_drivers = detail.get("macro_drivers")
        if isinstance(macro_drivers, list):
            haystack_parts.extend(str(item) for item in macro_drivers[:6])
        thesis_memory = detail.get("thesis_memory")
        if isinstance(thesis_memory, list):
            for item in thesis_memory[:3]:
                if not isinstance(item, Mapping):
                    continue
                haystack_parts.append(str(item.get("thesis_text") or ""))
                tags = item.get("tags")
                if isinstance(tags, list):
                    haystack_parts.extend(str(tag) for tag in tags[:4])
        haystack = " ".join(haystack_parts).casefold()
        if not haystack.strip():
            return False
        return any(str(keyword).strip().casefold() in haystack for keyword in keywords if str(keyword).strip())

    @staticmethod
    def _compute_learning_confidence_score(
        *,
        closed_count: int,
        hit_rate_pct: float | None,
        avg_return_pct: float | None,
    ) -> float:
        hit_component = max(-0.18, min(0.22, ((float(hit_rate_pct or 50.0) - 50.0) / 100.0)))
        return_component = max(-0.12, min(0.16, float(avg_return_pct or 0.0) / 25.0))
        sample_component = min(0.08, max(0, closed_count) * 0.015)
        score = 0.55 + hit_component + return_component + sample_component
        return max(0.2, min(0.95, score))

    @staticmethod
    def _confidence_label_for_score(score: float) -> str:
        if score >= 0.82:
            return "very_high"
        if score >= 0.7:
            return "high"
        if score >= 0.58:
            return "medium"
        return "low"

    @staticmethod
    def _signal_matches_labels(signal: MacroSurpriseSignal, labels: set[str]) -> bool:
        return (
            (signal.surprise_label in labels)
            or ((signal.baseline_surprise_label or "") in labels)
            or ((signal.consensus_surprise_label or "") in labels)
        )

    @staticmethod
    def _find_macro_reaction_asset(
        reaction: MacroMarketReaction,
        label: str,
    ) -> MacroReactionAssetMove | None:
        return next((item for item in reaction.reactions if item.label == label), None)

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
        dict[str, Any],
        list[MacroEvent],
        list[MacroSurpriseSignal],
        list[MacroMarketReaction],
    ]:
        news_task = news_client.fetch_latest_macro_news(limit=news_limit, when="1d")
        snapshot_task = market_data_client.get_core_market_snapshot()
        history_task = market_data_client.get_core_market_history(period=history_period, interval=history_interval, limit=history_limit)
        macro_task = market_data_client.get_macro_context()
        macro_intelligence_task = market_data_client.get_macro_intelligence()
        macro_event_task = market_data_client.get_macro_event_calendar(days_ahead=30)
        macro_surprise_task = market_data_client.get_macro_surprise_signals()
        macro_reaction_task = market_data_client.get_macro_market_reactions()
        research_task = self._gather_research_findings(
            research_client=research_client,
            research_query=research_query,
            limit=min(news_limit, 4),
        )
        (
            news,
            market_data,
            market_history,
            macro_context,
            research_findings,
            macro_intelligence,
            macro_event_calendar,
            macro_surprise_signals,
            macro_market_reactions,
        ) = await asyncio.gather(
            self._safe_async_call(news_task, default=[], source_name="news"),
            self._safe_async_call(snapshot_task, default={}, source_name="market_snapshot"),
            self._safe_async_call(history_task, default={}, source_name="market_history"),
            self._safe_async_call(
                macro_task,
                default={"vix": None, "tnx": None, "cpi_yoy": None},
                source_name="macro_context",
            ),
            self._safe_async_call(research_task, default=[], source_name="research"),
            self._safe_async_call(
                macro_intelligence_task,
                default={"headline": "macro backdrop unavailable", "signals": [], "highlights": [], "metrics": {}},
                source_name="macro_intelligence",
            ),
            self._safe_async_call(macro_event_task, default=[], source_name="macro_event_calendar"),
            self._safe_async_call(macro_surprise_task, default=[], source_name="macro_surprise_signals"),
            self._safe_async_call(macro_reaction_task, default=[], source_name="macro_market_reactions"),
        )

        news = self._guard_news_articles(news)
        market_data = self._guard_market_snapshot(market_data)
        macro_context = self._guard_macro_context(macro_context)
        research_findings = self._guard_research_findings(research_findings)
        macro_intelligence = self._guard_macro_intelligence(macro_intelligence)
        macro_event_calendar = [item for item in macro_event_calendar if isinstance(item, MacroEvent)]
        macro_surprise_signals = [item for item in macro_surprise_signals if isinstance(item, MacroSurpriseSignal)]
        macro_market_reactions = [item for item in macro_market_reactions if isinstance(item, MacroMarketReaction)]

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
        return (
            list(news),
            dict(market_data),
            trends,
            dict(macro_context),
            list(research_findings),
            dict(macro_intelligence),
            list(macro_event_calendar),
            list(macro_surprise_signals),
            list(macro_market_reactions),
        )

    async def _fetch_etf_exposures_for_market_data(
        self,
        *,
        market_data_client: MarketDataClient,
        market_data: Mapping[str, AssetQuote | None],
        limit: int = 4,
    ) -> list[ETFExposureProfile]:
        etf_tickers: list[str] = []
        for asset_name, quote in market_data.items():
            if "_etf" not in str(asset_name).casefold():
                continue
            ticker = quote.ticker if isinstance(quote, AssetQuote) else None
            normalized = str(ticker or "").strip().upper()
            if normalized and normalized not in etf_tickers:
                etf_tickers.append(normalized)
        if not etf_tickers:
            return []
        profiles = await self._safe_async_call(
            market_data_client.get_etf_exposure_profiles(etf_tickers[: max(1, limit)]),
            default={},
            source_name="etf_exposures",
        )
        if not isinstance(profiles, Mapping):
            return []
        return [
            item
            for item in profiles.values()
            if isinstance(item, ETFExposureProfile)
        ][: max(1, limit)]

    async def _safe_async_call(self, awaitable: Any, *, default: Any, source_name: str) -> Any:
        try:
            return await awaitable
        except Exception as exc:
            logger.warning("External provider degraded for {}: {}", source_name, exc)
            log_event("external_provider_degraded", source=source_name, error=str(exc))
            return default

    def _evaluate_data_quality(
        self,
        *,
        news: Sequence[NewsArticle],
        market_data: Mapping[str, AssetQuote | None],
        macro_context: Mapping[str, float | None] | None,
        macro_intelligence: Mapping[str, Any] | None,
        research_findings: Sequence[ResearchFinding] | None,
    ) -> dict[str, Any]:
        if self.data_quality_gate is None:
            return {
                "status": "disabled",
                "blocking": False,
                "score": 100.0,
                "issues": [],
                "checks": {},
                "gx": {"enabled": False, "executed": False, "success_percent": None, "warning": None},
            }
        report = self.data_quality_gate.evaluate(
            news=news,
            market_data=market_data,
            macro_context=macro_context,
            macro_intelligence=macro_intelligence,
            research_findings=research_findings or [],
        )
        payload = report.to_dict()
        log_event(
            "data_quality_gate",
            status=payload.get("status"),
            blocking=payload.get("blocking"),
            score=payload.get("score"),
            issues=len(payload.get("issues") or []),
        )
        return payload

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
        history_results = await asyncio.gather(
            *[market_data_client.get_history(ticker, period="6mo", interval="1d", limit=90) for ticker in live_tickers],
            return_exceptions=True,
        )
        fundamentals_map: dict[str, StockFundamentals | None] = {}
        for ticker, result in zip(live_tickers, fundamentals_results, strict=False):
            fundamentals_map[ticker] = None if isinstance(result, Exception) else result
        history_map: dict[str, list[OhlcvBar]] = {}
        for ticker, result in zip(live_tickers, history_results, strict=False):
            history_map[ticker] = result if isinstance(result, list) else []

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
            industry = fundamentals.industry if isinstance(fundamentals, StockFundamentals) else None
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

        holdings_payload: list[dict[str, Any]] = []
        for item in holding_reviews:
            fundamentals = fundamentals_map.get(item.ticker)
            holding_sector = fundamentals.sector if isinstance(fundamentals, StockFundamentals) else None
            holding_industry = fundamentals.industry if isinstance(fundamentals, StockFundamentals) else None
            holdings_payload.append(
                {
                    "ticker": item.ticker,
                    "category": item.category,
                    "sector": holding_sector,
                    "industry": holding_industry,
                    "themes": self._infer_portfolio_holding_themes(
                        ticker=item.ticker,
                        category=item.category,
                        sector=holding_sector,
                        industry=holding_industry,
                        note=item.note,
                    ),
                    "market_value": item.market_value,
                    "cost_basis": item.cost_basis,
                    "unrealized_pnl_pct": self._round_optional(item.unrealized_pnl_pct, 4),
                    "current_weight_pct": item.current_weight_pct,
                    "note": item.note,
                }
            )
        return {
            "total_market_value": round(total_market_value, 2),
            "holdings": holdings_payload,
            **self._build_portfolio_overlap_snapshot(
                holdings=holding_reviews,
                fundamentals_map=fundamentals_map,
                history_map=history_map,
            ),
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

    def _build_portfolio_constraint_summary(
        self,
        *,
        portfolio_snapshot: Mapping[str, Any] | None,
        factor_exposures: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(portfolio_snapshot, Mapping):
            return {}
        holdings = portfolio_snapshot.get("holdings")
        if not isinstance(holdings, list) or not holdings:
            return {}
        largest_position_pct = 0.0
        cash_weight_pct = 0.0
        growth_weight_pct = 0.0
        sector_weights: dict[str, float] = defaultdict(float)
        theme_weights: dict[str, float] = defaultdict(float)
        for item in holdings:
            if not isinstance(item, Mapping):
                continue
            weight_pct = self._as_float(item.get("current_weight_pct")) or 0.0
            largest_position_pct = max(largest_position_pct, weight_pct)
            category = str(item.get("category") or "").strip()
            if category == "cash":
                cash_weight_pct += weight_pct
            if category == "growth":
                growth_weight_pct += weight_pct
            sector = str(item.get("sector") or "").strip()
            if sector:
                sector_weights[sector] += weight_pct
            themes = item.get("themes")
            if isinstance(themes, list):
                for theme in themes:
                    theme_label = str(theme or "").strip()
                    if theme_label:
                        theme_weights[theme_label] += weight_pct
        dominant_sector = None
        dominant_sector_weight_pct = None
        if sector_weights:
            dominant_sector, dominant_sector_weight_pct = max(sector_weights.items(), key=lambda item: item[1])
        dominant_theme = None
        dominant_theme_weight_pct = None
        if theme_weights:
            dominant_theme, dominant_theme_weight_pct = max(theme_weights.items(), key=lambda item: item[1])
        average_pairwise_correlation = self._as_float(portfolio_snapshot.get("average_pairwise_correlation"))
        max_pairwise_correlation = self._as_float(portfolio_snapshot.get("max_pairwise_correlation"))
        high_correlation_pair_count = int(portfolio_snapshot.get("high_correlation_pair_count") or 0)
        high_correlation_pairs = (
            list(portfolio_snapshot.get("high_correlation_pairs") or [])
            if isinstance(portfolio_snapshot.get("high_correlation_pairs"), list)
            else []
        )
        flags: list[str] = []
        position_size_cap_pct = 5.0
        risk_budget_multiplier = 1.0
        if largest_position_pct >= 18.0:
            flags.append("single_name_concentration")
            position_size_cap_pct = min(position_size_cap_pct, 2.5)
            risk_budget_multiplier *= 0.7
        if growth_weight_pct >= 55.0:
            flags.append("growth_bucket_crowded")
            position_size_cap_pct = min(position_size_cap_pct, 3.0)
            risk_budget_multiplier *= 0.82
        if dominant_sector_weight_pct is not None and dominant_sector_weight_pct >= 35.0:
            flags.append("sector_concentration")
            position_size_cap_pct = min(position_size_cap_pct, 3.0)
            risk_budget_multiplier *= 0.85
        if dominant_theme_weight_pct is not None and dominant_theme_weight_pct >= 42.0:
            flags.append("theme_overlap")
            position_size_cap_pct = min(position_size_cap_pct, 2.5)
            risk_budget_multiplier *= 0.8
        if (
            max_pairwise_correlation is not None
            and max_pairwise_correlation >= 0.82
            and high_correlation_pair_count >= 1
        ) or high_correlation_pair_count >= 2:
            flags.append("correlation_cluster")
            position_size_cap_pct = min(position_size_cap_pct, 2.5)
            risk_budget_multiplier *= 0.78
        exposure_weights = (
            dict((factor_exposures or {}).get("exposure_weights") or {})
            if isinstance(factor_exposures, Mapping)
            else {}
        )
        duration_sensitive_pct = self._as_float(exposure_weights.get("duration_sensitive")) or 0.0
        equity_beta_pct = self._as_float(exposure_weights.get("equity_beta")) or 0.0
        mega_cap_ai_pct = self._as_float(exposure_weights.get("mega_cap_ai")) or 0.0
        if duration_sensitive_pct >= 45.0:
            flags.append("duration_factor_crowded")
            position_size_cap_pct = min(position_size_cap_pct, 2.8)
            risk_budget_multiplier *= 0.84
        if equity_beta_pct >= 78.0:
            flags.append("beta_factor_crowded")
            position_size_cap_pct = min(position_size_cap_pct, 2.8)
            risk_budget_multiplier *= 0.86
        if mega_cap_ai_pct >= 32.0:
            flags.append("mega_cap_ai_crowded")
            position_size_cap_pct = min(position_size_cap_pct, 2.6)
            risk_budget_multiplier *= 0.82
        if cash_weight_pct <= 2.5:
            flags.append("cash_buffer_thin")
            risk_budget_multiplier *= 0.9
        allow_new_risk = len(flags) == 0 or (
            largest_position_pct < 24.0
            and (dominant_theme_weight_pct or 0.0) < 50.0
            and (max_pairwise_correlation or 0.0) < 0.9
        )
        return {
            "largest_position_pct": round(largest_position_pct, 1),
            "cash_weight_pct": round(cash_weight_pct, 1),
            "growth_weight_pct": round(growth_weight_pct, 1),
            "dominant_sector": dominant_sector,
            "dominant_sector_weight_pct": round(dominant_sector_weight_pct, 1) if dominant_sector_weight_pct is not None else None,
            "dominant_theme": dominant_theme,
            "dominant_theme_weight_pct": round(dominant_theme_weight_pct, 1) if dominant_theme_weight_pct is not None else None,
            "theme_weights": {
                key: round(value, 1)
                for key, value in sorted(theme_weights.items(), key=lambda item: (-item[1], item[0]))[:6]
            },
            "factor_exposures": dict(factor_exposures or {}) if isinstance(factor_exposures, Mapping) else {},
            "average_pairwise_correlation": (
                round(average_pairwise_correlation, 2) if average_pairwise_correlation is not None else None
            ),
            "max_pairwise_correlation": round(max_pairwise_correlation, 2) if max_pairwise_correlation is not None else None,
            "high_correlation_pair_count": high_correlation_pair_count,
            "high_correlation_pairs": high_correlation_pairs[:4],
            "flags": flags,
            "allow_new_risk": allow_new_risk,
            "position_size_cap_pct": round(max(1.0, position_size_cap_pct), 1),
            "risk_budget_multiplier": round(max(0.45, min(1.0, risk_budget_multiplier)), 2),
        }

    @staticmethod
    def _infer_portfolio_holding_themes(
        *,
        ticker: str,
        category: str,
        sector: str | None,
        industry: str | None,
        note: str | None,
    ) -> list[str]:
        normalized_ticker = str(ticker or "").strip().upper()
        normalized_sector = str(sector or "").strip().casefold()
        normalized_industry = str(industry or "").strip().casefold()
        normalized_note = str(note or "").strip().casefold()
        normalized_category = str(category or "").strip().casefold()
        themes: list[str] = []
        if normalized_category == "cash":
            themes.append("cash")
        elif normalized_category == "gold":
            themes.append("gold")
        elif normalized_category == "core_etf":
            themes.append("broad_index")
        if normalized_sector == "technology":
            themes.append("technology")
        if normalized_sector == "energy":
            themes.append("energy")
        if normalized_sector == "financials":
            themes.append("financials")
        if normalized_sector == "healthcare":
            themes.append("healthcare")
        if normalized_sector == "utilities":
            themes.append("defensive_yield")
        if any(token in normalized_industry for token in ("semiconductor", "software", "internet", "cloud")):
            themes.append("ai_big_tech")
        if normalized_ticker in {
            "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSM", "META", "AMZN", "GOOGL", "GOOG", "PLTR", "ORCL", "CRM",
        }:
            themes.append("ai_big_tech")
        if "growth" in normalized_note or normalized_category == "growth":
            themes.append("growth")
        if "defensive" in normalized_note or normalized_category == "defensive":
            themes.append("defensive")
        return list(dict.fromkeys(theme for theme in themes if theme))

    def _build_portfolio_overlap_snapshot(
        self,
        *,
        holdings: Sequence[PortfolioHoldingReview],
        fundamentals_map: Mapping[str, StockFundamentals | None],
        history_map: Mapping[str, Sequence[OhlcvBar]],
    ) -> dict[str, Any]:
        holding_weights = {item.ticker: float(item.current_weight_pct) for item in holdings if item.ticker}
        theme_weights: dict[str, float] = defaultdict(float)
        for item in holdings:
            fundamentals = fundamentals_map.get(item.ticker)
            themes = self._infer_portfolio_holding_themes(
                ticker=item.ticker,
                category=item.category,
                sector=fundamentals.sector if isinstance(fundamentals, StockFundamentals) else None,
                industry=fundamentals.industry if isinstance(fundamentals, StockFundamentals) else None,
                note=item.note,
            )
            for theme in themes:
                theme_weights[theme] += float(item.current_weight_pct)

        return_series: dict[str, pd.Series] = {}
        for ticker, bars in history_map.items():
            closes = [self._as_float(getattr(bar, "close", None)) for bar in bars]
            normalized_closes = [float(close) for close in closes if close is not None and close > 0]
            if len(normalized_closes) < 20:
                continue
            series = pd.Series(normalized_closes, dtype=float).pct_change().dropna().tail(60).reset_index(drop=True)
            if len(series) >= 20:
                return_series[ticker] = series

        average_pairwise_correlation = None
        max_pairwise_correlation = None
        high_correlation_pairs: list[dict[str, Any]] = []
        if len(return_series) >= 2:
            frame = pd.concat(return_series, axis=1)
            correlation = frame.corr(min_periods=20)
            pair_values: list[tuple[str, str, float, float]] = []
            tickers = list(correlation.columns)
            for index, left in enumerate(tickers):
                for right in tickers[index + 1 :]:
                    corr_value = correlation.loc[left, right]
                    if pd.isna(corr_value):
                        continue
                    numeric_corr = float(corr_value)
                    pair_weight = max(holding_weights.get(left, 0.0), holding_weights.get(right, 0.0))
                    pair_values.append((left, right, numeric_corr, pair_weight))
            if pair_values:
                average_pairwise_correlation = sum(item[2] for item in pair_values) / len(pair_values)
                max_pairwise_correlation = max(item[2] for item in pair_values)
                pair_values.sort(key=lambda item: (-item[2], -item[3], item[0], item[1]))
                for left, right, corr_value, _pair_weight in pair_values:
                    if corr_value < 0.75:
                        continue
                    high_correlation_pairs.append(
                        {
                            "pair": f"{left}/{right}",
                            "correlation": round(corr_value, 2),
                        }
                    )
                    if len(high_correlation_pairs) >= 5:
                        break

        dominant_theme = None
        dominant_theme_weight_pct = None
        if theme_weights:
            dominant_theme, dominant_theme_weight_pct = max(theme_weights.items(), key=lambda item: item[1])
        return {
            "theme_weights": {
                key: round(value, 1)
                for key, value in sorted(theme_weights.items(), key=lambda item: (-item[1], item[0]))[:8]
            },
            "dominant_theme": dominant_theme,
            "dominant_theme_weight_pct": round(dominant_theme_weight_pct, 1) if dominant_theme_weight_pct is not None else None,
            "average_pairwise_correlation": (
                round(average_pairwise_correlation, 3) if average_pairwise_correlation is not None else None
            ),
            "max_pairwise_correlation": round(max_pairwise_correlation, 3) if max_pairwise_correlation is not None else None,
            "high_correlation_pair_count": len(high_correlation_pairs),
            "high_correlation_pairs": high_correlation_pairs,
        }

    def _build_factor_exposure_summary(
        self,
        *,
        portfolio_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(portfolio_snapshot, Mapping):
            return {}
        holdings = portfolio_snapshot.get("holdings")
        if not isinstance(holdings, list) or not holdings:
            return {}
        exposure_weights: dict[str, float] = defaultdict(float)
        for item in holdings:
            if not isinstance(item, Mapping):
                continue
            weight_pct = self._as_float(item.get("current_weight_pct")) or 0.0
            if weight_pct <= 0:
                continue
            ticker = str(item.get("ticker") or "").strip().upper()
            category = str(item.get("category") or "").strip().casefold()
            sector = str(item.get("sector") or "").strip().casefold()
            themes = {
                str(theme).strip().casefold()
                for theme in (item.get("themes") or [])
                if str(theme).strip()
            }
            if category == "cash":
                exposure_weights["cash"] += weight_pct
                continue
            exposure_weights["equity_beta"] += weight_pct
            if category == "gold" or "gold" in themes:
                exposure_weights["inflation_hedge"] += weight_pct
            if category == "defensive" or {"defensive", "defensive_yield"} & themes or sector in {"utilities", "healthcare", "consumer staples"}:
                exposure_weights["defensive"] += weight_pct
            if sector == "energy" or "energy" in themes:
                exposure_weights["commodity_inflation"] += weight_pct
            if sector == "financials" or "financials" in themes:
                exposure_weights["financials_value"] += weight_pct
            if category == "core_etf" or "broad_index" in themes or ticker in {"SPY", "VOO", "VTI"}:
                exposure_weights["broad_beta"] += weight_pct
            if (
                "growth" in themes
                or "technology" in themes
                or "ai_big_tech" in themes
                or ticker in {"QQQ", "VUG", "TLT"}
            ):
                exposure_weights["duration_sensitive"] += weight_pct
            if "ai_big_tech" in themes or ticker in {"QQQ", "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "META", "AMZN", "GOOGL", "GOOG"}:
                exposure_weights["mega_cap_ai"] += weight_pct

        top_exposures = [
            {"factor": key, "weight_pct": round(value, 1)}
            for key, value in sorted(exposure_weights.items(), key=lambda item: (-item[1], item[0]))
            if value > 0
        ]
        flags: list[str] = []
        if (self._as_float(exposure_weights.get("duration_sensitive")) or 0.0) >= 45.0:
            flags.append("duration_sensitive_crowded")
        if (self._as_float(exposure_weights.get("mega_cap_ai")) or 0.0) >= 32.0:
            flags.append("mega_cap_ai_crowded")
        if (self._as_float(exposure_weights.get("equity_beta")) or 0.0) >= 78.0:
            flags.append("equity_beta_high")
        if (self._as_float(exposure_weights.get("inflation_hedge")) or 0.0) <= 2.0:
            flags.append("inflation_hedge_light")
        return {
            "exposure_weights": {
                str(item["factor"]): float(item["weight_pct"])
                for item in top_exposures[:8]
                if isinstance(item, Mapping) and item.get("factor") is not None and item.get("weight_pct") is not None
            },
            "top_exposures": top_exposures[:5],
            "top_exposure_factor": top_exposures[0]["factor"] if top_exposures else None,
            "top_exposure_weight_pct": top_exposures[0]["weight_pct"] if top_exposures else None,
            "flags": flags,
        }

    def _build_source_health_summary(
        self,
        *,
        macro_intelligence: Mapping[str, Any] | None,
        macro_event_calendar: Sequence[MacroEvent] | None,
        macro_surprise_signals: Sequence[MacroSurpriseSignal] | None,
        macro_market_reactions: Sequence[MacroMarketReaction] | None,
        news_items: Sequence[NewsArticle] | None,
        research_items: Sequence[ResearchFinding] | None,
        company_intelligence: Sequence[CompanyIntelligence] | None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        score = 40.0
        rationale: list[str] = []
        fresh_sources: list[str] = []
        stale_sources: list[str] = []
        missing_sources: list[str] = []
        source_ages_hours: list[float] = []
        macro_sources = list((macro_intelligence or {}).get("sources_used") or []) if isinstance(macro_intelligence, Mapping) else []
        if macro_sources:
            score += min(16.0, len(macro_sources) * 3.0)
            fresh_sources.extend(str(item).strip() for item in macro_sources if str(item).strip())
            rationale.append("macro data stack is populated")
        else:
            missing_sources.append("macro_intelligence")
        if macro_event_calendar:
            future_events = [item for item in macro_event_calendar if item.scheduled_at >= now]
            if future_events:
                score += 8.0
                fresh_sources.append("macro_event_calendar")
                source_ages_hours.append(
                    min(abs((future_events[0].scheduled_at - now).total_seconds()) / 3600.0, 24.0)
                )
            else:
                stale_sources.append("macro_event_calendar")
        else:
            missing_sources.append("macro_event_calendar")
        if macro_surprise_signals:
            recent_surprises = [
                item for item in macro_surprise_signals
                if item.released_at is not None and abs((now - item.released_at).total_seconds()) <= 36 * 3600
            ]
            if recent_surprises:
                score += 12.0
                fresh_sources.append("macro_surprise_engine")
                rationale.append("recent macro surprise data is available")
                source_ages_hours.extend(
                    abs((now - item.released_at).total_seconds()) / 3600.0
                    for item in recent_surprises[:3]
                    if item.released_at is not None
                )
            else:
                stale_sources.append("macro_surprise_engine")
        else:
            missing_sources.append("macro_surprise_engine")
        if macro_market_reactions:
            recent_reactions = [
                item for item in macro_market_reactions
                if abs((now - item.released_at).total_seconds()) <= 36 * 3600
            ]
            if recent_reactions:
                score += 10.0
                fresh_sources.append("macro_market_reaction")
                source_ages_hours.extend(
                    abs((now - item.released_at).total_seconds()) / 3600.0
                    for item in recent_reactions[:3]
                    if item.released_at is not None
                )
            else:
                stale_sources.append("macro_market_reaction")
        else:
            missing_sources.append("macro_market_reaction")
        if news_items:
            score += min(8.0, len(news_items) * 1.5)
            fresh_sources.append("news")
            source_ages_hours.extend(
                self._estimate_hours_since(now=now, value=item.published_at)
                for item in list(news_items)[:3]
                if self._estimate_hours_since(now=now, value=item.published_at) is not None
            )
        else:
            missing_sources.append("news")
        if research_items:
            score += min(6.0, len(research_items) * 2.0)
            fresh_sources.append("research")
            source_ages_hours.extend(
                self._estimate_hours_since(now=now, value=item.published_at)
                for item in list(research_items)[:3]
                if self._estimate_hours_since(now=now, value=item.published_at) is not None
            )
        if company_intelligence:
            score += min(8.0, len(company_intelligence) * 1.5)
            fresh_sources.append("company_intelligence")
            source_ages_hours.extend(
                self._estimate_hours_since(
                    now=now,
                    value=item.latest_8k_filed_at or item.latest_10q_filed_at or item.latest_10k_filed_at,
                )
                for item in list(company_intelligence)[:3]
                if self._estimate_hours_since(
                    now=now,
                    value=item.latest_8k_filed_at or item.latest_10q_filed_at or item.latest_10k_filed_at,
                )
                is not None
            )
        avg_source_age_hours = (
            round(sum(source_ages_hours) / len(source_ages_hours), 1) if source_ages_hours else None
        )
        latency_penalty = 0.0
        if avg_source_age_hours is not None:
            latency_penalty = round(min(16.0, max(0.0, avg_source_age_hours - 6.0) * 0.8), 1)
        stale_penalty = round((len(stale_sources) * 5.0) + (len(missing_sources) * 2.5), 1)
        total_penalty = round(latency_penalty + stale_penalty, 1)
        freshness_pct = max(
            0.0,
            min(100.0, 100.0 - (len(stale_sources) * 18.0) - (len(missing_sources) * 10.0) - latency_penalty),
        )
        health_score = round(max(5.0, min(100.0, score - total_penalty)), 1)
        label = "strong" if health_score >= 75 else "mixed" if health_score >= 55 else "fragile"
        if stale_sources:
            rationale.append(f"stale: {', '.join(stale_sources[:3])}")
        if missing_sources:
            rationale.append(f"missing: {', '.join(missing_sources[:3])}")
        if latency_penalty > 0:
            rationale.append(f"latency penalty {latency_penalty} from avg source age {avg_source_age_hours}h")
        critical_sources = ["macro_intelligence", "macro_event_calendar", "macro_surprise_engine", "macro_market_reaction"]
        sla_breached_sources = [
            source
            for source in critical_sources
            if source in stale_sources or source in missing_sources
        ]
        coverage_pct = round((len(fresh_sources) / max(1, len(set(fresh_sources + stale_sources + missing_sources)))) * 100.0, 1)
        outage_detected = len(sla_breached_sources) >= 2 or ("macro_intelligence" in missing_sources and "macro_surprise_engine" in missing_sources)
        sla_status = "healthy"
        if outage_detected:
            sla_status = "outage"
        elif sla_breached_sources or latency_penalty >= 8.0:
            sla_status = "degraded"
        return {
            "score": health_score,
            "label": label,
            "freshness_pct": round(freshness_pct, 1),
            "coverage_pct": coverage_pct,
            "avg_source_age_hours": avg_source_age_hours,
            "latency_penalty": latency_penalty,
            "stale_penalty": stale_penalty,
            "total_penalty": total_penalty,
            "critical_sources": critical_sources,
            "sla_breached_sources": sla_breached_sources[:6],
            "sla_status": sla_status,
            "outage_detected": outage_detected,
            "fresh_sources": list(dict.fromkeys(fresh_sources))[:8],
            "stale_sources": stale_sources[:6],
            "missing_sources": missing_sources[:6],
            "rationale": rationale[:5],
        }

    @staticmethod
    def _estimate_hours_since(*, now: datetime, value: datetime | None) -> float | None:
        if value is None:
            return None
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return round(abs((now - normalized).total_seconds()) / 3600.0, 1)

    def _blend_confidence_with_source_health(
        self,
        *,
        assessment: ConfidenceAssessment,
        source_health: Mapping[str, Any] | None,
        portfolio_constraints: Mapping[str, Any] | None,
    ) -> ConfidenceAssessment:
        health_score = self._as_float((source_health or {}).get("score"))
        total_penalty = self._as_float((source_health or {}).get("total_penalty"))
        adjusted_score = float(assessment.score)
        rationale = list(assessment.rationale)
        if health_score is not None:
            adjusted_score += max(-0.08, min(0.1, ((health_score - 55.0) / 100.0)))
            if health_score >= 72:
                rationale.append("source_health_strong")
            elif health_score <= 45:
                rationale.append("source_health_fragile")
        if total_penalty is not None and total_penalty >= 8.0:
            adjusted_score -= max(0.03, min(0.09, total_penalty / 100.0))
            rationale.append("source_health_latency_penalty")
        if bool((source_health or {}).get("outage_detected")):
            adjusted_score -= 0.12
            rationale.append("source_outage_detected")
        if isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True)):
            adjusted_score -= 0.06
            rationale.append("portfolio_risk_budget_tight")
        clipped = round(min(0.95, max(0.2, adjusted_score)), 2)
        return ConfidenceAssessment(
            score=clipped,
            label=self._confidence_label_for_score(clipped),
            rationale=tuple(dict.fromkeys(rationale))[:6],
        )

    def _build_regime_specific_playbooks(
        self,
        *,
        macro_regime: MacroRegimeAssessment,
        macro_context: Mapping[str, float | None] | None,
        portfolio_constraints: Mapping[str, Any] | None,
    ) -> list[RegimeSpecificPlaybook]:
        regime = macro_regime.regime
        constrained = isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True))
        playbooks: list[RegimeSpecificPlaybook] = []
        if regime == "soft_landing":
            playbooks.append(
                RegimeSpecificPlaybook(
                    playbook_key="soft_landing_quality_growth",
                    regime=regime,
                    title="Soft Landing Quality Growth",
                    action="เอียงเข้าหา quality growth และ broad beta ได้ แต่ยังคง cash buffer บางส่วน",
                    risk_watch="breadth, QQQ/SPY leadership, payroll revisions",
                    conviction="high" if not constrained else "medium",
                    sizing_bias="moderately_add",
                    ttl_bias="medium",
                    rationale=("growth is intact", "breadth still matters", "avoid crowding the same theme"),
                )
            )
        elif regime == "disinflationary_growth":
            playbooks.append(
                RegimeSpecificPlaybook(
                    playbook_key="disinflation_extend_duration",
                    regime=regime,
                    title="Disinflation Quality Duration",
                    action="ให้น้ำหนักกับ quality growth และ duration-sensitive names ที่ execution ดี",
                    risk_watch="US10Y, core PCE, QQQ breadth",
                    conviction="high" if not constrained else "medium",
                    sizing_bias="add_selectively",
                    ttl_bias="medium",
                    rationale=("inflation pressure is easing", "duration tailwind can extend", "execution quality still matters"),
                )
            )
        elif regime == "inflation_rebound":
            playbooks.append(
                RegimeSpecificPlaybook(
                    playbook_key="inflation_rebound_defense",
                    regime=regime,
                    title="Inflation Rebound Defense",
                    action="ลด duration-sensitive growth, เพิ่มทอง/defensive/cash และยอมใช้ TTL สั้นลง",
                    risk_watch="core PCE, oil, US10Y, DXY",
                    conviction="high",
                    sizing_bias="reduce_beta",
                    ttl_bias="short",
                    rationale=("inflation is re-accelerating", "valuation pressure rises quickly", "favor pricing power"),
                )
            )
        elif regime == "recession_risk":
            playbooks.append(
                RegimeSpecificPlaybook(
                    playbook_key="recession_quality_cash",
                    regime=regime,
                    title="Recession Quality and Cash",
                    action="งดไล่ cyclical beta, เน้น cash, defensive, และ quality balance sheets",
                    risk_watch="credit spread, unemployment, breadth diffusion",
                    conviction="high",
                    sizing_bias="cut_risk",
                    ttl_bias="short",
                    rationale=("growth slowdown is dominant", "protect capital first", "avoid weak liquidity"),
                )
            )
        elif regime == "stagflation_risk":
            playbooks.append(
                RegimeSpecificPlaybook(
                    playbook_key="stagflation_pricing_power",
                    regime=regime,
                    title="Stagflation Pricing Power",
                    action="เน้น sectors ที่มี pricing power, commodity hedge, และหลีกเลี่ยง leverage สูง",
                    risk_watch="energy, gold, real rates, consumer weakness",
                    conviction="medium",
                    sizing_bias="defensive_barbell",
                    ttl_bias="short",
                    rationale=("growth and inflation signals conflict", "avoid leverage", "keep hedge exposure live"),
                )
            )
        else:
            playbooks.append(
                RegimeSpecificPlaybook(
                    playbook_key="mixed_transition_small_ball",
                    regime=regime,
                    title="Mixed Transition Small Ball",
                    action="ถือ size เล็กลง, รอ confirmation เพิ่ม, และยอม abstain ได้ง่ายขึ้น",
                    risk_watch="breadth, revisions, event risk",
                    conviction="medium" if not constrained else "low",
                    sizing_bias="small_size",
                    ttl_bias="short",
                    rationale=("macro signals are mixed", "protect optionality", "demand cleaner confirmation"),
                )
            )
        return playbooks

    def _build_no_trade_decision(
        self,
        *,
        market_confidence: ConfidenceAssessment,
        source_health: Mapping[str, Any] | None,
        portfolio_constraints: Mapping[str, Any] | None,
        thesis_invalidation: Mapping[str, Any] | None,
        asset_scope: str,
        question: str | None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if market_confidence.score <= 0.48:
            reasons.append("cross-asset confidence is still too low")
        if self._as_float((source_health or {}).get("score")) is not None and float((source_health or {}).get("score") or 0.0) < 48:
            reasons.append("source health is not strong enough")
        if self._as_float((source_health or {}).get("total_penalty")) is not None and float((source_health or {}).get("total_penalty") or 0.0) >= 10.0:
            reasons.append("source latency / stale penalty is still too high")
        if bool((source_health or {}).get("outage_detected")):
            reasons.append("critical data SLA is degraded and at least one source looks unavailable")
        if isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True)):
            reasons.append("portfolio constraints already limit new risk")
        if isinstance(thesis_invalidation, Mapping) and bool(thesis_invalidation.get("has_active_invalidation")):
            reasons.append(str(thesis_invalidation.get("summary") or "existing thesis is being invalidated"))
        should_abstain = bool(reasons)
        return {
            "should_abstain": should_abstain,
            "scope": asset_scope,
            "question": question,
            "summary": "prefer no-trade / wait mode" if should_abstain else "actionable with risk controls",
            "reasons": reasons[:4],
            "action": (
                "ลดการตัดสินใจใหม่ รอ data freshness และ portfolio headroom กลับมาก่อน"
                if should_abstain
                else "ยังเปิดสถานะได้ แต่ควรยึด quality, smaller size, และ confirm-driven execution"
            ),
        }

    def _build_champion_challenger_view(
        self,
        *,
        market_confidence: ConfidenceAssessment,
        source_health: Mapping[str, Any] | None,
        portfolio_constraints: Mapping[str, Any] | None,
        no_trade_decision: Mapping[str, Any] | None,
        source_learning_map: Mapping[str, float],
        regime_playbooks: Sequence[RegimeSpecificPlaybook],
        factor_exposures: Mapping[str, Any] | None,
        thesis_invalidation: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        baseline_score = round(float(market_confidence.score), 2)
        adaptive_score = baseline_score
        health_score = self._as_float((source_health or {}).get("score"))
        total_penalty = self._as_float((source_health or {}).get("total_penalty"))
        if health_score is not None:
            adaptive_score = round(adaptive_score + max(-0.08, min(0.08, ((health_score - 55.0) / 100.0))), 2)
        if total_penalty is not None:
            adaptive_score = round(adaptive_score - max(0.0, min(0.08, total_penalty / 100.0)), 2)
        if bool((source_health or {}).get("outage_detected")):
            adaptive_score = round(adaptive_score - 0.08, 2)
        if isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True)):
            adaptive_score = round(adaptive_score - 0.07, 2)
        if no_trade_decision and bool(no_trade_decision.get("should_abstain")):
            adaptive_score = round(adaptive_score - 0.05, 2)
        learning_score = self._average_source_learning_score(source_learning_map, tuple(source_learning_map.keys())[:4])
        if learning_score is not None:
            adaptive_score = round(adaptive_score + max(-0.04, min(0.05, ((learning_score - 60.0) / 100.0))), 2)
        top_exposure_weight = self._as_float((factor_exposures or {}).get("top_exposure_weight_pct"))
        if top_exposure_weight is not None and top_exposure_weight >= 38.0:
            adaptive_score = round(adaptive_score - 0.03, 2)
        invalidation_score = self._as_float((thesis_invalidation or {}).get("score"))
        if invalidation_score is not None and invalidation_score >= 55.0:
            adaptive_score = round(adaptive_score - max(0.03, min(0.08, invalidation_score / 1000.0)), 2)
        adaptive_score = round(max(0.2, min(0.95, adaptive_score)), 2)
        policies = [
            {
                "name": "baseline_confidence_only",
                "score": baseline_score,
                "uses": ["market_confidence"],
            },
            {
                "name": "adaptive_champion",
                "score": adaptive_score,
                "uses": ["source_health", "portfolio_constraints", "execution_learning", "regime_playbooks"],
            },
            {
                "name": "strict_abstain_guard",
                "score": round(max(0.2, adaptive_score - (0.06 if no_trade_decision and bool(no_trade_decision.get("should_abstain")) else 0.02)), 2),
                "uses": ["adaptive_champion", "no_trade_guard", "thesis_invalidation"],
            },
        ]
        policies.sort(key=lambda item: (float(item.get("score") or -999.0), str(item.get("name") or "")), reverse=True)
        champion = dict(policies[0])
        challenger = dict(policies[1] if len(policies) > 1 else policies[0])
        return {
            "recommended_policy": str(champion.get("name") or "adaptive_champion"),
            "delta_vs_baseline": round(float(champion.get("score") or adaptive_score) - baseline_score, 2),
            "champion": {
                **champion,
                "regime_playbook_count": len(regime_playbooks),
            },
            "challenger": challenger,
            "runner": {
                "policy_count": len(policies),
                "winner": str(champion.get("name") or "adaptive_champion"),
                "policies": policies,
            },
        }

    def _build_thesis_invalidation_summary(
        self,
        *,
        thesis_memory: Sequence[Mapping[str, Any]] | None,
        macro_regime: MacroRegimeAssessment,
        macro_surprise_signals: Sequence[MacroSurpriseSignal] | None,
        macro_market_reactions: Sequence[MacroMarketReaction] | None,
        company_intelligence: Sequence[CompanyIntelligence] | None,
    ) -> dict[str, Any]:
        thesis_tokens = {
            token
            for item in (thesis_memory or [])
            if isinstance(item, Mapping)
            for token in (
                [str(tag).strip().casefold() for tag in (item.get("tags") or []) if str(tag).strip()]
                + re.findall(r"[a-z_]{4,}", str(item.get("thesis_text") or "").casefold())
            )
        }
        signals: list[dict[str, Any]] = []
        score = 0.0
        if macro_regime.regime in {"inflation_rebound", "stagflation_risk"} and thesis_tokens & {"growth", "duration", "disinflation", "soft_landing"}:
            signals.append(
                {
                    "source": "macro_regime",
                    "label": "regime_shift_against_duration",
                    "severity": "high",
                    "reason": "macro regime moved away from the duration/growth thesis base",
                }
            )
            score += 26.0
        if macro_regime.regime == "recession_risk" and thesis_tokens & {"cyclical", "growth", "beta"}:
            signals.append(
                {
                    "source": "macro_regime",
                    "label": "recession_risk_against_beta",
                    "severity": "high",
                    "reason": "recession-risk regime now conflicts with beta/cyclical thesis",
                }
            )
            score += 24.0
        recent_negative_surprises = [
            item
            for item in list(macro_surprise_signals or [])[:4]
            if item.surprise_label in {"hotter_than_baseline", "stronger_than_baseline", "hawkish_shift"}
        ]
        if recent_negative_surprises and thesis_tokens & {"duration", "growth", "disinflation"}:
            signals.append(
                {
                    "source": "macro_surprise_engine",
                    "label": "macro_surprise_against_duration",
                    "severity": "medium",
                    "reason": "recent macro surprise is inconsistent with easing / duration-sensitive thesis",
                }
            )
            score += 18.0
        not_confirmed_reactions = [
            item
            for item in list(macro_market_reactions or [])[:4]
            if str(item.confirmation_label or "").strip().casefold() in {"not_confirmed", "mixed"}
        ]
        if not_confirmed_reactions:
            signals.append(
                {
                    "source": "macro_market_reaction",
                    "label": "market_not_confirming_prior_thesis",
                    "severity": "medium",
                    "reason": "cross-asset reaction is not confirming the prior thesis cleanly",
                }
            )
            score += 14.0
        negative_guidance = [
            item
            for item in list(company_intelligence or [])[:6]
            if str(item.guidance_signal or "").strip().casefold() in {"negative", "mixed"}
            or str(item.one_off_signal or "").strip().casefold() == "high"
        ]
        if negative_guidance:
            signals.append(
                {
                    "source": "company_intelligence",
                    "label": "guidance_or_quality_break",
                    "severity": "medium",
                    "reason": "company guidance / filing quality has weakened versus the stored thesis",
                }
            )
            score += 16.0
        clipped_score = round(max(0.0, min(100.0, score)), 1)
        severity = "high" if clipped_score >= 55 else "medium" if clipped_score >= 25 else "low"
        has_active_invalidation = clipped_score >= 25.0 and bool(signals)
        return {
            "has_active_invalidation": has_active_invalidation,
            "score": clipped_score,
            "severity": severity,
            "summary": (
                "stored thesis now has active invalidation pressure"
                if has_active_invalidation
                else "no strong thesis invalidation signal yet"
            ),
            "recommended_action": (
                "ลด conviction, รีเช็ก thesis เดิม, และอย่าเพิ่ม risk จนกว่าจะมี data ยืนยันใหม่"
                if has_active_invalidation
                else "ถือ thesis เดิมได้ แต่ยังควร revalidate เมื่อมี event ใหม่"
            ),
            "signals": signals[:4],
        }

    def _build_thesis_lifecycle_summary(
        self,
        *,
        thesis_memory: Sequence[Mapping[str, Any]] | None,
        source_health: Mapping[str, Any] | None,
        market_confidence: ConfidenceAssessment,
        thesis_invalidation: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        thesis_count = len([item for item in (thesis_memory or []) if isinstance(item, Mapping)])
        invalidation_active = bool((thesis_invalidation or {}).get("has_active_invalidation"))
        invalidation_score = self._as_float((thesis_invalidation or {}).get("score")) or 0.0
        source_health_score = self._as_float((source_health or {}).get("score")) or 0.0
        if thesis_count == 0:
            stage = "born"
        elif invalidation_active and invalidation_score >= 55.0:
            stage = "invalidated"
        elif invalidation_active:
            stage = "weakening"
        elif market_confidence.score >= 0.68 and source_health_score >= 65.0:
            stage = "confirmed"
        elif source_health_score < 45.0:
            stage = "archived"
        else:
            stage = "born"
        return {
            "stage": stage,
            "thesis_count": thesis_count,
            "summary": {
                "born": "thesis exists but still needs more confirmation",
                "confirmed": "thesis is being confirmed by the current evidence",
                "weakening": "thesis still exists but the evidence quality is deteriorating",
                "invalidated": "thesis no longer fits the latest evidence set",
                "archived": "thesis should be parked until data quality improves",
            }.get(stage, "thesis lifecycle is mixed"),
            "invalidation_score": round(invalidation_score, 1),
            "source_health_score": round(source_health_score, 1),
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
        management_commentary_lines = self._format_management_commentary_lines(payload.get("management_commentary"))
        microstructure_lines = self._format_microstructure_lines(payload.get("microstructure"))
        policy_feed_lines = self._format_policy_feed_lines(payload.get("policy_events"))
        ownership_lines = self._format_ownership_intelligence_lines(payload.get("ownership_intelligence"))
        order_flow_lines = self._format_order_flow_lines(payload.get("order_flow"))
        earnings_lines = self._format_earnings_lines(payload.get("earnings_calendar"))
        earnings_setup_lines = self._format_earnings_setup_lines(payload.get("earnings_setups"))
        earnings_surprise_lines = self._format_earnings_surprise_lines(payload.get("earnings_surprises"))
        return (
            f"รายงานชนิด: {report_kind}\n"
            "มหภาค:\n"
            f"{self._format_macro_lines(payload.get('macro_context'))}\n\n"
            "Macro intelligence:\n"
            f"{self._format_macro_intelligence_lines(payload.get('macro_intelligence'))}\n\n"
            "Upcoming macro events:\n"
            f"{self._format_macro_event_calendar_lines(payload.get('macro_event_calendar'))}\n\n"
            "Macro surprise engine:\n"
            f"{self._format_macro_surprise_lines(payload.get('macro_surprise_signals'))}\n\n"
            "Macro market reaction:\n"
            f"{self._format_macro_market_reaction_lines(payload.get('macro_market_reactions'))}\n\n"
            "Post-event playbooks:\n"
            f"{self._format_macro_playbook_lines(payload.get('macro_post_event_playbooks'))}\n\n"
            "Thesis memory:\n"
            f"{self._format_thesis_memory_lines(payload.get('thesis_memory'))}\n\n"
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
            "Fed / ECB policy feed:\n"
            f"{policy_feed_lines}\n\n"
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
            "Management commentary:\n"
            f"{management_commentary_lines}\n\n"
            "Microstructure:\n"
            f"{microstructure_lines}\n\n"
            "Ownership / 13F / 13D-G:\n"
            f"{ownership_lines}\n\n"
            "Options order-flow:\n"
            f"{order_flow_lines}\n\n"
            "Company intelligence:\n"
            f"{self._format_company_intelligence_lines(payload.get('company_intelligence'))}\n\n"
            "ETF exposure intelligence:\n"
            f"{self._format_etf_exposure_lines(payload.get('etf_exposures'))}\n\n"
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
            f"Macro Intelligence\n{self._format_macro_intelligence_lines(payload.get('macro_intelligence'))}\n\n"
            f"Upcoming Macro Events\n{self._format_macro_event_calendar_lines(payload.get('macro_event_calendar'))}\n\n"
            f"Macro Surprise Engine\n{self._format_macro_surprise_lines(payload.get('macro_surprise_signals'))}\n\n"
            f"Macro Market Reaction\n{self._format_macro_market_reaction_lines(payload.get('macro_market_reactions'))}\n\n"
            f"Post-Event Playbooks\n{self._format_macro_playbook_lines(payload.get('macro_post_event_playbooks'))}\n\n"
            f"Thesis Memory\n{self._format_thesis_memory_lines(payload.get('thesis_memory'))}\n\n"
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
            f"Management Commentary\n{self._format_management_commentary_lines(payload.get('management_commentary'))}\n\n"
            f"Microstructure\n{self._format_microstructure_lines(payload.get('microstructure'))}\n\n"
            f"Company Intelligence\n{self._format_company_intelligence_lines(payload.get('company_intelligence'))}\n\n"
            f"ETF Exposure\n{self._format_etf_exposure_lines(payload.get('etf_exposures'))}\n\n"
            f"Earnings ที่ต้องจับตา\n{self._format_earnings_lines(payload.get('earnings_calendar'))}\n\n"
            f"Pre-Earnings Setup Ranking\n{self._format_earnings_setup_lines(payload.get('earnings_setups'))}\n\n"
            f"Post-Earnings Read-Through\n{self._format_earnings_surprise_lines(payload.get('earnings_surprises'))}\n\n"
            f"ข่าวสำคัญ\n{self._format_fallback_news(payload.get('news_headlines')) or '- ไม่มี headline เด่น'}"
            f"\n\nPolicy Feed\n{self._format_policy_feed_lines(payload.get('policy_events'))}\n\n"
            f"Ownership / 13F / 13D-G\n{self._format_ownership_intelligence_lines(payload.get('ownership_intelligence'))}\n\n"
            f"Options Order Flow\n{self._format_order_flow_lines(payload.get('order_flow'))}"
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

    def _serialize_macro_intelligence(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        metrics = payload.get("metrics") if isinstance(payload, Mapping) else None
        return {
            "headline": str((payload or {}).get("headline") or "").strip() or "macro backdrop unavailable",
            "signals": [str(item).strip() for item in ((payload or {}).get("signals") or []) if str(item).strip()],
            "highlights": [str(item).strip() for item in ((payload or {}).get("highlights") or []) if str(item).strip()],
            "sources_used": [str(item).strip() for item in ((payload or {}).get("sources_used") or []) if str(item).strip()],
            "metrics": {
                str(key): self._round_optional(value)
                for key, value in (metrics.items() if isinstance(metrics, Mapping) else [])
                if isinstance(key, str)
            },
            "revisions": dict((payload or {}).get("revisions") or {}) if isinstance((payload or {}).get("revisions"), Mapping) else {},
            "positioning": dict((payload or {}).get("positioning") or {}) if isinstance((payload or {}).get("positioning"), Mapping) else {},
            "qualitative": dict((payload or {}).get("qualitative") or {}) if isinstance((payload or {}).get("qualitative"), Mapping) else {},
            "short_flow": dict((payload or {}).get("short_flow") or {}) if isinstance((payload or {}).get("short_flow"), Mapping) else {},
            "fedwatch": dict((payload or {}).get("fedwatch") or {}) if isinstance((payload or {}).get("fedwatch"), Mapping) else {},
            "structured_macro": dict((payload or {}).get("structured_macro") or {}) if isinstance((payload or {}).get("structured_macro"), Mapping) else {},
            "ex_us_macro": dict((payload or {}).get("ex_us_macro") or {}) if isinstance((payload or {}).get("ex_us_macro"), Mapping) else {},
            "global_event": dict((payload or {}).get("global_event") or {}) if isinstance((payload or {}).get("global_event"), Mapping) else {},
        }

    def _serialize_macro_event(self, item: MacroEvent) -> dict[str, Any]:
        return {
            "event_key": item.event_key,
            "event_name": item.event_name,
            "category": item.category,
            "source": item.source,
            "scheduled_at": item.scheduled_at.isoformat(),
            "importance": item.importance,
            "status": item.status,
            "source_url": item.source_url,
            "country": item.country,
            "previous_value": self._round_optional(item.previous_value),
            "forecast_value": self._round_optional(item.forecast_value),
            "actual_value": self._round_optional(item.actual_value),
        }

    def _serialize_macro_surprise(self, item: MacroSurpriseSignal) -> dict[str, Any]:
        return {
            "event_key": item.event_key,
            "event_name": item.event_name,
            "category": item.category,
            "source": item.source,
            "released_at": item.released_at.isoformat() if item.released_at else None,
            "next_event_at": item.next_event_at.isoformat() if item.next_event_at else None,
            "actual_value": self._round_optional(item.actual_value),
            "expected_value": self._round_optional(item.expected_value),
            "surprise_value": self._round_optional(item.surprise_value),
            "surprise_direction": item.surprise_direction,
            "surprise_label": item.surprise_label,
            "market_bias": item.market_bias,
            "rationale": list(item.rationale),
            "detail_url": item.detail_url,
            "baseline_expected_value": self._round_optional(item.baseline_expected_value),
            "baseline_surprise_value": self._round_optional(item.baseline_surprise_value),
            "baseline_surprise_label": item.baseline_surprise_label,
            "consensus_expected_value": self._round_optional(item.consensus_expected_value),
            "consensus_te_expected_value": self._round_optional(item.consensus_te_expected_value),
            "consensus_surprise_value": self._round_optional(item.consensus_surprise_value),
            "consensus_surprise_label": item.consensus_surprise_label,
        }

    def _serialize_macro_market_reaction(self, item: MacroMarketReaction) -> dict[str, Any]:
        return {
            "event_key": item.event_key,
            "event_name": item.event_name,
            "released_at": item.released_at.isoformat() if item.released_at else None,
            "market_bias": item.market_bias,
            "confirmation_label": item.confirmation_label,
            "confirmation_score_5m": self._round_optional(item.confirmation_score_5m),
            "confirmation_score_1h": self._round_optional(item.confirmation_score_1h),
            "rationale": list(item.rationale),
            "reactions": [
                {
                    "ticker": reaction.ticker,
                    "label": reaction.label,
                    "move_5m_pct": self._round_optional(reaction.move_5m_pct),
                    "move_1h_pct": self._round_optional(reaction.move_1h_pct),
                    "expected_direction": reaction.expected_direction,
                    "confirmed_5m": reaction.confirmed_5m,
                    "confirmed_1h": reaction.confirmed_1h,
                }
                for reaction in item.reactions
            ],
        }

    def _serialize_macro_playbook(self, item: MacroPostEventPlaybook) -> dict[str, Any]:
        return {
            "playbook_key": item.playbook_key,
            "title": item.title,
            "trigger": item.trigger,
            "action": item.action,
            "risk_watch": item.risk_watch,
            "confidence": item.confidence,
            "confidence_score": item.confidence_score,
            "learning_note": item.learning_note,
            "event_key": item.event_key,
        }

    @staticmethod
    def _serialize_regime_playbook(item: RegimeSpecificPlaybook) -> dict[str, Any]:
        return {
            "playbook_key": item.playbook_key,
            "regime": item.regime,
            "title": item.title,
            "action": item.action,
            "risk_watch": item.risk_watch,
            "conviction": item.conviction,
            "sizing_bias": item.sizing_bias,
            "ttl_bias": item.ttl_bias,
            "rationale": list(item.rationale),
        }

    def _serialize_company_intelligence(self, item: CompanyIntelligence) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "company_name": item.company_name,
            "cik": item.cik,
            "latest_10k_filed_at": item.latest_10k_filed_at.isoformat() if item.latest_10k_filed_at else None,
            "latest_10q_filed_at": item.latest_10q_filed_at.isoformat() if item.latest_10q_filed_at else None,
            "latest_8k_filed_at": item.latest_8k_filed_at.isoformat() if item.latest_8k_filed_at else None,
            "revenue_latest": self._round_optional(item.revenue_latest),
            "revenue_yoy_pct": self._round_optional(item.revenue_yoy_pct),
            "operating_cash_flow_latest": self._round_optional(item.operating_cash_flow_latest),
            "free_cash_flow_latest": self._round_optional(item.free_cash_flow_latest),
            "debt_latest": self._round_optional(item.debt_latest),
            "debt_delta_pct": self._round_optional(item.debt_delta_pct),
            "share_dilution_yoy_pct": self._round_optional(item.share_dilution_yoy_pct),
            "one_off_signal": item.one_off_signal,
            "guidance_signal": item.guidance_signal,
            "insider_signal": item.insider_signal,
            "sentiment_signal": item.sentiment_signal,
            "earnings_expectation_signal": item.earnings_expectation_signal,
            "analyst_rating_signal": item.analyst_rating_signal,
            "analyst_buy_count": item.analyst_buy_count,
            "analyst_hold_count": item.analyst_hold_count,
            "analyst_sell_count": item.analyst_sell_count,
            "analyst_upside_pct": self._round_optional(item.analyst_upside_pct),
            "insider_net_shares": self._round_optional(item.insider_net_shares),
            "insider_net_value": self._round_optional(item.insider_net_value),
            "insider_transaction_count": item.insider_transaction_count,
            "insider_last_filed_at": item.insider_last_filed_at.isoformat() if item.insider_last_filed_at else None,
            "corporate_action_signal": item.corporate_action_signal,
            "recent_corporate_actions": [
                self._serialize_corporate_action_event(event)
                for event in item.recent_corporate_actions[:4]
            ],
            "filing_highlights": list(item.filing_highlights),
            "recent_filings": [
                {
                    "form": filing.form,
                    "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
                    "report_date": filing.report_date.isoformat() if filing.report_date else None,
                    "primary_document_url": filing.primary_document_url,
                }
                for filing in item.recent_filings[:4]
            ],
        }

    def _serialize_policy_feed_event(self, item: PolicyFeedEvent) -> dict[str, Any]:
        return {
            "central_bank": item.central_bank,
            "title": item.title,
            "category": item.category,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "url": item.url,
            "summary": item.summary,
            "tone_signal": item.tone_signal,
        }

    def _serialize_ownership_intelligence(self, item: OwnershipIntelligence) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "company_name": item.company_name,
            "ownership_signal": item.ownership_signal,
            "highlights": list(item.highlights),
            "beneficial_owners": [
                {
                    "filer_name": owner.filer_name,
                    "form": owner.form,
                    "filed_at": owner.filed_at.isoformat() if owner.filed_at else None,
                    "stake_pct": self._round_optional(owner.stake_pct),
                    "shares": self._round_optional(owner.shares),
                    "source_url": owner.source_url,
                }
                for owner in item.beneficial_owners[:4]
            ],
            "institutional_holders": [
                {
                    "manager_name": holder.manager_name,
                    "filed_at": holder.filed_at.isoformat() if holder.filed_at else None,
                    "matched_issuer": holder.matched_issuer,
                    "value_usd_thousands": self._round_optional(holder.value_usd_thousands),
                    "shares": self._round_optional(holder.shares),
                    "source_url": holder.source_url,
                }
                for holder in item.institutional_holders[:4]
            ],
        }

    def _serialize_order_flow_snapshot(self, item: OrderFlowSnapshot) -> dict[str, Any]:
        return {
            "symbol": item.symbol,
            "bullish_premium": self._round_optional(item.bullish_premium),
            "bearish_premium": self._round_optional(item.bearish_premium),
            "call_put_ratio": self._round_optional(item.call_put_ratio),
            "unusual_count": item.unusual_count,
            "sweep_count": item.sweep_count,
            "opening_flow_ratio": self._round_optional(item.opening_flow_ratio),
            "sentiment": item.sentiment,
            "captured_at": item.captured_at.isoformat(),
            "source": item.source,
        }

    @staticmethod
    def _serialize_corporate_action_event(item: Any) -> dict[str, Any]:
        return {
            "ticker": getattr(item, "ticker", None),
            "action_type": getattr(item, "action_type", None),
            "ex_date": item.ex_date.isoformat() if getattr(item, "ex_date", None) else None,
            "record_date": item.record_date.isoformat() if getattr(item, "record_date", None) else None,
            "payable_date": item.payable_date.isoformat() if getattr(item, "payable_date", None) else None,
            "cash_amount": RecommendationService._round_optional(RecommendationService._as_float(getattr(item, "cash_amount", None))),
            "ratio": RecommendationService._round_optional(RecommendationService._as_float(getattr(item, "ratio", None))),
            "source": getattr(item, "source", None),
        }

    def _serialize_etf_exposure_profile(self, item: ETFExposureProfile) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "fund_family": item.fund_family,
            "category": item.category,
            "total_assets": self._round_optional(item.total_assets),
            "fund_flow_1m_pct": self._round_optional(item.fund_flow_1m_pct),
            "top_holdings": [
                {"name": name, "weight_pct": self._round_optional(weight)}
                for name, weight in item.top_holdings[:5]
            ],
            "sector_exposures": [
                {"name": name, "weight_pct": self._round_optional(weight)}
                for name, weight in item.sector_exposures[:6]
            ],
            "country_exposures": [
                {"name": name, "weight_pct": self._round_optional(weight)}
                for name, weight in item.country_exposures[:6]
            ],
            "concentration_score": self._round_optional(item.concentration_score),
            "exposure_signal": item.exposure_signal,
            "source": item.source,
        }

    def _serialize_transcript_insight(self, item: TranscriptInsight) -> dict[str, Any]:
        return {
            "ticker": item.ticker,
            "quarter": item.quarter,
            "year": item.year,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "source": item.source,
            "tone": item.tone,
            "guidance_signal": item.guidance_signal,
            "confidence": self._round_optional(item.confidence),
            "summary": item.summary,
            "highlights": list(item.highlights),
        }

    def _serialize_microstructure_snapshot(self, item: MicrostructureSnapshot) -> dict[str, Any]:
        return {
            "symbol": item.symbol,
            "dataset": item.dataset,
            "schema": item.schema,
            "best_bid": self._round_optional(item.best_bid),
            "best_ask": self._round_optional(item.best_ask),
            "bid_size": self._round_optional(item.bid_size),
            "ask_size": self._round_optional(item.ask_size),
            "spread_bps": self._round_optional(item.spread_bps),
            "imbalance": self._round_optional(item.imbalance),
            "last_price": self._round_optional(item.last_price),
            "last_size": self._round_optional(item.last_size),
            "sample_count": item.sample_count,
            "captured_at": item.captured_at.isoformat(),
        }

    @staticmethod
    def _serialize_thesis_memory_item(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "thesis_key": str(item.get("thesis_key") or "").strip(),
            "conversation_key": str(item.get("conversation_key") or "").strip() or None,
            "thesis_text": str(item.get("thesis_text") or "").strip(),
            "source_kind": str(item.get("source_kind") or "").strip() or "recommendation",
            "query_text": str(item.get("query_text") or "").strip() or None,
            "tags": [str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()],
            "confidence_score": RecommendationService._round_optional(RecommendationService._as_float(item.get("confidence_score"))),
            "similarity": RecommendationService._round_optional(RecommendationService._as_float(item.get("similarity"))),
            "created_at": str(item.get("created_at") or "").strip() or None,
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
        coverage = self._assess_stock_candidate_coverage(candidate)
        position_plan = self._determine_stock_pick_position_plan(
            source_kind="manual_pick",
            confidence_score=confidence.score,
            stance=candidate.stance,
            candidate=candidate,
        )
        return {
            "ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "sector": candidate.sector,
            "benchmark": candidate.benchmark,
            "benchmark_ticker": self._resolve_benchmark_ticker(candidate.benchmark),
            "peer_benchmark_ticker": candidate.peer_benchmark_ticker or self._resolve_sector_benchmark_ticker(candidate.sector),
            "price": candidate.price,
            "score": candidate.composite_score,
            "macro_overlay_score": candidate.macro_overlay_score,
            "universe_quality_score": candidate.universe_quality_score,
            "factor_risk_score": candidate.factor_risk_score,
            "regime_fit_score": candidate.regime_fit_score,
            "relative_strength_score": candidate.relative_strength_score,
            "peer_relative_score": candidate.peer_relative_score,
            "beta_1m": candidate.beta_1m,
            "liquidity_tier": candidate.liquidity_tier,
            "market_cap_bucket": candidate.market_cap_bucket,
            "confidence_score": confidence.score,
            "confidence_label": confidence.label,
            "coverage_score": coverage["score"],
            "coverage_label": coverage["label"],
            "coverage_summary": coverage["summary"],
            "stance": candidate.stance,
            "trend_direction": candidate.trend_direction,
            "rationale": list(candidate.rationale),
            "macro_drivers": list(candidate.macro_drivers),
            "universe_flags": list(candidate.universe_flags),
            "suggested_position_size_pct": position_plan["position_size_pct"],
            "signal_ttl_minutes": position_plan["ttl_minutes"],
            "execution_realism": position_plan.get("execution_realism"),
            "forward_pe": candidate.forward_pe,
            "revenue_growth": candidate.revenue_growth,
            "earnings_growth": candidate.earnings_growth,
        }

    @staticmethod
    def _resolve_benchmark_ticker(benchmark: str | None) -> str:
        normalized = str(benchmark or "").strip().casefold()
        return {
            "sp500": "SPY",
            "sp500_index": "SPY",
            "nasdaq100": "QQQ",
            "nasdaq_index": "QQQ",
            "watchlist": "SPY",
            "custom": "SPY",
        }.get(normalized, "SPY")

    @staticmethod
    def _resolve_sector_benchmark_ticker(sector: str | None) -> str:
        normalized = str(sector or "").strip().casefold()
        return {
            "technology": "XLK",
            "financials": "XLF",
            "energy": "XLE",
            "consumer discretionary": "XLY",
            "consumer staples": "XLP",
            "healthcare": "XLV",
            "industrials": "XLI",
            "materials": "XLB",
            "utilities": "XLU",
            "communication services": "XLC",
            "real estate": "XLRE",
        }.get(normalized, "SPY")

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
        coverage = self._assess_asset_snapshot_coverage(quote=quote, trend=trend)
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
            "coverage_score": coverage["score"],
            "coverage_label": coverage["label"],
            "coverage_summary": coverage["summary"],
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
        ordered_keys = [
            "vix",
            "tnx",
            "cpi_yoy",
            "core_cpi_yoy",
            "ppi_yoy",
            "yield_spread_10y_2y",
            "high_yield_spread",
            "unemployment_rate",
            "payrolls_mom_k",
            "avg_interest_rate_pct",
            "operating_cash_balance_b",
            "wti_usd",
            "brent_usd",
            "gasoline_usd_gal",
        ]
        lines = [
            f"- {MACRO_CONTEXT_LABELS.get(key, key)}: {macro_context.get(key, 'N/A')}"
            for key in ordered_keys
            if key in macro_context
        ]
        return "\n".join(lines) or "- ไม่มีข้อมูลมหภาค"

    @staticmethod
    def _format_macro_one_line(macro_context: Any) -> str:
        if not isinstance(macro_context, Mapping):
            return "มหภาค: ไม่มีข้อมูล"
        focus_pairs = [
            ("vix", "VIX"),
            ("tnx", "US10Y"),
            ("cpi_yoy", "CPI"),
            ("yield_spread_10y_2y", "2s10s"),
            ("unemployment_rate", "Unemp"),
        ]
        parts = [f"{label} {macro_context.get(key, 'N/A')}" for key, label in focus_pairs if key in macro_context]
        return "มหภาค: " + " | ".join(parts or ["ไม่มีข้อมูล"])

    @staticmethod
    def _format_macro_intelligence_lines(macro_intelligence: Any) -> str:
        if not isinstance(macro_intelligence, Mapping):
            return "- ไม่มี macro intelligence เพิ่มเติม"
        headline = str(macro_intelligence.get("headline") or "").strip() or "macro backdrop unavailable"
        highlights = ", ".join(str(item) for item in (macro_intelligence.get("highlights") or []) if item)
        signals = ", ".join(str(item) for item in (macro_intelligence.get("signals") or []) if item)
        sources = ", ".join(str(item) for item in (macro_intelligence.get("sources_used") or []) if item)
        revisions = macro_intelligence.get("revisions") if isinstance(macro_intelligence.get("revisions"), Mapping) else {}
        positioning = macro_intelligence.get("positioning") if isinstance(macro_intelligence.get("positioning"), Mapping) else {}
        qualitative = macro_intelligence.get("qualitative") if isinstance(macro_intelligence.get("qualitative"), Mapping) else {}
        short_flow = macro_intelligence.get("short_flow") if isinstance(macro_intelligence.get("short_flow"), Mapping) else {}
        fedwatch = macro_intelligence.get("fedwatch") if isinstance(macro_intelligence.get("fedwatch"), Mapping) else {}
        structured_macro = macro_intelligence.get("structured_macro") if isinstance(macro_intelligence.get("structured_macro"), Mapping) else {}
        ex_us_macro = macro_intelligence.get("ex_us_macro") if isinstance(macro_intelligence.get("ex_us_macro"), Mapping) else {}
        global_event = macro_intelligence.get("global_event") if isinstance(macro_intelligence.get("global_event"), Mapping) else {}
        lines = [f"- Headline: {headline}"]
        if highlights:
            lines.append(f"- Highlights: {highlights}")
        if signals:
            lines.append(f"- Signals: {signals}")
        if sources:
            lines.append(f"- Sources: {sources}")
        if revisions:
            lines.append(f"- Revisions: {', '.join(f'{key}={value}' for key, value in revisions.items())}")
        if positioning:
            lines.append(f"- Positioning: {', '.join(f'{key}={value}' for key, value in positioning.items())}")
        if qualitative:
            lines.append(f"- Fed Qualitative: {', '.join(f'{key}={value}' for key, value in qualitative.items())}")
        if short_flow:
            lines.append(f"- Short Flow: {', '.join(f'{key}={value}' for key, value in short_flow.items())}")
        if fedwatch:
            lines.append(f"- FedWatch: {', '.join(f'{key}={value}' for key, value in fedwatch.items())}")
        if structured_macro:
            dataset_text = ", ".join(
                f"{dataset}={details.get('value')}"
                for dataset, details in list(structured_macro.items())[:4]
                if isinstance(details, Mapping)
            )
            if dataset_text:
                lines.append(f"- Structured Macro: {dataset_text}")
        if ex_us_macro:
            ex_us_bits = []
            for key, value in ex_us_macro.items():
                if key in {"sources_used", "highlights"}:
                    continue
                ex_us_bits.append(f"{key}={value}")
                if len(ex_us_bits) >= 4:
                    break
            ex_us_highlights = ex_us_macro.get("highlights")
            if isinstance(ex_us_highlights, list) and ex_us_highlights:
                ex_us_bits.append(f"highlights={'; '.join(str(item) for item in ex_us_highlights[:2])}")
            if ex_us_bits:
                lines.append(f"- Ex-US Macro: {', '.join(ex_us_bits)}")
        if global_event:
            event_bits = []
            if global_event.get("top_theme"):
                event_bits.append(f"theme={global_event.get('top_theme')}")
            if global_event.get("top_location"):
                event_bits.append(f"location={global_event.get('top_location')}")
            if global_event.get("risk_score") is not None:
                event_bits.append(f"risk={global_event.get('risk_score')}")
            headlines = global_event.get("headlines")
            if isinstance(headlines, list) and headlines:
                event_bits.append(f"headlines={'; '.join(str(item) for item in headlines[:2])}")
            if event_bits:
                lines.append(f"- Global Event: {', '.join(event_bits)}")
        return "\n".join(lines)

    @staticmethod
    def _format_macro_event_calendar_lines(macro_event_calendar: Any, *, limit: int = 4) -> str:
        if not isinstance(macro_event_calendar, list) or not macro_event_calendar:
            return "- ไม่มี macro event calendar"
        lines: list[str] = []
        for item in macro_event_calendar[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            event_name = str(item.get("event_name") or item.get("event_key") or "-").strip()
            category = str(item.get("category") or "-").strip()
            scheduled_at = str(item.get("scheduled_at") or "-").strip()
            source = str(item.get("source") or "-").strip()
            country = str(item.get("country") or "").strip()
            forecast = item.get("forecast_value")
            previous = item.get("previous_value")
            actual = item.get("actual_value")
            extras: list[str] = [category]
            if country:
                extras.append(country)
            if forecast is not None:
                extras.append(f"fcst={forecast}")
            if previous is not None:
                extras.append(f"prev={previous}")
            if actual is not None:
                extras.append(f"actual={actual}")
            extras.append(f"source={source}")
            lines.append(f"- {event_name}: {scheduled_at} | " + " | ".join(extras))
        return "\n".join(lines) or "- ไม่มี macro event calendar"

    @staticmethod
    def _format_macro_surprise_lines(macro_surprise_signals: Any, *, limit: int = 4) -> str:
        if not isinstance(macro_surprise_signals, list) or not macro_surprise_signals:
            return "- ไม่มี macro surprise signals"
        lines: list[str] = []
        for item in macro_surprise_signals[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            event_name = str(item.get("event_name") or item.get("event_key") or "-").strip()
            label = str(item.get("surprise_label") or "n/a").strip()
            actual = item.get("actual_value")
            baseline_label = str(item.get("baseline_surprise_label") or "n/a").strip()
            baseline_expected = item.get("baseline_expected_value")
            consensus_label = str(item.get("consensus_surprise_label") or "n/a").strip()
            consensus_expected = item.get("consensus_expected_value")
            bias = str(item.get("market_bias") or "n/a").strip()
            lines.append(
                f"- {event_name}: summary={label} | actual={actual} | baseline={baseline_label} vs {baseline_expected} | "
                f"consensus={consensus_label} vs {consensus_expected} | bias={bias}"
            )
        return "\n".join(lines) or "- ไม่มี macro surprise signals"

    @staticmethod
    def _format_macro_market_reaction_lines(macro_market_reactions: Any, *, limit: int = 4) -> str:
        if not isinstance(macro_market_reactions, list) or not macro_market_reactions:
            return "- ไม่มี macro market reactions"
        lines: list[str] = []
        for item in macro_market_reactions[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            event_name = str(item.get("event_name") or item.get("event_key") or "-").strip()
            label = str(item.get("confirmation_label") or "n/a").strip()
            score_5m = item.get("confirmation_score_5m")
            score_1h = item.get("confirmation_score_1h")
            bias = str(item.get("market_bias") or "n/a").strip()
            lines.append(f"- {event_name}: {label} | 5m={score_5m} | 1h={score_1h} | bias={bias}")
        return "\n".join(lines) or "- ไม่มี macro market reactions"

    @staticmethod
    def _format_macro_playbook_lines(playbooks: Any, *, limit: int = 4) -> str:
        if not isinstance(playbooks, list) or not playbooks:
            return "- ไม่มี post-event playbook เด่น"
        lines: list[str] = []
        for item in playbooks[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or item.get("playbook_key") or "-").strip()
            action = str(item.get("action") or "").strip()
            risk_watch = str(item.get("risk_watch") or "").strip()
            confidence = str(item.get("confidence") or "").strip()
            confidence_score = item.get("confidence_score")
            learning_note = str(item.get("learning_note") or "").strip()
            confidence_text = (
                f" | confidence {confidence} ({confidence_score:.2f})"
                if isinstance(confidence_score, (float, int)) and confidence
                else f" | confidence {confidence}"
                if confidence
                else ""
            )
            learning_text = f" | {learning_note}" if learning_note else ""
            lines.append(
                f"- {title}: {action}"
                + (f" | watch {risk_watch}" if risk_watch else "")
                + confidence_text
                + learning_text
            )
        return "\n".join(lines) or "- ไม่มี post-event playbook เด่น"

    @staticmethod
    def _format_regime_playbook_lines(playbooks: Any, *, limit: int = 4) -> str:
        if not isinstance(playbooks, list) or not playbooks:
            return "- ไม่มี regime playbook เด่น"
        lines: list[str] = []
        for item in playbooks[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or item.get("playbook_key") or "-").strip()
            action = str(item.get("action") or "").strip()
            conviction = str(item.get("conviction") or "").strip()
            sizing_bias = str(item.get("sizing_bias") or "").strip()
            ttl_bias = str(item.get("ttl_bias") or "").strip()
            risk_watch = str(item.get("risk_watch") or "").strip()
            lines.append(
                f"- {title}: {action}"
                + (f" | conviction={conviction}" if conviction else "")
                + (f" | sizing={sizing_bias}" if sizing_bias else "")
                + (f" | ttl={ttl_bias}" if ttl_bias else "")
                + (f" | watch {risk_watch}" if risk_watch else "")
            )
        return "\n".join(lines) or "- ไม่มี regime playbook เด่น"

    @staticmethod
    def _format_thesis_memory_lines(thesis_memory: Any, *, limit: int = 4) -> str:
        if not isinstance(thesis_memory, list) or not thesis_memory:
            return "- ไม่มี thesis memory ที่เกี่ยวข้อง"
        lines: list[str] = []
        for item in thesis_memory[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            thesis_text = str(item.get("thesis_text") or "").strip()
            if not thesis_text:
                continue
            source_kind = str(item.get("source_kind") or "recommendation").strip()
            similarity = item.get("similarity")
            lines.append(
                f"- {thesis_text}"
                + (f" | source={source_kind}" if source_kind else "")
                + (f" | similarity={similarity}" if similarity is not None else "")
            )
        return "\n".join(lines) or "- ไม่มี thesis memory ที่เกี่ยวข้อง"

    @staticmethod
    def _format_policy_feed_lines(policy_items: Any, *, limit: int = 4) -> str:
        if not isinstance(policy_items, list) or not policy_items:
            return "- ไม่มี policy feed ล่าสุด"
        lines: list[str] = []
        for item in policy_items[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "-").strip()
            central_bank = str(item.get("central_bank") or "-").strip()
            category = str(item.get("category") or "-").strip()
            tone = str(item.get("tone_signal") or "-").strip()
            published_at = str(item.get("published_at") or "-").strip()
            lines.append(f"- {central_bank}/{category}: {title} | tone={tone} | at={published_at}")
        return "\n".join(lines) or "- ไม่มี policy feed ล่าสุด"

    @staticmethod
    def _format_ownership_intelligence_lines(ownership_items: Any, *, limit: int = 4) -> str:
        if isinstance(ownership_items, Mapping):
            values = ownership_items.values()
        elif isinstance(ownership_items, list):
            values = ownership_items
        else:
            return "- ไม่มี ownership / 13F / 13D-G signal"
        lines: list[str] = []
        for item in list(values)[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "-").strip()
            signal = str(item.get("ownership_signal") or "-").strip()
            highlights = ", ".join(str(part) for part in (item.get("highlights") or []) if part)
            lines.append(f"- {ticker}: signal={signal}" + (f" | {highlights}" if highlights else ""))
        return "\n".join(lines) or "- ไม่มี ownership / 13F / 13D-G signal"

    @staticmethod
    def _format_order_flow_lines(order_flow_items: Any, *, limit: int = 4) -> str:
        if isinstance(order_flow_items, Mapping):
            values = order_flow_items.values()
        elif isinstance(order_flow_items, list):
            values = order_flow_items
        else:
            return "- ไม่มี options order-flow ล่าสุด"
        lines: list[str] = []
        for item in list(values)[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("symbol") or item.get("ticker") or "-").strip()
            sentiment = str(item.get("sentiment") or "-").strip()
            call_put_ratio = item.get("call_put_ratio")
            unusual_count = item.get("unusual_count")
            sweep_count = item.get("sweep_count")
            lines.append(
                f"- {ticker}: sentiment={sentiment} | call_put={call_put_ratio if call_put_ratio is not None else '-'} | unusual={unusual_count if unusual_count is not None else '-'} | sweeps={sweep_count if sweep_count is not None else '-'}"
            )
        return "\n".join(lines) or "- ไม่มี options order-flow ล่าสุด"

    @staticmethod
    def _format_company_intelligence_lines(company_items: Any) -> str:
        if not isinstance(company_items, list) or not company_items:
            return "- ไม่มี company / filing intelligence"
        lines: list[str] = []
        for item in company_items[:4]:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "-").strip()
            highlights = ", ".join(str(part) for part in (item.get("filing_highlights") or []) if part)
            guidance = str(item.get("guidance_signal") or "n/a").strip()
            insider = str(item.get("insider_signal") or "n/a").strip()
            sentiment = str(item.get("sentiment_signal") or "n/a").strip()
            analyst = str(item.get("analyst_rating_signal") or "n/a").strip()
            upside = item.get("analyst_upside_pct")
            corporate_action = str(item.get("corporate_action_signal") or "n/a").strip()
            insider_value = item.get("insider_net_value")
            extras: list[str] = [
                f"guidance={guidance}",
                f"insider={insider}",
                f"sentiment={sentiment}",
                f"analyst={analyst}",
            ]
            if upside is not None:
                extras.append(f"upside={upside}%")
            if insider_value is not None:
                extras.append(f"insider_value={insider_value}")
            if corporate_action != "n/a":
                extras.append(f"corp_action={corporate_action}")
            lines.append(
                f"- {ticker}: " + " | ".join(extras)
                + (f" | {highlights}" if highlights else "")
            )
        return "\n".join(lines) or "- ไม่มี company / filing intelligence"

    @staticmethod
    def _format_etf_exposure_lines(etf_exposures: Any, *, limit: int = 4) -> str:
        if not isinstance(etf_exposures, list) or not etf_exposures:
            return "- ไม่มี ETF exposure intelligence"
        lines: list[str] = []
        for item in etf_exposures[: max(1, limit)]:
            if not isinstance(item, Mapping):
                continue
            ticker = str(item.get("ticker") or "-").strip()
            signal = str(item.get("exposure_signal") or "n/a").strip()
            concentration = item.get("concentration_score")
            top_holding = None
            sector = None
            holdings = item.get("top_holdings")
            if isinstance(holdings, list) and holdings:
                first = holdings[0]
                if isinstance(first, Mapping):
                    top_holding = f"{first.get('name')} {first.get('weight_pct')}%"
            sectors = item.get("sector_exposures")
            if isinstance(sectors, list) and sectors:
                first_sector = sectors[0]
                if isinstance(first_sector, Mapping):
                    sector = f"{first_sector.get('name')} {first_sector.get('weight_pct')}%"
            parts = [f"signal={signal}"]
            if concentration is not None:
                parts.append(f"top5={concentration}%")
            if top_holding:
                parts.append(f"top={top_holding}")
            if sector:
                parts.append(f"sector={sector}")
            lines.append(f"- {ticker}: " + " | ".join(parts))
        return "\n".join(lines) or "- ไม่มี ETF exposure intelligence"

    @staticmethod
    def summarize_source_coverage(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {"used_sources": [], "flags": {}, "counts": {}}
        used_sources: list[str] = []
        flags: dict[str, bool] = {}
        counts: dict[str, int] = {}

        macro_intelligence = payload.get("macro_intelligence")
        macro_sources = [
            str(item).strip()
            for item in ((macro_intelligence or {}).get("sources_used") or [])
            if str(item).strip()
        ] if isinstance(macro_intelligence, Mapping) else []
        if macro_sources:
            used_sources.extend(macro_sources)
            flags["macro_intelligence"] = True
            counts["macro_sources"] = len(macro_sources)

        macro_event_calendar = payload.get("macro_event_calendar")
        if isinstance(macro_event_calendar, list) and macro_event_calendar:
            used_sources.append("macro_event_calendar")
            event_sources = {
                str(item.get("source") or "").strip()
                for item in macro_event_calendar
                if isinstance(item, Mapping) and str(item.get("source") or "").strip()
            }
            used_sources.extend(sorted(event_sources))
            flags["macro_event_calendar"] = True
            counts["macro_events"] = len(macro_event_calendar)

        macro_surprise_signals = payload.get("macro_surprise_signals")
        if isinstance(macro_surprise_signals, list) and macro_surprise_signals:
            used_sources.append("macro_surprise_engine")
            used_sources.extend(
                sorted(
                    {
                        str(item.get("source") or "").strip()
                        for item in macro_surprise_signals
                        if isinstance(item, Mapping) and str(item.get("source") or "").strip()
                    }
                )
            )
            flags["macro_surprise_engine"] = True
            counts["macro_surprises"] = len(macro_surprise_signals)

        macro_market_reactions = payload.get("macro_market_reactions")
        if isinstance(macro_market_reactions, list) and macro_market_reactions:
            used_sources.append("macro_market_reaction")
            used_sources.extend(
                sorted(
                    {
                        str(reaction.get("label") or "").strip().casefold()
                        for item in macro_market_reactions
                        if isinstance(item, Mapping)
                        for reaction in (item.get("reactions") or [])
                        if isinstance(reaction, Mapping) and str(reaction.get("label") or "").strip()
                    }
                )
            )
            flags["macro_market_reaction"] = True
            counts["macro_market_reactions"] = len(macro_market_reactions)

        macro_playbooks = payload.get("macro_post_event_playbooks")
        if isinstance(macro_playbooks, list) and macro_playbooks:
            used_sources.append("macro_post_event_playbooks")
            flags["macro_post_event_playbooks"] = True
            counts["macro_playbooks"] = len(macro_playbooks)

        thesis_memory = payload.get("thesis_memory")
        if isinstance(thesis_memory, list) and thesis_memory:
            used_sources.append("thesis_memory")
            flags["thesis_memory"] = True
            counts["thesis_memory_items"] = len(thesis_memory)

        news_headlines = payload.get("news_headlines")
        if isinstance(news_headlines, list) and news_headlines:
            used_sources.append("news")
            flags["news"] = True
            counts["news_items"] = len(news_headlines)

        research_highlights = payload.get("research_highlights")
        if isinstance(research_highlights, list) and research_highlights:
            used_sources.append("research")
            flags["research"] = True
            counts["research_items"] = len(research_highlights)

        company_intelligence = payload.get("company_intelligence")
        if isinstance(company_intelligence, list) and company_intelligence:
            used_sources.extend(["company_intelligence", "sec"])
            flags["company_intelligence"] = True
            counts["company_intelligence_items"] = len(company_intelligence)

        policy_events = payload.get("policy_events")
        if isinstance(policy_events, list) and policy_events:
            used_sources.extend(["policy_feed", "federal_reserve", "ecb"])
            flags["policy_feed"] = True
            counts["policy_events"] = len(policy_events)

        ownership_intelligence = payload.get("ownership_intelligence")
        if isinstance(ownership_intelligence, list) and ownership_intelligence:
            used_sources.extend(["ownership_intelligence", "sec_13f_13d_13g"])
            flags["ownership_intelligence"] = True
            counts["ownership_items"] = len(ownership_intelligence)

        order_flow = payload.get("order_flow")
        if isinstance(order_flow, list) and order_flow:
            used_sources.extend(["order_flow", "cboe_trade_alert"])
            flags["order_flow"] = True
            counts["order_flow_items"] = len(order_flow)

        etf_exposures = payload.get("etf_exposures")
        if isinstance(etf_exposures, list) and etf_exposures:
            used_sources.extend(["etf_exposures", "yfinance"])
            flags["etf_exposures"] = True
            counts["etf_exposure_items"] = len(etf_exposures)

        management_commentary = payload.get("management_commentary")
        if isinstance(management_commentary, list) and management_commentary:
            used_sources.extend(["earnings_transcripts", "financial_modeling_prep"])
            flags["management_commentary"] = True
            counts["management_commentary_items"] = len(management_commentary)

        microstructure = payload.get("microstructure")
        if isinstance(microstructure, list) and microstructure:
            used_sources.extend(["microstructure", "databento"])
            flags["microstructure"] = True
            counts["microstructure_items"] = len(microstructure)

        earnings_calendar = payload.get("earnings_calendar")
        if isinstance(earnings_calendar, list) and earnings_calendar:
            used_sources.append("earnings_calendar")
            flags["earnings_calendar"] = True
            counts["earnings_events"] = len(earnings_calendar)

        earnings_surprises = payload.get("earnings_surprises")
        if isinstance(earnings_surprises, list) and earnings_surprises:
            used_sources.append("earnings_surprises")
            flags["earnings_surprises"] = True
            counts["earnings_surprises"] = len(earnings_surprises)

        stock_picks = payload.get("stock_picks")
        if isinstance(stock_picks, list) and stock_picks:
            used_sources.append("stock_screen")
            flags["stock_screen"] = True
            counts["stock_picks"] = len(stock_picks)

        source_health = payload.get("source_health")
        if isinstance(source_health, Mapping) and source_health:
            flags["source_health"] = True
            if source_health.get("score") is not None:
                counts["source_health_score"] = int(round(float(source_health.get("score") or 0.0)))
            if source_health.get("total_penalty") is not None:
                counts["source_health_penalty"] = int(round(float(source_health.get("total_penalty") or 0.0)))
            counts["source_outage_detected"] = int(bool(source_health.get("outage_detected")))

        no_trade_decision = payload.get("no_trade_decision")
        if isinstance(no_trade_decision, Mapping) and no_trade_decision:
            flags["no_trade_framework"] = True
            counts["abstain"] = int(bool(no_trade_decision.get("should_abstain")))

        factor_exposures = payload.get("factor_exposures")
        if isinstance(factor_exposures, Mapping) and factor_exposures:
            used_sources.append("factor_exposure_engine")
            flags["factor_exposures"] = True

        thesis_invalidation = payload.get("thesis_invalidation")
        if isinstance(thesis_invalidation, Mapping) and thesis_invalidation:
            used_sources.append("thesis_invalidation")
            flags["thesis_invalidation"] = True
            counts["thesis_invalidation_active"] = int(bool(thesis_invalidation.get("has_active_invalidation")))

        thesis_lifecycle = payload.get("thesis_lifecycle")
        if isinstance(thesis_lifecycle, Mapping) and thesis_lifecycle:
            flags["thesis_lifecycle"] = True
            counts["thesis_stage_known"] = int(bool(thesis_lifecycle.get("stage")))

        ordered_sources = list(dict.fromkeys(source for source in used_sources if source))
        return {
            "used_sources": ordered_sources,
            "flags": flags,
            "counts": counts,
        }

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
    def _format_source_health_lines(source_health: Any) -> str:
        if not isinstance(source_health, Mapping):
            return "- source health unavailable"
        rationale = ", ".join(str(item) for item in (source_health.get("rationale") or []) if item)
        fresh_sources = ", ".join(str(item) for item in (source_health.get("fresh_sources") or []) if item)
        stale_sources = ", ".join(str(item) for item in (source_health.get("stale_sources") or []) if item)
        sla_breaches = ", ".join(str(item) for item in (source_health.get("sla_breached_sources") or []) if item)
        return (
            f"- score={source_health.get('score')} | label={source_health.get('label')} | freshness={source_health.get('freshness_pct')} | penalty={source_health.get('total_penalty')} | sla={source_health.get('sla_status')}\n"
            f"- fresh={fresh_sources or '-'} | stale={stale_sources or '-'} | avg_age_h={source_health.get('avg_source_age_hours') if source_health.get('avg_source_age_hours') is not None else '-'}\n"
            f"- outage={source_health.get('outage_detected')} | coverage={source_health.get('coverage_pct') if source_health.get('coverage_pct') is not None else '-'} | sla_breaches={sla_breaches or '-'}\n"
            f"- rationale: {rationale or 'ไม่มีเหตุผลเพิ่มเติม'}"
        )

    @staticmethod
    def _format_data_quality_lines(data_quality: Any) -> str:
        if not isinstance(data_quality, Mapping):
            return "- data quality unavailable"
        issues = data_quality.get("issues") if isinstance(data_quality.get("issues"), list) else []
        issue_text = ", ".join(
            f"{issue.get('check')}={issue.get('severity')}"
            for issue in issues[:4]
            if isinstance(issue, Mapping)
        )
        gx = data_quality.get("gx") if isinstance(data_quality.get("gx"), Mapping) else {}
        return (
            f"- status={data_quality.get('status')} | score={data_quality.get('score')} | blocking={data_quality.get('blocking')}\n"
            f"- checks: {issue_text or 'all clear'}\n"
            f"- gx_enabled={gx.get('enabled')} | gx_executed={gx.get('executed')} | gx_success={gx.get('success_percent') if gx.get('success_percent') is not None else '-'}"
        )

    @staticmethod
    def _format_no_trade_lines(no_trade: Any) -> str:
        if not isinstance(no_trade, Mapping):
            return "- no-trade framework unavailable"
        reasons = ", ".join(str(item) for item in (no_trade.get("reasons") or []) if item)
        return (
            f"- summary: {no_trade.get('summary')}\n"
            f"- abstain={no_trade.get('should_abstain')} | reasons: {reasons or '-'}\n"
            f"- action: {no_trade.get('action') or '-'}"
        )

    @staticmethod
    def _format_champion_challenger_lines(item: Any) -> str:
        if not isinstance(item, Mapping):
            return "- champion / challenger unavailable"
        champion = item.get("champion") if isinstance(item.get("champion"), Mapping) else {}
        challenger = item.get("challenger") if isinstance(item.get("challenger"), Mapping) else {}
        runner = item.get("runner") if isinstance(item.get("runner"), Mapping) else {}
        policy_summary = ", ".join(
            f"{policy.get('name')}={policy.get('score')}"
            for policy in (runner.get("policies") or [])[:3]
            if isinstance(policy, Mapping)
        )
        return (
            f"- recommended={item.get('recommended_policy')} | delta={item.get('delta_vs_baseline')}\n"
            f"- champion {champion.get('name')}: score={champion.get('score')}\n"
            f"- challenger {challenger.get('name')}: score={challenger.get('score')}\n"
            f"- runner winner={runner.get('winner') or '-'} | policies: {policy_summary or '-'}"
        )

    @staticmethod
    def _format_factor_exposure_lines(item: Any, *, limit: int = 4) -> str:
        if not isinstance(item, Mapping) or not item:
            return "- factor exposure unavailable"
        top_exposures = item.get("top_exposures")
        exposure_text = ", ".join(
            f"{exposure.get('factor')}={exposure.get('weight_pct')}%"
            for exposure in (top_exposures or [])[: max(1, limit)]
            if isinstance(exposure, Mapping)
        )
        flags = ", ".join(str(flag) for flag in (item.get("flags") or []) if flag)
        return (
            f"- top={exposure_text or '-'}\n"
            f"- flags: {flags or '-'}"
        )

    @staticmethod
    def _format_thesis_invalidation_lines(item: Any, *, limit: int = 3) -> str:
        if not isinstance(item, Mapping) or not item:
            return "- thesis invalidation unavailable"
        signals = item.get("signals")
        signal_text = ", ".join(
            f"{signal.get('label')}({signal.get('severity')})"
            for signal in (signals or [])[: max(1, limit)]
            if isinstance(signal, Mapping)
        )
        return (
            f"- active={item.get('has_active_invalidation')} | score={item.get('score')} | severity={item.get('severity')}\n"
            f"- summary: {item.get('summary') or '-'}\n"
            f"- signals: {signal_text or '-'}\n"
            f"- action: {item.get('recommended_action') or '-'}"
        )

    @staticmethod
    def _format_thesis_lifecycle_lines(item: Any) -> str:
        if not isinstance(item, Mapping) or not item:
            return "- thesis lifecycle unavailable"
        return (
            f"- stage={item.get('stage') or '-'} | thesis_count={item.get('thesis_count') or 0}\n"
            f"- summary: {item.get('summary') or '-'}\n"
            f"- invalidation_score={item.get('invalidation_score') if item.get('invalidation_score') is not None else '-'} | "
            f"source_health_score={item.get('source_health_score') if item.get('source_health_score') is not None else '-'}"
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
        lines: list[str] = []
        for item in stock_items:
            if not isinstance(item, Mapping):
                continue
            rationale = ", ".join(item.get("rationale") or [])
            coverage_label = item.get("coverage_label")
            coverage_score = item.get("coverage_score")
            coverage_text = ""
            if isinstance(coverage_score, (float, int)) and coverage_label:
                coverage_text = f" | coverage={coverage_label} ({float(coverage_score):.2f})"
            elif coverage_label:
                coverage_text = f" | coverage={coverage_label}"
            lines.append(
                f"- {item.get('company_name')} ({item.get('ticker')}): {item.get('stance')} | score={item.get('score')} | "
                f"confidence={item.get('confidence_label')} ({item.get('confidence_score')})"
                f"{coverage_text} | {rationale}"
            )
        return "\n".join(lines) or "- ไม่มีหุ้นเด่น"

    @staticmethod
    def _format_management_commentary_lines(items: Any) -> str:
        if isinstance(items, Mapping):
            values = items.values()
        elif isinstance(items, list):
            values = items
        else:
            return "- ไม่มี management commentary ล่าสุด"
        lines: list[str] = []
        for item in values:
            if isinstance(item, TranscriptInsight):
                source_suffix = f" | source={item.source}" if item.source else ""
                lines.append(
                    f"- {item.ticker}: tone={item.tone} | guidance={item.guidance_signal} | confidence={item.confidence}{source_suffix} | {item.summary}"
                )
                continue
            if isinstance(item, Mapping):
                source = str(item.get("source") or "").strip()
                source_suffix = f" | source={source}" if source else ""
                lines.append(
                    f"- {item.get('ticker')}: tone={item.get('tone')} | guidance={item.get('guidance_signal')} | confidence={item.get('confidence')}{source_suffix} | {item.get('summary')}"
                )
        return "\n".join(lines) or "- ไม่มี management commentary ล่าสุด"

    @staticmethod
    def _format_microstructure_lines(items: Any) -> str:
        if isinstance(items, Mapping):
            values = items.values()
        elif isinstance(items, list):
            values = items
        else:
            return "- ไม่มี microstructure snapshot ล่าสุด"
        lines = [
            (
                f"- {item.get('symbol') or item.get('ticker')}: bid={item.get('best_bid')} ask={item.get('best_ask')} "
                f"| spread={item.get('spread_bps')} bps | imbalance={item.get('imbalance')} | samples={item.get('sample_count')}"
            )
            for item in values
            if isinstance(item, Mapping)
        ]
        return "\n".join(lines) or "- ไม่มี microstructure snapshot ล่าสุด"

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
    def _format_portfolio_constraints_lines(portfolio_constraints: Any) -> str:
        if not isinstance(portfolio_constraints, Mapping) or not portfolio_constraints:
            return "- ยังไม่มี portfolio constraints"
        flags = ", ".join(str(item) for item in (portfolio_constraints.get("flags") or []) if item)
        return (
            f"- allow_new_risk={portfolio_constraints.get('allow_new_risk')} | size_cap={portfolio_constraints.get('position_size_cap_pct')}% | risk_budget_multiplier={portfolio_constraints.get('risk_budget_multiplier')}\n"
            f"- largest={portfolio_constraints.get('largest_position_pct')}% | cash={portfolio_constraints.get('cash_weight_pct')}% | growth={portfolio_constraints.get('growth_weight_pct')}%\n"
            f"- dominant_sector={portfolio_constraints.get('dominant_sector') or '-'} ({portfolio_constraints.get('dominant_sector_weight_pct') or '-' }%) | dominant_theme={portfolio_constraints.get('dominant_theme') or '-'} ({portfolio_constraints.get('dominant_theme_weight_pct') or '-'}%)\n"
            f"- max_corr={portfolio_constraints.get('max_pairwise_correlation') if portfolio_constraints.get('max_pairwise_correlation') is not None else '-'} | high_corr_pairs={portfolio_constraints.get('high_correlation_pair_count') or 0} | flags: {flags or '-'}"
        )

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
        coverage_label = self._humanize_coverage_label(str(asset.get("coverage_label") or ""))
        coverage_score = self._as_float(asset.get("coverage_score"))
        if coverage_score is not None and coverage_label != "n/a":
            coverage_text = f"coverage {coverage_label} ({coverage_score:.2f})"
        elif coverage_label != "n/a":
            coverage_text = f"coverage {coverage_label}"
        else:
            coverage_text = "coverage n/a"
        level_text = f"แนวรับ {support} | แนวต้าน {resistance}" if support or resistance else "รอแนวราคาชัดเจน"
        return f"{asset.get('label')}: {stance} | {technical} | {coverage_text} | {level_text}"

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
    def _humanize_coverage_label(label: str) -> str:
        return {
            "high": "สูง",
            "medium": "ปานกลาง",
            "low": "ต่ำ",
        }.get(str(label or "").strip().casefold(), str(label or "").strip() or "n/a")

    @staticmethod
    def _assess_asset_snapshot_coverage(
        *,
        quote: AssetQuote | None,
        trend: TrendAssessment | None,
    ) -> dict[str, Any]:
        score = 0.0
        reasons: list[str] = []
        if quote is not None:
            if quote.price > 0:
                score += 0.35
                reasons.append("live price available")
            if quote.previous_close:
                score += 0.15
                reasons.append("day-over-day context available")
        if trend is not None:
            score += 0.30
            reasons.append("trend engine available")
            if trend.support_resistance.nearest_support is not None or trend.support_resistance.nearest_resistance is not None:
                score += 0.20
                reasons.append("support/resistance available")
        clipped = round(min(0.98, max(0.0, score)), 2)
        label = "high" if clipped >= 0.80 else "medium" if clipped >= 0.55 else "low"
        summary = ", ".join(reasons[:2]) if reasons else "limited market snapshot coverage"
        return {"score": clipped, "label": label, "summary": summary}

    @staticmethod
    def _assess_stock_candidate_coverage(candidate: StockCandidate) -> dict[str, Any]:
        score = 0.20
        reasons: list[str] = []
        if candidate.price is not None and candidate.price > 0:
            score += 0.08
            reasons.append("price available")
        if candidate.company_name and candidate.company_name.strip().upper() != candidate.ticker:
            score += 0.08
            reasons.append("company mapping available")
        if candidate.sector and candidate.sector.strip().casefold() != "unknown":
            score += 0.08
            reasons.append("sector mapping available")
        if candidate.benchmark and candidate.benchmark.strip().casefold() != "custom":
            score += 0.08
            reasons.append("benchmark mapping available")
        liquidity = str(candidate.liquidity_tier or "").strip().casefold()
        if liquidity in {"very_high", "high"}:
            score += 0.12
            reasons.append("liquidity supports execution")
        elif liquidity == "medium":
            score += 0.07
        market_cap = str(candidate.market_cap_bucket or "").strip().casefold()
        if market_cap in {"mega", "large"}:
            score += 0.10
            reasons.append("market cap coverage is broad")
        elif market_cap == "mid":
            score += 0.06
        fundamentals_present = sum(
            value is not None
            for value in (
                candidate.forward_pe,
                candidate.trailing_pe,
                candidate.revenue_growth,
                candidate.earnings_growth,
                candidate.return_on_equity,
                candidate.debt_to_equity,
                candidate.profit_margin,
            )
        )
        score += min(0.26, fundamentals_present * 0.04)
        if fundamentals_present >= 4:
            reasons.append("fundamental coverage is solid")
        elif fundamentals_present <= 1:
            reasons.append("fundamental coverage is thin")
        if candidate.universe_quality_score >= 0.75:
            score += 0.08
        elif candidate.universe_quality_score <= 0.40:
            score -= 0.06
            reasons.append("universe quality is fragile")
        clipped = round(min(0.98, max(0.25, score)), 2)
        label = "high" if clipped >= 0.82 else "medium" if clipped >= 0.62 else "low"
        summary = ", ".join(dict.fromkeys(reasons)) if reasons else "limited candidate coverage"
        return {"score": clipped, "label": label, "summary": summary}

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

    def _build_stock_candidate_action(
        self,
        candidate: StockCandidate,
        confidence_score: float,
        *,
        mode: str,
        portfolio_constraints: Mapping[str, Any] | None = None,
        approval_mode: str = "auto",
        max_position_size_pct: float | None = None,
    ) -> str:
        source_kind = {
            "daily": "daily_pick",
            "opportunity": "opportunity_pick",
            "watchlist": "watchlist_pick",
        }.get(mode, "stock_pick")
        position_plan = self._determine_stock_pick_position_plan(
            source_kind=source_kind,
            confidence_score=confidence_score,
            stance=candidate.stance,
            candidate=candidate,
            portfolio_constraints=portfolio_constraints,
            approval_mode=approval_mode,
            max_position_size_pct=max_position_size_pct,
        )
        if position_plan.get("blocked"):
            blocked_reasons = "; ".join(position_plan.get("blocked_reasons") or [])
            return (
                "งดเปิดสถานะใหม่ตอนนี้และรอ risk budget / execution setup ดีขึ้นก่อน"
                + (f" | {blocked_reasons}" if blocked_reasons else "")
            )
        approval_state = position_plan.get("approval_state")
        approval_suffix = ""
        if isinstance(approval_state, Mapping) and bool(approval_state.get("approval_required")):
            approval_suffix = " | ต้องมี human approval ก่อนลงมือ"
        if candidate.stance == "buy":
            if confidence_score >= 0.7:
                return (
                    f"ทยอยสะสมได้เป็นไม้เล็กถึงกลาง ราว {position_plan['position_size_pct']:.1f}% ของพอร์ต "
                    f"| signal shelf life ประมาณ {position_plan['ttl_minutes']} นาที"
                    f"{approval_suffix}"
                )
            return (
                f"เริ่มติดตามจุดเข้าได้ แต่ควรรอราคาและแรงซื้อยืนยันก่อนเพิ่มน้ำหนัก "
                f"| เริ่มไม่เกิน {position_plan['position_size_pct']:.1f}% ของพอร์ต"
                f"{approval_suffix}"
            )
        if mode == "watchlist":
            return (
                f"คงไว้ใน watchlist และรอสัญญาณราคา/ข่าวยืนยันก่อนเพิ่มน้ำหนัก "
                f"| ถ้าเข้าจริงเริ่มราว {position_plan['position_size_pct']:.1f}% ของพอร์ต"
                f"{approval_suffix}"
            )
        return (
            f"เฝ้าดูต่อและรอจังหวะที่ risk/reward ชัดขึ้นก่อนเปิดสถานะ "
            f"| initial size ราว {position_plan['position_size_pct']:.1f}% ของพอร์ต"
            f"{approval_suffix}"
        )

    def _determine_stock_pick_position_plan(
        self,
        *,
        source_kind: str,
        confidence_score: float,
        stance: str,
        candidate: StockCandidate | None = None,
        portfolio_constraints: Mapping[str, Any] | None = None,
        approval_mode: str = "auto",
        max_position_size_pct: float | None = None,
    ) -> dict[str, Any]:
        base_size = 2.0
        coverage = self._assess_stock_candidate_coverage(candidate) if candidate is not None else None
        if source_kind == "daily_pick":
            base_size = 3.5
        elif source_kind == "opportunity_pick":
            base_size = 2.5
        elif source_kind == "watchlist_pick":
            base_size = 1.5
        if confidence_score >= 0.8:
            base_size += 1.0
        elif confidence_score >= 0.65:
            base_size += 0.4
        elif confidence_score <= 0.45:
            base_size -= 0.6

        learning_score = self._load_stock_pick_source_learning_score(source_kind=source_kind)
        execution_feedback = self._load_stock_pick_execution_feedback(source_kind=source_kind)
        execution_prior = self._load_execution_heatmap_prior(
            alert_kind="stock_pick",
            preferred_sources=self._build_stock_pick_alert_source_coverage().get("used_sources") or (),
        )
        multiplier = 1.0
        if learning_score is not None:
            if learning_score >= 0.72:
                multiplier += 0.2
            elif learning_score <= 0.52:
                multiplier -= 0.2
        if isinstance(execution_feedback, Mapping):
            try:
                fast_decay_rate_pct = float(execution_feedback.get("fast_decay_rate_pct"))
            except (TypeError, ValueError):
                fast_decay_rate_pct = None
            try:
                durable_rate_pct = float(execution_feedback.get("durable_rate_pct"))
            except (TypeError, ValueError):
                durable_rate_pct = None
            if fast_decay_rate_pct is not None:
                if fast_decay_rate_pct >= 35:
                    multiplier -= 0.25
                elif fast_decay_rate_pct >= 20:
                    multiplier -= 0.12
            if durable_rate_pct is not None and durable_rate_pct >= 55:
                multiplier += 0.08
        if isinstance(execution_prior, Mapping):
            try:
                prior_score = float(execution_prior.get("score"))
            except (TypeError, ValueError):
                prior_score = None
            best_bucket = str(execution_prior.get("ttl_bucket") or "").strip()
            if prior_score is not None:
                if prior_score >= 75:
                    multiplier += 0.06
                elif prior_score <= 45:
                    multiplier -= 0.08
            if best_bucket == "short":
                multiplier -= 0.05
            elif best_bucket == "long":
                multiplier += 0.04
        if stance != "buy":
            multiplier -= 0.1
        if isinstance(coverage, Mapping):
            coverage_score = float(coverage.get("score") or 0.0)
            coverage_label = str(coverage.get("label") or "").strip().casefold()
            if coverage_label == "low":
                multiplier -= 0.18
            elif coverage_label == "medium" and coverage_score < 0.70:
                multiplier -= 0.08
        if isinstance(portfolio_constraints, Mapping):
            try:
                multiplier *= float(portfolio_constraints.get("risk_budget_multiplier") or 1.0)
            except (TypeError, ValueError):
                pass
        desired_position_size_pct = max(0.5, min(7.0, round(base_size * multiplier, 1)))
        execution_realism = self._estimate_execution_realism(
            candidate=candidate,
            desired_position_size_pct=desired_position_size_pct,
            portfolio_constraints=portfolio_constraints,
        )
        position_size_pct = float(execution_realism.get("adjusted_position_size_pct") or desired_position_size_pct)
        blocked_reasons: list[str] = []
        approval_required = str(approval_mode or "").strip().casefold() in {"review", "review_only", "manual"}
        if str(approval_mode or "").strip().casefold() in {"block", "blocked", "off"}:
            blocked_reasons.append("human override blocks fresh stock-pick execution")
        if isinstance(portfolio_constraints, Mapping) and not bool(portfolio_constraints.get("allow_new_risk", True)):
            blocked_reasons.append("portfolio risk budget already constrained")
        if candidate is not None and candidate.universe_quality_score <= 0.38:
            blocked_reasons.append("universe quality too weak for a fresh entry")
        if isinstance(coverage, Mapping):
            coverage_score = float(coverage.get("score") or 0.0)
            coverage_label = str(coverage.get("label") or "").strip().casefold()
            if coverage_label == "low" and confidence_score < 0.78:
                blocked_reasons.append("coverage is too thin for a fresh entry")
            elif coverage_score <= 0.50 and confidence_score < 0.70:
                blocked_reasons.append("coverage and conviction are both too weak")
        if float(execution_realism.get("execution_cost_bps") or 0.0) >= 32.0:
            blocked_reasons.append("modeled execution cost is too high")
        if desired_position_size_pct > float(execution_realism.get("position_size_cap_pct") or desired_position_size_pct):
            blocked_reasons.append("liquidity cap forces a smaller size")
        if max_position_size_pct is not None:
            position_size_pct = min(position_size_pct, max(0.5, float(max_position_size_pct)))
        blocked = position_size_pct < 0.75 or (
            stance == "buy" and len(blocked_reasons) >= 2 and confidence_score < 0.82
        )
        if approval_required:
            position_size_pct = min(position_size_pct, max(0.5, float(max_position_size_pct or position_size_pct)))
        ttl_minutes = self._compute_alert_ttl_minutes(
            alert_kind="stock_pick",
            severity="info" if stance == "buy" else "warning",
            confidence_score=max(confidence_score, learning_score or 0.0),
            execution_feedback=execution_feedback,
            execution_prior=execution_prior,
        )
        if blocked:
            ttl_minutes = max(45, min(ttl_minutes, 90))
        return {
            "position_size_pct": round(position_size_pct, 1),
            "size_tier": "high_conviction" if position_size_pct >= 4.5 else "starter" if position_size_pct >= 2.0 else "probe",
            "learning_score": learning_score,
            "ttl_minutes": ttl_minutes,
            "execution_feedback": execution_feedback,
            "execution_prior": execution_prior,
            "execution_realism": execution_realism,
            "coverage": dict(coverage) if isinstance(coverage, Mapping) else None,
            "blocked": blocked,
            "blocked_reasons": blocked_reasons,
            "approval_state": {
                "mode": approval_mode,
                "approval_required": approval_required,
                "max_position_size_pct": max_position_size_pct,
            },
        }

    def _load_stock_pick_source_learning_score(self, *, source_kind: str) -> float | None:
        rows_getter = getattr(self.runtime_history_store, "recent_stock_pick_scorecard", None)
        if not callable(rows_getter):
            return None
        try:
            rows = rows_getter(limit=160)
        except Exception:
            return None
        closed_rows = [
            row for row in rows
            if str(row.get("source_kind") or "").strip() == source_kind
            and str(row.get("status") or "").strip().casefold() == "closed"
        ]
        if not closed_rows:
            return None
        return_values: list[float] = []
        win_count = 0
        for row in closed_rows:
            try:
                return_value = float(row.get("return_pct")) if row.get("return_pct") is not None else None
            except (TypeError, ValueError):
                return_value = None
            if return_value is None:
                continue
            return_values.append(return_value)
            if return_value > 0:
                win_count += 1
        if not return_values:
            return None
        hit_rate_pct = round((win_count / len(closed_rows)) * 100.0, 1)
        avg_return_pct = round((sum(return_values) / len(return_values)) * 100.0, 2)
        return self._compute_learning_confidence_score(
            closed_count=len(closed_rows),
            hit_rate_pct=hit_rate_pct,
            avg_return_pct=avg_return_pct,
        )

    def _load_stock_pick_execution_feedback(self, *, source_kind: str) -> dict[str, Any] | None:
        rows_getter = getattr(self.runtime_history_store, "recent_stock_pick_scorecard", None)
        if not callable(rows_getter):
            return None
        try:
            rows = rows_getter(limit=200)
        except Exception:
            return None
        closed_rows = [
            row for row in rows
            if str(row.get("source_kind") or "").strip() == source_kind
            and str(row.get("status") or "").strip().casefold() == "closed"
            and isinstance(row.get("detail"), Mapping)
        ]
        return self._summarize_execution_feedback(closed_rows)

    def _load_alert_execution_feedback(self, *, alert_kind: str) -> dict[str, Any] | None:
        rows_getter = getattr(self.runtime_history_store, "recent_stock_pick_scorecard", None)
        if not callable(rows_getter):
            return None
        try:
            rows = rows_getter(limit=200)
        except Exception:
            return None
        closed_rows = []
        for row in rows:
            if str(row.get("status") or "").strip().casefold() != "closed":
                continue
            detail = row.get("detail")
            if not isinstance(detail, Mapping):
                continue
            if str(detail.get("alert_kind") or "").strip() != alert_kind:
                continue
            closed_rows.append(row)
        return self._summarize_execution_feedback(closed_rows)

    def _load_execution_heatmap_prior(
        self,
        *,
        alert_kind: str,
        preferred_sources: Sequence[str] = (),
    ) -> dict[str, Any] | None:
        snapshot = self._load_learning_snapshot()
        execution_panel = snapshot.get("execution_panel")
        if not isinstance(execution_panel, Mapping):
            return None
        heatmap = execution_panel.get("source_ttl_heatmap")
        if not isinstance(heatmap, list):
            return None
        normalized_sources = {
            str(item).strip()
            for item in preferred_sources
            if str(item).strip()
        }
        candidates = [
            item
            for item in heatmap
            if isinstance(item, Mapping)
            and str(item.get("alert_kind") or "").strip() == alert_kind
            and (
                not normalized_sources
                or str(item.get("source") or "").strip() in normalized_sources
            )
        ]
        if not candidates and normalized_sources:
            candidates = [
                item
                for item in heatmap
                if isinstance(item, Mapping)
                and str(item.get("alert_kind") or "").strip() == alert_kind
            ]
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.get("score") or -9999.0),
                int(item.get("sample_count") or 0),
                float(item.get("avg_return_pct") or -9999.0),
            ),
            reverse=True,
        )
        return dict(ranked[0])

    @staticmethod
    def _summarize_execution_feedback(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        ttl_evaluated_count = 0
        ttl_hit_count = 0
        fast_decay_count = 0
        durable_count = 0
        hold_count = 0
        discard_count = 0
        revalidate_count = 0
        ttl_total = 0.0
        ttl_count = 0
        for row in rows:
            detail = row.get("detail")
            if not isinstance(detail, Mapping):
                continue
            raw_ttl_hit = detail.get("ttl_hit")
            if isinstance(raw_ttl_hit, bool):
                ttl_evaluated_count += 1
                if raw_ttl_hit:
                    ttl_hit_count += 1
            try:
                ttl_minutes = float(detail.get("ttl_minutes"))
            except (TypeError, ValueError):
                ttl_minutes = None
            if ttl_minutes is not None and ttl_minutes > 0:
                ttl_total += ttl_minutes
                ttl_count += 1
            decay_label = str(detail.get("signal_decay_label") or "").strip()
            if decay_label == "fast_decay":
                fast_decay_count += 1
            elif decay_label == "durable":
                durable_count += 1
            postmortem_action = str(detail.get("postmortem_action") or "").strip()
            if postmortem_action == "hold_thesis":
                hold_count += 1
            elif postmortem_action == "discard_thesis":
                discard_count += 1
            elif postmortem_action == "revalidate_thesis":
                revalidate_count += 1
        sample_count = len(rows)
        if sample_count <= 0:
            return None
        return {
            "closed_count": sample_count,
            "ttl_evaluated_count": ttl_evaluated_count,
            "ttl_hit_rate_pct": round((ttl_hit_count / ttl_evaluated_count) * 100.0, 1) if ttl_evaluated_count else None,
            "fast_decay_rate_pct": round((fast_decay_count / sample_count) * 100.0, 1),
            "durable_rate_pct": round((durable_count / sample_count) * 100.0, 1),
            "hold_rate_pct": round((hold_count / sample_count) * 100.0, 1),
            "discard_rate_pct": round((discard_count / sample_count) * 100.0, 1),
            "revalidate_rate_pct": round((revalidate_count / sample_count) * 100.0, 1),
            "avg_ttl_minutes": round(ttl_total / ttl_count, 1) if ttl_count else None,
        }

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
            "core_cpi_yoy": (-10.0, 25.0),
            "pce_yoy": (-10.0, 25.0),
            "core_pce_yoy": (-10.0, 25.0),
            "ppi_yoy": (-20.0, 40.0),
            "gdp_qoq_annualized": (-30.0, 30.0),
            "personal_income_mom": (-20.0, 20.0),
            "personal_spending_mom": (-20.0, 20.0),
            "yield_spread_10y_2y": (-10.0, 10.0),
            "high_yield_spread": (0.0, 25.0),
            "mortgage_30y": (0.0, 20.0),
            "financial_conditions_index": (-10.0, 10.0),
            "unemployment_rate": (0.0, 30.0),
            "payrolls_mom_k": (-2000.0, 2000.0),
            "payrolls_revision_k": (-2000.0, 2000.0),
            "alfred_payroll_revision_k": (-2000.0, 2000.0),
            "alfred_cpi_revision_pct": (-20.0, 20.0),
            "avg_interest_rate_pct": (0.0, 20.0),
            "operating_cash_balance_b": (0.0, 5000.0),
            "public_debt_total_t": (0.0, 200.0),
            "wti_usd": (0.0, 500.0),
            "brent_usd": (0.0, 500.0),
            "gasoline_usd_gal": (0.0, 20.0),
            "natgas_usd_mmbtu": (0.0, 100.0),
            "cftc_equity_net_pct_oi": (-100.0, 100.0),
            "cftc_ust10y_net_pct_oi": (-100.0, 100.0),
            "cftc_usd_net_pct_oi": (-100.0, 100.0),
            "cftc_gold_net_pct_oi": (-100.0, 100.0),
            "finra_spy_short_volume_ratio": (0.0, 1.0),
            "finra_qqq_short_volume_ratio": (0.0, 1.0),
            "fedwatch_next_meeting_cut_prob_pct": (0.0, 100.0),
            "fedwatch_next_meeting_hold_prob_pct": (0.0, 100.0),
            "fedwatch_next_meeting_hike_prob_pct": (0.0, 100.0),
            "fedwatch_easing_12m_prob_pct": (0.0, 100.0),
            "fedwatch_hike_12m_prob_pct": (0.0, 100.0),
            "fedwatch_target_rate_mid_pct": (0.0, 20.0),
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

    def _guard_macro_intelligence(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        metrics = payload.get("metrics") if isinstance(payload, Mapping) else {}
        return {
            "headline": str((payload or {}).get("headline") or "").strip() or "macro backdrop unavailable",
            "signals": [str(item).strip() for item in ((payload or {}).get("signals") or []) if str(item).strip()],
            "highlights": [str(item).strip() for item in ((payload or {}).get("highlights") or []) if str(item).strip()],
            "sources_used": [str(item).strip() for item in ((payload or {}).get("sources_used") or []) if str(item).strip()],
            "metrics": self._guard_macro_context(metrics if isinstance(metrics, Mapping) else {}),
            "revisions": dict((payload or {}).get("revisions") or {}) if isinstance((payload or {}).get("revisions"), Mapping) else {},
            "positioning": dict((payload or {}).get("positioning") or {}) if isinstance((payload or {}).get("positioning"), Mapping) else {},
            "qualitative": dict((payload or {}).get("qualitative") or {}) if isinstance((payload or {}).get("qualitative"), Mapping) else {},
            "short_flow": dict((payload or {}).get("short_flow") or {}) if isinstance((payload or {}).get("short_flow"), Mapping) else {},
            "fedwatch": dict((payload or {}).get("fedwatch") or {}) if isinstance((payload or {}).get("fedwatch"), Mapping) else {},
            "structured_macro": dict((payload or {}).get("structured_macro") or {}) if isinstance((payload or {}).get("structured_macro"), Mapping) else {},
            "ex_us_macro": dict((payload or {}).get("ex_us_macro") or {}) if isinstance((payload or {}).get("ex_us_macro"), Mapping) else {},
            "global_event": dict((payload or {}).get("global_event") or {}) if isinstance((payload or {}).get("global_event"), Mapping) else {},
        }

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
