from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from invest_advisor_bot.analysis.technical_indicators import SupportResistanceLevels
from invest_advisor_bot.analysis.stock_screener import StockCandidate
from invest_advisor_bot.analysis.portfolio_profile import get_investor_profile
from invest_advisor_bot.bot.portfolio_state import PortfolioHolding
from invest_advisor_bot.analysis.trend_engine import TrendAssessment
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.data_quality import ReasoningDataQualityGate
from invest_advisor_bot.providers.market_data_client import AssetQuote, CompanyIntelligence, ETFExposureProfile, FilingEvent, MacroEvent, MacroMarketReaction, MacroReactionAssetMove, MacroSurpriseSignal, StockFundamentals
from invest_advisor_bot.providers.news_client import NewsArticle
from invest_advisor_bot.providers.research_client import ResearchFinding
from invest_advisor_bot.providers.transcript_client import EarningsTranscriptClient
from invest_advisor_bot.services.recommendation_service import RecommendationService
from invest_advisor_bot.universe import StockUniverseMember


class DummyLLM:
    async def generate_text(self, *args, **kwargs):
        return None


class FakeThesisMemoryStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []
        self.eval_artifacts: list[dict[str, object]] = []

    def search_thesis_memory(self, *, query_text: str, conversation_key: str | None = None, limit: int = 3):
        return [
            {
                "thesis_key": "abc123",
                "conversation_key": conversation_key,
                "thesis_text": "ตลาดเคย priced-in เงินเฟ้อสูง แต่ price action ไม่ confirm",
                "source_kind": "recommendation",
                "query_text": query_text,
                "tags": ["inflation", "playbook"],
                "confidence_score": 0.66,
                "similarity": 0.81,
                "created_at": "2026-03-23T10:00:00+00:00",
            }
        ][:limit]

    def record_thesis_memory(self, **kwargs) -> None:  # noqa: ANN003
        self.saved.append(kwargs)

    def record_evaluation_artifact(self, **kwargs) -> None:  # noqa: ANN003
        self.eval_artifacts.append(kwargs)

    def recent_stock_pick_scorecard(self, **kwargs):  # noqa: ANN003
        return []

    def build_evaluation_dashboard(self, **kwargs):  # noqa: ANN003
        return {
            "source_ranking": [
                {"source": "fred", "weighted_score": 78.0, "ttl_fit_score": 70.0, "best_ttl_bucket": "short"},
            ],
            "execution_panel": {
                "closed_postmortems": 3,
                "ttl_hit_rate_pct": 66.7,
                "fast_decay_rate_pct": 16.7,
                "hold_after_expiry_rate_pct": 66.7,
                "discard_after_expiry_rate_pct": 33.3,
                "by_alert_kind": [
                    {
                        "alert_kind": "stock_pick",
                        "closed_postmortems": 3,
                        "ttl_hit_rate_pct": 66.7,
                        "fast_decay_rate_pct": 16.7,
                        "hold_after_expiry_rate_pct": 66.7,
                        "discard_after_expiry_rate_pct": 33.3,
                        "best_ttl_bucket": "short",
                        "best_ttl_score": 72.0,
                        "best_ttl_sample_count": 3,
                    }
                ],
                "best_ttl_by_alert_kind": [
                    {"alert_kind": "stock_pick", "best_ttl_bucket": "short", "sample_count": 3},
                ],
                "source_ttl_heatmap": [
                    {
                        "source": "fred",
                        "alert_kind": "stock_pick",
                        "ttl_bucket": "short",
                        "sample_count": 3,
                        "ttl_hit_rate_pct": 66.7,
                        "hold_rate_pct": 66.7,
                        "avg_return_pct": 4.1,
                    }
                ],
            },
        }


class FakeMLflowObserver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def log_recommendation(self, **kwargs) -> str:  # noqa: ANN003
        self.calls.append(kwargs)
        return "run-mlflow-123"


class FakeAnalyticsStore:
    def __init__(self) -> None:
        self.recommendations: list[dict[str, object]] = []
        self.runtime_snapshots: list[dict[str, object]] = []

    def status(self) -> dict[str, object]:
        return {"available": True}

    def record_recommendation_event(self, **kwargs) -> None:  # noqa: ANN003
        self.recommendations.append(kwargs)

    def record_runtime_snapshot(self, snapshot) -> None:  # noqa: ANN001
        self.runtime_snapshots.append(snapshot)


class FakeEvidentlyObserver:
    def __init__(self) -> None:
        self.recommendations: list[dict[str, object]] = []

    def status(self) -> dict[str, object]:
        return {"enabled": True}

    def log_recommendation(self, **kwargs) -> None:  # noqa: ANN003
        self.recommendations.append(kwargs)


class FakePlaybookLearningStore(FakeThesisMemoryStore):
    def recent_stock_pick_scorecard(self, **kwargs):  # noqa: ANN003
        return [
            {
                "status": "closed",
                "return_pct": 0.09,
                "source_kind": "daily_pick",
                "detail": {
                    "thesis_summary": "sticky inflation lowers the setup for duration-sensitive growth",
                    "macro_drivers": ["sticky inflation lowers the setup for duration-sensitive growth", "rates"],
                    "thesis_memory": [{"thesis_text": "hot CPI but no market confirm", "tags": ["inflation", "cpi"]}],
                },
            },
            {
                "status": "closed",
                "return_pct": 0.05,
                "source_kind": "daily_pick",
                "detail": {
                    "thesis_summary": "sticky inflation lowers the setup for duration-sensitive growth",
                    "macro_drivers": ["sticky inflation", "duration-sensitive growth"],
                    "thesis_memory": [{"thesis_text": "CPI surprise did not get confirmation", "tags": ["cpi", "rates"]}],
                },
            },
        ]

    def build_evaluation_dashboard(self, **kwargs):  # noqa: ANN003
        return {
            "source_ranking": [
                {"source": "fred", "weighted_score": 86.0},
                {"source": "macro_surprise_engine", "weighted_score": 84.0},
                {"source": "macro_market_reaction", "weighted_score": 82.0},
                {"source": "macro_post_event_playbooks", "weighted_score": 80.0},
            ]
        }


class FakeDecayLearningStore(FakeThesisMemoryStore):
    def recent_stock_pick_scorecard(self, **kwargs):  # noqa: ANN003
        return [
            {
                "status": "closed",
                "return_pct": -0.03,
                "source_kind": "daily_pick",
                "detail": {
                    "alert_kind": "stock_pick",
                    "ttl_minutes": 240,
                    "ttl_hit": True,
                    "signal_decay_label": "fast_decay",
                    "postmortem_action": "discard_thesis",
                },
            },
            {
                "status": "closed",
                "return_pct": -0.01,
                "source_kind": "daily_pick",
                "detail": {
                    "alert_kind": "stock_pick",
                    "ttl_minutes": 210,
                    "ttl_hit": False,
                    "signal_decay_label": "expired_without_follow_through",
                    "postmortem_action": "discard_thesis",
                },
            },
            {
                "status": "closed",
                "return_pct": 0.02,
                "source_kind": "daily_pick",
                "detail": {
                    "alert_kind": "stock_pick",
                    "ttl_minutes": 200,
                    "ttl_hit": False,
                    "signal_decay_label": "mixed",
                    "postmortem_action": "revalidate_thesis",
                },
            },
        ]


def _sample_stock_candidate(
    *,
    ticker: str,
    company_name: str,
    sector: str = "Technology",
    stance: str = "buy",
) -> StockCandidate:
    return StockCandidate(
        asset=ticker.casefold(),
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        benchmark="sp500",
        market_cap_bucket="mega",
        liquidity_tier="very_high",
        price=190.0,
        trend_direction="uptrend",
        trend_score=3.0,
        rsi=58.0,
        day_change_pct=1.0,
        valuation_score=1.0,
        quality_score=1.0,
        growth_score=1.0,
        technical_score=1.5,
        macro_overlay_score=0.4,
        universe_quality_score=0.8,
        composite_score=2.5,
        stance=stance,
        rationale=("quality leader",),
        macro_drivers=("soft landing",),
        universe_flags=("liquidity_ok",),
        trailing_pe=28.0,
        forward_pe=24.0,
        revenue_growth=0.08,
        earnings_growth=0.11,
        return_on_equity=0.3,
        debt_to_equity=120.0,
        profit_margin=0.24,
    )


def _sample_quote(
    *,
    ticker: str = "SPY",
    price: float = 510.0,
    previous_close: float = 505.0,
) -> AssetQuote:
    return AssetQuote(
        ticker=ticker,
        name=ticker,
        currency="USD",
        exchange="NYSE",
        price=price,
        previous_close=previous_close,
        open_price=previous_close + 1.0,
        day_high=price + 1.0,
        day_low=previous_close - 1.0,
        volume=1000,
        timestamp=datetime.now(timezone.utc),
    )


def _sample_trend(
    *,
    ticker: str = "SPY",
    direction: str = "uptrend",
    score: float = 3.0,
    rsi: float = 58.0,
    current_price: float = 510.0,
) -> TrendAssessment:
    return TrendAssessment(
        ticker=ticker,
        direction=direction,  # type: ignore[arg-type]
        score=score,
        current_price=current_price,
        ema_fast=current_price - 2.0,
        ema_slow=current_price - 7.0,
        ema_gap_pct=0.01,
        rsi=rsi,
        macd=1.2,
        macd_signal=0.7,
        macd_hist=0.5,
        support_resistance=SupportResistanceLevels(
            current_price=current_price,
            nearest_support=current_price - 10.0,
            nearest_resistance=current_price + 10.0,
            supports=[current_price - 10.0],
            resistances=[current_price + 10.0],
            rolling_support=current_price - 12.0,
            rolling_resistance=current_price + 11.0,
        ),
        reasons=["price_above_fast_and_slow_ema", "macd_bullish"],
    )


@pytest.mark.asyncio
async def test_recommendation_service_builds_profile_aware_fallback_and_limits_history() -> None:
    service = RecommendationService(DummyLLM(), chat_history_limit=3, default_investor_profile="conservative")
    news = [NewsArticle("Fed steady", "https://example.com", "Reuters", None, None, "1")]
    market_data = {
        "spy_etf": _sample_quote(),
        "gld_etf": _sample_quote(ticker="GLD", price=220.0, previous_close=218.0),
    }
    trends = {
        "spy_etf": _sample_trend(),
        "gld_etf": _sample_trend(ticker="GLD", current_price=220.0),
    }

    for index in range(3):
        await service.generate_recommendation(
            news=news,
            market_data=market_data,
            trends=trends,
            question=f"portfolio question {index}",
            conversation_key="chat-1",
            asset_scope="us-stocks",
        )

    result = await service.generate_recommendation(
        news=news,
        market_data=market_data,
        trends=trends,
        question="analyze ETF portfolio in detail",
        conversation_key="chat-1",
        asset_scope="etf-only",
    )

    assert result.fallback_used is True
    assert result.input_payload["investor_profile"]["name"] == "conservative"
    assert result.input_payload["portfolio_plan"]["profile_name"] == "conservative"
    assert len(service._conversation_history["chat-1"]) == 3


@pytest.mark.asyncio
async def test_recommendation_service_persists_profile_selection_per_conversation() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    profile = service.set_investor_profile(conversation_key="chat-42", profile_name="growth")

    result = await service.generate_recommendation(
        news=[],
        market_data={"spy_etf": _sample_quote()},
        trends={"spy_etf": _sample_trend()},
        question="market outlook",
        conversation_key="chat-42",
        asset_scope="us-stocks",
    )

    assert profile.name == "growth"
    assert result.input_payload["investor_profile"]["name"] == "growth"


@pytest.mark.asyncio
async def test_recommendation_service_uses_thesis_memory_and_logs_learning_artifacts() -> None:
    thesis_store = FakeThesisMemoryStore()
    mlflow_observer = FakeMLflowObserver()
    service = RecommendationService(
        DummyLLM(),
        runtime_history_store=thesis_store,  # type: ignore[arg-type]
        mlflow_observer=mlflow_observer,  # type: ignore[arg-type]
    )

    result = await service.generate_recommendation(
        news=[NewsArticle("Fed steady", "https://example.com", "Reuters", None, None, "1")],
        market_data={"spy_etf": _sample_quote()},
        trends={"spy_etf": _sample_trend()},
        macro_intelligence={"headline": "macro backdrop mixed", "signals": ["inflation"], "highlights": [], "sources_used": ["fred"], "metrics": {}},
        question="เงินเฟ้อร้อน แต่ตลาดไม่ยืนยัน ควรทำอย่างไร",
        conversation_key="chat-thesis",
        asset_scope="all",
    )

    assert result.fallback_used is True
    assert result.input_payload["thesis_memory"]
    assert "ตลาดเคย priced-in เงินเฟ้อสูง" in result.recommendation_text
    assert thesis_store.saved
    assert thesis_store.eval_artifacts
    assert mlflow_observer.calls
    assert "execution_panel" in mlflow_observer.calls[0]
    assert "source_ranking" in mlflow_observer.calls[0]
    assert "source_health" in mlflow_observer.calls[0]
    assert "champion_challenger" in mlflow_observer.calls[0]
    assert "factor_exposures" in mlflow_observer.calls[0]
    assert "thesis_invalidation" in mlflow_observer.calls[0]
    assert mlflow_observer.calls[0]["artifact_key"]
    assert "response_id" in mlflow_observer.calls[0]
    assert "execution_panel" in thesis_store.eval_artifacts[0]["detail"]
    assert "source_ttl_heatmap" in thesis_store.eval_artifacts[0]["detail"]["execution_panel"]
    assert "source_health" in thesis_store.eval_artifacts[0]["detail"]
    assert "champion_challenger" in thesis_store.eval_artifacts[0]["detail"]
    assert "factor_exposures" in thesis_store.eval_artifacts[0]["detail"]
    assert "thesis_invalidation" in thesis_store.eval_artifacts[0]["detail"]
    assert thesis_store.eval_artifacts[-1]["detail"]["mlflow_run_id"] == "run-mlflow-123"


@pytest.mark.asyncio
async def test_recommendation_service_payload_includes_health_constraints_and_playbooks() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    result = await service.generate_recommendation(
        news=[NewsArticle("Fed steady", "https://example.com", "Reuters", None, None, "1")],
        market_data={"spy_etf": _sample_quote()},
        trends={"spy_etf": _sample_trend()},
        macro_intelligence={"headline": "macro backdrop mixed", "signals": ["inflation"], "highlights": [], "sources_used": ["fred"], "metrics": {}},
        macro_event_calendar=[
            MacroEvent(
                event_key="cpi",
                event_name="CPI",
                category="inflation",
                source="fred_release_calendar",
                scheduled_at=datetime.now(timezone.utc),
                importance="high",
                status="scheduled",
                source_url="https://fred.example/cpi",
            )
        ],
        portfolio_snapshot={
            "total_market_value": 100000.0,
            "holdings": [
                {"ticker": "AAPL", "category": "growth", "sector": "Technology", "market_value": 28000.0, "current_weight_pct": 28.0},
                {"ticker": "CASH", "category": "cash", "sector": None, "market_value": 2000.0, "current_weight_pct": 2.0},
            ],
        },
        question="ตอนนี้ควรงดเปิดความเสี่ยงเพิ่มไหม",
        conversation_key="chat-health",
        asset_scope="all",
    )

    payload = result.input_payload
    assert payload["source_health"]["score"] > 0
    assert payload["source_health"]["total_penalty"] >= 0
    assert payload["source_health"]["sla_status"] in {"healthy", "degraded", "outage"}
    assert payload["factor_exposures"]["top_exposure_weight_pct"] >= 28.0
    assert payload["portfolio_constraints"]["largest_position_pct"] >= 28.0
    assert payload["portfolio_constraints"]["allow_new_risk"] is False
    assert payload["portfolio_constraints"]["factor_exposures"]["top_exposure_factor"] == payload["factor_exposures"]["top_exposure_factor"]
    assert "dominant_theme" in payload["portfolio_constraints"]
    assert payload["regime_specific_playbooks"]
    assert "recommended_policy" in payload["champion_challenger"]
    assert payload["champion_challenger"]["runner"]["winner"]
    assert payload["thesis_invalidation"]["severity"] in {"low", "moderate", "high"}
    assert payload["thesis_lifecycle"]["stage"] in {"born", "confirmed", "weakening", "invalidated", "archived"}
    assert payload["no_trade_decision"]["should_abstain"] is True


def test_recommendation_service_serializes_enriched_macro_company_and_etf_context() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    payload = service._build_payload(
        news=[],
        market_data={"voo_etf": _sample_quote(ticker="VOO", price=510.0, previous_close=506.0)},
        trends={"voo_etf": _sample_trend(ticker="VOO", current_price=510.0)},
        macro_context={"vix": 19.0},
        macro_intelligence={"headline": "macro mixed", "signals": [], "highlights": [], "sources_used": ["fred"], "metrics": {}},
        macro_event_calendar=[
            MacroEvent(
                event_key="japan_interest_rate_decision",
                event_name="Interest Rate Decision",
                category="policy",
                source="trading_economics_global_calendar",
                scheduled_at=datetime(2026, 4, 3, 3, 0, tzinfo=timezone.utc),
                importance="high",
                status="scheduled",
                country="Japan",
                previous_value=0.25,
                forecast_value=0.50,
            )
        ],
        macro_surprise_signals=[],
        macro_market_reactions=[],
        research_findings=[],
        portfolio_snapshot={},
        asset_scope="all",
        question="สรุป ETF และ macro",
        investor_profile=get_investor_profile("balanced"),
        company_intelligence=[
            CompanyIntelligence(
                ticker="AAPL",
                company_name="Apple",
                cik="0000320193",
                latest_10k_filed_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
                latest_10q_filed_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
                latest_8k_filed_at=datetime(2026, 2, 15, tzinfo=timezone.utc),
                revenue_latest=100.0,
                revenue_yoy_pct=12.5,
                operating_cash_flow_latest=30.0,
                free_cash_flow_latest=24.0,
                debt_latest=50.0,
                debt_delta_pct=-2.0,
                share_dilution_yoy_pct=0.4,
                one_off_signal="low",
                guidance_signal="positive",
                insider_signal="accumulating",
                sentiment_signal="bullish",
                earnings_expectation_signal="supportive",
                analyst_rating_signal="bullish",
                analyst_buy_count=18,
                analyst_hold_count=6,
                analyst_sell_count=2,
                analyst_upside_pct=11.5,
                insider_net_shares=12500.0,
                insider_net_value=2500000.0,
                insider_transaction_count=3,
                insider_last_filed_at=datetime(2026, 2, 20, tzinfo=timezone.utc),
                corporate_action_signal="shareholder_return",
                filing_highlights=("guidance=positive", "analyst=bullish"),
            )
        ],
        etf_exposures=[
            ETFExposureProfile(
                ticker="VOO",
                fund_family="Vanguard",
                category="Large Blend",
                total_assets=500_000_000_000.0,
                fund_flow_1m_pct=1.4,
                top_holdings=(("Microsoft", 7.1), ("Apple", 6.6)),
                sector_exposures=(("Technology", 31.5), ("Financial Services", 13.2)),
                country_exposures=(("United States", 98.7),),
                concentration_score=19.0,
                exposure_signal="country_concentrated",
                source="yfinance",
            )
        ],
    )

    assert payload["macro_event_calendar"][0]["country"] == "Japan"
    assert payload["macro_event_calendar"][0]["forecast_value"] == 0.5
    assert payload["company_intelligence"][0]["analyst_rating_signal"] == "bullish"
    assert payload["company_intelligence"][0]["corporate_action_signal"] == "shareholder_return"
    assert payload["etf_exposures"][0]["ticker"] == "VOO"

    prompt = service._build_prompt(payload=payload, question="สรุป ETF และ macro", history_lines=[])

    assert "ETF holdings / exposure" in prompt
    assert "Japan" in prompt
    assert "country_concentrated" in prompt


class FakeMarketDataClient:
    async def get_history(self, ticker: str, *, period="6mo", interval="1d", limit=None):
        now = datetime.now(timezone.utc)
        normalized = ticker.upper()
        profiles = {
            "SPY": [430 + index * 1.1 for index in range(90)],
            "RSP": [160 + index * 0.45 for index in range(90)],
            "QQQ": [350 + index * 1.9 for index in range(90)],
            "RYT": [290 + index * 0.2 for index in range(90)],
        }
        closes = profiles.get(normalized, [100 + index * 0.1 for index in range(90)])
        bars = [
            type(
                "Bar",
                (),
                {
                    "ticker": normalized,
                    "timestamp": now,
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 2.0,
                    "close": close,
                    "volume": 10_000,
                },
            )()
            for close in closes
        ]
        return bars[-limit:] if isinstance(limit, int) and limit > 0 else bars

    async def get_core_market_snapshot(self):
        return {
            "sp500_index": _sample_quote(ticker="^GSPC", price=5100.0, previous_close=5070.0),
            "nasdaq_index": _sample_quote(ticker="^IXIC", price=18000.0, previous_close=17820.0),
            "xlk_etf": _sample_quote(ticker="XLK", price=210.0, previous_close=206.0),
            "xlf_etf": _sample_quote(ticker="XLF", price=43.0, previous_close=44.0),
        }

    async def get_core_market_history(self, *, period, interval, limit):
        now = datetime.now(timezone.utc)
        upward = [
            type(
                "Bar",
                (),
                {
                    "ticker": "SPY",
                    "timestamp": now,
                    "open": 100.0 + index,
                    "high": 101.0 + index,
                    "low": 99.0 + index,
                    "close": 100.0 + index,
                    "volume": 10_000,
                },
            )()
            for index in range(60)
        ]
        downward = [
            type(
                "Bar",
                (),
                {
                    "ticker": "XLF",
                    "timestamp": now,
                    "open": 100.0 - index,
                    "high": 101.0 - index,
                    "low": 99.0 - index,
                    "close": 100.0 - index,
                    "volume": 10_000,
                },
            )()
            for index in range(60)
        ]
        return {
            "sp500_index": upward,
            "nasdaq_index": upward,
            "xlk_etf": upward,
            "xlf_etf": downward,
        }

    async def get_macro_context(self):
        return {
            "vix": 19.5,
            "tnx": 4.2,
            "cpi_yoy": 3.1,
            "yield_spread_10y_2y": -0.35,
            "unemployment_rate": 4.1,
            "core_pce_yoy": 2.7,
            "gdp_qoq_annualized": 1.8,
            "personal_spending_mom": 0.2,
            "alfred_payroll_revision_k": -12.0,
            "cftc_equity_net_pct_oi": 9.0,
            "cftc_ust10y_net_pct_oi": -4.0,
            "finra_spy_short_volume_ratio": 0.51,
        }

    async def get_macro_intelligence(self):
        return {
            "headline": "growth slowing under the surface",
            "signals": ["yield_curve_inverted", "payroll_momentum_soft"],
            "highlights": ["2s10s inverted -0.35", "payroll add 110k"],
            "sources_used": ["fred", "bls", "treasury", "eia"],
            "metrics": {
                "vix": 19.5,
                "tnx": 4.2,
                "cpi_yoy": 3.1,
                "yield_spread_10y_2y": -0.35,
                "unemployment_rate": 4.1,
                "wti_usd": 82.4,
            },
        }

    async def get_macro_event_calendar(self, *, days_ahead=30):
        scheduled_at = datetime.now(timezone.utc).replace(hour=12, minute=30, second=0, microsecond=0)
        return [
            MacroEvent(
                event_key="cpi",
                event_name="CPI",
                category="inflation",
                source="fred_release_calendar",
                scheduled_at=scheduled_at,
                importance="high",
                status="scheduled",
                source_url="https://fred.example/cpi",
            ),
            MacroEvent(
                event_key="fomc",
                event_name="FOMC",
                category="policy",
                source="federal_reserve",
                scheduled_at=scheduled_at.replace(day=min(28, scheduled_at.day + 7)),
                importance="critical",
                status="scheduled",
                source_url="https://fed.example/fomc",
            ),
        ]

    async def get_macro_surprise_signals(self):
        released_at = datetime.now(timezone.utc).replace(hour=12, minute=30, second=0, microsecond=0)
        return [
            MacroSurpriseSignal(
                event_key="cpi",
                event_name="CPI",
                category="inflation",
                source="fred",
                released_at=released_at,
                next_event_at=released_at.replace(day=min(28, released_at.day + 10)),
                actual_value=3.4,
                expected_value=3.1,
                surprise_value=0.3,
                surprise_direction="hotter",
                surprise_label="hotter_than_baseline",
                market_bias="rates_up_risk_off",
                rationale=("actual=3.40%", "baseline=3.10%", "surprise=+0.30ppt"),
                detail_url="https://fred.example/cpi-latest",
            ),
            MacroSurpriseSignal(
                event_key="fomc",
                event_name="FOMC",
                category="policy",
                source="federal_reserve",
                released_at=released_at.replace(day=max(1, released_at.day - 5)),
                next_event_at=released_at.replace(day=min(28, released_at.day + 20)),
                actual_value=2.0,
                expected_value=0.0,
                surprise_value=2.0,
                surprise_direction="hawkish",
                surprise_label="hawkish_shift",
                market_bias="defensive_rates_up",
                rationale=("latest_tone=+2", "prior_tone=0", "shift=+2"),
                detail_url="https://fed.example/statement",
            ),
        ]

    async def get_macro_market_reactions(self):
        released_at = datetime.now(timezone.utc).replace(hour=12, minute=30, second=0, microsecond=0)
        return [
            MacroMarketReaction(
                event_key="cpi",
                event_name="CPI",
                released_at=released_at,
                market_bias="rates_up_risk_off",
                confirmation_label="not_confirmed",
                confirmation_score_5m=0.2,
                confirmation_score_1h=0.4,
                rationale=("5m_confirm=0.2", "1h_confirm=0.4", "macro surprise strong but cross-asset reaction did not confirm"),
                reactions=(
                    MacroReactionAssetMove("SPY", "SPY", 0.12, 0.18, "down", False, False),
                    MacroReactionAssetMove("QQQ", "QQQ", 0.2, 0.3, "down", False, False),
                    MacroReactionAssetMove("TLT", "TLT", -0.15, -0.22, "down", True, True),
                    MacroReactionAssetMove("DX-Y.NYB", "DXY", -0.03, 0.02, "up", False, False),
                    MacroReactionAssetMove("^VIX", "VIX", 0.1, 0.08, "up", True, True),
                ),
            )
        ]

    async def get_dynamic_stock_universe(self, *, indexes=("sp500", "nasdaq100"), max_members=None):
        return {
            "aapl": StockUniverseMember("AAPL", "Apple", "Technology", "nasdaq100"),
            "msft": StockUniverseMember("MSFT", "Microsoft", "Technology", "nasdaq100"),
            "xom": StockUniverseMember("XOM", "Exxon Mobil", "Energy", "sp500"),
        }

    async def get_stock_universe_snapshot(self, stock_universe):
        prices = {"AAPL": 195.0, "MSFT": 430.0, "XOM": 115.0}
        prevs = {"AAPL": 191.0, "MSFT": 425.0, "XOM": 116.5}
        return {
            alias: _sample_quote(
                ticker=member.ticker,
                price=prices.get(member.ticker, 100.0),
                previous_close=prevs.get(member.ticker, 99.0),
            )
            for alias, member in stock_universe.items()
        }

    async def get_stock_universe_history(self, *, stock_universe, period, interval, limit):
        now = datetime.now(timezone.utc)
        bars = {}
        for alias, member in stock_universe.items():
            closes = {
                "AAPL": [150 + index for index in range(50)],
                "MSFT": [200 + index * 1.2 for index in range(50)],
                "XOM": [130 - index * 0.4 for index in range(50)],
            }.get(member.ticker, [100.0] * 50)
            bars[alias] = [
                type(
                    "Bar",
                    (),
                    {
                        "ticker": member.ticker,
                        "timestamp": now,
                        "open": close - 1.0,
                        "high": close + 1.0,
                        "low": close - 2.0,
                        "close": close,
                        "volume": 10_000,
                    },
                )()
                for close in closes
            ]
        return bars

    async def get_stock_universe_fundamentals(self, stock_universe):
        fundamentals = {
            "AAPL": dict(sector="Technology", forward_pe=24.0, revenue_growth=0.12, earnings_growth=0.16, roe=1.5, revenue_qoq_change=0.06, operating_margin_qoq_change=0.02, free_cash_flow_qoq_change=0.18, revenue_quality_trend="improving", margin_quality_trend="improving", fcf_quality_trend="improving"),
            "MSFT": dict(sector="Technology", forward_pe=29.0, revenue_growth=0.15, earnings_growth=0.18, roe=1.7, revenue_qoq_change=0.04, operating_margin_qoq_change=0.01, free_cash_flow_qoq_change=0.12, revenue_quality_trend="improving", margin_quality_trend="improving", fcf_quality_trend="improving"),
            "XOM": dict(sector="Energy", forward_pe=13.0, revenue_growth=0.03, earnings_growth=0.02, roe=0.7, revenue_qoq_change=-0.05, operating_margin_qoq_change=-0.02, free_cash_flow_qoq_change=-0.2, revenue_quality_trend="deteriorating", margin_quality_trend="deteriorating", fcf_quality_trend="deteriorating"),
        }
        payload = {}
        for alias, member in stock_universe.items():
            values = fundamentals.get(member.ticker, {})
            payload[alias] = StockFundamentals(
                ticker=member.ticker,
                company_name=member.company_name,
                sector=values.get("sector", member.sector),
                industry="Software",
                market_cap=100_000_000_000.0,
                trailing_pe=28.0,
                forward_pe=values.get("forward_pe"),
                price_to_book=5.0,
                dividend_yield=None,
                revenue_growth=values.get("revenue_growth"),
                earnings_growth=values.get("earnings_growth"),
                profit_margin=0.22,
                operating_margin=0.24,
                return_on_equity=values.get("roe"),
                debt_to_equity=40.0,
                analyst_target_price=110.0,
                revenue_qoq_change=values.get("revenue_qoq_change"),
                operating_margin_qoq_change=values.get("operating_margin_qoq_change"),
                free_cash_flow_qoq_change=values.get("free_cash_flow_qoq_change"),
                revenue_quality_trend=values.get("revenue_quality_trend"),
                margin_quality_trend=values.get("margin_quality_trend"),
                fcf_quality_trend=values.get("fcf_quality_trend"),
            )
        return payload

    async def get_latest_price(self, ticker: str):
        return (await self.get_latest_prices([ticker])).get(ticker.upper())

    async def get_latest_prices(self, tickers):
        quotes = {
            "VOO": _sample_quote(ticker="VOO", price=510.0, previous_close=506.0),
            "QQQ": _sample_quote(ticker="QQQ", price=430.0, previous_close=426.0),
            "GLD": _sample_quote(ticker="GLD", price=220.0, previous_close=218.0),
            "AAPL": _sample_quote(ticker="AAPL", price=195.0, previous_close=191.0),
            "MSFT": _sample_quote(ticker="MSFT", price=430.0, previous_close=425.0),
            "XOM": _sample_quote(ticker="XOM", price=115.0, previous_close=116.5),
        }
        return {ticker.upper(): quotes.get(ticker.upper(), _sample_quote(ticker=ticker.upper(), price=100.0, previous_close=99.0)) for ticker in tickers}

    async def get_fundamentals(self, ticker: str):
        ticker_upper = ticker.upper()
        sector = "Technology" if ticker_upper in {"AAPL", "MSFT"} else "Energy" if ticker_upper == "XOM" else "Technology"
        revenue_growth = 0.12 if ticker_upper == "AAPL" else 0.03 if ticker_upper == "XOM" else 0.1
        earnings_growth = 0.16 if ticker_upper == "AAPL" else 0.02 if ticker_upper == "XOM" else 0.1
        profit_margin = 0.22 if ticker_upper != "XOM" else 0.08
        operating_margin = 0.24 if ticker_upper != "XOM" else 0.09
        return StockFundamentals(
            ticker=ticker_upper,
            company_name=ticker_upper,
            sector=sector,
            industry="Software",
            market_cap=100_000_000_000.0,
            trailing_pe=28.0,
            forward_pe=24.0 if ticker_upper != "XOM" else 13.0,
            price_to_book=5.0,
            dividend_yield=None,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            profit_margin=profit_margin,
            operating_margin=operating_margin,
            return_on_equity=1.2 if ticker_upper != "XOM" else 0.4,
            debt_to_equity=40.0 if ticker_upper != "XOM" else 120.0,
            analyst_target_price=110.0,
            free_cash_flow=1_000_000_000.0 if ticker_upper != "XOM" else -100_000_000.0,
            free_cash_flow_margin=0.12 if ticker_upper != "XOM" else -0.02,
            revenue_qoq_change=0.06 if ticker_upper != "XOM" else -0.05,
            operating_margin_qoq_change=0.02 if ticker_upper != "XOM" else -0.02,
            free_cash_flow_qoq_change=0.18 if ticker_upper != "XOM" else -0.15,
            revenue_quality_trend="improving" if ticker_upper != "XOM" else "deteriorating",
            margin_quality_trend="improving" if ticker_upper != "XOM" else "deteriorating",
            fcf_quality_trend="improving" if ticker_upper != "XOM" else "deteriorating",
        )

    async def get_earnings_calendar(self, tickers, *, days_ahead=7):
        base = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
        return {
            ticker.upper(): type(
                "EarningsEvent",
                (),
                {
                    "ticker": ticker.upper(),
                    "earnings_at": base,
                    "eps_estimate": 1.25,
                    "reported_eps": None,
                    "surprise_pct": None,
                },
            )()
            for ticker in tickers
        }

    async def get_recent_earnings_results(self, tickers, *, lookback_days=7):
        base = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)
        payload = {}
        for ticker in tickers:
            if ticker.upper() == "AAPL":
                payload[ticker.upper()] = type(
                    "RecentEarningsResult",
                    (),
                    {
                        "ticker": "AAPL",
                        "earnings_at": base,
                        "eps_estimate": 1.5,
                        "reported_eps": 1.7,
                        "surprise_pct": 13.3,
                    },
                )()
            elif ticker.upper() == "XOM":
                payload[ticker.upper()] = type(
                    "RecentEarningsResult",
                    (),
                    {
                        "ticker": "XOM",
                        "earnings_at": base,
                        "eps_estimate": 2.0,
                        "reported_eps": 1.7,
                        "surprise_pct": -15.0,
                    },
                )()
        return payload

    async def get_analyst_expectation_profiles(self, tickers):
        payload = {}
        for ticker in tickers:
            ticker_upper = ticker.upper()
            payload[ticker_upper] = type(
                "AnalystExpectationProfile",
                (),
                {
                    "ticker": ticker_upper,
                    "revenue_growth_estimate_current_q": 0.16 if ticker_upper == "AAPL" else 0.14 if ticker_upper == "XOM" else 0.08,
                    "eps_growth_estimate_current_q": 0.22 if ticker_upper == "AAPL" else 0.19 if ticker_upper == "XOM" else 0.1,
                    "revenue_analyst_count": 20,
                    "eps_analyst_count": 22,
                },
            )()
        return payload

    async def get_company_intelligence_batch(self, tickers, *, company_names=None):
        payload = {}
        for ticker in tickers:
            ticker_upper = ticker.upper()
            payload[ticker_upper] = CompanyIntelligence(
                ticker=ticker_upper,
                company_name=(company_names or {}).get(ticker_upper, ticker_upper),
                cik="0000000001",
                latest_10k_filed_at=datetime.now(timezone.utc),
                latest_10q_filed_at=datetime.now(timezone.utc),
                latest_8k_filed_at=datetime.now(timezone.utc),
                revenue_latest=100_000_000.0,
                revenue_yoy_pct=12.5 if ticker_upper != "XOM" else -4.0,
                operating_cash_flow_latest=12_000_000.0,
                free_cash_flow_latest=9_000_000.0,
                debt_latest=30_000_000.0,
                debt_delta_pct=-3.0 if ticker_upper != "XOM" else 8.0,
                share_dilution_yoy_pct=0.8 if ticker_upper != "XOM" else 3.5,
                one_off_signal="low" if ticker_upper != "XOM" else "moderate",
                guidance_signal="positive" if ticker_upper == "AAPL" else "negative" if ticker_upper == "XOM" else "mixed",
                insider_signal="accumulating" if ticker_upper == "AAPL" else "selling" if ticker_upper == "XOM" else "neutral",
                sentiment_signal="bullish" if ticker_upper != "XOM" else "bearish",
                earnings_expectation_signal="supportive" if ticker_upper != "XOM" else "weak",
                analyst_rating_signal="bullish" if ticker_upper != "XOM" else "bearish",
                analyst_buy_count=18,
                analyst_hold_count=6,
                analyst_sell_count=2,
                analyst_upside_pct=11.5 if ticker_upper != "XOM" else -3.0,
                insider_net_shares=12500.0 if ticker_upper != "XOM" else -8200.0,
                insider_net_value=2_500_000.0 if ticker_upper != "XOM" else -1_200_000.0,
                insider_transaction_count=3,
                insider_last_filed_at=datetime.now(timezone.utc),
                corporate_action_signal="shareholder_return",
                filing_highlights=("guidance=positive", "debt_delta=-3.0%"),
                recent_filings=(
                    FilingEvent(
                        ticker=ticker_upper,
                        cik="0000000001",
                        form="10-Q",
                        filing_date=datetime.now(timezone.utc),
                        report_date=datetime.now(timezone.utc),
                        accession_number="0000000001-26-000001",
                        primary_document="test.htm",
                        primary_document_url="https://example.com/test.htm",
                    ),
                ),
            )
        return payload

    async def get_etf_exposure_profiles(self, tickers):
        payload = {}
        for ticker in tickers:
            ticker_upper = ticker.upper()
            payload[ticker_upper] = ETFExposureProfile(
                ticker=ticker_upper,
                fund_family="Vanguard" if ticker_upper == "VOO" else "Invesco",
                category="Large Blend",
                total_assets=500_000_000_000.0,
                fund_flow_1m_pct=1.4,
                top_holdings=(("Microsoft", 7.1), ("Apple", 6.6), ("NVIDIA", 5.3)),
                sector_exposures=(("Technology", 31.5), ("Financial Services", 13.2)),
                country_exposures=(("United States", 98.7),),
                concentration_score=19.0,
                exposure_signal="country_concentrated",
                source="yfinance",
            )
        return payload


class FakeNewsClient:
    async def fetch_latest_macro_news(self, *, limit: int = 10, when: str = "1d"):
        return [
            NewsArticle(
                "Tech leadership improves as yields stabilize",
                "https://example.com/tech",
                "Reuters",
                None,
                None,
                "macro-1",
            ),
            NewsArticle(
                "Banks pause after mixed Fed outlook",
                "https://example.com/banks",
                "Reuters",
                None,
                None,
                "macro-2",
            ),
        ][:limit]

    async def fetch_stock_news(self, ticker: str, *, company_name: str | None = None, limit: int = 5, when: str = "7d"):
        if ticker.upper() == "AAPL":
            return [
                NewsArticle(
                    "AAPL raises guidance after earnings beat",
                    "https://example.com/aapl",
                    "Reuters",
                    None,
                    "Apple boosts outlook after better-than-expected quarter and revenue beat estimates by 3%",
                    ticker,
                )
            ]
        if ticker.upper() == "XOM":
            return [
                NewsArticle(
                    "XOM cuts guidance as softer demand hits margins",
                    "https://example.com/xom",
                    "Reuters",
                    None,
                    "Exxon warns of margin pressure, lowers forecast, and revenue missed estimates by 4%",
                    ticker,
                )
            ]
        return [NewsArticle(f"{ticker} earnings beat", "https://example.com", "Reuters", None, None, ticker)]


class FakeResearchClient:
    def available(self) -> bool:
        return True

    async def search_market_context(self, *, query: str, limit: int = 5):
        return []

    async def search_earnings_call_context(self, *, ticker: str, company_name: str | None = None, limit: int = 4):
        if ticker.upper() == "AAPL":
            return [
                ResearchFinding(
                    title="Apple earnings call summary",
                    url="https://example.com/apple-call",
                    source="Example",
                    snippet="Management highlighted strong demand, healthy pipeline, margin expansion, and reaffirmed guidance.",
                    provider="exa",
                )
            ]
        if ticker.upper() == "XOM":
            return [
                ResearchFinding(
                    title="Exxon earnings call summary",
                    url="https://example.com/exxon-call",
                    source="Example",
                    snippet="Management cited macro uncertainty, weaker demand, margin compression, and a one-time tax benefit after lowering forecast.",
                    provider="exa",
                )
            ]
        return []


class ExplodingNewsClient(FakeNewsClient):
    async def fetch_latest_macro_news(self, *, limit: int = 10, when: str = "1d"):
        raise RuntimeError("news unavailable")


class ExplodingMarketDataClient(FakeMarketDataClient):
    async def get_core_market_snapshot(self):
        raise RuntimeError("snapshot unavailable")

    async def get_core_market_history(self, *, period, interval, limit):
        raise RuntimeError("history unavailable")

    async def get_macro_context(self):
        raise RuntimeError("macro unavailable")

    async def get_macro_intelligence(self):
        raise RuntimeError("macro intelligence unavailable")


class ExplodingResearchClient(FakeResearchClient):
    async def search_market_context(self, *, query: str, limit: int = 5):
        raise RuntimeError("research unavailable")


class FragileBreadthMarketDataClient(FakeMarketDataClient):
    async def get_dynamic_stock_universe(self, *, indexes=("sp500", "nasdaq100"), max_members=None):
        return {
            "aapl": StockUniverseMember("AAPL", "Apple", "Technology", "nasdaq100"),
            "msft": StockUniverseMember("MSFT", "Microsoft", "Technology", "nasdaq100"),
            "xom": StockUniverseMember("XOM", "Exxon Mobil", "Energy", "sp500"),
            "cvx": StockUniverseMember("CVX", "Chevron", "Energy", "sp500"),
            "jpm": StockUniverseMember("JPM", "JPMorgan", "Financials", "sp500"),
            "bac": StockUniverseMember("BAC", "Bank of America", "Financials", "sp500"),
        }

    async def get_stock_universe_history(self, *, stock_universe, period, interval, limit):
        now = datetime.now(timezone.utc)
        bars = {}
        profiles = {
            "AAPL": [150 + index for index in range(50)],
            "MSFT": [200 + index * 0.8 for index in range(50)],
            "XOM": [130 - index * 0.2 for index in range(50)],
            "CVX": [125 - index * 0.25 for index in range(50)],
            "JPM": [180 - index * 0.15 for index in range(50)],
            "BAC": [42 - index * 0.08 for index in range(50)],
        }
        for alias, member in stock_universe.items():
            closes = profiles.get(member.ticker, [100.0] * 50)
            bars[alias] = [
                type(
                    "Bar",
                    (),
                    {
                        "ticker": member.ticker,
                        "timestamp": now,
                        "open": close - 1.0,
                        "high": close + 1.0,
                        "low": close - 2.0,
                        "close": close,
                        "volume": 10_000,
                    },
                )()
                for close in closes
            ]
        return bars


class MacroStressMarketDataClient(FakeMarketDataClient):
    async def get_macro_context(self):
        payload = dict(await super().get_macro_context())
        payload.update(
            {
                "core_pce_yoy": 3.2,
                "gdp_qoq_annualized": 0.9,
                "personal_spending_mom": -0.2,
                "alfred_payroll_revision_k": -62.0,
                "alfred_cpi_revision_pct": 0.18,
                "cftc_equity_net_pct_oi": 18.0,
                "cftc_ust10y_net_pct_oi": -12.0,
                "cftc_gold_net_pct_oi": 12.0,
                "finra_spy_short_volume_ratio": 0.57,
                "finra_qqq_short_volume_ratio": 0.58,
            }
        )
        return payload


@pytest.mark.asyncio
async def test_recommendation_service_handles_stock_screener_question() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    result = await service.screen_us_stocks(
        question="what 3 stocks should I buy now",
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=None,
        limit=3,
    )

    assert result.fallback_used is True
    assert len(result.picks) == 3
    assert "stock screener" in result.recommendation_text.casefold()


@pytest.mark.asyncio
async def test_answer_user_question_serializes_stock_pick_confidence() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    result = await service.answer_user_question(
        question="stock screener top 3",
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        conversation_key="chat-stock-picks",
    )

    assert result.fallback_used is True
    assert isinstance(result.input_payload.get("stock_picks"), list)
    assert result.input_payload["stock_picks"]
    first_pick = result.input_payload["stock_picks"][0]
    assert first_pick["confidence_label"] in {"very_high", "high", "medium", "low"}
    assert first_pick["confidence_score"] >= 0.25
    assert first_pick["price"] > 0


@pytest.mark.asyncio
async def test_recommendation_service_gracefully_degrades_when_external_sources_fail() -> None:
    service = RecommendationService(DummyLLM())

    result = await service.generate_market_update(
        news_client=ExplodingNewsClient(),  # type: ignore[arg-type]
        market_data_client=ExplodingMarketDataClient(),  # type: ignore[arg-type]
        research_client=ExplodingResearchClient(),  # type: ignore[arg-type]
        conversation_key="chat-degraded",
    )

    assert result.fallback_used is True
    assert result.input_payload["asset_snapshots"] == []
    assert result.input_payload["macro_context"]["vix"] is None

@pytest.mark.asyncio
async def test_recommendation_service_builds_periodic_report_fallback(tmp_path: Path) -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    result = await service.generate_periodic_report(
        report_kind="morning",
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        sector_rotation_state_store=SectorRotationStateStore(path=tmp_path / "sector_rotation_state.json"),
        news_limit=2,
        history_period="6mo",
        history_interval="1d",
        history_limit=60,
    )

    assert result.fallback_used is True
    assert "Morning Report" in result.recommendation_text
    assert "Macro Intelligence" in result.recommendation_text
    assert "Upcoming Macro Events" in result.recommendation_text
    assert "Macro Surprise Engine" in result.recommendation_text
    assert "Macro Market Reaction" in result.recommendation_text
    assert "Company Intelligence" in result.recommendation_text
    assert "Sector Rotation" in result.recommendation_text
    assert "Sector Persistence Daily" in result.recommendation_text
    assert "Sector Persistence Intraday" in result.recommendation_text
    assert "Market Breadth Diffusion" in result.recommendation_text
    assert "Market Breadth Trend" in result.recommendation_text
    assert "Index Leadership Divergence" in result.recommendation_text
    assert "Sector Breadth" in result.recommendation_text
    assert "Sector Breadth Trend" in result.recommendation_text
    assert "Post-Earnings Read-Through" in result.recommendation_text
    assert result.input_payload["company_intelligence"]
    assert result.input_payload["macro_event_calendar"]
    assert result.input_payload["macro_surprise_signals"]
    assert result.input_payload["macro_market_reactions"]
    assert result.input_payload["macro_intelligence"]["signals"]


@pytest.mark.asyncio
async def test_recommendation_service_market_update_includes_portfolio_snapshot_and_review() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    result = await service.generate_market_update(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        conversation_key="portfolio-chat",
        portfolio_holdings=(
            PortfolioHolding(ticker="VOO", quantity=10, avg_cost=470.0),
            PortfolioHolding(ticker="QQQ", quantity=4, avg_cost=400.0),
            PortfolioHolding(ticker="CASH", quantity=20_000.0),
        ),
    )

    assert result.fallback_used is True
    assert "Current Portfolio" in result.recommendation_text
    assert "Rebalance Review" in result.recommendation_text
    assert result.input_payload["market_confidence"]["score"] >= 0.25
    assert result.input_payload["portfolio_snapshot"]["total_market_value"] > 0
    assert result.input_payload["portfolio_review"]["holdings_count"] == 3
    assert result.input_payload["market_confidence"]["score"] >= 0.25
    assert result.input_payload["macro_intelligence"]["headline"] == "growth slowing under the surface"
    assert result.input_payload["macro_event_calendar"][0]["event_key"] == "cpi"
    assert result.input_payload["macro_surprise_signals"][0]["surprise_label"] == "hotter_than_baseline"
    assert result.input_payload["macro_surprise_signals"][0]["consensus_surprise_label"] is None
    assert result.input_payload["macro_market_reactions"][0]["confirmation_label"] == "not_confirmed"
    assert result.input_payload["macro_context"]["yield_spread_10y_2y"] == -0.35
    assert any(item["ticker"] == "VOO" for item in result.input_payload["portfolio_snapshot"]["holdings"])


@pytest.mark.asyncio
async def test_recommendation_service_applies_direct_macro_overlays_to_allocation_and_stock_picks() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    market_data_client = MacroStressMarketDataClient()  # type: ignore[assignment]

    result = await service.generate_market_update(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        conversation_key="macro-stress",
    )
    stock_candidates = await service._screen_stock_universe(  # type: ignore[attr-defined]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        top_k=3,
    )
    serialized_pick = service._serialize_stock_candidate(stock_candidates[0])  # type: ignore[attr-defined]

    buckets = {item["category"]: item["target_pct"] for item in result.input_payload["allocation_plan"]["buckets"]}
    assert buckets["cash"] >= 18
    assert buckets["growth"] <= 15
    assert "Active macro overlays" in result.input_payload["allocation_plan"]["rebalance_note"]
    assert serialized_pick["macro_overlay_score"] is not None
    assert serialized_pick["factor_risk_score"] is not None
    assert serialized_pick["beta_1m"] is not None
    assert serialized_pick["peer_benchmark_ticker"] is not None
    assert "macro_drivers" in serialized_pick
    assert serialized_pick["coverage_label"] in {"high", "medium", "low"}
    assert serialized_pick["coverage_score"] >= 0.25
    assert serialized_pick["coverage_summary"]
    assert serialized_pick["suggested_position_size_pct"] > 0
    assert serialized_pick["signal_ttl_minutes"] >= 60


def test_recommendation_service_renders_asset_focus_line_with_coverage() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    asset = service._serialize_asset_context(  # type: ignore[attr-defined]
        asset_name="spy_etf",
        quote=_sample_quote(),
        trend=_sample_trend(),
    )

    rendered = service._render_asset_focus_line(asset or {})  # type: ignore[arg-type]

    assert asset is not None
    assert asset["coverage_label"] in {"high", "medium", "low"}
    assert "coverage" in rendered


@pytest.mark.asyncio
async def test_recommendation_service_expands_allocation_shift_when_learning_evidence_is_strong() -> None:
    baseline_service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    learned_service = RecommendationService(
        DummyLLM(),
        runtime_history_store=FakePlaybookLearningStore(),  # type: ignore[arg-type]
        default_investor_profile="balanced",
    )
    market_data_client = MacroStressMarketDataClient()  # type: ignore[assignment]

    baseline_result = await baseline_service.generate_market_update(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        conversation_key="baseline-shift",
    )
    learned_result = await learned_service.generate_market_update(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        conversation_key="learned-shift",
    )

    baseline_buckets = {item["category"]: item["target_pct"] for item in baseline_result.input_payload["allocation_plan"]["buckets"]}
    learned_buckets = {item["category"]: item["target_pct"] for item in learned_result.input_payload["allocation_plan"]["buckets"]}
    assert learned_buckets["cash"] >= baseline_buckets["cash"]
    assert learned_buckets["growth"] <= baseline_buckets["growth"]
    assert "shift multiplier" in learned_result.input_payload["allocation_plan"]["rebalance_note"]


def test_recommendation_service_summarizes_source_coverage() -> None:
    coverage = RecommendationService.summarize_source_coverage(
        {
            "macro_intelligence": {"sources_used": ["fred", "bls", "treasury"]},
            "macro_event_calendar": [{"source": "fred_release_calendar"}, {"source": "federal_reserve"}],
            "macro_surprise_signals": [{"source": "fred"}, {"source": "federal_reserve"}],
            "macro_market_reactions": [{"reactions": [{"label": "SPY"}, {"label": "QQQ"}]}],
            "news_headlines": [{"title": "Fed steady"}],
            "research_highlights": [{"title": "Desk note"}],
            "company_intelligence": [{"ticker": "AAPL"}],
            "stock_picks": [{"ticker": "AAPL"}],
        }
    )

    assert "fred" in coverage["used_sources"]
    assert "macro_event_calendar" in coverage["used_sources"]
    assert "macro_surprise_engine" in coverage["used_sources"]
    assert "macro_market_reaction" in coverage["used_sources"]
    assert coverage["flags"]["company_intelligence"] is True
    assert coverage["counts"]["macro_events"] == 2
    assert coverage["counts"]["macro_surprises"] == 2
    assert coverage["counts"]["macro_market_reactions"] == 1


@pytest.mark.asyncio
async def test_recommendation_service_records_data_quality_and_learning_observers() -> None:
    analytics_store = FakeAnalyticsStore()
    evidently_observer = FakeEvidentlyObserver()
    service = RecommendationService(
        DummyLLM(),
        default_investor_profile="balanced",
        data_quality_gate=ReasoningDataQualityGate(
            enabled=True,
            gx_enabled=False,
            min_market_assets=1,
            min_macro_sources=1,
            min_news_items=1,
        ),
        analytics_store=analytics_store,  # type: ignore[arg-type]
        evidently_observer=evidently_observer,  # type: ignore[arg-type]
    )

    result = await service.generate_market_update(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        conversation_key="observer-test",
    )

    assert "data_quality" in result.input_payload
    assert result.input_payload["data_quality"]["status"] in {"pass", "warn"}
    assert analytics_store.recommendations
    assert analytics_store.runtime_snapshots
    assert evidently_observer.recommendations


@pytest.mark.asyncio
async def test_recommendation_service_generates_sector_and_earnings_alerts() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    market_data_client = FakeMarketDataClient()  # type: ignore[assignment]
    news_client = FakeNewsClient()  # type: ignore[assignment]

    sector_alerts = await service.generate_sector_rotation_alerts(
        news_client=news_client,  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=None,
    )
    stock_candidates = await service._screen_stock_universe(  # type: ignore[attr-defined]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        top_k=3,
    )
    earnings_alerts = await service.generate_earnings_calendar_alerts(
        market_data_client=market_data_client,  # type: ignore[arg-type]
        watchlist=("AAPL",),
        top_candidates=stock_candidates,
        days_ahead=7,
    )

    assert any(alert.key.startswith("sector:snapshot:") for alert in sector_alerts)
    assert any("Action" in alert.text for alert in sector_alerts)
    assert any("ปฏิทินงบ" in alert.text for alert in earnings_alerts)
    assert any("- Action:" in alert.text for alert in earnings_alerts)


@pytest.mark.asyncio
async def test_recommendation_service_formats_interest_alerts_in_thai_with_badges() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    alerts = await service.generate_interest_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        vix_threshold=10.0,
        risk_score_threshold=1.0,
        opportunity_score_threshold=2.0,
        news_impact_threshold=1.0,
    )

    assert alerts
    assert any(alert.text.startswith("🟠 ระวัง | ความเสี่ยงตลาด") for alert in alerts)
    assert any("macro surprise" in alert.text for alert in alerts)
    assert any("market reaction divergence" in alert.text for alert in alerts)
    assert any(
        alert.text.startswith("🔎 จับตา | สินทรัพย์เด่น")
        or alert.text.startswith("🟠 ระวัง | สินทรัพย์อ่อนแรง")
        or alert.text.startswith("✅ ยืนยัน | ข่าวหนุนโอกาส")
        or alert.text.startswith("🟠 ระวัง | ข่าวมหภาค")
        for alert in alerts
    )
    assert all("- Action:" in alert.text for alert in alerts)


def test_recommendation_service_playbook_confidence_learns_from_scorecard_history() -> None:
    service = RecommendationService(
        DummyLLM(),
        runtime_history_store=FakePlaybookLearningStore(),  # type: ignore[arg-type]
        default_investor_profile="balanced",
    )
    released_at = datetime.now(timezone.utc).replace(hour=12, minute=30, second=0, microsecond=0)
    playbooks = service._build_macro_post_event_playbooks(  # type: ignore[attr-defined]
        macro_surprise_signals=[
            MacroSurpriseSignal(
                event_key="cpi",
                event_name="CPI",
                category="inflation",
                source="fred",
                released_at=released_at,
                next_event_at=None,
                actual_value=3.4,
                expected_value=3.1,
                surprise_value=0.3,
                surprise_direction="hotter",
                surprise_label="hotter_than_baseline",
                market_bias="rates_up_risk_off",
                rationale=("actual=3.4",),
                detail_url=None,
            )
        ],
        macro_market_reactions=[
            MacroMarketReaction(
                event_key="cpi",
                event_name="CPI",
                released_at=released_at,
                market_bias="rates_up_risk_off",
                confirmation_label="not_confirmed",
                confirmation_score_5m=0.2,
                confirmation_score_1h=0.4,
                rationale=("cross-asset reaction did not confirm",),
                reactions=(
                    MacroReactionAssetMove("SPY", "SPY", 0.1, 0.15, "down", False, False),
                ),
            )
        ],
    )

    assert playbooks
    assert playbooks[0].confidence_score > 0.6
    assert "closed analogs" in playbooks[0].learning_note


@pytest.mark.asyncio
async def test_recommendation_service_escalates_macro_alerts_when_learning_evidence_is_strong() -> None:
    service = RecommendationService(
        DummyLLM(),
        runtime_history_store=FakePlaybookLearningStore(),  # type: ignore[arg-type]
        default_investor_profile="balanced",
    )

    alerts = await service.generate_interest_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        vix_threshold=10.0,
        risk_score_threshold=1.0,
        opportunity_score_threshold=2.0,
        news_impact_threshold=1.0,
    )

    assert any(alert.key.startswith("macro_surprise:cpi") and alert.severity == "critical" for alert in alerts)
    assert any(alert.key.startswith("macro_playbook:hot_cpi_no_market_confirm") and alert.severity == "critical" for alert in alerts)


@pytest.mark.asyncio
async def test_recommendation_service_formats_stock_pick_alerts_in_thai_with_badges() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    alerts = await service.generate_stock_pick_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        watchlist=("AAPL", "MSFT"),
        score_threshold=1.0,
        limit=3,
    )

    assert alerts
    assert any(alert.text.startswith("✅ ยืนยัน | หุ้นเด่นวันนี้") for alert in alerts)
    assert any(alert.text.startswith("🔎 จับตา |") for alert in alerts)
    assert all("- Action:" in alert.text for alert in alerts)
    stock_pick_alert = next(alert for alert in alerts if getattr(alert, "metadata", {}).get("stock_pick"))
    assert stock_pick_alert.metadata is not None
    assert stock_pick_alert.metadata["position_size_pct"] > 0
    assert stock_pick_alert.metadata["ttl_minutes"] >= 60
    assert "expires_at" in stock_pick_alert.metadata
    assert "stock_screener" in stock_pick_alert.metadata["source_coverage"]["used_sources"]
    assert "source_health" in stock_pick_alert.metadata
    assert "execution_realism" in stock_pick_alert.metadata


@pytest.mark.asyncio
async def test_recommendation_service_abstains_stock_picks_when_portfolio_is_already_crowded() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    alerts = await service.generate_stock_pick_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        score_threshold=1.0,
        limit=3,
        portfolio_holdings=(PortfolioHolding(ticker="AAPL", quantity=100.0, avg_cost=100.0),),
    )

    assert alerts
    assert alerts[0].key.startswith("stock:abstain:")
    assert "งดเปิด stock pick ตอนนี้" in alerts[0].text
    assert alerts[0].metadata["no_trade_decision"]["should_abstain"] is True


@pytest.mark.asyncio
async def test_recommendation_service_portfolio_constraints_use_correlation_and_theme_overlap() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    market_data_client = FakeMarketDataClient()

    snapshot = await service._gather_portfolio_snapshot(  # type: ignore[attr-defined]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        holdings=(
            PortfolioHolding(ticker="AAPL", quantity=100.0, avg_cost=100.0),
            PortfolioHolding(ticker="MSFT", quantity=80.0, avg_cost=100.0),
            PortfolioHolding(ticker="QQQ", quantity=20.0, avg_cost=100.0),
        ),
    )
    constraints = service._build_portfolio_constraint_summary(portfolio_snapshot=snapshot)  # type: ignore[attr-defined]

    assert constraints["dominant_theme"] in {"ai_big_tech", "technology", "growth"}
    assert constraints["dominant_theme_weight_pct"] is not None
    assert constraints["max_pairwise_correlation"] is not None
    assert constraints["high_correlation_pair_count"] >= 1
    assert any(flag in constraints["flags"] for flag in {"theme_overlap", "correlation_cluster"})


def test_recommendation_service_reduces_position_size_when_signals_decay_fast() -> None:
    service = RecommendationService(
        DummyLLM(),
        runtime_history_store=FakeDecayLearningStore(),  # type: ignore[arg-type]
        default_investor_profile="balanced",
    )

    conservative_plan = service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.82,
        stance="buy",
    )
    baseline_service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    baseline_plan = baseline_service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.82,
        stance="buy",
    )

    assert conservative_plan["position_size_pct"] < baseline_plan["position_size_pct"]
    assert conservative_plan["ttl_minutes"] < baseline_plan["ttl_minutes"]
    assert conservative_plan["execution_feedback"]["fast_decay_rate_pct"] >= 30


def test_recommendation_service_execution_realism_reduces_size_for_fragile_candidates() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    strong_candidate = StockCandidate(
        asset="aapl",
        ticker="AAPL",
        company_name="Apple",
        sector="Technology",
        benchmark="nasdaq100",
        market_cap_bucket="mega",
        liquidity_tier="very_high",
        price=190.0,
        trend_direction="uptrend",
        trend_score=3.1,
        rsi=60.0,
        day_change_pct=1.2,
        valuation_score=1.0,
        quality_score=1.1,
        growth_score=1.0,
        technical_score=1.8,
        macro_overlay_score=0.4,
        universe_quality_score=0.9,
        composite_score=2.8,
        stance="buy",
        rationale=("quality leader",),
        macro_drivers=("soft landing",),
        universe_flags=("liquidity_ok", "cap_ok"),
        trailing_pe=28.0,
        forward_pe=24.0,
        revenue_growth=0.08,
        earnings_growth=0.11,
        return_on_equity=0.3,
        debt_to_equity=120.0,
        profit_margin=0.24,
    )
    fragile_candidate = StockCandidate(
        asset="uber",
        ticker="UBER",
        company_name="Uber",
        sector="Industrials",
        benchmark="sp500",
        market_cap_bucket="mid",
        liquidity_tier="low",
        price=75.0,
        trend_direction="uptrend",
        trend_score=2.2,
        rsi=64.0,
        day_change_pct=1.0,
        valuation_score=0.4,
        quality_score=0.4,
        growth_score=0.7,
        technical_score=1.1,
        macro_overlay_score=0.1,
        universe_quality_score=0.34,
        composite_score=2.0,
        stance="buy",
        rationale=("execution risk",),
        macro_drivers=("mixed transition",),
        universe_flags=("liquidity_low", "cap_fragile"),
        trailing_pe=40.0,
        forward_pe=34.0,
        revenue_growth=0.1,
        earnings_growth=0.09,
        return_on_equity=0.12,
        debt_to_equity=150.0,
        profit_margin=0.08,
    )

    strong_plan = service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.82,
        stance="buy",
        candidate=strong_candidate,
    )
    fragile_plan = service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.82,
        stance="buy",
        candidate=fragile_candidate,
    )

    assert fragile_plan["position_size_pct"] < strong_plan["position_size_pct"]
    assert fragile_plan["execution_realism"]["execution_cost_bps"] > strong_plan["execution_realism"]["execution_cost_bps"]


def test_recommendation_service_blocks_low_coverage_candidate_when_confidence_is_not_strong() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    fragile_candidate = StockCandidate(
        asset="abcd",
        ticker="ABCD",
        company_name="ABCD",
        sector="Unknown",
        benchmark="custom",
        market_cap_bucket="small",
        liquidity_tier="low",
        price=12.0,
        trend_direction="sideways",
        trend_score=0.4,
        rsi=49.0,
        day_change_pct=0.2,
        valuation_score=0.1,
        quality_score=0.1,
        growth_score=0.1,
        technical_score=0.2,
        macro_overlay_score=0.0,
        universe_quality_score=0.28,
        composite_score=0.8,
        stance="buy",
        rationale=("coverage weak",),
        macro_drivers=(),
        universe_flags=("thin_coverage",),
        trailing_pe=None,
        forward_pe=None,
        revenue_growth=None,
        earnings_growth=None,
        return_on_equity=None,
        debt_to_equity=None,
        profit_margin=None,
    )

    plan = service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.62,
        stance="buy",
        candidate=fragile_candidate,
    )

    assert plan["blocked"] is True
    assert any("coverage" in reason for reason in plan["blocked_reasons"])
    assert plan["coverage"]["label"] == "low"


def test_recommendation_service_uses_heatmap_prior_for_stock_pick_ttl() -> None:
    service = RecommendationService(
        DummyLLM(),
        runtime_history_store=FakeThesisMemoryStore(),  # type: ignore[arg-type]
        default_investor_profile="balanced",
    )
    baseline_service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    prioritized_plan = service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.72,
        stance="buy",
    )
    baseline_plan = baseline_service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.72,
        stance="buy",
    )

    assert prioritized_plan["ttl_minutes"] <= baseline_plan["ttl_minutes"]
    assert prioritized_plan["execution_prior"]["ttl_bucket"] == "short"


def test_recommendation_service_abstains_when_top_stock_pick_coverage_is_too_thin() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    thin_pick = StockCandidate(
        asset="thin",
        ticker="THIN",
        company_name="THIN",
        sector="Unknown",
        benchmark="custom",
        market_cap_bucket="small",
        liquidity_tier="low",
        price=8.0,
        trend_direction="sideways",
        trend_score=0.2,
        rsi=50.0,
        day_change_pct=0.1,
        valuation_score=0.0,
        quality_score=0.0,
        growth_score=0.0,
        technical_score=0.1,
        macro_overlay_score=0.0,
        universe_quality_score=0.3,
        composite_score=0.6,
        stance="watch",
        rationale=("thin coverage",),
        macro_drivers=(),
        universe_flags=("thin",),
        trailing_pe=None,
        forward_pe=None,
        revenue_growth=None,
        earnings_growth=None,
        return_on_equity=None,
        debt_to_equity=None,
        profit_margin=None,
    )

    decision = service._build_stock_pick_no_trade_decision(  # type: ignore[attr-defined]
        picks=[thin_pick, thin_pick, thin_pick],
        portfolio_constraints=None,
        source_health={"score": 72.0},
    )

    assert decision["should_abstain"] is True
    assert any("coverage" in reason for reason in decision["reasons"])


def test_recommendation_service_requires_human_review_when_override_is_enabled() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")

    plan = service._determine_stock_pick_position_plan(  # type: ignore[attr-defined]
        source_kind="daily_pick",
        confidence_score=0.8,
        stance="buy",
        approval_mode="review",
        max_position_size_pct=2.5,
    )

    assert plan["approval_state"]["approval_required"] is True
    assert plan["position_size_pct"] <= 2.5


@pytest.mark.asyncio
async def test_recommendation_service_stock_pick_alerts_include_realert_cadence() -> None:
    service = RecommendationService(
        DummyLLM(),
        runtime_history_store=FakeDecayLearningStore(),  # type: ignore[arg-type]
        default_investor_profile="balanced",
    )

    alerts = await service.generate_stock_pick_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        score_threshold=1.0,
        limit=1,
    )

    assert alerts
    metadata = alerts[0].metadata or {}
    assert metadata["realert_after_minutes"] < metadata["ttl_minutes"]
    assert metadata["alert_kind"] == "stock_pick"
    assert "execution_prior" in metadata


@pytest.mark.asyncio
async def test_recommendation_service_generates_post_earnings_alerts_with_transcript_tone() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    alerts = await service.generate_post_earnings_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        watchlist=("AAPL", "XOM"),
        top_candidates=(),
        lookback_days=7,
    )

    assert len(alerts) == 2
    assert any(alert.text.startswith("✅ ยืนยัน | สรุปหลังประกาศงบ") for alert in alerts)
    assert any(alert.text.startswith("🟠 ระวัง | สรุปหลังประกาศงบ") for alert in alerts)
    assert any("guidance บวก" in alert.text for alert in alerts)
    assert any("guidance ลบ" in alert.text for alert in alerts)
    assert any("น้ำเสียงผู้บริหาร บวก" in alert.text for alert in alerts)
    assert any("น้ำเสียงผู้บริหาร ลบ" in alert.text for alert in alerts)
    assert any("รายได้ ดีกว่าคาด" in alert.text or "รายได้ ต่ำกว่าคาด" in alert.text for alert in alerts)
    assert any("FCF แข็งแรง" in alert.text or "FCF อ่อนแอ" in alert.text for alert in alerts)
    assert any("ดีขึ้น QoQ" in alert.text or "อ่อนลง QoQ" in alert.text for alert in alerts)
    assert any("vs sector" in alert.text.casefold() for alert in alerts)
    assert any("rev gap" in alert.text.casefold() for alert in alerts)
    assert any("กำไรหลักแข็งแรง" in alert.text or "คุณภาพกำไรต่ำหรือมีรายการพิเศษ" in alert.text for alert in alerts)
    assert any("one-off risk ปานกลาง" in alert.text or "one-off risk สูง" in alert.text for alert in alerts)
    assert all("- Action:" in alert.text for alert in alerts)


@pytest.mark.asyncio
async def test_recommendation_service_builds_research_proxy_transcript_insights_when_direct_api_unavailable() -> None:
    service = RecommendationService(
        DummyLLM(),
        transcript_client=EarningsTranscriptClient(api_key=""),
        default_investor_profile="balanced",
    )
    candidates = [
        _sample_stock_candidate(ticker="AAPL", company_name="Apple"),
        _sample_stock_candidate(ticker="XOM", company_name="Exxon Mobil", sector="Energy"),
    ]

    payload = await service._fetch_transcript_insights_for_candidates(  # type: ignore[attr-defined]
        candidates,
        limit=2,
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
    )

    assert payload["AAPL"].source.startswith("research_proxy")
    assert payload["AAPL"].guidance_signal == "supportive"
    assert payload["XOM"].source.startswith("research_proxy")
    assert payload["XOM"].guidance_signal == "softening"
    rendered = service._format_management_commentary_lines(payload)
    assert "source=research_proxy" in rendered


@pytest.mark.asyncio
async def test_recommendation_service_sector_rotation_intraday_persistence_tracks_rounds(tmp_path: Path) -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    store = SectorRotationStateStore(path=tmp_path / "sector_rotation_state.json")
    market_data_client = FakeMarketDataClient()  # type: ignore[assignment]
    news_client = FakeNewsClient()  # type: ignore[assignment]

    first_alerts = await service.generate_sector_rotation_alerts(
        news_client=news_client,  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=None,
        sector_rotation_state_store=store,
        min_streak=2,
    )
    second_alerts = await service.generate_sector_rotation_alerts(
        news_client=news_client,  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=None,
        sector_rotation_state_store=store,
        min_streak=2,
    )

    assert first_alerts
    assert any("delta" in alert.text for alert in second_alerts)
    assert any("Action" in alert.text for alert in second_alerts)


@pytest.mark.asyncio
async def test_recommendation_service_emits_market_internals_break_alert() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    alerts = await service.generate_sector_rotation_alerts(
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FragileBreadthMarketDataClient(),  # type: ignore[arg-type]
        research_client=None,
    )

    assert any(alert.key.startswith("market:internals-break:") for alert in alerts)
    assert any("Action" in alert.text for alert in alerts)


@pytest.mark.asyncio
async def test_recommendation_service_generates_pre_earnings_risk_alerts() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    stock_candidates = await service._screen_stock_universe(  # type: ignore[attr-defined]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        top_k=3,
    )
    alerts = await service.generate_pre_earnings_risk_alerts(
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        watchlist=("AAPL",),
        top_candidates=stock_candidates,
        days_ahead=7,
    )

    assert any(alert.text.startswith("🟠 ระวัง | ความเสี่ยงก่อนประกาศงบ") for alert in alerts)
    assert any("revenue est growth" in alert.text.casefold() for alert in alerts)
    assert any("forward pe" in alert.text.casefold() for alert in alerts)
    assert any("breadth" in alert.text.casefold() for alert in alerts)
    assert all("- Action:" in alert.text for alert in alerts)


@pytest.mark.asyncio
async def test_recommendation_service_periodic_report_exposes_daily_and_intraday_persistence(tmp_path: Path) -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    store = SectorRotationStateStore(path=tmp_path / "sector_rotation_state.json")
    market_data_client = FakeMarketDataClient()  # type: ignore[assignment]
    news_client = FakeNewsClient()  # type: ignore[assignment]

    await service.generate_sector_rotation_alerts(
        news_client=news_client,  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=None,
        sector_rotation_state_store=store,
        min_streak=2,
    )
    report = await service.generate_periodic_report(
        report_kind="closing",
        news_client=news_client,  # type: ignore[arg-type]
        market_data_client=market_data_client,  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        sector_rotation_state_store=store,
        sector_rotation_min_streak=1,
        earnings_result_lookback_days=7,
        news_limit=2,
        history_period="6mo",
        history_interval="1d",
        history_limit=60,
    )

    assert report.fallback_used is True
    assert "Sector Persistence Daily" in report.recommendation_text
    assert "Sector Persistence Intraday" in report.recommendation_text
    assert "Market Breadth Diffusion" in report.recommendation_text
    assert "Market Breadth Trend" in report.recommendation_text
    assert "Index Leadership Divergence" in report.recommendation_text
    assert "Sector Breadth" in report.recommendation_text
    assert "Sector Breadth Trend" in report.recommendation_text
    assert "Post-Earnings Read-Through" in report.recommendation_text


@pytest.mark.asyncio
async def test_recommendation_service_periodic_report_includes_macro_allocation_and_memory(tmp_path: Path) -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    sector_store = SectorRotationStateStore(path=tmp_path / "sector_rotation_state.json")
    memory_store = ReportMemoryStore(path=tmp_path / "report_memory.json")
    memory_store.remember(report_kind="morning", summary="Morning | breadth broad rally | sector leader Technology")

    report = await service.generate_periodic_report(
        report_kind="midday",
        news_client=FakeNewsClient(),  # type: ignore[arg-type]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        research_client=FakeResearchClient(),  # type: ignore[arg-type]
        sector_rotation_state_store=sector_store,
        report_memory_store=memory_store,
        news_limit=2,
        history_period="6mo",
        history_interval="1d",
        history_limit=60,
    )

    assert report.fallback_used is True
    assert "Macro Regime" in report.recommendation_text
    assert "Allocation Mix" in report.recommendation_text
    assert "Daily Narrative Memory" in report.recommendation_text
    assert "Pre-Earnings Setup Ranking" in report.recommendation_text
    stored = memory_store.get_day_entries()
    assert "midday" in stored


@pytest.mark.asyncio
async def test_recommendation_service_generates_earnings_setup_alerts() -> None:
    service = RecommendationService(DummyLLM(), default_investor_profile="balanced")
    stock_candidates = await service._screen_stock_universe(  # type: ignore[attr-defined]
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        top_k=3,
    )
    alerts = await service.generate_earnings_setup_alerts(
        market_data_client=FakeMarketDataClient(),  # type: ignore[arg-type]
        top_candidates=stock_candidates,
        days_ahead=7,
    )

    assert alerts
    assert any(
        alert.text.startswith("✅ ยืนยัน | Setup ก่อนงบ")
        or alert.text.startswith("🔎 จับตา | Setup ก่อนงบ")
        for alert in alerts
    )
    assert any("valuation" in alert.text.casefold() for alert in alerts)
    assert any("expectations" in alert.text.casefold() for alert in alerts)
    assert all("- Action:" in alert.text for alert in alerts)
