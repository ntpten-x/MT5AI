from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from invest_advisor_bot.analysis.technical_indicators import SupportResistanceLevels
from invest_advisor_bot.bot.portfolio_state import PortfolioHolding
from invest_advisor_bot.analysis.trend_engine import TrendAssessment
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.providers.market_data_client import AssetQuote, StockFundamentals
from invest_advisor_bot.providers.news_client import NewsArticle
from invest_advisor_bot.providers.research_client import ResearchFinding
from invest_advisor_bot.services.recommendation_service import RecommendationService
from invest_advisor_bot.universe import StockUniverseMember


class DummyLLM:
    async def generate_text(self, *args, **kwargs):
        return None


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
        return {"vix": 19.5, "tnx": 4.2, "cpi_yoy": 3.1}

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
    assert "Sector Rotation" in result.recommendation_text
    assert "Sector Persistence Daily" in result.recommendation_text
    assert "Sector Persistence Intraday" in result.recommendation_text
    assert "Market Breadth Diffusion" in result.recommendation_text
    assert "Market Breadth Trend" in result.recommendation_text
    assert "Index Leadership Divergence" in result.recommendation_text
    assert "Sector Breadth" in result.recommendation_text
    assert "Sector Breadth Trend" in result.recommendation_text
    assert "Post-Earnings Read-Through" in result.recommendation_text


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
    assert any(item["ticker"] == "VOO" for item in result.input_payload["portfolio_snapshot"]["holdings"])


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
    assert any(
        alert.text.startswith("🔎 จับตา | สินทรัพย์เด่น")
        or alert.text.startswith("🟠 ระวัง | สินทรัพย์อ่อนแรง")
        or alert.text.startswith("✅ ยืนยัน | ข่าวหนุนโอกาส")
        or alert.text.startswith("🟠 ระวัง | ข่าวมหภาค")
        for alert in alerts
    )
    assert all("- Action:" in alert.text for alert in alerts)


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
