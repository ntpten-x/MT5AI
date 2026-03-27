from __future__ import annotations

from pathlib import Path

import pytest

from invest_advisor_bot.bot.ai_simulated_portfolio_state import AISimulatedPortfolioStateStore
from invest_advisor_bot.providers.market_data_client import AssetQuote
from invest_advisor_bot.services.ai_simulated_portfolio import AISimulatedPortfolioService
from invest_advisor_bot.services.recommendation_service import RecommendationResult


class _FakeRecommendationService:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def generate_market_update(self, **_: object) -> RecommendationResult:
        return RecommendationResult(
            recommendation_text="market update",
            model=None,
            system_prompt_path="test",
            input_payload=dict(self.payload),
            fallback_used=True,
        )


class _FakeMarketDataClient:
    def __init__(self, quotes: dict[str, AssetQuote]) -> None:
        self.quotes = quotes

    async def get_latest_prices(self, tickers: list[str] | tuple[str, ...]) -> dict[str, AssetQuote | None]:
        return {ticker: self.quotes.get(ticker) for ticker in tickers}


def _quote(ticker: str, price: float) -> AssetQuote:
    return AssetQuote(
        ticker=ticker,
        name=ticker,
        exchange="TEST",
        currency="USD",
        price=price,
        previous_close=price * 0.99,
        open_price=price * 0.995,
        day_high=price * 1.01,
        day_low=price * 0.99,
        volume=1_000_000,
        timestamp=None,
    )


def test_ai_simulated_portfolio_state_store_round_trip(tmp_path: Path) -> None:
    store = AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json")
    state = store.ensure_portfolio("system", starting_cash=1000.0)
    assert state.cash == 1000.0
    assert state.holdings == ()

    saved = store.save_portfolio(
        "system",
        starting_cash=1000.0,
        cash=700.0,
        realized_pnl=25.0,
        holdings=[{"ticker": "SPY", "quantity": 2.5, "avg_cost": 100.0, "label": "SPY ETF", "asset_type": "etf"}],
        last_rebalanced_at=None,
        last_action_summary="BUY SPY",
        metadata={"engine": "test"},
    )
    assert saved.cash == 700.0
    assert saved.holdings[0].normalized_ticker == "SPY"

    store.append_trades(
        "system",
        [
            {
                "trade_id": "trade-1",
                "action": "buy",
                "ticker": "SPY",
                "quantity": 2.5,
                "price": 100.0,
                "notional": 250.0,
                "occurred_at": "2026-03-27T00:00:00+00:00",
            }
        ],
    )
    trades = store.list_trades("system")
    assert len(trades) == 1
    assert trades[0].ticker == "SPY"

    reloaded = AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json")
    restored = reloaded.get_portfolio("system")
    assert restored is not None
    assert restored.last_action_summary == "BUY SPY"
    assert restored.holdings[0].normalized_ticker == "SPY"


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_rebalance_buys_candidates(tmp_path: Path) -> None:
    payload = {
        "stock_picks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "stance": "buy",
                "confidence_score": 0.84,
                "confidence_label": "high",
                "coverage_score": 0.88,
                "coverage_label": "high",
                "score": 8.2,
            }
        ],
        "asset_snapshots": [
            {
                "ticker": "SPY",
                "label": "S&P 500 ETF",
                "trend": "uptrend",
                "trend_score": 4.0,
                "coverage_score": 0.82,
                "coverage_label": "high",
            }
        ],
        "market_confidence": {"score": 0.72, "label": "high"},
    }
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService(payload),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({"AAPL": _quote("AAPL", 100.0), "SPY": _quote("SPY", 50.0)}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json"),
        starting_cash_usd=1000.0,
        max_positions=4,
        max_position_pct=0.30,
        min_cash_pct=0.10,
        min_trade_notional_usd=25.0,
        rebalance_interval_minutes=60,
    )

    result = await service.maybe_rebalance(conversation_key="system", reason="test", force=True)

    assert result.action_count >= 1
    assert any(trade.action == "buy" for trade in result.trades)
    assert result.snapshot["position_count"] >= 1
    assert result.snapshot["cash"] >= 100.0
    assert result.snapshot["total_value"] > 0
    assert "AI Paper Portfolio" in result.rendered_summary
    assert "การตัดสินใจรอบล่าสุด" in result.rendered_summary


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_cooldown_skips_rebalance(tmp_path: Path) -> None:
    payload = {
        "stock_picks": [],
        "asset_snapshots": [],
        "market_confidence": {"score": 0.40, "label": "low"},
    }
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService(payload),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json"),
        starting_cash_usd=1000.0,
        rebalance_interval_minutes=360,
    )

    await service.maybe_rebalance(conversation_key="system", reason="first", force=True)
    second = await service.maybe_rebalance(conversation_key="system", reason="second", force=False)
    assert second.skipped_reason == "cooldown_active"


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_abstains_new_stock_entries_when_market_confidence_is_low(tmp_path: Path) -> None:
    payload = {
        "stock_picks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "stance": "buy",
                "confidence_score": 0.90,
                "coverage_score": 0.90,
                "score": 8.8,
            }
        ],
        "asset_snapshots": [
            {
                "ticker": "GLD",
                "label": "Gold ETF",
                "trend": "uptrend",
                "trend_score": 3.2,
                "coverage_score": 0.84,
            }
        ],
        "market_confidence": {"score": 0.20, "label": "very_low"},
    }
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService(payload),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({"AAPL": _quote("AAPL", 100.0), "GLD": _quote("GLD", 50.0)}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json"),
        starting_cash_usd=1000.0,
        profile_name="balanced",
    )

    result = await service.maybe_rebalance(conversation_key="system", reason="risk_off", force=True)

    tickers = {item["ticker"] for item in result.snapshot["holdings"]}
    assert "GLD" in tickers
    assert "AAPL" not in tickers
    assert "โหมดป้องกันสูง" in str(result.snapshot["last_action_summary"])


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_tracks_attribution_and_reentry_cooldown(tmp_path: Path) -> None:
    payload_buy = {
        "stock_picks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "stance": "buy",
                "confidence_score": 0.84,
                "coverage_score": 0.88,
                "score": 8.2,
            }
        ],
        "asset_snapshots": [],
        "market_confidence": {"score": 0.74, "label": "high"},
    }
    store = AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json")
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService(payload_buy),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({"AAPL": _quote("AAPL", 100.0)}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=store,
        starting_cash_usd=1000.0,
        profile_name="balanced",
    )

    await service.maybe_rebalance(conversation_key="system", reason="buy", force=True)

    state = store.get_portfolio("system")
    assert state is not None
    holding = state.holdings[0]
    cooled_state = store.save_portfolio(
        "system",
        starting_cash=state.starting_cash,
        cash=state.cash,
        realized_pnl=state.realized_pnl,
        holdings=[
            {
                "ticker": holding.normalized_ticker,
                "quantity": holding.quantity,
                "avg_cost": holding.avg_cost,
                "label": holding.label,
                "asset_type": holding.asset_type,
                "last_reason": holding.last_reason,
                "opened_at": "2026-03-20T00:00:00+00:00",
            }
        ],
        last_rebalanced_at=state.last_rebalanced_at,
        last_action_summary=state.last_action_summary,
        metadata=state.metadata,
    )
    assert cooled_state.holdings[0].opened_at is not None

    payload_sell = {
        "stock_picks": [],
        "asset_snapshots": [],
        "market_confidence": {"score": 0.20, "label": "very_low"},
    }
    service.recommendation_service = _FakeRecommendationService(payload_sell)  # type: ignore[assignment]
    service.market_data_client = _FakeMarketDataClient({"AAPL": _quote("AAPL", 120.0)})  # type: ignore[assignment]
    await service.maybe_rebalance(conversation_key="system", reason="sell", force=True)

    payload_rebuy = {
        "stock_picks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "stance": "buy",
                "confidence_score": 0.90,
                "coverage_score": 0.90,
                "score": 9.0,
            }
        ],
        "asset_snapshots": [],
        "market_confidence": {"score": 0.82, "label": "high"},
    }
    service.recommendation_service = _FakeRecommendationService(payload_rebuy)  # type: ignore[assignment]
    service.market_data_client = _FakeMarketDataClient({"AAPL": _quote("AAPL", 121.0)})  # type: ignore[assignment]
    result = await service.maybe_rebalance(conversation_key="system", reason="rebuy", force=True)

    assert not any(item["ticker"] == "AAPL" for item in result.snapshot["holdings"])
    assert any(item["ticker"] == "AAPL" for item in result.snapshot["attribution"])


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_render_texts_use_thai_paper_ui(tmp_path: Path) -> None:
    payload = {
        "stock_picks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "stance": "buy",
                "confidence_score": 0.84,
                "coverage_score": 0.88,
                "score": 8.2,
            }
        ],
        "asset_snapshots": [],
        "market_confidence": {"score": 0.72, "label": "high"},
    }
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService(payload),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({"AAPL": _quote("AAPL", 100.0)}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json"),
        starting_cash_usd=1000.0,
    )

    await service.maybe_rebalance(conversation_key="system", reason="buy", force=True)

    portfolio_text = await service.render_portfolio_text(conversation_key="system")
    trades_text = await service.render_trades_text(conversation_key="system")
    performance_text = await service.render_performance_text(conversation_key="system")

    assert "📘 AI Paper Portfolio" in portfolio_text
    assert "💼 บัญชี" in portfolio_text
    assert "📦 สินทรัพย์ที่ถือ" in portfolio_text
    assert "📒 สมุดรายการ AI Paper Portfolio" in trades_text
    assert "📊 ผลการดำเนินงาน AI Paper Portfolio" in performance_text
    assert "📈 Attribution" in performance_text


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_backfills_trade_history_from_existing_holdings(tmp_path: Path) -> None:
    store = AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json")
    state = store.ensure_portfolio("system", starting_cash=1000.0)
    store.save_portfolio(
        "system",
        starting_cash=state.starting_cash,
        cash=500.0,
        realized_pnl=0.0,
        holdings=[
            {
                "ticker": "GLD",
                "quantity": 1.25,
                "avg_cost": 200.0,
                "label": "Gold ETF",
                "asset_type": "gold",
                "last_reason": "ถือป้องกันความเสี่ยง",
                "opened_at": "2026-03-20T00:00:00+00:00",
            }
        ],
        last_rebalanced_at=None,
        last_action_summary="ไม่มีการซื้อขาย",
        metadata={"profile_name": "growth", "allowed_asset_types": ["stock", "etf", "gold"]},
    )
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService({"stock_picks": [], "asset_snapshots": [], "market_confidence": {"score": 0.5}}),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({"GLD": _quote("GLD", 201.0)}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=store,
        starting_cash_usd=1000.0,
    )

    trades_text = await service.render_trades_text(conversation_key="system")
    trades = store.list_trades("system", limit=20)

    assert trades
    assert trades[0].ticker == "GLD"
    assert trades[0].detail and trades[0].detail.get("synthetic_backfill") is True
    assert "GLD" in trades_text
    assert "backfill" in trades_text


@pytest.mark.asyncio
async def test_ai_simulated_portfolio_renders_trade_alerts_and_daily_digest(tmp_path: Path) -> None:
    payload = {
        "stock_picks": [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "stance": "buy",
                "confidence_score": 0.84,
                "coverage_score": 0.88,
                "score": 8.2,
            }
        ],
        "asset_snapshots": [],
        "market_confidence": {"score": 0.72, "label": "high"},
    }
    service = AISimulatedPortfolioService(
        recommendation_service=_FakeRecommendationService(payload),  # type: ignore[arg-type]
        market_data_client=_FakeMarketDataClient({"AAPL": _quote("AAPL", 100.0)}),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        state_store=AISimulatedPortfolioStateStore(path=tmp_path / "ai_simulated_portfolio.json"),
        starting_cash_usd=1000.0,
    )

    result = await service.maybe_rebalance(conversation_key="system", reason="buy", force=True)
    alerts = service.render_trade_alert_texts(snapshot=result.snapshot, trades=result.trades)
    digest = service.render_daily_digest_text(snapshot=result.snapshot, report_kind="closing")

    assert alerts
    assert "AI ซื้อ" in alerts[0]
    assert "พอร์ตตอนนี้" in alerts[0]
    assert "Closing Digest" in digest
