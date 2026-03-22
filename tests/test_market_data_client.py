from __future__ import annotations

from datetime import datetime, timezone

import pytest

from invest_advisor_bot.providers.market_data_client import AssetQuote, MarketDataClient, OhlcvBar


@pytest.mark.asyncio
async def test_market_data_client_caches_latest_price_and_history() -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    calls = {"latest": 0, "history": 0, "macro": 0}

    def fake_latest(ticker: str) -> AssetQuote:
        calls["latest"] += 1
        return AssetQuote(
            ticker=ticker,
            name=ticker,
            currency="USD",
            exchange="TEST",
            price=100.0,
            previous_close=99.0,
            open_price=99.5,
            day_high=101.0,
            day_low=98.5,
            volume=1_000,
            timestamp=datetime.now(timezone.utc),
        )

    def fake_history(ticker: str, period: str, interval: str, limit: int | None) -> list[OhlcvBar]:
        calls["history"] += 1
        return [
            OhlcvBar(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc),
                open=99.0,
                high=101.0,
                low=98.0,
                close=100.0,
                volume=1_000,
            )
        ]

    def fake_macro() -> dict[str, float | None]:
        calls["macro"] += 1
        return {"vix": 18.0, "tnx": 4.2, "cpi_yoy": 2.9}

    client._get_latest_price_sync = fake_latest  # type: ignore[method-assign]
    client._get_history_sync = fake_history  # type: ignore[method-assign]
    client._get_macro_context_sync = fake_macro  # type: ignore[method-assign]

    await client.get_latest_price("SPY")
    await client.get_latest_price("SPY")
    await client.get_history("SPY")
    await client.get_history("SPY")
    await client.get_macro_context()
    await client.get_macro_context()

    assert calls == {"latest": 1, "history": 1, "macro": 1}


@pytest.mark.asyncio
async def test_market_data_client_prefers_alpha_vantage_for_latest_price_when_configured() -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        alpha_vantage_api_key="demo",
        provider_order=("alpha_vantage", "yfinance"),
    )
    calls = {"alpha": 0, "yfinance": 0}

    def fake_alpha_vantage(ticker: str) -> AssetQuote:
        calls["alpha"] += 1
        return AssetQuote(
            ticker=ticker,
            name=ticker,
            currency="USD",
            exchange="AV",
            price=101.0,
            previous_close=100.0,
            open_price=100.5,
            day_high=102.0,
            day_low=99.5,
            volume=500,
            timestamp=datetime.now(timezone.utc),
        )

    def fake_yfinance(ticker: str) -> AssetQuote:
        calls["yfinance"] += 1
        return AssetQuote(
            ticker=ticker,
            name=ticker,
            currency="USD",
            exchange="YF",
            price=99.0,
            previous_close=98.0,
            open_price=98.5,
            day_high=100.0,
            day_low=97.5,
            volume=400,
            timestamp=datetime.now(timezone.utc),
        )

    client._get_latest_price_alpha_vantage_sync = fake_alpha_vantage  # type: ignore[method-assign]
    client._get_latest_price_yfinance_sync = fake_yfinance  # type: ignore[method-assign]

    quote = await client.get_latest_price("SPY")

    assert quote is not None
    assert quote.exchange == "AV"
    assert calls == {"alpha": 1, "yfinance": 0}


@pytest.mark.asyncio
async def test_market_data_client_falls_back_to_yfinance_history_when_alpha_vantage_has_no_data() -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        alpha_vantage_api_key="demo",
        provider_order=("alpha_vantage", "yfinance"),
    )
    calls = {"alpha": 0, "yfinance": 0}

    def fake_alpha_vantage(ticker: str, *, period: str, interval: str, limit: int | None) -> list[OhlcvBar]:
        calls["alpha"] += 1
        return []

    def fake_yfinance(ticker: str, period: str, interval: str, limit: int | None) -> list[OhlcvBar]:
        calls["yfinance"] += 1
        return [
            OhlcvBar(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc),
                open=10.0,
                high=11.0,
                low=9.5,
                close=10.5,
                volume=1_000,
            )
        ]

    client._get_history_alpha_vantage_sync = fake_alpha_vantage  # type: ignore[method-assign]
    client._get_history_yfinance_sync = fake_yfinance  # type: ignore[method-assign]

    bars = await client.get_history("SPY", period="6mo", interval="1d", limit=30)

    assert len(bars) == 1
    assert calls == {"alpha": 1, "yfinance": 1}
