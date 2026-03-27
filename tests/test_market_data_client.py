from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from invest_advisor_bot.providers.market_data_client import AssetQuote, MacroEvent, MarketDataClient, OhlcvBar, OptionContractSnapshot


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


def test_market_data_client_status_reports_provider_configuration() -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        polygon_api_key="polygon-demo",
        alpha_vantage_api_key="demo",
        fred_api_key="fred-demo",
        provider_order=("polygon", "alpha_vantage", "yfinance"),
        http_timeout_seconds=14.0,
    )

    status = client.status()

    assert status["available"] is True
    assert status["provider_order"] == ["polygon", "alpha_vantage", "yfinance"]
    assert status["configured_sources"]["polygon"] is True
    assert status["configured_sources"]["alpha_vantage"] is True
    assert status["configured_sources"]["fred"] is True
    assert status["configured_sources"]["yfinance"] is True
    assert status["http_timeout_seconds"] == 14.0


@pytest.mark.asyncio
async def test_market_data_client_prefers_polygon_for_latest_price_when_configured() -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        polygon_api_key="polygon-demo",
        provider_order=("polygon", "yfinance"),
    )
    calls = {"polygon": 0, "yfinance": 0}

    def fake_polygon(ticker: str) -> AssetQuote:
        calls["polygon"] += 1
        return AssetQuote(
            ticker=ticker,
            name=ticker,
            currency="USD",
            exchange="POLYGON",
            price=102.0,
            previous_close=101.0,
            open_price=101.2,
            day_high=103.0,
            day_low=100.5,
            volume=900,
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

    client._get_latest_price_polygon_sync = fake_polygon  # type: ignore[method-assign]
    client._get_latest_price_yfinance_sync = fake_yfinance  # type: ignore[method-assign]

    quote = await client.get_latest_price("SPY")

    assert quote is not None
    assert quote.exchange == "POLYGON"
    assert calls == {"polygon": 1, "yfinance": 0}


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


def test_market_data_client_macro_context_includes_extended_macro_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.fast_info = {
                "lastPrice": {
                    "^VIX": 17.5,
                    "^TNX": 4.33,
                }[ticker]
            }

    monkeypatch.setattr("invest_advisor_bot.providers.market_data_client.yf.Ticker", FakeTicker)
    monkeypatch.setattr(
        client,
        "_fetch_fred_series_frame",
        lambda series_id: pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=13, freq="MS", tz="UTC"),
                "value": {
                    "CPIAUCSL": [300 + index for index in range(13)],
                    "CPILFESL": [250 + (index * 0.8) for index in range(13)],
                    "T10Y2Y": [-0.25 + (index * 0.0) for index in range(13)],
                    "BAMLH0A0HYM2": [4.8 for _ in range(13)],
                    "MORTGAGE30US": [6.7 for _ in range(13)],
                    "NFCI": [0.25 for _ in range(13)],
                }[series_id],
            }
        ),
    )
    monkeypatch.setattr(client, "_fetch_bls_macro_snapshot", lambda: {"unemployment_rate": 4.1, "payrolls_mom_k": 110.0})
    monkeypatch.setattr(client, "_fetch_treasury_macro_snapshot", lambda: {"operating_cash_balance_b": 620.0})
    monkeypatch.setattr(client, "_fetch_eia_macro_snapshot", lambda: {"wti_usd": 82.0})

    context = client._get_macro_context_sync()

    assert context["vix"] == 17.5
    assert context["tnx"] == 4.33
    assert context["cpi_yoy"] == 4.0
    assert context["yield_spread_10y_2y"] == -0.25
    assert context["high_yield_spread"] == 4.8
    assert context["unemployment_rate"] == 4.1
    assert context["wti_usd"] == 82.0


def test_market_data_client_macro_intelligence_includes_new_data_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        bea_api_key="demo",
        cme_fedwatch_api_url="https://example.test/fedwatch",
        nasdaq_data_link_api_key="nasdaq-demo",
        nasdaq_data_link_datasets=("NDAQ/FACTOR1",),
    )

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.fast_info = {"lastPrice": {"^VIX": 18.2, "^TNX": 4.1}[ticker]}

    monkeypatch.setattr("invest_advisor_bot.providers.market_data_client.yf.Ticker", FakeTicker)
    monkeypatch.setattr(
        client,
        "_fetch_fred_series_frame",
        lambda series_id: pd.DataFrame({"date": pd.date_range("2024-01-01", periods=13, freq="MS", tz="UTC"), "value": [300 + index for index in range(13)]}),
    )
    monkeypatch.setattr(client, "_fetch_bls_macro_snapshot", lambda: {"unemployment_rate": 4.3, "payrolls_mom_k": 95.0})
    monkeypatch.setattr(client, "_fetch_bea_macro_snapshot", lambda: {"core_pce_yoy": 3.0, "gdp_qoq_annualized": 1.1})
    monkeypatch.setattr(client, "_fetch_treasury_macro_snapshot", lambda: {"operating_cash_balance_b": 450.0})
    monkeypatch.setattr(client, "_fetch_eia_macro_snapshot", lambda: {"gasoline_usd_gal": 3.8})
    monkeypatch.setattr(client, "_fetch_alfred_revision_snapshot", lambda: {"alfred_payroll_revision_k": 55.0})
    monkeypatch.setattr(client, "_fetch_cftc_cot_snapshot", lambda: {"cftc_equity_net_pct_oi": 18.0})
    monkeypatch.setattr(client, "_fetch_fed_qualitative_snapshot", lambda: {"fed_tone": "hawkish", "speech_titles": ["Inflation remains too high"]})
    monkeypatch.setattr(client, "_fetch_finra_short_volume_snapshot", lambda: {"finra_spy_short_volume_ratio": 0.58})
    monkeypatch.setattr(
        client,
        "_fetch_cme_fedwatch_snapshot",
        lambda: {
            "fedwatch_next_meeting_cut_prob_pct": 64.0,
            "fedwatch_easing_12m_prob_pct": 78.0,
            "fedwatch_target_rate_mid_pct": 4.625,
            "next_meeting": "2026-05-06",
        },
    )
    monkeypatch.setattr(
        client,
        "_fetch_nasdaq_data_link_snapshot",
        lambda: {"NDAQ/FACTOR1": {"date": "2026-03-20", "value": 1.23, "value_column": "Close"}},
    )

    intelligence = client._get_macro_intelligence_sync()

    assert "bea" in intelligence["sources_used"]
    assert "alfred" in intelligence["sources_used"]
    assert "cftc" in intelligence["sources_used"]
    assert "fed_qualitative" in intelligence["sources_used"]
    assert "finra" in intelligence["sources_used"]
    assert "cme_fedwatch" in intelligence["sources_used"]
    assert "nasdaq_data_link" in intelligence["sources_used"]
    assert intelligence["metrics"]["core_pce_yoy"] == 3.0
    assert intelligence["metrics"]["alfred_payroll_revision_k"] == 55.0
    assert intelligence["metrics"]["fedwatch_next_meeting_cut_prob_pct"] == 64.0
    assert intelligence["positioning"]["cftc_equity_net_pct_oi"] == 18.0
    assert intelligence["qualitative"]["fed_tone"] == "hawkish"
    assert intelligence["short_flow"]["finra_spy_short_volume_ratio"] == 0.58
    assert intelligence["fedwatch"]["next_meeting"] == "2026-05-06"
    assert intelligence["structured_macro"]["NDAQ/FACTOR1"]["value"] == 1.23


def test_market_data_client_macro_intelligence_includes_ex_us_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        ecb_series_map="inflation_yoy=ICP/M.U2.N.000000.4.ANR",
        imf_api_base_url="https://example.test/imf/{series}",
        imf_series_map="global_growth_pct=NGDP_RPCH",
        world_bank_indicator_map="gdp_growth=NY.GDP.MKTP.KD.ZG",
        world_bank_countries=("EMU", "JPN"),
    )

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.fast_info = {"lastPrice": {"^VIX": 18.2, "^TNX": 4.1}[ticker]}

    monkeypatch.setattr("invest_advisor_bot.providers.market_data_client.yf.Ticker", FakeTicker)
    monkeypatch.setattr(
        client,
        "_fetch_fred_series_frame",
        lambda series_id: pd.DataFrame({"date": pd.date_range("2024-01-01", periods=13, freq="MS", tz="UTC"), "value": [300 + index for index in range(13)]}),
    )
    monkeypatch.setattr(client, "_fetch_bls_macro_snapshot", lambda: {"unemployment_rate": 4.3, "payrolls_mom_k": 95.0})
    monkeypatch.setattr(client, "_fetch_bea_macro_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_treasury_macro_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_eia_macro_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_alfred_revision_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_cftc_cot_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_fed_qualitative_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_finra_short_volume_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_cme_fedwatch_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_nasdaq_data_link_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_gdelt_global_event_snapshot", lambda: {})
    monkeypatch.setattr(client, "_fetch_ecb_ex_us_macro_snapshot", lambda: {"ecb_inflation_yoy": 2.9})
    monkeypatch.setattr(client, "_fetch_imf_ex_us_macro_snapshot", lambda: {"imf_global_growth_pct": 2.8})
    monkeypatch.setattr(client, "_fetch_world_bank_ex_us_macro_snapshot", lambda: {"world_bank_gdp_growth": 2.2})

    intelligence = client._get_macro_intelligence_sync()

    assert "ecb" in intelligence["sources_used"]
    assert "imf" in intelligence["sources_used"]
    assert "world_bank" in intelligence["sources_used"]
    assert intelligence["ex_us_macro"]["ecb_inflation_yoy"] == 2.9
    assert intelligence["ex_us_macro"]["world_bank_gdp_growth"] == 2.2
    assert "eurozone_inflation_sticky" in intelligence["signals"]
    assert "global_growth_softening" in intelligence["signals"]


@pytest.mark.asyncio
async def test_market_data_client_option_chain_snapshot_uses_polygon_cache() -> None:
    client = MarketDataClient(cache_ttl_seconds=900, polygon_api_key="polygon-demo")
    calls = {"options": 0}

    def fake_option_chain(
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        limit: int,
    ) -> list[OptionContractSnapshot]:
        calls["options"] += 1
        return [
            OptionContractSnapshot(
                contract_ticker=f"{ticker}240621C00100000",
                underlying_ticker=ticker,
                contract_type=contract_type or "call",
                expiration_date=expiration_date or "2024-06-21",
                strike_price=100.0,
                bid=2.1,
                ask=2.3,
                midpoint=2.2,
                last_price=2.25,
                implied_volatility=0.24,
                open_interest=1000,
                volume=200,
                updated_at=datetime.now(timezone.utc),
            )
        ]

    client._get_option_chain_polygon_sync = fake_option_chain  # type: ignore[method-assign]

    first = await client.get_option_chain_snapshot("SPY", expiration_date="2024-06-21", contract_type="call", limit=5)
    second = await client.get_option_chain_snapshot("SPY", expiration_date="2024-06-21", contract_type="call", limit=5)

    assert len(first) == 1
    assert first[0].contract_ticker == "SPY240621C00100000"
    assert second[0].midpoint == 2.2
    assert calls["options"] == 1



def test_market_data_client_company_intelligence_parses_sec_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    filings_payload = {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "10-K"],
                "filingDate": ["2026-02-01", "2026-01-15", "2025-11-01"],
                "reportDate": ["2025-12-31", "2026-01-15", "2025-09-30"],
                "accessionNumber": ["0000000001-26-000001", "0000000001-26-000002", "0000000001-25-000003"],
                "primaryDocument": ["q.htm", "8k.htm", "10k.htm"],
            }
        }
    }
    companyfacts_payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "end": "2025-12-31", "filed": "2026-02-01", "val": 1200},
                            {"form": "10-Q", "end": "2024-12-31", "filed": "2025-02-01", "val": 1000},
                        ]
                    }
                },
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "end": "2025-12-31", "filed": "2026-02-01", "val": 300},
                            {"form": "10-Q", "end": "2024-12-31", "filed": "2025-02-01", "val": 240},
                        ]
                    }
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "end": "2025-12-31", "filed": "2026-02-01", "val": 40},
                            {"form": "10-Q", "end": "2024-12-31", "filed": "2025-02-01", "val": 35},
                        ]
                    }
                },
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "end": "2025-12-31", "filed": "2026-02-01", "val": 500},
                            {"form": "10-Q", "end": "2024-12-31", "filed": "2025-02-01", "val": 550},
                        ]
                    }
                },
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"form": "10-Q", "end": "2025-12-31", "filed": "2026-02-01", "val": 100},
                            {"form": "10-Q", "end": "2024-12-31", "filed": "2025-02-01", "val": 98},
                        ]
                    }
                },
                "RestructuringCharges": {
                    "units": {
                        "USD": [
                            {"form": "10-Q", "end": "2025-12-31", "filed": "2026-02-01", "val": 18},
                        ]
                    }
                },
            }
        }
    }

    monkeypatch.setattr(client, "_load_sec_ticker_mapping", lambda: {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}})
    monkeypatch.setattr(
        client,
        "_fetch_sec_json",
        lambda url: filings_payload if "submissions" in url else companyfacts_payload if "companyfacts" in url else None,
    )
    monkeypatch.setattr(client, "_infer_guidance_signal_from_filings", lambda filings: "positive")
    monkeypatch.setattr(client, "_fetch_market_alternative_signals", lambda ticker: ("accumulating", "bullish"))
    monkeypatch.setattr(client, "_infer_expectation_signal", lambda ticker: "supportive")
    monkeypatch.setattr(client, "_get_analyst_rating_profile_sync", lambda ticker: None)
    monkeypatch.setattr(client, "_get_insider_transaction_summary_sync", lambda ticker: None)
    monkeypatch.setattr(client, "_get_corporate_actions_sync", lambda ticker, lookback_days: [])

    intelligence = client._get_company_intelligence_sync("AAPL", "Apple Inc.")

    assert intelligence is not None
    assert intelligence.cik == "0000320193"
    assert intelligence.revenue_yoy_pct == 20.0
    assert intelligence.free_cash_flow_latest == 260.0
    assert intelligence.debt_delta_pct == -9.09
    assert intelligence.guidance_signal == "positive"
    assert intelligence.insider_signal == "accumulating"


def test_market_data_client_macro_event_calendar_combines_fred_and_fomc(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    now = datetime.now(timezone.utc)
    fred_event = MacroEvent(
        event_key="cpi",
        event_name="CPI",
        category="inflation",
        source="fred_release_calendar",
        scheduled_at=now + pd.Timedelta(days=2),
        importance="high",
        status="scheduled",
    )
    fomc_event = MacroEvent(
        event_key="fomc",
        event_name="FOMC",
        category="policy",
        source="federal_reserve",
        scheduled_at=now + pd.Timedelta(days=5),
        importance="critical",
        status="scheduled",
    )

    monkeypatch.setattr(client, "_fetch_macro_release_events_from_fred", lambda *, days_ahead: [fomc_event, fred_event])
    monkeypatch.setattr(client, "_fetch_fomc_events", lambda *, days_ahead: [])
    monkeypatch.setattr(client, "_fetch_global_macro_calendar_events", lambda *, days_ahead: [])

    events = client._get_macro_event_calendar_sync(30)

    assert [item.event_key for item in events] == ["cpi", "fomc"]


def test_market_data_client_guidance_signal_reads_8k_exhibits(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    filing = type(
        "Filing",
        (),
        {
            "primary_document_url": "https://www.sec.gov/Archives/example/8k.htm",
            "form": "8-K",
        },
    )()

    monkeypatch.setattr(client, "_fetch_sec_document_text", lambda url: "management discussed quarterly results")
    monkeypatch.setattr(
        client,
        "_fetch_sec_exhibit_texts",
        lambda filing_url, filing_text: ["company raised full-year guidance and improved outlook for revenue"],
    )

    signal = client._infer_guidance_signal_from_filings([filing])

    assert signal == "positive"


def test_market_data_client_analyst_rating_profile_uses_finnhub_consensus(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900, finnhub_api_key="demo")

    monkeypatch.setattr(
        client,
        "_fetch_finnhub_json",
        lambda endpoint, *, symbol: [{"strongBuy": 6, "buy": 8, "hold": 3, "sell": 1, "strongSell": 0}],
    )
    monkeypatch.setattr(client, "_get_rating_target_price_yfinance_sync", lambda ticker: 120.0)
    monkeypatch.setattr(client, "_get_rating_reference_price_sync", lambda ticker: 100.0)

    profile = client._get_analyst_rating_profile_sync("AAPL")

    assert profile is not None
    assert profile.consensus_signal == "bullish"
    assert profile.buy_count == 14
    assert profile.sell_count == 1
    assert profile.upside_pct == 20.0


def test_market_data_client_insider_summary_parses_sec_form4(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    filings_payload = {
        "filings": {
            "recent": {
                "form": ["4", "4/A"],
                "filingDate": ["2026-02-20", "2026-02-18"],
                "reportDate": ["2026-02-19", "2026-02-17"],
                "accessionNumber": ["0000000001-26-000004", "0000000001-26-000005"],
                "primaryDocument": ["form4.xml", "form4a.xml"],
            }
        }
    }
    xml_payloads = {
        "form4.xml": """
            <ownershipDocument>
              <periodOfReport>2026-02-19</periodOfReport>
              <nonDerivativeTable>
                <nonDerivativeTransaction>
                  <transactionDate><value>2026-02-19</value></transactionDate>
                  <transactionCoding><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionCoding>
                  <transactionAmounts>
                    <transactionShares><value>100</value></transactionShares>
                    <transactionPricePerShare><value>10.5</value></transactionPricePerShare>
                  </transactionAmounts>
                </nonDerivativeTransaction>
              </nonDerivativeTable>
            </ownershipDocument>
        """,
        "form4a.xml": """
            <ownershipDocument>
              <periodOfReport>2026-02-17</periodOfReport>
              <nonDerivativeTable>
                <nonDerivativeTransaction>
                  <transactionDate><value>2026-02-17</value></transactionDate>
                  <transactionCoding><transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionCoding>
                  <transactionAmounts>
                    <transactionShares><value>25</value></transactionShares>
                    <transactionPricePerShare><value>12</value></transactionPricePerShare>
                  </transactionAmounts>
                </nonDerivativeTransaction>
              </nonDerivativeTable>
            </ownershipDocument>
        """,
    }

    monkeypatch.setattr(client, "_load_sec_ticker_mapping", lambda: {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}})
    monkeypatch.setattr(client, "_fetch_sec_json", lambda url: filings_payload if "submissions" in url else None)
    monkeypatch.setattr(
        client,
        "_fetch_sec_raw_document_text",
        lambda url: xml_payloads["form4a.xml"] if url.endswith("form4a.xml") else xml_payloads["form4.xml"],
    )

    summary = client._get_insider_transaction_summary_sync("AAPL")

    assert summary is not None
    assert summary.signal == "accumulating"
    assert summary.net_shares == 75.0
    assert summary.net_value == 750.0
    assert summary.buy_count == 1
    assert summary.sell_count == 1
    assert summary.transaction_count == 2


def test_market_data_client_corporate_actions_reads_yfinance_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.actions = pd.DataFrame(
                {
                    "Dividends": [0.0, 0.82],
                    "Stock Splits": [4.0, 0.0],
                },
                index=pd.to_datetime(["2026-02-10", "2026-01-15"], utc=True),
            )

    monkeypatch.setattr("invest_advisor_bot.providers.market_data_client.yf.Ticker", FakeTicker)

    actions = client._get_corporate_actions_sync("AAPL", lookback_days=120)

    assert [item.action_type for item in actions] == ["split", "dividend"]
    assert client._classify_corporate_action_signal(actions) == "split"


def test_market_data_client_etf_exposure_profile_parses_holdings(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.info = {"fundFamily": "Vanguard", "category": "Large Blend", "totalAssets": 500_000_000_000}
            self.fund_top_holdings = pd.DataFrame(
                {
                    "Name": ["Microsoft", "Apple", "NVIDIA"],
                    "Holding Percent": [0.07, 0.065, 0.055],
                }
            )
            self.fund_sector_weightings = {"Technology": 0.32, "Financial Services": 0.14}
            self.fund_country_weightings = {"United States": 0.99, "Canada": 0.01}

    monkeypatch.setattr("invest_advisor_bot.providers.market_data_client.yf.Ticker", FakeTicker)

    profile = client._get_etf_exposure_profile_sync("VOO")

    assert profile is not None
    assert profile.fund_family == "Vanguard"
    assert profile.top_holdings[0] == ("Microsoft", 7.0)
    assert profile.sector_exposures[0] == ("Technology", 32.0)
    assert profile.country_exposures[0] == ("United States", 99.0)
    assert profile.exposure_signal == "country_concentrated"


def test_market_data_client_global_macro_calendar_parses_trading_economics(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        trading_economics_api_key="demo",
        global_macro_calendar_countries=("Japan", "Euro Area"),
        global_macro_calendar_importance=2,
    )
    monkeypatch.setattr(
        client,
        "_fetch_trading_economics_calendar_for_countries",
        lambda countries, *, days_back, days_ahead, importance: [
            {
                "Country": "Japan",
                "Event": "Interest Rate Decision",
                "Category": "Interest Rate",
                "Date": "2026-04-03T03:00:00Z",
                "Importance": 3,
                "Forecast": "0.50",
                "Previous": "0.25",
                "Actual": None,
                "URL": "/japan/interest-rate",
            }
        ],
    )

    events = client._fetch_global_macro_calendar_events(days_ahead=14)

    assert len(events) == 1
    assert events[0].country == "Japan"
    assert events[0].forecast_value == 0.5
    assert events[0].previous_value == 0.25
    assert events[0].source == "trading_economics_global_calendar"


def test_market_data_client_disables_trading_economics_calendar_after_free_plan_error() -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        trading_economics_api_key="demo",
        global_macro_calendar_countries=("Japan",),
    )

    request = httpx.Request("GET", "https://api.tradingeconomics.com/calendar/country/japan")
    response = httpx.Response(
        403,
        request=request,
        text="No Access to this country as free user.",
    )

    handled = client._handle_trading_economics_calendar_error(  # type: ignore[attr-defined]
        httpx.HTTPStatusError("forbidden", request=request, response=response)
    )

    status = client.status()

    assert handled is True
    assert status["trading_economics_calendar_disabled"] is True
    assert "restricted" in str(status["trading_economics_warning"]).casefold()
    assert status["configured_sources"]["trading_economics"] is False


def test_market_data_client_disables_alpha_vantage_after_free_daily_limit_note() -> None:
    client = MarketDataClient(
        cache_ttl_seconds=900,
        alpha_vantage_api_key="demo",
    )

    handled = client._handle_alpha_vantage_non_data_response(  # type: ignore[attr-defined]
        "We have detected your API key and our standard API rate limit is 25 requests per day."
    )

    status = client.status()

    assert handled is True
    assert status["alpha_vantage_disabled"] is True
    assert "daily-limit" in str(status["alpha_vantage_warning"]).casefold()
    assert status["configured_sources"]["alpha_vantage"] is False


def test_market_data_client_macro_surprise_signals_builds_cpi_and_fomc(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    event_at = datetime(2026, 4, 10, 12, 30, tzinfo=timezone.utc)
    cpi_frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=16, freq="MS", tz="UTC"),
            "value": [300.0, 300.5, 301.0, 301.6, 302.2, 302.7, 303.0, 303.4, 304.0, 304.5, 305.1, 305.8, 306.9, 307.8, 308.7, 309.9],
        }
    )
    ppi_rows = [{"date": datetime(2025, month, 1, tzinfo=timezone.utc), "value": 200.0 + month} for month in range(1, 13)]
    payroll_rows = [{"date": datetime(2025, month, 1, tzinfo=timezone.utc), "value": 150_000.0 + (month * 100.0)} for month in range(1, 13)]

    monkeypatch.setattr(
        client,
        "_get_macro_event_calendar_sync",
        lambda days_ahead: [
            MacroEvent("cpi", "CPI", "inflation", "fred_release_calendar", event_at, "high", "scheduled"),
            MacroEvent("fomc", "FOMC", "policy", "federal_reserve", event_at, "critical", "scheduled"),
        ],
    )
    monkeypatch.setattr(client, "_fetch_fred_series_frame", lambda series_id: cpi_frame if series_id == "CPIAUCSL" else pd.DataFrame())
    monkeypatch.setattr(client, "_fetch_bls_series_history", lambda series_ids, years=3: {"WPUFD4": ppi_rows, "CES0000000001": payroll_rows})
    monkeypatch.setattr(client, "_fetch_bls_macro_snapshot", lambda: {"payrolls_revision_k": -15.0})
    monkeypatch.setattr(
        client,
        "_fetch_consensus_calendar_rows",
        lambda days_back, days_ahead: {
            "cpi": {"Actual": "3.4%", "Forecast": "3.1%", "TEForecast": "3.2%", "URL": "/united-states/inflation-rate"},
            "fomc": {"Actual": "4.50%", "Forecast": "4.25%", "TEForecast": "4.25%", "URL": "/united-states/interest-rate"},
        },
    )
    monkeypatch.setattr(client, "_fetch_recent_fomc_statement_urls", lambda limit=2: ["https://www.federalreserve.gov/newsevents/pressreleases/monetary20260318a.htm", "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260128a.htm"])
    monkeypatch.setattr(
        client,
        "_fetch_fomc_statement_text",
        lambda url: "inflation remains elevated and job gains have remained solid" if "20260318" in url else "inflation has eased and labor market conditions have eased",
    )

    signals = client._get_macro_surprise_signals_sync()

    labels = {item.event_key: item.surprise_label for item in signals}
    cpi_signal = next(item for item in signals if item.event_key == "cpi")
    assert labels["cpi"] == "hotter_than_baseline"
    assert cpi_signal.consensus_expected_value == 3.1
    assert cpi_signal.consensus_surprise_value == 0.3
    assert labels["fomc"] == "hawkish_shift"


def test_market_data_client_macro_market_reactions_flags_non_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(cache_ttl_seconds=900)
    released_at = datetime.now(timezone.utc) - pd.Timedelta(days=1)
    surprise_signal = type(
        "Surprise",
        (),
        {
            "event_key": "fomc",
            "event_name": "FOMC",
            "released_at": released_at,
            "market_bias": "defensive_rates_up",
            "surprise_label": "hawkish_shift",
        },
    )()

    bars = [
        OhlcvBar(ticker="SPY", timestamp=released_at - pd.Timedelta(minutes=5), open=100.0, high=100.0, low=100.0, close=100.0, volume=1_000),
        OhlcvBar(ticker="SPY", timestamp=released_at + pd.Timedelta(minutes=5), open=100.2, high=100.2, low=100.2, close=100.2, volume=1_000),
        OhlcvBar(ticker="SPY", timestamp=released_at + pd.Timedelta(minutes=60), open=100.4, high=100.4, low=100.4, close=100.4, volume=1_000),
    ]

    monkeypatch.setattr(client, "_get_macro_surprise_signals_sync", lambda: [surprise_signal])
    monkeypatch.setattr(client, "_get_history_sync", lambda ticker, period, interval, limit: bars)

    reactions = client._get_macro_market_reactions_sync()

    assert reactions
    assert reactions[0].confirmation_label in {"not_confirmed", "mixed"}
