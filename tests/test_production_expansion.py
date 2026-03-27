from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from invest_advisor_bot.analysis.stock_screener import StockCandidate
from invest_advisor_bot.analytics_warehouse import AnalyticsWarehouse
from invest_advisor_bot.backtesting import BacktestingEngine
from invest_advisor_bot.braintrust_observer import BraintrustObserver
from invest_advisor_bot.event_bus import EventBus
from invest_advisor_bot.event_bus_worker import EventBusConsumerWorker
from invest_advisor_bot.feature_store import FeatureStoreBridge
from invest_advisor_bot.hot_path_cache import HotPathCache
from invest_advisor_bot.providers.broker_client import ExecutionSandboxClient
from invest_advisor_bot.providers.live_market_stream import LiveMarketStreamClient
from invest_advisor_bot.providers.market_data_client import MarketDataClient, OhlcvBar
from invest_advisor_bot.providers.microstructure_client import MicrostructureClient
from invest_advisor_bot.providers.transcript_client import EarningsTranscriptClient
from invest_advisor_bot.semantic_analyst import SemanticAnalyst
from invest_advisor_bot.services.recommendation_service import RecommendationService
from invest_advisor_bot.thesis_vector_store import ThesisVectorStore
from invest_advisor_bot.orchestration.prefect_flows import WorkflowOrchestrator


class DummyLLM:
    async def generate_text(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


@pytest.mark.asyncio
async def test_execution_sandbox_client_fetches_account_positions_and_order(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ExecutionSandboxClient(
        enabled=True,
        provider="alpaca",
        api_key="key",
        api_secret="secret",
    )

    def fake_request(method: str, path: str, json_payload=None):  # noqa: ANN001
        if method == "GET" and path == "account":
            return {
                "id": "acct-1",
                "status": "ACTIVE",
                "currency": "USD",
                "equity": "105000",
                "buying_power": "210000",
                "cash": "15000",
                "pattern_day_trader": False,
            }
        if method == "GET" and path == "positions":
            return [
                {
                    "symbol": "AAPL",
                    "qty": "2",
                    "side": "long",
                    "market_value": "380.5",
                    "cost_basis": "350.0",
                    "unrealized_pl": "30.5",
                    "unrealized_plpc": "0.087",
                }
            ]
        if method == "POST" and path == "orders":
            return {
                "id": "ord-1",
                "client_order_id": "client-1",
                "status": "accepted",
            }
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(client, "_request", fake_request)

    account = await client.get_account()
    positions = await client.list_positions()
    order = await client.submit_order(symbol="AAPL", qty=1, side="buy")

    assert account is not None
    assert account.account_id == "acct-1"
    assert positions and positions[0].symbol == "AAPL"
    assert order is not None
    assert order.order_id == "ord-1"
    assert client.status()["last_order_symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_execution_sandbox_client_supports_tradier_equity_and_option_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ExecutionSandboxClient(
        enabled=True,
        provider="tradier",
        tradier_access_token="token",
        tradier_account_id="acct-9",
    )

    def fake_request(method: str, path: str, json_payload=None):  # noqa: ANN001
        if method == "GET" and path == "account":
            return {
                "balances": {
                    "total_equity": "50000",
                    "stock_buying_power": "120000",
                    "cash": "9000",
                }
            }
        if method == "GET" and path == "positions":
            return {"positions": {"position": {"symbol": "SPY", "qty": "3", "cost_basis": "1500"}}}
        if method == "POST" and path == "orders":
            if json_payload and json_payload.get("class") == "option":
                return {"order": {"id": "opt-1", "status": "ok", "create_date": "2026-03-24T10:00:00Z"}}
            return {"order": {"id": "eq-1", "status": "ok", "create_date": "2026-03-24T10:00:00Z"}}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(client, "_request", fake_request)

    account = await client.get_account()
    positions = await client.list_positions()
    equity_order = await client.submit_order(symbol="SPY", qty=1, side="buy")
    option_order = await client.submit_option_order(
        contract_symbol="SPY250117C00500000",
        qty=1,
        side="buy_to_open",
        order_type="limit",
        limit_price=4.5,
    )

    assert account is not None
    assert account.account_id == "acct-9"
    assert positions and positions[0].symbol == "SPY"
    assert equity_order is not None and equity_order.order_id == "eq-1"
    assert option_order is not None and option_order.order_id == "opt-1"
    assert client.status()["last_order_kind"] == "option"


def test_earnings_transcript_client_builds_constructive_management_commentary() -> None:
    client = EarningsTranscriptClient(api_key="fmp-key")

    insight = client._build_insight(  # type: ignore[attr-defined]
        ticker="AAPL",
        payload={
            "symbol": "AAPL",
            "quarter": 1,
            "year": 2026,
            "date": "2026-01-31T21:00:00Z",
            "content": (
                "Demand remains strong and backlog improved. "
                "We saw margin expansion with a healthy pipeline. "
                "Management will raise guidance and remains confident."
            ),
        },
    )

    assert insight is not None
    assert insight.tone == "constructive"
    assert insight.guidance_signal == "supportive"
    assert insight.confidence >= 0.5
    assert insight.highlights


def test_earnings_transcript_client_available_with_alpha_vantage_only() -> None:
    client = EarningsTranscriptClient(api_key="", alpha_vantage_api_key="alpha-key")

    status = client.status()

    assert client.available() is True
    assert status["backend"] == "alpha_vantage"
    assert status["provider_order"] == ["alpha_vantage"]


@pytest.mark.asyncio
async def test_earnings_transcript_client_uses_alpha_vantage_when_fmp_returns_no_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EarningsTranscriptClient(api_key="fmp-key", alpha_vantage_api_key="alpha-key")

    monkeypatch.setattr(client, "_fetch_fmp_transcript_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        client,
        "_fetch_alpha_vantage_transcript_payload",
        lambda ticker: {
            "_provider": "alpha_vantage",
            "symbol": ticker,
            "date": "2026-01-31T21:00:00Z",
            "quarter": 1,
            "year": 2026,
            "transcript": (
                "Demand remains strong and backlog improved. "
                "Management remains confident and will raise guidance."
            ),
        },
    )

    insight = await client.get_latest_management_commentary("AAPL", max_quarters_back=1)

    assert insight is not None
    assert insight.source == "alpha_vantage"
    assert insight.guidance_signal == "supportive"
    assert client.status()["backend"] == "alpha_vantage"


@pytest.mark.asyncio
async def test_microstructure_client_builds_snapshot_from_databento_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MicrostructureClient(
        api_key="db-key",
        equities_dataset="XNAS.ITCH",
        options_dataset="OPRA.PILLAR",
    )

    frame = pd.DataFrame(
        [
            {
                "bid_px_00": 199.9,
                "ask_px_00": 200.1,
                "bid_sz_00": 1200,
                "ask_sz_00": 900,
                "price": 200.0,
                "size": 250,
            }
        ],
        index=[datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc)],
    )

    class _FakeRange:
        def to_df(self) -> pd.DataFrame:
            return frame

    class _FakeHistorical:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.timeseries = self

        def get_range(self, **kwargs):  # noqa: ANN003
            return _FakeRange()

    fake_db = SimpleNamespace(Historical=_FakeHistorical)
    monkeypatch.setattr(client, "_load_databento", lambda: fake_db)

    snapshot = await client.get_equity_snapshot("MSFT")

    assert snapshot is not None
    assert snapshot.symbol == "MSFT"
    assert snapshot.spread_bps == 10.0
    assert snapshot.imbalance == 0.571
    assert snapshot.sample_count == 1


def test_workflow_orchestrator_builds_prefect_wrapped_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePrefect:
        @staticmethod
        def flow(*, name: str):
            def decorator(fn):
                def wrapped():
                    return {"flow_name": name, "payload": fn()}

                return wrapped

            return decorator

    monkeypatch.setattr(WorkflowOrchestrator, "_load_prefect", lambda self: _FakePrefect())
    orchestrator = WorkflowOrchestrator(enabled=True)

    runtime_flow = orchestrator.build_runtime_snapshot_flow(snapshot_factory=lambda: {"ok": True})
    status = orchestrator.status()

    assert status["available"] is True
    assert len(status["flows"]) == 3
    assert runtime_flow() == {
        "flow_name": "invest-advisor-runtime-snapshot",
        "payload": {"ok": True},
    }


@pytest.mark.asyncio
async def test_live_market_stream_client_samples_iterable_records(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiveMarketStreamClient(
        enabled=True,
        api_key="db-key",
        dataset="XNAS.ITCH",
        schema="mbp-1",
        max_events_per_poll=2,
        sample_timeout_seconds=0.5,
    )

    class _Record:
        def __init__(self, symbol: str, bid: float, ask: float, price: float) -> None:
            self.symbol = symbol
            self.bid_px_00 = bid
            self.ask_px_00 = ask
            self.price = price
            self.size = 100
            self.ts_event = datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc)

    class _FakeLive:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self._records = iter([_Record("AAPL", 199.9, 200.1, 200.0), _Record("MSFT", 399.8, 400.2, 400.0)])

        def subscribe(self, **kwargs) -> None:  # noqa: ANN003
            return None

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._records)

        def stop(self) -> None:
            return None

    monkeypatch.setattr(client, "_load_databento", lambda: SimpleNamespace(Live=_FakeLive))

    events = await client.sample_events(("AAPL", "MSFT"))

    assert len(events) == 2
    assert events[0].symbol == "AAPL"
    assert events[0].spread_bps == 10.0
    assert client.status()["last_event_count"] == 2


def test_market_data_client_includes_gdelt_global_event_intelligence(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MarketDataClient(gdelt_query="(geopolitical OR sanctions) sourcecountry:US")

    context_payload = {
        "articles": [
            {"title": "Oil shipping disrupted after sanctions warning", "url": "https://example.com/1", "domain": "example.com"},
            {"title": "Election conflict risk rises in key region", "url": "https://example.com/2", "domain": "example.com"},
        ]
    }
    geo_payload = {
        "features": [
            {"properties": {"name": "Black Sea", "count": 8}, "geometry": {"coordinates": [31.0, 44.0]}},
        ]
    }

    def fake_get(url: str, params=None, headers=None, follow_redirects=True):  # noqa: ANN001
        payload = context_payload if "doc/doc" in url else geo_payload
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        )

    monkeypatch.setattr(client, "_get_http_client", lambda: SimpleNamespace(get=fake_get))

    snapshot = client._fetch_gdelt_global_event_snapshot()  # type: ignore[attr-defined]

    assert snapshot["article_count"] == 2
    assert snapshot["top_location"] == "Black Sea"
    assert snapshot["risk_score"] > 0
    assert snapshot["headlines"]


def test_braintrust_observer_writes_local_datasets(tmp_path) -> None:
    observer = BraintrustObserver(
        root_dir=tmp_path / "braintrust",
        enabled=True,
        api_key="",
        project_name="invest-advisor-bot",
        experiment_name="production-evals",
    )

    observer.log_recommendation(
        artifact_key="artifact-1",
        question="market outlook",
        response_text="response",
        model="gpt-test",
        fallback_used=False,
        payload={"source_health": {"score": 82.0}},
        data_quality={"status": "ok"},
        source_coverage={"used_sources": ["fred", "gdelt_context"]},
    )
    observer.log_outcome(
        artifact_key="artifact-1",
        outcome_label="win",
        return_after_cost_pct=0.04,
        detail={"alpha_after_cost_pct": 0.01, "execution_cost_bps": 12.0},
    )

    recommendation_rows = [json.loads(line) for line in (tmp_path / "braintrust" / "braintrust_recommendations.jsonl").read_text(encoding="utf-8").splitlines()]
    outcome_rows = [json.loads(line) for line in (tmp_path / "braintrust" / "braintrust_outcomes.jsonl").read_text(encoding="utf-8").splitlines()]

    assert recommendation_rows[0]["artifact_key"] == "artifact-1"
    assert outcome_rows[0]["outcome_label"] == "win"
    assert observer.status()["event_count"] == 2


def test_thesis_vector_store_and_feature_store_write_local_artifacts(tmp_path) -> None:
    thesis_store = ThesisVectorStore(root_dir=tmp_path / "qdrant", enabled=True)
    thesis_store.record_thesis(
        thesis_key="thesis-1",
        thesis_text="Rates cooling while quality growth leadership stays intact",
        source_kind="recommendation",
        conversation_key="chat-1",
        query_text="What should I buy?",
        tags=["fred", "quality_growth"],
        confidence_score=0.72,
        detail={"scope": "us-stocks"},
    )

    rows = thesis_store.search(query_text="quality growth", conversation_key="chat-1", limit=2)

    assert rows
    assert rows[0]["thesis_key"] == "thesis-1"
    assert thesis_store.status()["backend"] == "jsonl"

    feature_store = FeatureStoreBridge(
        root_dir=tmp_path / "feature_store",
        enabled=True,
        feast_enabled=True,
        project_name="invest_advisor_bot",
    )
    feature_store.record_recommendation_features(
        artifact_key="artifact-1",
        question="market outlook",
        model="gpt-test",
        payload={
            "source_health": {"score": 82.0, "freshness_pct": 91.0},
            "market_confidence": {"score": 0.74},
            "macro_intelligence": {"signals": ["yield_curve_inverted"]},
            "macro_event_calendar": [{"event_key": "cpi"}],
            "news_headlines": [{"title": "Fed steady"}],
            "research_highlights": [{"title": "macro note"}],
            "company_intelligence": [{"ticker": "AAPL"}],
            "thesis_memory": [{"thesis_text": "quality growth"}],
            "no_trade_decision": {"should_abstain": False},
        },
        source_coverage={"used_sources": ["fred", "sec"]},
        data_quality={"status": "pass", "score": 93.0},
        fallback_used=False,
        service_name="recommendation_service",
    )
    feature_store.record_outcome_features(
        artifact_key="artifact-1",
        outcome_label="win",
        adjusted_return_pct=0.04,
        detail={"alpha_after_cost_pct": 0.01, "execution_cost_bps": 12.0},
    )

    status = feature_store.status()

    assert (tmp_path / "feature_store" / "offline" / "recommendation_features.jsonl").exists()
    assert (tmp_path / "feature_store" / "feast_repo" / "feature_store.yaml").exists()
    assert status["feature_counts"]["recommendation"] == 1
    assert status["feature_counts"]["outcome"] == 1


def test_thesis_vector_store_uses_remote_embeddings_and_rerank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    thesis_store = ThesisVectorStore(
        root_dir=tmp_path / "qdrant",
        enabled=True,
        vector_size=8,
        embedding_api_key="embed-key",
        embedding_model="text-embedding-3-small",
        rerank_enabled=True,
    )

    def fake_post(url: str, headers=None, json=None):  # noqa: ANN001
        text = str((json or {}).get("input") or "").casefold()
        if "inflation" in text:
            embedding = [0.92, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif "quality growth" in text:
            embedding = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            embedding = [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"embedding": embedding}]},
        )

    monkeypatch.setattr(thesis_store, "_get_http_client", lambda: SimpleNamespace(post=fake_post))

    thesis_store.record_thesis(
        thesis_key="thesis-inflation",
        thesis_text="Inflation cooling supports duration-sensitive setups",
        source_kind="recommendation",
        conversation_key="chat-1",
        query_text="inflation cooling trade",
        tags=["inflation", "duration"],
        confidence_score=0.78,
    )
    thesis_store.record_thesis(
        thesis_key="thesis-growth",
        thesis_text="Quality growth leadership stays intact while margins expand",
        source_kind="recommendation",
        conversation_key="chat-1",
        query_text="quality growth trade",
        tags=["quality_growth"],
        confidence_score=0.74,
    )

    rows = thesis_store.search(query_text="inflation cooling setup", conversation_key="chat-1", limit=2)
    status = thesis_store.status()

    assert rows
    assert rows[0]["thesis_key"] == "thesis-inflation"
    assert rows[0]["rerank_score"] >= rows[1]["rerank_score"]
    assert status["embedding_configured"] is True
    assert status["last_embedding_backend"] == "remote"
    assert status["rerank_enabled"] is True


def test_backtesting_engine_builds_candidate_summary(tmp_path) -> None:
    engine = BacktestingEngine(root_dir=tmp_path / "backtesting", enabled=True)
    index = pd.date_range("2026-01-01", periods=40, freq="B", tz="UTC")

    def _bars(ticker: str, start: float, step: float) -> list:
        rows = []
        for offset, timestamp in enumerate(index):
            close = start + (offset * step)
            rows.append(
                OhlcvBar(
                    ticker=ticker,
                    timestamp=timestamp.to_pydatetime(),
                    open=close - 0.5,
                    high=close + 0.5,
                    low=close - 1.0,
                    close=close,
                    volume=1000 + offset,
                )
            )
        return rows

    summary = engine.evaluate_candidate_histories(
        candidate_histories={
            "AAPL": _bars("AAPL", 100.0, 1.2),
            "MSFT": _bars("MSFT", 100.0, 0.8),
        },
        benchmark_history=_bars("SPY", 100.0, 0.4),
    )

    assert summary["candidate_count"] == 2
    assert summary["best_candidate"]["ticker"] == "AAPL"
    assert summary["avg_candidate_alpha_pct"] is not None
    assert engine.status()["backend"] in {"pandas", "vectorbt"}


def test_recommendation_service_stock_screener_fallback_includes_management_commentary_and_microstructure() -> None:
    service = RecommendationService(
        DummyLLM(),
        default_investor_profile="balanced",
        backtesting_engine=BacktestingEngine(root_dir=Path("data") / "test_backtesting", enabled=False),
    )
    candidate = StockCandidate(
        asset="aapl",
        ticker="AAPL",
        company_name="Apple",
        sector="Technology",
        benchmark="nasdaq100",
        market_cap_bucket="mega",
        liquidity_tier="high",
        price=200.0,
        trend_direction="uptrend",
        trend_score=3.4,
        rsi=61.0,
        day_change_pct=1.2,
        valuation_score=0.8,
        quality_score=0.9,
        growth_score=0.9,
        technical_score=1.0,
        macro_overlay_score=0.3,
        universe_quality_score=0.9,
        composite_score=3.9,
        stance="buy",
        rationale=("earnings momentum", "quality balance sheet", "trend confirmation"),
        macro_drivers=("rates stable",),
        universe_flags=(),
        trailing_pe=30.0,
        forward_pe=28.0,
        revenue_growth=0.12,
        earnings_growth=0.14,
        return_on_equity=0.3,
        debt_to_equity=90.0,
        profit_margin=0.25,
    )

    rendered = service._build_stock_screener_fallback(  # type: ignore[attr-defined]
        question="ตอนนี้ควรซื้อหุ้นอะไร",
        picks=[candidate],
        profile=service.get_investor_profile("chat-1"),
        stock_news={"aapl": []},
        transcript_insights={
            "AAPL": {
                "ticker": "AAPL",
                "tone": "constructive",
                "guidance_signal": "supportive",
            }
        },
        microstructure_snapshots={
            "AAPL": {
                "symbol": "AAPL",
                "spread_bps": 4.2,
            }
        },
        backtest_summary={
            "benchmark_ticker": "SPY",
            "benchmark_return_pct": 3.2,
            "candidates": [{"ticker": "AAPL", "total_return_pct": 9.5, "alpha_pct": 6.3, "max_drawdown_pct": -4.1}],
        },
    )

    assert "tone constructive" in rendered
    assert "guidance supportive" in rendered
    assert "spread 4.2 bps" in rendered
    assert "Backtest snapshot" in rendered
    assert "alpha +6.3%" in rendered


def test_recommendation_service_status_exposes_new_optional_clients() -> None:
    service = RecommendationService(
        DummyLLM(),
        thesis_vector_store=SimpleNamespace(status=lambda: {"available": True, "backend": "jsonl"}),  # type: ignore[arg-type]
        feature_store=SimpleNamespace(status=lambda: {"available": True, "backend": "local+feast"}),  # type: ignore[arg-type]
        backtesting_engine=SimpleNamespace(status=lambda: {"available": True, "backend": "pandas", "benchmark_ticker": "SPY"}),  # type: ignore[arg-type]
        analytics_warehouse=SimpleNamespace(status=lambda: {"available": True, "backend": "jsonl"}),  # type: ignore[arg-type]
        event_bus=SimpleNamespace(status=lambda: {"available": True, "backend": "jsonl", "published_count": 2}),  # type: ignore[arg-type]
        event_bus_consumer=SimpleNamespace(status=lambda: {"available": True, "backend": "jsonl", "processed_count": 3}),  # type: ignore[arg-type]
        hot_path_cache=SimpleNamespace(status=lambda: {"available": True, "backend": "memory", "cache_keys": 1}),  # type: ignore[arg-type]
        semantic_analyst=SimpleNamespace(status=lambda: {"available": True, "backend": "local", "model_name": "local-heuristic"}),  # type: ignore[arg-type]
        braintrust_observer=SimpleNamespace(status=lambda: {"enabled": True, "configured": False}),  # type: ignore[arg-type]
        broker_client=SimpleNamespace(status=lambda: {"available": True, "provider": "alpaca"}),  # type: ignore[arg-type]
        transcript_client=SimpleNamespace(status=lambda: {"available": True, "cache_entries": 2}),  # type: ignore[arg-type]
        microstructure_client=SimpleNamespace(status=lambda: {"available": True, "equities_dataset": "XNAS.ITCH"}),  # type: ignore[arg-type]
    )

    status = service.status()

    assert status["broker"]["provider"] == "alpaca"
    assert status["transcripts"]["cache_entries"] == 2
    assert status["microstructure"]["equities_dataset"] == "XNAS.ITCH"
    assert status["braintrust"]["enabled"] is True
    assert status["thesis_vector_store"]["backend"] == "jsonl"
    assert status["feature_store"]["backend"] == "local+feast"
    assert status["backtesting"]["benchmark_ticker"] == "SPY"
    assert status["analytics_warehouse"]["backend"] == "jsonl"
    assert status["event_bus"]["published_count"] == 2
    assert status["event_bus_consumer"]["processed_count"] == 3
    assert status["hot_path_cache"]["backend"] == "memory"
    assert status["semantic_analyst"]["model_name"] == "local-heuristic"


def test_event_bus_writes_local_jsonl(tmp_path: Path) -> None:
    bus = EventBus(root_dir=tmp_path / "event_bus", enabled=True)

    bus.publish(topic="recommendation_event", key="artifact-1", payload={"ok": True})

    status = bus.status()
    event_log = tmp_path / "event_bus" / "event_bus.jsonl"
    assert status["backend"] == "jsonl"
    assert status["published_count"] == 1
    assert event_log.exists()
    assert '"topic": "invest_advisor.recommendation_event"' in event_log.read_text(encoding="utf-8")


def test_hot_path_cache_stores_local_values_and_stream(tmp_path: Path) -> None:
    cache = HotPathCache(root_dir=tmp_path / "hot_path_cache", enabled=True)

    cache.set_json(namespace="recommendation", key="artifact-1", payload={"ticker": "AAPL"}, ttl_seconds=120)
    cache.append_stream(stream="recommendations", payload={"artifact_key": "artifact-1"})

    assert cache.get_json(namespace="recommendation", key="artifact-1") == {"ticker": "AAPL"}
    assert cache.recent_stream(stream="recommendations", limit=1)[0]["payload"]["artifact_key"] == "artifact-1"
    assert cache.status()["backend"] in {"memory", "redis", "jsonl"}


def test_analytics_warehouse_writes_local_tables(tmp_path: Path) -> None:
    warehouse = AnalyticsWarehouse(root_dir=tmp_path / "warehouse", enabled=True)

    warehouse.record_recommendation_event(artifact_key="rec-1", question="q", detail={"ticker": "AAPL"})
    warehouse.record_evaluation_event(recommendation_key="rec-1", outcome_label="win", detail={"alpha_pct": 1.2})
    warehouse.record_market_event(symbol="AAPL", event_type="trade", detail={"price": 200.0})
    warehouse.record_runtime_snapshot({"runtime": {"ok": True}})

    status = warehouse.status()
    assert status["backend"] == "jsonl"
    assert (tmp_path / "warehouse" / "jsonl" / "recommendation_events.jsonl").exists()
    assert (tmp_path / "warehouse" / "jsonl" / "evaluation_events.jsonl").exists()
    assert (tmp_path / "warehouse" / "jsonl" / "market_events.jsonl").exists()
    assert (tmp_path / "warehouse" / "jsonl" / "runtime_snapshots.jsonl").exists()


@pytest.mark.asyncio
async def test_semantic_analyst_answers_local_questions(tmp_path: Path) -> None:
    warehouse = AnalyticsWarehouse(root_dir=tmp_path / "warehouse", enabled=True)
    warehouse.record_recommendation_event(
        artifact_key="rec-1",
        question="fallback question",
        fallback_used=True,
        detail={"source": "test"},
    )
    warehouse.record_evaluation_event(
        recommendation_key="rec-1",
        outcome_label="win",
        return_pct=2.5,
        alpha_pct=1.1,
        detail={"thesis": "trend follow"},
    )
    warehouse.record_runtime_snapshot({"runtime": {"db_state": {"healthy": True}}})
    analyst = SemanticAnalyst(root_dir=tmp_path / "semantic_analyst", warehouse=warehouse, enabled=True)

    fallback_answer = await analyst.analyze("show fallback trend")
    outcome_answer = await analyst.analyze("show outcome trend")

    assert "fallback" in fallback_answer.casefold()
    assert "evaluation" in outcome_answer.casefold() or "outcome" in outcome_answer.casefold()


def test_event_bus_consumer_worker_replays_jsonl_into_targets(tmp_path: Path) -> None:
    warehouse = AnalyticsWarehouse(root_dir=tmp_path / "warehouse", enabled=True)
    cache = HotPathCache(root_dir=tmp_path / "cache", enabled=True)
    bus = EventBus(root_dir=tmp_path / "event_bus", enabled=True)
    bus.publish(
        topic="recommendation_event",
        key="artifact-7",
        payload={"artifact_key": "artifact-7", "model": "test-model", "fallback_used": False},
    )
    worker = EventBusConsumerWorker(
        root_dir=tmp_path / "event_bus",
        enabled=True,
        analytics_warehouse=warehouse,
        hot_path_cache=cache,
        batch_size=10,
    )

    result = worker.process_pending()

    assert result["processed"] == 1
    assert worker.status()["processed_count"] == 1
    replay_rows = warehouse.recent_events(table="recommendation_events", limit=1)
    assert replay_rows
    assert replay_rows[-1]["payload"]["artifact_key"] == "artifact-7"
    replay_stream = cache.recent_stream(stream="recommendation_replay", limit=1)
    assert replay_stream[-1]["payload"]["artifact_key"] == "artifact-7"


def test_analytics_warehouse_creates_clickhouse_rollup_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[str] = []

    class _FakeClient:
        def command(self, sql: str) -> None:
            commands.append(sql)

        def insert(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    monkeypatch.setattr(AnalyticsWarehouse, "_load_clickhouse", lambda self: _FakeClient())

    warehouse = AnalyticsWarehouse(
        root_dir=tmp_path / "warehouse",
        enabled=True,
        clickhouse_url="http://localhost:8123",
    )

    status = warehouse.status()
    assert status["materialized_views_enabled"] is True
    assert "recommendation_topic_rollups" in status["rollup_tables"]
    assert any("CREATE MATERIALIZED VIEW IF NOT EXISTS default.recommendation_topic_rollups_mv" in command for command in commands)
    assert any("CREATE TABLE IF NOT EXISTS default.market_topic_rollups" in command for command in commands)
