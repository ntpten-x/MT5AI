from __future__ import annotations

from datetime import datetime, timezone

from invest_advisor_bot.analytics_store import AnalyticsStore
from invest_advisor_bot.data_quality import ReasoningDataQualityGate
from invest_advisor_bot.evidently_observer import EvidentlyObserver
from invest_advisor_bot.providers.market_data_client import AssetQuote


def test_reasoning_data_quality_gate_flags_missing_core_inputs() -> None:
    gate = ReasoningDataQualityGate(
        enabled=True,
        gx_enabled=False,
        min_market_assets=2,
        min_macro_sources=2,
        min_news_items=1,
    )

    report = gate.evaluate(
        news=[],
        market_data={},
        macro_context={"vix": None, "tnx": None, "cpi_yoy": None},
        macro_intelligence={"headline": "", "signals": [], "sources_used": []},
        research_findings=[],
    )

    assert report.status == "fail"
    assert report.blocking is True
    assert any(issue.check == "market_assets" for issue in report.issues)
    assert any(issue.check == "macro_core" for issue in report.issues)


def test_analytics_store_writes_jsonl_events(tmp_path) -> None:
    store = AnalyticsStore(root_dir=tmp_path / "analytics", enabled=True, parquet_export_interval_seconds=3600)

    store.record_recommendation_event(
        artifact_key="artifact-1",
        conversation_key_hash="abc123",
        question="What should I buy?",
        model="gpt-test",
        fallback_used=False,
        response_text="Hold quality growth with a cash buffer.",
        payload={"source_health": {"score": 82.0}},
        source_coverage={"used_sources": ["fred", "news"]},
        data_quality={"status": "pass", "score": 92.0},
    )

    status = store.status()
    jsonl_path = tmp_path / "analytics" / "jsonl" / "recommendation_events.jsonl"

    assert jsonl_path.exists()
    assert status["last_write_at"] is not None


def test_evidently_observer_writes_local_dataset(tmp_path) -> None:
    observer = EvidentlyObserver(root_dir=tmp_path / "evidently", enabled=True, report_every_n_events=50)

    observer.log_recommendation(
        artifact_key="artifact-1",
        question="ตลาดตอนนี้เป็นอย่างไร",
        response_text="ควรถือ cash buffer และ quality growth",
        model="gpt-test",
        fallback_used=False,
        payload={"source_health": {"score": 70.0}, "no_trade_decision": {"should_abstain": False}, "macro_intelligence": {"headline": "rates market leaning dovish"}},
        data_quality={"status": "warn", "score": 75.0},
    )
    observer.log_outcome(
        artifact_key="artifact-1",
        outcome_label="win",
        return_after_cost_pct=0.04,
        detail={"alpha_after_cost_pct": 0.02, "execution_cost_bps": 8.0},
    )

    status = observer.status()

    assert (tmp_path / "evidently" / "llm_evaluations.jsonl").exists()
    assert (tmp_path / "evidently" / "llm_outcomes.jsonl").exists()
    assert status["event_count"] == 1


def test_reasoning_data_quality_gate_passes_when_core_inputs_are_present() -> None:
    gate = ReasoningDataQualityGate(enabled=True, gx_enabled=False, min_market_assets=1, min_macro_sources=1, min_news_items=1)
    quote = AssetQuote(
        ticker="SPY",
        name="SPY",
        currency="USD",
        exchange="TEST",
        price=100.0,
        previous_close=99.0,
        open_price=99.5,
        day_high=101.0,
        day_low=98.5,
        volume=1000,
        timestamp=datetime.now(timezone.utc),
    )

    report = gate.evaluate(
        news=[object()],
        market_data={"spy_etf": quote},
        macro_context={"vix": 18.0, "tnx": 4.1, "cpi_yoy": 2.9},
        macro_intelligence={"headline": "macro backdrop mixed", "signals": ["yield_curve_inverted"], "sources_used": ["fred"]},
        research_findings=[],
    )

    assert report.status == "pass"
    assert report.blocking is False
