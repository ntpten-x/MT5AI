from __future__ import annotations

from invest_advisor_bot.runtime_diagnostics import RuntimeDiagnostics, render_prometheus_metrics


def test_runtime_diagnostics_snapshot_includes_mlflow_block() -> None:
    diagnostics = RuntimeDiagnostics()

    diagnostics.record_mlflow_state(
        {
            "enabled": False,
            "tracking_configured": True,
            "experiment_name": "invest-advisor-bot",
            "warning": "mlflow_dependency_missing: install with pip install -e .[observability]",
            "last_run_id": "run-123",
            "last_run_kind": "evaluation",
            "last_run_name": "stock_pick_scorecard_evaluation",
        }
    )

    snapshot = diagnostics.snapshot()

    assert "mlflow" in snapshot
    assert snapshot["mlflow"]["enabled"] is False
    assert snapshot["mlflow"]["tracking_configured"] is True
    assert snapshot["mlflow"]["experiment_name"] == "invest-advisor-bot"
    assert snapshot["mlflow"]["last_run_id"] == "run-123"
    assert snapshot["mlflow"]["last_run_kind"] == "evaluation"
    assert snapshot["mlflow"]["last_run_name"] == "stock_pick_scorecard_evaluation"


def test_runtime_diagnostics_snapshot_includes_uptime_and_service_state() -> None:
    diagnostics = RuntimeDiagnostics()

    diagnostics.record_service_state(
        service="llm_client",
        state={
            "available": True,
            "provider_order": ["gemini", "groq"],
        },
    )

    snapshot = diagnostics.snapshot()

    assert snapshot["started_at"]
    assert snapshot["uptime_seconds"] >= 0
    assert snapshot["services"]["llm_client"]["available"] is True
    assert snapshot["services"]["llm_client"]["provider_order"] == ["gemini", "groq"]


def test_runtime_diagnostics_tracks_provider_latency_and_renders_prometheus_labels() -> None:
    diagnostics = RuntimeDiagnostics()

    for latency_ms, success in (
        (100.0, True),
        (110.0, True),
        (120.0, True),
        (130.0, True),
        (140.0, True),
        (150.0, True),
        (160.0, True),
        (170.0, True),
        (180.0, True),
        (190.0, False),
    ):
        diagnostics.record_provider_latency(
            service="llm_client",
            provider="groq",
            operation="generate_text",
            latency_ms=latency_ms,
            success=success,
        )

    snapshot = diagnostics.snapshot()
    assert snapshot["provider_latency"][0]["avg_latency_ms"] == 145.0
    assert snapshot["provider_latency"][0]["success_count"] == 9
    assert snapshot["provider_latency"][0]["failure_count"] == 1
    assert snapshot["provider_latency"][0]["p95_latency_ms"] == 185.5
    assert snapshot["provider_latency"][0]["p99_latency_ms"] == 189.1
    assert snapshot["provider_latency"][0]["window_size"] == 10

    rendered = render_prometheus_metrics(snapshot)
    assert 'invest_advisor_bot_provider_latency_last_ms{operation="generate_text",provider="groq",service="llm_client"} 190.0' in rendered
    assert 'invest_advisor_bot_provider_latency_p95_ms{operation="generate_text",provider="groq",service="llm_client"} 185.5' in rendered
    assert 'invest_advisor_bot_provider_latency_p99_ms{operation="generate_text",provider="groq",service="llm_client"} 189.1' in rendered
    assert 'invest_advisor_bot_provider_latency_failure_total{operation="generate_text",provider="groq",service="llm_client"} 1.0' in rendered
