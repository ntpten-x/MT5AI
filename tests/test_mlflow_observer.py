from __future__ import annotations

from types import SimpleNamespace

import pytest

from invest_advisor_bot.mlflow_observer import MLflowObserver


class _FakeRun:
    def __init__(self, run_id: str = "run-123") -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakeMLflow:
    def __init__(self, *, run_id: str = "run-123", fail_on_start: bool = False, fail_on_text: bool = False) -> None:
        self.run_id = run_id
        self.fail_on_start = fail_on_start
        self.fail_on_text = fail_on_text
        self.metrics: list[tuple[str, float]] = []
        self.tags: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, **kwargs):  # noqa: ANN003
        if self.fail_on_start:
            raise RuntimeError("start failed")
        return _FakeRun(run_id=self.run_id)

    def set_tags(self, tags):  # noqa: ANN001
        for key, value in tags.items():
            self.tags[str(key)] = str(value)

    def set_tag(self, key, value):  # noqa: ANN001
        self.tags[str(key)] = str(value)

    def log_params(self, params):  # noqa: ANN001
        for key, value in params.items():
            self.tags[f"param:{key}"] = str(value)

    def log_metric(self, key, value):  # noqa: ANN001
        self.metrics.append((str(key), float(value)))

    def log_text(self, text, path):  # noqa: ANN001
        if self.fail_on_text:
            raise RuntimeError("text failed")
        self.texts[str(path)] = str(text)


def test_mlflow_observer_disables_itself_when_mlflow_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):  # noqa: ANN001
        assert name == "mlflow"
        raise ImportError("missing mlflow")

    monkeypatch.setattr("invest_advisor_bot.mlflow_observer.importlib.import_module", fake_import_module)

    observer = MLflowObserver(tracking_uri="http://mlflow.local")

    assert observer.enabled() is False
    assert observer.status()["tracking_configured"] is True
    assert "observability" in str(observer.status()["warning"])


def test_mlflow_observer_logs_execution_comparison_views_with_correlation_and_redaction() -> None:
    observer = MLflowObserver()
    fake_mlflow = _FakeMLflow(run_id="run-correlation")
    observer._mlflow = fake_mlflow  # type: ignore[attr-defined]
    observer._enabled = True  # type: ignore[attr-defined]

    run_id = observer.log_recommendation(
        service_name="recommendation_service",
        question="token=sk-live-very-secret-value market outlook?",
        conversation_key="chat-1",
        model="fake-model",
        fallback_used=False,
        payload={
            "kind": "market_update",
            "llm_api_key": "super-secret",
            "conversation_key": "chat-1",
            "notes": "x" * 2000,
            "rows": list(range(30)),
        },
        response_text="response " * 700,
        source_coverage={"used_sources": ["fred", "macro_surprise_engine"]},
        artifact_key="artifact-123",
        response_id="resp-456",
        source_ranking=[{"source": "fred", "weighted_score": 72.0}],
        source_health={"score": 78.0, "freshness_pct": 84.0, "label": "strong", "total_penalty": 6.0},
        champion_challenger={
            "recommended_policy": "adaptive_champion",
            "delta_vs_baseline": 0.07,
            "champion": {"name": "adaptive", "score": 0.74},
            "challenger": {"name": "baseline", "score": 0.67},
            "runner": {
                "winner": "adaptive_champion",
                "policies": [
                    {"name": "adaptive_champion", "score": 0.74},
                    {"name": "baseline_confidence_only", "score": 0.67},
                ],
            },
        },
        factor_exposures={"top_exposure_factor": "equity_beta", "top_exposure_weight_pct": 42.0},
        thesis_invalidation={"has_active_invalidation": True, "score": 28.0},
        walk_forward_eval={"window_count": 4, "avg_hit_rate_pct": 62.5, "avg_return_after_cost_pct": 1.8},
        execution_panel={
            "closed_postmortems": 5,
            "ttl_hit_rate_pct": 60.0,
            "fast_decay_rate_pct": 20.0,
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
            "source_ttl_heatmap": [{"source": "fred", "alert_kind": "stock_pick", "ttl_bucket": "short"}],
        },
    )

    assert run_id == "run-correlation"
    assert fake_mlflow.tags["artifact_key"] == "artifact-123"
    assert fake_mlflow.tags["response_id"] == "resp-456"
    assert fake_mlflow.tags["conversation_key_hash"]
    assert ("execution_ttl_hit_rate_pct__stock_pick", 66.7) in fake_mlflow.metrics
    assert ("source_health_score", 78.0) in fake_mlflow.metrics
    assert ("source_total_penalty", 6.0) in fake_mlflow.metrics
    assert ("champion_delta_vs_baseline", 0.07) in fake_mlflow.metrics
    assert ("factor_top_exposure_weight_pct", 42.0) in fake_mlflow.metrics
    assert ("thesis_invalidation_score", 28.0) in fake_mlflow.metrics
    assert ("walk_forward_window_count", 4.0) in fake_mlflow.metrics
    assert ("runner_policy_score__adaptive_champion", 0.74) in fake_mlflow.metrics
    assert fake_mlflow.tags["execution_best_ttl_bucket__stock_pick"] == "short"
    assert fake_mlflow.tags["factor_top_exposure_factor"] == "equity_beta"
    assert fake_mlflow.tags["thesis_invalidation_active"] == "true"
    assert fake_mlflow.tags["runner_winner"] == "adaptive_champion"
    assert "[redacted]" in fake_mlflow.texts["question.txt"]
    assert "sk-live-very-secret-value" not in fake_mlflow.texts["question.txt"]
    assert "[truncated]" in fake_mlflow.texts["response.txt"]
    assert "[redacted]" in fake_mlflow.texts["payload.json"]
    assert "super-secret" not in fake_mlflow.texts["payload.json"]
    assert "chat-1" not in fake_mlflow.texts["payload.json"]
    assert "... (+10 more items)" in fake_mlflow.texts["payload.json"]
    assert "execution_by_alert_kind.json" in fake_mlflow.texts
    assert "source_ttl_heatmap.json" in fake_mlflow.texts
    assert "source_health.json" in fake_mlflow.texts
    assert "champion_challenger.json" in fake_mlflow.texts
    assert "factor_exposures.json" in fake_mlflow.texts
    assert "thesis_invalidation.json" in fake_mlflow.texts
    assert "walk_forward_eval.json" in fake_mlflow.texts
    status = observer.status()
    assert status["last_run_id"] == "run-correlation"
    assert status["last_run_kind"] == "recommendation"


def test_mlflow_observer_logs_evaluation_with_tags_and_returns_run_id() -> None:
    observer = MLflowObserver()
    fake_mlflow = _FakeMLflow(run_id="run-eval")
    observer._mlflow = fake_mlflow  # type: ignore[attr-defined]
    observer._enabled = True  # type: ignore[attr-defined]

    run_id = observer.log_evaluation(
        name="stock_pick_scorecard_evaluation",
        metrics={"ttl_hit": True, "return_after_cost_pct": 0.12, "ignored": "text"},
        artifacts={"scorecard_detail": {"chat_id": "123456", "token": "abc", "rows": list(range(25))}},
        tags={"artifact_key": "alert-1", "ticker": "AAPL", "chat_id_hash": "hashed-chat"},
    )

    assert run_id == "run-eval"
    assert fake_mlflow.tags["artifact_key"] == "alert-1"
    assert fake_mlflow.tags["ticker"] == "AAPL"
    assert fake_mlflow.tags["chat_id_hash"] == "hashed-chat"
    assert ("ttl_hit", 1.0) in fake_mlflow.metrics
    assert ("return_after_cost_pct", 0.12) in fake_mlflow.metrics
    assert "[redacted]" in fake_mlflow.texts["scorecard_detail.json"]
    assert "... (+5 more items)" in fake_mlflow.texts["scorecard_detail.json"]


def test_mlflow_observer_swallows_logging_failures() -> None:
    observer = MLflowObserver()
    fake_mlflow = _FakeMLflow(fail_on_text=True)
    observer._mlflow = fake_mlflow  # type: ignore[attr-defined]
    observer._enabled = True  # type: ignore[attr-defined]

    result = observer.log_recommendation(
        service_name="recommendation_service",
        question="market outlook",
        conversation_key="chat-1",
        model="fake-model",
        fallback_used=False,
        payload={"kind": "market_update"},
        response_text="response",
        source_coverage={"used_sources": ["fred"]},
    )

    assert result is None
