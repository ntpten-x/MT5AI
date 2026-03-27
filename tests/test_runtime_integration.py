from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
import httpx

from invest_advisor_bot.bot.handlers import (
    BOT_SERVICES_KEY,
    _LAST_REQUEST_BY_CHAT,
    BotServices,
    analyst_command,
    backup_now_command,
    dashboard_command,
    market_update_command,
    report_now_command,
    scorecard_command,
    status_command,
)
from invest_advisor_bot.bot.jobs import (
    evaluate_stock_pick_scorecard,
    monitor_health_webhook,
    monitor_macro_event_refresh,
    send_morning_report,
)
from invest_advisor_bot.providers.market_data_client import OhlcvBar
from invest_advisor_bot.services.recommendation_service import RecommendationResult


class FakeBot:
    def __init__(self) -> None:
        self.chat_actions: list[tuple[str, str]] = []
        self.sent_messages: list[tuple[str, str]] = []

    async def send_chat_action(self, *, chat_id: str, action: str) -> None:
        self.chat_actions.append((str(chat_id), action))

    async def send_message(self, *, chat_id: str, text: str) -> None:
        self.sent_messages.append((str(chat_id), text))


class _WebhookResponse:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text or str(status_code)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://hooks.example/health")
            raise httpx.HTTPStatusError("error", request=request, response=self)


class _WebhookAsyncClient:
    def __init__(self, *, calls: list[dict[str, object]], queue: list[object] | None = None, **_: object) -> None:
        self._calls = calls
        self._queue = queue or []

    async def __aenter__(self) -> "_WebhookAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, content: bytes | None = None, headers: dict[str, str] | None = None) -> _WebhookResponse:
        payload = json.loads((content or b"{}").decode("utf-8"))
        self._calls.append({"url": url, "json": payload, "body": content or b"", "headers": headers or {}})
        if self._queue:
            queued = self._queue.pop(0)
            if isinstance(queued, Exception):
                raise queued
            return queued  # type: ignore[return-value]
        return _WebhookResponse()


class FakeMessage:
    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, reply_markup=None) -> None:  # noqa: ANN001
        self.replies.append(text)


class FakeRuntimeHistoryStore:
    def __init__(self) -> None:
        self.sent_reports: list[dict[str, object]] = []
        self.interactions: list[dict[str, object]] = []
        self.alert_audits: list[dict[str, object]] = []
        self.scorecard_rows: list[dict[str, object]] = []
        self.due_rows: list[dict[str, object]] = []
        self.completed_scorecards: list[dict[str, object]] = []
        self.recorded_stock_picks: list[dict[str, object]] = []
        self.dashboard_calls: list[dict[str, object]] = []
        self.dashboard_snapshot: dict[str, object] = {
            "available": True,
            "burn_in": {
                "started_at": "2026-03-15T00:00:00+00:00",
                "elapsed_days": 7.0,
                "target_days": 14,
                "progress_pct": 50.0,
            },
            "providers": [{"provider": "groq", "success_count": 12, "failure_count": 1, "last_event_at": "2026-03-22T10:00:00+00:00"}],
            "jobs": [{"job_name": "morning_report", "success_count": 5, "failure_count": 1, "last_event_at": "2026-03-22T10:00:00+00:00"}],
            "alerts": {"total": 4, "by_category": [{"category": "stock_pick", "total": 3}]},
            "reports": {"total": 6, "fallback_total": 2, "last_sent_at": "2026-03-22T10:00:00+00:00", "by_kind": [{"report_kind": "closing", "total": 2}]},
            "scorecard": {
                "closed_count": 4,
                "open_count": 2,
                "hit_rate_pct": 75.0,
                "avg_return_pct": 3.2,
                "avg_return_after_cost_pct": 2.8,
                "avg_alpha_pct": 0.9,
                "avg_alpha_after_cost_pct": 0.5,
            },
            "interactions": {"total": 8, "last_at": "2026-03-22T11:00:00+00:00"},
            "source_ranking": [{"source": "fred", "report_mentions": 4, "interaction_mentions": 3, "stock_pick_total": 2, "closed_count": 2, "hit_rate_pct": 100.0, "avg_return_pct": 4.1, "thesis_alignment_pct": 78.0, "thesis_alignment": "high", "ttl_hit_rate_pct": 66.7, "fast_decay_rate_pct": 16.7, "best_ttl_bucket": "short", "ttl_fit_score": 74.0, "weighted_score": 71.2}],
            "thesis_ranking": [{"thesis": "sticky inflation lowers the setup for duration-sensitive growth", "stock_pick_total": 3, "closed_count": 2, "hit_rate_pct": 100.0, "avg_return_pct": 5.4, "reliability_score": 0.87}],
            "decision_quality": {
                "source_health": {
                    "sample_count": 6,
                    "avg_score": 74.0,
                    "avg_freshness_pct": 79.0,
                    "degraded_sla_count": 1,
                    "outage_count": 0,
                    "strong_count": 4,
                    "mixed_count": 1,
                    "fragile_count": 1,
                },
                "no_trade": {
                    "decision_count": 5,
                    "abstain_count": 2,
                    "abstain_rate_pct": 40.0,
                    "top_reasons": [
                        {"reason": "supporting data coverage is still too thin", "count": 2},
                    ],
                },
            },
            "thesis_lifecycle": {
                "counts": [{"stage": "confirmed", "count": 3}, {"stage": "weakening", "count": 1}],
                "top_invalidations": [{"summary": "stored thesis now has active invalidation pressure", "score": 41.0, "severity": "medium"}],
            },
            "walk_forward_eval": {
                "window_size": 5,
                "window_count": 3,
                "avg_hit_rate_pct": 60.0,
                "avg_return_after_cost_pct": 1.7,
            },
            "execution_panel": {
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
                        "best_ttl_score": 78.0,
                    },
                    {
                        "alert_kind": "macro_surprise",
                        "closed_postmortems": 2,
                        "ttl_hit_rate_pct": 50.0,
                        "fast_decay_rate_pct": 25.0,
                        "hold_after_expiry_rate_pct": 50.0,
                        "discard_after_expiry_rate_pct": 50.0,
                        "best_ttl_bucket": "short",
                        "best_ttl_score": 46.0,
                    },
                ],
                "best_ttl_by_alert_kind": [
                    {
                        "alert_kind": "stock_pick",
                        "best_ttl_bucket": "short",
                        "sample_count": 3,
                        "hit_rate_pct": 66.7,
                        "hold_rate_pct": 66.7,
                        "avg_return_pct": 4.2,
                    }
                ],
                "source_ttl_heatmap": [
                    {
                        "source": "fred",
                        "alert_kind": "stock_pick",
                        "ttl_bucket": "short",
                        "sample_count": 3,
                        "ttl_hit_rate_pct": 66.7,
                        "hold_rate_pct": 66.7,
                        "avg_return_pct": 4.2,
                        "score": 78.0,
                    },
                    {
                        "source": "finra_short_flow",
                        "alert_kind": "stock_pick",
                        "ttl_bucket": "medium",
                        "sample_count": 4,
                        "ttl_hit_rate_pct": 33.3,
                        "hold_rate_pct": 20.0,
                        "avg_return_pct": -1.2,
                        "score": 28.0,
                    }
                    ,
                    {
                        "source": "macro_surprise_engine",
                        "alert_kind": "macro_surprise",
                        "ttl_bucket": "short",
                        "sample_count": 2,
                        "ttl_hit_rate_pct": 50.0,
                        "hold_rate_pct": 50.0,
                        "avg_return_pct": 1.1,
                        "score": 46.0,
                    }
                ],
            },
        }

    def record_sent_report(self, **kwargs) -> None:  # noqa: ANN003
        self.sent_reports.append(kwargs)

    def record_user_interaction(self, **kwargs) -> None:  # noqa: ANN003
        self.interactions.append(kwargs)

    def recent_stock_pick_scorecard(self, **kwargs):  # noqa: ANN003
        return list(self.scorecard_rows)

    def list_due_stock_pick_candidates(self, **kwargs):  # noqa: ANN003
        return list(self.due_rows)

    def complete_stock_pick_evaluation(self, **kwargs) -> None:  # noqa: ANN003
        self.completed_scorecards.append(kwargs)

    def record_stock_pick_candidate(self, **kwargs) -> None:  # noqa: ANN003
        self.recorded_stock_picks.append(kwargs)

    def record_alert_audit(self, **kwargs) -> None:  # noqa: ANN003
        self.alert_audits.append(kwargs)

    def build_evaluation_dashboard(self, **kwargs):  # noqa: ANN003
        self.dashboard_calls.append(dict(kwargs))
        snapshot = dict(self.dashboard_snapshot)
        execution_panel = dict(snapshot.get("execution_panel") or {})
        execution_filter = kwargs.get("execution_alert_kind")
        if isinstance(execution_filter, str) and execution_filter:
            execution_panel["alert_kind_filter"] = execution_filter
            by_alert_kind = execution_panel.get("by_alert_kind")
            if isinstance(by_alert_kind, list):
                execution_panel["by_alert_kind"] = [
                    item for item in by_alert_kind
                    if isinstance(item, dict) and str(item.get("alert_kind") or "") == execution_filter
                ]
            best_ttl = execution_panel.get("best_ttl_by_alert_kind")
            if isinstance(best_ttl, list):
                execution_panel["best_ttl_by_alert_kind"] = [
                    item for item in best_ttl
                    if isinstance(item, dict) and str(item.get("alert_kind") or "") == execution_filter
                ]
            heatmap = execution_panel.get("source_ttl_heatmap")
            if isinstance(heatmap, list):
                execution_panel["source_ttl_heatmap"] = [
                    item for item in heatmap
                    if isinstance(item, dict) and str(item.get("alert_kind") or "") == execution_filter
                ]
        snapshot["execution_panel"] = execution_panel
        return snapshot


class FakeUserStateStore:
    def __init__(self) -> None:
        self.dashboard_execution_filter: str | None = None
        self.approval_mode: str = "auto"
        self.max_position_size_pct: float | None = None

    def get(self, conversation_key: str):  # noqa: ANN001
        return SimpleNamespace(
            watchlist=(),
            preferred_sectors=(),
            stock_alert_threshold=1.8,
            daily_pick_enabled=True,
            dashboard_execution_filter=self.dashboard_execution_filter,
            approval_mode=self.approval_mode,
            max_position_size_pct=self.max_position_size_pct,
        )

    def update_preferences(self, conversation_key: str, **kwargs):  # noqa: ANN003
        if "dashboard_execution_filter" in kwargs:
            self.dashboard_execution_filter = kwargs.get("dashboard_execution_filter")
        if "approval_mode" in kwargs and kwargs.get("approval_mode") is not None:
            self.approval_mode = kwargs.get("approval_mode")
        if "max_position_size_pct" in kwargs:
            self.max_position_size_pct = kwargs.get("max_position_size_pct")
        return self.get(conversation_key)


class FakeBackupManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def create_backup(self, *, reason: str):
        self.calls.append(reason)
        return SimpleNamespace(
            path="backups/invest_advisor_backup_20260322T120000Z.json",
            created_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            row_counts={"bot_user_preferences": 1, "bot_stock_pick_scorecard": 2},
        )


class FakeQuoteMarketDataClient:
    async def get_latest_price(self, ticker: str):  # noqa: ANN001
        return SimpleNamespace(price=110.0 if ticker.upper() == "AAPL" else 100.0)


class FakeHistoryAwareMarketDataClient(FakeQuoteMarketDataClient):
    async def get_history(self, ticker: str, *, period: str, interval: str, limit: int | None = None):  # noqa: ANN003
        return [
            OhlcvBar(
                ticker=ticker.upper(),
                timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.5,
                volume=1000,
            )
        ]


class FakeAlertStateStore:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def filter_new_keys(self, keys):  # noqa: ANN001
        fresh: list[str] = []
        for key in keys:
            key_text = str(key)
            if key_text in self.seen:
                continue
            self.seen.add(key_text)
            fresh.append(key_text)
        return fresh


class FakeMLflowObserver:
    def __init__(
        self,
        *,
        enabled: bool = True,
        tracking_configured: bool = True,
        experiment_name: str = "invest-advisor-bot",
        warning: str | None = None,
        last_run_id: str | None = "mlflow-eval-123",
        last_run_kind: str | None = "evaluation",
    ) -> None:
        self.evaluation_calls: list[dict[str, object]] = []
        self._status = {
            "enabled": enabled,
            "tracking_configured": tracking_configured,
            "experiment_name": experiment_name,
            "warning": warning,
            "last_run_id": last_run_id,
            "last_run_kind": last_run_kind,
        }

    def log_evaluation(self, **kwargs) -> str:  # noqa: ANN003
        self.evaluation_calls.append(kwargs)
        return "mlflow-eval-123"

    def status(self) -> dict[str, object]:
        return dict(self._status)


class FakeRecommendationService:
    def __init__(
        self,
        *,
        mlflow_observer: object | None = None,
        status_payload: dict[str, object] | None = None,
        analyst_answer: str = "fallback rate is elevated in recent recommendation events",
    ) -> None:
        self.mlflow_observer = mlflow_observer
        self._status_payload = status_payload or {}
        self._analyst_answer = analyst_answer

    async def generate_market_update(self, **kwargs) -> RecommendationResult:  # noqa: ANN003
        return RecommendationResult(
            recommendation_text="สรุปตลาดล่าสุด: หุ้นสหรัฐยังทรงตัวและควรทยอยสะสมแบบระวังความเสี่ยง",
            model="fake-model",
            system_prompt_path="prompt.txt",
            input_payload={
                "kind": "market_update",
                "macro_intelligence": {"sources_used": ["fred", "bls"]},
                "macro_event_calendar": [{"source": "federal_reserve"}],
                "macro_market_reactions": [{"reactions": [{"label": "SPY"}]}],
                "news_headlines": [{"title": "Fed steady"}],
            },
            fallback_used=False,
        )

    async def generate_periodic_report(self, **kwargs) -> RecommendationResult:  # noqa: ANN003
        report_kind = kwargs.get("report_kind", "morning")
        return RecommendationResult(
            recommendation_text=f"{report_kind} report: sector rotation ยังหนุนฝั่ง defensive มากกว่า growth",
            model="fake-model",
            system_prompt_path="prompt.txt",
            input_payload={
                "kind": report_kind,
                "macro_intelligence": {"sources_used": ["fred", "bls", "treasury"]},
                "macro_event_calendar": [{"source": "fred_release_calendar"}],
                "macro_market_reactions": [{"reactions": [{"label": "SPY"}, {"label": "QQQ"}]}],
                "company_intelligence": [{"ticker": "AAPL"}],
                "stock_picks": [{"ticker": "AAPL", "price": 110.0, "score": 2.5, "confidence_score": 0.8, "confidence_label": "high", "stance": "buy"}],
            },
            fallback_used=True,
        )

    async def generate_macro_event_driven_alerts(self, **kwargs):  # noqa: ANN003
        return [
            SimpleNamespace(
                key="macro_event_refresh:cpi:2026-03-23T12:30:00+00:00",
                severity="warning",
                text="Macro refresh alert",
            ),
            SimpleNamespace(
                key="macro_playbook_window:hot_cpi_no_market_confirm",
                severity="warning",
                text="Macro playbook alert",
            ),
        ]

    async def answer_analytics_question(self, *, question: str) -> str:
        return f"{self._analyst_answer} | question={question}"

    def status(self) -> dict[str, object]:
        return dict(self._status_payload)


def _build_services(
    *,
    report_chat_id: str = "999",
    runtime_history_store: FakeRuntimeHistoryStore | None = None,
    database_url: str = "postgresql://example",
    backup_manager: FakeBackupManager | None = None,
    alert_state_store: FakeAlertStateStore | None = None,
    recommendation_service: object | None = None,
    market_data_client: object | None = None,
    user_state_store: object | None = None,
    health_alert_webhook_url: str = "",
    health_alert_webhook_secret: str = "",
    health_alert_cooldown_minutes: int = 30,
    health_alert_timeout_seconds: float = 8.0,
    health_alert_retry_count: int = 3,
    health_alert_retry_backoff_seconds: float = 1.5,
) -> BotServices:
    return BotServices(
        recommendation_service=(recommendation_service or FakeRecommendationService()),  # type: ignore[arg-type]
        market_data_client=(market_data_client or FakeQuoteMarketDataClient()),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        market_news_limit=5,
        market_history_period="6mo",
        market_history_interval="1d",
        market_history_limit=180,
        telegram_report_chat_id=report_chat_id,
        database_url=database_url,
        alert_state_store=alert_state_store,  # type: ignore[arg-type]
        user_state_store=user_state_store,  # type: ignore[arg-type]
        runtime_history_store=runtime_history_store,  # type: ignore[arg-type]
        backup_manager=backup_manager,  # type: ignore[arg-type]
        burn_in_target_days=14,
        health_alert_webhook_url=health_alert_webhook_url,
        health_alert_webhook_secret=health_alert_webhook_secret,
        health_alert_cooldown_minutes=health_alert_cooldown_minutes,
        health_alert_timeout_seconds=health_alert_timeout_seconds,
        health_alert_retry_count=health_alert_retry_count,
        health_alert_retry_backoff_seconds=health_alert_retry_backoff_seconds,
    )


def _build_context(services: BotServices, *, args: list[str] | None = None, bot: FakeBot | None = None):
    return SimpleNamespace(
        args=args or [],
        bot=bot or FakeBot(),
        application=SimpleNamespace(bot_data={BOT_SERVICES_KEY: services}),
    )


def _build_update(*, chat_id: str, text: str | None = None):
    message = FakeMessage(text=text)
    chat = SimpleNamespace(id=chat_id)
    return SimpleNamespace(effective_message=message, effective_chat=chat), message


@pytest.mark.asyncio
async def test_market_update_command_is_live_safe_and_records_interaction() -> None:
    _LAST_REQUEST_BY_CHAT.clear()
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(report_chat_id="1001", runtime_history_store=history_store)
    context = _build_context(services)
    update, message = _build_update(chat_id="1001")

    await market_update_command(update, context)

    assert context.bot.chat_actions == [("1001", "typing")]
    assert any("กำลังสรุปภาพรวมตลาด" in reply for reply in message.replies)
    assert any("สรุปตลาดล่าสุด" in reply for reply in message.replies)
    assert history_store.interactions
    assert history_store.interactions[0]["interaction_kind"] == "market_update"
    assert "fred" in history_store.interactions[0]["detail"]["source_coverage"]["used_sources"]


@pytest.mark.asyncio
async def test_report_now_command_sends_report_and_records_history() -> None:
    _LAST_REQUEST_BY_CHAT.clear()
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(report_chat_id="1002", runtime_history_store=history_store)
    context = _build_context(services, args=["closing"])
    update, message = _build_update(chat_id="1002")

    await report_now_command(update, context)

    assert any("กำลังสร้างรายงาน closing" in reply for reply in message.replies)
    assert any("closing report:" in reply for reply in message.replies)
    assert history_store.sent_reports
    assert history_store.sent_reports[0]["report_kind"] == "closing"
    assert "company_intelligence" in history_store.sent_reports[0]["detail"]["source_coverage"]["used_sources"]
    assert history_store.interactions[0]["interaction_kind"] == "report_now"


@pytest.mark.asyncio
async def test_status_command_renders_runtime_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    recommendation_service = FakeRecommendationService(
        mlflow_observer=FakeMLflowObserver(),
        status_payload={
            "analytics_warehouse": {"available": True, "configured": False, "backend": "jsonl", "counts": {"recommendation_events": 2}},
            "event_bus": {"available": True, "configured": False, "backend": "jsonl", "published_count": 4},
            "event_bus_consumer": {"available": True, "configured": True, "backend": "jsonl", "processed_count": 6},
            "hot_path_cache": {"available": True, "configured": False, "backend": "memory", "cache_keys": 2, "stream_event_count": 5},
            "semantic_analyst": {"available": True, "configured": False, "backend": "local-heuristic", "model_name": "local-heuristic"},
        },
    )
    services = _build_services(report_chat_id="1003", recommendation_service=recommendation_service, database_url="")
    context = _build_context(services)
    update, message = _build_update(chat_id="1003")

    snapshot = {
        "latest_provider_success": {
            "provider": "groq",
            "model": "llama-3.3",
            "service": "recommendation_service",
            "succeeded_at": "2026-03-22T10:00:00+00:00",
        },
        "response_stats": {
            "recommendation_service": {"total": 10, "fallback": 2, "fallback_rate": 20.0},
        },
        "jobs": {
            "morning_report": {
                "last_status": "ok",
                "last_run_at": "2026-03-22T10:05:00+00:00",
                "duration_ms": 1200,
                "success_count": 3,
                "failure_count": 1,
                "last_error": None,
            }
        },
        "alerts_today": {"total": 4, "by_category": {"risk": 1, "stock_pick": 3}},
        "db_state": {"backend": "postgres", "healthy": True, "checked_at": "2026-03-22T10:00:00+00:00", "error": None},
        "provider_circuit": {"groq": {"is_open": False, "failure_count": 0, "open_until": None}},
    }
    monkeypatch.setattr("invest_advisor_bot.bot.handlers.diagnostics.snapshot", lambda: snapshot)
    monkeypatch.setattr("invest_advisor_bot.bot.handlers.PostgresStateBackend.ping_database_url", lambda url: True)

    await status_command(update, context)

    rendered = "\n".join(message.replies)
    assert "Runtime Status" in rendered
    assert "DB: postgres | healthy=True" in rendered
    assert "MLflow: enabled=True | tracking=True" in rendered
    assert "last_run_id=mlflow-eval-123" in rendered
    assert "LLM ล่าสุด: groq" in rendered
    assert "Fallback Rate" in rendered
    assert "Provider Circuit" in rendered
    assert "Analytics Warehouse: available=True" in rendered
    assert "Event Bus: available=True" in rendered
    assert "Event Bus Consumer: available=True" in rendered
    assert "Hot Cache: available=True" in rendered
    assert "Semantic Analyst: available=True" in rendered


@pytest.mark.asyncio
async def test_status_command_surfaces_mlflow_warning() -> None:
    mlflow_observer = FakeMLflowObserver(
        enabled=False,
        tracking_configured=True,
        warning="mlflow_dependency_missing: install with pip install -e .[observability]",
        last_run_id=None,
        last_run_kind=None,
    )
    recommendation_service = FakeRecommendationService(mlflow_observer=mlflow_observer)
    services = _build_services(report_chat_id="1003", recommendation_service=recommendation_service)
    context = _build_context(services)
    update, message = _build_update(chat_id="1003")

    await status_command(update, context)

    rendered = "\n".join(message.replies)
    assert "MLflow: enabled=False | tracking=True" in rendered
    assert "MLflow warning: mlflow_dependency_missing" in rendered


@pytest.mark.asyncio
async def test_analyst_command_uses_semantic_analyst_answer() -> None:
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(
        report_chat_id="1003",
        runtime_history_store=history_store,
        recommendation_service=FakeRecommendationService(
            analyst_answer="warehouse shows recommendation fallback trending lower",
        ),
    )
    context = _build_context(services, args=["recommendation", "fallback", "trend"])
    update, message = _build_update(chat_id="1003")

    await analyst_command(update, context)

    rendered = "\n".join(message.replies)
    assert "Analytics Analyst" in rendered
    assert "warehouse shows recommendation fallback trending lower" in rendered
    assert history_store.interactions
    assert history_store.interactions[-1]["interaction_kind"] == "analytics_analyst"


@pytest.mark.asyncio
async def test_monitor_health_webhook_posts_incident_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    services = _build_services(
        health_alert_webhook_url="https://hooks.example/health",
        health_alert_webhook_secret="top-secret",
        health_alert_cooldown_minutes=30,
    )
    context = _build_context(services)

    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.collect_runtime_snapshot",
        lambda services, ping_database=False: {
            "started_at": "2026-03-24T11:00:00+00:00",
            "uptime_seconds": 120.0,
            "db_state": {"backend": "postgres", "healthy": False, "error": "timeout"},
            "provider_circuit": {"groq": {"is_open": True, "failure_count": 3, "open_until": "2026-03-24T11:05:00+00:00"}},
            "mlflow": {"enabled": False, "tracking_configured": True, "warning": "dependency missing"},
        },
    )
    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.httpx.AsyncClient",
        lambda **kwargs: _WebhookAsyncClient(calls=calls, **kwargs),
    )

    await monitor_health_webhook(context)

    assert len(calls) == 1
    payload = calls[0]["json"]
    assert payload["event_type"] == "incident"
    assert payload["issue_count"] == 3
    assert "database backend postgres is unhealthy" in payload["summary"]
    assert payload["issues"][1]["kind"] == "provider_circuit_open"
    assert payload["incident_key"].startswith("db_unhealthy:")
    assert payload["idempotency_key"] == calls[0]["headers"]["X-Invest-Advisor-Idempotency-Key"]
    signature = calls[0]["headers"]["X-Invest-Advisor-Signature"]
    timestamp = calls[0]["headers"]["X-Invest-Advisor-Timestamp"]
    expected = hmac.new(
        b"top-secret",
        timestamp.encode("utf-8") + b"." + calls[0]["body"],
        hashlib.sha256,
    ).hexdigest()
    assert signature == f"sha256={expected}"


@pytest.mark.asyncio
async def test_monitor_health_webhook_respects_cooldown_for_same_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    services = _build_services(
        health_alert_webhook_url="https://hooks.example/health",
        health_alert_cooldown_minutes=30,
    )
    context = _build_context(services)
    snapshot = {
        "started_at": "2026-03-24T11:00:00+00:00",
        "uptime_seconds": 120.0,
        "db_state": {"backend": "postgres", "healthy": False, "error": "timeout"},
        "provider_circuit": {},
        "mlflow": {"enabled": True, "tracking_configured": True, "warning": None},
    }

    monkeypatch.setattr("invest_advisor_bot.bot.jobs.collect_runtime_snapshot", lambda services, ping_database=False: snapshot)
    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.httpx.AsyncClient",
        lambda **kwargs: _WebhookAsyncClient(calls=calls, **kwargs),
    )

    await monitor_health_webhook(context)
    await monitor_health_webhook(context)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_monitor_health_webhook_posts_resolved_event_after_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    services = _build_services(
        health_alert_webhook_url="https://hooks.example/health",
        health_alert_cooldown_minutes=30,
    )
    context = _build_context(services)
    snapshots = [
        {
            "started_at": "2026-03-24T11:00:00+00:00",
            "uptime_seconds": 120.0,
            "db_state": {"backend": "postgres", "healthy": False, "error": "timeout"},
            "provider_circuit": {},
            "mlflow": {"enabled": True, "tracking_configured": True, "warning": None},
        },
        {
            "started_at": "2026-03-24T11:00:00+00:00",
            "uptime_seconds": 180.0,
            "db_state": {"backend": "postgres", "healthy": True, "error": None},
            "provider_circuit": {},
            "mlflow": {"enabled": True, "tracking_configured": True, "warning": None},
        },
    ]

    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.collect_runtime_snapshot",
        lambda services, ping_database=False: snapshots.pop(0),
    )
    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.httpx.AsyncClient",
        lambda **kwargs: _WebhookAsyncClient(calls=calls, **kwargs),
    )

    await monitor_health_webhook(context)
    await monitor_health_webhook(context)

    assert len(calls) == 2
    assert calls[0]["json"]["event_type"] == "incident"
    assert calls[1]["json"]["event_type"] == "resolved"
    assert calls[1]["json"]["summary"] == "runtime health recovered"
    assert calls[0]["json"]["idempotency_key"] != calls[1]["json"]["idempotency_key"]


@pytest.mark.asyncio
async def test_monitor_health_webhook_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    services = _build_services(
        health_alert_webhook_url="https://hooks.example/health",
        health_alert_retry_count=2,
        health_alert_retry_backoff_seconds=0.1,
    )
    context = _build_context(services)

    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.collect_runtime_snapshot",
        lambda services, ping_database=False: {
            "started_at": "2026-03-24T11:00:00+00:00",
            "uptime_seconds": 120.0,
            "db_state": {"backend": "postgres", "healthy": False, "error": "timeout"},
            "provider_circuit": {},
            "mlflow": {"enabled": True, "tracking_configured": True, "warning": None},
        },
    )
    async def _noop_sleep(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("invest_advisor_bot.bot.jobs.asyncio.sleep", _noop_sleep)
    monkeypatch.setattr(
        "invest_advisor_bot.bot.jobs.httpx.AsyncClient",
        lambda **kwargs: _WebhookAsyncClient(
            calls=calls,
            queue=[_WebhookResponse(status_code=503), _WebhookResponse(status_code=502), _WebhookResponse(status_code=200)],
            **kwargs,
        ),
    )

    await monitor_health_webhook(context)

    assert len(calls) == 3
    assert calls[0]["headers"]["X-Invest-Advisor-Idempotency-Key"] == calls[1]["headers"]["X-Invest-Advisor-Idempotency-Key"]
    assert calls[1]["headers"]["X-Invest-Advisor-Idempotency-Key"] == calls[2]["headers"]["X-Invest-Advisor-Idempotency-Key"]


@pytest.mark.asyncio
async def test_dashboard_command_renders_burn_in_and_scorecard_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    recommendation_service = FakeRecommendationService(mlflow_observer=FakeMLflowObserver())
    services = _build_services(
        report_chat_id="1007",
        runtime_history_store=FakeRuntimeHistoryStore(),
        recommendation_service=recommendation_service,
    )
    context = _build_context(services)
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    rendered = "\n".join(message.replies)
    assert "Evaluation Dashboard" in rendered
    assert "Burn-in" in rendered
    assert "Scorecard" in rendered
    assert "MLflow: enabled=True | tracking=True" in rendered
    assert "avg alpha=0.9%" in rendered
    assert "avg alpha after cost=0.5%" in rendered
    assert "Provider history" in rendered
    assert "Source ranking" in rendered
    assert "Thesis ranking" in rendered
    assert "Execution panel" in rendered
    assert "Execution by alert kind" in rendered
    assert "fast_decay=20.0%" in rendered
    assert "Best TTL by alert kind" in rendered
    assert "Source TTL heatmap" in rendered
    assert "Source health" in rendered
    assert "avg_score=74.0" in rendered
    assert "degraded_sla=1" in rendered
    assert "No-trade framework" in rendered
    assert "supporting data coverage is still too thin" in rendered
    assert "Thesis lifecycle" in rendered
    assert "Walk-forward eval" in rendered
    assert "fred -> stock_pick -> short" in rendered
    assert "best_ttl=short" in rendered
    assert "ttl_fit=74.0" in rendered
    assert "weighted=" in rendered


@pytest.mark.asyncio
async def test_dashboard_command_surfaces_mlflow_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    history_store = FakeRuntimeHistoryStore()
    recommendation_service = FakeRecommendationService(
        mlflow_observer=FakeMLflowObserver(
            enabled=False,
            tracking_configured=True,
            warning="mlflow_dependency_missing: install with pip install -e .[observability]",
            last_run_id=None,
            last_run_kind=None,
        )
    )
    services = _build_services(
        report_chat_id="1007",
        runtime_history_store=history_store,
        recommendation_service=recommendation_service,
    )
    context = _build_context(services)
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    rendered = "\n".join(message.replies)
    assert "MLflow: enabled=False | tracking=True" in rendered
    assert "MLflow warning: mlflow_dependency_missing" in rendered


@pytest.mark.asyncio
async def test_dashboard_command_filters_execution_panel_by_alert_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(report_chat_id="1007", runtime_history_store=history_store)
    context = _build_context(services, args=["stock_pick"])
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    rendered = "\n".join(message.replies)
    assert history_store.dashboard_calls[-1]["execution_alert_kind"] == "stock_pick"
    assert "Execution filter: stock_pick" in rendered
    assert "Execution by alert kind" in rendered
    assert "fred -> stock_pick -> short" in rendered
    assert "finra_short_flow -> stock_pick -> medium" in rendered
    assert "macro_surprise_engine -> macro_surprise -> short" not in rendered


@pytest.mark.asyncio
async def test_dashboard_command_persists_execution_filter_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    history_store = FakeRuntimeHistoryStore()
    user_state_store = FakeUserStateStore()
    services = _build_services(
        report_chat_id="1007",
        runtime_history_store=history_store,
        user_state_store=user_state_store,
    )
    context = _build_context(services, args=["macro_surprise"])
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    assert user_state_store.dashboard_execution_filter == "macro_surprise"

    second_context = _build_context(services)
    second_update, second_message = _build_update(chat_id="1007")
    await dashboard_command(second_update, second_context)

    rendered = "\n".join(second_message.replies)
    assert history_store.dashboard_calls[-1]["execution_alert_kind"] == "macro_surprise"
    assert "Execution filter: macro_surprise" in rendered


@pytest.mark.asyncio
async def test_dashboard_command_supports_order_independent_filter_and_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(report_chat_id="1007", runtime_history_store=history_store)
    context = _build_context(services, args=["ttl_hit", "stock_pick"])
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    rendered = "\n".join(message.replies)
    assert history_store.dashboard_calls[-1]["execution_alert_kind"] == "stock_pick"
    assert "Execution filter: stock_pick" in rendered
    assert "Execution sort: ttl_hit" in rendered
    assert rendered.index("fred -> stock_pick -> short") < rendered.index("finra_short_flow -> stock_pick -> medium")


@pytest.mark.asyncio
async def test_dashboard_command_supports_worst_sort_for_filtered_heatmap(monkeypatch: pytest.MonkeyPatch) -> None:
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(report_chat_id="1007", runtime_history_store=history_store)
    context = _build_context(services, args=["stock_pick", "worst"])
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    rendered = "\n".join(message.replies)
    assert "Execution sort: worst" in rendered
    assert rendered.index("finra_short_flow -> stock_pick -> medium") < rendered.index("fred -> stock_pick -> short")


@pytest.mark.asyncio
async def test_dashboard_command_uses_persisted_filter_with_explicit_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    history_store = FakeRuntimeHistoryStore()
    user_state_store = FakeUserStateStore()
    user_state_store.dashboard_execution_filter = "stock_pick"
    services = _build_services(
        report_chat_id="1007",
        runtime_history_store=history_store,
        user_state_store=user_state_store,
    )
    context = _build_context(services, args=["worst"])
    update, message = _build_update(chat_id="1007")

    monkeypatch.setattr(
        "invest_advisor_bot.bot.handlers.diagnostics.snapshot",
        lambda: {
            "latest_provider_success": {"provider": "groq", "model": "llama", "service": "recommendation_service"},
            "response_stats": {"recommendation_service": {"fallback_rate": 10.0, "fallback": 1, "total": 10}},
        },
    )

    await dashboard_command(update, context)

    rendered = "\n".join(message.replies)
    assert history_store.dashboard_calls[-1]["execution_alert_kind"] == "stock_pick"
    assert "Execution filter: stock_pick" in rendered
    assert "Execution sort: worst" in rendered
    assert rendered.index("finra_short_flow -> stock_pick -> medium") < rendered.index("fred -> stock_pick -> short")


@pytest.mark.asyncio
async def test_backup_now_command_runs_backup_manager() -> None:
    services = _build_services(
        report_chat_id="1008",
        runtime_history_store=FakeRuntimeHistoryStore(),
        backup_manager=FakeBackupManager(),
    )
    context = _build_context(services)
    update, message = _build_update(chat_id="1008")

    await backup_now_command(update, context)

    rendered = "\n".join(message.replies)
    assert "Backup Complete" in rendered
    assert "bot_user_preferences=1" in rendered


@pytest.mark.asyncio
async def test_scheduled_job_sends_message_to_telegram_and_records_report() -> None:
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(report_chat_id="1004", runtime_history_store=history_store)
    bot = FakeBot()
    context = _build_context(services, bot=bot)

    await send_morning_report(context)

    assert bot.sent_messages
    assert bot.sent_messages[0][0] == "1004"
    assert "morning report:" in bot.sent_messages[0][1]
    assert history_store.sent_reports
    assert history_store.sent_reports[0]["report_kind"] == "morning"
    assert "fred" in history_store.sent_reports[0]["detail"]["source_coverage"]["used_sources"]


@pytest.mark.asyncio
async def test_macro_event_refresh_job_sends_deduped_alerts_and_records_audits() -> None:
    history_store = FakeRuntimeHistoryStore()
    services = _build_services(
        report_chat_id="1010",
        runtime_history_store=history_store,
        alert_state_store=FakeAlertStateStore(),
    )
    bot = FakeBot()
    context = _build_context(services, bot=bot)

    await monitor_macro_event_refresh(context)
    await monitor_macro_event_refresh(context)

    assert len(bot.sent_messages) == 2
    assert bot.sent_messages[0][0] == "1010"
    assert history_store.alert_audits


@pytest.mark.asyncio
async def test_macro_event_refresh_job_skips_expired_alerts() -> None:
    class ExpiringRecommendationService(FakeRecommendationService):
        async def generate_macro_event_driven_alerts(self, **kwargs):  # noqa: ANN003
            return [
                SimpleNamespace(
                    key="macro_event_refresh:expired",
                    severity="warning",
                    text="expired alert",
                    metadata={"expires_at": "2026-03-22T00:00:00+00:00"},
                ),
                SimpleNamespace(
                    key="macro_event_refresh:active",
                    severity="warning",
                    text="active alert",
                    metadata={"expires_at": "2026-03-25T00:00:00+00:00"},
                ),
            ]

    history_store = FakeRuntimeHistoryStore()
    services = _build_services(
        report_chat_id="1011",
        runtime_history_store=history_store,
        alert_state_store=FakeAlertStateStore(),
        recommendation_service=ExpiringRecommendationService(),
    )
    bot = FakeBot()
    context = _build_context(services, bot=bot)

    await monitor_macro_event_refresh(context)

    assert len(bot.sent_messages) == 1
    assert "active alert" in bot.sent_messages[0][1]
    assert history_store.alert_audits[0]["detail"]["expires_at"] == "2026-03-25T00:00:00+00:00"


@pytest.mark.asyncio
async def test_scorecard_command_renders_recent_rows() -> None:
    history_store = FakeRuntimeHistoryStore()
    history_store.scorecard_rows = [
        {
            "ticker": "AAPL",
            "status": "closed",
            "return_pct": 0.084,
            "confidence_label": "high",
            "confidence_score": 0.81,
            "composite_score": 2.9,
            "detail": {
                "return_after_cost_pct": 0.081,
                "alpha_after_cost_pct": 0.024,
                "benchmark_ticker": "SPY",
                "thesis_summary": "sticky inflation lowers the setup for duration-sensitive growth",
                "postmortem_action": "hold_thesis",
                "signal_decay_label": "durable",
            },
            "created_at": "2026-03-22T10:00:00+00:00",
        }
    ]
    services = _build_services(report_chat_id="1005", runtime_history_store=history_store)
    context = _build_context(services)
    update, message = _build_update(chat_id="1005")

    await scorecard_command(update, context)

    rendered = "\n".join(message.replies)
    assert "Stock Pick Scorecard" in rendered
    assert "AAPL" in rendered
    assert "high" in rendered
    assert "after cost +8.1%" in rendered
    assert "alpha vs SPY +2.4%" in rendered
    assert "duration-sensitive growth" in rendered
    assert "hold_thesis" in rendered


@pytest.mark.asyncio
async def test_evaluate_stock_pick_scorecard_records_completed_rows_and_sends_summary() -> None:
    history_store = FakeRuntimeHistoryStore()
    history_store.due_rows = [
        {
            "id": 1,
            "ticker": "AAPL",
            "entry_price": 100.0,
            "source_kind": "daily_pick",
        }
    ]
    services = _build_services(report_chat_id="1006", runtime_history_store=history_store)
    bot = FakeBot()
    context = _build_context(services, bot=bot)

    await evaluate_stock_pick_scorecard(context)

    assert history_store.completed_scorecards
    assert history_store.completed_scorecards[0]["outcome_label"] in {"win", "strong_win"}
    assert history_store.completed_scorecards[0]["detail"]["return_after_cost_pct"] == 0.1
    assert bot.sent_messages
    assert "Stock Pick Scorecard Update" in bot.sent_messages[0][1]


@pytest.mark.asyncio
async def test_evaluate_stock_pick_scorecard_records_postmortem_for_expired_signal() -> None:
    history_store = FakeRuntimeHistoryStore()
    history_store.due_rows = [
        {
            "id": 2,
            "recommendation_key": "alert-aapl-1",
            "ticker": "AAPL",
            "entry_price": 100.0,
            "source_kind": "daily_pick",
            "stance": "buy",
            "created_at": "2026-03-23T03:00:00+00:00",
            "detail": {
                "ttl_minutes": 120,
                "expires_at": "2026-03-23T05:00:00+00:00",
                "alert_kind": "stock_pick",
                "sector": "Technology",
                "peer_benchmark_ticker": "XLK",
                "execution_realism": {"execution_cost_bps": 25.0},
            },
        }
    ]
    recommendation_service = FakeRecommendationService()
    mlflow_observer = FakeMLflowObserver()
    recommendation_service.mlflow_observer = mlflow_observer
    services = _build_services(
        report_chat_id="1007",
        runtime_history_store=history_store,
        market_data_client=FakeHistoryAwareMarketDataClient(),
        recommendation_service=recommendation_service,
    )
    bot = FakeBot()
    context = _build_context(services, bot=bot)

    await evaluate_stock_pick_scorecard(context)

    assert history_store.completed_scorecards
    detail = history_store.completed_scorecards[0]["detail"]
    assert detail["ttl_hit"] is True
    assert detail["return_after_cost_pct"] == 0.0975
    assert detail["benchmark_ticker"] == "SPY"
    assert detail["benchmark_return_pct"] == -0.0148
    assert detail["alpha_after_cost_pct"] == 0.1123
    assert detail["sector_benchmark_ticker"] == "XLK"
    assert detail["peer_benchmark_ticker"] == "XLK"
    assert detail["postmortem_action"] == "hold_thesis"
    assert detail["signal_decay_label"] == "durable"
    assert detail["mlflow_run_id"] == "mlflow-eval-123"
    assert mlflow_observer.evaluation_calls
    assert mlflow_observer.evaluation_calls[0]["tags"]["artifact_key"] == "alert-aapl-1"
    assert mlflow_observer.evaluation_calls[0]["metrics"]["return_after_cost_pct"] == 0.0975
    assert "alpha vs SPY +11.2%" in bot.sent_messages[0][1]
    assert "postmortem hold_thesis" in bot.sent_messages[0][1]
