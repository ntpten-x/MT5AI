from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from invest_advisor_bot.bot.handlers import (
    BOT_SERVICES_KEY,
    _LAST_REQUEST_BY_CHAT,
    BotServices,
    backup_now_command,
    dashboard_command,
    market_update_command,
    report_now_command,
    scorecard_command,
    status_command,
)
from invest_advisor_bot.bot.jobs import evaluate_stock_pick_scorecard, send_morning_report
from invest_advisor_bot.services.recommendation_service import RecommendationResult


class FakeBot:
    def __init__(self) -> None:
        self.chat_actions: list[tuple[str, str]] = []
        self.sent_messages: list[tuple[str, str]] = []

    async def send_chat_action(self, *, chat_id: str, action: str) -> None:
        self.chat_actions.append((str(chat_id), action))

    async def send_message(self, *, chat_id: str, text: str) -> None:
        self.sent_messages.append((str(chat_id), text))


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
        self.scorecard_rows: list[dict[str, object]] = []
        self.due_rows: list[dict[str, object]] = []
        self.completed_scorecards: list[dict[str, object]] = []
        self.recorded_stock_picks: list[dict[str, object]] = []
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
            "scorecard": {"closed_count": 4, "open_count": 2, "hit_rate_pct": 75.0, "avg_return_pct": 3.2},
            "interactions": {"total": 8, "last_at": "2026-03-22T11:00:00+00:00"},
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

    def build_evaluation_dashboard(self, **kwargs):  # noqa: ANN003
        return dict(self.dashboard_snapshot)


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


class FakeRecommendationService:
    async def generate_market_update(self, **kwargs) -> RecommendationResult:  # noqa: ANN003
        return RecommendationResult(
            recommendation_text="สรุปตลาดล่าสุด: หุ้นสหรัฐยังทรงตัวและควรทยอยสะสมแบบระวังความเสี่ยง",
            model="fake-model",
            system_prompt_path="prompt.txt",
            input_payload={"kind": "market_update"},
            fallback_used=False,
        )

    async def generate_periodic_report(self, **kwargs) -> RecommendationResult:  # noqa: ANN003
        report_kind = kwargs.get("report_kind", "morning")
        return RecommendationResult(
            recommendation_text=f"{report_kind} report: sector rotation ยังหนุนฝั่ง defensive มากกว่า growth",
            model="fake-model",
            system_prompt_path="prompt.txt",
            input_payload={"kind": report_kind},
            fallback_used=True,
        )


def _build_services(
    *,
    report_chat_id: str = "999",
    runtime_history_store: FakeRuntimeHistoryStore | None = None,
    database_url: str = "postgresql://example",
    backup_manager: FakeBackupManager | None = None,
) -> BotServices:
    return BotServices(
        recommendation_service=FakeRecommendationService(),  # type: ignore[arg-type]
        market_data_client=FakeQuoteMarketDataClient(),  # type: ignore[arg-type]
        news_client=object(),  # type: ignore[arg-type]
        research_client=None,
        market_news_limit=5,
        market_history_period="6mo",
        market_history_interval="1d",
        market_history_limit=180,
        telegram_report_chat_id=report_chat_id,
        database_url=database_url,
        runtime_history_store=runtime_history_store,  # type: ignore[arg-type]
        backup_manager=backup_manager,  # type: ignore[arg-type]
        burn_in_target_days=14,
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
    assert history_store.interactions[0]["interaction_kind"] == "report_now"


@pytest.mark.asyncio
async def test_status_command_renders_runtime_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _build_services(report_chat_id="1003")
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
    assert "LLM ล่าสุด: groq" in rendered
    assert "Fallback Rate" in rendered
    assert "Provider Circuit" in rendered


@pytest.mark.asyncio
async def test_dashboard_command_renders_burn_in_and_scorecard_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _build_services(report_chat_id="1007", runtime_history_store=FakeRuntimeHistoryStore())
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
    assert "Provider history" in rendered


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
            "detail": {},
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
    assert bot.sent_messages
    assert "Stock Pick Scorecard Update" in bot.sent_messages[0][1]
