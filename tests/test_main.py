from __future__ import annotations

from pathlib import Path

import pytest

from invest_advisor_bot.config import Settings
from invest_advisor_bot.main import main, resolve_database_url


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example")
    return Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")


def test_resolve_database_url_falls_back_when_psycopg_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(tmp_path, monkeypatch)

    def fake_import_module(name: str):  # noqa: ANN001
        if name == "psycopg":
            raise ImportError("missing psycopg")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("invest_advisor_bot.main.importlib.import_module", fake_import_module)

    assert resolve_database_url(settings) == ""


def test_main_starts_with_file_backend_when_psycopg_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    captured: dict[str, str | None] = {"database_url": None}

    class FakeApplication:
        bot_data: dict[str, object] = {}

        @staticmethod
        def run_polling(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    def fake_import_module(name: str):  # noqa: ANN001
        if name == "psycopg":
            raise ImportError("missing psycopg")
        raise AssertionError(f"unexpected import: {name}")

    def fake_build_application(runtime_settings: Settings, *, database_url: str | None = None) -> FakeApplication:
        assert runtime_settings is settings
        captured["database_url"] = database_url
        return FakeApplication()

    monkeypatch.setattr("invest_advisor_bot.main.get_settings", lambda: settings)
    monkeypatch.setattr("invest_advisor_bot.main.importlib.import_module", fake_import_module)
    monkeypatch.setattr("invest_advisor_bot.main.configure_logging", lambda runtime_settings: None)
    monkeypatch.setattr("invest_advisor_bot.main.log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("invest_advisor_bot.main.build_application", fake_build_application)

    assert main() == 0
    assert captured["database_url"] == ""


def test_main_runs_webhook_when_transport_is_webhook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(tmp_path, monkeypatch)
    settings.telegram_transport = "webhook"
    settings.telegram_webhook_url = "https://example.onrender.com"
    settings.telegram_webhook_path = "/telegram/webhook"
    settings.telegram_webhook_port = 10000
    captured: dict[str, object] = {}

    class FakeApplication:
        bot_data: dict[str, object] = {}

        @staticmethod
        def run_polling(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("run_polling should not be called in webhook mode")

        @staticmethod
        def run_webhook(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            captured["kwargs"] = kwargs
            return None

    monkeypatch.setattr("invest_advisor_bot.main.get_settings", lambda: settings)
    monkeypatch.setattr("invest_advisor_bot.main.resolve_database_url", lambda runtime_settings: "")
    monkeypatch.setattr("invest_advisor_bot.main.configure_logging", lambda runtime_settings: None)
    monkeypatch.setattr("invest_advisor_bot.main.log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("invest_advisor_bot.main.build_application", lambda runtime_settings, database_url=None: FakeApplication())
    monkeypatch.setattr("invest_advisor_bot.main.sync_service_diagnostics", lambda services: None)
    monkeypatch.setattr("invest_advisor_bot.main.stop_health_check_server", lambda: None)

    assert main() == 0
    assert captured["kwargs"]["webhook_url"] == "https://example.onrender.com/telegram/webhook"
    assert captured["kwargs"]["url_path"] == "telegram/webhook"
