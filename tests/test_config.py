from __future__ import annotations

from pathlib import Path

import pytest

from invest_advisor_bot.config import Settings


def test_settings_accept_runtime_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    settings.validate_runtime()
    assert settings.llm_available() is False
    assert settings.research_available() is False
    assert settings.default_investor_profile == "conservative"
    assert settings.project_root.name == "MT5AI"


def test_settings_detects_non_openai_llm_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")

    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    assert settings.llm_available() is True
    assert settings.research_available() is True
