from __future__ import annotations

from types import SimpleNamespace

import pytest

from invest_advisor_bot.bot import handlers


class DummyMessage:
    def __init__(self) -> None:
        self.replies: list[tuple[str, object | None]] = []

    async def reply_text(self, text: str, reply_markup=None) -> None:
        self.replies.append((text, reply_markup))


class DummyQuery:
    def __init__(self, data: str, message: DummyMessage) -> None:
        self.data = data
        self.message = message
        self.answered = False

    async def answer(self, *args, **kwargs) -> None:
        self.answered = True


@pytest.mark.asyncio
async def test_handle_menu_action_dispatches_to_expected_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    message = DummyMessage()
    query = DummyQuery(handlers.CALLBACK_MENU_HELP, message)
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace()
    called = {"help": False}

    async def fake_help_command(update, context) -> None:
        called["help"] = True
        await update.callback_query.message.reply_text("help ok")

    monkeypatch.setattr(handlers, "help_command", fake_help_command)

    await handlers.handle_menu_action(update, context)

    assert query.answered is True
    assert called["help"] is True
    assert message.replies[0][0] == "help ok"


def test_main_menu_includes_quick_actions_and_shortcut_buttons() -> None:
    markup = handlers._build_main_menu()
    callback_data = {
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    }

    assert handlers.CALLBACK_QUICK_SUMMARY in callback_data
    assert handlers.CALLBACK_MENU_PROFILE in callback_data
    assert handlers.CALLBACK_MENU_PORTFOLIO in callback_data
    assert handlers.CALLBACK_MENU_WATCHLIST in callback_data
    assert handlers.CALLBACK_MENU_PREFS in callback_data
    assert handlers.CALLBACK_MENU_REPORT_NOW in callback_data
    assert handlers.CALLBACK_MENU_MARKET_UPDATE in callback_data


def test_render_llm_provider_status_lines_includes_state_and_model() -> None:
    lines = handlers._render_llm_provider_status_lines(
        {
            "provider_statuses": [
                {
                    "provider": "cloudflare",
                    "state": "ok",
                    "last_model": "@cf/meta/llama-3.1-8b-instruct-fast",
                    "failure_count": 0,
                },
                {
                    "provider": "gemini",
                    "state": "rate_limited",
                    "last_status_code": 429,
                    "failure_count": 1,
                    "cooldown_until": "2026-03-27T10:00:00+00:00",
                },
            ]
        }
    )

    assert lines[0].startswith("- cloudflare: ok | model=@cf/meta/llama-3.1-8b-instruct-fast")
    assert "gemini: rate_limited" in lines[1]
    assert "code=429" in lines[1]


def test_classify_component_status_prefers_rate_limit_and_restricted_signals() -> None:
    assert handlers._classify_component_status({"warning": "429 quota exceeded", "available": False}) == "rate_limited"
    assert handlers._classify_component_status({"warning": "403 forbidden", "disabled": True}) == "restricted"
    assert handlers._classify_nasdaq_status(
        {"nasdaq_data_link_disabled": True, "nasdaq_data_link_datasets": ["FRED/GDP"]}
    ) == "disabled"
    assert handlers._classify_alpha_vantage_status(
        {"alpha_vantage_disabled": True, "configured_sources": {"alpha_vantage": False}}
    ) == "disabled"
    assert handlers._render_transcript_fallback_label(
        {"available": True, "provider_order": ["tavily", "exa"]}
    ) == "research_proxy[tavily+exa]"
