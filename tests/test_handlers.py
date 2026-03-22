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
