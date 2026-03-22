from __future__ import annotations

from invest_advisor_bot.bot.handlers import _build_main_menu


def test_main_menu_contains_expected_callbacks() -> None:
    menu = _build_main_menu()
    callback_rows = [[button.callback_data for button in row] for row in menu.inline_keyboard]

    assert callback_rows == [
        ["quick:summary", "quick:global_trend"],
        ["quick:gold", "quick:us_stocks"],
        ["quick:etf", "quick:stock_ideas"],
    ]
