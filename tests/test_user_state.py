from __future__ import annotations

from pathlib import Path

from invest_advisor_bot.bot.user_state import UserStateStore


def test_user_state_store_persists_dashboard_execution_filter(tmp_path: Path) -> None:
    path = tmp_path / "user_state.json"
    store = UserStateStore(path=path)

    prefs = store.update_preferences("chat-1", dashboard_execution_filter="macro_surprise")
    reloaded = UserStateStore(path=path)
    loaded = reloaded.get("chat-1")

    assert prefs.dashboard_execution_filter == "macro_surprise"
    assert loaded.dashboard_execution_filter == "macro_surprise"


def test_user_state_store_clears_dashboard_execution_filter(tmp_path: Path) -> None:
    path = tmp_path / "user_state.json"
    store = UserStateStore(path=path)
    store.update_preferences("chat-1", dashboard_execution_filter="stock_pick")

    prefs = store.update_preferences("chat-1", dashboard_execution_filter="all")

    assert prefs.dashboard_execution_filter is None


def test_user_state_store_persists_approval_preferences(tmp_path: Path) -> None:
    path = tmp_path / "user_state.json"
    store = UserStateStore(path=path)

    prefs = store.update_preferences("chat-1", approval_mode="review", max_position_size_pct=2.5)
    loaded = UserStateStore(path=path).get("chat-1")

    assert prefs.approval_mode == "review"
    assert prefs.max_position_size_pct == 2.5
    assert loaded.approval_mode == "review"
    assert loaded.max_position_size_pct == 2.5
