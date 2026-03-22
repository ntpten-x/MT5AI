from __future__ import annotations

from pathlib import Path

from invest_advisor_bot.bot.portfolio_state import PortfolioStateStore


def test_portfolio_state_store_file_backend_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "portfolio_state.json"
    store = PortfolioStateStore(path=path)

    holdings = store.upsert_holding("chat-1", ticker="VOO", quantity=10, avg_cost=470.0)
    assert len(holdings) == 1
    assert holdings[0].normalized_ticker == "VOO"
    assert holdings[0].avg_cost == 470.0

    holdings = store.upsert_holding("chat-1", ticker="CASH", quantity=20_000.0, note="reserve")
    assert len(holdings) == 2
    assert {item.normalized_ticker for item in holdings} == {"VOO", "CASH"}

    reloaded = PortfolioStateStore(path=path)
    reloaded_holdings = reloaded.list_holdings("chat-1")
    assert len(reloaded_holdings) == 2
    assert any(item.normalized_ticker == "CASH" and item.note == "reserve" for item in reloaded_holdings)

    remaining = reloaded.remove_holding("chat-1", ticker="VOO")
    assert len(remaining) == 1
    assert remaining[0].normalized_ticker == "CASH"
