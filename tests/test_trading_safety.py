from __future__ import annotations

from invest_advisor_bot.trading_safety import TradingSafetyPolicy, TradingSafetyStore


def test_trading_safety_blocks_kill_switch_and_records_audit(tmp_path) -> None:
    store = TradingSafetyStore(
        path=tmp_path / "safety.json",
        policy=TradingSafetyPolicy(kill_switch_enabled=True),
    )

    decision = store.evaluate_order(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="market",
        limit_price=None,
        estimated_price=100.0,
        account_equity=10_000.0,
        cash=5_000.0,
        source="test",
        actor="admin",
    )

    assert decision.allowed is False
    assert decision.reason == "kill_switch_enabled"
    status = store.status()
    assert status["kill_switch_enabled"] is True
    assert status["last_audit"]["event"] == "order_blocked"


def test_trading_safety_requires_approval_and_tracks_pending_order(tmp_path) -> None:
    store = TradingSafetyStore(
        path=tmp_path / "safety.json",
        policy=TradingSafetyPolicy(manual_approval_required=True),
    )

    decision = store.evaluate_order(
        symbol="SPY",
        side="buy",
        qty=2,
        order_type="limit",
        limit_price=500.0,
        estimated_price=500.0,
        account_equity=20_000.0,
        cash=10_000.0,
        source="paper_buy",
        actor="admin",
    )

    assert decision.allowed is False
    assert decision.approval_required is True
    assert decision.order_id
    pending = store.get_pending_order(decision.order_id)
    assert pending is not None
    assert pending["symbol"] == "SPY"
    assert store.status()["pending_order_count"] == 1

    approved = store.evaluate_order(
        symbol="SPY",
        side="buy",
        qty=2,
        order_type="limit",
        limit_price=500.0,
        estimated_price=500.0,
        account_equity=20_000.0,
        cash=10_000.0,
        source="paper_buy",
        actor="admin",
        approved=True,
        existing_order_id=decision.order_id,
    )

    assert approved.allowed is True
    assert approved.order_id == decision.order_id


def test_trading_safety_blocks_sizing_and_loss_limits(tmp_path) -> None:
    store = TradingSafetyStore(
        path=tmp_path / "safety.json",
        policy=TradingSafetyPolicy(max_order_notional_usd=1_000.0, daily_loss_limit_pct=0.05),
    )

    too_large = store.evaluate_order(
        symbol="QQQ",
        side="buy",
        qty=3,
        order_type="limit",
        limit_price=500.0,
        estimated_price=500.0,
        account_equity=10_000.0,
        cash=9_000.0,
        source="test",
        actor="admin",
    )
    assert too_large.reason == "max_order_notional_exceeded"

    # Same UTC day, equity is now down more than 5% from the stored day start.
    daily_loss = store.evaluate_order(
        symbol="QQQ",
        side="buy",
        qty=1,
        order_type="limit",
        limit_price=100.0,
        estimated_price=100.0,
        account_equity=9_400.0,
        cash=9_000.0,
        source="test",
        actor="admin",
    )
    assert daily_loss.reason == "daily_loss_limit_triggered"


def test_trading_safety_persists_kill_switch(tmp_path) -> None:
    path = tmp_path / "safety.json"
    store = TradingSafetyStore(path=path)
    store.set_kill_switch(True, reason="test", actor="admin")

    reloaded = TradingSafetyStore(path=path)

    assert reloaded.status()["kill_switch_enabled"] is True
