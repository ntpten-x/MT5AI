from __future__ import annotations

from config import Settings
from modules.risk_control import RiskManager


class _Database:
    def __init__(self, baseline_equity: float, high_watermark: float, consecutive_losses: int):
        self.baseline_equity = baseline_equity
        self.high_watermark = high_watermark
        self.consecutive_losses = consecutive_losses
        self.snapshots: list[dict] = []

    def record_equity_snapshot(self, account, timestamp, symbol, provider):
        self.snapshots.append(
            {
                "account": dict(account),
                "symbol": symbol,
                "provider": provider,
            }
        )

    def get_daily_equity_baseline(self, _date):
        return {"equity": self.baseline_equity}

    def get_daily_high_watermark_equity(self, _date):
        return self.high_watermark

    def count_consecutive_losses(self, limit=20):
        return self.consecutive_losses


def test_risk_manager_trips_and_latches_kill_switch_on_daily_drawdown():
    settings = Settings(
        _env_file=None,
        risk={
            "daily_drawdown_limit": 0.02,
            "daily_equity_max_drawdown_pct": 0.05,
            "max_consecutive_losses": 5,
        },
    )
    database = _Database(baseline_equity=1000.0, high_watermark=1050.0, consecutive_losses=0)
    manager = RiskManager(settings, database)

    halted, payload = manager.evaluate_kill_switch({"equity": 970.0})

    assert halted is True
    assert payload is not None
    assert payload["event_code"] == "kill_switch_daily_drawdown"
    assert payload["newly_tripped"] is True
    assert manager.kill_switch_active() is True

    halted_again, payload_again = manager.evaluate_kill_switch({"equity": 999.0})
    assert halted_again is True
    assert payload_again is not None
    assert payload_again["newly_tripped"] is False


def test_risk_manager_reset_clears_kill_switch():
    settings = Settings(_env_file=None)
    database = _Database(baseline_equity=1000.0, high_watermark=1000.0, consecutive_losses=0)
    manager = RiskManager(settings, database)
    manager._trip_kill_switch(
        event_code="kill_switch_daily_drawdown",
        reason="test",
        metric_name="daily_drawdown",
        metric_value=0.03,
        threshold=0.02,
    )

    manager.reset_kill_switch()

    assert manager.kill_switch_active() is False
    assert manager.kill_switch_status() is None


def test_risk_manager_loads_persisted_kill_switch(tmp_path):
    settings = Settings(
        _env_file=None,
        bot={"kill_switch_path": str(tmp_path / "kill_switch_state.json")},
    )
    database = _Database(baseline_equity=1000.0, high_watermark=1000.0, consecutive_losses=0)
    manager = RiskManager(settings, database)
    manager._trip_kill_switch(
        event_code="kill_switch_daily_drawdown",
        reason="persisted",
        metric_name="daily_drawdown",
        metric_value=0.03,
        threshold=0.02,
    )

    reloaded = RiskManager(settings, database)

    assert reloaded.kill_switch_active() is True
    assert reloaded.kill_switch_status()["reason"] == "persisted"


def test_risk_manager_returns_zero_when_minimum_volume_exceeds_risk_budget():
    settings = Settings(
        _env_file=None,
        trading={"symbols": ["EURUSD"]},
        risk={"risk_per_trade": 0.005},
    )
    database = _Database(baseline_equity=1000.0, high_watermark=1000.0, consecutive_losses=0)
    manager = RiskManager(settings, database)

    volume = manager.calculate_volume(
        "EURUSD",
        {
            "trade_tick_size": 0.00001,
            "trade_tick_value": 1.0,
            "trade_contract_size": 100000.0,
            "volume_step": 0.01,
            "volume_min": 0.01,
            "volume_max": 100.0,
        },
        {"equity": 10.0},
        entry_price=1.10000,
        stop_price=1.09900,
    )

    assert volume == 0.0
