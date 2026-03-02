from __future__ import annotations

from types import SimpleNamespace

from config import Settings, TradingSettings
from modules.mt5_bridge import MT5Bridge


class FakeMT5:
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    POSITION_TYPE_BUY = 0
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    TRADE_ACTION_DEAL = 1

    def __init__(self):
        self.last_request: dict | None = None
        self.selected_symbols: list[str] = []
        self._symbol_info = {
            "BTCUSDm": SimpleNamespace(
                name="BTCUSDm",
                visible=True,
                path="Cryptos\\BTCUSDm",
                currency_base="BTC",
                currency_profit="USD",
                trade_mode=4,
                digits=2,
                point=0.01,
                trade_tick_size=0.01,
                filling_mode=2,
                trade_exemode=self.SYMBOL_TRADE_EXECUTION_MARKET,
            )
        }

    def terminal_info(self):
        return object()

    def initialize(self, **kwargs):
        return True

    def login(self, **kwargs):
        return True

    def shutdown(self):
        return None

    def last_error(self):
        return (0, "ok")

    def symbols_get(self):
        return [SimpleNamespace(name="BTCUSDm", visible=True, trade_mode=4)]

    def symbol_info(self, symbol: str):
        return self._symbol_info.get(symbol)

    def symbol_select(self, symbol: str, enable: bool):
        self.selected_symbols.append(symbol)
        return enable and symbol in self._symbol_info

    def symbol_info_tick(self, symbol: str):
        if symbol not in self._symbol_info:
            return None
        return SimpleNamespace(symbol=symbol, bid=95000.0, ask=95010.0)

    def positions_get(self, symbol: str | None = None):
        if symbol not in (None, "BTCUSDm"):
            return []
        return [
            SimpleNamespace(
                symbol="BTCUSDm",
                ticket=101,
                volume=0.05,
                type=self.POSITION_TYPE_BUY,
                price_open=94500.0,
                price_current=95000.0,
                sl=93800.0,
                tp=96000.0,
                time=1_700_000_000,
            )
        ]

    def order_send(self, request: dict):
        self.last_request = dict(request)
        return SimpleNamespace(retcode=10009, order=9001, price=request.get("price", 0.0))


def test_bridge_resolves_broker_symbol_suffix_and_normalizes_payloads():
    settings = Settings(
        _env_file=None,
        trading=TradingSettings(
            symbols=["BTCUSD"],
            auto_discover_broker_symbols=True,
        ),
    )
    bridge = MT5Bridge(settings)
    bridge._mt5 = FakeMT5()

    info = bridge.prepare_symbol("BTCUSD")
    positions = bridge.positions_get(symbol="BTCUSD")
    tick = bridge.symbol_tick("BTCUSD")
    response = bridge.order_send({"symbol": "BTCUSD", "price": 95010.0, "volume": 0.01})

    assert bridge.broker_symbol("BTCUSD") == "BTCUSDm"
    assert info["symbol"] == "BTCUSD"
    assert info["broker_symbol"] == "BTCUSDm"
    assert positions[0]["symbol"] == "BTCUSD"
    assert positions[0]["broker_symbol"] == "BTCUSDm"
    assert tick["symbol"] == "BTCUSD"
    assert tick["broker_symbol"] == "BTCUSDm"
    assert bridge.module.last_request["symbol"] == "BTCUSDm"
    assert response["symbol"] == "BTCUSD"
    assert response["broker_symbol"] == "BTCUSDm"


def test_bridge_prefers_configured_symbol_alias_when_present():
    settings = Settings(
        _env_file=None,
        trading=TradingSettings(
            symbols=["BTCUSD"],
            symbol_aliases={"BTCUSD": "BTCUSD.micro"},
            auto_discover_broker_symbols=False,
        ),
    )
    bridge = MT5Bridge(settings)
    bridge._mt5 = FakeMT5()

    assert bridge.broker_symbol("BTCUSD") == "BTCUSD.micro"


def test_bridge_maps_market_execution_filling_flags_to_order_modes():
    settings = Settings(
        _env_file=None,
        trading=TradingSettings(
            symbols=["BTCUSD"],
            auto_discover_broker_symbols=True,
        ),
    )
    bridge = MT5Bridge(settings)
    bridge._mt5 = FakeMT5()

    modes = bridge.filling_modes(symbol="BTCUSD")
    response = bridge.close_position(
        position={
            "symbol": "BTCUSD",
            "broker_symbol": "BTCUSDm",
            "ticket": 101,
            "volume": 0.05,
            "type": bridge.module.POSITION_TYPE_BUY,
        },
        deviation=20,
        comment="test-close",
    )

    assert modes == [bridge.module.ORDER_FILLING_IOC]
    assert bridge.module.last_request["type_filling"] == bridge.module.ORDER_FILLING_IOC
    assert response["broker_symbol"] == "BTCUSDm"
