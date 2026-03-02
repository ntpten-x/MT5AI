from __future__ import annotations

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

PANDAS_FREQUENCIES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
}


def timeframe_minutes(name: str) -> int:
    normalized = name.upper()
    if normalized not in TIMEFRAME_MINUTES:
        raise KeyError(f"Unsupported timeframe: {name}")
    return TIMEFRAME_MINUTES[normalized]


def pandas_frequency(name: str) -> str:
    normalized = name.upper()
    if normalized not in PANDAS_FREQUENCIES:
        raise KeyError(f"Unsupported timeframe: {name}")
    return PANDAS_FREQUENCIES[normalized]


def resolve_mt5_timeframe(mt5, name: str):
    attribute = f"TIMEFRAME_{name.upper()}"
    if not hasattr(mt5, attribute):
        raise KeyError(f"MetaTrader5 does not expose {attribute}")
    return getattr(mt5, attribute)


def scheduler_trigger_args(name: str) -> dict[str, int | str]:
    normalized = name.upper()
    if normalized == "D1":
        return {"hour": 0, "minute": 0, "second": 5}

    minutes = timeframe_minutes(normalized)
    if minutes < 60:
        return {"minute": f"*/{minutes}", "second": 3}

    if minutes % 60 == 0:
        return {"hour": f"*/{minutes // 60}", "minute": 0, "second": 5}

    raise KeyError(f"Unsupported scheduler interval: {name}")
