from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class TradeSignal:
    symbol: str
    timeframe: str
    side: str
    probability: float
    long_probability: float
    short_probability: float
    score: float
    reason: str
    generated_at: datetime
    chronos_direction: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    risk_fraction: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionOutcome:
    status: str
    message: str
    ticket: int | None = None
    retcode: int | None = None
    requested_volume: float | None = None
    executed_price: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HeartbeatState:
    ok: bool
    message: str
    latency_ms: int
    timestamp: datetime
    account: dict[str, Any] | None = None
