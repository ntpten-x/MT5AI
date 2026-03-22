from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


@dataclass(slots=True, frozen=True)
class PortfolioHolding:
    ticker: str
    quantity: float
    avg_cost: float | None = None
    note: str | None = None

    @property
    def normalized_ticker(self) -> str:
        return self.ticker.strip().upper()

    @property
    def is_cash(self) -> bool:
        return self.normalized_ticker in {"CASH", "USD"}


class PortfolioStateStore:
    """Persist per-chat holdings for portfolio-aware advice."""

    def __init__(self, *, path: Path, database_url: str = "") -> None:
        self.path = path
        self._lock = RLock()
        self._state: dict[str, dict[str, dict[str, object]]] = {}
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()
        else:
            self._load()

    def list_holdings(self, conversation_key: str) -> tuple[PortfolioHolding, ...]:
        if self._db is not None:
            return self._list_holdings_db(conversation_key)
        with self._lock:
            payload = dict(self._state.get(conversation_key, {}))
        holdings: list[PortfolioHolding] = []
        for ticker, item in payload.items():
            if not isinstance(item, dict):
                continue
            quantity = self._coerce_float(item.get("quantity"))
            if quantity is None or quantity <= 0:
                continue
            holdings.append(
                PortfolioHolding(
                    ticker=str(ticker).upper(),
                    quantity=quantity,
                    avg_cost=self._coerce_float(item.get("avg_cost")),
                    note=str(item.get("note")).strip() if item.get("note") is not None else None,
                )
            )
        holdings.sort(key=lambda item: item.normalized_ticker)
        return tuple(holdings)

    def upsert_holding(
        self,
        conversation_key: str,
        *,
        ticker: str,
        quantity: float,
        avg_cost: float | None = None,
        note: str | None = None,
    ) -> tuple[PortfolioHolding, ...]:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker or quantity <= 0:
            return self.list_holdings(conversation_key)
        if self._db is not None:
            self._db.execute(
                """
                INSERT INTO bot_portfolio_holdings (conversation_key, ticker, quantity, avg_cost, note, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (conversation_key, ticker)
                DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    avg_cost = EXCLUDED.avg_cost,
                    note = EXCLUDED.note,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    conversation_key,
                    normalized_ticker,
                    float(quantity),
                    float(avg_cost) if avg_cost is not None else None,
                    note.strip() if note and note.strip() else None,
                    datetime.now(timezone.utc),
                ),
            )
            return self.list_holdings(conversation_key)
        with self._lock:
            bucket = dict(self._state.get(conversation_key, {}))
            bucket[normalized_ticker] = {
                "quantity": float(quantity),
                "avg_cost": float(avg_cost) if avg_cost is not None else None,
                "note": note.strip() if note and note.strip() else None,
            }
            self._state[conversation_key] = bucket
            self._persist()
        return self.list_holdings(conversation_key)

    def remove_holding(self, conversation_key: str, *, ticker: str) -> tuple[PortfolioHolding, ...]:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return self.list_holdings(conversation_key)
        if self._db is not None:
            self._db.execute(
                "DELETE FROM bot_portfolio_holdings WHERE conversation_key = %s AND ticker = %s",
                (conversation_key, normalized_ticker),
            )
            return self.list_holdings(conversation_key)
        with self._lock:
            bucket = dict(self._state.get(conversation_key, {}))
            bucket.pop(normalized_ticker, None)
            self._state[conversation_key] = bucket
            self._persist()
        return self.list_holdings(conversation_key)

    def _list_holdings_db(self, conversation_key: str) -> tuple[PortfolioHolding, ...]:
        assert self._db is not None
        rows = self._db.fetch_all(
            """
            SELECT ticker, quantity, avg_cost, note
            FROM bot_portfolio_holdings
            WHERE conversation_key = %s
            ORDER BY ticker ASC
            """,
            (conversation_key,),
        )
        holdings = [
            PortfolioHolding(
                ticker=str(ticker).upper(),
                quantity=float(quantity),
                avg_cost=self._coerce_float(avg_cost),
                note=str(note).strip() if note is not None else None,
            )
            for ticker, quantity, avg_cost, note in rows
            if self._coerce_float(quantity) is not None and float(quantity) > 0
        ]
        return tuple(holdings)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if isinstance(payload, dict):
            self._state = {
                str(conversation_key): dict(holdings)
                for conversation_key, holdings in payload.items()
                if isinstance(conversation_key, str) and isinstance(holdings, dict)
            }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
