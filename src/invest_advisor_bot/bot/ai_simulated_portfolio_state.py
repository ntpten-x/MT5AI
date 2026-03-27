from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


@dataclass(slots=True, frozen=True)
class AISimulatedPortfolioHolding:
    ticker: str
    quantity: float
    avg_cost: float | None = None
    label: str | None = None
    asset_type: str = "asset"
    last_reason: str | None = None
    opened_at: datetime | None = None

    @property
    def normalized_ticker(self) -> str:
        return self.ticker.strip().upper()


@dataclass(slots=True, frozen=True)
class AISimulatedPortfolioTrade:
    trade_id: str
    action: str
    ticker: str
    quantity: float
    price: float
    notional: float
    occurred_at: datetime
    label: str | None = None
    asset_type: str = "asset"
    rationale: str | None = None
    confidence_score: float | None = None
    coverage_score: float | None = None
    detail: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class AISimulatedPortfolioState:
    portfolio_key: str
    starting_cash: float
    cash: float
    realized_pnl: float
    created_at: datetime
    updated_at: datetime
    last_rebalanced_at: datetime | None
    last_action_summary: str | None
    holdings: tuple[AISimulatedPortfolioHolding, ...]
    metadata: dict[str, Any]


class AISimulatedPortfolioStateStore:
    """Persist the bot-owned simulated portfolio state and trade history."""

    def __init__(self, *, path: Path, database_url: str = "") -> None:
        self.path = path
        self._lock = RLock()
        self._state: dict[str, Any] = {"portfolios": {}, "trades": {}}
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()
        else:
            self._load()

    def ensure_portfolio(
        self,
        portfolio_key: str,
        *,
        starting_cash: float,
        base_currency: str = "USD",
        metadata: Mapping[str, Any] | None = None,
    ) -> AISimulatedPortfolioState:
        existing = self.get_portfolio(portfolio_key)
        if existing is not None:
            return existing
        now = datetime.now(timezone.utc)
        payload = {
            "starting_cash": float(starting_cash),
            "cash": float(starting_cash),
            "realized_pnl": 0.0,
            "holdings": [],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_rebalanced_at": None,
            "last_action_summary": None,
            "metadata": {
                "base_currency": base_currency.strip().upper() or "USD",
                **dict(metadata or {}),
            },
        }
        self._write_portfolio_payload(portfolio_key, payload)
        return self._deserialize_state(portfolio_key, payload)

    def get_portfolio(self, portfolio_key: str) -> AISimulatedPortfolioState | None:
        normalized_key = portfolio_key.strip()
        if not normalized_key:
            return None
        if self._db is not None:
            row = self._db.fetch_one(
                """
                SELECT starting_cash, cash, realized_pnl, holdings, metadata, created_at, updated_at, last_rebalanced_at
                FROM bot_ai_simulated_portfolio_state
                WHERE portfolio_key = %s
                """,
                (normalized_key,),
            )
            if row is None:
                return None
            starting_cash, cash, realized_pnl, holdings, metadata, created_at, updated_at, last_rebalanced_at = row
            return self._deserialize_state(
                normalized_key,
                {
                    "starting_cash": starting_cash,
                    "cash": cash,
                    "realized_pnl": realized_pnl,
                    "holdings": holdings,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "last_rebalanced_at": last_rebalanced_at,
                    "last_action_summary": (metadata or {}).get("last_action_summary") if isinstance(metadata, dict) else None,
                    "metadata": metadata or {},
                },
            )
        with self._lock:
            payload = dict((self._state.get("portfolios") or {}).get(normalized_key, {}))
        if not payload:
            return None
        return self._deserialize_state(normalized_key, payload)

    def save_portfolio(
        self,
        portfolio_key: str,
        *,
        starting_cash: float,
        cash: float,
        realized_pnl: float,
        holdings: list[Mapping[str, Any]],
        last_rebalanced_at: datetime | None,
        last_action_summary: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AISimulatedPortfolioState:
        existing = self.get_portfolio(portfolio_key)
        created_at = existing.created_at if existing is not None else datetime.now(timezone.utc)
        payload = {
            "starting_cash": float(starting_cash),
            "cash": float(cash),
            "realized_pnl": float(realized_pnl),
            "holdings": [dict(item) for item in holdings],
            "created_at": created_at.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_rebalanced_at": last_rebalanced_at.isoformat() if last_rebalanced_at is not None else None,
            "last_action_summary": last_action_summary.strip() if last_action_summary and last_action_summary.strip() else None,
            "metadata": {
                **dict(existing.metadata if existing is not None else {}),
                **dict(metadata or {}),
            },
        }
        if payload["last_action_summary"] is not None:
            payload["metadata"]["last_action_summary"] = payload["last_action_summary"]
        self._write_portfolio_payload(portfolio_key, payload)
        return self._deserialize_state(portfolio_key, payload)

    def reset_portfolio(
        self,
        portfolio_key: str,
        *,
        starting_cash: float,
        base_currency: str = "USD",
    ) -> AISimulatedPortfolioState:
        normalized_key = portfolio_key.strip()
        now = datetime.now(timezone.utc)
        payload = {
            "starting_cash": float(starting_cash),
            "cash": float(starting_cash),
            "realized_pnl": 0.0,
            "holdings": [],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_rebalanced_at": None,
            "last_action_summary": "portfolio reset",
            "metadata": {
                "base_currency": base_currency.strip().upper() or "USD",
                "last_action_summary": "portfolio reset",
            },
        }
        self._write_portfolio_payload(normalized_key, payload)
        if self._db is not None:
            self._db.execute("DELETE FROM bot_ai_simulated_portfolio_trade WHERE portfolio_key = %s", (normalized_key,))
        else:
            with self._lock:
                trades = dict(self._state.get("trades") or {})
                trades[normalized_key] = []
                self._state["trades"] = trades
                self._persist()
        return self._deserialize_state(normalized_key, payload)

    def append_trades(self, portfolio_key: str, trades: list[Mapping[str, Any]]) -> tuple[AISimulatedPortfolioTrade, ...]:
        normalized_key = portfolio_key.strip()
        if not normalized_key or not trades:
            return self.list_trades(normalized_key)
        serialized = [self._normalize_trade_payload(item) for item in trades]
        if self._db is not None:
            self._db.executemany(
                """
                INSERT INTO bot_ai_simulated_portfolio_trade (
                    portfolio_key, trade_id, action, ticker, quantity, price, notional, label,
                    asset_type, rationale, confidence_score, coverage_score, occurred_at, detail
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (trade_id) DO NOTHING
                """,
                [
                    (
                        normalized_key,
                        item["trade_id"],
                        item["action"],
                        item["ticker"],
                        item["quantity"],
                        item["price"],
                        item["notional"],
                        item.get("label"),
                        item.get("asset_type"),
                        item.get("rationale"),
                        item.get("confidence_score"),
                        item.get("coverage_score"),
                        self._parse_datetime(item.get("occurred_at")) or datetime.now(timezone.utc),
                        json.dumps(item.get("detail") or {}),
                    )
                    for item in serialized
                ],
            )
            return self.list_trades(normalized_key)
        with self._lock:
            trade_bucket = list((self._state.get("trades") or {}).get(normalized_key, []))
            trade_bucket.extend(serialized)
            self._state.setdefault("trades", {})[normalized_key] = trade_bucket[-500:]
            self._persist()
        return self.list_trades(normalized_key)

    def list_trades(self, portfolio_key: str, *, limit: int = 20) -> tuple[AISimulatedPortfolioTrade, ...]:
        normalized_key = portfolio_key.strip()
        if not normalized_key:
            return ()
        if self._db is not None:
            rows = self._db.fetch_all(
                """
                SELECT trade_id, action, ticker, quantity, price, notional, label, asset_type, rationale,
                       confidence_score, coverage_score, occurred_at, detail
                FROM bot_ai_simulated_portfolio_trade
                WHERE portfolio_key = %s
                ORDER BY occurred_at DESC, id DESC
                LIMIT %s
                """,
                (normalized_key, max(1, int(limit))),
            )
            return tuple(
                AISimulatedPortfolioTrade(
                    trade_id=str(trade_id),
                    action=str(action),
                    ticker=str(ticker).upper(),
                    quantity=float(quantity),
                    price=float(price),
                    notional=float(notional),
                    label=str(label).strip() if label is not None else None,
                    asset_type=str(asset_type).strip() if asset_type is not None else "asset",
                    rationale=str(rationale).strip() if rationale is not None else None,
                    confidence_score=self._coerce_float(confidence_score),
                    coverage_score=self._coerce_float(coverage_score),
                    occurred_at=self._parse_datetime(occurred_at) or datetime.now(timezone.utc),
                    detail=dict(detail) if isinstance(detail, dict) else {},
                )
                for trade_id, action, ticker, quantity, price, notional, label, asset_type, rationale, confidence_score, coverage_score, occurred_at, detail in rows
            )
        with self._lock:
            raw_trades = list((self._state.get("trades") or {}).get(normalized_key, []))
        deserialized = [self._deserialize_trade(item) for item in raw_trades]
        deserialized.sort(key=lambda item: item.occurred_at, reverse=True)
        return tuple(deserialized[: max(1, int(limit))])

    def status(self) -> dict[str, Any]:
        if self._db is not None:
            portfolio_count = self._db.fetch_one("SELECT COUNT(*) FROM bot_ai_simulated_portfolio_state")
            trade_count = self._db.fetch_one("SELECT COUNT(*) FROM bot_ai_simulated_portfolio_trade")
            return {
                "available": True,
                "configured": True,
                "backend": "postgres",
                "portfolio_count": int(portfolio_count[0]) if portfolio_count else 0,
                "trade_count": int(trade_count[0]) if trade_count else 0,
            }
        with self._lock:
            portfolios = self._state.get("portfolios") or {}
            trades = self._state.get("trades") or {}
        return {
            "available": True,
            "configured": True,
            "backend": "file",
            "portfolio_count": len(portfolios),
            "trade_count": sum(len(items) for items in trades.values() if isinstance(items, list)),
        }

    def _write_portfolio_payload(self, portfolio_key: str, payload: Mapping[str, Any]) -> None:
        normalized_key = portfolio_key.strip()
        if self._db is not None:
            metadata = dict(payload.get("metadata") or {})
            if payload.get("last_action_summary"):
                metadata["last_action_summary"] = payload.get("last_action_summary")
            self._db.execute(
                """
                INSERT INTO bot_ai_simulated_portfolio_state (
                    portfolio_key, starting_cash, cash, realized_pnl, holdings, metadata, created_at, updated_at, last_rebalanced_at
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
                ON CONFLICT (portfolio_key)
                DO UPDATE SET
                    starting_cash = EXCLUDED.starting_cash,
                    cash = EXCLUDED.cash,
                    realized_pnl = EXCLUDED.realized_pnl,
                    holdings = EXCLUDED.holdings,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at,
                    last_rebalanced_at = EXCLUDED.last_rebalanced_at
                """,
                (
                    normalized_key,
                    float(payload.get("starting_cash") or 0.0),
                    float(payload.get("cash") or 0.0),
                    float(payload.get("realized_pnl") or 0.0),
                    json.dumps(list(payload.get("holdings") or [])),
                    json.dumps(metadata),
                    self._parse_datetime(payload.get("created_at")) or datetime.now(timezone.utc),
                    self._parse_datetime(payload.get("updated_at")) or datetime.now(timezone.utc),
                    self._parse_datetime(payload.get("last_rebalanced_at")),
                ),
            )
            return
        with self._lock:
            portfolios = dict(self._state.get("portfolios") or {})
            portfolios[normalized_key] = dict(payload)
            self._state["portfolios"] = portfolios
            self._persist()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if isinstance(payload, dict):
            self._state = {
                "portfolios": dict(payload.get("portfolios") or {}),
                "trades": dict(payload.get("trades") or {}),
            }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _deserialize_state(self, portfolio_key: str, payload: Mapping[str, Any]) -> AISimulatedPortfolioState:
        holdings_payload = payload.get("holdings") or []
        metadata = dict(payload.get("metadata") or {})
        last_action_summary = str(payload.get("last_action_summary") or metadata.get("last_action_summary") or "").strip() or None
        holdings = tuple(
            AISimulatedPortfolioHolding(
                ticker=str(item.get("ticker") or "").upper(),
                quantity=float(item.get("quantity")),
                avg_cost=self._coerce_float(item.get("avg_cost")),
                label=str(item.get("label")).strip() if item.get("label") is not None else None,
                asset_type=str(item.get("asset_type") or "asset").strip() or "asset",
                last_reason=str(item.get("last_reason")).strip() if item.get("last_reason") is not None else None,
                opened_at=self._parse_datetime(item.get("opened_at")),
            )
            for item in holdings_payload
            if isinstance(item, Mapping) and self._coerce_float(item.get("quantity")) not in {None, 0.0}
        )
        return AISimulatedPortfolioState(
            portfolio_key=portfolio_key,
            starting_cash=float(payload.get("starting_cash") or 0.0),
            cash=float(payload.get("cash") or 0.0),
            realized_pnl=float(payload.get("realized_pnl") or 0.0),
            created_at=self._parse_datetime(payload.get("created_at")) or datetime.now(timezone.utc),
            updated_at=self._parse_datetime(payload.get("updated_at")) or datetime.now(timezone.utc),
            last_rebalanced_at=self._parse_datetime(payload.get("last_rebalanced_at")),
            last_action_summary=last_action_summary,
            holdings=tuple(sorted(holdings, key=lambda item: item.normalized_ticker)),
            metadata=metadata,
        )

    def _deserialize_trade(self, payload: Mapping[str, Any]) -> AISimulatedPortfolioTrade:
        return AISimulatedPortfolioTrade(
            trade_id=str(payload.get("trade_id") or ""),
            action=str(payload.get("action") or ""),
            ticker=str(payload.get("ticker") or "").upper(),
            quantity=float(payload.get("quantity") or 0.0),
            price=float(payload.get("price") or 0.0),
            notional=float(payload.get("notional") or 0.0),
            occurred_at=self._parse_datetime(payload.get("occurred_at")) or datetime.now(timezone.utc),
            label=str(payload.get("label")).strip() if payload.get("label") is not None else None,
            asset_type=str(payload.get("asset_type") or "asset").strip() or "asset",
            rationale=str(payload.get("rationale")).strip() if payload.get("rationale") is not None else None,
            confidence_score=self._coerce_float(payload.get("confidence_score")),
            coverage_score=self._coerce_float(payload.get("coverage_score")),
            detail=dict(payload.get("detail") or {}),
        )

    @staticmethod
    def _normalize_trade_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "trade_id": str(payload.get("trade_id") or "").strip(),
            "action": str(payload.get("action") or "").strip().lower(),
            "ticker": str(payload.get("ticker") or "").strip().upper(),
            "quantity": float(payload.get("quantity") or 0.0),
            "price": float(payload.get("price") or 0.0),
            "notional": float(payload.get("notional") or 0.0),
            "occurred_at": (
                payload.get("occurred_at").isoformat()
                if isinstance(payload.get("occurred_at"), datetime)
                else str(payload.get("occurred_at") or datetime.now(timezone.utc).isoformat())
            ),
            "label": str(payload.get("label")).strip() if payload.get("label") is not None else None,
            "asset_type": str(payload.get("asset_type") or "asset").strip() or "asset",
            "rationale": str(payload.get("rationale")).strip() if payload.get("rationale") is not None else None,
            "confidence_score": AISimulatedPortfolioStateStore._coerce_float(payload.get("confidence_score")),
            "coverage_score": AISimulatedPortfolioStateStore._coerce_float(payload.get("coverage_score")),
            "detail": dict(payload.get("detail") or {}),
        }

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None
