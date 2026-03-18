from __future__ import annotations

import json
import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Iterator

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_bars (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts_utc INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    tick_volume INTEGER NOT NULL,
    spread INTEGER NOT NULL,
    real_volume INTEGER NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts_utc)
);

CREATE TABLE IF NOT EXISTS trades (
    ticket INTEGER PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'MT5',
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    volume REAL NOT NULL,
    requested_price REAL,
    executed_price REAL,
    sl REAL,
    tp REAL,
    status TEXT NOT NULL,
    strategy TEXT,
    comment TEXT,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    profit REAL,
    request_payload TEXT,
    response_payload TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id SERIAL PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT 'MT5',
    symbol TEXT NOT NULL DEFAULT 'ACCOUNT',
    ts_utc TEXT NOT NULL,
    balance REAL NOT NULL,
    equity REAL NOT NULL,
    profit REAL NOT NULL,
    margin_free REAL NOT NULL,
    margin_level REAL
);

CREATE TABLE IF NOT EXISTS heartbeat (
    id SERIAL PRIMARY KEY,
    ts_utc TEXT NOT NULL,
    ok INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    model_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    artifact_path TEXT,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_events (
    id SERIAL PRIMARY KEY,
    ts_utc TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_market_bars_symbol_timeframe_ts
ON market_bars (symbol, timeframe, ts_utc DESC);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_ts
ON equity_snapshots (ts_utc DESC);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_opened_at
ON trades (symbol, opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_bot_events_ts
ON bot_events (ts_utc DESC);
"""

POST_MIGRATION_SQL = """
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_provider_symbol_ts
ON equity_snapshots (provider, symbol, ts_utc DESC);

CREATE INDEX IF NOT EXISTS idx_trades_provider_symbol_opened_at
ON trades (provider, symbol, opened_at DESC);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat()


class Database:
    def __init__(self, uri: str):
        self.uri = uri

    @contextmanager
    def connect(self) -> Iterator[Any]:
        connection = psycopg2.connect(self.uri)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                self._ensure_column(connection, "trades", "provider", "TEXT NOT NULL DEFAULT 'MT5'")
                self._ensure_column(connection, "equity_snapshots", "provider", "TEXT NOT NULL DEFAULT 'MT5'")
                self._ensure_column(connection, "equity_snapshots", "symbol", "TEXT NOT NULL DEFAULT 'ACCOUNT'")
                cur.execute(POST_MIGRATION_SQL)

    def _ensure_column(
        self,
        connection: Any,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        with connection.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table_name.lower(),),
            )
            columns = {str(row["column_name"]).lower() for row in cursor.fetchall()}
            if column_name.lower() in columns:
                return
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def upsert_bars(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0

        records = []
        normalized = frame.sort_values("time")
        for row in normalized.itertuples(index=False):
            timestamp = getattr(row, "time")
            if hasattr(timestamp, "timestamp"):
                ts_utc = int(timestamp.timestamp())
            else:
                ts_utc = int(getattr(row, "ts_utc"))

            records.append(
                (
                    symbol,
                    timeframe,
                    ts_utc,
                    float(getattr(row, "open")),
                    float(getattr(row, "high")),
                    float(getattr(row, "low")),
                    float(getattr(row, "close")),
                    int(getattr(row, "tick_volume", 0)),
                    int(getattr(row, "spread", 0)),
                    int(getattr(row, "real_volume", 0)),
                )
            )

        sql = """
        INSERT INTO market_bars (
            symbol, timeframe, ts_utc, open, high, low, close, tick_volume, spread, real_volume
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(symbol, timeframe, ts_utc) DO UPDATE SET
            open=EXCLUDED.open,
            high=EXCLUDED.high,
            low=EXCLUDED.low,
            close=EXCLUDED.close,
            tick_volume=EXCLUDED.tick_volume,
            spread=EXCLUDED.spread,
            real_volume=EXCLUDED.real_volume
        """
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.executemany(sql, records)
        return len(records)

    def load_bars(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        if limit:
            sql = """
            SELECT *
            FROM (
                SELECT symbol, timeframe, ts_utc, open, high, low, close, tick_volume, spread, real_volume
                FROM market_bars
                WHERE symbol = %s AND timeframe = %s
                ORDER BY ts_utc DESC
                LIMIT %s
            ) AS sub_query
            ORDER BY ts_utc ASC
            """
            params: tuple[Any, ...] = (symbol, timeframe, limit)
        else:
            sql = """
            SELECT symbol, timeframe, ts_utc, open, high, low, close, tick_volume, spread, real_volume
            FROM market_bars
            WHERE symbol = %s AND timeframe = %s
            ORDER BY ts_utc ASC
            """
            params = (symbol, timeframe)

        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                columns = [col[0] for col in cur.description]
                frame = pd.DataFrame(rows, columns=columns)

        if frame.empty:
            return frame

        frame["time"] = pd.to_datetime(frame["ts_utc"], unit="s", utc=True)
        return frame

    def record_trade(
        self,
        ticket: int,
        provider: str,
        symbol: str,
        side: str,
        volume: float,
        requested_price: float | None,
        executed_price: float | None,
        sl: float | None,
        tp: float | None,
        status: str,
        strategy: str,
        comment: str,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
        profit: float | None = None,
        closed_at: datetime | None = None,
    ) -> None:
        sql = """
        INSERT INTO trades (
            ticket, provider, symbol, side, volume, requested_price, executed_price, sl, tp, status,
            strategy, comment, opened_at, closed_at, profit, request_payload, response_payload
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(ticket) DO UPDATE SET
            provider=EXCLUDED.provider,
            executed_price=EXCLUDED.executed_price,
            sl=EXCLUDED.sl,
            tp=EXCLUDED.tp,
            status=EXCLUDED.status,
            closed_at=EXCLUDED.closed_at,
            profit=EXCLUDED.profit,
            response_payload=EXCLUDED.response_payload
        """
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        ticket,
                        provider,
                        symbol,
                        side,
                        volume,
                        requested_price,
                        executed_price,
                        sl,
                        tp,
                        status,
                        strategy,
                        comment,
                        utc_iso(),
                        utc_iso(closed_at) if closed_at else None,
                        profit,
                        json.dumps(request_payload or {}, ensure_ascii=True),
                        json.dumps(response_payload or {}, ensure_ascii=True),
                    ),
                )

    def record_equity_snapshot(
        self,
        account: dict[str, Any],
        timestamp: datetime | None = None,
        symbol: str = "ACCOUNT",
        provider: str = "MT5",
    ) -> None:
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO equity_snapshots (
                        provider, symbol, ts_utc, balance, equity, profit, margin_free, margin_level
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        provider,
                        symbol,
                        utc_iso(timestamp),
                        float(account.get("balance", 0.0)),
                        float(account.get("equity", 0.0)),
                        float(account.get("profit", 0.0)),
                        float(account.get("margin_free", 0.0)),
                        account.get("margin_level"),
                    ),
                )

    def get_daily_equity_baseline(
        self,
        day: date,
        symbol: str = "ACCOUNT",
        provider: str = "MT5",
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM equity_snapshots
                    WHERE provider = %s AND symbol = %s AND substr(ts_utc, 1, 10) = %s
                    ORDER BY ts_utc ASC
                    LIMIT 1
                    """,
                    (provider, symbol, day.isoformat()),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def get_latest_equity_snapshot(
        self,
        day: date,
        symbol: str = "ACCOUNT",
        provider: str = "MT5",
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM equity_snapshots
                    WHERE provider = %s AND symbol = %s AND substr(ts_utc, 1, 10) = %s
                    ORDER BY ts_utc DESC
                    LIMIT 1
                    """,
                    (provider, symbol, day.isoformat()),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def record_heartbeat(self, ok: bool, latency_ms: int, message: str) -> None:
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(
                    "INSERT INTO heartbeat (ts_utc, ok, latency_ms, message) VALUES (%s, %s, %s, %s)",
                    (utc_iso(), int(ok), latency_ms, message),
                )

    def record_model_run(
        self,
        model_type: str,
        symbol: str,
        timeframe: str,
        artifact_path: str | None,
        metrics: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_runs (model_type, symbol, timeframe, trained_at, artifact_path, metrics_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model_type,
                        symbol,
                        timeframe,
                        utc_iso(),
                        artifact_path,
                        json.dumps(metrics, ensure_ascii=True),
                    ),
                )

    def record_event(
        self,
        level: str,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO bot_events (ts_utc, level, event_type, message, payload_json)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        utc_iso(),
                        level.upper(),
                        event_type,
                        message,
                        json.dumps(payload or {}, ensure_ascii=True),
                    ),
                )

    def get_last_trade_opened_at(self, symbol: str) -> datetime | None:
        with self.connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT opened_at
                    FROM trades
                    WHERE symbol = %s
                    ORDER BY opened_at DESC
                    LIMIT 1
                    """,
                    (symbol,),
                )
                row = cur.fetchone()
        if row is None or not row["opened_at"]:
            return None
        return datetime.fromisoformat(row["opened_at"])

    def get_daily_high_watermark_equity(
        self,
        day: date,
        symbol: str = "ACCOUNT",
        provider: str = "MT5",
    ) -> float | None:
        with self.connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT MAX(equity) as max_equity
                    FROM equity_snapshots
                    WHERE provider = %s AND symbol = %s AND substr(ts_utc, 1, 10) = %s
                    """,
                    (provider, symbol, day.isoformat()),
                )
                row = cur.fetchone()
        if row is None or row["max_equity"] is None:
            return None
        return float(row["max_equity"])

    def count_consecutive_losses(self, limit: int = 10, provider: str = "MT5") -> int:
        with self.connect() as connection:
            with connection.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT profit, status
                    FROM trades
                    WHERE provider = %s AND status IN ('closed', 'filled')
                    ORDER BY opened_at DESC
                    LIMIT %s
                    """,
                    (provider, limit),
                )
                rows = cur.fetchall()
        
        consecutive_losses = 0
        for row in rows:
            profit = row["profit"]
            if profit is None:
                break
            if float(profit) < 0:
                consecutive_losses += 1
            else:
                break
        return consecutive_losses
