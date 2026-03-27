from __future__ import annotations

import importlib
from threading import RLock
from time import sleep
from typing import Any, Iterable, Sequence


class PostgresStateBackend:
    """Small synchronous Postgres helper for bot state persistence."""

    _shared_lock = RLock()
    _shared_connections: dict[str, Any] = {}
    _shared_connection_locks: dict[str, RLock] = {}

    def __init__(self, *, database_url: str) -> None:
        self.database_url = database_url.strip()
        self._lock = RLock()
        self._schema_ready = False
        self._vector_ready = False
        with self._shared_lock:
            self._connection_lock = self._shared_connection_locks.setdefault(self.database_url, RLock())

    def ensure_schema(self) -> None:
        with self._lock:
            if self._schema_ready:
                return
            psycopg = self._load_driver()
            statements = (
                """
                CREATE TABLE IF NOT EXISTS bot_user_preferences (
                    conversation_key TEXT PRIMARY KEY,
                    watchlist JSONB NOT NULL DEFAULT '[]'::jsonb,
                    preferred_sectors JSONB NOT NULL DEFAULT '[]'::jsonb,
                    stock_alert_threshold DOUBLE PRECISION NOT NULL DEFAULT 1.8,
                    daily_pick_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    dashboard_execution_filter TEXT,
                    approval_mode TEXT NOT NULL DEFAULT 'auto',
                    max_position_size_pct DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """,
                """
                ALTER TABLE bot_user_preferences
                ADD COLUMN IF NOT EXISTS dashboard_execution_filter TEXT
                """,
                """
                ALTER TABLE bot_user_preferences
                ADD COLUMN IF NOT EXISTS approval_mode TEXT NOT NULL DEFAULT 'auto'
                """,
                """
                ALTER TABLE bot_user_preferences
                ADD COLUMN IF NOT EXISTS max_position_size_pct DOUBLE PRECISION
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_alert_state (
                    alert_key TEXT PRIMARY KEY,
                    last_seen TIMESTAMPTZ NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_report_memory (
                    day_key DATE NOT NULL,
                    report_kind TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    summary TEXT NOT NULL,
                    PRIMARY KEY (day_key, report_kind)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_portfolio_holdings (
                    conversation_key TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    avg_cost DOUBLE PRECISION,
                    note TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (conversation_key, ticker)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_sector_rotation_state (
                    id BIGSERIAL PRIMARY KEY,
                    observed_at TIMESTAMPTZ NOT NULL,
                    regime TEXT NOT NULL,
                    sectors JSONB NOT NULL DEFAULT '{}'::jsonb,
                    market_breadth JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_provider_diagnostics_history (
                    id BIGSERIAL PRIMARY KEY,
                    event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    provider TEXT NOT NULL,
                    model TEXT,
                    service TEXT,
                    status TEXT NOT NULL,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_job_diagnostics_history (
                    id BIGSERIAL PRIMARY KEY,
                    event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    job_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms INTEGER,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_sent_reports_history (
                    id BIGSERIAL PRIMARY KEY,
                    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    report_kind TEXT NOT NULL,
                    chat_id TEXT,
                    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
                    model TEXT,
                    summary TEXT,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_user_interaction_history (
                    id BIGSERIAL PRIMARY KEY,
                    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    conversation_key TEXT,
                    interaction_kind TEXT NOT NULL,
                    question TEXT,
                    response_excerpt TEXT,
                    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
                    model TEXT,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_alert_audit (
                    id BIGSERIAL PRIMARY KEY,
                    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    alert_key TEXT NOT NULL,
                    severity TEXT,
                    category TEXT,
                    chat_id TEXT,
                    text_excerpt TEXT,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_stock_pick_scorecard (
                    id BIGSERIAL PRIMARY KEY,
                    recommendation_key TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    due_at TIMESTAMPTZ NOT NULL,
                    source_kind TEXT NOT NULL,
                    chat_id TEXT,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    stance TEXT,
                    confidence_score DOUBLE PRECISION,
                    confidence_label TEXT,
                    composite_score DOUBLE PRECISION,
                    entry_price DOUBLE PRECISION NOT NULL,
                    evaluated_at TIMESTAMPTZ,
                    exit_price DOUBLE PRECISION,
                    return_pct DOUBLE PRECISION,
                    outcome_label TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_thesis_memory (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    conversation_key TEXT,
                    thesis_key TEXT NOT NULL UNIQUE,
                    query_text TEXT,
                    thesis_text TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                    confidence_score DOUBLE PRECISION,
                    embedding_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bot_eval_artifact (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    artifact_key TEXT NOT NULL UNIQUE,
                    artifact_kind TEXT NOT NULL,
                    conversation_key TEXT,
                    model TEXT,
                    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
                    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    detail JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_bot_alert_state_last_seen ON bot_alert_state (last_seen)",
                "CREATE INDEX IF NOT EXISTS idx_bot_report_memory_day ON bot_report_memory (day_key)",
                "CREATE INDEX IF NOT EXISTS idx_bot_portfolio_holdings_conversation_key ON bot_portfolio_holdings (conversation_key)",
                "CREATE INDEX IF NOT EXISTS idx_bot_sector_rotation_regime_observed_at ON bot_sector_rotation_state (regime, observed_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_bot_provider_diag_event_at ON bot_provider_diagnostics_history (event_at)",
                "CREATE INDEX IF NOT EXISTS idx_bot_job_diag_event_at ON bot_job_diagnostics_history (event_at)",
                "CREATE INDEX IF NOT EXISTS idx_bot_sent_reports_sent_at ON bot_sent_reports_history (sent_at)",
                "CREATE INDEX IF NOT EXISTS idx_bot_user_interaction_occurred_at ON bot_user_interaction_history (occurred_at)",
                "CREATE INDEX IF NOT EXISTS idx_bot_alert_audit_sent_at ON bot_alert_audit (sent_at)",
                "CREATE INDEX IF NOT EXISTS idx_bot_stock_pick_scorecard_due_status ON bot_stock_pick_scorecard (status, due_at)",
                "CREATE INDEX IF NOT EXISTS idx_bot_stock_pick_scorecard_created_at ON bot_stock_pick_scorecard (created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_bot_thesis_memory_created_at ON bot_thesis_memory (created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_bot_thesis_memory_conversation_key ON bot_thesis_memory (conversation_key, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_bot_eval_artifact_created_at ON bot_eval_artifact (created_at DESC)",
            )
            def _ensure(cursor: Any) -> None:
                for statement in statements:
                    try:
                        cursor.execute(statement)
                    except Exception as exc:  # pragma: no cover - defensive around pooler DDL races
                        message = str(exc).casefold()
                        if "already exists" in message or "pg_type_typname_nsp_index" in message:
                            continue
                        raise

            self._run_with_retry(_ensure)
            self._ensure_vector_support()
            self._schema_ready = True

    def execute(self, query: str, params: Sequence[Any] | None = None) -> None:
        self._run_with_retry(lambda cursor: cursor.execute(query, params or ()))

    def fetch_one(self, query: str, params: Sequence[Any] | None = None) -> tuple[Any, ...] | None:
        row = self._run_with_retry(lambda cursor: (cursor.execute(query, params or ()), cursor.fetchone())[1])
        if row is None:
            return None
        return tuple(row)

    def fetch_all(self, query: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        rows = self._run_with_retry(lambda cursor: (cursor.execute(query, params or ()), cursor.fetchall())[1])
        return [tuple(row) for row in rows]

    def fetch_dicts(self, query: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        def _operation(cursor: Any) -> list[dict[str, Any]]:
            cursor.execute(query, params or ())
            description = cursor.description or ()
            columns = [getattr(item, "name", item[0]) for item in description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]

        return self._run_with_retry(_operation)

    def executemany(self, query: str, params_seq: Iterable[Sequence[Any]]) -> None:
        batch = list(params_seq)
        self._run_with_retry(lambda cursor: cursor.executemany(query, batch))

    def ping(self) -> bool:
        try:
            row = self.fetch_one("SELECT 1")
        except Exception:
            return False
        return bool(row and row[0] == 1)

    def vector_ready(self) -> bool:
        if not self._schema_ready:
            self.ensure_schema()
        return self._vector_ready

    @classmethod
    def ping_database_url(cls, database_url: str) -> bool:
        if not database_url.strip():
            return False
        return cls(database_url=database_url).ping()

    @staticmethod
    def _load_driver() -> Any:
        try:
            return importlib.import_module("psycopg")
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL is configured but psycopg is not installed. Install project dependencies first."
            ) from exc

    def _run_with_retry(self, operation: Any) -> Any:
        psycopg = self._load_driver()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._connection_lock:
                    connection = self._get_connection(psycopg)
                    with connection.cursor() as cursor:
                        return operation(cursor)
            except Exception as exc:  # pragma: no cover - runtime guard
                last_error = exc
                self._close_connection()
                if attempt < 2:
                    sleep(0.5 * (attempt + 1))
                    continue
                raise
        if last_error is not None:
            raise last_error

    def _ensure_vector_support(self) -> None:
        try:
            self.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.execute("ALTER TABLE bot_thesis_memory ADD COLUMN IF NOT EXISTS embedding vector(16)")
            self.execute("CREATE INDEX IF NOT EXISTS idx_bot_thesis_memory_embedding ON bot_thesis_memory USING ivfflat (embedding vector_cosine_ops)")
            self._vector_ready = True
        except Exception:
            self._vector_ready = False

    def _get_connection(self, psycopg: Any) -> Any:
        with self._shared_lock:
            connection = self._shared_connections.get(self.database_url)
            if connection is None or getattr(connection, "closed", False):
                connection = psycopg.connect(self.database_url, autocommit=True)
                self._shared_connections[self.database_url] = connection
            return connection

    def _close_connection(self) -> None:
        with self._shared_lock:
            connection = self._shared_connections.pop(self.database_url, None)
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            return
