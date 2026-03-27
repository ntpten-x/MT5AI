from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class AnalyticsWarehouse:
    """Optional ClickHouse warehouse sink with local JSONL fallbacks."""

    _TABLES = ("recommendation_events", "evaluation_events", "market_events", "runtime_snapshots")
    _ROLLUP_SPECS = {
        "recommendation_topic_rollups": "recommendation_events",
        "evaluation_outcome_rollups": "evaluation_events",
        "market_topic_rollups": "market_events",
        "runtime_topic_rollups": "runtime_snapshots",
    }

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        clickhouse_url: str = "",
        database: str = "default",
        username: str = "",
        password: str = "",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.clickhouse_url = clickhouse_url.strip()
        self.database = database.strip() or "default"
        self.username = username.strip()
        self.password = password.strip()
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._last_write_at: str | None = None
        self._counts = {table: 0 for table in self._TABLES}
        self._jsonl_dir = self.root_dir / "jsonl"
        if self.enabled:
            self._jsonl_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "jsonl"
            self._ensure_clickhouse_schema()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": bool(self.clickhouse_url),
                "backend": self._backend,
                "database": self.database,
                "counts": dict(self._counts),
                "table_counts": dict(self._counts),
                "rollup_tables": list(self._ROLLUP_SPECS),
                "materialized_views_enabled": bool(self.clickhouse_url),
                "last_write_at": self._last_write_at,
                "warning": self._warning,
            }

    def record_recommendation_event(self, **kwargs: Any) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(kwargs.get("artifact_key") or "").strip(),
            "topic": str(kwargs.get("model") or "").strip() or None,
            "payload_json": json.dumps(dict(kwargs), ensure_ascii=False),
        }
        self._write_event(table="recommendation_events", event=event)

    def record_evaluation_event(self, **kwargs: Any) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(kwargs.get("artifact_key") or "").strip(),
            "topic": str(kwargs.get("outcome_label") or "").strip() or None,
            "payload_json": json.dumps(dict(kwargs), ensure_ascii=False),
        }
        self._write_event(table="evaluation_events", event=event)

    def record_market_event(self, **kwargs: Any) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(kwargs.get("artifact_key") or "").strip(),
            "topic": str(kwargs.get("topic") or "").strip() or None,
            "payload_json": json.dumps(dict(kwargs), ensure_ascii=False),
        }
        self._write_event(table="market_events", event=event)

    def record_runtime_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": "runtime",
            "topic": "runtime",
            "payload_json": json.dumps(dict(snapshot), ensure_ascii=False),
        }
        self._write_event(table="runtime_snapshots", event=event)

    def recent_events(self, *, table: str, limit: int = 20) -> list[dict[str, Any]]:
        normalized = str(table or "").strip()
        if normalized not in self._TABLES:
            return []
        clickhouse_rows = self._query_clickhouse(table=normalized, limit=limit)
        if clickhouse_rows:
            return clickhouse_rows
        path = self._jsonl_dir / f"{normalized}.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, Mapping):
                        continue
                    normalized = dict(payload)
                    payload_json = normalized.get("payload_json")
                    if isinstance(payload_json, str):
                        try:
                            decoded_payload = json.loads(payload_json)
                        except Exception:
                            decoded_payload = {}
                        normalized["payload"] = decoded_payload if isinstance(decoded_payload, Mapping) else {}
                    rows.append(normalized)
        except OSError as exc:
            with self._lock:
                self._warning = f"warehouse_recent_failed: {exc}"
            return []
        return rows[-max(1, int(limit)) :]

    def _write_event(self, *, table: str, event: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self._append_jsonl(table=table, event=event)
        self._write_clickhouse(table=table, event=event)

    def _append_jsonl(self, *, table: str, event: Mapping[str, Any]) -> None:
        path = self._jsonl_dir / f"{table}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Analytics warehouse JSONL write failed for {}: {}", table, exc)
            with self._lock:
                self._warning = f"warehouse_jsonl_failed: {exc}"
        else:
            with self._lock:
                if self._backend == "disabled":
                    self._backend = "jsonl"
                self._counts[table] = int(self._counts.get(table, 0)) + 1
                self._last_write_at = self._utc_now().isoformat()

    def _ensure_clickhouse_schema(self) -> None:
        if not self.clickhouse_url:
            return
        try:
            client = self._load_clickhouse()
            for table in self._TABLES:
                client.command(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.database}.{table} (
                        event_at DateTime,
                        artifact_key String,
                        topic Nullable(String),
                        payload_json String
                    ) ENGINE = MergeTree
                    ORDER BY (event_at, artifact_key)
                    """
                )
            for rollup_table, source_table in self._ROLLUP_SPECS.items():
                client.command(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.database}.{rollup_table} (
                        event_date Date,
                        topic String,
                        total_events UInt64
                    ) ENGINE = SummingMergeTree
                    ORDER BY (event_date, topic)
                    """
                )
                client.command(
                    f"""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS {self.database}.{rollup_table}_mv
                    TO {self.database}.{rollup_table}
                    AS
                    SELECT
                        toDate(event_at) AS event_date,
                        ifNull(topic, '-') AS topic,
                        count() AS total_events
                    FROM {self.database}.{source_table}
                    GROUP BY event_date, topic
                    """
                )
        except Exception as exc:
            logger.warning("ClickHouse schema setup failed: {}", exc)
            with self._lock:
                self._warning = f"clickhouse_schema_failed: {exc}"
                self._backend = "jsonl"
        else:
            with self._lock:
                self._backend = "clickhouse"
                self._warning = None

    def _write_clickhouse(self, *, table: str, event: Mapping[str, Any]) -> None:
        if not self.clickhouse_url:
            return
        try:
            client = self._load_clickhouse()
            client.insert(
                f"{self.database}.{table}",
                [[event.get("event_at"), event.get("artifact_key"), event.get("topic"), event.get("payload_json")]],
                column_names=["event_at", "artifact_key", "topic", "payload_json"],
            )
        except Exception as exc:
            logger.warning("ClickHouse insert failed for {}: {}", table, exc)
            with self._lock:
                self._warning = f"clickhouse_insert_failed: {exc}"
        else:
            with self._lock:
                self._backend = "clickhouse"
                self._warning = None
                self._last_write_at = self._utc_now().isoformat()

    def _query_clickhouse(self, *, table: str, limit: int) -> list[dict[str, Any]]:
        if not self.clickhouse_url:
            return []
        try:
            client = self._load_clickhouse()
            result = client.query(
                f"""
                SELECT event_at, artifact_key, topic, payload_json
                FROM {self.database}.{table}
                ORDER BY event_at DESC
                LIMIT {max(1, int(limit))}
                """
            )
        except Exception as exc:
            with self._lock:
                self._warning = f"clickhouse_query_failed: {exc}"
            return []
        rows: list[dict[str, Any]] = []
        for row in result.result_rows:
            try:
                payload = json.loads(row[3])
            except Exception:
                payload = {}
            rows.append(
                {
                    "event_at": row[0].isoformat() if hasattr(row[0], "isoformat") else row[0],
                    "artifact_key": row[1],
                    "topic": row[2],
                    "payload": payload if isinstance(payload, Mapping) else {},
                }
            )
        with self._lock:
            if rows:
                self._backend = "clickhouse"
                self._warning = None
        rows.reverse()
        return rows

    def _load_clickhouse(self) -> Any:
        import clickhouse_connect

        return clickhouse_connect.get_client(
            interface="http",
            dsn=self.clickhouse_url,
            username=self.username or None,
            password=self.password or None,
            database=self.database,
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
