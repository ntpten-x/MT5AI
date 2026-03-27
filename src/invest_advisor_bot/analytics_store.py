from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class AnalyticsStore:
    """Append-only analytical store for recommendation and evaluation events."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = True,
        parquet_export_interval_seconds: int = 300,
        runtime_snapshot_interval_seconds: int = 300,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.parquet_export_interval_seconds = max(60, int(parquet_export_interval_seconds))
        self.runtime_snapshot_interval_seconds = max(60, int(runtime_snapshot_interval_seconds))
        self._lock = RLock()
        self._warning: str | None = None
        self._last_write_at: str | None = None
        self._last_export_at: str | None = None
        self._last_runtime_snapshot_at: datetime | None = None
        self._backend = "disabled"
        self._db_path = self.root_dir / "analytics.duckdb"
        self._jsonl_dir = self.root_dir / "jsonl"
        self._parquet_dir = self.root_dir / "parquet"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_dir.mkdir(parents=True, exist_ok=True)
            self._parquet_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_duckdb()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "backend": self._backend,
                "root_dir": str(self.root_dir),
                "db_path": str(self._db_path),
                "warning": self._warning,
                "last_write_at": self._last_write_at,
                "last_export_at": self._last_export_at,
            }

    def record_recommendation_event(
        self,
        *,
        artifact_key: str,
        conversation_key_hash: str | None,
        question: str | None,
        model: str | None,
        fallback_used: bool,
        response_text: str,
        payload: Mapping[str, Any],
        source_coverage: Mapping[str, Any],
        data_quality: Mapping[str, Any] | None,
    ) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": artifact_key,
            "conversation_key_hash": conversation_key_hash,
            "question": str(question or "").strip(),
            "model": str(model or "").strip() or None,
            "fallback_used": bool(fallback_used),
            "response_text": str(response_text or ""),
            "response_length": len(str(response_text or "")),
            "source_count": len(source_coverage.get("used_sources") or []) if isinstance(source_coverage, Mapping) else 0,
            "source_health_score": self._coerce_float((payload.get("source_health") or {}).get("score") if isinstance(payload.get("source_health"), Mapping) else None),
            "data_quality_status": str((data_quality or {}).get("status") or "").strip() or None,
            "data_quality_score": self._coerce_float((data_quality or {}).get("score")),
            "payload": dict(payload),
            "source_coverage": dict(source_coverage),
            "data_quality": dict(data_quality or {}),
        }
        self._append_event(bucket="recommendation_events", event=event)

    def record_evaluation_event(
        self,
        *,
        artifact_key: str,
        ticker: str | None,
        outcome_label: str | None,
        adjusted_return_pct: float | None,
        detail: Mapping[str, Any],
    ) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": artifact_key,
            "ticker": str(ticker or "").strip() or None,
            "outcome_label": str(outcome_label or "").strip() or None,
            "return_after_cost_pct": self._coerce_float(adjusted_return_pct),
            "alpha_after_cost_pct": self._coerce_float(detail.get("alpha_after_cost_pct")),
            "execution_cost_bps": self._coerce_float(detail.get("execution_cost_bps")),
            "detail": dict(detail),
        }
        self._append_event(bucket="evaluation_events", event=event)

    def record_market_event(
        self,
        *,
        artifact_key: str,
        topic: str,
        detail: Mapping[str, Any],
        numeric_1: float | None = None,
        numeric_2: float | None = None,
    ) -> None:
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(artifact_key or "").strip(),
            "topic": str(topic or "").strip() or None,
            "numeric_1": self._coerce_float(numeric_1),
            "numeric_2": self._coerce_float(numeric_2),
            "detail": dict(detail),
        }
        self._append_event(bucket="market_events", event=event)

    def record_runtime_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        now = self._utc_now()
        with self._lock:
            if self._last_runtime_snapshot_at is not None:
                delta = (now - self._last_runtime_snapshot_at).total_seconds()
                if delta < self.runtime_snapshot_interval_seconds:
                    return
            self._last_runtime_snapshot_at = now
        runtime = snapshot.get("runtime") if isinstance(snapshot.get("runtime"), Mapping) else snapshot
        event = {
            "event_at": now.isoformat(),
            "db_healthy": (runtime.get("db_state") or {}).get("healthy") if isinstance(runtime, Mapping) else None,
            "mlflow_enabled": (runtime.get("mlflow") or {}).get("enabled") if isinstance(runtime, Mapping) else None,
            "open_circuit_count": len(runtime.get("open_circuits") or []) if isinstance(runtime, Mapping) else 0,
            "snapshot": dict(runtime) if isinstance(runtime, Mapping) else {},
        }
        self._append_event(bucket="runtime_snapshots", event=event)

    def _append_event(self, *, bucket: str, event: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(event)
        self._append_jsonl(bucket=bucket, event=payload)
        self._append_duckdb(bucket=bucket, event=payload)

    def _append_jsonl(self, *, bucket: str, event: Mapping[str, Any]) -> None:
        path = self._jsonl_dir / f"{bucket}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            warning = f"analytics_jsonl_write_failed: {exc}"
            logger.warning("Analytics JSONL write failed for {}: {}", bucket, exc)
            with self._lock:
                self._warning = warning
        else:
            with self._lock:
                self._last_write_at = self._utc_now().isoformat()

    def _append_duckdb(self, *, bucket: str, event: Mapping[str, Any]) -> None:
        duckdb = self._load_duckdb()
        if duckdb is None:
            return
        payload_json = json.dumps(dict(event), ensure_ascii=False)
        try:
            connection = duckdb.connect(str(self._db_path))
            try:
                connection.execute(
                    f"""
                    INSERT INTO {bucket} (event_at, artifact_key, topic, numeric_1, numeric_2, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.get("event_at") or ""),
                        str(event.get("artifact_key") or ""),
                        self._derive_topic(bucket=bucket, event=event),
                        self._derive_numeric_1(bucket=bucket, event=event),
                        self._derive_numeric_2(bucket=bucket, event=event),
                        payload_json,
                    ),
                )
            finally:
                connection.close()
        except Exception as exc:
            warning = f"duckdb_insert_failed: {exc}"
            logger.warning("DuckDB insert failed for {}: {}", bucket, exc)
            with self._lock:
                self._warning = warning
                self._backend = "jsonl"
            return
        with self._lock:
            self._backend = "duckdb"
            self._last_write_at = self._utc_now().isoformat()
        self._maybe_export_parquet()

    def _initialize_duckdb(self) -> None:
        duckdb = self._load_duckdb()
        if duckdb is None:
            return
        try:
            connection = duckdb.connect(str(self._db_path))
            try:
                for table_name in ("recommendation_events", "evaluation_events", "runtime_snapshots", "market_events"):
                    connection.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            event_at TIMESTAMP,
                            artifact_key VARCHAR,
                            topic VARCHAR,
                            numeric_1 DOUBLE,
                            numeric_2 DOUBLE,
                            payload_json JSON
                        )
                        """
                    )
            finally:
                connection.close()
        except Exception as exc:
            warning = f"duckdb_init_failed: {exc}"
            logger.warning("DuckDB initialization failed: {}", exc)
            with self._lock:
                self._warning = warning
                self._backend = "jsonl"
        else:
            with self._lock:
                self._backend = "duckdb"

    def _maybe_export_parquet(self) -> None:
        duckdb = self._load_duckdb()
        if duckdb is None:
            return
        now = self._utc_now()
        with self._lock:
            last_export = self._last_export_at
        if last_export:
            try:
                last_export_dt = datetime.fromisoformat(last_export)
            except ValueError:
                last_export_dt = None
            if last_export_dt is not None and (now - last_export_dt).total_seconds() < self.parquet_export_interval_seconds:
                return
        try:
            connection = duckdb.connect(str(self._db_path))
            try:
                for table_name in ("recommendation_events", "evaluation_events", "runtime_snapshots", "market_events"):
                    target = self._parquet_dir / f"{table_name}.parquet"
                    connection.execute(
                        f"COPY (SELECT * FROM {table_name}) TO ? (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)",
                        (str(target),),
                    )
            finally:
                connection.close()
        except Exception as exc:
            warning = f"duckdb_parquet_export_failed: {exc}"
            logger.warning("DuckDB parquet export failed: {}", exc)
            with self._lock:
                self._warning = warning
            return
        with self._lock:
            self._last_export_at = now.isoformat()

    def _load_duckdb(self) -> Any | None:
        try:
            import duckdb
        except Exception as exc:
            with self._lock:
                if self._warning is None:
                    self._warning = f"duckdb_unavailable: {exc}"
                if self._backend == "disabled":
                    self._backend = "jsonl"
            return None
        return duckdb

    @staticmethod
    def _derive_topic(*, bucket: str, event: Mapping[str, Any]) -> str | None:
        if bucket == "recommendation_events":
            return str(event.get("model") or "").strip() or None
        if bucket == "evaluation_events":
            return str(event.get("outcome_label") or "").strip() or None
        if bucket == "runtime_snapshots":
            return "runtime"
        if bucket == "market_events":
            return str(event.get("topic") or "").strip() or None
        return None

    @staticmethod
    def _derive_numeric_1(*, bucket: str, event: Mapping[str, Any]) -> float | None:
        if bucket == "recommendation_events":
            return AnalyticsStore._coerce_float(event.get("source_health_score"))
        if bucket == "evaluation_events":
            return AnalyticsStore._coerce_float(event.get("return_after_cost_pct"))
        if bucket == "runtime_snapshots":
            return AnalyticsStore._coerce_float(event.get("open_circuit_count"))
        if bucket == "market_events":
            return AnalyticsStore._coerce_float(event.get("numeric_1"))
        return None

    @staticmethod
    def _derive_numeric_2(*, bucket: str, event: Mapping[str, Any]) -> float | None:
        if bucket == "recommendation_events":
            return AnalyticsStore._coerce_float(event.get("data_quality_score"))
        if bucket == "evaluation_events":
            return AnalyticsStore._coerce_float(event.get("alpha_after_cost_pct"))
        if bucket == "runtime_snapshots":
            value = event.get("db_healthy")
            if value is None:
                return None
            return 1.0 if bool(value) else 0.0
        if bucket == "market_events":
            return AnalyticsStore._coerce_float(event.get("numeric_2"))
        return None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
