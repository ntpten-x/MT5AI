from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


@dataclass(slots=True, frozen=True)
class BackupTableSpec:
    name: str
    columns: tuple[str, ...]
    json_columns: tuple[str, ...] = ()
    order_by: tuple[str, ...] = ()

    def select_sql(self) -> str:
        columns = ", ".join(self.columns)
        query = f"SELECT {columns} FROM {self.name}"
        if self.order_by:
            query += " ORDER BY " + ", ".join(self.order_by)
        return query

    def insert_sql(self) -> str:
        columns = ", ".join(self.columns)
        placeholders = ", ".join(
            "%s::jsonb" if column in self.json_columns else "%s"
            for column in self.columns
        )
        return f"INSERT INTO {self.name} ({columns}) VALUES ({placeholders})"

    def serialize_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append({column: _serialize_value(row.get(column)) for column in self.columns})
        return payload

    def restore_params(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for column in self.columns:
            value = row.get(column)
            if column in self.json_columns:
                values.append(json.dumps(value if value is not None else {}, ensure_ascii=False))
            else:
                values.append(value)
        return tuple(values)


@dataclass(slots=True, frozen=True)
class BackupManifest:
    path: Path
    created_at: datetime
    row_counts: dict[str, int]


BACKUP_TABLE_SPECS: tuple[BackupTableSpec, ...] = (
    BackupTableSpec(
        name="bot_user_preferences",
        columns=(
            "conversation_key",
            "watchlist",
            "preferred_sectors",
            "stock_alert_threshold",
            "daily_pick_enabled",
            "dashboard_execution_filter",
            "approval_mode",
            "max_position_size_pct",
            "updated_at",
        ),
        json_columns=("watchlist", "preferred_sectors"),
        order_by=("conversation_key",),
    ),
    BackupTableSpec(
        name="bot_alert_state",
        columns=("alert_key", "last_seen"),
        order_by=("alert_key",),
    ),
    BackupTableSpec(
        name="bot_report_memory",
        columns=("day_key", "report_kind", "created_at", "summary"),
        order_by=("day_key", "report_kind"),
    ),
    BackupTableSpec(
        name="bot_portfolio_holdings",
        columns=("conversation_key", "ticker", "quantity", "avg_cost", "note", "updated_at"),
        order_by=("conversation_key", "ticker"),
    ),
    BackupTableSpec(
        name="bot_sector_rotation_state",
        columns=("observed_at", "regime", "sectors", "market_breadth"),
        json_columns=("sectors", "market_breadth"),
        order_by=("observed_at",),
    ),
    BackupTableSpec(
        name="bot_provider_diagnostics_history",
        columns=("event_at", "provider", "model", "service", "status", "detail"),
        json_columns=("detail",),
        order_by=("event_at",),
    ),
    BackupTableSpec(
        name="bot_job_diagnostics_history",
        columns=("event_at", "job_name", "status", "duration_ms", "detail"),
        json_columns=("detail",),
        order_by=("event_at",),
    ),
    BackupTableSpec(
        name="bot_sent_reports_history",
        columns=("sent_at", "report_kind", "chat_id", "fallback_used", "model", "summary", "detail"),
        json_columns=("detail",),
        order_by=("sent_at",),
    ),
    BackupTableSpec(
        name="bot_user_interaction_history",
        columns=(
            "occurred_at",
            "conversation_key",
            "interaction_kind",
            "question",
            "response_excerpt",
            "fallback_used",
            "model",
            "detail",
        ),
        json_columns=("detail",),
        order_by=("occurred_at",),
    ),
    BackupTableSpec(
        name="bot_alert_audit",
        columns=("sent_at", "alert_key", "severity", "category", "chat_id", "text_excerpt", "detail"),
        json_columns=("detail",),
        order_by=("sent_at",),
    ),
    BackupTableSpec(
        name="bot_stock_pick_scorecard",
        columns=(
            "recommendation_key",
            "created_at",
            "due_at",
            "source_kind",
            "chat_id",
            "ticker",
            "company_name",
            "stance",
            "confidence_score",
            "confidence_label",
            "composite_score",
            "entry_price",
            "evaluated_at",
            "exit_price",
            "return_pct",
            "outcome_label",
            "status",
            "detail",
        ),
        json_columns=("detail",),
        order_by=("created_at", "recommendation_key"),
    ),
)


class BackupManager:
    """Create and restore JSON backups for Postgres-backed bot state."""

    def __init__(
        self,
        *,
        backup_dir: Path,
        database_url: str,
        retention_days: int = 14,
        backend: PostgresStateBackend | None = None,
    ) -> None:
        self.backup_dir = backup_dir
        self.retention_days = max(3, int(retention_days))
        self._db = backend or (PostgresStateBackend(database_url=database_url) if database_url.strip() else None)
        if self._db is not None:
            self._db.ensure_schema()

    def available(self) -> bool:
        return self._db is not None

    def status(self) -> dict[str, Any]:
        latest_backup = self.latest_backup_path()
        return {
            "available": self.available(),
            "backup_dir": str(self.backup_dir),
            "retention_days": self.retention_days,
            "latest_backup_path": str(latest_backup) if latest_backup is not None else None,
        }

    def latest_backup_path(self) -> Path | None:
        if not self.backup_dir.exists():
            return None
        candidates = sorted(self.backup_dir.glob("invest_advisor_backup_*.json"))
        return candidates[-1] if candidates else None

    def create_backup(self, *, reason: str = "manual") -> BackupManifest:
        if self._db is None:
            raise RuntimeError("DATABASE_URL is required for backup creation")

        created_at = datetime.now(timezone.utc)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        tables: dict[str, list[dict[str, Any]]] = {}
        row_counts: dict[str, int] = {}
        for spec in BACKUP_TABLE_SPECS:
            rows = self._db.fetch_dicts(spec.select_sql())
            serialized = spec.serialize_rows(rows)
            tables[spec.name] = serialized
            row_counts[spec.name] = len(serialized)

        payload = {
            "schema_version": 1,
            "created_at": created_at.isoformat(),
            "reason": reason,
            "tables": tables,
        }
        timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        path = self.backup_dir / f"invest_advisor_backup_{timestamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.cleanup_retention()
        return BackupManifest(path=path, created_at=created_at, row_counts=row_counts)

    def restore_backup(self, backup_path: Path | str) -> BackupManifest:
        if self._db is None:
            raise RuntimeError("DATABASE_URL is required for backup restore")

        path = Path(backup_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("tables"), Mapping):
            raise ValueError("Invalid backup payload")

        tables_payload = payload["tables"]
        for spec in reversed(BACKUP_TABLE_SPECS):
            self._db.execute(f"DELETE FROM {spec.name}")

        row_counts: dict[str, int] = {}
        for spec in BACKUP_TABLE_SPECS:
            rows_raw = tables_payload.get(spec.name, [])
            if not isinstance(rows_raw, Sequence):
                continue
            normalized_rows = [row for row in rows_raw if isinstance(row, Mapping)]
            if normalized_rows:
                self._db.executemany(spec.insert_sql(), [spec.restore_params(row) for row in normalized_rows])
            row_counts[spec.name] = len(normalized_rows)

        created_at = _coerce_datetime(payload.get("created_at")) or datetime.now(timezone.utc)
        return BackupManifest(path=path, created_at=created_at, row_counts=row_counts)

    def cleanup_retention(self) -> int:
        if not self.backup_dir.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        deleted = 0
        for path in self.backup_dir.glob("invest_advisor_backup_*.json"):
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if modified_at >= cutoff:
                continue
            try:
                path.unlink()
                deleted += 1
            except OSError:
                continue
        return deleted


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_value(item) for item in value]
    return value


def _coerce_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
