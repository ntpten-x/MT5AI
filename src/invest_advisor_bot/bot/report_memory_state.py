from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Mapping

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


@dataclass(slots=True, frozen=True)
class ReportMemoryEntry:
    report_kind: str
    created_at: datetime
    summary: str


class ReportMemoryStore:
    """Persist same-day report summaries so morning/midday/closing can form one narrative."""

    def __init__(self, *, path: Path, max_days: int = 10, database_url: str = "") -> None:
        self.path = path
        self.max_days = max(3, int(max_days))
        self._lock = RLock()
        self._state: dict[str, dict[str, dict[str, str]]] = {}
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()
        else:
            self._load()

    def get_day_entries(self, *, day_key: str | None = None) -> dict[str, ReportMemoryEntry]:
        effective_day = day_key or datetime.now(timezone.utc).date().isoformat()
        if self._db is not None:
            return self._get_day_entries_db(effective_day)
        with self._lock:
            payload = dict(self._state.get(effective_day, {}))
        entries: dict[str, ReportMemoryEntry] = {}
        for report_kind, item in payload.items():
            if not isinstance(item, Mapping):
                continue
            created_at_raw = item.get("created_at")
            summary = str(item.get("summary") or "").strip()
            if not isinstance(created_at_raw, str) or not summary:
                continue
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                continue
            entries[report_kind] = ReportMemoryEntry(report_kind=report_kind, created_at=created_at, summary=summary)
        return entries

    def remember(self, *, report_kind: str, summary: str, day_key: str | None = None) -> None:
        normalized_kind = report_kind.strip().casefold()
        if not normalized_kind or not summary.strip():
            return
        effective_day = day_key or datetime.now(timezone.utc).date().isoformat()
        now = datetime.now(timezone.utc)
        if self._db is not None:
            self._remember_db(report_kind=normalized_kind, summary=summary.strip(), day_key=effective_day, created_at=now)
            return
        with self._lock:
            day_payload = dict(self._state.get(effective_day, {}))
            day_payload[normalized_kind] = {"created_at": now.isoformat(), "summary": summary.strip()}
            self._state[effective_day] = day_payload
            self._trim_days()
            self._persist()

    def _trim_days(self) -> None:
        if len(self._state) <= self.max_days:
            return
        ordered = sorted(self._state)
        for key in ordered[:-self.max_days]:
            self._state.pop(key, None)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return
        if isinstance(payload, dict):
            self._state = {
                str(day): dict(entries)
                for day, entries in payload.items()
                if isinstance(day, str) and isinstance(entries, Mapping)
            }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_day_entries_db(self, day_key: str) -> dict[str, ReportMemoryEntry]:
        assert self._db is not None
        rows = self._db.fetch_all(
            """
            SELECT report_kind, created_at, summary
            FROM bot_report_memory
            WHERE day_key = %s
            ORDER BY created_at ASC
            """,
            (day_key,),
        )
        entries: dict[str, ReportMemoryEntry] = {}
        for report_kind, created_at, summary in rows:
            if not isinstance(report_kind, str) or not isinstance(summary, str):
                continue
            if not isinstance(created_at, datetime):
                continue
            entries[report_kind] = ReportMemoryEntry(
                report_kind=report_kind,
                created_at=created_at,
                summary=summary,
            )
        return entries

    def _remember_db(self, *, report_kind: str, summary: str, day_key: str, created_at: datetime) -> None:
        assert self._db is not None
        self._db.execute(
            """
            INSERT INTO bot_report_memory (day_key, report_kind, created_at, summary)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (day_key, report_kind)
            DO UPDATE SET
                created_at = EXCLUDED.created_at,
                summary = EXCLUDED.summary
            """,
            (day_key, report_kind, created_at, summary),
        )
        self._db.execute(
            """
            DELETE FROM bot_report_memory
            WHERE day_key < CURRENT_DATE - (%s * INTERVAL '1 day')
            """,
            (self.max_days,),
        )
