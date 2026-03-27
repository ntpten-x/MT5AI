from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Iterable

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


class AlertStateStore:
    """Persist recently-sent alert keys to avoid repeated Telegram spam."""

    def __init__(self, *, path: Path, suppression_minutes: int = 180, database_url: str = "") -> None:
        self.path = path
        self.suppression = timedelta(minutes=max(1, suppression_minutes))
        self._lock = RLock()
        self._state: dict[str, datetime] = {}
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()
        else:
            self._load()

    def filter_new_keys(self, keys: Iterable[str]) -> list[str]:
        now = datetime.now(timezone.utc)
        if self._db is not None:
            return self._filter_new_keys_db(keys, now=now)
        accepted: list[str] = []
        with self._lock:
            self._prune(now)
            for key in keys:
                last_seen = self._state.get(key)
                if last_seen is not None and now - last_seen < self.suppression:
                    continue
                self._state[key] = now
                accepted.append(key)
            if accepted:
                self._persist()
        return accepted

    def filter_alerts(self, alerts: Iterable[object]) -> list[object]:
        now = datetime.now(timezone.utc)
        normalized = [alert for alert in alerts if str(getattr(alert, "key", "") or "").strip()]
        if not normalized:
            return []
        if self._db is not None:
            return self._filter_alerts_db(normalized, now=now)

        accepted: list[object] = []
        with self._lock:
            self._prune(now)
            for alert in normalized:
                key = str(getattr(alert, "key", "") or "").strip()
                cadence = self._resolve_alert_cadence(alert)
                last_seen = self._state.get(key)
                if last_seen is not None and now - last_seen < cadence:
                    continue
                self._state[key] = now
                accepted.append(alert)
            if accepted:
                self._persist()
        return accepted

    def _filter_new_keys_db(self, keys: Iterable[str], *, now: datetime) -> list[str]:
        normalized = [key for key in dict.fromkeys(str(key).strip() for key in keys) if key]
        if not normalized:
            return []
        assert self._db is not None
        cutoff = now - self._retention_window()
        self._db.execute("DELETE FROM bot_alert_state WHERE last_seen < %s", (cutoff,))
        rows = self._db.fetch_all(
            "SELECT alert_key, last_seen FROM bot_alert_state WHERE alert_key = ANY(%s)",
            (normalized,),
        )
        seen_map = {
            str(key): value
            for key, value in rows
            if isinstance(key, str) and isinstance(value, datetime)
        }
        accepted: list[str] = []
        upserts: list[tuple[object, ...]] = []
        for key in normalized:
            last_seen = seen_map.get(key)
            if last_seen is not None and now - last_seen < self.suppression:
                continue
            accepted.append(key)
            upserts.append((key, now))
        if upserts:
            self._db.executemany(
                """
                INSERT INTO bot_alert_state (alert_key, last_seen)
                VALUES (%s, %s)
                ON CONFLICT (alert_key)
                DO UPDATE SET last_seen = EXCLUDED.last_seen
                """,
                upserts,
            )
        return accepted

    def _filter_alerts_db(self, alerts: Iterable[object], *, now: datetime) -> list[object]:
        normalized: list[object] = []
        key_map: dict[str, object] = {}
        for alert in alerts:
            key = str(getattr(alert, "key", "") or "").strip()
            if not key or key in key_map:
                continue
            key_map[key] = alert
            normalized.append(alert)
        if not normalized:
            return []
        assert self._db is not None
        cutoff = now - self._retention_window()
        self._db.execute("DELETE FROM bot_alert_state WHERE last_seen < %s", (cutoff,))
        rows = self._db.fetch_all(
            "SELECT alert_key, last_seen FROM bot_alert_state WHERE alert_key = ANY(%s)",
            ([str(getattr(alert, "key", "") or "").strip() for alert in normalized],),
        )
        seen_map = {
            str(key): value
            for key, value in rows
            if isinstance(key, str) and isinstance(value, datetime)
        }
        accepted: list[object] = []
        upserts: list[tuple[object, ...]] = []
        for alert in normalized:
            key = str(getattr(alert, "key", "") or "").strip()
            cadence = self._resolve_alert_cadence(alert)
            last_seen = seen_map.get(key)
            if last_seen is not None and now - last_seen < cadence:
                continue
            accepted.append(alert)
            upserts.append((key, now))
        if upserts:
            self._db.executemany(
                """
                INSERT INTO bot_alert_state (alert_key, last_seen)
                VALUES (%s, %s)
                ON CONFLICT (alert_key)
                DO UPDATE SET last_seen = EXCLUDED.last_seen
                """,
                upserts,
            )
        return accepted

    def _resolve_alert_cadence(self, alert: object) -> timedelta:
        metadata = getattr(alert, "metadata", None)
        if not isinstance(metadata, dict):
            return self.suppression
        raw_minutes = metadata.get("realert_after_minutes")
        try:
            minutes = int(float(raw_minutes))
        except (TypeError, ValueError):
            return self.suppression
        return timedelta(minutes=max(1, minutes))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        for key, raw_value in payload.items():
            if not isinstance(key, str) or not isinstance(raw_value, str):
                continue
            try:
                self._state[key] = datetime.fromisoformat(raw_value)
            except ValueError:
                continue

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value.isoformat() for key, value in self._state.items()}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _prune(self, now: datetime) -> None:
        retention = self._retention_window()
        expired = [key for key, timestamp in self._state.items() if now - timestamp >= retention]
        for key in expired:
            self._state.pop(key, None)

    def _retention_window(self) -> timedelta:
        return max(self.suppression, timedelta(days=7))
