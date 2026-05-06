from __future__ import annotations

import hashlib
import json
import re
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
                keys = self._alert_identity_keys(alert)
                cadence = self._resolve_alert_cadence(alert)
                last_seen = self._last_seen_for_any(keys)
                if last_seen is not None and now - last_seen < cadence:
                    continue
                for key in keys:
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
        identities_by_alert = {id(alert): self._alert_identity_keys(alert) for alert in normalized}
        lookup_keys = sorted({key for keys in identities_by_alert.values() for key in keys})
        cutoff = now - self._retention_window()
        self._db.execute("DELETE FROM bot_alert_state WHERE last_seen < %s", (cutoff,))
        rows = self._db.fetch_all(
            "SELECT alert_key, last_seen FROM bot_alert_state WHERE alert_key = ANY(%s)",
            (lookup_keys,),
        )
        seen_map = {
            str(key): value
            for key, value in rows
            if isinstance(key, str) and isinstance(value, datetime)
        }
        accepted: list[object] = []
        upserts: list[tuple[object, ...]] = []
        for alert in normalized:
            keys = identities_by_alert.get(id(alert)) or self._alert_identity_keys(alert)
            cadence = self._resolve_alert_cadence(alert)
            last_seen_values = [seen_map[key] for key in keys if key in seen_map]
            last_seen = max(last_seen_values) if last_seen_values else None
            if last_seen is not None and now - last_seen < cadence:
                continue
            accepted.append(alert)
            upserts.extend((key, now) for key in keys)
            for key in keys:
                seen_map[key] = now
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

    def _alert_identity_keys(self, alert: object) -> tuple[str, ...]:
        raw_key = str(getattr(alert, "key", "") or "").strip()
        identities = [raw_key]
        text = str(getattr(alert, "text", "") or "").strip()
        normalized_text = self._normalize_alert_text(text)
        if normalized_text:
            identities.append(f"text:{self._digest(normalized_text)}")
        metadata = getattr(alert, "metadata", None)
        if isinstance(metadata, dict):
            alert_kind = str(metadata.get("alert_kind") or raw_key.split(":", 1)[0] or "alert").strip()
            semantic_parts = [
                alert_kind,
                metadata.get("ticker"),
                metadata.get("indicator"),
                metadata.get("event_key"),
                metadata.get("event_name"),
                metadata.get("actual"),
                metadata.get("baseline"),
                metadata.get("surprise"),
                metadata.get("severity"),
            ]
            semantic = "|".join(str(item).strip().lower() for item in semantic_parts if item not in (None, ""))
            if semantic:
                identities.append(f"semantic:{self._digest(semantic)}")
        return tuple(dict.fromkeys(identity for identity in identities if identity))

    def _last_seen_for_any(self, keys: Iterable[str]) -> datetime | None:
        matches = [self._state[key] for key in keys if key in self._state]
        return max(matches) if matches else None

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _normalize_alert_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}t\d{2}:\d{2}(?::\d{2})?(?:\+\d{2}:\d{2}|z)?\b", "<timestamp>", normalized)
        return normalized

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
