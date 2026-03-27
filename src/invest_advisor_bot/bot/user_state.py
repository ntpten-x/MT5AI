from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from datetime import datetime, timezone

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


@dataclass(slots=True, frozen=True)
class UserPreferences:
    watchlist: tuple[str, ...] = ()
    preferred_sectors: tuple[str, ...] = ()
    stock_alert_threshold: float = 1.8
    daily_pick_enabled: bool = True
    dashboard_execution_filter: str | None = None
    approval_mode: str = "auto"
    max_position_size_pct: float | None = None


class UserStateStore:
    """Persist simple per-chat watchlists and alert preferences."""

    def __init__(self, *, path: Path, database_url: str = "") -> None:
        self.path = path
        self._lock = RLock()
        self._state: dict[str, dict[str, object]] = {}
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()
        else:
            self._load()

    def get(self, conversation_key: str) -> UserPreferences:
        if self._db is not None:
            return self._get_db(conversation_key)
        with self._lock:
            payload = self._state.get(conversation_key, {})
            watchlist = tuple(str(item).upper() for item in payload.get("watchlist", []) if str(item).strip()) if isinstance(payload, dict) else ()
            preferred_sectors = tuple(str(item).strip() for item in payload.get("preferred_sectors", []) if str(item).strip()) if isinstance(payload, dict) else ()
            threshold_raw = payload.get("stock_alert_threshold", 1.8) if isinstance(payload, dict) else 1.8
            daily_pick_enabled = bool(payload.get("daily_pick_enabled", True)) if isinstance(payload, dict) else True
            dashboard_execution_filter = self._normalize_dashboard_execution_filter(
                payload.get("dashboard_execution_filter") if isinstance(payload, dict) else None
            )
            approval_mode = self._normalize_approval_mode(payload.get("approval_mode") if isinstance(payload, dict) else None)
            max_position_size_pct = self._normalize_max_position_size(
                payload.get("max_position_size_pct") if isinstance(payload, dict) else None
            )
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = 1.8
        return UserPreferences(
            watchlist=watchlist,
            preferred_sectors=preferred_sectors,
            stock_alert_threshold=threshold,
            daily_pick_enabled=daily_pick_enabled,
            dashboard_execution_filter=dashboard_execution_filter,
            approval_mode=approval_mode,
            max_position_size_pct=max_position_size_pct,
        )

    def add_watchlist(self, conversation_key: str, ticker: str) -> UserPreferences:
        normalized = ticker.strip().upper()
        if not normalized:
            return self.get(conversation_key)
        if self._db is not None:
            prefs = self._get_db(conversation_key)
            watchlist = sorted({*prefs.watchlist, normalized})
            self._upsert_db(
                conversation_key,
                watchlist=watchlist,
                preferred_sectors=list(prefs.preferred_sectors),
                stock_alert_threshold=prefs.stock_alert_threshold,
                daily_pick_enabled=prefs.daily_pick_enabled,
                dashboard_execution_filter=prefs.dashboard_execution_filter,
                approval_mode=prefs.approval_mode,
                max_position_size_pct=prefs.max_position_size_pct,
            )
            return self.get(conversation_key)
        with self._lock:
            prefs = dict(self._state.get(conversation_key, {}))
            watchlist = {str(item).upper() for item in prefs.get("watchlist", []) if str(item).strip()}
            watchlist.add(normalized)
            prefs["watchlist"] = sorted(watchlist)
            self._state[conversation_key] = prefs
            self._persist()
        return self.get(conversation_key)

    def remove_watchlist(self, conversation_key: str, ticker: str) -> UserPreferences:
        normalized = ticker.strip().upper()
        if self._db is not None:
            prefs = self._get_db(conversation_key)
            watchlist = sorted(item for item in prefs.watchlist if item != normalized)
            self._upsert_db(
                conversation_key,
                watchlist=watchlist,
                preferred_sectors=list(prefs.preferred_sectors),
                stock_alert_threshold=prefs.stock_alert_threshold,
                daily_pick_enabled=prefs.daily_pick_enabled,
                dashboard_execution_filter=prefs.dashboard_execution_filter,
                approval_mode=prefs.approval_mode,
                max_position_size_pct=prefs.max_position_size_pct,
            )
            return self.get(conversation_key)
        with self._lock:
            prefs = dict(self._state.get(conversation_key, {}))
            watchlist = {str(item).upper() for item in prefs.get("watchlist", []) if str(item).strip()}
            watchlist.discard(normalized)
            prefs["watchlist"] = sorted(watchlist)
            self._state[conversation_key] = prefs
            self._persist()
        return self.get(conversation_key)

    def update_preferences(
        self,
        conversation_key: str,
        *,
        preferred_sectors: list[str] | None = None,
        stock_alert_threshold: float | None = None,
        daily_pick_enabled: bool | None = None,
        dashboard_execution_filter: str | None = None,
        approval_mode: str | None = None,
        max_position_size_pct: float | None = None,
    ) -> UserPreferences:
        if self._db is not None:
            current = self._get_db(conversation_key)
            self._upsert_db(
                conversation_key,
                watchlist=list(current.watchlist),
                preferred_sectors=preferred_sectors if preferred_sectors is not None else list(current.preferred_sectors),
                stock_alert_threshold=stock_alert_threshold if stock_alert_threshold is not None else current.stock_alert_threshold,
                daily_pick_enabled=daily_pick_enabled if daily_pick_enabled is not None else current.daily_pick_enabled,
                dashboard_execution_filter=(
                    self._normalize_dashboard_execution_filter(dashboard_execution_filter)
                    if dashboard_execution_filter is not None
                    else current.dashboard_execution_filter
                ),
                approval_mode=self._normalize_approval_mode(approval_mode) if approval_mode is not None else current.approval_mode,
                max_position_size_pct=(
                    self._normalize_max_position_size(max_position_size_pct)
                    if max_position_size_pct is not None
                    else current.max_position_size_pct
                ),
            )
            return self.get(conversation_key)
        with self._lock:
            prefs = dict(self._state.get(conversation_key, {}))
            if preferred_sectors is not None:
                prefs["preferred_sectors"] = [item.strip() for item in preferred_sectors if item.strip()]
            if stock_alert_threshold is not None:
                prefs["stock_alert_threshold"] = float(stock_alert_threshold)
            if daily_pick_enabled is not None:
                prefs["daily_pick_enabled"] = bool(daily_pick_enabled)
            if dashboard_execution_filter is not None:
                prefs["dashboard_execution_filter"] = self._normalize_dashboard_execution_filter(dashboard_execution_filter)
            if approval_mode is not None:
                prefs["approval_mode"] = self._normalize_approval_mode(approval_mode)
            if max_position_size_pct is not None:
                prefs["max_position_size_pct"] = self._normalize_max_position_size(max_position_size_pct)
            self._state[conversation_key] = prefs
            self._persist()
        return self.get(conversation_key)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if isinstance(payload, dict):
            self._state = payload

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_db(self, conversation_key: str) -> UserPreferences:
        assert self._db is not None
        row = self._db.fetch_one(
            """
            SELECT watchlist, preferred_sectors, stock_alert_threshold, daily_pick_enabled, dashboard_execution_filter,
                   approval_mode, max_position_size_pct
            FROM bot_user_preferences
            WHERE conversation_key = %s
            """,
            (conversation_key,),
        )
        if row is None:
            return UserPreferences()
        watchlist_raw, preferred_sectors_raw, threshold_raw, daily_pick_enabled, dashboard_execution_filter, approval_mode, max_position_size_pct = row
        watchlist = tuple(str(item).upper() for item in (watchlist_raw or []) if str(item).strip())
        preferred_sectors = tuple(str(item).strip() for item in (preferred_sectors_raw or []) if str(item).strip())
        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = 1.8
        return UserPreferences(
            watchlist=watchlist,
            preferred_sectors=preferred_sectors,
            stock_alert_threshold=threshold,
            daily_pick_enabled=bool(daily_pick_enabled),
            dashboard_execution_filter=self._normalize_dashboard_execution_filter(dashboard_execution_filter),
            approval_mode=self._normalize_approval_mode(approval_mode),
            max_position_size_pct=self._normalize_max_position_size(max_position_size_pct),
        )

    def _upsert_db(
        self,
        conversation_key: str,
        *,
        watchlist: list[str],
        preferred_sectors: list[str],
        stock_alert_threshold: float,
        daily_pick_enabled: bool,
        dashboard_execution_filter: str | None,
        approval_mode: str,
        max_position_size_pct: float | None,
    ) -> None:
        assert self._db is not None
        self._db.execute(
            """
            INSERT INTO bot_user_preferences (
                conversation_key,
                watchlist,
                preferred_sectors,
                stock_alert_threshold,
                daily_pick_enabled,
                dashboard_execution_filter,
                approval_mode,
                max_position_size_pct,
                updated_at
            )
            VALUES (%s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (conversation_key)
            DO UPDATE SET
                watchlist = EXCLUDED.watchlist,
                preferred_sectors = EXCLUDED.preferred_sectors,
                stock_alert_threshold = EXCLUDED.stock_alert_threshold,
                daily_pick_enabled = EXCLUDED.daily_pick_enabled,
                dashboard_execution_filter = EXCLUDED.dashboard_execution_filter,
                approval_mode = EXCLUDED.approval_mode,
                max_position_size_pct = EXCLUDED.max_position_size_pct,
                updated_at = EXCLUDED.updated_at
            """,
            (
                conversation_key,
                json.dumps(watchlist, ensure_ascii=False),
                json.dumps(preferred_sectors, ensure_ascii=False),
                float(stock_alert_threshold),
                bool(daily_pick_enabled),
                self._normalize_dashboard_execution_filter(dashboard_execution_filter),
                self._normalize_approval_mode(approval_mode),
                self._normalize_max_position_size(max_position_size_pct),
                datetime.now(timezone.utc),
            ),
        )

    @staticmethod
    def _normalize_dashboard_execution_filter(value: object) -> str | None:
        normalized = str(value or "").strip().casefold().replace("-", "_")
        if normalized in {"stock_pick", "macro_playbook", "macro_surprise"}:
            return normalized
        return None

    @staticmethod
    def _normalize_approval_mode(value: object) -> str:
        normalized = str(value or "").strip().casefold().replace("-", "_")
        if normalized in {"review", "review_only", "manual"}:
            return "review"
        if normalized in {"block", "blocked", "off"}:
            return "blocked"
        return "auto"

    @staticmethod
    def _normalize_max_position_size(value: object) -> float | None:
        if value in (None, "", 0, 0.0):
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        return round(max(0.5, min(10.0, normalized)), 1)
