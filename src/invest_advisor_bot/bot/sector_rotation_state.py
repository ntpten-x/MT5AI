from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


@dataclass(slots=True, frozen=True)
class StoredSectorRotationSnapshot:
    observed_at: datetime
    regime: str
    sectors: dict[str, dict[str, Any]]
    market_breadth: dict[str, Any] | None = None


class SectorRotationStateStore:
    """Persist sector rotation snapshots for multi-session leadership analysis."""

    def __init__(self, *, path: Path, max_snapshots: int = 160, database_url: str = "") -> None:
        self.path = path
        self.max_snapshots = max(8, int(max_snapshots))
        self._lock = RLock()
        self._snapshots: list[StoredSectorRotationSnapshot] = []
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()
        else:
            self._load()

    def recent_snapshots(self, *, limit: int = 8, regime: str | None = None) -> list[StoredSectorRotationSnapshot]:
        if self._db is not None:
            return self._recent_snapshots_db(limit=limit, regime=regime)
        with self._lock:
            snapshots = self._snapshots
            if regime is not None:
                normalized_regime = regime.strip().casefold()
                snapshots = [item for item in snapshots if item.regime == normalized_regime]
            return list(snapshots[-max(1, limit):])

    def append_snapshot(
        self,
        sector_items: Sequence[Mapping[str, Any]],
        *,
        observed_at: datetime | None = None,
        regime: str = "intraday",
        market_breadth: Mapping[str, Any] | None = None,
    ) -> None:
        timestamp = observed_at or datetime.now(timezone.utc)
        normalized_regime = regime.strip().casefold() or "intraday"
        snapshot_payload: dict[str, dict[str, Any]] = {}
        for item in sector_items:
            sector = str(item.get("sector") or "").strip()
            if not sector:
                continue
            snapshot_payload[sector] = {
                "ticker": item.get("ticker"),
                "stance": item.get("stance"),
                "trend_score": item.get("trend_score"),
                "trend_direction": item.get("trend_direction"),
                "participation_ratio": item.get("participation_ratio"),
                "average_trend_score": item.get("average_trend_score"),
                "equal_weight_confirmed": item.get("equal_weight_confirmed"),
                "breadth_label": item.get("breadth_label"),
            }

        if not snapshot_payload:
            return

        snapshot = StoredSectorRotationSnapshot(
            observed_at=timestamp,
            regime=normalized_regime,
            sectors=snapshot_payload,
            market_breadth=dict(market_breadth) if isinstance(market_breadth, Mapping) else None,
        )
        if self._db is not None:
            self._append_snapshot_db(snapshot)
            return
        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots = self._snapshots[-self.max_snapshots :]
            self._persist()

    def cleanup_retention(self) -> None:
        if self._db is not None:
            assert self._db is not None
            self._db.execute(
                """
                DELETE FROM bot_sector_rotation_state
                WHERE id IN (
                    SELECT id
                    FROM bot_sector_rotation_state
                    ORDER BY observed_at DESC
                    OFFSET %s
                )
                """,
                (self.max_snapshots,),
            )
            return
        with self._lock:
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots = self._snapshots[-self.max_snapshots :]
                self._persist()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, list):
            return
        snapshots: list[StoredSectorRotationSnapshot] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            observed_at_raw = item.get("observed_at")
            regime_raw = item.get("regime") or "intraday"
            sectors_raw = item.get("sectors")
            market_breadth_raw = item.get("market_breadth")
            if not isinstance(observed_at_raw, str) or not isinstance(regime_raw, str) or not isinstance(sectors_raw, Mapping):
                continue
            try:
                observed_at = datetime.fromisoformat(observed_at_raw)
            except ValueError:
                continue
            snapshots.append(
                StoredSectorRotationSnapshot(
                    observed_at=observed_at,
                    regime=regime_raw.strip().casefold() or "intraday",
                    sectors={
                        str(sector): dict(values)
                        for sector, values in sectors_raw.items()
                        if isinstance(sector, str) and isinstance(values, Mapping)
                    },
                    market_breadth=dict(market_breadth_raw) if isinstance(market_breadth_raw, Mapping) else None,
                )
            )
        self._snapshots = snapshots[-self.max_snapshots :]

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "observed_at": snapshot.observed_at.isoformat(),
                "regime": snapshot.regime,
                "sectors": snapshot.sectors,
                "market_breadth": snapshot.market_breadth,
            }
            for snapshot in self._snapshots
        ]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _recent_snapshots_db(self, *, limit: int, regime: str | None) -> list[StoredSectorRotationSnapshot]:
        assert self._db is not None
        effective_limit = max(1, limit)
        if regime is not None:
            rows = self._db.fetch_all(
                """
                SELECT observed_at, regime, sectors, market_breadth
                FROM bot_sector_rotation_state
                WHERE regime = %s
                ORDER BY observed_at DESC
                LIMIT %s
                """,
                (regime.strip().casefold(), effective_limit),
            )
        else:
            rows = self._db.fetch_all(
                """
                SELECT observed_at, regime, sectors, market_breadth
                FROM bot_sector_rotation_state
                ORDER BY observed_at DESC
                LIMIT %s
                """,
                (effective_limit,),
            )
        snapshots = [
            StoredSectorRotationSnapshot(
                observed_at=observed_at,
                regime=str(stored_regime or "intraday").strip().casefold() or "intraday",
                sectors={str(sector): dict(values) for sector, values in dict(sectors or {}).items()},
                market_breadth=dict(market_breadth or {}) if isinstance(market_breadth, Mapping) else None,
            )
            for observed_at, stored_regime, sectors, market_breadth in reversed(rows)
            if isinstance(observed_at, datetime)
        ]
        return snapshots

    def _append_snapshot_db(self, snapshot: StoredSectorRotationSnapshot) -> None:
        assert self._db is not None
        self._db.execute(
            """
            INSERT INTO bot_sector_rotation_state (observed_at, regime, sectors, market_breadth)
            VALUES (%s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                snapshot.observed_at,
                snapshot.regime,
                json.dumps(snapshot.sectors, ensure_ascii=False),
                json.dumps(snapshot.market_breadth or {}, ensure_ascii=False),
            ),
        )
