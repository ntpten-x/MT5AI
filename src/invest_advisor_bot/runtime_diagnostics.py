from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore


@dataclass(slots=True, frozen=True)
class ProviderSnapshot:
    provider: str
    model: str
    service: str
    succeeded_at: datetime


@dataclass(slots=True)
class JobSnapshot:
    name: str
    last_status: str = "unknown"
    last_run_at: datetime | None = None
    duration_ms: int | None = None
    success_count: int = 0
    failure_count: int = 0
    last_error: str | None = None


class RuntimeDiagnostics:
    def __init__(self) -> None:
        self._lock = RLock()
        self._history_store: RuntimeHistoryStore | None = None
        self._latest_provider_success: ProviderSnapshot | None = None
        self._response_stats: dict[str, dict[str, int]] = {}
        self._job_stats: dict[str, JobSnapshot] = {}
        self._alert_counts_by_day: dict[str, dict[str, int]] = {}
        self._db_state: dict[str, Any] = {"backend": "file", "healthy": None, "checked_at": None, "error": None}
        self._provider_circuit: dict[str, dict[str, Any]] = {}

    def attach_history_store(self, history_store: RuntimeHistoryStore | None) -> None:
        with self._lock:
            self._history_store = history_store

    def record_provider_success(self, *, provider: str, model: str, service: str) -> None:
        with self._lock:
            self._latest_provider_success = ProviderSnapshot(
                provider=provider,
                model=model,
                service=service,
                succeeded_at=datetime.now(timezone.utc),
            )
            history_store = self._history_store
        if history_store is not None:
            history_store.record_provider_event(
                provider=provider,
                model=model,
                service=service,
                status="success",
                detail={},
            )

    def record_provider_failure(
        self,
        *,
        provider: str,
        model: str,
        service: str | None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            history_store = self._history_store
        if history_store is not None:
            history_store.record_provider_event(
                provider=provider,
                model=model,
                service=service,
                status="failure",
                detail=detail,
            )

    def record_response(self, *, service: str, fallback_used: bool) -> None:
        with self._lock:
            stats = self._response_stats.setdefault(service, {"total": 0, "fallback": 0})
            stats["total"] += 1
            if fallback_used:
                stats["fallback"] += 1

    def record_job_run(
        self,
        *,
        job: str,
        status: str,
        duration_ms: int,
        error: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            snapshot = self._job_stats.setdefault(job, JobSnapshot(name=job))
            snapshot.last_status = status
            snapshot.last_run_at = datetime.now(timezone.utc)
            snapshot.duration_ms = duration_ms
            snapshot.last_error = error
            if status == "ok":
                snapshot.success_count += 1
            else:
                snapshot.failure_count += 1
            history_store = self._history_store
        if history_store is not None:
            payload = dict(detail or {})
            if error:
                payload.setdefault("error", error)
            history_store.record_job_event(
                job_name=job,
                status=status,
                duration_ms=duration_ms,
                detail=payload,
            )

    def record_alert_counts(self, *, categories: Mapping[str, int]) -> None:
        day_key = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            day_bucket = self._alert_counts_by_day.setdefault(day_key, {})
            for category, count in categories.items():
                day_bucket[category] = day_bucket.get(category, 0) + int(count)
            if len(self._alert_counts_by_day) > 14:
                for key in sorted(self._alert_counts_by_day)[:-14]:
                    self._alert_counts_by_day.pop(key, None)

    def record_db_state(self, *, backend: str, healthy: bool | None, error: str | None = None) -> None:
        with self._lock:
            self._db_state = {
                "backend": backend,
                "healthy": healthy,
                "checked_at": datetime.now(timezone.utc),
                "error": error,
            }

    def record_provider_circuit(
        self,
        *,
        provider: str,
        is_open: bool,
        failure_count: int,
        open_until: datetime | None,
    ) -> None:
        with self._lock:
            self._provider_circuit[provider] = {
                "is_open": is_open,
                "failure_count": failure_count,
                "open_until": open_until.isoformat() if open_until is not None else None,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest_provider = None
            if self._latest_provider_success is not None:
                latest_provider = {
                    "provider": self._latest_provider_success.provider,
                    "model": self._latest_provider_success.model,
                    "service": self._latest_provider_success.service,
                    "succeeded_at": self._latest_provider_success.succeeded_at.isoformat(),
                }
            response_stats: dict[str, dict[str, Any]] = {}
            for service, stats in self._response_stats.items():
                total = int(stats.get("total", 0))
                fallback = int(stats.get("fallback", 0))
                response_stats[service] = {
                    "total": total,
                    "fallback": fallback,
                    "fallback_rate": round((fallback / total) * 100.0, 1) if total else 0.0,
                }
            jobs = {
                name: {
                    "last_status": snapshot.last_status,
                    "last_run_at": snapshot.last_run_at.isoformat() if snapshot.last_run_at else None,
                    "duration_ms": snapshot.duration_ms,
                    "success_count": snapshot.success_count,
                    "failure_count": snapshot.failure_count,
                    "last_error": snapshot.last_error,
                }
                for name, snapshot in self._job_stats.items()
            }
            today_key = datetime.now(timezone.utc).date().isoformat()
            today_alerts = dict(self._alert_counts_by_day.get(today_key, {}))
            total_alerts_today = sum(today_alerts.values())
            db_state = dict(self._db_state)
            if isinstance(db_state.get("checked_at"), datetime):
                db_state["checked_at"] = db_state["checked_at"].isoformat()
            return {
                "latest_provider_success": latest_provider,
                "response_stats": response_stats,
                "jobs": jobs,
                "alerts_today": {
                    "total": total_alerts_today,
                    "by_category": today_alerts,
                },
                "db_state": db_state,
                "provider_circuit": dict(self._provider_circuit),
            }


diagnostics = RuntimeDiagnostics()
