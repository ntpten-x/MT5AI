from __future__ import annotations

from collections import deque
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


@dataclass(slots=True)
class ProviderLatencySnapshot:
    service: str
    provider: str
    operation: str
    last_latency_ms: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    window_size: int
    sample_count: int
    success_count: int
    failure_count: int
    last_status: str
    updated_at: datetime


class RuntimeDiagnostics:
    _provider_latency_window_size = 50

    def __init__(self) -> None:
        self._lock = RLock()
        self._started_at = datetime.now(timezone.utc)
        self._history_store: RuntimeHistoryStore | None = None
        self._latest_provider_success: ProviderSnapshot | None = None
        self._response_stats: dict[str, dict[str, int]] = {}
        self._job_stats: dict[str, JobSnapshot] = {}
        self._alert_counts_by_day: dict[str, dict[str, int]] = {}
        self._db_state: dict[str, Any] = {"backend": "file", "healthy": None, "checked_at": None, "error": None}
        self._provider_circuit: dict[str, dict[str, Any]] = {}
        self._provider_latency: dict[tuple[str, str, str], ProviderLatencySnapshot] = {}
        self._provider_latency_windows: dict[tuple[str, str, str], deque[float]] = {}
        self._service_state: dict[str, dict[str, Any]] = {}
        self._mlflow_state: dict[str, Any] = {
            "enabled": False,
            "tracking_configured": False,
            "experiment_name": None,
            "warning": None,
            "last_run_id": None,
            "last_run_kind": None,
            "last_run_name": None,
        }

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

    def record_provider_latency(
        self,
        *,
        service: str,
        provider: str,
        operation: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        service_name = str(service or "").strip()
        provider_name = str(provider or "").strip()
        operation_name = str(operation or "").strip()
        if not service_name or not provider_name or not operation_name:
            return
        normalized_latency = max(0.0, float(latency_ms))
        key = (service_name, provider_name, operation_name)
        with self._lock:
            existing = self._provider_latency.get(key)
            now = datetime.now(timezone.utc)
            latency_window = self._provider_latency_windows.get(key)
            if latency_window is None:
                latency_window = deque(maxlen=self._provider_latency_window_size)
                self._provider_latency_windows[key] = latency_window
            latency_window.append(normalized_latency)
            p95_latency = _percentile_from_window(latency_window, 95)
            p99_latency = _percentile_from_window(latency_window, 99)
            if existing is None:
                self._provider_latency[key] = ProviderLatencySnapshot(
                    service=service_name,
                    provider=provider_name,
                    operation=operation_name,
                    last_latency_ms=round(normalized_latency, 2),
                    avg_latency_ms=round(normalized_latency, 2),
                    p95_latency_ms=round(p95_latency, 2),
                    p99_latency_ms=round(p99_latency, 2),
                    window_size=len(latency_window),
                    sample_count=1,
                    success_count=1 if success else 0,
                    failure_count=0 if success else 1,
                    last_status="success" if success else "failure",
                    updated_at=now,
                )
                return
            sample_count = existing.sample_count + 1
            avg_latency = ((existing.avg_latency_ms * existing.sample_count) + normalized_latency) / sample_count
            self._provider_latency[key] = ProviderLatencySnapshot(
                service=service_name,
                provider=provider_name,
                operation=operation_name,
                last_latency_ms=round(normalized_latency, 2),
                avg_latency_ms=round(avg_latency, 2),
                p95_latency_ms=round(p95_latency, 2),
                p99_latency_ms=round(p99_latency, 2),
                window_size=len(latency_window),
                sample_count=sample_count,
                success_count=existing.success_count + (1 if success else 0),
                failure_count=existing.failure_count + (0 if success else 1),
                last_status="success" if success else "failure",
                updated_at=now,
            )

    def record_service_state(self, *, service: str, state: Mapping[str, Any] | None) -> None:
        normalized_service = str(service or "").strip()
        if not normalized_service:
            return
        payload = dict(state or {})
        with self._lock:
            self._service_state[normalized_service] = payload

    def record_mlflow_state(self, state: Mapping[str, Any] | None) -> None:
        normalized = {
            "enabled": False,
            "tracking_configured": False,
            "experiment_name": None,
            "warning": None,
            "last_run_id": None,
            "last_run_kind": None,
            "last_run_name": None,
        }
        if isinstance(state, Mapping):
            for key in tuple(normalized):
                normalized[key] = state.get(key)
        with self._lock:
            self._mlflow_state = normalized

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            started_at = self._started_at
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
            mlflow_state = dict(self._mlflow_state)
            service_state = {name: dict(state) for name, state in self._service_state.items()}
            provider_latency = [
                {
                    "service": item.service,
                    "provider": item.provider,
                    "operation": item.operation,
                    "last_latency_ms": item.last_latency_ms,
                    "avg_latency_ms": item.avg_latency_ms,
                    "p95_latency_ms": item.p95_latency_ms,
                    "p99_latency_ms": item.p99_latency_ms,
                    "window_size": item.window_size,
                    "sample_count": item.sample_count,
                    "success_count": item.success_count,
                    "failure_count": item.failure_count,
                    "last_status": item.last_status,
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in self._provider_latency.values()
            ]
            now = datetime.now(timezone.utc)
            return {
                "started_at": started_at.isoformat(),
                "uptime_seconds": round(max(0.0, (now - started_at).total_seconds()), 1),
                "latest_provider_success": latest_provider,
                "response_stats": response_stats,
                "jobs": jobs,
                "alerts_today": {
                    "total": total_alerts_today,
                    "by_category": today_alerts,
                },
                "db_state": db_state,
                "provider_circuit": dict(self._provider_circuit),
                "provider_latency": provider_latency,
                "services": service_state,
                "mlflow": mlflow_state,
            }


diagnostics = RuntimeDiagnostics()


def render_prometheus_metrics(snapshot: Mapping[str, Any] | None) -> str:
    payload = dict(snapshot or {})
    lines: list[str] = []

    def add_help(name: str, help_text: str, metric_type: str = "gauge") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")

    def add_metric(name: str, value: float | int, labels: Mapping[str, Any] | None = None) -> None:
        if labels:
            rendered_labels = ",".join(
                f'{_escape_prometheus_label(str(key))}="{_escape_prometheus_label_value(str(label_value))}"'
                for key, label_value in sorted(labels.items())
            )
            lines.append(f"{name}{{{rendered_labels}}} {value}")
            return
        lines.append(f"{name} {value}")

    add_help("invest_advisor_bot_uptime_seconds", "Runtime uptime in seconds.")
    uptime = payload.get("uptime_seconds")
    if isinstance(uptime, (int, float)):
        add_metric("invest_advisor_bot_uptime_seconds", float(uptime))

    add_help("invest_advisor_bot_alerts_today_total", "Total alerts emitted today.")
    alerts_today = payload.get("alerts_today")
    if isinstance(alerts_today, Mapping):
        total_alerts = alerts_today.get("total")
        if isinstance(total_alerts, (int, float)):
            add_metric("invest_advisor_bot_alerts_today_total", float(total_alerts))
        by_category = alerts_today.get("by_category")
        add_help("invest_advisor_bot_alerts_today_by_category", "Alerts emitted today by category.")
        if isinstance(by_category, Mapping):
            for category, count in by_category.items():
                if isinstance(count, (int, float)):
                    add_metric(
                        "invest_advisor_bot_alerts_today_by_category",
                        float(count),
                        labels={"category": category},
                    )

    add_help("invest_advisor_bot_db_healthy", "Database health state, 1 for healthy and 0 for unhealthy.")
    db_state = payload.get("db_state")
    if isinstance(db_state, Mapping) and isinstance(db_state.get("healthy"), bool):
        add_metric("invest_advisor_bot_db_healthy", 1 if db_state.get("healthy") else 0)

    add_help("invest_advisor_bot_mlflow_enabled", "MLflow observer enabled state.")
    add_help("invest_advisor_bot_mlflow_tracking_configured", "MLflow tracking URI configured state.")
    add_help("invest_advisor_bot_mlflow_warning", "MLflow warning present state.")
    mlflow_state = payload.get("mlflow")
    if isinstance(mlflow_state, Mapping):
        add_metric("invest_advisor_bot_mlflow_enabled", 1 if bool(mlflow_state.get("enabled")) else 0)
        add_metric(
            "invest_advisor_bot_mlflow_tracking_configured",
            1 if bool(mlflow_state.get("tracking_configured")) else 0,
        )
        add_metric("invest_advisor_bot_mlflow_warning", 1 if bool(mlflow_state.get("warning")) else 0)

    add_help("invest_advisor_bot_provider_circuit_open", "Provider circuit breaker open state.")
    add_help("invest_advisor_bot_provider_circuit_failures", "Provider circuit breaker failure count.")
    provider_circuit = payload.get("provider_circuit")
    if isinstance(provider_circuit, Mapping):
        for provider, state in provider_circuit.items():
            if not isinstance(state, Mapping):
                continue
            add_metric(
                "invest_advisor_bot_provider_circuit_open",
                1 if bool(state.get("is_open")) else 0,
                labels={"provider": provider},
            )
            failure_count = state.get("failure_count")
            if isinstance(failure_count, (int, float)):
                add_metric(
                    "invest_advisor_bot_provider_circuit_failures",
                    float(failure_count),
                    labels={"provider": provider},
                )

    add_help("invest_advisor_bot_provider_latency_last_ms", "Last observed provider latency in milliseconds.")
    add_help("invest_advisor_bot_provider_latency_avg_ms", "Average observed provider latency in milliseconds.")
    add_help("invest_advisor_bot_provider_latency_p95_ms", "Rolling p95 provider latency in milliseconds.")
    add_help("invest_advisor_bot_provider_latency_p99_ms", "Rolling p99 provider latency in milliseconds.")
    add_help("invest_advisor_bot_provider_latency_window_size", "Rolling latency window size.")
    add_help("invest_advisor_bot_provider_latency_samples_total", "Total provider latency samples.", metric_type="counter")
    add_help("invest_advisor_bot_provider_latency_success_total", "Successful provider calls recorded.", metric_type="counter")
    add_help("invest_advisor_bot_provider_latency_failure_total", "Failed provider calls recorded.", metric_type="counter")
    provider_latency = payload.get("provider_latency")
    if isinstance(provider_latency, list):
        for item in provider_latency:
            if not isinstance(item, Mapping):
                continue
            labels = {
                "service": item.get("service"),
                "provider": item.get("provider"),
                "operation": item.get("operation"),
            }
            for key, metric_name in (
                ("last_latency_ms", "invest_advisor_bot_provider_latency_last_ms"),
                ("avg_latency_ms", "invest_advisor_bot_provider_latency_avg_ms"),
                ("p95_latency_ms", "invest_advisor_bot_provider_latency_p95_ms"),
                ("p99_latency_ms", "invest_advisor_bot_provider_latency_p99_ms"),
                ("window_size", "invest_advisor_bot_provider_latency_window_size"),
                ("sample_count", "invest_advisor_bot_provider_latency_samples_total"),
                ("success_count", "invest_advisor_bot_provider_latency_success_total"),
                ("failure_count", "invest_advisor_bot_provider_latency_failure_total"),
            ):
                metric_value = item.get(key)
                if isinstance(metric_value, (int, float)):
                    add_metric(metric_name, float(metric_value), labels=labels)

    add_help("invest_advisor_bot_response_total", "Total responses by service.", metric_type="counter")
    add_help("invest_advisor_bot_response_fallback_total", "Fallback responses by service.", metric_type="counter")
    add_help("invest_advisor_bot_response_fallback_rate_pct", "Fallback rate percentage by service.")
    response_stats = payload.get("response_stats")
    if isinstance(response_stats, Mapping):
        for service, state in response_stats.items():
            if not isinstance(state, Mapping):
                continue
            labels = {"service": service}
            for key, metric_name in (
                ("total", "invest_advisor_bot_response_total"),
                ("fallback", "invest_advisor_bot_response_fallback_total"),
                ("fallback_rate", "invest_advisor_bot_response_fallback_rate_pct"),
            ):
                metric_value = state.get(key)
                if isinstance(metric_value, (int, float)):
                    add_metric(metric_name, float(metric_value), labels=labels)

    add_help("invest_advisor_bot_job_success_total", "Successful job runs.", metric_type="counter")
    add_help("invest_advisor_bot_job_failure_total", "Failed job runs.", metric_type="counter")
    add_help("invest_advisor_bot_job_duration_ms", "Last job duration in milliseconds.")
    jobs = payload.get("jobs")
    if isinstance(jobs, Mapping):
        for job_name, state in jobs.items():
            if not isinstance(state, Mapping):
                continue
            labels = {"job": job_name}
            success_count = state.get("success_count")
            if isinstance(success_count, (int, float)):
                add_metric("invest_advisor_bot_job_success_total", float(success_count), labels=labels)
            failure_count = state.get("failure_count")
            if isinstance(failure_count, (int, float)):
                add_metric("invest_advisor_bot_job_failure_total", float(failure_count), labels=labels)
            duration_ms = state.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                add_metric("invest_advisor_bot_job_duration_ms", float(duration_ms), labels=labels)

    add_help("invest_advisor_bot_service_available", "Service availability state.")
    services = payload.get("services")
    if isinstance(services, Mapping):
        for service_name, state in services.items():
            if not isinstance(state, Mapping):
                continue
            add_metric(
                "invest_advisor_bot_service_available",
                1 if bool(state.get("available")) else 0,
                labels={"service": service_name},
            )

    return "\n".join(lines) + "\n"


def _escape_prometheus_label(value: str) -> str:
    return value.replace("\\", "_").replace('"', "_").replace("\n", "_")


def _escape_prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _percentile_from_window(values: deque[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = ((len(ordered) - 1) * max(0, min(100, percentile))) / 100.0
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)
