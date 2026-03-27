from __future__ import annotations

from typing import Any

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend
from invest_advisor_bot.runtime_diagnostics import diagnostics


def sync_mlflow_diagnostics(services: object) -> dict[str, Any]:
    recommendation_service = getattr(services, "recommendation_service", None)
    observer = getattr(recommendation_service, "mlflow_observer", None)
    status_getter = getattr(observer, "status", None)
    mlflow_status = status_getter() if callable(status_getter) else None
    if not isinstance(mlflow_status, dict):
        mlflow_status = {
            "enabled": False,
            "tracking_configured": False,
            "experiment_name": None,
            "warning": "mlflow observer unavailable",
            "last_run_id": None,
            "last_run_kind": None,
            "last_run_name": None,
        }
    diagnostics.record_mlflow_state(mlflow_status)
    return dict(mlflow_status)


def sync_database_diagnostics(services: object, *, ping: bool) -> dict[str, Any]:
    database_url = str(getattr(services, "database_url", "") or "").strip()
    backend = "postgres" if database_url else "file"
    healthy: bool | None = None
    error: str | None = None
    if ping and database_url:
        try:
            healthy = bool(PostgresStateBackend.ping_database_url(database_url))
        except Exception as exc:
            healthy = False
            error = str(exc)
    diagnostics.record_db_state(backend=backend, healthy=healthy, error=error)
    return {"backend": backend, "healthy": healthy, "error": error}


def sync_service_diagnostics(services: object, *, ping_database: bool = False) -> dict[str, dict[str, Any]]:
    if services is None:
        diagnostics.record_mlflow_state(None)
        diagnostics.record_db_state(backend="file", healthy=None, error=None)
        return {}

    sync_mlflow_diagnostics(services)
    sync_database_diagnostics(services, ping=ping_database)

    service_instances = {
        "recommendation_service": getattr(services, "recommendation_service", None),
        "llm_client": getattr(getattr(services, "recommendation_service", None), "llm_client", None),
        "market_data_client": getattr(services, "market_data_client", None),
        "news_client": getattr(services, "news_client", None),
        "research_client": getattr(services, "research_client", None),
        "broker_client": getattr(services, "broker_client", None),
        "transcript_client": getattr(services, "transcript_client", None),
        "microstructure_client": getattr(services, "microstructure_client", None),
        "ownership_client": getattr(getattr(services, "recommendation_service", None), "ownership_client", None),
        "order_flow_client": getattr(getattr(services, "recommendation_service", None), "order_flow_client", None),
        "policy_feed_client": getattr(getattr(services, "recommendation_service", None), "policy_feed_client", None),
        "live_market_stream_client": getattr(services, "live_market_stream_client", None),
        "dbt_semantic_layer": getattr(getattr(services, "recommendation_service", None), "dbt_semantic_layer", None),
        "langfuse_observer": getattr(getattr(services, "recommendation_service", None), "langfuse_observer", None),
        "human_review_store": getattr(getattr(services, "recommendation_service", None), "human_review_store", None),
        "workflow_orchestrator": getattr(services, "workflow_orchestrator", None),
        "runtime_history_store": getattr(services, "runtime_history_store", None),
        "backup_manager": getattr(services, "backup_manager", None),
    }

    collected: dict[str, dict[str, Any]] = {}
    for service_name, instance in service_instances.items():
        status_getter = getattr(instance, "status", None)
        status = status_getter() if callable(status_getter) else None
        if not isinstance(status, dict):
            status = {"available": instance is not None}
        diagnostics.record_service_state(service=service_name, state=status)
        collected[service_name] = dict(status)
    return collected


def collect_runtime_snapshot(services: object, *, ping_database: bool = False) -> dict[str, Any]:
    sync_service_diagnostics(services, ping_database=ping_database)
    return diagnostics.snapshot()
