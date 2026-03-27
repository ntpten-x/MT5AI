from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from loguru import logger


@dataclass(slots=True, frozen=True)
class PrefectFlowSpec:
    name: str
    description: str


class WorkflowOrchestrator:
    """Optional Prefect workflow catalog for production orchestration."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._warning: str | None = None
        self._prefect_available = self._load_prefect() is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.enabled and self._prefect_available,
            "enabled": self.enabled,
            "prefect_available": self._prefect_available,
            "warning": self._warning,
            "flows": [asdict(spec) for spec in self.flow_catalog()],
        }

    @staticmethod
    def flow_catalog() -> tuple[PrefectFlowSpec, ...]:
        return (
            PrefectFlowSpec("invest-advisor-runtime-snapshot", "Collects runtime diagnostics and service status."),
            PrefectFlowSpec("invest-advisor-market-update", "Generates a market update payload and response text."),
            PrefectFlowSpec("invest-advisor-broker-account-snapshot", "Collects broker paper account and positions."),
        )

    def build_runtime_snapshot_flow(self, *, snapshot_factory: Callable[[], Mapping[str, Any]]) -> Callable[[], Mapping[str, Any]]:
        prefect = self._load_prefect()
        if prefect is None or not self.enabled:
            return lambda: dict(snapshot_factory())

        @prefect.flow(name="invest-advisor-runtime-snapshot")
        def runtime_snapshot_flow() -> Mapping[str, Any]:
            return dict(snapshot_factory())

        return runtime_snapshot_flow

    def build_market_update_flow(
        self,
        *,
        market_update_factory: Callable[[], Any],
    ) -> Callable[[], Any]:
        prefect = self._load_prefect()
        if prefect is None or not self.enabled:
            return market_update_factory

        @prefect.flow(name="invest-advisor-market-update")
        def market_update_flow() -> Any:
            return market_update_factory()

        return market_update_flow

    def build_broker_account_flow(
        self,
        *,
        broker_account_factory: Callable[[], Any],
    ) -> Callable[[], Any]:
        prefect = self._load_prefect()
        if prefect is None or not self.enabled:
            return broker_account_factory

        @prefect.flow(name="invest-advisor-broker-account-snapshot")
        def broker_account_flow() -> Any:
            return broker_account_factory()

        return broker_account_flow

    def _load_prefect(self) -> Any | None:
        try:
            import prefect
        except Exception as exc:
            self._warning = f"prefect_unavailable: {exc}"
            return None
        return prefect


def main() -> int:
    orchestrator = WorkflowOrchestrator(enabled=True)
    status = orchestrator.status()
    logger.info("Prefect orchestrator status: {}", status)
    return 0
