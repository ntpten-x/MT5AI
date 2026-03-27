from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from invest_advisor_bot.analytics_warehouse import AnalyticsWarehouse


class DbtSemanticLayer:
    """Scaffold a dbt project/semantic layer around analytics warehouse tables."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        project_name: str = "invest_advisor_bot_semantic",
        target_schema: str = "analytics",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.project_name = project_name.strip() or "invest_advisor_bot_semantic"
        self.target_schema = target_schema.strip() or "analytics"
        self._lock = RLock()
        self._last_synced_at: str | None = None
        self._warning: str | None = None
        self._metric_names = (
            "recommendation_count",
            "fallback_rate",
            "source_health_score_avg",
            "evaluation_return_after_cost_avg",
            "alert_count",
        )
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "project_name": self.project_name,
            "target_schema": self.target_schema,
            "root_dir": str(self.root_dir),
            "metric_names": list(self._metric_names),
            "last_synced_at": self._last_synced_at,
            "warning": self._warning,
        }

    def sync(self, *, warehouse: AnalyticsWarehouse | None = None) -> None:
        if not self.enabled:
            return
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            (self.root_dir / "models" / "marts").mkdir(parents=True, exist_ok=True)
            (self.root_dir / "models" / "staging").mkdir(parents=True, exist_ok=True)
            (self.root_dir / "dbt_project.yml").write_text(self._render_project_yml(), encoding="utf-8")
            (self.root_dir / "models" / "staging" / "sources.yml").write_text(
                self._render_sources_yml(warehouse=warehouse),
                encoding="utf-8",
            )
            (self.root_dir / "models" / "marts" / "semantic_metrics.yml").write_text(
                self._render_metrics_yml(),
                encoding="utf-8",
            )
            (self.root_dir / "semantic_manifest.json").write_text(
                json.dumps(
                    {
                        "project_name": self.project_name,
                        "target_schema": self.target_schema,
                        "metric_names": list(self._metric_names),
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                        "warehouse_status": warehouse.status() if warehouse is not None else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._last_synced_at = datetime.now(timezone.utc).isoformat()
        except OSError as exc:
            self._warning = f"dbt_semantic_sync_failed: {exc}"

    def _render_project_yml(self) -> str:
        return (
            f"name: {self.project_name}\n"
            "version: '1.0.0'\n"
            "config-version: 2\n\n"
            "profile: invest_advisor_bot\n\n"
            "model-paths: ['models']\n"
            "models:\n"
            f"  {self.project_name}:\n"
            "    +materialized: view\n"
        )

    def _render_sources_yml(self, *, warehouse: AnalyticsWarehouse | None) -> str:
        source_name = "clickhouse" if warehouse is not None and warehouse.status().get("configured") else "local_analytics"
        return (
            "version: 2\n\n"
            "sources:\n"
            f"  - name: {source_name}\n"
            f"    schema: {self.target_schema}\n"
            "    tables:\n"
            "      - name: recommendation_events\n"
            "      - name: evaluation_events\n"
            "      - name: runtime_snapshots\n"
        )

    def _render_metrics_yml(self) -> str:
        return (
            "version: 2\n\n"
            "metrics:\n"
            "  - name: recommendation_count\n"
            "    label: Recommendation Count\n"
            "    type: count\n"
            "    type_params:\n"
            "      measure:\n"
            "        name: artifact_key\n"
            "  - name: fallback_rate\n"
            "    label: Fallback Rate\n"
            "    type: derived\n"
            "    description: ratio of fallback recommendations over total recommendations\n"
            "  - name: source_health_score_avg\n"
            "    label: Average Source Health Score\n"
            "    type: simple\n"
            "  - name: evaluation_return_after_cost_avg\n"
            "    label: Average Evaluation Return After Cost\n"
            "    type: simple\n"
            "  - name: alert_count\n"
            "    label: Alert Count\n"
            "    type: count\n"
            "    type_params:\n"
            "      measure:\n"
            "        name: alert_key\n"
        )
