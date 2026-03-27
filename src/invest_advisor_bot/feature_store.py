from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class FeatureStoreBridge:
    """Local feature snapshot store with optional Feast repo scaffolding."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        feast_enabled: bool = False,
        project_name: str = "invest_advisor_bot",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.feast_enabled = bool(feast_enabled)
        self.project_name = project_name.strip() or "invest_advisor_bot"
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._last_write_at: str | None = None
        self._feature_counts = {"recommendation": 0, "outcome": 0}
        self._offline_dir = self.root_dir / "offline"
        self._repo_dir = self.root_dir / "feast_repo"
        self._recommendation_path = self._offline_dir / "recommendation_features.jsonl"
        self._outcome_path = self._offline_dir / "outcome_features.jsonl"
        if self.enabled:
            self._offline_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "local"
            if self.feast_enabled:
                self._scaffold_feast_repo()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": self.feast_enabled,
                "backend": self._backend,
                "project_name": self.project_name,
                "repo_path": str(self._repo_dir),
                "warning": self._warning,
                "last_write_at": self._last_write_at,
                "feature_counts": dict(self._feature_counts),
            }

    def record_recommendation_features(
        self,
        *,
        artifact_key: str,
        question: str | None,
        model: str | None,
        payload: Mapping[str, Any],
        source_coverage: Mapping[str, Any],
        data_quality: Mapping[str, Any],
        fallback_used: bool,
        service_name: str,
    ) -> dict[str, Any]:
        row = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(artifact_key or "").strip(),
            "question_length": len(str(question or "").strip()),
            "model": str(model or "").strip() or None,
            "service_name": str(service_name or "").strip() or None,
            "fallback_used": bool(fallback_used),
            "source_count": len(source_coverage.get("used_sources") or []) if isinstance(source_coverage, Mapping) else 0,
            "source_health_score": self._coerce_float(
                (payload.get("source_health") or {}).get("score")
                if isinstance(payload.get("source_health"), Mapping)
                else None
            ),
            "source_freshness_pct": self._coerce_float(
                (payload.get("source_health") or {}).get("freshness_pct")
                if isinstance(payload.get("source_health"), Mapping)
                else None
            ),
            "data_quality_status": str(data_quality.get("status") or "").strip() or None,
            "data_quality_score": self._coerce_float(data_quality.get("score")),
            "market_confidence_score": self._coerce_float(
                (payload.get("market_confidence") or {}).get("score")
                if isinstance(payload.get("market_confidence"), Mapping)
                else None
            ),
            "macro_signal_count": len((payload.get("macro_intelligence") or {}).get("signals") or [])
            if isinstance(payload.get("macro_intelligence"), Mapping)
            else 0,
            "macro_event_count": len(payload.get("macro_event_calendar") or [])
            if isinstance(payload.get("macro_event_calendar"), list)
            else 0,
            "news_count": len(payload.get("news_headlines") or [])
            if isinstance(payload.get("news_headlines"), list)
            else 0,
            "research_count": len(payload.get("research_highlights") or [])
            if isinstance(payload.get("research_highlights"), list)
            else 0,
            "company_intelligence_count": len(payload.get("company_intelligence") or [])
            if isinstance(payload.get("company_intelligence"), list)
            else 0,
            "thesis_memory_count": len(payload.get("thesis_memory") or [])
            if isinstance(payload.get("thesis_memory"), list)
            else 0,
            "should_abstain": bool(
                (payload.get("no_trade_decision") or {}).get("should_abstain")
                if isinstance(payload.get("no_trade_decision"), Mapping)
                else False
            ),
        }
        self._append_event(bucket="recommendation", event=row)
        return row

    def record_outcome_features(
        self,
        *,
        artifact_key: str,
        outcome_label: str | None,
        adjusted_return_pct: float | None,
        detail: Mapping[str, Any],
    ) -> dict[str, Any]:
        row = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(artifact_key or "").strip(),
            "outcome_label": str(outcome_label or "").strip() or None,
            "return_after_cost_pct": self._coerce_float(adjusted_return_pct),
            "alpha_after_cost_pct": self._coerce_float(detail.get("alpha_after_cost_pct")),
            "execution_cost_bps": self._coerce_float(detail.get("execution_cost_bps")),
            "ttl_hit": bool(detail.get("ttl_hit")) if isinstance(detail.get("ttl_hit"), bool) else None,
            "signal_decay_label": str(detail.get("signal_decay_label") or "").strip() or None,
            "postmortem_action": str(detail.get("postmortem_action") or "").strip() or None,
        }
        self._append_event(bucket="outcome", event=row)
        return row

    def _append_event(self, *, bucket: str, event: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._recommendation_path if bucket == "recommendation" else self._outcome_path
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Feature store write failed for {}: {}", bucket, exc)
            with self._lock:
                self._warning = f"feature_store_write_failed: {exc}"
            return
        with self._lock:
            self._last_write_at = self._utc_now().isoformat()
            self._feature_counts[bucket] = int(self._feature_counts.get(bucket, 0)) + 1
            if self._backend == "disabled":
                self._backend = "local"

    def _scaffold_feast_repo(self) -> None:
        try:
            self._repo_dir.mkdir(parents=True, exist_ok=True)
            feature_store_yaml = (
                "project: "
                + self.project_name
                + "\n"
                + "registry: registry.db\nprovider: local\nonline_store:\n  type: sqlite\n  path: online_store.db\noffline_store:\n  type: file\n"
            )
            features_py = '''from feast import Entity, FeatureView, Field, PushSource
from feast.types import Bool, Float32, Int64, String

artifact_key = Entity(name="artifact_key", join_keys=["artifact_key"])

recommendation_push_source = PushSource(name="recommendation_push_source")
outcome_push_source = PushSource(name="outcome_push_source")

recommendation_features = FeatureView(
    name="recommendation_features",
    entities=[artifact_key],
    ttl=None,
    schema=[
        Field(name="model", dtype=String),
        Field(name="service_name", dtype=String),
        Field(name="fallback_used", dtype=Bool),
        Field(name="source_count", dtype=Int64),
        Field(name="source_health_score", dtype=Float32),
        Field(name="source_freshness_pct", dtype=Float32),
        Field(name="data_quality_status", dtype=String),
        Field(name="data_quality_score", dtype=Float32),
        Field(name="market_confidence_score", dtype=Float32),
        Field(name="macro_signal_count", dtype=Int64),
        Field(name="macro_event_count", dtype=Int64),
        Field(name="news_count", dtype=Int64),
        Field(name="research_count", dtype=Int64),
        Field(name="company_intelligence_count", dtype=Int64),
        Field(name="thesis_memory_count", dtype=Int64),
        Field(name="should_abstain", dtype=Bool),
    ],
    source=recommendation_push_source,
)

outcome_features = FeatureView(
    name="outcome_features",
    entities=[artifact_key],
    ttl=None,
    schema=[
        Field(name="outcome_label", dtype=String),
        Field(name="return_after_cost_pct", dtype=Float32),
        Field(name="alpha_after_cost_pct", dtype=Float32),
        Field(name="execution_cost_bps", dtype=Float32),
        Field(name="ttl_hit", dtype=Bool),
        Field(name="signal_decay_label", dtype=String),
        Field(name="postmortem_action", dtype=String),
    ],
    source=outcome_push_source,
)
'''
            (self._repo_dir / "feature_store.yaml").write_text(feature_store_yaml, encoding="utf-8")
            (self._repo_dir / "features.py").write_text(features_py, encoding="utf-8")
            with self._lock:
                self._backend = "local+feast"
                self._warning = None
        except OSError as exc:
            logger.warning("Feast repo scaffold failed: {}", exc)
            with self._lock:
                self._warning = f"feast_scaffold_failed: {exc}"
                if self._backend == "disabled":
                    self._backend = "local"

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
