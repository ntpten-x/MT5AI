from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class BraintrustObserver:
    """Optional Braintrust-backed recommendation/evaluation logging with local fallbacks."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        api_key: str = "",
        project_name: str = "invest-advisor-bot",
        experiment_name: str = "production-evals",
        api_url: str = "https://api.braintrust.dev",
        report_every_n_events: int = 10,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.api_key = api_key.strip()
        self.project_name = project_name.strip() or "invest-advisor-bot"
        self.experiment_name = experiment_name.strip() or "production-evals"
        self.api_url = api_url.strip().rstrip("/") or "https://api.braintrust.dev"
        self.report_every_n_events = max(1, int(report_every_n_events))
        self._lock = RLock()
        self._warning: str | None = None
        self._last_sync_at: str | None = None
        self._event_count = 0
        self._recommendations_path = self.root_dir / "braintrust_recommendations.jsonl"
        self._outcomes_path = self.root_dir / "braintrust_outcomes.jsonl"
        self._sdk_logger: Any | None = None
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._sdk_logger = self._initialize_sdk_logger()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "configured": bool(self.api_key),
                "project_name": self.project_name,
                "experiment_name": self.experiment_name,
                "api_url": self.api_url,
                "warning": self._warning,
                "last_sync_at": self._last_sync_at,
                "event_count": self._event_count,
                "recommendations_path": str(self._recommendations_path),
                "outcomes_path": str(self._outcomes_path),
            }

    def log_recommendation(
        self,
        *,
        artifact_key: str,
        question: str | None,
        response_text: str,
        model: str | None,
        fallback_used: bool,
        payload: Mapping[str, Any],
        data_quality: Mapping[str, Any] | None,
        source_coverage: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(artifact_key or "").strip(),
            "kind": "recommendation",
            "question": str(question or "").strip(),
            "response_text": str(response_text or ""),
            "response_length": len(str(response_text or "")),
            "model": str(model or "").strip() or None,
            "fallback_used": bool(fallback_used),
            "source_coverage": dict(source_coverage or {}),
            "data_quality": dict(data_quality or {}),
            "payload_summary": {
                "source_health_score": self._coerce_float(
                    (payload.get("source_health") or {}).get("score")
                    if isinstance(payload.get("source_health"), Mapping)
                    else None
                ),
                "no_trade_active": bool((payload.get("no_trade_decision") or {}).get("should_abstain"))
                if isinstance(payload.get("no_trade_decision"), Mapping)
                else False,
                "macro_headline": str((payload.get("macro_intelligence") or {}).get("headline") or "").strip()
                if isinstance(payload.get("macro_intelligence"), Mapping)
                else "",
                "used_sources": list((source_coverage or {}).get("used_sources") or []),
            },
        }
        self._append_jsonl(self._recommendations_path, event)
        self._log_sdk_event(event)

    def log_outcome(
        self,
        *,
        artifact_key: str,
        outcome_label: str | None,
        return_after_cost_pct: float | None,
        detail: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        event = {
            "event_at": self._utc_now().isoformat(),
            "artifact_key": str(artifact_key or "").strip(),
            "kind": "outcome",
            "outcome_label": str(outcome_label or "").strip() or None,
            "return_after_cost_pct": self._coerce_float(return_after_cost_pct),
            "alpha_after_cost_pct": self._coerce_float(detail.get("alpha_after_cost_pct")),
            "execution_cost_bps": self._coerce_float(detail.get("execution_cost_bps")),
            "detail": dict(detail),
        }
        self._append_jsonl(self._outcomes_path, event)
        self._log_sdk_event(event)

    def _append_jsonl(self, path: Path, event: Mapping[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            warning = f"braintrust_dataset_write_failed: {exc}"
            logger.warning("Braintrust dataset write failed: {}", exc)
            with self._lock:
                self._warning = warning
            return
        with self._lock:
            self._event_count += 1

    def _initialize_sdk_logger(self) -> Any | None:
        if not self.api_key:
            return None
        try:
            from braintrust import init_logger
        except Exception as exc:
            with self._lock:
                if self._warning is None:
                    self._warning = f"braintrust_unavailable: {exc}"
            return None
        try:
            return init_logger(
                project=self.project_name,
                api_key=self.api_key,
                api_url=self.api_url,
            )
        except Exception as exc:
            with self._lock:
                self._warning = f"braintrust_init_failed: {exc}"
            return None

    def _log_sdk_event(self, event: Mapping[str, Any]) -> None:
        logger_instance = self._sdk_logger
        if logger_instance is None:
            return
        payload = dict(event)
        artifact_key = str(payload.get("artifact_key") or "").strip()
        try:
            log_fn = getattr(logger_instance, "log", None)
            if callable(log_fn):
                log_fn(
                    input={"artifact_key": artifact_key, "kind": payload.get("kind"), "question": payload.get("question")},
                    output={"outcome_label": payload.get("outcome_label"), "response_text": payload.get("response_text")},
                    metadata={"experiment": self.experiment_name, "event": payload},
                )
            else:
                insert_fn = getattr(logger_instance, "insert", None)
                if callable(insert_fn):
                    insert_fn(payload)
        except Exception as exc:
            logger.warning("Braintrust SDK logging failed: {}", exc)
            with self._lock:
                self._warning = f"braintrust_log_failed: {exc}"
            return
        with self._lock:
            self._last_sync_at = self._utc_now().isoformat()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
