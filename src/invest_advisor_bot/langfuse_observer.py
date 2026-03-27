from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class LangfuseObserver:
    """Optional Langfuse-backed tracing with JSONL fallback."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.public_key = public_key.strip()
        self.secret_key = secret_key.strip()
        self.host = host.strip().rstrip("/") or "https://cloud.langfuse.com"
        self._lock = RLock()
        self._warning: str | None = None
        self._last_sync_at: str | None = None
        self._event_count = 0
        self._client: Any | None = None
        self._recommendations_path = self.root_dir / "langfuse_recommendations.jsonl"
        self._outcomes_path = self.root_dir / "langfuse_outcomes.jsonl"
        self._reviews_path = self.root_dir / "langfuse_reviews.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._client = self._initialize_client()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "configured": bool(self.public_key and self.secret_key),
                "host": self.host,
                "warning": self._warning,
                "last_sync_at": self._last_sync_at,
                "event_count": self._event_count,
                "recommendations_path": str(self._recommendations_path),
                "outcomes_path": str(self._outcomes_path),
                "reviews_path": str(self._reviews_path),
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
    ) -> None:
        if not self.enabled:
            return
        event = {
            "event_at": self._utc_now().isoformat(),
            "kind": "recommendation",
            "artifact_key": artifact_key,
            "question": question,
            "response_text": response_text,
            "model": model,
            "fallback_used": fallback_used,
            "source_health": dict(payload.get("source_health") or {}) if isinstance(payload.get("source_health"), Mapping) else {},
            "market_confidence": dict(payload.get("market_confidence") or {}) if isinstance(payload.get("market_confidence"), Mapping) else {},
            "data_quality": dict(data_quality or {}),
        }
        self._append_jsonl(self._recommendations_path, event)
        self._log_trace_event(name="recommendation", event=event)

    def log_outcome(
        self,
        *,
        artifact_key: str,
        outcome_label: str | None,
        return_after_cost_pct: float | None,
        detail: Mapping[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "event_at": self._utc_now().isoformat(),
            "kind": "outcome",
            "artifact_key": artifact_key,
            "outcome_label": outcome_label,
            "return_after_cost_pct": return_after_cost_pct,
            "detail": dict(detail or {}),
        }
        self._append_jsonl(self._outcomes_path, event)
        self._log_trace_event(name="outcome", event=event)

    def log_human_review(
        self,
        *,
        review_id: str,
        artifact_key: str,
        decision: str,
        score: float | None,
        note: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "event_at": self._utc_now().isoformat(),
            "kind": "human_review",
            "review_id": review_id,
            "artifact_key": artifact_key,
            "decision": decision,
            "score": score,
            "note": note,
            "metadata": dict(metadata or {}),
        }
        self._append_jsonl(self._reviews_path, event)
        self._log_trace_event(name="human_review", event=event)

    def _append_jsonl(self, path: Path, event: Mapping[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            with self._lock:
                self._warning = f"langfuse_write_failed: {exc}"
            logger.warning("Langfuse fallback write failed: {}", exc)
            return
        with self._lock:
            self._event_count += 1

    def _initialize_client(self) -> Any | None:
        if not self.public_key or not self.secret_key:
            return None
        try:
            from langfuse import Langfuse
        except Exception as exc:
            with self._lock:
                self._warning = f"langfuse_unavailable: {exc}"
            return None
        try:
            return Langfuse(public_key=self.public_key, secret_key=self.secret_key, host=self.host)
        except Exception as exc:
            with self._lock:
                self._warning = f"langfuse_init_failed: {exc}"
            return None

    def _log_trace_event(self, *, name: str, event: Mapping[str, Any]) -> None:
        client = self._client
        if client is None:
            return
        try:
            trace = client.trace(
                name=name,
                id=str(event.get("artifact_key") or event.get("review_id") or ""),
                input={"question": event.get("question"), "kind": event.get("kind")},
                output={"response_text": event.get("response_text"), "outcome_label": event.get("outcome_label")},
                metadata=dict(event),
            )
            flush = getattr(trace, "flush", None)
            if callable(flush):
                flush()
        except Exception as exc:
            with self._lock:
                self._warning = f"langfuse_log_failed: {exc}"
            logger.warning("Langfuse SDK logging failed: {}", exc)
            return
        with self._lock:
            self._last_sync_at = self._utc_now().isoformat()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
