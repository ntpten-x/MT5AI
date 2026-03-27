from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4


class HumanReviewStore:
    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        low_confidence_threshold: float = 0.58,
        sample_every_n: int = 5,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.sample_every_n = max(1, int(sample_every_n))
        self._lock = RLock()
        self._enqueued_count = 0
        self._warning: str | None = None
        self._pending_path = self.root_dir / "human_review_pending.json"
        self._completed_path = self.root_dir / "human_review_completed.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "pending_count": len(self.list_pending(limit=1000)),
            "enqueued_count": self._enqueued_count,
            "warning": self._warning,
            "low_confidence_threshold": self.low_confidence_threshold,
            "sample_every_n": self.sample_every_n,
            "pending_path": str(self._pending_path),
            "completed_path": str(self._completed_path),
        }

    def should_enqueue(self, *, fallback_used: bool, confidence_score: float | None) -> bool:
        if not self.enabled:
            return False
        if fallback_used:
            return True
        if confidence_score is not None and confidence_score <= self.low_confidence_threshold:
            return True
        return ((self._enqueued_count + 1) % self.sample_every_n) == 0

    def enqueue(
        self,
        *,
        artifact_key: str,
        question: str | None,
        recommendation_text: str,
        model: str | None,
        fallback_used: bool,
        confidence_score: float | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        review_id = f"review-{uuid4().hex[:12]}"
        payload = self._load_pending()
        payload[review_id] = {
            "review_id": review_id,
            "artifact_key": artifact_key,
            "question": question,
            "recommendation_text": recommendation_text,
            "model": model,
            "fallback_used": fallback_used,
            "confidence_score": confidence_score,
            "metadata": dict(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_pending(payload)
        self._enqueued_count += 1
        return review_id

    def list_pending(self, *, limit: int = 10) -> list[dict[str, Any]]:
        payload = self._load_pending()
        rows = list(payload.values())
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows[: max(1, int(limit))]

    def complete_review(
        self,
        *,
        review_id: str,
        decision: str,
        score: float | None,
        note: str | None,
    ) -> dict[str, Any] | None:
        payload = self._load_pending()
        review = payload.pop(review_id, None)
        if review is None:
            return None
        completed = dict(review)
        completed.update(
            {
                "decision": str(decision or "").strip() or "accepted",
                "score": score,
                "note": note,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_pending(payload)
        self._append_completed(completed)
        return completed

    def _load_pending(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if not self._pending_path.exists():
                return {}
            try:
                payload = json.loads(self._pending_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._warning = f"human_review_load_failed: {exc}"
                return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): dict(value)
            for key, value in payload.items()
            if isinstance(value, Mapping)
        }

    def _save_pending(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            try:
                self._pending_path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                self._warning = f"human_review_save_failed: {exc}"

    def _append_completed(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            try:
                with self._completed_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(payload), ensure_ascii=False))
                    handle.write("\n")
            except OSError as exc:
                self._warning = f"human_review_complete_failed: {exc}"
