from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class EventBus:
    """Optional Redpanda/Kafka event publisher with a local JSONL fallback."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        brokers: str = "",
        topic_prefix: str = "invest_advisor",
        client_id: str = "invest-advisor-bot",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.brokers = brokers.strip()
        self.topic_prefix = topic_prefix.strip() or "invest_advisor"
        self.client_id = client_id.strip() or "invest-advisor-bot"
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._published_count = 0
        self._last_publish_at: str | None = None
        self._events_path = self.root_dir / "event_bus.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "jsonl"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": bool(self.brokers),
                "backend": self._backend,
                "topic_prefix": self.topic_prefix,
                "published_count": self._published_count,
                "last_publish_at": self._last_publish_at,
                "warning": self._warning,
            }

    def publish(self, *, topic: str, key: str | None, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        event = {
            "published_at": self._utc_now().isoformat(),
            "topic": self._normalize_topic(topic),
            "key": str(key or "").strip() or None,
            "payload": dict(payload),
        }
        self._append_jsonl(event)
        self._publish_kafka(event)

    def _publish_kafka(self, event: Mapping[str, Any]) -> None:
        if not self.brokers:
            return
        try:
            producer = self._load_producer()
            producer.send(
                event["topic"],
                key=str(event.get("key") or "").encode("utf-8") if event.get("key") else None,
                value=json.dumps(dict(event.get("payload") or {}), ensure_ascii=False).encode("utf-8"),
            )
            producer.flush(timeout=5.0)
        except Exception as exc:
            logger.warning("Event bus publish failed: {}", exc)
            with self._lock:
                self._warning = f"event_bus_publish_failed: {exc}"
                if self._backend == "disabled":
                    self._backend = "jsonl"
            return
        with self._lock:
            self._backend = "kafka"
            self._warning = None
            self._published_count += 1
            self._last_publish_at = self._utc_now().isoformat()

    def _append_jsonl(self, event: Mapping[str, Any]) -> None:
        try:
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Event bus JSONL write failed: {}", exc)
            with self._lock:
                self._warning = f"event_bus_jsonl_failed: {exc}"
        else:
            with self._lock:
                if self._backend == "disabled":
                    self._backend = "jsonl"
                self._published_count += 1
                self._last_publish_at = self._utc_now().isoformat()

    def _load_producer(self) -> Any:
        from kafka import KafkaProducer

        return KafkaProducer(
            bootstrap_servers=[item.strip() for item in self.brokers.split(",") if item.strip()],
            client_id=self.client_id,
        )

    def _normalize_topic(self, topic: str) -> str:
        raw = str(topic or "").strip().replace(" ", "_").replace("-", "_")
        return f"{self.topic_prefix}.{raw or 'events'}"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
