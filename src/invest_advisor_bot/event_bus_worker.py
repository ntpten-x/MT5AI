from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger

from invest_advisor_bot.analytics_warehouse import AnalyticsWarehouse
from invest_advisor_bot.hot_path_cache import HotPathCache


class EventBusConsumerWorker:
    """Optional consumer that replays event-bus traffic into local sinks."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        brokers: str = "",
        topic_prefix: str = "invest_advisor",
        consumer_group: str = "invest-advisor-bot",
        batch_size: int = 100,
        poll_timeout_seconds: float = 5.0,
        analytics_warehouse: AnalyticsWarehouse | None = None,
        hot_path_cache: HotPathCache | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.brokers = brokers.strip()
        self.topic_prefix = topic_prefix.strip() or "invest_advisor"
        self.consumer_group = consumer_group.strip() or "invest-advisor-bot"
        self.batch_size = max(1, int(batch_size))
        self.poll_timeout_seconds = max(1.0, float(poll_timeout_seconds))
        self.analytics_warehouse = analytics_warehouse
        self.hot_path_cache = hot_path_cache
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._processed_count = 0
        self._last_consumed_at: str | None = None
        self._last_topic: str | None = None
        self._state_path = self.root_dir / "consumer_state.json"
        self._events_path = self.root_dir / "event_bus.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "jsonl"
            if not self._state_path.exists():
                self._save_state({"jsonl_byte_offset": 0})

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": bool(self.brokers or self._events_path.exists()),
                "backend": self._backend,
                "consumer_group": self.consumer_group,
                "batch_size": self.batch_size,
                "processed_count": self._processed_count,
                "last_consumed_at": self._last_consumed_at,
                "last_topic": self._last_topic,
                "warning": self._warning,
            }

    def process_pending(self, *, limit: int | None = None) -> dict[str, int]:
        if not self.enabled:
            return {"processed": 0}
        remaining = max(1, int(limit or self.batch_size))
        processed = 0
        if self.brokers:
            processed += self._consume_kafka(max_records=remaining)
            remaining = max(0, remaining - processed)
        if remaining:
            processed += self._consume_jsonl(max_records=remaining)
        return {"processed": processed}

    def _consume_jsonl(self, *, max_records: int) -> int:
        if not self._events_path.exists():
            return 0
        state = self._load_state()
        offset = int(state.get("jsonl_byte_offset") or 0)
        try:
            file_size = self._events_path.stat().st_size
        except OSError:
            file_size = 0
        if offset > file_size:
            offset = 0
        processed = 0
        try:
            with self._events_path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                while processed < max_records:
                    line = handle.readline()
                    if not line:
                        break
                    offset = handle.tell()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, Mapping):
                        continue
                    if self._handle_event(event):
                        processed += 1
        except OSError as exc:
            logger.warning("Event bus consumer JSONL read failed: {}", exc)
            with self._lock:
                self._warning = f"event_bus_consumer_jsonl_failed: {exc}"
            return 0
        self._save_state({"jsonl_byte_offset": offset})
        return processed

    def _consume_kafka(self, *, max_records: int) -> int:
        try:
            consumer = self._load_consumer()
            consumer.subscribe(pattern=f"^{re.escape(self.topic_prefix)}\\..+")
            polled = consumer.poll(
                timeout_ms=int(self.poll_timeout_seconds * 1000),
                max_records=max(1, int(max_records)),
            )
        except Exception as exc:
            logger.warning("Event bus consumer Kafka poll failed: {}", exc)
            with self._lock:
                self._warning = f"event_bus_consumer_kafka_failed: {exc}"
                if self._backend == "disabled":
                    self._backend = "jsonl"
            return 0
        processed = 0
        for records in polled.values():
            for record in records:
                payload: dict[str, Any] = {}
                try:
                    decoded = json.loads(record.value.decode("utf-8")) if record.value else {}
                except Exception:
                    decoded = {}
                if isinstance(decoded, Mapping):
                    payload = dict(decoded)
                event = {
                    "published_at": getattr(record, "timestamp", None),
                    "topic": record.topic,
                    "key": record.key.decode("utf-8") if record.key else None,
                    "payload": payload,
                }
                if self._handle_event(event):
                    processed += 1
        if processed:
            try:
                consumer.commit()
            except Exception:
                pass
            with self._lock:
                self._backend = "kafka"
                self._warning = None
        return processed

    def _handle_event(self, event: Mapping[str, Any]) -> bool:
        topic = str(event.get("topic") or "").strip()
        payload = event.get("payload")
        if not topic or not isinstance(payload, Mapping):
            return False
        normalized_payload = dict(payload)
        event_key = str(event.get("key") or "").strip() or None
        if event_key and not normalized_payload.get("artifact_key"):
            normalized_payload["artifact_key"] = event_key
        topic_suffix = topic.split(".", 1)[1] if topic.startswith(f"{self.topic_prefix}.") else topic
        warehouse = self.analytics_warehouse
        cache = self.hot_path_cache
        try:
            if topic_suffix == "recommendation_event":
                if warehouse is not None:
                    warehouse.record_recommendation_event(**normalized_payload)
                if cache is not None:
                    artifact_key = str(normalized_payload.get("artifact_key") or "")
                    if artifact_key:
                        cache.set_json(namespace="recommendation_replay", key=artifact_key, payload=normalized_payload, ttl_seconds=900)
                    cache.append_stream(stream="recommendation_replay", payload=normalized_payload)
            elif topic_suffix == "evaluation_event":
                if warehouse is not None:
                    warehouse.record_evaluation_event(**normalized_payload)
                if cache is not None:
                    artifact_key = str(normalized_payload.get("artifact_key") or "")
                    if artifact_key:
                        cache.set_json(namespace="evaluation_replay", key=artifact_key, payload=normalized_payload, ttl_seconds=1800)
                    cache.append_stream(stream="evaluation_replay", payload=normalized_payload)
            elif topic_suffix == "market_event":
                if warehouse is not None:
                    warehouse.record_market_event(**normalized_payload)
                if cache is not None:
                    cache.append_stream(stream="market_replay", payload=normalized_payload)
            elif topic_suffix == "runtime_snapshot" and warehouse is not None:
                warehouse.record_runtime_snapshot(normalized_payload)
            else:
                if cache is not None:
                    cache.append_stream(stream="event_replay", payload={"topic": topic_suffix, **normalized_payload})
        except Exception as exc:
            logger.warning("Event bus consumer handle failed for {}: {}", topic, exc)
            with self._lock:
                self._warning = f"event_bus_consumer_handle_failed: {exc}"
            return False
        with self._lock:
            if self._backend == "disabled":
                self._backend = "jsonl"
            self._processed_count += 1
            self._last_consumed_at = datetime.now(timezone.utc).isoformat()
            self._last_topic = topic_suffix
            self._warning = None
        return True

    def _load_consumer(self) -> Any:
        from kafka import KafkaConsumer

        return KafkaConsumer(
            bootstrap_servers=[item.strip() for item in self.brokers.split(",") if item.strip()],
            group_id=self.consumer_group,
            enable_auto_commit=False,
            auto_offset_reset="latest",
        )

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {"jsonl_byte_offset": 0}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"jsonl_byte_offset": 0}
        return dict(payload) if isinstance(payload, Mapping) else {"jsonl_byte_offset": 0}

    def _save_state(self, payload: Mapping[str, Any]) -> None:
        try:
            self._state_path.write_text(json.dumps(dict(payload), ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            with self._lock:
                self._warning = f"event_bus_consumer_state_failed: {exc}"
