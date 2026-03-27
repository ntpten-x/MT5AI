from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from loguru import logger


class HotPathCache:
    """Optional Redis-backed hot cache and stream with local in-memory fallbacks."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        redis_url: str = "",
        stream_prefix: str = "invest_advisor",
        max_local_stream_events: int = 200,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.redis_url = redis_url.strip()
        self.stream_prefix = stream_prefix.strip() or "invest_advisor"
        self.max_local_stream_events = max(10, int(max_local_stream_events))
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._cache: dict[str, tuple[str, datetime | None]] = {}
        self._streams: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_local_stream_events)
        )
        self._last_write_at: str | None = None
        self._stream_event_count = 0
        self._stream_path = self.root_dir / "hot_path_streams.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "memory"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": bool(self.redis_url),
                "backend": self._backend,
                "stream_prefix": self.stream_prefix,
                "cache_keys": len(self._cache),
                "stream_event_count": self._stream_event_count,
                "last_write_at": self._last_write_at,
                "warning": self._warning,
            }

    def set_json(self, *, namespace: str, key: str, payload: Mapping[str, Any], ttl_seconds: int = 300) -> None:
        if not self.enabled:
            return
        cache_key = self._cache_key(namespace=namespace, key=key)
        encoded = json.dumps(dict(payload), ensure_ascii=False)
        expires_at = self._utc_now() + timedelta(seconds=max(1, int(ttl_seconds)))
        with self._lock:
            self._cache[cache_key] = (encoded, expires_at)
        self._write_local_stream({"kind": "cache_set", "key": cache_key, "payload": dict(payload)})
        self._write_redis_key(cache_key=cache_key, encoded=encoded, ttl_seconds=ttl_seconds)

    def get_json(self, *, namespace: str, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        cache_key = self._cache_key(namespace=namespace, key=key)
        redis_payload = self._read_redis_key(cache_key)
        if redis_payload is not None:
            return redis_payload
        with self._lock:
            item = self._cache.get(cache_key)
            if item is None:
                return None
            encoded, expires_at = item
            if expires_at is not None and expires_at <= self._utc_now():
                self._cache.pop(cache_key, None)
                return None
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError:
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    def append_stream(self, *, stream: str, payload: Mapping[str, Any], maxlen: int | None = None) -> None:
        if not self.enabled:
            return
        stream_name = self._stream_name(stream)
        event = {
            "stream": stream_name,
            "captured_at": self._utc_now().isoformat(),
            "payload": dict(payload),
        }
        with self._lock:
            deque_ref = self._streams.setdefault(
                stream_name,
                deque(maxlen=max(10, int(maxlen or self.max_local_stream_events))),
            )
            deque_ref.append(dict(event))
            self._stream_event_count += 1
            self._last_write_at = self._utc_now().isoformat()
        self._write_local_stream(event)
        self._write_redis_stream(stream_name=stream_name, payload=payload, maxlen=maxlen)

    def recent_stream(self, *, stream: str, limit: int = 20) -> list[dict[str, Any]]:
        stream_name = self._stream_name(stream)
        redis_rows = self._read_redis_stream(stream_name=stream_name, limit=limit)
        if redis_rows:
            return redis_rows
        with self._lock:
            rows = list(self._streams.get(stream_name) or [])
        return rows[-max(1, int(limit)) :]

    def _write_local_stream(self, event: Mapping[str, Any]) -> None:
        try:
            with self._stream_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Hot-path cache local stream write failed: {}", exc)
            with self._lock:
                self._warning = f"hot_cache_local_write_failed: {exc}"

    def _write_redis_key(self, *, cache_key: str, encoded: str, ttl_seconds: int) -> None:
        if not self.redis_url:
            return
        try:
            client = self._load_redis()
            client.setex(cache_key, max(1, int(ttl_seconds)), encoded)
        except Exception as exc:
            logger.warning("Redis cache set failed: {}", exc)
            with self._lock:
                self._warning = f"redis_set_failed: {exc}"
        else:
            with self._lock:
                self._backend = "redis"
                self._warning = None

    def _read_redis_key(self, cache_key: str) -> dict[str, Any] | None:
        if not self.redis_url:
            return None
        try:
            client = self._load_redis()
            value = client.get(cache_key)
        except Exception as exc:
            logger.warning("Redis cache read failed: {}", exc)
            with self._lock:
                self._warning = f"redis_get_failed: {exc}"
            return None
        if value is None:
            return None
        try:
            payload = json.loads(value)
        except Exception:
            return None
        with self._lock:
            self._backend = "redis"
            self._warning = None
        return dict(payload) if isinstance(payload, Mapping) else None

    def _write_redis_stream(self, *, stream_name: str, payload: Mapping[str, Any], maxlen: int | None) -> None:
        if not self.redis_url:
            return
        try:
            client = self._load_redis()
            client.xadd(
                stream_name,
                {"payload": json.dumps(dict(payload), ensure_ascii=False)},
                maxlen=max(10, int(maxlen or self.max_local_stream_events)),
                approximate=True,
            )
        except Exception as exc:
            logger.warning("Redis stream append failed: {}", exc)
            with self._lock:
                self._warning = f"redis_stream_failed: {exc}"
        else:
            with self._lock:
                self._backend = "redis"
                self._warning = None

    def _read_redis_stream(self, *, stream_name: str, limit: int) -> list[dict[str, Any]]:
        if not self.redis_url:
            return []
        try:
            client = self._load_redis()
            rows = client.xrevrange(stream_name, count=max(1, int(limit)))
        except Exception as exc:
            logger.warning("Redis stream read failed: {}", exc)
            with self._lock:
                self._warning = f"redis_stream_read_failed: {exc}"
            return []
        payloads: list[dict[str, Any]] = []
        for item_id, values in rows:
            raw_payload = values.get("payload")
            try:
                decoded = json.loads(raw_payload)
            except Exception:
                decoded = {}
            payloads.append(
                {
                    "stream": stream_name,
                    "id": item_id,
                    "payload": decoded if isinstance(decoded, Mapping) else {},
                }
            )
        with self._lock:
            if payloads:
                self._backend = "redis"
                self._warning = None
        payloads.reverse()
        return payloads

    def _load_redis(self) -> Any:
        import redis

        return redis.Redis.from_url(self.redis_url, decode_responses=True)

    def _cache_key(self, *, namespace: str, key: str) -> str:
        normalized_namespace = str(namespace or "").strip().replace(" ", "_")
        normalized_key = str(key or "").strip().replace(" ", "_")
        return f"{self.stream_prefix}:cache:{normalized_namespace}:{normalized_key}"

    def _stream_name(self, stream: str) -> str:
        normalized_stream = str(stream or "").strip().replace(" ", "_").replace("-", "_")
        return f"{self.stream_prefix}:stream:{normalized_stream or 'events'}"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
