from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import httpx
from loguru import logger


class ThesisVectorStore:
    """Optional external thesis memory using Qdrant with a local JSONL fallback."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        collection_name: str = "invest_advisor_thesis_memory",
        vector_size: int = 64,
        embedding_api_key: str = "",
        embedding_base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "",
        embedding_timeout_seconds: float = 12.0,
        rerank_enabled: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.qdrant_url = qdrant_url.strip()
        self.qdrant_api_key = qdrant_api_key.strip()
        self.collection_name = collection_name.strip() or "invest_advisor_thesis_memory"
        self.vector_size = max(8, int(vector_size))
        self.embedding_api_key = embedding_api_key.strip()
        self.embedding_base_url = embedding_base_url.strip().rstrip("/") or "https://api.openai.com/v1"
        self.embedding_model = embedding_model.strip()
        self.embedding_timeout_seconds = max(2.0, float(embedding_timeout_seconds))
        self.rerank_enabled = bool(rerank_enabled)
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._last_upsert_at: str | None = None
        self._last_search_at: str | None = None
        self._last_embedding_backend = "heuristic"
        self._point_count = 0
        self._http_client: httpx.Client | None = None
        self._jsonl_path = self.root_dir / "thesis_memory.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "jsonl"
            self._ensure_qdrant_collection()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": bool(self.qdrant_url),
                "backend": self._backend,
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "embedding_configured": self.embedding_available(),
                "embedding_model": self.embedding_model or None,
                "embedding_base_url": self.embedding_base_url if self.embedding_available() else None,
                "last_embedding_backend": self._last_embedding_backend,
                "rerank_enabled": self.rerank_enabled,
                "warning": self._warning,
                "last_upsert_at": self._last_upsert_at,
                "last_search_at": self._last_search_at,
                "point_count": self._point_count,
            }

    def embedding_available(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_model)

    def record_thesis(
        self,
        *,
        thesis_key: str,
        thesis_text: str,
        source_kind: str,
        conversation_key: str | None = None,
        query_text: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        confidence_score: float | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        normalized_text = str(thesis_text or "").strip()
        normalized_key = str(thesis_key or "").strip()
        if not normalized_text or not normalized_key:
            return
        embedding = self._embed_text(normalized_text)
        payload = {
            "thesis_key": normalized_key,
            "conversation_key": str(conversation_key or "").strip() or None,
            "query_text": str(query_text or "").strip() or None,
            "thesis_text": normalized_text,
            "source_kind": str(source_kind or "").strip() or "recommendation",
            "tags": [str(item).strip() for item in (tags or []) if str(item).strip()],
            "confidence_score": self._coerce_float(confidence_score),
            "detail": dict(detail or {}),
            "created_at": self._utc_now().isoformat(),
            "embedding_backend": self._last_embedding_backend,
            "embedding_model": self.embedding_model or None,
            "embedding_json": embedding,
        }
        self._append_jsonl(payload)
        self._upsert_qdrant(payload)

    def search(
        self,
        *,
        query_text: str,
        conversation_key: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return []
        query_embedding = self._embed_text(normalized_query)
        qdrant_rows = self._search_qdrant(
            query_text=normalized_query,
            query_embedding=query_embedding,
            conversation_key=conversation_key,
            limit=limit,
        )
        if qdrant_rows:
            return self._rerank_rows(query_text=normalized_query, rows=qdrant_rows, limit=limit)
        local_rows = self._search_local(
            query_text=normalized_query,
            query_embedding=query_embedding,
            conversation_key=conversation_key,
            limit=limit,
        )
        return self._rerank_rows(query_text=normalized_query, rows=local_rows, limit=limit)

    def _append_jsonl(self, payload: Mapping[str, Any]) -> None:
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(payload), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            logger.warning("Thesis vector store JSONL write failed: {}", exc)
            with self._lock:
                self._warning = f"jsonl_write_failed: {exc}"
        else:
            with self._lock:
                self._last_upsert_at = self._utc_now().isoformat()
                self._point_count += 1
                if self._backend == "disabled":
                    self._backend = "jsonl"

    def _ensure_qdrant_collection(self) -> None:
        if not self.qdrant_url:
            return
        try:
            client, models = self._load_qdrant()
            exists = False
            collection_exists = getattr(client, "collection_exists", None)
            if callable(collection_exists):
                exists = bool(collection_exists(self.collection_name))
            else:
                try:
                    client.get_collection(self.collection_name)
                    exists = True
                except Exception:
                    exists = False
            if not exists:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            with self._lock:
                self._backend = "qdrant"
                self._warning = None
        except Exception as exc:
            logger.warning("Qdrant collection setup failed: {}", exc)
            with self._lock:
                self._warning = f"qdrant_setup_failed: {exc}"
                self._backend = "jsonl"

    def _upsert_qdrant(self, payload: Mapping[str, Any]) -> None:
        if not self.qdrant_url:
            return
        if self.embedding_available() and str(payload.get("embedding_backend") or "") != "remote":
            return
        try:
            client, models = self._load_qdrant()
            point = models.PointStruct(
                id=str(payload.get("thesis_key") or ""),
                vector=list(payload.get("embedding_json") or []),
                payload={key: value for key, value in payload.items() if key != "embedding_json"},
            )
            client.upsert(collection_name=self.collection_name, points=[point])
            with self._lock:
                self._backend = "qdrant"
                self._last_upsert_at = self._utc_now().isoformat()
                self._warning = None
        except Exception as exc:
            logger.warning("Qdrant upsert failed: {}", exc)
            with self._lock:
                self._warning = f"qdrant_upsert_failed: {exc}"
                if self._backend == "disabled":
                    self._backend = "jsonl"

    def _search_qdrant(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        conversation_key: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self.qdrant_url:
            return []
        if self.embedding_available() and self._last_embedding_backend != "remote":
            return []
        try:
            client, _models = self._load_qdrant()
            hits = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=max(2, int(limit)) * 4,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed: {}", exc)
            with self._lock:
                self._warning = f"qdrant_search_failed: {exc}"
            return []
        rows: list[dict[str, Any]] = []
        for hit in hits or []:
            payload = dict(getattr(hit, "payload", {}) or {})
            stored_conversation_key = str(payload.get("conversation_key") or "").strip() or None
            if conversation_key and stored_conversation_key not in {None, conversation_key}:
                continue
            rows.append(
                {
                    "thesis_key": payload.get("thesis_key"),
                    "conversation_key": stored_conversation_key,
                    "thesis_text": payload.get("thesis_text"),
                    "source_kind": payload.get("source_kind"),
                    "query_text": payload.get("query_text"),
                    "tags": payload.get("tags") or [],
                    "confidence_score": payload.get("confidence_score"),
                    "detail": payload.get("detail") or {},
                    "created_at": payload.get("created_at"),
                    "similarity": self._coerce_float(getattr(hit, "score", None)),
                }
            )
            if len(rows) >= max(2, int(limit)) * 4:
                break
        with self._lock:
            self._last_search_at = self._utc_now().isoformat()
            if rows:
                self._backend = "qdrant"
                self._warning = None
        return rows

    def _search_local(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        conversation_key: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self._jsonl_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with self._jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, Mapping):
                        continue
                    stored_conversation_key = str(payload.get("conversation_key") or "").strip() or None
                    if conversation_key and stored_conversation_key not in {None, conversation_key}:
                        continue
                    score = self._cosine_similarity(
                        query_embedding,
                        list(payload.get("embedding_json") or []),
                    )
                    rows.append(
                        {
                            "thesis_key": payload.get("thesis_key"),
                            "conversation_key": stored_conversation_key,
                            "thesis_text": payload.get("thesis_text"),
                            "source_kind": payload.get("source_kind"),
                            "query_text": payload.get("query_text"),
                            "tags": payload.get("tags") or [],
                            "confidence_score": payload.get("confidence_score"),
                            "detail": payload.get("detail") or {},
                            "created_at": payload.get("created_at"),
                            "similarity": round(score, 4),
                        }
                    )
        except OSError as exc:
            logger.warning("Local thesis search failed: {}", exc)
            with self._lock:
                self._warning = f"jsonl_search_failed: {exc}"
            return []
        rows.sort(
            key=lambda item: (
                float(item.get("similarity") or -9999.0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        with self._lock:
            self._last_search_at = self._utc_now().isoformat()
            if self._backend == "disabled":
                self._backend = "jsonl"
        return rows[: max(2, int(limit)) * 4]

    def _load_qdrant(self) -> tuple[Any, Any]:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key or None,
            timeout=5.0,
        )
        return client, models

    def _embed_text(self, text: str) -> list[float]:
        remote_vector = self._embed_text_remote(text)
        if remote_vector is not None:
            with self._lock:
                self._last_embedding_backend = "remote"
                self._warning = None
            return remote_vector
        with self._lock:
            self._last_embedding_backend = "heuristic"
        return self._embed_text_heuristic(text, dimensions=self.vector_size)

    def _embed_text_remote(self, text: str) -> list[float] | None:
        if not self.embedding_available():
            return None
        payload: dict[str, Any] = {
            "input": str(text or ""),
            "model": self.embedding_model,
        }
        if self.vector_size > 0:
            payload["dimensions"] = self.vector_size
        try:
            response = self._get_http_client().post(
                f"{self.embedding_base_url}/embeddings",
                headers=self._embedding_headers(),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning("Remote thesis embedding failed: {}", exc)
            with self._lock:
                self._warning = f"embedding_request_failed: {exc}"
            return None
        data = body.get("data") if isinstance(body, Mapping) else None
        first = data[0] if isinstance(data, list) and data else None
        vector = first.get("embedding") if isinstance(first, Mapping) else None
        if not isinstance(vector, list):
            with self._lock:
                self._warning = "embedding_response_invalid"
            return None
        return self._normalize_vector_length(vector, dimensions=self.vector_size)

    @staticmethod
    def _embed_text_heuristic(text: str, *, dimensions: int = 64) -> list[float]:
        vector = [0.0] * max(8, int(dimensions))
        tokens = re.findall(r"[A-Za-z0-9_]+", str(text or "").casefold())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(len(vector)):
                vector[index] += (digest[index % len(digest)] / 255.0) - 0.5
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [round(value / norm, 6) for value in vector]

    def _rerank_rows(
        self,
        *,
        query_text: str,
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        if not self.rerank_enabled:
            return rows[: max(1, int(limit))]
        normalized_query = self._normalize_text(query_text)
        query_terms = self._lexical_terms(normalized_query)
        reranked: list[dict[str, Any]] = []
        for row in rows:
            semantic_score = float(self._coerce_float(row.get("similarity")) or 0.0)
            lexical_score = self._lexical_score(
                query_terms=query_terms,
                query_text=normalized_query,
                row=row,
            )
            recency_score = self._recency_score(row.get("created_at"))
            confidence_score = max(
                0.0,
                min(1.0, float(self._coerce_float(row.get("confidence_score")) or 0.0)),
            )
            rerank_score = round(
                (semantic_score * 0.68)
                + (lexical_score * 0.2)
                + (recency_score * 0.08)
                + (confidence_score * 0.04),
                4,
            )
            reranked.append(
                {
                    **row,
                    "lexical_score": round(lexical_score, 4),
                    "recency_score": round(recency_score, 4),
                    "rerank_score": rerank_score,
                }
            )
        reranked.sort(
            key=lambda item: (
                float(item.get("rerank_score") or -9999.0),
                float(item.get("similarity") or -9999.0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return reranked[: max(1, int(limit))]

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.embedding_timeout_seconds)
            return self._http_client

    def _embedding_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.embedding_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalize_vector_length(vector: list[Any], *, dimensions: int) -> list[float]:
        target = max(8, int(dimensions))
        normalized: list[float] = []
        for item in vector[:target]:
            try:
                normalized.append(float(item))
            except (TypeError, ValueError):
                normalized.append(0.0)
        if len(normalized) < target:
            normalized.extend([0.0] * (target - len(normalized)))
        return normalized

    @classmethod
    def _lexical_score(
        cls,
        *,
        query_terms: set[str],
        query_text: str,
        row: Mapping[str, Any],
    ) -> float:
        text_parts = [
            str(row.get("thesis_text") or ""),
            str(row.get("query_text") or ""),
            " ".join(str(tag).strip() for tag in (row.get("tags") or []) if str(tag).strip()),
        ]
        haystack = cls._normalize_text(" ".join(part for part in text_parts if part))
        if not haystack:
            return 0.0
        text_terms = cls._lexical_terms(haystack)
        overlap = (len(query_terms & text_terms) / max(1, len(query_terms))) if query_terms else 0.0
        phrase_bonus = 0.25 if query_text and query_text in haystack else 0.0
        fragment_hits = [
            fragment
            for fragment in re.split(r"\s+", query_text)
            if len(fragment) >= 3 and fragment in haystack
        ]
        fragment_bonus = min(0.2, len(fragment_hits) * 0.05)
        return max(0.0, min(1.0, overlap + phrase_bonus + fragment_bonus))

    @classmethod
    def _lexical_terms(cls, value: str) -> set[str]:
        english_terms = {token for token in re.findall(r"[A-Za-z0-9_]+", value.casefold()) if token}
        if english_terms:
            return english_terms
        return {
            token
            for token in re.split(r"\s+", cls._normalize_text(value))
            if len(token) >= 2
        }

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").casefold()).strip()

    @classmethod
    def _recency_score(cls, value: Any) -> float:
        timestamp = cls._parse_datetime(value)
        if timestamp is None:
            return 0.0
        age_days = abs((cls._utc_now() - timestamp).total_seconds()) / 86400.0
        return max(0.0, 1.0 - min(age_days, 30.0) / 30.0)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
