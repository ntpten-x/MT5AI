from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import httpx

from invest_advisor_bot.analytics_warehouse import AnalyticsWarehouse


class SemanticAnalyst:
    """Natural-language analytics layer with optional remote HTTP routing."""

    def __init__(
        self,
        *,
        root_dir: Path,
        warehouse: AnalyticsWarehouse | None = None,
        enabled: bool = False,
        api_url: str = "",
        api_key: str = "",
        model_name: str = "local-heuristic",
        timeout_seconds: float = 12.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.warehouse = warehouse
        self.enabled = bool(enabled)
        self.api_url = api_url.strip()
        self.api_key = api_key.strip()
        self.model_name = model_name.strip() or "local-heuristic"
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self._lock = RLock()
        self._warning: str | None = None
        self._backend = "disabled"
        self._last_answer_at: str | None = None
        self._question_count = 0
        self._history_path = self.root_dir / "semantic_analyst.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "local"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "configured": bool(self.api_url),
                "backend": self._backend,
                "model_name": self.model_name,
                "question_count": self._question_count,
                "last_answer_at": self._last_answer_at,
                "warning": self._warning,
            }

    async def analyze(self, question: str) -> str:
        if not self.enabled:
            return "semantic analyst ยังไม่เปิดใช้งาน"
        normalized_question = str(question or "").strip()
        if not normalized_question:
            return "โปรดระบุคำถามสำหรับ analytics analyst"
        remote_answer = await self._analyze_remote(normalized_question)
        answer = remote_answer or self._analyze_local(normalized_question)
        self._append_history(question=normalized_question, answer=answer)
        return answer

    async def _analyze_remote(self, question: str) -> str | None:
        if not self.api_url:
            return None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"question": question, "model": self.model_name}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            with self._lock:
                self._warning = f"semantic_remote_failed: {exc}"
            return None
        answer = body.get("answer") if isinstance(body, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            return None
        with self._lock:
            self._backend = "remote"
            self._warning = None
        return answer.strip()

    def _analyze_local(self, question: str) -> str:
        warehouse = self.warehouse
        if warehouse is None:
            return "semantic analyst local mode ยังไม่มี warehouse ให้ถามข้อมูล"
        normalized = question.casefold()
        if any(token in normalized for token in ("fallback", "llm", "recommendation")):
            rows = warehouse.recent_events(table="recommendation_events", limit=20)
            fallback_count = 0
            total = 0
            models: dict[str, int] = {}
            for row in rows:
                payload = row.get("payload")
                if not isinstance(payload, dict):
                    continue
                total += 1
                if payload.get("fallback_used"):
                    fallback_count += 1
                model = str(payload.get("model") or "").strip()
                if model:
                    models[model] = models.get(model, 0) + 1
            if not total:
                return "ยังไม่มี recommendation events ใน warehouse"
            top_model = max(models.items(), key=lambda item: item[1])[0] if models else "-"
            return f"ล่าสุดมี recommendation {total} รายการ, fallback {fallback_count} รายการ, model ที่พบบ่อยสุดคือ {top_model}"
        if any(token in normalized for token in ("outcome", "scorecard", "alpha", "return")):
            rows = warehouse.recent_events(table="evaluation_events", limit=20)
            outcomes = [row.get("payload") for row in rows if isinstance(row.get("payload"), dict)]
            if not outcomes:
                return "ยังไม่มี evaluation events ใน warehouse"
            returns = [
                float(item.get("adjusted_return_pct") if item.get("adjusted_return_pct") is not None else item.get("return_pct"))
                for item in outcomes
                if item.get("adjusted_return_pct") is not None or item.get("return_pct") is not None
            ]
            wins = sum(1 for item in outcomes if str(item.get("outcome_label") or "").strip().casefold() in {"win", "beat", "positive"})
            avg_return = round(sum(returns) / len(returns) * 100.0, 2) if returns else None
            return f"evaluation ล่าสุด {len(outcomes)} รายการ, win {wins}, avg adjusted return {avg_return if avg_return is not None else '-'}%"
        if any(token in normalized for token in ("market", "spread", "stream", "live")):
            rows = warehouse.recent_events(table="market_events", limit=20)
            payloads = [row.get("payload") for row in rows if isinstance(row.get("payload"), dict)]
            if not payloads:
                return "ยังไม่มี market events ใน warehouse"
            spreads = [
                float(item.get("numeric_1"))
                for item in payloads
                if item.get("numeric_1") is not None
            ]
            avg_spread = round(sum(spreads) / len(spreads), 2) if spreads else None
            symbols = [
                str((item.get("detail") or {}).get("symbol") or item.get("symbol") or "").strip()
                for item in payloads
            ]
            top_symbols = ", ".join(dict.fromkeys(symbol for symbol in symbols if symbol)) or "-"
            return f"market events ล่าสุด {len(payloads)} รายการ, avg spread {avg_spread if avg_spread is not None else '-'} bps, symbols {top_symbols}"
        if any(token in normalized for token in ("runtime", "db", "health", "job")):
            rows = warehouse.recent_events(table="runtime_snapshots", limit=10)
            payloads = [row.get("payload") for row in rows if isinstance(row.get("payload"), dict)]
            if not payloads:
                return "ยังไม่มี runtime snapshots ใน warehouse"
            latest = payloads[-1]
            snapshot = (
                latest.get("snapshot")
                if isinstance(latest.get("snapshot"), dict)
                else latest.get("runtime")
                if isinstance(latest.get("runtime"), dict)
                else {}
            )
            db_state = snapshot.get("db_state") if isinstance(snapshot, dict) else {}
            mlflow = snapshot.get("mlflow") if isinstance(snapshot, dict) else {}
            return (
                f"runtime ล่าสุด backend={db_state.get('backend') or '-'} healthy={db_state.get('healthy')} "
                f"mlflow_enabled={mlflow.get('enabled')} warning={mlflow.get('warning') or '-'}"
            )
        tables = ", ".join(["recommendation_events", "evaluation_events", "market_events", "runtime_snapshots"])
        return f"ยังตีความคำถามนี้เชิงโครงสร้างไม่ได้ ลองถามเรื่อง fallback, outcome, market stream, หรือ runtime health โดยอ้างอิงตาราง {tables}"

    def _append_history(self, *, question: str, answer: str) -> None:
        record = {
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
        }
        try:
            with self._history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            with self._lock:
                self._warning = f"semantic_history_failed: {exc}"
        else:
            with self._lock:
                self._question_count += 1
                self._last_answer_at = datetime.now(timezone.utc).isoformat()
                if self._backend == "disabled":
                    self._backend = "local"
