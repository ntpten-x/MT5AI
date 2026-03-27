from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import pandas as pd
from loguru import logger


class EvidentlyObserver:
    """Optional local Evidently integration for LLM monitoring datasets and reports."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = True,
        report_every_n_events: int = 10,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.report_every_n_events = max(1, int(report_every_n_events))
        self._lock = RLock()
        self._warning: str | None = None
        self._last_report_at: str | None = None
        self._last_report_path: str | None = None
        self._event_count = 0
        self._dataset_path = self.root_dir / "llm_evaluations.jsonl"
        self._outcomes_path = self.root_dir / "llm_outcomes.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "dataset_path": str(self._dataset_path),
                "outcomes_path": str(self._outcomes_path),
                "warning": self._warning,
                "last_report_at": self._last_report_at,
                "last_report_path": self._last_report_path,
                "event_count": self._event_count,
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
            "artifact_key": artifact_key,
            "question": str(question or "").strip(),
            "response_text": str(response_text or ""),
            "response_length": len(str(response_text or "")),
            "model": str(model or "").strip() or None,
            "fallback_used": bool(fallback_used),
            "source_health_score": self._coerce_float((payload.get("source_health") or {}).get("score") if isinstance(payload.get("source_health"), Mapping) else None),
            "data_quality_status": str((data_quality or {}).get("status") or "").strip() or None,
            "data_quality_score": self._coerce_float((data_quality or {}).get("score")),
            "no_trade_active": bool((payload.get("no_trade_decision") or {}).get("should_abstain")) if isinstance(payload.get("no_trade_decision"), Mapping) else False,
            "macro_headline": str((payload.get("macro_intelligence") or {}).get("headline") or "").strip() if isinstance(payload.get("macro_intelligence"), Mapping) else "",
        }
        self._append_jsonl(self._dataset_path, event)
        with self._lock:
            self._event_count += 1
            should_render = self._event_count % self.report_every_n_events == 0
        if should_render:
            self._render_report()

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
            "artifact_key": artifact_key,
            "outcome_label": str(outcome_label or "").strip() or None,
            "return_after_cost_pct": self._coerce_float(return_after_cost_pct),
            "alpha_after_cost_pct": self._coerce_float(detail.get("alpha_after_cost_pct")),
            "execution_cost_bps": self._coerce_float(detail.get("execution_cost_bps")),
        }
        self._append_jsonl(self._outcomes_path, event)

    def _append_jsonl(self, path: Path, event: Mapping[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            warning = f"evidently_dataset_write_failed: {exc}"
            logger.warning("Evidently dataset write failed: {}", exc)
            with self._lock:
                self._warning = warning

    def _render_report(self) -> None:
        evidently = self._load_evidently()
        if evidently is None:
            return
        try:
            records = [json.loads(line) for line in self._dataset_path.read_text(encoding="utf-8").splitlines()[-200:] if line.strip()]
        except OSError as exc:
            warning = f"evidently_dataset_read_failed: {exc}"
            logger.warning("Evidently dataset read failed: {}", exc)
            with self._lock:
                self._warning = warning
            return
        if not records:
            return
        frame = pd.DataFrame(records)
        try:
            dataset = evidently["Dataset"].from_pandas(frame)
            report = evidently["Report"]([evidently["TextEvals"]()])
            result = report.run(dataset)
            report_path = self.root_dir / "llm_evaluation_report.html"
            json_path = self.root_dir / "llm_evaluation_report.json"
            result.save_html(str(report_path))
            result.save_json(str(json_path))
        except Exception as exc:
            warning = f"evidently_report_failed: {exc}"
            logger.warning("Evidently report generation failed: {}", exc)
            with self._lock:
                self._warning = warning
            return
        with self._lock:
            self._last_report_at = self._utc_now().isoformat()
            self._last_report_path = str(report_path)

    def _load_evidently(self) -> dict[str, Any] | None:
        try:
            from evidently import Dataset, Report
            from evidently.presets import TextEvals
        except Exception as exc:
            with self._lock:
                if self._warning is None:
                    self._warning = f"evidently_unavailable: {exc}"
            return None
        return {
            "Dataset": Dataset,
            "Report": Report,
            "TextEvals": TextEvals,
        }

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
