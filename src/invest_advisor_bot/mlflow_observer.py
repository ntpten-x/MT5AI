from __future__ import annotations

import hashlib
import importlib
import json
import re
from datetime import date, datetime
from typing import Any, Mapping

from loguru import logger


class MLflowObserver:
    """Optional MLflow trace logger with no-op behavior when disabled."""

    _MAX_TEXT_LENGTH = 4_000
    _MAX_JSON_LENGTH = 24_000
    _MAX_STRING_LENGTH = 1_200
    _MAX_LIST_ITEMS = 20
    _MAX_MAPPING_ITEMS = 40
    _MAX_DEPTH = 5
    _SENSITIVE_KEYWORDS = (
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "chat_id",
        "conversation_key",
        "cookie",
        "password",
        "secret",
        "session",
        "telegram_token",
        "token",
        "user_id",
    )
    _RAW_SECRET_PATTERNS = (
        re.compile(r"(?i)\bsk-[a-z0-9_-]{12,}\b"),
        re.compile(r"(?i)\bgh[pousr]_[a-z0-9]{12,}\b"),
        re.compile(r"(?i)\b(?:xox[baprs]-[a-z0-9-]{10,})\b"),
    )

    def __init__(self, *, tracking_uri: str = "", experiment_name: str = "invest-advisor-bot") -> None:
        self.tracking_uri = tracking_uri.strip()
        self.experiment_name = experiment_name.strip() or "invest-advisor-bot"
        self._mlflow: Any | None = None
        self._enabled = False
        self._disabled_reason: str | None = "tracking_uri_not_configured"
        self._last_run_id: str | None = None
        self._last_run_kind: str | None = None
        self._last_run_name: str | None = None
        if not self.tracking_uri:
            return
        try:
            self._mlflow = importlib.import_module("mlflow")
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
            self._enabled = True
            self._disabled_reason = None
        except ImportError:
            self._disabled_reason = "mlflow_dependency_missing: install with pip install -e .[observability]"
        except Exception as exc:
            self._disabled_reason = f"mlflow_setup_failed: {exc}"
            logger.warning("MLflow observer disabled: {}", exc)

    def enabled(self) -> bool:
        return self._enabled and self._mlflow is not None

    def log_recommendation(
        self,
        *,
        service_name: str,
        question: str | None,
        conversation_key: str | None,
        model: str | None,
        fallback_used: bool,
        payload: Mapping[str, Any] | None,
        response_text: str,
        source_coverage: Mapping[str, Any] | None,
        artifact_key: str | None = None,
        response_id: str | None = None,
        data_quality: Mapping[str, Any] | None = None,
        execution_panel: Mapping[str, Any] | None = None,
        source_ranking: list[Mapping[str, Any]] | None = None,
        source_health: Mapping[str, Any] | None = None,
        champion_challenger: Mapping[str, Any] | None = None,
        factor_exposures: Mapping[str, Any] | None = None,
        thesis_invalidation: Mapping[str, Any] | None = None,
        walk_forward_eval: Mapping[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled():
            return None
        mlflow = self._mlflow
        assert mlflow is not None
        try:
            with mlflow.start_run(run_name=service_name, nested=True) as run:
                self._set_tags(
                    mlflow,
                    {
                        "service": service_name,
                        "artifact_key": artifact_key or "",
                        "conversation_key_hash": self._hash_identifier(conversation_key),
                        "model": model or "",
                        "response_id": response_id or "",
                        "fallback_used": str(bool(fallback_used)).lower(),
                    },
                )
                mlflow.log_params(
                    {
                        "has_question": int(bool((question or "").strip())),
                        "source_count": len((source_coverage or {}).get("used_sources") or []),
                    }
                )
                if isinstance(data_quality, Mapping):
                    quality_score = data_quality.get("score")
                    if isinstance(quality_score, (int, float)):
                        mlflow.log_metric("data_quality_score", float(quality_score))
                    mlflow.set_tag("data_quality_status", str(data_quality.get("status") or ""))
                    mlflow.set_tag("data_quality_blocking", str(bool(data_quality.get("blocking"))).lower())
                if isinstance(source_health, Mapping):
                    health_score = source_health.get("score")
                    freshness_pct = source_health.get("freshness_pct")
                    total_penalty = source_health.get("total_penalty")
                    if isinstance(health_score, (int, float)):
                        mlflow.log_metric("source_health_score", float(health_score))
                    if isinstance(freshness_pct, (int, float)):
                        mlflow.log_metric("source_freshness_pct", float(freshness_pct))
                    if isinstance(total_penalty, (int, float)):
                        mlflow.log_metric("source_total_penalty", float(total_penalty))
                if isinstance(factor_exposures, Mapping):
                    top_exposure_weight_pct = factor_exposures.get("top_exposure_weight_pct")
                    if isinstance(top_exposure_weight_pct, (int, float)):
                        mlflow.log_metric("factor_top_exposure_weight_pct", float(top_exposure_weight_pct))
                    top_exposure_factor = factor_exposures.get("top_exposure_factor")
                    if top_exposure_factor:
                        mlflow.set_tag("factor_top_exposure_factor", str(top_exposure_factor))
                if isinstance(thesis_invalidation, Mapping):
                    invalidation_score = thesis_invalidation.get("score")
                    if isinstance(invalidation_score, (int, float)):
                        mlflow.log_metric("thesis_invalidation_score", float(invalidation_score))
                    mlflow.set_tag(
                        "thesis_invalidation_active",
                        str(bool(thesis_invalidation.get("has_active_invalidation"))).lower(),
                    )
                if isinstance(walk_forward_eval, Mapping):
                    for metric_name in ("window_count", "avg_hit_rate_pct", "avg_return_after_cost_pct"):
                        value = walk_forward_eval.get(metric_name)
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"walk_forward_{metric_name}", float(value))
                execution_metrics = {
                    "execution_closed_postmortems": (execution_panel or {}).get("closed_postmortems"),
                    "execution_ttl_hit_rate_pct": (execution_panel or {}).get("ttl_hit_rate_pct"),
                    "execution_fast_decay_rate_pct": (execution_panel or {}).get("fast_decay_rate_pct"),
                    "execution_hold_after_expiry_rate_pct": (execution_panel or {}).get("hold_after_expiry_rate_pct"),
                    "execution_discard_after_expiry_rate_pct": (execution_panel or {}).get("discard_after_expiry_rate_pct"),
                }
                for key, value in execution_metrics.items():
                    if isinstance(value, bool):
                        mlflow.log_metric(key, int(value))
                    elif isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))
                by_alert_kind = (execution_panel or {}).get("by_alert_kind") if isinstance(execution_panel, Mapping) else None
                if isinstance(by_alert_kind, list):
                    for item in by_alert_kind[:8]:
                        if not isinstance(item, Mapping):
                            continue
                        kind = str(item.get("alert_kind") or "").strip()
                        if not kind:
                            continue
                        safe_kind = kind.replace("-", "_")
                        best_ttl_bucket = str(item.get("best_ttl_bucket") or "").strip()
                        if best_ttl_bucket:
                            mlflow.set_tag(f"execution_best_ttl_bucket__{safe_kind}", best_ttl_bucket)
                        for metric_name in (
                            "closed_postmortems",
                            "ttl_hit_rate_pct",
                            "fast_decay_rate_pct",
                            "hold_after_expiry_rate_pct",
                            "discard_after_expiry_rate_pct",
                            "best_ttl_score",
                            "best_ttl_sample_count",
                        ):
                            value = item.get(metric_name)
                            if isinstance(value, bool):
                                mlflow.log_metric(f"execution_{metric_name}__{safe_kind}", int(value))
                            elif isinstance(value, (int, float)):
                                mlflow.log_metric(f"execution_{metric_name}__{safe_kind}", float(value))
                if question:
                    mlflow.log_text(self._sanitize_text(question, limit=self._MAX_TEXT_LENGTH), "question.txt")
                mlflow.log_text(self._sanitize_text(response_text or "", limit=self._MAX_TEXT_LENGTH), "response.txt")
                self._log_json_artifact(mlflow, "payload.json", payload or {})
                self._log_json_artifact(mlflow, "source_coverage.json", source_coverage or {})
                self._log_json_artifact(mlflow, "data_quality.json", data_quality or {})
                self._log_json_artifact(mlflow, "execution_panel.json", execution_panel or {})
                self._log_json_artifact(mlflow, "execution_by_alert_kind.json", list(by_alert_kind or []))
                source_ttl_heatmap = (execution_panel or {}).get("source_ttl_heatmap") if isinstance(execution_panel, Mapping) else None
                self._log_json_artifact(mlflow, "source_ttl_heatmap.json", list(source_ttl_heatmap or []))
                self._log_json_artifact(mlflow, "source_ranking.json", list(source_ranking or []))
                self._log_json_artifact(mlflow, "source_health.json", source_health or {})
                self._log_json_artifact(mlflow, "factor_exposures.json", factor_exposures or {})
                self._log_json_artifact(mlflow, "champion_challenger.json", champion_challenger or {})
                self._log_json_artifact(mlflow, "thesis_invalidation.json", thesis_invalidation or {})
                self._log_json_artifact(mlflow, "walk_forward_eval.json", walk_forward_eval or {})
                champion = champion_challenger.get("champion") if isinstance(champion_challenger, Mapping) else None
                challenger = champion_challenger.get("challenger") if isinstance(champion_challenger, Mapping) else None
                runner = champion_challenger.get("runner") if isinstance(champion_challenger, Mapping) else None
                if isinstance(champion, Mapping) and isinstance(champion.get("score"), (int, float)):
                    mlflow.log_metric("champion_score", float(champion.get("score")))
                if isinstance(challenger, Mapping) and isinstance(challenger.get("score"), (int, float)):
                    mlflow.log_metric("challenger_score", float(challenger.get("score")))
                delta = champion_challenger.get("delta_vs_baseline") if isinstance(champion_challenger, Mapping) else None
                if isinstance(delta, (int, float)):
                    mlflow.log_metric("champion_delta_vs_baseline", float(delta))
                if isinstance(runner, Mapping):
                    policies = runner.get("policies")
                    if isinstance(policies, list):
                        for policy in policies[:5]:
                            if not isinstance(policy, Mapping):
                                continue
                            name = str(policy.get("name") or "").strip().replace("-", "_")
                            if not name:
                                continue
                            score = policy.get("score")
                            if isinstance(score, (int, float)):
                                mlflow.log_metric(f"runner_policy_score__{name}", float(score))
                    winner = runner.get("winner")
                    if winner:
                        mlflow.set_tag("runner_winner", str(winner))
                run_id = self._extract_run_id(run)
                self._remember_run(run_id=run_id, run_kind="recommendation", run_name=service_name)
                return run_id
        except Exception as exc:
            logger.warning("MLflow recommendation log failed: {}", exc)
        return None

    def log_evaluation(
        self,
        *,
        name: str,
        metrics: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled():
            return None
        mlflow = self._mlflow
        assert mlflow is not None
        try:
            with mlflow.start_run(run_name=name, nested=True) as run:
                self._set_tags(mlflow, tags or {})
                for key, value in (metrics or {}).items():
                    if isinstance(value, bool):
                        mlflow.log_metric(str(key), int(value))
                    elif isinstance(value, (int, float)):
                        mlflow.log_metric(str(key), float(value))
                for key, value in (artifacts or {}).items():
                    self._log_json_artifact(mlflow, f"{key}.json", value)
                run_id = self._extract_run_id(run)
                self._remember_run(run_id=run_id, run_kind="evaluation", run_name=name)
                return run_id
        except Exception as exc:
            logger.warning("MLflow evaluation log failed: {}", exc)
        return None

    def status(self) -> dict[str, Any]:
        warning: str | None = None
        if self.tracking_uri and not self.enabled():
            warning = self._disabled_reason or "mlflow disabled"
        return {
            "enabled": self.enabled(),
            "tracking_configured": bool(self.tracking_uri),
            "experiment_name": self.experiment_name,
            "disabled_reason": self._disabled_reason,
            "warning": warning,
            "last_run_id": self._last_run_id,
            "last_run_kind": self._last_run_kind,
            "last_run_name": self._last_run_name,
        }

    def _set_tags(self, mlflow: Any, tags: Mapping[str, Any]) -> None:
        sanitized = self._sanitize_tags(tags)
        if sanitized:
            mlflow.set_tags(sanitized)

    def _log_json_artifact(self, mlflow: Any, path: str, value: Any) -> None:
        sanitized = self._sanitize_value(value)
        rendered = json.dumps(sanitized, ensure_ascii=False, indent=2, default=self._json_default)
        if len(rendered) > self._MAX_JSON_LENGTH:
            rendered = json.dumps(
                {
                    "truncated": True,
                    "original_length": len(rendered),
                    "preview": self._sanitize_text(rendered, limit=self._MAX_JSON_LENGTH - 128),
                },
                ensure_ascii=False,
                indent=2,
            )
        mlflow.log_text(rendered, path)

    @classmethod
    def _sanitize_tags(cls, tags: Mapping[str, Any]) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for key, value in tags.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            sanitized_value = cls._sanitize_value(value, parent_key=normalized_key)
            if sanitized_value is None:
                continue
            if isinstance(sanitized_value, (dict, list)):
                rendered = json.dumps(sanitized_value, ensure_ascii=False, default=cls._json_default)
            else:
                rendered = str(sanitized_value)
            rendered = cls._sanitize_text(rendered, limit=240).strip()
            if rendered:
                sanitized[normalized_key] = rendered
        return sanitized

    @classmethod
    def _sanitize_value(cls, value: Any, *, parent_key: str = "", depth: int = 0) -> Any:
        if depth >= cls._MAX_DEPTH:
            return "[truncated-depth]"
        if cls._is_sensitive_key(parent_key):
            return "[redacted]"
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            items = list(value.items())
            for raw_key, raw_item in items[: cls._MAX_MAPPING_ITEMS]:
                normalized_key = str(raw_key).strip()
                if not normalized_key:
                    continue
                result[normalized_key] = cls._sanitize_value(raw_item, parent_key=normalized_key, depth=depth + 1)
            if len(items) > cls._MAX_MAPPING_ITEMS:
                result["__truncated_keys__"] = len(items) - cls._MAX_MAPPING_ITEMS
            return result
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            sanitized_items = [
                cls._sanitize_value(item, parent_key=parent_key, depth=depth + 1)
                for item in items[: cls._MAX_LIST_ITEMS]
            ]
            if len(items) > cls._MAX_LIST_ITEMS:
                sanitized_items.append(f"... (+{len(items) - cls._MAX_LIST_ITEMS} more items)")
            return sanitized_items
        if isinstance(value, str):
            return cls._sanitize_text(value, limit=cls._MAX_STRING_LENGTH)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return cls._sanitize_text(str(value), limit=cls._MAX_STRING_LENGTH)

    @classmethod
    def _sanitize_text(cls, value: str, *, limit: int) -> str:
        redacted = cls._redact_inline_secrets(str(value or ""))
        if len(redacted) <= limit:
            return redacted
        return f"{redacted[: max(0, limit - 16)].rstrip()}... [truncated]"

    @classmethod
    def _redact_inline_secrets(cls, value: str) -> str:
        redacted = re.sub(
            r"(?i)\b(api[_-]?key|authorization|bearer|password|secret|token)\s*[:=]\s*([^\s,;]+)",
            r"\1=[redacted]",
            value,
        )
        redacted = re.sub(r"(?i)\bbearer\s+[a-z0-9._\-]+\b", "Bearer [redacted]", redacted)
        for pattern in cls._RAW_SECRET_PATTERNS:
            redacted = pattern.sub("[redacted]", redacted)
        return redacted

    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        normalized = str(key or "").strip().casefold()
        if "hash" in normalized:
            return False
        return any(keyword in normalized for keyword in cls._SENSITIVE_KEYWORDS)

    @staticmethod
    def _hash_identifier(value: str | None) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_run_id(run: Any) -> str | None:
        info = getattr(run, "info", None)
        run_id = getattr(info, "run_id", None)
        if run_id is None:
            return None
        return str(run_id).strip() or None

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return str(value)

    def _remember_run(self, *, run_id: str | None, run_kind: str, run_name: str) -> None:
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            return
        self._last_run_id = normalized_run_id
        self._last_run_kind = run_kind
        self._last_run_name = run_name
