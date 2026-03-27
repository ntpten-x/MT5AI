from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from loguru import logger


@dataclass(slots=True, frozen=True)
class DataQualityIssue:
    severity: str
    check: str
    message: str


@dataclass(slots=True, frozen=True)
class DataQualityReport:
    status: str
    blocking: bool
    score: float
    issues: tuple[DataQualityIssue, ...]
    checks: dict[str, Any]
    gx_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blocking": self.blocking,
            "score": self.score,
            "issues": [
                {
                    "severity": issue.severity,
                    "check": issue.check,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
            "checks": dict(self.checks),
            "gx": dict(self.gx_summary),
        }


class ReasoningDataQualityGate:
    """Validate context quality before it is handed to the reasoning layer."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        gx_enabled: bool = False,
        min_market_assets: int = 3,
        min_macro_sources: int = 2,
        min_news_items: int = 1,
        min_research_items: int = 0,
        gx_root_dir: Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.gx_enabled = bool(gx_enabled)
        self.min_market_assets = max(1, int(min_market_assets))
        self.min_macro_sources = max(1, int(min_macro_sources))
        self.min_news_items = max(0, int(min_news_items))
        self.min_research_items = max(0, int(min_research_items))
        self.gx_root_dir = Path(gx_root_dir) if gx_root_dir is not None else None
        self._warning: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "gx_enabled": self.gx_enabled,
            "min_market_assets": self.min_market_assets,
            "min_macro_sources": self.min_macro_sources,
            "min_news_items": self.min_news_items,
            "min_research_items": self.min_research_items,
            "warning": self._warning,
        }

    def evaluate(
        self,
        *,
        news: Sequence[Any],
        market_data: Mapping[str, Any],
        macro_context: Mapping[str, float | None] | None,
        macro_intelligence: Mapping[str, Any] | None,
        research_findings: Sequence[Any],
    ) -> DataQualityReport:
        if not self.enabled:
            return DataQualityReport(
                status="disabled",
                blocking=False,
                score=100.0,
                issues=(),
                checks={"enabled": False},
                gx_summary={"enabled": False, "executed": False, "success_percent": None, "warning": None},
            )

        issues: list[DataQualityIssue] = []
        market_quotes = [quote for quote in market_data.values() if quote is not None and float(getattr(quote, "price", 0.0) or 0.0) > 0]
        market_asset_count = len(market_quotes)
        macro_payload = dict(macro_intelligence or {})
        macro_sources = [str(item).strip() for item in (macro_payload.get("sources_used") or []) if str(item).strip()]
        required_macro_keys = ("vix", "tnx", "cpi_yoy")
        macro_context_payload = dict(macro_context or {})
        required_macro_present = sum(1 for key in required_macro_keys if self._as_float(macro_context_payload.get(key)) is not None)
        headline = str(macro_payload.get("headline") or "").strip()
        macro_signal_count = len([item for item in (macro_payload.get("signals") or []) if str(item).strip()])
        news_count = len([item for item in news if item is not None])
        research_count = len([item for item in research_findings if item is not None])

        if market_asset_count == 0:
            issues.append(DataQualityIssue("critical", "market_assets", "no valid market quotes are available"))
        elif market_asset_count < self.min_market_assets:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "market_assets",
                    f"only {market_asset_count} valid market quotes are available; expected at least {self.min_market_assets}",
                )
            )

        if required_macro_present == 0:
            issues.append(DataQualityIssue("critical", "macro_core", "core macro metrics are missing"))
        elif required_macro_present < len(required_macro_keys):
            issues.append(
                DataQualityIssue(
                    "warning",
                    "macro_core",
                    f"macro coverage is partial ({required_macro_present}/{len(required_macro_keys)} core metrics present)",
                )
            )

        if len(macro_sources) < self.min_macro_sources:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "macro_sources",
                    f"macro source count is low ({len(macro_sources)} < {self.min_macro_sources})",
                )
            )

        if not headline:
            issues.append(DataQualityIssue("warning", "macro_headline", "macro headline is missing"))
        if macro_signal_count == 0:
            issues.append(DataQualityIssue("warning", "macro_signals", "macro intelligence has no explicit signals"))

        if news_count < self.min_news_items:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "news_context",
                    f"news context is thin ({news_count} < {self.min_news_items})",
                )
            )
        if research_count < self.min_research_items:
            issues.append(
                DataQualityIssue(
                    "warning",
                    "research_context",
                    f"research context is thin ({research_count} < {self.min_research_items})",
                )
            )

        gx_summary = self._run_great_expectations(
            market_asset_count=market_asset_count,
            macro_source_count=len(macro_sources),
            required_macro_present=required_macro_present,
            macro_context=macro_context_payload,
        )
        if gx_summary.get("executed") and gx_summary.get("success_percent") is not None:
            success_percent = float(gx_summary["success_percent"])
            if success_percent < 100.0:
                issues.append(
                    DataQualityIssue(
                        "warning" if success_percent >= 70.0 else "critical",
                        "great_expectations",
                        f"Great Expectations validation success is {success_percent:.1f}%",
                    )
                )

        critical_count = sum(1 for issue in issues if issue.severity == "critical")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        blocking = critical_count > 0
        if blocking:
            status = "fail"
        elif warning_count:
            status = "warn"
        else:
            status = "pass"
        score = max(0.0, 100.0 - (critical_count * 35.0) - (warning_count * 8.0))
        return DataQualityReport(
            status=status,
            blocking=blocking,
            score=round(score, 1),
            issues=tuple(issues),
            checks={
                "market_asset_count": market_asset_count,
                "macro_source_count": len(macro_sources),
                "required_macro_present": required_macro_present,
                "macro_signal_count": macro_signal_count,
                "news_count": news_count,
                "research_count": research_count,
            },
            gx_summary=gx_summary,
        )

    def _run_great_expectations(
        self,
        *,
        market_asset_count: int,
        macro_source_count: int,
        required_macro_present: int,
        macro_context: Mapping[str, float | None],
    ) -> dict[str, Any]:
        summary = {
            "enabled": self.gx_enabled,
            "executed": False,
            "success_percent": None,
            "warning": None,
        }
        if not self.gx_enabled:
            return summary
        try:
            import great_expectations as gx
        except Exception as exc:
            warning = f"great_expectations_unavailable: {exc}"
            self._warning = warning
            summary["warning"] = warning
            return summary

        frame = pd.DataFrame(
            [
                {
                    "market_asset_count": market_asset_count,
                    "macro_source_count": macro_source_count,
                    "required_macro_present": required_macro_present,
                    "vix": self._as_float(macro_context.get("vix")),
                    "tnx": self._as_float(macro_context.get("tnx")),
                    "cpi_yoy": self._as_float(macro_context.get("cpi_yoy")),
                }
            ]
        )

        try:
            context = gx.get_context(context_root_dir=str(self.gx_root_dir)) if self.gx_root_dir else gx.get_context()
            validator = context.sources.pandas_default.read_dataframe(frame)
            results = [
                validator.expect_column_values_to_be_between("market_asset_count", min_value=0, max_value=500),
                validator.expect_column_values_to_be_between("macro_source_count", min_value=0, max_value=50),
                validator.expect_column_values_to_be_between("required_macro_present", min_value=0, max_value=3),
                validator.expect_column_values_to_be_between("vix", min_value=0, max_value=120, mostly=0.0),
                validator.expect_column_values_to_be_between("tnx", min_value=-5, max_value=20, mostly=0.0),
                validator.expect_column_values_to_be_between("cpi_yoy", min_value=-10, max_value=25, mostly=0.0),
            ]
        except Exception as exc:
            warning = f"great_expectations_validation_failed: {exc}"
            logger.warning("Great Expectations validation failed: {}", exc)
            self._warning = warning
            summary["warning"] = warning
            return summary

        success_values = [bool(getattr(result, "success", False)) for result in results]
        success_percent = (sum(1 for item in success_values if item) / len(success_values)) * 100.0 if success_values else 100.0
        summary["executed"] = True
        summary["success_percent"] = round(success_percent, 1)
        return summary

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
