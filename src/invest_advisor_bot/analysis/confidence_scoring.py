from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from invest_advisor_bot.analysis.stock_screener import StockCandidate


@dataclass(slots=True, frozen=True)
class ConfidenceAssessment:
    score: float
    label: str
    rationale: tuple[str, ...]


def assess_stock_candidate_confidence(candidate: StockCandidate) -> ConfidenceAssessment:
    score = 0.45
    rationale: list[str] = []

    if candidate.stance == "buy":
        score += 0.14
        rationale.append("stance_buy")
    elif candidate.stance == "watch":
        score += 0.06
        rationale.append("stance_watch")
    else:
        score -= 0.08
        rationale.append("stance_avoid")

    score += max(-0.14, min(0.18, candidate.composite_score * 0.06))
    if candidate.trend_direction == "uptrend":
        score += 0.1
        rationale.append("trend_uptrend")
    elif candidate.trend_direction == "downtrend":
        score -= 0.12
        rationale.append("trend_downtrend")

    if candidate.rsi is not None:
        if 50 <= candidate.rsi <= 68:
            score += 0.08
            rationale.append("rsi_constructive")
        elif candidate.rsi >= 75:
            score -= 0.07
            rationale.append("rsi_overheated")
        elif candidate.rsi <= 38:
            score -= 0.06
            rationale.append("rsi_weak")

    if candidate.technical_score >= 1.4:
        score += 0.08
        rationale.append("technical_confirmed")
    elif candidate.technical_score <= -0.5:
        score -= 0.08
        rationale.append("technical_fragile")

    if candidate.growth_score >= 1.0:
        score += 0.07
        rationale.append("growth_supported")
    elif candidate.growth_score < 0:
        score -= 0.06
        rationale.append("growth_soft")

    if candidate.quality_score >= 1.0:
        score += 0.07
        rationale.append("quality_supported")
    elif candidate.quality_score < 0:
        score -= 0.06
        rationale.append("quality_soft")

    if candidate.valuation_score >= 1.0:
        score += 0.05
        rationale.append("valuation_reasonable")
    elif candidate.valuation_score < 0:
        score -= 0.08
        rationale.append("valuation_stretched")

    clipped = round(min(0.95, max(0.25, score)), 2)
    return ConfidenceAssessment(
        score=clipped,
        label=_label_for_score(clipped),
        rationale=tuple(rationale[:5]),
    )


def assess_market_recommendation_confidence(
    *,
    asset_snapshots: Sequence[Mapping[str, object]],
    macro_regime: Mapping[str, object] | None,
    news_items: Sequence[Mapping[str, object]],
    research_items: Sequence[Mapping[str, object]],
    portfolio_review: Mapping[str, object] | None,
) -> ConfidenceAssessment:
    score = 0.38
    rationale: list[str] = []

    macro_confidence = _as_float((macro_regime or {}).get("confidence"))
    if macro_confidence is not None:
        score += max(-0.05, min(0.18, (macro_confidence - 0.5) * 0.4))
        if macro_confidence >= 0.7:
            rationale.append("macro_clear")

    clear_assets = 0
    constructive_assets = 0
    for item in asset_snapshots[:6]:
        trend_score = _as_float(item.get("trend_score"))
        direction = str(item.get("trend") or "").strip().casefold()
        if trend_score is not None and abs(trend_score) >= 2.0:
            clear_assets += 1
        if direction == "uptrend" and trend_score is not None and trend_score >= 2.0:
            constructive_assets += 1
    if clear_assets:
        score += min(0.16, clear_assets * 0.03)
        rationale.append("cross_asset_trend_confirmed")
    if constructive_assets >= 2:
        score += 0.05
        rationale.append("leaders_constructive")

    news_count = len(news_items)
    research_count = len(research_items)
    if news_count >= 3:
        score += 0.05
        rationale.append("news_context_available")
    elif news_count == 0:
        score -= 0.04
        rationale.append("news_context_thin")

    if research_count >= 2:
        score += 0.05
        rationale.append("research_context_available")

    if isinstance(portfolio_review, Mapping) and portfolio_review.get("action_summary"):
        score += 0.03
        rationale.append("portfolio_alignment_checked")

    clipped = round(min(0.95, max(0.25, score)), 2)
    return ConfidenceAssessment(
        score=clipped,
        label=_label_for_score(clipped),
        rationale=tuple(rationale[:5]),
    )


def _label_for_score(score: float) -> str:
    if score >= 0.82:
        return "very_high"
    if score >= 0.7:
        return "high"
    if score >= 0.58:
        return "medium"
    return "low"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
