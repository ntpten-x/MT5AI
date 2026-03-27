from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from invest_advisor_bot.analysis.news_impact import NewsImpact, summarize_news_bias
from invest_advisor_bot.analysis.trend_engine import TrendAssessment


@dataclass(slots=True, frozen=True)
class RiskScoreAssessment:
    score: float
    level: str
    reasons: tuple[str, ...]


def calculate_risk_score(
    *,
    macro_context: Mapping[str, float | None],
    trends: Mapping[str, TrendAssessment],
    news_impacts: Sequence[NewsImpact],
) -> RiskScoreAssessment:
    score = 2.5
    reasons: list[str] = []

    vix = _as_float(macro_context.get("vix"))
    tnx = _as_float(macro_context.get("tnx"))
    cpi_yoy = _as_float(macro_context.get("cpi_yoy"))
    yield_spread = _as_float(macro_context.get("yield_spread_10y_2y"))
    high_yield_spread = _as_float(macro_context.get("high_yield_spread"))
    unemployment_rate = _as_float(macro_context.get("unemployment_rate"))
    payrolls_mom_k = _as_float(macro_context.get("payrolls_mom_k"))

    if vix is not None:
        if vix >= 30:
            score += 3.0
            reasons.append(f"VIX สูงมากที่ {vix:.2f}")
        elif vix >= 24:
            score += 2.0
            reasons.append(f"VIX สูงที่ {vix:.2f}")
        elif vix <= 17:
            score -= 0.5
            reasons.append(f"VIX ต่ำที่ {vix:.2f}")

    if tnx is not None and tnx >= 4.5:
        score += 1.5
        reasons.append(f"US10Y สูงที่ {tnx:.2f}")
    if cpi_yoy is not None and cpi_yoy >= 3.2:
        score += 1.0
        reasons.append(f"CPI ยังสูงที่ {cpi_yoy:.2f}%")
    if yield_spread is not None and yield_spread < 0:
        score += 0.8
        reasons.append(f"2s10s inverted ที่ {yield_spread:.2f}")
    if high_yield_spread is not None and high_yield_spread >= 4.5:
        score += 1.0
        reasons.append(f"HY spread กว้างที่ {high_yield_spread:.2f}")
    if unemployment_rate is not None and unemployment_rate >= 4.2:
        score += 0.7
        reasons.append(f"Unemployment สูงขึ้นที่ {unemployment_rate:.2f}%")
    if payrolls_mom_k is not None and payrolls_mom_k < 125:
        score += 0.5
        reasons.append(f"Payroll ชะลอที่ {payrolls_mom_k:.0f}k")

    downtrend_count = sum(1 for trend in trends.values() if trend.direction == "downtrend")
    uptrend_count = sum(1 for trend in trends.values() if trend.direction == "uptrend")
    if downtrend_count >= max(3, uptrend_count + 1):
        score += 2.0
        reasons.append("สินทรัพย์ส่วนใหญ่ยังอยู่ในขาลง")
    elif uptrend_count >= max(3, downtrend_count + 1):
        score -= 0.5
        reasons.append("สินทรัพย์หลักหลายตัวอยู่ในขาขึ้น")

    news_bias = summarize_news_bias(news_impacts)
    if news_bias["negative"] >= max(2, news_bias["positive"] + 1):
        score += 1.5
        reasons.append("ข่าวเชิงลบเด่นกว่าข่าวเชิงบวก")
    elif news_bias["positive"] >= max(2, news_bias["negative"] + 1):
        score -= 0.5
        reasons.append("ข่าวเชิงบวกเริ่มกลับมา")

    score = max(1.0, min(10.0, round(score, 1)))
    if score >= 8.0:
        level = "severe"
    elif score >= 6.0:
        level = "high"
    elif score >= 4.0:
        level = "elevated"
    else:
        level = "normal"
    return RiskScoreAssessment(score=score, level=level, reasons=tuple(reasons[:4]))


def _as_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)
