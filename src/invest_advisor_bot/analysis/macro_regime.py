from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

MacroRegimeName = Literal[
    "soft_landing",
    "disinflationary_growth",
    "inflation_rebound",
    "recession_risk",
    "stagflation_risk",
    "mixed_transition",
]


@dataclass(slots=True, frozen=True)
class MacroRegimeAssessment:
    regime: MacroRegimeName
    confidence: float
    growth_signal: float
    inflation_signal: float
    market_signal: float
    headline: str
    rationale: tuple[str, ...]


def assess_macro_regime(
    *,
    macro_context: Mapping[str, object] | None,
    asset_snapshots: Sequence[Mapping[str, object]],
) -> MacroRegimeAssessment:
    vix = _as_float((macro_context or {}).get("vix"))
    tnx = _as_float((macro_context or {}).get("tnx"))
    cpi_yoy = _as_float((macro_context or {}).get("cpi_yoy"))

    growth_signal = 0.0
    inflation_signal = 0.0
    market_signal = 0.0
    rationale: list[str] = []

    equity_uptrends = 0
    equity_downtrends = 0
    defensive_uptrends = 0
    for snapshot in asset_snapshots:
        asset = str(snapshot.get("asset") or "")
        trend = str(snapshot.get("trend") or "sideways")
        trend_score = _as_float(snapshot.get("trend_score")) or 0.0
        if asset in {"spy_etf", "qqq_etf", "sp500_index", "nasdaq_index", "vti_etf", "voo_etf"}:
            if trend == "uptrend":
                equity_uptrends += 1
                market_signal += min(1.2, trend_score / 4.0)
            elif trend == "downtrend":
                equity_downtrends += 1
                market_signal -= min(1.2, abs(trend_score) / 4.0)
        if asset in {"gld_etf", "iau_etf", "gold_futures", "tlt_etf"} and trend == "uptrend":
            defensive_uptrends += 1

    if vix is not None:
        if vix <= 18:
            growth_signal += 0.7
            rationale.append(f"VIX ต่ำ ({vix:.1f})")
        elif vix >= 26:
            growth_signal -= 1.0
            rationale.append(f"VIX สูง ({vix:.1f})")
    if tnx is not None:
        if tnx <= 4.1:
            growth_signal += 0.5
        elif tnx >= 4.6:
            growth_signal -= 0.6
            inflation_signal += 0.7
            rationale.append(f"US10Y สูง ({tnx:.2f})")
    if cpi_yoy is not None:
        if cpi_yoy <= 2.8:
            inflation_signal -= 0.8
            rationale.append(f"เงินเฟ้อชะลอ ({cpi_yoy:.1f}%)")
        elif cpi_yoy >= 3.4:
            inflation_signal += 1.0
            rationale.append(f"เงินเฟ้อเด้ง ({cpi_yoy:.1f}%)")

    if equity_uptrends >= max(2, equity_downtrends + 1):
        growth_signal += 0.8
        rationale.append("หุ้นหลักยังยืนแนวโน้มบวก")
    elif equity_downtrends >= max(2, equity_uptrends + 1):
        growth_signal -= 0.9
        rationale.append("หุ้นหลักเริ่มเสียโมเมนตัม")

    if defensive_uptrends >= 2:
        inflation_signal += 0.2
        market_signal -= 0.2

    regime: MacroRegimeName = "mixed_transition"
    headline = "ภาพมหภาคยังผสมกัน"
    if growth_signal >= 1.0 and inflation_signal <= 0.2 and market_signal >= 0.3:
        regime = "soft_landing"
        headline = "เศรษฐกิจชะลอแบบควบคุมได้ ตลาดรับความเสี่ยงได้"
    elif growth_signal >= 0.6 and inflation_signal <= -0.4 and market_signal >= 0.2:
        regime = "disinflationary_growth"
        headline = "การเติบโตยังพอไปต่อพร้อมเงินเฟ้อผ่อนลง"
    elif inflation_signal >= 0.8 and growth_signal >= -0.2:
        regime = "inflation_rebound"
        headline = "เงินเฟ้อมีความเสี่ยงกลับมา ตลาดต้องเลือกสินทรัพย์มากขึ้น"
    elif growth_signal <= -1.0 and inflation_signal <= 0.4:
        regime = "recession_risk"
        headline = "ความเสี่ยงเศรษฐกิจชะลอลงแรงเพิ่มขึ้น"
    elif growth_signal <= -0.5 and inflation_signal >= 0.8:
        regime = "stagflation_risk"
        headline = "เสี่ยงโตช้าแต่เงินเฟ้อยังดื้อ"

    confidence = min(0.95, 0.45 + (abs(growth_signal) + abs(inflation_signal) + abs(market_signal)) * 0.12)
    return MacroRegimeAssessment(
        regime=regime,
        confidence=round(confidence, 2),
        growth_signal=round(growth_signal, 2),
        inflation_signal=round(inflation_signal, 2),
        market_signal=round(market_signal, 2),
        headline=headline,
        rationale=tuple(dict.fromkeys(rationale))[:6] or ("มหภาคยังไม่ชัดเจน",),
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
