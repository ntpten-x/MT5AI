from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from invest_advisor_bot.analysis.trend_engine import TrendAssessment
from invest_advisor_bot.providers.market_data_client import AssetQuote, StockFundamentals
from invest_advisor_bot.universe import StockUniverseMember


@dataclass(slots=True, frozen=True)
class StockCandidate:
    asset: str
    ticker: str
    company_name: str
    sector: str
    benchmark: str
    price: float | None
    trend_direction: str
    trend_score: float
    rsi: float | None
    day_change_pct: float | None
    valuation_score: float
    quality_score: float
    growth_score: float
    technical_score: float
    composite_score: float
    stance: str
    rationale: tuple[str, ...]
    trailing_pe: float | None
    forward_pe: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    return_on_equity: float | None
    debt_to_equity: float | None
    profit_margin: float | None


def rank_stock_universe(
    *,
    stock_universe: Mapping[str, StockUniverseMember],
    quotes: Mapping[str, AssetQuote | None],
    trends: Mapping[str, TrendAssessment],
    fundamentals: Mapping[str, StockFundamentals | None],
    top_k: int = 5,
    max_per_sector: int = 2,
) -> list[StockCandidate]:
    candidates: list[StockCandidate] = []
    for asset, member in stock_universe.items():
        candidate = _build_candidate(
            asset=asset,
            member=member,
            quote=quotes.get(asset),
            trend=trends.get(asset),
            fundamentals=fundamentals.get(asset),
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.composite_score, reverse=True)
    selected: list[StockCandidate] = []
    sector_counts: dict[str, int] = {}
    for candidate in candidates:
        current_count = sector_counts.get(candidate.sector, 0)
        if current_count >= max_per_sector:
            continue
        selected.append(candidate)
        sector_counts[candidate.sector] = current_count + 1
        if len(selected) >= max(1, top_k):
            break
    return selected


def _build_candidate(
    *,
    asset: str,
    member: StockUniverseMember,
    quote: AssetQuote | None,
    trend: TrendAssessment | None,
    fundamentals: StockFundamentals | None,
) -> StockCandidate | None:
    if quote is None and trend is None:
        return None

    rationale: list[str] = []
    technical_score = 0.0
    trend_score = float(trend.score) if trend is not None else 0.0
    if trend is not None:
        technical_score += max(-2.5, min(3.5, trend.score * 0.9))
        if trend.direction == "uptrend":
            technical_score += 1.4
            rationale.append("แนวโน้มหลักยังเป็นขาขึ้น")
        elif trend.direction == "downtrend":
            technical_score -= 1.6
            rationale.append("แนวโน้มหลักยังอ่อนแรง")
        else:
            rationale.append("แนวโน้มยังแกว่งในกรอบ")
        if trend.rsi is not None:
            if 50 <= trend.rsi <= 68:
                technical_score += 0.8
                rationale.append("RSI ยังอยู่ในโซนบวกที่ไม่ร้อนเกินไป")
            elif trend.rsi >= 75:
                technical_score -= 0.6
                rationale.append("RSI เริ่มร้อนเกินไป")
            elif trend.rsi <= 40:
                technical_score -= 0.7
                rationale.append("แรงส่งยังไม่ฟื้นเต็มที่")

    day_change_pct = None
    if quote is not None and quote.previous_close:
        day_change_pct = round(((quote.price - quote.previous_close) / quote.previous_close) * 100.0, 2)

    valuation_score = 0.0
    quality_score = 0.0
    growth_score = 0.0
    if fundamentals is not None:
        if fundamentals.forward_pe is not None:
            if 0 < fundamentals.forward_pe <= 25:
                valuation_score += 1.4
                rationale.append("Forward PE ยังไม่ตึงเกินไป")
            elif fundamentals.forward_pe <= 35:
                valuation_score += 0.7
            elif fundamentals.forward_pe >= 50:
                valuation_score -= 1.0
                rationale.append("Valuation ค่อนข้างแพง")
        elif fundamentals.trailing_pe is not None:
            if 0 < fundamentals.trailing_pe <= 28:
                valuation_score += 0.9
            elif fundamentals.trailing_pe >= 55:
                valuation_score -= 0.8

        if fundamentals.revenue_growth is not None:
            if fundamentals.revenue_growth >= 0.10:
                growth_score += 1.0
                rationale.append("รายได้ยังเติบโตดี")
            elif fundamentals.revenue_growth <= 0:
                growth_score -= 0.8

        if fundamentals.earnings_growth is not None:
            if fundamentals.earnings_growth >= 0.10:
                growth_score += 1.1
                rationale.append("กำไรยังขยายตัวดี")
            elif fundamentals.earnings_growth <= 0:
                growth_score -= 0.9

        if fundamentals.return_on_equity is not None:
            if fundamentals.return_on_equity >= 0.15:
                quality_score += 1.0
                rationale.append("ROE อยู่ในเกณฑ์ดี")
            elif fundamentals.return_on_equity <= 0.05:
                quality_score -= 0.5

        if fundamentals.profit_margin is not None:
            if fundamentals.profit_margin >= 0.15:
                quality_score += 0.8
            elif fundamentals.profit_margin <= 0.05:
                quality_score -= 0.5

        if fundamentals.debt_to_equity is not None:
            if fundamentals.debt_to_equity <= 80:
                quality_score += 0.6
                rationale.append("งบดุลไม่ตึงเกินไป")
            elif fundamentals.debt_to_equity >= 180:
                quality_score -= 0.8
                rationale.append("หนี้สูง ต้องระวัง")

    composite = round((technical_score * 0.45) + (growth_score * 0.25) + (quality_score * 0.20) + (valuation_score * 0.10), 2)
    if composite >= 1.8 and (trend is None or trend.direction != "downtrend"):
        stance = "buy"
    elif composite >= 0.7 and (trend is None or trend.direction != "downtrend"):
        stance = "watch"
    else:
        stance = "avoid"

    if not rationale:
        rationale.append("ข้อมูลพื้นฐานและเทคนิคยังกลาง ๆ")

    return StockCandidate(
        asset=asset,
        ticker=member.ticker,
        company_name=member.company_name,
        sector=member.sector,
        benchmark=member.benchmark,
        price=quote.price if quote is not None else None,
        trend_direction=trend.direction if trend is not None else "sideways",
        trend_score=round(trend_score, 2),
        rsi=round(trend.rsi, 2) if trend is not None and trend.rsi is not None else None,
        day_change_pct=day_change_pct,
        valuation_score=round(valuation_score, 2),
        quality_score=round(quality_score, 2),
        growth_score=round(growth_score, 2),
        technical_score=round(technical_score, 2),
        composite_score=composite,
        stance=stance,
        rationale=tuple(rationale[:4]),
        trailing_pe=fundamentals.trailing_pe if fundamentals is not None else None,
        forward_pe=fundamentals.forward_pe if fundamentals is not None else None,
        revenue_growth=fundamentals.revenue_growth if fundamentals is not None else None,
        earnings_growth=fundamentals.earnings_growth if fundamentals is not None else None,
        return_on_equity=fundamentals.return_on_equity if fundamentals is not None else None,
        debt_to_equity=fundamentals.debt_to_equity if fundamentals is not None else None,
        profit_margin=fundamentals.profit_margin if fundamentals is not None else None,
    )
