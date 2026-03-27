from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

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
    market_cap_bucket: str
    liquidity_tier: str
    price: float | None
    trend_direction: str
    trend_score: float
    rsi: float | None
    day_change_pct: float | None
    valuation_score: float
    quality_score: float
    growth_score: float
    technical_score: float
    macro_overlay_score: float
    universe_quality_score: float
    composite_score: float
    stance: str
    rationale: tuple[str, ...]
    macro_drivers: tuple[str, ...]
    universe_flags: tuple[str, ...]
    trailing_pe: float | None
    forward_pe: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    return_on_equity: float | None
    debt_to_equity: float | None
    profit_margin: float | None
    factor_risk_score: float = 0.0
    regime_fit_score: float = 0.0
    relative_strength_score: float = 0.0
    peer_relative_score: float = 0.0
    beta_1m: float | None = None
    benchmark_ticker: str | None = None
    peer_benchmark_ticker: str | None = None


def rank_stock_universe(
    *,
    stock_universe: Mapping[str, StockUniverseMember],
    quotes: Mapping[str, AssetQuote | None],
    trends: Mapping[str, TrendAssessment],
    fundamentals: Mapping[str, StockFundamentals | None],
    macro_context: Mapping[str, object] | None = None,
    market_histories: Mapping[str, list[object]] | None = None,
    macro_regime: str | None = None,
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
            macro_context=macro_context,
            market_histories=market_histories,
            macro_regime=macro_regime,
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
    macro_context: Mapping[str, object] | None,
    market_histories: Mapping[str, list[object]] | None,
    macro_regime: str | None,
) -> StockCandidate | None:
    if quote is None and trend is None:
        return None

    sector_name = (fundamentals.sector if fundamentals is not None and fundamentals.sector else member.sector) or "Unknown"
    rationale: list[str] = []
    technical_score = 0.0
    trend_score = float(trend.score) if trend is not None else 0.0
    universe_quality_score, universe_flags = _score_universe_quality(member)
    if trend is not None:
        technical_score += max(-2.5, min(3.5, trend.score * 0.9))
        if trend.direction == "uptrend":
            technical_score += 1.4
            rationale.append("primary trend still points higher")
        elif trend.direction == "downtrend":
            technical_score -= 1.6
            rationale.append("primary trend remains weak")
        else:
            rationale.append("price trend is still range-bound")
        if trend.rsi is not None:
            if 50 <= trend.rsi <= 68:
                technical_score += 0.8
                rationale.append("RSI is constructive without looking overbought")
            elif trend.rsi >= 75:
                technical_score -= 0.6
                rationale.append("RSI looks stretched")
            elif trend.rsi <= 40:
                technical_score -= 0.7
                rationale.append("momentum has not recovered yet")

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
                rationale.append("forward PE is still reasonable")
            elif fundamentals.forward_pe <= 35:
                valuation_score += 0.7
            elif fundamentals.forward_pe >= 50:
                valuation_score -= 1.0
                rationale.append("valuation already prices in a lot")
        elif fundamentals.trailing_pe is not None:
            if 0 < fundamentals.trailing_pe <= 28:
                valuation_score += 0.9
            elif fundamentals.trailing_pe >= 55:
                valuation_score -= 0.8

        if fundamentals.revenue_growth is not None:
            if fundamentals.revenue_growth >= 0.10:
                growth_score += 1.0
                rationale.append("revenue growth remains healthy")
            elif fundamentals.revenue_growth <= 0:
                growth_score -= 0.8

        if fundamentals.earnings_growth is not None:
            if fundamentals.earnings_growth >= 0.10:
                growth_score += 1.1
                rationale.append("earnings growth still supports the story")
            elif fundamentals.earnings_growth <= 0:
                growth_score -= 0.9

        if fundamentals.return_on_equity is not None:
            if fundamentals.return_on_equity >= 0.15:
                quality_score += 1.0
                rationale.append("ROE remains strong")
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
                rationale.append("balance sheet still looks healthy")
            elif fundamentals.debt_to_equity >= 180:
                quality_score -= 0.8
                rationale.append("leverage is elevated")

    benchmark_ticker = _resolve_benchmark_ticker(member.benchmark)
    peer_benchmark_ticker = _resolve_peer_benchmark_ticker(sector_name)
    beta_1m, relative_strength_score, peer_relative_score, factor_risk_score, regime_fit_score, factor_rationale = _score_factor_risk(
        asset_key=asset,
        benchmark_ticker=benchmark_ticker,
        peer_benchmark_ticker=peer_benchmark_ticker,
        market_histories=market_histories,
        macro_regime=macro_regime,
    )
    macro_overlay_score, macro_drivers = _score_macro_overlay(
        sector=sector_name,
        trend_direction=trend.direction if trend is not None else "sideways",
        macro_context=macro_context,
        macro_regime=macro_regime,
        beta_1m=beta_1m,
    )
    rationale.extend(macro_drivers[:2])
    rationale.extend(factor_rationale[:2])
    if universe_quality_score >= 0.78:
        rationale.append("universe quality and liquidity support execution")
    elif universe_quality_score <= 0.45:
        rationale.append("universe quality suggests tighter execution discipline")

    composite = round(
        (technical_score * 0.40)
        + (growth_score * 0.24)
        + (quality_score * 0.18)
        + (valuation_score * 0.10)
        + (macro_overlay_score * 0.06)
        + (factor_risk_score * 0.07)
        + (regime_fit_score * 0.05)
        + ((universe_quality_score - 0.5) * 0.5),
        2,
    )
    if composite >= 1.8 and (trend is None or trend.direction != "downtrend"):
        stance = "buy"
    elif composite >= 0.7 and (trend is None or trend.direction != "downtrend"):
        stance = "watch"
    else:
        stance = "avoid"

    if not rationale:
        rationale.append("signals are still mixed")

    return StockCandidate(
        asset=asset,
        ticker=member.ticker,
        company_name=member.company_name,
        sector=sector_name,
        benchmark=member.benchmark,
        market_cap_bucket=member.market_cap_bucket,
        liquidity_tier=member.liquidity_tier,
        price=quote.price if quote is not None else None,
        trend_direction=trend.direction if trend is not None else "sideways",
        trend_score=round(trend_score, 2),
        rsi=round(trend.rsi, 2) if trend is not None and trend.rsi is not None else None,
        day_change_pct=day_change_pct,
        valuation_score=round(valuation_score, 2),
        quality_score=round(quality_score, 2),
        growth_score=round(growth_score, 2),
        technical_score=round(technical_score, 2),
        macro_overlay_score=round(macro_overlay_score, 2),
        universe_quality_score=round(universe_quality_score, 2),
        composite_score=composite,
        stance=stance,
        rationale=tuple(rationale[:4]),
        macro_drivers=tuple(macro_drivers[:3]),
        universe_flags=universe_flags[:3],
        trailing_pe=fundamentals.trailing_pe if fundamentals is not None else None,
        forward_pe=fundamentals.forward_pe if fundamentals is not None else None,
        revenue_growth=fundamentals.revenue_growth if fundamentals is not None else None,
        earnings_growth=fundamentals.earnings_growth if fundamentals is not None else None,
        return_on_equity=fundamentals.return_on_equity if fundamentals is not None else None,
        debt_to_equity=fundamentals.debt_to_equity if fundamentals is not None else None,
        profit_margin=fundamentals.profit_margin if fundamentals is not None else None,
        factor_risk_score=round(factor_risk_score, 2),
        regime_fit_score=round(regime_fit_score, 2),
        relative_strength_score=round(relative_strength_score, 2),
        peer_relative_score=round(peer_relative_score, 2),
        beta_1m=round(beta_1m, 2) if beta_1m is not None else None,
        benchmark_ticker=benchmark_ticker,
        peer_benchmark_ticker=peer_benchmark_ticker,
    )


def _score_macro_overlay(
    *,
    sector: str,
    trend_direction: str,
    macro_context: Mapping[str, object] | None,
    macro_regime: str | None,
    beta_1m: float | None,
) -> tuple[float, tuple[str, ...]]:
    if not macro_context:
        return 0.0, ()

    normalized_sector = sector.strip().casefold()
    growth_like = {"technology", "communication services", "consumer discretionary", "real estate"}
    defensive = {"healthcare", "consumer staples", "utilities", "energy"}
    cyclical = {"consumer discretionary", "industrials", "materials", "financials", "real estate"}

    core_pce_yoy = _as_float(macro_context.get("core_pce_yoy"))
    gdp_qoq_annualized = _as_float(macro_context.get("gdp_qoq_annualized"))
    personal_spending_mom = _as_float(macro_context.get("personal_spending_mom"))
    alfred_payroll_revision_k = _as_float(macro_context.get("alfred_payroll_revision_k"))
    cftc_equity_net_pct_oi = _as_float(macro_context.get("cftc_equity_net_pct_oi"))
    cftc_ust10y_net_pct_oi = _as_float(macro_context.get("cftc_ust10y_net_pct_oi"))
    cftc_gold_net_pct_oi = _as_float(macro_context.get("cftc_gold_net_pct_oi"))
    finra_spy_short_volume_ratio = _as_float(macro_context.get("finra_spy_short_volume_ratio"))
    finra_qqq_short_volume_ratio = _as_float(macro_context.get("finra_qqq_short_volume_ratio"))

    overlay = 0.0
    drivers: list[str] = []
    stress_signals = 0

    if (core_pce_yoy is not None and core_pce_yoy >= 2.9) or (cftc_ust10y_net_pct_oi is not None and cftc_ust10y_net_pct_oi <= -10):
        stress_signals += 1
        if normalized_sector in growth_like:
            overlay -= 0.7
            drivers.append("sticky inflation lowers the setup for duration-sensitive growth")
        elif normalized_sector in defensive:
            overlay += 0.45
            drivers.append("defensive sector benefits from sticky inflation and rates pressure")

    if (
        (gdp_qoq_annualized is not None and gdp_qoq_annualized < 1.5)
        or (personal_spending_mom is not None and personal_spending_mom < 0)
        or (alfred_payroll_revision_k is not None and alfred_payroll_revision_k <= -40)
    ):
        stress_signals += 1
        if normalized_sector in cyclical or normalized_sector in growth_like:
            overlay -= 0.55
            drivers.append("slower growth and softer revisions reduce cyclical conviction")
        elif normalized_sector in defensive:
            overlay += 0.5
            drivers.append("defensive earnings profile fits a slower-growth backdrop")

    crowded_equity_tape = (
        cftc_equity_net_pct_oi is not None
        and cftc_equity_net_pct_oi >= 15
        and (
            (finra_spy_short_volume_ratio is not None and finra_spy_short_volume_ratio >= 0.55)
            or (finra_qqq_short_volume_ratio is not None and finra_qqq_short_volume_ratio >= 0.55)
        )
    )
    if crowded_equity_tape:
        stress_signals += 1
        if normalized_sector in growth_like:
            overlay -= 0.35
            drivers.append("crowded equity positioning cuts upside for high-beta leaders")
        elif normalized_sector in defensive:
            overlay += 0.2
            drivers.append("defensive exposure is preferred while tape confirmation is weak")

    if macro_regime is None and stress_signals >= 3:
        if normalized_sector in growth_like:
            overlay -= 0.4
            drivers.append("macro stress cluster implies tighter positioning for duration-sensitive leaders")
        elif normalized_sector in defensive:
            overlay += 0.35
            drivers.append("macro stress cluster increases the value of defensive cash-flow stability")

    if cftc_gold_net_pct_oi is not None and cftc_gold_net_pct_oi >= 10 and normalized_sector == "energy":
        overlay += 0.15
        drivers.append("commodity-linked exposure stays useful when inflation hedges are bid")

    if macro_regime in {"inflation_rebound", "stagflation_risk"}:
        if normalized_sector in {"energy", "financials", "consumer staples"}:
            overlay += 0.35
            drivers.append("current macro regime favors pricing power and inflation-linked sectors")
        elif normalized_sector in growth_like:
            overlay -= 0.45
            drivers.append("current macro regime penalizes long-duration growth exposure")
    elif macro_regime in {"soft_landing", "disinflationary_growth"}:
        if normalized_sector in growth_like:
            overlay += 0.35
            drivers.append("current macro regime still supports quality growth leadership")
        elif normalized_sector in {"utilities", "consumer staples"}:
            overlay -= 0.15
    elif macro_regime == "recession_risk":
        if normalized_sector in {"healthcare", "consumer staples", "utilities"}:
            overlay += 0.45
            drivers.append("defensive cash-flow profile fits recession-risk conditions")
        elif normalized_sector in cyclical or normalized_sector in growth_like:
            overlay -= 0.55
            drivers.append("recession-risk regime cuts cyclical and beta appetite")

    if beta_1m is not None and beta_1m >= 1.35 and macro_regime in {"inflation_rebound", "recession_risk", "stagflation_risk"}:
        overlay -= 0.25
        drivers.append("high realized beta is a headwind in the current regime")

    if trend_direction == "downtrend" and overlay < 0:
        overlay -= 0.15
    if trend_direction == "uptrend" and overlay > 0:
        overlay += 0.1

    return round(overlay, 2), tuple(dict.fromkeys(drivers))


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_universe_quality(member: StockUniverseMember) -> tuple[float, tuple[str, ...]]:
    market_cap_bucket = str(member.market_cap_bucket or "").strip().casefold()
    liquidity_tier = str(member.liquidity_tier or "").strip().casefold()
    quality = max(0.0, min(1.0, float(member.quality_hint)))
    market_cap_score = {"mega": 1.0, "large": 0.86, "mid": 0.58, "small": 0.28, "micro": 0.08}.get(market_cap_bucket, 0.52)
    liquidity_score = {"very_high": 1.0, "high": 0.88, "medium": 0.62, "low": 0.26}.get(liquidity_tier, 0.5)
    score = round(max(0.0, min(1.0, (market_cap_score * 0.4) + (liquidity_score * 0.4) + (quality * 0.2))), 2)
    flags: list[str] = []
    if liquidity_tier in {"very_high", "high"}:
        flags.append("liquidity_ok")
    elif liquidity_tier == "low":
        flags.append("liquidity_low")
    if market_cap_bucket in {"mega", "large"}:
        flags.append("cap_ok")
    elif market_cap_bucket in {"small", "micro"}:
        flags.append("cap_fragile")
    blocked_tags = {"adr", "spac", "penny", "illiquid", "leveraged", "inverse"}
    matched_tags = blocked_tags.intersection({str(tag).strip().casefold() for tag in member.tags})
    if matched_tags:
        score = round(max(0.0, score - 0.25), 2)
        flags.extend(sorted(f"tag_{tag}" for tag in matched_tags))
    return score, tuple(dict.fromkeys(flags))


def _resolve_benchmark_ticker(benchmark: str | None) -> str:
    normalized = str(benchmark or "").strip().casefold()
    return {
        "sp500": "SPY",
        "sp500_index": "SPY",
        "nasdaq100": "QQQ",
        "nasdaq_index": "QQQ",
        "watchlist": "SPY",
        "custom": "SPY",
    }.get(normalized, "SPY")


def _resolve_peer_benchmark_ticker(sector: str) -> str:
    normalized = str(sector or "").strip().casefold()
    return {
        "technology": "XLK",
        "financials": "XLF",
        "energy": "XLE",
        "consumer discretionary": "XLY",
        "consumer staples": "XLP",
        "healthcare": "XLV",
        "industrials": "XLI",
        "materials": "XLB",
        "utilities": "XLU",
        "communication services": "XLC",
        "real estate": "XLRE",
    }.get(normalized, "SPY")


def _score_factor_risk(
    *,
    asset_key: str,
    benchmark_ticker: str,
    peer_benchmark_ticker: str,
    market_histories: Mapping[str, list[object]] | None,
    macro_regime: str | None,
) -> tuple[float | None, float, float, float, float, tuple[str, ...]]:
    if not market_histories:
        return None, 0.0, 0.0, 0.0, 0.0, ()
    asset_bars = market_histories.get(asset_key) or market_histories.get(str(asset_key).upper()) or []
    benchmark_bars = market_histories.get(benchmark_ticker) or market_histories.get(benchmark_ticker.casefold()) or []
    peer_bars = market_histories.get(peer_benchmark_ticker) or market_histories.get(peer_benchmark_ticker.casefold()) or []
    asset_returns = _extract_return_series(asset_bars, lookback=21)
    benchmark_returns = _extract_return_series(benchmark_bars, lookback=21)
    peer_returns = _extract_return_series(peer_bars, lookback=21)
    beta_1m = _compute_beta(asset_returns, benchmark_returns)
    relative_strength_score = _compute_relative_strength_score(asset_returns, benchmark_returns)
    peer_relative_score = _compute_relative_strength_score(asset_returns, peer_returns)
    factor_risk_score = 0.0
    regime_fit_score = 0.0
    rationale: list[str] = []
    if beta_1m is not None:
        if macro_regime in {"soft_landing", "disinflationary_growth"}:
            if 0.9 <= beta_1m <= 1.35:
                factor_risk_score += 0.4
                rationale.append("realized beta is constructive without looking overstretched")
            elif beta_1m >= 1.55:
                factor_risk_score -= 0.25
        elif macro_regime in {"inflation_rebound", "recession_risk", "stagflation_risk"}:
            if beta_1m >= 1.2:
                factor_risk_score -= 0.55
                rationale.append("realized beta is too high for the current macro regime")
            elif beta_1m <= 0.85:
                factor_risk_score += 0.2
    if relative_strength_score >= 0.5:
        factor_risk_score += 0.25
        rationale.append("stock is outperforming its broad benchmark")
    elif relative_strength_score <= -0.5:
        factor_risk_score -= 0.35
    if peer_relative_score >= 0.45:
        regime_fit_score += 0.3
        rationale.append("sector-relative strength confirms the setup")
    elif peer_relative_score <= -0.45:
        regime_fit_score -= 0.25
    if macro_regime in {"soft_landing", "disinflationary_growth"} and relative_strength_score >= 0.35:
        regime_fit_score += 0.2
    if macro_regime in {"recession_risk", "stagflation_risk"} and beta_1m is not None and beta_1m >= 1.15:
        regime_fit_score -= 0.25
    return beta_1m, relative_strength_score, peer_relative_score, factor_risk_score, regime_fit_score, tuple(dict.fromkeys(rationale))


def _extract_return_series(bars: list[object], *, lookback: int) -> list[float]:
    closes: list[float] = []
    for item in bars[-max(lookback + 1, 3):]:
        close_value = _as_float(getattr(item, "close", None))
        if close_value is None or close_value <= 0:
            continue
        closes.append(close_value)
    if len(closes) < 3:
        return []
    returns: list[float] = []
    for previous_close, close_value in zip(closes, closes[1:], strict=False):
        if previous_close <= 0:
            continue
        returns.append((close_value / previous_close) - 1.0)
    return returns[-lookback:]


def _compute_beta(asset_returns: list[float], benchmark_returns: list[float]) -> float | None:
    sample_count = min(len(asset_returns), len(benchmark_returns))
    if sample_count < 5:
        return None
    asset_sample = asset_returns[-sample_count:]
    benchmark_sample = benchmark_returns[-sample_count:]
    benchmark_mean = sum(benchmark_sample) / sample_count
    asset_mean = sum(asset_sample) / sample_count
    covariance = sum((asset - asset_mean) * (bench - benchmark_mean) for asset, bench in zip(asset_sample, benchmark_sample, strict=False))
    variance = sum((bench - benchmark_mean) ** 2 for bench in benchmark_sample)
    if math.isclose(variance, 0.0, abs_tol=1e-12):
        return None
    return round(covariance / variance, 3)


def _compute_relative_strength_score(asset_returns: list[float], benchmark_returns: list[float]) -> float:
    sample_count = min(len(asset_returns), len(benchmark_returns))
    if sample_count < 5:
        return 0.0
    asset_total = sum(asset_returns[-sample_count:])
    benchmark_total = sum(benchmark_returns[-sample_count:])
    return round((asset_total - benchmark_total) * 10.0, 2)
