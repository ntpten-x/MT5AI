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
    context = macro_context or {}
    vix = _as_float(context.get("vix"))
    tnx = _as_float(context.get("tnx"))
    cpi_yoy = _as_float(context.get("cpi_yoy"))
    core_cpi_yoy = _as_float(context.get("core_cpi_yoy"))
    core_pce_yoy = _as_float(context.get("core_pce_yoy"))
    yield_spread = _as_float(context.get("yield_spread_10y_2y"))
    high_yield_spread = _as_float(context.get("high_yield_spread"))
    unemployment_rate = _as_float(context.get("unemployment_rate"))
    payrolls_mom_k = _as_float(context.get("payrolls_mom_k"))
    gdp_qoq_annualized = _as_float(context.get("gdp_qoq_annualized"))
    personal_spending_mom = _as_float(context.get("personal_spending_mom"))
    alfred_payroll_revision_k = _as_float(context.get("alfred_payroll_revision_k"))
    alfred_cpi_revision_pct = _as_float(context.get("alfred_cpi_revision_pct"))
    cftc_equity_net_pct_oi = _as_float(context.get("cftc_equity_net_pct_oi"))
    cftc_ust10y_net_pct_oi = _as_float(context.get("cftc_ust10y_net_pct_oi"))
    cftc_gold_net_pct_oi = _as_float(context.get("cftc_gold_net_pct_oi"))
    finra_spy_short_volume_ratio = _as_float(context.get("finra_spy_short_volume_ratio"))
    finra_qqq_short_volume_ratio = _as_float(context.get("finra_qqq_short_volume_ratio"))

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
            rationale.append(f"VIX low ({vix:.1f})")
        elif vix >= 26:
            growth_signal -= 1.0
            rationale.append(f"VIX high ({vix:.1f})")
    if tnx is not None:
        if tnx <= 4.1:
            growth_signal += 0.5
        elif tnx >= 4.6:
            growth_signal -= 0.6
            inflation_signal += 0.7
            rationale.append(f"US 10Y yield high ({tnx:.2f})")
    if cpi_yoy is not None:
        if cpi_yoy <= 2.8:
            inflation_signal -= 0.8
            rationale.append(f"headline CPI cooling ({cpi_yoy:.1f}%)")
        elif cpi_yoy >= 3.4:
            inflation_signal += 1.0
            rationale.append(f"headline CPI re-accelerating ({cpi_yoy:.1f}%)")
    if core_cpi_yoy is not None and core_cpi_yoy >= 3.3:
        inflation_signal += 0.5
        rationale.append(f"core CPI sticky ({core_cpi_yoy:.1f}%)")
    if core_pce_yoy is not None:
        if core_pce_yoy >= 2.9:
            inflation_signal += 0.5
            rationale.append(f"core PCE sticky ({core_pce_yoy:.1f}%)")
        elif core_pce_yoy <= 2.4:
            inflation_signal -= 0.3
    if yield_spread is not None and yield_spread < 0:
        growth_signal -= 0.6
        market_signal -= 0.2
        rationale.append(f"yield curve inverted ({yield_spread:.2f})")
    if high_yield_spread is not None and high_yield_spread >= 4.5:
        growth_signal -= 0.5
        market_signal -= 0.3
        rationale.append(f"credit spread wide ({high_yield_spread:.2f})")
    if unemployment_rate is not None and unemployment_rate >= 4.2:
        growth_signal -= 0.4
        rationale.append(f"labor market softening ({unemployment_rate:.1f}%)")
    if payrolls_mom_k is not None and payrolls_mom_k < 125:
        growth_signal -= 0.3
        rationale.append(f"payroll growth soft ({payrolls_mom_k:.0f}k)")
    if gdp_qoq_annualized is not None and gdp_qoq_annualized < 1.5:
        growth_signal -= 0.5
        rationale.append(f"GDP growth slowing ({gdp_qoq_annualized:.1f}%)")
    if personal_spending_mom is not None and personal_spending_mom < 0:
        growth_signal -= 0.3
        rationale.append(f"personal spending negative ({personal_spending_mom:.1f}%)")
    if alfred_payroll_revision_k is not None and alfred_payroll_revision_k <= -40:
        growth_signal -= 0.5
        rationale.append(f"ALFRED payroll revision weak ({alfred_payroll_revision_k:.0f}k)")
    if alfred_cpi_revision_pct is not None and alfred_cpi_revision_pct >= 0.15:
        inflation_signal += 0.2
        rationale.append(f"CPI revisions moved higher ({alfred_cpi_revision_pct:.2f}%)")
    if cftc_equity_net_pct_oi is not None and cftc_equity_net_pct_oi >= 15:
        market_signal -= 0.3
        rationale.append(f"CFTC equity positioning crowded ({cftc_equity_net_pct_oi:.1f}%)")
    if cftc_ust10y_net_pct_oi is not None and cftc_ust10y_net_pct_oi <= -10:
        growth_signal -= 0.2
        inflation_signal += 0.2
        rationale.append(f"CFTC UST 10Y positioning pressures duration ({cftc_ust10y_net_pct_oi:.1f}%)")
    if cftc_gold_net_pct_oi is not None and cftc_gold_net_pct_oi >= 10:
        inflation_signal += 0.1
    if (
        (finra_spy_short_volume_ratio is not None and finra_spy_short_volume_ratio >= 0.55)
        or (finra_qqq_short_volume_ratio is not None and finra_qqq_short_volume_ratio >= 0.55)
    ):
        market_signal -= 0.35
        rationale.append("FINRA short flow elevated")
    if (
        cftc_equity_net_pct_oi is not None
        and cftc_equity_net_pct_oi >= 15
        and (
            (finra_spy_short_volume_ratio is not None and finra_spy_short_volume_ratio >= 0.55)
            or (finra_qqq_short_volume_ratio is not None and finra_qqq_short_volume_ratio >= 0.55)
        )
    ):
        market_signal -= 0.2
        rationale.append("crowded long risk with hedge pressure still high")

    if equity_uptrends >= max(2, equity_downtrends + 1):
        growth_signal += 0.8
        rationale.append("major equity indexes still trend higher")
    elif equity_downtrends >= max(2, equity_uptrends + 1):
        growth_signal -= 0.9
        rationale.append("major equity indexes losing momentum")

    if defensive_uptrends >= 2:
        inflation_signal += 0.2
        market_signal -= 0.2

    regime: MacroRegimeName = "mixed_transition"
    headline = "macro signals remain mixed"
    if growth_signal >= 1.0 and inflation_signal <= 0.2 and market_signal >= 0.3:
        regime = "soft_landing"
        headline = "growth is holding up while risk appetite stays constructive"
    elif growth_signal >= 0.6 and inflation_signal <= -0.4 and market_signal >= 0.2:
        regime = "disinflationary_growth"
        headline = "growth remains supported while inflation pressure fades"
    elif inflation_signal >= 0.8 and growth_signal >= -0.2:
        regime = "inflation_rebound"
        headline = "inflation risk is re-accelerating and valuation pressure is building"
    elif growth_signal <= -1.0 and inflation_signal <= 0.4:
        regime = "recession_risk"
        headline = "growth and market internals point to rising recession risk"
    elif growth_signal <= -0.5 and inflation_signal >= 0.8:
        regime = "stagflation_risk"
        headline = "growth is slowing while inflation remains sticky"

    confidence = min(0.95, 0.45 + (abs(growth_signal) + abs(inflation_signal) + abs(market_signal)) * 0.12)
    return MacroRegimeAssessment(
        regime=regime,
        confidence=round(confidence, 2),
        growth_signal=round(growth_signal, 2),
        inflation_signal=round(inflation_signal, 2),
        market_signal=round(market_signal, 2),
        headline=headline,
        rationale=tuple(dict.fromkeys(rationale))[:6] or ("macro backdrop is still inconclusive",),
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
