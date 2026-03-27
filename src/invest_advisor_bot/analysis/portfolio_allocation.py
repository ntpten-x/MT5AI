from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from invest_advisor_bot.analysis.macro_regime import MacroRegimeAssessment
from invest_advisor_bot.analysis.portfolio_profile import InvestorProfile

AllocationMixCategory = Literal["cash", "gold", "core_etf", "growth", "defensive"]


@dataclass(slots=True, frozen=True)
class AllocationMixBucket:
    category: AllocationMixCategory
    target_pct: int
    stance: str
    rationale: str
    preferred_assets: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class PortfolioAllocationPlan:
    profile_name: str
    macro_regime: str
    narrative: str
    buckets: tuple[AllocationMixBucket, ...]
    rebalance_note: str


@dataclass(slots=True, frozen=True)
class PortfolioHoldingReview:
    ticker: str
    category: AllocationMixCategory
    market_value: float
    cost_basis: float | None
    unrealized_pnl_pct: float | None
    current_weight_pct: float
    note: str | None = None


@dataclass(slots=True, frozen=True)
class PortfolioDriftBucket:
    category: AllocationMixCategory
    target_pct: float
    current_pct: float
    drift_pct: float
    action: str
    top_holdings: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class PortfolioRebalanceReview:
    total_market_value: float
    holdings_count: int
    buckets: tuple[PortfolioDriftBucket, ...]
    action_summary: str


_BASE_ALLOCATIONS: dict[str, dict[AllocationMixCategory, int]] = {
    "conservative": {"cash": 22, "gold": 18, "core_etf": 25, "growth": 10, "defensive": 25},
    "balanced": {"cash": 14, "gold": 12, "core_etf": 33, "growth": 21, "defensive": 20},
    "growth": {"cash": 6, "gold": 6, "core_etf": 26, "growth": 46, "defensive": 16},
}

_CATEGORY_ASSETS: dict[AllocationMixCategory, tuple[str, ...]] = {
    "cash": (),
    "gold": ("gold_futures", "gld_etf", "iau_etf"),
    "core_etf": ("voo_etf", "vti_etf", "spy_etf"),
    "growth": ("qqq_etf", "xlk_etf", "xlc_etf", "xly_etf"),
    "defensive": ("xlp_etf", "xlv_etf", "xlu_etf", "tlt_etf", "xlf_etf"),
}

_DIRECT_TICKER_CATEGORY: dict[str, AllocationMixCategory] = {
    "CASH": "cash",
    "USD": "cash",
    "GLD": "gold",
    "IAU": "gold",
    "GC=F": "gold",
    "XAUUSD": "gold",
    "VOO": "core_etf",
    "VTI": "core_etf",
    "SPY": "core_etf",
    "QQQ": "growth",
    "XLK": "growth",
    "XLC": "growth",
    "XLY": "growth",
    "XLP": "defensive",
    "XLV": "defensive",
    "XLU": "defensive",
    "XLF": "defensive",
    "XLE": "defensive",
    "TLT": "defensive",
    "AAPL": "growth",
    "MSFT": "growth",
    "NVDA": "growth",
    "META": "growth",
    "AMZN": "growth",
    "GOOGL": "growth",
    "NFLX": "growth",
    "AVGO": "growth",
    "AMD": "growth",
    "ADBE": "growth",
    "CRM": "growth",
    "JPM": "defensive",
    "BAC": "defensive",
    "XOM": "defensive",
    "CVX": "defensive",
    "KO": "defensive",
    "PG": "defensive",
    "JNJ": "defensive",
    "UNH": "defensive",
}

_SECTOR_CATEGORY: dict[str, AllocationMixCategory] = {
    "technology": "growth",
    "communication services": "growth",
    "consumer discretionary": "growth",
    "financials": "defensive",
    "energy": "defensive",
    "utilities": "defensive",
    "consumer staples": "defensive",
    "healthcare": "defensive",
    "real estate": "defensive",
    "materials": "core_etf",
    "industrials": "core_etf",
}

_REGIME_NARRATIVE: dict[str, str] = {
    "soft_landing": "Lean slightly into core ETF and growth while keeping a measured cash reserve.",
    "disinflationary_growth": "Favour quality growth and broad ETF exposure while the macro backdrop stays constructive.",
    "inflation_rebound": "Add gold and defensive exposure to cushion valuation pressure.",
    "recession_risk": "Raise cash and defensive exposure, and reduce aggressive growth risk.",
    "stagflation_risk": "Hold more gold and liquidity, and focus on defensive earnings quality.",
    "mixed_transition": "Keep the mix balanced and wait for a cleaner macro regime signal.",
}


def build_portfolio_allocation_plan(
    *,
    investor_profile: InvestorProfile,
    macro_regime: MacroRegimeAssessment,
    asset_snapshots: Sequence[Mapping[str, object]],
    macro_context: Mapping[str, object] | None = None,
    learning_multiplier: float = 1.0,
) -> PortfolioAllocationPlan:
    allocations = dict(_BASE_ALLOCATIONS[investor_profile.name])
    regime = macro_regime.regime
    if regime in {"recession_risk", "stagflation_risk"}:
        allocations["cash"] += 6
        allocations["gold"] += 4
        allocations["growth"] -= 8
        allocations["core_etf"] -= 2
    elif regime == "inflation_rebound":
        allocations["gold"] += 5
        allocations["defensive"] += 3
        allocations["growth"] -= 5
        allocations["cash"] -= 3
    elif regime in {"soft_landing", "disinflationary_growth"}:
        allocations["growth"] += 5
        allocations["core_etf"] += 3
        allocations["cash"] -= 4
        allocations["defensive"] -= 4
    effective_learning_multiplier = max(0.75, min(1.35, float(learning_multiplier)))
    overlay_reasons = _apply_direct_macro_overlays(
        allocations=allocations,
        macro_context=macro_context,
        learning_multiplier=effective_learning_multiplier,
    )
    allocations = _normalize_allocations(allocations)

    buckets: list[AllocationMixBucket] = []
    for category, target_pct in allocations.items():
        base_pct = _BASE_ALLOCATIONS[investor_profile.name][category]
        if target_pct >= base_pct + 4:
            stance = "Overweight"
        elif target_pct <= base_pct - 4:
            stance = "Underweight"
        else:
            stance = "Neutral"
        preferred_assets = _pick_assets(category=category, asset_snapshots=asset_snapshots)
        buckets.append(
            AllocationMixBucket(
                category=category,
                target_pct=target_pct,
                stance=stance,
                rationale=_build_bucket_rationale(category=category, regime=regime, target_pct=target_pct),
                preferred_assets=preferred_assets,
            )
        )

    narrative = _REGIME_NARRATIVE.get(regime, _REGIME_NARRATIVE["mixed_transition"])
    if overlay_reasons:
        narrative = f"{narrative} Direct overlay: {'; '.join(overlay_reasons[:2])}."
    rebalance_note = "Review allocation every day and rebalance any bucket drifting more than 5 percentage points."
    if overlay_reasons:
        rebalance_note += (
            " Active macro overlays: "
            + "; ".join(overlay_reasons[:3])
            + f" | shift multiplier x{effective_learning_multiplier:.2f}."
        )

    return PortfolioAllocationPlan(
        profile_name=investor_profile.name,
        macro_regime=regime,
        narrative=narrative,
        buckets=tuple(buckets),
        rebalance_note=rebalance_note,
    )


def infer_allocation_mix_category(*, symbol: str, sector: str | None = None) -> AllocationMixCategory:
    normalized_symbol = symbol.strip().upper()
    if normalized_symbol in _DIRECT_TICKER_CATEGORY:
        return _DIRECT_TICKER_CATEGORY[normalized_symbol]
    normalized_sector = (sector or "").strip().casefold()
    if normalized_sector in _SECTOR_CATEGORY:
        return _SECTOR_CATEGORY[normalized_sector]
    return "core_etf"


def build_portfolio_rebalance_review(
    *,
    allocation_plan: PortfolioAllocationPlan,
    holdings: Sequence[PortfolioHoldingReview],
) -> PortfolioRebalanceReview | None:
    if not holdings:
        return None
    total_market_value = sum(max(0.0, holding.market_value) for holding in holdings)
    if total_market_value <= 0:
        return None

    category_values: dict[AllocationMixCategory, float] = {bucket.category: 0.0 for bucket in allocation_plan.buckets}
    category_holdings: dict[AllocationMixCategory, list[PortfolioHoldingReview]] = {
        bucket.category: [] for bucket in allocation_plan.buckets
    }
    for holding in holdings:
        category_values[holding.category] = category_values.get(holding.category, 0.0) + holding.market_value
        category_holdings.setdefault(holding.category, []).append(holding)

    buckets: list[PortfolioDriftBucket] = []
    strongest_bucket: PortfolioDriftBucket | None = None
    for bucket in allocation_plan.buckets:
        current_pct = (category_values.get(bucket.category, 0.0) / total_market_value) * 100.0
        drift_pct = current_pct - float(bucket.target_pct)
        action = "Near target"
        if drift_pct >= 5.0:
            action = "Trim / rebalance down"
        elif drift_pct <= -5.0:
            action = "Add / build up"
        top_holdings = tuple(
            holding.ticker
            for holding in sorted(
                category_holdings.get(bucket.category, []),
                key=lambda item: item.market_value,
                reverse=True,
            )[:3]
        )
        current_bucket = PortfolioDriftBucket(
            category=bucket.category,
            target_pct=float(bucket.target_pct),
            current_pct=round(current_pct, 1),
            drift_pct=round(drift_pct, 1),
            action=action,
            top_holdings=top_holdings,
        )
        buckets.append(current_bucket)
        if strongest_bucket is None or abs(current_bucket.drift_pct) > abs(strongest_bucket.drift_pct):
            strongest_bucket = current_bucket

    if strongest_bucket is None:
        action_summary = "Current holdings do not have a usable rebalance signal yet."
    elif abs(strongest_bucket.drift_pct) < 5.0:
        action_summary = "Current portfolio is already close to the target mix."
    elif strongest_bucket.drift_pct > 0:
        action_summary = (
            f"{strongest_bucket.category} is overweight by about {strongest_bucket.drift_pct:.1f}% "
            "and should be trimmed gradually."
        )
    else:
        action_summary = (
            f"{strongest_bucket.category} is underweight by about {abs(strongest_bucket.drift_pct):.1f}% "
            "and can be added gradually."
        )

    return PortfolioRebalanceReview(
        total_market_value=round(total_market_value, 2),
        holdings_count=len(holdings),
        buckets=tuple(buckets),
        action_summary=action_summary,
    )


def _pick_assets(*, category: AllocationMixCategory, asset_snapshots: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    allowed = set(_CATEGORY_ASSETS[category])
    ranked: list[tuple[float, str]] = []
    for snapshot in asset_snapshots:
        asset = str(snapshot.get("asset") or "")
        if asset not in allowed:
            continue
        trend_score = _as_float(snapshot.get("trend_score")) or 0.0
        ranked.append((trend_score, asset))
    ranked.sort(reverse=True)
    return tuple(asset for _, asset in ranked[:2]) or _CATEGORY_ASSETS[category][:1]


def _build_bucket_rationale(*, category: AllocationMixCategory, regime: str, target_pct: int) -> str:
    if category == "cash":
        return f"Keep roughly {target_pct}% in cash to absorb volatility and fund new entries."
    if category == "gold":
        return f"Hold about {target_pct}% in gold as an inflation and tail-risk hedge in {regime}."
    if category == "core_etf":
        return f"Keep around {target_pct}% in broad core ETFs as the base of the portfolio."
    if category == "growth":
        return f"Use roughly {target_pct}% for growth exposure when the macro regime still supports momentum."
    return f"Keep about {target_pct}% in defensive exposure to stabilise the portfolio when breadth weakens."


def _normalize_allocations(allocations: Mapping[AllocationMixCategory, int]) -> dict[AllocationMixCategory, int]:
    normalized = {key: max(0, int(value)) for key, value in allocations.items()}
    total = sum(normalized.values()) or 1
    scaled = {key: int(round((value / total) * 100)) for key, value in normalized.items()}
    delta = 100 - sum(scaled.values())
    if delta:
        ordered = sorted(scaled, key=lambda item: scaled[item], reverse=(delta > 0))
        for key in ordered:
            if delta == 0:
                break
            if delta < 0 and scaled[key] == 0:
                continue
            scaled[key] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1
    return scaled


def _apply_direct_macro_overlays(
    *,
    allocations: dict[AllocationMixCategory, int],
    macro_context: Mapping[str, object] | None,
    learning_multiplier: float,
) -> list[str]:
    context = macro_context or {}
    reasons: list[str] = []
    core_pce_yoy = _as_float(context.get("core_pce_yoy"))
    gdp_qoq_annualized = _as_float(context.get("gdp_qoq_annualized"))
    personal_spending_mom = _as_float(context.get("personal_spending_mom"))
    alfred_payroll_revision_k = _as_float(context.get("alfred_payroll_revision_k"))
    cftc_equity_net_pct_oi = _as_float(context.get("cftc_equity_net_pct_oi"))
    cftc_ust10y_net_pct_oi = _as_float(context.get("cftc_ust10y_net_pct_oi"))
    finra_spy_short_volume_ratio = _as_float(context.get("finra_spy_short_volume_ratio"))
    finra_qqq_short_volume_ratio = _as_float(context.get("finra_qqq_short_volume_ratio"))

    if (core_pce_yoy is not None and core_pce_yoy >= 2.9) or (cftc_ust10y_net_pct_oi is not None and cftc_ust10y_net_pct_oi <= -10):
        allocations["gold"] += _scaled_shift(2, learning_multiplier)
        allocations["cash"] += _scaled_shift(2, learning_multiplier)
        allocations["growth"] -= _scaled_shift(3, learning_multiplier)
        allocations["core_etf"] -= _scaled_shift(1, learning_multiplier)
        reasons.append("sticky inflation / rates positioning keeps the portfolio more defensive")

    if (
        (gdp_qoq_annualized is not None and gdp_qoq_annualized < 1.5)
        or (personal_spending_mom is not None and personal_spending_mom < 0)
        or (alfred_payroll_revision_k is not None and alfred_payroll_revision_k <= -40)
    ):
        allocations["cash"] += _scaled_shift(2, learning_multiplier)
        allocations["defensive"] += _scaled_shift(2, learning_multiplier)
        allocations["growth"] -= _scaled_shift(3, learning_multiplier)
        allocations["core_etf"] -= _scaled_shift(1, learning_multiplier)
        reasons.append("growth and labor revisions argue for lower cyclical risk")

    crowded_equity_tape = (
        cftc_equity_net_pct_oi is not None
        and cftc_equity_net_pct_oi >= 15
        and (
            (finra_spy_short_volume_ratio is not None and finra_spy_short_volume_ratio >= 0.55)
            or (finra_qqq_short_volume_ratio is not None and finra_qqq_short_volume_ratio >= 0.55)
        )
    )
    if crowded_equity_tape:
        allocations["cash"] += _scaled_shift(2, learning_multiplier)
        allocations["defensive"] += _scaled_shift(1, learning_multiplier)
        allocations["growth"] -= _scaled_shift(2, learning_multiplier)
        allocations["core_etf"] -= _scaled_shift(1, learning_multiplier)
        reasons.append("crowded equity positioning plus heavy short flow lowers risk appetite")

    return reasons


def _scaled_shift(base_shift: int, learning_multiplier: float) -> int:
    magnitude = max(1, int(round(abs(base_shift) * max(0.75, min(1.35, float(learning_multiplier))))))
    return magnitude if base_shift >= 0 else -magnitude


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
