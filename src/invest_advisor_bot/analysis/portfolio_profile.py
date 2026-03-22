from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

InvestorProfileName = Literal["conservative", "balanced", "growth"]
MarketRegime = Literal["risk_off", "neutral", "risk_on"]

_BASE_ALLOCATIONS: dict[InvestorProfileName, dict[str, int]] = {
    "conservative": {"us_equity": 25, "gold": 20, "bonds": 35, "cash": 20},
    "balanced": {"us_equity": 45, "gold": 15, "bonds": 25, "cash": 15},
    "growth": {"us_equity": 72, "gold": 8, "bonds": 5, "cash": 15},
}

_PROFILE_DETAILS: dict[InvestorProfileName, tuple[str, str, str, str]] = {
    "conservative": (
        "สายรักษาเงินต้น",
        "เน้นรักษาเงินต้น ลดความผันผวน และถือสินทรัพย์ป้องกันความเสี่ยงมากขึ้น",
        "รับความเสี่ยงต่ำ เหมาะกับผู้ที่ยอมเสียโอกาสบางส่วนเพื่อคุม drawdown",
        "ทบทวนพอร์ตทุกเดือนและเพิ่มเงินสดเมื่อความผันผวนสูงผิดปกติ",
    ),
    "balanced": (
        "สายสมดุล",
        "ต้องการเติบโตระยะยาวโดยยังคุมความเสี่ยงให้อยู่ในระดับรับได้",
        "รับความเสี่ยงปานกลาง กระจายสินทรัพย์เพื่อไม่พึ่งหุ้นอย่างเดียว",
        "รีบาลานซ์รายเดือนหรือเมื่อสัดส่วนหลักเบี่ยงเกิน 5%",
    ),
    "growth": (
        "สายเติบโต",
        "เน้นการเติบโตของเงินทุนระยะยาวและยอมรับความผันผวนระยะสั้นได้มากขึ้น",
        "รับความเสี่ยงสูงกว่า แต่ยังต้องมี hedge และเงินสดสำรองบางส่วน",
        "ติดตาม momentum ของหุ้นหลักอย่างใกล้ชิดและลดความเสี่ยงเมื่อ regime เปลี่ยน",
    ),
}

_CATEGORY_LABELS: dict[str, str] = {
    "us_equity": "หุ้น/ETF สหรัฐ",
    "gold": "ทองคำ",
    "bonds": "พันธบัตร/ตราสารหนี้",
    "cash": "เงินสดสำรอง",
}

_CATEGORY_ASSETS: dict[str, tuple[str, ...]] = {
    "us_equity": (
        "voo_etf",
        "vti_etf",
        "spy_etf",
        "qqq_etf",
        "xlk_etf",
        "xlf_etf",
        "xle_etf",
        "sp500_index",
        "nasdaq_index",
    ),
    "gold": ("gold_futures", "gld_etf", "iau_etf"),
    "bonds": ("tlt_etf",),
    "cash": (),
}


@dataclass(slots=True, frozen=True)
class InvestorProfile:
    name: InvestorProfileName
    title_th: str
    objective: str
    risk_summary: str
    rebalance_hint: str
    base_allocations: Mapping[str, int]


@dataclass(slots=True, frozen=True)
class AllocationBucket:
    category: str
    label: str
    target_pct: int
    stance: str
    preferred_assets: tuple[str, ...]
    rationale: str


@dataclass(slots=True, frozen=True)
class PortfolioPlan:
    profile: InvestorProfile
    market_regime: MarketRegime
    regime_summary: str
    buckets: tuple[AllocationBucket, ...]
    action_plan: str
    risk_watch: str


def normalize_profile_name(value: str | None, *, default: InvestorProfileName = "balanced") -> InvestorProfileName:
    normalized = (value or "").strip().casefold()
    aliases = {
        "conservative": "conservative",
        "safe": "conservative",
        "low-risk": "conservative",
        "balanced": "balanced",
        "moderate": "balanced",
        "growth": "growth",
        "aggressive": "growth",
        "aggressive-growth": "growth",
        "aggressive_growth": "growth",
        "high-risk": "growth",
    }
    return aliases.get(normalized, default)  # type: ignore[return-value]


def detect_investor_profile(text: str | None) -> InvestorProfileName | None:
    normalized = (text or "").casefold()
    conservative_keywords = (
        "รักษาเงินต้น",
        "ปลอดภัย",
        "เสี่ยงต่ำ",
        "รับความเสี่ยงต่ำ",
        "conservative",
        "capital preservation",
        "wealth preservation",
        "safe",
    )
    balanced_keywords = (
        "สมดุล",
        "ปานกลาง",
        "balanced",
        "moderate",
        "รับความเสี่ยงกลาง",
    )
    growth_keywords = (
        "เติบโต",
        "เสี่ยงสูง",
        "รับความเสี่ยงสูง",
        "growth",
        "aggressive",
        "high risk",
        "high conviction",
        "profit",
    )
    if any(keyword in normalized for keyword in conservative_keywords):
        return "conservative"
    if any(keyword in normalized for keyword in growth_keywords):
        return "growth"
    if any(keyword in normalized for keyword in balanced_keywords):
        return "balanced"
    return None


def get_investor_profile(name: InvestorProfileName) -> InvestorProfile:
    title_th, objective, risk_summary, rebalance_hint = _PROFILE_DETAILS[name]
    return InvestorProfile(
        name=name,
        title_th=title_th,
        objective=objective,
        risk_summary=risk_summary,
        rebalance_hint=rebalance_hint,
        base_allocations=dict(_BASE_ALLOCATIONS[name]),
    )


def build_portfolio_plan(
    *,
    asset_snapshots: Sequence[Mapping[str, object]],
    macro_context: Mapping[str, object] | None,
    profile_name: InvestorProfileName,
    asset_scope: str = "all",
) -> PortfolioPlan:
    profile = get_investor_profile(profile_name)
    regime = _infer_market_regime(asset_snapshots=asset_snapshots, macro_context=macro_context)
    target_allocations = _build_target_allocations(profile_name=profile_name, regime=regime)
    visible_categories = _categories_for_scope(asset_scope)

    buckets: list[AllocationBucket] = []
    for category in visible_categories:
        target_pct = target_allocations[category]
        base_pct = profile.base_allocations[category]
        if target_pct >= base_pct + 5:
            stance = "Overweight"
        elif target_pct <= base_pct - 5:
            stance = "Underweight"
        else:
            stance = "Neutral"
        preferred_assets = _pick_preferred_assets(category=category, asset_snapshots=asset_snapshots)
        buckets.append(
            AllocationBucket(
                category=category,
                label=_CATEGORY_LABELS[category],
                target_pct=target_pct,
                stance=stance,
                preferred_assets=preferred_assets,
                rationale=_build_bucket_rationale(
                    category=category,
                    regime=regime,
                    target_pct=target_pct,
                    preferred_assets=preferred_assets,
                ),
            )
        )

    return PortfolioPlan(
        profile=profile,
        market_regime=regime,
        regime_summary=_build_regime_summary(regime, macro_context),
        buckets=tuple(buckets),
        action_plan=_build_action_plan(regime, profile_name, asset_scope),
        risk_watch=_build_risk_watch(regime, macro_context),
    )


def _infer_market_regime(
    *,
    asset_snapshots: Sequence[Mapping[str, object]],
    macro_context: Mapping[str, object] | None,
) -> MarketRegime:
    uptrend_count = 0
    downtrend_count = 0
    for snapshot in asset_snapshots:
        trend = str(snapshot.get("trend") or "sideways")
        if trend == "uptrend":
            uptrend_count += 1
        elif trend == "downtrend":
            downtrend_count += 1

    vix = _as_float((macro_context or {}).get("vix"))
    yield_10y = _as_float((macro_context or {}).get("tnx"))
    cpi_yoy = _as_float((macro_context or {}).get("cpi_yoy"))

    if (vix is not None and vix >= 26.0) or downtrend_count >= max(2, uptrend_count + 1):
        return "risk_off"
    if (
        (vix is not None and vix <= 18.0)
        and uptrend_count >= max(2, downtrend_count + 1)
        and (yield_10y is None or yield_10y <= 4.4)
        and (cpi_yoy is None or cpi_yoy <= 3.3)
    ):
        return "risk_on"
    return "neutral"


def _build_target_allocations(
    *,
    profile_name: InvestorProfileName,
    regime: MarketRegime,
) -> dict[str, int]:
    allocations = dict(_BASE_ALLOCATIONS[profile_name])
    if regime == "risk_off":
        allocations["us_equity"] -= 15
        allocations["gold"] += 5
        allocations["bonds"] += 5
        allocations["cash"] += 5
    elif regime == "risk_on":
        allocations["us_equity"] += 10
        allocations["gold"] -= 3
        allocations["bonds"] -= 4
        allocations["cash"] -= 3
    return _normalize_allocations(allocations)


def _normalize_allocations(allocations: Mapping[str, int]) -> dict[str, int]:
    normalized = {key: max(0, int(value)) for key, value in allocations.items()}
    total = sum(normalized.values()) or 1
    scaled = {key: int(round((value / total) * 100)) for key, value in normalized.items()}
    delta = 100 - sum(scaled.values())
    if delta:
        priority = sorted(scaled, key=lambda key: scaled[key], reverse=(delta > 0))
        for key in priority:
            if delta == 0:
                break
            if delta < 0 and scaled[key] == 0:
                continue
            scaled[key] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1
    return scaled


def _categories_for_scope(asset_scope: str) -> tuple[str, ...]:
    mapping = {
        "gold-only": ("gold", "cash"),
        "us-stocks": ("us_equity", "cash", "gold"),
        "etf-only": ("us_equity", "gold", "bonds", "cash"),
        "bonds": ("bonds", "cash", "gold"),
        "all": ("us_equity", "gold", "bonds", "cash"),
    }
    return mapping.get(asset_scope, ("us_equity", "gold", "bonds", "cash"))


def _pick_preferred_assets(
    *,
    category: str,
    asset_snapshots: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    category_assets = set(_CATEGORY_ASSETS[category])
    ranked: list[tuple[float, str]] = []
    for snapshot in asset_snapshots:
        asset_name = str(snapshot.get("asset") or "")
        if asset_name not in category_assets:
            continue
        trend_score = _as_float(snapshot.get("trend_score")) or 0.0
        day_change_pct = _as_float(snapshot.get("day_change_pct")) or 0.0
        ranked.append((trend_score + (day_change_pct * 0.05), asset_name))
    ranked.sort(reverse=True)
    top_assets = tuple(asset for _, asset in ranked[:2] if asset)
    return top_assets or _CATEGORY_ASSETS[category][:1]


def _build_bucket_rationale(
    *,
    category: str,
    regime: MarketRegime,
    target_pct: int,
    preferred_assets: Sequence[str],
) -> str:
    if category == "cash":
        if regime == "risk_off":
            return f"กันเงินสดไว้ประมาณ {target_pct}% เพื่อรองรับความผันผวนและรอจังหวะสะสม"
        return f"ถือเงินสดสำรองประมาณ {target_pct}% เพื่อคุมความเสี่ยงและใช้เติมพอร์ตเมื่อมีโอกาส"

    preferred = ", ".join(preferred_assets) if preferred_assets else _CATEGORY_LABELS[category]
    if regime == "risk_off":
        tone = "ตลาดยังต้องระวัง จึงเน้นสินทรัพย์ที่ช่วยกันความเสี่ยงและลดจังหวะไล่ราคา"
    elif regime == "risk_on":
        tone = "โมเมนตัมรวมยังเอื้อต่อการเพิ่มน้ำหนักสินทรัพย์เสี่ยงทีละส่วน"
    else:
        tone = "ภาพรวมยังผสมกัน จึงควรถือสัดส่วนสมดุลและเลือกตัวที่แข็งแรงกว่าเพื่อน"
    return f"{tone} กลุ่มนี้ตั้งเป้า {target_pct}% และโฟกัส {preferred}"


def _build_regime_summary(regime: MarketRegime, macro_context: Mapping[str, object] | None) -> str:
    vix = _as_float((macro_context or {}).get("vix"))
    if regime == "risk_off":
        vix_text = f" โดย VIX อยู่แถว {vix:.2f}" if vix is not None else ""
        return f"ภาวะตลาดอยู่ในโหมดระวังความเสี่ยง{vix_text} จึงควรเน้นการปกป้องเงินต้นมากกว่าการไล่ผลตอบแทน"
    if regime == "risk_on":
        return "ภาวะตลาดเปิดรับความเสี่ยงได้มากขึ้น แต่ยังควรเพิ่มน้ำหนักแบบค่อยเป็นค่อยไปและไม่ลด hedge จนหมด"
    return "ภาวะตลาดยังเป็นกลางถึงผันผวนสลับกัน ควรกระจายพอร์ตและเน้นวินัยการรีบาลานซ์"


def _build_action_plan(regime: MarketRegime, profile_name: InvestorProfileName, asset_scope: str) -> str:
    if regime == "risk_off":
        return "ทยอยเพิ่มเงินสดและสินทรัพย์กันความเสี่ยง ลดการเปิดรับหุ้นใหม่จนกว่าตลาดจะยืนแนวรับสำคัญได้"
    if regime == "risk_on" and profile_name == "growth":
        return "เพิ่มน้ำหนักหุ้นหรือ ETF หลักเป็นไม้ย่อย โดยยังคงเงินสดสำรองบางส่วนเผื่อรีบาวด์หลอก"
    if asset_scope == "gold-only":
        return "ใช้ทองคำเป็น hedge หลักของพอร์ตและเพิ่มน้ำหนักเมื่อแนวโน้มยังเหนือแนวรับสำคัญ"
    return "คงพอร์ตแบบสมดุล ทยอยสะสมเฉพาะตัวที่แนวโน้มแข็งแรง และรีบาลานซ์เมื่อสัดส่วนเริ่มเบี่ยงมาก"


def _build_risk_watch(regime: MarketRegime, macro_context: Mapping[str, object] | None) -> str:
    vix = _as_float((macro_context or {}).get("vix"))
    yield_10y = _as_float((macro_context or {}).get("tnx"))
    cpi_yoy = _as_float((macro_context or {}).get("cpi_yoy"))
    parts: list[str] = []
    if vix is not None:
        parts.append(f"VIX {vix:.2f}")
    if yield_10y is not None:
        parts.append(f"US10Y {yield_10y:.2f}")
    if cpi_yoy is not None:
        parts.append(f"CPI YoY {cpi_yoy:.2f}%")
    macro_line = " | ".join(parts) if parts else "macro data จำกัด"
    if regime == "risk_off":
        return f"จับตา {macro_line} หากความผันผวนเร่งขึ้นอีกควรเพิ่มเงินสดและลดสินทรัพย์เสี่ยงลง"
    if regime == "risk_on":
        return f"แม้ภาพรวมบวกขึ้น แต่ยังต้องจับตา {macro_line} เพราะการกลับตัวของ bond yield หรือ VIX สามารถกด valuation ได้เร็ว"
    return f"สัญญาณมหภาคยังไม่ขาดจากกัน ควรติดตาม {macro_line} ควบคู่กับแนวรับของดัชนีหลัก"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
