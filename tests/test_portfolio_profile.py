from __future__ import annotations

from invest_advisor_bot.analysis.portfolio_profile import (
    build_portfolio_plan,
    detect_investor_profile,
    get_investor_profile,
    normalize_profile_name,
)


def test_detect_investor_profile_handles_thai_and_alias_keywords() -> None:
    assert detect_investor_profile("ช่วยจัดพอร์ตแบบรักษาเงินต้น") == "conservative"
    assert detect_investor_profile("อยากได้พอร์ตเติบโต เสี่ยงสูงได้") == "growth"
    assert detect_investor_profile("ขอแบบสมดุลระยะยาว") == "balanced"
    assert normalize_profile_name("aggressive-growth", default="balanced") == "growth"


def test_build_portfolio_plan_prefers_defensive_mix_in_risk_off() -> None:
    plan = build_portfolio_plan(
        asset_snapshots=[
            {"asset": "spy_etf", "trend": "downtrend", "trend_score": -4.0, "day_change_pct": -1.2},
            {"asset": "qqq_etf", "trend": "downtrend", "trend_score": -4.4, "day_change_pct": -1.6},
            {"asset": "gld_etf", "trend": "uptrend", "trend_score": 3.1, "day_change_pct": 0.4},
            {"asset": "tlt_etf", "trend": "sideways", "trend_score": 0.6, "day_change_pct": 0.1},
        ],
        macro_context={"vix": 31.2, "tnx": 4.4, "cpi_yoy": 3.1},
        profile_name="conservative",
    )

    allocations = {bucket.category: bucket.target_pct for bucket in plan.buckets}

    assert plan.market_regime == "risk_off"
    assert allocations["us_equity"] < get_investor_profile("conservative").base_allocations["us_equity"]
    assert allocations["cash"] >= get_investor_profile("conservative").base_allocations["cash"]
