from __future__ import annotations

from datetime import datetime, timezone

from invest_advisor_bot.analysis.asset_ranking import rank_asset_snapshots
from invest_advisor_bot.analysis.news_impact import NewsImpact, score_news_impacts
from invest_advisor_bot.analysis.risk_score import calculate_risk_score
from invest_advisor_bot.analysis.technical_indicators import SupportResistanceLevels
from invest_advisor_bot.analysis.trend_engine import TrendAssessment
from invest_advisor_bot.bot.alert_state import AlertStateStore
from invest_advisor_bot.providers.news_client import NewsArticle


def _sample_trend(direction: str, score: float) -> TrendAssessment:
    return TrendAssessment(
        ticker="SPY",
        direction=direction,  # type: ignore[arg-type]
        score=score,
        current_price=500.0,
        ema_fast=498.0,
        ema_slow=492.0,
        ema_gap_pct=0.01,
        rsi=58.0,
        macd=1.0,
        macd_signal=0.5,
        macd_hist=0.5,
        support_resistance=SupportResistanceLevels(
            current_price=500.0,
            nearest_support=490.0,
            nearest_resistance=510.0,
            supports=[490.0],
            resistances=[510.0],
            rolling_support=488.0,
            rolling_resistance=512.0,
        ),
        reasons=["price_above_fast_and_slow_ema"],
    )


def test_news_impact_and_risk_score_detect_negative_macro_shift() -> None:
    articles = [
        NewsArticle(
            title="Markets Slide as Recession Fears Rise and Fed Warns on Inflation",
            link="https://example.com",
            source="Reuters",
            published_at=datetime.now(timezone.utc),
            summary=None,
            guid="1",
        )
    ]
    impacts = score_news_impacts(articles)
    assessment = calculate_risk_score(
        macro_context={"vix": 31.0, "tnx": 4.6, "cpi_yoy": 3.4},
        trends={
            "spy_etf": _sample_trend("downtrend", -4.0),
            "qqq_etf": _sample_trend("downtrend", -4.3),
            "gld_etf": _sample_trend("uptrend", 3.0),
        },
        news_impacts=impacts,
    )

    assert impacts
    assert impacts[0].sentiment == "negative"
    assert assessment.level in {"high", "severe"}
    assert assessment.score >= 6.0


def test_asset_ranking_prefers_positive_trend_assets() -> None:
    ranked = rank_asset_snapshots(
        [
            {"asset": "spy_etf", "label": "ETF SPY", "trend": "uptrend", "trend_score": 3.6, "day_change_pct": 1.1, "rsi": 60.0, "macd_hist": 0.7},
            {"asset": "qqq_etf", "label": "ETF QQQ", "trend": "downtrend", "trend_score": -3.8, "day_change_pct": -1.4, "rsi": 39.0, "macd_hist": -0.6},
        ]
    )

    assert ranked[0].asset == "spy_etf"
    assert ranked[0].stance == "watch-long"
    assert ranked[-1].stance in {"neutral", "avoid"}


def test_alert_state_store_filters_recent_duplicates(tmp_path) -> None:
    store = AlertStateStore(path=tmp_path / "alerts.json", suppression_minutes=180)

    first = store.filter_new_keys(["risk:1", "news:abc"])
    second = store.filter_new_keys(["risk:1", "news:abc", "rank:xyz"])

    assert first == ["risk:1", "news:abc"]
    assert second == ["rank:xyz"]
