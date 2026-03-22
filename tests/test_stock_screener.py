from __future__ import annotations

from datetime import datetime, timezone

from invest_advisor_bot.analysis.stock_screener import rank_stock_universe
from invest_advisor_bot.analysis.technical_indicators import SupportResistanceLevels
from invest_advisor_bot.analysis.trend_engine import TrendAssessment
from invest_advisor_bot.providers.market_data_client import AssetQuote, StockFundamentals
from invest_advisor_bot.universe import StockUniverseMember


def test_rank_stock_universe_prefers_quality_uptrend_names() -> None:
    universe = {
        "nvda": StockUniverseMember("NVDA", "NVIDIA", "Technology", "nasdaq100"),
        "xom": StockUniverseMember("XOM", "Exxon Mobil", "Energy", "sp500"),
        "dis": StockUniverseMember("DIS", "Disney", "Communication Services", "sp500"),
    }
    quotes = {
        asset: AssetQuote(
            ticker=member.ticker,
            name=member.company_name,
            currency="USD",
            exchange="TEST",
            price=100.0,
            previous_close=98.0,
            open_price=99.0,
            day_high=101.0,
            day_low=97.0,
            volume=1_000_000,
            timestamp=datetime.now(timezone.utc),
        )
        for asset, member in universe.items()
    }
    trends = {
        "nvda": _trend("uptrend", 4.0, 62.0),
        "xom": _trend("uptrend", 2.2, 55.0),
        "dis": _trend("downtrend", -1.8, 39.0),
    }
    fundamentals = {
        "nvda": _fundamentals("NVDA", revenue_growth=0.22, earnings_growth=0.28, roe=0.38, debt=35.0, fwd_pe=30.0),
        "xom": _fundamentals("XOM", revenue_growth=0.05, earnings_growth=0.07, roe=0.18, debt=25.0, fwd_pe=14.0),
        "dis": _fundamentals("DIS", revenue_growth=-0.03, earnings_growth=-0.02, roe=0.06, debt=190.0, fwd_pe=45.0),
    }

    ranked = rank_stock_universe(
        stock_universe=universe,
        quotes=quotes,
        trends=trends,
        fundamentals=fundamentals,
        top_k=3,
    )

    assert ranked[0].ticker == "NVDA"
    assert ranked[0].stance in {"buy", "watch"}
    assert ranked[-1].ticker != "DIS" or ranked[-1].stance == "avoid"


def _trend(direction: str, score: float, rsi: float) -> TrendAssessment:
    return TrendAssessment(
        ticker="TEST",
        direction=direction,  # type: ignore[arg-type]
        score=score,
        current_price=100.0,
        ema_fast=99.0,
        ema_slow=95.0,
        ema_gap_pct=0.03,
        rsi=rsi,
        macd=1.0,
        macd_signal=0.6,
        macd_hist=0.4,
        support_resistance=SupportResistanceLevels(
            current_price=100.0,
            nearest_support=95.0,
            nearest_resistance=105.0,
            supports=[95.0],
            resistances=[105.0],
            rolling_support=94.0,
            rolling_resistance=106.0,
        ),
        reasons=["price_above_fast_and_slow_ema"],
    )


def _fundamentals(
    ticker: str,
    *,
    revenue_growth: float,
    earnings_growth: float,
    roe: float,
    debt: float,
    fwd_pe: float,
) -> StockFundamentals:
    return StockFundamentals(
        ticker=ticker,
        company_name=ticker,
        sector="Technology",
        industry="Software",
        market_cap=100_000_000_000.0,
        trailing_pe=fwd_pe + 3.0,
        forward_pe=fwd_pe,
        price_to_book=6.0,
        dividend_yield=None,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        profit_margin=0.22,
        operating_margin=0.25,
        return_on_equity=roe,
        debt_to_equity=debt,
        analyst_target_price=115.0,
    )
