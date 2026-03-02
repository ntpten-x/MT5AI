from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from config import Settings
from modules.news_filter import NewsEvent, NewsFilter


def _settings() -> Settings:
    return Settings(
        news={
            "enabled": True,
            "cache_ttl_seconds": 300,
            "stale_cache_max_seconds": 86_400,
            "lookahead_minutes": 180,
            "lookback_minutes": 90,
            "min_importance": 3,
            "fail_closed": False,
            "symbol_currency_map": {
                "GOLD": ["USD"],
                "XAUUSD": ["USD"],
                "BTCUSD": ["USD"],
                "ETHUSD": ["USD"],
                "EURUSD": ["EUR", "USD"],
            },
        },
        risk={
            "news_spread_limit_points": 100,
            "spread_limit_points": 150,
        },
    )


def test_news_guard_marks_release_minute_and_closes_positions():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            timestamp=now + timedelta(seconds=20),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is True
    assert decision.phase == "release_minute"
    assert decision.event_tag == "NFP_PAYROLLS"
    assert decision.event_family == "NFP"
    assert decision.strict_spread_limit_points == 38
    assert "release_minute:NFP_PAYROLLS:USD:Non-Farm Payrolls:High:+0m" == decision.reason


def test_news_guard_marks_pre_news_before_release():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="FOMC Statement",
            currency="USD",
            timestamp=now + timedelta(minutes=20),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is False
    assert decision.phase == "pre_news"
    assert decision.event_tag == "FOMC_STATEMENT"
    assert decision.event_family == "FOMC"
    assert decision.strict_spread_limit_points == 38
    assert decision.reason == "pre_news:FOMC_STATEMENT:USD:FOMC Statement:High:+20m"


def test_news_guard_marks_pre_news_close_only_before_release():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Core CPI m/m",
            currency="USD",
            timestamp=now + timedelta(minutes=4),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is True
    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "CPI_CORE"
    assert decision.event_family == "CPI"
    assert decision.strict_spread_limit_points == 40
    assert decision.reason == "pre_news_close_only:CPI_CORE:USD:Core CPI m/m:High:+4m"


def test_news_guard_marks_post_news_cooldown():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="CPI",
            currency="USD",
            timestamp=now - timedelta(minutes=7),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is False
    assert decision.phase == "post_news_cooldown"
    assert decision.event_tag == "CPI"
    assert decision.event_family == "CPI"
    assert decision.reason == "post_news_cooldown:CPI:USD:CPI:High:-7m"


def test_news_guard_marks_post_release_freeze():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="FOMC Statement",
            currency="USD",
            timestamp=now - timedelta(minutes=4),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is False
    assert decision.phase == "post_release_freeze"
    assert decision.event_tag == "FOMC_STATEMENT"
    assert decision.event_family == "FOMC"
    assert decision.reason == "post_release_freeze:FOMC_STATEMENT:USD:FOMC Statement:High:-4m"


def test_news_guard_marks_post_news_reentry():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            timestamp=now - timedelta(minutes=35),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is False
    assert decision.close_positions is False
    assert decision.phase == "post_news_reentry"
    assert decision.event_tag == "NFP_PAYROLLS"
    assert decision.event_family == "NFP"
    assert decision.strict_spread_limit_points == 38
    assert decision.reason == "post_news_reentry:NFP_PAYROLLS:USD:Non-Farm Payrolls:High:-35m"


def test_news_guard_watch_window_applies_strict_spread_without_blocking():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="ISM Manufacturing PMI",
            currency="USD",
            timestamp=now + timedelta(minutes=75),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is False
    assert decision.close_positions is False
    assert decision.phase == "watch"
    assert decision.event_tag == "ISM_MANUFACTURING"
    assert decision.event_family == "ISM"
    assert decision.strict_spread_limit_points == 60
    assert decision.reason == "news_watch:ISM_MANUFACTURING:USD:ISM Manufacturing PMI:High:+75m"


def test_news_guard_prefers_nfp_over_generic_event_at_same_time():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="ISM Manufacturing PMI",
            currency="USD",
            timestamp=now + timedelta(minutes=1),
            impact="High",
            importance=3,
        ),
        NewsEvent(
            title="Non-Farm Employment Change",
            currency="USD",
            timestamp=now + timedelta(minutes=1),
            impact="High",
            importance=3,
        ),
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.event_tag == "NFP_PAYROLLS"
    assert decision.event_family == "NFP"
    assert decision.phase == "release_minute"


def test_news_filter_infers_fx_symbol_currencies():
    news_filter = NewsFilter(_settings())
    assert news_filter.symbol_currencies("EURUSD") == ["EUR", "USD"]


def test_news_filter_uses_disk_cache_when_feed_unavailable(tmp_path):
    settings = Settings(
        news={
            "enabled": True,
            "cache_ttl_seconds": 300,
            "stale_cache_max_seconds": 86_400,
            "cache_path": str(tmp_path / "news_cache.json"),
            "lookahead_minutes": 180,
            "lookback_minutes": 90,
            "pre_news_block_minutes": 30,
            "release_window_before_minutes": 1,
            "release_window_after_minutes": 2,
            "pre_news_close_only_minutes": 5,
            "post_release_freeze_minutes": 3,
            "post_news_cooldown_minutes": 15,
            "post_news_reentry_minutes": 20,
            "pre_news_close_positions_minutes": 5,
            "min_importance": 3,
            "symbol_currency_map": {"GOLD": ["USD"]},
        },
        risk={"news_spread_limit_points": 100},
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._write_disk_cache(
        [
            NewsEvent(
                title="Non-Farm Payrolls",
                currency="USD",
                timestamp=now + timedelta(minutes=14),
                impact="High",
                importance=3,
            )
        ],
        fetched_at=now,
    )

    def _raise():
        raise RuntimeError("429")

    news_filter._fetch_events = _raise
    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.blocked is True
    assert decision.phase == "pre_news"
    assert decision.event_tag == "NFP_PAYROLLS"
    assert decision.event_family == "NFP"
    assert decision.reason == "pre_news:NFP_PAYROLLS:USD:Non-Farm Payrolls:High:+14m"


def test_news_filter_extends_cache_during_rate_limit_backoff(tmp_path):
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "cache_ttl_seconds": 5,
            "stale_cache_max_seconds": 86_400,
            "calendar_backoff_base_seconds": 300,
            "calendar_backoff_max_seconds": 600,
            "cache_path": str(tmp_path / "news_cache.json"),
            "lookahead_minutes": 180,
            "lookback_minutes": 90,
            "min_importance": 3,
            "symbol_currency_map": {"GOLD": ["USD"]},
        },
        risk={"news_spread_limit_points": 100},
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._write_disk_cache(
        [
            NewsEvent(
                title="Non-Farm Payrolls",
                currency="USD",
                timestamp=now + timedelta(minutes=14),
                impact="High",
                importance=3,
            )
        ],
        fetched_at=now,
    )
    calls = {"count": 0}

    def _raise():
        calls["count"] += 1
        raise RuntimeError("429")

    news_filter._fetch_events = _raise

    first = news_filter.evaluate("GOLD", now=now)
    second = news_filter.evaluate("GOLD", now=now + timedelta(seconds=60))

    assert first.blocked is True
    assert second.blocked is True
    assert calls["count"] == 1
    assert news_filter._calendar_backoff_until == now + timedelta(seconds=300)
    assert news_filter._cache_expires_at == now + timedelta(seconds=300)


def test_news_filter_rate_limit_warning_is_not_repeated_within_interval(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "cache_ttl_seconds": 5,
            "stale_cache_max_seconds": 86_400,
            "calendar_backoff_base_seconds": 300,
            "calendar_backoff_max_seconds": 600,
            "calendar_warning_interval_seconds": 900,
            "cache_path": str(tmp_path / "news_cache.json"),
            "lookahead_minutes": 180,
            "lookback_minutes": 90,
            "min_importance": 3,
            "symbol_currency_map": {"GOLD": ["USD"]},
        },
        risk={"news_spread_limit_points": 100},
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)
    news_filter._write_disk_cache(
        [
            NewsEvent(
                title="Non-Farm Payrolls",
                currency="USD",
                timestamp=now + timedelta(minutes=14),
                impact="High",
                importance=3,
            )
        ],
        fetched_at=now,
    )
    warnings: list[str] = []

    def _warn(message, *args):
        warnings.append(str(message).format(*args))

    def _raise():
        raise RuntimeError("429")

    monkeypatch.setattr("modules.news_filter.logger.warning", _warn)
    news_filter._fetch_events = _raise

    news_filter.evaluate("GOLD", now=now)
    news_filter.evaluate("GOLD", now=now + timedelta(seconds=30))

    assert len(warnings) == 1
    assert "disk cache" in warnings[0]


def test_news_guard_detects_powell_press_conference_as_more_strict_than_fomc():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="FOMC Press Conference",
            currency="USD",
            timestamp=now + timedelta(minutes=18),
            impact="High",
            importance=3,
        ),
        NewsEvent(
            title="Federal Funds Rate Decision",
            currency="USD",
            timestamp=now + timedelta(minutes=18),
            impact="High",
            importance=3,
        ),
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "POWELL_PRESSER"
    assert decision.event_family == "FOMC"
    assert decision.strict_spread_limit_points == 34
    assert decision.reason == "pre_news_close_only:POWELL_PRESSER:USD:FOMC Press Conference:High:+18m"


def test_news_guard_prefers_rate_decision_over_statement():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Federal Funds Rate Decision",
            currency="USD",
            timestamp=now + timedelta(minutes=14),
            impact="High",
            importance=3,
        ),
        NewsEvent(
            title="FOMC Statement",
            currency="USD",
            timestamp=now + timedelta(minutes=14),
            impact="High",
            importance=3,
        ),
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "FOMC_RATE_DECISION"
    assert decision.event_family == "FOMC"
    assert decision.strict_spread_limit_points == 36
    assert decision.reason == "pre_news_close_only:FOMC_RATE_DECISION:USD:Federal Funds Rate Decision:High:+14m"


def test_news_guard_distinguishes_fomc_minutes_from_statement():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="FOMC Meeting Minutes",
            currency="USD",
            timestamp=now + timedelta(minutes=9),
            impact="High",
            importance=3,
        ),
        NewsEvent(
            title="FOMC Statement",
            currency="USD",
            timestamp=now + timedelta(minutes=9),
            impact="High",
            importance=3,
        ),
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "FOMC_STATEMENT"
    assert decision.event_family == "FOMC"
    assert decision.strict_spread_limit_points == 38
    assert decision.reason == "pre_news_close_only:FOMC_STATEMENT:USD:FOMC Statement:High:+9m"


def test_news_guard_matches_fomc_minutes_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="FOMC Meeting Minutes",
            currency="USD",
            timestamp=now + timedelta(minutes=9),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "FOMC_MINUTES"
    assert decision.event_family == "FOMC"
    assert decision.strict_spread_limit_points == 39
    assert decision.reason == "pre_news_close_only:FOMC_MINUTES:USD:FOMC Meeting Minutes:High:+9m"


def test_news_guard_distinguishes_cpi_headline_from_core():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="CPI y/y",
            currency="USD",
            timestamp=now + timedelta(minutes=6),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "CPI_YY"
    assert decision.event_family == "CPI"
    assert decision.strict_spread_limit_points == 41
    assert decision.reason == "pre_news_close_only:CPI_YY:USD:CPI y/y:High:+6m"


def test_news_guard_matches_cpi_mm_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="CPI m/m",
            currency="USD",
            timestamp=now + timedelta(minutes=7),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "CPI_MM"
    assert decision.event_family == "CPI"
    assert decision.strict_spread_limit_points == 40
    assert decision.reason == "pre_news_close_only:CPI_MM:USD:CPI m/m:High:+7m"


def test_news_guard_matches_core_pce_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Core PCE Price Index m/m",
            currency="USD",
            timestamp=now + timedelta(minutes=8),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "CORE_PCE"
    assert decision.event_family == "CPI"
    assert decision.strict_spread_limit_points == 39
    assert decision.reason == "pre_news_close_only:CORE_PCE:USD:Core PCE Price Index m/m:High:+8m"


def test_news_guard_matches_ppi_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Producer Price Index m/m",
            currency="USD",
            timestamp=now + timedelta(minutes=5),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "PPI"
    assert decision.event_family == "PPI"
    assert decision.strict_spread_limit_points == 50
    assert decision.reason == "pre_news_close_only:PPI:USD:Producer Price Index m/m:High:+5m"


def test_news_guard_matches_retail_sales_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Retail Sales m/m",
            currency="USD",
            timestamp=now + timedelta(minutes=4),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "RETAIL_SALES"
    assert decision.event_family == "RETAIL_SALES"
    assert decision.strict_spread_limit_points == 54
    assert decision.reason == "pre_news_close_only:RETAIL_SALES:USD:Retail Sales m/m:High:+4m"


def test_news_guard_matches_ism_services_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 3, 14, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="ISM Services PMI",
            currency="USD",
            timestamp=now + timedelta(minutes=4),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "ISM_SERVICES"
    assert decision.event_family == "ISM"
    assert decision.strict_spread_limit_points == 58
    assert decision.reason == "pre_news_close_only:ISM_SERVICES:USD:ISM Services PMI:High:+4m"


def test_news_guard_matches_adp_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="ADP Non-Farm Employment Change",
            currency="USD",
            timestamp=now + timedelta(minutes=4),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "ADP_EMPLOYMENT"
    assert decision.event_family == "LABOR"
    assert decision.strict_spread_limit_points == 58
    assert decision.reason == "pre_news_close_only:ADP_EMPLOYMENT:USD:ADP Non-Farm Employment Change:High:+4m"


def test_news_guard_matches_jolts_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 7, 14, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="JOLTS Job Openings",
            currency="USD",
            timestamp=now + timedelta(minutes=3),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "JOLTS_JOB_OPENINGS"
    assert decision.event_family == "LABOR"
    assert decision.strict_spread_limit_points == 60
    assert decision.reason == "pre_news_close_only:JOLTS_JOB_OPENINGS:USD:JOLTS Job Openings:High:+3m"


def test_news_guard_matches_initial_jobless_claims_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Initial Jobless Claims",
            currency="USD",
            timestamp=now + timedelta(minutes=3),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "INITIAL_JOBLESS_CLAIMS"
    assert decision.event_family == "LABOR"
    assert decision.strict_spread_limit_points == 62
    assert decision.reason == "pre_news_close_only:INITIAL_JOBLESS_CLAIMS:USD:Initial Jobless Claims:High:+3m"


def test_news_guard_matches_consumer_sentiment_subtype():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 4, 10, 14, 0, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Michigan Consumer Sentiment",
            currency="USD",
            timestamp=now + timedelta(minutes=3),
            impact="High",
            importance=3,
        )
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "pre_news_close_only"
    assert decision.event_tag == "CONSUMER_SENTIMENT"
    assert decision.event_family == "SENTIMENT"
    assert decision.strict_spread_limit_points == 64
    assert decision.reason == "pre_news_close_only:CONSUMER_SENTIMENT:USD:Michigan Consumer Sentiment:High:+3m"


def test_news_guard_distinguishes_nfp_companion_releases():
    settings = _settings()
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc)
    news_filter._cached_events = [
        NewsEvent(
            title="Average Hourly Earnings m/m",
            currency="USD",
            timestamp=now + timedelta(seconds=15),
            impact="High",
            importance=3,
        ),
        NewsEvent(
            title="Unemployment Rate",
            currency="USD",
            timestamp=now + timedelta(seconds=15),
            impact="High",
            importance=3,
        ),
    ]
    news_filter._cache_expires_at = now + timedelta(minutes=5)

    decision = news_filter.evaluate("GOLD", now=now)

    assert decision.phase == "release_minute"
    assert decision.event_tag == "NFP_WAGES"
    assert decision.event_family == "NFP"
    assert decision.strict_spread_limit_points == 39
    assert decision.reason == "release_minute:NFP_WAGES:USD:Average Hourly Earnings m/m:High:+0m"


def test_cryptopanic_payload_uses_current_developer_v2_endpoint_and_symbol_scoped_params(monkeypatch):
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "cryptopanic_enabled": True,
            "cryptopanic_api_url": "https://cryptopanic.com/api/developer/v2/posts/",
            "cryptopanic_api_key": "token",
            "cryptopanic_public": True,
            "cryptopanic_filter": "important",
            "cryptopanic_kind": "news",
            "cryptopanic_regions": "en",
        },
    )
    news_filter = NewsFilter(settings)
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"title": "ETH upgrade sentiment stable", "sentiment": "bullish", "votes": 80}]}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("modules.news_filter.requests.get", _fake_get)

    payload = news_filter._cryptopanic_payload("ETHUSD")

    assert isinstance(payload, list)
    assert captured["url"] == "https://cryptopanic.com/api/developer/v2/posts/"
    assert captured["params"] == {
        "auth_token": "token",
        "currencies": "ETH,BTC",
        "public": "true",
        "regions": "en",
        "filter": "important",
        "kind": "news",
    }


def test_cryptopanic_payload_raises_clear_error_on_404(monkeypatch):
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "cryptopanic_enabled": True,
            "cryptopanic_api_url": "https://cryptopanic.com/api/developer/v2/posts/",
            "cryptopanic_api_key": "token",
        },
    )
    news_filter = NewsFilter(settings)

    class _Response:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("raise_for_status should not be reached for 404 shortcut")

    monkeypatch.setattr("modules.news_filter.requests.get", lambda *args, **kwargs: _Response())

    try:
        news_filter._cryptopanic_payload("BTCUSD")
    except RuntimeError as exc:
        assert "endpoint was not found" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for 404 CryptoPanic response")


def test_cryptopanic_sentiment_derived_from_votes_payload():
    settings = Settings(_env_file=None)
    news_filter = NewsFilter(settings)

    payload = {"votes": {"positive": 1, "negative": 5}}

    assert news_filter._extract_sentiment_score(payload) == 17
    assert news_filter._extract_sentiment_label(payload) == "very_bearish"


def test_cryptopanic_payload_raises_clear_error_on_401(monkeypatch):
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "cryptopanic_enabled": True,
            "cryptopanic_api_url": "https://cryptopanic.com/api/developer/v2/posts/",
            "cryptopanic_api_key": "bad-token",
        },
    )
    news_filter = NewsFilter(settings)

    class _Response:
        status_code = 401

    monkeypatch.setattr("modules.news_filter.requests.get", lambda *args, **kwargs: _Response())

    try:
        news_filter._cryptopanic_payload("BTCUSD")
    except RuntimeError as exc:
        assert "rejected auth_token" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for 401 CryptoPanic response")


def test_crypto_keyword_guard_uses_description_text():
    settings = Settings(
        _env_file=None,
        trading={"crypto_symbols": ["BTCUSD"]},
        news={
            "enabled": True,
            "cryptopanic_enabled": True,
            "cryptopanic_api_key": "token",
            "crypto_keyword_guard_terms": ["SCAM"],
        },
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    news_filter._cryptopanic_payload = lambda symbol: [
        {
            "title": "Exchange issue update",
            "description": "Investigators warn users about a scam campaign",
            "votes": {"negative": 5, "positive": 0},
            "published_at": now.isoformat(),
            "instruments": [{"code": "BTC"}],
        }
    ]

    decision = news_filter.evaluate("BTCUSD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is True
    assert decision.phase == "crypto_keyword_guard"


def test_crypto_keyword_guard_uses_content_text():
    settings = Settings(
        _env_file=None,
        trading={"crypto_symbols": ["BTCUSD"]},
        news={
            "enabled": True,
            "cryptopanic_enabled": True,
            "cryptopanic_api_key": "token",
            "crypto_keyword_guard_terms": ["HACK"],
        },
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    news_filter._cryptopanic_payload = lambda symbol: [
        {
            "title": "Exchange update",
            "content": {"clean": "Major hack exposed hot wallet losses"},
            "votes": {"negative": 5, "positive": 0},
            "published_at": now.isoformat(),
            "instruments": [{"code": "BTC"}],
        }
    ]

    decision = news_filter.evaluate("BTCUSD", now=now)

    assert decision.blocked is True
    assert decision.close_positions is True
    assert decision.phase == "crypto_keyword_guard"


def test_calendar_warning_deduplicates_by_key(monkeypatch):
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "calendar_warning_interval_seconds": 900,
        },
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 3, 0, tzinfo=timezone.utc)
    captured: list[str] = []

    monkeypatch.setattr("modules.news_filter.logger.warning", lambda message, *args, **kwargs: captured.append(str(message)))

    news_filter._log_calendar_warning_once(
        now,
        "Calendar feed backoff active; using cached events for another 299s",
        key="calendar_backoff_active",
    )
    news_filter._log_calendar_warning_once(
        now + timedelta(seconds=30),
        "Calendar feed backoff active; using cached events for another 269s",
        key="calendar_backoff_active",
    )
    news_filter._log_calendar_warning_once(
        now + timedelta(seconds=30),
        "News fetch failed, using disk cache: 429 Too Many Requests",
        key="calendar_fetch_disk_429",
    )

    assert captured == [
        "Calendar feed backoff active; using cached events for another 299s",
        "News fetch failed, using disk cache: 429 Too Many Requests",
    ]


def test_calendar_backoff_uses_retry_after_header():
    settings = Settings(
        _env_file=None,
        news={
            "enabled": True,
            "calendar_backoff_base_seconds": 300,
            "calendar_backoff_max_seconds": 3600,
        },
    )
    news_filter = NewsFilter(settings)
    now = datetime(2026, 3, 2, 3, 0, tzinfo=timezone.utc)

    request = httpx.Request("GET", settings.news.calendar_url)
    response = httpx.Response(429, headers={"Retry-After": "1800"}, request=request)
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    news_filter._apply_calendar_backoff(now, exc)

    assert news_filter._calendar_backoff_seconds == 1800
    assert news_filter._calendar_backoff_until == now + timedelta(seconds=1800)


def test_internet_available_uses_cache_during_backoff(monkeypatch):
    settings = Settings(_env_file=None, news={"enabled": True})
    news_filter = NewsFilter(settings)
    news_filter._calendar_backoff_until = datetime.now(timezone.utc) + timedelta(minutes=10)

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("requests.get should not be called while backoff is active")

    monkeypatch.setattr("modules.news_filter.requests.get", _unexpected_request)

    assert news_filter.internet_available() is True
