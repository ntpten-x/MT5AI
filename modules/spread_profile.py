from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from config import Settings


def classify_trading_session(hour_utc: int, rollover_hours_utc: set[int]) -> str:
    if hour_utc in rollover_hours_utc:
        return "rollover"
    if 13 <= hour_utc < 16:
        return "overlap"
    if 7 <= hour_utc < 13:
        return "london"
    if 16 <= hour_utc < 22:
        return "new_york"
    return "tokyo"


def classify_market_regime(
    timestamp: datetime,
    rollover_hours_utc: set[int],
    weekend_open_days_utc: set[int],
    weekend_open_hours_utc: set[int],
) -> str:
    normalized = timestamp.astimezone(timezone.utc)
    if normalized.weekday() in weekend_open_days_utc and normalized.hour in weekend_open_hours_utc:
        return "weekend_open"
    if normalized.hour in rollover_hours_utc:
        return "rollover"
    return "weekday"


@dataclass(slots=True)
class SpreadBucket:
    label: str
    samples: int
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    maximum: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "samples": self.samples,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "maximum": self.maximum,
        }


@dataclass(slots=True)
class SpreadProfile:
    symbol: str
    timeframe: str
    generated_at: datetime
    rows: int
    regime_buckets: dict[str, SpreadBucket]
    session_buckets: dict[str, SpreadBucket]
    hour_buckets: dict[int, SpreadBucket]

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "rows": self.rows,
            "regime_buckets": {key: value.to_payload() for key, value in self.regime_buckets.items()},
            "session_buckets": {key: value.to_payload() for key, value in self.session_buckets.items()},
            "hour_buckets": {str(key): value.to_payload() for key, value in self.hour_buckets.items()},
        }


@dataclass(slots=True)
class SpreadLimitDecision:
    symbol: str
    base_regime_name: str
    runtime_regime_name: str
    session_name: str
    hour_utc: int
    soft_limit_points: int
    news_limit_points: int
    extreme_limit_points: int
    profile_ready: bool
    source: str

    def payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_regime_name": self.base_regime_name,
            "runtime_regime_name": self.runtime_regime_name,
            "session_name": self.session_name,
            "hour_utc": self.hour_utc,
            "soft_limit_points": self.soft_limit_points,
            "news_limit_points": self.news_limit_points,
            "extreme_limit_points": self.extreme_limit_points,
            "profile_ready": self.profile_ready,
            "source": self.source,
        }


class SpreadProfileManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._profiles: dict[tuple[str, str], SpreadProfile] = {}
        self.settings.spread_profile_cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_crypto_symbol(self, symbol: str) -> bool:
        return symbol.upper() in {item.upper() for item in self.settings.trading.crypto_symbols}

    def _spread_caps(self, symbol: str) -> tuple[int, int, int]:
        if self._is_crypto_symbol(symbol):
            return (
                int(self.settings.risk.crypto_spread_limit_points),
                int(self.settings.risk.crypto_news_spread_limit_points),
                int(self.settings.risk.crypto_extreme_spread_limit_points),
            )
        return (
            int(self.settings.risk.spread_limit_points),
            int(self.settings.risk.news_spread_limit_points),
            int(self.settings.risk.extreme_spread_limit_points),
        )

    def _symbol_floor(self, symbol: str, mapping: dict[str, float], fallback: int) -> float:
        configured = mapping.get(symbol.upper())
        if configured is not None:
            return float(configured)
        return float(fallback)

    def _enabled_for_symbol(self, symbol: str) -> bool:
        if not self.settings.spread_profile.enabled:
            return False
        configured = {item.upper() for item in self.settings.spread_profile.symbols}
        return symbol.upper() in configured

    def _bucket_from_series(self, label: str, values: pd.Series) -> SpreadBucket:
        clean = values.dropna().astype(float)
        return SpreadBucket(
            label=label,
            samples=int(clean.shape[0]),
            p50=float(clean.quantile(0.50)),
            p75=float(clean.quantile(0.75)),
            p90=float(clean.quantile(0.90)),
            p95=float(clean.quantile(0.95)),
            p99=float(clean.quantile(self.settings.spread_profile.extreme_quantile)),
            maximum=float(clean.max()),
        )

    def build_profile(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> SpreadProfile | None:
        if not self._enabled_for_symbol(symbol) or frame.empty or len(frame) < self.settings.spread_profile.min_rows:
            return None

        data = frame.copy()
        if "time" not in data.columns or "spread" not in data.columns:
            return None

        data["time"] = pd.to_datetime(data["time"], utc=True)
        data["hour_utc"] = data["time"].dt.hour
        rollover_hours = set(self.settings.spread_profile.rollover_hours_utc)
        weekend_open_days = set(self.settings.spread_profile.weekend_open_days_utc)
        weekend_open_hours = set(self.settings.spread_profile.weekend_open_hours_utc)
        data["session_name"] = data["hour_utc"].map(lambda hour: classify_trading_session(int(hour), rollover_hours))
        data["regime_name"] = data["time"].map(
            lambda timestamp: classify_market_regime(
                timestamp.to_pydatetime(),
                rollover_hours,
                weekend_open_days,
                weekend_open_hours,
            )
        )

        regime_buckets: dict[str, SpreadBucket] = {}
        for regime_name, group in data.groupby("regime_name"):
            regime_buckets[regime_name] = self._bucket_from_series(regime_name, group["spread"])

        session_buckets: dict[str, SpreadBucket] = {}
        for session_name, group in data.groupby("session_name"):
            session_buckets[session_name] = self._bucket_from_series(session_name, group["spread"])

        hour_buckets: dict[int, SpreadBucket] = {}
        for hour, group in data.groupby("hour_utc"):
            hour_buckets[int(hour)] = self._bucket_from_series(f"hour_{int(hour):02d}", group["spread"])

        return SpreadProfile(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            generated_at=datetime.now(timezone.utc),
            rows=int(len(data)),
            regime_buckets=regime_buckets,
            session_buckets=session_buckets,
            hour_buckets=hour_buckets,
        )

    def _read_disk_cache(self) -> dict[str, Any]:
        path = self.settings.spread_profile_cache_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read spread profile cache: {}", exc)
            return {}

    def _write_disk_cache(self) -> None:
        payload = {
            f"{symbol}:{timeframe}": profile.to_payload()
            for (symbol, timeframe), profile in self._profiles.items()
        }
        self.settings.spread_profile_cache_path.write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )

    def _deserialize_profile(self, payload: dict[str, Any]) -> SpreadProfile:
        regime_buckets = {
            key: SpreadBucket(
                label=value["label"],
                samples=int(value["samples"]),
                p50=float(value["p50"]),
                p75=float(value["p75"]),
                p90=float(value["p90"]),
                p95=float(value["p95"]),
                p99=float(value["p99"]),
                maximum=float(value["maximum"]),
            )
            for key, value in payload.get("regime_buckets", {}).items()
        }
        session_buckets = {
            key: SpreadBucket(
                label=value["label"],
                samples=int(value["samples"]),
                p50=float(value["p50"]),
                p75=float(value["p75"]),
                p90=float(value["p90"]),
                p95=float(value["p95"]),
                p99=float(value["p99"]),
                maximum=float(value["maximum"]),
            )
            for key, value in payload.get("session_buckets", {}).items()
        }
        hour_buckets = {
            int(key): SpreadBucket(
                label=value["label"],
                samples=int(value["samples"]),
                p50=float(value["p50"]),
                p75=float(value["p75"]),
                p90=float(value["p90"]),
                p95=float(value["p95"]),
                p99=float(value["p99"]),
                maximum=float(value["maximum"]),
            )
            for key, value in payload.get("hour_buckets", {}).items()
        }
        generated_at = datetime.fromisoformat(payload["generated_at"])
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        return SpreadProfile(
            symbol=str(payload["symbol"]).upper(),
            timeframe=str(payload["timeframe"]).upper(),
            generated_at=generated_at.astimezone(timezone.utc),
            rows=int(payload["rows"]),
            regime_buckets=regime_buckets,
            session_buckets=session_buckets,
            hour_buckets=hour_buckets,
        )

    def get_profile(
        self,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame | None = None,
    ) -> SpreadProfile | None:
        key = (symbol.upper(), timeframe.upper())
        if key in self._profiles:
            return self._profiles[key]

        cached = self._read_disk_cache()
        cached_payload = cached.get(f"{key[0]}:{key[1]}")
        if cached_payload:
            profile = self._deserialize_profile(cached_payload)
            age_seconds = (datetime.now(timezone.utc) - profile.generated_at).total_seconds()
            if age_seconds <= self.settings.spread_profile.stale_cache_max_seconds:
                self._profiles[key] = profile
                return profile

        if frame is None:
            return None

        profile = self.build_profile(symbol, timeframe, frame)
        if profile is not None:
            self._profiles[key] = profile
            self._write_disk_cache()
        return profile

    def refresh_symbol(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> dict[str, Any]:
        profile = self.build_profile(symbol, timeframe, frame)
        if profile is None:
            return {"status": "skipped", "reason": "insufficient_rows_or_symbol_disabled"}

        key = (symbol.upper(), timeframe.upper())
        self._profiles[key] = profile
        self._write_disk_cache()
        return {
            "status": "ready",
            "rows": profile.rows,
            "generated_at": profile.generated_at.isoformat(),
            "regimes": {
                name: {
                    "samples": bucket.samples,
                    "p95": round(bucket.p95, 2),
                    "p99": round(bucket.p99, 2),
                    "max": round(bucket.maximum, 2),
                }
                for name, bucket in profile.regime_buckets.items()
            },
            "sessions": {
                name: {
                    "samples": bucket.samples,
                    "p95": round(bucket.p95, 2),
                    "p99": round(bucket.p99, 2),
                    "max": round(bucket.maximum, 2),
                }
                for name, bucket in profile.session_buckets.items()
            },
        }

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        current_time: datetime,
        frame: pd.DataFrame | None = None,
        news_phase: str | None = None,
    ) -> SpreadLimitDecision:
        current_time = current_time.astimezone(timezone.utc)
        soft_cap, news_cap, extreme_cap = self._spread_caps(symbol)
        base_regime = classify_market_regime(
            current_time,
            set(self.settings.spread_profile.rollover_hours_utc),
            set(self.settings.spread_profile.weekend_open_days_utc),
            set(self.settings.spread_profile.weekend_open_hours_utc),
        )
        runtime_regime = (
            news_phase
            if news_phase
            in {
                "pre_news",
                "pre_news_close_only",
                "release_minute",
                "post_release_freeze",
                "post_news_cooldown",
                "post_news_reentry",
            }
            else base_regime
        )
        if not self._enabled_for_symbol(symbol):
            return SpreadLimitDecision(
                symbol=symbol,
                base_regime_name=base_regime,
                runtime_regime_name=runtime_regime,
                session_name="disabled",
                hour_utc=current_time.hour,
                soft_limit_points=soft_cap,
                news_limit_points=news_cap,
                extreme_limit_points=extreme_cap,
                profile_ready=False,
                source="global_defaults",
            )

        profile = self.get_profile(symbol, timeframe, frame=frame)
        session_name = classify_trading_session(
            current_time.hour, set(self.settings.spread_profile.rollover_hours_utc)
        )
        hour_utc = current_time.hour
        if profile is None:
            return SpreadLimitDecision(
                symbol=symbol,
                base_regime_name=base_regime,
                runtime_regime_name=runtime_regime,
                session_name=session_name,
                hour_utc=hour_utc,
                soft_limit_points=soft_cap,
                news_limit_points=news_cap,
                extreme_limit_points=extreme_cap,
                profile_ready=False,
                source="global_defaults",
            )

        regime_bucket = profile.regime_buckets.get(base_regime)
        session_bucket = profile.session_buckets.get(session_name)
        hour_bucket = profile.hour_buckets.get(hour_utc)
        source = "regime_session_profile"
        soft_candidates = [
            self._symbol_floor(
                symbol,
                self.settings.spread_profile.symbol_min_soft_limit_points,
                self.settings.spread_profile.min_soft_limit_points,
            )
        ]
        news_candidates = [
            self._symbol_floor(
                symbol,
                self.settings.spread_profile.symbol_min_news_limit_points,
                self.settings.spread_profile.min_news_limit_points,
            )
        ]
        extreme_candidates = [
            self._symbol_floor(
                symbol,
                self.settings.spread_profile.symbol_min_extreme_limit_points,
                self.settings.spread_profile.min_extreme_limit_points,
            )
        ]
        regime_buffer = self.settings.spread_profile.regime_buffer_points

        if base_regime == "weekend_open":
            regime_buffer += self.settings.spread_profile.weekend_open_extra_buffer_points
        elif base_regime == "rollover":
            regime_buffer += self.settings.spread_profile.rollover_extra_buffer_points

        if regime_bucket is not None:
            soft_candidates.append(regime_bucket.p95 + regime_buffer)
            news_candidates.append(regime_bucket.p75 + self.settings.spread_profile.news_buffer_points)
            extreme_candidates.append(regime_bucket.p99 + self.settings.spread_profile.extreme_buffer_points)

        if session_bucket is not None:
            soft_candidates.append(session_bucket.p95 + self.settings.spread_profile.session_buffer_points)
            news_candidates.append(session_bucket.p75 + self.settings.spread_profile.news_buffer_points)
            extreme_candidates.append(
                max(
                    session_bucket.p99 + self.settings.spread_profile.extreme_buffer_points,
                    session_bucket.p95 + self.settings.spread_profile.extreme_buffer_points,
                )
            )

        if hour_bucket is not None:
            source = "regime_session_hour_profile"
            soft_candidates.append(hour_bucket.p95 + self.settings.spread_profile.hour_buffer_points)
            news_candidates.append(hour_bucket.p75 + self.settings.spread_profile.news_buffer_points)
            extreme_candidates.append(hour_bucket.p99 + self.settings.spread_profile.extreme_buffer_points)

        soft_limit = min(
            soft_cap,
            int(round(max(soft_candidates))),
        )
        if runtime_regime == "pre_news":
            source = "pre_news_profile"
            tightened_soft = max(
                self.settings.spread_profile.min_soft_limit_points,
                int(round(max(news_candidates))) - self.settings.spread_profile.pre_news_extra_tightening_points,
            )
            soft_limit = min(soft_limit, tightened_soft)
        elif runtime_regime == "pre_news_close_only":
            source = "pre_news_close_only_profile"
            tightened_soft = max(
                self.settings.spread_profile.min_soft_limit_points,
                int(round(max(news_candidates)))
                - self.settings.spread_profile.pre_news_close_only_extra_tightening_points,
            )
            soft_limit = min(soft_limit, tightened_soft)
        elif runtime_regime == "release_minute":
            source = "release_minute_profile"
            tightened_soft = max(
                self.settings.spread_profile.min_soft_limit_points,
                int(round(max(news_candidates))) - self.settings.spread_profile.release_minute_extra_tightening_points,
            )
            soft_limit = min(soft_limit, tightened_soft)
        elif runtime_regime == "post_release_freeze":
            source = "post_release_freeze_profile"
            tightened_soft = max(
                self.settings.spread_profile.min_soft_limit_points,
                int(round(max(news_candidates)))
                - self.settings.spread_profile.post_release_freeze_extra_tightening_points,
            )
            soft_limit = min(soft_limit, tightened_soft)
        elif runtime_regime == "post_news_cooldown":
            source = "post_news_cooldown_profile"
            tightened_soft = max(
                self.settings.spread_profile.min_soft_limit_points,
                int(round(max(news_candidates)))
                - self.settings.spread_profile.post_news_cooldown_extra_tightening_points,
            )
            soft_limit = min(soft_limit, tightened_soft)
        elif runtime_regime == "post_news_reentry":
            source = "post_news_reentry_profile"
            tightened_soft = max(
                self.settings.spread_profile.min_soft_limit_points,
                int(round(max(news_candidates)))
                - self.settings.spread_profile.post_news_reentry_extra_tightening_points,
            )
            soft_limit = min(soft_limit, tightened_soft)
        news_limit = min(
            news_cap,
            int(round(max(news_candidates))),
        )
        extreme_limit = min(
            extreme_cap,
            int(round(max([soft_limit + 15, *extreme_candidates]))),
        )

        news_limit = min(news_limit, soft_limit)
        extreme_limit = max(extreme_limit, soft_limit + 10)
        return SpreadLimitDecision(
            symbol=symbol,
            base_regime_name=base_regime,
            runtime_regime_name=runtime_regime,
            session_name=session_name,
            hour_utc=hour_utc,
            soft_limit_points=soft_limit,
            news_limit_points=news_limit,
            extreme_limit_points=extreme_limit,
            profile_ready=True,
            source=source,
        )
