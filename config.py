from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent


def _parse_csv(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _parse_mapping(value):
    if not isinstance(value, str):
        return value

    mapping: dict[str, list[str]] = {}
    for chunk in value.split(";"):
        item = chunk.strip()
        if not item:
            continue
        key, separator, raw_values = item.partition(":")
        if not separator:
            continue
        values = [part.strip().upper() for part in raw_values.split(",") if part.strip()]
        if values:
            mapping[key.strip().upper()] = values
    return mapping


def _parse_string_mapping(value):
    if not isinstance(value, str):
        return value

    mapping: dict[str, str] = {}
    for chunk in value.split(";"):
        item = chunk.strip()
        if not item:
            continue
        key, separator, raw_value = item.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().upper()
        normalized_value = raw_value.strip()
        if normalized_key and normalized_value:
            mapping[normalized_key] = normalized_value
    return mapping


def _parse_int_mapping(value):
    if not isinstance(value, str):
        return value

    mapping: dict[str, list[int]] = {}
    for chunk in value.split(";"):
        item = chunk.strip()
        if not item:
            continue
        key, separator, raw_values = item.partition(":")
        if not separator:
            continue
        values: list[int] = []
        for part in raw_values.split(","):
            stripped = part.strip()
            if not stripped:
                continue
            values.append(int(stripped))
        if values:
            mapping[key.strip().upper()] = values
    return mapping


def _parse_float_mapping(value):
    if not isinstance(value, str):
        return value

    mapping: dict[str, float] = {}
    for chunk in value.split(";"):
        item = chunk.strip()
        if not item:
            continue
        key, separator, raw_value = item.partition(":")
        if not separator:
            continue
        try:
            mapping[key.strip().upper()] = float(raw_value.strip())
        except ValueError:
            continue
    return mapping


class MT5Settings(BaseModel):
    login: int = 0
    password: str = ""
    server: str = ""
    terminal_path: str | None = None
    connect_timeout_ms: int = 60_000


class TradingSettings(BaseModel):
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["XAUUSD", "BTCUSD", "ETHUSD"]
    )
    timeframe: str = "M15"
    confirmation_timeframes: Annotated[list[str], NoDecode] = Field(default_factory=list)
    confirmation_alignment_threshold: float = 0.5
    history_bars: int = 5_000
    train_bars: int = 15_000
    max_positions_per_symbol: int = 1
    allowed_sessions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["london", "new_york"]
    )
    symbol_allowed_sessions: Annotated[dict[str, list[str]], NoDecode] = Field(
        default_factory=lambda: {
            "XAUUSD": ["LONDON", "NEW_YORK"],
            "BTCUSD": ["ALL"],
            "ETHUSD": ["ALL"],
        }
    )
    min_trade_interval_seconds: int = 300
    allow_long: bool = True
    allow_short: bool = True
    allowed_hours_utc: Annotated[list[int], NoDecode] = Field(default_factory=lambda: list(range(7, 17)))
    symbol_allowed_hours_utc: Annotated[dict[str, list[int]], NoDecode] = Field(
        default_factory=lambda: {
            "XAUUSD": list(range(7, 17)),
            "BTCUSD": list(range(24)),
            "ETHUSD": list(range(24)),
        }
    )
    crypto_symbols: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["BTCUSD", "ETHUSD"])
    symbol_aliases: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)
    auto_discover_broker_symbols: bool = True
    dxy_symbol: str = "DXY"

    @field_validator("symbols", "allowed_sessions", "crypto_symbols", "confirmation_timeframes", mode="before")
    @classmethod
    def parse_csv(cls, value):
        return _parse_csv(value)

    @field_validator("symbol_aliases", mode="before")
    @classmethod
    def parse_symbol_aliases(cls, value):
        return _parse_string_mapping(value)

    @field_validator("allowed_hours_utc", mode="before")
    @classmethod
    def parse_int_csv(cls, value):
        parsed = _parse_csv(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
        return value

    @field_validator("symbol_allowed_sessions", mode="before")
    @classmethod
    def parse_symbol_allowed_sessions(cls, value):
        return _parse_mapping(value)

    @field_validator("symbol_allowed_hours_utc", mode="before")
    @classmethod
    def parse_int_mapping(cls, value):
        return _parse_int_mapping(value)

    @field_validator("timeframe")
    @classmethod
    def normalize_timeframe(cls, value: str) -> str:
        return value.upper()

    @field_validator("confirmation_timeframes")
    @classmethod
    def normalize_confirmation_timeframes(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value or []:
            timeframe = str(item).strip().upper()
            if not timeframe or timeframe in seen:
                continue
            normalized.append(timeframe)
            seen.add(timeframe)
        return normalized


class RiskSettings(BaseModel):
    risk_per_trade: float = 0.002  # 0.2%
    adaptive_risk_enabled: bool = True
    adaptive_risk_min: float = 0.002
    adaptive_risk_max: float = 0.010
    adaptive_risk_confidence_floor: float = 0.75
    adaptive_risk_confidence_ceiling: float = 1.20
    adaptive_risk_mtf_boost: float = 0.20
    adaptive_risk_drawdown_floor: float = 0.50
    daily_drawdown_limit: float = 0.002  # 0.2% daily hard stop
    daily_equity_max_drawdown_pct: float = 0.002  # 0.2% equity high-watermark guard
    max_consecutive_losses: int = 1  # Stop after 1 consecutive loss
    spread_limit_points: int = 150
    news_spread_limit_points: int = 100
    extreme_spread_limit_points: int = 220
    crypto_spread_limit_points: int = 6_000
    crypto_news_spread_limit_points: int = 6_000
    crypto_extreme_spread_limit_points: int = 9_000
    symbol_risk_per_trade: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    close_positions_on_extreme_spread: bool = True
    atr_stop_loss_mult: float = 1.5  # Tighter SL for sniper strategy
    atr_take_profit_mult: float = 1.0  # Tighter TP for sniper strategy
    crypto_atr_stop_loss_mult: float = 2.0
    crypto_atr_take_profit_mult: float = 3.0
    symbol_atr_stop_loss_mult: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_atr_take_profit_mult: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    min_reward_risk: float = 1.5
    auto_breakeven_enable: bool = True  # Move to breakeven when profit reaches 1:1
    auto_breakeven_rr_default: float = 1.0
    auto_breakeven_rr_crypto: float = 1.5
    time_exit_bars: int = 5  # Close position after 5 bars if no significant movement
    time_exit_min_progress_rr: float = 0.3
    trailing_trigger_rr: float = 0.0
    trailing_lock_rr: float = 0.0
    symbol_auto_breakeven_rr: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_time_exit_bars: Annotated[dict[str, int], NoDecode] = Field(default_factory=dict)
    symbol_time_exit_min_progress_rr: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_trailing_trigger_rr: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_trailing_lock_rr: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)

    @field_validator(
        "symbol_risk_per_trade",
        "symbol_atr_stop_loss_mult",
        "symbol_atr_take_profit_mult",
        "symbol_auto_breakeven_rr",
        "symbol_time_exit_min_progress_rr",
        "symbol_trailing_trigger_rr",
        "symbol_trailing_lock_rr",
        mode="before",
    )
    @classmethod
    def parse_symbol_risk_per_trade(cls, value):
        return _parse_float_mapping(value)

    @field_validator("symbol_time_exit_bars", mode="before")
    @classmethod
    def parse_symbol_time_exit_bars(cls, value):
        parsed = _parse_int_mapping(value)
        if not isinstance(parsed, dict):
            return value
        normalized: dict[str, int] = {}
        for key, values in parsed.items():
            if isinstance(values, list):
                if values:
                    normalized[key] = int(values[0])
            elif values is not None:
                normalized[key] = int(values)
        return normalized


class ExecutionSettings(BaseModel):
    dry_run: bool = True
    deviation_points: int = 20
    max_retries: int = 3
    retry_delay_seconds: float = 1.5
    fee_bps: float = 0.5
    slippage_bps: float = 1.0
    close_all_on_guard_trip: bool = True
    comment_prefix: str = "MT5AI"


class ModelSettings(BaseModel):
    xgb_artifact: str = "models/{symbol}/xgb_signal_model_{side}.joblib"
    lgbm_artifact: str = "models/{symbol}/lgbm_signal_model_{side}.joblib"
    sequence_artifact: str = "models/{symbol}/torch_sequence_model_{side}.pt"
    transformer_artifact: str = "models/{symbol}/torch_transformer_model_{side}.pt"
    optimization_profile: str = "models/{symbol}/optuna_profile.json"
    xgb_probability_threshold: float = 0.70  # Higher threshold for sniper strategy
    probability_dominance_margin: float = 0.30  # (Long_Prob - Short_Prob) > 0.30
    lightgbm_enabled: bool = False
    lightgbm_probability_weight: float = 0.20
    symbol_long_thresholds: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_short_thresholds: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_probability_dominance_margins: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_lightgbm_probability_weights: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_sequence_probability_weights: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_transformer_probability_weights: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    xgb_threshold_candidates: Annotated[list[float], NoDecode] = Field(
        default_factory=lambda: [0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.68, 0.7]
    )
    positive_return_horizon: int = 3
    positive_return_bps: int = 4
    training_min_rows: int = 800
    training_ratio: float = 0.7
    calibration_ratio: float = 0.15
    calibration_method: str = "auto"
    tuning_min_trades: int = 2
    chronos_enabled: bool = False
    chronos_model_id: str = "amazon/chronos-2"
    chronos_device: str = "cpu"
    chronos_prediction_length: int = 4
    chronos_deadband: float = 0.0003
    chronos_probability_weight: float = 0.10
    chronos_selective_enabled: bool = True
    chronos_allowed_sessions: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["london", "new_york"])
    chronos_min_atr_pct: float = 0.0005
    chronos_max_atr_pct: float = 0.004
    chronos_max_abs_ema_trend_strength: float = 1.25
    chronos_max_spread_pct: float = 0.002
    sequence_enabled: bool = False
    sequence_device: str = "cpu"
    sequence_length: int = 48
    sequence_hidden_size: int = 64
    sequence_num_layers: int = 2
    sequence_dropout: float = 0.1
    sequence_epochs: int = 8
    sequence_batch_size: int = 64
    sequence_learning_rate: float = 0.001
    sequence_probability_weight: float = 0.35
    transformer_enabled: bool = False
    transformer_heads: int = 4
    transformer_ffn_size: int = 128
    transformer_probability_weight: float = 0.20
    optuna_enabled: bool = True
    optuna_trials: int = 30
    optuna_timeout_seconds: int = 300
    optuna_n_startup_trials: int = 10
    optuna_risk_min: float = 0.001
    optuna_risk_max: float = 0.01
    optuna_sl_min: float = 1.0
    optuna_sl_max: float = 4.0
    optuna_tp_min: float = 1.0
    optuna_tp_max: float = 6.0
    optuna_lightgbm_weight_max: float = 0.4
    optuna_sequence_weight_max: float = 0.6
    optuna_transformer_weight_max: float = 0.4
    walk_forward_splits: int = 5
    walk_forward_min_positive_return_folds: int = 2
    walk_forward_min_positive_return_pct: float = 0.0
    volatility_breakout_pct: float = 0.005
    feature_selection_enabled: bool = True
    feature_selection_min_features: int = 12
    feature_selection_max_features: int = 24
    feature_selection_correlation_threshold: float = 0.95
    feature_selection_min_mutual_info: float = 0.0
    feature_selection_forced_features: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["ema_fast", "ema_slow", "rsi_14", "atr_14", "macd_hist"]
    )

    @field_validator(
        "symbol_long_thresholds",
        "symbol_short_thresholds",
        "symbol_probability_dominance_margins",
        "symbol_lightgbm_probability_weights",
        "symbol_sequence_probability_weights",
        "symbol_transformer_probability_weights",
        mode="before",
    )
    @classmethod
    def parse_float_mapping(cls, value):
        return _parse_float_mapping(value)

    @field_validator("xgb_threshold_candidates", mode="before")
    @classmethod
    def parse_float_csv(cls, value):
        parsed = _parse_csv(value)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
        return value

    @field_validator("chronos_allowed_sessions", "feature_selection_forced_features", mode="before")
    @classmethod
    def parse_text_csv(cls, value):
        parsed = _parse_csv(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return value


class DatabaseSettings(BaseModel):
    path: str = "database/trade_history.db"


class TelegramSettings(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class NewsSettings(BaseModel):
    enabled: bool = False
    calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 1_800
    stale_cache_max_seconds: int = 86_400
    calendar_backoff_base_seconds: int = 300
    calendar_backoff_max_seconds: int = 3_600
    calendar_warning_interval_seconds: int = 900
    cache_path: str = "data/news_calendar_cache.json"
    lookahead_minutes: int = 180
    lookback_minutes: int = 30
    block_before_minutes: int = 30
    block_after_minutes: int = 15
    close_positions_before_minutes: int = 5
    pre_news_block_minutes: int = 30
    release_window_before_minutes: int = 1
    release_window_after_minutes: int = 2
    pre_news_close_only_minutes: int = 5
    post_release_freeze_minutes: int = 3
    post_news_cooldown_minutes: int = 15
    post_news_reentry_minutes: int = 20
    pre_news_close_positions_minutes: int = 5
    min_importance: int = 3
    fail_closed: bool = False
    cryptopanic_enabled: bool = False
    cryptopanic_api_url: str = "https://cryptopanic.com/api/developer/v2/posts/"
    cryptopanic_api_key: str = ""
    cryptopanic_public: bool = True
    cryptopanic_filter: str = ""
    cryptopanic_kind: str = "news"
    cryptopanic_regions: str = "en"
    cryptopanic_min_sentiment_score: int = 30
    cryptopanic_very_bearish_label: str = "very_bearish"
    crypto_bearish_block_minutes: int = 60
    crypto_keyword_guard_excluded_terms: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["MT GOX", "MT. GOX", "2011", "HARD FORK", "FORMER BOSS"]
    )
    crypto_keyword_guard_lookback_minutes: int = 240
    crypto_keyword_guard_min_negative_votes: int = 3
    crypto_keyword_guard_min_panic_score: int = 70
    crypto_keyword_guard_require_instrument_match: bool = True
    crypto_keyword_guard_terms: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "HACK",
            "EXPLOIT",
            "SECURITY BREACH",
            "BANKRUPTCY",
            "INSOLVENCY",
            "RUG PULL",
            "HOT WALLET",
            "STOLEN FUNDS",
        ]
    )
    symbol_currency_map: Annotated[dict[str, list[str]], NoDecode] = Field(
        default_factory=lambda: {
            "GOLD": ["USD"],
            "XAUUSD": ["USD"],
            "EURUSD": ["EUR", "USD"],
            "BTCUSD": ["USD"],
            "ETHUSD": ["USD"],
        }
    )

    @field_validator("symbol_currency_map", mode="before")
    @classmethod
    def parse_symbol_currency_map(cls, value):
        return _parse_mapping(value)

    @field_validator("crypto_keyword_guard_terms", "crypto_keyword_guard_excluded_terms", mode="before")
    @classmethod
    def parse_keyword_csv(cls, value):
        parsed = _parse_csv(value)
        if isinstance(parsed, list):
            return [str(item).strip().upper() for item in parsed if str(item).strip()]
        return value


class SpreadProfileSettings(BaseModel):
    enabled: bool = True
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["GOLD", "XAUUSD", "BTCUSD", "ETHUSD"]
    )
    cache_path: str = "data/spread_profiles.json"
    stale_cache_max_seconds: int = 86_400
    min_rows: int = 1_000
    soft_quantile: float = 0.95
    extreme_quantile: float = 0.99
    session_buffer_points: int = 5
    hour_buffer_points: int = 3
    news_buffer_points: int = 4
    extreme_buffer_points: int = 20
    regime_buffer_points: int = 6
    weekend_open_extra_buffer_points: int = 4
    rollover_extra_buffer_points: int = 8
    news_window_extra_tightening_points: int = 3
    pre_news_extra_tightening_points: int = 2
    pre_news_close_only_extra_tightening_points: int = 4
    release_minute_extra_tightening_points: int = 6
    post_release_freeze_extra_tightening_points: int = 5
    post_news_cooldown_extra_tightening_points: int = 3
    post_news_reentry_extra_tightening_points: int = 1
    min_soft_limit_points: int = 45
    min_news_limit_points: int = 40
    min_extreme_limit_points: int = 70
    symbol_min_soft_limit_points: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_min_news_limit_points: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    symbol_min_extreme_limit_points: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    rollover_hours_utc: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [23, 0, 1])
    weekend_open_days_utc: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [6, 0])
    weekend_open_hours_utc: Annotated[list[int], NoDecode] = Field(
        default_factory=lambda: [22, 23, 0, 1]
    )

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, value):
        return _parse_csv(value)

    @field_validator("rollover_hours_utc", "weekend_open_days_utc", "weekend_open_hours_utc", mode="before")
    @classmethod
    def parse_int_csv(cls, value):
        parsed = _parse_csv(value)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
        return value

    @field_validator(
        "symbol_min_soft_limit_points",
        "symbol_min_news_limit_points",
        "symbol_min_extreme_limit_points",
        mode="before",
    )
    @classmethod
    def parse_float_mapping(cls, value):
        return _parse_float_mapping(value)


class BotSettings(BaseModel):
    environment: str = "demo"
    heartbeat_seconds: int = 60
    log_level: str = "INFO"
    log_dir: str = "logs"
    model_dir: str = "models"
    data_dir: str = "data"
    model_refresh_hour_utc: int = 0
    model_refresh_minute_utc: int = 10
    run_lock_path: str = "data/run-live.lock"
    kill_switch_path: str = "data/kill_switch_state.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    mt5: MT5Settings = Field(default_factory=MT5Settings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    spread_profile: SpreadProfileSettings = Field(default_factory=SpreadProfileSettings)
    bot: BotSettings = Field(default_factory=BotSettings)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT_DIR / path

    @property
    def database_path(self) -> Path:
        return self.resolve_path(self.database.path)

    @property
    def log_dir(self) -> Path:
        return self.resolve_path(self.bot.log_dir)

    @property
    def model_dir(self) -> Path:
        return self.resolve_path(self.bot.model_dir)

    @property
    def data_dir(self) -> Path:
        return self.resolve_path(self.bot.data_dir)

    @property
    def news_cache_path(self) -> Path:
        return self.resolve_path(self.news.cache_path)

    @property
    def spread_profile_cache_path(self) -> Path:
        return self.resolve_path(self.spread_profile.cache_path)

    @property
    def run_lock_path(self) -> Path:
        return self.resolve_path(self.bot.run_lock_path)

    @property
    def kill_switch_path(self) -> Path:
        return self.resolve_path(self.bot.kill_switch_path)

    @property
    def xgb_artifact_path(self) -> Path:
        return self.resolve_path(self.model.xgb_artifact)

    def xgb_artifact_path_for(self, symbol: str, target_side: str = "long") -> Path:
        return self._artifact_path_for(self.model.xgb_artifact, symbol, target_side, ".joblib")

    def lgbm_artifact_path_for(self, symbol: str, target_side: str = "long") -> Path:
        return self._artifact_path_for(self.model.lgbm_artifact, symbol, target_side, ".joblib")

    def sequence_artifact_path_for(self, symbol: str, target_side: str = "long") -> Path:
        return self._artifact_path_for(self.model.sequence_artifact, symbol, target_side, ".pt")

    def transformer_artifact_path_for(self, symbol: str, target_side: str = "long") -> Path:
        return self._artifact_path_for(self.model.transformer_artifact, symbol, target_side, ".pt")

    def optimization_profile_path_for(self, symbol: str) -> Path:
        configured = self.model.optimization_profile
        safe_symbol = "".join(char if char.isalnum() else "_" for char in symbol.upper())
        if "{symbol}" in configured:
            return self.resolve_path(configured.format(symbol=safe_symbol))

        path = self.resolve_path(configured)
        suffix = path.suffix or ".json"
        stem = path.stem if path.suffix else path.name
        if len(self.trading.symbols) <= 1:
            return path
        return path.with_name(f"{stem}_{safe_symbol}{suffix}")

    def _artifact_path_for(self, configured: str, symbol: str, target_side: str, default_suffix: str) -> Path:
        safe_symbol = "".join(char if char.isalnum() else "_" for char in symbol.upper())
        safe_side = "".join(char if char.isalnum() else "_" for char in target_side.lower())
        if "{symbol}" in configured:
            return self.resolve_path(configured.format(symbol=safe_symbol, side=safe_side))

        path = self.resolve_path(configured)
        suffix = path.suffix or default_suffix
        stem = path.stem if path.suffix else path.name
        if len(self.trading.symbols) <= 1:
            return path.with_name(f"{stem}_{safe_side}{suffix}")
        return path.with_name(f"{stem}_{safe_symbol}_{safe_side}{suffix}")

    def _optimization_profile(self, symbol: str) -> dict[str, float]:
        path = self.optimization_profile_path_for(symbol)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("profile_accepted") is False:
            return {}
        return payload

    def walk_forward_report_path_for(self, symbol: str, now: datetime | None = None) -> Path:
        safe_symbol = "".join(char if char.isalnum() else "_" for char in symbol.upper())
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d")
        return self.data_dir / f"walk_forward_{safe_symbol}_{stamp}.json"

    def trading_timeframes(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for timeframe in [self.trading.timeframe, *self.trading.confirmation_timeframes]:
            normalized = str(timeframe).strip().upper()
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
        return ordered

    def threshold_for(self, symbol: str, side: str) -> float:
        normalized_symbol = symbol.upper()
        normalized_side = side.lower()
        if normalized_side == "long":
            configured = self.model.symbol_long_thresholds.get(normalized_symbol)
        elif normalized_side == "short":
            configured = self.model.symbol_short_thresholds.get(normalized_symbol)
        else:
            raise ValueError(f"Unsupported threshold side: {side}")
        if configured is not None:
            return float(configured)
        profile_key = f"{normalized_side}_threshold"
        profile = self._optimization_profile(normalized_symbol)
        if profile_key in profile:
            return float(profile[profile_key])
        return float(self.model.xgb_probability_threshold)

    def dominance_margin_for(self, symbol: str) -> float:
        configured = self.model.symbol_probability_dominance_margins.get(symbol.upper())
        if configured is not None:
            return float(configured)
        return float(self.model.probability_dominance_margin)

    def risk_per_trade_for(self, symbol: str) -> float:
        configured = self.risk.symbol_risk_per_trade.get(symbol.upper())
        if configured is not None:
            return float(configured)
        profile = self._optimization_profile(symbol)
        if "risk_per_trade" in profile:
            return float(profile["risk_per_trade"])
        return float(self.risk.risk_per_trade)

    def atr_stop_loss_mult_for(self, symbol: str) -> float:
        normalized_symbol = symbol.upper()
        configured = self.risk.symbol_atr_stop_loss_mult.get(normalized_symbol)
        if configured is not None:
            return float(configured)
        profile = self._optimization_profile(normalized_symbol)
        if "atr_stop_loss_mult" in profile:
            return float(profile["atr_stop_loss_mult"])
        if normalized_symbol in {item.upper() for item in self.trading.crypto_symbols}:
            return float(self.risk.crypto_atr_stop_loss_mult)
        return float(self.risk.atr_stop_loss_mult)

    def atr_take_profit_mult_for(self, symbol: str) -> float:
        normalized_symbol = symbol.upper()
        configured = self.risk.symbol_atr_take_profit_mult.get(normalized_symbol)
        if configured is not None:
            return float(configured)
        profile = self._optimization_profile(normalized_symbol)
        if "atr_take_profit_mult" in profile:
            return float(profile["atr_take_profit_mult"])
        if normalized_symbol in {item.upper() for item in self.trading.crypto_symbols}:
            return float(self.risk.crypto_atr_take_profit_mult)
        return float(self.risk.atr_take_profit_mult)

    def auto_breakeven_rr_for(self, symbol: str) -> float:
        normalized_symbol = symbol.upper()
        configured = self.risk.symbol_auto_breakeven_rr.get(normalized_symbol)
        if configured is not None:
            return float(configured)
        if normalized_symbol in {item.upper() for item in self.trading.crypto_symbols}:
            return float(self.risk.auto_breakeven_rr_crypto)
        return float(self.risk.auto_breakeven_rr_default)

    def time_exit_bars_for(self, symbol: str) -> int:
        configured = self.risk.symbol_time_exit_bars.get(symbol.upper())
        if configured is not None:
            return int(configured)
        return int(self.risk.time_exit_bars)

    def time_exit_min_progress_rr_for(self, symbol: str) -> float:
        configured = self.risk.symbol_time_exit_min_progress_rr.get(symbol.upper())
        if configured is not None:
            return float(configured)
        return float(self.risk.time_exit_min_progress_rr)

    def trailing_trigger_rr_for(self, symbol: str) -> float:
        configured = self.risk.symbol_trailing_trigger_rr.get(symbol.upper())
        if configured is not None:
            return float(configured)
        return float(self.risk.trailing_trigger_rr)

    def trailing_lock_rr_for(self, symbol: str) -> float:
        configured = self.risk.symbol_trailing_lock_rr.get(symbol.upper())
        if configured is not None:
            return float(configured)
        return float(self.risk.trailing_lock_rr)

    def sequence_probability_weight_for(self, symbol: str) -> float:
        configured = self.model.symbol_sequence_probability_weights.get(symbol.upper())
        if configured is not None:
            return float(configured)
        profile = self._optimization_profile(symbol)
        if "sequence_probability_weight" in profile:
            return float(profile["sequence_probability_weight"])
        return float(self.model.sequence_probability_weight)

    def lightgbm_probability_weight_for(self, symbol: str) -> float:
        configured = self.model.symbol_lightgbm_probability_weights.get(symbol.upper())
        if configured is not None:
            return float(configured)
        profile = self._optimization_profile(symbol)
        if "lightgbm_probability_weight" in profile:
            return float(profile["lightgbm_probability_weight"])
        return float(self.model.lightgbm_probability_weight)

    def transformer_probability_weight_for(self, symbol: str) -> float:
        configured = self.model.symbol_transformer_probability_weights.get(symbol.upper())
        if configured is not None:
            return float(configured)
        profile = self._optimization_profile(symbol)
        if "transformer_probability_weight" in profile:
            return float(profile["transformer_probability_weight"])
        return float(self.model.transformer_probability_weight)


def ensure_runtime_directories(settings: Settings) -> None:
    for path in (
        settings.database_path.parent,
        settings.log_dir,
        settings.model_dir,
        settings.data_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for symbol in settings.trading.symbols:
        for side in ("long", "short"):
            settings.xgb_artifact_path_for(symbol, side).parent.mkdir(parents=True, exist_ok=True)
            settings.lgbm_artifact_path_for(symbol, side).parent.mkdir(parents=True, exist_ok=True)
            settings.sequence_artifact_path_for(symbol, side).parent.mkdir(parents=True, exist_ok=True)
            settings.transformer_artifact_path_for(symbol, side).parent.mkdir(parents=True, exist_ok=True)
        settings.optimization_profile_path_for(symbol).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    ensure_runtime_directories(settings)
    return settings
