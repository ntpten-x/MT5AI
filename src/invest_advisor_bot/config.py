from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from invest_advisor_bot.analysis.portfolio_profile import normalize_profile_name

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_investment_advisor.txt"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_ALERT_STATE_PATH = PROJECT_ROOT / "data" / "alert_state.json"
DEFAULT_USER_STATE_PATH = PROJECT_ROOT / "data" / "user_state.json"
DEFAULT_PORTFOLIO_STATE_PATH = PROJECT_ROOT / "data" / "portfolio_state.json"
DEFAULT_SECTOR_ROTATION_STATE_PATH = PROJECT_ROOT / "data" / "sector_rotation_state.json"
DEFAULT_REPORT_MEMORY_PATH = PROJECT_ROOT / "data" / "report_memory.json"
DEFAULT_RUNTIME_HISTORY_PATH = PROJECT_ROOT / "data" / "runtime_history.json"


class Settings(BaseSettings):
    """Runtime configuration for the Telegram investment advisor bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_token: str = Field(default="", validation_alias="TELEGRAM_TOKEN")
    telegram_report_chat_id: str = Field(default="", validation_alias="TELEGRAM_REPORT_CHAT_ID")
    database_url: str = Field(default="", validation_alias="DATABASE_URL")

    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_provider: str = Field(default="auto", validation_alias="LLM_PROVIDER")
    llm_provider_order: str = Field(default="", validation_alias="LLM_PROVIDER_ORDER")
    llm_model: str = Field(default="gpt-5-mini", validation_alias="LLM_MODEL")
    llm_model_fallbacks: str = Field(default="", validation_alias="LLM_MODEL_FALLBACKS")
    llm_base_url: str = Field(default="https://api.openai.com/v1", validation_alias="LLM_BASE_URL")
    llm_organization: str | None = Field(default=None, validation_alias="OPENAI_ORGANIZATION")
    llm_project: str | None = Field(default=None, validation_alias="OPENAI_PROJECT")
    llm_timeout_seconds: float = Field(default=25.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_output_tokens: int = Field(default=600, validation_alias="LLM_MAX_OUTPUT_TOKENS")
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_models: str = Field(
        default="google/gemini-2.0-flash-exp:free,meta-llama/llama-3.3-70b-instruct:free,qwen/qwen-2.5-72b-instruct:free,mistralai/mistral-small-3.1-24b-instruct:free",
        validation_alias="OPENROUTER_MODELS",
    )
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL")
    openrouter_http_referer: str | None = Field(default=None, validation_alias="OPENROUTER_HTTP_REFERER")
    openrouter_app_title: str | None = Field(default="Invest Advisor Bot", validation_alias="OPENROUTER_APP_TITLE")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_models: str = Field(default="gemini-2.0-flash,gemini-2.0-flash-lite", validation_alias="GEMINI_MODELS")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GEMINI_BASE_URL",
    )
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_models: str = Field(
        default="llama-3.3-70b-versatile,llama3-8b-8192",
        validation_alias="GROQ_MODELS",
    )
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", validation_alias="GROQ_BASE_URL")
    github_models_api_key: str = Field(default="", validation_alias="GITHUB_MODELS_API_KEY")
    github_models: str = Field(
        default="openai/gpt-4.1-mini,meta/Llama-3.3-70B-Instruct",
        validation_alias="GITHUB_MODELS",
    )
    github_models_base_url: str = Field(
        default="https://models.github.ai/inference",
        validation_alias="GITHUB_MODELS_BASE_URL",
    )
    github_models_api_version: str = Field(default="2026-03-10", validation_alias="GITHUB_MODELS_API_VERSION")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    exa_api_key: str = Field(default="", validation_alias="EXA_API_KEY")
    research_provider_order: str = Field(default="tavily,exa", validation_alias="RESEARCH_PROVIDER_ORDER")
    alpha_vantage_api_key: str = Field(default="", validation_alias="ALPHA_VANTAGE_API_KEY")
    market_data_provider_order: str = Field(
        default="alpha_vantage,yfinance",
        validation_alias="MARKET_DATA_PROVIDER_ORDER",
    )
    market_data_http_timeout_seconds: float = Field(default=12.0, validation_alias="MARKET_DATA_HTTP_TIMEOUT_SECONDS")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    logs_dir: Path = Field(default=DEFAULT_LOGS_DIR, validation_alias="LOGS_DIR")
    system_prompt_path: Path = Field(
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        validation_alias="SYSTEM_PROMPT_PATH",
    )

    news_timeout_seconds: float = Field(default=10.0, validation_alias="NEWS_TIMEOUT_SECONDS")
    market_history_period: str = Field(default="6mo", validation_alias="MARKET_HISTORY_PERIOD")
    market_history_interval: str = Field(default="1d", validation_alias="MARKET_HISTORY_INTERVAL")
    market_history_limit: int = Field(default=180, validation_alias="MARKET_HISTORY_LIMIT")
    market_news_limit: int = Field(default=5, validation_alias="MARKET_NEWS_LIMIT")
    market_cache_ttl_seconds: int = Field(default=900, validation_alias="MARKET_CACHE_TTL_SECONDS")
    news_cache_ttl_seconds: int = Field(default=900, validation_alias="NEWS_CACHE_TTL_SECONDS")
    default_investor_profile: str = Field(default="conservative", validation_alias="DEFAULT_INVESTOR_PROFILE")
    alert_state_path: Path = Field(default=DEFAULT_ALERT_STATE_PATH, validation_alias="ALERT_STATE_PATH")
    user_state_path: Path = Field(default=DEFAULT_USER_STATE_PATH, validation_alias="USER_STATE_PATH")
    portfolio_state_path: Path = Field(default=DEFAULT_PORTFOLIO_STATE_PATH, validation_alias="PORTFOLIO_STATE_PATH")
    sector_rotation_state_path: Path = Field(
        default=DEFAULT_SECTOR_ROTATION_STATE_PATH,
        validation_alias="SECTOR_ROTATION_STATE_PATH",
    )
    report_memory_path: Path = Field(default=DEFAULT_REPORT_MEMORY_PATH, validation_alias="REPORT_MEMORY_PATH")
    runtime_history_path: Path = Field(default=DEFAULT_RUNTIME_HISTORY_PATH, validation_alias="RUNTIME_HISTORY_PATH")
    runtime_history_retention_days: int = Field(default=30, validation_alias="RUNTIME_HISTORY_RETENTION_DAYS")
    burn_in_target_days: int = Field(default=14, validation_alias="BURN_IN_TARGET_DAYS")
    log_rotation_size: str = Field(default="10 MB", validation_alias="LOG_ROTATION_SIZE")
    log_retention: str = Field(default="14 days", validation_alias="LOG_RETENTION")
    maintenance_cleanup_interval_minutes: int = Field(default=360, validation_alias="MAINTENANCE_CLEANUP_INTERVAL_MINUTES")
    stock_pick_evaluation_interval_minutes: int = Field(default=360, validation_alias="STOCK_PICK_EVALUATION_INTERVAL_MINUTES")
    stock_pick_evaluation_horizon_days: int = Field(default=5, validation_alias="STOCK_PICK_EVALUATION_HORIZON_DAYS")
    backup_dir: Path = Field(default=PROJECT_ROOT / "backups", validation_alias="BACKUP_DIR")
    backup_interval_hours: int = Field(default=24, validation_alias="BACKUP_INTERVAL_HOURS")
    backup_retention_days: int = Field(default=14, validation_alias="BACKUP_RETENTION_DAYS")
    alert_suppression_minutes: int = Field(default=180, validation_alias="ALERT_SUPPRESSION_MINUTES")
    risk_score_alert_threshold: float = Field(default=6.5, validation_alias="RISK_SCORE_ALERT_THRESHOLD")
    opportunity_score_alert_threshold: float = Field(default=2.8, validation_alias="OPPORTUNITY_SCORE_ALERT_THRESHOLD")
    news_impact_alert_threshold: float = Field(default=2.0, validation_alias="NEWS_IMPACT_ALERT_THRESHOLD")
    sector_rotation_min_streak: int = Field(default=3, validation_alias="SECTOR_ROTATION_MIN_STREAK")
    earnings_result_lookback_days: int = Field(default=14, validation_alias="EARNINGS_RESULT_LOOKBACK_DAYS")

    bot_min_request_interval_seconds: float = Field(
        default=2.0,
        validation_alias="BOT_MIN_REQUEST_INTERVAL_SECONDS",
    )
    jobs_enabled: bool = Field(default=False, validation_alias="JOBS_ENABLED")
    daily_digest_hour_utc: int = Field(default=1, validation_alias="DAILY_DIGEST_HOUR_UTC")
    daily_digest_minute_utc: int = Field(default=0, validation_alias="DAILY_DIGEST_MINUTE_UTC")
    morning_report_hour_utc: int = Field(default=12, validation_alias="MORNING_REPORT_HOUR_UTC")
    morning_report_minute_utc: int = Field(default=30, validation_alias="MORNING_REPORT_MINUTE_UTC")
    midday_report_hour_utc: int = Field(default=17, validation_alias="MIDDAY_REPORT_HOUR_UTC")
    midday_report_minute_utc: int = Field(default=0, validation_alias="MIDDAY_REPORT_MINUTE_UTC")
    closing_report_hour_utc: int = Field(default=21, validation_alias="CLOSING_REPORT_HOUR_UTC")
    closing_report_minute_utc: int = Field(default=15, validation_alias="CLOSING_REPORT_MINUTE_UTC")
    risk_check_interval_minutes: int = Field(default=30, validation_alias="RISK_CHECK_INTERVAL_MINUTES")
    risk_vix_alert_threshold: float = Field(default=30.0, validation_alias="RISK_VIX_ALERT_THRESHOLD")
    earnings_alert_days_ahead: int = Field(default=14, validation_alias="EARNINGS_ALERT_DAYS_AHEAD")

    health_check_enabled: bool = Field(default=True, validation_alias="HEALTH_CHECK_ENABLED")
    health_check_host: str = Field(default="0.0.0.0", validation_alias="HEALTH_CHECK_HOST")
    health_check_port: int = Field(
        default_factory=lambda: int(os.environ.get("PORT", "10000")),
        validation_alias="HEALTH_CHECK_PORT",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return normalized

    @field_validator("market_history_limit", "market_news_limit", "llm_max_output_tokens")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Configured integer values must be greater than 0")
        return value

    @field_validator(
        "llm_timeout_seconds",
        "news_timeout_seconds",
        "bot_min_request_interval_seconds",
        "market_data_http_timeout_seconds",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Timeout and rate-limit values must be greater than 0")
        return value

    @field_validator(
        "market_cache_ttl_seconds",
        "news_cache_ttl_seconds",
        "alert_suppression_minutes",
        "daily_digest_hour_utc",
        "daily_digest_minute_utc",
        "morning_report_hour_utc",
        "morning_report_minute_utc",
        "midday_report_hour_utc",
        "midday_report_minute_utc",
        "closing_report_hour_utc",
        "closing_report_minute_utc",
        "risk_check_interval_minutes",
        "earnings_alert_days_ahead",
        "maintenance_cleanup_interval_minutes",
        "stock_pick_evaluation_interval_minutes",
        "stock_pick_evaluation_horizon_days",
        "backup_interval_hours",
        "backup_retention_days",
        "burn_in_target_days",
        "sector_rotation_min_streak",
        "earnings_result_lookback_days",
        "health_check_port",
    )
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Configured integer values must not be negative")
        return value

    @field_validator("default_investor_profile")
    @classmethod
    def validate_default_investor_profile(cls, value: str) -> str:
        normalized = normalize_profile_name(value, default="conservative")
        if normalized not in {"conservative", "balanced", "growth"}:
            raise ValueError("DEFAULT_INVESTOR_PROFILE must be conservative, balanced, or growth")
        return normalized

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().casefold() or "auto"
        if normalized not in {"auto", "openai", "openrouter", "gemini", "groq", "github_models"}:
            raise ValueError("LLM_PROVIDER must be auto, openai, openrouter, gemini, groq, or github_models")
        return normalized

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def validate_runtime(self) -> None:
        if not self.telegram_token.strip():
            raise ValueError("TELEGRAM_TOKEN is required")
        if not self.system_prompt_path.exists():
            raise ValueError(f"System prompt file not found: {self.system_prompt_path}")

    def llm_available(self) -> bool:
        return bool(
            self.llm_api_key.strip()
            or self.openrouter_api_key.strip()
            or self.gemini_api_key.strip()
            or self.groq_api_key.strip()
            or self.github_models_api_key.strip()
        )

    def research_available(self) -> bool:
        return bool(self.tavily_api_key.strip() or self.exa_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
