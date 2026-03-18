from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_investment_advisor.txt"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"


class Settings(BaseSettings):
    """Runtime configuration for the Telegram investment advisor bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    telegram_token: str = Field(default="", validation_alias="TELEGRAM_TOKEN")
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-5-mini", validation_alias="LLM_MODEL")
    llm_base_url: str = Field(default="https://api.openai.com/v1", validation_alias="LLM_BASE_URL")
    llm_organization: str | None = Field(default=None, validation_alias="OPENAI_ORGANIZATION")
    llm_project: str | None = Field(default=None, validation_alias="OPENAI_PROJECT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    news_timeout_seconds: float = Field(default=15.0, validation_alias="NEWS_TIMEOUT_SECONDS")
    llm_timeout_seconds: float = Field(default=30.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_output_tokens: int = Field(default=1000, validation_alias="LLM_MAX_OUTPUT_TOKENS")
    system_prompt_path: Path = Field(
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        validation_alias="SYSTEM_PROMPT_PATH",
    )
    logs_dir: Path = Field(default=DEFAULT_LOGS_DIR, validation_alias="LOGS_DIR")
    market_history_period: str = Field(default="6mo", validation_alias="MARKET_HISTORY_PERIOD")
    market_history_interval: str = Field(default="1d", validation_alias="MARKET_HISTORY_INTERVAL")
    market_history_limit: int = Field(default=180, validation_alias="MARKET_HISTORY_LIMIT")
    market_news_limit: int = Field(default=8, validation_alias="MARKET_NEWS_LIMIT")

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def validate_runtime(self) -> None:
        if not self.telegram_token.strip():
            raise ValueError("TELEGRAM_TOKEN is required")

    def llm_available(self) -> bool:
        return bool(self.llm_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

